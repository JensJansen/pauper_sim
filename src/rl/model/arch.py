"""Shared perception stack: Set Transformer encoder + FiLM conditioning over
rl.model.features's per-token representation. Card identity embeddings, a
self-attention encoder over the variable-length token set, and a FiLM
conditioner. No trunk, critic, or action head here -- those are per-deck
(rl.model.deck).

Every function takes a PADDED batch (shape (batch, max_tokens, ...)) plus a
boolean key_padding_mask (True = padding), the standard
nn.MultiheadAttention contract. pad_token_batch is the single place that
does the padding.
"""

import numpy as np
import torch
import torch.nn as nn

from rl.model.features import TOKEN_FEATURE_DIM


def pad_token_batch(token_lists, device="cpu"):
    """token_lists: one list per batch element of (vocab_index, feature_row,
    identity) triples (a build_token_set(...) result per game/step). Returns
    (vocab_idx [B, T] long, features [B, T, TOKEN_FEATURE_DIM] float,
    key_padding_mask [B, T] bool -- True at padded positions, identities --
    list of lists, batch element b's (Permanent or None) per padded
    position t, index-aligned with the tensors above, used by the pointer
    action head to map a legal target back to its row). T is the longest
    token list in this batch, not a fixed constant."""
    batch_size = len(token_lists)
    max_len = max((len(tl) for tl in token_lists), default=1)
    max_len = max(max_len, 1)  # an empty-board batch element still needs one padded slot
    # Fill numpy buffers first, convert to torch once for the whole batch (much
    # faster than one tensor write per token, same float32 output either way).
    vocab_idx_np = np.zeros((batch_size, max_len), dtype=np.int64)
    features_np = np.zeros((batch_size, max_len, TOKEN_FEATURE_DIM), dtype=np.float32)
    pad_mask_np = np.ones((batch_size, max_len), dtype=bool)
    identities = [[None] * max_len for _ in range(batch_size)]
    for b, tl in enumerate(token_lists):
        n = len(tl)
        if n == 0:
            continue
        vocab_idx_np[b, :n] = [idx for idx, _row, _identity in tl]
        features_np[b, :n] = [row for _idx, row, _identity in tl]
        pad_mask_np[b, :n] = False
        for t, (_idx, _row, identity) in enumerate(tl):
            identities[b][t] = identity
    vocab_idx = torch.from_numpy(vocab_idx_np).to(device)
    features = torch.from_numpy(features_np).to(device)
    key_padding_mask = torch.from_numpy(pad_mask_np).to(device)
    return vocab_idx, features, key_padding_mask, identities


class SetTransformer(nn.Module):
    """Card-identity embedding + linear projection into d_model, a stack of
    self-attention encoder layers (tokens attend to each other, so an
    attacker's threat depends on what can block it), then two independent
    Pooling-by-Multihead-Attention (PMA) heads: one learned query pools the
    "mine" tokens (trunk input), one pools the "theirs" tokens (FiLM
    conditioning input). Both sides share one joint self-attention stack so
    tokens can attend across the mine/theirs boundary before pooling splits
    the result back into two summaries.

    Pre-norm (norm_first=True) + residual connections for training
    stability under an RL objective."""

    def __init__(self, vocab_size, d_model=64, n_heads=4, n_layers=2, dim_feedforward=128, dropout=0.0):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model, padding_idx=0)
        self.input_proj = nn.Linear(d_model + TOKEN_FEATURE_DIM, d_model)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads, dim_feedforward=dim_feedforward, dropout=dropout,
            batch_first=True, norm_first=True,
        )
        # nested-tensor fast path doesn't support norm_first=True; disable explicitly to silence torch's warning.
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers, enable_nested_tensor=False)

        self.mine_query = nn.Parameter(torch.randn(1, 1, d_model) * 0.02)
        self.theirs_query = nn.Parameter(torch.randn(1, 1, d_model) * 0.02)
        self.mine_pool = nn.MultiheadAttention(d_model, n_heads, batch_first=True)
        self.theirs_pool = nn.MultiheadAttention(d_model, n_heads, batch_first=True)

        self.d_model = d_model

    def forward(self, vocab_idx, features, key_padding_mask, side_flag):
        """side_flag: [B, T] float, 1.0 for "mine" tokens, 0.0 for "theirs"
        (rl.model.features.build_token_set's last feature column, passed
        separately so pooling's mine/theirs mask stays explicit).

        Returns (mine_summary [B, d_model], theirs_summary [B, d_model],
        token_reps [B, T, d_model]). token_reps is exposed for the
        pointer-network action head (rl.model.deck), which scores actions
        against these post-attention representations.

        A batch element with ALL tokens padded would give MultiheadAttention
        an all-True padding row, producing a NaN softmax -- guarded by
        unmasking one dummy position for any such row before pooling."""
        x = self.embedding(vocab_idx)
        x = torch.cat([x, features], dim=-1)
        x = self.input_proj(x)

        safe_mask = key_padding_mask.clone()
        all_padded = safe_mask.all(dim=1)
        if all_padded.any():
            safe_mask[all_padded, 0] = False

        encoded = self.encoder(x, src_key_padding_mask=safe_mask)

        batch_size = vocab_idx.shape[0]
        mine_mask = safe_mask | (side_flag < 0.5)
        theirs_mask = safe_mask | (side_flag >= 0.5)
        mine_all_masked = mine_mask.all(dim=1)
        theirs_all_masked = theirs_mask.all(dim=1)
        if mine_all_masked.any():
            mine_mask = mine_mask.clone()
            mine_mask[mine_all_masked, 0] = False
        if theirs_all_masked.any():
            theirs_mask = theirs_mask.clone()
            theirs_mask[theirs_all_masked, 0] = False

        mine_q = self.mine_query.expand(batch_size, -1, -1)
        theirs_q = self.theirs_query.expand(batch_size, -1, -1)
        mine_summary, _ = self.mine_pool(mine_q, encoded, encoded, key_padding_mask=mine_mask)
        theirs_summary, _ = self.theirs_pool(theirs_q, encoded, encoded, key_padding_mask=theirs_mask)
        return mine_summary.squeeze(1), theirs_summary.squeeze(1), encoded


class FiLM(nn.Module):
    """MLP: conditioning vector -> per-layer (gamma, beta) pairs that
    modulate a trunk's hidden activations (gamma * h + beta), one pair per
    trunk layer width in `layer_dims`."""

    def __init__(self, condition_dim, layer_dims, hidden=64):
        super().__init__()
        self.layer_dims = list(layer_dims)
        total_out = sum(d * 2 for d in self.layer_dims)
        self.net = nn.Sequential(
            nn.Linear(condition_dim, hidden), nn.Tanh(), nn.Linear(hidden, total_out),
        )

    def forward(self, condition):
        raw = self.net(condition)
        out = []
        i = 0
        for d in self.layer_dims:
            gamma = 1.0 + raw[:, i:i + d]  # centered at 1.0: identity modulation at init
            beta = raw[:, i + d:i + 2 * d]
            out.append((gamma, beta))
            i += 2 * d
        return out
