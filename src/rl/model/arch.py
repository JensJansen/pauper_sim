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
        identities[b][:n] = [identity for _idx, _row, identity in tl]
    vocab_idx = torch.from_numpy(vocab_idx_np).to(device)
    features = torch.from_numpy(features_np).to(device)
    key_padding_mask = torch.from_numpy(pad_mask_np).to(device)
    return vocab_idx, features, key_padding_mask, identities


SUM_SCALE = 8.0  # a typical token count: keeps the sum-pooled half of a summary near unit scale
HEAD_DIM = 16  # attention heads are d_model // HEAD_DIM (the default 64 -> 4 heads), so a checkpoint's head count is implied by its d_model


def encoder_kwargs_from_state_dict(state_dict):
    """The SetTransformer constructor kwargs (d_model, n_layers, dim_feedforward, n_heads) a DeckNetwork state_dict was
    saved with, read off its tensor shapes -- the encoder analogue of checkpoint.trunk_hidden_from_deck_checkpoint.
    A default-sized checkpoint yields exactly SetTransformer's own defaults."""
    d_model = state_dict["encoder.embedding.weight"].shape[1]
    n_layers = 0
    while f"encoder.encoder.layers.{n_layers}.linear1.weight" in state_dict:
        n_layers += 1
    assert n_layers, "state_dict has no encoder.encoder.layers.* -- not a DeckNetwork state_dict?"
    return {"d_model": d_model, "n_layers": n_layers, "n_heads": d_model // HEAD_DIM,
            "dim_feedforward": state_dict["encoder.encoder.layers.0.linear1.weight"].shape[0]}


class SetTransformer(nn.Module):
    """Card-identity embedding + linear projection into d_model, a stack of
    self-attention encoder layers (tokens attend to each other, so an
    attacker's threat depends on what can block it), then two independent
    Pooling-by-Multihead-Attention (PMA) heads: one learned query pools the
    "mine" tokens (trunk input), one pools the "theirs" tokens (FiLM
    conditioning input). Both sides share one joint self-attention stack so
    tokens can attend across the mine/theirs boundary before pooling splits
    the result back into two summaries.

    Each side's summary is the attention pool CONCATENATED with a plain sum of
    that side's token representations (scaled by SUM_SCALE): a weighted mean
    cannot represent how many tokens there are (n identical cards pool like
    one), a sum can. summary_dim = 2 * d_model is the width consumers
    (DeckNetwork's trunk and FiLM, the mulligan net) are built against;
    token_reps stay d_model wide.

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
        self.summary_dim = 2 * d_model  # [attention pool | sum pool]

    def forward(self, vocab_idx, features, key_padding_mask, side_flag):
        """side_flag: [B, T] float, 1.0 for "mine" tokens, 0.0 for "theirs"
        (rl.model.features.build_token_set's last feature column, passed
        separately so pooling's mine/theirs mask stays explicit).

        Returns (mine_summary [B, summary_dim], theirs_summary [B, summary_dim],
        token_reps [B, T, d_model]). token_reps is exposed for the
        pointer-network action head (rl.model.deck), which scores actions
        against these post-attention representations.

        A batch element with ALL tokens padded would give MultiheadAttention
        an all-True padding row, producing a NaN softmax -- guarded by
        unmasking one dummy position for any such row before pooling. The same
        goes for a side (mine/theirs) with no real tokens (including a row with
        no real tokens at all): its pooled summary is overwritten with zeros,
        so it never depends on which token sits first or on padding content."""
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
        # A side with no tokens (e.g. the opponent's empty board early on) was unmasked at position 0 above only to
        # keep the softmax finite; that position is whichever token happens to be first, so the pooled value is an
        # order-dependent summary of the WRONG side. Its summary is zero instead: "nothing there".
        # Emptiness is judged from the ORIGINAL padding mask: safe_mask/mine_mask above had a dummy position unmasked,
        # which for a row with no real tokens at all would otherwise count as a (padding) "theirs" token.
        mine_empty = (key_padding_mask | (side_flag < 0.5)).all(dim=1)
        theirs_empty = (key_padding_mask | (side_flag >= 0.5)).all(dim=1)
        mine_summary = mine_summary.masked_fill(mine_empty[:, None, None], 0.0)
        theirs_summary = theirs_summary.masked_fill(theirs_empty[:, None, None], 0.0)
        # Sum pool over each side's REAL tokens (padding and the other side contribute exactly zero; where() rather
        # than a multiply so a non-finite padded activation could never leak in).
        real = ~key_padding_mask
        mine_sum = encoded.masked_fill(~(real & (side_flag >= 0.5))[:, :, None], 0.0).sum(dim=1) / SUM_SCALE
        theirs_sum = encoded.masked_fill(~(real & (side_flag < 0.5))[:, :, None], 0.0).sum(dim=1) / SUM_SCALE
        return (torch.cat([mine_summary.squeeze(1), mine_sum], dim=-1),
                torch.cat([theirs_summary.squeeze(1), theirs_sum], dim=-1), encoded)


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
