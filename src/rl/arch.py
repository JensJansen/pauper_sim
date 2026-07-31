"""Set Transformer encoder + FiLM conditioning over rl.features's
per-token representation -- the shared perception stack (docs discussion on
the attention-opponent-encoding branch): card identity embeddings, a
self-attention encoder over the variable-length token set, and a FiLM
conditioner. Deliberately does NOT include a trunk, critic, or action head
-- those are per-deck (see rl.deck), not shared.

Uses torch.nn.MultiheadAttention/TransformerEncoderLayer directly (already
a torch dependency, no new library) rather than hand-rolling attention --
same "stdlib/already-installed dependency before custom code" reasoning
this codebase already applies elsewhere (e.g. torch.distributions.Categorical
in the PPO update instead of a hand-rolled categorical).

Padding/masking: a batch of games has a variable number of tokens per
game (different board states). Every function here takes a PADDED batch
(shape (batch, max_tokens, ...)) plus a boolean key_padding_mask (True =
padding, to be ignored) -- the standard nn.MultiheadAttention contract --
never a ragged/variable-shape tensor. pad_token_batch below is the one
place that does the actual padding, so every caller (rollout collection,
PPO update) goes through the same, single, tested implementation."""

import numpy as np
import torch
import torch.nn as nn

from rl.features import TOKEN_FEATURE_DIM


def pad_token_batch(token_lists, device="cpu"):
    """token_lists: a list (one per batch element) of that element's own
    list of (vocab_index, feature_row, identity) triples, e.g. one
    build_token_set(...) result per game/step in a batch. Returns
    (vocab_idx [B, T] long, features [B, T, TOKEN_FEATURE_DIM] float,
    key_padding_mask [B, T] bool -- True at padded positions, identities --
    a plain list-of-lists, batch element b's own list of (Permanent or None)
    at each padded position t, index-aligned with the tensor dims above, for
    the pointer-network action head to map a legal target back to its own
    row). T = the longest token list in this specific batch, not a fixed
    constant -- every batch pads to its own max, matching how PPO rollout
    minibatches already vary in size (the final minibatch can be shorter
    than batch_size)."""
    batch_size = len(token_lists)
    max_len = max((len(tl) for tl in token_lists), default=1)
    max_len = max(max_len, 1)  # a batch element with ZERO tokens (empty board) still needs one padded slot
    vocab_idx = torch.zeros((batch_size, max_len), dtype=torch.long, device=device)
    features = torch.zeros((batch_size, max_len, TOKEN_FEATURE_DIM), dtype=torch.float32, device=device)
    key_padding_mask = torch.ones((batch_size, max_len), dtype=torch.bool, device=device)
    identities = [[None] * max_len for _ in range(batch_size)]
    for b, tl in enumerate(token_lists):
        for t, (idx, row, identity) in enumerate(tl):
            vocab_idx[b, t] = idx
            features[b, t] = torch.as_tensor(row, dtype=torch.float32)
            key_padding_mask[b, t] = False
            identities[b][t] = identity
    return vocab_idx, features, key_padding_mask, identities


class SetTransformer(nn.Module):
    """Card-identity embedding + linear projection into d_model, a stack of
    self-attention encoder layers (tokens attend to each other -- this is
    the whole reason Set Transformer was chosen over DeepSets: relative
    valuations between tokens, e.g. an attacker's real threat depends on
    what can block it), then TWO independent Pooling-by-Multihead-Attention
    (PMA) heads -- one learned query attending over the "mine" tokens for
    the trunk's own input, one over the "theirs" tokens for FiLM's
    conditioning input. A single joint self-attention stack over BOTH
    sides' tokens together (not two separate encoders) is what lets a
    token attend across the mine/theirs boundary before pooling -- pooling
    is what then splits the result back into two summaries.

    Stabilization: pre-norm (norm_first=True) + residual connections, per
    Parisotto et al.'s "Stabilizing Transformers for Reinforcement
    Learning" finding that RL objectives make plain post-norm transformers
    materially harder to optimize than the same architecture under a
    supervised objective. This isn't GTrXL's own gated-residual fix
    (that's for temporal/sequential transformers, a different attention use
    than this permutation-invariant, single-timestep one) but pre-norm is
    the same-spirit, directly-applicable half of that lesson."""

    def __init__(self, vocab_size, d_model=64, n_heads=4, n_layers=2, dim_feedforward=128, dropout=0.0):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model, padding_idx=0)
        self.input_proj = nn.Linear(d_model + TOKEN_FEATURE_DIM, d_model)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads, dim_feedforward=dim_feedforward, dropout=dropout,
            batch_first=True, norm_first=True,
        )
        # enable_nested_tensor=False: the nested-tensor fast path doesn't
        # support norm_first=True anyway (torch's own warning otherwise) --
        # disabling it explicitly silences the warning instead of leaving
        # noise in every future run.
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers, enable_nested_tensor=False)

        self.mine_query = nn.Parameter(torch.randn(1, 1, d_model) * 0.02)
        self.theirs_query = nn.Parameter(torch.randn(1, 1, d_model) * 0.02)
        self.mine_pool = nn.MultiheadAttention(d_model, n_heads, batch_first=True)
        self.theirs_pool = nn.MultiheadAttention(d_model, n_heads, batch_first=True)

        self.d_model = d_model

    def forward(self, vocab_idx, features, key_padding_mask, side_flag):
        """side_flag: [B, T] float, 1.0 for "mine" tokens, 0.0 for "theirs"
        -- rl.features.build_token_set's own last feature column, passed
        separately (not re-derived from `features`) so pooling's own
        mine/theirs mask stays explicit and doesn't rely on remembering
        which column index it lives at.

        Returns (mine_summary [B, d_model], theirs_summary [B, d_model],
        token_reps [B, T, d_model]) -- token_reps is exposed for the
        pointer-network action head (rl.deck), which scores actions
        against these SAME post-attention representations, not the raw
        pre-attention embeddings, so a token's scoring already reflects
        everything else on the board.

        A batch element with ALL tokens padded (an empty board on both
        sides, impossible in practice once mana/lands exist, but defensive
        regardless) would give MultiheadAttention an all-True padding row,
        which produces NaN softmax -- guarded by unmasking one dummy
        position for any such row before pooling (its value is discarded
        by the encoder's own all-zero embedding + all-zero mask contribution
        anyway, since nothing legitimate can attend to it)."""
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
    """Small MLP: conditioning vector -> per-layer (gamma, beta) pairs that
    modulate a trunk's own hidden activations (gamma * h + beta), one pair
    per trunk layer width given in `layer_dims`. Chosen over concatenation
    per the direct empirical comparison found during design (FiLM
    materially outperforming concatenation in an RL policy-conditioning
    study, plausibly because concatenation's added width can drown out the
    conditioning signal inside a wider, unrelated observation vector)."""

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
            gamma = 1.0 + raw[:, i:i + d]  # centered at 1.0 -- identity modulation at initialization
            beta = raw[:, i + d:i + 2 * d]
            out.append((gamma, beta))
            i += 2 * d
        return out
