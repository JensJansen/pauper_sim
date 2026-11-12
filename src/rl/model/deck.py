"""Per-deck perception encoder + trunk + critic + pointer-network action
head. Each deck owns its own SetTransformer (rl.model.arch) as a registered
child module, trained end to end with the policy by the same PPO update.

Every checkpoint is self-contained: a live.pt/snapshot holds its own
encoder, so any two populations can play each other with no "trained
against the same stack" guard. Card knowledge is not shared between decks --
a deck learns its opponents' cards only from games against them. One
encoder instance per DeckNetwork; sharing an instance across two networks
would put it in both optimizers and train it twice per round.

Action space, per deck: the union of a small fixed table (non-targeting
actions -- Play land, Cast X, Pass, mana payment choices, mulligan
decisions, etc., fixed-shape since a deck's own card pool never changes at
inference time) and a pointer-scored set (Attack/Assign Blocker/Choose
target/Choose opponent's, whose count depends on the opponent's decklist).
Both halves are scored into one combined logit vector before a single
softmax -- not two independent distributions -- since both kinds of action
can be simultaneously legal at a decision point, and sampling/log-prob/
entropy need the true combined legal set."""

import torch
import torch.nn as nn

import game
from rl.model.arch import FiLM
from rl.model.features import TOKEN_FEATURE_DIM

# Width of the previous-action embedding fed into the GRU (see DeckNetwork).
PREV_ACTION_DIM = 16
# Width of the "what did I point at" embedding fed into the GRU alongside it: the chosen token's card identity and full
# feature row at the moment it was chosen (zero when the previous action wasn't a pointer pick).
PREV_POINTER_DIM = 32


class DeckNetwork(nn.Module):
    def __init__(self, encoder, non_targeting_n_actions, trunk_hidden=(128, 128)):
        super().__init__()
        # Registered child so net.parameters()/state_dict() cover the encoder
        # for free: the optimizer trains it and every checkpoint carries it.
        self.encoder = encoder
        d_model = encoder.d_model
        # mine_summary (attention pool | sum pool) + non-tokenized globals (life, turn, phase, exact zone counts, ...)
        trunk_input_dim = encoder.summary_dim + SCALAR_FEATURE_DIM

        self.trunk_layers = nn.ModuleList()
        prev = trunk_input_dim
        for width in trunk_hidden:
            self.trunk_layers.append(nn.Linear(prev, width))
            prev = width
        self.film = FiLM(condition_dim=encoder.summary_dim, layer_dims=list(trunk_hidden))  # conditioned on theirs_summary
        self.activation = nn.Tanh()

        # Recurrent memory between the trunk and every head, so the critic,
        # fixed-action head and pointer query are all history-aware. The
        # observation alone is strictly Markov (current board only), which
        # can't represent facts like "they held two blue up and passed".
        #
        # GRU over LSTM: one state tensor, no measured loss at this size.
        # Recurrence over a transformer-over-history: stacking N observations
        # would multiply the encoder cost by N.
        #
        # The agent's own previous action rides in embedded alongside the
        # observation (standard for recurrent RL: R2D2, IMPALA) -- this is
        # self-knowledge, not hidden state leaked to the agent.
        #
        # Two symbols past the fixed table: n_fixed for "some pointer action"
        # and n_fixed+1 for "no previous action" (first decision of a game).
        # Every pointer action collapses to one symbol since a pointer index
        # names a different token each state and carries no stable identity;
        # what was pointed at is visible on the board anyway.
        self.n_fixed = non_targeting_n_actions
        self.prev_action_embed = nn.Embedding(non_targeting_n_actions + 2, PREV_ACTION_DIM)
        # Every pointer action shares ONE symbol above (a pointer index names a different token each state), so on its
        # own the memory could not say WHAT was attacked with / targeted. This embeds the chosen token itself -- its
        # card identity (the encoder's own embedding table) plus its whole feature row -- which carries over from
        # state to state because it describes the object, not its position in that state's token list.
        self.prev_pointer_proj = nn.Linear(encoder.d_model + TOKEN_FEATURE_DIM, PREV_POINTER_DIM)
        self.gru = nn.GRU(prev + PREV_ACTION_DIM + PREV_POINTER_DIM, prev, batch_first=True)
        self.hidden_size = prev

        self.critic_head = nn.Linear(prev, 1)
        self.non_targeting_head = nn.Linear(prev, non_targeting_n_actions)
        self.pointer_query = nn.Linear(prev, d_model)
        self.d_model = d_model

    def initial_hidden(self, batch_size=1, device="cpu"):
        """Zero starting recurrent state for a new game. Rollout collection
        and ppo_update's episode replay both start here, so replay is exact
        (no stale-hidden-state correction or burn-in needed)."""
        return torch.zeros(1, batch_size, self.hidden_size, device=device)

    def prev_action_symbols(self, raw_actions, device="cpu"):
        """Raw combined-action indices -> prev_action_embed symbols.
        `raw_actions` is one int per batch row; None (or negative) means "no
        previous action". Anything at or past the fixed table collapses to a
        single pointer-action symbol."""
        symbols = [
            self.n_fixed + 1 if a is None or a < 0 else (self.n_fixed if a >= self.n_fixed else a)
            for a in raw_actions
        ]
        return torch.as_tensor(symbols, dtype=torch.long, device=device)

    def prev_pointer_embedding(self, prev_pointer, n_rows, device):
        """[n_rows, PREV_POINTER_DIM]. prev_pointer: (vocab_idx [n_rows] long, rows [n_rows, TOKEN_FEATURE_DIM]) of the
        token the previous recorded action pointed at, vocab_idx 0 where it was not a pointer pick; None = no pointer
        pick anywhere (all zeros)."""
        if prev_pointer is None:
            return torch.zeros(n_rows, PREV_POINTER_DIM, device=device)
        vocab_idx, rows = prev_pointer
        emb = torch.tanh(self.prev_pointer_proj(torch.cat([self.encoder.embedding(vocab_idx), rows], dim=-1)))
        return emb * (vocab_idx > 0).unsqueeze(-1).to(emb.dtype)

    def forward(self, mine_summary, theirs_summary, scalar_features, token_reps, pointer_token_mask,
                hidden=None, seq=None, prev_action=None, prev_pointer=None):
        """mine_summary/theirs_summary/token_reps: this deck's own encoder
        outputs (caller runs self.encoder separately since rl.decision.agent
        needs token_reps and identities to build the pointer mask first, and
        the mulligan net reuses the same encoder outputs). scalar_features:
        [B, SCALAR_FEATURE_DIM], the non-tokenized globals (life totals, turn
        number, phase one-hot, mana pools, library sizes, opponent hand size,
        stack-targets-me/opponent, exact zone/type counts -- see rl.decision.agent._scalar_features
        for the exact composition). pointer_token_mask: [B, T] bool, True
        where a token is a currently-legal pointer target for whatever
        resolution is pending (computed by the caller from the real game
        engine, e.g. game.creature_attack_eligible -- never guessed here).

        prev_action: [B*T] long tensor of prev_action_embed symbols (build
        via prev_action_symbols), or None for "no previous action" on every
        row. hidden: [1, B, hidden_size] recurrent state, or None for a fresh
        zero state. seq: optional (B, T) naming the sequence layout of the
        flat batch -- everything else stays flat [B*T, ...] since only the
        GRU needs a time axis. seq=None means one timestep each (rollout
        shape).

        A flat batch passed with seq=(B, T) MUST be laid out episode-major
        (all T steps of episode 0, then episode 1, ...) -- `.reshape(B, T,
        -1)` assumes this; getting it backwards silently trains the GRU on
        interleaved episodes.

        Returns (combined_logits [B*T, non_targeting_n_actions + tokens],
        value [B*T], new_hidden [1, B, hidden_size]). combined_logits' first
        non_targeting_n_actions entries are the fixed-table actions (masked
        externally by the caller); the rest are one pointer score per token
        position, masked here via pointer_token_mask."""
        h = torch.cat([mine_summary, scalar_features], dim=-1)
        film_params = self.film(theirs_summary)
        for layer, (gamma, beta) in zip(self.trunk_layers, film_params):
            h = self.activation(gamma * layer(h) + beta)

        # None means "no previous action" for every row (first decision of a
        # game, or a caller that doesn't track it).
        if prev_action is None:
            prev_action = self.prev_action_symbols([None] * h.shape[0], h.device)
        h = torch.cat([h, self.prev_action_embed(prev_action), self.prev_pointer_embedding(prev_pointer, h.shape[0], h.device)],
                      dim=-1)

        batch, steps = seq if seq is not None else (h.shape[0], 1)
        if hidden is None:
            hidden = self.initial_hidden(batch, h.device)
        h_seq, new_hidden = self.gru(h.reshape(batch, steps, -1), hidden)
        h = h_seq.reshape(batch * steps, -1)

        value = self.critic_head(h).squeeze(-1)
        non_targeting_logits = self.non_targeting_head(h)

        query = self.pointer_query(h).unsqueeze(1)  # [B, 1, d_model]
        pointer_scores = torch.bmm(query, token_reps.transpose(1, 2)).squeeze(1) / (self.d_model ** 0.5)
        pointer_scores = pointer_scores.masked_fill(~pointer_token_mask, -1e8)

        combined_logits = torch.cat([non_targeting_logits, pointer_scores], dim=-1)
        return combined_logits, value, new_hidden


def previous_pointer_inputs(vocab_idx, features, prev_actions, seq, n_fixed):
    """The prev_pointer input for a REPLAYED batch (training update, evaluation probe): for every step, the card index
    and feature row of the token its episode's previous recorded action pointed at, read out of that previous step's own
    token tensors. vocab_idx [B*T, tokens], features [B*T, tokens, TOKEN_FEATURE_DIM] in the episode-major layout
    seq=(B, T) the GRU uses; prev_actions: B*T raw action indices of the previous recorded action (-1 or None = none).
    Steps whose previous action was not a pointer pick (or that have no previous step) get vocab 0 and a zero row."""
    n_seq, steps = seq
    width = vocab_idx.shape[1]
    v = vocab_idx.reshape(n_seq, steps, width)
    f = features.reshape(n_seq, steps, width, -1)
    prev_v, prev_f = torch.zeros_like(v), torch.zeros_like(f)
    prev_v[:, 1:], prev_f[:, 1:] = v[:, :-1], f[:, :-1]  # the previous step's tokens, aligned to this step
    actions = torch.as_tensor([-1 if a is None else a for a in prev_actions], dtype=torch.long,
                              device=vocab_idx.device).reshape(n_seq, steps)
    is_pointer = actions >= n_fixed
    pos = (actions - n_fixed).clamp(min=0, max=width - 1)
    chosen_v = torch.gather(prev_v, 2, pos.unsqueeze(-1)).squeeze(-1) * is_pointer
    chosen_f = torch.gather(prev_f, 2, pos[:, :, None, None].expand(-1, -1, 1, f.shape[-1])).squeeze(2)
    chosen_f = chosen_f * is_pointer.unsqueeze(-1).to(chosen_f.dtype)
    return chosen_v.reshape(-1), chosen_f.reshape(n_seq * steps, -1)


# Turn number/horizon, lands-played-this-turn, mulligans-taken, am-I-turn-
# player, my/opponent life totals, my/opponent floating mana pool (2 *
# len(POOL_COLORS)), phase one-hot (len(Phase)), my/opponent library size,
# opponent's hand size, stack-targets-me/opponent, am-I-on-the-play,
# opponent mulligans-taken, opponent cleanup-discard-turns -- scalar/global
# facts not re-derived via tokens since they aren't per-card
# (rl.decision.agent._scalar_features builds this vector).
#
# 4 = turn/lands/mulligans/am-i-turn-player, +2 = my/opponent life totals,
# +5 = my/opponent library size, opponent hand size, stack-targets-me/opponent,
# +3 = on_the_play, opponent mulligans, opponent cleanup-discard turns,
# + COUNT_SCALARS_DIM exact zone/type counts (see COUNT_SCALARS_DIM)
#
# EXACT COUNTS. The token encoder cannot count by itself (attention is a weighted mean), so the true cardinalities
# the policy needs are handed over directly, per side: battlefield permanents of each card type, UNTAPPED permanents
# of each card type, graveyard size, exile size, stack entries controlled -- plus my own hand size (the opponent's is
# already a scalar above). Each is log1p(n) / log1p(COUNT_CAP).
COUNT_CAP = 60
COUNT_SCALARS_PER_SIDE = 2 * len(game.CardType) + 3
COUNT_SCALARS_DIM = 2 * COUNT_SCALARS_PER_SIDE + 1
SCALAR_FEATURE_DIM = 4 + 2 * len(game.POOL_COLORS) + len(game.turn.Phase) + 2 + 5 + 3 + COUNT_SCALARS_DIM
