"""Per-deck perception encoder + trunk + critic + pointer-network action
head. Each deck owns its OWN SetTransformer (rl.model.arch) as a registered child
module, trained end to end with the policy by the same PPO update.

This replaced a shared, pretrained-then-frozen encoder (one SetTransformer
for the whole league, built by a separate run_pretrain.py phase and frozen
before league training began, 2026-08-17). Two things were wrong with it.
The encoder carried the entire burden of telling 139 distinct card effects
apart -- the structured features of the day collapsed 141 cards onto 78
distinct vectors, so the learned card embedding was the only thing
separating Lightning Bolt from Lava Dart -- and it was frozen after ~2,000
games/deck, then never updated across 12,000+ games of league play. The
component that had to learn the most was trained the least. Structured
features got much richer at the same time (rl.model.features), so the encoder is
no longer carrying that alone either.

Consequences worth knowing, all of them deliberate:

  - Every checkpoint is SELF-CONTAINED. A live.pt/snapshot holds its own
    encoder, so any two populations can play each other with no
    "were these trained against the same stack" guard, and re-training an
    encoder no longer invalidates cross-population comparisons.
  - PPO recomputes the encoder forward every minibatch of every epoch
    instead of caching one frozen forward per update (rl.training.ppo). That is a
    real throughput cost, accepted knowingly.
  - Card knowledge is no longer shared between decks: a deck learns its
    opponents' cards only from games against them.

One encoder per DeckNetwork. Passing the SAME SetTransformer instance to
two DeckNetworks would put it in both optimizers and train it twice per
round -- the exact coupling this design removes (see
tests/rl/test_deck.py's own check that the league builds distinct
encoders).

Action space, per deck: the UNION of a small FIXED table (non-targeting
actions -- Play land, Cast X, Pass, mana payment choices, mulligan
decisions, etc. -- legitimately fixed-shape since a deck's OWN card pool
never changes at inference time under this design) and a POINTER-scored
set (Attack/Assign Blocker/Choose target/Choose opponent's -- the four
action categories drl_env.build_action_table names with a "(name) (slot k)"
suffix, replaced here because their COUNT depends on the opponent's own
decklist, which this whole migration exists to stop coupling to).

Both halves are scored into ONE combined logit vector before a single
softmax -- not two independent distributions -- because at any real
decision point both kinds of action can be simultaneously legal (e.g. "Play
land" and "Attack: X" during the same declare-attackers window), and
sampling/log-prob/entropy all need to be computed over the TRUE combined
legal set, not two separately-normalized halves that would double-count
probability mass."""

import torch
import torch.nn as nn

import game
from rl.model.arch import FiLM

# Width of the previous-action embedding fed into the GRU (see DeckNetwork).
PREV_ACTION_DIM = 16


class DeckNetwork(nn.Module):
    def __init__(self, encoder, film_condition_dim, non_targeting_n_actions, trunk_hidden=(128, 128)):
        super().__init__()
        # A plain registered child, so net.parameters() covers the encoder for
        # free -- the optimizer trains it, clip_grad_norm_ clips it, and
        # net.state_dict() carries it into every checkpoint with no separate
        # file to keep in sync. The previous design deliberately did the
        # opposite (object.__setattr__, to keep a SHARED frozen stack out of
        # every checkpoint and out of three call sites that could mutate it);
        # with one encoder per deck there is nothing shared to protect, and
        # ownership is what makes a checkpoint self-contained.
        self.encoder = encoder
        d_model = encoder.d_model
        trunk_input_dim = d_model + SCALAR_FEATURE_DIM  # mine_summary + non-tokenized globals (life, turn, phase, ...)

        self.trunk_layers = nn.ModuleList()
        prev = trunk_input_dim
        for width in trunk_hidden:
            self.trunk_layers.append(nn.Linear(prev, width))
            prev = width
        self.film = FiLM(condition_dim=film_condition_dim, layer_dims=list(trunk_hidden))
        self.activation = nn.Tanh()

        # Recurrent memory, between the trunk and every head -- so the critic,
        # the fixed-action head AND the pointer query are all history-aware,
        # not just one of them. The observation is otherwise strictly Markov:
        # it describes the board right now and carries nothing about what has
        # happened, which is what makes "they held two blue up and passed"
        # unrepresentable no matter how rich the per-card features get.
        #
        # GRU rather than LSTM: one state tensor instead of two, for no
        # measured loss at this size. Recurrence rather than a transformer over
        # stacked history: recurrence remains the stronger and more stable
        # choice under partial observability (arxiv 2405.17358), and stacking N
        # observations would multiply this deck's encoder cost by N.
        #
        # The agent's OWN previous action, embedded and fed alongside the
        # observation -- standard practice for recurrent RL policies (R2D2,
        # IMPALA). Not a fact handed to the agent about hidden state: it is
        # self-knowledge, something it necessarily already had.
        #
        # It earns its place because removing known_top (2026-08-17) left
        # Brainstorm's placement recoverable ONLY as a hand-token delta between
        # consecutive frames -- "a card left my hand during a put_on_top, so it
        # is on top of my library now". Naming the action directly turns that
        # inference into a lookup.
        #
        # Two symbols past the fixed table: n_fixed for "some POINTER action"
        # and n_fixed+1 for "no previous action" (the first decision of a
        # game). Every pointer action collapses to one symbol on purpose -- a
        # pointer index means a different token from one state to the next, so
        # it carries no stable identity worth embedding, and what was actually
        # pointed AT is visible on the board anyway (the creature is now tapped
        # and attacking).
        self.n_fixed = non_targeting_n_actions
        self.prev_action_embed = nn.Embedding(non_targeting_n_actions + 2, PREV_ACTION_DIM)
        self.gru = nn.GRU(prev + PREV_ACTION_DIM, prev, batch_first=True)
        self.hidden_size = prev

        self.critic_head = nn.Linear(prev, 1)
        self.non_targeting_head = nn.Linear(prev, non_targeting_n_actions)
        self.pointer_query = nn.Linear(prev, d_model)
        self.d_model = d_model

    def initial_hidden(self, batch_size=1, device="cpu"):
        """The per-GAME starting recurrent state: zeros.

        Every sequence -- during rollout AND during the update -- starts here,
        which is what makes replay EXACT rather than approximate. ppo_update
        replays whole episodes from this same zero state, so the hidden states
        it recomputes are the ones collection would have produced under the
        current weights; there is no stale-hidden-state error to correct and
        therefore no need for stored states or R2D2-style burn-in."""
        return torch.zeros(1, batch_size, self.hidden_size, device=device)

    def prev_action_symbols(self, raw_actions, device="cpu"):
        """Raw combined-action indices -> prev_action_embed symbols.

        `raw_actions` is a sequence of ints, one per batch row, where None (or
        any negative value) means "no previous action". Anything at or past the
        fixed table is a POINTER action and collapses to a single symbol -- see
        the embedding's own comment for why that is deliberate rather than
        lossy."""
        symbols = [
            self.n_fixed + 1 if a is None or a < 0 else (self.n_fixed if a >= self.n_fixed else a)
            for a in raw_actions
        ]
        return torch.as_tensor(symbols, dtype=torch.long, device=device)

    def forward(self, mine_summary, theirs_summary, scalar_features, token_reps, pointer_token_mask,
                hidden=None, seq=None, prev_action=None):
        """mine_summary/theirs_summary/token_reps: this deck's own encoder's
        outputs. The caller runs self.encoder separately rather than this
        method calling it, because rl.decision.agent needs token_reps and the padded
        identities in hand to build the pointer mask before scoring, and the
        same three outputs feed the mulligan net's own pass. scalar_features:
        [B, SCALAR_FEATURE_DIM] -- the non-tokenized globals (life totals,
        turn number, phase one-hot, my/opponent mana pool, library sizes,
        opponent hand size, stack-targets-me/opponent -- the same non-per-card globals the
        scalar-feature builder (rl.decision.agent._scalar_features) carries, see its
        own docstring for the exact composition). pointer_token_mask:
        [B, T] bool, True where a token is a currently-legal POINTER TARGET
        for whatever specific resolution is pending right now (Attack-
        eligible creature, block-eligible creature, etc.) -- this is
        legality, computed by the caller from the real game engine
        (game.creature_attack_eligible et al.), never guessed by the
        network itself.

        prev_action: [B*T] long tensor of prev_action_embed SYMBOLS (build it
        with prev_action_symbols, which maps raw action indices), or None for
        "no previous action" on every row. hidden: [1, B, hidden_size]
        recurrent state, or None for a fresh zero state (see initial_hidden). seq: optional (B, T) naming the
        sequence layout of the FLAT batch. Every tensor above stays flat
        [B*T, ...] because the encoder, the trunk and all three heads are
        timestep-independent -- only the GRU needs a time axis, so only the
        GRU's input is reshaped. seq=None means one timestep each (B = batch,
        T = 1), which is the rollout shape.

        A flat batch passed with seq=(B, T) MUST be laid out episode-major:
        all T steps of episode 0, then all of episode 1, and so on. That is
        what `.reshape(B, T, -1)` assumes, and getting it backwards would
        silently train the GRU on interleaved episodes.

        Returns (combined_logits [B*T, non_targeting_n_actions + tokens],
        value [B*T], new_hidden [1, B, hidden_size]) -- combined_logits'
        first non_targeting_n_actions entries are the fixed-table actions
        (masked externally by the caller, same legal_action_mask contract
        drl_env already uses for that half), the remaining entries are one
        pointer score per token position (masked here internally via
        pointer_token_mask, since those aren't a fixed-identity action table
        the caller can mask by name -- masking IS this function's own job for
        that half)."""
        h = torch.cat([mine_summary, scalar_features], dim=-1)
        film_params = self.film(theirs_summary)
        for layer, (gamma, beta) in zip(self.trunk_layers, film_params):
            h = self.activation(gamma * layer(h) + beta)

        # The agent's own previous action rides in with the observation. None
        # means "no previous action" for every row -- the first decision of a
        # game, and the default for callers that do not track it.
        if prev_action is None:
            prev_action = self.prev_action_symbols([None] * h.shape[0], h.device)
        h = torch.cat([h, self.prev_action_embed(prev_action)], dim=-1)

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


# Turn number/horizon, lands-played-this-turn, mulligans-taken, am-I-turn-
# player, my/opponent life totals, my/opponent floating mana pool (2 *
# len(POOL_COLORS) -- floating mana is public information in real Magic,
# same reasoning library/hand size below already follows), phase one-hot
# (len(Phase)), my/opponent library size, opponent's hand size, stack-
# targets-me/stack-targets-opponent, and (2026-08-13) am-I-on-the-play,
# opponent mulligans-taken, opponent cleanup-discard-turns -- genuinely
# scalar/global facts, deliberately NOT re-derived via tokens since they
# aren't per-card ones (rl.decision.agent._scalar_features is the one place that
# builds this vector).
#
# The last three were added together because all three are PUBLIC information
# a real player has at the table and none of them was reaching the policy.
# on_the_play in particular was a plain oversight: rl.model.mulligan.MulliganNet has
# always received it while the main policy never did.
#
# 4 = turn/lands/mulligans/am-i-turn-player, +2 = my/opponent life totals,
# +5 = my/opponent library size, opponent hand size, stack-targets-me/opponent,
# +3 = on_the_play, opponent mulligans, opponent cleanup-discard turns
SCALAR_FEATURE_DIM = 4 + 2 * len(game.POOL_COLORS) + len(game.turn.Phase) + 2 + 5 + 3
