"""Per-deck trunk + critic + pointer-network action head, sitting on top of
a SHARED SetTransformer+FiLM perception stack (rl.arch) that this
module reuses, never owns (Phase 4's pretrain-then-freeze design: one
shared stack, N independent per-deck DeckNetwork instances built on top of
it).

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


class DeckNetwork(nn.Module):
    def __init__(self, shared_stack, film_condition_dim, non_targeting_n_actions, trunk_hidden=(128, 128)):
        super().__init__()
        self.shared_stack = shared_stack  # reference, not a copy -- see this module's own docstring
        d_model = shared_stack.d_model
        trunk_input_dim = d_model + SCALAR_FEATURE_DIM  # mine_summary + non-tokenized globals (life, turn, phase, ...)

        self.trunk_layers = nn.ModuleList()
        prev = trunk_input_dim
        for width in trunk_hidden:
            self.trunk_layers.append(nn.Linear(prev, width))
            prev = width
        self.film = FiLM(condition_dim=film_condition_dim, layer_dims=list(trunk_hidden))
        self.activation = nn.Tanh()

        self.critic_head = nn.Linear(prev, 1)
        self.non_targeting_head = nn.Linear(prev, non_targeting_n_actions)
        self.pointer_query = nn.Linear(prev, d_model)
        self.d_model = d_model

    def forward(self, mine_summary, theirs_summary, scalar_features, token_reps, pointer_token_mask):
        """mine_summary/theirs_summary/token_reps: shared_stack's own
        outputs (caller runs shared_stack separately -- see this module's
        docstring on why it's a reference, not owned here). scalar_features:
        [B, SCALAR_FEATURE_DIM] -- the non-tokenized globals (life totals,
        turn number, phase one-hot, pending-kind one-hot, mana pool -- the
        same non-per-card globals the scalar-feature builder
        (rl.train._scalar_features) carries). pointer_token_mask:
        [B, T] bool, True where a token is a currently-legal POINTER TARGET
        for whatever specific resolution is pending right now (Attack-
        eligible creature, block-eligible creature, etc.) -- this is
        legality, computed by the caller from the real game engine
        (game.creature_attack_eligible et al.), never guessed by the
        network itself.

        Returns (combined_logits [B, non_targeting_n_actions + T],
        value [B]) -- combined_logits' first non_targeting_n_actions
        entries are the fixed-table actions (masked externally by the
        caller, same legal_action_mask contract drl_env.py already uses for
        that half), the remaining T entries are one pointer score per token
        position (masked here internally via pointer_token_mask, since
        those aren't a fixed-identity action table the caller can mask by
        name -- masking IS this function's own job for that half)."""
        h = torch.cat([mine_summary, scalar_features], dim=-1)
        film_params = self.film(theirs_summary)
        for layer, (gamma, beta) in zip(self.trunk_layers, film_params):
            h = self.activation(gamma * layer(h) + beta)

        value = self.critic_head(h).squeeze(-1)
        non_targeting_logits = self.non_targeting_head(h)

        query = self.pointer_query(h).unsqueeze(1)  # [B, 1, d_model]
        pointer_scores = torch.bmm(query, token_reps.transpose(1, 2)).squeeze(1) / (self.d_model ** 0.5)
        pointer_scores = pointer_scores.masked_fill(~pointer_token_mask, -1e8)

        combined_logits = torch.cat([non_targeting_logits, pointer_scores], dim=-1)
        return combined_logits, value


# Turn number/horizon, lands-played-this-turn, mulligans-taken, am-I-turn-
# player, my/opponent life totals, floating mana pool (len(POOL_COLORS)),
# phase one-hot (len(Phase)) -- genuinely scalar/global facts, deliberately
# NOT re-derived via tokens since they aren't per-card ones.
import game  # noqa: E402
from rl.arch import FiLM  # noqa: E402

SCALAR_FEATURE_DIM = 4 + len(game.POOL_COLORS) + len(game.turn.Phase) + 2  # 4 = turn/lands/mulligans/am-i-turn-player, +2 = my/opponent life totals


if __name__ == "__main__":
    # ponytail self-check: run via `python rl.deck` from src/.
    import game as _game
    from rl.features import CardVocab, build_token_set
    from rl.arch import SetTransformer, pad_token_batch
    from game.state import GameState, PlayerState, Permanent

    decklist_a = _game.parse_decklist_file("../data/mono_red_madness.txt")
    decklist_b = _game.parse_decklist_file("../data/rakdos_madness.txt")
    vocab = CardVocab([decklist_a, decklist_b])

    seat0 = PlayerState(on_the_play=True)
    seat0.battlefield = [Permanent(_game.CARD_DEFS["Guttersnipe"]), Permanent(_game.CARD_DEFS["Voldaren Epicure"])]
    seat1 = PlayerState(on_the_play=False)
    seat1.battlefield = [Permanent(_game.CARD_DEFS["Kitchen Imp"])]
    state = GameState(on_the_play=True, players=[seat0, seat1])

    tokens = build_token_set(state, 0, vocab)
    vocab_idx, features, mask, identities = pad_token_batch([tokens])
    side_flag = features[:, :, -1]

    shared = SetTransformer(vocab.size, d_model=16, n_heads=2, n_layers=1, dim_feedforward=32)
    mine_summary, theirs_summary, token_reps = shared(vocab_idx, features, mask, side_flag)

    n_fixed_actions = 12  # arbitrary stand-in for a real per-deck fixed table's size
    net = DeckNetwork(shared, film_condition_dim=16, non_targeting_n_actions=n_fixed_actions, trunk_hidden=(32, 32))

    scalar_features = torch.zeros(1, SCALAR_FEATURE_DIM)
    # Only my own two battlefield permanents (Guttersnipe, Voldaren Epicure)
    # are legal "Attack:" pointer targets in this synthetic example -- the
    # opponent's Kitchen Imp and any padded slots must be masked out.
    pointer_mask = torch.zeros_like(mask)
    for t, ident in enumerate(identities[0]):
        pointer_mask[0, t] = ident is not None and ident in seat0.battlefield

    logits, value = net(mine_summary, theirs_summary, scalar_features, token_reps, pointer_mask)
    assert logits.shape == (1, n_fixed_actions + vocab_idx.shape[1])
    assert value.shape == (1,)
    assert torch.isfinite(value).all()

    # The two legal pointer positions must be finite/selectable; the
    # opponent's Kitchen Imp position must be masked to -inf (never sampled).
    pointer_logits = logits[0, n_fixed_actions:]
    legal_positions = pointer_mask[0].nonzero(as_tuple=True)[0]
    illegal_positions = (~pointer_mask[0]).nonzero(as_tuple=True)[0]
    assert torch.isfinite(pointer_logits[legal_positions]).all()
    assert (pointer_logits[illegal_positions] <= -1e7).all(), "masked pointer positions must be effectively -inf"

    # A full masked-categorical sample over the COMBINED action set must
    # only ever land on a legal position (either a legal fixed action or a
    # legal pointer target) -- exercising the actual masking contract this
    # network exists to support, not just checking shapes.
    fixed_mask = torch.zeros(1, n_fixed_actions, dtype=torch.bool)
    fixed_mask[0, 0] = True  # pretend exactly one fixed action ("Pass") is legal too
    full_mask = torch.cat([fixed_mask, pointer_mask], dim=-1)
    masked_logits = logits.masked_fill(~full_mask, -1e8)
    dist = torch.distributions.Categorical(logits=masked_logits)
    for _ in range(50):
        action = dist.sample()
        assert full_mask[0, action.item()], f"sampled an illegal combined action index {action.item()}"

    print(f"rl.deck self-check: OK (combined_logits={logits.shape}, value={value.shape})")
