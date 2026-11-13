"""The contract between the policy stack and the engine submodule's action space.

The engine (drl_env.action_bridge.pointer_identities) lists what a pointer can address WITHOUT an observation; the policy
(rl.model.features.build_token_set) lists the same objects as its tokens, plus identity-less ones. A pointer the engine
would offer but the policy cannot see (or the reverse) is a decision the policy can never take / a mask the bridge
cannot apply, so the two must agree on every real decision."""
import random

import drl_env
import game
from rl.decision import action_bridge as rl_bridge
from rl.decision.agent import RandomMulligan, _build_decision
from rl.model.features import build_token_set
from rl.roster import TOKEN_DEFS, build_pool


def test_the_rl_bridge_is_the_engines_bridge():
    from drl_env import action_bridge as engine_bridge
    for name in ("build_fixed_action_table", "pointer_legal_mask", "execute_pointer_choice", "any_pointer_legal",
                 "pointer_kind", "pointer_identities"):
        assert getattr(rl_bridge, name) is getattr(engine_bridge, name), name
    assert TOKEN_DEFS is engine_bridge.TOKEN_DEFS


def test_pointer_identities_match_the_observations_identity_tokens_on_real_games():
    decklists, vocab, ctxs, _tables = build_pool()
    names = sorted(decklists)[:4]
    checked = [0]
    for gi, (a, b) in enumerate((x, y) for x in names for y in names):
        rng, pick = random.Random(gi), random.Random(gi + 99)
        mull = RandomMulligan(random.Random(gi))
        seats = (a, b)

        def choose(state, seats=seats, mull=mull, pick=pick):
            pend = state.pending_resolution
            if pend is not None and pend["kind"] in ("mulligan_decision", "mulligan_bottom"):
                return mull.decide(state)
            seat = state.active_idx
            dec = _build_decision(state, seat, ctxs[seats[seat]], 10 ** 9)
            tokens = build_token_set(state, seat, vocab, include_rows=False)
            policy_ids = [ident for _i, _r, ident in tokens if ident is not None]
            engine_ids = drl_env.action_bridge.pointer_identities(state, seat)
            assert len(policy_ids) == len(engine_ids) and all(p is e for p, e in zip(policy_ids, engine_ids, strict=True)), (
                f"pointer identity lists diverge in kind {pend['kind'] if pend else None!r}")
            checked[0] += 1
            legal = [i for i, ok in enumerate(dec.full_mask) if ok]
            i = pick.choice(legal)
            if i < dec.n_fixed:
                if dec.fixed_table[i][0] == "Pass":
                    return None
                return lambda: dec.fixed_table[i][2](state)
            chosen = dec.identities[i - dec.n_fixed]
            return lambda: rl_bridge.execute_pointer_choice(state, chosen)

        game.run_multiplayer_game([decklists[a], decklists[b]], rng, gi % 2, choose, horizon=8, combat_enabled=True)
    assert checked[0] > 500
