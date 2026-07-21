"""Win-condition ("terminated_fn") functions, decoupled from any specific
deck's own module -- same separation-of-concerns rewards.py already has
for reward functions, so a deck can be paired with different termination
conditions (and a different deck with the same one) without either living
inside the other's module.

Contract: terminated_fn(state) -> bool, checked after anything that could
newly make it true (a permanent entering, combat_step) -- see
game.effects_common.enters_battlefield / combat_step.
"""

import game


def tron_terminated(state):
    """Tron's win condition: controls_all_tron_types. Structurally
    different from every other deck's (not a damage threshold), so it
    keeps its own dedicated function rather than the shared factory
    below."""
    return game.controls_all_tron_types(state)


def damage_threshold_terminated(threshold):
    """Factory for the "deal N damage" win condition every burn-style deck
    built so far (spy_combo, rakdos_madness, mono_red_madness) used
    identically as 3 separate, identically-shaped functions -- e.g.
    terminated_fn=terminated.damage_threshold_terminated(20).

    threshold is stamped onto the returned function itself (not just
    closed over) so callers that only have terminated_fn in hand -- e.g.
    harness.py's evaluate(), building a log's meta header -- can read it
    back via getattr(terminated_fn, "threshold", None) without needing
    their own copy of the number."""
    def terminated_fn(state):
        return state.damage_dealt >= threshold
    terminated_fn.threshold = threshold
    return terminated_fn


# Pre-baked named instances -- configs/*.json reference terminated_fn by
# plain name (getattr off this module), so a parameterized win condition
# needs one of these added by hand per DECK_REGISTRY_REFRESH_PLAN.md's
# "no structured spec" decision. Add only what an actual config needs --
# no speculative thresholds.
damage_threshold_20_terminated = damage_threshold_terminated(20)


if __name__ == "__main__":
    # ponytail self-check: no pytest in this project, mirrors the
    # assert-based demo convention -- run via `python terminated.py` from
    # src/.
    from game.state import GameState

    state = GameState(on_the_play=True)
    state.damage_dealt = 19
    terminated_20 = damage_threshold_terminated(20)
    assert terminated_20.threshold == 20
    assert not terminated_20(state)
    state.damage_dealt = 20
    assert terminated_20(state)

    # Independent thresholds don't share state -- confirms this is a
    # factory, not a single shared closure: at damage_dealt=5, the
    # threshold-5 function is satisfied and the threshold-20 one isn't.
    terminated_5 = damage_threshold_terminated(5)
    assert terminated_5.threshold == 5
    state.damage_dealt = 5
    assert terminated_5(state)
    assert not terminated_20(state)

    # Pre-baked named instance matches its own factory call exactly.
    state.damage_dealt = 19
    assert not damage_threshold_20_terminated(state)
    state.damage_dealt = 20
    assert damage_threshold_20_terminated(state)

    print("terminated.py self-check: OK")
