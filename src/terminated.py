"""Win-condition ("terminated_fn") functions, decoupled from any specific
deck's own module -- same separation-of-concerns rewards.py already has
for reward functions, so a deck can be paired with different termination
conditions (and a different deck with the same one) without either living
inside the other's module.

Contract: terminated_fn(state) -> bool, checked after anything that could
newly make it true (a permanent entering, combat damage) -- see
game.effects.casting.enters_battlefield / game.effects.combat.combat_damage_step.
"""

import game


def never_terminated(state):
    """Always False -- for a real 2-player adversarial config where the
    ONLY win condition should be the engine's own real one (an opponent's
    life_total hitting 0, or decking out): every other terminated_fn here
    is a 1-player heuristic proxy for "no real opponent to actually beat"
    (Tron assembly, a damage threshold), which game.effects.win_check.
    _check_end_of_game checks ahead of the real life_total check -- fine
    when it's the only way to win, wrong once a real opponent (and life
    total) exists. Since _check_end_of_game already treats terminated_fn
    as fully optional (`if active.terminated_fn is not None and ...`),
    this is the explicit way to opt out of it rather than the two paths
    racing to fire first."""
    return False


def tron_terminated(state):
    """Tron's win condition: controls_all_tron_types. Structurally
    different from every other deck's (not a damage threshold), so it
    keeps its own dedicated function rather than the shared factory
    below."""
    return game.controls_all_tron_types(state)


def damage_threshold_terminated(threshold=20):
    """Factory for the "deal N damage" win condition every burn-style deck
    built so far (spy_combo, rakdos_madness, mono_red_madness) used
    identically as 3 separate, identically-shaped functions -- e.g.
    terminated_fn=terminated.damage_threshold_terminated(). Defaults to
    20 (every real config today uses it); pass a different threshold for
    a deck that actually needs one.

    threshold is stamped onto the returned function itself (not just
    closed over) so callers that only have terminated_fn in hand -- e.g.
    harness.py's evaluate(), building a log's meta header -- can read it
    back via getattr(terminated_fn, "threshold", None) without needing
    their own copy of the number."""
    def terminated_fn(state):
        return state.damage_dealt >= threshold
    terminated_fn.threshold = threshold
    return terminated_fn


# Pre-baked named instance -- configs/*.json reference terminated_fn by
# plain name (getattr off this module), so a parameterized win condition
# needs one of these added by hand per DECK_REGISTRY_REFRESH_PLAN.md's
# "no structured spec" decision. Add only what an actual config needs --
# no speculative thresholds.
damage_threshold_20_terminated = damage_threshold_terminated()


if __name__ == "__main__":
    # ponytail self-check: no pytest in this project, mirrors the
    # assert-based demo convention -- run via `python terminated.py` from
    # src/.
    from game.state import GameState

    state = GameState(on_the_play=True)
    state.damage_dealt = 19
    terminated_20 = damage_threshold_terminated()  # relies on the default
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

    # never_terminated: always False, regardless of board state -- the
    # real 2-player win condition (life_total/deck-out, checked separately
    # by game.effects.win_check._check_end_of_game) is the only way to win.
    assert not never_terminated(state)
    state.damage_dealt = 10 ** 9
    assert not never_terminated(state)

    print("terminated.py self-check: OK")
