"""Win-condition ("terminated_fn") functions, decoupled from any specific
deck's own module -- same separation-of-concerns rewards.py already has for
reward functions, so a deck can be paired with a termination condition
without either living inside the other's module.

Contract: terminated_fn(state) -> bool, checked after anything that could
newly make it true (a permanent entering, combat damage) -- see
game.effects.casting.enters_battlefield / game.effects.combat.combat_damage_step.

Only never_terminated remains: the deck-specific 1-player win proxies (Tron
assembly, a damage threshold) went away with the 1-player pipeline. Every
2-player config wins the engine's own way -- an opponent's life_total hitting
0, or decking out -- so terminated_fn just opts out.
"""


def never_terminated(state):
    """Always False -- for a 2-player adversarial config where the ONLY win
    condition is the engine's own real one (an opponent's life_total hitting
    0, or decking out). game.effects.win_check._check_end_of_game already
    treats terminated_fn as fully optional (`if active.terminated_fn is not
    None and ...`), so this is the explicit way to opt out of it."""
    return False


if __name__ == "__main__":
    # ponytail self-check: run via `python terminated.py` from src/.
    from game.state import GameState

    state = GameState(on_the_play=True)
    assert not never_terminated(state)  # always False, regardless of board state
    print("terminated.py self-check: OK")
