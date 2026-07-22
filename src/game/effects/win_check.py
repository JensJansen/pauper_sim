"""Win-condition checking and the one function that can trigger it via
damage. Split out of the old effects_common.py because both casting.py
(a permanent entering can newly satisfy a terminated_fn) and combat.py
(damage can drop life_total to 0) need to reach _check_end_of_game, but
neither of those two needs the other -- keeping the check here, as a
shared leaf both depend on, avoids duplicating it or merging casting and
combat back into one module."""


def _check_end_of_game(state):
    """Central check for every way the game can end -- called from every
    place board state can change enough to matter: casting.enters_
    battlefield, combat.combat_damage_step, and deal_damage_to_opponent
    below. Replaces what used to be the same two lines duplicated at each
    of those call sites (see docs/MULTIPLAYER_ENGINE_PLAN.md).

    Two independent ways to win: the active player's own terminated_fn
    (their deck's combo-completion condition -- Tron assembly, a damage
    threshold) firing, or -- 2-player only -- the opponent's life_total
    hitting 0. No-ops once the game has already ended."""
    if state.turn_won is not None:
        return
    active_idx = state.active_idx
    active = state.players[active_idx]
    if active.terminated_fn is not None and active.terminated_fn(state):
        state.turn_won = state.turn_number
        state.winner = active_idx
        return
    if len(state.players) > 1 and state.opponent.life_total <= 0:
        state.turn_won = state.turn_number
        state.winner = active_idx


def deal_damage_to_opponent(state, n):
    """Every 'deals N damage to the opponent' effect routes through here
    -- the single choke point keeping state.damage_dealt (the historical
    1-player abstraction; terminated.damage_threshold_terminated and every
    burn deck's own win condition still reads it, unchanged) and the real
    per-player life_total (docs/MULTIPLAYER_ENGINE_PLAN.md) in sync. In
    1-player mode there's no second PlayerState to decrement -- the
    damage_dealt bump alone is still the whole story there, exactly as
    before this function existed."""
    state.damage_dealt += n
    if len(state.players) > 1:
        state.opponent.life_total -= n
    _check_end_of_game(state)


if __name__ == "__main__":
    # ponytail self-check: run via `python -m game.effects.win_check` from
    # src/. Both win paths directly: 1-player terminated_fn, and 2-player
    # life_total hitting 0 (deal_damage_to_opponent's own reason to exist).
    from ..state import GameState, PlayerState

    state = GameState(on_the_play=True, terminated_fn=lambda s: s.damage_dealt >= 5)
    deal_damage_to_opponent(state, 3)
    assert state.damage_dealt == 3 and state.turn_won is None
    deal_damage_to_opponent(state, 2)
    assert state.damage_dealt == 5 and state.turn_won == state.turn_number and state.winner == 0

    state2 = GameState(on_the_play=True, players=[PlayerState(True), PlayerState(False)])
    state2.players[1].life_total = 4
    deal_damage_to_opponent(state2, 3)
    assert state2.players[1].life_total == 1 and state2.turn_won is None
    deal_damage_to_opponent(state2, 1)
    assert state2.players[1].life_total == 0 and state2.turn_won == state2.turn_number and state2.winner == 0

    # No-op once already won -- a second lethal hit doesn't overwrite turn_won.
    won_turn = state2.turn_won
    deal_damage_to_opponent(state2, 10)
    assert state2.turn_won == won_turn

    print("win_check.py self-check: OK")
