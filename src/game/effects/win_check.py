"""Win-condition checking + the one damage function that can trigger it. A
shared leaf both casting.py and combat.py reach without needing each other."""


def _check_end_of_game(state):
    """Central end-of-game check, called wherever board state can change
    enough to matter. Two win paths (2-player only): the opponent's life
    hits 0, or the active player's own life hits 0 (opponent wins; a bare
    no-winner failure in 1-player). No-op once the game has ended."""
    if state.turn_won is not None:
        return
    active_idx = state.active_idx
    active = state.players[active_idx]
    if len(state.players) > 1 and state.opponent.life_total <= 0:
        state.turn_won = state.turn_number
        state.winner = active_idx
        return
    # The active player's own life at 0 is only reachable via a paid cost or
    # a self-damage effect (combat never damages the active player under
    # this engine's whole-turn model).
    if active.life_total <= 0:
        state.turn_won = state.turn_number
        state.winner = 1 - active_idx if len(state.players) > 1 else None


def _apply_life(state, player_idx, delta, reason):
    """Choke point every life-total change flows through; one log_event
    here captures all of them. `reason` ("gain"/"damage"/"cost"/...) lets a
    log reader tell them apart."""
    if not delta:
        return  # skip logging a no-op 0-life change
    player = state.players[player_idx]
    player.life_total += delta
    state.log_event("life_change", player_idx=player_idx, amount=delta,
                    new_total=player.life_total, reason=reason)


def gain_life(state, n, player_idx=None):
    """Every 'you gain N life' effect routes through here. Defaults to the
    active player; takes an explicit player_idx for the one case that
    credits someone else -- a blocker's lifelink, whose controller is the
    defending, non-active player. No _check_end_of_game call needed, since
    gaining life can never drop anyone's life to 0."""
    _apply_life(state, state.active_idx if player_idx is None else player_idx, n, "gain")


def lose_life(state, n, reason="cost"):
    """Every 'you lose / pay N life' effect routes through here -- the
    self-damage counterpart to deal_damage_to_opponent. Always the active
    player. Runs _check_end_of_game since paying life can be lethal."""
    _apply_life(state, state.active_idx, -n, reason)
    _check_end_of_game(state)


def deal_damage_to_opponent(state, n):
    """Every 'deals N damage to the opponent' effect routes through here --
    the single choke point for opponent-facing damage. A no-op in a
    1-player state (beyond the end-of-game check)."""
    if len(state.players) > 1:
        _apply_life(state, 1 - state.active_idx, -n, "damage")
    _check_end_of_game(state)


def deal_damage_to_player(state, player_idx, n):
    """"N damage to target player" for an arbitrary player -- the opponent
    or yourself. Damaging the opponent routes through
    deal_damage_to_opponent; damaging yourself is pure life loss, same
    shape as lose_life but attributed as "damage"."""
    if len(state.players) > 1 and player_idx == 1 - state.active_idx:
        deal_damage_to_opponent(state, n)
    else:
        _apply_life(state, player_idx, -n, "damage")
        _check_end_of_game(state)
