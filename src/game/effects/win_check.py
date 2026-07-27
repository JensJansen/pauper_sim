"""Win-condition checking + the one damage function that can trigger it. A
shared leaf both casting.py (enters_battlefield's co-located end-of-game check)
and combat.py (damage can drop life to 0) reach without needing each other."""


def _check_end_of_game(state):
    """Central end-of-game check, called wherever board state can change enough
    to matter (enters_battlefield, combat_damage_step, deal_damage_to_opponent,
    lose_life). Two win paths (2-player only): the opponent's life hits 0, or the
    active player's OWN life hits 0 (opponent wins; a bare no-winner failure in
    1-player). No-op once the game has ended."""
    if state.turn_won is not None:
        return
    active_idx = state.active_idx
    active = state.players[active_idx]
    if len(state.players) > 1 and state.opponent.life_total <= 0:
        state.turn_won = state.turn_number
        state.winner = active_idx
        return
    # Active player's own life at 0 -- only reachable via life paid as a cost
    # or lost to a self-damage effect (lose_life); combat never damages the
    # active player under this engine's whole-turn model, so this branch was
    # inert (and unwritten) until self-damage existed. The opponent wins in a
    # 2-player game; 1-player has no one to award it to, so it's a bare
    # failure with winner=None, same as decking out.
    if active.life_total <= 0:
        state.turn_won = state.turn_number
        state.winner = 1 - active_idx if len(state.players) > 1 else None


def _apply_life(state, player_idx, delta, reason):
    """The single choke point every life-total change flows through, so one
    log_event here captures all of them (combat, burn, lifegain, life paid as a
    cost). `reason` ("gain"/"damage"/"cost"/...) lets a log reader tell them
    apart -- a combat loss also carries its own co-located combat_damage event,
    so the two "damage" cases stay distinguishable."""
    if not delta:
        return  # a 0-life change is a no-op -- skip it (combat's own
                # deal_damage_to_opponent(0) when nothing connects would
                # otherwise spam the log with +0 entries)
    player = state.players[player_idx]
    player.life_total += delta
    state.log_event("life_change", player_idx=player_idx, amount=delta,
                    new_total=player.life_total, reason=reason)


def gain_life(state, n, player_idx=None):
    """Every 'you gain N life' effect routes through here -- mirrors
    deal_damage_to_opponent's own role as the choke point for the other
    direction. Defaults to the active player (state.life_total's own
    _active_player_property target), so it credits whoever is resolving the
    effect in both 1- and 2-player games; takes an explicit player_idx for
    the one case that credits someone else -- a BLOCKER's lifelink, whose
    controller is the defending, non-active player (combat._blocker_deal_
    damage). No _check_end_of_game call needed, since gaining life can never
    drop anyone's life to 0."""
    _apply_life(state, state.active_idx if player_idx is None else player_idx, n, "gain")


def lose_life(state, n, reason="cost"):
    """Every 'you lose / pay N life' effect routes through here -- the self-
    damage counterpart to deal_damage_to_opponent's opponent-facing loss,
    for the eventual generic 'pay X life' cost and cards like fetchland
    pain / Phyrexian mana / Dark Confidant. Always the active player
    (whoever is resolving the effect or paying the cost). Runs
    _check_end_of_game because -- unlike gaining life -- paying life CAN be
    lethal (a fetch at 1 life, Bob flipping a big drop), and that self-death
    is a real loss the engine must register (see _check_end_of_game's own
    active-life branch)."""
    _apply_life(state, state.active_idx, -n, reason)
    _check_end_of_game(state)


def deal_damage_to_opponent(state, n):
    """Every 'deals N damage to the opponent' effect routes through here --
    the single choke point for opponent-facing damage, decrementing the
    opponent's real per-player life_total. The opponent is players[1 -
    active_idx] (== state.opponent, but reached by index so _apply_life can
    log which player took the hit). A no-op in a 1-player state (no opponent
    to damage) beyond the end-of-game check."""
    if len(state.players) > 1:
        _apply_life(state, 1 - state.active_idx, -n, "damage")
    _check_end_of_game(state)


def deal_damage_to_player(state, player_idx, n):
    """Real Magic's "N damage to target player" for an ARBITRARY player --
    the opponent OR yourself (Lightning Bolt to your own face is legal, if
    rarely wise). Damaging the opponent routes through deal_damage_to_opponent;
    damaging yourself is pure life loss, same shape as lose_life but attributed
    as "damage". Both go through _apply_life + _check_end_of_game, so a lethal
    burn either way still registers the game end."""
    if len(state.players) > 1 and player_idx == 1 - state.active_idx:
        deal_damage_to_opponent(state, n)
    else:
        _apply_life(state, player_idx, -n, "damage")
        _check_end_of_game(state)


if __name__ == "__main__":
    # ponytail self-check: run via `python -m game.effects.win_check` from
    # src/. Every win path directly: the 2-player life_total hitting 0
    # (deal_damage_to_opponent's reason to exist), self-death, and the
    # life_change event log.
    from ..state import GameState, PlayerState

    state = GameState(on_the_play=True, players=[PlayerState(True), PlayerState(False)])

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

    gain_life(state, 3)
    assert state.life_total == 23  # STARTING_LIFE (20) + 3

    # lose_life (self-damage / life-as-a-cost) + the life_change event log.
    events = []
    s3 = GameState(on_the_play=True, players=[PlayerState(True), PlayerState(False)], event_log=events)
    lose_life(s3, 5, reason="pay_life")
    assert s3.players[0].life_total == 15 and s3.turn_won is None
    gain_life(s3, 2)                                    # active gains
    assert s3.players[0].life_total == 17
    gain_life(s3, 3, player_idx=1)                      # blocker-lifelink shape: credit the OTHER player
    assert s3.players[1].life_total == 23
    deal_damage_to_opponent(s3, 4)
    assert s3.players[1].life_total == 19
    lose_life(s3, 17)                                   # active pays itself to 0 -> opponent (idx 1) wins
    assert s3.players[0].life_total == 0 and s3.turn_won == s3.turn_number and s3.winner == 1

    # every life change emitted one life_change event, in order, each with the
    # affected player, signed amount, resulting total, and reason category.
    life_events = [(e["player_idx"], e["amount"], e["new_total"], e["reason"])
                   for e in events if e["kind"] == "life_change"]
    assert life_events == [
        (0, -5, 15, "pay_life"), (0, 2, 17, "gain"), (1, 3, 23, "gain"),
        (1, -4, 19, "damage"), (0, -17, 0, "cost"),
    ], life_events

    # 1-player self-death is a bare failure -- no opponent to award the win to.
    s1 = GameState(on_the_play=True)
    lose_life(s1, 20)
    assert s1.life_total == 0 and s1.turn_won == s1.turn_number and s1.winner is None

    print("win_check.py self-check: OK")
