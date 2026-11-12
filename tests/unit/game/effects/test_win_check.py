"""Life-total changes (gain/loss/damage) and the resulting win check."""
from game.effects.win_check import deal_damage_to_opponent, gain_life, lose_life
from game.state import GameState, PlayerState


def test_deal_damage_to_opponent_lethal_wins_and_is_idempotent():
    state2 = GameState(on_the_play=True, players=[PlayerState(True), PlayerState(False)])
    state2.players[1].life_total = 4
    deal_damage_to_opponent(state2, 3)
    assert state2.players[1].life_total == 1 and state2.turn_won is None
    deal_damage_to_opponent(state2, 1)
    assert state2.players[1].life_total == 0 and state2.turn_won == state2.turn_number and state2.winner == 0

    # a second lethal hit after the win doesn't overwrite turn_won
    won_turn = state2.turn_won
    deal_damage_to_opponent(state2, 10)
    assert state2.turn_won == won_turn


def test_gain_life_credits_active_player():
    state = GameState(on_the_play=True, players=[PlayerState(True), PlayerState(False)])
    gain_life(state, 3)
    assert state.life_total == 23  # STARTING_LIFE (20) + 3


def test_lose_life_and_life_change_event_log():
    events = []
    s3 = GameState(on_the_play=True, players=[PlayerState(True), PlayerState(False)], event_log=events)
    lose_life(s3, 5, reason="pay_life")
    assert s3.players[0].life_total == 15 and s3.turn_won is None
    gain_life(s3, 2)
    assert s3.players[0].life_total == 17
    gain_life(s3, 3, player_idx=1)                      # credit the OTHER player, e.g. blocker lifelink
    assert s3.players[1].life_total == 23
    deal_damage_to_opponent(s3, 4)
    assert s3.players[1].life_total == 19
    lose_life(s3, 17)                                   # active pays itself to 0 -> opponent (idx 1) wins
    assert s3.players[0].life_total == 0 and s3.turn_won == s3.turn_number and s3.winner == 1

    # one life_change event per change, in order
    life_events = [(e["player_idx"], e["amount"], e["new_total"], e["reason"])
                   for e in events if e["kind"] == "life_change"]
    assert life_events == [
        (0, -5, 15, "pay_life"), (0, 2, 17, "gain"), (1, 3, 23, "gain"),
        (1, -4, 19, "damage"), (0, -17, 0, "cost"),
    ], life_events


def test_one_player_self_death_is_bare_failure():
    # no opponent to award the win to
    s1 = GameState(on_the_play=True)
    lose_life(s1, 20)
    assert s1.life_total == 0 and s1.turn_won == s1.turn_number and s1.winner is None
