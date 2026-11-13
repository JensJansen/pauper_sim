"""Layer A7 -- hand-built timing scenarios for rows random play rarely reaches (start of the per-card scenario matrix).

Each scenario states a rule and checks the mask on both sides of its boundary, against the real rule rather than against the
engine: impulse windows (play until the deadline turn, not past it), plot (not castable the turn it was plotted),
flashback and instants at the right speed. The independent oracle is run on the same positions so the two cannot drift.
"""
import drl_env
import game
from rl.decision import action_bridge as AB
from rl.roster import TOKEN_DEFS
from support import action_oracle as O
from support import scenario as S

IMPULSE_DECK = [("Experimental Synthesizer", 2), ("Mountain", 4), ("Lightning Bolt", 4)]


def _table(decklist):
    return AB.build_fixed_action_table(decklist, token_card_defs=tuple(TOKEN_DEFS))


def _state(turn, *, hand=(), bf=("Mountain",), phase=game.turn.Phase.MAIN1):
    state = S.build(S.seat(), S.seat(bf=list(bf), hand=list(hand)))
    state.active_idx = state.turn_player_idx = 1
    state.phase = phase
    state.turn_number = turn
    return state


def _legal(state, table, label):
    row = next(i for i, r in enumerate(table) if r[0] == label)
    ours = bool(drl_env.legal_action_mask(state, table)[row])
    theirs = label in O.oracle_labels(state)
    assert ours == theirs, f"mask and oracle disagree on {label!r}: mask={ours} oracle={theirs}"
    return ours


def test_impulse_land_is_playable_through_its_deadline_turn_and_not_after():
    table = _table(IMPULSE_DECK)
    for turn, expected in ((4, True), (5, True), (6, False)):
        state = _state(turn, bf=())
        state.players[1].impulse = [(game.CARD_DEFS["Mountain"], 5)]
        assert _legal(state, table, "Play from exile: Mountain") is expected, turn


def test_impulse_spell_is_castable_through_its_deadline_turn_and_not_after():
    table = _table(IMPULSE_DECK)
    for turn, expected in ((5, True), (6, False)):
        state = _state(turn)
        state.players[1].impulse = [(game.CARD_DEFS["Lightning Bolt"], 5)]
        assert _legal(state, table, "Play from exile: Lightning Bolt") is expected, turn


def test_impulse_land_still_needs_a_land_drop_and_sorcery_timing():
    table = _table(IMPULSE_DECK)
    state = _state(5, bf=())
    state.players[1].impulse = [(game.CARD_DEFS["Mountain"], 5)]
    state.players[1].lands_played_this_turn = 1
    assert not _legal(state, table, "Play from exile: Mountain")
    state2 = _state(5, bf=(), phase=game.turn.Phase.COMBAT_DAMAGE)
    state2.players[1].impulse = [(game.CARD_DEFS["Mountain"], 5)]
    assert not _legal(state2, table, "Play from exile: Mountain")


def test_an_instant_from_exile_is_castable_at_instant_speed_but_a_sorcery_is_not():
    table = _table(IMPULSE_DECK)
    state = _state(5, phase=game.turn.Phase.END)
    state.players[1].impulse = [(game.CARD_DEFS["Lightning Bolt"], 5)]
    assert _legal(state, table, "Play from exile: Lightning Bolt")
