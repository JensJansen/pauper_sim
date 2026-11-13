"""Unit tests for drl_env._actions_common.copy_signature / distinct_copies: when are two same-named permanents the SAME choice?"""
import game
from drl_env._actions_common import copy_signature, distinct_copies
from game.state import GameState, Permanent, PlayerState


def _state(*forests):
    state = GameState(on_the_play=True, players=[PlayerState(True), PlayerState(False)])
    state.players[1].battlefield = list(forests)
    state.active_idx = 1
    return state


def _forest(slot):
    p = Permanent(game.CARD_DEFS["Forest"])
    p.slot = slot
    p.summoning_sick = False
    return p


def test_two_plain_copies_are_interchangeable_whatever_their_slot():
    a, b = _forest(1), _forest(2)
    state = _state(a, b)
    assert copy_signature(state, a) == copy_signature(state, b)
    assert distinct_copies(state, [a, b]) == [a]


def test_each_difference_that_can_change_an_outcome_separates_copies():
    def differs(mutate):
        a, b = _forest(1), _forest(2)
        mutate(b)
        state = _state(a, b)
        return copy_signature(state, a) != copy_signature(state, b)

    assert differs(lambda p: setattr(p, "tapped", True))
    assert differs(lambda p: setattr(p, "summoning_sick", True))
    assert differs(lambda p: setattr(p, "damage_marked", 1))
    assert differs(lambda p: p.counters.update({"+1/+1": 1}))
    assert differs(lambda p: setattr(p, "type_override", game.CardType.CREATURE))
    assert differs(lambda p: p.flags.update({"used_this_turn": True}))
    assert differs(lambda p: setattr(p, "temp_power", 1))
    assert differs(lambda p: p.temp_keywords.add("haste"))


def test_an_aura_on_either_battlefield_separates_copies_and_its_flags_matter():
    a, b = _forest(1), _forest(2)
    state = _state(a, b)
    sprawl = Permanent(game.CARD_DEFS["Utopia Sprawl"])
    sprawl.flags.update({"enchanting": b, "bonus_mana_color": "R"})
    state.players[0].battlefield = [sprawl]  # the OPPONENT controls the Aura
    assert copy_signature(state, a) != copy_signature(state, b)
    assert distinct_copies(state, [a, b]) == [a, b]
    sprawl.flags["bonus_mana_color"] = "U"
    other = Permanent(game.CARD_DEFS["Utopia Sprawl"])
    other.flags.update({"enchanting": a, "bonus_mana_color": "R"})
    state.players[1].battlefield.append(other)
    assert copy_signature(state, a) != copy_signature(state, b), "same Aura, different chosen color: still different copies"
