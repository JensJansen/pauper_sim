"""Migrated from game/effects/shared.py's __main__ ponytail self-check."""
import pytest

from game.cards import CardDef, CardType, EffectId
from game.effects.shared import any_creature_on_battlefield, discard_from_hand_to_graveyard, find_to_hand
from game.state import GameState


def _make_state():
    state = GameState(on_the_play=True)
    state.library = [
        CardDef("Forest", CardType.LAND, None, EffectId.FOREST),
        CardDef("Mountain", CardType.LAND, None, EffectId.MOUNTAIN),
    ]
    return state


def test_any_creature_on_battlefield_empty():
    state = _make_state()
    assert not any_creature_on_battlefield(state)


def test_find_to_hand_found():
    state = _make_state()
    find_to_hand(state, "Forest")
    assert [c.name for c in state.hand] == ["Forest"]
    assert [c.name for c in state.library] == ["Mountain"]


def test_find_to_hand_declined_still_shuffles():
    state = _make_state()
    find_to_hand(state, "Forest")
    find_to_hand(state, None)  # declined search still shuffles, finds nothing
    assert [c.name for c in state.hand] == ["Forest"]


def test_discard_from_hand_to_graveyard_moves_fresh_instance():
    state = _make_state()
    find_to_hand(state, "Forest")
    forest = state.hand[0]
    state.hand = [forest]
    discard_from_hand_to_graveyard(state, forest)
    # library->hand->graveyard: the graveyard holds a FRESH instance (move_card),
    # a distinct object from the hand CardDef, not the CardDef itself.
    assert state.hand == [] and [c.name for c in state.graveyard] == ["Forest"]
    assert state.graveyard[0] is not forest, "graveyard entry must be a new instance, not the discarded CardDef"


def test_discard_from_hand_to_graveyard_raises_when_not_in_hand():
    state = _make_state()
    find_to_hand(state, "Forest")
    forest = state.hand[0]
    state.hand = [forest]
    discard_from_hand_to_graveyard(state, forest)
    with pytest.raises(RuntimeError):
        discard_from_hand_to_graveyard(state, forest)
