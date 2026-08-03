"""Shared card-mechanic helpers reused by multiple color catalogs (shared.py)."""
import pytest

from game import registry
from game.cards import CardDef, CardType, EffectId
from game.catalog.colorless_cards import (
    _lotus_petal_on_tap, _treasure_on_tap, activate_candy_trail_sac, activate_expedition_map,
    activate_twisted_landscape_fetch,
)
from game.catalog.red_cards import activate_melded_moxite_sac, activate_reckless_lackey_sac
from game.effects.shared import any_creature_on_battlefield, discard_from_hand_to_graveyard, find_to_hand
from game.effects.tokens import activate_map_sac
from game.state import GameState, Permanent


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


def test_discard_from_hand_to_graveyard_logs_zone_move():
    # Regression: a genuine hand-cost discard (Islandcycling/Cycling/
    # Forestcycle -- none of which touch the stack) used to leave the
    # engine's own state.hand correctly reduced but emit no zone_move event
    # at all, so the replay/event log kept showing the card as still in
    # hand forever. Only this branch (not the resolving-spell one, already
    # logged at cast) needs the event.
    state = _make_state()
    state.event_log = []
    find_to_hand(state, "Forest")
    forest = state.hand[0]
    state.hand = [forest]
    discard_from_hand_to_graveyard(state, forest)
    moves = [e for e in state.event_log if e["kind"] == "zone_move"]
    assert moves[-1]["card"] == "Forest"
    assert moves[-1]["from_zone"] == "hand"
    assert moves[-1]["to_zone"] == "graveyard"
    assert moves[-1]["reason"] == "discard"


# Every one of these independently removes a sacrificed permanent from the
# battlefield instead of routing through state_based.sacrifice_to_graveyard
# (the one canonical "leave the battlefield by sacrifice" path every
# resolution.begin_sacrifice/discard_or_sacrifice pick also goes through) --
# each one used to skip fire_sacrifice_triggers entirely (Gixian Infiltrator
# silently never saw these sacrifices). Regression coverage: every site now
# queues an "on_sacrifice" entry, no exceptions.
_SACRIFICE_SITES = (
    ("activate_twisted_landscape_fetch", activate_twisted_landscape_fetch),
    ("activate_expedition_map", activate_expedition_map),
    ("activate_candy_trail_sac", activate_candy_trail_sac),
    ("_lotus_petal_on_tap", _lotus_petal_on_tap),
    ("_treasure_on_tap", _treasure_on_tap),
    ("activate_map_sac", activate_map_sac),
    ("activate_reckless_lackey_sac", activate_reckless_lackey_sac),
    ("activate_melded_moxite_sac", activate_melded_moxite_sac),
)


@pytest.mark.parametrize("name,fn", _SACRIFICE_SITES, ids=[n for n, _ in _SACRIFICE_SITES])
def test_every_sacrifice_site_fires_gixian_trigger(name, fn):
    state = GameState(on_the_play=True)
    gix = Permanent(registry.CARD_DEFS["Gixian Infiltrator"])
    gix.slot = 1
    sacrificed = Permanent(CardDef(f"Sacrificed ({name})", CardType.ARTIFACT, {"generic": 1}, EffectId.FILLER))
    sacrificed.slot = 1
    state.battlefield = [gix, sacrificed]
    fn(state, sacrificed)
    queued = [e for e in state.trigger_queue if e["type"] == "on_sacrifice"]
    assert queued and queued[0]["permanent"] is gix, f"{name} never called fire_sacrifice_triggers"
