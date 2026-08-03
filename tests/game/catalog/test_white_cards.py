"""Tests for game.catalog.white_cards. See the module under test for the
card-implementation rationale (real-rules citations, etc.) each test below
guards."""

from game import registry, resolution
from game.cards import CardDef, CardType, EffectId
from game.catalog.white_cards import cast_cartouche_of_solidarity, cast_ethereal_armor
from game.effects.stack import resolve_top_of_stack
from game.state import GameState, Permanent, PlayerState


def test_cartouche_of_solidarity_only_enchants_a_creature_you_control():
    """Real text is "Enchant creature you control" -- unlike Rancor/
    Ancestral Mask/Ethereal Armor/Armadillo Cloak's plain "Enchant
    creature" (either side). An opponent's creature must NOT be a legal
    target."""
    state = GameState(on_the_play=True, players=[PlayerState(True), PlayerState(False)])
    mine = Permanent(CardDef("Mine", CardType.CREATURE, None, EffectId.FILLER, power=1, toughness=1))
    mine.slot = 1
    theirs = Permanent(CardDef("Theirs", CardType.CREATURE, None, EffectId.FILLER, power=1, toughness=1))
    theirs.slot = 1
    state.players[0].battlefield = [mine]
    state.players[1].battlefield = [theirs]
    state.hand = [registry.CARD_DEFS["Cartouche of Solidarity"]]
    cast_cartouche_of_solidarity(state, registry.CARD_DEFS["Cartouche of Solidarity"])
    options = resolution.choose_any_target_creature_options(state)
    assert (0, "Mine", 1) in options
    assert (1, "Theirs", 1) not in options  # not a legal target -- not "you control"


def test_ethereal_armor_can_enchant_either_sides_creature():
    """Real text is plain "Enchant creature" -- no "you control"
    restriction, so an opponent's creature IS a legal (if unusual) target,
    unlike Cartouche of Solidarity above."""
    state = GameState(on_the_play=True, players=[PlayerState(True), PlayerState(False)])
    theirs = Permanent(CardDef("Theirs", CardType.CREATURE, None, EffectId.FILLER, power=1, toughness=1))
    theirs.slot = 1
    state.players[1].battlefield = [theirs]
    state.hand = [registry.CARD_DEFS["Ethereal Armor"]]
    assert registry.EFFECT_REGISTRY[EffectId.ETHEREAL_ARMOR]["cast"]["extra_legal"](state)
    cast_ethereal_armor(state, registry.CARD_DEFS["Ethereal Armor"])
    assert (1, "Theirs", 1) in resolution.choose_any_target_creature_options(state)
    resolution.execute_choose_any_target_creature(state, 1, "Theirs", 1)
    resolve_top_of_stack(state)
    assert any(p.card_def.name == "Ethereal Armor" for p in state.players[0].battlefield)
