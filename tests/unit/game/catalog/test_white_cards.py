"""Tests for game.catalog.white_cards."""

from game import mana, registry, resolution
from game.cards import CardDef, CardType, EffectId
from game.catalog.white_cards import cast_cartouche_of_solidarity, cast_ethereal_armor
from game.effects.combat import combat_damage_step, declare_attacker, declare_attackers_step
from game.effects.stack import resolve_top_of_stack
from game.effects.stats import has_keyword, permanent_power, permanent_toughness
from game.state import GameState, Permanent, PlayerState


def test_cartouche_of_solidarity_only_enchants_a_creature_you_control():
    """An opponent's creature must not be a legal target."""
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
    """An opponent's creature is a legal target."""
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


def test_cartouche_of_solidarity_extra_legal_gate_via_its_own_registry_entry():
    """extra_legal is satisfied only by a creature on the caster's own battlefield."""
    gate = registry.EFFECT_REGISTRY[EffectId.CARTOUCHE_OF_SOLIDARITY]["cast"]["extra_legal"]
    state = GameState(on_the_play=True, players=[PlayerState(True), PlayerState(False)])
    theirs = Permanent(CardDef("Theirs", CardType.CREATURE, None, EffectId.FILLER, power=1, toughness=1))
    theirs.slot = 1
    state.players[1].battlefield = [theirs]
    assert not gate(state)  # only an opponent's creature exists

    mine = Permanent(CardDef("Mine", CardType.CREATURE, None, EffectId.FILLER, power=1, toughness=1))
    mine.slot = 1
    state.players[0].battlefield = [mine]
    assert gate(state)


def test_cartouche_of_solidarity_grants_exactly_plus_one_plus_one():
    """An enchanted creature's power/toughness go up by exactly +1/+1."""
    state = GameState(on_the_play=True, players=[PlayerState(True), PlayerState(False)])
    creature = Permanent(CardDef("Mine", CardType.CREATURE, None, EffectId.FILLER, power=2, toughness=2))
    creature.slot = 1
    state.players[0].battlefield = [creature]
    assert permanent_power(state, creature) == 2 and permanent_toughness(state, creature) == 2

    cartouche = Permanent(registry.CARD_DEFS["Cartouche of Solidarity"])
    cartouche.flags["enchanting"] = creature
    state.players[0].battlefield.append(cartouche)
    assert permanent_power(state, creature) == 3
    assert permanent_toughness(state, creature) == 3


def test_cartouche_of_solidarity_cast_pays_w_through_the_mana_pipeline():
    """{W} cast cost is charged through the real mana pipeline."""
    state = GameState(on_the_play=True, players=[PlayerState(True), PlayerState(False)])
    plains = Permanent(registry.CARD_DEFS["Plains"])
    mine = Permanent(CardDef("Mine", CardType.CREATURE, None, EffectId.FILLER, power=1, toughness=1))
    mine.slot = 1
    card_def = registry.CARD_DEFS["Cartouche of Solidarity"]
    state.players[0].battlefield = [plains, mine]
    state.hand = [card_def]

    mana.activate_mana_source(state, plains)
    assert state.mana_pool == {"W": 1}
    assert state.mana_pool_single_pip == {"W": 1}
    mana.begin_pay_cost(state, card_def.cast_cost, on_complete=lambda s: cast_cartouche_of_solidarity(s, card_def))
    mana.execute_pool_spend(state, mana.pool_spend_options(state)[0])
    assert state.mana_pool == {}

    assert (0, "Mine", 1) in resolution.choose_any_target_creature_options(state)
    resolution.execute_choose_any_target_creature(state, 0, "Mine", 1)
    resolve_top_of_stack(state)
    assert card_def not in state.hand
    assert any(p.card_def.name == "Cartouche of Solidarity" for p in state.players[0].battlefield)


def test_cartouche_of_solidarity_etb_creates_a_vigilant_warrior_token():
    """ETB creates a 1/1 Warrior token with vigilance on the caster's battlefield."""
    state = GameState(on_the_play=True, players=[PlayerState(True), PlayerState(False)])
    mine = Permanent(CardDef("Mine", CardType.CREATURE, None, EffectId.FILLER, power=1, toughness=1))
    mine.slot = 1
    card_def = registry.CARD_DEFS["Cartouche of Solidarity"]
    state.players[0].battlefield = [mine]
    state.hand = [card_def]

    cast_cartouche_of_solidarity(state, card_def)
    resolution.execute_choose_any_target_creature(state, 0, "Mine", 1)
    resolve_top_of_stack(state)

    warrior = next((p for p in state.players[0].battlefield if p.card_def.name == "Warrior"), None)
    assert warrior is not None
    assert permanent_power(state, warrior) == 1 and permanent_toughness(state, warrior) == 1
    assert has_keyword(state, warrior, "vigilance")


def test_ethereal_armor_pt_bonus_counts_itself_and_other_enchantments():
    """pt_bonus/toughness_bonus counts Ethereal Armor itself, plus every
    other enchantment you control."""
    state = GameState(on_the_play=True, players=[PlayerState(True), PlayerState(False)])
    creature = Permanent(CardDef("Mine", CardType.CREATURE, None, EffectId.FILLER, power=1, toughness=1))
    creature.slot = 1
    armor = Permanent(registry.CARD_DEFS["Ethereal Armor"])
    armor.flags["enchanting"] = creature
    state.players[0].battlefield = [creature, armor]
    assert permanent_power(state, creature) == 2  # 1 base + 1 (itself)
    assert permanent_toughness(state, creature) == 2

    other_enchantment = Permanent(CardDef("Some Other Enchantment", CardType.ENCHANTMENT, None, EffectId.FILLER))
    state.players[0].battlefield.append(other_enchantment)
    assert permanent_power(state, creature) == 3  # 1 base + 2 (itself + the other)
    assert permanent_toughness(state, creature) == 3


def test_ethereal_armor_first_strike_kills_before_blocker_deals_damage():
    """First strike lets the attacker kill the blocker before taking damage
    back; requires both the +1/+1 bonus and first strike to reach a clean win."""
    _card_defs_backup = dict(registry.CARD_DEFS)
    try:
        state = GameState(on_the_play=True, players=[PlayerState(True), PlayerState(False)])
        fs_attacker = Permanent(CardDef("First Striker", CardType.CREATURE, None, EffectId.FILLER, power=2, toughness=2))
        fs_attacker.summoning_sick = False
        registry.CARD_DEFS["First Striker"] = fs_attacker.card_def
        armor_on_attacker = Permanent(CardDef("Ethereal Armor", CardType.ENCHANTMENT, {"W": 1}, EffectId.ETHEREAL_ARMOR))
        armor_on_attacker.flags["enchanting"] = fs_attacker
        lethal_blocker = Permanent(CardDef("Would-Be Killer", CardType.CREATURE, None, EffectId.FILLER, power=3, toughness=3))
        registry.CARD_DEFS["Would-Be Killer"] = lethal_blocker.card_def
        state.players[0].battlefield = [fs_attacker, armor_on_attacker]
        state.players[1].battlefield = [lethal_blocker]

        declare_attackers_step(state)
        declare_attacker(state, fs_attacker)
        state.blocked_by[fs_attacker] = [lethal_blocker]
        combat_damage_step(state)

        # Effective power 3 >= blocker's toughness 3: dies in the first-strike sub-step.
        assert lethal_blocker not in state.players[1].battlefield
        assert fs_attacker in state.players[0].battlefield and fs_attacker.damage_marked == 0
    finally:
        registry.CARD_DEFS.clear()
        registry.CARD_DEFS.update(_card_defs_backup)
