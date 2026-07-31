"""Tests for game.catalog.multicolor_cards. See the module under test for
the card-implementation rationale (real-rules citations, etc.) each test
below guards."""

from game import registry, resolution
from game.cards import CardDef, CardType, EffectId
from game.catalog.multicolor_cards import cast_agony_warp, cast_terminate, cast_writhing_chrysalis
from game.effects.combat import can_block
from game.effects.shared import card_colors
from game.effects.stack import resolve_top_of_stack
from game.effects.stats import permanent_power, permanent_toughness
from game.effects.triggers import promote_triggers_to_stack
from game.state import GameState, Permanent, PlayerState


def _two():
    return GameState(on_the_play=True, players=[PlayerState(True), PlayerState(False)])


def test_terminate_destroys_any_creature():
    """{B}{R}: Destroy target creature. It can't be regenerated -- a no-op,
    since no card in this engine ever grants regeneration. Destroys a black
    creature -- any creature qualifies, including black."""
    state = _two()
    victim = Permanent(CardDef("Black Creature", CardType.CREATURE, {"B": 1}, EffectId.FILLER, power=2, toughness=2))
    victim.slot = 1
    state.players[1].battlefield = [victim]
    state.players[0].hand = [registry.CARD_DEFS["Terminate"]]
    cast_terminate(state, registry.CARD_DEFS["Terminate"])
    resolution.execute_choose_any_target_creature(state, 1, "Black Creature", 1)
    resolve_top_of_stack(state)
    assert victim not in state.players[1].battlefield


def test_agony_warp_split_targets_survives_and_dies():
    """{U}{B}: "Target creature gets -3/-0 until end of turn. Target
    creature gets -0/-3 until end of turn." Two independent targets, both
    locked at cast (precast_choice) -- they MAY be the same creature (then
    -3/-3), so a single legal creature is enough to cast. On resolution each
    half applies only if its own target is still a legal creature (608.2c --
    a spell does as much as it can).

    Two DIFFERENT targets: 5/5 -> 2/5 (survives) and 1/3 -> 1/0 (dies) -- a
    -0/-3 that drops a creature to 0 toughness kills it via the
    state-based-action check."""
    state = _two()
    big = Permanent(CardDef("Big", CardType.CREATURE, None, EffectId.FILLER, power=5, toughness=5))
    big.slot = 1
    small = Permanent(CardDef("Small", CardType.CREATURE, None, EffectId.FILLER, power=1, toughness=3))
    small.slot = 1
    state.players[1].battlefield = [big, small]
    state.players[0].hand = [registry.CARD_DEFS["Agony Warp"]]
    cast_agony_warp(state, registry.CARD_DEFS["Agony Warp"])
    resolution.execute_choose_any_target_creature(state, 1, "Big", 1)     # -3/-0
    resolution.execute_choose_any_target_creature(state, 1, "Small", 1)   # -0/-3
    resolve_top_of_stack(state)
    assert big in state.players[1].battlefield and permanent_power(state, big) == 2 and permanent_toughness(state, big) == 5
    assert small not in state.players[1].battlefield  # 3 - 3 = 0 toughness -> dead


def test_agony_warp_same_creature_stacks_to_minus_three_three():
    """Both Agony Warp targets landing on the SAME creature (only one on
    board): -3/-3. 4/4 -> 1/1, survives."""
    state = _two()
    lone = Permanent(CardDef("Lone", CardType.CREATURE, None, EffectId.FILLER, power=4, toughness=4))
    lone.slot = 1
    state.players[1].battlefield = [lone]
    state.players[0].hand = [registry.CARD_DEFS["Agony Warp"]]
    cast_agony_warp(state, registry.CARD_DEFS["Agony Warp"])
    resolution.execute_choose_any_target_creature(state, 1, "Lone", 1)
    resolution.execute_choose_any_target_creature(state, 1, "Lone", 1)
    resolve_top_of_stack(state)
    assert lone in state.players[1].battlefield  # 4/4 -> 1/1, survives
    assert permanent_power(state, lone) == 1 and permanent_toughness(state, lone) == 1


def test_writhing_chrysalis_devoid_eldrazi_spawn_and_sacrifice_counter():
    """G8: Writhing Chrysalis -- {2}{R}{G} Devoid: "When you cast this
    spell, create two 0/1 Eldrazi Spawn" -- made as it's cast (before it
    resolves), then the 2/3 enters on resolution as colorless (Devoid).
    Reach + "whenever you sacrifice another Eldrazi, +1/+1" (its
    on_sacrifice), exercised here via the Eldrazi Spawn token's own sac
    ability. A real triggered ability -- queued, then placed on the stack at
    the next priority window, not applied immediately."""
    state = GameState(on_the_play=True)
    state.hand = [registry.CARD_DEFS["Writhing Chrysalis"]]
    cast_writhing_chrysalis(state, registry.CARD_DEFS["Writhing Chrysalis"])
    assert sum(1 for p in state.battlefield if p.card_def.name == "Eldrazi Spawn") == 2  # made at cast
    resolve_top_of_stack(state)  # Writhing enters
    wr = next(p for p in state.battlefield if p.card_def.name == "Writhing Chrysalis")
    assert card_colors(wr.card_def) == set()  # Devoid -> colorless
    spawn = next(p for p in state.battlefield if p.card_def.name == "Eldrazi Spawn")
    registry.EFFECT_REGISTRY[EffectId.ELDRAZI_SPAWN_TOKEN]["activated_abilities"]["sac"]["resolve"](state, spawn)
    promote_triggers_to_stack(state)
    resolve_top_of_stack(state)
    assert wr.counters.get("+1/+1") == 1  # "whenever you sacrifice another Eldrazi"


def test_sneaky_snacker_has_flying():
    """Real Sneaky Snacker (MH3) is a 2/1 flier -- only blockable by flying
    or reach, not a vanilla ground creature like Gurmag Angler."""
    state = _two()
    snacker = Permanent(registry.CARD_DEFS["Sneaky Snacker"])
    ground = Permanent(CardDef("Ground Blocker", CardType.CREATURE, None, EffectId.FILLER, power=5, toughness=5))
    assert not can_block(state, ground, snacker)
