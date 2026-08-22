"""Tests for game.catalog.multicolor_cards."""

import pytest

from game import mana, registry, resolution
from game.cards import CardDef, CardType, EffectId, card_colors, is_artifact
from game.catalog.blue_cards import islandcycle_lorien_revealed
from game.catalog.multicolor_cards import cast_agony_warp, cast_terminate, cast_writhing_chrysalis
from game.effects.casting import enters_battlefield
from game.effects.combat import can_block
from game.effects.shared import affinity_reduction
from game.effects.stack import push_to_stack, resolve_top_of_stack
from game.effects.stats import can_be_targeted, permanent_power, permanent_toughness
from game.effects.triggers import promote_triggers_to_stack
from game.state import GameState, Permanent, PlayerState


def _two():
    return GameState(on_the_play=True, players=[PlayerState(True), PlayerState(False)])


def test_terminate_destroys_any_creature():
    """Any creature qualifies, including black."""
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
    """Two different targets: 5/5 -> 2/5 (survives) and 1/3 -> 1/0 (dies)."""
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
    """Both targets on the same creature: -3/-3. 4/4 -> 1/1, survives."""
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
    """Cast creates two Eldrazi Spawn immediately; the 2/3 enters as
    colorless (Devoid) on resolution. Sacrificing another Eldrazi triggers
    +1/+1, queued and placed on the stack, not applied immediately."""
    state = GameState(on_the_play=True)
    state.hand = [registry.CARD_DEFS["Writhing Chrysalis"]]
    cast_writhing_chrysalis(state, registry.CARD_DEFS["Writhing Chrysalis"])
    assert sum(1 for p in state.battlefield if p.card_def.name == "Eldrazi Spawn") == 2  # made at cast
    resolve_top_of_stack(state)  # Writhing enters
    wr = next(p for p in state.battlefield if p.card_def.name == "Writhing Chrysalis")
    assert card_colors(wr.card_def) == set()  # Devoid -> colorless
    spawn = next(p for p in state.battlefield if p.card_def.name == "Eldrazi Spawn")
    registry.EFFECT_REGISTRY[EffectId.ELDRAZI_SPAWN_TOKEN]["activated_abilities"]["sac"]["resolve"](state, spawn)
    assert state.mana_pool_single_pip == {}  # the forced {C} float must not count as avoidable burn
    promote_triggers_to_stack(state)
    resolve_top_of_stack(state)
    assert wr.counters.get("+1/+1") == 1


def test_jagged_barrens_etb_deals_damage_to_target_opponent():
    """Target captured at ETB promotion; effect waits on the stack."""
    state = _two()
    state.active_idx = 0
    state.players[1].life_total = 20
    barrens = CardDef("Jagged Barrens", CardType.LAND, None, EffectId.JAGGED_BARRENS)
    enters_battlefield(state, barrens, from_zone="hand")
    assert [e["type"] for e in state.trigger_queue] == ["etb"]
    promote_triggers_to_stack(state)
    assert len(state.stack) == 1
    resolve_top_of_stack(state)
    assert state.players[1].life_total == 19


def test_jagged_barrens_etb_solo_no_opponent_is_noop():
    """No opponent -> the ETB does nothing."""
    solo = GameState(on_the_play=True)
    barrens = CardDef("Jagged Barrens", CardType.LAND, None, EffectId.JAGGED_BARRENS)
    enters_battlefield(solo, barrens, from_zone="hand")
    promote_triggers_to_stack(solo)
    assert solo.stack == [] and solo.pending_resolution is None


def test_sneaky_snacker_has_flying():
    """Only blockable by flying or reach, not a vanilla ground creature."""
    state = _two()
    snacker = Permanent(registry.CARD_DEFS["Sneaky Snacker"])
    ground = Permanent(CardDef("Ground Blocker", CardType.CREATURE, None, EffectId.FILLER, power=5, toughness=5))
    assert not can_block(state, ground, snacker)


@pytest.mark.parametrize("card_name,colors,enters_tapped", [
    ("Wooded Ridgeline", ("R", "G"), True),
    ("Jagged Barrens", ("B", "R"), False),
    ("Drossforge Bridge", ("B", "R"), False),
    ("Mistvault Bridge", ("U", "B"), False),
    ("Silverbluff Bridge", ("U", "R"), False),
    ("Slagwoods Bridge", ("R", "G"), False),
    ("Contaminated Aquifer", ("U", "B"), False),
    ("Ice Tunnel", ("U", "B"), False),
])
def test_flexible_dual_land_taps_for_two_colors(card_name, colors, enters_tapped):
    """A flexible dual's single tap can produce either color."""
    state = _two()
    card = registry.CARD_DEFS[card_name]
    for color in colors:
        land = Permanent(card)
        state.battlefield = [land]
        state.mana_pool = {}
        state.mana_pool_single_pip = {}
        mana.activate_mana_source(state, land, color)
        assert state.mana_pool == {color: 1} and land.tapped
        assert state.mana_pool_single_pip == {color: 1}
    if enters_tapped:
        entered = enters_battlefield(state, card, from_zone=None)
        assert entered.tapped


def test_rakdos_carnarium_taps_for_b_and_r_simultaneously():
    """One tap floats both {B} and {R} at once, not a choice of one."""
    state = _two()
    carnarium = Permanent(registry.CARD_DEFS["Rakdos Carnarium"])
    state.battlefield = [carnarium]
    mana.activate_mana_source(state, carnarium)
    assert state.mana_pool == {"B": 1, "R": 1} and carnarium.tapped
    assert state.mana_pool_single_pip == {}  # a 2-symbol event -- never single-pip-tagged


def test_rakdos_carnarium_etb_bounce_is_queued_through_real_card():
    """ETB land-bounce is queued, not inline: the choose_permanent decision
    only opens once promoted to the stack and resolved."""
    state = GameState(on_the_play=True)
    carnarium_def = registry.CARD_DEFS["Rakdos Carnarium"]
    state.hand = [carnarium_def]
    state.battlefield = [
        Permanent(CardDef("Forest", CardType.LAND, None, EffectId.FOREST)),
        Permanent(CardDef("Swamp", CardType.LAND, None, EffectId.SWAMP)),
    ]
    state.hand.remove(carnarium_def)
    enters_battlefield(state, carnarium_def)  # normal ETB path
    assert state.pending_resolution is None
    assert [e["type"] for e in state.trigger_queue] == ["etb"]
    promote_triggers_to_stack(state)
    resolve_top_of_stack(state)
    assert state.pending_resolution["kind"] == "choose_permanent"
    assert resolution.choose_permanent_options(state) == [
        ("Forest", 1), ("Rakdos Carnarium", 1), ("Swamp", 1),
    ]
    resolution.execute_choose_permanent_option(state, "Swamp", 1)
    assert state.pending_resolution is None
    assert sorted(p.card_def.name for p in state.battlefield) == ["Forest", "Rakdos Carnarium"]
    assert [c.name for c in state.hand] == ["Swamp"]


def test_drossforge_bridge_counts_for_affinity_and_metalcraft():
    """A land that's also an artifact counts toward "artifacts you control"."""
    assert is_artifact(registry.CARD_DEFS["Drossforge Bridge"])
    state = GameState(on_the_play=True)
    state.battlefield = [Permanent(registry.CARD_DEFS["Drossforge Bridge"]) for _ in range(3)]
    assert affinity_reduction(state) == 3


def test_contaminated_aquifer_is_legal_islandcycling_target():
    """Its Island subtype makes it a legal Islandcycling search target."""
    state = GameState(on_the_play=True)
    lorien = registry.CARD_DEFS["Lórien Revealed"]
    state.hand = [lorien]
    state.library = [
        registry.CARD_DEFS["Contaminated Aquifer"],
        CardDef("Mountain", CardType.LAND, None, EffectId.MOUNTAIN, basic=True, subtypes=("Mountain",)),
    ]
    islandcycle_lorien_revealed(state, lorien)
    assert state.pending_resolution["kind"] == "search_fetch"
    assert resolution.search_fetch_options(state) == ["Contaminated Aquifer"]  # Mountain excluded
    resolution.execute_search_fetch_option(state, "Contaminated Aquifer")
    assert any(c.name == "Contaminated Aquifer" for c in state.hand)


def test_sneaky_snacker_real_draw_trigger_orders_and_returns():
    """Two graveyard copies both cross the "third card drawn this turn"
    trigger on the same draw, need a placement-order choice, then each
    independently returns to the battlefield tapped."""
    state = GameState(on_the_play=True)
    snacker = registry.CARD_DEFS["Sneaky Snacker"]
    state.library = [CardDef(f"Filler {i}", CardType.SORCERY, {}, None) for i in range(5)]
    state.graveyard = [snacker, snacker]  # two physical copies
    state.draw(1)
    state.draw(1)
    state.draw(1)  # 3rd draw this turn -- both copies trigger
    assert len(state.trigger_queue) == 2
    promote_triggers_to_stack(state)
    assert state.pending_resolution["kind"] == "order_triggers"
    assert resolution.order_triggers_options(state) == ["Sneaky Snacker"]
    resolution.execute_order_triggers_option(state, "Sneaky Snacker")
    assert state.pending_resolution["kind"] == "order_triggers"  # one more to place
    resolution.execute_order_triggers_option(state, "Sneaky Snacker")
    assert state.pending_resolution is None
    assert len(state.stack) == 2
    while state.stack:
        resolve_top_of_stack(state)
    returned = [p for p in state.battlefield if p.card_def.name == "Sneaky Snacker"]
    assert len(returned) == 2
    assert all(p.tapped for p in returned)


def test_slippery_bogle_real_cast_pays_g_mana():
    """Casts through the real registry "cast" spec, paying {G} via the mana pipeline."""
    state = _two()
    bogle_def = registry.CARD_DEFS["Slippery Bogle"]
    state.hand = [bogle_def]
    state.mana_pool = {"G": 1}
    resolve_fn = registry.EFFECT_REGISTRY[EffectId.SLIPPERY_BOGLE]["cast"]["resolve"]
    mana.begin_pay_cost(state, bogle_def.cast_cost, on_complete=lambda s: push_to_stack(s, bogle_def, resolve_fn))
    assert state.pending_resolution["kind"] == "pay_cost"
    mana.execute_pool_spend(state, "G")
    assert bogle_def not in state.hand  # left hand once cost is fully paid
    resolve_top_of_stack(state)
    bogle = next(p for p in state.battlefield if p.card_def.name == "Slippery Bogle")
    assert bogle.card_def.effect_id == EffectId.SLIPPERY_BOGLE


def test_slippery_bogle_hexproof_blocks_opponent_targeting():
    """Hexproof: its own controller may target it, an opponent may not."""
    state = _two()
    bogle = Permanent(registry.CARD_DEFS["Slippery Bogle"])
    state.players[0].battlefield = [bogle]
    assert can_be_targeted(state, bogle, 0)       # its own controller may target it
    assert not can_be_targeted(state, bogle, 1)   # an opponent may not


def test_armadillo_cloak_real_cast_attaches_to_chosen_creature():
    """The real cast/target/pay-mana flow ends attached to the chosen creature."""
    state = _two()
    cloak = registry.CARD_DEFS["Armadillo Cloak"]
    target = Permanent(CardDef("Target Creature", CardType.CREATURE, None, EffectId.FILLER, power=2, toughness=2))
    target.slot = 1
    state.players[1].battlefield = [target]
    state.players[0].hand = [cloak]
    state.mana_pool = {"C": 1, "G": 1, "W": 1}  # "C" pays the generic pip
    cast_spec = registry.EFFECT_REGISTRY[EffectId.ARMADILLO_CLOAK]["cast"]
    assert cast_spec["extra_legal"](state)
    mana.begin_pay_cost(state, cloak.cast_cost, on_complete=lambda s: cast_spec["resolve"](s, cloak))
    guard = 0
    while state.pending_resolution is not None and state.pending_resolution["kind"] == "pay_cost":
        guard += 1
        assert guard < 12
        mana.execute_pool_spend(state, mana.pool_spend_options(state)[0])
    assert state.pending_resolution["kind"] == "choose_any_target"
    resolution.execute_choose_any_target_creature(state, 1, "Target Creature", 1)
    resolve_top_of_stack(state)
    aura = next(p for p in state.battlefield if p.card_def.name == "Armadillo Cloak")
    assert aura.flags["enchanting"] is target


def test_writhing_chrysalis_reach_blocks_flier():
    """Reach lets it block a flier."""
    state = _two()
    chrysalis = Permanent(registry.CARD_DEFS["Writhing Chrysalis"])
    flier = Permanent(registry.CARD_DEFS["Kitchen Imp"])
    assert can_block(state, chrysalis, flier)
