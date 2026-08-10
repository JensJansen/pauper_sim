"""Tests for game.catalog.multicolor_cards. See the module under test for
the card-implementation rationale (real-rules citations, etc.) each test
below guards."""

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
    # Sacrificing purely for the +1/+1 trigger still floats {C} as a forced
    # side effect, but it must never count as avoidable burn -- untagged.
    assert state.mana_pool_single_pip == {}
    promote_triggers_to_stack(state)
    resolve_top_of_stack(state)
    assert wr.counters.get("+1/+1") == 1  # "whenever you sacrifice another Eldrazi"


def test_jagged_barrens_etb_deals_damage_to_target_opponent():
    """Jagged Barrens: real "target opponent" (restricted, NOT "any target"
    like the red burn spells) -- captured at ETB promotion (etb_targets:
    True); the only ever-legal candidate in this 2-player engine, so no
    interactive choice is needed, but the effect is still properly deferred
    onto the stack rather than applied inline at promotion."""
    state = _two()
    state.active_idx = 0
    state.players[1].life_total = 20
    barrens = CardDef("Jagged Barrens", CardType.LAND, None, EffectId.JAGGED_BARRENS)
    enters_battlefield(state, barrens, from_zone="hand")
    assert [e["type"] for e in state.trigger_queue] == ["etb"]
    promote_triggers_to_stack(state)
    assert len(state.stack) == 1  # target captured at promotion, effect now waits on the stack
    resolve_top_of_stack(state)
    assert state.players[1].life_total == 19


def test_jagged_barrens_etb_solo_no_opponent_is_noop():
    """No opponent (1-player) -> nothing to target, so the ETB does
    nothing -- matches Mesmeric Fiend's own 1-player guard."""
    solo = GameState(on_the_play=True)
    barrens = CardDef("Jagged Barrens", CardType.LAND, None, EffectId.JAGGED_BARRENS)
    enters_battlefield(solo, barrens, from_zone="hand")
    promote_triggers_to_stack(solo)
    assert solo.stack == [] and solo.pending_resolution is None


def test_sneaky_snacker_has_flying():
    """Real Sneaky Snacker (MH3) is a 2/1 flier -- only blockable by flying
    or reach, not a vanilla ground creature like Gurmag Angler."""
    state = _two()
    snacker = Permanent(registry.CARD_DEFS["Sneaky Snacker"])
    ground = Permanent(CardDef("Ground Blocker", CardType.CREATURE, None, EffectId.FILLER, power=5, toughness=5))
    assert not can_block(state, ground, snacker)


def test_wooded_ridgeline_taps_for_r_or_g_and_enters_tapped():
    """Wooded Ridgeline: real {T}: Add {R} or {G} -- a flexible dual, one
    tap producing either color (mirrors how a flexible source like Saruli
    Caretaker/Bonder's Ornament is exercised via mana.activate_mana_source's
    color_choice), plus its own real "enters tapped" text."""
    state = _two()
    ridgeline = registry.CARD_DEFS["Wooded Ridgeline"]
    for color in ("R", "G"):
        land = Permanent(ridgeline)
        state.battlefield = [land]
        state.mana_pool = {}
        state.mana_pool_single_pip = {}
        mana.activate_mana_source(state, land, color)
        assert state.mana_pool == {color: 1} and land.tapped
        assert state.mana_pool_single_pip == {color: 1}  # a 1-symbol event -- tagged single-pip
    entered = enters_battlefield(state, ridgeline, from_zone=None)
    assert entered.tapped


def test_rakdos_carnarium_taps_for_b_and_r_simultaneously():
    """Rakdos Carnarium: real {T}: Add {B}{R} -- fixed_multi, one tap
    floats BOTH symbols at once, not a choice of one -- driven through the
    REAL "Rakdos Carnarium" registry entry (only exercised via a
    "Carnarium-ish" FILLER double elsewhere, see test_mana.py)."""
    state = _two()
    carnarium = Permanent(registry.CARD_DEFS["Rakdos Carnarium"])
    state.battlefield = [carnarium]
    mana.activate_mana_source(state, carnarium)
    assert state.mana_pool == {"B": 1, "R": 1} and carnarium.tapped
    assert state.mana_pool_single_pip == {}  # a 2-symbol event -- never single-pip-tagged


def test_rakdos_carnarium_etb_bounce_is_queued_through_real_card():
    """Rakdos Carnarium's ETB land-bounce (bounce_land_etb), driven through
    the REAL "Rakdos Carnarium" registry entry (EffectId.RAKDOS_CARNARIUM)
    instead of the FILLER double used in test_land_bounce_etb_is_queued_
    not_inline (tests/game/effects/test_casting.py). Queued, not inline:
    the choose_permanent decision only opens once promoted to the stack and
    resolved."""
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


def test_jagged_barrens_taps_for_b_or_r():
    """Jagged Barrens: real {T}: Add {B} or {R} -- flexible dual (its ETB
    damage trigger is already covered above; just the mana ability here)."""
    state = _two()
    barrens = registry.CARD_DEFS["Jagged Barrens"]
    for color in ("B", "R"):
        land = Permanent(barrens)
        state.battlefield = [land]
        state.mana_pool = {}
        state.mana_pool_single_pip = {}
        mana.activate_mana_source(state, land, color)
        assert state.mana_pool == {color: 1} and land.tapped
        assert state.mana_pool_single_pip == {color: 1}  # a 1-symbol event -- tagged single-pip


def test_drossforge_bridge_taps_for_b_or_r():
    """Drossforge Bridge: real {T}: Add {B} or {R} (its indestructible is
    already tested elsewhere -- just the mana ability here)."""
    state = _two()
    bridge = registry.CARD_DEFS["Drossforge Bridge"]
    for color in ("B", "R"):
        land = Permanent(bridge)
        state.battlefield = [land]
        state.mana_pool = {}
        state.mana_pool_single_pip = {}
        mana.activate_mana_source(state, land, color)
        assert state.mana_pool == {color: 1} and land.tapped
        assert state.mana_pool_single_pip == {color: 1}  # a 1-symbol event -- tagged single-pip


def test_drossforge_bridge_counts_for_affinity_and_metalcraft():
    """Drossforge Bridge is BOTH a land and an artifact (extra["artifact"])
    -- counts toward "artifacts you control" for affinity (Myr Enforcer,
    tested via this same affinity_reduction/is_artifact predicate in
    test_colorless_cards.py) and metalcraft (Galvanic Blast, tested via
    Great Furnace in test_red_cards.py)."""
    assert is_artifact(registry.CARD_DEFS["Drossforge Bridge"])
    state = GameState(on_the_play=True)
    state.battlefield = [Permanent(registry.CARD_DEFS["Drossforge Bridge"]) for _ in range(3)]
    assert affinity_reduction(state) == 3  # 3 artifact lands -> metalcraft threshold met / {3} affinity reduction


def test_mistvault_bridge_taps_for_u_or_b():
    """Mistvault Bridge: real {T}: Add {U} or {B} -- completely untested
    entry (mana ability, indestructible, artifact) before this."""
    state = _two()
    bridge = registry.CARD_DEFS["Mistvault Bridge"]
    for color in ("U", "B"):
        land = Permanent(bridge)
        state.battlefield = [land]
        state.mana_pool = {}
        state.mana_pool_single_pip = {}
        mana.activate_mana_source(state, land, color)
        assert state.mana_pool == {color: 1} and land.tapped
        assert state.mana_pool_single_pip == {color: 1}  # a 1-symbol event -- tagged single-pip


def test_silverbluff_bridge_taps_for_u_or_r():
    """Silverbluff Bridge: real {T}: Add {U} or {R} -- completely untested
    entry (mana ability, indestructible, artifact) before this."""
    state = _two()
    bridge = registry.CARD_DEFS["Silverbluff Bridge"]
    for color in ("U", "R"):
        land = Permanent(bridge)
        state.battlefield = [land]
        state.mana_pool = {}
        state.mana_pool_single_pip = {}
        mana.activate_mana_source(state, land, color)
        assert state.mana_pool == {color: 1} and land.tapped
        assert state.mana_pool_single_pip == {color: 1}  # a 1-symbol event -- tagged single-pip


def test_slagwoods_bridge_taps_for_r_or_g():
    """Slagwoods Bridge: real {T}: Add {R} or {G} -- completely untested
    entry (mana ability, indestructible, artifact) before this."""
    state = _two()
    bridge = registry.CARD_DEFS["Slagwoods Bridge"]
    for color in ("R", "G"):
        land = Permanent(bridge)
        state.battlefield = [land]
        state.mana_pool = {}
        state.mana_pool_single_pip = {}
        mana.activate_mana_source(state, land, color)
        assert state.mana_pool == {color: 1} and land.tapped
        assert state.mana_pool_single_pip == {color: 1}  # a 1-symbol event -- tagged single-pip


def test_contaminated_aquifer_taps_for_u_or_b():
    """Contaminated Aquifer: real {T}: Add {U} or {B} -- completely
    untested entry before this."""
    state = _two()
    aquifer = registry.CARD_DEFS["Contaminated Aquifer"]
    for color in ("U", "B"):
        land = Permanent(aquifer)
        state.battlefield = [land]
        state.mana_pool = {}
        state.mana_pool_single_pip = {}
        mana.activate_mana_source(state, land, color)
        assert state.mana_pool == {color: 1} and land.tapped
        assert state.mana_pool_single_pip == {color: 1}  # a 1-symbol event -- tagged single-pip


def test_contaminated_aquifer_is_legal_islandcycling_target():
    """Contaminated Aquifer's Island subtype (subtypes=("Island", "Swamp"))
    makes it a legal Islandcycling search target -- mirrors test_lorien_
    revealed_islandcycling_searches_island_subtype's own Ice Tunnel case in
    test_blue_cards.py; same pattern, sibling U/B dual."""
    state = GameState(on_the_play=True)
    lorien = registry.CARD_DEFS["Lórien Revealed"]
    state.hand = [lorien]
    state.library = [
        registry.CARD_DEFS["Contaminated Aquifer"],
        CardDef("Mountain", CardType.LAND, None, EffectId.MOUNTAIN, basic=True, subtypes=("Mountain",)),
    ]
    islandcycle_lorien_revealed(state, lorien)
    assert state.pending_resolution["kind"] == "search_fetch"
    assert resolution.search_fetch_options(state) == ["Contaminated Aquifer"]  # Mountain excluded (no Island subtype)
    resolution.execute_search_fetch_option(state, "Contaminated Aquifer")
    assert any(c.name == "Contaminated Aquifer" for c in state.hand)


def test_ice_tunnel_taps_for_u_or_b():
    """Ice Tunnel: real {T}: Add {U} or {B} (its Islandcycling-target
    relevance via subtypes is already tested elsewhere -- just the mana
    ability here)."""
    state = _two()
    tunnel = registry.CARD_DEFS["Ice Tunnel"]
    for color in ("U", "B"):
        land = Permanent(tunnel)
        state.battlefield = [land]
        state.mana_pool = {}
        state.mana_pool_single_pip = {}
        mana.activate_mana_source(state, land, color)
        assert state.mana_pool == {color: 1} and land.tapped
        assert state.mana_pool_single_pip == {color: 1}  # a 1-symbol event -- tagged single-pip


def test_sneaky_snacker_real_draw_trigger_orders_and_returns():
    """Sneaky Snacker's real on_draw_count {count: 3} + pending_kinds:
    order_triggers, driven through the REAL "Sneaky Snacker" registry entry
    (EffectId.SNEAKY_SNACKER) instead of the "Fake Snacker" FILLER double
    used by test_draw_counter_and_automatic_return (tests/game/effects/
    test_triggers.py). Two physical copies in the graveyard both cross the
    "third card drawn this turn" trigger on the same draw -- a real
    placement-order choice -- then each independently returns to the
    battlefield tapped. Its flying keyword is already tested above -- not
    repeated here."""
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
    assert state.pending_resolution["kind"] == "order_triggers"  # one more still to place
    resolution.execute_order_triggers_option(state, "Sneaky Snacker")
    assert state.pending_resolution is None
    assert len(state.stack) == 2
    while state.stack:
        resolve_top_of_stack(state)
    returned = [p for p in state.battlefield if p.card_def.name == "Sneaky Snacker"]
    assert len(returned) == 2
    assert all(p.tapped for p in returned)  # force_tapped=True on this automatic return


def test_slippery_bogle_real_cast_pays_g_mana():
    """Slippery Bogle {G}: cast through the real registry "cast" spec,
    paying {G} via game.mana.begin_pay_cost/execute_pool_spend and
    push_to_stack -- every other test in this suite constructs the
    Permanent directly or drops it straight onto zones, bypassing the real
    cast/cost-payment path entirely."""
    state = _two()
    bogle_def = registry.CARD_DEFS["Slippery Bogle"]
    state.hand = [bogle_def]
    state.mana_pool = {"G": 1}
    resolve_fn = registry.EFFECT_REGISTRY[EffectId.SLIPPERY_BOGLE]["cast"]["resolve"]
    mana.begin_pay_cost(state, bogle_def.cast_cost, on_complete=lambda s: push_to_stack(s, bogle_def, resolve_fn))
    assert state.pending_resolution["kind"] == "pay_cost"
    mana.execute_pool_spend(state, "G")
    assert bogle_def not in state.hand  # left hand once cost is fully paid (push_to_stack)
    resolve_top_of_stack(state)
    bogle = next(p for p in state.battlefield if p.card_def.name == "Slippery Bogle")
    assert bogle.card_def.effect_id == EffectId.SLIPPERY_BOGLE


def test_slippery_bogle_hexproof_blocks_opponent_targeting():
    """Slippery Bogle: hexproof (real stats.can_be_targeted restriction),
    tied to its own EffectId.SLIPPERY_BOGLE -- every existing hexproof test
    (e.g. test_can_be_targeted_hexproof_and_shroud) uses a FILLER "Hexed"
    double, never this card's own registry entry, despite Slippery Bogle
    being used extensively as a combat fixture elsewhere in the suite."""
    state = _two()
    bogle = Permanent(registry.CARD_DEFS["Slippery Bogle"])
    state.players[0].battlefield = [bogle]
    assert can_be_targeted(state, bogle, 0)       # its own controller may target it
    assert not can_be_targeted(state, bogle, 1)   # an opponent may not


def test_armadillo_cloak_real_cast_attaches_to_chosen_creature():
    """Armadillo Cloak {1}{G}{W}: the real cast/target/pay-mana flow --
    cast_aura's own resolve, extra_legal (any_creature_on_either_
    battlefield), precast_choice (real MTG: the target is locked in as
    part of casting, before the stack), pending_kinds: choose_any_target --
    driven to completion (resolve_top_of_stack), ending attached to the
    chosen creature. Every combat test in this file instead constructs the
    Aura permanent directly and presets flags["enchanting"], bypassing
    casting entirely; the combat effects themselves (trample/lifelink/
    pt_bonus) are already thoroughly tested there and not repeated here."""
    state = _two()
    cloak = registry.CARD_DEFS["Armadillo Cloak"]
    target = Permanent(CardDef("Target Creature", CardType.CREATURE, None, EffectId.FILLER, power=2, toughness=2))
    target.slot = 1
    state.players[1].battlefield = [target]
    state.players[0].hand = [cloak]
    state.mana_pool = {"C": 1, "G": 1, "W": 1}  # floating pool is real colors only -- "C" pays the generic pip
    cast_spec = registry.EFFECT_REGISTRY[EffectId.ARMADILLO_CLOAK]["cast"]
    assert cast_spec["extra_legal"](state)  # any_creature_on_either_battlefield
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
    """Writhing Chrysalis: Reach lets it block a real flier (Kitchen Imp) --
    mirrors test_can_block_evasion_and_reach's pattern in tests/game/
    effects/test_combat.py, exercised here through this card's own
    EffectId (every other aspect of this card is already thoroughly tested
    in this file, and not repeated here)."""
    state = _two()
    chrysalis = Permanent(registry.CARD_DEFS["Writhing Chrysalis"])
    flier = Permanent(registry.CARD_DEFS["Kitchen Imp"])
    assert can_block(state, chrysalis, flier)
