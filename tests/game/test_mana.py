"""Tests for game.mana's cast-then-pay mana system: activate_mana_source,
begin_pay_cost/execute_pool_spend, the can_pay/available_mana_units
affordability solver behind plan_payment, the Boggles-style mana-fixing Auras
(Utopia Sprawl automatic bonus, Abundant Growth competing granted ability), and
discount_departing_source (the sacrifice/untap mana-burn-tag discount)."""

import random

import pytest

from game import mana, registry
from game.cards import CardDef, CardType, EffectId
from game.state import GameState, Permanent


def _u(*colors):
    return frozenset(colors)


@pytest.mark.parametrize("units,cost,expected,why", [
    # -- the total-count condition, on its own --
    ([_u("G")], {"G": 1}, True, "trivial"),
    ([_u("G")], {"G": 1, "generic": 1}, False, "one unit cannot pay two pips"),
    ([], {}, True, "a free cost is payable with nothing"),
    ([_u("C"), _u("C")], {"generic": 2}, True, "colorless pays generic"),
    ([_u("C")], {"R": 1}, False, "colorless cannot pay a colored pip"),
    ([_u("G"), _u("R"), _u("W")], {"G": 1, "generic": 2}, True, "leftovers cover generic"),

    # -- Hall's condition alone: enough units in total, but not enough that
    #    can make the specific colors demanded --
    ([_u("G", "R"), _u("R")], {"G": 2}, False, "only one unit can make G"),
    ([_u("G", "R"), _u("G", "R")], {"G": 2}, True, "both duals can make G"),
    ([_u("G", "R"), _u("G", "R")], {"G": 1, "R": 1}, True, "the duals split"),
    ([_u("G"), _u("G")], {"G": 1, "R": 1}, False, "nothing can make R"),

    # -- the JOINT case: each color passes alone and the total passes, but
    #    the PAIR is short -- the case a greedy solver gets wrong.
    ([_u("B", "U"), _u("B", "U"), _u("B", "U"), _u("W")], {"U": 2, "B": 2}, False,
     "{U,B} jointly needs 4 units, only 3 can make either"),
    ([_u("B", "U"), _u("B", "U"), _u("B", "U"), _u("U")], {"U": 2, "B": 2}, True,
     "the fourth unit makes U, so the pair is covered"),
    ([_u("B", "U"), _u("B", "U"), _u("U"), _u("U")], {"U": 2, "B": 2}, True,
     "duals take the B half, singles take the U half"),
    ([_u("B", "U"), _u("U"), _u("U"), _u("U")], {"U": 2, "B": 2}, False,
     "only one unit can make B"),
    ([_u("W", "U"), _u("U", "B"), _u("B", "W")], {"W": 1, "U": 1, "B": 1}, True,
     "a three-colour cycle splits perfectly"),
    ([_u("W", "U"), _u("W", "U"), _u("W", "U")], {"W": 1, "U": 1, "B": 1}, False,
     "nothing can make B"),
])
def test_can_pay_is_exact_in_both_directions(units, cost, expected, why):
    """can_pay must be EXACT, not merely safe. A false positive lets a
    payment begin that cannot finish (all-False mask, hard error); a false
    negative hides a legal cast."""
    assert mana.can_pay(units, cost) is expected, why


def test_available_mana_units_counts_pool_and_sources_in_one_currency():
    """A floating pip and a symbol from an untapped source are
    interchangeable for affordability."""
    state = GameState(on_the_play=True)
    state.battlefield = [
        Permanent(CardDef("Mountain", CardType.LAND, None, EffectId.MOUNTAIN)),
        Permanent(CardDef("Swamp", CardType.LAND, None, EffectId.SWAMP)),
    ]
    assert sorted("".join(sorted(u)) for u in mana.available_mana_units(state)) == ["B", "R"]
    assert mana.plan_payment(state, {"R": 1, "B": 1}) is not None, "castable off untapped lands alone"
    assert mana.plan_payment(state, {"R": 2}) is None, "only one red source exists"

    # Tapping moves a unit from the source half to the pool half; total unchanged.
    mana.activate_mana_source(state, state.battlefield[0])
    assert sorted("".join(sorted(u)) for u in mana.available_mana_units(state)) == ["B", "R"]
    assert mana.plan_payment(state, {"R": 1, "B": 1}) is not None

    # A tapped source with nothing floating contributes nothing.
    state.mana_pool.clear()
    assert sorted("".join(sorted(u)) for u in mana.available_mana_units(state)) == ["B"]
    assert mana.plan_payment(state, {"R": 1}) is None


def test_fixed_multi_source_floats_both_symbols_at_once():
    # fixed_multi: one tap of a Rakdos-Carnarium-like source covers both
    # an outstanding B need and an outstanding R need at once, since it
    # floats both symbols from a single activation.
    filler_backup = registry.EFFECT_REGISTRY[EffectId.FILLER]
    registry.EFFECT_REGISTRY[EffectId.FILLER] = {"mana": ("fixed_multi", ("B", "R"))}
    try:
        state = GameState(on_the_play=True)
        state.battlefield = [Permanent(CardDef("Carnarium-ish", CardType.LAND, None, EffectId.FILLER))]
        assert mana.mana_output(state.battlefield[0], state) == ["B", "R"]
        mana.activate_mana_source(state, state.battlefield[0])  # one activation floats BOTH symbols
        assert state.mana_pool == {"B": 1, "R": 1} and state.battlefield[0].tapped
        assert mana.pool_can_pay(state.mana_pool, {"B": 1, "R": 1})
        assert state.mana_pool_single_pip == {}  # a 2-symbol event -- never single-pip-tagged
    finally:
        registry.EFFECT_REGISTRY[EffectId.FILLER] = filler_backup


def test_count_source_produces_one_symbol_per_matching_permanent():
    # Overgrown Battlement ("count" kind): 3 Defenders on the battlefield
    # means ONE tap alone produces 3 G, not just 1.
    state = GameState(on_the_play=True)
    state.battlefield = [
        Permanent(CardDef("Overgrown Battlement", CardType.CREATURE, {"G": 1}, EffectId.OVERGROWN_BATTLEMENT, defender=True)),
        Permanent(CardDef("Wall of Roots", CardType.CREATURE, {"generic": 1, "G": 1}, EffectId.WALL_OF_ROOTS, defender=True)),
        Permanent(CardDef("Wall of Roots", CardType.CREATURE, {"generic": 1, "G": 1}, EffectId.WALL_OF_ROOTS, defender=True)),
    ]
    for p in state.battlefield:
        p.summoning_sick = False  # Overgrown Battlement's {T} mana ability needs no summoning sickness (302.6)
    battlement = state.battlefield[0]
    assert mana.mana_output(battlement, state) == ["G", "G", "G"]  # 3 Defenders, itself included
    assert ("Overgrown Battlement", None) in mana.mana_ability_options(state)
    mana.activate_mana_source(state, battlement)  # one activation floats all 3 G into the pool
    assert state.mana_pool == {"G": 3} and battlement.tapped
    assert mana.pool_can_pay(state.mana_pool, {"G": 3}) and not mana.pool_can_pay(state.mana_pool, {"G": 4})
    assert state.mana_pool_single_pip == {}  # a 3-symbol event -- never single-pip-tagged


def test_available_mana_units_exclude_recomputes_a_count_sources_own_value():
    # Overgrown Battlement + Wall of Roots -- 2 Defenders, so Battlement's
    # own untapped contribution is 2G. Wall of Roots on its last -0/-1
    # counter (one more activation kills it): exclude={wall} must not just
    # drop Wall's own unit, it must ALSO recompute Battlement's still-
    # untapped contribution down to 1G, as if Wall were already gone --
    # the production strand (2026-08-25) this exists to prevent.
    state = GameState(on_the_play=True)
    battlement = Permanent(CardDef("Overgrown Battlement", CardType.CREATURE, {"G": 1}, EffectId.OVERGROWN_BATTLEMENT, defender=True))
    wall = Permanent(CardDef("Wall of Roots", CardType.CREATURE, {"generic": 1, "G": 1}, EffectId.WALL_OF_ROOTS, defender=True))
    state.battlefield = [battlement, wall]
    for p in state.battlefield:
        p.summoning_sick = False
    assert mana.mana_output(battlement, state) == ["G", "G"]  # 2 Defenders, both alive

    units_with_wall = mana.available_mana_units(state)
    assert sorted("".join(sorted(u)) for u in units_with_wall) == ["G", "G", "G"]  # 2 from Battlement + Wall's own 1

    units_wall_excluded = mana.available_mana_units(state, exclude={wall})
    assert sorted("".join(sorted(u)) for u in units_wall_excluded) == ["G"], (
        "Battlement's own contribution must drop to 1G when Wall is excluded, not stay at the stale 2G")


def test_units_after_own_ability_tap_accounts_for_a_lethal_self_removal():
    # Same board, mid-payment: hypothetically tapping Wall of Roots for its
    # OWN mana ability (own_ability_tap=wall) when doing so would kill it
    # must recompute Battlement's still-untapped value down to 1G in the
    # result, not the stale pre-death 2G a naive patch would produce.
    # Uses the REAL registered CardDefs (not ad-hoc ones, which default to
    # toughness=0 -- ad-hoc Wall of Roots would look "already dead" at 0
    # counters, defeating the point of this test), since this specifically
    # exercises the real mana_tap_removes_self registry lambda and real
    # base toughness (5).
    state = GameState(on_the_play=True)
    battlement = Permanent(registry.CARD_DEFS["Overgrown Battlement"])
    wall = Permanent(registry.CARD_DEFS["Wall of Roots"])
    wall.counters["-0/-1"] = 4  # toughness 1 -- one more activation is lethal
    state.battlefield = [battlement, wall]
    for p in state.battlefield:
        p.summoning_sick = False

    lethal_result = mana.units_after(state, tapped=[wall], produced=[frozenset({"G"})], own_ability_tap=wall)
    assert sorted("".join(sorted(u)) for u in lethal_result) == ["G", "G"], (
        "expected Battlement's devalued 1G + Wall's own one-time produced G, not the stale pre-death 2G")

    # A non-lethal tap (fresh Wall of Roots) must be completely unaffected --
    # the existing patch-based behavior, unchanged: Battlement's own 2G
    # (Wall still alive and counted) plus Wall's own produced G = 3 total.
    wall.counters["-0/-1"] = 0
    non_lethal_result = mana.units_after(state, tapped=[wall], produced=[frozenset({"G"})], own_ability_tap=wall)
    assert sorted("".join(sorted(u)) for u in non_lethal_result) == ["G", "G", "G"], (
        "Battlement's full 2G must be preserved when Wall's own tap does not kill it")


def test_single_pip_tag_on_fixed_source():
    # A plain land's tap is always a 1-symbol event -> tagged single-pip.
    state = GameState(on_the_play=True)
    state.battlefield = [Permanent(CardDef("Mountain", CardType.LAND, None, EffectId.MOUNTAIN))]
    mana.activate_mana_source(state, state.battlefield[0])
    assert state.mana_pool == {"R": 1}
    assert state.mana_pool_single_pip == {"R": 1}


def test_spend_one_pip_untagged_first():
    # spend_one_pip's spend-order convention: an UNTAGGED pip of a color is
    # always consumed before a TAGGED one.
    state = GameState(on_the_play=True)
    mana.float_mana(state, ["R", "G"])  # 2-symbol event -- untagged
    mana.float_mana(state, ["R"])       # 1-symbol event -- tagged
    assert state.mana_pool == {"R": 2, "G": 1}
    assert state.mana_pool_single_pip == {"R": 1}

    mana.spend_one_pip(state, "R")  # untagged R spent first
    assert state.mana_pool == {"R": 1, "G": 1}
    assert state.mana_pool_single_pip == {"R": 1}  # untouched -- the tagged pip is still floating

    mana.spend_one_pip(state, "R")  # only the tagged R remains
    assert "R" not in state.mana_pool
    assert "R" not in state.mana_pool_single_pip


def test_float_mana_taggable_false_forces_no_tag():
    # taggable=False overrides the len==1 rule -- used only by drl_env's
    # mana-filter output, which must never be tagged.
    state = GameState(on_the_play=True)
    mana.float_mana(state, ["U"], taggable=False)
    assert state.mana_pool == {"U": 1}
    assert state.mana_pool_single_pip == {}


def test_pool_can_pay_edges():
    # pool affordability edges: colorless pays generic but never a colored pip.
    assert mana.pool_can_pay({"C": 2}, {"generic": 2}) and not mana.pool_can_pay({"C": 1}, {"R": 1})
    assert mana.pool_can_pay({}, {})  # free cost


def test_begin_pay_cost_empty_cost_completes_immediately():
    # An empty cost (e.g. Lotus Petal's {} cast cost) completes immediately.
    state = GameState(on_the_play=True)
    resolved = []
    mana.begin_pay_cost(state, {}, on_complete=lambda s: resolved.append(True))
    assert state.pending_resolution is None
    assert resolved == [True]


def test_utopia_sprawl_automatic_bonus_mana():
    # Utopia Sprawl's bonus is automatic (always on top of the land's own
    # output), unlike Abundant Growth's competing ability. Exercised
    # against a real Forest with a synthetic Aura permanent.
    state = GameState(on_the_play=True)
    forest = Permanent(CardDef("Forest", CardType.LAND, None, EffectId.FOREST))
    utopia_sprawl = Permanent(CardDef("Utopia Sprawl", CardType.ENCHANTMENT, {"G": 1}, EffectId.UTOPIA_SPRAWL))
    utopia_sprawl.flags["enchanting"] = forest
    utopia_sprawl.flags["bonus_mana_color"] = "W"
    state.battlefield = [forest, utopia_sprawl]

    assert mana.mana_output(forest, state) == ["G", "W"]  # native G, plus Utopia Sprawl's automatic bonus
    mana.activate_mana_source(state, forest)  # one activation floats native G AND the automatic bonus W
    assert state.mana_pool == {"G": 1, "W": 1} and forest.tapped
    assert state.mana_pool_single_pip == {}  # a 2-symbol event -- never single-pip-tagged
    mana.begin_pay_cost(state, {"W": 1}, on_complete=lambda s: None)
    assert state.pending_resolution is not None
    mana.execute_pool_spend(state, "W")
    assert state.pending_resolution is None  # {W} paid from the pool
    assert state.mana_pool.get("G", 0) == 1  # the native G stays floating, unneeded here
    assert state.cost_paid_this_phase is True  # dense mana-burn-penalty exemption signal


def test_abundant_growth_competing_granted_ability():
    # Abundant Growth: Plains gets a competing "any of {G, W}" ability --
    # both its native W and the grant stay usable.
    state = GameState(on_the_play=True)
    plains = Permanent(CardDef("Plains", CardType.LAND, None, EffectId.PLAINS))
    abundant_growth = Permanent(CardDef("Abundant Growth", CardType.ENCHANTMENT, {"G": 1}, EffectId.ABUNDANT_GROWTH))
    abundant_growth.flags["enchanting"] = plains
    abundant_growth.flags["bonus_mana_colors"] = {"G", "W"}
    state.battlefield = [plains, abundant_growth]

    assert mana.mana_output(plains, state) == ["W"]  # native, no color_choice
    assert mana.mana_output(plains, state, "G") == ["G"]  # via the grant
    assert ("Plains", "G") in mana.mana_ability_options(state)  # the grant color is offered as a tap option
    mana.activate_mana_source(state, plains, "G")  # float G via the grant, chosen at tap time
    assert state.mana_pool == {"G": 1} and plains.tapped
    assert state.mana_pool_single_pip == {"G": 1}  # a 1-symbol event -- tagged single-pip
    mana.begin_pay_cost(state, {"G": 1}, on_complete=lambda s: None)
    mana.execute_pool_spend(state, "G")
    assert state.pending_resolution is None  # {G} covered via the granted color


def test_abundant_growth_granted_color_only_via_enchanted_permanent():
    # mana_ability_options must offer the granted color only via the
    # ENCHANTED Plains, even with an identical plain Plains also in play --
    # same-named sources are normally interchangeable; a granted-mana Aura
    # is the one case that breaks that.
    state = GameState(on_the_play=True)
    plain_plains = Permanent(CardDef("Plains", CardType.LAND, None, EffectId.PLAINS))
    grant_plains = Permanent(CardDef("Plains", CardType.LAND, None, EffectId.PLAINS))
    abundant_growth2 = Permanent(CardDef("Abundant Growth", CardType.ENCHANTMENT, {"G": 1}, EffectId.ABUNDANT_GROWTH))
    abundant_growth2.flags["enchanting"] = grant_plains
    abundant_growth2.flags["bonus_mana_colors"] = {"G", "W"}
    state.battlefield = [plain_plains, grant_plains, abundant_growth2]

    assert ("Plains", "G") in mana.mana_ability_options(state)  # only the ENCHANTED Plains can make G
    mana.activate_mana_source(state, grant_plains, "G")
    assert grant_plains.tapped and not plain_plains.tapped
    assert state.mana_pool_single_pip == {"G": 1}  # a 1-symbol event -- tagged single-pip
    try:
        mana.mana_output(plain_plains, state, "G")
        assert False, "the unenchanted Plains must not be able to produce G"
    except ValueError:
        pass


def test_discount_departing_source_ignores_permanent_without_mana_spec():
    # No registry "mana" spec at all (e.g. a plain non-mana creature) is a no-op.
    state = GameState(on_the_play=True)
    mana.float_mana(state, ["R"])
    creature = Permanent(CardDef("Some Creature", CardType.CREATURE, {"generic": 1}, EffectId.FILLER))
    mana.discount_departing_source(state, creature, 0)
    assert state.mana_pool_single_pip == {"R": 1}  # untouched


def test_discount_departing_source_no_op_when_nothing_tagged():
    state = GameState(on_the_play=True)
    mountain = Permanent(CardDef("Mountain", CardType.LAND, None, EffectId.MOUNTAIN))
    mana.discount_departing_source(state, mountain, 0)  # nothing floating at all
    assert state.mana_pool_single_pip == {}


def test_discount_departing_source_fixed_color_discounts_tag():
    # Fireblast's scenario: a Mountain tapped for {R}, then sacrificed --
    # the tag (not the mana itself) is excused.
    state = GameState(on_the_play=True)
    mountain = Permanent(CardDef("Mountain", CardType.LAND, None, EffectId.MOUNTAIN))
    state.battlefield = [mountain]
    mana.activate_mana_source(state, mountain)
    assert state.mana_pool_single_pip == {"R": 1}
    mana.discount_departing_source(state, mountain, 0)
    assert state.mana_pool_single_pip == {}
    assert state.mana_pool == {"R": 1}  # the floating mana itself is untouched


def test_discount_departing_source_flexible_picks_higher_tagged_candidate():
    # A dual land (flexible kind) discounts whichever candidate color
    # currently has the most tagged mana floating, not a fixed choice.
    filler_backup = registry.EFFECT_REGISTRY[EffectId.FILLER]
    registry.EFFECT_REGISTRY[EffectId.FILLER] = {"mana": ("flexible", {"W", "U"})}
    try:
        state = GameState(on_the_play=True)
        dual = Permanent(CardDef("Dual-ish", CardType.LAND, None, EffectId.FILLER))
        mana.float_mana(state, ["W"])
        mana.float_mana(state, ["W"])
        mana.float_mana(state, ["U"])
        assert state.mana_pool_single_pip == {"W": 2, "U": 1}
        mana.discount_departing_source(state, dual, 0)
        assert state.mana_pool_single_pip == {"W": 1, "U": 1}  # W had more tagged, so W is discounted
    finally:
        registry.EFFECT_REGISTRY[EffectId.FILLER] = filler_backup


def test_discount_departing_source_flexible_tie_break_uses_state_rng():
    # On a tie, the choice comes from state.rng, reproducible given the same seed.
    filler_backup = registry.EFFECT_REGISTRY[EffectId.FILLER]
    registry.EFFECT_REGISTRY[EffectId.FILLER] = {"mana": ("flexible", {"W", "U"})}
    try:
        state = GameState(on_the_play=True, rng=random.Random(0))
        dual = Permanent(CardDef("Dual-ish", CardType.LAND, None, EffectId.FILLER))
        mana.float_mana(state, ["W"])
        mana.float_mana(state, ["U"])
        assert state.mana_pool_single_pip == {"W": 1, "U": 1}  # a genuine tie

        expected = random.Random(0).choice(["U", "W"])  # sorted candidate order, same seed
        mana.discount_departing_source(state, dual, 0)
        remaining = "U" if expected == "W" else "W"
        assert state.mana_pool_single_pip == {remaining: 1}
    finally:
        registry.EFFECT_REGISTRY[EffectId.FILLER] = filler_backup


def test_discount_departing_source_out_of_scope_kind_is_a_no_op():
    # fixed_multi/tron/count/count_all are deliberately out of scope.
    filler_backup = registry.EFFECT_REGISTRY[EffectId.FILLER]
    registry.EFFECT_REGISTRY[EffectId.FILLER] = {"mana": ("fixed_multi", ("B", "R"))}
    try:
        state = GameState(on_the_play=True)
        source = Permanent(CardDef("Carnarium-ish", CardType.LAND, None, EffectId.FILLER))
        mana.float_mana(state, ["B"])  # pretend a genuine single-pip B is floating from elsewhere
        assert state.mana_pool_single_pip == {"B": 1}
        mana.discount_departing_source(state, source, 0)
        assert state.mana_pool_single_pip == {"B": 1}  # untouched
    finally:
        registry.EFFECT_REGISTRY[EffectId.FILLER] = filler_backup


# Mana filters (Conduit Pylons / Barrels of Blasting Jelly) are owned by
# the action layer (drl_env._filter_mana_*), not a mana primitive here --
# their own test lives with that code, not in this module.
