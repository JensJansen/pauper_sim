"""Tests for game.mana float-first mana system: activate_mana_source,
begin_pay_cost/execute_pool_spend, pool_can_pay/plan_payment, and the
Boggles-style mana-fixing Auras (Utopia Sprawl automatic bonus, Abundant
Growth competing granted ability)."""

from game import mana, registry
from game.cards import CardDef, CardType, EffectId
from game.state import GameState, Permanent


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
    finally:
        registry.EFFECT_REGISTRY[EffectId.FILLER] = filler_backup


def test_count_source_produces_one_symbol_per_matching_permanent():
    # Overgrown Battlement (real card, "count" kind -- one G per Defender
    # you control, itself included): 3 Defenders on the battlefield means
    # ONE tap of Battlement alone produces 3 G, not just 1.
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


def test_pool_can_pay_edges():
    # pool affordability edges: colorless pays generic but never a colored pip.
    assert mana.pool_can_pay({"C": 2}, {"generic": 2}) and not mana.pool_can_pay({"C": 1}, {"R": 1})
    assert mana.pool_can_pay({}, {})  # free cost


def test_begin_pay_cost_empty_cost_completes_immediately():
    # begin_pay_cost: an empty cost (e.g. Lotus Petal's {} cast cost) completes
    # immediately -- nothing to spend, no dangling pending.
    state = GameState(on_the_play=True)
    resolved = []
    mana.begin_pay_cost(state, {}, on_complete=lambda s: resolved.append(True))
    assert state.pending_resolution is None
    assert resolved == [True]


def test_utopia_sprawl_automatic_bonus_mana():
    # Boggles' two mana-fixing Auras need genuinely different treatment:
    # Utopia Sprawl's bonus is automatic (always on top of the land's own
    # output, no extra choice), Abundant Growth's is a competing ability
    # (the model picks native or granted each tap) -- see mana_output's
    # own module comments. Exercised directly against a real Forest, using
    # a synthetic Aura permanent (a real Utopia Sprawl CardDef, just not
    # attached via the real cast_aura flow).
    state = GameState(on_the_play=True)
    forest = Permanent(CardDef("Forest", CardType.LAND, None, EffectId.FOREST))
    utopia_sprawl = Permanent(CardDef("Utopia Sprawl", CardType.ENCHANTMENT, {"G": 1}, EffectId.UTOPIA_SPRAWL))
    utopia_sprawl.flags["enchanting"] = forest
    utopia_sprawl.flags["bonus_mana_color"] = "W"
    state.battlefield = [forest, utopia_sprawl]

    assert mana.mana_output(forest, state) == ["G", "W"]  # native G, plus Utopia Sprawl's automatic bonus
    mana.activate_mana_source(state, forest)  # one activation floats native G AND the automatic bonus W
    assert state.mana_pool == {"G": 1, "W": 1} and forest.tapped
    # pay a {W} cost by spending the floated W; the unneeded G stays in the pool.
    mana.begin_pay_cost(state, {"W": 1}, on_complete=lambda s: None)
    assert state.pending_resolution is not None
    mana.execute_pool_spend(state, "W")
    assert state.pending_resolution is None  # {W} paid from the pool
    assert state.mana_pool.get("G", 0) == 1  # the native G stays floating, unneeded here


def test_abundant_growth_competing_granted_ability():
    # Abundant Growth: Plains gets a genuinely competing "any of {G, W}"
    # ability -- both its own native W and the grant stay usable.
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
    mana.begin_pay_cost(state, {"G": 1}, on_complete=lambda s: None)
    mana.execute_pool_spend(state, "G")
    assert state.pending_resolution is None  # {G} covered via the granted color


def test_abundant_growth_granted_color_only_via_enchanted_permanent():
    # mana_ability_options must offer the granted color only via the ENCHANTED
    # Plains, so the caller can activate_mana_source that exact permanent
    # specifically -- even with an identical-by-name plain Plains also in
    # play. Same-named sources are normally fully interchangeable in this
    # engine; a granted-mana Aura is the one case that breaks that, since
    # only the enchanted copy can actually produce the granted color.
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
    try:
        mana.mana_output(plain_plains, state, "G")
        assert False, "the unenchanted Plains must not be able to produce G"
    except ValueError:
        pass


# Mana filters (Conduit Pylons / Barrels of Blasting Jelly) are a two-step
# pay-then-choose-color action ("Filter X, paying <color>" pays the {1}
# immediately; the output color is chosen afterward via the shared
# choose_color mana_subdecision stage) owned by the action layer
# (drl_env._filter_mana_*), not a mana primitive here -- so their own test
# lives with that code, not in this module.
