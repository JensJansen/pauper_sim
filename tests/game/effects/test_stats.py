"""Migrated from game/effects/stats.py's __main__ ponytail self-check.

Cross-player Aura reads (a real bug found while building mutual combat
damage): permanent_power/permanent_toughness/enchantment_count used to read
state.battlefield (active-player-proxied) to find enchanting Auras --
silently wrong for a BLOCKER (the defender's own creature), since
combat_damage_step always runs with active_idx on the ATTACKER."""
from game import registry
from game.cards import CardDef, CardType, EffectId
from game.effects.stats import can_be_targeted, has_keyword, permanent_power, permanent_toughness
from game.state import GameState, Permanent, PlayerState


def test_cross_player_aura_reads():
    state = GameState(on_the_play=True, players=[PlayerState(True), PlayerState(False)])
    defenders_creature = Permanent(CardDef("Defender's Creature", CardType.CREATURE, None, EffectId.FILLER, power=1, toughness=1))
    rancor_on_defender = Permanent(CardDef("Rancor", CardType.ENCHANTMENT, {"G": 1}, EffectId.RANCOR))
    rancor_on_defender.flags["enchanting"] = defenders_creature
    mask_on_defender = Permanent(CardDef("Ancestral Mask", CardType.ENCHANTMENT, {"generic": 2, "G": 1}, EffectId.ANCESTRAL_MASK))
    mask_on_defender.flags["enchanting"] = defenders_creature
    state.players[1].battlefield = [defenders_creature, rancor_on_defender, mask_on_defender]
    state.active_idx = 0  # the ATTACKER's own perspective -- defender's battlefield is NOT state.battlefield right now

    # power: 1 base + 2 (Rancor) + 2 (Ancestral Mask, 1 OTHER enchantment
    # -- Rancor -- so +2, not +4). toughness: 1 base + 0 (Rancor, power
    # only) + 2 (Ancestral Mask, symmetric).
    assert permanent_power(state, defenders_creature) == 5
    assert permanent_toughness(state, defenders_creature) == 3
    assert has_keyword(state, defenders_creature, "flying") is False


def test_counters_flat_pt_bonus():
    # Counters: a flat per-permanent bonus, no battlefield scan involved --
    # "+1/+1" (Nyxborn Hydra) and "-0/-1" (Wall of Roots) both fold straight
    # into base power/toughness alongside the existing Aura bonus.
    state = GameState(on_the_play=True, players=[PlayerState(True), PlayerState(False)])

    hydra = Permanent(CardDef("Some Hydra", CardType.CREATURE, None, EffectId.FILLER, power=0, toughness=1))
    hydra.counters["+1/+1"] = 3
    assert permanent_power(state, hydra) == 3
    assert permanent_toughness(state, hydra) == 4

    wall = Permanent(CardDef("Some Wall", CardType.CREATURE, None, EffectId.FILLER, power=0, toughness=5))
    wall.counters["-0/-1"] = 5
    assert permanent_power(state, wall) == 0
    assert permanent_toughness(state, wall) == 0  # 5 counters against toughness 5 -- lethal via the ordinary SBA check, not special-cased here


def test_animate_spec_threshold():
    # _animate_spec: Pinnacle Kill-Ship's own Station -- below its charge
    # threshold, ordinary card_def stats/keywords apply; at/above it,
    # animate's own power/toughness/keywords fully override, unrelated to
    # any counter's own _COUNTER_PT contribution (charge isn't listed there
    # at all, so it never double-counts as a stat bonus on its own).
    state = GameState(on_the_play=True, players=[PlayerState(True), PlayerState(False)])
    _filler_backup = registry.EFFECT_REGISTRY[EffectId.FILLER]
    registry.EFFECT_REGISTRY[EffectId.FILLER] = {
        "animate": {"counter": "charge", "threshold": 7, "power": 7, "toughness": 7, "keywords": {"flying"}},
    }
    try:
        ship = Permanent(CardDef("Some Ship", CardType.ARTIFACT, None, EffectId.FILLER))
        ship.counters["charge"] = 6
        assert permanent_power(state, ship) == 0 and permanent_toughness(state, ship) == 0
        assert has_keyword(state, ship, "flying") is False
        ship.counters["charge"] = 7
        assert permanent_power(state, ship) == 7 and permanent_toughness(state, ship) == 7
        assert has_keyword(state, ship, "flying") is True
    finally:
        registry.EFFECT_REGISTRY[EffectId.FILLER] = _filler_backup


def test_can_be_targeted_hexproof_and_shroud():
    # can_be_targeted: hexproof blocks OPPONENTS only; shroud blocks everyone;
    # a vanilla creature is targetable by anyone. Controller = whichever
    # battlefield the permanent sits on.
    ct_state = GameState(on_the_play=True, players=[PlayerState(True), PlayerState(False)])
    vanilla = Permanent(CardDef("Vanilla", CardType.CREATURE, None, EffectId.FILLER, power=1, toughness=1))
    ct_state.players[0].battlefield = [vanilla]
    assert can_be_targeted(ct_state, vanilla, 0) and can_be_targeted(ct_state, vanilla, 1)

    _filler_backup2 = registry.EFFECT_REGISTRY[EffectId.FILLER]
    try:
        registry.EFFECT_REGISTRY[EffectId.FILLER] = {"keywords": {"hexproof"}}
        hexed = Permanent(CardDef("Hexed", CardType.CREATURE, None, EffectId.FILLER, power=1, toughness=1))
        ct_state.players[0].battlefield = [hexed]
        assert can_be_targeted(ct_state, hexed, 0)       # its own controller may target it
        assert not can_be_targeted(ct_state, hexed, 1)   # an opponent may not

        registry.EFFECT_REGISTRY[EffectId.FILLER] = {"keywords": {"shroud"}}
        shrouded = Permanent(CardDef("Shrouded", CardType.CREATURE, None, EffectId.FILLER, power=1, toughness=1))
        ct_state.players[0].battlefield = [shrouded]
        assert not can_be_targeted(ct_state, shrouded, 0)  # nobody may target it, not even its controller
        assert not can_be_targeted(ct_state, shrouded, 1)
    finally:
        registry.EFFECT_REGISTRY[EffectId.FILLER] = _filler_backup2
