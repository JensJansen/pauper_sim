"""Battlefield entry and the direct cast paths (casting.py): land bounce ETB,
Aura targeting/attachment/fizzle, targeted-creature spells, and the shared
capture_any_target / target_still_legal targeting contract."""
from game import registry, resolution
from game.cards import CardDef, CardType, EffectId
from game.effects import stats
from game.effects.casting import (
    bounce_land_etb,
    capture_any_target,
    cast_aura,
    cast_targeting_creature,
    enters_battlefield,
    target_still_legal,
)
from game.effects.stack import resolve_top_of_stack
from game.effects.triggers import promote_triggers_to_stack
from game.state import GameState, Permanent, PlayerState


def test_land_bounce_etb_is_queued_not_inline():
    _filler_backup = registry.EFFECT_REGISTRY[EffectId.FILLER]
    registry.EFFECT_REGISTRY[EffectId.FILLER] = {"etb_trigger": lambda state, permanent: bounce_land_etb(state)}
    try:
        carnarium = CardDef("Fake Carnarium", CardType.LAND, None, EffectId.FILLER)
        state = GameState(on_the_play=True)
        state.hand = [carnarium]
        state.battlefield = [
            Permanent(CardDef("Forest", CardType.LAND, None, EffectId.FOREST)),
            Permanent(CardDef("Swamp", CardType.LAND, None, EffectId.SWAMP)),
        ]

        state.hand.remove(carnarium)
        enters_battlefield(state, carnarium)  # normal ETB path, like play_land_from_hand
        # ETB is queued, not run inline -- choose_permanent opens only once
        # promoted to the stack and resolved.
        assert state.pending_resolution is None
        assert [e["type"] for e in state.trigger_queue] == ["etb"]
        promote_triggers_to_stack(state)
        resolve_top_of_stack(state)
        assert state.pending_resolution["kind"] == "choose_permanent"
        assert resolution.choose_permanent_options(state) == [
            ("Fake Carnarium", 1), ("Forest", 1), ("Swamp", 1),
        ]
        resolution.execute_choose_permanent_option(state, "Swamp", 1)
        assert state.pending_resolution is None
        assert sorted(p.card_def.name for p in state.battlefield) == ["Fake Carnarium", "Forest"]
        assert [c.name for c in state.hand] == ["Swamp"]
    finally:
        registry.EFFECT_REGISTRY[EffectId.FILLER] = _filler_backup


def _bogle_with_rancor_attached():
    """Reproduces the Aura self-check's own setup: a Slippery Bogle on the
    battlefield with a cast, resolved Rancor attached (effective power 3)."""
    state = GameState(on_the_play=True)
    bogle = Permanent(CardDef("Slippery Bogle", CardType.CREATURE, {"G": 1}, EffectId.SLIPPERY_BOGLE, power=1, toughness=1))
    state.battlefield = [bogle]
    rancor = CardDef("Rancor", CardType.ENCHANTMENT, {"G": 1}, EffectId.RANCOR)
    state.hand = [rancor]
    cast_aura(state, rancor, lambda p: p.card_def.card_type == CardType.CREATURE)
    resolution.execute_choose_any_target_creature(state, 0, "Slippery Bogle", 1)
    resolve_top_of_stack(state)
    return state, bogle


def test_cast_aura_attaches_and_grants_bonus():
    state = GameState(on_the_play=True)
    bogle = Permanent(CardDef("Slippery Bogle", CardType.CREATURE, {"G": 1}, EffectId.SLIPPERY_BOGLE, power=1, toughness=1))
    state.battlefield = [bogle]
    assert stats.permanent_power(state, bogle) == 1
    assert stats.permanent_toughness(state, bogle) == 1

    rancor = CardDef("Rancor", CardType.ENCHANTMENT, {"G": 1}, EffectId.RANCOR)
    state.hand = [rancor]
    cast_aura(state, rancor, lambda p: p.card_def.card_type == CardType.CREATURE)
    assert resolution.choose_any_target_creature_options(state) == [(0, "Slippery Bogle", 1)]  # own creature, side 0
    resolution.execute_choose_any_target_creature(state, 0, "Slippery Bogle", 1)
    assert state.pending_resolution is None
    assert state.hand == [] and len(state.stack) == 1  # left hand at cast, sitting on the stack
    resolve_top_of_stack(state)
    assert state.hand == []
    rancor_permanent = next(p for p in state.battlefield if p.card_def.name == "Rancor")
    assert rancor_permanent.flags["enchanting"] is bogle
    assert stats.permanent_power(state, bogle) == 3  # 1 base + Rancor's own +2
    assert stats.permanent_toughness(state, bogle) == 1  # unchanged -- Rancor is +2/+0


def test_cast_aura_target_fizzle_when_target_dies_before_resolution():
    # the targeted permanent is gone by resolution: the spell fizzles
    # outright, straight to graveyard, never entering the battlefield.
    state, bogle = _bogle_with_rancor_attached()

    other_bogle = enters_battlefield(
        state, CardDef("Slippery Bogle", CardType.CREATURE, {"G": 1}, EffectId.SLIPPERY_BOGLE, power=1, toughness=1),
    )
    assert other_bogle.slot == 2  # bogle (still on the battlefield) already occupies slot 1

    ethereal_armor = CardDef("Ethereal Armor", CardType.ENCHANTMENT, {"W": 1}, EffectId.ETHEREAL_ARMOR)
    state.hand = [ethereal_armor]
    cast_aura(state, ethereal_armor, lambda p: p.card_def.card_type == CardType.CREATURE)
    assert (0, "Slippery Bogle", 2) in resolution.choose_any_target_creature_options(state)
    resolution.execute_choose_any_target_creature(state, 0, "Slippery Bogle", 2)  # targets other_bogle specifically
    state.battlefield.remove(other_bogle)  # dies before the cast resolves

    state.event_log = []
    resolve_top_of_stack(state)
    assert any(e["kind"] == "target_fizzle" for e in state.event_log)
    assert state.hand == []
    assert any(c.name == ethereal_armor.name for c in state.graveyard)  # graveyard holds a fresh instance
    assert not any(p.card_def.name == "Ethereal Armor" for p in state.battlefield)
    assert stats.permanent_power(state, bogle) == 3  # unaffected -- the fizzled Aura was never targeting bogle


def test_cast_targeting_creature_captured_none_when_no_legal_target_at_cast():
    # no creature anywhere at cast time: the mandatory pick auto-completes
    # with captured=None via begin_choose_any_target's zero-candidate
    # contract, before the spell even reaches the stack.
    state = GameState(on_the_play=True)
    snuff_out = CardDef("Snuff Out", CardType.INSTANT, {"B": 1}, EffectId.FILLER)
    state.hand = [snuff_out]
    resolved_calls = []
    cast_targeting_creature(state, snuff_out, on_resolve=lambda s, target: resolved_calls.append(target))
    assert state.pending_resolution is None  # auto-completed immediately, no resolution left dangling
    assert state.hand == [] and len(state.stack) == 1  # left hand at cast, sitting on the stack with zero targets
    resolve_top_of_stack(state)
    assert resolved_calls == []  # on_resolve never ran -- captured=None fizzled, same contract as a target dying first
    assert any(c.name == "Snuff Out" for c in state.graveyard)  # the spell itself still goes to the graveyard


def test_cast_aura_no_target_fallback_bestow():
    # cast_aura's no_target_fallback (Bestow, 702.103e): captured=None at
    # cast time enters the battlefield via the fallback instead of fizzling.
    no_target_fallback_calls = []

    def _bestow_no_target_fallback(state, card_def):
        no_target_fallback_calls.append(card_def)
        enters_battlefield(state, card_def, from_zone="hand")  # 702.103e: enters as a creature instead of fizzling

    state = GameState(on_the_play=True)
    bestow_aura = CardDef("Fake Bestow Aura", CardType.ENCHANTMENT, {"G": 1}, EffectId.FILLER, power=2, toughness=2)
    state.hand = [bestow_aura]
    cast_aura(
        state, bestow_aura, lambda p: p.card_type == CardType.CREATURE, no_target_fallback=_bestow_no_target_fallback,
    )
    assert state.pending_resolution is None
    assert state.hand == [] and len(state.stack) == 1
    resolve_top_of_stack(state)
    assert no_target_fallback_calls == [bestow_aura]  # the fallback ran instead of the plain graveyard fizzle
    assert not any(c.name == "Fake Bestow Aura" for c in state.graveyard)
    assert any(p.card_def.name == "Fake Bestow Aura" for p in state.battlefield)  # entered as a creature, per 702.103e


def test_capture_any_target_and_target_still_legal():
    # two same-named creatures on opposite sides: capture must lock the
    # exact one named by (side, name, slot), and legality flips only when
    # that specific object leaves, not its same-named twin.
    mine = Permanent(CardDef("Grizzly Bears", CardType.CREATURE, {"G": 1}, EffectId.FILLER, power=2, toughness=2))
    theirs = Permanent(CardDef("Grizzly Bears", CardType.CREATURE, {"G": 1}, EffectId.FILLER, power=2, toughness=2))
    mine.slot = theirs.slot = 1
    tstate = GameState(on_the_play=True, players=[PlayerState(True), PlayerState(False)])
    tstate.players[0].battlefield = [mine]
    tstate.players[1].battlefield = [theirs]
    captured = capture_any_target(tstate, ("creature", 1, "Grizzly Bears", 1))
    assert captured == ("creature", theirs)  # the opponent's copy, by identity -- not mine
    assert target_still_legal(tstate, captured)
    tstate.players[0].battlefield = []  # MY copy leaves -- the captured target (theirs) is untouched
    assert target_still_legal(tstate, captured)
    tstate.players[1].battlefield = []  # the captured copy leaves -> now illegal (fizzle)
    assert not target_still_legal(tstate, captured)
    assert capture_any_target(tstate, ("player", 0)) == ("player", 0)  # players pass through, always legal
    assert target_still_legal(tstate, ("player", 0))
    assert capture_any_target(tstate, None) is None and target_still_legal(tstate, None)  # no target -> no fizzle
