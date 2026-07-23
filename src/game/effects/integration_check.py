"""ponytail self-check for scenarios that exercise multiple effects/
submodules together and don't belong to any single one of them -- run via
`python -m game.effects.integration_check` from src/. Every other
submodule's own single-module scenarios live in its own __main__ block
instead; this file is only for the handful that genuinely need two or
more modules cooperating to mean anything."""

from . import combat, state_based, stack, triggers, madness_and_plot
from .. import mana, registry, resolution
from ..cards import CardDef, CardType, EffectId
from ..state import GameState, Permanent, PlayerState

# -- Madness chain end to end: triggers.py + stack.py + madness_and_plot.py
# + mana.py + resolution.py, all in one pass (discard -> exile + queue ->
# drain -> decision -> pay madness cost -> resolve -> drain again). No real
# madness card exists yet (deck assembly out of scope), so this borrows
# EffectId.FILLER for the fake spell, saving/restoring its real (empty)
# registry entry around the check; the mana source is a genuine Forest
# CardDef (EffectId.FOREST already has a real "mana" spec, no faking
# needed for that half).
resolved_calls = []


def _fake_resolve(s, c):
    resolved_calls.append(c.name)


_filler_backup = registry.EFFECT_REGISTRY[EffectId.FILLER]
registry.EFFECT_REGISTRY[EffectId.FILLER] = {"madness": {"cost": {"G": 1}, "resolve": _fake_resolve}}
try:
    madness_card = CardDef("Fake Madness Spell", CardType.INSTANT, {"generic": 1, "R": 1}, EffectId.FILLER)
    state = GameState(on_the_play=True)
    state.hand = [madness_card]
    state.battlefield = [Permanent(CardDef("Forest", CardType.LAND, None, EffectId.FOREST))]

    completed = []
    resolution.begin_discard(state, 1, optional=False, on_complete=lambda s, cards: completed.append(cards))
    resolution.execute_discard_option(state, "Fake Madness Spell")
    assert completed == [[madness_card]]
    assert state.pending_resolution is None  # discard's own resolution is done; nothing queued mid-discard
    assert len(state.trigger_queue) == 1 and state.trigger_queue[0]["kind"] == "madness"

    # Top-level orchestration point (game.turn's own priority round in
    # real play, once the enclosing top-level action is fully done):
    # promote. The trigger itself is now a real stack entry -- an
    # opponent gets a priority window before the cast-or-decline choice
    # is even offered (docs/PRIORITY_PLAN.md item 1) -- so resolving it
    # is what actually opens the decision, not promotion itself.
    triggers.promote_triggers_to_stack(state)
    assert state.pending_resolution is None  # just sitting on the stack, not open yet
    assert len(state.stack) == 1 and state.stack[0]["card_def"] is madness_card
    stack.resolve_top_of_stack(state)
    assert state.pending_resolution["kind"] == "madness_decision"
    assert resolution.madness_decision_options(state) == ["cast", "decline"]

    madness_and_plot.execute_madness_cast(state)
    # begin_pay_cost's own resolution is now pending (paying {G}) --
    # confirms this correctly nests through it via the captured
    # outer_on_complete, instead of crashing on a stale pending_resolution
    # reference once pay_cost's own resolution clears it.
    assert state.pending_resolution["kind"] == "pay_cost"
    assert mana.tap_cost_options(state) == [("Forest", None, False)]
    mana.execute_tap_cost_option(state, "Forest", None, False)
    # Pool-only model (MANA_POOL_PLAN.md): the tap only floats {G} into the
    # pool -- paying the cost is a separate, explicit spend.
    assert state.pending_resolution["kind"] == "pay_cost"
    mana.execute_pool_spend(state, "G")

    # Payment complete -> pushed to the stack (not resolved yet, a real
    # independent stack entry) -> the enclosing madness_decision's own
    # on_complete fires (a no-op today) -> fully back to no pending
    # resolution. The effect itself only fires once something actually
    # resolves the stack (a "Pass" in real play).
    assert resolved_calls == []
    assert len(state.stack) == 1 and state.stack[0]["card_def"] is madness_card
    assert state.pending_resolution is None
    assert state.trigger_queue == []
    stack.resolve_top_of_stack(state)
    assert resolved_calls == ["Fake Madness Spell"]
    assert state.stack == []
    assert state.exile == []
finally:
    registry.EFFECT_REGISTRY[EffectId.FILLER] = _filler_backup

print("integration_check.py madness chain (triggers+stack+madness_and_plot+mana+resolution): OK")

# -- Madness cast, then Abandon payment (regression -- execute_madness_cast
# used to remove the card from exile BEFORE begin_pay_cost, breaking
# mana.abandon_pay_cost's own documented contract that no zone is touched
# until on_complete fires; a model choosing Cast then Abandon made the card
# vanish from every zone instead of leaving it exiled, same as if Cast had
# never been chosen). Same discard -> exile -> decision setup as above.
_filler_backup = registry.EFFECT_REGISTRY[EffectId.FILLER]
registry.EFFECT_REGISTRY[EffectId.FILLER] = {"madness": {"cost": {"G": 1}, "resolve": _fake_resolve}}
try:
    madness_card = CardDef("Fake Madness Spell", CardType.INSTANT, {"generic": 1, "R": 1}, EffectId.FILLER)
    state = GameState(on_the_play=True)
    state.hand = [madness_card]
    state.battlefield = [Permanent(CardDef("Forest", CardType.LAND, None, EffectId.FOREST))]

    resolution.begin_discard(state, 1, optional=False, on_complete=lambda s, cards: None)
    resolution.execute_discard_option(state, "Fake Madness Spell")
    triggers.promote_triggers_to_stack(state)
    stack.resolve_top_of_stack(state)
    assert state.pending_resolution["kind"] == "madness_decision"

    madness_and_plot.execute_madness_cast(state)
    assert state.pending_resolution["kind"] == "pay_cost"
    assert state.exile == [(madness_card, None)]  # still exiled -- payment not committed yet

    mana.abandon_pay_cost(state)
    assert state.pending_resolution is None
    assert state.exile == [(madness_card, None)]  # unchanged, not vanished
    assert state.graveyard == [] and state.hand == []
finally:
    registry.EFFECT_REGISTRY[EffectId.FILLER] = _filler_backup

print("integration_check.py madness cast + abandon payment (no-vanish regression): OK")

# -- Blocking's own mutual combat damage + creature death (docs/COMBAT_
# PLAN.md steps 5/6): combat.py's own combat_damage_step handing off to
# state_based.py's check_state_based_actions -- a blocked attacker deals
# no damage to the opponent (absorbed), but IT and its blocker now fight
# each other for real, and check_state_based_actions kills whichever
# side(s) that's lethal for. Needs a genuine 2-player GameState: a dying
# BLOCKER's own zones belong to the DEFENDER, not whichever side
# state.active_idx (the attacker, throughout combat_damage_step)
# currently proxies to -- exactly the bug state_based._destroy_creature's
# own docstring explains. Two blocked pairs in ONE combat: pair A's
# attacker dies, its blocker survives; pair B's blocker dies, its
# attacker survives -- proving the death check applies independently per
# creature, not "whoever's weaker overall."
_card_defs_backup = dict(registry.CARD_DEFS)
try:
    state = GameState(on_the_play=True, players=[PlayerState(True), PlayerState(False)])
    attacker_a = Permanent(CardDef("Attacker A", CardType.CREATURE, None, EffectId.FILLER, power=1, toughness=3))
    attacker_a.summoning_sick = False
    attacker_b = Permanent(CardDef("Attacker B", CardType.CREATURE, None, EffectId.FILLER, power=5, toughness=3))
    attacker_b.summoning_sick = False
    unblocked_attacker = Permanent(CardDef("Unblocked", CardType.CREATURE, None, EffectId.FILLER, power=2, toughness=1))
    unblocked_attacker.summoning_sick = False
    blocker_a = Permanent(CardDef("Blocker A", CardType.CREATURE, None, EffectId.FILLER, power=5, toughness=2))
    blocker_b = Permanent(CardDef("Blocker B", CardType.CREATURE, None, EffectId.FILLER, power=1, toughness=2))
    for p in (attacker_a, attacker_b, unblocked_attacker, blocker_a, blocker_b):
        registry.CARD_DEFS[p.card_def.name] = p.card_def
    state.players[0].battlefield = [attacker_a, attacker_b, unblocked_attacker]
    state.players[1].battlefield = [blocker_a, blocker_b]

    combat.declare_attackers_step(state)
    combat.declare_attacker(state, attacker_a)
    combat.declare_attacker(state, attacker_b)
    combat.declare_attacker(state, unblocked_attacker)
    state.blocked_by[attacker_a] = blocker_a
    state.blocked_by[attacker_b] = blocker_b
    combat.combat_damage_step(state)

    assert state.damage_dealt == 2  # only the unblocked attacker's power counts -- A/B's damage is absorbed
    assert state.attackers == []
    # Pair A: attacker_a (toughness 3) took blocker_a's 5 power -> dead.
    # blocker_a (toughness 2) took attacker_a's 1 power -> survives.
    assert attacker_a not in state.players[0].battlefield and blocker_a in state.players[1].battlefield
    assert [c.name for c in state.players[0].graveyard] == ["Attacker A"]  # landed in ITS OWN owner's graveyard, not the active player's by coincidence
    assert blocker_a.damage_marked == 1 and not blocker_a.tapped
    # Pair B: attacker_b (toughness 3) took blocker_b's 1 power -> survives.
    # blocker_b (toughness 2) took attacker_b's 5 power -> dead, and its
    # zones are the DEFENDER's, proving _destroy_creature found the right
    # owner rather than assuming state.active_idx (still the attacker here).
    assert attacker_b in state.players[0].battlefield and blocker_b not in state.players[1].battlefield
    assert [c.name for c in state.players[1].graveyard] == ["Blocker B"]
    assert attacker_b.damage_marked == 1 and attacker_b.tapped

    # A fresh combat resets blocked_by too, not just attackers.
    combat.declare_attackers_step(state)
    assert state.blocked_by == {}

    # cleanup_step clears damage_marked for EVERY permanent, BOTH players.
    assert blocker_a.damage_marked == 1
    state_based.cleanup_step(state)
    assert state.pending_resolution is None
    assert blocker_a.damage_marked == 0
    assert attacker_b.damage_marked == 0
finally:
    registry.CARD_DEFS.clear()
    registry.CARD_DEFS.update(_card_defs_backup)

print("integration_check.py combat+state_based mutual-damage/creature-death handoff: OK")
