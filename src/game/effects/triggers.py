"""The trigger queue: moving a queued trigger (Sneaky Snacker's automatic
return, a Madness decision, an ETB/LTB/upkeep/take_initiative/venture/Ward/
cast/sacrifice ability) onto the priority stack. Sits above casting.py and
stack.py -- _trigger_resolve's "automatic" branch needs casting.
enters_battlefield."""

from . import casting
from .stack import counter_spell, push_to_stack
from .. import registry, resolution


def _trigger_resolve(entry):
    """Build the resolve(state, card_def) for one queued trigger, deferred
    until this stack entry resolves. Branches on entry["type"]:

    "automatic" (Sneaky Snacker's on-draw return): runs the effect directly.
    "decision" (Madness): opens the cast-or-decline choice only here,
    matching "you may cast it as this ability resolves"."""
    if entry["type"] == "etb":
        # ETB trigger (casting.enters_battlefield queued it). Hook takes
        # (state, permanent); looked up fresh at resolution.
        permanent = entry["permanent"]

        def resolve(state, card_def):
            etb = registry.EFFECT_REGISTRY.get(card_def.effect_id, {}).get("etb_trigger")
            if etb is not None:
                etb(state, permanent)
        return resolve
    if entry["type"] == "ltb":
        # LTB trigger (state_based._queue_leave_triggers queued it).
        # `permanent` is already off the battlefield but still carries
        # whatever the trigger needs (e.g. Mesmeric Fiend's linked card).
        permanent = entry["permanent"]

        def resolve(state, card_def):
            ltb = registry.EFFECT_REGISTRY.get(card_def.effect_id, {}).get("ltb_trigger")
            if ltb is not None:
                ltb(state, permanent)
        return resolve
    if entry["type"] == "upkeep":
        # "At the beginning of your upkeep" (Delver of Secrets), queued by
        # game.turn.upkeep_step.
        permanent = entry["permanent"]

        def resolve(state, card_def):
            upkeep = registry.EFFECT_REGISTRY.get(card_def.effect_id, {}).get("upkeep_trigger")
            if upkeep is not None:
                upkeep(state, permanent)
        return resolve
    if entry["type"] == "take_initiative":
        # CR 722.2's second Initiative trigger: queued by combat.
        # combat_damage_step once a player's creatures deal combat damage to
        # the current holder. Resolving it flips state.initiative_idx and
        # queues that player's own venture trigger. Lazy import -- undercity
        # pulls in casting/tokens, above this module.
        player_idx = entry["player_idx"]

        def resolve(state, card_def):
            from . import undercity
            undercity.take_initiative(state, player_idx)
        return resolve
    if entry["type"] == "venture":
        # "Venture into Undercity", queued by undercity.queue_venture (on
        # taking the initiative or at upkeep). Lazy import as above.
        player_idx = entry["player_idx"]

        def resolve(state, card_def):
            from . import undercity
            undercity.venture(state, player_idx)
        return resolve
    if entry["type"] == "ward":
        # Ward (Tolarian Terror): queued by casting._maybe_trigger_ward.
        # Promoted above the triggering spell, so on resolution that spell
        # is state.stack[-1]. payer_idx pays the ward cost or the spell is
        # countered; fizzles if the spell already left the stack.
        payer_idx = entry["payer_idx"]
        cost = entry["cost"]

        def resolve(state, card_def):
            if not state.stack:
                return
            spell_entry = state.stack[-1]

            def _on_pay(state, paid):
                if not paid:
                    counter_spell(state, spell_entry)

            resolution.begin_pay_unless(state, payer_idx, cost, _on_pay)
        return resolve
    if entry["type"] == "cast_trigger":
        # "Whenever you cast an instant/sorcery" (Guttersnipe --
        # stack.on_cast_trigger queued it).
        permanent = entry["permanent"]

        def resolve(state, card_def):
            trigger = registry.EFFECT_REGISTRY.get(card_def.effect_id, {}).get("on_cast")
            if trigger is not None:
                trigger(state, permanent)
        return resolve
    if entry["type"] == "on_sacrifice":
        # "Whenever you sacrifice another permanent/Eldrazi" (Gixian
        # Infiltrator, Writhing Chrysalis) -- shared.fire_sacrifice_triggers
        # queued it. permanent is the creature carrying the ability;
        # sacrificed_card_def is what was sacrificed.
        permanent = entry["permanent"]
        sacrificed_card_def = entry["sacrificed_card_def"]

        def resolve(state, card_def):
            hook = registry.EFFECT_REGISTRY.get(card_def.effect_id, {}).get("on_sacrifice")
            if hook is not None:
                hook(state, permanent, sacrificed_card_def)
        return resolve
    if entry["type"] == "automatic":
        if entry["kind"] == "on_draw_count":
            def resolve(state, card_def):
                # The card can leave the graveyard (exile, reanimation)
                # between queueing and resolving; fizzle gracefully instead
                # of crashing on graveyard.remove.
                if card_def not in state.graveyard:
                    return
                state.graveyard.remove(card_def)
                casting.enters_battlefield(state, card_def, force_tapped=True, from_zone="graveyard")
            return resolve
        raise ValueError(f"unknown automatic trigger queue entry: {entry}")
    if entry["type"] == "decision":
        if entry["kind"] == "madness":
            def resolve(state, card_def):
                resolution.begin_madness_decision(state, card_def, on_complete=lambda s: None)
            return resolve
        raise ValueError(f"unknown trigger queue entry: {entry}")
    raise ValueError(f"unknown trigger queue entry: {entry}")


def _promotion_targets(entry):
    """A queued ETB or LTB triggered ability that chooses targets. Its
    targets are picked when the ability goes on the stack (603.3d), so its
    hook runs at promotion -- opening target selection and pushing its own
    effect entry -- rather than at resolution like a non-targeting
    trigger. Gated by a registry "etb_targets"/"ltb_targets" flag."""
    if entry["type"] not in ("etb", "ltb"):
        return False
    key = "etb_targets" if entry["type"] == "etb" else "ltb_targets"
    return registry.EFFECT_REGISTRY.get(entry["card_def"].effect_id, {}).get(key, False)


def promote_triggers_to_stack(state):
    """Moves every currently-queued trigger, for every player, onto
    state.stack. Called once per priority round, right before anyone would
    receive priority (704.3: SBAs, then triggers to the stack, then
    priority).

    Reads each player's own trigger_queue, not just the active-player
    proxy: most producers fire because of the active player's own action,
    but a state-based death's LTB trigger (state_based._queue_leave_
    triggers) can belong to the non-active player (their blocker died on
    the active player's own turn), so it writes into that true owner's
    queue directly.

    APNAP ordering (603.3b): each player's queue becomes its own group, the
    active player's group placed first (resolves last), the other player's
    second (resolves first) -- see _place_trigger_groups. A group only gets
    an ordering decision (begin_order_triggers) when that owner has 2+
    simultaneous triggers -- a player only ever orders their own. active_idx
    is set to each group's owner for the duration of its placement (so
    push_to_stack's controller stamp is correct) and restored after.

    A targeting ETB/LTB (_promotion_targets) is placed differently: instead
    of pushing a plain resolve, its own etb_trigger/ltb_trigger hook runs at
    placement, opening target selection. Mixed with plain triggers for the
    same owner, all go through one begin_order_triggers choice."""
    original_active_idx = state.active_idx
    player_indices = [original_active_idx]
    if len(state.players) > 1:
        player_indices.append(1 - original_active_idx)

    def _entry_for(e):
        if _promotion_targets(e):
            hook = "etb_trigger" if e["type"] == "etb" else "ltb_trigger"
            return {"card_def": e["card_def"], "permanent": e["permanent"], "targeting": True, "hook": hook}
        return {"card_def": e["card_def"], "resolve": _trigger_resolve(e)}

    groups = []
    for player_idx in player_indices:
        queue = state.players[player_idx].trigger_queue
        if not queue:
            continue
        # Mana-burn-penalty exemption signal (read by game.turn.
        # _empty_mana_pools): a trigger is hitting the stack this phase, so
        # any mana floated this phase isn't "for nothing" even if unspent.
        state.players[player_idx].triggers_fired_this_phase = True
        groups.append((player_idx, [_entry_for(e) for e in queue]))
        queue.clear()

    _place_trigger_groups(state, groups, original_active_idx)


def _place_trigger_groups(state, groups, original_active_idx):
    """Places `groups` (APNAP-ordered (owner_idx, entries) pairs) one at a
    time, each under its own owner's active_idx, restoring
    original_active_idx once every group is placed. A group's 2+ entries
    chain through begin_order_triggers's on_complete rather than being
    placed in one synchronous pass, since only one pending_resolution can
    be open at a time.

    A single-entry group's targeting placement needs the same chaining:
    its hook can itself open a fresh pending_resolution that's still
    unanswered when the hook call returns. Proceeding straight to `rest`/
    restoring active_idx then would stomp active_idx out from under that
    still-open decision (whose owner can differ from original_active_idx)
    and silently misdirect it to the wrong player's board -- so this
    chains onto that resolution's own on_complete too, same as the
    2+-entries branch below."""
    if not groups:
        state.active_idx = original_active_idx
        return
    (owner_idx, entries), rest = groups[0], groups[1:]
    state.active_idx = owner_idx
    if len(entries) == 1:
        # Same placement logic execute_order_triggers_option (handlers_
        # triggers.py) uses for the 2+ case below -- inlined here rather
        # than shared, to avoid an import cycle back into effects.triggers.
        entry = entries[0]
        if entry.get("targeting"):
            registry.EFFECT_REGISTRY[entry["card_def"].effect_id][entry["hook"]](state, entry["permanent"])
            if state.pending_resolution is not None:
                # A real target existed and the hook's own decision is still
                # open (a target-less hook auto-completes synchronously,
                # leaving pending_resolution None). Defer `rest` until that
                # decision finishes, chaining onto its own on_complete.
                inner_on_complete = state.pending_resolution["on_complete"]

                def _continue(s, *args):
                    inner_on_complete(s, *args)
                    _place_trigger_groups(s, rest, original_active_idx)

                state.pending_resolution["on_complete"] = _continue
                return
        else:
            push_to_stack(state, entry["card_def"], entry["resolve"], reserves_hand_card=False, is_spell=False)  # a triggered ability, not a spell
        _place_trigger_groups(state, rest, original_active_idx)
    else:
        resolution.begin_order_triggers(
            state, entries, on_complete=lambda s: _place_trigger_groups(s, rest, original_active_idx))
