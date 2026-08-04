"""The trigger queue: moving a queued trigger (Sneaky Snacker's automatic
return, a Madness decision, an ETB/LTB/upkeep/venture/Ward/cast/sacrifice
ability) onto the priority stack. Sits ABOVE casting.py and stack.py -- _trigger_resolve's
"automatic" branch needs casting.enters_battlefield, so it can't live under
casting.py the way stack.py does (see casting.py's docstring)."""

from . import casting
from .stack import counter_spell, push_to_stack
from .. import registry, resolution


def _trigger_resolve(entry):
    """Build the resolve(state, card_def) for one queued trigger, deferred
    until THIS stack entry resolves (real Magic: a triggered ability goes on
    the stack and can be responded to, like a spell). Branches on entry["type"]:

    "automatic" (Sneaky Snacker's on-draw return): runs the effect directly.
    "decision" (Madness): opens the cast-or-decline choice only here, matching
    "you may cast it as this ability resolves" -- so an opponent gets a
    priority window on the trigger before the choice is offered. on_complete is
    a no-op; promote_triggers_to_stack runs fresh each priority round, so
    anything still (or newly) queued is picked up there."""
    if entry["type"] == "etb":
        # An enters-the-battlefield triggered ability (casting.enters_
        # battlefield queued it). The registry etb_trigger hook takes
        # (state, permanent) -- the permanent that entered, carried on the
        # entry so an ETB that needs its own source (Mesmeric Fiend) can
        # reach it. Looked up fresh at resolution (not captured at queue
        # time) so a temporarily-reassigned FILLER registry entry in a self-
        # check still resolves to whatever's live now, same lazy-lookup
        # convention the rest of this engine uses. Fizzles gracefully to a
        # no-op if the hook is somehow gone (belt-and-suspenders --
        # enters_battlefield only queues this when the hook exists).
        permanent = entry["permanent"]

        def resolve(state, card_def):
            etb = registry.EFFECT_REGISTRY.get(card_def.effect_id, {}).get("etb_trigger")
            if etb is not None:
                etb(state, permanent)
        return resolve
    if entry["type"] == "ltb":
        # A leaves-the-battlefield triggered ability (queue_leave_triggers
        # queued it when the permanent left). The registry ltb_trigger hook
        # takes (state, permanent) -- the permanent that LEFT (already off
        # the battlefield, but still the object carrying whatever the trigger
        # needs, e.g. Mesmeric Fiend's linked exiled card on its flags). Same
        # fresh lazy lookup + graceful no-op as the etb branch above.
        permanent = entry["permanent"]

        def resolve(state, card_def):
            ltb = registry.EFFECT_REGISTRY.get(card_def.effect_id, {}).get("ltb_trigger")
            if ltb is not None:
                ltb(state, permanent)
        return resolve
    if entry["type"] == "upkeep":
        # "At the beginning of your upkeep, ..." (Delver of Secrets), queued by
        # game.turn.upkeep_step. Same (state, permanent) hook shape + fresh
        # lazy lookup as etb/ltb above.
        permanent = entry["permanent"]

        def resolve(state, card_def):
            upkeep = registry.EFFECT_REGISTRY.get(card_def.effect_id, {}).get("upkeep_trigger")
            if upkeep is not None:
                upkeep(state, permanent)
        return resolve
    if entry["type"] == "venture":
        # "Venture into Undercity" (The Initiative / Avenging Hunter), queued by
        # undercity.queue_venture (on taking the initiative or at upkeep). The
        # player is carried on the entry; resolving it advances their dungeon.
        # Lazy import -- undercity pulls in casting/tokens, above this module.
        player_idx = entry["player_idx"]

        def resolve(state, card_def):
            from . import undercity
            undercity.venture(state, player_idx)
        return resolve
    if entry["type"] == "ward":
        # Ward (Tolarian Terror): queued by casting._maybe_trigger_ward when an
        # opponent targets the Warded creature. It's promoted onto the stack
        # ABOVE the triggering spell (queued before that spell's push, promoted
        # in the next priority round), so when it resolves the triggering spell
        # is the entry just below -- state.stack[-1]. The targeting player
        # (payer_idx) pays the ward cost (begin_pay_unless) or that spell/
        # ability is countered. Fizzles gracefully if the spell already left
        # the stack (countered in response) -- nothing to counter.
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
        # A "whenever you cast an instant/sorcery" ability (Guttersnipe --
        # stack.on_cast_trigger queued it). Its own source Permanent is
        # carried on the entry so the hook can act from it (the on_cast
        # hook's signature is (state, permanent), unlike etb's state-only
        # one); the hook is looked up fresh at resolution, same reason as
        # the etb branch above.
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
        # sacrificed_card_def is what was sacrificed. Same fresh lazy lookup
        # + graceful no-op as the etb/ltb/upkeep branches above -- if
        # `permanent` has since left the battlefield itself, the registry
        # hook still runs (a non-targeted effect, so no fizzle), it just has
        # no observable effect on an object no longer in the game.
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
                # The card can LEAVE the graveyard between this trigger
                # being queued (at draw time) and resolving here on the
                # stack -- an opponent exiling it with graveyard hate
                # (monster_tron's Relic of Progenitus, spy_combo's
                # graveyard effects, ...) during the intervening priority
                # window, or another effect reanimating it. Real Magic:
                # the return trigger then simply does nothing (the object
                # it would move is gone). Fizzle gracefully instead of
                # crashing on graveyard.remove -- this race is reachable in
                # real play, not just hypothetical.
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
    """A queued ETB or LTB triggered ability that CHOOSES TARGETS. Its
    targets are picked when the ability goes on the stack (603.3d), so its
    hook runs at PROMOTION -- opening target selection and pushing its own
    single effect entry, which fizzles per-target at resolution (608.2b/c)
    -- rather than at resolution like a non-targeting trigger. Every
    "enters/leaves and targets" effect works this way (Rooftop Percher,
    Pinnacle Kill-Ship, Sewer-veillance Cam's ETB *and* LTB "tap or untap
    target creature" halves). Gated by a registry "etb_targets"/
    "ltb_targets" flag, one per trigger kind (a card could in principle
    target on one half and not the other)."""
    if entry["type"] not in ("etb", "ltb"):
        return False
    key = "etb_targets" if entry["type"] == "etb" else "ltb_targets"
    return registry.EFFECT_REGISTRY.get(entry["card_def"].effect_id, {}).get(key, False)


def promote_triggers_to_stack(state):
    """Moves every currently-queued trigger -- for EVERY player, not just
    the active one -- onto state.stack, deferred there rather than run
    immediately (see _trigger_resolve for how each trigger kind resolves).
    Called once per priority round, right before anyone would receive
    priority (game.turn's own priority round),
    matching real Magic's actual ordering (704.3: state-based actions,
    then triggers move to the stack, THEN priority is given).

    Reads each player's OWN trigger_queue (game.state.PlayerState.
    trigger_queue), not just state.trigger_queue (the active-player proxy).
    Most producers (ETB, upkeep, cast triggers, Madness, venture, Ward,
    sacrifice) only
    ever fire because of an action the CURRENTLY active player is taking, so
    writing through the proxy already lands them in the right owner's list.
    The one exception is a leaves-the-battlefield trigger from a state-based
    DEATH (game.effects.state_based._queue_leave_triggers): state-based
    checks scan BOTH battlefields every priority round regardless of whose
    turn it is, so the dying permanent's owner can be the NON-active player
    (their blocker died in combat, or a removal spell killed their creature,
    on the active player's own turn) -- _queue_leave_triggers writes into
    that true owner's list directly for exactly this reason (writing through
    the proxy instead would hand the active player an order_triggers choice
    naming a card from the opponent's own deck).

    Real Magic's own APNAP ordering (603.3b) matters given the above, so each
    player's queue becomes its own GROUP: the active
    player's group is placed FIRST (deepest on the stack, resolves LAST),
    then the other player's group is placed second (resolves FIRST) --
    see _place_trigger_groups. Each group gets its own placement-order
    decision (resolution.begin_order_triggers) only when THAT SAME player
    has 2+ simultaneous triggers -- a player only ever orders their own
    triggers, never an opponent's (this is what keeps order_triggers a
    plain by-name resolution: every candidate it can ever offer is
    guaranteed to be a card from the deciding player's own deck, the same
    guarantee drl_env._actions._CHOOSE_NAME_PENDING_KINDS documents and
    enforces). state.active_idx is set to each group's owner for the
    duration of its placement (push_to_stack's own "controller" stamp reads
    active_idx, same convention resolve_top_of_stack's controller-restore
    already established) and restored once every group has been placed.
    Both queues empty (by far the common case): no-op, safe to call
    unconditionally at the start of every priority round.

    A TARGETING ETB/LTB (_promotion_targets) is placed like any other queued
    trigger, just via a different placement action: instead of pushing a
    plain resolve, its OWN etb_trigger/ltb_trigger hook runs AT PLACEMENT --
    opening target selection and pushing its own effect entry (targets
    locked now, fizzle per-target at resolution, 608.2b/c). When 2+ triggers
    (targeting and/or plain, any mix) are queued together for the SAME
    owner, ALL of them go through ONE begin_order_triggers ordering choice
    (603.3b covers every simultaneous trigger of one player's, not just the
    non-targeting ones) -- see resolution.execute_order_triggers_option's
    own targeting branch for how a targeting entry gets placed."""
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
        # Dense mana-burn-penalty exemption signal (PlayerState.
        # triggers_fired_this_phase, read by game.turn._empty_mana_pools):
        # a real trigger is about to hit the stack for this player this
        # phase, so any mana they float this same phase can't be "for
        # nothing" even if it goes unspent (Writhing Chrysalis/Gixian
        # Infiltrator sacrificing a mana source purely for this trigger).
        state.players[player_idx].triggers_fired_this_phase = True
        groups.append((player_idx, [_entry_for(e) for e in queue]))
        queue.clear()

    _place_trigger_groups(state, groups, original_active_idx)


def _place_trigger_groups(state, groups, original_active_idx):
    """Places `groups` (a list of (owner_idx, entries) pairs, APNAP-ordered
    by promote_triggers_to_stack) one at a time, each under its OWN owner's
    active_idx, restoring original_active_idx once every group is placed.
    A group's own 2+ entries chain through begin_order_triggers's on_complete
    (begin_resolution's own docstring: "it may itself begin a further
    resolution") rather than being placed in one synchronous pass, since
    only ONE pending_resolution can be open at a time -- the next group's
    decision (if it needs one) must wait for this one to fully resolve, the
    same reason game.turn's own priority round only ever calls
    promote_triggers_to_stack when state.pending_resolution is None.

    A single-entry group's own TARGETING placement needs that identical
    wait: its hook (e.g. masked_vandal_etb) can itself open a fresh
    pending_resolution (a real target existed) that's still unanswered when
    the hook call returns -- begin_resolution is a flat, single-slot
    assignment (state.pending_resolution's own docstring), so proceeding
    straight to `rest`/restoring active_idx right then would stomp
    state.active_idx out from under that still-open decision (whose owner
    can be a DIFFERENT player than original_active_idx -- the entire reason
    groups are per-owner), and, since choose_opponent_permanent's own
    predicate re-reads state.opponent fresh from state.active_idx on every
    call, silently reassign BOTH who answers it and whose board is being
    offered. So this chains onto that resolution's own on_complete too,
    exactly like the 2+-entries branch below already does -- confirmed live
    via an all-False mask + wrong-seat/wrong-board crash in real cross-deck
    league play (a non-active-player group's lone targeting ETB opened
    choose_opponent_permanent, then active_idx got stomped back to
    original_active_idx before the agent ever answered it, so the NEXT
    query evaluated the wrong player's board as "the opponent")."""
    if not groups:
        state.active_idx = original_active_idx
        return
    (owner_idx, entries), rest = groups[0], groups[1:]
    state.active_idx = owner_idx
    if len(entries) == 1:
        # Same two-way placement branch execute_order_triggers_option (in
        # resolution/handlers.py) uses for the 2+ case below -- inlined here,
        # not shared, because triggers.py already imports resolution at
        # module level (this file's own top import), so handlers.py importing
        # a shared helper back out of effects.triggers would cycle.
        entry = entries[0]
        if entry.get("targeting"):
            registry.EFFECT_REGISTRY[entry["card_def"].effect_id][entry["hook"]](state, entry["permanent"])
            if state.pending_resolution is not None:
                # A real target existed and the hook's own decision is still
                # open (a target-less hook auto-completes synchronously via
                # complete_resolution before returning here, leaving
                # state.pending_resolution None -- see this function's own
                # docstring). Defer `rest` until THAT decision genuinely
                # finishes, chaining onto its own on_complete.
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
