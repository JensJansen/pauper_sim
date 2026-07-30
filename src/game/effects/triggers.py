"""The trigger queue: moving a queued trigger (Sneaky Snacker's automatic
return, a Madness decision, an ETB/LTB/upkeep/venture/Ward/cast ability) onto
the priority stack. Sits ABOVE casting.py and stack.py -- _trigger_resolve's
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
                # crashing on graveyard.remove -- confirmed the hard way, a
                # cross-deck league game ~2500 games in raced exactly this.
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


def _etb_targets(entry):
    """A queued ETB triggered ability that CHOOSES TARGETS. Its targets are
    picked when the ability goes on the stack (603.3d), so its hook runs at
    PROMOTION -- opening target selection and pushing its own single effect
    entry, which fizzles per-target at resolution (608.2b/c) -- rather than at
    resolution like a non-targeting ETB. Owner directive (see
    project_targeted_triggered_abilities): every "enters and targets" effect
    works this way (Rooftop Percher, Pinnacle Kill-Ship)."""
    if entry["type"] != "etb":
        return False
    return registry.EFFECT_REGISTRY.get(entry["card_def"].effect_id, {}).get("etb_targets", False)


def promote_triggers_to_stack(state):
    """Moves every currently-queued trigger -- for EVERY player, not just
    the active one -- onto state.stack, replacing the old
    drain_trigger_queue (which ran each entry's own effect immediately
    instead of deferring it onto the stack -- see _trigger_resolve for
    what changed per trigger kind). Called once per priority round, right
    before anyone would receive priority (game.turn's own priority round),
    matching real Magic's actual ordering (704.3: state-based actions,
    then triggers move to the stack, THEN priority is given).

    Reads each player's OWN trigger_queue (game.state.PlayerState.
    trigger_queue), not just state.trigger_queue (the active-player proxy).
    Most producers (ETB, upkeep, cast triggers, Madness, venture, Ward) only
    ever fire because of an action the CURRENTLY active player is taking, so
    writing through the proxy already lands them in the right owner's list.
    The one exception is a leaves-the-battlefield trigger from a state-based
    DEATH (game.effects.state_based._queue_leave_triggers): state-based
    checks scan BOTH battlefields every priority round regardless of whose
    turn it is, so the dying permanent's owner can be the NON-active player
    (their blocker died in combat, or a removal spell killed their creature,
    on the active player's own turn) -- _queue_leave_triggers writes into
    that true owner's list directly for exactly this reason (confirmed
    live: a real league game crashed drl_env's coverage guard because the
    active player was handed an order_triggers choice naming the opponent's
    Clockwork Percussionist).

    Real Magic's own APNAP ordering (603.3b) now actually matters given the
    above, so each player's queue becomes its own GROUP: the active
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

    A TARGETING ETB (_etb_targets) is placed like any other queued trigger,
    just via a different placement action: instead of pushing a plain
    resolve, its OWN etb_trigger hook runs AT PLACEMENT -- opening target
    selection and pushing its own effect entry (targets locked now, fizzle
    per-target at resolution, 608.2b/c). When 2+ triggers (targeting and/or
    plain, any mix) are queued together for the SAME owner, ALL of them go
    through ONE begin_order_triggers ordering choice (603.3b covers every
    simultaneous trigger of one player's, not just the non-targeting ones)
    -- see resolution.execute_order_triggers_option's own targeting branch
    for how a targeting entry gets placed."""
    original_active_idx = state.active_idx
    player_indices = [original_active_idx]
    if len(state.players) > 1:
        player_indices.append(1 - original_active_idx)

    def _entry_for(e):
        if _etb_targets(e):
            return {"card_def": e["card_def"], "permanent": e["permanent"], "targeting": True}
        return {"card_def": e["card_def"], "resolve": _trigger_resolve(e)}

    groups = []
    for player_idx in player_indices:
        queue = state.players[player_idx].trigger_queue
        if not queue:
            continue
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
    promote_triggers_to_stack when state.pending_resolution is None."""
    if not groups:
        state.active_idx = original_active_idx
        return
    (owner_idx, entries), rest = groups[0], groups[1:]
    state.active_idx = owner_idx
    if len(entries) == 1:
        # Same two-way placement branch execute_order_triggers_option uses for
        # the 2+ case below -- inlined here (not shared) for the same "can't
        # import across this pair without a cycle" reason state_based/
        # handlers duplicate their own leaves-battlefield three-liner.
        entry = entries[0]
        if entry.get("targeting"):
            registry.EFFECT_REGISTRY[entry["card_def"].effect_id]["etb_trigger"](state, entry["permanent"])
        else:
            push_to_stack(state, entry["card_def"], entry["resolve"], reserves_hand_card=False, is_spell=False)  # a triggered ability, not a spell
        _place_trigger_groups(state, rest, original_active_idx)
    else:
        resolution.begin_order_triggers(
            state, entries, on_complete=lambda s: _place_trigger_groups(s, rest, original_active_idx))


if __name__ == "__main__":
    # ponytail self-check: run via `python -m game.effects.triggers` from
    # src/. Per-turn draw counter + Sneaky Snacker-style automatic return
    # -- the scenario specific to THIS
    # module (_trigger_resolve's "automatic" branch + multi-trigger
    # ordering). The Madness "decision" branch is exercised together with
    # madness_and_plot.execute_madness_cast in effects/integration_check.py
    # instead, since that chain needs both modules working together.
    from .. import registry
    from ..cards import CardDef, CardType, EffectId
    from ..state import GameState
    from .stack import resolve_top_of_stack

    _filler_backup = registry.EFFECT_REGISTRY[EffectId.FILLER]
    registry.EFFECT_REGISTRY[EffectId.FILLER] = {"on_draw_count": {"count": 3}}
    try:
        snacker = CardDef("Fake Snacker", CardType.CREATURE, {"generic": 1}, EffectId.FILLER)
        state = GameState(on_the_play=True)
        state.library = [CardDef(f"Filler {i}", CardType.SORCERY, {}, None) for i in range(5)]
        state.graveyard = [snacker, snacker]  # two physical copies

        state.draw(1)
        assert state.cards_drawn_this_turn == 1 and state.trigger_queue == []
        state.draw(1)
        assert state.cards_drawn_this_turn == 2 and state.trigger_queue == []
        state.draw(1)  # the third card this turn -- both copies trigger
        assert state.cards_drawn_this_turn == 3
        assert len(state.trigger_queue) == 2
        assert all(e == {"type": "automatic", "kind": "on_draw_count", "card_def": snacker} for e in state.trigger_queue)

        state.draw(1)  # a 4th draw must NOT re-trigger (exactly == 3, not >= 3)
        assert len(state.trigger_queue) == 2

        # 2 simultaneous triggers -- a real placement-order choice
        #, not fixed queue order.
        promote_triggers_to_stack(state)
        assert state.pending_resolution["kind"] == "order_triggers"
        assert resolution.order_triggers_options(state) == ["Fake Snacker"]
        resolution.execute_order_triggers_option(state, "Fake Snacker")
        assert state.pending_resolution["kind"] == "order_triggers"  # one more still to place
        resolution.execute_order_triggers_option(state, "Fake Snacker")
        assert state.pending_resolution is None
        assert len(state.stack) == 2
        assert state.trigger_queue == []

        # No decision at any point once each stack entry resolves -- both
        # copies return to the battlefield tapped.
        while state.stack:
            resolve_top_of_stack(state)
        assert state.pending_resolution is None
        assert state.graveyard == []
        assert len(state.battlefield) == 2
        assert all(p.card_def is snacker and p.tapped for p in state.battlefield)
    finally:
        registry.EFFECT_REGISTRY[EffectId.FILLER] = _filler_backup

    print("triggers.py draw-counter self-check: OK")

    # 2 SIMULTANEOUS TARGETING ETBs (F8 generalization): both go through ONE
    # begin_order_triggers ordering choice -- no assert-len<=1 guard anymore --
    # each PLACED via its own etb_trigger hook (not a plain resolve). Placement
    # order is the active player's choice; LIFO means placed-LAST resolves-FIRST.
    from ..state import Permanent

    etb_calls = []

    def _fake_targeting_etb(tag):
        def hook(state, permanent):
            etb_calls.append(f"{tag}-placed")
            push_to_stack(
                state, permanent.card_def,
                lambda s, cd: etb_calls.append(f"{tag}-resolved"),
                reserves_hand_card=False, is_spell=False,
            )
        return hook

    _filler_backup2 = registry.EFFECT_REGISTRY[EffectId.FILLER]
    _ent_backup = registry.EFFECT_REGISTRY[EffectId.GENEROUS_ENT]
    registry.EFFECT_REGISTRY[EffectId.FILLER] = {"etb_trigger": _fake_targeting_etb("A"), "etb_targets": True}
    registry.EFFECT_REGISTRY[EffectId.GENEROUS_ENT] = {"etb_trigger": _fake_targeting_etb("B"), "etb_targets": True}
    try:
        card_a = CardDef("Targeting A", CardType.CREATURE, None, EffectId.FILLER)
        card_b = CardDef("Targeting B", CardType.CREATURE, None, EffectId.GENEROUS_ENT)
        perm_a = Permanent(card_a)
        perm_b = Permanent(card_b)
        state = GameState(on_the_play=True)
        state.battlefield = [perm_a, perm_b]
        state.trigger_queue = [
            {"type": "etb", "card_def": card_a, "permanent": perm_a},
            {"type": "etb", "card_def": card_b, "permanent": perm_b},
        ]

        promote_triggers_to_stack(state)
        assert state.pending_resolution["kind"] == "order_triggers"
        assert resolution.order_triggers_options(state) == ["Targeting A", "Targeting B"]
        assert etb_calls == []  # neither hook has run yet -- only PLACEMENT runs a targeting hook

        resolution.execute_order_triggers_option(state, "Targeting A")  # placed FIRST -- resolves LAST
        assert etb_calls == ["A-placed"]
        assert len(state.stack) == 1
        assert state.pending_resolution["kind"] == "order_triggers"  # one more still to place

        resolution.execute_order_triggers_option(state, "Targeting B")  # placed LAST -- resolves FIRST
        assert etb_calls == ["A-placed", "B-placed"]
        assert len(state.stack) == 2
        assert state.pending_resolution is None

        while state.stack:
            resolve_top_of_stack(state)
        assert etb_calls == ["A-placed", "B-placed", "B-resolved", "A-resolved"]  # LIFO: B (placed last) resolves first
    finally:
        registry.EFFECT_REGISTRY[EffectId.FILLER] = _filler_backup2
        registry.EFFECT_REGISTRY[EffectId.GENEROUS_ENT] = _ent_backup

    print("triggers.py 2 simultaneous targeting-ETB triggers (F8 generalization) self-check: OK")

    # Cross-owner LTB trigger (the actual bug this session found): a
    # NON-active player's creature dying via state-based action (combat,
    # removal -- check_state_based_actions scans BOTH battlefields every
    # round) must queue into THAT player's own trigger_queue, not the
    # active player's proxy -- and promote_triggers_to_stack must place each
    # owner's group under ITS OWN active_idx (APNAP, 603.3b): the active
    # player's single trigger placed first (resolves last), the opponent's
    # single trigger placed second (resolves first), each stack entry
    # controller-stamped to its TRUE owner, active_idx restored to the real
    # active player once both groups are placed.
    from ..state import PlayerState
    from . import state_based

    ltb_fired = []
    _filler_backup3 = registry.EFFECT_REGISTRY[EffectId.FILLER]
    _ent_backup2 = registry.EFFECT_REGISTRY[EffectId.GENEROUS_ENT]
    registry.EFFECT_REGISTRY[EffectId.FILLER] = {"ltb_trigger": lambda s, p: ltb_fired.append(p.card_def.name)}
    registry.EFFECT_REGISTRY[EffectId.GENEROUS_ENT] = {"ltb_trigger": lambda s, p: ltb_fired.append(p.card_def.name)}
    _card_defs_backup2 = dict(registry.CARD_DEFS)
    try:
        mine_def = CardDef("Mine Dies", CardType.CREATURE, None, EffectId.FILLER, power=1, toughness=1)
        theirs_def = CardDef("Theirs Dies", CardType.CREATURE, None, EffectId.GENEROUS_ENT, power=1, toughness=1)
        registry.CARD_DEFS["Mine Dies"] = mine_def
        registry.CARD_DEFS["Theirs Dies"] = theirs_def
        mine = Permanent(mine_def)
        theirs = Permanent(theirs_def)
        state = GameState(on_the_play=True, players=[PlayerState(True), PlayerState(False)])
        state.active_idx = 0
        state.players[0].battlefield = [mine]
        state.players[1].battlefield = [theirs]
        mine.damage_marked = 1  # lethal for both -- simultaneous SBA deaths, cross-owner
        theirs.damage_marked = 1
        state_based.check_state_based_actions(state)
        # Each landed in its OWN owner's queue, never the active-player proxy.
        assert [e["card_def"].name for e in state.players[0].trigger_queue] == ["Mine Dies"]
        assert [e["card_def"].name for e in state.players[1].trigger_queue] == ["Theirs Dies"]

        promote_triggers_to_stack(state)
        assert state.pending_resolution is None  # 1 entry per owner -- no ordering decision needed by either
        assert len(state.stack) == 2
        assert state.stack[0]["card_def"].name == "Mine Dies" and state.stack[0]["controller"] == 0  # active player's own -- placed first (deepest)
        assert state.stack[1]["card_def"].name == "Theirs Dies" and state.stack[1]["controller"] == 1  # opponent's -- placed second (on top)
        assert state.active_idx == 0  # restored to the true active player, not left on the opponent

        while state.stack:  # LIFO: the opponent's (placed last) resolves FIRST
            resolve_top_of_stack(state)
        assert ltb_fired == ["Theirs Dies", "Mine Dies"]
    finally:
        registry.EFFECT_REGISTRY[EffectId.FILLER] = _filler_backup3
        registry.EFFECT_REGISTRY[EffectId.GENEROUS_ENT] = _ent_backup2
        registry.CARD_DEFS.clear()
        registry.CARD_DEFS.update(_card_defs_backup2)

    print("triggers.py cross-owner LTB trigger (APNAP, per-owner queues) self-check: OK")

    # The actual crash scenario: the OPPONENT has 2+ simultaneous triggers of
    # their OWN -- order_triggers must be answered by THAT player (active_idx
    # set to them), offering ONLY their own card names, never the active
    # player's -- this is what keeps order_triggers a safe by-name resolution
    # (drl_env._actions._CHOOSE_NAME_PENDING_KINDS's own guaranteed invariant:
    # every candidate it can ever offer is confined to the deciding player's
    # own deck).
    _filler_backup4 = registry.EFFECT_REGISTRY[EffectId.FILLER]
    _ent_backup3 = registry.EFFECT_REGISTRY[EffectId.GENEROUS_ENT]
    registry.EFFECT_REGISTRY[EffectId.FILLER] = {"ltb_trigger": lambda s, p: None}
    registry.EFFECT_REGISTRY[EffectId.GENEROUS_ENT] = {"ltb_trigger": lambda s, p: None}
    _card_defs_backup3 = dict(registry.CARD_DEFS)
    try:
        opp_a_def = CardDef("Opp Dies A", CardType.CREATURE, None, EffectId.FILLER, power=1, toughness=1)
        opp_b_def = CardDef("Opp Dies B", CardType.CREATURE, None, EffectId.GENEROUS_ENT, power=1, toughness=1)
        registry.CARD_DEFS["Opp Dies A"] = opp_a_def
        registry.CARD_DEFS["Opp Dies B"] = opp_b_def
        opp_a = Permanent(opp_a_def)
        opp_b = Permanent(opp_b_def)
        state = GameState(on_the_play=True, players=[PlayerState(True), PlayerState(False)])
        state.active_idx = 0
        state.players[1].battlefield = [opp_a, opp_b]
        opp_a.damage_marked = 1
        opp_b.damage_marked = 1
        state_based.check_state_based_actions(state)

        promote_triggers_to_stack(state)
        assert state.pending_resolution["kind"] == "order_triggers"
        assert state.active_idx == 1  # the OPPONENT orders these -- never the active player
        assert resolution.order_triggers_options(state) == ["Opp Dies A", "Opp Dies B"]  # only THEIR own names

        resolution.execute_order_triggers_option(state, "Opp Dies A")
        assert state.active_idx == 1  # still theirs to place the second one
        resolution.execute_order_triggers_option(state, "Opp Dies B")
        assert state.pending_resolution is None
        assert state.active_idx == 0  # restored once every group (here, just the one) is placed
        assert all(e["controller"] == 1 for e in state.stack)
    finally:
        registry.EFFECT_REGISTRY[EffectId.FILLER] = _filler_backup4
        registry.EFFECT_REGISTRY[EffectId.GENEROUS_ENT] = _ent_backup3
        registry.CARD_DEFS.clear()
        registry.CARD_DEFS.update(_card_defs_backup3)

    print("triggers.py opponent-owned order_triggers (APNAP: only their own names) self-check: OK")
