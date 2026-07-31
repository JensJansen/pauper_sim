"""The priority stack: push, pop-and-resolve, and the one cast-time trigger
hook. Deliberately dumb -- these functions never inspect what `resolve` does,
so this module doesn't depend on casting.py (even though cast_aura calls
push_to_stack). triggers.py sits ABOVE it instead: _trigger_resolve's
"automatic" branch needs casting.enters_battlefield, which put here would
recreate a cycle.

References registry.EFFECT_REGISTRY only inside function bodies -- see
game/registry.py's docstring for why."""

from .. import registry
from ..cards import CardType


def on_cast_trigger(state, card_def):
    """A "whenever you cast an instant/sorcery" ability (Guttersnipe). Real
    Magic 603.3: a triggered ability doesn't take effect when it triggers -- it
    goes on the stack at the next priority point, ABOVE the triggering spell,
    with a response window. So this only QUEUES it (state.trigger_queue);
    game.turn's priority round promotes it once the cast is fully done (never
    mid-cast, or it'd land under a spell not yet pushed). Every cast path
    (normal / alt_cast / Flashback / Plot) calls this identically."""
    if card_def.card_type not in (CardType.INSTANT, CardType.SORCERY):
        return
    for permanent in state.battlefield:
        trigger = registry.EFFECT_REGISTRY.get(permanent.card_def.effect_id, {}).get("on_cast")
        if trigger is not None:
            state.log_event(
                "trigger_fired", source=(permanent.card_def.name, permanent.slot), trigger_kind="on_cast",
                triggering_card=card_def.name,
            )
            state.trigger_queue.append({"type": "cast_trigger", "card_def": permanent.card_def, "permanent": permanent})


def push_to_stack(state, card_def, resolve, reserves_hand_card=True, is_spell=True, exiles_on_resolve=False, targets=()):
    """Defer `resolve(state, card_def)` onto state.stack instead of running it
    now, giving both players a priority window before it resolves. Pushed only
    once the spell's cost is fully paid -- never mid-payment (an alt cost that
    is itself a resolution, e.g. Fireblast's sacrifice-2-Mountains, pushes from
    that resolution's own on_complete).

    targets: this entry's DECLARED targets (real Magic: public the instant
    they're chosen, 601.2c -- stays visible even if one later becomes illegal
    and the spell fizzles at resolution, so this is never re-derived from
    target_still_legal). A tuple of tagged descriptors, one per target chosen
    at cast/activation time (empty for the common non-targeted push -- lands,
    vanilla creatures, untargeted triggers):
      ("player", seat_idx)              -- capture_any_target's player case
      ("creature", permanent)           -- capture_any_target's permanent case
                                            (any permanent type, not just
                                            creatures -- e.g. Cleansing
                                            Wildfire's land target; the tag
                                            name is capture_any_target's own)
      ("graveyard_card", card_instance) -- begin_choose_up_to_graveyard's
                                            picks (Rooftop Percher)
      ("stack_entry", other_entry)      -- begin_choose_stack_target's pick
                                            (Counterspell/Dispel/Spell Pierce)
    Consumed by rl.features.build_token_set/_stack_target_map to surface
    "what is currently being targeted" to the agent -- never read by engine
    resolution logic itself (each resolve closure already carries its own
    captured target(s) independently; this is purely an observation-side
    mirror of the same data).

    A pushed card stays physically in its origin zone until `resolve` moves it
    (resolve does its own hand/graveyard/exile removal, not here) -- so a paid-
    but-unresolved card is still "in hand." Two places must NOT count it as
    available: drl_env._hand_count_available (cast legality) and
    resolution.discard_options (instant-speed discard) -- both subtract
    same-named stack entries still reserving a hand card.

    reserves_hand_card=False when the card is NOT awaiting removal from the
    caster's hand: Flashback/reanimate (already out of the graveyard),
    Plot/Adventure/Madness cast-from-exile (already out of exile), an alt cost
    that discards eagerly (Fireblast, Crop Rotation), or a promoted trigger.
    (Confirmed live: two same-named Madness cards discarded back-to-back --
    default True miscounted the first's promoted entry as reserving the second,
    still-in-hand copy, leaving zero legal actions.)

    Records active_idx as the entry's controller: a priority round can flip
    active_idx before this resolves, but resolve must run against the CASTER's
    zones (state.py's active_idx proxy) -- resolve_top_of_stack restores it."""
    state.stack.append({
        "card_def": card_def, "resolve": resolve, "controller": state.active_idx,
        "reserves_hand_card": reserves_hand_card, "is_spell": is_spell,
        # Flashback (702.34): the spell is EXILED as it leaves the stack, not put
        # into the graveyard. resolve_top_of_stack does that exile (+ logs it), so
        # every Flashback push sets this instead of leaving the card untracked.
        "exiles_on_resolve": exiles_on_resolve,
        "targets": tuple(targets),
    })
    # A normally-cast spell LEAVES its controller's hand the instant it goes on
    # the stack (real Magic: a cast spell is a stack object, not a hand card),
    # and NEVER re-enters hand. Removing it HERE -- at cast, not at resolution --
    # is what prevents it being cast a SECOND time while the first copy still
    # sits on the stack (the reserved-hand-card double-cast bug). Its resolve's
    # own "send this card onward" step recognises it as the resolving spell (via
    # state.resolving_card, set by resolve_top_of_stack) instead of expecting it
    # in hand. reserves_hand_card=False (flashback/madness/plot/alt-cost) has
    # already left its origin zone, so there is nothing to remove.
    if reserves_hand_card and card_def in state.hand:
        state.hand.remove(card_def)
    # Only a SPELL is a card that goes on the stack. An activated/triggered
    # ability (is_spell=False) puts an ability object on the stack while its
    # source card stays where it is -- NO card changes zones -- so it must not
    # be logged as a card zone_move (logging one made the replay converter mint
    # a phantom library copy for every activation: Makeshift Munitions,
    # Krark-Clan Shaman, Nihil Spellbomb, ...). For a spell, from_zone is "hand"
    # only when it actually just left hand here; a spell cast from graveyard/
    # exile (Flashback/Madness/Plot/free alt-cost, reserves_hand_card=False)
    # already left its origin zone before this push.
    if is_spell:
        state.log_event(
            "zone_move", card=card_def.name, from_zone="hand" if reserves_hand_card else None,
            to_zone="stack", controller=state.active_idx,
        )


def push_ability_to_stack(state, source_card_def, effect):
    """Put a NON-MANA ability's effect on the stack (real Magic 605: activated/
    triggered abilities use the stack + a priority window; only mana abilities
    skip it). `source_card_def` labels the entry; `effect(state)` runs on
    resolution. Costs and targets are already paid/chosen before this push.
    reserves_hand_card=False (source is on the battlefield, not a hand card);
    is_spell=False (an ability isn't a legal Counterspell/Dispel/Spell Pierce
    target)."""
    push_to_stack(state, source_card_def, lambda st, cd: effect(st), reserves_hand_card=False, is_spell=False)


def counter_spell(state, entry):
    """Counter a spell: remove its entry from the stack so it never resolves,
    and send its card to the right zone. A normally-cast spell
    (reserves_hand_card) already left its controller's hand at cast
    (push_to_stack), so this just sends the on-stack card to that controller's
    graveyard (real Magic: a countered spell goes to its owner's graveyard); the
    `if cd in controller.hand` guard below is now only defensive.
    A spell cast from elsewhere (flashback/madness/plot/free-alt,
    reserves_hand_card=False) has already left its origin zone at cast time,
    so there's nothing to move -- it simply ceases (a flashback spell would
    be exiled either way; a self-discarding spell like Crop Rotation is
    already in the graveyard, which is exactly where a countered spell goes).
    No-op if the entry has already left the stack (already countered/resolved
    in response). Returns True iff it actually countered something."""
    if entry not in state.stack:
        return False
    state.stack.remove(entry)
    cd = entry["card_def"]
    if entry.get("reserves_hand_card"):
        controller = state.players[entry["controller"]]
        if cd in controller.hand:
            controller.hand.remove(cd)
        state.move_card(cd, controller.graveyard)
    state.log_event("countered", card=cd.name, controller=entry["controller"])
    return True


def resolve_top_of_stack(state):
    """Pop and resolve the most recently pushed spell -- LIFO, no
    reordering action needed (real Magic's own stack order). Called once
    per "Pass" while state.stack is non-empty (game.turn._run_turn_gen),
    never automatically -- the model must explicitly let it happen instead
    of casting something else in response.

    Restores active_idx to this entry's own controller (push_to_stack)
    before resolving: by the time all players have passed in a row,
    active_idx may be sitting on whoever passed last, not the original
    caster -- resolve must run from the
    controller's own zone perspective regardless."""
    entry = state.stack.pop()
    state.active_idx = entry["controller"]
    # Only a spell is a card leaving the stack on resolution; an ability
    # resolving moves no card (same is_spell reasoning as push_to_stack). A
    # Flashback spell is EXILED as it leaves the stack (702.34), not put into the
    # graveyard -- log that explicit destination (the exile itself happens below,
    # after the effect resolves), so the replay shows exile, not the graveyard.
    if entry["is_spell"]:
        if entry.get("exiles_on_resolve"):
            state.log_event("zone_move", card=entry["card_def"].name, from_zone="stack", to_zone="exile", reason="flashback")
        else:
            state.log_event("zone_move", card=entry["card_def"].name, from_zone="stack", reason="resolve")
    # Mark the spell currently resolving so its resolve's own "send this card
    # onward" step (discard_from_hand_to_graveyard / cast_permanent_from_hand /
    # cast_aura) treats it as the resolving spell -- off hand since cast
    # (push_to_stack) and NEVER to re-enter hand -- rather than expecting it in
    # hand. Save/restore (not just clear) so a nested resolution, if one ever
    # happens, can't strand the outer spell's marker.
    prev_resolving = state.resolving_card
    state.resolving_card = entry["card_def"]
    try:
        entry["resolve"](state, entry["card_def"])
    finally:
        state.resolving_card = prev_resolving
    # Flashback exile (702.34): the card was removed from the graveyard at cast
    # and its resolve never re-homes it, so it is now gone from every zone -- it
    # cannot be flashed back again (out of the graveyard) and stays out of the
    # graveyard for good. state.exile here is the CASTABLE exile zone (Plot/
    # Madness/Adventure, stored as (card_def, turn) tuples) -- a flashback card is
    # NOT castable-from-exile, so it must not be added there; leaving it untracked
    # is the faithful representation given no plain-exile zone exists. The
    # stack->exile zone_move logged above is what makes the replay render it as
    # exiled rather than defaulting it to the graveyard.
