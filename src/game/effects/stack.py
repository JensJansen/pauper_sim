"""The priority stack: push, pop-and-resolve, and the one cast-time trigger
hook. These functions never inspect what `resolve` does, so this module
doesn't depend on casting.py. triggers.py sits above it instead.

References registry.EFFECT_REGISTRY only inside function bodies -- see
game/registry.py's docstring for why."""

from .. import registry
from ..cards import CardType


def on_cast_trigger(state, card_def):
    """A "whenever you cast an instant/sorcery" ability (Guttersnipe).
    Queues the trigger (state.trigger_queue) rather than running it inline;
    game.turn's priority round promotes it above the triggering spell once
    the cast is fully done. Every cast path calls this identically."""
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
    """Defer `resolve(state, card_def)` onto state.stack, giving both
    players a priority window before it resolves. Pushed only once the
    spell's cost is fully paid.

    targets: this entry's declared targets, a tuple of tagged descriptors
    (empty for a non-targeted push):
      ("player", seat_idx)              -- capture_any_target's player case
      ("creature", permanent)           -- capture_any_target's permanent case
      ("graveyard_card", card_instance) -- begin_choose_up_to_graveyard's picks
      ("stack_entry", other_entry)      -- begin_choose_stack_target's pick
    Consumed by rl.model.features.build_token_set to surface "what is
    currently being targeted" to the agent; never read by resolution logic
    itself.

    A pushed card stays in its origin zone until `resolve` moves it, so a
    paid-but-unresolved card still reads as "in hand" -- drl_env.
    _hand_count_available and resolution.discard_options both subtract
    same-named stack entries still reserving a hand card.

    reserves_hand_card=False when the card isn't awaiting removal from the
    caster's hand: Flashback/reanimate, Plot/Adventure/Madness cast-from-
    exile, an eagerly-discarding alt cost (Fireblast, Crop Rotation), or a
    promoted trigger.

    Records active_idx as the entry's controller since a priority round can
    flip active_idx before this resolves; resolve_top_of_stack restores it."""
    state.stack.append({
        "card_def": card_def, "resolve": resolve, "controller": state.active_idx,
        "reserves_hand_card": reserves_hand_card, "is_spell": is_spell,
        # Flashback (702.34): exiled on resolve, not put into the graveyard.
        "exiles_on_resolve": exiles_on_resolve,
        "targets": tuple(targets),
    })
    # A cast spell leaves hand the instant it goes on the stack and never
    # re-enters it -- prevents casting a second copy while this one still
    # sits on the stack. reserves_hand_card=False has already left its
    # origin zone, so there's nothing to remove here.
    if reserves_hand_card and card_def in state.hand:
        state.hand.remove(card_def)
    # Only a spell is a card leaving hand for the stack; an ability
    # (is_spell=False) leaves its source card where it is -- no zone_move.
    if is_spell:
        state.log_event(
            "zone_move", card=card_def.name, from_zone="hand" if reserves_hand_card else None,
            to_zone="stack", controller=state.active_idx,
        )


def push_ability_to_stack(state, source_card_def, effect):
    """Put a non-mana ability's effect on the stack (mana abilities alone
    skip the stack). `source_card_def` labels the entry; `effect(state)`
    runs on resolution. Costs/targets are already paid/chosen. Never
    reserves a hand card (source is on the battlefield); is_spell=False
    (an ability isn't a legal counterspell target)."""
    push_to_stack(state, source_card_def, lambda st, cd: effect(st), reserves_hand_card=False, is_spell=False)


def counter_spell(state, entry):
    """Counter a spell: remove its entry from the stack so it never
    resolves, and send its card to its owner's graveyard. A normally-cast
    spell already left hand at cast, so this just moves the on-stack card
    to the graveyard (the `if cd in controller.hand` guard is defensive
    only). A spell cast from elsewhere has already left its origin zone,
    so there's nothing to move. No-op if the entry already left the stack.
    Returns True iff it actually countered something."""
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
    """Pop and resolve the most recently pushed spell (LIFO). Called once
    per "Pass" while state.stack is non-empty, never automatically.

    Restores active_idx to this entry's own controller before resolving,
    since by the time all players have passed in a row, active_idx may be
    sitting on whoever passed last."""
    entry = state.stack.pop()
    state.active_idx = entry["controller"]
    # Only a spell is a card leaving the stack; an ability resolving moves no
    # card. Flashback (702.34) exiles rather than going to the graveyard.
    if entry["is_spell"]:
        if entry.get("exiles_on_resolve"):
            state.log_event("zone_move", card=entry["card_def"].name, from_zone="stack", to_zone="exile", reason="flashback")
        else:
            state.log_event("zone_move", card=entry["card_def"].name, from_zone="stack", reason="resolve")
    # Mark the resolving spell so its own "send this card onward" step
    # treats it as such rather than expecting it in hand. Save/restore so a
    # nested resolution can't strand the outer spell's marker.
    prev_resolving = state.resolving_card
    state.resolving_card = entry["card_def"]
    try:
        entry["resolve"](state, entry["card_def"])
    finally:
        state.resolving_card = prev_resolving
    # Flashback exile: removed from the graveyard at cast and never re-homed
    # by resolve, so it's now out of every zone -- not added to the castable
    # exile zone (state.exile), since it can't be flashed back again.
