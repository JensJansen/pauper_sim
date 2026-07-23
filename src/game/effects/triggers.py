"""The trigger queue: moving a queued trigger (Sneaky Snacker's automatic
return, a Madness decision) onto the real priority stack. Sits above both
casting.py and stack.py -- _trigger_resolve's "automatic" branch needs
casting.enters_battlefield (an automatic trigger can put a card back onto
the battlefield), so this module can't live underneath casting.py the way
stack.py does. See casting.py's own module docstring for why that
dependency has to point this direction."""

from . import casting
from .stack import push_to_stack
from .. import resolution


def _trigger_resolve(entry):
    """Builds the stack entry's own resolve(state, card_def) function for
    one queued trigger (docs/PRIORITY_PLAN.md item 1) -- deferred until
    THIS SPECIFIC stack entry actually resolves, instead of running the
    instant the trigger was queued (real Magic: triggered abilities go on
    the stack and can be responded to, same as a cast spell).

    "automatic" (Sneaky Snacker's on-draw-count return): runs the exact
    effect this engine always ran immediately before this plan.
    "decision" (Madness): now OPENS the cast-or-decline choice only here,
    matching real Magic's "you may cast it as this ability resolves"
    wording -- not the instant the card was discarded, which is what lets
    an opponent get a real priority window (a chance to respond to the
    trigger itself) before the decision is even offered. Its own
    on_complete is a no-op: the old recursive "keep draining" continuation
    isn't needed anymore -- promote_triggers_to_stack (below) is called
    fresh at the start of every priority round, so anything left queued
    (or queued anew) is picked up there instead of needing to be chained
    through by hand."""
    if entry["type"] == "automatic":
        if entry["kind"] == "on_draw_count":
            def resolve(state, card_def):
                state.graveyard.remove(card_def)
                casting.enters_battlefield(state, card_def, force_tapped=True)
            return resolve
        raise ValueError(f"unknown automatic trigger queue entry: {entry}")
    if entry["type"] == "decision":
        if entry["kind"] == "madness":
            def resolve(state, card_def):
                resolution.begin_madness_decision(state, card_def, on_complete=lambda s: None)
            return resolve
        raise ValueError(f"unknown trigger queue entry: {entry}")
    raise ValueError(f"unknown trigger queue entry: {entry}")


def promote_triggers_to_stack(state):
    """Moves every currently-queued trigger for the active player onto
    state.stack (docs/PRIORITY_PLAN.md item 1), replacing the old
    drain_trigger_queue (which ran each entry's own effect immediately
    instead of deferring it onto the stack -- see _trigger_resolve for
    what changed per trigger kind). Called once per priority round, right
    before anyone would receive priority (game.turn's own priority round),
    matching real Magic's actual ordering (704.3: state-based actions,
    then triggers move to the stack, THEN priority is given).

    Only ever looks at state.trigger_queue (the ACTIVE player's own,
    active-player-proxied) -- callers always invoke this with
    state.active_idx == state.turn_player_idx (priority always resets
    there before this runs), and nothing in the current card pool ever
    queues a trigger for a non-active player (only the active player's
    own draw()/discard() ever populate trigger_queue), so real Magic's own
    APNAP ordering (whose triggers get placed first, when different
    players have simultaneous ones) is moot given what this engine can
    actually produce today -- revisit if a future card changes that.

    2+ queued at once: the active player picks the placement order
    (resolution.begin_order_triggers) -- real Magic's own rule (603.3b),
    not a fixed queue order (a real deck can hit this: Faithless
    Looting's discard-2 landing on two Madness cards at once, or two
    Sneaky Snackers both crossing their own draw-count trigger on the
    same draw). 0 or 1: pushed immediately, no ordering decision needed.
    No-op if the queue is empty -- safe to call unconditionally at the
    start of every priority round."""
    if not state.trigger_queue:
        return
    stack_entries = [{"card_def": entry["card_def"], "resolve": _trigger_resolve(entry)} for entry in state.trigger_queue]
    state.trigger_queue.clear()
    if len(stack_entries) == 1:
        entry = stack_entries[0]
        push_to_stack(state, entry["card_def"], entry["resolve"], reserves_hand_card=False)
        return
    resolution.begin_order_triggers(state, stack_entries, on_complete=lambda s: None)


if __name__ == "__main__":
    # ponytail self-check: run via `python -m game.effects.triggers` from
    # src/. Per-turn draw counter + Sneaky Snacker-style automatic return
    # (MADNESS_DECKS_PLAN.md item 7) -- the scenario specific to THIS
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
        # (docs/PRIORITY_PLAN.md item 1), not fixed queue order.
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
