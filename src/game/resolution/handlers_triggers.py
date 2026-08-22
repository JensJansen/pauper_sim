"""Trigger placement ordering: when 2+ of one player's own triggers are ready
to move onto the stack at once, the player chooses the placement order
(603.3b). Re-exported via game.resolution so `from ..resolution import X`
in the catalogs keeps resolving."""

from .. import registry
from ._core import begin_resolution, complete_resolution


def begin_order_triggers(state, entries, on_complete):
    """2+ of the active player's own triggers are ready to move onto the
    stack at once -- real Magic lets that player choose the PLACEMENT order
    (603.3b), not a fixed queue order. Only a single player's own triggers;
    APNAP ordering BETWEEN players is handled one level up, by
    game.effects.triggers.promote_triggers_to_stack's group placement.

    entries: list of {"card_def", "resolve"} dicts (plain triggers, already
    stack-ready) or {"card_def", "permanent", "targeting": True, "hook"}
    dicts (target-at-promotion ETBs/LTBs -- placing one runs its own
    etb_trigger/ltb_trigger hook instead; see execute_order_triggers_
    option's targeting branch). Built by promote_triggers_to_stack.

    Picks one at a time; each pick is placed immediately
    (execute_order_triggers_option), not deferred to the end -- PLACEMENT
    order, not resolution order. Since the stack is LIFO, whichever PLAIN
    entry is placed LAST resolves FIRST. on_complete(state) once every entry
    has been placed."""
    begin_resolution(state, "order_triggers", on_complete, remaining=list(entries))


def order_triggers_options(state):
    return sorted({e["card_def"].name for e in state.pending_resolution["remaining"]})


def execute_order_triggers_option(state, name):
    """Places one still-`remaining` entry, then either advances to the next
    pick (this "order_triggers" resolution stays open) or completes once
    none are left.

    A targeting entry's hook can itself open a fresh pending_resolution,
    which REPLACES state.pending_resolution out from under `pending` (still
    a live reference to the detached dict). So _finish (below) is deferred,
    chained onto that fresh decision's own on_complete -- but ONLY when
    state.pending_resolution is genuinely a DIFFERENT object than `pending`
    afterward (an identity check, not just `is not None`): a target-less
    hook auto-completes (and clears) the fresh one synchronously before
    returning here, and a hook that opens no resolution at all leaves
    state.pending_resolution as this SAME `pending` dict, unchanged --
    either way there's no fresh decision to defer to, and _finish
    unconditionally reinstalls `pending` first regardless of which case
    this was."""
    pending = state.pending_resolution
    idx = next(i for i, e in enumerate(pending["remaining"]) if e["card_def"].name == name)
    entry = pending["remaining"].pop(idx)

    def _finish():
        state.pending_resolution = pending  # reinstall -- see this function's own docstring
        if not pending["remaining"]:
            complete_resolution(state)

    if entry.get("targeting"):
        # Target-at-promotion: run the entry's own etb_trigger/ltb_trigger
        # hook now; it opens target selection and pushes its own stack entry
        # once targets lock.
        registry.EFFECT_REGISTRY[entry["card_def"].effect_id][entry["hook"]](state, entry["permanent"])
        if state.pending_resolution is not None and state.pending_resolution is not pending:
            inner_on_complete = state.pending_resolution["on_complete"]

            def _continue(s, *args):
                inner_on_complete(s, *args)
                _finish()

            state.pending_resolution["on_complete"] = _continue
            return
    else:
        entry["controller"] = state.active_idx  # active_idx is still the trigger owner here
        entry["reserves_hand_card"] = False  # a queued trigger, never a real cast
        entry["is_spell"] = False  # a triggered ability, never a Counterspell target
        state.stack.append(entry)  # already the stack's native {"card_def", "resolve"} shape
    _finish()
