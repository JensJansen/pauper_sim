"""Trigger placement ordering: when 2+ of one player's own triggers are ready
to move onto the stack at once, the player chooses the placement order
(603.3b). Re-exported via game.resolution so `from ..resolution import X`
in the catalogs keeps resolving."""

from .. import registry
from ._core import begin_resolution, complete_resolution


def begin_order_triggers(state, entries, on_complete):
    """2+ of the active player's own
    triggers are ready to move onto the stack at once (e.g. Faithless
    Looting's discard-2 hitting two Madness cards in the same discard, two
    Sneaky Snackers both crossing their own draw-count trigger on the same
    draw, or two targeting ETBs entering simultaneously) -- real Magic lets
    that player choose the PLACEMENT order (603.3b), not a fixed queue
    order. Only ever a single player's own triggers here -- APNAP ordering
    BETWEEN players is handled one level up, by
    game.effects.triggers.promote_triggers_to_stack's own group placement,
    before this is ever opened for either player's group.

    entries: list of {"card_def", "resolve"} dicts (plain triggers, already
    stack-ready) or {"card_def", "permanent", "targeting": True, "hook"}
    dicts (target-at-promotion ETBs/LTBs -- placing one runs its OWN
    etb_trigger/ltb_trigger hook (whichever "hook" names) instead, see
    execute_order_triggers_option's targeting branch) -- built
    by game.effects.triggers.promote_triggers_to_stack, which is
    also what turns each queued trigger's own (type, kind) into the right
    resolve function -- this module only ever deals in the stack's own
    generic shape, never trigger-specific semantics, same reverse-import
    reason execute_madness_cast's own cost-payment lives in
    game/effects/madness_and_plot.py instead of here).

    Picks one at a time; each pick is placed immediately
    (execute_order_triggers_option below), not deferred to the end --
    PLACEMENT order, not resolution order. Since the stack is LIFO, whichever
    PLAIN entry is placed LAST resolves FIRST (a targeting entry's own effect
    goes on the stack at ITS placement moment too, once it locks targets,
    following that same LIFO rule). on_complete(state) once every entry has
    been placed."""
    begin_resolution(state, "order_triggers", on_complete, remaining=list(entries))


def order_triggers_options(state):
    return sorted({e["card_def"].name for e in state.pending_resolution["remaining"]})


def execute_order_triggers_option(state, name):
    """Places one still-`remaining` entry, then either advances to the next
    pick (this same "order_triggers" resolution stays open) or completes it
    once none are left.

    A targeting entry's own hook can itself open a fresh pending_resolution
    (a real target existed) -- begin_resolution is a flat, single-slot
    assignment (state.pending_resolution's own docstring), so that call
    REPLACES this "order_triggers" resolution in state.pending_resolution
    out from under `pending` (still a live reference to the detached dict)
    the instant it runs. Finishing this pick (checking `pending["remaining"]`
    and maybe completing it) right then would either complete/clear the
    WRONG resolution (the freshly-opened one, with the wrong on_complete
    signature) or -- if not yet empty -- silently strand the rest of this
    order_triggers pick with no live state.pending_resolution ever pointing
    back to it again. So _finish (below) is deferred, chained onto that
    fresh decision's own on_complete, exactly like game.effects.triggers.
    _place_trigger_groups' own single-entry targeting branch -- but ONLY
    when state.pending_resolution is genuinely a DIFFERENT object than
    `pending` afterward (an `is not` identity check, not just `is not
    None`): a target-less hook auto-completes (and clears) the fresh one
    synchronously via complete_resolution before returning here, and a hook
    that opens no resolution at all (e.g. a plain push_to_stack, no
    targeting) leaves state.pending_resolution as this SAME still-open
    `pending` dict, unchanged -- either way there's no fresh decision to
    defer to, `is not None` alone can't tell the two apart from "a fresh
    one just replaced it," and _finish unconditionally reinstalls `pending`
    first regardless of which case this was, so it's always the correct
    resolution being checked/completed."""
    pending = state.pending_resolution
    idx = next(i for i, e in enumerate(pending["remaining"]) if e["card_def"].name == name)
    entry = pending["remaining"].pop(idx)

    def _finish():
        state.pending_resolution = pending  # reinstall -- see this function's own docstring
        if not pending["remaining"]:
            complete_resolution(state)

    if entry.get("targeting"):
        # Target-at-promotion (game.effects.triggers.promote_triggers_to_
        # stack's own docstring): placing this entry means running its OWN
        # etb_trigger/ltb_trigger hook (entry["hook"]) NOW -- it opens target
        # selection and pushes its own stack entry once targets are locked,
        # rather than this function pushing a plain resolve directly. Same
        # two-way branch promote_triggers_to_stack's own single-entry fast
        # path uses.
        registry.EFFECT_REGISTRY[entry["card_def"].effect_id][entry["hook"]](state, entry["permanent"])
        if state.pending_resolution is not None and state.pending_resolution is not pending:
            inner_on_complete = state.pending_resolution["on_complete"]

            def _continue(s, *args):
                inner_on_complete(s, *args)
                _finish()

            state.pending_resolution["on_complete"] = _continue
            return
    else:
        # Same controller field push_to_stack itself stamps on every entry
        # -- state.active_idx here is still the
        # trigger owner (nothing else can interleave mid-resolution), so this
        # is the correct moment to record it, same reasoning push_to_stack's
        # own docstring gives.
        entry["controller"] = state.active_idx
        # Every entry reaching here originates from triggers.promote_triggers_
        # to_stack (begin_order_triggers's own docstring: queued triggers only,
        # never a real cast) -- same "never reserves a hand card" reasoning
        # push_to_stack(..., reserves_hand_card=False) applies for the
        # single-trigger branch right above this function's own caller.
        entry["reserves_hand_card"] = False
        entry["is_spell"] = False  # a triggered ability, not a spell -- never a Counterspell target
        state.stack.append(entry)  # already the stack's own native {"card_def", "resolve"} shape
        # A triggered ability going on the stack moves no card (is_spell=False), so
        # emit no card zone_move -- same reasoning as push_to_stack / resolve_top_of_stack.
    _finish()
