"""The pending-resolution state machine core: begin_resolution starts a
multi-action decision, complete_resolution finishes it and fires the
on_complete callback. Every handlers_<category> module builds on these;
these never build on the handlers."""

from ..cards import CardDef


def begin_resolution(state, kind, on_complete, **fields):
    """Start a pending resolution. on_complete(state) runs once it's fully
    resolved via repeated calls into that kind's own option/execute
    functions; it may itself begin a further resolution, chaining multi-step
    effects through nested callbacks."""
    state.pending_resolution = {"kind": kind, "on_complete": on_complete, **fields}
    # One instrumentation point for every resolution kind: "a decision window
    # of this kind just opened." The specific resulting zone-move is logged
    # separately by each kind's own execute_*/on_complete.
    state.log_event("resolution_begin", resolution_kind=kind)


def _loggable(value):
    """Converts a CardDef (or list of them) to its .name for logging, since
    CardDef isn't JSON-serializable; every other args shape (strings,
    (name, slot) tuples, bools, ints, None, lists of those) passes through
    unchanged. Only touches the logged copy, not what on_complete receives."""
    if isinstance(value, CardDef):
        return value.name
    if isinstance(value, list):
        return [_loggable(v) for v in value]
    return value


def complete_resolution(state, *args):
    """*args is an optional result payload (e.g. search_fetch's chosen card
    name) for kinds whose completion carries one; omitted for kinds that
    don't (e.g. pay_cost)."""
    kind = state.pending_resolution["kind"]
    on_complete = state.pending_resolution["on_complete"]
    state.pending_resolution = None
    logged_result = [_loggable(a) for a in args] if args else None
    state.log_event("resolution_complete", resolution_kind=kind, result=logged_result)
    on_complete(state, *args)
