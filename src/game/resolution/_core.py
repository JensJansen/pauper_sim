"""The pending-resolution state machine core: begin_resolution starts a
multi-action decision, complete_resolution finishes it and fires the
on_complete callback. A leaf -- every concrete handler in handlers.py builds
on these; they never build on the handlers."""

from ..cards import CardDef


def begin_resolution(state, kind, on_complete, **fields):
    """Start a pending resolution. on_complete(state) runs once it's fully
    resolved (via repeated calls into that kind's own option/execute
    functions) -- it may itself begin a further resolution, so multi-step
    effects chain naturally through nested callbacks rather than needing a
    single monolithic resolution type."""
    state.pending_resolution = {"kind": kind, "on_complete": on_complete, **fields}
    # Generic catch-all covering every resolution kind (pay_cost, scry/
    # surveil, choose_permanent, search_fetch, sacrifice, ...) from ONE
    # instrumentation point -- "a decision window of this kind just
    # opened," never the specific zone-move that eventually results from
    # it (that's each kind's own execute_*/on_complete's job to log).
    state.log_event("resolution_begin", resolution_kind=kind)


def _loggable(value):
    """complete_resolution's own *args can carry raw CardDef objects
    directly -- e.g. execute_discard_option's own discarded_cards list
    (appends the real card, never just its name, since _discard_one and
    the madness trigger queue both need the actual object) -- not
    JSON-serializable, confirmed the hard way: a real 50-game mono_red_
    madness/rakdos_madness match (both madness decks, discarding
    constantly) crashed run_league.py's own event-log write on exactly
    this path. Converts a CardDef (or a list of them) to its own .name;
    every other args shape already used elsewhere (strings, (name, slot)
    tuples, bools, ints, None, plain lists of those) passes through
    unchanged -- this only ever touches the LOGGED copy, never what
    on_complete actually receives."""
    if isinstance(value, CardDef):
        return value.name
    if isinstance(value, list):
        return [_loggable(v) for v in value]
    return value


def complete_resolution(state, *args):
    """*args is an optional payload for kinds whose completion carries a
    result the caller needs (e.g. search_fetch's chosen card name) --
    on_complete(state) for kinds that don't (e.g. pay_cost)."""
    kind = state.pending_resolution["kind"]
    on_complete = state.pending_resolution["on_complete"]
    state.pending_resolution = None
    logged_result = [_loggable(a) for a in args] if args else None
    state.log_event("resolution_complete", resolution_kind=kind, result=logged_result)
    on_complete(state, *args)
