"""A deep, identity-canonical fingerprint of a game state.

`state_fingerprint(state)` serializes everything reachable from the state (players, zones, permanents, flags, counters,
stack, pending resolution, mana pools, rng position, event-log length) into a nested tuple. Objects are replaced by a
stable first-visit index, so two states that are structurally the same (a state and its deepcopy, or a state before and
after a pure computation) fingerprint identically, while any mutation changes it. Functions/closures are recorded by
qualified name only (their captured cells are not walked: they would drag in the whole object graph and the
fingerprint's job is observable state, not callback identity).
"""
import types
from enum import Enum

_SKIP = {"event_log", "on_mana_burn", "on_single_pip_burn"}


def state_fingerprint(state):
    ids = {}

    def walk(o, depth=0):
        if o is None or isinstance(o, (bool, int, float, str, bytes)):
            return o
        if isinstance(o, Enum):
            return (type(o).__name__, o.name)
        if isinstance(o, types.FunctionType):
            return ("fn", o.__qualname__)
        if isinstance(o, (types.BuiltinFunctionType, types.MethodType)):
            return ("fn", getattr(o, "__qualname__", repr(o)))
        if depth > 40:
            return "<deep>"
        if isinstance(o, (list, tuple)):
            return tuple(walk(x, depth + 1) for x in o)
        if isinstance(o, (set, frozenset)):
            return ("set", tuple(sorted((repr(walk(x, depth + 1)) for x in o))))
        if isinstance(o, dict):
            return ("dict", tuple(sorted(((repr(walk(k, depth + 1)), walk(v, depth + 1)) for k, v in o.items()),
                                         key=lambda kv: kv[0])))
        key = id(o)
        if key in ids:
            return ("ref", ids[key])
        ids[key] = len(ids)
        if hasattr(o, "getstate") and hasattr(o, "random"):  # random.Random
            return ("rng", hash(o.getstate()))
        attrs = getattr(o, "__dict__", None)
        if attrs is None:
            slots = getattr(type(o), "__slots__", ())
            attrs = {s: getattr(o, s) for s in slots if hasattr(o, s)}
        body = tuple(sorted(((k, walk(v, depth + 1)) for k, v in attrs.items() if k not in _SKIP), key=lambda kv: kv[0]))
        return (type(o).__name__, ids[key], body)

    fp = walk(state)
    log = getattr(state, "event_log", None)
    return fp, (None if log is None else len(log))
