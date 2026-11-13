"""Forking a live game state so an action can be tried and thrown away.

A state is mid-flight: pending resolutions and stack entries hold CLOSURES (`on_complete`, `resolve`, ...), and
`copy.deepcopy` leaves a closure pointing at the ORIGINAL objects it captured. Measured on real games, ~1.7% of decision
states have a closure that captured a live game object (a Permanent, or a dict/list that is also part of the state), so a
deepcopy of one of them would let a trial action silently mutate or read the original game.

`is_fork_safe(state)` finds exactly those states; `fork_state(state)` deep-copies the rest. For an unsafe state the caller
re-creates the position by replaying the recorded game up to that decision (`support.action_corpus.replay_fork`), which is
exact because the engine is seeded and deterministic.

CardDef objects are interned and compared by identity everywhere (no __eq__), so they are shared with the copy, never
duplicated.
"""
import copy
import types
from enum import Enum

from game.cards import CardDef
from game.state import CardInstance, GameState, Permanent, PlayerState

_LIVE = (Permanent, CardInstance, PlayerState, GameState)


def _walk(state):
    """(ids of every mutable object reachable from state, the functions found on the way). Does not descend into
    closures: those are inspected separately."""
    live, funcs, seen = set(), [], set()

    def go(o, depth=0):
        if o is None or isinstance(o, (bool, int, float, str, bytes, Enum, CardDef)) or depth > 40:
            return
        if isinstance(o, types.FunctionType):
            funcs.append(o)
            return
        if isinstance(o, (types.BuiltinFunctionType, types.MethodType)):
            return
        if id(o) in seen:
            return
        seen.add(id(o))
        if isinstance(o, (list, tuple, set, frozenset)):
            if not isinstance(o, (tuple, frozenset)):
                live.add(id(o))
            for x in o:
                go(x, depth + 1)
            return
        if isinstance(o, dict):
            live.add(id(o))
            for k, v in o.items():
                go(k, depth + 1)
                go(v, depth + 1)
            return
        live.add(id(o))
        attrs = getattr(o, "__dict__", None)
        if attrs is not None:
            for v in attrs.values():
                go(v, depth + 1)

    go(state)
    return live, funcs


def _cell_values(f, seen):
    """Everything a function carries with it: closure cells AND default / keyword-default arguments (`def g(s, inst=inst)`
    is the same capture by another route, and is common in this engine)."""
    if id(f) in seen:
        return
    seen.add(id(f))
    carried = []
    for cell in f.__closure__ or ():
        try:
            carried.append(cell.cell_contents)
        except ValueError:
            continue
    carried.extend(f.__defaults__ or ())
    carried.extend((f.__kwdefaults__ or {}).values())
    for v in carried:
        yield v
        if isinstance(v, types.FunctionType):
            yield from _cell_values(v, seen)


def is_fork_safe(state):
    """True when no closure reachable from the state captured a live game object."""
    live, funcs = _walk(state)
    seen = set()
    for f in funcs:
        for v in _cell_values(f, seen):
            if isinstance(v, _LIVE) or (id(v) in live and not isinstance(v, (CardDef, types.FunctionType))):
                return False
            if isinstance(v, (list, tuple, set)) and any(isinstance(x, _LIVE) or id(x) in live for x in v):
                return False
    return True


def fork_state(state):
    """Deep copy with the interned CardDefs shared. Only valid when is_fork_safe(state)."""
    memo = {}

    seen = set()
    stack = [state]
    while stack:
        o = stack.pop()
        if o is None or isinstance(o, (bool, int, float, str, bytes, Enum)) or id(o) in seen:
            continue
        seen.add(id(o))
        if isinstance(o, CardDef):
            memo[id(o)] = o
        elif isinstance(o, (list, tuple, set, frozenset)):
            stack.extend(o)
        elif isinstance(o, dict):
            stack.extend(o.keys())
            stack.extend(o.values())
        elif isinstance(o, (types.FunctionType, types.BuiltinFunctionType, types.MethodType)):
            continue
        elif hasattr(o, "__dict__"):
            stack.extend(o.__dict__.values())
    return copy.deepcopy(state, memo)
