"""Exhaustive search over mask-legal action sequences from hand-built positions.

A position that is mid-decision holds closures over live permanents and cannot be copied, so every search node is
REBUILT: `explore` calls `build()` (a deterministic factory) and replays the node's prefix of move descriptors. A move is
described by its row label, or for a pointer pick by `pointer <owner>:<name>#<slot>`, so a descriptor means the same
thing in every rebuild.
"""
import drl_env
from rl.decision import action_bridge as AB
from support.fingerprint import state_fingerprint


def _identities(state):
    """Token positions as the pointer head sees them: the active player's battlefield, then the other player's."""
    a = state.active_idx
    return [(a, p) for p in state.players[a].battlefield] + [(1 - a, p) for p in state.players[1 - a].battlefield]


def moves(state, table):
    """[(descriptor, apply(state))] for every legal move now: fixed rows then pointer picks."""
    out = []
    mask = drl_env.legal_action_mask(state, table)
    for i, ok in enumerate(mask):
        if ok:
            out.append((table[i][0], lambda s, i=i: table[i][2](s)))
    if AB.any_pointer_legal(state):
        ids = _identities(state)
        legal = AB.pointer_legal_mask(state, [p for _owner, p in ids])
        for (owner, p), ok in zip(ids, legal, strict=True):
            if ok:
                desc = f"pointer {owner}:{p.card_def.name}#{p.slot}"
                out.append((desc, lambda s, owner=owner, name=p.card_def.name, slot=p.slot:
                            AB.execute_pointer_choice(s, find(s, owner, name, slot))))
    return out


def find(state, owner, name, slot):
    """The permanent of `owner` (seat index) with this name and slot in `state`."""
    return next(p for p in state.players[owner].battlefield if p.card_def.name == name and p.slot == slot)


def explore(build, table, is_terminal, observe, cap=20000, terminal_moves=(), keep=lambda desc: True):
    """Depth-first over every sequence of legal moves from build()'s position, deduplicating positions by fingerprint.
    `is_terminal(state)` ends a path; `terminal_moves` are move labels (e.g. "Pass") that end a path too, observed in the
    position they were chosen in; `keep(descriptor)` restricts the search to some moves; `observe(state)` is called at every terminal and its return values are collected.
    A state with no legal move that is not terminal is a DEAD END (returned). Returns (observations, dead_ends, nodes)."""
    seen, observations, dead = set(), [], []
    stack = [()]
    while stack:
        prefix = stack.pop()
        ends_with_terminal = bool(prefix) and prefix[-1] in terminal_moves
        state = build()
        for want in prefix[:-1] if ends_with_terminal else prefix:
            apply = next(a for d, a in moves(state, table) if d == want)
            apply(state)
        fp = state_fingerprint(state)
        if ends_with_terminal:
            observations.append(observe(state))
            continue
        if fp in seen:
            continue
        seen.add(fp)
        if len(seen) > cap:
            raise AssertionError(f"search exceeded {cap} positions; shrink the scenario")
        if is_terminal(state):
            observations.append(observe(state))
            continue
        nxt = [(d, a) for d, a in moves(state, table) if keep(d)]
        if not nxt:
            dead.append(prefix)
            continue
        stack.extend(prefix + (d,) for d, _a in nxt)
    return observations, dead, len(seen)
