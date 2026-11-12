"""Structural anti-loop property test for the declare-blockers step.

Both infinite declare-blockers loops found previously had the same shape:
two readers of the same rule consulting different state (e.g. can_block
honoring reach while a blocker-assignment predicate didn't, or one function
iterating attackers while another iterated battlefield). In both cases a
legal action recorded nothing and declare-blockers re-opened on a
byte-identical state forever -- unbounded by `horizon`, which bounds turns,
not priority rounds.

So this file does not test a rule. It tests the invariant those bugs broke:

    while a declare_blockers resolution is pending, every legal action must
    change observable state, and the step must terminate.

Boards are sampled from registry.CARD_DEFS itself, so a newly added
creature with a new evasion keyword is covered automatically.
"""
import random

import pytest

import game
from drl_env import _actions_table
from game import registry
from game.state import GameState, Permanent, PlayerState

N_TRIALS = 300
MAX_BLOCKING_DECISIONS = 60  # generous: far above (creatures + Done) for these board sizes


def _creature_names():
    return sorted(
        name for name, cd in registry.CARD_DEFS.items()
        if cd.card_type == game.CardType.CREATURE
    )


def _names_with_keyword(keyword):
    """Creature names whose registry entry grants `keyword`. Read from the
    registry rather than hardcoded, so a new flier/reach creature joins the
    board generator automatically."""
    out = []
    for name in _creature_names():
        entry = registry.EFFECT_REGISTRY.get(registry.CARD_DEFS[name].effect_id) or {}
        if keyword in (entry.get("keywords") or ()):
            out.append(name)
    return out


def _blocking_fingerprint(state):
    """Everything the declare-blockers step can legitimately change. Two
    consecutive decisions with an identical fingerprint mean the action in
    between did nothing -- the loop signature."""
    return (
        state.pending_resolution["kind"] if state.pending_resolution else None,
        tuple(sorted(
            (a.card_def.name, a.slot, tuple(sorted((b.card_def.name, b.slot) for b in bs)))
            for pl in state.players for a, bs in pl.blocked_by.items()
        )),
        tuple(sorted(
            (p.card_def.name, p.slot) for pl in state.players for p in pl.attackers
        )),
        tuple(sorted(
            (p.card_def.name, p.slot, p.tapped) for pl in state.players for p in pl.battlefield
        )),
    )


def _build_board(rng, names):
    """A random mid-combat board: player 0 attacking, player 1 defending,
    with attackers already declared and (sometimes) one already dead
    (506.4)."""
    # Bias the pool toward evasion matchups, since a uniform sample rarely
    # mixes an evasive attacker with a reach-but-not-flying blocker. Seeding
    # one flier and one reach creature into every board raises that
    # coverage; remaining slots stay uniformly sampled.
    pool = rng.sample(names, k=min(2, len(names)))
    for keyword in ("flying", "reach"):
        candidates = _names_with_keyword(keyword)
        if candidates:
            pool.append(rng.choice(candidates))
    pool = list(dict.fromkeys(pool))  # dedupe, preserve order
    decklist = [(n, 4) for n in pool]

    state = GameState(on_the_play=True, players=[PlayerState(True), PlayerState(False)])
    for idx in (0, 1):
        permanents = []
        for name in pool:
            for slot in range(1, rng.randint(1, 2) + 1):
                p = Permanent(registry.CARD_DEFS[name])
                p.slot = slot
                p.summoning_sick = False
                permanents.append(p)
        state.players[idx].battlefield = permanents

    # Declare a random non-empty subset of player 0's creatures as attackers.
    candidates = [p for p in state.players[0].battlefield if not p.tapped]
    if not candidates:
        return None, None
    for p in rng.sample(candidates, k=rng.randint(1, len(candidates))):
        game.declare_attacker(state, p)
    if not state.players[0].attackers:
        return None, None

    # Sometimes kill an attacker before blocks -- 506.4.
    if rng.random() < 0.4:
        doomed = rng.choice(list(state.players[0].attackers))
        state.players[0].battlefield.remove(doomed)
        game.remove_from_combat(state, doomed)

    state.active_idx = 1  # _declare_blockers_gen's own flip to the defender
    return state, decklist


def _legal_blocker_actions(actions, state):
    """Only actions gated to the declare_blockers resolution, mirroring how the
    engine's own mask sweep gates them. Calling every legal_fn in the table
    unconditionally is wrong: a mana row assumes an open mana subdecision and
    raises without one.

    A _pending_gate is either a frozenset of pending kinds or the
    _GATE_NO_PENDING sentinel (a bare object, for rows legal only with NO
    pending resolution), so it has to be type-checked, not just membership-
    tested."""
    out = []
    for name, legal, execute in actions:
        gate = getattr(legal, "_pending_gate", None)
        if not isinstance(gate, (frozenset, set)) or "declare_blockers" not in gate:
            continue
        if legal(state):
            out.append((name, execute))
    return out


@pytest.mark.parametrize("seed", range(N_TRIALS))
def test_no_legal_declare_blockers_action_is_ever_a_no_op(seed):
    rng = random.Random(seed)
    names = _creature_names()
    state, decklist = _build_board(rng, names)
    if state is None:
        pytest.skip("degenerate board (no untapped creature to attack with)")

    actions = _actions_table.build_action_table(
        decklist, game.EFFECT_REGISTRY, opponent_decklist=decklist,
    )
    game.begin_declare_blockers(state, on_complete=lambda s: None)

    for step in range(MAX_BLOCKING_DECISIONS):
        pending = state.pending_resolution
        if pending is None:
            return  # the step finished -- the only correct way out
        before = _blocking_fingerprint(state)

        if pending["kind"] == "choose_opponent_permanent":
            options = game.choose_opponent_permanent_options(state)
            assert options, (
                "a choose_opponent_permanent opened with zero options -- it will fizzle "
                "with None, record nothing, and re-open declare_blockers unchanged. "
                "This is the loop signature (both 2026-08-19 bugs)."
            )
            name, slot = rng.choice(options)
            game.execute_choose_opponent_permanent_option(state, name, slot)
        else:
            assert pending["kind"] == "declare_blockers", pending["kind"]
            legal = _legal_blocker_actions(actions, state)
            if not legal:
                return  # zero legal actions -> _declare_blockers_gen abandons; bounded
            action_name, execute = rng.choice(legal)
            execute(state)
            # An assign opens a nested choice; that alone is progress. The
            # fizzle-to-nothing case lands back on declare_blockers unchanged,
            # which the fingerprint below catches.

        after = _blocking_fingerprint(state)
        assert after != before, (
            f"seed={seed} step={step}: a legal action left observable state identical.\n"
            f"pending={pending['kind']} fingerprint={before}\n"
            "This is exactly how both infinite declare-blockers loops behaved."
        )

    pytest.fail(
        f"seed={seed}: declare-blockers did not terminate in {MAX_BLOCKING_DECISIONS} decisions -- "
        "non-progressing round, even though every individual action changed state."
    )


def test_the_property_test_would_catch_a_reintroduced_no_op():
    """Mutation check on the harness itself: if a legal Assign Blocker records
    no block (the shared bug), the fingerprint must be unchanged. Without this,
    a fingerprint that silently captured nothing would make every trial above
    vacuously pass."""
    state = GameState(on_the_play=True, players=[PlayerState(True), PlayerState(False)])
    attacker = Permanent(game.CARD_DEFS["Sneaky Snacker"])
    attacker.tapped = True
    blocker = Permanent(game.CARD_DEFS["Generous Ent"])
    state.players[0].battlefield = [attacker]
    state.players[0].attackers = [attacker]
    state.players[1].battlefield = [blocker]
    state.active_idx = 1

    game.begin_declare_blockers(state, on_complete=lambda s: None)
    before = _blocking_fingerprint(state)

    # Simulate the bug: re-open declare_blockers having recorded nothing.
    game.begin_declare_blockers(state, on_complete=lambda s: None)
    assert _blocking_fingerprint(state) == before, (
        "the fingerprint must be blind to a no-op re-open, or the property test proves nothing"
    )

    # And it must NOT be blind to a real block being recorded.
    state.players[0].blocked_by[attacker] = [blocker]
    assert _blocking_fingerprint(state) != before
