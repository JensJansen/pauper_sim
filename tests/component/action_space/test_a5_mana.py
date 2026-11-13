"""Layer A5 -- mana and payment, checked exhaustively where it can be and by volume where it cannot.

  A5a  the engine's affordability test (`game.can_pay`, Hall's condition) against the oracle's brute-force search
       (`support.action_oracle.payable`), on abstract problems: an exhaustive small grid plus many random ones.
  A5b  the same question on REAL boards: random boards of real mana cards (lands, elves, Battlement, Priest, Tron, filters,
       Wall of Roots, Petal, ...) with random tapped / summoning-sick state, random pools and Auras on either side's lands,
       real card costs. `game.plan_payment` against the oracle's independent per-card source model.
  A5c  no stranding, and completeness of payment: from a board and a cost the oracle calls payable, a depth-first search
       over EVERY sequence of mask-legal mana actions (taps, copy choices, Saruli and filter sub-decisions, pool spends)
       must always be able to finish paying, never reach a dead end, and must finish on some path; from a cost the oracle
       calls unpayable the payment must not even begin.
"""
import functools
import itertools
import random


import drl_env
import game
from rl.decision import action_bridge as AB
from support import action_oracle as O
from support import scenario as S
from support.fingerprint import state_fingerprint

COLORS = game.POOL_COLORS


# ---- A5a: abstract problems -----------------------------------------------------------------------------------------

def _engine(units, cost):
    return game.can_pay([frozenset(u) for u in units], cost)


def _oracle(units, cost):
    return O.payable([], [[[c] for c in sorted(u)] for u in units], cost)


def test_can_pay_agrees_with_brute_force_on_an_exhaustive_small_grid():
    colors = ("G", "R", "C")
    unit_kinds = [frozenset(s) for r in (1, 2, 3) for s in itertools.combinations(colors, r)]
    costs = [{"generic": g, **({"G": a} if a else {}), **({"R": b} if b else {})}
             for g in range(3) for a in range(3) for b in range(3) if g + a + b <= 4]
    n = 0
    for k in range(5):
        for units in itertools.combinations_with_replacement(unit_kinds, k):
            for cost in costs:
                n += 1
                assert _engine(units, cost) == _oracle(units, cost), (units, cost)
    assert n > 5_000


def test_can_pay_agrees_with_brute_force_on_random_larger_problems():
    rng = random.Random(5)
    for _ in range(6000):
        units = [frozenset(rng.sample(COLORS, rng.choice([1, 1, 1, 2, 2, 3, 5]))) for _ in range(rng.randint(0, 9))]
        cost = {"generic": rng.randint(0, 5)}
        for c in rng.sample(COLORS, rng.randint(0, 3)):
            cost[c] = rng.randint(1, 3)
        assert _engine(units, cost) == _oracle(units, cost), (units, cost)


# ---- A5b: real boards -----------------------------------------------------------------------------------------------

# Cards that make mana (the oracle models each from the registry). Saruli Caretaker is deliberately in the pool: both sides
# must refuse to count it as free supply (authorized, owner 2026-08-17).
MANA_CARDS = [n for n, cd in sorted(game.CARD_DEFS.items())
              if any(k in game.EFFECT_REGISTRY.get(cd.effect_id, {}) for k in ("mana", "filter_mana", "mana_extra_choose"))]
LAND_NAMES = [n for n in MANA_CARDS if game.CARD_DEFS[n].card_type == game.CardType.LAND]
COST_POOL = [dict(cd.cast_cost) for cd in game.CARD_DEFS.values() if cd.cast_cost]


def _random_board(rng):
    """Seat 1 (active, owner of the mana sources) and seat 0 (opponent, who may also control Auras on seat 1's lands)."""
    n = rng.randint(1, 6)
    mine = []
    for _ in range(n):
        name = rng.choice(MANA_CARDS)
        flags = {"tapped": rng.random() < 0.25, "sick": rng.random() < 0.3}
        if name == "Wall of Roots":
            flags["counters"] = {"-0/-1": rng.randint(0, 4)}
        mine.append((name, flags))
    mine_extra = []
    theirs = []
    lands = [i for i, (nm, _f) in enumerate(mine) if nm in LAND_NAMES]
    for i in rng.sample(lands, k=min(len(lands), rng.randint(0, 2))):
        aura = rng.choice(["Utopia Sprawl", "Abundant Growth"])
        target_forest = aura == "Utopia Sprawl"
        if target_forest and mine[i][0] != "Forest":
            continue
        (theirs if rng.random() < 0.5 else mine_extra).append((aura, {"aura_on": (1, i)}))
    state = S.build(S.seat(bf=theirs, hand=[]), S.seat(bf=mine + mine_extra))
    for pl in state.players:
        for p in pl.battlefield:
            if p.card_def.name == "Utopia Sprawl":
                p.flags["bonus_mana_color"] = rng.choice(game.COLORS)
            elif p.card_def.name == "Abundant Growth":
                p.flags["bonus_mana_colors"] = set(game.COLORS)
    state.active_idx = state.turn_player_idx = 1
    state.phase = game.turn.Phase.MAIN1
    for c in rng.sample(COLORS, rng.randint(0, 2)):
        state.players[1].mana_pool[c] = rng.randint(1, 2)
    return state


def _random_cost(rng):
    if rng.random() < 0.6:
        return dict(rng.choice(COST_POOL))
    cost = {"generic": rng.randint(0, 4)}
    for c in rng.sample(COLORS, rng.randint(0, 3)):
        cost[c] = rng.randint(1, 2)
    return cost


def test_plan_payment_agrees_with_the_oracle_on_random_real_boards():
    rng = random.Random(77)
    disagreements = []
    payable = 0
    N = 7000
    for _ in range(N):
        state = _random_board(rng)
        cost = _random_cost(rng)
        eng = game.plan_payment(state, cost) is not None
        orc = O.can_afford(state, cost)
        payable += orc
        if eng != orc:
            disagreements.append((eng, orc, cost, [(p.card_def.name, p.tapped, p.summoning_sick) for p in state.players[1].battlefield],
                                  dict(state.players[1].mana_pool)))
    assert payable > N * 0.15 and payable < N * 0.95, f"degenerate sample: {payable}/{N} payable"
    detail = "\n".join(str(d) for d in disagreements[:5])
    assert not disagreements, f"{len(disagreements)} unexplained disagreements, e.g.\n{detail}"


# ---- A5c: depth-first search over every payment -----------------------------------------------------------------------

def _mana_decklist(state):
    names = sorted({p.card_def.name for pl in state.players for p in pl.battlefield})
    return tuple((n, 4) for n in names)


@functools.lru_cache(maxsize=None)
def _table_for(decklist):
    """Action tables are stateless closures, and building one is the expensive part of rebuilding a search node."""
    return AB.build_fixed_action_table(list(decklist))


def _moves(state, table):
    """Every mask-legal action as (descriptor, apply(state)). Pointer picks are described by (name, slot)."""
    out = []
    mask = drl_env.legal_action_mask(state, table)
    for i, ok in enumerate(mask):
        if ok:
            out.append((table[i][0], lambda s, i=i: table[i][2](s)))
    if AB.any_pointer_legal(state):
        ids = [*state.players[state.active_idx].battlefield, *state.players[1 - state.active_idx].battlefield]
        for j, ok in enumerate(AB.pointer_legal_mask(state, ids)):
            if ok:
                p = ids[j]
                out.append((f"pointer {p.card_def.name}#{p.slot}", lambda s, p=p: AB.execute_pointer_choice(s, _same(s, p))))
    return out


def _build_case(seed):
    """A deterministic (state with a payment open, cost, table, oracle verdict): rebuilt from the seed whenever a search
    node needs its state, because a state mid-payment holds closures over live permanents and cannot be copied."""
    rng = random.Random(seed)
    state = _random_board(rng)
    state.players[1].battlefield = [p for p in state.players[1].battlefield if p.card_def.name != "Saruli Caretaker" or rng.random() < 0.5]
    cost = _random_cost(rng)
    table = _table_for(_mana_decklist(state))
    orc = O.can_afford(state, cost)
    if orc:  # the engine and the oracle agree on affordability (test above), so this is also what the engine lets begin
        game.begin_pay_cost(state, dict(cost), on_complete=lambda s: None)  # opens the payment exactly as a cast does
    return state, cost, table, orc


def _same(state, ref):
    """The permanent in `state` standing where `ref` stood in a sibling rebuild of the same case (same owner, name, slot)."""
    for pl in state.players:
        for p in pl.battlefield:
            if p.card_def.name == ref.card_def.name and p.slot == ref.slot:
                return p
    raise AssertionError("pointer target vanished")


def _explore(seed, cap=250):
    """DFS over every sequence of legal actions while a payment is open (each node is rebuilt by replaying its prefix).
    Returns (finished, dead_ends, nodes)."""
    seen = set()
    stack = [()]
    finished = dead = 0
    while stack:
        prefix = stack.pop()
        state, _cost, table, _orc = _build_case(seed)
        for want in prefix:
            move = next(apply for desc, apply in _moves(state, table) if desc == want)
            move(state)
        fp = state_fingerprint(state)
        if fp in seen:
            continue
        seen.add(fp)
        if len(seen) > cap:
            break
        if state.pending_resolution is None and state.mana_subdecision is None:
            finished += 1
            continue
        moves = _moves(state, table)
        if not moves:
            dead += 1
            continue
        stack.extend(prefix + (desc,) for desc, _apply in moves)
    return finished, dead, len(seen)


def test_every_payment_can_be_finished_and_none_strands():
    stats = {"started": 0, "unpayable": 0, "finished": 0, "nodes": 0}
    failures = []
    seed = 10_000
    while stats["started"] < 160 and seed < 14_000:
        seed += 1
        state, cost, _table, orc = _build_case(seed)
        if state.pending_resolution is None:  # a free cost, or one the pool already paid
            continue
        if not orc:
            stats["unpayable"] += 1
            continue
        stats["started"] += 1
        finished, dead, nodes = _explore(seed)
        stats["finished"] += finished
        stats["nodes"] += nodes
        if dead or not finished:
            failures.append({"seed": seed, "cost": cost, "dead_ends": dead, "finished": finished, "nodes": nodes,
                             "board": [(p.card_def.name, p.tapped, p.summoning_sick) for p in state.players[1].battlefield],
                             "pool": dict(state.players[1].mana_pool)})
    assert stats["started"] >= 150, stats
    detail = "\n".join(str(f) for f in failures[:4])
    assert not failures, f"{len(failures)} payments stranded or could not finish, e.g.\n{detail}"
