"""Layer A6 -- combat, checked against the rules and searched exhaustively on small boards.

Random boards of REAL creatures (every creature in the card pool is sampled, evasion and defenders deliberately seeded in):

  A6a  declare attackers: the pointer mask equals the rule (an untapped creature that is not a Defender, not already
       attacking, not summoning-sick unless it has haste); an attacker taps unless it has vigilance; Pass is refused while a
       goaded creature that can attack has not been declared. A depth-first search over every declaration sequence reaches
       EXACTLY the attacker sets the rules allow (every subset of the eligible, containing all goaded-and-able ones).
  A6b  declare blockers: the blocker mask and the nested "which attacker" mask equal the rules (untapped creature, not yet
       committed, can block something; fliers need flying/reach); "Done blocking" is refused exactly while a menace
       attacker has a lone blocker. A depth-first search over every declaration sequence reaches EXACTLY the legal block
       assignments (a blocker blocks one attacker; flying; menace = 0 or 2+ blockers), with no dead end. (A declaration the
       engine abandons because a menace attacker is stuck with one blocker and no second is available is judged after
       enforce_menace, as the engine does.)
  A6c  combat damage among several blockers: a depth-first search over every assignment reaches EXACTLY the splits the
       owner-approved model allows (each blocker up to its lethal cap, never more before all are capped; leftover to the
       player with trample, else onto the last blocker).

Shared with the engine (listed so the oracle is not mistaken for more than it is): keyword lookup (`stats.has_keyword`,
which folds in Auras and until-end-of-turn effects) and permanent power/toughness.
"""
import itertools
import random

import pytest

import drl_env
import game
from game.effects import stats
from game.effects.combat import blocker_lethal_capacities, enforce_menace
from game.state import GameState, Permanent, PlayerState
from rl.decision import action_bridge as AB
from support import search as SR

CREATURES = sorted(n for n, cd in game.CARD_DEFS.items() if cd.card_type == game.CardType.CREATURE)


def _with(keyword):
    return [n for n in CREATURES if keyword in (game.EFFECT_REGISTRY.get(game.CARD_DEFS[n].effect_id, {}).get("keywords") or ())]


FLIERS, REACHERS = _with("flying"), _with("reach")
DEFENDERS = [n for n in CREATURES if game.CARD_DEFS[n].extra.get("defender", False)]
assert FLIERS and REACHERS and DEFENDERS, "the evasion seeds need at least one creature each"
# No creature in the card pool has menace / haste / vigilance printed, but Auras and until-end-of-turn effects grant them
# (permanent.temp_keywords), so the boards grant them at random: the rules code must honor a granted keyword like a printed one.
GRANTED = ("menace", "haste", "vigilance", "flying", "reach", "cant_be_blocked_except_by_flying")


def _pick(rng, k):
    names = rng.sample(CREATURES, k)
    for pool in (FLIERS, REACHERS, DEFENDERS):
        if rng.random() < 0.35:
            names.append(rng.choice(pool))
    return names


def _board(seed, attackers_declared=False):
    """Seat 0 attacks, seat 1 defends. Deterministic in `seed`."""
    rng = random.Random(seed)
    state = GameState(on_the_play=True, players=[PlayerState(True), PlayerState(False)])
    decklist = []
    for idx in (0, 1):
        perms = []
        for name in _pick(rng, rng.randint(1, 3)):
            p = Permanent(game.CARD_DEFS[name])
            p.slot = 1 + sum(1 for q in perms if q.card_def.name == name)
            p.tapped = rng.random() < 0.2
            p.summoning_sick = rng.random() < 0.3
            if rng.random() < 0.25:
                p.temp_keywords.add(rng.choice(GRANTED))
            perms.append(p)
            decklist.append((name, 4))
        state.players[idx].battlefield = perms
    state.active_idx = state.turn_player_idx = 0
    state.phase = game.turn.Phase.DECLARE_ATTACKERS
    for p in state.players[0].battlefield:  # some goaded
        if rng.random() < 0.15:
            p.flags["goaded_by"] = 1
    table = AB.build_fixed_action_table(sorted(set(decklist)))
    if attackers_declared:
        eligible = [p for p in state.players[0].battlefield if _can_attack(state, p)]
        for p in eligible:
            if rng.random() < 0.7:
                game.declare_attacker(state, p)
        if not state.players[0].attackers:
            return None, table
        state.active_idx = 1  # the defender decides
        state.phase = game.turn.Phase.DECLARE_BLOCKERS
        game.begin_declare_blockers(state, on_complete=lambda s: None)
    return state, table


# ---- the rules, restated --------------------------------------------------------------------------------------------

def _can_attack(state, p):
    """508.1a, 302.6, 702.3 (defender), 702.10 (haste)."""
    return (p.card_type == game.CardType.CREATURE and not p.tapped and not p.card_def.extra.get("defender", False)
            and not any(p in pl.attackers for pl in state.players)
            and (not p.summoning_sick or stats.has_haste(state, p)))


def _may_block(state, blocker, attacker):
    """702.9 (flying) / 702.17 (reach), and the 'can't be blocked except by flying' creatures."""
    needs = stats.has_keyword(state, attacker, "flying") or stats.has_keyword(state, attacker, "cant_be_blocked_except_by_flying")
    return (not needs) or stats.has_keyword(state, blocker, "flying") or stats.has_keyword(state, blocker, "reach")


def _legal_assignments(state):
    """Every complete legal block declaration as a frozenset of (blocker key, attacker key)."""
    attackers = list(state.players[0].attackers)
    blockers = [p for p in state.players[1].battlefield if p.card_type == game.CardType.CREATURE and not p.tapped]
    key = lambda p: (p.card_def.name, p.slot)  # noqa: E731
    out = set()
    for choice in itertools.product([None, *attackers], repeat=len(blockers)):
        if any(a is not None and not _may_block(state, b, a) for b, a in zip(blockers, choice, strict=True)):
            continue
        count = {}
        for a in choice:
            if a is not None:
                count[a] = count.get(a, 0) + 1
        if any(n == 1 and stats.has_keyword(state, a, "menace") for a, n in count.items()):  # 509.1c, 702.110
            continue
        out.add(frozenset((key(b), key(a)) for b, a in zip(blockers, choice, strict=True) if a is not None))
    return out


# ---- A6a: attackers ----------------------------------------------------------------------------------------------------

SEEDS = range(300)


def test_attack_mask_equals_the_rule_and_pass_respects_goad():
    checked = 0
    for seed in SEEDS:
        state, table = _board(seed)
        mine, theirs = state.players[0].battlefield, state.players[1].battlefield
        got = AB.pointer_legal_mask(state, [*mine, *theirs])
        want = [_can_attack(state, p) for p in mine] + [False] * len(theirs)
        assert got == want, (seed, [(p.card_def.name, p.tapped, p.summoning_sick) for p in mine])
        goad_blocks_pass = any(p.flags.get("goaded_by") is not None and _can_attack(state, p) for p in mine)
        pass_row = next(i for i, r in enumerate(table) if r[0] == "Pass")
        assert bool(drl_env.legal_action_mask(state, table)[pass_row]) == (not goad_blocks_pass), seed
        checked += any(want)
    assert checked > 100, "too few boards with an eligible attacker"


def test_declaring_an_attacker_taps_it_unless_it_has_vigilance():
    for seed in SEEDS:
        state, _table = _board(seed)
        for p in [q for q in state.players[0].battlefield if _can_attack(state, q)]:
            AB.execute_pointer_choice(state, p)
            assert p in state.players[0].attackers
            assert p.tapped == (not stats.has_keyword(state, p, "vigilance")), (seed, p.card_def.name)


def test_every_reachable_attacker_set_is_exactly_what_the_rules_allow():
    done = 0
    for seed in SEEDS:
        state, table = _board(seed)
        eligible = [p for p in state.players[0].battlefield if _can_attack(state, p)]
        if not 1 <= len(eligible) <= 4:
            continue
        must = {(p.card_def.name, p.slot) for p in eligible if p.flags.get("goaded_by") is not None}
        key = {(p.card_def.name, p.slot) for p in eligible}
        want = {frozenset(c) for r in range(len(key) + 1) for c in itertools.combinations(sorted(key), r) if must <= set(c)}
        observed, dead, _n = SR.explore(
            lambda seed=seed: _board(seed)[0], table, lambda s: False,
            lambda s: frozenset((p.card_def.name, p.slot) for p in s.players[0].attackers), terminal_moves=("Pass",), cap=3000,
            keep=lambda d: d == "Pass" or d.startswith("pointer "))  # declarations only: tapping a goaded creature first is a separate (legal) line
        assert not dead, (seed, dead[:2])
        assert set(observed) == want, (seed, sorted(map(sorted, set(observed) ^ want))[:3])
        done += 1
    assert done > 60


# ---- A6b: blockers -----------------------------------------------------------------------------------------------------

def test_blocker_masks_and_done_blocking_equal_the_rules():
    checked = 0
    for seed in SEEDS:
        state, table = _board(seed, attackers_declared=True)
        if state is None:
            continue
        attackers = state.players[0].attackers
        mine, theirs = state.players[1].battlefield, state.players[0].battlefield
        got = AB.pointer_legal_mask(state, [*mine, *theirs])
        want = [p.card_type == game.CardType.CREATURE and not p.tapped and any(_may_block(state, p, a) for a in attackers)
                for p in mine] + [False] * len(theirs)
        assert got == want, (seed, [(p.card_def.name, p.tapped) for p in mine])
        done_row = next(i for i, r in enumerate(table) if r[0] == "Done blocking")
        assert bool(drl_env.legal_action_mask(state, table)[done_row]), seed  # nothing blocked yet: no lone menace block
        checked += any(want)
        for blocker in [p for p, ok in zip(mine, want, strict=False) if ok]:  # the nested "which attacker" choice
            st, tb = _board(seed, attackers_declared=True)
            b2 = next(p for p in st.players[1].battlefield if (p.card_def.name, p.slot) == (blocker.card_def.name, blocker.slot))
            AB.execute_pointer_choice(st, b2)
            assert st.pending_resolution["kind"] == "choose_opponent_permanent"
            ids = [*st.players[1].battlefield, *st.players[0].battlefield]
            got2 = AB.pointer_legal_mask(st, ids)
            want2 = [False] * len(st.players[1].battlefield) + [
                _may_block(st, b2, a) and a in st.players[0].attackers for a in st.players[0].battlefield]
            assert got2 == want2, (seed, blocker.card_def.name)
    assert checked > 100


def _final_blocks(state):
    """The declaration as the engine would carry it into combat: abandoned lone menace blocks are dropped (enforce_menace)."""
    state.active_idx = 0  # combat damage runs with the attacker active; enforce_menace reads its blocked_by
    enforce_menace(state)
    return frozenset(((b.card_def.name, b.slot), (a.card_def.name, a.slot))
                     for a, bs in state.players[0].blocked_by.items() for b in bs)


def _finish_normally(state):
    """A declaration that ends through "Done blocking" must already be legal: no menace attacker may hold exactly one
    blocker (509.1c). Checked BEFORE enforce_menace, which would otherwise quietly repair the mistake and hide it."""
    lone = [a.card_def.name for a, bs in state.players[0].blocked_by.items()
            if len(bs) == 1 and stats.has_keyword(state, a, "menace")]
    assert not lone, f"'Done blocking' let a declaration finish with a lone block on menace attacker(s) {lone}"
    return _final_blocks(state)


def test_done_blocking_is_refused_exactly_while_a_menace_attacker_has_a_lone_blocker():
    attacker = Permanent(game.CARD_DEFS["Generous Ent"])
    attacker.tapped = True
    attacker.temp_keywords.add("menace")
    b1, b2 = (Permanent(game.CARD_DEFS["Slippery Bogle"]) for _ in range(2))
    b1.slot, b2.slot = 1, 2
    table = AB.build_fixed_action_table([("Slippery Bogle", 4), ("Generous Ent", 4)])
    done_row = next(i for i, r in enumerate(table) if r[0] == "Done blocking")

    def position(blockers_on_attacker):
        state = GameState(on_the_play=True, players=[PlayerState(True), PlayerState(False)])
        state.players[0].battlefield = [attacker]
        state.players[0].attackers = [attacker]
        state.players[1].battlefield = [b1, b2]
        state.active_idx = 1
        game.begin_declare_blockers(state, on_complete=lambda s: None)
        state.players[0].blocked_by = {attacker: list(blockers_on_attacker)} if blockers_on_attacker else {}
        return state

    assert drl_env.legal_action_mask(position([]), table)[done_row], "no block at all is legal"
    assert not drl_env.legal_action_mask(position([b1]), table)[done_row], "a lone block on a menace attacker is not"
    assert drl_env.legal_action_mask(position([b1, b2]), table)[done_row], "two blockers satisfy menace"


def test_every_reachable_block_declaration_is_exactly_what_the_rules_allow():
    done = 0
    for seed in SEEDS:
        state, table = _board(seed, attackers_declared=True)
        if state is None:
            continue
        blockers = [p for p in state.players[1].battlefield if p.card_type == game.CardType.CREATURE and not p.tapped]
        if not 1 <= len(blockers) <= 3 or len(state.players[0].attackers) > 3:
            continue
        want = _legal_assignments(state)
        observed, dead, _n = SR.explore(
            lambda seed=seed: _board(seed, attackers_declared=True)[0], table,
            lambda s: s.pending_resolution is None, _finish_normally, cap=3000)
        # A declaration with no legal action left is ABANDONED by the engine (game.turn._declare_blockers_gen): a menace
        # attacker holding a lone block with no second blocker to add. It is judged after enforce_menace, like any other end.
        for prefix in dead:
            st = _board(seed, attackers_declared=True)[0]
            for want_move in prefix:
                next(a for d, a in SR.moves(st, table) if d == want_move)(st)
            assert st.pending_resolution["kind"] == "declare_blockers" and game.menace_block_incomplete(st), (seed, prefix)
            observed.append(_final_blocks(st))
        assert set(observed) == want, (seed, "reachable-not-legal", sorted(map(sorted, set(observed) - want))[:2],
                                       "legal-not-reachable", sorted(map(sorted, want - set(observed)))[:2])
        done += 1
    assert done > 40


# ---- A6c: damage among several blockers -------------------------------------------------------------------------------

def _damage_case(power, toughs, trample, marked, deathtouch):
    state = GameState(on_the_play=True, players=[PlayerState(True), PlayerState(False)])
    attacker = Permanent(game.CARD_DEFS["Generous Ent"])
    attacker.slot = 1
    if deathtouch:
        attacker.temp_keywords.add("deathtouch")
    blockers = []
    for i, _t in enumerate(toughs):
        b = Permanent(game.CARD_DEFS["Slippery Bogle"])
        b.slot = i + 1
        b.counters = {"+1/+1": toughs[i] - 1}
        b.damage_marked = marked[i]
        blockers.append(b)
    state.players[0].battlefield = [attacker]
    state.players[1].battlefield = blockers
    state.players[0].attackers = [attacker]
    state.active_idx = state.turn_player_idx = 0
    caps = blocker_lethal_capacities(state, attacker, blockers)  # the engine's own caps: what the mask will enforce
    game.begin_assign_combat_damage(state, attacker, blockers, power, trample, caps, on_complete=lambda s: None)
    return state, attacker, blockers, caps


@pytest.mark.parametrize("power,toughs,trample,marked,deathtouch", [
    (3, (1, 1), False, (0, 0), False), (3, (2, 2), False, (0, 0), False), (4, (1, 2, 3), False, (0, 0, 0), False),
    (6, (1, 2), False, (0, 0), False), (6, (1, 2), True, (0, 0), False), (5, (2, 1, 1), True, (0, 0, 0), False),
    (2, (3, 3), True, (0, 0), False), (5, (1, 1, 1), False, (0, 0, 0), False),
    # damage already marked makes a blocker cheaper to kill: lethal = toughness - marked
    (4, (3, 3), False, (2, 0), False), (5, (3, 3), True, (2, 1), False), (4, (4, 2), True, (3, 0), False),
    # deathtouch: one point is lethal to every blocker, whatever its toughness (702.2c)
    (4, (3, 3), True, (0, 0), True), (3, (5, 2, 4), False, (0, 0, 0), True),
])
def test_every_reachable_damage_split_is_exactly_what_the_model_allows(power, toughs, trample, marked, deathtouch):
    table = AB.build_fixed_action_table([("Slippery Bogle", 4), ("Generous Ent", 4)])

    def observe(state):
        split, opp = state.players[0].attackers[0].flags["combat_damage_split"]
        return tuple(split.get(b, 0) for b in state.players[1].battlefield) + (opp,)

    observed, dead, _n = SR.explore(lambda: _damage_case(power, toughs, trample, marked, deathtouch)[0], table,
                                    lambda s: s.pending_resolution is None, observe, cap=3000)
    assert not dead
    caps = [1 if deathtouch else t - m for t, m in zip(toughs, marked, strict=True)]  # 702.2c / 510.1: lethal damage
    total_cap = sum(caps)
    want = set()
    for amounts in itertools.product(*[range(c + 1) for c in caps]):
        if sum(amounts) == min(power, total_cap):
            if power > total_cap:  # everyone capped; leftover forced
                extra = power - total_cap
                if trample:
                    want.add((*amounts, extra))
                else:
                    want.add((*amounts[:-1], amounts[-1] + extra, 0))
            else:
                want.add((*amounts, 0))
    assert set(observed) == want, (sorted(set(observed) - want)[:3], sorted(want - set(observed))[:3])
