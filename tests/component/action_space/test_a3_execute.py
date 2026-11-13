"""Layer A3 -- execute every legal action.

At sampled decisions of real games, EVERY action the mask allows is executed on a throw-away fork of the live state
(`support/fork.py`; a state whose pending closures captured live objects is re-created by replaying its game instead).
For each branch:

  * it must not raise;
  * it must change the observable state (a legal action that does nothing is how declare-blockers once looped forever;
    `Pass` is excluded: it is carried out by the turn loop, not by its closure);
  * the position after it must still have a legal action (no dead end), unless the game is over;
  * its documented post-condition holds (a land play moves a land from hand to battlefield, a tap adds mana, a pool
    spend takes the pip, an attack declaration puts the creature in the attackers);
  * the mask on the NEW position agrees with the independent oracle -- branches reach positions random play does not
    (mid-payment after an unusual tap, a half-made cast), which widens A2's coverage.

The fork is checked too: its mask equals the original's, and a replayed fork fingerprints identically to the position it
re-creates.
"""
import os
from collections import Counter

import numpy as np
import pytest

import game
import rl.decision.agent as A
from rl.decision import action_bridge as AB
from support import action_corpus as AC
from support.fingerprint import state_fingerprint
from support.fork import fork_state, is_fork_safe
from support.paths import SRC

EVERY = int(os.environ.get("ACTION_FORK_EVERY", "8"))   # branch every Nth decision (per game, staggered)
MAX_FAILS = 25


def _settle(f):
    """What the turn loop does between two actions, so the next position is one the loop could actually present."""
    if f.casting_depth <= 0:
        game.turn.check_state_based_actions(f)
        game.turn.refizzle_if_now_targetless(f)
    if f.pending_resolution is None:
        game.promote_triggers_to_stack(f)


def _zone_count(zone, name):
    return sum(1 for c in zone if getattr(c, "name", getattr(getattr(c, "card_def", None), "name", None)) == name)


def _post_condition(label, ptr_kind, chosen, before, f, seat):
    """None if the documented effect of this action shows in the new state, else a description of what is missing."""
    me1 = f.players[seat]
    if label and label.startswith("Play land: "):
        name = label[len("Play land: "):]
        ok = (before["hand"].count(name) - 1 == _zone_count(me1.hand, name)
              and before["bf"].count(name) + 1 == _zone_count(me1.battlefield, name)
              and me1.lands_played_this_turn == before["lands"] + 1)
        return None if ok else f"land {name!r} did not move hand -> battlefield / count a land drop"
    if label and label.startswith("Tap "):
        grew = sum(me1.mana_pool.values()) > before["pool"]
        opened = f.active_mana_subdecision is not None
        return None if grew or opened else "tap added no mana and opened no sub-decision"
    if label and label.startswith("Spend ") and label.endswith(" from pool"):
        color = label.split()[1]
        return None if me1.mana_pool.get(color, 0) == before["pool_by"].get(color, 0) - 1 else f"{color} pip was not spent"
    if ptr_kind == "declare_attackers" and chosen is not None:
        return None if any(a is chosen for pl in f.players for a in pl.attackers) else "chosen creature is not attacking"
    return None


@pytest.fixture(scope="module")
def branch_run():
    prev = os.getcwd()
    os.chdir(SRC)
    fails = {k: [] for k in ("exception", "noop", "dead_end", "post_condition", "post_oracle", "fork_mask", "fork_fidelity")}
    n = Counter()
    by_label_family = Counter()

    def note(kind, **info):
        n["fail_" + kind] += 1
        if len(fails[kind]) < MAX_FAILS:
            fails[kind].append(info)

    def on_decision(state, seat, deck, dec, legal, ctxs, tables):
        cur = AC.CURRENT
        if (cur.count + cur.pi) % EVERY:
            return
        n["states"] += 1
        safe = is_fork_safe(state)
        n["unsafe_states"] += 0 if safe else 1
        before_fp = state_fingerprint(state)
        labels = [r[0] for r in dec.fixed_table]
        me = state.players[seat]
        before = {"me": me, "hand": [c.name for c in me.hand], "bf": [p.card_def.name for p in me.battlefield],
                  "lands": me.lands_played_this_turn, "pool": sum(me.mana_pool.values()), "pool_by": dict(me.mana_pool)}
        base = AC.context(state, seat, deck)
        for idx in (int(i) for i in legal):
            label = labels[idx] if idx < dec.n_fixed else None
            if label == "Pass":
                n["pass_skipped"] += 1
                continue
            if safe:
                f = fork_state(state)
                seat_f, deck_f = seat, deck
                dec_f = A._build_decision(f, seat, ctxs[deck], 10 ** 9)
            else:
                f, seat_f, deck_f, dec_f = AC.replay_fork(cur, cur.count)
                if state_fingerprint(f) != before_fp:
                    note("fork_fidelity", label=label, **base)
                    continue
            n["branches"] += 1
            if not np.array_equal(dec_f.full_mask, dec.full_mask):
                note("fork_mask", label=label, **base)
                continue
            chosen = dec_f.identities[idx - dec_f.n_fixed] if idx >= dec_f.n_fixed else None
            ptr_kind = AB.pointer_kind(f) if idx >= dec_f.n_fixed else None
            by_label_family[AC.O.family(label, AC._extra_tap_names()) if label else "pointer:" + str(ptr_kind)] += 1
            try:
                A._executor_for(f, idx, dec_f.fixed_table, dec_f.identities)()
                _settle(f)
            except Exception as e:  # noqa: BLE001
                note("exception", label=label or f"pointer:{ptr_kind}", error=repr(e)[:300], **base)
                continue
            if state_fingerprint(f) == before_fp:
                note("noop", label=label or f"pointer:{ptr_kind}", **base)
            problem = _post_condition(label, ptr_kind, chosen, before, f, seat)
            if problem:
                note("post_condition", label=label or f"pointer:{ptr_kind}", problem=problem, **base)
            if f.winner is not None or f.turn_won is not None or f.decked_out:
                continue
            seat2 = f.active_idx
            deck2 = cur.seats[seat2]
            dec2 = A._build_decision(f, seat2, ctxs[deck2], 10 ** 9)
            if dec2.n_legal == 0:
                note("dead_end", label=label or f"pointer:{ptr_kind}", **base)
                continue
            labels2 = [r[0] for r in dec2.fixed_table]
            mask2 = {labels2[i] for i in np.flatnonzero(dec2.full_mask) if i < dec2.n_fixed}
            for kind, lab, finding in AC.diff_against_oracle(f, labels2, mask2):
                n["post_oracle_" + ("explained" if finding else "unexplained")] += 1
                if finding is None:
                    note("post_oracle", kind=kind, label=lab, after=label or f"pointer:{ptr_kind}", **AC.context(f, seat2, deck2))

    try:
        played, errors = AC.play_games(on_decision, games_per_pair=1, seed=23)
    finally:
        os.chdir(prev)
    return fails, n, errors, by_label_family


def _show(items):
    return "\n".join(str(x)[:500] for x in items[:3])


def test_the_run_was_substantial(branch_run):
    _fails, n, errors, fam = branch_run
    assert not errors, errors[:3]
    assert n["states"] >= 3000 and n["branches"] >= 2500, dict(n)
    for needed in ("play_land", "cast", "tap", "spend", "activate"):
        assert fam[needed] >= 30, (needed, dict(fam))


def test_unsafe_states_are_replayed_not_copied(branch_run):
    """~1.7% of states hold closures over live objects; the run must have met some and handled them by replay."""
    n = branch_run[1]
    assert n["unsafe_states"] >= 1, "no unsafe state met -- the fork-safety gate was never exercised"
    assert not branch_run[0]["fork_fidelity"], _show(branch_run[0]["fork_fidelity"])


def test_a_fork_has_the_same_mask_as_the_position_it_copies(branch_run):
    assert not branch_run[0]["fork_mask"], _show(branch_run[0]["fork_mask"])


def test_every_legal_action_executes_without_raising(branch_run):
    assert not branch_run[0]["exception"], _show(branch_run[0]["exception"])


def test_no_legal_action_is_a_no_op(branch_run):
    assert not branch_run[0]["noop"], _show(branch_run[0]["noop"])


def test_no_legal_action_leads_to_a_dead_end(branch_run):
    assert not branch_run[0]["dead_end"], _show(branch_run[0]["dead_end"])


def test_post_conditions_hold(branch_run):
    assert not branch_run[0]["post_condition"], _show(branch_run[0]["post_condition"])


def test_the_mask_agrees_with_the_oracle_on_positions_reached_by_every_action(branch_run):
    assert not branch_run[0]["post_oracle"], _show(branch_run[0]["post_oracle"])


# ---- the fork gate itself ---------------------------------------------------------------------------------------------

class _Box:
    def __init__(self, **kw):
        self.__dict__.update(kw)


def test_fork_gate_flags_closures_that_captured_live_state():
    zone = [1, 2, 3]
    by_cell = _Box(zone=zone, cb=(lambda: zone))                      # closure cell
    by_default = _Box(zone=zone, cb=(lambda z=zone: z))              # default argument (the Flashback shape)
    by_kwdefault = _Box(zone=zone, cb=(lambda *, z=zone: z))         # keyword default
    nested = _Box(zone=zone, cb=(lambda: (lambda z=zone: z)))        # captured by a function a closure holds
    for box in (by_cell, by_default, by_kwdefault, nested):
        assert not is_fork_safe(box)
    harmless = _Box(zone=zone, cb=(lambda n=3: n + 1))
    assert is_fork_safe(harmless)
    assert fork_state(harmless).zone == zone and fork_state(harmless).zone is not zone
