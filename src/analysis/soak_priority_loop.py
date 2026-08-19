"""Sampled soak that catches the unbounded-priority-round hang IN THE ACT.

The greedy --eval logs are clean (analysis/scan_action_loops.py: worst
same-turn run 61 decisions). Training does NOT play greedy -- it SAMPLES from
the policy -- so a loop reachable only off the sampled tail would never appear
there. This runs the sampled path and instruments it.

Detection, not timeout: every decision is tagged with state.turn_number. A
healthy turn resolves in tens of decisions. When one turn passes
--warn-decisions, the wrapper dumps what the game is actually doing -- phase,
pending_resolution kind, whose priority, stack depth, battlefield size, mana
pool -- for the whole run of decisions in that turn, and (with --abort) raises
so the process stops with a real traceback instead of eating RAM.

Why tag on turn_number rather than wall time: the loop's defining property is
that turn_number STOPS ADVANCING while decisions continue (game/turn.py's
inner `while True` resets consecutive_passes on every non-pass action and the
actor keeps priority, so the round never ends and horizon -- which bounds
turns only -- never fires). A slow turn is not the bug; a turn that never ends
is.

Usage:
  python analysis/soak_priority_loop.py --rounds 200 --warn-decisions 400
  python analysis/soak_priority_loop.py --deck elves --abort
"""
import sys
import time
import random
import argparse
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # src/

from repo_paths import CHECKPOINTS_DIR
from rl.pool import build_pool
from rl.league import LeaguePool
from rl.agent import SeatAgent
from rl.mulligan import MulliganNet
from rl.rewards import deploy_reward_v6
from rl.train import collect_rollout_league
from rl.league_runner import HORIZON, build_deck_net
from game.effects.combat import creature_block_eligible as _block_eligible
import drl_env
import torch
from rl import checkpoint as ckpt_io

ROSTER = ["rakdos_madness", "dmir_terror", "elves", "mono_red_rally"]
HITS = []
CALLS = Counter()


def _count_calls():
    """Wrap the two functions that decide whether declare-blockers ENDS.

    "Done blocking" being the only legal action while the step never ends
    means either complete_resolution is not clearing the pending, or
    something re-opens it immediately. Counting both distinguishes those two
    without guessing: begin >> complete means a re-open loop; begin ==
    complete means Done is firing and being undone."""
    import game as _g
    for name in ("begin_declare_blockers", "complete_resolution"):
        real = getattr(_g, name)
        def make(real=real, name=name):
            def wrapped(*a, **kw):
                CALLS[name] += 1
                return real(*a, **kw)
            return wrapped
        setattr(_g, name, make())


def _kw(state, perm):
    """Keywords actually in effect on this permanent right now (intrinsic +
    granted), via the same accessor combat.can_block consults -- reading the
    CardDef's static extra dict instead would miss aura/static/temp grants and
    is what made an earlier pass of this investigation wrongly conclude no
    creature in the roster had reach."""
    try:
        from game.effects.stats import creature_keywords
        return set(creature_keywords(state, perm))
    except Exception:  # noqa: BLE001
        return set()


def _snap(state):
    """One line of what the engine is doing right now.

    blocked_by is the load-bearing field for the declare-blockers loop: if it
    is IDENTICAL on every iteration the defender is re-submitting a no-op, and
    if it oscillates the defender is toggling an assignment back and forth.
    Those are different bugs with different fixes, and the state dump cannot
    tell them apart without it."""
    pend = state.pending_resolution
    try:
        blocked = tuple(sorted(
            (a.card_def.name, a.slot, tuple(sorted((b.card_def.name, b.slot) for b in bs)))
            for a, bs in (state.opponent.blocked_by or {}).items()))
    except Exception:  # noqa: BLE001 -- diagnostics must never mask the bug they hunt
        blocked = ("<unavailable>",)
    return {
        "phase": getattr(getattr(state, "phase", None), "value", None),
        "pending": pend["kind"] if pend else None,
        "active": state.active_idx,
        "turn_player": state.turn_player_idx,
        "stack": len(getattr(state, "stack", ()) or ()),
        "battlefield": len(getattr(state, "battlefield", ()) or ()),
        "pool": sum((getattr(state, "mana_pool", None) or {}).values()) if getattr(state, "mana_pool", None) else 0,
        "blocked_by": blocked,
        "eligible_blockers": sum(1 for p in state.battlefield if _block_eligible(state, p)),
        "attackers": tuple(sorted(
            (a.card_def.name, tuple(sorted(_kw(state, a))))
            for a in (state.opponent.attackers or ()))),
        "eligible": tuple(sorted(
            (p.card_def.name, p.slot, tuple(sorted(_kw(state, p))))
            for p in state.battlefield if _block_eligible(state, p))),
        "n_begin": CALLS["begin_declare_blockers"],
        "n_complete": CALLS["complete_resolution"],
    }


def make_patched_decide(real_decide, warn_at, abort):
    """SeatAgent.decide replacement that counts decisions WITHIN one turn.

    Counter state is keyed by (id(agent), turn_number) rather than held per
    instance: collect_rollout_league builds its own SeatAgents per game, so
    there is no instance to attach to from out here, and a plain global
    counter would pool two seats' decisions together and false-positive on
    ordinary long turns."""
    state_by_agent = {}

    def decide(self, state, seat, horizon, device, greedy=False):
        st = state_by_agent.get(id(self))
        t = state.turn_number
        if st is None or st["turn"] != t:
            st = state_by_agent[id(self)] = {"turn": t, "n": 0, "trail": [], "warned": False}
        st["n"] += 1
        if st["n"] > warn_at - 60:          # keep only the tail, not the whole run
            st["trail"].append(_snap(state))
            del st["trail"][:-60]
        if st["n"] >= warn_at and not st["warned"]:
            st["warned"] = True
            hit = {"turn": t, "decisions": st["n"],
                   "phases": Counter(s["phase"] for s in st["trail"]).most_common(5),
                   "pendings": Counter(str(s["pending"]) for s in st["trail"]).most_common(5),
                   "active_flips": len({s["active"] for s in st["trail"]}),
                   "stack": Counter(s["stack"] for s in st["trail"]).most_common(3),
                   "battlefield": Counter(s["battlefield"] for s in st["trail"]).most_common(3),
                   "pool": Counter(s["pool"] for s in st["trail"]).most_common(3),
                   "eligible_blockers": Counter(s["eligible_blockers"] for s in st["trail"]).most_common(3),
                   "distinct_blocked_by": len({s["blocked_by"] for s in st["trail"]}),
                   "blocked_by_sample": [s["blocked_by"] for s in st["trail"][:2]],
                   "n_legal": Counter(s.get("n_legal") for s in st["trail"]).most_common(3),
                   "legal_sets": Counter(s.get("legal") for s in st["trail"]).most_common(3),
                   "chosen_idx": Counter(s.get("chosen") for s in st["trail"]).most_common(5),
                   "begin_delta_over_window": (st["trail"][-1]["n_begin"] - st["trail"][0]["n_begin"]),
                   "complete_delta_over_window": (st["trail"][-1]["n_complete"] - st["trail"][0]["n_complete"]),
                   "window_decisions": len(st["trail"]),
                   "attackers": Counter(s.get("attackers") for s in st["trail"]).most_common(2),
                   "eligible": Counter(s.get("eligible") for s in st["trail"]).most_common(2)}
            HITS.append(hit)
            print(f"\n  *** LOOP SUSPECT: turn {t}, {st['n']} decisions in ONE turn ***", flush=True)
            for k, v in hit.items():
                print(f"        {k}: {v}", flush=True)
            if abort:
                raise RuntimeError(f"priority round exceeded {warn_at} decisions in turn {t}")
        try:
            fixed_table = self.deck_ctx[1]
            mask = drl_env.legal_action_mask(state, fixed_table)
            legal = [fixed_table[i][0] for i, ok in enumerate(mask) if ok]
        except Exception as exc:  # noqa: BLE001
            legal = [f"<mask failed: {type(exc).__name__}: {exc}>"]
        out = real_decide(self, state, seat, horizon, device, greedy=greedy)
        if st["n"] > warn_at - 60:
            st["trail"][-1]["legal"] = tuple(legal[:8])
            st["trail"][-1]["n_legal"] = len(legal)
            st["trail"][-1]["chosen"] = getattr(out, "action_idx", None)
        return out

    return decide


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--league", default="4_deck_subleague_test")
    p.add_argument("--deck", default=None)
    p.add_argument("--games", type=int, default=24)
    p.add_argument("--rounds", type=int, default=200)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--warn-decisions", type=int, default=400)
    p.add_argument("--abort", action="store_true", help="Raise on the first suspect (default: log and continue).")
    args = p.parse_args()

    # Seed TORCH, not just the python rng: collection samples from the policy
    # via torch, so leaving torch's global generator unseeded makes every run
    # different even at a fixed --seed. That is why an identical command found
    # 2 loop suspects once and 0 the next time -- the loop is stochastic, and
    # without this the soak cannot be replayed to confirm a fix.
    torch.manual_seed(args.seed)
    _count_calls()
    decklists, vocab, deck_ctxs, fixed_tables = build_pool()
    league_dir = str(CHECKPOINTS_DIR / args.league)
    live_nets, mulligan_nets = {}, {}
    for name in ROSTER:
        path = f"{league_dir}/{name}/live.pt"
        net = build_deck_net(vocab.size, len(fixed_tables[name]),
                             ckpt_io.trunk_hidden_from_deck_checkpoint(path))
        ckpt_io.load_deck_checkpoint(path, net)
        net.eval()
        live_nets[name] = net
        mnet = MulliganNet(net.encoder)
        ckpt_io.load_deck_checkpoint(f"{league_dir}/{name}/mulligan.pt", mnet)
        mnet.eval()
        mulligan_nets[name] = mnet

    # collect_rollout_league builds its own SeatAgents internally, so patch the
    # CLASS rather than an instance -- every seat in every game gets counted.
    SeatAgent.decide = make_patched_decide(SeatAgent.decide, args.warn_decisions, args.abort)

    pool = LeaguePool(league_dir, ROSTER)
    rng = random.Random(args.seed)
    decks = [args.deck] if args.deck else ROSTER
    # collect_rollout_league exposes no greedy switch at all: league collection
    # is always sampled. That asymmetry with the greedy --eval logs is the
    # reason this soak exists.
    mode = "SAMPLED (what training does)"
    print(f"soak: {args.rounds} rounds x {args.games} games, {mode}, "
          f"warn at {args.warn_decisions} decisions/turn, horizon={HORIZON}", flush=True)

    t_start = time.time()
    for rnd in range(args.rounds):
        for name in decks:
            t0 = time.time()
            bufs, _m, played, _o = collect_rollout_league(
                name, live_nets, mulligan_nets, deck_ctxs, decklists, pool, deploy_reward_v6,
                HORIZON, args.games, rng, device="cpu",
                checkpoint_rate=0.15, pfsp=True)
            big = max((len(b) for b in bufs.values()), default=0)
            flag = "  <-- OVERSIZED" if big > 12000 else ""
            print(f"  round {rnd:4d} [{name:16}] {played:3d}g {time.time() - t0:6.1f}s "
                  f"largest_buf={big:7d} suspects={len(HITS)}{flag}", flush=True)
    print(f"\ndone in {time.time() - t_start:.0f}s; {len(HITS)} loop suspect(s)", flush=True)


if __name__ == "__main__":
    main()
