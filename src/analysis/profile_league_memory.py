"""Find what grows in the league PARENT process across training iterations.

Context (2026-08-19): a league session's parent grew ~600MB per deck-round
while its 6 collect workers stayed flat, reaching 17GB before the machine ran
out of memory. Worker-side collection is therefore NOT the cause; something in
the aggregating process retains memory across iterations.

A first pass with tracemalloc attributed ~570MB PER COLLECT, still live one
iteration later, to multiprocessing/connection.py's unpickle path -- tens of
millions of ~32-byte Python objects, NOT torch tensors (a tensor census showed
only 16MB, which is why a tensors-only profile would have come back clean).
But tracemalloc attributes to the ALLOCATION site, not the retainer: the
unpickler is merely where the objects were born. This pass measures the
suspected retainer directly instead.

Prime suspect: _run_session's mull_by_deck_accum. It is extended every
deck-round but drained only every MULLIGAN_UPDATE_EVERY (8) iterations, so
between flushes it holds every mulligan transition from every deck. That would
explain parent-only growth, a non-tensor payload, retention across iterations,
and the episodic climb-then-release step pattern the watchdog saw.

mull_by_deck_accum is a LOCAL of _run_session, so it is read by walking up the
stack from the patched callee rather than by editing production code.

    python -u analysis/profile_league_memory.py --n-iterations 6 --device cpu \
        --run-config ../training_configs/run_bench_cpu.json \
        --league-config ../training_configs/run_bench_cpu.json

Arguments are forwarded verbatim to run_league.main(), so this profiles the
REAL code path rather than a reimplementation of it.
"""
import collections
import gc
import inspect
import itertools
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch  # noqa: E402

import rl.league_runner as league_runner  # noqa: E402


def _deep_object_count(obj, depth=6):
    """Rough count of distinct Python objects reachable from `obj`. The leak is
    measured in OBJECT COUNT (tens of millions of ~32-byte items), so a count
    is the number that matters, not sys.getsizeof of the outer container."""
    seen = set()
    stack = [(obj, 0)]
    total = 0
    while stack:
        item, d = stack.pop()
        if id(item) in seen or d > depth:
            continue
        seen.add(id(item))
        total += 1
        if isinstance(item, dict):
            stack.extend((v, d + 1) for v in itertools.islice(item.values(), 200))
            stack.extend((k, d + 1) for k in itertools.islice(item.keys(), 200))
        elif isinstance(item, (list, tuple, set)):
            stack.extend((v, d + 1) for v in itertools.islice(item, 500))
        elif hasattr(item, "__dict__"):
            stack.extend((v, d + 1) for v in itertools.islice(vars(item).values(), 200))
    return total


def _session_locals():
    """_run_session's frame locals, found by walking up from the caller."""
    frame = inspect.currentframe()
    while frame is not None:
        if "mull_by_deck_accum" in frame.f_locals:
            return frame.f_locals
        frame = frame.f_back
    return None


def _tensor_bytes():
    total = 0
    for obj in gc.get_objects():
        try:
            if torch.is_tensor(obj):
                total += obj.numel() * obj.element_size()
        except Exception:
            continue
    return total


def main():
    state = {"n": 0}
    original = league_runner.collect_rollout_league_parallel

    def instrumented(*args, **kwargs):
        result = original(*args, **kwargs)
        state["n"] += 1
        gc.collect()

        print(f"\n[mem] ===== after parallel-collect #{state['n']} =====", flush=True)

        session = _session_locals()
        if session is None:
            print("[mem] could not locate _run_session frame", flush=True)
        else:
            accum = session.get("mull_by_deck_accum") or {}
            per_deck = {name: len(v) for name, v in accum.items()}
            total_transitions = sum(per_deck.values())
            print(f"[mem] mull_by_deck_accum: {total_transitions} transitions {per_deck}", flush=True)
            if total_transitions:
                sample = next((v[0] for v in accum.values() if v), None)
                if sample is not None:
                    per_item = _deep_object_count(sample)
                    print(f"[mem]   ~{per_item} objects per transition "
                          f"-> ~{total_transitions * per_item / 1e6:.1f}M objects held here", flush=True)
            # Anything else in the frame big enough to matter.
            for key in ("buffers_by_deck", "mull_by_deck", "worker_logs", "outcomes"):
                val = session.get(key)
                if isinstance(val, dict):
                    print(f"[mem]   {key}: " + str({k: len(v) for k, v in val.items()}), flush=True)
                elif isinstance(val, list):
                    print(f"[mem]   {key}: {len(val)} entries", flush=True)

        print(f"[mem] live torch tensors total {_tensor_bytes()/1e6:.0f}MB "
              f"| gc objects {len(gc.get_objects())}", flush=True)
        return result

    league_runner.collect_rollout_league_parallel = instrumented

    import run_league
    run_league.main()


if __name__ == "__main__":
    main()
