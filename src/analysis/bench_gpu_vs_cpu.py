"""GPU-vs-CPU A/B for rl.ppo.ppo_update, measured INSIDE a real league session.

Why this shape rather than the obvious "run a session twice, time both":
GPU and CPU float arithmetic differ slightly, so the two arms' weights diverge
after the very first update -> different actions -> different game LENGTHS ->
different amounts of work. By iteration 125 you would be comparing wall-clock
across two different workloads and calling the difference a speedup. That test
gets less trustworthy the longer it runs, which is the opposite of what a
benchmark should do.

So instead: monkeypatch ppo_update inside a REAL session and, on every call,
run BOTH arms from the IDENTICAL starting weights on the IDENTICAL buffer --
    1. GPU arm on a throwaway deepcopy of the net (timed, then discarded)
    2. CPU arm on the real net (timed, and it is the one that counts)
Only the CPU result is applied, so training proceeds exactly as it normally
would and every subsequent buffer stays on-distribution. Each pair is a
controlled, zero-divergence comparison of the same work on two devices.

The buffers are the real thing -- real board states, real token lists, real
sizes, real batch_size ramp values -- because they come from the real loop,
not from a synthetic generator.

TIMING CORRECTNESS. CUDA is asynchronous: without torch.cuda.synchronize()
around the timed region you measure kernel LAUNCHES, not kernel work, and the
GPU looks absurdly fast. Both syncs are mandatory, not defensive. CUDA context
init and autotune are paid in an explicit warmup before the session so they
land outside every measurement.

EQUIVALENCE CHECK. Each pair asserts the two arms agree on epochs_run and land
within tolerance on the returned losses. Without it a GPU arm that silently
early-stopped on target_kl after 1 epoch instead of 4 would look like a 4x
speedup. Disagreements are recorded, not swallowed.

WHAT THIS DELIBERATELY DOES NOT MEASURE: the per-iteration state_dict
broadcast to the collection workers (rl.rollout_parallel line ~184). That cost
exists only in the GPU arm -- GPU-resident nets need a device->host copy 500x
per session -- and is measured separately by --broadcast-only, then ADDED to
the GPU projection. Omitting it would flatter GPU dishonestly.

Usage:
  python analysis/bench_gpu_vs_cpu.py --iterations 30
  python analysis/bench_gpu_vs_cpu.py --broadcast-only
"""
import sys
import time
import copy
import json
import argparse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # src/

import torch

import rl.league_runner as league_runner
from rl.ppo import ppo_update as _real_ppo_update

RECORDS = []
_WARMED = False


def _sync():
    if torch.cuda.is_available():
        torch.cuda.synchronize()


_GPU_CACHE = {}  # id(cpu_net) -> (gpu_net, gpu_optimizer)


def _timed_gpu(net, optimizer, buf, kwargs):
    """One GPU update on a shadow copy -- never touches the real net.

    The shadow net/optimizer are built ONCE per deck and reused. Building a
    fresh deepcopy + Adam on every call (the first version of this) leaked
    host memory until the REAL arm could not allocate its own minibatch pad
    -- a 5,744-transition buffer needs a 304 MiB (5744, 92, 151) array, and
    it OOMed mid-session. Reuse keeps the harness at 4 shadow nets total.

    Reloading both state dicts before each timed run is what preserves the
    property the whole benchmark rests on: both arms start from byte-identical
    weights and optimizer moments, so they do the same work. Adam's
    load_state_dict casts state tensors onto the param's device, so the
    moments follow the params to the GPU -- a fresh-Adam GPU arm would skip
    bias-correction work the CPU arm is doing and time a strictly smaller job.
    """
    ent = _GPU_CACHE.get(id(net))
    if ent is None:
        gnet = copy.deepcopy(net).to("cuda")
        gopt = torch.optim.Adam([p for p in gnet.parameters() if p.requires_grad],
                                lr=optimizer.param_groups[0]["lr"])
        _GPU_CACHE[id(net)] = ent = (gnet, gopt)
    gnet, gopt = ent
    gnet.load_state_dict(net.state_dict())
    gopt.load_state_dict(optimizer.state_dict())
    _sync()
    t0 = time.perf_counter()
    out = _real_ppo_update(gnet, gopt, buf, "cuda", **kwargs)
    _sync()
    return time.perf_counter() - t0, out


def _patched(net, optimizer, buf, device, **kwargs):
    """Drop-in for rl.league_runner.ppo_update: times both arms, applies CPU."""
    global _WARMED
    gpu_t, gpu_out, gpu_err = None, None, None
    if torch.cuda.is_available():
        if not _WARMED:
            # Context init + autotune + (on a GPU arch whose cubins this wheel
            # does not ship, e.g. Blackwell sm_120 on an older cu wheel) PTX
            # JIT are all paid OUTSIDE any measurement. Timed and printed
            # because a multi-minute warmup is itself a finding.
            t_w = time.perf_counter()
            try:
                # ONE warmup update, not two. Two was a mistake: on this
                # hardware a GPU update is far slower than a CPU one, so each
                # extra warmup call costs minutes of pure overhead. One is
                # enough to pay context init (measured 0.2s) and autotune of
                # the kernels this net actually uses.
                _timed_gpu(net, optimizer, buf, kwargs)
                print(f"    [bench] cuda warmup {time.perf_counter() - t_w:.1f}s", flush=True)
            except Exception as exc:  # noqa: BLE001 -- recorded, not hidden
                gpu_err = f"warmup: {type(exc).__name__}: {exc}"
                print(f"    [bench] cuda warmup FAILED: {gpu_err}", flush=True)
            _WARMED = True
        if gpu_err is None:
            try:
                gpu_t, gpu_out = _timed_gpu(net, optimizer, buf, kwargs)
            except Exception as exc:  # noqa: BLE001
                gpu_err = f"{type(exc).__name__}: {exc}"

    # CPU arm: the REAL update. This is the one that mutates the net, so the
    # session advances exactly as an unbenchmarked run would.
    t0 = time.perf_counter()
    cpu_out = _real_ppo_update(net, optimizer, buf, device, **kwargs)
    cpu_t = time.perf_counter() - t0

    rec = {"n_transitions": len(buf), "batch_size": kwargs.get("batch_size"),
           "cpu_s": cpu_t, "gpu_s": gpu_t, "gpu_error": gpu_err,
           "cpu_epochs": cpu_out[5], "gpu_epochs": gpu_out[5] if gpu_out else None}
    if gpu_out is not None:
        # Same work? policy/value loss and epochs_run must agree. A GPU arm
        # that early-stopped after 1 epoch would otherwise read as a 4x win.
        rec["equivalent"] = (cpu_out[5] == gpu_out[5]
                             and abs(cpu_out[0] - gpu_out[0]) < 5e-2
                             and abs(cpu_out[1] - gpu_out[1]) < 5e-2)
        rec["cpu_policy_loss"], rec["gpu_policy_loss"] = cpu_out[0], gpu_out[0]
    RECORDS.append(rec)
    # Printed per call, not just dumped at the end: a slow arm has to be
    # visible WHILE the run is happening, otherwise a pathological GPU time is
    # indistinguishable from a hang for the whole length of the session.
    diag = ""
    if rec.get("equivalent") is False:
        # Print WHY, not just that it failed: an epochs_run mismatch means the
        # two arms ran different amounts of work (target_kl early-stopped on
        # one side) and the timing pair must be discarded, whereas a pure loss
        # drift at equal epochs is ordinary float non-determinism.
        diag = (f" [epochs {cpu_out[5]}vs{gpu_out[5]} "
                f"ploss {cpu_out[0]:+.4f}vs{gpu_out[0]:+.4f} "
                f"vloss {cpu_out[1]:.4f}vs{gpu_out[1]:.4f}]")
    print(f"    [bench {len(RECORDS):4d}] n={rec['n_transitions']:5d} bs={rec['batch_size']} "
          f"cpu={cpu_t:7.2f}s gpu={f'{gpu_t:7.2f}s' if gpu_t else '   --   '} "
          f"{'x%.2f' % (cpu_t / gpu_t) if gpu_t else ''} "
          f"eq={rec.get('equivalent')}{diag} {rec['gpu_error'] or ''}", flush=True)
    return cpu_out


def bench_broadcast(reps=200):
    """The cost that exists ONLY under GPU: extracting CPU state_dicts for the
    collection workers. rl.rollout_parallel rebuilds these on EVERY collect
    call -- n_iterations * len(train_decks) times per session -- so a
    GPU-resident net pays a device->host copy of every parameter each time."""
    from rl.pool import build_pool
    from rl import checkpoint as ckpt_io
    decklists, vocab, deck_ctxs, fixed_tables = build_pool()
    roster = ["rakdos_madness", "dmir_terror", "elves", "mono_red_rally"]
    cpu_nets, gpu_nets = {}, {}
    for name in roster:
        p = f"../checkpoints/4_deck_bench/{name}/live.pt"
        net = league_runner.build_deck_net(vocab.size, len(fixed_tables[name]),
                                           ckpt_io.trunk_hidden_from_deck_checkpoint(p))
        ckpt_io.load_deck_checkpoint(p, net)
        cpu_nets[name] = net
        if torch.cuda.is_available():
            gpu_nets[name] = copy.deepcopy(net).to("cuda")

    def _cpu_broadcast():
        return {n: net.state_dict() for n, net in cpu_nets.items()}

    def _gpu_broadcast():
        # What a GPU-resident league would have to do: pull every tensor back
        # to host so it can be pickled to the worker processes.
        return {n: {k: v.cpu() for k, v in net.state_dict().items()}
                for n, net in gpu_nets.items()}

    for fn in (_cpu_broadcast, _gpu_broadcast):  # warmup
        if gpu_nets or fn is _cpu_broadcast:
            fn()
    _sync()
    t0 = time.perf_counter()
    for _ in range(reps):
        _cpu_broadcast()
    cpu_s = (time.perf_counter() - t0) / reps
    gpu_s = None
    if gpu_nets:
        _sync()
        t0 = time.perf_counter()
        for _ in range(reps):
            _gpu_broadcast()
        _sync()
        gpu_s = (time.perf_counter() - t0) / reps
    return {"reps": reps, "cpu_broadcast_s": cpu_s, "gpu_broadcast_s": gpu_s}


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--iterations", type=int, default=30)
    p.add_argument("--out", type=str, default="../logs/bench_gpu_vs_cpu.json")
    p.add_argument("--broadcast-only", action="store_true")
    args = p.parse_args()

    print(f"torch {torch.__version__}  cuda={torch.cuda.is_available()}  "
          f"threads={torch.get_num_threads()}", flush=True)
    if torch.cuda.is_available():
        print(f"gpu: {torch.cuda.get_device_name(0)}", flush=True)

    if args.broadcast_only:
        out = bench_broadcast()
        print(json.dumps(out, indent=2))
        Path(args.out).write_text(json.dumps({"broadcast": out}, indent=2))
        return

    bcast = bench_broadcast()
    print(f"broadcast/call: cpu={bcast['cpu_broadcast_s']*1000:.1f}ms  "
          f"gpu={bcast['gpu_broadcast_s']*1000:.1f}ms", flush=True)

    league_runner.ppo_update = _patched  # league_runner did `from rl.ppo import ppo_update`
    import run_league
    sys.argv = ["run_league.py", "--n-iterations", str(args.iterations),
                "--run-config", "../training_configs/run_bench.json",
                "--league-config", "../training_configs/run_bench.json"]
    t0 = time.time()
    run_league.main()
    wall = time.time() - t0

    Path(args.out).write_text(json.dumps(
        {"records": RECORDS, "broadcast": bcast, "wall_s": wall,
         "iterations": args.iterations,
         "torch": torch.__version__, "threads": torch.get_num_threads(),
         "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None},
        indent=2))
    print(f"\nwrote {len(RECORDS)} paired records to {args.out} (wall {wall:.0f}s)")


if __name__ == "__main__":
    main()
