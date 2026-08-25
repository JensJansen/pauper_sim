"""GPU-vs-CPU A/B for rl.training.ppo.ppo_update, measured inside a real league session.

Monkeypatches ppo_update so every call runs both arms from identical starting
weights on the identical buffer: a GPU arm on a throwaway deepcopy of the net
(timed, then discarded), and a CPU arm on the real net (timed, and applied).
Only the CPU result changes the net, so training proceeds normally and every
subsequent buffer stays on-distribution. Buffers come from the real training
loop, not a synthetic generator.

CUDA is synchronized around each timed region (CUDA calls are async). CUDA
context init and kernel autotune are paid in an explicit warmup before the
session so they land outside every measurement.

Each pair asserts the two arms agree on epochs_run and land within tolerance
on the returned losses; disagreements are recorded, not swallowed.

Does not measure the per-iteration state_dict broadcast to collection workers
(rl.training.rollout_parallel) -- GPU-resident nets pay a device->host copy of
every parameter each session. That cost is measured separately via
--broadcast-only and added to the GPU projection.

Usage:
  python analysis/eval/bench_gpu_vs_cpu.py --iterations 30
  python analysis/eval/bench_gpu_vs_cpu.py --broadcast-only
"""
import sys
import time
import copy
import json
import argparse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))  # src/

import torch

import rl.league.league_runner as league_runner
from rl.training.ppo import ppo_update as _real_ppo_update

RECORDS = []
_WARMED = False


def _sync():
    if torch.cuda.is_available():
        torch.cuda.synchronize()


_GPU_CACHE = {}  # id(cpu_net) -> (gpu_net, gpu_optimizer)


def _timed_gpu(net, optimizer, buf, kwargs):
    """One GPU update on a shadow copy -- never touches the real net.

    The shadow net/optimizer are built once per deck and reused. Both state
    dicts are reloaded from the CPU net/optimizer before each timed run, so
    both arms start from identical weights and optimizer moments and do the
    same work.
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
    """Drop-in for rl.league.league_runner.ppo_update: times both arms, applies CPU."""
    global _WARMED
    gpu_t, gpu_out, gpu_err = None, None, None
    if torch.cuda.is_available():
        if not _WARMED:
            # Pays context init, PTX JIT, and kernel autotune outside any measurement.
            t_w = time.perf_counter()
            try:
                _timed_gpu(net, optimizer, buf, kwargs)
                print(f"    [bench] cuda warmup {(time.perf_counter() - t_w) * 1000:,.0f}ms", flush=True)
            except Exception as exc:  # noqa: BLE001 -- recorded, not hidden
                gpu_err = f"warmup: {type(exc).__name__}: {exc}"
                print(f"    [bench] cuda warmup FAILED: {gpu_err}", flush=True)
            _WARMED = True
        if gpu_err is None:
            try:
                gpu_t, gpu_out = _timed_gpu(net, optimizer, buf, kwargs)
            except Exception as exc:  # noqa: BLE001
                gpu_err = f"{type(exc).__name__}: {exc}"

    # CPU arm: the real update; mutates the net so the session advances normally.
    t0 = time.perf_counter()
    cpu_out = _real_ppo_update(net, optimizer, buf, device, **kwargs)
    cpu_t = time.perf_counter() - t0

    rec = {"n_transitions": len(buf), "batch_size": kwargs.get("batch_size"),
           "cpu_ms": cpu_t * 1000, "gpu_ms": gpu_t * 1000 if gpu_t else None, "gpu_error": gpu_err,
           "cpu_epochs": cpu_out[5], "gpu_epochs": gpu_out[5] if gpu_out else None}
    if gpu_out is not None:
        # Both arms must agree on epochs_run and loss within tolerance to be comparable.
        rec["equivalent"] = (cpu_out[5] == gpu_out[5]
                             and abs(cpu_out[0] - gpu_out[0]) < 5e-2
                             and abs(cpu_out[1] - gpu_out[1]) < 5e-2)
        rec["cpu_policy_loss"], rec["gpu_policy_loss"] = cpu_out[0], gpu_out[0]
    RECORDS.append(rec)  # printed per call so a slow/hung GPU arm is visible live
    diag = ""
    if rec.get("equivalent") is False:
        diag = (f" [epochs {cpu_out[5]}vs{gpu_out[5]} "
                f"ploss {cpu_out[0]:+.4f}vs{gpu_out[0]:+.4f} "
                f"vloss {cpu_out[1]:.4f}vs{gpu_out[1]:.4f}]")
    print(f"    [bench {len(RECORDS):4d}] n={rec['n_transitions']:5d} bs={rec['batch_size']} "
          f"cpu={cpu_t * 1000:8,.0f}ms gpu={f'{gpu_t * 1000:8,.0f}ms' if gpu_t else '    --    '} "
          f"{'x%.2f' % (cpu_t / gpu_t) if gpu_t else ''} "
          f"eq={rec.get('equivalent')}{diag} {rec['gpu_error'] or ''}", flush=True)
    return cpu_out


def bench_broadcast(reps=200):
    """Cost, GPU-only, of extracting CPU state_dicts for the collection
    workers -- rl.training.rollout_parallel rebuilds these on every collect
    call, so a GPU-resident net pays a device->host copy of every parameter
    each time."""
    from rl.roster import build_pool
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
        # Device->host copy of every tensor, needed to pickle to worker processes.
        return {n: {k: v.cpu() for k, v in net.state_dict().items()}
                for n, net in gpu_nets.items()}

    for fn in (_cpu_broadcast, _gpu_broadcast):  # warmup
        if gpu_nets or fn is _cpu_broadcast:
            fn()
    _sync()
    t0 = time.perf_counter()
    for _ in range(reps):
        _cpu_broadcast()
    cpu_ms = (time.perf_counter() - t0) / reps * 1000
    gpu_ms = None
    if gpu_nets:
        _sync()
        t0 = time.perf_counter()
        for _ in range(reps):
            _gpu_broadcast()
        _sync()
        gpu_ms = (time.perf_counter() - t0) / reps * 1000
    return {"reps": reps, "cpu_broadcast_ms": cpu_ms, "gpu_broadcast_ms": gpu_ms}


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
    print(f"broadcast/call: cpu={bcast['cpu_broadcast_ms']:.1f}ms  "
          f"gpu={bcast['gpu_broadcast_ms']:.1f}ms", flush=True)

    league_runner.ppo_update = _patched  # league_runner did `from rl.training.ppo import ppo_update`
    import run_league
    sys.argv = ["run_league.py", "--n-iterations", str(args.iterations),
                "--run-config", "../training_configs/benchmarking_league.json",
                "--league-config", "../training_configs/benchmarking_league.json"]
    t0 = time.time()
    run_league.main()
    wall_ms = (time.time() - t0) * 1000

    Path(args.out).write_text(json.dumps(
        {"records": RECORDS, "broadcast": bcast, "wall_ms": wall_ms,
         "iterations": args.iterations,
         "torch": torch.__version__, "threads": torch.get_num_threads(),
         "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None},
        indent=2))
    print(f"\nwrote {len(RECORDS)} paired records to {args.out} (wall {wall_ms:,.0f}ms)")


if __name__ == "__main__":
    main()
