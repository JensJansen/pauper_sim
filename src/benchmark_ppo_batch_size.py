"""One-off diagnostic (not part of the training pipeline): sweeps
ppo_update's batch_size on CPU vs GPU to find the approximate crossover
where GPU stops losing to CPU. Motivated by the earlier CPU-vs-GPU
benchmark, which only ever tested batch_size=64 (production's default) --
that test found CPU 2.4x faster on a real ~4400-transition buffer, but
never checked whether a MUCH bigger minibatch (fewer, larger kernel
launches -- the actual lever that favors GPU) changes that outcome.

Run ONLY when nothing else is consuming the CPU's physical cores (e.g.
after any other run_league.py session finishes) -- CPU-side numbers
collected under contention are not trustworthy for this comparison.

Tests at TWO total buffer sizes (a "typical" single-update buffer, ~2000
transitions, and a "large" one, ~8000, simulating deliberately
accumulating several iterations' worth before updating) to check whether
the crossover point itself depends on total data volume, not just
minibatch size -- it shouldn't, in theory (more minibatches is a roughly
linear multiplier on both devices), but that's exactly the kind of
assumption worth actually checking rather than trusting.

The "large" buffer is built by concatenating the real collected buffer
with itself, not synthesized from scratch -- preserves real per-token
shape/padding characteristics (variable token counts per game state)
while giving enough volume to test large batch sizes meaningfully;
duplicated content doesn't affect timing, which depends on shapes/counts,
not values.

Every GPU timing includes a full CPU->GPU->CPU round trip (net.to(device)
+ move_optimizer_state both ways), not just the update itself -- rollout
collection always needs the net back on CPU afterward (batch-of-1
inference stays CPU regardless of this schedule, a separate, already-
settled finding), so an idealized "already resident on the target device"
number would understate what mechanically switching actually costs in
run_league.py's real iteration loop."""

import random
import time

import torch

from rewards import action_count_win_reward_200_floor02
from terminated import never_terminated
from token_arch import SetTransformer
from token_deck import DeckNetwork
from token_league import LeaguePool
from token_pool import build_pool
from token_train import RolloutBuffer, collect_rollout_league, move_optimizer_state, ppo_update

CHECKPOINT_DIR = "../checkpoints"
FROZEN_STACK = f"{CHECKPOINT_DIR}/shared_stack_frozen.pt"
LEAGUE_DIR = f"{CHECKPOINT_DIR}/league"
D_MODEL = 64
TRAINING_DECK = "mono_red_madness"
HORIZON = 120
# Starts at 128, not 32/64: the earlier benchmark already settled that CPU
# wins decisively at small batch sizes, and those exact cells are also the
# slowest to run (batch_size=32 on the 8000-transition large buffer x 4
# epochs = ~1000 forward+backward passes for ONE cell -- the single most
# expensive, least informative combination). The crossover, if any, lives
# in the large-minibatch range where GPU's fewer/bigger kernel launches
# can finally amortize their overhead -- that's what this sweep targets.
BATCH_SIZES = [128, 256, 512, 1024, 2048, 4096]
N_EPOCHS = 4  # matches production's ppo_update default -- crossover is per-minibatch, but keep this realistic anyway
CUDA_AVAILABLE = torch.cuda.is_available()


def load_frozen_stack(vocab_size):
    ckpt = torch.load(FROZEN_STACK, weights_only=True)
    assert ckpt["vocab_size"] == vocab_size
    shared = SetTransformer(vocab_size, d_model=ckpt["d_model"], n_heads=4, n_layers=2, dim_feedforward=128)
    shared.load_state_dict(ckpt["shared"])
    for p in shared.parameters():
        p.requires_grad = False
    shared.eval()
    return shared


def duplicate_buffer(buf, times):
    out = RolloutBuffer()
    for _ in range(times):
        for i in range(len(buf)):
            out.add(buf.token_lists[i], buf.scalar[i], buf.mask[i], buf.action[i],
                    buf.logp[i], buf.value[i], buf.reward[i], buf.done[i])
    return out


def time_update(net, buf, device, batch_size):
    """Times the REAL production sequence, not an idealized one: rollout
    collection always leaves the net on CPU (device_for_batch_size's own
    docstring -- collection stays CPU regardless of this schedule, that's
    settled and unrelated to batch size), so a GPU update always pays a
    CPU->GPU move before AND a GPU->CPU move after (the net must be back
    on CPU for the very next collection call). Timing only the update
    itself, as if the net were already resident on the target device,
    would understate the real cost of switching and could pick too low a
    threshold for run_league.py's own mechanical switch."""
    net.to("cpu")
    optimizer = torch.optim.Adam([p for p in net.parameters() if p.requires_grad], lr=3e-4)

    # Untimed warm-up -- CUDA context/kernel JIT on first real call to this
    # device, done with its own (untimed) move so it doesn't contaminate
    # the timed round-trip below.
    warm = RolloutBuffer()
    for i in range(min(8, len(buf))):
        warm.add(buf.token_lists[i], buf.scalar[i], buf.mask[i], buf.action[i],
                  buf.logp[i], buf.value[i], buf.reward[i], buf.done[i])
    net.to(device)
    move_optimizer_state(optimizer, device)
    ppo_update(net, [optimizer], warm, device, n_epochs=1, batch_size=min(batch_size, len(warm)))
    net.to("cpu")
    move_optimizer_state(optimizer, "cpu")

    if device == "cuda":
        torch.cuda.synchronize()
    t0 = time.time()
    net.to(device)
    move_optimizer_state(optimizer, device)
    ppo_update(net, [optimizer], buf, device, n_epochs=N_EPOCHS, batch_size=batch_size)
    net.to("cpu")
    move_optimizer_state(optimizer, "cpu")
    if device == "cuda":
        torch.cuda.synchronize()
    return time.time() - t0


def main():
    if not CUDA_AVAILABLE:
        print("CUDA not available -- this benchmark needs both devices to compare.", flush=True)
        return

    decklists, vocab, deck_ctxs, fixed_tables = build_pool()
    shared = load_frozen_stack(vocab.size)
    net = DeckNetwork(shared, film_condition_dim=D_MODEL, non_targeting_n_actions=len(fixed_tables[TRAINING_DECK]))

    print("Collecting a real buffer (~20 games) for realistic token shapes...", flush=True)
    rng = random.Random(0)
    # Roster restricted to TRAINING_DECK only -- this benchmark only ever
    # built a live net for that one deck, so letting the pool sample the
    # OTHER real deck (build_pool() always returns both) as an opponent
    # would KeyError on a live_nets lookup that was never populated for it.
    pool = LeaguePool(LEAGUE_DIR, [TRAINING_DECK])
    live_nets = {TRAINING_DECK: net}
    typical_buf, played = collect_rollout_league(
        TRAINING_DECK, net, deck_ctxs[TRAINING_DECK], decklists[TRAINING_DECK], action_count_win_reward_200_floor02,
        pool, decklists, deck_ctxs, live_nets, [never_terminated, never_terminated], HORIZON, 20, rng, device="cpu",
    )
    large_buf = duplicate_buffer(typical_buf, 4)
    print(f"typical buffer: {len(typical_buf)} transitions ({played} games); large buffer: {len(large_buf)} transitions\n",
          flush=True)

    for label, buf in (("typical", typical_buf), ("large", large_buf)):
        print(f"=== Buffer: {label} ({len(buf)} transitions) ===", flush=True)
        print(f"{'batch_size':>10} | {'cpu (s)':>8} | {'gpu (s)':>8} | {'cpu/gpu ratio':>13} | winner", flush=True)
        crossed = False
        for batch_size in BATCH_SIZES:
            print(f"  measuring batch_size={batch_size} (cpu then gpu)...", flush=True)  # live progress -- each cell can take ~20s on the large buffer
            cpu_time = time_update(net, buf, "cpu", batch_size)
            gpu_time = time_update(net, buf, "cuda", batch_size)
            ratio = cpu_time / gpu_time
            winner = "GPU" if gpu_time < cpu_time else "CPU"
            print(f"{batch_size:>10} | {cpu_time:>8.2f} | {gpu_time:>8.2f} | {ratio:>13.2f} | {winner}", flush=True)
            if winner == "GPU" and not crossed:
                print(f"  <-- crossover: GPU first wins at batch_size={batch_size}", flush=True)
                crossed = True
        if not crossed:
            print("  (GPU never won within the tested batch_size range)", flush=True)
        print(flush=True)


if __name__ == "__main__":
    main()
