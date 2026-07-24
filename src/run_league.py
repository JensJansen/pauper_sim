"""League driver: replaces run_stage1.py + run_stage2.py's discrete
"mirror baseline, then one pairwise fine-tune" curriculum with one
continuous loop. Every deck in the roster (data/league_decks.json) trains
every round, against an opponent RESAMPLED each game from a LeaguePool
(token_league.py) -- historical snapshots of every deck plus everyone's
current live weights, picked uniformly. No separate "Stage 1"/"Stage 2":
early on, with few decks and no snapshots yet, most games are naturally
close to mirror play; cross-deck and cross-snapshot exposure grows
organically as the pool fills in, without a hardcoded phase boundary.

Same frozen shared stack as before (Phase 4's shared_stack_frozen.pt,
unaffected by this change). Each deck's own live net/optimizer persists
in checkpoints/league/<deck_name>/live.pt; historical opponents live
alongside as snapshot_<id>.pt (LeaguePool's own concern).

Parallel rollout collection (token_train.collect_rollout_league_parallel)
is used whenever n_workers > 1 -- benchmarked at ~3.2-3.5x wall-clock
speedup at 6-8 worker processes on this machine (6 physical cores),
plateauing beyond that (hyperthreaded/logical cores past the physical
count add nothing reliable). Defaults to 6. The executor is created ONCE
for the whole session and reused across every iteration, so process-
spawn/import overhead (each worker re-importing torch + the game engine)
is paid once, not once per collection round.

--matchup DECK_A DECK_B --games N [--log PATH]: bypasses league opponent
sampling entirely -- runs N games as a DIRECT, fixed pairing between two
named decks (train_selfplay's cross-matchup path), still updating and
checkpointing both decks' live nets normally. --log PATH captures the
game engine's own existing event log (game/state.py's GameState.
log_event, already instrumented across mana.py/turn.py/resolution.py/
game/effects/*.py -- see token_train.collect_rollout's own docstring) for
every game played, written as one JSON file. Logging is not yet threaded
through the parallel worker path, so --log forces sequential collection.

Usage:
  python run_league.py [--n-iterations N] [--games-per-iteration N] [--snapshot-every N] [--n-workers N]
  python run_league.py --matchup DECK_A DECK_B [--games N] [--log PATH]
"""
import argparse
import json
import os
import random
import time
from concurrent.futures import ProcessPoolExecutor

import torch

from rewards import action_count_win_reward_200_floor02
from terminated import never_terminated
from token_arch import SetTransformer
from token_deck import DeckNetwork
from token_league import LeaguePool
from token_pool import build_pool
from token_train import (
    batch_size_for_iteration, collect_rollout_league, collect_rollout_league_parallel, device_for_batch_size,
    move_optimizer_state, ppo_update, train_selfplay,
)

CHECKPOINT_DIR = "../checkpoints"
FROZEN_STACK = f"{CHECKPOINT_DIR}/shared_stack_frozen.pt"
LEAGUE_DIR = f"{CHECKPOINT_DIR}/league"
D_MODEL = 64
SHARED_HPARAMS = {"d_model": D_MODEL, "n_heads": 4, "n_layers": 2, "dim_feedforward": 128}


def load_frozen_stack(vocab_size):
    assert os.path.exists(FROZEN_STACK), (
        f"{FROZEN_STACK} not found -- run `python run_pretrain.py ... --freeze` (Phase 4) first"
    )
    ckpt = torch.load(FROZEN_STACK, weights_only=True)
    assert ckpt["vocab_size"] == vocab_size, (
        f"frozen stack was built with vocab_size={ckpt['vocab_size']}, current pool vocab is {vocab_size} -- "
        "the deck roster changed since Phase 4 ran; re-run Phase 4 or fix the mismatch before continuing"
    )
    shared = SetTransformer(vocab_size, d_model=ckpt["d_model"], n_heads=4, n_layers=2, dim_feedforward=128)
    shared.load_state_dict(ckpt["shared"])
    for p in shared.parameters():
        p.requires_grad = False
    shared.eval()
    return shared


def _run_session(n_iterations, games_per_iteration, snapshot_every, executor, n_workers,
                  gpu_threshold=None, batch_size_start=32, batch_size_cap=2048, batch_size_steps=6):
    decklists, vocab, deck_ctxs, fixed_tables = build_pool()
    deck_names = list(decklists)
    shared = load_frozen_stack(vocab.size)

    live_nets, optimizers = {}, {}
    for name in deck_names:
        net = DeckNetwork(shared, film_condition_dim=D_MODEL, non_targeting_n_actions=len(fixed_tables[name]))
        live_path = f"{LEAGUE_DIR}/{name}/live.pt"
        if os.path.exists(live_path):
            ckpt = torch.load(live_path, weights_only=True)
            net.load_state_dict(ckpt["net"])
        optimizer = torch.optim.Adam([p for p in net.parameters() if p.requires_grad], lr=3e-4)
        if os.path.exists(live_path):
            optimizer.load_state_dict(ckpt["optimizer"])
        live_nets[name] = net
        optimizers[name] = optimizer

    pool = LeaguePool(LEAGUE_DIR, deck_names)
    session_path = f"{LEAGUE_DIR}/session.txt"
    session = int(open(session_path).read()) + 1 if os.path.exists(session_path) else 0
    if session > 0:
        print(f"resumed league (session {session}); snapshots on disk: "
              f"{ {name: len(pool.snapshots[name]) for name in deck_names} }")

    rng = random.Random()
    terminated_fns = [never_terminated, never_terminated]
    reward_fn = action_count_win_reward_200_floor02
    reward_fn_name = "action_count_win_reward_200_floor02"
    horizon = 120

    print(f"League session {session}: n_iterations={n_iterations} games_per_iteration={games_per_iteration} "
          f"snapshot_every={snapshot_every} decks={deck_names} n_workers={n_workers} "
          f"batch_size={batch_size_start}->{batch_size_cap} ({batch_size_steps} steps) gpu_threshold={gpu_threshold}")
    t0 = time.time()
    total_games = 0
    collect_time_total = 0.0
    update_time_total = 0.0
    for iteration in range(n_iterations):
        batch_size = batch_size_for_iteration(iteration, n_iterations, batch_size_start, batch_size_cap, batch_size_steps)
        update_device = device_for_batch_size(batch_size, gpu_threshold)
        for name in deck_names:
            t_collect0 = time.time()
            if executor is not None:
                buf, played = collect_rollout_league_parallel(
                    name, live_nets, reward_fn_name, LEAGUE_DIR, horizon, games_per_iteration,
                    executor, n_workers, SHARED_HPARAMS,
                )
            else:
                buf, played = collect_rollout_league(
                    name, live_nets[name], deck_ctxs[name], decklists[name], reward_fn,
                    pool, decklists, deck_ctxs, live_nets, terminated_fns, horizon, games_per_iteration, rng, device="cpu",
                )
            collect_time_total += time.time() - t_collect0
            total_games += played
            t_update0 = time.time()
            if len(buf):
                # Collection above (and any OTHER deck's collection this
                # same iteration, which may sample THIS deck's live net as
                # its opponent -- collect_rollout_league's own live_nets
                # lookup) always needs this net on CPU -- so a GPU update
                # is a real round trip, move there, update, move straight
                # back, never left resident.
                if update_device == "cuda":
                    live_nets[name].to("cuda")
                    move_optimizer_state(optimizers[name], "cuda")
                policy_loss, value_loss, entropy = ppo_update(
                    live_nets[name], [optimizers[name]], buf, update_device, batch_size=batch_size,
                )
                if update_device == "cuda":
                    live_nets[name].to("cpu")
                    move_optimizer_state(optimizers[name], "cpu")
            else:
                policy_loss = value_loss = entropy = 0.0
            update_time_total += time.time() - t_update0
            print(f"  iter {iteration} [{name}]: games={played} buf={len(buf)} batch_size={batch_size} "
                  f"device={update_device} policy_loss={policy_loss:.4f} value_loss={value_loss:.4f}", flush=True)

        if (iteration + 1) % snapshot_every == 0:
            for name in deck_names:
                pool.register_snapshot(name, live_nets[name])
            print(f"  iter {iteration}: snapshotted all decks "
                  f"(counts now: { {n: len(pool.snapshots[n]) for n in deck_names} })", flush=True)

    elapsed = time.time() - t0
    print(f"session {session} done in {elapsed:.1f}s ({elapsed / total_games:.2f}s/game across {total_games} games) -- "
          f"collect={collect_time_total:.1f}s ({100 * collect_time_total / elapsed:.0f}%), "
          f"update={update_time_total:.1f}s ({100 * update_time_total / elapsed:.0f}%)")

    os.makedirs(LEAGUE_DIR, exist_ok=True)
    for name in deck_names:
        deck_dir = f"{LEAGUE_DIR}/{name}"
        os.makedirs(deck_dir, exist_ok=True)
        torch.save({"net": live_nets[name].state_dict(), "optimizer": optimizers[name].state_dict()},
                   f"{deck_dir}/live.pt")
    with open(session_path, "w") as f:
        f.write(str(session))
    print("live checkpoints saved for all decks")


def _json_default(obj):
    """Fallback for json.dump when a raw engine object slips into a logged
    event -- some log_event field ends up holding an object instead of a
    plain value (confirmed the hard way: a real 50-game run hit
    "TypeError: Object of type CardDef is not JSON serializable" partway
    through the write, after every log_event call site in game/*.py was
    checked by hand and found using .name correctly -- the actual culprit
    wasn't isolated in time to fix at the source, so this boundary fix
    keeps a real log from being lost to one bad field, and prints enough
    to trace the culprit next time it fires). Converts to the object's own
    .name where recognizable, else repr()."""
    name = getattr(obj, "name", None)
    if isinstance(name, str):
        print(f"  [event log] non-serializable {type(obj).__name__} encountered, using its .name: {name!r}")
        return name
    print(f"  [event log] non-serializable {type(obj).__name__} encountered, using repr(): {obj!r}")
    return repr(obj)


def _run_matchup_session(deck_a_name, deck_b_name, total_games, log_path):
    decklists, vocab, deck_ctxs, fixed_tables = build_pool()
    for name in (deck_a_name, deck_b_name):
        assert name in decklists, f"unknown deck {name!r}, expected one of {list(decklists)}"
    shared = load_frozen_stack(vocab.size)

    live_nets, optimizers = {}, {}
    for name in (deck_a_name, deck_b_name):
        net = DeckNetwork(shared, film_condition_dim=D_MODEL, non_targeting_n_actions=len(fixed_tables[name]))
        live_path = f"{LEAGUE_DIR}/{name}/live.pt"
        if os.path.exists(live_path):
            ckpt = torch.load(live_path, weights_only=True)
            net.load_state_dict(ckpt["net"])
        optimizer = torch.optim.Adam([p for p in net.parameters() if p.requires_grad], lr=3e-4)
        if os.path.exists(live_path):
            optimizer.load_state_dict(ckpt["optimizer"])
        live_nets[name] = net
        optimizers[name] = optimizer

    rng = random.Random()
    terminated_fns = [never_terminated, never_terminated]
    reward_fn = action_count_win_reward_200_floor02
    horizon = 120

    games_per_iteration = min(10, total_games)
    n_iterations = max(1, total_games // games_per_iteration)
    actual_total = n_iterations * games_per_iteration
    game_logs = [] if log_path else None

    print(f"Direct matchup: {deck_a_name} vs {deck_b_name}, {actual_total} games "
          f"({n_iterations} iterations x {games_per_iteration} games/iteration), logging={'on' if log_path else 'off'}")
    t0 = time.time()
    train_selfplay(
        live_nets[deck_a_name], deck_ctxs[deck_a_name], decklists[deck_a_name], reward_fn,
        live_nets[deck_b_name], deck_ctxs[deck_b_name], decklists[deck_b_name], reward_fn,
        [optimizers[deck_a_name]], [optimizers[deck_b_name]], terminated_fns, horizon,
        n_iterations=n_iterations, games_per_iteration=games_per_iteration, rng=rng, device="cpu", game_logs=game_logs,
    )
    elapsed = time.time() - t0
    print(f"matchup session done in {elapsed:.1f}s ({elapsed / actual_total:.2f}s/game across {actual_total} games)")

    os.makedirs(LEAGUE_DIR, exist_ok=True)
    for name in (deck_a_name, deck_b_name):
        deck_dir = f"{LEAGUE_DIR}/{name}"
        os.makedirs(deck_dir, exist_ok=True)
        torch.save({"net": live_nets[name].state_dict(), "optimizer": optimizers[name].state_dict()},
                   f"{deck_dir}/live.pt")
    print("live checkpoints saved for both decks")

    if log_path:
        os.makedirs(os.path.dirname(log_path) or ".", exist_ok=True)
        doc = {
            "meta": {"deck_a": deck_a_name, "deck_b": deck_b_name, "horizon": horizon, "n_games": actual_total},
            "games": [{"game_index": i, "events": events} for i, events in enumerate(game_logs)],
        }
        with open(log_path, "w") as f:
            json.dump(doc, f, default=_json_default)  # compact (no indent) -- the earlier oversized-log issue was pretty-printed output; avoiding that even at this modest scale
        size_kb = os.path.getsize(log_path) / 1024
        print(f"event log written to {log_path} ({len(game_logs)} games, "
              f"{sum(len(g) for g in game_logs)} total events, {size_kb:.1f} KB)")


def build_arg_parser():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--n-iterations", type=int, default=1)
    parser.add_argument("--games-per-iteration", type=int, default=2)
    parser.add_argument("--snapshot-every", type=int, default=20)
    parser.add_argument("--n-workers", type=int, default=6)
    parser.add_argument("--matchup", nargs=2, metavar=("DECK_A", "DECK_B"), default=None,
                         help="Direct fixed pairing instead of league opponent sampling.")
    parser.add_argument("--games", type=int, default=50, help="Total games for --matchup mode.")
    parser.add_argument("--log", type=str, default=None, metavar="PATH",
                         help="Write the game engine's own event log for every game this session to PATH as JSON.")
    parser.add_argument("--gpu-threshold", type=int, default=None, metavar="BATCH_SIZE",
                         help="ppo_update batch_size at/above which to mechanically switch that update to GPU. "
                              "Find via benchmark_ppo_batch_size.py's own measured crossover -- default None means "
                              "always CPU (no threshold known/set).")
    parser.add_argument("--batch-size-start", type=int, default=32,
                         help="ppo_update batch_size at the first iteration (small/granular early).")
    parser.add_argument("--batch-size-cap", type=int, default=2048,
                         help="ppo_update batch_size ceiling by the end of the session.")
    parser.add_argument("--batch-size-steps", type=int, default=6,
                         help="Number of doublings from --batch-size-start to --batch-size-cap, spread evenly across the session.")
    return parser


def main():
    args = build_arg_parser().parse_args()

    if args.matchup:
        _run_matchup_session(args.matchup[0], args.matchup[1], args.games, args.log)
        return

    assert args.log is None, (
        "--log is only wired through --matchup mode right now (train_selfplay's game_logs param) -- "
        "collect_rollout_league (plain league opponent-sampling mode) doesn't forward it yet"
    )
    n_workers = args.n_workers
    schedule_kwargs = dict(gpu_threshold=args.gpu_threshold, batch_size_start=args.batch_size_start,
                            batch_size_cap=args.batch_size_cap, batch_size_steps=args.batch_size_steps)
    if n_workers > 1:
        # ONE executor for the whole session, reused across every
        # iteration -- process-spawn/import overhead (each worker
        # re-importing torch + the game engine) is paid once here, not
        # once per collection round. n_workers > 6 (this machine's
        # physical core count) measured no reliable further speedup.
        with ProcessPoolExecutor(max_workers=n_workers) as executor:
            _run_session(args.n_iterations, args.games_per_iteration, args.snapshot_every, executor, n_workers,
                         **schedule_kwargs)
    else:
        _run_session(args.n_iterations, args.games_per_iteration, args.snapshot_every, None, n_workers, **schedule_kwargs)


if __name__ == "__main__":
    main()
