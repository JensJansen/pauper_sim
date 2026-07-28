"""League driver: one continuous training loop in place of a discrete
"mirror baseline, then one pairwise fine-tune" curriculum. Every deck in
the roster (data/league_decks.json) trains every round, against an
opponent RESAMPLED each game from a LeaguePool (rl.league) -- historical
snapshots of every deck plus everyone's current live weights, picked
uniformly. No separate "Stage 1"/"Stage 2":
early on, with few decks and no snapshots yet, most games are naturally
close to mirror play; cross-deck and cross-snapshot exposure grows
organically as the pool fills in, without a hardcoded phase boundary.

Same frozen shared stack as before (pretraining's shared_stack_frozen.pt,
unaffected by this change). Each deck's own live net/optimizer persists
in checkpoints/league/<deck_name>/live.pt; historical opponents live
alongside as snapshot_<id>.pt (LeaguePool's own concern).

Parallel rollout collection (rl.train.collect_rollout_league_parallel)
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
log_event, already instrumented across mana.py/turn.py/resolution/*.py/
game/effects/*.py -- see rl.train.collect_rollout's own docstring) for
every game played, written as one JSON file. Logging is threaded through
both the sequential and parallel worker paths (event dicts are picklable).

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

from rl.rewards import deploy_reward_v1
from rl.arch import SetTransformer
from rl.deck import DeckNetwork
from rl.league import LeaguePool
from rl.pool import build_pool
from rl.agent import SeatAgent
from rl.train import (
    batch_size_for_iteration, collect_rollout, collect_rollout_league,
    collect_rollout_league_parallel, ppo_update, _constant_pairing,
)
from rl.mulligan import MulliganNet, update as mulligan_update

CHECKPOINT_DIR = "../checkpoints"
FROZEN_STACK = f"{CHECKPOINT_DIR}/shared_stack_frozen.pt"
LEAGUE_DIR = f"{CHECKPOINT_DIR}/league"
D_MODEL = 64
SHARED_HPARAMS = {"d_model": D_MODEL, "n_heads": 4, "n_layers": 2, "dim_feedforward": 128}


def load_frozen_stack(vocab_size):
    assert os.path.exists(FROZEN_STACK), (
        f"{FROZEN_STACK} not found -- run `python run_pretrain.py ... --freeze` (pretrain) first"
    )
    ckpt = torch.load(FROZEN_STACK, weights_only=True)
    assert ckpt["vocab_size"] == vocab_size, (
        f"frozen stack was built with vocab_size={ckpt['vocab_size']}, current pool vocab is {vocab_size} -- "
        "the deck roster changed since pretraining ran; re-run pretraining or fix the mismatch before continuing"
    )
    shared = SetTransformer(vocab_size, d_model=ckpt["d_model"], n_heads=4, n_layers=2, dim_feedforward=128)
    shared.load_state_dict(ckpt["shared"])
    for p in shared.parameters():
        p.requires_grad = False
    shared.eval()
    return shared


def build_fresh_stack(vocab_size):
    """A shared stack with the EXACT architecture/config of the real frozen
    one (SHARED_HPARAMS) but random, UNTRAINED weights -- frozen + eval, same
    as load_frozen_stack returns. Lets the benchmark harness drive the real
    training loop (_run_session) without a trained shared_stack_frozen.pt on
    disk: identical to the real league in every way except the weights aren't
    trained, which is exactly the intended benchmarking difference."""
    shared = SetTransformer(vocab_size, **SHARED_HPARAMS)
    for p in shared.parameters():
        p.requires_grad = False
    shared.eval()
    return shared


def _save_live_checkpoints(live_nets, optimizers, deck_names, session, session_path, league_dir,
                           mulligan_nets=None, mulligan_optimizers=None):
    """Persist every deck's current live net + optimizer + the session
    counter. Called at each snapshot point AND at session end -- NOT only
    at the end: a mid-session crash (a rare card-interaction bug ~2500
    games into a 3000-game batch is exactly how this was learned) would
    otherwise discard the whole session's live-net training, since only the
    periodic snapshots (historical opponents) get written incrementally.
    With this, a crash loses at most snapshot_every iterations."""
    os.makedirs(league_dir, exist_ok=True)
    for name in deck_names:
        deck_dir = f"{league_dir}/{name}"
        os.makedirs(deck_dir, exist_ok=True)
        torch.save({"net": live_nets[name].state_dict(), "optimizer": optimizers[name].state_dict()},
                   f"{deck_dir}/live.pt")
        if mulligan_nets is not None:
            torch.save({"net": mulligan_nets[name].state_dict(), "optimizer": mulligan_optimizers[name].state_dict()},
                       f"{deck_dir}/mulligan.pt")
    with open(session_path, "w") as f:
        f.write(str(session))


def _run_session(n_iterations, games_per_iteration, snapshot_every, executor, n_workers,
                  batch_size_start=32, batch_size_cap=2048, batch_size_steps=6,
                  fresh_stack=False, league_dir=None, seed=None,
                  salvage_opponents=True, train_deck=True, train_mulligan=True, train_decks=None,
                  matchup=None, game_logs=None):
    # fresh_stack + league_dir + seed let the benchmark harness drive this EXACT
    # loop with untrained (but identical-config) models over a throwaway dir,
    # reproducibly -- the only intended differences from a real training
    # session. Defaults preserve the real run: load the trained frozen stack,
    # checkpoint into LEAGUE_DIR, nondeterministic rng.
    #
    # Independent per-layer / per-deck training (freeze modes): train_deck /
    # train_mulligan gate which LAYER updates; train_decks (a subset, default the
    # whole roster) gates which DECKS train -- the rest stay loaded as FROZEN
    # opponents (never updated / snapshotted), which is exactly onboarding a new
    # deck against an established field. "Frozen" needs no requires_grad juggling:
    # collection is inference_mode, so a deck simply doesn't move if we never call
    # its update. Every deck's mulligan/deck net is still loaded so it can PLAY.
    league_dir = league_dir or LEAGUE_DIR
    if seed is not None:
        torch.manual_seed(seed)  # identical fresh stack + in-process sampling across benchmark configs
        random.seed(seed)  # collect_rollout_league_parallel draws its worker seeds from the global random module
    decklists, vocab, deck_ctxs, fixed_tables = build_pool()
    deck_names = list(decklists)
    # matchup = (A, B): a fixed A-vs-B pairing instead of league opponent sampling
    # (snapshotting off); it trains exactly those two decks, so it IS a train_decks
    # subset. Both share the SAME loading/optimizer/checkpoint setup below -- no
    # separate matchup driver.
    if matchup is not None:
        assert len(matchup) == 2, "matchup must name exactly two decks"
        train_decks = list(matchup)
    train_decks = list(train_decks) if train_decks is not None else deck_names
    assert set(train_decks) <= set(deck_names), f"train_decks {train_decks} not all in roster {deck_names}"
    train_set = set(train_decks)
    shared = build_fresh_stack(vocab.size) if fresh_stack else load_frozen_stack(vocab.size)

    live_nets, optimizers = {}, {}
    for name in deck_names:
        net = DeckNetwork(shared, film_condition_dim=D_MODEL, non_targeting_n_actions=len(fixed_tables[name]))
        live_path = f"{league_dir}/{name}/live.pt"
        if os.path.exists(live_path):
            ckpt = torch.load(live_path, weights_only=True)
            net.load_state_dict(ckpt["net"])
        optimizer = torch.optim.Adam([p for p in net.parameters() if p.requires_grad], lr=3e-4)
        if os.path.exists(live_path) and "optimizer" in ckpt:  # migrated live.pt drops optimizer -> fresh Adam
            optimizer.load_state_dict(ckpt["optimizer"])
        live_nets[name] = net
        optimizers[name] = optimizer

    # Per-deck mulligan model (rl.mulligan): owns the pregame keep/mulligan +
    # bottoming, trained by its OWN REINFORCE with a direct game-outcome reward
    # (decoupled from the main PPO above). Shares the same frozen stack. Resets
    # with the per-deck policies (mulligan.pt is deleted alongside live.pt).
    mulligan_nets, mulligan_optimizers = {}, {}
    for name in deck_names:
        mnet = MulliganNet(shared)
        mull_path = f"{league_dir}/{name}/mulligan.pt"
        if os.path.exists(mull_path):
            mck = torch.load(mull_path, weights_only=True)
            mnet.load_state_dict(mck["net"])
        mopt = torch.optim.Adam([p for p in mnet.parameters() if p.requires_grad], lr=1e-3)
        if os.path.exists(mull_path):
            mopt.load_state_dict(mck["optimizer"])
        mulligan_nets[name] = mnet
        mulligan_optimizers[name] = mopt

    pool = LeaguePool(league_dir, deck_names)
    session_path = f"{league_dir}/session.txt"
    session = int(open(session_path).read()) + 1 if os.path.exists(session_path) else 0
    if session > 0:
        print(f"resumed league (session {session}); snapshots on disk: "
              f"{ {name: len(pool.snapshots[name]) for name in deck_names} }")

    rng = random.Random(seed)  # seed=None -> nondeterministic, identical to the prior random.Random()
    reward_fn = deploy_reward_v1
    reward_fn_name = "deploy_reward_v1"
    horizon = 120

    mode = []
    if not train_deck:
        mode.append("deck FROZEN")
    if not train_mulligan:
        mode.append("mulligan FROZEN")
    if train_set != set(deck_names):
        mode.append(f"train_decks={train_decks}")
    print(f"League session {session}: n_iterations={n_iterations} games_per_iteration={games_per_iteration} "
          f"snapshot_every={snapshot_every} decks={deck_names} n_workers={n_workers} "
          f"batch_size={batch_size_start}->{batch_size_cap} ({batch_size_steps} steps)"
          f"{' [' + ', '.join(mode) + ']' if mode else ''}")
    t0 = time.time()
    total_games = 0
    collect_time_total = 0.0
    update_time_total = 0.0
    for iteration in range(n_iterations):
        batch_size = batch_size_for_iteration(iteration, n_iterations, batch_size_start, batch_size_cap, batch_size_steps)
        mull_by_deck_iter = {name: [] for name in train_decks}  # mulligan transitions accumulated across this iteration
        # Snapshot the mulligan nets for the workers now (ALL decks -- a frozen
        # opponent still needs its mulligan net to play). Updated once after the
        # deck-loop (on-policy within the iteration, same as the main nets).
        mulligan_state_dicts = {n: mulligan_nets[n].state_dict() for n in deck_names}
        for name in train_decks:  # only TRAIN decks get a round; the rest are frozen opponents
            t_collect0 = time.time()
            if matchup is not None:
                # Fixed pairing: this deck vs the OTHER named deck (no opponent
                # sampling), both sides carrying their REAL mulligan models via the
                # unified loop -- so logged matchup games use the same pregame policy
                # training does. game_logs (if given) accumulates across every round.
                opp = matchup[1] if name == matchup[0] else matchup[0]
                pairing = _constant_pairing(
                    [SeatAgent(live_nets[name], mulligan_nets[name], deck_ctxs[name]),
                     SeatAgent(live_nets[opp], mulligan_nets[opp], deck_ctxs[opp])],
                    [decklists[name], decklists[opp]], [reward_fn, reward_fn], [name, opp])
                buffers_by_deck, mull_by_deck, played = collect_rollout(
                    pairing, games_per_iteration, horizon, rng, device="cpu", game_logs=game_logs)
            elif executor is not None:
                buffers_by_deck, mull_by_deck, played = collect_rollout_league_parallel(
                    name, live_nets, reward_fn_name, league_dir, horizon, games_per_iteration,
                    executor, n_workers, SHARED_HPARAMS, mulligan_state_dicts, game_logs=game_logs,
                )
            else:
                buffers_by_deck, mull_by_deck, played = collect_rollout_league(
                    name, live_nets, mulligan_nets, deck_ctxs, decklists, pool, reward_fn,
                    horizon, games_per_iteration, rng, device="cpu", game_logs=game_logs,
                )
            collect_time_total += time.time() - t_collect0
            total_games += played
            # Accumulate mulligan transitions for each TRAIN deck that generated
            # some this round (training deck + any live-opponent salvage); a frozen
            # opponent's are discarded (it isn't being trained).
            for deck_name, tr in mull_by_deck.items():
                if deck_name in train_set:
                    mull_by_deck_iter[deck_name].extend(tr)
            t_update0 = time.time()
            # PPO-update every TRAIN deck that received transitions this round (only
            # when deck training is on). The training deck's own bucket always; a
            # live-opponent bucket only when salvage_opponents. A frozen deck / a
            # non-subset opponent is never updated -- just an opponent.
            policy_loss = value_loss = entropy = 0.0
            salvaged = 0
            if train_deck:
                for deck_name, buf in buffers_by_deck.items():
                    if not len(buf) or deck_name not in train_set:
                        continue
                    if deck_name == name:
                        policy_loss, value_loss, entropy = ppo_update(
                            live_nets[name], [optimizers[name]], buf, "cpu", batch_size=batch_size)
                    elif salvage_opponents:
                        ppo_update(live_nets[deck_name], [optimizers[deck_name]], buf, "cpu", batch_size=batch_size)
                        salvaged += len(buf)
            update_time_total += time.time() - t_update0
            print(f"  iter {iteration} [{name}]: games={played} buf={len(buffers_by_deck.get(name, ()))} "
                  f"salvaged={salvaged} batch_size={batch_size} "
                  f"policy_loss={policy_loss:.4f} value_loss={value_loss:.4f}", flush=True)

        # Mulligan-model REINFORCE: one step per TRAIN deck on its own transitions
        # this iteration (its games + live-opponent salvage). Gated on train_mulligan;
        # decoupled from the main PPO updates above -- its own optimizer, its own reward.
        mull_stats = {}
        if train_mulligan:
            for name in train_decks:
                if mull_by_deck_iter[name]:
                    mull_stats[name] = mulligan_update(mulligan_nets[name], mulligan_optimizers[name], mull_by_deck_iter[name])
        if mull_stats:  # readout so the mulligan subsystem is visible while it trains
            total_n = sum(s["n"] for s in mull_stats.values())
            mean_loss = sum(s["loss"] for s in mull_stats.values()) / len(mull_stats)
            print(f"  iter {iteration}: mulligan model -- {total_n} transitions across {len(mull_stats)} decks, "
                  f"mean REINFORCE loss {mean_loss:.4f}", flush=True)

        if matchup is None and (iteration + 1) % snapshot_every == 0:  # matchup: no historical snapshots
            for name in train_decks:  # only TRAIN decks change -> only they need new snapshots
                pool.register_snapshot(name, live_nets[name], mulligan_nets[name])
            _save_live_checkpoints(live_nets, optimizers, deck_names, session, session_path, league_dir,
                                   mulligan_nets, mulligan_optimizers)
            print(f"  iter {iteration}: snapshotted {len(train_decks)} train deck(s) + saved live checkpoints "
                  f"(counts now: { {n: len(pool.snapshots[n]) for n in deck_names} })", flush=True)

    elapsed = time.time() - t0
    print(f"session {session} done in {elapsed:.1f}s ({elapsed / total_games:.2f}s/game across {total_games} games) -- "
          f"collect={collect_time_total:.1f}s ({100 * collect_time_total / elapsed:.0f}%), "
          f"update={update_time_total:.1f}s ({100 * update_time_total / elapsed:.0f}%)")

    _save_live_checkpoints(live_nets, optimizers, deck_names, session, session_path, league_dir,
                           mulligan_nets, mulligan_optimizers)
    print("live checkpoints saved for all decks")


def _run_eval(eval_decks, games_per_pairing, greedy, seed, game_logs, matchup=None,
              fresh_stack=False, league_dir=None):
    """Eval / faithful log generation: play games with NO training (record=False,
    no updates, no checkpointing) over the CURRENT live agents (deck net + its
    mulligan model -- so logged games use the same pregame policy training does).
    Pairing is a round-robin with mirrors (combinations_with_replacement) over
    eval_decks, or a single A-vs-B pairing when `matchup` is given. Sampled by
    default; greedy=True argmaxes. game_logs (a list) collects one engine
    event_log per game."""
    import itertools
    league_dir = league_dir or LEAGUE_DIR
    decklists, vocab, deck_ctxs, fixed_tables = build_pool()
    deck_names = list(decklists)
    eval_decks = list(matchup) if matchup else (list(eval_decks) if eval_decks else deck_names)
    assert set(eval_decks) <= set(deck_names), f"eval decks {eval_decks} not all in roster {deck_names}"
    pairings = [tuple(matchup)] if matchup else list(itertools.combinations_with_replacement(eval_decks, 2))
    shared = build_fresh_stack(vocab.size) if fresh_stack else load_frozen_stack(vocab.size)
    rng = random.Random(seed)
    horizon = 120

    live_nets, mulligan_nets = {}, {}
    for name in set(eval_decks):
        net = DeckNetwork(shared, film_condition_dim=D_MODEL, non_targeting_n_actions=len(fixed_tables[name]))
        live_path = f"{league_dir}/{name}/live.pt"
        if os.path.exists(live_path):
            net.load_state_dict(torch.load(live_path, weights_only=True)["net"])
        net.eval()
        live_nets[name] = net
        mnet = MulliganNet(shared)
        mull_path = f"{league_dir}/{name}/mulligan.pt"
        if os.path.exists(mull_path):
            mnet.load_state_dict(torch.load(mull_path, weights_only=True)["net"])
        mnet.eval()
        mulligan_nets[name] = mnet

    print(f"Eval: {len(pairings)} pairing(s) x {games_per_pairing} games "
          f"({'greedy' if greedy else 'sampled'}, seed={seed}) over decks={eval_decks}")
    t0 = time.time()
    total = 0
    for a, b in pairings:
        # record_as = [None, None] and reward_fns = [None, None]: pure play, no
        # buffers, no reward -- record=False ignores them entirely.
        pairing = _constant_pairing(
            [SeatAgent(live_nets[a], mulligan_nets[a], deck_ctxs[a]),
             SeatAgent(live_nets[b], mulligan_nets[b], deck_ctxs[b])],
            [decklists[a], decklists[b]], [None, None], [None, None])
        _bufs, _mull, played = collect_rollout(pairing, games_per_pairing, horizon, rng, device="cpu",
                                               record=False, greedy=greedy, game_logs=game_logs)
        total += played
        print(f"  {a} vs {b}: {played} games", flush=True)
    print(f"eval done: {total} games in {time.time() - t0:.1f}s")


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


def _write_event_log(log_path, game_logs, meta):
    """Write the engine event logs collected this session to PATH as one compact
    JSON doc (no indent -- an earlier oversized-log issue was pretty-printed
    output). _json_default salvages any stray non-serializable field."""
    os.makedirs(os.path.dirname(log_path) or ".", exist_ok=True)
    doc = {"meta": meta, "games": [{"game_index": i, "events": ev} for i, ev in enumerate(game_logs)]}
    with open(log_path, "w") as f:
        json.dump(doc, f, default=_json_default)
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
                         help="Fixed A-vs-B pairing instead of league opponent sampling (snapshotting off). Trains "
                              "both decks with their real mulligan models via the unified loop.")
    parser.add_argument("--games", type=int, default=50, help="Total games (per deck round) for --matchup mode.")
    parser.add_argument("--decks", type=str, default=None, metavar="A,B,...",
                         help="Train only this comma-separated subset of the roster; the rest stay loaded as FROZEN "
                              "opponents (onboarding a new deck / targeted retraining). Default: the whole roster.")
    parser.add_argument("--train-deck-only", action="store_true",
                         help="Train the per-deck policies only; freeze the mulligan models.")
    parser.add_argument("--train-mulligan-only", action="store_true",
                         help="Train the mulligan models only; freeze the per-deck policies (a clean bandit vs fixed skill).")
    parser.add_argument("--eval", action="store_true",
                         help="Eval / log-generation: play games with NO training over the current live agents "
                              "(round-robin with mirrors over --decks, or one A-vs-B pairing with --matchup). "
                              "--games games per pairing.")
    parser.add_argument("--greedy", action="store_true",
                         help="Eval: argmax the policy instead of sampling (default: sampled, matching trained behavior).")
    parser.add_argument("--seed", type=int, default=None,
                         help="Seed the rng (reproducible eval logs; also reproducible training sampling).")
    parser.add_argument("--log", type=str, default=None, metavar="PATH",
                         help="Write the game engine's own event log for every game this session to PATH as JSON "
                              "(sequential collection only).")
    parser.add_argument("--batch-size-start", type=int, default=32,
                         help="ppo_update batch_size at the first iteration (small/granular early).")
    parser.add_argument("--batch-size-cap", type=int, default=2048,
                         help="ppo_update batch_size ceiling by the end of the session.")
    parser.add_argument("--batch-size-steps", type=int, default=6,
                         help="Number of doublings from --batch-size-start to --batch-size-cap, spread evenly across the session.")
    parser.add_argument("--no-opponent-salvage", action="store_true",
                         help="Disable Path A: when the sampled opponent is another deck's LIVE net, train THAT deck "
                              "too from its (on-policy) transitions this round, at no extra collection cost -- on by "
                              "default. Pass this to fall back to training only the round's own deck (for A/B).")
    return parser


def main():
    args = build_arg_parser().parse_args()

    train_deck = not args.train_mulligan_only
    train_mulligan = not args.train_deck_only
    assert train_deck or train_mulligan, "cannot freeze BOTH layers (--train-deck-only + --train-mulligan-only)"
    train_decks = args.decks.split(",") if args.decks else None
    matchup = tuple(args.matchup) if args.matchup else None
    game_logs = [] if args.log else None

    if args.eval:  # no training: round-robin (or single matchup) over current live agents
        _run_eval(train_decks, args.games, args.greedy, args.seed, game_logs, matchup=matchup)
        if args.log:
            _write_event_log(args.log, game_logs, {"mode": "eval", "matchup": list(matchup) if matchup else None,
                                                   "decks": train_decks, "greedy": args.greedy, "games_logged": len(game_logs)})
        return

    # --matchup counts by --games (per deck round), otherwise by --n-iterations x
    # --games-per-iteration. Logging is threaded through BOTH the sequential and MP
    # league paths (event dicts are picklable), so --log no longer forces sequential.
    n_iterations, games_per_iteration = args.n_iterations, args.games_per_iteration
    if matchup is not None:
        games_per_iteration = min(max(args.games_per_iteration, 10), args.games)
        n_iterations = max(1, args.games // games_per_iteration)

    schedule_kwargs = dict(batch_size_start=args.batch_size_start,
                            batch_size_cap=args.batch_size_cap, batch_size_steps=args.batch_size_steps,
                            salvage_opponents=not args.no_opponent_salvage, seed=args.seed,
                            train_deck=train_deck, train_mulligan=train_mulligan, train_decks=train_decks,
                            matchup=matchup, game_logs=game_logs)

    sequential = matchup is not None or args.n_workers <= 1  # matchup uses collect_rollout directly (no worker path)
    if not sequential:
        # ONE executor for the whole session, reused across every iteration --
        # process-spawn/import overhead is paid once, not per collection round.
        with ProcessPoolExecutor(max_workers=args.n_workers) as executor:
            _run_session(n_iterations, games_per_iteration, args.snapshot_every, executor, args.n_workers, **schedule_kwargs)
    else:
        _run_session(n_iterations, games_per_iteration, args.snapshot_every, None, 1, **schedule_kwargs)

    if args.log:
        meta = {"mode": "matchup" if matchup else "league", "matchup": list(matchup) if matchup else None,
                "train_decks": train_decks, "games_logged": len(game_logs)}
        _write_event_log(args.log, game_logs, meta)


if __name__ == "__main__":
    main()
