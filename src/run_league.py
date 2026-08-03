"""League driver: one continuous training loop, with no separate "Stage 1"/
"Stage 2" curriculum. Every deck in the roster (data/league_decks.json)
trains every round, against an opponent RESAMPLED each game from a
LeaguePool (rl.league) -- historical snapshots of every deck plus everyone's
current live weights, picked uniformly. No separate stages are needed:
early on, with few decks and no snapshots yet, most games are naturally
close to mirror play; cross-deck and cross-snapshot exposure grows
organically as the pool fills in, without a hardcoded phase boundary.

Uses the same frozen shared stack as pretraining (shared_stack_frozen.pt).
Each deck's own live net/optimizer persists in checkpoints/league/
<deck_name>/live.pt; historical opponents live alongside as
snapshot_<id>.pt (LeaguePool's own concern).

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
  python run_league.py --run-config PATH --league-config PATH
      Normal training: run-mechanics defaults (training_configs/run_default.json) + one league's identity
      (training_configs/league_*.json: league_name, roster, optional train_decks, total_games).
      No game count given -- it's computed automatically from the league's own progress.json (doubles each
      batch, 1/2/4/8/..., stopping once total_games is reached). Either config's
      individual values, or --league-name/--roster/--n-workers/etc. directly, still override for one call.
  python run_league.py --n-iterations N [--snapshot-every N] [--n-workers N]
      Debug / one-off: force an exact iteration count, bypassing auto-sizing (and its progress.json bookkeeping)
      entirely. Config files still apply for anything not explicitly overridden.
  python run_league.py --matchup DECK_A DECK_B [--games N] [--log PATH]
      Fixed A-vs-B pairing, no league opponent sampling, no auto-sizing.
"""
import argparse
import json
import os
import random
import time
from concurrent.futures import ProcessPoolExecutor

import torch

from rl.rewards import deploy_reward_v2
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


def _load_progress(league_dir):
    """This league's own auto-sizing state (checkpoints/<league_name>/
    progress.json) -- never hand-edited, only ever written by _save_progress
    below. Absent (a brand-new league) reads as "nothing run yet": the
    doubling sequence's own start-at-1 case, per _next_batch_games."""
    path = f"{league_dir}/progress.json"
    if not os.path.exists(path):
        return {"last_batch_size": 0, "cumulative_games_per_deck": 0}
    return json.load(open(path))


def _save_progress(league_dir, last_batch_size, cumulative_games_per_deck):
    os.makedirs(league_dir, exist_ok=True)
    with open(f"{league_dir}/progress.json", "w") as f:
        json.dump({"last_batch_size": last_batch_size, "cumulative_games_per_deck": cumulative_games_per_deck}, f)


def _next_batch_games(league_dir, total_games):
    """The next auto-sized batch's games-per-deck: doubles from the last real
    batch this league actually ran (1 if none yet), never overshooting
    total_games on the final batch. Returns None once total_games is already
    met -- the caller's signal to run nothing at all.

    No separate cap here (there used to be a max_batch_size argument): the
    doubling ladder's own safety property -- never jump straight from a
    small, verified-healthy batch to a huge one -- is now enforced by
    whatever repeatedly re-invokes this script and health-checks between
    calls (the `/train` skill, or the webapp's auto-escalation loop),
    exactly the same mechanism that already has to exist for the ladder to
    mean anything. A second, hand-picked ceiling on top of that was
    redundant -- every prior value (1024, 2048, 4000) was picked ad hoc per
    league file with no principled basis, never actually exercised by real
    training at scale."""
    progress = _load_progress(league_dir)
    remaining = total_games - progress["cumulative_games_per_deck"]
    if remaining <= 0:
        return None
    next_size = progress["last_batch_size"] * 2 if progress["last_batch_size"] > 0 else 1
    return min(next_size, remaining)


def _run_session(n_iterations, games_per_iteration, snapshot_every, executor, n_workers,
                  fresh_stack=False, league_dir=None, seed=None,
                  train_deck=True, train_mulligan=True, train_decks=None,
                  matchup=None, game_logs=None, checkpoint_rate=0.0, roster=None):
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
    # roster: a TRUE isolated sub-league -- restricts the entire opponent pool (not
    # just which decks receive a training round, see train_decks below) to this
    # subset. vocab/shared_stack/deck_ctxs/fixed_tables still come from the FULL
    # build_pool() roster above, unchanged -- only WHICH deck_names this session ever
    # constructs a live net/optimizer for, or ever samples as an opponent, narrows.
    # This preserves checkpoint/vocab compatibility with the full-roster league (a
    # sub-league can seed its decks straight from full-league live.pt checkpoints)
    # while genuinely never loading, training, or pairing against anyone outside the
    # subset -- unlike train_decks alone, which still samples opponents from the
    # WHOLE roster (that's train_decks' own intended use: onboarding a new deck
    # against an already-established field, not an isolated sub-league).
    if roster is not None:
        assert set(roster) <= set(deck_names), f"roster {roster} not all in the full pool roster {deck_names}"
        deck_names = list(roster)
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

    # Fixed for the whole session (shared never trains, trunk widths never
    # resize) -- computed once here rather than re-derived by
    # collect_rollout_league_parallel on every one of its
    # n_iterations * len(train_decks) calls below.
    shared_state_dict = shared.state_dict()
    all_trunk_hidden = {name: tuple(layer.out_features for layer in net.trunk_layers) for name, net in live_nets.items()}

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
    reward_fn = deploy_reward_v2
    reward_fn_name = "deploy_reward_v2"
    horizon = 120

    mode = []
    if not train_deck:
        mode.append("deck FROZEN")
    if not train_mulligan:
        mode.append("mulligan FROZEN")
    if train_set != set(deck_names):
        mode.append(f"train_decks={train_decks}")
    print(f"League session {session}: n_iterations={n_iterations} games_per_iteration={games_per_iteration} "
          f"snapshot_every={snapshot_every} checkpoint_rate={checkpoint_rate} decks={deck_names} n_workers={n_workers}"
          f"{' [' + ', '.join(mode) + ']' if mode else ''}")
    t0 = time.time()
    total_games = 0
    collect_time_total = 0.0
    update_time_total = 0.0
    for iteration in range(n_iterations):
        # PPO minibatch ramp (32 -> 2048 over 6 steps): batch_size_for_iteration's
        # own hardcoded defaults -- see its docstring (Smith et al. 2017 "grow
        # batch size instead of decaying LR"). Used to be 3 separate CLI-tunable
        # knobs; the codebase's own comment on them admitted nobody had ever
        # actually overridden them in practice, so they're fixed here instead.
        batch_size = batch_size_for_iteration(iteration, n_iterations)
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
                    executor, n_workers, SHARED_HPARAMS, shared_state_dict, all_trunk_hidden,
                    mulligan_state_dicts, game_logs=game_logs, checkpoint_rate=checkpoint_rate,
                )
            else:
                buffers_by_deck, mull_by_deck, played = collect_rollout_league(
                    name, live_nets, mulligan_nets, deck_ctxs, decklists, pool, reward_fn,
                    horizon, games_per_iteration, rng, device="cpu", game_logs=game_logs,
                    checkpoint_rate=checkpoint_rate,
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
            # when deck training is on). ONE gate, unconditionally: deck_name must be
            # in train_set (a frozen deck / a non-subset opponent -- e.g. an
            # established field a new deck is onboarding against, see train_decks'
            # own docstring above -- is NEVER updated, just an opponent). Past that
            # single preliminary check, the invariant is absolute: the training
            # deck's own bucket always updates, and so does ANY live-opponent bucket
            # that shows up here at all -- a live opponent outside train_set never
            # reaches this loop in the first place (buffers_by_deck only ever
            # contains a bucket for a live opponent, per _make_league_pairing's own
            # record_as logic; a frozen-snapshot opponent records nothing regardless).
            # Within train_set, "live opponent got real transitions this round" and
            # "that deck should learn from them" are the same fact -- never a
            # separate, configurable choice.
            policy_loss = value_loss = entropy = 0.0
            salvaged = 0
            if train_deck:
                for deck_name, buf in buffers_by_deck.items():
                    if not len(buf) or deck_name not in train_set:
                        continue
                    if deck_name == name:
                        policy_loss, value_loss, entropy = ppo_update(
                            live_nets[name], [optimizers[name]], buf, "cpu", batch_size=batch_size)
                    else:
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
    eval_decks, or a single A-vs-B pairing when `matchup` is given. Greedy
    (argmax) by default, matching rl.agent/rl.mulligan's own "greedy=True is
    for eval" contract -- eval logs should show the policy's actual best play,
    not an exploration sample; pass greedy=False for the old sampled behavior.
    game_logs (a list) collects one engine
    event_log per game. Returns (RESOLVED deck roster actually played, per-game
    (deck_a, deck_b) pairing list aligned 1:1 with game_logs, or None when
    game_logs is None) -- the roster falls back to the full roster when called
    with None (the caller logs it into the event log's meta so a log written
    without an explicit --decks still records which decks it actually used);
    the pairing list is what lets _write_event_log stamp each game with which
    matchup it actually was, rather than a bare, unlabeled game_index."""
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
    game_pairings = [] if game_logs is not None else None
    for a, b in pairings:
        # record_as = [None, None] and reward_fns = [None, None]: pure play, no
        # buffers, no reward -- record=False ignores them entirely.
        pairing = _constant_pairing(
            [SeatAgent(live_nets[a], mulligan_nets[a], deck_ctxs[a]),
             SeatAgent(live_nets[b], mulligan_nets[b], deck_ctxs[b])],
            [decklists[a], decklists[b]], [None, None], [None, None])
        before = len(game_logs) if game_logs is not None else 0
        _bufs, _mull, played = collect_rollout(pairing, games_per_pairing, horizon, rng, device="cpu",
                                               record=False, greedy=greedy, game_logs=game_logs)
        total += played
        if game_pairings is not None:
            # len(game_logs) - before, not `played`: robust even if collect_rollout
            # ever appends a different count than it reports played.
            game_pairings.extend([(a, b)] * (len(game_logs) - before))
        print(f"  {a} vs {b}: {played} games", flush=True)
    print(f"eval done: {total} games in {time.time() - t0:.1f}s")
    return eval_decks, game_pairings


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


def _write_event_log(log_path, game_logs, meta, game_pairings=None):
    """Write the engine event logs collected this session to PATH as one compact
    JSON doc (no indent -- pretty-printing bloats an event log substantially).
    _json_default salvages any stray non-serializable field. game_pairings (from
    _run_eval, or a single repeated pairing for a --matchup training run), if
    given, stamps each game's own deck_a/deck_b directly onto its record -- so a
    round-robin log with many different pairings in one file is self-describing
    (webapp/replay_engine.list_games reads it), rather than requiring every
    consumer to reconstruct which pairing game N was from meta or the run's
    own stdout print order."""
    os.makedirs(os.path.dirname(log_path) or ".", exist_ok=True)
    games = []
    for i, ev in enumerate(game_logs):
        entry = {"game_index": i}
        if game_pairings is not None:
            entry["deck_a"], entry["deck_b"] = game_pairings[i]
        entry["events"] = ev
        games.append(entry)
    doc = {"meta": meta, "games": games}
    with open(log_path, "w") as f:
        json.dump(doc, f, default=_json_default)
    size_kb = os.path.getsize(log_path) / 1024
    print(f"event log written to {log_path} ({len(game_logs)} games, "
          f"{sum(len(g) for g in game_logs)} total events, {size_kb:.1f} KB)")


def build_arg_parser():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    # Every flag below whose default is None (instead of a real value) is
    # CONFIG-BACKED: main() resolves it as explicit-flag > --run-config /
    # --league-config value > the hardcoded fallback commented alongside each
    # one -- see main()'s own resolution block, the ONE place all three tiers
    # are reconciled. None here means "the user didn't say," never "off" or
    # "zero" -- so a config-backed value can still validly BE 0 or empty
    # without being mistaken for "not given" (e.g. checkpoint_opponent_rate=0.0).
    parser.add_argument("--n-iterations", type=int, default=None,
                         help="Force this exact number of iterations, bypassing auto-sizing entirely (debugging / "
                              "one-off runs only -- never written to progress.json, never fed back into the "
                              "doubling sequence). Omit this for normal training: the size is computed from "
                              "--league-config's total_games and how far that league has already "
                              "gotten (checkpoints/<league_name>/progress.json).")
    parser.add_argument("--snapshot-every", type=int, default=None,
                         help="Snapshot cadence in ITERATIONS. Prefer --run-config's snapshot_every_games (a fixed "
                              "games-count, independent of games_per_iteration) -- this raw flag is a lower-level "
                              "override. Default 20 if neither is given.")
    parser.add_argument("--n-workers", type=int, default=None, help="Default 6 (run config).")
    parser.add_argument("--matchup", nargs=2, metavar=("DECK_A", "DECK_B"), default=None,
                         help="Fixed A-vs-B pairing instead of league opponent sampling (snapshotting off, no "
                              "auto-sizing). Trains both decks with their real mulligan models via the unified loop.")
    parser.add_argument("--games", type=int, default=50, help="Total games (per deck round) for --matchup mode.")
    parser.add_argument("--decks", type=str, default=None, metavar="A,B,...",
                         help="Train only this comma-separated subset of the roster; the rest stay loaded as FROZEN "
                              "opponents (onboarding a new deck / targeted retraining). Falls back to --league-config's "
                              "own train_decks, then to the full roster (train everyone in it).")
    parser.add_argument("--train-deck-only", action="store_true",
                         help="Train the per-deck policies only; freeze the mulligan models.")
    parser.add_argument("--train-mulligan-only", action="store_true",
                         help="Train the mulligan models only; freeze the per-deck policies (a clean bandit vs fixed skill).")
    parser.add_argument("--eval", action="store_true",
                         help="Eval / log-generation: play games with NO training over the current live agents "
                              "(round-robin with mirrors over --decks, or one A-vs-B pairing with --matchup). "
                              "--games games per pairing.")
    parser.add_argument("--sampled", action="store_true",
                         help="Eval: sample from the policy instead of argmaxing (default: greedy/argmax, so logged "
                              "eval games show the policy's actual best play; pass this for the old exploratory-"
                              "sampled behavior).")
    parser.add_argument("--seed", type=int, default=None,
                         help="Seed the rng (reproducible eval logs; also reproducible training sampling).")
    parser.add_argument("--log", type=str, default=None, metavar="PATH",
                         help="Write the game engine's own event log for every game this session to PATH as JSON "
                              "(sequential collection only).")
    parser.add_argument("--checkpoint-opponent-rate", type=float, default=None,
                         help="Probability that a sampled opponent is a frozen historical snapshot rather than that "
                              "deck's current live net, independent of how many snapshots exist (rl.league.LeaguePool."
                              "sample_opponent). Default 0.0 (run config): every game is real-model-vs-real-model, no "
                              "checkpoint opponents at all -- deliberately off during early training, when the "
                              "snapshot pool is mostly barely-trained early copies. The one run-config value expected "
                              "to be overridden deliberately per-invocation (e.g. a scheduled 0.0 -> 0.20 switch "
                              "partway through a long run), rather than edited in the file.")
    parser.add_argument("--league-name", type=str, default=None,
                         help="Store this session's checkpoints under checkpoints/<league-name>/ instead of the "
                              "default checkpoints/league/. Prefer --league-config, which sets this AND the roster "
                              "together; this flag is the lower-level override (or use it alone with no config at "
                              "all, matching the pre-config-file interface).")
    parser.add_argument("--roster", type=str, default=None, metavar="A,B,...",
                         help="Restrict the ENTIRE opponent pool (not just which decks train -- see --decks) to "
                              "this comma-separated subset: a true isolated sub-league where no deck outside the "
                              "set is ever loaded, trained, or sampled as an opponent. Reuses the full roster's "
                              "vocab/shared stack unchanged. Prefer --league-config; this is the lower-level override.")
    parser.add_argument("--total-games", type=int, default=None,
                         help="Games/deck this league trains to in total -- drives auto-sizing and the stop "
                              "condition. Default: --league-config's own total_games.")
    parser.add_argument("--run-config", type=str, default=None, metavar="PATH",
                         help="JSON of run-mechanics defaults shared across leagues (n_workers, snapshot_every_games, "
                              "checkpoint_opponent_rate) -- see training_configs/run_default.json. Any individual "
                              "value can still be overridden by passing its own flag explicitly for one invocation.")
    parser.add_argument("--league-config", type=str, default=None, metavar="PATH",
                         help="JSON describing one league (league_name, roster, optional train_decks, total_games) "
                              "-- see training_configs/league_*.json. Drives automatic batch sizing "
                              "whenever --n-iterations is not given.")
    return parser


def main():
    args = build_arg_parser().parse_args()

    # Both configs optional -- {} when omitted, so every .get() below falls
    # through to its own hardcoded default exactly as if no config existed
    # at all (the pre-config-file interface keeps working unchanged).
    run_cfg = json.load(open(args.run_config)) if args.run_config else {}
    league_cfg = json.load(open(args.league_config)) if args.league_config else {}

    train_deck = not args.train_mulligan_only
    train_mulligan = not args.train_deck_only
    assert train_deck or train_mulligan, "cannot freeze BOTH layers (--train-deck-only + --train-mulligan-only)"

    # Deck identity: explicit flag > --league-config > (roster itself, for
    # train_decks -- omitting it from the config means "train everyone in
    # the roster," _run_session's own existing default).
    roster = args.roster.split(",") if args.roster else league_cfg.get("roster")
    train_decks = args.decks.split(",") if args.decks else league_cfg.get("train_decks", roster)
    matchup = tuple(args.matchup) if args.matchup else None
    game_logs = [] if args.log else None
    league_name = args.league_name or league_cfg.get("league_name")
    league_dir = f"{CHECKPOINT_DIR}/{league_name}" if league_name else None

    if args.eval:  # no training: round-robin (or single matchup) over current live agents
        resolved_decks, game_pairings = _run_eval(train_decks, args.games, not args.sampled, args.seed, game_logs,
                                                    matchup=matchup, league_dir=league_dir)
        if args.log:
            # "decks" logs the RESOLVED roster _run_eval actually played, not the raw
            # --decks arg -- train_decks is None whenever --decks was omitted (the
            # common case), which is not the same as "no decks played."
            _write_event_log(args.log, game_logs, {"mode": "eval", "matchup": list(matchup) if matchup else None,
                                                   "decks": resolved_decks, "greedy": not args.sampled, "games_logged": len(game_logs)},
                              game_pairings=game_pairings)
        return

    # Run mechanics: explicit flag > --run-config > hardcoded default -- the
    # SAME three-tier resolution as deck identity above, just against
    # run_cfg instead of league_cfg.
    n_workers = args.n_workers if args.n_workers is not None else run_cfg.get("n_workers", 6)
    checkpoint_rate = args.checkpoint_opponent_rate if args.checkpoint_opponent_rate is not None else run_cfg.get("checkpoint_opponent_rate", 0.0)

    # Sizing: --matchup counts by --games (per deck round, its own scheme,
    # unaffected by any of this); an explicit --n-iterations forces an exact
    # size and is a pure debug escape hatch (see its own --help text -- never
    # written to progress.json, never fed back into the doubling sequence);
    # otherwise this league's own total_games + how far it's already gotten
    # (progress.json) determine the next batch automatically.
    # Logging is threaded through BOTH the sequential and MP league paths
    # (event dicts are picklable), so --log does not force sequential collection.
    auto_sizing = False
    if matchup is not None:
        # Matchup mode never parallelizes across workers (always sequential,
        # see `sequential` below) -- a flat floor of 10 is all this has ever
        # needed; used to be a CLI-configurable value, but the clamp already
        # forced it to 10 for anyone who didn't explicitly push it higher.
        games_per_iteration = min(10, args.games)
        n_iterations = max(1, args.games // games_per_iteration)
    else:
        # One game per worker: collect_rollout_league_parallel splits n_games
        # across n_workers via plain `n_games // n_workers` -- fewer games than
        # workers silently starves the rest (they get zero games and are never
        # even submitted). Used to be an independent, CLI-tunable value
        # (default 2) that could silently conflict with --n-workers; benchmarked
        # (src/benchmarking/training_run.py) against 1x/2x/3x n_workers on this
        # machine and 1x was both the simplest (never under-provisions) and the
        # fastest measured (2x/3x add PPO-update-side cost -- a bigger buffer to
        # update on -- without adding real collection parallelism once every
        # worker already has work).
        games_per_iteration = max(1, n_workers)
        if args.n_iterations is not None:
            n_iterations = args.n_iterations
        else:
            total_games = args.total_games if args.total_games is not None else league_cfg.get("total_games")
            assert league_dir is not None, (
                "no --n-iterations given and no league to auto-size from -- pass --league-config (or at least "
                "--league-name together with --total-games), or force an exact size with "
                "--n-iterations for a one-off debug run"
            )
            assert total_games is not None, (
                f"auto-sizing needs total_games -- got {total_games!r} (from --league-config, or --total-games directly)"
            )
            next_batch_games = _next_batch_games(league_dir, total_games)
            if next_batch_games is None:
                progress = _load_progress(league_dir)
                print(f"{league_name!r} already at {progress['cumulative_games_per_deck']}/{total_games} "
                      f"games/deck -- nothing to run")
                return
            n_iterations = max(1, next_batch_games // games_per_iteration)
            auto_sizing = True

    # snapshot_every_games (a fixed games-count) is the preferred run-config
    # path; the raw --snapshot-every flag (iterations) is a lower-level
    # override for either it or the config -- converted using games_per_
    # iteration ABOVE this line since it's already fully resolved by here.
    if args.snapshot_every is not None:
        snapshot_every = args.snapshot_every
    elif "snapshot_every_games" in run_cfg:
        snapshot_every = max(1, run_cfg["snapshot_every_games"] // games_per_iteration)
    else:
        snapshot_every = 20

    schedule_kwargs = dict(seed=args.seed,
                            train_deck=train_deck, train_mulligan=train_mulligan, train_decks=train_decks,
                            matchup=matchup, game_logs=game_logs, checkpoint_rate=checkpoint_rate,
                            league_dir=league_dir, roster=roster)

    sequential = matchup is not None or n_workers <= 1  # matchup uses collect_rollout directly (no worker path)
    if not sequential:
        # ONE executor for the whole session, reused across every iteration --
        # process-spawn/import overhead is paid once, not per collection round.
        with ProcessPoolExecutor(max_workers=n_workers) as executor:
            _run_session(n_iterations, games_per_iteration, snapshot_every, executor, n_workers, **schedule_kwargs)
    else:
        _run_session(n_iterations, games_per_iteration, snapshot_every, None, 1, **schedule_kwargs)

    # Feed this batch's own size back into the doubling sequence and advance
    # the league's cumulative count -- skipped entirely for a forced
    # --n-iterations run (auto_sizing stays False), per --n-iterations' own
    # "never written to progress.json" contract above.
    if auto_sizing:
        progress = _load_progress(league_dir)
        played_this_batch = n_iterations * games_per_iteration
        _save_progress(league_dir, played_this_batch, progress["cumulative_games_per_deck"] + played_this_batch)

    if args.log:
        meta = {"mode": "matchup" if matchup else "league", "matchup": list(matchup) if matchup else None,
                "train_decks": train_decks, "games_logged": len(game_logs)}
        # One fixed pairing for the whole run (--log is only ever wired through
        # --matchup mode) -- same schema as _run_eval's per-game pairings, just
        # constant, so every log this script can produce is consistently shaped.
        game_pairings = [matchup] * len(game_logs) if matchup else None
        _write_event_log(args.log, game_logs, meta, game_pairings=game_pairings)


if __name__ == "__main__":
    main()
