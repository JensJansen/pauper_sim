"""run_league.py's reusable core: session driving (_run_session), eval-mode
functions (_run_eval and the vs_history/vs_gauntlet/vs_heuristic checks),
checkpoint/progress helpers, and the per-deck network builder. Extracted so
other callers (e.g. benchmarking/training_run.py) can import these
side-effect-free functions without importing the run_league.py CLI script."""
import copy
import json
import os
import random
import time

import torch

from rl.rewards import deploy_reward_v6
from rl.model.arch import SetTransformer
from rl.model.deck import DeckNetwork
from rl.league.league import LeaguePool, PFSP_POWER
from rl.roster import build_pool
from rl.decision.agent import SeatAgent
from rl.decision.heuristic_agent import HeuristicAgent
from rl.training.train import batch_size_for_iteration, collect_rollout, collect_rollout_league, _constant_pairing, ent_coef_schedule
from rl.training.rollout_parallel import collect_rollout_league_parallel
from rl.training.ppo import ppo_update
from rl.model.mulligan import MulliganNet, update as mulligan_update
from rl import checkpoint as ckpt_io
from repo_paths import CHECKPOINTS_DIR

LEAGUE_DIR = CHECKPOINTS_DIR / "league"
D_MODEL = 64  # SetTransformer width; must match rl.model.arch.SetTransformer's own d_model default
# Turn limit for a single game, for training AND every eval alike -- a
# training/eval mismatch here is silent, since games would simply end at a
# different point and every win rate would measure something else.
HORIZON = 120

# Games per in-training eval check (vs_history / vs_gauntlet / vs_heuristic).
# Raising EVAL_GAMES and eval_every_sessions together trades reading
# frequency for precision at the same total compute cost. Config-driven
# (run-config "eval_games" / "eval_every_sessions") since the right point on
# that trade depends on session size.
EVAL_GAMES = 20
EVAL_EVERY_SESSIONS = 1

# DeckNetwork trunk widths, per deck (each deck's SetTransformer encoder adds
# a further 117,056 params on top of this, and trains with it). Per-league
# configurable. Changing it invalidates existing live.pt/snapshot shapes, so
# it only applies to a fresh league -- _run_session reads the width back off
# an existing live.pt when resuming, and uses this only for a deck that has
# none yet.
TRUNK_HIDDEN = (128, 128)

# Optimizer/PPO knobs, overridable per league via a run-config "ppo" object.
# The value in force is recorded in metrics.jsonl's session_start record.
PPO_DEFAULTS = {
    # league_runner's per-deck Adam learning rate.
    #
    # Resuming a league whose live.pt already carries optimizer state:
    # load_deck_checkpoint's optimizer restore overwrites this freshly
    # constructed Adam's lr with whatever the checkpoint saved, so an
    # existing league keeps its old lr on resume until that checkpoint's
    # "optimizer" key is gone (a migrated/legacy live.pt, or a fresh Adam).
    "lr": 2e-4,
    # Minibatch ramp, in EPISODES per minibatch, not transitions -- ppo_update
    # replays whole episodes through the GRU in order, so its unit of
    # sampling is a trajectory, not a step.
    "seq_batch_size_start": 4,
    "seq_batch_size_cap": 16,
    "mulligan_lr": 1e-3,   # the mulligan net's own, deliberately separate optimizer
    "gamma": 0.99,
    "gae_lambda": 0.95,    # with gamma, a 0.9405/step advantage decay
    "target_kl": 0.03,
    "n_epochs": 4,
    "adv_norm_floor": 0.0,  # floor on the advantage-normalization divisor (rl.training.ppo.ppo_update)
    # Entropy bonus. None uses ent_coef_schedule's 0.02 -> 0.005 anneal; a
    # float pins it constant for the whole run instead.
    "ent_coef": None,
}


def build_deck_net(vocab_size, n_actions, trunk_hidden=None):
    """One deck's full network: its own perception encoder (rl.model.arch)
    plus the trunk/critic/pointer heads built over it. Every construction
    site goes through here so no caller has to remember that the encoder is
    built from the pool vocab size and film_condition_dim tracks its
    d_model.

    Each deck gets its own encoder rather than sharing one -- a checkpoint
    carries its own, so cross-population comparison is valid by
    construction."""
    encoder = SetTransformer(vocab_size)
    return DeckNetwork(encoder, film_condition_dim=encoder.d_model, non_targeting_n_actions=n_actions,
                       trunk_hidden=trunk_hidden or TRUNK_HIDDEN)


def _save_live_checkpoints(live_nets, optimizers, deck_names, session, session_path, league_dir,
                           mulligan_nets=None, mulligan_optimizers=None):
    """Persists every deck's current live net + optimizer + the session
    counter. Called at each snapshot point and at session end -- not only at
    the end, so a mid-session crash loses at most snapshot_every_games
    games/deck of training rather than the whole session."""
    os.makedirs(league_dir, exist_ok=True)
    for name in deck_names:
        deck_dir = f"{league_dir}/{name}"
        ckpt_io.save_deck_checkpoint(f"{deck_dir}/live.pt", live_nets[name], optimizers[name])
        if mulligan_nets is not None:
            ckpt_io.save_deck_checkpoint(f"{deck_dir}/mulligan.pt", mulligan_nets[name], mulligan_optimizers[name])
    with open(session_path, "w") as f:
        f.write(str(session))


def _load_progress(league_dir):
    """This league's auto-sizing state (checkpoints/<league_name>/
    progress.json), only ever written by _save_progress. Absent (a brand-new
    league) reads as "nothing run yet"."""
    path = f"{league_dir}/progress.json"
    if not os.path.exists(path):
        return {"last_batch_size": 0, "cumulative_games_per_deck": 0}
    return json.load(open(path))


def _save_progress(league_dir, last_batch_size, cumulative_games_per_deck):
    os.makedirs(league_dir, exist_ok=True)
    with open(f"{league_dir}/progress.json", "w") as f:
        json.dump({"last_batch_size": last_batch_size, "cumulative_games_per_deck": cumulative_games_per_deck}, f)


def _append_metric(league_dir, **fields):
    """Appends one JSON line to checkpoints/<league>/metrics.jsonl -- every
    call site tags its own `kind` ("ppo" / "mulligan" / "vs_history") so
    report_metrics.py can group by it. Does not create league_dir itself --
    the caller's LeaguePool(league_dir, ...) construction already does."""
    with open(f"{league_dir}/metrics.jsonl", "a") as f:
        f.write(json.dumps(fields) + "\n")


def should_snapshot(games_before, games_per_iteration, snapshot_every):
    """True when the iteration about to be trained -- which takes this league
    from `games_before` to `games_before + games_per_iteration` games/deck --
    crosses a multiple of the snapshot cadence.

    `games_before` must be cumulative across every session this league has
    ever run (progress.json), not session-local -- `iteration` restarts at 0
    every process invocation, so a session-local test would never fire for a
    session shorter than snapshot_every and would lose the remainder at
    every session boundary.

    snapshot_every is in iterations (run_league.py converts run-config's
    snapshot_every_games by floor division); multiplying back recovers the
    games-count."""
    snapshot_every_games = snapshot_every * games_per_iteration
    return (games_before + games_per_iteration) // snapshot_every_games > games_before // snapshot_every_games


def checkpoint_progress(league_dir, cumulative_games_per_deck):
    """Mid-session write of the cumulative counter, at the snapshot cadence
    -- keeps progress.json in step with _save_live_checkpoints' own
    incremental writes, so a mid-session crash doesn't leave the counter
    behind the weights on disk.

    Writes the counter absolutely (caller passes games-so-far, not a delta),
    so calling it repeatedly within one session is idempotent. Leaves
    last_batch_size alone -- only advance_progress touches that."""
    _save_progress(league_dir, _load_progress(league_dir)["last_batch_size"], cumulative_games_per_deck)


def advance_progress(league_dir, n_iterations, games_per_iteration, auto_sizing,
                     session_start_games=None):
    """Records what a finished batch did. Returns the new progress dict.

    Two separate things live in progress.json and only one is gated on
    auto_sizing:

      last_batch_size            the doubling ladder's feedback; a forced
                                 size (--n-iterations) never perturbs it.
      cumulative_games_per_deck  how much this league has ever trained --
                                 the horizon batch_size_for_iteration and
                                 ent_coef_schedule ramp against, so it must
                                 always advance.

    session_start_games: given the session's own starting count, the new
    value is computed absolutely (start + played), so it's idempotent no
    matter how many mid-session checkpoint_progress writes already landed.
    Omitting it keeps the old additive behavior for callers that never
    checkpoint mid-session."""
    progress = _load_progress(league_dir)
    played = n_iterations * games_per_iteration
    last_batch_size = played if auto_sizing else progress["last_batch_size"]
    base = progress["cumulative_games_per_deck"] if session_start_games is None else session_start_games
    cumulative = base + played
    _save_progress(league_dir, last_batch_size, cumulative)
    return {"last_batch_size": last_batch_size, "cumulative_games_per_deck": cumulative}


def _next_batch_games(league_dir, total_games):
    """The next auto-sized batch's games-per-deck: doubles from the last
    real batch this league actually ran (1 if none yet), never overshooting
    total_games on the final batch. Returns None once total_games is already
    met -- the caller's signal to run nothing at all.

    No separate ceiling here -- the doubling ladder's safety property (never
    jump from a small, verified-healthy batch to a huge one) is enforced by
    whatever repeatedly re-invokes this and health-checks between calls."""
    progress = _load_progress(league_dir)
    remaining = total_games - progress["cumulative_games_per_deck"]
    if remaining <= 0:
        return None
    next_size = progress["last_batch_size"] * 2 if progress["last_batch_size"] > 0 else 1
    return min(next_size, remaining)


def _run_session(n_iterations, games_per_iteration, snapshot_every, executor, n_workers,
                  league_dir=None, seed=None,
                  train_deck=True, train_mulligan=True, train_decks=None,
                  matchup=None, game_logs=None, checkpoint_rate=0.0, roster=None, pfsp=True,
                  gauntlet_league_dir=None, heuristic_decks=(), cumulative_games=0,
                  ppo_hparams=None, eval_games=EVAL_GAMES, eval_every_sessions=EVAL_EVERY_SESSIONS,
                  pfsp_power=PFSP_POWER, trunk_hidden=TRUNK_HIDDEN, device="cpu"):
    # device: where the nets live, and where ppo_update's gradient work runs.
    # Only the update moves -- collection always runs on CPU (inference over
    # one game state at a time in worker processes), so eval paths and
    # workers keep using CPU nets.
    #
    # cumulative_games: this league's games/deck already played before this
    # session started -- batch_size_for_iteration and ent_coef_schedule ramp
    # against cumulative_games + iteration * games_per_iteration, so the
    # ramp grows smoothly within a session and resumes correctly across
    # sessions.
    #
    # league_dir + seed let a benchmark harness drive this exact loop over a
    # throwaway dir, reproducibly. Defaults preserve the real run.
    #
    # train_deck / train_mulligan gate which layer updates; train_decks (a
    # subset, default the whole roster) gates which decks train -- the rest
    # stay loaded as frozen opponents (still played, never updated or
    # snapshotted).
    league_dir = league_dir or LEAGUE_DIR
    if seed is not None:
        torch.manual_seed(seed)  # identical fresh stack + in-process sampling across benchmark configs
        random.seed(seed)  # collect_rollout_league_parallel draws its worker seeds from the global random module
    decklists, vocab, deck_ctxs, fixed_tables = build_pool()
    deck_names = list(decklists)
    # roster: a true isolated sub-league -- restricts the entire opponent
    # pool (not just which decks train) to this subset. vocab/deck_ctxs/
    # fixed_tables still come from the full build_pool() roster, preserving
    # checkpoint/vocab compatibility, but this session never loads, trains,
    # or pairs against anyone outside the subset. Unlike train_decks alone,
    # which still samples opponents from the whole roster (for onboarding a
    # new deck against an established field).
    if roster is not None:
        assert set(roster) <= set(deck_names), f"roster {roster} not all in the full pool roster {deck_names}"
        deck_names = list(roster)
    # matchup = (A, B): a fixed pairing instead of league sampling
    # (snapshotting off); trains exactly those two decks.
    if matchup is not None:
        assert len(matchup) == 2, "matchup must name exactly two decks"
        train_decks = list(matchup)
    train_decks = list(train_decks) if train_decks is not None else deck_names
    assert set(train_decks) <= set(deck_names), f"train_decks {train_decks} not all in roster {deck_names}"
    train_set = set(train_decks)
    # ppo_hparams: partial override of PPO_DEFAULTS. Unknown keys are a hard
    # error, not a silent no-op.
    hp = {**PPO_DEFAULTS, **(ppo_hparams or {})}
    unknown = set(hp) - set(PPO_DEFAULTS)
    assert not unknown, f"unknown ppo hyperparameter(s) {sorted(unknown)}; known: {sorted(PPO_DEFAULTS)}"
    live_nets, optimizers = {}, {}
    for name in deck_names:
        # Trunk width: reused from an existing live.pt when resuming; the
        # configured width only for a genuinely fresh deck, so a config edit
        # can't shape-mismatch a league already on disk.
        live_path = f"{league_dir}/{name}/live.pt"
        width = ckpt_io.trunk_hidden_from_deck_checkpoint(live_path) or trunk_hidden
        net = build_deck_net(vocab.size, len(fixed_tables[name]), width)
        # .to(device) must happen before the optimizer is built and before
        # the checkpoint load: Adam's state tensors are created lazily on
        # the param's device, so building the optimizer first would strand
        # them on CPU while the params sit on GPU.
        net.to(device)
        optimizer = torch.optim.Adam([p for p in net.parameters() if p.requires_grad], lr=hp["lr"])
        ckpt_io.load_deck_checkpoint(live_path, net, optimizer)  # no-op if live_path doesn't exist yet; optimizer load is migration-guarded
        resumed_lr = optimizer.param_groups[0]["lr"]
        if resumed_lr != hp["lr"]:
            # A resumed checkpoint's own saved Adam state wins over hp["lr"].
            # Print-only: the mismatch might be intentional, so this just
            # makes it visible rather than overriding either way.
            print(f"  [{name}] resumed optimizer lr={resumed_lr} differs from config lr={hp['lr']} (checkpoint's lr wins)")
        live_nets[name] = net
        optimizers[name] = optimizer

    # Fixed for the whole session -- computed once rather than re-derived on
    # every collect_rollout_league_parallel call. The encoder isn't
    # broadcast this way since it trains and ships fresh each call.
    all_trunk_hidden = {name: tuple(layer.out_features for layer in net.trunk_layers) for name, net in live_nets.items()}

    # Per-deck mulligan model: owns the pregame keep/mulligan + bottoming,
    # trained by its own REINFORCE with a direct game-outcome reward,
    # decoupled from the main PPO above.
    mulligan_nets, mulligan_optimizers = {}, {}
    for name in deck_names:
        mnet = MulliganNet(live_nets[name].encoder)
        # The encoder is shared with the deck net (already on `device`), but
        # MulliganNet's own heads were just built on CPU -- move before the
        # optimizer exists, same ordering rule as the deck net above.
        mnet.to(device)
        mopt = torch.optim.Adam([p for p in mnet.parameters() if p.requires_grad], lr=hp["mulligan_lr"])
        mull_path = f"{league_dir}/{name}/mulligan.pt"
        ckpt_io.load_deck_checkpoint(mull_path, mnet, mopt)  # same migration-guarded optimizer load as the live-net path above
        resumed_mull_lr = mopt.param_groups[0]["lr"]
        if resumed_mull_lr != hp["mulligan_lr"]:
            print(f"  [{name}] resumed mulligan optimizer lr={resumed_mull_lr} differs from config lr={hp['mulligan_lr']} (checkpoint's lr wins)")
        mulligan_nets[name] = mnet
        mulligan_optimizers[name] = mopt

    # CPU mirrors for every path that plays rather than trains (the
    # sequential collect fallback, --matchup pairings, and the end-of-session
    # eval checks all run through collect_rollout(device="cpu")). A
    # GPU-resident net would be a device mismatch there, so on a GPU league
    # these mirrors are refreshed from the training weights on demand. When
    # device == "cpu" they're the same objects, not copies.
    if device == "cpu":
        cpu_nets, cpu_mulligan_nets = live_nets, mulligan_nets

        def sync_cpu_mirrors():
            pass
    else:
        cpu_nets = {n: copy.deepcopy(net).cpu().eval() for n, net in live_nets.items()}
        # MulliganNet holds its encoder as a plain reference, not a
        # registered child, so nn.Module.cpu()/load_state_dict on a deepcopy
        # leaves the mirror's trunk wired to the original CUDA encoder.
        # Repointing at the mirrored deck net's encoder fixes it, mirroring
        # the same shared-encoder relationship the live pair has.
        cpu_mulligan_nets = {}
        for n, m in mulligan_nets.items():
            mirror = copy.deepcopy(m).cpu().eval()
            # object.__setattr__, matching MulliganNet.__init__: a plain
            # assignment would register the encoder as a child, and
            # sync_cpu_mirrors below would then fail on missing encoder.* keys.
            object.__setattr__(mirror, "encoder", cpu_nets[n].encoder)
            cpu_mulligan_nets[n] = mirror

        def sync_cpu_mirrors():
            """Pull the current training weights down for the play paths."""
            for n in deck_names:
                cpu_nets[n].load_state_dict({k: v.cpu() for k, v in live_nets[n].state_dict().items()})
                cpu_mulligan_nets[n].load_state_dict(
                    {k: v.cpu() for k, v in mulligan_nets[n].state_dict().items()})

    pool = LeaguePool(league_dir, deck_names, pfsp_power=pfsp_power)
    session_path = f"{league_dir}/session.txt"
    session = int(open(session_path).read()) + 1 if os.path.exists(session_path) else 0
    if session > 0:
        print(f"resumed league (session {session}); snapshots on disk: "
              f"{ {name: len(pool.snapshots[name]) for name in deck_names} }")

    rng = random.Random(seed)  # seed=None -> nondeterministic, identical to the prior random.Random()
    reward_fn = deploy_reward_v6
    reward_fn_name = "deploy_reward_v6"
    horizon = HORIZON

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
    # One session_start header per session makes metrics.jsonl
    # self-describing: which reward/roster/config produced a stretch of
    # records, and the league's cumulative games/deck at that point.
    _append_metric(league_dir, kind="session_start", session=session,
                   cumulative_games=cumulative_games, reward_fn=reward_fn_name,
                   roster=list(deck_names), train_decks=sorted(train_set),
                   n_iterations=n_iterations, games_per_iteration=games_per_iteration,
                   snapshot_every=snapshot_every, checkpoint_rate=checkpoint_rate,
                   pfsp=pfsp, n_workers=n_workers, horizon=horizon,
                   gauntlet=gauntlet_league_dir is not None,
                   eval_games=eval_games, eval_every_sessions=eval_every_sessions,
                   pfsp_power=pfsp_power, **hp)
    t0 = time.time()
    total_games = 0
    collect_time_total = 0.0
    update_time_total = 0.0
    # Mulligan transitions accumulate across iterations, flushed to a
    # REINFORCE update only every MULLIGAN_UPDATE_EVERY iterations rather
    # than on each iteration's small, correlated batch.
    MULLIGAN_UPDATE_EVERY = 8
    mull_by_deck_accum = {name: [] for name in train_decks}
    for iteration in range(n_iterations):
        iter_t0 = time.time()
        games_this_iter = 0
        print(f"  iter {iteration}: start at {time.strftime('%H:%M:%S', time.localtime(iter_t0))}", flush=True)
        # Both the PPO minibatch ramp and the entropy-coefficient anneal
        # (the latter only when hp["ent_coef"] is None) track true
        # cumulative games/deck: this session's starting point plus games
        # played so far this session.
        games_so_far_this_session = iteration * games_per_iteration
        batch_size = batch_size_for_iteration(cumulative_games + games_so_far_this_session,
                                              start=hp["seq_batch_size_start"], cap=hp["seq_batch_size_cap"])
        # hp["ent_coef"] pins a constant and skips the anneal; None keeps the
        # 0.02 -> 0.005 schedule.
        ent_coef = (hp["ent_coef"] if hp["ent_coef"] is not None
                    else ent_coef_schedule(cumulative_games + games_so_far_this_session))
        # Snapshot the mulligan nets for the workers (all decks -- a frozen
        # opponent still needs one to play). .cpu() since these cross a
        # process boundary to CPU-only workers.
        mulligan_state_dicts = {n: {k: v.cpu() for k, v in mulligan_nets[n].state_dict().items()}
                                for n in deck_names}
        for name in train_decks:  # only TRAIN decks get a round; the rest are frozen opponents
            t_collect0 = time.time()
            # Both CPU-playing collect paths read the mirrors, so refresh
            # first. Not called on the parallel path, which ships state_dicts
            # itself and never touches a mirror.
            if matchup is not None or executor is None:
                sync_cpu_mirrors()
            if matchup is not None:
                # Fixed pairing: this deck vs the other named deck, both
                # carrying their real mulligan models so logged games match
                # training's pregame policy.
                opp = matchup[1] if name == matchup[0] else matchup[0]
                pairing = _constant_pairing(
                    [SeatAgent(cpu_nets[name], cpu_mulligan_nets[name], deck_ctxs[name]),
                     SeatAgent(cpu_nets[opp], cpu_mulligan_nets[opp], deck_ctxs[opp])],
                    [decklists[name], decklists[opp]], [reward_fn, reward_fn], [name, opp])
                buffers_by_deck, mull_by_deck, played = collect_rollout(
                    pairing, games_per_iteration, horizon, rng, device="cpu", game_logs=game_logs)
            elif executor is not None:
                buffers_by_deck, mull_by_deck, played, outcomes = collect_rollout_league_parallel(
                    name, live_nets, reward_fn_name, league_dir, horizon, games_per_iteration,
                    executor, n_workers, all_trunk_hidden,
                    mulligan_state_dicts, game_logs=game_logs, checkpoint_rate=checkpoint_rate, pfsp=pfsp,
                    pfsp_power=pfsp_power,
                )
            else:
                buffers_by_deck, mull_by_deck, played, outcomes = collect_rollout_league(
                    name, cpu_nets, cpu_mulligan_nets, deck_ctxs, decklists, pool, reward_fn,
                    horizon, games_per_iteration, rng, device="cpu", game_logs=game_logs,
                    checkpoint_rate=checkpoint_rate, pfsp=pfsp,
                )
            if matchup is None:
                # Feed every game's real outcome into the one authoritative
                # pool so PFSP weighting updates within the session, not
                # just across process invocations.
                for opp_name, snap_id, won in outcomes:
                    pool.record_outcome(name, ("deck", opp_name), won)
                    if snap_id is not None:
                        pool.record_outcome(name, ("snapshot", opp_name, snap_id), won)
            collect_time_total += time.time() - t_collect0
            total_games += played
            games_this_iter += played
            # Accumulate mulligan transitions for each train deck that
            # generated some this round; a frozen opponent's are discarded.
            # Gated on train_mulligan to match the flush below.
            for deck_name, tr in mull_by_deck.items():
                if train_mulligan and deck_name in train_set:
                    mull_by_deck_accum[deck_name].extend(tr)
            t_update0 = time.time()
            # PPO-updates every train deck that received transitions this
            # round: the training deck's own bucket, plus any live-opponent
            # bucket in train_set (buffers_by_deck only ever holds
            # live-opponent buckets; a frozen-snapshot opponent records
            # nothing).
            policy_loss = value_loss = entropy = approx_kl = clip_fraction = 0.0
            epochs_run = 0
            salvaged = 0
            if train_deck:
                for deck_name, buf in buffers_by_deck.items():
                    if not len(buf) or deck_name not in train_set:
                        continue
                    if deck_name == name:
                        (policy_loss, value_loss, entropy, approx_kl, clip_fraction,
                         epochs_run, explained_variance, adv_std) = ppo_update(
                            live_nets[name], optimizers[name], buf, device, batch_size=batch_size,
                            ent_coef=ent_coef, gamma=hp["gamma"], gae_lambda=hp["gae_lambda"],
                            target_kl=hp["target_kl"], n_epochs=hp["n_epochs"],
                            adv_norm_floor=hp["adv_norm_floor"])
                    else:
                        ppo_update(live_nets[deck_name], optimizers[deck_name], buf, device,
                                   batch_size=batch_size, ent_coef=ent_coef, gamma=hp["gamma"],
                                   gae_lambda=hp["gae_lambda"], target_kl=hp["target_kl"],
                                   n_epochs=hp["n_epochs"], adv_norm_floor=hp["adv_norm_floor"])
                        salvaged += len(buf)
            update_time_total += time.time() - t_update0
            buffer_size = len(buffers_by_deck.get(name, ()))
            print(f"  iter {iteration} [{name}]: games={played} buf={buffer_size} "
                  f"salvaged={salvaged} batch_size={batch_size} ent_coef={ent_coef:.4f} "
                  f"policy_loss={policy_loss:.4f} value_loss={value_loss:.4f} "
                  f"approx_kl={approx_kl:.4f} clip_frac={clip_fraction:.3f} epochs_run={epochs_run}", flush=True)
            # buffer_size alongside batch_size shows whether the minibatch
            # ramp has saturated past the collected data (once batch_size >=
            # buffer_size, ppo_update runs full-batch steps). approx_kl/
            # clip_fraction/epochs_run give visibility into why entropy
            # moves, not just that it did.
            _append_metric(league_dir, kind="ppo", session=session, iteration=iteration, deck=name,
                           cumulative_games=cumulative_games + games_so_far_this_session,
                           games=played, buffer_size=buffer_size, batch_size=batch_size, salvaged=salvaged,
                           ent_coef=ent_coef, policy_loss=policy_loss, value_loss=value_loss, entropy=entropy,
                           approx_kl=approx_kl, clip_fraction=clip_fraction, epochs_run=epochs_run,
                           explained_variance=explained_variance, adv_std=adv_std,
                           target_kl=hp["target_kl"])

        # Mulligan-model REINFORCE: one step per train deck on transitions
        # accumulated across the last MULLIGAN_UPDATE_EVERY iterations,
        # flushed early on the session's final iteration so nothing is
        # dropped at a session boundary.
        mull_stats = {}
        is_mulligan_flush = (iteration + 1) % MULLIGAN_UPDATE_EVERY == 0 or iteration == n_iterations - 1
        if train_mulligan and is_mulligan_flush:
            for name in train_decks:
                if mull_by_deck_accum[name]:
                    mull_stats[name] = mulligan_update(mulligan_nets[name], mulligan_optimizers[name], mull_by_deck_accum[name])
                    _append_metric(league_dir, kind="mulligan", session=session, iteration=iteration, deck=name,
                                   cumulative_games=cumulative_games + games_so_far_this_session,
                                   n=mull_stats[name]["n"], loss=mull_stats[name]["loss"])
                    mull_by_deck_accum[name] = []
        if mull_stats:  # readout so the mulligan subsystem is visible while it trains
            total_n = sum(s["n"] for s in mull_stats.values())
            mean_loss = sum(s["loss"] for s in mull_stats.values()) / len(mull_stats)
            print(f"  iter {iteration}: mulligan model -- {total_n} transitions across {len(mull_stats)} decks, "
                  f"mean REINFORCE loss {mean_loss:.4f}", flush=True)

        # Snapshot cadence is cumulative, not session-local -- gated on
        # cumulative games crossing a multiple of snapshot_every_games
        # (should_snapshot). snapshot_every arrives in iterations;
        # multiplying back recovers the games-count.
        games_before = cumulative_games + iteration * games_per_iteration
        crossed = should_snapshot(games_before, games_per_iteration, snapshot_every)
        if matchup is None and crossed:  # matchup: no historical snapshots
            for name in train_decks:  # only TRAIN decks change -> only they need new snapshots
                pool.register_snapshot(name, live_nets[name], mulligan_nets[name])
            _save_live_checkpoints(live_nets, optimizers, deck_names, session, session_path, league_dir,
                                   mulligan_nets, mulligan_optimizers)
            pool.save_opponent_stats()  # same cadence as the checkpoints above, not every iteration
            # And the cumulative counter, so progress.json is never left
            # behind the weights those two lines just wrote.
            checkpoint_progress(league_dir, games_before + games_per_iteration)
            print(f"  iter {iteration}: snapshotted {len(train_decks)} train deck(s) + saved live checkpoints "
                  f"(counts now: { {n: len(pool.snapshots[n]) for n in deck_names} })", flush=True)

        iter_done_t = time.time()
        iter_elapsed = iter_done_t - iter_t0
        avg_s_per_game = iter_elapsed / games_this_iter if games_this_iter else float("nan")
        print(f"  iter {iteration}: done at {time.strftime('%H:%M:%S', time.localtime(iter_done_t))}, "
              f"took {iter_elapsed:.1f}s across {games_this_iter} games ({avg_s_per_game:.2f}s/game) "
              f"(session elapsed so far: {iter_done_t - t0:.1f}s)", flush=True)

    elapsed = time.time() - t0
    print(f"session {session} done in {elapsed:.1f}s ({elapsed / total_games:.2f}s/game across {total_games} games) -- "
          f"collect={collect_time_total:.1f}s ({100 * collect_time_total / elapsed:.0f}%), "
          f"update={update_time_total:.1f}s ({100 * update_time_total / elapsed:.0f}%)")

    # Once per session: does the live net beat its own past selves. Only
    # meaningful in league mode (matchup mode never snapshots). Runs
    # sequentially in this process via plain collect_rollout, not the
    # session's own executor/n_workers, so it adds real extra games on top
    # of however small the training batch was.
    cumulative_at_session_end = cumulative_games + n_iterations * games_per_iteration
    # eval_every_sessions > 1 spends the same total eval compute on fewer,
    # bigger, more precise readings. Session 0 always evals.
    run_evals = eval_every_sessions <= 1 or session % eval_every_sessions == 0
    if not run_evals:
        print(f"  (evals skipped: session {session} % eval_every_sessions={eval_every_sessions} != 0)", flush=True)
    if matchup is None and run_evals:
        # Every check below plays on CPU nets, so the mirrors must carry
        # this session's final trained weights before evaluating.
        sync_cpu_mirrors()
        for name in train_decks:
            for r in _run_eval_vs_history(name, cpu_nets[name], cpu_mulligan_nets[name], deck_ctxs[name],
                                          decklists[name], league_dir, horizon,
                                          games_per_snapshot=eval_games, seed=seed):
                _append_metric(league_dir, kind="vs_history", session=session, iteration=iteration, deck=name,
                               cumulative_games=cumulative_at_session_end, **r)
                print(f"  vs-history [{name}] vs {r['label']}: {r['live_wins']}/{r['games']} live wins "
                      f"({r['snapshot_wins']} snapshot wins, {r['no_winner']} no-winner)", flush=True)
            # The gauntlet check (an independently-trained twin population)
            # -- None until that population's training reaches this deck, or
            # no gauntlet_league_dir is configured.
            r = _run_eval_vs_gauntlet(name, cpu_nets[name], cpu_mulligan_nets[name], deck_ctxs[name],
                                      decklists[name], gauntlet_league_dir, horizon,
                                      games=eval_games, seed=seed)
            if r is not None:
                _append_metric(league_dir, kind="vs_gauntlet", session=session, iteration=iteration, deck=name,
                               cumulative_games=cumulative_at_session_end, **r)
                print(f"  vs-gauntlet [{name}]: {r['live_wins']}/{r['games']} live wins "
                      f"({r['gauntlet_wins']} gauntlet wins, {r['no_winner']} no-winner)", flush=True)
            # The tier-1 gauntlet member (HeuristicAgent) -- only for
            # whichever deck(s) heuristic_decks names.
            if name in heuristic_decks:
                r = _run_eval_vs_heuristic(name, cpu_nets[name], cpu_mulligan_nets[name], deck_ctxs[name],
                                           decklists[name], horizon, games=eval_games, seed=seed)
                _append_metric(league_dir, kind="vs_heuristic", session=session, iteration=iteration, deck=name,
                               cumulative_games=cumulative_at_session_end, **r)
                print(f"  vs-heuristic [{name}]: {r['live_wins']}/{r['games']} live wins "
                      f"({r['heuristic_wins']} heuristic wins, {r['no_winner']} no-winner)", flush=True)

    _save_live_checkpoints(live_nets, optimizers, deck_names, session, session_path, league_dir,
                           mulligan_nets, mulligan_optimizers)
    if matchup is None:
        pool.save_opponent_stats()
    print("live checkpoints saved for all decks")


def _run_eval(eval_decks, games_per_pairing, greedy, seed, game_logs, matchup=None,
              league_dir=None):
    """Eval / faithful log generation: plays games with no training
    (record=False, no updates, no checkpointing) over the current live
    agents (deck net + its mulligan model). Pairing is a round-robin with
    mirrors (combinations_with_replacement) over eval_decks, or a single
    A-vs-B pairing when `matchup` is given. Greedy (argmax) by default;
    pass greedy=False for sampled play. game_logs (a list) collects one
    engine event_log per game. Returns (resolved deck roster actually
    played, per-game (deck_a, deck_b) pairing list aligned 1:1 with
    game_logs, or None when game_logs is None) -- the pairing list is what
    lets _write_event_log stamp each game with its real matchup."""
    import itertools
    league_dir = league_dir or LEAGUE_DIR
    decklists, vocab, deck_ctxs, fixed_tables = build_pool()
    deck_names = list(decklists)
    eval_decks = list(matchup) if matchup else (list(eval_decks) if eval_decks else deck_names)
    assert set(eval_decks) <= set(deck_names), f"eval decks {eval_decks} not all in roster {deck_names}"
    pairings = [tuple(matchup)] if matchup else list(itertools.combinations_with_replacement(eval_decks, 2))
    rng = random.Random(seed)
    horizon = HORIZON

    live_nets, mulligan_nets = {}, {}
    for name in set(eval_decks):
        live_path = f"{league_dir}/{name}/live.pt"
        net = build_deck_net(vocab.size, len(fixed_tables[name]),
                             ckpt_io.trunk_hidden_from_deck_checkpoint(live_path))
        ckpt_io.load_deck_checkpoint(live_path, net)  # optimizer=None: eval only ever needs inference weights
        net.eval()
        live_nets[name] = net
        mnet = MulliganNet(net.encoder)
        mull_path = f"{league_dir}/{name}/mulligan.pt"
        ckpt_io.load_deck_checkpoint(mull_path, mnet)
        mnet.eval()
        mulligan_nets[name] = mnet

    print(f"Eval: {len(pairings)} pairing(s) x {games_per_pairing} games "
          f"({'greedy' if greedy else 'sampled'}, seed={seed}) over decks={eval_decks}")
    t0 = time.time()
    total = 0
    game_pairings = [] if game_logs is not None else None
    for a, b in pairings:
        # record_as/reward_fns = [None, None]: pure play, record=False ignores them.
        pairing = _constant_pairing(
            [SeatAgent(live_nets[a], mulligan_nets[a], deck_ctxs[a]),
             SeatAgent(live_nets[b], mulligan_nets[b], deck_ctxs[b])],
            [decklists[a], decklists[b]], [None, None], [None, None])
        before = len(game_logs) if game_logs is not None else 0
        _bufs, _mull, played = collect_rollout(pairing, games_per_pairing, horizon, rng, device="cpu",
                                               record=False, greedy=greedy, game_logs=game_logs)
        total += played
        if game_pairings is not None:
            # len(game_logs) - before, not `played`: robust even if
            # collect_rollout appends a different count than it reports.
            game_pairings.extend([(a, b)] * (len(game_logs) - before))
        print(f"  {a} vs {b}: {played} games", flush=True)
    print(f"eval done: {total} games in {time.time() - t0:.1f}s")
    return eval_decks, game_pairings


def _list_snapshot_ids(dir_path):
    """Sorted snapshot ids (the N in snapshot_N.pt) found directly in
    dir_path, or [] if the directory doesn't exist yet -- shared by the
    active pool dir and its archive/ subdir, which have the same naming."""
    if not os.path.isdir(dir_path):
        return []
    return sorted(int(fn[len("snapshot_"):-len(".pt")]) for fn in os.listdir(dir_path)
                  if fn.startswith("snapshot_") and fn.endswith(".pt"))


def league_roster(league_dir):
    """The deck names this league has actually trained, sorted -- i.e. those
    with a live.pt on disk. build_pool() spans the whole manifest regardless
    of roster, so an eval script iterating its decklists directly would try
    to load checkpoints a smaller league never wrote."""
    if not os.path.isdir(league_dir):
        return []
    return sorted(d for d in os.listdir(league_dir)
                  if os.path.exists(os.path.join(league_dir, d, "live.pt")))


def load_vintage_agent(league_dir, deck_name, vintage, deck_ctx, pool=None):
    """A frozen SeatAgent for one historical point of deck_name: an int/str
    snapshot id, or "live" for the current checkpoint pair.

    Path resolution is the only thing this adds over
    LeaguePool.load_snapshot_agent: a snapshot sits either directly in the
    deck dir (still inside the active sampling window) or under archive/
    once evicted. Callers that walk many vintages should pass a single
    reused `pool` -- load_snapshot_agent caches by path.

    Every vintage brings its own perception encoder (it lives inside the
    checkpoint), so a historical agent is played exactly as it was, rather
    than being replayed through whatever encoder is current."""
    vocab, fixed_table = deck_ctx
    deck_dir = os.path.join(league_dir, deck_name)
    if vintage == "live":
        live_path = os.path.join(deck_dir, "live.pt")
        net = build_deck_net(vocab.size, len(fixed_table),
                             ckpt_io.trunk_hidden_from_deck_checkpoint(live_path))
        ckpt_io.load_deck_checkpoint(live_path, net)
        net.eval()
        mull = MulliganNet(net.encoder)
        ckpt_io.load_deck_checkpoint(os.path.join(deck_dir, "mulligan.pt"), mull)
        mull.eval()
        return SeatAgent(net, mull, deck_ctx)
    for sub in ("", "archive"):
        path = os.path.join(deck_dir, sub, f"snapshot_{int(vintage)}.pt")
        if os.path.exists(path):
            return (pool or LeaguePool(league_dir, [deck_name])).load_snapshot_agent(path, deck_ctx)
    raise FileNotFoundError(f"no snapshot_{vintage}.pt for {deck_name} under {deck_dir} or its archive/")


def _play_paired_eval_games(live_agent, opp_agent, decklist, n_games, horizon, seed, opp_wins_key,
                            greedy=True):
    """_play_eval_games with common random numbers: half the games with
    live_agent at seat 0, half with the sides swapped, both halves driven
    from the same seed.

    Because collect_rollout draws starting_idx per game from that rng,
    replaying the identical seed with the seats exchanged hands the
    play/draw to the other agent on the same shuffles -- on-the-play is
    balanced exactly rather than in expectation, and deck-order variance
    largely cancels between the paired halves. The unpaired
    _play_eval_games always seats live at 0 and leaves both to chance."""
    half = n_games // 2
    fwd = _play_eval_games(live_agent, opp_agent, decklist, half, horizon,
                           random.Random(seed), opp_wins_key, greedy=greedy)
    rev = _play_eval_games(opp_agent, live_agent, decklist, n_games - half, horizon,
                           random.Random(seed), opp_wins_key, greedy=greedy)
    # rev is scored from the opponent's seat, so live's wins there are rev's
    # opp_wins_key column and vice versa.
    return {"games": fwd["games"] + rev["games"],
            "live_wins": fwd["live_wins"] + rev[opp_wins_key],
            opp_wins_key: fwd[opp_wins_key] + rev["live_wins"],
            "no_winner": fwd["no_winner"] + rev["no_winner"],
            "paired": True}


def _play_eval_games(live_agent, opp_agent, decklist, n_games, horizon, rng, opp_wins_key,
                     greedy=True):
    """Shared tail for every vs_history/vs_gauntlet/vs_heuristic eval pass:
    plays a fixed live_agent-vs-opp_agent pairing (no training), scores
    winners via collect_rollout's "game_over" event, and returns
    {"games", "live_wins", <opp_wins_key>, "no_winner"}. opp_wins_key names
    the opponent side (e.g. "snapshot_wins"/"gauntlet_wins"/"heuristic_wins").

    greedy defaults to True (measure the policy's actual best play, not an
    exploration sample); run_anchor_eval.py overrides it since argmax over a
    randomly initialized head is degenerate. Applies to both seats."""
    pairing = _constant_pairing([live_agent, opp_agent], [decklist, decklist], [None, None], [None, None])
    game_logs = []
    _bufs, _mull, played = collect_rollout(pairing, n_games, horizon, rng, device="cpu",
                                           record=False, greedy=greedy, game_logs=game_logs)
    outcomes = [e for ev in game_logs for e in ev if e["kind"] == "game_over"]
    live_wins = sum(1 for e in outcomes if e["winner"] == 0)
    opp_wins = sum(1 for e in outcomes if e["winner"] == 1)
    return {"games": played, "live_wins": live_wins, opp_wins_key: opp_wins,
            "no_winner": played - live_wins - opp_wins}


def _run_eval_vs_history(deck_name, live_net, mulligan_net, deck_ctx, decklist, league_dir,
                          horizon, games_per_snapshot=EVAL_GAMES, seed=None):
    """Plays deck_name's current live net against its own past selves: the
    oldest snapshot still in the active LeaguePool sampling window, plus
    (once any have been evicted) the oldest archived snapshot -- the direct
    answer to "has this policy actually improved."

    Returns [] when there's no history yet (before the first snapshot
    exists). Reuses the caller's already-loaded live_net/deck_ctx/decklist;
    only the historical snapshot is loaded from disk, via
    LeaguePool.load_snapshot_agent. greedy=True, measuring the policy's
    actual best play."""
    deck_dir = f"{league_dir}/{deck_name}"
    milestones = []
    archive_ids = _list_snapshot_ids(os.path.join(deck_dir, "archive"))
    if archive_ids:
        milestones.append(("archive_oldest", archive_ids[0], True,
                           os.path.join(deck_dir, "archive", f"snapshot_{archive_ids[0]}.pt")))
    active_ids = _list_snapshot_ids(deck_dir)
    if active_ids:
        milestones.append(("active_oldest", active_ids[0], False,
                           os.path.join(deck_dir, f"snapshot_{active_ids[0]}.pt")))
    if not milestones:
        return []

    pool = LeaguePool(league_dir, [deck_name])  # only its load_snapshot_agent loader is used here
    live_agent = SeatAgent(live_net, mulligan_net, deck_ctx)
    results = []
    for label, snapshot_id, is_archive, snapshot_path in milestones:
        hist_agent = pool.load_snapshot_agent(snapshot_path, deck_ctx)
        result = _play_paired_eval_games(live_agent, hist_agent, decklist, games_per_snapshot, horizon,
                                         seed, "snapshot_wins")
        # snapshot_id/is_archive recorded since the two labels are not
        # comparable: archive_oldest is pinned to snapshot_0 forever (a fixed
        # reference), while active_oldest tracks a rolling older self.
        results.append({"label": label, "snapshot_id": snapshot_id, "is_archive": is_archive, **result})
    return results


def _run_eval_vs_gauntlet(deck_name, live_net, mulligan_net, deck_ctx, decklist, gauntlet_league_dir,
                           horizon, games=EVAL_GAMES, seed=None):
    """Plays deck_name's current live net against the same-named deck from an
    independently-trained population (gauntlet_league_dir) -- a genuinely
    external reference, not another point in this league's own self-play
    history. A shared population-wide blind spot is something an
    independently-evolved population is more likely to expose than any
    opponent drawn from this league's own history.

    Returns None (not []) if the gauntlet league has no live.pt for this
    deck yet, or gauntlet_league_dir is unset -- distinct from
    _run_eval_vs_history's [] since there's only ever one gauntlet opponent
    per deck. greedy=True."""
    if gauntlet_league_dir is None:
        return None
    # No stack-compatibility check needed: the gauntlet's checkpoint brings
    # its own perception encoder, so its weights are never reinterpreted
    # through this league's.
    gauntlet_live_path = f"{gauntlet_league_dir}/{deck_name}/live.pt"
    if not os.path.exists(gauntlet_live_path):
        return None

    vocab, fixed_table = deck_ctx
    # The gauntlet is a different population that may have been trained at a
    # different trunk width, so its own checkpoint is the only authority on
    # the shape to build.
    gauntlet_net = build_deck_net(vocab.size, len(fixed_table),
                                  ckpt_io.trunk_hidden_from_deck_checkpoint(gauntlet_live_path))
    ckpt_io.load_deck_checkpoint(gauntlet_live_path, gauntlet_net)  # existence already checked above
    gauntlet_net.eval()
    gauntlet_mull = MulliganNet(gauntlet_net.encoder)
    gauntlet_mull_path = f"{gauntlet_league_dir}/{deck_name}/mulligan.pt"
    ckpt_io.load_deck_checkpoint(gauntlet_mull_path, gauntlet_mull)
    gauntlet_mull.eval()

    live_agent = SeatAgent(live_net, mulligan_net, deck_ctx)
    gauntlet_agent = SeatAgent(gauntlet_net, gauntlet_mull, deck_ctx)
    return _play_paired_eval_games(live_agent, gauntlet_agent, decklist, games, horizon, seed, "gauntlet_wins")


def _run_eval_vs_heuristic(deck_name, live_net, mulligan_net, deck_ctx, decklist, horizon,
                            games=EVAL_GAMES, seed=None):
    """Plays deck_name's current live net against a HeuristicAgent
    (rl.decision.heuristic_agent) for the same deck -- the gauntlet's tier-1
    member, distinct from _run_eval_vs_gauntlet's tier-2 (an
    independently-trained population). Only called for whichever deck(s)
    the caller's heuristic_decks names -- the heuristic's rules are general
    MTG principles hand-picked for mono_red_rally, not audited for every
    deck."""
    live_agent = SeatAgent(live_net, mulligan_net, deck_ctx)
    heuristic_agent = HeuristicAgent(deck_ctx)
    return _play_paired_eval_games(live_agent, heuristic_agent, decklist, games, horizon, seed, "heuristic_wins")


def _json_default(obj):
    """Fallback for json.dump when a raw engine object slips into a logged
    event instead of a plain value. Keeps a real log from being lost to one
    bad field, and prints enough to trace the culprit. Converts to the
    object's own .name where recognizable, else repr()."""
    name = getattr(obj, "name", None)
    if isinstance(name, str):
        print(f"  [event log] non-serializable {type(obj).__name__} encountered, using its .name: {name!r}")
        return name
    print(f"  [event log] non-serializable {type(obj).__name__} encountered, using repr(): {obj!r}")
    return repr(obj)


def _write_event_log(log_path, game_logs, meta, game_pairings=None):
    """Writes the engine event logs collected this session to PATH as one
    compact JSON doc (no indent -- pretty-printing bloats it substantially).
    _json_default salvages any stray non-serializable field. game_pairings,
    if given, stamps each game's own deck_a/deck_b onto its record so a
    round-robin log with many pairings is self-describing (read by
    webapp/replay_engine.list_games)."""
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
