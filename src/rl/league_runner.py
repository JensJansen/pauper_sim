"""run_league.py's reusable core: session driving (_run_session), eval-mode
functions (_run_eval and the vs_history/vs_gauntlet/vs_heuristic checks),
checkpoint/progress helpers, and the shared/frozen-stack loaders. Extracted
so benchmarking/training_run.py can import this directly (`from rl import
league_runner`) instead of importing the 954-line run_league.py CLI script
for its side-effect-free functions -- run_league.py itself now just calls
into this module from its own main(). Pure reorganization: no behavior
changed by moving the code here.
"""
import json
import os
import random
import time

import torch

from rl.rewards import deploy_reward_v2
from rl.arch import SetTransformer
from rl.deck import DeckNetwork
from rl.league import LeaguePool
from rl.pool import build_pool
from rl.agent import HeuristicAgent, SeatAgent
from rl.train import batch_size_for_iteration, collect_rollout, collect_rollout_league, _constant_pairing, ent_coef_schedule
from rl.rollout_parallel import collect_rollout_league_parallel
from rl.ppo import ppo_update
from rl.mulligan import MulliganNet, update as mulligan_update
from rl import checkpoint as ckpt_io
from repo_paths import CHECKPOINTS_DIR

FROZEN_STACK = CHECKPOINTS_DIR / "shared_stack_frozen.pt"
LEAGUE_DIR = CHECKPOINTS_DIR / "league"
D_MODEL = 64
SHARED_HPARAMS = {"d_model": D_MODEL, "n_heads": 4, "n_layers": 2, "dim_feedforward": 128}


def load_frozen_stack(vocab_size):
    assert os.path.exists(FROZEN_STACK), (
        f"{FROZEN_STACK} not found -- run `python run_pretrain.py ... --freeze` (pretrain) first"
    )
    ckpt = ckpt_io.load_frozen_stack(FROZEN_STACK)
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
        ckpt_io.save_deck_checkpoint(f"{deck_dir}/live.pt", live_nets[name], optimizers[name])
        if mulligan_nets is not None:
            ckpt_io.save_deck_checkpoint(f"{deck_dir}/mulligan.pt", mulligan_nets[name], mulligan_optimizers[name])
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


def _append_metric(league_dir, **fields):
    """Appends one JSON line to checkpoints/<league>/metrics.jsonl -- every
    call site tags its own `kind` ("ppo" / "mulligan" / "vs_history") so
    report_metrics.py can group by it. This is the durable form of what used
    to only ever appear in stdout (policy_loss/value_loss/entropy were always
    COMPUTED per iteration -- entropy in particular was thrown away right
    after computing it, never even printed); nothing here changes training,
    only what's recorded about it. Does not create league_dir itself -- the
    caller's LeaguePool(league_dir, ...) construction already does, once,
    before the first call ever lands here."""
    with open(f"{league_dir}/metrics.jsonl", "a") as f:
        f.write(json.dumps(fields) + "\n")


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
                  matchup=None, game_logs=None, checkpoint_rate=0.0, roster=None, pfsp=True,
                  gauntlet_league_dir=None, heuristic_decks=(), cumulative_games=0):
    # cumulative_games: this league's games/deck ALREADY played before this
    # session started (progress.json, threaded in by run_league.py's main())
    # -- the horizon batch_size_for_iteration and ent_coef_schedule (rl.train)
    # ramp against, recomputed every iteration as cumulative_games + iteration
    # * games_per_iteration so the ramp both grows smoothly WITHIN a big
    # session and picks up from the right point when a NEW session starts,
    # rather than resetting to each schedule's start value every single
    # process invocation the way both used to (see their own docstrings).
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
        optimizer = torch.optim.Adam([p for p in net.parameters() if p.requires_grad], lr=3e-4)
        live_path = f"{league_dir}/{name}/live.pt"
        ckpt_io.load_deck_checkpoint(live_path, net, optimizer)  # no-op if live_path doesn't exist yet; optimizer load is migration-guarded (a migrated live.pt dropping "optimizer" -> fresh Adam)
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
        mopt = torch.optim.Adam([p for p in mnet.parameters() if p.requires_grad], lr=1e-3)
        mull_path = f"{league_dir}/{name}/mulligan.pt"
        ckpt_io.load_deck_checkpoint(mull_path, mnet, mopt)  # same migration-guarded optimizer load as the live-net path above
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
    # Mulligan transitions accumulate ACROSS iterations now (2026-08-06),
    # flushed to a real REINFORCE update only every MULLIGAN_UPDATE_EVERY
    # iterations (see the update block below) instead of every single
    # iteration on whatever that one iteration happened to generate (median
    # 11 transitions/update, measured on a real run -- both the policy
    # gradient AND its learned baseline were being fit from the same tiny,
    # correlated batch, a genuinely high-variance regime given reward is
    # dominated by a 0/1 win outcome). See TRAINING_IMPROVEMENT_OPTIONS.md
    # section 4.
    MULLIGAN_UPDATE_EVERY = 8
    mull_by_deck_accum = {name: [] for name in train_decks}
    for iteration in range(n_iterations):
        # PPO minibatch ramp (32 -> 2048 over 6 steps) and entropy-coefficient
        # anneal: both tracked against TRUE cumulative games/deck (this
        # session's own starting point, cumulative_games, plus games played so
        # far THIS session) -- see batch_size_for_iteration's and
        # ent_coef_schedule's own docstrings (rl.train) for why session-local
        # tracking was replaced. games_per_iteration is the per-deck games
        # count each iteration adds, regardless of how many decks train.
        games_so_far_this_session = iteration * games_per_iteration
        batch_size = batch_size_for_iteration(cumulative_games + games_so_far_this_session)
        ent_coef = ent_coef_schedule(cumulative_games + games_so_far_this_session)
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
                buffers_by_deck, mull_by_deck, played, outcomes = collect_rollout_league_parallel(
                    name, live_nets, reward_fn_name, league_dir, horizon, games_per_iteration,
                    executor, n_workers, SHARED_HPARAMS, shared_state_dict, all_trunk_hidden,
                    mulligan_state_dicts, game_logs=game_logs, checkpoint_rate=checkpoint_rate, pfsp=pfsp,
                )
            else:
                buffers_by_deck, mull_by_deck, played, outcomes = collect_rollout_league(
                    name, live_nets, mulligan_nets, deck_ctxs, decklists, pool, reward_fn,
                    horizon, games_per_iteration, rng, device="cpu", game_logs=game_logs,
                    checkpoint_rate=checkpoint_rate, pfsp=pfsp,
                )
            if matchup is None:
                # Feed every game's real outcome into the ONE authoritative pool
                # (sequential or parallel -- a worker's own pool is read-only, see
                # collect_rollout_league_parallel's own docstring) so the NEXT
                # sample_opponent call already reflects it -- PFSP weighting updates
                # within the same session, not just across separate process
                # invocations (save_opponent_stats, below, is what carries it across
                # those).
                for opp_name, snap_id, won in outcomes:
                    pool.record_outcome(name, ("deck", opp_name), won)
                    if snap_id is not None:
                        pool.record_outcome(name, ("snapshot", opp_name, snap_id), won)
            collect_time_total += time.time() - t_collect0
            total_games += played
            # Accumulate mulligan transitions for each TRAIN deck that generated
            # some this round (training deck + any live-opponent salvage); a frozen
            # opponent's are discarded (it isn't being trained).
            for deck_name, tr in mull_by_deck.items():
                if deck_name in train_set:
                    mull_by_deck_accum[deck_name].extend(tr)
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
            policy_loss = value_loss = entropy = approx_kl = clip_fraction = 0.0
            epochs_run = 0
            salvaged = 0
            if train_deck:
                for deck_name, buf in buffers_by_deck.items():
                    if not len(buf) or deck_name not in train_set:
                        continue
                    if deck_name == name:
                        policy_loss, value_loss, entropy, approx_kl, clip_fraction, epochs_run = ppo_update(
                            live_nets[name], [optimizers[name]], buf, "cpu", batch_size=batch_size, ent_coef=ent_coef)
                    else:
                        ppo_update(live_nets[deck_name], [optimizers[deck_name]], buf, "cpu",
                                   batch_size=batch_size, ent_coef=ent_coef)
                        salvaged += len(buf)
            update_time_total += time.time() - t_update0
            buffer_size = len(buffers_by_deck.get(name, ()))
            print(f"  iter {iteration} [{name}]: games={played} buf={buffer_size} "
                  f"salvaged={salvaged} batch_size={batch_size} ent_coef={ent_coef:.4f} "
                  f"policy_loss={policy_loss:.4f} value_loss={value_loss:.4f} "
                  f"approx_kl={approx_kl:.4f} clip_frac={clip_fraction:.3f} epochs_run={epochs_run}", flush=True)
            # entropy: computed by ppo_update every call, previously discarded
            # right after this print -- see _append_metric's own docstring.
            # buffer_size alongside batch_size is what actually answers "has
            # the minibatch ramp saturated past the real amount of collected
            # data" (once batch_size >= buffer_size, ppo_update's inner loop
            # stops sub-batching and just runs n_epochs full-batch steps).
            # approx_kl/clip_fraction/epochs_run (2026-08-06): visibility into
            # WHY entropy moves the way it does, not just that it did -- see
            # ppo_update's own docstring and TRAINING_IMPROVEMENT_OPTIONS.md
            # section 4 for the entropy-collapse investigation this exists to
            # let the owner actually check going forward, instead of guessing.
            _append_metric(league_dir, kind="ppo", session=session, iteration=iteration, deck=name,
                           games=played, buffer_size=buffer_size, batch_size=batch_size, salvaged=salvaged,
                           ent_coef=ent_coef, policy_loss=policy_loss, value_loss=value_loss, entropy=entropy,
                           approx_kl=approx_kl, clip_fraction=clip_fraction, epochs_run=epochs_run)

        # Mulligan-model REINFORCE: one step per TRAIN deck on transitions
        # accumulated across the last MULLIGAN_UPDATE_EVERY iterations (see
        # this loop's own top-of-function comment for why), flushed early on
        # the session's FINAL iteration regardless of the cadence so no
        # transitions are silently dropped at a session boundary (this
        # accumulator is a local variable -- nothing persists it across
        # separate `run_league.py` process invocations). Gated on
        # train_mulligan; decoupled from the main PPO updates above -- its
        # own optimizer, its own reward.
        mull_stats = {}
        is_mulligan_flush = (iteration + 1) % MULLIGAN_UPDATE_EVERY == 0 or iteration == n_iterations - 1
        if train_mulligan and is_mulligan_flush:
            for name in train_decks:
                if mull_by_deck_accum[name]:
                    mull_stats[name] = mulligan_update(mulligan_nets[name], mulligan_optimizers[name], mull_by_deck_accum[name])
                    _append_metric(league_dir, kind="mulligan", session=session, iteration=iteration, deck=name,
                                   n=mull_stats[name]["n"], loss=mull_stats[name]["loss"])
                    mull_by_deck_accum[name] = []
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
            # Same cadence as the checkpoints above (not every iteration -- see
            # LeaguePool.save_opponent_stats' own docstring for why this specific
            # cadence, not "every game" or "never until session end").
            pool.save_opponent_stats()
            print(f"  iter {iteration}: snapshotted {len(train_decks)} train deck(s) + saved live checkpoints "
                  f"(counts now: { {n: len(pool.snapshots[n]) for n in deck_names} })", flush=True)

    elapsed = time.time() - t0
    print(f"session {session} done in {elapsed:.1f}s ({elapsed / total_games:.2f}s/game across {total_games} games) -- "
          f"collect={collect_time_total:.1f}s ({100 * collect_time_total / elapsed:.0f}%), "
          f"update={update_time_total:.1f}s ({100 * update_time_total / elapsed:.0f}%)")

    # Once per session (not per snapshot -- see _run_eval_vs_history's own
    # docstring): does the live net actually beat its own past selves. Only
    # meaningful in league mode (matchup mode never snapshots, so there's
    # never any history to compare against) -- naturally a no-op (empty
    # milestones) until a deck has been through at least one snapshot cycle.
    # NOT free once that happens: every _run_eval_vs_* call below runs its
    # games sequentially in THIS process via plain collect_rollout (not
    # collect_rollout_league_parallel), so it does not use the session's own
    # executor/n_workers -- up to (2 vs_history milestones + vs_gauntlet +
    # vs_heuristic) x games-per-check extra games per train deck, tacked onto
    # the end of the session regardless of how small that session's own
    # training batch was (confirmed empirically: a 16-game training session
    # in this repo's own checkpoints/4_deck_subleague_test/metrics.jsonl paid
    # 80 additional vs_history eval games alone). See _play_eval_games.
    if matchup is None:
        for name in train_decks:
            for r in _run_eval_vs_history(name, live_nets[name], mulligan_nets[name], deck_ctxs[name],
                                          decklists[name], shared, league_dir, horizon, seed=seed):
                _append_metric(league_dir, kind="vs_history", session=session, iteration=iteration, deck=name, **r)
                print(f"  vs-history [{name}] vs {r['label']}: {r['live_wins']}/{r['games']} live wins "
                      f"({r['snapshot_wins']} snapshot wins, {r['no_winner']} no-winner)", flush=True)
            # The gauntlet check (an INDEPENDENTLY-trained twin population, see
            # _run_eval_vs_gauntlet's own docstring) -- None until that population's
            # training has reached this deck, or no gauntlet_league_dir is configured
            # for this league at all (most leagues won't have one).
            r = _run_eval_vs_gauntlet(name, live_nets[name], mulligan_nets[name], deck_ctxs[name],
                                      decklists[name], shared, gauntlet_league_dir, horizon, seed=seed)
            if r is not None:
                _append_metric(league_dir, kind="vs_gauntlet", session=session, iteration=iteration, deck=name, **r)
                print(f"  vs-gauntlet [{name}]: {r['live_wins']}/{r['games']} live wins "
                      f"({r['gauntlet_wins']} gauntlet wins, {r['no_winner']} no-winner)", flush=True)
            # The tier-1 gauntlet member (a hand-authored HeuristicAgent) --
            # only for whichever deck(s) heuristic_decks names (most leagues
            # name none; see _run_eval_vs_heuristic's own docstring).
            if name in heuristic_decks:
                r = _run_eval_vs_heuristic(name, live_nets[name], mulligan_nets[name], deck_ctxs[name],
                                           decklists[name], horizon, seed=seed)
                _append_metric(league_dir, kind="vs_heuristic", session=session, iteration=iteration, deck=name, **r)
                print(f"  vs-heuristic [{name}]: {r['live_wins']}/{r['games']} live wins "
                      f"({r['heuristic_wins']} heuristic wins, {r['no_winner']} no-winner)", flush=True)

    _save_live_checkpoints(live_nets, optimizers, deck_names, session, session_path, league_dir,
                           mulligan_nets, mulligan_optimizers)
    if matchup is None:
        pool.save_opponent_stats()
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
        ckpt_io.load_deck_checkpoint(live_path, net)  # optimizer=None: eval only ever needs inference weights
        net.eval()
        live_nets[name] = net
        mnet = MulliganNet(shared)
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


def _list_snapshot_ids(dir_path):
    """Sorted snapshot ids (the N in snapshot_N.pt) found directly in
    dir_path, or [] if the directory doesn't exist yet -- shared by the
    active pool dir and its archive/ subdir, which have the same naming."""
    if not os.path.isdir(dir_path):
        return []
    return sorted(int(fn[len("snapshot_"):-len(".pt")]) for fn in os.listdir(dir_path)
                  if fn.startswith("snapshot_") and fn.endswith(".pt"))


def _play_eval_games(live_agent, opp_agent, decklist, n_games, horizon, rng, opp_wins_key):
    """Shared tail for every vs_history/vs_gauntlet/vs_heuristic eval pass:
    plays a fixed live_agent-vs-opp_agent pairing greedily (no training,
    matching _run_eval's own eval convention), scores winners via
    collect_rollout's "game_over" event, and returns the standard
    {"games", "live_wins", <opp_wins_key>, "no_winner"} tally shape every
    caller already builds by hand. opp_wins_key names the opponent side in
    the returned dict (e.g. "snapshot_wins"/"gauntlet_wins"/"heuristic_wins")
    since each caller's opponent is a different kind of thing."""
    pairing = _constant_pairing([live_agent, opp_agent], [decklist, decklist], [None, None], [None, None])
    game_logs = []
    _bufs, _mull, played = collect_rollout(pairing, n_games, horizon, rng, device="cpu",
                                           record=False, greedy=True, game_logs=game_logs)
    outcomes = [e for ev in game_logs for e in ev if e["kind"] == "game_over"]
    live_wins = sum(1 for e in outcomes if e["winner"] == 0)
    opp_wins = sum(1 for e in outcomes if e["winner"] == 1)
    return {"games": played, "live_wins": live_wins, opp_wins_key: opp_wins,
            "no_winner": played - live_wins - opp_wins}


def _run_eval_vs_history(deck_name, live_net, mulligan_net, deck_ctx, decklist, shared, league_dir,
                          horizon, games_per_snapshot=20, seed=None):
    """Plays deck_name's CURRENT live net against its own past selves: the
    oldest snapshot still in the active LeaguePool sampling window, plus (once
    LeaguePool.register_snapshot has evicted anything) the oldest ARCHIVED
    snapshot -- the deepest history available at all. This is the direct
    answer to "has this policy actually improved," as opposed to eyeballing
    stable loss curves: a live net that can't beat its own 20,000-games-ago
    self is a much stronger cycling signal than static win rates against
    whatever the league happens to be sampling right now.

    Returns [] when there's no history yet at all (early in a run, before the
    first snapshot exists) -- nothing to compare against, not an error. Reuses
    the caller's already-loaded live_net/shared/deck_ctx/decklist (no disk
    round-trip for the live side); only the historical snapshot is loaded from
    disk, via LeaguePool.load_snapshot_agent (the same loader league training
    itself uses for checkpoint opponents). Relies on collect_rollout's own
    "game_over" event (see rl.train) to score winners -- greedy=True so this
    measures the policy's actual best play, matching _run_eval's own eval
    convention, not an exploration sample."""
    deck_dir = f"{league_dir}/{deck_name}"
    milestones = []
    archive_ids = _list_snapshot_ids(os.path.join(deck_dir, "archive"))
    if archive_ids:
        milestones.append(("archive_oldest", os.path.join(deck_dir, "archive", f"snapshot_{archive_ids[0]}.pt")))
    active_ids = _list_snapshot_ids(deck_dir)
    if active_ids:
        milestones.append(("active_oldest", os.path.join(deck_dir, f"snapshot_{active_ids[0]}.pt")))
    if not milestones:
        return []

    pool = LeaguePool(league_dir, [deck_name])  # only its load_snapshot_agent loader is used here
    live_agent = SeatAgent(live_net, mulligan_net, deck_ctx)
    rng = random.Random(seed)
    results = []
    for label, snapshot_path in milestones:
        hist_agent = pool.load_snapshot_agent(snapshot_path, shared, deck_ctx)
        result = _play_eval_games(live_agent, hist_agent, decklist, games_per_snapshot, horizon, rng, "snapshot_wins")
        results.append({"label": label, **result})
    return results


def _run_eval_vs_gauntlet(deck_name, live_net, mulligan_net, deck_ctx, decklist, shared, gauntlet_league_dir,
                           horizon, games=20, seed=None):
    """Plays deck_name's CURRENT live net against the SAME-NAMED deck from an
    INDEPENDENTLY-trained population (gauntlet_league_dir -- see
    training_configs/run_gauntlet.json's own _note) -- a genuinely EXTERNAL
    reference, not another point in this league's own self-play history the
    way _run_eval_vs_history's snapshots are. Two runs trained via the
    identical algorithm/settings from the SAME frozen shared stack still
    diverge into different regions of strategy space from nothing but a
    different nondeterministic training trajectory -- so a shared,
    population-wide blind spot (this WHOLE mini-league co-adapting into a
    closed, mutually-exploitable bubble, the failure mode this whole
    mechanism exists to catch) is something an independently-evolved
    population is far more likely to expose than any opponent drawn from
    this league's OWN history ever could, no matter how far back.

    Returns None (not []) if the gauntlet league has no live.pt for this deck
    yet (its training hasn't reached this deck, or gauntlet_league_dir is
    unset) -- distinct from _run_eval_vs_history's [] since there's only ever
    ONE gauntlet opponent per deck, not a list of milestones. greedy=True,
    same eval convention as _run_eval_vs_history: the policy's actual best
    play, not an exploration sample."""
    if gauntlet_league_dir is None:
        return None
    gauntlet_live_path = f"{gauntlet_league_dir}/{deck_name}/live.pt"
    if not os.path.exists(gauntlet_live_path):
        return None

    _vocab, fixed_table = deck_ctx
    gauntlet_net = DeckNetwork(shared, film_condition_dim=D_MODEL, non_targeting_n_actions=len(fixed_table))
    ckpt_io.load_deck_checkpoint(gauntlet_live_path, gauntlet_net)  # existence already checked above
    gauntlet_net.eval()
    gauntlet_mull = MulliganNet(shared)
    gauntlet_mull_path = f"{gauntlet_league_dir}/{deck_name}/mulligan.pt"
    ckpt_io.load_deck_checkpoint(gauntlet_mull_path, gauntlet_mull)
    gauntlet_mull.eval()

    live_agent = SeatAgent(live_net, mulligan_net, deck_ctx)
    gauntlet_agent = SeatAgent(gauntlet_net, gauntlet_mull, deck_ctx)
    rng = random.Random(seed)
    return _play_eval_games(live_agent, gauntlet_agent, decklist, games, horizon, rng, "gauntlet_wins")


def _run_eval_vs_heuristic(deck_name, live_net, mulligan_net, deck_ctx, decklist, horizon, games=20, seed=None):
    """Plays deck_name's CURRENT live net against a HeuristicAgent (rl.agent)
    for the SAME deck -- the gauntlet's tier-1 member: a hand-authored, non-
    self-play reference opponent, distinct from _run_eval_vs_gauntlet's tier-2
    (an independently-TRAINED population). Only called for whichever deck(s)
    _run_session's own caller configured (see heuristic_decks below) -- the
    heuristic's rules are general MTG principles the owner hand-picked for
    ONE deck (mono_red_rally), not audited or intended for every deck in a
    roster."""
    live_agent = SeatAgent(live_net, mulligan_net, deck_ctx)
    heuristic_agent = HeuristicAgent(deck_ctx)
    rng = random.Random(seed)
    return _play_eval_games(live_agent, heuristic_agent, decklist, games, horizon, rng, "heuristic_wins")


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
