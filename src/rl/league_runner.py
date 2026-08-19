"""run_league.py's reusable core: session driving (_run_session), eval-mode
functions (_run_eval and the vs_history/vs_gauntlet/vs_heuristic checks),
checkpoint/progress helpers, and the per-deck network builder. Extracted
so benchmarking/training_run.py can import this directly (`from rl import
league_runner`) instead of importing the 954-line run_league.py CLI script
for its side-effect-free functions -- run_league.py itself now just calls
into this module from its own main(). Pure reorganization: no behavior
changed by moving the code here.
"""
import copy
import json
import os
import random
import time

import torch

from rl.rewards import deploy_reward_v6
from rl.arch import SetTransformer
from rl.deck import DeckNetwork
from rl.league import LeaguePool, PFSP_POWER
from rl.pool import build_pool
from rl.agent import SeatAgent
from rl.heuristic_agent import HeuristicAgent
from rl.train import batch_size_for_iteration, collect_rollout, collect_rollout_league, _constant_pairing, ent_coef_schedule
from rl.rollout_parallel import collect_rollout_league_parallel
from rl.ppo import ppo_update
from rl.mulligan import MulliganNet, update as mulligan_update
from rl import checkpoint as ckpt_io
from repo_paths import CHECKPOINTS_DIR

LEAGUE_DIR = CHECKPOINTS_DIR / "league"
D_MODEL = 64  # SetTransformer width; must match rl.arch.SetTransformer's own d_model default
# Turn limit for a single game, for TRAINING and EVERY eval alike. One constant
# because a training/eval mismatch here is silent: games would simply end at a
# different point than the policy was trained for, and every win rate would
# measure something other than what it claimed. It was previously written out
# as a literal `120` in several places (both _run_session and _run_eval here,
# plus run_cross_league_eval, run_anchor_eval, run_snapshot_round_robin), so
# changing it would have silently desynchronized them.
HORIZON = 120

# Games per in-training eval check (vs_history / vs_gauntlet / vs_heuristic).
# Was a hardcoded 20 at each of the three call sites, giving SE 11.2pp and a
# 95% CI of +/-21.9pp -- wide enough that no realistic single-session change
# could ever clear it, while still costing ~19% of total compute across the
# thirteen such readings taken per session. Nothing was learnable from any
# individual one; the trend across many was the only usable signal, and nothing
# computed that trend until report_metrics was rewritten.
#
# Raising this trades reading FREQUENCY for PRECISION at fixed cost:
# eval_games=200 with eval_every_sessions=4 costs the same compute and drops
# per-reading SE to 3.5pp. Config-driven (run-config "eval_games" /
# "eval_every_sessions") because the right point on that trade depends on
# session size.
EVAL_GAMES = 20
EVAL_EVERY_SESSIONS = 1

# DeckNetwork trunk widths, per deck. Each deck's SetTransformer encoder is a
# further 117,056 params on top of this (and trains with it, 2026-08-17 -- it
# used to be a shared, pretrained, permanently-frozen 59% of the model).
# Made per-league configurable 2026-08-15 to test whether the late-training
# plateau (reached by ~4,000 games/deck
# and NOT caused by KL truncation, entropy, advantage scale or Adam-step count)
# is a capacity ceiling. (512, 512) is 515,261 params, 6.2x this.
#
# Changing it invalidates existing live.pt/snapshot shapes, so it only applies
# to a FRESH league: _run_session reads the width back off an existing live.pt
# when resuming and uses this only for a deck that has none yet.
TRUNK_HIDDEN = (128, 128)

# Optimizer/PPO knobs, previously hardcoded at their use sites and therefore
# unchangeable without a code edit -- `lr` sat as a literal 3e-4 inside the Adam
# construction, and gamma/gae_lambda/target_kl/n_epochs were only ever
# ppo_update's own signature defaults, so no config file could reach them and
# nothing recorded which values a given run used.
#
# These defaults were EXACTLY the values every run used up to 2026-08-15 (see
# `lr` below for the one that has since changed on measured evidence). Exposing
# them made changing them a config edit instead of a code edit, and made the
# value actually in force appear in metrics.jsonl's session_start record.
# Override per league via a run-config "ppo" object.
PPO_DEFAULTS = {
    # league_runner's per-deck Adam. 3e-4 -> 2e-4 (2026-08-15), the first
    # default in this file changed on a controlled experiment rather than a
    # guess -- three single-variable arms at 10,000 games/deck each, scored
    # against the IDENTICAL reference (the 20,016-game baseline's live nets):
    #
    #     lr        kl med  epochs  truncation   vs baseline
    #     3e-4      0.0274   2.67       99%        (is the baseline)
    #     2e-4      0.0243   3.94        6%        50/80 = 62.5%
    #     1.5e-4    0.0152   4.00        0%        36/80 = 45.0%
    #
    # The curve is sharply nonlinear -- 2e-4's KL is only 11% under 3e-4's, but
    # that is enough to drop back under the target_kl=0.03 epoch-mean cliff,
    # collapsing truncation 99% -> 6%.
    #
    # CORRECTED (section 1A.13): the 62.5% above is 2e-4 measured at cum 9,984,
    # which extending it to a budget-matched 20,016 showed was the HIGHEST of
    # eight samples on an oscillating series (47.5 / 55.0 / 62.5 / 47.5 / 51.2 /
    # 53.8 / 43.8 from cum 3,984 on). At matched budget it is 43.8%, i.e. parity
    # (z=-1.12). The "better final policy" claim is WITHDRAWN.
    #
    # 2e-4 is kept as the default on what survived: it reaches parity with the
    # baseline's FINAL policy at cum ~3,984 (z=-0.45 from parity), a ~5x
    # sample-efficiency gain, it eliminates truncation, and it is still the only
    # value tested at which no deck materially regresses (3e-4 costs dmir_terror
    # -80 elo, 1.5e-4 costs rakdos_madness -71). It converges much faster to the
    # same place; it does not end up stronger.
    #
    # Note for anyone tuning further: 2e-4 plateaus and oscillates exactly as
    # 3e-4 does despite having no truncation at all, so the late-training
    # plateau is NOT a trust-region problem and lowering lr further will not fix
    # it.
    #
    # Scope: LEAGUE training only. The pretrain phase kept its own untested
    # 3e-4 and was deleted with run_pretrain.py (2026-08-17), so this is now
    # the only lr in the system.
    #
    # Resuming a league trained at 3e-4 (checkpoints/4_deck_subleague_test)
    # now picks this up and silently changes its optimizer mid-run. Pin
    # "ppo": {"lr": 0.0003} in that league's config if it is ever continued.
    "lr": 2e-4,
    # Minibatch ramp, in EPISODES per minibatch -- NOT transitions.
    #
    # The recurrent policy (2026-08-17) made a transition-level minibatch
    # meaningless: ppo_update replays whole episodes through the GRU in order,
    # so its unit of sampling is a trajectory, not a step. These keys were
    # renamed from batch_size_start/batch_size_cap precisely so a config
    # carrying the old numbers fails loudly on PPO_DEFAULTS' unknown-key assert
    # instead of being read as 32 EPISODES -- which, at ~25-50 episodes per
    # buffer, would silently collapse every update to one full-batch minibatch.
    #
    # 4 -> 16 with ~100-step episodes is roughly 400 -> 1600 transitions per
    # minibatch, against the 32 transitions the feedforward policy used. That
    # is a large change in gradient granularity and it perturbs values tuned
    # against the old one (target_kl's early stopping, games_per_iteration=24).
    # Expect the KL/truncation diagnostics to move on the first recurrent run.
    #
    # The ramp itself (small early, large late) is unchanged in spirit and
    # still PINNED OFF-ish: start 4, cap 16 does fire, unlike the old pinned
    # 32/32. BUG 1 (cumulative_games stuck at 0) is fixed, so the ramp is live.
    "seq_batch_size_start": 4,
    "seq_batch_size_cap": 16,
    "mulligan_lr": 1e-3,   # the mulligan net's own, deliberately separate optimizer
    "gamma": 0.99,
    "gae_lambda": 0.95,    # with gamma, a 0.9405/step advantage decay
    "target_kl": 0.03,
    "n_epochs": 4,
    # Floor on the advantage-normalization divisor (rl.ppo.ppo_update). 0.0 is
    # exactly the historical behavior; see that function for why a nonzero value
    # is wanted and why it is not being guessed. Set it from the adv_std
    # distribution `report_metrics` now prints, after a session of real data.
    "adv_norm_floor": 0.0,
    # Entropy bonus. None = use rl.train.ent_coef_schedule's 0.02 -> 0.005
    # anneal (the historical behavior, unchanged). A float pins it CONSTANT for
    # the whole run, which is Wave 2b.
    #
    # Why the knob exists: on the 20,016-games/deck run the anneal was measured
    # running the wrong way. ent_coef fell 0.0200 -> 0.0140 while policy entropy
    # fell 0.985 -> 0.450; a more deterministic policy moves probability ratios
    # further per unit parameter change, so median approx_kl doubled
    # (0.0099 -> 0.0274) until the epoch-mean crossed target_kl and KL early
    # stopping truncated nearly every update -- mean epochs_run 4.00 (session 5)
    # -> 2.24 (session 12), BELOW the 2.8 that motivated the restart in the
    # first place. The anneal cuts the entropy bonus exactly when low entropy is
    # what is breaking the updates.
    "ent_coef": None,
}


def build_deck_net(vocab_size, n_actions, trunk_hidden=None):
    """One deck's full network: its OWN perception encoder (rl.arch) plus the
    trunk/critic/pointer heads built over it. Every construction site goes
    through here so no caller has to remember that the encoder is built from
    the pool vocab size and that film_condition_dim tracks its d_model.

    There is deliberately no shared-encoder loader beside this any more. The
    league used to pretrain ONE SetTransformer, freeze it to
    checkpoints/shared_stack_frozen.pt, and build every deck on that single
    instance -- which also required stack_id.txt bookkeeping, because loading
    one population's per-deck weights onto ANOTHER population's stack
    silently produces a meaningless number (that really happened: every
    vs_gauntlet reading across sessions 0-14, 24,579 games/deck, was
    confounded before anyone noticed). A checkpoint now carries its own
    encoder, so cross-population comparison is valid by construction and
    there is nothing left to guard."""
    encoder = SetTransformer(vocab_size)
    return DeckNetwork(encoder, film_condition_dim=encoder.d_model, non_targeting_n_actions=n_actions,
                       trunk_hidden=trunk_hidden or TRUNK_HIDDEN)


def _save_live_checkpoints(live_nets, optimizers, deck_names, session, session_path, league_dir,
                           mulligan_nets=None, mulligan_optimizers=None):
    """Persist every deck's current live net + optimizer + the session
    counter. Called at each snapshot point AND at session end -- NOT only
    at the end: a mid-session crash (a rare card-interaction bug ~2500
    games into a 3000-game batch is exactly how this was learned) would
    otherwise discard the whole session's live-net training, since only the
    periodic snapshots (historical opponents) get written incrementally.
    With this, a crash loses at most snapshot_every_games games/deck (the
    snapshot cadence is cumulative-games-based, not session-local -- see the
    `crossed` gate in _run_session)."""
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


def should_snapshot(games_before, games_per_iteration, snapshot_every):
    """True when the iteration about to be trained -- which takes this league
    from `games_before` to `games_before + games_per_iteration` games/deck --
    crosses a multiple of the snapshot cadence.

    `games_before` is CUMULATIVE across every session this league has ever run
    (progress.json), not session-local. That is the whole point: `iteration`
    restarts at 0 every process invocation, so the old
    `(iteration + 1) % snapshot_every == 0` test never fired for a session
    shorter than snapshot_every and threw away the remainder at every session
    boundary. The escalation ladder runs exactly those short sessions early on
    (1, 2, 4 iterations against snapshot_every=8), so a fresh league reached
    168 games/deck with an empty opponent pool -- which also makes
    checkpoint_opponent_rate a silent no-op, since there is nothing to sample.
    Same class of bug as BUG 1 (see advance_progress) and as the ent_coef /
    batch_size ramps before it.

    snapshot_every is in ITERATIONS (run_league.py converts run-config's
    snapshot_every_games by floor division); multiplying back recovers the
    games-count, so a raw --snapshot-every override still behaves identically.
    Extracted so the multi-session invariant is testable without spawning
    training."""
    snapshot_every_games = snapshot_every * games_per_iteration
    return (games_before + games_per_iteration) // snapshot_every_games > games_before // snapshot_every_games


def checkpoint_progress(league_dir, cumulative_games_per_deck):
    """Mid-session write of the cumulative counter, at the snapshot cadence.

    BUG 3 (fixed 2026-08-17). _save_live_checkpoints has always written the
    live nets incrementally at every snapshot point, precisely so a mid-session
    crash keeps the training it already did. progress.json did NOT: it was
    written once, by run_league.main(), only after _run_session RETURNED. So a
    crash left the two out of step -- weights from N games/deck on disk beside
    a counter still reading the session's starting value. The counter is the
    horizon batch_size_for_iteration, ent_coef_schedule and should_snapshot all
    ramp against, so the resumed league re-ran a stretch of schedule it had
    already trained through. It had to be hand-corrected three times before
    being fixed here.

    Writes the counter ABSOLUTELY (caller passes games-so-far, not a delta) so
    calling it repeatedly within one session is idempotent. Leaves
    last_batch_size alone: that is the doubling ladder's feedback and belongs
    to the session as a whole, so only advance_progress touches it."""
    _save_progress(league_dir, _load_progress(league_dir)["last_batch_size"], cumulative_games_per_deck)


def advance_progress(league_dir, n_iterations, games_per_iteration, auto_sizing,
                     session_start_games=None):
    """Record what a finished batch did. Returns the new progress dict.

    Two separate things live in progress.json and only ONE of them belongs to
    auto-sizing:

      last_batch_size            the doubling ladder's feedback. STAYS gated on
                                 auto_sizing -- --n-iterations' documented
                                 contract is that a forced size never perturbs
                                 the ladder, so a one-off 3000-game debug run
                                 cannot become the next auto-sized batch's base.
      cumulative_games_per_deck  how much this league has EVER trained. Nothing
                                 to do with the ladder. It is the horizon that
                                 batch_size_for_iteration and ent_coef_schedule
                                 ramp against, so it must advance ALWAYS.

    Gating both on auto_sizing was BUG 1: the /train skill and the webapp
    escalation loop both always pass --n-iterations, so progress.json was never
    written, _load_progress returned 0 forever, and both schedules restarted at
    their origin every session for the entire run. Extracted from
    run_league.main() so the invariant is testable without spawning training.

    session_start_games is BUG 3's half of the fix. checkpoint_progress now
    advances the same counter mid-session, so a session-end "add played to
    whatever is on disk" would double-count every snapshot it already wrote.
    Given the session's own starting count, this computes the new value
    ABSOLUTELY (start + played) and is therefore idempotent no matter how many
    mid-session writes landed. Omitting it keeps the old additive behavior for
    callers that never checkpoint mid-session."""
    progress = _load_progress(league_dir)
    played = n_iterations * games_per_iteration
    last_batch_size = played if auto_sizing else progress["last_batch_size"]
    base = progress["cumulative_games_per_deck"] if session_start_games is None else session_start_games
    cumulative = base + played
    _save_progress(league_dir, last_batch_size, cumulative)
    return {"last_batch_size": last_batch_size, "cumulative_games_per_deck": cumulative}


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
                  league_dir=None, seed=None,
                  train_deck=True, train_mulligan=True, train_decks=None,
                  matchup=None, game_logs=None, checkpoint_rate=0.0, roster=None, pfsp=True,
                  gauntlet_league_dir=None, heuristic_decks=(), cumulative_games=0,
                  ppo_hparams=None, eval_games=EVAL_GAMES, eval_every_sessions=EVAL_EVERY_SESSIONS,
                  pfsp_power=PFSP_POWER, trunk_hidden=TRUNK_HIDDEN, device="cpu"):
    # device: where the NETS live, and therefore where ppo_update's gradient
    # work runs. "cpu" (the default) is byte-for-byte the previous behavior.
    #
    # Only the UPDATE moves. Collection always runs on CPU -- it is inference
    # over one game state at a time in six worker processes, which a GPU cannot
    # help with -- so the workers keep receiving CPU state_dicts, and every
    # eval path keeps running on CPU nets. Measured on this repo's own
    # 22,848-games/deck checkpoint (analysis/bench_gpu_vs_cpu.py): updates are
    # 86% of session wall time, and CUDA runs them 1.6-2.25x faster with
    # epochs_run identical on both arms, the speedup growing with buffer size.
    # The device->host state_dict copy this forces on every collect call costs
    # ~10ms against ~0.8ms on CPU, i.e. ~5s across a whole session -- measured,
    # not assumed, because it is the cost that would have made this a wash.
    # cumulative_games: this league's games/deck ALREADY played before this
    # session started (progress.json, threaded in by run_league.py's main())
    # -- the horizon batch_size_for_iteration and ent_coef_schedule (rl.train)
    # ramp against, recomputed every iteration as cumulative_games + iteration
    # * games_per_iteration so the ramp both grows smoothly WITHIN a big
    # session and picks up from the right point when a NEW session starts,
    # rather than resetting to each schedule's start value every single
    # process invocation the way both used to (see their own docstrings).
    # league_dir + seed let the benchmark harness drive this EXACT loop over a
    # throwaway dir, reproducibly -- the only intended differences from a real
    # training session. Defaults preserve the real run: checkpoint into
    # LEAGUE_DIR, nondeterministic rng. (There used to be a fresh_stack flag
    # too, for driving the loop without a trained shared_stack_frozen.pt on
    # disk; every deck now starts from a freshly-initialized encoder anyway
    # unless its own live.pt supplies one, so there is nothing to toggle.)
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
    # subset. vocab/deck_ctxs/fixed_tables still come from the FULL
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
    # ppo_hparams: a partial override of PPO_DEFAULTS (run-config "ppo" object).
    # Unknown keys are a hard error rather than a silent no-op -- a typo'd
    # hyperparameter that quietly does nothing is exactly the failure mode that
    # left two anti-plateau schedules un-executed for 40,104 iterations.
    hp = {**PPO_DEFAULTS, **(ppo_hparams or {})}
    unknown = set(hp) - set(PPO_DEFAULTS)
    assert not unknown, f"unknown ppo hyperparameter(s) {sorted(unknown)}; known: {sorted(PPO_DEFAULTS)}"
    live_nets, optimizers = {}, {}
    for name in deck_names:
        # Trunk width: whatever this deck's existing live.pt was built with when
        # RESUMING, the configured width only for a genuinely fresh deck. Reading
        # it back off the checkpoint rather than trusting the config is what keeps
        # a config edit from silently shape-mismatching a league already on disk
        # -- the same failure mode as the vocab/feature-dim asserts, but for a
        # value that is now per-league tunable (see trunk_hidden's own docstring
        # in run_league.py).
        live_path = f"{league_dir}/{name}/live.pt"
        width = ckpt_io.trunk_hidden_from_deck_checkpoint(live_path) or trunk_hidden
        net = build_deck_net(vocab.size, len(fixed_tables[name]), width)
        # .to(device) BEFORE the optimizer is built and BEFORE the checkpoint
        # load, and the order is load-bearing: Adam's own state tensors
        # (exp_avg/exp_avg_sq) are created lazily on the param's device, and
        # Optimizer.load_state_dict casts loaded state onto the device of the
        # param it belongs to. Building the optimizer first and moving the net
        # afterwards leaves the moments stranded on CPU while the params sit on
        # GPU, which fails at the first step() with a device mismatch.
        net.to(device)
        optimizer = torch.optim.Adam([p for p in net.parameters() if p.requires_grad], lr=hp["lr"])
        ckpt_io.load_deck_checkpoint(live_path, net, optimizer)  # no-op if live_path doesn't exist yet; optimizer load is migration-guarded (a migrated live.pt dropping "optimizer" -> fresh Adam)
        live_nets[name] = net
        optimizers[name] = optimizer

    # Fixed for the whole session (trunk widths never resize) -- computed once
    # here rather than re-derived by collect_rollout_league_parallel on every
    # one of its n_iterations * len(train_decks) calls below. Each net's
    # ENCODER is not broadcast this way: it trains, so it ships fresh inside
    # every net's own state_dict on each call.
    all_trunk_hidden = {name: tuple(layer.out_features for layer in net.trunk_layers) for name, net in live_nets.items()}

    # Per-deck mulligan model (rl.mulligan): owns the pregame keep/mulligan +
    # bottoming, trained by its OWN REINFORCE with a direct game-outcome reward
    # (decoupled from the main PPO above). Shares the same frozen stack. Resets
    # with the per-deck policies (mulligan.pt is deleted alongside live.pt).
    mulligan_nets, mulligan_optimizers = {}, {}
    for name in deck_names:
        mnet = MulliganNet(live_nets[name].encoder)
        # The encoder is SHARED with the deck net (same object), so it is
        # already on `device`, but MulliganNet's own heads were just built on
        # CPU -- leaving the module split across two devices. rl.mulligan.update
        # picks its device off `next(net.parameters()).device`, so a split
        # module fails on whichever half loses that coin flip. Same ordering
        # rule as the deck net above: move before the optimizer exists.
        mnet.to(device)
        mopt = torch.optim.Adam([p for p in mnet.parameters() if p.requires_grad], lr=hp["mulligan_lr"])
        mull_path = f"{league_dir}/{name}/mulligan.pt"
        ckpt_io.load_deck_checkpoint(mull_path, mnet, mopt)  # same migration-guarded optimizer load as the live-net path above
        mulligan_nets[name] = mnet
        mulligan_optimizers[name] = mopt

    # CPU mirrors for every path that PLAYS rather than trains.
    #
    # Game play -- the sequential collect fallback, --matchup pairings, and all
    # the end-of-session vs_history/vs_gauntlet/vs_heuristic checks -- runs
    # through collect_rollout(device="cpu"), which builds its tensors on CPU.
    # Handing those a GPU-resident net is a device mismatch, so on a GPU league
    # they get a mirror that is refreshed from the training weights on demand.
    #
    # When device == "cpu" these are the SAME OBJECTS, not copies: the default
    # path allocates nothing, copies nothing, and behaves exactly as it did
    # before this existed.
    if device == "cpu":
        cpu_nets, cpu_mulligan_nets = live_nets, mulligan_nets

        def sync_cpu_mirrors():
            pass
    else:
        cpu_nets = {n: copy.deepcopy(net).cpu().eval() for n, net in live_nets.items()}
        cpu_mulligan_nets = {n: copy.deepcopy(m).cpu().eval() for n, m in mulligan_nets.items()}

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
    # metrics.jsonl was not self-describing: a reader had no way to tell which
    # reward, roster, or config produced a given stretch of records, and no
    # record carried the league's cumulative games/deck at all -- reconstructing
    # "which snapshot era is session 34" meant counting PPO iterations by hand
    # and multiplying by games_per_iteration. One header per session fixes both.
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
        # anneal (the latter only when hp["ent_coef"] is None -- a configured
        # constant bypasses the schedule): both tracked against TRUE cumulative
        # games/deck (this
        # session's own starting point, cumulative_games, plus games played so
        # far THIS session) -- see batch_size_for_iteration's and
        # ent_coef_schedule's own docstrings (rl.train) for why session-local
        # tracking was replaced. games_per_iteration is the per-deck games
        # count each iteration adds, regardless of how many decks train.
        games_so_far_this_session = iteration * games_per_iteration
        batch_size = batch_size_for_iteration(cumulative_games + games_so_far_this_session,
                                              start=hp["seq_batch_size_start"], cap=hp["seq_batch_size_cap"])
        # hp["ent_coef"] pins a CONSTANT and skips the anneal entirely (Wave 2b);
        # None keeps the 0.02 -> 0.005 schedule. See PPO_DEFAULTS for the measured
        # reason the anneal is suspect.
        ent_coef = (hp["ent_coef"] if hp["ent_coef"] is not None
                    else ent_coef_schedule(cumulative_games + games_so_far_this_session))
        # Snapshot the mulligan nets for the workers now (ALL decks -- a frozen
        # opponent still needs its mulligan net to play). Updated once after the
        # deck-loop (on-policy within the iteration, same as the main nets).
        # .cpu() for the same reason the main nets are copied down in
        # rl.rollout_parallel: these cross a process boundary to CPU-only
        # workers. No-op when training is already on CPU.
        mulligan_state_dicts = {n: {k: v.cpu() for k, v in mulligan_nets[n].state_dict().items()}
                                for n in deck_names}
        for name in train_decks:  # only TRAIN decks get a round; the rest are frozen opponents
            t_collect0 = time.time()
            # Both CPU-playing collect paths below read the mirrors, and the
            # weights moved since the last round, so refresh first. Deliberately
            # NOT called on the parallel path: that one ships state_dicts to the
            # workers itself and never touches a mirror, so syncing there would
            # be ~10ms of pure waste on every one of a session's 500 rounds.
            # A no-op on CPU either way.
            if matchup is not None or executor is None:
                sync_cpu_mirrors()
            if matchup is not None:
                # Fixed pairing: this deck vs the OTHER named deck (no opponent
                # sampling), both sides carrying their REAL mulligan models via the
                # unified loop -- so logged matchup games use the same pregame policy
                # training does. game_logs (if given) accumulates across every round.
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
            # Gated on train_mulligan to match the FLUSH below, which is. Without
            # that symmetry a --train-deck-only session extended this list every
            # deck-round and never drained it, growing unboundedly for the whole
            # session (found 2026-08-19 while hunting a parent-process memory
            # leak; not the cause of that, but a real latent one).
            for deck_name, tr in mull_by_deck.items():
                if train_mulligan and deck_name in train_set:
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
                           cumulative_games=cumulative_games + games_so_far_this_session,
                           games=played, buffer_size=buffer_size, batch_size=batch_size, salvaged=salvaged,
                           ent_coef=ent_coef, policy_loss=policy_loss, value_loss=value_loss, entropy=entropy,
                           approx_kl=approx_kl, clip_fraction=clip_fraction, epochs_run=epochs_run,
                           explained_variance=explained_variance, adv_std=adv_std,
                           target_kl=hp["target_kl"])

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
                                   cumulative_games=cumulative_games + games_so_far_this_session,
                                   n=mull_stats[name]["n"], loss=mull_stats[name]["loss"])
                    mull_by_deck_accum[name] = []
        if mull_stats:  # readout so the mulligan subsystem is visible while it trains
            total_n = sum(s["n"] for s in mull_stats.values())
            mean_loss = sum(s["loss"] for s in mull_stats.values()) / len(mull_stats)
            print(f"  iter {iteration}: mulligan model -- {total_n} transitions across {len(mull_stats)} decks, "
                  f"mean REINFORCE loss {mean_loss:.4f}", flush=True)

        # Snapshot cadence is CUMULATIVE, not session-local (2026-08-13). This
        # used to read `(iteration + 1) % snapshot_every == 0`, but `iteration`
        # restarts at 0 every process invocation, so a session SHORTER than
        # snapshot_every never snapshotted at all and every session boundary
        # discarded the remainder -- the escalation ladder's early rungs (1, 2,
        # 4 iterations against snapshot_every=8) reached 168 games/deck without
        # ever registering one, leaving the opponent pool empty and
        # checkpoint_opponent_rate silently inoperative. Same session-local
        # counter flaw as BUG 1 (see advance_progress) and as the ent_coef /
        # batch_size ramps before them; gated on cumulative games crossing a
        # multiple of snapshot_every_games instead, which is what the
        # run-config field this is derived from actually names.
        # snapshot_every arrives in ITERATIONS (run_league.py converts), so
        # multiplying back recovers the games-count without a signature change
        # and keeps a raw --snapshot-every override behaving identically.
        games_before = cumulative_games + iteration * games_per_iteration
        crossed = should_snapshot(games_before, games_per_iteration, snapshot_every)
        if matchup is None and crossed:  # matchup: no historical snapshots
            for name in train_decks:  # only TRAIN decks change -> only they need new snapshots
                pool.register_snapshot(name, live_nets[name], mulligan_nets[name])
            _save_live_checkpoints(live_nets, optimizers, deck_names, session, session_path, league_dir,
                                   mulligan_nets, mulligan_optimizers)
            # Same cadence as the checkpoints above (not every iteration -- see
            # LeaguePool.save_opponent_stats' own docstring for why this specific
            # cadence, not "every game" or "never until session end").
            pool.save_opponent_stats()
            # BUG 3: and the cumulative counter, so progress.json can never be
            # left behind the weights those two lines just wrote. The iteration
            # is finished by here, so games_before + games_per_iteration is what
            # this league has actually trained.
            checkpoint_progress(league_dir, games_before + games_per_iteration)
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
    # Per-deck games/deck as of these evals: every deck advances by
    # games_per_iteration each iteration regardless of roster size, so this is
    # the same quantity progress.json's cumulative_games_per_deck tracks -- NOT
    # total_games, which sums across decks.
    cumulative_at_session_end = cumulative_games + n_iterations * games_per_iteration
    # eval_every_sessions > 1 spends the SAME total eval compute on fewer,
    # bigger, more precise readings (see EVAL_GAMES). Session 0 always evals so
    # a fresh league is never left with no reference at all.
    run_evals = eval_every_sessions <= 1 or session % eval_every_sessions == 0
    if not run_evals:
        print(f"  (evals skipped: session {session} % eval_every_sessions={eval_every_sessions} != 0)", flush=True)
    if matchup is None and run_evals:
        # Every check below PLAYS games on CPU nets, so the mirrors must carry
        # this session's final trained weights -- otherwise a GPU league would
        # measure whatever the mirrors last held (their session-start state),
        # silently reporting the wrong policy's win rate rather than crashing.
        sync_cpu_mirrors()
        for name in train_decks:
            for r in _run_eval_vs_history(name, cpu_nets[name], cpu_mulligan_nets[name], deck_ctxs[name],
                                          decklists[name], league_dir, horizon,
                                          games_per_snapshot=eval_games, seed=seed):
                _append_metric(league_dir, kind="vs_history", session=session, iteration=iteration, deck=name,
                               cumulative_games=cumulative_at_session_end, **r)
                print(f"  vs-history [{name}] vs {r['label']}: {r['live_wins']}/{r['games']} live wins "
                      f"({r['snapshot_wins']} snapshot wins, {r['no_winner']} no-winner)", flush=True)
            # The gauntlet check (an INDEPENDENTLY-trained twin population, see
            # _run_eval_vs_gauntlet's own docstring) -- None until that population's
            # training has reached this deck, or no gauntlet_league_dir is configured
            # for this league at all (most leagues won't have one).
            r = _run_eval_vs_gauntlet(name, cpu_nets[name], cpu_mulligan_nets[name], deck_ctxs[name],
                                      decklists[name], gauntlet_league_dir, horizon,
                                      games=eval_games, seed=seed)
            if r is not None:
                _append_metric(league_dir, kind="vs_gauntlet", session=session, iteration=iteration, deck=name,
                               cumulative_games=cumulative_at_session_end, **r)
                print(f"  vs-gauntlet [{name}]: {r['live_wins']}/{r['games']} live wins "
                      f"({r['gauntlet_wins']} gauntlet wins, {r['no_winner']} no-winner)", flush=True)
            # The tier-1 gauntlet member (a hand-authored HeuristicAgent) --
            # only for whichever deck(s) heuristic_decks names (most leagues
            # name none; see _run_eval_vs_heuristic's own docstring).
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


def league_roster(league_dir):
    """The deck names this league has actually trained, sorted -- i.e. those
    with a live.pt on disk. build_pool() deliberately spans the WHOLE manifest
    (all 11 decks, so vocab indices line up regardless of roster), so any
    eval script that iterates its decklists directly will try to load
    checkpoints a 4-deck league never wrote."""
    if not os.path.isdir(league_dir):
        return []
    return sorted(d for d in os.listdir(league_dir)
                  if os.path.exists(os.path.join(league_dir, d, "live.pt")))


def load_vintage_agent(league_dir, deck_name, vintage, deck_ctx, pool=None):
    """A frozen SeatAgent for ONE historical point of deck_name: an int/str
    snapshot id, or "live" for the current checkpoint pair.

    Path resolution is the only thing this adds over
    LeaguePool.load_snapshot_agent, which already does the loading: a snapshot
    sits either directly in the deck dir (still inside the active sampling
    window) or under archive/ once evicted, since LeaguePool.register_snapshot
    MOVES rather than deletes. Callers that walk many vintages should pass a
    single reused `pool` -- load_snapshot_agent caches by path, so the same
    snapshot is only ever read off disk once.

    Every vintage brings its OWN perception encoder (it lives inside the
    checkpoint), so a historical agent is played exactly as it was, rather
    than being replayed through whatever encoder is current -- which is what
    made a vintage comparison honest only while the stack was frozen forever.
    """
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
    """_play_eval_games with COMMON RANDOM NUMBERS: half the games with
    live_agent at seat 0, half with the sides swapped, both halves driven from
    the SAME seed.

    Because collect_rollout draws starting_idx per game from that rng
    (train.py), replaying the identical seed with the seats exchanged hands the
    play/draw to the OTHER agent on the very same shuffles. On-the-play is then
    balanced EXACTLY rather than in expectation, and the deck-order variance
    that dominates a card game largely cancels between the paired halves --
    worth an estimated 2-4x effective sample size for no extra games.

    The unpaired _play_eval_games always seats live at 0 and leaves both to
    chance. That is unbiased but noisy, and at the n=20 per check this project
    ran, noise was the binding constraint on every conclusion drawn."""
    half = n_games // 2
    fwd = _play_eval_games(live_agent, opp_agent, decklist, half, horizon,
                           random.Random(seed), opp_wins_key, greedy=greedy)
    rev = _play_eval_games(opp_agent, live_agent, decklist, n_games - half, horizon,
                           random.Random(seed), opp_wins_key, greedy=greedy)
    # rev is scored from the OPPONENT's seat, so live's wins there are rev's
    # opp_wins_key column and vice versa.
    return {"games": fwd["games"] + rev["games"],
            "live_wins": fwd["live_wins"] + rev[opp_wins_key],
            opp_wins_key: fwd[opp_wins_key] + rev["live_wins"],
            "no_winner": fwd["no_winner"] + rev["no_winner"],
            "paired": True}


def _play_eval_games(live_agent, opp_agent, decklist, n_games, horizon, rng, opp_wins_key,
                     greedy=True):
    """Shared tail for every vs_history/vs_gauntlet/vs_heuristic eval pass:
    plays a fixed live_agent-vs-opp_agent pairing greedily (no training,
    matching _run_eval's own eval convention), scores winners via
    collect_rollout's "game_over" event, and returns the standard
    {"games", "live_wins", <opp_wins_key>, "no_winner"} tally shape every
    caller already builds by hand. opp_wins_key names the opponent side in
    the returned dict (e.g. "snapshot_wins"/"gauntlet_wins"/"heuristic_wins")
    since each caller's opponent is a different kind of thing.

    greedy defaults to True -- the eval convention every in-training caller
    uses (measure the policy's actual best play, not an exploration sample).
    run_anchor_eval.py overrides it: argmax over a RANDOMLY initialized head
    is a degenerate constant-ish policy rather than an honest untrained
    baseline, so the random-init anchor wants sampling. collect_rollout takes
    one flag for the whole rollout, so this applies to BOTH seats."""
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
    the caller's already-loaded live_net/deck_ctx/decklist (no disk
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
        # snapshot_id/is_archive recorded (2026-08-13) because the two labels
        # are NOT comparable and were being read as one number. archive_oldest
        # is pinned to snapshot_0 forever -- eviction is oldest-first and this
        # takes the archive minimum -- so it is a FIXED ~200-game reference,
        # while active_oldest tracks a rolling ~6,400-game-old self. Without
        # the id on the record there is no way to tell from metrics.jsonl which
        # policy a given win rate was measured against.
        results.append({"label": label, "snapshot_id": snapshot_id, "is_archive": is_archive, **result})
    return results


def _run_eval_vs_gauntlet(deck_name, live_net, mulligan_net, deck_ctx, decklist, gauntlet_league_dir,
                           horizon, games=EVAL_GAMES, seed=None):
    """Plays deck_name's CURRENT live net against the SAME-NAMED deck from an
    INDEPENDENTLY-trained population (gauntlet_league_dir -- see
    training_configs/run_gauntlet.json's own _note) -- a genuinely EXTERNAL
    reference, not another point in this league's own self-play history the
    way _run_eval_vs_history's snapshots are. Two runs trained via the
    identical algorithm/settings still diverge into different regions of
    strategy space from nothing but a different nondeterministic training
    trajectory -- so a shared,
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
    # No stack-compatibility check is needed: the gauntlet's checkpoint brings
    # its own perception encoder, so its weights are never reinterpreted
    # through this league's. That confound (and the stack_id.txt bookkeeping
    # that guarded it) died with the shared frozen stack -- see build_deck_net.
    gauntlet_live_path = f"{gauntlet_league_dir}/{deck_name}/live.pt"
    if not os.path.exists(gauntlet_live_path):
        return None

    vocab, fixed_table = deck_ctx
    # The gauntlet is a DIFFERENT population and may legitimately have been
    # trained at a different trunk width than the league calling this, so its
    # own checkpoint is the only authority on the shape to build.
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
    """Plays deck_name's CURRENT live net against a HeuristicAgent (rl.heuristic_agent)
    for the SAME deck -- the gauntlet's tier-1 member: a hand-authored, non-
    self-play reference opponent, distinct from _run_eval_vs_gauntlet's tier-2
    (an independently-TRAINED population). Only called for whichever deck(s)
    _run_session's own caller configured (see heuristic_decks below) -- the
    heuristic's rules are general MTG principles the owner hand-picked for
    ONE deck (mono_red_rally), not audited or intended for every deck in a
    roster."""
    live_agent = SeatAgent(live_net, mulligan_net, deck_ctx)
    heuristic_agent = HeuristicAgent(deck_ctx)
    return _play_paired_eval_games(live_agent, heuristic_agent, decklist, games, horizon, seed, "heuristic_wins")


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
