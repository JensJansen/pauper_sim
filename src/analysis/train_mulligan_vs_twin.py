"""One-time mulligan-net bootstrap: freeze BOTH populations' DeckNetworks
(main AND twin) and train ONLY a fresh MulliganNet attached to main's net,
via its own REINFORCE, by playing main's frozen policy against twin's frozen
policy. rl/mulligan.py's 2026-08-20 fix gave the net eyes (it now reads the
same structured, self-attended hand representation the main policy sees,
instead of a bare mean-pooled card-identity embedding that made it blind to
land count); every mulligan.pt on disk predates that fix and now fails to
load (see MulliganNet.hand_repr_version). This is the retrain.

Two phases, one script:

  1. TRAIN. For each main-league deck, --train-games games (default 1000),
     round-robin split EVENLY across the TWIN opponent roster -- --twin-decks
     if given, else the SAME roster --decks restricts training to (so with
     neither flag given, every main deck round-robins against all four twin
     decks; with only --decks narrowed, training is same-archetype-only
     unless --twin-decks is ALSO given explicitly -- every run before
     2026-08-21 used --decks alone to isolate one deck for ablation, so
     every one of those was unknowingly same-archetype-only, not the
     opponent-diverse roster this paragraph used to claim was the default
     regardless of --decks). An opponent-diverse bootstrap generalizes
     better than training against one fixed archetype -- see --twin-decks.
     Twin's pregame decider is a
     scripted stand-in, never twin's own mulligan.pt (pre-fix, and cannot be
     loaded under the new class without touching twin's checkpoint, which
     this script must not do -- see below): --twin-mulligan chooses between
     AlwaysKeep (default, every prior run) and MulliganZeroLands (mulligans
     only an unambiguous 0-land hand -- removes the twin's own worst-hand
     losses as a noise source in the reward the MAIN seat's mulligan net
     learns from, without granting twin any real hand-evaluation skill).
     Main's own DeckNetwork is loaded with requires_grad=False and NEVER
     gets an optimizer -- structurally incapable of moving, not just "we
     won't call step() on it".

     TWIN IS READ-ONLY, ENFORCED STRUCTURALLY, NOT JUST BY CONVENTION: this
     script contains no save_deck_checkpoint/save_snapshot/register_snapshot
     call whose path is ever built from --twin. Grep for `twin_dir` before
     trusting that claim again after an edit.

  2. EVAL. For each main-league deck, --eval-games games (default 100) under
     EACH of three pregame deciders, all sharing the SAME frozen main net so
     the pregame policy is the only thing that varies: RandomMulligan (no
     signal, pure noise -- the floor), AlwaysKeep (no signal, but not
     actively harmful -- the cheap heuristic), and the just-trained
     MulliganNet (greedy). Reports win rate per arm vs twin (AlwaysKeep),
     plus -- for the trained arm only -- a land-count/keep-rate/confidence
     breakdown reconstructed from the event log, the exact audit that first
     surfaced the pre-fix bug (0-land hands kept 50% of the time at ~92%
     confidence, 0/8 wins). That gives a second, more granular signal than
     the aggregate win rate: "is it better on average" vs "did it actually
     stop keeping unplayable hands".

TRAJECTORY, not just the endpoint. Every prior run of this script only ever
inspected the FINAL greedy snapshot -- across four different training-regime
changes (twin/AlwaysKeep at 1000 and 3000 games, cost=0 ablation, a same-
strength self-mirror variant) every one ended at "never crosses 50% on any
hand, ever", with no visibility into whether P(mulligan) was trending toward
that threshold, away from it, or never moving at all. _mulligan_common.
build_probe_hands builds eight FIXED synthetic hands per deck (every land
count 0 through 7, built from real cards in that deck's own list); after
every chunk's REINFORCE
update, probe_p_mulligan reads P(mulligan) back out for each and it gets
logged (printed live, and saved under each deck's "probe_trajectory" in
--out's JSON) -- so "is more training even the right lever" becomes a
directly answerable question instead of another blind multi-thousand-game
bet.

Output: checkpoints/<main>/<deck>/<--bootstrap-name>.pt (default
mulligan_bootstrap.pt) -- a NEW filename, never overwrites the real
mulligan.pt. Review the eval numbers below before promoting it (rename to
mulligan.pt, or fold it into normal joint league training -- freezing the
main net forever would reintroduce the encoder-drift risk MulliganNet's own
docstring already flags).

--decks restricts the roster (default: every deck both leagues have),
--entropy-coef overrides rl.mulligan.ENTROPY_COEF for this run only (the
same process-local monkeypatch pattern -- see main(), read fresh by
update() on every call, never written back to mulligan.py), --stratify-
0land-pct/--stratify-7land-pct force that fraction of the main seat's
training deals to 0 lands / 7 lands (all-land) respectively (game.state.
build_shuffled_library's force_land_count, via collect_rollout -- 0-land
is naturally ~14% of hands, so a 50-game REINFORCE batch still only
carries a handful of that decisive case; 7-land is naturally ~0.001%, so
without --stratify-7land-pct the net gets essentially ZERO real exposure
to the other extreme at all -- a 2026-08-21 run stratified on 0-land alone
learned P(mulligan) monotonic in land count, correctly high at 0 lands but
driven to ~0 at 7, the same "always keep" mistake everywhere else, purely
by extrapolating a feature it had only ever seen pushed one direction),
--twin-mulligan swaps twin's pregame decider (see above), and
--bootstrap-name lets two runs against the SAME deck coexist under
different output filenames. Together these are what a same-deck A/B
ablation needs, e.g. does more training resolve the "trained_mulligan ==
always_keep, every game" result from the first full run (no -- see the
probe-trajectory paragraph below), does a stronger entropy bonus prevent
the collapse the probe trajectory actually found (only partially -- it
trades collapse-to-always-keep for washing out hand-quality
differentiation instead), or does forcing enough 0-land exposure into
every batch fix it on its own (partially -- real, growing differentiation
between hands, but the whole distribution still drifts toward keep at the
default entropy coefficient):

    python -u analysis/train_mulligan_vs_twin.py --decks mono_red_rally \
        --train-games 3000 --bootstrap-name mulligan_bootstrap_default \
        --out ../logs/mulligan_ablation_default.json
    python -u analysis/train_mulligan_vs_twin.py --decks mono_red_rally \
        --train-games 1200 --entropy-coef 0.3 --bootstrap-name mulligan_bootstrap_entropy_0p3 \
        --out ../logs/mulligan_ablation_entropy_0p3.json

rl.mulligan's own per-mulligan-count reward penalty (MULLIGAN_COST) --
formerly also overridable here via --mulligan-cost -- is GONE as of
2026-08-21, not just defaulted to zero: an early 3000-game ablation (cost=0
vs cost=0.02, no stratification, no zero_lands twin) found byte-identical
outcomes, but neither arm had learned a working policy yet, so that result
never actually distinguished "the penalty doesn't matter" from "nothing
mattered yet, penalty included." Once stratify_0land_pct + a
MulliganZeroLands twin + enough games (~16-25k) produced a real, decisive
policy, the SAME question mattered again in a way that could no longer be
deferred: every one of those checkpoints was trained under cost=0, but
rl.mulligan's module default was still cost=0.02 -- attaching a checkpoint
into league training under the OLD default would have silently changed the
reward function (and the loaded Adam optimizer's momentum) out from under
it the moment training resumed. Removed from rl.mulligan.mulligan_reward
entirely rather than leave that mismatch in place. See rl/mulligan.py's
own docstring.

    python -u analysis/train_mulligan_vs_twin.py \
        --main 4_deck_subleague_test --twin 4_deck_subleague_gauntlet \
        --out ../logs/mulligan_bootstrap_vs_twin.json
"""
import argparse
import json
import os
import random
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch  # noqa: E402

from _mulligan_common import (  # noqa: E402
    audit_land_counts, build_probe_hands, load_frozen_nets, print_land_audit, probe_p_mulligan,
)
from rl import checkpoint as ckpt_io  # noqa: E402
from rl import mulligan as mulligan_mod  # noqa: E402 -- module itself, so --entropy-coef can patch its ENTROPY_COEF global
from rl.agent import AlwaysKeep, MulliganZeroLands, RandomMulligan, SeatAgent  # noqa: E402
from rl.league_runner import HORIZON, PPO_DEFAULTS  # noqa: E402
from rl.mulligan import MulliganNet, update as mulligan_update  # noqa: E402
from rl.pool import build_pool  # noqa: E402
from rl.rewards import deploy_reward_v6  # noqa: E402
from rl.train import collect_rollout  # noqa: E402
from repo_paths import CHECKPOINTS_DIR  # noqa: E402


def _twin_cycling_pairing(main_agent, main_bucket, main_decklist, reward_fn,
                          twin_agents, twin_decklists, twin_names, main_seat):
    """Cycles through twin's roster round-robin, one opponent per game
    (collect_rollout calls pairing(rng) fresh every game) -- so even a short
    batch touches every twin archetype rather than running n games in a row
    against just one. main_seat pins which physical seat carries the main
    deck for every game this pairing plays; on-the-play itself still varies
    game to game via collect_rollout's own starting_idx randomization, so
    fixing main_seat here does not bias which side goes first.
    main_bucket/reward_fn=None (the eval-phase call) makes record_as/
    reward_fns [None, None] on both sides -- correct for record=False,
    where neither is ever read."""
    counter = [0]

    def pairing(rng):
        opp = twin_names[counter[0] % len(twin_names)]
        counter[0] += 1
        twin_agent = twin_agents[opp]
        if main_seat == 0:
            return ([main_agent, twin_agent], [main_decklist, twin_decklists[opp]],
                    [reward_fn, None], [main_bucket, None])
        return ([twin_agent, main_agent], [twin_decklists[opp], main_decklist],
                [None, reward_fn], [None, main_bucket])
    return pairing


def _train_mulligan(main_net, main_decklist, main_bucket, deck_ctx,
                    twin_agents, twin_decklists, twin_names,
                    n_games, update_every, seed, horizon, mulligan_lr, probe_hands=None,
                    stratify_0land_pct=0.0, stratify_7land_pct=0.0,
                    resume_from=None, games_already=0):
    """Fresh MulliganNet on main_net's encoder, trained by REINFORCE
    (rl.mulligan.update) against twin's full roster (AlwaysKeep), main
    pinned to seat 0 throughout -- seat swapping only matters for removing
    an on-the-play confound from a WIN-RATE comparison (the eval phase
    below); it buys nothing for training the mulligan policy itself, since
    on_the_play is already one of the two scalars the net observes and
    already varies game to game regardless of which physical seat is fixed.
    Reward is filled in by collect_rollout itself (rl.mulligan.mulligan_
    reward, called internally per mull_game[seat]) -- nothing here computes
    it by hand.

    probe_hands (optional, from _mulligan_common.build_probe_hands): if
    given, P(mulligan) on each fixed probe hand is read back (probe_p_
    mulligan) after EVERY chunk's REINFORCE update and appended to the
    returned trajectory -- the actual point being to see whether P(mulligan
    | 0 lands) is rising, falling, or flat over training, instead of only
    ever inspecting the final snapshot (2026-08-20, after four different
    training-regime changes all ended at "never crosses 50%" with no visibility
    into whether any of them were even trending the right way).

    Returns (mnet, optimizer, win_tally, probe_trajectory) -- probe_trajectory
    is [] when probe_hands is None.

    stratify_0land_pct, stratify_7land_pct: passed straight through to
    collect_rollout -- see its own docstring. 0.0/0.0 (default) is the
    untouched natural-shuffle behavior every prior run here used.

    resume_from (optional path): CONTINUES an earlier run's actual net +
    optimizer state (rl.checkpoint.load_deck_checkpoint, the same format
    save_deck_checkpoint below writes) instead of starting a fresh
    MulliganNet. Exists because torch.manual_seed is never called anywhere
    in this script -- MulliganNet's random init and every non-greedy
    torch.distributions.Categorical(...).sample() draw from torch's global
    RNG, which --seed does NOT control (only the python `random.Random(seed)`
    game-dealing rng is reproducible run to run) -- so simply relaunching
    with a bigger --train-games produces an INDEPENDENT run, not a
    continuation of a specific prior trajectory. games_already: purely a
    labeling offset (the checkpoint itself doesn't record how many games
    trained it) so printed lines and probe_trajectory's "games" continue
    the earlier run's numbering instead of restarting at 0. win_tally
    still starts fresh at 0/0 -- the earlier run's tally isn't in the
    checkpoint either, and mixing it in would misrepresent the CURRENT
    win rate as an average across two different net-weight regimes."""
    mnet = MulliganNet(main_net.encoder)
    opt = torch.optim.Adam([p for p in mnet.parameters() if p.requires_grad], lr=mulligan_lr)
    if resume_from is not None:
        loaded = ckpt_io.load_deck_checkpoint(resume_from, mnet, opt)
        if not loaded:
            raise SystemExit(f"--resume-from {resume_from} does not exist")
    main_agent = SeatAgent(main_net, mnet, deck_ctx)
    pairing = _twin_cycling_pairing(main_agent, main_bucket, main_decklist, deploy_reward_v6,
                                    twin_agents, twin_decklists, twin_names, main_seat=0)
    rng = random.Random(seed)
    win_tally = {"wins": 0, "decided": 0}

    def on_game_end(state):
        if state.winner is None:
            return
        win_tally["decided"] += 1
        if state.winner == 0:
            win_tally["wins"] += 1

    played_this_run = 0  # chunk-sizing math below; n_games is games to play in THIS call, not a cumulative target
    probe_trajectory = []
    n_chunks = max(1, n_games // update_every)
    for chunk in range(n_chunks):
        chunk_games = update_every if chunk < n_chunks - 1 else n_games - played_this_run
        _buffers, mull_by_deck, n_played = collect_rollout(
            pairing, chunk_games, horizon, rng, device="cpu",
            record=True, greedy=False, on_game_end=on_game_end,
            stratify_0land_pct=stratify_0land_pct, stratify_7land_pct=stratify_7land_pct)
        played_this_run += n_played
        played = games_already + played_this_run  # cumulative, for labeling only
        transitions = mull_by_deck.get(main_bucket, [])
        stats = mulligan_update(mnet, opt, transitions)
        rate = win_tally["wins"] / win_tally["decided"] if win_tally["decided"] else float("nan")
        line = (f"    [{main_bucket}] {played}/{games_already + n_games} games, win_rate={rate:.3f}, "
                f"mulligan n={stats['n']} loss={stats.get('loss', float('nan')):.4f}")
        if probe_hands is not None:
            probes = probe_p_mulligan(mnet, probe_hands)
            probe_trajectory.append({"games": played, **probes})
            line += " | P(mulligan): " + " ".join(f"{k}={v:.3f}" for k, v in probes.items())
        print(line, flush=True)
    return mnet, opt, win_tally, probe_trajectory


def _play_eval_arm(agent, main_decklist, twin_agents, twin_decklists, twin_names,
                   n_games, seed, horizon, game_logs=None):
    """n_games split evenly across both seats (on-the-play fairness -- same
    reasoning cross_league_round_robin.py's own docstring gives for its
    identical split) and round-robin across twin's roster. record=False,
    greedy=True throughout (this measures the policy's best play, not an
    exploration sample -- matching every other eval in this repo). Returns
    (wins, decided) for the agent's side; a horizon timeout (state.winner is
    None) counts toward neither."""
    half = n_games // 2
    tally = {"wins": 0, "decided": 0}
    for main_seat in (0, 1):
        def on_game_end(state, main_seat=main_seat):
            if state.winner is None:
                return
            tally["decided"] += 1
            if state.winner == main_seat:
                tally["wins"] += 1
        pairing = _twin_cycling_pairing(agent, None, main_decklist, None,
                                        twin_agents, twin_decklists, twin_names, main_seat=main_seat)
        collect_rollout(pairing, half, horizon, random.Random(seed + main_seat), device="cpu",
                        record=False, greedy=True, on_game_end=on_game_end, game_logs=game_logs)
    return tally["wins"], tally["decided"]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--main", default="4_deck_subleague_test", help="main league checkpoint dir name")
    ap.add_argument("--twin", default="4_deck_subleague_gauntlet", help="twin league checkpoint dir name -- READ ONLY")
    ap.add_argument("--train-games", type=int, default=1000, help="training games per main deck")
    ap.add_argument("--update-every", type=int, default=50, help="games per REINFORCE update")
    ap.add_argument("--eval-games", type=int, default=100, help="eval games per policy arm per main deck")
    ap.add_argument("--seed", type=int, default=20260820)
    ap.add_argument("--out", default=None, help="write full results here as JSON")
    ap.add_argument("--decks", nargs="+", default=None,
                    help="restrict to these MAIN deck names to train (default: every deck both leagues have a "
                         "live.pt for). Also sets the TWIN opponent roster unless --twin-decks overrides it "
                         "separately -- see --twin-decks")
    ap.add_argument("--twin-decks", nargs="+", default=None,
                    help="restrict the TWIN opponent roster independently of --decks (default: same as --decks, "
                         "i.e. a same-archetype-only bootstrap -- every run before 2026-08-21 used this default, "
                         "DESPITE the module docstring above claiming round-robin-across-all-four is the norm; "
                         "that claim was only ever true when --decks was left unset entirely). Pass all four twin "
                         "decks here (independent of a narrower --decks) for an opponent-diverse bootstrap")
    ap.add_argument("--entropy-coef", type=float, default=None,
                    help="override rl.mulligan.ENTROPY_COEF for this run only (ablation -- default: leave it alone)")
    ap.add_argument("--stratify-0land-pct", type=float, default=0.0,
                    help="fraction of training games (main seat only) whose opening hand is forced to 0 lands "
                         "(game.state.build_shuffled_library's force_land_count, via collect_rollout's "
                         "stratify_0land_pct) -- 0.0 (default) leaves dealing fully natural, same as every prior run")
    ap.add_argument("--stratify-7land-pct", type=float, default=0.0,
                    help="fraction of training games (main seat only) whose opening hand is forced to 7 lands "
                         "(all-land, no spells) -- the OTHER extreme from --stratify-0land-pct. A flooded hand is "
                         "naturally on the order of 1000x rarer than a 0-land hand, so without this the net never "
                         "sees one and has no signal to learn that hand quality isn't monotonic in land count")
    ap.add_argument("--twin-mulligan", choices=["always_keep", "zero_lands"], default="always_keep",
                    help="twin's pregame decider (default: always_keep, matching every prior run). zero_lands "
                         "(rl.agent.MulliganZeroLands) mulligans a 0-land hand and keeps everything else -- removes "
                         "the twin's own worst-hand losses as a noise source in the main seat's mulligan reward")
    ap.add_argument("--bootstrap-name", default="mulligan_bootstrap",
                    help="output checkpoint filename stem (no .pt) -- vary this to run two configs "
                         "against the same deck without one overwriting the other's result")
    ap.add_argument("--resume-from", default=None,
                    help="path to an existing bootstrap .pt (this exact deck) to CONTINUE training from, "
                         "loading its actual net+optimizer state, instead of a fresh MulliganNet -- required "
                         "because torch.manual_seed is never called here, so a fresh run with a bigger "
                         "--train-games is an independent draw, not a continuation of a specific prior run. "
                         "Only valid with --decks restricted to exactly one deck.")
    ap.add_argument("--resume-games-already", type=int, default=0,
                    help="how many games --resume-from's checkpoint already represents -- a pure labeling "
                         "offset (the checkpoint itself doesn't record this) so printed/logged \"games\" and "
                         "probe_trajectory continue the earlier run's numbering instead of restarting at 0")
    args = ap.parse_args()

    if args.resume_from is not None and (args.decks is None or len(args.decks) != 1):
        raise SystemExit("--resume-from requires --decks restricted to exactly one deck")

    if args.entropy_coef is not None:
        # Same process-local monkeypatch pattern -- update() reads ENTROPY_COEF
        # as a module global at call time too, so this is safe and leaves
        # mulligan.py itself untouched.
        print(f"ABLATION: overriding rl.mulligan.ENTROPY_COEF {mulligan_mod.ENTROPY_COEF} -> {args.entropy_coef}")
        mulligan_mod.ENTROPY_COEF = args.entropy_coef

    decklists, vocab, deck_ctxs, fixed_tables = build_pool()
    main_dir = os.path.join(CHECKPOINTS_DIR, args.main)
    twin_dir = os.path.join(CHECKPOINTS_DIR, args.twin)  # READ ONLY -- never passed to a save_* call below

    deck_names = sorted(n for n in decklists
                        if os.path.exists(f"{main_dir}/{n}/live.pt") and os.path.exists(f"{twin_dir}/{n}/live.pt"))
    if args.decks is not None:
        missing = set(args.decks) - set(deck_names)
        if missing:
            raise SystemExit(f"--decks named {sorted(missing)}, not present (with a live.pt) in BOTH leagues")
        deck_names = [n for n in deck_names if n in set(args.decks)]
    if not deck_names:
        raise SystemExit(f"no deck has a live.pt in BOTH {args.main} and {args.twin}")

    twin_deck_names = deck_names  # default: same-archetype-only, matching every run before 2026-08-21
    if args.twin_decks is not None:
        available_twin = sorted(n for n in decklists if os.path.exists(f"{twin_dir}/{n}/live.pt"))
        missing = set(args.twin_decks) - set(available_twin)
        if missing:
            raise SystemExit(f"--twin-decks named {sorted(missing)}, not present (with a live.pt) in {args.twin}")
        twin_deck_names = sorted(args.twin_decks)

    print(f"main={args.main} twin={args.twin} (twin read-only) decks={deck_names} twin_decks={twin_deck_names}")
    print(f"train_games/deck={args.train_games} update_every={args.update_every} "
          f"eval_games/arm={args.eval_games} seed={args.seed} "
          f"entropy_coef={mulligan_mod.ENTROPY_COEF} stratify_0land_pct={args.stratify_0land_pct} "
          f"stratify_7land_pct={args.stratify_7land_pct} twin_mulligan={args.twin_mulligan} "
          f"resume_from={args.resume_from} resume_games_already={args.resume_games_already}", flush=True)

    main_nets = load_frozen_nets(main_dir, deck_names, vocab, fixed_tables)
    twin_nets = load_frozen_nets(twin_dir, twin_deck_names, vocab, fixed_tables)
    # A single shared decider instance/rng across every twin deck+arm, same
    # convention AlwaysKeep() (stateless) and the existing RandomMulligan
    # eval arm (one Random(args.seed), reused) already use here.
    twin_decider = AlwaysKeep() if args.twin_mulligan == "always_keep" else MulliganZeroLands(random.Random(args.seed + 3000))
    twin_agents = {n: SeatAgent(twin_nets[n], twin_decider, deck_ctxs[n]) for n in twin_deck_names}
    mulligan_lr = PPO_DEFAULTS["mulligan_lr"]

    t0 = time.time()
    results = {"main_league": args.main, "twin_league": args.twin,
               "train_games_per_deck": args.train_games, "eval_games_per_arm": args.eval_games,
               "seed": args.seed,
               "entropy_coef": mulligan_mod.ENTROPY_COEF, "stratify_0land_pct": args.stratify_0land_pct,
               "stratify_7land_pct": args.stratify_7land_pct,
               "twin_mulligan": args.twin_mulligan, "twin_decks": twin_deck_names, "decks": {}}

    for name in deck_names:
        print(f"\n=== {name}: TRAIN vs twin roster {twin_deck_names} ({args.train_games} games) ===", flush=True)
        probe_hands = build_probe_hands(decklists[name], vocab)
        mnet, opt, train_tally, probe_trajectory = _train_mulligan(
            main_nets[name], decklists[name], name, deck_ctxs[name],
            twin_agents, decklists, twin_deck_names,
            args.train_games, args.update_every, args.seed, HORIZON, mulligan_lr, probe_hands=probe_hands,
            stratify_0land_pct=args.stratify_0land_pct, stratify_7land_pct=args.stratify_7land_pct,
            resume_from=args.resume_from, games_already=args.resume_games_already)
        train_rate = train_tally["wins"] / train_tally["decided"] if train_tally["decided"] else float("nan")
        print(f"  training done: {train_tally['wins']}/{train_tally['decided']} = {train_rate:.3f} "
              f"vs twin ({args.twin_mulligan})")

        out_path = f"{main_dir}/{name}/{args.bootstrap_name}.pt"
        ckpt_io.save_deck_checkpoint(out_path, mnet, opt)
        print(f"  saved {out_path}")

        print(f"  === {name}: EVAL, 3 pregame policies x {args.eval_games} games "
              f"vs twin ({args.twin_mulligan}) ===", flush=True)
        arms = [
            ("random_mulligan", SeatAgent(main_nets[name], RandomMulligan(random.Random(args.seed)), deck_ctxs[name])),
            ("always_keep", SeatAgent(main_nets[name], AlwaysKeep(), deck_ctxs[name])),
            ("trained_mulligan", SeatAgent(main_nets[name], mnet, deck_ctxs[name])),
        ]
        deck_result = {"train_win_rate": train_rate, "train_games_decided": train_tally["decided"],
                       "probe_trajectory": probe_trajectory, "arms": {}}
        for arm_name, agent in arms:
            logs = [] if arm_name == "trained_mulligan" else None
            wins, decided = _play_eval_arm(agent, decklists[name], twin_agents, decklists, twin_deck_names,
                                           args.eval_games, args.seed, HORIZON, game_logs=logs)
            rate = wins / decided if decided else float("nan")
            print(f"    {arm_name:16s}: {wins:3d}/{decided:3d} = {rate:.3f}")
            deck_result["arms"][arm_name] = {"wins": wins, "decided": decided, "win_rate": rate}
            if logs:
                audit = audit_land_counts(logs)
                print_land_audit(audit)
                deck_result["land_audit"] = {
                    str(lc): {k: v for k, v in d.items() if k != "keep_probs"} for lc, d in audit.items()
                }
        results["decks"][name] = deck_result

    print(f"\ndone in {time.time() - t0:.1f}s")
    if args.out:
        with open(args.out, "w") as f:
            json.dump(results, f, indent=2)
        print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
