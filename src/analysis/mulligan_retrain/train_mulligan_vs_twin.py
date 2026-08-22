"""One-time mulligan-net bootstrap: freeze both populations' DeckNetworks
(main and twin) and train only a fresh MulliganNet attached to main's net,
via REINFORCE, by playing main's frozen policy against twin's frozen policy.

Two phases, one script:

  1. TRAIN. For each main-league deck, --train-games games (default 1000),
     round-robin split evenly across the twin opponent roster (--twin-decks
     if given, else the same roster --decks restricts training to). Twin's
     pregame decider is a scripted stand-in (--twin-mulligan), never twin's
     own mulligan.pt. Main's own DeckNetwork is loaded with
     requires_grad=False and never gets an optimizer. Twin is read-only,
     enforced structurally: this script contains no
     save_deck_checkpoint/save_snapshot/register_snapshot call whose path is
     ever built from --twin.

  2. EVAL. For each main-league deck, --eval-games games (default 100) under
     each of three pregame deciders, all sharing the same frozen main net:
     RandomMulligan (the floor), AlwaysKeep (the cheap heuristic), and the
     just-trained MulliganNet (greedy). Reports win rate per arm vs twin,
     plus -- for the trained arm only -- a land-count/keep-rate/confidence
     breakdown reconstructed from the event log.

Trajectory, not just the endpoint: _mulligan_common.build_probe_hands builds
eight fixed synthetic hands per deck; after every chunk's REINFORCE update,
probe_p_mulligan reads P(mulligan) back out for each and logs it (printed
live, and saved under each deck's "probe_trajectory" in --out's JSON).

Output: checkpoints/<main>/<deck>/<--bootstrap-name>.pt (default
mulligan_bootstrap.pt) -- a new filename, never overwrites the real
mulligan.pt. Review the eval numbers before promoting it.

See --help for the full flag list (roster/opponent selection, land-count
stratification, entropy override, resume).

    python -u analysis/mulligan_retrain/train_mulligan_vs_twin.py --decks mono_red_rally \
        --train-games 3000 --bootstrap-name mulligan_bootstrap_default \
        --out ../logs/mulligan_ablation_default.json
    python -u analysis/mulligan_retrain/train_mulligan_vs_twin.py --decks mono_red_rally \
        --train-games 1200 --entropy-coef 0.3 --bootstrap-name mulligan_bootstrap_entropy_0p3 \
        --out ../logs/mulligan_ablation_entropy_0p3.json

    python -u analysis/mulligan_retrain/train_mulligan_vs_twin.py \
        --main 4_deck_subleague_test --twin 4_deck_subleague_gauntlet \
        --out ../logs/mulligan_bootstrap_vs_twin.json
"""
import argparse
import json
import os
import random
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import torch  # noqa: E402

from _mulligan_common import (  # noqa: E402
    audit_land_counts, build_probe_hands, load_frozen_nets, print_land_audit, probe_p_mulligan,
)
from rl import checkpoint as ckpt_io  # noqa: E402
from rl.model import mulligan as mulligan_mod  # noqa: E402 -- so --entropy-coef can patch ENTROPY_COEF
from rl.decision.agent import AlwaysKeep, MulliganZeroLands, RandomMulligan, SeatAgent  # noqa: E402
from rl.league.league_runner import HORIZON, PPO_DEFAULTS  # noqa: E402
from rl.model.mulligan import MulliganNet, update as mulligan_update  # noqa: E402
from rl.roster import build_pool  # noqa: E402
from rl.rewards import deploy_reward_v6  # noqa: E402
from rl.training.train import collect_rollout  # noqa: E402
from repo_paths import CHECKPOINTS_DIR  # noqa: E402


def _twin_cycling_pairing(main_agent, main_bucket, main_decklist, reward_fn,
                          twin_agents, twin_decklists, twin_names, main_seat):
    """Cycles through twin's roster round-robin, one opponent per game, so
    even a short batch touches every twin archetype. main_seat pins which
    physical seat carries the main deck; on-the-play itself still varies via
    collect_rollout's starting_idx randomization. main_bucket/reward_fn=None
    (the eval-phase call) is correct for record=False, where neither is read."""
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
    (rl.model.mulligan.update) against twin's full roster (AlwaysKeep), main
    pinned to seat 0 throughout -- on_the_play still varies game to game
    regardless of which physical seat is fixed, so this doesn't bias
    training; only a win-rate comparison needs the seat swap (eval phase).
    Reward is filled in by collect_rollout itself.

    probe_hands (optional, from _mulligan_common.build_probe_hands): if
    given, P(mulligan) on each fixed probe hand is read back after every
    chunk's REINFORCE update and appended to the returned trajectory.
    Returns (mnet, optimizer, win_tally, probe_trajectory); probe_trajectory
    is [] when probe_hands is None.

    stratify_0land_pct, stratify_7land_pct are passed through to
    collect_rollout. resume_from (optional path): continues an earlier run's net + optimizer
    state instead of starting fresh -- needed because torch.manual_seed is
    never called here, so MulliganNet's random init and sampling draw from
    torch's global RNG, uncontrolled by --seed; relaunching with a bigger
    --train-games would otherwise be an independent run, not a continuation.
    games_already is a labeling offset only. win_tally still starts fresh at 0/0."""
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

    played_this_run = 0  # n_games is this call's target, not cumulative
    probe_trajectory = []
    n_chunks = max(1, n_games // update_every)
    for chunk in range(n_chunks):
        chunk_games = update_every if chunk < n_chunks - 1 else n_games - played_this_run
        _buffers, mull_by_deck, n_played = collect_rollout(
            pairing, chunk_games, horizon, rng, device="cpu",
            record=True, greedy=False, on_game_end=on_game_end,
            stratify_0land_pct=stratify_0land_pct, stratify_7land_pct=stratify_7land_pct)
        played_this_run += n_played
        played = games_already + played_this_run  # for labeling only
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
    """n_games split evenly across both seats for on-the-play fairness, and
    round-robin across twin's roster. record=False, greedy=True throughout.
    Returns (wins, decided) for the agent's side; a horizon timeout
    (state.winner is None) counts toward neither."""
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
                         "i.e. same-archetype-only). Pass all four twin decks for an opponent-diverse bootstrap")
    ap.add_argument("--entropy-coef", type=float, default=None,
                    help="override rl.model.mulligan.ENTROPY_COEF for this run only (ablation -- default: leave it alone)")
    ap.add_argument("--stratify-0land-pct", type=float, default=0.0,
                    help="fraction of training games (main seat only) whose opening hand is forced to 0 lands "
                         "(game.state.build_shuffled_library's force_land_count, via collect_rollout's "
                         "stratify_0land_pct) -- 0.0 (default) leaves dealing fully natural")
    ap.add_argument("--stratify-7land-pct", type=float, default=0.0,
                    help="fraction of training games (main seat only) whose opening hand is forced to 7 lands "
                         "(all-land, no spells) -- the other extreme from --stratify-0land-pct")
    ap.add_argument("--twin-mulligan", choices=["always_keep", "zero_lands"], default="always_keep",
                    help="twin's pregame decider (default: always_keep). zero_lands "
                         "(rl.decision.agent.MulliganZeroLands) mulligans a 0-land hand and keeps everything else -- removes "
                         "the twin's own worst-hand losses as a noise source in the main seat's mulligan reward")
    ap.add_argument("--bootstrap-name", default="mulligan_bootstrap",
                    help="output checkpoint filename stem (no .pt) -- vary this to run two configs "
                         "against the same deck without one overwriting the other's result")
    ap.add_argument("--resume-from", default=None,
                    help="path to an existing bootstrap .pt (this exact deck) to continue training from, "
                         "loading its actual net+optimizer state, instead of a fresh MulliganNet. "
                         "Only valid with --decks restricted to exactly one deck.")
    ap.add_argument("--resume-games-already", type=int, default=0,
                    help="how many games --resume-from's checkpoint already represents -- a pure labeling "
                         "offset so printed/logged \"games\" and probe_trajectory continue the earlier run's "
                         "numbering instead of restarting at 0")
    args = ap.parse_args()

    if args.resume_from is not None and (args.decks is None or len(args.decks) != 1):
        raise SystemExit("--resume-from requires --decks restricted to exactly one deck")

    if args.entropy_coef is not None:
        print(f"ABLATION: overriding rl.model.mulligan.ENTROPY_COEF {mulligan_mod.ENTROPY_COEF} -> {args.entropy_coef}")
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

    twin_deck_names = deck_names  # default: same-archetype-only
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
    # One shared decider instance/rng across every twin deck and arm.
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
