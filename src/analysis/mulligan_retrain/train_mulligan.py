"""Mulligan-net bootstrap: freeze the DeckNetwork(s) involved and train only
a fresh MulliganNet attached to the trainee's net, via REINFORCE. Two
opponent modes, selected by --opponent-mode (flag or --config):

  twin         Opponent is an INDEPENDENTLY-trained twin league's frozen
               roster (--twin), cycled round-robin. Twin's pregame decider
               is a scripted stand-in (--twin-mulligan), never twin's own
               mulligan.pt. Twin is read-only, enforced structurally: no
               save_deck_checkpoint/save_snapshot/register_snapshot call in
               this file is ever passed a twin path.
  self-mirror  Opponent is the SAME frozen net as the trainee (one deck,
               loaded once, shared by both SeatAgents), so expected in-game
               skill parity is 50/50 by construction; only the pregame
               decider differs (opponent always uses RandomMulligan).
               Single-archetype by construction (deck X only ever faces deck
               X) -- narrower than twin mode; stage on one deck before
               trusting a full-roster result.

Two phases, one script, either mode:

  1. TRAIN. For each --league deck, --train-games games (default depends on
     --opponent-mode: 1000 for twin, 3000 for self-mirror -- each mode's own
     history), against that mode's opponent. Trajectory, not just the
     endpoint: _mulligan_common.build_probe_hands builds eight fixed
     synthetic hands per deck; after every chunk's REINFORCE update,
     probe_p_mulligan reads P(mulligan) back out for each and logs it
     (printed live, and saved under each deck's "probe_trajectory" in
     --out's JSON).
  2. EVAL. For each deck, --eval-games games under three pregame deciders,
     all sharing the same frozen trainee net: RandomMulligan (the floor),
     AlwaysKeep (the cheap heuristic), and the just-trained MulliganNet
     (greedy). Reports win rate per arm, plus -- for the trained arm only --
     a land-count/keep-rate/confidence breakdown reconstructed from the
     event log.

Config file (--config, e.g. training_configs/mulligan_bootstrap_default.json):
JSON of any flag below by its long name with dashes -> underscores
(opponent_mode, league, twin, train_games, ...); a config may itself
"extend" another (see config_loader.load_config, shared with run_league.py).
Flag > config > hardcoded default, same precedence as run_league.py.

Output: checkpoints/<league>/<deck>/<--bootstrap-name>.pt (default
mulligan_bootstrap.pt for twin mode, mulligan_bootstrap_selfmirror.pt for
self-mirror) -- a new filename, never overwrites the real mulligan.pt.
Review the eval numbers before promoting it.

    python -u analysis/mulligan_retrain/train_mulligan.py --opponent-mode twin \\
        --config ../training_configs/mulligan_bootstrap_default.json

    python -u analysis/mulligan_retrain/train_mulligan.py --opponent-mode self-mirror \\
        --decks mono_red_rally --train-games 3000 \\
        --out ../logs/mulligan_selfmirror_mono_red_rally.json

    python -u analysis/mulligan_retrain/train_mulligan.py --opponent-mode twin \\
        --decks mono_red_rally --train-games 1200 --entropy-coef 0.3 \\
        --bootstrap-name mulligan_bootstrap_entropy_0p3 \\
        --out ../logs/mulligan_ablation_entropy_0p3.json

See --help for the full flag list (roster/opponent selection, land-count
stratification, entropy override, resume).
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
from config_loader import load_config  # noqa: E402
from rl import checkpoint as ckpt_io  # noqa: E402
from rl.model import mulligan as mulligan_mod  # noqa: E402 -- so --entropy-coef can patch ENTROPY_COEF
from rl.decision.agent import AlwaysKeep, MulliganZeroLands, RandomMulligan, SeatAgent  # noqa: E402
from rl.league.league_runner import HORIZON, PPO_DEFAULTS  # noqa: E402
from rl.model.mulligan import MulliganNet, update as mulligan_update  # noqa: E402
from rl.roster import build_pool  # noqa: E402
from rl.rewards import deploy_reward_v6  # noqa: E402
from rl.training.train import _constant_pairing, collect_rollout  # noqa: E402
from repo_paths import CHECKPOINTS_DIR  # noqa: E402


def _twin_cycling_pairing(main_agent, main_bucket, main_decklist, reward_fn,
                          twin_agents, twin_decklists, twin_names, main_seat):
    """Cycles through twin's roster round-robin, one opponent per game, so
    even a short batch touches every twin archetype. main_seat pins which
    physical seat carries the trainee deck; on-the-play itself still varies
    via collect_rollout's starting_idx randomization. main_bucket/reward_fn
    =None (the eval-phase call) is correct for record=False, where neither
    is read."""
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


def _twin_pairing_factory(twin_agents, twin_decklists, twin_names):
    """pairing_factory(main_agent, main_decklist, reward_fn, main_bucket,
    main_seat) -> a pairing cycling through twin's roster round-robin. Same
    shape as _self_mirror_pairing_factory below, so _train_mulligan and
    _play_eval_arm don't need to know which opponent mode is in effect."""
    def factory(main_agent, main_decklist, reward_fn, main_bucket, main_seat):
        return _twin_cycling_pairing(main_agent, main_bucket, main_decklist, reward_fn,
                                     twin_agents, twin_decklists, twin_names, main_seat)
    return factory


def _self_mirror_pairing_factory(opponent_agent, opponent_decklist):
    """Same pairing_factory shape as _twin_pairing_factory, against one fixed
    opponent instead of a cycled roster."""
    def factory(main_agent, main_decklist, reward_fn, main_bucket, main_seat):
        agents = [main_agent, opponent_agent] if main_seat == 0 else [opponent_agent, main_agent]
        decklists = [main_decklist, opponent_decklist] if main_seat == 0 else [opponent_decklist, main_decklist]
        rewards = [reward_fn, None] if main_seat == 0 else [None, reward_fn]
        buckets = [main_bucket, None] if main_seat == 0 else [None, main_bucket]
        return _constant_pairing(agents, decklists, rewards, buckets)
    return factory


def _train_mulligan(main_net, main_decklist, main_bucket, deck_ctx, pairing_factory,
                    n_games, update_every, seed, horizon, mulligan_lr, probe_hands,
                    stratify_0land_pct=0.0, stratify_7land_pct=0.0,
                    resume_from=None, games_already=0):
    """Fresh MulliganNet on main_net's encoder, trained by REINFORCE
    (rl.model.mulligan.update) against whatever opponent pairing_factory
    describes, main pinned to seat 0 throughout -- on_the_play still varies
    game to game regardless of which physical seat is fixed, so this doesn't
    bias training; only a win-rate comparison needs the seat swap (eval
    phase). Reward is filled in by collect_rollout itself.

    P(mulligan) on each of probe_hands (from _mulligan_common.build_probe_hands)
    is read back after every chunk's REINFORCE update and appended to the
    returned trajectory.

    stratify_0land_pct, stratify_7land_pct are passed through to
    collect_rollout. resume_from (optional path): continues an earlier run's
    net + optimizer state instead of starting fresh -- needed because
    torch.manual_seed is never called here, so MulliganNet's random init and
    sampling draw from torch's global RNG, uncontrolled by --seed;
    relaunching with a bigger --train-games would otherwise be an
    independent run, not a continuation. games_already is a labeling offset
    only. win_tally still starts fresh at 0/0."""
    mnet = MulliganNet(main_net.encoder)
    opt = torch.optim.Adam([p for p in mnet.parameters() if p.requires_grad], lr=mulligan_lr)
    if resume_from is not None:
        loaded = ckpt_io.load_deck_checkpoint(resume_from, mnet, opt)
        if not loaded:
            raise SystemExit(f"--resume-from {resume_from} does not exist")
    main_agent = SeatAgent(main_net, mnet, deck_ctx)
    pairing = pairing_factory(main_agent, main_decklist, deploy_reward_v6, main_bucket, main_seat=0)
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
        probes = probe_p_mulligan(mnet, probe_hands)
        probe_trajectory.append({"games": played, **probes})
        line = (f"    [{main_bucket}] {played}/{games_already + n_games} games, win_rate={rate:.3f}, "
                f"mulligan n={stats['n']} loss={stats.get('loss', float('nan')):.4f}"
                f" | P(mulligan): " + " ".join(f"{k}={v:.3f}" for k, v in probes.items()))
        print(line, flush=True)
    return mnet, opt, win_tally, probe_trajectory


def _play_eval_arm(agent, main_decklist, pairing_factory, n_games, seed, horizon, game_logs=None):
    """n_games split evenly across both seats for on-the-play fairness.
    record=False, greedy=True throughout. Returns (wins, decided) for
    agent's side; a horizon timeout (state.winner is None) counts toward
    neither."""
    half = n_games // 2
    tally = {"wins": 0, "decided": 0}
    for main_seat in (0, 1):
        def on_game_end(state, main_seat=main_seat):
            if state.winner is None:
                return
            tally["decided"] += 1
            if state.winner == main_seat:
                tally["wins"] += 1
        pairing = pairing_factory(agent, main_decklist, None, None, main_seat)
        collect_rollout(pairing, half, horizon, random.Random(seed + main_seat), device="cpu",
                        record=False, greedy=True, on_game_end=on_game_end, game_logs=game_logs)
    return tally["wins"], tally["decided"]


def _build_arg_parser():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=None, metavar="PATH",
                    help="JSON of any flag below by its long name (dashes -> underscores) -- see "
                         "training_configs/mulligan_bootstrap_default.json. May itself \"extend\" another "
                         "config (config_loader.load_config, shared with run_league.py). "
                         "Flag > config > hardcoded default.")
    ap.add_argument("--opponent-mode", choices=["twin", "self-mirror"], default=None,
                    help="twin: opponent is an independently-trained twin league's roster (see --twin). "
                         "self-mirror: opponent is the SAME frozen net, RandomMulligan pregame. Required "
                         "(flag or --config's own opponent_mode) -- no hardcoded default, since silently "
                         "picking the wrong mode would invalidate the whole run.")
    ap.add_argument("--league", default=None, help="the population being trained (checkpoint dir name). "
                    "Default 4_deck_subleague_test.")
    ap.add_argument("--twin", default=None, help="opponent-mode=twin only: the twin league (checkpoint dir "
                    "name), READ ONLY. Default 4_deck_subleague_gauntlet.")
    ap.add_argument("--train-games", type=int, default=None,
                    help="training games per deck. Default 1000 (twin) / 3000 (self-mirror).")
    ap.add_argument("--update-every", type=int, default=None, help="games per REINFORCE update. Default 50.")
    ap.add_argument("--eval-games", type=int, default=None,
                    help="eval games per policy arm per deck. Default 100.")
    ap.add_argument("--seed", type=int, default=None, help="Default 20260820.")
    ap.add_argument("--out", default=None, help="write full results here as JSON")
    ap.add_argument("--decks", nargs="+", default=None,
                    help="restrict to these deck names to train (default: every deck --league has a live.pt "
                         "for, and -- opponent-mode=twin only -- --twin also has a live.pt for). Twin mode "
                         "also sets the TWIN opponent roster from this unless --twin-decks overrides it "
                         "separately -- see --twin-decks")
    ap.add_argument("--twin-decks", nargs="+", default=None,
                    help="opponent-mode=twin only: restrict the TWIN opponent roster independently of "
                         "--decks (default: same as --decks, i.e. same-archetype-only). Pass all four twin "
                         "decks for an opponent-diverse bootstrap")
    ap.add_argument("--entropy-coef", type=float, default=None,
                    help="override rl.model.mulligan.ENTROPY_COEF for this run only (ablation -- default: "
                         "leave it alone)")
    ap.add_argument("--stratify-0land-pct", type=float, default=None,
                    help="fraction of training games (trainee seat only) whose opening hand is forced to 0 "
                         "lands (game.state.build_shuffled_library's force_land_count, via collect_rollout's "
                         "stratify_0land_pct) -- default 0.0 leaves dealing fully natural")
    ap.add_argument("--stratify-7land-pct", type=float, default=None,
                    help="fraction of training games (trainee seat only) whose opening hand is forced to 7 "
                         "lands (all-land, no spells) -- the other extreme from --stratify-0land-pct. Default 0.0")
    ap.add_argument("--twin-mulligan", choices=["always_keep", "zero_lands"], default=None,
                    help="opponent-mode=twin only: twin's pregame decider (default: always_keep). zero_lands "
                         "(rl.decision.agent.MulliganZeroLands) mulligans a 0-land hand and keeps everything "
                         "else -- removes the twin's own worst-hand losses as a noise source in the trainee "
                         "seat's mulligan reward")
    ap.add_argument("--bootstrap-name", default=None,
                    help="output checkpoint filename stem (no .pt) -- vary this to run two configs against "
                         "the same deck without one overwriting the other's result. Default mulligan_bootstrap "
                         "(twin) / mulligan_bootstrap_selfmirror (self-mirror)")
    ap.add_argument("--resume-from", default=None,
                    help="path to an existing bootstrap .pt (this exact deck) to continue training from, "
                         "loading its actual net+optimizer state, instead of a fresh MulliganNet. Only valid "
                         "with --decks restricted to exactly one deck.")
    ap.add_argument("--resume-games-already", type=int, default=None,
                    help="how many games --resume-from's checkpoint already represents -- a pure labeling "
                         "offset so printed/logged \"games\" and probe_trajectory continue the earlier run's "
                         "numbering instead of restarting at 0. Default 0.")
    return ap


def _resolve_options(args, cfg):
    """Flag > config > hardcoded default precedence for every CLI option,
    plus the two validations (opponent_mode required; resume_from needs a
    single deck) that must hold before any network is touched. Split out of
    main() so this branching is testable without build_pool()/collect_rollout."""
    def resolve(flag_name, cfg_key, default):
        v = getattr(args, flag_name)
        return v if v is not None else cfg.get(cfg_key, default)

    opponent_mode = args.opponent_mode or cfg.get("opponent_mode")
    if opponent_mode not in ("twin", "self-mirror"):
        raise SystemExit("--opponent-mode {twin,self-mirror} is required (flag, or --config's own "
                         "opponent_mode key)")
    is_twin = opponent_mode == "twin"

    opts = {
        "opponent_mode": opponent_mode,
        "is_twin": is_twin,
        "league": resolve("league", "league", "4_deck_subleague_test"),
        "twin": resolve("twin", "twin", "4_deck_subleague_gauntlet"),
        "train_games": resolve("train_games", "train_games", 1000 if is_twin else 3000),
        "update_every": resolve("update_every", "update_every", 50),
        "eval_games": resolve("eval_games", "eval_games", 100),
        "seed": resolve("seed", "seed", 20260820),
        "out": resolve("out", "out", None),
        "decks": args.decks if args.decks is not None else cfg.get("decks"),
        "twin_decks": args.twin_decks if args.twin_decks is not None else cfg.get("twin_decks"),
        "entropy_coef": resolve("entropy_coef", "entropy_coef", None),
        "stratify_0land_pct": resolve("stratify_0land_pct", "stratify_0land_pct", 0.0),
        "stratify_7land_pct": resolve("stratify_7land_pct", "stratify_7land_pct", 0.0),
        "twin_mulligan": resolve("twin_mulligan", "twin_mulligan", "always_keep"),
        "resume_from": resolve("resume_from", "resume_from", None),
        "resume_games_already": resolve("resume_games_already", "resume_games_already", 0),
    }
    opts["bootstrap_name"] = resolve("bootstrap_name", "bootstrap_name",
                                     "mulligan_bootstrap" if is_twin else "mulligan_bootstrap_selfmirror")

    if opts["resume_from"] is not None and (opts["decks"] is None or len(opts["decks"]) != 1):
        raise SystemExit("--resume-from requires --decks restricted to exactly one deck")
    return opts


def main():
    args = _build_arg_parser().parse_args()
    opts = _resolve_options(args, load_config(args.config))
    opponent_mode, is_twin = opts["opponent_mode"], opts["is_twin"]
    league, twin = opts["league"], opts["twin"]
    train_games, update_every = opts["train_games"], opts["update_every"]
    eval_games, seed, out = opts["eval_games"], opts["seed"], opts["out"]
    decks, twin_decks = opts["decks"], opts["twin_decks"]
    entropy_coef = opts["entropy_coef"]
    stratify_0land_pct, stratify_7land_pct = opts["stratify_0land_pct"], opts["stratify_7land_pct"]
    twin_mulligan, bootstrap_name = opts["twin_mulligan"], opts["bootstrap_name"]
    resume_from, resume_games_already = opts["resume_from"], opts["resume_games_already"]

    if entropy_coef is not None:
        print(f"ABLATION: overriding rl.model.mulligan.ENTROPY_COEF {mulligan_mod.ENTROPY_COEF} -> {entropy_coef}")
        mulligan_mod.ENTROPY_COEF = entropy_coef

    decklists, vocab, deck_ctxs, fixed_tables = build_pool()
    league_dir = os.path.join(CHECKPOINTS_DIR, league)

    if is_twin:
        twin_dir = os.path.join(CHECKPOINTS_DIR, twin)  # READ ONLY -- never passed to a save_* call below
        deck_names = sorted(n for n in decklists
                            if os.path.exists(f"{league_dir}/{n}/live.pt") and os.path.exists(f"{twin_dir}/{n}/live.pt"))
        if decks is not None:
            missing = set(decks) - set(deck_names)
            if missing:
                raise SystemExit(f"--decks named {sorted(missing)}, not present (with a live.pt) in BOTH {league} and {twin}")
            deck_names = [n for n in deck_names if n in set(decks)]
        if not deck_names:
            raise SystemExit(f"no deck has a live.pt in BOTH {league} and {twin}")

        twin_deck_names = deck_names  # default: same-archetype-only
        if twin_decks is not None:
            available_twin = sorted(n for n in decklists if os.path.exists(f"{twin_dir}/{n}/live.pt"))
            missing = set(twin_decks) - set(available_twin)
            if missing:
                raise SystemExit(f"--twin-decks named {sorted(missing)}, not present (with a live.pt) in {twin}")
            twin_deck_names = sorted(twin_decks)

        print(f"opponent_mode=twin league={league} twin={twin} (twin read-only) decks={deck_names} "
              f"twin_decks={twin_deck_names}")
    else:
        deck_names = sorted(n for n in decklists if os.path.exists(f"{league_dir}/{n}/live.pt"))
        if decks is not None:
            missing = set(decks) - set(deck_names)
            if missing:
                raise SystemExit(f"--decks named {sorted(missing)}, not present (with a live.pt) in {league}")
            deck_names = [n for n in deck_names if n in set(decks)]
        if not deck_names:
            raise SystemExit(f"no deck has a live.pt in {league}")
        print(f"opponent_mode=self-mirror league={league} (opponent=same net, RandomMulligan) decks={deck_names}")

    print(f"train_games/deck={train_games} update_every={update_every} "
          f"eval_games/arm={eval_games} seed={seed} "
          f"entropy_coef={mulligan_mod.ENTROPY_COEF} stratify_0land_pct={stratify_0land_pct} "
          f"stratify_7land_pct={stratify_7land_pct}"
          + (f" twin_mulligan={twin_mulligan}" if is_twin else "")
          + f" resume_from={resume_from} resume_games_already={resume_games_already}", flush=True)

    nets = load_frozen_nets(league_dir, deck_names, vocab, fixed_tables)
    mulligan_lr = PPO_DEFAULTS["mulligan_lr"]

    if is_twin:
        twin_nets = load_frozen_nets(twin_dir, twin_deck_names, vocab, fixed_tables)
        # One shared decider instance/rng across every twin deck and arm.
        twin_decider = AlwaysKeep() if twin_mulligan == "always_keep" else MulliganZeroLands(random.Random(seed + 3000))
        twin_agents = {n: SeatAgent(twin_nets[n], twin_decider, deck_ctxs[n]) for n in twin_deck_names}

    t0 = time.time()
    results = {"opponent_mode": opponent_mode, "league": league,
               "train_games_per_deck": train_games, "eval_games_per_arm": eval_games,
               "seed": seed, "entropy_coef": mulligan_mod.ENTROPY_COEF,
               "stratify_0land_pct": stratify_0land_pct, "stratify_7land_pct": stratify_7land_pct, "decks": {}}
    if is_twin:
        results.update(twin=twin, twin_mulligan=twin_mulligan, twin_decks=twin_deck_names)

    for name in deck_names:
        if is_twin:
            # Same shared twin_agents/roster for both train and eval, same as
            # the pre-merge script -- twin's pregame decider (twin_decider)
            # already owns its own fixed seed (seed + 3000).
            pairing_factory = _twin_pairing_factory(twin_agents, decklists, twin_deck_names)
            train_pairing_factory = eval_pairing_factory = pairing_factory
            opponent_desc = f"twin roster {twin_deck_names}"
        else:
            # Deliberately DIFFERENT RandomMulligan opponent rng for train vs
            # eval (seed+1000 vs seed+2000), matching the pre-merge script --
            # not a shared opponent between the two phases.
            train_opponent = SeatAgent(nets[name], RandomMulligan(random.Random(seed + 1000)), deck_ctxs[name])
            eval_opponent = SeatAgent(nets[name], RandomMulligan(random.Random(seed + 2000)), deck_ctxs[name])
            train_pairing_factory = _self_mirror_pairing_factory(train_opponent, decklists[name])
            eval_pairing_factory = _self_mirror_pairing_factory(eval_opponent, decklists[name])
            opponent_desc = "self-mirror (RandomMulligan)"

        print(f"\n=== {name}: TRAIN vs {opponent_desc} ({train_games} games) ===", flush=True)
        probe_hands = build_probe_hands(decklists[name], vocab)
        mnet, opt, train_tally, probe_trajectory = _train_mulligan(
            nets[name], decklists[name], name, deck_ctxs[name], train_pairing_factory,
            train_games, update_every, seed, HORIZON, mulligan_lr, probe_hands,
            stratify_0land_pct=stratify_0land_pct, stratify_7land_pct=stratify_7land_pct,
            resume_from=resume_from, games_already=resume_games_already)
        train_rate = train_tally["wins"] / train_tally["decided"] if train_tally["decided"] else float("nan")
        print(f"  training done: {train_tally['wins']}/{train_tally['decided']} = {train_rate:.3f} "
              f"vs {opponent_desc}")

        out_path = f"{league_dir}/{name}/{bootstrap_name}.pt"
        ckpt_io.save_deck_checkpoint(out_path, mnet, opt)
        print(f"  saved {out_path}")

        print(f"  === {name}: EVAL, 3 pregame policies x {eval_games} games vs {opponent_desc} ===", flush=True)
        arms = [
            ("random_mulligan", SeatAgent(nets[name], RandomMulligan(random.Random(seed)), deck_ctxs[name])),
            ("always_keep", SeatAgent(nets[name], AlwaysKeep(), deck_ctxs[name])),
            ("trained_mulligan", SeatAgent(nets[name], mnet, deck_ctxs[name])),
        ]
        deck_result = {"train_win_rate": train_rate, "train_games_decided": train_tally["decided"],
                       "probe_trajectory": probe_trajectory, "arms": {}}
        for arm_name, agent in arms:
            logs = [] if arm_name == "trained_mulligan" else None
            wins, decided = _play_eval_arm(agent, decklists[name], eval_pairing_factory,
                                           eval_games, seed, HORIZON, game_logs=logs)
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
    if out:
        with open(out, "w") as f:
            json.dump(results, f, indent=2)
        print(f"wrote {out}")


if __name__ == "__main__":
    main()
