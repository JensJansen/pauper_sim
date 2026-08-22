"""Mulligan-net bootstrap #2: same-strength opponent, deconfounded from
in-game skill. Both seats run the SAME frozen DeckNetwork (one deck, loaded
once, shared by both SeatAgents), so expected in-game skill parity is 50/50
by construction. Only the pregame decider differs:

    training seat:  <deck>'s frozen live.pt  +  live, training MulliganNet
    opponent seat:  <deck>'s SAME live.pt    +  RandomMulligan

Scope, narrower than train_mulligan_vs_twin.py: single-archetype by
construction (deck X only ever faces deck X), so this is not a production
training regime -- --decks defaults to every deck the league has, but stage
on one deck first (e.g. mono_red_rally) before trusting a full-roster result.

Same two phases, same eval/audit machinery as train_mulligan_vs_twin.py
(shared via _mulligan_common.py): TRAIN (--train-games, default 3000), then
EVAL under each of RandomMulligan / AlwaysKeep / the just-trained MulliganNet
(greedy), seat-swapped 50/50. Land-count audit on the trained arm only.

Output: checkpoints/<league>/<deck>/<--bootstrap-name>.pt (default
mulligan_bootstrap_selfmirror.pt) -- never overwrites the real mulligan.pt or
either twin-based bootstrap file.

    python -u analysis/mulligan_retrain/train_mulligan_self_mirror.py --decks mono_red_rally \
        --out ../logs/mulligan_selfmirror_mono_red_rally.json
"""
import argparse
import json
import os
import random
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import torch  # noqa: E402

from _mulligan_common import audit_land_counts, load_frozen_nets, print_land_audit  # noqa: E402
from rl import checkpoint as ckpt_io  # noqa: E402
from rl.decision.agent import AlwaysKeep, RandomMulligan, SeatAgent  # noqa: E402
from rl.league.league_runner import HORIZON, PPO_DEFAULTS  # noqa: E402
from rl.model.mulligan import MulliganNet, update as mulligan_update  # noqa: E402
from rl.roster import build_pool  # noqa: E402
from rl.rewards import deploy_reward_v6  # noqa: E402
from rl.training.train import _constant_pairing, collect_rollout  # noqa: E402
from repo_paths import CHECKPOINTS_DIR  # noqa: E402


def _train_mulligan(net, decklist, bucket, deck_ctx, opponent_seed,
                    n_games, update_every, seed, horizon, mulligan_lr):
    """Fresh MulliganNet on net's own encoder, trained by REINFORCE against a
    SeatAgent wrapping the SAME net object with RandomMulligan as its
    pregame decider -- one fixed opponent, so _constant_pairing is enough.
    Main pinned to seat 0 throughout training; on_the_play already varies
    game to game via collect_rollout's starting_idx, so fixing the seat
    doesn't bias what the net learns. The eval phase handles the seat swap
    needed for a win-rate comparison."""
    mnet = MulliganNet(net.encoder)
    opt = torch.optim.Adam([p for p in mnet.parameters() if p.requires_grad], lr=mulligan_lr)
    main_agent = SeatAgent(net, mnet, deck_ctx)
    opponent_agent = SeatAgent(net, RandomMulligan(random.Random(opponent_seed)), deck_ctx)
    pairing = _constant_pairing([main_agent, opponent_agent], [decklist, decklist],
                                [deploy_reward_v6, None], [bucket, None])
    rng = random.Random(seed)
    win_tally = {"wins": 0, "decided": 0}

    def on_game_end(state):
        if state.winner is None:
            return
        win_tally["decided"] += 1
        if state.winner == 0:
            win_tally["wins"] += 1

    played = 0
    n_chunks = max(1, n_games // update_every)
    for chunk in range(n_chunks):
        chunk_games = update_every if chunk < n_chunks - 1 else n_games - played
        _buffers, mull_by_deck, n_played = collect_rollout(
            pairing, chunk_games, horizon, rng, device="cpu",
            record=True, greedy=False, on_game_end=on_game_end)
        played += n_played
        transitions = mull_by_deck.get(bucket, [])
        stats = mulligan_update(mnet, opt, transitions)
        rate = win_tally["wins"] / win_tally["decided"] if win_tally["decided"] else float("nan")
        print(f"    [{bucket}] {played}/{n_games} games, win_rate={rate:.3f} (vs self-mirror+RandomMulligan), "
              f"mulligan n={stats['n']} loss={stats.get('loss', float('nan')):.4f}", flush=True)
    return mnet, opt, win_tally


def _play_eval_arm(agent, opponent_agent, decklist, n_games, seed, horizon, game_logs=None):
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
        agents = [agent, opponent_agent] if main_seat == 0 else [opponent_agent, agent]
        pairing = _constant_pairing(agents, [decklist, decklist], [None, None], [None, None])
        collect_rollout(pairing, half, horizon, random.Random(seed + main_seat), device="cpu",
                        record=False, greedy=True, on_game_end=on_game_end, game_logs=game_logs)
    return tally["wins"], tally["decided"]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--league", default="4_deck_subleague_test", help="league checkpoint dir name")
    ap.add_argument("--decks", nargs="+", default=None,
                    help="restrict to these deck names (default: every deck the league has a live.pt for)")
    ap.add_argument("--train-games", type=int, default=3000, help="training games per deck")
    ap.add_argument("--update-every", type=int, default=50, help="games per REINFORCE update")
    ap.add_argument("--eval-games", type=int, default=100, help="eval games per policy arm per deck")
    ap.add_argument("--seed", type=int, default=20260820)
    ap.add_argument("--out", default=None, help="write full results here as JSON")
    ap.add_argument("--bootstrap-name", default="mulligan_bootstrap_selfmirror",
                    help="output checkpoint filename stem (no .pt)")
    args = ap.parse_args()

    decklists, vocab, deck_ctxs, fixed_tables = build_pool()
    league_dir = os.path.join(CHECKPOINTS_DIR, args.league)

    deck_names = sorted(n for n in decklists if os.path.exists(f"{league_dir}/{n}/live.pt"))
    if args.decks is not None:
        missing = set(args.decks) - set(deck_names)
        if missing:
            raise SystemExit(f"--decks named {sorted(missing)}, not present (with a live.pt) in {args.league}")
        deck_names = [n for n in deck_names if n in set(args.decks)]
    if not deck_names:
        raise SystemExit(f"no deck has a live.pt in {args.league}")

    print(f"league={args.league} (self-mirror, opponent=RandomMulligan) decks={deck_names}")
    print(f"train_games/deck={args.train_games} update_every={args.update_every} "
          f"eval_games/arm={args.eval_games} seed={args.seed}", flush=True)

    nets = load_frozen_nets(league_dir, deck_names, vocab, fixed_tables)
    mulligan_lr = PPO_DEFAULTS["mulligan_lr"]

    t0 = time.time()
    results = {"league": args.league, "opponent": "self-mirror + RandomMulligan",
               "train_games_per_deck": args.train_games, "eval_games_per_arm": args.eval_games,
               "seed": args.seed, "decks": {}}

    for name in deck_names:
        print(f"\n=== {name}: TRAIN vs same-strength self-mirror ({args.train_games} games) ===", flush=True)
        mnet, opt, train_tally = _train_mulligan(
            nets[name], decklists[name], name, deck_ctxs[name], args.seed + 1000,
            args.train_games, args.update_every, args.seed, HORIZON, mulligan_lr)
        train_rate = train_tally["wins"] / train_tally["decided"] if train_tally["decided"] else float("nan")
        print(f"  training done: {train_tally['wins']}/{train_tally['decided']} = {train_rate:.3f} "
              f"vs self-mirror (RandomMulligan) -- expect ~0.5 before mulligan skill diverges")

        out_path = f"{league_dir}/{name}/{args.bootstrap_name}.pt"
        ckpt_io.save_deck_checkpoint(out_path, mnet, opt)
        print(f"  saved {out_path}")

        print(f"  === {name}: EVAL, 3 pregame policies x {args.eval_games} games "
              f"vs self-mirror (RandomMulligan) ===", flush=True)
        arms = [
            ("random_mulligan", SeatAgent(nets[name], RandomMulligan(random.Random(args.seed)), deck_ctxs[name])),
            ("always_keep", SeatAgent(nets[name], AlwaysKeep(), deck_ctxs[name])),
            ("trained_mulligan", SeatAgent(nets[name], mnet, deck_ctxs[name])),
        ]
        # One opponent per deck, shared across that deck's three eval arms.
        opponent_agent = SeatAgent(nets[name], RandomMulligan(random.Random(args.seed + 2000)), deck_ctxs[name])
        deck_result = {"train_win_rate": train_rate, "train_games_decided": train_tally["decided"], "arms": {}}
        for arm_name, agent in arms:
            logs = [] if arm_name == "trained_mulligan" else None
            wins, decided = _play_eval_arm(agent, opponent_agent, decklists[name],
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
