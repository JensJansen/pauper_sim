"""Best-response diagnostic: how much better than the league's own agent for DECK can a fresh agent get against the
league's FROZEN final field?

Copies <source league>'s live nets for every deck EXCEPT `deck` into checkpoints/<out>/ (no snapshots, no history),
then trains a from-scratch DECK agent (net + mulligan net) against that field only: the other decks never update,
and checkpoint_opponent_rate is 0, so every opponent is the frozen final net (DECK's own mirror opponent is the
fresh agent itself). Same run mechanics as the source league (ppo hyper-parameters, games_per_iteration, snapshot
cadence, n_workers) taken from the source league's config. Every snapshot of the fresh agent is kept, so
strength-vs-age against the frozen field can be read afterwards; compare with the source league's own DECK agent
against the same field (evaluation.play_matchups with AgentRef(deck, <league_dir>)).

Not a faithful reproduction of the league's training conditions: the fresh agent never sees the old DECK agent,
and the field does not adapt to it. That is the point -- a fixed field isolates "can this deck be played better".

Usage:
  python analysis/eval/run_best_response.py --deck elves [--source small_baseline] [--games 50000]
                  [--config ../training_configs/small_baseline.json] [--out br_elves]
"""
import argparse
import shutil
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from config_loader import load_config
from repo_paths import CHECKPOINTS_DIR
from rl.league.league import PFSP_POWER
from rl.league.league_runner import TRUNK_HIDDEN, _run_session, advance_progress
from run_league import _resolve_device


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--deck", required=True)
    ap.add_argument("--source", default="small_baseline")
    ap.add_argument("--config", default="../training_configs/small_baseline.json")
    ap.add_argument("--games", type=int, default=50000, help="games/deck to train the fresh agent")
    ap.add_argument("--out", default=None, help="league dir name under checkpoints/ (default br_<deck>)")
    args = ap.parse_args()

    cfg = load_config(args.config)
    roster = cfg["roster"]
    assert args.deck in roster, f"{args.deck} not in roster {roster}"
    out = CHECKPOINTS_DIR / (args.out or f"br_{args.deck}")
    assert not out.exists() or not any(out.iterdir()), f"{out} already exists; delete it to rerun"
    for d in roster:
        if d != args.deck:
            (out / d).mkdir(parents=True, exist_ok=True)
            for f in ("live.pt", "mulligan.pt"):
                shutil.copy2(CHECKPOINTS_DIR / args.source / d / f, out / d / f)
    shutil.copy2(CHECKPOINTS_DIR / args.source / "session.txt", out / "session.txt") if (
        CHECKPOINTS_DIR / args.source / "session.txt").exists() else None

    gpi = cfg.get("games_per_iteration") or cfg.get("n_workers", 6)
    n_workers = cfg.get("n_workers", 6)
    n_iterations = args.games // gpi
    kwargs = {
        "train_decks": [args.deck], "roster": roster, "league_dir": out, "checkpoint_rate": 0.0,
        "pfsp": cfg.get("pfsp", True), "cumulative_games": 0, "ppo_hparams": cfg.get("ppo"),
        "pfsp_power": cfg.get("pfsp_power", PFSP_POWER), "trunk_hidden": tuple(cfg.get("trunk_hidden", TRUNK_HIDDEN)),
        "device": _resolve_device(cfg.get("device", "cpu")), "overlap_collect": cfg.get("overlap_collect", False),
    }
    snapshot_every = max(1, cfg["snapshot_every_games"] // gpi) if "snapshot_every_games" in cfg else 20
    print(f"best response: {args.deck} fresh vs frozen {args.source} field -> {out}; {n_iterations} iterations x {gpi} games",
          flush=True)
    with ProcessPoolExecutor(max_workers=n_workers) as executor:
        _run_session(n_iterations, gpi, snapshot_every, executor, n_workers, **kwargs)
    advance_progress(out, n_iterations, gpi, False, session_start_games=0)
    print("best response done", flush=True)


if __name__ == "__main__":
    main()
