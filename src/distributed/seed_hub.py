"""One-shot, explicit seeding of the hub's checkpoint tree from an existing
trained league -- replaces hand-copying files into checkpoints/<hub-league>/.

    python -m distributed.seed_hub --from-league main-league \\
        --to-league main-league-distributed \\
        --ownership-pool ../distributed_configs/main_league_ownership.json

For every deck in the ownership pool it copies the deck's sync surface
(live.pt, mulligan.pt, the ACTIVE snapshot_*.pt files, opponent_stats.json --
never archive/ or checks/, same exclusion as hub_store) and records the deck's
starting cumulative_games in the hub's runtime state, so a registering node
seeds its progress.json (hyperparameter schedule, snapshot cadence) from the
deck's real history instead of 0. The progress value is the source league's
progress.json (one counter shared by every deck in a league trained by
run_training_pipeline.py -- i.e. correct for every deck seeded from it);
--games overrides it.

Run it with the hub STOPPED: a running hub holds its state in memory and would
overwrite what this writes. It refuses to touch a deck that already has a
live.pt in the hub tree, or a runtime state that already has claims, unless
--force (which wipes the pool decks' synced files and all claims/node mappings,
invalidating every node's token).
"""
import argparse
import json
import os
import shutil

from . import hub_ownership, hub_store


def seed_hub(source_league_dir, hub_league_dir, deck_pool, state, games=None, force=False):
    """Returns {deck: games}. `state` is a hub_ownership.HubRuntimeState."""
    if games is None:
        with open(os.path.join(source_league_dir, "progress.json")) as f:
            games = json.load(f)["cumulative_games_per_deck"]
    for deck in deck_pool:
        if not os.path.exists(os.path.join(source_league_dir, deck, "live.pt")):
            raise SystemExit(f"source league has no live.pt for {deck!r}: {source_league_dir}")
    if not force:
        existing = [d for d in deck_pool if os.path.exists(os.path.join(hub_league_dir, d, "live.pt"))]
        if existing or state.claims:
            raise SystemExit(f"hub tree/state already populated (decks with live.pt: {existing}, claims: "
                             f"{sorted(state.claims)}) -- pass --force to wipe and reseed")
    for deck in deck_pool:
        src, dst = os.path.join(source_league_dir, deck), os.path.join(hub_league_dir, deck)
        os.makedirs(dst, exist_ok=True)
        for fn in os.listdir(dst):  # force reseed: no stale snapshots from a previous seed may linger
            if hub_store._is_deck_file(fn):
                os.remove(os.path.join(dst, fn))
        for fn in os.listdir(src):
            if os.path.isfile(os.path.join(src, fn)) and hub_store._is_deck_file(fn):
                tmp = os.path.join(dst, fn + ".tmp")
                shutil.copyfile(os.path.join(src, fn), tmp)
                os.replace(tmp, os.path.join(dst, fn))
    hub_ownership.seed_hub_state(state, {deck: games for deck in deck_pool})
    return {deck: games for deck in deck_pool}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--from-league", required=True, help="e.g. main-league (a checkpoints/<name> directory)")
    ap.add_argument("--to-league", required=True, help="the hub's checkpoints/<name> directory")
    ap.add_argument("--ownership-pool", required=True)
    ap.add_argument("--runtime-state", default="../distributed_configs/hub_runtime_state.json")
    ap.add_argument("--games", type=int, default=None,
                    help="Override the starting cumulative_games (default: the source league's progress.json).")
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    from repo_paths import CHECKPOINTS_DIR
    pool = hub_ownership.load_deck_pool(args.ownership_pool)
    state = hub_ownership.HubRuntimeState(args.runtime_state)
    seeded = seed_hub(str(CHECKPOINTS_DIR / args.from_league), str(CHECKPOINTS_DIR / args.to_league),
                      pool, state, games=args.games, force=args.force)
    print(f"seeded {len(seeded)} decks at {next(iter(seeded.values()))} cumulative games each")


if __name__ == "__main__":
    main()
