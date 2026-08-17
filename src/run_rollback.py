"""Promote a historical snapshot back to a deck's LIVE checkpoint.

WHY THIS EXISTS. Training can make a policy worse, and did: at 60,001
games/deck, three of four decks in 4_deck_subleague_test lost head-to-head to
their own ~200-game-old selves (dmir_terror 38%, elves 43%,
rakdos_madness 35% against snapshot_0, measured by a snapshot round
robin). Continuing to train from `live.pt` in that state means
building on the worst version of the policy the run ever produced.

NO NEW BACKUP IS NEEDED and this deliberately does not make one of the pool.
`LeaguePool.register_snapshot` MOVES evictions into `<deck>/archive/` rather
than deleting them, so every snapshot a run ever took is already on disk --
290 per deck here, one per 200 games, ~1MB each. The rollback material exists;
what was missing was a way to USE it. That is all this script is.

What it does per deck: reconstruct the DeckNetwork (and its era-matched
MulliganNet) from the snapshot and write them as live.pt / mulligan.pt.

  - Optimizer state is deliberately NOT restored -- snapshots never carried
    any. load_deck_checkpoint's migration guard re-warms a fresh Adam, which
    is the correct behavior for a policy that is being rewound anyway; the
    stale moment/variance estimates belonged to a trajectory being abandoned.
  - The CURRENT live.pt/mulligan.pt are copied aside first (see --force), so a
    rollback is itself reversible.
  - session.txt and progress.json are untouched: the session counter is a log
    of what has run, not a claim about the weights, and rewriting history there
    would make metrics.jsonl unreadable.

Usage:
  python run_rollback.py --league 4_deck_subleague_test --deck elves --snapshot 58
  python run_rollback.py --league 4_deck_subleague_test \\
      --set elves=58,rakdos_madness=232,dmir_terror=116 --dry-run
"""
import argparse
import os
import shutil

from repo_paths import CHECKPOINTS_DIR
from rl.mulligan import MulliganNet
from rl.pool import build_pool
from rl.league_runner import build_deck_net, league_roster
from rl import checkpoint as ckpt_io

BACKUP_SUFFIX = ".before_rollback"


def _snapshot_path(deck_dir, snapshot_id):
    """Snapshots live in the active pool dir until evicted, then under
    archive/ -- try both, same as league_runner.load_vintage_agent."""
    for sub in ("", "archive"):
        path = os.path.join(deck_dir, sub, f"snapshot_{snapshot_id}.pt")
        if os.path.exists(path):
            return path
    raise FileNotFoundError(f"no snapshot_{snapshot_id}.pt under {deck_dir} or its archive/")


def rollback_deck(league_dir, deck, snapshot_id, deck_ctx, force=False, dry_run=False):
    """Returns a one-line description of what was (or would be) done."""
    vocab, fixed_table = deck_ctx
    deck_dir = os.path.join(league_dir, deck)
    src = _snapshot_path(deck_dir, snapshot_id)
    live_path = os.path.join(deck_dir, "live.pt")
    mull_path = os.path.join(deck_dir, "mulligan.pt")
    if dry_run:
        return f"{deck}: would restore {os.path.relpath(src, league_dir)} -> live.pt"

    # Back up the current live pair first. Refuse to clobber an existing backup
    # unless --force: a second rollback would otherwise destroy the only copy of
    # the pre-rollback state, which is exactly the mistake this guards against.
    for path in (live_path, mull_path):
        backup = path + BACKUP_SUFFIX
        if os.path.exists(path):
            if os.path.exists(backup) and not force:
                raise FileExistsError(
                    f"{backup} already exists -- a previous rollback's backup would be overwritten. "
                    f"Move or delete it, or pass --force if you are sure.")
            shutil.copy2(path, backup)

    saved = ckpt_io.load_snapshot(src)
    net = build_deck_net(vocab.size, len(fixed_table), saved["trunk_hidden"])
    net.load_state_dict(saved["state_dict"])
    ckpt_io.save_deck_checkpoint(live_path, net)  # optimizer=None -> re-warms fresh

    note = ""
    if "mulligan_state_dict" in saved:
        mull = MulliganNet(net.encoder, hidden=saved.get("mulligan_hidden", 64))
        mull.load_state_dict(saved["mulligan_state_dict"])
        ckpt_io.save_deck_checkpoint(mull_path, mull)
    else:
        # A deck-only snapshot carries no pregame policy. Leaving the CURRENT
        # mulligan.pt in place would pair a rewound main policy with a mulligan
        # net from the abandoned trajectory, so say so rather than do it quietly.
        note = "  (!! snapshot has no mulligan state -- mulligan.pt left at its current version)"
    return f"{deck}: restored snapshot_{snapshot_id} -> live.pt (backup at live.pt{BACKUP_SUFFIX}){note}"


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--league", default="4_deck_subleague_test")
    p.add_argument("--deck", help="single deck to roll back (with --snapshot)")
    p.add_argument("--snapshot", type=int, help="snapshot id for --deck")
    p.add_argument("--set", dest="pairs",
                   help="comma-separated deck=snapshot pairs, e.g. elves=58,rakdos_madness=232")
    p.add_argument("--force", action="store_true", help="overwrite an existing rollback backup")
    p.add_argument("--dry-run", action="store_true", help="print what would happen, change nothing")
    args = p.parse_args()

    if args.pairs:
        targets = {}
        for pair in args.pairs.split(","):
            name, _, sid = pair.partition("=")
            assert sid, f"malformed --set entry {pair!r}, expected deck=snapshot"
            targets[name.strip()] = int(sid)
    else:
        assert args.deck and args.snapshot is not None, "pass --deck with --snapshot, or --set"
        targets = {args.deck: args.snapshot}

    league_dir = str(CHECKPOINTS_DIR / args.league)
    roster = league_roster(league_dir)
    unknown = set(targets) - set(roster)
    assert not unknown, f"deck(s) {sorted(unknown)} not in {args.league} (has {roster})"

    decklists, vocab, deck_ctxs, _fixed = build_pool()
    print(f"{'DRY RUN: ' if args.dry_run else ''}rollback in {args.league}\n")
    for deck, sid in sorted(targets.items()):
        print(" ", rollback_deck(league_dir, deck, sid, deck_ctxs[deck],
                                 force=args.force, dry_run=args.dry_run))
    if not args.dry_run:
        print(f"\nOptimizer state intentionally not restored (re-warms fresh). "
              f"To undo: move each live.pt{BACKUP_SUFFIX} back over live.pt.")


if __name__ == "__main__":
    main()
