"""One-time checkpoint migration for the pregame-action removal (the harness
refactor). BEHAVIORALLY LOSSLESS: the two DeckNetwork head rows for "Keep hand" /
"Mulligan" never trained in league (the mulligan model intercepts every pregame
decision before the main net's forward), so dropping them changes no behavior.

MUST run BEFORE the actions are removed from drl_env._actions -- it needs the
CURRENT (pre-removal) fixed table to locate the two rows to slice. Backs up
checkpoints/league/ first and refuses to run twice.

Per file:
  live.pt       -- slice the 2 dead rows from non_targeting_head.{weight,bias};
                   DROP the optimizer state (Adam re-warms in a few steps --
                   simpler and safer than slicing its per-param momentum).
  snapshot_*.pt -- slice the same 2 rows; wrap into the new SeatAgent format by
                   pairing with the deck's CURRENT mulligan.pt (an era-matched
                   mulligan was never saved; current is the best available, and a
                   snapshot is only ever an opponent).
  mulligan.pt   -- untouched (MulliganNet head unaffected).

Run from repo root:  python scripts/migrate_checkpoints.py
"""
import os
import shutil
import sys

import torch

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(_ROOT, "src"))
os.chdir(os.path.join(_ROOT, "src"))  # build_pool reads ../data, ../checkpoints via relative paths

from rl.pool import build_pool  # noqa: E402

LEAGUE_DIR = os.path.join(_ROOT, "checkpoints", "league")
BACKUP_DIR = os.path.join(_ROOT, "checkpoints", "league_backup_preremoval")


def _drop_indices(fixed_table):
    names = [n for n, _l, _e in fixed_table]
    idx = [i for i, n in enumerate(names) if n in ("Keep hand", "Mulligan")]
    assert len(idx) == 2, (
        f"expected exactly 2 pregame actions ('Keep hand', 'Mulligan') in the fixed table, "
        f"found {idx} = {[names[i] for i in idx]}. Are the actions already removed? "
        "Run this BEFORE removing them from drl_env._actions."
    )
    return idx


def _slice_head(state_dict, drop):
    w = state_dict["non_targeting_head.weight"]
    keep = torch.tensor([i for i in range(w.shape[0]) if i not in drop], dtype=torch.long)
    sd = dict(state_dict)
    sd["non_targeting_head.weight"] = w[keep].clone()
    sd["non_targeting_head.bias"] = state_dict["non_targeting_head.bias"][keep].clone()
    return sd


def main():
    decklists, vocab, deck_ctxs, fixed_tables = build_pool()
    if not os.path.isdir(LEAGUE_DIR):
        print(f"no league dir at {LEAGUE_DIR} -- nothing to migrate")
        return
    if os.path.exists(BACKUP_DIR):
        print(f"backup {BACKUP_DIR} already exists -- refusing to overwrite (migration already run?). "
              "Delete the backup to force a re-run.")
        return
    shutil.copytree(LEAGUE_DIR, BACKUP_DIR)
    print(f"backed up {LEAGUE_DIR} -> {BACKUP_DIR}")

    migrated = 0
    for name in sorted(os.listdir(LEAGUE_DIR)):
        deck_dir = os.path.join(LEAGUE_DIR, name)
        if not os.path.isdir(deck_dir) or name not in fixed_tables:
            continue
        drop = _drop_indices(fixed_tables[name])
        old_n = len(fixed_tables[name])

        mull_sd = mull_hidden = None
        mull_path = os.path.join(deck_dir, "mulligan.pt")
        if os.path.exists(mull_path):
            mck = torch.load(mull_path, weights_only=True)
            mull_sd = mck["net"]
            mull_hidden = mull_sd["trunk.0.weight"].shape[0]  # first trunk Linear's out_features

        for fn in sorted(os.listdir(deck_dir)):
            fp = os.path.join(deck_dir, fn)
            if fn == "live.pt":
                ck = torch.load(fp, weights_only=True)
                assert ck["net"]["non_targeting_head.weight"].shape[0] == old_n, (
                    f"{name}/live.pt head {ck['net']['non_targeting_head.weight'].shape[0]} != table size {old_n}"
                )
                sd = _slice_head(ck["net"], drop)
                assert sd["non_targeting_head.weight"].shape[0] == old_n - 2
                torch.save({"net": sd}, fp)  # optimizer dropped -> fresh Adam on resume
            elif fn.startswith("snapshot_") and fn.endswith(".pt"):
                ck = torch.load(fp, weights_only=True)
                sd = _slice_head(ck["state_dict"], drop)
                out = {"state_dict": sd, "trunk_hidden": ck["trunk_hidden"]}
                if mull_sd is not None:
                    out["mulligan_state_dict"] = mull_sd
                    out["mulligan_hidden"] = mull_hidden
                torch.save(out, fp)
        migrated += 1
        print(f"  migrated {name}: head {old_n} -> {old_n - 2} (dropped rows {drop}); "
              f"snapshots wrapped with {'current mulligan' if mull_sd is not None else 'AlwaysKeep (no mulligan.pt)'}")
    print(f"migration done: {migrated} decks. Backup at {BACKUP_DIR}.")


if __name__ == "__main__":
    main()
