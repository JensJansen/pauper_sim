"""Publishing a run's data to the replay web app (the src/webapp git submodule).

  mirror.py  best-effort copies of what a run writes (checks/*.json -> consolidated jsonl, metrics lines,
             progress, the run manifest, review replays) into src/webapp/logs/; called by the writers themselves
  sync.py    after each validation tick: prune each league to its newest 10 sets, commit and push the data to the
             replay repo, bump the submodule pointer

Everything here is a no-op unless src/webapp is a checked-out submodule and the league dir is exactly
checkpoints/<name>, and never raises into training.
"""
