"""Keeps the replay repo (the src/webapp submodule) current with the latest
training output, automatically: after every validation cadence tick it prunes
each league down to its newest KEEP_SETS sets, commits ONLY the data
(logs/replays/ and every logs/validation/<league>/ -- never viewer code, never
logs/validation/README.md), pushes to the replay repo's `main`, keeps the local
checkout on the latest `main`, and bumps this repo's submodule pointer.

A "set" is one cadence tick (cumulative_games = N) for one league: that tick's
entry in every consolidated checks/<check>.jsonl (league-level and per-deck) plus
its replay `logs/replays/<league>/round_robin_review_<N>games.json`. Retention:
  - replays and checks: the newest KEEP_SETS N per league. Pruning only touches
    the webapp copy -- checkpoints/ keeps everything, and git history keeps
    whatever was ever committed.
  - metrics.jsonl and progress.json: never pruned (the continuous training
    curves; owner decision 2026-10-17).
  - anything else in logs/replays/ (hand-made files, e.g. cross_league_50k.json)
    is never touched.
Legacy flat replays (`logs/replays/<league>_round_robin_review_<N>games.json`)
are moved into their league folder the first time sync runs.

Every replay also gets a `<replay>.json.gameindex` sidecar (byte range + label info
per game), built here before the commit and pruned with its replay, so the replay
viewer can read one game without parsing the whole ~30 MB file (webapp/replay_index.py).

Best-effort like webapp_mirror: sync() never raises (a git/network problem is
printed and retried at the next tick), and sync_async() runs it on a background
thread so training doesn't wait on a push. Git never prompts for credentials
(GIT_TERMINAL_PROMPT=0) and every git call has a timeout. The thread is
non-daemon, so a final sync finishes before the process exits.

Conflicts: the normal case is a plain push. Only if the push is rejected
(someone else pushed to the replay repo's main) it fetches and rebases our
commit on origin/main (--autostash, so uncommitted viewer-code edits survive); a
rebase conflict is aborted and reported, never forced.

The parent-repo pointer bump is a LOCAL commit restricted to src/webapp on
whatever branch is checked out; it is not pushed (PUSH_PARENT) so this never
pushes unrelated unpushed work.
"""
import importlib.util
import json
import os
import re
import subprocess
import threading
from pathlib import Path

import webapp_mirror
from repo_paths import REPO_ROOT

KEEP_SETS = 10
GIT_TIMEOUT_SECONDS = 300
PUSH_PARENT = False
BRANCH = "main"

_FLAT_REVIEW_RE = re.compile(r"^(?P<league>.+)_round_robin_review_(?P<n>\d+)games\.json$")
_REVIEW_RE = re.compile(r"^round_robin_review_(?P<n>\d+)games\.json$")
SIDECAR_SUFFIX = ".gameindex"  # keep in sync with webapp/replay_index.SIDECAR_SUFFIX


# ---- retention (plain filesystem, no git) ----

def migrate_flat_replays(replays_dir):
    """Moves legacy `<league>_round_robin_review_<N>games.json` files into
    `<league>/round_robin_review_<N>games.json`. Returns how many moved."""
    replays_dir = Path(replays_dir)
    moved = 0
    if not replays_dir.is_dir():
        return 0
    for path in list(replays_dir.glob("*.json")):
        m = _FLAT_REVIEW_RE.match(path.name)
        if not m:
            continue
        dest = replays_dir / m["league"] / f"round_robin_review_{m['n']}games.json"
        dest.parent.mkdir(parents=True, exist_ok=True)
        os.replace(path, dest)
        moved += 1
    return moved


def prune_replays(replays_dir, keep=KEEP_SETS):
    """Per league folder, deletes all but the newest `keep` review replays (by
    N, not mtime). Returns the removed paths."""
    removed = []
    replays_dir = Path(replays_dir)
    if not replays_dir.is_dir():
        return removed
    for league_dir in (d for d in replays_dir.iterdir() if d.is_dir()):
        files = sorted(((int(m["n"]), p) for p in league_dir.iterdir() if (m := _REVIEW_RE.match(p.name))),
                       key=lambda t: t[0], reverse=True)
        for _n, path in files[keep:]:
            path.unlink()
            Path(str(path) + SIDECAR_SUFFIX).unlink(missing_ok=True)
            removed.append(path)
    return removed


def build_sidecars(webapp_dir):
    """Builds/refreshes the per-game byte-offset index (`<replay>.json.gameindex`) of
    every replay under logs/replays/ that lacks a valid one, and drops orphaned ones,
    so the viewer can serve one game without parsing a ~30 MB file. The indexer is the
    replay repo's own replay_index.py, loaded by path (it is stdlib-only and imports
    nothing from either repo); a submodule checkout too old to have it is skipped.
    Returns (built, orphans_removed)."""
    module_path = Path(webapp_dir) / "replay_index.py"
    if not module_path.is_file():
        return 0, 0
    spec = importlib.util.spec_from_file_location("_webapp_replay_index", module_path)
    indexer = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(indexer)
    return indexer.ensure_sidecars(Path(webapp_dir) / "logs" / "replays")


def prune_checks(league_dir, keep=KEEP_SETS):
    """Rewrites every checks/<check>.jsonl under league_dir (league-level and
    per-deck) to hold only its newest `keep` distinct cumulative_games, last
    line per N winning (the same rule the stats API reads with). Lines with no
    cumulative_games are kept; an unparseable (truncated) line is dropped.
    Returns the number of files rewritten. Caller holds webapp_mirror.WRITE_LOCK."""
    rewritten = 0
    for jsonl in Path(league_dir).rglob("*.jsonl"):
        if jsonl.parent.name != "checks":
            continue
        records = []  # (n or None, raw line)
        raw_lines = 0
        with open(jsonl) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                raw_lines += 1
                try:
                    n = json.loads(line).get("cumulative_games")
                except (json.JSONDecodeError, AttributeError):
                    continue
                records.append((n, line))
        newest = sorted({n for n, _ in records if n is not None}, reverse=True)[:keep]
        last_index = {}
        for i, (n, _line) in enumerate(records):
            if n is not None:
                last_index[n] = i
        kept = [line for i, (n, line) in enumerate(records)
                if n is None or (n in newest and last_index[n] == i)]
        if len(kept) != raw_lines:
            tmp = jsonl.with_name(jsonl.name + ".tmp")
            tmp.write_text("\n".join(kept) + ("\n" if kept else ""))
            os.replace(tmp, jsonl)
            rewritten += 1
    return rewritten


def prune_all(webapp_dir, keep=KEEP_SETS):
    webapp_dir = Path(webapp_dir)
    migrate_flat_replays(webapp_dir / "logs" / "replays")
    removed = prune_replays(webapp_dir / "logs" / "replays", keep)
    rewritten = 0
    validation_dir = webapp_dir / "logs" / "validation"
    if validation_dir.is_dir():
        with webapp_mirror.WRITE_LOCK:
            for league_dir in (d for d in validation_dir.iterdir() if d.is_dir()):
                rewritten += prune_checks(league_dir, keep)
    return removed, rewritten


# ---- git ----

def _git(cwd, *args, check=False):
    env = dict(os.environ, GIT_TERMINAL_PROMPT="0")
    result = subprocess.run(["git", "-C", str(cwd), *args], capture_output=True, text=True, env=env,
                            timeout=GIT_TIMEOUT_SECONDS)
    if check and result.returncode != 0:
        raise RuntimeError(f"git {' '.join(args)} failed ({result.returncode}): {result.stderr.strip()}")
    return result


def _data_pathspecs(webapp_dir):
    """logs/replays plus each logs/validation/<league> -- NOT logs/validation/README.md
    or anything outside logs/, so viewer code and docs are never auto-committed."""
    specs = ["logs/replays"] if (Path(webapp_dir) / "logs" / "replays").is_dir() else []
    validation_dir = Path(webapp_dir) / "logs" / "validation"
    if validation_dir.is_dir():
        specs += [f"logs/validation/{d.name}" for d in sorted(validation_dir.iterdir()) if d.is_dir()]
    return specs


def commit_and_push(webapp_dir, message, branch=BRANCH):
    """Returns {"committed": bool, "pushed": bool, "head": short sha | None, "error": str | None}."""
    out = {"committed": False, "pushed": False, "head": None, "error": None}
    try:
        _git(webapp_dir, "fetch", "-q", "origin", branch)  # failure is fine: we just commit locally
        current = _git(webapp_dir, "rev-parse", "--abbrev-ref", "HEAD", check=True).stdout.strip()
        if current != branch:
            _git(webapp_dir, "checkout", "-q", branch, check=True)

        specs = _data_pathspecs(webapp_dir)
        if specs:
            with webapp_mirror.WRITE_LOCK:  # no half-appended line gets snapshotted
                _git(webapp_dir, "add", "-A", "--", *specs, check=True)
            if _git(webapp_dir, "diff", "--cached", "--quiet", "--", *specs).returncode != 0:
                _git(webapp_dir, "commit", "-q", "-m", message, "--", *specs, check=True)
                out["committed"] = True

        remote = f"origin/{branch}"
        behind = _git(webapp_dir, "rev-list", "--count", f"{branch}..{remote}")
        if behind.returncode == 0 and int(behind.stdout.strip() or 0) > 0:
            with webapp_mirror.WRITE_LOCK:
                rebase = _git(webapp_dir, "rebase", "--autostash", remote)
                if rebase.returncode != 0:
                    _git(webapp_dir, "rebase", "--abort")
                    raise RuntimeError(f"rebase onto {remote} failed, left unpushed: {rebase.stderr.strip()}")

        ahead = _git(webapp_dir, "rev-list", "--count", f"{remote}..{branch}")
        if ahead.returncode != 0 or int(ahead.stdout.strip() or 0) > 0:
            _git(webapp_dir, "push", "-q", "origin", branch, check=True)
            out["pushed"] = True
        out["head"] = _git(webapp_dir, "rev-parse", "--short", "HEAD", check=True).stdout.strip()
    except (RuntimeError, OSError, subprocess.TimeoutExpired) as e:
        out["error"] = str(e)
    return out


def bump_parent_pointer(repo_root, webapp_rel, message, push=PUSH_PARENT):
    """Local commit of the new submodule pointer, restricted to the submodule
    path. Returns True if a commit was made."""
    changed = _git(repo_root, "diff", "--quiet", "--ignore-submodules=dirty", "HEAD", "--", webapp_rel)
    if changed.returncode == 0:
        return False
    _git(repo_root, "commit", "-q", "-m", message, "--", webapp_rel, check=True)
    if push:
        _git(repo_root, "push", "-q", check=True)
    return True


# ---- entry points ----

_STATE_LOCK = threading.Lock()
_running = False
_pending_label = None


def sync(label, webapp_dir=None, repo_root=None, keep=KEEP_SETS):
    """One full pass: prune -> commit/push the replay repo -> bump the parent
    pointer. Never raises; returns a result dict (or None when skipped)."""
    try:
        if webapp_dir is None and not webapp_mirror.webapp_ready():
            return None
        webapp_dir = Path(webapp_dir or webapp_mirror.WEBAPP_DIR)
        repo_root = Path(repo_root or REPO_ROOT)
        removed, rewritten = prune_all(webapp_dir, keep)
        sidecars_built, sidecars_dropped = build_sidecars(webapp_dir)
        git = commit_and_push(webapp_dir, f"Mirror {label}")
        bumped = False
        if git["error"] is None and git["head"] and webapp_dir.is_relative_to(repo_root):
            bumped = bump_parent_pointer(repo_root, webapp_dir.relative_to(repo_root).as_posix(),
                                         f"Bump replay repo to {git['head']} ({label})")
        print(f"webapp_sync: {label}: pruned {len(removed)} replays + {rewritten} check files; "
              f"sidecars built={sidecars_built} dropped={sidecars_dropped}; "
              f"committed={git['committed']} pushed={git['pushed']} head={git['head']} "
              f"parent_bumped={bumped}" + (f" ERROR: {git['error']}" if git["error"] else ""), flush=True)
        return {"removed": removed, "rewritten": rewritten, "sidecars": (sidecars_built, sidecars_dropped),
                "git": git, "parent_bumped": bumped}
    except Exception as e:  # noqa: BLE001 -- never let a mirroring hiccup take training down
        print(f"webapp_sync: {label}: failed, will retry at the next tick: {e!r}", flush=True)
        return None


def sync_async(label):
    """sync() on a background thread. If one is already running, remembers this
    label and runs once more when it finishes (coalesced). Returns the thread
    that was started, or None if it was coalesced."""
    global _running, _pending_label
    with _STATE_LOCK:
        if _running:
            _pending_label = label
            return None
        _running = True
    thread = threading.Thread(target=_worker, args=(label,), name="webapp-sync")
    thread.start()
    return thread


def _worker(label):
    global _running, _pending_label
    while True:
        sync(label)
        with _STATE_LOCK:
            if _pending_label is None:
                _running = False
                return
            label, _pending_label = _pending_label, None
