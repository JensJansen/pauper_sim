"""Best-effort mirror of validation output into the webapp submodule's
logs/validation/, so /stats (src/webapp/README.md) shows real data without
the manual copy step it needed before this module existed. Used by
rl/league/league_runner.py (metrics.jsonl, progress.json) and
validation/_common.py (checks/*.json) -- the two places that already write
checkpoints/<league>/, so mirroring rides along with those writes rather
than becoming every caller's own responsibility.

mirror_json consolidates: checkpoints/<league>/checks/<check>_<N>games.json
(a new file every cadence tick, forever) mirrors into ONE growing
checks/<check>.jsonl per check (one compact line per cadence tick), not a
same-shaped new file per tick -- see mirror_json's own docstring. This is a
webapp-mirror-only change: checkpoints/ itself (the canonical, unbounded,
gitignored local copy) keeps writing one file per snapshot exactly as
before -- see validation/_common.py's write_league_json/write_deck_json.
app.py's /checks routes read both the old (pre-migration) and new shape.

Every function here is a no-op when either guard fails:
  - webapp_ready(): src/webapp isn't a checked-out git submodule (a fresh
    clone with the submodule never `git submodule update --init`'d leaves
    an empty placeholder directory, not a checkout).
  - _real_league_name(): league_dir isn't literally CHECKPOINTS_DIR/<name>
    -- true for every real league, false for a benchmark harness's
    throwaway league_dir (see league_runner._run_session's own docstring on
    that param) or a test's tmp_path. A throwaway/test run has no business
    landing in the committed webapp copy.
Both are re-checked on every call (a handful of stat()s) rather than cached
-- cheap next to the training/validation work already happening around
these calls, and it means a submodule init/deinit mid-session takes effect
immediately instead of needing a restart.

A write that gets past both guards is still wrapped in try/except OSError:
this mirroring is a convenience for the webapp, never allowed to raise and
take training down with it (disk full, permissions, a submodule mid-checkout
-- none of that is this module's problem to propagate).

Stdlib-only, no torch import -- same rationale as repo_paths.py, and it
keeps this importable from a plain `python -c` sanity check.
"""
import json
import re
from pathlib import Path

from repo_paths import CHECKPOINTS_DIR, SRC_DIR

WEBAPP_DIR = SRC_DIR / "webapp"
WEBAPP_VALIDATION_DIR = WEBAPP_DIR / "logs" / "validation"

_CHECK_SNAPSHOT_RE = re.compile(r"^((?:.*/)?checks/)([^/]+)_\d+games\.json$")


def webapp_ready():
    """True when src/webapp is an actually-checked-out git submodule (not an
    uninitialized placeholder) with its own logs/ directory present -- what
    `git submodule update --init` leaves behind. Wrapped in try/except
    OSError like every write below: Path.exists()/is_dir() only swallow
    ENOENT/ENOTDIR/EBADF/ELOOP internally, so a PermissionError on the stat
    (e.g. a locked/inaccessible submodule dir) would otherwise escape this
    guard uncaught -- straight into callers like league_runner._save_progress
    that have no try/except of their own."""
    try:
        return (WEBAPP_DIR / ".git").exists() and (WEBAPP_DIR / "logs").is_dir()
    except OSError:
        return False


def _real_league_name(league_dir):
    """The league name if league_dir is exactly CHECKPOINTS_DIR/<name> --
    None for anything else, so a benchmark/test's throwaway dir never gets
    mirrored into the committed webapp copy (see module docstring)."""
    try:
        rel = Path(league_dir).resolve().relative_to(CHECKPOINTS_DIR.resolve())
    except (ValueError, OSError):
        return None
    return rel.parts[0] if len(rel.parts) == 1 else None


def _consolidated_check_path(relative_path):
    """checks/<check>_<N>games.json (or <deck>/checks/<check>_<N>games.json)
    -> checks/<check>.jsonl -- every snapshot for a given check (+ league,
    + deck) accumulates as one growing JSONL file instead of a new file
    per cadence tick, forever. Every check payload already stamps its own
    "cumulative_games" (see each check module's own run()), so a line is
    self-describing without needing the tick encoded in a filename --
    app.py's /checks routes resolve an (old-shaped) per-snapshot request
    path back to the matching line in this file. None for a relative_path
    that isn't a checks/*.json snapshot shape -- mirror_json's only two
    real callers (write_league_json/write_deck_json) always pass this
    shape, so this is a defensive fallback, not an expected path."""
    m = _CHECK_SNAPSHOT_RE.match(relative_path.replace("\\", "/"))
    if not m:
        return None
    prefix, check = m.groups()
    return f"{prefix}{check}.jsonl"


def mirror_json(league_dir, relative_path, payload):
    """Mirror of checkpoints/<league>/<relative_path> (already written by
    the caller) into logs/validation/<league>/ -- appended as one compact
    JSON line to a consolidated <check>.jsonl file rather than written as
    its own same-named file, see _consolidated_check_path. A relative_path
    that doesn't match that checks/*.json shape is silently skipped."""
    league_name = _real_league_name(league_dir)
    if league_name is None or not webapp_ready():
        return
    consolidated_rel = _consolidated_check_path(relative_path)
    if consolidated_rel is None:
        return
    try:
        dest = WEBAPP_VALIDATION_DIR / league_name / consolidated_rel
        dest.parent.mkdir(parents=True, exist_ok=True)
        with open(dest, "a") as f:
            f.write(json.dumps(payload) + "\n")
    except OSError:
        pass


def mirror_metrics_line(league_dir, fields):
    """Mirror of one metrics.jsonl line (already appended by the caller,
    league_runner._append_metric) into
    logs/validation/<league>/metrics.jsonl."""
    league_name = _real_league_name(league_dir)
    if league_name is None or not webapp_ready():
        return
    try:
        dest_dir = WEBAPP_VALIDATION_DIR / league_name
        dest_dir.mkdir(parents=True, exist_ok=True)
        with open(dest_dir / "metrics.jsonl", "a") as f:
            f.write(json.dumps(fields) + "\n")
    except OSError:
        pass


def mirror_progress(league_dir, payload):
    """Mirror of progress.json (already written by the caller,
    league_runner._save_progress) into
    logs/validation/<league>/progress.json."""
    league_name = _real_league_name(league_dir)
    if league_name is None or not webapp_ready():
        return
    try:
        dest_dir = WEBAPP_VALIDATION_DIR / league_name
        dest_dir.mkdir(parents=True, exist_ok=True)
        with open(dest_dir / "progress.json", "w") as f:
            json.dump(payload, f)
    except OSError:
        pass
