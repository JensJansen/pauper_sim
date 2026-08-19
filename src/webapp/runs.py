"""Subprocess-backed run management for the training-ops web UI (app.py).

A "run" is a plain OS subprocess: run_league.py invoked
with explicit CLI flags built from the submitted form values -- NEVER a
--run-config/--league-config PATH. That's what makes a loaded
training_configs/*.json preconfiguration purely a form-prefill convenience
(see app.py's /api/configs) rather than something the running process reads
again: once "Start" is clicked, the values on screen are the whole story.

Runs are tracked in-memory (self._procs) for the life of the server process
-- polling a live subprocess.Popen is the only reliable way to know it's
still running / read its exit code, there's no portable PID-liveness check
that doesn't need extra dependencies. A JSON registry on disk
(logs/webapp_runs/registry.json) survives a server restart for history and
log access, but a run started before a restart can no longer be stopped
through the UI -- it just keeps running to completion untouched.

League mode field grouping (LEAGUE_GLOBAL / LEAGUE_MODES, imported from
rl.league_cli_spec) is hand-authored domain knowledge about which flags
matter for which of run_league.py's three real run modes (see its own module
docstring's Usage section + main()'s --eval / --matchup branches) -- argparse
has no way to introspect that, only flag names/types/help. test_runs.py
cross-checks it against the real parser so a future flag addition can't
silently go ungrouped.

Auto-sizing escalation: when a League-mode submission leaves --n-iterations
blank and gives --total-games, a single run_league.py invocation only ever
plays ONE batch of its own internal doubling ladder (rl.league_runner._next_batch_games)
-- exactly the "start tiny, verify, double" behavior the `/train` skill drives
by hand across many separate invocations. _escalating_loop automates that same
loop: after each batch, check it was actually healthy (see _batch_healthy's own
docstring for why exit code alone is NOT reliable here), then re-invoke the
identical command again (run_league resumes from its own progress.json/session.txt)
until this league's cumulative games/deck reaches the target, a batch comes back
unhealthy, or the user hits Stop.
"""
import argparse
import json
import os
import re
import subprocess
import sys
import threading
import time
import uuid
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # src/, for `repo_paths` / `rl.league_cli_spec`
from repo_paths import REPO_ROOT, SRC_DIR, CHECKPOINTS_DIR  # noqa: E402
from rl.league_cli_spec import build_arg_parser, LEAGUE_GLOBAL, LEAGUE_MODES  # noqa: E402 -- torch-free, safe on every league-run start
import analysis.report_metrics as report_metrics  # noqa: E402 -- torch-free, safe on every league-run start

LOG_DIR = REPO_ROOT / "logs" / "webapp_runs"  # registry.json only -- per-run logs live under RUNS_ROOT, see _run_dir
REGISTRY_PATH = LOG_DIR / "registry.json"
RUNS_ROOT = REPO_ROOT / "logs"

# Only one training script now. There used to be a "pretrain" entry here too,
# for run_pretrain.py, which built and froze the league's single shared
# perception stack; per-deck encoders (2026-08-17) removed that whole phase.
SCRIPTS = {"league": SRC_DIR / "run_league.py"}

# run_league.py's own --run-config/--league-config: never exposed, since runs
# are always started with fully-resolved explicit flags (see module docstring).
_SKIP_DESTS = {"help", "run_config", "league_config"}

# LEAGUE_GLOBAL / LEAGUE_MODES themselves now live in rl.league_cli_spec (see
# its own module docstring), imported above -- re-exported here unchanged so
# app.py's `from runs import LEAGUE_GLOBAL, LEAGUE_MODES, ...` keeps working.


def argspec_from_parser(parser):
    """Introspect an argparse.ArgumentParser into a JSON-able field spec, so
    the web form always matches the script's real CLI -- one source of
    truth (build_arg_parser's own flags/types/help), no hand-maintained
    duplicate field list to drift out of sync as flags are added."""
    spec = []
    for action in parser._actions:  # argparse has no public introspection API
        if action.dest in _SKIP_DESTS:
            continue
        if isinstance(action, argparse.BooleanOptionalAction):
            # Also nargs==0 like store_true, but has a SECOND --no-X option
            # string for "explicitly off" -- store_true's kind can only ever
            # omit the flag (Python bool "False" -> "don't pass it"), which
            # would silently collapse "explicitly off" into "unspecified,
            # use the script's own default" for a flag whose default may be
            # True (e.g. --pfsp). Needs its own kind so build_argv can emit
            # --no-X, not just decide whether to emit --X.
            kind = "tri_bool"
        elif action.nargs == 0:  # store_true
            kind = "store_true"
        elif action.type is int:
            kind = "int"
        elif action.type is float:
            kind = "float"
        else:
            kind = "text"
        spec.append({
            "dest": action.dest,
            "flags": list(action.option_strings),
            "type": kind,
            "nargs": action.nargs if isinstance(action.nargs, int) and action.nargs > 0 else None,
            "default": action.default,
            "metavar": action.metavar,
            "help": action.help,
        })
    return spec


def _league_parser():
    return build_arg_parser()  # rl.league_cli_spec -- torch-free, no run_league.py import needed


def build_argv(script, values):
    """values: {dest: value} from the submitted form (already JSON-decoded).
    A missing/empty/None value means "flag omitted" -- the script's own
    hardcoded default applies, exactly as if left off the CLI by hand."""
    assert script == "league", f"unknown script {script!r}"
    argv = []
    for field in argspec_from_parser(_league_parser()):
        val = values.get(field["dest"])
        if val is None or val == "" or val == []:
            continue
        flag = field["flags"][0]
        if field["type"] == "store_true":
            if val:
                argv.append(flag)
        elif field["type"] == "tri_bool":
            # val is never None/""/[] here (skipped above) -- an explicit
            # True/False, so emit the matching --X / --no-X flag rather than
            # store_true's "omit unless truthy" (which could never express
            # "explicitly off" for a default-True flag like --pfsp).
            argv.append(field["flags"][0] if val else field["flags"][1])
        elif field["nargs"] == 2:
            a, b = val
            argv += [flag, str(a), str(b)]
        else:
            argv += [flag, str(val)]
    return argv


def is_auto_sizing_league_run(values):
    """True iff this submission takes run_league.py's auto-sizing path
    (main()'s else-branch: no --n-iterations debug override, --total-games
    given) rather than a forced one-off size -- the only case where a
    single invocation plays just ONE batch of a doubling ladder instead of
    everything requested in one shot."""
    return bool(not values.get("matchup") and not values.get("eval")
                and not values.get("n_iterations")
                and values.get("total_games"))


def _run_dir(script, run_id):
    """logs/<timestamp>-<script>-<short-id>/ -- one folder per run, holding
    stdout.log and (if --log was requested) event_log.json together, so
    app.py's /api/replay/runs browser can list and open either without a
    separate registry lookup, and so the two kinds of log stop being
    distinguishable only by hand-picked filename prefix."""
    d = RUNS_ROOT / f"{time.strftime('%Y%m%d-%H%M%S')}-{script}-{run_id[:6]}"
    d.mkdir(parents=True, exist_ok=True)
    return d


def _league_dir_for(values):
    league_name = values.get("league_name")
    return CHECKPOINTS_DIR / league_name if league_name else CHECKPOINTS_DIR / "league"


def _read_cumulative_games(league_dir):
    path = league_dir / "progress.json"
    if not path.exists():
        return 0
    return json.loads(path.read_text()).get("cumulative_games_per_deck", 0)


def _rollback_hint(league_dir, deck, rows, peak_at):
    """"snapshot_116 (~23,200 games)" for the reading that scored best, or None
    if the records predate `cumulative_games` being written.

    Snapshots are numbered by the league's own counter and taken every
    `snapshot_every_games`, so the id is cumulative_games // snapshot_every --
    read off the session_start record rather than assumed, since it is a
    per-league config value."""
    pts = [r for r in rows if r.get("games")]
    if peak_at is None or peak_at >= len(pts):
        return None
    cumulative = pts[peak_at].get("cumulative_games")
    if not cumulative:
        return None  # record predates the field; nothing to compute from
    every = _snapshot_every_games(league_dir)
    if not every:
        return f"the checkpoint at ~{cumulative:,} games/deck"
    return f"snapshot_{cumulative // every} (~{cumulative:,} games/deck)"


def _snapshot_every_games(league_dir):
    """This league's own snapshot cadence, from the most recent session_start
    record. None if nothing recorded it (pre-2026-08-13 records)."""
    path = Path(league_dir) / "metrics.jsonl"
    if not path.exists():
        return None
    starts = [r for r in report_metrics.load(str(path)) if r.get("kind") == "session_start"]
    for r in reversed(starts):
        # snapshot_every is in ITERATIONS; multiply back up to games/deck.
        if r.get("snapshot_every") and r.get("games_per_iteration"):
            return r["snapshot_every"] * r["games_per_iteration"]
    return None


def learning_health(league_dir, min_records=10):
    """(verdict, [human-readable line per deck]) for a league's own
    metrics.jsonl -- "is this run still buying anything," which is a completely
    different question from _batch_healthy's "did the process crash."

    Verdict is the WORST across decks: "regressed" > "stalled" > "ok", or
    "unknown" when there is not yet enough history to say. Delegates the
    statistics to report_metrics.peak_comparison / trend_z rather than
    reimplementing them, so the gate and the report can never disagree.

    Why this exists: `total_games: 600000` in run_default.json is ~10.3 days of
    wall clock at measured throughput. Nothing in this loop could tell a run
    that was learning from one that was not, so a 60,001-games/deck run
    continued for weeks while three of its four decks were getting WORSE --
    each of them ending up weaker than a snapshot already sitting on disk.
    Crash-free is not the same as healthy."""
    path = Path(league_dir) / "metrics.jsonl"
    if not path.exists():
        return "unknown", []
    records = report_metrics.load(str(path))
    by_deck = {}
    for r in records:
        # archive_oldest specifically: it is pinned to snapshot_0 forever, which
        # makes it the one FIXED reference in the file. active_oldest moves as
        # the pool rolls, so a change in it is ambiguous between the policy
        # improving and the reference getting stronger.
        if r.get("kind") == "vs_history" and r.get("label") == "archive_oldest":
            by_deck.setdefault(r.get("deck", "?"), []).append(r)

    states, lines = [], []
    for deck in sorted(by_deck):
        rows = sorted(by_deck[deck], key=lambda r: (r.get("session", 0), r.get("iteration", 0)))
        if len(rows) < min_records:
            continue
        peak_z, peak_at, crit = report_metrics.peak_comparison(rows)
        t_z = report_metrics.trend_z(rows)
        if peak_z <= -crit:
            state = "regressed"
            detail = f"REGRESSING ({abs(peak_z):.1f} sigma below its own peak)"
            # Name the actual file to roll back to. Without this the gate says
            # "a better policy is on disk" and leaves the operator to work out
            # WHICH -- which previously meant reconstructing the session ->
            # games mapping by hand from PPO iteration counts.
            hint = _rollback_hint(league_dir, deck, rows, peak_at)
            if hint:
                detail += f" -- best was ~{hint}"
        elif t_z <= -2:
            state, detail = "regressed", f"REGRESSING (trend z={t_z:+.2f})"
        elif t_z >= 2:
            state, detail = "ok", f"ok (trend z={t_z:+.2f})"
        else:
            state, detail = "stalled", f"stalled (trend z={t_z:+.2f})"
        states.append(state)
        lines.append(f"{deck}: {detail}")
    if not states:
        return "unknown", lines
    # Worst deck decides: one regressing deck makes the RUN unhealthy even if
    # the others are fine -- which is exactly the real case (mono_red_rally
    # gained 241 Elo while elves and rakdos both fell below their own past selves).
    for worst in ("regressed", "stalled", "ok"):
        if worst in states:
            return worst, lines
    return "unknown", lines


def _batch_healthy(log_tail):
    """Mirrors the `/train` skill's own health check, NOT the subprocess exit
    code: on Windows, a parallel run (--n-workers > 1) reliably exits 1 from
    ProcessPoolExecutor teardown even when the session fully completed and
    checkpointed cleanly -- ONLY the log content is trustworthy. Healthy iff
    a "session N done" summary line is present and no Traceback was printed."""
    if "Traceback" in log_tail:
        return False
    return re.search(r"session \d+ done", log_tail) is not None


# run_league.py does not log PER-ITERATION timing today
# (the "iter N [deck]: games=... policy_loss=..." lines carry no elapsed
# time) -- only a summary line at the end of each "batch" (one script
# invocation's own internal session, or one --eval pass). "Batch" is
# therefore the finest timing granularity actually available without
# changing the training scripts' own print statements.
_SESSION_DONE_RE = re.compile(r"session \d+ done in ([\d.]+)s \(([\d.]+)s/game(?: across (\d+) games)?\)")
_EVAL_DONE_RE = re.compile(r"eval done: (\d+) games in ([\d.]+)s")


def parse_last_batch_timing(log_text):
    """Timing for the most recently completed batch found in a run's log --
    whichever of the two summary-line formats above appears LAST in the
    text (an escalating session's log has one per batch; matches run_league's
    own "session N done" format for both League and Matchup training, or
    "eval done" for Eval mode). None if no batch has finished yet."""
    candidates = []
    for m in _SESSION_DONE_RE.finditer(log_text):
        total_s, s_per_game, games = float(m.group(1)), float(m.group(2)), m.group(3)
        candidates.append((m.start(), {
            "total_seconds": total_s, "avg_seconds_per_game": s_per_game,
            "games": int(games) if games else None,
        }))
    for m in _EVAL_DONE_RE.finditer(log_text):
        games, total_s = int(m.group(1)), float(m.group(2))
        candidates.append((m.start(), {
            "total_seconds": total_s, "avg_seconds_per_game": (total_s / games) if games else None,
            "games": games,
        }))
    if not candidates:
        return None
    return max(candidates, key=lambda c: c[0])[1]  # latest-in-file wins


class RunManager:
    def __init__(self):
        LOG_DIR.mkdir(parents=True, exist_ok=True)
        self._procs = {}  # run_id -> Popen of the CURRENTLY active batch (only runs THIS process started)
        self._cancel_events = {}  # run_id -> threading.Event, escalating sessions only
        self._registry = json.loads(REGISTRY_PATH.read_text()) if REGISTRY_PATH.exists() else {}

    def _save_registry(self):
        # ponytail: no lock -- single local user, worst case is a rare stale
        # read of the registry file, never corrupted training state (the
        # actual checkpoints/progress.json are only ever written by the
        # subprocess itself). Add a lock if multi-user access ever matters.
        REGISTRY_PATH.write_text(json.dumps(self._registry, indent=2))

    def start(self, script, values):
        assert script in SCRIPTS, f"unknown script {script!r}"
        if script == "league" and is_auto_sizing_league_run(values):
            return self._start_escalating(values)
        return self._start_single(script, values)

    def _start_single(self, script, values):
        run_id = uuid.uuid4().hex[:12]
        run_dir = _run_dir(script, run_id)
        if values.get("log"):
            # A filled-in log field means "log this run" -- the auto-organized
            # folder path replaces whatever the user typed, same opt-in signal
            # as before, now landing somewhere the browser can find it.
            values = {**values, "log": str(run_dir / "event_log.json")}
        argv = build_argv(script, values)
        log_path = run_dir / "stdout.log"
        cmd = [sys.executable, "-u", str(SCRIPTS[script])] + argv
        log_path.write_text(f"$ {' '.join(cmd)}\n\n")
        logfile = open(log_path, "a")
        # New process group on Windows so stop() can taskkill the WHOLE tree --
        # run_league.py --n-workers>1 spawns a ProcessPoolExecutor; terminating
        # only the parent would orphan its worker processes.
        creationflags = subprocess.CREATE_NEW_PROCESS_GROUP if os.name == "nt" else 0
        proc = subprocess.Popen(cmd, cwd=SRC_DIR, stdout=logfile, stderr=subprocess.STDOUT,
                                 creationflags=creationflags)
        self._procs[run_id] = proc
        self._registry[run_id] = {
            "id": run_id, "script": script, "mode": "single", "argv": argv, "pid": proc.pid,
            "started": time.time(), "ended": None, "status": "running",
            "exit_code": None, "log_path": str(log_path),
        }
        self._save_registry()
        return run_id

    def _start_escalating(self, values):
        run_id = uuid.uuid4().hex[:12]
        run_dir = _run_dir("league", run_id)
        if values.get("log"):
            values = {**values, "log": str(run_dir / "event_log.json")}
        argv = build_argv("league", values)
        log_path = run_dir / "stdout.log"
        log_path.write_text("")
        league_dir = _league_dir_for(values)
        entry = {
            "id": run_id, "script": "league", "mode": "auto_escalate", "argv": argv, "pid": None,
            "started": time.time(), "ended": None, "status": "running",
            "exit_code": None, "log_path": str(log_path),
            "batches_run": 0, "total_games": int(values["total_games"]),
            "cumulative_games_per_deck": _read_cumulative_games(league_dir),
        }
        self._registry[run_id] = entry
        self._save_registry()
        cancel_event = threading.Event()
        self._cancel_events[run_id] = cancel_event
        thread = threading.Thread(target=self._escalating_loop, args=(run_id, argv, league_dir, cancel_event), daemon=True)
        thread.start()
        return run_id

    def _escalating_loop(self, run_id, argv, league_dir, cancel_event):
        entry = self._registry[run_id]
        log_path = Path(entry["log_path"])
        cmd = [sys.executable, "-u", str(SCRIPTS["league"])] + argv
        creationflags = subprocess.CREATE_NEW_PROCESS_GROUP if os.name == "nt" else 0
        batch = 0
        while not cancel_event.is_set():
            # Check BEFORE spawning: if a previous session already reached the
            # target, stop here rather than spawning a batch -- run_league.py
            # itself would print "nothing to run" and exit with no "session N
            # done" line, which _batch_healthy would (wrongly) read as a failure.
            cumulative = _read_cumulative_games(league_dir)
            entry["cumulative_games_per_deck"] = cumulative
            if cumulative >= entry["total_games"]:
                entry.update(status="finished", ended=time.time(), exit_code=0)
                self._save_registry()
                return
            batch += 1
            with open(log_path, "a") as f:
                f.write(f"\n=== batch {batch} ===\n$ {' '.join(cmd)}\n\n")
            batch_start_offset = log_path.stat().st_size
            logfile = open(log_path, "a")
            proc = subprocess.Popen(cmd, cwd=SRC_DIR, stdout=logfile, stderr=subprocess.STDOUT,
                                     creationflags=creationflags)
            self._procs[run_id] = proc
            entry["pid"] = proc.pid
            entry["batches_run"] = batch
            self._save_registry()
            exit_code = proc.wait()
            logfile.close()

            if cancel_event.is_set():
                entry.update(status="stopped", ended=time.time(), exit_code=exit_code)
                self._save_registry()
                return

            with open(log_path, "r") as f:
                f.seek(batch_start_offset)
                batch_log = f.read()
            if not _batch_healthy(batch_log):
                entry.update(status="failed", ended=time.time(), exit_code=exit_code)
                self._save_registry()
                return

            entry["cumulative_games_per_deck"] = _read_cumulative_games(league_dir)

            # A crash-free batch is not necessarily a USEFUL batch. Check
            # whether the run is still buying anything before spending another
            # one (see learning_health).
            verdict, health_lines = learning_health(league_dir)
            entry["learning_health"] = verdict
            entry["learning_health_detail"] = health_lines
            if health_lines:
                with open(log_path, "a") as f:
                    f.write(f"\n=== learning health after batch {batch}: {verdict.upper()} ===\n")
                    for line in health_lines:
                        f.write(f"  {line}\n")
                    if verdict == "regressed":
                        f.write("  !! At least one deck is now WEAKER than a checkpoint already on disk.\n"
                                "     Continuing will train further from the degraded policy.\n")
            # DEFAULT IS WARN, NOT STOP. Stopping automatically is an
            # operator-policy call the repo owner has not made yet, and silently
            # halting someone's overnight run is the more destructive default of
            # the two. Set stop_on_regression on the run entry to opt in.
            if verdict == "regressed" and entry.get("stop_on_regression"):
                entry.update(status="stalled", ended=time.time(), exit_code=0)
                self._save_registry()
                return
            self._save_registry()
            # Healthy batch done -- loop back to the top, which re-checks
            # cumulative-vs-target before deciding whether to spawn another.
            # run_league.py resumes on its own from progress.json/session.txt.

        # Reached only if cancel_event was set exactly between two batches (the
        # `while` condition itself went false) -- every early return above
        # already records its own terminal status, this is just the fallback
        # for that one race window.
        entry.update(status="stopped", ended=time.time())
        self._save_registry()

    def stop(self, run_id):
        event = self._cancel_events.get(run_id)
        if event is not None:
            event.set()  # escalating session: stop launching further batches
        proc = self._procs.get(run_id)
        if proc is None:
            # This server process never held a live handle for this run -- almost
            # always because it was restarted after the run started (see the
            # README's "Known limitation"). Nothing we can do to actually kill it.
            return event is not None
        if proc.poll() is None:  # still actually running -- go kill it
            if os.name == "nt":
                subprocess.run(["taskkill", "/PID", str(proc.pid), "/T", "/F"], capture_output=True)
            else:
                proc.terminate()
            try:
                proc.wait(timeout=3)
            except subprocess.TimeoutExpired:
                pass  # checked via poll() below regardless of whether wait() caught the exit
            if proc.poll() is None:
                return False  # taskkill/terminate did NOT actually kill it -- don't claim success
            if event is None:  # single-batch run: WE killed it, so it's genuinely "stopped"
                entry = self._registry[run_id]
                entry.update(status="stopped", ended=time.time())
                self._save_registry()
        elif event is None:
            # It had already finished on its own by the time we checked -- resolve
            # its REAL status (finished/failed) via _refresh's exit-code logic
            # rather than mislabeling a natural finish as a manual "stopped".
            self._refresh(run_id)
        return True

    def _refresh(self, run_id):
        entry = self._registry.get(run_id)
        if entry is None or entry.get("mode") == "auto_escalate":
            return entry  # escalating sessions keep their own status current (see _escalating_loop)
        proc = self._procs.get(run_id)
        if entry["status"] == "running" and proc is not None:
            code = proc.poll()
            if code is not None:
                entry["status"] = "finished" if code == 0 else "failed"
                entry["exit_code"] = code
                entry["ended"] = time.time()
                self._save_registry()
        return entry

    def _with_timing(self, entry):
        """A COPY of entry with a "timing" field computed fresh from its log
        file -- never persisted to the registry (it's derived, cheap enough
        to recompute per request for a local single-user tool's log sizes,
        and staying derived-only means it can never go stale on disk)."""
        if entry is None:
            return None
        enriched = dict(entry)
        try:
            log_text = Path(entry["log_path"]).read_text(errors="replace")
        except OSError:
            log_text = ""
        enriched["timing"] = parse_last_batch_timing(log_text)
        return enriched

    def get(self, run_id):
        return self._with_timing(self._refresh(run_id))

    def list_runs(self):
        for run_id in list(self._registry):
            self._refresh(run_id)
        entries = sorted(self._registry.values(), key=lambda r: r["started"], reverse=True)
        return [self._with_timing(e) for e in entries]

    def tail_log(self, run_id):
        """Yield new lines from a run's log file as they're written, until the
        run is no longer active and no further content is available. Works
        unchanged for an escalating session's log -- it's one file appended
        to across every batch, so the stream just runs straight through."""
        entry = self._registry.get(run_id)
        if entry is None:
            return
        with open(entry["log_path"], "r") as f:
            while True:
                line = f.readline()
                if line:
                    yield line
                    continue
                if self._refresh(run_id)["status"] != "running":
                    return
                time.sleep(0.5)
