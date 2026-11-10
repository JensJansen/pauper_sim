"""checkpoints/<league>/manifest.json: what a run IS -- so two runs (or the same run before and after a model change)
can be told apart and compared honestly long after the code has moved on.

Written by run_training_pipeline.py at the start of every invocation and closed at the end. One league directory is
one run: its weights, metrics.jsonl, checks/ and this manifest travel together (archive or discard them as one
unit). Records:

  config          the resolved training options in force at the latest session; `config_history` keeps every
                  earlier value (with the time it was replaced) so a mid-run change is visible
  model           the network architecture (trunk widths, encoder size, parameter count, a hash of the module
                  structure, vocabulary size, action-table sizes); `model_history` likewise
  code            git commit / branch / whether the tree had uncommitted changes, at each session start
  environment     python / torch / CUDA device
  sessions        one entry per invocation: started_at, finished_at, games/deck at start and end, argv

Stdlib only apart from the optional model summary. Best-effort like the other run records: callers wrap it so a
failure here never stops training."""
import hashlib
import json
import os
import platform
import subprocess
import sys
import time

from webapp_publish.mirror import mirror_manifest

MANIFEST_FILE = "manifest.json"
SCHEMA_VERSION = 1


def _git(repo_root, *args):
    try:
        out = subprocess.run(["git", *args], cwd=repo_root, capture_output=True, text=True, timeout=20, check=False)
        return out.stdout.strip() if out.returncode == 0 else None
    except (OSError, subprocess.SubprocessError):
        return None


def code_state(repo_root):
    """{"commit", "branch", "dirty"} for the repo at repo_root; any field None when git is unavailable."""
    status = _git(repo_root, "status", "--porcelain")
    return {"commit": _git(repo_root, "rev-parse", "HEAD"), "branch": _git(repo_root, "rev-parse", "--abbrev-ref", "HEAD"),
            "dirty": None if status is None else bool(status)}


def model_info(net, vocab_size, trunk_hidden, n_fixed_actions):
    """A summary of one deck network's architecture. `structure_sha` hashes the module structure (layer types and
    shapes, not weights), so it changes exactly when the architecture does."""
    return {"trunk_hidden": list(trunk_hidden), "vocab_size": vocab_size,
            "encoder_d_model": getattr(net.encoder, "d_model", None),
            "n_params_example_deck": sum(p.numel() for p in net.parameters()),
            "structure_sha": hashlib.sha256(repr(net).encode()).hexdigest()[:16],
            "n_fixed_actions": dict(n_fixed_actions)}


def _environment():
    env = {"python": platform.python_version(), "platform": platform.platform()}
    try:
        import torch
        env["torch"] = torch.__version__
        env["cuda_device"] = torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
    except Exception:  # noqa: BLE001 -- a manifest never fails a run
        pass
    return env


def _now():
    return time.strftime("%Y-%m-%dT%H:%M:%S%z")


def _jsonable(config):
    return json.loads(json.dumps(config, default=str))


def read(league_dir):
    path = os.path.join(league_dir, MANIFEST_FILE)
    if not os.path.exists(path):
        return None
    with open(path) as f:
        return json.load(f)


def _write(league_dir, manifest):
    os.makedirs(league_dir, exist_ok=True)
    with open(os.path.join(league_dir, MANIFEST_FILE), "w") as f:
        json.dump(manifest, f, indent=2)
    mirror_manifest(league_dir, manifest)


def start_session(league_dir, config, model, games_start, repo_root, argv=None):
    """Creates or updates the manifest for a new invocation and returns it."""
    m = read(league_dir) or {"schema_version": SCHEMA_VERSION, "created_at": _now(), "config_history": [],
                              "model_history": [], "sessions": []}
    config, now = _jsonable(config), _now()
    if m.get("config") is not None and m["config"] != config:
        m["config_history"].append({"replaced_at": now, "config": m["config"]})
    if m.get("model") is not None and m["model"] != model:
        m["model_history"].append({"replaced_at": now, "model": m["model"]})
    m["config"], m["model"], m["environment"] = config, model, _environment()
    m["sessions"].append({"started_at": now, "finished_at": None, "games_start": games_start, "games_end": None,
                          "code": code_state(repo_root), "argv": list(sys.argv if argv is None else argv)})
    _write(league_dir, m)
    return m


def finish_session(league_dir, games_end):
    """Stamps the latest session as finished (no-op if there is no manifest or session)."""
    m = read(league_dir)
    if not m or not m.get("sessions"):
        return None
    m["sessions"][-1].update({"finished_at": _now(), "games_end": games_end})
    _write(league_dir, m)
    return m
