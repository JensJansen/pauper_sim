"""Engine code-version compatibility gate. Under this repo's rules-faithfulness
mandate a node running stale engine code produces training data that looks
valid but isn't, so the hub refuses to hand a deck to a node whose code isn't
on an operator-approved list. (Config drift is a different matter and is
deliberately NOT checked -- see DISTRIBUTED_TRAINING_PLAN.md.)

A registering node reports three things:
  - reported_sha: `git rev-parse HEAD` of its checkout (a commit SHA even if it
    was provisioned from a tag);
  - reported_dirty: whether any TRACKED file differs from HEAD (untracked files
    are ignored -- the venv and node-local config live in the checkout; the
    trust model is non-adversarial, this catches an honest local edit);
  - reported_vocab_hash: hash of its card vocabulary (index assignment must
    match the hub's or the nets read cards as the wrong inputs).

The hub accepts only SHAs listed in an operator-maintained file
(`distributed_configs/accepted_versions.local.json`, gitignored, hub-local:
`{"shas": ["<40-char sha or a >=7-char prefix>", ...]}`), re-read on every
registration so approving a new SHA needs no hub restart and a waiting node
picks it up on its next retry. A missing/empty file rejects everyone (fail
closed). The hub's OWN git HEAD is deliberately not the reference: the hub
never trains, and its working tree has uncommitted work.

`check` with accepted_shas=None and expected_vocab_hash=None is the gate
switched off (tests / `--no-version-gate`).
"""
import hashlib
import json
import os
import subprocess

MIN_SHA_PREFIX = 7


def vocab_hash(vocab):
    """Stable hash of a rl.model.features.CardVocab's name -> index mapping."""
    return hashlib.sha256(json.dumps(vocab.name_to_index, sort_keys=True).encode("utf-8")).hexdigest()


def local_version(repo_root):
    """(sha, dirty) for the git checkout at repo_root; (None, False) if it
    isn't one (the hub then rejects the node -- it must be able to say what
    code it runs)."""
    try:
        sha = subprocess.run(["git", "-C", str(repo_root), "rev-parse", "HEAD"], capture_output=True,
                             text=True, check=True).stdout.strip()
        status = subprocess.run(["git", "-C", str(repo_root), "status", "--porcelain", "--untracked-files=no"],
                                capture_output=True, text=True, check=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None, False
    return sha, bool(status)


def load_accepted_shas(path):
    """The accepted SHA list, [] if the file is missing or unreadable (fail
    closed). Entries shorter than MIN_SHA_PREFIX characters are ignored."""
    try:
        with open(path) as f:
            shas = json.load(f)["shas"]
    except (OSError, ValueError, KeyError, TypeError):
        return []
    return [s.strip().lower() for s in shas if isinstance(s, str) and len(s.strip()) >= MIN_SHA_PREFIX]


def check(reported_sha, reported_vocab_hash, reported_dirty=False, accepted_shas=None, expected_vocab_hash=None):
    """(accepted: bool, reason: str | None)."""
    if accepted_shas is not None:
        if not reported_sha:
            return False, "node did not report a code version (not running from a git checkout?)"
        if reported_dirty:
            return False, f"node's checkout at {reported_sha[:10]} has local modifications to tracked files"
        sha = reported_sha.lower()
        if not any(sha.startswith(a) for a in accepted_shas):
            return False, (f"code version {reported_sha[:10]} is not on the hub's accepted list "
                           f"({len(accepted_shas)} accepted)")
    if expected_vocab_hash is not None and reported_vocab_hash != expected_vocab_hash:
        return False, "card vocabulary does not match the hub's (indices would be misread)"
    return True, None
