"""Centralized checkpoint save/load for this repo's on-disk schemas.

Two schemas:
  - deck checkpoint (save/load_deck_checkpoint): {"net": state_dict[, "optimizer": state_dict]},
    used for both live.pt (DeckNetwork+Adam) and mulligan.pt (MulliganNet+Adam).
  - snapshot (save/load_snapshot): {"state_dict", "trunk_hidden", optional
    "mulligan_state_dict"/"mulligan_hidden"} -- a frozen historical opponent
    (rl.league.league.LeaguePool).

load_snapshot returns the raw dict rather than building the net: the caller
needs trunk_hidden to construct the right-shaped net before load_state_dict
can run.
"""
import os
import time

import torch

# ponytail: OneDrive transiently locks a file mid-write on Windows (torch.save's
# _open_zipfile_writer fails with error 1224, ERROR_USER_MAPPED_FILE). Retry
# with backoff before giving up; a non-transient failure (disk full, bad path,
# permissions) still raises after _SAVE_RETRY_ATTEMPTS.
_SAVE_RETRY_ATTEMPTS = 5
_SAVE_RETRY_BASE_DELAY = 0.5


def _save_with_retry(obj, path):
    """Writes to a temp file beside `path`, then os.replace()s it into place
    -- atomic on both Windows and POSIX, so a write that dies partway leaves
    the previous good checkpoint intact instead of a truncated one. The temp
    name carries the pid so concurrent writers to the same directory can't
    collide."""
    tmp = f"{path}.{os.getpid()}.tmp"
    for attempt in range(_SAVE_RETRY_ATTEMPTS):
        try:
            torch.save(obj, tmp)
            os.replace(tmp, path)  # retried too: OneDrive can lock the DESTINATION, not just the write
            return
        except (RuntimeError, OSError):
            if attempt == _SAVE_RETRY_ATTEMPTS - 1:
                try:
                    os.remove(tmp)  # don't leave a half-written temp file behind
                except OSError:
                    pass
                raise
            time.sleep(_SAVE_RETRY_BASE_DELAY * (2 ** attempt))


def load_optimizer_if_present(optimizer, ckpt, key="optimizer"):
    """Loads optimizer state from ckpt[key] if present, else leaves
    `optimizer` at its fresh-constructed state. Tolerates a checkpoint saved
    before that optimizer entry existed."""
    if key in ckpt:
        optimizer.load_state_dict(ckpt[key])


def _to_cpu(obj):
    """Recursively moves every tensor in a state-dict-shaped structure to
    CPU, so a checkpoint is device-agnostic regardless of training device.
    Handles the Adam optimizer state-dict shape (tensors nested under
    {"state": {idx: {"exp_avg": ..., ...}}}) too."""
    if torch.is_tensor(obj):
        return obj.detach().cpu()
    if isinstance(obj, dict):
        return {k: _to_cpu(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return type(obj)(_to_cpu(v) for v in obj)
    return obj


def save_deck_checkpoint(path, net, optimizer=None):
    """Writes net's (and optimizer's, if given) state to path as
    {"net": ...[, "optimizer": ...]}. Always written on CPU (see _to_cpu)."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    saved = {"net": _to_cpu(net.state_dict())}
    if optimizer is not None:
        saved["optimizer"] = _to_cpu(optimizer.state_dict())
    _save_with_retry(saved, path)


def load_deck_checkpoint(path, net, optimizer=None):
    """Loads net's (and optimizer's, if given) state from path in place.
    No-op if path doesn't exist yet. optimizer load goes through
    load_optimizer_if_present so a legacy checkpoint with no optimizer state
    re-warms with a fresh Adam instead of KeyError-ing. Returns True iff
    path existed and was loaded."""
    if not os.path.exists(path):
        return False
    ckpt = torch.load(path, weights_only=True, map_location="cpu")  # CPU load works into any target device
    net.load_state_dict(ckpt["net"])
    if optimizer is not None:
        load_optimizer_if_present(optimizer, ckpt)
    return True


def save_snapshot(path, net, mulligan_net=None):
    """Writes net's frozen weights plus its trunk_hidden shape (needed to
    reconstruct a same-shaped DeckNetwork on load), and mulligan_net's
    frozen weights plus hidden width if given. mulligan_net=None writes a
    deck-only snapshot; load_snapshot's caller falls back to AlwaysKeep
    for one."""
    trunk_hidden = tuple(layer.out_features for layer in net.trunk_layers)
    saved = {"state_dict": _to_cpu(net.state_dict()), "trunk_hidden": trunk_hidden}  # CPU, see save_deck_checkpoint
    if mulligan_net is not None:
        saved["mulligan_state_dict"] = _to_cpu(mulligan_net.state_dict())
        saved["mulligan_hidden"] = mulligan_net.trunk[0].out_features
    _save_with_retry(saved, path)


def load_snapshot(path):
    """Returns the raw saved dict (state_dict, trunk_hidden, optional
    mulligan_state_dict/mulligan_hidden) -- does not build the net itself;
    the caller needs trunk_hidden to construct the right shape first."""
    return torch.load(path, weights_only=True, map_location="cpu")


def trunk_hidden_from_deck_checkpoint(path):
    """The DeckNetwork trunk widths a live.pt was saved with, read off its
    tensor shapes. Returns None if the file does not exist. Needed because
    trunk_hidden is per-league configurable, so a caller must build the net
    before loading into it and can't assume a default width."""
    if not os.path.exists(path):
        return None
    sd = torch.load(path, map_location="cpu", weights_only=False)["net"]
    widths, i = [], 0
    while f"trunk_layers.{i}.weight" in sd:
        widths.append(sd[f"trunk_layers.{i}.weight"].shape[0])
        i += 1
    assert widths, f"{path} has no trunk_layers.* -- not a DeckNetwork checkpoint?"
    return tuple(widths)
