"""Centralizes checkpoint save/load for the distinct on-disk schemas that used
to be hand-rolled (build net, torch.load, load_state_dict, optional
optimizer-migration guard) at every call site across rl/league.py and
rl/league_runner.py.

Two schemas, two helper pairs (deliberately NOT unified into one signature
-- the shapes are genuinely different):

  - deck checkpoint (save/load_deck_checkpoint): one file per net,
    {"net": state_dict[, "optimizer": state_dict]} -- rl.league_runner's
    per-deck live.pt (DeckNetwork+Adam) and mulligan.pt (MulliganNet+Adam)
    share this exact shape.
  - snapshot (save/load_snapshot): rl.league.LeaguePool's frozen historical
    opponent, {"state_dict":, "trunk_hidden":, optional "mulligan_state_dict":,
    optional "mulligan_hidden":}.

There were two more until 2026-08-17, both belonging to the pretrain-then-
freeze design that per-deck encoders replaced: a shared/frozen-stack pair
(shared_stack_frozen.pt) and a combined pretrain checkpoint holding the whole
roster's throwaway heads plus the one shared stack. Neither has a writer any
more -- a DeckNetwork's encoder is a registered child, so it rides along in
the deck checkpoint and snapshot schemas above with no separate file.

The snapshot loader returns the raw saved dict rather than building the net
itself: the caller needs trunk_hidden FROM the dict to know what shape to
construct before load_state_dict can run, so net-building stays with the
caller.
"""
import os
import time

import torch

# ponytail: this repo lives inside a OneDrive-synced folder, and OneDrive's
# sync daemon transiently locks a file mid-write on Windows (torch.save's
# _open_zipfile_writer failing with "open file failed with error code: 1224"
# -- ERROR_USER_MAPPED_FILE), observed for real during a multi-hour unattended
# training campaign (2026-08). Every checkpoint write below retries through
# that specific transient window with backoff before giving up -- a real,
# non-transient failure (disk full, bad path, permissions) still raises after
# _SAVE_RETRY_ATTEMPTS, it just isn't mistaken for one on the first collision.
_SAVE_RETRY_ATTEMPTS = 5
_SAVE_RETRY_BASE_DELAY = 0.5


def _save_with_retry(obj, path):
    """ATOMIC: writes to a temp file beside `path`, then os.replace()s it into
    place. torch.save streams into the destination, truncating it first, so a
    write that dies partway (crash, OOM kill, Ctrl-C, power loss) used to leave
    a truncated file that had ALREADY destroyed the previous good checkpoint --
    unrecoverable, and live.pt carries the whole training history. os.replace
    is a single filesystem operation on both Windows and POSIX, so `path` names
    either the complete old file or the complete new one and never a partial
    write. A crash now costs only a stray .tmp.

    That also narrows the OneDrive window described above: the sync daemon can
    no longer observe a checkpoint mid-truncation, only a finished file
    appearing at once.

    The temp name carries the pid so two processes writing the same league
    directory cannot scribble on each other's partial file. Same filesystem
    (same directory) is required -- os.replace is only atomic within one."""
    tmp = f"{path}.{os.getpid()}.tmp"
    for attempt in range(_SAVE_RETRY_ATTEMPTS):
        try:
            torch.save(obj, tmp)
            os.replace(tmp, path)  # retried too: OneDrive can lock the DESTINATION, not just the write
            return
        except (RuntimeError, OSError):
            if attempt == _SAVE_RETRY_ATTEMPTS - 1:
                # Give up without leaving a half-written temp file behind.
                try:
                    os.remove(tmp)
                except OSError:
                    pass
                raise
            time.sleep(_SAVE_RETRY_BASE_DELAY * (2 ** attempt))


def load_optimizer_if_present(optimizer, ckpt, key="optimizer"):
    """Loads optimizer state from ckpt[key] iff present, else leaves
    `optimizer` at whatever fresh state the caller already constructed it
    with (the intended re-warm behavior). Guards every optimizer load below
    against a checkpoint saved before that optimizer entry existed -- a bare
    ckpt[key] would KeyError on such a legacy file."""
    if key in ckpt:
        optimizer.load_state_dict(ckpt[key])


def _to_cpu(obj):
    """Recursively moves every tensor in a state-dict-shaped structure to CPU.

    A checkpoint must be device-agnostic. torch.save records each tensor's
    device, so a GPU-trained net saved verbatim produces a live.pt that cannot
    even be DESERIALIZED on a machine without CUDA, and that silently drags
    weights back onto the GPU for every CPU-only consumer (analysis/*.py, the
    webapp, the eval paths). Nothing downstream should have to know which
    device a league happened to train on -- so the conversion lives here, at
    the one choke point every checkpoint write goes through, rather than at
    each call site.

    Handles the Adam-state shape too (optimizer.state_dict() nests tensors
    inside {"state": {idx: {"exp_avg": ..., ...}}}), which a flat
    {k: v.cpu()} comprehension would miss.
    """
    if torch.is_tensor(obj):
        return obj.detach().cpu()
    if isinstance(obj, dict):
        return {k: _to_cpu(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return type(obj)(_to_cpu(v) for v in obj)
    return obj


def save_deck_checkpoint(path, net, optimizer=None):
    """Writes net's (and, if given, optimizer's) state to path as
    {"net": ...[, "optimizer": ...]} -- the schema shared by run_league.py's
    per-deck live.pt (DeckNetwork+Adam) and mulligan.pt (MulliganNet+Adam).

    Always written on CPU regardless of the training device -- see _to_cpu."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    saved = {"net": _to_cpu(net.state_dict())}
    if optimizer is not None:
        saved["optimizer"] = _to_cpu(optimizer.state_dict())
    _save_with_retry(saved, path)


def load_deck_checkpoint(path, net, optimizer=None):
    """Loads net's (and, if given, optimizer's) state from path IN PLACE; a
    no-op if path doesn't exist yet (net/optimizer keep the fresh state the
    caller already constructed them with -- the cold-start case). optimizer's
    load goes through load_optimizer_if_present so a legacy checkpoint saved
    without optimizer state re-warms with a fresh Adam instead of
    KeyError-ing -- applies identically whether net is the main DeckNetwork
    or a MulliganNet, closing a gap where only the DeckNetwork (live.pt) path
    used to carry this guard. Returns True iff path existed (and was
    loaded), False otherwise, so callers that pass optimizer=None (eval
    passes, which only ever need inference weights) can still tell trained
    weights from an untrained fresh net."""
    if not os.path.exists(path):
        return False
    # map_location="cpu": the saver already writes CPU tensors, but a legacy
    # file written before that (or by any other tool) would otherwise demand
    # the exact device it was saved on just to deserialize. Loading to CPU and
    # letting load_state_dict copy into whatever device `net` lives on is
    # correct for every combination, including CPU-file -> GPU-net.
    ckpt = torch.load(path, weights_only=True, map_location="cpu")
    net.load_state_dict(ckpt["net"])
    if optimizer is not None:
        load_optimizer_if_present(optimizer, ckpt)
    return True


def save_snapshot(path, net, mulligan_net=None):
    """Writes net's frozen weights (plus its trunk_hidden shape, needed to
    reconstruct a same-shaped DeckNetwork on load) and, if given,
    mulligan_net's frozen weights (plus its hidden width) to path --
    rl.league.LeaguePool.register_snapshot's own schema. mulligan_net=None
    writes a deck-only snapshot; load_snapshot's caller falls back to
    AlwaysKeep for one (see rl.league.LeaguePool.load_snapshot_agent)."""
    trunk_hidden = tuple(layer.out_features for layer in net.trunk_layers)
    # CPU for the same reason save_deck_checkpoint is (see _to_cpu): snapshots
    # are registered mid-session straight off the live nets, so on a
    # GPU-trained league they would otherwise be written as CUDA tensors and
    # become unloadable for every CPU-only consumer -- and snapshots are read
    # far more widely than live.pt is (PFSP opponent sampling, vs_history,
    # load_vintage_agent, the cross-league eval's vintages).
    saved = {"state_dict": _to_cpu(net.state_dict()), "trunk_hidden": trunk_hidden}
    if mulligan_net is not None:
        saved["mulligan_state_dict"] = _to_cpu(mulligan_net.state_dict())
        saved["mulligan_hidden"] = mulligan_net.trunk[0].out_features  # restore the exact hidden width on load
    _save_with_retry(saved, path)


def load_snapshot(path):
    """Returns the raw saved dict (state_dict, trunk_hidden, optional
    mulligan_state_dict/mulligan_hidden) -- unlike load_deck_checkpoint, does
    NOT build the net itself: the caller needs trunk_hidden to construct a
    DeckNetwork of the right shape BEFORE state_dict can be loaded into it
    (see rl.league.LeaguePool.load_snapshot_agent)."""
    return torch.load(path, weights_only=True, map_location="cpu")  # device-agnostic, see load_deck_checkpoint


def trunk_hidden_from_deck_checkpoint(path):
    """The DeckNetwork trunk widths a live.pt was saved with, read straight off
    its tensor shapes. Returns None if the file does not exist.

    save_deck_checkpoint stores only {"net": state_dict} -- unlike save_snapshot,
    which records "trunk_hidden" explicitly -- so a caller that must CONSTRUCT
    the net before loading into it has no other way to learn the shape. Inferring
    it works for checkpoints written before this function existed, which storing
    a new field would not, and keeps one mechanism instead of two.

    Needed because trunk_hidden became per-league configurable (2026-08-15, the
    capacity experiment): a loader can no longer assume DeckNetwork's default,
    and cross-league loads -- _run_eval_vs_gauntlet pulling another population's
    live.pt -- may legitimately face a different width than the running league's.
    """
    if not os.path.exists(path):
        return None
    sd = torch.load(path, map_location="cpu", weights_only=False)["net"]
    widths, i = [], 0
    while f"trunk_layers.{i}.weight" in sd:
        widths.append(sd[f"trunk_layers.{i}.weight"].shape[0])
        i += 1
    assert widths, f"{path} has no trunk_layers.* -- not a DeckNetwork checkpoint?"
    return tuple(widths)
