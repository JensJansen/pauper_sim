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
    for attempt in range(_SAVE_RETRY_ATTEMPTS):
        try:
            torch.save(obj, path)
            return
        except (RuntimeError, OSError):
            if attempt == _SAVE_RETRY_ATTEMPTS - 1:
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


def save_deck_checkpoint(path, net, optimizer=None):
    """Writes net's (and, if given, optimizer's) state to path as
    {"net": ...[, "optimizer": ...]} -- the schema shared by run_league.py's
    per-deck live.pt (DeckNetwork+Adam) and mulligan.pt (MulliganNet+Adam)."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    saved = {"net": net.state_dict()}
    if optimizer is not None:
        saved["optimizer"] = optimizer.state_dict()
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
    ckpt = torch.load(path, weights_only=True)
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
    saved = {"state_dict": net.state_dict(), "trunk_hidden": trunk_hidden}
    if mulligan_net is not None:
        saved["mulligan_state_dict"] = mulligan_net.state_dict()
        saved["mulligan_hidden"] = mulligan_net.trunk[0].out_features  # restore the exact hidden width on load
    _save_with_retry(saved, path)


def load_snapshot(path):
    """Returns the raw saved dict (state_dict, trunk_hidden, optional
    mulligan_state_dict/mulligan_hidden) -- unlike load_deck_checkpoint, does
    NOT build the net itself: the caller needs trunk_hidden to construct a
    DeckNetwork of the right shape BEFORE state_dict can be loaded into it
    (see rl.league.LeaguePool.load_snapshot_agent)."""
    return torch.load(path, weights_only=True)


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
