"""Shared context and output plumbing for every check in this package.

A check is a module exposing `NAME` (str) and `run(ctx) -> dict` (a short
summary for the orchestrator's own console line -- the check writes its own
detail JSON and metrics.jsonl records itself via the helpers below, so the
orchestrator never needs to know a given check's output shape).
"""
import json
import os
from dataclasses import dataclass, field

from repo_paths import CHECKPOINTS_DIR
from rl import checkpoint as ckpt_io
from rl.decision.agent import SeatAgent
from rl.league.league_runner import _append_metric, build_deck_net
from rl.model.mulligan import MulliganNet
from webapp_mirror import mirror_json


@dataclass
class ValidationContext:
    """Everything a check needs, built once per cadence point by
    run_training_pipeline.py and passed to every check in turn.

    collected_game_logs / collected_deck_league: raw per-game event logs and
    a parallel seat->(league, deck_name) attribution, accumulated by
    round_robin_primary and round_robin_training as they play (registry
    order matters: mulligan_audit runs after both and reads these instead of
    replaying games). ctx's own copy is discarded with the rest of ctx once
    a cadence point's checks finish -- never written to disk itself. A check
    that wants ITS games kept around for replay embeds them directly into
    its own write_league_json payload instead (a "games" key, same shape
    rl.league.league_runner._write_event_log uses) -- see
    round_robin_primary.py and write_league_json's own docstring below on
    why that key is stripped before mirroring.

    executor / n_workers: the SAME ProcessPoolExecutor run_training_pipeline.py
    already builds for training collection, threaded through so a check can
    parallelize its own game-playing across it instead of running single-
    process. Provably idle for a check's whole duration: run_all only ever
    runs synchronously between training chunks, never concurrently with one.
    executor=None (the default) means "run sequentially" -- every check must
    keep working with no executor at all, since standalone callers (run_league.py
    --eval, analysis/ scripts, a check invoked directly in a test) never
    build or pass one."""
    primary_league_name: str
    train_decks: list
    decklists: dict
    vocab: object
    deck_ctxs: dict
    fixed_tables: dict
    games_per_check: int
    seed: object
    cumulative_games: int
    training_league_name: str = None
    collected_game_logs: list = field(default_factory=list)
    collected_deck_league: list = field(default_factory=list)
    executor: object = None
    n_workers: int = 1

    @property
    def primary_league_dir(self):
        return str(CHECKPOINTS_DIR / self.primary_league_name)

    @property
    def training_league_dir(self):
        return str(CHECKPOINTS_DIR / self.training_league_name) if self.training_league_name else None


def _write_json(path, payload):
    # No indent when a "games" key is present: pretty-printing a full event
    # log substantially bloats it and slows the write down for no benefit
    # (nobody hand-reads this file at that size) -- same call
    # rl.league.league_runner._write_event_log already makes for the same
    # reason.
    indent = None if "games" in payload else 2
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(payload, f, indent=indent)


def _strip_games_for_mirror(payload):
    """A check that embeds a "games" key (round_robin_primary -- see its own
    docstring) wants that exact file, unmodified, openable in the webapp's
    replay viewer. But write_league_json/write_deck_json also mirror
    everything they write into the webapp submodule's small, git-committed
    logs/validation/ for /stats (webapp_mirror.mirror_json) -- and a full
    round robin's embedded games can be hundreds of MB, which must never
    land in a public git repo. /stats never reads a "games" key anyway (it
    only ever uses check/decks/pairings/deck_totals/etc.), so dropping it
    from the MIRRORED copy costs /stats nothing while keeping that copy
    small; the full payload, games included, still goes to the real,
    local-only checkpoints/ file untouched."""
    if "games" not in payload:
        return payload
    return {k: v for k, v in payload.items() if k != "games"}


def write_league_json(ctx, check_name, payload):
    """checkpoints/<primary_league>/checks/<check_name>_<games>games.json,
    mirrored into the webapp submodule's logs/validation/ if it's checked
    out (see webapp_mirror.mirror_json) -- minus payload["games"] if
    present, see _strip_games_for_mirror."""
    relative = f"checks/{check_name}_{ctx.cumulative_games}games.json"
    path = f"{ctx.primary_league_dir}/{relative}"
    _write_json(path, payload)
    mirror_json(ctx.primary_league_dir, relative, _strip_games_for_mirror(payload))
    return path


def write_deck_json(ctx, deck_name, check_name, payload):
    """checkpoints/<primary_league>/<deck>/checks/<check_name>_<games>games.json,
    mirrored into the webapp submodule's logs/validation/ if it's checked
    out (see webapp_mirror.mirror_json) -- minus payload["games"] if
    present, see _strip_games_for_mirror."""
    relative = f"{deck_name}/checks/{check_name}_{ctx.cumulative_games}games.json"
    path = f"{ctx.primary_league_dir}/{relative}"
    _write_json(path, payload)
    mirror_json(ctx.primary_league_dir, relative, _strip_games_for_mirror(payload))
    return path


def append_metric(ctx, **fields):
    """One compact record into checkpoints/<primary_league>/metrics.jsonl,
    tagging cumulative_games automatically so callers don't repeat it.
    Unlike league_runner._append_metric itself (which assumes its caller's
    LeaguePool construction already created league_dir), this creates the
    directory if needed -- consistent with write_league_json/write_deck_json
    above, and safe for a check invoked standalone against a league whose
    directory structure isn't otherwise guaranteed to exist yet."""
    os.makedirs(ctx.primary_league_dir, exist_ok=True)
    _append_metric(ctx.primary_league_dir, cumulative_games=ctx.cumulative_games, **fields)


def load_deck_net(league_dir, name, vocab, fixed_table):
    """One deck's current live DeckNetwork + MulliganNet, eval mode, read
    from league_dir/<name>/{live,mulligan}.pt. Shared by every check that
    needs a deck's current weights -- eval only ever needs inference, so no
    optimizer is loaded."""
    live_path = f"{league_dir}/{name}/live.pt"
    net = build_deck_net(vocab.size, len(fixed_table), ckpt_io.trunk_hidden_from_deck_checkpoint(live_path))
    ckpt_io.load_deck_checkpoint(live_path, net)
    net.eval()
    mnet = MulliganNet(net.encoder)
    ckpt_io.load_deck_checkpoint(f"{league_dir}/{name}/mulligan.pt", mnet)
    mnet.eval()
    return net, mnet


def load_agent(league_dir, name, vocab, deck_ctx, fixed_table):
    """A ready-to-play SeatAgent for one deck's current live weights."""
    net, mnet = load_deck_net(league_dir, name, vocab, fixed_table)
    return SeatAgent(net, mnet, deck_ctx)
