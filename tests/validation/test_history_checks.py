"""strength_history / policy_movement / progress_indicator. The indicator's decision rule is a pure function, tested
directly; the three checks are run end to end (slow: real engine + torch) over a tiny league with hand-made
snapshots."""
import json
import random
from pathlib import Path

import pytest
import torch

from evaluation import play, verdicts
from evaluation.probe import record_probe_set, save_probe
from rl import checkpoint as ckpt_io
from rl.decision.agent import SeatAgent
from rl.league import league_runner
from rl.league.league import LeaguePool
from rl.model.mulligan import MulliganNet
from rl.roster import build_pool as _real_build_pool
from validation import _common, policy_movement, progress_indicator, strength_history

_SRC_DIR = Path(__file__).resolve().parent.parent.parent / "src"
DECKS = ["rakdos_madness", "dmir_terror"]


# ---- the indicator's rule ----------------------------------------------------------------------------------

def _own(deck, anchor, wins, games=100):
    return {"deck": deck, "anchor_games": anchor, "wins": wins, "no_winner": 0, "games": games}


def _policy(deck, anchor, js, random_js=0.3):
    return {deck: {"anchors": [{"anchor_games": anchor, "js": js, "top1": 0.5}], "random": {"js": random_js}}}


def test_indicator_warmup_until_a_reference_old_enough_exists():
    rows = [_own("a", 5000, 90)]  # below MIN_REF_GAMES
    assert progress_indicator.assess(rows, {}, 50000)["label"] == "WARMUP"
    assert progress_indicator.assess([_own("a", 40000, 90)], {}, 50000)["label"] == "WARMUP"  # newer than 0.6x
    assert progress_indicator.assess([], {}, 50000)["label"] == "WARMUP"


def test_indicator_reference_is_the_oldest_anchor_within_the_fraction_of_current_age():
    rows = [_own("a", 10000, 60), _own("a", 20000, 55), _own("a", 30000, 50), _own("a", 40000, 50)]
    r = progress_indicator.assess(rows, {}, 60000)  # 0.6 * 60000 = 36000 -> the 30000 anchor
    assert r["decks"]["a"]["ref_games"] == 30000


def test_indicator_labels():
    rows = [_own("a", 20000, 70), _own("b", 20000, 70)]
    assert progress_indicator.assess(rows, {}, 40000)["label"] == "IMPROVING"  # 70% over 200 games: interval above 50
    flat = [_own("a", 20000, 50), _own("b", 20000, 49)]
    assert progress_indicator.assess(flat, {}, 40000)["label"] == "FLAT"  # no probe
    moving = {**_policy("a", 20000, 0.2), **_policy("b", 20000, 0.2)}
    assert progress_indicator.assess(flat, moving, 40000)["label"] == "CHURNING"
    frozen = {**_policy("a", 20000, 0.01), **_policy("b", 20000, 0.01)}
    assert progress_indicator.assess(flat, frozen, 40000)["label"] == "STUCK"
    banner = progress_indicator.banner(progress_indicator.assess(flat, moving, 40000), 40000)
    assert "CHURNING" in banner and "provisional" in banner


# ---- the three checks end to end ----------------------------------------------------------------------------

@pytest.fixture
def league(tmp_path, monkeypatch):
    """Two decks with tiny nets, six snapshots each (ids 0..5 = 10..60 games at 10 games per snapshot), a live
    checkpoint, and a probe file for each deck."""
    monkeypatch.chdir(_SRC_DIR)
    monkeypatch.setattr(league_runner, "HORIZON", 30)
    vocab_path = str(tmp_path / "vocab.json")
    monkeypatch.setattr(play, "build_pool", lambda: _real_build_pool(vocab_path=vocab_path))
    monkeypatch.setattr(play, "_pool_cache", None)
    monkeypatch.setattr(play, "_agent_cache", {})
    monkeypatch.setattr(_common, "CHECKPOINTS_DIR", tmp_path)
    decklists, vocab, deck_ctxs, tables = _real_build_pool(vocab_path=vocab_path)
    league_dir = str(tmp_path / "primary")
    probes = tmp_path / "probes"
    probes.mkdir()
    torch.manual_seed(0)
    for name in DECKS:
        pool = LeaguePool(league_dir, [name], max_snapshots_per_deck=50)
        net = league_runner.build_deck_net(vocab.size, len(tables[name]), trunk_hidden=(24, 24))
        mnet = MulliganNet(net.encoder)
        for _ in range(6):
            for p in net.parameters():
                p.data += 0.05 * torch.randn_like(p)  # a different policy at every snapshot
            pool.register_snapshot(name, net, mnet)
        ckpt_io.save_deck_checkpoint(f"{league_dir}/{name}/live.pt", net)
        ckpt_io.save_deck_checkpoint(f"{league_dir}/{name}/mulligan.pt", mnet)
    for name in DECKS:
        factories = {(d, "a"): (lambda d=d: SeatAgent(*play._load(league_dir, d, 5), deck_ctxs[d])) for d in DECKS}
        probe = record_probe_set(name, factories, decklists, deck_ctxs, 4, 1, 30)
        for e in probe["episodes"]:
            e["probe"] = [True] * len(e["action"])  # a tiny random-play game may never reach a developed board
        save_probe(str(probes / f"probe_{name}.pkl.gz"), probe, vocab, tables[name])
    random.seed(0)
    return {"tmp": tmp_path, "league_dir": league_dir, "probes": str(probes), "decklists": decklists, "vocab": vocab,
            "deck_ctxs": deck_ctxs, "tables": tables}


def _ctx(lg, **overrides):
    return _common.ValidationContext(
        primary_league_name="primary", train_decks=list(DECKS), decklists=lg["decklists"], vocab=lg["vocab"],
        deck_ctxs=lg["deck_ctxs"], fixed_tables=lg["tables"], games_per_check=2, seed=3, cumulative_games=70,
        games_per_snapshot=10, history_ladder=(20, 40), history_games=2, field_games=2,
        probe_states_dir=lg["probes"], **overrides)


@pytest.mark.slow
def test_history_checks_run_end_to_end_and_feed_the_indicator(league):
    ctx = _ctx(league)
    s = strength_history.run(ctx)
    assert s["anchors"] == 2 and "wrote" in s
    payload = json.loads((league["tmp"] / "primary" / "checks" / "strength_history_70games.json").read_text())
    assert len(payload["own"]) == 2 * 2  # decks x anchors
    assert len(payload["field"]) == 2 * 2 * 1  # opponent deck x anchors x the one other deck's current agent
    assert all(r["games"] == 2 for r in payload["own"])

    m = policy_movement.run(ctx)
    assert m["decks"] == 2
    for r in ctx.extras["policy_movement"].values():
        assert [a["anchor_games"] for a in r["anchors"]] == [20, 40]
        assert 0.0 <= r["anchors"][0]["js"] <= 0.7 and r["adjacent"] is not None and r["random"]["js"] > 0

    p = progress_indicator.run(ctx)
    assert p["label"] == "WARMUP"  # 20-40 games is far below MIN_REF_GAMES: the warm-up guard, not an error
    lines = [json.loads(x) for x in (league["tmp"] / "primary" / "metrics.jsonl").read_text().splitlines()]
    assert {"strength_history", "strength_field", "policy_movement", "policy_scale", "progress_indicator"} <= \
        {x["kind"] for x in lines}


@pytest.mark.slow
def test_history_checks_skip_without_their_inputs(league):
    ctx = _ctx(league)
    ctx.games_per_snapshot = None
    assert "skipped" in strength_history.run(ctx) and "skipped" in policy_movement.run(ctx)
    ctx = _ctx(league)
    ctx.probe_states_dir = None
    assert "skipped" in policy_movement.run(ctx)
    ctx.cumulative_games = 25  # nothing is two snapshot intervals older than that except snapshot 0 at 10 games
    assert strength_history.run(ctx).get("anchors", 0) <= 1
    assert "skipped" in progress_indicator.run(_ctx(league))  # strength_history has not run on this ctx


@pytest.mark.slow
def test_reference_cell_gets_extra_games(league, monkeypatch):
    monkeypatch.setattr(verdicts, "MIN_REF_GAMES", 10)  # 0.6 x 70 games = 42: the 40-game anchor is the reference
    ctx = _ctx(league)
    strength_history.run(ctx)
    games = {(r["deck"], r["anchor_games"]): r["games"] for r in ctx.extras["strength_history"]["own"]}
    assert all(g == 2 * progress_indicator.REF_GAMES_FACTOR for (_d, a), g in games.items() if a == 40)
    assert all(g == 2 for (_d, a), g in games.items() if a == 20)
