"""league_runner's small helpers that had no direct test (plan P0/P5): the mulligan flush, the event-log writer and its JSON
fallback, and the snapshot step's bookkeeping."""
import json

import pytest

from rl.league import league_runner as R


class _Obj:
    name = "Lightning Bolt"


class _Anon:
    def __repr__(self):
        return "<anon>"


def test_json_default_uses_a_name_when_there_is_one_and_repr_otherwise(capsys):
    assert R._json_default(_Obj()) == "Lightning Bolt"
    assert R._json_default(_Anon()) == "<anon>"
    assert "non-serializable" in capsys.readouterr().out


def test_event_log_is_one_compact_document_with_meta_and_per_game_entries(tmp_path):
    path = tmp_path / "sub" / "log.json"
    R._write_event_log(str(path), [[{"e": 1, "card": _Obj()}], [{"e": 2}]], {"who": "t"}, game_pairings=[("a", "b"), ("c", "d")])
    doc = json.loads(path.read_text())
    assert doc["meta"] == {"who": "t"} and [g["game_index"] for g in doc["games"]] == [0, 1]
    assert doc["games"][0]["deck_a"] == "a" and doc["games"][1]["deck_b"] == "d"
    assert doc["games"][0]["events"] == [{"e": 1, "card": "Lightning Bolt"}]
    assert "\n" not in path.read_text(), "compact: no pretty-printing"


def test_event_log_without_pairings_has_no_deck_fields(tmp_path):
    path = tmp_path / "log.json"
    R._write_event_log(str(path), [[{"e": 1}]], {})
    assert "deck_a" not in json.loads(path.read_text())["games"][0]


def _flush_ctx(tmp_path, train_mulligan=True, decks=("a", "b")):
    return {"train_mulligan": train_mulligan, "train_decks": list(decks), "league_dir": str(tmp_path), "session": 3,
            "mulligan_nets": dict.fromkeys(decks, "net"), "mulligan_optimizers": dict.fromkeys(decks, "opt")}


def test_flush_steps_each_deck_that_has_transitions_logs_it_and_clears_the_accumulator(tmp_path, monkeypatch):
    calls = []
    monkeypatch.setattr(R, "mulligan_update", lambda net, opt, tr: (calls.append((net, opt, len(tr))), {"n": len(tr), "loss": 0.5})[1])
    accum = {"a": [1, 2, 3], "b": []}
    R._flush_mulligan(_flush_ctx(tmp_path), accum, iteration=R.MULLIGAN_UPDATE_EVERY - 1, n_iterations=100, cumulative=40)
    assert calls == [("net", "opt", 3)], "a deck with nothing accumulated is not stepped"
    assert accum == {"a": [], "b": []}
    rec = json.loads((tmp_path / "metrics.jsonl").read_text().splitlines()[0])
    assert rec["kind"] == "mulligan" and rec["deck"] == "a" and rec["n"] == 3 and rec["cumulative_games"] == 40


def test_flush_does_nothing_when_not_due_and_keeps_what_has_accumulated(tmp_path, monkeypatch):
    monkeypatch.setattr(R, "mulligan_update", lambda *a: pytest.fail("not due: must not step"))
    accum = {"a": [1], "b": [2]}
    R._flush_mulligan(_flush_ctx(tmp_path), accum, iteration=1, n_iterations=100, cumulative=0)
    assert accum == {"a": [1], "b": [2]} and not (tmp_path / "metrics.jsonl").exists()


def test_flush_on_the_last_iteration_empties_the_accumulators_so_nothing_is_lost_at_a_session_end(tmp_path, monkeypatch):
    seen = []
    monkeypatch.setattr(R, "mulligan_update", lambda net, opt, tr: (seen.append(len(tr)), {"n": len(tr), "loss": 0.0})[1])
    accum = {"a": [1, 2], "b": [3]}
    R._flush_mulligan(_flush_ctx(tmp_path), accum, iteration=2, n_iterations=3, cumulative=6)
    assert sorted(seen) == [1, 2] and accum == {"a": [], "b": []}


def test_flush_is_off_when_the_mulligan_is_not_training(tmp_path, monkeypatch):
    monkeypatch.setattr(R, "mulligan_update", lambda *a: pytest.fail("mulligan frozen: must not step"))
    accum = {"a": [1], "b": [2]}
    R._flush_mulligan(_flush_ctx(tmp_path, train_mulligan=False), accum, iteration=7, n_iterations=8, cumulative=0)
    assert accum == {"a": [1], "b": [2]}


class _Pool:
    def __init__(self):
        self.registered, self.saved, self.snapshots = [], 0, {"a": [], "b": []}

    def register_snapshot(self, name, net, mnet):
        self.registered.append((name, net, mnet))

    def save_opponent_stats(self):
        self.saved += 1


def _snap_ctx(tmp_path, **over):
    ctx = {"games_per_iteration": 10, "cumulative_games": 90, "matchup": None, "pool": _Pool(), "league_dir": str(tmp_path),
           "train_decks": ["a"], "deck_names": ["a", "b"], "live_nets": {"a": "na", "b": "nb"}, "mulligan_nets": {"a": "ma", "b": "mb"},
           "optimizers": {}, "mulligan_optimizers": {}, "session": 0, "session_path": str(tmp_path / "session.txt")}
    ctx.update(over)
    return ctx


def test_a_due_snapshot_registers_only_the_train_decks_and_advances_progress(tmp_path, monkeypatch):
    saved = []
    monkeypatch.setattr(R, "_save_live_checkpoints", lambda *a, **k: saved.append(1))
    ctx = _snap_ctx(tmp_path)
    R._snapshot_if_due(ctx, iteration=0, snapshot_every=10)  # 90 -> 100 crosses the 100-game cadence
    assert ctx["pool"].registered == [("a", "na", "ma")], "only train decks change, so only they get a snapshot"
    assert saved == [1] and ctx["pool"].saved == 1
    assert R._load_progress(str(tmp_path))["cumulative_games_per_deck"] == 100


def test_a_snapshot_that_is_not_due_touches_nothing(tmp_path, monkeypatch):
    monkeypatch.setattr(R, "_save_live_checkpoints", lambda *a, **k: pytest.fail("not due"))
    ctx = _snap_ctx(tmp_path, cumulative_games=0)
    R._snapshot_if_due(ctx, iteration=0, snapshot_every=10)
    assert ctx["pool"].registered == [] and not (tmp_path / "progress.json").exists()


def test_a_matchup_session_never_snapshots(tmp_path, monkeypatch):
    monkeypatch.setattr(R, "_save_live_checkpoints", lambda *a, **k: pytest.fail("matchup: no snapshots"))
    ctx = _snap_ctx(tmp_path, matchup=("a", "b"))
    R._snapshot_if_due(ctx, iteration=0, snapshot_every=10)
    assert ctx["pool"].registered == []
