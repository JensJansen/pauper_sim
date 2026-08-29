"""Self-check for validation.mulligan_audit.run(): end-to-end, real game
engine + torch, same pattern as test_round_robin_review.py's own tests --
mulligan_audit needs a loaded MulliganNet for its probe-hand pass, not just
parsed game logs. Pins the by_land_count/probe_hands split between
write_deck_json (keeps them) and append_metric (drops them, see the
module's own comment) that the JSONL-consolidation review flagged as
unverified by any test."""
from pathlib import Path

import pytest

from rl import checkpoint as ckpt_io
from rl.league import league_runner
from rl.model.mulligan import MulliganNet
from rl.roster import build_pool as _real_build_pool
from validation import _common, mulligan_audit

_SRC_DIR = Path(__file__).resolve().parent.parent.parent / "src"
LAND = "Forest"
SPELL = "Llanowar Elves"


def _game(seat, hand_lands, chosen, p_keep, winner):
    """One synthetic game: `seat` draws hand_lands lands plus one spell,
    makes a single mulligan_keep decision, then ends. Same shape
    tests/test_mulligan_common.py's own _game helper builds -- audit_land_counts
    only parses these events, it doesn't need them to match the attributed
    deck's real decklist."""
    cards = [LAND] * hand_lands + [SPELL]
    return [
        {"kind": "zone_move", "active_idx": seat, "reason": "draw", "to_zone": "hand", "cards": cards},
        {"kind": "decision_weights", "active_idx": seat, "network": "mulligan_keep", "chosen_index": chosen,
         "candidates": [{"fixed_label": "Keep", "probability": p_keep}]},
        {"kind": "game_over", "winner": winner},
    ]


def _make_ctx(tmp_path, monkeypatch, deck):
    monkeypatch.chdir(_SRC_DIR)
    monkeypatch.setattr(_common, "CHECKPOINTS_DIR", tmp_path)
    decklists, vocab, deck_ctxs, fixed_tables = _real_build_pool(vocab_path=str(tmp_path / "vocab.json"))

    net = league_runner.build_deck_net(vocab.size, len(fixed_tables[deck]), trunk_hidden=(24, 24))
    ckpt_io.save_deck_checkpoint(str(tmp_path / "primary" / deck / "live.pt"), net)
    ckpt_io.save_deck_checkpoint(str(tmp_path / "primary" / deck / "mulligan.pt"), MulliganNet(net.encoder))

    return _common.ValidationContext(
        primary_league_name="primary", train_decks=[deck], decklists=decklists, vocab=vocab,
        deck_ctxs=deck_ctxs, fixed_tables=fixed_tables, games_per_check=1, seed=1, cumulative_games=100,
        collected_game_logs=[_game(0, hand_lands=2, chosen=0, p_keep=0.8, winner=0)],
        collected_deck_league=[{0: ("primary", deck)}],
    )


@pytest.mark.slow
def test_run_keeps_by_land_count_and_probe_hands_in_write_deck_json_but_strips_them_from_metrics(
        tmp_path, monkeypatch):
    ctx = _make_ctx(tmp_path, monkeypatch, "elves")

    result = mulligan_audit.run(ctx)

    assert result["decks"] == 1

    deck_path = Path(f"{ctx.primary_league_dir}/elves/checks/mulligan_audit_100games.json")
    import json
    deck_payload = json.loads(deck_path.read_text())
    assert "by_land_count" in deck_payload and deck_payload["by_land_count"]
    assert "probe_hands" in deck_payload and deck_payload["probe_hands"]
    # The natural-game audit bucket for the 2-land hand played above.
    assert deck_payload["by_land_count"]["2"]["kept"] == 1

    metrics_lines = Path(f"{ctx.primary_league_dir}/metrics.jsonl").read_text().splitlines()
    records = [json.loads(l) for l in metrics_lines]
    mulligan_record = next(r for r in records if r["kind"] == "mulligan_audit")
    assert "by_land_count" not in mulligan_record
    assert "probe_hands" not in mulligan_record
    assert mulligan_record["decisions"] == 1
    assert mulligan_record["mulligan_rate"] == 0.0


def test_run_skips_cleanly_with_no_collected_games(tmp_path, monkeypatch):
    monkeypatch.setattr(_common, "CHECKPOINTS_DIR", tmp_path)
    ctx = _common.ValidationContext(
        primary_league_name="primary", train_decks=["elves"], decklists={}, vocab=None,
        deck_ctxs={}, fixed_tables={}, games_per_check=1, seed=1, cumulative_games=100,
    )

    result = mulligan_audit.run(ctx)

    assert "skipped" in result
