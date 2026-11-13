"""The records a real session writes carry exactly the fields rl.league.metrics.KINDS documents (plan P11).

Readers (evaluation, validation, report_metrics, the web app mirror) use .get on an append-only file, so a field that is renamed,
dropped or added undocumented breaks them silently. A tiny real session is the only honest source of what training emits."""
import json
import random

import numpy as np
import pytest
import torch

from rl.league import league_runner, metrics
from support.paths import SRC

NUMERIC_PPO = ("policy_loss", "value_loss", "entropy", "approx_kl", "clip_fraction", "explained_variance", "adv_std",
               "policy_loss_mean", "value_loss_mean", "entropy_mean", "approx_kl_mean", "clip_fraction_mean")


@pytest.fixture(scope="module")
def records(tmp_path_factory):
    league = tmp_path_factory.mktemp("schema")
    mp = pytest.MonkeyPatch()
    mp.setattr(league_runner, "HORIZON", 30)
    mp.chdir(SRC)
    try:
        torch.manual_seed(1)
        random.seed(1)
        np.random.seed(1)
        league_runner._run_session(8, 2, 4, None, 1, league_dir=str(league), seed=1, roster=["elves", "boggles"],
                                   trunk_hidden=(8, 8), ppo_hparams={"n_epochs": 1})
    finally:
        mp.undo()
    return [json.loads(line) for line in (league / "metrics.jsonl").read_text().splitlines() if line.strip()]


def _documented(kind):
    names = set()
    for key in metrics.KINDS[kind]["fields"]:
        names |= {part.strip() for part in key.split("/")}
    return names


def _by_kind(records, kind):
    return [r for r in records if r["kind"] == kind]


@pytest.mark.parametrize("kind", ["session_start", "ppo", "mulligan"])
def test_every_field_a_session_writes_is_documented_and_every_documented_field_is_written(records, kind):
    rows = _by_kind(records, kind)
    assert rows, f"the session wrote no {kind!r} record"
    documented = _documented(kind) | {"kind", "cumulative_games", "session", "iteration"}  # shared by every training kind
    for r in rows:
        assert set(r) <= documented, f"{kind}: undocumented field(s) {sorted(set(r) - documented)}"
    written = set().union(*(set(r) for r in rows))
    assert _documented(kind) <= written, f"{kind}: documented but never written: {sorted(_documented(kind) - written)}"


def test_ppo_records_are_one_per_deck_per_iteration_with_finite_numbers_and_the_new_mean_fields(records):
    rows = _by_kind(records, "ppo")
    assert sorted((r["iteration"], r["deck"]) for r in rows) == [(i, d) for i in range(8) for d in ("boggles", "elves")]
    for r in rows:
        for field in NUMERIC_PPO:
            assert isinstance(r[field], float) and np.isfinite(r[field]), (field, r[field])
        assert r["epochs_run"] >= 1 and r["buffer_size"] > 0 and r["games"] == 2
        assert 0.0 <= r["clip_fraction"] <= 1.0 and 0.0 <= r["clip_fraction_mean"] <= 1.0
        assert r["entropy_mean"] > 0 and r["approx_kl_mean"] >= 0


def test_cumulative_games_never_decreases_within_a_deck_and_matches_the_iteration(records):
    for deck in ("elves", "boggles"):
        seq = [r["cumulative_games"] for r in _by_kind(records, "ppo") if r["deck"] == deck]
        assert seq == sorted(seq) and seq[-1] - seq[0] == 2 * 7, "2 games per iteration across 8 iterations"


def test_mulligan_records_appear_on_the_documented_cadence(records):
    its = sorted({r["iteration"] for r in _by_kind(records, "mulligan")})
    assert its == [league_runner.MULLIGAN_UPDATE_EVERY - 1], "one flush at the 8th iteration, which is also the session's last"
