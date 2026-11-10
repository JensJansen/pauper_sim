"""rl.league.metrics: the single writer/reader of metrics.jsonl and the registry that documents every kind."""
import re
from pathlib import Path

import validation
from rl.league import metrics

_REPO = Path(__file__).resolve().parents[3]


def test_append_then_read_round_trips_and_skips_truncated_lines(tmp_path):
    metrics.append(str(tmp_path), kind="ppo", deck="elves", cumulative_games=60)
    metrics.append(str(tmp_path), kind="mulligan", deck="elves", cumulative_games=60)
    with open(tmp_path / metrics.METRICS_FILE, "a") as f:
        f.write('{"kind": "ppo", "deck": "tru')  # an interrupted append
    assert [r["kind"] for r in metrics.read(str(tmp_path))] == ["ppo", "mulligan"]
    assert [r["deck"] for r in metrics.read(str(tmp_path), kinds={"mulligan"})] == ["elves"]
    assert list(metrics.read(str(tmp_path / "nowhere"))) == []


def test_an_unregistered_kind_is_written_anyway_but_warns_once(tmp_path, capsys, monkeypatch):
    monkeypatch.setattr(metrics, "_warned", set())
    metrics.append(str(tmp_path), kind="brand_new", cumulative_games=1)
    metrics.append(str(tmp_path), kind="brand_new", cumulative_games=2)
    assert capsys.readouterr().out.count("brand_new") == 1
    assert len(list(metrics.read(str(tmp_path)))) == 2


def test_every_kind_the_code_emits_is_registered():
    emitted = {c.NAME for c in validation.CHECKS if c.NAME != "primary_round_robin_review"}  # the review check writes no metric
    for folder in ("src/validation", "src/rl"):
        for path in (_REPO / folder).rglob("*.py"):
            emitted |= set(re.findall(r'kind="([a-z_]+)"', path.read_text(encoding="utf-8")))
    assert emitted <= set(metrics.KINDS), f"unregistered kinds: {sorted(emitted - set(metrics.KINDS))}"


def test_readme_documents_every_live_kind():
    readme = (_REPO / "README.md").read_text(encoding="utf-8")
    missing = [k for k, spec in metrics.KINDS.items() if not spec.get("retired") and f"`{k}`" not in readme]
    assert not missing, f"README's metrics reference omits: {missing}"
