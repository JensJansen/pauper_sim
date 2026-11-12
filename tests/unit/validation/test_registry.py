"""Self-check for validation.run_all's failure isolation: one check raising
must never stop the others, and every registered check must actually run in
order (mulligan_audit depends on round_robin_primary/round_robin_training
having run first -- see validation/__init__.py's own docstring)."""
import types

import validation


def _fake_check(name, behavior):
    """A minimal stand-in module with just NAME/run(ctx), matching the real
    contract without needing build_pool()/torch/real games."""
    mod = types.SimpleNamespace(NAME=name)
    mod.run = behavior
    return mod


def test_a_failing_check_does_not_stop_the_others(monkeypatch):
    calls = []

    def ok(ctx):
        calls.append("ok")
        return {"status": "fine"}

    def boom(ctx):
        calls.append("boom")
        raise RuntimeError("simulated failure")

    monkeypatch.setattr(validation, "CHECKS", [
        _fake_check("first", ok), _fake_check("broken", boom), _fake_check("last", ok),
    ])

    results = validation.run_all(ctx=object())

    assert calls == ["ok", "boom", "ok"]  # every check ran, "broken" didn't stop "last"
    assert results["first"] == {"status": "fine"}
    assert results["broken"] is None  # the failure is recorded, not silently dropped
    assert results["last"] == {"status": "fine"}


def test_checks_registry_runs_round_robins_before_mulligan_audit():
    """mulligan_audit is a post-hoc analysis of ctx.collected_game_logs,
    which only round_robin_primary/round_robin_training populate -- if
    registry order ever regresses, mulligan_audit would silently see no data."""
    names = [c.NAME for c in validation.CHECKS]
    assert names.index("primary_vs_primary_round_robin") < names.index("mulligan_audit")
    assert names.index("primary_vs_training_round_robin") < names.index("mulligan_audit")
