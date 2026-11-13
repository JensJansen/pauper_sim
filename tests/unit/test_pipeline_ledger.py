"""Ledger gate (plan P0): every public function and class of the agent pipeline is named by at least one test.

The scope is the non-engine, non-replay source: src/rl, src/evaluation, src/validation and the run_* / config scripts. "Named"
is a word match in any test file -- a coarse floor, not coverage, but it stops a new public entry point arriving with no test at
all. A name that genuinely has none goes in UNTESTED with the reason, so the gap is visible and reviewed rather than silent."""
import ast
import re

from support.paths import REPO

SCOPE_DIRS = ("src/rl", "src/evaluation", "src/validation")
SCOPE_FILES = ("src/run_league.py", "src/run_rollback.py", "src/run_training_pipeline.py", "src/config_loader.py",
               "src/repo_paths.py")
UNTESTED = {}  # "path:name" -> reason; empty on 2026-11-12


def _public_names():
    paths = [p for d in SCOPE_DIRS for p in (REPO / d).rglob("*.py")] + [REPO / f for f in SCOPE_FILES]
    for path in sorted(paths):
        for node in ast.parse(path.read_text(encoding="utf-8")).body:
            if isinstance(node, (ast.FunctionDef, ast.ClassDef)) and not node.name.startswith("_"):
                yield f"{path.relative_to(REPO).as_posix()}:{node.name}", node.name


def test_every_public_function_and_class_of_the_pipeline_is_named_by_a_test():
    tests = "\n".join(p.read_text(encoding="utf-8", errors="ignore") for p in (REPO / "tests").rglob("*.py")
                      if p.name != "test_pipeline_ledger.py")
    missing = [key for key, name in _public_names() if key not in UNTESTED and not re.search(rf"\b{re.escape(name)}\b", tests)]
    assert not missing, "no test names: " + ", ".join(missing) + " (add a test, or list it in UNTESTED with the reason)"


def test_the_untested_ledger_has_no_stale_entries():
    live = {key for key, _name in _public_names()}
    assert set(UNTESTED) <= live, f"entries for names that no longer exist: {sorted(set(UNTESTED) - live)}"
    assert all(UNTESTED.values()), "every entry needs a reason"
