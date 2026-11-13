"""Test the tests. Plants ONE deliberate bug at a time in a scratch COPY of src/ and engine/src/ (the real tree is never touched) and
runs a component's tests against it; a mutation that no test notices is a hole in that component's suite.

    python tests/support/mutation_sweep.py <component>             # every mutation
    python tests/support/mutation_sweep.py <component> 3 7         # only these numbers
    python tests/support/mutation_sweep.py <component> --baseline  # the unmutated copy must pass (harness sanity check)

A component opts in by providing tests/component/<component>/mutations.py with COMPONENT, TEST_PATHS (ordered fast
first), ENV (extra environment for the run) and MUTATIONS (see tests/component/encoder/mutations.py). The suite stops at
the first failure, so a caught mutation costs seconds and only an uncaught one runs everything. A run that exceeds the
component's TIMEOUT (default 1500 s) counts as caught: a mutation that makes the code hang is noticed by the hang."""
import importlib.util
import os
import pathlib
import shutil
import subprocess
import sys
import tempfile
import time

REPO = pathlib.Path(__file__).resolve().parents[2]


def load(component):
    path = REPO / "tests" / "component" / component / "mutations.py"
    if not path.exists():
        raise SystemExit(f"no planted-bug list for component {component!r}: expected {path}")
    spec = importlib.util.spec_from_file_location(f"mutations_{component}", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def run(module, mutation_idx=None):
    scratch = pathlib.Path(tempfile.mkdtemp(prefix="mut_"))
    src = scratch / "src"
    shutil.copytree(REPO / "src", src, ignore=shutil.ignore_patterns("__pycache__", "webapp", "*.pyc"))
    engine_src = scratch / "engine_src"  # the engine submodule's code: a mutation path starting "engine/src/" plants a bug there
    shutil.copytree(REPO / "engine" / "src", engine_src, ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
    desc = "baseline (no mutation)"
    if mutation_idx is not None:
        rel, old, new, desc = module.MUTATIONS[mutation_idx]
        path = engine_src / rel[len("engine/src/"):] if rel.startswith("engine/src/") else src / rel
        text = path.read_text(encoding="utf-8")
        assert text.count(old) == 1, f"mutation {mutation_idx}: target text found {text.count(old)}x in {rel}"
        path.write_text(text.replace(old, new), encoding="utf-8")
    # scratch src first (the code under mutation), then the real tests/ so `support` imports; the repo ini's own
    # pythonpath (the REAL src) is overridden to empty below
    env = dict(os.environ, PYTHONDONTWRITEBYTECODE="1", PYTHONPATH=os.pathsep.join([str(src), str(engine_src), str(REPO / "tests")]), **module.ENV)
    t = time.time()
    try:
        r = subprocess.run([sys.executable, "-m", "pytest", *module.TEST_PATHS, "-x", "-q", "-p", "no:cacheprovider",
                            "-o", "pythonpath="], check=False, cwd=REPO, env=env, capture_output=True, text=True,
                           timeout=getattr(module, "TIMEOUT", 1500))
    except subprocess.TimeoutExpired:
        # A planted bug that makes the code loop or explode (a creature declared as an attacker forever) is noticed by the
        # suite hanging; real turns have no iteration cap, so a hang IS the symptom. Counted as caught, and said so.
        shutil.rmtree(scratch, ignore_errors=True)
        return 1, desc, "TIMEOUT (the mutated code hangs or blows up the search)", round(time.time() - t), []
    out = (r.stdout + r.stderr).splitlines()
    # a test FAILED, or the mutated code made a test/fixture ERROR at run time (exit 1 either way). A collection/import
    # problem is exit code 2 and is rejected below.
    failed = next((ln for ln in out if ln.startswith(("FAILED", "ERROR tests"))), None)
    shutil.rmtree(scratch, ignore_errors=True)
    # pytest exit codes: 0 all passed, 1 some test FAILED. Anything else (2 interrupted/collection error, 3 internal, 4
    # usage, 5 nothing collected) means the harness or the environment is broken -- never count that as a caught bug.
    if r.returncode not in (0, 1) or (r.returncode == 1 and failed is None):
        raise SystemExit(f"HARNESS ERROR (pytest exit {r.returncode}) running {desc!r}:" + "\n" + "\n".join(out[-25:]))
    return r.returncode, desc, failed, round(time.time() - t), out[-3:]


def main(argv):
    if not argv:
        raise SystemExit(__doc__)
    module, args = load(argv[0]), argv[1:]
    if args == ["--baseline"]:
        rc, desc, failed, secs, tail = run(module, None)
        print(f"{desc}: {'PASS' if rc == 0 else 'FAIL'} in {secs}s", *tail, sep="\n  ")
        return rc
    todo = [int(a) for a in args] or range(len(module.MUTATIONS))
    survivors = []
    for i in todo:
        rc, desc, failed, secs, _tail = run(module, i)
        verdict = "CAUGHT  " if rc != 0 else "SURVIVED"
        print(f"[{i:2d}] {verdict} {secs:4d}s  {desc}" + (f"\n         by {failed.split('::', 1)[-1][:110]}" if failed else ""), flush=True)
        if rc == 0:
            survivors.append((i, desc))
    print(f"\n{len(todo) - len(survivors)}/{len(todo)} mutations caught")
    for i, d in survivors:
        print(f"  SURVIVED [{i}] {d}")
    return 1 if survivors else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
