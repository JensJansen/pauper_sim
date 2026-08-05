"""Single source of truth for this repo's directory layout.

Lives directly in src/, so anything that can already `import repo_paths`
(i.e. already has src/ on sys.path) needs no separate sys.path surgery for
THIS import -- callers that also need OTHER src/ modules by bare name (e.g.
run_league) may still need their own sys.path.insert(0, SRC_DIR) for that.

Dependency-free (stdlib only) so importing it never drags in torch/etc.
"""
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
SRC_DIR = REPO_ROOT / "src"
CHECKPOINTS_DIR = REPO_ROOT / "checkpoints"
