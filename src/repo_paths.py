"""Single source of truth for this repo's directory layout.

Lives in src/, so any caller with src/ already on sys.path can `import
repo_paths` directly; importing other src/ modules by bare name may still
need a separate sys.path.insert(0, SRC_DIR). Stdlib-only so importing it
never pulls in torch.
"""
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
SRC_DIR = REPO_ROOT / "src"
CHECKPOINTS_DIR = REPO_ROOT / "checkpoints"
