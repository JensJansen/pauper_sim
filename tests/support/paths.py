"""Where things are, computed once. Tests import these instead of doing `Path(__file__).parent...` arithmetic, which
breaks the moment a file moves between tiers."""
import pathlib

REPO = pathlib.Path(__file__).resolve().parents[2]
SRC = REPO / "src"
DATA_DIR = REPO / "data"
CACHE_DIR = REPO / "tests" / ".cache"
