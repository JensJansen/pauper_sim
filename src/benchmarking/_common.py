"""Path/stdout bootstrap for the benchmarking scripts.

Imported first by every benchmark (before any rl.*/game import) so it can put
src/ on sys.path and chdir into src/, making the `../data` and
`../checkpoints` relative paths every rl.* module uses resolve the same way
they do for run_league.py. Import it for its side effect.
"""

import os
import sys

_SRC = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # .../src
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

from repo_paths import SRC_DIR  # noqa: E402

os.chdir(SRC_DIR)

# Line-buffer stdout so print() flushes even when output is redirected to a
# file. hasattr guard: a wrapped/captured stdout may lack reconfigure.
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(line_buffering=True)
