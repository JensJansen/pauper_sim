"""Path/stdout bootstrap for the benchmarking scripts.

Imported FIRST by every benchmark (before any rl.*/game import) so it can
(1) put src/ on sys.path and (2) chdir into src/, making the `../data` and
`../checkpoints` relative paths every rl.* module already uses resolve the
same way they do for run_league.py -- no matter which directory the benchmark
was launched from. Import it for its side effect (e.g. training_run.py does).
"""

import os
import sys

_SRC = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # .../src
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)  # also makes `import repo_paths` (lives in src/) work below

from repo_paths import SRC_DIR  # noqa: E402

os.chdir(SRC_DIR)  # so build_pool()'s ../data and ../checkpoints resolve like run_league.py's

# Line-buffer stdout so every print() flushes at its newline even when output is
# redirected to a file (block-buffered by default off a TTY -- benchmark progress
# would otherwise stay invisible until the process exits). Covers every script
# that imports _common. hasattr guard: a wrapped/captured stdout may lack it.
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(line_buffering=True)
