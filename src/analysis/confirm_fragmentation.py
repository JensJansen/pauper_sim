"""Confirm (or refute) that the league parent's memory growth is pymalloc
arena fragmentation rather than object retention.

Background (2026-08-19). The parent grows ~690MB per deck-round while its
collect workers stay flat. Three things are already established:

  * live torch tensors stay at ~21MB          -> not tensors
  * gc-tracked objects stay flat (~640-940k)  -> not retained containers
  * mull_by_deck_accum holds ~1.2k transitions -> not the mulligan accumulator

and tracemalloc attributes ~16.7 MILLION objects per deck-round, averaging 32
bytes, to multiprocessing.Connection.recv's unpickle path -- the rollout
buffers arriving from the workers as individual Python floats/ints rather than
as compact arrays.

That combination POINTS AT fragmentation: allocate and free tens of millions of
tiny objects each round and pymalloc's 256KB arenas end up sparsely occupied.
An arena returns to the OS only when every block in it is free, which at that
granularity effectively never happens -- so RSS climbs monotonically while the
set of LIVE objects does not.

Pointing at is not proof, so this measures the three quantities that
discriminate:

    dRSS        -- what the OS thinks the process holds
    dTRACED     -- bytes tracemalloc still considers LIVE
    dARENAS     -- pymalloc arenas currently allocated (256KB each in CPython
                   3.10), scraped from sys._debug_mallocstats()

Interpretation:
  dRSS ~= dTRACED                      -> retention; a live object graph is
                                          holding it and the leak hunt goes on.
  dRSS >> dTRACED and arenas track RSS -> fragmentation confirmed; the fix is
                                          to stop shipping millions of tiny
                                          objects per round.
  dRSS >> dTRACED and arenas do NOT    -> the growth is outside pymalloc
                                          (a C-extension allocation), and the
                                          hunt moves there.

tracemalloc runs with a 1-frame trace here: the earlier 20-frame run cost
several GB of its own and made RSS meaningless.
"""
import gc
import io
import os
import re
import sys
import tracemalloc

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import rl.league_runner as league_runner  # noqa: E402

ARENA_BYTES = 256 * 1024  # CPython 3.10 ARENA_SIZE


def _rss_bytes():
    """WorkingSetSize via psapi. argtypes/restype must be declared -- without
    them the HANDLE is truncated on 64-bit and the call silently returns 0
    (which is exactly what the first version of this probe did)."""
    import ctypes
    from ctypes import wintypes

    class PMC(ctypes.Structure):
        _fields_ = [
            ("cb", wintypes.DWORD), ("PageFaultCount", wintypes.DWORD),
            ("PeakWorkingSetSize", ctypes.c_size_t), ("WorkingSetSize", ctypes.c_size_t),
            ("QuotaPeakPagedPoolUsage", ctypes.c_size_t), ("QuotaPagedPoolUsage", ctypes.c_size_t),
            ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t), ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
            ("PagefileUsage", ctypes.c_size_t), ("PeakPagefileUsage", ctypes.c_size_t),
        ]

    psapi = ctypes.WinDLL("psapi")
    psapi.GetProcessMemoryInfo.argtypes = [wintypes.HANDLE, ctypes.POINTER(PMC), wintypes.DWORD]
    psapi.GetProcessMemoryInfo.restype = wintypes.BOOL
    kernel32 = ctypes.WinDLL("kernel32")
    kernel32.GetCurrentProcess.restype = wintypes.HANDLE

    counters = PMC()
    counters.cb = ctypes.sizeof(counters)
    if not psapi.GetProcessMemoryInfo(kernel32.GetCurrentProcess(), ctypes.byref(counters), counters.cb):
        return 0
    return counters.WorkingSetSize


_ARENA_RE = re.compile(r"arenas allocated current\s*=\s*(\d+)")


def _arenas_allocated():
    """sys._debug_mallocstats() writes to the C-level stderr, so it cannot be
    captured by reassigning sys.stderr. Redirect fd 2 to a temp file instead."""
    import tempfile

    saved_fd = os.dup(2)
    try:
        with tempfile.TemporaryFile(mode="w+") as tmp:
            os.dup2(tmp.fileno(), 2)
            try:
                sys._debugmallocstats()
            finally:
                os.dup2(saved_fd, 2)
            tmp.seek(0)
            match = _ARENA_RE.search(tmp.read())
            return int(match.group(1)) if match else -1
    finally:
        os.close(saved_fd)


def main():
    tracemalloc.start(1)
    state = {"n": 0, "base_rss": None, "base_traced": None, "base_arenas": None}
    original = league_runner.collect_rollout_league_parallel

    def instrumented(*args, **kwargs):
        result = original(*args, **kwargs)
        state["n"] += 1
        gc.collect()  # anything merely awaiting collection must not be blamed

        rss = _rss_bytes()
        traced = tracemalloc.get_traced_memory()[0]
        arenas = _arenas_allocated()
        if state["base_rss"] is None:
            state["base_rss"], state["base_traced"], state["base_arenas"] = rss, traced, arenas

        d_rss = (rss - state["base_rss"]) / 1e6
        d_traced = (traced - state["base_traced"]) / 1e6
        d_arenas = arenas - state["base_arenas"]
        arena_mb = d_arenas * ARENA_BYTES / 1e6

        print(
            f"[frag] round {state['n']:3d}  dRSS={d_rss:+8.0f}MB  dTRACED={d_traced:+8.0f}MB  "
            f"dARENAS={d_arenas:+7d} ({arena_mb:+.0f}MB)  unexplained={d_rss - d_traced:+.0f}MB",
            flush=True,
        )
        return result

    league_runner.collect_rollout_league_parallel = instrumented

    import run_league
    run_league.main()


if __name__ == "__main__":
    main()
