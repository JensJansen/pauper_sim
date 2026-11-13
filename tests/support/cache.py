"""Cache keys for expensive test fixtures (the real-game corpus).

A cached fixture is only valid for the code that produced it, so its key must cover ALL of that code -- not just the
files the fixture is "about". The corpus stores both the encoder's output and the oracle's expectation as computed by
the code of the day; if the engine changes and the cache doesn't, the stale data keeps agreeing with itself and the
tests quietly verify old code. So the key is a hash of every Python file under src/ (except the web app, which is its
own repo) and under engine/src/ (the engine submodule), the decklists the games are built from, and any extra files the fixture names."""
import hashlib

from support.paths import REPO


def source_hash(repo=REPO, extra=()):
    """Hex digest over src/**/*.py and engine/src/**/*.py (minus src/webapp and bytecode caches), data/*.txt, data/league_decks.json and
    `extra` (paths relative to `repo`). File paths are hashed along with contents, so a rename changes the key."""
    src = repo / "src"
    files = sorted(p for p in src.rglob("*.py") if "webapp" not in p.relative_to(src).parts and "__pycache__" not in p.parts)
    engine_src = repo / "engine" / "src"  # the engine submodule: its code shapes every recorded decision too
    files += sorted(p for p in engine_src.rglob("*.py") if "__pycache__" not in p.parts)
    data = repo / "data"
    files += sorted(data.glob("*.txt")) + ([data / "league_decks.json"] if (data / "league_decks.json").exists() else [])
    files += [repo / e for e in extra]
    h = hashlib.sha256()
    for p in files:
        h.update(p.relative_to(repo).as_posix().encode() + b"\0" + p.read_bytes() + b"\0")
    return h.hexdigest()
