"""Session fixtures for the encoder suite. The corpus (real decisions from real games, see encoder_corpus.py) is built
once per session and cached on disk, keyed by a hash of the source it depends on (so editing the encoder code, the
agent, or this oracle rebuilds it). Knobs: ENCODER_CORPUS_GAMES (games per deck pair, default 1),
ENCODER_CORPUS_DECKS (comma list, default every deck), ENCODER_CORPUS_NOCACHE=1."""
import hashlib
import os
import pickle

import pytest

from support import corpus as ec
from support.fixtures import _cwd_in_src  # noqa: F401  (these tests rely on cwd=src/)

_DEPENDS = ["src/rl/model/features.py", "src/rl/model/card_behavior.py", "src/rl/model/arch.py", "src/rl/model/deck.py",
            "src/rl/decision/agent.py", "src/rl/decision/action_bridge.py", "src/rl/roster.py", "tests/support/corpus.py"]
CACHE_DIR = ec.REPO / "tests" / ".cache"


def _cache_path(games):
    h = hashlib.sha256()
    for rel in _DEPENDS:
        h.update((ec.REPO / rel).read_bytes())
    h.update(f"{games}|{os.environ.get('ENCODER_CORPUS_DECKS', '')}".encode())
    return CACHE_DIR / f"encoder_corpus_{h.hexdigest()[:16]}.pkl"


@pytest.fixture(scope="session")
def corpus():
    games = int(os.environ.get("ENCODER_CORPUS_GAMES", "1"))
    path = _cache_path(games)
    use_cache = not os.environ.get("ENCODER_CORPUS_NOCACHE")
    if use_cache and path.exists():
        frames = pickle.loads(path.read_bytes())
        prev = os.getcwd()
        os.chdir(ec.SRC)
        try:
            vocab = ec.build_pool(vocab_path=None)[1]
        finally:
            os.chdir(prev)
        return frames, vocab
    frames, vocab, _agents = ec.build_corpus(games_per_pair=games)
    if use_cache:
        CACHE_DIR.mkdir(exist_ok=True)
        path.write_bytes(pickle.dumps(frames))
    return frames, vocab


@pytest.fixture(scope="session")
def frames(corpus):
    return corpus[0]


@pytest.fixture(scope="session")
def vocab(corpus):
    return corpus[1]
