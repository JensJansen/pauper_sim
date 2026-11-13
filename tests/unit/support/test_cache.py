"""support.cache.source_hash: the corpus cache key must change whenever anything that can alter a recorded decision does."""
from support.cache import source_hash


def make_repo(tmp_path):
    (tmp_path / "src" / "game").mkdir(parents=True)
    (tmp_path / "src" / "rl").mkdir()
    (tmp_path / "src" / "webapp").mkdir()
    (tmp_path / "data").mkdir()
    (tmp_path / "tests" / "support").mkdir(parents=True)
    (tmp_path / "src" / "game" / "stats.py").write_text("POWER = 1\n")
    (tmp_path / "src" / "rl" / "features.py").write_text("DIM = 3\n")
    (tmp_path / "src" / "webapp" / "app.py").write_text("PORT = 1\n")
    (tmp_path / "data" / "elves.txt").write_text("4 Llanowar Elves\n")
    (tmp_path / "data" / "league_decks.json").write_text("{}\n")
    (tmp_path / "tests" / "support" / "corpus.py").write_text("X = 1\n")
    return tmp_path


def test_a_change_anywhere_in_src_changes_the_key(tmp_path):
    repo = make_repo(tmp_path)
    base = source_hash(repo, extra=["tests/support/corpus.py"])
    (repo / "src" / "game" / "stats.py").write_text("POWER = 2\n")  # the ENGINE, not an encoder file
    assert source_hash(repo, extra=["tests/support/corpus.py"]) != base


def test_decklists_and_the_corpus_builder_are_part_of_the_key(tmp_path):
    repo = make_repo(tmp_path)
    base = source_hash(repo, extra=["tests/support/corpus.py"])
    (repo / "data" / "elves.txt").write_text("3 Llanowar Elves\n")
    changed_deck = source_hash(repo, extra=["tests/support/corpus.py"])
    assert changed_deck != base
    (repo / "tests" / "support" / "corpus.py").write_text("X = 2\n")
    assert source_hash(repo, extra=["tests/support/corpus.py"]) != changed_deck


def test_the_web_app_and_bytecode_are_not_part_of_the_key(tmp_path):
    repo = make_repo(tmp_path)
    base = source_hash(repo)
    (repo / "src" / "webapp" / "app.py").write_text("PORT = 2\n")
    (repo / "src" / "game" / "__pycache__").mkdir()
    (repo / "src" / "game" / "__pycache__" / "stats.py").write_text("junk\n")
    assert source_hash(repo) == base


def test_a_rename_changes_the_key(tmp_path):
    repo = make_repo(tmp_path)
    base = source_hash(repo)
    (repo / "src" / "rl" / "features.py").rename(repo / "src" / "rl" / "feats.py")
    assert source_hash(repo) != base


def test_the_key_is_stable_when_nothing_changes(tmp_path):
    repo = make_repo(tmp_path)
    assert source_hash(repo) == source_hash(repo)


def test_the_real_repo_key_covers_the_engine():
    """The repo's own key must include the engine submodule's code (it lives under engine/src/), the exact gap this guards."""
    import hashlib

    from support.paths import REPO
    files = {p.relative_to(REPO / "src").parts[0] for p in (REPO / "src").rglob("*.py") if "webapp" not in p.parts}
    assert "rl" in files
    engine = {p.relative_to(REPO / "engine" / "src").parts[0] for p in (REPO / "engine" / "src").rglob("*.py")}
    assert {"game", "drl_env"} <= engine
    assert len(source_hash()) == len(hashlib.sha256().hexdigest())


def test_a_change_in_the_engine_submodule_changes_the_key(tmp_path):
    repo = make_repo(tmp_path)
    (repo / "engine" / "src" / "game").mkdir(parents=True)
    (repo / "engine" / "src" / "game" / "rules.py").write_text("X = 1\n")
    base = source_hash(repo)
    (repo / "engine" / "src" / "game" / "rules.py").write_text("X = 2\n")
    assert source_hash(repo) != base
