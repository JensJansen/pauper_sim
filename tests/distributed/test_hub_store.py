"""Self-check for distributed.hub_store: push atomicity, plain-append
metrics merging, roster-scoped pulls (never a bare directory listing), and
the fresh-deck seed's real-checkpoint round trip + no-clobber guard.

seed_fresh_deck builds a real (if tiny-vocab) DeckNetwork/MulliganNet via
rl.checkpoint -- same lightweight-fake-object pattern tests/rl/test_checkpoint.py
already uses (a real net, not a mocked one, but no game engine or build_pool()
involved, so this isn't @pytest.mark.slow -- consistent with that file's own
precedent)."""
import json
import os

from distributed import hub_store


class _FakeVocab:
    size = 5


_FIXED_TABLE = [None, None, None]  # only its length matters to build_deck_net


def test_push_deck_files_writes_files_and_swaps_atomically(tmp_path):
    league_dir = str(tmp_path)
    hub_store.push_deck_files(league_dir, "boggles", {"live.pt": b"weights-v1", "opponent_stats.json": b"{}"}, [])
    assert (tmp_path / "boggles" / "live.pt").read_bytes() == b"weights-v1"
    assert (tmp_path / "boggles" / "opponent_stats.json").read_bytes() == b"{}"
    # no leftover temp files
    assert not list((tmp_path / "boggles").glob("*.tmp"))


def test_push_deck_files_leaves_prior_state_untouched_on_a_mid_write_failure(tmp_path, monkeypatch):
    league_dir = str(tmp_path)
    hub_store.push_deck_files(league_dir, "boggles", {"live.pt": b"weights-v1"}, [])

    real_open = open

    def flaky_open(path, mode="r", *a, **kw):
        if isinstance(path, str) and path.endswith("mulligan.pt.tmp") and "wb" in mode:
            raise OSError("simulated connection drop mid-push")
        return real_open(path, mode, *a, **kw)

    monkeypatch.setattr("builtins.open", flaky_open)
    try:
        hub_store.push_deck_files(league_dir, "boggles", {"live.pt": b"weights-v2", "mulligan.pt": b"m"}, [])
        assert False, "expected the simulated write failure to raise"
    except OSError:
        pass
    monkeypatch.undo()

    # the prior live.pt must be untouched -- v2 never got swapped in, since
    # the batch never fully wrote (mulligan.pt failed before any os.replace)
    assert (tmp_path / "boggles" / "live.pt").read_bytes() == b"weights-v1"
    assert not (tmp_path / "boggles" / "mulligan.pt").exists()
    assert not list((tmp_path / "boggles").glob("*.tmp"))


def test_append_metrics_is_a_plain_append(tmp_path):
    league_dir = str(tmp_path)
    hub_store.append_metrics(league_dir, ['{"kind": "ppo", "deck": "boggles"}'])
    hub_store.append_metrics(league_dir, ['{"kind": "mulligan", "deck": "elves"}'])
    lines = (tmp_path / "metrics.jsonl").read_text().splitlines()
    assert [json.loads(ln) for ln in lines] == [
        {"kind": "ppo", "deck": "boggles"},
        {"kind": "mulligan", "deck": "elves"},
    ]


def test_push_deck_files_also_appends_metrics_lines(tmp_path):
    league_dir = str(tmp_path)
    hub_store.push_deck_files(league_dir, "boggles", {"live.pt": b"w"}, ['{"kind": "ppo", "iteration": 0}'])
    assert (tmp_path / "metrics.jsonl").read_text().strip() == '{"kind": "ppo", "iteration": 0}'


def test_pull_full_roster_only_reads_known_deck_files_never_archive_or_checks(tmp_path):
    league_dir = tmp_path
    boggles = league_dir / "boggles"
    boggles.mkdir()
    (boggles / "live.pt").write_bytes(b"live")
    (boggles / "snapshot_5.pt").write_bytes(b"snap5")
    (boggles / "archive").mkdir()
    (boggles / "archive" / "snapshot_0.pt").write_bytes(b"should-never-be-pulled")
    (boggles / "checks").mkdir()
    (boggles / "checks" / "round_robin_50games.json").write_text("{}")
    # a league-level checks/ dir sibling to deck dirs -- must never be
    # mistaken for a deck named "checks"
    (league_dir / "checks").mkdir()
    (league_dir / "checks" / "summary.json").write_text("{}")
    (league_dir / "metrics.jsonl").write_text("")

    roster = hub_store.pull_full_roster(str(league_dir), ["boggles", "elves"])

    assert set(roster["boggles"]) == {"live.pt", "snapshot_5.pt"}
    assert roster["boggles"]["live.pt"] == b"live"
    assert roster["elves"] == {}  # never seeded, no directory yet -- empty, not an error
    assert "checks" not in roster


def test_seed_fresh_deck_writes_a_loadable_checkpoint(tmp_path):
    from rl import checkpoint as ckpt_io
    from rl.model.arch import SetTransformer
    from rl.model.deck import DeckNetwork

    league_dir = str(tmp_path)
    hub_store.seed_fresh_deck(league_dir, "boggles", _FakeVocab(), _FIXED_TABLE)

    live_path = tmp_path / "boggles" / "live.pt"
    mull_path = tmp_path / "boggles" / "mulligan.pt"
    assert live_path.exists() and mull_path.exists()

    trunk_hidden = ckpt_io.trunk_hidden_from_deck_checkpoint(str(live_path))
    encoder = SetTransformer(_FakeVocab.size)
    net = DeckNetwork(encoder, film_condition_dim=encoder.d_model,
                       non_targeting_n_actions=len(_FIXED_TABLE), trunk_hidden=trunk_hidden)
    assert ckpt_io.load_deck_checkpoint(str(live_path), net) is True


def test_seed_fresh_deck_never_overwrites_existing_weights(tmp_path):
    league_dir = str(tmp_path)
    deck_dir = tmp_path / "boggles"
    deck_dir.mkdir()
    (deck_dir / "live.pt").write_bytes(b"real-trained-weights-do-not-touch")

    hub_store.seed_fresh_deck(league_dir, "boggles", _FakeVocab(), _FIXED_TABLE)

    assert (deck_dir / "live.pt").read_bytes() == b"real-trained-weights-do-not-touch"
    assert not (deck_dir / "mulligan.pt").exists()  # no-op means no-op, not "seed the other file"


def _consistent(files):
    """Every push v_i leaves exactly live.pt=b"v<i>" and snapshot_<i>.pt."""
    snaps = [fn for fn in files if fn.startswith("snapshot_")]
    return len(snaps) == 1 and snaps[0] == "snapshot_" + files["live.pt"][1:].decode() + ".pt"


def _hammer(tmp_path):
    import threading
    league = str(tmp_path)
    hub_store.push_deck_files(league, "elves", {"live.pt": b"v0", "snapshot_0.pt": b"x"}, [],
                              snapshot_names=["snapshot_0.pt"])
    stop, bad = threading.Event(), []

    def reader():
        while not stop.is_set():
            files = hub_store.pull_full_roster(league, ["elves"])["elves"]
            if not _consistent(files):
                bad.append(sorted(files))
    t = threading.Thread(target=reader)
    t.start()
    try:
        for i in range(1, 200):
            hub_store.push_deck_files(league, "elves", {"live.pt": f"v{i}".encode(), f"snapshot_{i}.pt": b"x"},
                                      [], snapshot_names=[f"snapshot_{i}.pt"])
    finally:
        stop.set()
        t.join()
    return bad


def test_concurrent_pulls_never_see_a_half_swapped_deck(tmp_path):
    """The hub is threaded: a pull reading while a push swaps/prunes must see
    one push's complete file set for a deck, never a mix."""
    assert _hammer(tmp_path) == []


def test_the_hammer_test_can_actually_detect_a_torn_read(tmp_path, monkeypatch):
    """Sensitivity check for the test above: with the per-deck lock disabled it must fail."""
    import contextlib
    monkeypatch.setattr(hub_store, "_deck_lock", lambda deck: contextlib.nullcontext())
    try:
        torn = _hammer(tmp_path)
    except PermissionError:
        # On Windows (the hub machine) an unlocked push can't even os.replace a file a
        # concurrent pull has open -- the lock also prevents a push crashing with a 500.
        torn = ["os.replace blocked by an open reader"]
    assert torn != []
