"""rl.league.manifest: a run's identity record, extended (never overwritten) by later sessions."""
from rl.league import manifest
from rl.model.arch import SetTransformer
from rl.model.deck import DeckNetwork


def _net(trunk):
    enc = SetTransformer(20)
    return DeckNetwork(enc, film_condition_dim=enc.d_model, non_targeting_n_actions=5, trunk_hidden=trunk)


def _model(trunk):
    return manifest.model_info(_net(trunk), 20, trunk, {"elves": 5})


def test_model_structure_hash_tracks_the_architecture_not_the_weights():
    assert _model((16, 16))["structure_sha"] == _model((16, 16))["structure_sha"]  # fresh random weights, same structure
    assert _model((16, 16))["structure_sha"] != _model((24, 24))["structure_sha"]
    m = _model((16, 16))
    assert m["trunk_hidden"] == [16, 16] and m["vocab_size"] == 20 and m["n_params_example_deck"] > 0


def test_sessions_accumulate_and_changes_are_kept(tmp_path):
    league = str(tmp_path / "run")
    first = manifest.start_session(league, {"lr": 1, "trunk": (16, 16)}, _model((16, 16)), 0, tmp_path, argv=["a"])
    assert first["sessions"][0]["games_start"] == 0 and first["config_history"] == [] and first["model_history"] == []
    manifest.finish_session(league, 600)
    second = manifest.start_session(league, {"lr": 2, "trunk": (16, 16)}, _model((24, 24)), 600, tmp_path, argv=["b"])
    assert [s["games_start"] for s in second["sessions"]] == [0, 600]
    assert second["sessions"][0]["games_end"] == 600 and second["sessions"][0]["finished_at"]
    assert second["sessions"][1]["finished_at"] is None
    assert second["config_history"][0]["config"]["lr"] == 1 and second["config"]["lr"] == 2
    assert second["config"]["trunk"] == [16, 16]  # tuples stored as JSON lists
    assert second["model_history"][0]["model"]["trunk_hidden"] == [16, 16]
    assert manifest.read(league)["created_at"] == first["created_at"]


def test_code_state_is_all_none_outside_a_git_repo(tmp_path):
    assert manifest.code_state(tmp_path) == {"commit": None, "branch": None, "dirty": None}
