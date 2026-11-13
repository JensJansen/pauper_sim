"""The worker boundary of league rollout collection (plan P3): what crosses between the main process and the workers.

Pure-function tests with synthetic buffers and a recording stand-in for the executor; the one test that plays real games
through the worker is tests/component/agent_pipeline/test_worker_equivalence.py."""
import pickle

import numpy as np
import pytest
import torch

from rl.training import rollout_parallel as rp
from rl.training.train import RolloutBuffer
from support.ppo_reference import make_buffer, tiny_net


class _Perm:  # stands in for a live Permanent: not picklable on purpose
    def __reduce__(self):
        raise TypeError("a live Permanent must never cross the process boundary")


def _buffer_with_identities(lengths=(2, 3, 1), seed=0):
    buf = make_buffer(list(lengths), seed=seed)
    for i in range(len(buf)):
        buf.token_lists[i] = [(1 + i % 4, np.full(3, float(i), dtype=np.float32), _Perm()) for _ in range(i % 3)]
    return buf


def _same_transitions(a, b):
    assert len(a) == len(b)
    for i in range(len(a)):
        assert [(v, tuple(np.ravel(r))) for v, r, _id in a.token_lists[i]] == [
            (v, tuple(np.ravel(r))) for v, r, _id in b.token_lists[i]]
        assert np.array_equal(a.scalar[i], b.scalar[i]) and np.array_equal(a.mask[i], b.mask[i])
        assert (a.action[i], a.logp[i], a.value[i], a.reward[i], a.done[i]) == (
            b.action[i], b.logp[i], b.value[i], b.reward[i], b.done[i])


def test_strip_identities_drops_exactly_the_identity():
    tokens = [(3, np.arange(3.0), _Perm()), (4, np.arange(3.0) + 1, None)]
    out = rp._strip_identities(tokens)
    assert [(v, tuple(r)) for v, r, _ in out] == [(3, (0.0, 1.0, 2.0)), (4, (1.0, 2.0, 3.0))]
    assert all(identity is None for _v, _r, identity in out)


def test_buffer_round_trip_is_lossless_except_identities_and_picklable():
    buf = _buffer_with_identities()
    entries = rp._buffer_to_entries(buf)
    shipped = pickle.loads(pickle.dumps(entries))  # a live identity would raise here
    out = rp._extend_buffer_from_entries(RolloutBuffer(), shipped)
    _same_transitions(buf, out)
    assert all(identity is None for tl in out.token_lists for _v, _r, identity in tl)


def test_round_trip_of_an_empty_buffer_and_of_extending_an_existing_one():
    assert len(rp._extend_buffer_from_entries(RolloutBuffer(), rp._buffer_to_entries(RolloutBuffer()))) == 0
    a, b = _buffer_with_identities(seed=1), _buffer_with_identities(seed=2)
    merged = rp._extend_buffer_from_entries(rp._extend_buffer_from_entries(RolloutBuffer(), rp._buffer_to_entries(a)),
                                            rp._buffer_to_entries(b))
    expect = RolloutBuffer()
    expect.extend(a)
    expect.extend(b)
    _same_transitions(expect, merged)


def test_sanitize_events_keeps_primitives_and_structure_and_makes_the_rest_picklable():
    class Named:
        name = "Lightning Bolt"

    class Anon:
        pass

    logs = [[{"event": "cast", "card": Named(), "n": 3, "ok": True, "x": None, "xs": [1, (2, 3)], "d": {"a": Anon()}}],
            [], [{"event": "end"}]]
    out = rp._sanitize_events(logs)
    pickle.dumps(out)
    assert len(out) == 3 and out[1] == [] and out[2] == [{"event": "end"}]
    ev = out[0][0]
    assert ev["card"] == "Lightning Bolt" and ev["n"] == 3 and ev["ok"] is True and ev["x"] is None
    assert ev["xs"] == [1, [2, 3]] and isinstance(ev["d"]["a"], str)


class _RecordingExecutor:
    def __init__(self):
        self.calls = []

    def submit(self, fn, *args, **kwargs):
        self.calls.append((fn, args, kwargs))
        return ("future", len(self.calls))


@pytest.mark.parametrize("n_games,n_workers", [(10, 3), (3, 5), (1, 4), (8, 8), (7, 1), (0, 3)])
def test_games_are_split_across_workers_with_none_lost_and_no_empty_task(n_games, n_workers):
    ex = _RecordingExecutor()
    futures = rp.submit_league_rollout("a", {"a": tiny_net()}, "deploy_reward_v6", "/tmp/x", 20, n_games, ex, n_workers,
                                       {"a": (16, 16)})
    chunks = [args[6] for _fn, args, _kw in ex.calls]
    assert sum(chunks) == n_games and all(c > 0 for c in chunks) and len(futures) == len(chunks) <= n_workers
    assert max(chunks, default=0) - min(chunks, default=0) <= 1, "chunks are as even as possible"


def test_every_keyword_reaches_every_worker_with_the_value_given():
    ex = _RecordingExecutor()
    mull = {"a": {"w": torch.zeros(1)}}
    rp.submit_league_rollout("a", {"a": tiny_net()}, "deploy_reward_v6", "/root", 77, 6, ex, 3, {"a": (16, 16)},
                             mulligan_state_dicts=mull, game_logs=[], checkpoint_rate=0.37, pfsp=False, pfsp_power=2.5,
                             stratify_0land_pct=0.11)
    assert len(ex.calls) == 3
    for fn, args, kw in ex.calls:
        assert fn is rp._league_rollout_worker
        assert args[0] == "a" and args[3] == "deploy_reward_v6" and args[4] == "/root" and args[5] == 77
        assert kw == {"mulligan_state_dicts": mull, "collect_logs": True, "checkpoint_rate": 0.37, "pfsp": False,
                      "pfsp_power": 2.5, "stratify_0land_pct": 0.11}
    assert len({args[7] for _fn, args, _kw in ex.calls}) == 3, "each worker gets its own seed"


def test_game_logs_are_collected_only_when_a_list_is_passed():
    ex = _RecordingExecutor()
    rp.submit_league_rollout("a", {"a": tiny_net()}, "deploy_reward_v6", "/r", 20, 2, ex, 1, {"a": (16, 16)})
    assert ex.calls[0][2]["collect_logs"] is False


def test_the_shipped_weights_are_cpu_copies_of_the_live_weights_at_call_time():
    """Finding AP-1 (fixed): on a CPU run `.cpu()` alone returns the live tensor, and the executor pickles the task on a feeder
    thread after submit() returns, so with overlap_collect the pickle could see weights the concurrent PPO update had moved."""
    net = tiny_net()
    ex = _RecordingExecutor()
    rp.submit_league_rollout("a", {"a": net}, "deploy_reward_v6", "/r", 20, 2, ex, 1, {"a": (16, 16)})
    shipped = ex.calls[0][1][1]["a"]
    with torch.no_grad():
        for p in net.parameters():
            p.add_(1.0)  # the live net trains on after the call; what was shipped must not move
    for key, tensor in net.state_dict().items():
        assert shipped[key].device.type == "cpu"
        if tensor.dtype.is_floating_point and tensor.numel():
            assert not torch.equal(shipped[key], tensor), f"{key}: shipped weights alias the live ones"


class _Future:
    def __init__(self, result):
        self._result = result

    def result(self):
        return self._result


def test_gather_merges_every_worker_with_nothing_lost_or_duplicated():
    w1 = ({"a": rp._buffer_to_entries(make_buffer([2], seed=1)), "b": rp._buffer_to_entries(make_buffer([1], seed=2))},
          {"a": [["m1"]]}, [[{"e": 1}]], 2, [("b", None, True, False)])
    w2 = ({"a": rp._buffer_to_entries(make_buffer([3], seed=3))}, {"a": [["m2"]], "b": [["m3"]]}, [[{"e": 2}], [{"e": 3}]],
          3, [("a", 4, False, False), ("b", None, True, True)])
    logs = []
    buffers, mull, played, outcomes = rp.gather_league_rollout([_Future(w1), _Future(w2)], game_logs=logs)
    assert played == 5 and len(buffers["a"]) == 5 and len(buffers["b"]) == 1
    assert mull == {"a": [["m1"], ["m2"]], "b": [["m3"]]}
    assert outcomes == w1[4] + w2[4] and logs == [[{"e": 1}], [{"e": 2}], [{"e": 3}]]
    expect = RolloutBuffer()
    expect.extend(make_buffer([2], seed=1))
    expect.extend(make_buffer([3], seed=3))
    _same_transitions(expect, buffers["a"])


def test_gather_ignores_logs_unless_a_list_was_passed():
    w = ({"a": rp._buffer_to_entries(make_buffer([1], seed=1))}, {}, [[{"e": 1}]], 1, [])
    buffers, _m, played, _o = rp.gather_league_rollout([_Future(w)])
    assert played == 1 and len(buffers["a"]) == 1
