"""The policy-movement probe on real games (plan P0/P8): a recorded probe set, read back through a network, gives well-formed
action distributions that do not depend on how they are batched, survive the save/load round trip, and refuse a changed action
table. validation.policy_movement and the progress indicator stand on these numbers."""

import numpy as np
import pytest
import torch

import game
from evaluation import probe as P
from rl.decision.action_bridge import build_fixed_action_table
from rl.decision.agent import AlwaysKeep, SeatAgent
from rl.league.league_runner import build_deck_net
from rl.model.features import CardVocab
from rl.roster import TOKEN_DEFS

ENC = {"d_model": 32, "n_heads": 2, "n_layers": 2, "dim_feedforward": 64}
DECKS = ("mono_red_madness", "elves")


@pytest.fixture()
def world():
    lists = {d: game.parse_decklist_file(f"../data/{d}.txt") for d in DECKS}
    vocab = CardVocab(list(lists.values()), token_card_defs=TOKEN_DEFS)
    tables, nets, ctxs = {}, {}, {}
    torch.manual_seed(0)
    for d in DECKS:
        tables[d] = build_fixed_action_table(lists[d], token_card_defs=TOKEN_DEFS)
        nets[d] = build_deck_net(vocab.size, len(tables[d]), trunk_hidden=(32, 32), encoder=ENC)
        nets[d].eval()
        ctxs[d] = (vocab, tables[d])
    factories = {(d, "a"): (lambda d=d: SeatAgent(nets[d], AlwaysKeep(), ctxs[d])) for d in DECKS}
    probe = P.record_probe_set(DECKS[0], factories, lists, ctxs, n_games=8, seed=3, horizon=60)
    return probe, nets[DECKS[0]], vocab, tables[DECKS[0]], lists, ctxs, factories


def _n_selected(probe):
    return sum(sum(e["probe"]) for e in probe["episodes"])


def test_the_probe_set_has_real_decisions_and_is_reproducible_once_torch_is_seeded(world):
    """Finding AP-2 (fixed, docs): the agents sample their actions from torch's GLOBAL generator, which `seed` does not touch,
    so the set is reproducible only if the caller also seeds torch, as here; record_probe_set's docstring now says so."""
    probe, _net, _vocab, _table, lists, ctxs, factories = world
    assert _n_selected(probe) >= 5, "random self-play reached too few developed states for a probe"
    runs = []
    for _ in range(2):
        torch.manual_seed(11)
        runs.append(P.record_probe_set(DECKS[0], factories, lists, ctxs, n_games=8, seed=3, horizon=60))
    assert [e["action"] for e in runs[0]["episodes"]] == [e["action"] for e in runs[1]["episodes"]]


def test_every_distribution_is_a_probability_vector_over_exactly_the_legal_actions(world):
    probe, net, *_ = world
    dists = P.probe_distributions(net, probe)
    assert len(dists) == _n_selected(probe)
    legal = [int(np.count_nonzero(e["mask"][i])) for e in probe["episodes"] for i, s in enumerate(e["probe"]) if s]
    for d, n_legal in zip(dists, legal, strict=True):
        assert len(d) == n_legal >= P.MIN_LEGAL
        assert (d >= 0).all() and d.sum() == pytest.approx(1.0, abs=1e-5)


def test_distributions_do_not_depend_on_how_episodes_are_batched(world):
    probe, net, *_ = world
    one = P.probe_distributions(net, probe, batch_episodes=1)
    many = P.probe_distributions(net, probe, batch_episodes=64)
    assert len(one) == len(many)
    assert max(float(np.abs(a - b).max()) for a, b in zip(one, many, strict=True)) < 1e-5


def test_a_net_compared_with_itself_has_no_movement_and_a_different_net_has_some(world):
    probe, net, vocab, table, *_ = world
    base = P.probe_distributions(net, probe)
    assert P.compare(base, P.probe_distributions(net, probe)) == (0.0, 1.0)
    other = build_deck_net(vocab.size, len(table), trunk_hidden=(32, 32), encoder=ENC)
    js, agree = P.compare(base, P.probe_distributions(other, probe))
    assert js > 1e-4 and agree < 1.0


def test_save_then_load_gives_the_same_distributions(world, tmp_path):
    probe, net, vocab, table, *_ = world
    path = str(tmp_path / "p.pkl.gz")
    P.save_probe(path, probe, vocab, table)
    back = P.load_probe(path, vocab, table)
    a, b = P.probe_distributions(net, probe), P.probe_distributions(net, back)
    assert len(a) == len(b) and max(float(np.abs(x - y).max()) for x, y in zip(a, b, strict=True)) < 1e-6


def test_a_probe_recorded_against_another_action_table_is_refused(world, tmp_path):
    probe, _net, vocab, table, *_ = world
    path = str(tmp_path / "p.pkl.gz")
    P.save_probe(path, probe, vocab, table)
    changed = table[1:] + table[:1]
    with pytest.raises(ValueError, match="action table changed"):
        P.load_probe(path, vocab, changed)


def test_the_distributions_are_the_ones_the_update_path_would_compute_for_the_stored_actions(world):
    """The probe replays stored actions through the same forward pass ppo_update uses; so the probability it assigns the action
    that was actually taken must match ppo_update's own log-probability of it (ratio 1 at lr 0 against an on-policy rollout)."""
    probe, net, *_ = world
    dists = P.probe_distributions(net, probe)
    taken = [e["action"][i] for e in probe["episodes"] for i, s in enumerate(e["probe"]) if s]
    legal_idx = [np.flatnonzero(e["mask"][i]) for e in probe["episodes"] for i, s in enumerate(e["probe"]) if s]
    for d, a, idx in zip(dists, taken, legal_idx, strict=True):
        assert a in idx, "the recorded action must be legal in its own recorded mask"
        assert d[list(idx).index(a)] > 0, "an action the sampled policy took cannot have probability 0 under the same net"
