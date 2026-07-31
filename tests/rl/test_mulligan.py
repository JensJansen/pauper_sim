"""Tests for rl.mulligan: the per-deck mulligan reward function and
MulliganNet's REINFORCE training.

Flakiness note (do not touch the seeding): the REINFORCE learning assertions
in test_mulligan_net_shapes_and_reinforce_learning are otherwise flaky
(~1-in-several spurious failures on random init) even with
torch.manual_seed(0)/random.seed(0) in place. The seeding calls' position --
before the net/optimizer/rng are built -- matters for that reproducibility;
moving them is likely to reintroduce the flakiness, not something to paper
over with retries or loosened tolerances.
"""
import random

import torch
import pytest

import game as _game
from rl.arch import SetTransformer
from rl.features import CardVocab
from rl.mulligan import MULLIGAN_COST, MulliganNet, mulligan_reward, update


@pytest.mark.slow
def test_mulligan_reward_shape():
    # reward shape: big win, CONVEX per-mulligan cost, negative on mull-heavy loss
    assert abs(mulligan_reward(True, 0) - 1.0) < 1e-9
    assert abs(mulligan_reward(True, 3) - (1.0 - MULLIGAN_COST * 9)) < 1e-9   # quadratic penalty
    assert mulligan_reward(False, 3) < 0                                      # loss-after-mulligans is negative
    assert abs(mulligan_reward(False, 0)) < 1e-9                              # kept-and-lost is neutral
    assert mulligan_reward(True, 7) >= 0                                      # a win survives even at the cap
    # convexity: the Nth mulligan must hurt strictly MORE than the (N-1)th
    marg = [mulligan_reward(False, m) - mulligan_reward(False, m + 1) for m in range(7)]
    assert all(marg[i] < marg[i + 1] for i in range(len(marg) - 1)), marg


@pytest.mark.slow
def test_mulligan_net_shapes_and_reinforce_learning():
    torch.manual_seed(0)  # deterministic net init -- the REINFORCE learning asserts below
    random.seed(0)        # are otherwise flaky (~1-in-several spurious failures on random init)

    decklist = _game.parse_decklist_file("../data/mono_blue_terror.txt")
    vocab = CardVocab([decklist])
    shared = SetTransformer(vocab.size, d_model=16, n_heads=2, n_layers=2, dim_feedforward=32)
    for p in shared.parameters():
        p.requires_grad = False
    net = MulliganNet(shared, hidden=32)
    opt = torch.optim.Adam([p for p in net.parameters() if p.requires_grad], lr=1e-2)

    names = sorted({n for n, *_ in decklist})
    rng = random.Random(0)

    def rand_hand(k=7):
        return [vocab.index(rng.choice(names)) for _ in range(k)]

    # heads produce the right shapes and respect the bottom mask
    hi = torch.tensor([rand_hand()], dtype=torch.long)
    sc = torch.tensor([[0.0, 1.0]], dtype=torch.float32)
    logits, value = net.decision(hi, sc)
    assert logits.shape == (1, 2) and value.shape == (1,)
    ci = torch.tensor([[vocab.index(names[0]), vocab.index(names[1]), 0]], dtype=torch.long)
    cm = torch.tensor([[True, True, False]], dtype=torch.bool)
    scores, _ = net.bottom(hi, sc, ci, cm)
    assert scores.shape == (1, 3) and scores[0, 2].item() < -1e7  # padded candidate masked out

    # learning: a synthetic world where mulliganing (action=1) is GOOD -- keeping
    # always loses (reward -0), mulliganing wins (reward WIN - cost). REINFORCE
    # must push the keep/mull head toward "mulligan". Then flip it.
    def train_toward(good_action, steps=300):
        for _ in range(steps):
            batch = []
            for _ in range(32):
                hand = rand_hand()
                # sample the CURRENT policy so it's on-policy
                with torch.inference_mode():
                    lg, _v = net.decision(torch.tensor([hand]), torch.tensor([[0.0, 1.0]]))
                    a = int(torch.distributions.Categorical(logits=lg).sample().item())
                won = (a == good_action)
                r = mulligan_reward(won, mulligans_taken=1 if a == 1 else 0)
                batch.append(["decision", hand, [0.0, 1.0], True, a, r])
            update(net, opt, batch)

    def mull_prob():
        with torch.inference_mode():
            lg, _ = net.decision(torch.tensor([rand_hand()]), torch.tensor([[0.0, 1.0]]))
            return torch.softmax(lg, -1)[0, 1].item()

    train_toward(good_action=1)
    p_mull_when_good = mull_prob()
    assert p_mull_when_good > 0.75, f"should learn to mulligan when mulliganing wins, got {p_mull_when_good:.2f}"
    train_toward(good_action=0)
    p_mull_when_bad = mull_prob()
    assert p_mull_when_bad < 0.25, f"should learn to keep when keeping wins, got {p_mull_when_bad:.2f}"
