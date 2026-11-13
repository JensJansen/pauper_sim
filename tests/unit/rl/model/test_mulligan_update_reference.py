"""The mulligan model's REINFORCE step against a plain reference (plan P7): loss = (-sum logp * (r - V) + 0.5 * sum (V - r)^2
- ENTROPY_COEF * sum H) / n, a forced keep re-masked, the value head fit by its own loss, a no-op on an empty batch.

Built on a tiny net with empty hands (an empty token list) and lr 0, so the weights do not move while the loss is measured."""
import numpy as np
import pytest
import torch

from rl.model import mulligan as M
from rl.model.arch import SetTransformer, pad_token_batch


def _net(seed=0):
    torch.manual_seed(seed)
    enc = SetTransformer(vocab_size=5, d_model=8, n_heads=2, n_layers=1, dim_feedforward=16)
    return M.MulliganNet(enc, hidden=8)


def _transitions(n, seed=0, forced_keep=()):
    rng = np.random.default_rng(seed)
    return [["decision", [], [float(rng.random()), float(rng.integers(0, 2))], i not in forced_keep,
             0 if i in forced_keep else int(rng.integers(0, 2)), float(rng.choice([-1.0, 1.0]))] for i in range(n)]


def _reference(net, trs):
    vocab, feats, pad, _ = pad_token_batch([t[1] for t in trs])
    with torch.no_grad():
        mine, _reps = net.encode(vocab, feats, pad, feats[:, :, -1])
        logits, value = net.decision(mine, torch.tensor([t[2] for t in trs], dtype=torch.float32))
    logits = logits.double().clone()
    for i, t in enumerate(trs):
        if not t[3]:
            logits[i, 1] = -1e8
    logp_all = torch.log_softmax(logits, -1)
    p = logp_all.exp()
    actions = torch.tensor([t[4] for t in trs])
    rewards = torch.tensor([t[5] for t in trs], dtype=torch.float64)
    value = value.double()
    logp = logp_all[torch.arange(len(trs)), actions]
    entropy = -(p * logp_all).sum(-1)
    adv = rewards - value
    policy = -(logp * adv).sum()
    vloss = ((value - rewards) ** 2).sum()
    n = len(trs)
    return {"loss": float((policy + 0.5 * vloss - M.ENTROPY_COEF * entropy.sum()) / n), "policy_loss": float(policy / n),
            "value_loss": float(vloss / n)}


@pytest.mark.parametrize("n,forced", [(1, ()), (6, ()), (9, (2, 5))])
def test_the_step_reports_the_reference_loss_terms(n, forced):
    net = _net()
    trs = _transitions(n, seed=n, forced_keep=forced)
    ref = _reference(net, trs)
    got = M.update(net, torch.optim.SGD(net.parameters(), lr=0.0), trs)
    assert got["n"] == n
    for key in ("loss", "policy_loss", "value_loss"):
        assert got[key] == pytest.approx(ref[key], abs=1e-4), key


def test_an_empty_batch_is_a_no_op():
    net = _net()
    snap = {k: v.clone() for k, v in net.state_dict().items()}
    assert M.update(net, torch.optim.SGD(net.parameters(), lr=1.0), []) == {"n": 0}
    assert all(torch.equal(snap[k], v) for k, v in net.state_dict().items())


def test_the_encoder_is_never_trained_by_the_mulligan_step():
    net = _net()
    before = [p.detach().clone() for p in net.encoder.parameters()]
    M.update(net, torch.optim.SGD(net.parameters(), lr=1.0), _transitions(6))
    assert all(torch.equal(a, b) for a, b in zip(before, net.encoder.parameters(), strict=True))


def test_the_baseline_does_not_receive_the_policy_gradient():
    """The value head is trained only by its own squared error (the policy term uses a detached baseline), so repeated steps
    on a fixed batch must bring V toward the rewards."""
    net = _net(2)
    trs = _transitions(10, seed=4)
    before = _reference(net, trs)["value_loss"]
    opt = torch.optim.SGD(net.parameters(), lr=0.02)
    for _ in range(30):
        M.update(net, opt, trs)
    assert _reference(net, trs)["value_loss"] < before
