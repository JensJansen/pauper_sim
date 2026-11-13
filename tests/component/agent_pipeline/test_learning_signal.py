"""Learning-signal checks (plan P10): on a task with a known optimum, the PPO loop measurably learns.

The task is a bandit written straight into rollout buffers: one fixed state, four legal actions, action 2 pays +1 and the
rest -1. Actions are sampled from the network's own policy each round and the log-probability stored is the policy's own,
exactly as collect_rollout would record them, then ppo_update runs with a real optimizer. If the sign of any term, the
advantage, the clip, or the critic target were wrong, the policy would not find action 2. Negative controls: a reward that
ignores the action leaves the policy near where it started, and a reward that pays the OTHER action moves it the other way."""
import numpy as np
import pytest
import torch

from rl.model.deck import SCALAR_FEATURE_DIM
from rl.training.ppo import ppo_update
from rl.training.train import RolloutBuffer
from support.ppo_reference import N_FIXED, policy_outputs, tiny_net

GOOD = 2
SCALAR = np.linspace(-1, 1, SCALAR_FEATURE_DIM).astype(np.float32)
ALL_LEGAL = np.ones(N_FIXED, dtype=bool)
ONLY_ZERO = np.array([True] + [False] * (N_FIXED - 1))


def _probe_buffer():
    buf = RolloutBuffer()
    buf.add([], SCALAR, ALL_LEGAL, 0, 0.0, 0.0, 0.0, True)
    return buf


def _p_good(net):
    logits, _v = policy_outputs(net, _probe_buffer())
    return float(torch.softmax(logits[0].double(), dim=-1)[GOOD])


def _round(net, rng, n_episodes, reward_fn, extra_steps=0):
    """One on-policy collection round. Each episode is the decision at issue followed by `extra_steps` forced moves (only
    action 0 legal, log-prob 0) before the terminal reward, so the reward arrives `extra_steps` steps after the choice."""
    probe = RolloutBuffer()
    for _ in range(n_episodes):
        probe.add([], SCALAR, ALL_LEGAL, 0, 0.0, 0.0, 0.0, True)
    logits, values = policy_outputs(net, probe)
    logp = torch.log_softmax(logits.double(), dim=-1).numpy()
    buf = RolloutBuffer()
    for i in range(n_episodes):
        action = int(rng.choice(N_FIXED, p=np.exp(logp[i])[:N_FIXED] / np.exp(logp[i])[:N_FIXED].sum()))
        reward = reward_fn(action, rng)
        last = extra_steps == 0
        buf.add([], SCALAR, ALL_LEGAL, action, float(logp[i][action]), float(values[i]), reward if last else 0.0, last)
        for k in range(extra_steps):
            end = k == extra_steps - 1
            buf.add([], SCALAR, ONLY_ZERO, 0, 0.0, 0.0, reward if end else 0.0, end)
    return buf


def _train(reward_fn, rounds=40, n_episodes=48, extra_steps=0, seed=0, lr=3e-3):
    rng = np.random.default_rng(seed)
    net = tiny_net(seed)
    opt = torch.optim.Adam(net.parameters(), lr=lr)
    start = _p_good(net)
    for _ in range(rounds):
        buf = _round(net, rng, n_episodes, reward_fn, extra_steps)
        ppo_update(net, opt, buf, "cpu", n_epochs=4, batch_size=16, target_kl=None, ent_coef=0.0)
    return start, _p_good(net)


def _pays_good(action, rng):
    return 1.0 if action == GOOD else -1.0


def test_the_loop_finds_the_rewarded_action():
    start, end = _train(_pays_good)
    assert end > 0.9 and end > start + 0.5, f"P(good action) went {start:.3f} -> {end:.3f}"


def test_the_critic_learns_the_return():
    """A policy that has converged earns +1 every episode, so the critic's value for the state must approach +1 (the
    explained-variance number is 0 there by design: constant returns explain nothing)."""
    rng = np.random.default_rng(0)
    net = tiny_net(0)
    opt = torch.optim.Adam(net.parameters(), lr=3e-3)
    first = float(policy_outputs(net, _probe_buffer())[1][0])
    for _ in range(80):
        ppo_update(net, opt, _round(net, rng, 48, _pays_good), "cpu", n_epochs=4, batch_size=16, target_kl=None, ent_coef=0.0)
    last = float(policy_outputs(net, _probe_buffer())[1][0])
    assert last > 0.6 and last > first + 0.4, f"value of the state went {first:.3f} -> {last:.3f}, return is about +1"


def test_a_reward_that_ignores_the_action_does_not_move_the_policy_much():
    start, end = _train(lambda a, rng: float(rng.choice([-1.0, 1.0])), rounds=30)
    assert abs(end - start) < 0.25, f"P(good action) drifted {start:.3f} -> {end:.3f} on a reward that ignores it"


def test_a_reward_for_the_other_actions_moves_the_policy_away_from_this_one():
    start, end = _train(lambda a, rng: -1.0 if a == GOOD else 1.0)
    assert end < start and end < 0.05


@pytest.mark.parametrize("delay", [1, 3])
def test_credit_reaches_a_decision_whose_reward_arrives_later(delay):
    start, end = _train(_pays_good, rounds=60, extra_steps=delay)
    assert end > start + 0.3, f"delay {delay}: P(good action) went {start:.3f} -> {end:.3f}"
