"""Reference implementations and constructed batches for the PPO-update tests (tests/unit/rl/training/test_ppo_math.py and
tests/component/agent_pipeline).

The references are written from the maths (Schulman et al.: GAE, the clipped surrogate, the k3 KL estimate), not copied from
rl.training.ppo; the only thing shared with the code under test is the network's own forward signature, used to read the
policy's outputs on a stored batch. The independence limit is the usual one: the same author wrote both.

A constructed buffer holds tiny synthetic transitions (an empty token list, random scalar inputs, a random legal action).
Length-1 episodes have no recurrence, so the policy outputs for them can be read straight off the network in one forward pass
(`policy_outputs`); multi-step episodes are for the properties that need no such reference (batching invariance)."""
import numpy as np
import torch

from rl.model.arch import SetTransformer, pad_token_batch
from rl.model.deck import SCALAR_FEATURE_DIM, DeckNetwork, previous_pointer_inputs
from rl.training.train import RolloutBuffer

N_FIXED = 4


def tiny_net(seed=0, n_fixed=N_FIXED):
    torch.manual_seed(seed)
    return DeckNetwork(SetTransformer(vocab_size=5, d_model=8, n_heads=2, n_layers=1, dim_feedforward=16),
                       non_targeting_n_actions=n_fixed, trunk_hidden=(16, 16))


def gae_reference(rewards, values, dones, gamma, lam):
    """GAE from the textbook recursion: delta_t = r_t + gamma * V_{t+1} * (1 - done_t) - V_t and
    A_t = delta_t + gamma * lam * (1 - done_t) * A_{t+1}. Nothing follows the last element of the array, and nothing follows
    a step whose `done` is True (the next element is a different game)."""
    n = len(rewards)
    adv = [0.0] * n
    nxt_adv = 0.0
    for t in range(n - 1, -1, -1):
        last = dones[t] or t == n - 1
        nxt_v = 0.0 if last else values[t + 1]
        delta = rewards[t] + gamma * nxt_v - values[t]
        nxt_adv = delta + (0.0 if last else gamma * lam * nxt_adv)
        adv[t] = nxt_adv
    return np.array(adv, dtype=np.float64)


def make_buffer(lengths, seed=0, n_fixed=N_FIXED, reward_scale=1.0):
    """One episode per entry of `lengths` (steps in that episode; the last step of each is done=True). Every transition has an
    empty token list, random scalar features, a random mask with at least one legal action, a random legal action, and random
    value/reward/old-logp. Returns the RolloutBuffer."""
    rng = np.random.default_rng(seed)
    buf = RolloutBuffer()
    for n in lengths:
        for step in range(n):
            mask = rng.random(n_fixed) < 0.6
            if not mask.any():
                mask[rng.integers(n_fixed)] = True
            action = int(rng.choice(np.flatnonzero(mask)))
            buf.add([], rng.normal(size=SCALAR_FEATURE_DIM).astype(np.float32), mask, action,
                    float(rng.normal(-1.0, 0.3)), float(rng.normal(0.0, 0.5)),
                    float(rng.normal(0.0, reward_scale)), step == n - 1)
    return buf


def policy_outputs(net, buf):
    """(masked logits [B, n_fixed + 1], value [B]) of `net` on a buffer of length-1 episodes with empty token lists: what
    ppo_update's own forward pass sees on the first step of every episode (no previous action, zero recurrent state)."""
    n_fixed = net.non_targeting_head.out_features
    total = len(buf)
    vocab, features, pad, _ids = pad_token_batch(buf.token_lists, device="cpu")
    side_flag = features[:, :, -1]
    mine, theirs, reps = net.encoder(vocab, features, pad, side_flag)
    mask = np.zeros((total, n_fixed + vocab.shape[1]), dtype=bool)
    for i, stored in enumerate(buf.mask):
        mask[i, :len(stored)] = stored
    mask = torch.as_tensor(mask)
    scalar = torch.as_tensor(np.array(buf.scalar), dtype=torch.float32)
    prev_action = net.prev_action_symbols([-1] * total, "cpu")
    prev_pointer = previous_pointer_inputs(vocab, features, [-1] * total, (total, 1), n_fixed)
    with torch.no_grad():
        logits, values, _h = net(mine, theirs, scalar, reps, mask[:, n_fixed:], seq=(total, 1),
                                 prev_action=prev_action, prev_pointer=prev_pointer)
    return logits.masked_fill(~mask, -1e8), values


def sgd(net, lr=0.0):
    return torch.optim.SGD(net.parameters(), lr=lr)
