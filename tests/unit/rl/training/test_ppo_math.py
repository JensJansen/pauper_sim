"""PPO update math against independent references (plan P1, planning/AGENT_PIPELINE_TEST_PLAN.md).

GAE, advantage normalisation, the clipped surrogate, the value loss, the entropy term, the k3 KL estimate, the clip
fraction, gradient clipping and the target_kl early stop are each checked on a constructed batch (tests/support/
ppo_reference.py) against a plain reference. Constructed batches use length-1 episodes and a stepper with lr 0, so the
policy's outputs are known and the weights do not move while the loss terms are being measured."""
import copy

import numpy as np
import pytest
import torch

from rl.training.ppo import _compute_gae, ppo_update
from support.ppo_reference import gae_reference, make_buffer, policy_outputs, sgd, tiny_net


# ---- GAE ---------------------------------------------------------------------------------------------------------------

@pytest.mark.parametrize("gamma,lam", [(0.99, 0.95), (1.0, 1.0), (0.9, 0.0), (0.5, 0.5), (0.0, 0.9), (1.0, 0.0)])
def test_gae_equals_the_textbook_recursion_on_random_batches(gamma, lam):
    rng = np.random.default_rng(7)
    for _ in range(25):
        n = int(rng.integers(1, 40))
        rewards, values = rng.normal(size=n), rng.normal(size=n)
        dones = rng.random(n) < 0.2
        got = _compute_gae(rewards.astype(np.float32), values.astype(np.float32), dones.astype(np.float32), gamma, lam)
        assert np.allclose(got, gae_reference(rewards, values, dones, gamma, lam), atol=1e-4)


def test_gae_hand_computed_cases():
    # One step: A = r - V.
    assert _compute_gae([1.0], [0.25], [True], 0.99, 0.95)[0] == pytest.approx(0.75)
    # Two steps, terminal at the end, gamma = lam = 1: the Monte-Carlo return minus the value, at every step.
    adv = _compute_gae([0.0, 1.0], [0.4, 0.7], [False, True], 1.0, 1.0)
    assert adv == pytest.approx([1.0 - 0.4, 1.0 - 0.7])
    # lambda = 0 is one-step TD: A_t = r_t + gamma * V_{t+1} - V_t.
    adv = _compute_gae([0.0, 0.0, 1.0], [0.1, 0.2, 0.9], [False, False, True], 0.9, 0.0)
    assert adv == pytest.approx([0.9 * 0.2 - 0.1, 0.9 * 0.9 - 0.2, 1.0 - 0.9])


def test_gae_never_bootstraps_across_a_game_boundary():
    r1, v1 = [0.0, 1.0], [0.3, 0.6]
    r2, v2 = [0.0, 0.0, -1.0], [0.2, 0.1, 0.5]
    together = _compute_gae(r1 + r2, v1 + v2, [False, True, False, False, True], 0.99, 0.95)
    apart = np.concatenate([_compute_gae(r1, v1, [False, True], 0.99, 0.95),
                            _compute_gae(r2, v2, [False, False, True], 0.99, 0.95)])
    assert np.allclose(together, apart)


def test_gae_treats_the_end_of_an_unfinished_buffer_as_terminal():
    """No terminal flush (defensive case in ppo_update): nothing follows the last element, so it bootstraps 0."""
    adv = _compute_gae([0.0, 0.5], [0.2, 0.3], [False, False], 1.0, 1.0)
    assert adv == pytest.approx([0.5 - 0.2, 0.5 - 0.3])


# ---- the loss terms, measured on a constructed batch -------------------------------------------------------------------

def _batch(n=12, seed=3, deltas=None, reward_scale=1.0):
    """n length-1 episodes; old_logp is set so that the ratio of every transition is exp(deltas[i]) at the net's own
    weights. Returns (net, buf, logp_new, values_net)."""
    net = tiny_net(seed)
    buf = make_buffer([1] * n, seed=seed, reward_scale=reward_scale)
    logits, values = policy_outputs(net, buf)
    logp_all = torch.log_softmax(logits.double(), dim=-1)
    new_logp = logp_all[torch.arange(n), torch.as_tensor(buf.action)]
    deltas = np.linspace(-1.0, 1.0, n) if deltas is None else np.asarray(deltas, dtype=np.float64)
    buf.logp = [float(x) for x in (new_logp.numpy() - deltas)]
    return net, buf, new_logp.numpy(), logp_all


def _reference_losses(buf, logp_all, gamma=0.99, lam=0.95, clip=0.2, floor=0.0):
    n = len(buf)
    adv_raw = gae_reference(buf.reward, buf.value, buf.done, gamma, lam)
    ret = adv_raw + np.array(buf.value)
    adv = (adv_raw - adv_raw.mean()) / (max(adv_raw.std(), floor) + 1e-8)
    act = np.array(buf.action)
    new_logp = logp_all[np.arange(n), act].numpy()
    ratio = np.exp(new_logp - np.array(buf.logp))
    surrogate = np.minimum(ratio * adv, np.clip(ratio, 1 - clip, 1 + clip) * adv)
    probs = torch.softmax(logp_all, dim=-1).numpy()
    entropy = -(probs * np.where(probs > 0, np.log(np.where(probs > 0, probs, 1.0)), 0.0)).sum(-1)
    return {
        "policy_loss": float(-surrogate.mean()),
        "entropy": float(entropy.mean()),
        "approx_kl": float(((ratio - 1) - np.log(ratio)).mean()),
        "clip_fraction": float((np.abs(ratio - 1.0) > clip).mean()),
        "explained_variance": 1.0 - float(((ret - np.array(buf.value)) ** 2).mean()) / float(ret.var()),
        "adv_std": float(adv_raw.std()),
        "ret": ret,
    }


def _one_step(net, buf, **kw):
    """One epoch, one minibatch holding every episode, lr 0: every reported number is a pre-step measurement."""
    kw.setdefault("target_kl", None)
    stats = {}
    out = ppo_update(net, sgd(net), buf, "cpu", n_epochs=1, batch_size=len(buf), stats_out=stats, **kw)
    names = ("policy_loss", "value_loss", "entropy", "approx_kl", "clip_fraction", "epochs_run", "explained_variance",
             "adv_std")
    return dict(zip(names, out, strict=True)), stats


def test_policy_loss_entropy_kl_and_clip_fraction_match_the_reference_across_the_clip_range():
    net, buf, _new_logp, logp_all = _batch(n=16)
    got, _ = _one_step(net, buf)
    ref = _reference_losses(buf, logp_all)
    for key in ("policy_loss", "entropy", "approx_kl", "clip_fraction", "explained_variance", "adv_std"):
        assert got[key] == pytest.approx(ref[key], abs=1e-4), key
    assert 0.0 < got["clip_fraction"] < 1.0, "the batch must straddle the clip range or it proves nothing"


def test_the_surrogate_clips_on_the_correct_side_for_each_advantage_sign():
    """ratio far above 1+clip with a positive advantage is capped (no further gain); far below 1-clip with a negative
    advantage is capped as well; the other two combinations are not (the pessimistic min keeps the unclipped term)."""
    for ratio_delta, adv_sign in ((1.0, +1), (1.0, -1), (-1.0, +1), (-1.0, -1)):
        net, buf, _nl, logp_all = _batch(n=8, deltas=[ratio_delta] * 8)
        # a hand-made reward pattern: constant sign advantage is removed by normalisation, so make half +/- instead
        for i in range(8):
            buf.value[i] = 0.0
            buf.reward[i] = adv_sign * (1.0 if i % 2 == 0 else 0.2)
        got, _ = _one_step(net, buf)
        ref = _reference_losses(buf, logp_all)
        assert got["policy_loss"] == pytest.approx(ref["policy_loss"], abs=1e-4), (ratio_delta, adv_sign)


def test_value_loss_is_the_mean_squared_error_to_the_gae_return():
    net, buf, _nl, _la = _batch(n=10, seed=5)
    got, _ = _one_step(net, buf)
    logits_unused, values_net = policy_outputs(net, buf)
    ref = _reference_losses(buf, _la)
    assert got["value_loss"] == pytest.approx(float(((values_net.numpy() - ref["ret"]) ** 2).mean()), abs=1e-4)


def test_adv_norm_floor_changes_the_policy_loss_as_the_reference_says():
    net, buf, _nl, logp_all = _batch(n=12, seed=9, reward_scale=0.01)  # tiny spread, well under the floor
    for floor in (0.0, 0.5):
        got, _ = _one_step(copy.deepcopy(net), buf, adv_norm_floor=floor)
        assert got["policy_loss"] == pytest.approx(_reference_losses(buf, logp_all, floor=floor)["policy_loss"], abs=1e-4)


def test_masked_actions_get_no_probability_and_no_entropy():
    net, buf, _nl, logp_all = _batch(n=10, seed=11)
    masks = np.array(buf.mask)
    probs = torch.softmax(logp_all, dim=-1).numpy()
    assert probs[:, :masks.shape[1]][~masks].max() < 1e-12
    assert probs[:, masks.shape[1]:].max() < 1e-12  # the dummy padded token slot is never legal either


# ---- stats_out: update-wide means (Q5) -----------------------------------------------------------------------------------

def test_stats_out_means_equal_the_last_minibatch_values_when_there_is_one_minibatch():
    net, buf, *_ = _batch(n=9, seed=4)
    got, means = _one_step(net, buf)
    for key in ("policy_loss", "value_loss", "entropy", "approx_kl", "clip_fraction"):
        assert means[f"{key}_mean"] == pytest.approx(got[key], abs=1e-6), key


def test_stats_out_means_do_not_depend_on_how_the_episodes_are_batched():
    """With lr 0 the weights never move, so the step-weighted mean over every minibatch is a property of the data: it must
    be identical for minibatches of 1, 2, 3 or all episodes, ragged lengths included (padding must not dilute any mean)."""
    lengths = [1, 3, 2, 4, 1, 2]
    buf = make_buffer(lengths, seed=21)
    results = []
    for batch_size in (1, 2, 3, len(lengths)):
        net = tiny_net(2)
        stats = {}
        ppo_update(net, sgd(net), buf, "cpu", n_epochs=1, batch_size=batch_size, target_kl=None, stats_out=stats)
        results.append(stats)
    for key in results[0]:
        assert [r[key] for r in results] == pytest.approx([results[0][key]] * len(results), abs=1e-5), key


def test_stats_out_is_filled_only_when_asked_and_does_not_change_the_return_value():
    net, buf, *_ = _batch(n=6, seed=2)
    net_a, net_b = copy.deepcopy(net), copy.deepcopy(net)
    a = ppo_update(net_a, sgd(net_a), buf, "cpu", n_epochs=1, batch_size=6, target_kl=None)
    stats = {}
    b = ppo_update(net_b, sgd(net_b), buf, "cpu", n_epochs=1, batch_size=6, target_kl=None, stats_out=stats)
    assert a == pytest.approx(b, abs=1e-6)
    assert set(stats) == {f"{k}_mean" for k in ("policy_loss", "value_loss", "entropy", "approx_kl", "clip_fraction")}


# ---- gradient clipping and the KL early stop ------------------------------------------------------------------------------

def _step_norm(max_grad_norm):
    net, buf, *_ = _batch(n=10, seed=6, deltas=[0.5] * 10)
    before = [p.detach().clone() for p in net.parameters()]
    opt = sgd(net, lr=1.0)  # plain SGD, lr 1: the parameter change IS the clipped gradient
    ppo_update(net, opt, buf, "cpu", n_epochs=1, batch_size=10, target_kl=None, max_grad_norm=max_grad_norm)
    return float(torch.sqrt(sum(((p.detach() - b) ** 2).sum() for p, b in zip(net.parameters(), before, strict=True))))


def test_gradient_clipping_caps_the_step_at_max_grad_norm():
    unclipped = _step_norm(1e9)
    assert unclipped > 1e-3, "the raw gradient must exceed the cap or this proves nothing"
    assert _step_norm(1e-3) == pytest.approx(1e-3, rel=1e-2)


def test_target_kl_stops_the_update_after_the_epoch_that_exceeded_it():
    far = [1.5] * 12  # ratio e^1.5 against the stored log-probs: a large k3 estimate, and lr 0 keeps it large every epoch
    net, buf, *_ = _batch(n=12, seed=8, deltas=far)
    out = ppo_update(net, sgd(net), buf, "cpu", n_epochs=4, batch_size=12, target_kl=0.03)
    assert out[5] == 1, "the first epoch's KL is far above target_kl: no second epoch"
    for target in (None, 1e9):
        net, buf, *_ = _batch(n=12, seed=8, deltas=far)
        assert ppo_update(net, sgd(net), buf, "cpu", n_epochs=4, batch_size=12, target_kl=target)[5] == 4


def test_target_kl_does_not_stop_an_update_that_stays_inside_it():
    net, buf, *_ = _batch(n=12, seed=8, deltas=[0.0] * 12)  # ratio 1 everywhere, lr 0: KL stays 0
    assert ppo_update(net, sgd(net), buf, "cpu", n_epochs=4, batch_size=12, target_kl=0.03)[5] == 4


# ---- a stored action must be legal under its own stored mask (Q6) -----------------------------------------------------------

def test_an_action_the_stored_mask_forbids_raises_instead_of_training_on_it():
    net, buf, *_ = _batch(n=8, seed=13)
    victim = 3
    illegal = [a for a in range(4) if not buf.mask[victim][a]]
    if not illegal:
        buf.mask[victim] = np.array([True, True, True, False])
        illegal = [3]
    buf.action[victim] = illegal[0]
    with pytest.raises(ValueError, match=rf"transition {victim} of 8"):
        ppo_update(net, sgd(net), buf, "cpu", n_epochs=1, batch_size=8, target_kl=None)


@pytest.mark.parametrize("bad_action", [-1, 99])
def test_an_action_outside_the_mask_width_raises(bad_action):
    net, buf, *_ = _batch(n=5, seed=14)
    buf.action[0] = bad_action
    with pytest.raises(ValueError, match="stored mask"):
        ppo_update(net, sgd(net), buf, "cpu", n_epochs=1, batch_size=5, target_kl=None)


def test_a_buffer_whose_actions_are_all_legal_is_accepted():
    net, buf, *_ = _batch(n=8, seed=15)
    ppo_update(net, sgd(net), buf, "cpu", n_epochs=1, batch_size=8, target_kl=None)


# ---- the coefficients act in the right direction and at the right scale ---------------------------------------------------------

def _flat_advantage_batch(n=10, seed=21):
    """Every advantage is identical, so after normalisation it is exactly 0 and the policy-gradient term vanishes: what moves the
    weights is only the value loss (vf_coef) and the entropy bonus (ent_coef), which isolates each coefficient."""
    net, buf, *_ = _batch(n, seed, deltas=[0.0] * n)
    for i in range(n):
        buf.value[i], buf.reward[i] = 0.0, 0.5
    return net, buf


def _delta(coefs, lr=0.05):
    net, buf = _flat_advantage_batch()
    before = torch.cat([p.detach().flatten() for p in net.parameters()])
    ppo_update(net, sgd(net, lr), buf, "cpu", n_epochs=1, batch_size=len(buf), target_kl=None, max_grad_norm=1e9, **coefs)
    return torch.cat([p.detach().flatten() for p in net.parameters()]) - before


def _mean_entropy(net, buf):
    logits, _ = policy_outputs(net, buf)
    p = torch.softmax(logits.double(), -1)
    return float(-(p * torch.log(p.clamp_min(1e-30))).sum(-1).mean())


def test_a_positive_entropy_coefficient_raises_the_policy_entropy():
    net, buf = _flat_advantage_batch()
    before = _mean_entropy(net, buf)
    ppo_update(net, sgd(net, 0.5), buf, "cpu", n_epochs=1, batch_size=len(buf), target_kl=None, ent_coef=1.0, vf_coef=0.0)
    assert _mean_entropy(net, buf) > before


def test_the_value_loss_weight_scales_the_value_gradient_and_zero_switches_it_off():
    off = _delta({"vf_coef": 0.0, "ent_coef": 0.0})
    one = _delta({"vf_coef": 1.0, "ent_coef": 0.0})
    two = _delta({"vf_coef": 2.0, "ent_coef": 0.0})
    assert float(off.abs().max()) == 0.0, "with no advantage, no entropy bonus and vf_coef 0 nothing may move"
    assert float(one.abs().max()) > 0
    assert torch.allclose(two, 2 * one, atol=1e-6), "the loss is linear in vf_coef, so one SGD step is too"


def test_the_entropy_weight_scales_the_entropy_gradient_and_zero_switches_it_off():
    off = _delta({"vf_coef": 0.0, "ent_coef": 0.0})
    one = _delta({"vf_coef": 0.0, "ent_coef": 1.0})
    two = _delta({"vf_coef": 0.0, "ent_coef": 2.0})
    assert float(off.abs().max()) == 0.0 and float(one.abs().max()) > 0
    assert torch.allclose(two, 2 * one, atol=1e-6)
