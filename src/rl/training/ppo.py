"""GAE + PPO update math for the token/attention architecture, extracted
from rl.training.train so the multiprocessing plumbing (rollout_parallel)
and the rollout-collection loop (train) don't carry this module's numerics."""

import numpy as np
import torch

from rl.model.arch import pad_token_batch


def _compute_gae(rewards_, values_, dones_, gamma, gae_lambda):
    """Standard GAE. Safe to concatenate multiple games: each ends with
    done=True, so a reverse pass never bootstraps across a game boundary."""
    n = len(rewards_)
    adv = np.zeros(n, dtype=np.float32)
    last_gae = 0.0
    for t in reversed(range(n)):
        next_value = 0.0 if dones_[t] or t + 1 >= n else values_[t + 1]
        next_nonterminal = 0.0 if dones_[t] else 1.0
        delta = rewards_[t] + gamma * next_value * next_nonterminal - values_[t]
        last_gae = delta + gamma * gae_lambda * next_nonterminal * last_gae
        adv[t] = last_gae
    return adv


def ppo_update(net, optimizer, buf, device, n_epochs=4, batch_size=64, gamma=0.99, gae_lambda=0.95,
                clip_range=0.2, ent_coef=0.01, vf_coef=0.5, max_grad_norm=0.5, target_kl=0.03,
                adv_norm_floor=0.0):
    # ent_coef default 0.01: without an entropy bonus the policy collapses
    # onto a narrow low-branching behavior (pass, shrink its own board). The
    # mulligan model has its own separate ENTROPY_COEF. League training
    # instead computes a per-session value via rl.training.train.ent_coef_schedule
    # and passes it in explicitly.
    #
    # target_kl: an epoch-level trust-region backstop independent of
    # ent_coef, checked once per epoch (mean approx-KL across that epoch's
    # minibatches) since a per-minibatch check is noisier.
    """PPO update over a buffer of variable-length token lists -- pads the
    whole buffer ONCE up front (device tensors), then slices each minibatch
    to its own widest state, so the encoder sees exactly what per-minibatch
    padding would give it without re-padding every minibatch x epoch.

    optimizer: one optimizer over net.parameters() (the encoder is a
    registered child of net, so it's covered).

    Returns (policy_loss, value_loss, entropy, approx_kl, clip_fraction,
    epochs_run, explained_variance, adv_std). epochs_run < n_epochs means
    target_kl triggered early stopping. explained_variance is
    1 - Var(ret - value) / Var(ret), measured before the update -- unlike
    raw value_loss (an unscaled MSE), it distinguishes a critic that has
    actually learned from one whose targets have simply collapsed toward a
    constant. adv_std is the raw advantage spread before normalization."""
    values = np.array(buf.value, dtype=np.float32)
    rewards_ = np.array(buf.reward, dtype=np.float32)
    dones = np.array(buf.done, dtype=np.float32)
    adv = _compute_gae(rewards_, values, dones, gamma, gae_lambda)
    ret = adv + values
    ret_var = float(ret.var())
    # 0.0 (not 1.0) when targets are constant: no variance to explain.
    explained_variance = (1.0 - float(((ret - values) ** 2).mean()) / ret_var) if ret_var > 1e-12 else 0.0

    # Advantage normalization, with a FLOOR on the divisor. The unguarded
    # form -- (adv - mean) / (std + 1e-8) -- rescales every batch to unit
    # variance even when the real signal is tiny (e.g. a near-always-lost
    # matchup), promoting critic noise to a full-scale gradient. With a
    # floor, a batch with genuinely tiny advantages stays tiny instead of
    # producing a confident update. Default 0.0 reproduces the previous
    # (unguarded) behavior exactly; adv_std is returned so the floor can be
    # set from the actual measured distribution rather than a guess.
    adv_std = float(adv.std())
    adv = (adv - adv.mean()) / (max(adv_std, adv_norm_floor) + 1e-8)

    total = len(buf)
    # EPISODE segmentation: collect_rollout appends each seat's trajectory
    # contiguously and flushes it done=True, and every downstream merge
    # preserves order, so splitting on `done` recovers exact episode
    # boundaries. This is also why the update can't shuffle transitions --
    # the GRU needs each episode replayed in order from its own start.
    bounds, start_idx = [], 0
    for i in range(total):
        if buf.done[i]:
            bounds.append((start_idx, i + 1))
            start_idx = i + 1
    if start_idx < total:
        # No terminal flush (defensive) -- treat the remainder as its own
        # episode rather than dropping transitions.
        bounds.append((start_idx, total))
    episodes = np.arange(len(bounds))
    n_fixed = net.non_targeting_head.out_features
    # Everything per-transition is built ONCE here and indexed per minibatch.
    # The encoder re-runs every minibatch/epoch (it trains), but its INPUTS
    # never change: re-padding them from Python lists every minibatch x epoch
    # was ~40% of the update's wall time (measured). Padding to the whole
    # buffer's widest state and slicing to each minibatch's own widest state
    # yields tensors identical to padding that minibatch alone.
    n_tokens = np.array([len(tl) for tl in buf.token_lists], dtype=np.int64)
    vocab_all, features_all, pad_all, _ids = pad_token_batch(buf.token_lists, device=device)
    # Mask: n_fixed fixed-action columns, then one pointer column per token.
    mask_np = np.zeros((total, n_fixed + vocab_all.shape[1]), dtype=bool)
    for i, stored in enumerate(buf.mask):
        mask_np[i, :n_fixed] = stored[:n_fixed]
        pointer_part = stored[n_fixed:]
        mask_np[i, n_fixed:n_fixed + len(pointer_part)] = pointer_part
    mask_all = torch.as_tensor(mask_np, device=device)
    scalar_all = torch.as_tensor(np.array(buf.scalar), dtype=torch.float32, device=device)
    act_all = torch.as_tensor(np.array(buf.action), dtype=torch.int64, device=device)
    old_logp_all = torch.as_tensor(np.array(buf.logp), dtype=torch.float32, device=device)
    adv_all = torch.as_tensor(adv, dtype=torch.float32, device=device)
    ret_all = torch.as_tensor(ret, dtype=torch.float32, device=device)
    zero = torch.zeros((), device=device)
    last_policy_loss = last_value_loss = last_entropy = last_approx_kl = last_clip_fraction = zero
    epochs_run = 0
    # Listed once (not re-walked per clip_grad_norm_ call); the encoder is a
    # registered child of net, so it's covered here too.
    all_params = list(net.parameters())
    for _epoch in range(n_epochs):
        epochs_run += 1
        epoch_kl_sum, epoch_kl_n = zero, 0
        np.random.shuffle(episodes)
        for start in range(0, len(episodes), batch_size):
            chunk = [bounds[e] for e in episodes[start:start + batch_size]]
            # Episode-major layout (all of episode 0, then episode 1, ...),
            # matching DeckNetwork.forward's seq=(B, T) reshape. Ragged
            # episodes are padded by repeating the last real index; `valid`
            # below zeroes their contribution to every loss term.
            n_seq = len(chunk)
            max_steps = max(e - s for s, e in chunk)
            idx_grid = np.empty((n_seq, max_steps), dtype=np.int64)
            valid_grid = np.zeros((n_seq, max_steps), dtype=bool)
            # prev_action is derived, not stored: the previous recorded
            # action within an episode is simply the previous transition's.
            prev_grid = np.full((n_seq, max_steps), -1, dtype=np.int64)  # -1 == no previous action
            for row, (s_i, e_i) in enumerate(chunk):
                n = e_i - s_i
                idx_grid[row, :n] = np.arange(s_i, e_i)
                idx_grid[row, n:] = e_i - 1
                valid_grid[row, :n] = True
                prev_grid[row, 1:n] = [buf.action[j] for j in range(s_i, e_i - 1)]
            mb = idx_grid.reshape(-1)
            mb_t = torch.as_tensor(mb, device=device)
            prev_action_mb = net.prev_action_symbols(prev_grid.reshape(-1).tolist(), device)
            valid = torch.as_tensor(valid_grid.reshape(-1), dtype=torch.float32, device=device)
            n_valid = valid.sum().clamp(min=1.0)
            scalar_mb = scalar_all[mb_t]
            act_mb = act_all[mb_t]
            old_logp_mb = old_logp_all[mb_t]
            adv_mb = adv_all[mb_t]
            ret_mb = ret_all[mb_t]

            # An empty-board batch still needs one padded slot (max_tokens >= 1).
            max_tokens = max(int(n_tokens[mb].max()), 1)
            vocab_idx = vocab_all[mb_t, :max_tokens]
            features = features_all[mb_t, :max_tokens]
            key_padding_mask = pad_all[mb_t, :max_tokens]
            side_flag = features[:, :, -1]
            mine_summary, theirs_summary, token_reps = net.encoder(vocab_idx, features, key_padding_mask, side_flag)

            # n_fixed read off the net directly, never inferred from
            # mask_length - token_count (a zero-token entry pads to one
            # dummy slot, which would make that inference off-by-one).
            full_mask_mb = mask_all[mb_t, :n_fixed + max_tokens]
            pointer_mask_mb = full_mask_mb[:, n_fixed:]
            # hidden=None -> zeros: every episode replays from the same
            # start state used during collection, avoiding stale hidden state.
            logits, values_pred, _h = net(mine_summary, theirs_summary, scalar_mb, token_reps,
                                          pointer_mask_mb, seq=(n_seq, max_steps),
                                          prev_action=prev_action_mb)
            masked_logits = logits.masked_fill(~full_mask_mb, -1e8)
            dist = torch.distributions.Categorical(logits=masked_logits)
            new_logp = dist.log_prob(act_mb)
            # Every mean below is over real steps only, via `valid` -- a plain
            # .mean() would dilute by however much padding this batch needed.
            entropy = (dist.entropy() * valid).sum() / n_valid

            ratio = torch.exp(new_logp - old_logp_mb)
            surr1 = ratio * adv_mb
            surr2 = torch.clamp(ratio, 1 - clip_range, 1 + clip_range) * adv_mb
            policy_loss = -(torch.min(surr1, surr2) * valid).sum() / n_valid
            value_loss = (((values_pred - ret_mb) ** 2) * valid).sum() / n_valid
            loss = policy_loss + vf_coef * value_loss - ent_coef * entropy

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(all_params, max_grad_norm)
            optimizer.step()

            with torch.no_grad():
                # k3 estimator (Schulman): unbiased, lower-variance than the
                # naive mean, and always >= 0. Diagnostic only (computed
                # post-step), not a pre-step gate.
                approx_kl = (((ratio - 1) - torch.log(ratio)) * valid).sum() / n_valid
                clip_fraction = (((ratio - 1.0).abs() > clip_range).float() * valid).sum() / n_valid
            # Kept as device tensors: a per-minibatch .item() forced a GPU sync
            # that stopped the host from queueing the next minibatch. One sync
            # per epoch (the target_kl check below) and one at return.
            epoch_kl_sum = epoch_kl_sum + approx_kl
            epoch_kl_n += 1
            last_policy_loss, last_value_loss, last_entropy = policy_loss.detach(), value_loss.detach(), entropy.detach()
            last_approx_kl, last_clip_fraction = approx_kl, clip_fraction

        # target_kl early stop, checked once per epoch (mean KL across that
        # epoch's minibatches). Only affects the NEXT epoch.
        if target_kl is not None and epoch_kl_n and float(epoch_kl_sum) / epoch_kl_n > target_kl:
            break
    last_policy_loss, last_value_loss, last_entropy, last_approx_kl, last_clip_fraction = (
        float(x) for x in (last_policy_loss, last_value_loss, last_entropy, last_approx_kl, last_clip_fraction))
    return (last_policy_loss, last_value_loss, last_entropy, last_approx_kl, last_clip_fraction,
            epochs_run, explained_variance, adv_std)
