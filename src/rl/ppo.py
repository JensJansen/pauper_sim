"""GAE + PPO update math for the token/attention architecture -- extracted
from rl.train so the multiprocessing plumbing (rl.rollout_parallel) and the
rollout-collection game loop (rl.train itself) don't have to carry this
module's own numerics along with them. Pure reorganization: no behavior
changed by moving the code here."""

import numpy as np
import torch

from rl.arch import pad_token_batch


def _compute_gae(rewards_, values_, dones_, gamma, gae_lambda):
    """Standard GAE. Concatenating multiple games' worth of a buffer is safe:
    every game's own end is flushed with done=True, so a reverse GAE pass
    never bootstraps across a game boundary."""
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


def _precompute_frozen_shared(net, token_lists, device, chunk_size=256):
    """Run the FROZEN shared stack over every transition ONCE, returning
    per-transition (mine[i], theirs[i], token_reps[i]) so ppo_update can reuse
    them across all epochs instead of recomputing the SetTransformer n_epochs
    times per minibatch. token_reps[i] is trimmed to that transition's real
    token count (min 1, matching pad_token_batch's 0->1 dummy padding), so it
    can be re-padded to each minibatch's own max later.

    Uses no_grad, NOT inference_mode: the cached tensors are fed back into the
    trainable head's forward, and inference-mode tensors cannot participate in
    an autograd graph (it raises) -- no_grad tensors become plain constant
    leaves, which is exactly what a frozen stack's output is.
    # ponytail: caches the whole buffer's token_reps at once; chunk the reuse
    # too if a huge buffer ever OOMs on GPU."""
    mine_all, theirs_all, reps_all = [], [], []
    with torch.no_grad():
        for start in range(0, len(token_lists), chunk_size):
            chunk = token_lists[start:start + chunk_size]
            vocab_idx, features, key_padding_mask, _identities = pad_token_batch(chunk, device=device)
            side_flag = features[:, :, -1]
            mine, theirs, token_reps = net.shared_stack(vocab_idx, features, key_padding_mask, side_flag)
            for j, toks in enumerate(chunk):
                n_tok = max(len(toks), 1)  # a 0-token board pads to ONE dummy slot, same as pad_token_batch
                mine_all.append(mine[j])
                theirs_all.append(theirs[j])
                reps_all.append(token_reps[j, :n_tok])
    return mine_all, theirs_all, reps_all


def ppo_update(net, optimizers, buf, device, n_epochs=4, batch_size=64, gamma=0.99, gae_lambda=0.95,
                clip_range=0.2, ent_coef=0.01, vf_coef=0.5, max_grad_norm=0.5, target_kl=0.03):
    # ent_coef default 0.01: with no entropy bonus the main policy collapses
    # onto a narrow low-branching behavior (pass, shrink its own board) -- the
    # action-space-minimization pathology; see rl.rewards.deploy_reward_v2's own
    # comment. Still the relevant risk under deploy_reward_v5 (what league
    # training uses as of 2026-08-11), which has no efficiency scaling either:
    # its flat_win_loss_reward base is a flat +1/-1, so nothing in the terminal
    # reward rewards taking FEWER actions, and this bonus stays the thing
    # bounding pointless ones. v5 was itself a response to a related but
    # distinct failure -- passivity that the LOSS band made cheaper than
    # trying, which an entropy bonus alone could not have fixed. The
    # mulligan model has its own ENTROPY_COEF; this is the DeckNetwork policy's.
    # 0.01 is this function's own fallback for callers that don't schedule it
    # (pretrain's train_selfplay path) -- league training instead computes a
    # per-session value via rl.train.ent_coef_schedule and passes it in
    # explicitly; see that function's docstring for why a FIXED coefficient
    # was found to let entropy collapse to a floor by ~250 games/deck and
    # never recover for the following 30,000+ (TRAINING_IMPROVEMENT_OPTIONS.md
    # section 4).
    #
    # target_kl (2026-08-06 addition): a real, non-optional trust-region
    # backstop independent of ent_coef -- 4 fixed epochs over a small
    # (median ~500-transition), non-stationary buffer (a new opponent mix
    # every iteration under league/PFSP sampling) is a textbook setup for
    # each update sharply overfitting to whatever that one small buffer
    # happened to reward, which shrinks the policy's entropy through the
    # POLICY LOSS itself, not through ent_coef's soft regularizer -- raising
    # ent_coef alone can't fully counter it. Checked once per EPOCH (mean
    # approx-KL across that epoch's minibatches), not per-minibatch: a
    # per-minibatch check is noisier and would risk stopping after a single
    # unlucky minibatch; the epoch-level check is the convention used by
    # OpenAI's Spinning Up and Stable-Baselines3's PPO (their own
    # `target_kl` knob). 0.03 is their own commonly-used default; not yet
    # tuned against this specific game/reward's KL distribution.
    """PPO update over a buffer of variable-length token lists -- pads ONCE
    per minibatch (not once for the whole buffer up front), since a buffer
    spanning many games can have wildly different token counts across
    entries and padding the WHOLE buffer to its own global max would waste
    memory/compute proportional to the single largest board state seen.

    optimizers: a LIST of optimizers, all zero_grad'd before and step'd
    after the SAME backward() call -- never one optimizer per net.
    Needed because a DeckNetwork's shared_stack is a REFERENCE to a module
    shared across multiple nets (pretraining's per-deck throwaway heads all
    point at the same SetTransformer+FiLM instance); giving each net's
    call site its own single optimizer over net.parameters() would create
    TWO independent Adam instances tracking separate, unsynchronized
    momentum/variance state for the identical shared_stack tensors, stepping
    on them in alternation. Passing a single-net-only optimizer as [optimizer]
    (league training, where the shared stack is frozen and only one optimizer
    ever touches this net's own params) still works unchanged.

    Returns (policy_loss, value_loss, entropy, approx_kl, clip_fraction,
    epochs_run) -- the last 3 are 2026-08-06 additions (see target_kl above)
    giving callers visibility into how hard each update is pushing the
    policy, previously invisible (only entropy was ever logged, and a fixed
    ent_coef gave no signal on WHY it was collapsing). epochs_run < n_epochs
    means target_kl triggered early stopping that update."""
    values = np.array(buf.value, dtype=np.float32)
    rewards_ = np.array(buf.reward, dtype=np.float32)
    dones = np.array(buf.done, dtype=np.float32)
    adv = _compute_gae(rewards_, values, dones, gamma, gae_lambda)
    ret = adv + values
    adv = (adv - adv.mean()) / (adv.std() + 1e-8)

    # A FROZEN shared stack (league) produces the SAME per-transition outputs
    # every epoch, so precompute them ONCE and reuse -- skipping n_epochs-1
    # redundant SetTransformer forwards per minibatch (the bulk of the update's
    # forward cost, and ~46% of a real training iteration is the update). A
    # TRAINABLE shared stack (pretrain) must recompute so gradients reach it, so
    # this is gated on requires_grad and needs no caller change. Fidelity is
    # exact: the SetTransformer masks padding in attention/pooling, so a
    # transition's cached reps equal what a fresh per-minibatch forward would
    # produce.
    cache_shared = not any(p.requires_grad for p in net.shared_stack.parameters())
    if cache_shared:
        cached_mine, cached_theirs, cached_reps = _precompute_frozen_shared(net, buf.token_lists, device)

    total = len(buf)
    indices = np.arange(total)
    last_policy_loss = last_value_loss = last_entropy = last_approx_kl = last_clip_fraction = 0.0
    epochs_run = 0
    # net's parameter set is fixed for this whole call -- listing it once
    # avoids re-walking the module tree (net.parameters() -> named_modules())
    # on every one of the n_epochs * n_minibatches clip_grad_norm_ calls below.
    all_params = list(net.parameters())
    for _epoch in range(n_epochs):
        epochs_run += 1
        epoch_kl_sum, epoch_kl_n = 0.0, 0
        np.random.shuffle(indices)
        for start in range(0, total, batch_size):
            mb = indices[start:start + batch_size]
            scalar_mb = torch.as_tensor(np.array([buf.scalar[i] for i in mb]), dtype=torch.float32, device=device)
            act_mb = torch.as_tensor(np.array([buf.action[i] for i in mb]), dtype=torch.int64, device=device)
            old_logp_mb = torch.as_tensor(np.array([buf.logp[i] for i in mb]), dtype=torch.float32, device=device)
            adv_mb = torch.as_tensor(adv[mb], dtype=torch.float32, device=device)
            ret_mb = torch.as_tensor(ret[mb], dtype=torch.float32, device=device)

            n_fixed = net.non_targeting_head.out_features
            if cache_shared:
                # Reuse the frozen shared stack's precomputed per-transition
                # outputs -- no SetTransformer forward this epoch. Re-pad
                # token_reps to THIS minibatch's own max token count, exactly as
                # pad_token_batch would have (real tokens first, dummy/pad after).
                mine_summary = torch.stack([cached_mine[i] for i in mb])
                theirs_summary = torch.stack([cached_theirs[i] for i in mb])
                reps_list = [cached_reps[i] for i in mb]
                max_tokens = max(r.shape[0] for r in reps_list)
                token_reps = torch.zeros((len(mb), max_tokens, mine_summary.shape[-1]),
                                         dtype=mine_summary.dtype, device=device)
                for row, r in enumerate(reps_list):
                    token_reps[row, :r.shape[0]] = r
            else:
                # Trainable shared stack (pretrain): recompute so gradients flow into it.
                vocab_idx, features, key_padding_mask, _identities = pad_token_batch(
                    [buf.token_lists[i] for i in mb], device=device)
                side_flag = features[:, :, -1]
                max_tokens = vocab_idx.shape[1]
                mine_summary, theirs_summary, token_reps = net.shared_stack(vocab_idx, features, key_padding_mask, side_flag)

            # Full action mask per minibatch entry -- padded to max_tokens (this
            # batch's own max token count, matching the token_reps padding
            # above). n_fixed read directly off the net (never inferred from
            # mask_length - token_count): pad_token_batch pads a ZERO-token entry
            # (a legitimate empty-board state, e.g. before either seat has played
            # a land) to ONE dummy slot, which would make that inference
            # silently off-by-one.
            full_mask_mb = torch.zeros((len(mb), n_fixed + max_tokens), dtype=torch.bool, device=device)
            for row, i in enumerate(mb):
                stored = buf.mask[i]
                full_mask_mb[row, :n_fixed] = torch.as_tensor(stored[:n_fixed], dtype=torch.bool, device=device)
                pointer_part = stored[n_fixed:]
                full_mask_mb[row, n_fixed:n_fixed + len(pointer_part)] = torch.as_tensor(
                    pointer_part, dtype=torch.bool, device=device,
                )

            pointer_mask_mb = full_mask_mb[:, n_fixed:]
            logits, values_pred = net(mine_summary, theirs_summary, scalar_mb, token_reps, pointer_mask_mb)
            masked_logits = logits.masked_fill(~full_mask_mb, -1e8)
            dist = torch.distributions.Categorical(logits=masked_logits)
            new_logp = dist.log_prob(act_mb)
            entropy = dist.entropy().mean()

            ratio = torch.exp(new_logp - old_logp_mb)
            surr1 = ratio * adv_mb
            surr2 = torch.clamp(ratio, 1 - clip_range, 1 + clip_range) * adv_mb
            policy_loss = -torch.min(surr1, surr2).mean()
            value_loss = ((values_pred - ret_mb) ** 2).mean()
            loss = policy_loss + vf_coef * value_loss - ent_coef * entropy

            for opt in optimizers:
                opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(all_params, max_grad_norm)
            for opt in optimizers:
                opt.step()

            with torch.no_grad():
                # k3 estimator (Schulman, http://joschu.net/blog/kl-approx.html):
                # unbiased, lower-variance than the naive (old_logp - new_logp)
                # mean, and always >= 0 (a real divergence, not a noisy signed
                # estimate) -- computed post-step (uses the SAME ratio the
                # update just used) purely as a diagnostic of how far that step
                # just moved the policy, not a pre-step gate.
                approx_kl = ((ratio - 1) - torch.log(ratio)).mean().item()
                clip_fraction = ((ratio - 1.0).abs() > clip_range).float().mean().item()
            epoch_kl_sum += approx_kl
            epoch_kl_n += 1
            last_policy_loss, last_value_loss, last_entropy = policy_loss.item(), value_loss.item(), entropy.item()
            last_approx_kl, last_clip_fraction = approx_kl, clip_fraction

        # target_kl early stop: checked once per EPOCH (mean KL across that
        # epoch's own minibatches), not per-minibatch -- see ppo_update's own
        # docstring for why. Only takes effect for the NEXT epoch (this one
        # already ran in full), so epochs_run always reflects epochs that
        # actually executed a full pass, never a partial one.
        if target_kl is not None and epoch_kl_n and (epoch_kl_sum / epoch_kl_n) > target_kl:
            break
    return last_policy_loss, last_value_loss, last_entropy, last_approx_kl, last_clip_fraction, epochs_run
