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


def ppo_update(net, optimizer, buf, device, n_epochs=4, batch_size=64, gamma=0.99, gae_lambda=0.95,
                clip_range=0.2, ent_coef=0.01, vf_coef=0.5, max_grad_norm=0.5, target_kl=0.03,
                adv_norm_floor=0.0):
    # ent_coef default 0.01: with no entropy bonus the main policy collapses
    # onto a narrow low-branching behavior (pass, shrink its own board) -- an
    # action-space-minimization pathology an earlier, efficiency-scored
    # reward generation hit directly (removed 2026-08-22; see git history).
    # Still the relevant risk under deploy_reward_v6 (what league training
    # uses as of 2026-08-12), which has no efficiency scaling either: its
    # flat_win_loss_reward base is a flat +1/-1, so nothing in the terminal
    # reward rewards taking FEWER actions, and this bonus stays the thing
    # bounding pointless ones. deploy_reward_v6 was itself a response to a
    # related but distinct failure -- passivity that the LOSS band made
    # cheaper than trying, which an entropy bonus alone could not have fixed.
    # The mulligan model has its own ENTROPY_COEF; this is the DeckNetwork
    # policy's. 0.01 is this function's own fallback for callers that don't
    # schedule it -- league training instead computes a per-session value via
    # rl.train.ent_coef_schedule and passes it in explicitly; see that
    # function's docstring for why a FIXED coefficient was found to let
    # entropy collapse to a floor by ~250 games/deck and never recover for
    # the following 30,000+.
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

    optimizer: ONE optimizer over net.parameters(). It used to be a LIST,
    because a DeckNetwork's encoder was a reference to a SetTransformer
    shared across several nets, and a shared module needs a single optimizer
    of its own rather than one per net stepping unsynchronized Adam state on
    the same tensors. Each deck now owns its encoder outright (rl.deck), so
    no module is reachable from two nets and every call site passed a
    one-element list.

    Returns (policy_loss, value_loss, entropy, approx_kl, clip_fraction,
    epochs_run, explained_variance, adv_std) -- approx_kl/clip_fraction/
    epochs_run are 2026-08-06 additions (see target_kl above) giving callers
    visibility into how hard each update is pushing the policy, previously
    invisible (only entropy was ever logged, and a fixed ent_coef gave no
    signal on WHY it was collapsing). epochs_run < n_epochs means target_kl
    triggered early stopping that update. adv_std (2026-08-13) is the RAW
    advantage spread before normalization -- the number adv_norm_floor has to
    be set from, and previously discarded.

    explained_variance (2026-08-13) is 1 - Var(ret - value) / Var(ret),
    measured BEFORE the update. value_loss alone is uninterpretable: it is a
    raw MSE with no scale attached, so a critic that has genuinely learned and
    one whose targets have simply collapsed to a constant both report a small
    number. On this project's own 60,001-game run value_loss sat at 0.01-0.03
    and flat, which was read as a healthy critic; the competing explanation --
    that gamma/gae_lambda discount early-game returns to ~0, making them
    trivially predictable -- fits the same number and has the opposite
    implication. Explained variance separates them (a constant predictor
    scores 0 however small its MSE) and is the acceptance check for any
    future change to the credit-assignment horizon."""
    values = np.array(buf.value, dtype=np.float32)
    rewards_ = np.array(buf.reward, dtype=np.float32)
    dones = np.array(buf.done, dtype=np.float32)
    adv = _compute_gae(rewards_, values, dones, gamma, gae_lambda)
    ret = adv + values
    ret_var = float(ret.var())
    # 0.0 rather than 1.0 when the targets are constant: no variance to explain
    # means the critic has demonstrated nothing, which is the honest reading.
    explained_variance = (1.0 - float(((ret - values) ** 2).mean()) / ret_var) if ret_var > 1e-12 else 0.0

    # Advantage normalization, with a FLOOR on the divisor.
    #
    # The unguarded form -- (adv - mean) / (std + 1e-8) -- rescales every batch
    # to unit variance no matter how little real signal it contains. That is
    # fine when outcomes differ, and actively harmful when they do not: in a
    # matchup lost ~96% of the time nearly every trajectory returns -1, so the
    # raw spread is dominated by critic error rather than by anything the
    # policy did, and dividing by that tiny std promotes pure noise to a
    # full-scale gradient before ~67 Adam steps run on it. Measured at 60,001
    # games/deck, three of four decks spent 58-77% of training in exactly such
    # matchups, and flattening PFSP only
    # takes the worst case to ~56% -- the floor is structural, since two of
    # elves' four opponents are unwinnable.
    #
    # With a floor, a batch whose advantages are genuinely tiny STAYS tiny and
    # produces a proportionally small update instead of a confident one.
    #
    # DEFAULT 0.0 = exactly the previous behavior, deliberately. The right
    # value depends on this reward's actual advantage scale, which nothing has
    # ever recorded -- so adv_std is now returned and logged, and the floor gets
    # set from that distribution rather than from a guess. Picking it blind is
    # the mistake that made PFSP_POWER=2.0 the leading cause of a 60,001-game
    # regression; not repeating it here.
    adv_std = float(adv.std())
    adv = (adv - adv.mean()) / (max(adv_std, adv_norm_floor) + 1e-8)

    total = len(buf)
    # EPISODE segmentation. collect_rollout appends each seat's per-game
    # trajectory to its bucket contiguously and flushes it done=True (see its
    # own docstring), and every merge downstream -- RolloutBuffer.extend, the
    # worker serialization in rl.rollout_parallel -- preserves order. So the
    # buffer is already a concatenation of whole episodes and needs no new
    # bookkeeping to recover them; splitting on `done` is exact.
    #
    # This is also why the update can no longer shuffle transitions: the GRU
    # needs each episode replayed in order, from its own start.
    bounds, start_idx = [], 0
    for i in range(total):
        if buf.done[i]:
            bounds.append((start_idx, i + 1))
            start_idx = i + 1
    if start_idx < total:
        # A trajectory with no terminal flush. collect_rollout always ends one
        # with done=True, so this is defensive -- treat the remainder as its
        # own episode rather than silently dropping transitions.
        bounds.append((start_idx, total))
    episodes = np.arange(len(bounds))
    last_policy_loss = last_value_loss = last_entropy = last_approx_kl = last_clip_fraction = 0.0
    epochs_run = 0
    # net's parameter set is fixed for this whole call -- listing it once
    # avoids re-walking the module tree (net.parameters() -> named_modules())
    # on every one of the n_epochs * n_minibatches clip_grad_norm_ calls below.
    # The encoder is a registered child of net (rl.deck.DeckNetwork), so this
    # covers it -- both for the optimizer and for the grad-norm clip.
    all_params = list(net.parameters())
    for _epoch in range(n_epochs):
        epochs_run += 1
        epoch_kl_sum, epoch_kl_n = 0.0, 0
        np.random.shuffle(episodes)
        for start in range(0, len(episodes), batch_size):
            chunk = [bounds[e] for e in episodes[start:start + batch_size]]
            # Flat batch laid out EPISODE-MAJOR -- all steps of episode 0, then
            # all of episode 1 -- which is the layout DeckNetwork.forward's
            # `seq=(B, T)` reshape assumes. Ragged episodes are padded by
            # REPEATING the last real index: the padded steps compute a real
            # forward (cheap, and it keeps every tensor rectangular) and are
            # then zeroed out of every loss term by `valid` below, so they
            # contribute no gradient.
            n_seq = len(chunk)
            max_steps = max(e - s for s, e in chunk)
            idx_grid = np.empty((n_seq, max_steps), dtype=np.int64)
            valid_grid = np.zeros((n_seq, max_steps), dtype=bool)
            # prev_action is DERIVED, not stored: within an episode the
            # previous recorded action is simply the previous transition's, and
            # the first step of an episode has none. That holds exactly because
            # a forced move advances neither the buffer nor the agent's own
            # prev_action during collection (rl.agent._seat_step), so the
            # recorded sequence IS the sequence the agent conditioned on.
            prev_grid = np.full((n_seq, max_steps), -1, dtype=np.int64)  # -1 == no previous action
            for row, (s_i, e_i) in enumerate(chunk):
                n = e_i - s_i
                idx_grid[row, :n] = np.arange(s_i, e_i)
                idx_grid[row, n:] = e_i - 1
                valid_grid[row, :n] = True
                prev_grid[row, 1:n] = [buf.action[j] for j in range(s_i, e_i - 1)]
            mb = idx_grid.reshape(-1)
            prev_action_mb = net.prev_action_symbols(prev_grid.reshape(-1).tolist(), device)
            valid = torch.as_tensor(valid_grid.reshape(-1), dtype=torch.float32, device=device)
            n_valid = valid.sum().clamp(min=1.0)
            scalar_mb = torch.as_tensor(np.array([buf.scalar[i] for i in mb]), dtype=torch.float32, device=device)
            act_mb = torch.as_tensor(np.array([buf.action[i] for i in mb]), dtype=torch.int64, device=device)
            old_logp_mb = torch.as_tensor(np.array([buf.logp[i] for i in mb]), dtype=torch.float32, device=device)
            adv_mb = torch.as_tensor(adv[mb], dtype=torch.float32, device=device)
            ret_mb = torch.as_tensor(ret[mb], dtype=torch.float32, device=device)

            n_fixed = net.non_targeting_head.out_features
            # The encoder trains with the rest of the net, so its forward is
            # recomputed here every minibatch of every epoch -- gradients have
            # to reach it. A frozen encoder used to allow one cached forward
            # per update instead; per-deck trainable encoders gave that up
            # deliberately (see rl.deck's module docstring).
            vocab_idx, features, key_padding_mask, _identities = pad_token_batch(
                [buf.token_lists[i] for i in mb], device=device)
            side_flag = features[:, :, -1]
            max_tokens = vocab_idx.shape[1]
            mine_summary, theirs_summary, token_reps = net.encoder(vocab_idx, features, key_padding_mask, side_flag)

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
            # hidden=None -> zeros: every episode is replayed from the SAME
            # start state collection used, so the recomputed hidden states are
            # exactly the ones the current weights would have produced. That
            # is what removes the usual stale-hidden-state problem (and with it
            # any need to store states or burn in).
            logits, values_pred, _h = net(mine_summary, theirs_summary, scalar_mb, token_reps,
                                          pointer_mask_mb, seq=(n_seq, max_steps),
                                          prev_action=prev_action_mb)
            masked_logits = logits.masked_fill(~full_mask_mb, -1e8)
            dist = torch.distributions.Categorical(logits=masked_logits)
            new_logp = dist.log_prob(act_mb)
            # Every mean below is over REAL steps only -- a plain .mean() would
            # dilute each term by however much padding this minibatch needed,
            # which varies with how ragged its episodes happen to be.
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
                # k3 estimator (Schulman, http://joschu.net/blog/kl-approx.html):
                # unbiased, lower-variance than the naive (old_logp - new_logp)
                # mean, and always >= 0 (a real divergence, not a noisy signed
                # estimate) -- computed post-step (uses the SAME ratio the
                # update just used) purely as a diagnostic of how far that step
                # just moved the policy, not a pre-step gate.
                approx_kl = ((((ratio - 1) - torch.log(ratio)) * valid).sum() / n_valid).item()
                clip_fraction = ((((ratio - 1.0).abs() > clip_range).float() * valid).sum() / n_valid).item()
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
    return (last_policy_loss, last_value_loss, last_entropy, last_approx_kl, last_clip_fraction,
            epochs_run, explained_variance, adv_std)
