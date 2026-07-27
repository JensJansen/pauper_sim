"""Self-play rollout collection + PPO update for the token/attention
architecture (rl.features + rl.arch + rl.deck +
rl.action_bridge) -- the piece that actually trains a DeckNetwork.

One orchestration primitive, train_selfplay, covers both training regimes
this architecture needs:
- Mirror self-play (net_a is net_b, same object/weights): both seats'
  transitions get pooled into ONE buffer and ONE ppo_update call -- true
  single-policy self-play, not two independently-updated copies of the same
  weights drifting apart mid-iteration. Used for Phase 4 (a throwaway
  DeckNetwork per pool deck, all sharing one SetTransformer+FiLM instance so
  gradients warm up the shared stack from every deck) and for Stage 1 (one
  deck's own real trunk/critic/pointer head, mirror-only, against a FROZEN
  shared stack).
- Cross-matchup (net_a is not net_b): each net keeps its own buffer and gets
  its own independent ppo_update call, both learning from every game -- the
  same "both models learn from every game" mechanism the prior flat-MLP trainer's
  train_simultaneous_selfplay already validated for the flat-MLP
  architecture (Effort A), now over the token/pointer representation. Used
  for Stage 2.

Reward attribution is computed directly: 0.0 if drl_env._lost(state, seat)
else reward_fn(state, done, horizon), _for_player-flipped at the true end.
Deliberately NOT gated by an extra "state.turn_won is None -> 0" check on top
of whatever reward_fn already decides, which would silently diverge from the
game loop's own semantics for any reward function that isn't already
win/loss-gated."""

import random

import numpy as np
import torch

import drl_env
import game
from rl.action_bridge import (
    any_pointer_legal, build_fixed_action_table, execute_pointer_choice, pointer_legal_mask,
)
from rl.arch import pad_token_batch
from rl.features import build_token_set


class RolloutBuffer:
    def __init__(self):
        self.token_lists, self.scalar, self.mask, self.action, self.logp, self.value, self.reward, self.done = (
            [], [], [], [], [], [], [], [],
        )

    def __len__(self):
        return len(self.token_lists)

    def add(self, token_list, scalar, mask, action, logp, value, reward, done):
        self.token_lists.append(token_list)
        self.scalar.append(scalar)
        self.mask.append(mask)
        self.action.append(action)
        self.logp.append(logp)
        self.value.append(value)
        self.reward.append(reward)
        self.done.append(done)

    def extend(self, other):
        """Append every transition from another RolloutBuffer -- the shared
        "copy one buffer's entries into another" primitive for pooling both
        seats (mirror self-play) and merging per-game league buffers."""
        for i in range(len(other)):
            self.add(other.token_lists[i], other.scalar[i], other.mask[i], other.action[i],
                     other.logp[i], other.value[i], other.reward[i], other.done[i])

    def clear(self):
        self.__init__()


def _reward_for(state, seat, reward_fn, horizon, done):
    """Per-seat reward -- see this module's own docstring for why the reward
    is taken directly from reward_fn (0.0 if the seat lost, else reward_fn's
    value), never gated by an extra turn_won check."""
    if drl_env._lost(state, seat):
        return 0.0
    if done:
        return drl_env._for_player(state, seat, lambda s: reward_fn(s, True, horizon))
    return reward_fn(state, False, horizon)


def _scalar_features(state, seat_idx, horizon):
    """Non-tokenized globals -- turn number, lands-played, mulligans, am-I-
    turn-player, floating mana pool, phase one-hot, my/opponent life. Same
    composition rl.deck.SCALAR_FEATURE_DIM documents (mana-pool cap of 8,
    matched here). state.mana_pool is a
    GameState property proxying to state.players[state.active_idx]
    (game/state.py's _active_player_property) -- read unconditionally, not
    gated, since _for_player below already guarantees active_idx == seat_idx
    for the whole duration of _read."""
    def _read(s):
        me = s.players[seat_idx]
        other = s.players[1 - seat_idx]
        out = [
            min(s.turn_number / horizon, 1.0),
            1.0 if s.lands_played_this_turn > 0 else 0.0,
            min(s.mulligans_taken, 7) / 7,
            1.0 if s.active_idx == s.turn_player_idx else 0.0,
        ]
        for color in game.POOL_COLORS:
            out.append(min(s.mana_pool.get(color, 0), 8) / 8)
        for phase in game.turn.Phase:
            out.append(1.0 if phase == s.phase else 0.0)
        out.append(max(me.life_total, 0) / game.state.STARTING_LIFE)
        out.append(max(other.life_total, 0) / game.state.STARTING_LIFE)
        return out
    return drl_env._for_player(state, seat_idx, _read)


def _seat_step(state, seat, deck_ctx, net, horizon, device):
    """One seat's own forward pass at the current decision -- builds the
    token set + scalar features, runs the shared stack then the per-deck
    net, masks the combined logits (fixed table via ONE drl_env.
    legal_action_mask sweep, pointer half via pointer_legal_mask), samples,
    and returns everything a RolloutBuffer entry needs plus the action
    actually taken. deck_ctx: (vocab, fixed_table, pending_kinds) for this
    seat's own deck."""
    vocab, fixed_table, _pending_kinds = deck_ctx
    tokens = build_token_set(state, seat, vocab)
    scalar = _scalar_features(state, seat, horizon)

    vocab_idx, features, key_padding_mask, identities = pad_token_batch([tokens], device=device)
    side_flag = features[:, :, -1]
    with torch.no_grad():
        mine_summary, theirs_summary, token_reps = net.shared_stack(vocab_idx, features, key_padding_mask, side_flag)

        fixed_mask_np = drl_env.legal_action_mask(state, fixed_table)
        pointer_mask_list = pointer_legal_mask(state, identities[0]) if any_pointer_legal(state) else [False] * len(identities[0])
        fixed_mask = torch.as_tensor(fixed_mask_np, dtype=torch.bool, device=device).unsqueeze(0)
        pointer_mask = torch.as_tensor(pointer_mask_list, dtype=torch.bool, device=device).unsqueeze(0)
        full_mask = torch.cat([fixed_mask, pointer_mask], dim=-1)

        if not bool(full_mask.any()):
            # DIAGNOSTIC (temporary): an all-False mask means the engine
            # reached a decision state the action space can't represent AT
            # ALL -- a real gap. masked_fill(-1e8) then Categorical would
            # otherwise sample UNIFORMLY over every (illegal) position and
            # crash downstream (execute_pointer_choice) with a misleading
            # error. Surface the true culprit precisely instead.
            pend = state.pending_resolution
            print("  *** ALL-FALSE MASK ***", flush=True)
            print(f"    pending_kind={pend['kind'] if pend else None} phase={state.phase} seat={seat}", flush=True)
            if pend:
                print(f"    pending keys={list(pend.keys())}", flush=True)
                for k in ("remaining", "ordered", "kept", "disposed"):
                    if k in pend:
                        v = pend[k]
                        print(f"    pending[{k}]={[getattr(c, 'name', c) for c in v] if isinstance(v, list) else v}", flush=True)
            raise RuntimeError(f"all-False action mask for pending kind {pend['kind'] if pend else None!r}")

        scalar_t = torch.as_tensor(scalar, dtype=torch.float32, device=device).unsqueeze(0)
        logits, value = net(mine_summary, theirs_summary, scalar_t, token_reps, pointer_mask)
        masked_logits = logits.masked_fill(~full_mask, -1e8)
        dist = torch.distributions.Categorical(logits=masked_logits)
        action = dist.sample()
        logp = dist.log_prob(action)

    action_idx = int(action.item())
    n_fixed = len(fixed_table)
    if action_idx < n_fixed:
        execute_fn = fixed_table[action_idx][2]
        executor = (lambda state=state, execute_fn=execute_fn: execute_fn(state))
    else:
        chosen_permanent = identities[0][action_idx - n_fixed]
        executor = (lambda state=state, chosen=chosen_permanent: execute_pointer_choice(state, chosen))

    buffer_entry = (tokens, scalar, full_mask.squeeze(0).cpu().numpy(), action_idx, float(logp.item()), float(value.item()))
    is_pass = n_fixed > action_idx and fixed_table[action_idx][0] == "Pass"
    return executor, buffer_entry, is_pass


def collect_rollout(seat_nets, decklists, reward_fns, deck_ctxs, horizon, n_games, rng, device="cpu",
                     game_logs=None):
    """Plays n_games real self-play games (game.run_multiplayer_game),
    recording a transition into whichever seat's own buffer made each
    decision -- 100% utilization, same core mechanism validated for the
    flat-MLP architecture (the prior flat-MLP trainer), now over the token/pointer
    representation. seat_nets[i]/deck_ctxs[i]: seat i's own DeckNetwork and
    (vocab, fixed_table, pending_kinds) -- may be the SAME object for both
    seats (mirror self-play, Stage 1) or different (Stage 2).

    game_logs: optional list -- if given, one fresh event_log list gets
    appended per game played, threaded straight through to game.
    run_multiplayer_game's own event_log param (game/state.py's
    GameState.log_event -- already-instrumented across mana.py/turn.py/
    resolution.py/game/effects/*.py). No new logging: this only wires the
    engine's own existing, zero-cost-when-off event log into the training
    rollout path, exactly as it already works for the game loop/
    run_multiplayer_game's other callers."""
    buffers = [RolloutBuffer(), RolloutBuffer()]
    pending = [None, None]

    def choose_action(state):
        seat = state.active_idx
        if pending[seat] is not None:
            reward = _reward_for(state, seat, reward_fns[seat], horizon, False)
            tokens, scalar, mask, action_idx, logp, value = pending[seat]
            buffers[seat].add(tokens, scalar, mask, action_idx, logp, value, reward, False)
            pending[seat] = None
        executor, entry, is_pass = _seat_step(state, seat, deck_ctxs[seat], seat_nets[seat], horizon, device)
        pending[seat] = entry
        return None if is_pass else executor

    games_played = 0
    for _ in range(n_games):
        starting_idx = rng.randint(0, 1)
        event_log = [] if game_logs is not None else None
        state = game.run_multiplayer_game(
            decklists=decklists, rng=rng, starting_player_idx=starting_idx,
            choose_action=choose_action, horizon=horizon, combat_enabled=True, event_log=event_log,
        )
        if game_logs is not None:
            game_logs.append(event_log)
        for seat in (0, 1):
            if pending[seat] is not None:
                reward = _reward_for(state, seat, reward_fns[seat], horizon, True)
                tokens, scalar, mask, action_idx, logp, value = pending[seat]
                buffers[seat].add(tokens, scalar, mask, action_idx, logp, value, reward, True)
                pending[seat] = None
        games_played += 1
    return buffers, games_played


def collect_rollout_league(training_deck_name, training_net, training_ctx, training_decklist, reward_fn,
                            pool, decklists_by_name, ctxs_by_name, live_nets, horizon, n_games,
                            rng, device="cpu"):
    """League-play counterpart to collect_rollout: RESAMPLES the opponent
    from `pool` before every single game (not once for the whole call) --
    the actual mechanism league training needs instead of a fixed pairing.
    Reuses collect_rollout unchanged, one real game (n_games=1) at a time,
    rather than duplicating its pending-transition bookkeeping.

    live_nets: dict deck_name -> that deck's own CURRENT (being-trained)
    DeckNetwork -- needed because a sampled opponent may be "some OTHER
    deck's live net", not just training_net or a frozen snapshot.

    Only the training seat's transitions are ever recorded, UNLESS the
    sampled opponent turns out to be training_net's own literal live
    object (a true mirror pairing, not a frozen snapshot of the same
    deck) -- then both seats' transitions get pooled, matching the data
    efficiency the old dedicated mirror-mode path already had. A frozen
    opponent (any snapshot, or a DIFFERENT deck's live net) never
    contributes a buffer entry: its weights aren't being updated by this
    call, so recording its transitions would train nothing."""
    buf = RolloutBuffer()
    games_played = 0
    for _ in range(n_games):
        opponent_deck_name, snapshot_path = pool.sample_opponent(training_deck_name, rng)
        is_self = snapshot_path is None and opponent_deck_name == training_deck_name
        if is_self:
            opponent_net = training_net
        elif snapshot_path is None:
            opponent_net = live_nets[opponent_deck_name]
        else:
            opponent_net = pool.load_snapshot_net(snapshot_path, training_net.shared_stack, ctxs_by_name[opponent_deck_name])

        training_seat = rng.randint(0, 1)  # randomized so the training net isn't always seat 0/1
        opponent_seat = 1 - training_seat
        seat_nets, decklists, ctxs, reward_fns = [None, None], [None, None], [None, None], [None, None]
        seat_nets[training_seat], seat_nets[opponent_seat] = training_net, opponent_net
        decklists[training_seat], decklists[opponent_seat] = training_decklist, decklists_by_name[opponent_deck_name]
        ctxs[training_seat], ctxs[opponent_seat] = training_ctx, ctxs_by_name[opponent_deck_name]
        reward_fns[training_seat] = reward_fns[opponent_seat] = reward_fn  # opponent's own reward is computed but never recorded

        buffers, played = collect_rollout(seat_nets, decklists, reward_fns, ctxs, horizon,
                                           n_games=1, rng=rng, device=device)
        games_played += played
        seats_to_record = (0, 1) if is_self else (training_seat,)
        for seat in seats_to_record:
            buf.extend(buffers[seat])
    return buf, games_played


def _strip_identities(token_list):
    """Drops each token's Permanent-object identity before it crosses a
    process boundary -- ppo_update already discards identities entirely
    (pad_token_batch's own _identities return value, never read again once
    a transition is buffered; identity only matters live, during
    collection, for pointer_legal_mask/execute_pointer_choice), so there is
    nothing to lose, and no need to find out the hard way whether a live
    Permanent (embedded in a real GameState object graph) is even safely
    picklable at all."""
    return [(idx, row, None) for idx, row, _identity in token_list]


def _league_rollout_worker(training_deck_name, all_state_dicts, all_trunk_hidden, shared_state_dict,
                            shared_hparams, reward_fn_name, league_root_dir, horizon, n_games, seed):
    """Runs in a SEPARATE PROCESS (spawned fresh -- Windows only supports
    the "spawn" start method, no fork, so this re-imports the whole module
    graph from scratch rather than inheriting any parent-process memory).
    Must be a module-level function: ProcessPoolExecutor on spawn needs to
    locate it by import path (rl.train._league_rollout_worker), not a
    closure or lambda.

    Rebuilds decklists/vocab/deck_ctxs/fixed_tables locally via
    build_pool() rather than receiving them from the parent -- fixed_table
    entries hold legal_fn/execute_fn closures (drl_env.build_action_table's
    own _attack_execute/_choose_permanent_execute/... each return a nested
    `def execute(state): ...`), which the standard pickle module cannot
    serialize at all. Only plain tensors (state_dicts), scalars, and
    strings actually cross the process boundary; reward_fn is passed BY
    NAME for the identical reason (rl.rewards's own named instances, e.g.
    action_count_win_reward_200_floor02, are themselves closures returned
    by action_count_win_reward(...)). league_root_dir is passed explicitly
    rather than imported from run_league.py to avoid a circular import
    (run_league.py already imports FROM this module at module scope)."""
    torch.set_num_threads(1)  # this worker IS the unit of parallelism -- it must not also spawn its own intra-op thread pool and oversubscribe the physical cores every other worker is also competing for
    import rl.rewards as rewards_module
    from rl.arch import SetTransformer
    from rl.deck import DeckNetwork
    from rl.league import LeaguePool
    from rl.pool import build_pool

    reward_fn = getattr(rewards_module, reward_fn_name)
    decklists, vocab, deck_ctxs, fixed_tables = build_pool()

    shared = SetTransformer(vocab.size, **shared_hparams)
    shared.load_state_dict(shared_state_dict)
    shared.eval()
    for p in shared.parameters():
        p.requires_grad = False

    live_nets = {}
    for name, state_dict in all_state_dicts.items():
        net = DeckNetwork(shared, film_condition_dim=shared.d_model, non_targeting_n_actions=len(fixed_tables[name]),
                           trunk_hidden=all_trunk_hidden[name])
        net.load_state_dict(state_dict)
        net.eval()
        live_nets[name] = net

    pool = LeaguePool(league_root_dir, list(decklists))  # read-only here -- this worker never calls register_snapshot
    rng = random.Random(seed)

    buf, played = collect_rollout_league(
        training_deck_name, live_nets[training_deck_name], deck_ctxs[training_deck_name], decklists[training_deck_name],
        reward_fn, pool, decklists, deck_ctxs, live_nets, horizon, n_games, rng, device="cpu",
    )
    entries = [
        (_strip_identities(buf.token_lists[i]), buf.scalar[i], buf.mask[i], buf.action[i],
         buf.logp[i], buf.value[i], buf.reward[i], buf.done[i])
        for i in range(len(buf))
    ]
    return entries, played


def collect_rollout_league_parallel(training_deck_name, live_nets, reward_fn_name, league_root_dir, horizon, n_games,
                                     executor, n_workers, shared_hparams):
    """Orchestrator (runs in the MAIN process): splits n_games across
    n_workers, submits one _league_rollout_worker task per worker via the
    given (already-created, reused-across-calls) ProcessPoolExecutor --
    reused rather than created fresh per call so process-spawn/import
    overhead (re-importing torch/the game engine in every worker) is paid
    ONCE, not once per collection round. Every deck's live net crosses the
    boundary (not just training_deck_name's own), since a worker needs the
    SAME opponent-sampling capability collect_rollout_league already has
    in-process -- including sampling some OTHER deck's current live net,
    not just training_deck_name's own or frozen snapshots."""
    shared = live_nets[training_deck_name].shared_stack
    shared_state_dict = shared.state_dict()
    all_state_dicts = {name: net.state_dict() for name, net in live_nets.items()}
    all_trunk_hidden = {name: tuple(layer.out_features for layer in net.trunk_layers) for name, net in live_nets.items()}

    base = n_games // n_workers
    remainder = n_games % n_workers
    chunks = [base + (1 if i < remainder else 0) for i in range(n_workers)]

    futures = [
        executor.submit(_league_rollout_worker, training_deck_name, all_state_dicts, all_trunk_hidden,
                         shared_state_dict, shared_hparams, reward_fn_name, league_root_dir, horizon, chunk,
                         random.randrange(2 ** 31))
        for chunk in chunks if chunk > 0
    ]
    buf = RolloutBuffer()
    games_played = 0
    for future in futures:
        entries, played = future.result()
        games_played += played
        for entry in entries:
            buf.add(*entry)
    return buf, games_played


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


def ppo_update(net, optimizers, buf, device, n_epochs=4, batch_size=64, gamma=0.99, gae_lambda=0.95,
                clip_range=0.2, ent_coef=0.0, vf_coef=0.5, max_grad_norm=0.5):
    """PPO update over a buffer of variable-length token lists -- pads ONCE
    per minibatch (not once for the whole buffer up front), since a buffer
    spanning many games can have wildly different token counts across
    entries and padding the WHOLE buffer to its own global max would waste
    memory/compute proportional to the single largest board state seen.

    optimizers: a LIST of optimizers, all zero_grad'd before and step'd
    after the SAME backward() call -- never one optimizer per net.
    Needed because a DeckNetwork's shared_stack is a REFERENCE to a module
    shared across multiple nets (Phase 4's per-deck throwaway heads all
    point at the same SetTransformer+FiLM instance); giving each net's
    call site its own single optimizer over net.parameters() would create
    TWO independent Adam instances tracking separate, unsynchronized
    momentum/variance state for the identical shared_stack tensors,
    stepping on them in alternation -- confirmed the hard way (see git
    history) as the exact bug this signature change fixes. Passing a
    single-net-only optimizer as [optimizer] (Stage 1/2, where the shared
    stack is frozen and only one optimizer ever touches this net's own
    params) still works unchanged."""
    values = np.array(buf.value, dtype=np.float32)
    rewards_ = np.array(buf.reward, dtype=np.float32)
    dones = np.array(buf.done, dtype=np.float32)
    adv = _compute_gae(rewards_, values, dones, gamma, gae_lambda)
    ret = adv + values
    adv = (adv - adv.mean()) / (adv.std() + 1e-8)

    total = len(buf)
    indices = np.arange(total)
    last_policy_loss = last_value_loss = last_entropy = 0.0
    for _epoch in range(n_epochs):
        np.random.shuffle(indices)
        for start in range(0, total, batch_size):
            mb = indices[start:start + batch_size]
            token_lists_mb = [buf.token_lists[i] for i in mb]
            scalar_mb = torch.as_tensor(np.array([buf.scalar[i] for i in mb]), dtype=torch.float32, device=device)
            act_mb = torch.as_tensor(np.array([buf.action[i] for i in mb]), dtype=torch.int64, device=device)
            old_logp_mb = torch.as_tensor(np.array([buf.logp[i] for i in mb]), dtype=torch.float32, device=device)
            adv_mb = torch.as_tensor(adv[mb], dtype=torch.float32, device=device)
            ret_mb = torch.as_tensor(ret[mb], dtype=torch.float32, device=device)

            vocab_idx, features, key_padding_mask, _identities = pad_token_batch(token_lists_mb, device=device)
            side_flag = features[:, :, -1]
            # Full action mask per minibatch entry -- padded to the batch's
            # own max token count, matching pad_token_batch's own padding
            # (a shorter mask entry's own "extra" positions at the end
            # correspond to padded, always-illegal token slots). n_fixed
            # read directly off the net (never inferred from
            # mask_length - token_count): pad_token_batch pads a
            # ZERO-token entry (a legitimate empty-board state, e.g. before
            # either seat has played a land) to ONE dummy slot, which would
            # make that inference silently off-by-one for exactly that case
            # -- caught by this module's own smoke test hitting it in the
            # very first rollout.
            n_fixed = net.non_targeting_head.out_features
            max_tokens = vocab_idx.shape[1]
            full_mask_mb = torch.zeros((len(mb), n_fixed + max_tokens), dtype=torch.bool, device=device)
            for row, i in enumerate(mb):
                stored = buf.mask[i]
                full_mask_mb[row, :n_fixed] = torch.as_tensor(stored[:n_fixed], dtype=torch.bool, device=device)
                pointer_part = stored[n_fixed:]
                full_mask_mb[row, n_fixed:n_fixed + len(pointer_part)] = torch.as_tensor(
                    pointer_part, dtype=torch.bool, device=device,
                )

            mine_summary, theirs_summary, token_reps = net.shared_stack(vocab_idx, features, key_padding_mask, side_flag)
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
            all_params = list(net.parameters())
            torch.nn.utils.clip_grad_norm_(all_params, max_grad_norm)
            for opt in optimizers:
                opt.step()
            last_policy_loss, last_value_loss, last_entropy = policy_loss.item(), value_loss.item(), entropy.item()
    return last_policy_loss, last_value_loss, last_entropy


def batch_size_for_iteration(iteration, n_iterations, start=32, cap=2048, n_steps=6):
    """Small, granular minibatches early in training, doubling toward a
    larger cap as training progresses -- the "increase batch size instead
    of decaying the learning rate" schedule shape (Smith et al. 2017: more
    frequent, noisier early gradient steps to cover ground quickly on a
    policy far from any optimum; larger, smoother steps later as it
    approaches convergence), not the reverse -- see this session's own
    design discussion for why that direction has actual empirical
    precedent and the opposite one doesn't. n_steps doublings spread
    evenly across the run (a step schedule, not continuous growth --
    simplest thing that matches "incrementally increasing")."""
    if n_iterations <= 1:
        return start
    progress = iteration / n_iterations
    doublings = int(progress * n_steps)
    return min(start * (2 ** doublings), cap)


def device_for_batch_size(batch_size, gpu_threshold):
    """Mechanical CPU/GPU switch for ppo_update ONLY -- rollout collection
    (batch-of-1 inference) stays CPU unconditionally regardless of this
    schedule; that's a settled, batch-size-independent finding (GPU loses
    on tiny per-decision batches no matter how large training's own
    minibatch grows), not something this function touches. gpu_threshold:
    set from an empirically measured CPU/GPU crossover -- never assume one.
    None (not yet measured) or no CUDA -> always CPU."""
    if gpu_threshold is None or not torch.cuda.is_available():
        return "cpu"
    return "cuda" if batch_size >= gpu_threshold else "cpu"


def move_optimizer_state(optimizer, device):
    """torch.optim.Optimizer has no built-in .to(device): net.to(device)
    moves PARAMETERS but never touches an optimizer's own per-parameter
    state (Adam's exp_avg/exp_avg_sq momentum buffers), which stay
    wherever they were first created. Skipping this is not a performance
    nitpick -- it's a correctness bug: the very next optimizer.step()
    after switching a net's device raises a device-mismatch RuntimeError,
    confirmed by tracing exactly what Adam's own step() does with stored
    state tensors versus current parameter tensors."""
    for state in optimizer.state.values():
        for key, value in state.items():
            if torch.is_tensor(value):
                state[key] = value.to(device)


def _pooled(buffers):
    merged = RolloutBuffer()
    for buf in buffers:
        merged.extend(buf)
    return merged


def train_selfplay(net_a, deck_ctx_a, decklist_a, reward_fn_a, net_b, deck_ctx_b, decklist_b, reward_fn_b,
                    optimizers_a, optimizers_b, horizon, n_iterations, games_per_iteration,
                    rng, device="cpu", game_logs=None):
    """Runs n_iterations rounds of collect_rollout (games_per_iteration real
    games each) + ppo_update. See this module's own docstring for when
    net_a is net_b (mirror self-play, one pooled update) vs. not (Stage 2
    cross-matchup, two independent updates). optimizers_a/optimizers_b:
    LISTS of optimizers (see ppo_update's own docstring for why -- a net's
    shared_stack may need its own separate optimizer from the net's own
    head). Returns nothing -- both nets and all optimizers are updated in
    place, same convention the prior flat-MLP trainer's train_simultaneous_selfplay
    already uses. game_logs: forwarded straight to collect_rollout (see its
    own docstring) -- one entry appended per game, across every iteration."""
    mirror = net_a is net_b
    for iteration in range(n_iterations):
        buffers, games_played = collect_rollout(
            [net_a, net_b], [decklist_a, decklist_b], [reward_fn_a, reward_fn_b],
            [deck_ctx_a, deck_ctx_b], horizon, games_per_iteration, rng, device=device, game_logs=game_logs,
        )
        if mirror:
            merged = _pooled(buffers)
            stats_a = stats_b = ppo_update(net_a, optimizers_a, merged, device) if len(merged) else (0.0, 0.0, 0.0)
        else:
            stats_a = ppo_update(net_a, optimizers_a, buffers[0], device) if len(buffers[0]) else (0.0, 0.0, 0.0)
            stats_b = ppo_update(net_b, optimizers_b, buffers[1], device) if len(buffers[1]) else (0.0, 0.0, 0.0)
        mean_r_a = float(np.mean(buffers[0].reward)) if len(buffers[0]) else 0.0
        mean_r_b = float(np.mean(buffers[1].reward)) if len(buffers[1]) else 0.0
        print(f"  iter {iteration}: games={games_played} buf=({len(buffers[0])},{len(buffers[1])}) "
              f"mean_reward=({mean_r_a:.3f},{mean_r_b:.3f}) "
              f"policy_loss=({stats_a[0]:.4f},{stats_b[0]:.4f}) value_loss=({stats_a[1]:.4f},{stats_b[1]:.4f})")


if __name__ == "__main__":
    # ponytail self-check: run via `python rl.train` from src/. Tiny
    # end-to-end smoke test -- real 2-player games (mono_red_madness mirror,
    # a genuine cross-matchup vs rakdos_madness), tiny network dims, few
    # games/iterations, just enough to prove the whole pipeline (rollout
    # collection -> padded/masked batching -> PPO update) runs without
    # crashing, hanging, or producing NaN/inf, before any real training.
    import random as _random
    import time

    import game
    from rl.rewards import action_count_win_reward_200_floor02
    from rl.arch import SetTransformer
    from rl.deck import DeckNetwork
    from rl.features import CardVocab
    from rl.action_bridge import build_fixed_action_table

    device = "cpu"
    decklist_a = game.parse_decklist_file("../data/mono_red_madness.txt")
    decklist_b = game.parse_decklist_file("../data/rakdos_madness.txt")
    token_defs = (game.BLOOD_TOKEN_CARD_DEF, game.ROBOT_TOKEN_CARD_DEF)
    vocab = CardVocab([decklist_a, decklist_b], token_card_defs=token_defs)

    pending_kinds_a = game.derive_pending_kinds(decklist_a)
    pending_kinds_b = game.derive_pending_kinds(decklist_b)
    fixed_table_a = build_fixed_action_table(decklist_a, token_card_defs=token_defs, pending_kinds=pending_kinds_a)
    fixed_table_b = build_fixed_action_table(decklist_b, token_card_defs=token_defs, pending_kinds=pending_kinds_b)
    deck_ctx_a = (vocab, fixed_table_a, pending_kinds_a)
    deck_ctx_b = (vocab, fixed_table_b, pending_kinds_b)

    shared = SetTransformer(vocab.size, d_model=16, n_heads=2, n_layers=1, dim_feedforward=32)
    net_a = DeckNetwork(shared, film_condition_dim=16, non_targeting_n_actions=len(fixed_table_a), trunk_hidden=(24, 24))
    net_b = DeckNetwork(shared, film_condition_dim=16, non_targeting_n_actions=len(fixed_table_b), trunk_hidden=(24, 24))
    opt_a = torch.optim.Adam(net_a.parameters(), lr=3e-4)
    opt_b = torch.optim.Adam(net_b.parameters(), lr=3e-4)
    # NOTE: blocks 1/2 below share ONE net_a/net_b pair across both a mirror
    # test and a cross-matchup test purely for smoke-test brevity -- net_a
    # here is never mirrored AND cross-matched against the SAME shared stack
    # instance two different nets also pretrain against (that split-
    # optimizer scenario, the actual bug this session's feedback caught, is
    # exercised separately and explicitly in block 3 below).

    reward_fn = action_count_win_reward_200_floor02
    rng = _random.Random(0)
    horizon = 20

    # 1) Mirror self-play smoke test -- net_a plays itself, one pooled
    # update, exercises the "same weights both seats" path.
    t0 = time.time()
    buffers, games_played = collect_rollout(
        [net_a, net_a], [decklist_a, decklist_a], [reward_fn, reward_fn],
        [deck_ctx_a, deck_ctx_a], horizon, n_games=2, rng=rng, device=device,
    )
    assert games_played == 2
    assert len(buffers[0]) > 0 and len(buffers[1]) > 0, "both seats must have recorded at least one transition"
    for buf in buffers:
        assert all(np.isfinite(v) for v in buf.value), "collected values must be finite"
        assert all(np.isfinite(r) for r in buf.reward), "collected rewards must be finite"
        assert buf.done[-1] is True, "every buffer must end with a flushed terminal transition"
    merged = _pooled(buffers)
    policy_loss, value_loss, entropy = ppo_update(net_a, [opt_a], merged, device, n_epochs=2, batch_size=16)
    assert np.isfinite(policy_loss) and np.isfinite(value_loss) and np.isfinite(entropy)
    for p in net_a.parameters():
        assert torch.isfinite(p).all(), "a parameter went non-finite after the mirror PPO update"
    print(f"rl.train mirror smoke test: OK ({games_played} games, buf_sizes={len(buffers[0]), len(buffers[1])}, "
          f"policy_loss={policy_loss:.4f}, {time.time() - t0:.1f}s)")

    # 2) Cross-matchup smoke test -- net_a vs net_b, two independent
    # buffers/updates, exercises the "different decks/action spaces on each
    # seat" path (this is what Stage 2, and pretrain_shared_stack's
    # cross-deck gradient flow into the shared stack, actually rely on).
    t0 = time.time()
    train_selfplay(
        net_a, deck_ctx_a, decklist_a, reward_fn, net_b, deck_ctx_b, decklist_b, reward_fn,
        [opt_a], [opt_b], horizon, n_iterations=2, games_per_iteration=2, rng=rng, device=device,
    )
    for net in (net_a, net_b):
        for p in net.parameters():
            assert torch.isfinite(p).all(), "a parameter went non-finite after the cross-matchup PPO update"
    print(f"rl.train cross-matchup smoke test: OK ({time.time() - t0:.1f}s)")

    # 2b) game_logs smoke test -- wiring the game engine's OWN existing
    # event_log (game/state.py's log_event, already instrumented across
    # mana.py/turn.py/resolution.py/game/effects/*.py) through to
    # collect_rollout, not any new logging. One entry per game played,
    # each a real list of structured event dicts.
    game_logs = []
    _buffers, played = collect_rollout(
        [net_a, net_a], [decklist_a, decklist_a], [reward_fn, reward_fn],
        [deck_ctx_a, deck_ctx_a], horizon, n_games=2, rng=rng, device=device, game_logs=game_logs,
    )
    assert len(game_logs) == played == 2, "one event_log entry must be appended per game played"
    for one_game_events in game_logs:
        assert len(one_game_events) > 0, "a real game must produce at least one engine event"
        for event in one_game_events:
            assert "kind" in event and "turn" in event and "phase" in event, "every event must carry log_event's own envelope"
    kinds_seen = {event["kind"] for one_game_events in game_logs for event in one_game_events}
    assert "turn_start" in kinds_seen, "a multi-turn game must log at least one turn_start event"
    print(f"rl.train game_logs smoke test: OK ({sum(len(g) for g in game_logs)} events across {played} games, "
          f"kinds={sorted(kinds_seen)})")

    # 3) Split-optimizer smoke test -- the actual pattern run_pretrain.py
    # needs: TWO throwaway heads sharing ONE SetTransformer instance, but
    # only ONE optimizer (opt_shared2) ever touches the shared stack's own
    # params, so its Adam momentum stays coherent across both decks'
    # alternating mirror sessions instead of being split across two
    # unsynchronized Adam instances (the exact bug this signature change
    # exists to fix -- see ppo_update's own docstring).
    t0 = time.time()
    shared2 = SetTransformer(vocab.size, d_model=16, n_heads=2, n_layers=1, dim_feedforward=32)
    net_a2 = DeckNetwork(shared2, film_condition_dim=16, non_targeting_n_actions=len(fixed_table_a), trunk_hidden=(24, 24))
    net_b2 = DeckNetwork(shared2, film_condition_dim=16, non_targeting_n_actions=len(fixed_table_b), trunk_hidden=(24, 24))
    opt_shared2 = torch.optim.Adam(shared2.parameters(), lr=3e-4)
    opt_a2_head = torch.optim.Adam([p for n, p in net_a2.named_parameters() if not n.startswith("shared_stack.")], lr=3e-4)
    opt_b2_head = torch.optim.Adam([p for n, p in net_b2.named_parameters() if not n.startswith("shared_stack.")], lr=3e-4)
    shared2_before = [p.clone() for p in shared2.parameters()]

    train_selfplay(net_a2, deck_ctx_a, decklist_a, reward_fn, net_a2, deck_ctx_a, decklist_a, reward_fn,
                    [opt_shared2, opt_a2_head], [opt_shared2, opt_a2_head], horizon,
                    n_iterations=1, games_per_iteration=2, rng=rng, device=device)
    train_selfplay(net_b2, deck_ctx_b, decklist_b, reward_fn, net_b2, deck_ctx_b, decklist_b, reward_fn,
                    [opt_shared2, opt_b2_head], [opt_shared2, opt_b2_head], horizon,
                    n_iterations=1, games_per_iteration=2, rng=rng, device=device)

    assert id(opt_shared2) == id(opt_shared2), "sanity: the SAME optimizer object must be reused across both decks"
    assert any(not torch.equal(a, b) for a, b in zip(shared2_before, shared2.parameters())), (
        "shared stack must have actually moved after two decks' worth of updates through the ONE shared optimizer"
    )
    for net in (net_a2, net_b2):
        for p in net.parameters():
            assert torch.isfinite(p).all(), "a parameter went non-finite after the split-optimizer PPO update"
    print(f"rl.train split-optimizer (Phase 4 pattern) smoke test: OK ({time.time() - t0:.1f}s)")

    # 4) League smoke test -- collect_rollout_league against a REAL
    # LeaguePool, exercising all three opponent kinds it must handle:
    # true mirror (both seats recorded), another deck's live net (training
    # seat only), and a frozen historical snapshot (training seat only).
    # sample_opponent is monkeypatched per sub-case rather than left to
    # chance, so each path is deterministically exercised instead of
    # hoping enough random games happen to hit all three.
    import shutil
    import tempfile

    from rl.league import LeaguePool

    t0 = time.time()
    live_nets = {"a": net_a, "b": net_b}
    decklists_by_name = {"a": decklist_a, "b": decklist_b}
    ctxs_by_name = {"a": deck_ctx_a, "b": deck_ctx_b}
    tmp_dir = tempfile.mkdtemp()
    try:
        pool = LeaguePool(tmp_dir, ["a", "b"], max_snapshots_per_deck=3)
        pool.register_snapshot("a", net_a)  # gives the "frozen snapshot of self" path something real to load
        snapshot_path = pool.snapshots["a"][0][1]

        pool.sample_opponent = lambda training_deck_name, rng: ("a", None)  # true mirror
        buf_self, played = collect_rollout_league("a", net_a, deck_ctx_a, decklist_a, reward_fn, pool,
                                                    decklists_by_name, ctxs_by_name, live_nets,
                                                    horizon, n_games=1, rng=rng, device=device)
        assert played == 1 and len(buf_self) > 0, "true mirror must record a non-empty pooled buffer"

        pool.sample_opponent = lambda training_deck_name, rng: ("b", None)  # another deck's live net
        buf_cross, played = collect_rollout_league("a", net_a, deck_ctx_a, decklist_a, reward_fn, pool,
                                                     decklists_by_name, ctxs_by_name, live_nets,
                                                     horizon, n_games=1, rng=rng, device=device)
        assert played == 1 and len(buf_cross) > 0, "cross-deck opponent must still record the training seat's own transitions"

        pool.sample_opponent = lambda training_deck_name, rng: ("a", snapshot_path)  # frozen snapshot of self
        buf_snap, played = collect_rollout_league("a", net_a, deck_ctx_a, decklist_a, reward_fn, pool,
                                                    decklists_by_name, ctxs_by_name, live_nets,
                                                    horizon, n_games=1, rng=rng, device=device)
        assert played == 1 and len(buf_snap) > 0, "a frozen snapshot opponent must still record the training seat's own transitions"
        assert snapshot_path in pool._net_cache, "load_snapshot_net must have populated the cache"

        for buf in (buf_self, buf_cross, buf_snap):
            assert all(np.isfinite(v) for v in buf.value)
            assert all(np.isfinite(r) for r in buf.reward)

        policy_loss, value_loss, entropy = ppo_update(net_a, [opt_a], buf_self, device, n_epochs=1, batch_size=16)
        assert np.isfinite(policy_loss) and np.isfinite(value_loss)
        for p in net_a.parameters():
            assert torch.isfinite(p).all(), "a parameter went non-finite after a league-buffer PPO update"
    finally:
        shutil.rmtree(tmp_dir)

    print(f"rl.train league smoke test: OK (mirror/cross-deck/snapshot opponents all exercised, {time.time() - t0:.1f}s)")
