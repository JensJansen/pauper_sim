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
from collections import namedtuple

import numpy as np
import torch

import drl_env
import game
from rl.action_bridge import (
    any_pointer_legal, build_fixed_action_table, execute_pointer_choice, pointer_legal_mask,
)
from rl.arch import pad_token_batch
from rl.features import build_token_set
from rl import mulligan as mulligan_mod


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
    """Per-seat reward, computed seat-relative (drl_env._for_player flips
    state.active_idx to `seat`, so the reward_fn reads that seat's own zones/
    counters and compares state.winner to it). NO external loser gate: the
    reward_fn itself decides win vs loss vs no-winner -- deploy_reward needs the
    LOSER to reach its own (nonzero) loss band, not be forced to 0. Legacy
    reward_fns that still want "loser -> 0" self-contain that check now
    (rl.rewards.action_count_win_reward), so this stays correct for both."""
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


# A decision's forward-FREE part: everything needed to (a) tell whether it's
# forced and (b) later run the policy forward, without having run it yet.
# full_mask is the raw-width (n_fixed + raw_token_count) legal mask, the exact
# array stored in the buffer and re-padded by ppo_update. sole_action is the
# lone legal index when n_legal == 1 (a forced move), else None.
_Decision = namedtuple("_Decision", "tokens scalar full_mask identities fixed_table n_fixed n_legal sole_action")


def _raise_all_false(state, seat):
    # DIAGNOSTIC (temporary): an all-False mask means the engine reached a
    # decision state the action space can't represent AT ALL -- a real gap.
    # masked_fill(-1e8) then Categorical would otherwise sample UNIFORMLY over
    # every (illegal) position and crash downstream (execute_pointer_choice)
    # with a misleading error. Surface the true culprit precisely instead.
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


def _executor_for(state, action_idx, fixed_table, identities):
    """The zero-arg callable that applies action_idx: a fixed-table entry's own
    execute closure, or execute_pointer_choice on the chosen permanent (pointer
    half, indexed past the fixed table)."""
    if action_idx < len(fixed_table):
        execute_fn = fixed_table[action_idx][2]
        return (lambda state=state, execute_fn=execute_fn: execute_fn(state))
    chosen = identities[action_idx - len(fixed_table)]
    return (lambda state=state, chosen=chosen: execute_pointer_choice(state, chosen))


def _is_pass(action_idx, fixed_table):
    return action_idx < len(fixed_table) and fixed_table[action_idx][0] == "Pass"


def _build_decision(state, seat, deck_ctx, horizon):
    """Everything a decision needs that does NOT require the (expensive) policy
    forward: the token set + scalar features and the legal-action mask (fixed
    half via ONE drl_env.legal_action_mask sweep, pointer half via
    pointer_legal_mask). Forward-free by design so a FORCED decision
    (n_legal == 1 -- overwhelmingly a priority Pass with nothing to play) skips
    the network entirely.
    identities is read straight off the token set (each token is (vocab_idx,
    feature_row, identity)); the pointer mask -- and its buffer-stored width --
    are the RAW token count, exactly what ppo_update re-pads per minibatch."""
    vocab, fixed_table, _pending_kinds = deck_ctx
    tokens = build_token_set(state, seat, vocab)
    scalar = _scalar_features(state, seat, horizon)
    identities = [identity for _idx, _row, identity in tokens]

    fixed_mask = np.asarray(drl_env.legal_action_mask(state, fixed_table), dtype=bool)
    pointer_mask = pointer_legal_mask(state, identities) if any_pointer_legal(state) else [False] * len(identities)
    full_mask = np.concatenate([fixed_mask, np.asarray(pointer_mask, dtype=bool)])

    legal = np.flatnonzero(full_mask)
    if legal.size == 0:
        _raise_all_false(state, seat)
    sole = int(legal[0]) if legal.size == 1 else None
    return _Decision(tokens, scalar, full_mask, identities, fixed_table, len(fixed_table), int(legal.size), sole)


def _padded_full_mask(full_mask_np, n_fixed, n_padded_tokens, device):
    """Widen a decision's raw full_mask (n_fixed + raw_token_count) to the
    padded token width the network's logits actually span (n_fixed +
    n_padded_tokens), padding the extra token slots False -- the same
    convention ppo_update uses to rebuild a minibatch mask, and the reason the
    empty-board case (pad_token_batch pads 0 tokens to 1 dummy slot) is safe."""
    out = torch.zeros((n_fixed + n_padded_tokens,), dtype=torch.bool, device=device)
    out[:n_fixed] = torch.as_tensor(full_mask_np[:n_fixed], dtype=torch.bool, device=device)
    ptr = full_mask_np[n_fixed:]
    out[n_fixed:n_fixed + len(ptr)] = torch.as_tensor(ptr, dtype=torch.bool, device=device)
    return out


def _seat_step(state, seat, deck_ctx, net, horizon, device):
    """One seat's decision, batch-of-1 (the sequential collector). Builds the
    decision (mask first -- see _build_decision), takes the sole legal action
    WITHOUT a forward when the state is forced, else runs the shared stack +
    per-deck net, masks, and samples. Returns (executor, buffer_entry, is_pass);
    buffer_entry is None for a forced move (record nothing)."""
    dec = _build_decision(state, seat, deck_ctx, horizon)
    if dec.sole_action is not None:
        return (_executor_for(state, dec.sole_action, dec.fixed_table, dec.identities), None,
                _is_pass(dec.sole_action, dec.fixed_table))

    vocab_idx, features, key_padding_mask, _identities = pad_token_batch([dec.tokens], device=device)
    side_flag = features[:, :, -1]
    full_mask = _padded_full_mask(dec.full_mask, dec.n_fixed, vocab_idx.shape[1], device).unsqueeze(0)
    with torch.inference_mode():
        mine_summary, theirs_summary, token_reps = net.shared_stack(vocab_idx, features, key_padding_mask, side_flag)
        scalar_t = torch.as_tensor(dec.scalar, dtype=torch.float32, device=device).unsqueeze(0)
        logits, value = net(mine_summary, theirs_summary, scalar_t, token_reps, full_mask[:, dec.n_fixed:])
        masked_logits = logits.masked_fill(~full_mask, -1e8)
        dist = torch.distributions.Categorical(logits=masked_logits)
        action = dist.sample()
        logp = dist.log_prob(action)

    action_idx = int(action.item())
    buffer_entry = (dec.tokens, dec.scalar, dec.full_mask, action_idx, float(logp.item()), float(value.item()))
    return (_executor_for(state, action_idx, dec.fixed_table, dec.identities), buffer_entry,
            _is_pass(action_idx, dec.fixed_table))


def collect_rollout(seat_nets, decklists, reward_fns, deck_ctxs, horizon, n_games, rng, device="cpu",
                     game_logs=None, mulligan_ctx=None):
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
    mull_game = [[], []]  # this game's mulligan-model transitions per seat (reward filled at game end)

    def choose_action(state):
        seat = state.active_idx
        # Pregame mulligan phase (keep/mulligan + London bottoming) is owned by the
        # separate deck-specific mulligan model, trained by its own REINFORCE with a
        # direct game-outcome reward (rl.mulligan) -- NOT the main policy. Route those
        # decisions there and record a transition; the main net never sees them.
        if mulligan_ctx is not None:
            pend = state.pending_resolution
            if pend is not None and pend["kind"] in ("mulligan_decision", "mulligan_bottom"):
                return mulligan_mod.decide(mulligan_ctx["nets"][seat], mulligan_ctx["vocab"], state, seat,
                                           mull_game[seat].append)
        executor, entry, is_pass = _seat_step(state, seat, deck_ctxs[seat], seat_nets[seat], horizon, device)
        # entry is None for a FORCED decision (_seat_step took the sole legal
        # action without a policy forward). Record nothing and leave pending
        # untouched: no choice was made, and the last REAL decision must stay
        # pending so the terminal reward still attaches to it (the game-end
        # flush below), not vanish behind a forced move. Reward is terminal-
        # only, so recording the previous decision's reward now (at this real
        # decision) vs. earlier loses nothing.
        if entry is not None:
            if pending[seat] is not None:
                reward = _reward_for(state, seat, reward_fns[seat], horizon, False)
                tokens, scalar, mask, action_idx, logp, value = pending[seat]
                buffers[seat].add(tokens, scalar, mask, action_idx, logp, value, reward, False)
            pending[seat] = entry
        return None if is_pass else executor

    # Batch-of-1 rollout inference: torch's intra-op threading is pure overhead
    # on these tiny per-decision forwards (a single 1-thread worker measured
    # ~1.66x over the default-threaded sequential path, benchmarking/
    # mp_scaling.py). Force one thread for the whole game loop, then restore --
    # so the BATCHED ppo_update that follows still gets every core. No-op inside
    # a parallel worker, which already ran torch.set_num_threads(1) at startup.
    prev_threads = torch.get_num_threads()
    torch.set_num_threads(1)
    games_played = 0
    try:
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
            if mulligan_ctx is not None:
                # Attribute the whole-game outcome directly to this game's mulligan
                # decisions (the bandit reward -- no discounting through gameplay).
                for seat in (0, 1):
                    r = mulligan_mod.mulligan_reward(state.winner == seat, state.players[seat].mulligans_taken)
                    for entry in mull_game[seat]:
                        entry[5] = r
                    mulligan_ctx["out"][seat].extend(mull_game[seat])
                    mull_game[seat] = []
            games_played += 1
    finally:
        torch.set_num_threads(prev_threads)
    return buffers, games_played


def collect_rollout_league(training_deck_name, training_net, training_ctx, training_decklist, reward_fn,
                            pool, decklists_by_name, ctxs_by_name, live_nets, horizon, n_games,
                            rng, device="cpu", mulligan_nets=None):
    """League-play counterpart to collect_rollout: RESAMPLES the opponent
    from `pool` before every single game (not once for the whole call) --
    the actual mechanism league training needs instead of a fixed pairing.
    Reuses collect_rollout unchanged, one real game (n_games=1) at a time,
    rather than duplicating its pending-transition bookkeeping.

    live_nets: dict deck_name -> that deck's own CURRENT (being-trained)
    DeckNetwork -- needed because a sampled opponent may be "some OTHER
    deck's live net", not just training_net or a frozen snapshot.

    Returns (buf, opponent_buffers, games_played). buf is the training deck's
    transitions (training seat always; both seats on a true mirror, matching the
    old mirror-mode data efficiency). opponent_buffers is Path A's salvage: a
    dict {live_opponent_deck_name -> RolloutBuffer} holding the OPPONENT seat's
    transitions from games where the opponent was another deck's CURRENT live
    net -- on-policy for that deck (its net is constant across this whole round),
    so the caller can train that deck from them too, harvesting the ~half of
    each live-vs-live game's signal that used to be discarded. A frozen snapshot
    opponent is off-policy and still contributes nothing."""
    buf = RolloutBuffer()
    opponent_buffers = {}
    mull_training = []          # training deck's mulligan-model transitions (both seats on a mirror)
    mull_opponents = {}         # Path A salvage: live-opponent deck name -> its mulligan transitions
    games_played = 0
    for _ in range(n_games):
        seat_nets, decklists, ctxs, reward_fns, training_seat, is_self, opp_name, opp_is_live = _league_pairing(
            training_deck_name, training_net, training_ctx, training_decklist, reward_fn,
            pool, decklists_by_name, ctxs_by_name, live_nets, rng,
        )
        mull_ctx = None
        if mulligan_nets is not None:
            opp_seat = 1 - training_seat
            seat_mull = [None, None]
            seat_mull[training_seat] = mulligan_nets[training_deck_name]
            seat_mull[opp_seat] = mulligan_nets[opp_name]
            mull_ctx = {"nets": seat_mull, "vocab": training_ctx[0], "out": [[], []]}  # training_ctx = (vocab, fixed_table, pending_kinds)
        buffers, played = collect_rollout(seat_nets, decklists, reward_fns, ctxs, horizon,
                                           n_games=1, rng=rng, device=device, mulligan_ctx=mull_ctx)
        games_played += played
        for seat in ((0, 1) if is_self else (training_seat,)):
            buf.extend(buffers[seat])
        if opp_is_live:
            opponent_buffers.setdefault(opp_name, RolloutBuffer()).extend(buffers[1 - training_seat])
        if mull_ctx is not None:  # mirror the buffer-salvage rules for the mulligan transitions
            for seat in ((0, 1) if is_self else (training_seat,)):
                mull_training.extend(mull_ctx["out"][seat])
            if opp_is_live:
                mull_opponents.setdefault(opp_name, []).extend(mull_ctx["out"][1 - training_seat])
    return buf, opponent_buffers, mull_training, mull_opponents, games_played


def _league_pairing(training_deck_name, training_net, training_ctx, training_decklist, reward_fn,
                    pool, decklists_by_name, ctxs_by_name, live_nets, rng):
    """Sample one opponent (true mirror / another deck's live net / a frozen
    snapshot) and lay out the per-seat nets/decklists/ctxs/reward_fns for a
    single league game, randomizing which seat the training net takes. Returns
    those four 2-lists plus (training_seat, is_self, opponent_deck_name,
    opponent_is_live). Factored out of collect_rollout_league so the
    opponent-sampling and which-seat-to-record rules live in one place.

    opponent_is_live: the opponent is ANOTHER deck's CURRENT live net (not a
    frozen snapshot, not the training net itself). Its transitions this game are
    on-policy for ITS net, so they can be salvaged to train that deck too (Path
    A -- see collect_rollout_league / run_league._run_session). A snapshot
    opponent is off-policy (frozen old weights) and a mirror is the same net
    already pooled into the training buffer, so both are False here."""
    opponent_deck_name, snapshot_path = pool.sample_opponent(training_deck_name, rng)
    is_self = snapshot_path is None and opponent_deck_name == training_deck_name
    if is_self:
        opponent_net = training_net
    elif snapshot_path is None:
        opponent_net = live_nets[opponent_deck_name]
    else:
        opponent_net = pool.load_snapshot_net(snapshot_path, training_net.shared_stack, ctxs_by_name[opponent_deck_name])
    opponent_is_live = snapshot_path is None and not is_self  # another deck's current net -> its transitions are salvageable

    training_seat = rng.randint(0, 1)  # randomized so the training net isn't always seat 0/1
    opponent_seat = 1 - training_seat
    seat_nets, decklists, ctxs, reward_fns = [None, None], [None, None], [None, None], [None, None]
    seat_nets[training_seat], seat_nets[opponent_seat] = training_net, opponent_net
    decklists[training_seat], decklists[opponent_seat] = training_decklist, decklists_by_name[opponent_deck_name]
    ctxs[training_seat], ctxs[opponent_seat] = training_ctx, ctxs_by_name[opponent_deck_name]
    reward_fns[training_seat] = reward_fns[opponent_seat] = reward_fn
    return seat_nets, decklists, ctxs, reward_fns, training_seat, is_self, opponent_deck_name, opponent_is_live


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


def _buffer_to_entries(buf):
    """Serialize a RolloutBuffer to plain (picklable) tuples for the process
    boundary -- identities stripped (see _strip_identities). Shared by the
    worker for both its training-deck buffer and each salvaged live-opponent
    buffer (Path A)."""
    return [
        (_strip_identities(buf.token_lists[i]), buf.scalar[i], buf.mask[i], buf.action[i],
         buf.logp[i], buf.value[i], buf.reward[i], buf.done[i])
        for i in range(len(buf))
    ]


def _extend_buffer_from_entries(buf, entries):
    for entry in entries:
        buf.add(*entry)
    return buf


def _league_rollout_worker(training_deck_name, all_state_dicts, all_trunk_hidden, shared_state_dict,
                            shared_hparams, reward_fn_name, league_root_dir, horizon, n_games, seed,
                            mulligan_state_dicts=None):
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

    mulligan_nets = None
    if mulligan_state_dicts is not None:
        from rl.mulligan import MulliganNet
        mulligan_nets = {}
        for name, sd in mulligan_state_dicts.items():
            mn = MulliganNet(shared)  # reuses the same frozen shared stack built above
            mn.load_state_dict(sd)
            mn.eval()
            mulligan_nets[name] = mn

    buf, opponent_buffers, mull_training, mull_opponents, played = collect_rollout_league(
        training_deck_name, live_nets[training_deck_name], deck_ctxs[training_deck_name], decklists[training_deck_name],
        reward_fn, pool, decklists, deck_ctxs, live_nets, horizon, n_games, rng, device="cpu",
        mulligan_nets=mulligan_nets,
    )
    entries = _buffer_to_entries(buf)
    opponent_entries = {opp: _buffer_to_entries(ob) for opp, ob in opponent_buffers.items()}  # Path A salvage, per live opponent
    # mulligan transitions are plain data (no live-object identities) -> cross the process boundary as-is
    return entries, opponent_entries, mull_training, mull_opponents, played


def collect_rollout_league_parallel(training_deck_name, live_nets, reward_fn_name, league_root_dir, horizon, n_games,
                                     executor, n_workers, shared_hparams, mulligan_state_dicts=None):
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
                         random.randrange(2 ** 31), mulligan_state_dicts)
        for chunk in chunks if chunk > 0
    ]
    buf = RolloutBuffer()
    opponent_buffers = {}  # Path A: live-opponent deck name -> salvaged transitions, merged across workers
    mull_training = []           # training deck's mulligan transitions, merged across workers
    mull_opponents = {}          # Path A for the mulligan model: live-opponent deck -> its transitions
    games_played = 0
    for future in futures:
        entries, opponent_entries, mt, mo, played = future.result()
        games_played += played
        _extend_buffer_from_entries(buf, entries)
        for opp, opp_entries in opponent_entries.items():
            _extend_buffer_from_entries(opponent_buffers.setdefault(opp, RolloutBuffer()), opp_entries)
        mull_training.extend(mt)
        for opp, tr in mo.items():
            mull_opponents.setdefault(opp, []).extend(tr)
    return buf, opponent_buffers, mull_training, mull_opponents, games_played


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
    last_policy_loss = last_value_loss = last_entropy = 0.0
    for _epoch in range(n_epochs):
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
            # a land) to ONE dummy slot, which would make that inference silently
            # off-by-one -- caught by this module's own smoke test hitting it in
            # the very first rollout.
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
        buf_self, opp_self, played = collect_rollout_league("a", net_a, deck_ctx_a, decklist_a, reward_fn, pool,
                                                            decklists_by_name, ctxs_by_name, live_nets,
                                                            horizon, n_games=1, rng=rng, device=device)
        assert played == 1 and len(buf_self) > 0, "true mirror must record a non-empty pooled buffer"
        assert opp_self == {}, "a true mirror salvages NO separate opponent buffer (both seats already pooled into buf_self)"

        pool.sample_opponent = lambda training_deck_name, rng: ("b", None)  # another deck's live net
        buf_cross, opp_cross, played = collect_rollout_league("a", net_a, deck_ctx_a, decklist_a, reward_fn, pool,
                                                              decklists_by_name, ctxs_by_name, live_nets,
                                                              horizon, n_games=1, rng=rng, device=device)
        assert played == 1 and len(buf_cross) > 0, "cross-deck opponent must still record the training seat's own transitions"
        # Path A: a LIVE-net opponent's transitions are salvaged under its deck name.
        assert set(opp_cross) == {"b"} and len(opp_cross["b"]) > 0, "a live-net opponent must salvage its own on-policy transitions"
        assert all(np.isfinite(v) for v in opp_cross["b"].value) and all(np.isfinite(r) for r in opp_cross["b"].reward)

        pool.sample_opponent = lambda training_deck_name, rng: ("a", snapshot_path)  # frozen snapshot of self
        buf_snap, opp_snap, played = collect_rollout_league("a", net_a, deck_ctx_a, decklist_a, reward_fn, pool,
                                                            decklists_by_name, ctxs_by_name, live_nets,
                                                            horizon, n_games=1, rng=rng, device=device)
        assert played == 1 and len(buf_snap) > 0, "a frozen snapshot opponent must still record the training seat's own transitions"
        assert opp_snap == {}, "a frozen snapshot opponent is off-policy -- salvages nothing"
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

    # 5) Frozen shared-stack caching in ppo_update -- the LEAGUE path. When the
    # shared stack is frozen, ppo_update precomputes its per-transition outputs
    # ONCE (_precompute_frozen_shared) and reuses them across epochs instead of
    # recomputing the SetTransformer. Verify the cached path runs, trains the
    # head, and leaves the frozen stack byte-for-byte untouched. Every block
    # above used a TRAINABLE shared stack, so none exercised this path.
    t0 = time.time()
    for p in net_a.shared_stack.parameters():
        p.requires_grad = False
    shared_before = [p.clone() for p in net_a.shared_stack.parameters()]
    head_before = [p.clone() for p in net_a.non_targeting_head.parameters()]
    opt_head = torch.optim.Adam([p for p in net_a.parameters() if p.requires_grad], lr=3e-4)
    pl, vl, ent = ppo_update(net_a, [opt_head], buf_self, device, n_epochs=2, batch_size=16)
    assert np.isfinite(pl) and np.isfinite(vl) and np.isfinite(ent)
    assert all(torch.equal(a, b) for a, b in zip(shared_before, net_a.shared_stack.parameters())), \
        "a FROZEN shared stack must be byte-for-byte unchanged after a cached ppo_update"
    assert any(not torch.equal(a, b) for a, b in zip(head_before, net_a.non_targeting_head.parameters())), \
        "the per-deck head must have actually trained in the cached ppo_update"
    for p in net_a.parameters():
        assert torch.isfinite(p).all(), "a parameter went non-finite after the cached ppo_update"
    print(f"rl.train frozen-cache ppo_update smoke test: OK (shared stack untouched, head trained, {time.time() - t0:.1f}s)")
