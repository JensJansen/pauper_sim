"""Self-play rollout collection + PPO update for the token/attention
architecture (rl.features + rl.arch + rl.deck +
rl.action_bridge) -- the piece that actually trains a DeckNetwork.

One orchestration primitive, train_selfplay, covers both training regimes
this architecture needs:
- Mirror self-play (net_a is net_b, same object/weights): both seats'
  transitions get pooled into ONE buffer and ONE ppo_update call -- true
  single-policy self-play, not two independently-updated copies of the same
  weights drifting apart mid-iteration. Used for pretraining (a throwaway
  DeckNetwork per pool deck, all sharing one SetTransformer+FiLM instance so
  gradients warm up the shared stack from every deck) and for the league's
  mirror games (one deck's own real trunk/critic/pointer head, mirror-only,
  against a FROZEN shared stack).
- Cross-matchup (net_a is not net_b): each net keeps its own buffer and gets
  its own independent ppo_update call, both learning from every game. Used
  for the league's cross-deck games.

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
from rl.arch import pad_token_batch
from rl import mulligan as mulligan_mod
from rl.agent import SeatAgent, AlwaysKeep


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


# The per-seat DECISION primitives (_seat_step, _build_decision,
# _scalar_features, _executor_for, _padded_full_mask, _Decision,
# _raise_all_false, _is_pass) now live in rl.agent -- collect_rollout drives
# them through SeatAgent.decide, so nothing here calls them directly anymore.
# _reward_for stays: it's ATTRIBUTION (a rollout concern), not decision.


def _constant_pairing(agents, decklists, reward_fns, record_as):
    """A pairing that yields the SAME layout every game -- mirror, cross-matchup,
    or a fixed A-vs-B matchup. (League resampling is _make_league_pairing.)"""
    def pairing(rng):
        return agents, decklists, reward_fns, record_as
    return pairing


def collect_rollout(pairing, n_games, horizon, rng, device="cpu", record=True, greedy=False, game_logs=None):
    """The ONE game loop. Plays n_games real self-play games
    (game.run_multiplayer_game); a per-game `pairing(rng)` supplies that game's
    layout, and the SeatAgents it returns own the mulligan-vs-policy dispatch
    (rl.agent). This function owns only ATTRIBUTION (bandit mulligan reward +
    per-decision terminal-flush PPO reward) and bucketing.

    pairing(rng) -> (agents, decklists, reward_fns, record_as):
      agents[seat]      -- SeatAgent making seat `seat`'s decisions
      decklists[seat]   -- that seat's decklist (for run_multiplayer_game)
      reward_fns[seat]  -- that seat's reward_fn (may be None when record=False)
      record_as[seat]   -- the deck-name BUCKET to record this seat's transitions
                           into, or None to record nothing for that seat (a
                           frozen-snapshot / off-policy opponent, or eval).

    record=False -> pure play (eval / log generation), buffers nothing;
    reward_fns may be None. greedy=True -> deterministic argmax (eval). game_logs:
    optional list, one engine event_log appended per game (game/state.py's
    log_event -- already instrumented; zero-cost when None).

    Returns (buffers_by_deck, mull_by_deck, games_played): dicts keyed by the
    deck-name buckets from record_as. A mirror routes BOTH seats to one bucket
    (pooled single-policy self-play); a live cross-deck opponent routes its own
    seat to its own bucket (the former Path-A salvage, now uniform). Each seat's
    per-game trajectory is appended to its bucket CONTIGUOUSLY and ends done=True,
    so a later GAE pass never bootstraps across a trajectory boundary."""
    buffers_by_deck = {}   # bucket -> RolloutBuffer (main PPO transitions)
    mull_by_deck = {}      # bucket -> list of mulligan transitions (bandit)
    games_played = 0

    # Batch-of-1 rollout inference: torch's intra-op threading is pure overhead
    # on these tiny per-decision forwards (~1.66x measured). Force one thread
    # for the whole game loop, then restore --
    # so the BATCHED ppo_update that follows still gets every core. No-op inside
    # a parallel worker, which already ran torch.set_num_threads(1) at startup.
    prev_threads = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        for _ in range(n_games):
            agents, decklists, reward_fns, record_as = pairing(rng)
            game_buffers = [RolloutBuffer(), RolloutBuffer()]  # per-game per-seat, kept contiguous
            pending = [None, None]
            mull_game = [[], []]

            def choose_action(state, agents=agents, reward_fns=reward_fns, record_as=record_as,
                              game_buffers=game_buffers, pending=pending,
                              mull_game=mull_game):
                seat = state.active_idx
                dr = agents[seat].decide(state, seat, horizon, device, greedy=greedy)
                if record and record_as[seat] is not None:
                    # Mulligan transition: whole-game bandit reward, filled at game end.
                    if dr.mull_entry is not None:
                        mull_game[seat].append(dr.mull_entry)
                    # Main-policy transition: ppo_entry is None for a FORCED decision
                    # (sole legal action) OR a pregame decision (mulligan-owned) --
                    # record nothing and leave pending untouched, so the last REAL
                    # decision stays pending for the terminal reward (flushed below).
                    if dr.ppo_entry is not None:
                        if pending[seat] is not None:
                            reward = _reward_for(state, seat, reward_fns[seat], horizon, False)
                            game_buffers[seat].add(*pending[seat], reward, False)
                        pending[seat] = dr.ppo_entry
                return None if dr.is_pass else dr.executor

            starting_idx = rng.randint(0, 1)
            event_log = [] if game_logs is not None else None
            state = game.run_multiplayer_game(
                decklists=decklists, rng=rng, starting_player_idx=starting_idx,
                choose_action=choose_action, horizon=horizon, combat_enabled=True, event_log=event_log,
            )
            if game_logs is not None:
                game_logs.append(event_log)
            if record:
                for seat in (0, 1):
                    bucket = record_as[seat]
                    if bucket is None:
                        continue
                    if pending[seat] is not None:  # terminal flush for this seat's last real decision
                        reward = _reward_for(state, seat, reward_fns[seat], horizon, True)
                        game_buffers[seat].add(*pending[seat], reward, True)
                    if len(game_buffers[seat]):  # append this seat's whole (contiguous) trajectory to its bucket
                        buffers_by_deck.setdefault(bucket, RolloutBuffer()).extend(game_buffers[seat])
                    if mull_game[seat]:  # attribute the game's outcome to this seat's mulligan picks (bandit)
                        r = mulligan_mod.mulligan_reward(state.winner == seat, state.players[seat].mulligans_taken)
                        for entry in mull_game[seat]:
                            entry[5] = r
                        mull_by_deck.setdefault(bucket, []).extend(mull_game[seat])
            games_played += 1
    finally:
        torch.set_num_threads(prev_threads)
    return buffers_by_deck, mull_by_deck, games_played


def collect_rollout_league(training_deck_name, live_nets, mulligan_nets, deck_ctxs, decklists_by_name,
                            pool, reward_fn, horizon, n_games, rng, device="cpu", record=True, game_logs=None,
                            checkpoint_rate=0.0):
    """League collection: builds a pairing that RESAMPLES the opponent from
    `pool` before every game (true mirror / another deck's live net / a frozen
    snapshot), then runs the ONE loop (collect_rollout). Returns
    (buffers_by_deck, mull_by_deck, games_played) keyed by deck name -- the
    training deck's bucket (training seat always; BOTH seats on a mirror) plus,
    for any game whose opponent was another deck's CURRENT live net, that deck's
    own bucket (on-policy for it -- the former Path-A salvage, now just another
    bucket). A frozen-snapshot opponent is off-policy and records nothing.

    live_nets / mulligan_nets / deck_ctxs / decklists_by_name: dicts keyed by
    deck name over the WHOLE roster (an opponent may be any other deck's live
    net). mulligan_nets=None -> every seat uses AlwaysKeep (no mulligan dispatch/
    training), e.g. a matchup or deck-only collection. checkpoint_rate: forwarded
    to _make_league_pairing/pool.sample_opponent verbatim -- see rl.league.
    LeaguePool.sample_opponent's own docstring."""
    pairing = _make_league_pairing(training_deck_name, live_nets, mulligan_nets, deck_ctxs,
                                    decklists_by_name, pool, reward_fn, checkpoint_rate=checkpoint_rate)
    return collect_rollout(pairing, n_games, horizon, rng, device=device, record=record, game_logs=game_logs)


def _make_league_pairing(training_deck_name, live_nets, mulligan_nets, deck_ctxs, decklists_by_name, pool, reward_fn,
                          checkpoint_rate=0.0):
    """Builds a pairing closure: samples one opponent per game, randomizes which
    seat the training deck takes, wraps each side as a SeatAgent (its live/loaded
    DeckNetwork + its mulligan decider), and sets record_as -- the training
    deck's name for its seat; the opponent's name only if the opponent is a
    mirror (pooled into the training bucket) or another deck's LIVE net (its own
    bucket -- salvage); None for a frozen snapshot (off-policy). Opponent-sampling
    and which-seat-to-record rules live here, in one place.

    checkpoint_rate: forwarded to pool.sample_opponent verbatim (see its own
    docstring) -- the live/checkpoint split, independent of snapshot-window
    occupancy. Default 0.0: every game is real-model-vs-real-model."""
    training_net = live_nets[training_deck_name]
    training_ctx = deck_ctxs[training_deck_name]
    training_decklist = decklists_by_name[training_deck_name]

    def _mull(name):
        return mulligan_nets[name] if mulligan_nets is not None else AlwaysKeep()

    def pairing(rng):
        opp_name, snapshot_path = pool.sample_opponent(training_deck_name, rng, checkpoint_rate=checkpoint_rate)
        is_self = snapshot_path is None and opp_name == training_deck_name
        opp_is_live = snapshot_path is None and not is_self  # another deck's current net -> salvageable

        train_agent = SeatAgent(training_net, _mull(training_deck_name), training_ctx)
        if is_self:
            opp_agent = train_agent  # true mirror: same net + mulligan on both seats
        elif snapshot_path is None:
            opp_agent = SeatAgent(live_nets[opp_name], _mull(opp_name), deck_ctxs[opp_name])
        else:
            # A frozen snapshot loads as a whole frozen SeatAgent (deck + its
            # era-matched mulligan, or AlwaysKeep for a pre-refactor snapshot).
            opp_agent = pool.load_snapshot_agent(snapshot_path, training_net.shared_stack, deck_ctxs[opp_name])

        training_seat = rng.randint(0, 1)  # randomized so the training net isn't always seat 0/1
        opp_seat = 1 - training_seat
        agents = [None, None]
        decklists = [None, None]
        reward_fns = [None, None]
        record_as = [None, None]
        agents[training_seat], agents[opp_seat] = train_agent, opp_agent
        decklists[training_seat], decklists[opp_seat] = training_decklist, decklists_by_name[opp_name]
        reward_fns[training_seat] = reward_fns[opp_seat] = reward_fn
        record_as[training_seat] = training_deck_name
        if is_self:
            record_as[opp_seat] = training_deck_name  # mirror -> both seats pooled into the training bucket
        elif opp_is_live:
            record_as[opp_seat] = opp_name             # another deck's live net -> salvage under its own name
        # else: frozen snapshot -> record_as[opp_seat] stays None (off-policy)
        return agents, decklists, reward_fns, record_as

    return pairing


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


def _sanitize_events(game_logs):
    """Deep-convert event-log values to picklable primitives before they cross a
    process boundary -- a few log_event fields can hold a card closure/lambda (the
    same non-serializable case run_league._json_default guards at JSON-write time),
    which pickle can't ship from an MP worker. Converts an unknown object to its
    string .name if it has one, else repr(); leaves primitives/containers intact."""
    def conv(v):
        if isinstance(v, (str, int, float, bool)) or v is None:
            return v
        if isinstance(v, dict):
            return {k: conv(x) for k, x in v.items()}
        if isinstance(v, (list, tuple)):
            return [conv(x) for x in v]
        name = getattr(v, "name", None)
        return name if isinstance(name, str) else repr(v)
    return [[conv(event) for event in one_game] for one_game in game_logs]


def _league_rollout_worker(training_deck_name, all_state_dicts, all_trunk_hidden, shared_state_dict,
                            shared_hparams, reward_fn_name, league_root_dir, horizon, n_games, seed,
                            mulligan_state_dicts=None, collect_logs=False, checkpoint_rate=0.0):
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

    # Deck names come from all_state_dicts' own keys, NOT list(decklists) (this
    # worker's fresh build_pool() call always returns the FULL roster, regardless of
    # any roster= restriction the orchestrating _run_session applied) -- live_nets
    # above already reflects exactly the (possibly narrowed) roster the orchestrator
    # built, so mirroring its keys here is what keeps this worker's own opponent
    # pool scoped the same way, without needing a separate parameter to carry the
    # restriction across the process boundary redundantly.
    pool = LeaguePool(league_root_dir, list(all_state_dicts.keys()))  # read-only here -- this worker never calls register_snapshot
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

    worker_logs = [] if collect_logs else None  # engine event logs are plain dicts -> picklable, cross the boundary as-is
    buffers_by_deck, mull_by_deck, played = collect_rollout_league(
        training_deck_name, live_nets, mulligan_nets, deck_ctxs, decklists, pool, reward_fn,
        horizon, n_games, rng, device="cpu", game_logs=worker_logs, checkpoint_rate=checkpoint_rate,
    )
    # Serialize each deck's buffer to picklable entries (identities stripped);
    # mulligan transitions and event logs are already plain data.
    entries_by_deck = {name: _buffer_to_entries(buf) for name, buf in buffers_by_deck.items()}
    if worker_logs:
        worker_logs = _sanitize_events(worker_logs)  # strip unpicklable closures before crossing the boundary
    return entries_by_deck, mull_by_deck, worker_logs, played


def collect_rollout_league_parallel(training_deck_name, live_nets, reward_fn_name, league_root_dir, horizon, n_games,
                                     executor, n_workers, shared_hparams, mulligan_state_dicts=None, game_logs=None,
                                     checkpoint_rate=0.0):
    """Orchestrator (runs in the MAIN process): splits n_games across
    n_workers, submits one _league_rollout_worker task per worker via the
    given (already-created, reused-across-calls) ProcessPoolExecutor --
    reused rather than created fresh per call so process-spawn/import
    overhead (re-importing torch/the game engine in every worker) is paid
    ONCE, not once per collection round. Every deck's live net crosses the
    boundary (not just training_deck_name's own), since a worker needs the
    SAME opponent-sampling capability collect_rollout_league already has
    in-process -- including sampling some OTHER deck's current live net,
    not just training_deck_name's own or frozen snapshots.

    checkpoint_rate: forwarded to every worker's own pool.sample_opponent
    call verbatim -- see rl.league.LeaguePool.sample_opponent's docstring."""
    shared = live_nets[training_deck_name].shared_stack
    shared_state_dict = shared.state_dict()
    all_state_dicts = {name: net.state_dict() for name, net in live_nets.items()}
    all_trunk_hidden = {name: tuple(layer.out_features for layer in net.trunk_layers) for name, net in live_nets.items()}

    base = n_games // n_workers
    remainder = n_games % n_workers
    chunks = [base + (1 if i < remainder else 0) for i in range(n_workers)]

    collect_logs = game_logs is not None
    futures = [
        executor.submit(_league_rollout_worker, training_deck_name, all_state_dicts, all_trunk_hidden,
                         shared_state_dict, shared_hparams, reward_fn_name, league_root_dir, horizon, chunk,
                         random.randrange(2 ** 31), mulligan_state_dicts, collect_logs, checkpoint_rate)
        for chunk in chunks if chunk > 0
    ]
    buffers_by_deck = {}   # deck name -> merged RolloutBuffer across workers
    mull_by_deck = {}      # deck name -> merged mulligan transitions across workers
    games_played = 0
    for future in futures:
        entries_by_deck, worker_mull_by_deck, worker_logs, played = future.result()
        games_played += played
        for name, entries in entries_by_deck.items():
            _extend_buffer_from_entries(buffers_by_deck.setdefault(name, RolloutBuffer()), entries)
        for name, tr in worker_mull_by_deck.items():
            mull_by_deck.setdefault(name, []).extend(tr)
        if game_logs is not None and worker_logs:  # one event_log per game, merged across workers
            game_logs.extend(worker_logs)
    return buffers_by_deck, mull_by_deck, games_played


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
                clip_range=0.2, ent_coef=0.01, vf_coef=0.5, max_grad_norm=0.5):
    # ent_coef default 0.01 (was 0.0): with no entropy bonus the main policy
    # collapses onto a narrow low-branching behavior (pass, shrink its own board) --
    # the action-space-minimization pathology; see rl.rewards.deploy_reward_v2. The
    # mulligan model has its own ENTROPY_COEF; this is the DeckNetwork policy's.
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
    momentum/variance state for the identical shared_stack tensors,
    stepping on them in alternation -- confirmed the hard way (see git
    history) as the exact bug this signature change fixes. Passing a
    single-net-only optimizer as [optimizer] (league training, where the
    shared stack is frozen and only one optimizer ever touches this net's own
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


def train_selfplay(net_a, deck_ctx_a, decklist_a, reward_fn_a, net_b, deck_ctx_b, decklist_b, reward_fn_b,
                    optimizers_a, optimizers_b, horizon, n_iterations, games_per_iteration,
                    rng, device="cpu", game_logs=None):
    """Runs n_iterations rounds of collect_rollout (games_per_iteration real
    games each) + ppo_update. See this module's own docstring for when
    net_a is net_b (mirror self-play, one pooled update) vs. not (cross-matchup,
    two independent updates). optimizers_a/optimizers_b:
    LISTS of optimizers (see ppo_update's own docstring for why -- a net's
    shared_stack may need its own separate optimizer from the net's own
    head). Returns nothing -- both nets and all optimizers are updated in
    place. game_logs: forwarded straight to collect_rollout (see its
    own docstring) -- one entry appended per game, across every iteration."""
    mirror = net_a is net_b
    # AlwaysKeep pregame: train_selfplay trains only the main policy (pretrain /
    # a plain matchup), never a mulligan model. A mirror pools BOTH seats into
    # bucket "a" (single-policy self-play); a cross-matchup keeps "a"/"b".
    agent_a = SeatAgent(net_a, AlwaysKeep(), deck_ctx_a)
    agent_b = agent_a if mirror else SeatAgent(net_b, AlwaysKeep(), deck_ctx_b)
    record_as = ["a", "a"] if mirror else ["a", "b"]
    pairing = _constant_pairing([agent_a, agent_b], [decklist_a, decklist_b],
                                [reward_fn_a, reward_fn_b], record_as)
    for iteration in range(n_iterations):
        buffers_by_deck, _mull, games_played = collect_rollout(
            pairing, games_per_iteration, horizon, rng, device=device, game_logs=game_logs)
        buf_a = buffers_by_deck.get("a", RolloutBuffer())
        buf_b = buffers_by_deck.get("b", RolloutBuffer())
        stats_a = ppo_update(net_a, optimizers_a, buf_a, device) if len(buf_a) else (0.0, 0.0, 0.0)
        if mirror:
            stats_b = stats_a
        else:
            stats_b = ppo_update(net_b, optimizers_b, buf_b, device) if len(buf_b) else (0.0, 0.0, 0.0)
        mean_r_a = float(np.mean(buf_a.reward)) if len(buf_a) else 0.0
        mean_r_b = float(np.mean(buf_b.reward)) if len(buf_b) else 0.0
        print(f"  iter {iteration}: games={games_played} buf=({len(buf_a)},{len(buf_b)}) "
              f"mean_reward=({mean_r_a:.3f},{mean_r_b:.3f}) "
              f"policy_loss=({stats_a[0]:.4f},{stats_b[0]:.4f}) value_loss=({stats_a[1]:.4f},{stats_b[1]:.4f})")
