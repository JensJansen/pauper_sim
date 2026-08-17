"""Multiprocessing plumbing for league rollout collection -- extracted from
rl.train so the game-loop/attribution code (rl.train itself) doesn't have to
carry the ProcessPoolExecutor/pickling-boundary concerns along with it.
_league_rollout_worker (run in each spawned worker process) and
collect_rollout_league_parallel (the main-process orchestrator) both build
on rl.train.collect_rollout_league -- the SAME game loop the sequential path
uses, just re-entered fresh per worker process. Pure reorganization: no
behavior changed by moving the code here."""

import random

import torch

from rl.train import RolloutBuffer, collect_rollout_league


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
    same non-serializable case rl.league_runner._json_default guards at JSON-write time),
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


def _league_rollout_worker(training_deck_name, all_state_dicts, all_trunk_hidden,
                            reward_fn_name, league_root_dir, horizon, n_games, seed,
                            mulligan_state_dicts=None, collect_logs=False, checkpoint_rate=0.0, pfsp=True,
                            pfsp_power=None):
    """Runs in a SEPARATE PROCESS (spawned fresh -- Windows only supports
    the "spawn" start method, no fork, so this re-imports the whole module
    graph from scratch rather than inheriting any parent-process memory).
    Must be a module-level function: ProcessPoolExecutor on spawn needs to
    locate it by import path (rl.rollout_parallel._league_rollout_worker),
    not a closure or lambda.

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
    rather than imported from rl.league_runner to avoid a circular import
    (rl.league_runner already imports FROM this module at module scope)."""
    torch.set_num_threads(1)  # this worker IS the unit of parallelism -- it must not also spawn its own intra-op thread pool and oversubscribe the physical cores every other worker is also competing for
    import rl.rewards as rewards_module
    from rl.arch import SetTransformer
    from rl.deck import DeckNetwork
    from rl.league import LeaguePool
    from rl.pool import build_pool

    reward_fn = getattr(rewards_module, reward_fn_name)
    decklists, vocab, deck_ctxs, fixed_tables = build_pool()

    # One encoder PER DECK, built fresh as a shape and filled from that deck's
    # own state_dict -- the encoder is a registered child of DeckNetwork, so
    # it crosses the process boundary inside all_state_dicts along with the
    # rest of the net. Nothing extra has to be shipped for it.
    live_nets = {}
    for name, state_dict in all_state_dicts.items():
        encoder = SetTransformer(vocab.size)
        net = DeckNetwork(encoder, film_condition_dim=encoder.d_model,
                           non_targeting_n_actions=len(fixed_tables[name]),
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
    # pfsp_power MUST cross the process boundary: each worker builds its own
    # LeaguePool, so a power left at the module default here would silently keep
    # n_workers-1 of every n_workers games on the old weighting.
    pool_kwargs = {} if pfsp_power is None else {"pfsp_power": pfsp_power}
    pool = LeaguePool(league_root_dir, list(all_state_dicts.keys()), **pool_kwargs)  # read-only here -- this worker never calls register_snapshot
    rng = random.Random(seed)

    mulligan_nets = None
    if mulligan_state_dicts is not None:
        from rl.mulligan import MulliganNet
        mulligan_nets = {}
        for name, sd in mulligan_state_dicts.items():
            mn = MulliganNet(live_nets[name].encoder)  # its own deck's encoder, loaded just above
            mn.load_state_dict(sd)
            mn.eval()
            mulligan_nets[name] = mn

    worker_logs = [] if collect_logs else None  # engine event logs are plain dicts -> picklable, cross the boundary as-is
    buffers_by_deck, mull_by_deck, played, outcomes = collect_rollout_league(
        training_deck_name, live_nets, mulligan_nets, deck_ctxs, decklists, pool, reward_fn,
        horizon, n_games, rng, device="cpu", game_logs=worker_logs, checkpoint_rate=checkpoint_rate, pfsp=pfsp,
    )
    # Serialize each deck's buffer to picklable entries (identities stripped);
    # mulligan transitions and event logs are already plain data. outcomes is
    # already plain (str, int-or-None, bool) tuples -- picklable as-is. This
    # worker's own `pool` above is read-only (per its own comment); outcomes
    # is how what it observed gets back to the MAIN process's real pool
    # (collect_rollout_league_parallel aggregates it; _run_session applies it).
    entries_by_deck = {name: _buffer_to_entries(buf) for name, buf in buffers_by_deck.items()}
    if worker_logs:
        worker_logs = _sanitize_events(worker_logs)  # strip unpicklable closures before crossing the boundary
    return entries_by_deck, mull_by_deck, worker_logs, played, outcomes


def collect_rollout_league_parallel(training_deck_name, live_nets, reward_fn_name, league_root_dir, horizon, n_games,
                                     executor, n_workers, all_trunk_hidden,
                                     mulligan_state_dicts=None, game_logs=None, checkpoint_rate=0.0, pfsp=True,
                                     pfsp_power=None):
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

    all_trunk_hidden: every live net's trunk widths -- fixed for the whole
    session (trunk widths never resize), so the caller computes it ONCE
    (rl.league_runner._run_session, right after live_nets is built) instead of
    this function re-deriving it from live_nets on every one of the
    n_iterations * len(train_decks) calls it gets per session. Each net's
    PERCEPTION ENCODER needs no equivalent: it is part of the net, so it ships
    inside all_state_dicts below and is re-sent every call (it trains, so a
    once-per-session broadcast would go stale immediately).

    checkpoint_rate, pfsp: forwarded to every worker's own pool.sample_opponent
    call verbatim -- see rl.league.LeaguePool.sample_opponent's docstring.

    Returns (buffers_by_deck, mull_by_deck, games_played, outcomes) -- outcomes
    merges every worker's own collect_rollout_league outcomes list (see its
    docstring); the caller applies these to the real pool via record_outcome,
    since every worker's pool here is a separate-process, read-only replica."""
    all_state_dicts = {name: net.state_dict() for name, net in live_nets.items()}

    base = n_games // n_workers
    remainder = n_games % n_workers
    chunks = [base + (1 if i < remainder else 0) for i in range(n_workers)]

    collect_logs = game_logs is not None
    futures = [
        executor.submit(_league_rollout_worker, training_deck_name, all_state_dicts, all_trunk_hidden,
                         reward_fn_name, league_root_dir, horizon, chunk,
                         random.randrange(2 ** 31), mulligan_state_dicts, collect_logs, checkpoint_rate, pfsp,
                         pfsp_power)
        for chunk in chunks if chunk > 0
    ]
    buffers_by_deck = {}   # deck name -> merged RolloutBuffer across workers
    mull_by_deck = {}      # deck name -> merged mulligan transitions across workers
    outcomes = []          # (opponent_deck_name, snapshot_id_or_None, training_deck_won), merged across workers --
                            # see collect_rollout_league's own docstring: the caller (_run_session) applies these
                            # to its ONE authoritative pool object, since every worker's own pool is a read-only replica.
    games_played = 0
    for future in futures:
        entries_by_deck, worker_mull_by_deck, worker_logs, played, worker_outcomes = future.result()
        games_played += played
        for name, entries in entries_by_deck.items():
            _extend_buffer_from_entries(buffers_by_deck.setdefault(name, RolloutBuffer()), entries)
        for name, tr in worker_mull_by_deck.items():
            mull_by_deck.setdefault(name, []).extend(tr)
        if game_logs is not None and worker_logs:  # one event_log per game, merged across workers
            game_logs.extend(worker_logs)
        outcomes.extend(worker_outcomes)
    return buffers_by_deck, mull_by_deck, games_played, outcomes
