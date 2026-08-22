"""Multiprocessing plumbing for league rollout collection, extracted from
rl.training.train so the game-loop/attribution code doesn't carry the
ProcessPoolExecutor/pickling-boundary concerns. _league_rollout_worker (run
in each spawned worker) and collect_rollout_league_parallel (the
main-process orchestrator) both build on rl.training.train.collect_rollout_league
-- the same game loop the sequential path uses, re-entered fresh per
worker."""

import random

import torch

from rl.training.train import RolloutBuffer, collect_rollout_league


def _strip_identities(token_list):
    """Drops each token's Permanent-object identity before it crosses a
    process boundary -- unused downstream (pad_token_batch discards it once
    buffered; identity only matters live, during collection), and a live
    Permanent may not even be picklable."""
    return [(idx, row, None) for idx, row, _identity in token_list]


def _buffer_to_entries(buf):
    """Serializes a RolloutBuffer to plain (picklable) tuples for the
    process boundary -- identities stripped (see _strip_identities)."""
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
    """Deep-converts event-log values to picklable primitives -- a few
    log_event fields can hold a card closure/lambda, which pickle can't
    ship. Converts an unknown object to its .name if present, else repr()."""
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
    "spawn", so this re-imports the whole module graph). Must be
    module-level: ProcessPoolExecutor on spawn locates it by import path.

    Rebuilds decklists/vocab/deck_ctxs/fixed_tables locally via build_pool()
    rather than receiving them from the parent -- fixed_table entries hold
    legal_fn/execute_fn closures, which pickle cannot serialize. Only plain
    tensors, scalars, and strings cross the process boundary; reward_fn is
    passed by name for the same reason. league_root_dir is passed
    explicitly to avoid a circular import with rl.league.league_runner."""
    torch.set_num_threads(1)  # avoid oversubscribing cores: this worker IS the unit of parallelism
    import rl.rewards as rewards_module
    from rl.model.arch import SetTransformer
    from rl.model.deck import DeckNetwork
    from rl.league.league import LeaguePool
    from rl.roster import build_pool

    reward_fn = getattr(rewards_module, reward_fn_name)
    decklists, vocab, deck_ctxs, fixed_tables = build_pool()

    # One encoder per deck, built fresh as a shape and filled from that
    # deck's own state_dict (a registered child of DeckNetwork).
    live_nets = {}
    for name, state_dict in all_state_dicts.items():
        encoder = SetTransformer(vocab.size)
        net = DeckNetwork(encoder, film_condition_dim=encoder.d_model,
                           non_targeting_n_actions=len(fixed_tables[name]),
                           trunk_hidden=all_trunk_hidden[name])
        net.load_state_dict(state_dict)
        net.eval()
        live_nets[name] = net

    # Deck names come from all_state_dicts' own keys (the possibly-narrowed
    # roster the orchestrator built), not this worker's own full build_pool().
    # pfsp_power must cross the process boundary: each worker builds its own
    # LeaguePool, so a default left here would silently apply to n_workers-1
    # of every n_workers games.
    pool_kwargs = {} if pfsp_power is None else {"pfsp_power": pfsp_power}
    pool = LeaguePool(league_root_dir, list(all_state_dicts.keys()), **pool_kwargs)  # read-only here -- this worker never calls register_snapshot
    rng = random.Random(seed)

    mulligan_nets = None
    if mulligan_state_dicts is not None:
        from rl.model.mulligan import MulliganNet
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
    # Serialize each deck's buffer to picklable entries; mulligan
    # transitions and event logs are already plain data. `pool` above is
    # read-only -- outcomes carries what happened back to the main process.
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
    given (reused-across-calls) ProcessPoolExecutor, so process-spawn/import
    overhead is paid once, not once per collection round. Every deck's live
    net crosses the boundary, not just training_deck_name's own, since a
    worker needs the same opponent-sampling capability the in-process path
    has.

    all_trunk_hidden: every live net's trunk widths, computed once by the
    caller (rl.league.league_runner._run_session) rather than re-derived
    here on every call. Each net's encoder ships inside all_state_dicts
    instead, since it trains and would go stale if only sent once per
    session.

    checkpoint_rate, pfsp: forwarded to every worker's pool.sample_opponent
    verbatim.

    Returns (buffers_by_deck, mull_by_deck, games_played, outcomes) --
    outcomes merges every worker's own outcomes list; the caller applies
    these to the real pool via record_outcome, since each worker's pool is
    a separate-process, read-only replica."""
    # .cpu(): workers are separate processes with no CUDA context, and they
    # always play on CPU regardless of where training runs. A no-op when
    # training is already on CPU.
    all_state_dicts = {name: {k: v.cpu() for k, v in net.state_dict().items()}
                       for name, net in live_nets.items()}

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
    outcomes = []          # (opponent_deck_name, snapshot_id_or_None, training_deck_won), merged across workers
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
