"""Self-play rollout collection for the token/attention architecture -- the
piece that trains a DeckNetwork. GAE/PPO-update math lives in rl.training.ppo;
ProcessPoolExecutor plumbing lives in rl.training.rollout_parallel; both
build on the game loop (collect_rollout) and buffer type (RolloutBuffer)
defined here.

League training (rl.league.league_runner._run_session) drives collect_rollout
via _make_league_pairing/collect_rollout_league, which resamples an
opponent from rl.league.league.LeaguePool every game -- mirror self-play and
cross-deck matchups both fall out of what LeaguePool samples. Each seat
keeps its own recurrent state (SeatAgent keys hidden state by seat): two
seats sharing a net are one policy, never one history.

The per-game GRU state lives on the SeatAgent and is cleared once per game
by the loop below -- load-bearing, since ppo_update replays every episode
from a zero state.

Reward attribution: 0.0 if drl_env._lost(state, seat), else
reward_fn(state, done, horizon), _for_player-flipped at the true end."""

import torch

import drl_env
import game
from rl.model import mulligan as mulligan_mod
from rl.decision.agent import SeatAgent, AlwaysKeep


# Decisions one seat may take within a SINGLE turn before collect_rollout
# treats the priority round as non-progressing and raises -- `horizon`
# bounds turns, not decisions within one, so this is what stops a
# legal-but-non-progressing action from hanging forever. ~80x the largest
# legitimate turn ever measured here (61 decisions across 140 games).
MAX_DECISIONS_PER_TURN = 5000


def _non_progressing_report(state, seat, n_decisions):
    """The message the turn guard raises -- a full diagnostic (attacking
    creatures, eligible blockers, existing block assignments), not just a
    count, since a loop like this is rare and policy-dependent and
    re-catching one can cost hours."""
    pend = state.pending_resolution
    lines = [
        f"non-progressing priority round: {n_decisions} decisions in turn {state.turn_number} "
        f"(limit {MAX_DECISIONS_PER_TURN}). Some action is legal but makes no progress.",
        f"  phase={getattr(state.phase, 'value', state.phase)} "
        f"pending={pend['kind'] if pend else None} seat={seat} active={state.active_idx} "
        f"turn_player={state.turn_player_idx} stack={len(state.stack)} "
        f"battlefield={len(state.battlefield)}",
    ]
    try:  # diagnostics must never mask the failure they describe
        from game.effects.stats import creature_keywords
        from game.effects.combat import creature_block_eligible

        def desc(p):
            kws = ",".join(sorted(creature_keywords(state, p))) or "-"
            return f"{p.card_def.name}(slot {p.slot}, {kws})"

        opp = state.opponent
        lines.append(f"  attackers: {[desc(a) for a in (opp.attackers or ())]}")
        lines.append(f"  eligible_blockers: {[desc(p) for p in state.battlefield if creature_block_eligible(state, p)]}")
        lines.append("  blocked_by: " + str({
            f"{a.card_def.name}#{a.slot}": [f"{b.card_def.name}#{b.slot}" for b in bs]
            for a, bs in (opp.blocked_by or {}).items()}))
    except Exception as exc:  # noqa: BLE001
        lines.append(f"  (combat detail unavailable: {type(exc).__name__}: {exc})")
    return "\n".join(lines)


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
        """Appends every transition from another buffer -- pools both seats
        (mirror self-play) or merges per-game league buffers."""
        for i in range(len(other)):
            self.add(other.token_lists[i], other.scalar[i], other.mask[i], other.action[i],
                     other.logp[i], other.value[i], other.reward[i], other.done[i])

    def clear(self):
        self.__init__()


def _reward_for(state, seat, reward_fn, horizon, done):
    """Per-seat reward, computed seat-relative (drl_env._for_player flips
    state.active_idx to `seat`) on every call, terminal or not -- lets a
    reward_fn be dense. No external loser gate: reward_fn itself decides win
    vs loss vs no-winner."""
    return drl_env._for_player(state, seat, lambda s: reward_fn(s, done, horizon))


# Per-seat DECISION primitives live in rl.decision.agent -- collect_rollout
# drives them through SeatAgent.decide. _reward_for stays here (attribution).


def _make_on_mana_burn(agents, record, record_as):
    """Builds the on_mana_burn hook game.run_multiplayer_game consults --
    a top-level function so it's unit-testable independent of a full game."""
    def on_mana_burn(state, seat):
        """Answers whether anything was legally castable with the floating
        pool. Checks only fixed-table rows that could have spent
        state.mana_pool (Cast/Activate rows via game.mana.plan_payment) --
        excludes "Play land:" and "Tap " rows, and Pass. Skips the sweep
        for an untracked seat (nobody reads its mana_mistake_burn)."""
        if not record or record_as[seat] is None:
            return True
        fixed_table = agents[seat].deck_ctx[1]
        mask = drl_env._for_player(state, seat, lambda s: drl_env.legal_action_mask(s, fixed_table))
        return any(
            legal and name != "Pass"
            and not name.startswith("Play land:") and not name.startswith("Tap ")
            for legal, (name, _l, _e) in zip(mask, fixed_table)
        )
    return on_mana_burn


def _wants_mana_mistake(reward_fns, record_as):
    """True when at least one tracked seat's reward_fn opts in via
    consumes_mana_mistake=True -- lets collect_rollout skip building the
    on_mana_burn hook when nothing would read the signal."""
    return any(
        record_as[s] is not None and getattr(reward_fns[s], "consumes_mana_mistake", False)
        for s in (0, 1)
    )


def _charge_fns_for(reward_fns):
    """Per-seat charge_single_pip_burn attribute (or None) from
    with_dense_mana_burn_penalty -- same opt-in-attribute pattern as
    _wants_mana_mistake, one level more specific since collect_rollout needs
    to call it."""
    return [getattr(reward_fns[s], "charge_single_pip_burn", None) for s in (0, 1)]


def _wants_single_pip_burn_hook(charge_fns, record_as):
    """True when at least one tracked seat's reward_fn exposes
    charge_single_pip_burn -- lets collect_rollout skip wiring the
    on_single_pip_burn hook otherwise."""
    return any(charge_fns[s] is not None and record_as[s] is not None for s in (0, 1))


def _winner_only_burn_for(reward_fns):
    """Per-seat "does this reward_fn charge mana burn to the winner only"
    flag (with_dense_mana_burn_penalty tags its closure with
    mana_burn_winner_only when refund_on_loss=True). When True,
    collect_rollout defers that seat's per-Tap charges and applies them at
    game end only if it won -- see deferred_charges below."""
    return [bool(getattr(reward_fns[s], "mana_burn_winner_only", False)) for s in (0, 1)]


def _constant_pairing(agents, decklists, reward_fns, record_as):
    """A pairing that yields the SAME layout every game -- mirror, cross-matchup,
    or a fixed A-vs-B matchup. (League resampling is _make_league_pairing.)"""
    def pairing(rng):
        return agents, decklists, reward_fns, record_as
    return pairing


def collect_rollout(pairing, n_games, horizon, rng, device="cpu", record=True, greedy=False, game_logs=None,
                     on_game_end=None, stratify_0land_pct=0.0, stratify_7land_pct=0.0):
    """The ONE game loop. Plays n_games real self-play games
    (game.run_multiplayer_game); a per-game `pairing(rng)` supplies that
    game's layout, and the SeatAgents it returns own the mulligan-vs-policy
    dispatch. This function owns only ATTRIBUTION (bandit mulligan reward +
    per-decision terminal-flush PPO reward) and bucketing.

    pairing(rng) -> (agents, decklists, reward_fns, record_as):
      agents[seat]      -- SeatAgent making seat `seat`'s decisions
      decklists[seat]   -- that seat's decklist (for run_multiplayer_game)
      reward_fns[seat]  -- that seat's reward_fn (may be None when record=False)
      record_as[seat]   -- the deck-name bucket to record this seat's
                           transitions into, or None for a frozen-snapshot /
                           off-policy opponent, or eval.

    record=False -> pure play (eval / log generation), buffers nothing.
    greedy=True -> deterministic argmax (eval). game_logs: optional list,
    one engine event_log appended per game. on_game_end: optional
    on_game_end(state) callback, fired once per game right after it ends.
    This function stays pairing-agnostic (no knowledge of leagues/pools) --
    the league-specific use (LeaguePool PFSP stat updates) is wired in by
    the caller.

    stratify_0land_pct=0.0, stratify_7land_pct=0.0 (both default, no extra
    rng draw when both are 0.0): a TRAINING-ONLY knob that, with the given
    per-game probability, forces the single recorded seat's opening hand to
    0 or 7 lands (game.state.build_shuffled_library's force_land_count).
    Mutually exclusive per game. Exists because both tails are naturally
    rare -- a flooded 7-land hand occurs on the order of 0.001% of natural
    deals, so without forcing it, a batch essentially never contains one for
    the net to learn hand quality isn't monotonic in land count. Left
    unstratified if more than one seat (or zero) is recorded.

    Returns (buffers_by_deck, mull_by_deck, games_played): dicts keyed by
    the deck-name buckets from record_as. A mirror routes both seats to one
    bucket; a live cross-deck opponent routes its own seat to its own
    bucket. Each seat's trajectory is appended to its bucket contiguously
    and ends done=True, so GAE never bootstraps across a trajectory
    boundary.

    Dense mana-burn charges (with_dense_mana_burn_penalty) are normally
    written into the buffer as they happen. For a WINNER-ONLY seat
    (refund_on_loss=True) they are instead deferred for the whole game and
    applied at the terminal flush only if that seat won -- see
    deferred_charges below."""
    buffers_by_deck = {}   # bucket -> RolloutBuffer (main PPO transitions)
    mull_by_deck = {}      # bucket -> list of mulligan transitions (bandit)
    games_played = 0

    # Batch-of-1 inference: torch's intra-op threading is pure overhead here
    # (~1.66x measured). Forced to 1 thread for the game loop; restored after.
    prev_threads = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        for _ in range(n_games):
            agents, decklists, reward_fns, record_as = pairing(rng)
            # Clear every agent's recurrent state before the game starts (the
            # ONE place this happens -- LeaguePool hands back the same agent
            # game after game). Deduped by identity since a mirror pairing
            # puts one agent on both seats. Guarded by hasattr since not
            # every agent type defines reset().
            for agent in {id(a): a for a in agents}.values():
                if hasattr(agent, "reset"):
                    agent.reset()
            game_buffers = [RolloutBuffer(), RolloutBuffer()]  # per-game per-seat, kept contiguous
            pending = [None, None]
            mull_game = [[], []]
            charge_fns = _charge_fns_for(reward_fns)
            # Per-seat bookkeeping for charge_single_pip_burn: open_taps
            # holds (buffer_index, pips) for every single-pip Tap recorded
            # since that seat's last phase clear, reset on each clear.
            open_taps = [[], []]
            # prev_single_pip_sum: sum(mana_pool_single_pip) at the START of
            # this seat's last choose_action call -- an increase since then
            # can only be that seat's own immediately preceding action.
            prev_single_pip_sum = [0, 0]
            # deferred_charges[seat]: (buffer_index, share) for burn charges
            # not yet written, for a winner-only reward_fn -- whole-game
            # scoped (not reset per phase clear), applied or dropped at the
            # terminal flush once the outcome is known.
            deferred_charges = [[], []]
            winner_only_burn = _winner_only_burn_for(reward_fns)

            # Turn-decision guard state: [turn_number_being_counted, count].
            # A list, not two locals, so the closure below can mutate it.
            turn_watch = [None, 0]

            def choose_action(state, agents=agents, reward_fns=reward_fns, record_as=record_as,
                              game_buffers=game_buffers, pending=pending,
                              mull_game=mull_game, open_taps=open_taps,
                              prev_single_pip_sum=prev_single_pip_sum, charge_fns=charge_fns,
                              turn_watch=turn_watch):
                seat = state.active_idx
                # NON-PROGRESSING-LOOP GUARD: `horizon` bounds turn_number,
                # which doesn't advance inside a priority round, so a
                # legal-but-non-progressing action would otherwise spin
                # forever, growing the buffer until the worker dies. Capping
                # a priority round in the ENGINE would be a real rules
                # deviation, so the guard lives in the harness instead.
                # MAX_DECISIONS_PER_TURN is ~80x the largest legitimate turn
                # ever measured here.
                if state.turn_number != turn_watch[0]:
                    turn_watch[0], turn_watch[1] = state.turn_number, 0
                turn_watch[1] += 1
                if turn_watch[1] > MAX_DECISIONS_PER_TURN:
                    raise RuntimeError(_non_progressing_report(state, seat, turn_watch[1]))
                current_sum = sum(state.players[seat].mana_pool_single_pip.values())
                dr = agents[seat].decide(state, seat, horizon, device, greedy=greedy)
                if record and record_as[seat] is not None:
                    # Mulligan transition: whole-game bandit reward, filled at game end.
                    if dr.mull_entry is not None:
                        mull_game[seat].append(dr.mull_entry)
                    # Main-policy transition: ppo_entry is None for a forced
                    # decision or pregame (mulligan-owned) -- record nothing,
                    # leaving `pending` for the last real decision's terminal reward.
                    if dr.ppo_entry is not None:
                        if pending[seat] is not None:
                            reward = _reward_for(state, seat, reward_fns[seat], horizon, False)
                            # pending[seat] (about to be appended) was a
                            # single-pip Tap iff the pool grew since last
                            # look AND the fixed-table label says so.
                            if charge_fns[seat] is not None and current_sum > prev_single_pip_sum[seat]:
                                prev_action_idx = pending[seat][3]
                                fixed_table = agents[seat].deck_ctx[1]
                                if prev_action_idx < len(fixed_table) and fixed_table[prev_action_idx][0].startswith("Tap"):
                                    open_taps[seat].append((len(game_buffers[seat]), 1))
                            game_buffers[seat].add(*pending[seat], reward, False)
                        pending[seat] = dr.ppo_entry
                    prev_single_pip_sum[seat] = current_sum
                return None if dr.is_pass else dr.executor

            def on_single_pip_burn(state, seat, amount, game_buffers=game_buffers, open_taps=open_taps,
                                    record_as=record_as, charge_fns=charge_fns,
                                    deferred_charges=deferred_charges, winner_only_burn=winner_only_burn):
                taps, open_taps[seat] = open_taps[seat], []
                if charge_fns[seat] is None or record_as[seat] is None or not taps:
                    return
                # charge_single_pip_burn reads/mutates PlayerState fields for
                # `amount` (already applied by _empty_mana_pools). Called
                # even when this burn is 0 so credited/charged_total stay in sync.
                charge = charge_fns[seat](state.players[seat])
                if charge <= 0:
                    return
                total_pips = sum(pips for _, pips in taps)
                distributed = 0.0
                for k, (idx, pips) in enumerate(taps):
                    # last tap absorbs the rounding remainder so shares sum to
                    # EXACTLY `charge`, never more.
                    share = charge - distributed if k == len(taps) - 1 else charge * pips / total_pips
                    # Winner-only seats bank the share instead of spending it now.
                    if winner_only_burn[seat]:
                        deferred_charges[seat].append((idx, share))
                    else:
                        game_buffers[seat].reward[idx] -= share
                    distributed += share

            on_mana_burn = _make_on_mana_burn(agents, record, record_as)
            wants_mana_mistake = _wants_mana_mistake(reward_fns, record_as)
            wants_single_pip_hook = _wants_single_pip_burn_hook(charge_fns, record_as)

            starting_idx = rng.randint(0, 1)
            event_log = [] if game_logs is not None else None
            stratify = None
            if stratify_0land_pct > 0.0 or stratify_7land_pct > 0.0:
                recorded_seats = [s for s in (0, 1) if record_as[s] is not None]
                if len(recorded_seats) == 1:
                    roll = rng.random()
                    if roll < stratify_0land_pct:
                        stratify = (recorded_seats[0], 0)
                    elif roll < stratify_0land_pct + stratify_7land_pct:
                        stratify = (recorded_seats[0], 7)
            state = game.run_multiplayer_game(
                decklists=decklists, rng=rng, starting_player_idx=starting_idx,
                choose_action=choose_action, horizon=horizon, combat_enabled=True, event_log=event_log,
                on_mana_burn=on_mana_burn if wants_mana_mistake else None,
                on_single_pip_burn=on_single_pip_burn if wants_single_pip_hook else None,
                stratify=stratify,
            )
            # The engine's own win-check sets state.winner/state.turn_won
            # with no log_event of its own -- log it here (the same choke
            # point every other state change uses) so a consumer of
            # game_logs never has to reconstruct the outcome by replaying
            # deltas. mana_burnt_total*/single_pip are whole-game cumulative
            # diagnostics, seat-indexed.
            state.log_event(
                "game_over", winner=state.winner, turn_won=state.turn_won,
                mana_burnt_total=[p.mana_burnt_total for p in state.players],
                mana_burnt_total_single_pip=[p.mana_burnt_total_single_pip for p in state.players],
            )
            if on_game_end is not None:
                on_game_end(state)
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
                    # Winner-only mana burn: spend this seat's banked
                    # charges iff it won; on a loss they're simply dropped
                    # (bit-for-bit identical to burning nothing -- a
                    # terminal refund would not be, see
                    # with_dense_mana_burn_penalty's docstring). Ordered
                    # after the terminal add (index must exist) and before
                    # the extend below (which copies values, not references).
                    if deferred_charges[seat] and state.winner == seat:
                        for idx, share in deferred_charges[seat]:
                            game_buffers[seat].reward[idx] -= share
                    if len(game_buffers[seat]):  # append this seat's whole (contiguous) trajectory to its bucket
                        buffers_by_deck.setdefault(bucket, RolloutBuffer()).extend(game_buffers[seat])
                    if mull_game[seat]:  # attribute the game's outcome to this seat's mulligan picks (bandit)
                        r = mulligan_mod.mulligan_reward(state.winner == seat)
                        for entry in mull_game[seat]:
                            entry[5] = r
                        mull_by_deck.setdefault(bucket, []).extend(mull_game[seat])
            games_played += 1
    finally:
        torch.set_num_threads(prev_threads)
    return buffers_by_deck, mull_by_deck, games_played


def collect_rollout_league(training_deck_name, live_nets, mulligan_nets, deck_ctxs, decklists_by_name,
                            pool, reward_fn, horizon, n_games, rng, device="cpu", record=True, game_logs=None,
                            checkpoint_rate=0.0, pfsp=True):
    """League collection: builds a pairing that resamples the opponent from
    `pool` before every game, then runs collect_rollout. Returns
    (buffers_by_deck, mull_by_deck, games_played, outcomes) keyed by deck
    name -- the training deck's bucket, plus (for a game whose opponent was
    another deck's current live net) that deck's own bucket. A frozen-
    snapshot opponent is off-policy and records nothing.

    outcomes: one (opponent_deck_name, snapshot_id_or_None, training_deck_won)
    tuple per game that reached a winner, for the caller to feed into
    pool.record_outcome -- a horizon-timeout game is excluded rather than
    counted as a loss. Neither this function nor its parallel-worker twin
    write to `pool` themselves; only the one authoritative pool object
    (rl.league.league_runner._run_session) gets updated, after collection.

    live_nets / mulligan_nets / deck_ctxs / decklists_by_name: dicts keyed
    by deck name over the whole roster. mulligan_nets=None -> every seat
    uses AlwaysKeep. checkpoint_rate, pfsp: forwarded to
    _make_league_pairing/pool.sample_opponent verbatim."""
    choice_sink = {}
    outcomes = []
    pairing = _make_league_pairing(training_deck_name, live_nets, mulligan_nets, deck_ctxs,
                                    decklists_by_name, pool, reward_fn, checkpoint_rate=checkpoint_rate,
                                    choice_sink=choice_sink, pfsp=pfsp)

    def on_game_end(state):
        if state.winner is None:  # horizon timeout, nobody won -- exclude, don't count as a loss
            return
        opp_name = choice_sink["opponent_deck_name"]
        snapshot_path = choice_sink["snapshot_path"]
        training_seat = choice_sink["training_seat"]
        snap_id = None
        if snapshot_path is not None:
            snap_id = next((sid for sid, path in pool.snapshots[opp_name] if path == snapshot_path), None)
        outcomes.append((opp_name, snap_id, state.winner == training_seat))

    buffers_by_deck, mull_by_deck, played = collect_rollout(
        pairing, n_games, horizon, rng, device=device, record=record, game_logs=game_logs, on_game_end=on_game_end)
    return buffers_by_deck, mull_by_deck, played, outcomes


def _make_league_pairing(training_deck_name, live_nets, mulligan_nets, deck_ctxs, decklists_by_name, pool, reward_fn,
                          checkpoint_rate=0.0, choice_sink=None, pfsp=True):
    """Builds a pairing closure: samples one opponent per game, randomizes
    which seat the training deck takes, wraps each side as a SeatAgent, and
    sets record_as -- the training deck's name for its seat; the
    opponent's name only if it's a mirror or another deck's live net
    (salvage); None for a frozen snapshot (off-policy).

    checkpoint_rate, pfsp: forwarded to pool.sample_opponent verbatim.

    choice_sink: optional mutable dict this closure overwrites every call
    with {'opponent_deck_name', 'snapshot_path', 'training_seat'}, so a
    caller (collect_rollout_league's on_game_end) can recover which
    opponent a given game was against."""
    training_net = live_nets[training_deck_name]
    training_ctx = deck_ctxs[training_deck_name]
    training_decklist = decklists_by_name[training_deck_name]

    def _mull(name):
        return mulligan_nets[name] if mulligan_nets is not None else AlwaysKeep()

    def pairing(rng):
        opp_name, snapshot_path = pool.sample_opponent(training_deck_name, rng, checkpoint_rate=checkpoint_rate, pfsp=pfsp)
        is_self = snapshot_path is None and opp_name == training_deck_name
        opp_is_live = snapshot_path is None and not is_self  # another deck's current net -> salvageable

        train_agent = SeatAgent(training_net, _mull(training_deck_name), training_ctx)
        if is_self:
            opp_agent = train_agent  # true mirror: same net + mulligan on both seats
        elif snapshot_path is None:
            opp_agent = SeatAgent(live_nets[opp_name], _mull(opp_name), deck_ctxs[opp_name])
        else:
            # A frozen snapshot loads as a whole frozen SeatAgent (deck +
            # its era-matched mulligan/encoder).
            opp_agent = pool.load_snapshot_agent(snapshot_path, deck_ctxs[opp_name])

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
        if choice_sink is not None:
            choice_sink["opponent_deck_name"] = opp_name
            choice_sink["snapshot_path"] = snapshot_path
            choice_sink["training_seat"] = training_seat
        return agents, decklists, reward_fns, record_as

    return pairing


def batch_size_for_iteration(cumulative_games, horizon_games=50_000, start=4, cap=16, n_steps=6):
    """Minibatch size for this point in the run, in EPISODES (see
    rl.training.ppo.ppo_update). Small, granular minibatches early in
    training, doubling toward `cap` as training progresses -- more frequent,
    noisier steps while the policy is far from converged, larger smoother
    steps as it approaches it. n_steps doublings spread evenly across the
    run (a step schedule, not continuous growth).

    Tracked against CUMULATIVE games/deck (progress.json's
    cumulative_games_per_deck), not a session-local iteration count, since a
    real run spans many separate process invocations."""
    if horizon_games <= 0:
        return cap
    progress = min(1.0, cumulative_games / horizon_games)
    doublings = int(progress * n_steps)
    return min(start * (2 ** doublings), cap)


def ent_coef_schedule(cumulative_games, start=0.02, floor=0.005, horizon_games=50_000):
    """PPO entropy-bonus coefficient, linearly annealed from `start` down to
    `floor` over `horizon_games` cumulative games/deck, then held at
    `floor` -- replaces a fixed ent_coef for the whole life of a run, since a
    policy that locks in low entropy early has no exploratory margin left to
    adapt when the self-play opponent pool later shifts.

    start/floor/horizon_games are first estimates, not swept.
    horizon_games matches batch_size_for_iteration's own."""
    if horizon_games <= 0:
        return floor
    progress = min(1.0, cumulative_games / horizon_games)
    return start + (floor - start) * progress
