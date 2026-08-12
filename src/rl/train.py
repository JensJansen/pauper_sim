"""Self-play rollout collection for the token/attention architecture
(rl.features + rl.arch + rl.deck + rl.action_bridge) -- the piece that
actually trains a DeckNetwork. The GAE/PPO-update math lives in rl.ppo, and
the ProcessPoolExecutor multiprocessing plumbing for league collection lives
in rl.rollout_parallel; both build on the game loop (collect_rollout) and
buffer type (RolloutBuffer) defined here.

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

import numpy as np
import torch

import drl_env
import game
from rl import mulligan as mulligan_mod
from rl.agent import SeatAgent, AlwaysKeep
from rl.ppo import ppo_update


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
    LOSER to reach its own (nonzero) loss band, not be forced to 0. A reward_fn
    that wants "loser -> 0" self-contains that check itself
    (rl.rewards.action_count_win_reward), so this stays correct for both."""
    if done:
        return drl_env._for_player(state, seat, lambda s: reward_fn(s, True, horizon))
    return reward_fn(state, False, horizon)


# The per-seat DECISION primitives (_seat_step, _build_decision,
# _scalar_features, _executor_for, _padded_full_mask, _Decision,
# _raise_all_false, _is_pass) live in rl.agent -- collect_rollout drives them
# through SeatAgent.decide rather than calling them directly.
# _reward_for stays here: it's ATTRIBUTION (a rollout concern), not decision.


def _make_on_mana_burn(agents, record, record_as):
    """Builds the on_mana_burn hook game.run_multiplayer_game consults from
    game.turn._empty_mana_pools -- a top-level function (not a closure inline
    in collect_rollout's per-game loop) so it's unit-testable against a real
    deck_ctx/legal_action_mask without needing a full stochastic game (see
    tests/rl/test_train.py::test_on_mana_burn_closure_*)."""
    def on_mana_burn(state, seat):
        """Answers "was anything legally castable with the floating pool" --
        only consulted once neither cost_paid_this_phase nor
        triggers_fired_this_phase already exempted the burn. Checks only rows
        that could actually have spent state.mana_pool: the FIXED half of
        that seat's action table minus its "Play land:" rows (never touch
        the pool, gated on hand membership) and "Tap " mana-ability rows
        (ADD to the pool, gated on source availability -- never spend it) --
        only Cast/Activate rows there are pool-gated via game.mana.
        plan_payment. The pointer half is targeting-only and never itself
        consumes pool mana, so it can't change this answer. Pass is excluded
        explicitly since it's always legal and would make the check vacuous.
        Skips the sweep entirely for an untracked seat (frozen snapshot
        opponent, eval) -- nobody will ever read that seat's
        mana_mistake_burn, so True (no mistake tallied) costs nothing and
        saves the legal_action_mask call."""
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
    """True when at least one TRACKED seat's reward_fn would actually drain
    PlayerState.mana_mistake_burn (rl.rewards.with_mana_mistake_penalty tags
    its own returned closure -- see consumes_mana_mistake there). Lets
    collect_rollout skip building/wiring the on_mana_burn hook -- and the
    legal_action_mask sweep it costs per un-exempted phase boundary -- for a
    pairing where nothing would ever read the signal, e.g. pretraining's
    action_count_win_reward_*."""
    return any(
        record_as[s] is not None and getattr(reward_fns[s], "consumes_mana_mistake", False)
        for s in (0, 1)
    )


def _charge_fns_for(reward_fns):
    """Per-seat rl.rewards.with_dense_mana_burn_penalty.charge_single_pip_burn
    attribute, or None -- the same opt-in-attribute pattern _wants_mana_
    mistake reads off consumes_mana_mistake, one level more specific (the
    actual charge callable, not just a bool) since collect_rollout's own
    on_single_pip_burn hook needs to CALL it, not just know it exists."""
    return [getattr(reward_fns[s], "charge_single_pip_burn", None) for s in (0, 1)]


def _wants_single_pip_burn_hook(charge_fns, record_as):
    """True when at least one TRACKED seat's reward_fn exposes
    charge_single_pip_burn -- lets collect_rollout skip building/wiring the
    on_single_pip_burn hook for a pairing that never uses with_dense_mana_
    burn_penalty (e.g. pretraining's action_count_win_reward_*), same
    reasoning _wants_mana_mistake gives for the older on_mana_burn hook."""
    return any(charge_fns[s] is not None and record_as[s] is not None for s in (0, 1))


def _winner_only_burn_for(reward_fns):
    """Per-seat "does this seat's reward_fn charge mana burn to the WINNER
    ONLY" flag (rl.rewards.with_dense_mana_burn_penalty tags its own closure
    with mana_burn_winner_only when built with refund_on_loss=True -- see
    there). Same opt-in-attribute pattern as _charge_fns_for/
    _wants_mana_mistake, read per seat since a cross-deck pairing could in
    principle mix reward functions.

    When True for a seat, collect_rollout DEFERS that seat's per-Tap burn
    charges instead of writing them into the buffer as they happen, and
    applies them at game end only if that seat won -- see deferred_charges in
    collect_rollout, and with_dense_mana_burn_penalty's own docstring for why
    deferring is not interchangeable with charging-then-refunding."""
    return [bool(getattr(reward_fns[s], "mana_burn_winner_only", False)) for s in (0, 1)]


def _constant_pairing(agents, decklists, reward_fns, record_as):
    """A pairing that yields the SAME layout every game -- mirror, cross-matchup,
    or a fixed A-vs-B matchup. (League resampling is _make_league_pairing.)"""
    def pairing(rng):
        return agents, decklists, reward_fns, record_as
    return pairing


def collect_rollout(pairing, n_games, horizon, rng, device="cpu", record=True, greedy=False, game_logs=None,
                     on_game_end=None):
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
    log_event -- already instrumented; zero-cost when None). on_game_end:
    optional on_game_end(state) callback fired once per game, right after it
    ends -- zero-cost when None, same as game_logs. Generic on purpose (this
    function stays pairing-agnostic, no knowledge of leagues/pools): the
    league-specific use (rl.league.LeaguePool PFSP stat updates, see
    collect_rollout_league) is wired in by the CALLER, not here.

    Returns (buffers_by_deck, mull_by_deck, games_played): dicts keyed by the
    deck-name buckets from record_as. A mirror routes BOTH seats to one bucket
    (pooled single-policy self-play); a live cross-deck opponent routes its own
    seat to its own bucket. Each seat's per-game trajectory is appended to its
    bucket CONTIGUOUSLY and ends done=True, so a later GAE pass never
    bootstraps across a trajectory boundary.

    Dense mana-burn charges (rl.rewards.with_dense_mana_burn_penalty) are
    normally written into the buffer as they happen, attributed to the Tap
    actions that caused them. For a seat whose reward_fn is WINNER-ONLY
    (refund_on_loss=True there), they are instead DEFERRED for the whole game
    and applied at the terminal flush only if that seat won -- see
    deferred_charges below."""
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
            charge_fns = _charge_fns_for(reward_fns)
            # Per-seat credit-assignment bookkeeping for with_dense_mana_burn_
            # penalty's charge_single_pip_burn (see rl.rewards' own docstring
            # for why this exists at all): open_taps holds (buffer_index, pips)
            # for every single-pip Tap action recorded since that seat's own
            # last phase clear -- reset every clear via on_single_pip_burn
            # below, whether or not it burnt anything, so a clean phase can't
            # leave stale entries for a LATER burn to wrongly inherit.
            # prev_single_pip_sum is the sum(mana_pool_single_pip) observed at
            # the START of that seat's own last choose_action call -- an
            # INCREASE since then can only be that seat's own immediately
            # preceding action (nothing else touches ITS pool between two of
            # ITS OWN decisions: taking any non-Pass action never hands
            # priority away, so nothing else runs in between; a burn instead
            # DROPS this to whatever's newly floating post-clear, never up).
            open_taps = [[], []]
            prev_single_pip_sum = [0, 0]
            # deferred_charges[seat]: (buffer_index, share) for every burn
            # charge NOT yet written into the buffer, when that seat's
            # reward_fn is winner-only (rl.rewards' refund_on_loss). WHOLE-GAME
            # scoped -- deliberately NOT reset per phase clear the way
            # open_taps is, since the win/loss that decides whether these ever
            # apply isn't known until the game ends. Applied (or dropped
            # entirely, on a loss) in the terminal-flush block below.
            deferred_charges = [[], []]
            winner_only_burn = _winner_only_burn_for(reward_fns)

            def choose_action(state, agents=agents, reward_fns=reward_fns, record_as=record_as,
                              game_buffers=game_buffers, pending=pending,
                              mull_game=mull_game, open_taps=open_taps,
                              prev_single_pip_sum=prev_single_pip_sum, charge_fns=charge_fns):
                seat = state.active_idx
                current_sum = sum(state.players[seat].mana_pool_single_pip.values())
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
                            # pending[seat] (about to be appended, at the index
                            # game_buffers[seat] is about to get) was a single-pip
                            # Tap iff the pool grew since we last looked AND its
                            # own fixed-table label says so -- gated on the label
                            # too, not just the delta, since only Pass can ever
                            # cross a phase boundary (see prev_single_pip_sum's
                            # own comment above) and Pass structurally can't
                            # itself have added to the pool.
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
                # charge_single_pip_burn reads/mutates PlayerState fields keyed
                # off amount (already applied by game.turn._empty_mana_pools
                # before this hook fires) -- called even when this specific
                # burn is 0 so mana_burn_penalty_credited/_charged_total stay
                # exactly in sync with a caller that DID rely on reward_fn's
                # own (now-passthrough) per-decision calls elsewhere.
                charge = charge_fns[seat](state.players[seat])
                if charge <= 0:
                    return
                total_pips = sum(pips for _, pips in taps)
                distributed = 0.0
                for k, (idx, pips) in enumerate(taps):
                    # last tap absorbs the rounding remainder so shares sum to
                    # EXACTLY `charge`, never more (fine to be exact here --
                    # no downstream code depends on per-share precision).
                    share = charge - distributed if k == len(taps) - 1 else charge * pips / total_pips
                    # Winner-only seats bank the share instead of spending it
                    # now; identical arithmetic either way, only the WHEN
                    # differs (see deferred_charges above).
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
            state = game.run_multiplayer_game(
                decklists=decklists, rng=rng, starting_player_idx=starting_idx,
                choose_action=choose_action, horizon=horizon, combat_enabled=True, event_log=event_log,
                on_mana_burn=on_mana_burn if wants_mana_mistake else None,
                on_single_pip_burn=on_single_pip_burn if wants_single_pip_hook else None,
            )
            # The engine's own win-check (game/effects/win_check.py) sets
            # state.winner/state.turn_won as plain attributes with no log_event
            # call of its own -- nothing in the event stream otherwise records
            # who won (or that the game timed out with no winner). One more
            # state.log_event call here, through the SAME choke point every
            # other instrumented state change already uses, so a consumer of
            # game_logs (win-rate aggregation, the vs-history eval below) never
            # has to reconstruct the outcome by replaying life_change deltas.
            # No-ops (state.event_log is None) when logging is off, same as
            # every other log_event call site. mana_burnt_total/_single_pip
            # are whole-game cumulative diagnostics (game.state.PlayerState),
            # seat-indexed lists here (index == seat, same convention
            # game.turn._empty_mana_pools' own "pools" event uses) -- lets a
            # consumer (run_cross_league_eval.py) compare mana-waste rates
            # across reward policies without replaying the whole event log.
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
                    # Winner-only mana burn (rl.rewards' refund_on_loss): now
                    # that the outcome is known, spend this seat's banked
                    # charges iff it WON. On a loss they are simply dropped --
                    # never written, so the trajectory is bit-for-bit
                    # identical to one that burnt nothing (a terminal refund
                    # would NOT be: GAE discounts it back by
                    # (gamma*gae_lambda)^k per step of distance, leaving early
                    # burns mostly uncancelled -- see with_dense_mana_burn_
                    # penalty's docstring). Ordered deliberately: AFTER the
                    # terminal add above (its index must exist to be charged)
                    # and BEFORE the extend below (which copies values, not
                    # references, so a later write wouldn't reach the bucket).
                    if deferred_charges[seat] and state.winner == seat:
                        for idx, share in deferred_charges[seat]:
                            game_buffers[seat].reward[idx] -= share
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
                            checkpoint_rate=0.0, pfsp=True):
    """League collection: builds a pairing that RESAMPLES the opponent from
    `pool` before every game (true mirror / another deck's live net / a frozen
    snapshot), then runs the ONE loop (collect_rollout). Returns
    (buffers_by_deck, mull_by_deck, games_played, outcomes) keyed by deck name --
    the training deck's bucket (training seat always; BOTH seats on a mirror)
    plus, for any game whose opponent was another deck's CURRENT live net, that
    deck's own bucket (on-policy for it). A frozen-snapshot opponent is
    off-policy and records nothing.

    outcomes: one (opponent_deck_name, snapshot_id_or_None, training_deck_won)
    tuple per game played that reached a winner, for the CALLER to feed into
    pool.record_outcome -- a horizon-timeout game (state.winner is None) is
    excluded entirely rather than counted as a loss, matching the no_winner
    bucketing rl.league_runner._run_eval_vs_history/_run_eval_vs_gauntlet already
    use elsewhere for the same "nobody actually won" case.
    this function and _league_rollout_worker (its parallel-worker twin) never
    write to `pool` themselves (pool.sample_opponent-only here; a worker's own
    pool copy is a separate-process, read-only replica per its own docstring),
    only ever return what happened so the ONE authoritative pool object (living
    in rl.league_runner._run_session) gets updated once, after collection, regardless
    of which path collected the games.

    live_nets / mulligan_nets / deck_ctxs / decklists_by_name: dicts keyed by
    deck name over the WHOLE roster (an opponent may be any other deck's live
    net). mulligan_nets=None -> every seat uses AlwaysKeep (no mulligan dispatch/
    training), e.g. a matchup or deck-only collection. checkpoint_rate, pfsp:
    forwarded to _make_league_pairing/pool.sample_opponent verbatim -- see
    rl.league.LeaguePool.sample_opponent's own docstring."""
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
    """Builds a pairing closure: samples one opponent per game, randomizes which
    seat the training deck takes, wraps each side as a SeatAgent (its live/loaded
    DeckNetwork + its mulligan decider), and sets record_as -- the training
    deck's name for its seat; the opponent's name only if the opponent is a
    mirror (pooled into the training bucket) or another deck's LIVE net (its own
    bucket -- salvage); None for a frozen snapshot (off-policy). Opponent-sampling
    and which-seat-to-record rules live here, in one place.

    checkpoint_rate, pfsp: forwarded to pool.sample_opponent verbatim (see its
    own docstring) -- checkpoint_rate is the live/checkpoint split (default 0.0:
    every game is real-model-vs-real-model), pfsp is whether that split's deck/
    snapshot choice is weighted toward whoever's currently winning against
    training_deck_name (default True) or plain uniform.

    choice_sink: optional mutable dict this closure overwrites every call with
    {'opponent_deck_name', 'snapshot_path', 'training_seat'} -- lets a caller
    (collect_rollout_league's on_game_end) recover which specific opponent THIS
    game was against, after collect_rollout's own game loop finishes it.
    collect_rollout stays pairing-agnostic (no knowledge of leagues/pools), so
    this can't be threaded through its return contract instead."""
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
            # A frozen snapshot loads as a whole frozen SeatAgent (deck + its
            # era-matched mulligan, or AlwaysKeep for a snapshot with no
            # mulligan state).
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
        if choice_sink is not None:
            choice_sink["opponent_deck_name"] = opp_name
            choice_sink["snapshot_path"] = snapshot_path
            choice_sink["training_seat"] = training_seat
        return agents, decklists, reward_fns, record_as

    return pairing


def batch_size_for_iteration(cumulative_games, horizon_games=50_000, start=32, cap=2048, n_steps=6):
    """Small, granular minibatches early in training, doubling toward a
    larger cap as training progresses -- the "increase batch size instead
    of decaying the learning rate" schedule shape (Smith et al. 2017: more
    frequent, noisier early gradient steps to cover ground quickly on a
    policy far from any optimum; larger, smoother steps later as it
    approaches convergence), not the reverse. n_steps doublings spread evenly
    across the run (a step schedule, not continuous growth -- simplest thing
    that matches "incrementally increasing").

    Tracked against CUMULATIVE games/deck (progress.json's own
    cumulative_games_per_deck, threaded in by the caller), not a session-
    local iteration/n_iterations pair as this used to be -- a real training
    run is many separate `run_league.py` process invocations (the
    escalation-ladder methodology: start tiny, double each clean batch), and
    the old session-local version reset all the way back to `start` at the
    beginning of EVERY one of those invocations regardless of how far the
    overall run had actually progressed, undermining the whole point of a
    ramp meant to span one run's worth of convergence, not one batch's.
    horizon_games=50,000 is a first estimate, not derived from an ablation --
    picked because the observed entropy-collapse/plateau pattern this was
    introduced to help with (TRAINING_IMPROVEMENT_OPTIONS.md section 4) was
    already fully established well before 50,000 cumulative games/deck in
    the run that motivated this change."""
    if horizon_games <= 0:
        return cap
    progress = min(1.0, cumulative_games / horizon_games)
    doublings = int(progress * n_steps)
    return min(start * (2 ** doublings), cap)


def ent_coef_schedule(cumulative_games, start=0.02, floor=0.005, horizon_games=50_000):
    """PPO entropy-bonus coefficient, linearly annealed from `start` down to
    `floor` over `horizon_games` cumulative games/deck, then held at
    `floor` -- replaces a FIXED ent_coef=0.01 for the whole life of a run.

    Motivation (TRAINING_IMPROVEMENT_OPTIONS.md section 4): on a real
    34,579-games/deck run, PPO exploration entropy (masked-categorical, over
    the combined fixed-action + pointer-target logits) collapsed from ~1.2
    nats at session 0 to a floor of ~0.2-0.3 nats by session ~9 (roughly
    250-300 cumulative games/deck) and never recovered for the remaining
    30,000+ games, under the OLD fixed ent_coef=0.01. A near-deterministic
    policy that locked in that early has no exploratory margin left to
    adapt when the self-play opponent pool's mix later shifts (new
    snapshots rotating in, PFSP re-weighting) -- a plausible mechanism for
    the rise-then-regress cycling also observed in that run's vs_gauntlet
    trend, independent of anything wrong with the reward or opponent pool
    on their own.

    start/floor/horizon_games are first estimates, not swept -- start
    (0.02, double the old fixed value) and floor (0.005, half the old fixed
    value) bracket the old constant on either side rather than assuming
    which direction was wrong; horizon_games matches batch_size_for_
    iteration's own (see its docstring for why 50,000). Deliberately a
    SEPARATE function from batch_size_for_iteration, even though both take
    the same cumulative_games/horizon_games shape, since the two schedules
    have no reason to share a single curve -- callers pass cumulative_games
    to both independently."""
    if horizon_games <= 0:
        return floor
    progress = min(1.0, cumulative_games / horizon_games)
    return start + (floor - start) * progress


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
        stats_a = ppo_update(net_a, optimizers_a, buf_a, device) if len(buf_a) else (0.0, 0.0, 0.0, 0.0, 0.0, 0)
        if mirror:
            stats_b = stats_a
        else:
            stats_b = ppo_update(net_b, optimizers_b, buf_b, device) if len(buf_b) else (0.0, 0.0, 0.0, 0.0, 0.0, 0)
        mean_r_a = float(np.mean(buf_a.reward)) if len(buf_a) else 0.0
        mean_r_b = float(np.mean(buf_b.reward)) if len(buf_b) else 0.0
        print(f"  iter {iteration}: games={games_played} buf=({len(buf_a)},{len(buf_b)}) "
              f"mean_reward=({mean_r_a:.3f},{mean_r_b:.3f}) "
              f"policy_loss=({stats_a[0]:.4f},{stats_b[0]:.4f}) value_loss=({stats_a[1]:.4f},{stats_b[1]:.4f})")
