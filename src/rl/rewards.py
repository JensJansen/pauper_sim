"""Reward functions for training a DRL policy against the game engine.

Contract: any callable

    reward_fn(state: game.GameState, done: bool, horizon: int) -> float

Called once per environment step with the state *after* that step's
action was applied. A sparse reward function returns 0.0 unless `done`;
a dense one could return something every call. No base class -- any
matching callable works.

Terminal WIN/LOSS scoring: a win is scaled by how efficiently it was reached,
minus a "sloppiness" penalty for hoarded cards forced to cleanup discard (see
_hill); a loss/timeout is 0.0 minus that same penalty (action_count_win_reward
has no such penalty -- it's the plain, pretrain-only predecessor). Pretraining
uses action_count_win_reward_200_floor02; league self-play uses
deploy_reward_v2 (= deploy_reward with win_floor=1.0 -> flat efficiency term;
the v1 efficiency scaling caused an action-space-minimization pathology).

DENSE shaping (with_dense_mana_burn_penalty, below) IS wired into
deploy_reward_v2 -- see deploy_reward_v2's own comment for its history: an
unconditional first version was reverted 2026-08 (archetype bias against
Elves' Priest of Titania), then re-enabled 2026-08 reading a PER-PIP
single-pip-tagged subset of mana burnt (PlayerState.
mana_burnt_this_turn_single_pip) instead of the raw total -- game.mana.
float_mana/spend_one_pip tag and spend individual floating pips (see their
own docstrings, and PlayerState.mana_pool_single_pip's), which structurally
excludes Priest-style board-state-scaled bursts from the penalty while still
catching avoidable single-pip waste even in a phase that ALSO saw a burst
tap (a whole-phase-exclusion predecessor of this design could not). It wraps
a base reward_fn with a per-transition penalty for mana burnt (unconditional
WITHIN the single-pip-tagged bucket -- no exemption for whether the burn was
avoidable, unlike the narrower with_mana_mistake_penalty it had itself
replaced). The penalty telescopes a Hill curve (_hill) across a turn's
transitions, so its sum by turn's end equals a single terminal charge for
that turn's total single-pip-tagged burn, but attributed to the transitions
that actually caused it."""


def _hill(x, c, p):
    """Hill-function saturating curve: 0 at x=0, convex (slow-then-
    accelerating) while x << c, inflects around x == c (h(c) == 0.5), then
    concaves back toward -- but never reaches -- 1 as x grows without bound.
    x is assumed >= 0 (a raw count/amount, never negative)."""
    if x <= 0:
        return 0.0
    xp = x ** p
    return xp / (xp + c ** p)


def action_count_win_reward(plateau_actions=80, max_actions=200, min_reward=0.25):
    """Win/loss reward whose "prefer efficiency" axis is the WINNING seat's
    own action count (PlayerState.actions_taken -- real, non-Pass actions
    only, per-seat, never combined across both seats, and never counting an
    automatic draw-for-turn as an "action" -- see actions_taken's own
    docstring) rather than a global turn count: turn-based decay lets a policy
    take arbitrarily many actions within a single turn "for free" (nothing
    below turn granularity was ever measured), so it can't actually discourage
    padding out a turn with pointless actions the way a per-action metric can.

    Piecewise linear, not a pure decay -- a plain decay**actions_taken starts
    penalizing from action 1, and even clamped to a floor came out too close
    to a loss's flat 0 to read as "still clearly a win," while rewarding
    nothing for finishing within a perfectly reasonable number of actions.
    This version has three flat request-driven reference points instead: an
    actions_taken <= plateau_actions (80, "sufficiently fast" -- no reason to
    reward going even faster) -> max_reward (1.0); actions_taken >= max_actions
    (200) -> min_reward (0.25, clearly above a loss's 0); linear ramp between
    the two. A loss or draw is still exactly 0.0 either way -- only a win's own
    value moves."""
    span = max_actions - plateau_actions
    def reward_fn(state, done, horizon):
        if not done or state.turn_won is None:
            return 0.0
        # Self-contained loser gate: a non-winning seat always scores 0, decided
        # here rather than relying on the caller to zero it externally (deploy_reward
        # needs the loser to reach its own loss band instead, so that gate can't live
        # in the caller). state.active_idx is the seat being scored (rl.train._reward_for
        # flips it).
        if state.winner != state.active_idx:
            return 0.0
        winner_actions = state.players[state.winner].actions_taken
        over_plateau = min(max(0, winner_actions - plateau_actions), span)
        return 1.0 - over_plateau / span * (1.0 - min_reward)
    return reward_fn


def deploy_reward(plateau_actions=80, max_actions=200, win_floor=0.5,
                   discard_c=4.0, discard_p=2.0):
    """Two-band terminal reward. Scored PER SEAT: rl.train._reward_for flips
    state.active_idx to the seat being scored (via drl_env._for_player) and no
    longer zeroes the loser first, so this callable decides win vs loss itself.

    Both bands subtract the SAME sloppiness penalty, q = _hill(...) in
    [0, 1) -- PlayerState.cleanup_discard_turns (cards hoarded past hand
    size) -- so a win with a lot of hoarded cards scores barely above a clean
    loss, not barely below a clean win. Mana burn no longer feeds this
    terminal penalty at all -- see with_mana_mistake_penalty below for its
    (dense, per-transition) replacement.

    WIN (state.winner == this seat): win_floor (0.5) -> 1.0 - q, scaled by the
    winner's own GAMEPLAY efficiency -- actions_taken MINUS pregame_actions (the
    mulligan/keep/bottom picks are excluded so a winner that had to mulligan
    isn't docked for it): <= plateau_actions (80) gameplay actions -> 1.0 (a fast
    win), >= max_actions (200) -> win_floor (a long, grindy win); linear between;
    q subtracted after.

    LOSS or no-winner timeout (state.winner != this seat, including None -- the
    only non-win 2-player outcome, a horizon cap; genuine draws can't happen,
    win_check awards every life/deck-out end to a seat): exactly -q (0.0 for a
    clean loss, same q as above otherwise).

    q asymptotes toward but never reaches 1, so -- when win_floor is 1.0 (the
    only value league self-play actually uses, deploy_reward_v2 below) -- every
    win (which lands in (0, 1]) still strictly outscores every loss (which lands
    in (-1, 0]) regardless of how sloppy either one was. That guarantee does NOT
    hold for win_floor < 1.0 (a sloppy-enough win can dip below win_floor - 1,
    underneath a clean loss's 0) -- deploy_reward_v1 below is reference-only,
    unused by training, specifically because of its OWN efficiency-scaling
    pathology (see deploy_reward_v2's comment), so this is not considered a bug.

    The MULLIGAN decision is not scored here: it's owned by the separate per-deck
    mulligan model (rl.mulligan), which the main policy doesn't drive, so
    penalizing the main policy for mulligans it can't control would be pure noise.
    Scoring it through this terminal reward would also need ~100 steps of
    discounting to reach the mulligan decision -- too diluted a signal -- which is
    why mulligan gets its own dedicated model instead. Terminal only (0.0 until
    done)."""
    span = max_actions - plateau_actions
    win_span = 1.0 - win_floor
    def reward_fn(state, done, horizon):
        if not done:
            return 0.0
        seat = state.active_idx  # the seat being scored (rl.train._reward_for flipped it here)
        p = state.players[seat]
        q = _hill(p.cleanup_discard_turns, discard_c, discard_p)
        if state.winner == seat:
            gameplay_actions = p.actions_taken - p.pregame_actions
            over = min(max(0, gameplay_actions - plateau_actions), span)
            return win_floor + win_span * (1.0 - over / span) - q
        return -q
    return reward_fn


def with_mana_mistake_penalty(base_reward_fn, penalty_per_pip=0.01, per_event_cap=0.05):
    """NOT wired into deploy_reward_v2 as of 2026-08 -- see
    with_dense_mana_burn_penalty below, which replaced it (owner call: too
    weak in practice, and the exemption logic here made it forgive floated
    mana whenever nothing was castable, which is exactly the case the owner
    wants punished). Kept for reference/comparison, same as deploy_reward_v1.

    Wraps a base reward_fn with a DENSE, per-transition penalty for mana
    burnt that game.turn._empty_mana_pools could find no justification for
    (PlayerState.mana_mistake_burn -- see that function's own docstring for
    the three-way exemption: paid for something, triggered something, or
    nothing was legally castable anyway all avoid this counter). Drains
    (reads then zeroes) the counter on every call, done or not, so the same
    burnt pip is never counted twice and none are ever skipped regardless of
    how many -- or how few -- transitions separate one burn from the next.
    This is the one place in this module that mutates state as a side effect
    of scoring it, deliberately: it's a shaping-reward mailbox, not a pure
    read. Applied on the terminal call too -- a mistake on the game's final
    phase still has to be paid for, exactly like any other, or the policy
    would learn burning is free in the last few actions of a game.

    penalty_per_pip is linear, not the terminal band's Hill curve -- at the
    scale a single drain actually produces (usually 0, occasionally a small
    handful of pips from one phase's mana sources), a Hill curve calibrated
    for whole-game totals is indistinguishable from linear anyway, so the
    extra machinery isn't buying anything here.

    per_event_cap bounds what a single drain can cost, independent of
    penalty_per_pip: an uncapped additive dense term is not guaranteed
    policy-invariant (unlike formal potential-based shaping), and
    deploy_reward's own "every win outscores every loss" guarantee already
    has zero margin in its worst case (a maximally-sloppy win's badness
    approaches, but never reaches, a clean loss's 0) -- summed dense
    penalties across an episode could in principle erode that further. The
    cap bounds the worst SINGLE event; it does not bound the aggregate
    across many separate events in one game, which would need an
    episode-level budget (tracked on PlayerState, reset for free every game)
    instead -- deliberately not built until real training data shows the
    aggregate case actually matters, rather than guessing at a second
    constant with no grounding.

    Both constants are conservative placeholders, not derived from data --
    there's no training run yet under this rule to measure how often a
    genuine (non-exempted) mistake actually fires. Revisit both once one
    exists (compare mean dense penalty per game against deploy_reward_v2's
    own terminal range)."""
    def reward_fn(state, done, horizon):
        p = state.players[state.active_idx]
        mistake, p.mana_mistake_burn = p.mana_mistake_burn, 0
        penalty = min(penalty_per_pip * mistake, per_event_cap)
        return base_reward_fn(state, done, horizon) - penalty
    # Lets rl.train.collect_rollout skip building/wiring game.turn's
    # on_mana_burn hook (and the per-phase legal_action_mask sweep it costs)
    # for a pairing where no seat's reward_fn would ever drain
    # mana_mistake_burn anyway -- e.g. pretraining's action_count_win_reward.
    reward_fn.consumes_mana_mistake = True
    return reward_fn


def with_dense_mana_burn_penalty(base_reward_fn, mana_burn_c=3.3, mana_burn_p=4.0, game_penalty_cap=2.0):
    """League self-play's current dense mana-burn shaping (2026-08, replacing
    with_mana_mistake_penalty above -- see its own docstring for why).
    Restores the PRE-cbd7379 mana-burn curve's shape and intent (a Hill
    curve, asymptotically saturating, so a badly-sloppy game is barely worse
    than a slightly-sloppy one rather than unboundedly worse) but keeps it
    genuinely DENSE and per-transition rather than terminal, and switches
    its input from that era's mana_burnt_total (unconditional, whole-GAME
    cumulative) to PlayerState.mana_burnt_this_turn_single_pip (per-TURN
    cumulative, reset by game.turn._run_turn_gen) -- deliberately still
    unconditional WITHIN the single-pip-tagged bucket, unlike
    mana_mistake_burn's cost-paid/trigger-fired/nothing-castable exemptions:
    the point here is punishing floating more mana in a turn than you spend,
    full stop, not just provably avoidable waste (owner call, 2026-08 --
    mana_mistake_burn's own exemptions were forgiving exactly the cases
    meant to be punished).

    PER-PIP SINGLE-PIP TAG (2026-08, second iteration -- this wrapper was
    first wired unconditionally against raw mana_burnt_this_turn, reverted
    after a real training batch showed the `elves` deck regressing sharply
    across every cross-league matchup, traced to Priest of Titania's mana
    ability ("count_all" in the card registry -- sums every Elf on BOTH
    battlefields in one all-or-nothing tap): a large, unavoidable per-turn
    burst that's correct, intrinsic archetype play, not a mistake raw totals
    could tell apart from real waste. A first fix, since replaced, excluded
    an entire PHASE's burn whenever any "unmetered" source was tapped that
    phase -- coarse, and blind to a mixed phase where an avoidable
    single-pip tap sat alongside an unavoidable burst.)

    mana_burnt_this_turn_single_pip (game.mana.float_mana / spend_one_pip,
    game.turn._empty_mana_pools) now attributes PER PIP: a mana-producing
    EVENT is tagged "single-pip" iff it adds exactly 1 symbol to the pool in
    that one event (dynamic, len(produced) == 1) -- a plain land or Llanowar
    Elves always qualifies; Rakdos Carnarium/Utopia Sprawl's automatic bonus
    (2+ symbols) never does; a Tron land qualifies only while not all three
    Tron types are online; Priest of Titania/Overgrown Battlement qualify
    only in the edge case their count resolves to exactly 1.
    PlayerState.mana_pool_single_pip shadow-counts how many of the CURRENTLY
    FLOATING pips of each color are tagged. Spending a pip (game.mana.
    spend_one_pip) always consumes an UNTAGGED pip of that color first -- so
    Priest's own burst mana is always spent/burnt ahead of any tagged
    single-pip mana of the same color, meaning
    mana_burnt_this_turn_single_pip never blames Priest's unavoidable excess
    while still correctly catching a genuinely avoidable single-pip tap
    (e.g. 2 unneeded Llanowar Elves alongside a Priest burst that alone
    would have covered the spend). Reflexive tapping of ordinary single-pip
    sources -- the actual problem this wrapper exists for -- is fully
    penalized, now with per-pip precision instead of the earlier
    whole-phase exclusion's coarser under-penalizing of mixed phases.

    TELESCOPING, not a flat per-event charge: each call charges only the
    MARGINAL increase in _hill(mana_burnt_this_turn_single_pip, c, p) since
    the last call for this player this turn (PlayerState.mana_burn_penalty_
    credited is the running baseline, reset alongside
    mana_burnt_this_turn_single_pip each new turn). Burning early in a turn
    is nearly free (the curve's shallow start); each additional pip burnt
    LATER in the same turn costs more than the last, because it's added to
    an already-elevated baseline -- a natural, compounding "you should have
    planned this turn's mana better" shape. Summed across a whole turn's
    worth of calls, the total charged is EXACTLY
    _hill(total_burnt_this_turn_single_pip, c, p) -- the same number a
    one-shot terminal score would have given for that turn, just attributed
    to whichever individual transitions actually caused it (proper credit
    assignment, not blurred across the whole game the way this file's
    original terminal mana-burn penalty was -- see git history on cbd7379).

    c=3.3, p=4.0 (owner-specified anchors, 2026-08): _hill(1, 3.3, 4.0) ~=
    0.008 (first burnt pip in a turn is nearly free) and _hill(6..7, 3.3,
    4.0) ~= 0.92-0.95 (burning most of a turn's mana is close to the max
    single-turn charge). Full curve: 1->0.008, 2->0.12, 3->0.41, 4->0.68,
    5->0.84, 6->0.92, 7->0.95, 10->0.99. Placeholders in the same sense
    deploy_reward's own discard curve is -- revisit once real training data
    exists under this rule.

    game_penalty_cap (2026-08 addition) bounds the WHOLE-GAME running total
    this wrapper ever charges (PlayerState.mana_burn_penalty_charged_total,
    never reset once the game is underway -- see its own docstring for why
    a cap is needed at all: resetting mana_burn_penalty_credited every turn
    means these per-turn charges do NOT telescope to a bounded total the way
    a true potential function would -- Ng, Harada & Russell 1999's
    F(s,a,s')=Φ(s')-Φ(s) invariance theorem needs Φ to be a function that
    telescopes across the WHOLE trajectory with no artificial resets;
    resetting the baseline every turn silently drops the "refund" a real
    potential function would pay back at that exact transition, so nothing
    otherwise stops a long run of bad turns from summing to a penalty that
    dwarfs the terminal win/loss signal). Deliberately NOT "fixed" by making
    this a true potential function instead (one Hill curve over the whole-
    GAME cumulative burn, never reset): Ng et al.'s own guarantee is that
    potential-based shaping does not change the optimal policy at all --
    exactly the opposite of the point of adding this term, which is to
    inject a real, lasting preference against wasting mana beyond whatever
    plain win/loss already implies. A whole-game (non-reset) curve would
    also let a policy "pay down" its mana-burn budget in one early bad turn
    and burn near-free for the rest of the game, the opposite of the
    turn-by-turn discipline this is meant to teach. So: real, accumulating,
    non-potential-based pressure (deliberate), just capped in aggregate
    (deliberate) -- a blunt safety backstop, not principled shaping, chosen
    because a reward term whose magnitude scales with episode length with no
    ceiling is a known source of PPO/GAE instability (noisier advantage
    estimates, a value function fitting a heavier-tailed target), and this
    repo has already hit that exact failure mode once for real: git history
    on cbd7379 walked back an earlier, similarly unbounded mana-burn penalty
    specifically because of training problems it caused (dmir_terror). 2.0
    is a conservative placeholder (comparable order of magnitude to q's own
    [0, 1) range, wide enough that a genuinely sloppy game can still land
    well below a clean loss -- that outcome is fine, only the UNBOUNDED
    magnitude was the problem) -- revisit once real training data exists
    under this rule, same as mana_burn_c/mana_burn_p above.

    No exemption checking means no need for game.turn's on_mana_burn hook at
    all (unlike with_mana_mistake_penalty) -- mana_burnt_this_turn_single_pip
    is always tallied by game.turn._empty_mana_pools regardless of
    reward_fn, so this wrapper needs no consumes_mana_mistake-style opt-in
    flag; it's pure, already-available bookkeeping to read, not something
    rl.train needs to wire up conditionally."""
    def reward_fn(state, done, horizon):
        p = state.players[state.active_idx]
        current = _hill(p.mana_burnt_this_turn_single_pip, mana_burn_c, mana_burn_p)
        penalty = current - p.mana_burn_penalty_credited
        p.mana_burn_penalty_credited = current
        # Whole-game cap: clamp this charge so mana_burn_penalty_charged_total
        # never exceeds game_penalty_cap, regardless of how many separate bad
        # turns preceded it. remaining floored at 0 -- once the cap is fully
        # spent, every further charge is exactly 0, never negative (this is a
        # ceiling on badness, never a bonus for burning MORE).
        remaining = max(0.0, game_penalty_cap - p.mana_burn_penalty_charged_total)
        penalty = min(penalty, remaining)
        p.mana_burn_penalty_charged_total += penalty
        return base_reward_fn(state, done, horizon) - penalty
    return reward_fn


# Pre-baked named instance (callers reference reward_fns by plain name via
# getattr off this module -- see rl.train's own reward_fn_name plumbing).
# Floor lowered to 0.2 (vs. the default 0.25) per the "sliding scale from
# 1 - 0.2" spec -- this is the reward pretraining (run_pretrain.py) uses.
action_count_win_reward_200_floor02 = action_count_win_reward(min_reward=0.2)

# Not wired into run_league.py; kept for reference/comparison against
# deploy_reward_v2. Win band 0.5->1.0 by gameplay efficiency; loss band
# shares whatever deploy_reward's own current default computes (see its
# docstring) -- v1 and v2 only ever differ in win_floor, both sharing the
# one factory. Deliberately left unwrapped by with_mana_mistake_penalty --
# it's unused by training, so there's no reason to change its behavior beyond
# what dropping mana burn from deploy_reward's own badness already implies.
deploy_reward_v1 = deploy_reward()


# League self-play's reward (run_league.py): deploy_reward with win_floor=1.0,
# which collapses the win band's EFFICIENCY term to a flat 1.0 (win_span = 0,
# gameplay-action-count no longer moves the score) -- while still subtracting
# the cleanup-discard sloppiness penalty q from both bands. v1's
# efficiency scaling induces an action-space-minimization pathology (the policy
# generalizes "win in fewer actions -> more reward" into "shrink your own
# board"); flat win removes that gradient. Float-first means no undo action
# exists, so there's no tap/untap-abandon churn to structurally guard against;
# PPO entropy alone bounds pointless actions without capping them. Because q
# never reaches 1, every win here still strictly outscores every loss no
# matter how sloppy either was -- see deploy_reward's own docstring for the
# exact guarantee and its win_floor=1.0 precondition.
#
# WRAPPED by with_dense_mana_burn_penalty (2026-08 addition; reverted the
# same month; RE-ENABLED 2026-08 under a fix -- see below). First version: a
# 2026-08-06 cross-league benchmark (before/after a +10,000-game/deck batch
# trained under an UNCONDITIONAL dense wrap, reading raw mana_burnt_this_turn)
# showed elves regressing consistently across all 4 gauntlet matchups --
# traced to Priest of Titania's mana ability ("count_all", src/game/mana.py,
# sums Elves on BOTH battlefields with no partial-tap option) making large,
# unavoidable per-turn bursts an intrinsic, correct part of the archetype,
# not a mistake the dense curve could distinguish from real waste. Reverted
# to plain deploy_reward(win_floor=1.0) rather than patched with an
# exemption (e.g. "was anything castable") because Priest's burst is large
# enough that even a coarse castability check would rarely exempt it.
#
# Re-enabled, second iteration: a first fix wrapped this in a whole-phase
# metered/unmetered exclusion (excuse an entire phase's burn if any burst
# source was tapped that phase) -- since replaced by with_dense_mana_burn_
# penalty's own PER-PIP single-pip tag (see its docstring): the penalty now
# reads PlayerState.mana_burnt_this_turn_single_pip, which game.turn.
# _empty_mana_pools sums straight from PlayerState.mana_pool_single_pip --
# only pips from a mana-producing EVENT that added exactly 1 symbol, spent
# last (game.mana.spend_one_pip always drains untagged/burst mana of a
# color first). Priest of Titania's burst never reaches the penalty at all
# now, regardless of magnitude, AND (unlike the whole-phase predecessor) a
# reflexive single-pip tap sitting alongside a burst tap in the same phase
# is still correctly penalized -- reflexive tapping of ordinary metered
# sources (every land, every simple mana dork -- the actual problem this
# wrapper exists for) is fully penalized with per-pip precision.
#
# This does NOT resolve the wrap's separate, already-accepted tradeoff that
# its own game_penalty_cap can still push a sufficiently sloppy win below a
# clean loss in principle (with_dense_mana_burn_penalty's own docstring: "a
# genuinely sloppy game can still land well below a clean loss -- that
# outcome is fine, only the UNBOUNDED magnitude was the problem") -- that was
# a deliberate design choice when the wrapper was authored, orthogonal to
# the archetype-bias bug this split fixes.
deploy_reward_v2 = with_dense_mana_burn_penalty(deploy_reward(win_floor=1.0))
