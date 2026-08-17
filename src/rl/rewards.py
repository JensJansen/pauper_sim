"""Reward functions for training a DRL policy against the game engine.

Contract: any callable

    reward_fn(state: game.GameState, done: bool, horizon: int) -> float

Called once per environment step with the state *after* that step's
action was applied. A sparse reward function returns 0.0 unless `done`;
a dense one could return something every call. No base class -- any
matching callable works.

Terminal WIN/LOSS scoring differs by reward generation. v1-v4 scale a win by
how efficiently it was reached, minus a "sloppiness" penalty q for hoarded
cards forced to cleanup discard (see _hill), with a loss/timeout at 0.0 minus
that same q. Pretraining uses action_count_win_reward_200_floor02 (the plain,
pretrain-only predecessor, no such penalty). League self-play uses
deploy_reward_v5 (2026-08-11), which drops q entirely on BOTH bands for a flat
+1.0 win / -1.0 loss (flat_win_loss_reward) and applies the dense mana-burn
wrap below to the WINNER ONLY -- see its own comment for the measured reason
(charging burn on a loss made losing passively score ~2x better than losing
while trying, since a seat that never taps cannot burn).

DENSE shaping (with_dense_mana_burn_penalty, below) is wired into
deploy_reward_v3, v4, and -- winner-only, via refund_on_loss=True -- v5. See
deploy_reward_v2's own comment for its history: an
unconditional first version was reverted 2026-08 (archetype bias against
Elves' Priest of Titania), then re-enabled 2026-08 reading a PER-PIP
single-pip-tagged subset of mana burnt (PlayerState.
mana_burnt_this_turn_single_pip) instead of the raw total -- game.mana.
float_mana/spend_one_pip tag and spend individual floating pips (see their
own docstrings, and PlayerState.mana_pool_single_pip's), which structurally
excludes Priest-style board-state-scaled bursts from the penalty while still
catching avoidable single-pip waste even in a phase that ALSO saw a burst
tap (a whole-phase-exclusion predecessor of this design could not). The
penalty is unconditional WITHIN the single-pip-tagged bucket -- no exemption
for whether the burn was avoidable, unlike the narrower with_mana_mistake_
penalty it had itself replaced. The penalty telescopes a Hill curve (_hill)
across a turn's worth of burns, so its sum by turn's end equals a single
terminal charge for that turn's total single-pip-tagged burn -- ATTRIBUTED
to the specific Tap actions that caused it via rl.train's on_single_pip_burn
hook (2026-08-10 second iteration; see with_dense_mana_burn_penalty's own
docstring for why the first version's naive "charge inside reward_fn"
approach didn't actually achieve that attribution despite claiming to,
confirmed by direct measurement against the real production reward -- the
numbers are quoted in that docstring; the one-off probe that produced them
was retired 2026-08-17). reward_fn itself is now a
plain passthrough of the base reward_fn -- the dense charge is applied as a
direct correction to already-recorded buffer entries, not through reward_fn's
own return value. deploy_reward_v3's and deploy_reward_v4's own comments
(below) cover their respective 2026-08-10 curve/cap reshapes on top of this
same mechanism; deploy_reward_v5's covers the 2026-08-11 winner-only split
and the wider (c=2.9, cap=1.5) curve it uses."""


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
                   discard_c=4.0, discard_p=2.0, discard_weight=1.0):
    """Two-band terminal reward. Scored PER SEAT: rl.train._reward_for flips
    state.active_idx to the seat being scored (via drl_env._for_player) and no
    longer zeroes the loser first, so this callable decides win vs loss itself.

    Both bands subtract the SAME sloppiness penalty, q = discard_weight *
    _hill(...) in [0, discard_weight) -- PlayerState.cleanup_discard_turns
    (cards hoarded past hand size) -- so a win with a lot of hoarded cards
    scores barely above a clean loss, not barely below a clean win. Mana burn
    no longer feeds this terminal penalty at all -- see with_mana_mistake_
    penalty below for its (dense, per-transition) replacement.
    discard_weight (default 1.0, byte-identical to every existing caller --
    v1/v2 below don't pass it) scales q's own asymptote DOWN from 1 without
    reshaping its curve, so a caller layering ANOTHER bounded penalty on top
    (deploy_reward_v3's own mana-burn wrap, rl.rewards below) can size each
    penalty's own share of a shared worst-case badness budget instead of both
    independently asymptoting toward 1 and compounding past whatever spread
    that caller actually wants.

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

    q asymptotes toward but never reaches discard_weight, so -- when win_floor
    is 1.0 and discard_weight is 1.0 (deploy_reward_v2's own values below) --
    every win (which lands in (0, 1]) still strictly outscores every loss
    (which lands in (-1, 0]) regardless of how sloppy either one was. That
    guarantee does NOT hold for win_floor < 1.0 (a sloppy-enough win can dip
    below win_floor - discard_weight, underneath a clean loss's 0) --
    deploy_reward_v1 below is reference-only, unused by training,
    specifically because of its OWN efficiency-scaling pathology (see
    deploy_reward_v2's comment), so this is not considered a bug.

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
        q = discard_weight * _hill(p.cleanup_discard_turns, discard_c, discard_p)
        if state.winner == seat:
            gameplay_actions = p.actions_taken - p.pregame_actions
            over = min(max(0, gameplay_actions - plateau_actions), span)
            return win_floor + win_span * (1.0 - over / span) - q
        return -q
    return reward_fn


def flat_win_loss_reward():
    """The simplest possible terminal reward: +1.0 for a win, -1.0 for a loss
    OR a no-winner horizon timeout, 0.0 until done. Scored PER SEAT exactly
    like deploy_reward above (rl.train._reward_for flips state.active_idx to
    the seat being scored, so `state.winner == state.active_idx` IS "did this
    seat win"). deploy_reward_v5's base -- see its own comment for why the
    league moved to this shape.

    Deliberately a NEW function rather than deploy_reward(discard_weight=0.0):
    that call would zero q correctly but still return deploy_reward's own LOSS
    band, which is `-q` -> 0.0, not -1.0. The two-band ±1 shape here is not
    reachable by parameterizing deploy_reward at all, so it gets its own
    factory instead of another knob on that one.

    No sloppiness/efficiency terms of any kind: every win scores exactly the
    same before shaping, and every loss scores exactly the same, full stop.
    That flatness is the entire point (deploy_reward_v5's comment has the
    measured reason) -- any WITHIN-band gradation is a lever a policy can
    optimize independently of actually winning, and this reward exists
    specifically because one such lever (the loss band's own mana-burn
    exposure) was found to reward losing passively over losing while trying.
    Shaping still layers on top via with_dense_mana_burn_penalty, but only on
    the WIN band (refund_on_loss=True, see there)."""
    def reward_fn(state, done, horizon):
        if not done:
            return 0.0
        return 1.0 if state.winner == state.active_idx else -1.0
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


def _charge_single_pip_burn(player, mana_burn_c, mana_burn_p, game_penalty_cap, mana_burn_weight=1.0):
    """The marginal Hill-curve-with-cap charge for player.mana_burnt_this_
    turn_single_pip, AS IT STANDS RIGHT NOW -- called from rl.train's
    on_single_pip_burn hook, which fires from inside game.turn.
    _empty_mana_pools AFTER that counter has already been incremented for
    the clear this call is for, so "current" below already reflects this
    specific burn. Exactly the math with_dense_mana_burn_penalty's own
    reward_fn used to run inline (see its docstring for why that was moved
    here) -- mutates player.mana_burn_penalty_credited/
    mana_burn_penalty_charged_total as a side effect, unchanged from
    before.

    mana_burn_weight (2026-08-11, default 1.0 -- byte-identical for v2/v3/v4,
    which never pass it) SCALES the Hill curve itself, so one turn's own
    worst case approaches mana_burn_weight rather than _hill's own fixed 1.0
    asymptote. Distinct from game_penalty_cap, which bounds the WHOLE-GAME
    running total and does not reshape any single charge: with weight=1.0 and
    cap=0.6 (v4), a single 4-pip turn already computes 0.683 and gets clipped
    by the game cap, so the cap -- not the curve -- was setting the effective
    per-turn magnitude. Separating the two lets v5 place the curve's own
    asymptote (1.5) where the owner's spec wants a 5-pip turn to land (~1.35,
    90% of it) while still using the cap for its intended job: bounding the
    sum across many bad turns."""
    current = mana_burn_weight * _hill(player.mana_burnt_this_turn_single_pip, mana_burn_c, mana_burn_p)
    charge = current - player.mana_burn_penalty_credited
    player.mana_burn_penalty_credited = current
    # Whole-game cap: clamp this charge so mana_burn_penalty_charged_total
    # never exceeds game_penalty_cap, regardless of how many separate bad
    # turns preceded it. remaining floored at 0 -- once the cap is fully
    # spent, every further charge is exactly 0, never negative (this is a
    # ceiling on badness, never a bonus for burning MORE).
    remaining = max(0.0, game_penalty_cap - player.mana_burn_penalty_charged_total)
    charge = min(charge, remaining)
    player.mana_burn_penalty_charged_total += charge
    return charge


def with_dense_mana_burn_penalty(base_reward_fn, mana_burn_c=3.3, mana_burn_p=4.0, game_penalty_cap=2.0,
                                  refund_on_loss=False, mana_burn_weight=1.0):
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

    TELESCOPING, not a flat per-event charge: each BURN (game.turn.
    _empty_mana_pools detecting a non-empty pool at a phase boundary, via
    the on_single_pip_burn hook -- see ATTRIBUTION below) charges only the
    MARGINAL increase in _hill(mana_burnt_this_turn_single_pip, c, p) since
    the last burn for this player this turn (PlayerState.mana_burn_penalty_
    credited is the running baseline, reset alongside
    mana_burnt_this_turn_single_pip each new turn). Burning early in a turn
    is nearly free (the curve's shallow start); each additional pip burnt
    LATER in the same turn costs more than the last, because it's added to
    an already-elevated baseline -- a natural, compounding "you should have
    planned this turn's mana better" shape. Summed across a whole turn's
    worth of burns, the total charged is EXACTLY
    _hill(total_burnt_this_turn_single_pip, c, p) -- the same number a
    one-shot terminal score would have given for that turn, just distributed
    across whichever individual Tap transitions actually caused it (proper
    credit assignment, not blurred across the whole game the way this
    file's original terminal mana-burn penalty was -- see git history on
    cbd7379).

    ATTRIBUTION (2026-08-10, second iteration -- replacing an earlier
    version of this same charge that ran inline inside reward_fn, called at
    a seat's own NEXT decision after the burn): reward_fn below is now a
    plain passthrough of base_reward_fn. The actual charge is computed by
    _charge_single_pip_burn above and applied by rl.train.collect_rollout's
    own on_single_pip_burn hook (exposed here as reward_fn.
    charge_single_pip_burn, the same opt-in-attribute pattern with_mana_
    mistake_penalty's own consumes_mana_mistake flag already uses) --
    called SYNCHRONOUSLY from inside game.turn._empty_mana_pools, at the
    exact moment of the burn, so it can charge the actual recent single-pip
    Tap actions directly instead of whatever's pending at a seat's next
    unrelated decision.

    Why this needed fixing: a seat keeps priority after any non-Pass action
    (only Pass hands it to the other player), so a whole run of taps
    happens back-to-back with mana_burnt_this_turn_single_pip completely
    unchanged between them -- it only increments at the phase-boundary
    clear, an ENGINE-internal event with no decision point of its own. The
    FIRST version's every-tap reward call therefore correctly computed a
    marginal delta of exactly 0 for each individual tap, and the entire
    phase's charge landed on whichever action was still pending once the
    clear had already happened by the time the seat was asked to decide
    again -- almost always that seat's own end-of-phase Pass, never the
    taps that actually produced the float. Measured directly against
    dmir_terror's real production reward function (one-off probe, 2026-08-10
    session; script retired 2026-08-17): only 1.8% of recorded
    transitions (36/1968 across 25 games) ever carried a nonzero charge,
    and a Tap action received it no more often than Pass or anything else
    (3.2% vs. 2.2% vs. 3.4%) -- the charge was landing almost independently
    of which action actually caused it. PPO's only path to the right lesson
    was indirect: the value function bootstrapping a rare, mislocated
    charge backward through GAE onto the PRECEDING taps, a far weaker and
    noisier channel than a correctly-attributed direct charge -- a
    plausible mechanism for why steepening this curve (deploy_reward_v2 ->
    v3) measurably halved elves' burn rate but left dmir_terror's
    completely unchanged (rl.run_cross_league_eval / analyze_mana_burn_
    by_turn.py, same session): a harsher curve doesn't help when the
    signal isn't reaching the action it's meant to discourage.

    rl.train.collect_rollout's on_single_pip_burn hook distributes each
    burn's charge across every single-pip Tap action that seat recorded
    since its own last phase clear, weighted by pips produced (always
    exactly 1 per tap, per the tag rule above). This is an approximation,
    not exact per-pip attribution: PlayerState.mana_pool_single_pip is a
    per-color shadow COUNT, not a per-pip list (see its own docstring), so
    it can't say exactly WHICH tap's specific pip survived to be burnt
    versus got spent first when multiple same-color taps happened in one
    phase. A large improvement over the prior 100%-lands-on-an-unrelated-
    Pass behavior regardless.

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

    Unlike with_mana_mistake_penalty, still no EXEMPTION checking (no need
    for game.turn's on_mana_burn hook -- mana_burnt_this_turn_single_pip is
    always tallied by _empty_mana_pools regardless of reward_fn). It DOES
    now need the separate on_single_pip_burn hook for ATTRIBUTION (see
    above) -- gated the same way with_mana_mistake_penalty's own
    consumes_mana_mistake flag gates on_mana_burn: rl.train.collect_rollout
    wires on_single_pip_burn in only when at least one tracked seat's
    reward_fn exposes a charge_single_pip_burn attribute, so a pairing that
    doesn't use this wrapper (e.g. pretraining's action_count_win_reward_*)
    never pays for the extra bookkeeping.

    refund_on_loss (2026-08-11, opt-in, default False -- so v2/v3/v4 above
    keep byte-identical semantics): when True, this penalty applies ONLY to a
    seat that WON its game; a seat that lost (or timed out with no winner)
    pays exactly nothing for whatever it burnt. Tags the returned closure with
    a `mana_burn_winner_only = True` attribute, read by rl.train.collect_
    rollout (_wants_winner_only_burn) -- the same opt-in-attribute pattern
    charge_single_pip_burn/consumes_mana_mistake already use.

    Why (deploy_reward_v5's own comment has the full measurement): charging
    this penalty on a LOSS made losing PASSIVELY score strictly better than
    losing while trying. A seat that never taps mana cannot burn mana, so it
    scored exactly 0.0 on this term by construction, every time; a seat that
    developed its board and made any ordinary sequencing mistake paid up to
    the full cap. Measured on a real 10,003-games/deck run: dmir_terror's
    passive losses averaged 0.321 total penalty vs. 0.598 for its active
    losses (n=14 vs 64) -- a structural discount for not playing.

    Why "winner-only" and not "smaller on a loss": any nonzero loss-band
    charge reintroduces the same ordering (passive still pays less than
    active), just with a smaller gap. The asymmetry is removed by
    construction only when the loss band carries no burn term at all.

    IMPLEMENTATION IS NOT A TERMINAL REFUND -- rl.train.collect_rollout DEFERS
    the per-Tap writes instead, holding each (buffer_index, share) until the
    game ends and applying them only on a win (see its own deferred_charges
    comment). A lump-sum refund at the terminal transition would NOT be
    equivalent: PPO trains on GAE advantages, where a charge written at step t
    lands in delta_t directly and immediately, while a terminal refund reaches
    step t only through GAE's backward recursion, discounted by
    (gamma*gae_lambda)^k over the k steps between them (0.9405^k at rl.ppo's
    own gamma=0.99/gae_lambda=0.95 -- ~11% of the refund survives 40 steps
    back). That would have left EARLY burns in a long losing game mostly
    un-refunded while late ones cancelled cleanly -- a distance-dependent
    residue, not the intended "costs nothing on a loss." Deferring the write
    leaves a losing trajectory bit-for-bit identical to one that never burnt
    at all, which is the actual guarantee wanted here."""
    def reward_fn(state, done, horizon):
        return base_reward_fn(state, done, horizon)
    reward_fn.charge_single_pip_burn = lambda player: _charge_single_pip_burn(
        player, mana_burn_c, mana_burn_p, game_penalty_cap, mana_burn_weight)
    if refund_on_loss:
        reward_fn.mana_burn_winner_only = True
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
# board"); flat win removes that gradient. There is no undo action
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
# the archetype-bias bug this split fixes. SUPERSEDED as of deploy_reward_v3
# below (2026-08-10) for the actual trained path -- v3's own discard_weight/
# game_penalty_cap split resolves this tradeoff by construction instead of
# accepting it. v2 itself is kept unchanged, for reference/comparison, same
# treatment v1 already gets above.
deploy_reward_v2 = with_dense_mana_burn_penalty(deploy_reward(win_floor=1.0))


# League self-play's CURRENT reward (run_league.py, 2026-08-10), replacing
# deploy_reward_v2 above: same shape (flat win_floor=1.0, no efficiency
# scaling), but reshaped so mana-burn hits considerably harder and the whole
# thing is normalized to a specific, provable best-win/worst-loss spread
# (owner spec, 2026-08-10: "best wins ~2 > worst losses... up the cost of
# burning the first mana pip considerably... more aggressive punishment").
#
# discard_weight=0.4 caps q's own asymptote at 0.4 (down from 1.0) rather
# than reshaping its curve (discard_c/discard_p unchanged) -- discard-
# sloppiness keeps the exact same shape, just a smaller maximum share of the
# combined badness budget below. mana_burn_c=2.0/mana_burn_p=2.5 (down from
# 3.3/4.0) makes the per-pip curve considerably front-loaded (verified
# numerically): pip1=0.150, pip2=0.500, pip3=0.734, pip4=0.850, pip6=0.940 --
# vs. v2's pip1=0.008, pip2=0.119, pip3=0.406, pip4=0.683, pip6=0.916. The
# first wasted pip in a turn now costs ~18x what it cost under v2 (0.150 vs.
# 0.008) instead of being "nearly free," and the curve reaches near-max
# badness within 4-5 pips instead of 6-7. game_penalty_cap=0.6 (down from
# 2.0) is mana-burn's own share of the combined budget.
#
# Combined: clean win (q=0, mana_penalty=0) = 1.0, unchanged. Worst-case
# loss (q -> its 0.4 asymptote, mana_penalty -> its 0.6 cap) -> -1.0,
# asymptotically -- so best win minus worst loss -> 2.0, exactly matching
# the requested spread as a provable bound rather than an empirical
# approximation. Mana-burn's own share of that budget (0.6) now outweighs
# discard's (0.4), matching mana burn being called out as the bigger
# concern of the two.
#
# Side effect worth noting explicitly: this also restores "every win
# strictly outscores every loss" as an actual guarantee, which v2's own
# comment above explicitly ABANDONED as an accepted tradeoff. Worst win =
# 1.0 - q - mana_penalty > 1.0 - 0.4 - 0.6 = 0.0 STRICTLY (q < 0.4 always,
# even though mana_penalty can reach its 0.6 cap exactly); best/clean loss =
# 0.0 - 0.0 = 0.0 exactly. Worst win is therefore always strictly greater
# than best loss -- a consequence of discard_weight + game_penalty_cap
# summing to exactly win_floor (0.4 + 0.6 = 1.0), not something separately
# enforced.
#
# Accepted, NOT fixed, tradeoff: game_penalty_cap=0.6 combined with the
# steeper curve means a single very bad turn (~3 pips wasted, raw 0.734) can
# nearly exhaust the WHOLE GAME's mana-burn budget on its own, making any
# further waste that same game free. Kept as-is, same "blunt safety
# backstop, not principled shaping" philosophy game_penalty_cap's own
# docstring already states -- a PER-TURN (rather than whole-game) cap would
# reintroduce the exact unbounded-episode-length PPO/GAE instability the
# whole-game cap exists to prevent (see with_dense_mana_burn_penalty's own
# docstring and the cbd7379 history it cites).
#
# SUPERSEDED as of deploy_reward_v4 below (2026-08-10, same day): a real
# training run under v3 (sessions 13-25, ~10,001 -> 20,004 games/deck, the
# same run that validated the on_single_pip_burn attribution fix) drove
# elves and rakdos_madness into a degenerate zero-mana-development collapse
# starting session 20 -- vs_gauntlet win rate for elves flatlined at exactly
# 0/20 for 5 consecutive sessions, confirmed (not sampling noise) via a
# hand-inspected game where elves held a playable Forest and passed 367
# times across 46 turns rather than ever play it, forfeiting rather than
# risk the burn penalty. v3's steepened curve (this comment's own numbers
# above) was calibrated to compensate for the credit-assignment bug that
# on_single_pip_burn fixed THE SAME SESSION -- once the charge started
# landing precisely on the causal Tap action instead of almost-uniformly on
# whatever was pending, that compensation became a double-count, strong
# enough on at least two archetypes to make guaranteed inaction a better-
# scoring local optimum than a real attempt to win despite deploy_reward's
# own worst-win-beats-best-loss guarantee holding throughout (this is an
# optimization/exploration failure, not a reward-ordering bug). v4 below was
# the response to this; CORRECTED 2026-08-11 -- v4 turned out to be
# necessary but NOT sufficient, and the "v3's steepening caused it" reading
# here is at best partial. Retraining from scratch under v4 (a fresh league,
# never touched by this collapse or its checkpoint revert) reproduced the
# same passivity in dmir_terror at 11.2% of games (14/125), while elves --
# the deck that actually collapsed here -- came back at 0.8%. The deeper
# cause was in the reward's own band structure, not this curve's steepness:
# see deploy_reward_v5's comment for the measured asymmetry and the fix.
# v3 itself kept unchanged, same reference/comparison treatment v1 and v2
# already get above.
deploy_reward_v3 = with_dense_mana_burn_penalty(
    deploy_reward(win_floor=1.0, discard_weight=0.4),
    mana_burn_c=2.0, mana_burn_p=2.5, game_penalty_cap=0.6,
)


# SUPERSEDED by deploy_reward_v5 below (2026-08-11) -- kept unchanged for
# reference/comparison, same treatment v1/v2/v3 get. v4 was the response to
# v3's collapse (see v3's own SUPERSEDED note, and its 2026-08-11
# correction): softening the curve back to v2's shape did help -- elves,
# retrained from scratch under v4, showed the passivity pattern in only 0.8%
# of games (1/125) vs. the flatlined collapse it suffered under v3 -- but it
# did NOT fix dmir_terror, which still showed it in 11.2% (14/125) of a
# fresh-reset run that had never seen the original collapse at all. That
# residue is what led to v5's structural change; the remaining diagnosis is
# in v5's own comment.
#
# League self-play's reward from 2026-08-10 (second revision) until v5
# replaced it 2026-08-11, itself replacing deploy_reward_v3 above. Same
# win/loss shape and the
# same discard_weight=0.4/game_penalty_cap=0.6 split as v3 -- that split is
# what makes "every win strictly outscores every loss" a provable guarantee
# (v3's own comment: worst win 1.0-0.4-0.6=0.0 > best loss 0.0), and it's
# unrelated to the curve's own steepness, so it stays. ONLY mana_burn_c/
# mana_burn_p revert to v2's original 3.3/4.0 (from v3's 2.0/2.5) -- see
# deploy_reward_v3's own SUPERSEDED note directly above for why v3's
# steepening (an 18x front-load, tuned to compensate for a credit-
# assignment bug that made the charge land almost anywhere BUT the causal
# Tap action) became a double-count once that bug was fixed, and drove two
# of four archetypes into a losing-on-purpose local optimum. Reverting the
# curve to v2's shape (pip1=0.008, nearly free, vs. v3's 0.150) keeps the
# CORRECT, now-precisely-targeted attribution while removing the
# compensation that was tuned for the broken one -- the bet being that
# accurate delivery alone is enough of a deterrent without also needing
# v3's steepness. That bet was measured and came back split (see the
# SUPERSEDED note above this block): right for elves, insufficient for
# dmir_terror.
deploy_reward_v4 = with_dense_mana_burn_penalty(
    deploy_reward(win_floor=1.0, discard_weight=0.4),
    mana_burn_c=3.3, mana_burn_p=4.0, game_penalty_cap=0.6,
)


# SUPERSEDED by deploy_reward_v6 below (2026-08-12) -- kept unchanged for
# reference/comparison, same treatment v1-v4 get. v5's two STRUCTURAL changes
# (flat +1/-1 band, winner-only burn) were both validated over 20,065
# games/deck and are carried into v6 verbatim; only its curve WEIGHT was
# wrong. See v6's own comment for the measurement.
#
# League self-play's reward from 2026-08-11 until v6 replaced it 2026-08-12,
# itself replacing deploy_reward_v4 above. Two changes, both structural rather
# than a re-tuning: the terminal band is a flat +1/-1 (flat_win_loss_reward --
# no cleanup-discard penalty q at all, on either band), and the dense
# mana-burn penalty applies ONLY to a seat that WON (refund_on_loss=True).
#
# WHY THE LOSS BAND LOST ITS MANA-BURN TERM. v4 charged mana burn on both
# bands. A seat that never taps mana cannot burn mana, so a fully PASSIVE
# loss scored exactly 0.0 on that term by construction, every single time,
# while a seat that actually developed its board and made any ordinary
# sequencing mistake paid up to the whole cap. Measured across a real
# 10,003-games/deck run (fresh reset, v4 throughout), dmir_terror's own
# losses: passive ("never played a land, never tapped, never cast anything")
# averaged 0.321 total penalty across 14 such games; active losses averaged
# 0.598 across 64. Nearly 2x worse for trying. The reward function was
# telling a policy that expects to lose that the cheapest way to lose is to
# do nothing at all -- and the observed behavior matched exactly: 26 of 26
# flagged games cast ZERO spells, including free ones (Snuff Out costs only
# life), which no mana-shaped incentive can explain on its own but a general
# "losing quietly is cheaper" gradient can.
#
# WHY q WENT AWAY ENTIRELY (both bands, not just the loss band). q existed to
# discourage hoarding -- "do something with your cards" -- but the win/loss
# outcome already carries that: hoarded cards stay VISIBLE in game state
# (an overflowing hand, uncast threats, a board that never developed), so a
# terminal win/loss signal is positioned to attribute their cost on its own
# given enough training. Burnt mana is the opposite and that asymmetry is
# exactly why the dense term survives while q doesn't: once a phase boundary
# clears the pool, the waste leaves NO trace in game state for any terminal
# signal to see or blame, which is the entire reason it needs dense,
# per-Tap-attributed shaping in the first place. Dropping q also removes the
# tight discard_weight + game_penalty_cap == win_floor equation v3/v4
# depended on (see v3's comment) -- one less piece of fragile arithmetic to
# re-derive every time a constant moves.
#
# GUARANTEE, re-derived for this shape (it no longer needs the equation
# above): worst-case win = 1.0 - game_penalty_cap = -0.5; every loss = -1.0
# exactly, with no range at all. Worst win beats best loss by 0.5, a real
# margin rather than v3/v4's "approaches 0.0 from above vs. exactly 0.0."
# Note a sufficiently sloppy win CAN now score negative (-0.5 at the cap),
# which v4 could not (its floor was +0.4) -- deliberate, and harmless: the
# only ordering that matters is win-vs-loss, and -0.5 > -1.0 always.
#
# CURVE recalibrated for the wider range (owner spec, 2026-08-11: "cap at 5
# mana burnt and a score of -1.5"). Three constants move together:
# mana_burn_weight 1.0 -> 1.5 (NEW param, see _charge_single_pip_burn -- it
# scales the Hill curve itself, so a single turn's own asymptote is 1.5
# rather than _hill's fixed 1.0), mana_burn_c 3.3 -> 2.9 (p=4.0 unchanged)
# so the curve reaches that asymptote near 5 burnt pips, and
# game_penalty_cap 0.6 -> 1.5. Per-turn charge by pips burnt: pip1=0.021,
# pip2=0.277, pip3=0.801, pip4=1.175, pip5=1.348 (~90% of the weight),
# pip6=1.422, asymptoting toward 1.5.
#
# Weight and cap are deliberately EQUAL here (both 1.5) but do different
# jobs, and separating them is the point: under v4 (weight implicitly 1.0,
# cap 0.6) a single 4-pip turn computed 0.683 and was immediately clipped by
# the whole-game cap, so the cap -- not the curve -- set the per-turn
# magnitude, and every later mistake that game was free. Here the curve owns
# per-turn magnitude and the cap owns the whole-game sum, so one bad turn
# (1.35) is now distinguishable from several (up to 1.5) instead of both
# pinning at the same ceiling.
#   CORRECTED 2026-08-11, measured: that last sentence is only ~10% true.
#   One 5-pip turn charges 1.348 of a 1.5 cap -- 90% of the whole-game
#   budget -- so "one bad turn" vs "several" spans just 1.35..1.50. v5 did
#   NOT escape v4's failure mode, it only widened it slightly. See
#   deploy_reward_v6 below for the measurement and the fix.
#
# UNPROVEN RISK worth stating plainly: removing q removes the ONLY penalty
# for hoarding anywhere in the reward. The reasoning above (terminal outcome
# can see hoarding because it persists in state) is sound but has not been
# measured. If a new hoarding pathology appears under v5, this is the first
# thing to suspect, and the every-10k occurrence-rate check is where it
# should show up.
deploy_reward_v5 = with_dense_mana_burn_penalty(
    flat_win_loss_reward(),
    mana_burn_c=2.9, mana_burn_p=4.0, game_penalty_cap=1.5,
    refund_on_loss=True, mana_burn_weight=1.5,
)


# League self-play's CURRENT reward (run_league.py, 2026-08-12), replacing
# deploy_reward_v5 above. Staged 2026-08-11 behind an explicit gate -- apply
# only if a saturation replay (analyze_burn_saturation.py, since retired)
# still showed dmir_terror/elves
# saturating the cap in >50% of games at the 20,000-games/deck checkpoint,
# since a self-correcting v5 would have made the swap a pure confound.
# GATE RESULT (20,065 games/deck, 400 logged games): saturation went UP, not
# down -- dmir_terror 64% -> 71%, elves 69% -> 64%, and burn rose on every
# deck (dmir_terror 21.4 -> 30.3 pips/game, 2.00 -> 2.63 per own turn). The
# share of burn happening on the burner's OWN turn also rose (84% -> 89%),
# which rules out the benign reading that the extra burn was mana held up
# for instant-speed interaction on the opponent's turn: it is sequencing
# waste, which is what this term exists to price.
#
# ONE constant moves from v5: mana_burn_weight 1.5 -> 0.5. Base band
# (flat +1/-1), winner-only charging, curve shape (c=2.9, p=4.0) and
# game_penalty_cap (1.5) are all byte-identical, so the GUARANTEE is
# unchanged and needs no re-derivation: worst win = 1.0 - 1.5 = -0.5, every
# loss = -1.0 exactly, margin 0.5.
#
# WHY, MEASURED at v5's 10,003-games/deck checkpoint (analyze_burn_
# saturation.py over logs/v5_*_25games.json, 200 seat-games per deck): the
# cap was not bounding an occasional disaster, it was the normal case.
# dmir_terror exhausted it in 64% of games and elves in 69%, about two
# thirds of the way through each one -- so in most games the last third
# burnt mana entirely free of charge. A penalty that is already maxed out
# is a flat toll, not a gradient, and a flat toll cannot teach sequencing.
# dmir_terror's UNCAPPED charge averaged 4.26 against a 1.5 cap: 73% of the
# computed penalty was clipped away before it ever reached the buffer.
#
# WHY NOT JUST RAISE THE CAP (the owner's first instinct, and mine). The
# guarantee fixes a hard ceiling at cap < 2.0 (worst win 1.0-cap must beat
# every loss at -1.0), so 1.5 is already 75% of the maximum the shape can
# ever express, and the remaining 25% IS the safety margin. Spending it buys
# almost nothing: at weight 1.5, cap 1.8 moves dmir_terror from 64% to 62%
# saturating and elves from 69% to 66%, while cutting the worst-win-to-loss
# margin from 0.50 to 0.20. v3's collapse happened while its ordering
# guarantee held -- it was an optimization failure, not a reward-ordering
# bug -- so that margin is load-bearing against exactly the pathology v5 was
# built to fix. Three points of saturation is not worth 60% of it.
#
# WHY LOWERING THE WEIGHT IS NOT "WEAKENING THE PENALTY". The delivered tax
# barely moves (dmir_terror 1.13 -> 0.91 mean charge per game) because the
# cap was already discarding most of it. What changes is how much of it is
# PROPORTIONAL to actual waste: clipped fraction 73% -> 36%, saturation
# 64% -> 42%, and when the cap does die it dies at 81% through the game
# instead of 66%. Same total pressure, spread across the decisions that
# caused it. Per-turn the signal stays sharp -- a 3-pip turn costs 0.267
# against a +1.0 win -- and one 5-pip turn now costs 30% of the whole-game
# budget instead of 90%, which is what v5's comment above claimed to have
# achieved and did not.
#
# Per-turn charge by pips burnt: pip1=0.007, pip2=0.092, pip3=0.267,
# pip4=0.392, pip5=0.449 (~90% of the weight, same place on the curve v5's
# owner spec put it), pip6=0.474, asymptoting toward 0.5.
#
# WHAT v6 IS NOT RESPONDING TO. Win rate vs the frozen gauntlet twin went UP
# over the same 10k games (43.2% -> 47.4%), never-play-lands fell to 0.0% on
# three of four decks, and hoarding stayed low without q. v5's structural
# changes work and v6 keeps them untouched. But that overall win rate hides a
# split -- rakdos_madness 50->62%, dmir_terror 16->22%, mono_red_rally
# 78->85%, elves 29->20% -- and elves is a deck that burnt heavily and LOST
# ground, which is the case the shaping term should have caught and did not.
#
# OPEN QUESTION this does not settle: whether v5's flat toll CAUSED the high
# burn ("capped either way, so tap freely") or merely failed to prevent it.
# If burn does not move under v6, that is the distinction to test next, and
# at that point cap=1.8 becomes worth its margin cost as the only lever left.
deploy_reward_v6 = with_dense_mana_burn_penalty(
    flat_win_loss_reward(),
    mana_burn_c=2.9, mana_burn_p=4.0, game_penalty_cap=1.5,
    refund_on_loss=True, mana_burn_weight=0.5,
)
