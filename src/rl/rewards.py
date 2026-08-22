"""Reward functions for training a DRL policy against the game engine.

Contract: any callable

    reward_fn(state: game.GameState, done: bool, horizon: int) -> float

Called once per environment step with the state *after* that step's action
was applied. rl.train._reward_for flips state.active_idx to the seat being
scored (via drl_env._for_player) before calling this, so a reward_fn always
reads its own seat's zones/counters and compares state.winner to itself,
whether the call is terminal or not. A sparse reward function returns 0.0
unless `done`; a dense one (see with_dense_mana_burn_penalty) can return
something on every call. No base class -- any matching callable works.

deploy_reward_v6 (below) is the only reward function used by production
training (run_league.py) and by the mulligan-net retrain scripts in
src/analysis/. Five earlier reward generations (action_count_win_reward,
deploy_reward v1-v5, with_mana_mistake_penalty) explored terminal-band shape
and dense mana-burn shaping between 2026-08-06 and 2026-08-12; they were
superseded in production well before removal and are gone from this file as
of 2026-08-22 -- see git history for their own docstrings if the full
blow-by-blow derivation is ever needed again. deploy_reward_v6's own
docstring below summarizes the measured findings its current constants are
calibrated against."""


def _hill(x, c, p):
    """Hill-function saturating curve: 0 at x=0, convex (slow-then-
    accelerating) while x << c, inflects around x == c (h(c) == 0.5), then
    concaves back toward -- but never reaches -- 1 as x grows without bound.
    x is assumed >= 0 (a raw count/amount, never negative)."""
    if x <= 0:
        return 0.0
    xp = x ** p
    return xp / (xp + c ** p)


def flat_win_loss_reward():
    """The simplest possible terminal reward: +1.0 for a win, -1.0 for a loss
    OR a no-winner horizon timeout, 0.0 until done. Scored PER SEAT exactly
    as this module's own contract describes (state.active_idx flipped to the
    scored seat before this is called, so `state.winner == state.active_idx`
    IS "did this seat win").

    The league moved to this flat shape (2026-08-11) away from an earlier
    two-band design that scaled a win's score by the winning seat's own
    action-count efficiency: that scaling let a policy generalize "win in
    fewer actions -> more reward" into "shrink your own board," an
    action-space-minimization pathology unrelated to actually winning. Flat
    win removes that gradient entirely -- there is no undo action to
    structurally guard against tap/untap-abandon churn, and PPO entropy
    alone bounds pointless actions without capping them.

    No sloppiness/efficiency terms of any kind: every win scores exactly the
    same before shaping, and every loss scores exactly the same, full stop.
    Any WITHIN-band gradation is a lever a policy can optimize independently
    of actually winning -- the earlier design's own cleanup-discard penalty
    and its loss-band mana-burn exposure were both found to reward passive
    play (losing quietly, or never developing a board) over a real attempt
    to win. Shaping still layers on top via with_dense_mana_burn_penalty,
    but only on the WIN band (deploy_reward_v6 passes refund_on_loss=True)."""
    def reward_fn(state, done, horizon):
        if not done:
            return 0.0
        return 1.0 if state.winner == state.active_idx else -1.0
    return reward_fn


def _charge_single_pip_burn(player, mana_burn_c, mana_burn_p, game_penalty_cap, mana_burn_weight=1.0):
    """The marginal Hill-curve-with-cap charge for player.mana_burnt_this_
    turn_single_pip, AS IT STANDS RIGHT NOW -- called from rl.train's
    on_single_pip_burn hook, which fires from inside game.turn.
    _empty_mana_pools AFTER that counter has already been incremented for
    the clear this call is for, so "current" below already reflects this
    specific burn. Mutates player.mana_burn_penalty_credited/
    mana_burn_penalty_charged_total as a side effect.

    mana_burn_weight (2026-08-11) SCALES the Hill curve itself, so one
    turn's own worst case approaches mana_burn_weight rather than _hill's
    own fixed 1.0 asymptote. Distinct from game_penalty_cap, which bounds
    the WHOLE-GAME running total and does not reshape any single charge --
    separating the two lets the curve own per-turn magnitude while the cap
    owns the whole-game sum, instead of one bad turn immediately pinning
    the cap and making every later mistake that game free."""
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
    """League self-play's dense mana-burn shaping: a Hill curve, per-TURN
    cumulative and reset each turn (PlayerState.mana_burnt_this_turn_
    single_pip, game.turn._run_turn_gen resets it), so a badly-sloppy turn
    is barely worse than a slightly-sloppy one rather than unboundedly
    worse, but genuinely DENSE and per-transition rather than terminal.
    Replaced an earlier per-mistake dense penalty (with_mana_mistake_
    penalty, removed 2026-08-22): that predecessor was too weak in
    practice, and its exemption logic forgave floated mana whenever nothing
    was castable, which is exactly the case meant to be punished here.

    PER-PIP SINGLE-PIP TAG: mana_burnt_this_turn_single_pip (game.mana.
    float_mana/spend_one_pip, game.turn._empty_mana_pools) attributes PER
    PIP: a mana-producing EVENT is tagged "single-pip" iff it adds exactly 1
    symbol to the pool in that one event (dynamic, len(produced) == 1) -- a
    plain land or Llanowar Elves always qualifies; Rakdos Carnarium/Utopia
    Sprawl's automatic bonus (2+ symbols) never does; a Tron land qualifies
    only while not all three Tron types are online; Priest of Titania/
    Overgrown Battlement qualify only in the edge case their count resolves
    to exactly 1. PlayerState.mana_pool_single_pip shadow-counts how many of
    the CURRENTLY FLOATING pips of each color are tagged. Spending a pip
    (game.mana.spend_one_pip) always consumes an UNTAGGED pip of that color
    first -- so a mana ability's large, correct, intrinsic archetype burst
    (Priest of Titania's "count_all" tap, summing every Elf on both
    battlefields with no partial-tap option) is always spent/burnt ahead of
    any tagged single-pip mana of the same color, meaning
    mana_burnt_this_turn_single_pip never blames that unavoidable excess
    while still correctly catching a genuinely avoidable single-pip tap
    (e.g. 2 unneeded Llanowar Elves alongside a Priest burst that alone
    would have covered the spend). Reflexive tapping of ordinary single-pip
    sources -- the actual problem this wrapper exists for -- is fully
    penalized with per-pip precision.

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
    one-shot terminal score would have given for that turn, just
    distributed across whichever individual Tap transitions actually caused
    it (proper credit assignment, not blurred across the whole game).

    ATTRIBUTION: reward_fn below is a plain passthrough of base_reward_fn.
    The actual charge is computed by _charge_single_pip_burn above and
    applied by rl.train.collect_rollout's own on_single_pip_burn hook
    (exposed here as reward_fn.charge_single_pip_burn, an opt-in-attribute
    pattern) -- called SYNCHRONOUSLY from inside game.turn.
    _empty_mana_pools, at the exact moment of the burn, so it can charge the
    actual recent single-pip Tap actions directly instead of whatever's
    pending at a seat's next unrelated decision (a seat keeps priority after
    any non-Pass action, so a whole run of taps happens back-to-back with
    mana_burnt_this_turn_single_pip completely unchanged between them -- an
    earlier version of this charge that ran inline inside reward_fn at a
    seat's own NEXT decision measured only 1.8% of transitions ever
    carrying a nonzero charge, landing on a Tap action no more often than
    Pass or anything else, before this synchronous hook fixed it).

    game_penalty_cap bounds the WHOLE-GAME running total this wrapper ever
    charges (PlayerState.mana_burn_penalty_charged_total, never reset once
    the game is underway). Deliberately NOT a true potential function (one
    Hill curve over the whole-GAME cumulative burn, never reset) instead:
    Ng, Harada & Russell 1999's invariance theorem guarantees potential-
    based shaping does not change the optimal policy at all -- exactly the
    opposite of the point of adding this term, which is to inject a real,
    lasting preference against wasting mana beyond whatever plain win/loss
    already implies. So: real, accumulating, non-potential-based pressure,
    capped in aggregate as a blunt safety backstop (not principled shaping)
    -- a reward term whose magnitude scales with episode length with no
    ceiling is a known source of PPO/GAE instability, and this repo already
    hit that failure mode once for real (an earlier, similarly unbounded
    mana-burn penalty was walked back for the training problems it caused).

    Unlike with_mana_mistake_penalty (removed), still no EXEMPTION checking
    (mana_burnt_this_turn_single_pip is always tallied by
    _empty_mana_pools regardless of reward_fn) -- it DOES need the separate
    on_single_pip_burn hook for ATTRIBUTION, gated the same way
    consumes_mana_mistake gates on_mana_burn: rl.train.collect_rollout
    wires on_single_pip_burn in only when at least one tracked seat's
    reward_fn exposes a charge_single_pip_burn attribute.

    refund_on_loss (2026-08-11, opt-in, default False): when True, this
    penalty applies ONLY to a seat that WON its game; a seat that lost (or
    timed out with no winner) pays exactly nothing for whatever it burnt.
    Tags the returned closure with a `mana_burn_winner_only = True`
    attribute, read by rl.train.collect_rollout (_wants_winner_only_burn).

    Why: charging this penalty on a LOSS made losing PASSIVELY score
    strictly better than losing while trying. A seat that never taps mana
    cannot burn mana, so it scored exactly 0.0 on this term by
    construction, every time; a seat that developed its board and made any
    ordinary sequencing mistake paid up to the full cap. Measured on a real
    10,003-games/deck run: one archetype's passive losses averaged 0.321
    total penalty vs. 0.598 for its active losses (n=14 vs 64) -- a
    structural discount for not playing. "Winner-only" rather than "smaller
    on a loss": any nonzero loss-band charge reintroduces the same
    ordering, just with a smaller gap -- the asymmetry is removed by
    construction only when the loss band carries no burn term at all.

    IMPLEMENTATION IS NOT A TERMINAL REFUND -- rl.train.collect_rollout
    DEFERS the per-Tap writes instead, holding each (buffer_index, share)
    until the game ends and applying them only on a win (see its own
    deferred_charges comment). A lump-sum refund at the terminal transition
    would NOT be equivalent: PPO trains on GAE advantages, where a charge
    written at step t lands in delta_t directly and immediately, while a
    terminal refund reaches step t only through GAE's backward recursion,
    discounted by (gamma*gae_lambda)^k over the k steps between them
    (~11% of the refund survives 40 steps back at this repo's own
    gamma=0.99/gae_lambda=0.95). That would leave EARLY burns in a long
    losing game mostly un-refunded while late ones cancel cleanly -- a
    distance-dependent residue, not the intended "costs nothing on a loss."
    Deferring the write leaves a losing trajectory bit-for-bit identical to
    one that never burnt at all, which is the actual guarantee wanted here."""
    def reward_fn(state, done, horizon):
        return base_reward_fn(state, done, horizon)
    reward_fn.charge_single_pip_burn = lambda player: _charge_single_pip_burn(
        player, mana_burn_c, mana_burn_p, game_penalty_cap, mana_burn_weight)
    if refund_on_loss:
        reward_fn.mana_burn_winner_only = True
    return reward_fn


# League self-play's CURRENT reward (run_league.py). Terminal band is a
# flat +1/-1 (flat_win_loss_reward, above -- no cleanup-discard penalty of
# any kind, on either band). The dense mana-burn penalty (with_dense_mana_
# burn_penalty, above) applies ONLY to a seat that WON its game
# (refund_on_loss=True) -- see that function's own docstring for the full
# mechanics (per-pip single-pip tagging, telescoping Hill curve, whole-game
# cap, and why "winner-only" rather than "smaller on a loss").
#
# GUARANTEE: worst-case win = 1.0 - game_penalty_cap = -0.5; every loss =
# -1.0 exactly, with no range at all. Worst win beats best loss by 0.5. A
# sufficiently sloppy win CAN score negative (-0.5 at the cap) -- harmless,
# since the only ordering that matters is win-vs-loss and -0.5 > -1.0
# always.
#
# CONSTANTS AND WHY (calibrated against five earlier reward generations,
# 2026-08-06 through 2026-08-12, removed 2026-08-22 -- see git history for
# their own docstrings if the full derivation is ever needed again):
#
# - Flat +1/-1 with no efficiency-scaling or cleanup-discard term (unlike
#   the removed deploy_reward family): an earlier two-band design that
#   scaled a win by the winner's own action-count kept a hoarding/
#   sloppiness penalty on both bands; dropping it works because hoarded
#   cards stay VISIBLE in game state (an overflowing hand, uncast threats,
#   an undeveloped board), so a terminal win/loss signal is positioned to
#   attribute their cost on its own given enough training -- unlike burnt
#   mana, which disappears from state the instant a phase boundary clears
#   the pool, and is exactly why it still needs dense, per-Tap-attributed
#   shaping instead. UNPROVEN RISK: this removed the only penalty for
#   hoarding anywhere in the reward. If a hoarding pathology appears, this
#   is the first thing to suspect.
#
# - mana_burn_weight=0.5 (down from an earlier 1.5): at weight 1.5, the
#   whole-game cap (1.5) was measured saturating in the majority of games
#   (one archetype at 64%, another at 69%, at a 20,065-games/deck
#   checkpoint) -- meaning for the last third of most games, burnt mana was
#   entirely free of charge, because a maxed-out penalty is a flat toll,
#   not a gradient, and can't teach sequencing. Raising the cap instead was
#   rejected: the worst-win-vs-best-loss guarantee fixes a hard ceiling at
#   cap < 2.0, so 1.5 already spends 75% of the margin the shape can
#   express, and that margin is load-bearing against the same
#   optimization-failure pathology (guaranteed inaction beating a real
#   attempt to win) that an earlier reward generation hit despite its
#   ordering guarantee holding throughout. Lowering the weight instead
#   barely changes the delivered tax (one archetype's mean charge per
#   game: 1.13 -> 0.91) but sharply changes how much of it is proportional
#   to actual waste (clipped fraction 73% -> 36%, saturation 64% -> 42%),
#   spreading the same total pressure across the decisions that actually
#   caused it instead of maxing out two-thirds through the game.
#
# - mana_burn_c=2.9, mana_burn_p=4.0, game_penalty_cap=1.5: unchanged from
#   the weight change above. Per-turn charge by pips burnt at weight=0.5:
#   pip1=0.007, pip2=0.092, pip3=0.267, pip4=0.392, pip5=0.449 (~90% of the
#   weight), pip6=0.474, asymptoting toward 0.5.
#
# MEASURED RESULT of the winner-only + flat-band structural changes
# (carried unchanged from the reward generation that introduced them): win
# rate vs. the frozen gauntlet twin went up over 10k games/deck (43.2% ->
# 47.4%), never-play-lands fell to 0.0% on three of four decks, and
# hoarding stayed low without a discard penalty. That aggregate hides a
# split -- one deck burnt heavily and LOST ground (29% -> 20% vs the
# gauntlet twin) while the other three gained -- and mana-burn shaping
# should have caught that deck's regression and did not.
#
# OPEN QUESTION this does not settle: whether the flat per-turn cap CAUSED
# high burn rates ("capped either way, so tap freely") or merely failed to
# prevent them. If burn does not move under this weight, that's the
# distinction to test next -- at that point game_penalty_cap=1.8 becomes
# worth its margin cost as the only lever left.
deploy_reward_v6 = with_dense_mana_burn_penalty(
    flat_win_loss_reward(),
    mana_burn_c=2.9, mana_burn_p=4.0, game_penalty_cap=1.5,
    refund_on_loss=True, mana_burn_weight=0.5,
)
