"""Reward functions for training a DRL policy against the game engine.

Contract: any callable reward_fn(state, done, horizon) -> float, called once
per step with the state *after* that step's action. rl.training.train._reward_for
flips state.active_idx to the seat being scored, so reward_fn reads its own
seat's zones/counters and compares state.winner to itself. A sparse reward
returns 0.0 unless `done`; a dense one (see with_dense_mana_burn_penalty) can
return nonzero every call.

deploy_reward_v6 is the reward function used by production training and the
mulligan-net retrain scripts."""


def _hill(x, c, p):
    """Hill-function saturating curve: 0 at x=0, convex while x << c,
    inflects at x == c (h(c) == 0.5), concaves toward but never reaches 1.
    x is assumed >= 0."""
    if x <= 0:
        return 0.0
    xp = x ** p
    return xp / (xp + c ** p)


def flat_win_loss_reward():
    """Terminal reward: +1.0 for a win, -1.0 for a loss or a no-winner
    horizon timeout, 0.0 until done. Scored per seat
    (state.winner == state.active_idx). No efficiency/sloppiness terms --
    every win scores the same, every loss scores the same. Shaping layers
    on top via with_dense_mana_burn_penalty, applied only on the win band
    when refund_on_loss=True."""
    def reward_fn(state, done, horizon):
        if not done:
            return 0.0
        return 1.0 if state.winner == state.active_idx else -1.0
    return reward_fn


def _charge_single_pip_burn(player, mana_burn_c, mana_burn_p, game_penalty_cap, mana_burn_weight=1.0):
    """Marginal Hill-curve-with-cap charge for
    player.mana_burnt_this_turn_single_pip, called from the
    on_single_pip_burn hook right after that counter is incremented for the
    clear this call is for. Mutates player.mana_burn_penalty_credited /
    mana_burn_penalty_charged_total. mana_burn_weight scales the Hill
    curve's own asymptote; game_penalty_cap separately bounds the
    whole-game running total."""
    current = mana_burn_weight * _hill(player.mana_burnt_this_turn_single_pip, mana_burn_c, mana_burn_p)
    charge = current - player.mana_burn_penalty_credited
    player.mana_burn_penalty_credited = current
    # Clamp to the whole-game cap, floored at 0: a ceiling on badness, never
    # a bonus once the budget is fully spent.
    remaining = max(0.0, game_penalty_cap - player.mana_burn_penalty_charged_total)
    charge = min(charge, remaining)
    player.mana_burn_penalty_charged_total += charge
    return charge


def with_dense_mana_burn_penalty(base_reward_fn, mana_burn_c=3.3, mana_burn_p=4.0, game_penalty_cap=2.0,
                                  refund_on_loss=False, mana_burn_weight=1.0):
    """League self-play's dense mana-burn shaping: a Hill curve, cumulative
    per TURN and reset each turn (PlayerState.mana_burnt_this_turn_
    single_pip, reset by game.turn._run_turn_gen), so a badly-sloppy turn is
    only slightly worse than a mildly-sloppy one, but still dense and
    per-transition rather than terminal.

    SINGLE-PIP TAGGING: mana_burnt_this_turn_single_pip counts only
    mana-producing events that add exactly one symbol to the pool (a plain
    land or Llanowar Elves; not Rakdos Carnarium's automatic 2+, and a Tron
    land only while not all three types are online). Spending always
    consumes an untagged pip first, so a large correct burst (e.g. Priest of
    Titania's count_all tap) is never blamed -- only genuinely avoidable
    single-pip taps are.

    TELESCOPING: each burn (detected by game.turn._empty_mana_pools at a
    phase boundary) charges only the marginal increase in
    _hill(mana_burnt_this_turn_single_pip, c, p) since the last charge this
    turn (PlayerState.mana_burn_penalty_credited is the running baseline).
    Summed across a turn, the total charged equals
    _hill(total_burnt_this_turn, c, p) exactly, distributed across whichever
    Tap actions actually caused it.

    ATTRIBUTION: reward_fn below is a plain passthrough of base_reward_fn.
    The actual charge is computed by _charge_single_pip_burn and applied by
    rl.training.train.collect_rollout's on_single_pip_burn hook (exposed
    here as reward_fn.charge_single_pip_burn), called synchronously at the
    moment of the burn so it attributes the charge to the actual recent Tap
    actions rather than whatever's pending at the seat's next decision.

    game_penalty_cap bounds the whole-game running total
    (PlayerState.mana_burn_penalty_charged_total, never reset). Deliberately
    not a true potential function -- this is real, accumulating pressure
    against wasting mana, capped only as a blunt safety backstop against
    unbounded per-episode magnitude (a known source of PPO/GAE instability).

    refund_on_loss (opt-in, default False): when True, this penalty applies
    only to a seat that WON its game -- a losing seat pays nothing, so
    passive play doesn't score better than a real attempt to win. Tags the
    returned closure with mana_burn_winner_only=True, read by
    rl.training.train.collect_rollout (_wants_winner_only_burn).

    IMPLEMENTATION NOTE: this is not a terminal refund.
    rl.training.train.collect_rollout DEFERS the per-Tap writes and applies
    them only on a win, at game end (see its own deferred_charges). A
    lump-sum terminal refund would NOT be equivalent under GAE: a charge
    written at step t lands in delta_t directly, while a terminal refund
    reaches step t only discounted by (gamma*gae_lambda)^k -- leaving early
    burns mostly un-refunded. Deferring keeps a losing trajectory identical
    to one that never burnt at all."""
    def reward_fn(state, done, horizon):
        return base_reward_fn(state, done, horizon)
    reward_fn.charge_single_pip_burn = lambda player: _charge_single_pip_burn(
        player, mana_burn_c, mana_burn_p, game_penalty_cap, mana_burn_weight)
    if refund_on_loss:
        reward_fn.mana_burn_winner_only = True
    return reward_fn


# League self-play's CURRENT reward (run_league.py). Terminal band is flat
# +1/-1 (flat_win_loss_reward); dense mana-burn penalty applies only to the
# winning seat (refund_on_loss=True).
#
# GUARANTEE: worst-case win = 1.0 - game_penalty_cap = -0.5; every loss =
# -1.0 exactly. Worst win still beats best loss by 0.5.
deploy_reward_v6 = with_dense_mana_burn_penalty(
    flat_win_loss_reward(),
    mana_burn_c=2.9, mana_burn_p=4.0, game_penalty_cap=1.5,
    refund_on_loss=True, mana_burn_weight=0.5,
)
