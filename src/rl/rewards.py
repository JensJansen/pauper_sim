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
deploy_reward_v2 (= with_mana_mistake_penalty wrapping deploy_reward with
win_floor=1.0 -> flat efficiency term; the v1 efficiency scaling caused an
action-space-minimization pathology).

DENSE shaping: with_mana_mistake_penalty (below) wraps a base reward_fn with
a per-transition penalty for mana burnt where the engine could find no
justification for it (see game.turn._empty_mana_pools's three-way exemption).
Deliberately NOT folded into the terminal badness score above -- the whole
point is to attribute a mistake to roughly the transition that caused it,
not blur it across the entire game the way a terminal-only signal already
was shown to (Tolarian Terror-style credit misattribution)."""


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
    """Wraps a base reward_fn with a DENSE, per-transition penalty for mana
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
# exact guarantee and its win_floor=1.0 precondition. with_mana_mistake_penalty
# (above) wraps this with a dense per-transition term on top; see its own
# docstring for why that guarantee's margin, already thin by construction, is
# a real (if deliberately bounded) tradeoff of adding it.
deploy_reward_v2 = with_mana_mistake_penalty(deploy_reward(win_floor=1.0))
