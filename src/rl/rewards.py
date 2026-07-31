"""Reward functions for training a DRL policy against the game engine.

Contract: any callable

    reward_fn(state: game.GameState, done: bool, horizon: int) -> float

Called once per environment step with the state *after* that step's
action was applied. A sparse reward function returns 0.0 unless `done`;
a dense one could return something every call. No base class -- any
matching callable works.

Only WIN/LOSS rewards live here (no dense shaping): a win is scaled by how
efficiently it was reached, minus a "sloppiness" penalty shared with the loss
band (see _badness); a loss/timeout is 0.0 minus that same penalty
(action_count_win_reward has no such penalty -- it's the plain, pretrain-only
predecessor). Pretraining uses action_count_win_reward_200_floor02; league
self-play uses deploy_reward_v2 (= deploy_reward with win_floor=1.0 -> flat
efficiency term; the v1 efficiency scaling caused an action-space-
minimization pathology).
"""


def _hill(x, c, p):
    """Hill-function saturating curve: 0 at x=0, convex (slow-then-
    accelerating) while x << c, inflects around x == c (h(c) == 0.5), then
    concaves back toward -- but never reaches -- 1 as x grows without bound.
    x is assumed >= 0 (a raw count/amount, never negative)."""
    if x <= 0:
        return 0.0
    xp = x ** p
    return xp / (xp + c ** p)


def _badness(mana_burnt, cleanup_discard_turns, mana_burn_c, mana_burn_p, discard_c, discard_p):
    """Combines two independent [0, 1)-saturating badness scores -- overtapped/
    dissipated mana (rule 500.4 pool-empties, PlayerState.mana_burnt_total) and
    hoarded cards forced to cleanup discard (PlayerState.cleanup_discard_turns)
    -- into one [0, 1) score, via noisy-or (1 minus the product of each
    factor's "goodness" complement): 0 only when BOTH inputs are 0, saturates
    toward 1 if EITHER input alone saturates, and -- unlike a plain sum --
    can never exceed 1 without clamping. Deliberately owner-specified default
    curves (mana_burn_c=5, mana_burn_p=3, discard_c=4, discard_p=2): a couple
    of stray burnt mana or one cleanup discard is nearly free, ~10 cumulative
    burnt mana is already a severe penalty, and it keeps growing (slower)
    past that -- see the design discussion that produced this shape for the
    exact target numbers."""
    h_burn = _hill(mana_burnt, mana_burn_c, mana_burn_p)
    h_discard = _hill(cleanup_discard_turns, discard_c, discard_p)
    return 1.0 - (1.0 - h_burn) * (1.0 - h_discard)


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
                   mana_burn_c=5.0, mana_burn_p=3.0, discard_c=4.0, discard_p=2.0):
    """Two-band terminal reward. Scored PER SEAT: rl.train._reward_for flips
    state.active_idx to the seat being scored (via drl_env._for_player) and no
    longer zeroes the loser first, so this callable decides win vs loss itself.

    Both bands subtract the SAME sloppiness penalty, q = _badness(...) in
    [0, 1) -- combining PlayerState.mana_burnt_total (mana tapped and left to
    dissipate, rule 500.4) and .cleanup_discard_turns (cards hoarded past hand
    size) -- so a win with a lot of overtapped mana or hoarded cards scores
    barely above a clean loss, not barely below a clean win.

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
        q = _badness(p.mana_burnt_total, p.cleanup_discard_turns, mana_burn_c, mana_burn_p, discard_c, discard_p)
        if state.winner == seat:
            gameplay_actions = p.actions_taken - p.pregame_actions
            over = min(max(0, gameplay_actions - plateau_actions), span)
            return win_floor + win_span * (1.0 - over / span) - q
        return -q
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
# one factory.
deploy_reward_v1 = deploy_reward()


# League self-play's reward (run_league.py): deploy_reward with win_floor=1.0,
# which collapses the win band's EFFICIENCY term to a flat 1.0 (win_span = 0,
# gameplay-action-count no longer moves the score) -- while still subtracting
# the mana-burn/cleanup-discard sloppiness penalty q from both bands. v1's
# efficiency scaling induces an action-space-minimization pathology (the policy
# generalizes "win in fewer actions -> more reward" into "shrink your own
# board"); flat win removes that gradient. Float-first means no undo action
# exists, so there's no tap/untap-abandon churn to structurally guard against;
# PPO entropy alone bounds pointless actions without capping them. Because q
# never reaches 1, every win here still strictly outscores every loss no
# matter how sloppy either was -- see deploy_reward's own docstring for the
# exact guarantee and its win_floor=1.0 precondition.
deploy_reward_v2 = deploy_reward(win_floor=1.0)
