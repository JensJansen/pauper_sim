"""Reward functions for training a DRL policy against the game engine.

Contract: any callable

    reward_fn(state: game.GameState, done: bool, horizon: int) -> float

Called once per environment step with the state *after* that step's
action was applied. A sparse reward function returns 0.0 unless `done`;
a dense one could return something every call. No base class -- any
matching callable works.

Only WIN/LOSS rewards live here (no dense shaping): a win is scaled by how
efficiently it was reached; a loss/timeout is either a flat 0.0
(action_count_win_reward) or 0.0 minus a small, front-loaded penalty for any
hand-size cleanup discards (deploy_reward). Pretraining uses
action_count_win_reward_200_floor02; league self-play uses deploy_reward_v2
(= deploy_reward with win_floor=1.0 -> flat win; the v1 efficiency scaling
caused an action-space-minimization pathology).
"""


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


def deploy_reward(plateau_actions=80, max_actions=200, win_floor=0.5, discard_base=0.02):
    """Two-band terminal reward. Scored PER SEAT: rl.train._reward_for flips
    state.active_idx to the seat being scored (via drl_env._for_player) and no
    longer zeroes the loser first, so this callable decides win vs loss itself.

    WIN (state.winner == this seat): win_floor (0.5) -> 1.0, scaled by the
    winner's own GAMEPLAY efficiency -- actions_taken MINUS pregame_actions (the
    mulligan/keep/bottom picks are excluded so a winner that had to mulligan
    isn't docked for it): <= plateau_actions (80) gameplay actions -> 1.0 (a fast
    win), >= max_actions (200) -> win_floor (a long, grindy win); linear between.
    Every win outscores every loss (win_floor > 0 >= every loss score below).

    LOSS or no-winner timeout (state.winner != this seat, including None -- the
    only non-win 2-player outcome, a horizon cap; genuine draws can't happen,
    win_check awards every life/deck-out end to a seat): exactly 0.0 if this
    seat was never forced to discard to cleanup (cleanup_discard_turns == 0),
    else -(discard_base ** cleanup_discard_turns). discard_base (0.02) < 1, so
    this SHRINKS toward 0 as more discard-turns pile up -- the first
    discard-turn is the loudest signal (-0.02), a real deterrent against ANY
    hoarding at all, and each further one matters less. Deliberately the
    opposite growth direction from rl.mulligan's own convex MULLIGAN_COST
    penalty (which gets WORSE per mulligan): front-loads the punishment onto
    committing the infraction at all, not onto how much. Always within
    (-discard_base, 0] for cleanup_discard_turns >= 0, so no explicit floor is
    needed here -- unlike the mulligan side, this can never approach a runaway
    negative.

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
        if state.winner == seat:
            gameplay_actions = p.actions_taken - p.pregame_actions
            over = min(max(0, gameplay_actions - plateau_actions), span)
            return win_floor + win_span * (1.0 - over / span)
        n = p.cleanup_discard_turns
        return -(discard_base ** n) if n > 0 else 0.0
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
# which collapses the win band to a flat 1.0 (win_span = 0) -- no efficiency
# scaling -- while keeping the identical loss band. v1's efficiency scaling
# induces an action-space-minimization pathology (the policy generalizes "win
# in fewer actions -> more reward" into "shrink your own board"); flat win
# removes that gradient. Float-first means no undo action exists, so there's
# no tap/untap-abandon churn to structurally guard against; PPO entropy alone
# bounds pointless actions without capping them. Loss band is exactly 0.0 with
# no discards, else -(0.02 ** cleanup_discard_turns) -- see deploy_reward's
# own docstring for why this front-loads the penalty onto the first hoarding
# discard rather than scaling with how many.
deploy_reward_v2 = deploy_reward(win_floor=1.0)
