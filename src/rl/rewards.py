"""Reward functions for training a DRL policy against the game engine.

Contract: any callable

    reward_fn(state: game.GameState, done: bool, horizon: int) -> float

Called once per environment step with the state *after* that step's
action was applied. A sparse reward function returns 0.0 unless `done`;
a dense one could return something every call. No base class -- any
matching callable works.

Only WIN/LOSS rewards live here (no dense shaping): a win is scaled by how
efficiently it was reached; a loss/timeout is either a flat 0.0
(action_count_win_reward) or a small floor scaled down by wasted cards
(deploy_reward). The old single-player Tron-era heuristics (resource-quality
tie-breakers) and the turn-decay fast_win_reward went away with the 1-player
pipeline. Pretraining uses action_count_win_reward_200_floor02; league
self-play uses deploy_reward_v2 (= deploy_reward with win_floor=1.0 -> flat win;
the v1 efficiency scaling caused an action-space-minimization pathology).
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
        # Self-contained loser gate: rl.train._reward_for used to zero the loser
        # externally (drl_env._lost) before ever calling this. That gate is gone
        # now (deploy_reward needs the loser to reach its own loss band), so this
        # legacy reward must decide it itself -- a non-winning seat still scores 0.
        # state.active_idx is the seat being scored (rl.train._reward_for flips it).
        if state.winner != state.active_idx:
            return 0.0
        winner_actions = state.players[state.winner].actions_taken
        over_plateau = min(max(0, winner_actions - plateau_actions), span)
        return 1.0 - over_plateau / span * (1.0 - min_reward)
    return reward_fn


def deploy_reward(plateau_actions=80, max_actions=200, win_floor=0.5, loss_default=0.25,
                  discard_penalty=0.05):
    """Two-band terminal reward. Scored PER SEAT: rl.train._reward_for flips
    state.active_idx to the seat being scored (via drl_env._for_player) and no
    longer zeroes the loser first, so this callable decides win vs loss itself.

    WIN (state.winner == this seat): win_floor (0.5) -> 1.0, scaled by the
    winner's own GAMEPLAY efficiency -- actions_taken MINUS pregame_actions (the
    mulligan/keep/bottom picks are excluded so a winner that had to mulligan
    isn't docked for it): <= plateau_actions (80) gameplay actions -> 1.0 (a fast
    win), >= max_actions (200) -> win_floor (a long, grindy win); linear between.
    Every win outscores every loss (win_floor > loss_default).

    LOSS or no-winner timeout (state.winner != this seat, including None -- the
    only non-win 2-player outcome, a horizon cap; genuine draws can't happen,
    win_check awards every life/deck-out end to a seat): loss_default (0.25)
    DECREMENTED by discard_penalty (0.05) per turn this seat discarded to cleanup
    (hoarded cards it never played), floored at 0, so a hoarder that never deploys
    bottoms out at 0 while a deck that deployed its hand and lost keeps 0.25.

    The MULLIGAN decision is NOT scored here anymore: it's owned by the separate
    per-deck mulligan model (rl.mulligan), which the main policy doesn't drive, so
    penalizing the main policy for mulligans it can't control would be pure noise.
    (The old terminal mulligan penalty also never propagated to the mulligan
    decision through ~100 steps of discounting -- the whole reason that model
    exists.) Terminal only (0.0 until done)."""
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
        return max(0.0, loss_default - discard_penalty * p.cleanup_discard_turns)
    return reward_fn


# Pre-baked named instance (callers reference reward_fns by plain name via
# getattr off this module -- see rl.train's own reward_fn_name plumbing).
# Floor lowered to 0.2 (vs. the default 0.25) per the "sliding scale from
# 1 - 0.2" spec -- this is the reward pretraining (run_pretrain.py) uses.
action_count_win_reward_200_floor02 = action_count_win_reward(min_reward=0.2)

# The reward league self-play used BEFORE deploy_reward_v2 replaced it below --
# kept for reference/comparison, not wired into run_league.py. Win band
# 0.5->1.0 by gameplay efficiency; loss band 0.25 minus 0.05 per
# cleanup-discard turn, floored at 0.
deploy_reward_v1 = deploy_reward()


# The reward league self-play uses NOW (run_league.py): deploy_reward with
# win_floor=1.0, which collapses the win band to a FLAT 1.0 (win_span = 0) -- no
# efficiency scaling -- while keeping the identical loss band. v1's efficiency
# scaling induced an action-space-minimization pathology (the policy generalized
# "win in fewer actions -> more reward" into "shrink your own board"); flat win
# removes that gradient. Float-first removed the tap/untap-abandon churn
# structurally (no undo action exists), so the old dense abandon penalty is gone;
# PPO entropy alone now bounds pointless actions without capping them.
deploy_reward_v2 = deploy_reward(win_floor=1.0)


if __name__ == "__main__":
    # ponytail self-check: run via `python -m rl.rewards` from src/.
    # action_count_win_reward: per-seat (state.players[winner].actions_taken),
    # not turn_number -- a real 2-player state (state.winner needs a second
    # seat to mean anything).
    from game.state import GameState, PlayerState

    # Default instance (0.25 floor) -- built locally just for this check; the
    # only pre-baked module-level instance the pipeline ships is the 0.2-floor one.
    rf = action_count_win_reward()

    state2 = GameState(on_the_play=True, players=[PlayerState(True), PlayerState(False)])
    state2.turn_won = None
    assert rf(state2, done=True, horizon=120) == 0.0  # no winner -> 0
    assert rf(state2, done=False, horizon=120) == 0.0  # not done -> 0

    state2.turn_won = 5
    state2.winner = 0

    # Plateau: anything at or under plateau_actions (80) scores the full
    # 1.0, no reward at all for going even faster -- "sufficiently fast",
    # per request.
    state2.players[1].actions_taken = 999  # the LOSER's own count must never matter
    state2.players[0].actions_taken = 1
    assert rf(state2, done=True, horizon=120) == 1.0
    state2.players[0].actions_taken = 80
    assert rf(state2, done=True, horizon=120) == 1.0

    # Linear ramp from (80, 1.0) to (200, 0.25) -- midpoint (140) should
    # land exactly halfway between.
    state2.players[0].actions_taken = 140
    assert abs(rf(state2, done=True, horizon=120) - 0.625) < 1e-9

    # Floor: exactly 0.25 at max_actions (200), and bottoms out there --
    # never continues down toward 0 for a wildly long game past the cap.
    state2.players[0].actions_taken = 200
    win_at_cap = rf(state2, done=True, horizon=120)
    assert abs(win_at_cap - 0.25) < 1e-9
    state2.players[0].actions_taken = 5000
    win_past_cap = rf(state2, done=True, horizon=120)
    assert win_at_cap == win_past_cap  # bottomed out -- doesn't keep decaying below this

    # legacy reward now self-contains the loser gate (the external loser gate is
    # gone): a non-winning seat (state.winner != state.active_idx) scores 0 on its own.
    state2.active_idx = 1  # score seat 1; winner is 0 -> a loss for this seat
    assert rf(state2, done=True, horizon=120) == 0.0
    state2.active_idx = 0

    print("rewards.py action_count_win_reward self-check: OK")

    # --- deploy_reward: two-band terminal reward, scored per seat ---
    dr = deploy_reward()  # defaults: plateau 80, max 200, win_floor 0.5, 0.05 penalties
    s = GameState(on_the_play=True, players=[PlayerState(True), PlayerState(False)])
    s.active_idx = 0  # score seat 0 throughout (rl.train._reward_for flips this in real use)

    assert dr(s, done=False, horizon=120) == 0.0  # terminal only

    # WIN band (seat 0 is the winner): 0.5 + 0.5*efficiency, efficiency on
    # GAMEPLAY actions (actions_taken - pregame_actions).
    s.winner = 0
    s.players[1].actions_taken = 999  # loser's count must never matter
    s.players[0].pregame_actions = 5  # 5 mulligan/keep/bottom picks -- excluded below
    s.players[0].actions_taken = 5 + 10   # 10 gameplay actions -> under plateau
    assert dr(s, done=True, horizon=120) == 1.0
    s.players[0].actions_taken = 5 + 80   # exactly plateau gameplay actions
    assert dr(s, done=True, horizon=120) == 1.0
    s.players[0].actions_taken = 5 + 140  # 140 gameplay -> midpoint of [80,200]
    assert abs(dr(s, done=True, horizon=120) - 0.75) < 1e-9  # 0.5 + 0.5*0.5
    s.players[0].actions_taken = 5 + 200  # >= max gameplay -> win floor
    assert abs(dr(s, done=True, horizon=120) - 0.5) < 1e-9
    s.players[0].actions_taken = 5 + 5000
    assert abs(dr(s, done=True, horizon=120) - 0.5) < 1e-9  # floored, never below win_floor

    # LOSS band (seat 0 is NOT the winner): 0.25 minus 0.05/discard-turn, floored
    # at 0. Mulligans are NOT scored here (the mulligan model owns that now).
    s.winner = 1
    s.players[0].cleanup_discard_turns = 0
    s.players[0].mulligans_taken = 7   # mulligans no longer affect the loss band
    assert abs(dr(s, done=True, horizon=120) - 0.25) < 1e-9  # played its hand, lost (mulligans ignored)
    s.players[0].cleanup_discard_turns = 3
    assert abs(dr(s, done=True, horizon=120) - 0.10) < 1e-9  # 0.25 - 3*0.05
    s.players[0].cleanup_discard_turns = 6
    assert dr(s, done=True, horizon=120) == 0.0  # over-penalized on discards -> clamped, never negative

    # No-winner timeout (winner None) uses the loss band for whichever seat.
    s.winner = None
    s.players[0].cleanup_discard_turns = 1
    assert abs(dr(s, done=True, horizon=120) - 0.20) < 1e-9  # 0.25 - 1*0.05

    print("rewards.py deploy_reward self-check: OK")

    # --- deploy_reward_v2 == deploy_reward(win_floor=1.0): the win band flattens to
    # 1.0 regardless of game length (the loss band is deploy_reward's, checked above) ---
    v2 = deploy_reward(win_floor=1.0)
    s.winner = 0  # reuse the fixture above (pregame_actions=5)
    s.players[0].actions_taken = 5 + 5000   # a long, grindy win
    assert v2(s, done=True, horizon=120) == 1.0  # flat -- no efficiency dock
    s.players[0].actions_taken = 5 + 1          # a fast win scores identically
    assert v2(s, done=True, horizon=120) == 1.0
    print("rewards.py deploy_reward_v2 (win_floor=1.0 -> flat win) self-check: OK")
