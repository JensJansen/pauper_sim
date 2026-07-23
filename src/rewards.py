"""Reward functions for training a DRL policy against game.py's simulator.

Contract (see DRL_PLAN.md "Reward function contract"): any callable

    reward_fn(state: game.GameState, done: bool, horizon: int) -> float

Called once per environment step with the state *after* that step's
action was applied. A sparse reward function returns 0.0 unless `done`;
a dense one could return something every call. No base class -- any
matching callable works.
"""

import game


def resource_quality_components(state):
    """Raw (unnormalized) values behind resource_quality -- also useful on
    its own for display purposes (e.g. the visualizer's live readout),
    where "3 non-land permanents, 5 available mana, 4 cards in hand" is
    more legible than a single combined 0-3 number."""
    non_land_permanents = sum(
        1 for p in state.battlefield if p.card_def.card_type != game.CardType.LAND
    )

    available_mana = 0
    for p in state.battlefield:
        effect = p.card_def.effect_id
        if p.tapped or effect not in game.SIMPLE_MANA_SOURCE_EFFECTS:
            continue
        spec = game.EFFECT_REGISTRY[effect]
        # Saruli Caretaker: its own extra cost (tap another untapped
        # creature) must be currently payable, same guard mana.py's
        # tap_cost_options uses before ever offering it as a source.
        extra_available = spec.get("mana_extra_available")
        if extra_available is not None and not extra_available(state, p):
            continue
        if spec["mana"][0] == "flexible":
            available_mana += 1  # flexible sources always produce exactly 1 mana, any color
        else:
            available_mana += len(game.mana_output(p, state))

    hand_size = len(state.hand)

    return {
        "non_land_permanents": non_land_permanents,
        "available_mana": available_mana,
        "hand_size": hand_size,
    }


NON_LAND_PERMANENTS_CAP = 3
AVAILABLE_MANA_CAP = 12
HAND_SIZE_CAP = 5


def resource_quality(state):
    """Quality-of-victory tie-breaker (DRL_PLAN.md): non-land permanents
    still in play + mana available right now + cards in hand, each
    normalized to [0, 1] against its own cap and summed, so the total is
    in [0, 3] regardless of what the individual caps are -- that's what
    keeps assembled_with_resource_quality's turn-dominance guarantee valid
    even if the caps themselves change."""
    c = resource_quality_components(state)
    return (
        min(c["non_land_permanents"], NON_LAND_PERMANENTS_CAP) / NON_LAND_PERMANENTS_CAP
        + min(c["available_mana"], AVAILABLE_MANA_CAP) / AVAILABLE_MANA_CAP
        + min(c["hand_size"], HAND_SIZE_CAP) / HAND_SIZE_CAP
    )


def resource_quality_pct(state):
    """Display-only "score 2": resource_quality rescaled from its native
    [0, 3] range to [0, 100], where 100 means every component hit its cap
    (a "perfect" board state). Ignores turn number entirely (unlike
    assembled_with_resource_quality, "score 1") -- but a failed game
    (never assembled) always scores 0, same failure gate as score 1, so a
    resource-rich board on a loss doesn't outrank a modest board on a win."""
    if state.turn_won is None:
        return 0.0
    return 100.0 * resource_quality(state) / 3.0


def tron_online_score(state):
    """MULTI_DECK_PLAN.md Phase M7: Tron's second scoring function --
    successor to the old turn_online concept (the turn all three Tron
    types were simultaneously untapped, tracked as a second termination
    tier). New design, not a behavior-preserving port: a post-hoc check
    on the final board state (1.0 if all three types are present AND
    every one of them is untapped, 0.0 otherwise) instead of a tracked
    turn number, since scoring functions only ever run once, at game end.
    A failed game already scores 0.0 via finalize_scores' central gate;
    the explicit check here is the same belt-and-suspenders pattern
    resource_quality_pct/assembled_with_resource_quality both use."""
    if state.turn_won is None:
        return 0.0
    if not game.controls_all_tron_types(state):
        return 0.0
    all_untapped = all(
        not p.tapped for p in state.battlefield if p.card_def.effect_id == game.EffectId.TRON_LAND
    )
    return 1.0 if all_untapped else 0.0


def assembled_with_resource_quality(state, done, horizon):
    """Failure -> 0. Success at turn T -> 0.85**T + 0.02 * resource_quality.

    The 0.02 coefficient is derived (DRL_PLAN.md), not tuned: it's small
    enough that no amount of resource_quality can ever outweigh finishing
    one turn earlier -- turn always dominates.
    """
    if not done:
        return 0.0
    if state.turn_won is None:
        return 0.0
    return (0.85 ** state.turn_number) + 0.02 * resource_quality(state)


def strict_binary_reward(state, done, horizon):
    """Turn-of-win only, no tiebreaker, no board-state dependency at all --
    deck-agnostic despite originating as spy_combo's reward function
    (renamed off "spy_combo_reward" once reused by other decks;
    DECK_REGISTRY_REFRESH_PLAN.md). Failure (horizon reached, decked out,
    or any other non-win) -> 0. Success at turn T -> 0.85**T.

    Originally spy_combo's own reward, simplified by request from an
    earlier version that also scored board state/mana/hand as a
    same-turn tiebreaker (SPY_REWARD_PLAN.md Tasks 3/5/6) -- dropped as
    unnecessary complexity. Turn-dominance is trivial (0.85**T is
    strictly decreasing in T on its own), so there's no coefficient to
    derive or re-derive if horizon changes -- the reason it generalizes
    cleanly to any deck with no per-deck tuning."""
    if not done:
        return 0.0
    if state.turn_won is None:
        return 0.0
    return 0.85 ** state.turn_number


def fast_win_reward(decay=0.99):
    """Factory for a pure win/loss reward with a gentle preference for
    winning sooner -- same shape as strict_binary_reward (0 for a loss,
    draw, or horizon cutoff; decay**turn_number for a win), but with its
    own tunable decay instead of reusing that function's fixed 0.85.

    0.85 was calibrated for the short, ~6-10 turn horizons every 1-player
    deck uses -- fine there, but at a real 2-player adversarial game's own
    natural horizon (this decklist's deck-out ceiling is ~109 turns,
    docs/MULTIPLAYER_ENGINE_PLAN.md's boggles_mirror config), 0.85**109 is
    effectively 0: a turn-100 win would be reward-indistinguishable from a
    loss, destroying the entire win/loss signal for any normal-length
    game. decay close to 1 keeps that from happening -- adjacent turns
    stay nearly equal (an intentional "prefer speed, but only as a
    tiebreaker" request) while a win at ANY reachable turn stays clearly
    above 0.

    decay stamped onto the returned function itself (not just closed
    over), same convention terminated.damage_threshold_terminated's own
    threshold already uses -- lets a caller holding only reward_fn read
    back getattr(reward_fn, "decay", None) without a separate copy of the
    number."""
    def reward_fn(state, done, horizon):
        if not done or state.turn_won is None:
            return 0.0
        return decay ** state.turn_number
    reward_fn.decay = decay
    return reward_fn


# Pre-baked named instance -- configs/*.json reference reward_fns by plain
# name (getattr off this module), same convention terminated.py's own
# damage_threshold_20_terminated uses. 0.99: turn 1 vs turn 3 differ by
# only ~2% ("marginal", as requested), while a win even at boggles_mirror's
# own ~109-turn deck-out ceiling (0.99**109 ~= 0.34) stays clearly above a
# loss's flat 0.
# ponytail: 0.99 chosen for boggles_mirror specifically, not derived --
# retune (or add another named instance) if a deck with a very different
# natural horizon ever needs this reward too.
fast_win_reward_099 = fast_win_reward(0.99)


def action_count_win_reward(plateau_actions=80, max_actions=200, min_reward=0.25):
    """Win/loss reward like fast_win_reward, but the "prefer efficiency"
    axis is the WINNING seat's own action count (PlayerState.actions_taken
    -- real, non-Pass actions only, per-seat, never combined across both
    seats, and never counting an automatic draw-for-turn as an "action" --
    see actions_taken's own docstring) instead of the global turn_number --
    request: turn-based decay lets a policy take arbitrarily many actions
    within a single turn "for free" (nothing below turn granularity was
    ever measured), so it can't actually discourage padding out a turn with
    pointless actions the way a per-action metric can.

    Piecewise linear, not a pure decay -- a plain decay**actions_taken (the
    first version of this function) starts penalizing from action 1, and
    even clamped to a floor, that floor came out at only 0.99**204 ~= 0.13:
    too close to a loss's flat 0 to read as "still clearly a win," and
    nothing rewarded finishing within a perfectly reasonable number of
    actions any more than finishing in fewer. This version has three flat
    request-driven reference points instead of one derived decay constant:
    actions_taken <= plateau_actions (80, "sufficiently fast" -- no
    reason to reward going even faster) -> max_reward (1.0); actions_taken
    >= max_actions (200) -> min_reward (0.25, clearly above a loss's 0, per
    request); linear ramp between the two. A loss or draw is still exactly
    0.0 either way -- only a win's OWN value moves.

    plateau_actions/max_actions/min_reward stamped onto the returned
    function itself, same convention terminated.damage_threshold_
    terminated's own threshold already uses."""
    span = max_actions - plateau_actions
    def reward_fn(state, done, horizon):
        if not done or state.turn_won is None:
            return 0.0
        winner_actions = state.players[state.winner].actions_taken
        over_plateau = min(max(0, winner_actions - plateau_actions), span)
        return 1.0 - over_plateau / span * (1.0 - min_reward)
    reward_fn.plateau_actions = plateau_actions
    reward_fn.max_actions = max_actions
    reward_fn.min_reward = min_reward
    return reward_fn


# Pre-baked named instance -- same configs/*.json-by-plain-name convention
# as fast_win_reward_099/damage_threshold_20_terminated.
action_count_win_reward_200 = action_count_win_reward()


if __name__ == "__main__":
    # ponytail self-check: run via `python rewards.py` from src/.
    from game.state import GameState

    assert fast_win_reward_099.decay == 0.99

    state = GameState(on_the_play=True)
    state.turn_number = 1
    state.turn_won = None
    assert fast_win_reward_099(state, done=True, horizon=120) == 0.0  # no winner (loss/draw) -> 0
    assert fast_win_reward_099(state, done=False, horizon=120) == 0.0  # not done yet -> 0

    state.turn_won = 1
    assert abs(fast_win_reward_099(state, done=True, horizon=120) - 0.99) < 1e-9  # win at turn 1

    # Strictly decreasing in turn number, but only marginally (turn 1 vs
    # turn 3 within ~2%) -- the whole point of a 0.99 decay over 0.85.
    win_turn_1 = fast_win_reward_099(state, done=True, horizon=120)
    state.turn_number = 3
    win_turn_3 = fast_win_reward_099(state, done=True, horizon=120)
    assert win_turn_3 < win_turn_1
    assert (win_turn_1 - win_turn_3) / win_turn_1 < 0.03

    # Even at boggles_mirror's own ~109-turn natural deck-out ceiling, a
    # win still stays clearly above a loss's flat 0 -- 0.85 (every other
    # deck's shared constant) could not make this same claim.
    state.turn_number = 109
    win_turn_109 = fast_win_reward_099(state, done=True, horizon=120)
    assert win_turn_109 > 0.3

    print("rewards.py fast_win_reward self-check: OK")

    # action_count_win_reward: per-seat (state.players[winner].actions_taken),
    # not turn_number -- a real 2-player state this time (state.winner needs
    # a second seat to mean anything).
    from game.state import PlayerState

    assert action_count_win_reward_200.plateau_actions == 80
    assert action_count_win_reward_200.max_actions == 200
    assert action_count_win_reward_200.min_reward == 0.25

    state2 = GameState(on_the_play=True, players=[PlayerState(True), PlayerState(False)])
    state2.turn_won = None
    assert action_count_win_reward_200(state2, done=True, horizon=120) == 0.0  # no winner -> 0
    assert action_count_win_reward_200(state2, done=False, horizon=120) == 0.0  # not done -> 0

    state2.turn_won = 5
    state2.winner = 0

    # Plateau: anything at or under plateau_actions (80) scores the full
    # 1.0, no reward at all for going even faster -- "sufficiently fast",
    # per request.
    state2.players[1].actions_taken = 999  # the LOSER's own count must never matter
    state2.players[0].actions_taken = 1
    assert action_count_win_reward_200(state2, done=True, horizon=120) == 1.0
    state2.players[0].actions_taken = 80
    assert action_count_win_reward_200(state2, done=True, horizon=120) == 1.0

    # Linear ramp from (80, 1.0) to (200, 0.25) -- midpoint (140) should
    # land exactly halfway between.
    state2.players[0].actions_taken = 140
    assert abs(action_count_win_reward_200(state2, done=True, horizon=120) - 0.625) < 1e-9

    # Floor: exactly 0.25 at max_actions (200), and bottoms out there --
    # never continues down toward 0 for a wildly long game past the cap.
    state2.players[0].actions_taken = 200
    win_at_cap = action_count_win_reward_200(state2, done=True, horizon=120)
    assert abs(win_at_cap - 0.25) < 1e-9
    state2.players[0].actions_taken = 5000
    win_past_cap = action_count_win_reward_200(state2, done=True, horizon=120)
    assert win_at_cap == win_past_cap  # bottomed out -- doesn't keep decaying below this

    print("rewards.py action_count_win_reward self-check: OK")
