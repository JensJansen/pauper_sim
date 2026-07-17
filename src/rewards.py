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
        if p.tapped or p.card_def.effect_id not in game.SIMPLE_MANA_SOURCE_EFFECTS:
            continue
        if p.card_def.effect_id in (game.EffectId.WOODED_RIDGELINE, game.EffectId.BONDERS_ORNAMENT):
            available_mana += 1  # flexible-color sources always produce exactly 1 mana
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


def _phase_d1_sanity_check():
    import random

    # Not done -> 0.0 regardless of contents.
    mid_state = game.GameState(on_the_play=True, rng=random.Random(0))
    mid_state.turn_number = 3
    assert assembled_with_resource_quality(mid_state, done=False, horizon=6) == 0.0

    # Done, never assembled -> 0.0.
    failed_state = game.GameState(on_the_play=True, rng=random.Random(0))
    failed_state.turn_number = 6
    failed_state.turn_won = None
    assert assembled_with_resource_quality(failed_state, done=True, horizon=6) == 0.0

    # Done, assembled turn 3, zero resources -> 0.85**3.
    turn3_min = game.GameState(on_the_play=True, rng=random.Random(0))
    turn3_min.turn_number = 3
    turn3_min.turn_won = 3
    r_turn3_min = assembled_with_resource_quality(turn3_min, done=True, horizon=6)
    assert abs(r_turn3_min - 0.85 ** 3) < 1e-9, r_turn3_min

    # Done, assembled turn 6, maxed-out resource_quality -> 0.85**6 + 0.06.
    # Caps are 3 non-land permanents / 12 available mana / 5 cards in hand:
    # 3 untapped Bonder's Ornaments (1 flexible mana each) + 9 untapped
    # Forests (1 mana each) = 3 non-land permanents and 12 mana exactly;
    # 5 cards in hand exactly.
    turn6_max = game.GameState(on_the_play=True, rng=random.Random(0))
    turn6_max.turn_number = 6
    turn6_max.turn_won = 6
    turn6_max.battlefield = (
        [game.Permanent(game.CARD_DEFS["Bonder's Ornament"]) for _ in range(3)]
        + [game.Permanent(game.CARD_DEFS["Forest"]) for _ in range(9)]
    )
    turn6_max.hand = [game.CARD_DEFS["Forest"]] * 5
    quality = resource_quality(turn6_max)
    assert quality == 3.0, quality  # every term hit its cap
    r_turn6_max = assembled_with_resource_quality(turn6_max, done=True, horizon=6)
    assert abs(r_turn6_max - (0.85 ** 6 + 0.02 * 3.0)) < 1e-9, r_turn6_max

    # The guarantee itself: worst turn-3 success still beats best turn-6 success.
    assert r_turn3_min > r_turn6_max, (r_turn3_min, r_turn6_max)

    # score 2 (resource_quality_pct): 0 at the floor, 100 at the same
    # maxed-out state used above.
    assert resource_quality_pct(turn3_min) == 0.0
    assert resource_quality_pct(turn6_max) == 100.0

    # score 2's failure gate: a resource-maxed board still scores 0 if the
    # game never actually assembled Tron -- resource quality alone can't
    # buy a failure a nonzero score.
    failed_but_resource_rich = game.GameState(on_the_play=True, rng=random.Random(0))
    failed_but_resource_rich.turn_number = 6
    failed_but_resource_rich.turn_won = None
    failed_but_resource_rich.battlefield = turn6_max.battlefield
    failed_but_resource_rich.hand = turn6_max.hand
    assert resource_quality(failed_but_resource_rich) == 3.0  # resources genuinely maxed
    assert resource_quality_pct(failed_but_resource_rich) == 0.0  # but the gate zeroes it anyway


def _phase_m7_sanity_check():
    """MULTI_DECK_PLAN.md Phase M7: tron_online_score, the successor to
    the old turn_online concept, now a post-hoc scoring function."""
    import random

    # All three types present and untapped -> online.
    online = game.GameState(on_the_play=True, rng=random.Random(0))
    online.turn_number = 4
    online.turn_won = 4
    online.battlefield = [
        game.Permanent(game.CARD_DEFS["Urza's Mine"]),
        game.Permanent(game.CARD_DEFS["Urza's Power Plant"]),
        game.Permanent(game.CARD_DEFS["Urza's Tower"]),
    ]
    assert tron_online_score(online) == 1.0

    # All three present, but one tapped (e.g. used to pay for the 3rd
    # piece's own land drop cost isn't possible, but a prior activation
    # could tap one) -> not online.
    not_online = game.GameState(on_the_play=True, rng=random.Random(0))
    not_online.turn_number = 4
    not_online.turn_won = 4
    not_online.battlefield = [
        game.Permanent(game.CARD_DEFS["Urza's Mine"], tapped=True),
        game.Permanent(game.CARD_DEFS["Urza's Power Plant"]),
        game.Permanent(game.CARD_DEFS["Urza's Tower"]),
    ]
    assert tron_online_score(not_online) == 0.0

    # Failure gate: an untapped-Tron board still scores 0 if the game
    # never actually terminated (turn_won is None) -- same centrally
    # redundant belt-and-suspenders pattern as the other scoring functions.
    failed = game.GameState(on_the_play=True, rng=random.Random(0))
    failed.turn_number = 6
    failed.turn_won = None
    failed.battlefield = online.battlefield
    assert tron_online_score(failed) == 0.0


if __name__ == "__main__":
    _phase_d1_sanity_check()
    print("Phase D1 OK: reward_fn matches the formula, and turn always dominates resource_quality.")

    _phase_m7_sanity_check()
    print("Phase M7 OK: tron_online_score correctly reads all-untapped board state, and is gated by the failure check.")
