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
