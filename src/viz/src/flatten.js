// Pure: turns a game record into one ordered list of "steps" (opening
// hand as a synthetic step 0 built from the log's own real opening-hand
// snapshot, then game.steps as-is -- already a flat, turn-stamped
// sequence, draw pseudo-actions included, nothing to reassemble here).
export function flattenSteps(game) {
  const openingStep = {
    turn: 1,
    action: "Opening hand",
    fetched: [],
    left_battlefield: [],
    tapped_for_cost: [],
    decision: null,
    fallback: false,
    state_after: game.opening_hand_state,
  };

  return [openingStep, ...game.steps];
}

// 2-player counterpart: opening_state (both seats' own snapshot) instead
// of opening_hand_state, starting_player_idx stands in for turn_player_idx
// (whoever's about to take turn 1) since no real decision -- and therefore
// no actor -- has happened yet.
export function flattenTwoPlayerSteps(game) {
  const openingStep = {
    turn_number: 1,
    turn_player_idx: game.starting_player_idx,
    actor_idx: null,
    phase: null,
    action: "Opening hand",
    fetched: [],
    left_battlefield: [],
    tapped_for_cost: [],
    decision: null,
    fallback: false,
    state_after: game.opening_state,
  };
  return [openingStep, ...game.steps];
}

export function endingTurn(game) {
  if (game.turn_won != null) return game.turn_won;
  // 2-player draw (safety-cap horizon reached, no winner): final_turn_number
  // is always present on a 2-player game record, win or draw alike, unlike
  // 1-player's turn_won-or-last-logged-step fallback below.
  if (game.final_turn_number != null) return game.final_turn_number;
  if (game.steps.length) return game.steps[game.steps.length - 1].turn;
  return 1;
}
