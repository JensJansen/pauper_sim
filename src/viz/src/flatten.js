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

export function endingTurn(game) {
  if (game.turn_won !== null) return game.turn_won;
  if (game.steps.length) return game.steps[game.steps.length - 1].turn;
  return 1;
}
