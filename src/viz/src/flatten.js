// Pure: turns a game record into one ordered list of "steps" (opening
// hand as a synthetic step 0, then every turn's actions in order -- draw
// pseudo-actions are already embedded in the log, nothing extra needed).
export function flattenSteps(game) {
  const openingStep = {
    action: "Opening hand",
    fetched: [],
    left_battlefield: [],
    tapped_for_cost: [],
    state_after: {
      turn_number: 1,
      hand: game.opening_hand,
      battlefield: [],
      graveyard: [],
      resource_quality: {
        non_land_permanents: 0,
        available_mana: 0,
        hand_size: game.opening_hand.length,
      },
    },
  };

  const steps = [openingStep];
  for (const turn of game.turns) {
    for (const action of turn.actions) steps.push(action);
  }
  return steps;
}

export function endingTurn(game) {
  if (game.turn_won !== null) return game.turn_won;
  if (game.turns.length) return game.turns[game.turns.length - 1].turn;
  return 1;
}
