// Pure: AND-combined search filters over a loaded batch of games. All
// criteria are optional; only active (non-null) ones are applied.
import { flattenSteps } from "./flatten.js";

function everStep(game, predicate) {
  return flattenSteps(game).some(predicate);
}

function everInHand(game, cardName) {
  return everStep(game, (step) => step.state_after.hand.includes(cardName));
}

function everInPlay(game, cardName) {
  return everStep(game, (step) => step.state_after.battlefield.some((p) => p.name === cardName));
}

export function filterGames(games, criteria = {}) {
  // turnWonMin/Max replaces the old turnOnlineMin/Max (the "online" concept
  // -- a second, stricter termination tier -- no longer exists as tracked
  // state; turn_won is the single remaining termination turn). scoreMin/Max
  // now filters on the primary score (reward_fn's named entry, always
  // first/present in the scores dict) rather than a display-only "score2"
  // that no longer has a guaranteed scale (a deck's own scoring_fns could
  // be anything, e.g. Tron's tron_online_score is a plain 0/1, not a
  // 0-100 percentage).
  const { turnWonMin, turnWonMax, cardInHand, cardInPlay, scoreMin, scoreMax } = criteria;

  return games.filter((g) => {
    const primaryScore = Object.values(g.scores)[0];
    if (turnWonMin != null && (g.turn_won == null || g.turn_won < turnWonMin)) return false;
    if (turnWonMax != null && (g.turn_won == null || g.turn_won > turnWonMax)) return false;
    if (scoreMin != null && primaryScore < scoreMin) return false;
    if (scoreMax != null && primaryScore > scoreMax) return false;
    if (cardInHand && !everInHand(g, cardInHand)) return false;
    if (cardInPlay && !everInPlay(g, cardInPlay)) return false;
    return true;
  });
}
