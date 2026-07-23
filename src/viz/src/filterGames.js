// Pure: AND-combined search filters over a loaded batch of games. All
// criteria are optional; only active (non-null) ones are applied. Works
// over either log shape (1-player or 2-player -- see loadLog.js), detected
// per game off g.scores' own shape: a plain dict (1-player, one score set)
// vs. an array of two (2-player, one per seat) -- the same discriminator
// loadLog.js uses meta.seats for, just at the per-game level since this
// module never sees meta.
import { flattenSteps, flattenTwoPlayerSteps } from "./flatten.js";

function isTwoPlayer(game) {
  return Array.isArray(game.scores);
}

function everStep(game, predicate) {
  const steps = isTwoPlayer(game) ? flattenTwoPlayerSteps(game) : flattenSteps(game);
  return steps.some(predicate);
}

// "Ever," for a 2-player game, means "in EITHER seat's own zone" -- a
// search for one card name isn't scoped to a particular agent.
function everInHand(game, cardName) {
  return everStep(game, (step) =>
    isTwoPlayer(game)
      ? step.state_after.players.some((p) => p.hand.includes(cardName))
      : step.state_after.hand.includes(cardName)
  );
}

function everInPlay(game, cardName) {
  return everStep(game, (step) =>
    isTwoPlayer(game)
      ? step.state_after.players.some((p) => p.battlefield.some((perm) => perm.name === cardName))
      : step.state_after.battlefield.some((p) => p.name === cardName)
  );
}

export function filterGames(games, criteria = {}) {
  // turnWonMin/Max replaces the old turnOnlineMin/Max (the "online" concept
  // -- a second, stricter termination tier -- no longer exists as tracked
  // state; turn_won is the single remaining termination turn). scoreMin/Max
  // now filters on the primary score (reward_fn's named entry, always
  // first/present in the scores dict) rather than a display-only "score2"
  // that no longer has a guaranteed scale (a deck's own scoring_fns could
  // be anything, e.g. Tron's tron_online_score is a plain 0/1, not a
  // 0-100 percentage). For a 2-player game, "the primary score" means seat
  // 0's (agent_a's) own reward_fn value -- same "sort/filter off seat 0"
  // convention harness.evaluate_two_player's own log-writing already uses,
  // since a 2-player game has no single score shared by both seats.
  const { turnWonMin, turnWonMax, cardInHand, cardInPlay, scoreMin, scoreMax } = criteria;

  return games.filter((g) => {
    const primaryScore = isTwoPlayer(g) ? Object.values(g.scores[0])[0] : Object.values(g.scores)[0];
    if (turnWonMin != null && (g.turn_won == null || g.turn_won < turnWonMin)) return false;
    if (turnWonMax != null && (g.turn_won == null || g.turn_won > turnWonMax)) return false;
    if (scoreMin != null && primaryScore < scoreMin) return false;
    if (scoreMax != null && primaryScore > scoreMax) return false;
    if (cardInHand && !everInHand(g, cardInHand)) return false;
    if (cardInPlay && !everInPlay(g, cardInPlay)) return false;
    return true;
  });
}
