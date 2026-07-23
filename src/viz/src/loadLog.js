// Parsing/validation for a harness.evaluate(log_path=...) JSON log: a
// top-level {meta, games} object, meta carrying the run identity (reward/
// scoring fn names, horizon, seed, ...) and games the per-game records.
//
// Two shapes exist: harness.evaluate's own 1-player log (one hand/
// battlefield/etc per step, meta.reward_fn a plain string) and harness.
// evaluate_two_player's 2-player log (both seats' own zones per step,
// meta.seats a 2-element array -- the one field reliably present in the
// 2-player shape and absent from the 1-player one, so it's the shape
// discriminator rather than guessing off any single game record).
const REQUIRED_GAME_KEYS = ["game_index", "scores", "turn_won", "opening_hand_state", "steps", "end_state"];
const REQUIRED_TWO_PLAYER_GAME_KEYS = [
  "game_index", "scores", "winner", "turn_won", "final_turn_number", "opening_state", "steps", "end_state",
];

export function parseLog(text) {
  let data;
  try {
    data = JSON.parse(text);
  } catch (e) {
    throw new Error(`Not valid JSON (${e.message})`);
  }

  if (data === null || typeof data !== "object" || !Array.isArray(data.games)) {
    throw new Error("Must be a JSON object with a \"games\" array -- is this a harness.evaluate(log_path=...) log?");
  }
  if (data.games.length === 0) {
    throw new Error("\"games\" must be non-empty");
  }

  const twoPlayer = Array.isArray(data.meta?.seats);
  const requiredKeys = twoPlayer ? REQUIRED_TWO_PLAYER_GAME_KEYS : REQUIRED_GAME_KEYS;
  const missing = requiredKeys.filter((k) => !(k in data.games[0]));
  if (missing.length) {
    throw new Error(
      `Game records are missing expected key(s): ${missing.join(", ")} -- is this a harness.evaluate(log_path=...) log?`
    );
  }

  return { meta: data.meta || {}, games: data.games, twoPlayer };
}
