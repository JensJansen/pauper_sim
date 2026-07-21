// Parsing/validation for a harness.evaluate(log_path=...) JSON log: a
// top-level {meta, games} object, meta carrying the run identity (reward/
// scoring fn names, horizon, seed, ...) and games the per-game records.
const REQUIRED_GAME_KEYS = ["game_index", "scores", "turn_won", "opening_hand_state", "steps", "end_state"];

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

  const missing = REQUIRED_GAME_KEYS.filter((k) => !(k in data.games[0]));
  if (missing.length) {
    throw new Error(
      `Game records are missing expected key(s): ${missing.join(", ")} -- is this a harness.evaluate(log_path=...) log?`
    );
  }

  return { meta: data.meta || {}, games: data.games };
}
