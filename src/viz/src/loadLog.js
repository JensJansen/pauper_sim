// Parsing/validation for a harness.evaluate(log_path=...) JSON log.
// Pure function -- no DOM/File API here, so it's directly unit-testable.
const REQUIRED_KEYS = ["game_index", "scores", "turn_won", "opening_hand", "turns", "end_state"];

export function parseLog(text) {
  let data;
  try {
    data = JSON.parse(text);
  } catch (e) {
    throw new Error(`Not valid JSON (${e.message})`);
  }

  if (!Array.isArray(data) || data.length === 0) {
    throw new Error("Must be a non-empty JSON array of game records");
  }

  const missing = REQUIRED_KEYS.filter((k) => !(k in data[0]));
  if (missing.length) {
    throw new Error(
      `Records are missing expected key(s): ${missing.join(", ")} -- is this a harness.evaluate(log_path=...) log?`
    );
  }

  return data;
}
