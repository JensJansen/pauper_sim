import { describe, it, expect } from "vitest";
import { parseLog } from "./loadLog.js";
import fixture from "./__fixtures__/sampleGames.json";

describe("parseLog", () => {
  it("accepts a real harness.evaluate log", () => {
    const { meta, games } = parseLog(JSON.stringify(fixture));
    expect(games).toHaveLength(2);
    expect(meta).toHaveProperty("reward_fn");
    expect(games[0]).toHaveProperty("opening_hand_state");
    expect(games[0].opening_hand_state.hand.length).toBeGreaterThan(0);
  });

  it("rejects invalid JSON", () => {
    expect(() => parseLog("not json")).toThrow(/not valid JSON/i);
  });

  it("rejects a non-object", () => {
    expect(() => parseLog(JSON.stringify(["not", "an", "object"]))).toThrow(/must be a json object/i);
  });

  it("rejects an empty games array", () => {
    expect(() => parseLog(JSON.stringify({ meta: {}, games: [] }))).toThrow(/must be non-empty/i);
  });

  it("rejects game records missing expected keys, naming which ones", () => {
    expect(() => parseLog(JSON.stringify({ meta: {}, games: [{ missing: "keys" }] }))).toThrow(
      /game_index.*scores.*turn_won.*opening_hand_state.*steps.*end_state/s
    );
  });
});
