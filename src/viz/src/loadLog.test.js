import { describe, it, expect } from "vitest";
import { parseLog } from "./loadLog.js";
import fixture from "./__fixtures__/sampleGames.json";

describe("parseLog", () => {
  it("accepts a real harness.evaluate log", () => {
    const games = parseLog(JSON.stringify(fixture));
    expect(games).toHaveLength(2);
    expect(games[0]).toHaveProperty("opening_hand");
    expect(games[0].opening_hand).toHaveLength(7);
  });

  it("rejects invalid JSON", () => {
    expect(() => parseLog("not json")).toThrow(/not valid JSON/i);
  });

  it("rejects a non-array", () => {
    expect(() => parseLog(JSON.stringify({ not: "a list" }))).toThrow(/non-empty JSON array/i);
  });

  it("rejects an empty array", () => {
    expect(() => parseLog("[]")).toThrow(/non-empty JSON array/i);
  });

  it("rejects records missing expected keys, naming which ones", () => {
    expect(() => parseLog(JSON.stringify([{ missing: "keys" }]))).toThrow(
      /game_index.*scores.*turn_won.*opening_hand.*turns.*end_state/s
    );
  });
});
