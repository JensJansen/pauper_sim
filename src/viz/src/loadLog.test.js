import { describe, it, expect } from "vitest";
import { parseLog } from "./loadLog.js";
import fixture from "./__fixtures__/sampleGames.json";
import twoPlayerFixture from "./__fixtures__/sampleTwoPlayerGames.json";

describe("parseLog", () => {
  it("accepts a real harness.evaluate log", () => {
    const { meta, games, twoPlayer } = parseLog(JSON.stringify(fixture));
    expect(games).toHaveLength(2);
    expect(meta).toHaveProperty("reward_fn");
    expect(games[0]).toHaveProperty("opening_hand_state");
    expect(games[0].opening_hand_state.hand.length).toBeGreaterThan(0);
    expect(twoPlayer).toBe(false);
  });

  it("accepts a real harness.evaluate_two_player log, discriminated by meta.seats", () => {
    const { meta, games, twoPlayer } = parseLog(JSON.stringify(twoPlayerFixture));
    expect(twoPlayer).toBe(true);
    expect(meta.seats).toHaveLength(2);
    expect(games.length).toBeGreaterThan(0);
    expect(games[0]).toHaveProperty("opening_state");
    expect(games[0].opening_state.players).toHaveLength(2);
    expect(games[0].scores).toHaveLength(2);
  });

  it("rejects a two-player game record missing an expected key", () => {
    const broken = { meta: { seats: [{}, {}] }, games: [{ game_index: 0 }] };
    expect(() => parseLog(JSON.stringify(broken))).toThrow(/winner.*opening_state/s);
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
