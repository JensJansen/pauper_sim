import { describe, it, expect } from "vitest";
import { flattenSteps, endingTurn } from "./flatten.js";
import fixture from "./__fixtures__/sampleGames.json";

const [successGame, failureGame] = fixture.games;

describe("flattenSteps", () => {
  it("starts with a synthetic opening-hand step matching the real opening hand snapshot", () => {
    const steps = flattenSteps(successGame);
    expect(steps[0].action).toBe("Opening hand");
    expect(steps[0].state_after).toBe(successGame.opening_hand_state);
    expect(steps[0].state_after.battlefield).toEqual([]);
  });

  it("has one step per opening-hand-step plus every logged step, in order", () => {
    const steps = flattenSteps(successGame);
    expect(steps).toHaveLength(1 + successGame.steps.length);
  });

  it("turn 1 (on the play) has no draw pseudo-action; later turns do, first", () => {
    const steps = flattenSteps(successGame);
    // step 1 is turn 1's first real action, not a draw (matches the source fixture)
    expect(steps[1].action).not.toBe("Draw a card");
    const turn2Start = steps.findIndex((s) => s.action === "Draw a card");
    expect(turn2Start).toBeGreaterThan(0);
    expect(steps[turn2Start].turn).toBe(2);
  });
});

describe("endingTurn", () => {
  it("uses turn_won for a success", () => {
    expect(endingTurn(successGame)).toBe(successGame.turn_won);
  });

  it("falls back to the last logged turn for a failure", () => {
    expect(failureGame.turn_won).toBeNull();
    const lastTurn = failureGame.steps[failureGame.steps.length - 1].turn;
    expect(endingTurn(failureGame)).toBe(lastTurn);
  });
});
