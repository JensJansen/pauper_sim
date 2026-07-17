import { describe, it, expect } from "vitest";
import { flattenSteps, endingTurn } from "./flatten.js";
import fixture from "./__fixtures__/sampleGames.json";

const [successGame, failureGame] = fixture;

describe("flattenSteps", () => {
  it("starts with a synthetic opening-hand step matching the real opening hand", () => {
    const steps = flattenSteps(successGame);
    expect(steps[0].action).toBe("Opening hand");
    expect(steps[0].state_after.hand).toEqual(successGame.opening_hand);
    expect(steps[0].state_after.battlefield).toEqual([]);
  });

  it("has one step per opening-hand-step plus every logged action, in order", () => {
    const steps = flattenSteps(successGame);
    const totalActions = successGame.turns.reduce((n, t) => n + t.actions.length, 0);
    expect(steps).toHaveLength(1 + totalActions);
  });

  it("turn 1 (on the play) has no draw pseudo-action; later turns do, first", () => {
    const steps = flattenSteps(successGame);
    // step 1 is turn 1's first real action, not a draw (matches the source fixture)
    expect(steps[1].action).not.toBe("Draw a card");
    // turn 2 starts right after turn 1's actions (2 of them in this fixture)
    const turn2Start = 1 + successGame.turns[0].actions.length;
    expect(steps[turn2Start].action).toBe("Draw a card");
    expect(steps[turn2Start].fetched).toEqual([successGame.turns[1].drew]);
  });
});

describe("endingTurn", () => {
  it("uses turn_won for a success", () => {
    expect(endingTurn(successGame)).toBe(successGame.turn_won);
  });

  it("falls back to the last logged turn for a failure", () => {
    expect(failureGame.turn_won).toBeNull();
    const lastTurn = failureGame.turns[failureGame.turns.length - 1].turn;
    expect(endingTurn(failureGame)).toBe(lastTurn);
  });
});
