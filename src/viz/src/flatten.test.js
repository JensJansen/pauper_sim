import { describe, it, expect } from "vitest";
import { flattenSteps, flattenTwoPlayerSteps, endingTurn } from "./flatten.js";
import fixture from "./__fixtures__/sampleGames.json";
import twoPlayerFixture from "./__fixtures__/sampleTwoPlayerGames.json";

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

  it("uses final_turn_number for a 2-player draw (no turn_won)", () => {
    const drawGame = twoPlayerFixture.games.find((g) => g.turn_won == null);
    if (drawGame) {
      expect(endingTurn(drawGame)).toBe(drawGame.final_turn_number);
    }
  });
});

describe("flattenTwoPlayerSteps", () => {
  const [game] = twoPlayerFixture.games;

  it("starts with a synthetic opening-hand step for BOTH seats, matching opening_state", () => {
    const steps = flattenTwoPlayerSteps(game);
    expect(steps[0].action).toBe("Opening hand");
    expect(steps[0].state_after).toBe(game.opening_state);
    expect(steps[0].state_after.players).toHaveLength(2);
    expect(steps[0].turn_player_idx).toBe(game.starting_player_idx);
    expect(steps[0].actor_idx).toBeNull(); // no single decision made yet -- both hands shown at once
  });

  it("has one step per opening-hand-step plus every logged step, in order", () => {
    const steps = flattenTwoPlayerSteps(game);
    expect(steps).toHaveLength(1 + game.steps.length);
  });

  it("every real step names an actor seat and echoes both players' own state", () => {
    const steps = flattenTwoPlayerSteps(game);
    for (const step of steps.slice(1)) {
      expect(step.state_after.players).toHaveLength(2);
      expect([0, 1]).toContain(step.actor_idx);
      expect([0, 1]).toContain(step.turn_player_idx);
    }
  });
});
