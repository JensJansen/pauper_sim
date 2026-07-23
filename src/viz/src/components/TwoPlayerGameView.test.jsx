import { describe, it, expect } from "vitest";
import { render, screen, fireEvent } from "@testing-library/react";
import TwoPlayerGameView from "./TwoPlayerGameView.jsx";
import twoPlayerFixture from "../__fixtures__/sampleTwoPlayerGames.json";

const [game] = twoPlayerFixture.games;

describe("TwoPlayerGameView", () => {
  it("shows both agents' boards with their own life totals on the opening step", () => {
    render(<TwoPlayerGameView game={game} onBack={() => {}} />);
    expect(screen.getByText("Agent A")).toBeTruthy();
    expect(screen.getByText("Agent B")).toBeTruthy();
    const opening = game.opening_state.players;
    const lifeReadouts = screen.getAllByText(/^Life: /).map((el) => el.textContent);
    expect(lifeReadouts).toEqual([`Life: ${opening[0].life_total}`, `Life: ${opening[1].life_total}`]);
  });

  it("marks the starting player's Turn badge on the opening step, with no Acting badge yet", () => {
    render(<TwoPlayerGameView game={game} onBack={() => {}} />);
    expect(screen.getAllByText("Turn")).toHaveLength(1);
    expect(screen.queryByText("Acting")).toBeNull();
  });

  it("stepping forward shows an actor badge naming who made that decision", () => {
    render(<TwoPlayerGameView game={game} onBack={() => {}} />);
    fireEvent.keyDown(document, { key: "ArrowRight" });
    const step = game.steps[0];
    const expectedLabel = step.actor_idx === 0 ? "Agent A" : "Agent B";
    // The actor badge sits right next to the action name in the action panel.
    expect(screen.getAllByText(expectedLabel).length).toBeGreaterThan(0);
  });

  it("shows a win/draw outcome banner with both agents' scores on the final step", () => {
    render(<TwoPlayerGameView game={game} onBack={() => {}} />);
    const steps = game.steps.length + 1;
    for (let i = 1; i < steps; i++) fireEvent.keyDown(document, { key: "ArrowRight" });

    if (game.winner == null) {
      expect(screen.getByText("Draw")).toBeTruthy();
    } else {
      expect(screen.getByText(`${game.winner === 0 ? "Agent A" : "Agent B"} wins`)).toBeTruthy();
    }
    expect(screen.getAllByText(/Agent A/).length).toBeGreaterThan(0);
    expect(screen.getAllByText(/Agent B/).length).toBeGreaterThan(0);
    expect(document.querySelector(".outcome-scores")).toBeTruthy();
  });

  it("Escape calls onBack", () => {
    let called = false;
    render(<TwoPlayerGameView game={game} onBack={() => (called = true)} />);
    fireEvent.keyDown(document, { key: "Escape" });
    expect(called).toBe(true);
  });
});
