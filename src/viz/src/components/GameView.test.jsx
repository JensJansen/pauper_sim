import { describe, it, expect } from "vitest";
import { render, screen, fireEvent } from "@testing-library/react";
import GameView, { sortedBattlefield } from "./GameView.jsx";
import fixture from "../__fixtures__/sampleGames.json";

describe("sortedBattlefield", () => {
  it("orders lands before artifacts before creatures", () => {
    const battlefield = [
      { name: "Bonder's Ornament", tapped: false }, // artifact
      { name: "Generous Ent", tapped: false }, // creature (never actually appears in practice, but must sort correctly if it did)
      { name: "Urza's Mine", tapped: false }, // land
      { name: "Expedition Map", tapped: false }, // artifact
      { name: "Forest", tapped: true }, // land
    ];
    const sorted = sortedBattlefield(battlefield).map((p) => p.name);
    expect(sorted).toEqual([
      "Urza's Mine",
      "Forest",
      "Bonder's Ornament",
      "Expedition Map",
      "Generous Ent",
    ]);
  });

  it("preserves original order among ties (stable sort)", () => {
    const battlefield = [
      { name: "Urza's Tower", tapped: false },
      { name: "Urza's Mine", tapped: false },
      { name: "Forest", tapped: false },
    ];
    const sorted = sortedBattlefield(battlefield).map((p) => p.name);
    expect(sorted).toEqual(["Urza's Tower", "Urza's Mine", "Forest"]);
  });

  it("does not mutate the input array", () => {
    const battlefield = [{ name: "Bonder's Ornament" }, { name: "Forest" }];
    const original = [...battlefield];
    sortedBattlefield(battlefield);
    expect(battlefield).toEqual(original);
  });
});

describe("floating mana readout", () => {
  it("shows the floating pool once mana is tapped, and clears once it's spent", () => {
    // fixture.games[0] -- step 5 taps Urza's Power Plant for {generic: 1}
    // against Expedition Map's cost, floating 1 colorless; step 6 spends it
    // (see harness.py's per-action logging: every tap/spend is its own step).
    const [game] = fixture.games;
    render(<GameView game={game} onBack={() => {}} />);

    for (let i = 0; i < 5; i++) fireEvent.keyDown(document, { key: "ArrowRight" });
    expect(screen.getByText("C:1")).toBeTruthy();

    fireEvent.keyDown(document, { key: "ArrowRight" });
    expect(screen.getByText("none")).toBeTruthy();
  });
});
