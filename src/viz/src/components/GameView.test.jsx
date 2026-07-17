import { describe, it, expect } from "vitest";
import { sortedBattlefield } from "./GameView.jsx";

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
