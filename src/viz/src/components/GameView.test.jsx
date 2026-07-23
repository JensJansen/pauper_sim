import { describe, it, expect } from "vitest";
import { render, screen, fireEvent } from "@testing-library/react";
import GameView, { sortedBattlefield, computeBattlefieldLinks } from "./GameView.jsx";
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

describe("computeBattlefieldLinks", () => {
  // A real battlefield snapshot pulled from an actual boggles_mirror
  // --log run (harness._snapshot_player_state): two Forests (ambiguous --
  // needs a slot badge), one Slippery Bogle enchanted by one Rancor
  // (unambiguous names, but the enchant link still needs to resolve).
  const battlefield = [
    { name: "Forest", tapped: true, slot: 1, enchanting: null },
    { name: "Slippery Bogle", tapped: false, slot: 1, enchanting: null },
    { name: "Rancor", tapped: false, slot: 1, enchanting: "Slippery Bogle (slot 1)" },
    { name: "Forest", tapped: false, slot: 2, enchanting: null },
  ];

  it("counts same-named permanents so the viewer knows when a slot badge is needed", () => {
    const { nameCounts } = computeBattlefieldLinks(battlefield);
    expect(nameCounts["Forest"]).toBe(2); // ambiguous -- ends up with a slot badge
    expect(nameCounts["Slippery Bogle"]).toBe(1); // not ambiguous -- no badge needed
    expect(nameCounts["Rancor"]).toBe(1);
  });

  it("maps a target's own (name, slot) label to the names of whatever enchants it", () => {
    const { enchantedBy } = computeBattlefieldLinks(battlefield);
    expect(enchantedBy["Slippery Bogle (slot 1)"]).toEqual(["Rancor"]);
    expect(enchantedBy["Forest (slot 1)"]).toBeUndefined(); // nothing enchants either Forest here
  });

  it("gracefully no-ops on a 1-player battlefield entry (no slot/enchanting fields at all)", () => {
    const oneLayerBattlefield = [{ name: "Urza's Mine", tapped: false }];
    const { nameCounts, enchantedBy } = computeBattlefieldLinks(oneLayerBattlefield);
    expect(nameCounts["Urza's Mine"]).toBe(1);
    expect(enchantedBy).toEqual({});
  });
});

describe("exile zone and opponent life", () => {
  function makeGame({ exile = [], damageDealt = 0 }) {
    const stateAfter = {
      turn_number: 1,
      hand: [],
      battlefield: [],
      graveyard: [],
      exile,
      mana_pool: {},
      damage_dealt: damageDealt,
      resource_quality: { non_land_permanents: 0, available_mana: 0, hand_size: 0 },
    };
    return {
      game_index: 0,
      scores: { reward_fn: 1 },
      turn_won: null,
      opening_hand_state: stateAfter,
      steps: [],
      end_state: stateAfter,
    };
  }

  it("shows exiled cards when present", () => {
    render(<GameView game={makeGame({ exile: ["Fiery Temper"] })} onBack={() => {}} />);
    expect(screen.getByText("Exile")).toBeTruthy();
    expect(screen.getByAltText("Fiery Temper")).toBeTruthy();
  });

  it("shows the exile zone as empty when nothing's exiled", () => {
    render(<GameView game={makeGame({})} onBack={() => {}} />);
    expect(screen.getByText("Exile")).toBeTruthy();
    expect(screen.queryByAltText("Fiery Temper")).toBeNull();
  });

  it("shows opponent life (win_threshold - damage_dealt) when meta.win_threshold is set", () => {
    render(
      <GameView game={makeGame({ damageDealt: 14 })} meta={{ win_threshold: 20 }} onBack={() => {}} />
    );
    expect(screen.getByText("Opponent life: 6")).toBeTruthy();
  });

  it("floors opponent life at 0 rather than going negative", () => {
    render(
      <GameView game={makeGame({ damageDealt: 27 })} meta={{ win_threshold: 20 }} onBack={() => {}} />
    );
    expect(screen.getByText("Opponent life: 0")).toBeTruthy();
  });

  it("hides opponent life entirely for a deck with no win_threshold (e.g. Tron)", () => {
    render(<GameView game={makeGame({ damageDealt: 0 })} onBack={() => {}} />);
    expect(screen.queryByText(/Opponent life/)).toBeNull();
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
