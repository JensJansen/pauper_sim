import { describe, it, expect } from "vitest";
import { filterGames } from "./filterGames.js";

function makeGame({ gameIndex, turnWon, score1, handEver, playEver = [] }) {
  const steps = playEver.length
    ? [
        {
          turn: 1,
          action: "Play land",
          fetched: [],
          left_battlefield: [],
          tapped_for_cost: [],
          decision: null,
          fallback: false,
          state_after: {
            turn_number: 1,
            hand: [],
            battlefield: playEver.map((name) => ({ name, tapped: false })),
            graveyard: [],
            resource_quality: { non_land_permanents: 0, available_mana: 0, hand_size: 0 },
          },
        },
      ]
    : [];
  return {
    game_index: gameIndex,
    scores: { reward_fn: score1 },
    turn_won: turnWon,
    opening_hand_state: { turn_number: 1, hand: handEver, battlefield: [], graveyard: [] },
    steps,
    end_state: { turn_number: 1, hand: [], battlefield: [], graveyard: [] },
  };
}

const games = [
  makeGame({ gameIndex: 0, turnWon: 3, score1: 0.6, handEver: ["Forest", "Urza's Mine"], playEver: ["Urza's Mine"] }),
  makeGame({ gameIndex: 1, turnWon: 3, score1: 0.5, handEver: ["Bramble Wurm"] }),
  makeGame({ gameIndex: 2, turnWon: 5, score1: 0.4, handEver: ["Forest"] }),
  makeGame({ gameIndex: 3, turnWon: null, score1: 0.0, handEver: ["Forest"] }),
];

describe("filterGames", () => {
  it("with no criteria, returns everything", () => {
    expect(filterGames(games, {})).toHaveLength(4);
  });

  it("turn_won range alone", () => {
    const result = filterGames(games, { turnWonMin: 3, turnWonMax: 3 });
    expect(result.map((g) => g.game_index)).toEqual([0, 1]);
  });

  it("score range alone (the primary/first named score's raw scale)", () => {
    const result = filterGames(games, { scoreMin: 0.5 });
    expect(result.map((g) => g.game_index)).toEqual([0, 1]);
  });

  it("card-in-hand alone matches every game that ever had it, including a failure", () => {
    const result = filterGames(games, { cardInHand: "Forest" });
    expect(result.map((g) => g.game_index)).toEqual([0, 2, 3]);
  });

  it("combining turn_won range + card-in-hand is an intersection (AND), not a union", () => {
    // turn_won in [3,3] alone matches {0, 1}; cardInHand=Forest alone matches {0, 2, 3}.
    // The AND-combination must be exactly their intersection: {0}.
    const result = filterGames(games, { turnWonMin: 3, turnWonMax: 3, cardInHand: "Forest" });
    expect(result.map((g) => g.game_index)).toEqual([0]);
  });

  it("card-in-play alone matches only games where the card was ever on the battlefield", () => {
    const result = filterGames(games, { cardInPlay: "Urza's Mine" });
    expect(result.map((g) => g.game_index)).toEqual([0]);
  });

  it("card-in-play does not match a card that was only ever in hand", () => {
    const result = filterGames(games, { cardInPlay: "Forest" });
    expect(result).toEqual([]);
  });

  it("a filter combination matching nothing returns an empty array, not an error", () => {
    const result = filterGames(games, { turnWonMin: 3, turnWonMax: 3, cardInHand: "Bramble Wurm", scoreMin: 0.9 });
    expect(result).toEqual([]);
  });
});

describe("filterGames -- 2-player games", () => {
  function makeTwoPlayerGame({ gameIndex, turnWon, scoreA, scoreB, seat0Hand = [], seat1Battlefield = [] }) {
    const openingState = {
      players: [
        { hand: seat0Hand, battlefield: [], graveyard: [], exile: [], mana_pool: {}, life_total: 20 },
        { hand: [], battlefield: seat1Battlefield, graveyard: [], exile: [], mana_pool: {}, life_total: 20 },
      ],
      stack: [],
    };
    return {
      game_index: gameIndex,
      starting_player_idx: 0,
      winner: null,
      turn_won: turnWon,
      final_turn_number: turnWon ?? 10,
      scores: [{ reward_fn: scoreA }, { reward_fn: scoreB }],
      opening_state: openingState,
      steps: [],
      end_state: openingState,
    };
  }

  const twoPlayerGames = [
    makeTwoPlayerGame({ gameIndex: 0, turnWon: 3, scoreA: 0.6, scoreB: 0.1, seat0Hand: ["Forest"] }),
    makeTwoPlayerGame({ gameIndex: 1, turnWon: 5, scoreA: 0.2, scoreB: 0.9, seat1Battlefield: [{ name: "Urza's Mine", tapped: false }] }),
  ];

  it("primary score filters off seat 0 (agent_a), not seat 1", () => {
    const result = filterGames(twoPlayerGames, { scoreMin: 0.5 });
    expect(result.map((g) => g.game_index)).toEqual([0]); // seat 1's higher score (game 1) must not qualify it
  });

  it("card-in-hand/card-in-play check EITHER seat's own zone, not just seat 0's", () => {
    expect(filterGames(twoPlayerGames, { cardInHand: "Forest" }).map((g) => g.game_index)).toEqual([0]);
    expect(filterGames(twoPlayerGames, { cardInPlay: "Urza's Mine" }).map((g) => g.game_index)).toEqual([1]);
  });

  it("turn_won range works the same as the 1-player shape", () => {
    expect(filterGames(twoPlayerGames, { turnWonMin: 4 }).map((g) => g.game_index)).toEqual([1]);
  });
});
