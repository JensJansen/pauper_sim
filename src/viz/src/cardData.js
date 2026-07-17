// Static reference data for the 22 distinct cards in this project's deck
// (see data/deck_list.txt / game.py's DECKLIST, the source of truth).
// Fixed metadata -- a card's type never depends on game state, so this
// never needs to track anything at runtime, just kept in sync by hand if
// the decklist itself ever changes.
import { slug } from "./slug.js";

const SORT_PRIORITY = {
  LAND: 0,
  ARTIFACT: 1,
  CREATURE: 2,
  SORCERY: 3,
  INSTANT: 3,
  FILLER: 4,
};

// [name, CardType] pairs, exactly matching game.py's DECKLIST order/values.
const RAW = [
  ["Urza's Mine", "LAND"],
  ["Urza's Power Plant", "LAND"],
  ["Urza's Tower", "LAND"],
  ["Forest", "LAND"],
  ["Wooded Ridgeline", "LAND"],
  ["Bojuka Bog", "LAND"],
  ["Tocasia's Dig Site", "LAND"],
  ["Conduit Pylons", "LAND"],
  ["Expedition Map", "ARTIFACT"],
  ["Crop Rotation", "INSTANT"],
  ["Ancient Stirrings", "SORCERY"],
  ["Bonder's Ornament", "ARTIFACT"],
  ["Candy Trail", "ARTIFACT"],
  ["Barrels of Blasting Jelly", "ARTIFACT"],
  ["Relic of Progenitus", "ARTIFACT"],
  ["Generous Ent", "CREATURE"], // never actually cast in this deck (forestcycled only), but correct to include
  ["Rooftop Percher", "FILLER"],
  ["Boulderbranch Golem", "FILLER"],
  ["Maelstrom Colossus", "FILLER"],
  ["Bramble Wurm", "FILLER"],
  ["Pinnacle Kill-Ship", "FILLER"],
  ["Breath Weapon", "FILLER"],
];

export const CARD_DATA = Object.fromEntries(
  RAW.map(([name, type]) => [
    name,
    { type, sortPriority: SORT_PRIORITY[type], artFile: `${slug(name)}.png` },
  ])
);

export const CARD_NAMES = RAW.map(([name]) => name);

export function cardInfo(name) {
  return CARD_DATA[name] || { type: "FILLER", sortPriority: 4, artFile: `${slug(name)}.png` };
}
