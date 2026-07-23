// Static reference data for every distinct card across every deck this
// project plays (data/*.txt -- monster_tron, spy_combo, rakdos_madness,
// mono_red_madness -- plus the two synthetic token CardDefs created
// mid-game, game.BLOOD_TOKEN_CARD_DEF/ROBOT_TOKEN_CARD_DEF, which never
// appear in a decklist file but do show up on a logged battlefield).
// Fixed metadata -- a card's type never depends on game state, so this
// never needs to track anything at runtime, just kept in sync by hand
// whenever a deck adds a card not already listed here (cross-check
// against game.CARD_DEFS if unsure: `python -c "import game;
// print(sorted(game.CARD_DEFS))"` from src/).
import { slug } from "./slug.js";

const SORT_PRIORITY = {
  LAND: 0,
  ARTIFACT: 1,
  ENCHANTMENT: 2,
  CREATURE: 3,
  SORCERY: 4,
  INSTANT: 4,
  FILLER: 5,
};

// [name, CardType] pairs, alphabetical -- spans every deck, not just one.
const RAW = [
  ["Abundant Growth", "ENCHANTMENT"],
  ["Alms of the Vein", "SORCERY"],
  ["Ancestral Mask", "ENCHANTMENT"],
  ["Ancient Stirrings", "SORCERY"],
  ["Armadillo Cloak", "ENCHANTMENT"],
  ["Ash Barrens", "LAND"],
  ["Balustrade Spy", "CREATURE"],
  ["Barrels of Blasting Jelly", "ARTIFACT"],
  ["Blood", "ARTIFACT"], // token (game.effects.tokens.BLOOD_TOKEN_CARD_DEF)
  ["Bojuka Bog", "LAND"],
  ["Bonder's Ornament", "ARTIFACT"],
  ["Boulderbranch Golem", "CREATURE"],
  ["Bramble Wurm", "CREATURE"],
  ["Breath Weapon", "INSTANT"],
  ["Candy Trail", "ARTIFACT"],
  ["Cartouche of Solidarity", "ENCHANTMENT"],
  ["Conduit Pylons", "LAND"],
  ["Crop Rotation", "INSTANT"],
  ["Dread Return", "SORCERY"],
  ["Eldrazi Spawn", "CREATURE"], // token (game.effects.tokens.ELDRAZI_SPAWN_TOKEN_CARD_DEF)
  ["End the Festivities", "SORCERY"],
  ["Ethereal Armor", "ENCHANTMENT"],
  ["Expedition Map", "ARTIFACT"],
  ["Faithless Looting", "SORCERY"],
  ["Fiery Temper", "INSTANT"],
  ["Fireblast", "INSTANT"],
  ["Forest", "LAND"],
  ["Gatecreeper Vine", "CREATURE"],
  ["Generous Ent", "CREATURE"],
  ["Gladecover Scout", "CREATURE"],
  ["Grab the Prize", "SORCERY"],
  ["Guttersnipe", "CREATURE"],
  ["Highway Robbery", "SORCERY"],
  ["Jagged Barrens", "LAND"],
  ["Kitchen Imp", "CREATURE"],
  ["Land Grant", "SORCERY"],
  ["Lava Dart", "INSTANT"],
  ["Lead the Stampede", "SORCERY"],
  ["Lightning Bolt", "INSTANT"],
  ["Lotleth Giant", "CREATURE"],
  ["Lotus Petal", "ARTIFACT"],
  ["Maelstrom Colossus", "CREATURE"],
  ["Malevolent Rumble", "SORCERY"],
  ["Masked Vandal", "CREATURE"],
  ["Melded Moxite", "ARTIFACT"],
  ["Mesmeric Fiend", "CREATURE"],
  ["Mountain", "LAND"],
  ["Nyxborn Hydra", "CREATURE"],
  ["Overgrown Battlement", "CREATURE"],
  ["Pinnacle Kill-Ship", "ARTIFACT"],
  ["Plains", "LAND"],
  ["Quirion Ranger", "CREATURE"],
  ["Rakdos Carnarium", "LAND"],
  ["Ram Through", "INSTANT"],
  ["Rancor", "ENCHANTMENT"],
  ["Relic of Progenitus", "ARTIFACT"],
  ["Robot", "CREATURE"], // token (game.effects.tokens.ROBOT_TOKEN_CARD_DEF)
  ["Rooftop Percher", "CREATURE"],
  ["Sagu Wildling", "SORCERY"],
  ["Saruli Caretaker", "CREATURE"],
  ["Silhana Ledgewalker", "CREATURE"],
  ["Slippery Bogle", "CREATURE"],
  ["Sneaky Snacker", "CREATURE"],
  ["Swamp", "LAND"],
  ["Tocasia's Dig Site", "LAND"],
  ["Urza's Mine", "LAND"],
  ["Urza's Power Plant", "LAND"],
  ["Urza's Tower", "LAND"],
  ["Utopia Sprawl", "ENCHANTMENT"],
  ["Vampire's Kiss", "SORCERY"],
  ["Voldaren Epicure", "CREATURE"],
  ["Wall of Roots", "CREATURE"],
  ["Warrior", "CREATURE"], // token (game.effects.tokens.WARRIOR_TOKEN_CARD_DEF)
  ["Winding Way", "SORCERY"],
  ["Wooded Ridgeline", "LAND"],
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
