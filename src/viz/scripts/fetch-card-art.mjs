// One-time prep step: fetch each of the 22 known cards' art from Scryfall
// and cache it locally under public/card_art/ (gitignored -- regenerate
// by rerunning this script, never committed). The app never needs
// internet access to *view* a game; only this script talks to Scryfall.
//
// Usage: node scripts/fetch-card-art.mjs

import { mkdir, writeFile, stat } from "node:fs/promises";
import path from "node:path";
import { fileURLToPath } from "node:url";
import { CARD_NAMES } from "../src/cardData.js";
import { slug } from "../src/slug.js";

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const OUT_DIR = path.join(__dirname, "..", "public", "card_art");
const DELAY_MS = 150; // stay well under Scryfall's rate-limit guidance

async function exists(p) {
  try {
    await stat(p);
    return true;
  } catch {
    return false;
  }
}

function sleep(ms) {
  return new Promise((resolve) => setTimeout(resolve, ms));
}

// Scryfall rejects requests with a default/generic User-Agent (returns
// HTTP 400, subcode "generic_user_agent") -- their API docs require a
// custom one identifying the application.
const HEADERS = {
  "User-Agent": "azul-modeling-tron-viz/1.0 (local hobby project card art cache)",
  Accept: "application/json",
};

async function fetchOne(name) {
  const url = `https://api.scryfall.com/cards/named?fuzzy=${encodeURIComponent(name)}`;
  const res = await fetch(url, { headers: HEADERS });
  if (!res.ok) {
    throw new Error(`Scryfall lookup failed for ${JSON.stringify(name)}: HTTP ${res.status}`);
  }
  const card = await res.json();
  const imageUrl =
    card.image_uris?.png ?? card.card_faces?.[0]?.image_uris?.png;
  if (!imageUrl) {
    throw new Error(`No image_uris found for ${JSON.stringify(name)} (card: ${card.name})`);
  }

  const imgRes = await fetch(imageUrl, { headers: HEADERS });
  if (!imgRes.ok) {
    throw new Error(`Image download failed for ${JSON.stringify(name)}: HTTP ${imgRes.status}`);
  }
  const buf = Buffer.from(await imgRes.arrayBuffer());
  const outPath = path.join(OUT_DIR, `${slug(name)}.png`);
  await writeFile(outPath, buf);
  return { name, outPath, bytes: buf.length };
}

async function main() {
  await mkdir(OUT_DIR, { recursive: true });
  console.log(`Fetching art for ${CARD_NAMES.length} cards into ${OUT_DIR}...`);

  const failures = [];
  for (const name of CARD_NAMES) {
    const outPath = path.join(OUT_DIR, `${slug(name)}.png`);
    if (await exists(outPath)) {
      console.log(`  skip ${name} -> ${path.basename(outPath)} (already cached)`);
      continue;
    }
    try {
      const { bytes } = await fetchOne(name);
      console.log(`  ok   ${name} -> ${path.basename(outPath)} (${bytes} bytes)`);
    } catch (err) {
      console.error(`  FAIL ${name}: ${err.message}`);
      failures.push(name);
    }
    await sleep(DELAY_MS);
  }

  if (failures.length) {
    console.error(`\n${failures.length} card(s) failed: ${failures.join(", ")}`);
    process.exit(1);
  }
  console.log("\nAll card art fetched successfully.");
}

main();
