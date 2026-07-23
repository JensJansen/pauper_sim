// One-time prep step: look up each known card's art URL in Scryfall's
// bulk "unique artwork" dump (public/card_art/*.jsonl, one JSON card per
// line -- see https://scryfall.com/docs/api/bulk-data) and cache the
// image locally under public/card_art/ (gitignored -- regenerate by
// rerunning this script, never committed). The app never needs internet
// access to *view* a game; only this script talks to the network, and
// only to fetch image bytes (the card lookup itself is local).
//
// Usage: node scripts/fetch-card-art.mjs

import { mkdir, writeFile, stat, readdir } from "node:fs/promises";
import { createReadStream } from "node:fs";
import { createInterface } from "node:readline";
import path from "node:path";
import { fileURLToPath } from "node:url";
import { CARD_NAMES } from "../src/cardData.js";
import { slug } from "../src/slug.js";

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const OUT_DIR = path.join(__dirname, "..", "public", "card_art");
const DELAY_MS = 150; // stay well under Scryfall's rate-limit guidance

const HEADERS = {
  "User-Agent": "azul-modeling-tron-viz/1.0 (local hobby project card art cache)",
};

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

// Builds name -> image URL from the newest *.jsonl bulk-data file present
// in OUT_DIR. First entry per name wins; that's fine, we just need *an*
// image for each card, not a specific printing.
async function loadArtIndex() {
  const files = (await readdir(OUT_DIR)).filter((f) => f.endsWith(".jsonl"));
  if (files.length === 0) {
    throw new Error(
      `No *.jsonl bulk-data file found in ${OUT_DIR}. Download Scryfall's ` +
        `"Unique Artwork" bulk file (https://scryfall.com/docs/api/bulk-data) and drop it there.`
    );
  }
  files.sort();
  const jsonlPath = path.join(OUT_DIR, files.at(-1));

  const index = new Map();
  const rl = createInterface({ input: createReadStream(jsonlPath), crlfDelay: Infinity });
  for await (const line of rl) {
    if (!line) continue;
    const card = JSON.parse(line);
    if (index.has(card.name)) continue;
    const imageUrl = card.image_uris?.png ?? card.card_faces?.[0]?.image_uris?.png;
    if (imageUrl) index.set(card.name, imageUrl);
  }
  return index;
}

async function fetchOne(name, imageUrl) {
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
  const artIndex = await loadArtIndex();
  console.log(`Fetching art for ${CARD_NAMES.length} cards into ${OUT_DIR}...`);

  const failures = [];
  for (const name of CARD_NAMES) {
    const outPath = path.join(OUT_DIR, `${slug(name)}.png`);
    if (await exists(outPath)) {
      console.log(`  skip ${name} -> ${path.basename(outPath)} (already cached)`);
      continue;
    }
    const imageUrl = artIndex.get(name);
    if (!imageUrl) {
      console.error(`  FAIL ${name}: not found in bulk-data index`);
      failures.push(name);
      continue;
    }
    try {
      const { bytes } = await fetchOne(name, imageUrl);
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
