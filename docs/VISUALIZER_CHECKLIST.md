# Game Visualizer — Implementation Checklist (v2: React rebuild)

Companion to [VISUALIZER_PLAN.md](VISUALIZER_PLAN.md) (v2). Concrete
build order, phase by phase, each ending in a runnable check before
moving on. Planning only; nothing gets built until you say go. Supersedes
the v1 checklist — `src/viz/visualize.py` and `viewer_template.html` get
removed as part of Phase R7, not kept alongside the React app.

## Phase R0 — Naming convention for card art files

Both the static `cardData.js` lookup and the `fetch-card-art.mjs` prep
script need the *same* filename slug per card, or the app will fail to
load images silently. One function, used by both:

```js
const slug = (name) => name.toLowerCase().replace(/'/g, "")
  .replace(/[^a-z0-9]+/g, "-").replace(/(^-|-$)/g, "");
// "Urza's Mine" -> "urzas-mine", "Pinnacle Kill-Ship" -> "pinnacle-kill-ship"
```

- [ ] Put this in one shared module (`src/viz/src/slug.js`) imported by
      both the prep script and `cardData.js`, not duplicated.

## Phase R1 — Project scaffolding

- [ ] Scaffold a React + Vite project in `src/viz/` (`npm create vite@latest`
      equivalent: `package.json`, `vite.config.js`, `index.html`,
      `src/main.jsx`, `src/App.jsx`).
- [ ] Add `src/viz/node_modules/`, `src/viz/dist/`, and
      `src/viz/public/card_art/` to `.gitignore` (project root's existing
      `.gitignore`, extending its "generated artifacts" section).
- [ ] Add Vitest (Vite's own first-class test runner) for the pure-logic
      unit checks in later phases — no separate config needed beyond
      what `vite` already provides.
- [ ] Sanity check: `npm install && npm run dev` starts without error and
      serves a placeholder page.

## Phase R2 — Card art + static card data

- [ ] `src/viz/src/cardData.js`: hand-authored from `game.py`'s
      `DECKLIST`/`CardType` (source of truth) — all 22 names mapped to
      `{ type, sortPriority, artFile: slug(name) + ".jpg" }`.
      `sortPriority`: `0` for every `LAND` effect, `1` for `ARTIFACT`,
      `2` for `CREATURE` (only `Generous Ent` — flagged in the plan as
      always-empty on the battlefield in practice, but correct to
      include), `3` for `SORCERY`/`INSTANT` (never on the battlefield at
      all, included for completeness/no-crash if ever rendered), `4` for
      `FILLER`.
- [ ] `src/viz/scripts/fetch-card-art.mjs`: for each of the 22 names, hit
      Scryfall's `cards/named?fuzzy=<name>` endpoint, take
      `image_uris.normal` (or `card_faces[0].image_uris.normal` for any
      double-faced entries, though none exist in this pool), download to
      `public/card_art/<slug>.jpg`. A small delay between requests
      (Scryfall's guidance is to stay well under their rate limit, e.g.
      ~100ms between calls) — 22 requests total, this finishes in a few
      seconds regardless.
- [ ] Sanity check: run the script, confirm all 22 files exist in
      `public/card_art/`, each with a plausible size (a few KB minimum —
      catches silently-saved error pages, which would be tiny HTML, not
      a real several-KB+ JPEG).

## Phase R3 — Data loading

- [ ] `src/viz/src/loadLog.js`: pure function `parseLog(text) -> games[]`
      — `JSON.parse`, validate it's a non-empty array, validate the first
      element has the expected keys (`game_index`, `score`,
      `opening_hand`, `turns`, `end_state`), throw a clear `Error` naming
      what's wrong otherwise (ported from `visualize.py`'s validation,
      same spirit).
- [ ] `src/viz/src/components/DropZone.jsx`: drag-and-drop target +
      "browse for a file" `<input type="file">` fallback, reads the
      dropped/selected file via the File API, calls `parseLog`, lifts
      the result into `App`'s state (or shows the thrown error message).
- [ ] Sanity check (Vitest, no browser needed): `parseLog` against a
      real fixture (a small real log, e.g. copy of
      `logs/viz_demo_25.json`) succeeds; against malformed inputs
      (invalid JSON, non-array, missing keys) throws with a message
      naming the problem.

## Phase R4 — Card rendering

- [ ] `src/viz/src/components/Card.jsx`: given `{ name, tapped }`, looks
      up `cardData[name]`, renders `<img src="/card_art/{artFile}">`
      with an `onError` fallback to a plain text label (never a broken-
      image icon) if the art is missing. `tapped` applies
      `transform: rotate(90deg)` via CSS, plus enough surrounding margin
      that rotated cards don't visually overlap their neighbors.
- [ ] Sanity check: Vitest render test (via `@testing-library/react` or
      Vite's built-in test utilities) confirming a known card name
      resolves to its expected `artFile` and that `tapped=true` applies
      the rotation class/style — structural correctness, not a visual
      check (see Phase R7's flagged limitation).

## Phase R5 — Game view

- [ ] `src/viz/src/flatten.js`: pure function `flattenSteps(game) ->
      steps[]` — port of the old vanilla-JS flattening logic (synthetic
      "Opening hand" step 0, then every turn's actions in order,
      draw pseudo-actions included since they're already in the log).
- [ ] `src/viz/src/components/GameView.jsx`: renders the current step's
      `state_after` — hand (`Card` list), battlefield (`Card` list
      **sorted by `cardData[name].sortPriority`**, ties broken by
      original order), graveyard (`Card` list) — plus the action detail
      panel (`fetched`/`left_battlefield`/`tapped_for_cost`/`scry`, only
      the fields present on that step), turn number, live
      resource-quality readout (already in the log's `state_after`, no
      recomputation needed), and the final-step outcome panel
      (score + success/failure + `turn_assembled`/`turn_online`).
- [ ] Keyboard handling: `ArrowRight`/`ArrowDown` = next step,
      `ArrowLeft`/`ArrowUp` = previous (clamped, no wraparound),
      `Escape` = back to the batch list.
- [ ] Sanity check: Vitest unit test on `flattenSteps` against a real
      fixture game — correct step count, correct ordering, draw
      pseudo-actions in the right place (same assertions the Python
      `_phase_visualizer_logging_sanity_check` already makes on the
      source data, just confirming the JS-side flattening preserves
      them).

## Phase R6 — Batch list, search filters, pagination

- [ ] `src/viz/src/filterGames.js`: pure function `filterGames(games,
      criteria) -> games[]` — `criteria = { turnOnlineMin, turnOnlineMax,
      cardInHand, scoreMin, scoreMax }`, all optional, AND-combined.
      `cardInHand` checks `opening_hand` plus every step's
      `state_after.hand` (via `flattenSteps`) for the given name.
- [ ] `src/viz/src/components/SearchFilters.jsx`: turn-online min/max
      number inputs, a `<select>` of the 22 known card names (plus a
      "any" default) for card-in-hand, score min/max number inputs —
      updates filter criteria in `App` state on change.
- [ ] `src/viz/src/components/Pagination.jsx`: given the filtered set and
      a page size (default 25, adjustable constant), renders page-number
      controls, capped at ~20 visible page buttons before truncating
      (e.g. `1 2 3 … 19 20`), tracks current page in `App` state.
- [ ] `src/viz/src/components/BatchList.jsx`: renders the current page's
      rows (runtime label, `game_index`, ending turn, score), click → 
      `GameView`.
- [ ] Sanity check: Vitest unit tests on `filterGames` — a hand-built
      small fixture set, assert AND-intersection behaves correctly (e.g.
      a `turn_online` filter alone matches more games than the same
      filter combined with a `cardInHand` filter that only some of those
      games satisfy).

## Phase R7 — Cleanup and end-to-end verification

- [ ] Remove `src/viz/visualize.py` and `src/viz/viewer_template.html`
      (superseded).
- [ ] Update `README.md`'s viz-related instructions (currently, if any,
      pointing at the old `python src/viz/visualize.py` command) to the
      new `cd src/viz && npm install && npm run dev` flow, plus the
      one-time `node scripts/fetch-card-art.mjs` prep step.
- [ ] Run every Vitest suite from Phases R3-R6 together, confirm all
      pass.
- [ ] Load a real log (e.g. regenerate `logs/viz_demo_25.json` or a
      fresh small batch) via drag-and-drop in the running dev server,
      manually confirm: card art renders, battlefield sorting matches
      lands-then-artifacts-then-creatures, tapped cards are visually
      rotated, search filters narrow the batch list correctly and
      combine with AND, pagination controls work, arrow-key step-through
      still works end to end including the final outcome panel.
- [ ] **Known limitation, same as v1**: no browser automation available
      in this environment — the checklist can confirm every pure-logic
      piece via Vitest and that the dev server starts without error, but
      the actual visual result (art layout, rotation, sort order as
      *seen*) needs to be confirmed by you, not verified here.
