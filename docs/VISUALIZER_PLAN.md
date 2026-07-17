# Game Visualizer — Implementation Plan (v2: React rebuild)

Companion to [DRL_PLAN.md](DRL_PLAN.md) and the logging feature already
built into `harness.py`'s `_GameLogger`. **Supersedes v1 of this plan**,
which shipped as a static HTML/vanilla-JS tool
(`src/viz/visualize.py` + `viewer_template.html`) — working, but visually
minimal (text chips, no card art) and not built for further growth. This
revision replaces that with a proper React app per your explicit go-ahead
to "scrap it and move into node/js." Planning only — nothing gets built
until this is confirmed.

## Goal (updated)

A persistent, reactive web app — not a regenerated static file — that you
load a JSON log into via drag-and-drop, and that renders games the way an
actual game of Tron looks on a table: real card art, cards sorted by type
within each zone, tapped cards visually rotated, arrow-key step-through
per game, plus a page-based browser over the whole batch with AND-combined
search filters (turn online, card in hand, score).

## Decisions locked in (still true from v1)

1. **No hidden information, ever** — never show library contents/order,
   matching what the trained model's own observation space actually sees.
2. **Arrow-key navigation**, one action at a time, forward and back.
3. **v1 shows only the chosen action**, not alternatives/model confidence
   — still a separately-scoped follow-on, not part of this rebuild.
4. **Live resource-quality readout** throughout a playthrough, not just
   at the end (already precomputed server-side in the log, per v1's
   finding that this avoids client/server logic drift).
5. **State captured directly during `evaluate()`**, never reconstructed
   after the fact — `_GameLogger`'s `state_after`/`left_battlefield`/
   `tapped_for_cost` snapshots from v1 stay exactly as they are; nothing
   about the log format needs to change for this rebuild except one
   addition (see "Card type" below).

## New decisions (this conversation)

1. **Framework: React + Vite.** Chosen over Svelte for the larger
   ecosystem and better odds of familiar tooling for whoever touches this
   next (including future me).
2. **Card art: fetched once and cached locally, gitignored.** Our card
   pool is fixed at 22 distinct names — a one-time prep script pulls each
   from Scryfall and saves it under `src/viz/public/card_art/` (or
   similar), the app reads local files at runtime (no network needed to
   *view* a game), and the cached images are never committed (regenerable
   from the prep script, same philosophy as `models/`/`logs/` already
   being gitignored).
3. **Data loading: in-app drag-and-drop / file picker.** One running app
   (`npm run dev`), no CLI step to view a specific file — drop a JSON log
   onto the page, or click to browse for one. Swapping datasets means
   dropping a different file, not restarting anything.
4. **Battlefield sort order**: lands first, then artifacts, then
   creatures (real MTG table convention — closest-to-viewer to
   furthest). Card type is fixed, static metadata (never depends on game
   state), so this is a **small static lookup table shipped in the
   frontend**, not a backend log change — avoids regenerating existing
   logs and avoids the client/server-drift risk that made resource_quality
   worth precomputing server-side (that risk doesn't apply here, since a
   card's type never varies).
   - Worth knowing going in: nothing that's actually a creature (Generous
     Ent) ever reaches the battlefield in this deck — it's only ever
     forestcycled from hand, never cast. The "creatures" tier is correct
     convention but will always render empty against this dataset.
5. **Tapped cards render rotated** (`transform: rotate(90deg)`,
   the standard tabletop convention), not just dimmed/struck-through.
6. **Pagination, not infinite scroll/virtualization.** A page-based
   browser (page-number controls, up to ~20 page buttons before
   truncating) over the *filtered* result set. Deferred: true
   virtualization for datasets large enough that pagination itself
   becomes unwieldy — not a near-term need.
7. **Search/filter, AND-combined**: three filters, all optional, all
   active filters intersect (not union) —
   - **Turn tron was online**: min/max range over `turn_online` (an
     exact-turn search is just min = max).
   - **Card in hand**: a dropdown of the 22 known card names (not
     free-text), matching a game where that name appears in
     `opening_hand` **or** any `state_after.hand` across the whole game
     — "was this ever in hand," not just at one specific point.
   - **Score**: min/max range over the game's `score`.

## Card type lookup (new static frontend data, not a log format change)

One small JSON/JS module in the frontend, e.g. `src/viz/src/cardData.js`,
mapping each of the 22 `DECKLIST` names to:
```js
{ type: "Land" | "Artifact" | "Creature" | "Sorcery" | "Instant" | "Filler",
  sortPriority: 0 (land) | 1 (artifact) | 2 (creature) | ...,
  artFile: "urzas-mine.jpg" }
```
Hand-authored once from `game.py`'s `DECKLIST`/`CardType` (source of
truth), not derived at runtime — this is fixed metadata, so there's
nothing to keep in sync beyond "if the decklist ever changes, update this
table too," same as any other static reference data.

## Architecture

```
src/viz/                      -- becomes a self-contained Node/Vite project
  package.json
  vite.config.js
  index.html
  scripts/
    fetch-card-art.mjs          -- one-time prep: Scryfall -> public/card_art/
  public/
    card_art/                    -- gitignored, populated by the prep script
  src/
    main.jsx
    App.jsx
    cardData.js                  -- static name -> type/sortPriority/artFile
    components/
      DropZone.jsx                -- drag-and-drop / file-picker loader
      SearchFilters.jsx           -- turn-online / card-in-hand / score, AND-combined
      Pagination.jsx
      BatchList.jsx
      GameView.jsx                 -- step-through: hand/battlefield/graveyard/action detail
      Card.jsx                     -- one card: art, name fallback, tapped rotation
```

The old `src/viz/visualize.py` and `viewer_template.html` are removed as
part of this rebuild (superseded, not kept alongside).

**Data flow**: everything after the initial drag-and-drop is client-side
— the dropped JSON file is parsed in the browser via the File API, held
in React state, filtered/paginated/rendered from there. No backend, no
API calls at view time (only the one-time prep script talks to Scryfall,
and only for images, never game data).

## UI layout (updated)

**Landing / drop zone**: shown until a file is loaded. Drag-and-drop
target plus a "browse for a file" fallback.

**Batch list view** (after a file loads):
- Search filters at the top (turn-online range, card-in-hand dropdown,
  score range) — AND-combined, updating the filtered set live.
- Paginated list of the filtered games (sorted by score, as the log
  already is), page-number controls below.
- Click a row → game view.

**Game view** (unchanged in spirit from v1, updated in rendering):
- Hand, battlefield, graveyard rendered as real card art, not text chips.
  Battlefield grouped/sorted lands → artifacts → creatures. Tapped cards
  rotated 90°.
- Current action's detail (`fetched`/`left_battlefield`/`tapped_for_cost`/
  `scry`), turn number, live resource-quality readout — all as in v1.
- Final step: terminal score + success/failure + `turn_assembled`/
  `turn_online`.
- Arrow keys step forward/back; a way back to the batch list.

## Explicitly out of scope for this plan

- Alternative-actions/model-confidence display.
- Any change to the trained model, reward function, or `game.py`.
- A hosted/deployed version — local dev-server use only (`npm run dev`).
- Editing/annotating games from within the viewer.
- Virtualized rendering for very large datasets (deferred per the
  pagination decision above).
- Fetching card art live at view time (deferred per the caching decision
  above; the app never needs internet access to view a game, only the
  prep script does).

## Before writing a checklist

This is a substantially bigger rebuild than v1 — confirm the architecture
above (directory layout, the static card-type/art lookup replacing a log
format change, drag-and-drop as the sole data-loading path) matches your
intent before it's turned into concrete build phases.
