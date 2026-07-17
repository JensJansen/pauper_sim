# Tron Assembly Simulator — Implementation Checklist

Companion to [TRON_SIMULATOR_PLAN.md](TRON_SIMULATOR_PLAN.md). This is the
build order — each phase should be checked against a quick manual sanity
check before moving to the next, since a bug in an early phase (e.g. mana
payment) silently corrupts every phase after it.

Planning only — nothing here is implemented yet.

## Phase 0 — Decklist as data

- [ ] Encode the 60-card list from the plan's four tables as data: one
      entry per card *name* with its quantity, not 60 individual rows —
      the engine expands quantity into copies at deck-construction time.
- [ ] Each entry needs: name, card type (Land / Artifact / Sorcery /
      Instant / Filler), mana cost (if castable), and an `effect_id`
      tag that Phase 3 will map to actual behavior (e.g.
      `TRON_LAND`, `EXPEDITION_MAP`, `ANCIENT_STIRRINGS`, `FILLER`, …).
- [ ] Sanity check: summing quantities across all entries equals 60.

## Phase 1 — Zone & state model

- [ ] Define the zones: `library` (ordered), `hand` (unordered), 
      `battlefield` (unordered, permanents only), `graveyard` (unordered —
      contents are never read back in this simulator, so it only needs to
      exist for bookkeeping/sanity checks, not real zone semantics).
- [ ] Define a battlefield permanent as: reference to its card,
      `tapped: bool`, and any per-permanent runtime flag needed (e.g.
      "Expedition Map already cracked" isn't needed since cracking
      sacrifices it — but note any card that needs one-per-turn tracking,
      e.g. Barrels of Blasting Jelly's once-per-turn mana ability).
- [ ] Define `GameState`: the four zones, `lands_played_this_turn`,
      `turn_number`, `on_the_play: bool`, `mana_pool` (see Phase 2),
      and the two metric results (`turn_assembled`, `turn_online`),
      both initialized to `None`/"not yet."
- [ ] Sanity check: construct a `GameState` from a shuffled 60-card
      library, draw 7 into hand, confirm zone counts are 53/7/0/0.

## Phase 2 — Mana system

- [ ] Represent a mana cost as counts per color plus a generic amount
      (e.g. `{G: 1, generic: 1}` for `{1}{G}`).
- [ ] Represent "mana a permanent can produce" as a list of options (most
      lands: one fixed color/colorless; Wooded Ridgeline: choice of two;
      Tocasia's Dig Site / Conduit Pylons: mutually-exclusive ability
      choice, see Phase 3).
- [ ] Implement "can this cost be paid right now" as a check against
      currently-untapped mana-producing permanents (not an actual pool —
      pay-as-you-go by tapping specific permanents, since *which* land
      gets tapped for a given cost is exactly what Metric A vs. B and the
      "prefer non-Tron mana" policy rule depend on).
- [ ] Implement "pay this cost," which the policy calls with an explicit
      choice of which permanents to tap — this function only executes a
      choice already made by Phase 5, it does not choose for itself.
- [ ] Sanity check: with 2 Forest + 1 Urza's Mine untapped, paying `{G}`
      taps a Forest, not the Mine, under the "prefer non-Tron" rule
      exercised manually.

## Phase 3 — Card effects

Implement one effect per `effect_id` from Phase 0. Each is a small,
independent function — build and manually test them one at a time in this
order (simplest/most load-bearing first):

- [ ] `TRON_LAND` (Mine/Power Plant/Tower): enters untapped; tap ability
      adds {C}, plus an additional {C} if the other two Tron types are
      also on the battlefield.
- [ ] Plain colored land (Forest): enters untapped, taps for its color.
- [ ] `ENTERS_TAPPED` lands (Wooded Ridgeline, Bojuka Bog): enters tapped;
      Wooded Ridgeline offers a color choice on tap; Bojuka Bog's ETB
      graveyard-exile is a documented no-op (confirm it's skipped, not
      silently missing).
- [ ] Dual-tap-ability lands (Tocasia's Dig Site, Conduit Pylons): model
      the "only one {T} ability per turn" constraint generically (a
      permanent that's tapped can't offer either ability again), then
      wire each card's two specific abilities into it. Conduit Pylons
      also needs its ETB-surveil-1 trigger, independent of its tap
      abilities.
- [ ] `EXPEDITION_MAP`: pay {1}, tap, sacrifice → search library for a
      land, put in hand, shuffle.
- [ ] `CROP_ROTATION`: pay {G}, sacrifice a land you choose → search
      library for a land, put directly onto the battlefield (untapped,
      per its real text), shuffle.
- [ ] `ANCIENT_STIRRINGS`: pay {G} → look at top 5 of library, optionally
      take one noncreature colorless card to hand, put the rest on the
      **bottom** in random order (no shuffle).
- [ ] `BONDERS_ORNAMENT`: two abilities sharing one {T} — add one mana of
      any color, *or* pay {4} and tap to draw a card.
- [ ] `CANDY_TRAIL`: ETB → scry 2 (reorder/bottom the top 2 as chosen);
      separately, pay {2}, tap, sacrifice → draw a card (life gain
      omitted, doesn't affect either metric).
- [ ] `BARRELS_OF_BLASTING_JELLY`: pay {1} (no tap) → add one mana of any
      color, usable once per turn — needs the "already used this turn"
      flag from Phase 1.
- [ ] `RELIC_OF_PROGENITUS`: pay {1}, exile self → draw a card.
- [ ] `GENEROUS_ENT` forestcycling: from hand, discard this card (pay
      {1}) → search library for a Forest, put in hand, shuffle. (Its
      creature body is never cast in this simulator — see Phase 5, it's
      only ever used for its forestcycling or left in hand/filler-drawn.)
- [ ] `FILLER`: no effect; can occupy hand/library/battlefield-never
      (filler is never played) and can be discarded by any generic
      discard effect if one is ever needed.
- [ ] Search-effect helper: given "find a land," implement the shared
      "prefer the highest-priority missing Tron type, else prefer a
      green source if none untapped, else nothing/arbitrary" lookup used
      by every search effect (Map, Crop Rotation, Stirrings, Ent), so the
      priority logic lives in one place, not duplicated five times.

## Phase 4 — Turn loop

- [ ] Untap step: set every battlefield permanent's `tapped = False`,
      reset any once-per-turn flags (e.g. Barrels of Blasting Jelly).
- [ ] Draw step: draw 1 card, except turn 1 when `on_the_play` is true.
- [ ] Main phase driver: loop "ask the policy (Phase 5) for the next
      action; if it returns an action, execute it (via the relevant
      Phase 3 effect and/or Phase 2 mana payment) and loop again; if it
      returns 'pass,' end the main phase." Cap the loop (e.g. 50
      iterations) as a guard against an infinite policy loop, not because
      one is expected.
- [ ] End step: no cleanup needed for this deck (confirmed in the plan —
      hand size never gets large enough to matter), but check and record
      the two metrics here: Metric A the instant the 3rd unique Tron type
      lands (so it should actually be checked right after every
      battlefield-entering event during the main phase, not only at end
      step — flag the first turn number it becomes true); Metric B
      checked once per turn, immediately after the untap step and before
      any mana is spent that turn.
- [ ] Turn increment and loop back to untap, until `turn_number` exceeds
      the configured horizon (6) or both metrics are already recorded.
- [ ] Sanity check: run one turn manually with a hand containing exactly
      one Forest and one Urza's Mine, confirm the land drop plays the
      Mine (per policy priority) and Metric A stays unrecorded (only 1 of
      3 Tron types).

## Phase 5 — Decision policy

Translate the plan's numbered priority list into an explicit decision
function called once per main-phase loop iteration:

- [ ] Land-drop sub-decision: implement the priority order exactly as
      listed (missing Tron type > green source > tapped-lands-first among
      remaining options), only when `lands_played_this_turn == 0`.
- [ ] Mana-spend sub-decision: implement steps 2a–2f from the plan in
      order, where each step first checks affordability/legality before
      being chosen; re-run the whole priority list from the top after
      every action (since state changed) rather than working through a
      fixed queue.
- [ ] Missing-Tron-type helper: a single function `missing_tron_types()`
      used by both the land-drop and search-effect logic, so "what's
      still missing" is computed one way everywhere.
- [ ] "Prefer non-Tron mana for costs" helper: given a cost and the set
      of untapped mana-producing permanents, choose which to tap,
      deprioritizing Tron lands — used by every mana payment in Phase 2.
- [ ] Sanity check: construct a mid-game state with 2 Tron lands +
      Ancient Stirrings in hand + 1 untapped Forest, confirm the policy
      casts Stirrings using the Forest, not a Tron land.

## Phase 6 — Monte Carlo driver

- [ ] Deck constructor: expand Phase 0 data into 60 concrete card
      instances, shuffle (seeded RNG for reproducibility).
- [ ] Single-game runner: opening hand of 7, run Phase 4's turn loop for
      up to 6 turns, return `(turn_assembled, turn_online)` — either may
      be `None`.
- [ ] Batch runner: repeat the single-game runner N times (default
      50,000), collect all `(turn_assembled, turn_online)` pairs.
- [ ] Config surface: N simulations, turn horizon, on-the-play vs
      on-the-draw, RNG seed — all as explicit parameters/flags, no hidden
      defaults buried in code.

## Phase 7 — Aggregation & output

- [ ] For each turn 1..horizon, compute cumulative % of games with
      `turn_assembled <= turn` and, separately, `turn_online <= turn`.
- [ ] Compute mean/median turn for each metric (over games where it
      happened) and % that never happened by the horizon, for both
      metrics.
- [ ] Print/report a table: turn number, cumulative % assembled,
      cumulative % online.

## Phase 8 — Final sanity pass (before calling it done)

- [ ] Manually trace one full seeded game turn-by-turn against the
      printed log and confirm every action matches what a human Tron
      pilot would actually do with that exact hand.
- [ ] Confirm Metric B is never less than Metric A is never true in
      reverse — i.e. `turn_online >= turn_assembled` in every recorded
      game (Metric B can never happen before Metric A, since you need the
      3rd land in play before all 3 can be simultaneously untapped) — a
      violation here means a metric-tracking bug, not real game behavior.
- [ ] One small runnable check (per your usual bar for non-trivial logic
      — a self-check script or a couple of `assert`s, not a full test
      suite) covering: deck totals to 60, opening hand is 7, `turn_online
      >= turn_assembled` holds across a batch of games, and a hand-fed
      "always draw Mine/Power Plant/Tower turns 1–3" scenario produces
      `turn_assembled == 3` deterministically.
