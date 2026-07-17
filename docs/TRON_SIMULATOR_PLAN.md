# Tron Assembly Simulator — Plan

## Goal

Estimate, for a given turn N, the probability that all three Tron lands
(Urza's Mine, Urza's Power Plant, Urza's Tower) are in play simultaneously.
Output is a probability curve across turns (e.g. "62% by turn 3, 89% by
turn 4"), not a single number.

**Two metrics are tracked per simulated game, not one** (per your
correction — these can genuinely differ once the real decklist is in play,
since two of its lands enter tapped and Tron lands can get tapped for
mana before the third one lands):

- **Metric A — "Tron assembled"**: the turn on which the third unique Tron
  land type (Mine/Power Plant/Tower) first enters the battlefield,
  regardless of whether anything is tapped at that instant.
- **Metric B — "Tron online"**: the first turn where, immediately after
  that turn's untap step and before anything is spent, all three Tron
  land types are simultaneously untapped and controlled by you. This is
  the turn you could actually cast something costing 7 generic mana off
  Tron alone.

A and B are usually the same turn, but can diverge: e.g. if you already
have Mine and Power Plant in play and tap both for mana to cast Crop
Rotation, which finds Tower and puts it onto the battlefield — Metric A
happens that turn, but Metric B doesn't happen until your *next* untap
step, since Mine and Power Plant were tapped when Tower arrived. The
greedy policy (below) tries to avoid this by preferring non-Tron mana
sources for costs whenever one is available, but it can still happen when
Tron lands are the only mana on board.

## Method: Monte Carlo, not exact combinatorics

Run N simulated games (default 50,000–100,000, configurable), each with an
independently shuffled library, a fixed greedy play policy, and no
opponent. For each game, record the turn Tron first came online (or "never"
within the turn horizon). Aggregate into a per-turn cumulative probability.

Chosen over exact hypergeometric math because the deck includes search
effects (some shuffle, some don't) and turn-by-turn sequencing decisions —
modeling that space exactly gets complicated fast, whereas Monte Carlo
handles it for free by just playing the games out.

## Scope of MTG rules modeled

In scope:
- Zones: library, hand, battlefield, graveyard.
- Turn structure: draw step (skipped turn 1 on the play), one merged main
  phase (see note below), end step. Untap step exists only to reset land
  tapped-status.
- Sorcery-speed timing: lands and sorcery-speed spells only during your
  main phase, only when nothing else is being resolved (no stack needed —
  see "not modeled").
- One land drop per turn.
- Lands entering tapped vs untapped: Wooded Ridgeline and Bojuka Bog both
  enter tapped, so this rule is actually exercised by the real decklist
  (not just supported speculatively).
- Lands/artifacts with two abilities that share the same {T} cost (e.g.
  Tocasia's Dig Site: tap for {C} *or* pay {3}+tap to surveil 1; Conduit
  Pylons: tap for {C} *or* pay {1}+tap for any color) — only one such
  ability can be used per permanent per turn, modeled generically rather
  than as a special case per card.
- Mana: each land taps for specific colors; spells have real costs; you can
  only cast/activate what your untapped mana can pay for.
- Search effects that shuffle vs. don't (Expedition Map/Crop
  Rotation/Generous Ent's Forestcycling shuffle after; Ancient Stirrings
  does not — bottoms the unchosen cards instead).
- Card draw, discard-to-hand-size at cleanup is **not** modeled (see below)
  since hand sizes never get large enough in this deck to matter.

Explicitly **not** modeled (no opponent exists, so these have no effect on
the question being asked):
- The stack / priority passing between players. Instants are simply usable
  "any time during your main phase" instead of holding priority correctly,
  since there's nothing to respond to.
- Combat entirely (no attackers/blockers matter for counting lands).
- Mulligans (per your answer: always keep opening 7, on the play).
- Opponent's deck, hand, or actions.
- Any card interaction not related to finding/playing lands (creature
  bodies, removal, combat tricks) — those are all abstracted into a single
  generic filler card (see decklist).

## Card pool

Sourced from your `deck list.txt` (60 main-deck cards, verified against
Scryfall oracle text). Every card that can plausibly change *when* Tron
comes online — produces mana, searches/tutors, shuffles, surveils/scries,
or draws extra cards — is implemented with its real rules text. Anything
left over (pure combat/lifegain/removal payoffs with no card-selection or
mana component) collapses to a single generic filler card, since it
cannot affect the outcome being measured.

### Tron lands (12)

| Qty | Card | Implemented effect |
|----:|------|---------------------|
| 4 | Urza's Mine | Tap: Add {C}. Add an additional {C} if you control an Urza's Power Plant and an Urza's Tower. |
| 4 | Urza's Power Plant | Same pattern, checks for Mine + Tower. |
| 4 | Urza's Tower | Same pattern, checks for Mine + Power Plant. |

### Other real lands (6)

| Qty | Card | Implemented effect |
|----:|------|---------------------|
| 2 | Forest | Tap: Add {G}. |
| 1 | Wooded Ridgeline | **Enters tapped.** Tap: Add {R} or {G}. |
| 1 | Bojuka Bog | **Enters tapped.** ETB: exile target player's graveyard (no-op, no opposing graveyard exists — omitted). Tap: Add {B} (never actually needed by any spell in this deck, but it's still a land drop / Crop Rotation target). |
| 1 | Tocasia's Dig Site | Tap: Add {C}. *Or* {3}, tap: Surveil 1 (shares the tap cost — one or the other per turn). |
| 1 | Conduit Pylons | ETB: Surveil 1. Tap: Add {C}. *Or* {1}, tap: Add one mana of any color (shares the tap cost). |

### Other relevant nonland cards (24)

| Qty | Card | Implemented effect |
|----:|------|---------------------|
| 4 | Expedition Map | {1}, T, Sacrifice: search library for a land, put into hand, shuffle. |
| 2 | Crop Rotation | {G}, sacrifice a land: search library for a land, put it **directly onto the battlefield**, shuffle. |
| 4 | Ancient Stirrings | {G}: look at top 5, may take one noncreature colorless card to hand, rest to bottom in random order — **no shuffle**. |
| 4 | Bonder's Ornament | Tap: Add one mana of any color. *Or* {4}, tap: draw a card (shares the tap cost — this deck only ever has one copy in play, but the real "each player who controls one draws" text still nets you a card solo). |
| 4 | Candy Trail | ETB: Scry 2. {2}, T, Sacrifice: gain 3 life (irrelevant, ignored) and draw a card. |
| 2 | Barrels of Blasting Jelly | {1}: Add one mana of any color, activate only once per turn (no tap — doesn't compete with anything). The {5}/sac/damage ability is combat-relevant only — omitted. |
| 2 | Relic of Progenitus | {1}, Exile this artifact: draw a card. (Its graveyard-exile tap ability is a no-op solo — omitted.) |
| 2 | Generous Ent | 5/7 body is irrelevant (folded into filler-style "just a card" for combat purposes). Forestcycling {1}: discard this card from hand, search library for a Forest, put into hand, shuffle. |

### Filler (18)

| Qty | Card |
|----:|------|
| 2 | Rooftop Percher |
| 2 | Boulderbranch Golem |
| 4 | Maelstrom Colossus *(Cascade explicitly can't hit a land — nonland only — so even this card's one flashy ability is provably irrelevant here)* |
| 4 | Bramble Wurm |
| 4 | Pinnacle Kill-Ship |
| 2 | Breath Weapon |

Total: 12 + 6 + 24 + 18 = 60. ✓.

### Judgment calls made here (flag if you'd rather draw the line differently)

- I counted **mana rocks/fixing, scry/surveil, and extra card draw** as
  "relevant," not just direct land tutors — because they change how often
  you can afford/find your search spells or dig an extra card deep, which
  does shift the probability curve, even if indirectly. If you'd rather
  treat only direct land-tutoring as relevant and fold Bonder's Ornament,
  Candy Trail, Relic of Progenitus, Barrels of Blasting Jelly, and
  Generous Ent's forestcycling into plain filler, that's a smaller/faster
  model to build — say so and I'll simplify.
- Generous Ent's Forestcycling only ever finds a **Forest**, never a Tron
  piece directly — it's included because it's still a real search+shuffle
  effect and a guaranteed land drop, not because it can find Tron lands.
- Sideboard (15 cards) is out of scope entirely — only the 60-card
  maindeck is simulated, no sideboarding.

## Engine model (concepts, not code yet)

- **Card**: name, type, cost (mana symbols), and an `effect` — a small
  function describing what it does when played/activated (land: tap
  ability + enters-tapped flag; spell: search/shuffle/etc.). Filler card
  has no effect.
- **Land-on-battlefield**: reference to its Card, `tapped: bool`.
- **GameState**: library (ordered list), hand, battlefield, graveyard,
  lands-played-this-turn counter, mana currently available.
- **Turn loop**: untap all lands → draw step (skip on turn 1 if on the
  play) → main phase (repeat: policy picks next legal action — play a
  land, cast a sorcery-speed spell/activate an ability, or pass — until
  policy passes or no legal actions remain) → end step (no cleanup needed
  for this deck) → check "Tron online?" → next turn.
- **Shuffling**: any effect that says "shuffle" re-randomizes the
  remaining library order. Ancient Stirrings instead sends unchosen cards
  to the bottom in a random order (immaterial for future draws either way
  in Monte Carlo, since library is already random each game — implemented
  correctly anyway to keep the engine honest for e.g. future "know top
  card" effects).

## Decision policy (fixed greedy heuristic)

Since there's no opponent, the "AI" just needs a deterministic priority
order for its one job: assemble Tron ASAP. Draft priority list, evaluated fresh each time the policy gets to act.
Throughout, whenever a cost could be paid with either a Tron land or a
non-Tron land/mana source, **prefer the non-Tron source** — this is what
keeps Metric A and Metric B close together, mirroring how a real pilot
plays.

1. **Land drop**: if a Tron land you don't yet control is in hand, play it
   (missing-type order Mine > Power Plant > Tower for reproducibility when
   more than one is available). Otherwise play a land that can produce
   {G} if you don't have an untapped green source yet (needed for Crop
   Rotation/Ancient Stirrings/Forestcycling). Otherwise play whichever
   land is otherwise least useful to hold (tapped lands like Wooded
   Ridgeline/Bojuka Bog first, since playing them early hides their
   enters-tapped downside; Tocasia's Dig Site/Conduit Pylons last, since
   their extra abilities are more useful once you have spare mana).
2. **Spend remaining mana**, in priority order, always re-checking after
   each action whether Tron is complete:
   a. Crack Expedition Map if you control one and haven't cracked it (put
      a land — the highest-priority missing Tron piece if any, else
      nothing worth fetching — into hand).
   b. Cast Ancient Stirrings if affordable (cheap, digs 5).
   c. Cast Crop Rotation only if it can find a still-missing Tron piece
      this turn (sacrifice a non-Tron land as fodder if one exists;
      otherwise skip — never sacrifice a Tron piece to find another Tron
      piece).
   d. Forestcycle a Generous Ent from hand if you need a green source and
      have no other way to get one this turn.
   e. Activate Bonder's Ornament/Barrels of Blasting Jelly's mana-fixing
      ability only if it enables casting something above that couldn't
      otherwise be cast this turn.
   f. Crack Candy Trail or Relic of Progenitus for an extra card only
      once all three Tron pieces are already in play and untapped (i.e.
      only spend mana on "just draw a card" once the goal is achieved —
      before that, mana goes toward search effects, not raw card
      draw, since a search effect is strictly better at finding lands).
3. Any search/dig effect (Map, Stirrings, Crop Rotation, Candy Trail's
   scry, Conduit Pylons' surveil, Tocasia's Dig Site's surveil) always
   keeps/fetches the highest-priority missing Tron piece if one is
   visible; once all three types are in play, these effects just take
   whatever (doesn't affect either metric, so pick arbitrarily).
4. If nothing productive to do, pass.

This policy is intentionally simple and hardcoded (per your answer — no
configurable strategy layer). It can be revisited later if the numbers
look off versus known real-world Tron statistics.

## Monte Carlo driver

- Configurable: number of simulations, turn horizon (**default 6**, per
  your correction), on the play vs on the draw (default: on the play),
  RNG seed (for reproducibility), decklist (swap-in point for the tables
  above).
- For each simulated game: shuffle deck, draw opening 7, run the turn loop
  up to the horizon, recording the turn Metric A (assembled) happens and,
  separately, the turn Metric B (online/untapped) happens — either or
  both may be "not by horizon."
- Aggregate, separately for A and B: for each turn 1..6, % of games where
  that metric was reached by that turn or earlier (cumulative). Also
  report mean/median turn (among games where it happened) and overall %
  that never got there by the horizon, for both metrics.
- Output: a table with turn number and two cumulative-% columns (Assembled
  / Online), plus the summary stats above. A chart is a nice-to-have once
  the numbers exist, not needed for v1.

## Explicitly out of scope for v1

- Mulligans, play/draw comparison as a first-class report (can pass a flag
  and rerun, not both at once automatically).
- Any opponent modeling, interaction, or removal of your permanents.
- Alternate decklists/archetypes beyond the one table above (structured so
  swapping the decklist later is just editing that table's equivalent
  data, not rearchitecting).
- Configurable/pluggable AI strategies — one fixed heuristic only.

## Before writing any code

Decklist ✓, turn horizon (6) ✓, and the two-metric definition ✓ are all
settled. Still open:

1. Default simulation count — proposing 50,000 (reruns in well under a
   second even in plain Python at this scale; can raise later if the
   curve looks noisy near turn 6).
2. The "judgment calls" callout in the Card pool section above — mainly
   whether Bonder's Ornament / Candy Trail / Relic of Progenitus / Barrels
   of Blasting Jelly / Generous Ent's forestcycling should be modeled
   precisely (current plan) or simplified into filler. Silence = I'll
   proceed with modeling them precisely.
3. Anything in the greedy policy (land-drop order, when to crack Candy
   Trail/Relic vs. hold for search spells, always preferring non-Tron
   mana for costs) you'd rather see played differently.
