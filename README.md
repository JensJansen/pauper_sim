# MTG-Subset Simulator + Attention-based Self-Play RL

A from-scratch **2-player Magic: The Gathering rules engine** for a curated
subset of cards, plus a **token/attention deep-RL system** that trains a
separate policy per deck by self-play and continuous league play against a
pool of historical opponents. The engine, the card catalog, the neural
architecture, and the training loop all live in this repo with essentially no
framework dependencies beyond PyTorch.

---

## What's here

| Piece | Where | What it is |
|-------|-------|------------|
| **Game engine** | `src/game/` | A self-contained MTG-subset simulator: zones, turn/phase loop, priority, the stack, mana, combat, and 150+ card/token effects. No ML dependency. |
| **Card catalog** | `src/game/catalog/` | Card definitions grouped by color. Decks are just decklists resolved against this shared catalog — adding a deck from already-implemented cards needs no code. |
| **Action space** | `src/drl_env/` | Turns a decklist + `EFFECT_REGISTRY` into a flat action table with per-action legality + execute closures and legal masks. Not a gym `Env`, just the assembly between engine and training loop. |
| **DRL system** | `src/rl/` | A per-deck Set-Transformer + FiLM perception encoder, trunk/critic/pointer-network action heads, and a PPO self-play + league training loop. |
| **Decks** | `data/*.txt` + `league_decks.json` | An 11-deck roster (see below). |
| **Training drivers** | `src/run_league.py` | Trains every deck continuously in a league, encoder and policy together. |
| **Training-ops UI** | `src/webapp/` | A local Flask web app to start/stop/configure training runs and watch their logs live in a browser, instead of hand-building CLI invocations. |

**Roster** (`data/league_decks.json`): `mono_red_madness`, `rakdos_madness`,
`spy_combo`, `boggles`, `monster_tron`, `dmir_terror`, `elves`,
`grixis_affinity`, `jund_wildfire`, `mono_blue_terror`, `mono_red_rally`.

---

## Goals

- **A faithful-enough MTG-subset engine** to run real 2-player games:
  phases, priority passing, the stack, combat with attackers/blockers/damage,
  and a broad set of real card mechanics (madness, plot, flashback, bestow/
  auras, scry/surveil, fetch/search, token generation, initiative/Undercity,
  mulligans, decking out, state-based actions, life-total win checks). Rules
  fidelity is a standing project mandate (see `CLAUDE.md`).
- **A card representation that generalizes across decks.** Rather than a
  fixed per-slot observation, each public-zone card becomes a *token* (a
  learned identity embedding + a hand-authored static feature vector), and a
  Set Transformer lets tokens attend to one another so relative valuations
  (an attacker's threat depends on what can block it) are learnable.
- **One policy per deck, each with its own perception encoder.** Embedding,
  attention and head all train together under the same PPO update, so a
  checkpoint is self-contained: any two populations can be played against each
  other with no compatibility bookkeeping. Until 2026-08-17 there was instead
  ONE shared encoder, pretrained across all decks by a separate phase and then
  frozen for the whole run — 59% of the model, trained on ~2,000 games/deck of
  the weakest play and never updated across the 12,000+ that followed.
- **Continuous league training**, not a fixed curriculum: every deck trains
  every round against opponents resampled from a pool of historical snapshots
  of every deck, so a policy can't quietly forget how to beat an opponent's
  earlier self.

---

## Repository layout

```
src/
  game/                    The engine (package). Zero ML deps.
    cards.py                 EffectId / CardType / CardDef — the shared card data model.
    state.py                 PlayerState + GameState: zones, turn/stack/pending-
                             resolution bookkeeping. Proxies "my board" to whichever
                             player currently holds priority.
    turn.py                  Phase/Speed enums; priority rounds; mulligan; the turn
                             loop; game_coroutine + run_multiplayer_game (the 2-player
                             driver the training loop drives).
    mana.py                  Cast-then-pay mana (real CR 601.2): a cast is legal when
                             the pool PLUS still-untapped sources could cover it
                             (plan_payment, over available_mana_units + can_pay), and
                             the sources are tapped inside the payment (601.2f) before
                             the pool is spent (601.2g). The agent makes every tap;
                             the solver only decides whether a cost is payable.
    decklist.py              Parse data/*.txt decklists against CARD_DEFS.
    registry.py              Union of every color catalog -> CARD_DEFS + EFFECT_REGISTRY;
                             derive_pending_kinds.
    catalog/                 Card definitions by color (black/blue/colorless/green/
                             multicolor/red/white).
    effects/                 Generic effect plumbing (each card catalog calls in):
                             casting, combat, stack + triggers, state_based (SBAs +
                             cleanup), stats (Aura/keyword/P/T), tokens, win_check,
                             madness_and_plot, undercity (initiative + its two
                             Undercity-only resolution kinds, see resolution/ below),
                             shared.
    resolution/              Multi-step decisions the MODEL makes one action at a time:
                             _core (begin/complete state machine) + one handlers_<kind>
                             module per resolution category -- handlers_mulligan,
                             handlers_targeting (incl. stack targeting), handlers_combat
                             (declare-blockers/damage-assignment), handlers_casting
                             (cast-copy/mode/X/Delve/mana subdecisions/Madness),
                             handlers_library (search/graveyard/scry/surveil/ponder/
                             discard/sacrifice/explore), handlers_triggers (placement
                             ordering) -- all re-exported flat via resolution/__init__.py.
                             Two Undercity-only kinds (choose_room, throne_reveal) live in
                             effects/undercity.py instead, not here (single-caller, no
                             reason to sit in the shared pool).

  drl_env/                 Action-table / legal-mask machinery (a package, not a gym Env),
                           split by category, all re-exported flat via __init__.py:
    _actions_common.py       Shared _GATE_NO_PENDING sentinel + _hand_count_available.
    _actions_cast.py         Play land / plain Cast (incl. modal/X-cost/Delve) /
                             Activate / Forestcycle / impulse ("play from exile")
                             legal/execute pairs.
    _actions_cast_altzone.py Casting from a non-hand zone or non-default cost:
                             alt-cost/Flashback/Escape/Plot/Omen/Prototype.
    _actions_combat.py       Attack / Assign Blocker / Done blocking / trample
                             damage-to-player legal/execute pairs.
    _actions_resolution.py   Generic pending-resolution dispatch: Pass, the shared
                             "Choose: X" by-name dispatch, exact-(name, slot)
                             permanent targeting, pool-mana spend, and every small
                             universal decision row (pay_unless, tuck_position,
                             may_transform/copy/cast, choose_room, target player/
                             any-target, madness, discard-or-sacrifice, Declines).
    _actions_mana.py         Mana-ability/extra-cost/filter legal/execute pairs +
                             Chromatic Star's choose_mana_color.
    _actions_table.py        build_action_table + legal_action_mask (kept together --
                             the table-builder touches every category above).
    _seat.py                 Per-seat helpers (_for_player, _lost).

  rl/                      The token/attention DRL system:
    features.py              CardVocab (stable card->index) + static per-token feature
                             vectors + build_token_set.
    arch.py                  SetTransformer (embeddings + self-attention + two PMA
                             pooling heads) and FiLM — the perception encoder.
    deck.py                  DeckNetwork: one deck's own encoder + trunk + critic +
                             pointer-net action head, trained end to end.
    mulligan.py              Per-deck pregame mulligan model + its REINFORCE trainer.
    agent.py                 SeatAgent: per-seat decision dispatch (pregame ->
                             mulligan model, everything else -> DeckNetwork).
                             HeuristicAgent: the gauntlet's hand-authored, non-learned
                             opponent (see README's Gauntlet section).
    action_bridge.py         Maps the network's combined (fixed + pointer) action space
                             back to real engine calls.
    train.py                 Rollout collection game loop (collect_rollout) + mirror &
                             cross self-play orchestration (train_selfplay); the
                             RolloutBuffer type rl.ppo/rl.rollout_parallel build on.
    ppo.py                   GAE + the PPO update itself (ppo_update), incl. the frozen-
                             shared-stack precompute-and-reuse cache.
    rollout_parallel.py      ProcessPoolExecutor multiprocessing plumbing for league
                             collection (collect_rollout_league_parallel + its worker).
    pool.py                  Builds the shared vocab + per-deck action tables from the
                             league roster (data/league_decks.json).
    league.py                LeaguePool: historical opponent snapshots, PFSP-weighted
                             sampling, eviction/archival, disk persistence.
    league_runner.py         run_league.py's reusable core: _run_session, the eval-mode
                             functions (_run_eval/_run_eval_vs_history/_vs_gauntlet/
                             _vs_heuristic), checkpoint/progress helpers, shared/frozen-
                             stack loaders. Imported directly by benchmarking/
                             training_run.py instead of importing run_league.py itself.
    rewards.py               Reward functions (win/loss with a speed tiebreaker).

  run_league.py            Thin CLI wrapper (arg resolution + main()) around
                           rl/league_runner.py.
  run_rollback.py          Promote a historical snapshot back to live.pt (the archive
                           already holds every snapshot; this is how to USE it).
                           The only run_* at this level that does NOT write state is
                           none -- that is the split: root mutates checkpoints,
                           analysis/ only reads them.
  analysis/                Read-only inspection tools. Run them from src/, e.g.
                           `python analysis/report_metrics.py ../checkpoints/<league>`
                           -- each adds src/ to sys.path itself, the same way
                           webapp/runs.py does.
    report_metrics.py      Plain-text summary of a league's metrics.jsonl -- per-record
                           trends first, then pooled stats with IMPROVING/FLAT/
                           REGRESSING/PAST PEAK verdicts and CIs.
    run_anchor_eval.py     Absolute scale: checkpoints vs a fully UNTRAINED
                           DeckNetwork, encoder included (a floor, saturates fast).
    run_snapshot_round_robin.py
                           Round robin among a deck's own snapshots -- 3-cycle count,
                           Bradley-Terry Elo + residual vs noise floor, monotonicity.
                           Showed 3 of 4 decks REGRESSING; the acceptance test for any
                           training change.
    run_cross_league_eval.py
                           Live weights of one league vs another, per deck.
                           (The per-question forensics scripts -- analyze_*.py for burn
                           saturation, hoarding, land patterns, decision entropy and
                           target fizzles, check_credit_assignment.py, and the _shared.py
                           helpers they imported -- were deleted in the 2026-08-17
                           cleanup once their investigations closed. The conclusions they
                           reached are recorded in rl/rewards.py's own comments.)
  benchmarking/            training_run.py (benchmarks the real league loop under
                           different collection configs) + _common.py (path/stdout
                           bootstrap it imports for its side effect).
  webapp/                  Local Flask UI: app.py (routes) + runs.py (subprocess/
                           registry logic) + static/*.html (landing, training-ops,
                           replay viewer -- no build step). app_public.py is a
                           separate deploy-only entrypoint exposing just the
                           replay viewer (safe to host publicly). See its own
                           section below.

data/                      Decklists (*.txt) + league_decks.json roster.
checkpoints/               Trained weights + vocab.json (gitignored; see below).
logs/                      Game event logs from --log runs (gitignored).
```

**Run scripts from `src/`.** The driver/training scripts (`run_league.py`,
`benchmarking/*`) use relative paths like `../data` and
`../checkpoints`, and the `rl.*` modules
import each other and `game`/`drl_env` by name — both of which resolve when
you run from `src/` (Python puts the script's directory on `sys.path`). The
engine itself (`game/`) is a proper importable package.

---

## The game engine (`src/game/`)

A self-contained simulator with **no ML dependency** — you can import `game`
and play out matches on its own. Highlights:

- **Turn structure & priority.** A full phase sequence (untap, upkeep, draw,
  main1, declare-attackers, declare-blockers, combat-damage, main2, end) with
  real priority rounds — both players get a chance to act at each step.
- **The stack & triggers.** Spells/abilities go on a stack and resolve;
  triggers queue and get promoted to the stack.
- **Combat.** Declare attackers/blockers, summoning sickness, combat damage,
  per-permanent identity (so an Aura can attach to one specific copy).
- **Mana — cast then pay (real CR 601.2).** A mana pool spent by explicit
  actions, Tron-land detection, flexible/filter sources — and, since
  2026-08-17, the real casting order. See below.
- **Card mechanics.** Madness, plot, flashback, auras/bestow, scry/surveil,
  fetch/search, initiative/Undercity, mulligans, token generation (Blood,
  Robot, Warrior, Eldrazi Spawn, Food, Clue, Treasure, and more), affinity/
  delve/escape cost reductions, state-based actions, decking out, and
  life-total win checks.
- **Hidden information is respected.** The OPPONENT's hand and both players'
  library CONTENTS stay hidden; only public zones (battlefield/graveyard/
  stack/exile) plus the agent's OWN hand are ever tokenized (own hand isn't
  hidden information from yourself). Opponent hand/library SIZE isn't hidden
  in real Magic either (either player can count a library or a hand), so
  it's surfaced to the agent separately, as a scalar.

  The `revealed` pseudo-zone (2026-08-17) carries cards a pending resolution
  is **currently holding outside every real zone** — a scry/surveil's revealed
  cards, a Deem Inferior tuck. Tokenized for the DECIDING seat only. This
  closed a real blindness rather than adding information: `begin_scry_surveil`
  deletes the cards from the library into `pending["remaining"]`, and the
  action row is a bare `["keep", "dispose"]`, so the agent was choosing what to
  bottom **without being able to see any of it**.

  There was also a `known_top` pseudo-zone (2026-08-13 – 2026-08-17), holding
  what a player had seen placed on top of their own library by Brainstorm. It
  was removed with the recurrent policy: it is a *computed, persisted fact*,
  and the agent is meant to learn from what it observes. It does observe the
  placement — those cards leave its own tokenized hand one at a time — so the
  sequence carries the information even though no single frame does, which is
  what the GRU is there to integrate. The agent's own previous action is fed
  in alongside the observation to make that recoverable as a lookup rather
  than an inference over hand-token deltas.

### Mana: cast then pay

Real Magic announces a spell, settles modes/X/targets, determines the total
cost (601.2e), and only **then** activates mana abilities (601.2f) and pays
(601.2g). This engine ran 601.2f *first* until 2026-08-17 — "float-first": a
cast was illegal unless the floating pool already covered it, and no source was
ever tapped during payment. That is the reverse of the real sequence, and it
forced the agent to commit to a mana configuration before it knew what it was
paying for.

**How it works now.** One function carries the change: every affordability gate
in the codebase is `game.plan_payment(state, cost) is not None`, and that now
means *"could the floating pool **plus** whatever is still tappable cover this?"*
No call site moved. The agent still makes every tap — the solver decides only
**whether** a cost is payable, never **how**. (A pre-float-first version returned
an actual tap *plan* that the engine applied, which is exactly how the mana
decision used to be taken away from the agent.)

**The solver is exact**, via the deficiency form of Hall's theorem: a cost is
payable iff there are at least as many mana units as pips, and for every subset
of the colours demanded, at least that many units can produce one of them.
Exactness matters in both directions — a false positive lets a payment begin
that cannot finish (an all-False mask and a hard error, since there is no
*Abandon payment*), and a false negative hides a legal cast, which is a
rules-faithfulness bug of its own. At most six colours can be demanded, so the
subset sweep is ≤63 iterations, and ≤7 for real costs.

**The stranding invariant.** Because a payment cannot be abandoned, anything
done during one that reduces available mana could make it unfinishable. Every
such action is gated on the payment surviving it. Tapping is *not* automatically
safe: a source with a colour choice counts while untapped as one unit that could
be any of its colours, and tapping collapses it to one — Jagged Barrens (`{B}`
or `{R}`) tapped for `{B}` against an `{R}` cost is exactly how a real game
stranded. Filters tap their own source, and Conduit Pylons is *also* a `{C}`
source. Saruli Caretaker's cost taps another creature that may itself be a mana
source, so it is excluded from the affordability count entirely and both its
choices are gated. `begin_pay_cost` asserts the invariant itself, so a caller
that skips its gate fails at that call site instead of several actions later.

Mana abilities are additionally illegal for the whole of an in-flight cast
*before* 601.2f — while modes, X, a delve amount, which graveyard copy, or
delve's exile are being chosen. That is faithful (no player receives priority
during 601.2) and it removes a whole class of stranding rather than guarding
each case.

> **AUTHORIZED SIMPLIFICATION** (owner-approved 2026-08-17). *Speculative*
> floating — a mana ability activated with no payment open — is restricted to
> the active player's own main phase. Real Magic allows it in any priority
> window. It costs little in practice: holding mana up means leaving lands
> *untapped*, not floating, and paying for an instant on the opponent's turn
> goes through the payment window, so what it actually removes is floating for a
> mana ability's *side effect* (Lotus Petal's sacrifice, Wall of Roots' counter)
> outside a main phase. This change **removed two** authorized deviations —
> the `state.in_cleanup` mana ban, and combat's shared mana window — and added
> this one.

The engine's effect functions defensively handle a no-opponent (1-player)
configuration — useful for a card-level unit test that doesn't need a second
seat — but the active surface, and everything the DRL system drives, is
2-player.

---

## The DRL system (`src/rl/`)

The observation has two parts. The first is a **variable-length set of card
tokens**, one per public-zone card for both players plus every card in the
agent's own hand (own hand is never hidden from the agent itself; the
opponent's hand stays hidden). Each token = a learned **identity embedding**
(via `CardVocab`) concatenated with a deterministic **static feature vector**
plus **dynamic per-instance state**.

The static half has two parts. *Printed stats*: mana cost, type, base P/T,
keywords. Then *what the card does*, *derived* from `EFFECT_REGISTRY` and
`CardDef.extra` rather than hand-authored, so a new card is described the
moment it is registered: mana production (produces-mana, which colors, whether
the amount is board-scaled, enters-tapped, can-filter), an effect-capability
multi-hot over every registry spec key, the pending-resolution kinds the card
can create (its behavioral signature — `choose_any_target` reads as removal,
`search_fetch` as a tutor), flags (`artifact`/`basic`/`defender`/
`indestructible`/`devoid`), and creature subtypes.

> Those derived blocks were added 2026-08-17. Before them the static vector
> was printed stats only, which collapsed the 141-card catalog onto **78
> distinct vectors** — all 26 lands shared *one*, as did Lightning Bolt /
> Galvanic Blast / Lava Dart — leaving the learned embedding as the only thing
> telling them apart. It is now **131 of 141**, and every remaining collision
> is a genuine near-functional-duplicate pair (Llanowar/Fyndhorn Elves,
> Gladecover Scout/Slippery Bogle). Two were load-bearing rather than cosmetic:
> `artifact` (affinity counts artifact *lands*, so `Island` and `Seat of the
> Synod` were the same card) and the `Elf` subtype (Priest of Titania taps for
> {G} per Elf, so the elves engine was invisible).
>
> This is deliberately the **cheap** end — presence/absence, auto-derived. The
> richer version is a hand-authored semantic vector per card (effect class,
> magnitude, what it targets); `rl/features.py` marks it as the upgrade path.

The dynamic half: a `cost_reduction_delta` (own-hand tokens only, how much
cheaper this card currently is than its printed cost, e.g. affinity/Tolarian
Terror), tapped, effective P/T, combat commitments — **is-attacking** and
**summoning-sick**, both added 2026-08-17, without which an *unblocked*
attacker was feature-identical to a creature that stayed home (combat reads
`unblocked = [p for p in attackers if p not in blocked_by]`, so it never
appears in `blocked_by` at all) — how many Auras a permanent carries and
whether it *is* an attached Aura, per-kind counter counts, whether currently
targeted by a spell/ability on the stack (mine or the opponent's, including a
spell targeting another spell), zone, and a mine/theirs side flag.

The second is a **scalar vector** (`rl/agent.py`'s `_scalar_features`) of
non-tokenized globals: turn number, lands-played, mulligans taken, whose
turn it is, each player's floating mana pool (by color), phase, life
totals, each player's library size, the opponent's hand size, and whether
anything on the stack currently targets either player directly (a burn
spell to the face has no token to carry a bit on, so that one case is
scalar-only). Library/hand size, floating mana, and a declared target are
all public knowledge in real Magic, unlike hand/library *contents* — see
"Hidden information is respected" above. The agent's own hand size isn't
included here (redundant with counting its own hand tokens above).

Every deck has its own network, encoder included — nothing is shared between
decks but the card *index* mapping:

- **`SetTransformer` (`rl/arch.py`)** — embeds + projects tokens,
  runs a joint self-attention encoder over *both* sides' tokens (so a token
  can attend across the mine/theirs boundary), then pools with two
  independent learned-query heads: a "mine" summary (trunk input) and a
  "theirs" summary (FiLM conditioning input). Pre-norm transformer for RL
  stability. Uses `torch.nn.MultiheadAttention`/`TransformerEncoderLayer`
  directly rather than hand-rolling attention.
- **`FiLM`** — turns the "theirs" summary into per-layer (gamma, beta)
  modulations of the trunk, chosen over concatenation.
- **A `GRU` between the trunk and every head** (2026-08-17) — so the critic,
  the fixed-action head *and* the pointer query are all history-aware. Without
  it the observation is strictly Markov: it describes the board right now and
  says nothing about what has happened, which makes "they held two blue up and
  passed" unrepresentable no matter how rich the per-card features get. Chosen
  over a transformer over stacked history because recurrence remains stronger
  and more stable under partial observability, and stacking N observations
  would multiply each deck's encoder cost by N.

  Two invariants it depends on. The state is keyed **by seat**, because a
  mirror pairing puts one `SeatAgent` on both seats and a shared state would
  leak seat 0's hand into seat 1's inputs. And it is cleared **per game**,
  because `ppo_update` replays every episode from a zero state — a state
  surviving a game boundary would make the update recompute hidden states that
  never occurred.
- **`DeckNetwork` (per-deck, `rl/deck.py`)** — a small trunk + critic +
  a **pointer-network action head**. The action space is the union of a
  **fixed table** of non-targeting actions (play land, cast X, pass, mana
  payments, mulligans, …) and a **pointer-scored** set of targeting actions
  (attack / assign-blocker / choose-target), scored against the
  post-attention token representations. Both halves feed **one combined
  softmax**, so a masked-categorical sample over the true legal set is
  correct.
- **Pregame mulligan model (per-deck, `rl/mulligan.py`)** — a separate small
  head reading its own deck's encoder embeddings, owning every pregame
  keep/mulligan/bottom decision. It holds that encoder by plain reference, not
  as a registered child, so its REINFORCE optimizer never steps the encoder
  PPO owns (one near-bandit sample per game should not be steering a 117k-param
  perception encoder that ~100 in-game decisions per game are also steering). A `SeatAgent` (`rl/agent.py`) routes pregame decisions to it and
  everything else to the `DeckNetwork`. It trains by its own REINFORCE with a
  direct whole-game reward, decoupled from the main PPO update — a mulligan is a
  near-bandit: one pregame choice, the game's outcome as its number.

Training is **PPO self-play** (`rl/train.py`'s rollout game loop, `rl/ppo.py`'s
update math). Mirror matches pool both seats into one buffer/update;
cross-matchups give each net its own buffer, both learning from every game.
Rollout collection parallelizes across worker processes (`rl/rollout_parallel.py`,
~3.2–3.5× on 6 physical cores).

The **league** (`rl/league.py`, `rl/league_runner.py`) keeps a rolling window of
historical snapshots per deck. Each game resamples an opponent two-level: pick
a deck (including the training deck itself, for mirror play), then pick one of
its snapshots (or its current live weights). No hardcoded Stage-1/Stage-2
phases — cross-deck and cross-snapshot exposure grows as the pool fills.

Both levels are **PFSP-weighted** (`LeaguePool.sample_opponent`'s `pfsp=True`
default), not uniform: a deck (or a specific one of its snapshots) the
training deck is CURRENTLY losing to more often gets sampled more —
`weight = PFSP_FLOOR + (1 - win_rate) ** PFSP_POWER`, a small floor so even a
thoroughly-beaten opponent stays sampleable at a low rate (driving it to
exactly zero would reintroduce the catastrophic-forgetting failure the
snapshot history exists to prevent). A never-yet-played candidate gets a
neutral 0.5 win-rate prior, so early in a run (or with `pfsp=False`) this is
statistically indistinguishable from the old uniform behavior. Running
win/loss tallies (`LeaguePool.record_outcome`, fed from every REAL training
game via `collect_rollout`'s `on_game_end` hook — no separate eval pass) are
what drive the weighting; they persist to `<league_dir>/<deck>/
opponent_stats.json` on the same checkpoint cadence as everything else, since
each training run is many separate short-lived process invocations, not one
long-running daemon.

A snapshot evicted from that rolling window is **archived, not deleted**
(`checkpoints/<league>/<deck>/archive/`) — the active sampling window stays
bounded deliberately (a stale snapshot is a weak opponent), but the archive
keeps deep history around so a deck's win rate against its own much-older
self stays measurable for the life of a run, not just its last ~6,400 games
(`DEFAULT_MAX_SNAPSHOTS_PER_DECK=32` × `snapshot_every_games=200`, raised
2026-08-06 from 8/~1,600 after a real 34,579-games/deck run showed that
window capping 95%+ of a deck's own history as permanently unreachable for
training, only usable via the vs_history eval spot-check).
See **Instrumentation** below.

---

## Training pipeline

One phase. All decks share one vocabulary (`checkpoints/vocab.json`,
append-only so old checkpoints stay valid) — the card *index* mapping, not any
learned weights. Run everything **from `src/`**.

```
cd src
python run_league.py --n-iterations N --snapshot-every 15 --n-workers 6
```

Every deck trains every iteration against a resampled league opponent. A deck
with no `live.pt` yet starts from a freshly-initialized net — encoder included
— so there is nothing to prepare before the first run.

**PPO samples whole episodes, not transitions** (2026-08-17, with the recurrent
policy): the GRU needs each game replayed in order from its own start, so the
update segments the buffer on `done` and minibatches over trajectories.
`seq_batch_size_start`/`seq_batch_size_cap` therefore count **episodes** — they
replaced `batch_size_start`/`batch_size_cap`, and the rename is deliberate so a
config carrying the old keys fails on the unknown-key assert rather than being
read as 32 *episodes* and collapsing every update to one full-batch minibatch.

This used to be a **two-phase** pipeline: `run_pretrain.py` trained one shared
`SetTransformer`+`FiLM` across all decks via throwaway per-deck heads, froze it
to `checkpoints/shared_stack_frozen.pt`, and the league then trained per-deck
heads on top of that fixed encoder. Per-deck encoders (2026-08-17) removed the
phase entirely — see the architecture note above for why. Two consequences
worth knowing:

- **PPO is slower per update.** A frozen encoder let `ppo_update` precompute
  its per-transition outputs once and reuse them across every epoch; a
  trainable one must be recomputed each minibatch so gradients reach it.
- **Snapshots are bigger.** A `live.pt`/`snapshot_*.pt` now carries its own
  117k-parameter encoder (~830 KB rather than ~360 KB).

Each deck's live net/optimizer persists in
`checkpoints/league/<deck>/live.pt`; snapshots live alongside, and a
`session.txt` counter makes runs resumable across sessions.

Key flags:
- `--n-workers` — parallel rollout processes (default 6; no reliable gain past
  physical core count).
- `--snapshot-every` — iterations between registering a snapshot of every deck
  and checkpointing live nets (default 20).
- `--total-games` (with `--league-config`, e.g. `training_configs/league_main.json`)
  — auto-sizing target: instead of `--n-iterations`, doubles the batch size
  each invocation until this many games/deck have been played.
- `--checkpoint-opponent-rate` — probability a sampled opponent is a frozen
  historical snapshot instead of that deck's current live net (default 0.0:
  always live; the one value meant to be changed deliberately mid-training).
- `--pfsp` / `--no-pfsp` — PFSP-weight opponent sampling toward whoever's
  currently beating the training deck most, instead of uniform (default True;
  see the **Continuous league training** section above and `rl.league.
  LeaguePool.sample_opponent`'s own docstring).
- `--gauntlet-league-name` — an independently-trained twin league
  (`checkpoints/<name>/`) to periodically measure this league's live nets
  against, a genuinely external reference unlike this league's own historical
  snapshots (optional; most leagues won't have one — see **Gauntlet** below).

`--games-per-iteration` isn't a flag — it defaults to `max(1, n_workers)` (one
game per worker; a smaller value used to silently starve some worker processes
of any work at all), overridable via the run-config's `games_per_iteration`.
That override exists because the default of 6 means every `ppo_update` spends
its whole trust region on **six games of evidence**. The old `--max-batch-size`
auto-sizing cap is likewise no longer a parameter — see `rl/league_runner.py`'s
(`_next_batch_games`) and `run_league.py`'s (`main()`) own comments at each
removal site for why.

**The PPO minibatch ramp (32 → 2048 over 6 steps) is documented below but has
never actually executed** — see the warning under **Instrumentation**.

Optimizer/PPO knobs (`lr`, `mulligan_lr`, `gamma`, `gae_lambda`, `target_kl`,
`n_epochs`, `adv_norm_floor`, `ent_coef`) come from
`rl.league_runner.PPO_DEFAULTS` and are overridable per league via a run-config
`"ppo"` object; an unknown key is a hard error rather than a silent no-op. Eval
budget (`eval_games`, `eval_every_sessions`) is config-driven the same way.

`ent_coef` defaults to `None`, meaning "use `rl.train.ent_coef_schedule`'s
0.02 → 0.005 anneal"; a float pins it constant for the whole run instead.
Setting it to 0.05 was run as a controlled single-variable A/B
(config since deleted with the rest of the concluded arms) and
**was not adopted** — it held
policy entropy 25% higher but left `approx_kl` unchanged and merely
redistributed elo between decks.

**`lr` changed 3e-4 → 2e-4 (2026-08-15)** — the first default here set by a
controlled experiment rather than a guess. Three single-variable arms at 10,000
games/deck, each scored against the *identical* reference (the 20,016-game
baseline's live nets):

| `lr` | median `approx_kl` | mean `epochs_run` | truncation | vs baseline |
|---|---|---|---|---|
| 3e-4 | 0.0274 | 2.67 | 99% | *is the baseline* |
| **2e-4** | 0.0243 | 3.94 | **6%** | **62.5%** |
| 1.5e-4 | 0.0152 | 4.00 | 0% | 45.0% |

The curve is sharply nonlinear: 2e-4's KL is only 11% below 3e-4's, but that is
enough to drop back under the `target_kl=0.03` epoch-mean cliff and collapse
truncation from 99% to 6%.

> **The 62.5% is a peak, not a level** (§1A.13). Extending 2e-4 to a
> budget-matched 20,016 games showed that reading was the highest of eight
> samples on an oscillating series; at matched budget it is 43.8%, i.e. parity.
> The "better final policy" claim is **withdrawn**.

2e-4 is kept as the default on what survived: it reaches parity with the
baseline's **final** policy at cum ~3,984 — a **~5× sample-efficiency gain** —
it eliminates truncation, and it is still the only value at which no deck
materially regresses (3e-4 costs dmir_terror −80 elo, 1.5e-4 costs
rakdos_madness −71). It converges much faster to the same place; it does not end
up stronger. Note that 2e-4 plateaus and oscillates exactly as 3e-4 does despite
having no truncation, so **the late-training plateau is not a trust-region
problem** and lowering `lr` further will not fix it.

Scope was **league training only**; the pretrain phase kept its own untested
3e-4 and was deleted along with `run_pretrain.py` in 2026-08-17's per-deck
encoder change. Resuming a league trained at 3e-4 now picks up
the new default; pin `"ppo": {"lr": 0.0003}` in its config to continue it
unchanged.

Training runs on **CPU** by design — the model is small (~200–250K params) and
a batch-size sweep found no GPU crossover at this size.

### Instrumentation

Every league session (`_run_session`, both league and `--matchup` modes)
appends to `checkpoints/<league>/metrics.jsonl`, one JSON line per record:
> ⚠️ **BUG 1 (fixed 2026-08-13): both schedules used to be inert.** Across all
> 40,104 PPO iterations of the 60,001-games/deck run, `batch_size` was 32 on
> every iteration and `ent_coef` stayed in [0.0191, 0.0200], because
> `--n-iterations` left `auto_sizing` False → `_save_progress` was never called
> → `progress.json` never existed → the horizon both schedules ramp against read
> 0 every session. `advance_progress` now advances it unconditionally
> (`tests/test_run_league.py` pins this). **Any hyperparameter conclusion drawn
> between 2026-08-06 and 2026-08-13 is against a configuration that is not the
> one described here.**
>
> Current state, measured over the 20,016-games/deck run that followed:
> the **`ent_coef` anneal now genuinely executes** (0.0200 → 0.0140), while the
> **minibatch ramp stays deliberately pinned off** (`batch_size_start ==
> batch_size_cap == 32`) so that fixing the counter did not switch a 24× cut in
> Adam steps on as a side effect — raise `batch_size_cap` to run it as its own
> experiment. Note the anneal running correctly is not the same as it being
> *right*: it was measured moving the wrong way relative to policy entropy, and
> the fix attempted for it did not pan out.

- `kind: "session_start"` — one header per session recording the reward
  function, roster, cumulative games/deck, and every resolved PPO/eval
  hyperparameter, so `metrics.jsonl` is self-describing rather than requiring
  the reader to reconstruct which config produced a stretch of records.
- `kind: "ppo"` — per deck per iteration: `policy_loss`, `value_loss`,
  `entropy` (computed every call, previously never recorded past stdout),
  `explained_variance` (`1 - Var(ret - value)/Var(ret)` — `value_loss` alone is
  a raw MSE with no scale attached, so a genuinely-learned critic and one whose
  targets collapsed to a constant both report a small number),
  `buffer_size` (transitions collected that iteration) and `batch_size` side
  by side, so a saturated minibatch ramp (`batch_size >= buffer_size`, at
  which point `ppo_update` stops sub-batching and just runs `n_epochs`
  full-batch steps) is directly visible instead of assumed. Every record also
  carries `cumulative_games`.
- `kind: "mulligan"` — per deck per iteration REINFORCE loss/n.
- `kind: "vs_history"` — once per session per deck (league mode only): the
  live net played against its own oldest still-active snapshot and, once one
  exists, its oldest **archived** snapshot (`rl.league`'s eviction archive,
  see above) — a direct win-rate-vs-past-self measurement, not an inference
  from loss curves. Skipped (empty) automatically until a deck has been
  through at least one snapshot cycle, so it costs nothing during a run's
  cheap early sessions. Each record carries `snapshot_id`/`is_archive`, and
  **the two labels must never be pooled**: `archive_oldest` is pinned to
  `snapshot_0` forever (eviction is oldest-first and the archive minimum is
  taken), making it a FIXED ~200-game reference, while `active_oldest` tracks a
  rolling ~6,400-game-old self. Games are played **side-swapped from a paired
  seed**, so on-the-play is balanced exactly rather than in expectation.
- `kind: "vs_gauntlet"` / `kind: "vs_heuristic"` — the gauntlet mechanism's
  two tiers, both EXTERNAL to this league's own self-play history (unlike
  `vs_history`'s snapshots). See **Gauntlet** below.

Every game the engine plays (any collection path, since `collect_rollout` is
the one game loop) also gets one `game_over` event appended to its own
`event_log` (`winner`, `turn_won`) — the outcome was previously never written
to the log stream at all, only reconstructible by replaying `life_change`
deltas by hand.

`python analysis/report_metrics.py <league_dir> [--window N]` prints a plain-text
summary read from `metrics.jsonl` — stdlib only, no plotting dependency. It
leads with the **per-record sequence** for every win-rate series and only then
pools, because pooling is what hides a decline: `dmir_terror` vs
`archive_oldest` ran 60/80/85/65/80/60/60/80/75/55/50/60/60/45 across sessions
24–37, which pools to a bland ~65%. Four verdicts are distinguished —
`IMPROVING` / `FLAT` / `REGRESSING` (linear decline) / `PAST PEAK` (below a
window this run already reached, whatever the overall trend). `PAST PEAK` is
the one that matters here: every deck in the league rose then fell, and a
linear trend test reads that shape as no trend at all. Its threshold is
Šidák-corrected for the number of windows searched, since the best of many
noisy windows is high by selection. `FLAT` is annotated with the minimum
effect the sample size could have detected, so "no change" is never confused
with "cannot tell".

The webapp's escalation loop calls the same statistics via
`webapp/runs.py:learning_health` after every batch, so a run that has stopped
buying anything says so in its own log. It **warns rather than stops** by
default — set `stop_on_regression` on the run entry to opt in.

### Gauntlet

`vs_history` (above) and PFSP-weighted sampling both only ever measure a
league against **its own** self-play history — real signal, but it can't
tell "genuinely improving" apart from "well-adapted to beating a closed
population that co-evolved with itself." The gauntlet is two EXTERNAL
reference points, outside that history entirely:

- **Tier 2 — an independently-trained twin population**: the actively-trained
  league (`training_configs/run_default.json`, `league_name:
  "4_deck_subleague_test"`) is measured each session against a SEPARATE
  checkpoint tree (`gauntlet_league_name: "4_deck_subleague_gauntlet"`) that
  never plays against its live nets during training — a genuinely external
  reference. Two runs from an identical algorithm/config still diverge into
  different regions of strategy space purely from a different nondeterministic
  training trajectory, so a blind spot the WHOLE active population shares (the
  risk PFSP and `vs_history` can't rule out) is far more likely to show up
  against a genuinely independent opponent than against anything drawn from
  the league's own history. Wired via a config's `gauntlet_league_name` field
  (`rl.league_runner._run_eval_vs_gauntlet`, once per session per deck, only
  once the twin population has a checkpoint for that deck).

  These two checkpoint trees swapped roles on 2026-08-05: `4_deck_subleague_test`
  had trained ~55,000 games/deck (its first ~10,000 without PFSP at all) and
  plateaued flat against its gauntlet twin; the twin (PFSP from game 1, stopped
  at a fixed ~10,000 games/deck) took over as the actively-trained population
  instead, and the checkpoint directories were renamed to match their new
  roles — so `4_deck_subleague_test` is, and remains, whichever population is
  actively training, `4_deck_subleague_gauntlet` the frozen reference. (The
  original writeup lived in `SWAP_EXPERIMENT.md` at the repo root; that file
  no longer exists — `training_configs/run_default.json`'s own `_league_note`
  is the surviving record.) See `training_configs/run_gauntlet.json`'s own
  note for why that config is retired rather than reused to train the (now
  frozen) other side.

  **The confound that disabled this in 2026-08-07 is now structurally
  impossible.** `_run_eval_vs_gauntlet` used to load the gauntlet league's
  saved per-deck weights onto a `DeckNetwork` built on the CALLING population's
  frozen shared stack. When the 2026-08-06 restart re-pretrained that stack
  from scratch without retraining the gauntlet, gauntlet's decisions began
  running through an embedding space its FiLM/pointer-query layers had never
  seen — same tensor shapes, so nothing crashed, and every `vs_gauntlet`
  reading across sessions 0–14 (24,579 games/deck) was confounded before anyone
  noticed. A `stack_id.txt` guard was added to catch it.

  Per-deck encoders (2026-08-17) removed both the failure and the guard: a
  checkpoint carries its own encoder, so a gauntlet opponent can only ever be
  played through the perception it was trained with. `gauntlet_league_name` is
  still unset in `run_default.json` and both twin populations were deleted in
  the 2026-08-17 cleanup, so re-enabling it needs a newly grown twin — but
  nothing about the mechanism is unsafe any more.

  **2026-08-06 comprehensive restart**: separately from the swap above, a
  7-agent audit (`TRAINING_IMPROVEMENT_OPTIONS.md` at the repo root) found the
  swapped-in population was still rising-then-regressing (peaked 61% vs its
  gauntlet twin at 24,579 games/deck, fell to 26-36% by 24,877-27,649) rather
  than genuinely plateaued — traced to a hard 1,600-game opponent-memory
  ceiling, PFSP over-concentrating on one structurally-unwinnable matchup, and
  PPO exploration entropy collapsing to a floor by ~250 games/deck and never
  recovering. All three fixed (`rl/league.py`'s `DEFAULT_MAX_SNAPSHOTS_PER_DECK`
  and `PFSP_POWER`; `rl/ppo.py`'s `target_kl` and `rl/train.py`'s
  `ent_coef_schedule`), plus the dense mana-burn reward reverted (see the
  rewards section above) and a substantially larger pretrain budget before the
  refreeze (that phase no longer exists). `4_deck_subleague_test` restarted
  from zero under all of it at
  once; `4_deck_subleague_gauntlet` was left untouched as the (now
  representation-mismatched, see above) stale reference.
- **Tier 1 — `rl.agent.HeuristicAgent`**: a hand-authored, non-learned
  opponent, reusing the exact same legal-action machinery a trained
  `SeatAgent` does (`_build_decision`, `_executor_for`) but scoring among
  legal choices by fixed, general MTG principles instead of a policy: play a
  land if possible; else cast the highest-cost thing affordable; attack a
  creature only if it's safe (nothing can kill it) or a fair-or-better trade
  (the cheapest creature that could profitably kill it costs at least as
  much mana as the attacker); block to kill when possible; else pass.
  Deliberately rough, not optimal-play — the point is catching whether the
  population still does obviously-good, never-adapting things, not
  benchmarking against a strong opponent. Currently scoped to
  `mono_red_rally` only (`heuristic_decks` in a league config) — an
  aggressive creature deck is the natural fit for a simple greedy heuristic;
  not audited for every deck in a roster. Pregame (mulligan) delegates to
  `AlwaysKeep`.

Both report into `metrics.jsonl` (`kind: "vs_gauntlet"` / `"vs_heuristic"`,
picked up by `analysis/report_metrics.py` the same way as everything else) and cost
nothing when unconfigured — most leagues won't set `gauntlet_league_name` or
`heuristic_decks` at all.

### Direct matchup (no league sampling)

```
python run_league.py --matchup DECK_A DECK_B [--games 50] [--log path/to/games.json]
```

Runs a fixed pairing between two named decks, still updating and
checkpointing both. `--log` captures the engine's own event log for every
game as one JSON file — the input the replay converter consumes. (`--log` is
wired only through `--matchup` mode.)

### Rewards & win condition

- **Rewards** (`rl/rewards.py`): league play uses `deploy_reward_v6` (2026-08-12 — supersedes `deploy_reward_v5`, see
  below): a **flat `+1.0` on any win, `-1.0` on any loss or no-winner
  timeout** (`flat_win_loss_reward`), with the dense mana-burn penalty below
  applied to the **winner only**. There is no efficiency scaling (the earlier
  `deploy_reward_v1`'s efficiency band induced an action-space-minimization
  pathology where policies learned to shrink their own board to "win in fewer
  actions") and, as of v5, **no cleanup-discard penalty `q` at all** on either
  band — v1-v4's `q` (a Hill curve over `PlayerState.cleanup_discard_turns`,
  cards hoarded past hand size) is gone. See "Why `q` was dropped" below.
  `PlayerState.mana_burnt_total`/`mana_burnt_this_turn` feed no reward —
  they remain raw diagnostics for logging/viz (`mana_burnt_this_turn_single_pip`,
  a filtered subset of the latter, does feed reward — see dense mana-burn
  shaping below). The **mulligan model** trains
  on its own reward (`rl/mulligan.py`): win payout minus a convex
  (quadratic) per-mulligan penalty, on transitions accumulated across
  several league iterations per REINFORCE update (not one iteration's worth
  each time) since 2026-08-06 — see `rl.league_runner._run_session`'s
  `MULLIGAN_UPDATE_EVERY`.

  **Dense mana-burn shaping — tried, reverted, fixed twice, reweighted, then
  made winner-only (2026-08).** A per-transition penalty for mana burnt at a
  phase boundary (rule 500.4) is wired into `deploy_reward_v3`, `v4`, and —
  winner-only, via `refund_on_loss=True` — `v5`, as
  `with_dense_mana_burn_penalty`, replacing an earlier, narrower,
  exemption-aware version (`with_mana_mistake_penalty`). Its first version
  read raw `PlayerState.mana_burnt_this_turn` unconditionally — every pip
  burnt counted, no exemption for whether the burn was avoidable. A real
  training batch under it showed the `elves` deck regressing consistently
  across every matchup in a cross-league benchmark — traced to Priest of
  Titania's mana ability (`"count_all"`, `game/mana.py`, sums Elves on BOTH
  battlefields with no partial-tap option): for that archetype, large
  per-turn mana bursts are an intrinsic, unavoidable part of correct play,
  not a mistake a dense curve on raw totals could distinguish from real
  waste. The wrap was reverted to plain `deploy_reward(win_floor=1.0)`.

  A second version re-enabled it reading `PlayerState.
  mana_burnt_this_turn_metered`, a WHOLE-PHASE fix: `game.turn.
  _empty_mana_pools` excused a player's entire phase's burn whenever any
  board-state-scaled burst source (Priest of Titania, Overgrown Battlement)
  was tapped that phase, regardless of whether an ordinary metered source
  (every land, every simple mana dork) was also tapped the same phase —
  correct for Priest, but blind to a mixed phase where a genuinely
  avoidable single-pip tap sat alongside an unavoidable burst.

  A third version replaced that whole-phase exclusion with **per-pip
  attribution**: `PlayerState.mana_pool_single_pip` is a shadow count,
  parallel to `mana_pool`, tracking how many of the currently floating pips
  of each color trace back to a "single-pip" mana-producing EVENT — one
  that added exactly 1 symbol to the pool in that one event
  (`game.mana.float_mana`, `len(symbols) == 1`, computed dynamically per
  event, not from a static per-source-kind list). A plain land or Llanowar
  Elves always qualifies; Rakdos Carnarium and Utopia Sprawl's automatic
  bonus (2+ symbols per tap) never do; a Tron land qualifies only while not
  all three Tron types are controlled; Priest of Titania/Overgrown
  Battlement qualify only in the edge case their count happens to resolve
  to exactly 1. A mana filter's output (Conduit Pylons/Barrels of Blasting
  Jelly) is the one explicit exception, forced untagged regardless of count
  (`taggable=False`) — a deliberate pool→pool conversion, not reflexive
  tapping. Spending a pip (`game.mana.spend_one_pip`) always consumes an
  UNTAGGED pip of that color first, so a burst source's own unavoidable
  excess absorbs blame for a burnt leftover ahead of a genuinely avoidable
  single-pip tap of the same color — `PlayerState.
  mana_burnt_this_turn_single_pip` (`game.turn._empty_mana_pools`) sums
  whatever remains tagged at the moment of the burn.

  A **2026-08-10 engine fix** closed a real gap in that same accounting:
  `game.turn.Phase.END` used to run `cleanup_step` immediately on entry, with
  no priority window before it — since mana abilities are gate-free (legal in
  any window, 605.1a/605.3b), a player could float mana during the forced
  hand-size discard itself, which (because `_run_turn_gen` resets the
  per-turn mana-burn counters for the *next* turn before that next turn's own
  boundary sweep runs) escaped the dense penalty entirely. `Phase.END` now
  runs a real end-step priority round first (rule 513), sweeps mana at an
  explicit sub-boundary, THEN runs cleanup. Mana abilities are illegal for the
  whole cleanup portion — as of 2026-08-17 that falls out of the general
  main-phase rule below rather than needing `state.in_cleanup` to say so, and
  the separate authorized simplification that used to sit on that flag is gone.
  `in_cleanup` itself remains, since it is the only thing distinguishing the end
  step from cleanup (they share `Phase.END`). The trigger-driven extra priority
  round there is kept, specifically so a Madness card discarded by forced
  cleanup can still resolve its own cast-or-graveyard decision.

  `deploy_reward_v3` (2026-08-10) reshapes the curve itself considerably on
  top of all the above, per owner request: `mana_burn_c=2.0`/`mana_burn_p=2.5`
  (down from v2's `3.3`/`4.0`) makes the first wasted pip in a turn cost
  `≈0.150` instead of `≈0.008` (~18x), reaching near-max badness within 4-5
  pips instead of 6-7. `game_penalty_cap=0.6` (down from `2.0`) and
  `discard_weight=0.4` (new — see above) are sized to sum to exactly
  `win_floor` (`1.0`), so a clean win scores `1.0`, the worst realistic loss
  (heavy discarding *and* mana burn saturating its cap) approaches `-1.0`,
  and the spread between them is exactly `2.0` by construction rather than an
  empirical hope — this also RESTORES "every win outscores every loss" as an
  actual guarantee, which v2's own combined budget (`1.0` possible from `q`
  alone, `+2.0` more from an uncapped mana wrap) had explicitly abandoned as
  an accepted tradeoff. `deploy_reward_v2` is kept, unchanged, for reference
  (same treatment `deploy_reward_v1` already gets). `deploy_reward_v3` is
  **superseded** as of `deploy_reward_v4` below (same day) but is likewise
  kept, unchanged, for reference.

  `deploy_reward_v4` (2026-08-10, second revision) supersedes
  `deploy_reward_v3` above: a real training run under v3 (sessions 13-25,
  ~10,001 → 20,004 games/deck — the same run that validated the
  `on_single_pip_burn` attribution fix described above) drove `elves` and
  `rakdos_madness` into a degenerate zero-mana-development collapse —
  `vs_gauntlet` win rate for `elves` flatlined at exactly `0/20` for 5
  consecutive sessions, confirmed via a hand-inspected game where it held a
  playable land and passed rather than ever play it, forfeiting rather than
  risk the burn penalty. v3's steepened curve had been calibrated to
  compensate for the very credit-assignment bug `on_single_pip_burn` fixed
  the same session — once the charge started landing precisely on the
  causal Tap action instead of almost uniformly on whatever was pending,
  that compensation became a double-count, strong enough on two archetypes
  to make guaranteed inaction score better than a real attempt to win. `v4`
  keeps v3's `discard_weight=0.4`/`game_penalty_cap=0.6` split (unrelated to
  the curve's own steepness, and what makes "every win beats every loss" a
  provable guarantee) but reverts `mana_burn_c`/`mana_burn_p` to v2's
  original `3.3`/`4.0` (from v3's `2.0`/`2.5`), removing the compensation
  that was tuned for the now-fixed bug while keeping the correct,
  precisely-targeted attribution.

  That bet was then measured, and came back **split**: retrained from scratch
  under v4, `elves` showed the passivity pattern in only `0.8%` of games
  (1/125), but `dmir_terror` still showed it in `11.2%` (14/125) — in a fresh
  league that had never seen v3's collapse at all. So v4 helped but was not
  the whole fix, and the earlier "v3's steepened curve caused it" reading was
  at best partial.

  `deploy_reward_v5` (2026-08-11, **superseded** by `deploy_reward_v6` below
  — its two structural changes were validated over 20,065 games/deck and are
  carried into v6 verbatim; only its curve *weight* was wrong) supersedes
  `deploy_reward_v4` with two **structural** changes rather than a re-tuning:

  1. **Mana burn is charged to the WINNER only.** v4 charged it on both
     bands, and that turned out to reward losing *passively* over losing
     while trying: a seat that never taps mana cannot burn mana, so a fully
     passive loss scored exactly `0.0` on that term by construction, every
     time, while a seat that developed its board and made any ordinary
     sequencing mistake paid up to the whole cap. Measured across a real
     10,003-games/deck run, `dmir_terror`'s own losses averaged `0.321` total
     penalty across 14 passive games vs. `0.598` across 64 active ones —
     nearly 2× worse for trying. The observed behavior matched exactly: 26 of
     26 flagged games cast **zero** spells, including free ones (Snuff Out
     costs only life), which no mana-shaped incentive explains on its own but
     a general "losing quietly is cheaper" gradient does.
  2. **Why `q` was dropped** (both bands, not just the loss band). `q`
     existed to discourage hoarding, but the win/loss outcome already carries
     that: hoarded cards stay **visible** in game state (an overflowing hand,
     uncast threats, a board that never developed), so a terminal signal can
     attribute their cost on its own given enough training. Burnt mana is the
     opposite, and that asymmetry is exactly why the dense term survives
     while `q` doesn't: once a phase boundary clears the pool, the waste
     leaves **no trace** in game state for any terminal signal to see. This
     also removes the tight `discard_weight + game_penalty_cap == win_floor`
     equation v3/v4 depended on.

  **Curve/guarantee.** The `_hill` shape is unchanged ("first pip nearly
  free, accelerating, asymptotic") but is now scaled by a new
  `mana_burn_weight=1.5` (`_hill` itself asymptotes at `1.0`), with
  `mana_burn_c=2.9`/`p=4.0` placing that asymptote near 5 burnt pips and
  `game_penalty_cap=1.5` bounding the whole-game sum. Per-turn charge by pips
  burnt: `1→0.021`, `2→0.277`, `3→0.801`, `4→1.175`, `5→1.348` (~90% of the
  weight), asymptoting toward `1.5`. Weight and cap are equal here but do
  different jobs — under v4 (weight implicitly `1.0`, cap `0.6`) a single
  4-pip turn computed `0.683` and was immediately clipped by the whole-game
  cap, so the *cap* set per-turn magnitude and every later mistake that game
  was free; now the curve owns per-turn magnitude and the cap owns the sum,
  so one bad turn (`1.35`) is distinguishable from several (up to `1.5`).
  *(Measured 2026-08-11: that last claim is only ~10% true — `1.348` of a
  `1.5` cap is 90% of the budget, so v5 narrowed v4's failure mode rather
  than escaping it. This is what `deploy_reward_v6` below fixes.)*
  Guarantee, re-derived (it no longer needs the equation above): worst-case
  win `= 1.0 - 1.5 = -0.5`; **every** loss `= -1.0` exactly, with no range at
  all. A sloppy win *can* now score negative, which v4 could not (floor
  `+0.4`) — deliberate and harmless, since the only ordering that matters is
  win-vs-loss.

  **Implementation is a DEFERRAL, not a terminal refund** (`rl/train.py`'s
  `deferred_charges`). Charges are computed and attributed per-Tap exactly as
  before, but held for the whole game and applied at the terminal flush only
  if that seat won; on a loss they are simply never written, leaving the
  trajectory bit-for-bit identical to one that burnt nothing. A refund would
  *not* be equivalent: PPO trains on GAE advantages, where a charge written
  at step `t` lands in `delta_t` immediately while a terminal refund reaches
  it only through GAE's backward recursion, discounted by
  `(gamma*gae_lambda)^k` (`0.9405^k` at `gamma=0.99`/`gae_lambda=0.95` — ~11%
  surviving 40 steps back), which would leave early burns in a long losing
  game mostly un-cancelled.

  **The `q` risk, now measured.** Removing `q` removed the only penalty for
  hoarding anywhere in the reward. Over 20,065 games/deck the bet held:
  cleanup-discard turns per game came out at `0.87`/`2.39`/`3.00`/`0.10`
  (rakdos/dmir/elves/mono_red) against v4's `7.75`/`10.09` on the two worst
  decks — hoarding got *better* without its penalty, because it was a symptom
  of passivity rather than an independent problem.

  `deploy_reward_v6` (2026-08-12) supersedes `deploy_reward_v5`, moving
  **one** constant: `mana_burn_weight` `1.5 → 0.5`. Band, winner-only
  charging, curve shape (`c=2.9`/`p=4.0`) and `game_penalty_cap` (`1.5`) are
  byte-identical, so the guarantee is unchanged (worst win `-0.5` vs every
  loss `-1.0`).

  The problem it fixes is **saturation**: the whole-game cap was not bounding
  an occasional disaster, it was the normal case. At v5's 20,065-games/deck
  checkpoint `dmir_terror` exhausted the cap in **71%** of games and `elves`
  in **64%**, roughly two thirds of the way through each — so in most games
  the final third burnt mana entirely free of charge. A penalty that is
  already maxed out is a flat toll, not a gradient. `dmir_terror`'s
  *uncapped* charge averaged `4.26` against a `1.5` cap: **73% of the
  computed penalty was clipped away** before reaching the buffer.

  **Why not just raise the cap** (the obvious move): the guarantee fixes a
  hard ceiling at `cap < 2.0`, so `1.5` is already 75% of the maximum the
  shape can express and the remaining 25% *is* the safety margin. Spending it
  buys almost nothing — at weight `1.5`, `cap=1.8` moves `dmir_terror` from
  64% to 62% saturating while cutting the worst-win-to-loss margin from
  `0.50` to `0.20`. v3's collapse happened *while its ordering guarantee
  held* (an optimization failure, not a reward-ordering bug), so that margin
  is load-bearing against exactly the pathology v5 was built to fix.

  **Lowering the weight is not weakening the penalty.** The delivered tax
  barely moves (`dmir_terror` `1.13 → 0.91` mean charge/game) because the cap
  was already discarding most of it; what changes is how much is
  *proportional to actual waste* — clipped fraction `73% → 36%`, saturation
  `64% → 42%`, and when the cap does die it dies at 81% through the game
  instead of 66%. Per-turn charge by pips burnt: `1→0.007`, `2→0.092`,
  `3→0.267`, `4→0.392`, `5→0.449`, asymptoting toward `0.5`. One 5-pip turn
  now costs 30% of the whole-game budget instead of 90%.

  `deploy_reward_v1`-`v5` are all kept unchanged for reference/comparison.
- **Win condition**: the engine's real one — an opponent's life total hitting
  0, or a player decking out. There is no separate termination heuristic.

---

## Training-ops UI (`src/webapp/`)

A local Flask web app for starting, stopping, configuring, and watching
training runs from a browser instead of hand-building `run_league.py`
invocations, plus a game replay viewer (below) for
stepping through a logged game's board state.

```
cd src/webapp
python app.py          # http://127.0.0.1:5000 -- localhost only, no auth
```

`/` is a landing page linking to the two tools; the training-run panel
described below lives at `/train`, the replay viewer (below) at `/replay`.
Each page links back to `/` and to the other tool.

- **Runs are plain subprocesses.** Starting one spawns `run_league.py` with
  fully explicit CLI flags built from whatever's in
  the form — never a `--run-config`/`--league-config` *path*. The league
  form is generated by introspecting `rl.league_cli_spec.build_arg_parser()`
  directly (`webapp/runs.py`'s `argspec_from_parser`) — a torch-free module
  holding the same `build_arg_parser()` run_league.py's own CLI uses, so the
  webapp never pays run_league.py's full torch/rl.* import cost just to read
  its flag spec — so the form always matches the script's real CLI with no
  hand-maintained duplicate field list to drift out of sync as flags change.
- **Fields are grouped by run_league.py's real modes**, in collapsible
  sections (`rl.league_cli_spec`'s `LEAGUE_GLOBAL` / `LEAGUE_MODES`,
  re-exported from `webapp/runs.py`), so a field
  a mode doesn't read never shows up while configuring it — e.g. **League
  training** never shows `--matchup`, and **Matchup training**/**Eval** never
  show `--roster` or the auto-sizing fields. A few fields deliberately appear
  in more than one section (e.g. `train-deck-only`/`train-mulligan-only` matter
  to both League and Matchup training, which both actually train); `league_name`/
  `seed`/`log` apply to every mode and stay always-visible above the sections
  instead. The form only ever shows a *reduced* parameter surface in the first
  place — see "Reduced parameter surface" below.
- **`training_configs/*.json` are optional preconfigurations, not live
  config.** Picking one from the dropdown copies its values into the
  League-training section's fields client-side (e.g. `league_main.json`'s
  `total_games: 3000` fills the Total Games field) — the fields stay
  ordinary, editable inputs after that. Once "Start run" is clicked, the
  values on screen are the whole story; the JSON file itself is never
  referenced again.
- **League training auto-escalates to `--total-games`.** Leaving
  `--n-iterations` blank there takes `run_league.py`'s own auto-sizing path,
  which (by design) only ever plays ONE batch of its own doubling ladder per
  invocation — the same "start tiny, verify, double" behavior the `/train`
  skill drives by hand across many separate invocations
  (`rl.league_runner._next_batch_games`). The webapp automates that loop itself
  (`RunManager._escalating_loop`): after each batch it re-invokes the same
  command until this league's cumulative games/deck reaches the target, a
  batch comes back unhealthy, or Stop is clicked. Health is judged from log
  *content* (a `session N done` line, no `Traceback`), never the subprocess
  exit code — a parallel run (`--n-workers > 1`) reliably exits 1 on Windows
  from `ProcessPoolExecutor` teardown even on full success, exactly the
  false-failure the `/train` skill already works around by grepping the log.
  Setting `--n-iterations` explicitly (a forced one-off size, never fed back
  into the doubling sequence) or using Matchup/Eval mode always runs as a
  single batch instead. The runs table shows live progress
  (`cumulative/total games/deck`, batch count) for an escalating session.
- **Stop** kills the whole process tree (Windows: `taskkill /T /F`), not
  just the parent — `--n-workers > 1` spawns a `ProcessPoolExecutor`, and
  terminating only the parent would orphan its worker processes. For an
  escalating session, it also stops the next batch from being queued.
- **Each run gets its own folder**, `logs/<timestamp>-<script>-<short-id>/`
  (`RunManager._run_dir`), holding `stdout.log` and, if the form's `log`
  field was filled in, `event_log.json` — replacing the old scheme of a
  hand-typed `--log` path plus a same-named stdout file off in
  `logs/webapp_runs/`. Filling in `log` still just means "log this run";
  the path itself is generated, not taken literally, so every logged run
  ends up somewhere the replay viewer's server-side browser (below) can
  find it.
- **Logs** stream live over Server-Sent Events, reading that run's
  `stdout.log` — one continuous file per run, so an escalating session's
  log runs straight through every batch with a `=== batch N ===` marker
  between them; a finished run's log stays readable afterward.
- **Known limitation**: run tracking (start/stop/status) only works for
  runs started by the currently-running server process — restarting the
  server orphans any run still in flight (it keeps training to completion
  untouched, just no longer stoppable/pollable from the UI). The on-disk
  registry (`logs/webapp_runs/registry.json`, unchanged location — only
  the per-run stdout/event logs it points to moved) still shows its
  history and log once it's done.
- **CPU contention**: training already saturates every core at
  `--n-workers 6`; the UI warns (not blocks) before starting a second run
  while one is already active.

### Reduced parameter surface

`run_league.py`'s CLI/config surface was audited 2026-07-31 for knobs that
were either arbitrary (offered with no real justification) or miscoupled
(free-standing when they should have been derived from another value).
Three were removed/derived, three were kept as-is:

- **`--games-per-iteration` — removed, now always `max(1, n_workers)`.**
  `collect_rollout_league_parallel` splits games across workers via plain
  `n_games // n_workers` — a value smaller than `n_workers` silently starves
  the remainder (zero games, never even submitted). One game per worker is
  the simplest value that never under-provisions; benchmarked against 2x/3x
  `n_workers` (`src/benchmarking/training_run.py`) and it was also the
  *fastest* measured on this machine (2x/3x only add PPO-update cost — a
  bigger buffer to update on — without adding real collection parallelism
  once every worker already has work).
- **PPO minibatch ramp (`--batch-size-start`/`-cap`/`-steps`) — removed,
  hardcoded** (32 → 2048 over 6 steps, `rl.train.batch_size_for_iteration`'s
  own defaults). The *schedule* is a real, citable technique (Smith et al.
  2017 — grow batch size instead of decaying the learning rate), but the
  code's own prior comment on these flags admitted nobody had ever actually
  overridden them in practice. Follows the same pattern the codebase already
  uses for equally-important hyperparameters that were never exposed as
  flags at all (`horizon=120`, the PPO/mulligan learning rates). Tracked
  against cumulative games/deck since 2026-08-06 (`progress.json`), not the
  session-local iteration count it used to reset against at the start of
  every separate `run_league.py` invocation — see the function's own
  docstring. The PPO entropy coefficient (`rl.train.ent_coef_schedule`, also
  added 2026-08-06) follows the identical cumulative-games shape, annealing
  0.02 → 0.005 instead of a single fixed `ent_coef=0.01` for a run's whole
  life — see `TRAINING_IMPROVEMENT_OPTIONS.md` section 4 for the entropy-
  collapse data that motivated it, and `rl.ppo.ppo_update`'s own `target_kl`
  parameter (a per-epoch trust-region early stop, also new) for the other
  half of that fix.
- **`--max-batch-size` — removed entirely, no replacement cap.** Its job was
  protecting the auto-sizing doubling ladder's safety property (never jump
  from a small, verified-healthy batch straight to a huge one) — but that's
  now provided by whatever repeatedly re-invokes `run_league.py` and
  health-checks between calls (the `/train` skill, or the webapp's
  auto-escalation loop above), which has to exist for the ladder to mean
  anything regardless. A second, hand-picked ceiling on top of that was
  redundant, and every prior value (1024, 2048, 4000 across the three
  `training_configs/league_*.json` files) was picked ad hoc with no
  principled basis.

Kept as real, per-run decisions — not derivable from anything else:
`--total-games` (the actual training-size target), `--n-workers` (hardware-
dependent), `--checkpoint-opponent-rate` (deliberately owner-controlled —
see `rl/league.py`'s own design writeup), `--games`/`--seed` (direct user
choices for matchup/eval), `--n-iterations` (a documented debug escape
hatch). `--snapshot-every`/`snapshot_every_games` (~200 games between
snapshots) has a real but looser rationale and was left alone.

### Game replay viewer (`/replay`)

Step through a logged game's board state one event at a time. MVP scope:
retroactive viewing of an already-completed `--log` file only (no live game
viewing).

- **The webapp parses the raw event-log JSON directly** — no intermediate
  replay format involved. `replay_engine.py`'s `GameReducer` folds the event
  stream (the same `state.log_event` records) into one board-state snapshot
  per event: life totals, mana pools, hand/battlefield/graveyard/exile
  contents, and the stack (name-based zone identity tracking, DFC face
  reverts, aura-orphan handling, same-phase-recast identity). The stack is
  tracked as a single shared ordered list (top = last, matching
  `GameState.stack`); the pregame mulligan sequence is **not** netted into
  one summary step — see below.
- **A hamburger button fixed at the top-left opens a drawer** (link home,
  open-a-new-file, browse-server-logs, and the game list — replacing an
  earlier plain dropdown) that overlays the board without disturbing the
  scrub position underneath. **File selection is a native browser file
  picker**, reachable from the drawer's "Open new file". Pick any `--log`
  JSON file from disk; the browser reads it and posts the content to the
  backend, which returns that file's game list (one file can hold an entire
  round-robin `--eval` run) before reducing any board state, then reduces
  just the selected game.
- **"Browse server logs" (local-only)** lists every
  `logs/<run>/event_log.json` a webapp-launched run has written
  (`GET /api/replay/runs`, newest first), so a run started from `/train`
  doesn't need its output file hunted down by hand afterward — clicking an
  entry fetches it (`GET /api/replay/runs/<name>/raw`) and feeds it through
  the exact same client-side flow as a picked file. `app_public.py` (the
  publicly-hostable subset, see its own section below) has no equivalent
  route — the hosted viewer stays file-picker-only.
- Each game in the list is labeled from its own
  `deck_a`/`deck_b` fields (`rl.league_runner`'s `_write_event_log` stamps
  every game with which pairing it actually was) as `"deck A vs deck B (game
  N)"` — the `(game N)` disambiguates repeat games of the same pairing (a
  double round-robin plays each one twice) — rather than an unlabeled
  `"game 7"` a round-robin log's many different pairings can't otherwise be
  told apart by. Logs written before this field existed still work, falling
  back to the old file-level `game — game N` label. The drawer's game list
  is a client-side search box over the full per-file index (cheap even for a
  multi-thousand-game log) paginated 50 at a time, rather than one giant
  dropdown.
- **Both hands are always fully visible**, and the stack is always visible —
  this is post-hoc review of a finished game, not live play, so there's no
  hidden-information concern.
- **Card art is hotlinked from Scryfall's image endpoint** per card name (no
  local caching, nothing committed to the repo, no asset pipeline) — the
  browser handles image caching on its own. Hovering any card thumbnail pops
  up a larger version for readability.
- **Cockatrice-style board layout**: each player's zones stack vertically, with
  a right-hand rail (library/graveyard/exile/mana pool) per player. Zone order
  is mirrored around the middle so the two creature rows face off (top to
  bottom: P0 hand, P0 lands, P0 other, P0 creatures, P1 creatures, P1 other, P1
  lands, P1 hand) — hand is always the zone nearest that player's own outer
  edge. Graveyard and exile collapse to a pile showing the most-recently-added
  card's art (hover to see every card); library only ever shows a card count,
  never identities, since it's a hidden zone. Tapped permanents render rotated
  90°. Auras nest in a small stack peeking out beneath the permanent they
  enchant (resolved to the target's actual controller, so a Pacifism-style
  aura enchanting the opponent nests under their creature, not the caster's
  side).
- **The stack column (far left) is split in half, one per player**, each
  roughly the height of that player's own row, Cockatrice-style — the
  underlying model is still ONE shared LIFO stack (`GameReducer`'s own
  `self.stack`, real rules fidelity per this file's module docstring); the
  split is purely how the *viewer* groups entries by controller. Each entry
  shows card art, not just a name.
- **Far-right panel: the agent's top-5 candidate actions at the current
  decision**, split in half exactly like the stack column (one half per
  player) since a `decision_weights` step always has exactly one deciding
  player (`step.active_player_idx`, from the event envelope's `active_idx`
  -- `_seat_step`'s docstring guarantees `active_idx == the deciding seat`
  for its whole duration). The deciding player's half shows the top-5
  candidates by the network's post-mask probability, the chosen one
  highlighted, with the value estimate below; the other half just shows "no
  decision data." Opt-in instrumentation, off by
  default: `rl/agent.py`'s `_seat_step` (main policy) and
  `rl/mulligan.py`'s `decide()` (both mulligan branches) log a
  `decision_weights` event only when `state.event_log is not None` (i.e.
  `--log` eval/matchup runs, never blanket-on during ordinary self-play
  collection) and only for a real (non-forced, >1-legal-option) decision —
  reads values already computed in that decision's own forward pass, so no
  extra inference call and no effect on sampling. `rl/action_bridge.py`'s
  `pointer_kind(state)` names which targeting category (if any) governs a
  pointer candidate, mirroring `pointer_legal_mask`'s own dispatch.
  `replay_engine.py`'s handler formats each candidate (`fixed_label`
  verbatim; `pointer_identity`'s `{name, slot, controller}` into
  `"{name} (slot {slot}) (P{controller})"`, parts omitted when absent — a
  graveyard card or stack entry has no `slot`) — the logging side never
  bakes a string, matching every other event kind in this file.
- **Every mulligan-round draw, reject, and bottom-card pick is its own step**
  (owner directive), not netted into one "opening hands" summary — each is a
  genuinely separate decision (its own `rl/mulligan.py` forward pass), exactly
  the kind of moment this viewer exists to make visible, and each gets its own
  top-5 panel from the decision-weights logging above.
- **Library count is a simplified approximation** (owner-authorized): assume a
  real 60-card constructed deck, decrement once per card actually drawn
  (including every mulligan re-draw), and give the count back on a mulligan
  take/bottom put-back, so a multi-mulligan pregame still nets to the right
  remaining count. Non-draw depletion elsewhere (search, mill, impulse exile,
  scry-to-graveyard, non-mulligan put-backs) isn't counted — fine for
  "roughly how many cards are left," not a source of truth.
- Event kinds with no board-visible effect (`priority_flip`,
  `resolution_begin`/`complete`, `explore`/`animated` — the log entry
  doesn't carry enough to render unambiguously, etc.) still advance the
  scrubber with a plain label so the timeline never silently skips a step.
- **A creature's power/toughness badge tracks its CURRENT effective stats,
  not just what it printed on entry.** `game.effects.state_based`'s
  `check_state_based_actions` (already scanning every creature each priority
  round, and also called once more from `cleanup_step` right after clearing
  damage/until-EOT effects, so a pump wearing off at cleanup logs its own
  drop immediately instead of appearing to last into the opponent's whole
  next turn) recomputes each one's `permanent_power`/`permanent_toughness` —
  folding in `+1/+1`/`-0/-1` counters, until-EOT pump, attached Auras,
  animate/transform, and conditional static-self boosts — and logs a
  `stats_changed` event whenever it moved off what was last logged. The
  viewer keeps the entering, printed stats alongside as `base_power`/
  `base_toughness` and colors the badge green/red when the live value is
  above/below it, so a charged-up (or shrunk) creature is visible at a
  glance instead of stuck showing its zone-entry numbers.
- **`pass` is dropped entirely, not just given a no-op step** (owner-authorized
  simplification): a priority pass carries no information beyond what already
  gets its own step — the stack's top item resolving or the phase advancing —
  so showing it separately was pure noise, often a large fraction of a game's
  total step count.
- **A phase with nothing but passes in it is skipped straight through to
  whatever phase actually has something happen** (same owner authorization):
  `phase_change` is buffered rather than shown immediately, since whether it's
  worth a step isn't known until either a real event during that phase
  surfaces it, or the next `phase_change`/`turn_start` arrives with nothing
  having happened — an empty phase, dropped rather than shown as a bare
  "— Upkeep —" the player never actually did anything in.
- **`decision_weights` is buffered right alongside the phase header it
  belongs to** (owner directive, 2026-08-19, refining the collapse above):
  `_log_decision_weights` fires for any decision with more than one legal
  option *regardless of whether the chosen action was pass*, so a player
  merely considering — and declining — an action was flushing an
  otherwise-empty phase on its own, defeating the collapse for almost every
  phase in a real game. Queued decision panels now ride along with the
  phase's buffered header: shown together if a real action follows later in
  the same phase, discarded together if it doesn't. A decision that led to
  an actual board change still always shows — only pass-only phases vanish.
  Arrow-key/scrubber navigation lands on the next phase with real content,
  e.g. an empty Main Phase 1 steps straight to Attacks.
- **A persistent Turn/Phase/active-player line** above the scrubber
  (`#turn-phase`) reads e.g. "Turn 5 — main1 — P0's turn, P1 acting" —
  `turn_player_idx` (whose turn it structurally is) and
  `active_player_idx` (whoever currently holds priority) are shown
  separately since they diverge whenever a player is doing something (a
  response, a block) on the other player's turn; identical when they
  match, so an uneventful step just reads "P0's turn".
- Deferred, not in MVP scope: live re-inference against an arbitrary
  checkpoint (the decision-point overlay above has shipped).

### Public hosting (`app_public.py`)

`src/webapp/app_public.py` is a deploy-only entrypoint exposing **just** the
`/replay` viewer and its two `/api/replay/*` endpoints — never `/train`,
which starts real OS subprocesses from POST'd args and must stay
localhost-only (see `app.py`'s module docstring). It has no filesystem
access of its own: both endpoints take raw log JSON text in the POST body
(the browser reads the file, not the server), and `replay_engine.py` has no
imports beyond the stdlib, so hosting it needs only
`src/webapp/requirements-public.txt` (`flask` + `gunicorn`), not the repo's
full CUDA-pinned `requirements.txt`. `render.yaml` at the repo root deploys
it as a Render free-tier web service (`gunicorn --chdir src/webapp
app_public:app`).

---

## Setup

```
pip install -r requirements.txt
```

`requirements.txt` pins a CUDA (cu128) PyTorch build plus `numpy`, but
**training runs on CPU** and doesn't need a GPU — the pinned wheel just
prevents pip from silently swapping to a CPU-only build on a machine that does
have one. The replay converter additionally needs `grpcio-tools` (for
protobuf codegen). The training-ops UI (`src/webapp/`) additionally needs
`flask`.

---

## Tests

A real `pytest` suite lives under `tests/`, mirroring `src/`'s layout (e.g.
`src/game/mana.py` -> `tests/game/test_mana.py`). `pyproject.toml` sets
`pythonpath = ["src"]` and `testpaths = ["tests"]`, so tests import modules the
same way production code does and `pytest` just works from the repo root —
no `cd src` needed.

```
pip install -r requirements.txt   # pytest included
pytest                             # whole suite (engine + catalog + effects +
                                    # resolution + drl_env + rl + webapp)
pytest -m "not slow"               # fast tier only: game engine/catalog/effects,
                                    # deterministic, no torch, sub-second
pytest -m slow                     # rl/ tier only: imports torch, plays real
                                    # mini-games, runs PPO/REINFORCE updates,
                                    # seconds per test
```

The whole suite runs together in well under a minute.
No pre-commit hook wired up yet — running the suite is still manual.

`benchmarking/training_run.py` measures the real league loop under different
collection configs (`seq`, `mp<N>`, plus a `--batched` toggle) over a fresh
untrained stack — a benchmark, not a test.

---

## Generated artifacts (gitignored)

`checkpoints/` (trained weights + `vocab.json`) and `logs/` (event logs from
`--log` runs) are gitignored — regenerable by rerunning training. `.gitignore`
also lists `models/`, `reports/`, and `graphify-out/`.

---

## Project status

The engine and DRL architecture are the current, active surface.

- **The DRL pipeline is 2-player only.** The engine tolerates a no-opponent
  (1-player) configuration (useful for isolated card-behavior tests), but the
  token architecture always encodes an opponent seat.
