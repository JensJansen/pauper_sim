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
| **DRL system** | `src/rl/` | A shared Set-Transformer + FiLM perception stack, per-deck trunk/critic/pointer-network action heads, and a PPO self-play + league training loop. |
| **Decks** | `data/*.txt` + `league_decks.json` | An 11-deck roster (see below). |
| **Training drivers** | `src/run_pretrain.py`, `src/run_league.py` | Two-phase pipeline: pretrain & freeze the shared stack, then train every deck continuously in a league. |
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
- **One policy per deck, sharing one perception stack.** The embedding +
  attention layers are pretrained across all decks and frozen; each deck then
  trains its own lightweight head on top.
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
    mana.py                  Float-first mana: mana abilities float into a per-player
                             pool at any priority window (never the stack), paying a
                             cost spends floated pool mana explicitly, pure pool-
                             affordability legality checks (pool_can_pay).
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
                             pooling heads) and FiLM — the shared perception stack.
    deck.py                  DeckNetwork: per-deck trunk + critic + pointer-net action
                             head on top of a shared stack.
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

  run_pretrain.py          Pretrain + freeze the shared stack.
  run_league.py            Thin CLI wrapper (arg resolution + main()) around
                           rl/league_runner.py.
  report_metrics.py        Plain-text summary of a league's metrics.jsonl (entropy/loss
                           trends, win rate vs. archived past selves).
  benchmarking/            training_run.py (benchmarks the real league loop under
                           different collection configs) + _common.py (path/stdout
                           bootstrap it imports for its side effect).
  webapp/                  Local Flask UI: app.py (routes) + runs.py (subprocess/
                           registry logic) + static/*.html (landing, training-ops,
                           replay viewer -- no build step). See its own section below.

data/                      Decklists (*.txt) + league_decks.json roster.
checkpoints/               Trained weights + vocab.json (gitignored; see below).
logs/                      Game event logs from --log runs (gitignored).
```

**Run scripts from `src/`.** The driver/training scripts (`run_pretrain.py`,
`run_league.py`, `benchmarking/*`) use relative paths like `../data` and
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
- **Mana.** A mana pool that must be spent by explicit actions, Tron-land
  detection, and flexible/filter mana sources.
- **Card mechanics.** Madness, plot, flashback, auras/bestow, scry/surveil,
  fetch/search, initiative/Undercity, mulligans, token generation (Blood,
  Robot, Warrior, Eldrazi Spawn, Food, Clue, Treasure, and more), affinity/
  delve/escape cost reductions, state-based actions, decking out, and
  life-total win checks.
- **Hidden information is respected.** Hand and library CONTENTS stay hidden;
  only public zones (battlefield/graveyard/stack/exile) are ever tokenized.
  Their SIZE isn't hidden in real Magic (either player can count a library or
  a hand), so it's surfaced to the agent separately, as a scalar.

The engine's effect functions defensively handle a no-opponent (1-player)
configuration — useful for a card-level unit test that doesn't need a second
seat — but the active surface, and everything the DRL system drives, is
2-player.

---

## The DRL system (`src/rl/`)

The observation has two parts. The first is a **variable-length set of card
tokens**, one per public-zone card for both players. Each token = a learned
**identity embedding** (via `CardVocab`) concatenated with a deterministic
**static feature vector** (mana cost, type, base P/T, keywords) plus
**dynamic per-instance state** (tapped, effective P/T, combat commitments,
whether currently targeted by a spell/ability on the stack — mine or the
opponent's, including a spell targeting another spell — zone, mine/theirs
side flag).

The second is a **scalar vector** (`rl/agent.py`'s `_scalar_features`) of
non-tokenized globals: turn number, lands-played, mulligans taken, whose
turn it is, each player's floating mana pool (by color), phase, life
totals, each player's library size, the opponent's hand size, and whether
anything on the stack currently targets either player directly (a burn
spell to the face has no token to carry a bit on, so that one case is
scalar-only). Library/hand size, floating mana, and a declared target are
all public knowledge in real Magic, unlike hand/library *contents* — see
"Hidden information is respected" above.

The network is split into a **shared** stack and a **per-deck** head:

- **`SetTransformer` (shared, `rl/arch.py`)** — embeds + projects tokens,
  runs a joint self-attention encoder over *both* sides' tokens (so a token
  can attend across the mine/theirs boundary), then pools with two
  independent learned-query heads: a "mine" summary (trunk input) and a
  "theirs" summary (FiLM conditioning input). Pre-norm transformer for RL
  stability. Uses `torch.nn.MultiheadAttention`/`TransformerEncoderLayer`
  directly rather than hand-rolling attention.
- **`FiLM` (shared)** — turns the "theirs" summary into per-layer
  (gamma, beta) modulations of the trunk, chosen over concatenation.
- **`DeckNetwork` (per-deck, `rl/deck.py`)** — a small trunk + critic +
  a **pointer-network action head**. The action space is the union of a
  **fixed table** of non-targeting actions (play land, cast X, pass, mana
  payments, mulligans, …) and a **pointer-scored** set of targeting actions
  (attack / assign-blocker / choose-target), scored against the
  post-attention token representations. Both halves feed **one combined
  softmax**, so a masked-categorical sample over the true legal set is
  correct.
- **Pregame mulligan model (per-deck, `rl/mulligan.py`)** — a separate small
  head on the same shared stack that owns every pregame keep/mulligan/bottom
  decision. A `SeatAgent` (`rl/agent.py`) routes pregame decisions to it and
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
small deliberately (a stale snapshot is a weak opponent), but the archive
keeps deep history around so a deck's win rate against its own much-older
self stays measurable for the life of a run, not just its last ~1,600 games.
See **Instrumentation** below.

---

## Training pipeline

Both phases share one vocabulary/embedding table across all decks
(`checkpoints/vocab.json`, append-only so old checkpoints stay valid). Run
everything **from `src/`**.

### 1. Pretrain & freeze the shared perception stack

```
cd src
python run_pretrain.py <n_iterations> <games_per_iteration>           # build up the shared stack
python run_pretrain.py <n_iterations> <games_per_iteration> --freeze  # freeze once satisfied
```

Runs mirror self-play for **every** deck each iteration, flowing gradients
from a throwaway per-deck head into the one shared `SetTransformer`+`FiLM`.
Checkpoints to `checkpoints/pretrain_shared_stack.pt` after every session
(resumable). `--freeze` additionally writes
`checkpoints/shared_stack_frozen.pt` — run it only once you're satisfied.

### 2. League training

```
cd src
python run_league.py --n-iterations N --snapshot-every 15 --n-workers 6
```

Every deck trains every iteration against a resampled league opponent, on top
of the frozen shared stack. Each deck's live net/optimizer persists in
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

`--games-per-iteration` isn't a flag — it's always derived as `max(1, n_workers)`
(one game per worker; a smaller value used to silently starve some worker
processes of any work at all). The PPO minibatch ramp (32 → 2048 over 6 steps)
and the old `--max-batch-size` auto-sizing cap are likewise no longer
parameters — see `rl/league_runner.py`'s (`_next_batch_games`) and
`run_league.py`'s (`main()`) own comments at each removal site for why.

Training runs on **CPU** by design — the model is small (~200–250K params) and
a batch-size sweep found no GPU crossover at this size.

### Instrumentation

Every league session (`_run_session`, both league and `--matchup` modes)
appends to `checkpoints/<league>/metrics.jsonl`, one JSON line per record:
- `kind: "ppo"` — per deck per iteration: `policy_loss`, `value_loss`,
  `entropy` (computed every call, previously never recorded past stdout),
  `buffer_size` (transitions collected that iteration) and `batch_size` side
  by side, so a saturated minibatch ramp (`batch_size >= buffer_size`, at
  which point `ppo_update` stops sub-batching and just runs `n_epochs`
  full-batch steps) is directly visible instead of assumed.
- `kind: "mulligan"` — per deck per iteration REINFORCE loss/n.
- `kind: "vs_history"` — once per session per deck (league mode only): the
  live net played against its own oldest still-active snapshot and, once one
  exists, its oldest **archived** snapshot (`rl.league`'s eviction archive,
  see above) — a direct win-rate-vs-past-self measurement, not an inference
  from loss curves. Skipped (empty) automatically until a deck has been
  through at least one snapshot cycle, so it costs nothing during a run's
  cheap early sessions.
- `kind: "vs_gauntlet"` / `kind: "vs_heuristic"` — the gauntlet mechanism's
  two tiers, both EXTERNAL to this league's own self-play history (unlike
  `vs_history`'s snapshots). See **Gauntlet** below.

Every game the engine plays (any collection path, since `collect_rollout` is
the one game loop) also gets one `game_over` event appended to its own
`event_log` (`winner`, `turn_won`) — the outcome was previously never written
to the log stream at all, only reconstructible by replaying `life_change`
deltas by hand.

`python report_metrics.py <league_dir>` prints a plain-text summary (entropy
trend, latest losses, vs-history win rates) read from `metrics.jsonl` — stdlib
only, no plotting dependency.

### Gauntlet

`vs_history` (above) and PFSP-weighted sampling both only ever measure a
league against **its own** self-play history — real signal, but it can't
tell "genuinely improving" apart from "well-adapted to beating a closed
population that co-evolved with itself." The gauntlet is two EXTERNAL
reference points, outside that history entirely:

- **Tier 2 — an independently-trained twin population** (`training_configs/
  run_gauntlet.json`): the SAME roster, run mechanics, and frozen shared
  stack as the main league, but its own checkpoint tree (`league_name:
  "4_deck_subleague_gauntlet"`) that never plays against the main league's
  live nets. Trained to a fixed depth (~8,000 games/deck) then left alone —
  no code-level "freeze," just stop invoking further sessions against that
  `league_name`. Two runs from an identical algorithm/config still diverge
  into different regions of strategy space purely from a different
  nondeterministic training trajectory — so a blind spot the WHOLE main
  population shares (the risk PFSP and `vs_history` can't rule out) is far
  more likely to show up against a genuinely independent opponent than
  against anything drawn from the league's own history. Wired via a config's
  `gauntlet_league_name` field (`rl.league_runner._run_eval_vs_gauntlet`, once per
  session per deck, only once the twin population has a checkpoint for that
  deck).
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
picked up by `report_metrics.py` the same way as everything else) and cost
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

- **Rewards** (`rl/rewards.py`): **pretraining** uses
  `action_count_win_reward_200_floor02` — loss/draw → `0.0`, win → `1.0` down
  to a `0.2` floor scaled by the *winning seat's* action count (so a policy
  can't pad a turn with free actions to inflate its reward). **League** play
  uses `deploy_reward_v2`: a flat `1.0` on any win minus `q` (no efficiency
  scaling — the earlier `deploy_reward_v1`'s efficiency band induced an
  action-space-minimization pathology where policies learned to shrink their
  own board to "win in fewer actions"), and exactly `-q` on a loss. `q` is a
  terminal "sloppiness" penalty shared by both bands: a saturating
  (Hill-function) curve over `PlayerState.cleanup_discard_turns` (cards
  hoarded past hand size) — near-zero for a couple of stray discards, severe
  by ~6 cumulative, asymptoting toward but never reaching `1.0`, so every win
  still strictly outscores every loss no matter how sloppy either was.
  `deploy_reward_v2` additionally wraps that terminal score with
  `with_mana_mistake_penalty`: a *dense*, per-transition penalty (not folded
  into `q`, deliberately — a terminal-only signal blurs blame for a mistake
  across the whole game, e.g. across every subsequent cast of a card whose
  cost the policy simply mis-tapped for once) for mana burnt at a phase
  boundary (rule 500.4) that `game.turn._empty_mana_pools` could find no
  justification for: nothing was paid toward a cast/ability that phase,
  nothing triggered as a result of the mana-producing action itself (e.g.
  sacrificing a mana source purely for its own combat-trick trigger), and
  nothing was legally castable with the floating pool at the moment it was
  lost (`PlayerState.mana_mistake_burn`, `GameState.on_mana_burn`).
  `PlayerState.mana_burnt_total` (the *unconditional* pips-lost tally) no
  longer feeds any reward — it remains a raw diagnostic for logging/viz. The
  **mulligan model** trains on its own reward (`rl/mulligan.py`): win payout
  minus a convex (quadratic) per-mulligan penalty.
- **Win condition**: the engine's real one — an opponent's life total hitting
  0, or a player decking out. There is no separate termination heuristic.

---

## Training-ops UI (`src/webapp/`)

A local Flask web app for starting, stopping, configuring, and watching
training runs from a browser instead of hand-building `run_league.py`/
`run_pretrain.py` invocations, plus a game replay viewer (below) for
stepping through a logged game's board state.

```
cd src/webapp
python app.py          # http://127.0.0.1:5000 -- localhost only, no auth
```

`/` is a landing page linking to the two tools; the training-run panel
described below lives at `/train`, the replay viewer (below) at `/replay`.
Each page links back to `/` and to the other tool.

- **Runs are plain subprocesses.** Starting one spawns `run_league.py` or
  `run_pretrain.py` with fully explicit CLI flags built from whatever's in
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
- **Logs** stream live over Server-Sent Events, reading the same kind of
  log file the `/train` skill already writes by hand
  (`logs/webapp_runs/<run_id>.log`) — one continuous file per run, so an
  escalating session's log runs straight through every batch with a
  `=== batch N ===` marker between them; a finished run's log stays readable
  afterward.
- **Known limitation**: run tracking (start/stop/status) only works for
  runs started by the currently-running server process — restarting the
  server orphans any run still in flight (it keeps training to completion
  untouched, just no longer stoppable/pollable from the UI). The on-disk
  registry (`logs/webapp_runs/registry.json`) still shows its history and
  log once it's done.
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
  flags at all (`horizon=120`, the PPO/mulligan learning rates).
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

Step through a logged game's board state one event at a time. MVP scope per
`todo/game_visualization.md`: retroactive viewing of an already-completed
`--log` file only (no live game viewing).

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
  open-a-new-file, and the game list — replacing an earlier plain dropdown)
  that overlays the board without disturbing the scrub position underneath.
  **File selection is a native browser file picker**, reachable from the
  drawer's "Open new file". Pick any `--log` JSON file from disk
  (`logs/*.json`); the browser reads it and posts the content to the
  backend, which returns that file's game list (one file can hold an entire
  round-robin `--eval` run) before reducing any board state, then reduces
  just the selected game. Each game in the list is labeled from its own
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
  round) recomputes each one's `permanent_power`/`permanent_toughness` —
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
- Deferred, not in MVP scope: live re-inference against an arbitrary
  checkpoint (the decision-point overlay above has shipped) — tracked in
  `todo/game_visualization.md`.

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
