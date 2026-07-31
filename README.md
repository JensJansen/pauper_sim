# MTG-Subset Simulator + Attention-based Self-Play RL

A from-scratch **2-player Magic: The Gathering rules engine** for a curated
subset of cards, plus a **token/attention deep-RL system** that trains a
separate policy per deck by self-play and continuous league play against a
pool of historical opponents. The engine, the card catalog, the neural
architecture, the training loop, and a Cockatrice replay exporter all live in
this repo with essentially no framework dependencies beyond PyTorch.

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
| **Replay export** | `src/sim_replay_converter/` | Converts engine game logs into Cockatrice `.cor` replay files so games can be watched in Cockatrice's viewer. |

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
                             madness_and_plot, undercity (initiative), shared.
    resolution/              Multi-step decisions the MODEL makes one action at a time
                             (mulligan, scry/surveil, search/fetch, discard, targeting,
                             trigger ordering, ...): _core (begin/complete state machine)
                             + handlers (every concrete resolution kind).

  drl_env/                 Action-table / legal-mask machinery (a package, not a gym Env):
    _actions.py              build_action_table + per-action legal/execute predicates.
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
    action_bridge.py         Maps the network's combined (fixed + pointer) action space
                             back to real engine calls.
    train.py                 Rollout collection + PPO update; mirror & cross self-play;
                             parallel and in-process-batched rollout collection.
    pool.py                  Builds the shared vocab + per-deck action tables from the
                             league roster (data/league_decks.json).
    league.py                LeaguePool: historical opponent snapshots, sampling,
                             eviction, disk persistence.
    rewards.py               Reward functions (win/loss with a speed tiebreaker).

  run_pretrain.py          Pretrain + freeze the shared stack.
  run_league.py            League driver (and a --matchup direct-pairing mode).
  benchmarking/            training_run.py (benchmarks the real league loop under
                           different collection configs) + _common.py (path/stdout
                           bootstrap it imports for its side effect).
  sim_replay_converter/    JSON game log -> Cockatrice .cor replay (convert.py + a
                           vendored copy of Cockatrice's .proto files under proto/).

data/                      Decklists (*.txt) + league_decks.json roster.
checkpoints/               Trained weights + vocab.json (gitignored; see below).
logs/                      Game event logs from --log runs (gitignored).
```

**Run scripts from `src/`.** The driver/training scripts (`run_pretrain.py`,
`run_league.py`, `benchmarking/*`, `sim_replay_converter/convert.py`) use
relative paths like `../data` and `../checkpoints`, and the `rl.*` modules
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
turn it is, floating mana pool, phase, life totals, each player's library
size, the opponent's hand size, and whether anything on the stack currently
targets either player directly (a burn spell to the face has no token to
carry a bit on, so that one case is scalar-only). Library/hand size and a
declared target are all public knowledge in real Magic, unlike hand/library
*contents* — see "Hidden information is respected" above.

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

Training is **PPO self-play** (`rl/train.py`). Mirror matches pool both seats
into one buffer/update; cross-matchups give each net its own buffer, both
learning from every game. Rollout collection parallelizes across worker
processes (~3.2–3.5× on 6 physical cores).

The **league** (`rl/league.py`, `run_league.py`) keeps a rolling window of
historical snapshots per deck. Each game resamples an opponent two-level
uniformly: pick a deck (including the training deck itself, for mirror play),
then pick one of its snapshots (or its current live weights). No hardcoded
Stage-1/Stage-2 phases — cross-deck and cross-snapshot exposure grows as the
pool fills.

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
python run_league.py --n-iterations N --games-per-iteration 6 --snapshot-every 15 --n-workers 6
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
- `--batch-size-start` / `--batch-size-cap` / `--batch-size-steps` — the PPO
  minibatch schedule, doubling from start (32) toward the cap (2048) in a
  fixed number of steps (6) across the session.
- `--batched-collect` — collect every deck's games under the start-of-iteration
  weights, then do one PPO update per deck (cleaner on-policy, removes
  intra-iteration drift). Experimental; benchmark before adopting.
- `--no-opponent-salvage` — by default, when the sampled opponent is another
  deck's **live** net, that deck is also trained from its (on-policy)
  transitions this round at no extra collection cost; pass this to disable it.

Training runs on **CPU** by design — the model is small (~200–250K params) and
a batch-size sweep found no GPU crossover at this size.

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
  uses `deploy_reward_v2`: a flat `1.0` on any win (no efficiency scaling — the
  earlier `deploy_reward_v1`'s efficiency band induced an action-space-
  minimization pathology where policies learned to shrink their own board to
  "win in fewer actions"), and the same `0.25` floor decremented per turn the
  seat discarded to cleanup (punishing hoarding) on a loss. The **mulligan
  model** trains on its own reward (`rl/mulligan.py`): win payout minus a convex
  (quadratic) per-mulligan penalty.
- **Win condition**: the engine's real one — an opponent's life total hitting
  0, or a player decking out. There is no separate termination heuristic.

---

## Watching games in Cockatrice

`src/sim_replay_converter/convert.py` turns an engine JSON log (from a
`--matchup --log` or `--eval --log` run) into Cockatrice `.cor` replays so a
game can be watched in Cockatrice's built-in replay viewer:

```
cd src/sim_replay_converter
python convert.py <sim.json> <output_dir> [--game INDEX]
```

It auto-detects two log shapes (snapshot-diff and event-stream) and, on first
run, generates protobuf bindings from the vendored Cockatrice `.proto` files
(`proto/`, overridable via `COCKATRICE_PROTO_DIR`). Requires `grpcio-tools`.

One JSON file can hold many games — a `--matchup` run logs one deck pair, an
`--eval` run logs a whole round-robin (every pairing, with mirrors) in one
file — and each game's output filename/description is labeled with its own
deck pairing, not just the file's. `pytest tests/sim_replay_converter` checks
the converter against `logs/*.json` (skipped if none present) plus a few
fabricated-event checks that always run — rerun it after an engine change to
the event log shape.

---

## Setup

```
pip install -r requirements.txt
```

`requirements.txt` pins a CUDA (cu128) PyTorch build plus `numpy`, but
**training runs on CPU** and doesn't need a GPU — the pinned wheel just
prevents pip from silently swapping to a CPU-only build on a machine that does
have one. The replay converter additionally needs `grpcio-tools` (for
protobuf codegen).

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
                                    # resolution + drl_env + sim_replay_converter + rl)
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
