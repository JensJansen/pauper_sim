# MTG-Subset Simulator + Attention-based Self-Play RL

A from-scratch **2-player Magic: The Gathering rules engine** for a curated
subset of cards, plus a **token/attention deep-RL system** that trains a
separate policy per deck by self-play and continuous league play against a
pool of historical opponents. Everything — the engine, the card catalog, the
neural architecture, the training loop, and a Cockatrice replay exporter —
lives in this repo with essentially no framework dependencies beyond PyTorch.

The project started life as a single-deck "Tron assembly" probability
simulator; it has since grown into a general multi-deck engine and an
adversarial RL agent. (Some in-code docstrings still reference the older
1-player Tron pipeline and its now-deleted files — `harness.py`, `run.py`,
`lean_ppo.py` — see [Project status](#project-status).)

---

## What's here

| Piece | Where | What it is |
|-------|-------|------------|
| **Game engine** | `src/game/` | A self-contained MTG-subset simulator: zones, turn/phase loop, priority, the stack, mana, combat, and ~130 card/token effects. No ML dependency. |
| **Card catalog** | `src/game/catalog/` | Card definitions grouped by color. Decks are just decklists resolved against this shared catalog — adding a deck from already-implemented cards needs no code. |
| **Decks** | `data/*.txt` + `league_decks.json` | Five archetypes: `mono_red_madness`, `rakdos_madness`, `spy_combo`, `boggles`, `monster_tron`. |
| **DRL architecture** | `src/token_*.py` | A shared Set-Transformer + FiLM perception stack, per-deck trunk/critic/pointer-network action heads, and a PPO self-play + league training loop. |
| **Training drivers** | `src/run_pretrain.py`, `src/run_league.py` | Two-phase pipeline: pretrain & freeze the shared stack, then train every deck continuously in a league. |
| **Replay export** | `src/sim_replay_converter/` | Converts engine game logs into Cockatrice `.cor` replay files so games can be watched in Cockatrice's viewer. |

---

## Goals

- **A faithful-enough MTG-subset engine** to run real 2-player games:
  phases, priority passing, the stack, combat with attackers/blockers/damage,
  and a broad set of real card mechanics (madness, plot, flashback, bestow/
  auras, scry/surveil, fetch/search, token generation, mulligans, decking
  out, state-based actions, life-total win checks).
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
  game/                    The engine (package, ~9k LOC). Zero ML deps.
    state.py                 PlayerState + GameState: zones, turn/stack/
                             pending-resolution bookkeeping. Proxies "my
                             board" to whichever player has priority.
    turn.py                  Phase enum + turn loop (untap→...→cleanup),
                             priority rounds, 1p and 2p game drivers.
    mana.py                  Mana pool, tapping, cost payment, Tron lands,
                             flexible/filter mana sources.
    resolution.py            Pending-resolution state machine (mulligan,
                             scry/surveil, search/fetch, discard, targeting,
                             trigger ordering, madness decisions, ...).
    combat.py-adjacent       effects/combat.py: attackers, blockers, damage.
    effects/                 Card-effect building blocks: casting, combat,
                             stack, triggers, state-based actions, tokens,
                             madness_and_plot, stats, win_check.
    catalog/                 Card definitions by color (black/blue/green/red/
                             white/colorless/multicolor).
    registry.py              Unions every color catalog into CARD_DEFS +
                             EFFECT_REGISTRY.
    decklist.py, reporting.py, cards.py

  drl_env.py               Action-table / legal-mask machinery: turns a
                           decklist + EFFECT_REGISTRY into a legal action
                           space (no gym Env of its own anymore).
  rewards.py               Reward functions (win/loss with speed tiebreakers).
  terminated.py            Win-condition ("terminated_fn") functions.

  token_features.py        Per-card tokenization: CardVocab (stable card→index)
                           + static feature vectors + build_token_set.
  token_arch.py            SetTransformer (embeddings + self-attention + two
                           PMA pooling heads) and FiLM. The shared stack.
  token_deck.py            DeckNetwork: per-deck trunk + critic + pointer-net
                           action head on top of a shared stack.
  token_action_bridge.py   Maps the network's combined (fixed + pointer)
                           action space back to real engine calls.
  token_train.py           Rollout collection + PPO update; mirror & cross
                           self-play; parallel rollout collection.
  token_pool.py            Builds the shared vocab + per-deck action tables
                           from the league roster (data/league_decks.json).
  token_league.py          LeaguePool: historical opponent snapshots, sampling,
                           eviction, disk persistence.

  run_pretrain.py          Phase 4: pretrain + freeze the shared stack.
  run_league.py            League driver (and a --matchup direct-pairing mode).
  benchmark_*.py           Rollout-parallelism and PPO batch-size benchmarks.
  generate_regression_snapshot.py

  sim_replay_converter/    JSON game log → Cockatrice .cor replay.

data/                      Decklists (*.txt) + league_decks.json roster.
docs/                      Design/plan documents.
checkpoints/               Trained weights (gitignored; see below).
logs/                      Game event logs from --log runs (gitignored).
```

**Why the flat `src/` for the token modules?** They import each other
directly (`import game`, `import drl_env`, ...), which works because Python
adds a script's own directory to `sys.path`. The engine itself *is* a proper
package (`game/`); the driver/training scripts are kept flat because they're
only ever run from `src/`, never imported as a library.

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
  fetch/search, mulligans, token generation (Blood, Robot, Warrior, Eldrazi
  Spawn), state-based actions, decking out, and life-total win checks.
- **Hidden information is respected.** Hand and library contents stay hidden;
  only public zones (battlefield/graveyard/stack/exile) are ever encoded.

Every module ships an `assert`-based self-check runnable directly, e.g.
`python game/state.py` isn't wired that way but the token/engine-facing
modules are (see [Self-checks](#tests--self-checks)).

---

## The DRL system (`src/token_*.py`)

The observation is a **variable-length set of card tokens**, one per
public-zone card for both players. Each token = a learned **identity
embedding** (via `CardVocab`) concatenated with a deterministic **static
feature vector** (mana cost, type, base P/T, keywords) plus **dynamic
per-instance state** (tapped, effective P/T, combat commitments, zone,
mine/theirs side flag).

The network is split into a **shared** stack and a **per-deck** head:

- **`SetTransformer` (shared, `token_arch.py`)** — embeds + projects tokens,
  runs a joint self-attention encoder over *both* sides' tokens (so a token
  can attend across the mine/theirs boundary), then pools with two
  independent learned-query heads: a "mine" summary (trunk input) and a
  "theirs" summary (FiLM conditioning input). Pre-norm transformer for RL
  stability.
- **`FiLM` (shared)** — turns the "theirs" summary into per-layer
  (gamma, beta) modulations of the trunk, chosen over concatenation.
- **`DeckNetwork` (per-deck, `token_deck.py`)** — a small trunk + critic +
  a **pointer-network action head**. The action space is the union of a
  **fixed table** of non-targeting actions (play land, cast X, pass, mana
  payments, mulligans, …) and a **pointer-scored** set of targeting actions
  (attack / assign-blocker / choose-target), scored against the post-attention
  token representations. Both halves feed **one combined softmax**, so a
  masked-categorical sample over the true legal set is correct.

Training is **PPO self-play** (`token_train.py`). Mirror matches pool both
seats into one buffer/update; cross-matchups give each net its own buffer,
both learning from every game. Rollout collection parallelizes across worker
processes (~3.2–3.5× on 6 physical cores).

The **league** (`token_league.py`, `run_league.py`) keeps a rolling window of
historical snapshots per deck. Each game resamples an opponent two-level
uniformly: pick a deck (including the training deck itself, for mirror play),
then pick one of its snapshots (or its current live weights). No hardcoded
Stage-1/Stage-2 phases — cross-deck and cross-snapshot exposure grows as the
pool fills.

---

## Training pipeline

Both phases assume **five decks** (`data/league_decks.json`) and one shared
vocabulary/embedding table across all of them (`checkpoints/vocab.json`,
append-only so old checkpoints stay valid).

Run everything **from `src/`** (scripts use relative paths like `../data`,
`../checkpoints`).

### 1. Pretrain & freeze the shared perception stack

```
cd src
python run_pretrain.py <n_iterations> <games_per_iteration>        # build up the shared stack
python run_pretrain.py <n_iterations> <games_per_iteration> --freeze  # freeze once satisfied
```

Runs mirror self-play for **every** deck each iteration, flowing gradients
from a throwaway per-deck head into the one shared `SetTransformer`+`FiLM`.
`--freeze` writes `checkpoints/shared_stack_frozen.pt`. Checkpointed and
resumable between invocations — start small, watch for stalls, scale up.

### 2. League training

```
cd src
python run_league.py --n-iterations N --games-per-iteration 6 --snapshot-every 15 --n-workers 6
```

Every deck trains every iteration against a resampled league opponent, on top
of the frozen shared stack. Each deck's live net/optimizer persists in
`checkpoints/league/<deck>/live.pt`; snapshots live alongside. Resumable
across sessions.

Key flags: `--n-workers` (parallel rollout processes; default 6, no reliable
gain past physical core count), `--snapshot-every`, and a `--batch-size-*`
schedule (small→large across the session). Training runs on **CPU** by design
— the model is small (~200–250K params) and a batch-size sweep found no GPU
crossover; `--gpu-threshold` exists but defaults to off.

### Direct matchup (no league sampling)

```
python run_league.py --matchup DECK_A DECK_B --games 50 [--log path/to/games.json]
```

Runs a fixed pairing between two named decks, still updating and
checkpointing both. `--log` captures the engine's own event log for every
game as one JSON file — the input the replay converter consumes.

### Rewards & termination

- **Rewards** (`rewards.py`): win/loss with a speed tiebreaker. The league
  default is `action_count_win_reward_200_floor02` — loss/draw → 0, win →
  1.0 down to a 0.2 floor scaled by the *winning seat's* action count (so a
  policy can't pad a turn with free actions).
- **Termination** (`terminated.py`): league play uses `never_terminated` so
  the only win condition is the engine's real one (opponent to 0 life, or
  decking out). Other functions (Tron assembly, damage thresholds) are
  1-player heuristics kept from the earlier pipeline.

---

## Watching games in Cockatrice

`src/sim_replay_converter/convert.py` turns an engine JSON log into a
Cockatrice `.cor` replay so a game can be watched in Cockatrice's built-in
replay viewer:

```
cd src/sim_replay_converter
python convert.py <sim.json> <output_dir> [--game INDEX]
```

It auto-detects two log shapes (snapshot-diff and event-stream) and, on first
run, generates protobuf bindings from the vendored Cockatrice `.proto` files
(`proto/`, overridable via `COCKATRICE_PROTO_DIR`). Requires `grpcio-tools`.

---

## Setup

```
pip install -r requirements.txt
```

`requirements.txt` pins a CUDA (cu128) PyTorch build, but **training runs on
CPU** and doesn't need a GPU — the pinned wheel just prevents pip from
silently swapping to a CPU-only build on a machine that does have one. The
replay converter additionally needs `grpcio-tools` (for protobuf codegen).

---

## Tests / self-checks

There's no test framework — each module carries an `assert`-based self-check
you run directly (from `src/`). The important ones:

```
cd src
python game/effects/integration_check.py   # engine integration checks
python drl_env.py                           # action-table / legal-mask checks
python token_features.py                    # tokenization + vocab persistence
python token_arch.py                        # Set Transformer + FiLM (incl. permutation invariance)
python token_deck.py                        # DeckNetwork + pointer masking
python token_action_bridge.py               # action bridge on a real 2p game
python token_pool.py                        # shared vocab / roster wiring
python token_league.py                      # league sampling / eviction
python token_train.py                       # rollout + PPO smoke test
python rewards.py                           # reward functions
python terminated.py                        # win conditions
```

`benchmark_parallel.py` and `benchmark_ppo_batch_size.py` measure rollout
parallelism and the PPO batch-size/device crossover;
`generate_regression_snapshot.py` produces regression fixtures.

---

## Generated artifacts (gitignored)

`checkpoints/` (trained weights + `vocab.json`) and `logs/` (event logs from
`--log` runs) are gitignored — regenerable by rerunning training. Old 2-deck
weights are archived under `checkpoints/archive_2deck/`. (`.gitignore` also
still lists `models/`/`reports/`/`graphify-out/` from earlier tooling; those
reappear only if that tooling runs again.)

---

## Project status

The engine and DRL architecture are the current, active surface. A few things
to know:

- **The DRL pipeline is 2-player only.** The engine still contains a
  1-player mode (used by the original Tron experiments), but the token
  architecture always encodes an opponent seat.
- **Some docstrings are stale.** References in `game/__init__.py` and others
  to `harness.py`, `run.py`, `lean_ppo.py`, and `configs/*.json` describe the
  removed 1-player pipeline; those files no longer exist and `configs/` is
  empty.
- **Five-deck retrain.** `docs/FIVE_DECK_EXPANSION.md` describes the move from
  a 2-deck to a 5-deck roster. The roster wiring (`league_decks.json`, all
  token defs) is applied and the old 2-deck weights are archived; the fresh
  5-deck shared stack + league weights are (re)generated by running the
  pipeline above.
