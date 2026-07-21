# Tron Assembly Simulator

Estimates the probability of assembling all three Tron lands (Urza's Mine,
Urza's Power Plant, Urza's Tower) by a given turn, for a specific 60-card
Modern Tron decklist (`data/deck_list.txt`). Includes a from-scratch MTG
rules engine, a hand-tuned greedy heuristic pilot, and a DRL policy trained
against the same simulator.

## Layout

```
src/            All source code (flat -- see "Why flat?" below)
  game.py         The simulator: card definitions, zones, turn loop, mana,
                  card effects, and the greedy heuristic policy. Zero
                  dependencies, self-contained sanity checks. Nothing else
                  in this repo modifies it.
  rewards.py      Reward function(s) for the DRL policy (pluggable).
  tron_env.py     Gymnasium environment adapter wrapping game.py.
  harness.py      TrainingHarness: train / evaluate / save / load a DRL
                  model, plus optional detailed per-game JSON logging.
  run.py          Unified train/eval runner, keyed off a JSON config under
                  configs/ (superseded the old per-deck train_*.py/
                  evaluate_drl.py scripts -- see
                  docs/DECK_REGISTRY_REFRESH_PLAN.md).
  viz/            Game Viewer: a separate React+Vite app (its own
                  package.json, not part of the Python project) for
                  stepping through a harness.evaluate(log_path=...) log
                  action by action -- see its own section below.
docs/           Planning documents (one PLAN + one CHECKLIST per feature,
                written before implementation). Includes the lookahead-search
                experiment, which was implemented, evaluated, and reverted --
                kept as a record of what was tried and why.
data/           Source decklist.
models/         Saved trained models (gitignored -- model.zip + metadata.json
                per run, regenerate via train_drl.py).
logs/           Detailed per-game evaluation logs and raw training output
                (gitignored, can be large).
reports/        Human-readable aggregate summaries (gitignored).
```

**Why flat `src/`?** All 6 modules import each other directly (`import
game`, `import tron_env`, ...), which works because Python adds a script's
own directory to `sys.path` automatically. Splitting into subpackages would
need real packaging for a project this size that isn't published or
imported elsewhere -- not worth it yet.

## Setup

```
pip install -r requirements.txt
```

Requires a CUDA-capable GPU build of torch for reasonable DRL training
speed (see the comment in `requirements.txt`); the pure simulator and
heuristic have no such requirement.

## Running things

All commands run from the repository root (paths like `models/...` inside
the scripts are relative to the working directory, not the script's own
location):

```
python src/harness.py                        # DRL harness sanity checks (construction, train, evaluate, save/load, logging)
python src/run.py <config> <runs> --train    # train a deck (configs/<config>.json), continuing an existing model if one exists
python src/run.py <config> <runs> --log      # evaluate a trained deck against fresh games
```

## Game Viewer (`src/viz/`)

A React+Vite app for stepping through one game at a time from a
`harness.evaluate(..., log_path=...)` JSON log: real card art, hand/
battlefield/graveyard rendered like an actual game (battlefield sorted
lands-then-artifacts-then-creatures, tapped cards rotated), arrow-key
step-through, and a searchable/paginated batch browser. Separate Node
project, not part of the Python side:

```
cd src/viz
npm install
node scripts/fetch-card-art.mjs   # one-time: caches card art locally (gitignored)
npm run dev                        # opens a local dev server; drag a games.json log onto the page
npm test                           # Vitest -- all the pure logic (parsing, filtering, sorting, pagination)
```

The app never needs internet access to *view* a game -- only the one-time
art-fetch script talks to Scryfall.

## Design history

`docs/` has the full design rationale in the order it was actually built:
`TRON_SIMULATOR_*` (the engine), `TRON_LOOKAHEAD_*` (a rollout-search
policy -- implemented, found to only match the heuristic at real
compute cost, reverted), `DRL_*` (the trained policy that replaced it,
which does measurably outperform the heuristic), `VISUALIZER_*` (the
game viewer above -- v1 shipped as a static HTML/vanilla-JS tool, v2
rebuilt it as this React app for further growth).
