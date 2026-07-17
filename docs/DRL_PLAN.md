# Deep RL Policy — Implementation Plan

Companion to [TRON_SIMULATOR_PLAN.md](TRON_SIMULATOR_PLAN.md) and the (reverted)
lookahead experiment. Planning only — nothing gets built until this is
confirmed. `game.py` is **not modified** by this plan; every new piece
lives in new files that import `game.py` and call its existing primitives.

## Goal

Train a policy (via reinforcement learning, not search) to make the same
turn-by-turn decisions `policy_choose_action` makes today, and see whether
it can beat, match, or falls short of the hand-tuned heuristic — using a
proper training process instead of lookahead's per-decision noisy
resampling, which we already showed only ties the heuristic at real cost.

Built as **four independent, swappable pieces** rather than one coupled
script, per your follow-up: the simulator, the reward function, the DRL
model, and the training harness that wires the other three together.

## Decisions locked in (across both rounds of this conversation)

1. **RL library**: PyTorch + Stable-Baselines3, plus `sb3-contrib` for
   `MaskablePPO` (the action space has state-dependent legality, so action
   masking is a hard requirement).
2. **Reward** (one specific reward function — see "Reward function
   contract" below for the pluggable interface it implements): paid once,
   at episode end.
   - Failure (horizon reached, Tron never assembled): **0**.
   - Success at turn T: **`(0.85 ** T) + 0.02 * resource_quality(T)`**.
   - `resource_quality` = `norm(non_land_permanents, cap=5) +
     norm(available_mana, cap=10) + norm(hand_size, cap=10)`, each term
     clamped to `[0, 1]` via `min(x, cap) / cap`, so `resource_quality ∈
     [0, 3]`.
   - **The 0.02 coefficient is derived, not guessed**: the tightest gap is
     between the last two turns, `0.85**5 * (1 - 0.85) ≈ 0.067`; dividing
     by the max possible resource_quality swing (3) gives `ε < 0.0223`.
     **0.02** sits inside that bound with margin — no amount of extra
     cards/mana at turn 6 can ever outscore a resource-poor success at
     turn 3.
3. **Rollout scope**: build the full pipeline now, validate with a small
   pilot training run before committing to a real one — same discipline
   as the lookahead timing pilot that caught a real cost problem early.
4. **Swap mechanism**: plain dependency injection (constructor
   parameters), not a config-driven registry. Swapping pieces means
   writing a few lines in a run script, not editing framework code.
5. **Model scope**: any Stable-Baselines3-family algorithm (PPO,
   MaskablePPO, A2C, DQN, ...) — not a fully custom non-SB3 model
   interface. The harness handles both maskable and non-maskable
   algorithms (see "DRL model contract" below).
6. **Harness scope**: one harness covers both training and
   playing/evaluating — including reloading a previously *saved* model
   and running it fresh against a large batch of games, independent of
   the training run that produced it.
7. **Game-agnosticism**: cleanliness now, not speculative generality. The
   simulator/harness boundary is kept honest and undocumented internals
   don't leak across it, but there's no abstract multi-game plugin system
   — Tron is the only real implementation.

## Two things resolved without asking (flag if you'd rather weigh in)

- **"Training policy" and "reward function" are the same concept** — your
  first message paired them with a slash, and the follow-up's final
  enumeration only lists 4 things (game, reward, model, harness), not 5.
  Treating "reward function" as the canonical name.
- **The environment adapter is not one of the 4 pieces.** The Gym-style
  environment (observation encoding, the 22-action space, legality
  masking) has to combine a simulator with a reward function to exist at
  all — it's assembly logic the harness owns, not a standalone
  interchangeable piece, and not something bolted onto `game.py` (which
  stays a pure rules engine with zero RL vocabulary in it).
- **The "two gammas" subtlety from the first round of this plan still
  applies unchanged**: `0.85 ** T` is computed from `state.turn_number`
  and baked into the reward function's return value; the SB3 algorithm's
  own `gamma` hyperparameter stays near 1 (e.g. `0.99`), since it's now a
  separate, unrelated concern (ordinary advantage-estimation smoothing).

## Architecture: the four pieces and their contracts

```
   Simulator          Reward function         DRL model (SB3)
   (game.py,           (rewards.py,             (sb3 / sb3-contrib
   unchanged)           new file)                 class, e.g. MaskablePPO)
        \                    |                        /
         \                   |                       /
          \                  |                      /
           v                 v                     v
              tron_env.py (environment adapter)
              -- built BY the harness, not a 5th piece --
                              |
                              v
                     harness.py (TrainingHarness)
                    .train() / .evaluate() / .save() / .load()
                              |
                              v
              train_drl.py / evaluate_drl.py (thin run scripts)
```

### 1. Simulator — `game.py` (unchanged)

Already a clean, well-tested rules engine with zero RL concepts in it.
The adapter reads `GameState` directly (hand/battlefield/graveyard/turn
number) and calls existing primitives (`play_land_from_hand`,
`plan_payment`, `cast_ancient_stirrings`, etc.) — nothing new needed here.

### 2. Reward function contract — `rewards.py` (new file)

A reward function is any Python callable matching:

```python
def reward_fn(state: game.GameState, done: bool, horizon: int) -> float:
    ...
```

Called once per environment step, with the state *after* that step's
action was applied. A sparse reward function just returns `0.0` unless
`done`; a dense one could return something every call. This is the whole
contract — no base class needed, any matching callable works, per the
dependency-injection decision above.

`rewards.py` holds one implementation to start:
`assembled_with_resource_quality(state, done, horizon)`, exactly the
formula from "Decisions locked in" §2. Built as its own top-level
function (not a closure or class) so it can be imported and passed
directly to the environment builder. Adding a second reward function
later means adding a second function to this file — nothing else changes.

### 3. DRL model contract — any SB3-family class

The harness accepts a **model class + constructor kwargs** (not a
pre-built instance), because SB3 algorithms are constructed *with* their
environment (`PPO("MlpPolicy", env, **kwargs)`) — the harness builds the
env first, then instantiates the model with it. Concretely: `harness =
TrainingHarness(reward_fn=..., model_cls=MaskablePPO, model_kwargs={...})`.

Action masking specifically only works with `MaskablePPO`/`MaskableDQN`
from `sb3-contrib`. For a non-maskable algorithm (plain `PPO`, `A2C`),
the environment adapter still computes `action_masks()` internally, but
the harness instead wraps `step()` so an illegal action silently resolves
to "pass" rather than crashing — makes every SB3 algorithm usable, just
with a worse exploration signal for the ones that can't see the mask
directly. This wrapping only activates for non-maskable model classes.

### 4. Training harness — `harness.py` (new file), class `TrainingHarness`

```python
class TrainingHarness:
    def __init__(self, reward_fn, model_cls, model_kwargs=None,
                 horizon=6, on_the_play=True):
        ...  # builds the TronEnv (tron_env.py) from reward_fn/horizon/on_the_play,
             # builds model_cls(env=self.env, **model_kwargs)

    def train(self, total_timesteps, save_path=None):
        ...  # self.model.learn(total_timesteps=...); if save_path, calls self.save(save_path)

    def evaluate(self, num_games, seed=0):
        ...  # runs self.model through num_games real games via game.py's run_game,
             # same (turn_assembled, turn_online) pairs / aggregate_results / print_report
             # shape as every other comparison in this project

    def save(self, path):
        ...  # self.model.save(f"{path}/model.zip") + a metadata.json sidecar (see below)

    @classmethod
    def load(cls, path, reward_fn, model_cls, horizon=6, on_the_play=True):
        ...  # rebuilds the harness/env, loads model_cls.load(f"{path}/model.zip", env=...),
             # cross-checks metadata.json against the given reward_fn/horizon and warns/errors
             # on mismatch (see "Model persistence" below)
```

`train_drl.py` and `evaluate_drl.py` become thin wiring scripts: pick a
reward function from `rewards.py`, pick a model class from
`sb3_contrib`/`sb3`, construct (or load) a `TrainingHarness`, call
`.train()` or `.evaluate()`. Swapping any of the 3 injected pieces is a
one-line change in these scripts, never a change to `harness.py` or
`tron_env.py`.

## Model persistence (new requirement from this round)

The point of exporting a trained model is to reuse it later — e.g. "load
this model and run it against 100,000 fresh simulated shuffles to measure
its real efficacy," independent of and after the training run that
produced it. Two files per saved run, in `models/<run_name>/`:

- `model.zip` — SB3's own native save format (network weights + the
  algorithm's hyperparameters + policy architecture). Nothing custom
  here, this is what `model.save()`/`model_cls.load()` already do.
- `metadata.json` — a small sidecar recording what the model *assumes*,
  since `model.zip` alone doesn't say what environment/reward it was
  trained against, and silently evaluating it against a mismatched
  action space or reward would produce meaningless numbers instead of an
  error:
  ```json
  {
    "reward_fn": "assembled_with_resource_quality",
    "model_cls": "MaskablePPO",
    "model_kwargs": {"...": "..."},
    "horizon": 6,
    "on_the_play": true,
    "action_space_size": 22,
    "observation_dim": 90,
    "total_timesteps_trained": 20000,
    "train_seed": 0,
    "timestamp": "2026-07-16T..."
  }
  ```
- `TrainingHarness.load(...)` checks the given `reward_fn`'s name and the
  current `tron_env.py`'s action/observation sizes against
  `metadata.json` and raises clearly (not a silent shape mismatch deep in
  a numpy call) if they don't match — e.g. if `tron_env.py`'s action space
  has grown since the model was trained.

## Environment design (`tron_env.py`) — unchanged from the first round

### Observation space

A fixed-size `Box` vector, all components normalized to roughly `[0, 1]`:

- **Hand** (22 dims): count of each distinct card name in hand, ÷ that
  card's total copies in the deck.
- **Battlefield** (44 dims): for each of the 22 distinct card names,
  `(untapped_count, tapped_count)`, ÷ total copies in the deck.
- **Library remaining** (22 dims): `deck_total_per_name - hand_count -
  battlefield_count - graveyard_count`, ÷ total copies. Not hidden
  information leaking through — a real pilot knows their own decklist and
  everything they've seen, so by elimination they already know which
  specific cards remain; only the hidden *order* stays unencoded.
- **Turn number** (1 dim): `turn_number / horizon`.
- **Land drop used this turn** (1 dim): `0` or `1`.

Total: 90 dims. Small enough for a small MLP (`net_arch=[64, 64]` for the
pilot).

### Action space — `Discrete(22)`, enumerated exactly

| # | Action | Executes via (existing `game.py` primitive) |
|--:|--------|------------------------------------------------|
| 0–7 | Play land: Mine / Power Plant / Tower / Forest / Wooded Ridgeline / Bojuka Bog / Tocasia's Dig Site / Conduit Pylons | `play_land_from_hand` |
| 8 | Cast Expedition Map | `plan_payment` + `cast_permanent_from_hand` |
| 9 | Cast Crop Rotation | `plan_payment` + `cast_crop_rotation` (fodder choice stays heuristic-pinned: prefer an already-tapped non-Tron land, same rule `try_crop_rotation` already uses — never a Tron land) |
| 10 | Cast Ancient Stirrings | `plan_payment` + `cast_ancient_stirrings` |
| 11 | Cast Candy Trail | `plan_payment` + `cast_permanent_from_hand` |
| 12 | Cast Bonder's Ornament | `plan_payment` + `cast_permanent_from_hand` |
| 13 | Cast Barrels of Blasting Jelly | `plan_payment` + `cast_permanent_from_hand` |
| 14 | Cast Relic of Progenitus | `plan_payment` + `cast_permanent_from_hand` |
| 15 | Forestcycle Generous Ent | `plan_payment` + `forestcycle_generous_ent` |
| 16 | Activate Expedition Map | `plan_payment` + `activate_expedition_map` |
| 17 | Activate Candy Trail (sac) | `plan_payment` + `activate_candy_trail_sac` |
| 18 | Activate Relic of Progenitus | `plan_payment` + `activate_relic_of_progenitus` |
| 19 | Activate Bonder's Ornament (draw) | `plan_payment` + `activate_bonders_ornament_draw` |
| 20 | Activate Tocasia's Dig Site (surveil) | `plan_payment` + `activate_tocasia_dig_site_surveil` |
| 21 | Pass | ends the current decision point |

Search-effect *targets* (which land Expedition Map/Crop
Rotation/Stirrings/Forestcycling actually finds) stay exactly as
implemented today — baked inside the primitives themselves
(`is_priority_land`/`find_and_remove_priority_land`), nothing extra to
wire up. Barrels of Blasting Jelly's and Conduit Pylons' mana-filter
modes are **not** separate actions — same as the heuristic, they only
matter as an automatic fallback *inside* `plan_payment`, unmodified.

`action_masks()` returns a length-22 boolean array: action `i` is legal
iff its card is in the right zone and `plan_payment` can currently afford
its cost. `tron_env.py` takes the injected `reward_fn` as a constructor
parameter and calls it at the end of every `step()` — it does not compute
reward itself.

### Step/reset semantics

- `reset()`: build a fresh shuffled game via `game.py`'s `new_game_state`,
  return the initial observation.
- `step(action)`: if `action == PASS` or no legal actions remain, advance
  through `game.py`'s own turn machinery (`untap_step`, `draw_step`) to
  the next decision point — reusing `game.py`'s logic, not reimplementing
  it. Otherwise execute the chosen action via its primitive from the
  table above. Call the injected `reward_fn(state, done, horizon)` for
  the reward. `terminated=True` when Tron assembles or the horizon is
  reached with no assembly.
- Horizon and on-the-play are fixed per `TronEnv(...)` construction.

## Training design (`harness.py` + `train_drl.py`)

- **Pilot run** (this plan's immediate deliverable): small network
  (`net_arch=[64, 64]`), short run (~10,000–20,000 timesteps), single
  environment (no vectorization yet). Purpose is **infrastructure
  validation, not a good policy**. Concretely, pilot success means:
  - A handful of random-action episodes (no training) run end-to-end
    without error, with `action_masks()` never allowing an illegal move.
  - Reward values observed match the formula by hand-checking a couple of
    episodes.
  - A short `harness.train(total_timesteps=~15000)` call completes
    without error, saves a `model.zip` + `metadata.json`, and produces a
    non-degenerate loss/reward log.
- **Full run**: deferred. Sizing (timesteps, parallel envs, network size)
  gets decided after the pilot reports actual wall-clock per timestep —
  same "measure before committing" discipline as the lookahead timing
  pilot.

## Evaluation design (`harness.py` + `evaluate_drl.py`)

- `harness.evaluate(num_games, seed)` wraps `model.predict(obs,
  action_masks=...)` in a closure matching `game.py`'s
  `choose_action(state)` signature and runs it through `game.py`'s
  existing `run_game` for `num_games` real, independently shuffled games —
  same `(turn_assembled, turn_online)` pairs, same `aggregate_results`/
  `print_report` shape as every other comparison in this project, directly
  comparable to `game.policy_choose_action`'s numbers.
- `evaluate_drl.py` supports both: evaluating right after a training run,
  and **loading a previously saved model cold** (`TrainingHarness.load(...)`)
  and evaluating it fresh — this is the "train once, later run it against
  100,000 shuffles" flow from this round's discussion. 100,000 games is a
  real commitment (SB3's `.predict()` has more per-decision overhead than
  the pure-Python heuristic's function calls); the actual per-game cost
  gets measured with a small pilot batch first, same discipline as
  everywhere else in this project, before committing to the full 100k run.

## Explicitly out of scope for this plan

- Hyperparameter tuning, network architecture search.
- Vectorized/parallel environments (single-env pilot first; revisit if
  the full run's timing needs it).
- Any change to `game.py`.
- Deciding the full run's exact size, or the 100,000-game evaluation's
  exact size — both are post-pilot decisions.
- A config-file/registry-driven way to select pieces (explicitly declined
  in favor of dependency injection this round).
