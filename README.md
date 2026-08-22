# MTG-Subset Simulator + Attention-based Self-Play RL

A from-scratch **2-player Magic: The Gathering rules engine** for a curated
subset of cards, plus a **token/attention deep-RL system** that trains a
separate policy per deck by self-play and continuous league play against a
pool of historical opponents. Essentially no framework dependencies beyond
PyTorch.

---

## What's here

| Piece | Where | What it is |
|-------|-------|------------|
| **Game engine** | `src/game/` | A self-contained MTG-subset simulator: zones, turn/phase loop, priority, the stack, mana, combat, and 150+ card/token effects. No ML dependency. |
| **Card catalog** | `src/game/catalog/` | Card definitions grouped by color. Decks are decklists resolved against this shared catalog — adding a deck from already-implemented cards needs no code. |
| **Action space** | `src/drl_env/` | Turns a decklist + `EFFECT_REGISTRY` into a flat action table with per-action legality + execute closures and legal masks. Not a gym `Env`, just the assembly between engine and training loop. |
| **DRL system** | `src/rl/` | A per-deck Set-Transformer + FiLM perception encoder, trunk/critic/pointer-network action heads, and a PPO self-play + league training loop. |
| **Decks** | `data/*.txt` + `league_decks.json` | An 11-deck roster (see below). |
| **Training drivers** | `src/run_league.py` | Trains every deck continuously in a league, encoder and policy together. |
| **Replay viewer** | `src/webapp/` | A local Flask web app that steps through a logged game's board state one event at a time, plus a publicly-hostable subset for sharing a run. |

**Roster** (`data/league_decks.json`): `mono_red_madness`, `rakdos_madness`,
`spy_combo`, `boggles`, `monster_tron`, `dmir_terror`, `elves`,
`grixis_affinity`, `jund_wildfire`, `mono_blue_terror`, `mono_red_rally`.

---

## Goals

- **A faithful MTG-subset engine**: phases, priority passing, the stack,
  combat with attackers/blockers/damage, and a broad set of real card
  mechanics (madness, plot, flashback, bestow/auras, scry/surveil,
  fetch/search, token generation, initiative/Undercity, mulligans, decking
  out, state-based actions, life-total win checks). Rules fidelity is a
  standing project mandate (see `CLAUDE.md`).
- **A card representation that generalizes across decks.** Each public-zone
  card becomes a *token* (a learned identity embedding + a hand-authored
  static feature vector), and a Set Transformer lets tokens attend to one
  another so relative valuations (an attacker's threat depends on what can
  block it) are learnable.
- **One policy per deck, each with its own perception encoder.** Embedding,
  attention and head all train together under the same PPO update, so a
  checkpoint is self-contained: any two populations can be played against
  each other with no compatibility bookkeeping.
- **Continuous league training**, not a fixed curriculum: every deck trains
  every round against opponents resampled from a pool of historical
  snapshots of every deck, so a policy can't quietly forget how to beat an
  opponent's earlier self.

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
                             the pool plus still-untapped sources could cover it
                             (plan_payment); the agent makes every tap, the solver
                             only decides whether a cost is payable.
    decklist.py              Parse data/*.txt decklists against CARD_DEFS.
    registry.py              Union of every color catalog -> CARD_DEFS + EFFECT_REGISTRY;
                             derive_pending_kinds.
    catalog/                 Card definitions by color (black/blue/colorless/green/
                             multicolor/red/white).
    effects/                 Generic effect plumbing each card catalog calls into:
                             casting, combat, stack + triggers, state_based (SBAs +
                             cleanup), stats (Aura/keyword/P/T), tokens, win_check,
                             madness_and_plot, undercity, shared.
    resolution/              Multi-step decisions the model makes one action at a
                             time: _core (begin/complete state machine) + one
                             handlers_<kind> module per resolution category —
                             mulligan, targeting (incl. stack targeting), combat
                             (declare-blockers/damage-assignment), casting
                             (cast-copy/mode/X/Delve/mana subdecisions/Madness),
                             library (search/graveyard/scry/surveil/ponder/discard/
                             sacrifice/explore), triggers (placement ordering) — all
                             re-exported flat via resolution/__init__.py.

  drl_env/                 Action-table / legal-mask machinery (a package, not a gym
                           Env), split by category, all re-exported flat via
                           __init__.py:
    _actions_common.py       Shared _GATE_NO_PENDING sentinel + _hand_count_available.
    _actions_cast.py         Play land / plain Cast (incl. modal/X-cost/Delve) /
                             Activate / Forestcycle / impulse legal/execute pairs.
    _actions_cast_altzone.py Casting from a non-hand zone or non-default cost:
                             alt-cost/Flashback/Escape/Plot/Omen/Prototype.
    _actions_combat.py       Attack / Assign Blocker / Done blocking, plus the
                             permanently-masked trample damage-to-player row.
    _actions_resolution.py   Generic pending-resolution dispatch: Pass, the shared
                             "Choose: X" by-name dispatch, exact-(name, slot)
                             permanent targeting, pool-mana spend, and every small
                             universal decision row.
    _actions_mana.py         Mana-ability/extra-cost/filter legal/execute pairs +
                             Chromatic Star's choose_mana_color.
    _actions_table.py        build_action_table + legal_action_mask.
    _seat.py                 Per-seat helpers (_for_player, _lost).

  rl/                      The token/attention DRL system, grouped by dependency:
    model/                   Network/observation shape — what the net IS.
      features.py              CardVocab (stable card->index) + static per-token
                               feature vectors + build_token_set.
      arch.py                  SetTransformer (embeddings + self-attention + two PMA
                               pooling heads) and FiLM — the perception encoder.
      deck.py                  DeckNetwork: one deck's own encoder + trunk + critic +
                               pointer-net action head, trained end to end.
      mulligan.py              Per-deck pregame mulligan model + its REINFORCE trainer.
    decision/                 Turning an observation into a chosen action.
      action_bridge.py         Maps the network's combined (fixed + pointer) action
                               space back to real engine calls.
      agent.py                 SeatAgent: per-seat decision dispatch (pregame ->
                               mulligan model, everything else -> DeckNetwork).
      heuristic_agent.py       HeuristicAgent: the gauntlet's hand-authored,
                               non-learned opponent (see the Gauntlet section below).
    training/                 The rollout loop and PPO update math.
      train.py                 Rollout collection game loop (collect_rollout) + league
                               opponent-pairing orchestration; the RolloutBuffer type
                               ppo.py/rollout_parallel.py build on.
      ppo.py                   GAE + the PPO update itself (ppo_update).
      rollout_parallel.py      ProcessPoolExecutor multiprocessing plumbing for league
                               collection (collect_rollout_league_parallel + its
                               worker).
    league/                   Opponent pool + session orchestration.
      league.py                LeaguePool: historical opponent snapshots, PFSP-
                               weighted sampling, eviction/archival, disk persistence.
      league_runner.py         run_league.py's reusable core: _run_session, the
                               eval-mode functions, checkpoint/progress helpers,
                               shared/frozen-stack loaders. Imported directly by
                               benchmarking/training_run.py.
    roster.py                Builds the shared vocab + per-deck action tables from the
                             league roster (data/league_decks.json) — named apart
                             from league/ (opponent-pool management) on purpose.
    rewards.py                The win/loss reward + dense mana-burn shaping
                             (deploy_reward_v6) that league play trains against.
    checkpoint.py             Save/load for a deck's live net+optimizer and frozen
                             league snapshots; the one place device placement
                             (CPU-only on disk) is handled.
    league_cli_spec.py        run_league.py's own CLI surface, torch-free.

  run_league.py            Thin CLI wrapper (arg resolution + main()) around
                           rl/league/league_runner.py.
  run_rollback.py          Promote a historical snapshot back to live.pt. Root-level
                           run_* scripts mutate checkpoints; analysis/ only reads them.
  analysis/                Read-only inspection tools, grouped by concern (except
                           mulligan_retrain/train_mulligan.py, which writes a new
                           mulligan_bootstrap*.pt per deck -- never live.pt/mulligan.pt
                           themselves). Run them from src/, e.g.
                           `python analysis/eval/report_metrics.py ../checkpoints/<league>`
                           — each adds src/ to sys.path itself.
    eval/                    Play games / summarize logs; never trains anything.
      report_metrics.py       Plain-text summary of a league's metrics.jsonl —
                               per-record trends first, then pooled stats with
                               IMPROVING/FLAT/REGRESSING/PAST PEAK verdicts and CIs.
      run_anchor_eval.py      Absolute scale: checkpoints vs a fully untrained
                               DeckNetwork (a floor, saturates fast).
      run_snapshot_round_robin.py
                               Round robin among a deck's own snapshots — Bradley-
                               Terry Elo + residual vs noise floor, monotonicity.
      run_cross_league_eval.py
                               Live weights of one league vs another, per deck, plus
                               mana-burn comparison and budget-matched vintage support.
      bench_gpu_vs_cpu.py      CPU-vs-GPU A/B timing for rl.training.ppo.ppo_update.
    mulligan_retrain/        An open investigation: rebuilding the mulligan model
                             against a frozen main policy.
      train_mulligan.py        One script, --opponent-mode twin | self-mirror: twin
                               trains against an independently-trained twin league's
                               roster, self-mirror against the SAME frozen net with
                               RandomMulligan as its pregame decider — two ablations
                               probing the same collapse-to-always-keep failure mode.
                               Config-file driven (--config, same "extends" pattern
                               and flag > config > default precedence as
                               run_league.py — see config_loader.py and
                               training_configs/mulligan_bootstrap_default.json).
      _mulligan_common.py      Shared net-loading/land-audit/probe-hand helpers.
  benchmarking/            training_run.py (benchmarks the real league loop under
                           different collection configs) + _common.py (path/stdout
                           bootstrap it imports for its side effect).
  webapp/                  GIT SUBMODULE (github.com/JensJansen/pauper-sim-replay) —
                           `git submodule update --init` after a plain clone. Local
                           Flask UI: app.py (routes) + replay_engine.py
                           (event-log-to-board-state reducer) + static/replay.html
                           (no build step). Training runs are launched via
                           run_league.py directly (CLI or the `/train` skill), not
                           through this app. app_public.py is a separate deploy-only
                           entrypoint with the same routes (safe to host publicly
                           since a deploy only ever contains this repo's own
                           committed files). See its own section below.

data/                      Decklists (*.txt) + league_decks.json roster.
checkpoints/               Trained weights + vocab.json (gitignored; see below).
logs/                      Game event logs from --log runs (gitignored).
```

**Run scripts from `src/`.** The driver/training scripts (`run_league.py`,
`benchmarking/*`) use relative paths like `../data` and `../checkpoints`,
and the `rl.*` modules import each other and `game`/`drl_env` by name —
both of which resolve when you run from `src/` (Python puts the script's
directory on `sys.path`). The engine itself (`game/`) is a proper
importable package.

---

## The game engine (`src/game/`)

A self-contained simulator with **no ML dependency**. Highlights:

- **Turn structure & priority.** A full phase sequence (untap, upkeep, draw,
  main1, declare-attackers, declare-blockers, combat-damage, main2, end) with
  real priority rounds — both players get a chance to act at each step.
- **The stack & triggers.** Spells/abilities go on a stack and resolve;
  triggers queue and get promoted to the stack.
- **Combat.** Declare attackers/blockers, summoning sickness, combat damage,
  per-permanent identity (so an Aura can attach to one specific copy),
  gang-blocking and menace, a first-strike damage sub-step, and removal from
  combat — a permanent that leaves the battlefield stops attacking/blocking
  (506.4), while a creature that was blocked stays blocked even if every
  blocker dies (509.1h).
- **Mana — cast then pay (real CR 601.2).** A mana pool spent by explicit
  actions, Tron-land detection, flexible/filter sources. See below.
- **Card mechanics.** Madness, plot, flashback, auras/bestow, scry/surveil,
  fetch/search, initiative/Undercity, mulligans, token generation (Blood,
  Robot, Warrior, Eldrazi Spawn, Food, Clue, Treasure, and more), affinity/
  delve/escape cost reductions, state-based actions, decking out, and
  life-total win checks.
- **Hidden information is respected.** The opponent's hand and both players'
  library contents stay hidden; only public zones (battlefield/graveyard/
  stack/exile) plus the agent's own hand are ever tokenized. Opponent
  hand/library size isn't hidden in real Magic (either player can count a
  library or a hand), so it's surfaced to the agent separately, as a scalar.

  The `revealed` pseudo-zone carries cards a pending resolution is currently
  holding outside every real zone — a scry/surveil's revealed cards, a Deem
  Inferior tuck. Tokenized for the deciding seat only.

  A GRU carries information across turns instead of a persisted "known"
  zone: the agent observes a card's placement (it leaves its own tokenized
  hand one event at a time), so the sequence carries the information even
  though no single frame does. The agent's own previous action is fed in
  alongside the observation to make that recoverable as a lookup rather than
  an inference over hand-token deltas.

### Mana: cast then pay

Real Magic announces a spell, settles modes/X/targets, determines the total
cost (601.2e), and only then activates mana abilities (601.2f) and pays
(601.2g).

Every affordability gate in the codebase is `game.plan_payment(state, cost)
is not None`, meaning *"could the floating pool plus whatever is still
tappable cover this?"* The agent still makes every tap — the solver decides
only **whether** a cost is payable, never **how**.

**The solver is exact**, via the deficiency form of Hall's theorem: a cost is
payable iff there are at least as many mana units as pips, and for every
subset of the colours demanded, at least that many units can produce one of
them. Exactness matters in both directions — a false positive lets a payment
begin that cannot finish (an all-False mask and a hard error, since there is
no *Abandon payment*), and a false negative hides a legal cast. At most six
colours can be demanded, so the subset sweep is ≤63 iterations, and ≤7 for
real costs.

**The stranding invariant.** Because a payment cannot be abandoned, anything
done during one that reduces available mana could make it unfinishable. Every
such action is gated on the payment surviving it. Tapping is *not*
automatically safe: a source with a colour choice counts while untapped as
one unit that could be any of its colours, and tapping collapses it to one —
Jagged Barrens (`{B}` or `{R}`) tapped for `{B}` against an `{R}` cost strands
a payment. Filters tap their own source, and Conduit Pylons is also a `{C}`
source. Saruli Caretaker's cost taps another creature that may itself be a
mana source, so it is excluded from the affordability count entirely and both
its choices are gated. `begin_pay_cost` asserts the invariant itself, so a
caller that skips its gate fails at that call site instead of several actions
later.

Mana abilities are illegal for the whole of an in-flight cast before
601.2f — while modes, X, a delve amount, which graveyard copy, or delve's
exile are being chosen. That is faithful (no player receives priority during
601.2) and removes a whole class of stranding rather than guarding each case.

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
agent's own hand. Each token = a learned **identity embedding** (via
`CardVocab`) concatenated with a deterministic **static feature vector** plus
**dynamic per-instance state**.

The static half has two parts. *Printed stats*: mana cost, type, base P/T,
keywords. Then *what the card does*, *derived* from `EFFECT_REGISTRY` and
`CardDef.extra` rather than hand-authored, so a new card is described the
moment it is registered: mana production (produces-mana, which colors,
whether the amount is board-scaled, enters-tapped, can-filter), an
effect-capability multi-hot over every registry spec key, the
pending-resolution kinds the card can create (its behavioral signature —
`choose_any_target` reads as removal, `search_fetch` as a tutor), flags
(`artifact`/`basic`/`defender`/`indestructible`/`devoid`), and creature
subtypes. This is deliberately the **cheap** end — presence/absence,
auto-derived; the richer version would be a hand-authored semantic vector
per card (effect class, magnitude, what it targets) — `rl/model/features.py`
marks it as the upgrade path.

The dynamic half: a `cost_reduction_delta` (own-hand tokens only, how much
cheaper this card currently is than its printed cost), tapped, effective P/T,
combat commitments (is-attacking, summoning-sick), how many Auras a
permanent carries and whether it *is* an attached Aura, per-kind counter
counts, whether currently targeted by a spell/ability on the stack (mine or
the opponent's, including a spell targeting another spell), zone, and a
mine/theirs side flag.

The second is a **scalar vector** (`rl/decision/agent.py`'s
`_scalar_features`) of non-tokenized globals: turn number, lands-played,
mulligans taken, whose turn it is, each player's floating mana pool (by
color), phase, life totals, each player's library size, the opponent's hand
size, and whether anything on the stack currently targets either player
directly. Library/hand size, floating mana, and a declared target are all
public knowledge in real Magic, unlike hand/library *contents*. The agent's
own hand size isn't included here (redundant with counting its own hand
tokens above).

Every deck has its own network, encoder included — nothing is shared between
decks but the card *index* mapping:

- **`SetTransformer` (`rl/model/arch.py`)** — embeds + projects tokens, runs
  a joint self-attention encoder over *both* sides' tokens (so a token can
  attend across the mine/theirs boundary), then pools with two independent
  learned-query heads: a "mine" summary (trunk input) and a "theirs" summary
  (FiLM conditioning input). Pre-norm transformer for RL stability. Uses
  `torch.nn.MultiheadAttention`/`TransformerEncoderLayer` directly rather
  than hand-rolling attention.
- **`FiLM`** — turns the "theirs" summary into per-layer (gamma, beta)
  modulations of the trunk, chosen over concatenation.
- **A `GRU` between the trunk and every head** — makes the critic, the
  fixed-action head, and the pointer query all history-aware (the raw
  observation is otherwise strictly Markov, so it can't represent
  something like "they held two blue up and passed").

  Two invariants it depends on: state is keyed **by seat** (a mirror
  pairing puts one `SeatAgent` on both seats, so a shared state would leak
  seat 0's hand into seat 1's inputs), and cleared **per game** (`ppo_update`
  replays every episode from a zero state).
- **`DeckNetwork` (per-deck, `rl/model/deck.py`)** — a small trunk + critic +
  a **pointer-network action head**. The action space is the union of a
  **fixed table** of non-targeting actions (play land, cast X, pass, mana
  payments, mulligans, …) and a **pointer-scored** set of targeting actions
  (attack / assign-blocker / choose-target), scored against the
  post-attention token representations. Both halves feed **one combined
  softmax**, so a masked-categorical sample over the true legal set is
  correct.
- **Pregame mulligan model (per-deck, `rl/model/mulligan.py`)** — a separate
  small head owning every pregame keep/mulligan/bottom decision. It reads
  the same structured, self-attended hand representation the main policy
  sees at every in-game decision — its own deck's `SetTransformer` run over
  `rl.model.features.build_token_set`'s full per-card token set. It holds
  that encoder by plain reference, not as a registered child, and wraps its
  own forward pass in `torch.no_grad()`, so its REINFORCE optimizer never
  steps the encoder PPO owns. A `SeatAgent` (`rl/decision/agent.py`) routes
  pregame decisions to it and everything else to the `DeckNetwork`. It
  trains by its own REINFORCE with a direct whole-game reward, decoupled
  from the main PPO update.

Training is **PPO self-play** (`rl/training/train.py`'s rollout game loop,
`rl/training/ppo.py`'s update math). Mirror matches pool both seats into one
buffer/update; cross-matchups give each net its own buffer, both learning
from every game. Rollout collection parallelizes across worker processes
(`rl/training/rollout_parallel.py`, ~3.2–3.5× on 6 physical cores).

The **league** (`rl/league/league.py`, `rl/league/league_runner.py`) keeps a
rolling window of historical snapshots per deck. Each game resamples an
opponent two-level: pick a deck (including the training deck itself, for
mirror play), then pick one of its snapshots (or its current live weights).
No hardcoded stages — cross-deck and cross-snapshot exposure grows as the
pool fills.

Both levels are **PFSP-weighted** (`LeaguePool.sample_opponent`'s `pfsp=True`
default), not uniform: a deck (or a specific one of its snapshots) the
training deck is currently losing to more often gets sampled more —
`weight = PFSP_FLOOR + (1 - win_rate) ** PFSP_POWER`, with a small floor so
a thoroughly-beaten opponent stays sampleable at a low rate. A
never-yet-played candidate gets a neutral 0.5 win-rate prior. Running
win/loss tallies (`LeaguePool.record_outcome`, fed from every real training
game via `collect_rollout`'s `on_game_end` hook) drive the weighting; they
persist to `<league_dir>/<deck>/opponent_stats.json`.

A snapshot evicted from the rolling window is **archived, not deleted**
(`checkpoints/<league>/<deck>/archive/`,
`DEFAULT_MAX_SNAPSHOTS_PER_DECK=32` × `snapshot_every_games=200`), so a
deck's win rate against its own much-older self stays measurable for the
life of a run. See **Instrumentation** below.

---

## Training pipeline

All decks share one vocabulary (`checkpoints/vocab.json`, append-only so old
checkpoints stay valid) — the card *index* mapping, not any learned weights.
Run everything **from `src/`**.

```
cd src
python run_league.py --n-iterations N --snapshot-every 15 --n-workers 6
```

Every deck trains every iteration against a resampled league opponent. A
deck with no `live.pt` yet starts from a freshly-initialized net — encoder
included.

**PPO samples whole episodes, not transitions**: the GRU needs each game
replayed in order from its own start, so the update segments the buffer on
`done` and minibatches over trajectories. `seq_batch_size_start`/
`seq_batch_size_cap` count **episodes**.

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
  always live).
- `--pfsp` / `--no-pfsp` — PFSP-weight opponent sampling toward whoever's
  currently beating the training deck most, instead of uniform (default
  True; see `rl.league.league.LeaguePool.sample_opponent`'s own docstring).
- `--gauntlet-league-name` — an independently-trained twin league
  (`checkpoints/<name>/`) to periodically measure this league's live nets
  against (optional; most leagues won't have one — see **Gauntlet** below).

`--games-per-iteration` isn't a flag — it defaults to `max(1, n_workers)`
(one game per worker), overridable via the run-config's
`games_per_iteration`.

Optimizer/PPO knobs (`lr`, `mulligan_lr`, `gamma`, `gae_lambda`, `target_kl`,
`n_epochs`, `adv_norm_floor`, `ent_coef`) come from
`rl.league.league_runner.PPO_DEFAULTS` and are overridable per league via a
run-config `"ppo"` object; an unknown key is a hard error rather than a
silent no-op. Eval budget (`eval_games`, `eval_every_sessions`) is
config-driven the same way.

`ent_coef` defaults to `None`, meaning "use `rl.training.train.ent_coef_schedule`'s
0.02 → 0.005 anneal"; a float pins it constant for the whole run instead.

`lr` defaults to `2e-4`: it reaches parity with a reference policy in
roughly a fifth the games and eliminates PPO trust-region truncation, at the
cost of converging to the same final quality rather than a better one.
Lowering `lr` further does not by itself fix a training plateau — plateaus
recur at this value too.

`--device cuda` (or `"device"` in the run config; falls back to CPU if
omitted) moves the PPO update — and only the update — onto the GPU.
Collection always stays on CPU across `n_workers` processes (single-game-at-
a-time inference, which a GPU cannot help with). Every currently-active run
config sets `"device": "cuda"`: CUDA runs `ppo_update` 1.6–2.25× faster than
CPU on real training buffers, with `epochs_run` identical on both arms.
Checkpoints are always written as CPU tensors regardless of training
device, so a league can move between CPU and GPU between sessions with no
conversion.

### Reduced parameter surface

Not exposed as CLI flags because they're derived or hardcoded:
`--games-per-iteration` (`max(1, n_workers)`, overridable via run-config's
`games_per_iteration`), the PPO minibatch ramp bounds
(`rl.training.train.batch_size_for_iteration`'s own 32→2048 defaults), and a
batch-size safety cap (now the job of whatever repeatedly re-invokes
`run_league.py` and health-checks between calls, e.g. the `/train` skill's
escalation loop).

Kept as real per-run decisions: `--total-games`, `--n-workers`,
`--checkpoint-opponent-rate`, `--games`/`--seed`, `--n-iterations` (a
documented debug escape hatch), `--snapshot-every`/`snapshot_every_games`.

### Instrumentation

Every league session (`_run_session`, both league and `--matchup` modes)
appends to `checkpoints/<league>/metrics.jsonl`, one JSON line per record:

- `kind: "session_start"` — one header per session recording the reward
  function, roster, cumulative games/deck, and every resolved PPO/eval
  hyperparameter.
- `kind: "ppo"` — per deck per iteration: `policy_loss`, `value_loss`,
  `entropy`, `explained_variance` (`1 - Var(ret - value)/Var(ret)` —
  `value_loss` alone is a raw MSE with no scale attached), `buffer_size`
  (transitions collected that iteration) and `batch_size` side by side, so a
  saturated minibatch ramp (`batch_size >= buffer_size`, at which point
  `ppo_update` stops sub-batching and just runs `n_epochs` full-batch steps)
  is directly visible. Every record also carries `cumulative_games`.
- `kind: "mulligan"` — per deck per iteration REINFORCE loss/n.
- `kind: "vs_history"` — once per session per deck (league mode only): the
  live net played against its own oldest still-active snapshot and, once one
  exists, its oldest **archived** snapshot — a direct win-rate-vs-past-self
  measurement. Skipped (empty) automatically until a deck has been through
  at least one snapshot cycle. Each record carries `snapshot_id`/
  `is_archive`, and the two labels must never be pooled: `archive_oldest` is
  pinned to `snapshot_0` forever (a fixed ~200-game reference), while
  `active_oldest` tracks a rolling ~6,400-game-old self. Games are played
  side-swapped from a paired seed, so on-the-play is balanced exactly rather
  than in expectation.
- `kind: "vs_gauntlet"` / `kind: "vs_heuristic"` — the gauntlet mechanism's
  two tiers, both external to this league's own self-play history. See
  **Gauntlet** below.

Every game the engine plays also gets one `game_over` event appended to its
own `event_log` (`winner`, `turn_won`).

`python analysis/eval/report_metrics.py <league_dir> [--window N]` prints a
plain-text summary read from `metrics.jsonl` — stdlib only. It leads with
the **per-record sequence** for every win-rate series and only then pools,
because pooling hides a decline. Four verdicts are distinguished:
`IMPROVING` / `FLAT` / `REGRESSING` (linear decline) / `PAST PEAK` (below a
window this run already reached, whatever the overall trend) — `PAST PEAK`
catches a rise-then-fall shape a linear trend test reads as no trend at
all. Its threshold is Šidák-corrected for the number of windows searched,
since the best of many noisy windows is high by selection. `FLAT` is
annotated with the minimum effect the sample size could have detected.

### Gauntlet

`vs_history` (above) and PFSP-weighted sampling both only ever measure a
league against **its own** self-play history. The gauntlet adds two
external reference points:

- **Tier 2 — an independently-trained twin population**: the
  actively-trained league is measured each session against a separate
  checkpoint tree (`gauntlet_league_name`) that never plays against its
  live nets during training — a genuinely external reference, unlike
  anything drawn from the league's own history. Wired via a config's
  `gauntlet_league_name` field (`rl.league.league_runner._run_eval_vs_gauntlet`,
  once per session per deck, only once the twin population has a
  checkpoint for that deck). Costs nothing when unconfigured — most
  leagues won't set it.
- **Tier 1 — `rl.decision.agent.HeuristicAgent`**: a hand-authored,
  non-learned opponent, reusing the same legal-action machinery a trained
  `SeatAgent` does (`_build_decision`, `_executor_for`) but scoring among
  legal choices by fixed, general MTG principles: play a land if possible;
  else cast the highest-cost thing affordable; attack a creature only if
  it's safe or a fair-or-better trade; block to kill when possible; else
  pass. Deliberately rough, for catching obviously-bad regressions rather
  than benchmarking strength. Currently scoped to `mono_red_rally` only
  (`heuristic_decks` in a league config). Pregame delegates to `AlwaysKeep`.

Both report into `metrics.jsonl` (`kind: "vs_gauntlet"` / `"vs_heuristic"`,
picked up by `analysis/eval/report_metrics.py`).

### Direct matchup (no league sampling)

```
python run_league.py --matchup DECK_A DECK_B [--games 50] [--log path/to/games.json]
```

Runs a fixed pairing between two named decks, still updating and
checkpointing both. `--log` captures the engine's own event log for every
game as one JSON file — the input the replay converter consumes (`--log` is
wired only through `--matchup` mode).

### Rewards & win condition

- **Rewards** (`rl/rewards.py`): league play uses `deploy_reward_v6`: a
  **flat `+1.0` on any win, `-1.0` on any loss or no-winner timeout**
  (`flat_win_loss_reward`), with a dense mana-burn penalty (below) applied
  to the **winner only**. No efficiency scaling and no cleanup-discard
  penalty on either band — hoarded cards stay **visible** in game state (an
  overflowing hand, uncast threats, an undeveloped board), so a terminal
  win/loss signal can attribute their cost on its own given enough
  training. `PlayerState.mana_burnt_total`/`mana_burnt_this_turn` feed no
  reward — they remain raw diagnostics for logging/viz. The **mulligan
  model** trains on its own reward (`rl/model/mulligan.py`): `WIN_REWARD`
  if the seat won, 0 otherwise, on transitions accumulated across several
  league iterations per REINFORCE update (see
  `rl.league.league_runner._run_session`'s `MULLIGAN_UPDATE_EVERY`).

  **Dense mana-burn shaping** (`with_dense_mana_burn_penalty`, applied via
  `refund_on_loss=True`). A per-transition penalty for mana burnt at a
  phase boundary (rule 500.4), read from
  `PlayerState.mana_burnt_this_turn_single_pip` — a **per-pip attributed**
  subset of the total: `PlayerState.mana_pool_single_pip` tracks how many
  floating pips of each color trace back to a "single-pip" mana-producing
  event, one that added exactly 1 symbol to the pool (`game.mana.
  float_mana`, `len(symbols) == 1`, computed dynamically per event). A
  plain land or Llanowar Elves always qualifies; Rakdos Carnarium and
  Utopia Sprawl's automatic bonus (2+ symbols per tap) never do; a Tron
  land qualifies only while not all three Tron types are controlled. A
  mana filter's output (Conduit Pylons/Barrels of Blasting Jelly) is
  forced untagged regardless of count (`taggable=False`) — a deliberate
  pool→pool conversion, not reflexive tapping. Spending a pip
  (`game.mana.spend_one_pip`) always consumes an untagged pip of that
  color first, so a burst mana ability's own unavoidable excess absorbs
  blame ahead of a genuinely avoidable single-pip tap of the same color.

  `game.turn.Phase.END` runs a real end-step priority round (rule 513)
  first, sweeps mana at an explicit sub-boundary, then runs cleanup, so
  mana floated during the forced hand-size discard is charged correctly.
  Mana abilities are illegal for the whole cleanup portion. `in_cleanup`
  is the only thing distinguishing the end step from cleanup (they share
  `Phase.END`). The trigger-driven extra priority round there is kept so a
  Madness card discarded by forced cleanup can still resolve its own
  cast-or-graveyard decision.

  **Mana burn is charged to the winner only** (`refund_on_loss=True`):
  charging both bands rewards losing *passively* over losing while trying,
  since a seat that never taps mana cannot burn mana. Implementation is a
  **deferral, not a terminal refund** (`rl/training/train.py`'s
  `deferred_charges`): charges are computed and attributed per-tap as they
  happen, but held for the whole game and applied at the terminal flush
  only if that seat won; on a loss they are simply never written. A
  terminal refund would not be equivalent: PPO trains on GAE advantages,
  where a charge written at step `t` lands in `delta_t` immediately, but a
  terminal refund reaches it only through GAE's discounted backward
  recursion, leaving early burns in a long losing game mostly
  un-cancelled.

  **Curve/guarantee.** `mana_burn_c=2.9`/`p=4.0`/`mana_burn_weight=0.5`/
  `game_penalty_cap=1.5`. Per-turn charge by pips burnt: `1→0.007`,
  `2→0.092`, `3→0.267`, `4→0.392`, `5→0.449` (~90% of the weight),
  asymptoting toward `0.5`. Worst-case win `= 1.0 - 1.5 = -0.5`; every loss
  `= -1.0` exactly — a sloppy win *can* score negative, which is deliberate
  and harmless since the only ordering that matters is win-vs-loss.
- **Win condition**: the engine's real one — an opponent's life total
  hitting 0, or a player decking out. There is no separate termination
  heuristic.

---

## Replay viewer (`src/webapp/`)

A local Flask web app that steps through a logged game's board state one
event at a time, backed by the same event-log JSON `--log` writes. Training
runs are launched via `run_league.py` directly (CLI, or the `/train` skill)
— this app has no training-launch surface of its own.

**A git submodule**, not a regular tracked directory —
[github.com/JensJansen/pauper-sim-replay](https://github.com/JensJansen/pauper-sim-replay).
`git clone`ing this repo alone leaves `src/webapp/` empty; either clone with
`--recurse-submodules`, or after a plain clone run
`git submodule update --init`. A change to the viewer itself is a commit in
*that* repo, then a second commit here bumping the pinned submodule SHA.

```
cd src/webapp
python app.py          # http://127.0.0.1:5000 -- localhost only, no auth
```

`/` serves the replay viewer directly. **`--log` output needs to land
inside the submodule's own checkout** for the server-side "Browse server
logs" list to find it (`logs/` resolves relative to `src/webapp/`, not this
repo's root) — any filename, any depth under that `logs/`, e.g. from `src/`:
```
python run_league.py --matchup deck_a deck_b --log webapp/logs/<run-name>/event_log.json
```
(the filename `event_log.json` is a convention, not a requirement).

### Game replay viewer (`/`)

Step through a logged game's board state one event at a time. MVP scope:
retroactive viewing of an already-completed `--log` file only (no live game
viewing).

- **The webapp parses the raw event-log JSON directly** — no intermediate
  replay format. `replay_engine.py`'s `GameReducer` folds the event stream
  into one board-state snapshot per event: life totals, mana pools,
  hand/battlefield/graveyard/exile contents, and the stack (name-based zone
  identity tracking, DFC face reverts, aura-orphan handling,
  same-phase-recast identity). The pregame mulligan sequence is **not**
  netted into one summary step — see below.
- **A hamburger button fixed at the top-left opens a drawer** (link home,
  open-a-new-file, browse-server-logs, and the game list) that overlays the
  board without disturbing the scrub position underneath. File selection is
  a native browser file picker; the browser posts the picked file's content
  to the backend, which returns that file's game list (one file can hold an
  entire round-robin `--eval` run), then reduces just the selected game.
- **"Browse server logs"** lists every `*.json` file under the submodule's
  own `logs/`, any depth, any filename (`GET /api/replay/runs`, newest
  first) — clicking an entry fetches it (`GET
  /api/replay/runs/<path:name>/raw`) through the same client-side flow as a
  picked file. `app_public.py` has the identical routes, pointed at its
  own `logs/`.
- Each game in the list is labeled from its own `deck_a`/`deck_b` fields
  (stamped by `rl.league.league_runner`'s `_write_event_log`) as `"deck A
  vs deck B (game N)"`; logs written before that field existed fall back to
  a file-level `game — game N` label. The drawer's game list is a
  client-side search box over the full per-file index, paginated 50 at a
  time.
- **Both hands are always fully visible**, and the stack is always visible —
  this is post-hoc review of a finished game, not live play.
- **Card art is hotlinked from Scryfall's image endpoint** per card name (no
  local caching, no asset pipeline). Hovering any card thumbnail pops up a
  larger version.
- **Cockatrice-style board layout**: each player's zones stack vertically,
  with a right-hand rail (library/graveyard/exile/mana pool) per player,
  mirrored around the middle so the two creature rows face off. Graveyard
  and exile collapse to a pile showing the most-recently-added card's art
  (hover to see every card); library only ever shows a card count. Tapped
  permanents render rotated 90°. Auras nest in a small stack beneath the
  permanent they enchant (resolved to the target's actual controller).
- **The stack column (far left) is split in half, one per player** — the
  underlying model is still one shared LIFO stack (`GameReducer`'s own
  `self.stack`); the split is purely how the viewer groups entries by
  controller. Each entry shows card art, not just a name.
- **Far-right panel: the agent's top-5 candidate actions at the current
  decision**, split in half like the stack column, since a
  `decision_weights` step always has exactly one deciding player. That
  player's half shows the top-5 candidates by the network's post-mask
  probability, the chosen one highlighted, with the value estimate below;
  the other half shows "no decision data." Opt-in instrumentation, off by
  default: `rl/decision/agent.py`'s `_seat_step` and `rl/model/mulligan.py`'s
  `decide()` log a `decision_weights` event only when `state.event_log is
  not None` and only for a real (non-forced, >1-legal-option) decision —
  reading values already computed in that decision's own forward pass, so
  no extra inference call.
- **Every mulligan-round draw, reject, and bottom-card pick is its own step**
  (owner directive), not netted into one "opening hands" summary — each is a
  genuinely separate decision (its own `rl/model/mulligan.py` forward pass), exactly
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
  `resolution_begin`/`complete`, `explore`/`animated`, etc.) still advance
  the scrubber with a plain label so the timeline never silently skips a
  step.
- **A creature's power/toughness badge tracks its current effective stats,
  not just what it printed on entry.** `game.effects.state_based`'s
  `check_state_based_actions` recomputes each one's
  `permanent_power`/`permanent_toughness` — folding in counters, until-EOT
  pump, attached Auras, and animate/transform — and logs a `stats_changed`
  event whenever it moved. The viewer keeps the entering, printed stats
  alongside as `base_power`/`base_toughness` and colors the badge red/green
  when the live value is below/above it.
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
  whose turn it structurally is and who currently holds priority are shown
  separately since they diverge whenever a player is doing something (a
  response, a block) on the other player's turn.
- Deferred, not in MVP scope: live re-inference against an arbitrary
  checkpoint.

### Public hosting (`app_public.py`)

`app_public.py` is a deploy-only entrypoint with the same routes as
`app.py`, including the server-side log browser (`/api/replay/runs*`) —
that only ever lists/serves files already committed to this public repo's
own `logs/` (deliberately not gitignored there). The two POST endpoints
(`/api/replay/games`, `/api/replay/game`) take raw log JSON text in the
body regardless, and `replay_engine.py` has no imports beyond the stdlib.

Hosting is driven from the
[pauper-sim-replay](https://github.com/JensJansen/pauper-sim-replay) repo
directly, not from here — that repo's own `render.yaml` deploys
`app_public.py` as a Render free-tier web service off its own
`requirements-public.txt` (`flask` + `gunicorn`), independent of this
repo's full CUDA-pinned `requirements.txt`. Pushing a change here bumps
this repo's pinned submodule commit; Render only redeploys on a push to
the other repo.

---

## Setup

```
pip install -r requirements.txt
```

`requirements.txt` pins a CUDA (cu128) PyTorch build plus `numpy`.
**Training defaults to CPU and needs no GPU**, so a CPU-only machine is
fully supported; the pinned wheel keeps pip from silently swapping to a
CPU-only build on a machine that does have one, and is what `--device cuda`
needs. The replay converter additionally needs `grpcio-tools` (for
protobuf codegen). The replay viewer (`src/webapp/`, a git submodule —
`git submodule update --init` first if it's empty) has its own, separate
dependency: `flask`.

---

## Tests

A real `pytest` suite lives under `tests/`, mirroring `src/`'s layout (e.g.
`src/game/mana.py` -> `tests/game/test_mana.py`). `pyproject.toml` sets
`pythonpath = ["src"]` and `testpaths = ["tests"]`, so `pytest` runs from
the repo root with no `cd src` needed.

```
pip install -r requirements.txt   # pytest included
pytest                             # whole suite
pytest -m "not slow"               # fast tier: game engine/catalog/effects,
                                    # deterministic, no torch, sub-second
pytest -m slow                     # rl/ tier: imports torch, plays real
                                    # mini-games, runs PPO/REINFORCE updates
```

The whole suite runs together in well under a minute. No pre-commit hook
wired up yet — running the suite is still manual.

`benchmarking/training_run.py` measures the real league loop under
different collection configs (`seq`, `mp<N>`) over a fresh untrained stack
— a benchmark, not a test.

---

## Generated artifacts (gitignored)

`checkpoints/` (trained weights + `vocab.json`) and `logs/` (event logs
from `--log` runs) are gitignored — regenerable by rerunning training.
`.gitignore` also lists `models/`, `reports/`, and `graphify-out/`.

---

## Project status

The engine and DRL architecture are the current, active surface.

- **The DRL pipeline is 2-player only.** The engine tolerates a no-opponent
  (1-player) configuration (useful for isolated card-behavior tests), but the
  token architecture always encodes an opponent seat.
