# MTG-Subset Simulator + Attention-based Self-Play RL

A from-scratch **2-player Magic: The Gathering rules engine** for a curated
card subset, plus a **token/attention deep-RL system** that trains a
separate policy per deck by self-play and continuous league play against a
pool of historical opponents. No framework dependencies beyond PyTorch.

---

## What's here

| Piece | Where | What it is |
|-------|-------|------------|
| **Game engine** | `src/game/` | Zones, turn/phase loop, priority, the stack, mana, combat, 150+ card/token effects. No ML dependency. |
| **Card catalog** | `src/game/catalog/` | Card definitions by color. Decks are decklists resolved against this shared catalog. |
| **Action space** | `src/drl_env/` | Decklist + `EFFECT_REGISTRY` -> a flat action table with per-action legality, execute closures, and legal masks. Not a gym `Env`. |
| **DRL system** | `src/rl/` | Per-deck Set-Transformer + FiLM encoder, trunk/critic/pointer-network action heads, PPO self-play + league training loop. |
| **Decks** | `data/*.txt` + `league_decks.json` | An 11-deck roster (see below). |
| **Training pipeline** | `src/run_training_pipeline.py` | Trains a whole league end to end (encoder + policy together), running the full `validation/` check suite on a cadence. The primary way to train. |
| **Training driver** | `src/run_league.py` | One training session; `--matchup`/one-off debug runs. |
| **Validation checks** | `src/validation/` | Mid-run quality/stats checks (round robins, mulligan quality, vs. history), run by `run_training_pipeline.py`. |
| **Replay viewer** | `src/webapp/` | Local Flask app that steps through a logged game's board state one event at a time, plus a publicly-hostable subset. |

**Roster** (`data/league_decks.json`): `mono_red_madness`, `rakdos_madness`,
`spy_combo`, `boggles`, `monster_tron`, `dmir_terror`, `elves`,
`grixis_affinity`, `jund_wildfire`, `mono_blue_terror`, `mono_red_rally`.

---

## Goals

- A rules-faithful engine: phases, priority, the stack, combat, and real card
  mechanics (madness, plot, flashback, bestow, scry/surveil, fetch, tokens,
  initiative/Undercity, mulligans, decking, state-based actions). Rules
  fidelity is a standing project mandate — see `CLAUDE.md`.
- A card representation that generalizes across decks: each public-zone card
  becomes a token (learned identity embedding + static features), attended
  over by a Set Transformer so relative valuations are learnable.
- One self-contained policy per deck (encoder included), so any two
  checkpoints can play each other with no compatibility bookkeeping.
- Continuous league training against resampled historical snapshots of every
  deck, so a policy can't quietly forget how to beat an opponent's earlier
  self.

---

## Repository layout

```
src/
  config_loader.py         Shared training-config JSON loader ("extends" merge).
  repo_paths.py             Stdlib-only repo-root path helper.
  game/                    The engine (package). Zero ML deps.
    cards.py                 EffectId / CardType / CardDef — the card data model.
    state.py                 PlayerState + GameState: zones, turn/stack bookkeeping.
    turn.py                  Phase/Speed enums; priority rounds; mulligan; the turn
                             loop; game_coroutine + run_multiplayer_game.
    mana.py                  Cast-then-pay mana (CR 601.2): plan_payment checks
                             affordability; the agent makes every tap.
    decklist.py              Parse data/*.txt decklists against CARD_DEFS.
    registry.py              Union of every color catalog -> CARD_DEFS + EFFECT_REGISTRY.
    catalog/                 Card definitions by color.
    effects/                 Generic effect plumbing: casting, combat, stack +
                             triggers, state_based (SBAs + cleanup), stats,
                             tokens, win_check, madness_and_plot, undercity, shared.
    resolution/              Multi-step decisions made one action at a time: _core
                             (begin/complete state machine) + one handlers_<kind>
                             module per category — mulligan, targeting, combat,
                             casting, library, triggers — re-exported flat via
                             resolution/__init__.py.

  drl_env/                 Action-table / legal-mask machinery, split by category,
                           re-exported flat via __init__.py:
    _actions_common.py       Shared sentinel + hand-count helper.
    _actions_cast.py         Play land / Cast (modal/X/Delve) / Activate / Forestcycle.
    _actions_cast_altzone.py Casting from non-hand zones or alt costs: Flashback,
                             Escape, Plot, Omen, Prototype.
    _actions_combat.py       Attack / Assign Blocker / Done blocking / trample row.
    _actions_resolution.py   Generic pending-resolution dispatch (Pass, by-name
                             choices, permanent targeting, mana spend).
    _actions_mana.py         Mana-ability/extra-cost/filter legal/execute pairs.
    _actions_table.py        build_action_table + legal_action_mask.
    _seat.py                 Per-seat helpers.

  rl/                      The token/attention DRL system, grouped by dependency:
    model/                   Network/observation shape.
      features.py              CardVocab + static per-token feature vectors + build_token_set.
      arch.py                  SetTransformer (embeddings + self-attention + two PMA
                               pooling heads) and FiLM — the perception encoder.
      deck.py                  DeckNetwork: one deck's encoder + trunk + critic +
                               pointer-net action head, trained end to end.
      mulligan.py              Per-deck pregame mulligan model + its REINFORCE trainer.
    decision/                 Turning an observation into a chosen action.
      action_bridge.py         Maps the combined action space back to engine calls.
      agent.py                 SeatAgent: per-seat dispatch (pregame -> mulligan
                               model, everything else -> DeckNetwork).
    training/                 The rollout loop and PPO update math.
      train.py                 collect_rollout + league opponent-pairing orchestration.
      ppo.py                   GAE + ppo_update.
      rollout_parallel.py      ProcessPoolExecutor plumbing for league collection.
    league/                   Opponent pool + session orchestration.
      league.py                LeaguePool: historical snapshots, PFSP-weighted
                               sampling, eviction/archival, disk persistence.
      league_runner.py         run_league.py's core: _run_session, eval-mode
                               functions, checkpoint/progress helpers. Imported
                               directly by benchmarking/training_run.py.
    roster.py                Builds shared vocab + per-deck action tables from
                             data/league_decks.json.
    rewards.py                deploy_reward_v6: win/loss reward + dense mana-burn shaping.
    checkpoint.py             Save/load for live nets + frozen league snapshots
                             (CPU-only on disk).
    league_cli_spec.py        run_league.py's CLI surface, torch-free.

  run_training_pipeline.py Trains a whole league end to end, running the full
                           validation/ check suite on a cadence -- see
                           "Validation checks" above. The primary training entry
                           point.
  run_league.py            One training session; --matchup / one-off debug runs.
  run_rollback.py          Promote a historical snapshot back to live.pt.
  validation/              Every mid-run quality/stats check, one module per
                           check, run through by run_training_pipeline.py.
                           To add a new check: write a module (NAME + run(ctx)),
                           register it in __init__.py's CHECKS list.
    __init__.py              CHECKS registry + run_all(ctx) (log-and-continue on
                             a check that raises).
    _common.py               ValidationContext + shared net-loading/output-
                             writing/metrics.jsonl helpers every check uses.
    round_robin_primary.py   primary_vs_primary_round_robin.
    round_robin_training.py  primary_vs_training_round_robin (full cross product).
    mulligan_audit.py        mulligan_audit (per-deck + league rollup).
    vs_history.py            vs_history.
  analysis/                Read-only inspection tools (never train, except
                           mulligan_retrain/, which writes a new
                           mulligan_bootstrap*.pt, never live.pt/mulligan.pt).
                           Run from src/, e.g.
                           `python analysis/eval/report_metrics.py ../checkpoints/<league>`.
    eval/                    Play games / summarize logs.
      report_metrics.py       Plain-text summary of a league's metrics.jsonl.
      run_snapshot_round_robin.py
                               Round robin among a deck's own snapshots — Bradley-
                               Terry Elo + residual vs noise floor. A retrospective,
                               whole-run analysis (not per-checkpoint), so it stays
                               a standalone tool rather than a validation/ check.
      bench_gpu_vs_cpu.py      CPU-vs-GPU A/B timing for ppo_update.
    mulligan_retrain/        Rebuilding the mulligan model against a frozen main
                             policy — an open investigation.
      train_mulligan.py        --opponent-mode twin | self-mirror: two ablations
                               probing the same collapse-to-always-keep failure
                               mode. Config-file driven (--config, same "extends"
                               precedence as run_league.py).
      _mulligan_common.py      Shared net-loading/land-audit/probe-hand helpers
                               (validation/mulligan_audit.py extends
                               audit_land_counts from here with per-deck
                               attribution rather than reimplementing it).
  benchmarking/            training_run.py benchmarks the real league loop under
                           different collection configs.
  webapp/                  GIT SUBMODULE (github.com/JensJansen/pauper-sim-replay)
                           — `git submodule update --init` after a plain clone.
                           app.py (routes) + replay_engine.py (event-log-to-
                           board-state reducer) + static/replay.html (no build
                           step). app_public.py is a separate deploy-only
                           entrypoint with the same routes. See its own section
                           below.

data/                      Decklists (*.txt) + league_decks.json roster.
training_configs/          Run-mechanics + league-identity JSON configs.
checkpoints/               Trained weights + vocab.json (gitignored; see below).
logs/                      Game event logs from --log runs (gitignored).
```

**Run scripts from `src/`.** `run_training_pipeline.py`, `run_league.py`, and
`benchmarking/*` use relative paths like `../data`, `../checkpoints`, and
`../training_configs`, and the `rl.*`/`validation.*` modules import each
other and `game`/`drl_env` by name — both resolve when run from `src/`.
`game/` is a proper importable package.

---

## The game engine (`src/game/`)

Self-contained, no ML dependency.

- **Turn structure & priority.** Full phase sequence (untap, upkeep, draw,
  main1, declare-attackers, declare-blockers, combat-damage, main2, end)
  with real priority rounds.
- **The stack & triggers.** Spells/abilities go on the stack and resolve;
  triggers queue and get promoted to the stack.
- **Combat.** Attackers/blockers, summoning sickness, per-permanent
  identity, gang-blocking, menace, a first-strike sub-step, and removal
  from combat (506.4); a blocked creature stays blocked even if every
  blocker dies (509.1h).
- **Mana — cast then pay (CR 601.2).** See below.
- **Card mechanics.** Madness, plot, flashback, auras/bestow, scry/surveil,
  fetch/search, initiative/Undercity, mulligans, token generation (Blood,
  Robot, Warrior, Eldrazi Spawn, Food, Clue, Treasure, and more),
  affinity/delve/escape cost reduction, state-based actions, decking out,
  life-total win checks.
- **Hidden information is respected.** Only public zones (battlefield/
  graveyard/stack/exile) plus the agent's own hand are ever tokenized.
  Opponent hand/library *size* is public knowledge in real Magic, so it's
  surfaced separately as a scalar; contents stay hidden.

  A `revealed` pseudo-zone carries cards a pending resolution is holding
  outside every real zone (a scry's revealed cards, a Deem Inferior tuck),
  tokenized only for the deciding seat.

  A GRU carries information across turns instead of a persisted "known"
  zone — the agent observes a card leaving its own tokenized hand one event
  at a time, so the sequence carries the information even though no single
  frame does. The agent's previous action is fed in alongside the
  observation for the same reason.

### Mana: cast then pay

Real Magic announces a spell, settles modes/X/targets, determines total cost
(601.2e), then activates mana abilities (601.2f) and pays (601.2g). Every
affordability gate is `game.plan_payment(state, cost) is not None` — could
the floating pool plus whatever's still tappable cover this? The agent makes
every tap; the solver only decides whether a cost is payable.

**Exact solver**, via the deficiency form of Hall's theorem: payable iff
there are at least as many mana units as pips, and for every subset of
colours demanded, at least that many units can produce one of them. At most
six colours can be demanded, so the subset sweep is <=63 iterations.

**Stranding invariant.** A payment can't be abandoned, so any action taken
mid-payment that reduces available mana is gated on the payment surviving
it. A dual-colour source counts as one unit until tapped, so tapping it for
the wrong colour can strand a payment; filters and cost-paid mana sources
(e.g. Saruli Caretaker) are handled the same way. `begin_pay_cost` asserts
the invariant itself.

Mana abilities are illegal for the whole of an in-flight cast before 601.2f
(no player receives priority during 601.2).

> **AUTHORIZED SIMPLIFICATION** (owner-approved 2026-08-17). Speculative
> floating — a mana ability activated with no payment open — is restricted
> to the active player's own main phase, instead of any priority window as
> in real Magic. Low cost in practice: holding mana up just means leaving
> lands untapped, and paying for an instant on the opponent's turn goes
> through the payment window anyway.

The engine tolerates a no-opponent (1-player) configuration for card-level
unit tests, but the active surface, and everything the DRL system drives,
is 2-player.

---

## The DRL system (`src/rl/`)

Observation has two parts. First, a **variable-length set of card tokens**
(one per public-zone card for both players, plus every card in the agent's
own hand). Each token = a learned **identity embedding** (`CardVocab`) +
**static feature vector** + **dynamic per-instance state**.

Static half: printed stats (cost, type, base P/T, keywords), then what the
card *does* — derived from `EFFECT_REGISTRY`/`CardDef.extra` rather than
hand-authored, so a new card is described the moment it's registered: mana
production, an effect-capability multi-hot over every registry spec key,
pending-resolution kinds the card can create, type flags, creature
subtypes. This is the cheap end (presence/absence, auto-derived); a
hand-authored semantic vector per card is the upgrade path noted in
`rl/model/features.py`.

Dynamic half: `cost_reduction_delta` (own-hand only), tapped, effective
P/T, combat commitments, Aura count/attached flag, per-kind counters,
whether currently targeted by anything on the stack, zone, mine/theirs.

Second, a **scalar vector** (`rl/decision/agent.py`'s `_scalar_features`):
turn number, lands played, mulligans taken, whose turn, floating mana pool
by colour, phase, life totals, library sizes, opponent's hand size, and
whether the stack targets either player directly. All public knowledge in
real Magic. The agent's own hand size is omitted (redundant with its hand
tokens).

Every deck has its own network, encoder included — nothing is shared
between decks but the card index mapping:

- **`SetTransformer` (`rl/model/arch.py`)** — embeds/projects tokens, runs
  joint self-attention over both sides' tokens, pools with two
  learned-query heads: a "mine" summary (trunk input) and a "theirs"
  summary (FiLM conditioning input). Pre-norm, `torch.nn.MultiheadAttention`
  / `TransformerEncoderLayer`.
- **`FiLM`** — turns the "theirs" summary into per-layer (gamma, beta)
  trunk modulation.
- **A `GRU`** between trunk and every head, making the critic, fixed-action
  head, and pointer query history-aware (the raw observation is otherwise
  Markov). Keyed **by seat** (a mirror pairing puts one `SeatAgent` on both
  seats) and cleared **per game**.
- **`DeckNetwork` (`rl/model/deck.py`)** — small trunk + critic + a
  pointer-network action head. Action space = a **fixed table**
  (non-targeting actions: play land, cast X, pass, mana payments,
  mulligans, ...) union a **pointer-scored** set (attack / assign-blocker /
  choose-target, scored against post-attention tokens). Both halves feed
  one combined softmax.
- **Pregame mulligan model (`rl/model/mulligan.py`)** — separate small head
  for every pregame keep/mulligan/bottom decision, reading its own deck's
  `SetTransformer` output over the full hand token set. Holds that encoder
  by plain reference (not a registered child) and runs under
  `torch.no_grad()`, so its own REINFORCE optimizer never steps the
  encoder PPO owns. `SeatAgent` routes pregame decisions to it, everything
  else to `DeckNetwork`. Trains on whole-game reward, decoupled from PPO.

Training is **PPO self-play** (`rl/training/train.py` rollout loop,
`rl/training/ppo.py` update math). Mirror matches pool both seats into one
buffer/update; cross-matchups give each net its own buffer. Rollout
collection parallelizes across worker processes
(`rl/training/rollout_parallel.py`).

The **league** (`rl/league/league.py`, `rl/league/league_runner.py`) keeps
a rolling window of historical snapshots per deck. Each game resamples an
opponent two-level: a deck (including the training deck itself, for
mirrors), then one of its snapshots or current live weights.

Both levels are **PFSP-weighted** (`LeaguePool.sample_opponent`'s
`pfsp=True` default): `weight = PFSP_FLOOR + (1 - win_rate) ** PFSP_POWER`,
so an opponent currently beating the training deck more gets sampled more,
with a floor so a thoroughly-beaten opponent stays sampleable. A
never-played candidate gets a neutral 0.5 win-rate prior. Win/loss tallies
(`LeaguePool.record_outcome`) persist to
`<league_dir>/<deck>/opponent_stats.json`.

An evicted snapshot is **archived, not deleted**
(`checkpoints/<league>/<deck>/archive/`,
`DEFAULT_MAX_SNAPSHOTS_PER_DECK=32`), so win rate against a much-older self
stays measurable for the life of a run. See **Instrumentation** below.

---

## Training pipeline

All decks share one vocabulary (`checkpoints/vocab.json`, append-only) —
the card index mapping, not learned weights. Run from **`src/`**.

**`run_training_pipeline.py --config PATH [--fresh]`** is the primary way to
train a league end to end: one command trains from wherever
`progress.json` says a league is at to `--config`'s `total_games`/deck,
running the full **validation/** check suite every `checks_cadence_pct` of
the way there (see **Validation checks** below) — no repeated manual
invocation needed.

```
cd src
python run_training_pipeline.py --config ../training_configs/main_league.json [--fresh]
```

`run_league.py` itself remains for `--matchup` training and one-off debug
runs (`--n-iterations`, bypassing auto-sizing):

```
cd src
python run_league.py --n-iterations N --snapshot-every 15 --n-workers 6
```

Every deck trains every iteration against a resampled league opponent. A
deck with no `live.pt` yet starts from a fresh net, encoder included.

**PPO samples whole episodes, not transitions**: the GRU needs each game
replayed from its own start, so the update segments the buffer on `done`
and minibatches over trajectories. `seq_batch_size_start`/
`seq_batch_size_cap` count **episodes**.

Each deck's live net/optimizer persists in
`checkpoints/league/<deck>/live.pt`; snapshots live alongside; a
`session.txt` counter makes runs resumable.

Key flags:
- `--n-workers` — parallel rollout processes (default 6).
- `--snapshot-every` — iterations between registering a snapshot of every
  deck and checkpointing live nets (default 20).
- `--total-games` (with `--league-config`, e.g.
  `../training_configs/main_league.json`) — auto-sizing target: instead of
  `--n-iterations`, doubles the batch size each invocation until this many
  games/deck have been played.
- `--checkpoint-opponent-rate` — probability a sampled opponent is a frozen
  historical snapshot instead of live weights (default 0.0).
- `--pfsp` / `--no-pfsp` — PFSP-weight opponent sampling instead of uniform
  (default True).

`create_training_league`/`checks_cadence_pct`/`checks_games`/`stratify_0land_pct`
are config-only fields (no `run_league.py` flag) — they're read by
`run_training_pipeline.py`, not `run_league.py` itself. See **Validation
checks** below.

`--games-per-iteration` isn't a flag — defaults to `max(1, n_workers)`,
overridable via the run-config's `games_per_iteration`.

PPO knobs (`lr`, `mulligan_lr`, `gamma`, `gae_lambda`, `target_kl`,
`n_epochs`, `adv_norm_floor`, `ent_coef`) come from
`rl.league.league_runner.PPO_DEFAULTS`, overridable per league via a
run-config `"ppo"` object; an unknown key is a hard error.

`ent_coef` defaults to `None` (`rl.training.train.ent_coef_schedule`'s
0.02 -> 0.005 anneal); a float pins it constant instead.

`lr` defaults to `2e-4`. `--device cuda` (or `"device"` in the run config;
falls back to CPU) moves the PPO update — and only the update — to GPU.
Collection stays on CPU across `n_workers` processes. Checkpoints are
always written as CPU tensors, so a league can move between CPU/GPU
between sessions with no conversion.

### Reduced parameter surface

Not exposed as CLI flags: `--games-per-iteration` (derived), the PPO
minibatch ramp (`rl.training.train.batch_size_for_iteration`'s own 4->16
episode defaults), and a batch-size safety cap (now the job of whatever
repeatedly re-invokes `run_league.py` and health-checks between calls).

Kept as real per-run decisions: `--total-games`, `--n-workers`,
`--checkpoint-opponent-rate`, `--games`/`--seed`, `--n-iterations` (a debug
escape hatch), `--snapshot-every`/`snapshot_every_games`.

### Instrumentation

Every league session appends to `checkpoints/<league>/metrics.jsonl`, one
JSON line per record:

- `kind: "session_start"` — one header per session: reward function,
  roster, cumulative games/deck, every resolved PPO hyperparameter.
- `kind: "ppo"` — per deck per iteration: `policy_loss`, `value_loss`,
  `entropy`, `explained_variance`, `buffer_size`/`batch_size` side by side
  (so a saturated minibatch ramp is visible), `cumulative_games`.
- `kind: "mulligan"` — per deck per iteration REINFORCE loss/n.

`vs_history`/the primary-vs-training-league comparison no
longer append here automatically every session -- they're validation/'s
checks now, on their own much coarser cadence. See **Validation checks**.

Every game also gets one `game_over` event (`winner`, `turn_won`) appended
to its own `event_log`.

`python analysis/eval/report_metrics.py <league_dir> [--window N]` prints a
plain-text summary from `metrics.jsonl` — stdlib only. Leads with the
per-record sequence for every win-rate series before pooling (pooling
hides a decline). Four verdicts: `IMPROVING` / `FLAT` / `REGRESSING`
(linear decline) / `PAST PEAK` (below a window this run already reached).
`PAST PEAK`'s threshold is Šidák-corrected for the number of windows
searched. `FLAT` is annotated with the minimum effect the sample size
could have detected.

### Validation checks

`run_training_pipeline.py` runs the full `validation/` check suite every
`checks_cadence_pct` of `total_games` (default 5%, so ~20 passes over a
full run), `checks_games` games each (default 50) -- a single umbrella
covering every mid-run quality/stats question, in one place, easy to extend
(add a module to `validation/`, register it in `validation/__init__.py`'s
`CHECKS` list). A check that raises is logged and skipped; training itself
is never aborted by a bad check.

Five checks today:
- **`primary_vs_primary_round_robin`** — every training deck plays every
  other, mirrors included, on current live weights.
- **`primary_vs_training_round_robin`** — the FULL cross product against an
  independently-trained **training league** -- every primary deck vs every
  training-league deck, not just same-name pairs. A genuinely external
  reference: a shared population-wide blind spot is something an
  independently-evolved population is more likely to expose than any
  opponent drawn from this league's own history. Opt in per league config
  with `"create_training_league": true` (same roster/mechanics as the
  primary league, auto-managed under `checkpoints/<league_name>-training/`).
  When set, `run_training_pipeline.py` trains that twin to a **fixed**
  `TRAINING_LEAGUE_GAMES` (10,000) games/deck first, once, before the
  primary league's own training starts; it is never extended past that cap
  even if the primary's `total_games` later rises -- a stable benchmark
  population, not a moving target. Skipped (logged, not fatal) whenever
  `create_training_league` is unset/false, or until the twin has a
  checkpoint for a given deck.
- **`mulligan_audit`** — % of hands kept vs. mulliganed by land count (0-7),
  from two sources: (1) the SAME games the two round-robin checks above just
  played (primary-controlled seats only, even in a cross-league game) -- not
  a separate batch of its own; and (2) a sculpted-hand probe that loads each
  deck's current live + mulligan weights and queries fixed, seeded synthetic
  hands covering every land count 0-7 on demand, filling in the buckets
  natural self-play rarely or never draws. Both report keep/mulligan rate
  plus an entropy figure for the decision.
- **`vs_history`** — each deck's current live net vs. its own frozen old
  self (the oldest still-active snapshot, and once one exists, the oldest
  **archived** snapshot). The one check immune to the "everyone is
  improving together" confound the two round robins both have, since the
  opponent here can never change -- any win-rate movement is unambiguously
  this deck's own progress.

Output, at two levels: `checkpoints/<primary_league>/checks/` for
league-wide results (both round robins; a league-wide rollup for the other
two), `checkpoints/<primary_league>/<deck>/checks/` for a single deck's
own results (`mulligan_audit`/`vs_history`). Every file is
stamped with the games/deck count at that cadence point
(`<check>_<games>games.json`) and never overwritten, so a whole run's
history stays on disk. Each check also drops a compact per-deck summary
into `metrics.jsonl` (`kind` matching the check's own name above) for
`report_metrics.py`'s existing trend tooling.

### Direct matchup (no league sampling)

```
python run_league.py --matchup DECK_A DECK_B [--games 50] [--log path/to/games.json]
```

Runs a fixed pairing between two named decks, still updating and
checkpointing both. `--log` captures the event log for every game as one
JSON file (wired only through `--matchup` mode) — the replay viewer's input.

### Rewards & win condition

- **`deploy_reward_v6`** (`rl/rewards.py`): flat **+1.0 win, -1.0 loss or
  no-winner timeout** (`flat_win_loss_reward`), plus a dense mana-burn
  penalty applied to the **winner only**. No efficiency scaling, no
  cleanup-discard penalty — hoarded cards stay visible in game state, so
  the win/loss signal attributes their cost on its own given enough
  training. `mana_burnt_total`/`mana_burnt_this_turn` are diagnostics only,
  fed to no reward. The **mulligan model** trains on its own reward
  (`WIN_REWARD` if the seat won, else 0), accumulated across several league
  iterations per REINFORCE update.

  **Dense mana-burn shaping** (`with_dense_mana_burn_penalty`,
  `refund_on_loss=True`): a per-transition penalty for mana burnt at a
  phase boundary (rule 500.4), read from a **per-pip attributed** subset of
  the total (`mana_pool_single_pip` tracks pips from a "single-pip" mana
  event — exactly 1 symbol produced). A plain land or Llanowar Elves
  qualifies; Rakdos Carnarium/Utopia Sprawl's bonus (2+ symbols per tap)
  never does; a Tron land qualifies only while not all three Tron types are
  controlled; a mana filter's output (Conduit Pylons) is always untagged.
  Spending a pip consumes an untagged pip of that colour first, so burst
  mana's own unavoidable excess absorbs blame ahead of an avoidable tap.

  `Phase.END` runs a real end-step priority round (513) first, sweeps mana
  at an explicit sub-boundary, then runs cleanup, so mana floated during
  the forced hand-size discard is charged correctly. `in_cleanup` is the
  only thing distinguishing the end step from cleanup (they share
  `Phase.END`); mana abilities are illegal for the whole cleanup portion.

  Charged to the winner only (`refund_on_loss=True`) — charging both bands
  would reward losing passively over losing while trying. Implemented as a
  **deferral, not a terminal refund** (`train.py`'s `deferred_charges`):
  charges are computed per-tap as they happen, held for the whole game, and
  applied at the terminal flush only on a win — a terminal refund wouldn't
  be equivalent under GAE, which discounts a refund back through time far
  less accurately than crediting it at the step it happened.

  Curve (`mana_burn_c=2.9`, `p=4.0`, `mana_burn_weight=0.5`,
  `game_penalty_cap=1.5`): per-turn charge by pips burnt —
  `1->0.007`, `2->0.092`, `3->0.267`, `4->0.392`, `5->0.449`, asymptoting
  toward `0.5`. Worst-case win = `1.0 - 1.5 = -0.5`; every loss = `-1.0`
  exactly — only the win-vs-loss ordering matters, so a sloppy win scoring
  negative is fine.
- **Win condition**: the engine's real one — life total hitting 0, or
  decking out. No separate termination heuristic.

---

## Replay viewer (`src/webapp/`)

Local Flask app that steps through a logged game's board state one event
at a time, from the same event-log JSON `--log` writes. Training runs
launch via `run_league.py` directly — this app has no training-launch
surface.

**A git submodule**, not a tracked directory —
[github.com/JensJansen/pauper-sim-replay](https://github.com/JensJansen/pauper-sim-replay).
A plain `git clone` of this repo leaves `src/webapp/` empty; clone with
`--recurse-submodules`, or run `git submodule update --init` after. A
viewer change is a commit in *that* repo, then a second commit here
bumping the pinned submodule SHA.

```
cd src/webapp
python app.py          # http://127.0.0.1:5000 -- localhost only, no auth
```

`--log` output needs to land inside the submodule's own checkout for the
server-side log browser to find it (`logs/` resolves relative to
`src/webapp/`), e.g. from `src/`:
```
python run_league.py --matchup deck_a deck_b --log webapp/logs/<run-name>/event_log.json
```

### Game replay viewer (`/`)

Retroactive viewing of an already-completed `--log` file only (no live
game viewing).

- `replay_engine.py`'s `GameReducer` folds the raw event-log JSON directly
  into one board-state snapshot per event (name-based zone identity, DFC
  face reverts, aura-orphan handling, same-phase-recast identity) — no
  intermediate replay format.
- A hamburger drawer overlays the board (open-a-file, browse-server-logs,
  game list) without disturbing scrub position. File selection posts a
  picked file's content to the backend for reduction.
- **Browse server logs** lists every `*.json` under the submodule's own
  `logs/` (`GET /api/replay/runs`, newest first); `app_public.py` has the
  same routes against its own `logs/`. Games are labeled from `deck_a`/
  `deck_b` fields; older logs fall back to a file-level label.
- Both hands and the stack are always fully visible — this is post-hoc
  review, not live play.
- Card art is hotlinked from Scryfall by name (no local cache).
- Cockatrice-style layout: zones stack vertically per player with a
  right-hand rail (library/graveyard/exile/mana pool), mirrored around the
  middle. Graveyard/exile collapse to a hover pile; library shows a count.
  Tapped permanents render rotated 90°. Auras nest under the permanent they
  enchant.
- The stack column is split per player (one shared LIFO stack underneath;
  the split is purely a viewer grouping).
- Far-right panel shows the deciding player's top-5 candidate actions by
  post-mask probability, chosen one highlighted, value estimate below. Opt-
  in instrumentation (`decision_weights` event), logged only when
  `state.event_log is not None` and the decision has >1 legal option — no
  extra inference call.
- Every mulligan-round draw/reject/bottom-pick is its own step, not netted
  into one summary (owner directive).
- **Library count is a simplified approximation** (owner-authorized):
  assumes a real 60-card deck, decrements once per card drawn, restores on
  a mulligan take/bottom put-back. Non-draw depletion (search, mill,
  impulse exile, scry-to-graveyard) isn't counted — a rough count, not a
  source of truth.
- Events with no board-visible effect (`priority_flip`, `resolution_begin`
  /`complete`, etc.) still advance the scrubber with a plain label.
- A creature's P/T badge tracks current effective stats (`state_based`'s
  `check_state_based_actions` recomputes them each check), colored red/
  green against the printed `base_power`/`base_toughness` it entered with.
- **`pass` is dropped entirely** (owner-authorized): a pass carries no
  information beyond the stack resolving or the phase advancing, and was
  often a large fraction of a game's step count.
- **A phase with nothing but passes is skipped straight through** to
  whatever phase actually has something happen (`phase_change` is
  buffered until a real event or the next phase-change resolves it).
- `decision_weights` panels are buffered alongside their phase's header
  (owner directive, 2026-08-19): they fire for any >1-option decision
  regardless of whether the choice was pass, so a merely-considered-and-
  declined action no longer defeats the empty-phase collapse above.
- A persistent Turn/Phase/active-player line above the scrubber
  distinguishes whose turn it structurally is from who currently holds
  priority (they diverge on a response or block).
- Deferred, not in scope: live re-inference against an arbitrary checkpoint.

### Public hosting (`app_public.py`)

Deploy-only entrypoint with the same routes as `app.py`, including the
server-side log browser, scoped to files already committed to the public
repo's own `logs/` (deliberately not gitignored there). `replay_engine.py`
has no imports beyond the stdlib.

Hosting is driven from the
[pauper-sim-replay](https://github.com/JensJansen/pauper-sim-replay) repo
directly — its own `render.yaml` deploys `app_public.py` as a Render
free-tier service off `requirements-public.txt` (`flask` + `gunicorn`),
independent of this repo's CUDA-pinned `requirements.txt`. A push here
bumps the pinned submodule commit; Render only redeploys on a push to the
other repo.

---

## Setup

```
pip install -r requirements.txt
```

`requirements.txt` pins a CUDA (cu128) PyTorch build plus `numpy`.
**Training defaults to CPU and needs no GPU** — the pinned wheel just keeps
pip from silently swapping to a CPU-only build on a machine that has one,
and is what `--device cuda` needs. The replay converter additionally needs
`grpcio-tools`. The replay viewer (`src/webapp/`, a submodule) has its own
dependency: `flask`.

---

## Tests

A `pytest` suite lives under `tests/`, mirroring `src/`'s layout (e.g.
`src/game/mana.py` -> `tests/game/test_mana.py`). `pyproject.toml` sets
`pythonpath = ["src"]` and `testpaths = ["tests"]`, so `pytest` runs from
the repo root, no `cd src` needed.

```
pip install -r requirements.txt   # pytest included
pytest                             # whole suite
pytest -m "not slow"               # fast tier: engine/catalog/effects,
                                    # deterministic, no torch, sub-second
pytest -m slow                     # rl/ tier: imports torch, plays real
                                    # mini-games, runs PPO/REINFORCE updates
```

The whole suite runs in well under a minute. No pre-commit hook wired up
yet — running the suite is manual.

`benchmarking/training_run.py` measures the real league loop under
different collection configs (`seq`, `mp<N>`) over a fresh untrained stack
— a benchmark, not a test.

---

## Generated artifacts (gitignored)

`checkpoints/` (trained weights + `vocab.json`) and `logs/` (event logs
from `--log` runs) — regenerable by rerunning training. `.gitignore` also
lists `models/`, `reports/`, `graphify-out/`.

---

## Project status

The engine and DRL architecture are the current, active surface.

- **The DRL pipeline is 2-player only.** The engine tolerates a
  no-opponent (1-player) configuration for isolated card-behavior tests,
  but the token architecture always encodes an opponent seat.
