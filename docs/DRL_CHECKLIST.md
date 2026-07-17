# Deep RL Policy — Implementation Checklist

Companion to [DRL_PLAN.md](DRL_PLAN.md). Concrete build order, phase by
phase, each ending in a runnable check before moving on — same discipline
as the original engine and the lookahead build. Planning only; nothing
gets built until you say go. `game.py` is never modified by any phase
below.

## Phase D0 — Dependencies

- [ ] Create `requirements.txt`: `torch`, `gymnasium`, `stable-baselines3`,
      `sb3-contrib`.
- [ ] Install into the project's environment.
- [ ] Sanity check: `python -c "import torch, gymnasium, stable_baselines3,
      sb3_contrib; print('ok')"` succeeds.

## Phase D1 — Reward function(s): `rewards.py`

- [ ] Define the contract as a module-level docstring: `def
      reward_fn(state: game.GameState, done: bool, horizon: int) -> float`.
- [ ] Implement `assembled_with_resource_quality(state, done, horizon)`:
      - Returns `0.0` if not `done`.
      - If `done` and `state.turn_assembled is None`: returns `0.0`
        (horizon reached, no assembly).
      - If `done` and `state.turn_assembled is not None`: returns
        `(0.85 ** state.turn_number) + 0.02 * resource_quality(state)`.
- [ ] Implement `resource_quality(state) -> float` (module-private
      helper, not part of the public reward contract): sum of three
      `min(x, cap) / cap` terms —
      - `non_land_permanents`: count of `state.battlefield` entries where
        `card_def.card_type != game.CardType.LAND`, cap 5.
      - `available_mana`: sum over every *untapped* `state.battlefield`
        permanent of `len(game.mana_output(p, state))` for permanents
        whose effect is a simple mana source (reuse
        `game.SIMPLE_MANA_SOURCE_EFFECTS` to filter, so the Tron bonus is
        automatically included since `mana_output` already accounts for
        it), cap 10.
      - `hand_size`: `len(state.hand)`, cap 10.
- [ ] Sanity check (hand-built `GameState`s, no environment needed yet):
      - Not done → `0.0` regardless of state contents.
      - Done, `turn_assembled is None` → `0.0`.
      - Done, assembled at turn 3 with 0 non-land permanents/mana/hand →
        reward ≈ `0.85**3 ≈ 0.614`.
      - Done, assembled at turn 6 with maxed-out resource_quality (3.0)
        → reward ≈ `0.85**6 + 0.06 ≈ 0.377`. Assert this is still less
        than the turn-3 minimal-resource case above — the concrete
        instance of the guarantee derived in the plan.

## Phase D2 — Environment adapter: `tron_env.py`

### D2.1 — Card indexing and observation builder

- [ ] `CARD_NAMES = [name for name, *_ in game.DECKLIST]` — the stable,
      deterministic 22-name ordering used everywhere below.
- [ ] `CARD_COPIES = {name: qty for name, qty, *_ in game.DECKLIST}`.
- [ ] `build_observation(state, horizon) -> np.ndarray` (shape `(90,)`,
      `float32`): hand counts ÷ copies (22) + battlefield
      untapped/tapped counts ÷ copies (44) + library-remaining counts ÷
      copies (22) + `turn_number / horizon` (1) + `lands_played_this_turn
      > 0` as 0/1 (1). Library-remaining per name computed as
      `CARD_COPIES[name] - hand_count - battlefield_count -
      graveyard_count` (matches DRL_PLAN's "known by elimination" note).
- [ ] Sanity check: build a fresh `game.new_game_state`, confirm
      `build_observation(...).shape == (90,)`, all values in `[0, 1]`,
      and hand-count entries for the 7 opening-hand cards sum to `7 /
      <appropriate normalization>` when de-normalized.

### D2.2 — Action table

- [ ] Define `ACTIONS`, an ordered list of 22 entries, each
      `(name, legality_fn, execute_fn)`:
      - `legality_fn(state) -> bool`
      - `execute_fn(state) -> None` (mutates state; only ever called when
        `legality_fn` was just confirmed true)
      - Entries 0–7 (land drops): `legality_fn` checks
        `state.lands_played_this_turn == 0` and the named card is in
        `state.hand`; `execute_fn` calls `game.play_land_from_hand`.
      - Entries 8–15 (casts): `legality_fn` checks the card is in hand
        and `game.plan_payment(state, card_def.cast_cost) is not None`;
        `execute_fn` re-derives the plan, calls `game.execute_payment`,
        then the card's cast primitive (table in DRL_PLAN.md's Action
        space section).
      - Entries 16–20 (activations): `legality_fn` checks a matching
        untapped permanent exists on the battlefield and its ability
        cost is payable via `plan_payment`; `execute_fn` pays then calls
        the activation primitive.
      - Entry 21 (Pass): `legality_fn` always `True`; `execute_fn` is a
        no-op (`step()` interprets Pass as "advance the turn," handled
        in D2.4, not by mutating via this table).
- [ ] Sanity check: hand-build a state with exactly one Tron land in hand
      and nothing else castable; confirm `ACTIONS[i][1](state)` is `True`
      for exactly the matching land-drop entry and Pass, `False`
      everywhere else.

### D2.3 — `TronEnv` class skeleton

- [ ] `TronEnv(gymnasium.Env)` constructor: `__init__(self, reward_fn,
      horizon=6, on_the_play=True, seed=None)`. Stores `reward_fn`
      (injected, per the contract in Phase D1 — `tron_env.py` never
      computes reward itself). `observation_space =
      gymnasium.spaces.Box(low=0, high=1, shape=(90,), dtype=np.float32)`.
      `action_space = gymnasium.spaces.Discrete(22)`.
- [ ] `action_masks(self) -> np.ndarray[bool]`: `[fn(self.state) for
      _, fn, _ in ACTIONS]` (length 22 — this exact method name/shape is
      what `sb3-contrib`'s `MaskablePPO` auto-detects).
- [ ] Sanity check: construct a `TronEnv`, confirm
      `observation_space.contains(env.observation_space.sample())` and
      `action_masks()` returns a length-22 bool array before `reset()` is
      even called (should reflect *some* valid pre-game default, or raise
      a clear error if called before `reset()` — decide and document
      which).

### D2.4 — `reset()` / `step()`

- [ ] `reset(self, seed=None, options=None)`: build `self.rng =
      random.Random(seed or self._seed)`, `self.state =
      game.new_game_state(self.on_the_play, self.rng)`, advance through
      `game.untap_step`/`game.draw_step` for turn 1 (matching what
      `game.run_turn` does before the main-phase loop), return
      `(build_observation(self.state, self.horizon), {})`.
- [ ] `step(self, action)`:
      - If `action_masks()[action]` is `False`: **silently substitute
        Pass** (this is the "works with any SB3 algorithm" safety net
        from DRL_PLAN.md, implemented once here in the environment
        itself — simpler than conditionally wrapping in the harness per
        model class, since it's always correct regardless of which
        algorithm drives the env).
      - If Pass (given or substituted): advance `self.state` to the next
        decision point via `game.untap_step`/`game.draw_step`/turn
        increment, matching `game.run_turn`'s structure, OR terminate if
        `self.state.turn_number >= self.horizon`.
      - Otherwise: call `ACTIONS[action][2](self.state)`.
      - `done = self.state.turn_assembled is not None or
        self.state.turn_number >= self.horizon` (post-action/post-advance).
      - `reward = self.reward_fn(self.state, done, self.horizon)`.
      - Return `(build_observation(...), reward, done, False, {})`
        (5-tuple, Gymnasium's `terminated, truncated` split —
        `truncated` stays `False` here, horizon exhaustion is a true
        terminal state for this task, not a truncation).
- [ ] Sanity check: hand-feed a specific seed, step through Pass
      repeatedly, confirm turn number advances exactly like an
      all-pass `game.run_game` does (cross-check against
      `game._phase4_sanity_check`'s known values: 6 turns, hand grows by
      7+5).
- [ ] Sanity check: a full episode of **uniform-random legal actions**
      (sample only where `action_masks()` is `True`) runs to completion
      without error across 50 episodes, terminates with `done=True`
      exactly once per episode, and every returned reward is either `0.0`
      or matches the Phase D1 formula.

## Phase D3 — Training harness: `harness.py`

- [ ] `TrainingHarness.__init__(self, reward_fn, model_cls,
      model_kwargs=None, horizon=6, on_the_play=True, seed=0)`: builds
      `self.env = TronEnv(reward_fn, horizon, on_the_play, seed)`, then
      `self.model = model_cls("MlpPolicy", self.env, **(model_kwargs or
      {}))`. Stores `reward_fn`, `horizon`, `on_the_play`, `model_cls`,
      `model_kwargs` (needed verbatim for the metadata sidecar in D4).
- [ ] Sanity check: construct a harness with `model_cls=MaskablePPO`,
      `model_kwargs={"policy_kwargs": {"net_arch": [64, 64]}}`; confirm
      `self.model` is a real `MaskablePPO` instance wired to `self.env`
      with no error (construction only, no `.learn()` yet).

## Phase D4 — Persistence: `TrainingHarness.save()` / `.load()`

- [ ] `save(self, path)`: `os.makedirs(path, exist_ok=True)`;
      `self.model.save(f"{path}/model.zip")`; write `metadata.json` with
      the exact schema from DRL_PLAN.md ("Model persistence" section):
      `reward_fn` (its `__name__`), `model_cls` (its `__name__`),
      `model_kwargs`, `horizon`, `on_the_play`, `action_space_size` (22,
      `len(ACTIONS)`), `observation_dim` (90), `total_timesteps_trained`
      (tracked as an instance attribute, updated by `train()`),
      `train_seed`, `timestamp` (`datetime.now().isoformat()`).
- [ ] `TrainingHarness.load(cls, path, reward_fn, model_cls, horizon=6,
      on_the_play=True)` (classmethod): read `metadata.json`; if
      `metadata["reward_fn"] != reward_fn.__name__` or
      `metadata["horizon"] != horizon` or `metadata["action_space_size"]
      != len(ACTIONS)` or `metadata["observation_dim"] != 90`: raise
      `ValueError` naming exactly which field mismatched. Otherwise build
      a harness the same way `__init__` does, then replace `self.model`
      with `model_cls.load(f"{path}/model.zip", env=self.env)`.
- [ ] Sanity check: train a tiny harness for a handful of timesteps
      (reuses Phase D5's mechanism once it exists — this check can be
      written now and just skipped/deferred until D5 lands), save it,
      reload it via `load(...)` with matching args (succeeds), then
      reload it again deliberately passing `horizon=5` (must raise
      `ValueError` mentioning "horizon").

## Phase D5 — Training: `TrainingHarness.train()`

- [ ] `train(self, total_timesteps, save_path=None)`:
      `self.model.learn(total_timesteps=total_timesteps)`;
      `self.total_timesteps_trained = (existing or 0) +
      total_timesteps`; if `save_path`, call `self.save(save_path)`.
- [ ] Sanity check: `harness.train(total_timesteps=500)` on a fresh
      `MaskablePPO` harness completes without error in well under a
      minute and produces a `model.zip` when `save_path` is given.

## Phase D6 — Evaluation: `TrainingHarness.evaluate()`

- [ ] `evaluate(self, num_games, horizon=None, seed=0) -> list[tuple]`:
      build a `choose_action(state)` closure — `build_observation(state,
      horizon)` → `self.model.predict(obs, action_masks=self.env
      .action_masks_for(state))` (see note below) → map the returned
      action index back to a real action via `ACTIONS`, mirroring
      `game.policy_choose_action`'s return shape (a zero-arg callable or
      `None`). Run it through `game.run_game(rng, on_the_play, horizon,
      choose_action)` for `num_games` independent games (one shared
      `random.Random(seed)` stream, exactly like `game.simulate_many`).
      Return the same `(turn_assembled, turn_online)` pair list.
- [ ] **Implementation note**: `action_masks()` as written in D2.3 reads
      `self.state`, which belongs to the *training* `TronEnv` instance.
      Evaluation plays games via `game.run_game` directly (not through
      `env.step`), so it needs a stateless variant —
      refactor `action_masks()` in D2.3 to delegate to a free function
      `legal_action_mask(state) -> np.ndarray[bool]` that both the env
      method and `evaluate()` call, rather than duplicating the logic.
      (Flagging here since it's easy to miss until evaluation is actually
      being wired up.)
- [ ] Sanity check: `harness.evaluate(num_games=20, seed=0)` returns 20
      well-formed pairs (`None` or `1..horizon` for each element,
      matching Phase 6's original sanity-check shape from
      `game._phase6_sanity_check`); feed the result through `game
      .aggregate_results`/`game.print_report` unmodified and confirm they
      accept it without error (proves the return shape is truly
      compatible, not just superficially similar).

## Phase D7 — Thin run scripts

- [ ] `train_drl.py`: imports `rewards.assembled_with_resource_quality`,
      `sb3_contrib.MaskablePPO`, constructs a `TrainingHarness`, calls
      `.train(total_timesteps=..., save_path="models/pilot")`. Every
      piece-selection choice (which reward, which model class, which
      kwargs) is a plain variable at the top of this file — this is
      exactly the "swap by editing a few lines" mechanism from the
      dependency-injection decision.
- [ ] `evaluate_drl.py`: two modes — (a) evaluate a harness right after
      training (imported from a just-run `train_drl.py` session or
      re-trained inline), (b) `TrainingHarness.load("models/pilot",
      reward_fn=..., model_cls=...)` then `.evaluate(num_games=...)` cold,
      with no training step — the "load and test later" flow from
      DRL_PLAN.md. Prints via `game.print_report`.
- [ ] Sanity check: running `train_drl.py` then `evaluate_drl.py` (mode
      b, loading what (a) just saved) end-to-end produces a report table
      with no errors.

## Phase D8 — Pilot run (this plan's actual deliverable)

Matches DRL_PLAN.md's "Training design" pilot criteria exactly:

- [ ] 50 random-legal-action episodes (Phase D2.4's last check, at
      full scale) confirm the environment itself is solid before
      spending any training time on it.
- [ ] `train_drl.py` with `net_arch=[64, 64]`, `total_timesteps` in the
      10,000–20,000 range, single (non-vectorized) environment. Record
      actual wall-clock.
- [ ] `evaluate_drl.py` against a modest game count (e.g. 200–500, not
      100,000 yet) — sanity-check the trained policy's numbers are
      *plausible* (not necessarily good: turn_assembled distribution
      should look like a real game distribution, not e.g. always `None`
      or always turn 1).
- [ ] Report: pilot wall-clock, whether training completed without
      error, and the small-scale evaluation numbers. **Explicitly not**
      a verdict on policy quality — that's what the full run (sized after
      this) and the eventual 100,000-game evaluation are for.
- [ ] Decision point (same shape as every prior pilot in this project):
      given the measured wall-clock, agree on the full run's
      `total_timesteps`/parallel-env count before spending real time on
      it, and separately agree on the real evaluation game count before
      spending real time on that.
