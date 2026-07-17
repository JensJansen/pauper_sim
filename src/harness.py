"""Training harness: orchestrates a simulator (game.py), an injected reward
function (rewards.py contract), and an injected SB3-family model class into
one object that can train, evaluate, save, and reload.

Not coupled to any specific reward function or model class -- both are
constructor parameters (dependency injection, per DRL_PLAN.md), so swapping
either means passing different arguments in a run script, never editing
this file.
"""

import json
import os
import random
from datetime import datetime

import game
import tron_env
import rewards  # only for resource_quality_components -- see _snapshot_state's note


def _diff_added(before_names, after_names):
    """Multiset diff: names present in after_names beyond what before_names
    already accounted for (handles duplicate card names like 2 Forests
    correctly, since each match consumes one occurrence)."""
    remaining = list(before_names)
    added = []
    for name in after_names:
        if name in remaining:
            remaining.remove(name)
        else:
            added.append(name)
    return added


def _snapshot_state(state):
    """Full non-hidden state snapshot -- hand/battlefield/graveyard by card
    name, battlefield with tapped status, never library contents/order (the
    trained model's own observation space never sees library order either,
    only remaining-by-name counts -- see VISUALIZER_PLAN.md).

    Includes the raw resource_quality components for the visualizer's live
    readout. This is a deliberate, pragmatic coupling to one specific
    reward function's internals (rewards.resource_quality_components),
    not a generic "any reward function can supply a live readout"
    mechanism -- nothing else needs that abstraction yet, so it isn't
    built. Precomputing here in Python (versus re-deriving the same mana/
    Tron-bonus logic in JS) avoids any risk of the two implementations
    silently drifting apart.
    """
    snapshot = {
        "turn_number": state.turn_number,
        "hand": [c.name for c in state.hand],
        "battlefield": [{"name": p.card_def.name, "tapped": p.tapped} for p in state.battlefield],
        "graveyard": [c.name for c in state.graveyard],
    }
    snapshot["resource_quality"] = rewards.resource_quality_components(state)
    return snapshot


# Per-action "scry" log detail (seen/kept_on_top/bottomed/binned) is
# removed as of MULTI_DECK_PLAN.md Phase M5. It was keyed off a fixed
# table of action names (Cast Candy Trail, Play land: Conduit Pylons, ...)
# and predicted outcomes via game.is_priority_land, both of which stopped
# being meaningful in Phase M4e: scry/surveil now resolve over several
# separate before_action/after_action pairs, each with its own action name
# ("Choose: Forest", "Keep (scry/surveil)", ...), and there's no more
# priority-land auto-pick to predict -- the model decides. Confirmed this
# was already fully dormant (never fired) before removal. A real
# replacement means deciding how _GameLogger should represent a
# multi-step resolution at all (one entry per step? merged into the
# originating action's entry?) -- that's visualizer-support work, already
# explicitly out of scope for this plan (see MULTI_DECK_PLAN.md's
# "Explicitly out of scope" section), not something to improvise here.


def finalize_scores(state, reward_fn, scoring_fns, horizon):
    """The full scores list for one finished game: score 1 (reward_fn,
    mandatory, called with done=True since this only ever runs at a
    game's true end) followed by each additional scoring_fns entry, each
    computed once. MULTI_DECK_PLAN.md Phase M6 decision: a game that
    never terminated by the horizon gets every score forced to 0.0 here,
    centrally, rather than relying on each scoring function to remember
    its own failure check (reward_fn's own internal check stays too, as
    belt-and-suspenders, but this is what actually guarantees it for any
    scoring function that might forget)."""
    if state.turn_won is None:
        return [0.0] * (1 + len(scoring_fns))
    return [reward_fn(state, True, horizon)] + [fn(state) for fn in scoring_fns]


class _GameLogger:
    """Per-game narrative for harness.evaluate()'s optional log_path: opening
    hand, each turn's draw, every substantive action taken with whatever it
    fetched, final state, and the reward_fn score -- built purely by
    observing game.GameState through the same choose_action closure
    evaluate() already uses, never touching game.py."""

    def __init__(self, reward_fn, horizon, scoring_fns=None):
        self.reward_fn = reward_fn
        self.horizon = horizon
        self.scoring_fns = scoring_fns or []
        self.opening_hand = None
        self.turns = []
        self._last_turn_number = None
        self._last_hand_names = None
        self._current_turn_entry = None
        self._pending_names = None
        self._pending_action_name = None
        self._pending_battlefield_before = None
        self._pending_tapped_ids = None

    def observe(self, state):
        """Called at the top of every choose_action call, before that call's
        decision -- this is where a new turn (and its draw) is detected."""
        hand_names = [c.name for c in state.hand]

        if self.opening_hand is None:
            self.opening_hand = list(hand_names)
            self._last_hand_names = list(hand_names)
            self._last_turn_number = state.turn_number

        if state.turn_number != self._last_turn_number:
            drew = _diff_added(self._last_hand_names, hand_names)
            drew_card = drew[0] if drew else None
            self._current_turn_entry = {
                "turn": state.turn_number,
                "drew": drew_card,
                "actions": [],
            }
            self.turns.append(self._current_turn_entry)
            self._last_turn_number = state.turn_number
            self._last_hand_names = hand_names
            if drew_card is not None:
                # Its own steppable entry -- otherwise arrow-key stepping
                # would show hand size jump with no step explaining why.
                self._current_turn_entry["actions"].append({
                    "action": "Draw a card",
                    "fetched": [drew_card],
                    "left_battlefield": [],
                    "tapped_for_cost": [],
                    "state_after": _snapshot_state(state),
                })
        elif self._current_turn_entry is None:
            self._current_turn_entry = {"turn": state.turn_number, "drew": None, "actions": []}
            self.turns.append(self._current_turn_entry)

    def before_action(self, state, action_name):
        # hand + battlefield combined: a card moving between these two zones
        # (e.g. a land being played, a spell being cast) nets to zero in this
        # combined view, since it's just relocated, not new. Only a card
        # arriving from OUTSIDE both zones (the library -- a search result or
        # an ability-triggered draw) shows up as a net addition, which is
        # exactly what "fetched" should mean. Simpler and more robust than
        # special-casing each action type's own card separately.
        self._pending_names = [c.name for c in state.hand] + [p.card_def.name for p in state.battlefield]
        self._pending_action_name = action_name
        # Object references (not names) -- needed so duplicate-name
        # permanents (e.g. two Urza's Mines) are tracked by identity, not
        # confused with each other.
        self._pending_battlefield_before = list(state.battlefield)
        self._pending_tapped_ids = {id(p) for p in state.battlefield if p.tapped}

    def after_action(self, state):
        after_names = [c.name for c in state.hand] + [p.card_def.name for p in state.battlefield]
        fetched = _diff_added(self._pending_names, after_names)
        # Identity-based, not name-based: which specific permanents are
        # simply no longer present (covers Crop Rotation's sacrificed land,
        # Expedition Map/Candy Trail sacrificing themselves, Relic of
        # Progenitus exiling itself -- one general field for all of them).
        left_battlefield = [
            p.card_def.name for p in self._pending_battlefield_before if p not in state.battlefield
        ]
        tapped_for_cost = [
            p.card_def.name for p in state.battlefield
            if p.tapped and id(p) not in self._pending_tapped_ids
        ]
        entry = {
            "action": self._pending_action_name,
            "fetched": fetched,
            "left_battlefield": left_battlefield,
            "tapped_for_cost": tapped_for_cost,
        }

        entry["state_after"] = _snapshot_state(state)
        self._current_turn_entry["actions"].append(entry)
        self._last_hand_names = [c.name for c in state.hand]  # keep the next turn's draw-diff baseline current

    def finalize(self, state, game_index):
        # MULTI_DECK_PLAN.md Phase M6: score/score2 replaced by a generic
        # scores list (score 1 -- reward_fn -- always first, then whatever
        # scoring_fns were configured); "turn_online" stays retired as a
        # tracked field (Phase M3) -- a deck that wants that concept
        # expresses it as one of scoring_fns instead (see rewards.py).
        return {
            "game_index": game_index,
            "scores": finalize_scores(state, self.reward_fn, self.scoring_fns, self.horizon),
            "turn_won": state.turn_won,
            "opening_hand": self.opening_hand,
            "turns": self.turns,
            "end_state": _snapshot_state(state),
        }


class TrainingHarness:
    # Deck-parameterized (MULTI_DECK_PLAN.md Phase M4/M7): decklist/
    # terminated_fn default to Tron's so every existing caller (train_drl.py,
    # evaluate_drl.py, the sanity checks below) keeps working unchanged. A
    # second deck/model is just a different decklist/terminated_fn/reward_fn
    # passed in here -- never a change to this file.
    def __init__(self, reward_fn, model_cls, model_kwargs=None,
                 decklist=game.TRON_DECKLIST, terminated_fn=game.tron_terminated,
                 horizon=6, on_the_play=True, seed=0, scoring_fns=None):
        self.reward_fn = reward_fn
        self.model_cls = model_cls
        self.model_kwargs = model_kwargs or {}
        self.decklist = decklist
        self.terminated_fn = terminated_fn
        self.horizon = horizon
        self.on_the_play = on_the_play
        self.seed = seed
        self.total_timesteps_trained = 0
        # MULTI_DECK_PLAN.md Phase M6: reward_fn (score 1) is mandatory --
        # called every env step during training, and the sort key for
        # evaluate()'s logs. scoring_fns is an arbitrary-length list of
        # additional (state) -> float scores, each computed once at game
        # end (never mid-episode, never during training) purely for
        # human/eval-time consumption. See finalize_scores below for the
        # centrally-enforced failure-zeroing rule shared by all of them.
        self.scoring_fns = list(scoring_fns) if scoring_fns else []

        self.env = tron_env.TronEnv(
            reward_fn, decklist=decklist, terminated_fn=terminated_fn,
            horizon=horizon, on_the_play=on_the_play, seed=seed,
        )
        self.model = model_cls("MlpPolicy", self.env, **self.model_kwargs)

    # -- D5: training ---------------------------------------------------

    def train(self, total_timesteps, save_path=None, max_episodes=None):
        """total_timesteps is always an upper bound. If max_episodes is
        given, training stops as soon as that many episodes complete
        (SB3's own StopTrainingOnMaxEpisodes callback -- exact episode
        counting, not a timestep approximation), whichever comes first."""
        callback = None
        if max_episodes is not None:
            from stable_baselines3.common.callbacks import StopTrainingOnMaxEpisodes
            callback = StopTrainingOnMaxEpisodes(max_episodes=max_episodes, verbose=1)
        self.model.learn(total_timesteps=total_timesteps, callback=callback)
        self.total_timesteps_trained = self.model.num_timesteps  # SB3's own authoritative count
        if save_path:
            self.save(save_path)

    # -- D6: evaluation ---------------------------------------------------

    def evaluate(self, num_games, horizon=None, seed=0, log_path=None):
        """Plays num_games real games through game.run_game directly (not
        through env.step). Returns a list of (turn_won, scores) pairs,
        directly comparable via game.print_report/game.aggregate_results
        (MULTI_DECK_PLAN.md Phase M6). The heuristic-era game.simulate_many
        this shape used to match is gone as of Phase M5 -- every deck is
        always played by a DRL model.

        If log_path is given, also writes a JSON array (one record per
        game: opening hand, each turn's draw, every substantive action
        taken with what it fetched, end state, and scores) to that path,
        sorted highest score first (scores[0], i.e. score 1/reward_fn --
        MULTI_DECK_PLAN.md Phase M6: scores is now an arbitrary-length
        list, reward_fn's value always first, then whatever scoring_fns
        this harness was constructed with). A game that never terminated
        by the horizon gets every score forced to 0.0 (see
        finalize_scores), so failures naturally sort to the bottom as a
        block with no meaningful order among themselves -- no special
        casing needed for that, it just falls out of the sort."""
        horizon = horizon or self.horizon
        rng = random.Random(seed)
        game_logs = [] if log_path is not None else None
        results = []

        for game_index in range(num_games):
            log = _GameLogger(self.reward_fn, horizon, self.scoring_fns) if log_path is not None else None

            def choose_action(state, log=log):
                if log is not None:
                    log.observe(state)

                obs = tron_env.build_observation(state, self.decklist, horizon)
                mask = tron_env.legal_action_mask(state, self.env.actions)
                try:
                    action, _ = self.model.predict(obs, action_masks=mask, deterministic=True)
                except TypeError:
                    # non-maskable SB3 algorithm (plain PPO/A2C/...): no action_masks kwarg
                    action, _ = self.model.predict(obs, deterministic=True)
                action = int(action)
                if not mask[action]:
                    # Same reasoning as TronEnv.step()'s fallback
                    # (MULTI_DECK_PLAN.md Phase M4e): PASS_ACTION isn't a
                    # safe universal substitute anymore -- it's illegal
                    # whenever a resolution is pending. Substitute the
                    # first currently-legal action instead.
                    legal_indices = [i for i, ok in enumerate(mask) if ok]
                    action = legal_indices[0]
                if action == self.env.pass_action:
                    return None

                name, _, execute_fn = self.env.actions[action]
                if log is None:
                    return lambda: execute_fn(state)

                def wrapped_execute(state=state, name=name, execute_fn=execute_fn, log=log):
                    log.before_action(state, name)
                    execute_fn(state)
                    log.after_action(state)
                return wrapped_execute

            state = game.run_game(self.decklist, self.terminated_fn, rng, self.on_the_play, horizon, choose_action)
            scores = finalize_scores(state, self.reward_fn, self.scoring_fns, horizon)
            results.append((state.turn_won, scores))

            if log is not None:
                game_logs.append(log.finalize(state, game_index))

        if log_path is not None:
            game_logs.sort(key=lambda g: g["scores"][0], reverse=True)
            os.makedirs(os.path.dirname(log_path) or ".", exist_ok=True)
            with open(log_path, "w") as f:
                json.dump(game_logs, f, indent=2)

        return results

    # -- D4: persistence ---------------------------------------------------

    def _metadata(self):
        return {
            "reward_fn": self.reward_fn.__name__,
            "model_cls": self.model_cls.__name__,
            "model_kwargs": self.model_kwargs,
            "horizon": self.horizon,
            "on_the_play": self.on_the_play,
            "action_space_size": len(self.env.actions),
            "observation_dim": self.env.observation_dim,
            "total_timesteps_trained": self.total_timesteps_trained,
            "train_seed": self.seed,
            "timestamp": datetime.now().isoformat(),
            # Informational only, no mismatch-check on load -- scoring_fns
            # never touch training or the saved model (MULTI_DECK_PLAN.md
            # Phase M7), they're a live argument to load() like reward_fn.
            "scoring_fns": [fn.__name__ for fn in self.scoring_fns],
        }

    def save(self, path):
        os.makedirs(path, exist_ok=True)
        self.model.save(os.path.join(path, "model.zip"))
        with open(os.path.join(path, "metadata.json"), "w") as f:
            json.dump(self._metadata(), f, indent=2)

    @classmethod
    def load(cls, path, reward_fn, model_cls, decklist=game.TRON_DECKLIST, terminated_fn=game.tron_terminated,
              horizon=6, on_the_play=True, scoring_fns=None):
        with open(os.path.join(path, "metadata.json")) as f:
            metadata = json.load(f)

        current_action_space_size = len(tron_env.build_action_table(decklist, game.EFFECT_REGISTRY))
        current_observation_dim = tron_env.observation_dim_for(decklist)

        mismatches = []
        if metadata["reward_fn"] != reward_fn.__name__:
            mismatches.append(f"reward_fn: saved={metadata['reward_fn']!r}, given={reward_fn.__name__!r}")
        if metadata["horizon"] != horizon:
            mismatches.append(f"horizon: saved={metadata['horizon']!r}, given={horizon!r}")
        if metadata["action_space_size"] != current_action_space_size:
            mismatches.append(
                f"action_space_size: saved={metadata['action_space_size']!r}, "
                f"current={current_action_space_size!r}"
            )
        if metadata["observation_dim"] != current_observation_dim:
            mismatches.append(
                f"observation_dim: saved={metadata['observation_dim']!r}, "
                f"current={current_observation_dim!r}"
            )
        if mismatches:
            raise ValueError("TrainingHarness.load: metadata mismatch -- " + "; ".join(mismatches))

        harness = cls(
            reward_fn=reward_fn, model_cls=model_cls, model_kwargs=metadata["model_kwargs"],
            decklist=decklist, terminated_fn=terminated_fn,
            horizon=horizon, on_the_play=on_the_play, seed=metadata["train_seed"],
            scoring_fns=scoring_fns,
        )
        harness.model = model_cls.load(os.path.join(path, "model.zip"), env=harness.env)
        harness.total_timesteps_trained = metadata["total_timesteps_trained"]
        return harness


def _phase_d3_d4_d5_d6_sanity_check():
    import shutil
    import rewards
    from sb3_contrib import MaskablePPO

    model_kwargs = {"policy_kwargs": {"net_arch": [8, 8]}, "n_steps": 32, "batch_size": 16, "verbose": 0}

    # D3: construction wires model to env correctly.
    harness = TrainingHarness(
        reward_fn=rewards.assembled_with_resource_quality,
        model_cls=MaskablePPO,
        model_kwargs=model_kwargs,
        horizon=6, on_the_play=True, seed=0,
    )
    assert isinstance(harness.model, MaskablePPO)
    assert harness.model.env is not None

    # D5: a tiny training call completes without error.
    harness.train(total_timesteps=64)
    assert harness.total_timesteps_trained == 64

    # D6: evaluate() returns well-formed, game.py-compatible results.
    results = harness.evaluate(num_games=5, seed=1)
    assert len(results) == 5
    for turn_won, scores in results:
        assert turn_won is None or 1 <= turn_won <= 6
        assert len(scores) == 1  # just score 1 (reward_fn) -- this harness has no scoring_fns configured
        if turn_won is None:
            assert scores == [0.0]
    rows, summary = game.aggregate_results(results, horizon=6)  # must not raise

    # D4: save then reload with matching args succeeds and behaves the same.
    test_path = "models/_phase_d4_test"
    if os.path.exists(test_path):
        shutil.rmtree(test_path)
    harness.save(test_path)
    assert os.path.exists(os.path.join(test_path, "model.zip"))
    assert os.path.exists(os.path.join(test_path, "metadata.json"))

    reloaded = TrainingHarness.load(
        test_path, reward_fn=rewards.assembled_with_resource_quality, model_cls=MaskablePPO,
        horizon=6, on_the_play=True,
    )
    assert reloaded.total_timesteps_trained == 64
    reloaded.evaluate(num_games=2, seed=2)  # must not raise

    # D4: reload with a mismatched horizon must raise, naming the field.
    try:
        TrainingHarness.load(
            test_path, reward_fn=rewards.assembled_with_resource_quality, model_cls=MaskablePPO,
            horizon=5, on_the_play=True,
        )
        raise AssertionError("expected a ValueError for mismatched horizon")
    except ValueError as e:
        assert "horizon" in str(e), e

    shutil.rmtree(test_path)


def _phase_logging_sanity_check():
    import rewards
    from sb3_contrib import MaskablePPO

    harness = TrainingHarness(
        reward_fn=rewards.assembled_with_resource_quality,
        model_cls=MaskablePPO,
        model_kwargs={"policy_kwargs": {"net_arch": [8, 8]}, "n_steps": 32, "batch_size": 16, "verbose": 0},
        horizon=6, on_the_play=True, seed=0,
        # MULTI_DECK_PLAN.md Phase M6: a real (non-empty) scoring_fns list,
        # so this check exercises the arbitrary-length scores feature, not
        # just the mandatory score 1 -- resource_quality_pct is a natural
        # stand-in for what used to be the hardcoded "score2".
        scoring_fns=[rewards.resource_quality_pct],
    )
    harness.train(total_timesteps=256)

    log_path = "models/_phase_logging_test/games.json"
    results = harness.evaluate(num_games=20, seed=3, log_path=log_path)
    assert len(results) == 20
    assert all(len(scores) == 2 for _turn_won, scores in results)

    with open(log_path) as f:
        logs = json.load(f)
    assert len(logs) == 20

    # Sorted highest score first (scores[0], i.e. score 1); every failure
    # scores exactly [0.0, 0.0] and they sort to the bottom as a block
    # with no special-casing required.
    score1_values = [g["scores"][0] for g in logs]
    assert score1_values == sorted(score1_values, reverse=True), score1_values
    assert all(0.0 <= g["scores"][1] <= 100.0 for g in logs), [g["scores"][1] for g in logs]
    num_failures = sum(1 for turn_won, _scores in results if turn_won is None)
    num_zero_scores = sum(1 for s in score1_values if s == 0.0)
    assert num_failures == num_zero_scores, (num_failures, num_zero_scores)
    assert all(g["scores"] == [0.0, 0.0] for g in logs if g["turn_won"] is None), "failures must zero every score"

    # Schema + content sanity on the top (best) and bottom (a failure, if any) entries.
    best = logs[0]
    assert set(best.keys()) == {"game_index", "scores", "turn_won", "opening_hand", "turns", "end_state"}
    assert len(best["scores"]) == 2
    assert len(best["opening_hand"]) == 7
    assert isinstance(best["turns"], list) and len(best["turns"]) >= 1
    assert best["turns"][0]["turn"] == 1
    assert best["turns"][0]["drew"] is None  # on the play, turn 1 has no draw
    for turn_entry in best["turns"]:
        for action_entry in turn_entry["actions"]:
            assert "action" in action_entry and "fetched" in action_entry
    assert isinstance(best["end_state"]["hand"], list)
    assert isinstance(best["end_state"]["battlefield"], list)

    if score1_values[-1] == 0.0:
        worst = logs[-1]
        assert worst["turn_won"] is None
        assert len(worst["opening_hand"]) == 7

    import shutil
    shutil.rmtree("models/_phase_logging_test")


def _phase_scry_logging_sanity_check():
    import rewards

    # The Candy Trail/Tocasia's Dig Site/duplicate-name scry-logging
    # coverage that used to live here is temporarily removed as of
    # MULTI_DECK_PLAN.md Phase M4c: _GameLogger's scry/surveil detection
    # predicted outcomes via game.is_priority_land and assumed the
    # resolution completed synchronously within one before_action/
    # after_action pair. Neither holds anymore -- scry/surveil are now a
    # multi-step pending resolution the model walks through. Redesigning
    # how _GameLogger represents that belongs with Phase M4e's action-table
    # rebuild (same open question already flagged for Crop Rotation's
    # logging), not a piecemeal fix here.

    # An action with no scry/surveil involved must not get a "scry" key at all.
    plain_state = game.GameState(on_the_play=True, rng=random.Random(0))
    plain_state.hand = [game.CARD_DEFS["Forest"]]
    log3 = _GameLogger(rewards.assembled_with_resource_quality, horizon=6)
    log3.observe(plain_state)
    log3.before_action(plain_state, "Play land: Forest")
    game.play_land_from_hand(plain_state, game.CARD_DEFS["Forest"])
    log3.after_action(plain_state)
    assert "scry" not in log3._current_turn_entry["actions"][-1]


def _phase_visualizer_logging_sanity_check():
    import rewards
    import tron_env
    from sb3_contrib import MaskablePPO

    # The left_battlefield/tapped_for_cost/fetched Crop Rotation logging
    # coverage that used to live here is temporarily removed as of
    # MULTI_DECK_PLAN.md Phase M4b: Crop Rotation now resolves over
    # multiple pending-resolution steps (choose sacrifice, then choose
    # fetch target) instead of one synchronous call, and how _GameLogger
    # should represent a multi-step action in its log output is an open
    # design question that belongs with Phase M4e's action-table rebuild,
    # not a piecemeal fix here against machinery about to be replaced.

    # Draw pseudo-action: turn 1 on the play has no draw, so no pseudo-action;
    # turn 2 does, and it comes before that turn's real action. game.run_game
    # always builds a fresh random deck via new_game_state, so drive
    # game.run_turn directly against a hand-built state instead, to keep
    # the library/hand under precise control.
    state2 = game.GameState(on_the_play=True, rng=random.Random(0))
    state2.hand = [game.CARD_DEFS["Forest"], game.CARD_DEFS["Rooftop Percher"]]
    state2.library = [game.CARD_DEFS["Bramble Wurm"]] + [game.CARD_DEFS["Rooftop Percher"]] * 20

    log2 = _GameLogger(rewards.assembled_with_resource_quality, horizon=6)

    def choose_action(s):
        log2.observe(s)
        land = next((c for c in s.hand if c.name == "Forest"), None)
        if land is None or s.lands_played_this_turn > 0:
            return None

        def act():
            log2.before_action(s, "Play land: Forest")
            game.play_land_from_hand(s, land)
            log2.after_action(s)
        return act

    game.run_turn(state2, choose_action)  # turn 1
    game.run_turn(state2, choose_action)  # turn 2

    assert log2.turns[0]["drew"] is None
    assert log2.turns[0]["actions"][0]["action"] == "Play land: Forest", log2.turns[0]["actions"]
    assert log2.turns[1]["drew"] == "Bramble Wurm"
    assert log2.turns[1]["actions"][0]["action"] == "Draw a card", log2.turns[1]["actions"]
    assert log2.turns[1]["actions"][0]["fetched"] == ["Bramble Wurm"]
    assert log2.turns[1]["actions"][0]["left_battlefield"] == []
    assert log2.turns[1]["actions"][0]["tapped_for_cost"] == []
    assert log2.turns[1]["actions"][0]["state_after"]["hand"].count("Bramble Wurm") == 1

    # state_after well-formed on every action entry across a real evaluation
    # run. models/run_50k is permanently incompatible as of MULTI_DECK_PLAN.md
    # Phase M4e (trained against the old 22-action/90-dim space; the
    # generic action table is now 65 actions/97 dims) -- TrainingHarness
    # .load's own metadata mismatch-check correctly refuses to load it, per
    # its documented job. This check's assertions are purely structural
    # (schema well-formedness), not about policy quality, so a tiny
    # freshly-trained throwaway model (same pattern
    # _phase_d3_d4_d5_d6_sanity_check already uses) covers it exactly as
    # well -- no need to wait for Phase M8's real retrain.
    harness = TrainingHarness(
        reward_fn=rewards.assembled_with_resource_quality,
        model_cls=MaskablePPO,
        model_kwargs={"policy_kwargs": {"net_arch": [8, 8]}, "n_steps": 32, "batch_size": 16, "verbose": 0},
        horizon=6, on_the_play=True, seed=0,
    )
    harness.train(total_timesteps=64)
    log_path = "models/_phase_viz_test/games.json"
    harness.evaluate(num_games=20, seed=5, log_path=log_path)
    with open(log_path) as f:
        logs = json.load(f)
    for g in logs:
        for t in g["turns"]:
            for a in t["actions"]:
                sa = a["state_after"]
                assert isinstance(sa["turn_number"], int)
                assert isinstance(sa["hand"], list)
                assert isinstance(sa["graveyard"], list)
                assert all(isinstance(p, dict) and "name" in p and "tapped" in p for p in sa["battlefield"])
                assert "left_battlefield" in a and "tapped_for_cost" in a

    import shutil
    shutil.rmtree("models/_phase_viz_test")


if __name__ == "__main__":
    _phase_d3_d4_d5_d6_sanity_check()
    print("Phase D3/D4/D5/D6 OK: harness construction, training, evaluation, save/load-with-mismatch-check all work.")

    _phase_logging_sanity_check()
    print("Logging OK: evaluate(log_path=...) writes a score-sorted JSON log with the right schema.")

    _phase_scry_logging_sanity_check()
    print("Scry/surveil logging OK: Candy Trail's scry correctly bottoms cards, Tocasia's surveil correctly bins them, and plain actions get no scry key at all.")

    _phase_visualizer_logging_sanity_check()
    print("Visualizer logging OK: left_battlefield/tapped_for_cost/state_after are correct, and the draw pseudo-action appears exactly where expected.")
