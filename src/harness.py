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

from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv

import game
import drl_env
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
    name, battlefield with tapped status, floating mana_pool by color,
    never library contents/order (the trained model's own observation
    space never sees library order either, only remaining-by-name counts
    -- see VISUALIZER_PLAN.md).

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
        "exile": [c.name for c, _plotted_turn in state.exile],
        "mana_pool": dict(state.mana_pool),  # copy -- state.mana_pool mutates in place (mana.py, turn.untap_step)
        # Running non-combat damage counter against the implicit opponent
        # (game/state.py's own note: no modeled opponent state beyond
        # this) -- 0 and never read by any deck whose terminated_fn isn't
        # one of terminated.damage_threshold_terminated's instances, but
        # cheap enough to always include rather than special-case by deck.
        "damage_dealt": state.damage_dealt,
    }
    snapshot["resource_quality"] = rewards.resource_quality_components(state)
    return snapshot


def _snapshot_pending(state):
    """Whatever the model can currently see about an in-progress multi-step
    resolution (state.pending_resolution, game/resolution.py) -- the
    hidden information a decision at this step is actually based on:
    scry/surveil's revealed-but-undecided cards and what's already been
    kept/disposed, search_fetch's full set of matching library cards (not
    just the one eventually chosen), etc. None when no resolution is
    pending (an ordinary Play-land/Cast-spell/mana-tap action). Every
    *_options helper used here is re-exported flat off the `game` package
    already (game/__init__.py), so no new imports are needed."""
    pending = state.pending_resolution
    if pending is None:
        return None
    kind = pending["kind"]
    snap = {"kind": kind}
    if kind in ("scry", "surveil"):
        snap["current_card"] = pending["remaining"][0].name if pending["remaining"] else None
        snap["remaining"] = [c.name for c in pending["remaining"]]
        snap["kept"] = [c.name for c in pending["kept"]]
        snap["disposed"] = [c.name for c in pending["disposed"]]
    elif kind == "search_fetch":
        snap["library_matches"] = game.search_fetch_options(state)
    elif kind == "choose_permanent":
        snap["battlefield_matches"] = game.choose_permanent_options(state)
    elif kind == "discard":
        snap["hand_options"] = game.discard_options(state)
    elif kind == "sacrifice":
        snap["sacrifice_options"] = game.sacrifice_options(state)
    elif kind == "madness_decision":
        snap["card"] = pending["card_def"].name
    return snap


def finalize_scores(state, reward_fn, scoring_fns, horizon):
    """The full scores dict for one finished game: reward_fn (mandatory,
    called with done=True since this only ever runs at a game's true end,
    keyed by its own __name__) plus each additional scoring_fns entry,
    each computed once and keyed by its own __name__ -- a name-keyed dict
    rather than a positional list so a log is self-describing across
    configs with different reward/scoring functions (dict insertion order
    is preserved, so reward_fn's entry is always first). A game that
    never terminated by the horizon gets every score forced to 0.0 here,
    centrally, rather than relying on each scoring function to remember
    its own failure check (reward_fn's own internal check stays too, as
    belt-and-suspenders, but this is what actually guarantees it for any
    scoring function that might forget)."""
    names = [reward_fn.__name__] + [fn.__name__ for fn in scoring_fns]
    if state.turn_won is None:
        return {name: 0.0 for name in names}
    values = [reward_fn(state, True, horizon)] + [fn(state) for fn in scoring_fns]
    return dict(zip(names, values))


class _GameLogger:
    """Per-game narrative for harness.evaluate()'s optional log_path: opening
    hand, every substantive action taken (each turn's draw included, as its
    own step) with whatever it fetched and whatever hidden information the
    model actually saw when deciding, final state, and the named scores --
    built purely by observing game.GameState through the same choose_action
    closure evaluate() already uses, never touching game.py. `steps` is a
    single flat list (no per-turn wrapper) since nothing downstream ever
    needs actions grouped by turn separately from the linear sequence the
    viewer steps through -- each step just carries its own turn number."""

    def __init__(self, reward_fn, horizon, scoring_fns=None):
        self.reward_fn = reward_fn
        self.horizon = horizon
        self.scoring_fns = scoring_fns or []
        self.opening_hand_state = None
        self.steps = []
        self._last_turn_number = None
        self._last_hand_names = None
        self._pending_names = None
        self._pending_action_name = None
        self._pending_decision = None
        self._pending_fallback = False
        self._pending_battlefield_before = None
        self._pending_tapped_ids = None

    def observe(self, state):
        """Called at the top of every choose_action call, before that call's
        decision -- this is where a new turn (and its draw) is detected."""
        hand_names = [c.name for c in state.hand]

        if self.opening_hand_state is None:
            self.opening_hand_state = _snapshot_state(state)
            self._last_hand_names = list(hand_names)
            self._last_turn_number = state.turn_number

        if state.turn_number != self._last_turn_number:
            drew = _diff_added(self._last_hand_names, hand_names)
            drew_card = drew[0] if drew else None
            self._last_turn_number = state.turn_number
            self._last_hand_names = hand_names
            if drew_card is not None:
                # Its own steppable entry -- otherwise arrow-key stepping
                # would show hand size jump with no step explaining why.
                self.steps.append({
                    "turn": state.turn_number,
                    "action": "Draw a card",
                    "fetched": [drew_card],
                    "left_battlefield": [],
                    "tapped_for_cost": [],
                    "decision": None,
                    "fallback": False,
                    "state_after": _snapshot_state(state),
                })

    def before_action(self, state, action_name, fallback=False):
        # hand + battlefield combined: a card moving between these two zones
        # (e.g. a land being played, a spell being cast) nets to zero in this
        # combined view, since it's just relocated, not new. Only a card
        # arriving from OUTSIDE both zones (the library -- a search result or
        # an ability-triggered draw) shows up as a net addition, which is
        # exactly what "fetched" should mean. Simpler and more robust than
        # special-casing each action type's own card separately.
        self._pending_names = [c.name for c in state.hand] + [p.card_def.name for p in state.battlefield]
        self._pending_action_name = action_name
        self._pending_fallback = fallback
        # Captured before execute_fn runs and consumes/mutates
        # state.pending_resolution -- this is the hidden information (scry
        # reveals, search matches, ...) the model actually saw to make this
        # choice.
        self._pending_decision = _snapshot_pending(state)
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
        self.steps.append({
            "turn": state.turn_number,
            "action": self._pending_action_name,
            "fetched": fetched,
            "left_battlefield": left_battlefield,
            "tapped_for_cost": tapped_for_cost,
            "decision": self._pending_decision,
            "fallback": self._pending_fallback,
            "state_after": _snapshot_state(state),
        })
        self._last_hand_names = [c.name for c in state.hand]  # keep the next turn's draw-diff baseline current

    def finalize(self, state, game_index):
        return {
            "game_index": game_index,
            "scores": finalize_scores(state, self.reward_fn, self.scoring_fns, self.horizon),
            "turn_won": state.turn_won,
            "opening_hand_state": self.opening_hand_state,
            "steps": self.steps,
            "end_state": _snapshot_state(state),
        }


def _make_env(reward_fn, decklist, terminated_fn, horizon, on_the_play, seed, pending_kinds, combat_enabled,
              token_card_defs=()):
    """Zero-arg factory for DummyVecEnv -- each call builds one fresh
    DeckEnv wrapped in Monitor. Monitor is required here: SB3 only
    auto-wraps an env in Monitor when it is NOT already a VecEnv (see
    stable_baselines3.common.base_class.BaseAlgorithm._wrap_env), so once
    TrainingHarness hands model_cls a pre-built DummyVecEnv, skipping this
    wrap would silently blank the rollout/ep_rew_mean console stats for
    the whole n_envs>1 path."""
    def _init():
        return Monitor(drl_env.DeckEnv(
            reward_fn, decklist=decklist, terminated_fn=terminated_fn,
            horizon=horizon, on_the_play=on_the_play, seed=seed, pending_kinds=pending_kinds,
            combat_enabled=combat_enabled, token_card_defs=token_card_defs,
        ))
    return _init


class TrainingHarness:
    # Deck-parameterized (MULTI_DECK_PLAN.md Phase M4/M7): no deck gets a
    # default here (not even Tron) -- decklist/terminated_fn/pending_kinds
    # are always the caller's own (e.g. game.parse_decklist_file(...),
    # terminated.tron_terminated, game.derive_pending_kinds(decklist)). A
    # second deck/model is just different arguments passed in here --
    # never a change to this file.
    def __init__(self, reward_fn, model_cls, decklist, terminated_fn, pending_kinds, model_kwargs=None,
                 horizon=6, on_the_play=True, seed=0, scoring_fns=None, n_envs=1, combat_enabled=False,
                 token_card_defs=()):
        self.reward_fn = reward_fn
        self.model_cls = model_cls
        self.model_kwargs = model_kwargs or {}
        self.decklist = decklist
        self.terminated_fn = terminated_fn
        self.horizon = horizon
        self.on_the_play = on_the_play
        self.seed = seed
        self.n_envs = n_envs
        self.pending_kinds = pending_kinds
        # rakdos madness / mono red madness only -- default off, same as
        # DeckEnv's own combat_enabled (see its docstring). Not part of
        # load()'s metadata mismatch-check: combat is behavioral only (like
        # terminated_fn), it never changes action_space_size/observation_dim.
        self.combat_enabled = combat_enabled
        # Tokens this deck's own cards can create at runtime whose activated
        # ability (if any) needs an action-table entry -- e.g. boggles'
        # Eldrazi Spawn (Malevolent Rumble). Defaults to () so every
        # existing caller (none of which currently pass this) is
        # unaffected. IS part of load()'s mismatch-check below (via
        # action_space_size), same as pending_kinds: adding/removing a
        # token's own actions changes the action space just like a new
        # card would.
        self.token_card_defs = token_card_defs
        self.total_timesteps_trained = 0
        # MULTI_DECK_PLAN.md Phase M6: reward_fn (score 1) is mandatory --
        # called every env step during training, and the sort key for
        # evaluate()'s logs. scoring_fns is an arbitrary-length list of
        # additional (state) -> float scores, each computed once at game
        # end (never mid-episode, never during training) purely for
        # human/eval-time consumption. See finalize_scores below for the
        # centrally-enforced failure-zeroing rule shared by all of them.
        self.scoring_fns = list(scoring_fns) if scoring_fns else []

        # Computed directly from (decklist, EFFECT_REGISTRY) rather than
        # borrowed off self.env's own attributes (as DeckEnv.actions/
        # .pass_action/.observation_dim used to be read below) -- required
        # once self.env can be a VecEnv (n_envs>1), which has no such
        # attributes of its own.
        self.actions = drl_env.build_action_table(
            decklist, game.EFFECT_REGISTRY, token_card_defs=token_card_defs, pending_kinds=pending_kinds,
        )
        self.pass_action = next(i for i, (name, _legal, _execute) in enumerate(self.actions) if name == "Pass")
        self.observation_dim = drl_env.observation_dim_for(decklist, pending_kinds)

        if n_envs == 1:
            self.env = drl_env.DeckEnv(
                reward_fn, decklist=decklist, terminated_fn=terminated_fn,
                horizon=horizon, on_the_play=on_the_play, seed=seed, pending_kinds=pending_kinds,
                combat_enabled=combat_enabled, token_card_defs=token_card_defs,
            )
        else:
            self.env = DummyVecEnv([
                _make_env(
                    reward_fn, decklist, terminated_fn, horizon, on_the_play, seed + i, pending_kinds,
                    combat_enabled, token_card_defs=token_card_defs,
                )
                for i in range(n_envs)
            ])
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
            # StopTrainingOnMaxEpisodes multiplies by n_envs internally (its
            # own docstring: "in total for max_episodes * n_envs episodes"),
            # so divide here to keep max_episodes meaning "total episodes
            # across the whole run" regardless of n_envs -- a no-op at the
            # default n_envs=1.
            callback = StopTrainingOnMaxEpisodes(max_episodes=max(1, max_episodes // self.n_envs), verbose=1)
        self.model.learn(total_timesteps=total_timesteps, callback=callback)
        self.total_timesteps_trained = self.model.num_timesteps  # SB3's own authoritative count
        if save_path:
            self.save(save_path)

    # -- D6: evaluation ---------------------------------------------------

    def evaluate(self, num_games, horizon=None, seed=0, log_path=None, config_name=None):
        """Plays num_games real games through game.run_game directly (not
        through env.step). Returns a list of (turn_won, scores) pairs,
        directly comparable via game.print_report/game.aggregate_results.
        scores is a dict keyed by each score function's own __name__
        (reward_fn's entry always first) rather than a positional list, so
        it stays self-describing across configs with different reward/
        scoring functions. The heuristic-era game.simulate_many this shape
        used to match is gone as of Phase M5 -- every deck is always
        played by a DRL model.

        If log_path is given, also writes a JSON object `{"meta": ...,
        "games": [...]}` to that path: `meta` records the run identity
        (reward/scoring function names, horizon, on_the_play, seed, and
        config_name if given) so a log is self-describing without relying
        on its filename or a side-channel metadata.json; `games` is one
        record per game (opening hand, every substantive action taken with
        what it fetched and whatever hidden information the model saw to
        decide it, end state, and scores), sorted highest primary score
        first (each game's first scores entry, i.e. reward_fn's value --
        dict insertion order keeps that first). A game that never
        terminated by the horizon gets every score forced to 0.0 (see
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

                obs = drl_env.build_observation(state, self.decklist, horizon, self.pending_kinds)
                mask = drl_env.legal_action_mask(state, self.actions)
                try:
                    action, _ = self.model.predict(obs, action_masks=mask, deterministic=True)
                except TypeError:
                    # non-maskable SB3 algorithm (plain PPO/A2C/...): no action_masks kwarg
                    action, _ = self.model.predict(obs, deterministic=True)
                action = int(action)
                fallback = not mask[action]
                if fallback:
                    # Same reasoning as DeckEnv.step()'s fallback
                    # (MULTI_DECK_PLAN.md Phase M4e): PASS_ACTION isn't a
                    # safe universal substitute anymore -- it's illegal
                    # whenever a resolution is pending. Substitute the
                    # first currently-legal action instead.
                    legal_indices = [i for i, ok in enumerate(mask) if ok]
                    action = legal_indices[0]
                if action == self.pass_action:
                    return None

                name, _, execute_fn = self.actions[action]
                if log is None:
                    return lambda: execute_fn(state)

                def wrapped_execute(state=state, name=name, execute_fn=execute_fn, log=log, fallback=fallback):
                    log.before_action(state, name, fallback=fallback)
                    execute_fn(state)
                    log.after_action(state)
                return wrapped_execute

            state = game.run_game(
                self.decklist, self.terminated_fn, rng, self.on_the_play, horizon, choose_action,
                combat_enabled=self.combat_enabled,
            )
            scores = finalize_scores(state, self.reward_fn, self.scoring_fns, horizon)
            results.append((state.turn_won, scores))

            if log is not None:
                game_logs.append(log.finalize(state, game_index))

        if log_path is not None:
            game_logs.sort(key=lambda g: next(iter(g["scores"].values())), reverse=True)
            log_doc = {
                "meta": {
                    "config_name": config_name,
                    "reward_fn": self.reward_fn.__name__,
                    "scoring_fns": [fn.__name__ for fn in self.scoring_fns],
                    "horizon": horizon,
                    "on_the_play": self.on_the_play,
                    "seed": seed,
                    # Only present for a damage-race deck (see
                    # terminated.damage_threshold_terminated) -- None for
                    # e.g. Tron's controls_all_tron_types, which has no
                    # such notion.
                    "win_threshold": getattr(self.terminated_fn, "threshold", None),
                },
                "games": game_logs,
            }
            os.makedirs(os.path.dirname(log_path) or ".", exist_ok=True)
            with open(log_path, "w") as f:
                json.dump(log_doc, f, indent=2)

        return results

    # -- D4: persistence ---------------------------------------------------

    def _metadata(self):
        return {
            "reward_fn": self.reward_fn.__name__,
            "model_cls": self.model_cls.__name__,
            "model_kwargs": self.model_kwargs,
            "horizon": self.horizon,
            "on_the_play": self.on_the_play,
            "action_space_size": len(self.actions),
            "observation_dim": self.observation_dim,
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
    def load(cls, path, reward_fn, model_cls, decklist, terminated_fn, pending_kinds,
              horizon=6, on_the_play=True, scoring_fns=None, combat_enabled=False, token_card_defs=()):
        with open(os.path.join(path, "metadata.json")) as f:
            metadata = json.load(f)

        current_action_space_size = len(drl_env.build_action_table(
            decklist, game.EFFECT_REGISTRY, token_card_defs=token_card_defs, pending_kinds=pending_kinds,
        ))
        current_observation_dim = drl_env.observation_dim_for(decklist, pending_kinds)

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
            scoring_fns=scoring_fns, pending_kinds=pending_kinds, combat_enabled=combat_enabled,
            token_card_defs=token_card_defs,
        )
        harness.model = model_cls.load(os.path.join(path, "model.zip"), env=harness.env)
        harness.total_timesteps_trained = metadata["total_timesteps_trained"]
        return harness
