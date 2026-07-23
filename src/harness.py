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
        # Bottom-to-top order (index 0 = bottom of stack, last = next to
        # resolve) -- every spell fully paid for but not yet resolved (see
        # game.push_to_stack); empty outside of a cast awaiting a "Pass".
        "stack": [entry["card_def"].name for entry in state.stack],
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


def _slot_labeled(pairs):
    """(name, slot) pairs -> "Name (slot k)" strings -- docs/MULTIPLAYER_
    GAPS.md's "Permanent identity" convention (two same-named permanents
    aren't interchangeable once only one might be the actual target), same
    display format drl_env's own "Choose target: ..." action names already
    use. Shared by every _PENDING_OPTIONS entry below that targets a
    specific permanent rather than offering a plain name."""
    return [f"{name} (slot {slot})" for name, slot in pairs]


# kind -> (snapshot field name, options(state) -> list[str]). Every
# pending_resolution kind whose entire decision detail is "the list of
# names/labeled targets the model is currently choosing among" -- 8 of
# game/resolution.py's 11 kinds -- dispatched uniformly through this table
# instead of one near-identical elif branch apiece. Mirrors src/viz's own
# SIMPLE_DECISION_FIELDS (GameView.jsx's "kind -> [label, field]"): same
# data, same dispatch shape, both ends of this pipe. scry/surveil
# (multiple fields, not option-list-shaped) and madness_decision (a single
# scalar read straight off the pending dict, no state query needed) don't
# fit this shape and stay their own branches in _snapshot_pending below.
_PENDING_OPTIONS = {
    "search_fetch": ("library_matches", game.search_fetch_options),
    "choose_permanent": ("battlefield_matches", lambda s: _slot_labeled(game.choose_permanent_options(s))),
    "choose_opponent_permanent": (
        # 2-player combat only (blocking's own nested "which attacker"
        # consult, game.resolution.declare_blocker_assignment) -- state.
        # opponent already means the right thing here, since this kind is
        # only ever entered with state.active_idx already flipped to the
        # referencing player (see begin_choose_opponent_permanent's own
        # docstring).
        "opponent_battlefield_matches", lambda s: _slot_labeled(game.choose_opponent_permanent_options(s))
    ),
    "discard": ("hand_options", game.discard_options),
    "sacrifice": ("sacrifice_options", game.sacrifice_options),
    "mulligan_decision": ("options", game.mulligan_decision_options),
    "mulligan_bottom": ("bottom_options", game.bottom_options),
    "order_triggers": ("trigger_options", game.order_triggers_options),
    "declare_blockers": (
        # No dedicated *_options helper in resolution.py for this kind --
        # the actual "which blocker" choice is drl_env's own action-table
        # eligibility check (creature_block_eligible), not a resolution
        # primitive. This is the complementary half: which of the
        # attacker's declared attackers still need a blocker assigned,
        # exactly declare_blocker_assignment's own nested predicate
        # (state.opponent means the attacker here, from the
        # already-flipped defender's point of view).
        "unblocked_attackers",
        lambda s: _slot_labeled((p.card_def.name, p.slot) for p in s.opponent.attackers if p not in s.opponent.blocked_by),
    ),
}


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
    elif kind == "madness_decision":
        snap["card"] = pending["card_def"].name
    elif kind in _PENDING_OPTIONS:
        field, options_fn = _PENDING_OPTIONS[kind]
        snap[field] = options_fn(state)
    return snap


def _snapshot_player_state(state, player_state):
    """One seat's own zones + life/combat state -- the 2-player-log
    counterpart to _snapshot_state, scoped to a single PlayerState instead
    of reading the active-relative GameState properties (which only ever
    expose whichever seat is currently active -- see game/state.py's own
    "_active_player_property" docstring). Includes life_total/attackers/
    blocked_by, which _snapshot_state never captures because they're inert
    in 1-player mode; never includes state.stack (shared across both
    players, captured once at the step/game level instead -- see
    _snapshot_two_player_state). Takes the full `state` (not just
    `player_state`) because game.permanent_power/permanent_toughness below
    need it -- an Aura and its target are always on the same side, but
    those helpers still search state.players themselves rather than trust
    the caller's own active_idx (see effects/stats.py's own docstring).

    Each battlefield entry's "enchanting" is the (name, slot)-addressed
    target an Aura permanent is attached to (Permanent.flags["enchanting"],
    set once by cast_aura/state_based.py's own orphan-check reader,
    game/effects/casting.py's own docstring) -- None for every non-Aura
    permanent, which never sets this flag at all. One-directional by
    design (aura -> target, never the reverse on the target's own entry):
    a creature can carry several Auras at once, so "what enchants me" is a
    list computed by scanning for it, not a single field to store back on
    the creature -- left to the viewer (src/viz) to derive from this,
    rather than duplicating the same relationship both ways here.

    CREATURE entries additionally carry power/toughness (Aura bonuses
    already folded in, via game.permanent_power/permanent_toughness),
    base_power/base_toughness (the card's own printed stats, no bonuses),
    and keywords -- lets the viewer show a buffed creature's true current
    stats rather than just its base card text. Never present on a
    non-creature entry (base 0/no keywords for a land/artifact/enchantment
    is meaningless, not just uninteresting)."""
    return {
        "hand": [c.name for c in player_state.hand],
        "battlefield": [
            {
                "name": p.card_def.name, "tapped": p.tapped, "slot": p.slot,
                "enchanting": (
                    f"{p.flags['enchanting'].card_def.name} (slot {p.flags['enchanting'].slot})"
                    if p.flags.get("enchanting") is not None else None
                ),
                **(
                    {
                        "power": game.permanent_power(state, p),
                        "toughness": game.permanent_toughness(state, p),
                        "base_power": p.card_def.extra.get("power", 0),
                        "base_toughness": p.card_def.extra.get("toughness", 0),
                        "keywords": sorted(game.creature_keywords(state, p)),
                    }
                    if p.card_def.card_type == game.CardType.CREATURE
                    else {}
                ),
            }
            for p in player_state.battlefield
        ],
        "graveyard": [c.name for c in player_state.graveyard],
        "exile": [c.name for c, _plotted_turn in player_state.exile],
        "mana_pool": dict(player_state.mana_pool),
        "life_total": player_state.life_total,
        "damage_dealt": player_state.damage_dealt,
        "attackers": [f"{p.card_def.name} (slot {p.slot})" for p in player_state.attackers],
        "blocked_by": {
            f"{attacker.card_def.name} (slot {attacker.slot})": f"{blocker.card_def.name} (slot {blocker.slot})"
            for attacker, blocker in player_state.blocked_by.items()
        },
    }


def _snapshot_two_player_state(state):
    """Both seats' own snapshot (index-by-seat, same convention
    evaluate_two_player's own wins/turn_counts/action_counts already use)
    plus the one genuinely shared zone (the stack -- see GameState.stack's
    own docstring: one object shared by both players, not per-seat)."""
    return {
        "players": [_snapshot_player_state(state, p) for p in state.players],
        "stack": [entry["card_def"].name for entry in state.stack],
    }


def finalize_scores(state, reward_fn, scoring_fns, horizon, seat_idx=None):
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
    scoring function that might forget).

    seat_idx: 2-player games only (None -- the default -- preserves 1p
    behavior exactly). state.turn_won/turn_number alone never say WHO won
    once a real opponent exists (see drl_env._lost's own docstring), so a
    seat that lost (state.winner set to the OTHER seat) also gets every
    score forced to 0.0, on top of the existing "never terminated" gate.

    Also re-points state.active_idx at seat_idx (via drl_env._for_player)
    before calling reward_fn/scoring_fns, then restores it -- every
    reward/scoring function that reads board state at all (resource_
    quality-based ones: assembled_with_resource_quality, resource_
    quality_pct, tron_online_score) does so through state.hand/
    state.battlefield/etc., which are active-relative properties (game/
    state.py's own "_active_player_property" docstring) meaning "whichever
    seat happens to be active right now," not "seat_idx specifically."
    Without this flip, both seats would silently get whichever seat's
    board state.active_idx happened to be pointing at when the game
    ended -- wrong for exactly one of the two seats, every time they
    differ."""
    names = [reward_fn.__name__] + [fn.__name__ for fn in scoring_fns]
    if state.turn_won is None:
        return {name: 0.0 for name in names}
    if seat_idx is not None and drl_env._lost(state, seat_idx):
        return {name: 0.0 for name in names}
    compute = lambda s: [reward_fn(s, True, horizon)] + [fn(s) for fn in scoring_fns]
    values = compute(state) if seat_idx is None else drl_env._for_player(state, seat_idx, compute)
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


class _TwoPlayerGameLogger:
    """_GameLogger's 2-player counterpart, for harness.evaluate_two_player's
    own optional log_path -- same "observe game.GameState through the same
    choose_action closure, never touch game.py" construction, but every
    snapshot is BOTH seats' own zones (_snapshot_two_player_state), never
    just whichever one happens to be state.active_idx, plus the turn/actor
    attribution a 2-player game actually needs that 1-player never had to
    record: state.turn_player_idx (whose turn it structurally is) and
    state.active_idx AT THE MOMENT OF EACH DECISION (who is actually being
    asked to act right now -- a priority consult, most visibly blocking
    (game.turn._declare_blockers_gen), genuinely flips this away from the
    turn owner; see game/state.py's own GameState docstring). Index-by-seat
    throughout (self.opening_state[seat], scores[seat], ...), same
    convention evaluate_two_player's own wins/turn_counts/action_counts
    already use -- never a dict keyed by "agent_a"/"agent_b" or similar.

    Per-seat draw detection is diffed continuously (every observe() call,
    against that SPECIFIC seat's own last-recorded hand) rather than
    _GameLogger's "only re-check once state.turn_number changes" -- that
    once-per-turn-boundary approach silently drops a turn's automatic draw
    whenever a real priority round visits a player BEFORE that turn's own
    DRAW phase actually runs (any combat_enabled=True deck's UPKEEP phase,
    which -- unlike a non-combat deck's phase set -- has a real priority
    round of its own, called before DRAW's automatic draw_step; game/
    turn.py's own FULL_PHASES/_run_priority_round_gen). Continuous per-seat
    diffing has no such blind spot: the only way a seat's hand can grow
    between two observe() calls for that seat with no before_action/
    after_action pair in between (which already update the baseline
    themselves, see after_action below) is an automatic effect -- almost
    always that turn's own draw_step, occasionally another automatic
    hand-return effect (e.g. an on_draw_count trigger resolving off the
    stack) -- either way, real information nothing else in the log would
    otherwise capture at all."""

    def __init__(self, reward_fns, horizon, scoring_fns_by_seat):
        self.reward_fns = reward_fns
        self.horizon = horizon
        self.scoring_fns_by_seat = scoring_fns_by_seat
        self.opening_state = [None, None]
        self.steps = []
        self._last_hand_names = [None, None]  # None until that seat's first observe() -- marks "not seen yet"
        self._pending_names = None
        self._pending_action_name = None
        self._pending_decision = None
        self._pending_fallback = False
        self._pending_battlefield_before = None
        self._pending_tapped_ids = None
        self._pending_actor_idx = None

    def _make_step(self, state, actor_idx, action, fetched=(), left_battlefield=(), tapped_for_cost=(),
                    decision=None, fallback=False):
        return {
            "turn_number": state.turn_number,
            "turn_player_idx": state.turn_player_idx,
            "actor_idx": actor_idx,
            "phase": state.phase.value if state.phase is not None else None,
            "action": action,
            "fetched": list(fetched),
            "left_battlefield": list(left_battlefield),
            "tapped_for_cost": list(tapped_for_cost),
            "decision": decision,
            "fallback": fallback,
            "state_after": _snapshot_two_player_state(state),
        }

    def observe(self, state):
        """Called at the top of every choose_action call, before that
        call's decision -- same hook point _GameLogger.observe uses, just
        keyed by state.active_idx (whichever seat is actually being
        consulted right now, turn owner or not) instead of assumed to
        always be the same lone player."""
        seat = state.active_idx
        hand_names = [c.name for c in state.hand]
        if self._last_hand_names[seat] is None:
            # First time this seat has ever been consulted -- always during
            # the pregame mulligan phase (game.turn._run_mulligan_gen visits
            # every player index before turn 1 ever starts), so this is
            # that seat's true opening hand, not a mid-game draw.
            self.opening_state[seat] = _snapshot_player_state(state, state.players[seat])
            self._last_hand_names[seat] = hand_names
            return
        drew = _diff_added(self._last_hand_names[seat], hand_names)
        self._last_hand_names[seat] = hand_names
        if drew:
            self.steps.append(self._make_step(state, seat, "Draw a card", fetched=drew))

    def before_action(self, state, action_name, fallback=False):
        self._pending_names = [c.name for c in state.hand] + [p.card_def.name for p in state.battlefield]
        self._pending_action_name = action_name
        self._pending_fallback = fallback
        self._pending_decision = _snapshot_pending(state)
        self._pending_battlefield_before = list(state.battlefield)
        self._pending_tapped_ids = {id(p) for p in state.battlefield if p.tapped}
        self._pending_actor_idx = state.active_idx

    def after_action(self, state):
        after_names = [c.name for c in state.hand] + [p.card_def.name for p in state.battlefield]
        fetched = _diff_added(self._pending_names, after_names)
        # (name, slot)-addressed, not name-only: with two same-named
        # permanents in play (this deck's own norm -- 4-ofs, mirror match),
        # a bare name here can't say which specific physical copy left/got
        # tapped, exactly the ambiguity that made a real removal (the
        # untapped blocker, correctly, dying to combat damage) misreadable
        # in the viewer as "the tapped one must have been removed." Same
        # "Name (slot k)" convention _make_step's own attackers/blocked_by
        # already use.
        left_battlefield = [
            f"{p.card_def.name} (slot {p.slot})" for p in self._pending_battlefield_before
            if p not in state.battlefield
        ]
        tapped_for_cost = [
            f"{p.card_def.name} (slot {p.slot})" for p in state.battlefield
            if p.tapped and id(p) not in self._pending_tapped_ids
        ]
        self.steps.append(self._make_step(
            state, self._pending_actor_idx, self._pending_action_name,
            fetched=fetched, left_battlefield=left_battlefield, tapped_for_cost=tapped_for_cost,
            decision=self._pending_decision, fallback=self._pending_fallback,
        ))
        # Keep the ACTING seat's draw-diff baseline current -- same
        # reasoning _GameLogger.after_action's own trailing line gives,
        # just scoped to the one seat this action actually belonged to.
        self._last_hand_names[self._pending_actor_idx] = [c.name for c in state.hand]

    def finalize(self, state, game_index, starting_player_idx):
        return {
            "game_index": game_index,
            "starting_player_idx": starting_player_idx,
            "winner": state.winner,
            "turn_won": state.turn_won,
            "final_turn_number": state.turn_number,
            "scores": [
                finalize_scores(state, self.reward_fns[seat], self.scoring_fns_by_seat[seat], self.horizon,
                                 seat_idx=seat)
                for seat in range(2)
            ],
            "opening_state": {"players": self.opening_state, "stack": []},
            "steps": self.steps,
            "end_state": _snapshot_two_player_state(state),
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


def _make_two_player_env(reward_fn, decklist, terminated_fn, pending_kinds, opponent_decklist,
                          opponent_terminated_fn, opponent_pending_kinds, my_seat_idx, horizon, on_the_play,
                          seed, token_card_defs, opponent_token_card_defs, shaping_weight, shaping_gamma):
    """_make_env's two-player counterpart -- same Monitor-wrapping reason
    (n_envs>1 needs it, see _make_env's own docstring). opponent_model
    starts unset here (TrainingHarness.set_opponent_model wires it in once
    both sides' models exist -- see that method's own docstring)."""
    def _init():
        return Monitor(drl_env.TwoPlayerDeckEnv(
            reward_fn, decklist=decklist, terminated_fn=terminated_fn, pending_kinds=pending_kinds,
            opponent_decklist=opponent_decklist, opponent_terminated_fn=opponent_terminated_fn,
            opponent_pending_kinds=opponent_pending_kinds, my_seat_idx=my_seat_idx, horizon=horizon,
            on_the_play=on_the_play, seed=seed, token_card_defs=token_card_defs,
            opponent_token_card_defs=opponent_token_card_defs,
            shaping_weight=shaping_weight, shaping_gamma=shaping_gamma,
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
                 token_card_defs=(), opponent_decklist=None, opponent_terminated_fn=None,
                 opponent_pending_kinds=None, opponent_token_card_defs=(), my_seat_idx=0, shaping_weight=0.0,
                 vec_env_cls=DummyVecEnv):
        # Two-player mode (docs/MULTIPLAYER_ENGINE_PLAN.md's harness pass):
        # triggered purely by opponent_decklist being given, same "presence
        # of a second decklist is the mode switch" rule run.py applies to a
        # config's own JSON. This harness still trains exactly ONE model
        # (my_seat_idx's own) -- a full match needs a SECOND TrainingHarness
        # for the opponent's own decklist/model, cross-wired via
        # set_opponent_model and run together by train_two_player below;
        # nothing here builds that second harness itself.
        self.two_player = opponent_decklist is not None
        self.opponent_decklist = opponent_decklist
        self.opponent_terminated_fn = opponent_terminated_fn
        self.opponent_pending_kinds = opponent_pending_kinds
        self.opponent_token_card_defs = opponent_token_card_defs
        self.my_seat_idx = my_seat_idx
        # Opponent-visibility observation blocks (MULTIPLAYER_GAPS.md) --
        # same values TwoPlayerDeckEnv computes for itself, kept here too
        # so evaluate_two_player (which drives games directly through
        # game.run_multiplayer_game, never through self.env) can build the
        # identical observation shape its model actually trained on
        # without reaching into self.env's own internals.
        if self.two_player:
            self.opponent_total_cards = sum(qty for _name, qty, *_rest in opponent_decklist)
            self.opponent_creature_names, self.opponent_creature_copies = drl_env.creature_names_and_copies(
                opponent_decklist,
            )
            self.opponent_card_names, self.opponent_card_copies = drl_env._card_lookup(opponent_decklist)
        else:
            self.opponent_total_cards = None
            self.opponent_creature_names = None
            self.opponent_creature_copies = None
            self.opponent_card_names = None
            self.opponent_card_copies = None
        self.reward_fn = reward_fn
        self.model_cls = model_cls
        self.model_kwargs = model_kwargs or {}
        # Potential-based dense reward (MULTIPLAYER_GAPS.md), 2-player only
        # -- opt-in (default 0.0 -- no behavior change unless a config
        # asks for it), see TwoPlayerDeckEnv's own docstring for the full
        # design. shaping_gamma deliberately reuses the model's own PPO
        # discount (falling back to SB3's own default of 0.99 if
        # model_kwargs doesn't set one) rather than an independent value --
        # see TwoPlayerDeckEnv.__init__'s own comment on why they must
        # agree.
        self.shaping_weight = shaping_weight
        self.shaping_gamma = self.model_kwargs.get("gamma", 0.99)
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
        # Two-player mode: opponent_decklist/opponent_token_card_defs too --
        # evaluate_two_player dispatches through THIS table (harness.actions/
        # harness.pass_action), and a real blocking consult needs a "Choose
        # opponent's: <name> (slot k)" action addressing the OTHER side's
        # battlefield once a blocker's been assigned (declare_blocker_
        # assignment's own nested resolution) -- omitting it here left that
        # resolution with zero legal actions the instant blocking actually
        # fired during evaluation (same bug TwoPlayerDeckEnv.__init__ had
        # for its own self.actions/self.opponent_actions, caught by a live
        # smoke test rather than any narrower unit check).
        self.actions = drl_env.build_action_table(
            decklist, game.EFFECT_REGISTRY, token_card_defs=token_card_defs, pending_kinds=pending_kinds,
            opponent_decklist=opponent_decklist, opponent_token_card_defs=opponent_token_card_defs,
        )
        self.pass_action = next(i for i, (name, _legal, _execute) in enumerate(self.actions) if name == "Pass")
        # 2-player observation includes the opponent-visibility blocks on
        # top of the plain 1-player shape (drl_env.two_player_observation_
        # dim/TwoPlayerDeckEnv's own observation_dim) -- must match exactly,
        # since this value is what _metadata()/load()'s mismatch-check
        # actually compares.
        if self.two_player:
            self.observation_dim = drl_env.two_player_observation_dim(
                decklist, pending_kinds, opponent_decklist,
            )
        else:
            self.observation_dim = drl_env.observation_dim_for(decklist, pending_kinds)

        if self.two_player:
            if n_envs == 1:
                self.env = drl_env.TwoPlayerDeckEnv(
                    reward_fn, decklist=decklist, terminated_fn=terminated_fn, pending_kinds=pending_kinds,
                    opponent_decklist=opponent_decklist, opponent_terminated_fn=opponent_terminated_fn,
                    opponent_pending_kinds=opponent_pending_kinds, my_seat_idx=my_seat_idx, horizon=horizon,
                    on_the_play=on_the_play, seed=seed, token_card_defs=token_card_defs,
                    opponent_token_card_defs=opponent_token_card_defs,
                    shaping_weight=self.shaping_weight, shaping_gamma=self.shaping_gamma,
                )
            else:
                self.env = vec_env_cls([
                    _make_two_player_env(
                        reward_fn, decklist, terminated_fn, pending_kinds, opponent_decklist,
                        opponent_terminated_fn, opponent_pending_kinds, my_seat_idx, horizon, on_the_play,
                        seed + i, token_card_defs, opponent_token_card_defs,
                        self.shaping_weight, self.shaping_gamma,
                    )
                    for i in range(n_envs)
                ])
        elif n_envs == 1:
            self.env = drl_env.DeckEnv(
                reward_fn, decklist=decklist, terminated_fn=terminated_fn,
                horizon=horizon, on_the_play=on_the_play, seed=seed, pending_kinds=pending_kinds,
                combat_enabled=combat_enabled, token_card_defs=token_card_defs,
            )
        else:
            self.env = vec_env_cls([
                _make_env(
                    reward_fn, decklist, terminated_fn, horizon, on_the_play, seed + i, pending_kinds,
                    combat_enabled, token_card_defs=token_card_defs,
                )
                for i in range(n_envs)
            ])
        self.model = model_cls("MlpPolicy", self.env, **self.model_kwargs)
        if self.two_player:
            # Always MY OWN model, unlike set_opponent_model below (which
            # needs a SECOND harness -- train_two_player wires that one in).
            # Needed so TwoPlayerDeckEnv._play_opponent_turn can simulate my
            # own blocking decision while the opponent's whole turn runs
            # synchronously inside step() (TwoPlayerDeckEnv.own_model's own
            # docstring) -- re-set again in load() once self.model is
            # replaced by the reloaded one.
            self.set_own_model(self.model)

    def set_own_model(self, model):
        """Two-player mode only: point every underlying TwoPlayerDeckEnv's
        own_model at the given model object (always THIS harness's own
        model -- see __init__'s call right after self.model is built, and
        load()'s re-call once self.model is replaced by the reloaded one).
        Same Monitor/DummyVecEnv unwrapping as set_opponent_model."""
        envs = self.env.envs if isinstance(self.env, DummyVecEnv) else [self.env]
        for env in envs:
            base = env.env if isinstance(env, Monitor) else env
            base.own_model = model

    def set_opponent_model(self, model):
        """Two-player mode only: point every underlying TwoPlayerDeckEnv at
        the opponent's own live SB3 model OBJECT (not a snapshot) -- both
        sides then improve together with zero re-wiring needed between
        train_two_player's alternating bursts, since the object each env
        holds a reference to is the exact one the opponent's own
        TrainingHarness keeps mutating via .learn(). Reaches through
        Monitor/DummyVecEnv the same way n_envs>1 already wraps every env
        (see _make_env's own docstring for why Monitor is always there)."""
        envs = self.env.envs if isinstance(self.env, DummyVecEnv) else [self.env]
        for env in envs:
            base = env.env if isinstance(env, Monitor) else env
            base.opponent_model = model

    def episode_count(self):
        """Total episodes completed so far across every underlying env, via
        self.model.get_env() rather than self.env directly: at n_envs>1
        self.env IS the Monitor-wrapped DummyVecEnv we built ourselves
        (_make_env/_make_two_player_env), but at n_envs==1 self.env is the
        bare DeckEnv/TwoPlayerDeckEnv (no Monitor of our own -- see
        TrainingHarness.__init__), and SB3 only wraps that in its OWN
        Monitor+DummyVecEnv internally once handed to model_cls(...). Going
        through the model's own env is the one path that's always
        Monitor-wrapped either way, so episode_lengths is always there to
        read. train_two_player's own episode-based stop condition needs this
        directly (unlike 1-player train() below, two-player training drives
        model.learn() itself in bursts, so SB3's own StopTrainingOnMaxEpisodes
        callback -- reset at the start of every separate learn() call -- has
        no cumulative count to stop on across bursts the way a single
        1-player .learn() call does)."""
        return sum(len(env.episode_lengths) for env in self.model.get_env().envs)

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
        casing needed for that, it just falls out of the sort.

        Two-player mode only: use the module-level evaluate_two_player
        instead -- this plays a solitaire game.run_game, which has no
        opponent to speak of."""
        if self.two_player:
            raise NotImplementedError(
                "TrainingHarness.evaluate() is 1-player only -- use harness.evaluate_two_player(harness_a, "
                "harness_b, ...) for a two-player harness (see docs/MULTIPLAYER_ENGINE_PLAN.md)."
            )
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
            # two_player IS mismatch-checked on load (below) -- silently
            # loading a model trained against a live opponent back into
            # 1-player mode, or vice versa, would use an env whose
            # observation the model was never actually trained against
            # (TwoPlayerDeckEnv and DeckEnv share observation_dim/
            # action_space_size for the same decklist, so those two checks
            # alone wouldn't catch this). my_seat_idx is informational only
            # (which seat a saved model played never changes what it
            # learned about its OWN cards).
            "two_player": self.two_player,
            "my_seat_idx": self.my_seat_idx,
            # Informational only, no mismatch-check on load -- same
            # "behavioral, doesn't change action_space_size/observation_dim"
            # bucket as combat_enabled above. shaping_gamma itself isn't
            # saved separately: it's always re-derived from model_kwargs
            # (already saved/restored above), never an independent value.
            "shaping_weight": self.shaping_weight,
        }

    def save(self, path):
        os.makedirs(path, exist_ok=True)
        self.model.save(os.path.join(path, "model.zip"))
        with open(os.path.join(path, "metadata.json"), "w") as f:
            json.dump(self._metadata(), f, indent=2)

    @classmethod
    def load(cls, path, reward_fn, model_cls, decklist, terminated_fn, pending_kinds,
              horizon=6, on_the_play=True, scoring_fns=None, combat_enabled=False, token_card_defs=(),
              opponent_decklist=None, opponent_terminated_fn=None, opponent_pending_kinds=None,
              opponent_token_card_defs=(), my_seat_idx=0, shaping_weight=0.0, vec_env_cls=DummyVecEnv):
        with open(os.path.join(path, "metadata.json")) as f:
            metadata = json.load(f)

        current_action_space_size = len(drl_env.build_action_table(
            decklist, game.EFFECT_REGISTRY, token_card_defs=token_card_defs, pending_kinds=pending_kinds,
            opponent_decklist=opponent_decklist, opponent_token_card_defs=opponent_token_card_defs,
        ))
        current_two_player = opponent_decklist is not None
        if current_two_player:
            current_observation_dim = drl_env.two_player_observation_dim(decklist, pending_kinds, opponent_decklist)
        else:
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
        if metadata.get("two_player", False) != current_two_player:
            mismatches.append(
                f"two_player: saved={metadata.get('two_player', False)!r}, current={current_two_player!r}"
            )
        if mismatches:
            raise ValueError("TrainingHarness.load: metadata mismatch -- " + "; ".join(mismatches))

        harness = cls(
            reward_fn=reward_fn, model_cls=model_cls, model_kwargs=metadata["model_kwargs"],
            decklist=decklist, terminated_fn=terminated_fn,
            horizon=horizon, on_the_play=on_the_play, seed=metadata["train_seed"],
            scoring_fns=scoring_fns, pending_kinds=pending_kinds, combat_enabled=combat_enabled,
            token_card_defs=token_card_defs, opponent_decklist=opponent_decklist,
            opponent_terminated_fn=opponent_terminated_fn, opponent_pending_kinds=opponent_pending_kinds,
            opponent_token_card_defs=opponent_token_card_defs, my_seat_idx=my_seat_idx,
            shaping_weight=shaping_weight, vec_env_cls=vec_env_cls,
        )
        harness.model = model_cls.load(os.path.join(path, "model.zip"), env=harness.env)
        if harness.two_player:
            harness.set_own_model(harness.model)  # __init__'s own_model wiring pointed at the PRE-reload model object
        harness.total_timesteps_trained = metadata["total_timesteps_trained"]
        return harness


# ---------------------------------------------------------------------------
# Two-player training/evaluation coordinators. Each operates on a PAIR of
# TrainingHarness instances already built in two-player mode (opponent_
# decklist given, my_seat_idx 0 and 1 respectively, one pointed at the
# other's own decklist as ITS opponent) -- neither function builds a
# harness itself, run.py's own two-player dispatch does that (see
# run.py's decklist_2-triggered path).
# ---------------------------------------------------------------------------

def train_two_player(harness_a, harness_b, total_timesteps, burst_timesteps=2000,
                      max_episodes=None, save_path_a=None, save_path_b=None):
    """Alternates short .learn() bursts between two harnesses trained
    against each other -- the confirmed "opponent-as-environment" design
    (see TwoPlayerDeckEnv's own docstring): harness_a.env auto-plays
    harness_b's CURRENT model during harness_b's turns, and vice versa.
    set_opponent_model stores the model OBJECT itself (not a snapshot), so
    both sides face an ever-improving live opponent with zero re-wiring
    needed between bursts -- each .learn() call mutates the very object
    the other side's env already holds a reference to.

    burst_timesteps: how often the two sides swap training turns, not a
    tunable that changes the end result in any principled way -- small
    enough that neither side trains for a very long stretch against a
    frozen-in-place opponent snapshot, without bursts so short that SB3's
    own per-.learn() setup overhead starts to matter. total_timesteps is
    split evenly between both sides, same as one 1-player .train() call's
    total_timesteps is that side's own full budget.

    max_episodes: same "stop as soon as this many episodes complete, total_
    timesteps is only the upper bound" contract as 1-player TrainingHarness.
    train() -- but checked here via harness.episode_count() between bursts,
    rather than via SB3's own StopTrainingOnMaxEpisodes callback: that
    callback's own episode counter resets every separate .learn() call, and
    this function calls .learn() once per burst per side, so it has no
    cumulative count to stop on across bursts the way a single 1-player
    .learn() call does. Checked against EITHER side reaching the target
    (both sides train the same number of bursts, so they stay close
    together regardless), same spirit as 1-player's own max_episodes."""
    harness_a.set_opponent_model(harness_b.model)
    harness_b.set_opponent_model(harness_a.model)

    trained = 0
    while trained < total_timesteps:
        if max_episodes is not None and (
            harness_a.episode_count() >= max_episodes or harness_b.episode_count() >= max_episodes
        ):
            break
        step = min(burst_timesteps, total_timesteps - trained)
        harness_a.model.learn(total_timesteps=step, reset_num_timesteps=False)
        harness_b.model.learn(total_timesteps=step, reset_num_timesteps=False)
        trained += step

    harness_a.total_timesteps_trained = harness_a.model.num_timesteps
    harness_b.total_timesteps_trained = harness_b.model.num_timesteps
    if save_path_a:
        harness_a.save(save_path_a)
    if save_path_b:
        harness_b.save(save_path_b)


def evaluate_two_player(harness_a, harness_b, num_games, horizon=None, seed=0, log_path=None, config_name=None):
    """Plays num_games real 2-player games (game.run_multiplayer_game)
    between harness_a's and harness_b's CURRENT models, deterministic
    (same convention 1-player evaluate() uses). Returns (wins_a, wins_b,
    draws, turn_counts, action_counts) -- action_counts is each game's
    total choose_action call count (both sides combined, one per real
    decision including Pass -- the same granularity _TwoPlayerGameLogger's
    steps use), for efficiency metrics like actions-per-turn; paired
    index-for-index with turn_counts.

    If log_path is given, also writes a JSON object `{"meta": ...,
    "games": [...]}` to that path -- same top-level shape as 1-player
    evaluate()'s own log, but every game record is 2-player-shaped
    (_TwoPlayerGameLogger.finalize): both seats' opening hand/battlefield/
    graveyard/exile/life_total, every substantive action attributed to the
    seat that actually made it (not just the turn owner -- a blocking
    consult acts through the DEFENDER), and both seats' own scores. Sorted
    by seat 0's own primary score (agent_a's reward_fn value), descending
    -- an arbitrary but consistent tiebreak, same "first scores entry"
    convention 1-player evaluate() uses, picking one side since a 2-player
    game has no single shared score to sort a pair of games by.

    Building the log requires knowing which action index the model picked
    and whether it was a fallback substitution -- drl_env.model_choose_action
    doesn't expose either (by design: it only ever returns the zero-arg
    callable/None the turn generator's send() protocol expects), so the
    logging path re-implements that same predict/mask/fallback sequence
    inline, exactly as 1-player evaluate() already does for the identical
    reason. The far more common no-log path (training-time opponent
    evaluation, plain CLI runs) is untouched -- still the cheaper
    model_choose_action call, zero behavior change."""
    horizon = horizon or harness_a.horizon
    rng = random.Random(seed)
    harnesses = (harness_a, harness_b)
    wins = [0, 0]
    draws = 0
    turn_counts = []
    action_counts = []
    game_logs = [] if log_path is not None else None

    for game_index in range(num_games):
        starting_idx = rng.randint(0, 1)
        action_count = [0]
        log = _TwoPlayerGameLogger(
            reward_fns=[harness_a.reward_fn, harness_b.reward_fn], horizon=horizon,
            scoring_fns_by_seat=[harness_a.scoring_fns, harness_b.scoring_fns],
        ) if log_path is not None else None

        def choose_action(state, action_count=action_count, log=log):
            if log is not None:
                log.observe(state)
            action_count[0] += 1
            seat = state.active_idx
            harness = harnesses[seat]
            obs = drl_env.build_two_player_observation(
                state, seat, harness.decklist, horizon, harness.pending_kinds,
                harness.opponent_total_cards, harness.opponent_creature_names, harness.opponent_creature_copies,
                harness.opponent_card_names, harness.opponent_card_copies,
            )
            if log is None:
                return drl_env.model_choose_action(
                    state, obs, harness.model, harness.actions, harness.pass_action, deterministic=True,
                )

            mask = drl_env.legal_action_mask(state, harness.actions)
            try:
                action, _ = harness.model.predict(obs, action_masks=mask, deterministic=True)
            except TypeError:
                action, _ = harness.model.predict(obs, deterministic=True)
            action = int(action)
            fallback = not mask[action]
            if fallback:
                legal_indices = [i for i, ok in enumerate(mask) if ok]
                action = legal_indices[0]
            if action == harness.pass_action:
                return None

            name, _legal, execute_fn = harness.actions[action]

            def wrapped_execute(state=state, name=name, execute_fn=execute_fn, log=log, fallback=fallback):
                log.before_action(state, name, fallback=fallback)
                execute_fn(state)
                log.after_action(state)
            return wrapped_execute

        state = game.run_multiplayer_game(
            decklists=[harness_a.decklist, harness_b.decklist],
            terminated_fns=[harness_a.terminated_fn, harness_b.terminated_fn],
            rng=rng, starting_player_idx=starting_idx, choose_action=choose_action,
            horizon=horizon, combat_enabled=True,
        )
        turn_counts.append(state.turn_number)
        action_counts.append(action_count[0])
        if state.winner is None:
            draws += 1
        else:
            wins[state.winner] += 1

        if log is not None:
            game_logs.append(log.finalize(state, game_index, starting_idx))

    if log_path is not None:
        game_logs.sort(key=lambda g: next(iter(g["scores"][0].values())), reverse=True)
        log_doc = {
            "meta": {
                "config_name": config_name,
                "horizon": horizon,
                "seed": seed,
                "seats": [
                    {
                        "reward_fn": h.reward_fn.__name__,
                        "scoring_fns": [fn.__name__ for fn in h.scoring_fns],
                        "win_threshold": getattr(h.terminated_fn, "threshold", None),
                    }
                    for h in harnesses
                ],
            },
            "games": game_logs,
        }
        os.makedirs(os.path.dirname(log_path) or ".", exist_ok=True)
        with open(log_path, "w") as f:
            json.dump(log_doc, f, indent=2)

    return wins[0], wins[1], draws, turn_counts, action_counts


if __name__ == "__main__":
    # ponytail self-check: no pytest in this project, mirrors the
    # assert-based demo convention every other module here uses -- run via
    # `python harness.py` from src/. End-to-end two-player smoke test: two
    # tiny real MaskablePPO models (same mirror decklists drl_env.py's own
    # TwoPlayerDeckEnv self-check uses -- Mountain+Bolt vs. a pure-Mountain
    # punching bag) actually trained against each other via
    # train_two_player, saved, reloaded through load()'s own mismatch
    # check, then played against each other via evaluate_two_player. Not
    # a claim about learned play quality -- total_timesteps here is
    # deliberately tiny, just enough to exercise every moving part (env
    # wiring, opponent cross-referencing, save/load, evaluate) at least
    # once.
    import shutil
    import tempfile

    from sb3_contrib import MaskablePPO

    import terminated as terminated_module  # noqa: F401 -- not used here (terminated_fn is a trivial lambda below), imported only to confirm the module still loads alongside this self-check

    # finalize_scores(seat_idx=...) correctness (hand-built state, no model
    # needed): a board-state-reading reward (assembled_with_resource_quality,
    # via resource_quality) must score each seat off THAT seat's own
    # battlefield, not whichever seat state.active_idx happens to be
    # pointing at -- the bug this pass fixed (drl_env._for_player wasn't
    # being used at all before). Seat 0 gets 2 non-land permanents, seat 1
    # gets 0 -- if the flip weren't happening, both seats would read seat
    # 0's board (whichever one state.active_idx pointed at) and come out
    # identical.
    from game.cards import CardDef, CardType
    from game.state import GameState, PlayerState, Permanent

    def _creature(name):
        return Permanent(CardDef(name, CardType.CREATURE, None, None))

    seat0 = PlayerState(on_the_play=True)
    seat0.battlefield = [_creature("Bear"), _creature("Wolf")]
    seat1 = PlayerState(on_the_play=False)
    seat1.battlefield = []
    fs_state = GameState(on_the_play=True, players=[seat0, seat1])
    fs_state.active_idx = 0  # whoever happens to be active when the game ends -- must not leak into seat 1's score
    fs_state.turn_number = 3
    fs_state.turn_won = 3
    fs_state.winner = None  # a draw (horizon/safety-cap) -- neither seat is "lost", both scored normally

    score_0 = finalize_scores(fs_state, rewards.assembled_with_resource_quality, [], horizon=40, seat_idx=0)
    score_1 = finalize_scores(fs_state, rewards.assembled_with_resource_quality, [], horizon=40, seat_idx=1)
    assert score_0["assembled_with_resource_quality"] > score_1["assembled_with_resource_quality"], (
        "seat 0 (2 creatures) must outscore seat 1 (0 creatures) -- if this fails, finalize_scores stopped "
        "reading each seat's OWN board"
    )
    assert fs_state.active_idx == 0  # _for_player must restore it, not leave it pointed at whichever seat scored last

    # _lost zeroing still applies per seat on top of the per-seat board read.
    fs_state.winner = 1
    assert finalize_scores(fs_state, rewards.assembled_with_resource_quality, [], horizon=40, seat_idx=0) == {
        "assembled_with_resource_quality": 0.0
    }
    assert finalize_scores(fs_state, rewards.assembled_with_resource_quality, [], horizon=40, seat_idx=1)[
        "assembled_with_resource_quality"
    ] > 0.0
    print("harness.py finalize_scores(seat_idx=...) self-check: OK")

    deck_a = [("Mountain", 20), ("Lightning Bolt", 10)]
    deck_b = [("Mountain", 20)]
    pending_a = game.derive_pending_kinds(deck_a)
    pending_b = game.derive_pending_kinds(deck_b)
    tiny_model_kwargs = {
        "policy_kwargs": {"net_arch": [8, 8]}, "verbose": 0, "device": "cpu", "n_steps": 32, "batch_size": 16,
    }

    tmp_dir = tempfile.mkdtemp(prefix="azul_2p_selfcheck_")
    try:
        harness_a = TrainingHarness(
            reward_fn=rewards.strict_binary_reward, model_cls=MaskablePPO, decklist=deck_a,
            terminated_fn=lambda s: False, pending_kinds=pending_a, model_kwargs=tiny_model_kwargs,
            horizon=40, on_the_play=True, seed=0, opponent_decklist=deck_b,
            opponent_terminated_fn=lambda s: False, opponent_pending_kinds=pending_b, my_seat_idx=0,
        )
        harness_b = TrainingHarness(
            reward_fn=rewards.strict_binary_reward, model_cls=MaskablePPO, decklist=deck_b,
            terminated_fn=lambda s: False, pending_kinds=pending_b, model_kwargs=tiny_model_kwargs,
            horizon=40, on_the_play=False, seed=1, opponent_decklist=deck_a,
            opponent_terminated_fn=lambda s: False, opponent_pending_kinds=pending_a, my_seat_idx=1,
        )
        assert harness_a.two_player and harness_b.two_player
        assert harness_a.env.opponent_model is None and harness_b.env.opponent_model is None
        # Potential-based shaping (MULTIPLAYER_GAPS.md): default off
        # (shaping_weight=0.0, no config asked for it here), shaping_gamma
        # falls back to SB3's own PPO default (0.99) since tiny_model_kwargs
        # sets no "gamma" -- and both reach all the way through to the
        # actual TwoPlayerDeckEnv the model trains against, not just the
        # harness's own attributes.
        assert harness_a.shaping_weight == 0.0 and harness_a.shaping_gamma == 0.99
        assert harness_a.env.shaping_weight == 0.0 and harness_a.env.shaping_gamma == 0.99

        path_a, path_b = os.path.join(tmp_dir, "a"), os.path.join(tmp_dir, "b")
        train_two_player(harness_a, harness_b, total_timesteps=64, burst_timesteps=32,
                          save_path_a=path_a, save_path_b=path_b)
        assert harness_a.env.opponent_model is harness_b.model  # live reference, not a snapshot
        assert harness_a.total_timesteps_trained == 64 and harness_b.total_timesteps_trained == 64
        print("harness.py train_two_player self-check: OK")

        # Loading with the WRONG mode (no opponent_decklist -- i.e. 1p)
        # must fail loudly, not silently produce a mismatched env -- this
        # is exactly the failure two_player's new mismatch-check exists to
        # catch (see _metadata's own docstring).
        try:
            TrainingHarness.load(
                path_a, reward_fn=rewards.strict_binary_reward, model_cls=MaskablePPO, decklist=deck_a,
                terminated_fn=lambda s: False, pending_kinds=pending_a, horizon=40,
            )
            raise AssertionError("load() should have rejected a 1-player reload of a two-player model")
        except ValueError as e:
            assert "two_player" in str(e)

        loaded_a = TrainingHarness.load(
            path_a, reward_fn=rewards.strict_binary_reward, model_cls=MaskablePPO, decklist=deck_a,
            terminated_fn=lambda s: False, pending_kinds=pending_a, horizon=40,
            opponent_decklist=deck_b, opponent_terminated_fn=lambda s: False, opponent_pending_kinds=pending_b,
            my_seat_idx=0,
        )
        loaded_b = TrainingHarness.load(
            path_b, reward_fn=rewards.strict_binary_reward, model_cls=MaskablePPO, decklist=deck_b,
            terminated_fn=lambda s: False, pending_kinds=pending_b, horizon=40,
            opponent_decklist=deck_a, opponent_terminated_fn=lambda s: False, opponent_pending_kinds=pending_a,
            my_seat_idx=1,
        )
        assert loaded_a.total_timesteps_trained == 64 and loaded_b.total_timesteps_trained == 64
        print("harness.py two-player save/load round-trip self-check: OK")

        wins_a, wins_b, draws, turn_counts, action_counts = evaluate_two_player(
            loaded_a, loaded_b, num_games=4, horizon=40, seed=7,
        )
        assert wins_a + wins_b + draws == 4
        assert all(t <= 40 for t in turn_counts)
        assert len(action_counts) == 4 and all(a > 0 for a in action_counts)  # every game takes at least 1 action
        print(f"harness.py evaluate_two_player self-check: OK (wins_a={wins_a}, wins_b={wins_b}, draws={draws})")

        # Same 4 games, this time with log_path -- the actual gap this pass
        # closed (evaluate_two_player had no per-game JSON log at all
        # before). Re-running (not reusing the pass above) keeps this
        # self-check independent of whether logging changes game outcomes
        # (it doesn't, but asserting that isn't this check's job).
        log_path = os.path.join(tmp_dir, "2p_log.json")
        evaluate_two_player(loaded_a, loaded_b, num_games=4, horizon=40, seed=7, log_path=log_path,
                             config_name="2p_selfcheck")
        with open(log_path) as f:
            log_doc = json.load(f)
        assert log_doc["meta"]["config_name"] == "2p_selfcheck"
        assert len(log_doc["meta"]["seats"]) == 2
        assert len(log_doc["games"]) == 4
        g = log_doc["games"][0]
        assert set(["game_index", "starting_player_idx", "winner", "turn_won", "final_turn_number", "scores",
                     "opening_state", "steps", "end_state"]) <= set(g.keys())
        assert len(g["scores"]) == 2 and len(g["opening_state"]["players"]) == 2 and len(g["end_state"]["players"]) == 2
        # Both seats' opening hands are real 7-card snapshots, not just one
        # side -- exactly what _snapshot_state (1-player, active-relative)
        # could never produce for a 2-player game.
        assert all(len(p["hand"]) <= 7 and "life_total" in p for p in g["opening_state"]["players"])
        assert len(g["steps"]) > 0
        step = g["steps"][0]
        assert set(["turn_number", "turn_player_idx", "actor_idx", "phase", "action"]) <= set(step.keys())
        assert step["actor_idx"] in (0, 1)
        assert len(step["state_after"]["players"]) == 2
        print(f"harness.py evaluate_two_player log_path self-check: OK ({log_path})")
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)
