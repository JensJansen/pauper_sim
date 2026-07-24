"""Zone & state model: PlayerState (one player's battlefield/hand/library/
graveyard/mana pool/life total) plus GameState (the shared turn/stack/
pending-resolution bookkeeping and a list of PlayerStates).

See docs/MULTIPLAYER_ENGINE_PLAN.md for the full design. GameState exposes
every zone (hand/battlefield/library/graveyard/exile/mana_pool/
trigger_queue/lands_played_this_turn/cards_drawn_this_turn/decked_out/
damage_dealt/attackers/terminated_fn/on_the_play/life_total) as a property
that reads/writes state.players[state.active_idx] -- the player whose turn
it currently is. This is what lets every existing card-effect function and
all of mana.py/resolution.py/game/effects/*.py keep working completely
unchanged: they only ever meant "my own board" to begin with, and the
proxy makes that automatically correct for whichever player has priority,
in both 1-player (today's only mode -- active_idx never leaves 0) and
2-player games. state.opponent is the one new accessor genuinely-
opponent-facing code (game.effects.win_check.deal_damage_to_opponent) needs.

mana_pool holds mana tapped-but-not-yet-spent (e.g. a tap that produces more
than a cost's live needs) -- spending it, even toward a cost it could cover,
is always its own explicit model action, never automatic. Cleared at the
start of each turn (see turn.untap_step); see game.mana for how it's filled
and spent.
"""

import random

from . import registry

STARTING_LIFE = 20


class DeckedOut(Exception):
    """Raised by PlayerState.draw() the instant it's asked for a card with
    an empty library -- not just a flag to poll later. Real Magic:
    attempting to draw from an empty library is an immediate loss, mid-
    effect, not something that waits for the rest of the current card's
    effect (or trigger-queue drain, or phase) to finish first. game.turn.
    _run_turn_gen wraps its entire body in one try/except DeckedOut: return,
    so this unwinds cleanly from however deep a card's own effect/
    resolution chain is at the moment the draw happens, with the drawing
    player's decked_out already set by draw() below before the raise. In a
    2-player game the drawing player instantly loses (state.winner is set
    to the other player); in 1-player there's no one to award the win to,
    same bare-failure outcome as before this exception existed."""


class Permanent:
    """A specific physical card sitting on the battlefield."""

    def __init__(self, card_def, tapped=False):
        self.card_def = card_def
        self.tapped = tapped
        # True until the untap step that first sees this permanent already
        # on the battlefield -- matches real Magic's "under your control
        # since your most recent turn began" (a permanent entering mid-turn,
        # by any path, is sick for the rest of that turn regardless of how
        # it got there). Only declare_attackers_step (game.effects.combat)
        # reads this today.
        self.summoning_sick = True
        # Generic per-permanent runtime flags, e.g. "used_mana_filter_this_turn"
        # for Barrels of Blasting Jelly's once-per-turn ability.
        self.flags = {}
        # Which numbered copy of this exact card name this is, among this
        # player's currently-live permanents of that name (docs/COMBAT_PLAN.md's
        # "permanent identity" design) -- lets drl_env address a SPECIFIC
        # physical creature ("Attack: Slippery Bogle (slot 2)") instead of
        # picking an arbitrary same-named match, since two copies stop being
        # interchangeable the moment an Aura attaches to only one of them.
        # Assigned for real by game.effects.casting.enters_battlefield (pooled:
        # the lowest number not already in use among currently-live
        # same-named permanents -- reused once a permanent leaves, never
        # incremented forever, so it stays bounded even through repeated
        # bounce/blink). Defaults to 1 here so every existing self-check
        # that constructs a Permanent directly (bypassing enters_battlefield)
        # -- always exactly one instance of a given name -- gets a sensible
        # value for free; a real game always overwrites this on entry.
        self.slot = 1
        # Combat damage marked on this creature this turn (docs/COMBAT_PLAN.md
        # step 6) -- compared against game.effects.stats.permanent_toughness by
        # check_state_based_actions to decide creature death. Cleared for
        # every permanent, both players, each turn's cleanup_step (real
        # Magic: damage clears at cleanup regardless of whose turn it is).
        # 0 for every non-creature permanent too -- harmless, never read.
        self.damage_marked = 0

        # dict[str counter-kind -> int count], e.g. {"+1/+1": 2}. Lives
        # directly on the permanent (unlike an enchanting Aura, which has to
        # be found by scanning the battlefield) -- game.effects.stats folds
        # this straight into permanent_power/permanent_toughness with no new
        # battlefield scan of its own. Kind strings are card-text-shaped
        # ("+1/+1", "-0/-1", Pinnacle Kill-Ship's own "charge") -- no
        # +1/+1-vs--1/-1 annihilation (real Magic rule 122.3) is modeled,
        # since no card in this pool ever puts both kinds on the same
        # permanent; add it if one ever does.
        self.counters = {}

        # None (the common case) means "use card_def.card_type" -- see the
        # card_type property below. Set for real by Pinnacle Kill-Ship's own
        # Station ability (CardType.CREATURE, once animated -- colorless_
        # cards.py) and Nyxborn Hydra's own Bestow (CardType.ENCHANTMENT
        # while attached, cleared back to None -- i.e. back to card_def's
        # own CREATURE -- once orphaned, game.effects.state_based's own
        # "becomes_creature_when_orphaned" branch -- green_cards.py).
        # card_def.card_type itself is fixed/shared across every physical
        # copy of a name, so a per-permanent type change can never be
        # expressed by mutating it directly.
        self.type_override = None

    @property
    def card_type(self):
        """This permanent's REAL current type -- type_override if one is
        set, else its own card_def.card_type. Every "what type is this
        permanent right now" check (attack/block eligibility, state-based
        death candidacy, "target a creature/enchantment you control"
        predicates, "how many enchantments do you control") reads this, NOT
        card_def.card_type directly, so Pinnacle Kill-Ship's Station and
        Nyxborn Hydra's Bestow both take effect everywhere automatically
        once type_override is set/cleared, with no per-site special-casing.
        A bare CardDef sitting in hand/library/graveyard (not yet a
        Permanent) has no override concept at all -- those zones' own
        search/discard/reanimation-target predicates correctly keep reading
        card_def.card_type directly, unaffected by this property."""
        return self.type_override if self.type_override is not None else self.card_def.card_type

    def __repr__(self):
        return f"Permanent({self.card_def.name!r}, tapped={self.tapped})"


class PlayerState:
    """One player's zones, turn-scoped counters, and life total -- what
    used to be the entirety of GameState before MULTIPLAYER_ENGINE_PLAN.md.
    A 1-player game is just a GameState with a single PlayerState in it;
    nothing in this class itself knows or cares how many other
    PlayerStates, if any, exist alongside it."""

    def __init__(self, on_the_play, terminated_fn=None, life_total=STARTING_LIFE):
        self.library = []       # ordered list[CardDef], index 0 = top of deck
        self.hand = []          # list[CardDef]
        self.battlefield = []   # list[Permanent]
        # list[CardDef]. Dread Return (reanimation, Flashback) reads and
        # removes from it, not just bookkeeping.
        self.graveyard = []

        # list[tuple[CardDef, int | None]] -- (card_def, plotted_turn).
        # plotted_turn is None for Madness entries (never outlive one
        # trigger-queue drain) and turn_number-at-the-time for Plot entries
        # (persist across turns until cast).
        self.exile = []

        # list[dict], each {"type": "decision"|"automatic", "kind": str, ...}.
        # Populated by things that happen mid-resolution but must not be
        # acted on until the enclosing action's entire effect is fully
        # done. Promoted onto state.stack by game.effects.triggers.
        # promote_triggers_to_stack, called once per priority round
        # (docs/PRIORITY_PLAN.md item 1) -- never mutated directly by any
        # card's own resolve function.
        self.trigger_queue = []

        self.lands_played_this_turn = 0
        # Reset each turn this player takes (turn._run_turn_gen), incremented
        # once per card actually drawn (see draw() below).
        self.cards_drawn_this_turn = 0

        # spy_combo deck: damage this player has dealt via non-combat
        # damage effects and combat (see game.effects.win_check.
        # deal_damage_to_opponent) -- kept exactly as before
        # MULTIPLAYER_ENGINE_PLAN.md for 1-player back-compat
        # (terminated.damage_threshold_terminated reads this, unchanged).
        # In a 2-player game the same call also decrements the opponent's
        # real life_total below; in 1-player there is no opponent to
        # decrement, this counter is the whole story, same as always.
        self.damage_dealt = 0
        self.decked_out = False

        # Real per-player life total (MULTIPLAYER_ENGINE_PLAN.md) -- only
        # ever decremented by game.effects.win_check.deal_damage_to_opponent acting
        # on the *other* player's PlayerState. Unused/inert in 1-player
        # mode (nothing ever reads a lone player's own life_total there).
        self.life_total = life_total

        # Creatures this player declared as attackers this combat
        # (game.turn.Phase.DECLARE_ATTACKERS through COMBAT_DAMAGE) --
        # empty outside combat.
        self.attackers = []

        # dict[Permanent (one of this player's own attackers) -> Permanent
        # (the OPPONENT's creature blocking it)] -- docs/COMBAT_PLAN.md's
        # blocking mechanics. An attacker absent from this dict is
        # unblocked. At most one blocker per attacker, at most one
        # attacker per blocker (no gang-blocking/menace modeled -- nothing
        # in the current card pool needs it). Reset alongside attackers,
        # each combat (declare_attackers_step).
        self.blocked_by = {}

        # terminated_fn(state) -> bool is this player's own injected win
        # condition (their deck's own combo-completion check -- Tron
        # assembly, a damage threshold), checked generically via
        # game.effects.win_check._check_end_of_game rather than hardcoded to any
        # one deck. Defaults to None so hand-built test states (which set
        # state.turn_won directly, bypassing this mechanism) don't need to
        # supply one; real games always pass one explicitly.
        self.terminated_fn = terminated_fn

        # Whether this player skips their very first draw (real Magic: the
        # player on the play doesn't draw turn 1). Checked against
        # turns_taken (this player's own turn count), not the game's
        # global turn_number -- once a second player also takes turns, the
        # global counter no longer means "my first turn" (see turn.
        # draw_step).
        self.on_the_play = on_the_play
        self.turns_taken = 0

        # How many mulligans this player has taken in the pregame mulligan
        # phase (game.turn.run_mulligan_phase) -- 0 for a kept opening hand.
        # Determines how many cards a "keep" must bottom (London Mulligan --
        # see game.resolution.execute_mulligan_keep). Never reset once the
        # game is underway; nothing after the pregame phase reads it again.
        self.mulligans_taken = 0

        # dict[str symbol -> int count], e.g. {"G": 2}. Absent/zero entries
        # mean "none floating" -- never holds a "generic" key, only real
        # color/colorless symbols (generic is a cost-side concept, never
        # something a source produces).
        self.mana_pool = {}

        # Total REAL actions this player has personally taken over the
        # whole game -- incremented once per non-Pass action executed in
        # game.turn's own priority round, declare_blockers, mulligan, and
        # end-of-turn discard loops. Pass itself never counts (declining to
        # act isn't the "pointless action" a reward shaped by this is meant
        # to discourage -- it's usually the correct choice); mulligan/
        # declare_blockers/discard never offer a bare Pass at all, so every
        # yield there already counts unconditionally. Matches harness.
        # evaluate_two_player's own per-game step count (games' own "steps"
        # in a --log JSON, which likewise never records a Pass), just
        # persisted onto the player instead of a transient loop variable, so
        # a reward_fn (rewards.py) can read it mid-game. 2-player only --
        # unused/inert in 1-player mode (nothing there currently reads it).
        self.actions_taken = 0

    def draw(self, n=1):
        """Real Magic: attempting to draw from an empty library is an
        instant loss, mid-effect -- sets decked_out, then raises DeckedOut
        so the enclosing turn generator (game.turn._run_turn_gen) unwinds
        immediately, wherever in a card's own effect/resolution chain this
        draw happened.

        Increments cards_drawn_this_turn per card actually drawn and
        queues an "automatic" trigger-queue entry (drained once the
        enclosing action is fully done, never inline here) for every
        graveyard card whose registry entry has an "on_draw_count" spec
        matching the new count exactly. Generic and registry-driven --
        scans the full (un-deduped) graveyard, so multiple physical copies
        each queue their own return, matching a per-card triggered
        ability."""
        for _ in range(n):
            if not self.library:
                self.decked_out = True
                raise DeckedOut()
            self.hand.append(self.library.pop(0))
            self.cards_drawn_this_turn += 1
            for card_def in self.graveyard:
                spec = registry.EFFECT_REGISTRY.get(card_def.effect_id, {}).get("on_draw_count")
                if spec is not None and self.cards_drawn_this_turn == spec["count"]:
                    self.trigger_queue.append({"type": "automatic", "kind": "on_draw_count", "card_def": card_def})


def _active_player_property(attr):
    """One GameState property per PlayerState field, reading/writing
    state.players[state.active_idx].<attr> -- see this module's own
    docstring for why this is the load-bearing trick that lets every
    existing card-effect function and mana.py/resolution.py/
    game/effects/*.py stay unchanged under a 2-player game."""
    def getter(self):
        return getattr(self.players[self.active_idx], attr)

    def setter(self, value):
        setattr(self.players[self.active_idx], attr, value)

    return property(getter, setter)


class GameState:
    """Shared turn/stack/pending-resolution bookkeeping, plus a list of
    PlayerStates (length 1 today for every existing config; 2 for a
    multiplayer game -- see docs/MULTIPLAYER_ENGINE_PLAN.md). Every zone
    accessor below (state.hand, state.battlefield, ...) is a property
    proxying to state.players[state.active_idx] -- "whoever currently
    holds priority," which game.turn._declare_blockers_gen already
    temporarily flips away from the turn owner for its own narrow consult,
    and which docs/PRIORITY_PLAN.md's general priority round flips far
    more broadly. state.turn_player_idx (below) is the OTHER, distinct
    fact -- whose turn it structurally is -- needed the instant those two
    can genuinely differ: real Magic's land-drop/sorcery-speed rules are
    "your own turn," not just "the right phase," so anything gating on
    those needs turn_player_idx specifically, never active_idx."""

    def __init__(self, on_the_play, rng=None, terminated_fn=None, players=None, event_log=None):
        # `players`, if given, replaces the single-player list this
        # constructor would otherwise build from on_the_play/terminated_fn
        # -- used by new_multiplayer_game_state below. on_the_play/
        # terminated_fn are then ignored (each player supplies their own).
        self.players = players if players is not None else [PlayerState(on_the_play, terminated_fn=terminated_fn)]
        self.active_idx = 0

        # None (the default -- every existing caller) means "logging is
        # off," checked first thing in log_event below so every
        # instrumented call site across mana.py/turn.py/resolution.py/
        # game/effects/*.py costs one attribute read on the hot training
        # path, never a dict build. Pass a plain list to turn logging on
        # for one game (e.g. `GameState(..., event_log=[])`, or via
        # new_game_state/new_multiplayer_game_state's own event_log param)
        # -- log_event appends one structured dict per instrumented state
        # change, in occurrence order, which is what makes the whole game
        # reconstructible afterward (see log_event's own docstring for the
        # shared envelope every event gets).
        self.event_log = event_log

        # Whose turn it structurally is -- distinct from active_idx (see
        # this class's own docstring) the instant a priority consult flips
        # active_idx away from the turn owner. Set once per turn by
        # game.turn._run_turn_gen's own turn-start setup (same place
        # turn_number/turns_taken/lands_played_this_turn reset), NEVER
        # touched by a priority-consult flip -- defaults to active_idx's
        # own starting value here so a hand-built self-check state (which
        # never goes through _run_turn_gen at all) still gets a sensible
        # value for free.
        self.turn_player_idx = self.active_idx

        self.turn_number = 0
        self.rng = rng or random.Random()

        # Set by game.turn._run_turn_gen at the start of each phase (see
        # game.turn.Phase) -- None until the first phase is entered.
        self.phase = None

        # The turn the game ended on, and which player (index into
        # state.players) won it -- None/None while the game is still in
        # progress. turn_won's meaning is unchanged from before
        # MULTIPLAYER_ENGINE_PLAN.md (every existing 1-player reader --
        # game/effects/*.py, drl_env.py, harness.py, rewards.py,
        # generate_regression_snapshot.py -- keeps working unmodified);
        # winner is the new field 2-player games need to say *who*.
        # winner stays None for a bare failure (horizon reached in
        # 1-player, or -- 1-player only -- decking out with no opponent to
        # award the win to).
        self.turn_won = None
        self.winner = None

        # None, or a dict describing an in-progress multi-step decision
        # (paying a cost one tap at a time, resolving a scry/surveil,
        # choosing a search target, ...) that must be fully resolved
        # before any other action becomes legal again. Always the active
        # player's own decision -- this engine's full-sequential-turns
        # model (no interrupt window) means a defending player never gets
        # a decision of their own during someone else's turn. See
        # game.resolution.begin_resolution/complete_resolution.
        self.pending_resolution = None

        # list[dict {"card_def": CardDef, "resolve": (state, card_def) ->
        # None}], top of stack = last element. Real Magic's stack is one
        # object shared by both players -- kept here (not per-player) for
        # that reason, even though only the active player ever pushes to
        # it under this engine's no-interrupt-window rule. See
        # game.effects.stack.push_to_stack/resolve_top_of_stack.
        self.stack = []

    hand = _active_player_property("hand")
    battlefield = _active_player_property("battlefield")
    library = _active_player_property("library")
    graveyard = _active_player_property("graveyard")
    exile = _active_player_property("exile")
    trigger_queue = _active_player_property("trigger_queue")
    lands_played_this_turn = _active_player_property("lands_played_this_turn")
    cards_drawn_this_turn = _active_player_property("cards_drawn_this_turn")
    mana_pool = _active_player_property("mana_pool")
    decked_out = _active_player_property("decked_out")
    damage_dealt = _active_player_property("damage_dealt")
    life_total = _active_player_property("life_total")
    attackers = _active_player_property("attackers")
    blocked_by = _active_player_property("blocked_by")
    terminated_fn = _active_player_property("terminated_fn")
    on_the_play = _active_player_property("on_the_play")
    turns_taken = _active_player_property("turns_taken")
    mulligans_taken = _active_player_property("mulligans_taken")

    @property
    def opponent(self):
        """The non-active PlayerState -- only meaningful (and only ever
        called) in a 2-player game; game.effects.win_check.deal_damage_to_opponent
        guards every call with len(state.players) > 1 first."""
        return self.players[1 - self.active_idx]

    def draw(self, n=1):
        return self.players[self.active_idx].draw(n)

    def log_event(self, kind, **fields):
        """Appends one structured event to self.event_log, if logging is on
        (see __init__'s own event_log docstring) -- else an immediate
        no-op, so every instrumented call site across mana.py/turn.py/
        resolution.py/game/effects/*.py costs exactly one attribute check
        when logging is off (the default, and every existing bulk-training
        path).

        Every event automatically carries the same envelope -- turn,
        phase, active_idx (who currently holds priority), turn_player_idx
        (whose turn it structurally is -- see this class's own docstring
        on why that's a distinct fact from active_idx) -- so an
        instrumented call site only ever supplies `kind` plus whatever
        fields are specific to that one event (a permanent's (name, slot),
        a color/amount, a from/to zone, ...). This is deliberately an
        EVENT log, not another periodic state snapshot: the previous
        (now-removed) harness.py-based game logger snapshotted state after
        each MODEL decision only, which made every state change that
        happened as an automatic side effect of "Pass" (a stack
        resolution, a state-based-action death, the once-per-turn mana
        clear) invisible between two logged snapshots. Recording the
        change itself, at the exact call site that makes it, has no such
        blind spot regardless of whether a real decision or an automatic
        engine step caused it."""
        if self.event_log is None:
            return
        self.event_log.append({
            "kind": kind,
            "turn": self.turn_number,
            "phase": self.phase.value if self.phase is not None else None,
            "active_idx": self.active_idx,
            "turn_player_idx": self.turn_player_idx,
            **fields,
        })


def build_shuffled_library(decklist, rng):
    """Expand a decklist's quantities into CardDef refs and shuffle. Only
    which decklist's quantities to expand is parameterized -- CARD_DEFS
    stays the single shared name->CardDef lookup (game.registry)."""
    library = []
    for name, qty, *_rest in decklist:
        library.extend([registry.CARD_DEFS[name]] * qty)
    rng.shuffle(library)
    return library


def new_game_state(decklist, terminated_fn, on_the_play, rng, event_log=None):
    """1-player entry point -- unchanged signature/behavior from before
    MULTIPLAYER_ENGINE_PLAN.md (drl_env.py, out of scope for that plan,
    calls this directly and must keep working unmodified) aside from the
    new, defaulted-off event_log param (GameState's own docstring)."""
    state = GameState(on_the_play, rng=rng, terminated_fn=terminated_fn, event_log=event_log)
    state.library = build_shuffled_library(decklist, rng)
    state.draw(7)
    return state


def new_multiplayer_game_state(decklists, terminated_fns, starting_player_idx, rng, event_log=None):
    """N-player entry point (docs/MULTIPLAYER_ENGINE_PLAN.md) -- decklists/
    terminated_fns are one entry per player and may differ (nothing here
    requires a mirror match; CARD_DEFS is already deck-agnostic). Only
    starting_player_idx is "on the play" (skips their own first draw --
    see turn.draw_step) and takes the first turn; every other player
    starts with on_the_play=False. Every player draws their own opening 7,
    same as new_game_state's single-player opening hand, regardless of who
    goes first."""
    players = [
        PlayerState(on_the_play=(i == starting_player_idx), terminated_fn=terminated_fns[i])
        for i in range(len(decklists))
    ]
    state = GameState(
        on_the_play=players[starting_player_idx].on_the_play, rng=rng, players=players, event_log=event_log,
    )
    state.active_idx = starting_player_idx
    state.turn_player_idx = starting_player_idx
    for player, decklist in zip(state.players, decklists):
        player.library = build_shuffled_library(decklist, rng)
        player.draw(7)
    return state


if __name__ == "__main__":
    # ponytail self-check: run via `python -m game.state` from src/. Proves
    # the event_log plumbing itself (this module's own new surface) end to
    # end -- NOT a re-test of engine correctness (every game/*.py file
    # already self-checks its own instrumented behavior unchanged; this
    # only confirms logging actually happens, in the right shape, in the
    # right order, and costs nothing when off).
    import random

    from . import mana, registry, resolution
    from .effects.casting import play_land_from_hand
    from .effects.stack import push_to_stack
    from .turn import run_game

    # event_log=None (the default) is a true no-op -- log_event never
    # builds a dict, self.event_log stays None.
    off_state = GameState(on_the_play=True)
    assert off_state.event_log is None
    off_state.log_event("whatever", foo="bar")
    assert off_state.event_log is None
    print("state.py event_log=None no-op self-check: OK")

    # Direct log_event call: every event carries the same envelope
    # (turn/phase/active_idx/turn_player_idx) automatically, on top of
    # whatever fields the call site passed.
    log = []
    on_state = GameState(on_the_play=True, event_log=log)
    on_state.turn_number = 3
    on_state.log_event("tap", permanent=("Mountain", 1))
    assert len(log) == 1
    event = log[0]
    assert event["kind"] == "tap" and event["permanent"] == ("Mountain", 1)
    assert event["turn"] == 3 and event["active_idx"] == 0 and event["turn_player_idx"] == 0
    print("state.py log_event envelope self-check: OK")

    # End to end through a REAL game (run_game -> new_game_state -> every
    # instrumented mutation site across mana.py/turn.py/resolution.py/
    # game/effects/*.py) -- tap a Mountain, cast Lightning Bolt, let it
    # resolve. This is exactly the scenario the removed harness.py-based
    # game logger got wrong (see docs discussion): the stack push/resolve
    # and the once-per-turn mana clear both happen as a side effect of
    # "Pass," which that old logger silently dropped. Confirms this one
    # doesn't.
    bolt_def = registry.CARD_DEFS["Lightning Bolt"]
    bolt_resolve = registry.EFFECT_REGISTRY[bolt_def.effect_id]["cast"]["resolve"]

    def _burn_policy(state):
        if state.pending_resolution is not None:
            if state.pending_resolution["kind"] == "mulligan_decision":
                return lambda: resolution.execute_mulligan_keep(state)
            tap_opts = mana.tap_cost_options(state)
            if tap_opts:
                name, color, is_filter = tap_opts[0]
                return lambda: mana.execute_tap_cost_option(state, name, color, is_filter)
            color = mana.pool_spend_options(state)[0]
            return lambda: mana.execute_pool_spend(state, color)
        if state.lands_played_this_turn == 0 and any(c.name == "Mountain" for c in state.hand):
            return lambda: play_land_from_hand(state, registry.CARD_DEFS["Mountain"])
        # hand_count - stacked_count, same accounting drl_env._hand_count_available
        # and turn.py's own self-check use: a copy already paid-for-but-
        # unresolved on the stack is still physically in hand (push_to_stack
        # only removes it once it actually resolves) but isn't really available.
        hand_count = sum(1 for c in state.hand if c.name == "Lightning Bolt")
        stacked_count = sum(1 for e in state.stack if e["card_def"].name == "Lightning Bolt")
        if hand_count > stacked_count and mana.plan_payment(state, bolt_def.cast_cost) is not None:
            def _cast_bolt():
                mana.begin_pay_cost(state, bolt_def.cast_cost, on_complete=lambda s: push_to_stack(s, bolt_def, bolt_resolve))
            return _cast_bolt
        return None  # Pass -- resolves the stack if non-empty, else advances the phase

    events = []
    state = run_game(
        decklist=[("Mountain", 10), ("Lightning Bolt", 5)], terminated_fn=None, rng=random.Random(0),
        on_the_play=True, horizon=2, choose_action=_burn_policy, event_log=events,
    )
    assert state.damage_dealt >= 3  # at least one Bolt landed
    kinds = [e["kind"] for e in events]
    assert "turn_start" in kinds and "phase_change" in kinds
    assert "zone_move" in kinds  # land entering the battlefield, and the Bolt hitting the stack then leaving it
    assert "mana_tap" in kinds and "mana_spend" in kinds
    assert "pass" in kinds  # the exact event class the old snapshot-based logger silently dropped
    # Every event's envelope is well-formed regardless of kind.
    for e in events:
        assert set(e) >= {"kind", "turn", "phase", "active_idx", "turn_player_idx"}
    # Order is causally sensible: the Mountain tap (mana_tap) happens before
    # the spend (mana_spend) that actually pays for the Bolt.
    assert kinds.index("mana_tap") < kinds.index("mana_spend")
    print(f"state.py real-game event_log self-check: OK ({len(events)} events, kinds={sorted(set(kinds))})")
