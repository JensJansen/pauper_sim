"""Zone & state model: PlayerState (one player's zones + life) plus GameState
(shared turn/stack/pending-resolution bookkeeping + a list of PlayerStates).

GameState exposes every zone (hand/battlefield/library/graveyard/exile/
mana_pool/trigger_queue/attackers/... ) as a property proxying to
state.players[state.active_idx] -- whoever currently holds priority. This is
what lets every card-effect function + mana/resolution/effects stay unchanged:
they only ever meant "my own board," and the proxy makes that correct for the
active player in both 1- and 2-player games. state.opponent is the one
opponent-facing accessor (win_check.deal_damage_to_opponent).

mana_pool holds tapped-but-unspent mana; spending it is always an explicit
model action, never automatic, and it clears each turn (turn.untap_step)."""

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
        # player's currently-live permanents of that name -- lets drl_env address a SPECIFIC
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
        # Combat damage marked on this creature this turn -- compared against game.effects.stats.permanent_toughness by
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

        # Until-end-of-turn temporary modifiers (Agony Warp's -3/-0 & -0/-3;
        # Toxin Analysis' granted deathtouch/lifelink), all cleared by
        # game.effects.state_based.cleanup_step at end of turn. temp_power/
        # temp_toughness fold into stats.permanent_power/toughness; temp_
        # keywords into stats.creature_keywords. Separate from `counters`,
        # which persist (they're real +1/+1 / lifelink counters, not
        # until-EOT effects).
        self.temp_power = 0
        self.temp_toughness = 0
        self.temp_keywords = set()

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
    used to be the entirety of GameState before the multiplayer refactor.
    A 1-player game is just a GameState with a single PlayerState in it;
    nothing in this class itself knows or cares how many other
    PlayerStates, if any, exist alongside it."""

    def __init__(self, on_the_play, life_total=STARTING_LIFE):
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

        # Impulse zone: cards exiled "you may play until end of [this / your
        # next] turn" (Reckless Impulse, Experimental Synthesizer, Clockwork
        # Percussionist). list of (card_def, playable_until_turn_number) --
        # playable while state.turn_number <= that number; pruned once past it
        # by game.turn.untap_step. Kept distinct from `exile` (whose Plot
        # entries persist forever until cast) since impulse cards expire.
        self.impulse = []

        # list[dict], each {"type": "decision"|"automatic", "kind": str, ...}.
        # Populated by things that happen mid-resolution but must not be
        # acted on until the enclosing action's entire effect is fully
        # done. Promoted onto state.stack by game.effects.triggers.
        # promote_triggers_to_stack, called once per priority round
        # -- never mutated directly by any
        # card's own resolve function.
        self.trigger_queue = []

        self.lands_played_this_turn = 0
        # Reset each turn this player takes (turn._run_turn_gen), incremented
        # once per card actually drawn (see draw() below).
        self.cards_drawn_this_turn = 0

        # Undercity dungeon progress (The Initiative / Avenging Hunter): the
        # name of the room this player currently occupies, or None if they're
        # not in a dungeon. "Venture into Undercity" (game.effects.undercity)
        # enters at Secret Entrance when None, else advances to the next room;
        # cleared back to None once the final room (Throne of the Dead Three)
        # completes, so a later venture starts a fresh run. Persists across
        # turns and independently of who currently holds the initiative.
        self.dungeon_room = None

        self.decked_out = False

        # Real per-player life total -- only
        # ever decremented by game.effects.win_check.deal_damage_to_opponent acting
        # on the *other* player's PlayerState. Unused/inert in 1-player
        # mode (nothing ever reads a lone player's own life_total there).
        self.life_total = life_total

        # Creatures this player declared as attackers this combat
        # (game.turn.Phase.DECLARE_ATTACKERS through COMBAT_DAMAGE) --
        # empty outside combat.
        self.attackers = []

        # dict[Permanent (one of this player's own attackers) -> Permanent
        # (the OPPONENT's creature blocking it)]
        # blocking mechanics. An attacker absent from this dict is
        # unblocked. At most one blocker per attacker, at most one
        # attacker per blocker (no gang-blocking/menace modeled -- nothing
        # in the current card pool needs it). Reset alongside attackers,
        # each combat (declare_attackers_step).
        self.blocked_by = {}

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
        # yield there already counts unconditionally -- matching the per-game
        # step count a --log JSON records (which likewise never records a
        # Pass), just
        # persisted onto the player instead of a transient loop variable, so
        # a reward_fn (rl.rewards) can read it mid-game. 2-player only --
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
    multiplayer game). Every zone
    accessor below (state.hand, state.battlefield, ...) is a property
    proxying to state.players[state.active_idx] -- "whoever currently
    holds priority," which game.turn._declare_blockers_gen already
    temporarily flips away from the turn owner for its own narrow consult,
    and which general priority round flips far
    more broadly. state.turn_player_idx (below) is the OTHER, distinct
    fact -- whose turn it structurally is -- needed the instant those two
    can genuinely differ: real Magic's land-drop/sorcery-speed rules are
    "your own turn," not just "the right phase," so anything gating on
    those needs turn_player_idx specifically, never active_idx."""

    def __init__(self, on_the_play, rng=None, players=None, event_log=None):
        # `players`, if given, replaces the single-player list this
        # constructor would otherwise build from on_the_play -- used by
        # new_multiplayer_game_state below. on_the_play is then ignored
        # (each player supplies their own).
        self.players = players if players is not None else [PlayerState(on_the_play)]
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
        # progress. turn_won's meaning is unchanged from before the
        # multiplayer refactor (every existing reader -- game/effects/*.py,
        # drl_env.py, rl.rewards -- keeps working unmodified); winner is the
        # new field 2-player games need to say *who*.
        # winner stays None for a bare failure (horizon reached in
        # 1-player, or -- 1-player only -- decking out with no opponent to
        # award the win to).
        self.turn_won = None
        self.winner = None

        # Which player (index into state.players) currently has THE INITIATIVE
        # (Avenging Hunter / The Initiative), or None if no one does. A single
        # shared designation, like the monarch. game.effects.undercity.
        # take_initiative sets it (and queues that player's venture); combat
        # damage to the holder passes it (game.effects.combat); the holder
        # ventures into Undercity at the start of each of their upkeeps
        # (game.turn.upkeep_step).
        self.initiative_idx = None

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

        # The CardDef currently resolving off the stack, set by
        # game.effects.stack.resolve_top_of_stack for the duration of that
        # resolve. It left its controller's hand at cast (push_to_stack) and
        # must NEVER re-enter hand, so its resolve's own "send this card to
        # graveyard/battlefield" step (discard_from_hand_to_graveyard,
        # cast_permanent_from_hand, cast_aura) checks identity against this
        # instead of expecting the card in hand. None whenever nothing resolves.
        self.resolving_card = None

    hand = _active_player_property("hand")
    battlefield = _active_player_property("battlefield")
    library = _active_player_property("library")
    graveyard = _active_player_property("graveyard")
    exile = _active_player_property("exile")
    impulse = _active_player_property("impulse")
    trigger_queue = _active_player_property("trigger_queue")
    lands_played_this_turn = _active_player_property("lands_played_this_turn")
    cards_drawn_this_turn = _active_player_property("cards_drawn_this_turn")
    mana_pool = _active_player_property("mana_pool")
    decked_out = _active_player_property("decked_out")
    life_total = _active_player_property("life_total")
    attackers = _active_player_property("attackers")
    blocked_by = _active_player_property("blocked_by")
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
        # Single generic draw-logging hook: every card that enters a hand
        # from its library -- the turn draw_step, a spell like Faithless
        # Looting, the opening 7, a mulligan redraw -- passes through here
        # and is recorded as one "zone_move" (library->hand, reason="draw")
        # naming the cards. draw() is otherwise silent, so this is the ONLY
        # record of what was drawn, in the pregame too: there is no separate
        # "mulligan_hand" event, the opening and redrawn hands ARE these draw
        # events (see game.resolution.begin_mulligan). try/finally so a draw
        # that decks out partway still logs whatever reached hand before
        # DeckedOut unwound the stack.
        player = self.players[self.active_idx]
        if self.event_log is None:
            return player.draw(n)
        before = len(player.hand)
        try:
            player.draw(n)
        finally:
            drawn = player.hand[before:]
            if drawn:
                self.log_event("zone_move", cards=[c.name for c in drawn],
                               from_zone="library", to_zone="hand", reason="draw")

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
        (now-removed) earlier snapshot-based game logger snapshotted state after
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


def new_multiplayer_game_state(decklists, starting_player_idx, rng, event_log=None):
    """N-player entry point -- decklists are one entry per player and may
    differ (nothing here requires a mirror match; CARD_DEFS is already
    deck-agnostic). Only starting_player_idx is "on the play" (skips their
    own first draw -- see turn.draw_step) and takes the first turn; every
    other player starts with on_the_play=False. Every player draws their
    own opening 7, same as a single-player opening hand, regardless of who
    goes first."""
    players = [
        PlayerState(on_the_play=(i == starting_player_idx))
        for i in range(len(decklists))
    ]
    state = GameState(
        on_the_play=players[starting_player_idx].on_the_play, rng=rng, players=players, event_log=event_log,
    )
    state.turn_player_idx = starting_player_idx
    for idx, decklist in enumerate(decklists):
        state.players[idx].library = build_shuffled_library(decklist, rng)
        state.active_idx = idx   # attribute each opening draw to its own drawer
        state.draw(7)            # routed through GameState.draw so the opening hand is logged like any other draw
    state.active_idx = starting_player_idx
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
    from .turn import run_multiplayer_game

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

    # End to end through a REAL 2-player game (run_multiplayer_game -> every
    # instrumented mutation site across mana.py/turn.py/resolution.py/
    # game/effects/*.py) -- player 0 taps a Mountain, casts Lightning Bolt at
    # player 1, and lets it resolve; player 1 only ever passes. This is
    # exactly the scenario the removed earlier snapshot-based game logger got wrong:
    # the stack push/resolve and the once-per-turn mana clear both happen as a
    # side effect of "Pass," which that old logger silently dropped. Confirms
    # this one doesn't.
    bolt_def = registry.CARD_DEFS["Lightning Bolt"]

    # Lightning Bolt's production "cast" resolve is a precast "any target"
    # choice (game.catalog.red_cards) that would open a choose_any_target
    # pending this toy logging policy doesn't model -- same reason turn.py's
    # own real-game self-check uses a local direct-to-opponent resolve. This
    # test is about the event_log plumbing, not targeting, so decouple it the
    # same way.
    from .effects.win_check import deal_damage_to_opponent

    def bolt_resolve(s, cd):
        if cd in s.hand:
            s.hand.remove(cd)
        s.graveyard.append(cd)
        deal_damage_to_opponent(s, 3)

    def _burn_policy(state):
        if state.pending_resolution is not None and state.pending_resolution["kind"] == "mulligan_decision":
            return lambda: resolution.execute_mulligan_keep(state)  # both players always keep
        if state.pending_resolution is not None and state.pending_resolution["kind"] == "discard":
            name = resolution.discard_options(state)[0]  # cleanup hand-size discard, either player
            return lambda: resolution.execute_discard_option(state, name)
        if state.active_idx != 0:
            return None  # player 1 never acts otherwise
        if state.pending_resolution is not None:  # paying the Bolt's {R}
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
    state = run_multiplayer_game(
        decklists=[[("Mountain", 10), ("Lightning Bolt", 5)], [("Mountain", 20)]],
        rng=random.Random(0), starting_player_idx=0,
        choose_action=_burn_policy, horizon=12, event_log=events,
    )
    assert state.players[1].life_total < 20  # at least one Bolt landed on the opponent
    kinds = [e["kind"] for e in events]
    assert "turn_start" in kinds and "phase_change" in kinds
    assert "zone_move" in kinds  # land entering the battlefield, and the Bolt hitting the stack then leaving it
    assert "mana_tap" in kinds and "mana_spend" in kinds
    assert "pass" in kinds  # the exact event class the old snapshot-based logger silently dropped
    assert "life_change" in kinds  # the Bolt's own damage to the opponent's life_total
    # Every draw -- both opening hands AND every in-game draw -- is recorded as
    # one library->hand zone_move (reason="draw") naming the cards, via the
    # single generic hook; each opening 7 IS a draw event at turn 0.
    draw_moves = [e for e in events if e.get("reason") == "draw"]
    assert draw_moves, "expected draws to be logged"
    assert all(e["from_zone"] == "library" and e["to_zone"] == "hand" and e["cards"] for e in draw_moves)
    assert any(e["turn"] == 0 and len(e["cards"]) == 7 for e in draw_moves)  # an opening hand
    assert any(e["turn"] > 0 for e in draw_moves)  # at least one in-game draw (a turn's draw_step)
    # Every event's envelope is well-formed regardless of kind.
    for e in events:
        assert set(e) >= {"kind", "turn", "phase", "active_idx", "turn_player_idx"}
    # Order is causally sensible: a Mountain tap (mana_tap) happens before the
    # spend (mana_spend) that actually pays for the Bolt.
    assert kinds.index("mana_tap") < kinds.index("mana_spend")
    print(f"state.py real-game event_log self-check: OK ({len(events)} events, kinds={sorted(set(kinds))})")
