"""Zone & state model: PlayerState (one player's zones + life) plus GameState
(shared turn/stack/pending-resolution bookkeeping + a list of PlayerStates).

GameState exposes every zone (hand/battlefield/library/graveyard/exile/
mana_pool/trigger_queue/attackers/...) as a property proxying to
state.players[state.active_idx] -- whoever currently holds priority.
state.opponent is the one opponent-facing accessor.

mana_pool holds floated-but-unspent mana; spending it is an explicit model
action, and it empties at every step/phase boundary for both players
(turn._empty_mana_pools, rule 500.4)."""

import random

from . import registry

STARTING_LIFE = 20


class DeckedOut(Exception):
    """Raised by PlayerState.draw() the instant it's asked for a card with
    an empty library -- real Magic: an immediate loss, mid-effect, not
    deferred to the end of the current effect or phase.
    game.turn._run_turn_gen wraps its whole body in try/except DeckedOut,
    unwinding cleanly from wherever the draw happened; decked_out is
    already set by draw() before the raise. In a 2-player game the drawing
    player loses outright; in 1-player there's no one to award the win to."""


class CardInstance:
    """A specific physical card, distinct from every other copy of the same
    name, in whatever zone holds it. Wraps the shared, interned CardDef
    (game.registry.CARD_DEFS) and adds a per-game `iid` so same-named
    copies are tellable apart.

    IDENTITY IS OBJECT IDENTITY. A target/selection captures the exact
    instance object and stays legal only while that object is still in its
    zone; every zone change mints a NEW instance (GameState.move_card), so
    a card that changes zones is genuinely a new object (MTG 400.7) and any
    target locked on the old object fizzles. `iid` is engine-internal
    (logging/debugging) -- never shown to the agent, never the targeting
    handle. Deliberately does NOT override __eq__/__hash__.

    Attribute access proxies to card_def (`.name`, `.card_type`, ...) via
    __getattr__, so existing code that reads those off a zone element keeps
    working unchanged.

    Scope: the graveyard holds CardInstances, the battlefield holds
    Permanents (a subclass). hand/library/exile are still list[CardDef]
    until a card needs distinct instances there."""

    def __init__(self, card_def, iid=None):
        self.card_def = card_def
        self.iid = iid

    def __getattr__(self, attr):
        # Reached only when `attr` isn't a real attribute of this object
        # (card_def/iid, or a Permanent subclass's own battlefield fields) --
        # forward the rest to the shared CardDef. Guard card_def to avoid
        # infinite recursion if it's ever missing (copy/unpickle).
        if attr == "card_def":
            raise AttributeError(attr)
        return getattr(self.card_def, attr)

    def __repr__(self):
        return f"CardInstance({self.card_def.name!r}#{self.iid})"


class Permanent(CardInstance):
    """A specific physical card sitting on the battlefield -- a CardInstance
    (shared card_def + per-game iid) plus battlefield-only runtime state
    (tapped, counters, damage, summoning sickness, slot, ...)."""

    def __init__(self, card_def, tapped=False, iid=None):
        super().__init__(card_def, iid)
        self.tapped = tapped
        # True until the untap step first sees this permanent already on
        # the battlefield (302.6: continuous control since the turn began).
        self.summoning_sick = True
        self.flags = {}  # generic per-permanent runtime flags, e.g. "used_mana_filter_this_turn"
        # Which numbered copy of this card name this is among the player's
        # currently-live permanents of that name -- lets drl_env address a
        # SPECIFIC physical creature. Assigned by
        # game.effects.casting.enters_battlefield (pooled: lowest unused
        # number, reused once a permanent leaves). Defaults to 1 here for
        # direct-construction self-checks; a real game always overwrites it.
        self.slot = 1
        # Combat damage marked this turn -- compared against
        # stats.permanent_toughness by check_state_based_actions. Cleared
        # for every permanent each turn's cleanup_step.
        self.damage_marked = 0

        # dict[str counter-kind -> int count], e.g. {"+1/+1": 2}. Kind
        # strings are card-text-shaped ("+1/+1", Pinnacle Kill-Ship's
        # "charge"). +1/+1-vs--1/-1 annihilation (122.3) is not modeled --
        # no card here ever puts both kinds on one permanent.
        self.counters = {}

        # None means "use card_def.card_type" (see the card_type property
        # below). Set by Pinnacle Kill-Ship's Station (animated ->
        # CREATURE) and Nyxborn Hydra's Bestow (attached -> ENCHANTMENT,
        # cleared back to None once orphaned).
        self.type_override = None

        # Until-end-of-turn modifiers (Agony Warp's -3/-0 & -0/-3, Toxin
        # Analysis' granted deathtouch/lifelink), cleared by cleanup_step.
        # Separate from `counters`, which persist.
        self.temp_power = 0
        self.temp_toughness = 0
        self.temp_keywords = set()

    @property
    def card_type(self):
        """This permanent's real current type -- type_override if set, else
        card_def.card_type. Every "what type is this permanent" check
        should read this, not card_def.card_type directly, so Station/
        Bestow take effect everywhere with no per-site special-casing.
        A bare CardDef (hand/library/graveyard) has no override concept."""
        return self.type_override if self.type_override is not None else self.card_def.card_type

    def __repr__(self):
        return f"Permanent({self.card_def.name!r}, tapped={self.tapped})"


class PlayerState:
    """One player's zones, turn-scoped counters, and life total. GameState
    holds a list of these -- two for a real game, one for a single-board
    unit test."""

    def __init__(self, on_the_play, life_total=STARTING_LIFE):
        self.library = []       # ordered list[CardDef], index 0 = top of deck
        self.hand = []          # list[CardDef]
        self.battlefield = []   # list[Permanent]
        self.graveyard = []     # list[CardDef]; Dread Return reads/removes from it directly

        # list[tuple[CardDef, int | None]] -- (card_def, plotted_turn).
        # None for Madness entries (transient); turn_number for Plot
        # entries (persist across turns until cast).
        self.exile = []

        # Impulse zone: cards exiled "you may play until end of [this/your
        # next] turn" (Reckless Impulse, etc). list of (card_def,
        # playable_until_turn_number); pruned by game.turn.untap_step once
        # past the deadline. Distinct from `exile` since impulse expires.
        self.impulse = []

        # list[dict], each {"type": "decision"|"automatic", "kind": str, ...}.
        # Populated mid-resolution but not acted on until that action's
        # effect is fully done; promoted onto state.stack by
        # game.effects.triggers.promote_triggers_to_stack each priority round.
        self.trigger_queue = []

        # Phase-scoped mistake-detection flags for the dense mana-burn
        # reward signal, read then reset each phase boundary by
        # game.turn._empty_mana_pools. cost_paid_this_phase: a real
        # cast/ability payment happened. triggers_fired_this_phase: a
        # trigger was promoted to the stack from this player's queue.
        # Either flag alone exempts a burn from counting as a mistake.
        self.cost_paid_this_phase = False
        self.triggers_fired_this_phase = False

        self.lands_played_this_turn = 0
        self.cards_drawn_this_turn = 0  # reset each turn (turn._run_turn_gen), incremented per card drawn

        # Undercity dungeon progress (The Initiative / Avenging Hunter): room
        # name currently occupied, or None. game.effects.undercity's
        # "venture" enters at Secret Entrance when None, else advances;
        # cleared once the final room (Throne of the Dead Three) completes.
        self.dungeon_room = None

        self.decked_out = False

        # Decremented only by game.effects.win_check.deal_damage_to_opponent
        # acting on the *other* player's PlayerState. Unused in 1-player mode.
        self.life_total = life_total

        self.attackers = []  # declared attackers (Phase.DECLARE_ATTACKERS through COMBAT_DAMAGE); empty outside combat

        # dict[Permanent (attacker) -> list[Permanent]] (the OPPONENT's
        # blockers). An attacker absent from this dict is unblocked;
        # gang-blocking (multiple blockers per attacker) is supported.
        # Reset alongside attackers, each combat (declare_attackers_step).
        self.blocked_by = {}

        # Whether this player skips their first draw (the player on the
        # play doesn't draw turn 1). Checked against turns_taken (this
        # player's own turn count), not the game's global turn_number --
        # see turn.draw_step.
        self.on_the_play = on_the_play
        self.turns_taken = 0

        # Mulligans taken in the pregame phase -- 0 for a kept opening hand.
        # Determines how many cards a "keep" must bottom (London Mulligan,
        # game.resolution.execute_mulligan_keep).
        self.mulligans_taken = 0

        # dict[str symbol -> int count], e.g. {"G": 2}. Never holds a
        # "generic" key -- generic is a cost-side concept, not something a
        # source produces.
        self.mana_pool = {}

        # Parallel shadow-count to mana_pool: dict[str symbol -> int count]
        # of currently-floating pips that trace back to a "single-pip"
        # mana-producing EVENT (exactly 1 symbol added in that event,
        # computed dynamically by game.mana.float_mana -- not a static
        # per-source-kind rule). A plain land is always single-pip; Rakdos
        # Carnarium/Utopia Sprawl's bonus (2+ symbols) never is. A mana
        # filter's output pip and an Eldrazi Spawn sac-for-{C} are forced
        # untagged (taggable=False) since both are pool->pool conversions,
        # not reflexive tapping.
        #
        # A source that's SACRIFICED or UNTAPPED after already being
        # tapped this phase gets an after-the-fact discount via
        # game.mana.discount_departing_source instead (see its docstring).
        #
        # Invariant: mana_pool_single_pip[c] <= mana_pool[c] for every
        # color c. Decremented by game.mana.spend_one_pip's spend-order
        # convention (untagged before tagged). Cleared alongside mana_pool
        # at every phase boundary (game.turn._empty_mana_pools).
        self.mana_pool_single_pip = {}

        # Total REAL actions this player has taken this game -- incremented
        # once per non-Pass action across priority rounds, declare_blockers,
        # mulligan, and end-of-turn discard loops. Pass itself never
        # counts. 2-player only; unused in 1-player mode.
        self.actions_taken = 0

        # actions_taken as it stood when the pregame mulligan phase finished
        # (game.turn.game_coroutine). Lets a reward measure GAMEPLAY actions
        # only, excluding mulligan/keep/bottom picks.
        self.pregame_actions = 0

        # Turns this player discarded to hand size at cleanup (per-turn
        # count, not per card) -- a proxy for hoarding cards. Read by
        # rl.rewards.deploy_reward.
        self.cleanup_discard_turns = 0

        # Total mana LOST to rule 500.4's pool-empty over the whole game,
        # summed across every clear. Unconditional -- a raw diagnostic
        # (logging/viz), not read by any reward function. Never reset.
        self.mana_burnt_total = 0

        # Whole-game single-pip-tagged subset of mana_burnt_total, never
        # reset. Diagnostic-only (logging/viz); with_dense_mana_burn_penalty
        # reads the per-turn field below instead.
        self.mana_burnt_total_single_pip = 0

        # DENSE reward mailbox, meant to be drained (read then zeroed) by a
        # reward_fn tagged consumes_mana_mistake=True (none currently is).
        # A narrower subset of mana_burnt_total: only added to once
        # cost_paid_this_phase, triggers_fired_this_phase, AND the optional
        # GameState.on_mana_burn hook have all failed to justify the burn.
        # Never reset by the engine itself.
        self.mana_mistake_burn = 0

        # Cumulative mana burnt THIS TURN (both players), reset by
        # game.turn._run_turn_gen at the start of every new turn.
        # Unconditional, unlike mana_mistake_burn -- punishes floating more
        # than can be spent within a turn. mana_burnt_this_turn_single_pip
        # below is the filtered subset with_dense_mana_burn_penalty reads.
        self.mana_burnt_this_turn = 0

        # Subset of mana_burnt_this_turn: only the portion still TAGGED
        # single-pip at each phase-boundary burn. Because spend_one_pip
        # always spends an untagged pip before a tagged one, whatever's
        # still tagged at phase end is genuinely avoidable waste.
        # rl.rewards.with_dense_mana_burn_penalty's actual input.
        self.mana_burnt_this_turn_single_pip = 0

        # The Hill-curve badness value (rl.rewards._hill) already charged
        # against reward for this turn -- with_dense_mana_burn_penalty's
        # running baseline, so each call charges only the marginal increase
        # since the last one. Reset alongside mana_burnt_this_turn_single_pip.
        self.mana_burn_penalty_credited = 0.0

        # Running total of dense mana-burn penalty actually charged against
        # reward this GAME (never reset per-turn, unlike
        # mana_burn_penalty_credited). Lets with_dense_mana_burn_penalty
        # clamp each charge so the running total never exceeds a configured
        # ceiling -- a deliberately blunt cap for PPO/GAE stability, since an
        # unbounded-with-episode-length reward term caused real training
        # problems here before.
        self.mana_burn_penalty_charged_total = 0.0

    def draw(self, n=1):
        """Draws `n` cards. An empty library raises DeckedOut immediately
        (real Magic: drawing from an empty library is an instant loss,
        mid-effect) -- sets decked_out first, so game.turn._run_turn_gen's
        try/except can unwind cleanly.

        Increments cards_drawn_this_turn per card drawn and queues an
        "automatic" trigger-queue entry for every graveyard card whose
        registry entry has an "on_draw_count" spec matching the new count."""
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
    state.players[state.active_idx].<attr> -- lets every card-effect
    function read/write "my own board" unchanged under a 2-player game."""
    def getter(self):
        return getattr(self.players[self.active_idx], attr)

    def setter(self, value):
        setattr(self.players[self.active_idx], attr, value)

    return property(getter, setter)


class GameState:
    """Shared turn/stack/pending-resolution bookkeeping, plus a list of
    PlayerStates -- two for a real game, one for a single-board unit test.
    Every zone accessor (state.hand, state.battlefield, ...) is a property
    proxying to state.players[state.active_idx] -- whoever currently holds
    priority, which a priority consult (e.g. _declare_blockers_gen) can
    temporarily flip away from the turn owner. state.turn_player_idx is the
    distinct fact of whose turn it structurally is: real Magic's land-drop/
    sorcery-speed rules gate on "your own turn," not just phase, so those
    checks need turn_player_idx specifically, never active_idx."""

    def __init__(self, on_the_play, rng=None, players=None, event_log=None):
        # `players`, if given, replaces the single-player list built from
        # on_the_play (used by new_multiplayer_game_state, where each
        # player supplies their own on_the_play instead).
        self.players = players if players is not None else [PlayerState(on_the_play)]
        self.active_idx = 0

        # None (default) means logging is off -- checked first in log_event
        # so instrumented call sites cost one attribute read, never a dict
        # build. Pass a list (e.g. `GameState(..., event_log=[])`) to turn
        # logging on: log_event appends one structured dict per state
        # change, in order, making the whole game reconstructible.
        self.event_log = event_log

        # Optional (state, player_idx) -> bool hook, consulted by
        # game.turn._empty_mana_pools right before a genuinely-unattributed
        # pool clear -- answers "was anything legally castable with this
        # floating mana," which needs the action table this module has no
        # knowledge of. None (default) means "don't know, don't tally a
        # mistake." Stamped on by game.turn.run_multiplayer_game's
        # on_mana_burn param.
        self.on_mana_burn = None

        # Optional (state, player_idx, single_pip_burnt) -> None hook,
        # called by game.turn._empty_mana_pools for EVERY player on EVERY
        # phase-boundary clear (0 when nothing was floating) -- lets a
        # consumer (rl.training.train) charge the dense mana-burn penalty
        # against the actual recent Tap actions rather than whatever action
        # happens to be pending at the seat's next decision. Fired
        # unconditionally so a consumer's per-seat bookkeeping resets
        # cleanly every phase boundary. None (default) is a no-op. Stamped
        # on by game.turn.run_multiplayer_game's on_single_pip_burn param.
        self.on_single_pip_burn = None

        # Whose turn it structurally is -- distinct from active_idx (see
        # class docstring). Set once per turn by game.turn._run_turn_gen's
        # turn-start setup; defaults to active_idx here so a hand-built
        # self-check state still gets a sensible value.
        self.turn_player_idx = self.active_idx

        self.turn_number = 0
        self.rng = rng or random.Random()

        self.phase = None  # set by game.turn._run_turn_gen at the start of each phase; None until then

        # turn_won: the turn the game ended on (life hitting 0, horizon cap,
        # decking out), None while in progress. winner: which player
        # (index) won, only ever set in a 2-player game -- stays None for a
        # bare failure (horizon reached, or 1-player decking out).
        self.turn_won = None
        self.winner = None

        # Which player (index) currently has THE INITIATIVE (Avenging
        # Hunter), or None. game.effects.undercity.take_initiative sets it
        # (from Avenging Hunter's ETB, or a queued trigger on combat damage
        # to the current holder); the holder ventures into Undercity each
        # of their upkeeps (game.turn.upkeep_step).
        self.initiative_idx = None

        # None, or a dict describing an in-progress multi-step decision
        # (paying a cost, resolving a scry/surveil, choosing a search
        # target, ...) that must fully resolve before any other action is
        # legal. Always the active player's own decision. See
        # game.resolution.begin_resolution/complete_resolution.
        self.pending_resolution = None

        # Reentrant counter: >0 while a spell/ability is still being CAST/
        # ACTIVATED (601.2f-601.2i / 602.2) -- from game.mana.begin_pay_cost
        # opening through the matching game.effects.stack.push_to_stack that
        # commits it. Real Magic pays a single spell/ability's total cost
        # (including choices like "which permanent to sacrifice" for an
        # additional cost) as ONE atomic action: no state-based actions, no
        # priority, run in between. game.turn._run_priority_round_gen
        # suppresses check_state_based_actions/refizzle_if_now_targetless
        # while this is >0, so a creature that's the caster's only
        # sacrifice fodder can't be claimed by an SBA mid-payment and leave
        # a later additional-cost choice with zero legal targets (2026-08-27
        # production crash -- cast_eviscerators_insight's own sacrifice
        # fizzled to None mid-cast; see tests.game.catalog.test_black_cards.
        # test_sac_cost_fodder_dying_mid_payment_crashes_the_resolve).
        #
        # A counter, not a bool: begin_pay_cost can be reentered before the
        # outer spell reaches the stack (e.g. a cost that itself triggers
        # another cost payment). Incremented by begin_pay_cost by default;
        # its one documented exception is game.resolution.handlers_casting.
        # pay_unless_pay (Ward/Spell Pierce's "pay to avoid an outcome"),
        # which pays a cost for something ALREADY on the stack and never
        # reaches push_to_stack -- begin_pay_cost's counts_as_cast=False
        # opts that call out. Self-heals to 0 at the top of every fresh
        # _run_priority_round_gen (once per phase/step) as a blast-radius
        # bound against any uncaught exception or as-yet-undiscovered
        # begin_pay_cost path that skips push_to_stack.
        self.casting_depth = 0

        # None, or a dict describing an in-progress GATE-FREE mana ability's
        # own multi-step choice -- Saruli Caretaker's "tap another
        # creature, then choose a color", or a mana filter's "pay the
        # activation cost, then choose the output color".
        #
        # Deliberately a SEPARATE field from pending_resolution: a
        # gate-free mana ability (605.1a/605.3b -- legal in any priority
        # window, even mid-resolution of something else) must be able to
        # open its own multi-step choice without clobbering whatever's
        # already pending. While set, it takes exclusive priority over
        # everything else (drl_env.legal_action_mask); once complete,
        # control returns to whatever pending_resolution already held.
        #
        # Every subdecision carries {"owner": seat index} -- whoever was
        # active when it opened. It's a single GLOBAL slot, so without an
        # owner its exclusive priority would land on whichever seat is
        # asked NEXT rather than the one that opened it (a Saruli
        # subdecision left open by the defender once crashed the attacker's
        # combat-damage assignment with an all-False action mask,
        # 2026-08-16). Read via active_mana_subdecision, never raw, from
        # anywhere deciding what a given seat may legally do.
        self.mana_subdecision = None

        # True for the duration of game.turn.Phase.END's own CLEANUP portion
        # (after the real end-step priority round and cleanup_step's
        # hand-size discard have both started -- see _run_turn_gen's own
        # handling). Rule 514.3: normally no player receives priority at all
        # during cleanup. AUTHORIZED SIMPLIFICATION (owner-approved
        # 2026-08-10): this flag additionally revokes gate-free mana
        # abilities' own real-rules any-window legality (605.1a/605.3b --
        # see drl_env._actions_mana's own gate on it) specifically while
        # True, so nothing can float new mana during cleanup at all. Kept as
        # a dedicated flag rather than a Phase.CLEANUP enum value (which
        # would change len(game.turn.Phase) and silently break every
        # existing checkpoint's rl.model.deck.SCALAR_FEATURE_DIM / rl.decision.agent's own
        # phase one-hot) and rather than gating on pending_resolution["kind"]
        # == "discard" (that kind is shared with unrelated non-cleanup
        # effects -- Faithless Looting, Grab the Prize -- where mana
        # abilities must stay legal), matching mana_subdecision's own
        # "dedicated exclusive-mode field, not a pending_resolution kind"
        # reasoning above. Does NOT block a Madness cast-or-graveyard
        # decision queued by a cleanup discard (game.resolution.
        # handlers_library._discard_one) -- that's a mandatory game-state
        # resolution, not a discretionary action, and still gets its own
        # priority round via _run_turn_gen's existing trigger_queue check;
        # this flag just stays True through that round too. Reset to False
        # in a finally, so an early return (a game-ending state-based action
        # mid cleanup) can never leave it stuck True into the next turn.
        self.in_cleanup = False

        # list[dict {"card_def": CardDef, "resolve": (state, card_def) ->
        # None}], top of stack = last element. Shared by both players (real
        # Magic's stack is one object), even though only the active player
        # ever pushes under this engine's no-interrupt-window rule. See
        # game.effects.stack.push_to_stack/resolve_top_of_stack.
        self.stack = []

        # The CardDef currently resolving off the stack, set by
        # game.effects.stack.resolve_top_of_stack. It left its controller's
        # hand at cast and must never re-enter it, so the resolve's own
        # zone-move step checks identity against this instead of expecting
        # the card in hand. None whenever nothing resolves.
        self.resolving_card = None

        # Monotonic per-game counter minting the `iid` of every CardInstance/
        # Permanent (see new_instance/new_permanent/mint_iid). Never reused, so
        # a flickered/returned card gets a genuinely fresh identity in logs.
        self._next_iid = 0

    hand = _active_player_property("hand")
    battlefield = _active_player_property("battlefield")
    library = _active_player_property("library")
    graveyard = _active_player_property("graveyard")
    exile = _active_player_property("exile")
    impulse = _active_player_property("impulse")
    trigger_queue = _active_player_property("trigger_queue")
    lands_played_this_turn = _active_player_property("lands_played_this_turn")
    cards_drawn_this_turn = _active_player_property("cards_drawn_this_turn")
    # cost_paid_this_phase is proxied like the above -- always pays out of
    # whichever player holds priority. triggers_fired_this_phase is
    # deliberately NOT proxied: promote_triggers_to_stack writes it for a
    # player who may not be active_idx, so it writes
    # state.players[player_idx] directly instead.
    cost_paid_this_phase = _active_player_property("cost_paid_this_phase")
    mana_pool = _active_player_property("mana_pool")
    mana_pool_single_pip = _active_player_property("mana_pool_single_pip")
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

    @property
    def active_mana_subdecision(self):
        """mana_subdecision, but only when the ACTIVE seat is the one that
        opened it -- otherwise None.

        A mana subdecision claims exclusive priority: while one is open its
        owner may take no other action. The raw slot is global and can't
        tell owner from non-owner, so anything asking "what may THIS seat
        legally do" must read this property; only execute_* handlers
        (which run as the owner) may touch the raw slot.

        Without this, a subdecision the defender left open once silently
        governed the attacker's next decision (an all-False action mask
        crash, 2026-08-16 -- see mana_subdecision's own comment)."""
        sub = self.mana_subdecision
        if sub is None or sub.get("owner") != self.active_idx:
            return None
        return sub

    def draw(self, n=1):
        # Single generic draw-logging hook: every card entering a hand from
        # its library (draw_step, Faithless Looting, opening 7, a mulligan
        # redraw) is recorded as one "zone_move" (library->hand,
        # reason="draw") naming the cards. try/finally so a draw that decks
        # out partway still logs whatever reached hand first.
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

    def mint_iid(self):
        """Next never-reused per-game instance id. The single source of iids
        for both new_instance (non-battlefield) and new_permanent (board), so
        every physical card in a game has a distinct one."""
        iid = self._next_iid
        self._next_iid += 1
        return iid

    def new_instance(self, card_def):
        """Mint a fresh CardInstance for `card_def` -- the one place a
        non-battlefield card instance is born. Reached via move_card, so a
        card that changes zones is genuinely a new object (MTG 400.7) and
        stale targets on it fizzle."""
        return CardInstance(card_def, self.mint_iid())

    def new_permanent(self, card_def, tapped=False):
        """Battlefield counterpart of new_instance: mint a fresh Permanent with
        the next iid. game.effects.casting.enters_battlefield uses this so board
        permanents share the same per-game identity counter as graveyard cards."""
        return Permanent(card_def, tapped=tapped, iid=self.mint_iid())

    def move_card(self, card, destination):
        """The choke point for a card entering a tracked non-battlefield
        zone as a NEW object. Mints a fresh CardInstance from `card`'s
        CardDef into `destination` and returns it. `card` may be a CardDef
        or an existing CardInstance/Permanent -- the source object is NOT
        reused, so any target captured on it is now stale (MTG 400.7).

        Removal from the SOURCE zone stays at the existing removal sites
        (state-based death, stack resolution, hand discard); this helper
        owns only the minting half. Battlefield entry has its own choke
        point, enters_battlefield.

        Today `destination` is always a graveyard list; hand/library/exile
        are still CardDef-based."""
        card_def = card.card_def if isinstance(card, CardInstance) else card
        inst = self.new_instance(card_def)
        destination.append(inst)
        return inst

    def log_event(self, kind, **fields):
        """Appends one structured event to self.event_log, if logging is on
        -- else an immediate no-op, so every instrumented call site costs
        exactly one attribute check when logging is off (the default).

        Every event automatically carries the same envelope -- turn, phase,
        active_idx, turn_player_idx -- so a call site only supplies `kind`
        plus whatever fields are specific to that event. This is an EVENT
        log, not a periodic state snapshot: recording each change at its
        exact call site captures automatic side effects (a stack
        resolution, a state-based death, a mana clear) that a
        decision-only snapshot would miss between two logged points."""
        if self.event_log is None:
            return
        # Sanitize field values at the source so no event carries a raw engine
        # object (a CardDef, a closure), which would break the JSON write or
        # an MP worker's pickle. Primitives pass through; an object becomes
        # its string .name if it has one, else a short repr; containers recurse.
        def _safe(value):
            if isinstance(value, (str, int, float, bool)) or value is None:
                return value
            if isinstance(value, dict):
                return {k: _safe(v) for k, v in value.items()}
            if isinstance(value, (list, tuple)):
                return [_safe(v) for v in value]
            name = getattr(value, "name", None)
            return name if isinstance(name, str) else repr(value)
        self.event_log.append({
            "kind": kind,
            "turn": self.turn_number,
            "phase": self.phase.value if self.phase is not None else None,
            "active_idx": self.active_idx,
            "turn_player_idx": self.turn_player_idx,
            **{k: _safe(v) for k, v in fields.items()},
        })


def build_shuffled_library(decklist, rng, force_land_count=None):
    """Expands a decklist's quantities into CardDef refs (via
    game.registry.CARD_DEFS) and shuffles.

    force_land_count=None (default): plain uniform shuffle.
    force_land_count=N: a TRAINING-ONLY knob (rl.training.train.collect_rollout's
    stratify_0land_pct) that lands exactly N lands in the top 7 -- still a
    real, legal 60-card library, just drawn from the slice of shuffles
    where the opening hand has N lands. Every card past position 7 is
    independently shuffled."""
    library = []
    for name, qty, *_rest in decklist:
        library.extend([registry.CARD_DEFS[name]] * qty)
    if force_land_count is None:
        rng.shuffle(library)
        return library
    lands = [c for c in library if c.card_type.name == "LAND"]
    nonlands = [c for c in library if c.card_type.name != "LAND"]
    assert 0 <= force_land_count <= 7, f"force_land_count must be 0-7, got {force_land_count}"
    assert len(lands) >= force_land_count and len(nonlands) >= 7 - force_land_count, \
        "decklist doesn't have enough lands/nonlands to force this opening land count"
    rng.shuffle(lands)
    rng.shuffle(nonlands)
    hand = lands[:force_land_count] + nonlands[:7 - force_land_count]
    rest = lands[force_land_count:] + nonlands[7 - force_land_count:]
    rng.shuffle(hand)
    rng.shuffle(rest)
    return hand + rest


def new_multiplayer_game_state(decklists, starting_player_idx, rng, event_log=None, stratify=None):
    """N-player entry point -- one decklist per player, decks may differ.
    Only starting_player_idx is "on the play" (skips their first draw --
    see turn.draw_step) and takes the first turn; every other player
    starts with on_the_play=False. Every player draws their own opening 7.

    stratify=None (default): every seat gets a plain uniform shuffle.
    stratify=(seat_idx, land_count): TRAINING-ONLY (see
    build_shuffled_library's force_land_count) -- forces just that seat's
    opening hand to land_count lands."""
    players = [
        PlayerState(on_the_play=(i == starting_player_idx))
        for i in range(len(decklists))
    ]
    state = GameState(
        on_the_play=players[starting_player_idx].on_the_play, rng=rng, players=players, event_log=event_log,
    )
    state.turn_player_idx = starting_player_idx
    for idx, decklist in enumerate(decklists):
        force = stratify[1] if stratify is not None and stratify[0] == idx else None
        state.players[idx].library = build_shuffled_library(decklist, rng, force_land_count=force)
        state.active_idx = idx   # attribute each opening draw to its own drawer
        state.draw(7)            # routed through GameState.draw so the opening hand is logged like any other draw
    state.active_idx = starting_player_idx
    return state

