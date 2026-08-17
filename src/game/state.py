"""Zone & state model: PlayerState (one player's zones + life) plus GameState
(shared turn/stack/pending-resolution bookkeeping + a list of PlayerStates).

GameState exposes every zone (hand/battlefield/library/graveyard/exile/
mana_pool/trigger_queue/attackers/... ) as a property proxying to
state.players[state.active_idx] -- whoever currently holds priority. This is
what lets every card-effect function + mana/resolution/effects stay unchanged:
they only ever meant "my own board," and the proxy makes that correct for the
active player in both 1- and 2-player games. state.opponent is the one
opponent-facing accessor (win_check.deal_damage_to_opponent).

mana_pool holds floated-but-unspent mana; spending it is always an explicit
model action, never automatic, and it empties at every step/phase boundary
for both players (turn._empty_mana_pools, rule 500.4), not just once per turn."""

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


class CardInstance:
    """A specific physical card, distinct from every other copy of the same
    name, in whatever zone holds it. Wraps the shared, interned CardDef (its
    unchanging rules identity, one object per name -- game.registry.CARD_DEFS)
    and adds a per-game `iid` so two same-named copies are tellable apart.

    IDENTITY IS OBJECT IDENTITY. A target/selection captures the exact instance
    object and stays legal only while that object is still in its zone; every
    zone change mints a NEW instance (see GameState.move_card), so a card that
    changes zones is genuinely a new object (MTG rule 400.7) and any target
    locked on the old object fizzles. `iid` is engine-internal (deterministic
    logging/debugging so `(name, slot)`/name aren't ambiguous once duplicates
    exist) -- NEVER shown to the agent and NEVER the targeting handle (the
    object reference is). Deliberately does NOT override __eq__/__hash__:
    default identity is the whole point.

    Attribute access PROXIES to card_def (`.name`, `.card_type`, `.cast_cost`,
    `.effect_id`, `.extra`, ...) via __getattr__, so the large body of engine
    code that reads those straight off a zone element keeps working unchanged
    once that element became a CardInstance.

    SCOPE: the graveyard holds CardInstances and the battlefield
    holds Permanents (which subclass this). hand/library/exile are DEFERRED --
    still list[CardDef] -- until a card needs distinct instances there.
    FUTURE (400.7 exceptions): no pool
    card yet has a linked ability that TRACKS an object across a zone change
    (Adventure/Foretell/"return THIS card"); that would need the instance to
    stay linked across the specific move rather than being reminted here."""

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
    """One player's zones, turn-scoped counters, and life total. GameState
    holds a list of these, one per player -- two for a real game, one for
    a unit test that only needs to exercise a single board. Nothing in
    this class itself knows or cares how many other PlayerStates, if any,
    exist alongside it."""

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

        # Phase-scoped mistake-detection flags for the dense mana-burn
        # penalty (rl.rewards.with_mana_mistake_penalty) -- both read then
        # reset every phase boundary by game.turn._empty_mana_pools.
        # cost_paid_this_phase: a real cast/ability payment happened
        # (game.mana.execute_pool_spend). triggers_fired_this_phase: a
        # trigger was promoted to the stack from this player's own queue
        # (game.effects.triggers.promote_triggers_to_stack) -- covers the
        # Writhing Chrysalis/Gixian Infiltrator case, sacrificing a mana
        # source as a combat trick, where the mana is an incidental
        # byproduct of an action taken for its trigger, not for the mana.
        # Either flag alone exempts a burn from counting as a mistake.
        self.cost_paid_this_phase = False
        self.triggers_fired_this_phase = False

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

        # Parallel shadow-count to mana_pool: dict[str symbol -> int count]
        # -- how many of the CURRENTLY FLOATING pips of each color trace
        # back to a "single-pip" mana-producing EVENT (one that added
        # EXACTLY 1 symbol to the pool in that one event -- computed
        # dynamically per event by game.mana.float_mana, len(symbols)==1,
        # NOT a static per-source-kind classification). A plain land or
        # Llanowar Elves is always single-pip; Rakdos Carnarium and Utopia
        # Sprawl's automatic bonus (always 2+ symbols in one tap) never is;
        # a Tron land is single-pip only while NOT all three Tron types are
        # controlled; Priest of Titania/Overgrown Battlement are single-pip
        # only in the edge case their count happens to resolve to exactly 1
        # (e.g. Priest with zero other Elves out) -- deliberate, not a bug
        # to guard against. A mana filter's output pip (drl_env.
        # _actions_mana._filter_mana_execute) is the one explicit
        # exception: always exactly 1 symbol but forced untagged
        # (float_mana(..., taggable=False)), since it's a deliberate
        # pool->pool conversion, not reflexive tapping. An Eldrazi Spawn's
        # sac-for-{C} (effects.tokens.activate_eldrazi_spawn_sac) is the
        # same exception for a different reason: the float and the
        # sacrifice are one atomic action, so it's forced untagged too.
        #
        # A mana-producing permanent that's SACRIFICED (game.effects.
        # state_based.sacrifice_to_graveyard) or gets UNTAPPED by an effect
        # after already being tapped this phase (Quirion Ranger, Sewer-
        # veillance Cam) instead gets an after-the-fact discount via
        # game.mana.discount_departing_source -- tapping it for mana right
        # before/around either event would have been free, so any of its
        # own colors still tagged here are retroactively excused, picking
        # whichever candidate color has the most tagged mana when the
        # source can produce more than one (see that function's docstring).
        #
        # A lossless shadow COUNT, not a literal per-pip list -- sufficient
        # for this one boolean tag dimension (doesn't scale to more tag
        # dimensions, fine for now). Invariant: mana_pool_single_pip[c] <=
        # mana_pool[c] for every color c, always. Decremented by game.mana.
        # spend_one_pip per the spend-order convention -- an UNTAGGED pip
        # of a color is always spent before a TAGGED one, so this only
        # shrinks once no untagged pip of that color remains (see
        # spend_one_pip's own docstring for why). Cleared alongside
        # mana_pool at every phase boundary (game.turn._empty_mana_pools).
        # Replaces the earlier (2026-08) whole-phase
        # unmetered_mana_tapped_this_phase boolean with per-pip
        # attribution -- see mana_burnt_this_turn_single_pip below for what
        # this feeds.
        self.mana_pool_single_pip = {}

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

        # actions_taken as it stood the instant the pregame mulligan phase
        # finished (snapshotted in game.turn.game_coroutine). Lets a reward
        # measure GAMEPLAY actions only -- actions_taken counts every mulligan/
        # keep/bottom pick too (game.turn._run_mulligan_gen), which would
        # otherwise inflate a mulligan-heavy winner's action count and wrongly
        # dock its efficiency reward. 0 until that snapshot (and for any state
        # that never runs the pregame phase).
        self.pregame_actions = 0

        # Number of TURNS this player discarded to hand size at its own cleanup
        # (game.effects.state_based.cleanup_step) -- a per-turn count, not per
        # card. A proxy for hoarding drawn cards it never deployed; read by
        # rl.rewards.deploy_reward. Only the hand-size cleanup discard bumps
        # this, never any other discard effect.
        self.cleanup_discard_turns = 0

        # Total mana LOST to rule 500.4's automatic pool-empty (game.turn.
        # _empty_mana_pools) over the whole game -- summed across every
        # non-empty clear, all colors combined (a {"R": 2} clear adds 2, not
        # 1). Unconditional (every burnt pip, justified or not) -- a raw
        # diagnostic for logging/viz, no longer read by any reward function
        # (see mana_mistake_burn below for the reward-facing, conditional
        # signal). Never reset once the game is underway.
        self.mana_burnt_total = 0

        # Whole-game cumulative subset of mana_burnt_total above: only the
        # single-pip-tagged portion (same tag rule as mana_burnt_this_turn_
        # single_pip below -- see PlayerState.mana_pool_single_pip's own
        # docstring), but never reset at a turn boundary. Diagnostic-only
        # (logging/viz -- rl.run_cross_league_eval's mana-burn comparison),
        # not read by any reward function; with_dense_mana_burn_penalty
        # reads the per-turn field below, not this one. Never reset once the
        # game is underway, same lifecycle as mana_burnt_total.
        self.mana_burnt_total_single_pip = 0

        # DENSE reward mailbox -- rl.rewards.with_mana_mistake_penalty drains
        # it (reads then zeroes) on every transition. A narrower subset of
        # mana_burnt_total above: game.turn._empty_mana_pools only adds to
        # this one once cost_paid_this_phase, triggers_fired_this_phase, AND
        # "was anything legally castable with the floating pool" (the
        # optional GameState.on_mana_burn hook -- see its own docstring for
        # why that check can't live in this module) have all failed to
        # justify the burn. Never reset by the engine itself -- only ever
        # drained by the reward wrapper reading it. NOT what league self-play
        # currently trains against -- see mana_burnt_this_turn below, which is
        # (2026-08).
        self.mana_mistake_burn = 0

        # Cumulative mana burnt THIS TURN (both players -- the non-active
        # player can float mana too, see game.turn._empty_mana_pools's own
        # docstring), reset by game.turn._run_turn_gen at the start of every
        # new turn. Unconditional like mana_burnt_total above (every burnt
        # pip counts, no cost-paid/trigger-fired/nothing-castable exemption)
        # -- a deliberately blunter signal than mana_mistake_burn: the point
        # is punishing floating more than you can spend WITHIN a turn, not
        # just avoidable waste. Raw diagnostic -- mana_burnt_this_turn_single_pip
        # below (a filtered subset of this) is what with_dense_mana_burn_penalty
        # actually reads for reward.
        self.mana_burnt_this_turn = 0

        # Subset of mana_burnt_this_turn above: at each phase-boundary burn,
        # only the portion of the burnt pool that was still TAGGED
        # single-pip (sum(player.mana_pool_single_pip.values()) at that
        # moment) -- game.turn._empty_mana_pools' per-pip attribution
        # (2026-08, replacing the earlier whole-phase
        # unmetered_mana_tapped_this_phase exclusion). Because
        # game.mana.spend_one_pip always spends an UNTAGGED pip of a color
        # before a TAGGED one, whatever single-pip mana is still floating
        # (and tagged) at phase end is genuinely avoidable waste: a
        # Priest-of-Titania-style burst's own excess is untagged and gets
        # spent/burnt first, so it never reaches this counter, while a
        # single-pip tap left floating unnecessarily does. rl.rewards.
        # with_dense_mana_burn_penalty's actual input. Reset alongside
        # mana_burnt_this_turn, same turn-boundary lifecycle.
        self.mana_burnt_this_turn_single_pip = 0

        # The Hill-curve badness value (rl.rewards._hill) ALREADY charged
        # against reward for this turn's mana_burnt_this_turn_single_pip so
        # far -- with_dense_mana_burn_penalty's own running baseline, so each
        # reward call charges only the MARGINAL increase in badness since the
        # last call (a genuinely dense, per-transition credit whose sum across
        # a turn still telescopes to exactly _hill(total_this_turn, c, p) by
        # the turn's end -- same total a one-shot terminal score would have
        # given, just attributed to the transitions that actually caused it).
        # Reset alongside mana_burnt_this_turn_single_pip.
        self.mana_burn_penalty_credited = 0.0

        # Running total of dense mana-burn penalty ACTUALLY charged against
        # reward so far this GAME (not per-turn -- never reset by
        # game.turn._run_turn_gen, same "whole game" lifetime as
        # mana_burnt_total above). with_dense_mana_burn_penalty's own safety
        # backstop: mana_burn_penalty_credited resetting every turn means the
        # per-turn charges do NOT telescope to a bounded whole-game total the
        # way a true potential function would (a turn boundary drops
        # whatever was owed back instead of refunding it) -- so nothing
        # otherwise stops a long run of bad turns from summing to a penalty
        # that dwarfs the terminal win/loss signal entirely. This field lets
        # the wrapper clamp each charge so the running total never exceeds
        # its own configured ceiling -- a deliberately blunt cap, not
        # principled shaping, chosen for PPO/GAE stability (an unbounded-
        # with-episode-length reward term is real-world proven trouble here:
        # see git history on cbd7379, which walked back an earlier, similarly
        # unbounded mana-burn penalty specifically because of training
        # problems it caused). Never reset once the game is underway.
        self.mana_burn_penalty_charged_total = 0.0

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
    existing card-effect function and mana.py/resolution/*.py/
    game/effects/*.py stay unchanged under a 2-player game."""
    def getter(self):
        return getattr(self.players[self.active_idx], attr)

    def setter(self, value):
        setattr(self.players[self.active_idx], attr, value)

    return property(getter, setter)


class GameState:
    """Shared turn/stack/pending-resolution bookkeeping, plus a list of
    PlayerStates -- two for a real game (self-play training always runs
    2-player via turn.run_multiplayer_game), one for a unit test that only
    needs a single board. Every zone accessor below (state.hand,
    state.battlefield, ...) is a property
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
        # instrumented call site across mana.py/turn.py/resolution/*.py/
        # game/effects/*.py costs one attribute read on the hot training
        # path, never a dict build. Pass a plain list to turn logging on
        # for one game (e.g. `GameState(..., event_log=[])`, or via
        # new_multiplayer_game_state's own event_log param)
        # -- log_event appends one structured dict per instrumented state
        # change, in occurrence order, which is what makes the whole game
        # reconstructible afterward (see log_event's own docstring for the
        # shared envelope every event gets).
        self.event_log = event_log

        # Optional (state, player_idx) -> bool hook, consulted by game.turn.
        # _empty_mana_pools right before a genuinely-unattributed pool clear
        # (nothing paid, nothing triggered this phase) -- answers "was
        # anything legally castable with this floating mana," which needs
        # the action table this module deliberately has no knowledge of.
        # None (the default -- every plain rules-engine/test caller) means
        # "don't know, don't tally a mistake." Stamped onto state directly by
        # game.turn.run_multiplayer_game's own on_mana_burn param (the DRL
        # layer wires one in via rl.train.collect_rollout) -- not threaded
        # through game_coroutine/_run_turn_gen's own signatures, since
        # nothing in that call chain besides _empty_mana_pools needs it.
        self.on_mana_burn = None

        # Optional (state, player_idx, single_pip_burnt) -> None hook,
        # called by game.turn._empty_mana_pools for EVERY player on EVERY
        # phase-boundary clear (single_pip_burnt=0 when nothing was
        # floating) -- the credit-assignment fix for rl.rewards.
        # with_dense_mana_burn_penalty: that wrapper's own math is dense and
        # correctly telescoping, but reward_fn is only ever called at a
        # seat's OWN next decision, which can land many actions after the
        # taps that actually produced the float (see rl.train.collect_
        # rollout's pending-reward bookkeeping) -- in practice the whole
        # charge was landing on whatever action happened to be pending at
        # that moment (almost always that seat's own end-of-phase Pass, not
        # the Tap actions that caused it). This hook fires synchronously,
        # inside the SAME engine call that computes single_pip_burnt, so a
        # consumer (rl.train's on_single_pip_burn) can charge it against
        # the actual recent Tap actions directly instead. Fired
        # unconditionally (not just when something burnt) so a consumer's
        # own per-seat bookkeeping resets cleanly every phase boundary, not
        # just the ones that happened to burn something -- a clean phase
        # must still reset the tracker, or a later phase's burn would
        # wrongly inherit blame for taps that were fully spent long before
        # it. None (the default) is a no-op, same convention as
        # on_mana_burn above. Stamped onto state by game.turn.
        # run_multiplayer_game's own on_single_pip_burn param.
        self.on_single_pip_burn = None

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
        # progress. turn_won is set whenever the game ends for any reason
        # (a life total hitting 0, a horizon cap, decking out); winner
        # names WHICH player won and is only ever set in a 2-player game.
        # winner stays None for a bare failure (horizon reached, or --
        # in a 1-player unit-test state -- decking out with no opponent to
        # award the win to).
        self.turn_won = None
        self.winner = None

        # Which player (index into state.players) currently has THE INITIATIVE
        # (Avenging Hunter / The Initiative), or None if no one does. A single
        # shared designation, like the monarch. game.effects.undercity.
        # take_initiative sets it (and queues that player's venture) -- called
        # directly from Avenging Hunter's own ETB, or from a queued
        # "take_initiative" triggered ability when combat damage to the
        # current holder resolves it (game.effects.combat queues that trigger,
        # game.effects.triggers resolves it); the holder ventures into
        # Undercity at the start of each of their upkeeps (game.turn.
        # upkeep_step).
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

        # None, or a dict describing an in-progress GATE-FREE mana ability's
        # own multi-step choice -- currently two shapes: Saruli Caretaker's
        # "tap another creature, then choose a color" (602.5g cost choice
        # then the ability's own color-choice effect), and a mana filter's
        # "pay the activation cost [fixed table row], then choose the
        # output color" (the cost half is a flat action, not a stage of
        # this dict at all -- see drl_env._actions_mana._filter_mana_execute).
        # Deliberately a SEPARATE field from pending_resolution, not a
        # nested/stacked use of it: pending_resolution is a single slot,
        # and a gate-free mana ability (605.1a/605.3b -- legal in ANY
        # priority window, even mid-resolution of something else, e.g. mid
        # pay_unless) must be able to open its own multi-step choice
        # WITHOUT clobbering whatever's already pending. While this is set,
        # it takes exclusive priority over everything else (drl_env.
        # legal_action_mask/rl.action_bridge's own dispatch) -- mirrors how
        # activating a mana ability is atomic from everyone else's view in
        # real Magic; once the choice completes, control returns to
        # whatever pending_resolution already held, completely untouched.
        #
        # The ONE stage every such ability shares -- "choose_color" -- is a
        # generic primitive (game.resolution.begin_mana_color_choice) that
        # knows nothing about who opened it: {"stage": "choose_color",
        # "can_produce": (state, color) -> bool, "on_choose_color": (state,
        # color) -> None}, both bound as closures by whichever ability is
        # completing (Saruli: tap its stored target then activate its own
        # "mana"-spec source; a filter: add the color straight to the
        # pool). Saruli's OWN first stage additionally carries {"stage":
        # "choose_target", "source": Permanent, "target_predicate":
        # callable, "target": Permanent | None} -- game.resolution.
        # begin_mana_subdecision/execute_mana_subdecision_target -- a mana
        # filter has no equivalent first stage; it opens straight into
        # choose_color from its own flat fixed-table row's execute.
        # Every subdecision additionally carries {"owner": seat index} --
        # whoever was active when it opened. It is a single GLOBAL slot, and
        # its whole point is exclusive priority ("suppress everything except
        # the matching-stage closures", see _actions_table.legal_action_mask),
        # so without an owner that exclusivity lands on whichever seat is asked
        # NEXT rather than the one that opened it. A Saruli Caretaker
        # subdecision left open by the DEFENDER did exactly that on 2026-08-16:
        # the ATTACKER was then asked to assign combat damage, got a pointer
        # mask built for the defender's tap-target choice instead, and crashed
        # with an all-False action mask. Read it through
        # active_mana_subdecision, never raw, from anywhere that is deciding
        # what a given seat may legally do.
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
        # existing checkpoint's rl.deck.SCALAR_FEATURE_DIM / rl.agent's own
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
    # cost_paid_this_phase is proxied like the above -- game.mana.
    # execute_pool_spend always pays out of whichever player currently holds
    # priority (state.active_idx), same as state.mana_pool itself.
    # triggers_fired_this_phase is deliberately NOT proxied: promote_
    # triggers_to_stack writes it for a player who may not be active_idx
    # (the non-turn player's own queue) -- trigger_queue IS proxied above,
    # but promote_triggers_to_stack writes state.players[player_idx].
    # trigger_queue directly rather than through that proxy, for the same
    # reason -- see that function's own docstring.
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
        owner may take no other action, which is how the engine models a mana
        ability being atomic (real Magic never interleaves anything between
        announcing one and its resolution). That exclusivity is correct for the
        owner and wrong for everyone else, and the raw slot cannot tell the
        difference -- it is global, with no stack. Anything asking "what may
        THIS seat legally do right now" must read this property; only the
        execute_* handlers, which by construction run as the owner, may touch
        the raw slot.

        Without this, a subdecision the defender left open silently governed
        the attacker's next decision instead (2026-08-16: an all-False action
        mask during assign_combat_damage, which is a hard crash -- see
        mana_subdecision's own comment in __init__)."""
        sub = self.mana_subdecision
        if sub is None or sub.get("owner") != self.active_idx:
            return None
        return sub

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

    def mint_iid(self):
        """Next never-reused per-game instance id. The single source of iids
        for both new_instance (non-battlefield) and new_permanent (board), so
        every physical card in a game has a distinct one."""
        iid = self._next_iid
        self._next_iid += 1
        return iid

    def new_instance(self, card_def):
        """Mint a fresh CardInstance for `card_def` -- the ONE place a
        non-battlefield card instance is born. Reached via move_card whenever a
        card ENTERS a tracked non-battlefield zone (the graveyard, today) as a
        new object, so a card that changes zones is genuinely a new object
        (MTG 400.7) and stale targets on it fizzle."""
        return CardInstance(card_def, self.mint_iid())

    def new_permanent(self, card_def, tapped=False):
        """Battlefield counterpart of new_instance: mint a fresh Permanent with
        the next iid. game.effects.casting.enters_battlefield uses this so board
        permanents share the same per-game identity counter as graveyard cards."""
        return Permanent(card_def, tapped=tapped, iid=self.mint_iid())

    def move_card(self, card, destination):
        """THE choke point for a card entering a tracked non-battlefield zone as
        a NEW object. Mints a fresh CardInstance (new iid) from `card`'s
        underlying CardDef into `destination` (a zone list) and returns it.
        `card` may be a CardDef or an existing CardInstance/Permanent (its
        .card_def is used) -- the source object is NOT reused, so any target
        captured on the old object is now stale and fizzles (MTG 400.7).

        Removal from the SOURCE zone stays at the existing removal sites
        (state-based death, stack resolution, hand discard): those already drop
        the old object, which is what makes a board target fizzle. This helper
        owns only the minting half. Battlefield entry has its own minting choke
        point, enters_battlefield (a Permanent, for its ETB/slot logic).

        Today `destination` is always a graveyard list; hand/library/exile are
        DEFERRED (still CardDef-based)."""
        card_def = card.card_def if isinstance(card, CardInstance) else card
        inst = self.new_instance(card_def)
        destination.append(inst)
        return inst

    def log_event(self, kind, **fields):
        """Appends one structured event to self.event_log, if logging is on
        (see __init__'s own event_log docstring) -- else an immediate
        no-op, so every instrumented call site across mana.py/turn.py/
        resolution/*.py/game/effects/*.py costs exactly one attribute check
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
        # Sanitize field values AT THE SOURCE so no event ever carries a raw
        # engine object -- a CardDef, or a card-effect closure/lambda -- which
        # would break the JSON write (rl.league_runner._json_default) or an MP worker's
        # pickle (rl.rollout_parallel._sanitize_events). Those two remain only as backstops.
        # Runs only when logging is ON (the early-return above), so the default
        # training path pays nothing. Primitives pass through; an object becomes
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

