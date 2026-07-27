"""Turn loop: a fixed sequence of phases (game.turn.Phase), each running its
own turn-based automatic effect (if any) then a real priority round
 -- both players get a real chance to act or
respond at every phase/step, not just the turn player, except Untap (never
any priority) and Cleanup (priority only if something triggers there)."""

import enum

from . import registry
from .effects.combat import attackers_needing_damage_assignment, combat_damage_step, declare_attackers_step, enforce_menace
from .effects.stack import resolve_top_of_stack
from .effects.state_based import check_state_based_actions, cleanup_step
from .effects.triggers import promote_triggers_to_stack
from .resolution import begin_assign_combat_damage, begin_declare_blockers, begin_mulligan, complete_resolution
from .state import DeckedOut, new_multiplayer_game_state


class Phase(enum.Enum):
    """One full turn's phases, in order. Phases gate three things: (1)
    each phase's own turn-based automatic effect below (if any) plus,
    for DECLARE_BLOCKERS specifically, the defending player's own
    block-assignment decision (_declare_blockers_gen) -- both run BEFORE
    that phase's own real priority round; (2)
    what "Pass" advances past (empty stack, everyone's passed in a row);
    and (3) via Speed/speed_legal below, which top-level actions
    (Cast/Activate/Play land) are legal at all -- a Speed.SORCERY action
    is only legal during MAIN1/MAIN2 of your OWN turn, anything else
    stays legal in every phase regardless of whose turn it is. Real-Magic
    upkeep *trigger* timing is intentionally not modeled -- no card in
    this repo currently cares (see the gap analysis this design followed
    from), so UPKEEP has no automatic effect at all yet, just a seam for
    one later. END's own effect (cleanup_step: discard to hand size,
    clear combat damage) is real, not a placeholder."""
    UNTAP = "untap"
    UPKEEP = "upkeep"
    DRAW = "draw"
    MAIN1 = "main1"
    DECLARE_ATTACKERS = "declare_attackers"
    DECLARE_BLOCKERS = "declare_blockers"
    COMBAT_DAMAGE = "combat_damage"
    MAIN2 = "main2"
    END = "end"


class Speed(enum.Enum):
    """When a top-level action (cast a spell, activate an ability, play a
    land) is legal relative to phase -- real Magic's own casting-speed
    rules, deliberately without the stack (see Phase's own docstring on
    what's still not modeled). YOUR_TURN and INSTANT no longer behave
    identically now that real priority means the
    non-turn player can genuinely hold priority mid-someone-else's-turn:
    INSTANT stays legal regardless of whose turn it structurally is (real
    Magic's whole point of the keyword); YOUR_TURN requires
    state.turn_player_idx specifically, with no phase/stack restriction of
    its own (unlike SORCERY, which is both your-turn-only AND main-phase/
    empty-stack-only). No card currently sets `"speed": Speed.YOUR_TURN`
    in its own registry entry -- every non-instant card falls through to
    SORCERY, the default (see drl_env._cast_speed) -- so this is a real,
    already-correct mechanism waiting for its first user, not dead code.

    Only ever checked for a top-level action that WOULD initiate a
    resolution (Cast/Activate/Play land/Plot) -- never for a
    pending-resolution continuation (Choose/Keep/Dispose/Decline/Abandon
    payment/Cast or Decline (madness), mana taps). Those stay governed
    purely by pending_resolution, same as always: Pass is already illegal
    whenever one is open (see drl_env._pass_legal), so a phase can never
    advance out from under a resolution in progress, and nothing mid-
    resolution needs its own timing check."""
    SORCERY = "sorcery"
    YOUR_TURN = "your_turn"
    INSTANT = "instant"


# Where a Speed.SORCERY action is legal -- both main phases, matching real
# Magic's own "any time you could cast a sorcery" (main phase, empty
# stack) minus the stack half. A deck whose own phase sequence never
# includes MAIN2 (combat_enabled=False -- MINIMAL_PHASES) needs no special
# case: state.phase simply never equals Phase.MAIN2 for it, so this
# degrades to "MAIN1 only" for free.
SORCERY_SPEED_PHASES = {Phase.MAIN1, Phase.MAIN2}


def speed_legal(state, speed):
    """The one gate every timing-restricted legal_fn in drl_env.py calls
    into.

    Real Magic's sorcery-speed rule is "your main phase, empty stack, you
    have priority" -- ALL THREE conditions, not just the phase/stack half:
    under real priority, the non-turn player can
    hold priority during the turn player's own MAIN1/MAIN2 (state.phase is
    a single shared field, describing the TURN's phase, not "whichever
    player is currently being asked"), and must not be allowed to play a
    land or cast a sorcery just because state.phase happens to match --
    that's a your-own-turn-only privilege, checked via
    state.turn_player_idx (game.drl_env._land_drop_legal also gates on
    Speed.SORCERY and needs the identical check for the same reason).
    YOUR_TURN carries the turn-ownership restriction alone, with no
    phase/stack restriction of its own -- no card uses it yet (see this
    enum's own docstring)."""
    if speed is Speed.SORCERY:
        return (
            state.active_idx == state.turn_player_idx
            and state.phase in SORCERY_SPEED_PHASES
            and not state.stack
        )
    if speed is Speed.YOUR_TURN:
        return state.active_idx == state.turn_player_idx
    return True


# Every deck with combat_enabled=True (rakdos madness / mono red madness /
# boggles) gets the full turn; everything else collapses to just the
# phases that ever do anything for a deck with no combat step at all --
# UNTAP (new-turn triggers, summoning sickness clears), DRAW (the turn's
# card), MAIN1 (every spell/ability -- no phase gating means a second main
# phase would add nothing without a combat phase to sandwich), END (horizon
# check). Not skipped via a forced Pass each -- these phases are simply
# never in the sequence a non-combat deck's generator iterates.
FULL_PHASES = tuple(Phase)
MINIMAL_PHASES = (Phase.UNTAP, Phase.DRAW, Phase.MAIN1, Phase.END)

# Per-phase cap on model-action loop iterations, replacing the old single
# MAX_MAIN_PHASE_ACTIONS -- guards against an infinite policy loop, not
# expected. A single "logical" action (cast a spell, activate an ability)
# can cost multiple loop iterations to fully resolve (one per mana tap,
# plus any search/scry/take decisions) -- and Speed.INSTANT actions/
# activated abilities (unrestricted by phase) can still bring that full
# complexity to any phase, not just the main phases (only Speed.SORCERY
# actions are confined to MAIN1/MAIN2). Kept uniform at 200 (matching the
# old constant) for every phase for now, precisely to avoid capping a
# non-main phase low enough to truncate a resolution mid-flight; trim
# per-phase once real training runs show what each phase actually needs.
PHASE_ACTION_CAPS = {phase: 200 for phase in Phase}

# Safety cap on ONE priority round's own inner loop (_run_priority_round_gen
# below) -- same "guard against an infinite policy loop, not expected"
# reasoning as PHASE_ACTION_CAPS, just scoped to a single round (bounded by
# how many times priority could plausibly pass back and forth, and by how
# many stack items could plausibly need resolving, before real resource
# limits -- mana, cards in hand -- stop a policy cold) rather than a whole
# phase, since general priority round can now run
# more than once per phase (once per stack resolution).
#
# ponytail: dropped from 200 to 20 as a temporary stopgap while
# investigating a suspected deterministic-eval stall (a tap-a-source ->
# Abandon payment loop: abandoning fully reverses the tap/pool delta, so a
# deterministic policy facing the identical resulting observation could
# keep re-choosing the same doomed payment attempt, burning the old
# 200-iteration budget almost silently every phase). 4 was tried first and
# was too aggressive -- turn.py's own regression self-check failed casting
# a single Lightning Bolt (tap + spend + both-players-pass-to-resolve
# already costs 4 by itself). 20 leaves room for a couple of real casts
# plus combat per phase while still bounding a runaway loop far below 200.
# Revisit once the loop's actual root cause is confirmed: either raise this
# back toward 200 once fixed properly, or replace it with a smarter "no
# observable progress" detector instead of a blunt iteration count.
PRIORITY_ROUND_ACTION_CAP = 20


def untap_step(state):
    untapped = [(p.card_def.name, p.slot) for p in state.battlefield if p.tapped]
    mana_cleared = dict(state.mana_pool)
    for permanent in state.battlefield:
        permanent.tapped = False
        permanent.summoning_sick = False
        permanent.flags.pop("used_this_turn", None)  # Barrels of Blasting Jelly
        # "doesn't untap during its controller's next untap step" (Sleep of
        # the Dead): skip this permanent's untap ONCE, consuming the flag.
        if permanent.flags.pop("skip_next_untap", False):
            permanent.tapped = True
    state.mana_pool.clear()  # floating mana doesn't carry across turns
    # The Initiative's "until your next turn" durations (Arena's Goad, Throne's
    # hexproof) expire at their owning player's turn start. Lazy import:
    # undercity pulls in casting/tokens, loaded after turn.py.
    from .effects import undercity
    undercity.expire_until_next_turn(state)
    # Impulse cards ("play until end of [this/your next] turn") whose deadline
    # has passed expire now -- they leave the impulse zone (ceasing, untracked)
    # and can no longer be played. turn_number is already this player's current
    # turn here, so an entry with deadline < turn_number is past its window.
    expired = [(cd.name, u) for (cd, u) in state.impulse if state.turn_number > u]
    if expired:
        state.impulse = [(cd, u) for (cd, u) in state.impulse if state.turn_number <= u]
        state.log_event("impulse_expired", cards=[n for n, _u in expired])
    if untapped or mana_cleared:
        state.log_event("untap_step", untapped=untapped, mana_cleared=mana_cleared)


def upkeep_step(state):
    """"At the beginning of your upkeep, ..." triggers (Delver of Secrets).
    Queues an "upkeep" trigger for each of the TURN player's own battlefield
    permanents whose registry has an "upkeep_trigger" -- the priority round
    that follows this phase's auto-effect then promotes them onto the stack
    (real timing: upkeep triggers go on the stack, get a priority window).
    A cheap no-op when nothing has one."""
    for permanent in state.battlefield:  # active_idx == turn player at UPKEEP
        if registry.EFFECT_REGISTRY.get(permanent.card_def.effect_id, {}).get("upkeep_trigger") is not None:
            state.trigger_queue.append({"type": "upkeep", "card_def": permanent.card_def, "permanent": permanent})
    # The Initiative: "at the beginning of your upkeep, venture into Undercity"
    # (only the current holder). Queued like the card upkeep triggers above.
    if state.initiative_idx == state.turn_player_idx:
        from .effects import undercity
        undercity.queue_venture(state, state.turn_player_idx)


def draw_step(state):
    # Checked against this player's OWN turn count (turns_taken), not the
    # game's global turn_number: once a second player also takes turns
    #, turn_number==1 no longer means "my
    # first turn" -- P2's first turn is turn_number==2. In 1-player mode
    # turns_taken tracks turn_number exactly (there's only one player), so
    # this is behaviorally identical to the old check there.
    if state.turns_taken == 1 and state.on_the_play:
        return
    state.draw(1)


_PHASE_AUTO_EFFECTS = {
    Phase.UNTAP: untap_step,
    Phase.UPKEEP: upkeep_step,
    Phase.DRAW: draw_step,
    Phase.DECLARE_ATTACKERS: declare_attackers_step,
    Phase.COMBAT_DAMAGE: combat_damage_step,
    Phase.END: cleanup_step,
}


def _run_mulligan_gen(state):
    """Pregame: every player decides keep-or-mulligan for their own opening
    hand (already dealt by state.new_game_state/new_multiplayer_game_state's
    own eager draw(7)), one player fully at a time -- same per-player
    active_idx flip pattern as _declare_blockers_gen, just scoped to the
    whole pregame instead of one phase. APNAP order: whoever active_idx
    already points at (the real starting player) goes first. Runs entirely
    before turn 1 (state.turn_number is still 0, state.phase is still None)
    -- nothing here touches any turn-scoped field, so it's driven by
    run_mulligan_phase below rather than folded into _run_turn_gen."""
    starting_idx = state.active_idx
    order = [starting_idx] + [i for i in range(len(state.players)) if i != starting_idx]
    for idx in order:
        state.active_idx = idx
        begin_mulligan(state, on_complete=lambda s: None)
        while state.pending_resolution is not None:
            action = yield
            state.players[state.active_idx].actions_taken += 1
            action()  # keep/mulligan/bottom -- None (Pass) is never expected, same as _declare_blockers_gen
    state.active_idx = starting_idx


def run_mulligan_phase(state, choose_action):
    """Synchronous driver for _run_mulligan_gen -- same run_turn/
    _run_turn_gen pairing shape. Called once, by run_multiplayer_game
    below, before its own turn loop starts."""
    gen = _run_mulligan_gen(state)
    try:
        next(gen)
        while True:
            gen.send(choose_action(state))
    except StopIteration:
        pass


def _declare_blockers_gen(state):
    """The defending player's own block-assignment decision
   , yielded through the SAME generic decision
    protocol as everything else in this generator-based turn loop --
    folded directly into _run_turn_gen's own Phase.DECLARE_BLOCKERS
    handling, replacing the old
    run_declare_blockers (which took a defender_choose_action callback
    and drove its own separate while loop, entirely outside the main
    generator's yield protocol). Declaring blockers is a turn-based
    special action belonging to the DEFENDER specifically, not the turn
    player -- real Magic's own rule -- so state.active_idx is temporarily
    flipped to them for its scope, the same hidden-information fix this
    mechanism has always depended on: state.hand/state.battlefield only
    mean the defender's OWN zones once this flip has happened, not the
    attacker's.

    Flips back to the attacker once the defender is done (0 or more
    assignments), before this phase's own regular priority round runs
    next (attacker gets it first, per rule 1 -- "combat
    tricks are only asymmetrically supportable" limitation no longer
    applies once that round exists: the defender's own consult is no
    longer the only response window either side gets).

    No-op (no yield at all) if there's no real opponent to consult
    (len(state.players) < 2) -- matches every 1-player caller exactly as
    before blocking existed.

    Bounded at PRIORITY_ROUND_ACTION_CAP iterations, same constant and same
    reasoning _run_priority_round_gen's own cap-exhaustion path already
    documents (a deterministic or barely-trained policy can keep re-issuing
    the same unproductive action forever) -- confirmed live via
    boggles_mirror evaluation: with no cap at all here (unlike that other
    loop), a defender stuck re-assigning the same blocker assignment spun
    for tens of thousands of iterations, turn_number never advancing.
    Exhausting the cap force-completes with whatever's already been
    assigned (complete_resolution, same "can't finish, so the attempt ends
    outright" precedent mana.abandon_pay_cost and the priority round's own
    cap-exhaustion already use) rather than leaving declare_blockers open
    forever."""
    if len(state.players) < 2:
        return
    attacker_idx = state.active_idx
    state.active_idx = 1 - attacker_idx
    state.log_event("priority_flip", reason="declare_blockers", to_idx=state.active_idx)
    try:
        begin_declare_blockers(state, on_complete=lambda s: None)
        for _ in range(PRIORITY_ROUND_ACTION_CAP):
            if state.pending_resolution is None:
                return
            action = yield
            state.players[state.active_idx].actions_taken += 1
            action()  # "Done" is its own explicit action here -- None (Pass) is never expected, same as before
        if state.pending_resolution is not None:
            # Cap exhausted mid-blocking. The pending resolution here may be
            # the top-level "declare_blockers" (on_complete takes only
            # state) OR a NESTED "choose which attacker to block"
            # (begin_choose_opponent_permanent, whose on_complete is
            # _on_attacker_chosen(s, choice) and REQUIRES a choice arg) --
            # complete_resolution(state) with no payload crashes the latter
            # (confirmed live: a barely-trained defender looped on
            # Assign-Blocker and hit the cap with the attacker-choice still
            # open). Abandon outright instead: keep whatever blocks are
            # already recorded in blocked_by, drop the in-progress one, end
            # blocking regardless of nesting depth. Same "can't finish, so
            # the attempt ends" precedent abandon_pay_cost / the priority
            # round's own cap-exhaustion already use.
            state.log_event("declare_blockers_cap_abandoned", pending_kind=state.pending_resolution["kind"])
            state.pending_resolution = None
    finally:
        state.active_idx = attacker_idx
        state.log_event("priority_flip", reason="declare_blockers_done", to_idx=state.active_idx)


def _assign_combat_damage_gen(state):
    """After blockers are declared (active_idx already back to the ATTACKER,
    _declare_blockers_gen's finally): for each attacker blocked by 2+
    creatures, the attacking player freely assigns that attacker's combat
    damage across its blockers (+ trample) -- one point at a time, same
    generic yield protocol as every other in-turn decision. A lone blocker
    (or 0-power attacker) needs no decision (combat_damage_step auto-
    assigns). These forced sub-resolution picks are deliberately NOT counted
    toward actions_taken -- like resolving a cost or the automatic draw,
    they're a mechanical consequence of a multi-block, not a discretionary
    action the action-count reward should penalize.

    Cap-bounded purely as defense: unlike blocking's own no-op trap, each
    pick strictly decrements the pending's `remaining`, so this always
    finishes in exactly `power` picks -- it cannot loop. On the (unreachable)
    cap exhaustion, abandon and let combat_damage_step auto-assign."""
    for attacker, blockers, power, has_trample in attackers_needing_damage_assignment(state):
        begin_assign_combat_damage(state, attacker, blockers, power, has_trample, on_complete=lambda s: None)
        for _ in range(PRIORITY_ROUND_ACTION_CAP):
            if state.pending_resolution is None:
                break
            action = yield
            action()
        else:
            state.pending_resolution = None  # cap defense only; fall back to auto-assign


def _run_priority_round_gen(state):
    """One or more rounds of real priority-passing, run at the start of every phase/step (after its own
    turn-based actions, see _run_turn_gen) and repeated after every single
    stack resolution.

    Starts with priority at state.turn_player_idx (rule 1). Before each
    consultation: state-based actions are checked and any newly-queued
    triggers are promoted onto the stack (real Magic 704.3's actual
    ordering -- SBAs, then triggers move to the stack, THEN priority is
    given) -- cheap no-ops when there's nothing to do, so unconditional
    every time is simpler and more rules-accurate than trying to detect
    "did anything change" by hand.

    Whoever currently holds priority (state.active_idx) either acts
    (yields once, gets back a zero-arg callable -- the stack grows,
    priority stays with them, rule 2, and "holding priority" falls out
    for free) or passes (yields once, gets back None -- priority moves to
    the other player, own 2-player-only scope).
    Once every player has passed in a row: if the stack is non-empty,
    its top item resolves and priority resets to turn_player_idx (rule 1)
    -- the round repeats; if the stack is empty, this generator ends and
    the phase/step can advance.

    Never called at all for Phase.UNTAP (rule 4 -- see _run_turn_gen).
    Phase.END calls this only conditionally (see its own handling) --
    "usually none during Cleanup" is enforced there, not here; once this
    generator IS entered for Cleanup, it behaves identically to every
    other phase."""
    state.active_idx = state.turn_player_idx
    consecutive_passes = 0
    for _ in range(PRIORITY_ROUND_ACTION_CAP):
        check_state_based_actions(state)
        # Move queued triggers (ETBs, cast triggers, Madness decisions,
        # Sneaky Snacker returns) onto the stack ONLY at a genuine priority
        # point -- never while a resolution is still in progress (real Magic
        # 704.3/603.3: triggered abilities are put on the stack the next time
        # a player WOULD receive priority, which is not mid-cost/mid-choice).
        # A pending_resolution means we're mid-action -- paying a cost,
        # locking a target, walking a discard -- and promoting then would
        # (a) land a cast/ETB trigger UNDER the spell or effect that
        # resolution hasn't pushed yet (wrong stack order), and (b) for 2+
        # simultaneous triggers, open begin_order_triggers' own pending right
        # on top of the one already in progress, clobbering it. Deferring to
        # the next pending-free iteration is both correct and safe: this loop
        # runs promote every iteration, so a trigger queued mid-resolution is
        # picked up the instant that resolution clears.
        if state.pending_resolution is None:
            promote_triggers_to_stack(state)
        action = yield
        if action is None:
            state.log_event("pass")
            consecutive_passes += 1
            if consecutive_passes >= len(state.players):
                if state.stack:
                    resolve_top_of_stack(state)
                    # Rule 1 (priority resets to the turn player) applies to
                    # handing out the NEXT priority window -- it does not
                    # apply if resolving the stack top just opened a fresh
                    # pending_resolution (e.g. a Madness cast-or-decline
                    # choice), which is the entry's own CONTROLLER's forced
                    # decision, not a priority window, and stays open past
                    # this yield. resolve_top_of_stack already restored
                    # state.active_idx to that controller (game/effects/
                    # stack.py); when the controller is the non-turn player
                    # (their own instant-speed trigger, resolving during the
                    # opponent's turn), stomping active_idx back to
                    # turn_player_idx here reassigns the decision -- and the
                    # zone reads (state.exile/state.hand) it makes -- to the
                    # wrong player. Confirmed live via mono_red_madness_mirror
                    # training: a Blood-token discard's Madness trigger,
                    # resolving on the opponent's turn, misattributed to
                    # turn_player_idx and crashing resolution._remove_one_
                    # from_exile ("should be unreachable" StopIteration) once
                    # decline was chosen against the wrong player's exile.
                    if state.pending_resolution is None:
                        state.active_idx = state.turn_player_idx
                    consecutive_passes = 0
                    continue
                # Stack empty, everyone passed -- the phase/step is over.
                # Reset priority to the turn player before returning, not
                # just when a stack item resolves above: the NEXT phase's
                # own turn-based auto_effect (and the audit invariant
                # _run_turn_gen's own docstring documents) both require
                # state.active_idx == state.turn_player_idx to already
                # hold by the time this generator's caller resumes --
                # otherwise the last player to merely PASS (not act) would
                # incorrectly still be "active" going into the next phase.
                state.active_idx = state.turn_player_idx
                return
            state.active_idx = 1 - state.active_idx  # the only other player, in a 2-player game
        else:
            state.players[state.active_idx].actions_taken += 1  # Pass itself doesn't count -- see PlayerState.actions_taken's own docstring
            action()
            consecutive_passes = 0  # the priority holder keeps priority (rule 2) -- state.active_idx unchanged

    # PRIORITY_ROUND_ACTION_CAP exhausted without ever reaching the clean
    # "everyone passed, stack empty" exit above -- same invariant that exit
    # already enforces before returning: state.active_idx must be back to
    # state.turn_player_idx by the time this generator ends, or whoever
    # last held priority (not necessarily the turn player) would incorrectly
    # stay "active" going into the next phase. Unreachable in practice at
    # the old cap of 200 (essentially never hit); became reachable once the
    # cap dropped low enough to actually bind during real multi-action
    # turns -- found via turn.py's own regression self-check, not guessed.
    state.active_idx = state.turn_player_idx
    # A pending_resolution can ALSO still be open here (e.g. a deterministic
    # or barely-trained policy oscillating tap-a-source -> Abandon payment
    # for 20 straight iterations, confirmed live via boggles_mirror
    # training: Ash Barrens' landcycling stuck this way, its still-open
    # pay_cost silently surviving into later phases/turns until it finally
    # completed with the card long gone from hand -- discard_from_hand_to_
    # graveyard's own "should be unreachable" RuntimeError). Every other
    # exit from this loop guarantees pending_resolution is None (the clean
    # exit above only returns once the stack's empty AND no cost/choice is
    # outstanding); this is the one path that doesn't, so it has to drop it
    # itself rather than let it leak across a phase boundary no caller
    # expects. ponytail: a dropped pay_cost's own taps/floated pool mana
    # are simply left as an over-tap (already a normal, safe state
    # elsewhere in this engine) rather than precisely reversed via
    # mana.abandon_pay_cost -- upgrade to that (or a smarter "no observable
    # progress" detector, per PRIORITY_ROUND_ACTION_CAP's own note above)
    # if leaving lands tapped for nothing turns out to matter.
    state.pending_resolution = None


def _run_turn_gen(state, combat_enabled=False):
    """Generator form of one full turn -- the single implementation shared
    by run_turn's synchronous choose_action loop below and the token
    training pipeline's own per-seat driver (rl.train). Iterates FULL_PHASES or
    MINIMAL_PHASES depending on combat_enabled; for each phase, runs that
    phase's own turn-based automatic effect (if any), then a real
    priority round -- except Untap (never any
    priority at all, rule 4) and Cleanup (priority only if something
    newly triggered there, rule 4 -- see its own handling below).
    Phase.DECLARE_BLOCKERS additionally runs the defending player's own
    block-assignment decision (_declare_blockers_gen) BEFORE its own
    priority round, since that's a turn-based special action belonging to
    the defender, not a priority action itself.

    Every yield (from this generator OR the sub-generators it drives via
    `yield from`) uses the exact same protocol: the caller sends back
    either None ("pass") or a zero-arg callable, via gen.send(...) --
    this generator is completely agnostic to WHO answers a given yield;
    state.active_idx (whoever currently holds priority) tells the CALLER
    that, and dispatching accordingly is entirely the caller's own
    business (run_turn's plain choose_action(state) below; the token
    training loop's own fork between the two seats). Ends
    (StopIteration) once every phase has run its course.

    Wrapped in one try/except DeckedOut: a draw (this phase's own, or a
    card effect's, in any phase -- no phase gates casting) can raise
    DeckedOut from arbitrarily deep in a resolution chain (see
    state.GameState.draw's own docstring); catching it here, around the
    whole turn, ends the turn/generator immediately and uniformly,
    wherever it happened, with state.decked_out already set by draw()
    itself. Callers (run_turn, the training loop) never see DeckedOut, only
    the StopIteration this produces either way.

    combat_enabled: per-deck opt-in (default off, matching every other
    deck-specific knob here) -- only rakdos madness/mono red madness/
    boggles pass True. Phase.DECLARE_ATTACKERS is a real per-creature
    decision (declare_attackers_step/creature_attack_eligible/
    declare_attacker, game/effects/combat.py); Phase.COMBAT_DAMAGE totals
    unblocked attackers' power into the opponent's life_total."""
    try:
        # Whoever active_idx is right now, at the very start of this
        # generator, is the true turn owner for the whole turn -- callers
        # (run_turn/run_multiplayer_game) always invoke this with active_idx
        # already pointing at them, before any priority consult could ever
        # flip it away. Set once here, never touched again until next turn.
        state.turn_player_idx = state.active_idx
        state.turn_number += 1
        state.turns_taken += 1  # this player's own turn count -- see draw_step's own note on why turn_number alone isn't enough once a second player exists
        state.lands_played_this_turn = 0
        state.cards_drawn_this_turn = 0
        state.log_event("turn_start", turn_player_idx=state.turn_player_idx)

        phases = FULL_PHASES if combat_enabled else MINIMAL_PHASES
        for phase in phases:
            from_phase = state.phase
            state.phase = phase
            state.log_event("phase_change", from_phase=from_phase.value if from_phase is not None else None)
            auto_effect = _PHASE_AUTO_EFFECTS.get(phase)
            if auto_effect is not None:
                auto_effect(state)
            if state.turn_won is not None:
                return

            if phase is Phase.UNTAP:
                continue  # rule 4: no priority during Untap, full stop -- not even a check for it

            if phase is Phase.DECLARE_BLOCKERS:
                yield from _declare_blockers_gen(state)
                if state.turn_won is not None:
                    return
                # Menace: a menace attacker left with exactly one blocker is
                # unblocked (game.effects.combat.enforce_menace) before any
                # damage is assigned -- must run after blocks are finalized,
                # before _assign_combat_damage_gen reads them.
                enforce_menace(state)
                # A multi-blocked attacker's controller now freely assigns
                # that attacker's combat damage across its blockers (gang-
                # blocking) -- before COMBAT_DAMAGE's own combat_damage_step
                # auto-effect applies it.
                yield from _assign_combat_damage_gen(state)
                if state.turn_won is not None:
                    return

            if phase is Phase.END:
                # Rule 4: "usually none during Cleanup, unless something
                # triggers there" (real Magic 514.2/514.3). cleanup_step
                # (hand-size discard + damage clear) already ran once as
                # this phase's own auto_effect above -- drive its own
                # discard resolution to completion first (a turn-based
                # action for the active player alone, NOT a priority
                # round: the opponent never gets a window between
                # individual discard picks, same generic yield protocol
                # as everything else, just without any stack/pass-
                # counting semantics), then check whether anything got
                # queued (a discarded card with its own Madness trigger,
                # say). If so, a real priority round for it, then
                # cleanup_step repeats (matching the real rule instead of
                # a single hardcoded pass). No current card ever queues
                # anything during cleanup, so this loop always runs
                # exactly once in practice.
                while True:
                    for _ in range(PRIORITY_ROUND_ACTION_CAP):
                        if state.pending_resolution is None:
                            break
                        action = yield
                        state.players[state.active_idx].actions_taken += 1
                        action()
                    else:
                        # Same defensive cap _declare_blockers_gen now has,
                        # for the same reason (an uncapped yield loop here
                        # could spin forever the same way that one did) --
                        # naturally bounded by hand size in practice (a
                        # discard resolution can only ask for as many cards
                        # as are actually over the hand-size limit), so this
                        # is defense-in-depth rather than a confirmed-live
                        # bug like that one was.
                        if state.pending_resolution is not None:
                            complete_resolution(state)
                    if state.turn_won is not None:
                        return
                    if not state.trigger_queue:
                        break
                    yield from _run_priority_round_gen(state)
                    if state.turn_won is not None:
                        return
                    cleanup_step(state)
                    if state.turn_won is not None:
                        return
                continue

            yield from _run_priority_round_gen(state)
            if state.turn_won is not None:
                return
    except DeckedOut:
        # Real Magic: decking out is an instant loss for whoever draws
        # from an empty library. In a 2-player game the OTHER player wins
        # outright (state.active_idx is still whoever was drawing -- only
        # the active player ever draws); in 1-player there's no one to
        # award the win to, same bare-failure outcome (turn_won/winner
        # stay None) as before this player-count distinction existed.
        if len(state.players) > 1:
            state.turn_won = state.turn_number
            state.winner = 1 - state.active_idx
        return


def run_turn(state, choose_action, combat_enabled=False):
    """One full turn, pull-style: repeatedly calls choose_action(state)
    itself and feeds the result into _run_turn_gen. See that generator's
    docstring for the actual turn logic -- this is just its synchronous
    driver. choose_action(state) is called for EVERY yield regardless of
    whose decision it is -- a closure that needs
    to act differently per player reads state.active_idx itself (same
    contract run_multiplayer_game's own choose_action already relies on),
    no separate callback for blocking or any other reactive window needed
    anymore."""
    gen = _run_turn_gen(state, combat_enabled=combat_enabled)
    try:
        next(gen)  # advance to first yield (or StopIteration if the turn ended during a phase's own automatic effect)
        while True:
            gen.send(choose_action(state))
    except StopIteration:
        pass


def _yield_decisions(inner, state):
    """Adapt a choose_action-driven inner generator (_run_mulligan_gen /
    _run_turn_gen -- each yields at a decision and expects the chosen action
    back via .send()) into one that yields the live STATE outward and forwards
    the action in. The inner's own yielded value was always ignored by its
    synchronous driver (run_turn / run_mulligan_phase), so yielding `state`
    instead -- what a driver actually needs to choose on -- changes nothing
    about the decision sequence."""
    try:
        next(inner)
        while True:
            action = yield state
            inner.send(action)
    except StopIteration:
        pass


def game_coroutine(state, horizon=None, combat_enabled=False):
    """run_multiplayer_game's decision flow as a resumable generator: yields
    the state at every point a player must choose (pregame mulligan, then every
    turn's priority/combat decisions) and expects the chosen action back via
    .send() -- the SAME value choose_action returns (None for a Pass, or a
    zero-arg executor callable). run_multiplayer_game drives this synchronously
    (so every existing caller and self-check exercises this exact path); the
    batched rollout collector (rl.train.collect_rollout_batched) drives many of
    these at once, running ONE network forward across all their pending
    decisions instead of a batch-of-1 forward per decision. The turn loop here
    mirrors run_multiplayer_game's own former inline loop verbatim -- same
    horizon/turn_won/decked_out guard, same lazy active_idx flip."""
    yield from _yield_decisions(_run_mulligan_gen(state), state)
    first_turn = True
    while (horizon is None or state.turn_number < horizon) and state.turn_won is None and not state.decked_out:
        if not first_turn:
            state.active_idx = 1 - state.active_idx
        first_turn = False
        yield from _yield_decisions(_run_turn_gen(state, combat_enabled=combat_enabled), state)


def run_multiplayer_game(decklists, rng, starting_player_idx, choose_action,
                          horizon=None, combat_enabled=False, event_log=None):
    """N-player entry point. Full
    sequential turns -- one player's whole turn runs to completion (same
    run_turn/choose_action(state) contract as the 1-player path; a
    choose_action closure that needs to act differently per player can
    read state.active_idx itself, no separate callable per player needed)
    before active_idx flips to the next one. horizon=None (default) means
    uncapped: the loop instead ends only on an actual game-loss condition
    (state.turn_won, set by a life_total hitting 0 -- via
    game.effects.win_check._check_end_of_game -- or a decked-out draw).
    This can't hang: draw_step draws exactly one card
    every turn for whichever player is active, so total combined library
    size across every player is a hard upper bound on turns regardless of
    board state, independent of PHASE_ACTION_CAPS' own per-phase bound.
    Pass an int horizon to still cap it (e.g. for a bounded self-check).

    Flips active_idx lazily -- right before the NEXT turn starts, not
    right after the current one ends -- so state.active_idx always names
    whoever just played once this function returns, including on a
    horizon-capped exit (an eager flip would leave it pointing at a player
    who never actually got a turn, misattributing every state.hand/
    state.decked_out/etc. read a caller does on the returned state)."""
    state = new_multiplayer_game_state(decklists, starting_player_idx, rng, event_log=event_log)
    # Drive the game as a coroutine (game_coroutine) -- choose_action(state) is
    # still called for EVERY decision regardless of whose it is, exactly as the
    # former inline mulligan+turn loop did. The loop now lives in
    # game_coroutine (single source of truth) so the batched rollout collector
    # can interleave many games over the same generator; every existing caller
    # keeps this identical synchronous behavior.
    gen = game_coroutine(state, horizon=horizon, combat_enabled=combat_enabled)
    try:
        req_state = next(gen)
        while True:
            req_state = gen.send(choose_action(req_state))
    except StopIteration:
        pass
    return state


if __name__ == "__main__":
    # ponytail self-check: no pytest in this project, mirrors the
    # assert-based demo convention -- run via `python -m game.turn` from
    # src/. Exercises the 2-player engine
    # end to end: turn alternation, on_the_play/turns_taken, life_total
    # loss, deck-out-as-instant-loss, and a real game played through actual
    # cards (not a synthetic no-op deck) -- Mountain + Lightning Bolt is
    # the smallest real combination that can deal opponent damage.
    import random

    from . import mana, registry, resolution
    from .cards import CardDef, CardType, EffectId
    from .effects.casting import play_land_from_hand
    from .effects.stack import push_to_stack
    from .effects.win_check import deal_damage_to_opponent
    from .state import GameState, PlayerState

    # -- construction: opening hands, on_the_play, starting life ----------
    state = new_multiplayer_game_state(
        decklists=[[("Mountain", 20)], [("Mountain", 20)]],
        starting_player_idx=0,
        rng=random.Random(0),
    )
    assert len(state.players) == 2
    assert state.active_idx == 0
    assert state.players[0].on_the_play and not state.players[1].on_the_play
    assert len(state.players[0].hand) == 7 and len(state.players[1].hand) == 7
    assert state.players[0].life_total == 20 and state.players[1].life_total == 20
    print("turn.py 2-player construction self-check: OK")

    # -- turn alternation + on_the_play (pass-only policy) ----------------
    def _pass_except_discard(state):
        # Always Pass -- pure alternation, no card actions -- EXCEPT
        # cleanup_step's own hand-size discard is mandatory in real Magic
        # (the real mask-driven path, drl_env._pass_legal, already refuses
        # Pass while a resolution is pending; this hand-written policy has
        # to honor that too now that _run_turn_gen's Phase.END loop
        # actually drives the resolution to completion instead of
        # silently abandoning it on a stray Pass).
        if state.pending_resolution is not None:
            # run_multiplayer_game now runs the pregame mulligan phase
            # (game.turn.run_mulligan_phase) before turn 1 -- always keep
            # (0 mulligans taken), same net opening-hand outcome this
            # policy already had before mulligans existed.
            if state.pending_resolution["kind"] == "mulligan_decision":
                return lambda: resolution.execute_mulligan_keep(state)
            name = resolution.discard_options(state)[0]
            return lambda: resolution.execute_discard_option(state, name)
        return None

    state = run_multiplayer_game(
        decklists=[[("Mountain", 20)], [("Mountain", 20)]],
        rng=random.Random(0),
        starting_player_idx=0,
        choose_action=_pass_except_discard,
        horizon=6,
    )
    assert state.turn_number == 6
    assert state.active_idx == 1  # player 1 played turn 6 last -- lazy-flip means this must still say so, not have advanced past it
    assert state.players[0].turns_taken == 3 and state.players[1].turns_taken == 3  # turns 1/3/5 vs 2/4/6
    # Player 0 (on the play) skipped their very first draw; player 1 never
    # does -- both then draw once per turn after that, and cleanup_step
    # discards back down to HAND_SIZE_LIMIT (7) every time a draw pushes
    # past it, so both settle back at exactly 7.
    assert len(state.players[0].hand) == 7
    assert len(state.players[1].hand) == 7
    assert state.turn_won is None and state.winner is None  # horizon reached, no win condition ever fired
    print("turn.py turn-alternation self-check: OK")

    # -- life_total loss (direct deal_damage_to_opponent call) ------------
    state = GameState(
        on_the_play=True,
        players=[PlayerState(on_the_play=True), PlayerState(on_the_play=False)],
    )
    assert state.active_idx == 0
    deal_damage_to_opponent(state, 15)
    assert state.players[1].life_total == 5  # 20 - 15
    assert state.turn_won is None  # not lethal yet
    deal_damage_to_opponent(state, 5)
    assert state.players[1].life_total == 0
    assert state.turn_won == state.turn_number and state.winner == 0  # opponent's life hit 0 -- active player (0) wins
    print("turn.py life-total-loss self-check: OK")

    # -- deck-out is an instant loss for the drawing player, in 2p --------
    # on_the_play=False for the drawing player specifically so draw_step's
    # skip-first-draw rule never applies here -- keeps this check about
    # deck-out alone, not entangled with the on_the_play interaction
    # already covered by the turn-alternation check above.
    state = GameState(
        on_the_play=False,
        players=[PlayerState(on_the_play=False), PlayerState(on_the_play=True)],
    )
    state.active_idx = 0
    state.players[0].library = []  # empty -- draw_step will immediately deck this player out
    state.players[1].library = [CardDef("Mountain", CardType.LAND, None, EffectId.MOUNTAIN)]

    run_turn(state, lambda state: None, combat_enabled=False)  # UNTAP has no auto-effect; DRAW's draw_step raises DeckedOut immediately
    assert state.players[0].decked_out
    assert state.turn_won == state.turn_number and state.winner == 1  # the OTHER player wins outright
    print("turn.py deck-out self-check: OK")

    # -- a real 2-player game, played through actual cards -----------------
    # Player 0: Mountain + Lightning Bolt (real red_cards.py cards) racing
    # to burn player 1's life_total to 0. Player 1: Mountain only, never
    # acts (always passes) -- a pure punching bag, proving damage actually
    # lands on a REAL opponent PlayerState through the normal cast/mana-
    # payment path (game.mana.begin_pay_cost + push_to_stack), not a
    # direct deal_damage_to_opponent call like the unit check above.
    bolt_def = registry.CARD_DEFS["Lightning Bolt"]

    # This priority/stack game test uses Lightning Bolt purely as a burn
    # vehicle to prove damage lands on a REAL opponent through the normal
    # cast/mana-payment/stack path. Its production "cast" resolve is now a
    # precast target choice ("any target", game.catalog.red_cards -- tested
    # there), which would open a choose_any_target pending this toy policy
    # doesn't model. Use a local direct-to-opponent resolve so this stays a
    # turn/priority/stack test, decoupled from targeting.
    def bolt_resolve(s, cd):
        if cd in s.hand:
            s.hand.remove(cd)
        s.graveyard.append(cd)
        deal_damage_to_opponent(s, 3)

    def _bolts_available(state):
        # Lightning Bolt is instant-speed (CardType.INSTANT), so it stays
        # legal to "cast" again even with a copy already pushed, paid-for-
        # but-unresolved on the stack -- a copy already there is still
        # physically in state.hand (its resolve only removes it once it
        # actually resolves; see game.effects.stack.push_to_stack) but isn't
        # really available. Same accounting drl_env._hand_count_available
        # does for exactly this reason.
        hand_count = sum(1 for c in state.hand if c.name == "Lightning Bolt")
        stacked_count = sum(1 for entry in state.stack if entry["card_def"].name == "Lightning Bolt")
        return hand_count - stacked_count

    def _burn_policy(state):
        if state.pending_resolution is not None and state.pending_resolution["kind"] == "mulligan_decision":
            # run_multiplayer_game now runs the pregame mulligan phase for
            # BOTH players first -- always keep (0 mulligans taken), same
            # net opening-hand outcome this policy already had before
            # mulligans existed.
            return lambda: resolution.execute_mulligan_keep(state)
        if state.pending_resolution is not None and state.pending_resolution["kind"] == "discard":
            # cleanup_step's own hand-size discard
            # can now happen to EITHER player at the end of THEIR OWN
            # turn -- unlike every other resolution below (which only
            # ever belongs to player 0, the only one who ever casts/taps
            # anything), so this has to be checked before the "player 1
            # never acts" rule, not folded inside it. An arbitrary card is
            # fine, this policy has no preference among its own
            # Mountains/Bolts either way.
            name = resolution.discard_options(state)[0]
            return lambda: resolution.execute_discard_option(state, name)
        if state.active_idx != 0:
            return None  # player 1 never acts otherwise
        if state.pending_resolution is not None:
            # Only ever a pay_cost here now (paying Lightning Bolt's {R}) --
            # discard (above) is handled regardless of whose turn it is.
            # Pool-only model: a tap only floats mana
            # into the pool, so tap first if there's still an untapped
            # Mountain, then spend the floated R from the pool.
            tap_opts = mana.tap_cost_options(state)
            if tap_opts:
                name, color, is_filter = tap_opts[0]
                return lambda: mana.execute_tap_cost_option(state, name, color, is_filter)
            color = mana.pool_spend_options(state)[0]
            return lambda: mana.execute_pool_spend(state, color)
        if state.lands_played_this_turn == 0 and any(c.name == "Mountain" for c in state.hand):
            return lambda: play_land_from_hand(state, registry.CARD_DEFS["Mountain"])
        if _bolts_available(state) > 0 and mana.plan_payment(state, bolt_def.cast_cost) is not None:
            def _cast_bolt():
                mana.begin_pay_cost(state, bolt_def.cast_cost, on_complete=lambda s: push_to_stack(s, bolt_def, bolt_resolve))
            return _cast_bolt
        return None  # Pass -- resolves the stack if non-empty, else advances the phase

    state = run_multiplayer_game(
        decklists=[[("Mountain", 20), ("Lightning Bolt", 10)], [("Mountain", 20)]],
        rng=random.Random(0),
        starting_player_idx=0,
        choose_action=_burn_policy,
        horizon=60,  # safety cap only -- the game is expected to end well before this via life_total, not by hitting it
    )
    assert state.winner == 0
    assert state.players[1].life_total <= 0  # burned out (10 Bolts * 3 available; only ~7 needed for lethal)
    assert state.turn_won is not None and state.turn_won < 60  # ended on life_total, not the safety cap
    print(f"turn.py real 2-player game self-check: OK (player 1 dead on turn {state.turn_won}, "
          f"life_total={state.players[1].life_total})")

    # -- _declare_blockers_gen: the defender's own consult, active_idx flip
    # --.
    # resolution.py's own self-check already covers begin_declare_blockers/
    # declare_blocker_assignment directly; this one is specifically about
    # the flip-drive-restore shape unique to this generator, driven the
    # same way drl_env._assign_blocker_execute's own nested on_complete
    # re-opens begin_declare_blockers after each assignment -- now via the
    # generator's own yield protocol (like every other decision point in
    # this file) instead of a directly-invoked callback.
    from .state import Permanent

    bear = Permanent(CardDef("Bear", CardType.CREATURE, None, None))
    grizzly = Permanent(CardDef("Grizzly Bears", CardType.CREATURE, None, None))
    state = GameState(on_the_play=True, players=[PlayerState(True), PlayerState(False)])
    state.players[0].battlefield = [bear]
    state.players[1].battlefield = [grizzly]
    state.players[0].attackers = [bear]  # declared, bypassing declare_attackers_step itself
    state.active_idx = 0  # the attacker's own turn, before the consult ever starts

    active_idx_during_consult = []

    def _defender_policy(state):
        active_idx_during_consult.append(state.active_idx)
        pending = state.pending_resolution
        if pending["kind"] == "declare_blockers":
            if state.players[0].blocked_by:
                return lambda: resolution.complete_resolution(state)  # "Done blocking" -- already assigned my one blocker
            outer_on_complete = pending["on_complete"]
            return lambda: resolution.declare_blocker_assignment(
                state, grizzly, on_complete=lambda s: resolution.begin_declare_blockers(s, outer_on_complete),
            )
        name, slot = resolution.choose_opponent_permanent_options(state)[0]
        return lambda: resolution.execute_choose_opponent_permanent_option(state, name, slot)

    gen = _declare_blockers_gen(state)
    try:
        next(gen)
        while True:
            gen.send(_defender_policy(state))
    except StopIteration:
        pass
    assert state.active_idx == 0  # flipped back to the attacker once the consult is done
    assert active_idx_during_consult and all(idx == 1 for idx in active_idx_during_consult)  # the defender's own seat throughout
    assert state.players[0].blocked_by == {bear: [grizzly]}
    assert state.pending_resolution is None

    # No-op guard: fewer than 2 players -- never starts a resolution at
    # all, let alone leaves one stuck open (no next(gen) call ever
    # reaches a yield -- StopIteration on the very first one).
    state_1p = GameState(on_the_play=True)  # len(state.players) == 1
    gen_1p = _declare_blockers_gen(state_1p)
    try:
        next(gen_1p)
    except StopIteration:
        pass
    assert state_1p.pending_resolution is None

    print("turn.py _declare_blockers_gen self-check: OK")

    # Turn-owner / priority-holder split:
    # speed_legal's Speed.SORCERY (and the new Speed.YOUR_TURN) branches
    # must refuse the non-turn player even when state.phase/state.stack
    # alone would otherwise say "legal" -- state.phase is a single shared
    # field describing the TURN's phase, not whichever player is currently
    # being asked, so this can't be caught by the phase/stack checks
    # alone. Simulates a priority consult (active_idx flipped away from
    # turn_player_idx) the same way _declare_blockers_gen's own flip does,
    # without needing the full priority round built yet.
    state = GameState(on_the_play=True, players=[PlayerState(True), PlayerState(False)])
    state.phase = Phase.MAIN1
    state.turn_player_idx = 0
    state.active_idx = 0
    assert speed_legal(state, Speed.SORCERY)  # the turn player's own MAIN1, empty stack -- legal
    assert speed_legal(state, Speed.YOUR_TURN)
    assert speed_legal(state, Speed.INSTANT)  # always legal, regardless of turn ownership

    state.active_idx = 1  # simulating a priority consult of the OTHER player, mid the turn player's own MAIN1
    assert not speed_legal(state, Speed.SORCERY)  # refused -- not their turn, even though phase/stack still say MAIN1/empty
    assert not speed_legal(state, Speed.YOUR_TURN)
    assert speed_legal(state, Speed.INSTANT)  # unaffected -- instant speed never cares whose turn it is

    print("turn.py turn-owner speed_legal self-check: OK")
