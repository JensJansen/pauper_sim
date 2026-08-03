"""Turn loop: a fixed sequence of phases (game.turn.Phase), each running its
own turn-based automatic effect (if any) then a real priority round
 -- both players get a real chance to act or
respond at every phase/step, not just the turn player, except Untap (never
any priority) and Cleanup (priority only if something triggers there)."""

import enum

from . import registry
from .effects.combat import (
    attackers_needing_damage_assignment, combat_damage_step, creature_block_eligible,
    declare_attackers_step, enforce_menace, menace_block_incomplete,
)
from .effects.stack import resolve_top_of_stack
from .effects.state_based import check_state_based_actions, cleanup_step
from .effects.triggers import promote_triggers_to_stack
from .resolution import (
    begin_assign_combat_damage, begin_declare_blockers, begin_mulligan, complete_resolution,
    refizzle_if_now_targetless,
)
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
    this repo currently has one, so UPKEEP has no automatic effect at all
    yet, just a seam for one later. END's own effect (cleanup_step:
    discard to hand size, clear combat damage) is real, not a
    placeholder."""
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

# Combat's three sub-phases form ONE mana window: mana floated in
# declare-attackers is still there in declare-blockers/combat-damage.
# AUTHORIZED SIMPLIFICATION (owner-approved): real Magic empties the pool
# between each of these steps too (500.4); we don't, treating combat as a
# single window. Every OTHER phase boundary still empties (see _run_turn_gen).
_COMBAT_PHASES = {Phase.DECLARE_ATTACKERS, Phase.DECLARE_BLOCKERS, Phase.COMBAT_DAMAGE}


def _empty_mana_pools(state):
    """Rule 500.4: at the end of each step/phase, all unused mana empties --
    for BOTH players (the non-active player can float mana to cast an instant
    on your turn). Float-first makes the pool a persistent within-phase
    resource, so this per-phase empty is what makes a mis-floated color a real,
    faithful cost (the mana is gone, the source stays tapped) rather than an
    undo. Replaces the old once-per-turn clear (untap_step) -- and, since it
    now owns every mana clear, logs each non-empty emptying (keyed by player
    index) so the event log stays a faithful record of when mana was lost, and
    tallies each player's own PlayerState.mana_burnt_total (rl.rewards reads
    it) by the total pips lost, not just whether anything was."""
    emptied = {}
    for idx, player in enumerate(state.players):
        if player.mana_pool:
            emptied[idx] = dict(player.mana_pool)
            player.mana_burnt_total += sum(player.mana_pool.values())
            player.mana_pool.clear()
    if emptied:
        state.log_event("mana_emptied", pools=emptied)


def speed_legal(state, speed):
    """The one gate every timing-restricted legal_fn in drl_env calls
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

# Per-phase cap on model-action loop iterations -- guards against an
# infinite policy loop, not expected. A single "logical" action (cast a
# spell, activate an ability) can cost multiple loop iterations to fully
# resolve (one per mana tap, plus any search/scry/take decisions) -- and
# Speed.INSTANT actions/activated abilities (unrestricted by phase) can
# still bring that full complexity to any phase, not just the main phases
# (only Speed.SORCERY actions are confined to MAIN1/MAIN2). Kept uniform
# at 200 for every phase, precisely to avoid capping a non-main phase low
# enough to truncate a resolution mid-flight; trim per-phase if a real
# training run ever shows a phase needs a tighter bound.
PHASE_ACTION_CAPS = {phase: 200 for phase in Phase}

# Safety cap on ONE priority round's own inner loop (_run_priority_round_gen
# below) -- same "guard against an infinite policy loop, not expected"
# reasoning as PHASE_ACTION_CAPS, just scoped to a single round (bounded by
# how many times priority could plausibly pass back and forth, and by how
# many stack items could plausibly need resolving, before real resource
# limits -- mana, cards in hand -- stop a policy cold) rather than a whole
# phase, since a priority round can now run more than once per phase (once
# per stack resolution).
#
# ponytail: capped well below PHASE_ACTION_CAPS (20, not 200) so a policy
# stuck re-issuing the same non-progressing action gets cut off quickly
# rather than burning a large budget silently. 20 leaves room for a couple
# of real casts plus combat per phase without truncating a resolution
# mid-flight (casting a single spell -- tap + spend + both players passing
# to resolve -- already costs 4 by itself). Raise it back toward 200, or
# replace it with a smarter "no observable progress" detector, if a real
# training run ever needs more headroom than 20 provides.
PRIORITY_ROUND_ACTION_CAP = 20


def untap_step(state):
    untapped = [(p.card_def.name, p.slot) for p in state.battlefield if p.tapped]
    for permanent in state.battlefield:
        permanent.tapped = False
        permanent.summoning_sick = False
        permanent.flags.pop("used_this_turn", None)  # Barrels of Blasting Jelly
        # "doesn't untap during its controller's next untap step" (Sleep of
        # the Dead): skip this permanent's untap ONCE, consuming the flag.
        if permanent.flags.pop("skip_next_untap", False):
            permanent.tapped = True
    # This step only handles untapping -- mana pools empty via
    # _empty_mana_pools at every phase boundary (including into this step),
    # not here.
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
    if untapped:
        state.log_event("untap_step", untapped=untapped)


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
    # game's global turn_number: once a second player also takes turns,
    # turn_number==1 no longer means "my first turn" -- P2's first turn is
    # turn_number==2. In a 1-player state turns_taken tracks turn_number
    # exactly (there's only one player).
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
    hand (already dealt by state.new_multiplayer_game_state's
    own eager draw(7)), one player fully at a time -- same per-player
    active_idx flip pattern as _declare_blockers_gen, just scoped to the
    whole pregame instead of one phase. APNAP order: whoever active_idx
    already points at (the real starting player) goes first. Runs entirely
    before turn 1 (state.turn_number is still 0, state.phase is still None)
    -- nothing here touches any turn-scoped field, so it's driven by
    game_coroutine (via _yield_decisions) rather than folded into _run_turn_gen."""
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


def _declare_blockers_gen(state):
    """The defending player's own block-assignment decision, yielded
    through the SAME generic decision protocol as everything else in this
    generator-based turn loop -- folded directly into _run_turn_gen's own
    Phase.DECLARE_BLOCKERS handling. Declaring blockers is a turn-based
    special action belonging to the DEFENDER specifically, not the turn
    player -- real Magic's own rule -- so state.active_idx is temporarily
    flipped to them for its scope: state.hand/state.battlefield only mean
    the defender's OWN zones once this flip has happened, not the
    attacker's.

    Flips back to the attacker once the defender is done (0 or more
    assignments), before this phase's own regular priority round runs
    next (attacker gets it first, per rule 1).

    No-op (no yield at all) if there's no real opponent to consult
    (len(state.players) < 2).

    Bounded at PRIORITY_ROUND_ACTION_CAP iterations, same constant and same
    reasoning _run_priority_round_gen's own cap-exhaustion path already
    documents: a deterministic or barely-trained policy can keep
    re-issuing the same unproductive action forever, so this loop cannot
    be left unbounded. Exhausting the cap force-completes with whatever's
    already been assigned (complete_resolution, same "can't finish, so the
    attempt ends outright" precedent the priority round's own
    cap-exhaustion below already uses) rather than leaving
    declare_blockers open forever."""
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
            if (state.pending_resolution["kind"] == "declare_blockers" and menace_block_incomplete(state)
                    and not any(creature_block_eligible(state, p) for p in state.battlefield)):
                # Genuinely stuck, not just unproductive: a menace attacker has
                # exactly one committed blocker (so "Done" stays illegal, per
                # menace_block_incomplete), and no creature remains that could be
                # added as a second one -- zero legal actions exist at all, not
                # merely a bad one to keep re-submitting. The cap below assumes
                # SOME action can always be resubmitted to consume its budget
                # (the boggles_mirror precedent this cap was built for -- a
                # policy re-issuing the SAME unproductive assignment forever);
                # it can never fire here because there is nothing to submit in
                # the first place. Abandon right now instead of waiting for a
                # cap that's unreachable -- identical recovery to the
                # cap-exhaustion path below (enforce_menace already documents
                # this exact scenario and drops the illegal lone block at
                # combat damage, so the OUTCOME doesn't change, only when the
                # abandon happens).
                state.log_event("declare_blockers_cap_abandoned", pending_kind=state.pending_resolution["kind"])
                state.pending_resolution = None
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
            # the attempt ends" precedent the priority round's own
            # cap-exhaustion already uses.
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
    consultation: state-based actions are checked, any pending resolution
    left targetless BY those SBAs is re-fizzled (refizzle_if_now_targetless
    -- an SBA can kill the only legal target of a choose_permanent/choose_
    opponent_permanent/choose_any_target that validated non-empty when it
    opened, one priority-loop iteration earlier; see that function's own
    docstring for the crash this closes), and any newly-queued triggers are
    promoted onto the stack (real Magic 704.3's actual ordering -- SBAs,
    then triggers move to the stack, THEN priority is given) -- cheap no-ops
    when there's nothing to do, so unconditional every time is simpler and
    more rules-accurate than trying to detect "did anything change" by hand.

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
        refizzle_if_now_targetless(state)
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
                    # turn_player_idx here would reassign the decision --
                    # and the zone reads (state.exile/state.hand) it makes --
                    # to the wrong player.
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
    # stay "active" going into the next phase.
    state.active_idx = state.turn_player_idx
    # A pending_resolution can ALSO still be open here (e.g. a deterministic
    # or barely-trained policy oscillating on the same cost payment for the
    # full cap without ever completing it). Every other exit from this loop
    # guarantees pending_resolution is None (the clean exit above only
    # returns once the stack's empty AND no cost/choice is outstanding);
    # this is the one path that doesn't, so it has to drop it itself rather
    # than let it leak across a phase boundary no caller expects.
    # ponytail: float-first has no undo mechanism at all (payment is an
    # irreversible pool spend, not a reversible tap), so a dropped pay_cost
    # is simply left as-is here: any mana already spent toward it via
    # execute_pool_spend is gone, and any floated-but-unspent mana just sits
    # in the pool until the next phase-boundary clear -- both are ordinary
    # "burned floating mana" outcomes real Magic itself produces whenever a
    # player floats more than they use, not a new failure mode. Upgrade to a
    # smarter "no observable progress" detector if this ever needs to be
    # tighter.
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
            # Rule 500.4: unused mana empties at every step/phase boundary (both
            # players), EXCEPT between combat's own sub-phases (one mana window
            # -- see _COMBAT_PHASES). Logged BEFORE state.phase/phase_change
            # advance, tagged with from_phase (the phase actually ending) --
            # matching real Magic's own "at the end of X" timing, and keeping
            # the replay viewer from showing combat's floated mana as still
            # present once Main Phase 2 (or any other next phase) has begun.
            if not (from_phase in _COMBAT_PHASES and phase in _COMBAT_PHASES):
                _empty_mana_pools(state)
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
        # the active player ever draws); in a 1-player state there's no one
        # to award the win to, so turn_won/winner simply stay None.
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
    the action in. The inner's own yielded value is ignored by its
    synchronous driver (run_turn's own loop / _run_mulligan_gen's caller
    here), so yielding `state` instead -- what a driver actually needs to
    choose on -- changes nothing about the decision sequence."""
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
    zero-arg executor callable). run_multiplayer_game drives this
    synchronously, so every existing caller and self-check exercises this
    exact path, with the same horizon/turn_won/decked_out guard and the
    same lazy active_idx flip run_multiplayer_game itself documents."""
    yield from _yield_decisions(_run_mulligan_gen(state), state)
    # Baseline for gameplay-only action counts: everything counted up to here is
    # pregame mulligan/keep/bottom picks (see PlayerState.pregame_actions).
    for player in state.players:
        player.pregame_actions = player.actions_taken
    first_turn = True
    while (horizon is None or state.turn_number < horizon) and state.turn_won is None and not state.decked_out:
        if not first_turn:
            state.active_idx = 1 - state.active_idx
        first_turn = False
        yield from _yield_decisions(_run_turn_gen(state, combat_enabled=combat_enabled), state)


def run_multiplayer_game(decklists, rng, starting_player_idx, choose_action,
                          horizon=None, combat_enabled=False, event_log=None):
    """N-player entry point. Full
    sequential turns -- one player's whole turn runs to completion (the
    same run_turn/choose_action(state) contract run_turn itself uses; a
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
    # Drive the game as a coroutine (game_coroutine) -- choose_action(state)
    # is called for EVERY decision regardless of whose it is. The loop
    # lives in game_coroutine (single source of truth) so the batched
    # rollout collector can interleave many games over the same generator.
    gen = game_coroutine(state, horizon=horizon, combat_enabled=combat_enabled)
    try:
        req_state = next(gen)
        while True:
            req_state = gen.send(choose_action(req_state))
    except StopIteration:
        pass
    return state

