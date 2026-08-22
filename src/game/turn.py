"""Turn loop: a fixed sequence of phases (game.turn.Phase), each running
its own turn-based automatic effect (if any), then a real priority round
where both players get a chance to act or respond -- except Untap (never
any priority) and Cleanup (priority only if something triggers)."""

import enum

from . import registry
from .effects.shared import set_tapped
from .effects.combat import (
    attackers_needing_damage_assignment, blocker_lethal_capacities, combat_damage_step, creature_block_eligible,
    declare_attackers_step, enforce_menace, menace_block_incomplete,
)
from .effects.stack import resolve_top_of_stack
from .effects.state_based import check_state_based_actions, cleanup_step
from .effects.triggers import promote_triggers_to_stack
from .resolution import (
    begin_assign_combat_damage, begin_declare_blockers, begin_mulligan,
    refizzle_if_now_targetless,
)
from .state import DeckedOut, new_multiplayer_game_state


class Phase(enum.Enum):
    """One full turn's phases, in order. Phases gate: (1) each phase's own
    turn-based automatic effect (if any), plus DECLARE_BLOCKERS' own
    block-assignment decision, both running before that phase's priority
    round; (2) what "Pass" advances past; and (3) via Speed/speed_legal,
    which top-level actions are legal at all -- Speed.SORCERY is legal
    only during MAIN1/MAIN2 of your own turn. UPKEEP has no automatic
    effect yet (no card here has an upkeep trigger). END's effect
    (cleanup_step) is real, not a placeholder."""
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
    """When a top-level action (cast, activate, play a land) is legal
    relative to phase -- real Magic's casting-speed rules, without the
    stack. INSTANT is legal regardless of whose turn it structurally is;
    YOUR_TURN requires state.turn_player_idx with no phase/stack
    restriction of its own (unlike SORCERY, which needs both). No card
    currently sets Speed.YOUR_TURN -- every non-instant card falls
    through to SORCERY (drl_env._cast_speed's default).

    Only checked for a top-level action that WOULD initiate a resolution,
    never for a pending-resolution continuation -- those are governed
    purely by pending_resolution, and Pass is already illegal while one
    is open."""
    SORCERY = "sorcery"
    YOUR_TURN = "your_turn"
    INSTANT = "instant"


# Where Speed.SORCERY is legal -- both main phases (the stack half of
# "any time you could cast a sorcery" is checked separately). A deck
# with no MAIN2 (MINIMAL_PHASES) degrades to "MAIN1 only" for free.
SORCERY_SPEED_PHASES = {Phase.MAIN1, Phase.MAIN2}

# Rule 500.4 (unused mana empties at every step/phase boundary) applies with
# no exceptions -- see _empty_mana_pools below.


def _tally_mana_mistake(state, idx, player, burnt):
    """DENSE, narrower reward-facing signal for PlayerState.mana_mistake_burn
    (drained by a reward_fn tagged consumes_mana_mistake=True; none
    currently is). Called once per non-empty pool clear from
    _empty_mana_pools, only counting a burn as a genuine mistake once
    THREE exemptions have all failed: nothing was paid toward a
    cast/ability this phase (cost_paid_this_phase), no trigger was queued
    for this player this phase (triggers_fired_this_phase), and -- checked
    only in that residual case, via the optional state.on_mana_burn hook
    -- nothing was legally castable with the floating pool. A real payment
    or trigger exempts on its own, before ever consulting the hook."""
    if not player.cost_paid_this_phase and not player.triggers_fired_this_phase:
        hook = state.on_mana_burn
        if hook is not None and not hook(state, idx):
            player.mana_mistake_burn += burnt


def _empty_mana_pools(state):
    """Rule 500.4: at the end of each step/phase, all unused mana empties
    for BOTH players. Logs each non-empty emptying and tallies
    PlayerState.mana_burnt_total (raw diagnostic) by total pips lost.

    Drives PlayerState.mana_mistake_burn via _tally_mana_mistake, and
    resets cost_paid_this_phase/triggers_fired_this_phase for every player.
    Also tallies PlayerState.mana_burnt_this_turn (reset by _run_turn_gen,
    not here) and its single-pip-tagged subset
    mana_burnt_this_turn_single_pip (rl.rewards.with_dense_mana_burn_
    penalty's actual input) -- summed from PlayerState.mana_pool_single_pip
    at the moment of the burn, per-pip attribution rather than a
    whole-phase cut. Feeds mana_burnt_total_single_pip the same way.

    The logged event also carries pools_single_pip (idx -> pip count) as a
    per-CLEAR breakdown, letting a consumer reconstruct a per-turn burn
    timeline from game_logs alone.

    Fires state.on_single_pip_burn(state, idx, single_pip_burnt) for EVERY
    player, every call (0 when nothing was floating) -- the
    credit-assignment fix for with_dense_mana_burn_penalty, letting a
    consumer charge the penalty against the actual Tap actions instead of
    whatever's pending at the seat's next decision."""
    emptied, emptied_single_pip = {}, {}
    for idx, player in enumerate(state.players):
        single_pip_burnt = 0
        if player.mana_pool:
            emptied[idx] = dict(player.mana_pool)
            burnt = sum(player.mana_pool.values())
            single_pip_burnt = sum(player.mana_pool_single_pip.values())
            emptied_single_pip[idx] = single_pip_burnt
            player.mana_burnt_total += burnt
            player.mana_burnt_total_single_pip += single_pip_burnt
            player.mana_burnt_this_turn += burnt
            player.mana_burnt_this_turn_single_pip += single_pip_burnt
            _tally_mana_mistake(state, idx, player, burnt)
            player.mana_pool.clear()
            player.mana_pool_single_pip.clear()
        if state.on_single_pip_burn is not None:
            state.on_single_pip_burn(state, idx, single_pip_burnt)
        player.cost_paid_this_phase = False
        player.triggers_fired_this_phase = False
    if emptied:
        state.log_event("mana_emptied", pools=emptied, pools_single_pip=emptied_single_pip)


def speed_legal(state, speed):
    """The one gate every timing-restricted legal_fn in drl_env calls into.

    Real Magic's sorcery-speed rule is "your main phase, empty stack, you
    have priority" -- all three conditions: state.phase is a single shared
    field describing the TURN's phase, not whichever player is being
    asked, so a non-turn player holding priority during the turn player's
    MAIN1/MAIN2 must still be refused (checked via state.turn_player_idx;
    game.drl_env._land_drop_legal needs the identical check).
    YOUR_TURN carries the turn-ownership restriction alone, with no
    phase/stack restriction of its own."""
    if speed is Speed.SORCERY:
        return (
            state.active_idx == state.turn_player_idx
            and state.phase in SORCERY_SPEED_PHASES
            and not state.stack
        )
    if speed is Speed.YOUR_TURN:
        return state.active_idx == state.turn_player_idx
    return True


# Decks with combat_enabled=True get the full turn; everything else
# collapses to UNTAP/DRAW/MAIN1/END -- the phases that do anything without
# a combat step. Not skipped via a forced Pass; simply not in the sequence.
FULL_PHASES = tuple(Phase)
MINIMAL_PHASES = (Phase.UNTAP, Phase.DRAW, Phase.MAIN1, Phase.END)

# No action/iteration cap exists on these loops: a step/phase can only end
# when the stack is empty and all players have passed in succession (CR
# 500.3/405.4), so an artificial cap that advances early is a rules
# violation. This means a policy that never converges on a legal,
# progress-making action can hang these loops indefinitely -- deliberate:
# correctness of turn structure over that defense-in-depth. A recurrence
# should be fixed above this layer (e.g. a wall-clock timeout), not with
# another silent mid-resolution truncation here.


def untap_step(state):
    # `untapped` only records a permanent once its FINAL tapped state for
    # this step is known (after skip_next_untap has had its say).
    untapped = []
    for permanent in state.battlefield:
        was_tapped = permanent.tapped
        permanent.tapped = False
        permanent.summoning_sick = False
        permanent.flags.pop("used_this_turn", None)  # Barrels of Blasting Jelly
        # "doesn't untap during its controller's next untap step" (Sleep of
        # the Dead): skip this permanent's untap once, consuming the flag.
        if permanent.flags.pop("skip_next_untap", False):
            set_tapped(state, permanent, True, reason="skip_next_untap")
        elif was_tapped:
            untapped.append((permanent.card_def.name, permanent.slot))
    # The Initiative's "until your next turn" durations expire at their
    # owning player's turn start. Lazy import: undercity loads after turn.py.
    from .effects import undercity
    undercity.expire_until_next_turn(state)
    # Impulse cards past their "play until end of [this/your next] turn"
    # deadline expire now, leaving the impulse zone untracked.
    expired = [(cd.name, u) for (cd, u) in state.impulse if state.turn_number > u]
    if expired:
        state.impulse = [(cd, u) for (cd, u) in state.impulse if state.turn_number <= u]
        state.log_event("impulse_expired", cards=[n for n, _u in expired])
    if untapped:
        state.log_event("untap_step", untapped=untapped)


def upkeep_step(state):
    """"At the beginning of your upkeep, ..." triggers (Delver of Secrets).
    Queues an "upkeep" trigger for each of the turn player's battlefield
    permanents whose registry has an "upkeep_trigger"; the following
    priority round promotes them onto the stack."""
    for permanent in state.battlefield:  # active_idx == turn player at UPKEEP
        if registry.EFFECT_REGISTRY.get(permanent.card_def.effect_id, {}).get("upkeep_trigger") is not None:
            state.trigger_queue.append({"type": "upkeep", "card_def": permanent.card_def, "permanent": permanent})
    # The Initiative: venture into Undercity at the holder's own upkeep.
    if state.initiative_idx == state.turn_player_idx:
        from .effects import undercity
        undercity.queue_venture(state, state.turn_player_idx)


def draw_step(state):
    # Checked against this player's own turn count (turns_taken), not the
    # game's global turn_number -- P2's first turn is turn_number==2.
    if state.turns_taken == 1 and state.on_the_play:
        return
    state.draw(1)


_PHASE_AUTO_EFFECTS = {
    Phase.UNTAP: untap_step,
    Phase.UPKEEP: upkeep_step,
    Phase.DRAW: draw_step,
    Phase.DECLARE_ATTACKERS: declare_attackers_step,
    Phase.COMBAT_DAMAGE: combat_damage_step,
    # Phase.END has no entry -- cleanup_step runs explicitly, mid its own custom handling below.
}


def _run_mulligan_gen(state):
    """Pregame: every player decides keep-or-mulligan for their own opening
    hand (already dealt by new_multiplayer_game_state's eager draw(7)), one
    player fully at a time. APNAP order: whoever active_idx already points
    at goes first. Runs before turn 1, driven by game_coroutine rather than
    folded into _run_turn_gen."""
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
    """The defending player's own block-assignment decision, folded into
    _run_turn_gen's Phase.DECLARE_BLOCKERS handling. Declaring blockers
    belongs to the DEFENDER, not the turn player, so state.active_idx is
    temporarily flipped to them for its scope.

    Flips back to the attacker once the defender is done, before this
    phase's regular priority round runs (attacker gets it first).

    No-op if there's no real opponent (len(state.players) < 2).

    Unbounded: no iteration cap, since capping legal actions per priority
    round would itself be a rules deviation. Detection lives in the
    training harness instead (rl.training.train.collect_rollout raises past a
    decision budget).

    Runs until the defender is done, or a genuinely-stuck state is
    detected below (zero legal actions left). An action that is LEGAL but
    makes no progress spins here forever -- the fix for that is making
    legality and execution agree, not capping this loop."""
    if len(state.players) < 2:
        return
    attacker_idx = state.active_idx
    state.active_idx = 1 - attacker_idx
    state.log_event("priority_flip", reason="declare_blockers", to_idx=state.active_idx)
    try:
        begin_declare_blockers(state, on_complete=lambda s: None)
        while True:
            if state.pending_resolution is None:
                return
            if (state.pending_resolution["kind"] == "declare_blockers" and menace_block_incomplete(state)
                    and not any(creature_block_eligible(state, p) for p in state.battlefield)):
                # Genuinely stuck, not just unproductive: a menace attacker
                # has exactly one committed blocker ("Done" stays illegal)
                # and no creature remains to add as a second one -- zero
                # legal actions exist. Abandon now rather than loop forever;
                # enforce_menace drops the illegal lone block at combat
                # damage, so the OUTCOME doesn't change.
                state.log_event("declare_blockers_abandoned", pending_kind=state.pending_resolution["kind"])
                state.pending_resolution = None
                return
            action = yield
            state.players[state.active_idx].actions_taken += 1
            action()  # "Done" is its own explicit action; None (Pass) is never expected here
    finally:
        state.active_idx = attacker_idx
        state.log_event("priority_flip", reason="declare_blockers_done", to_idx=state.active_idx)


def _assign_combat_damage_gen(state):
    """After blockers are declared: for each attacker blocked by 2+
    creatures, the attacking player assigns that attacker's combat damage
    across its blockers (+ trample), one point at a time. A lone blocker
    (or 0-power attacker) needs no decision. These picks are deliberately
    NOT counted toward actions_taken -- a mechanical consequence of a
    multi-block, not a discretionary action.

    Unbounded but needs no cap: each pick strictly decrements the
    pending's `remaining`, so this always finishes in exactly `power`
    picks."""
    for attacker, blockers, power, has_trample in attackers_needing_damage_assignment(state):
        # Rule 510.1a: an attacker assigns damage only among creatures
        # CURRENTLY blocking it; blocked_by can hold a dead entry (506.4
        # doesn't drop it), which would otherwise offer an illegal, unmasked
        # target.
        #
        # A blocker dying AFTER this decision but BEFORE combat_damage_step
        # deals it is handled separately in
        # game.effects.combat._damage_assignment_for: a departed blocker's
        # earmarked share is added to a trampler's excess rather than lost.
        #
        # Below 2 living blockers there's no free choice to make --
        # combat_damage_step's _default_damage_assignment handles 1 and 0
        # directly.
        living = [b for b in blockers if b in state.opponent.battlefield]
        if len(living) < 2:
            continue
        lethal_by_blocker = blocker_lethal_capacities(state, attacker, living)
        begin_assign_combat_damage(
            state, attacker, living, power, has_trample, lethal_by_blocker, on_complete=lambda s: None,
        )
        while state.pending_resolution is not None:
            action = yield
            action()


def _run_priority_round_gen(state):
    """One or more rounds of real priority-passing, run at the start of
    every phase/step and repeated after every single stack resolution.

    Starts with priority at state.turn_player_idx (rule 1). Before each
    consultation: state-based actions are checked, any pending resolution
    left targetless by those SBAs is re-fizzled
    (refizzle_if_now_targetless), and any newly-queued triggers are
    promoted onto the stack (704.3's ordering: SBAs, then triggers, then
    priority).

    Whoever holds priority either acts (stack grows, priority stays with
    them, rule 2) or passes (priority moves to the other player). Once
    every player has passed in a row: if the stack is non-empty, its top
    item resolves and priority resets to turn_player_idx (rule 1), and the
    round repeats; if empty, this generator ends and the phase can advance.

    Never called for Phase.UNTAP (rule 4). Phase.END calls this only
    conditionally; once entered, it behaves like any other phase."""
    state.active_idx = state.turn_player_idx
    consecutive_passes = 0
    while True:
        check_state_based_actions(state)
        refizzle_if_now_targetless(state)
        # Move queued triggers onto the stack only at a genuine priority
        # point, never mid-resolution (704.3/603.3): promoting then could
        # land a trigger under a not-yet-pushed spell, or clobber an
        # in-progress begin_order_triggers pending. Deferred to the next
        # pending-free iteration, picked up the instant it clears.
        if state.pending_resolution is None:
            promote_triggers_to_stack(state)
        action = yield
        if action is None:
            state.log_event("pass")
            consecutive_passes += 1
            if consecutive_passes >= len(state.players):
                if state.stack:
                    resolve_top_of_stack(state)
                    # Rule 1 only applies to handing out the NEXT priority
                    # window -- not if resolving the stack top opened a
                    # fresh pending_resolution (e.g. a Madness decision),
                    # which is the entry's own controller's forced decision
                    # and stays open past this yield. resolve_top_of_stack
                    # already restored active_idx to that controller;
                    # stomping it back to turn_player_idx here would
                    # reassign the decision to the wrong player.
                    if state.pending_resolution is None:
                        state.active_idx = state.turn_player_idx
                    consecutive_passes = 0
                    continue
                # Stack empty, everyone passed -- the phase/step is over.
                # Reset priority to the turn player: the next phase's
                # auto_effect requires active_idx == turn_player_idx by the
                # time this generator's caller resumes.
                state.active_idx = state.turn_player_idx
                return
            state.active_idx = 1 - state.active_idx  # the only other player, in a 2-player game
        else:
            state.players[state.active_idx].actions_taken += 1  # Pass doesn't count -- see PlayerState.actions_taken
            action()
            consecutive_passes = 0  # priority holder keeps priority (rule 2)


def _run_turn_gen(state, combat_enabled=False):
    """Generator form of one full turn, shared by run_turn's synchronous
    driver and the training pipeline's per-seat driver. Iterates
    FULL_PHASES or MINIMAL_PHASES depending on combat_enabled; for each
    phase, runs its own turn-based automatic effect (if any), then a real
    priority round -- except Untap (never any priority, rule 4).
    Phase.END packs two real-rules steps (513 end step, 514 cleanup) into
    one Phase value: a normal priority round, then cleanup_step's
    hand-size discard with no priority beyond a Madness decision (see its
    own handling below, and state.in_cleanup's docstring).
    Phase.DECLARE_BLOCKERS runs the defender's block-assignment decision
    (_declare_blockers_gen) before its own priority round.

    Every yield uses the same protocol: the caller sends back None
    ("pass") or a zero-arg callable via gen.send(...). This generator is
    agnostic to WHO answers -- state.active_idx tells the caller that.
    Ends (StopIteration) once every phase has run its course.

    Wrapped in try/except DeckedOut: a draw in any phase can raise
    DeckedOut from arbitrarily deep in a resolution chain; catching it
    here ends the turn/generator immediately, with state.decked_out
    already set by draw() itself.

    combat_enabled: per-deck opt-in (default off) -- only rakdos
    madness/mono red madness/boggles pass True."""
    try:
        # Whoever active_idx is right now is the true turn owner for the
        # whole turn -- callers always invoke this before any priority
        # consult could flip it away. Set once, untouched until next turn.
        state.turn_player_idx = state.active_idx
        state.turn_number += 1
        state.turns_taken += 1  # this player's own turn count, distinct from turn_number once a 2nd player exists
        state.lands_played_this_turn = 0
        state.cards_drawn_this_turn = 0
        # BOTH players: the non-active player can float/burn mana too.
        for player in state.players:
            player.mana_burnt_this_turn = 0
            player.mana_burnt_this_turn_single_pip = 0
            player.mana_burn_penalty_credited = 0.0
        state.log_event("turn_start", turn_player_idx=state.turn_player_idx)

        phases = FULL_PHASES if combat_enabled else MINIMAL_PHASES
        for phase in phases:
            from_phase = state.phase
            # Rule 500.4: unused mana empties at every step/phase boundary,
            # for both players. Logged BEFORE state.phase advances, tagged
            # with from_phase, matching real Magic's "at the end of X" timing.
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
                # unblocked (enforce_menace), before any damage is assigned.
                enforce_menace(state)
                # A multi-blocked attacker's controller assigns that
                # attacker's damage across its blockers (gang-blocking).
                yield from _assign_combat_damage_gen(state)
                if state.turn_won is not None:
                    return

            if phase is Phase.END:
                # Rule 513: the end step itself -- a normal priority round,
                # same shape as any other phase's trailing round (mana
                # abilities/instants/responses all legal).
                yield from _run_priority_round_gen(state)
                if state.turn_won is not None:
                    return

                # Rule 500.4: unused mana empties here too -- a genuine
                # sub-step boundary between the end step and cleanup, even
                # though both share one Phase.END value (state.phase never
                # becomes a distinct "cleanup" phase -- see
                # state.in_cleanup's docstring). Called explicitly since
                # state.phase not changing means the generic sweep above
                # never fires for this boundary on its own.
                _empty_mana_pools(state)

                # Rule 514: cleanup (hand-size discard + damage/until-EOT
                # clear), run explicitly rather than as this phase's auto_effect.
                cleanup_step(state)
                if state.turn_won is not None:
                    return

                # Normally no player receives priority at all during
                # cleanup (514.3). AUTHORIZED SIMPLIFICATION (owner-
                # approved 2026-08-10): gate-free mana abilities (605.1a/
                # 605.3b -- legal in ANY OTHER priority window) are
                # additionally revoked for this whole cleanup portion, via
                # state.in_cleanup (see its own docstring) -- real Magic
                # wouldn't revoke them, but nothing here needs to float
                # mana it can never spend, and doing so was exactly what
                # let mana burnt during cleanup dodge the dense mana-burn
                # reward signal (rl.rewards.with_dense_mana_burn_penalty)
                # entirely (this engine has no interrupt window, so the
                # player who just finished their turn never gets another
                # decision before the NEXT turn's own reset silently
                # discards whatever got tallied here). The forced discard
                # picks themselves, and a discarded card's own Madness
                # cast-or-graveyard decision (game.resolution.
                # handlers_library._discard_one), are the ONLY things still
                # possible here: rule 514.3a's real "priority round if
                # something triggers" exception is kept (below) purely to
                # let that Madness decision resolve, not to reopen mana/
                # instant-speed play -- in_cleanup stays True through it.
                state.in_cleanup = True
                try:
                    while True:
                        while state.pending_resolution is not None:
                            action = yield
                            state.players[state.active_idx].actions_taken += 1
                            action()
                        if state.turn_won is not None:
                            return
                        if not state.trigger_queue:
                            break
                        # Only a Madness decision (queued by the discard
                        # picks above) is expected to reach here -- in_cleanup
                        # stays True through this round too, so nothing else
                        # newly legal can sneak in alongside it.
                        yield from _run_priority_round_gen(state)
                        if state.turn_won is not None:
                            return
                        cleanup_step(state)
                        if state.turn_won is not None:
                            return
                finally:
                    state.in_cleanup = False
                continue

            yield from _run_priority_round_gen(state)
            if state.turn_won is not None:
                return
    except DeckedOut:
        # Decking out is an instant loss for whoever draws from an empty
        # library. In 2-player the OTHER player wins outright; in 1-player
        # there's no one to award the win to, so turn_won/winner stay None.
        if len(state.players) > 1:
            state.turn_won = state.turn_number
            state.winner = 1 - state.active_idx
        return


def run_turn(state, choose_action, combat_enabled=False):
    """One full turn, pull-style: repeatedly calls choose_action(state) and
    feeds the result into _run_turn_gen (see its docstring for the actual
    turn logic). choose_action is called for EVERY yield regardless of
    whose decision it is -- a closure that acts differently per player
    reads state.active_idx itself."""
    gen = _run_turn_gen(state, combat_enabled=combat_enabled)
    try:
        next(gen)  # advance to first yield, or StopIteration if the turn ended during an automatic effect
        while True:
            gen.send(choose_action(state))
    except StopIteration:
        pass


def _yield_decisions(inner, state):
    """Adapts a choose_action-driven inner generator (_run_mulligan_gen /
    _run_turn_gen) into one that yields the live STATE outward and forwards
    the chosen action back in via .send()."""
    try:
        next(inner)
        while True:
            action = yield state
            inner.send(action)
    except StopIteration:
        pass


def game_coroutine(state, horizon=None, combat_enabled=False):
    """run_multiplayer_game's decision flow as a resumable generator: yields
    the state at every point a player must choose (pregame mulligan, then
    every turn's decisions) and expects the chosen action back via .send()
    -- the same value choose_action returns. run_multiplayer_game drives
    this synchronously."""
    yield from _yield_decisions(_run_mulligan_gen(state), state)
    # Baseline for gameplay-only action counts: everything counted up to
    # here is pregame mulligan/keep/bottom picks (PlayerState.pregame_actions).
    for player in state.players:
        player.pregame_actions = player.actions_taken
    first_turn = True
    while (horizon is None or state.turn_number < horizon) and state.turn_won is None and not state.decked_out:
        if not first_turn:
            state.active_idx = 1 - state.active_idx
        first_turn = False
        yield from _yield_decisions(_run_turn_gen(state, combat_enabled=combat_enabled), state)


def run_multiplayer_game(decklists, rng, starting_player_idx, choose_action,
                          horizon=None, combat_enabled=False, event_log=None, on_mana_burn=None,
                          on_single_pip_burn=None, stratify=None):
    """N-player entry point. Full sequential turns -- one player's whole
    turn runs to completion before active_idx flips to the next one.
    horizon=None (default) means uncapped: the loop ends only on an actual
    game-loss condition (state.turn_won). Turn count can't run away
    (draw_step draws one card per turn, bounding turns by combined library
    size), but a single priority round's own inner loop has no cap, so a
    policy that never converges on a legal action can still hang this
    indefinitely. Pass an int horizon to bound the turn count for a
    self-check; it doesn't protect against that inner case.

    Flips active_idx lazily -- right before the NEXT turn starts -- so
    state.active_idx always names whoever just played once this function
    returns, including on a horizon-capped exit.

    on_mana_burn/on_single_pip_burn: stamped straight onto the matching
    GameState attributes (see their docstrings). stratify: passed straight
    through to new_multiplayer_game_state."""
    state = new_multiplayer_game_state(decklists, starting_player_idx, rng, event_log=event_log, stratify=stratify)
    state.on_mana_burn = on_mana_burn
    state.on_single_pip_burn = on_single_pip_burn
    # Drive the game via game_coroutine -- the single source of truth, so
    # the batched rollout collector can interleave many games over it.
    gen = game_coroutine(state, horizon=horizon, combat_enabled=combat_enabled)
    try:
        req_state = next(gen)
        while True:
            req_state = gen.send(choose_action(req_state))
    except StopIteration:
        pass
    return state

