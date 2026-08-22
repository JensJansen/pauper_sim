"""Tests for game.turn: the 2-player engine end to end -- turn alternation,
on_the_play/turns_taken, life_total loss, deck-out-as-instant-loss, a real
game played through actual cards (not a synthetic no-op deck), the
_declare_blockers_gen flip-drive-restore shape, the turn-owner vs
priority-holder split in speed_legal, and rule 500.4's per-phase-boundary
mana clear."""

import random

from drl_env._actions_mana import _mana_ability_legal
from game import mana, registry, resolution
from game.cards import CardDef, CardType, EffectId
from game.effects.casting import play_land_from_hand
from game.effects.stack import push_to_stack
from game.effects.win_check import deal_damage_to_opponent
from game.resolution.handlers_casting import execute_madness_decline
from game.state import GameState, Permanent, PlayerState
from game.turn import (
    Phase, Speed, _declare_blockers_gen, _run_priority_round_gen, new_multiplayer_game_state,
    run_multiplayer_game, run_turn, speed_legal,
)


def test_two_player_construction():
    # construction: opening hands, on_the_play, starting life
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


def test_stratify_forces_opening_land_count():
    # stratify=(seat, land_count): only that seat's opening hand is forced;
    # the other seat still gets a plain, unforced shuffle (Mountain-only
    # deck, so its hand is trivially 7 lands either way -- the real check
    # is seat 0's hand, dealt from a mixed deck).
    decklist = [("Mountain", 33), ("Lightning Bolt", 27)]
    state = new_multiplayer_game_state(
        decklists=[decklist, decklist],
        starting_player_idx=0,
        rng=random.Random(0),
        stratify=(0, 0),
    )
    assert all(c.name != "Mountain" for c in state.players[0].hand)
    assert len(state.players[0].hand) == 7

    state = new_multiplayer_game_state(
        decklists=[decklist, decklist],
        starting_player_idx=0,
        rng=random.Random(0),
        stratify=(1, 3),
    )
    assert sum(1 for c in state.players[1].hand if c.name == "Mountain") == 3


def test_turn_alternation_and_on_the_play():
    def _pass_except_discard(state):
        # Always Pass -- pure alternation, no card actions -- EXCEPT
        # cleanup_step's own hand-size discard is mandatory in real Magic
        # (the real mask-driven path, drl_env._pass_legal, already refuses
        # Pass while a resolution is pending; this hand-written policy has
        # to honor that too now that _run_turn_gen's Phase.END loop
        # actually drives the resolution to completion instead of
        # silently abandoning it on a stray Pass).
        if state.pending_resolution is not None:
            # run_multiplayer_game runs the pregame mulligan phase
            # (game.turn._run_mulligan_gen) before turn 1 -- always keep
            # (0 mulligans taken).
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


def test_life_total_loss_direct_damage():
    # life_total loss (direct deal_damage_to_opponent call)
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


def test_deck_out_is_instant_loss_for_drawing_player():
    # deck-out is an instant loss for the drawing player, in 2p.
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


def test_real_two_player_game_through_actual_cards():
    # A real 2-player game, played through actual cards. Player 0: Mountain
    # + Lightning Bolt (real red_cards.py cards) racing to burn player 1's
    # life_total to 0. Player 1: Mountain only, never acts (always passes)
    # -- a pure punching bag, proving damage actually lands on a REAL
    # opponent PlayerState through the normal cast/mana-payment path
    # (game.mana.begin_pay_cost + push_to_stack), not a direct
    # deal_damage_to_opponent call like the unit check above.
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
        s.move_card(cd, s.graveyard)
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
            # run_multiplayer_game runs the pregame mulligan phase for BOTH
            # players first -- always keep (0 mulligans taken).
            return lambda: resolution.execute_mulligan_keep(state)
        if state.pending_resolution is not None and state.pending_resolution["kind"] == "discard":
            # cleanup_step's own hand-size discard
            # can happen to EITHER player at the end of THEIR OWN
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
            # Cast-then-pay (601.2f/g): the spell is announced first, so THIS is
            # the window where mana gets produced -- tap a source when the pool
            # can't cover the remainder yet, then spend what's floating.
            options = mana.pool_spend_options(state)
            if options:
                return lambda: mana.execute_pool_spend(state, options[0])
            mtn = next((p for p in state.battlefield if p.card_def.name == "Mountain" and not p.tapped), None)
            return (lambda: mana.activate_mana_source(state, mtn)) if mtn is not None else None
        if state.lands_played_this_turn == 0 and any(c.name == "Mountain" for c in state.hand):
            return lambda: play_land_from_hand(state, registry.CARD_DEFS["Mountain"])
        if _bolts_available(state) > 0:
            # plan_payment now counts untapped sources as well as the pool, so
            # no pre-float step is needed -- announce, then pay above.
            if mana.plan_payment(state, bolt_def.cast_cost) is not None:
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


def test_declare_blockers_gen_flips_active_idx_to_defender_and_restores():
    # _declare_blockers_gen: the defender's own consult, active_idx flip.
    # test_handlers_combat.py already covers begin_declare_blockers/
    # declare_blocker_assignment directly; this one is specifically about
    # the flip-drive-restore shape unique to this generator, driven via
    # the generator's own yield protocol -- the same nested-on_complete
    # re-open of begin_declare_blockers that drl_env._assign_blocker_execute
    # uses after each assignment, just routed through this generator's
    # yield protocol like every other decision point in this file.
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


def test_declare_blockers_gen_noop_with_fewer_than_two_players():
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


def test_declare_blockers_gen_abandons_stuck_menace_instead_of_deadlocking():
    # A menace attacker with exactly ONE committed blocker makes "Done"
    # illegal (menace_block_incomplete), by design -- 509.1c is 0 or 2+,
    # never 1. If the defender has no OTHER creature left to add as a
    # second blocker, "Assign Blocker" is also illegal for everyone
    # (creature_block_eligible excludes an already-committed blocker) -- a
    # genuine zero-legal-action deadlock: this is the ONLY case
    # _declare_blockers_gen still proactively abandons on its own. No
    # iteration cap exists anywhere in this loop (removed 2026-08-19,
    # game/turn.py) -- an unproductive-but-technically-legal action (some
    # OTHER action always resubmittable) is no longer force-ended and will
    # loop forever if a policy keeps re-issuing it; only this specific
    # zero-legal-action case gets caught. This test proves the generator
    # detects that exact dead end proactively and abandons immediately
    # instead of hanging -- the OUTCOME (a lone menace block gets dropped)
    # is enforce_menace's job at combat damage, a separate call site
    # already covered by test_menace_block_incomplete_and_enforce_menace in
    # tests/game/effects/test_combat.py; this test only proves the
    # generator itself doesn't hang on the way there.
    _fb = registry.EFFECT_REGISTRY[EffectId.FILLER]
    registry.EFFECT_REGISTRY[EffectId.FILLER] = {"keywords": {"menace"}}
    try:
        bear = Permanent(CardDef("Menacer", CardType.CREATURE, None, EffectId.FILLER, power=4, toughness=1))
        grizzly = Permanent(CardDef("Grizzly Bears", CardType.CREATURE, None, None))
        state = GameState(on_the_play=True, players=[PlayerState(True), PlayerState(False)])
        state.players[0].battlefield = [bear]
        state.players[1].battlefield = [grizzly]  # the defender's ONLY creature -- no second blocker exists
        state.players[0].attackers = [bear]
        state.active_idx = 0

        def _defender_policy(state):
            pending = state.pending_resolution
            if pending["kind"] == "declare_blockers":
                # Only ever reached once (assign grizzly) -- the SECOND time
                # this generator would ask for a declare_blockers decision, the
                # proactive stuck-check must fire first and abandon without
                # yielding again, so this branch is never called twice.
                outer_on_complete = pending["on_complete"]
                return lambda: resolution.declare_blocker_assignment(
                    state, grizzly, on_complete=lambda s: resolution.begin_declare_blockers(s, outer_on_complete),
                )
            name, slot = resolution.choose_opponent_permanent_options(state)[0]  # the nested "which attacker" pick -- only bear
            return lambda: resolution.execute_choose_opponent_permanent_option(state, name, slot)

        gen = _declare_blockers_gen(state)
        try:
            next(gen)
            while True:
                gen.send(_defender_policy(state))
        except StopIteration:
            pass  # reached cleanly, no RuntimeError

        assert state.pending_resolution is None  # abandoned, not left stuck open
        assert state.active_idx == 0  # flipped back to the attacker via the generator's own finally
        # The lone illegal block is still recorded here -- dropping it is
        # enforce_menace's job at combat damage (a later, separate call site),
        # not this generator's -- so the precondition this test exists to
        # cover is confirmed still genuinely present at this point.
        assert state.players[0].blocked_by == {bear: [grizzly]}
    finally:
        registry.EFFECT_REGISTRY[EffectId.FILLER] = _fb


def test_priority_round_has_no_iteration_cap():
    """The bug this closes (2026-08-19): a since-removed
    PRIORITY_ROUND_ACTION_CAP=20 could silently end a priority round --
    letting the phase/step advance -- with real spells still on the stack,
    once enough granular decisions (mana taps, targeting, a Ward payment,
    ...) consumed the whole budget in one round. Confirmed live: turn 19 of
    a dmir_terror mirror (logs/4_deck_subleague_test_double_round_robin_
    20016games.json, game index 8) -- a warded Snuff Out's cast alone ate
    the cap, so Spell Pierce's response got carried into the NEXT step
    instead of resolving in this one, and the mana emptied at that
    (illegitimate) step boundary along the way denied Snuff Out's
    controller the floating mana they'd otherwise have had to pay Spell
    Pierce's tax. No cap exists anywhere in this loop now (game/turn.py) --
    this drives a priority round through more real stack activity than the
    old cap ever allowed (25 casts, well past 20) and confirms it only
    ever ends with an empty stack, never truncates mid-resolution."""
    state = GameState(on_the_play=True, players=[PlayerState(True), PlayerState(False)])
    state.turn_player_idx = 0
    state.active_idx = 0

    ITEMS = 25  # comfortably past the old cap of 20
    pushed = []

    def _push(name):
        card_def = CardDef(name, CardType.INSTANT, None, None)

        def _resolve(s, cd):
            pushed.append(cd.name)

        def _do():
            push_to_stack(state, card_def, _resolve)
        return _do

    # P0 casts ITEMS filler spells in a row (retaining priority each time,
    # rule 2 -- no pass needed between them), then both players pass in
    # pairs to let every stack item resolve (2 passes each), then a final
    # pair to end the round on the now-empty stack.
    plan = [_push(f"filler{i}") for i in range(ITEMS)] + [None] * (ITEMS * 2 + 2)

    gen = _run_priority_round_gen(state)
    yields = 0
    try:
        next(gen)
        yields = 1
        for step in plan:
            gen.send(step)
            yields += 1
    except StopIteration:
        pass

    assert yields > 20, f"only {yields} yields -- test didn't actually exceed the old cap"
    assert state.stack == [], "priority round returned with the stack still non-empty"
    assert pushed == [f"filler{i}" for i in reversed(range(ITEMS))]  # LIFO: last pushed resolves first
    assert state.active_idx == state.turn_player_idx  # rule 1 invariant restored on clean exit


def test_speed_legal_turn_owner_vs_priority_holder():
    # Turn-owner / priority-holder split:
    # speed_legal's Speed.SORCERY (and Speed.YOUR_TURN) branches
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


def test_rule_500_4_mana_pool_clears_at_every_phase_boundary():
    # Rule 500.4: unused mana empties at every step/phase boundary for BOTH
    # players (not just whoever floated it), with NO exceptions. Combat's three
    # sub-phases used to share one mana window (an authorized simplification);
    # it was removed 2026-08-17 with cast-then-pay, since speculative floating
    # is now the active player's own main phase only and payment-time mana is
    # produced and spent inside a single payment -- nothing can deliberately
    # float INTO combat any more, so the exception had nothing left to protect.
    #
    # The policy below calls activate_mana_source DIRECTLY rather than going
    # through the action mask, so it still floats in DECLARE_ATTACKERS to prove
    # the ENGINE clears it there; the mask's own refusal to offer that float is
    # a separate rule, covered by
    # test_speculative_mana_ability_is_main_phase_only. Two pre-placed, already-
    # untapped Mountains (no land drop / draw needed: player 0 is
    # on_the_play so its first-turn draw is skipped, and an empty hand never
    # triggers cleanup_step's discard) let this float mana at two different
    # phases and check it against every phase boundary in between, for BOTH
    # players -- not just whoever floated it.
    mana_pool_log = []  # one (phase, player0_pool, player1_pool) entry per NEW phase seen
    floated_at = set()

    def _mana_clear_policy(state):
        phase = state.phase
        if not mana_pool_log or mana_pool_log[-1][0] is not phase:
            mana_pool_log.append((phase, dict(state.players[0].mana_pool), dict(state.players[1].mana_pool)))
        if state.active_idx != 0:
            return None  # player 1 never acts; its float (below) is injected directly, not via an action
        mtn = next((p for p in state.battlefield if p.card_def.name == "Mountain" and not p.tapped), None)
        if mtn is None:
            return None
        if phase is Phase.MAIN1 and "main1" not in floated_at:
            floated_at.add("main1")
            state.players[1].mana_pool["W"] = 1  # the NON-active player's own float, checked alongside player 0's
            return lambda: mana.activate_mana_source(state, mtn)
        if phase is Phase.DECLARE_ATTACKERS and "declare_attackers" not in floated_at:
            floated_at.add("declare_attackers")
            return lambda: mana.activate_mana_source(state, mtn)
        return None

    state = GameState(on_the_play=True, players=[PlayerState(True), PlayerState(False)])
    state.active_idx = 0
    state.players[0].battlefield = [
        Permanent(CardDef("Mountain", CardType.LAND, None, EffectId.MOUNTAIN)),
        Permanent(CardDef("Mountain", CardType.LAND, None, EffectId.MOUNTAIN)),
    ]
    run_turn(state, _mana_clear_policy, combat_enabled=True)
    by_phase = {phase: (p0, p1) for phase, p0, p1 in mana_pool_log}
    assert by_phase[Phase.MAIN1] == ({}, {})  # nothing floated yet
    assert by_phase[Phase.DECLARE_ATTACKERS] == ({}, {})  # MAIN1's float (both players') cleared crossing into combat
    assert by_phase[Phase.DECLARE_BLOCKERS][0] == {}  # DECLARE_ATTACKERS' float is gone too -- 500.4 applies BETWEEN combat steps now
    assert by_phase[Phase.COMBAT_DAMAGE][0] == {}  # and across DECLARE_BLOCKERS -> COMBAT_DAMAGE
    assert by_phase[Phase.MAIN2] == ({}, {})
    assert floated_at == {"main1", "declare_attackers"}  # both floats actually happened, not skipped
    # mana_burnt_total tallies pips lost, not clear-events: player 0 lost the
    # MAIN1 "R":1 AND the DECLARE_ATTACKERS "R":1 (two separate clears) -> 2;
    # player 1 lost only its MAIN1 "W":1 -> 1. Diagnostic-only (logging/viz) --
    # no reward function reads mana_burnt_total; see PlayerState.mana_mistake_burn
    # for the reward-facing, conditional signal.
    assert state.players[0].mana_burnt_total == 2
    assert state.players[1].mana_burnt_total == 1
    # mana_burnt_total_single_pip: player 0's two floats both went through
    # the real activate_mana_source/float_mana tagging flow (plain Mountain
    # taps -- single-pip by construction) -- fully tagged, so it equals
    # mana_burnt_total here. Player 1's "W":1 was injected directly onto
    # mana_pool for the test, bypassing float_mana entirely, so it was never
    # tagged into mana_pool_single_pip -- 0, unlike the raw total's 1.
    assert state.players[0].mana_burnt_total_single_pip == 2
    assert state.players[1].mana_burnt_total_single_pip == 0


def test_mana_mistake_burn_three_way_exemption():
    # _empty_mana_pools's dense reward-facing tally (PlayerState.
    # mana_mistake_burn, meant for a reward_fn tagged consumes_mana_mistake=
    # True -- none currently is): only counts a burn once
    # cost_paid_this_phase, triggers_fired_this_phase, AND the
    # on_mana_burn hook (if any) have all failed to justify it. Called
    # directly (not driven through a real turn) since this is pure
    # conditional bookkeeping over already-set flags -- see
    # test_rule_500_4_mana_pool_clears_at_phase_boundaries_except_combat
    # above for the phase-boundary-timing side of the same function.
    from game.turn import _empty_mana_pools

    def fresh_state(on_mana_burn=None):
        state = GameState(on_the_play=True, players=[PlayerState(True), PlayerState(False)])
        state.on_mana_burn = on_mana_burn
        return state

    # 1) Paid for something this phase -> exempt, even if the hook (were it
    # consulted) would say nothing else was legal. Hook deliberately raises
    # if called, so this also proves the hook is skipped entirely here.
    state = fresh_state(on_mana_burn=lambda s, idx: (_ for _ in ()).throw(AssertionError("hook should not run")))
    state.players[0].mana_pool = {"R": 2}
    state.players[0].cost_paid_this_phase = True
    _empty_mana_pools(state)
    assert state.players[0].mana_mistake_burn == 0
    assert state.players[0].mana_burnt_total == 2  # unconditional diagnostic still tallies regardless
    assert state.players[0].cost_paid_this_phase is False  # reset for the next phase

    # 2) A trigger fired this phase (Writhing Chrysalis-style combat trick)
    # -> exempt, same hook-skipped guarantee as above.
    state = fresh_state(on_mana_burn=lambda s, idx: (_ for _ in ()).throw(AssertionError("hook should not run")))
    state.players[0].mana_pool = {"C": 1}
    state.players[0].triggers_fired_this_phase = True
    _empty_mana_pools(state)
    assert state.players[0].mana_mistake_burn == 0
    assert state.players[0].triggers_fired_this_phase is False  # reset for the next phase

    # 3) Nothing paid, nothing triggered, hook says something WAS legally
    # castable (e.g. declining a legal-but-suboptimal card) -> exempt.
    state = fresh_state(on_mana_burn=lambda s, idx: True)
    state.players[0].mana_pool = {"B": 1}
    _empty_mana_pools(state)
    assert state.players[0].mana_mistake_burn == 0

    # 4) Nothing paid, nothing triggered, hook says nothing was legally
    # castable -> genuine mistake, tallied.
    state = fresh_state(on_mana_burn=lambda s, idx: False)
    state.players[0].mana_pool = {"B": 2}
    _empty_mana_pools(state)
    assert state.players[0].mana_mistake_burn == 2

    # 5) No hook at all (plain rules-engine/test use, e.g. every other test in
    # this file) -> never tallies a mistake, regardless of anything else.
    state = fresh_state(on_mana_burn=None)
    state.players[0].mana_pool = {"G": 3}
    _empty_mana_pools(state)
    assert state.players[0].mana_mistake_burn == 0

    # The hook receives the RIGHT player_idx even when it's the non-active
    # player's pool being cleared, and state.mana_pool (the active-player
    # proxy) still resolves correctly if the hook flips active_idx itself
    # (the real rl.train hook does this via drl_env._for_player).
    seen = []
    def hook(s, idx):
        seen.append((idx, dict(s.players[idx].mana_pool)))
        return False
    state = fresh_state(on_mana_burn=hook)
    state.active_idx = 0
    state.players[1].mana_pool = {"U": 1}  # the NON-active player's own pool
    _empty_mana_pools(state)
    assert seen == [(1, {"U": 1})]
    assert state.players[1].mana_mistake_burn == 1


def test_mana_burnt_this_turn_unconditional():
    # PlayerState.mana_burnt_this_turn (rl.rewards.with_dense_mana_burn_penalty's
    # input, 2026-08) is DELIBERATELY unconditional, unlike mana_mistake_burn
    # -- it tallies every burnt pip even when cost_paid_this_phase or
    # triggers_fired_this_phase would exempt mana_mistake_burn. Same
    # direct-call style as test_mana_mistake_burn_three_way_exemption above
    # (pure bookkeeping over already-set flags, no real turn needed).
    from game.turn import _empty_mana_pools

    state = GameState(on_the_play=True, players=[PlayerState(True), PlayerState(False)])
    state.players[0].mana_pool = {"R": 2}
    state.players[0].cost_paid_this_phase = True  # would exempt mana_mistake_burn
    _empty_mana_pools(state)
    assert state.players[0].mana_mistake_burn == 0  # exempted, as before
    assert state.players[0].mana_burnt_this_turn == 2  # NOT exempted -- unconditional like mana_burnt_total

    # Accumulates across multiple phase-boundary clears within the same
    # (simulated) turn -- _run_turn_gen resets it only at turn start, not
    # per-phase, so several small burns in one turn compound.
    state.players[0].mana_pool = {"U": 1}
    _empty_mana_pools(state)
    assert state.players[0].mana_burnt_this_turn == 3


def test_mana_burnt_this_turn_single_pip_per_pip_attribution():
    # rl.rewards.with_dense_mana_burn_penalty's actual input (2026-08,
    # replacing the earlier whole-phase unmetered_mana_tapped_this_phase
    # exclusion -- see PlayerState.mana_pool_single_pip's own docstring for
    # the dynamic len(produced)==1 tag rule and game.mana.spend_one_pip's
    # untagged-first spend-order convention): sums whatever remains TAGGED
    # in mana_pool_single_pip at the moment of the burn, per pip, not a
    # whole-phase exclusion.
    from game.turn import _empty_mana_pools

    # (a) A single-pip source (Mountain): tagged, and burning it counts.
    state = GameState(on_the_play=True, players=[PlayerState(True), PlayerState(False)])
    state.active_idx = 0
    mountain = Permanent(registry.CARD_DEFS["Mountain"])
    state.players[0].battlefield = [mountain]
    mana.activate_mana_source(state, mountain)
    assert state.players[0].mana_pool_single_pip == {"R": 1}
    _empty_mana_pools(state)
    assert state.players[0].mana_burnt_this_turn == 1
    assert state.players[0].mana_burnt_this_turn_single_pip == 1
    assert state.players[0].mana_pool_single_pip == {}  # cleared alongside mana_pool

    # (b) Priest of Titania's burst, WITH another Elf out (a genuine
    # multi-symbol event): never tagged, so burning it never counts,
    # regardless of size.
    state = GameState(on_the_play=True, players=[PlayerState(True), PlayerState(False)])
    state.active_idx = 0
    priest = Permanent(registry.CARD_DEFS["Priest of Titania"])
    elves = Permanent(registry.CARD_DEFS["Llanowar Elves"])
    state.players[0].battlefield = [priest, elves]
    mana.activate_mana_source(state, priest)  # 2 G: itself + the other Elf
    assert state.players[0].mana_pool == {"G": 2}
    assert state.players[0].mana_pool_single_pip == {}
    _empty_mana_pools(state)
    assert state.players[0].mana_burnt_this_turn == 2
    assert state.players[0].mana_burnt_this_turn_single_pip == 0

    # (b2) Confirmed edge case (owner-approved): Priest ALONE, no other
    # Elves out, resolves to exactly 1 G -- a genuine 1-symbol event, so IS
    # tagged, same as any other single-pip source. Not a bug to guard
    # against.
    state = GameState(on_the_play=True, players=[PlayerState(True), PlayerState(False)])
    state.active_idx = 0
    priest = Permanent(registry.CARD_DEFS["Priest of Titania"])
    state.players[0].battlefield = [priest]
    mana.activate_mana_source(state, priest)  # 1 G: just itself
    assert state.players[0].mana_pool_single_pip == {"G": 1}
    _empty_mana_pools(state)
    assert state.players[0].mana_burnt_this_turn_single_pip == 1

    # (c) MIXED phase, the case the old whole-phase mechanism got wrong:
    # Priest's multi-symbol burst AND a Mountain both tapped and burnt the
    # same phase -- only the Mountain's 1 tagged pip is counted, not 0
    # (the old exclusion) and not both pips either.
    state = GameState(on_the_play=True, players=[PlayerState(True), PlayerState(False)])
    state.active_idx = 0
    priest = Permanent(registry.CARD_DEFS["Priest of Titania"])
    elves = Permanent(registry.CARD_DEFS["Llanowar Elves"])
    mountain = Permanent(registry.CARD_DEFS["Mountain"])
    state.players[0].battlefield = [priest, elves, mountain]
    mana.activate_mana_source(state, priest)   # 2 G (untagged)
    mana.activate_mana_source(state, mountain)  # 1 R (tagged)
    _empty_mana_pools(state)
    assert state.players[0].mana_burnt_this_turn == 3
    assert state.players[0].mana_burnt_this_turn_single_pip == 1  # only the Mountain's pip

    # (d) Spend-order scenario: Priest's burst (untagged, 3 G from itself +
    # 2 other Elves) plus 2 avoidable single-pip taps (tagged) float
    # together; spending exactly what Priest's burst alone would have
    # covered consumes the untagged pips first, so the 2 avoidable tagged
    # pips are what's left floating and correctly blamed.
    state = GameState(on_the_play=True, players=[PlayerState(True), PlayerState(False)])
    state.active_idx = 0
    mana.float_mana(state, ["G", "G", "G"])  # simulates Priest's 3-symbol burst -- untagged
    mana.float_mana(state, ["G"])  # 2 separate single-pip taps -- tagged
    mana.float_mana(state, ["G"])
    assert state.players[0].mana_pool == {"G": 5}
    assert state.players[0].mana_pool_single_pip == {"G": 2}
    for _ in range(3):  # spend exactly what Priest's burst alone covered
        mana.spend_one_pip(state, "G")
    assert state.players[0].mana_pool == {"G": 2}
    assert state.players[0].mana_pool_single_pip == {"G": 2}  # untouched -- untagged spent first
    _empty_mana_pools(state)
    assert state.players[0].mana_burnt_this_turn == 2
    assert state.players[0].mana_burnt_this_turn_single_pip == 2  # both avoidable taps blamed


def test_mana_burn_state_resets_at_turn_boundary():
    # _run_turn_gen resets all three per-turn mana-burn tracking fields
    # (mana_burnt_this_turn, mana_burnt_this_turn_single_pip, and
    # mana_burn_penalty_credited) for BOTH players at the start of every new
    # turn. run_turn (not run_multiplayer_game) drives exactly ONE turn on an
    # already-built state directly -- no pregame mulligan phase to drive, no
    # second player's own turn to script -- so a plain "always pass"
    # choose_action is safe for the whole turn (a fresh opening hand has
    # nothing forcing a real decision: no lands played yet, nothing drawn
    # beyond the norm, no cleanup discard). The accumulation side of these
    # counters is covered elsewhere; this test is specifically about the
    # RESET, for all three fields, for both players.
    state = new_multiplayer_game_state(
        decklists=[[("Mountain", 20)], [("Mountain", 20)]],
        starting_player_idx=0, rng=random.Random(0),
    )
    # Fabricated leftover values (however they'd really accumulate is the
    # accumulation tests' job) -- the only question here is whether
    # _run_turn_gen's own turn-start block zeroes them for BOTH players.
    state.players[0].mana_burnt_this_turn = 5
    state.players[0].mana_burnt_this_turn_single_pip = 5
    state.players[0].mana_burn_penalty_credited = 0.3
    state.players[1].mana_burnt_this_turn = 2
    state.players[1].mana_burnt_this_turn_single_pip = 2
    state.players[1].mana_burn_penalty_credited = 0.1

    seen_at_first_yield = {}

    def _pass_and_snoop(state):
        if not seen_at_first_yield:
            seen_at_first_yield["p0"] = state.players[0].mana_burnt_this_turn
            seen_at_first_yield["p0_single_pip"] = state.players[0].mana_burnt_this_turn_single_pip
            seen_at_first_yield["p0_credited"] = state.players[0].mana_burn_penalty_credited
            seen_at_first_yield["p1"] = state.players[1].mana_burnt_this_turn
            seen_at_first_yield["p1_single_pip"] = state.players[1].mana_burnt_this_turn_single_pip
            seen_at_first_yield["p1_credited"] = state.players[1].mana_burn_penalty_credited
        return None  # always pass -- nothing to play from a bare opening hand

    run_turn(state, choose_action=_pass_and_snoop, combat_enabled=False)
    assert seen_at_first_yield == {
        "p0": 0, "p0_single_pip": 0, "p0_credited": 0.0,
        "p1": 0, "p1_single_pip": 0, "p1_credited": 0.0,
    }, (
        "both players' per-turn mana-burn tracking must already be reset by the very first "
        "decision point of the new turn, for both players, not just the turn owner"
    )


def test_mana_floated_in_real_end_step_swept_same_turn():
    # Fix (2026-08-10): Phase.END now runs a REAL end-step priority round
    # (rule 513) BEFORE cleanup, instead of cleanup_step firing immediately
    # on phase entry with no priority window at all. Before this fix, mana
    # floated here would persist unswept until the FOLLOWING turn's own
    # UNTAP entry -- by which point that turn's own
    # mana_burnt_this_turn_single_pip counter had already reset (see
    # test_mana_burn_state_resets_at_turn_boundary above),
    # silently dropping the burn from the dense reward signal entirely
    # (this engine has no interrupt window, so the player who just
    # finished their turn never gets another decision before that reset
    # fires). Proves the float is now swept and counted for the SAME turn.
    state = new_multiplayer_game_state(
        decklists=[[("Mountain", 20)], [("Mountain", 20)]],
        starting_player_idx=0, rng=random.Random(0),
    )
    mountain = Permanent(registry.CARD_DEFS["Mountain"])
    state.players[0].battlefield = [mountain]

    floated = {"done": False}

    def _float_once_in_end_step_then_pass(state):
        if state.pending_resolution is not None:
            name = resolution.discard_options(state)[0]
            return lambda: resolution.execute_discard_option(state, name)
        if not floated["done"] and state.phase is Phase.END and not state.in_cleanup:
            floated["done"] = True
            return lambda: mana.activate_mana_source(state, mountain)
        return None  # always pass otherwise -- nothing else to play from a bare opening hand

    run_turn(state, choose_action=_float_once_in_end_step_then_pass, combat_enabled=False)
    assert floated["done"]
    assert state.players[0].mana_burnt_this_turn_single_pip == 1
    assert state.players[0].mana_pool_single_pip == {}  # cleared alongside mana_pool


def test_speculative_mana_ability_is_main_phase_only():
    # AUTHORIZED SIMPLIFICATION (owner-approved 2026-08-17, see
    # drl_env._actions_mana._mana_timing_legal): SPECULATIVE floating -- a mana
    # ability activated with no payment open -- is restricted to the active
    # player's own main phase. Real 605.1a/605.3b allows it in any priority
    # window. Payment-time activation (601.2f) stays legal in every phase and is
    # covered by test_mana_ability_legal_mid_pay_unless.
    #
    # This REPLACES the old test_mana_ability_illegal_during_cleanup and its own
    # separate 2026-08-10 simplification (mana abilities revoked while
    # state.in_cleanup, so floated mana could not dodge the burn signal there).
    # Cleanup is not a main phase and no payment is open in it, so the new rule
    # already excludes it -- the flag check is gone, one deviation fewer, and the
    # cleanup case is asserted below as a consequence rather than a special case.
    state = GameState(on_the_play=True, players=[PlayerState(True), PlayerState(False)])
    mountain = Permanent(registry.CARD_DEFS["Mountain"])
    state.battlefield = [mountain]
    legal = _mana_ability_legal("Mountain", None)

    # Ground truth (game.mana_ability_options) says the source is available in
    # every case below -- only the drl_env-facing timing gate differs, which is
    # what makes this a restriction on the ACTION SPACE, not on the engine.
    for phase in (Phase.MAIN1, Phase.MAIN2):
        state.phase = phase
        assert legal(state), f"speculative float must be legal in {phase}"
        assert ("Mountain", None) in mana.mana_ability_options(state)

    for phase in (Phase.UPKEEP, Phase.DRAW, Phase.DECLARE_ATTACKERS,
                  Phase.COMBAT_DAMAGE, Phase.END):
        state.phase = phase
        assert not legal(state), f"speculative float must be illegal in {phase}"
        assert ("Mountain", None) in mana.mana_ability_options(state), "source availability is unchanged"

    # Not the opponent's main phase either -- "the ACTIVE player's own main
    # phase" is turn ownership, not merely the phase.
    state.phase = Phase.MAIN1
    state.active_idx = 1
    assert not legal(state), "the non-turn player may not speculatively float in the turn player's main phase"

    # Cleanup, now a consequence of the phase rule rather than its own flag.
    state.active_idx = 0
    state.in_cleanup = True
    state.phase = Phase.END
    assert not legal(state)


def test_madness_decision_from_forced_cleanup_discard_still_resolves():
    # A card discarded via cleanup's own forced hand-size discard queues a
    # Madness cast-or-graveyard decision exactly like any other discard
    # (game.resolution.handlers_library._discard_one, "regardless of why
    # the card was discarded"). state.in_cleanup blocks every OTHER action
    # (mana abilities included) for the whole cleanup window, but the
    # existing trigger_queue-driven extra priority round (rule 514.3a) is
    # deliberately KEPT specifically so this decision still gets a chance
    # to resolve -- see _run_turn_gen's own Phase.END handling. Declines
    # the cast (no black mana available) -- this test is about the decision
    # reaching a real pending_resolution at all, not about completing a
    # cast (see test_kitchen_imp_madness_cast_from_discard for that).
    state = new_multiplayer_game_state(
        decklists=[[("Mountain", 20)], [("Mountain", 20)]],
        starting_player_idx=0, rng=random.Random(0),
    )
    state.players[0].hand = [registry.CARD_DEFS["Mountain"]] * 7 + [registry.CARD_DEFS["Kitchen Imp"]]
    assert len(state.players[0].hand) == 8  # 1 over HAND_SIZE_LIMIT -- cleanup must force exactly 1 discard

    madness_decision_seen = {"done": False}

    def _discard_kitchen_imp_then_decline_madness(state):
        pending = state.pending_resolution
        if pending is not None and pending["kind"] == "discard":
            return lambda: resolution.execute_discard_option(state, "Kitchen Imp")
        if pending is not None and pending["kind"] == "madness_decision":
            madness_decision_seen["done"] = True
            return lambda: execute_madness_decline(state)
        return None  # always pass otherwise

    run_turn(state, choose_action=_discard_kitchen_imp_then_decline_madness, combat_enabled=False)
    assert madness_decision_seen["done"]
    assert not any(c.name == "Kitchen Imp" for c in state.players[0].hand)
    assert any(c.name == "Kitchen Imp" for c in state.players[0].graveyard)  # declined -> exile to graveyard


def test_on_mana_burn_hook_wired_through_run_multiplayer_game():
    # Integration check that run_multiplayer_game's on_mana_burn kwarg really
    # reaches _empty_mana_pools -- test_mana_mistake_burn_three_way_exemption
    # above only exercises the conditional logic directly, not this wiring.
    # Mono-Mountain deck, no spells at all: play a land, tap it once in
    # MAIN1, then always Pass -- nothing is ever paid for or triggered, so
    # the float is a live candidate mistake the moment it clears.
    tapped_once = []
    calls = []

    def _play_then_tap_then_pass(state):
        if state.pending_resolution is not None and state.pending_resolution["kind"] == "mulligan_decision":
            return lambda: resolution.execute_mulligan_keep(state)  # both players always keep
        if state.active_idx != 0:
            return None
        if state.lands_played_this_turn == 0 and any(c.name == "Mountain" for c in state.hand):
            return lambda: play_land_from_hand(state, registry.CARD_DEFS["Mountain"])
        if not tapped_once:
            mtn = next((p for p in state.battlefield if p.card_def.name == "Mountain" and not p.tapped), None)
            if mtn is not None:
                tapped_once.append(True)
                return lambda: mana.activate_mana_source(state, mtn)
        return None

    def _hook(state, player_idx):
        calls.append(player_idx)
        return False  # a mono-land deck has nothing to cast, ever

    state = run_multiplayer_game(
        decklists=[[("Mountain", 20)], [("Mountain", 20)]],
        rng=random.Random(0), starting_player_idx=0,
        choose_action=_play_then_tap_then_pass, horizon=1, on_mana_burn=_hook,
    )
    assert tapped_once  # the tap actually happened
    assert 0 in calls  # the hook was actually consulted for player 0's own burn
    assert state.players[0].mana_mistake_burn == 1


def test_cast_then_pay_end_to_end_with_an_empty_pool():
    # The headline sequence, through the REAL action table rather than direct
    # engine calls: with NOTHING floating and two untapped Mountains, casting
    # Lightning Bolt is already legal (601.2e says the cost is known before any
    # mana is produced), the payment window is where the Mountain gets tapped
    # (601.2f), and the pool spend finishes it (601.2g).
    #
    # Under float-first this exact position was ILLEGAL -- the agent had to
    # guess and float first, before knowing what it was paying for. That guess
    # is the misplay this whole change removes.
    from drl_env import build_action_table, legal_action_mask

    actions = build_action_table([("Mountain", 4), ("Lightning Bolt", 2)], registry.EFFECT_REGISTRY)
    cast_idx = next(i for i, (n, _l, _e) in enumerate(actions) if n == "Cast Lightning Bolt")
    tap_idx = next(i for i, (n, _l, _e) in enumerate(actions) if n == "Tap Mountain")

    state = GameState(on_the_play=True, players=[PlayerState(True), PlayerState(False)])
    state.phase = Phase.MAIN1
    state.battlefield = [Permanent(registry.CARD_DEFS["Mountain"]),
                         Permanent(registry.CARD_DEFS["Mountain"])]
    state.hand = [registry.CARD_DEFS["Lightning Bolt"]]
    assert state.mana_pool == {}

    assert legal_action_mask(state, actions)[cast_idx], "castable off untapped lands, nothing floating"
    actions[cast_idx][2](state)
    assert state.pending_resolution["kind"] == "pay_cost"
    assert state.pending_resolution["remaining"] == {"R": 1}
    assert state.pending_resolution["announced"] == {"R": 1}, "announced is stashed for the strand bug report"

    # 601.2f: the mana ability is legal DURING the payment -- that is the window
    # the mana is produced in now.
    assert legal_action_mask(state, actions)[tap_idx]
    actions[tap_idx][2](state)
    assert state.mana_pool == {"R": 1}

    # 601.2g: spend it. Nothing reached the stack before this point (601.2i).
    assert not state.stack, "nothing reaches the stack until the cost is paid"
    mana.execute_pool_spend(state, "R")
    assert state.mana_pool == {}
    # The payment is over; Bolt's own "any target" choice follows it (this
    # engine settles that after the cost rather than at 601.2c -- a separate
    # ordering question this test deliberately does not assert on). What matters
    # here is that pay_cost is done and it was driven entirely by tapping.
    assert state.pending_resolution is not None
    assert state.pending_resolution["kind"] != "pay_cost", "the payment completed"


def test_opponent_never_acts_inside_the_cast_window():
    # CR 601.2 is atomic: no player receives priority between announcing a spell
    # and it becoming cast, so the opponent cannot respond until it is ON THE
    # STACK (601.2i, then 117.3c). Cast-then-pay makes that window much longer
    # -- it now contains every tap -- so the invariant is pinned explicitly
    # rather than left implicit.
    #
    # Two mechanisms hold it, and this asserts both: a player who takes an
    # ACTION keeps priority (_priority_round only flips active_idx on a Pass),
    # and Pass is illegal while a resolution is pending
    # (drl_env._pass_legal._pending_gate).
    from drl_env import build_action_table, legal_action_mask

    actions = build_action_table([("Mountain", 4), ("Lightning Bolt", 2)], registry.EFFECT_REGISTRY)
    cast_idx = next(i for i, (n, _l, _e) in enumerate(actions) if n == "Cast Lightning Bolt")
    pass_idx = next(i for i, (n, _l, _e) in enumerate(actions) if n == "Pass")

    state = GameState(on_the_play=True, players=[PlayerState(True), PlayerState(False)])
    state.phase = Phase.MAIN1
    state.battlefield = [Permanent(registry.CARD_DEFS["Mountain"])]
    state.hand = [registry.CARD_DEFS["Lightning Bolt"]]
    caster = state.active_idx

    actions[cast_idx][2](state)
    assert state.pending_resolution["kind"] == "pay_cost"
    assert state.active_idx == caster, "announcing must not hand priority to anyone"
    assert not legal_action_mask(state, actions)[pass_idx], (
        "Pass is illegal mid-payment, which is what stops the phase -- and priority -- "
        "advancing out from under an in-flight cast"
    )

    # Take the whole payment; priority never leaves the caster at any point.
    tap_idx = next(i for i, (n, _l, _e) in enumerate(actions) if n == "Tap Mountain")
    actions[tap_idx][2](state)
    assert state.active_idx == caster
    assert not legal_action_mask(state, actions)[pass_idx], "still sealed mid-payment"
    mana.execute_pool_spend(state, "R")
    assert state.active_idx == caster

    # Bolt's own target choice follows the payment, and Pass stays illegal
    # through it too -- the cast is still in flight, so the opponent still
    # cannot act. Priority reopens only once nothing is pending at all.
    assert state.pending_resolution is not None
    assert not legal_action_mask(state, actions)[pass_idx]
    assert state.active_idx == caster, "priority never left the caster for the whole cast"
