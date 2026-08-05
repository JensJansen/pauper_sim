"""Tests for game.turn: the 2-player engine end to end -- turn alternation,
on_the_play/turns_taken, life_total loss, deck-out-as-instant-loss, a real
game played through actual cards (not a synthetic no-op deck), the
_declare_blockers_gen flip-drive-restore shape, the turn-owner vs
priority-holder split in speed_legal, and rule 500.4's per-phase-boundary
mana clear."""

import random

from game import mana, registry, resolution
from game.cards import CardDef, CardType, EffectId
from game.effects.casting import play_land_from_hand
from game.effects.stack import push_to_stack
from game.effects.win_check import deal_damage_to_opponent
from game.state import GameState, Permanent, PlayerState
from game.turn import (
    Phase, Speed, _declare_blockers_gen, new_multiplayer_game_state,
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
            # Float-first: a tap floats mana into the pool via a top-level mana
            # ability BEFORE the cast (below); during payment there is only pool
            # mana to spend.
            color = mana.pool_spend_options(state)[0]
            return lambda: mana.execute_pool_spend(state, color)
        if state.lands_played_this_turn == 0 and any(c.name == "Mountain" for c in state.hand):
            return lambda: play_land_from_hand(state, registry.CARD_DEFS["Mountain"])
        if _bolts_available(state) > 0:
            if mana.plan_payment(state, bolt_def.cast_cost) is not None:
                def _cast_bolt():
                    mana.begin_pay_cost(state, bolt_def.cast_cost, on_complete=lambda s: push_to_stack(s, bolt_def, bolt_resolve))
                return _cast_bolt
            # Can't pay yet -- float {R} from an untapped Mountain first (float-first).
            mtn = next((p for p in state.battlefield if p.card_def.name == "Mountain" and not p.tapped), None)
            if mtn is not None:
                return lambda: mana.activate_mana_source(state, mtn)
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
    # genuine zero-legal-action deadlock, not merely an unproductive one
    # (unlike the ordinary action-cap-exhaustion path the sibling test
    # above covers, where some action is always resubmittable). This
    # proves the generator detects that exact dead end proactively and
    # abandons immediately instead of hanging -- the OUTCOME (a lone
    # menace block gets dropped) is enforce_menace's job at combat damage,
    # a separate call site already covered by
    # test_menace_block_incomplete_and_enforce_menace in
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


def test_rule_500_4_mana_pool_clears_at_phase_boundaries_except_combat():
    # Rule 500.4: unused mana empties at every step/phase boundary for BOTH
    # players (not just whoever floated it), EXCEPT across combat's own
    # three sub-phases -- an authored simplification (see _COMBAT_PHASES'
    # own comment in turn.py) that treats DECLARE_ATTACKERS/DECLARE_BLOCKERS/
    # COMBAT_DAMAGE as one shared mana window. Two pre-placed, already-
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
    assert by_phase[Phase.DECLARE_BLOCKERS][0] == {"R": 1}  # DECLARE_ATTACKERS' float survives -- combat's shared window
    assert by_phase[Phase.COMBAT_DAMAGE][0] == {"R": 1}  # still there crossing DECLARE_BLOCKERS -> COMBAT_DAMAGE too
    assert by_phase[Phase.MAIN2] == ({}, {})  # cleared again once combat's shared window ends
    assert floated_at == {"main1", "declare_attackers"}  # both floats actually happened, not skipped
    # mana_burnt_total tallies pips lost, not clear-events: player 0 lost the
    # MAIN1 "R":1 AND the DECLARE_ATTACKERS "R":1 (two separate clears) -> 2;
    # player 1 lost only its MAIN1 "W":1 -> 1. Diagnostic-only (logging/viz) --
    # no reward function reads mana_burnt_total; see PlayerState.mana_mistake_burn
    # for the reward-facing, conditional signal.
    assert state.players[0].mana_burnt_total == 2
    assert state.players[1].mana_burnt_total == 1


def test_mana_mistake_burn_three_way_exemption():
    # _empty_mana_pools's dense reward-facing tally (PlayerState.
    # mana_mistake_burn, rl.rewards.with_mana_mistake_penalty): only counts a
    # burn once cost_paid_this_phase, triggers_fired_this_phase, AND the
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
