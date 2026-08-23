"""Tests for rl.decision.action_bridge: the fixed-table/pointer-head action space bridge to the game engine."""
import json
import random

import pytest

import drl_env
import game
from game.state import CardInstance, GameState, Permanent, PlayerState
from rl.decision.action_bridge import (
    _TARGETING_PREFIXES,
    any_pointer_legal,
    build_fixed_action_table,
    execute_pointer_choice,
    pointer_legal_mask,
)
from rl.model.features import SLOTS_AFTER_TARGETING
from rl.model.features import CardVocab, build_token_set


def _mono_red_fixture():
    """The mono_red_madness decklist/fixed-table/vocab fixture most of these
    checks share -- Blood/Robot are the two tokens the deck creates mid-game."""
    decklist = game.parse_decklist_file("../data/mono_red_madness.txt")
    # mono_red_madness creates both of these mid-game (Blood from madness
    # discards, Robot from Melded Moxite's sacrifice ability); the deck
    # config's own "token_card_defs" only lists "Blood", so this fixture
    # passes the complete set explicitly.
    token_defs = (game.BLOOD_TOKEN_CARD_DEF, game.ROBOT_TOKEN_CARD_DEF)
    fixed_table = build_fixed_action_table(decklist, token_card_defs=token_defs)
    vocab = CardVocab([decklist], token_card_defs=token_defs)
    return decklist, token_defs, fixed_table, vocab


@pytest.mark.slow
def test_build_fixed_action_table_basic():
    _decklist, _token_defs, fixed_table, _vocab = _mono_red_fixture()
    fixed_names = [name for name, _l, _e in fixed_table]
    assert not any(name.startswith(_TARGETING_PREFIXES) for name in fixed_names), (
        "fixed table must contain zero targeting actions"
    )
    # NO-UNDO POLICY tripwire (todo/no_undo_policy.md): the engine must never
    # expose an action letting the agent reconsider/reverse an earlier
    # commitment -- an "Unassign Blocker" action would let the agent cycle
    # assign/unassign indefinitely, and no iteration cap exists anywhere in
    # the turn loop to stop it. Hardcodes the literal substring independently
    # of any prefix-filter tuple, so a future reintroduction fails loudly here.
    assert not any("nassign" in name for name in fixed_names), (
        "no action may let the agent reconsider/reverse an earlier commitment -- see todo/no_undo_policy.md"
    )
    assert "Pass" in fixed_names
    assert any(name.startswith("Play land:") for name in fixed_names)

    # DFC back faces (Delver of Secrets -> Insectile Aberration) must get
    # their own "Choose: X" row, same as a token name -- a transformed
    # permanent's card_def swaps to the back face, so any by-name resolution
    # (order_triggers, sacrifice, discard, search_fetch, ...) can need to
    # reference it.
    dfc_decklist = game.parse_decklist_file("../data/mono_blue_terror.txt")
    dfc_table = build_fixed_action_table(dfc_decklist)
    dfc_names = [name for name, _l, _e in dfc_table]
    assert "Choose: Delver of Secrets" in dfc_names and "Choose: Insectile Aberration" in dfc_names, (
        "both DFC faces need their own 'Choose: X' row -- got: "
        f"{[n for n in dfc_names if 'Delver' in n or 'Insectile' in n]}"
    )
    assert any(name.startswith("Cast ") for name in fixed_names)


@pytest.mark.slow
def test_pointer_legal_mask_real_game_attack_block_choose_opponent():
    # pointer_legal_mask / execute_pointer_choice against a real 2-player
    # game (build_token_set always indexes an opponent seat, so a 1-player
    # game state doesn't fit its contract). Mirror match, one shared
    # choose_action dispatching on state.active_idx. Always prefers a
    # pointer action when one is legal, else falls back to the fixed
    # table's "prefer Pass, else first legal" rule -- exercises attacking,
    # blocking, and the cross-player "Choose opponent's" consult nested
    # inside blocking.
    decklist, _token_defs, fixed_table, vocab = _mono_red_fixture()
    pass_action = next(i for i, (name, _l, _e) in enumerate(fixed_table) if name == "Pass")
    saw_attack, saw_block, saw_choose_opponent = [False], [False], [False]

    def make_choose_action(rng):
        def choose_action(state):
            seat = state.active_idx
            pend = state.pending_resolution
            if pend is not None and pend["kind"] in ("mulligan_decision", "mulligan_bottom"):
                # Pregame decisions aren't in the fixed table -- owned by the
                # mulligan model in real play. Keep every hand so this test's
                # random play still reaches real turns.
                return lambda state=state: game.execute_mulligan_keep(state)
            if any_pointer_legal(state):
                tokens = build_token_set(state, seat, vocab)
                identities_row = [ident for _idx, _row, ident in tokens]
                mask = pointer_legal_mask(state, identities_row)
                legal = [p for p, ok in zip(identities_row, mask) if ok]
                if legal:
                    chosen = rng.choice(legal)  # randomize which target too, not always the first -- more varied lines
                    pending = state.pending_resolution
                    if pending is None:
                        saw_attack[0] = True
                    elif pending["kind"] == "declare_blockers":
                        saw_block[0] = True
                    elif pending["kind"] == "choose_opponent_permanent":
                        saw_choose_opponent[0] = True
                    return lambda state=state, chosen=chosen: execute_pointer_choice(state, chosen)

            # Prefer any non-Pass legal action (randomly) so the game
            # actually develops a board; Pass is the fallback of last resort.
            #
            # Legality via one drl_env.legal_action_mask sweep, never by
            # calling each action's own legal_fn independently: drl_env's
            # sweep-scoped caches are only valid for the duration of one
            # sweep.
            mask = drl_env.legal_action_mask(state, fixed_table)
            non_pass_indices = [i for i, ok in enumerate(mask) if ok and i != pass_action]
            if non_pass_indices:
                i = rng.choice(non_pass_indices)
                execute_fn = fixed_table[i][2]
                return lambda state=state, execute_fn=execute_fn: execute_fn(state)
            if mask[pass_action]:
                return None
            raise AssertionError("no fixed action legal and no pointer target legal -- the model would be stuck")
        return choose_action

    # Random unguided play doesn't reliably reach every combat sub-case in
    # any one game, so retry across several independent games.
    for seed in range(10):
        if saw_attack[0] and saw_block[0] and saw_choose_opponent[0]:
            break
        game.run_multiplayer_game(
            decklists=[decklist, decklist],
            rng=random.Random(seed), starting_player_idx=seed % 2, choose_action=make_choose_action(random.Random(seed)),
            horizon=40, combat_enabled=True,
        )
    assert saw_attack[0], "expected at least one real declare-attackers window across 10 real 2p games"
    assert saw_block[0], "expected at least one real declare-blockers window across 10 real 2p games"
    assert saw_choose_opponent[0], "expected at least one real cross-player 'Choose opponent's' consult across 10 real 2p games"


@pytest.mark.slow
def test_no_undo_blocking_regression():
    # NO-UNDO regression (todo/no_undo_policy.md): a committed blocker must
    # never be offered as a pointer target again -- only "add more blockers"
    # or "Done blocking" remain legal. Hand-built to construct the
    # already-committed state and assert the negative. blocked_by/attackers
    # are keyed by the attacking player's own permanents, so the "opponent"
    # PlayerState here is the attacker's side even though the deciding seat
    # (0) is the blocker's.
    _decklist, _token_defs, _fixed_table, vocab = _mono_red_fixture()
    ub_me = PlayerState(on_the_play=True)
    ub_opp = PlayerState(on_the_play=False)
    bogle = Permanent(game.CARD_DEFS["Slippery Bogle"])
    attacker = Permanent(game.CARD_DEFS["Slippery Bogle"])
    ub_me.battlefield = [bogle]
    ub_opp.attackers = [attacker]
    ub_opp.blocked_by = {attacker: [bogle]}
    ub_state = GameState(on_the_play=True, players=[ub_me, ub_opp])
    ub_state.pending_resolution = {"kind": "declare_blockers", "on_complete": lambda s: None}
    ub_ids = [ident for _i, _r, ident in build_token_set(ub_state, 0, vocab)]
    ub_legal = [r for r, ok in zip(ub_ids, pointer_legal_mask(ub_state, ub_ids)) if ok]
    assert bogle not in ub_legal, "a committed blocker must NEVER be reachable again -- no undo, see todo/no_undo_policy.md"


@pytest.mark.slow
def test_saruli_mana_subdecision_pointer():
    # Saruli Caretaker mana_subdecision -- pointer mechanics (choose_target
    # stage): domain is my own battlefield, excludes the source by identity
    # (not name -- two same-named Sarulis are legitimately valid targets of
    # each other), excludes already-tapped creatures, applies the source's
    # own mana_extra_choose predicate. This is specifically the pointer half.
    _decklist, _token_defs, _fixed_table, vocab = _mono_red_fixture()
    sd_me = PlayerState(on_the_play=True)
    sd_opp = PlayerState(on_the_play=False)
    saruli_a = Permanent(game.CARD_DEFS["Saruli Caretaker"])
    saruli_b = Permanent(game.CARD_DEFS["Saruli Caretaker"])
    saruli_b.slot = 2
    other = Permanent(game.CARD_DEFS["Slippery Bogle"])
    tapped_other = Permanent(game.CARD_DEFS["Slippery Bogle"])
    tapped_other.slot = 2
    tapped_other.tapped = True
    sd_me.battlefield = [saruli_a, saruli_b, other, tapped_other]
    sd_state = GameState(on_the_play=True, players=[sd_me, sd_opp])
    extra_pred = game.EFFECT_REGISTRY[saruli_a.card_def.effect_id]["mana_extra_choose"]
    game.begin_mana_subdecision(sd_state, saruli_a, extra_pred)
    sd_ids = [ident for _i, _r, ident in build_token_set(sd_state, 0, vocab)]
    sd_legal = [r for r, ok in zip(sd_ids, pointer_legal_mask(sd_state, sd_ids)) if ok]
    assert set(sd_legal) == {saruli_b, other}, (
        f"expected the OTHER Saruli + the untapped Bogle, excluding self (identity) and the tapped Bogle, got {sd_legal}"
    )
    execute_pointer_choice(sd_state, saruli_b)  # tap a DIFFERENT Saruli to pay for this one's ability -- legal
    assert sd_state.mana_subdecision["stage"] == "choose_color" and sd_state.mana_subdecision["target"] is saruli_b


@pytest.mark.slow
def test_saruli_mid_pay_unless():
    # Quirion Ranger targets an opponent's Ward creature -> pay_unless opens
    # with the Saruli player as payer -> they activate Saruli (both
    # mana_subdecision stages) while pay_unless is still open -> control
    # returns to pay_unless, untouched, once mana_subdecision closes. Why
    # mana_subdecision is its own field: reusing pending_resolution's own
    # building blocks would wrongly mask mana abilities mid-any-resolution.
    _decklist, _token_defs, _fixed_table, vocab = _mono_red_fixture()
    pu_me = PlayerState(on_the_play=True)
    pu_opp = PlayerState(on_the_play=False)
    pu_saruli = Permanent(game.CARD_DEFS["Saruli Caretaker"])
    pu_saruli.summoning_sick = False  # Saruli's own {T} ability needs this (302.6)
    pu_other = Permanent(game.CARD_DEFS["Slippery Bogle"])
    pu_me.battlefield = [pu_saruli, pu_other]
    pu_state = GameState(on_the_play=True, players=[pu_me, pu_opp])
    pu_result = []
    game.begin_pay_unless(pu_state, payer_idx=0, cost={"generic": 1}, on_result=lambda s, paid: pu_result.append(paid))
    assert pu_state.pending_resolution["kind"] == "pay_unless"

    pu_decklist = [("Saruli Caretaker", 2), ("Slippery Bogle", 2)]
    pu_table = build_fixed_action_table(pu_decklist)
    pu_tap_idx = next(i for i, (n, _l, _e) in enumerate(pu_table) if n == "Tap Saruli Caretaker")
    _, pu_tap_legal, pu_tap_execute = pu_table[pu_tap_idx]
    assert pu_tap_legal(pu_state), "Saruli's mana ability must stay legal mid-pay_unless -- the whole point of this redesign"
    pu_tap_execute(pu_state)
    assert pu_state.pending_resolution["kind"] == "pay_unless", "pay_unless must be UNTOUCHED while the mana_subdecision is open"
    assert pu_state.mana_subdecision["stage"] == "choose_target"

    pu_ids = [ident for _i, _r, ident in build_token_set(pu_state, 0, vocab)]
    pu_legal = [r for r, ok in zip(pu_ids, pointer_legal_mask(pu_state, pu_ids)) if ok]
    assert pu_legal == [pu_other], "only the untapped OTHER creature must be a legal tap-target"
    execute_pointer_choice(pu_state, pu_other)
    assert pu_state.mana_subdecision["stage"] == "choose_color"

    pu_color_idx = next(i for i, (n, _l, _e) in enumerate(pu_table) if n == "Produce G")
    _, pu_color_legal, pu_color_execute = pu_table[pu_color_idx]
    assert pu_color_legal(pu_state)
    pu_color_execute(pu_state)
    assert pu_state.mana_subdecision is None
    assert pu_state.mana_pool == {"G": 1}
    assert pu_state.pending_resolution["kind"] == "pay_unless", (
        "control must return to the untouched pay_unless resolution, not something new"
    )

    pu_pay_idx = next(i for i, (n, _l, _e) in enumerate(pu_table) if n == "Pay (unless)")
    _, pu_pay_legal, pu_pay_execute = pu_table[pu_pay_idx]
    assert pu_pay_legal(pu_state), "the floated G must actually be usable to pay off pay_unless"
    pu_pay_execute(pu_state)
    while pu_state.pending_resolution is not None:
        game.execute_pool_spend(pu_state, game.pool_spend_options(pu_state)[0])
    assert pu_result == [True], "pay_unless must resolve paid=True, using the mana Saruli floated mid-resolution"


@pytest.mark.slow
def test_choose_graveyard_card_pointer_opponent_graveyard():
    # choose_graveyard_card is a pointer target (matched by object identity),
    # not a fixed "Choose: X" row. Hand-build a real pending targeting the
    # opponent's graveyard and confirm the pointer offers exactly that card
    # and executes the pick.
    _decklist, _token_defs, _fixed_table, vocab = _mono_red_fixture()
    gy_me = PlayerState(on_the_play=True)
    gy_opp = PlayerState(on_the_play=False)
    gy_opp.graveyard = [CardInstance(game.CARD_DEFS["Lightning Bolt"])]  # graveyard holds CardInstances
    gy_state = GameState(on_the_play=True, players=[gy_me, gy_opp])
    picked = []
    game.begin_choose_graveyard_card(gy_state, predicate=lambda c: True,
                                     on_complete=lambda st, chosen: picked.append(chosen), graveyard=gy_opp.graveyard)
    assert gy_state.pending_resolution["kind"] == "choose_graveyard_card" and any_pointer_legal(gy_state)
    gy_ids = [ident for _idx, _row, ident in build_token_set(gy_state, 0, vocab)]
    legal_refs = [ref for ref, ok in zip(gy_ids, pointer_legal_mask(gy_state, gy_ids)) if ok]
    assert len(legal_refs) == 1 and legal_refs[0] is gy_opp.graveyard[0], \
        "the opponent's graveyard Lightning Bolt instance must be the sole legal pointer target"
    execute_pointer_choice(gy_state, legal_refs[0])
    assert len(picked) == 1 and picked[0] is gy_opp.graveyard[0] and gy_state.pending_resolution is None, \
        "executing the pointer choice must resolve the graveyard pick to that exact instance and clear the pending"


@pytest.mark.slow
def test_choose_graveyard_card_hand_reveal():
    # Faithful hand reveal (Mesmeric Fiend): the same primitive can pick from
    # a player's hand, which the effect reveals. The whole revealed hand is
    # tokenized so the pointer can address it, but only predicate-legal
    # (nonland) cards are selectable.
    _decklist, _token_defs, _fixed_table, vocab = _mono_red_fixture()
    hr_me = PlayerState(on_the_play=True)
    hr_opp = PlayerState(on_the_play=False)
    hr_opp.hand = [game.CARD_DEFS["Lightning Bolt"], game.CARD_DEFS["Mountain"]]
    hr_state = GameState(on_the_play=True, players=[hr_me, hr_opp])
    hr_picked = []
    game.begin_choose_graveyard_card(hr_state, predicate=lambda c: c.card_type != game.CardType.LAND,
                                     on_complete=lambda st, chosen: hr_picked.append(chosen), graveyard=hr_opp.hand)
    hr_refs = [ident for _idx, _row, ident in build_token_set(hr_state, 0, vocab)]
    # Deferred hand: identities are the CardDefs; this state has no other
    # zones, so every non-None token is one of the two revealed hand cards.
    assert {r.name for r in hr_refs if r is not None} == {"Lightning Bolt", "Mountain"}, \
        "the whole revealed hand must be tokenized"
    hr_legal = [ref for ref, ok in zip(hr_refs, pointer_legal_mask(hr_state, hr_refs)) if ok]
    assert len(hr_legal) == 1 and hr_legal[0] is hr_opp.hand[0], \
        "only the nonland revealed-hand card must be a legal pointer target"
    execute_pointer_choice(hr_state, hr_legal[0])
    assert len(hr_picked) == 1 and hr_picked[0] is hr_opp.hand[0], "executing the reveal-hand pointer choice must resolve to that exact card"


@pytest.mark.slow
def test_stranded_payment_filter_regression():
    # STRANDED-PAYMENT regression: a mana filter is a pool->pool conversion,
    # legal mid-payment, that could otherwise consume the very colored pip an
    # already-begun payment still owes -- leaving remaining={'G': 1} with no
    # green source and no legal action at all (all-False mask), since
    # float-first mana has no "Abandon payment" action.
    # drl_env._actions_mana._filter_would_strand_payment forbids exactly that.
    tron_decklist = game.parse_decklist_file("../data/monster_tron.txt")
    tron_table = drl_env.build_action_table(tron_decklist, game.EFFECT_REGISTRY)
    # "Filter Barrels of Blasting Jelly, paying G" -- the exact action from the crash.
    filt_idx = next(i for i, (n, _l, _e) in enumerate(tron_table)
                    if n == "Filter Barrels of Blasting Jelly, paying G")
    st = GameState(on_the_play=True, players=[PlayerState(True), PlayerState(False)])
    # Speculative floating is the active player's own main phase only; a
    # bare GameState has no phase at all, which would fail that gate before
    # reaching what this test is actually about.
    st.phase = game.turn.Phase.MAIN1
    st.battlefield = [Permanent(game.CARD_DEFS["Barrels of Blasting Jelly"])]
    st.mana_pool.update({"G": 1})
    # With no payment pending, converting the lone G is a perfectly legal choice.
    assert drl_env.legal_action_mask(st, tron_table)[filt_idx], "filter must stay legal with no payment pending"
    # Mid-payment owing exactly {G} with one G floating. Barrels can output
    # any of five colors, one of them G, so the conversion itself stays
    # legal -- what must not be reachable is the specific color choice that
    # strands, one gate later.
    game.begin_pay_cost(st, {"G": 1}, on_complete=lambda s: None)
    assert st.pending_resolution["kind"] == "pay_cost" and st.pending_resolution["remaining"].get("G") == 1
    assert drl_env.legal_action_mask(st, tron_table)[filt_idx], (
        "a conversion with a safe output color available must stay legal"
    )

    # Take it, and check the color stage offers ONLY the color that pays.
    tron_table[filt_idx][2](st)
    assert st.mana_subdecision["stage"] == "choose_color"
    offered = {c for c in game.COLORS
               if drl_env.legal_action_mask(st, tron_table)[
                   next(i for i, (n, _l, _e) in enumerate(tron_table) if n == f"Produce {c}")]}
    assert offered == {"G"}, (
        f"only G can still pay the owed {{G}}; every other output would strand -- got {offered}"
    )
    # Non-empty by construction: the agent is never left with a payment it
    # owes and no legal way to make it.
    assert offered


@pytest.mark.slow
def test_wall_of_roots_tapped_by_other_means_still_a_legal_mana_action():
    # DUPLICATED-GATE regression (2026-08-23, a real production-training
    # crash): Wall of Roots' mana ability has no {T} in its own cost, so
    # game.tappable_for_mana correctly exempts it from the blanket
    # "must be untapped" rule -- but drl_env._actions_mana._find_mana_source
    # used to re-implement that same tapped/summoning-lock gate INLINE
    # instead of calling tappable_for_mana, and had silently drifted out of
    # sync (missing the mana_no_tap exception): game.mana_ability_options
    # correctly listed a tapped Wall of Roots as available, but the actual
    # per-action legal_action_mask entry ("Tap Wall of Roots") stayed
    # illegal regardless -- an in-flight payment relying on it got stranded
    # (an all-False mask, a hard crash) even though the engine's own
    # "what's available" view said it should have worked.
    spy_decklist = game.parse_decklist_file("../data/spy_combo.txt")
    spy_table = drl_env.build_action_table(spy_decklist, game.EFFECT_REGISTRY)
    tap_idx = next(i for i, (n, _l, _e) in enumerate(spy_table) if n == "Tap Wall of Roots")
    st = GameState(on_the_play=True, players=[PlayerState(True), PlayerState(False)])
    st.phase = game.turn.Phase.MAIN1
    wall = Permanent(game.CARD_DEFS["Wall of Roots"])
    wall.tapped = True  # tapped by something else entirely (e.g. Saruli Caretaker's cost)
    st.battlefield = [wall]
    assert ("Wall of Roots", None) in game.mana_ability_options(st), (
        "sanity check: the engine's own availability view must already say this is usable"
    )
    assert drl_env.legal_action_mask(st, spy_table)[tap_idx], (
        "a tapped-by-something-else Wall of Roots, not yet used this turn, must still be a legal mana action"
    )
    spy_table[tap_idx][2](st)
    assert st.mana_pool.get("G", 0) == 1, "activating it must actually float the G"


@pytest.mark.slow
def test_lotleth_giant_cast_survives_saruli_tapping_wall_of_roots():
    # END-TO-END REPRODUCTION of the reported bug: casting Lotleth Giant
    # ({6}{B}) in the actual Balustrade Spy deck, where Saruli Caretaker taps
    # Wall of Roots as its additional cost mid-payment. Wall of Roots' own
    # mana ability has no {T} in ITS OWN cost (mana_no_tap), so being tapped
    # by Saruli's unrelated cost must not disable it -- pre-fix, this exact
    # sequence left remaining={'generic': 1} owed with Wall of Roots as the
    # only source left on the board, and "Tap Wall of Roots" came back
    # illegal anyway: an all-False action mask, a hard crash, even though
    # plan_payment had already promised this cast was payable.
    spy_decklist = game.parse_decklist_file("../data/spy_combo.txt")
    spy_table = drl_env.build_action_table(spy_decklist, game.EFFECT_REGISTRY)

    def idx(name):
        return next(i for i, (n, _l, _e) in enumerate(spy_table) if n == name)

    st = GameState(on_the_play=True, players=[PlayerState(True), PlayerState(False)])
    st.phase = game.turn.Phase.MAIN1
    forests = [Permanent(game.CARD_DEFS["Forest"]) for _ in range(5)]
    swamp = Permanent(game.CARD_DEFS["Swamp"])
    wall = Permanent(game.CARD_DEFS["Wall of Roots"])
    saruli = Permanent(game.CARD_DEFS["Saruli Caretaker"])
    saruli.summoning_sick = False  # Saruli's own cost is "{T}, tap another creature" -- it needs this (302.6)
    for i, p in enumerate(forests + [swamp, wall, saruli]):
        p.slot = i
    st.battlefield = forests + [swamp, wall, saruli]
    st.hand = [game.CARD_DEFS["Lotleth Giant"]]

    cast_idx = idx("Cast Lotleth Giant")
    assert drl_env.legal_action_mask(st, spy_table)[cast_idx], "5 Forest + Swamp + Wall of Roots must already cover {6}{B}"
    spy_table[cast_idx][2](st)
    assert st.pending_resolution["kind"] == "pay_cost"
    assert st.pending_resolution["remaining"] == {"generic": 6, "B": 1}

    # Pay six of the seven pips off the lands, leaving exactly {generic: 1} --
    # tight enough that Wall of Roots' own ability is the only thing left.
    forest_idx, swamp_idx = idx("Tap Forest"), idx("Tap Swamp")
    for _ in range(5):
        spy_table[forest_idx][2](st)
    spy_table[swamp_idx][2](st)
    assert st.mana_pool == {"G": 5, "B": 1}
    while st.mana_pool:
        game.execute_pool_spend(st, game.pool_spend_options(st)[0])
    assert st.pending_resolution["remaining"].get("generic") == 1, "one generic pip must remain owed"

    # Saruli taps Wall of Roots (the only other untapped creature) as its
    # additional cost -- a normal, legal mid-payment choice.
    saruli_tap_idx = idx("Tap Saruli Caretaker")
    assert drl_env.legal_action_mask(st, spy_table)[saruli_tap_idx]
    spy_table[saruli_tap_idx][2](st)
    assert st.mana_subdecision["stage"] == "choose_target"
    game.execute_mana_subdecision_target(st, wall)  # same call the pointer dispatch makes
    assert st.mana_subdecision["stage"] == "choose_color"
    produce_idx = idx("Produce W")
    assert drl_env.legal_action_mask(st, spy_table)[produce_idx]
    spy_table[produce_idx][2](st)
    assert wall.tapped, "Saruli's cost must have tapped Wall of Roots"

    # The regression: Wall of Roots' own no-tap ability must STILL be a
    # legal action, even though Wall is now tapped for an unrelated reason,
    # and it's the only way left to pay the last owed pip.
    wall_tap_idx = idx("Tap Wall of Roots")
    assert drl_env.legal_action_mask(st, spy_table)[wall_tap_idx], (
        "Wall of Roots' own mana ability must stay legal after being tapped by Saruli's cost -- "
        "otherwise the payment strands with an all-False mask"
    )
    spy_table[wall_tap_idx][2](st)
    assert "G" in st.mana_pool

    while st.pending_resolution is not None:
        game.execute_pool_spend(st, game.pool_spend_options(st)[0])
    assert st.pending_resolution is None, "the cast must fully pay off, not strand"
    game.resolve_top_of_stack(st)
    assert any(p.card_def.name == "Lotleth Giant" for p in st.battlefield), \
        "Lotleth Giant must actually resolve onto the battlefield"


@pytest.mark.slow
def test_choose_cast_copy_own_graveyard_matching():
    # choose_cast_copy: which same-named graveyard copy is being cast (MTG
    # 601.2a). Pointer-only -- assert first that it adds zero fixed action
    # rows to any deck.
    for _deck_file in ("mono_red_madness.txt", "monster_tron.txt", "jund_wildfire.txt"):
        _dl = game.parse_decklist_file(f"../data/{_deck_file}")
        _tbl = build_fixed_action_table(_dl)
        assert not any("cast copy" in n.lower() or "cast_copy" in n.lower() for n, _l, _e in _tbl), (
            f"{_deck_file}: choose_cast_copy must add NO fixed action rows (it is pointer-only)"
        )

    # Unit: only same-named instances in the CASTER's OWN graveyard are legal,
    # matched by object identity -- an unrelated card and the opponent's
    # same-named copy are both excluded.
    _decklist, _token_defs, _fixed_table, vocab = _mono_red_fixture()
    cc_me = PlayerState(on_the_play=True)
    cc_opp = PlayerState(on_the_play=False)
    dart_def = game.CARD_DEFS["Lava Dart"]
    dart_a, dart_b = CardInstance(dart_def), CardInstance(dart_def)
    other = CardInstance(game.CARD_DEFS["Mountain"])
    cc_me.graveyard = [dart_a, other, dart_b]
    cc_opp.graveyard = [CardInstance(dart_def)]  # opponent's copy -- never castable by me
    cc_state = GameState(on_the_play=True, players=[cc_me, cc_opp])
    cc_picked = []
    game.begin_choose_cast_copy(cc_state, "Lava Dart", on_complete=lambda s, inst: cc_picked.append(inst))
    assert cc_state.pending_resolution["kind"] == "choose_cast_copy" and any_pointer_legal(cc_state)
    assert game.choose_cast_copy_options(cc_state) == [dart_a, dart_b]
    cc_ids = [ident for _i, _r, ident in build_token_set(cc_state, 0, vocab)]
    cc_legal = [r for r, ok in zip(cc_ids, pointer_legal_mask(cc_state, cc_ids)) if ok]
    assert cc_legal == [dart_a, dart_b] or cc_legal == [dart_b, dart_a], (
        f"exactly my own two Lava Dart instances must be legal, got {cc_legal}"
    )
    execute_pointer_choice(cc_state, dart_b)  # aim at the SECOND copy specifically
    assert cc_picked == [dart_b] and cc_state.pending_resolution is None, (
        "executing the pointer choice must deliver that exact instance and clear the pending"
    )


@pytest.mark.slow
def test_choose_cast_copy_percher_response():
    # The scenario this exists for: a Rooftop Percher trigger on the stack
    # has targeted one specific Lava Dart in my graveyard for exile. Lava
    # Dart is an instant, so its Flashback is castable in response -- casting
    # that copy removes it, making Percher's target illegal so the exile
    # fizzles (608.2b/c); casting the other copy leaves the targeted one to
    # be exiled. Both must be reachable, which is what picking the copy by
    # identity buys.
    _decklist, _token_defs, _fixed_table, vocab = _mono_red_fixture()
    dart_def = game.CARD_DEFS["Lava Dart"]
    for aim_at_targeted in (True, False):
        sc_me = PlayerState(on_the_play=True)
        sc_opp = PlayerState(on_the_play=False)
        t_a, t_b = CardInstance(dart_def), CardInstance(dart_def)
        sc_me.graveyard = [t_a, t_b]
        sc_me.battlefield = [Permanent(game.CARD_DEFS["Mountain"])]  # Lava Dart flashback: sacrifice a Mountain
        sc_state = GameState(on_the_play=True, players=[sc_me, sc_opp])
        # Percher's exile effect, already on the stack with t_a locked as its target.
        exiled = []

        def _percher_resolve(st, _cd, targets=(t_a,), sink=exiled):
            survivors = [i for i in targets if any(i in pl.graveyard for pl in st.players)]
            for i in survivors:
                next(pl for pl in st.players if i in pl.graveyard).graveyard.remove(i)
                sink.append(i)

        # Percher is the opponent's permanent, so its trigger's controller
        # must be seat 1 -- push_to_stack records state.active_idx as the
        # controller, deciding targeted_by_mine vs targeted_by_theirs from
        # seat 0's perspective below.
        sc_state.active_idx = 1
        game.push_to_stack(sc_state, game.CARD_DEFS["Rooftop Percher"], _percher_resolve,
                           reserves_hand_card=False, is_spell=False,
                           targets=(("graveyard_card", t_a),))
        sc_state.active_idx = 0  # priority back to me to respond with the flashback
        # The targeted copy shows targeted_by_theirs; the other copy shows
        # neither. Looked up by `is`, not a dict keyed by identity: the stack
        # also carries the Percher entry itself, whose identity is an
        # unhashable dict.
        sc_tokens = build_token_set(sc_state, 0, vocab)
        sc_row_a = next(row for _i, row, ident in sc_tokens if ident is t_a)
        sc_row_b = next(row for _i, row, ident in sc_tokens if ident is t_b)
        # Counted back from the row's end, past everything _token_row appends
        # after the targeting pair.
        tgt_theirs = -SLOTS_AFTER_TARGETING - 1
        assert sc_row_a[tgt_theirs] == 1.0 and sc_row_b[tgt_theirs] == 0.0, (
            "only the Percher-targeted graveyard copy may show targeted_by_theirs"
        )

        aimed = t_a if aim_at_targeted else t_b
        game.begin_choose_cast_copy(sc_state, "Lava Dart", on_complete=lambda s, inst: s.graveyard.remove(inst))
        execute_pointer_choice(sc_state, aimed)  # the flashback removes the chosen copy from the graveyard
        assert aimed not in sc_state.players[0].graveyard

        # Now let Percher resolve and see whether its target survived.
        game.resolve_top_of_stack(sc_state)
        if aim_at_targeted:
            assert exiled == [], "casting the targeted copy must make Percher's target illegal -> it exiles nothing"
            assert sc_state.players[0].graveyard == [t_b], "the untargeted copy is untouched"
        else:
            assert exiled == [t_a], "casting the OTHER copy leaves the targeted one to be exiled"
            assert sc_state.players[0].graveyard == [], "targeted copy exiled, other copy already cast"


@pytest.mark.slow
def test_no_mana_ability_during_a_casting_step_before_601_2f():
    # CR 601.2f: mana abilities are activated at one point in the casting
    # process -- after modes, X, delve and which-copy are settled and the
    # total cost is known, immediately before paying, not partway through
    # choosing them. No player receives priority during 601.2, so the
    # engine's otherwise correct "a mana ability is legal in any priority
    # window" (605.1a/605.3b) does not apply mid-cast. Refusing the whole
    # window generalizes better than guarding individual actions inside it
    # (the same hole would otherwise reappear at choose_delve_amount, where
    # every "Delve N" button re-checks affordability and a tap taken
    # mid-choice could leave them all illegal).
    bw_def = game.CARD_DEFS["Bramble Wurm"]
    bw_a, bw_b = CardInstance(bw_def), CardInstance(bw_def)
    bw_me = PlayerState(on_the_play=True)
    bw_me.graveyard = [bw_a, bw_b]
    bw_me.battlefield = [Permanent(game.CARD_DEFS["Barrels of Blasting Jelly"])]
    bw_state = GameState(on_the_play=True, players=[bw_me, PlayerState(on_the_play=False)])
    # Speculative floating is the active player's own main phase only; a
    # bare GameState has no phase at all, which would fail that gate before
    # reaching what this test is actually about.
    bw_state.phase = game.turn.Phase.MAIN1
    bw_state.mana_pool.update({"C": 2, "G": 1})  # 2 C pays the generic, G pays the G -- fully funded before the cast
    bw_decklist = game.parse_decklist_file("../data/monster_tron.txt")
    bw_table = drl_env.build_action_table(bw_decklist, game.EFFECT_REGISTRY)
    bw_filt_idx = next(i for i, (n, _l, _e) in enumerate(bw_table)
                        if n == "Filter Barrels of Blasting Jelly, paying G")
    # Not yet activated: the filter is a free, ordinary pool->pool conversion.
    assert drl_env.legal_action_mask(bw_state, bw_table)[bw_filt_idx], "filter must stay legal before any commitment"
    resolve_calls = []
    game.begin_choose_cast_copy(
        bw_state, "Bramble Wurm",
        on_complete=lambda s, inst: game.begin_pay_cost(
            s, bw_def.extra["gy_ability_cost"], on_complete=lambda s2: resolve_calls.append(inst)),
    )
    assert bw_state.pending_resolution["kind"] == "choose_cast_copy"
    # The cast is committed but 601.2f has not been reached: no mana ability
    # is legal. Checked for every mana row in the table, not just the filter
    # -- the rule is about the window, not one hazardous action inside it.
    bw_mask = drl_env.legal_action_mask(bw_state, bw_table)
    bw_mana_rows = [i for i, (n, _l, _e) in enumerate(bw_table)
                    if n.startswith("Tap ") or n.startswith("Filter ")]
    assert bw_mana_rows, "sanity: monster_tron must have mana rows for this to be testing anything"
    assert not any(bw_mask[i] for i in bw_mana_rows), (
        "no mana ability may be activated between announcing a cast and reaching 601.2f"
    )

    execute_pointer_choice(bw_state, bw_a)  # pick a copy -> begin_pay_cost opens, fully funded
    assert bw_state.pending_resolution["kind"] == "pay_cost"
    while bw_state.pending_resolution is not None:
        game.execute_pool_spend(bw_state, game.pool_spend_options(bw_state)[0])
    assert resolve_calls == [bw_a], "the ability must actually resolve -- the reserved mana was never stranded"


@pytest.mark.slow
def test_assign_combat_damage_pointer_mask_respects_lethal_cap():
    # rl.decision.action_bridge.pointer_legal_mask's "assign_combat_damage"
    # branch (the RL-layer twin of
    # game.resolution.handlers_combat.assign_combat_damage_options): a
    # blocker already at its own lethal_by_blocker cap must drop out of the
    # pointer mask entirely (no overkill), while an under-cap blocker stays
    # offered.
    _decklist, _token_defs, _fixed_table, vocab = _mono_red_fixture()
    ac_me = PlayerState(on_the_play=True)
    ac_opp = PlayerState(on_the_play=False)
    attacker = Permanent(game.CARD_DEFS["Reckless Lackey"])
    b1 = Permanent(game.CARD_DEFS["Slippery Bogle"])  # 1 toughness -- lethal cap 1
    b1.slot = 1
    b2 = Permanent(game.CARD_DEFS["Slippery Bogle"])  # a second, distinct instance -- cap 2
    b2.slot = 2
    ac_me.battlefield = [attacker]
    ac_opp.battlefield = [b1, b2]
    ac_state = GameState(on_the_play=True, players=[ac_me, ac_opp])
    ac_state.attackers = [attacker]
    ac_state.blocked_by = {attacker: [b1, b2]}
    game.begin_assign_combat_damage(
        ac_state, attacker, [b1, b2], 3, has_trample=False,
        lethal_by_blocker={b1: 1, b2: 2}, on_complete=lambda s: None,
    )
    assert ac_state.pending_resolution["kind"] == "assign_combat_damage"

    ac_ids = [ident for _i, _r, ident in build_token_set(ac_state, 0, vocab)]
    ac_legal = {r for r, ok in zip(ac_ids, pointer_legal_mask(ac_state, ac_ids)) if ok}
    assert ac_legal == {b1, b2}, f"both blockers start under their own cap, got {ac_legal}"

    execute_pointer_choice(ac_state, b1)  # 1 point onto b1 -- now AT its cap (1)
    ac_ids2 = [ident for _i, _r, ident in build_token_set(ac_state, 0, vocab)]
    ac_legal2 = {r for r, ok in zip(ac_ids2, pointer_legal_mask(ac_state, ac_ids2)) if ok}
    assert ac_legal2 == {b2}, f"b1 is at its own lethal cap (no overkill) -- only b2 should remain, got {ac_legal2}"


@pytest.mark.slow
def test_universal_cross_player_decision_rows():
    # Cross-player decision rows must exist in every deck's table. A card can
    # pose a question its opponent answers -- Spell Pierce/Ward (pay_unless:
    # the spell's controller pays), Deem Inferior (tuck_position: the tucked
    # permanent's owner chooses), Chain Lightning (may_copy), Cleansing
    # Wildfire (search_fetch: the destroyed land's controller searches),
    # Refurbished Familiar/Relic of Progenitus (discard/graveyard decline),
    # Undercity venture (initiative can be stolen, so a deck without any of
    # these cards can still be the one venturing). derive_pending_kinds reads
    # a deck's own cards, so gating these rows on it would leave the
    # answering seat with no row and an all-False mask -- they are
    # unconditional in every deck's table.
    #
    # Not here (deliberately): Transform/Cast(may)/Cast(madness)/Keep(select
    # to hand)/Decline(discard or sacrifice)/Decline(Ancient Stirrings)/
    # Decline(Malevolent Rumble)/Add mana -- each confirmed self-only (single
    # call site, always the deciding player's own card) -- see
    # test_self_only_decision_rows_gated_per_deck below.
    universal_decision_rows = (
        "Pay (unless)", "Don't pay (unless)",
        "Tuck: 2nd from top", "Tuck: bottom",
        "Copy spell", "Don't copy spell",
        "Keep (scry/surveil)", "Dispose (scry/surveil)",
        "Decline (search)", "Decline (graveyard)", "Decline (discard)",
        "Shuffle (Ponder)",
        "Target: yourself", "Target: opponent",
        "Target any: yourself", "Target any: opponent", "Choose no target",
    )
    with open("../data/league_decks.json") as _f:
        roster = json.load(_f)
    for deck_name, deck_file in roster.items():
        dl = game.parse_decklist_file(f"../data/{deck_file}")
        names = {n for n, _l, _e in build_fixed_action_table(dl)}
        missing = [r for r in universal_decision_rows if r not in names]
        assert not missing, (
            f"{deck_name}: missing universal decision row(s) {missing} -- an opponent's card can pose "
            "these questions, so every deck must be able to answer them"
        )


@pytest.mark.slow
def test_self_only_decision_rows_gated_per_deck():
    # The flip side of the test above: these decision kinds are audited and
    # confirmed self-only (each has exactly one call site in the whole
    # codebase, always the deciding player's own card). Unlike the genuinely
    # cross-player rows, making these universal would add real output rows
    # for cards a deck never runs, permanently masked illegal. Derived
    # generically from game.derive_pending_kinds rather than a hardcoded
    # per-deck list, so this stays correct if the roster ever changes.
    self_only_rows = {
        "may_transform": ("Transform", "Don't transform"),
        "may_cast": ("Cast (may)", "Decline (may)"),
        "madness_decision": ("Cast (madness)", "Decline (madness)"),
        "select_to_hand": ("Keep (select to hand)", "Bottom (select to hand)"),
        "discard_or_sacrifice": ("Decline (discard or sacrifice)",),
        "ancient_stirrings": ("Decline (Ancient Stirrings)",),
        "malevolent_rumble": ("Decline (Malevolent Rumble)",),
        "choose_mana_color": tuple(f"Add mana: {c}" for c in game.COLORS),
    }
    with open("../data/league_decks.json") as _f:
        roster = json.load(_f)
    for deck_name, deck_file in roster.items():
        dl = game.parse_decklist_file(f"../data/{deck_file}")
        own_kinds = game.derive_pending_kinds(dl)
        names = {n for n, _l, _e in build_fixed_action_table(dl)}
        for kind, rows in self_only_rows.items():
            present = [r for r in rows if r in names]
            if kind in own_kinds:
                assert present == list(rows), (
                    f"{deck_name}: owns a {kind!r} card but is missing row(s) "
                    f"{[r for r in rows if r not in names]}"
                )
            else:
                assert not present, (
                    f"{deck_name}: doesn't own a {kind!r} card but still carries {present} -- "
                    "should be gated on this deck's own derive_pending_kinds, not universal"
                )


@pytest.mark.slow
def test_choose_stack_target():
    # choose_stack_target: a spell to counter is very often the opponent's --
    # no per-deck "Choose: X" row could ever represent that. Pointer-
    # addressed, matching choose_graveyard_card's precedent.
    sct_me_dl = game.parse_decklist_file("../data/mono_blue_terror.txt")  # has Counterspell, no Cast Down
    sct_opp_dl = game.parse_decklist_file("../data/dmir_terror.txt")      # has Cast Down
    assert not any(n == "Cast Down" for n, *_ in sct_me_dl), "fixture: Cast Down must be the OPPONENT's card, not mine"
    sct_vocab = CardVocab([sct_me_dl, sct_opp_dl])
    sct_me = PlayerState(on_the_play=True)
    sct_opp = PlayerState(on_the_play=False)
    cast_down = game.CARD_DEFS["Cast Down"]
    # Two simultaneous copies of the opponent's spell -- exercises that
    # same-named duplicates stay independently addressable.
    entry_a = {"card_def": cast_down, "resolve": None, "controller": 1, "is_spell": True}
    entry_b = {"card_def": cast_down, "resolve": None, "controller": 1, "is_spell": True}
    sct_state = GameState(on_the_play=True, players=[sct_me, sct_opp])
    sct_state.stack = [entry_a, entry_b]
    sct_picked = []
    game.begin_choose_stack_target(sct_state, predicate=lambda e: True, on_complete=lambda s, e: sct_picked.append(e))
    assert sct_state.pending_resolution["kind"] == "choose_stack_target" and any_pointer_legal(sct_state)
    assert game.choose_stack_target_options(sct_state) == [entry_a, entry_b]
    sct_ids = [ident for _i, _r, ident in build_token_set(sct_state, 0, sct_vocab)]
    sct_legal = [r for r, ok in zip(sct_ids, pointer_legal_mask(sct_state, sct_ids)) if ok]
    assert sct_legal == [entry_a, entry_b] or sct_legal == [entry_b, entry_a], (
        f"both copies of the opponent's spell must be legal, independently addressable targets, got {sct_legal}"
    )
    execute_pointer_choice(sct_state, entry_b)  # aim at the SECOND copy specifically
    assert sct_picked == [entry_b] and sct_state.pending_resolution is None, (
        "executing the pointer choice must deliver that exact stack entry and clear the pending"
    )


@pytest.mark.slow
def test_order_triggers_unrepresentable_candidate_raises():
    # The durable runtime guard: a by-name kind whose candidates escape the
    # deciding player's own table must raise immediately, not silently
    # produce an all-False mask deep in some later game. Forge one into an
    # otherwise-ordinary pending kind to exercise it.
    guard_dl = game.parse_decklist_file("../data/mono_red_madness.txt")
    guard_table = build_fixed_action_table(guard_dl)
    guard_state = GameState(on_the_play=True, players=[PlayerState(True), PlayerState(False)])
    guard_state.pending_resolution = {"kind": "order_triggers", "remaining": [
        {"card_def": game.CARD_DEFS["Sagu Wildling"], "resolve": None},  # a real card, NOT in mono_red_madness
    ]}
    raised = False
    try:
        drl_env.legal_action_mask(guard_state, guard_table)
    except RuntimeError as e:
        raised = "Sagu Wildling" in str(e) and "order_triggers" in str(e)
    assert raised, "a by-name kind returning a candidate outside this deck's own table must raise immediately, not silently mask-fail"
