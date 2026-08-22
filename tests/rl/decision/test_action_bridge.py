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
    # mono_red_madness creates BOTH of these mid-game (Blood from madness
    # discards, Robot from Melded Moxite's own sacrifice ability) -- the deck
    # config's own "token_card_defs" only lists "Blood", missing "Robot", so
    # this fixture passes the complete, correct set explicitly rather than
    # relying on that config.
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
    # assign/unassign indefinitely (a real, observed pathology: tens of
    # thousands of iterations in one boggles_mirror evaluation, back when a
    # per-round action cap forced an end to it). That cap is gone now
    # (game/turn.py, 2026-08-19 -- no iteration cap exists anywhere in the
    # turn loop), so this no-undo rule is the ONLY thing ruling the
    # pathology out; nothing would stop it if reintroduced.
    # Hardcodes the literal substring independently of any prefix-filter
    # tuple, so a future reintroduction of this exact shape fails loudly here
    # rather than silently passing whatever filter happens to exist at the
    # time.
    assert not any("nassign" in name for name in fixed_names), (
        "no action may let the agent reconsider/reverse an earlier commitment -- see todo/no_undo_policy.md"
    )
    assert "Pass" in fixed_names
    assert any(name.startswith("Play land:") for name in fixed_names)

    # DFC back faces (Delver of Secrets -> Insectile Aberration) must get their
    # own "Choose: X" row, same as a token name -- a transformed permanent's
    # card_def swaps to the back face, so any BY-NAME resolution (order_
    # triggers, sacrifice, discard, search_fetch, ...) can need to reference it.
    # Omitting it from choosable_names produces an all-False action mask the
    # moment a by-name resolution needs to reference the back face
    # (drl_env._actions_table.build_action_table).
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
    # pointer_legal_mask / execute_pointer_choice against a REAL 2-PLAYER
    # game (build_token_set is inherently 2-player -- it always indexes an
    # opponent seat, so a 1-player game state doesn't fit its contract at
    # all). Mirror match, one shared choose_action dispatching on
    # state.active_idx. Always prefers a pointer action when one is legal
    # (Attack/Assign Blocker/Choose opponent's), else falls back to the fixed
    # table's own "prefer Pass, else first legal" rule -- exercises
    # attacking, blocking, AND the cross-player "Choose opponent's" consult
    # nested inside blocking, across real turns.
    decklist, _token_defs, fixed_table, vocab = _mono_red_fixture()
    pass_action = next(i for i, (name, _l, _e) in enumerate(fixed_table) if name == "Pass")
    saw_attack, saw_block, saw_choose_opponent = [False], [False], [False]

    def make_choose_action(rng):
        def choose_action(state):
            seat = state.active_idx
            pend = state.pending_resolution
            if pend is not None and pend["kind"] in ("mulligan_decision", "mulligan_bottom"):
                # Pregame decisions aren't in the fixed table -- owned by the
                # mulligan model in real play. Keep every hand here so this
                # test's random fixed/pointer play still reaches real turns.
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

            # Prefer any NON-Pass legal action (randomly) so the game
            # actually develops a board -- always preferring Pass would mean
            # lands/creatures never get played at all, and combat would never
            # happen for this test to exercise. Pass is the fallback of last
            # resort, not the default preference.
            #
            # Legality via ONE drl_env.legal_action_mask sweep, never by
            # calling each action's own legal_fn independently: drl_env's
            # sweep-scoped caches (e.g. _mana_ability_options_cache) are
            # explicitly documented as "valid only for the duration of one
            # legal_action_mask sweep," so calling legal_fn ad hoc, action by
            # action, outside that sweep would read stale cache state.
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
    # any ONE game (declare_blockers only yields a real decision if attacks
    # were declared AND the defender happens to have an eligible blocker
    # alive at that moment). Retrying across several independent games is
    # the robust way to cover every sub-case, not enlarging one game and
    # hoping.
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
    # NO-UNDO regression (todo/no_undo_policy.md): a COMMITTED blocker must
    # never be offered as a pointer target at all -- no way to reconsider it
    # once assigned, only "add more blockers" or "Done blocking" remain
    # legal. Hand-built (not random play) specifically to construct the
    # already-committed state and assert the negative. blocked_by/attackers
    # are keyed by the ATTACKING player's own permanents
    # (game.effects.combat.creature_block_eligible's own docstring), so the
    # "opponent" PlayerState here is the attacker's side even though the
    # deciding seat (0, active_idx's default) is the blocker's own.
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
    # Saruli Caretaker mana_subdecision -- POINTER mechanics (choose_target
    # stage): domain is my own battlefield, excludes the source by IDENTITY
    # (not name -- two same-named Sarulis are legitimately valid targets of
    # each other), excludes already-tapped creatures, applies the source's
    # own mana_extra_choose predicate. drl_env's own self-check exercises
    # the rest of the flow (the fixed-table "Tap <name>"/"Produce <color>"
    # rows) directly; this is specifically the pointer half.
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
    # THE SCENARIO THAT JUSTIFIES THIS WHOLE DESIGN: Quirion Ranger targets
    # an opponent's Ward creature -> pay_unless opens with the Saruli
    # player as payer -> they activate Saruli (BOTH mana_subdecision stages)
    # WHILE pay_unless is still open -> control returns to pay_unless,
    # completely untouched, once the mana_subdecision closes. Reachable in
    # the real 11-deck league (spy_combo's Quirion Ranger vs.
    # mono_blue_terror/dmir_terror's Ward creatures). Gating mana abilities
    # through the same building blocks pending_resolution itself uses
    # (begin_choose_permanent/begin_choose_mana_color) would wrongly mask
    # them mid-ANY-resolution, which is why mana_subdecision is its own
    # separate field instead.
    _decklist, _token_defs, _fixed_table, vocab = _mono_red_fixture()
    pu_me = PlayerState(on_the_play=True)
    pu_opp = PlayerState(on_the_play=False)
    pu_saruli = Permanent(game.CARD_DEFS["Saruli Caretaker"])
    pu_saruli.summoning_sick = False  # Saruli's own {T} ability needs this (302.6) -- see drl_env's own self-check for the same note
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
    # OPPONENT's graveyard -- the cross-deck case that needs per-object, not
    # per-name, addressing -- and confirm the pointer offers exactly that card
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
    # Faithful hand reveal (Mesmeric Fiend): the same primitive can pick from a
    # player's HAND, which the effect reveals. The whole revealed hand is
    # tokenized so the pointer can address it, but only predicate-legal (nonland)
    # cards are selectable -- no whole-league fixed "Choose: X" rows needed.
    _decklist, _token_defs, _fixed_table, vocab = _mono_red_fixture()
    hr_me = PlayerState(on_the_play=True)
    hr_opp = PlayerState(on_the_play=False)
    hr_opp.hand = [game.CARD_DEFS["Lightning Bolt"], game.CARD_DEFS["Mountain"]]
    hr_state = GameState(on_the_play=True, players=[hr_me, hr_opp])
    hr_picked = []
    game.begin_choose_graveyard_card(hr_state, predicate=lambda c: c.card_type != game.CardType.LAND,
                                     on_complete=lambda st, chosen: hr_picked.append(chosen), graveyard=hr_opp.hand)
    hr_refs = [ident for _idx, _row, ident in build_token_set(hr_state, 0, vocab)]
    # DEFERRED hand: identities are the CardDefs; this state has no other zones,
    # so every non-None token is one of the two revealed hand cards.
    assert {r.name for r in hr_refs if r is not None} == {"Lightning Bolt", "Mountain"}, \
        "the whole revealed hand must be tokenized"
    hr_legal = [ref for ref, ok in zip(hr_refs, pointer_legal_mask(hr_state, hr_refs)) if ok]
    assert len(hr_legal) == 1 and hr_legal[0] is hr_opp.hand[0], \
        "only the nonland revealed-hand card must be a legal pointer target"
    execute_pointer_choice(hr_state, hr_legal[0])
    assert len(hr_picked) == 1 and hr_picked[0] is hr_opp.hand[0], "executing the reveal-hand pointer choice must resolve to that exact card"


@pytest.mark.slow
def test_stranded_payment_filter_regression():
    # STRANDED-PAYMENT regression: a mana FILTER is a pool->pool conversion,
    # legal mid-payment, that could otherwise consume the very colored pip an
    # already-begun payment still owes -- leaving remaining={'G': 1} with no
    # green source and, since float-first mana has no "Abandon payment"
    # action, no legal action at all (all-False mask).
    # drl_env._actions_mana._filter_would_strand_payment forbids exactly that.
    tron_decklist = game.parse_decklist_file("../data/monster_tron.txt")
    tron_table = drl_env.build_action_table(tron_decklist, game.EFFECT_REGISTRY)
    # "Filter Barrels of Blasting Jelly, paying G" -- the exact action from the crash.
    filt_idx = next(i for i, (n, _l, _e) in enumerate(tron_table)
                    if n == "Filter Barrels of Blasting Jelly, paying G")
    st = GameState(on_the_play=True, players=[PlayerState(True), PlayerState(False)])
    # Speculative floating is the active player's own main phase only
    # (drl_env._actions_mana._mana_timing_legal); a bare GameState has no
    # phase at all, which would fail that gate before reaching what this
    # test is actually about.
    st.phase = game.turn.Phase.MAIN1
    st.battlefield = [Permanent(game.CARD_DEFS["Barrels of Blasting Jelly"])]
    st.mana_pool.update({"G": 1})
    # With no payment pending, converting the lone G is a perfectly legal choice.
    assert drl_env.legal_action_mask(st, tron_table)[filt_idx], "filter must stay legal with no payment pending"
    # Mid-payment owing exactly {G} with exactly one G floating. Barrels can
    # output ANY of five colors, and one of them is G -- so the conversion
    # itself is still perfectly safe (G in, G back out) and stays legal. What
    # must not be reachable is the specific COLOR choice that strands, and that
    # is where the guard now bites: one gate later, on exactly the wrong
    # branch, instead of blanket-refusing a whole action that has a safe line.
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
    # Non-empty by construction, which is the invariant that actually matters:
    # the agent is never left with a payment it owes and no legal way to make it.
    assert offered


@pytest.mark.slow
def test_choose_cast_copy_own_graveyard_matching():
    # choose_cast_copy: WHICH same-named graveyard copy is being cast (MTG
    # 601.2a). Pointer-only -- assert first that it added ZERO fixed action rows
    # to any deck, which is what kept every existing checkpoint valid.
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
    # THE SCENARIO this exists for: a Rooftop Percher trigger on the stack has
    # targeted ONE specific Lava Dart in my graveyard for exile. Lava Dart is an
    # INSTANT, so its Flashback is castable in response -- and casting THAT copy
    # removes it, making Percher's target illegal so the exile fizzles (608.2b/c).
    # Casting the OTHER copy instead leaves the targeted one to be exiled. Both
    # are legal Magic; the point is that both must be REACHABLE, which is exactly
    # what picking the copy by identity buys.
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

        # Percher is the OPPONENT's permanent, so its trigger's controller must be
        # seat 1 -- push_to_stack records state.active_idx as the controller, and
        # that is what decides targeted_by_MINE vs targeted_by_THEIRS from seat 0's
        # perspective below.
        sc_state.active_idx = 1
        game.push_to_stack(sc_state, game.CARD_DEFS["Rooftop Percher"], _percher_resolve,
                           reserves_hand_card=False, is_spell=False,
                           targets=(("graveyard_card", t_a),))
        sc_state.active_idx = 0  # priority back to me to respond with the flashback
        # The targeted copy shows targeted_by_theirs; the other copy shows neither
        # -- this is what makes "cast the one they're pointing at" learnable at all.
        # Looked up by `is`, not a dict keyed by identity: the stack now also
        # carries a token (the Percher entry itself) whose identity is an
        # unhashable dict, so `{ident: row for ...}` would crash outright.
        sc_tokens = build_token_set(sc_state, 0, vocab)
        sc_row_a = next(row for _i, row, ident in sc_tokens if ident is t_a)
        sc_row_b = next(row for _i, row, ident in sc_tokens if ident is t_b)
        # Counted back from the row's END, past everything _token_row appends
        # after the targeting pair -- features.py owns that count, so another
        # inserted block shifts this automatically.
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
    # CR 601.2f: mana abilities are activated at ONE point in the casting
    # process -- after modes, X, delve and which-copy are settled and the total
    # cost is known, immediately before paying. Not partway through choosing
    # them. No player receives priority during 601.2, so the engine's otherwise
    # correct "a mana ability is legal in ANY priority window" (605.1a/605.3b)
    # simply does not apply mid-cast.
    #
    # This REPLACES an older regression that asserted the opposite -- that a
    # filter stayed legal during choose_cast_copy and had to be individually
    # guarded against stranding the reserved cost (2 Bramble Wurms in the
    # graveyard, activate the {2}{G} graveyard ability, then filter the {G}
    # away while still picking a copy). Refusing the whole window is both more
    # faithful and strictly safer than guarding one action inside it, and it
    # generalises: the same hole reappeared at choose_delve_amount, where every
    # "Delve N" button re-checks affordability and a tap taken mid-choice could
    # leave them ALL illegal (found live -- Gurmag Angler, black source tapped
    # for blue after announcing).
    bw_def = game.CARD_DEFS["Bramble Wurm"]
    bw_a, bw_b = CardInstance(bw_def), CardInstance(bw_def)
    bw_me = PlayerState(on_the_play=True)
    bw_me.graveyard = [bw_a, bw_b]
    bw_me.battlefield = [Permanent(game.CARD_DEFS["Barrels of Blasting Jelly"])]
    bw_state = GameState(on_the_play=True, players=[bw_me, PlayerState(on_the_play=False)])
    # Speculative floating is the active player's own main phase only
    # (drl_env._actions_mana._mana_timing_legal); a bare GameState has no
    # phase at all, which would fail that gate before reaching what this
    # test is actually about.
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
    # The cast is committed but 601.2f has not been reached: NO mana ability is
    # legal, so there is nothing here that could consume the mana the payment is
    # about to need. Every mana row in the table, not just the filter -- the rule
    # is about the window, not about one hazardous action inside it.
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
    # rl.decision.action_bridge.pointer_legal_mask's own "assign_combat_damage" branch
    # (the RL-layer twin of game.resolution.handlers_combat.
    # assign_combat_damage_options, added alongside a6f4639's trample fix): a
    # blocker already at its own lethal_by_blocker cap must drop out of the
    # pointer mask entirely (no overkill), while an under-cap blocker stays
    # offered. Only the engine layer had coverage for this before (tests/
    # game/effects/test_combat_damage_detail.py) -- nothing exercised the RL
    # pointer-head side.
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
    # CROSS-PLAYER DECISION ROWS must exist in EVERY deck's table. A card can pose
    # a question its OPPONENT answers -- Spell Pierce/Ward (pay_unless: the
    # spell's controller pays), Deem Inferior (tuck_position: the tucked
    # permanent's owner chooses), Chain Lightning (may_copy / a new target for the
    # copy), Cleansing Wildfire (search_fetch: the destroyed land's controller
    # searches), Refurbished Familiar/Relic of Progenitus (discard/graveyard
    # decline -- both target the opponent), Undercity venture (choose_room/
    # choose_target_player/scry/search_fetch -- initiative can be STOLEN, so a
    # deck with none of these among its own cards can still be the one
    # venturing). derive_pending_kinds reads a deck's OWN cards, so gating these
    # rows on it would leave the ANSWERING seat with no row for the question
    # and an all-False mask. They are unconditional in every deck's table;
    # this asserts every deck has them.
    #
    # NOT here (deliberately): Transform/Cast(may)/Cast(madness)/Keep(select to
    # hand)/Decline(discard or sacrifice)/Decline(Ancient Stirrings)/Decline
    # (Malevolent Rumble)/Add mana -- each confirmed SELF-ONLY (single call
    # site, always the deciding player's own card: Delver's own upkeep,
    # Cascade's own may-cast, madness always triggers off your own discarded
    # card, Lead the Stampede/Highway Robbery/Ancient Stirrings/Malevolent
    # Rumble/Chromatic Star are each one specific card's own effect) -- see
    # test_self_only_decision_rows_gated_per_deck below, and drl_env._actions_table.
    # build_action_table's own "UNIVERSAL DECISION ROWS" header comment.
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
    # The flip side of the test above: these decision kinds were AUDITED and
    # confirmed self-only (each has exactly one call site in the whole
    # codebase, and it's always the deciding player's own card -- see
    # build_action_table's own header comment for the per-kind evidence).
    # Unlike the genuinely cross-player rows, making these universal would
    # just repeat impulse's own history: real output rows for a card the
    # deck never runs, permanently masked illegal. Derived generically from
    # game.derive_pending_kinds rather than a hardcoded per-deck list, so
    # this stays correct if the roster or its decklists ever change.
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
    # choose_stack_target: a spell to counter is very often the OPPONENT's --
    # no per-deck "Choose: X" row could ever represent that. Pointer-
    # addressed, matching choose_graveyard_card's own precedent. Real
    # cross-deck fixture, not a synthetic name.
    sct_me_dl = game.parse_decklist_file("../data/mono_blue_terror.txt")  # has Counterspell, no Cast Down
    sct_opp_dl = game.parse_decklist_file("../data/dmir_terror.txt")      # has Cast Down
    assert not any(n == "Cast Down" for n, *_ in sct_me_dl), "fixture: Cast Down must be the OPPONENT's card, not mine"
    sct_vocab = CardVocab([sct_me_dl, sct_opp_dl])
    sct_me = PlayerState(on_the_play=True)
    sct_opp = PlayerState(on_the_play=False)
    cast_down = game.CARD_DEFS["Cast Down"]
    # TWO simultaneous copies of the opponent's spell -- also exercises that
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
    # The durable runtime guard itself: a by-name kind whose candidates escape
    # the deciding player's own table must raise IMMEDIATELY, not silently
    # produce an all-False mask deep in some later game (the shape of bug
    # order_triggers/DFC and choose_stack_target/opponent can both take).
    # Forge one into an otherwise-ordinary pending kind to exercise it.
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
