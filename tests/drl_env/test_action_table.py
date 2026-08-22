"""Tests for drl_env's action table (drl_env._actions_table and its sibling
category modules): build_action_table, legal_action_mask, and the Plot/
on-cast-trigger/token/targeting/combat mechanics they wire together, end
to end through the real production functions."""

from pathlib import Path

import pytest

import game
from drl_env import _actions_mana, _actions_table
from drl_env._actions_common import *
from drl_env._actions_cast import *
from drl_env._actions_cast_altzone import *
from drl_env._actions_combat import *
from drl_env._actions_resolution import *
from drl_env._actions_mana import *
from drl_env._actions_table import *
from game.cards import CardDef, CardType, EffectId
from game.state import CardInstance, GameState, Permanent, PlayerState

# tests/drl_env/test_action_table.py -> tests/drl_env -> tests -> repo root,
# same depth data/ sits at.
DATA_DIR = Path(__file__).resolve().parents[2] / "data"


def _action_index(actions, action_name):
    return next(i for i, (nm, _l, _e) in enumerate(actions) if nm == action_name)


def _boggles_decklist():
    return game.parse_decklist_file(str(DATA_DIR / "boggles.txt"))


def test_plot_and_on_cast_trigger():
    # Exercises Plot and the on-cast trigger hook through the real _plot_legal/
    # _plot_execute/_cast_from_exile_legal/_cast_from_exile_execute functions.
    # No real Plot/Guttersnipe card exists yet, so this temporarily injects
    # into the global game.CARD_DEFS/game.EFFECT_REGISTRY, saving/restoring both.
    _card_defs_backup = dict(game.CARD_DEFS)
    _filler_backup = game.EFFECT_REGISTRY[EffectId.FILLER]
    _generous_ent_backup = game.EFFECT_REGISTRY[EffectId.GENEROUS_ENT]

    PLOT_COST = {"generic": 1, "B": 1}  # {B}, not {R}: EffectId.SWAMP is a real, correctly-wired mana source

    on_cast_calls = []
    plot_spell = CardDef("Fake Plot Spell", CardType.SORCERY, PLOT_COST, EffectId.FILLER)
    game.CARD_DEFS["Fake Plot Spell"] = plot_spell
    game.EFFECT_REGISTRY[EffectId.FILLER] = {
        "cast": {"resolve": lambda s, c: None},
        "plot": {"cost": PLOT_COST, "resolve": lambda s, c: (s.hand.remove(c), s.exile.append((c, s.turn_number)))},
    }
    # Guttersnipe stand-in: a permanent whose registry entry has an
    # "on_cast" trigger -- borrows EffectId.GENEROUS_ENT for the duration.
    game.EFFECT_REGISTRY[EffectId.GENEROUS_ENT] = {
        "on_cast": lambda s, permanent: on_cast_calls.append(permanent.card_def.name),
    }
    try:
        state = GameState(on_the_play=True)
        state.phase = game.turn.Phase.MAIN1  # Plot defaults to SORCERY speed, needs a sorcery-speed phase
        state.hand = [plot_spell]
        state.battlefield = [
            Permanent(CardDef("Swamp", CardType.LAND, None, EffectId.SWAMP)),
            Permanent(CardDef("Swamp", CardType.LAND, None, EffectId.SWAMP)),
            Permanent(CardDef("Guttersnipe-ish", CardType.CREATURE, None, EffectId.GENEROUS_ENT)),
        ]

        # Plot it: pay {1}{B}, exile with this turn's stamp. Float {B}{B} from
        # the two Swamps before plotting, then pay by spending from the pool.
        game.activate_mana_source(state, state.battlefield[0])
        game.activate_mana_source(state, state.battlefield[1])
        assert _plot_legal("Fake Plot Spell", PLOT_COST, game.turn.Speed.SORCERY)(state)
        _plot_execute("Fake Plot Spell", PLOT_COST, game.EFFECT_REGISTRY[EffectId.FILLER]["plot"]["resolve"])(state)
        assert state.pending_resolution["kind"] == "pay_cost"
        while state.pending_resolution is not None:
            game.execute_pool_spend(state, game.pool_spend_options(state)[0])
        assert state.pending_resolution is None
        assert state.hand == []
        assert [c.name for c, _stamp in state.exile] == ["Fake Plot Spell"]
        assert on_cast_calls == []  # plotting itself never fires on_cast -- it isn't casting the spell

        # Same turn: not castable yet ("on a later turn").
        assert not _cast_from_exile_legal("Fake Plot Spell", None, game.turn.Speed.SORCERY)(state)

        # A later turn: castable for free, queues on_cast_trigger (Guttersnipe).
        state.turn_number += 1
        assert _cast_from_exile_legal("Fake Plot Spell", None, game.turn.Speed.SORCERY)(state)
        _cast_from_exile_execute("Fake Plot Spell", game.EFFECT_REGISTRY[EffectId.FILLER]["cast"]["resolve"])(state)
        assert state.exile == []
        # on_cast queues the trigger; it fires only once game.turn's priority
        # round promotes it onto the stack and resolves, not inline at cast.
        assert on_cast_calls == []
        assert [e["type"] for e in state.trigger_queue] == ["cast_trigger"]
        game.promote_triggers_to_stack(state)
        game.resolve_top_of_stack(state)
        assert on_cast_calls == ["Guttersnipe-ish"]

        # extra_legal gate on the cast-from-exile path: Plot waives the mana
        # cost, not other costs a normal cast's extra_legal already checks.
        # Re-plot, then simulate an extra_legal that's never satisfiable.
        game.EFFECT_REGISTRY[EffectId.FILLER] = {
            "cast": {"resolve": lambda s, c: None, "extra_legal": lambda s: False},
            "plot": {"cost": PLOT_COST, "resolve": lambda s, c: (s.hand.remove(c), s.exile.append((c, s.turn_number)))},
        }
        state = GameState(on_the_play=True)
        state.phase = game.turn.Phase.MAIN1
        state.hand = [plot_spell]
        state.battlefield = [
            Permanent(CardDef("Swamp", CardType.LAND, None, EffectId.SWAMP)),
            Permanent(CardDef("Swamp", CardType.LAND, None, EffectId.SWAMP)),
        ]
        game.activate_mana_source(state, state.battlefield[0])
        game.activate_mana_source(state, state.battlefield[1])
        _plot_execute("Fake Plot Spell", PLOT_COST, game.EFFECT_REGISTRY[EffectId.FILLER]["plot"]["resolve"])(state)
        while state.pending_resolution is not None:
            game.execute_pool_spend(state, game.pool_spend_options(state)[0])
        state.turn_number += 1
        assert not _cast_from_exile_legal("Fake Plot Spell", game.EFFECT_REGISTRY[EffectId.FILLER]["cast"]["extra_legal"], game.turn.Speed.SORCERY)(state)
    finally:
        game.CARD_DEFS.clear()
        game.CARD_DEFS.update(_card_defs_backup)
        game.EFFECT_REGISTRY[EffectId.FILLER] = _filler_backup
        game.EFFECT_REGISTRY[EffectId.GENEROUS_ENT] = _generous_ent_backup


def test_on_cast_trigger_fires_only_after_cost_paid():
    # on_cast_trigger (Guttersnipe) must fire only once a spell's cost is
    # actually paid, queued in _cast_execute's _after_pay. Float {B}, announce
    # the cast, confirm the trigger has not fired while payment is pending,
    # then spend to complete it and confirm it fires only on resolve.
    _card_defs_backup = dict(game.CARD_DEFS)
    _filler_backup = game.EFFECT_REGISTRY[EffectId.FILLER]
    _generous_ent_backup = game.EFFECT_REGISTRY[EffectId.GENEROUS_ENT]

    on_cast_calls = []
    fake_bolt = CardDef("Fake Bolt", CardType.INSTANT, {"B": 1}, EffectId.FILLER)
    game.CARD_DEFS["Fake Bolt"] = fake_bolt
    game.EFFECT_REGISTRY[EffectId.FILLER] = {"cast": {"resolve": lambda s, c: None}}
    game.EFFECT_REGISTRY[EffectId.GENEROUS_ENT] = {
        "on_cast": lambda s, permanent: on_cast_calls.append(permanent.card_def.name),
    }
    try:
        state = GameState(on_the_play=True)
        state.phase = game.turn.Phase.MAIN1
        state.hand = [fake_bolt]
        swamp = Permanent(CardDef("Swamp", CardType.LAND, None, EffectId.SWAMP))
        state.battlefield = [
            swamp,
            Permanent(CardDef("Guttersnipe-ish", CardType.CREATURE, None, EffectId.GENEROUS_ENT)),
        ]
        cast_legal = _cast_legal("Fake Bolt", None, game.turn.Speed.INSTANT)
        cast_execute = _cast_execute("Fake Bolt", game.EFFECT_REGISTRY[EffectId.FILLER]["cast"]["resolve"])

        game.activate_mana_source(state, swamp)  # float {B} before casting
        assert cast_legal(state)
        cast_execute(state)
        assert state.pending_resolution["kind"] == "pay_cost"
        assert on_cast_calls == []  # not fired yet -- the cost isn't paid
        game.execute_pool_spend(state, "B")  # pay {B} from the pool -> cost complete
        # Paid: the spell is on the stack and the on_cast trigger is queued
        # (not fired inline). Promote + resolve to fire it, above the spell.
        assert on_cast_calls == []
        assert [e["type"] for e in state.trigger_queue] == ["cast_trigger"]
        assert len(state.stack) == 1  # the spell itself, paid and on the stack
        game.promote_triggers_to_stack(state)
        game.resolve_top_of_stack(state)
        assert on_cast_calls == ["Guttersnipe-ish"]
        assert len(state.stack) == 1  # the spell still waiting below the resolved trigger
    finally:
        game.CARD_DEFS.clear()
        game.CARD_DEFS.update(_card_defs_backup)
        game.EFFECT_REGISTRY[EffectId.FILLER] = _filler_backup
        game.EFFECT_REGISTRY[EffectId.GENEROUS_ENT] = _generous_ent_backup


def test_faithless_looting_flashback_requires_mana():
    """Flashback cost is {2}{R} and must actually be checked by
    _flashback_legal/plan_payment, not skipped."""
    decklist = [("Faithless Looting", 1), ("Mountain", 3)]
    actions = build_action_table(decklist, game.EFFECT_REGISTRY)
    _fb_name, fb_legal, fb_execute = next(
        (nm, lg, ex) for nm, lg, ex in actions if nm == "Flashback Faithless Looting"
    )

    state = GameState(on_the_play=True)
    state.phase = game.turn.Phase.MAIN1
    fl = CardInstance(game.CARD_DEFS["Faithless Looting"])
    state.graveyard = [fl]
    state.library = [CardDef(f"F{i}", CardType.LAND, None, EffectId.MOUNTAIN) for i in range(4)]
    assert not fb_legal(state)  # 0 floating mana -- must NOT be legal

    mountains = [Permanent(game.CARD_DEFS["Mountain"]) for _ in range(3)]
    state.battlefield = mountains
    for m in mountains:
        game.activate_mana_source(state, m)  # float 3 R before flashing back
    assert fb_legal(state)
    fb_execute(state)
    assert state.pending_resolution["kind"] == "pay_cost"
    while state.pending_resolution is not None:
        game.execute_pool_spend(state, game.pool_spend_options(state)[0])
    assert state.mana_pool == {}  # the full {2}{R} = 3 pips actually spent, not skipped
    assert fl not in state.graveyard and len(state.stack) == 1


def test_tokens_blood_sacrifice():
    # build_action_table's token_card_defs param is what makes "Activate
    # Blood (sac)" exist as an action -- "Blood" is never a decklist name.
    empty_decklist = []
    no_token_actions = build_action_table(empty_decklist, game.EFFECT_REGISTRY)
    assert not any("Blood" in nm for nm, _l, _e in no_token_actions)  # opt-in: omitted => absent

    token_actions = build_action_table(empty_decklist, game.EFFECT_REGISTRY, token_card_defs=(game.BLOOD_TOKEN_CARD_DEF,))
    activate_name, activate_legal, activate_execute = next(
        (nm, lg, ex) for nm, lg, ex in token_actions if nm == "Activate Blood (sac)"
    )

    state = GameState(on_the_play=True)
    game.create_token(state, game.BLOOD_TOKEN_CARD_DEF)
    state.battlefield.append(Permanent(CardDef("Swamp", CardType.LAND, None, EffectId.SWAMP)))
    state.hand = [CardDef("Card To Discard", CardType.SORCERY, {}, None)]
    state.library = [CardDef("Library Card", CardType.SORCERY, {}, None)]

    swamp = next(p for p in state.battlefield if p.card_def.name == "Swamp")
    game.activate_mana_source(state, swamp)  # float {B} before activating (Blood's {1})
    assert activate_legal(state)
    activate_execute(state)  # pays {1} via the real begin_pay_cost path
    assert state.pending_resolution["kind"] == "pay_cost"
    # This engine never auto-spends floated mana toward generic; spend explicitly.
    game.execute_pool_spend(state, game.pool_spend_options(state)[0])

    assert state.pending_resolution["kind"] == "discard"  # Blood's discard -- a cost, paid before the effect
    game.execute_discard_option(state, "Card To Discard")
    assert state.pending_resolution is None
    assert [p.card_def.name for p in state.battlefield] == ["Swamp"]  # Blood is gone, never added to any zone
    # The draw is Blood's effect -- on the stack now, not fired inline. Resolve it.
    assert len(state.stack) == 1 and state.hand == []
    game.resolve_top_of_stack(state)
    assert [c.name for c in state.hand] == ["Library Card"]  # discarded one, drew one


def test_cross_player_targeting():
    # build_action_table's opponent_decklist/opponent_token_card_defs params
    # register "Choose opponent's: X (slot k)" actions from the other side's
    # own card pool. Boggles on both sides since a real decklist with real
    # creature quantities on both sides is all this needs.
    boggles_decklist = _boggles_decklist()

    no_opponent_actions = build_action_table(boggles_decklist, game.EFFECT_REGISTRY)
    assert not any(nm.startswith("Choose opponent's:") for nm, _l, _e in no_opponent_actions)  # 1p mode: never registered

    my_actions = build_action_table(boggles_decklist, game.EFFECT_REGISTRY, opponent_decklist=boggles_decklist)
    bogle_slot_actions = [nm for nm, _l, _e in my_actions if nm.startswith("Choose opponent's: Slippery Bogle")]
    assert bogle_slot_actions == [f"Choose opponent's: Slippery Bogle (slot {k})" for k in range(1, 5)]  # boggles.txt: 4 copies
    # Registered for every opponent choosable name, not just creatures (e.g.
    # Masked Vandal's ETB targets an opponent artifact/enchantment).
    forest_slot_actions = [nm for nm, _l, _e in my_actions if nm.startswith("Choose opponent's: Forest")]
    assert forest_slot_actions == [f"Choose opponent's: Forest (slot {k})" for k in range(1, 13)]  # boggles.txt: 12 Forests

    target_slot_2 = _action_index(my_actions, "Choose opponent's: Slippery Bogle (slot 2)")
    target_slot_1 = _action_index(my_actions, "Choose opponent's: Slippery Bogle (slot 1)")

    attacker_bogle_1 = Permanent(game.CARD_DEFS["Slippery Bogle"])
    attacker_bogle_2 = Permanent(game.CARD_DEFS["Slippery Bogle"])
    attacker_bogle_2.slot = 2
    state = GameState(on_the_play=True, players=[PlayerState(True), PlayerState(False)])
    state.players[0].battlefield = [attacker_bogle_1, attacker_bogle_2]
    state.active_idx = 1  # simulating the defender's own already-flipped perspective

    _, legal_slot_2, execute_slot_2 = my_actions[target_slot_2]
    _, legal_slot_1, _ = my_actions[target_slot_1]
    assert not legal_slot_2(state) and not legal_slot_1(state)  # nothing pending yet

    completed = []
    game.begin_choose_opponent_permanent(
        state, lambda p: p.card_def.card_type == game.CardType.CREATURE, lambda s, choice: completed.append(choice),
    )
    assert legal_slot_1(state) and legal_slot_2(state)
    execute_slot_2(state)
    assert completed == [("Slippery Bogle", 2)]  # the specific slot targeted, not an arbitrary same-named match
    assert not legal_slot_2(state)  # resolution is complete, nothing pending anymore


def test_turn_owner_priority_holder_split():
    # _land_drop_legal (via speed_legal) and _attack_legal must both refuse
    # the non-turn player even when state.phase/lands_played_this_turn/their
    # own eligible creature would otherwise look legal -- simulates a
    # priority consult (active_idx flipped away from turn_player_idx).
    boggles_decklist = _boggles_decklist()
    my_actions = build_action_table(boggles_decklist, game.EFFECT_REGISTRY, opponent_decklist=boggles_decklist)

    play_forest_idx = _action_index(my_actions, "Play land: Forest")
    attack_bogle_idx = _action_index(my_actions, "Attack: Slippery Bogle (slot 1)")
    _, play_forest_legal, _ = my_actions[play_forest_idx]
    _, attack_bogle_legal, _ = my_actions[attack_bogle_idx]

    turn_owner_state = GameState(on_the_play=True, players=[PlayerState(True), PlayerState(False)])
    turn_owner_state.turn_player_idx = 0
    turn_owner_state.active_idx = 0
    turn_owner_state.phase = game.turn.Phase.DECLARE_ATTACKERS
    attacking_bogle = Permanent(game.CARD_DEFS["Slippery Bogle"])
    attacking_bogle.summoning_sick = False
    turn_owner_state.players[0].hand = [game.CARD_DEFS["Forest"]]
    turn_owner_state.players[0].battlefield = [attacking_bogle]
    assert attack_bogle_legal(turn_owner_state)  # the turn player's own creature, their own DECLARE_ATTACKERS -- legal

    turn_owner_state.phase = game.turn.Phase.MAIN1
    assert play_forest_legal(turn_owner_state)  # the turn player's own MAIN1, land in hand, none played yet -- legal

    turn_owner_state.active_idx = 1  # simulating a priority consult of the OTHER player
    turn_owner_state.players[1].hand = [game.CARD_DEFS["Forest"]]  # even with their OWN land available
    assert not play_forest_legal(turn_owner_state)  # refused -- not their turn, regardless of their own hand/lands_played_this_turn

    turn_owner_state.phase = game.turn.Phase.DECLARE_ATTACKERS
    non_turn_bogle = Permanent(game.CARD_DEFS["Slippery Bogle"])
    non_turn_bogle.summoning_sick = False
    turn_owner_state.players[1].battlefield = [non_turn_bogle]  # even with their OWN eligible creature at the same (name, slot)
    assert not attack_bogle_legal(turn_owner_state)  # refused -- declaring attackers is the turn player's own special action


def test_blocking():
    # build_action_table's "Assign Blocker: <name> (slot j)" / "Done blocking"
    # entries, end to end through the real production functions. Two
    # attacking Slippery Bogles, one defending Slippery Bogle blocks only one.
    boggles_decklist = _boggles_decklist()
    my_actions = build_action_table(boggles_decklist, game.EFFECT_REGISTRY, opponent_decklist=boggles_decklist)
    boggles_pending_kinds = game.derive_pending_kinds(boggles_decklist)  # exercised for its own sake

    assign_slot_1 = _action_index(my_actions, "Assign Blocker: Slippery Bogle (slot 1)")
    done_blocking_idx = _action_index(my_actions, "Done blocking")
    _, assign_legal, assign_execute = my_actions[assign_slot_1]
    _, done_legal, done_execute = my_actions[done_blocking_idx]

    atk_bogle_1 = Permanent(game.CARD_DEFS["Slippery Bogle"])
    atk_bogle_2 = Permanent(game.CARD_DEFS["Slippery Bogle"])
    atk_bogle_2.slot = 2
    defender_bogle = Permanent(game.CARD_DEFS["Slippery Bogle"])  # slot 1 by default
    state = GameState(on_the_play=True, players=[PlayerState(True), PlayerState(False)])
    state.players[0].battlefield = [atk_bogle_1, atk_bogle_2]
    state.players[0].attackers = [atk_bogle_1, atk_bogle_2]
    atk_bogle_1.tapped = True
    atk_bogle_2.tapped = True  # declare_attacker's effect, simulated directly here
    state.players[1].battlefield = [defender_bogle]
    state.active_idx = 1  # simulating _declare_blockers_gen's own flip to the defender

    assert not assign_legal(state) and not done_legal(state)  # nothing pending yet

    completed = []
    game.begin_declare_blockers(state, on_complete=lambda s: completed.append(True))
    assert completed == []
    assert assign_legal(state) and done_legal(state)

    assign_execute(state)  # "Assign Blocker: Slippery Bogle (slot 1)" -- parks defender_bogle as a blocker
    assert state.pending_resolution["kind"] == "choose_opponent_permanent"
    target_slot_1 = _action_index(my_actions, "Choose opponent's: Slippery Bogle (slot 1)")
    _, target_legal, target_execute = my_actions[target_slot_1]
    assert target_legal(state)
    target_execute(state)  # assigns it to block atk_bogle_1 specifically, not atk_bogle_2

    # Re-opened: defender_bogle is now spoken for, so it's no longer offered
    # again -- confirms creature_block_eligible gates the same action twice.
    assert state.pending_resolution["kind"] == "declare_blockers"
    assert not assign_legal(state)
    assert done_legal(state)
    done_execute(state)  # "Done blocking" -- atk_bogle_2 goes unblocked
    assert completed == [True]
    assert state.pending_resolution is None
    assert state.players[0].blocked_by == {atk_bogle_1: [defender_bogle]}


def test_flying_blocker_restriction():
    # _assign_blocker_execute's extra_predicate (game.has_keyword), end to
    # end: Silhana Ledgewalker ("can't be blocked except by flying") can
    # only be blocked by a flier (Kitchen Imp) -- a plain Slippery Bogle is
    # otherwise a legal untapped blocker but can never be assigned here.
    flying_decklist = [("Silhana Ledgewalker", 2), ("Slippery Bogle", 2), ("Kitchen Imp", 2)]
    flying_actions = build_action_table(flying_decklist, game.EFFECT_REGISTRY, opponent_decklist=flying_decklist)

    _, bogle_legal, bogle_execute = flying_actions[_action_index(flying_actions, "Assign Blocker: Slippery Bogle (slot 1)")]
    _, imp_legal, imp_execute = flying_actions[_action_index(flying_actions, "Assign Blocker: Kitchen Imp (slot 1)")]

    attacking_ledgewalker = Permanent(game.CARD_DEFS["Silhana Ledgewalker"])
    attacking_ledgewalker.tapped = True  # already attacked
    defending_bogle = Permanent(game.CARD_DEFS["Slippery Bogle"])
    defending_imp = Permanent(game.CARD_DEFS["Kitchen Imp"])
    state = GameState(on_the_play=True, players=[PlayerState(True), PlayerState(False)])
    state.players[0].battlefield = [attacking_ledgewalker]
    state.players[0].attackers = [attacking_ledgewalker]
    state.players[1].battlefield = [defending_bogle, defending_imp]
    state.active_idx = 1  # simulating _declare_blockers_gen's own flip to the defender

    game.begin_declare_blockers(state, on_complete=lambda s: None)
    # Slippery Bogle (no flying) is not even offered: the only attacker can
    # only be blocked by a flier, so Bogle has no legal target.
    assert not bogle_legal(state)
    assert imp_legal(state)

    imp_execute(state)  # Kitchen Imp HAS flying -- opens a real nested choice
    assert state.pending_resolution["kind"] == "choose_opponent_permanent"
    assert game.choose_opponent_permanent_options(state) == [("Silhana Ledgewalker", 1)]
    game.execute_choose_opponent_permanent_option(state, "Silhana Ledgewalker", 1)
    assert state.players[0].blocked_by == {attacking_ledgewalker: [defending_imp]}


def test_aura_targeting_exact_slot_addressing():
    # A target is chosen once, at cast time, exact (name, slot) addressed,
    # and re-validated by identity only once the spell resolves off the
    # stack. "Cast Rancor" pays its {G} cost, then (precast_choice) opens
    # choose_permanent with both Slippery Bogles offered by distinct slot;
    # picking slot 2 pushes to the stack; resolving attaches it to exactly
    # the one chosen, not an arbitrary same-named match.
    targeting_decklist = [("Slippery Bogle", 2), ("Rancor", 2), ("Forest", 10)]
    targeting_actions = build_action_table(targeting_decklist, game.EFFECT_REGISTRY)

    bogle_1 = Permanent(game.CARD_DEFS["Slippery Bogle"])
    bogle_2 = Permanent(game.CARD_DEFS["Slippery Bogle"])
    bogle_2.slot = 2
    forest = Permanent(game.CARD_DEFS["Forest"])
    rancor_card = game.CARD_DEFS["Rancor"]
    state = GameState(on_the_play=True)
    state.phase = game.turn.Phase.MAIN1  # sorcery-speed cast requires this -- GameState defaults phase=None
    state.battlefield = [bogle_1, bogle_2, forest]
    state.hand = [rancor_card]

    _, cast_rancor_legal, cast_rancor_execute = targeting_actions[_action_index(targeting_actions, "Cast Rancor")]
    targeting_actions[_action_index(targeting_actions, "Tap Forest")][2](state)  # float {G} before casting
    assert cast_rancor_legal(state)
    cast_rancor_execute(state)
    assert state.pending_resolution["kind"] == "pay_cost"
    targeting_actions[_action_index(targeting_actions, "Spend G from pool")][2](state)  # pays Rancor's {G} from the pool

    # Cost fully paid -- precast_choice means cast_aura runs its target choice
    # immediately here, not deferred to when this eventually pops off the
    # stack. The creature half rides the identity pointer scheme
    # (rl.decision.action_bridge), so this drives it via the same execute the
    # pointer path calls, not a "Choose target:" fixed action.
    assert state.pending_resolution["kind"] == "choose_any_target"
    assert set(game.choose_any_target_creature_options(state)) == {(0, "Slippery Bogle", 1), (0, "Slippery Bogle", 2)}

    game.execute_choose_any_target_creature(state, 0, "Slippery Bogle", 2)  # the specific slot-2 bogle
    # Target chosen -- pushed to the stack, not yet attached. The card left
    # hand at cast; resolve_top_of_stack restores it transiently for the
    # attach resolve.
    assert state.pending_resolution is None
    assert state.hand == [] and len(state.stack) == 1

    game.resolve_top_of_stack(state)
    assert state.hand == []
    rancor_permanent = next(p for p in state.battlefield if p.card_def.name == "Rancor")
    assert rancor_permanent.flags["enchanting"] is bogle_2  # the specific one chosen, not bogle_1


def test_aura_target_fizzle():
    # Same path as test_aura_targeting_exact_slot_addressing, but the exact
    # chosen permanent (bogle_1) is gone by the time the cast resolves -- the
    # whole spell fails, no effect, straight to the graveyard, never attaches.
    targeting_decklist = [("Slippery Bogle", 2), ("Rancor", 2), ("Forest", 10)]
    targeting_actions = build_action_table(targeting_decklist, game.EFFECT_REGISTRY)
    rancor_card = game.CARD_DEFS["Rancor"]

    state = GameState(on_the_play=True)
    state.phase = game.turn.Phase.MAIN1
    bogle_1 = Permanent(game.CARD_DEFS["Slippery Bogle"])
    forest = Permanent(game.CARD_DEFS["Forest"])
    state.battlefield = [bogle_1, forest]
    state.hand = [rancor_card]

    targeting_actions[_action_index(targeting_actions, "Tap Forest")][2](state)  # float {G} first
    targeting_actions[_action_index(targeting_actions, "Cast Rancor")][2](state)
    targeting_actions[_action_index(targeting_actions, "Spend G from pool")][2](state)
    game.execute_choose_any_target_creature(state, 0, "Slippery Bogle", 1)  # Aura targets via the any-target pointer path
    assert len(state.stack) == 1
    state.battlefield.remove(bogle_1)  # dies before the cast resolves

    game.resolve_top_of_stack(state)
    assert state.hand == []
    assert any(c.card_def is rancor_card for c in state.graveyard)  # fizzled to graveyard as a fresh instance (400.7)
    assert not any(p.card_def.name == "Rancor" for p in state.battlefield)


def test_mana_ability_options_cache_no_stale_leak():
    # _mana_ability_options_cache memoization must never return a stale
    # answer: sweep the mask with 1 untapped Mountain ("Tap Mountain" legal),
    # activate it (0 untapped sources left), then sweep again -- it must see
    # the mutation, not the first sweep's cached answer.
    perf_decklist = [("Mountain", 10), ("Lightning Bolt", 5)]
    perf_actions = build_action_table(perf_decklist, game.EFFECT_REGISTRY)
    perf_tap_mountain = _action_index(perf_actions, "Tap Mountain")
    perf_state = GameState(on_the_play=True, players=[PlayerState(True)])
    perf_mtn = Permanent(game.CARD_DEFS["Mountain"])
    perf_state.battlefield = [perf_mtn]
    # Speculative floating is main-phase-only; the default GameState carries
    # no phase, which would fail that gate before this test's own cache check.
    perf_state.phase = game.turn.Phase.MAIN1
    assert legal_action_mask(perf_state, perf_actions)[perf_tap_mountain]
    assert _actions_mana._mana_ability_options_cache is None  # cleared once the sweep returns
    game.activate_mana_source(perf_state, perf_mtn)  # taps the only Mountain
    assert ("Mountain", None) not in game.mana_ability_options(perf_state)  # ground truth: nothing left to tap
    assert not legal_action_mask(perf_state, perf_actions)[perf_tap_mountain]  # would be wrongly True if stale


def test_saruli_caretaker_two_stage_mana_subdecision():
    # Saruli Caretaker's extra cost -- tap another untapped creature -- is a
    # cost choice (602.5g), decided first ("Tap Saruli Caretaker", gate-free,
    # opens the mana_subdecision), then the tap-target (driven directly via
    # game.execute_mana_subdecision_target, same call the pointer dispatch
    # makes), then the color (shared "Produce <color>" buttons, gated via
    # legal_action_mask's _mana_subdecision_gate dispatch) -- matching real
    # sequencing, never the stack (605.1a).
    saruli_decklist = [("Saruli Caretaker", 2), ("Slippery Bogle", 2)]
    saruli_actions = build_action_table(saruli_decklist, game.EFFECT_REGISTRY)

    saruli = Permanent(game.CARD_DEFS["Saruli Caretaker"])
    saruli.summoning_sick = False  # Saruli's own {T} ability needs this (302.6)
    bogle_1 = Permanent(game.CARD_DEFS["Slippery Bogle"])
    bogle_2 = Permanent(game.CARD_DEFS["Slippery Bogle"])
    bogle_2.slot = 2
    bogle_2.tapped = True  # already tapped, not relevant to this row's aggregate legality
    state = GameState(on_the_play=True, event_log=[])
    # Speculative floating is the active player's own main phase only; a bare
    # GameState has no phase, which would fail that gate first.
    state.phase = game.turn.Phase.MAIN1
    state.battlefield = [saruli, bogle_1, bogle_2]

    tap_idx = _action_index(saruli_actions, "Tap Saruli Caretaker")
    _, tap_legal, tap_execute = saruli_actions[tap_idx]
    assert tap_legal(state)  # at least one other untapped creature exists (bogle_1)
    tap_execute(state)
    sub = state.mana_subdecision
    assert sub is not None and sub["stage"] == "choose_target" and sub["source"] is saruli and sub["target"] is None, (
        "must open a mana_subdecision at the choose_target stage, source resolved to this Saruli"
    )

    # No longer reachable while the sub-decision is open (legal_action_mask's
    # exclusive-priority dispatch; tap_legal itself doesn't check this).
    assert not legal_action_mask(state, saruli_actions)[tap_idx]

    game.execute_mana_subdecision_target(state, bogle_1)  # same call the pointer dispatch makes
    assert state.mana_subdecision["stage"] == "choose_color" and state.mana_subdecision["target"] is bogle_1
    assert not bogle_1.tapped, "the target isn't tapped until the color is actually chosen"

    color_idx = _action_index(saruli_actions, "Produce G")
    mask = legal_action_mask(state, saruli_actions)
    assert mask[color_idx], "Produce G must be legal mid choose_color stage (Saruli's own flexible 5-color spec)"
    assert not mask[tap_idx], "Tap Saruli Caretaker must stay illegal -- mana_subdecision still open"
    _, _color_legal, color_execute = saruli_actions[color_idx]
    color_execute(state)

    assert state.mana_subdecision is None
    assert saruli.tapped and bogle_1.tapped and bogle_2.tapped  # bogle_2 was already tapped, untouched otherwise
    assert state.mana_pool == {"G": 1}
    assert not tap_legal(state)  # Saruli itself now tapped -- no untapped source left
    # bogle_1 is the mana-subdecision target, not the activator's own Saruli.
    # Only one tap_or_untap event: bogle_1's tap (Saruli's tap is logged
    # separately, as a "mana_tap" event, by activate_mana_source).
    tap_events = [e for e in state.event_log if e["kind"] == "tap_or_untap"]
    assert len(tap_events) == 1
    assert tap_events[0]["permanent"] == ["Slippery Bogle", 1]
    assert tap_events[0]["now_tapped"] is True
    assert tap_events[0]["owner_idx"] == state.active_idx


def test_mana_filter_two_step_color_choice():
    # Mana filter (Barrels of Blasting Jelly), two-step design reusing
    # Saruli Caretaker's choose_color mana_subdecision machinery: step 1, a
    # flat fixed-table row "Filter X, paying <input>", pays the cost
    # immediately and opens the shared choose_color stage; step 2, a
    # "Produce <color>" button, produces the chosen output color into the
    # pool. No nested pay_cost -- state.mana_subdecision stays separate from
    # state.pending_resolution throughout.
    filter_decklist = [("Barrels of Blasting Jelly", 2)]
    filter_actions = build_action_table(filter_decklist, game.EFFECT_REGISTRY)

    barrels = Permanent(game.CARD_DEFS["Barrels of Blasting Jelly"])
    state = GameState(on_the_play=True)
    # Speculative floating is the active player's own main phase only; a
    # bare GameState has no phase, which would fail that gate first.
    state.phase = game.turn.Phase.MAIN1
    state.battlefield = [barrels]
    state.mana_pool = {"U": 1}

    pay_u_idx = _action_index(filter_actions, "Filter Barrels of Blasting Jelly, paying U")
    pay_w_idx = _action_index(filter_actions, "Filter Barrels of Blasting Jelly, paying W")
    _, legal_pay_u, execute_pay_u = filter_actions[pay_u_idx]
    _, legal_pay_w, _execute_pay_w = filter_actions[pay_w_idx]

    # Exactly one row per POOL_COLOR (no output_color cross product): 6 rows
    # for this one source, not 30.
    filter_rows = [n for n, _l, _e in filter_actions if n.startswith("Filter Barrels of Blasting Jelly")]
    assert len(filter_rows) == len(game.POOL_COLORS) == 6, filter_rows

    # A deck with a filter card but no Saruli-shaped mana_extra_choose card
    # must still get the shared "Produce <color>" rows.
    for color in game.COLORS:
        assert f"Produce {color}" in [n for n, _l, _e in filter_actions]

    assert legal_pay_u(state)
    assert not legal_pay_w(state)  # no floating W to spend

    execute_pay_u(state)
    assert state.mana_pool == {}, "the U pip is spent immediately as the cost -- no output produced yet"
    assert barrels.flags.get("used_this_turn", False) is True, "cost paid immediately, same as before the redesign"
    assert state.mana_subdecision is not None and state.mana_subdecision["stage"] == "choose_color"

    # Exclusive priority: no other filter row is reachable while the
    # subdecision is open, via legal_action_mask's own dispatch.
    mask = legal_action_mask(state, filter_actions)
    assert not any(mask[i] for i, (n, _l, _e) in enumerate(filter_actions) if n.startswith("Filter "))
    for color in game.COLORS:
        assert mask[_action_index(filter_actions, f"Produce {color}")], (
            f"Produce {color} must be legal mid choose_color stage -- Barrels' filter_mana spec is all 5 true colors"
        )

    produce_b_idx = _action_index(filter_actions, "Produce B")
    _, _produce_b_legal, produce_b_execute = filter_actions[produce_b_idx]
    produce_b_execute(state)

    assert state.mana_pool == {"B": 1}  # U spent at step 1, B produced at step 2 -- net one pip converted
    # A filter's output is never single-pip-tagged: a deliberate pool->pool
    # conversion, not reflexive tapping (float_mana's taggable=False).
    assert state.mana_pool_single_pip == {}
    assert state.mana_subdecision is None
    assert not legal_pay_u(state)  # Barrels' once-per-turn gate now closed


def test_mana_filter_same_color_round_trip_erases_single_pip_tag():
    # KNOWN, OWNER-APPROVED GAP (2026-08, see _filter_mana_execute's own
    # on_choose_color comment): running an already-TAGGED (avoidable,
    # single-pip-sourced) floating pip through a filter -- even choosing the
    # SAME output color back -- erases its tag for zero pool-size cost,
    # because game.spend_one_pip pays the filter's {1} out of the only
    # (tagged) pip of that color, and float_mana(..., taggable=False) never
    # re-tags the output. Locked in here as deliberate (filters are
    # once-per-turn per source, so the exploitable surface is small), not a
    # regression to fix.
    filter_decklist = [("Barrels of Blasting Jelly", 2)]
    filter_actions = build_action_table(filter_decklist, game.EFFECT_REGISTRY)

    barrels = Permanent(game.CARD_DEFS["Barrels of Blasting Jelly"])
    forest = Permanent(game.CARD_DEFS["Forest"])
    state = GameState(on_the_play=True)
    # Speculative floating is the active player's own main phase only; a
    # bare GameState has no phase, which would fail that gate first.
    state.phase = game.turn.Phase.MAIN1
    state.battlefield = [barrels, forest]

    game.activate_mana_source(state, forest)  # a real, avoidable single land tap
    assert state.mana_pool == {"G": 1}
    assert state.mana_pool_single_pip == {"G": 1}  # tagged -- this pip WOULD count if later burnt

    pay_g_idx = _action_index(filter_actions, "Filter Barrels of Blasting Jelly, paying G")
    _, legal_pay_g, execute_pay_g = filter_actions[pay_g_idx]
    assert legal_pay_g(state)
    execute_pay_g(state)  # spends the only (tagged) G pip -- both dict entries deleted
    assert state.mana_pool == {}
    assert state.mana_pool_single_pip == {}

    produce_g_idx = _action_index(filter_actions, "Produce G")
    _, _produce_g_legal, produce_g_execute = filter_actions[produce_g_idx]
    produce_g_execute(state)  # choose the SAME color, G, back out

    assert state.mana_pool == {"G": 1}  # net zero pool-size change...
    assert state.mana_pool_single_pip == {}  # ...but the avoidable tag is gone, by design


def test_mana_filter_row_count_stays_flat_not_cross_product():
    # Regression guard against re-introducing the output_color x input_color
    # cross product: monster_tron runs both real filter cards. POOL_COLORS
    # (6) rows per source, nothing more -- output color is chosen via the
    # shared "Produce <color>" buttons, not a per-row dimension.
    tron_decklist = game.parse_decklist_file(str(DATA_DIR / "monster_tron.txt"))
    tron_actions = build_action_table(tron_decklist, game.EFFECT_REGISTRY)
    filter_rows = [n for n, _l, _e in tron_actions if n.startswith("Filter ")]
    assert len(filter_rows) == 2 * len(game.POOL_COLORS) == 12, filter_rows
    produce_rows = [n for n, _l, _e in tron_actions if n.startswith("Produce ")]
    assert len(produce_rows) == len(game.COLORS) == 5, produce_rows


def test_mana_subdecision_rows_cache_stays_bounded():
    # _mana_subdecision_rows_cache persists across sweeps, so it must never
    # grow without bound when many distinct `actions` tables are looked up
    # over a process's lifetime.
    tron_decklist = game.parse_decklist_file(str(DATA_DIR / "monster_tron.txt"))
    _actions_table._mana_subdecision_rows_cache.clear()
    for _ in range(_actions_table._MANA_SUBDECISION_ROWS_CACHE_CAP + 10):
        fresh_actions = build_action_table(tron_decklist, game.EFFECT_REGISTRY)
        _actions_table._mana_subdecision_rows(fresh_actions)
        assert len(_actions_table._mana_subdecision_rows_cache) <= _actions_table._MANA_SUBDECISION_ROWS_CACHE_CAP
    _actions_table._mana_subdecision_rows_cache.clear()


def test_mana_filter_mid_pay_cost_completes_without_disturbing_original_payment():
    # state.pending_resolution is a single slot, so filtering can't be an
    # ordinary nested pay_cost. This drives the two-step subdecision all the
    # way through mid an already-open pay_cost: the original
    # pending_resolution must stay untouched and still completable after.
    tron_decklist = game.parse_decklist_file(str(DATA_DIR / "monster_tron.txt"))
    tron_actions = build_action_table(tron_decklist, game.EFFECT_REGISTRY)
    state = GameState(on_the_play=True, players=[PlayerState(True), PlayerState(False)])
    # Speculative floating is the active player's own main phase only; a
    # bare GameState has no phase, which would fail that gate first.
    state.phase = game.turn.Phase.MAIN1
    state.battlefield = [Permanent(game.CARD_DEFS["Barrels of Blasting Jelly"])]
    state.mana_pool.update({"G": 2})  # a SPARE green pip -- converting one away can't strand the {G} still owed

    game.begin_pay_cost(state, {"G": 1}, on_complete=lambda s: s.log_event("payment_complete"))
    assert state.pending_resolution["kind"] == "pay_cost" and state.pending_resolution["remaining"] == {"G": 1}

    pay_idx = _action_index(tron_actions, "Filter Barrels of Blasting Jelly, paying G")
    _, legal_pay, execute_pay = tron_actions[pay_idx]
    assert legal_pay(state), "a SPARE pip of the owed color must stay convertible mid-payment"
    execute_pay(state)
    assert state.mana_subdecision is not None and state.mana_subdecision["stage"] == "choose_color"
    assert state.pending_resolution["kind"] == "pay_cost" and state.pending_resolution["remaining"] == {"G": 1}, (
        "the original payment must be completely untouched while the subdecision is open"
    )
    assert state.mana_pool == {"G": 1}  # the spare pip spent; the still-owed one untouched

    produce_idx = _action_index(tron_actions, "Produce W")
    _, _produce_legal, produce_execute = tron_actions[produce_idx]
    produce_execute(state)
    assert state.mana_subdecision is None
    assert state.mana_pool == {"G": 1, "W": 1}
    assert state.pending_resolution["kind"] == "pay_cost", "control returns to the ORIGINAL pending, unchanged"

    # The original payment is still completable -- the invariant that matters.
    assert "G" in game.pool_spend_options(state)
    game.execute_pool_spend(state, "G")
    assert state.pending_resolution is None


def test_mana_row_generation_derives_from_registry_kind():
    # Row count per mana source is derived from the registry's "mana" spec
    # kind, not a blanket no-color-plus-6-colors for every source. Three
    # shapes in one deck: a non-land fixed source (Llanowar Elves) needs only
    # its no-color row; a land fixed source (Mountain) still needs every
    # color the full registry could ever grant it (an opponent's Abundant
    # Growth can enchant this deck's own Mountain); a non-land flexible
    # source (Bonder's Ornament) needs only its real colors and no no-color
    # row.
    decklist = [("Llanowar Elves", 4), ("Mountain", 4), ("Bonder's Ornament", 2)]
    actions = build_action_table(decklist, game.EFFECT_REGISTRY)
    labels = [name for name, _l, _e in actions]

    # Llanowar Elves: fixed G, non-land, no grant ever reaches a creature --
    # exactly one row.
    assert labels.count("Tap Llanowar Elves") == 1
    assert not any(l.startswith("Tap Llanowar Elves for") for l in labels)

    # Mountain: fixed R, but a land -- keeps a row for every color the full
    # registry declares grantable, even with no Abundant Growth in this deck.
    assert labels.count("Tap Mountain") == 1
    for color in game.COLORS:
        assert f"Tap Mountain for {color}" in labels
    assert "Tap Mountain for C" not in labels  # colorless is never a color choice

    # Bonder's Ornament: flexible over all 5 colors, non-land -- one row per
    # real color, and no no-color row.
    assert "Tap Bonder's Ornament" not in labels
    for color in game.COLORS:
        assert f"Tap Bonder's Ornament for {color}" in labels
    assert "Tap Bonder's Ornament for C" not in labels


def test_impulse_actions_gated_on_own_decklist():
    # Impulse is never cross-player (state.impulse is always the active
    # player's own zone), unlike pay_unless. A decklist with no impulse card
    # gets zero "Play from exile" rows.
    decklist = [("Mountain", 4), ("Lightning Bolt", 4)]
    actions = build_action_table(decklist, game.EFFECT_REGISTRY)
    assert not any(name.startswith("Play from exile") for name, _l, _e in actions)

    # A decklist that DOES have an impulse card gets them.
    impulse_decklist = [("Reckless Impulse", 4), ("Mountain", 4)]
    impulse_actions = build_action_table(impulse_decklist, game.EFFECT_REGISTRY)
    assert any(name == "Play from exile: Mountain" for name, _l, _e in impulse_actions)


def test_mana_ability_legal_mid_pay_unless():
    # A mana ability must stay legal during any pending resolution (605.1a/
    # 605.3b), including while paying a cost. Covers a Ward/Spell-Pierce-style
    # pay_unless: the payer floats mana in response, then pays.
    pu_decklist = [("Mountain", 4)]
    pu_actions = build_action_table(pu_decklist, game.EFFECT_REGISTRY)
    pu_tap_idx = _action_index(pu_actions, "Tap Mountain")
    pu_pay_idx = _action_index(pu_actions, "Pay (unless)")
    pu_state = GameState(on_the_play=True, players=[PlayerState(True), PlayerState(False)])
    pu_mtn = Permanent(game.CARD_DEFS["Mountain"])
    pu_state.players[0].battlefield = [pu_mtn]

    pu_results = []
    game.begin_pay_unless(pu_state, 0, {"generic": 1}, lambda s, paid: pu_results.append(paid))
    assert pu_state.pending_resolution["kind"] == "pay_unless"
    # No phase is set, so nothing here is legal via the main-phase rule --
    # everything below is the payment window (605.3a).
    assert legal_action_mask(pu_state, pu_actions)[pu_tap_idx]  # CAN tap right now, mid-resolution
    # Payable before anything is floated: cast-then-pay counts the untapped
    # Mountain, so the payer isn't required to pre-float to be offered the choice.
    assert legal_action_mask(pu_state, pu_actions)[pu_pay_idx]

    pu_actions[pu_tap_idx][2](pu_state)
    assert pu_state.pending_resolution["kind"] == "pay_unless"  # untouched by the tap
    assert legal_action_mask(pu_state, pu_actions)[pu_pay_idx]  # and still payable once floated

    pu_actions[pu_pay_idx][2](pu_state)
    while pu_state.pending_resolution is not None and pu_state.pending_resolution["kind"] == "pay_cost":
        game.execute_pool_spend(pu_state, game.pool_spend_options(pu_state)[0])
    assert pu_results == [True]


def test_mana_ability_legal_mid_any_pending_resolution():
    # Generalize past pay_unless: a mana ability must stay legal mid-
    # resolution of an entirely unrelated pending kind too (discard), and
    # activating it must not disturb that resolution at all.
    mid_decklist = [("Mountain", 4), ("Lightning Bolt", 2)]
    mid_actions = build_action_table(mid_decklist, game.EFFECT_REGISTRY)
    mid_tap_idx = _action_index(mid_actions, "Tap Mountain")
    mid_state = GameState(on_the_play=True)
    mid_mtn = Permanent(game.CARD_DEFS["Mountain"])
    mid_state.battlefield = [mid_mtn]
    mid_state.hand = [game.CARD_DEFS["Lightning Bolt"]]
    # The discard is an unrelated pending, not a payment, so this float is
    # speculative and needs the main phase (_mana_timing_legal).
    mid_state.phase = game.turn.Phase.MAIN1

    discard_completed = []
    game.begin_discard(mid_state, 1, False, on_complete=lambda s, cards: discard_completed.append(cards))
    assert mid_state.pending_resolution["kind"] == "discard"
    assert legal_action_mask(mid_state, mid_actions)[mid_tap_idx]  # legal even mid-discard now

    mid_actions[mid_tap_idx][2](mid_state)
    assert mid_state.mana_pool == {"R": 1}
    assert mid_state.pending_resolution["kind"] == "discard"  # untouched -- still open, still needs its own answer

    game.execute_discard_option(mid_state, "Lightning Bolt")
    assert discard_completed == [[game.CARD_DEFS["Lightning Bolt"]]]
    assert mid_state.pending_resolution is None


def test_mana_subdecision_does_not_hijack_the_other_seat():
    """A mana subdecision claims exclusive priority -- its owner may take no
    other action until it completes, modeling a mana ability as atomic.
    state.mana_subdecision is a single global slot, so it must carry an
    owner: without one, exclusivity would land on whichever seat asks next
    rather than the one that opened it.

    Pins both halves: the non-owner is unaffected, and the owner still gets
    exclusive priority (is forced to finish)."""
    saruli_decklist = [("Saruli Caretaker", 2), ("Slippery Bogle", 2)]
    actions = build_action_table(saruli_decklist, game.EFFECT_REGISTRY)

    saruli = Permanent(game.CARD_DEFS["Saruli Caretaker"])
    saruli.summoning_sick = False
    bogle = Permanent(game.CARD_DEFS["Slippery Bogle"])
    state = GameState(on_the_play=True, players=[PlayerState(True), PlayerState(False)])
    # Speculative floating is the active player's own main phase only; a
    # bare GameState has no phase, which would fail that gate first.
    state.phase = game.turn.Phase.MAIN1
    state.players[1].battlefield = [saruli, bogle]

    # Seat 1 opens the subdecision while it is the active seat, and must also
    # be the turn player: speculative floating requires turn ownership, not
    # just the phase.
    state.active_idx = 1
    state.turn_player_idx = 1
    tap_idx = _action_index(actions, "Tap Saruli Caretaker")
    assert actions[tap_idx][1](state)
    actions[tap_idx][2](state)
    assert state.mana_subdecision is not None
    assert state.mana_subdecision["owner"] == 1

    # Owner's view: still exclusive, still forced to finish.
    assert state.active_mana_subdecision is state.mana_subdecision
    assert not legal_action_mask(state, actions)[tap_idx], "owner keeps exclusive priority"

    # The OTHER seat must not see it at all -- this is the crash.
    state.active_idx = 0
    assert state.active_mana_subdecision is None, "a subdecision must never govern another seat's decision"
    assert state.mana_subdecision is not None, "...while still being genuinely open for its owner"

    # And the other seat's own action legality is evaluated normally rather
    # than being suppressed by someone else's exclusive priority.
    from rl.decision.action_bridge import any_pointer_legal
    state.active_idx = 0
    assert any_pointer_legal(state) is False, (
        "with no pending of its own, seat 0 has no pointer decision -- and must NOT "
        "inherit the choose_target answer from seat 1's subdecision"
    )


def test_reach_blocker_vs_flying_attacker_is_not_a_no_op():
    # The "Assign Blocker" legality check and the per-attacker predicate its
    # executor passes down must agree, or the action becomes legal-but-
    # unfulfillable and the declare-blockers round never ends.
    #
    # game.can_block lets reach block a flier, and creature_block_eligible
    # (hence _assign_blocker_legal) consults it, so a reach blocker facing
    # only fliers must be correctly offered and must actually be able to
    # record a block against one.
    decklist = [("Generous Ent", 2), ("Sneaky Snacker", 2), ("Forest", 8)]
    actions = _actions_table.build_action_table(decklist, game.EFFECT_REGISTRY, opponent_decklist=decklist)
    _, ent_legal, ent_execute = actions[_action_index(actions, "Assign Blocker: Generous Ent (slot 1)")]

    attacker = Permanent(game.CARD_DEFS["Sneaky Snacker"])
    attacker.tapped = True  # already attacking
    ent = Permanent(game.CARD_DEFS["Generous Ent"])
    state = GameState(on_the_play=True, players=[PlayerState(True), PlayerState(False)])
    state.players[0].battlefield = [attacker]
    state.players[0].attackers = [attacker]
    state.players[1].battlefield = [ent]
    state.active_idx = 1  # _declare_blockers_gen's own flip to the defender

    assert game.has_keyword(state, attacker, "flying"), "fixture assumes Sneaky Snacker flies"
    assert game.has_keyword(state, ent, "reach"), "fixture assumes Generous Ent has reach"
    assert game.can_block(state, ent, attacker), "reach must be able to block a flier"

    game.begin_declare_blockers(state, on_complete=lambda s: None)
    assert ent_legal(state), "reach blocker vs flying attacker must be offered"

    ent_execute(state)
    # Must open a real choice that includes the flier, not silently re-open
    # declare_blockers with nothing recorded.
    assert state.pending_resolution is not None
    assert state.pending_resolution["kind"] == "choose_opponent_permanent", (
        f"expected a real attacker choice, got {state.pending_resolution['kind']!r} -- "
        "a no-op re-open is the infinite-loop bug"
    )
    assert game.choose_opponent_permanent_options(state) == [("Sneaky Snacker", 1)]
    game.execute_choose_opponent_permanent_option(state, "Sneaky Snacker", 1)
    assert state.players[0].blocked_by == {attacker: [ent]}, "the block must actually be recorded"


def test_attacker_that_left_the_battlefield_is_removed_from_combat():
    # Same legal-but-unfulfillable shape as the reach test above, reached by
    # a different divergence: 506.4 says a permanent removed from the
    # battlefield stops being an attacking creature, so state.attackers must
    # be pruned when a declared attacker dies. Otherwise the two halves of
    # "Assign Blocker" disagree -- creature_block_eligible iterates
    # attackers, choose_opponent_permanent_options iterates battlefield --
    # and the action stays legal with nothing to fulfil it.
    decklist = [("Generous Ent", 2), ("Sneaky Snacker", 2), ("Forest", 8)]
    actions = _actions_table.build_action_table(decklist, game.EFFECT_REGISTRY, opponent_decklist=decklist)
    _, ent_legal, ent_execute = actions[_action_index(actions, "Assign Blocker: Generous Ent (slot 1)")]

    attacker = Permanent(game.CARD_DEFS["Sneaky Snacker"])
    attacker.tapped = True  # already attacking
    ent = Permanent(game.CARD_DEFS["Generous Ent"])
    state = GameState(on_the_play=True, players=[PlayerState(True), PlayerState(False)])
    state.players[0].battlefield = [attacker]
    state.players[0].attackers = [attacker]
    state.players[1].battlefield = [ent]
    state.active_idx = 1  # _declare_blockers_gen's own flip to the defender

    # The removal spell resolves: the attacker leaves the battlefield.
    from game.effects.state_based import destroy_permanent
    destroy_permanent(state, attacker)
    assert attacker not in state.players[0].battlefield, "fixture: the attacker must be gone"

    assert attacker not in state.players[0].attackers, (
        "506.4: a permanent removed from the battlefield stops being an attacking creature"
    )

    game.begin_declare_blockers(state, on_complete=lambda s: None)
    assert state.pending_resolution is None, (
        "nothing is attacking any more -- the declare-blockers step must not open at all"
    )
    assert not ent_legal(state), "no attackers on the battlefield means no blocker is assignable"


def test_assign_damage_to_opponent_is_permanently_dead():
    """Trample-to-player is never an agent choice: it's a forced, automatic
    outcome of assign_combat_damage's own resolution once every blocker is
    at its lethal cap (702.19b/510.1c). This row stays registered only so
    the fixed action table's length doesn't change; it must never actually
    be reachable."""
    decklist = [("Reckless Lackey", 4), ("Mountain", 8)]
    actions = _actions_table.build_action_table(decklist, game.EFFECT_REGISTRY)
    _, legal, execute = actions[_action_index(actions, "Assign combat damage to opponent")]
    state = GameState(on_the_play=True, players=[PlayerState(True), PlayerState(False)])
    assert not legal(state), "always False -- no pending resolution needed to prove it"
    with pytest.raises(AssertionError):
        execute(state)
