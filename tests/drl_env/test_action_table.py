"""Tests for drl_env's action table (drl_env._actions_table and its sibling
category modules): build_action_table, legal_action_mask, and the Plot/
on-cast-trigger/token/targeting/combat mechanics they wire together, end to
end through the REAL production functions -- never a parallel
reimplementation."""

from pathlib import Path

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
    # Exercises Plot and the on-cast trigger hook (item 11) through the REAL
    # _plot_legal/_plot_execute/_cast_from_exile_legal/_cast_from_exile_execute
    # functions -- not a parallel reimplementation. No real Plot/Guttersnipe
    # card exists yet (deck assembly out of scope), so this temporarily
    # injects into the global game.CARD_DEFS/game.EFFECT_REGISTRY, saving/
    # restoring both.
    _card_defs_backup = dict(game.CARD_DEFS)
    _filler_backup = game.EFFECT_REGISTRY[EffectId.FILLER]
    _generous_ent_backup = game.EFFECT_REGISTRY[EffectId.GENEROUS_ENT]

    PLOT_COST = {"generic": 1, "B": 1}  # {B}, not {R} -- EffectId.SWAMP is a real, already-correctly-wired
    # mana source (registry.py's per-effect_id EFFECT_REGISTRY entries are consulted directly by
    # mana_ability_options/activate_mana_source; injecting a fake "mana" spec onto FILLER wouldn't
    # be picked up as a real source's real color, so reusing a real fixed-color land sidesteps that).

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
        state.phase = game.turn.Phase.MAIN1  # Plot Speed defaults to SORCERY (a CardType.SORCERY card, no override) -- needs a sorcery-speed phase to be legal at all now
        state.hand = [plot_spell]
        state.battlefield = [
            Permanent(CardDef("Swamp", CardType.LAND, None, EffectId.SWAMP)),
            Permanent(CardDef("Swamp", CardType.LAND, None, EffectId.SWAMP)),
            Permanent(CardDef("Guttersnipe-ish", CardType.CREATURE, None, EffectId.GENEROUS_ENT)),
        ]

        # Plot it: pay {1}{B}, exile with this turn's stamp. Float-first: float
        # {B}{B} from the two Swamps BEFORE plotting (1 generic + 1 B needed --
        # a mana ability floats immediately and never uses the stack), then pay
        # by spending from the pool.
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
        # on_cast now QUEUES the trigger (faithful timing) -- it fires only
        # once game.turn's priority round promotes it onto the stack (above
        # the spell) and it resolves, not inline at cast.
        assert on_cast_calls == []
        assert [e["type"] for e in state.trigger_queue] == ["cast_trigger"]
        game.promote_triggers_to_stack(state)
        game.resolve_top_of_stack(state)
        assert on_cast_calls == ["Guttersnipe-ish"]

        # extra_legal gate on the cast-from-exile path (Highway Robbery's
        # own need: Plot waives the mana cost, not other costs a normal
        # cast's extra_legal already checks). Re-plot, then simulate an
        # extra_legal that's never satisfiable.
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
    # Regression: on_cast_trigger (Guttersnipe) must fire only once a spell's
    # cost is actually paid -- queued in _cast_execute's _after_pay, never
    # inline the instant the cast is announced. Float-first: float {B}, announce
    # the cast (opens pay_cost), confirm the trigger has NOT fired while payment
    # is still pending, then spend to complete it and confirm it queues, then
    # fires only on resolve.
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

        game.activate_mana_source(state, swamp)  # float {B} BEFORE casting (float-first)
        assert cast_legal(state)
        cast_execute(state)
        assert state.pending_resolution["kind"] == "pay_cost"
        assert on_cast_calls == []  # NOT fired yet -- the cost isn't paid
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
    """Regression: real Flashback cost is {2}{R} -- FAITHLESS_LOOTING's
    registry entry used to omit "cost" entirely, so _flashback_legal skipped
    plan_payment and the action showed legal with 0 floating mana (live
    game: flashed back for free, no cost ever paid)."""
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
        game.activate_mana_source(state, m)  # float 3 R BEFORE flashing back (float-first)
    assert fb_legal(state)
    fb_execute(state)
    assert state.pending_resolution["kind"] == "pay_cost"
    while state.pending_resolution is not None:
        game.execute_pool_spend(state, game.pool_spend_options(state)[0])
    assert state.mana_pool == {}  # the full {2}{R} = 3 pips actually spent, not skipped
    assert fl not in state.graveyard and len(state.stack) == 1


def test_tokens_blood_sacrifice():
    # Tokens (item 8): build_action_table's token_card_defs param is what
    # actually makes "Activate Blood (sac)" exist as an action at all --
    # "Blood" is never a decklist name, so the plain distinct_names-driven
    # loop alone (used by every other activated ability) can't find it.
    empty_decklist = []
    no_token_actions = build_action_table(empty_decklist, game.EFFECT_REGISTRY)
    assert not any("Blood" in nm for nm, _l, _e in no_token_actions)  # opt-in: omitted => absent, zero effect on existing decks

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
    game.activate_mana_source(state, swamp)  # float-first: float {B} BEFORE activating (Blood's {1})
    assert activate_legal(state)
    activate_execute(state)  # pays {1} via the real begin_pay_cost path, same as every other cost_key ability
    assert state.pending_resolution["kind"] == "pay_cost"
    # The floated B pays the {generic:1} (this engine never auto-spends floated
    # mana toward generic -- mana.py's own design -- so spend it explicitly).
    game.execute_pool_spend(state, game.pool_spend_options(state)[0])

    assert state.pending_resolution["kind"] == "discard"  # Blood's discard -- a COST, paid before the effect
    game.execute_discard_option(state, "Card To Discard")
    assert state.pending_resolution is None
    assert [p.card_def.name for p in state.battlefield] == ["Swamp"]  # Blood is gone, never added to any zone
    # The DRAW is Blood's effect -- on the stack now (faithful timing), not
    # fired the instant its costs (sac + discard) were paid. Resolve it.
    assert len(state.stack) == 1 and state.hand == []
    game.resolve_top_of_stack(state)
    assert [c.name for c in state.hand] == ["Library Card"]  # discarded one, drew one


def test_cross_player_targeting():
    # Cross-player targeting: build_action_table's
    # opponent_decklist/opponent_token_card_defs params register "Choose
    # opponent's: X (slot k)" actions from the OTHER side's own card pool
    # -- blocking's first consumer, but exercised standalone here since
    # blocking itself isn't built yet. Boggles on both sides -- what's
    # under test is MY OWN action table's opponent-facing entries, not
    # anything about my own cards, so a real decklist with real creature
    # quantities on both sides is all this needs.
    boggles_decklist = _boggles_decklist()

    no_opponent_actions = build_action_table(boggles_decklist, game.EFFECT_REGISTRY)
    assert not any(nm.startswith("Choose opponent's:") for nm, _l, _e in no_opponent_actions)  # 1p mode: never registered at all

    my_actions = build_action_table(boggles_decklist, game.EFFECT_REGISTRY, opponent_decklist=boggles_decklist)
    bogle_slot_actions = [nm for nm, _l, _e in my_actions if nm.startswith("Choose opponent's: Slippery Bogle")]
    assert bogle_slot_actions == [f"Choose opponent's: Slippery Bogle (slot {k})" for k in range(1, 5)]  # boggles.txt: 4 copies
    # Registered for every opponent choosable name, not just creatures -- same
    # "pre-register broadly, mask precisely" pattern as "Choose target: X"
    # (own side) above. A creature-only filter here used to omit non-creature
    # names (e.g. this Forest), which silently broke the instant a predicate
    # legitimately wanted a non-creature opponent permanent (Masked Vandal's
    # ETB targets an opponent artifact/enchantment) -- an all-False mask
    # crash, since begin_choose_opponent_permanent found a real option but no
    # action row existed to select it.
    forest_slot_actions = [nm for nm, _l, _e in my_actions if nm.startswith("Choose opponent's: Forest")]
    assert forest_slot_actions == [f"Choose opponent's: Forest (slot {k})" for k in range(1, 13)]  # boggles.txt: 12 Forests

    target_slot_2 = _action_index(my_actions, "Choose opponent's: Slippery Bogle (slot 2)")
    target_slot_1 = _action_index(my_actions, "Choose opponent's: Slippery Bogle (slot 1)")

    attacker_bogle_1 = Permanent(game.CARD_DEFS["Slippery Bogle"])
    attacker_bogle_2 = Permanent(game.CARD_DEFS["Slippery Bogle"])
    attacker_bogle_2.slot = 2
    state = GameState(on_the_play=True, players=[PlayerState(True), PlayerState(False)])
    state.players[0].battlefield = [attacker_bogle_1, attacker_bogle_2]
    state.active_idx = 1  # simulating the defender's own already-flipped perspective (see game.begin_choose_opponent_permanent's own docstring)

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
    # Turn-owner / priority-holder split:
    # _land_drop_legal (via speed_legal) and _attack_legal must both
    # refuse the non-turn player even when state.phase/state.
    # lands_played_this_turn/their own eligible creature would otherwise
    # look legal -- simulates a priority consult (active_idx flipped away
    # from turn_player_idx) without needing the full priority round built
    # yet. Rebuilds its own boggles action table (rather than depending on
    # test_cross_player_targeting's) so the two run independently of order.
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
    # Blocking: build_action_table's "Assign Blocker:
    # <name> (slot j)" / "Done blocking" entries, end to end through the
    # REAL production functions (_assign_blocker_legal/_execute,
    # _done_blocking_legal/_execute) -- not a parallel reimplementation.
    # Two attacking Slippery Bogles (real power=1 stats), one defending
    # Slippery Bogle blocks only ONE of them. Rebuilds its own boggles
    # action table (rather than depending on other tests') so this runs
    # independently of order.
    boggles_decklist = _boggles_decklist()
    my_actions = build_action_table(boggles_decklist, game.EFFECT_REGISTRY, opponent_decklist=boggles_decklist)
    boggles_pending_kinds = game.derive_pending_kinds(boggles_decklist)  # exercised for its own sake, unused beyond this line

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
    atk_bogle_2.tapped = True  # declare_attacker's own effect -- simulated directly, attacking itself isn't under test here
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

    # Re-opened (drl_env._assign_blocker_execute's own nested on_complete):
    # defender_bogle is now spoken for, so it's no longer offered again --
    # confirms creature_block_eligible actually gates the SAME action a
    # second time, not just the first.
    assert state.pending_resolution["kind"] == "declare_blockers"
    assert not assign_legal(state)
    assert done_legal(state)
    done_execute(state)  # "Done blocking" -- atk_bogle_2 goes unblocked
    assert completed == [True]
    assert state.pending_resolution is None
    assert state.players[0].blocked_by == {atk_bogle_1: [defender_bogle]}


def test_flying_blocker_restriction():
    # Flying: _assign_blocker_execute's own
    # extra_predicate (game.has_keyword), end to end through the REAL
    # action table -- Silhana Ledgewalker (real "can't be blocked except
    # by creatures with flying," modeled as the "flying" keyword) can only
    # be blocked by a creature that itself has flying (Kitchen Imp, real
    # flying) -- a plain Slippery Bogle is otherwise a perfectly legal
    # (untapped) blocker, but can never be assigned to THIS specific
    # attacker. Mixes cards from different real color catalogs
    # (green/multicolor + black) purely to exercise the engine mechanism
    # -- not a claim either card is ever actually run together in a real
    # deck.
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
    # GANG-BLOCKING eligibility: Slippery Bogle (no flying) is NOT even
    # offered here -- the only attacker (Silhana Ledgewalker, modeled with
    # flying) can only be blocked by a flyer, so Bogle has no legal target
    # and "Assign Blocker: Slippery Bogle" is illegal. Kitchen Imp (flying)
    # IS a legal blocker.
    assert not bogle_legal(state)
    assert imp_legal(state)

    imp_execute(state)  # Kitchen Imp HAS flying -- opens a real nested choice
    assert state.pending_resolution["kind"] == "choose_opponent_permanent"
    assert game.choose_opponent_permanent_options(state) == [("Silhana Ledgewalker", 1)]
    game.execute_choose_opponent_permanent_option(state, "Silhana Ledgewalker", 1)
    assert state.players[0].blocked_by == {attacking_ledgewalker: [defending_imp]}


def test_aura_targeting_exact_slot_addressing():
    # Targeting (real MTG rule, per drl_env._precast_choice_execute /
    # game.effects.casting.cast_aura's own docstrings): a target is chosen once,
    # at cast time, exact (name, slot) addressed -- not just by name -- and
    # re-validated by identity only once the spell resolves off the stack.
    # End to end through the REAL action table: "Cast Rancor" pays its {G}
    # cost, then (precast_choice, not deferred) immediately opens
    # choose_permanent with BOTH Slippery Bogles offered by their own
    # distinct slot; "Choose target: Slippery Bogle (slot 2)" picks the
    # specific one, which pushes to the stack (not yet attached, still in
    # hand); resolving the stack attaches it to exactly the one chosen, not
    # an arbitrary same-named match.
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
    targeting_actions[_action_index(targeting_actions, "Tap Forest")][2](state)  # float-first: float {G} BEFORE casting
    assert cast_rancor_legal(state)
    cast_rancor_execute(state)
    assert state.pending_resolution["kind"] == "pay_cost"
    targeting_actions[_action_index(targeting_actions, "Spend G from pool")][2](state)  # pays Rancor's {G} from the pool

    # Cost fully paid -- precast_choice means cast_aura runs its target
    # choice IMMEDIATELY here, NOT deferred to when this eventually pops
    # off the stack (that's the whole point of this redesign).
    # Aura now targets any creature on EITHER battlefield (real "Enchant
    # creature"), hexproof-aware -- the creature half rides the identity
    # pointer scheme (rl.action_bridge), so we drive it via the same
    # execute the pointer path calls, not a "Choose target:" fixed action.
    assert state.pending_resolution["kind"] == "choose_any_target"
    assert set(game.choose_any_target_creature_options(state)) == {(0, "Slippery Bogle", 1), (0, "Slippery Bogle", 2)}

    game.execute_choose_any_target_creature(state, 0, "Slippery Bogle", 2)  # the SPECIFIC slot-2 bogle
    # Target chosen -- pushed to the stack, not yet attached. The card LEFT
    # hand at cast (push_to_stack), so it can't be re-cast off the stack;
    # resolve_top_of_stack restores it transiently for the attach resolve.
    assert state.pending_resolution is None
    assert state.hand == [] and len(state.stack) == 1

    game.resolve_top_of_stack(state)
    assert state.hand == []
    rancor_permanent = next(p for p in state.battlefield if p.card_def.name == "Rancor")
    assert rancor_permanent.flags["enchanting"] is bogle_2  # the SPECIFIC one chosen -- not bogle_1, despite the identical name


def test_aura_target_fizzle():
    # Fizzle, same end-to-end path as test_aura_targeting_exact_slot_addressing:
    # the exact chosen permanent (bogle_1 this time) is gone by the time the
    # cast resolves -- the whole spell fails, no effect, straight to the
    # graveyard, never attaches. Rebuilds its own action table/rancor_card
    # (fresh, deterministic) rather than depending on the other test's
    # objects, so the two run independently of order.
    targeting_decklist = [("Slippery Bogle", 2), ("Rancor", 2), ("Forest", 10)]
    targeting_actions = build_action_table(targeting_decklist, game.EFFECT_REGISTRY)
    rancor_card = game.CARD_DEFS["Rancor"]

    state = GameState(on_the_play=True)
    state.phase = game.turn.Phase.MAIN1
    bogle_1 = Permanent(game.CARD_DEFS["Slippery Bogle"])
    forest = Permanent(game.CARD_DEFS["Forest"])
    state.battlefield = [bogle_1, forest]
    state.hand = [rancor_card]

    targeting_actions[_action_index(targeting_actions, "Tap Forest")][2](state)  # float-first: float {G} first
    targeting_actions[_action_index(targeting_actions, "Cast Rancor")][2](state)
    targeting_actions[_action_index(targeting_actions, "Spend G from pool")][2](state)
    game.execute_choose_any_target_creature(state, 0, "Slippery Bogle", 1)  # Aura targets via the any-target pointer path now
    assert len(state.stack) == 1
    state.battlefield.remove(bogle_1)  # dies before the cast resolves

    game.resolve_top_of_stack(state)
    assert state.hand == []
    assert any(c.card_def is rancor_card for c in state.graveyard)  # fizzled to graveyard as a fresh instance (400.7)
    assert not any(p.card_def.name == "Rancor" for p in state.battlefield)


def test_mana_ability_options_cache_no_stale_leak():
    # _mana_ability_options_cache memoization never returns a stale answer:
    # build a state with exactly 1 untapped Mountain, sweep the mask
    # (populating the cache -- "Tap Mountain" legal), activate it (a real
    # mutation -- 0 untapped sources left, so the ability is gone), then sweep
    # again -- the second sweep must see the mutation, not the first sweep's
    # cached answer, proving the cache doesn't leak across separate
    # legal_action_mask calls.
    perf_decklist = [("Mountain", 10), ("Lightning Bolt", 5)]
    perf_actions = build_action_table(perf_decklist, game.EFFECT_REGISTRY)
    perf_tap_mountain = _action_index(perf_actions, "Tap Mountain")
    perf_state = GameState(on_the_play=True, players=[PlayerState(True)])
    perf_mtn = Permanent(game.CARD_DEFS["Mountain"])
    perf_state.battlefield = [perf_mtn]
    # Speculative floating is main-phase-only (see _mana_timing_legal); the
    # default GameState carries no phase at all, which would fail the timing
    # gate before the cache this test is actually about ever gets consulted.
    perf_state.phase = game.turn.Phase.MAIN1
    assert legal_action_mask(perf_state, perf_actions)[perf_tap_mountain]
    assert _actions_mana._mana_ability_options_cache is None  # cleared again once the sweep itself returns
    game.activate_mana_source(perf_state, perf_mtn)  # taps the only Mountain -- 0 untapped sources left
    assert ("Mountain", None) not in game.mana_ability_options(perf_state)  # ground truth: nothing left to tap
    assert not legal_action_mask(perf_state, perf_actions)[perf_tap_mountain]  # would be wrongly True if the first sweep's stale cache leaked through


def test_saruli_caretaker_two_stage_mana_subdecision():
    # Saruli Caretaker (Option A: gate-free two-stage mana_subdecision, see
    # state.mana_subdecision's own docstring): its extra cost -- tap ANOTHER
    # untapped creature -- is a COST CHOICE (602.5g), decided first ("Tap
    # Saruli Caretaker", gate-free, opens the mana_subdecision), then the
    # tap-TARGET (pointer-routed in real play, rl.action_bridge -- exercised
    # there; here the resulting state transition is driven directly via
    # game.execute_mana_subdecision_target, same call the pointer dispatch
    # makes), THEN the color (shared "Produce <color>" buttons, gated via
    # legal_action_mask's own _mana_subdecision_gate dispatch, not this
    # closure's own pending-agnostic check) -- matching real sequencing (cost
    # paid before the ability's own color-choice effect resolves), never the
    # stack (605.1a).
    saruli_decklist = [("Saruli Caretaker", 2), ("Slippery Bogle", 2)]
    saruli_actions = build_action_table(saruli_decklist, game.EFFECT_REGISTRY)

    saruli = Permanent(game.CARD_DEFS["Saruli Caretaker"])
    saruli.summoning_sick = False  # Saruli's own {T} ability needs this (302.6) -- the TARGET's own sickness doesn't matter (being tapped as a cost isn't a {T} ability of its own)
    bogle_1 = Permanent(game.CARD_DEFS["Slippery Bogle"])
    bogle_2 = Permanent(game.CARD_DEFS["Slippery Bogle"])
    bogle_2.slot = 2
    bogle_2.tapped = True  # already tapped -- would be excluded downstream, not relevant to this row's own aggregate legality
    state = GameState(on_the_play=True)
    # Speculative floating is the active player's own main phase only
    # (drl_env._actions_mana._mana_timing_legal); a bare GameState has no
    # phase at all, which would fail that gate before reaching what this
    # test is actually about.
    state.phase = game.turn.Phase.MAIN1
    state.battlefield = [saruli, bogle_1, bogle_2]

    tap_idx = _action_index(saruli_actions, "Tap Saruli Caretaker")
    _, tap_legal, tap_execute = saruli_actions[tap_idx]
    assert tap_legal(state)  # at least one OTHER untapped creature exists (bogle_1) -- aggregate check, no per-target/color row anymore
    tap_execute(state)
    sub = state.mana_subdecision
    assert sub is not None and sub["stage"] == "choose_target" and sub["source"] is saruli and sub["target"] is None, (
        "must open a mana_subdecision at the choose_target stage, source resolved to THIS Saruli"
    )

    # Same row is no longer even reachable while the sub-decision is open --
    # legal_action_mask's own exclusive-priority dispatch, not this closure's
    # own logic (tap_legal itself doesn't check mana_subdecision at all).
    assert not legal_action_mask(state, saruli_actions)[tap_idx]

    game.execute_mana_subdecision_target(state, bogle_1)  # same call rl.action_bridge's pointer dispatch makes -- exercised end to end there
    assert state.mana_subdecision["stage"] == "choose_color" and state.mana_subdecision["target"] is bogle_1
    assert not bogle_1.tapped, "the target isn't tapped until the color is actually chosen (execute_mana_subdecision_color's own order)"

    color_idx = _action_index(saruli_actions, "Produce G")
    mask = legal_action_mask(state, saruli_actions)
    assert mask[color_idx], "Produce G must be legal mid choose_color stage (Saruli's own flexible 5-color spec)"
    assert not mask[tap_idx], "Tap Saruli Caretaker must stay illegal -- mana_subdecision still open"
    _, _color_legal, color_execute = saruli_actions[color_idx]
    color_execute(state)

    assert state.mana_subdecision is None
    assert saruli.tapped and bogle_1.tapped and bogle_2.tapped  # bogle_2 was already tapped, untouched otherwise
    assert state.mana_pool == {"G": 1}
    assert not tap_legal(state)  # Saruli itself now tapped -- no untapped source left, no longer offered at all


def test_mana_filter_two_step_color_choice():
    # Mana filter (Barrels of Blasting Jelly / Conduit Pylons), two-step
    # redesign reusing Saruli Caretaker's own choose_color mana_subdecision
    # machinery (game.begin_mana_color_choice): step 1, a flat fixed-table
    # row "Filter X, paying <input>", pays the cost immediately (taps/flags
    # the source, spends the input pip) and opens the shared choose_color
    # stage; step 2, one of the same "Produce <color>" buttons Saruli uses,
    # produces the chosen output color into the pool. No nested pay_cost --
    # state.mana_subdecision stays a field separate from
    # state.pending_resolution throughout. Float {U}: paying with it is
    # legal only while that U pip is actually floating; illegal for a color
    # not currently floating.
    filter_decklist = [("Barrels of Blasting Jelly", 2)]
    filter_actions = build_action_table(filter_decklist, game.EFFECT_REGISTRY)

    barrels = Permanent(game.CARD_DEFS["Barrels of Blasting Jelly"])
    state = GameState(on_the_play=True)
    # Speculative floating is the active player's own main phase only
    # (drl_env._actions_mana._mana_timing_legal); a bare GameState has no
    # phase at all, which would fail that gate before reaching what this
    # test is actually about.
    state.phase = game.turn.Phase.MAIN1
    state.battlefield = [barrels]
    state.mana_pool = {"U": 1}

    pay_u_idx = _action_index(filter_actions, "Filter Barrels of Blasting Jelly, paying U")
    pay_w_idx = _action_index(filter_actions, "Filter Barrels of Blasting Jelly, paying W")
    _, legal_pay_u, execute_pay_u = filter_actions[pay_u_idx]
    _, legal_pay_w, _execute_pay_w = filter_actions[pay_w_idx]

    # Exactly one row per POOL_COLOR (no output_color cross product anymore):
    # the redesign's own stated goal -- 6 rows for this one source, not 30.
    filter_rows = [n for n, _l, _e in filter_actions if n.startswith("Filter Barrels of Blasting Jelly")]
    assert len(filter_rows) == len(game.POOL_COLORS) == 6, filter_rows

    # A deck with a filter card but NO Saruli-shaped mana_extra_choose card
    # must still get the shared "Produce <color>" rows -- build_action_table's
    # needs_mana_subdecision_color_buttons gate must trigger on filter_mana
    # too, not just mana_extra_choose.
    for color in game.COLORS:
        assert f"Produce {color}" in [n for n, _l, _e in filter_actions]

    assert legal_pay_u(state)
    assert not legal_pay_w(state)  # no floating W to spend

    execute_pay_u(state)
    assert state.mana_pool == {}, "the U pip is spent immediately as the cost -- no output produced yet"
    assert barrels.flags.get("used_this_turn", False) is True, "cost paid immediately, same as before the redesign"
    assert state.mana_subdecision is not None and state.mana_subdecision["stage"] == "choose_color"

    # Exclusive priority: no OTHER filter row (this source's own gate is
    # already closed anyway, but even a hypothetical second untapped source
    # would be suppressed) is reachable while the subdecision is open --
    # legal_action_mask's own dispatch, not either closure's own logic.
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
    # A filter's output is never single-pip-tagged, even though it's
    # exactly 1 symbol -- a deliberate pool->pool conversion, not
    # reflexive tapping (game.mana.float_mana's taggable=False).
    assert state.mana_pool_single_pip == {}
    assert state.mana_subdecision is None
    assert not legal_pay_u(state)  # Barrels' own once-per-turn gate now closed -- no more floating U to spend either way


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
    # Speculative floating is the active player's own main phase only
    # (drl_env._actions_mana._mana_timing_legal); a bare GameState has no
    # phase at all, which would fail that gate before reaching what this
    # test is actually about.
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
    # Durable regression guard against re-introducing the output_color x
    # input_color cross product: monster_tron runs BOTH real filter cards
    # (Conduit Pylons + Barrels of Blasting Jelly), so its table is where the
    # old design's ~60-row block lived. Now: POOL_COLORS (6) rows per source,
    # nothing more -- output color is chosen via the shared "Produce <color>"
    # buttons, not a per-row dimension.
    tron_decklist = game.parse_decklist_file(str(DATA_DIR / "monster_tron.txt"))
    tron_actions = build_action_table(tron_decklist, game.EFFECT_REGISTRY)
    filter_rows = [n for n, _l, _e in tron_actions if n.startswith("Filter ")]
    assert len(filter_rows) == 2 * len(game.POOL_COLORS) == 12, filter_rows
    produce_rows = [n for n, _l, _e in tron_actions if n.startswith("Produce ")]
    assert len(produce_rows) == len(game.COLORS) == 5, produce_rows


def test_mana_subdecision_rows_cache_stays_bounded():
    # Regression guard: _mana_subdecision_rows_cache persists ACROSS sweeps
    # (unlike its per-sweep-reset siblings -- see its own docstring), so it
    # must never grow without bound when many distinct `actions` tables are
    # looked up over a process's lifetime (e.g. a multiprocessing worker
    # rebuilding a fresh table per task instead of reusing one per deck).
    tron_decklist = game.parse_decklist_file(str(DATA_DIR / "monster_tron.txt"))
    _actions_table._mana_subdecision_rows_cache.clear()
    for _ in range(_actions_table._MANA_SUBDECISION_ROWS_CACHE_CAP + 10):
        fresh_actions = build_action_table(tron_decklist, game.EFFECT_REGISTRY)
        _actions_table._mana_subdecision_rows(fresh_actions)
        assert len(_actions_table._mana_subdecision_rows_cache) <= _actions_table._MANA_SUBDECISION_ROWS_CACHE_CAP
    _actions_table._mana_subdecision_rows_cache.clear()


def test_mana_filter_mid_pay_cost_completes_without_disturbing_original_payment():
    # The reason filtering couldn't just be an ordinary nested pay_cost in
    # the first place: state.pending_resolution is a single slot. This
    # proves the two-step redesign genuinely preserves that guarantee --
    # opening state.mana_subdecision mid an already-open pay_cost, then
    # completing BOTH subdecision steps, must leave the ORIGINAL
    # pending_resolution completely untouched and still completable
    # afterward (not just legal to attempt, per
    # test_stranded_payment_filter_regression -- this drives it all the way
    # through, the same real scenario a training game encounters).
    tron_decklist = game.parse_decklist_file(str(DATA_DIR / "monster_tron.txt"))
    tron_actions = build_action_table(tron_decklist, game.EFFECT_REGISTRY)
    state = GameState(on_the_play=True, players=[PlayerState(True), PlayerState(False)])
    # Speculative floating is the active player's own main phase only
    # (drl_env._actions_mana._mana_timing_legal); a bare GameState has no
    # phase at all, which would fail that gate before reaching what this
    # test is actually about.
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
    # Row count per mana source is now DERIVED from the registry's own "mana"
    # spec kind (build_action_table), not a blanket no-color-plus-6-colors for
    # every source name -- a row a source can never make legal is a dead
    # output neuron. Three shapes in one deck: a non-land FIXED source
    # (Llanowar Elves) needs only its own no-color row; a LAND fixed source
    # (Mountain) still needs every color Abundant Growth's registry entry
    # could ever grant it (game.EFFECT_REGISTRY is the FULL registry, not
    # just this decklist -- an opponent's Abundant Growth can enchant this
    # deck's own Mountain); a non-land FLEXIBLE source (Bonder's Ornament)
    # needs only its own real colors, and -- unlike a fixed source -- no
    # no-color row at all.
    decklist = [("Llanowar Elves", 4), ("Mountain", 4), ("Bonder's Ornament", 2)]
    actions = build_action_table(decklist, game.EFFECT_REGISTRY)
    labels = [name for name, _l, _e in actions]

    # Llanowar Elves: fixed G, non-land, no grant ever reaches a creature --
    # exactly one row.
    assert labels.count("Tap Llanowar Elves") == 1
    assert not any(l.startswith("Tap Llanowar Elves for") for l in labels)

    # Mountain: fixed R, but a LAND -- keeps a row for every color
    # game.EFFECT_REGISTRY declares grantable (Abundant Growth's own entry),
    # even though this decklist has no Abundant Growth of its own.
    assert labels.count("Tap Mountain") == 1
    for color in game.COLORS:
        assert f"Tap Mountain for {color}" in labels
    assert "Tap Mountain for C" not in labels  # colorless is never a color CHOICE

    # Bonder's Ornament: flexible over all 5 colors, non-land -- one row per
    # real color, and NO no-color row (its native ability never offers one).
    assert "Tap Bonder's Ornament" not in labels
    for color in game.COLORS:
        assert f"Tap Bonder's Ornament for {color}" in labels
    assert "Tap Bonder's Ornament for C" not in labels


def test_impulse_actions_gated_on_own_decklist():
    # Impulse is never cross-player (state.impulse is always the ACTIVE
    # player's own zone) -- unlike pay_unless. build_action_table has no
    # pending_kinds parameter at all (an opponent's card can never pose an
    # impulse decision to a deck without one), so a decklist with no impulse
    # card gets zero "Play from exile" rows, purely from its own contents.
    decklist = [("Mountain", 4), ("Lightning Bolt", 4)]
    actions = build_action_table(decklist, game.EFFECT_REGISTRY)
    assert not any(name.startswith("Play from exile") for name, _l, _e in actions)

    # A decklist that DOES have an impulse card gets them.
    impulse_decklist = [("Reckless Impulse", 4), ("Mountain", 4)]
    impulse_actions = build_action_table(impulse_decklist, game.EFFECT_REGISTRY)
    assert any(name == "Play from exile: Mountain" for name, _l, _e in impulse_actions)


def test_mana_ability_legal_mid_pay_unless():
    # A mana ability must stay legal during ANY pending resolution (605.1a/
    # 605.3b: a mana ability may be activated even mid-resolution of something
    # else, INCLUDING while paying a cost). Covers a Ward/Spell-Pierce-style
    # pay_unless: the payer floats mana IN RESPONSE (not pre-floated), then pays.
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
    # No phase is set on this state at all, so nothing here is legal by way of
    # the main-phase rule -- everything below is the PAYMENT window (605.3a's
    # "whenever a rule or effect asks for a mana payment"), which is exactly
    # what this test exists to pin.
    assert legal_action_mask(pu_state, pu_actions)[pu_tap_idx]  # CAN tap right now, mid-resolution
    # Payable BEFORE anything is floated: cast-then-pay counts the untapped
    # Mountain, so the payer is not required to pre-float to be offered the
    # choice. Under float-first this assertion read `not ...` -- that was the
    # bug being fixed, not a property worth keeping.
    assert legal_action_mask(pu_state, pu_actions)[pu_pay_idx]

    pu_actions[pu_tap_idx][2](pu_state)
    assert pu_state.pending_resolution["kind"] == "pay_unless"  # untouched by the tap
    assert legal_action_mask(pu_state, pu_actions)[pu_pay_idx]  # and still payable once floated

    pu_actions[pu_pay_idx][2](pu_state)
    while pu_state.pending_resolution is not None and pu_state.pending_resolution["kind"] == "pay_cost":
        game.execute_pool_spend(pu_state, game.pool_spend_options(pu_state)[0])
    assert pu_results == [True]


def test_mana_ability_legal_mid_any_pending_resolution():
    # Generalize past pay_unless specifically: a mana ability must stay legal
    # mid-resolution of an ENTIRELY UNRELATED pending kind too (discard), and
    # activating it must not disturb that unrelated resolution at all.
    mid_decklist = [("Mountain", 4), ("Lightning Bolt", 2)]
    mid_actions = build_action_table(mid_decklist, game.EFFECT_REGISTRY)
    mid_tap_idx = _action_index(mid_actions, "Tap Mountain")
    mid_state = GameState(on_the_play=True)
    mid_mtn = Permanent(game.CARD_DEFS["Mountain"])
    mid_state.battlefield = [mid_mtn]
    mid_state.hand = [game.CARD_DEFS["Lightning Bolt"]]
    # The discard here is an unrelated pending, NOT a payment, so this float is
    # speculative and needs the main phase (_mana_timing_legal). The property
    # being pinned -- a mana ability neither needs nor disturbs an unrelated
    # open resolution -- is unchanged.
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
    """A mana subdecision claims EXCLUSIVE priority -- its owner may take no
    other action until it completes, which is how the engine models a mana
    ability being atomic. state.mana_subdecision is a single global slot, so
    before it carried an owner that exclusivity landed on whichever seat was
    asked NEXT rather than the one that opened it.

    Live consequence, 2026-08-16: the DEFENDER opened a Saruli Caretaker
    subdecision and never completed it; the ATTACKER was then asked to assign
    combat damage, rl.action_bridge built the pointer mask for the defender's
    tap-target choice instead of the attacker's blockers, nothing matched, and
    the all-False mask crashed an 11-deck training run twice.

    Pins both halves: the non-owner is unaffected, and the owner still gets
    exclusive priority (i.e. is forced to finish)."""
    saruli_decklist = [("Saruli Caretaker", 2), ("Slippery Bogle", 2)]
    actions = build_action_table(saruli_decklist, game.EFFECT_REGISTRY)

    saruli = Permanent(game.CARD_DEFS["Saruli Caretaker"])
    saruli.summoning_sick = False
    bogle = Permanent(game.CARD_DEFS["Slippery Bogle"])
    state = GameState(on_the_play=True, players=[PlayerState(True), PlayerState(False)])
    # Speculative floating is the active player's own main phase only
    # (drl_env._actions_mana._mana_timing_legal); a bare GameState has no
    # phase at all, which would fail that gate before reaching what this
    # test is actually about.
    state.phase = game.turn.Phase.MAIN1
    state.players[1].battlefield = [saruli, bogle]

    # Seat 1 opens the subdecision while it is the active seat. It must be seat
    # 1's own turn too: speculative floating is the ACTIVE PLAYER's main phase,
    # which is turn ownership, not just the phase (_mana_timing_legal).
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
    from rl.action_bridge import any_pointer_legal
    state.active_idx = 0
    assert any_pointer_legal(state) is False, (
        "with no pending of its own, seat 0 has no pointer decision -- and must NOT "
        "inherit the choose_target answer from seat 1's subdecision"
    )
