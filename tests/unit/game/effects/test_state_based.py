"""Aura-orphaning and token-death: scenarios specific to
check_state_based_actions/_destroy_creature. The combat+SBA handoff is
covered in test_integration_check.py."""
from game import registry, resolution
from game.cards import CardDef, CardType, EffectId
from game.effects.casting import enters_battlefield
from game.effects.stack import resolve_top_of_stack
from game.effects.state_based import check_state_based_actions, cleanup_step, destroy_permanent, sacrifice_to_graveyard
from game.effects.tokens import WARRIOR_TOKEN_CARD_DEF
from game.effects.triggers import promote_triggers_to_stack
from game.state import GameState, Permanent, PlayerState


def test_aura_orphaning_and_cleanup():
    # Rancor returns to its controller's hand (returns_to_hand_when_orphaned);
    # every other Aura (Ancestral Mask here) goes to the graveyard by default.
    rancor_def = CardDef("Rancor", CardType.ENCHANTMENT, {"G": 1}, EffectId.RANCOR)
    ancestral_mask_def = CardDef("Ancestral Mask", CardType.ENCHANTMENT, {"generic": 2, "G": 1}, EffectId.ANCESTRAL_MASK)
    _card_defs_backup = dict(registry.CARD_DEFS)
    try:
        state = GameState(on_the_play=True, players=[PlayerState(True), PlayerState(False)])
        attacker_with_rancor = Permanent(CardDef("Rancor'd Attacker", CardType.CREATURE, None, EffectId.FILLER, power=1, toughness=1))
        blocker_with_mask = Permanent(CardDef("Masked Blocker", CardType.CREATURE, None, EffectId.FILLER, power=1, toughness=1))
        registry.CARD_DEFS["Rancor'd Attacker"] = attacker_with_rancor.card_def
        registry.CARD_DEFS["Masked Blocker"] = blocker_with_mask.card_def
        rancor_permanent = Permanent(rancor_def)
        rancor_permanent.flags["enchanting"] = attacker_with_rancor
        mask_permanent = Permanent(ancestral_mask_def)
        mask_permanent.flags["enchanting"] = blocker_with_mask
        state.players[0].battlefield = [attacker_with_rancor, rancor_permanent]
        state.players[1].battlefield = [blocker_with_mask, mask_permanent]

        attacker_with_rancor.damage_marked = 1  # lethal (toughness 1)
        blocker_with_mask.damage_marked = 1  # lethal (toughness 1)
        check_state_based_actions(state)

        assert attacker_with_rancor not in state.players[0].battlefield
        assert [c.name for c in state.players[0].graveyard] == ["Rancor'd Attacker"]
        # 400.7: the graveyard card is a fresh instance, not the battlefield Permanent re-added
        assert state.players[0].graveyard[0] is not attacker_with_rancor
        assert rancor_permanent not in state.players[0].battlefield
        assert [c.name for c in state.players[0].hand] == ["Rancor"]  # returned to hand, not the graveyard
        assert state.players[0].hand[0] is not rancor_permanent  # fresh instance in hand, not the Permanent
        assert rancor_def not in state.players[0].graveyard

        assert blocker_with_mask not in state.players[1].battlefield
        assert mask_permanent not in state.players[1].battlefield
        assert sorted(c.name for c in state.players[1].graveyard) == ["Ancestral Mask", "Masked Blocker"]  # ordinary Aura -- graveyard, not hand
        assert all(c is not blocker_with_mask and c is not mask_permanent for c in state.players[1].graveyard)  # fresh instances (400.7)
        assert state.players[1].hand == []

        # cleanup_step clears damage_marked for every permanent, both players
        cleanup_step(state)
        assert state.pending_resolution is None
    finally:
        registry.CARD_DEFS.clear()
        registry.CARD_DEFS.update(_card_defs_backup)


def test_stats_changed_logged_on_counter_change_only_when_it_actually_changes():
    # a +1/+1 counter logs one stats_changed event; a later call with no
    # stat change must not re-log
    state = GameState(on_the_play=True, players=[PlayerState(True), PlayerState(False)], event_log=[])
    bear = Permanent(CardDef("Bear", CardType.CREATURE, None, EffectId.FILLER, power=2, toughness=2))
    bear.slot = 1
    state.players[0].battlefield = [bear]

    check_state_based_actions(state)  # first pass logs the printed base once (no prior cached value)
    changed = [e for e in state.event_log if e["kind"] == "stats_changed"]
    assert len(changed) == 1
    assert changed[0]["power"] == 2 and changed[0]["toughness"] == 2

    bear.counters["+1/+1"] = 1
    check_state_based_actions(state)
    changed = [e for e in state.event_log if e["kind"] == "stats_changed"]
    assert len(changed) == 2
    assert changed[-1]["permanent"] == ["Bear", 1]  # log_event's own _safe() turns tuples into lists
    assert changed[-1]["power"] == 3 and changed[-1]["toughness"] == 3

    check_state_based_actions(state)  # unchanged since -- no duplicate
    assert len([e for e in state.event_log if e["kind"] == "stats_changed"]) == 2


def test_cleanup_step_logs_stats_changed_when_a_temp_pump_expires():
    # a temp pump wearing off at cleanup must log its stats_changed event
    # immediately, not wait for the next check_state_based_actions call
    state = GameState(on_the_play=True, players=[PlayerState(True), PlayerState(False)], event_log=[])
    bear = Permanent(CardDef("Bear", CardType.CREATURE, None, EffectId.FILLER, power=2, toughness=2))
    bear.slot = 1
    bear.temp_power = 3
    bear.temp_toughness = 3
    state.players[0].battlefield = [bear]
    bear.flags["_logged_pt"] = (5, 5)  # as if already logged pumped to 5/5 mid-turn

    cleanup_step(state)

    changed = [e for e in state.event_log if e["kind"] == "stats_changed"]
    assert len(changed) == 1
    assert changed[0]["permanent"] == ["Bear", 1]
    assert changed[0]["power"] == 2 and changed[0]["toughness"] == 2  # back to printed base


def test_entering_via_enters_battlefield_does_not_log_a_redundant_stats_changed():
    # enters_battlefield seeds flags["_logged_pt"] from its own zone_move
    # event, so a plain ETB produces no redundant stats_changed
    state = GameState(on_the_play=True, players=[PlayerState(True), PlayerState(False)], event_log=[])
    bear_def = CardDef("Bear", CardType.CREATURE, None, EffectId.FILLER, power=2, toughness=2)
    enters_battlefield(state, bear_def, from_zone="hand")
    check_state_based_actions(state)
    assert [e for e in state.event_log if e["kind"] == "stats_changed"] == []


def test_token_creature_death_ceases_to_exist():
    state = GameState(on_the_play=True, players=[PlayerState(True), PlayerState(False)])
    warrior_token = Permanent(WARRIOR_TOKEN_CARD_DEF)
    state.players[0].battlefield = [warrior_token]
    warrior_token.damage_marked = 1  # lethal (toughness 1)
    check_state_based_actions(state)
    assert warrior_token not in state.players[0].battlefield
    assert state.players[0].graveyard == []  # ceased to exist -- never added to any zone


def test_destroy_permanent_indestructible_and_land():
    # destroy_permanent: indestructible can't be destroyed; a land goes to its
    # owner's graveyard; a creature routes through _destroy_creature.
    state = GameState(on_the_play=True, players=[PlayerState(True), PlayerState(False)])
    bridge = Permanent(CardDef("Drossforge Bridge", CardType.LAND, None, EffectId.DROSSFORGE_BRIDGE, artifact=True, indestructible=True))
    plain_land = Permanent(CardDef("Great Furnace", CardType.LAND, None, EffectId.GREAT_FURNACE, artifact=True))
    _card_defs_backup = dict(registry.CARD_DEFS)
    registry.CARD_DEFS["Great Furnace"] = plain_land.card_def  # so it goes to graveyard (not treated as a token)
    try:
        state.players[0].battlefield = [bridge, plain_land]
        assert destroy_permanent(state, bridge) is False  # indestructible -- survives
        assert bridge in state.players[0].battlefield
        assert destroy_permanent(state, plain_land) is True
        assert plain_land not in state.players[0].battlefield
        assert [c.name for c in state.players[0].graveyard] == ["Great Furnace"]
    finally:
        registry.CARD_DEFS.clear()
        registry.CARD_DEFS.update(_card_defs_backup)


def test_deathtouch_any_damage_is_lethal():
    # Deathtouch SBA: a creature with ANY marked damage from a deathtouch
    # source dies, even below its toughness.
    state = GameState(on_the_play=True, players=[PlayerState(True), PlayerState(False)])
    big = Permanent(CardDef("Big", CardType.CREATURE, None, EffectId.FILLER, power=5, toughness=5))
    state.players[1].battlefield = [big]
    big.damage_marked = 1
    big.flags["deathtouched"] = True  # as combat.py would set for a deathtouch hit
    check_state_based_actions(state)
    assert big not in state.players[1].battlefield  # 1 < 5 toughness, but deathtouched -> dead


def test_temp_toughness_reduction_lethal_via_sba():
    # temp P/T: -0/-3 to a 3-toughness creature is lethal via SBA (0 toughness).
    state = GameState(on_the_play=True, players=[PlayerState(True), PlayerState(False)])
    warped = Permanent(CardDef("Warped", CardType.CREATURE, None, EffectId.FILLER, power=2, toughness=3))
    state.players[0].battlefield = [warped]
    warped.temp_toughness = -3
    check_state_based_actions(state)
    assert warped not in state.players[0].battlefield


def test_cleanup_clears_temp_modifiers_and_deathtouch_marker():
    state = GameState(on_the_play=True, players=[PlayerState(True), PlayerState(False)])
    surv = Permanent(CardDef("Survivor", CardType.CREATURE, None, EffectId.FILLER, power=2, toughness=5))
    surv.temp_power, surv.temp_toughness, surv.temp_keywords = -3, -1, {"deathtouch"}
    surv.flags["deathtouched"] = True
    state.players[0].battlefield = [surv]
    cleanup_step(state)
    assert surv.temp_power == 0 and surv.temp_toughness == 0 and surv.temp_keywords == set()
    assert "deathtouched" not in surv.flags


def test_sacrifice_to_graveyard_fires_dies_and_on_sacrifice_triggers():
    # fires a dies (ltb) trigger and on_sacrifice on the sacrificer's other
    # permanents -- both queued onto the stack, not applied immediately
    _filler_backup3 = registry.EFFECT_REGISTRY[EffectId.FILLER]
    dies_fired = []
    sac_seen = []
    try:
        state = GameState(on_the_play=True)
        registry.EFFECT_REGISTRY[EffectId.FILLER] = {
            "ltb_trigger": lambda s, p: dies_fired.append(p.card_def.name),
            "on_sacrifice": lambda s, p, sacced: sac_seen.append(sacced.name),
        }
        watcher = Permanent(CardDef("Watcher", CardType.CREATURE, None, EffectId.FILLER, power=1, toughness=1))
        victim = Permanent(CardDef("Victim", CardType.ARTIFACT, {"generic": 1}, EffectId.FILLER))
        registry.CARD_DEFS["Watcher"] = watcher.card_def
        registry.CARD_DEFS["Victim"] = victim.card_def
        state.battlefield = [watcher, victim]
        sacrifice_to_graveyard(state, victim)
        gy_victim = next((c for c in state.graveyard if c.card_def is victim.card_def), None)
        assert gy_victim is not None  # real card -> graveyard
        assert gy_victim is not victim  # 400.7: a FRESH instance, not the battlefield Permanent re-added
        assert [e for e in state.trigger_queue if e["type"] == "ltb"]  # its dies-trigger queued
        assert [e for e in state.trigger_queue if e["type"] == "on_sacrifice"]  # Watcher's trigger queued too
        assert sac_seen == []  # not applied yet -- still just queued
        promote_triggers_to_stack(state)
        # same controller for both triggers -- 603.3b lets them choose placement order
        while state.pending_resolution is not None and state.pending_resolution["kind"] == "order_triggers":
            resolution.execute_order_triggers_option(state, resolution.order_triggers_options(state)[0])
        while state.stack:
            resolve_top_of_stack(state)
        assert dies_fired == ["Victim"]
        assert sac_seen == ["Victim"]  # Watcher's on_sacrifice saw the sacrifice ("another permanent")
    finally:
        registry.EFFECT_REGISTRY[EffectId.FILLER] = _filler_backup3
        registry.CARD_DEFS.pop("Watcher", None)
        registry.CARD_DEFS.pop("Victim", None)
