"""Migrated from game/effects/state_based.py's __main__ ponytail self-check.

Aura-orphaning and token-death are the two scenarios specific to THIS module
(check_state_based_actions/_destroy_creature), as opposed to the combat+SBA
handoff exercised together with combat.py in
tests/game/effects/test_integration_check.py."""
from game import registry
from game.cards import CardDef, CardType, EffectId
from game.effects.state_based import check_state_based_actions, cleanup_step, destroy_permanent, sacrifice_to_graveyard
from game.effects.tokens import WARRIOR_TOKEN_CARD_DEF
from game.state import GameState, Permanent, PlayerState


def test_aura_orphaning_and_cleanup():
    # Aura-orphaning: Rancor returns to its controller's hand
    # (returns_to_hand_when_orphaned, green_cards.py); every other Aura
    # (Ancestral Mask here) to the graveyard, real Magic's default.
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
        # 400.7: the graveyard card is a FRESH instance, not the battlefield
        # Permanent re-added (an identity check the by-name assert above can't
        # make -- the exact minting-discipline bug this guards against).
        assert state.players[0].graveyard[0] is not attacker_with_rancor
        assert rancor_permanent not in state.players[0].battlefield
        assert [c.name for c in state.players[0].hand] == ["Rancor"]  # returned to hand, not the graveyard
        assert state.players[0].hand[0] is not rancor_permanent  # fresh instance in hand, not the Permanent
        assert rancor_def not in state.players[0].graveyard

        assert blocker_with_mask not in state.players[1].battlefield
        assert mask_permanent not in state.players[1].battlefield
        assert sorted(c.name for c in state.players[1].graveyard) == ["Ancestral Mask", "Masked Blocker"]  # ordinary Aura -- graveyard, not hand
        # both fresh instances, not the battlefield Permanents re-added (400.7)
        assert all(c is not blocker_with_mask and c is not mask_permanent for c in state.players[1].graveyard)
        assert state.players[1].hand == []

        # cleanup_step clears damage_marked for EVERY permanent, both
        # players -- not just the active player's own side.
        cleanup_step(state)
        assert state.pending_resolution is None
    finally:
        registry.CARD_DEFS.clear()
        registry.CARD_DEFS.update(_card_defs_backup)


def test_token_creature_death_ceases_to_exist():
    # Token creature death: ceases to exist entirely -- same real-Magic
    # rule every existing token-removal path already follows, NOT the
    # graveyard-goes-there-normally case above.
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
    # cleanup clears temp modifiers + deathtouch marker.
    state = GameState(on_the_play=True, players=[PlayerState(True), PlayerState(False)])
    surv = Permanent(CardDef("Survivor", CardType.CREATURE, None, EffectId.FILLER, power=2, toughness=5))
    surv.temp_power, surv.temp_toughness, surv.temp_keywords = -3, -1, {"deathtouch"}
    surv.flags["deathtouched"] = True
    state.players[0].battlefield = [surv]
    cleanup_step(state)
    assert surv.temp_power == 0 and surv.temp_toughness == 0 and surv.temp_keywords == set()
    assert "deathtouched" not in surv.flags


def test_sacrifice_to_graveyard_fires_dies_and_on_sacrifice_triggers():
    # sacrifice_to_graveyard: fires a dies (ltb) trigger AND on_sacrifice
    # triggers on the sacrificer's other battlefield permanents.
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
        assert sac_seen == ["Victim"]  # Watcher's on_sacrifice saw the sacrifice ("another permanent")
    finally:
        registry.EFFECT_REGISTRY[EffectId.FILLER] = _filler_backup3
        registry.CARD_DEFS.pop("Watcher", None)
        registry.CARD_DEFS.pop("Victim", None)
