"""Tests for game.catalog.green_cards. See the module under test for the
card-implementation rationale (real-rules citations, etc.) each test below
guards."""

import contextlib
import io

import drl_env
from game import registry
from game.cards import CardDef, CardType, EffectId
from game.catalog.green_cards import (
    SAGU_WILDLING_CREATURE_CARD_DEF,
    _ram_through_extra_legal,
    activate_bramble_wurm_gy,
    ancient_stirrings_options,
    cast_abundant_growth,
    cast_ancestral_mask,
    cast_ancient_stirrings,
    cast_crop_rotation,
    cast_land_grant,
    cast_land_grant_alt,
    cast_lead_the_stampede,
    cast_malevolent_rumble,
    cast_nyxborn_hydra_bestow,
    cast_nyxborn_hydra_creature,
    cast_pulse_of_murasa,
    cast_ram_through,
    cast_rancor,
    cast_roost_seek,
    cast_sagu_wildling_creature,
    cast_winding_way_land,
    execute_ancient_stirrings_option,
    execute_malevolent_rumble_option,
    execute_select_to_hand_option,
    forestcycle_generous_ent,
    gatecreeper_vine_etb,
    land_grant_alt_cost_legal,
    malevolent_rumble_options,
    quirion_ranger_untap_legal,
    quirion_ranger_untap_resolve,
    select_to_hand_options,
    timberwatch_elf_activate,
    wellwisher_activate,
)
from game.effects.casting import cast_permanent_from_hand, enters_battlefield, play_land_from_hand
from game.effects.shared import any_creature_on_battlefield
from game.effects.stack import push_to_stack, resolve_top_of_stack
from game.effects.state_based import check_state_based_actions
from game.effects.stats import can_be_targeted, has_keyword, permanent_power, permanent_toughness
from game.effects.tokens import (
    FOOD_TOKEN_CARD_DEF, SKELETON_TOKEN_CARD_DEF, activate_eldrazi_spawn_sac, activate_food_sac, create_token,
)
from game.effects.triggers import promote_triggers_to_stack
from game.mana import (
    COLORS, activate_mana_source, execute_pool_spend, mana_ability_options, mana_output, pool_spend_options,
    tap_summoning_locked,
)
from game.resolution import (
    choose_any_target_creature_options, choose_graveyard_card_options, execute_choose_any_target_creature,
    execute_choose_graveyard_card_decline, execute_choose_graveyard_card_option,
    execute_choose_opponent_permanent_option, execute_choose_permanent_option, execute_search_fetch_option,
    search_fetch_options,
)
from game.state import CardInstance, GameState, Permanent, PlayerState
from game.turn import Phase, untap_step


def _malevolent_rumble_cast():
    state = GameState(on_the_play=True)
    rumble = CardDef("Malevolent Rumble", CardType.SORCERY, {"generic": 1, "G": 1}, EffectId.MALEVOLENT_RUMBLE)
    state.hand = [rumble]
    state.library = [
        CardDef("A Creature", CardType.CREATURE, {"G": 1}, None),
        CardDef("An Instant", CardType.INSTANT, {"G": 1}, None),
        CardDef("A Land", CardType.LAND, None, None),
        CardDef("Filler 4", CardType.SORCERY, {}, None),
        CardDef("Filler 5", CardType.SORCERY, {}, None),  # 5th card -- never revealed, stays in library
    ]
    cast_malevolent_rumble(state, rumble)
    return state, rumble


def test_malevolent_rumble_take_creature():
    """Malevolent Rumble: reveal top 4, may take one permanent card to hand
    (rest to graveyard, NOT the library bottom -- unlike Ancient Stirrings'
    own take-one-or-decline shape, verified via Scryfall)."""
    state, rumble = _malevolent_rumble_cast()
    assert [p.card_def.name for p in state.battlefield] == ["Eldrazi Spawn"]
    assert state.pending_resolution["kind"] == "malevolent_rumble"
    # Instants/sorceries aren't "permanent cards" -- ineligible; only the
    # creature and the land are offered, plus the ever-present decline.
    assert malevolent_rumble_options(state) == ["A Creature", "A Land", "decline"]
    execute_malevolent_rumble_option(state, "A Creature")
    assert state.pending_resolution is None
    assert [c.name for c in state.hand] == ["A Creature"]
    # Everything revealed but not taken -- including the ineligible ones --
    # goes to the graveyard, alongside Malevolent Rumble itself.
    assert sorted(c.name for c in state.graveyard) == ["A Land", "An Instant", "Filler 4", "Malevolent Rumble"]
    assert [c.name for c in state.library] == ["Filler 5"]  # never revealed, untouched


def test_eldrazi_spawn_token_sac_ability():
    """The 0/1 Eldrazi Spawn token created by Malevolent Rumble: its own
    "Sacrifice: Add {C}" ability floats mana with no {T} at all."""
    state, _rumble = _malevolent_rumble_cast()
    spawn = state.battlefield[0]
    assert state.mana_pool == {}
    activate_eldrazi_spawn_sac(state, spawn)
    assert state.battlefield == []  # sacrificed, not graveyarded -- a token ceases to exist
    assert state.mana_pool == {"C": 1}


def test_malevolent_rumble_decline():
    """Declining leaves everything revealed in the graveyard, nothing kept."""
    state = GameState(on_the_play=True)
    rumble2 = CardDef("Malevolent Rumble", CardType.SORCERY, {"generic": 1, "G": 1}, EffectId.MALEVOLENT_RUMBLE)
    state.hand = [rumble2]
    state.library = [CardDef(f"Card {i}", CardType.CREATURE, {"G": 1}, None) for i in range(4)]
    cast_malevolent_rumble(state, rumble2)
    execute_malevolent_rumble_option(state, "decline")
    assert state.pending_resolution is None
    assert state.hand == []
    assert sorted(c.name for c in state.graveyard) == ["Card 0", "Card 1", "Card 2", "Card 3", "Malevolent Rumble"]


def test_bramble_wurm_graveyard_ability():
    """Bramble Wurm's graveyard ability: exile from graveyard (removed,
    untracked), gain 5 life -- the one piece of real logic
    activate_bramble_wurm_gy adds beyond already-self-checked helpers
    (cast_permanent_from_hand, gain_life itself)."""
    state = GameState(on_the_play=True)
    # Graveyard holds a real CardInstance (as a live game does), and that
    # instance is what activate_bramble_wurm_gy now receives -- the exile
    # cost is an identity removal, not a by-name lookup.
    wurm = CardInstance(CardDef(
        "Bramble Wurm", CardType.CREATURE, {"generic": 6, "G": 1}, EffectId.BRAMBLE_WURM, power=7, toughness=6,
        gy_ability_cost={"generic": 2, "G": 1},
    ))
    state.graveyard = [wurm]
    activate_bramble_wurm_gy(state, wurm)
    assert state.graveyard == []  # exiled -- a cost, paid immediately on activation
    # The life gain is the ability's EFFECT -- now on the stack (faithful
    # timing), not applied the instant the cost was paid.
    assert len(state.stack) == 1 and state.life_total == 20
    resolve_top_of_stack(state)
    assert state.life_total == 25  # STARTING_LIFE (20) + 5, once the effect resolves


def test_sagu_wildling_omen_search_and_redraw():
    """Sagu Wildling's Omen: cast_roost_seek shuffles ITSELF into the
    library (not exile, not the graveyard) once its search resolves --
    real Adventure's own exile doesn't apply to Omen. Redrawing it later
    puts the same physical card back in hand; cast_sagu_wildling_creature
    then finds it there BY NAME (a different CardDef object, same display
    name), removes it, and puts the real creature on the battlefield with
    its own ETB gain-3-life."""
    state = GameState(on_the_play=True)
    roost_seek = CardDef("Sagu Wildling", CardType.SORCERY, {"G": 1}, EffectId.ROOST_SEEK)
    state.hand = [roost_seek]
    state.library = [CardDef("Forest", CardType.LAND, None, EffectId.FOREST, basic=True)]
    cast_roost_seek(state, roost_seek)
    assert state.hand == []
    assert state.exile == []  # no exile step at all -- unlike real Adventure
    assert state.pending_resolution["kind"] == "search_fetch"
    execute_search_fetch_option(state, "Forest")
    assert [c.name for c in state.hand] == ["Forest"]
    assert [c.name for c in state.library] == ["Sagu Wildling"]  # shuffled itself in, per "(Also shuffle this card.)"

    # Redraw it (same physical card, ordinary draw) -- the real creature
    # half is now just a second cast option for that same hand card.
    state.hand.append(state.library.pop(0))
    cast_sagu_wildling_creature(state, SAGU_WILDLING_CREATURE_CARD_DEF)
    assert [c.name for c in state.hand] == ["Forest"]  # the redrawn "Sagu Wildling" left hand, "Forest" untouched
    # The ETB gain-3 is queued now (faithful timing) -- resolve it off the
    # stack, exactly as game.turn's priority round would.
    assert state.life_total == 20
    promote_triggers_to_stack(state)
    resolve_top_of_stack(state)
    assert state.life_total == 23  # STARTING_LIFE (20) + 3, once the ETB resolves
    sagu_permanent = next(p for p in state.battlefield if p.card_def.name == "Sagu Wildling")
    assert sagu_permanent.card_def is SAGU_WILDLING_CREATURE_CARD_DEF


def test_sagu_wildling_omen_full_cast_path():
    """Full cast PATH (not just the resolve): _omen_cast_execute removes the
    physical hand card AT CAST (by name -- the creature side is a distinct
    CardDef), the creature-side def goes on the stack, and resolving it puts
    the creature on the battlefield. The card must NEVER re-enter hand, and
    cast_sagu_wildling_creature must be a no-op on the (already empty of it)
    hand. Mirrors _omen_cast_execute's own post-payment steps."""
    state = GameState(on_the_play=True)
    state.hand = [CardDef("Sagu Wildling", CardType.SORCERY, {"G": 1}, EffectId.ROOST_SEEK),
                  CardDef("Forest", CardType.LAND, None, EffectId.FOREST, basic=True)]
    hc = next(c for c in state.hand if c.name == SAGU_WILDLING_CREATURE_CARD_DEF.name)
    state.hand.remove(hc)  # _omen_cast_execute's cast-time by-name removal
    push_to_stack(state, SAGU_WILDLING_CREATURE_CARD_DEF, cast_sagu_wildling_creature)
    assert [c.name for c in state.hand] == ["Forest"] and len(state.stack) == 1  # Sagu left hand at cast; Forest untouched
    resolve_top_of_stack(state)
    assert [c.name for c in state.hand] == ["Forest"]  # never re-entered hand during/after resolution
    assert any(p.card_def is SAGU_WILDLING_CREATURE_CARD_DEF for p in state.battlefield)  # creature resolved onto the battlefield


def _ram_through_card():
    return CardDef("Ram Through", CardType.INSTANT, {"generic": 1, "G": 1}, EffectId.RAM_THROUGH)


def test_ram_through_plain_kill():
    """Ram Through ({1}{G} Instant): one-sided fight -- target creature you
    control deals its power to target creature you don't control, trample
    overflow to that creature's controller. Two targets locked at cast,
    resolved via the shared combat damage-marking + state-based death.

    (a) plain: my 3/3 kills the opponent's 2/2, taking nothing back."""
    ram = _ram_through_card()
    state = GameState(on_the_play=True, players=[PlayerState(True), PlayerState(False)])
    mine = Permanent(CardDef("My Beater", CardType.CREATURE, {"G": 1}, EffectId.FILLER, power=3, toughness=3))
    theirs = Permanent(CardDef("Their Bear", CardType.CREATURE, {"G": 1}, EffectId.FILLER, power=2, toughness=2))
    state.players[0].battlefield = [mine]
    state.players[1].battlefield = [theirs]
    state.hand = [ram]
    assert _ram_through_extra_legal(state)
    cast_ram_through(state, ram)
    assert state.pending_resolution["kind"] == "choose_any_target"
    execute_choose_any_target_creature(state, 0, "My Beater", 1)  # source: creature I control
    assert state.pending_resolution["kind"] == "choose_any_target"
    execute_choose_any_target_creature(state, 1, "Their Bear", 1)  # target: creature I don't control
    assert state.hand == [] and len(state.stack) == 1  # left hand at cast, on the stack
    resolve_top_of_stack(state)
    assert any(c.name == ram.name for c in state.graveyard)
    assert theirs not in state.players[1].battlefield  # 3 >= 2 toughness -> dead
    assert mine in state.players[0].battlefield  # one-sided: my creature takes nothing


def test_ram_through_trample_overflow():
    """(b) trample overflow: Rancor grants trample; excess (power - lethal)
    hits the opponent PLAYER."""
    ram = _ram_through_card()
    state = GameState(on_the_play=True, players=[PlayerState(True), PlayerState(False)])
    trampler = Permanent(CardDef("Trampler", CardType.CREATURE, {"G": 1}, EffectId.FILLER, power=5, toughness=5))
    rancor = Permanent(CardDef("Rancor", CardType.ENCHANTMENT, {"G": 1}, EffectId.RANCOR))
    rancor.flags["enchanting"] = trampler
    victim = Permanent(CardDef("Victim", CardType.CREATURE, {"G": 1}, EffectId.FILLER, power=1, toughness=2))
    state.players[0].battlefield = [trampler, rancor]
    state.players[1].battlefield = [victim]
    state.hand = [ram]
    cast_ram_through(state, ram)
    execute_choose_any_target_creature(state, 0, "Trampler", 1)
    execute_choose_any_target_creature(state, 1, "Victim", 1)
    resolve_top_of_stack(state)
    # Effective power 7 (5 + Rancor's +2): 2 lethal to Victim, 5 tramples through.
    assert victim not in state.players[1].battlefield
    assert state.players[1].life_total == 15  # 20 - 5 trampled


def test_ram_through_fizzle_target_removed():
    """(c) fizzle: the target creature leaves before resolution (608.2c)."""
    ram = _ram_through_card()
    state = GameState(on_the_play=True, players=[PlayerState(True), PlayerState(False)])
    mine2 = Permanent(CardDef("Mine2", CardType.CREATURE, {"G": 1}, EffectId.FILLER, power=3, toughness=3))
    theirs2 = Permanent(CardDef("Theirs2", CardType.CREATURE, {"G": 1}, EffectId.FILLER, power=2, toughness=2))
    state.players[0].battlefield = [mine2]
    state.players[1].battlefield = [theirs2]
    state.hand = [ram]
    cast_ram_through(state, ram)
    execute_choose_any_target_creature(state, 0, "Mine2", 1)
    execute_choose_any_target_creature(state, 1, "Theirs2", 1)
    state.players[1].battlefield.remove(theirs2)  # removed in response, before resolution
    log = io.StringIO()
    with contextlib.redirect_stdout(log):
        resolve_top_of_stack(state)
    assert "fizzle" in log.getvalue().lower()
    assert any(c.name == ram.name for c in state.graveyard) and mine2 in state.players[0].battlefield  # my creature unaffected


def test_ram_through_not_castable_solo():
    """(d) never castable in 1-player (no "creature you don't control")."""
    solo = GameState(on_the_play=True)
    solo.battlefield = [Permanent(CardDef("Lonely", CardType.CREATURE, {"G": 1}, EffectId.FILLER, power=3, toughness=3))]
    assert not _ram_through_extra_legal(solo)


def _masked_vandal_card():
    return CardDef("Masked Vandal", CardType.CREATURE, {"generic": 1, "G": 1}, EffectId.MASKED_VANDAL, power=1, toughness=3)


def _setup_vandal():
    """Common setup for the Masked Vandal ETB scenarios: a creature in the
    caster's graveyard, an artifact on the opponent's battlefield, cast the
    Vandal, and promote its ETB to the stack -- target-at-promotion opens
    the opponent-permanent TARGET choice FIRST."""
    state = GameState(on_the_play=True, players=[PlayerState(True), PlayerState(False)])
    state.players[0].graveyard = [state.new_instance(CardDef("Dead Guy", CardType.CREATURE, {"G": 1}, EffectId.FILLER, power=2, toughness=2))]
    art = Permanent(CardDef("Their Relic", CardType.ARTIFACT, {"generic": 1}, EffectId.FILLER))
    state.players[1].battlefield = [art]
    enters_battlefield(state, _masked_vandal_card(), from_zone="hand")
    assert [e["type"] for e in state.trigger_queue] == ["etb"]
    promote_triggers_to_stack(state)  # target-at-promotion: opens the opponent-permanent TARGET choice FIRST
    assert state.pending_resolution["kind"] == "choose_opponent_permanent"
    return state, art


def test_masked_vandal_etb_take():
    """(a) take it: lock the target, then (at resolution) exile a creature
    from GY -> exile the target."""
    state, opp_artifact = _setup_vandal()
    execute_choose_opponent_permanent_option(state, "Their Relic", 1)  # target locked at promotion
    assert state.pending_resolution is None and len(state.stack) == 1  # the effect is on the stack
    resolve_top_of_stack(state)  # opens the "you may exile a creature" choice
    assert state.pending_resolution["kind"] == "choose_graveyard_card"
    vandal_opts = choose_graveyard_card_options(state)
    assert [c.name for c in vandal_opts] == ["Dead Guy"]  # creature cards only
    execute_choose_graveyard_card_option(state, vandal_opts[0])
    assert state.players[0].graveyard == []  # creature exiled from GY (the "if you do" cost)
    assert opp_artifact not in state.players[1].battlefield  # the targeted artifact is exiled
    assert state.pending_resolution is None


def test_masked_vandal_etb_decline_graveyard_exile():
    """(b) decline the "you may": the target was locked, but declining the
    graveyard exile means nothing happens."""
    state, opp_artifact = _setup_vandal()
    execute_choose_opponent_permanent_option(state, "Their Relic", 1)
    resolve_top_of_stack(state)
    assert state.pending_resolution["kind"] == "choose_graveyard_card"
    execute_choose_graveyard_card_decline(state)
    assert len(state.players[0].graveyard) == 1 and opp_artifact in state.players[1].battlefield
    assert state.pending_resolution is None


def test_masked_vandal_etb_fizzle_target_removed():
    """(c) fizzle: target locked at promotion, then leaves before resolution
    -> the creature is still exiled (the cost is paid), but the target exile
    fizzles (608.2b)."""
    state, opp_artifact = _setup_vandal()
    execute_choose_opponent_permanent_option(state, "Their Relic", 1)
    state.players[1].battlefield.remove(opp_artifact)  # target gone before this resolves
    resolve_top_of_stack(state)
    execute_choose_graveyard_card_option(state, choose_graveyard_card_options(state)[0])
    assert state.players[0].graveyard == []  # the creature was still exiled -- the cost was paid
    assert opp_artifact not in state.players[1].battlefield  # nothing re-added; the exile fizzled gracefully
    assert state.pending_resolution is None


def test_masked_vandal_etb_no_legal_target():
    """(d) no legal target (opponent controls no artifact/enchantment): the
    whole ETB is a no-op -- it isn't even put on the stack (603.3c), nothing
    offered."""
    state = GameState(on_the_play=True, players=[PlayerState(True), PlayerState(False)])
    state.players[0].graveyard = [state.new_instance(CardDef("Dead Guy", CardType.CREATURE, {"G": 1}, EffectId.FILLER, power=2, toughness=2))]
    enters_battlefield(state, _masked_vandal_card(), from_zone="hand")
    promote_triggers_to_stack(state)  # no legal target -> the hook returns early, nothing pushed
    assert state.pending_resolution is None and state.stack == [] and len(state.players[0].graveyard) == 1


def _cast_generous_ent_and_resolve_etb():
    state = GameState(on_the_play=True)
    ent = CardDef(
        "Generous Ent", CardType.CREATURE, {"generic": 5, "G": 1}, EffectId.GENEROUS_ENT,
        forestcycling_cost={"generic": 1}, power=5, toughness=5,
    )
    state.hand = [ent]
    cast_permanent_from_hand(state, ent)
    return state


def test_generous_ent_etb_creates_food_token():
    """Generous Ent: hard-cast (Reach + ETB "create a Food token"). (Forestcycling,
    the only mode before, is unchanged.)"""
    state = _cast_generous_ent_and_resolve_etb()
    ent_perm = next(p for p in state.battlefield if p.card_def.name == "Generous Ent")
    assert has_keyword(state, ent_perm, "reach")
    # ETB queued -> resolve -> a Food token is created.
    assert not any(p.card_def.name == "Food" for p in state.battlefield)
    promote_triggers_to_stack(state)
    resolve_top_of_stack(state)
    food = next(p for p in state.battlefield if p.card_def.name == "Food")
    assert food.card_def is FOOD_TOKEN_CARD_DEF


def test_food_token_sac_ability():
    """The Food token's own "{2},{T},Sacrifice: gain 3 life" ability: the
    gain is the effect, on the stack (the {2} + tap are drl_env's concern)."""
    state = _cast_generous_ent_and_resolve_etb()
    promote_triggers_to_stack(state)
    resolve_top_of_stack(state)
    food = next(p for p in state.battlefield if p.card_def.name == "Food")
    activate_food_sac(state, food)
    assert not any(p.card_def.name == "Food" for p in state.battlefield)  # ceased to exist
    assert state.graveyard == [] and state.life_total == 20  # no GY trip; gain still on the stack
    resolve_top_of_stack(state)
    assert state.life_total == 23  # +3 once the effect resolves


def _wall_of_roots():
    return Permanent(CardDef(
        "Wall of Roots", CardType.CREATURE, {"generic": 1, "G": 1}, EffectId.WALL_OF_ROOTS,
        defender=True, power=0, toughness=5,
    ))


def test_wall_of_roots_no_tap_ability_and_counter_death():
    """Wall of Roots: no {T} in the real ability's own cost at all (unlike
    every other mana dork here), and a real -0/-1 counter each use (not a
    private activation count). Float-first: its mana ability IS
    activate_mana_source (immediate float, no tap-during-payment), so both
    are exercised through that path."""
    state = GameState(on_the_play=True)
    wall = _wall_of_roots()
    state.battlefield = [wall]

    assert ("Wall of Roots", None) in mana_ability_options(state)
    activate_mana_source(state, wall)  # float {G} -- no {T} in this ability's cost, so it stays untapped
    assert wall.tapped is False  # no {T} in this ability's own cost
    assert wall.counters["-0/-1"] == 1
    assert state.mana_pool.get("G", 0) == 1  # produced mana floated into the pool

    # Once each turn -- not offered again this same turn, even though it's
    # still untapped (mana_extra_available's own used_this_turn gate; there
    # is no tapped state here to gate on at all).
    assert ("Wall of Roots", None) not in mana_ability_options(state)

    # 4 more activations, one per (simulated) turn, reach the 5th counter --
    # lethal against this wall's own 5 toughness, via a genuine counter and
    # the ordinary state-based-action check.
    for _ in range(4):
        untap_step(state)  # resets used_this_turn, re-enabling the ability
        activate_mana_source(state, wall)

    assert wall.counters["-0/-1"] == 5
    assert permanent_toughness(state, wall) == 0
    assert wall in state.battlefield  # state-based actions haven't been asked to check yet
    check_state_based_actions(state)
    assert wall not in state.battlefield
    assert any(c.name == wall.card_def.name for c in state.graveyard)


def test_nyxborn_hydra_creature_mode():
    """Nyxborn Hydra, creature mode: 0/1 base (a design choice, not Scryfall
    data) plus X real "+1/+1" counters."""
    state = GameState(on_the_play=True)
    hydra_card = CardDef("Nyxborn Hydra", CardType.CREATURE, {"G": 1}, EffectId.NYXBORN_HYDRA, power=0, toughness=1)
    state.hand = [hydra_card]
    cast_nyxborn_hydra_creature(3)(state, hydra_card)
    assert state.hand == []
    hydra_permanent = next(p for p in state.battlefield if p.card_def.name == "Nyxborn Hydra")
    assert hydra_permanent.counters["+1/+1"] == 3
    assert permanent_power(state, hydra_permanent) == 3
    assert permanent_toughness(state, hydra_permanent) == 4


def test_nyxborn_hydra_bestow_attach_and_falloff():
    """Nyxborn Hydra, Bestow: GGX, enchant target creature you control --
    drives the real cast_aura flow (choose target -> stack -> resolve), same
    pattern casting.py's own Rancor self-check already uses."""
    state = GameState(on_the_play=True)
    target_creature = Permanent(CardDef("Target Creature", CardType.CREATURE, None, EffectId.FILLER, power=2, toughness=2))
    hydra_card2 = CardDef("Nyxborn Hydra", CardType.CREATURE, {"G": 1}, EffectId.NYXBORN_HYDRA, power=0, toughness=1)
    state.battlefield = [target_creature]
    state.hand = [hydra_card2]

    cast_nyxborn_hydra_bestow(2)(state, hydra_card2)
    assert state.pending_resolution["kind"] == "choose_any_target"  # Bestow = cast_aura, now any-target
    execute_choose_any_target_creature(state, 0, "Target Creature", 1)  # side 0 (this 1-player fixture)
    assert state.hand == []  # left hand at cast -- sitting on the stack, unresolved
    assert len(state.stack) == 1
    resolve_top_of_stack(state)
    assert state.hand == []

    bestowed = next(p for p in state.battlefield if p.card_def.name == "Nyxborn Hydra")
    assert bestowed.flags["enchanting"] is target_creature
    assert bestowed.counters["+1/+1"] == 2
    assert bestowed.card_type == CardType.ENCHANTMENT  # NOT a creature while attached
    assert bestowed.type_override == CardType.ENCHANTMENT
    # The enchanted creature gains +1/+1 PER counter on the Hydra itself --
    # a dynamic pt_bonus/toughness_bonus (2 counters here), not a fixed
    # constant like Rancor's own +2.
    assert permanent_power(state, target_creature) == 4  # 2 base + 2
    assert permanent_toughness(state, target_creature) == 4
    # Correctly excluded from every "another creature"/"target creature"
    # check while attached -- Dread Return's own sacrifice-3-creatures cost
    # (real interaction: both cards are in spy_combo.txt together).
    assert any_creature_on_battlefield(state) is True  # target_creature itself still qualifies
    assert sum(1 for p in state.battlefield if p.card_type == CardType.CREATURE) == 1  # bestowed Hydra doesn't count

    # Fall-off: the enchanted creature dies -- real Bestow rule, the Hydra
    # stays on the battlefield and becomes a creature again, keeping its own
    # counters (state_based._destroy_creature's own
    # "becomes_creature_when_orphaned" branch).
    target_creature.damage_marked = 4  # lethal against its OWN buffed toughness (2 base + 2 from the Bestow attached to it)
    check_state_based_actions(state)
    assert target_creature not in state.battlefield
    assert bestowed in state.battlefield  # NOT graveyarded/returned to hand like an ordinary orphaned Aura
    assert bestowed.type_override is None
    assert bestowed.card_type == CardType.CREATURE  # card_def.card_type itself, now that the override is cleared
    assert bestowed.flags.get("enchanting") is None
    assert bestowed.counters["+1/+1"] == 2  # unchanged -- "maintain its own +1/+1s"
    assert permanent_power(state, bestowed) == 2 and permanent_toughness(state, bestowed) == 3  # 0/1 base + 2, its own pt_bonus no longer applies to anything


def test_nyxborn_hydra_via_action_table_mode_then_x():
    """Nyxborn Hydra via the REAL action table (not calling resolve closures
    directly, unlike the direct-call tests above): "Cast Nyxborn Hydra" ->
    choose_cast_mode ("Mode 1" = creature, registry-first) -> choose_cast_x
    ("X=3") -> pay {G}{3} -> resolves as the creature mode with 3 counters.
    Confirms the decomposed mode-then-X sequencing (drl_env._x_modal_execute)
    actually reaches the exact resolve closures the direct-call tests above
    already verified in isolation."""
    hydra_dl = [("Nyxborn Hydra", 4), ("Forest", 8)]
    hydra_byname = {a[0]: (a[1], a[2]) for a in drl_env.build_action_table(hydra_dl, registry.EFFECT_REGISTRY)}
    hx_state = GameState(on_the_play=True)
    hx_state.phase = Phase.MAIN1
    hx_state.turn_player_idx = 0
    hx_state.active_idx = 0
    hx_state.hand = [registry.CARD_DEFS["Nyxborn Hydra"]]
    hx_state.battlefield = [Permanent(registry.CARD_DEFS["Forest"]) for _ in range(4)]  # {G} + X=3 generic = 4 mana
    for f in hx_state.battlefield:
        activate_mana_source(hx_state, f)
    assert hx_state.mana_pool.get("G", 0) == 4
    cast_legal, cast_execute = hydra_byname["Cast Nyxborn Hydra"]
    assert cast_legal(hx_state)
    cast_execute(hx_state)
    assert hx_state.pending_resolution["kind"] == "choose_cast_mode"
    mode1_legal, mode1_execute = hydra_byname["Mode 1"]
    assert mode1_legal(hx_state)
    mode1_execute(hx_state)
    assert hx_state.pending_resolution["kind"] == "choose_cast_x"
    x3_legal, x3_execute = hydra_byname["X=3"]
    assert x3_legal(hx_state)
    x3_execute(hx_state)
    assert hx_state.pending_resolution["kind"] == "pay_cost"
    guard = 0
    while hx_state.pending_resolution is not None:
        guard += 1
        assert guard < 30
        execute_pool_spend(hx_state, pool_spend_options(hx_state)[0])
    resolve_top_of_stack(hx_state)
    hx_permanent = next(p for p in hx_state.battlefield if p.card_def.name == "Nyxborn Hydra")
    assert hx_permanent.counters["+1/+1"] == 3


def test_winding_way_via_action_table():
    """Winding Way via the real action table: "Cast Winding Way" -> choose_
    cast_mode ("Mode 1" = creature, registry-first, no per-mode cost
    override) -> pay {1}{G} -> hits the stack normally (no precast_choice)
    -> resolving reveals top 4, matches creatures to hand."""
    ww_dl = [("Winding Way", 4), ("Forest", 8)]
    ww_byname = {a[0]: (a[1], a[2]) for a in drl_env.build_action_table(ww_dl, registry.EFFECT_REGISTRY)}
    ww_state = GameState(on_the_play=True)
    ww_state.phase = Phase.MAIN1
    ww_state.turn_player_idx = 0
    ww_state.active_idx = 0
    ww_state.hand = [registry.CARD_DEFS["Winding Way"]]
    ww_state.battlefield = [Permanent(registry.CARD_DEFS["Forest"]) for _ in range(2)]  # {1}{G} = 2 mana
    ww_state.library = [CardDef(f"Bear{i}", CardType.CREATURE, {"G": 1}, EffectId.FILLER, power=2, toughness=2) for i in range(4)]
    for f in ww_state.battlefield:
        activate_mana_source(ww_state, f)
    cast_legal, cast_execute = ww_byname["Cast Winding Way"]
    assert cast_legal(ww_state)
    cast_execute(ww_state)
    assert ww_state.pending_resolution["kind"] == "choose_cast_mode"
    mode1_legal, mode1_execute = ww_byname["Mode 1"]
    assert mode1_legal(ww_state)
    mode1_execute(ww_state)
    assert ww_state.pending_resolution["kind"] == "pay_cost"
    guard = 0
    while ww_state.pending_resolution is not None:
        guard += 1
        assert guard < 30
        execute_pool_spend(ww_state, pool_spend_options(ww_state)[0])
    assert len(ww_state.stack) == 1  # no precast_choice -- Winding Way sits on the stack like an ordinary sorcery
    resolve_top_of_stack(ww_state)
    assert any(c.name == "Winding Way" for c in ww_state.graveyard), "the resolved sorcery must land in its own graveyard"
    assert len(ww_state.hand) == 4 and all(c.card_type == CardType.CREATURE for c in ww_state.hand), (
        "creature mode must match all 4 revealed (all-creature-seeded) library cards to hand"
    )


def test_utopia_sprawl_via_action_table():
    """Utopia Sprawl via the real action table: exercises BOTH extra_legal
    (needs a Forest to enchant) and precast_choice (the aura's real target
    is locked in as part of casting, before the stack) within the decomposed
    modal path -- "Cast Utopia Sprawl" -> choose_cast_mode ("Mode 1" =
    green, registry-first) -> pay {G} -> resolve() runs directly as
    pay_cost's on_complete (precast_choice), which is cast_utopia_sprawl ->
    cast_aura -> opens ITS OWN choose_any_target for the Forest --
    pre-existing cast_aura machinery, untouched by this change; stopping
    once that resolve() hand-off is reached is enough to prove the new
    mode-choice/cost/precast_choice wiring is correct."""
    us_dl = [("Utopia Sprawl", 4), ("Forest", 8)]
    us_byname = {a[0]: (a[1], a[2]) for a in drl_env.build_action_table(us_dl, registry.EFFECT_REGISTRY)}
    us_state = GameState(on_the_play=True)
    us_state.phase = Phase.MAIN1
    us_state.turn_player_idx = 0
    us_state.active_idx = 0
    us_state.hand = [registry.CARD_DEFS["Utopia Sprawl"]]
    us_forest = Permanent(registry.CARD_DEFS["Forest"])
    us_state.battlefield = [us_forest]
    activate_mana_source(us_state, us_forest)
    cast_legal, cast_execute = us_byname["Cast Utopia Sprawl"]
    assert cast_legal(us_state)
    cast_execute(us_state)
    assert us_state.pending_resolution["kind"] == "choose_cast_mode"
    mode1_legal, mode1_execute = us_byname["Mode 1"]
    assert mode1_legal(us_state)
    mode1_execute(us_state)
    assert us_state.pending_resolution["kind"] == "pay_cost"
    guard = 0
    while us_state.pending_resolution is not None and us_state.pending_resolution["kind"] == "pay_cost":
        guard += 1
        assert guard < 30
        execute_pool_spend(us_state, pool_spend_options(us_state)[0])
    assert us_state.pending_resolution["kind"] == "choose_any_target", (
        "precast_choice must hand off to cast_aura's own target choice directly, not push to the stack first"
    )


def _drive_etb(state):
    promote_triggers_to_stack(state)
    while state.stack:
        resolve_top_of_stack(state)


def test_gingerbread_cabin_enters_tapped_no_food():
    """Gingerbread Cabin: "enters tapped unless you control 3+ other
    Forests; when it enters untapped, create a Food token." <3 other
    Forests -> enters tapped, no Food."""
    state = GameState(on_the_play=True)
    state.hand = [CardDef("Gingerbread Cabin", CardType.LAND, None, EffectId.GINGERBREAD_CABIN, subtypes=("Forest",))]
    cabin = play_land_from_hand(state, state.hand[0])
    assert cabin.tapped and cabin.flags["entered_tapped"] is True
    _drive_etb(state)
    assert not any(p.card_def.name == "Food" for p in state.battlefield)


def test_gingerbread_cabin_enters_untapped_creates_food():
    """3+ other Forests -> enters untapped, creates a Food. A prior
    Gingerbread Cabin counts as a Forest too (subtype Forest), so 2 basics +
    1 Cabin = 3."""
    state = GameState(on_the_play=True)
    state.battlefield = [
        Permanent(CardDef("Forest", CardType.LAND, None, EffectId.FOREST, basic=True, subtypes=("Forest",))),
        Permanent(CardDef("Forest", CardType.LAND, None, EffectId.FOREST, basic=True, subtypes=("Forest",))),
        Permanent(CardDef("Gingerbread Cabin", CardType.LAND, None, EffectId.GINGERBREAD_CABIN, subtypes=("Forest",))),
    ]
    state.hand = [CardDef("Gingerbread Cabin", CardType.LAND, None, EffectId.GINGERBREAD_CABIN, subtypes=("Forest",))]
    cabin = play_land_from_hand(state, state.hand[0])
    assert not cabin.tapped and cabin.flags["entered_tapped"] is False
    _drive_etb(state)
    assert any(p.card_def.name == "Food" for p in state.battlefield)


def test_pulse_of_murasa_own_graveyard():
    """Pulse of Murasa: return a creature/land card from your graveyard to
    hand, gain 6. Only creature/land cards are eligible (not an instant)."""
    state = GameState(on_the_play=True)
    pulse = CardDef("Pulse of Murasa", CardType.INSTANT, {"generic": 2, "G": 1}, EffectId.PULSE_OF_MURASA)
    state.hand = [pulse]
    state.graveyard = [state.new_instance(cd) for cd in [
        CardDef("A Creature", CardType.CREATURE, {"G": 1}, EffectId.FILLER, power=1, toughness=1),
        CardDef("A Land", CardType.LAND, None, EffectId.FOREST, basic=True),
        CardDef("An Instant", CardType.INSTANT, {"G": 1}, EffectId.FILLER),  # ineligible
    ]]
    state.life_total = 10
    cast_pulse_of_murasa(state, pulse)
    assert state.pending_resolution["kind"] == "choose_graveyard_card"
    pulse_opts = choose_graveyard_card_options(state)
    assert sorted(c.name for c in pulse_opts) == ["A Creature", "A Land"]  # instant excluded
    execute_choose_graveyard_card_option(state, next(o for o in pulse_opts if o.name == "A Creature"))
    resolve_top_of_stack(state)
    assert any(c.name == "A Creature" for c in state.hand)
    assert state.life_total == 16  # +6
    assert any(c.name == pulse.name for c in state.graveyard)  # the instant itself resolved to the graveyard


def test_pulse_of_murasa_cross_graveyard():
    """Cross-graveyard (Oracle "from A graveyard ... to its owner's hand"):
    the caster can return a card from the OPPONENT's graveyard -- it goes to
    the OPPONENT's (owner's) hand, and the caster still gains the 6."""
    state = GameState(on_the_play=True, players=[PlayerState(True), PlayerState(False)])
    state.active_idx = 0
    pulse2 = CardDef("Pulse of Murasa", CardType.INSTANT, {"generic": 2, "G": 1}, EffectId.PULSE_OF_MURASA)
    state.players[0].hand = [pulse2]
    opp_beast = CardDef("Opp Beast", CardType.CREATURE, {"G": 1}, EffectId.FILLER, power=2, toughness=2)
    state.players[1].graveyard = [state.new_instance(opp_beast)]
    state.players[0].life_total = 10
    cast_pulse_of_murasa(state, pulse2)
    beast_opts = choose_graveyard_card_options(state)
    assert [c.name for c in beast_opts] == ["Opp Beast"]  # the opponent's graveyard card is a legal target
    execute_choose_graveyard_card_option(state, beast_opts[0])
    resolve_top_of_stack(state)
    assert opp_beast in state.players[1].hand  # returned to ITS OWNER (the opponent), not the caster
    assert state.players[0].life_total == 16  # caster gained 6
    assert state.active_idx == 0


def test_priest_of_titania_counts_elves_both_battlefields():
    """Priest of Titania: {T} adds {G} per Elf on THE battlefield (both
    sides), incl. Masked Vandal (Changeling) and the opponent's Elves."""
    state = GameState(on_the_play=True, players=[PlayerState(True), PlayerState(False)])
    state.active_idx = 0
    priest = Permanent(registry.CARD_DEFS["Priest of Titania"])
    priest.slot = 1
    state.players[0].battlefield = [priest, Permanent(registry.CARD_DEFS["Llanowar Elves"]), Permanent(registry.CARD_DEFS["Masked Vandal"])]
    state.players[1].battlefield = [Permanent(registry.CARD_DEFS["Quirion Ranger"])]
    assert mana_output(priest, state) == ["G"] * 4  # Priest + Llanowar + Masked Vandal (changeling) + opp Quirion


def test_wellwisher_gain_life_per_elf():
    """Wellwisher: {T} gain 1 life per Elf."""
    state = GameState(on_the_play=True)
    well = Permanent(registry.CARD_DEFS["Wellwisher"])
    well.slot = 1
    state.battlefield = [well, Permanent(registry.CARD_DEFS["Llanowar Elves"])]  # 2 Elves
    wellwisher_activate(state, well)
    assert well.tapped
    resolve_top_of_stack(state)
    assert state.life_total == 22  # 20 + 2


def test_timberwatch_elf_pump():
    """Timberwatch Elf: {T} target creature +X/+X, X = # Elves."""
    state = GameState(on_the_play=True)
    tw = Permanent(registry.CARD_DEFS["Timberwatch Elf"])
    tw.slot = 1
    target = Permanent(CardDef("Beater", CardType.CREATURE, None, EffectId.FILLER, power=2, toughness=2))
    target.slot = 1
    state.battlefield = [tw, target, Permanent(registry.CARD_DEFS["Llanowar Elves"])]  # 2 Elves
    timberwatch_elf_activate(state, tw)
    execute_choose_any_target_creature(state, 0, "Beater", 1)
    resolve_top_of_stack(state)
    assert permanent_power(state, target) == 4 and permanent_toughness(state, target) == 4  # +2/+2


def test_rancor_castable_when_only_the_opponent_controls_a_creature():
    """Rancor's real text is plain "Enchant creature" -- either side, no
    "you control" restriction (unlike Cartouche of Solidarity) -- so it
    must still be legal to cast with zero creatures of the caster's own,
    as long as the OPPONENT controls one. any_creature_on_battlefield
    (caster-only) would wrongly report this as uncastable;
    any_creature_on_either_battlefield is the correct gate."""
    state = GameState(on_the_play=True, players=[PlayerState(True), PlayerState(False)])
    theirs = Permanent(CardDef("Theirs", CardType.CREATURE, None, EffectId.FILLER, power=2, toughness=2))
    theirs.slot = 1
    state.players[1].battlefield = [theirs]
    state.players[0].hand = [registry.CARD_DEFS["Rancor"]]
    assert registry.EFFECT_REGISTRY[EffectId.RANCOR]["cast"]["extra_legal"](state)
    cast_rancor(state, registry.CARD_DEFS["Rancor"])
    assert (1, "Theirs", 1) in choose_any_target_creature_options(state)


def test_quirion_ranger_untap_lets_player_choose_which_forest():
    """"Return a Forest you control to hand: Untap target creature." WHICH
    Forest pays the cost is a real player choice (602.5g), not an
    arbitrary auto-pick -- two Forests stop being fungible the instant one
    of them is worth keeping (e.g. it's enchanted by Utopia Sprawl)."""
    state = GameState(on_the_play=True)
    ranger = Permanent(registry.CARD_DEFS["Quirion Ranger"])
    ranger.slot = 1
    keep = Permanent(CardDef("Forest", CardType.LAND, None, EffectId.FOREST, basic=True, subtypes=("Forest",)))
    keep.slot = 1
    bounce = Permanent(CardDef("Forest", CardType.LAND, None, EffectId.FOREST, basic=True, subtypes=("Forest",)))
    bounce.slot = 2
    target = Permanent(CardDef("Beater", CardType.CREATURE, None, EffectId.FILLER, power=2, toughness=2))
    target.slot = 1
    target.tapped = True
    state.battlefield = [ranger, keep, bounce, target]
    quirion_ranger_untap_resolve(state, ranger)
    assert state.pending_resolution["kind"] == "choose_permanent"
    execute_choose_permanent_option(state, "Forest", 2)  # explicitly bounce slot 2, keep slot 1
    assert bounce not in state.battlefield and keep in state.battlefield
    assert [c.name for c in state.hand] == ["Forest"]
    assert state.pending_resolution["kind"] == "choose_any_target"
    execute_choose_any_target_creature(state, 0, "Beater", 1)
    resolve_top_of_stack(state)
    assert not target.tapped


def test_avenging_hunter_etb_initiative_venture():
    """Avenging Hunter: ETB -> you take the initiative -> a venture is
    queued and, on resolving, enters Secret Entrance (basic-land search)."""
    state = GameState(on_the_play=True, players=[PlayerState(True), PlayerState(False)])
    state.active_idx = state.turn_player_idx = 0
    state.players[0].library = [
        CardDef("Forest", CardType.LAND, None, EffectId.FOREST, basic=True, subtypes=("Forest",)),
        CardDef("z", CardType.SORCERY, {}, EffectId.FILLER),
    ]
    ah = registry.CARD_DEFS["Avenging Hunter"]
    assert has_keyword(state, Permanent(ah), "trample")  # 5/4 Trample
    state.players[0].hand = [ah]
    cast_permanent_from_hand(state, ah)  # ETB queued
    promote_triggers_to_stack(state)
    resolve_top_of_stack(state)  # ETB -> take_initiative (which queues the venture)
    assert state.initiative_idx == 0
    promote_triggers_to_stack(state)
    resolve_top_of_stack(state)  # venture -> Secret Entrance -> basic-land search
    assert state.players[0].dungeon_room == "Secret Entrance"
    assert state.pending_resolution["kind"] == "search_fetch"


def test_summoning_sickness_mana_dork_gated():
    """Summoning sickness (302.6): a creature's {T} ability (a mana dork)
    can't be used the turn it enters, unless it has haste."""
    state = GameState(on_the_play=True)
    llan = Permanent(registry.CARD_DEFS["Llanowar Elves"])  # summoning_sick=True by construction
    assert tap_summoning_locked(state, llan)  # sick mana dork -> can't tap for {G}
    llan.summoning_sick = False
    assert not tap_summoning_locked(state, llan)  # controlled since last turn -> can tap


def test_summoning_sickness_tap_ability_gated():
    """A non-mana {T} ability (Wellwisher's lifegain) is gated the same way."""
    state = GameState(on_the_play=True)
    well = Permanent(registry.CARD_DEFS["Wellwisher"])  # sick {T} ability
    assert tap_summoning_locked(state, well)
    assert not registry.EFFECT_REGISTRY[EffectId.WELLWISHER]["activated_abilities"]["lifegain"]["legal"](state, well)
    well.summoning_sick = False
    assert registry.EFFECT_REGISTRY[EffectId.WELLWISHER]["activated_abilities"]["lifegain"]["legal"](state, well)


def test_summoning_sickness_no_tap_exempt():
    """An ability with NO {T} in its cost is exempt (Wall of Roots)."""
    state = GameState(on_the_play=True)
    wor = Permanent(registry.CARD_DEFS["Wall of Roots"])  # sick, but its mana ability has NO {T}
    assert not tap_summoning_locked(state, wor)  # mana_no_tap -> not summoning-sickness gated


# --- REGRESSION: Sagu Wildling's Omen search used to accept ANY land ---

def test_roost_seek_omen_search_only_offers_basic_lands():
    """REGRESSION: cast_roost_seek's search used to accept any
    `c.card_type == CardType.LAND`; real oracle text ("search your library
    for a basic land card") restricts it to BASIC lands only (any of the
    five, no color restriction). A nonbasic land sitting in the library
    alongside a basic Forest must NOT be offered -- only the Forest."""
    state = GameState(on_the_play=True)
    roost_seek = CardDef("Sagu Wildling", CardType.SORCERY, {"G": 1}, EffectId.ROOST_SEEK)
    state.hand = [roost_seek]
    nonbasic = CardDef("Nonbasic Land", CardType.LAND, None, EffectId.FILLER)  # a land, but no basic=True
    basic_forest = CardDef("Forest", CardType.LAND, None, EffectId.FOREST, basic=True)
    state.library = [nonbasic, basic_forest]
    cast_roost_seek(state, roost_seek)
    assert state.pending_resolution["kind"] == "search_fetch"
    assert search_fetch_options(state) == ["Forest"]  # the nonbasic land is correctly excluded
    execute_search_fetch_option(state, "Forest")
    assert [c.name for c in state.hand] == ["Forest"]
    assert any(c.name == "Nonbasic Land" for c in state.library)  # untouched, never offered


# --- REGRESSION: Gatecreeper Vine's ETB search used to accept ANY land ---

def test_gatecreeper_vine_etb_only_offers_basic_lands():
    """REGRESSION: gatecreeper_vine_etb's search used to accept any land;
    real oracle text ("a basic land card or a Gate card", no Gate-subtype
    card in this pool) restricts it to basics in practice. A nonbasic land
    must NOT be offered. Also covers the "optional even when a target
    exists" half of the missing-coverage item (the search is begun with
    optional=True, offering a decline, unlike Land Grant/Crop Rotation's
    mandatory fetches)."""
    state = GameState(on_the_play=True)
    nonbasic = CardDef("Nonbasic Land", CardType.LAND, None, EffectId.FILLER)
    basic_forest = CardDef("Forest", CardType.LAND, None, EffectId.FOREST, basic=True)
    state.library = [nonbasic, basic_forest]
    gatecreeper_vine_etb(state)
    assert state.pending_resolution["kind"] == "search_fetch"
    assert state.pending_resolution["optional"] is True  # "may search" -- optional even though a basic is available
    assert search_fetch_options(state) == ["Forest"]  # the nonbasic land is correctly excluded
    execute_search_fetch_option(state, "Forest")
    assert [c.name for c in state.hand] == ["Forest"]


def test_generous_ent_forestcycle():
    """Generous Ent's OTHER mode: {1}, discard from hand: search library
    for a Forest specifically (the fixed cost recorded on the card, plus
    the actual discard+search). Its ETB Food token is covered by
    test_generous_ent_etb_creates_food_token above -- this is the
    forestcycle mode alone."""
    state = GameState(on_the_play=True)
    ent = CardDef(
        "Generous Ent", CardType.CREATURE, {"generic": 5, "G": 1}, EffectId.GENEROUS_ENT,
        forestcycling_cost={"generic": 1}, power=5, toughness=5,
    )
    assert ent.extra["forestcycling_cost"] == {"generic": 1}
    state.hand = [ent]
    state.library = [CardDef("Forest", CardType.LAND, None, EffectId.FOREST, basic=True)]
    forestcycle_generous_ent(state, ent)
    assert [c.name for c in state.hand] == ["Forest"]
    assert [c.name for c in state.graveyard] == ["Generous Ent"]  # discarded itself, not the fetched land
    assert state.library == []


def test_nyxborn_hydra_bestow_no_legal_target_enters_as_creature():
    """Bestow, no_target_fallback (real MTG 702.103e): casting for the
    bestow cost with GENUINELY ZERO legal creatures to enchant at cast
    time -- not one that died later, the fizzle-after-cast branch already
    covered elsewhere -- does not fizzle to the graveyard like an ordinary
    Aura. It still enters the battlefield, as a creature, with its X
    +1/+1 counters."""
    state = GameState(on_the_play=True)
    hydra_card = CardDef("Nyxborn Hydra", CardType.CREATURE, {"G": 1}, EffectId.NYXBORN_HYDRA, power=0, toughness=1)
    state.hand = [hydra_card]
    # No creature anywhere -- begin_choose_any_target auto-completes with
    # None (empty candidate pool), so this never even opens a pending choice.
    cast_nyxborn_hydra_bestow(2)(state, hydra_card)
    assert state.pending_resolution is None
    assert len(state.stack) == 1
    resolve_top_of_stack(state)
    hydra_permanent = next(p for p in state.battlefield if p.card_def.name == "Nyxborn Hydra")
    assert hydra_permanent.card_type == CardType.CREATURE  # entered as a creature, NOT an Aura/Enchantment
    assert hydra_permanent.type_override is None
    assert hydra_permanent.counters["+1/+1"] == 2


def test_quirion_ranger_untap_illegal_no_forest_to_return():
    """quirion_ranger_untap_legal: false when there's no Forest to return
    (the happy path is covered by test_quirion_ranger_untap_lets_player_
    choose_which_forest above)."""
    state = GameState(on_the_play=True)
    ranger = Permanent(registry.CARD_DEFS["Quirion Ranger"])
    state.battlefield = [ranger]  # no Forest anywhere
    assert not quirion_ranger_untap_legal(state, ranger)


def test_quirion_ranger_untap_illegal_already_used_this_turn():
    """quirion_ranger_untap_legal: false once already used this turn, even
    with a Forest available."""
    state = GameState(on_the_play=True)
    ranger = Permanent(registry.CARD_DEFS["Quirion Ranger"])
    forest = Permanent(CardDef("Forest", CardType.LAND, None, EffectId.FOREST, basic=True))
    state.battlefield = [ranger, forest]
    ranger.flags["used_this_turn"] = True
    assert not quirion_ranger_untap_legal(state, ranger)


def test_winding_way_land_mode():
    """Winding Way, land mode (creature mode is covered by
    test_winding_way_via_action_table above): reveal top 4, lands to hand,
    the rest to the graveyard alongside Winding Way itself."""
    state = GameState(on_the_play=True)
    ww = CardDef("Winding Way", CardType.SORCERY, {"generic": 1, "G": 1}, EffectId.WINDING_WAY)
    state.hand = [ww]
    state.library = [
        CardDef("Land1", CardType.LAND, None, EffectId.FOREST, basic=True),
        CardDef("Bear1", CardType.CREATURE, {"G": 1}, EffectId.FILLER, power=2, toughness=2),
        CardDef("Land2", CardType.LAND, None, EffectId.FOREST, basic=True),
        CardDef("Instant1", CardType.INSTANT, {"G": 1}, EffectId.FILLER),
        CardDef("Never Revealed", CardType.LAND, None, EffectId.FOREST, basic=True),  # 5th card, stays in library
    ]
    cast_winding_way_land(state, ww)
    assert sorted(c.name for c in state.hand) == ["Land1", "Land2"]
    assert sorted(c.name for c in state.graveyard) == ["Bear1", "Instant1", "Winding Way"]
    assert [c.name for c in state.library] == ["Never Revealed"]


def test_lead_the_stampede_cast_and_resolve():
    """{2}{G}: look at top 5, may reveal any number of creatures to hand,
    rest to the bottom in any order the model chooses."""
    state = GameState(on_the_play=True)
    lts = CardDef("Lead the Stampede", CardType.SORCERY, {"generic": 2, "G": 1}, EffectId.LEAD_THE_STAMPEDE)
    state.hand = [lts]
    state.library = [
        CardDef("CreatureA", CardType.CREATURE, {"G": 1}, EffectId.FILLER, power=1, toughness=1),
        CardDef("CreatureB", CardType.CREATURE, {"G": 1}, EffectId.FILLER, power=1, toughness=1),
        CardDef("CreatureC", CardType.CREATURE, {"G": 1}, EffectId.FILLER, power=1, toughness=1),
        CardDef("LandX", CardType.LAND, None, EffectId.FOREST, basic=True),
        CardDef("InstantY", CardType.INSTANT, {"G": 1}, EffectId.FILLER),
        CardDef("Never Revealed", CardType.SORCERY, {}, EffectId.FILLER),  # 6th card, stays in library
    ]
    cast_lead_the_stampede(state, lts)
    assert state.pending_resolution["kind"] == "select_to_hand"

    assert select_to_hand_options(state) == ["keep", "bottom"]  # front = CreatureA, eligible
    execute_select_to_hand_option(state, "keep")
    assert select_to_hand_options(state) == ["keep", "bottom"]  # front = CreatureB, eligible
    execute_select_to_hand_option(state, "bottom")
    assert select_to_hand_options(state) == ["keep", "bottom"]  # front = CreatureC, eligible
    execute_select_to_hand_option(state, "keep")
    assert select_to_hand_options(state) == ["bottom"]  # front = LandX, NOT eligible (not a creature)
    execute_select_to_hand_option(state, "bottom")
    assert select_to_hand_options(state) == ["bottom"]  # front = InstantY, not eligible
    execute_select_to_hand_option(state, "bottom")

    # 3 bottomed (CreatureB, LandX, InstantY) -> ordering phase, one option per distinct name.
    assert sorted(select_to_hand_options(state)) == ["CreatureB", "InstantY", "LandX"]
    execute_select_to_hand_option(state, "CreatureB")
    execute_select_to_hand_option(state, "LandX")
    execute_select_to_hand_option(state, "InstantY")

    assert state.pending_resolution is None
    assert sorted(c.name for c in state.hand) == ["CreatureA", "CreatureC"]  # the two kept
    assert sorted(c.name for c in state.graveyard) == ["Lead the Stampede"]  # only itself -- bottomed cards aren't graveyarded
    assert [c.name for c in state.library] == ["Never Revealed", "CreatureB", "LandX", "InstantY"]  # bottomed in chosen order


def test_land_grant_normal_cast_searches_forest():
    """Land Grant, normal cast: search library for a Forest specifically --
    a single fixed target name (unlike Roost Seek's real model choice), so
    this resolves immediately once invoked."""
    state = GameState(on_the_play=True)
    land_grant = CardDef("Land Grant", CardType.SORCERY, {"generic": 1, "G": 1}, EffectId.LAND_GRANT)
    state.hand = [land_grant]
    state.library = [
        CardDef("Forest", CardType.LAND, None, EffectId.FOREST, basic=True),
        CardDef("Plains", CardType.LAND, None, EffectId.PLAINS, basic=True),
    ]
    cast_land_grant(state, land_grant)
    assert [c.name for c in state.hand] == ["Forest"]
    assert [c.name for c in state.graveyard] == ["Land Grant"]
    assert [c.name for c in state.library] == ["Plains"]


def test_land_grant_alt_cost_free_reveal_hand():
    """Land Grant's free alt-cost ("reveal your hand" instead of paying):
    legal only with no land cards in hand; pushes to the stack (unlike the
    direct-resolve normal cast) since nothing else defers it."""
    state = GameState(on_the_play=True)
    land_grant = CardDef("Land Grant", CardType.SORCERY, {"generic": 1, "G": 1}, EffectId.LAND_GRANT)
    state.hand = [land_grant]
    state.library = [CardDef("Forest", CardType.LAND, None, EffectId.FOREST, basic=True)]
    assert land_grant_alt_cost_legal(state)  # no OTHER land card in hand

    other_land = CardDef("Swamp", CardType.LAND, None, EffectId.SWAMP, basic=True)
    state.hand.append(other_land)
    assert not land_grant_alt_cost_legal(state)  # a land card in hand -> the free alt-cost is no longer legal
    state.hand.remove(other_land)

    cast_land_grant_alt(state, land_grant)
    assert state.hand == []  # left hand at cast, sitting on the stack unresolved
    assert len(state.stack) == 1
    resolve_top_of_stack(state)
    assert [c.name for c in state.hand] == ["Forest"]
    assert [c.name for c in state.graveyard] == ["Land Grant"]


def test_crop_rotation_sacrifice_and_unrestricted_land_search():
    """{G}, sacrifice a land (a real ADDITIONAL COST, chosen and paid before
    the spell is even fully cast): search library for a land -- put it
    DIRECTLY onto the battlefield (unlike Gatecreeper Vine/Sagu Wildling/
    Land Grant, which all restrict to basics and go to hand; Crop
    Rotation's own real text has no such restriction -- "search your
    library for a land card")."""
    state = GameState(on_the_play=True)
    crop = CardDef("Crop Rotation", CardType.INSTANT, {"G": 1}, EffectId.CROP_ROTATION)
    state.hand = [crop]
    # Named "Forest" (matching registry.CARD_DEFS) so sacrifice_to_graveyard
    # treats it as a real card, not a token, and moves it to the graveyard.
    old_land = Permanent(CardDef("Forest", CardType.LAND, None, EffectId.FOREST, basic=True))
    state.battlefield = [old_land]
    nonbasic = CardDef("Reflecting Pool", CardType.LAND, None, EffectId.FILLER)  # nonbasic -- must still be offered
    state.library = [nonbasic]

    cast_crop_rotation(state, crop)
    assert state.pending_resolution["kind"] == "choose_permanent"  # the sacrifice IS the additional cost, chosen now
    execute_choose_permanent_option(state, "Forest", 1)
    assert old_land not in state.battlefield
    assert any(c.name == "Forest" for c in state.graveyard)  # sacrificed -- a real card, to the graveyard
    assert state.pending_resolution is None and len(state.stack) == 1  # only the search itself waits on the stack

    resolve_top_of_stack(state)
    assert state.pending_resolution["kind"] == "search_fetch"
    assert search_fetch_options(state) == ["Reflecting Pool"]  # any land, nonbasic included -- no restriction
    execute_search_fetch_option(state, "Reflecting Pool")
    assert state.pending_resolution is None
    assert any(p.card_def.name == "Reflecting Pool" for p in state.battlefield)  # straight onto the battlefield
    assert all(c.name != "Reflecting Pool" for c in state.hand)  # NOT to hand -- unlike Gatecreeper/Land Grant


def test_ancient_stirrings_resolve_picks_a_card():
    """{G}: look at top 5, may take one noncreature colorless card to hand,
    rest to the bottom in random order. Exercises the real resolve/options/
    execute trio (only its action-table row existence was checked before),
    including is_noncreature_colorless's two independent exclusions: a
    creature (even a colorless-cost one) is never eligible, and neither is
    a colored noncreature."""
    state = GameState(on_the_play=True)
    stirrings = CardDef("Ancient Stirrings", CardType.SORCERY, {"G": 1}, EffectId.ANCIENT_STIRRINGS)
    state.hand = [stirrings]
    state.library = [
        CardDef("Land Card", CardType.LAND, None, EffectId.FOREST, basic=True),  # eligible: no cast_cost -> colorless
        CardDef("Colorless Artifact", CardType.ARTIFACT, {"generic": 3}, EffectId.FILLER),  # eligible: noncreature, colorless
        CardDef("Green Creature", CardType.CREATURE, {"G": 1}, EffectId.FILLER, power=1, toughness=1),  # ineligible: creature
        CardDef("Green Sorcery", CardType.SORCERY, {"G": 1}, EffectId.FILLER),  # ineligible: colored
        CardDef("Colorless Creature", CardType.CREATURE, {"generic": 2}, EffectId.FILLER, power=1, toughness=1),  # ineligible: creature, even though colorless
        CardDef("Untouched", CardType.LAND, None, EffectId.FOREST, basic=True),  # 6th card, never revealed
    ]
    cast_ancient_stirrings(state, stirrings)
    assert state.pending_resolution["kind"] == "ancient_stirrings"
    assert ancient_stirrings_options(state) == ["Colorless Artifact", "Land Card", "decline"]
    execute_ancient_stirrings_option(state, "Colorless Artifact")
    assert state.pending_resolution is None
    assert [c.name for c in state.hand] == ["Colorless Artifact"]
    assert [c.name for c in state.graveyard] == ["Ancient Stirrings"]  # unlike Malevolent Rumble, the rest go to the bottom, not here
    assert sorted(c.name for c in state.library) == [
        "Colorless Creature", "Green Creature", "Green Sorcery", "Land Card", "Untouched",
    ]


def test_bramble_wurm_etb_and_keywords():
    """Bramble Wurm: ETB gain 5 life, plus Trample + Reach (its graveyard
    ability is already thoroughly tested elsewhere -- see
    test_bramble_wurm_graveyard_ability above)."""
    state = GameState(on_the_play=True)
    wurm = CardDef(
        "Bramble Wurm", CardType.CREATURE, {"generic": 6, "G": 1}, EffectId.BRAMBLE_WURM, power=7, toughness=6,
        gy_ability_cost={"generic": 2, "G": 1},
    )
    state.hand = [wurm]
    cast_permanent_from_hand(state, wurm)
    perm = next(p for p in state.battlefield if p.card_def.name == "Bramble Wurm")
    assert has_keyword(state, perm, "trample")
    assert has_keyword(state, perm, "reach")
    assert state.life_total == 20
    promote_triggers_to_stack(state)
    resolve_top_of_stack(state)
    assert state.life_total == 25  # +5, once the ETB resolves


def test_gladecover_scout_hexproof():
    """Gladecover Scout: vanilla body with hexproof -- a real targeting
    restriction, not just a flavor keyword."""
    state = GameState(on_the_play=True, players=[PlayerState(True), PlayerState(False)])
    scout = Permanent(registry.CARD_DEFS["Gladecover Scout"])
    state.players[0].battlefield = [scout]
    assert has_keyword(state, scout, "hexproof")
    assert can_be_targeted(state, scout, 0)  # its own controller may still target it
    assert not can_be_targeted(state, scout, 1)  # an opponent may not


def test_silhana_ledgewalker_hexproof():
    """Silhana Ledgewalker's own hexproof (its "can't be blocked except by
    flying" evasion is already covered via the real action table
    elsewhere)."""
    state = GameState(on_the_play=True, players=[PlayerState(True), PlayerState(False)])
    silhana = Permanent(registry.CARD_DEFS["Silhana Ledgewalker"])
    state.players[0].battlefield = [silhana]
    assert has_keyword(state, silhana, "hexproof")
    assert can_be_targeted(state, silhana, 0)
    assert not can_be_targeted(state, silhana, 1)


def test_ancestral_mask_real_cast():
    """Drives the real cast_ancestral_mask itself (the registry "cast"
    entry) -- pt_bonus/orphan-to-graveyard behavior is already covered via
    direct construction in tests/game/effects/test_stats.py and
    test_state_based.py."""
    state = GameState(on_the_play=True)
    target = Permanent(CardDef("Target Creature", CardType.CREATURE, None, EffectId.FILLER, power=2, toughness=2))
    mask = CardDef("Ancestral Mask", CardType.ENCHANTMENT, {"generic": 2, "G": 1}, EffectId.ANCESTRAL_MASK)
    state.battlefield = [target]
    state.hand = [mask]
    cast_ancestral_mask(state, mask)
    assert state.pending_resolution["kind"] == "choose_any_target"
    execute_choose_any_target_creature(state, 0, "Target Creature", 1)
    assert state.hand == []
    resolve_top_of_stack(state)
    attached = next(p for p in state.battlefield if p.card_def.name == "Ancestral Mask")
    assert attached.flags["enchanting"] is target


def test_utopia_sprawl_non_green_mode_via_action_table():
    """Utopia Sprawl's other 4 color modes (green is covered by
    test_utopia_sprawl_via_action_table above): choosing "Mode 2" (white --
    registry cast_modes order: green, white, blue, black, red) all the way
    through resolution, confirming the chosen color -- not just green -- is
    what gets recorded on the Aura."""
    us_dl = [("Utopia Sprawl", 4), ("Forest", 8)]
    us_byname = {a[0]: (a[1], a[2]) for a in drl_env.build_action_table(us_dl, registry.EFFECT_REGISTRY)}
    us_state = GameState(on_the_play=True)
    us_state.phase = Phase.MAIN1
    us_state.turn_player_idx = 0
    us_state.active_idx = 0
    us_state.hand = [registry.CARD_DEFS["Utopia Sprawl"]]
    us_forest = Permanent(registry.CARD_DEFS["Forest"])
    us_state.battlefield = [us_forest]
    activate_mana_source(us_state, us_forest)
    cast_legal, cast_execute = us_byname["Cast Utopia Sprawl"]
    assert cast_legal(us_state)
    cast_execute(us_state)
    assert us_state.pending_resolution["kind"] == "choose_cast_mode"
    mode2_legal, mode2_execute = us_byname["Mode 2"]
    assert mode2_legal(us_state)
    mode2_execute(us_state)
    assert us_state.pending_resolution["kind"] == "pay_cost"
    guard = 0
    while us_state.pending_resolution is not None and us_state.pending_resolution["kind"] == "pay_cost":
        guard += 1
        assert guard < 30
        execute_pool_spend(us_state, pool_spend_options(us_state)[0])
    assert us_state.pending_resolution["kind"] == "choose_any_target"
    execute_choose_any_target_creature(us_state, 0, "Forest", 1)
    resolve_top_of_stack(us_state)
    sprawl = next(p for p in us_state.battlefield if p.card_def.name == "Utopia Sprawl")
    assert sprawl.flags["bonus_mana_color"] == "W"


def test_abundant_growth_real_cast_attaches_and_draws():
    """Drives the real cast_abundant_growth -> cast_aura -> attach flow
    (the granted-mana mechanism itself is already covered elsewhere via
    directly-constructed flags, bypassing casting -- this confirms the ETB
    draw and the attach actually happen through the real cast)."""
    state = GameState(on_the_play=True)
    land = Permanent(CardDef("Some Land", CardType.LAND, None, EffectId.FOREST, basic=True))
    growth = CardDef("Abundant Growth", CardType.ENCHANTMENT, {"G": 1}, EffectId.ABUNDANT_GROWTH)
    state.battlefield = [land]
    state.hand = [growth]
    state.library = [CardDef("Drawn Card", CardType.SORCERY, {}, EffectId.FILLER)]
    cast_abundant_growth(state, growth)
    assert state.pending_resolution["kind"] == "choose_any_target"
    execute_choose_any_target_creature(state, 0, "Some Land", 1)
    assert state.hand == []  # left hand at cast, sitting on the stack unresolved
    resolve_top_of_stack(state)
    assert [c.name for c in state.hand] == ["Drawn Card"]  # the ETB draw
    attached = next(p for p in state.battlefield if p.card_def.name == "Abundant Growth")
    assert attached.flags["enchanting"] is land
    assert attached.flags["bonus_mana_colors"] == set(COLORS)


def test_pulse_of_murasa_fizzle_target_removed():
    """(fizzle, 608.2b): the chosen graveyard card leaves the graveyard
    before resolution -- the whole spell does nothing, lifegain included
    (both non-fizzle branches are covered above)."""
    state = GameState(on_the_play=True)
    pulse = CardDef("Pulse of Murasa", CardType.INSTANT, {"generic": 2, "G": 1}, EffectId.PULSE_OF_MURASA)
    state.hand = [pulse]
    creature_inst = state.new_instance(CardDef("A Creature", CardType.CREATURE, {"G": 1}, EffectId.FILLER, power=1, toughness=1))
    state.graveyard = [creature_inst]
    state.life_total = 10
    cast_pulse_of_murasa(state, pulse)
    assert state.pending_resolution["kind"] == "choose_graveyard_card"
    execute_choose_graveyard_card_option(state, creature_inst)
    state.graveyard.remove(creature_inst)  # removed in response, before resolution
    log = io.StringIO()
    with contextlib.redirect_stdout(log):
        resolve_top_of_stack(state)
    assert "fizzle" in log.getvalue().lower()
    assert state.life_total == 10  # no lifegain -- the whole spell fizzled, not just the return
    assert all(c.name != "A Creature" for c in state.hand)
    assert any(c.name == pulse.name for c in state.graveyard)  # Pulse itself still resolves to its own graveyard


def test_fyndhorn_elves_mana_tap():
    """Fyndhorn Elves: {T}: Add {G} (a functional twin of Llanowar Elves) --
    zero direct references before this test. Confirms both the mana output
    and the same summoning-sickness gating Llanowar Elves gets."""
    state = GameState(on_the_play=True)
    fyndhorn = Permanent(registry.CARD_DEFS["Fyndhorn Elves"])  # summoning_sick=True by construction
    assert tap_summoning_locked(state, fyndhorn)  # sick mana dork -> can't tap for {G} yet
    fyndhorn.summoning_sick = False
    state.battlefield = [fyndhorn]
    assert ("Fyndhorn Elves", None) in mana_ability_options(state)
    activate_mana_source(state, fyndhorn)
    assert fyndhorn.tapped
    assert state.mana_pool.get("G", 0) == 1


def test_skeleton_token_menace():
    """The Skeleton token (EffectId.SKELETON_TOKEN, "Undercity Catacombs
    Skeleton", registry-only -- not in GREEN_CARD_CATALOG): 4/1 with
    menace. Zero references before this test."""
    state = GameState(on_the_play=True)
    skeleton = create_token(state, SKELETON_TOKEN_CARD_DEF)
    assert has_keyword(state, skeleton, "menace")
    assert permanent_power(state, skeleton) == 4 and permanent_toughness(state, skeleton) == 1
