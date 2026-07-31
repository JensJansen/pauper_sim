"""Tests for game.catalog.red_cards. See the module under test for the
card-implementation rationale (real-rules citations, etc.) each test below
guards."""

import contextlib
import io

import drl_env
from game import registry, resolution
from game.cards import CardDef, CardType, EffectId
from game.catalog.red_cards import (
    _goblin_bushwhacker_kicked,
    activate_reckless_lackey_sac,
    cast_breath_weapon,
    cast_chain_lightning,
    cast_cleansing_wildfire,
    cast_end_the_festivities,
    cast_galvanic_blast,
    cast_highway_robbery,
    cast_lightning_bolt,
    cast_rally_at_the_hornburg,
    cast_reckless_impulse,
    experimental_synthesizer_sac,
    krark_clan_shaman_activate,
)
from game.effects.casting import cast_permanent_from_hand
from game.effects.stack import resolve_top_of_stack
from game.effects.state_based import sacrifice_to_graveyard
from game.effects.stats import has_keyword, permanent_power
from game.effects.triggers import promote_triggers_to_stack
from game.mana import activate_mana_source, execute_pool_spend, pool_spend_options
from game.state import GameState, Permanent, PlayerState
from game.turn import Phase, untap_step


def test_breath_weapon_symmetric_two_damage_wipe():
    """Real text: deals 2 damage to each NON-DRAGON creature. No card in
    this catalog is ever a Dragon, so that filter is always satisfied: this
    hits every creature currently in play, on either player's battlefield,
    a real symmetric board wipe (this deck's own creatures included)."""
    state = GameState(on_the_play=True, players=[PlayerState(True), PlayerState(False)])
    breath_weapon = CardDef("Breath Weapon", CardType.INSTANT, {"generic": 2, "R": 1}, EffectId.BREATH_WEAPON)
    state.hand = [breath_weapon]
    mine_dies = Permanent(CardDef("Mine (dies)", CardType.CREATURE, None, EffectId.FILLER, power=1, toughness=2))
    mine_survives = Permanent(CardDef("Mine (survives)", CardType.CREATURE, None, EffectId.FILLER, power=1, toughness=3))
    theirs_dies = Permanent(CardDef("Theirs (dies)", CardType.CREATURE, None, EffectId.FILLER, power=1, toughness=1))
    not_a_creature = Permanent(CardDef("Some Land", CardType.LAND, None, EffectId.FILLER))
    state.players[0].battlefield = [mine_dies, mine_survives, not_a_creature]
    state.players[1].battlefield = [theirs_dies]

    cast_breath_weapon(state, breath_weapon)
    assert state.hand == [] and any(c.name == breath_weapon.name for c in state.graveyard)
    assert mine_dies not in state.players[0].battlefield  # 2 damage >= 2 toughness -- this deck's own creature dies too
    assert mine_survives in state.players[0].battlefield and mine_survives.damage_marked == 2
    assert theirs_dies not in state.players[1].battlefield
    assert not_a_creature in state.players[0].battlefield  # a land is never a valid target


def test_end_the_festivities_hits_only_the_opponent_not_the_caster():
    # Real text (Innistrad: Crimson Vow, {R}) is "deals 1 damage to each
    # opponent and each creature and planeswalker they control" --
    # asymmetric, unlike Breath Weapon above: the caster's own life total and
    # creatures are completely untouched, only the opponent's face and board
    # take the 1 damage.
    state = GameState(on_the_play=True, players=[PlayerState(True), PlayerState(False)])
    festivities = CardDef("End the Festivities", CardType.SORCERY, {"R": 1}, EffectId.END_THE_FESTIVITIES)
    state.hand = [festivities]
    mine_untouched = Permanent(CardDef("Mine (untouched)", CardType.CREATURE, None, EffectId.FILLER, power=1, toughness=1))
    theirs_dies = Permanent(CardDef("Theirs (dies)", CardType.CREATURE, None, EffectId.FILLER, power=1, toughness=1))
    theirs_survives = Permanent(CardDef("Theirs (survives)", CardType.CREATURE, None, EffectId.FILLER, power=1, toughness=2))
    their_land = Permanent(CardDef("Their Land", CardType.LAND, None, EffectId.FILLER))
    state.players[0].battlefield = [mine_untouched]
    state.players[1].battlefield = [theirs_dies, theirs_survives, their_land]
    opponent_life_before = state.players[1].life_total

    cast_end_the_festivities(state, festivities)
    assert state.hand == [] and any(c.name == festivities.name for c in state.graveyard)
    assert state.players[1].life_total == opponent_life_before - 1  # the opponent's face takes the 1 damage
    assert state.players[0].life_total == 20  # the CASTER is never damaged -- not symmetric
    assert mine_untouched in state.players[0].battlefield and mine_untouched.damage_marked == 0  # own board untouched
    assert theirs_dies not in state.players[1].battlefield  # 1 damage >= 1 toughness
    assert theirs_survives in state.players[1].battlefield and theirs_survives.damage_marked == 1
    assert their_land in state.players[1].battlefield  # a land is never a valid target


def test_highway_robbery_discard_path_triggers_madness():
    """Highway Robbery: "discard a card or sacrifice a land. If you do, draw
    two cards." Discard path, discarding a Madness card: Madness's own
    exile-not-graveyard replacement effect fires regardless of WHY the card
    was discarded (an optional cost here, not Faithless Looting's own
    discard-2 effect)."""
    state = GameState(on_the_play=True)
    hr = CardDef("Highway Robbery", CardType.SORCERY, {"generic": 1, "R": 1}, EffectId.HIGHWAY_ROBBERY)
    fiery_temper = CardDef("Fiery Temper", CardType.INSTANT, {"generic": 1, "R": 2}, EffectId.FIERY_TEMPER)
    state.hand = [hr, fiery_temper]
    state.library = [CardDef(f"Filler {i}", CardType.LAND, None, EffectId.MOUNTAIN) for i in range(2)]
    cast_highway_robbery(state, hr)
    assert state.pending_resolution["kind"] == "discard_or_sacrifice"
    resolution.execute_discard_or_sacrifice_option(state, "discard", "Fiery Temper")
    assert len(state.hand) == 2  # drew 2
    assert state.exile and state.exile[0][0].name == "Fiery Temper"  # exiled, not graveyarded -- Madness
    assert state.trigger_queue and state.trigger_queue[0]["kind"] == "madness"


def test_highway_robbery_sacrifice_land_path():
    """Sacrifice-a-land path -- the alternative cost to discarding a card.
    Also regression coverage: execute_discard_or_sacrifice_option's own
    sacrifice branch now fires fire_sacrifice_triggers too (Gixian
    Infiltrator used to never see a sacrifice paid this way)."""
    state2 = GameState(on_the_play=True)
    hr2 = CardDef("Highway Robbery", CardType.SORCERY, {"generic": 1, "R": 1}, EffectId.HIGHWAY_ROBBERY)
    mountain = Permanent(CardDef("Mountain", CardType.LAND, None, EffectId.MOUNTAIN))
    gix = Permanent(registry.CARD_DEFS["Gixian Infiltrator"])
    gix.slot = 1
    state2.hand = [hr2]
    state2.battlefield = [mountain, gix]
    state2.library = [CardDef(f"Filler {i}", CardType.LAND, None, EffectId.MOUNTAIN) for i in range(2)]
    cast_highway_robbery(state2, hr2)
    resolution.execute_discard_or_sacrifice_option(state2, "sacrifice", "Mountain")
    assert state2.battlefield == [gix]
    assert sorted(c.name for c in state2.graveyard) == ["Highway Robbery", "Mountain"]
    assert len(state2.hand) == 2
    queued = [e for e in state2.trigger_queue if e["type"] == "on_sacrifice"]
    assert queued and queued[0]["permanent"] is gix


def test_highway_robbery_decline_no_draw():
    """Decline -- genuinely optional even with something payable on hand (a
    spare land AND a spare card), no draw either way."""
    state3 = GameState(on_the_play=True)
    hr3 = CardDef("Highway Robbery", CardType.SORCERY, {"generic": 1, "R": 1}, EffectId.HIGHWAY_ROBBERY)
    spare_card = CardDef("Lightning Bolt", CardType.INSTANT, {"R": 1}, EffectId.LIGHTNING_BOLT)
    spare_land = Permanent(CardDef("Mountain", CardType.LAND, None, EffectId.MOUNTAIN))
    state3.hand = [hr3, spare_card]
    state3.battlefield = [spare_land]
    cast_highway_robbery(state3, hr3)
    assert state3.pending_resolution["kind"] == "discard_or_sacrifice"  # genuinely offered, not auto-completed
    resolution.execute_discard_or_sacrifice_decline(state3)
    assert [c.name for c in state3.hand] == ["Lightning Bolt"]  # untouched, no draw
    assert spare_land in state3.battlefield  # untouched
    assert state3.pending_resolution is None


def _fresh_bolt_state():
    bolt = CardDef("Lightning Bolt", CardType.INSTANT, {"R": 1}, EffectId.LIGHTNING_BOLT)
    s = GameState(on_the_play=True, players=[PlayerState(True), PlayerState(False)])
    s.hand = [bolt]
    return s, bolt


def test_lightning_bolt_hits_opponent_creature():
    """Lightning Bolt: faithful "3 damage to any target" -- target locked at
    cast (capture_any_target), applied or fizzled at resolution.

    (a) opponent creature -> 3 damage marked on that exact creature."""
    state, bolt = _fresh_bolt_state()
    opp_creature = Permanent(CardDef("Grizzly Bears", CardType.CREATURE, None, EffectId.FILLER, power=2, toughness=4))
    opp_creature.slot = 1
    state.players[1].battlefield = [opp_creature]
    cast_lightning_bolt(state, bolt)  # begins choose_any_target (precast)
    resolution.execute_choose_any_target_creature(state, 1, "Grizzly Bears", 1)  # lock the opponent's creature
    assert state.hand == []  # left hand at cast, paid-but-unresolved on the stack
    resolve_top_of_stack(state)
    assert opp_creature.damage_marked == 3 and any(c.name == bolt.name for c in state.graveyard)


def test_lightning_bolt_hits_opponent_face():
    """(b) opponent player -> 3 to the face (20 -> 17)."""
    state, bolt = _fresh_bolt_state()
    cast_lightning_bolt(state, bolt)
    resolution.execute_choose_any_target_player(state, 1)
    resolve_top_of_stack(state)
    assert state.players[1].life_total == 17  # 20 - 3 to the opponent's face


def test_lightning_bolt_can_target_self():
    """(c) yourself -> legal, 3 to own face (20 -> 17), pure self-damage."""
    state, bolt = _fresh_bolt_state()
    cast_lightning_bolt(state, bolt)
    resolution.execute_choose_any_target_player(state, 0)
    resolve_top_of_stack(state)
    assert state.players[0].life_total == 17  # 3 to your OWN face -- pure self-damage, never opponent


def test_lightning_bolt_fizzles_if_target_removed():
    """(d) FIZZLE: chosen creature gone before resolution -> no effect."""
    state, bolt = _fresh_bolt_state()
    doomed = Permanent(CardDef("Grizzly Bears", CardType.CREATURE, None, EffectId.FILLER, power=2, toughness=4))
    doomed.slot = 1
    state.players[1].battlefield = [doomed]
    cast_lightning_bolt(state, bolt)
    resolution.execute_choose_any_target_creature(state, 1, "Grizzly Bears", 1)
    state.players[1].battlefield = []  # target leaves before the bolt resolves
    fizzle_log = io.StringIO()
    with contextlib.redirect_stdout(fizzle_log):
        resolve_top_of_stack(state)
    assert "fizzle" in fizzle_log.getvalue().lower()
    assert any(c.name == bolt.name for c in state.graveyard) and state.players[1].life_total == 20  # nothing happened


def test_lightning_bolt_hexproof_targeting():
    """(e) hexproof: an OPPONENT'S hexproof creature is NOT a legal Bolt
    target, but the caster's OWN hexproof creature still is (boggles' whole
    point -- opponents can't burn its bogles)."""
    original_filler = registry.EFFECT_REGISTRY[EffectId.FILLER]
    try:
        registry.EFFECT_REGISTRY[EffectId.FILLER] = {"keywords": {"hexproof"}}
        state, bolt = _fresh_bolt_state()
        opp_hex = Permanent(CardDef("Hex Bear", CardType.CREATURE, None, EffectId.FILLER, power=1, toughness=1))
        mine_hex = Permanent(CardDef("Hex Bear", CardType.CREATURE, None, EffectId.FILLER, power=1, toughness=1))
        opp_hex.slot = mine_hex.slot = 1
        state.players[1].battlefield = [opp_hex]  # opponent's -- untargetable by me
        state.players[0].battlefield = [mine_hex]  # my own -- still targetable by me
        cast_lightning_bolt(state, bolt)
        creature_opts = resolution.choose_any_target_creature_options(state)
        assert (0, "Hex Bear", 1) in creature_opts  # my own hexproof creature is fair game to me
        assert (1, "Hex Bear", 1) not in creature_opts  # the opponent's is not
    finally:
        registry.EFFECT_REGISTRY[EffectId.FILLER] = original_filler


def _drain_pool_pay(state):
    """Walk begin_pay_cost's pool spend to completion."""
    guard = 0
    while state.pending_resolution is not None and state.pending_resolution["kind"] == "pay_cost":
        guard += 1
        assert guard < 12
        execute_pool_spend(state, pool_spend_options(state)[0])


def test_chain_lightning_copy_rider_pay_copy_retarget_decline():
    """Chain Lightning: 3 to any target, then the copy rider. Caster (0)
    targets opponent (1); opp pays {R}{R} to copy + retarget the caster
    (0); the copy deals 3 back and the caster declines to copy -- the chain
    stops."""
    state = GameState(on_the_play=True, players=[PlayerState(True), PlayerState(False)])
    state.active_idx = 0
    state.players[1].mana_pool = {"R": 2}  # opponent can afford the copy rider once
    chain = CardDef("Chain Lightning", CardType.SORCERY, {"R": 1}, EffectId.CHAIN_LIGHTNING)
    state.players[0].hand = [chain]
    cast_chain_lightning(state, chain)  # precast target choice
    resolution.execute_choose_any_target_player(state, 1)  # caster targets the opponent
    assert chain not in state.players[0].hand  # left hand at cast, now on the stack
    resolve_top_of_stack(state)
    assert state.players[1].life_total == 17 and any(c.name == chain.name for c in state.players[0].graveyard)  # 3 to opp, itself to gy
    # rider FIRST "may": the affected player (opp, idx 1) may pay {R}{R}
    assert state.pending_resolution["kind"] == "pay_unless" and state.active_idx == 1
    resolution.pay_unless_pay(state)
    _drain_pool_pay(state)  # opp taps {R}{R} from pool
    # rider SECOND, INDEPENDENT "may": having paid, opp may copy
    assert state.pending_resolution["kind"] == "may_copy" and state.active_idx == 1
    resolution.execute_may_copy(state, True)  # choose to copy
    # now the opponent chooses the copy's new target (their perspective)
    assert state.pending_resolution["kind"] == "choose_any_target" and state.active_idx == 1
    resolution.execute_choose_any_target_player(state, 0)  # retarget the original caster
    assert state.active_idx == 0  # active_idx restored to the resolving spell's controller
    assert len(state.stack) == 1 and state.stack[0]["controller"] == 1  # copy on stack, opp controls it
    resolve_top_of_stack(state)
    assert state.players[0].life_total == 17  # the copy dealt 3 back to the caster
    # the copy's rider: now the caster (idx 0) is the affected player; they decline to pay
    assert state.pending_resolution["kind"] == "pay_unless" and state.active_idx == 0
    resolution.pay_unless_decline(state)
    assert state.pending_resolution is None and state.stack == []  # chain stops, no card left behind (copy ceased)


def test_chain_lightning_pay_but_decline_copy():
    """The two "may"s are independent: paying {R}{R} but DECLINING to copy
    is a legal (if rarely-useful) line -- the mana is spent, no copy is
    made."""
    state = GameState(on_the_play=True, players=[PlayerState(True), PlayerState(False)])
    state.active_idx = 0
    state.players[1].mana_pool = {"R": 2}
    chain2 = CardDef("Chain Lightning", CardType.SORCERY, {"R": 1}, EffectId.CHAIN_LIGHTNING)
    state.players[0].hand = [chain2]
    cast_chain_lightning(state, chain2)
    resolution.execute_choose_any_target_player(state, 1)
    resolve_top_of_stack(state)
    assert state.players[1].life_total == 17
    resolution.pay_unless_pay(state)
    _drain_pool_pay(state)
    assert state.pending_resolution["kind"] == "may_copy" and state.active_idx == 1
    resolution.execute_may_copy(state, False)  # paid, but decline to copy
    assert state.pending_resolution is None and state.stack == []  # no copy made
    assert state.active_idx == 0  # restored to controller
    assert sum(state.players[1].mana_pool.values()) == 0  # the {R}{R} was still spent


def test_cleansing_wildfire_own_land():
    """Cleansing Wildfire: destroy target land (any). Its controller
    searches for a basic (tapped); the caster draws.

    (a) Own tapped land -> you search + fetch a basic tapped + draw."""
    state = GameState(on_the_play=True)
    dual = Permanent(CardDef("Twisted Landscape", CardType.LAND, None, EffectId.TWISTED_LANDSCAPE))
    dual.slot = 1
    dual.tapped = True
    state.battlefield = [dual]
    cw = CardDef("Cleansing Wildfire", CardType.SORCERY, {"generic": 1, "R": 1}, EffectId.CLEANSING_WILDFIRE)
    state.hand = [cw]
    state.library = [
        CardDef("Mountain", CardType.LAND, None, EffectId.MOUNTAIN, basic=True, subtypes=("Mountain",)),
        CardDef("Guttersnipe", CardType.CREATURE, {"generic": 2, "R": 1}, EffectId.GUTTERSNIPE, power=2, toughness=2),  # nonbasic -- ineligible
    ]
    cast_cleansing_wildfire(state, cw)
    assert state.pending_resolution["kind"] == "choose_any_target"
    assert (0, "Twisted Landscape", 1) in resolution.choose_any_target_creature_options(state)
    resolution.execute_choose_any_target_creature(state, 0, "Twisted Landscape", 1)
    resolve_top_of_stack(state)  # destroy + open the (may) search
    assert dual not in state.battlefield  # destroyed
    assert state.pending_resolution["kind"] == "search_fetch"
    assert resolution.search_fetch_options(state) == ["Mountain"]  # only the basic
    resolution.execute_search_fetch_option(state, "Mountain")
    fetched = [p for p in state.battlefield if p.card_def.name == "Mountain"]
    assert len(fetched) == 1 and fetched[0].tapped  # onto the battlefield tapped
    assert len(state.hand) == 1  # "Draw a card"


def test_cleansing_wildfire_opponent_land():
    """(b) TARGET AN OPPONENT'S LAND: it's destroyed, the OPPONENT (its
    controller) searches THEIR library and the basic enters THEIR
    battlefield tapped; the CASTER draws."""
    state = GameState(on_the_play=True, players=[PlayerState(True), PlayerState(False)])
    state.active_idx = 0
    opp_land = Permanent(CardDef("Opp Dual", CardType.LAND, None, EffectId.TWISTED_LANDSCAPE))
    opp_land.slot = 1
    state.players[1].battlefield = [opp_land]
    state.players[1].library = [CardDef("Forest", CardType.LAND, None, EffectId.FOREST, basic=True, subtypes=("Forest",))]
    cw2 = CardDef("Cleansing Wildfire", CardType.SORCERY, {"generic": 1, "R": 1}, EffectId.CLEANSING_WILDFIRE)
    state.players[0].hand = [cw2]
    state.players[0].library = [CardDef("CasterDraw", CardType.LAND, None, EffectId.MOUNTAIN, basic=True, subtypes=("Mountain",))]
    cast_cleansing_wildfire(state, cw2)
    resolution.execute_choose_any_target_creature(state, 1, "Opp Dual", 1)  # target the OPPONENT's land
    resolve_top_of_stack(state)  # destroy + open the search FOR THE OPPONENT
    assert opp_land not in state.players[1].battlefield  # destroyed
    assert state.pending_resolution["kind"] == "search_fetch" and state.active_idx == 1  # the land's controller searches
    resolution.execute_search_fetch_option(state, "Forest")
    opp_basics = [p for p in state.players[1].battlefield if p.card_def.name == "Forest"]
    assert len(opp_basics) == 1 and opp_basics[0].tapped  # onto the OPPONENT's battlefield, tapped
    assert state.active_idx == 0  # restored to the caster
    assert len(state.players[0].hand) == 1 and state.players[0].hand[0].name == "CasterDraw"  # CASTER drew


def test_reckless_impulse_exile_and_expiry():
    """Reckless Impulse: exile top 2 into the impulse zone (playable until
    the player's next turn), and untap prunes them once expired."""
    state = GameState(on_the_play=True)
    state.turn_number = 3
    ri = CardDef("Reckless Impulse", CardType.SORCERY, {"generic": 1, "R": 1}, EffectId.RECKLESS_IMPULSE)
    state.hand = [ri]
    state.library = [CardDef("A", CardType.LAND, None, EffectId.MOUNTAIN), CardDef("B", CardType.INSTANT, {"R": 1}, EffectId.LIGHTNING_BOLT), CardDef("C", CardType.LAND, None, EffectId.MOUNTAIN)]
    cast_reckless_impulse(state, ri)
    assert len(state.impulse) == 2 and all(u == 3 + len(state.players) for _cd, u in state.impulse)  # until next turn
    assert [c.name for c in state.graveyard] == ["Reckless Impulse"]
    state.turn_number = 3 + len(state.players) + 1  # past the deadline
    untap_step(state)
    assert state.impulse == []  # expired, pruned


def test_goblin_bushwhacker_kicked_team_pump():
    """Goblin Bushwhacker kicked: team +1/+0 and haste until EOT (incl.
    itself)."""
    state = GameState(on_the_play=True)
    ally = Permanent(CardDef("Ally", CardType.CREATURE, None, EffectId.FILLER, power=1, toughness=1))
    ally.slot = 1
    state.battlefield = [ally]
    bw = CardDef("Goblin Bushwhacker", CardType.CREATURE, {"R": 1}, EffectId.GOBLIN_BUSHWHACKER, power=1, toughness=1)
    state.hand = [bw]
    _goblin_bushwhacker_kicked(state, bw)
    promote_triggers_to_stack(state)
    while state.stack:
        resolve_top_of_stack(state)
    gob = next(p for p in state.battlefield if p.card_def.name == "Goblin Bushwhacker")
    assert permanent_power(state, gob) == 2 and has_keyword(state, gob, "haste")
    assert permanent_power(state, ally) == 2 and has_keyword(state, ally, "haste")


def test_goblin_bushwhacker_via_action_table_cost_override_mode():
    """Goblin Bushwhacker via the REAL action table (not calling
    _goblin_bushwhacker_kicked directly, unlike the test above): "Cast
    Goblin Bushwhacker" -> choose_cast_mode ("Mode 2" = kicked,
    registry-second, its own {R}{R} cost OVERRIDE) -> pay {R}{R} -> resolves
    as the kicked mode. Exercises the cost-override branch of drl_env.
    _modal_execute/_modal_legal (Winding Way/Utopia Sprawl's own modes never
    override cost; this card's kicked mode does)."""
    bw_dl = [("Goblin Bushwhacker", 4), ("Mountain", 8)]
    bw_byname = {a[0]: (a[1], a[2]) for a in drl_env.build_action_table(
        bw_dl, registry.EFFECT_REGISTRY, pending_kinds=registry.derive_pending_kinds(bw_dl))}
    bw_state = GameState(on_the_play=True)
    bw_state.phase = Phase.MAIN1
    bw_state.turn_player_idx = 0
    bw_state.active_idx = 0
    bw_state.hand = [registry.CARD_DEFS["Goblin Bushwhacker"]]
    bw_ally = Permanent(CardDef("Ally", CardType.CREATURE, None, EffectId.FILLER, power=1, toughness=1))
    bw_ally.slot = 1
    bw_state.battlefield = [bw_ally, Permanent(registry.CARD_DEFS["Mountain"]), Permanent(registry.CARD_DEFS["Mountain"])]
    for p in bw_state.battlefield:
        if p.card_def.name == "Mountain":
            activate_mana_source(bw_state, p)
    assert bw_state.mana_pool.get("R", 0) == 2
    cast_legal, cast_execute = bw_byname["Cast Goblin Bushwhacker"]
    assert cast_legal(bw_state)
    cast_execute(bw_state)
    assert bw_state.pending_resolution["kind"] == "choose_cast_mode"
    mode2_legal, mode2_execute = bw_byname["Mode 2"]  # registry order: unkicked (Mode 1), kicked (Mode 2)
    assert mode2_legal(bw_state)
    mode2_execute(bw_state)
    assert bw_state.pending_resolution["kind"] == "pay_cost"
    assert bw_state.pending_resolution["remaining"] == {"R": 2}, (
        "must owe the kicked mode's OWN {R}{R} cost override, not the card's base {R}"
    )
    guard = 0
    while bw_state.pending_resolution is not None:
        guard += 1
        assert guard < 30
        execute_pool_spend(bw_state, pool_spend_options(bw_state)[0])
    while bw_state.stack:  # resolve the spell itself first (creates the permanent, queues its ETB)
        resolve_top_of_stack(bw_state)
    promote_triggers_to_stack(bw_state)  # THEN promote the now-queued ETB trigger
    while bw_state.stack:
        resolve_top_of_stack(bw_state)
    bw_gob = next(p for p in bw_state.battlefield if p.card_def.name == "Goblin Bushwhacker")
    assert permanent_power(bw_state, bw_gob) == 2 and has_keyword(bw_state, bw_gob, "haste"), (
        "kicked mode's ETB team pump must have fired"
    )


def test_rally_at_the_hornburg_haste_only_humans():
    """Rally at the Hornburg: two 1/1 Human Soldiers; Humans you control
    (incl. a Human already out) gain haste."""
    state = GameState(on_the_play=True)
    human = Permanent(CardDef("Human Ally", CardType.CREATURE, {"R": 1}, EffectId.FILLER, power=1, toughness=1, subtypes=("Human",)))
    human.slot = 1
    goblin = Permanent(CardDef("Goblin Ally", CardType.CREATURE, {"R": 1}, EffectId.FILLER, power=1, toughness=1, subtypes=("Goblin",)))
    goblin.slot = 1
    state.battlefield = [human, goblin]
    rally = CardDef("Rally at the Hornburg", CardType.SORCERY, {"generic": 1, "R": 1}, EffectId.RALLY_AT_THE_HORNBURG)
    state.hand = [rally]
    cast_rally_at_the_hornburg(state, rally)
    assert sum(1 for p in state.battlefield if p.card_def.name == "Human Soldier") == 2
    assert has_keyword(state, human, "haste")  # Human gets haste
    assert not has_keyword(state, goblin, "haste")  # non-Human doesn't


def test_reckless_lackey_sac_draws_and_makes_treasure():
    """Reckless Lackey: first strike + haste; sac -> draw + Treasure."""
    state = GameState(on_the_play=True)
    lackey = Permanent(registry.CARD_DEFS["Reckless Lackey"])
    lackey.slot = 1
    state.battlefield = [lackey]
    state.library = [CardDef("Top", CardType.LAND, None, EffectId.MOUNTAIN)]
    assert has_keyword(state, lackey, "first_strike") and has_keyword(state, lackey, "haste")
    activate_reckless_lackey_sac(state, lackey)
    assert lackey not in state.battlefield and any(c.name == "Reckless Lackey" for c in state.graveyard)
    resolve_top_of_stack(state)
    assert len(state.hand) == 1 and any(p.card_def.name == "Treasure" for p in state.battlefield)


def test_goblin_tomb_raider_conditional_static():
    """Goblin Tomb Raider: +1/+0 and haste only while you control an
    artifact."""
    state = GameState(on_the_play=True)
    gtr = Permanent(registry.CARD_DEFS["Goblin Tomb Raider"])
    gtr.slot = 1
    state.battlefield = [gtr]
    assert permanent_power(state, gtr) == 1 and not has_keyword(state, gtr, "haste")
    state.battlefield.append(Permanent(registry.CARD_DEFS["Great Furnace"]))  # an artifact land
    assert permanent_power(state, gtr) == 2 and has_keyword(state, gtr, "haste")


def test_burning_tree_emissary_etb_adds_rg():
    """Burning-Tree Emissary: ETB adds {R}{G} to the pool."""
    state = GameState(on_the_play=True)
    bte = registry.CARD_DEFS["Burning-Tree Emissary"]
    state.hand = [bte]
    cast_permanent_from_hand(state, bte)
    promote_triggers_to_stack(state)
    while state.stack:
        resolve_top_of_stack(state)
    assert state.mana_pool.get("R") == 1 and state.mana_pool.get("G") == 1


def _drive_stack(state):
    promote_triggers_to_stack(state)
    while state.stack:
        resolve_top_of_stack(state)


def test_krark_clan_shaman_sac_artifact_sweeps_nonflyers():
    """Krark-Clan Shaman: Sacrifice an artifact -> 1 damage to each creature
    without flying (a real flyer is spared; itself dies as a 1/1). Uses real
    distinct cards: Gurmag Angler (nonflying) and Utrom Monitor (flying)."""
    state = GameState(on_the_play=True, players=[PlayerState(True), PlayerState(False)])
    state.active_idx = 0
    krark = Permanent(registry.CARD_DEFS["Krark-Clan Shaman"])
    krark.slot = 1
    art = Permanent(registry.CARD_DEFS["Great Furnace"])
    state.players[0].battlefield = [krark, art]
    ground = Permanent(registry.CARD_DEFS["Gurmag Angler"])  # 5/5 nonflying
    ground.slot = 1
    flyer = Permanent(registry.CARD_DEFS["Utrom Monitor"])  # 3/3 flying
    flyer.slot = 1
    state.players[1].battlefield = [ground, flyer]
    krark_clan_shaman_activate(state, krark)
    resolution.execute_sacrifice_option(state, "Great Furnace")
    _drive_stack(state)
    assert ground.damage_marked == 1 and flyer.damage_marked == 0  # ground hit, flyer spared
    assert krark not in state.players[0].battlefield  # 1/1 took 1 -> dead


def test_experimental_synthesizer_impulse_and_samurai():
    """Experimental Synthesizer: ETB impulse; {2}{R},Sac -> Samurai + LTB
    impulse."""
    state = GameState(on_the_play=True)
    state.turn_number = 1
    state.hand = [registry.CARD_DEFS["Experimental Synthesizer"]]
    state.library = [CardDef(f"t{i}", CardType.LAND, None, EffectId.MOUNTAIN, basic=True) for i in range(3)]
    cast_permanent_from_hand(state, registry.CARD_DEFS["Experimental Synthesizer"])
    _drive_stack(state)
    assert len(state.impulse) == 1  # ETB impulse
    es = next(p for p in state.battlefield if p.card_def.name == "Experimental Synthesizer")
    experimental_synthesizer_sac(state, es)
    _drive_stack(state)
    assert any(p.card_def.name == "Samurai" for p in state.battlefield) and len(state.impulse) == 2  # + LTB impulse


def test_clockwork_percussionist_dies_into_impulse():
    """Clockwork Percussionist: haste; dies -> impulse (until next turn)."""
    state = GameState(on_the_play=True)
    state.turn_number = 1
    state.library = [CardDef("q", CardType.LAND, None, EffectId.MOUNTAIN, basic=True)]
    cp = Permanent(registry.CARD_DEFS["Clockwork Percussionist"])
    cp.slot = 1
    state.battlefield = [cp]
    assert has_keyword(state, cp, "haste")
    sacrifice_to_graveyard(state, cp)
    _drive_stack(state)
    assert len(state.impulse) == 1


def test_galvanic_blast_no_metalcraft():
    """Galvanic Blast: 2 damage without Metalcraft (checked at resolution)."""
    state = GameState(on_the_play=True, players=[PlayerState(True), PlayerState(False)])
    state.active_idx = 0
    victim = Permanent(CardDef("Victim", CardType.CREATURE, None, EffectId.FILLER, power=1, toughness=9))
    victim.slot = 1
    state.players[1].battlefield = [victim]
    gb = registry.CARD_DEFS["Galvanic Blast"]
    state.players[0].hand = [gb]
    cast_galvanic_blast(state, gb)
    resolution.execute_choose_any_target_creature(state, 1, "Victim", 1)
    resolve_top_of_stack(state)
    assert victim.damage_marked == 2  # no metalcraft


def test_galvanic_blast_metalcraft_four_damage():
    """Galvanic Blast: 4 damage with Metalcraft (3+ artifacts controlled)."""
    state = GameState(on_the_play=True, players=[PlayerState(True), PlayerState(False)])
    state.active_idx = 0
    state.players[0].battlefield = [Permanent(registry.CARD_DEFS["Great Furnace"]) for _ in range(3)]  # metalcraft
    victim = Permanent(CardDef("Victim", CardType.CREATURE, None, EffectId.FILLER, power=1, toughness=9))
    victim.slot = 1
    state.players[1].battlefield = [victim]
    gb = registry.CARD_DEFS["Galvanic Blast"]
    state.players[0].hand = [gb]
    cast_galvanic_blast(state, gb)
    resolution.execute_choose_any_target_creature(state, 1, "Victim", 1)
    resolve_top_of_stack(state)
    assert victim.damage_marked == 4  # metalcraft -> 4
