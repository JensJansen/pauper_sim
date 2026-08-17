"""Tests for game.catalog.red_cards. See the module under test for the
card-implementation rationale (real-rules citations, etc.) each test below
guards."""

import contextlib
import io

import drl_env
from game import registry, resolution
from game.cards import CardDef, CardType, EffectId
from game.catalog.red_cards import (
    _fireblast_alt_extra_legal,
    _goblin_bushwhacker_kicked,
    _grab_the_prize_extra_legal,
    _makeshift_munitions_legal,
    activate_melded_moxite_sac,
    activate_reckless_lackey_sac,
    cast_breath_weapon,
    cast_chain_lightning,
    cast_cleansing_wildfire,
    cast_end_the_festivities,
    cast_faithless_looting,
    cast_fiery_temper,
    cast_fireblast,
    cast_fireblast_alt,
    cast_galvanic_blast,
    cast_grab_the_prize,
    cast_highway_robbery,
    cast_highway_robbery_from_exile,
    cast_lava_dart,
    cast_lightning_bolt,
    cast_rally_at_the_hornburg,
    cast_reckless_impulse,
    experimental_synthesizer_sac,
    flashback_faithless_looting,
    flashback_lava_dart,
    krark_clan_shaman_activate,
    makeshift_munitions_activate,
    melded_moxite_etb,
    voldaren_epicure_etb,
)
from game.effects.casting import cast_permanent_from_hand
from game.effects.madness_and_plot import execute_madness_cast, plot_to_exile
from game.effects.stack import on_cast_trigger, resolve_top_of_stack
from game.effects.state_based import sacrifice_to_graveyard
from game.effects.stats import has_keyword, permanent_power
from game.effects.triggers import promote_triggers_to_stack
from game.mana import activate_mana_source, begin_pay_cost, execute_pool_spend, pool_spend_options
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


def test_breath_weapon_spares_dragons_avenging_hunter_regression():
    """REGRESSION: cast_breath_weapon used to be a purely symmetric "2
    damage to each creature" wipe, on the (now-false) assumption that no
    Dragon exists in this card pool. Green's Avenging Hunter
    (registry.CARD_DEFS["Avenging Hunter"], subtypes=("Dragon","Ranger"))
    IS a Dragon, so the real card's non-Dragon filter must actually bite:
    the Dragon takes ZERO damage while an ordinary non-Dragon creature
    still takes the full 2 -- same symmetric-wipe shape otherwise (either
    battlefield). Sanity check that this actually distinguishes old vs new
    behavior: under the old (purely symmetric) code the Dragon would have
    taken 2 damage too, same as the non-Dragon; here it must take 0."""
    state = GameState(on_the_play=True, players=[PlayerState(True), PlayerState(False)])
    breath_weapon = CardDef("Breath Weapon", CardType.INSTANT, {"generic": 2, "R": 1}, EffectId.BREATH_WEAPON)
    state.hand = [breath_weapon]
    dragon = Permanent(registry.CARD_DEFS["Avenging Hunter"])  # a real Dragon in this pool
    dragon.slot = 1
    non_dragon = Permanent(CardDef("Grizzly Bears", CardType.CREATURE, None, EffectId.FILLER, power=2, toughness=4))
    non_dragon.slot = 1
    state.players[1].battlefield = [dragon, non_dragon]

    cast_breath_weapon(state, breath_weapon)
    assert dragon in state.players[1].battlefield and dragon.damage_marked == 0  # Dragon filter is LIVE, not vacuous
    assert dragon.damage_marked != 2, "old symmetric behavior would have marked 2 damage on the Dragon"
    assert non_dragon.damage_marked == 2  # a real non-Dragon on the same battlefield still takes the full 2


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
    resolution.execute_discard_or_sacrifice_discard(state, "Fiery Temper")
    assert len(state.hand) == 2  # drew 2
    assert state.exile and state.exile[0][0].name == "Fiery Temper"  # exiled, not graveyarded -- Madness
    assert state.trigger_queue and state.trigger_queue[0]["kind"] == "madness"


def test_highway_robbery_sacrifice_land_path():
    """Sacrifice-a-land path -- the alternative cost to discarding a card.
    Picking "sacrifice" opens a real choose_permanent sub-decision for
    WHICH land pays it (not an arbitrary first-same-name match -- see
    execute_discard_or_sacrifice_trigger_sacrifice's own docstring).
    Also regression coverage: the sacrifice path fires
    fire_sacrifice_triggers too (Gixian Infiltrator used to never see a
    sacrifice paid this way)."""
    state2 = GameState(on_the_play=True)
    hr2 = CardDef("Highway Robbery", CardType.SORCERY, {"generic": 1, "R": 1}, EffectId.HIGHWAY_ROBBERY)
    mountain = Permanent(CardDef("Mountain", CardType.LAND, None, EffectId.MOUNTAIN))
    gix = Permanent(registry.CARD_DEFS["Gixian Infiltrator"])
    gix.slot = 1
    state2.hand = [hr2]
    state2.battlefield = [mountain, gix]
    state2.library = [CardDef(f"Filler {i}", CardType.LAND, None, EffectId.MOUNTAIN) for i in range(2)]
    cast_highway_robbery(state2, hr2)
    resolution.execute_discard_or_sacrifice_trigger_sacrifice(state2)
    assert state2.pending_resolution["kind"] == "choose_permanent"
    resolution.execute_choose_permanent_option(state2, "Mountain", mountain.slot)
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


def test_chain_lightning_rider_decline_no_copy():
    """The rider's FIRST "may": the affected player DECLINES to pay {R}{R}
    at all (whether because they can't afford it or simply choose not to --
    both existing Chain Lightning tests above always float exactly {R}{R}
    and always choose to pay). No copy, no retarget -- the chain just
    stops, and no mana is spent."""
    state = GameState(on_the_play=True, players=[PlayerState(True), PlayerState(False)])
    state.active_idx = 0
    # Opponent has NO mana floated at all -- can't afford (and here, declines) {R}{R}.
    chain = CardDef("Chain Lightning", CardType.SORCERY, {"R": 1}, EffectId.CHAIN_LIGHTNING)
    state.players[0].hand = [chain]
    cast_chain_lightning(state, chain)
    resolution.execute_choose_any_target_player(state, 1)
    resolve_top_of_stack(state)
    assert state.players[1].life_total == 17  # the original 3 damage still happened
    assert state.pending_resolution["kind"] == "pay_unless" and state.active_idx == 1
    resolution.pay_unless_decline(state)
    assert state.pending_resolution is None and state.stack == []  # no copy made, chain stops here
    assert state.active_idx == 0  # restored to the resolving spell's controller
    assert sum(state.players[1].mana_pool.values()) == 0  # nothing was ever spent


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
    bw_byname = {a[0]: (a[1], a[2]) for a in drl_env.build_action_table(bw_dl, registry.EFFECT_REGISTRY)}
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


def test_burning_tree_emissary_etb_adds_rr():
    """Burning-Tree Emissary: ETB adds {R}{R} to the pool (authorized simplification)."""
    state = GameState(on_the_play=True)
    bte = registry.CARD_DEFS["Burning-Tree Emissary"]
    state.hand = [bte]
    cast_permanent_from_hand(state, bte)
    promote_triggers_to_stack(state)
    while state.stack:
        resolve_top_of_stack(state)
    assert state.mana_pool.get("R") == 2
    assert state.mana_pool_single_pip == {}  # a 2-symbol event -- never single-pip-tagged


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
    assert state.pending_resolution["kind"] == "choose_permanent"
    resolution.execute_choose_permanent_option(state, "Great Furnace", art.slot)
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
    samurai = next(p for p in state.battlefield if p.card_def.name == "Samurai")
    assert has_keyword(state, samurai, "vigilance")  # 2/2 white Samurai WITH vigilance (real card text)


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


def test_voldaren_epicure_etb_damages_opponent_and_creates_blood():
    """Voldaren Epicure ETB (voldaren_epicure_etb): "1 damage to each
    opponent. Create a Blood token." Only ever appeared as inert battlefield
    filler elsewhere -- this actually resolves the ETB."""
    state = GameState(on_the_play=True, players=[PlayerState(True), PlayerState(False)])
    state.active_idx = 0
    voldaren_epicure_etb(state)
    assert state.players[1].life_total == 19  # 1 damage to the opponent
    assert any(p.card_def.name == "Blood" for p in state.players[0].battlefield)  # Blood token created


def test_fiery_temper_baseline_cast_hits_any_target():
    """Fiery Temper's baseline {1}{R}{R} cast (cast_fiery_temper): 3 damage
    to any target -- never invoked by any other test (only its Madness
    mode and its own discard-bait role are exercised elsewhere)."""
    state = GameState(on_the_play=True, players=[PlayerState(True), PlayerState(False)])
    ft = CardDef("Fiery Temper", CardType.INSTANT, {"generic": 1, "R": 2}, EffectId.FIERY_TEMPER)
    state.players[0].hand = [ft]
    cast_fiery_temper(state, ft)
    resolution.execute_choose_any_target_player(state, 1)
    assert ft not in state.players[0].hand
    resolve_top_of_stack(state)
    assert state.players[1].life_total == 17 and any(c.name == "Fiery Temper" for c in state.players[0].graveyard)


def test_fiery_temper_madness_cast_via_real_discard_chain():
    """Fiery Temper's Madness {R} (madness_fiery_temper), driven through a
    REAL discard trigger end to end -- unlike every other place in this
    suite, which only confirms the madness trigger QUEUES. Faithless
    Looting's own mandatory discard-2 discards Fiery Temper (Madness
    replacement: hand -> exile, not graveyard); casting it for {R} deals 3
    to the chosen target and the card ends up in the graveyard, never
    having touched hand again."""
    state = GameState(on_the_play=True, players=[PlayerState(True), PlayerState(False)])
    state.active_idx = 0
    fl = registry.CARD_DEFS["Faithless Looting"]
    ft = registry.CARD_DEFS["Fiery Temper"]
    state.players[0].hand = [fl, ft]
    state.players[0].library = [CardDef(f"D{i}", CardType.LAND, None, EffectId.MOUNTAIN) for i in range(2)]
    mountain = Permanent(registry.CARD_DEFS["Mountain"])
    state.players[0].battlefield = [mountain]

    cast_faithless_looting(state, fl)  # draw 2, mandatory discard 2
    assert len(state.players[0].hand) == 3  # Fiery Temper + 2 drawn
    resolution.execute_discard_option(state, "Fiery Temper")  # Madness replacement: -> exile, not graveyard
    resolution.execute_discard_option(state, "D0")  # the second mandatory discard, an ordinary card
    assert ft not in state.players[0].hand
    assert any(cd.name == "Fiery Temper" for cd, _stamp in state.exile)
    assert state.trigger_queue and state.trigger_queue[0]["kind"] == "madness"

    promote_triggers_to_stack(state)
    resolve_top_of_stack(state)  # opens the cast-or-graveyard decision
    assert state.pending_resolution["kind"] == "madness_decision"

    activate_mana_source(state, mountain)  # pre-float (still valid, just no longer required): float {R} BEFORE casting
    execute_madness_cast(state)
    assert state.pending_resolution["kind"] == "pay_cost"
    guard = 0
    while state.pending_resolution is not None and state.pending_resolution["kind"] == "pay_cost":
        guard += 1
        assert guard < 10
        execute_pool_spend(state, pool_spend_options(state)[0])
    assert state.pending_resolution["kind"] == "choose_any_target"  # precast_choice -- target locked before it resolves
    resolution.execute_choose_any_target_player(state, 1)
    resolve_top_of_stack(state)
    assert state.players[1].life_total == 17  # 3 damage
    assert any(c.name == "Fiery Temper" for c in state.players[0].graveyard)  # ends up in the graveyard
    assert not any(cd.name == "Fiery Temper" for cd, _stamp in state.exile)  # left exile


def test_faithless_looting_cast_draws_two_discards_two():
    """Faithless Looting's baseline {R} cast (cast_faithless_looting):
    draw 2, then mandatory discard 2 -- never invoked by any other test."""
    state = GameState(on_the_play=True)
    fl = CardDef("Faithless Looting", CardType.SORCERY, {"R": 1}, EffectId.FAITHLESS_LOOTING)
    spare_a = CardDef("Spare A", CardType.LAND, None, EffectId.MOUNTAIN)
    spare_b = CardDef("Spare B", CardType.LAND, None, EffectId.MOUNTAIN)
    state.hand = [fl, spare_a, spare_b]
    state.library = [CardDef(f"Draw{i}", CardType.LAND, None, EffectId.MOUNTAIN) for i in range(2)]

    cast_faithless_looting(state, fl)
    assert fl not in state.hand and any(c.name == "Faithless Looting" for c in state.graveyard)
    assert len(state.hand) == 4  # Spare A/B + 2 drawn
    resolution.execute_discard_option(state, "Spare A")
    resolution.execute_discard_option(state, "Spare B")
    assert state.pending_resolution is None
    assert len(state.hand) == 2 and all(c.name.startswith("Draw") for c in state.hand)
    assert sorted(c.name for c in state.graveyard) == ["Faithless Looting", "Spare A", "Spare B"]


def test_faithless_looting_flashback_effect_resolves_and_exiles():
    """flashback_faithless_looting's own EFFECT (draw 2/discard 2) actually
    executing, and the flashed-back card ending up EXILED, never back in
    the graveyard (real Flashback, 702.34). Distinct from
    test_faithless_looting_flashback_requires_mana
    (tests/drl_env/test_action_table.py), which only exercises the {2}{R}
    mana gate and never resolves the effect."""
    state = GameState(on_the_play=True)
    fl_inst = state.new_instance(registry.CARD_DEFS["Faithless Looting"])
    state.graveyard = [fl_inst]
    state.hand = [CardDef("Spare A", CardType.LAND, None, EffectId.MOUNTAIN), CardDef("Spare B", CardType.LAND, None, EffectId.MOUNTAIN)]
    state.library = [CardDef(f"Draw{i}", CardType.LAND, None, EffectId.MOUNTAIN) for i in range(2)]

    flashback_faithless_looting(state, fl_inst)
    assert fl_inst not in state.graveyard  # left the graveyard the moment Flashback was chosen
    assert len(state.stack) == 1
    resolve_top_of_stack(state)
    assert len(state.hand) == 4  # Spare A/B + 2 drawn
    resolution.execute_discard_option(state, "Spare A")
    resolution.execute_discard_option(state, "Spare B")
    assert len(state.hand) == 2 and all(c.name.startswith("Draw") for c in state.hand)
    assert not any(c.name == "Faithless Looting" for c in state.graveyard)  # never graveyarded -- exiled instead


def test_highway_robbery_plot_cost_and_cast_from_exile():
    """Highway Robbery's OWN Plot {1}{R} (its registry "plot" spec's own
    cost) and cast_highway_robbery_from_exile -- exercised with the REAL
    card, not the generic FILLER double
    test_plot_to_exile_moves_hand_card_with_turn_stamp already covers
    (tests/game/effects/test_madness_and_plot.py). cast_highway_robbery_
    from_exile reruns the SAME discard-or-sacrifice may as a normal cast,
    moving the card exile -> graveyard, never touching hand again."""
    plot_spec = registry.EFFECT_REGISTRY[EffectId.HIGHWAY_ROBBERY]["plot"]
    assert plot_spec["cost"] == {"generic": 1, "R": 1}  # real Plot cost

    state = GameState(on_the_play=True)
    hr = registry.CARD_DEFS["Highway Robbery"]
    spare = CardDef("Spare Card", CardType.INSTANT, {"R": 1}, EffectId.FILLER)
    state.hand = [hr, spare]
    plot_to_exile(state, hr)
    assert state.hand == [spare]
    assert state.exile == [(hr, state.turn_number)]

    state.library = [CardDef(f"Filler {i}", CardType.LAND, None, EffectId.MOUNTAIN) for i in range(2)]
    cast_highway_robbery_from_exile(state, hr)
    assert any(c.name == "Highway Robbery" for c in state.graveyard)  # exile -> graveyard, never hand
    assert hr not in state.hand
    assert state.pending_resolution["kind"] == "discard_or_sacrifice"
    resolution.execute_discard_or_sacrifice_discard(state, "Spare Card")
    assert any(c.name == "Spare Card" for c in state.graveyard)
    assert len(state.hand) == 2 and all(c.name.startswith("Filler") for c in state.hand)  # drew 2


def test_grab_the_prize_extra_legal_requires_second_hand_card():
    """Grab the Prize's discard is a real additional cost (_grab_the_prize_
    extra_legal): needs a card in hand BESIDES the one being cast -- with
    nothing else in hand, casting it is illegal."""
    state = GameState(on_the_play=True)
    gtp = registry.CARD_DEFS["Grab the Prize"]
    state.hand = [gtp]
    assert not _grab_the_prize_extra_legal(state)
    state.hand = [gtp, CardDef("Spare", CardType.LAND, None, EffectId.MOUNTAIN)]
    assert _grab_the_prize_extra_legal(state)


def test_grab_the_prize_discard_nonland_deals_damage():
    """cast_grab_the_prize / _grab_the_prize_effect: draw 2; if the
    discarded card wasn't a land, 2 damage to each opponent."""
    state = GameState(on_the_play=True, players=[PlayerState(True), PlayerState(False)])
    state.active_idx = 0
    gtp = registry.CARD_DEFS["Grab the Prize"]
    nonland = CardDef("Spare Bolt", CardType.INSTANT, {"R": 1}, EffectId.FILLER)
    state.players[0].hand = [gtp, nonland]
    state.players[0].library = [CardDef(f"D{i}", CardType.LAND, None, EffectId.MOUNTAIN) for i in range(2)]

    cast_grab_the_prize(state, gtp)
    assert state.pending_resolution["kind"] == "discard"
    resolution.execute_discard_option(state, "Spare Bolt")
    assert len(state.players[0].hand) == 2  # drew 2
    assert state.players[1].life_total == 18  # 2 damage -- discarded card wasn't a land
    assert any(c.name == "Grab the Prize" for c in state.players[0].graveyard)


def test_grab_the_prize_discard_land_no_damage():
    """(b) discarding a LAND: still draw 2, but no damage."""
    state = GameState(on_the_play=True, players=[PlayerState(True), PlayerState(False)])
    state.active_idx = 0
    gtp = registry.CARD_DEFS["Grab the Prize"]
    land = CardDef("Spare Land", CardType.LAND, None, EffectId.MOUNTAIN)
    state.players[0].hand = [gtp, land]
    state.players[0].library = [CardDef(f"D{i}", CardType.LAND, None, EffectId.MOUNTAIN) for i in range(2)]

    cast_grab_the_prize(state, gtp)
    resolution.execute_discard_option(state, "Spare Land")
    assert len(state.players[0].hand) == 2  # drew 2
    assert state.players[1].life_total == 20  # discarded card WAS a land -- no damage


def test_melded_moxite_etb_discard_draws_two():
    """melded_moxite_etb: "you may discard a card. If you do, draw two
    cards." (a) discarding -- draws 2."""
    state = GameState(on_the_play=True)
    spare = CardDef("Spare", CardType.LAND, None, EffectId.MOUNTAIN)
    state.hand = [spare]
    state.library = [CardDef(f"D{i}", CardType.LAND, None, EffectId.MOUNTAIN) for i in range(2)]
    melded_moxite_etb(state)
    assert state.pending_resolution["kind"] == "discard" and state.pending_resolution["optional"] is True
    resolution.execute_discard_option(state, "Spare")
    assert len(state.hand) == 2  # drew 2 after discarding


def test_melded_moxite_etb_decline_no_draw():
    """(b) declining outright -- genuinely optional, no draw."""
    state = GameState(on_the_play=True)
    spare = CardDef("Spare", CardType.LAND, None, EffectId.MOUNTAIN)
    state.hand = [spare]
    melded_moxite_etb(state)
    resolution.execute_discard_decline(state)
    assert state.hand == [spare]  # untouched, no draw
    assert state.pending_resolution is None


def test_melded_moxite_sac_pays_three_and_creates_tapped_robot():
    """Melded Moxite's {3} sac ability (activate_melded_moxite_sac): its
    own {3} mana-cost gating and the tapped-Robot-token effect, neither of
    which the unrelated Gixian-sacrifice-trigger regression elsewhere
    exercises (that one only confirms the sacrifice fires the OTHER
    creature's trigger, never Melded Moxite's own cost/effect)."""
    state = GameState(on_the_play=True)
    mm = Permanent(registry.CARD_DEFS["Melded Moxite"])
    mm.slot = 1
    mountains = [Permanent(registry.CARD_DEFS["Mountain"]) for _ in range(3)]
    state.battlefield = [mm] + mountains
    for m in mountains:
        activate_mana_source(state, m)
    assert state.mana_pool.get("R", 0) == 3

    begin_pay_cost(state, mm.card_def.extra["sac_ability_cost"], on_complete=lambda s: activate_melded_moxite_sac(s, mm))
    assert state.pending_resolution["kind"] == "pay_cost" and state.pending_resolution["remaining"] == {"generic": 3}
    guard = 0
    while state.pending_resolution is not None:
        guard += 1
        assert guard < 10
        execute_pool_spend(state, pool_spend_options(state)[0])
    assert mm not in state.battlefield and any(c.name == "Melded Moxite" for c in state.graveyard)
    resolve_top_of_stack(state)
    robot = next(p for p in state.battlefield if p.card_def.name == "Robot")
    assert robot.tapped is True and permanent_power(state, robot) == 2  # tapped 2/2 Robot


def test_fireblast_hard_cast_deals_four():
    """Fireblast's baseline {4}{R}{R} cast (cast_fireblast): 4 damage to
    any target -- never invoked by any other test."""
    state = GameState(on_the_play=True, players=[PlayerState(True), PlayerState(False)])
    fb = CardDef("Fireblast", CardType.INSTANT, {"generic": 4, "R": 2}, EffectId.FIREBLAST)
    state.players[0].hand = [fb]
    cast_fireblast(state, fb)
    resolution.execute_choose_any_target_player(state, 1)
    resolve_top_of_stack(state)
    assert state.players[1].life_total == 16  # 20 - 4
    assert any(c.name == "Fireblast" for c in state.players[0].graveyard)


def test_fireblast_alt_cost_sacrifices_two_mountains_no_mana():
    """Fireblast's alt cost (cast_fireblast_alt): sacrifice 2 Mountains
    instead of paying mana -- same 4 damage to any target."""
    state = GameState(on_the_play=True, players=[PlayerState(True), PlayerState(False)])
    fb = registry.CARD_DEFS["Fireblast"]
    mountains = [Permanent(registry.CARD_DEFS["Mountain"]) for _ in range(2)]
    mountains[0].slot = 1
    mountains[1].slot = 2
    state.players[0].hand = [fb]
    state.players[0].battlefield = mountains
    assert _fireblast_alt_extra_legal(state)

    cast_fireblast_alt(state, fb)
    assert fb not in state.players[0].hand and any(c.name == "Fireblast" for c in state.players[0].graveyard)
    assert state.pending_resolution["kind"] == "choose_permanent"
    resolution.execute_choose_permanent_option(state, "Mountain", 1)
    assert state.pending_resolution["kind"] == "choose_permanent"
    resolution.execute_choose_permanent_option(state, "Mountain", 2)
    assert state.players[0].battlefield == []  # both Mountains sacrificed
    assert state.pending_resolution["kind"] == "choose_any_target"
    resolution.execute_choose_any_target_player(state, 1)
    resolve_top_of_stack(state)
    assert state.players[1].life_total == 16  # 4 damage, no mana ever paid
    assert sum(state.players[0].mana_pool.values()) == 0


def test_fireblast_alt_cost_discounts_tag_from_a_tapped_mountain():
    """Tapping one of the two sacrificed Mountains for {R} first (correct
    play -- free value, since it's being sacrificed either way) tags an R
    pip; sacrificing both Mountains via the alt cost discounts that tag by
    exactly 1 (mana.discount_departing_source), not 2 -- only one Mountain
    actually produced tagged mana, the other never got a chance to activate
    before its sacrifice-time discount already found nothing left to take."""
    state = GameState(on_the_play=True, players=[PlayerState(True), PlayerState(False)])
    fb = registry.CARD_DEFS["Fireblast"]
    mountains = [Permanent(registry.CARD_DEFS["Mountain"]) for _ in range(2)]
    mountains[0].slot = 1
    mountains[1].slot = 2
    state.players[0].hand = [fb]
    state.players[0].battlefield = mountains
    activate_mana_source(state, mountains[0])  # float+tag {R} before sacrificing it
    assert state.players[0].mana_pool_single_pip == {"R": 1}

    cast_fireblast_alt(state, fb)
    resolution.execute_choose_permanent_option(state, "Mountain", 1)
    resolution.execute_choose_permanent_option(state, "Mountain", 2)
    assert state.players[0].mana_pool_single_pip == {}  # the one tagged R pip is excused
    assert state.players[0].mana_pool == {"R": 1}  # the floated mana itself is still there, unspent


def test_fireblast_alt_illegal_with_fewer_than_two_mountains():
    """_fireblast_alt_extra_legal: fewer than 2 Mountains controlled ->
    illegal to even attempt the alt cost."""
    state = GameState(on_the_play=True)
    state.battlefield = [Permanent(registry.CARD_DEFS["Mountain"])]  # only 1
    assert not _fireblast_alt_extra_legal(state)
    state.battlefield = []
    assert not _fireblast_alt_extra_legal(state)


def test_guttersnipe_on_cast_deals_two_to_opponent():
    """Guttersnipe's on_cast trigger (guttersnipe_on_cast): "Whenever you
    cast an instant or sorcery spell, this creature deals 2 damage to each
    opponent" -- fires via the generic on_cast_trigger chokepoint every
    real cast path calls once a spell's cost is paid. Only ever appeared as
    inert filler elsewhere; this actually casts an instant with Guttersnipe
    on the battlefield."""
    state = GameState(on_the_play=True, players=[PlayerState(True), PlayerState(False)])
    state.active_idx = 0
    guttersnipe = Permanent(registry.CARD_DEFS["Guttersnipe"])
    guttersnipe.slot = 1
    state.players[0].battlefield = [guttersnipe]
    bolt = registry.CARD_DEFS["Lightning Bolt"]
    state.players[0].hand = [bolt]

    on_cast_trigger(state, bolt)  # what every real cast path calls once the cost is paid
    cast_lightning_bolt(state, bolt)
    resolution.execute_choose_any_target_player(state, 1)
    _drive_stack(state)
    assert state.players[1].life_total == 20 - 3 - 2  # Bolt's 3 + Guttersnipe's 2


def test_lava_dart_cast_deals_one():
    """Lava Dart's baseline {R} cast (cast_lava_dart): 1 damage to any
    target -- never invoked by any other test."""
    state = GameState(on_the_play=True, players=[PlayerState(True), PlayerState(False)])
    ld = CardDef("Lava Dart", CardType.INSTANT, {"R": 1}, EffectId.LAVA_DART)
    state.players[0].hand = [ld]
    cast_lava_dart(state, ld)
    resolution.execute_choose_any_target_player(state, 1)
    resolve_top_of_stack(state)
    assert state.players[1].life_total == 19
    assert any(c.name == "Lava Dart" for c in state.players[0].graveyard)


def test_lava_dart_flashback_sacrifices_mountain_no_mana():
    """Lava Dart's Flashback (flashback_lava_dart): sacrifice a Mountain,
    no mana at all -- same 1 damage to any target."""
    state = GameState(on_the_play=True, players=[PlayerState(True), PlayerState(False)])
    ld_inst = state.new_instance(registry.CARD_DEFS["Lava Dart"])
    state.players[0].graveyard = [ld_inst]
    mountain = Permanent(registry.CARD_DEFS["Mountain"])
    mountain.slot = 1
    state.players[0].battlefield = [mountain]

    flashback_lava_dart(state, ld_inst)
    assert ld_inst not in state.players[0].graveyard  # left the graveyard the moment Flashback was chosen
    assert state.pending_resolution["kind"] == "choose_permanent"
    resolution.execute_choose_permanent_option(state, "Mountain", 1)
    assert mountain not in state.players[0].battlefield
    assert state.pending_resolution["kind"] == "choose_any_target"
    resolution.execute_choose_any_target_player(state, 1)
    resolve_top_of_stack(state)
    assert state.players[1].life_total == 19  # 1 damage, no mana spent
    assert sum(state.players[0].mana_pool.values()) == 0


def test_goblin_bushwhacker_unkicked_via_action_table_only_charges_r():
    """Goblin Bushwhacker's UNKICKED baseline {R} mode -- both existing
    Bushwhacker tests only exercise the KICKED mode. "Mode 1" charges just
    the card's own {R} (no cost override, unlike kicked's {R}{R} override),
    and goblin_bushwhacker_etb correctly no-ops -- no team pump -- when not
    kicked."""
    bw_dl = [("Goblin Bushwhacker", 4), ("Mountain", 8)]
    bw_byname = {a[0]: (a[1], a[2]) for a in drl_env.build_action_table(bw_dl, registry.EFFECT_REGISTRY)}
    state = GameState(on_the_play=True)
    state.phase = Phase.MAIN1
    state.turn_player_idx = 0
    state.active_idx = 0
    state.hand = [registry.CARD_DEFS["Goblin Bushwhacker"]]
    ally = Permanent(CardDef("Ally", CardType.CREATURE, None, EffectId.FILLER, power=1, toughness=1))
    ally.slot = 1
    state.battlefield = [ally, Permanent(registry.CARD_DEFS["Mountain"])]
    activate_mana_source(state, state.battlefield[1])
    assert state.mana_pool.get("R", 0) == 1
    cast_legal, cast_execute = bw_byname["Cast Goblin Bushwhacker"]
    assert cast_legal(state)
    cast_execute(state)
    assert state.pending_resolution["kind"] == "choose_cast_mode"
    mode1_legal, mode1_execute = bw_byname["Mode 1"]  # registry order: unkicked (Mode 1), kicked (Mode 2)
    assert mode1_legal(state)
    mode1_execute(state)
    assert state.pending_resolution["kind"] == "pay_cost"
    assert state.pending_resolution["remaining"] == {"R": 1}, (
        "unkicked mode must charge only the card's own base {R}, no override"
    )
    guard = 0
    while state.pending_resolution is not None:
        guard += 1
        assert guard < 30
        execute_pool_spend(state, pool_spend_options(state)[0])
    while state.stack:
        resolve_top_of_stack(state)
    promote_triggers_to_stack(state)
    while state.stack:
        resolve_top_of_stack(state)
    gob = next(p for p in state.battlefield if p.card_def.name == "Goblin Bushwhacker")
    assert permanent_power(state, gob) == 1 and not has_keyword(state, gob, "haste"), (
        "unkicked -- goblin_bushwhacker_etb must no-op, no team pump"
    )
    assert permanent_power(state, ally) == 1 and not has_keyword(state, ally, "haste")


def test_reckless_lackey_sac_via_action_table_pays_generic_and_r():
    """Reckless Lackey's {2}{R} sac ability's OWN sac_ability_cost
    mana-gating dispatch, driven through the REAL action table (mirrors
    test_goblin_bushwhacker_via_action_table_cost_override_mode's style).
    Its resolve effect (draw + Treasure) IS already tested directly
    (test_reckless_lackey_sac_draws_and_makes_treasure) and via the
    Gixian-trigger regression -- this only exercises the {2}{R} payment
    path itself."""
    rl_dl = [("Reckless Lackey", 1), ("Mountain", 8)]
    rl_byname = {a[0]: (a[1], a[2]) for a in drl_env.build_action_table(rl_dl, registry.EFFECT_REGISTRY)}
    state = GameState(on_the_play=True)
    state.phase = Phase.MAIN1
    state.turn_player_idx = 0
    state.active_idx = 0
    lackey = Permanent(registry.CARD_DEFS["Reckless Lackey"])
    lackey.slot = 1
    mountains = [Permanent(registry.CARD_DEFS["Mountain"]) for _ in range(3)]
    state.battlefield = [lackey] + mountains
    state.library = [CardDef("Top", CardType.LAND, None, EffectId.MOUNTAIN)]
    for m in mountains:
        activate_mana_source(state, m)
    assert state.mana_pool.get("R", 0) == 3
    legal, execute = rl_byname["Activate Reckless Lackey (sac)"]
    assert legal(state)
    execute(state)
    assert state.pending_resolution["kind"] == "pay_cost"
    assert state.pending_resolution["remaining"] == {"generic": 2, "R": 1}
    guard = 0
    while state.pending_resolution is not None:
        guard += 1
        assert guard < 10
        execute_pool_spend(state, pool_spend_options(state)[0])
    assert lackey not in state.battlefield and any(c.name == "Reckless Lackey" for c in state.graveyard)
    resolve_top_of_stack(state)
    assert len(state.hand) == 1 and any(p.card_def.name == "Treasure" for p in state.battlefield)


def test_krark_clan_shaman_illegal_with_no_artifact():
    """Krark-Clan Shaman's sac-artifact ability's own "legal" gate
    (_controls_artifact, via its registry "sweep" spec): zero artifacts
    controlled -> illegal to even activate. The resolve effect itself IS
    already well tested via direct calls with an artifact present
    (test_krark_clan_shaman_sac_artifact_sweeps_nonflyers)."""
    state = GameState(on_the_play=True)
    krark = Permanent(registry.CARD_DEFS["Krark-Clan Shaman"])
    krark.slot = 1
    state.battlefield = [krark]  # no artifact anywhere
    ability = registry.EFFECT_REGISTRY[EffectId.KRARK_CLAN_SHAMAN]["activated_abilities"]["sweep"]
    assert ability["legal"](state, krark) is False


def test_makeshift_munitions_activate_pays_one_sacrifices_and_deals_damage():
    """Makeshift Munitions' {1} + sacrifice-an-artifact-or-creature
    activated ability (makeshift_munitions_activate/_makeshift_munitions_
    legal): pay {1}, then sacrifice (a real per-instance choice), then 1
    damage to a chosen target -- completely untested elsewhere."""
    state = GameState(on_the_play=True, players=[PlayerState(True), PlayerState(False)])
    state.active_idx = 0
    mm = Permanent(registry.CARD_DEFS["Makeshift Munitions"])
    creature = Permanent(CardDef("Fodder", CardType.CREATURE, None, EffectId.FILLER, power=1, toughness=1))
    creature.slot = 1
    mountain = Permanent(registry.CARD_DEFS["Mountain"])
    state.players[0].battlefield = [mm, creature, mountain]
    activate_mana_source(state, mountain)
    assert state.mana_pool.get("R", 0) == 1
    assert _makeshift_munitions_legal(state, mm)

    makeshift_munitions_activate(state, mm)
    assert state.pending_resolution["kind"] == "pay_cost"
    guard = 0
    while state.pending_resolution is not None and state.pending_resolution["kind"] == "pay_cost":
        guard += 1
        assert guard < 10
        execute_pool_spend(state, pool_spend_options(state)[0])
    assert state.pending_resolution["kind"] == "choose_permanent"  # WHICH artifact/creature pays the sacrifice
    resolution.execute_choose_permanent_option(state, "Fodder", 1)
    assert creature not in state.players[0].battlefield
    assert state.pending_resolution["kind"] == "choose_any_target"
    resolution.execute_choose_any_target_player(state, 1)
    resolve_top_of_stack(state)
    assert state.players[1].life_total == 19  # 1 damage


def test_experimental_synthesizer_sac_via_action_table_sorcery_speed_and_cost():
    """Experimental Synthesizer's {2}{R} sac ability's cost_key mana-gating
    dispatch AND its "speed": Speed.SORCERY ("Activate only as a sorcery")
    timing restriction, driven through the REAL action table -- unlike
    test_experimental_synthesizer_impulse_and_samurai above, which calls
    experimental_synthesizer_sac directly and never touches either gate."""
    es_dl = [("Experimental Synthesizer", 1), ("Mountain", 8)]
    es_byname = {a[0]: (a[1], a[2]) for a in drl_env.build_action_table(es_dl, registry.EFFECT_REGISTRY)}
    state = GameState(on_the_play=True)
    state.phase = Phase.MAIN1
    state.turn_player_idx = 0
    state.active_idx = 0
    es = Permanent(registry.CARD_DEFS["Experimental Synthesizer"])
    es.slot = 1
    mountains = [Permanent(registry.CARD_DEFS["Mountain"]) for _ in range(3)]
    state.battlefield = [es] + mountains
    for m in mountains:
        activate_mana_source(state, m)
    assert state.mana_pool.get("R", 0) == 3
    legal, execute = es_byname["Activate Experimental Synthesizer (make_samurai)"]

    state.phase = Phase.DECLARE_ATTACKERS  # a real phase, but not a main phase
    assert not legal(state), "\"Activate only as a sorcery\" must be illegal outside MAIN1/MAIN2"

    state.phase = Phase.MAIN1
    assert legal(state)
    execute(state)
    assert state.pending_resolution["kind"] == "pay_cost"
    assert state.pending_resolution["remaining"] == {"generic": 2, "R": 1}
    guard = 0
    while state.pending_resolution is not None:
        guard += 1
        assert guard < 10
        execute_pool_spend(state, pool_spend_options(state)[0])
    assert es not in state.battlefield  # sacrificed as part of the effect's own resolve
    _drive_stack(state)
    assert any(p.card_def.name == "Samurai" for p in state.battlefield)
