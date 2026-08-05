"""Initiative-taking, dungeon room branching/effects, and the goad/hexproof
"until your next turn" expiry the Undercity subsystem introduces."""
from game import resolution
from game.cards import CardDef, CardType, EffectId
from game.effects.stack import resolve_top_of_stack
from game.effects.stats import creature_keywords, permanent_power
from game.effects.triggers import promote_triggers_to_stack
from game.effects.undercity import (
    apply_goad,
    choose_room_options,
    execute_choose_room_option,
    execute_throne_reveal_option,
    expire_until_next_turn,
    take_initiative,
    venture,
    _room_throne,
)
from game.state import GameState, Permanent, PlayerState


def _venture_to_secret_entrance():
    """Setup for tests beyond the initial take-initiative self-check: the
    same steps, minus the intermediate assertions, ending with player 0
    already resolved through Secret Entrance (search_fetch complete,
    pending_resolution back to None)."""
    state = GameState(on_the_play=True, players=[PlayerState(True), PlayerState(False)])
    state.active_idx = state.turn_player_idx = 0
    basic = CardDef("Forest", CardType.LAND, None, EffectId.FILLER, basic=True, subtypes=("Forest",))
    state.players[0].library = [basic, CardDef("X", CardType.SORCERY, {}, EffectId.FILLER)]
    take_initiative(state, 0)
    promote_triggers_to_stack(state)
    resolve_top_of_stack(state)
    resolution.execute_search_fetch_option(state, "Forest")
    return state


def test_take_initiative_and_secret_entrance():
    # take_initiative sets it + queues a venture that enters Secret Entrance
    # (a 1-way "not in dungeon" entry) and searches a basic land to hand.
    state = GameState(on_the_play=True, players=[PlayerState(True), PlayerState(False)])
    state.active_idx = state.turn_player_idx = 0
    basic = CardDef("Forest", CardType.LAND, None, EffectId.FILLER, basic=True, subtypes=("Forest",))
    state.players[0].library = [basic, CardDef("X", CardType.SORCERY, {}, EffectId.FILLER)]
    take_initiative(state, 0)
    assert state.initiative_idx == 0
    promote_triggers_to_stack(state)
    assert len(state.stack) == 1  # the venture trigger
    resolve_top_of_stack(state)   # venture -> Secret Entrance -> begin_search_fetch
    assert state.players[0].dungeon_room == "Secret Entrance"
    assert state.pending_resolution["kind"] == "search_fetch"
    resolution.execute_search_fetch_option(state, "Forest")
    assert basic in state.players[0].hand


def test_branch_choice_forge_puts_counters_on_creature():
    # Branch choice: from Secret Entrance, venture again -> choose Forge or Lost
    # Well. Pick Forge, then put two +1/+1 counters on a creature.
    state = _venture_to_secret_entrance()
    creature = Permanent(CardDef("Bear", CardType.CREATURE, None, EffectId.FILLER, power=2, toughness=2))
    creature.slot = 1
    state.players[0].battlefield = [creature]
    venture(state, 0)
    assert state.pending_resolution["kind"] == "choose_room"
    assert set(choose_room_options(state)) == {"Forge", "Lost Well"}
    execute_choose_room_option(state, "Forge")
    assert state.players[0].dungeon_room == "Forge"
    assert state.pending_resolution["kind"] == "choose_any_target"
    resolution.execute_choose_any_target_creature(state, 0, "Bear", 1)
    assert creature.counters["+1/+1"] == 2


def test_goad_expiry_at_goaders_next_turn():
    # Goad expiry: goad set on turn 3 by player 0 persists, then clears at
    # player 0's next turn (a strictly-later turn_number).
    state = GameState(on_the_play=True, players=[PlayerState(True), PlayerState(False)])
    state.turn_number = 3
    state.turn_player_idx = 0
    victim = Permanent(CardDef("Goaded", CardType.CREATURE, None, EffectId.FILLER, power=1, toughness=1))
    state.players[1].battlefield = [victim]
    apply_goad(state, victim, 0)
    assert victim.flags.get("goaded_by") == 0
    state.turn_number = 4  # opponent's turn -- goad still active
    state.turn_player_idx = 1
    expire_until_next_turn(state)
    assert victim.flags.get("goaded_by") == 0  # not the goader's turn -- unchanged
    state.turn_number = 5  # player 0's next turn
    state.turn_player_idx = 0
    expire_until_next_turn(state)
    assert victim.flags.get("goaded_by") is None  # expired


def test_throne_places_creature_with_counters_and_hexproof():
    # Throne: reveal top 10, put a creature onto the battlefield with 3 counters
    # + hexproof until next turn.
    state = GameState(on_the_play=True, players=[PlayerState(True), PlayerState(False)])
    state.active_idx = state.turn_player_idx = 0
    state.turn_number = 2
    lib_creature = CardDef("Giant", CardType.CREATURE, {"G": 1}, EffectId.FILLER, power=3, toughness=3)
    state.players[0].library = [lib_creature] + [CardDef(f"L{i}", CardType.LAND, None, EffectId.FILLER) for i in range(9)]
    _room_throne(state, 0)
    assert state.pending_resolution["kind"] == "throne_reveal"
    execute_throne_reveal_option(state, "Giant")
    giant = next(p for p in state.players[0].battlefield if p.card_def.name == "Giant")
    assert giant.counters["+1/+1"] == 3 and permanent_power(state, giant) == 6  # 3 base + 3
    assert "hexproof" in creature_keywords(state, giant)
    assert len(state.players[0].library) == 9  # 9 non-chosen revealed cards shuffled back
    state.turn_number = 4  # a later turn of player 0
    expire_until_next_turn(state)
    assert "hexproof" not in creature_keywords(state, giant)  # expired
