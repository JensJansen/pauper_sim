"""The Initiative + the Undercity dungeon (Avenging Hunter).

A player can hold THE INITIATIVE (state.initiative_idx), a shared designation
like the monarch. Taking it, and the start of each of your upkeeps, makes you
"venture into Undercity": enter the first room if you're not in the dungeon,
else advance to the next room, applying that room's effect. Combat damage to
the initiative-holder passes the initiative to the attacker
(game.effects.combat.combat_damage_step).

This module owns the pure logic: take_initiative, venture + the room effects,
and the two "until your next turn" durations it introduces (Arena's Goad and
Throne's hexproof, both stamped with the turn they began and expired at the
owning player's next turn by expire_until_next_turn). The combat-facing
pieces -- the menace block rule and goad's forced attack -- live in
game.effects.combat (has_unfulfilled_goad / enforce_menace), which sits below
this module. The venture triggered ability is queued here and resolved by
game.effects.triggers' own "venture" branch.

Real Undercity dungeon (Scryfall), branching room graph in _DUNGEON below."""

from . import casting
from .stats import can_be_targeted
from .tokens import SKELETON_TOKEN_CARD_DEF, TREASURE_TOKEN_CARD_DEF, create_token
from .win_check import lose_life
from .. import resolution
from ..cards import CardDef, CardType

# Label for the venture triggered ability's stack entry (a designation, not a
# real card -- only used as the entry's card_def for logging/display). Exposed
# publicly (INITIATIVE_MARKER_CARD) so the token pipeline can reserve a vocab
# index + a choosable-name action for it (it appears on the stack as a venture
# trigger, and in order_triggers when it coincides with another trigger).
_INITIATIVE_CARD = CardDef("The Initiative", CardType.ENCHANTMENT, None, None)
INITIATIVE_MARKER_CARD = _INITIATIVE_CARD

ENTRANCE = "Secret Entrance"
_TERMINAL = "Throne of the Dead Three"

# room name -> the room(s) it leads to (a 1- or 2-way branch; () = terminal).
_DUNGEON = {
    "Secret Entrance": ("Forge", "Lost Well"),
    "Forge": ("Trap!", "Arena"),
    "Lost Well": ("Arena", "Stash"),
    "Trap!": ("Archives",),
    "Arena": ("Archives", "Catacombs"),
    "Stash": ("Catacombs",),
    "Archives": (_TERMINAL,),
    "Catacombs": (_TERMINAL,),
    _TERMINAL: (),
}

# Every room name -- drl_env.build_action_table pre-registers one "Enter room:
# X" action per name (only ever ≤2 legal at once, at a branch).
ROOM_NAMES = tuple(_DUNGEON)


def take_initiative(state, player_idx):
    """Set the initiative on `player_idx` and queue their venture. The Initiative
    reads "Whenever you take the initiative ... venture into Undercity"; that
    venture is a triggered ability, so it's queued (not run inline) and goes on
    the stack with a priority window like any other trigger. Always happens on
    that player's own turn (Avenging Hunter's ETB, or the combat that stole it),
    so promote_triggers_to_stack -- run with active_idx == player_idx -- picks
    it up. "You can take the initiative even if you already have it": re-taking
    still queues a fresh venture, which this does unconditionally."""
    state.initiative_idx = player_idx
    state.log_event("take_initiative", player_idx=player_idx)
    queue_venture(state, player_idx)


def queue_venture(state, player_idx):
    """Queue one "venture into Undercity" triggered ability for player_idx (the
    upkeep venture -- game.turn.upkeep_step -- and take_initiative both use
    this). Resolved by game.effects.triggers._trigger_resolve's "venture"
    branch, which calls venture()."""
    state.players[player_idx].trigger_queue.append(
        {"type": "venture", "card_def": _INITIATIVE_CARD, "player_idx": player_idx}
    )


def venture(state, player_idx):
    """Resolve one venture: advance the player one room (enter at Secret
    Entrance if not in the dungeon), letting them pick when the current room
    branches. Called from the venture trigger's resolution, with active_idx
    already == player_idx (the trigger's controller), so the room effects'
    active-player-proxied zone reads/choices are all this player's."""
    cur = state.players[player_idx].dungeon_room
    # cur is never a terminal room here: _enter_room clears dungeon_room to None
    # on completing the terminal room, so a later venture re-enters at Secret
    # Entrance (a fresh dungeon run) via the `cur is None` branch.
    nexts = (ENTRANCE,) if cur is None else _DUNGEON[cur]
    if len(nexts) == 1:
        _enter_room(state, player_idx, nexts[0])
    else:
        resolution.begin_choose_room(state, nexts, lambda s, room: _enter_room(s, player_idx, room))


def _enter_room(state, player_idx, room):
    # dungeon_room reflects the current position; completing the terminal room
    # leaves the dungeon (None) so the next venture starts a new run.
    state.players[player_idx].dungeon_room = None if not _DUNGEON[room] else room
    state.log_event("undercity_enter_room", player_idx=player_idx, room=room)
    _ROOM_EFFECTS[room](state, player_idx)


# --- room effects (active_idx == the venturer throughout) ---

def _room_secret_entrance(state, player_idx):
    """Search your library for a basic land card, reveal it, put it into your
    hand, then shuffle."""
    def _fetch(state, name):
        if name is None:  # no basic land in library -- fizzles, still shuffles nothing to do
            return
        card = next(c for c in state.library if c.name == name)
        state.library.remove(card)
        state.hand.append(card)
        state.rng.shuffle(state.library)

    resolution.begin_search_fetch(
        state, lambda c: c.card_type == CardType.LAND and c.extra.get("basic"), _fetch,
    )


def _room_forge(state, player_idx):
    """Put two +1/+1 counters on target creature."""
    def _on_target(state, target):
        captured = casting.capture_any_target(state, target)
        if captured is not None and captured[0] == "creature":
            perm = captured[1]
            perm.counters["+1/+1"] = perm.counters.get("+1/+1", 0) + 2

    resolution.begin_choose_any_target(
        state, lambda p: p.card_type == CardType.CREATURE and can_be_targeted(state, p, player_idx),
        _on_target, allow_players=False,
    )


def _room_lost_well(state, player_idx):
    """Scry 2."""
    resolution.begin_scry_surveil(state, "scry", 2, on_complete=lambda s: None)


def _room_trap(state, player_idx):
    """Target player loses 5 life."""
    def _on_player(state, idx):
        original = state.active_idx
        state.active_idx = idx  # lose_life always hits the active player -- flip to the target
        lose_life(state, 5, reason="undercity_trap")
        state.active_idx = original

    resolution.begin_choose_target_player(state, _on_player)


def _room_arena(state, player_idx):
    """Goad target creature."""
    def _on_target(state, target):
        captured = casting.capture_any_target(state, target)
        if captured is not None and captured[0] == "creature":
            apply_goad(state, captured[1], player_idx)

    resolution.begin_choose_any_target(
        state, lambda p: p.card_type == CardType.CREATURE and can_be_targeted(state, p, player_idx),
        _on_target, allow_players=False,
    )


def _room_stash(state, player_idx):
    """Create a Treasure token."""
    create_token(state, TREASURE_TOKEN_CARD_DEF)


def _room_archives(state, player_idx):
    """Draw a card."""
    state.draw(1)


def _room_catacombs(state, player_idx):
    """Create a 4/1 black Skeleton creature token with menace."""
    create_token(state, SKELETON_TOKEN_CARD_DEF)


def _room_throne(state, player_idx):
    """Reveal the top ten cards; put a creature card from among them onto the
    battlefield with three +1/+1 counters and hexproof until your next turn;
    then shuffle. begin_throne_reveal handles the reveal + shuffle-back; this
    only places the chosen creature."""
    def _place(state, chosen):
        if chosen is None:  # no creature revealed
            return
        perm = casting.enters_battlefield(state, chosen, from_zone="library")
        perm.counters["+1/+1"] = perm.counters.get("+1/+1", 0) + 3
        perm.flags["throne_hexproof"] = True  # read as hexproof by stats.creature_keywords
        perm.flags["throne_hexproof_turn"] = state.turn_number  # expires at this player's NEXT turn

    resolution.begin_throne_reveal(state, 10, _place)


_ROOM_EFFECTS = {
    "Secret Entrance": _room_secret_entrance,
    "Forge": _room_forge,
    "Lost Well": _room_lost_well,
    "Trap!": _room_trap,
    "Arena": _room_arena,
    "Stash": _room_stash,
    "Archives": _room_archives,
    "Catacombs": _room_catacombs,
    _TERMINAL: _room_throne,
}


# --- goad + the "until your next turn" durations ---

def apply_goad(state, permanent, goader_idx):
    """Goad `permanent` (Arena): its controller must attack with it if able,
    until the GOADER's next turn. Enforcement is in game.effects.combat
    (has_unfulfilled_goad); this stamps who goaded it and when so
    expire_until_next_turn can clear it at the goader's next turn."""
    permanent.flags["goaded_by"] = goader_idx
    permanent.flags["goaded_set_turn"] = state.turn_number
    state.log_event("goaded", permanent=(permanent.card_def.name, permanent.slot), by=goader_idx)


def expire_until_next_turn(state):
    """Turn-start cleanup (game.turn.untap_step, turn player active) for the two
    "until your next turn" durations this subsystem adds. Goad ends at the
    GOADER's next turn; Throne hexproof ends at the granted creature's
    CONTROLLER's next turn. Both were stamped with the turn_number they began
    on, so a strictly-later turn_number for the owning player clears them (never
    the same turn they were applied). Scans every battlefield -- a goaded
    creature is often the opponent's."""
    tp = state.turn_player_idx
    for idx, player in enumerate(state.players):
        for p in player.battlefield:
            if (p.flags.get("goaded_by") == tp
                    and p.flags.get("goaded_set_turn", state.turn_number) < state.turn_number):
                p.flags.pop("goaded_by", None)
                p.flags.pop("goaded_set_turn", None)
            if (idx == tp and p.flags.get("throne_hexproof")
                    and p.flags.get("throne_hexproof_turn", state.turn_number) < state.turn_number):
                p.flags.pop("throne_hexproof", None)
                p.flags.pop("throne_hexproof_turn", None)


if __name__ == "__main__":
    # ponytail self-check: run via `python -m game.effects.undercity` from src/.
    from ..state import GameState, Permanent, PlayerState
    from ..cards import EffectId
    from .stack import resolve_top_of_stack
    from .triggers import promote_triggers_to_stack

    def _drive(state):  # walk any pending resolution/stack to quiescence with a fixed default
        for _ in range(200):
            if state.stack:
                resolve_top_of_stack(state)
                continue
            promote_triggers_to_stack(state)
            if state.stack:
                continue
            break

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
    print("undercity.py take-initiative + Secret Entrance self-check: OK")

    # Branch choice: from Secret Entrance, venture again -> choose Forge or Lost
    # Well. Pick Forge, then put two +1/+1 counters on a creature.
    creature = Permanent(CardDef("Bear", CardType.CREATURE, None, EffectId.FILLER, power=2, toughness=2))
    creature.slot = 1
    state.players[0].battlefield = [creature]
    venture(state, 0)
    assert state.pending_resolution["kind"] == "choose_room"
    assert set(resolution.choose_room_options(state)) == {"Forge", "Lost Well"}
    resolution.execute_choose_room_option(state, "Forge")
    assert state.players[0].dungeon_room == "Forge"
    assert state.pending_resolution["kind"] == "choose_any_target"
    resolution.execute_choose_any_target_creature(state, 0, "Bear", 1)
    assert creature.counters["+1/+1"] == 2
    print("undercity.py branch + Forge self-check: OK")

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
    print("undercity.py goad expiry self-check: OK")

    # Throne: reveal top 10, put a creature onto the battlefield with 3 counters
    # + hexproof until next turn.
    from .stats import creature_keywords, permanent_power
    state = GameState(on_the_play=True, players=[PlayerState(True), PlayerState(False)])
    state.active_idx = state.turn_player_idx = 0
    state.turn_number = 2
    lib_creature = CardDef("Giant", CardType.CREATURE, {"G": 1}, EffectId.FILLER, power=3, toughness=3)
    state.players[0].library = [lib_creature] + [CardDef(f"L{i}", CardType.LAND, None, EffectId.FILLER) for i in range(9)]
    _room_throne(state, 0)
    assert state.pending_resolution["kind"] == "throne_reveal"
    resolution.execute_throne_reveal_option(state, "Giant")
    giant = next(p for p in state.players[0].battlefield if p.card_def.name == "Giant")
    assert giant.counters["+1/+1"] == 3 and permanent_power(state, giant) == 6  # 3 base + 3
    assert "hexproof" in creature_keywords(state, giant)
    assert len(state.players[0].library) == 9  # 9 non-chosen revealed cards shuffled back
    state.turn_number = 4  # a later turn of player 0
    expire_until_next_turn(state)
    assert "hexproof" not in creature_keywords(state, giant)  # expired
    print("undercity.py Throne self-check: OK")
