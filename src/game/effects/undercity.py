"""The Initiative + the Undercity dungeon (Avenging Hunter).

A player can hold THE INITIATIVE (state.initiative_idx), a shared
designation like the monarch. Taking it, and the start of each upkeep,
makes you "venture into Undercity": enter the first room if not already in
the dungeon, else advance to the next. Combat damage to the initiative-
holder queues a "take the initiative" trigger for the attacker
(combat.combat_damage_step, CR 722.2); initiative_idx only flips once that
trigger resolves.

This module owns the pure logic: take_initiative, venture + room effects,
and the two "until your next turn" durations (Arena's Goad, Throne's
hexproof), expired by expire_until_next_turn. Combat-facing pieces --
menace blocking, goad's forced attack -- live in game.effects.combat.

Real Undercity dungeon (Scryfall); branching room graph in _DUNGEON below."""

from . import casting
from .shared import find_to_hand, shuffle_library
from .stats import can_be_targeted
from .tokens import SKELETON_TOKEN_CARD_DEF, TREASURE_TOKEN_CARD_DEF, create_token
from .win_check import lose_life
from .. import resolution
from ..cards import CardDef, CardType

# Label for the venture triggered ability's stack entry (a designation, not
# a real card). Exposed publicly so the token pipeline can reserve a vocab
# index + choosable-name action for it.
_INITIATIVE_CARD = CardDef("The Initiative", CardType.ENCHANTMENT, None, None)
INITIATIVE_MARKER_CARD = _INITIATIVE_CARD

ENTRANCE = "Secret Entrance"
_TERMINAL = "Throne of the Dead Three"

# room name -> the room(s) it leads to (1- or 2-way branch; () = terminal).
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
# X" action per name (≤2 legal at once, at a branch).
ROOM_NAMES = tuple(_DUNGEON)


def take_initiative(state, player_idx):
    """Set the initiative on `player_idx` and queue their venture (the
    Initiative's "whenever you take the initiative ... venture into
    Undercity"). Only ever called with active_idx == player_idx already.
    Re-taking still queues a fresh venture unconditionally."""
    state.initiative_idx = player_idx
    state.log_event("take_initiative", player_idx=player_idx)
    queue_venture(state, player_idx)


def queue_take_initiative(state, player_idx):
    """Queue the Initiative's combat-damage trigger (CR 722.2's second one)
    for player_idx, once combat.combat_damage_step determines their
    creatures dealt combat damage to the current holder this step. Only
    actually flips state.initiative_idx once it resolves (triggers.py's
    "take_initiative" branch, which calls take_initiative() above)."""
    state.players[player_idx].trigger_queue.append(
        {"type": "take_initiative", "card_def": _INITIATIVE_CARD, "player_idx": player_idx}
    )


def queue_venture(state, player_idx):
    """Queue one "venture into Undercity" trigger for player_idx (used by
    both the upkeep venture and take_initiative). Resolved by triggers.py's
    "venture" branch, which calls venture()."""
    state.players[player_idx].trigger_queue.append(
        {"type": "venture", "card_def": _INITIATIVE_CARD, "player_idx": player_idx}
    )


def venture(state, player_idx):
    """Resolve one venture: advance the player one room (enter at Secret
    Entrance if not in the dungeon), letting them pick when the current
    room branches. Called with active_idx already == player_idx."""
    cur = state.players[player_idx].dungeon_room
    # _enter_room clears dungeon_room to None on completing the terminal
    # room, so a later venture re-enters at Secret Entrance.
    nexts = (ENTRANCE,) if cur is None else _DUNGEON[cur]
    if len(nexts) == 1:
        _enter_room(state, player_idx, nexts[0])
    else:
        begin_choose_room(state, nexts, lambda s, room: _enter_room(s, player_idx, room))


def _enter_room(state, player_idx, room):
    # Completing the terminal room clears dungeon_room to None, so the next
    # venture starts a new run.
    state.players[player_idx].dungeon_room = None if not _DUNGEON[room] else room
    state.log_event("undercity_enter_room", player_idx=player_idx, room=room)
    _ROOM_EFFECTS[room](state, player_idx)


def begin_choose_room(state, options, on_complete):
    """The venturing player picks which of `options` (the 1 or 2 rooms the
    current room leads to) to enter next. on_complete(state, room_name)
    fires with the choice."""
    resolution.begin_resolution(state, "choose_room", on_complete, options=tuple(options))


def choose_room_options(state):
    return list(state.pending_resolution["options"])


def execute_choose_room_option(state, name):
    resolution.complete_resolution(state, name)


# --- room effects (active_idx == the venturer throughout) ---

def _room_secret_entrance(state, player_idx):
    """Search your library for a basic land card, reveal it, put it into
    your hand, then shuffle."""
    resolution.begin_search_fetch(
        state, lambda c: c.card_type == CardType.LAND and c.extra.get("basic"), find_to_hand,
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


def begin_throne_reveal(state, n, on_complete):
    """Throne of the Dead Three: reveal the top `n` library cards; the
    venturer picks one creature card from among them
    (throne_reveal_options). Every exit path returns the unchosen revealed
    cards to the library and shuffles it, so on_complete(state,
    chosen_card_def | None) only has to place the chosen creature."""
    revealed = state.library[:n]
    del state.library[:n]
    resolution.begin_resolution(state, "throne_reveal", on_complete, revealed=revealed)
    if not throne_reveal_options(state):
        _finish_throne(state, None)  # no creature among the revealed cards


def throne_reveal_options(state):
    return sorted({c.name for c in state.pending_resolution["revealed"] if c.card_type == CardType.CREATURE})


def execute_throne_reveal_option(state, name):
    chosen = next(c for c in state.pending_resolution["revealed"] if c.name == name and c.card_type == CardType.CREATURE)
    _finish_throne(state, chosen)


def _finish_throne(state, chosen):
    revealed = state.pending_resolution["revealed"]
    rest = [c for c in revealed if c is not chosen] if chosen is not None else list(revealed)
    state.library.extend(rest)
    shuffle_library(state)
    resolution.complete_resolution(state, chosen)


def _room_throne(state, player_idx):
    """Reveal the top ten cards; put a creature card from among them onto
    the battlefield with three +1/+1 counters and hexproof until your next
    turn; then shuffle. begin_throne_reveal handles the reveal + shuffle-
    back; this only places the chosen creature."""
    def _place(state, chosen):
        if chosen is None:  # no creature revealed
            return
        perm = casting.enters_battlefield(state, chosen, from_zone="library")
        perm.counters["+1/+1"] = perm.counters.get("+1/+1", 0) + 3
        perm.flags["throne_hexproof"] = True  # read as hexproof by stats.creature_keywords
        perm.flags["throne_hexproof_turn"] = state.turn_number  # expires at this player's NEXT turn

    begin_throne_reveal(state, 10, _place)


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
    """Goad `permanent` (Arena): its controller must attack with it if
    able, until the goader's next turn. Enforcement lives in
    game.effects.combat (has_unfulfilled_goad); this stamps who goaded it
    and when so expire_until_next_turn can clear it."""
    permanent.flags["goaded_by"] = goader_idx
    permanent.flags["goaded_set_turn"] = state.turn_number
    state.log_event("goaded", permanent=(permanent.card_def.name, permanent.slot), by=goader_idx)


def expire_until_next_turn(state):
    """Turn-start cleanup (game.turn.untap_step) for the two "until your
    next turn" durations: goad ends at the goader's next turn, Throne
    hexproof at the granted creature's controller's next turn. Both are
    cleared once the owning player's turn_number strictly exceeds the
    stamped one. Scans every battlefield -- a goaded creature is often the
    opponent's."""
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
