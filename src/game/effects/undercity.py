"""The Initiative + the Undercity dungeon (Avenging Hunter).

A player can hold THE INITIATIVE (state.initiative_idx), a shared designation
like the monarch. Taking it, and the start of each of your upkeeps, makes you
"venture into Undercity": enter the first room if you're not in the dungeon,
else advance to the next room, applying that room's effect. Combat damage to
the initiative-holder queues a "you take the initiative" triggered ability
for the attacker (game.effects.combat.combat_damage_step, CR 722.2's second
triggered ability) -- initiative_idx only actually changes once that trigger
resolves off the stack (game.effects.triggers' own "take_initiative" branch,
which calls take_initiative below), same as Avenging Hunter's own ETB grant
(its own etb_trigger calls take_initiative directly, already resolving from
within its own stack entry).

This module owns the pure logic: take_initiative, venture + the room effects,
and the two "until your next turn" durations it introduces (Arena's Goad and
Throne's hexproof, both stamped with the turn they began and expired at the
owning player's next turn by expire_until_next_turn). The combat-facing
pieces -- the menace block rule and goad's forced attack -- live in
game.effects.combat (has_unfulfilled_goad / enforce_menace), which sits below
this module. The take_initiative and venture triggered abilities are queued
here and resolved by game.effects.triggers' own "take_initiative"/"venture"
branches. Two pending-resolution kinds (begin_choose_room, begin_throne_
reveal) live here rather than in game.resolution's shared handler pool --
both are Undercity-only, each with exactly one caller (venture / _room_throne,
respectively) in this module.

Real Undercity dungeon (Scryfall), branching room graph in _DUNGEON below."""

from . import casting
from .shared import find_to_hand, shuffle_library
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
    the stack with a priority window like any other trigger. Only ever called
    with active_idx == player_idx already -- Avenging Hunter's own etb_trigger
    (green_cards.py) calls this directly from within its OWN stack entry's
    resolution (active_idx == that trigger's controller, the permanent's
    controller); the combat-damage case instead queues a "take_initiative"
    trigger of its own (queue_take_initiative below) and this only runs when
    THAT resolves (game.effects.triggers' "take_initiative" branch), by which
    point resolve_top_of_stack has already restored active_idx to that
    trigger's own controller. Either way, queue_venture below always appends
    into the right player's own trigger_queue. "You can take the initiative
    even if you already have it": re-taking still queues a fresh venture,
    which this does unconditionally."""
    state.initiative_idx = player_idx
    state.log_event("take_initiative", player_idx=player_idx)
    queue_venture(state, player_idx)


def queue_take_initiative(state, player_idx):
    """Queue The Initiative's own combat-damage-triggered ability -- CR 722.2's
    second one, "whenever one or more creatures a player controls deal combat
    damage to the player who has the initiative, the controller of those
    creatures takes the initiative" -- for player_idx (game.effects.combat.
    combat_damage_step, once it's determined player_idx's creatures actually
    dealt combat damage to the current holder this step). A real triggered
    ability like any other: it goes on the stack and only actually flips
    state.initiative_idx once it resolves (game.effects.triggers._trigger_
    resolve's "take_initiative" branch, which calls take_initiative() above),
    not the instant the damage was dealt."""
    state.players[player_idx].trigger_queue.append(
        {"type": "take_initiative", "card_def": _INITIATIVE_CARD, "player_idx": player_idx}
    )


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
        begin_choose_room(state, nexts, lambda s, room: _enter_room(s, player_idx, room))


def _enter_room(state, player_idx, room):
    # dungeon_room reflects the current position; completing the terminal room
    # leaves the dungeon (None) so the next venture starts a new run.
    state.players[player_idx].dungeon_room = None if not _DUNGEON[room] else room
    state.log_event("undercity_enter_room", player_idx=player_idx, room=room)
    _ROOM_EFFECTS[room](state, player_idx)


def begin_choose_room(state, options, on_complete):
    """Undercity venture: the venturing player picks which of `options` (the
    1 or 2 rooms the current room leads to) to enter next. on_complete(state,
    room_name) fires with the chosen room. Undercity-only (venture above is
    its one caller) -- lives here rather than in game.resolution's shared
    handler pool."""
    resolution.begin_resolution(state, "choose_room", on_complete, options=tuple(options))


def choose_room_options(state):
    return list(state.pending_resolution["options"])


def execute_choose_room_option(state, name):
    resolution.complete_resolution(state, name)


# --- room effects (active_idx == the venturer throughout) ---

def _room_secret_entrance(state, player_idx):
    """Search your library for a basic land card, reveal it, put it into your
    hand, then shuffle -- find_to_hand (game.effects.shared) is the shared
    tail every other "search library for X, put it into hand, shuffle"
    effect already routes through; it also logs the zone_move (library ->
    hand) so replays/the visualizer can track the fetched card, and already
    handles the no-match case (name=None: still shuffles, finds nothing)."""
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
    """Undercity's Throne of the Dead Three: reveal the top `n` library cards;
    the venturer picks one CREATURE card from among them
    (throne_reveal_options). Every exit path (a pick, or the empty-reveal
    auto-complete when no creature is revealed) returns the unchosen revealed
    cards to the library and shuffles it -- done here so the library is always
    left consistent; on_complete(state, chosen_card_def | None) then only has
    to place the chosen creature (battlefield + counters + hexproof, in
    _room_throne below). Undercity-only (_room_throne is its one caller) --
    lives here rather than in game.resolution's shared handler pool."""
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
