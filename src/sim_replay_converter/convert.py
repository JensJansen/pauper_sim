#!/usr/bin/env python3
"""
Convert MTG simulator JSON game logs into Cockatrice .cor replay files, so the
simulator's games can be watched in Cockatrice's built-in replay viewer.

Two source JSON shapes are supported, auto-detected per game:

  * snapshot-diff (boggles_mirror-style): each game has "opening_state" +
    "steps", where every step carries a full state_after snapshot. Consecutive
    snapshots are diffed to reconstruct the MoveCard/SetCardAttr/etc. stream
    a live Cockatrice server would have produced.

  * event-stream (mono_red_madness-style): each game has "events", an actual
    atomic event log (zone_move, mana_tap, attack_declared, ...). These map
    close to 1:1 onto Cockatrice events, with light bookkeeping for cards
    whose identity is only ever revealed at the moment they leave a zone
    (this format never logs a card entering a player's hand).

One JSON file can hold many games (--matchup runs write one deck-pair per
file; --eval runs write a whole round-robin, all pairings interleaved with
mirrors, in one file). The per-game output filename/description prefers
meta["config_name"], then meta["matchup"] (--matchup runs), then
meta["deck_a"]/["deck_b"] (older logs); a round-robin --eval log currently
carries none of those (see run_league.py's _run_eval docstring), so its
per-game deck pair is instead recovered from the sibling .log file's
"A vs B: N games" lines, in the same order games were logged.

Usage:
    python convert.py <sim.json> <output_dir> [--game INDEX]

On first run this generates Python protobuf bindings from the repo's
.proto files (cached alongside this script) via grpc_tools.protoc.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
from collections import Counter
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
# Vendored copy of libcockatrice_protocol/libcockatrice/protocol/pb/*.proto, so this
# whole directory is self-contained and portable to another repo. Override with
# COCKATRICE_PROTO_DIR to re-sync against a live Cockatrice checkout instead (the
# vendored copy can drift if Cockatrice's wire protocol changes later).
PROTO_DIR = Path(os.environ["COCKATRICE_PROTO_DIR"]) if "COCKATRICE_PROTO_DIR" in os.environ else SCRIPT_DIR / "proto"
PB_CACHE_DIR = SCRIPT_DIR / "_pb2_cache"


def bootstrap_pb2() -> None:
    """Generate (and cache) Python protobuf bindings for Cockatrice's .proto files."""
    proto_files = sorted(PROTO_DIR.glob("*.proto"))
    if not proto_files:
        sys.exit(f"no .proto files found under {PROTO_DIR}")

    stale = not PB_CACHE_DIR.exists()
    if not stale:
        newest_proto = max(p.stat().st_mtime for p in proto_files)
        generated = list(PB_CACHE_DIR.glob("*_pb2.py"))
        stale = not generated or max(p.stat().st_mtime for p in generated) < newest_proto

    if stale:
        from grpc_tools import protoc

        PB_CACHE_DIR.mkdir(parents=True, exist_ok=True)
        args = ["protoc", f"-I{PROTO_DIR}", f"--python_out={PB_CACHE_DIR}"] + [str(p) for p in proto_files]
        if protoc.main(args) != 0:
            sys.exit("protoc codegen failed")

    sys.path.insert(0, str(PB_CACHE_DIR))


bootstrap_pb2()

import card_attributes_pb2 as pb_attr  # noqa: E402
import event_attach_card_pb2 as pb_attach  # noqa: E402
import event_create_counter_pb2 as pb_createcounter  # noqa: E402
import event_create_arrow_pb2 as pb_createarrow  # noqa: E402
import event_create_token_pb2 as pb_createtoken  # noqa: E402
import event_delete_arrow_pb2 as pb_deletearrow  # noqa: E402
import event_destroy_card_pb2 as pb_destroy  # noqa: E402
import event_draw_cards_pb2 as pb_draw  # noqa: E402
import event_flip_card_pb2 as pb_flip  # noqa: E402
import event_game_state_changed_pb2 as pb_gsc  # noqa: E402
import event_reveal_cards_pb2 as pb_reveal  # noqa: E402
import event_move_card_pb2 as pb_move  # noqa: E402
import event_set_active_phase_pb2 as pb_setphase  # noqa: E402
import event_set_active_player_pb2 as pb_setplayer  # noqa: E402
import event_set_card_attr_pb2 as pb_setattr  # noqa: E402
import event_set_counter_pb2 as pb_setcounter  # noqa: E402
import event_shuffle_pb2 as pb_shuffle  # noqa: E402
import game_replay_pb2 as pb_replay  # noqa: E402
import serverinfo_zone_pb2 as pb_zone  # noqa: E402

# --- Zone name constants (must match libcockatrice/utility/zone_names.h) ---
Z_TABLE = "table"
Z_GRAVE = "grave"
Z_EXILE = "rfg"
Z_HAND = "hand"
Z_DECK = "deck"
Z_SIDEBOARD = "sb"
Z_STACK = "stack"

# --- Phase indices (must match cockatrice/src/game/phase.cpp Phases::phases) ---
PHASE_BY_NAME = {
    "untap": 0,
    "upkeep": 1,
    "draw": 2,
    "main1": 3,
    "declare_attackers": 5,
    "declare_blockers": 6,
    "combat_damage": 7,
    "main2": 9,
    "end": 10,
}

SLOT_RE = re.compile(r"^(.*) \(slot (\d+)\)$")

# Battlefield rows (table y-coordinate).
ROW_CREATURE = 0
ROW_OTHER = 1
ROW_LAND = 2

# Name-based land heuristic for the event-stream format, which gives no card-type
# info at all (no power/toughness, no "enchanting"/attach field, nothing). Catches
# every land actually seen in the red/black aggro decks this was built against;
# a nonbasic utility land with an unusual name would fall back to ROW_CREATURE.
BASIC_LAND_NAMES = {"Plains", "Island", "Swamp", "Mountain", "Forest", "Wastes"}


def is_probable_land(name):
    return name in BASIC_LAND_NAMES or name.startswith("Snow-Covered ")


# The event-stream format never states a deck's true size, and after fixing the
# mulligan/reveal issues below, the number of cards actually referenced in a game is
# well under a real 60-card constructed deck (most cards in the deck are never drawn).
# Floor the displayed library size at a realistic constructed-deck count rather than
# showing it drained down to exactly however many cards got used.
DEFAULT_DECK_SIZE = 60


def parse_slot_key(s):
    m = SLOT_RE.match(s)
    if not m:
        raise ValueError(f"unexpected slot-key format: {s!r}")
    return (m.group(1), int(m.group(2)))


def bf_key(card):
    return (card["name"], card["slot"])


def battlefield_row(card):
    """Row classifier for the snapshot-diff format, which does give power/toughness
    and an "enchanting" attach field (unlike the event-stream format above)."""
    if "power" in card:
        return ROW_CREATURE
    if card.get("enchanting"):
        return ROW_OTHER
    return ROW_LAND


def pop_by_name(pool, name):
    """Pop and return the first {'id','name'} entry matching name from a list-zone pool."""
    for i, c in enumerate(pool):
        if c["name"] == name:
            return pool.pop(i)
    return None


class PlayerConv:
    def __init__(self, idx, name):
        self.idx = idx
        self.name = name
        self.next_card_id = 0
        self.hand = []  # list of {"id", "name"}
        self.graveyard = []
        self.exile = []
        self.stack = []  # event-stream format only: cards currently "on the stack"
        self.pending_resolution = []  # event-stream format only: resolved-off-stack,
        # awaiting a battlefield entry to claim them, else flushed to graveyard
        self.battlefield = {}  # (name, slot) -> dict(id, x, y, tapped, power, toughness, keywords, enchanting)
        self.next_row_x = {ROW_LAND: 0, ROW_CREATURE: 0, ROW_OTHER: 0}
        self.library_draws = 0  # total cards ever pulled from the library this game
        self.counter_ids = {"life": 0, "w": 1, "u": 2, "b": 3, "r": 4, "g": 5, "x": 6}
        self.next_counter_id = 8
        self.deck_zone_info = None  # live reference, card_count patched once library_draws is final
        self.life = 20  # event-stream format only: tracked locally (no snapshot to read from)
        self.mana_pool = {}  # event-stream format only: tracked locally

    def new_id(self):
        cid = self.next_card_id
        self.next_card_id += 1
        return cid


def add_event(cont, player_id, ext_field):
    """Append a new GameEvent to cont and return the mutable extension payload."""
    ev = cont.event_list.add()
    if player_id is not None:
        ev.player_id = player_id
    return ev.Extensions[ext_field]


class BaseReplayBuilder:
    """Shared scaffolding: GameReplay/game_info setup, the opening
    Event_GameStateChanged snapshot, counters, and low-level event emission.
    Subclasses provide the actual step/event processing loop."""

    def __init__(self, game, meta):
        self.game = game
        self.replay = pb_replay.GameReplay()
        self.replay.replay_id = game["game_index"]
        info = self.replay.game_info
        info.game_id = game["game_index"]
        description = meta.get("config_name") or f'{meta.get("deck_a", "?")}_vs_{meta.get("deck_b", "?")}'
        info.description = f'{description} - game {game["game_index"]}'
        info.max_players = 2
        info.player_count = 2
        # NOTE: must stay False here. AbstractGame::loadReplay() calls
        # gameMetaInfo->setFromProto(game_info) at TabGame construction time,
        # which does a raw CopyFrom (bypassing setStarted()'s signal emission).
        # If game_info.started were already true, the first Event_GameStateChanged's
        # "game_started && !gameMetaInfo->started()" check would be false, setStarted()
        # would never fire, gameStarted() would never emit, and TabGame::startGame()
        # (which switches mainWidget to the actual play area) would never run --
        # the client would silently stay on the empty deck-view page forever.
        info.started = False
        info.spectators_omniscient = True

        self.players = [PlayerConv(0, "Player 1"), PlayerConv(1, "Player 2")]
        self.active_player = None
        self.active_phase = None
        self.next_arrow_id = 0
        self.live_arrow_ids = []
        self.live_attacker_ids = []  # (player_idx, card_id) currently flagged AttrAttacking="1"
        self.step_seconds = 0

    def _new_container(self):
        cont = self.replay.event_list.add()
        cont.seconds_elapsed = self.step_seconds
        self.step_seconds += 1
        return cont

    def other(self, p):
        return self.players[1 - p.idx]

    def _mint_from_library(self, p):
        card_id = p.new_id()
        p.library_draws += 1
        return card_id

    def _prune_empty_containers(self):
        kept = [c for c in self.replay.event_list if len(c.event_list) > 0]
        del self.replay.event_list[:]
        for c in kept:
            self.replay.event_list.add().CopyFrom(c)

    # ------------------------------------------------------------------
    # Initial snapshot: one Event_GameStateChanged carrying full opening
    # board state, exactly like a spectator joining an in-progress game.
    # ------------------------------------------------------------------
    def _emit_initial_snapshot(self, starting_player_idx, life_by_idx):
        cont = self._new_container()
        gsc = add_event(cont, None, pb_gsc.Event_GameStateChanged.ext)
        gsc.game_started = True
        gsc.active_player_id = starting_player_idx
        gsc.active_phase = 0
        gsc.seconds_elapsed = 0
        self.active_player = starting_player_idx
        self.active_phase = 0

        for p in self.players:
            pinfo = gsc.player_list.add()
            pinfo.properties.player_id = p.idx
            pinfo.properties.user_info.name = p.name
            pinfo.properties.ready_start = True

            for zone_name, has_coords, ztype in [
                (Z_HAND, False, pb_zone.ServerInfo_Zone.PrivateZone),
                (Z_TABLE, True, pb_zone.ServerInfo_Zone.PublicZone),
                (Z_STACK, False, pb_zone.ServerInfo_Zone.PublicZone),
                (Z_GRAVE, False, pb_zone.ServerInfo_Zone.PublicZone),
                (Z_EXILE, False, pb_zone.ServerInfo_Zone.PublicZone),
                (Z_DECK, False, pb_zone.ServerInfo_Zone.HiddenZone),
                (Z_SIDEBOARD, False, pb_zone.ServerInfo_Zone.HiddenZone),
            ]:
                zinfo = pinfo.zone_list.add()
                zinfo.name = zone_name
                zinfo.type = ztype
                zinfo.with_coords = has_coords
                zinfo.card_count = 0
                if zone_name == Z_DECK:
                    p.deck_zone_info = zinfo

            life = life_by_idx.get(p.idx, 20)
            p.life = life
            for cname, cval in [("life", life), ("w", 0), ("u", 0), ("b", 0), ("r", 0), ("g", 0), ("x", 0)]:
                cnt = pinfo.counter_list.add()
                cnt.id = p.counter_ids[cname]
                cnt.name = cname
                cnt.count = cval

    def _draw_cards(self, cont, p, names):
        drawn = []
        for name in names:
            card_id = self._mint_from_library(p)
            drawn.append({"id": card_id, "name": name})
        dc = add_event(cont, p.idx, pb_draw.Event_DrawCards.ext)
        dc.number = len(drawn)
        for card in drawn:
            ci = dc.cards.add()
            ci.id = card["id"]
            ci.name = card["name"]
            p.hand.append(card)

    def ensure_counter(self, cont, p, key):
        key = key.lower()
        cid = p.counter_ids.get(key)
        if cid is not None:
            return cid
        cid = p.next_counter_id
        p.next_counter_id += 1
        p.counter_ids[key] = cid
        cc = add_event(cont, p.idx, pb_createcounter.Event_CreateCounter.ext)
        cc.counter_info.id = cid
        cc.counter_info.name = key
        cc.counter_info.count = 0
        cc.counter_info.radius = 20
        col = cc.counter_info.counter_color
        col.r, col.g, col.b, col.a = 200, 200, 200, 255
        return cid

    def set_counter(self, cont, p, key, value):
        cid = self.ensure_counter(cont, p, key)
        sc = add_event(cont, p.idx, pb_setcounter.Event_SetCounter.ext)
        sc.counter_id = cid
        sc.value = value

    def _resolve_battlefield_ref(self, p, key):
        if key in p.battlefield:
            return p, p.battlefield[key]["id"]
        opp = self.other(p)
        if key in opp.battlefield:
            return opp, opp.battlefield[key]["id"]
        return p, None

    def _move(self, cont, src_p, src_zone, dst_p, dst_zone, card_id, new_card_id, card_name=None, x=None, y=None):
        mc = add_event(cont, src_p.idx, pb_move.Event_MoveCard.ext)
        mc.card_id = card_id
        mc.start_player_id = src_p.idx
        mc.start_zone = src_zone
        mc.target_player_id = dst_p.idx
        mc.target_zone = dst_zone
        mc.new_card_id = new_card_id
        if card_name:
            mc.card_name = card_name
        if x is not None:
            mc.x = x
        if y is not None:
            mc.y = y

    def _set_attr(self, cont, p, zone_name, card_id, attribute, value):
        sa = add_event(cont, p.idx, pb_setattr.Event_SetCardAttr.ext)
        sa.zone_name = zone_name
        sa.card_id = card_id
        sa.attribute = attribute
        sa.attr_value = value

    def _create_arrow(self, cont, start_p, start_card_id, target_p, target_card_id=None):
        ca = add_event(cont, start_p.idx, pb_createarrow.Event_CreateArrow.ext)
        arrow_id = self.next_arrow_id
        self.next_arrow_id += 1
        ai = ca.arrow_info
        ai.id = arrow_id
        ai.start_player_id = start_p.idx
        ai.start_zone = Z_TABLE
        ai.start_card_id = start_card_id
        ai.target_player_id = target_p.idx
        if target_card_id is not None:
            ai.target_zone = Z_TABLE
            ai.target_card_id = target_card_id
        ai.arrow_color.r, ai.arrow_color.g, ai.arrow_color.b = 255, 0, 0
        # Track the owner (the player CREATE_ARROW was dispatched to): Cockatrice
        # stores each arrow in that player's own arrow map, so the matching
        # DELETE_ARROW must be dispatched to the SAME player to remove it.
        self.live_arrow_ids.append((arrow_id, start_p.idx))

    def _clear_arrows(self, cont):
        if not self.live_arrow_ids:
            return
        # DELETE_ARROW isn't one of GameEventHandler's specially-routed event types --
        # it falls through to the generic per-player dispatch, which deletes the
        # arrow from THAT player's own arrow map. So each delete must be dispatched
        # to the arrow's owner (start_p.idx at creation), not a fixed player -- a
        # player-1-owned attack arrow deleted under player 0 simply never clears
        # (it lingered across every later phase for the seat-1 attacker).
        for arrow_id, owner_idx in self.live_arrow_ids:
            da = add_event(cont, owner_idx, pb_deletearrow.Event_DeleteArrow.ext)
            da.arrow_id = arrow_id
        self.live_arrow_ids = []

    def _clear_attackers(self, cont):
        # Event-stream builder only, mirroring _clear_arrows' lifecycle (an
        # attacker stays flagged through declare_blockers so its blocker's
        # target reads clearly, then clears on the next phase change). The
        # snapshot-diff builder needs no equivalent -- it already derives this
        # per-step from a full before/after attacker-set diff (PASS 5).
        if not self.live_attacker_ids:
            return
        for player_idx, card_id in self.live_attacker_ids:
            p = self.players[player_idx]
            self._set_attr(cont, p, Z_TABLE, card_id, pb_attr.AttrAttacking, "0")
        self.live_attacker_ids = []


class ReplayBuilder(BaseReplayBuilder):
    """Snapshot-diff format: each step carries a full state_after snapshot;
    consecutive snapshots are diffed into the Cockatrice event stream."""

    def __init__(self, game, meta):
        super().__init__(game, meta)
        opening = game["opening_state"]["players"]
        life_by_idx = {i: opening[i].get("life_total", 20) for i in (0, 1)}
        self._emit_initial_snapshot(game["starting_player_idx"], life_by_idx)

        # Draw the opening hand the same way a live server does: game starts with
        # hand=0/deck=full, then each player shuffles and draws their opener.
        # (Verified against real exported replays -- Cockatrice never pre-populates
        # the hand zone's card_list at GameStateChanged time.)
        for p in self.players:
            opening_hand = opening[p.idx]["hand"]
            if not opening_hand:
                continue
            draw_cont = self._new_container()
            sh = add_event(draw_cont, p.idx, pb_shuffle.Event_Shuffle.ext)
            sh.zone_name = Z_DECK
            self._draw_cards(draw_cont, p, opening_hand)

    # ------------------------------------------------------------------
    # Step processing: diff consecutive full-state snapshots into events.
    # ------------------------------------------------------------------
    def process_steps(self):
        prev_state = self.game["opening_state"]
        for step in self.game["steps"]:
            self._process_one_step(prev_state, step)
            prev_state = step["state_after"]

        for p in self.players:
            p.deck_zone_info.card_count = p.library_draws

    def _process_one_step(self, before, step):
        cont = self._new_container()

        turn_player = step["turn_player_idx"]
        if turn_player != self.active_player:
            sp = add_event(cont, None, pb_setplayer.Event_SetActivePlayer.ext)
            sp.active_player_id = turn_player
            self.active_player = turn_player

        phase_name = step["phase"]
        if phase_name is not None:
            phase_idx = PHASE_BY_NAME.get(phase_name)
            if phase_idx is not None and phase_idx != self.active_phase:
                sph = add_event(cont, None, pb_setphase.Event_SetActivePhase.ext)
                sph.phase = phase_idx
                self.active_phase = phase_idx
                if phase_idx != PHASE_BY_NAME["declare_blockers"]:
                    self._clear_arrows(cont)

        after_players = step["state_after"]["players"]
        for p in self.players:
            self._diff_player(cont, p, before["players"][p.idx], after_players[p.idx])

    def _diff_player(self, cont, p, before, after):
        before_bf = {bf_key(c): c for c in before["battlefield"]}
        after_bf = {bf_key(c): c for c in after["battlefield"]}

        entered_keys = sorted(k for k in after_bf if k not in before_bf)
        left_keys = [k for k in before_bf if k not in after_bf]
        kept_keys = [k for k in after_bf if k in before_bf]

        hand_removed = Counter(before["hand"]) - Counter(after["hand"])
        hand_added = Counter(after["hand"]) - Counter(before["hand"])
        grave_removed = Counter(before["graveyard"]) - Counter(after["graveyard"])
        grave_added = Counter(after["graveyard"]) - Counter(before["graveyard"])
        exile_removed = Counter(before["exile"]) - Counter(after["exile"])
        exile_added = Counter(after["exile"]) - Counter(before["exile"])

        # PASS 1: permanents that left the battlefield -> graveyard/exile/hand/ceases-to-exist
        # (hand is a real destination here: e.g. Rancor returns to its owner's hand
        # when the enchanted creature dies, instead of going to the graveyard)
        for key in left_keys:
            name = key[0]
            old = p.battlefield.pop(key)
            if grave_added[name] > 0:
                grave_added[name] -= 1
                self._move(cont, p, Z_TABLE, p, Z_GRAVE, old["id"], old["id"])
                p.graveyard.append({"id": old["id"], "name": name})
            elif exile_added[name] > 0:
                exile_added[name] -= 1
                self._move(cont, p, Z_TABLE, p, Z_EXILE, old["id"], old["id"])
                p.exile.append({"id": old["id"], "name": name})
            elif hand_added[name] > 0:
                hand_added[name] -= 1
                self._move(cont, p, Z_TABLE, p, Z_HAND, old["id"], old["id"])
                p.hand.append({"id": old["id"], "name": name})
            else:
                d = add_event(cont, p.idx, pb_destroy.Event_DestroyCard.ext)
                d.zone_name = Z_TABLE
                d.card_id = old["id"]

        # PASS 2: permanents that entered the battlefield <- hand/graveyard/exile/library
        for key in entered_keys:
            name, _slot = key
            card = after_bf[key]
            y = battlefield_row(card)
            x = p.next_row_x[y]
            p.next_row_x[y] += 3

            if hand_removed[name] > 0:
                hand_removed[name] -= 1
                src_card = pop_by_name(p.hand, name)
                card_id = src_card["id"]
                self._move(cont, p, Z_HAND, p, Z_TABLE, card_id, card_id, x=x, y=y)
            elif grave_removed[name] > 0:
                grave_removed[name] -= 1
                src_card = pop_by_name(p.graveyard, name)
                card_id = src_card["id"]
                self._move(cont, p, Z_GRAVE, p, Z_TABLE, card_id, card_id, x=x, y=y)
            elif exile_removed[name] > 0:
                exile_removed[name] -= 1
                src_card = pop_by_name(p.exile, name)
                card_id = src_card["id"]
                self._move(cont, p, Z_EXILE, p, Z_TABLE, card_id, card_id, x=x, y=y)
            else:
                card_id = self._mint_from_library(p)
                self._move(cont, p, Z_DECK, p, Z_TABLE, 0, card_id, card_name=name, x=x, y=y)

            entry = {
                "id": card_id,
                "x": x,
                "y": y,
                "tapped": False,
                "power": card.get("power"),
                "toughness": card.get("toughness"),
                "keywords": [],
                "enchanting": None,
            }
            p.battlefield[key] = entry
            if card["tapped"]:
                self._set_attr(cont, p, Z_TABLE, card_id, pb_attr.AttrTapped, "1")
                entry["tapped"] = True
            if "power" in card:
                pt = f'{card["power"]}/{card["toughness"]}'
                self._set_attr(cont, p, Z_TABLE, card_id, pb_attr.AttrPT, pt)
            if card.get("keywords"):
                ann = ", ".join(sorted(card["keywords"]))
                self._set_attr(cont, p, Z_TABLE, card_id, pb_attr.AttrAnnotation, ann)
                entry["keywords"] = list(card["keywords"])

        # PASS 3: permanents kept on the battlefield -> attribute diffs
        for key in kept_keys:
            old = p.battlefield[key]
            new = after_bf[key]
            if old["tapped"] != new["tapped"]:
                self._set_attr(cont, p, Z_TABLE, old["id"], pb_attr.AttrTapped, "1" if new["tapped"] else "0")
                old["tapped"] = new["tapped"]
            if "power" in new:
                newpt = (new["power"], new["toughness"])
                if (old["power"], old["toughness"]) != newpt:
                    self._set_attr(cont, p, Z_TABLE, old["id"], pb_attr.AttrPT, f"{newpt[0]}/{newpt[1]}")
                    old["power"], old["toughness"] = newpt
            new_kw = new.get("keywords") or []
            if new_kw != old["keywords"]:
                ann = ", ".join(sorted(new_kw))
                self._set_attr(cont, p, Z_TABLE, old["id"], pb_attr.AttrAnnotation, ann)
                old["keywords"] = list(new_kw)

        # PASS 4: attachments (auras/equipment), for both newly entered and kept permanents
        for key in entered_keys + kept_keys:
            new_ench = after_bf[key].get("enchanting")
            old = p.battlefield[key]
            if new_ench != old["enchanting"]:
                ac = add_event(cont, p.idx, pb_attach.Event_AttachCard.ext)
                ac.start_zone = Z_TABLE
                ac.card_id = old["id"]
                if new_ench:
                    target_owner, target_id = self._resolve_battlefield_ref(p, parse_slot_key(new_ench))
                    if target_id is not None:
                        ac.target_player_id = target_owner.idx
                        ac.target_zone = Z_TABLE
                        ac.target_card_id = target_id
                old["enchanting"] = new_ench

        # PASS 5: attacking flag (+ an arrow to the opposing player, matching live Cockatrice)
        before_attackers = {parse_slot_key(s) for s in before.get("attackers", [])}
        after_attackers = {parse_slot_key(s) for s in after.get("attackers", [])}
        for key in after_attackers - before_attackers:
            if key in p.battlefield:
                self._set_attr(cont, p, Z_TABLE, p.battlefield[key]["id"], pb_attr.AttrAttacking, "1")
                self._create_arrow(cont, p, p.battlefield[key]["id"], self.other(p))
        for key in before_attackers - after_attackers:
            if key in p.battlefield:
                self._set_attr(cont, p, Z_TABLE, p.battlefield[key]["id"], pb_attr.AttrAttacking, "0")

        # PASS 6: blocking arrows (blocker -> attacker), attacker key lives on this player's side
        for attacker_str, blocker_val in (after.get("blocked_by") or {}).items():
            blockers = blocker_val if isinstance(blocker_val, list) else [blocker_val]
            attacker_key = parse_slot_key(attacker_str)
            if attacker_key not in p.battlefield:
                continue
            for blocker_str in blockers:
                blocker_key = parse_slot_key(blocker_str)
                opp = self.other(p)
                if blocker_key not in opp.battlefield:
                    continue
                self._create_arrow(
                    cont, opp, opp.battlefield[blocker_key]["id"], p, p.battlefield[attacker_key]["id"]
                )

        # PASS 7: hand-only movement left unexplained by the battlefield passes above
        # (discards, draws, mulligans, direct-to-hand fetches)
        for name, cnt in list(hand_removed.items()):
            for _ in range(cnt):
                card = pop_by_name(p.hand, name)
                if card is None:
                    continue
                if grave_added[name] > 0:
                    grave_added[name] -= 1
                    self._move(cont, p, Z_HAND, p, Z_GRAVE, card["id"], card["id"])
                    p.graveyard.append(card)
                elif exile_added[name] > 0:
                    exile_added[name] -= 1
                    self._move(cont, p, Z_HAND, p, Z_EXILE, card["id"], card["id"])
                    p.exile.append(card)
                else:
                    # unexplained hand departure (e.g. a mulligan put-back): return it to the
                    # library rather than destroying it -- unlike battlefield tokens, a real
                    # card in hand doesn't just cease to exist.
                    self._move(cont, p, Z_HAND, p, Z_DECK, card["id"], card["id"])

        # cards newly present in hand with no explained source all came from the library;
        # batch them into one real Event_DrawCards (matches what a live Cockatrice server
        # emits for a draw/mulligan, and gives a much cleaner log line than N MoveCards)
        drawn_names = [name for name, cnt in hand_added.items() for _ in range(cnt)]
        if drawn_names:
            self._draw_cards(cont, p, drawn_names)

        # PASS 8: any remaining graveyard/exile growth (mill-type effects straight from library)
        for name, cnt in list(grave_added.items()):
            for _ in range(cnt):
                card_id = self._mint_from_library(p)
                self._move(cont, p, Z_DECK, p, Z_GRAVE, 0, card_id, card_name=name)
                p.graveyard.append({"id": card_id, "name": name})
        for name, cnt in list(exile_added.items()):
            for _ in range(cnt):
                card_id = self._mint_from_library(p)
                self._move(cont, p, Z_DECK, p, Z_EXILE, 0, card_id, card_name=name)
                p.exile.append({"id": card_id, "name": name})

        # PASS 9: life total + mana pool
        if before["life_total"] != after["life_total"]:
            self.set_counter(cont, p, "life", after["life_total"])
        before_mana = before.get("mana_pool") or {}
        after_mana = after.get("mana_pool") or {}
        for color in set(before_mana) | set(after_mana):
            bv, av = before_mana.get(color, 0), after_mana.get(color, 0)
            if bv != av:
                self.set_counter(cont, p, color, av)


# JSON zone name -> Cockatrice zone constant, for the event-stream format.
JSON_ZONE_TO_COCKATRICE = {
    "hand": Z_HAND,
    "graveyard": Z_GRAVE,
    "exile": Z_EXILE,
    "stack": Z_STACK,
    "battlefield": Z_TABLE,
    "library": Z_DECK,
    "library_bottom": Z_DECK,
}


def _norm_zone(z):
    """The engine tags some exiles as exile_untracked / exile_mesmeric (delve and
    other exile-costs, impulse draws, Mesmeric-style effects). To a replay viewer
    they're all just the exile zone; collapse them so the move actually renders
    instead of being dropped by the `dst_zone is None` guards below."""
    return "exile" if z and z.startswith("exile") else z


class EventStreamReplayBuilder(BaseReplayBuilder):
    """Event-stream format: an actual atomic event log (zone_move, mana_tap,
    attack_declared, life_change, ...). Every draw is logged explicitly and by
    name as a zone_move (library->hand, reason='draw'), so hand identity is known
    the moment a card is drawn -- no lookahead needed. Card identity in the
    stack/graveyard/exile is still only revealed as a card leaves a zone (no
    explicit "cast" event names its source), so those pools are built up lazily
    as names are first referenced. life_change gives an authoritative running
    total for every life change, combat or otherwise, so life tracking is exact.

    Mulligans are logged as repeated full-hand 'draw' batches (one per attempt)
    with the cards not kept logged as mulligan_take / mulligan_bottom put-backs;
    __init__ nets these into the single kept opening hand rather than replaying
    every round (an agent that mulligans to zero would otherwise dump ~100 churn
    events at t=0).
    """

    def __init__(self, game, meta):
        super().__init__(game, meta)
        events = game["events"]
        self.starting_player_idx = next(e["turn_player_idx"] for e in events if e["kind"] == "turn_start")
        self._emit_initial_snapshot(self.starting_player_idx, {0: 20, 1: 20})

        # Everything before the first turn_start is the opening-hand / mulligan
        # sequence: each mulligan attempt is logged as a fresh full-hand 'draw'
        # (library->hand), with the cards not kept logged as mulligan_take /
        # mulligan_bottom put-backs (hand->library). Net those out per player to
        # get the actual kept opening hand and render it as a single shuffle +
        # draw, the way a live server does. Post-first-turn events (self.main_events)
        # are the real game and are replayed verbatim by process_events.
        first_turn = next(i for i, e in enumerate(events) if e["kind"] == "turn_start")
        self.main_events = events[first_turn:]
        opening = {0: Counter(), 1: Counter()}
        for e in events[:first_turn]:
            if e["kind"] != "zone_move":
                continue
            names = e.get("cards") or ([e["card"]] if e.get("card") else [])
            reason = e.get("reason")
            if reason == "draw":
                opening[e["active_idx"]] += Counter(names)
            elif reason in ("mulligan_take", "mulligan_bottom"):
                opening[e["active_idx"]] -= Counter(names)

        for p in self.players:
            hand = list(opening[p.idx].elements())
            if not hand:
                continue
            cont = self._new_container()
            sh = add_event(cont, p.idx, pb_shuffle.Event_Shuffle.ext)
            sh.zone_name = Z_DECK
            self._draw_cards(cont, p, hand)

    def _resolve_hand_card(self, p, name):
        """Get (card_id, revealed_name_or_None) for a specific named card about to
        leave hand: reuse the matching hand entry (revealed when it was drawn),
        else mint fresh as a last resort if hand tracking somehow undercounted."""
        c = pop_by_name(p.hand, name)
        if c is not None:
            return c["id"], None
        return self._mint_from_library(p), name

    def process_events(self):
        for e in self.main_events:
            kind = e["kind"]
            handler = self._HANDLERS.get(kind)
            if handler is not None:
                handler(self, e)
            # pass, priority_flip, resolution_begin, resolution_complete,
            # cleanup_damage_cleared, trigger_fired, declare_blockers_cap_abandoned,
            # destroy_failed_indestructible, menace_unblocked, aura_no_target:
            # no Cockatrice-visible effect (bookkeeping / a decision or safety-cap
            # outcome with no board-state change of its own).
            # combat_damage / fight_damage: superseded by life_change, which fires for
            # the same damage with the resulting total already computed; creature-vs-
            # creature damage was never visualized here (state_based_death covers deaths).
            # impulse_expired: no-op -- an unplayed impulse card stays in exile (see
            # _handle_impulse_exile). Deliberately still unrendered (no clean Cockatrice
            # primitive, and no reliable data to render them from):
            #   pump          -- P/T buff, but the event-stream entry doesn't carry base
            #                    P/T and the amount's shape (+N/+N vs +N/+0) is ambiguous.
            #   explore       -- "plus_counter" result is the same kind of P/T buff as
            #                    pump, for the same reason.
            #   animated      -- a permanent becomes a creature type-change; the
            #                    battlefield row it's drawn in never gets reclassified.
            #   block_unassigned -- a removed block; arrows aren't keyed by pair and clear
            #                    at the next phase boundary anyway.
            #   undercity_enter_room, take_initiative, goaded, ward_triggered -- dungeon /
            #                    initiative / goad / ward status with no board-visible analog.
            # ponder: the shuffled/ordered LOOK is invisible either way (deck order isn't
            #                    tracked); nothing here ever needs to move or reveal a card.

        # flush any spells still awaiting a fate when the recorded game ends
        self._flush_pending_resolutions(self._new_container())

        for p in self.players:
            p.deck_zone_info.card_count = max(DEFAULT_DECK_SIZE, p.library_draws)

        self._prune_empty_containers()

    def _flush_pending_resolutions(self, cont):
        for p in self.players:
            for c in p.pending_resolution:
                self._move(cont, p, Z_STACK, p, Z_GRAVE, c["id"], c["id"])
                p.graveyard.append({"id": c["id"], "name": c["name"]})
            p.pending_resolution.clear()

    def _handle_turn_start(self, e):
        turn_player = e["turn_player_idx"]
        if turn_player == self.active_player:
            return
        cont = self._new_container()
        sp = add_event(cont, None, pb_setplayer.Event_SetActivePlayer.ext)
        sp.active_player_id = turn_player
        self.active_player = turn_player
        self._flush_pending_resolutions(cont)

    def _handle_phase_change(self, e):
        phase_idx = PHASE_BY_NAME.get(e["phase"])
        if phase_idx is None or phase_idx == self.active_phase:
            return
        cont = self._new_container()
        sph = add_event(cont, None, pb_setphase.Event_SetActivePhase.ext)
        sph.phase = phase_idx
        self.active_phase = phase_idx
        if phase_idx != PHASE_BY_NAME["declare_blockers"]:
            self._clear_arrows(cont)
            self._clear_attackers(cont)
        self._flush_pending_resolutions(cont)
        # Draws are NOT synthesized here: this format logs every draw explicitly
        # as a named library->hand zone_move (handled by _handle_card_move).

    def _handle_mana_tap(self, e):
        p = self.players[e["active_idx"]]
        cont = self._new_container()
        name, slot = e["permanent"]
        entry = p.battlefield.get((name, slot))
        if entry is not None and not entry["tapped"]:
            self._set_attr(cont, p, Z_TABLE, entry["id"], pb_attr.AttrTapped, "1")
            entry["tapped"] = True
        for color in e.get("produced") or []:
            p.mana_pool[color] = p.mana_pool.get(color, 0) + 1
            self.set_counter(cont, p, color, p.mana_pool[color])

    def _handle_mana_spend(self, e):
        p = self.players[e["active_idx"]]
        cont = self._new_container()
        color = e["color"]
        p.mana_pool[color] = max(0, p.mana_pool.get(color, 0) - 1)
        self.set_counter(cont, p, color, p.mana_pool[color])

    def _handle_untap_step(self, e):
        p = self.players[e["active_idx"]]
        cont = self._new_container()
        for name, slot in e.get("untapped") or []:
            entry = p.battlefield.get((name, slot))
            if entry is not None and entry["tapped"]:
                self._set_attr(cont, p, Z_TABLE, entry["id"], pb_attr.AttrTapped, "0")
                entry["tapped"] = False

    def _handle_mana_emptied(self, e):
        # Rule 500.4: unused mana empties at every step/phase boundary, for
        # whichever players actually had a nonzero pool (keyed by player index,
        # stringified by the JSON round-trip). Replaces the old once-per-turn
        # untap_step clear, so this is now the only place mana pool counters
        # reset to 0 in the replay.
        cont = self._new_container()
        for idx_str, pool in (e.get("pools") or {}).items():
            p = self.players[int(idx_str)]
            for color in pool:
                p.mana_pool[color] = 0
                self.set_counter(cont, p, color, 0)

    def _handle_attack_declared(self, e):
        p = self.players[e["active_idx"]]
        name, slot = e["attacker"]
        entry = p.battlefield.get((name, slot))
        if entry is None:
            return
        cont = self._new_container()
        self._set_attr(cont, p, Z_TABLE, entry["id"], pb_attr.AttrAttacking, "1")
        self.live_attacker_ids.append((p.idx, entry["id"]))
        self._create_arrow(cont, p, entry["id"], self.other(p))

    def _handle_block_assigned(self, e):
        blocker_p = self.players[e["active_idx"]]
        attacker_p = self.other(blocker_p)
        b_entry = blocker_p.battlefield.get(tuple(e["blocker"]))
        a_entry = attacker_p.battlefield.get(tuple(e["attacker"]))
        if b_entry is None or a_entry is None:
            return
        cont = self._new_container()
        self._create_arrow(cont, blocker_p, b_entry["id"], attacker_p, a_entry["id"])

    def _handle_aura_attached(self, e):
        # Aura resolving onto its target: draw an arrow from the aura to what it's
        # enchanting, then a moment (one container == ~1 replay-second) later snap it
        # on and clear the arrow. _resolve_battlefield_ref finds the target on either
        # player's side (auras like Cartouche can enchant an opponent's creature).
        aura_p = self.players[e["active_idx"]]
        aura = aura_p.battlefield.get(tuple(e["aura"]))
        if aura is None:
            return
        target_owner, target_id = self._resolve_battlefield_ref(aura_p, tuple(e["target"]))
        if target_id is None:
            return
        self._create_arrow(self._new_container(), aura_p, aura["id"], target_owner, target_id)
        cont = self._new_container()
        ac = add_event(cont, aura_p.idx, pb_attach.Event_AttachCard.ext)
        ac.start_zone = Z_TABLE
        ac.card_id = aura["id"]
        ac.target_player_id = target_owner.idx
        ac.target_zone = Z_TABLE
        ac.target_card_id = target_id
        # ponytail: only arrows live mid-main-phase are the one we just made; switch to
        # a targeted delete if a combat arrow can ever overlap an aura resolution.
        self._clear_arrows(cont)

    def _handle_life_change(self, e):
        # covers every source of life change, combat or otherwise (confirmed against
        # the log: a combat_damage event hitting a player is always immediately
        # followed by one of these with the resulting total) -- no need to also
        # track life off of combat_damage, which would double-count it.
        p = self.players[e["player_idx"]]
        p.life = e["new_total"]
        cont = self._new_container()
        self.set_counter(cont, p, "life", p.life)

    def _handle_state_based_death(self, e):
        owner = self.players[e["owner_idx"]]
        name, slot = e["permanent"]
        old = owner.battlefield.pop((name, slot), None)
        if old is None:
            return
        cont = self._new_container()
        if old.get("is_token"):
            d = add_event(cont, owner.idx, pb_destroy.Event_DestroyCard.ext)
            d.zone_name = Z_TABLE
            d.card_id = old["id"]
            return
        front = self._revert_dfc_face(cont, owner, old)
        self._move(cont, owner, Z_TABLE, owner, Z_GRAVE, old["id"], old["id"])
        owner.graveyard.append({"id": old["id"], "name": front or name})

    def _handle_zone_move(self, e):
        if "disposed" in e:
            # Scry/surveil: no "card"/"cards"/"permanent" field at all -- the moved
            # names live in "disposed" (dispose target: "disposed_to") and
            # "kept_to_library_top" instead, so the generic dispatch below never
            # fires for this shape. Without this branch a surveil-to-graveyard
            # silently vanished from the replay.
            self._handle_disposed(e)
            return
        from_zone = _norm_zone(e.get("from_zone"))
        to_zone = _norm_zone(e.get("to_zone"))
        if "permanent" in e:
            self._handle_permanent_move(e, from_zone, to_zone)
        else:
            name = e.get("card")
            if name is not None:
                self._handle_card_move(e, name, from_zone, to_zone)
            else:
                for n in e.get("cards") or []:
                    self._handle_card_move(e, n, from_zone, to_zone)

    def _resolve_incoming_permanent(self, p, name, from_zone):
        """Find or mint a card id for a permanent about to enter the battlefield.
        Returns (card_id, source_zone_or_None, is_token, reveal_name_or_None)."""
        for i, c in enumerate(p.pending_resolution):
            if c["name"] == name:
                return p.pending_resolution.pop(i)["id"], Z_STACK, False, None
        if from_zone == "hand":
            card_id, reveal = self._resolve_hand_card(p, name)
            return card_id, Z_HAND, False, reveal
        pool = {"graveyard": p.graveyard, "exile": p.exile}.get(from_zone)
        if pool is not None:
            c = pop_by_name(pool, name)
            if c:
                return c["id"], JSON_ZONE_TO_COCKATRICE[from_zone], False, None
        if from_zone is None:
            return p.new_id(), None, True, None  # a token: no card-pool identity at all
        return self._mint_from_library(p), Z_DECK, False, name

    def _handle_permanent_move(self, e, from_zone, to_zone):
        p = self.players[e["active_idx"]]
        name, slot = e["permanent"]
        key = (name, slot)

        if to_zone == "battlefield":
            card_id, src_zone, is_token, reveal = self._resolve_incoming_permanent(p, name, from_zone)
            card_type = e.get("card_type")
            if card_type is not None:
                y = ROW_LAND if card_type == "LAND" else (ROW_CREATURE if card_type == "CREATURE" else ROW_OTHER)
            else:
                # older data without card_type: fall back to a name guess
                y = ROW_LAND if is_probable_land(name) else ROW_CREATURE
            x = p.next_row_x[y]
            p.next_row_x[y] += 3
            cont = self._new_container()
            if is_token:
                ct = add_event(cont, p.idx, pb_createtoken.Event_CreateToken.ext)
                ct.zone_name = Z_TABLE
                ct.card_id = card_id
                ct.card_name = name
                ct.destroy_on_zone_change = True
                ct.x = x
                ct.y = y
            else:
                self._move(cont, p, src_zone, p, Z_TABLE, card_id, card_id, card_name=reveal, x=x, y=y)
            entry = {"id": card_id, "x": x, "y": y, "tapped": False, "is_token": is_token}
            p.battlefield[key] = entry
            if e.get("tapped"):
                self._set_attr(cont, p, Z_TABLE, card_id, pb_attr.AttrTapped, "1")
                entry["tapped"] = True
            if e.get("power") is not None and e.get("toughness") is not None:
                self._set_attr(cont, p, Z_TABLE, card_id, pb_attr.AttrPT, f'{e["power"]}/{e["toughness"]}')
            return

        old = p.battlefield.pop(key, None)
        if old is None:
            return
        cont = self._new_container()
        if to_zone == "ceases_to_exist" or old.get("is_token"):
            d = add_event(cont, p.idx, pb_destroy.Event_DestroyCard.ext)
            d.zone_name = Z_TABLE
            d.card_id = old["id"]
            return
        dst_zone = JSON_ZONE_TO_COCKATRICE.get(to_zone)
        if dst_zone is None:
            return
        front = self._revert_dfc_face(cont, p, old)
        self._move(cont, p, Z_TABLE, p, dst_zone, old["id"], old["id"])
        dst_pool = {"hand": p.hand, "graveyard": p.graveyard, "exile": p.exile}.get(to_zone)
        if dst_pool is not None:
            dst_pool.append({"id": old["id"], "name": front or name})

    def _resolve_incoming_hand_like(self, p, name, from_zone):
        """Get (card_id, src_zone, revealed_name_or_None) for a card about to
        go onto the stack. from_zone is only ever "hand" or None here (see
        game.effects.stack.push_to_stack: it logs "hand" when the cast
        reserves a hand card, else None for every reserves_hand_card=False
        cast -- Flashback/Escape from the graveyard, Madness/Plot/Adventure
        from exile, an eagerly-discarded alt cost, a copy). None can't be
        told apart further from the event alone, so those all fall back to
        the same name-based search across the non-hand pools.

        from_zone == "hand" must check ONLY p.hand, never fall through to
        the other pools: a same-named card can easily sit in more than one
        place at once (e.g. a second physical copy still in hand while the
        first is being flashed back from the graveyard), and checking hand
        unconditionally used to steal that unrelated hand copy's identity
        for the graveyard-sourced cast -- confirmed live, a mono red madness
        Lava Dart hard-cast-then-flashback: the flashback grabbed the OTHER
        Lava Dart still sitting in hand instead of the resolved one in the
        graveyard, leaving the real graveyard copy untouched and making the
        replay look like the same card was recast while still on the stack,
        unresolved."""
        if from_zone == "hand":
            c = pop_by_name(p.hand, name)
            if c:
                return c["id"], Z_HAND, None
            return self._mint_from_library(p), Z_DECK, name
        c = pop_by_name(p.exile, name)
        if c:
            return c["id"], Z_EXILE, None
        # A same-turn recast (the resolved card hasn't hit a phase boundary
        # yet, so _flush_pending_resolutions hasn't moved it into p.graveyard)
        # -- same identity source _resolve_incoming_permanent already draws on
        # just above, for the same reason (a resolution marker draws no
        # visual move of its own; see _handle_card_move's "resolution marker
        # only" branch below).
        for i, c in enumerate(p.pending_resolution):
            if c["name"] == name:
                return p.pending_resolution.pop(i)["id"], Z_STACK, None
        c = pop_by_name(p.graveyard, name)
        if c:
            return c["id"], Z_GRAVE, None
        # not tracked anywhere -- came straight from the hidden deck zone,
        # revealing its name for the first time now (or a token-like copy
        # spell with no physical card behind it at all).
        return self._mint_from_library(p), Z_DECK, name

    def _handle_card_move(self, e, name, from_zone, to_zone):
        if to_zone == "stack":
            p = self.players[e.get("controller", e["active_idx"])]
            cont = self._new_container()
            card_id, src_zone, reveal = self._resolve_incoming_hand_like(p, name, from_zone)
            self._move(cont, p, src_zone, p, Z_STACK, card_id, card_id, card_name=reveal)
            p.stack.append({"id": card_id, "name": name})
            return

        if from_zone == "stack" and to_zone is None:
            # resolution marker only -- no destination given. Deferred: claimed by a
            # battlefield entry (_resolve_incoming_permanent) or a same-phase recast
            # (_resolve_incoming_hand_like, e.g. Flashback) for the same name, else
            # flushed to graveyard at the next phase boundary (see
            # _flush_pending_resolutions).
            p = self.players[e["active_idx"]]
            c = pop_by_name(p.stack, name)
            if c:
                p.pending_resolution.append(c)
            return

        p = self.players[e["active_idx"]]
        # (Pre-game mulligan put-backs to the library are consumed in __init__ and
        # never reach here; a to-library move in-game is real play -- a Lembas-style
        # graveyard shuffle, a Brainstorm put-back, etc. -- and is rendered below.)

        if from_zone == "hand":
            card_id, reveal = self._resolve_hand_card(p, name)
            src_zone_cockatrice = Z_HAND
        else:
            src_pool = {"graveyard": p.graveyard, "exile": p.exile, "stack": p.stack}.get(from_zone)
            c = pop_by_name(src_pool, name) if src_pool is not None else None
            if c is not None:
                card_id = c["id"]
                src_zone_cockatrice = JSON_ZONE_TO_COCKATRICE[from_zone]
                reveal = None
            else:
                # never actually tracked in that zone -- the client has no real CardItem
                # sitting there to find, so it must come from the hidden deck zone
                # instead, revealing its name for the first time right now.
                card_id = self._mint_from_library(p)
                src_zone_cockatrice = Z_DECK
                reveal = name

        dst_zone = JSON_ZONE_TO_COCKATRICE.get(to_zone)
        if dst_zone is None:
            return

        cont = self._new_container()
        self._move(cont, p, src_zone_cockatrice, p, dst_zone, card_id, card_id, card_name=reveal)
        dst_pool = {"hand": p.hand, "graveyard": p.graveyard, "exile": p.exile}.get(to_zone)
        if dst_pool is not None:
            dst_pool.append({"id": card_id, "name": name})
        elif to_zone in ("library", "library_bottom"):
            # card put back into the deck -- un-count it so the displayed deck size stays right
            p.library_draws = max(0, p.library_draws - 1)

    def _handle_countered(self, e):
        # A countered spell goes to its owner's graveyard (real Magic 701.5b). The
        # engine records this only as a bare "countered" event (card + controller),
        # never a zone_move, so render the move here: take the card off ITS
        # CONTROLLER's stack (not the counterer's -- controller, not active_idx)
        # into their graveyard. Without this the countered card lingered on the
        # stack and never reached the graveyard, so a graveyard-counting cost
        # reducer (Cryptic Serpent / Tolarian Terror) looked cheaper in the replay
        # than the visible graveyard justified.
        p = self.players[e.get("controller", e["active_idx"])]
        c = pop_by_name(p.stack, e["card"])
        if c is None:
            return
        cont = self._new_container()
        self._move(cont, p, Z_STACK, p, Z_GRAVE, c["id"], c["id"])
        p.graveyard.append({"id": c["id"], "name": c["name"]})

    def _handle_put_on_top(self, e):
        # Brainstorm-style "put N cards from your hand on top of your library."
        # The engine logs this as its own "put_on_top" event (cards=[names], the
        # caster is active_idx), NOT a zone_move -- so without this the cards were
        # never taken back off the replay hand (hand grew by the net every
        # Brainstorm). Render each as hand->deck and un-count it from the shown
        # deck size (it went back into the library). Deck order isn't tracked
        # (the deck is a hidden zone), so "on top" is just "into the deck" here.
        p = self.players[e["active_idx"]]
        moved = [c for c in (pop_by_name(p.hand, name) for name in e.get("cards") or []) if c is not None]
        if not moved:
            return
        cont = self._new_container()
        for c in moved:
            self._move(cont, p, Z_HAND, p, Z_DECK, c["id"], c["id"])
            p.library_draws = max(0, p.library_draws - 1)

    def _find_perm(self, name_slot):
        """Locate a permanent on either player's battlefield by [name, slot];
        effect taps/untaps and orphaned auras can target either side."""
        key = tuple(name_slot)
        for p in self.players:
            if key in p.battlefield:
                return p, key
        return None, None

    def _revert_dfc_face(self, cont, p, entry):
        """A transformed double-faced permanent shows its FRONT face again the
        moment it leaves the battlefield (real Magic 712.4a). Emit an
        Event_FlipCard back to the stored front name so the card reverts before it
        lands in the new zone, and return that front name (for the destination
        pool's tracked name); None if this entry was never a transformed DFC."""
        front = entry.get("front_name")
        if front is None:
            return None
        fc = add_event(cont, p.idx, pb_flip.Event_FlipCard.ext)
        fc.zone_name = Z_TABLE
        fc.card_id = entry["id"]
        fc.card_name = front
        return front

    def _handle_transform(self, e):
        # A double-faced permanent flips to its back face (Delver of Secrets ->
        # Insectile Aberration). The engine logs permanent=(front name, slot) +
        # to_card + the back-face power/toughness. Cockatrice keeps the card's id
        # across a flip, so emit Event_FlipCard to rename the SAME card to the back
        # face, refresh its P/T, and re-key our battlefield entry to the new name so
        # every later event for this permanent (which the engine now logs under the
        # back-face name) still resolves. _revert_dfc_face flips it back on leave.
        from_name, slot = e["permanent"]
        to_card = e.get("to_card")
        if to_card is None:
            return  # older logs logged transform without the back-face name -- nothing to flip to
        p, key = self._find_perm([from_name, slot])
        if key is None:
            return
        entry = p.battlefield.pop(key)
        cont = self._new_container()
        fc = add_event(cont, p.idx, pb_flip.Event_FlipCard.ext)
        fc.zone_name = Z_TABLE
        fc.card_id = entry["id"]
        fc.card_name = to_card
        if e.get("power") is not None and e.get("toughness") is not None:
            self._set_attr(cont, p, Z_TABLE, entry["id"], pb_attr.AttrPT, f'{e["power"]}/{e["toughness"]}')
        entry["front_name"] = from_name
        p.battlefield[(to_card, slot)] = entry

    def _handle_reveal(self, e):
        # Delver's upkeep look: the revealed top-of-library card, shown to both
        # players (Event_RevealCards from the deck zone). The card stays in the
        # library -- this is purely the reveal, no zone change.
        p = self.players[e["active_idx"]]
        name = e.get("card")
        if name is None:
            return
        cont = self._new_container()
        rc = add_event(cont, p.idx, pb_reveal.Event_RevealCards.ext)
        rc.zone_name = Z_DECK
        rc.number_of_cards = 1
        ci = rc.cards.add()
        ci.id = 0
        ci.name = name

    def _mint_and_move_to_graveyard(self, cont, p, names):
        for name in names:
            card_id = self._mint_from_library(p)
            self._move(cont, p, Z_DECK, p, Z_GRAVE, 0, card_id, card_name=name)
            p.graveyard.append({"id": card_id, "name": name})

    def _handle_mill(self, e):
        # Cards milled off the top of the library into the graveyard. Logged as its
        # own event (named cards, count) rather than a zone_move, so without this the
        # graveyard never grew -- and, downstream, delve/flashback/dredge that pull
        # those names couldn't find them and minted phantoms straight from the deck.
        p = self.players[e["player_idx"]]
        names = e.get("cards") or []
        if not names:
            return
        self._mint_and_move_to_graveyard(self._new_container(), p, names)

    def _handle_disposed(self, e):
        # Scry puts disposed cards on the library bottom (disposed_to ==
        # "library_bottom"); surveil sends them to the graveyard. Library-bottom
        # is invisible here (deck order isn't tracked, same as kept_to_library_top),
        # so only the graveyard case needs an actual move.
        if e.get("disposed_to") != "graveyard":
            return
        names = e.get("disposed") or []
        if not names:
            return
        p = self.players[e["active_idx"]]
        self._mint_and_move_to_graveyard(self._new_container(), p, names)

    def _handle_graveyard_exiled(self, e):
        # Graveyard-exile (Bojuka Bog / Cling to Dust-style). Two shapes:
        #   * {target_player_idx, exiled:[names]} -- specific named cards exiled.
        #   * {player_idx}                        -- the whole graveyard, no list.
        # In both cases move the affected cards we're tracking from graveyard to exile.
        if "exiled" in e:
            p = self.players[e["target_player_idx"]]
            cont = self._new_container()
            for name in e["exiled"]:
                c = pop_by_name(p.graveyard, name)
                if c is None:
                    continue  # not tracked in our graveyard model -- nothing on screen to move
                self._move(cont, p, Z_GRAVE, p, Z_EXILE, c["id"], c["id"])
                p.exile.append(c)
            return
        p = self.players[e["player_idx"]]
        if not p.graveyard:
            return
        cont = self._new_container()
        for c in list(p.graveyard):
            self._move(cont, p, Z_GRAVE, p, Z_EXILE, c["id"], c["id"])
            p.exile.append(c)
        p.graveyard.clear()

    def _handle_graveyards_exiled(self, e):
        # Relic of Progenitus-style "exile all graveyards" -- both players' whole
        # graveyards at once, no per-player fields at all (unlike the singular
        # graveyard_exiled above, which always names a player/target).
        cont = self._new_container()
        for p in self.players:
            if not p.graveyard:
                continue
            for c in list(p.graveyard):
                self._move(cont, p, Z_GRAVE, p, Z_EXILE, c["id"], c["id"])
                p.exile.append(c)
            p.graveyard.clear()

    def _handle_impulse_exile(self, e):
        # "Impulse draw": exile the top card(s) of the library face-up, playable for
        # a while. Render as library->exile so they show in the exile zone; if one is
        # later played its from_zone is exile and _resolve_* finds it there. The
        # matching impulse_expired is a no-op -- an unplayed impulse card physically
        # stays in exile, it just stops being castable.
        p = self.players[e["active_idx"]]
        names = e.get("cards") or []
        if not names:
            return
        cont = self._new_container()
        for name in names:
            card_id = self._mint_from_library(p)
            self._move(cont, p, Z_DECK, p, Z_EXILE, 0, card_id, card_name=name)
            p.exile.append({"id": card_id, "name": name})

    def _set_tapped(self, e, tapped):
        p, key = self._find_perm(e["permanent"])
        if key is None:
            return
        entry = p.battlefield[key]
        if entry["tapped"] == tapped:
            return
        cont = self._new_container()
        self._set_attr(cont, p, Z_TABLE, entry["id"], pb_attr.AttrTapped, "1" if tapped else "0")
        entry["tapped"] = tapped

    def _handle_tap(self, e):
        self._set_tapped(e, True)

    def _handle_untap(self, e):
        self._set_tapped(e, False)

    def _handle_tap_or_untap(self, e):
        self._set_tapped(e, bool(e.get("now_tapped")))

    def _handle_tuck(self, e):
        # Put a permanent into its owner's library (Deem Inferior-style). No
        # accompanying zone_move is logged, so without this the tucked permanent
        # lingered on the battlefield (and got redrawn as a second copy). Deck order
        # isn't modeled (hidden zone), so "top2" is just "back into the deck" here.
        p = self.players[e.get("owner_idx", e["active_idx"])]
        key = next((k for k in p.battlefield if k[0] == e["card"]), None)
        if key is None:
            return
        old = p.battlefield.pop(key)
        cont = self._new_container()
        if old.get("is_token"):
            d = add_event(cont, p.idx, pb_destroy.Event_DestroyCard.ext)
            d.zone_name = Z_TABLE
            d.card_id = old["id"]
            return
        self._move(cont, p, Z_TABLE, p, Z_DECK, old["id"], old["id"])
        p.library_draws = max(0, p.library_draws - 1)

    def _handle_destroy(self, e):
        # A destroy logged as its own event (permanent + owner_idx + to_zone) rather
        # than a zone_move / state_based_death. Route it off the battlefield to the
        # given zone, crediting the permanent's OWNER (active_idx is the destroyer).
        p = self.players[e.get("owner_idx", e["active_idx"])]
        name, slot = e["permanent"]
        old = p.battlefield.pop((name, slot), None)
        if old is None:
            return
        cont = self._new_container()
        if old.get("is_token"):
            d = add_event(cont, p.idx, pb_destroy.Event_DestroyCard.ext)
            d.zone_name = Z_TABLE
            d.card_id = old["id"]
            return
        dst_zone = JSON_ZONE_TO_COCKATRICE.get(_norm_zone(e.get("to_zone")), Z_GRAVE)
        self._move(cont, p, Z_TABLE, p, dst_zone, old["id"], old["id"])
        pool = {Z_GRAVE: p.graveyard, Z_EXILE: p.exile, Z_HAND: p.hand}.get(dst_zone)
        if pool is not None:
            pool.append({"id": old["id"], "name": name})

    def _handle_aura_orphaned(self, e):
        # The creature an aura was enchanting left, so the aura falls off. No
        # zone_move is logged for the aura; its fate is in `outcome` (Rancor -> hand,
        # most auras -> graveyard). Without this the aura stayed stuck on the now-empty
        # slot on the battlefield.
        p, key = self._find_perm(e["aura"])
        if key is None:
            return
        old = p.battlefield.pop(key)
        cont = self._new_container()
        if old.get("is_token"):
            d = add_event(cont, p.idx, pb_destroy.Event_DestroyCard.ext)
            d.zone_name = Z_TABLE
            d.card_id = old["id"]
            return
        outcome = e.get("outcome", "graveyard")
        dst_zone = {"hand": Z_HAND, "exile": Z_EXILE}.get(outcome, Z_GRAVE)
        self._move(cont, p, Z_TABLE, p, dst_zone, old["id"], old["id"])
        pool = {Z_GRAVE: p.graveyard, Z_EXILE: p.exile, Z_HAND: p.hand}.get(dst_zone)
        if pool is not None:
            pool.append({"id": old["id"], "name": key[0]})

    _HANDLERS = {
        "turn_start": _handle_turn_start,
        "phase_change": _handle_phase_change,
        "zone_move": _handle_zone_move,
        "mana_tap": _handle_mana_tap,
        "mana_spend": _handle_mana_spend,
        "untap_step": _handle_untap_step,
        "mana_emptied": _handle_mana_emptied,
        "attack_declared": _handle_attack_declared,
        "block_assigned": _handle_block_assigned,
        "aura_attached": _handle_aura_attached,
        "aura_orphaned": _handle_aura_orphaned,
        "life_change": _handle_life_change,
        "state_based_death": _handle_state_based_death,
        "destroy": _handle_destroy,
        "countered": _handle_countered,
        "put_on_top": _handle_put_on_top,
        "mill": _handle_mill,
        "graveyard_exiled": _handle_graveyard_exiled,
        "graveyards_exiled": _handle_graveyards_exiled,
        "impulse_exile": _handle_impulse_exile,
        "tap": _handle_tap,
        "untap": _handle_untap,
        "tap_or_untap": _handle_tap_or_untap,
        "tuck": _handle_tuck,
        "transform": _handle_transform,
        "reveal": _handle_reveal,
    }


def build_replay_for_game(game, meta):
    if "events" in game:
        rb = EventStreamReplayBuilder(game, meta)
        rb.process_events()
    else:
        rb = ReplayBuilder(game, meta)
        rb.process_steps()
    return rb.replay


PAIRING_LINE_RE = re.compile(r"^\s*(\S+) vs (\S+): (\d+) games$")


def _pairing_labels_from_log(json_path, expected_total):
    """run_league.py's --eval round-robin path (no --decks/--matchup override)
    writes meta.decks/matchup as None even though a real deck roster drove the
    run: _write_event_log logs the raw, unresolved CLI args instead of the
    roster _run_eval actually resolved and played (see that function's own
    docstring). Recover a label per game_index from the sibling .log file's
    "A vs B: N games" lines instead -- printed in the exact pairing order
    games were appended in, so slicing by count reconstructs it exactly.
    Returns a list of length expected_total, or None if no sibling .log
    exists or its game count doesn't match (stale/foreign .log)."""
    log_path = json_path.with_suffix(".log")
    if not log_path.exists():
        return None
    labels = []
    for line in log_path.read_text(encoding="utf-8").splitlines():
        m = PAIRING_LINE_RE.match(line)
        if m:
            a, b, n = m.group(1), m.group(2), int(m.group(3))
            labels.extend([f"{a}_vs_{b}"] * n)
    return labels if len(labels) == expected_total else None


def _resolve_config_name(meta, pairing_labels, game_index):
    """The single label used for both the in-replay description (read by
    BaseReplayBuilder via meta["config_name"]) and the output filename."""
    if meta.get("config_name"):
        return meta["config_name"]
    matchup = meta.get("matchup")
    if matchup:
        return f"{matchup[0]}_vs_{matchup[1]}"
    if meta.get("deck_a"):
        return f'{meta["deck_a"]}_vs_{meta.get("deck_b", "sim")}'
    if pairing_labels is not None and game_index < len(pairing_labels):
        return pairing_labels[game_index]
    return "sim_vs_sim"


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("json_path", help="path to the simulator's JSON game log")
    ap.add_argument("output_dir", help="directory to write .cor replay files into")
    ap.add_argument("--game", type=int, help="only convert this game_index")
    args = ap.parse_args()

    json_path = Path(args.json_path)
    with open(json_path, encoding="utf-8") as f:
        data = json.load(f)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    meta = data["meta"]
    pairing_labels = None
    if not (meta.get("config_name") or meta.get("matchup") or meta.get("deck_a")):
        pairing_labels = _pairing_labels_from_log(json_path, len(data["games"]))

    count = 0
    for game in data["games"]:
        if args.game is not None and game["game_index"] != args.game:
            continue
        name_prefix = _resolve_config_name(meta, pairing_labels, game["game_index"])
        replay = build_replay_for_game(game, {**meta, "config_name": name_prefix})
        out_path = out_dir / f'{name_prefix}_game{game["game_index"]}.cor'
        out_path.write_bytes(replay.SerializeToString())
        print(
            f"wrote {out_path} ({len(replay.event_list)} event containers, "
            f"{out_path.stat().st_size} bytes)"
        )
        count += 1

    if count == 0:
        sys.exit("no games converted (check --game filter)")


if __name__ == "__main__":
    main()
