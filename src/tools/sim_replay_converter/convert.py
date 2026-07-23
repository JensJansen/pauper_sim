#!/usr/bin/env python3
"""
Convert MTG simulator JSON game logs (boggles_mirror-style) into Cockatrice
.cor replay files, so the simulator's games can be watched in Cockatrice's
built-in replay viewer.

The JSON format is a full-state snapshot per decision step (state_after),
not the atomic diff-event stream Cockatrice replays expect. This script
diffs consecutive snapshots to reconstruct the MoveCard/SetCardAttr/etc.
event stream a live Cockatrice server would have produced.

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
import event_delete_arrow_pb2 as pb_deletearrow  # noqa: E402
import event_destroy_card_pb2 as pb_destroy  # noqa: E402
import event_draw_cards_pb2 as pb_draw  # noqa: E402
import event_game_state_changed_pb2 as pb_gsc  # noqa: E402
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

# Battlefield rows (table y-coordinate). The JSON doesn't tag card type, so lands are
# inferred: no power/toughness (not a creature) and not itself attached to anything
# (auras/equipment always have "enchanting" set, even the ones enchanting a land) --
# true for every land in the boggles_mirror/Pauper-style decks this was built against.
ROW_CREATURE = 0
ROW_OTHER = 1
ROW_LAND = 2


def parse_slot_key(s):
    m = SLOT_RE.match(s)
    if not m:
        raise ValueError(f"unexpected slot-key format: {s!r}")
    return (m.group(1), int(m.group(2)))


def bf_key(card):
    return (card["name"], card["slot"])


def battlefield_row(card):
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
        self.battlefield = {}  # (name, slot) -> dict(id, x, y, tapped, power, toughness, keywords, enchanting)
        self.next_row_x = {ROW_LAND: 0, ROW_CREATURE: 0, ROW_OTHER: 0}
        self.library_draws = 0  # total cards ever pulled from the library this game
        self.counter_ids = {"life": 0, "w": 1, "u": 2, "b": 3, "r": 4, "g": 5, "x": 6}
        self.next_counter_id = 8
        self.deck_zone_info = None  # live reference, card_count patched once library_draws is final

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


class ReplayBuilder:
    def __init__(self, game, meta):
        self.game = game
        self.replay = pb_replay.GameReplay()
        self.replay.replay_id = game["game_index"]
        info = self.replay.game_info
        info.game_id = game["game_index"]
        info.description = f'{meta.get("config_name", "sim")} - game {game["game_index"]}'
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
        self.step_seconds = 0

        self._emit_initial_snapshot()

    def _new_container(self):
        cont = self.replay.event_list.add()
        cont.seconds_elapsed = self.step_seconds
        self.step_seconds += 1
        return cont

    def other(self, p):
        return self.players[1 - p.idx]

    # ------------------------------------------------------------------
    # Initial snapshot: one Event_GameStateChanged carrying full opening
    # board state, exactly like a spectator joining an in-progress game.
    # ------------------------------------------------------------------
    def _emit_initial_snapshot(self):
        cont = self._new_container()
        gsc = add_event(cont, None, pb_gsc.Event_GameStateChanged.ext)
        gsc.game_started = True
        gsc.active_player_id = self.game["starting_player_idx"]
        gsc.active_phase = 0
        gsc.seconds_elapsed = 0
        self.active_player = self.game["starting_player_idx"]
        self.active_phase = 0

        opening = self.game["opening_state"]["players"]
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

            life = opening[p.idx].get("life_total", 20)
            for cname, cval in [("life", life), ("w", 0), ("u", 0), ("b", 0), ("r", 0), ("g", 0), ("x", 0)]:
                cnt = pinfo.counter_list.add()
                cnt.id = p.counter_ids[cname]
                cnt.name = cname
                cnt.count = cval

        # Draw the opening hand the same way a live server does: game starts with
        # hand=0/deck=full (above), then each player shuffles and draws their opener.
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

    def _draw_cards(self, cont, p, names):
        drawn = []
        for name in names:
            card_id = p.new_id()
            p.library_draws += 1
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
                if phase_idx != PHASE_BY_NAME["declare_blockers"] and self.live_arrow_ids:
                    # DELETE_ARROW isn't one of GameEventHandler's specially-routed event
                    # types -- it falls through to the generic per-player dispatch, which
                    # requires event.player_id() to resolve to a real PlayerLogic (unlike
                    # SET_ACTIVE_PLAYER/SET_ACTIVE_PHASE, -1/unset gets silently dropped).
                    for arrow_id in self.live_arrow_ids:
                        da = add_event(cont, self.players[0].idx, pb_deletearrow.Event_DeleteArrow.ext)
                        da.arrow_id = arrow_id
                    self.live_arrow_ids = []

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
                card_id = p.new_id()
                p.library_draws += 1
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
                ca = add_event(cont, p.idx, pb_createarrow.Event_CreateArrow.ext)
                arrow_id = self.next_arrow_id
                self.next_arrow_id += 1
                ai = ca.arrow_info
                ai.id = arrow_id
                ai.start_player_id = p.idx
                ai.start_zone = Z_TABLE
                ai.start_card_id = p.battlefield[key]["id"]
                ai.target_player_id = self.other(p).idx
                ai.arrow_color.r, ai.arrow_color.g, ai.arrow_color.b = 255, 0, 0
                self.live_arrow_ids.append(arrow_id)
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
                ca = add_event(cont, opp.idx, pb_createarrow.Event_CreateArrow.ext)
                arrow_id = self.next_arrow_id
                self.next_arrow_id += 1
                ai = ca.arrow_info
                ai.id = arrow_id
                ai.start_player_id = opp.idx
                ai.start_zone = Z_TABLE
                ai.start_card_id = opp.battlefield[blocker_key]["id"]
                ai.target_player_id = p.idx
                ai.target_zone = Z_TABLE
                ai.target_card_id = p.battlefield[attacker_key]["id"]
                ai.arrow_color.r, ai.arrow_color.g, ai.arrow_color.b = 255, 0, 0
                self.live_arrow_ids.append(arrow_id)

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
                card_id = p.new_id()
                p.library_draws += 1
                self._move(cont, p, Z_DECK, p, Z_GRAVE, 0, card_id, card_name=name)
                p.graveyard.append({"id": card_id, "name": name})
        for name, cnt in list(exile_added.items()):
            for _ in range(cnt):
                card_id = p.new_id()
                p.library_draws += 1
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


def build_replay_for_game(game, meta):
    rb = ReplayBuilder(game, meta)
    rb.process_steps()
    return rb.replay


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("json_path", help="path to the simulator's JSON game log")
    ap.add_argument("output_dir", help="directory to write .cor replay files into")
    ap.add_argument("--game", type=int, help="only convert this game_index")
    args = ap.parse_args()

    with open(args.json_path, encoding="utf-8") as f:
        data = json.load(f)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    meta = data["meta"]
    count = 0
    for game in data["games"]:
        if args.game is not None and game["game_index"] != args.game:
            continue
        replay = build_replay_for_game(game, meta)
        out_path = out_dir / f'{meta.get("config_name", "sim")}_game{game["game_index"]}.cor'
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
