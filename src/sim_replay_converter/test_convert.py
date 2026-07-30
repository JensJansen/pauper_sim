#!/usr/bin/env python3
"""Regression check for the event-stream replay converter's draw handling.

Guards the two bugs fixed 2026-07-27:
  1. draws were double-counted (a phantom phase-inferred draw on top of the real
     logged library->hand draw) -> ~2 cards into hand per turn.
  2. every mulligan redraw piled permanently into hand (put-backs were dropped)
     -> "drew 56 immediately", inflated deck size.

For every logs/match_*.json it rebuilds the replay and asserts the cards the
converter puts into each hand exactly equal what the engine log accounts for:
the net kept opening hand (draws minus mulligan put-backs, before turn 1) plus
every in-game entry into hand. No framework -- just asserts. Run: python test_convert.py
"""
from __future__ import annotations

import glob
import json
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import convert as C  # noqa: E402
import event_draw_cards_pb2 as pb_draw  # noqa: E402
import event_move_card_pb2 as pb_move  # noqa: E402


def converter_into_hand(replay):
    into = {0: 0, 1: 0}
    for cont in replay.event_list:
        for ev in cont.event_list:
            mc = ev.Extensions[pb_move.Event_MoveCard.ext]
            if mc.HasField("target_zone") and mc.target_zone == "hand":
                into[mc.start_player_id] += 1
            dc = ev.Extensions[pb_draw.Event_DrawCards.ext]
            if dc.number > 0:
                into[ev.player_id] += dc.number
    return into


def engine_expected(events):
    first_turn = next(i for i, e in enumerate(events) if e["kind"] == "turn_start")
    kept = {0: Counter(), 1: Counter()}
    for e in events[:first_turn]:
        if e["kind"] != "zone_move":
            continue
        n = e.get("cards") or ([e["card"]] if e.get("card") else [])
        if e.get("reason") == "draw":
            kept[e["active_idx"]] += Counter(n)
        elif e.get("reason") in ("mulligan_take", "mulligan_bottom"):
            kept[e["active_idx"]] -= Counter(n)
    exp = {p: sum(kept[p].values()) for p in (0, 1)}
    for e in events[first_turn:]:
        # An orphaned aura with outcome "hand" (Rancor) returns to its controller's
        # hand -- the converter emits a real into-hand move for it, so account for it.
        if e["kind"] == "aura_orphaned" and e.get("outcome") == "hand":
            exp[e["active_idx"]] += 1
            continue
        if e["kind"] != "zone_move" or e.get("to_zone") != "hand":
            continue
        n = e.get("cards") or ([e["card"]] if e.get("card") else []) or ([e["permanent"]] if e.get("permanent") else [])
        exp[e["active_idx"]] += len(n)
    return exp


def main():
    logs = sorted(glob.glob(str(Path(__file__).resolve().parents[2] / "logs" / "match_*.json")))
    if not logs:
        print("SKIP: no logs/match_*.json present (rerun matches to regenerate, then re-run this check)")
        return
    checked = 0
    for log in logs:
        d = json.load(open(log, encoding="utf-8"))
        for g in d["games"]:
            if "events" not in g:
                continue  # snapshot-diff format: not this converter path
            replay = C.build_replay_for_game(g, d["meta"])
            got = converter_into_hand(replay)
            exp = engine_expected(g["events"])
            assert got == exp, f"{Path(log).name}: into-hand {got} != engine-accounted {exp}"
            checked += 1
    print(f"OK: converter hand-draw counts match the engine log for {checked} games")


if __name__ == "__main__":
    main()
