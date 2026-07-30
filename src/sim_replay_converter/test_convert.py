#!/usr/bin/env python3
"""Regression check for the event-stream replay converter.

Guards the two draw-handling bugs fixed 2026-07-27:
  1. draws were double-counted (a phantom phase-inferred draw on top of the real
     logged library->hand draw) -> ~2 cards into hand per turn.
  2. every mulligan redraw piled permanently into hand (put-backs were dropped)
     -> "drew 56 immediately", inflated deck size.

For every logs/*.json it rebuilds the replay and asserts the cards the
converter puts into each hand exactly equal what the engine log accounts for:
the net kept opening hand (draws minus mulligan put-backs, before turn 1) plus
every in-game entry into hand.

Also guards three engine-format changes fixed 2026-07-30 (surveil/scry's
"disposed" zone_move shape, "mana_emptied" replacing the old untap_step mana
clear, and the new plural "graveyards_exiled"), and one converter-only bug
fixed the same day (an attacker's AttrAttacking flag was set on
attack_declared but never cleared, so it stayed visually flagged as attacking
for the rest of the game), with small self-contained fabricated-event checks
that don't depend on any log file being present.

No framework -- just asserts. Run: python test_convert.py
"""
from __future__ import annotations

import glob
import json
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import convert as C  # noqa: E402
import card_attributes_pb2 as pb_attr  # noqa: E402
import event_draw_cards_pb2 as pb_draw  # noqa: E402
import event_move_card_pb2 as pb_move  # noqa: E402
import event_set_card_attr_pb2 as pb_setattr  # noqa: E402


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


def _envelope(kind, active_idx=0, turn_player_idx=0, turn=1, phase="main1", **fields):
    return {"kind": kind, "turn": turn, "phase": phase, "active_idx": active_idx,
            "turn_player_idx": turn_player_idx, **fields}


def check_surveil_disposed_to_graveyard():
    """A surveil zone_move carries no "card"/"cards"/"permanent" field at all --
    just "disposed"/"disposed_to"/"kept_to_library_top" -- so it never reached
    the general card/cards/permanent dispatch and silently vanished. A
    disposed_to="graveyard" card must now render as a real deck->grave move."""
    events = [
        _envelope("turn_start", phase=None),
        _envelope("zone_move", phase="upkeep", reason="surveil",
                   kept_to_library_top=[], disposed_to="graveyard", disposed=["Test Card"]),
    ]
    replay = C.build_replay_for_game({"game_index": 0, "events": events}, {})
    moves = [
        ev.Extensions[pb_move.Event_MoveCard.ext]
        for cont in replay.event_list for ev in cont.event_list
        if ev.HasField("player_id")
    ]
    hits = [mc for mc in moves if mc.start_zone == "deck" and mc.target_zone == "grave"
            and mc.card_name == "Test Card"]
    assert len(hits) == 1, f"expected one deck->grave move for the disposed card, got {len(hits)}"
    # disposed_to="library_bottom" (scry) has no visible zone change -- must NOT
    # render as a move (deck order isn't tracked; see kept_to_library_top).
    events[-1]["disposed_to"] = "library_bottom"
    replay = C.build_replay_for_game({"game_index": 0, "events": events}, {})
    assert len(replay.event_list) == 1, "a library_bottom disposal should stay a no-op"


def check_mana_emptied_zeroes_pool():
    """mana_emptied replaced the old untap_step "mana_cleared" field (removed from
    the engine entirely -- see game/turn.py's _empty_mana_pools). Without a
    handler, a tapped source's mana counter stayed visibly nonzero forever."""
    events = [
        _envelope("turn_start", phase=None),
        _envelope("mana_tap", permanent=["Mountain", 1], mode="tap", produced=["R"]),
        _envelope("mana_emptied", phase="main2", pools={"0": {"R": 1}}),
    ]
    rb = C.EventStreamReplayBuilder({"game_index": 0, "events": events}, {})
    rb.process_events()
    assert rb.players[0].mana_pool.get("R", 0) == 0, "mana_emptied must zero the tracked pool"


def check_graveyards_exiled_clears_both_players():
    """Relic of Progenitus's "exile all graveyards" logs a bare graveyards_exiled
    (plural) with no player/card fields, distinct from the singular,
    always-targeted graveyard_exiled -- both graveyards must empty into exile."""
    events = [
        _envelope("turn_start", phase=None),
        _envelope("mill", active_idx=0, player_idx=0, count=1, cards=["Card A"]),
        _envelope("mill", active_idx=1, player_idx=1, count=1, cards=["Card B"]),
        _envelope("graveyards_exiled"),
    ]
    rb = C.EventStreamReplayBuilder({"game_index": 0, "events": events}, {})
    rb.process_events()
    assert rb.players[0].graveyard == [] and rb.players[1].graveyard == []
    assert any(c["name"] == "Card A" for c in rb.players[0].exile)
    assert any(c["name"] == "Card B" for c in rb.players[1].exile)


def check_attack_flag_clears_after_combat():
    """AttrAttacking was set "1" on attack_declared but never reset -- a creature
    that ever attacked stayed visually flagged as attacking for the rest of the
    game (confirmed live: three Silhana Ledgewalkers looked "stuck" in a boggles
    mirror). Must clear once combat phase moves on, mirroring _clear_arrows'
    lifecycle (stays through declare_blockers, clears after)."""
    events = [
        _envelope("turn_start", phase=None),
        _envelope("zone_move", phase="main1", permanent=["Test Creature", 1],
                   from_zone="hand", to_zone="battlefield", tapped=False,
                   card_type="CREATURE", power=1, toughness=1),
        _envelope("attack_declared", phase="declare_attackers", attacker=["Test Creature", 1], tapped=True),
        _envelope("phase_change", phase="declare_blockers", from_phase="declare_attackers"),
        _envelope("phase_change", phase="combat_damage", from_phase="declare_blockers"),
    ]
    replay = C.build_replay_for_game({"game_index": 0, "events": events}, {})
    flag_values = [
        ev.Extensions[pb_setattr.Event_SetCardAttr.ext].attr_value
        for cont in replay.event_list for ev in cont.event_list
        if ev.HasField("player_id") and ev.Extensions[pb_setattr.Event_SetCardAttr.ext].attribute == pb_attr.AttrAttacking
    ]
    assert flag_values == ["1", "0"], f"expected the attacking flag to set then clear, got {flag_values}"


def check_same_phase_flashback_reuses_resolved_card():
    """A card recast (Flashback) in the SAME phase it just resolved in -- the
    engine logs stack->None "resolve" (no explicit destination; see
    game.effects.stack.resolve_top_of_stack) followed later by a phase
    boundary that would flush it to the graveyard, but a same-phase recast
    can fire first. _resolve_incoming_hand_like used to only check hand/exile
    and, finding no match, minted a brand-new phantom copy from the library --
    so the replay showed TWO Lava Darts (one stuck forever on the stack) for
    what the sim log recorded as one card resolving, then being flashed back
    from the graveyard. Must instead reuse the same card id."""
    events = [
        _envelope("turn_start", phase=None),
        _envelope("zone_move", card="Lava Dart", from_zone="hand", to_zone="stack", controller=0),
        _envelope("zone_move", card="Lava Dart", from_zone="stack", reason="resolve"),
        _envelope("zone_move", card="Lava Dart", from_zone=None, to_zone="stack", controller=0),
        _envelope("zone_move", card="Lava Dart", from_zone="stack", to_zone="exile", reason="flashback"),
    ]
    replay = C.build_replay_for_game({"game_index": 0, "events": events}, {})
    moves = [
        ev.Extensions[pb_move.Event_MoveCard.ext]
        for cont in replay.event_list for ev in cont.event_list
        if ev.HasField("player_id")
    ]
    to_stack = [mc for mc in moves if mc.target_zone == "stack"]
    assert len(to_stack) == 2, f"expected two hand->stack/graveyard->stack moves, got {len(to_stack)}"
    assert to_stack[0].card_id == to_stack[1].card_id, (
        "the flashback recast must reuse the resolved card's own id, not mint a phantom second copy"
    )
    to_exile = [mc for mc in moves if mc.target_zone == C.Z_EXILE]
    assert len(to_exile) == 1 and to_exile[0].card_id == to_stack[0].card_id, (
        "the flashback resolution must exile the SAME card that was cast, not an orphaned stack entry"
    )


def check_flashback_does_not_steal_a_second_hand_copy():
    """The real bug behind the mono_red_madness_vs_monster_tron report: with
    TWO physical Lava Darts -- one hard-cast+resolved, one still genuinely
    sitting untouched in hand -- _resolve_incoming_hand_like used to check
    p.hand unconditionally (ignoring from_zone), so the graveyard-sourced
    Flashback (from_zone=None) grabbed the OTHER, still-in-hand copy's id
    instead of the resolved one waiting in pending_resolution/graveyard. That
    left the real graveyard copy's identity untouched (never claimed, so it
    would eventually phantom-flush to the graveyard on its own) and rendered
    the replay as if the still-in-hand copy had been "cast" a second time
    while the first was still on the stack, unresolved -- exactly what was
    reported. from_zone == "hand" must be the ONLY thing that makes hand a
    candidate pool."""
    events = [
        _envelope("turn_start", phase=None),
        _envelope("zone_move", cards=["Lava Dart", "Lava Dart"], from_zone="library", to_zone="hand", reason="draw"),
        _envelope("zone_move", card="Lava Dart", from_zone="hand", to_zone="stack", controller=0),
        _envelope("zone_move", card="Lava Dart", from_zone="stack", reason="resolve"),
        _envelope("zone_move", card="Lava Dart", from_zone=None, to_zone="stack", controller=0),
        _envelope("zone_move", card="Lava Dart", from_zone="stack", to_zone="exile", reason="flashback"),
    ]
    rb = C.EventStreamReplayBuilder({"game_index": 0, "events": events}, {})
    rb.process_events()
    p = rb.players[0]
    assert [c["name"] for c in p.hand] == ["Lava Dart"], (
        f"the untouched second copy must still be sitting in hand, got {p.hand!r}"
    )
    assert p.stack == [] and p.pending_resolution == [] and p.graveyard == [], (
        "the cast+flashbacked copy must be fully exiled -- nothing left on the stack, "
        f"pending, or in the graveyard, got stack={p.stack!r} pending={p.pending_resolution!r} "
        f"graveyard={p.graveyard!r}"
    )


def main():
    check_surveil_disposed_to_graveyard()
    check_mana_emptied_zeroes_pool()
    check_graveyards_exiled_clears_both_players()
    check_attack_flag_clears_after_combat()
    check_same_phase_flashback_reuses_resolved_card()
    check_flashback_does_not_steal_a_second_hand_copy()
    print("OK: surveil/scry disposal, mana_emptied, graveyards_exiled, attack-flag, and flashback self-checks passed")

    logs = sorted(glob.glob(str(Path(__file__).resolve().parents[2] / "logs" / "*.json")))
    if not logs:
        print("SKIP: no logs/*.json present (rerun matches to regenerate, then re-run this check)")
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
            assert got == exp, f"{Path(log).name} game {g['game_index']}: into-hand {got} != engine-accounted {exp}"
            checked += 1
    print(f"OK: converter hand-draw counts match the engine log for {checked} games")


if __name__ == "__main__":
    main()
