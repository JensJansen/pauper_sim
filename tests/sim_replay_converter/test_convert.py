"""Regression tests for the event-stream replay converter.

Guards two draw-handling invariants:
  1. draws must not be double-counted (no phantom phase-inferred draw on top
     of the real logged library->hand draw), so hand counts stay exactly what
     the log accounts for.
  2. every mulligan redraw must net against its put-backs rather than piling
     up permanently in hand or inflating the shown deck size.

test_converter_matches_engine_log_hand_counts rebuilds the replay for every
logs/*.json (if any are present) and asserts the cards the converter puts into
each hand exactly equal what the engine log accounts for: the net kept opening
hand (draws minus mulligan put-backs, before turn 1) plus every in-game entry
into hand.

The other tests cover three engine event shapes (surveil/scry's "disposed"
zone_move shape, "mana_emptied" zeroing a player's tracked mana pool, and the
plural "graveyards_exiled") and one converter-only invariant (an attacker's
AttrAttacking flag must clear once combat moves past declare_blockers, not
stay set for the rest of the game), with small self-contained fabricated-event
checks that don't depend on any log file being present.
"""
from __future__ import annotations

import glob
import json
from collections import Counter
from pathlib import Path

import pytest

# convert.py's own module-level bootstrap_pb2() (relative to ITS OWN __file__,
# in src/sim_replay_converter/) generates/caches the Cockatrice protobuf
# bindings and inserts that cache dir onto sys.path as a side effect -- so the
# bare pb2 imports below resolve correctly regardless of where this test file
# lives, no sys.path hack of our own needed (sim_replay_converter/ has no
# __init__.py, but src/ being on pythonpath via pyproject.toml makes it an
# importable implicit namespace package).
from sim_replay_converter import convert as C
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


def test_surveil_disposed_to_graveyard():
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


def test_mana_emptied_zeroes_pool():
    """mana_emptied (game/turn.py's _empty_mana_pools) is the event this
    converter reads to zero a tracked mana pool counter. Without a handler for
    it, a tapped source's mana counter would stay visibly nonzero forever."""
    events = [
        _envelope("turn_start", phase=None),
        _envelope("mana_tap", permanent=["Mountain", 1], mode="tap", produced=["R"]),
        _envelope("mana_emptied", phase="main2", pools={"0": {"R": 1}}),
    ]
    rb = C.EventStreamReplayBuilder({"game_index": 0, "events": events}, {})
    rb.process_events()
    assert rb.players[0].mana_pool.get("R", 0) == 0, "mana_emptied must zero the tracked pool"


def test_graveyards_exiled_clears_both_players():
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


def test_attack_flag_clears_after_combat():
    """AttrAttacking is set "1" on attack_declared and must clear once combat
    phase moves on, mirroring _clear_arrows' lifecycle (stays through
    declare_blockers, clears after) -- otherwise a creature that ever attacked
    would stay visually flagged as attacking for the rest of the game."""
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


def test_same_phase_flashback_reuses_resolved_card():
    """A card recast (Flashback) in the SAME phase it just resolved in -- the
    engine logs stack->None "resolve" (no explicit destination; see
    game.effects.stack.resolve_top_of_stack) followed later by a phase
    boundary that would flush it to the graveyard, but a same-phase recast
    can fire first. _resolve_incoming_hand_like must also check this
    same-phase pending-resolution pool (not just hand/exile) and reuse the
    resolved card's id -- otherwise the replay would show TWO Lava Darts (one
    stuck forever on the stack) for what the sim log recorded as one card
    resolving, then being flashed back from the graveyard."""
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


def test_flashback_does_not_steal_a_second_hand_copy():
    """With TWO physical Lava Darts -- one hard-cast+resolved, one still
    genuinely sitting untouched in hand -- from_zone == "hand" must be the
    ONLY thing that makes hand a candidate pool for
    _resolve_incoming_hand_like. Checking p.hand unconditionally (ignoring
    from_zone) would let a graveyard-sourced Flashback (from_zone=None) steal
    the OTHER, still-in-hand copy's id instead of the resolved one waiting in
    pending_resolution/graveyard -- leaving the real graveyard copy unclaimed
    and rendering the replay as if the still-in-hand copy had been "cast" a
    second time while the first was still on the stack, unresolved."""
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


def test_pairing_labels_survive_mismatched_log_basename(tmp_path):
    """Confirmed live: a real subleague run wrote
    subleague_..._3010games.json alongside eval_..._3010games.log -- an
    UNRELATED basename, not the same-stem sibling _pairing_labels_from_log
    used to assume. Every game in that batch silently fell back to
    "sim_vs_sim" as a result. The lookup must also find a same-directory
    .log with a different name, by matching on content (the "event log
    written to <path>" line names the JSON file it belongs to)."""
    json_path = tmp_path / "subleague_run.json"
    log_path = tmp_path / "eval_run.log"  # deliberately NOT subleague_run.log
    log_path.write_text(
        "Eval: 1 pairing(s) x 2 games (sampled, seed=None) over decks=['a', 'b']\n"
        "  a vs b: 2 games\n"
        "eval done: 2 games in 1.0s\n"
        f"event log written to ../logs/{json_path.name} (2 games, 10 total events, 1.0 KB)\n",
        encoding="utf-8",
    )
    labels = C._pairing_labels_from_log(json_path, expected_total=2)
    assert labels == ["a_vs_b", "a_vs_b"]


def test_converter_matches_engine_log_hand_counts():
    """Cross-checks every real logs/*.json (if any are present) against the
    engine's own accounting, on top of the fabricated-event checks above.
    Skips cleanly if no logs exist yet (rerun matches to regenerate)."""
    logs = sorted(glob.glob(str(Path(__file__).resolve().parents[2] / "logs" / "*.json")))
    if not logs:
        pytest.skip("no logs/*.json present (rerun matches to regenerate, then re-run this check)")
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
