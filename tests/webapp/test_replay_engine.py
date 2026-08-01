"""Self-check for the replay viewer's event-stream reducer (webapp/replay_engine.py).

Fabricated event logs, same style as tests/sim_replay_converter/test_convert.py's
fabricated-event checks (mirrors real GameState.log_event field shapes, confirmed
against that module's already-working, log-tested code) -- exercises the pieces
most likely to silently misrender: mulligan netting, cast-then-resolve, mana,
combat flag lifecycle, a countered spell, a state-based death, and the
same-phase-recast identity fix ported from convert.py's Lava Dart bug fix.
"""
from webapp.replay_engine import GameReducer, list_games, reduce_game


def _ev(kind, active_idx=0, turn_player_idx=0, turn=1, phase="main1", **fields):
    return {"kind": kind, "turn": turn, "phase": phase, "active_idx": active_idx,
            "turn_player_idx": turn_player_idx, **fields}


def test_mulligan_nets_to_single_kept_hand():
    events = [
        _ev("zone_move", phase=None, cards=["A", "B", "C"], from_zone="library", to_zone="hand", reason="draw"),
        _ev("zone_move", phase=None, cards=["A", "B", "C"], from_zone="hand", to_zone="library", reason="mulligan_bottom"),
        _ev("zone_move", phase=None, cards=["A", "B", "D"], from_zone="library", to_zone="hand", reason="draw"),
        _ev("zone_move", phase=None, cards=["A"], from_zone="hand", to_zone="library", reason="mulligan_bottom"),
        _ev("turn_start", phase=None),
    ]
    steps = GameReducer(events).run()
    assert steps[0]["kind"] == "opening_hands"
    assert sorted(steps[0]["players"][0]["hand"]) == ["B", "D"]
    assert "mulliganed 1x" in steps[0]["description"]


def test_cast_and_resolve_to_battlefield():
    events = [
        _ev("turn_start", phase=None),
        _ev("zone_move", phase="main1", card="Grizzly Bears", from_zone="hand", to_zone="stack", controller=0),
        _ev("zone_move", phase="main1", card="Grizzly Bears", from_zone="stack", reason="resolve"),
        _ev("zone_move", phase="main1", permanent=["Grizzly Bears", 0], from_zone="stack", to_zone="battlefield",
            tapped=False, card_type="CREATURE", power=2, toughness=2),
    ]
    steps = GameReducer(events).run()
    p0 = steps[-1]["players"][0]
    assert p0["hand"] == []
    assert steps[-1]["stack"] == []
    assert p0["battlefield"] == [
        {"name": "Grizzly Bears", "slot": 0, "tapped": False, "power": 2, "toughness": 2,
         "is_token": False, "card_type": "CREATURE", "attacking": False, "blocking": None}
    ]


def test_mana_tap_and_spend():
    events = [
        _ev("turn_start", phase=None),
        _ev("zone_move", phase="main1", permanent=["Forest", 0], from_zone="hand", to_zone="battlefield",
            tapped=False, card_type="LAND"),
        _ev("mana_tap", phase="main1", permanent=["Forest", 0], produced=["G"]),
        _ev("mana_spend", phase="main1", color="G"),
    ]
    steps = GameReducer(events).run()
    assert steps[-2]["players"][0]["mana_pool"]["G"] == 1
    assert steps[-2]["players"][0]["battlefield"][0]["tapped"] is True
    assert steps[-1]["players"][0]["mana_pool"]["G"] == 0


def test_attack_flag_clears_after_declare_blockers():
    events = [
        _ev("turn_start", phase=None),
        _ev("zone_move", phase="main1", permanent=["Grizzly Bears", 0], from_zone="hand", to_zone="battlefield",
            tapped=False, card_type="CREATURE", power=2, toughness=2),
        _ev("attack_declared", phase="declare_attackers", attacker=["Grizzly Bears", 0], tapped=True),
        _ev("phase_change", phase="declare_blockers"),
        _ev("phase_change", phase="combat_damage"),
    ]
    steps = GameReducer(events).run()
    by_kind = {s["kind"]: s for s in steps}  # each kind appears once in this fixture
    attacking_after_declare = by_kind["attack_declared"]["players"][0]["battlefield"][0]["attacking"]
    # two phase_change steps share a kind -- combat_damage is the last step overall
    attacking_at_blockers = steps[-2]["players"][0]["battlefield"][0]["attacking"]
    attacking_after_damage = steps[-1]["players"][0]["battlefield"][0]["attacking"]
    assert (attacking_after_declare, attacking_at_blockers, attacking_after_damage) == (True, True, False)


def test_countered_spell_goes_to_controllers_graveyard():
    events = [
        _ev("turn_start", phase=None),
        _ev("zone_move", phase="main1", card="Lightning Bolt", from_zone="hand", to_zone="stack", controller=0),
        _ev("countered", phase="main1", card="Lightning Bolt", controller=0),
    ]
    steps = GameReducer(events).run()
    assert steps[-1]["stack"] == []
    assert [c for c in steps[-1]["players"][0]["graveyard"]] == ["Lightning Bolt"]


def test_state_based_death_moves_to_graveyard():
    events = [
        _ev("turn_start", phase=None),
        _ev("zone_move", phase="main1", permanent=["Grizzly Bears", 0], from_zone="hand", to_zone="battlefield",
            tapped=False, card_type="CREATURE", power=2, toughness=2),
        _ev("state_based_death", phase="combat_damage", owner_idx=0, permanent=["Grizzly Bears", 0]),
    ]
    steps = GameReducer(events).run()
    p0 = steps[-1]["players"][0]
    assert p0["battlefield"] == []
    assert p0["graveyard"] == ["Grizzly Bears"]


def test_same_phase_flashback_does_not_duplicate_card():
    """Ported identity-tracking check: convert.py's real Lava Dart bug (see
    its test_same_phase_flashback_reuses_resolved_card) -- a card recast in
    the SAME phase it just resolved must reuse that resolution, not spawn a
    phantom second copy stuck on the stack."""
    events = [
        _ev("turn_start", phase=None),
        _ev("zone_move", phase="main1", card="Lava Dart", from_zone="hand", to_zone="stack", controller=0),
        _ev("zone_move", phase="main1", card="Lava Dart", from_zone="stack", reason="resolve"),
        _ev("zone_move", phase="main1", card="Lava Dart", from_zone=None, to_zone="stack", controller=0),
        _ev("zone_move", phase="main1", card="Lava Dart", from_zone="stack", to_zone="exile", reason="flashback"),
    ]
    steps = GameReducer(events).run()
    p0 = steps[-1]["players"][0]
    assert steps[-1]["stack"] == []
    assert p0["exile"] == ["Lava Dart"]
    assert p0["graveyard"] == []


def test_list_games_labels_from_matchup_meta():
    doc = {"meta": {"matchup": ["red_aggro", "blue_control"]},
           "games": [{"game_index": 0, "events": [_ev("turn_start", phase=None)]}]}
    result = list_games(doc)
    assert "red_aggro vs blue_control" in result["games"][0]["label"]
    assert result["games"][0]["num_events"] == 1


def test_reduce_game_rejects_non_event_stream_log():
    doc = {"meta": {}, "games": [{"game_index": 0, "events": [{"kind": "not_a_real_kind"}]}]}
    try:
        reduce_game(doc, 0)
        assert False, "expected a ValueError for a non-event-stream log"
    except ValueError:
        pass
