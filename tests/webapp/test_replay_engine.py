"""Self-check for the replay viewer's event-stream reducer (webapp/replay_engine.py).

Fabricated event logs (mirrors real GameState.log_event field shapes) --
exercises the pieces most likely to silently misrender: mulligan netting,
cast-then-resolve, mana, combat flag lifecycle, a countered spell, a
state-based death, and the same-phase-recast identity fix for the Lava Dart
double-copy bug.
"""
from webapp.replay_engine import GameReducer, list_games, reduce_game


def _ev(kind, active_idx=0, turn_player_idx=0, turn=1, phase="main1", **fields):
    return {"kind": kind, "turn": turn, "phase": phase, "active_idx": active_idx,
            "turn_player_idx": turn_player_idx, **fields}


def test_mulligan_rounds_are_individually_visible():
    """Owner directive (2026-08-02): no opening-hand netting -- every
    mulligan-round draw, reject, and bottom-card pick is its own step, real
    event shapes (mulligan_take rejects the whole hand at once, cards=[...];
    mulligan_bottom is logged once per bottomed card, card=<single name>,
    per game/resolution/handlers_mulligan.py)."""
    events = [
        _ev("zone_move", phase=None, cards=["A", "B", "C", "D", "E", "F", "G"],
            from_zone="library", to_zone="hand", reason="draw"),
        _ev("zone_move", phase=None, cards=["A", "B", "C", "D", "E", "F", "G"],
            from_zone="hand", to_zone="library", reason="mulligan_take"),
        _ev("zone_move", phase=None, cards=["H", "I", "J", "K", "L", "M", "N"],
            from_zone="library", to_zone="hand", reason="draw"),
        _ev("zone_move", phase=None, card="N", from_zone="hand", to_zone="library_bottom", reason="mulligan_bottom"),
        _ev("turn_start", phase=None),
    ]
    steps = GameReducer(events).run()
    kinds = [s["kind"] for s in steps]
    assert kinds == ["zone_move", "zone_move", "zone_move", "zone_move", "turn_start"]
    # after the re-draw, before bottoming N:
    assert sorted(steps[2]["players"][0]["hand"]) == ["H", "I", "J", "K", "L", "M", "N"]
    # after bottoming N, at turn_start -- the actual kept hand:
    assert sorted(steps[-1]["players"][0]["hand"]) == ["H", "I", "J", "K", "L", "M"]
    assert steps[-1]["players"][0]["library_remaining"] == 60 - 6


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
         "base_power": 2, "base_toughness": 2,
         "is_token": False, "card_type": "CREATURE", "attacking": False, "blocking": None,
         "enchanting": None}
    ]


def test_stats_changed_updates_live_pt_but_not_base():
    events = [
        _ev("turn_start", phase=None),
        _ev("zone_move", phase="main1", permanent=["Grizzly Bears", 0], from_zone="hand", to_zone="battlefield",
            tapped=False, card_type="CREATURE", power=2, toughness=2),
        _ev("stats_changed", phase="main1", permanent=["Grizzly Bears", 0], power=3, toughness=3),
    ]
    steps = GameReducer(events).run()
    bear = steps[-1]["players"][0]["battlefield"][0]
    assert bear["power"] == 3 and bear["toughness"] == 3
    assert bear["base_power"] == 2 and bear["base_toughness"] == 2


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
    """A phase_change with nothing real in it collapses away (see
    test_empty_phases_collapse), so each phase here needs its own real event
    (life_change, arbitrary choice) to surface its own phase_change step."""
    events = [
        _ev("turn_start", phase=None),
        _ev("zone_move", phase="main1", permanent=["Grizzly Bears", 0], from_zone="hand", to_zone="battlefield",
            tapped=False, card_type="CREATURE", power=2, toughness=2),
        _ev("attack_declared", phase="declare_attackers", attacker=["Grizzly Bears", 0], tapped=True),
        _ev("phase_change", phase="declare_blockers"),
        _ev("life_change", phase="declare_blockers", player_idx=1, new_total=20, amount=0),
        _ev("phase_change", phase="combat_damage"),
        _ev("life_change", phase="combat_damage", player_idx=1, new_total=18, amount=-2),
    ]
    steps = GameReducer(events).run()
    kinds = [s["kind"] for s in steps]
    assert kinds == [
        "turn_start", "zone_move", "attack_declared",
        "phase_change", "life_change", "phase_change", "life_change",
    ]
    attacking_after_declare = steps[2]["players"][0]["battlefield"][0]["attacking"]
    attacking_at_blockers = steps[4]["players"][0]["battlefield"][0]["attacking"]
    attacking_after_damage = steps[6]["players"][0]["battlefield"][0]["attacking"]
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
    """Identity-tracking check for a real Lava Dart bug: a card recast in
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


def test_library_remaining_nets_correctly_across_a_mulligan():
    """DEFAULT_DECK_SIZE=60, net of every draw/mulligan_take/mulligan_bottom:
    draw 7 (+7), mulligan_take all 7 back (-7), draw 7 more (+7), bottom 1
    (-1) -> 6 net so far, +1 more in-game draw -> 7 -> 60 - 7 = 53.
    Owner-authorized simplification: mill/put-back (Brainstorm-style, not a
    mulligan) still don't move this counter."""
    events = [
        _ev("zone_move", phase=None, cards=["A", "B", "C", "D", "E", "F", "G"],
            from_zone="library", to_zone="hand", reason="draw"),
        _ev("zone_move", phase=None, cards=["A", "B", "C", "D", "E", "F", "G"],
            from_zone="hand", to_zone="library", reason="mulligan_take"),
        _ev("zone_move", phase=None, cards=["H", "I", "J", "K", "L", "M", "N"],
            from_zone="library", to_zone="hand", reason="draw"),
        _ev("zone_move", phase=None, card="N", from_zone="hand", to_zone="library_bottom", reason="mulligan_bottom"),
        _ev("turn_start", phase=None),
        _ev("zone_move", phase="draw", card="O", from_zone="library", to_zone="hand", reason="draw"),
        _ev("mill", phase="main1", player_idx=0, count=2, cards=["P", "Q"]),
        _ev("put_on_top", phase="main1", cards=["O"]),
    ]
    steps = GameReducer(events).run()
    assert steps[-1]["players"][0]["library_remaining"] == 53


def test_aura_attached_resolves_cross_player_target():
    """A Pacifism-style aura (controlled by P1) enchanting P0's creature must
    resolve "enchanting" to the TARGET's controller, not the aura's -- the
    viewer nests the aura under P0's creature even though P1 controls it."""
    events = [
        _ev("turn_start", phase=None),
        _ev("zone_move", active_idx=0, phase="main1", permanent=["Grizzly Bears", 0], from_zone="hand",
            to_zone="battlefield", tapped=False, card_type="CREATURE", power=2, toughness=2),
        _ev("zone_move", active_idx=1, phase="main1", permanent=["Pacifism", 0], from_zone="hand",
            to_zone="battlefield", tapped=False, card_type="ENCHANTMENT"),
        _ev("aura_attached", active_idx=1, phase="main1", aura=["Pacifism", 0], target=["Grizzly Bears", 0]),
    ]
    steps = GameReducer(events).run()
    pacifism = steps[-1]["players"][1]["battlefield"][0]
    assert pacifism["enchanting"] == {"controller_idx": 0, "name": "Grizzly Bears", "slot": 0}


def test_decision_weights_step_formats_candidates_and_flushes_pending_phase():
    """rl/agent.py and rl/mulligan.py log structured facts (fixed_label a
    plain string, pointer_identity a {name, slot, controller} fact, never a
    baked string) -- this handler does the label formatting, and (like any
    other real event) flushes a still-buffered empty phase_change."""
    events = [
        _ev("turn_start", phase=None),
        _ev("phase_change", phase="main1"),  # buffered until a real event flushes it
        _ev("decision_weights", phase="main1", network="main", chosen_index=1, value_estimate=0.42,
            pointer_kind="declare_attackers",
            candidates=[
                {"index": 0, "probability": 0.1, "fixed_label": "Pass", "pointer_identity": None},
                {"index": 1, "probability": 0.7, "fixed_label": None,
                 "pointer_identity": {"name": "Grizzly Bears", "slot": 0, "controller": 0}},
                {"index": 2, "probability": 0.2, "fixed_label": None,
                 "pointer_identity": {"name": "Lightning Bolt", "slot": None, "controller": 1}},
            ]),
    ]
    steps = GameReducer(events).run()
    kinds = [s["kind"] for s in steps]
    assert kinds == ["turn_start", "phase_change", "decision_weights"]  # the buffered phase_change got flushed
    dw = steps[-1]
    assert dw["network"] == "main" and dw["chosen_index"] == 1 and dw["value_estimate"] == 0.42
    assert dw["pointer_kind"] == "declare_attackers"
    labels = [c["label"] for c in dw["candidates"]]
    assert labels == ["Pass", "Grizzly Bears (slot 0) (P0)", "Lightning Bolt (P1)"]
    assert dw["players"] == steps[0]["players"], "decision_weights is non-board-mutating"


def test_pass_produces_no_step_regardless_of_stack():
    """Priority passes are pure engine machinery -- whatever they lead to
    (a stack item resolving, a phase advancing) already gets its own step,
    so "pass" itself must never appear in the scrubber timeline."""
    events = [
        _ev("turn_start", phase=None),
        _ev("pass", phase="main1"),  # stack empty here -- leads to a phase change
        _ev("phase_change", phase="combat_start"),
        _ev("zone_move", phase="main1", card="Lightning Bolt", from_zone="hand", to_zone="stack", controller=0),
        _ev("pass", phase="main1"),
        _ev("pass", phase="main1", active_idx=1),  # stack non-empty -- leads to a resolution
        _ev("zone_move", phase="main1", card="Lightning Bolt", from_zone="stack", reason="resolve"),
    ]
    steps = GameReducer(events).run()
    kinds = [s["kind"] for s in steps]
    assert "pass" not in kinds
    assert kinds == ["turn_start", "phase_change", "zone_move", "zone_move"]


def test_empty_phases_collapse():
    """A phase with nothing but priority passes in it (upkeep, end_combat)
    is skipped entirely -- straight through to whichever phase actually has
    something happen in it -- while a phase with a real event still shows
    its own phase_change step."""
    events = [
        _ev("turn_start", phase=None),
        _ev("phase_change", phase="upkeep"),          # empty -- collapses away
        _ev("phase_change", phase="draw"),
        _ev("zone_move", phase="draw", card="Island", from_zone="library", to_zone="hand", reason="draw"),
        _ev("phase_change", phase="main1"),            # empty -- collapses away
        _ev("phase_change", phase="combat_start"),      # empty -- collapses away
        _ev("phase_change", phase="declare_attackers"),  # empty (log ends before anything happens in it)
    ]
    steps = GameReducer(events).run()
    kinds_and_phases = [(s["kind"], s.get("phase")) for s in steps]
    assert kinds_and_phases == [
        ("turn_start", None),
        ("phase_change", "draw"),
        ("zone_move", "draw"),
    ]


def test_list_games_labels_from_matchup_meta():
    doc = {"meta": {"matchup": ["red_aggro", "blue_control"]},
           "games": [{"game_index": 0, "events": [_ev("turn_start", phase=None)]}]}
    result = list_games(doc)
    assert "red_aggro vs blue_control" in result["games"][0]["label"]
    assert result["games"][0]["num_events"] == 1


def test_list_games_labels_round_robin_games_by_their_own_pairing():
    # A round-robin --eval log holds many different pairings in one file --
    # meta has no single matchup to fall back to, so each game must be
    # labeled from its OWN deck_a/deck_b (run_league.py's _write_event_log).
    # Two games of the SAME pairing (a double round-robin) must disambiguate
    # rather than producing two identical, unindexable labels.
    doc = {"meta": {"mode": "eval", "matchup": None, "decks": ["dmir_terror", "elves"]},
           "games": [
               {"game_index": 0, "deck_a": "dmir_terror", "deck_b": "dmir_terror",
                "events": [_ev("turn_start", phase=None)]},
               {"game_index": 1, "deck_a": "dmir_terror", "deck_b": "dmir_terror",
                "events": [_ev("turn_start", phase=None)]},
               {"game_index": 2, "deck_a": "dmir_terror", "deck_b": "elves",
                "events": [_ev("turn_start", phase=None)]},
           ]}
    labels = [g["label"] for g in list_games(doc)["games"]]
    assert "dmir_terror vs dmir_terror" in labels[0] and "game 1" in labels[0]
    assert "dmir_terror vs dmir_terror" in labels[1] and "game 2" in labels[1]
    assert labels[0] != labels[1]
    assert "dmir_terror vs elves" in labels[2] and "game 1" in labels[2]


def test_reduce_game_rejects_non_event_stream_log():
    doc = {"meta": {}, "games": [{"game_index": 0, "events": [{"kind": "not_a_real_kind"}]}]}
    try:
        reduce_game(doc, 0)
        assert False, "expected a ValueError for a non-event-stream log"
    except ValueError:
        pass
