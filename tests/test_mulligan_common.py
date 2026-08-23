"""Self-check for _mulligan_common.audit_land_counts's per-deck attribution
extension -- the shared land-count/keep-rate reconstruction
validation.mulligan_audit depends on. Backward compatibility (no
deck_by_game_seat) is pinned too, since train_mulligan.py still relies on
that flat shape unchanged.
"""
from analysis.mulligan_retrain._mulligan_common import audit_land_counts

LAND = "Forest"
SPELL = "Llanowar Elves"


def _game(seat, hand_lands, chosen, p_keep, winner=None):
    """One synthetic game: `seat` draws hand_lands lands plus one spell,
    makes a single mulligan_keep decision, then (optionally) game_over."""
    cards = [LAND] * hand_lands + [SPELL]
    events = [
        {"kind": "zone_move", "active_idx": seat, "reason": "draw", "to_zone": "hand", "cards": cards},
        {"kind": "decision_weights", "active_idx": seat, "network": "mulligan_keep", "chosen_index": chosen,
         "candidates": [{"fixed_label": "Keep", "probability": p_keep}]},
    ]
    if winner is not None:
        events.append({"kind": "game_over", "winner": winner})
    return events


def test_unbucketed_matches_the_pre_extension_flat_shape():
    games = [_game(0, 2, chosen=0, p_keep=0.8, winner=0)]  # kept, 2 lands, seat 0 wins
    result = audit_land_counts(games)
    assert result == {2: {"kept": 1, "mulliganed": 0, "keep_probs": [0.8], "wins": 1, "losses": 0}}


def test_mulliganed_hand_is_never_scored_for_win_loss():
    games = [_game(0, 0, chosen=1, p_keep=0.1, winner=1)]  # mulliganed, 0 lands
    result = audit_land_counts(games)
    assert result[0]["kept"] == 0
    assert result[0]["mulliganed"] == 1
    assert result[0]["wins"] == 0 and result[0]["losses"] == 0


def test_bucketed_attributes_each_seat_to_its_own_deck():
    games = [_game(0, 3, chosen=0, p_keep=0.7, winner=0)]
    result = audit_land_counts(games, deck_by_game_seat=[{0: "elves"}])
    assert set(result) == {"elves"}
    assert result["elves"][3]["kept"] == 1
    assert result["elves"][3]["wins"] == 1


def test_bucketed_excludes_a_seat_absent_from_the_map():
    """The caller's way to filter out an opponent-league seat in a
    cross-league game: only seat 0 is present in the map here, so seat 1's
    decision must not appear anywhere in the result -- not under any deck,
    not even unbucketed."""
    seat0 = {"kind": "zone_move", "active_idx": 0, "reason": "draw", "to_zone": "hand", "cards": [LAND, SPELL]}
    seat0_decision = {"kind": "decision_weights", "active_idx": 0, "network": "mulligan_keep", "chosen_index": 0,
                      "candidates": [{"fixed_label": "Keep", "probability": 0.6}]}
    seat1 = {"kind": "zone_move", "active_idx": 1, "reason": "draw", "to_zone": "hand",
             "cards": [LAND, LAND, SPELL]}
    seat1_decision = {"kind": "decision_weights", "active_idx": 1, "network": "mulligan_keep", "chosen_index": 0,
                      "candidates": [{"fixed_label": "Keep", "probability": 0.9}]}
    events = [seat0, seat0_decision, seat1, seat1_decision, {"kind": "game_over", "winner": 0}]

    result = audit_land_counts([events], deck_by_game_seat=[{0: "elves"}])

    assert set(result) == {"elves"}
    assert result["elves"][1]["kept"] == 1  # seat 0's 1-land hand
    assert 2 not in result["elves"]  # seat 1's 2-land decision excluded entirely


def test_mulligan_bottom_removes_a_card_from_the_reconstructed_hand():
    events = [
        {"kind": "zone_move", "active_idx": 0, "reason": "draw", "to_zone": "hand", "cards": [LAND, LAND, SPELL]},
        {"kind": "zone_move", "active_idx": 0, "reason": "mulligan_bottom", "card": LAND},
        {"kind": "decision_weights", "active_idx": 0, "network": "mulligan_keep", "chosen_index": 0,
         "candidates": [{"fixed_label": "Keep", "probability": 0.5}]},
        {"kind": "game_over", "winner": 0},
    ]
    result = audit_land_counts([events])
    assert result == {1: {"kept": 1, "mulliganed": 0, "keep_probs": [0.5], "wins": 1, "losses": 0}}
