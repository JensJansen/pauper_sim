"""Self-check for _mulligan_common.audit_land_counts's per-deck attribution
extension -- the shared land-count/keep-rate reconstruction
validation.mulligan_audit depends on. Backward compatibility (no
deck_by_game_seat) is pinned too, since train_mulligan.py still relies on
that flat shape unchanged.
"""
from pathlib import Path

import pytest

import game as game_module
from analysis.mulligan_retrain._mulligan_common import (
    _binary_entropy_bits, audit_land_counts, build_probe_hands_sampled, probe_land_count_stats,
)
from rl.model.arch import SetTransformer
from rl.model.features import CardVocab
from rl.model.mulligan import MulliganNet

LAND = "Forest"
SPELL = "Llanowar Elves"

_SRC_DIR = Path(__file__).resolve().parent.parent / "src"


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
    assert result == {2: {"kept": 1, "mulliganed": 0, "keep_probs": [0.8],
                          "entropy_bits": [pytest.approx(0.7219280948873623)], "wins": 1, "losses": 0}}


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
    assert result == {1: {"kept": 1, "mulliganed": 0, "keep_probs": [0.5],
                          "entropy_bits": [pytest.approx(1.0)], "wins": 1, "losses": 0}}


def test_binary_entropy_bits_extremes_and_midpoint():
    assert _binary_entropy_bits(0.5) == pytest.approx(1.0)
    assert _binary_entropy_bits(0.0) == pytest.approx(0.0)
    assert _binary_entropy_bits(1.0) == pytest.approx(0.0)


@pytest.mark.slow
def test_build_probe_hands_sampled_is_deterministic_and_respects_land_count(monkeypatch):
    monkeypatch.chdir(_SRC_DIR)  # parse_decklist_file's "../data/..." is relative to src/, same as rl.roster's own convention
    decklist = game_module.parse_decklist_file("../data/mono_blue_terror.txt")
    vocab = CardVocab([decklist])
    land_indices = {vocab.index(n) for n, *_ in decklist if game_module.CARD_DEFS[n].card_type.name == "LAND"}

    probes_a = build_probe_hands_sampled(decklist, vocab, land_counts=[0, 3, 7], n_variants=3, seed=0)
    probes_b = build_probe_hands_sampled(decklist, vocab, land_counts=[0, 3, 7], n_variants=3, seed=0)

    assert set(probes_a) == {0, 3, 7}
    for lc, hands in probes_a.items():
        assert len(hands) == 3
        for tokens in hands:
            n_lands = sum(1 for row in tokens if row[0] in land_indices)
            assert n_lands == lc
    # seeded -> identical hands every call, same as build_probe_hands' fixed-target rationale
    assert [[t[0] for t in hand] for hands in probes_a.values() for hand in hands] == \
           [[t[0] for t in hand] for hands in probes_b.values() for hand in hands]


@pytest.mark.slow
def test_build_probe_hands_sampled_skips_a_land_count_the_deck_cant_draw():
    """A real deck this shape exists in the league roster: spy_combo runs
    only 4 Forest/Swamp (the rest of its "mana" is creatures, not Lands),
    which used to crash build_probe_hands_sampled at land_count=5 --
    rng.sample(lands, 5) from a population of 4 raises ValueError. A deck
    that can never physically be dealt a 5+-land hand shouldn't be asked
    to produce one; the bucket is skipped, not forced or faked."""
    decklist = [(LAND, 4), (SPELL, 56)]  # mirrors spy_combo's real 4-land/56-nonland shape
    vocab = CardVocab([decklist])

    probes = build_probe_hands_sampled(decklist, vocab, land_counts=range(8), n_variants=2, seed=0)

    assert set(probes) == {0, 1, 2, 3, 4}  # 5, 6, 7 are impossible with only 4 lands -- skipped, not crashed
    for lc, hands in probes.items():
        assert len(hands) == 2


@pytest.mark.slow
def test_probe_land_count_stats_shape(monkeypatch):
    torch = pytest.importorskip("torch")
    monkeypatch.chdir(_SRC_DIR)
    decklist = game_module.parse_decklist_file("../data/mono_blue_terror.txt")
    vocab = CardVocab([decklist])
    shared = SetTransformer(vocab.size, d_model=16, n_heads=2, n_layers=1, dim_feedforward=32)
    net = MulliganNet(shared, hidden=16)

    probes = build_probe_hands_sampled(decklist, vocab, land_counts=[0, 7], n_variants=2, seed=0)
    stats = probe_land_count_stats(net, probes)

    assert set(stats) == {0, 7}
    for lc, s in stats.items():
        assert s["n"] == 2
        assert 0.0 <= s["p_mulligan_mean"] <= 1.0
        assert 0.0 <= s["entropy_bits_mean"] <= 1.0
        assert s["p_mulligan_spread"] >= 0.0
