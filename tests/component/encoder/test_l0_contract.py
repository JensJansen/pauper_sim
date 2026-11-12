"""Layer 0 -- contract. Cheap invariants that every real token row must satisfy, whatever the game state."""
import numpy as np
import pytest

from support import corpus as ec
from rl.model import features as F



def test_row_width_and_dtype(frames):
    for f in frames:
        assert f.rows.shape == (len(f.vocab_idx), F.TOKEN_FEATURE_DIM) and f.rows.dtype == np.float32


def test_every_value_is_finite_and_within_zero_one(frames):
    """Every token feature and every scalar lies in [0, 1], with ONE deliberate exception: eff_power is signed, in
    [-1, 1] (a creature can have negative power and a later pump acts on the true value). Life is capped at 200 and
    cost reductions at the largest printed cost; both used to leak above 1."""
    for k, f in enumerate(frames):
        assert np.isfinite(f.rows).all() and np.isfinite(f.scalar).all(), f"frame {k}"
        unsigned = np.delete(f.rows, ec.COL["eff_power"], axis=1)
        assert unsigned.min() >= 0.0 and unsigned.max() <= 1.0, f"frame {k}: a token feature outside [0,1]"
        assert f.rows[:, ec.COL["eff_power"]].min() >= -1.0 and f.rows[:, ec.COL["eff_power"]].max() <= 1.0, f"frame {k}: eff_power outside [-1,1]"
        assert f.scalar.min() >= 0.0 and f.scalar.max() <= 1.0, f"frame {k}: a scalar outside [0,1]"


def test_scalar_length_matches_the_declared_dim(frames):
    from rl.model.deck import SCALAR_FEATURE_DIM
    assert all(len(f.scalar) == SCALAR_FEATURE_DIM for f in frames)


def test_exactly_one_zone_and_a_binary_side_flag_per_token(frames):
    zone_cols = [ec.COL[f"zone[{z}]"] for z in ec.ZONE_ORDER]
    for k, f in enumerate(frames):
        if not len(f.rows):
            continue
        assert (f.rows[:, zone_cols].sum(axis=1) == 1.0).all(), f"frame {k}: a token is in 0 or 2 zones"
        assert set(np.unique(f.rows[:, ec.COL["side_mine"]])) <= {0.0, 1.0}


def test_binary_columns_are_binary(frames):
    binary = ["untapped", "tapped", "blocked_attacker", "committed_blocker", "attacking", "summoning_sick",
              "is_attached", "targeted_by_mine", "targeted_by_theirs", "revealed_focus", "revealed_kept",
              "revealed_disposed", "counter[other]"] + [f"zone[{z}]" for z in ec.ZONE_ORDER]
    cols = [ec.COL[c] for c in binary]
    for k, f in enumerate(frames):
        assert np.isin(f.rows[:, cols], (0.0, 1.0)).all(), f"frame {k}"


def test_battlefield_state_is_only_set_on_battlefield_tokens(frames):
    """untapped/tapped/power/toughness/combat flags/counters belong to permanents; every other zone reads zero."""
    names = (["untapped", "tapped", "eff_power", "eff_toughness", "blocked_attacker", "committed_blocker", "attacking",
              "summoning_sick", "auras_attached", "is_attached"] + [f"counter[{k}]" for k in ec.COUNTER_KINDS]
             + ["counter[other]"])
    perm_cols = [ec.COL[c] for c in names]
    bf = ec.COL["zone[battlefield]"]
    for k, f in enumerate(frames):
        off = f.rows[:, bf] == 0.0
        assert (f.rows[off][:, perm_cols] == 0.0).all(), f"frame {k}: non-battlefield token carries permanent state"
        on = f.rows[:, bf] == 1.0
        assert ((f.rows[on][:, ec.COL["untapped"]] + f.rows[on][:, ec.COL["tapped"]]) == 1.0).all()


def test_cost_reduction_only_on_own_hand_tokens(frames):
    cr, hand, side = ec.COL["cost_reduction"], ec.COL["zone[hand]"], ec.COL["side_mine"]
    for k, f in enumerate(frames):
        stray = (f.rows[:, cr] != 0.0) & ~((f.rows[:, hand] == 1.0) & (f.rows[:, side] == 1.0))
        assert not stray.any(), f"frame {k}: cost_reduction set outside the viewer's own hand"


def test_revealed_tokens_are_the_deciders_own_and_only_during_their_pending_kinds(frames):
    rev, side = ec.COL["zone[revealed]"], ec.COL["side_mine"]
    allowed = {"scry", "surveil", "ponder", "tuck_position"}
    for k, f in enumerate(frames):
        is_rev = f.rows[:, rev] == 1.0
        if is_rev.any():
            assert f.pending_kind in allowed, f"frame {k}: revealed tokens during {f.pending_kind}"
            assert (f.rows[is_rev][:, side] == 1.0).all()


def test_opponent_hand_is_never_tokenized_except_a_reveal(frames):
    hand, side = ec.COL["zone[hand]"], ec.COL["side_mine"]
    for k, f in enumerate(frames):
        opp_hand = (f.rows[:, hand] == 1.0) & (f.rows[:, side] == 0.0)
        if opp_hand.any():
            assert f.pending_kind == "choose_graveyard_card", f"frame {k}: opponent hand visible during {f.pending_kind}"


def test_copy_counts_are_consistent(frames):
    card, state = ec.COL["copies_same_card"], ec.COL["copies_same_state"]
    for k, f in enumerate(frames):
        assert (f.rows[:, state] <= f.rows[:, card] + 1e-6).all(), f"frame {k}: more identical-state copies than copies of the card"
        assert ((f.rows[:, card] == 0) <= (f.rows[:, state] == 0)).all(), f"frame {k}: state copies without card copies"


def test_mask_width_matches_tokens(frames):
    for k, f in enumerate(frames):
        assert len(f.full_mask) == f.n_fixed + len(f.rows), f"frame {k}"
        assert f.full_mask.any(), f"frame {k}: nothing legal"


def test_life_totals_are_exact_below_the_cap_and_saturate_at_one_above_it():
    """Real games almost never reach the cap, so the corpus cannot exercise it: set life directly."""
    import random

    import game
    from rl.decision.agent import LIFE_CAP, _scalar_features
    
    decklist = game.parse_decklist_file("../data/mono_red_madness.txt")
    state = game.new_multiplayer_game_state([decklist, decklist], 0, random.Random(0))
    base = 4 + 2 * len(game.POOL_COLORS) + len(game.turn.Phase)  # first life column (my life, then the opponent's)
    for life, want in ((-5, 0.0), (0, 0.0), (20, 20 / LIFE_CAP), (199, 199 / LIFE_CAP), (200, 1.0), (201, 1.0), (5000, 1.0)):
        state.players[0].life_total = state.players[1].life_total = life
        vec = _scalar_features(state, 0, 40)
        assert vec[base] == pytest.approx(want) and vec[base + 1] == pytest.approx(want), f"life {life}"
        assert min(vec) >= 0.0 and max(vec) <= 1.0, f"life {life}: a scalar left [0,1]"


def test_no_printed_cost_saturates_and_so_no_cost_reduction_can_exceed_one():
    """cost_reduction = (printed generic - effective generic) / MANA_PIP_CAP, at most printed generic / MANA_PIP_CAP. That
    is within [0, 1] only if the cap covers the largest cost any card has; the static cost columns use the same cap, so
    this also means no printed cost is clipped."""
    import game
    largest = max(n for cd in game.CARD_DEFS.values() for n in (cd.cast_cost or {}).values())
    assert largest <= F.MANA_PIP_CAP, f"MANA_PIP_CAP {F.MANA_PIP_CAP} < largest printed cost {largest}"
    for cd in game.CARD_DEFS.values():
        vec = F.static_card_features(cd)[:len(game.POOL_COLORS) + 1]
        want = [min((cd.cast_cost or {}).get(c, 0), F.MANA_PIP_CAP) / F.MANA_PIP_CAP for c in (*game.POOL_COLORS, "generic")]
        assert vec == pytest.approx(want) and max(want) <= 1.0, cd.name
