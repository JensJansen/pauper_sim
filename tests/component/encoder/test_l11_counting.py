"""Layer 11 -- counting. n identical cards must be distinguishable from m identical cards, at every level the policy
can read: the per-token copy-count columns, the encoder's summaries (attention pool AND sum pool), and the exact-count
scalars. (Without this, 6 Mountains looked like 1: attention is a weighted mean and identical tokens add nothing.)"""
import itertools
import math

import numpy as np
import pytest

import game
from support import scenario as sc
from rl.decision.agent import _scalar_features
from rl.model import features as F
from rl.model.deck import COUNT_CAP, COUNT_SCALARS_DIM, COUNT_SCALARS_PER_SIDE, SCALAR_FEATURE_DIM

COPIES = {
    "lands I control": lambda n: {"s0": sc.seat(bf=["Mountain"] * n), "s1": sc.seat()},
    "copies of a spell in my hand": lambda n: {"s0": sc.seat(hand=["Lightning Bolt"] * n), "s1": sc.seat()},
    "cards in my graveyard": lambda n: {"s0": sc.seat(gy=["Counterspell"] * n), "s1": sc.seat()},
    "cards in their graveyard": lambda n: {"s0": sc.seat(), "s1": sc.seat(gy=["Counterspell"] * n)},
    "their lands": lambda n: {"s0": sc.seat(), "s1": sc.seat(bf=["Island"] * n)},
    "cards in their exile": lambda n: {"s0": sc.seat(), "s1": sc.seat(ex=["Lightning Bolt"] * n)},
    "creatures I control": lambda n: {"s0": sc.seat(bf=["Guttersnipe"] * n), "s1": sc.seat()},
}


@pytest.fixture(scope="module")
def world():
    vocab = sc.make_vocab()
    return vocab, sc.make_encoder(vocab)


def _log(n):
    return math.log(n) / math.log(F.COPIES_CAP)


# ---- the token columns ---------------------------------------------------------------------------------------------------

def test_copy_counts_are_exact_and_tokens_stay_separate():
    vocab = sc.make_vocab()
    state = sc.build(s0=sc.seat(bf=["Mountain", ("Mountain", {"tapped": True}), ("Mountain", {"tapped": True}),
                                   ("Mountain", {"tapped": True}), "Mountain", "Guttersnipe"]), s1=sc.seat(bf=["Mountain"]))
    tokens = F.build_token_set(state, 0, vocab)
    assert len(tokens) == 7, "identical tokens must NOT be merged: each stays pointer-addressable"
    assert len({id(ident) for _i, _r, ident in tokens}) == 7
    c_card, c_state = F.MULTIPLICITY_START, F.MULTIPLICITY_START + 1
    mine_mountains = [r for i, r, _x in tokens if i == vocab.index("Mountain") and r[-1] == 1.0]
    theirs_mountain = next(r for i, r, _x in tokens if i == vocab.index("Mountain") and r[-1] == 0.0)
    assert len(mine_mountains) == 5
    assert all(abs(r[c_card] - _log(5)) < 1e-6 for r in mine_mountains), "5 Mountains: every one reads 'five copies'"
    tapped_col = F.STATIC_FEATURE_DIM + F.HAND_FEATURE_DIM + F.MULTIPLICITY_FEATURE_DIM + F.LINK_FEATURE_DIM + 1
    tapped = [r for r in mine_mountains if r[tapped_col] == 1.0]
    untapped = [r for r in mine_mountains if r[tapped_col] == 0.0]
    assert len(tapped) == 3 and all(abs(r[c_state] - _log(3)) < 1e-6 for r in tapped), "3 tapped read 'three alike'"
    assert len(untapped) == 2 and all(abs(r[c_state] - _log(2)) < 1e-6 for r in untapped)
    assert theirs_mountain[c_card] == 0.0 and theirs_mountain[c_state] == 0.0, "my Mountains and theirs are counted apart"
    lone = next(r for i, r, _x in tokens if i == vocab.index("Guttersnipe"))
    assert lone[c_card] == 0.0 and lone[c_state] == 0.0


def test_copy_count_saturates_at_the_cap_and_is_monotone():
    vocab = sc.make_vocab()
    vals = []
    for n in (1, 2, 3, 5, 8, 13, 20, 30):
        tokens = F.build_token_set(sc.build(s0=sc.seat(hand=["Lightning Bolt"] * n), s1=sc.seat()), 0, vocab)
        vals.append(tokens[0][1][F.MULTIPLICITY_START])
    assert vals == sorted(vals) and vals[0] == 0.0 and vals[-1] == 1.0 and len(set(vals)) == len(vals) - 1  # 20 and 30 tie


# ---- the encoder ----------------------------------------------------------------------------------------------------------

@pytest.mark.parametrize("what", sorted(COPIES))
def test_every_copy_count_from_one_to_eight_encodes_differently(world, what):
    vocab, enc = world
    enc_of = {n: sc.encode(sc.build(**COPIES[what](n)), 0, enc, vocab) for n in range(1, 9)}
    for a, b in itertools.combinations(enc_of, 2):
        assert sc.differs(enc_of[a], enc_of[b], tol=1e-5), f"{a} vs {b} {what} encode alike"


def test_the_split_between_tapped_and_untapped_copies_is_visible(world):
    vocab, enc = world

    def split(tapped, untapped):
        bf = [("Mountain", {"tapped": True})] * tapped + ["Mountain"] * untapped
        return sc.encode(sc.build(s0=sc.seat(bf=bf), s1=sc.seat()), 0, enc, vocab)

    encs = {t: split(t, 5 - t) for t in range(6)}
    for a, b in itertools.combinations(encs, 2):
        assert sc.differs(encs[a], encs[b], tol=1e-5), f"{a} vs {b} tapped of 5 encode alike"


def test_six_of_one_card_differs_from_six_of_another(world):
    vocab, enc = world
    a = sc.encode(sc.build(s0=sc.seat(bf=["Mountain"] * 6), s1=sc.seat()), 0, enc, vocab)
    b = sc.encode(sc.build(s0=sc.seat(bf=["Island"] * 6), s1=sc.seat()), 0, enc, vocab)
    assert sc.differs(a, b)


def test_the_sum_pooled_half_grows_with_the_count(world):
    """The second half of a summary is a sum, so it must carry cardinality by itself."""
    vocab, enc = world
    norms = [float(sc.encode(sc.build(**COPIES["lands I control"](n)), 0, enc, vocab)[0][enc.d_model:].norm()) for n in range(1, 9)]
    assert all(b > a for a, b in zip(norms, norms[1:])), norms


# ---- the exact-count scalars -----------------------------------------------------------------------------------------------

def _cnt(n):
    return math.log1p(n) / math.log1p(COUNT_CAP)


def _real_state():
    import random
    dl = game.parse_decklist_file("../data/mono_red_madness.txt")
    return game.new_multiplayer_game_state([dl, dl], 0, random.Random(0))


def test_scalar_counts_track_the_board_exactly():
    state = _real_state()
    types = list(game.CardType)
    base = SCALAR_FEATURE_DIM - COUNT_SCALARS_DIM
    me, opp = state.players[0], state.players[1]
    me.battlefield.clear(), opp.battlefield.clear(), me.graveyard.clear(), opp.graveyard.clear()
    for n in range(9):
        me.battlefield[:] = [game.state.Permanent(game.CARD_DEFS["Mountain"]) for _ in range(n)]
        vec = _scalar_features(state, 0, 40)
        assert vec[base + types.index(game.CardType.LAND)] == pytest.approx(_cnt(n)), f"{n} lands"
        assert vec[base + len(types) + types.index(game.CardType.LAND)] == pytest.approx(_cnt(n)), "all untapped"
        assert vec[base + COUNT_SCALARS_PER_SIDE + types.index(game.CardType.LAND)] == pytest.approx(0.0), "their lands: none"
    for p in me.battlefield[:3]:
        p.tapped = True
    vec = _scalar_features(state, 0, 40)
    assert vec[base + len(types) + types.index(game.CardType.LAND)] == pytest.approx(_cnt(5)), "8 lands, 3 tapped"
    assert vec[base + types.index(game.CardType.LAND)] == pytest.approx(_cnt(8))


def test_scalar_counts_cover_graveyard_exile_stack_and_my_hand():
    state = _real_state()
    types = list(game.CardType)
    base = SCALAR_FEATURE_DIM - COUNT_SCALARS_DIM
    gy, ex, st = base + 2 * len(types), base + 2 * len(types) + 1, base + 2 * len(types) + 2
    me, opp = state.players[0], state.players[1]
    me.graveyard[:] = [game.state.CardInstance(game.CARD_DEFS["Lightning Bolt"]) for _ in range(4)]
    opp.exile[:] = [(game.CARD_DEFS["Lightning Bolt"], None)] * 2
    state.stack[:] = [{"card_def": game.CARD_DEFS["Counterspell"], "resolve": None, "controller": 1, "targets": ()}] * 3
    me.hand[:] = [game.CARD_DEFS["Mountain"]] * 6
    v0, v1 = _scalar_features(state, 0, 40), _scalar_features(state, 1, 40)
    assert v0[gy] == pytest.approx(_cnt(4)) and v0[gy + COUNT_SCALARS_PER_SIDE] == pytest.approx(0.0)
    assert v0[ex] == pytest.approx(0.0) and v0[ex + COUNT_SCALARS_PER_SIDE] == pytest.approx(_cnt(2))
    assert v0[st] == pytest.approx(0.0) and v0[st + COUNT_SCALARS_PER_SIDE] == pytest.approx(_cnt(3))
    assert v0[-1] == pytest.approx(_cnt(6)), "my own hand size is the last count scalar"
    # the other seat sees the same position mirrored
    assert v1[gy + COUNT_SCALARS_PER_SIDE] == pytest.approx(_cnt(4)) and v1[st] == pytest.approx(_cnt(3))


def test_scalar_counts_are_in_range_and_seen_from_each_seat():
    state = _real_state()
    for seat in (0, 1):
        vec = np.asarray(_scalar_features(state, seat, 40))
        assert len(vec) == SCALAR_FEATURE_DIM
        counts = vec[SCALAR_FEATURE_DIM - COUNT_SCALARS_DIM:]
        assert counts.min() >= 0.0 and counts.max() <= 1.0
