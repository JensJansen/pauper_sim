"""Layer 5 -- sensitivity. No dead inputs: perturbing any one input (each of the TOKEN_FEATURE_DIM columns, or a token's
card index) must move that token's representation and the summary of its own side. This catches mis-sliced columns,
a feature accidentally dropped before the projection, and an embedding that is never looked up."""
import numpy as np
import pytest

from support import synth
from rl.model.features import STATIC_FEATURE_DIM, TOKEN_FEATURE_DIM

EPS = 1e-7


@pytest.fixture(scope="module")
def setup():
    enc = synth.make_encoder(seed=3)
    rng = np.random.default_rng(14)
    v, f = synth.make_tokens(rng, 4, 4)
    return enc, v, f, synth.run(enc, *synth.pad([(v, f)]))


# column TOKEN_FEATURE_DIM-1 is the side flag: it is ALSO what routes a token to the mine/theirs pool, so perturbing
# it by 0.7 would flip the token's side -- test it separately below.
@pytest.mark.parametrize("col", range(TOKEN_FEATURE_DIM - 1))
@pytest.mark.parametrize("side", ["mine", "theirs"])
def test_every_feature_column_moves_the_token_and_its_sides_summary(setup, col, side):
    enc, v, f, (bm, bt, br) = setup
    i = 1 if side == "mine" else 5
    f2 = f.copy()
    f2[i, col] += 0.5
    m, t, r = synth.run(enc, *synth.pad([(v, f2)]))
    assert (r[0, i] - br[0, i]).abs().max() > EPS, f"column {col} never reaches token {i}'s representation"
    own, base_own = (m, bm) if side == "mine" else (t, bt)
    assert (own - base_own).abs().max() > EPS, f"column {col} never reaches the {side} summary"


def test_the_side_flag_routes_a_token_between_the_two_summaries(setup):
    enc, v, f, (bm, bt, br) = setup
    f2 = f.copy()
    f2[1, -1] = 0.0  # a mine token becomes a theirs token
    m, t, r = synth.run(enc, *synth.pad([(v, f2)]))
    assert (m - bm).abs().max() > EPS and (t - bt).abs().max() > EPS


@pytest.mark.parametrize("side", ["mine", "theirs"])
def test_the_card_embedding_is_used(setup, side):
    enc, v, f, (bm, bt, br) = setup
    i = 1 if side == "mine" else 5
    v2 = v.copy()
    v2[i] = 1 + (v2[i] % (synth.VOCAB - 2))  # a different real card
    assert v2[i] != v[i]
    m, t, r = synth.run(enc, *synth.pad([(v2, f)]))
    assert (r[0, i] - br[0, i]).abs().max() > EPS
    assert ((m - bm) if side == "mine" else (t - bt)).abs().max() > EPS


def test_padding_index_zero_embeds_to_nothing():
    enc = synth.make_encoder(seed=4)
    assert enc.embedding.weight[0].abs().max() == 0, "index 0 is the padding/unknown sentinel and must embed to zero"
    assert STATIC_FEATURE_DIM < TOKEN_FEATURE_DIM
