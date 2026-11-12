"""Layer 4 -- the maths of SetTransformer. Synthetic inputs give full control (empty sides, one token, all padding);
the second half repeats the key properties on real decisions from the corpus. The empty-side cases are the ones the
original permutation test never reached -- they hid a real bug (an empty side was pooled over whichever token came
first)."""
import numpy as np
import pytest
import torch

from support import synth
from rl.model.arch import SetTransformer

ATOL = 1e-5

# (n_mine, n_theirs): every side-occupancy regime, including empty sides and a single token
CONFIGS = [(1, 0), (0, 1), (3, 0), (0, 3), (1, 1), (2, 5), (7, 1), (12, 12)]


@pytest.fixture(scope="module")
def enc():
    return synth.make_encoder()


def _rng(seed=0):
    return np.random.default_rng(seed)


@pytest.mark.parametrize("cfg", CONFIGS)
def test_permuting_tokens_permutes_reps_and_keeps_summaries(enc, cfg):
    rng = _rng(1)
    v, f = synth.make_tokens(rng, *cfg)
    base = synth.run(enc, *synth.pad([(v, f)]))
    for _ in range(4):
        order = rng.permutation(len(v))
        got = synth.run(enc, *synth.pad([(v[order], f[order])]))
        assert torch.allclose(got[0], base[0], atol=ATOL) and torch.allclose(got[1], base[1], atol=ATOL)
        assert torch.allclose(got[2][0, :len(v)], base[2][0, order], atol=ATOL)


@pytest.mark.parametrize("cfg", CONFIGS)
def test_padding_is_invisible_even_when_it_holds_garbage(enc, cfg):
    rng = _rng(2)
    v, f = synth.make_tokens(rng, *cfg)
    n = len(v)
    base = synth.run(enc, *synth.pad([(v, f)]))
    for extra in (1, 5, 17):
        zeros = synth.run(enc, *synth.pad([(v, f)], width=n + extra))
        junk = synth.run(enc, *synth.pad([(v, f)], width=n + extra, garbage=_rng(3)))
        for got in (zeros, junk):
            assert torch.allclose(got[0], base[0], atol=ATOL) and torch.allclose(got[1], base[1], atol=ATOL)
            assert torch.allclose(got[2][0, :n], base[2][0, :n], atol=ATOL)


@pytest.mark.parametrize("mode", ["eval", "train"])
def test_batched_equals_one_at_a_time(mode):
    enc = synth.make_encoder()
    getattr(enc, mode)()
    rng = _rng(4)
    sets = [synth.make_tokens(rng, *c) for c in CONFIGS + [(5, 3), (0, 9), (9, 0)]]
    bm, bt, br = synth.run(enc, *synth.pad(sets, garbage=_rng(5)))
    for i, s in enumerate(sets):
        sm, st, sr = synth.run(enc, *synth.pad([s]))
        n = len(s[0])
        assert torch.allclose(bm[i], sm[0], atol=ATOL) and torch.allclose(bt[i], st[0], atol=ATOL), f"set {i}"
        assert torch.allclose(br[i, :n], sr[0, :n], atol=ATOL), f"set {i}"


@pytest.mark.parametrize("cfg,empty_mine,empty_theirs", [((0, 4), True, False), ((4, 0), False, True), ((2, 2), False, False)])
def test_a_side_with_no_tokens_summarises_to_exactly_zero(enc, cfg, empty_mine, empty_theirs):
    v, f = synth.make_tokens(_rng(6), *cfg)
    mine, theirs, _ = synth.run(enc, *synth.pad([(v, f)]))
    assert (mine.abs().max() == 0) == empty_mine and (theirs.abs().max() == 0) == empty_theirs


def test_an_all_padding_batch_row_is_finite_and_zero(enc):
    sets = [synth.make_tokens(_rng(7), 2, 2), (np.zeros(0, dtype=np.int64), np.zeros((0, synth.TOKEN_FEATURE_DIM), np.float32))]
    mine, theirs, reps = synth.run(enc, *synth.pad(sets))
    assert torch.isfinite(mine).all() and torch.isfinite(theirs).all() and torch.isfinite(reps).all()
    assert mine[1].abs().max() == 0 and theirs[1].abs().max() == 0


@pytest.mark.parametrize("cfg", [(3, 4), (1, 6), (5, 1)])
def test_pooling_attends_only_to_its_own_side_and_sums_to_one(enc, cfg):
    """Rebuild the pooling masks from the spec and read the attention weights directly."""
    v, f = synth.make_tokens(_rng(8), *cfg)
    vi, ft, mask = synth.pad([(v, f)])
    with torch.no_grad():
        x = enc.input_proj(torch.cat([enc.embedding(vi), ft], dim=-1))
        encoded = enc.encoder(x, src_key_padding_mask=mask)
        side = ft[:, :, -1]
        for pool, query, wanted in ((enc.mine_pool, enc.mine_query, side >= 0.5), (enc.theirs_pool, enc.theirs_query, side < 0.5)):
            pool_mask = ~wanted | mask
            _out, w = pool(query.expand(1, -1, -1), encoded, encoded, key_padding_mask=pool_mask, need_weights=True, average_attn_weights=True)
            w = w[0, 0]
            assert torch.allclose(w.sum(), torch.tensor(1.0), atol=1e-5)
            assert w[pool_mask[0]].abs().max() == 0, "attention leaked onto the other side or onto padding"


def test_cross_token_attention_is_live(enc):
    """Changing one token changes OTHER tokens' representations (they attend to each other), and a change to a
    'theirs' token reaches the 'mine' summary and vice versa (one joint attention stack)."""
    rng = _rng(9)
    v, f = synth.make_tokens(rng, 3, 3)
    base = synth.run(enc, *synth.pad([(v, f)]))
    f2 = f.copy()
    f2[4, 10] += 0.7  # a 'theirs' token
    got = synth.run(enc, *synth.pad([(v, f2)]))
    assert (got[2][0, 0] - base[2][0, 0]).abs().max() > 1e-6, "token 0 ignored a change to token 4"
    assert (got[0] - base[0]).abs().max() > 1e-6, "a theirs-token change never reached the mine summary"
    f3 = f.copy()
    f3[0, 10] += 0.7  # a 'mine' token
    got = synth.run(enc, *synth.pad([(v, f3)]))
    assert (got[1] - base[1]).abs().max() > 1e-6, "a mine-token change never reached the theirs summary"


def test_deterministic_and_dropout_free_in_eval(enc):
    v, f = synth.make_tokens(_rng(10), 4, 4)
    a = synth.run(enc, *synth.pad([(v, f)]))
    b = synth.run(enc, *synth.pad([(v, f)]))
    assert all(torch.equal(x, y) for x, y in zip(a, b))


def test_every_parameter_gets_a_finite_gradient_and_padding_gets_none():
    enc = synth.make_encoder()
    enc.train()
    rng = _rng(11)
    sets = [synth.make_tokens(rng, *c) for c in [(3, 2), (0, 4), (4, 0), (1, 1)]]
    vi, ft, mask = synth.pad(sets, garbage=_rng(12))
    ft = ft.clone().requires_grad_(True)
    mine, theirs, reps = enc(vi, ft, mask, ft[:, :, -1].detach())
    (mine.sum() + theirs.sum() + reps[~mask].sum()).backward()
    dead = [n for n, p in enc.named_parameters() if p.grad is None or not torch.isfinite(p.grad).all()]
    assert not dead, dead
    assert ft.grad[mask].abs().max() == 0, "gradient reached padded positions"
    assert ft.grad[~mask].abs().max() > 0


def test_encoder_config_is_read_back_from_its_own_state_dict():
    from rl.model.arch import encoder_kwargs_from_state_dict
    enc = synth.make_encoder(d_model=48, n_heads=3, n_layers=3, dim_ff=96)
    kw = encoder_kwargs_from_state_dict({f"encoder.{k}": v for k, v in enc.state_dict().items()})  # a DeckNetwork's keys
    assert kw == {"d_model": 48, "n_layers": 3, "n_heads": 3, "dim_feedforward": 96}
    twin = SetTransformer(synth.VOCAB, **kw)
    twin.load_state_dict(enc.state_dict())


# ---- the same properties on REAL decisions ------------------------------------------------------------------------------

@pytest.mark.parametrize("mode", ["eval", "train"])
def test_real_decisions_are_order_padding_and_batch_invariant(frames, vocab, mode):
    torch.manual_seed(0)
    enc = SetTransformer(vocab.size, d_model=32, n_heads=2, n_layers=2, dim_feedforward=64)
    getattr(enc, mode)()
    rng = _rng(13)
    sets = [(f.vocab_idx, f.rows) for f in frames if len(f.vocab_idx)][:: max(1, len(frames) // 400)]
    empty_theirs = sum(1 for _v, r in sets if (r[:, -1] == 1.0).all())
    assert len(sets) >= 300 and empty_theirs >= 5, f"corpus sample has {empty_theirs} empty-opponent states"
    chunks = [sets[i:i + 16] for i in range(0, len(sets), 16)]
    for chunk in chunks:
        bm, bt, br = synth.run(enc, *synth.pad(chunk, garbage=rng))
        for i, (v, r) in enumerate(chunk):
            n = len(v)
            order = rng.permutation(n)
            sm, st, sr = synth.run(enc, *synth.pad([(v[order], r[order])], width=n + 3))
            assert torch.allclose(bm[i], sm[0], atol=1e-4) and torch.allclose(bt[i], st[0], atol=1e-4)
            assert torch.allclose(br[i, :n][order], sr[0, :n], atol=1e-4)
