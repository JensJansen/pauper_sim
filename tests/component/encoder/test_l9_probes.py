"""Layer 9 -- efficacy probes. Layers 0-8 show the encoder is correct and consistent; they cannot show it CARRIES
the information needed to play. These probes train a fresh encoder (plus a small head) to read facts off real board
states, and must beat a predict-the-mean baseline on games it never saw:

  * counts / sums the policy needs (untapped permanents, their total power, my hand size, their graveyard size,
    creatures in combat, stack size),
  * a boolean (do they have a flyer),
  * a POINTER task: pick their highest-power permanent -- the same query-dot-token_reps mechanism the real pointer
    head uses for targeting, attacking and blocking.

If a probe cannot be fit, information is lost inside the encoder (pooling bottleneck, a masking bug, a dead input).
Costs a few minutes of CPU (about 3). Diagnostic knobs: ENCODER_PROBE_SEED, ENCODER_PROBE_EPOCHS, ENCODER_PROBE_ABLATE."""
import os

import numpy as np
import torch
import torch.nn as nn

from support import corpus as ec
import game
from support import synth
from rl.model import features as F
from rl.model.arch import SetTransformer

D_MODEL = 32
CREATURE_COL = len(game.POOL_COLORS) + 1 + list(F.CARD_TYPES).index(game.CardType.CREATURE)
FLYING_COL = (len(game.POOL_COLORS) + 1 + len(F.CARD_TYPES) + 2 + list(F.KEYWORD_VOCAB).index("flying")
              if "flying" in F.KEYWORD_VOCAB else None)


def _labels(rows):
    """Ground truth read from the (oracle-verified) rows themselves."""
    C = ec.COL
    bf = rows[:, C["zone[battlefield]"]] == 1
    mine, theirs = rows[:, C["side_mine"]] == 1, rows[:, C["side_mine"]] == 0
    return [
        float((bf & mine & (rows[:, C["untapped"]] == 1)).sum()),                       # my untapped permanents
        float(rows[bf & theirs, C["eff_power"]].sum() * ec.PT_CAP),                      # their total power
        float((FLYING_COL is not None) and (rows[bf & theirs, FLYING_COL] == 1).any()),  # they have a flyer
        float((rows[:, C["zone[hand]"]] == 1).sum()),                                    # my hand size
        float(((rows[:, C["zone[graveyard]"]] == 1) & theirs).sum()),                    # their graveyard size
        float((rows[:, C["attacking"]] == 1).sum()),                                     # creatures attacking
        float((rows[:, C["zone[stack]"]] == 1).sum()),                                   # stack size
    ]


LABELS = ["my untapped permanents", "their total power", "they have a flyer", "my hand size", "their graveyard size",
          "creatures attacking", "stack size"]


def _split(frames):
    sets = [f for f in frames if len(f.rows) and not f.forced]
    return [f for f in sets if f.game_id % 5 != 0], [f for f in sets if f.game_id % 5 == 0]  # held out by GAME


def _batches(frs, size, rng):
    order = rng.permutation(len(frs))
    for i in range(0, len(frs), size):
        yield [frs[j] for j in order[i:i + size]]


def _tensors(batch):
    vi, ft, mask = synth.pad([(f.vocab_idx, f.rows) for f in batch])
    ablate = os.environ.get("ENCODER_PROBE_ABLATE", "")  # diagnostic only: zero a group of input columns
    if "behavior" in ablate:
        ft[:, :, F.STATIC_FEATURE_DIM - 119:F.STATIC_FEATURE_DIM] = 0.0
    if "links" in ablate:
        ft[:, :, F.LINK_START:F.LINK_START + F.LINK_FEATURE_DIM] = 0.0
    if "copies" in ablate:
        ft[:, :, F.MULTIPLICITY_START:F.MULTIPLICITY_START + F.MULTIPLICITY_FEATURE_DIM] = 0.0
    return vi, ft, mask


def _r2(pred, y):
    return 1.0 - float(((pred - y) ** 2).sum() / max(((y - y.mean()) ** 2).sum(), 1e-9))


def test_the_encoder_can_be_trained_to_read_the_board(frames, vocab):
    torch.manual_seed(0)
    train, test = _split(frames)
    Y = {id(f): np.asarray(_labels(f.rows), dtype=np.float32) for f in train + test}
    ytr = np.stack([Y[id(f)] for f in train])
    mu, sd = ytr.mean(0), ytr.std(0) + 1e-6
    enc = SetTransformer(vocab.size, d_model=D_MODEL, n_heads=2, n_layers=2, dim_feedforward=64)
    head = nn.Sequential(nn.Linear(2 * enc.summary_dim, 64), nn.ReLU(), nn.Linear(64, len(LABELS)))
    opt = torch.optim.Adam(list(enc.parameters()) + list(head.parameters()), lr=3e-3)
    rng = np.random.default_rng(0)
    for _epoch in range(25):
        enc.train()
        for batch in _batches(train, 64, rng):
            vi, ft, mask = _tensors(batch)
            m, t, _ = enc(vi, ft, mask, ft[:, :, -1])
            y = torch.as_tensor((np.stack([Y[id(f)] for f in batch]) - mu) / sd)
            loss = ((head(torch.cat([m, t], -1)) - y) ** 2).mean()
            opt.zero_grad()
            loss.backward()
            opt.step()
    enc.eval()
    preds, ys = [], []
    with torch.no_grad():
        for batch in _batches(test, 128, rng):
            vi, ft, mask = _tensors(batch)
            m, t, _ = enc(vi, ft, mask, ft[:, :, -1])
            preds.append(head(torch.cat([m, t], -1)).numpy() * sd + mu)
            ys.append(np.stack([Y[id(f)] for f in batch]))
    pred, y = np.concatenate(preds), np.concatenate(ys)
    scores = {name: _r2(pred[:, i], y[:, i]) for i, name in enumerate(LABELS)}
    print("held-out R^2:", {k: round(v, 3) for k, v in scores.items()})
    # Pure COUNT labels (hand size, graveyard size) used to sit near 0.74-0.79 because the encoder could not count
    # identical cards; with copy-count columns and a sum-pooled half they must now be read off almost exactly.
    floor = {"my untapped permanents": 0.9, "their total power": 0.8, "they have a flyer": 0.7, "my hand size": 0.9,
             "their graveyard size": 0.9, "creatures attacking": 0.9, "stack size": 0.9}
    weak = {k: round(v, 3) for k, v in scores.items() if v < floor[k]}
    assert not weak, f"the encoder cannot decode these facts from a board it never saw (R^2 below floor): {weak}"


def test_the_pointer_mechanism_learns_to_pick_the_strongest_opposing_permanent(frames, vocab):
    torch.manual_seed(int(os.environ.get("ENCODER_PROBE_SEED", "1")))
    C = ec.COL

    def candidates(f):
        return (f.rows[:, C["zone[battlefield]"]] == 1) & (f.rows[:, C["side_mine"]] == 0)

    def usable(f):
        c = candidates(f)
        return len(f.rows) and c.sum() >= 2 and len(set(np.round(f.rows[c, C["eff_power"]], 4))) >= 2

    tr, te = _split([f for f in frames if usable(f)])
    assert len(tr) >= 150 and len(te) >= 40, f"only {len(tr)}/{len(te)} usable states; raise ENCODER_CORPUS_GAMES"
    enc = SetTransformer(vocab.size, d_model=D_MODEL, n_heads=2, n_layers=2, dim_feedforward=64)
    query = nn.Linear(2 * enc.summary_dim, D_MODEL)
    opt = torch.optim.Adam(list(enc.parameters()) + list(query.parameters()), lr=3e-3)
    rng = np.random.default_rng(int(os.environ.get("ENCODER_PROBE_SEED", "1")))

    def scores_and_good(batch):
        vi, ft, mask = _tensors(batch)
        m, t, reps = enc(vi, ft, mask, ft[:, :, -1])
        s = torch.bmm(query(torch.cat([m, t], -1)).unsqueeze(1), reps.transpose(1, 2)).squeeze(1) / D_MODEL ** 0.5
        legal = torch.zeros_like(mask)
        good = torch.zeros_like(mask)
        for b, f in enumerate(batch):
            c = candidates(f)
            legal[b, :len(c)] = torch.as_tensor(c)
            top = f.rows[:, C["eff_power"]] * c
            good[b, :len(c)] = torch.as_tensor(c & (np.round(top, 5) == np.round(top.max(), 5)))
        return s.masked_fill(~legal, -1e9), good

    for _epoch in range(int(os.environ.get("ENCODER_PROBE_EPOCHS", "40"))):
        enc.train()
        for batch in _batches(tr, 32, rng):
            s, good = scores_and_good(batch)
            logp = torch.log_softmax(s, -1)
            loss = -(torch.logsumexp(logp.masked_fill(~good, -1e9), -1)).mean()  # probability mass on ANY best target
            opt.zero_grad()
            loss.backward()
            opt.step()
    enc.eval()
    hit = n = 0
    with torch.no_grad():
        for batch in _batches(te, 64, rng):
            s, good = scores_and_good(batch)
            hit += int(good.gather(1, s.argmax(-1, keepdim=True)).sum())
            n += len(batch)
    acc = hit / n
    print(f"pointer accuracy on held-out games: {acc:.3f} ({n} states)")
    assert acc >= 0.85, f"pointer head picked the strongest permanent only {acc:.0%} of the time"
