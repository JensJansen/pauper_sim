"""Layer 10 -- regression pin. Pins what the encoder pipeline produces for fixed inputs, so any change (a feature column
added or reordered, a card's static vector shifting, an architecture tweak) is a visible, deliberate diff rather than
silent drift. Regenerate on purpose with ENCODER_UPDATE_GOLDEN=1 and commit golden.json.

Pinned: the feature-width constants, a hash of EVERY card's static vector, a hash of the token rows for a fixed busy
position seen from both seats, and the encoder's outputs on it (tolerance 1e-4) under fixed-seed initial weights."""
import hashlib
import json
import os
import pathlib

import numpy as np
import torch

import game
from support import scenario as sc
from rl.model import features as F
from rl.model.arch import pad_token_batch
from rl.roster import TOKEN_DEFS

GOLDEN = pathlib.Path(__file__).with_name("golden.json")
ATOL = 1e-4


def _static_hash():
    h = hashlib.sha256()
    for cd in sorted(list(game.CARD_DEFS.values()) + list(TOKEN_DEFS), key=lambda c: c.name):
        h.update(cd.name.encode())
        h.update(np.asarray(F.static_card_features(cd), dtype=np.float32).tobytes())
    return h.hexdigest()


def _busy():
    from support.scenario import BUSY
    return BUSY


def _current():
    vocab = sc.make_vocab()
    enc = sc.make_encoder(vocab)
    state = sc.build(**_busy())
    out = {"dims": [F.STATIC_FEATURE_DIM, F.DYNAMIC_FEATURE_DIM, F.TOKEN_FEATURE_DIM], "static_hash": _static_hash(),
           "n_cards": len(game.CARD_DEFS)}
    for seat in (0, 1):
        tokens = F.build_token_set(state, seat, vocab)
        rows = np.asarray([r for _i, r, _x in tokens], dtype=np.float32)
        order = np.lexsort(rows.T[::-1])  # order-free hash: sort rows lexicographically first
        out[f"rows_hash_seat{seat}"] = hashlib.sha256(np.round(rows[order], 6).tobytes()).hexdigest()
        vi, ft, mask, _ = pad_token_batch([tokens])
        with torch.no_grad():
            mine, theirs, _reps = enc(vi, ft, mask, ft[:, :, -1])
        out[f"mine_seat{seat}"] = [round(x, 6) for x in mine[0].tolist()]
        out[f"theirs_seat{seat}"] = [round(x, 6) for x in theirs[0].tolist()]
    return out


def test_encoder_pipeline_matches_the_pinned_golden():
    cur = _current()
    if os.environ.get("ENCODER_UPDATE_GOLDEN"):
        GOLDEN.write_text(json.dumps(cur, indent=1))
    assert GOLDEN.exists(), "no golden.json yet: run once with ENCODER_UPDATE_GOLDEN=1 and commit it"
    want = json.loads(GOLDEN.read_text())
    assert cur["dims"] == want["dims"], f"feature widths changed: {want['dims']} -> {cur['dims']}"
    assert cur["n_cards"] == want["n_cards"], "the card pool changed size"
    assert cur["static_hash"] == want["static_hash"], "some card's static feature vector changed"
    for seat in (0, 1):
        assert cur[f"rows_hash_seat{seat}"] == want[f"rows_hash_seat{seat}"], f"seat {seat}: token rows changed"
        for k in (f"mine_seat{seat}", f"theirs_seat{seat}"):
            np.testing.assert_allclose(cur[k], want[k], atol=ATOL, err_msg=f"{k}: encoder output changed")
