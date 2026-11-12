"""The policy must not depend on WHERE a token sits in the observation, nor on how much padding follows it.

rl.model.arch.SetTransformer has no positional encoding and the pointer head scores each token by content
(query . token_rep), so reordering the token list must permute the pointer logits the same way and leave the
fixed-table logits and the critic value unchanged; extra trailing padding must change nothing. This test checks
that on REAL decisions from short self-play games (real token sets, real legal masks), with a randomly initialised
network -- a leak (a mask/padding bug, a positional dependence, an index baked into a feature) shows up as a logit
mismatch, whatever the weights are.

Not covered: slot numbers (never an input to the network) and the fixed table's row order (see
tests/component/action_space/test_action_table_order.py)."""
import pathlib
import random

import numpy as np
import pytest
import torch

import game
import rl.decision.agent as agent_mod
from rl.decision.action_bridge import build_fixed_action_table
from rl.decision.agent import AlwaysKeep, SeatAgent
from rl.model.arch import SetTransformer, pad_token_batch
from rl.model.deck import DeckNetwork
from rl.model.features import CardVocab
from rl.rewards import flat_win_loss_reward
from rl.training.train import _constant_pairing, collect_rollout

DECKLIST = pathlib.Path(__file__).resolve().parents[3] / "data" / "mono_red_madness.txt"  # absolute: this module-scoped fixture runs before the per-test chdir into src/
N_GAMES = 3
HORIZON = 40
PERMUTATIONS_PER_DECISION = 3  # random shuffles, on top of the reversal
EXTRA_PAD = 5
ATOL = 1e-4  # float32; attention sums reorder under a permutation


def _logits(net, dec, order, extra_pad=0):
    """(logits [n_fixed + T], value) for dec with its tokens taken in `order` (a permutation of range(n_tokens)) and
    `extra_pad` trailing padded positions. Fresh recurrent state, no previous action -- one decision in isolation."""
    tokens = [dec.tokens[i] for i in order]
    vocab_idx, features, key_padding_mask, _ = pad_token_batch([tokens])
    if extra_pad:
        vocab_idx = torch.nn.functional.pad(vocab_idx, (0, extra_pad))
        features = torch.nn.functional.pad(features, (0, 0, 0, extra_pad))
        key_padding_mask = torch.nn.functional.pad(key_padding_mask, (0, extra_pad), value=True)
    pointer_half = np.asarray(dec.full_mask[dec.n_fixed:], dtype=bool)[list(order)]
    pointer_mask = torch.zeros((1, vocab_idx.shape[1]), dtype=torch.bool)
    pointer_mask[0, :len(order)] = torch.as_tensor(pointer_half)
    scalar = torch.as_tensor(dec.scalar, dtype=torch.float32).unsqueeze(0)
    with torch.no_grad():
        mine, theirs, token_reps = net.encoder(vocab_idx, features, key_padding_mask, features[:, :, -1])
        logits, value, _ = net(mine, theirs, scalar, token_reps, pointer_mask)
    return logits[0], value[0]


@pytest.fixture(scope="module")
def captured():
    """(net, [non-forced _Decision, ...]) from a few real mirror games. Only _build_decision is wrapped (to record
    what it returns); play itself is untouched."""
    torch.manual_seed(0)
    decklist = game.parse_decklist_file(str(DECKLIST))
    token_defs = (game.BLOOD_TOKEN_CARD_DEF, game.ROBOT_TOKEN_CARD_DEF)
    vocab = CardVocab([decklist], token_card_defs=token_defs)
    fixed_table = build_fixed_action_table(decklist, token_card_defs=token_defs)
    encoder = SetTransformer(vocab.size, d_model=32, n_heads=2, n_layers=2, dim_feedforward=64)
    net = DeckNetwork(encoder, non_targeting_n_actions=len(fixed_table), trunk_hidden=(32, 32))
    net.eval()
    agent = SeatAgent(net, AlwaysKeep(), (vocab, fixed_table))
    pairing = _constant_pairing([agent, agent], [decklist, decklist], [flat_win_loss_reward()] * 2, ["m", "m"])

    decisions = []
    real_build = agent_mod._build_decision

    def recording_build(*args, **kwargs):
        dec = real_build(*args, **kwargs)
        if dec.sole_action is None:  # forced decisions run no network and carry no tokens
            decisions.append(dec)
        return dec

    agent_mod._build_decision = recording_build
    try:
        for seed in range(N_GAMES):
            collect_rollout(pairing, 1, horizon=HORIZON, rng=random.Random(seed), device="cpu")
    finally:
        agent_mod._build_decision = real_build
    return net, decisions


def test_there_are_real_decisions_to_check(captured):
    _net, decisions = captured
    assert len(decisions) >= 20, f"only {len(decisions)} non-forced decisions captured; raise N_GAMES/HORIZON"
    assert any(d.full_mask[d.n_fixed:].any() for d in decisions), "no pointer decision captured; test would be blind to it"


def test_logits_are_equivariant_to_token_order_and_blind_to_padding(captured):
    net, decisions = captured
    rng = random.Random(0)
    spread = 0.0  # guards against a vacuous pass (all logits equal)
    for k, dec in enumerate(decisions):
        n = len(dec.tokens)
        assert len(dec.identities) == n and len(dec.full_mask) == dec.n_fixed + n
        identity = list(range(n))
        base, base_value = _logits(net, dec, identity)
        legal = torch.as_tensor(dec.full_mask, dtype=torch.bool)
        if legal.sum() > 1:
            spread = max(spread, float(base[legal].max() - base[legal].min()))

        # extra trailing padding: nothing may change (pointer width differs, so compare the shared prefix)
        padded, padded_value = _logits(net, dec, identity, extra_pad=EXTRA_PAD)
        width = dec.n_fixed + n
        assert torch.allclose(padded[:width][legal], base[legal], atol=ATOL), f"decision {k}: padding changed logits"
        assert torch.allclose(padded_value, base_value, atol=ATOL), f"decision {k}: padding changed the value"

        orders = [identity[::-1]] + [rng.sample(identity, n) for _ in range(PERMUTATIONS_PER_DECISION)]
        for order in orders:
            got, got_value = _logits(net, dec, order)
            # fixed-table half and the critic are order-free
            assert torch.allclose(got[:dec.n_fixed][legal[:dec.n_fixed]], base[:dec.n_fixed][legal[:dec.n_fixed]],
                                  atol=ATOL), f"decision {k}: fixed-table logits changed under reorder"
            assert torch.allclose(got_value, base_value, atol=ATOL), f"decision {k}: value changed under reorder"
            # pointer half: new position j holds old token order[j], so its score must be that token's old score
            for j, old in enumerate(order):
                if legal[dec.n_fixed + old]:
                    assert abs(float(got[dec.n_fixed + j] - base[dec.n_fixed + old])) <= ATOL, (
                        f"decision {k}: token {old} scored differently at position {j}")
    assert spread > 1e-3, f"logits were all ~equal (spread {spread:.2e}) -- the comparison above proved nothing"
