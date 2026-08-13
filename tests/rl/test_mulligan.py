"""Tests for rl.mulligan: the per-deck mulligan reward function and
MulliganNet's REINFORCE training.

Flakiness note: the REINFORCE learning assertions in
test_mulligan_net_shapes_and_reinforce_learning are genuinely sensitive to the
net's random init (~1-in-several spurious failures unseeded), so the seeding is
load-bearing and must not be papered over with retries or loosened tolerances.

REVISED 2026-08-13. Seeding ONCE at the top was not enough, and the reason is
worth recording: it made the assertions depend on how many random draws every
object built in between happened to consume. SetTransformer's input_proj is
`Linear(d_model + TOKEN_FEATURE_DIM, ...)`, so when ZONES gained a sixth entry
("known_top") and TOKEN_FEATURE_DIM went 40 -> 41, the RNG stream shifted and
MulliganNet was re-rolled onto an init these asserts fail from -- an unrelated
observation-feature change breaking a mulligan test. torch.manual_seed(0) is
now ALSO called immediately before MulliganNet is constructed, so this test
depends only on its own subject. Feature dims will keep changing.
"""
import random

import torch
import pytest

import game as _game
from game.state import GameState, PlayerState
from rl.arch import SetTransformer
from rl.features import CardVocab
from rl.mulligan import HAND, MULLIGAN_COST, MulliganNet, decide, mulligan_reward, update


@pytest.mark.slow
def test_mulligan_reward_shape():
    # reward shape: big win, CONVEX per-mulligan cost, negative on mull-heavy loss
    assert abs(mulligan_reward(True, 0) - 1.0) < 1e-9
    assert abs(mulligan_reward(True, 3) - (1.0 - MULLIGAN_COST * 9)) < 1e-9   # quadratic penalty
    assert mulligan_reward(False, 3) < 0                                      # loss-after-mulligans is negative
    assert abs(mulligan_reward(False, 0)) < 1e-9                              # kept-and-lost is neutral
    assert mulligan_reward(True, 7) >= 0                                      # a win survives even at the cap
    # convexity: the Nth mulligan must hurt strictly MORE than the (N-1)th
    marg = [mulligan_reward(False, m) - mulligan_reward(False, m + 1) for m in range(7)]
    assert all(marg[i] < marg[i + 1] for i in range(len(marg) - 1)), marg


@pytest.mark.slow
def test_mulligan_net_shapes_and_reinforce_learning():
    torch.manual_seed(0)  # deterministic net init -- the REINFORCE learning asserts below
    random.seed(0)        # are otherwise flaky (~1-in-several spurious failures on random init)

    decklist = _game.parse_decklist_file("../data/mono_blue_terror.txt")
    vocab = CardVocab([decklist])
    shared = SetTransformer(vocab.size, d_model=16, n_heads=2, n_layers=2, dim_feedforward=32)
    for p in shared.parameters():
        p.requires_grad = False
    # Re-seed HERE, immediately before the net whose init the assertions below
    # actually depend on. Seeding only at the top couples this test to how many
    # random draws SetTransformer happens to consume, which is a function of
    # TOKEN_FEATURE_DIM -- so an unrelated observation-feature change (ZONES
    # gained "known_top" on 2026-08-13, 40 -> 41) shifted the RNG stream and
    # re-rolled MulliganNet onto an init the REINFORCE asserts fail from. The
    # feature dim will keep changing; this test's subject is the mulligan net.
    torch.manual_seed(0)
    net = MulliganNet(shared, hidden=32)
    opt = torch.optim.Adam([p for p in net.parameters() if p.requires_grad], lr=1e-2)

    names = sorted({n for n, *_ in decklist})
    rng = random.Random(0)

    def rand_hand(k=7):
        return [vocab.index(rng.choice(names)) for _ in range(k)]

    # heads produce the right shapes and respect the bottom mask
    hi = torch.tensor([rand_hand()], dtype=torch.long)
    sc = torch.tensor([[0.0, 1.0]], dtype=torch.float32)
    logits, value = net.decision(hi, sc)
    assert logits.shape == (1, 2) and value.shape == (1,)
    ci = torch.tensor([[vocab.index(names[0]), vocab.index(names[1]), 0]], dtype=torch.long)
    cm = torch.tensor([[True, True, False]], dtype=torch.bool)
    scores, _ = net.bottom(hi, sc, ci, cm)
    assert scores.shape == (1, 3) and scores[0, 2].item() < -1e7  # padded candidate masked out

    # learning: a synthetic world where mulliganing (action=1) is GOOD -- keeping
    # always loses (reward -0), mulliganing wins (reward WIN - cost). REINFORCE
    # must push the keep/mull head toward "mulligan". Then flip it.
    def train_toward(good_action, steps=300):
        for _ in range(steps):
            batch = []
            for _ in range(32):
                hand = rand_hand()
                # sample the CURRENT policy so it's on-policy
                with torch.inference_mode():
                    lg, _v = net.decision(torch.tensor([hand]), torch.tensor([[0.0, 1.0]]))
                    a = int(torch.distributions.Categorical(logits=lg).sample().item())
                won = (a == good_action)
                r = mulligan_reward(won, mulligans_taken=1 if a == 1 else 0)
                batch.append(["decision", hand, [0.0, 1.0], True, a, r])
            update(net, opt, batch)

    def mull_prob():
        with torch.inference_mode():
            lg, _ = net.decision(torch.tensor([rand_hand()]), torch.tensor([[0.0, 1.0]]))
            return torch.softmax(lg, -1)[0, 1].item()

    train_toward(good_action=1)
    p_mull_when_good = mull_prob()
    assert p_mull_when_good > 0.75, f"should learn to mulligan when mulliganing wins, got {p_mull_when_good:.2f}"
    train_toward(good_action=0)
    p_mull_when_bad = mull_prob()
    assert p_mull_when_bad < 0.25, f"should learn to keep when keeping wins, got {p_mull_when_bad:.2f}"


def _net_and_vocab(decklist_path="../data/mono_blue_terror.txt"):
    decklist = _game.parse_decklist_file(decklist_path)
    vocab = CardVocab([decklist])
    shared = SetTransformer(vocab.size, d_model=16, n_heads=2, n_layers=1, dim_feedforward=32)
    return MulliganNet(shared, hidden=16), vocab, decklist


@pytest.mark.slow
def test_decide_logs_keep_or_mulligan_weights_when_legal():
    net, vocab, decklist = _net_and_vocab()
    names = sorted({n for n, *_ in decklist})
    p0, p1 = PlayerState(on_the_play=True), PlayerState(on_the_play=False)
    p0.hand = [_game.CARD_DEFS[n] for n in names[:7]]
    state = GameState(on_the_play=True, players=[p0, p1], event_log=[])
    state.active_idx = 0
    state.pending_resolution = {"kind": "mulligan_decision"}

    decide(net, vocab, state, seat=0, record=lambda *_: None, greedy=True)
    assert len(state.event_log) == 1
    ev = state.event_log[0]
    assert ev["kind"] == "decision_weights" and ev["network"] == "mulligan_keep"
    assert {c["fixed_label"] for c in ev["candidates"]} == {"Keep", "Mulligan"}
    assert ev["chosen_index"] in (0, 1)


@pytest.mark.slow
def test_decide_skips_logging_a_mulligan_decision_forced_past_the_cap():
    net, vocab, decklist = _net_and_vocab()
    names = sorted({n for n, *_ in decklist})
    p0, p1 = PlayerState(on_the_play=True), PlayerState(on_the_play=False)
    p0.hand = [_game.CARD_DEFS[n] for n in names[:7]]
    p0.mulligans_taken = HAND  # past the cap -- "keep" is the only legal option
    state = GameState(on_the_play=True, players=[p0, p1], event_log=[])
    state.active_idx = 0
    state.pending_resolution = {"kind": "mulligan_decision"}

    decide(net, vocab, state, seat=0, record=lambda *_: None, greedy=True)
    assert state.event_log == [], "a forced (past-cap) mulligan decision must not log decision_weights"


@pytest.mark.slow
def test_decide_logs_bottom_weights_when_multiple_candidates():
    net, vocab, decklist = _net_and_vocab()
    names = sorted({n for n, *_ in decklist})
    p0, p1 = PlayerState(on_the_play=True), PlayerState(on_the_play=False)
    p0.hand = [_game.CARD_DEFS[n] for n in names[:3]]  # 3 unique names -- a real choice
    state = GameState(on_the_play=True, players=[p0, p1], event_log=[])
    state.active_idx = 0
    state.pending_resolution = {"kind": "mulligan_bottom", "remaining": 1}

    decide(net, vocab, state, seat=0, record=lambda *_: None, greedy=True)
    assert len(state.event_log) == 1
    ev = state.event_log[0]
    assert ev["kind"] == "decision_weights" and ev["network"] == "mulligan_bottom"
    assert {c["fixed_label"] for c in ev["candidates"]} <= set(names[:3])


@pytest.mark.slow
def test_decide_skips_logging_a_bottom_pick_forced_single_candidate():
    net, vocab, decklist = _net_and_vocab()
    only_name = decklist[0][0]
    p0, p1 = PlayerState(on_the_play=True), PlayerState(on_the_play=False)
    p0.hand = [_game.CARD_DEFS[only_name]] * 3  # every card shares one name -- nothing real to choose
    state = GameState(on_the_play=True, players=[p0, p1], event_log=[])
    state.active_idx = 0
    state.pending_resolution = {"kind": "mulligan_bottom", "remaining": 1}

    decide(net, vocab, state, seat=0, record=lambda *_: None, greedy=True)
    assert state.event_log == [], "a forced (single-candidate) bottom pick must not log decision_weights"
