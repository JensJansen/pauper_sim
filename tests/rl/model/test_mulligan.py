"""Tests for rl.model.mulligan: the per-deck mulligan reward function and
MulliganNet's REINFORCE training.

The REINFORCE assertions in test_mulligan_net_shapes_and_reinforce_learning
check direction after a single update step (does a positive-advantage
reward for an action raise/lower that action's own probability?), which is
what REINFORCE actually guarantees, rather than convergence past a
threshold after many steps. Net init is seeded (torch.manual_seed(0))
immediately before MulliganNet is constructed -- not just once at the top --
so the test doesn't depend on how many random draws unrelated objects
(e.g. SetTransformer, sized by TOKEN_FEATURE_DIM) consume first.
"""
import random

import torch
import pytest

import game as _game
from game.state import GameState, PlayerState
from rl.model.arch import SetTransformer, pad_token_batch
from rl.model.features import CardVocab, build_token_set
from rl.model.mulligan import HAND, MulliganNet, decide, mulligan_reward, update


@pytest.mark.slow
def test_mulligan_reward_shape():
    # No per-mulligan-count penalty: win pays WIN_REWARD regardless of how
    # many mulligans it took, a loss is always exactly 0.
    assert abs(mulligan_reward(True) - 1.0) < 1e-9
    assert abs(mulligan_reward(False)) < 1e-9


def _hand_tokens(decklist, vocab, names, rng, k=7):
    """A real (state, tokens) pair for a random k-card hand -- build_token_set's
    real output, not a hand-rolled stand-in, since that's what decide()/
    update() actually record and replay."""
    p0 = PlayerState(on_the_play=True)
    p1 = PlayerState(on_the_play=False)
    p0.hand = [_game.CARD_DEFS[rng.choice(names)] for _ in range(k)]
    state = GameState(on_the_play=True, players=[p0, p1], event_log=None)
    state.active_idx = 0
    return build_token_set(state, 0, vocab)


@pytest.mark.slow
def test_mulligan_net_shapes_and_reinforce_learning():
    torch.manual_seed(0)  # deterministic net init: REINFORCE direction asserts are otherwise flaky
    random.seed(0)

    decklist = _game.parse_decklist_file("../data/mono_blue_terror.txt")
    vocab = CardVocab([decklist])
    shared = SetTransformer(vocab.size, d_model=16, n_heads=2, n_layers=2, dim_feedforward=32)
    # Re-seed immediately before the net whose init the assertions depend on,
    # so this test doesn't depend on how many random draws SetTransformer
    # (sized by TOKEN_FEATURE_DIM) consumes first.
    torch.manual_seed(0)
    net = MulliganNet(shared, hidden=32)

    names = sorted({n for n, *_ in decklist})
    rng = random.Random(0)

    # heads produce the right shapes and respect the bottom mask
    tokens = _hand_tokens(decklist, vocab, names, rng)
    vi, feat, kpm, _identities = pad_token_batch([tokens])
    side = feat[:, :, -1]
    mine_summary, token_reps = net.encode(vi, feat, kpm, side)
    sc = torch.tensor([[0.0, 1.0]], dtype=torch.float32)
    logits, value = net.decision(mine_summary, sc)
    assert logits.shape == (1, 2) and value.shape == (1,)
    cand_reps = token_reps[:, [0, 1, 0], :]  # row 2 is a dummy, masked below
    cm = torch.tensor([[True, True, False]], dtype=torch.bool)
    scores, _ = net.bottom(mine_summary, sc, cand_reps, cm)
    assert scores.shape == (1, 3) and scores[0, 2].item() < -1e7  # padded candidate masked out

    # MulliganNet.encode wraps the shared encoder forward in no_grad so this
    # net's own optimizer never steers it -- backward must leave every
    # encoder param ungraded (no grad_fn reaches them at all).
    (logits.sum() + value.sum() + scores.sum()).backward()
    assert all(p.grad is None for p in shared.parameters()), (
        "a mulligan forward pass must never populate the shared encoder's .grad")

    # REINFORCE direction: one update step, checking whether a
    # positive-advantage reward moves probability mass toward the action
    # that earned it -- on a fixed probe hand, once per direction, each from
    # its own freshly re-seeded net so the two checks don't interfere.
    # MulliganNet has no dropout/stochastic layers, so decision() is a pure
    # deterministic function of the weights.
    probe_tokens, probe_scalars = _hand_tokens(decklist, vocab, names, rng), [0.0, 1.0]

    def _fresh_net():
        torch.manual_seed(0)
        return MulliganNet(shared, hidden=32)

    def _mull_prob(n):
        with torch.inference_mode():
            vi, feat, kpm, _identities = pad_token_batch([probe_tokens])
            mine_summary, _token_reps = n.encode(vi, feat, kpm, feat[:, :, -1])
            lg, _ = n.decision(mine_summary, torch.tensor([probe_scalars]))
            return torch.softmax(lg, -1)[0, 1].item()

    # Reward favors action=1 (mulligan): P(mulligan) must go up after one step.
    net_up = _fresh_net()
    opt_up = torch.optim.Adam([p for p in net_up.parameters() if p.requires_grad], lr=1e-2)
    before_up = _mull_prob(net_up)
    update(net_up, opt_up, [["decision", probe_tokens, probe_scalars, True, 1, mulligan_reward(True)]])
    after_up = _mull_prob(net_up)
    assert after_up > before_up, (
        f"one REINFORCE step with a positive-advantage reward for action=1 must raise "
        f"P(mulligan): {before_up:.4f} -> {after_up:.4f}")

    # Reward favors action=0 (keep): P(mulligan) must go down after one step.
    net_down = _fresh_net()
    opt_down = torch.optim.Adam([p for p in net_down.parameters() if p.requires_grad], lr=1e-2)
    before_down = _mull_prob(net_down)
    update(net_down, opt_down, [["decision", probe_tokens, probe_scalars, True, 0, mulligan_reward(True)]])
    after_down = _mull_prob(net_down)
    assert after_down < before_down, (
        f"one REINFORCE step with a positive-advantage reward for action=0 must lower "
        f"P(mulligan): {before_down:.4f} -> {after_down:.4f}")


@pytest.mark.slow
def test_mulligan_net_bottom_branch_reinforce_direction():
    """Exercises update()'s 'bottom' transition branch (torch.gather over
    token_reps using recorded cand_pos indices) end to end with a real
    recorded transition (decide()'s own output). Checks the same
    direction-not-convergence property as the 'decision' branch test: one
    REINFORCE step with a positive-advantage reward for a candidate must
    raise that candidate's probability."""
    torch.manual_seed(0)
    random.seed(0)

    decklist = _game.parse_decklist_file("../data/mono_blue_terror.txt")
    vocab = CardVocab([decklist])
    shared = SetTransformer(vocab.size, d_model=16, n_heads=2, n_layers=2, dim_feedforward=32)
    torch.manual_seed(0)
    net = MulliganNet(shared, hidden=32)

    names = sorted({n for n, *_ in decklist})
    p0, p1 = PlayerState(on_the_play=True), PlayerState(on_the_play=False)
    p0.hand = [_game.CARD_DEFS[n] for n in names[:3]]  # 3 unique names -- a real bottom choice
    state = GameState(on_the_play=True, players=[p0, p1], event_log=None)
    state.active_idx = 0
    state.pending_resolution = {"kind": "mulligan_bottom", "remaining": 1}

    sink = []
    decide(net, vocab, state, seat=0, record=sink.append, greedy=True)
    assert len(sink) == 1 and sink[0][0] == "bottom"
    _, tokens, scalars, cand_pos, _chosen, _reward = sink[0]
    assert len(cand_pos) == 3  # one row per distinct hand name

    def _fresh_net():
        torch.manual_seed(0)
        return MulliganNet(shared, hidden=32)

    def _bottom_probs(n):
        with torch.inference_mode():
            vi, feat, kpm, _identities = pad_token_batch([tokens])
            mine_summary, token_reps = n.encode(vi, feat, kpm, feat[:, :, -1])
            sc = torch.tensor([scalars], dtype=torch.float32)
            cand_reps = token_reps[:, cand_pos, :]
            cm = torch.ones((1, len(cand_pos)), dtype=torch.bool)
            scores, _ = n.bottom(mine_summary, sc, cand_reps, cm)
            return torch.softmax(scores, -1)[0]

    # Reward a fixed target candidate (index 0 into cand_pos, not whatever
    # decide() happened to greedily choose): P(target) must go up after one step.
    target = 0
    net_up = _fresh_net()
    opt_up = torch.optim.Adam([p for p in net_up.parameters() if p.requires_grad], lr=1e-2)
    before = _bottom_probs(net_up)[target].item()
    update(net_up, opt_up, [["bottom", tokens, scalars, cand_pos, target, 1.0]])
    after = _bottom_probs(net_up)[target].item()
    assert after > before, (
        f"one REINFORCE step on update()'s 'bottom' branch with a positive-advantage "
        f"reward for candidate {target} must raise its probability: {before:.4f} -> {after:.4f}")


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
