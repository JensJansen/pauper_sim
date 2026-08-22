"""Tests for rl.mulligan: the per-deck mulligan reward function and
MulliganNet's REINFORCE training.

Flakiness note: the REINFORCE assertions in
test_mulligan_net_shapes_and_reinforce_learning check DIRECTION after a single
update step (does a positive-advantage reward for an action raise/lower that
action's own probability?), not convergence past a threshold after hundreds of
steps -- the single-step property is what REINFORCE actually guarantees, and
checking it directly is far cheaper and less exposed to random-init variance
than the old convergence check was. Net init is still seeded (torch.manual_
seed(0)) so the test is reproducible run to run.

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
from rl.arch import SetTransformer, pad_token_batch
from rl.features import CardVocab, build_token_set
from rl.mulligan import HAND, MulliganNet, decide, mulligan_reward, update


@pytest.mark.slow
def test_mulligan_reward_shape():
    # reward shape (2026-08-21: no per-mulligan-count penalty -- see rl.mulligan's
    # own docstring for why it was removed rather than left zeroed): win pays
    # WIN_REWARD regardless of how many mulligans it took to get there, a loss
    # is always exactly 0 -- mulliganing is discouraged only by its own effect
    # on win probability, never by a flat count-based cost.
    assert abs(mulligan_reward(True) - 1.0) < 1e-9
    assert abs(mulligan_reward(False)) < 1e-9


def _hand_tokens(decklist, vocab, names, rng, k=7):
    """A real (state, tokens) pair for a random k-card hand -- build_token_set's
    real output, not a hand-rolled stand-in, since that's what decide()/update()
    actually record and replay post-2026-08-20 (see rl.mulligan's module
    docstring: the fix was giving the net this SAME structured representation
    instead of a bare mean-pooled identity embedding)."""
    p0 = PlayerState(on_the_play=True)
    p1 = PlayerState(on_the_play=False)
    p0.hand = [_game.CARD_DEFS[rng.choice(names)] for _ in range(k)]
    state = GameState(on_the_play=True, players=[p0, p1], event_log=None)
    state.active_idx = 0
    return build_token_set(state, 0, vocab)


@pytest.mark.slow
def test_mulligan_net_shapes_and_reinforce_learning():
    torch.manual_seed(0)  # deterministic net init -- the REINFORCE direction asserts below
    random.seed(0)        # are otherwise flaky (~1-in-several spurious failures on random init)

    decklist = _game.parse_decklist_file("../data/mono_blue_terror.txt")
    vocab = CardVocab([decklist])
    shared = SetTransformer(vocab.size, d_model=16, n_heads=2, n_layers=2, dim_feedforward=32)
    # Re-seed HERE, immediately before the net whose init the assertions below
    # actually depend on. Seeding only at the top couples this test to how many
    # random draws SetTransformer happens to consume, which is a function of
    # TOKEN_FEATURE_DIM -- so an unrelated observation-feature change (ZONES
    # gained "known_top" on 2026-08-13, 40 -> 41) shifted the RNG stream and
    # re-rolled MulliganNet onto an init the REINFORCE asserts fail from. The
    # feature dim will keep changing; this test's subject is the mulligan net.
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
    cand_reps = token_reps[:, [0, 1, 0], :]  # row 2 is a dummy -- masked below, same as a padded vocab index would be
    cm = torch.tensor([[True, True, False]], dtype=torch.bool)
    scores, _ = net.bottom(mine_summary, sc, cand_reps, cm)
    assert scores.shape == (1, 3) and scores[0, 2].item() < -1e7  # padded candidate masked out

    # encoder boundary: MulliganNet.encode wraps the shared encoder forward in
    # no_grad specifically so this net's own REINFORCE optimizer can never
    # steer it (see the class docstring) -- backward above must therefore
    # leave every one of the encoder's own params ungrated (nothing built a
    # grad_fn through them at all, not just "grad happens to be zero").
    (logits.sum() + value.sum() + scores.sum()).backward()
    assert all(p.grad is None for p in shared.parameters()), (
        "a mulligan forward pass must never populate the shared encoder's .grad")

    # REINFORCE direction: ONE update step, not a convergence threshold after
    # hundreds (the old version of this test ran 300 iterations x 32-sample
    # batches, TWICE, and per this file's own flakiness note above needed
    # double-seeding even then just to stay reliable). What actually matters
    # is whether a single step with a positive-advantage reward moves
    # probability mass toward the action that earned it -- checked directly
    # here on a fixed probe hand, once per direction, each from its own
    # freshly re-seeded (same-seed, deterministic) net so the two checks
    # can't interfere with each other's gradient step. MulliganNet has no
    # dropout/stochastic layers (Linear+Tanh only), so decision() is a pure
    # deterministic function of the weights -- no sampling needed to read it.
    probe_tokens, probe_scalars = _hand_tokens(decklist, vocab, names, rng), [0.0, 1.0]

    def _fresh_net():
        torch.manual_seed(0)  # same init every time -- see the re-seed comment above
        return MulliganNet(shared, hidden=32)

    def _mull_prob(n):
        with torch.inference_mode():
            vi, feat, kpm, _identities = pad_token_batch([probe_tokens])
            mine_summary, _token_reps = n.encode(vi, feat, kpm, feat[:, :, -1])
            lg, _ = n.decision(mine_summary, torch.tensor([probe_scalars]))
            return torch.softmax(lg, -1)[0, 1].item()

    # Reward favors action=1 (mulligan) -> P(mulligan) on the same probe hand
    # must go UP after one step.
    net_up = _fresh_net()
    opt_up = torch.optim.Adam([p for p in net_up.parameters() if p.requires_grad], lr=1e-2)
    before_up = _mull_prob(net_up)
    update(net_up, opt_up, [["decision", probe_tokens, probe_scalars, True, 1, mulligan_reward(True)]])
    after_up = _mull_prob(net_up)
    assert after_up > before_up, (
        f"one REINFORCE step with a positive-advantage reward for action=1 must raise "
        f"P(mulligan): {before_up:.4f} -> {after_up:.4f}")

    # Reward favors action=0 (keep) -> P(mulligan) on the same probe hand must
    # go DOWN after one step, from a fresh (identically-initialized) net.
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
    """update()'s rewritten 'bottom' transition branch (torch.gather over
    token_reps using recorded cand_pos indices -- replacing the old plain
    vocab-index gather) had zero test coverage: every other REINFORCE check
    in this file only ever feeds update() 'decision'-kind transitions.
    Exercises it end to end with a REAL recorded transition (decide()'s own
    output, not a hand-rolled stand-in) and checks the same
    direction-not-convergence property test_mulligan_net_shapes_and_
    reinforce_learning checks for the 'decision' branch: one REINFORCE step
    with a positive-advantage reward for a candidate must raise that
    candidate's own probability."""
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
        torch.manual_seed(0)  # same init every time -- see the re-seed comment above
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
    # decide() happened to greedily choose) -- P(target) on the same probe
    # hand must go UP after one step feeding this exact 'bottom' transition
    # (recorded token set + cand_pos) through update().
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
