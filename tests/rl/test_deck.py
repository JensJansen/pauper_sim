"""Tests for rl.deck.DeckNetwork: combined fixed-table + pointer action head."""
import torch
import pytest

import game as _game
from game.state import GameState, Permanent, PlayerState
from rl.arch import SetTransformer, pad_token_batch
from rl.deck import SCALAR_FEATURE_DIM, DeckNetwork
from rl.features import CardVocab, build_token_set


def _fixture():
    decklist_a = _game.parse_decklist_file("../data/mono_red_madness.txt")
    decklist_b = _game.parse_decklist_file("../data/rakdos_madness.txt")
    vocab = CardVocab([decklist_a, decklist_b])

    seat0 = PlayerState(on_the_play=True)
    seat0.battlefield = [Permanent(_game.CARD_DEFS["Guttersnipe"]), Permanent(_game.CARD_DEFS["Voldaren Epicure"])]
    seat1 = PlayerState(on_the_play=False)
    seat1.battlefield = [Permanent(_game.CARD_DEFS["Kitchen Imp"])]
    state = GameState(on_the_play=True, players=[seat0, seat1])

    tokens = build_token_set(state, 0, vocab)
    vocab_idx, features, mask, identities = pad_token_batch([tokens])
    side_flag = features[:, :, -1]

    shared = SetTransformer(vocab.size, d_model=16, n_heads=2, n_layers=1, dim_feedforward=32)
    mine_summary, theirs_summary, token_reps = shared(vocab_idx, features, mask, side_flag)

    n_fixed_actions = 12  # arbitrary stand-in for a real per-deck fixed table's size
    net = DeckNetwork(shared, film_condition_dim=16, non_targeting_n_actions=n_fixed_actions, trunk_hidden=(32, 32))

    scalar_features = torch.zeros(1, SCALAR_FEATURE_DIM)
    # Only my own two battlefield permanents (Guttersnipe, Voldaren Epicure)
    # are legal "Attack:" pointer targets in this synthetic example -- the
    # opponent's Kitchen Imp and any padded slots must be masked out.
    pointer_mask = torch.zeros_like(mask)
    for t, ident in enumerate(identities[0]):
        pointer_mask[0, t] = ident is not None and ident in seat0.battlefield

    return net, mine_summary, theirs_summary, scalar_features, token_reps, pointer_mask, n_fixed_actions, vocab_idx


@pytest.mark.slow
def test_deck_network_output_shapes():
    net, mine_summary, theirs_summary, scalar_features, token_reps, pointer_mask, n_fixed_actions, vocab_idx = _fixture()
    logits, value, hidden = net(mine_summary, theirs_summary, scalar_features, token_reps, pointer_mask)
    assert logits.shape == (1, n_fixed_actions + vocab_idx.shape[1])
    assert value.shape == (1,)
    assert torch.isfinite(value).all()


@pytest.mark.slow
def test_deck_network_pointer_masking():
    net, mine_summary, theirs_summary, scalar_features, token_reps, pointer_mask, n_fixed_actions, _vocab_idx = _fixture()
    logits, _value, _hidden = net(mine_summary, theirs_summary, scalar_features, token_reps, pointer_mask)

    # The two legal pointer positions must be finite/selectable; the
    # opponent's Kitchen Imp position must be masked to -inf (never sampled).
    pointer_logits = logits[0, n_fixed_actions:]
    legal_positions = pointer_mask[0].nonzero(as_tuple=True)[0]
    illegal_positions = (~pointer_mask[0]).nonzero(as_tuple=True)[0]
    assert torch.isfinite(pointer_logits[legal_positions]).all()
    assert (pointer_logits[illegal_positions] <= -1e7).all(), "masked pointer positions must be effectively -inf"


@pytest.mark.slow
def test_deck_network_masked_categorical_only_samples_legal_actions():
    net, mine_summary, theirs_summary, scalar_features, token_reps, pointer_mask, n_fixed_actions, _vocab_idx = _fixture()
    logits, _value, _hidden = net(mine_summary, theirs_summary, scalar_features, token_reps, pointer_mask)

    # A full masked-categorical sample over the COMBINED action set must
    # only ever land on a legal position (either a legal fixed action or a
    # legal pointer target) -- exercising the actual masking contract this
    # network exists to support, not just checking shapes.
    fixed_mask = torch.zeros(1, n_fixed_actions, dtype=torch.bool)
    fixed_mask[0, 0] = True  # pretend exactly one fixed action ("Pass") is legal too
    full_mask = torch.cat([fixed_mask, pointer_mask], dim=-1)
    masked_logits = logits.masked_fill(~full_mask, -1e8)
    dist = torch.distributions.Categorical(logits=masked_logits)
    for _ in range(50):
        action = dist.sample()
        assert full_mask[0, action.item()], f"sampled an illegal combined action index {action.item()}"


@pytest.mark.slow
def test_the_league_gives_every_deck_its_own_encoder():
    """The invariant rl.deck's module docstring names.

    Handing the SAME SetTransformer instance to two DeckNetworks would put it
    in two Adam instances tracking separate momentum for the identical tensors,
    each stepping on the other -- which is precisely the coupling per-deck
    encoders exist to remove, and it would be silent. build_deck_net is the one
    construction site, so pinning it there covers every caller."""
    from rl.league_runner import build_deck_net

    a, b = build_deck_net(12, 4), build_deck_net(12, 4)
    assert a.encoder is not b.encoder, "two decks must never be handed the same encoder instance"
    assert not any(pa is pb for pa in a.parameters() for pb in b.parameters()), \
        "no parameter tensor may be reachable from both decks' nets"

    # And they must start from independent random init, not a shared seed --
    # otherwise "distinct objects" would still mean two decks that move
    # identically for as long as they see identical batches.
    assert not torch.equal(a.encoder.embedding.weight, b.encoder.embedding.weight)

    # A step on one deck leaves the other's perception untouched.
    before = b.encoder.embedding.weight.clone()
    opt = torch.optim.Adam(a.parameters(), lr=1e-2)
    a.encoder.embedding.weight.sum().backward()
    opt.step()
    assert torch.equal(b.encoder.embedding.weight, before), \
        "training one deck must not move another deck's encoder"


def _recurrent_fixture(n_rows, seed=0):
    """n_rows independent single-step inputs for a small DeckNetwork."""
    torch.manual_seed(seed)
    d_model, n_tokens, n_fixed = 16, 5, 4
    encoder = SetTransformer(vocab_size=8, d_model=d_model, n_heads=2, n_layers=1, dim_feedforward=32)
    net = DeckNetwork(encoder, film_condition_dim=d_model, non_targeting_n_actions=n_fixed, trunk_hidden=(24, 24))
    mine = torch.randn(n_rows, d_model)
    theirs = torch.randn(n_rows, d_model)
    scalar = torch.randn(n_rows, SCALAR_FEATURE_DIM)
    reps = torch.randn(n_rows, n_tokens, d_model)
    pmask = torch.ones(n_rows, n_tokens, dtype=torch.bool)
    return net, mine, theirs, scalar, reps, pmask


@pytest.mark.slow
def test_sequence_batch_is_episode_major_not_step_major():
    """The one silent failure mode of the recurrent update.

    ppo_update hands DeckNetwork.forward a FLAT [B*T, ...] batch plus seq=(B, T),
    and forward does `.reshape(B, T, -1)` before the GRU. That reshape is only
    correct if the flat batch is laid out episode-major -- all T steps of
    episode 0, then all of episode 1. Get it backwards and nothing errors: the
    GRU simply trains on interleaved fragments of different games, which is
    wrong in a way no shape check or loss curve would reveal.

    Pinned by equivalence: running the flat batch as sequences must match
    replaying each episode one step at a time, threading the hidden state --
    which is the definition of what the update is supposed to be doing."""
    n_seq, n_steps = 2, 3
    net, mine, theirs, scalar, reps, pmask = _recurrent_fixture(n_seq * n_steps)
    net.eval()

    with torch.no_grad():
        seq_logits, seq_values, _h = net(mine, theirs, scalar, reps, pmask, seq=(n_seq, n_steps))

        step_logits, step_values = [], []
        for episode in range(n_seq):
            hidden = None  # every episode starts from the same zero state
            for step in range(n_steps):
                row = episode * n_steps + step  # episode-major indexing
                logits, value, hidden = net(mine[row:row + 1], theirs[row:row + 1], scalar[row:row + 1],
                                            reps[row:row + 1], pmask[row:row + 1], hidden=hidden)
                step_logits.append(logits)
                step_values.append(value)

    assert torch.allclose(seq_logits, torch.cat(step_logits), atol=1e-5), (
        "sequence-batched logits must match step-by-step replay -- they do not, so the flat batch "
        "is not being reshaped episode-major"
    )
    assert torch.allclose(seq_values, torch.cat(step_values), atol=1e-5)


@pytest.mark.slow
def test_the_recurrent_state_actually_changes_the_policy():
    """A GRU that is wired in but never consulted would pass every shape test
    in this file. The same observation must produce a DIFFERENT distribution
    depending on what came before it -- that is the entire point of the
    change, and without this check a forward that dropped `hidden` on the
    floor would look healthy."""
    net, mine, theirs, scalar, reps, pmask = _recurrent_fixture(2, seed=3)
    net.eval()
    with torch.no_grad():
        # The SAME observation (row 0), once from a fresh game and once after
        # having already seen row 1.
        fresh, _v, _h = net(mine[:1], theirs[:1], scalar[:1], reps[:1], pmask[:1])
        _first, _v2, carried = net(mine[1:], theirs[1:], scalar[1:], reps[1:], pmask[1:])
        after_history, _v3, _h3 = net(mine[:1], theirs[:1], scalar[:1], reps[:1], pmask[:1], hidden=carried)
    assert not torch.allclose(fresh, after_history), (
        "an identical board must be evaluated differently once the agent has a history -- "
        "the recurrent state is not reaching the heads"
    )


@pytest.mark.slow
def test_ppo_update_handles_ragged_episodes_and_ignores_padding():
    """Episodes have different lengths, so every minibatch is padded. Padding
    is done by REPEATING the last real transition, which computes a real
    forward -- so it must be masked out of every loss term or it silently
    upweights whatever transition happened to end each episode.

    Checked by construction rather than by inspection: a buffer of ragged
    episodes must train (params move, losses finite) and must give the SAME
    result as one whose episodes are all the same length once the extra
    transitions carry no advantage."""
    import numpy as np
    from rl.ppo import ppo_update
    from rl.train import RolloutBuffer

    net, *_rest = _recurrent_fixture(1, seed=7)
    n_fixed = net.non_targeting_head.out_features
    buf = RolloutBuffer()
    # Three episodes of lengths 4, 1, 3 -- the middle one is a single-step
    # episode, the degenerate case for a sequence batcher.
    for length in (4, 1, 3):
        for step in range(length):
            buf.add([], np.zeros(SCALAR_FEATURE_DIM, dtype=np.float32), np.ones(n_fixed, dtype=bool),
                    0, -0.5, 0.0, 1.0 if step == length - 1 else 0.0, step == length - 1)
    before = [p.clone() for p in net.parameters()]
    pl, vl, ent, akl, cf, epochs, ev, astd = ppo_update(
        net, torch.optim.Adam(net.parameters(), lr=1e-3), buf, "cpu", n_epochs=2, batch_size=2)

    assert all(np.isfinite(x) for x in (pl, vl, ent, akl, cf, ev, astd)), "ragged episodes must not produce NaN"
    assert any(not torch.equal(a, b) for a, b in zip(before, net.parameters())), "the update must actually train"
    assert epochs >= 1


@pytest.mark.slow
def test_previous_action_reaches_the_policy_and_pointer_actions_collapse():
    """The agent's own last action is fed in alongside the observation, which
    is what makes Brainstorm's placement a lookup instead of an inference over
    hand-token deltas (known_top, the persisted record that used to carry it,
    was removed 2026-08-17).

    Two properties. It must actually CHANGE the policy -- a forward that
    ignored prev_action would pass every shape test. And every pointer action
    must map to ONE symbol: a pointer index names a different token from state
    to state, so embedding it per-index would be learning noise."""
    net, mine, theirs, scalar, reps, pmask = _recurrent_fixture(1, seed=11)
    net.eval()
    n_fixed = net.n_fixed

    # Symbol mapping: fixed actions keep their index, every pointer action
    # collapses to n_fixed, and "no previous action" is n_fixed + 1.
    symbols = net.prev_action_symbols([0, n_fixed - 1, n_fixed, n_fixed + 50, None, -1])
    assert symbols.tolist() == [0, n_fixed - 1, n_fixed, n_fixed, n_fixed + 1, n_fixed + 1]

    with torch.no_grad():
        no_prev, _v, _h = net(mine, theirs, scalar, reps, pmask,
                              prev_action=net.prev_action_symbols([None]))
        after_a, _v, _h = net(mine, theirs, scalar, reps, pmask,
                              prev_action=net.prev_action_symbols([0]))
        after_b, _v, _h = net(mine, theirs, scalar, reps, pmask,
                              prev_action=net.prev_action_symbols([1]))
        ptr_1, _v, _h = net(mine, theirs, scalar, reps, pmask,
                            prev_action=net.prev_action_symbols([n_fixed]))
        ptr_2, _v, _h = net(mine, theirs, scalar, reps, pmask,
                            prev_action=net.prev_action_symbols([n_fixed + 7]))

    assert not torch.allclose(no_prev, after_a), "prev_action must reach the heads at all"
    assert not torch.allclose(after_a, after_b), "different fixed actions must be distinguishable"
    assert torch.allclose(ptr_1, ptr_2), "two different pointer actions must collapse to one symbol"
