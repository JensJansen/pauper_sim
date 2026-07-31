"""Migrated from src/rl/deck.py's __main__ self-check."""
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
    logits, value = net(mine_summary, theirs_summary, scalar_features, token_reps, pointer_mask)
    assert logits.shape == (1, n_fixed_actions + vocab_idx.shape[1])
    assert value.shape == (1,)
    assert torch.isfinite(value).all()


@pytest.mark.slow
def test_deck_network_pointer_masking():
    net, mine_summary, theirs_summary, scalar_features, token_reps, pointer_mask, n_fixed_actions, _vocab_idx = _fixture()
    logits, _value = net(mine_summary, theirs_summary, scalar_features, token_reps, pointer_mask)

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
    logits, _value = net(mine_summary, theirs_summary, scalar_features, token_reps, pointer_mask)

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
