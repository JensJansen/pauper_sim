"""Plot's own resolve shape (plot_to_exile). Full discard->exile->resolve
chain (execute_madness_cast) is covered in test_integration_check.py."""
from game.cards import CardDef, CardType, EffectId
from game.effects.madness_and_plot import plot_to_exile
from game.state import GameState


def test_plot_to_exile_moves_hand_card_with_turn_stamp():
    state = GameState(on_the_play=True)
    plot_card = CardDef("Fake Plot Spell", CardType.SORCERY, {"generic": 1}, EffectId.FILLER)
    state.hand = [plot_card]
    plot_to_exile(state, plot_card)
    assert state.hand == []
    assert state.exile == [(plot_card, state.turn_number)]
