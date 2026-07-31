"""Plot's own resolve shape (plot_to_exile).

execute_madness_cast is exercised together with triggers.py/stack.py's own
machinery -- the full discard -> exile + queue -> drain -> decision -> pay ->
resolve chain -- in tests/game/effects/test_integration_check.py, since it's
genuinely a multi-module scenario, not just this one function."""
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
