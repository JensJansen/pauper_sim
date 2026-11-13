"""The encoder half of the Rooftop Percher scenario (the engine's own test covers the pointer/legality half):
a graveyard copy that a stack entry targets carries the targeted_by_theirs bit, its same-named sibling does not."""
import game
from game.state import CardInstance, GameState, Permanent, PlayerState
from rl.model.features import SLOTS_AFTER_TARGETING, CardVocab, build_token_set


def test_percher_target_shows_targeted_by_theirs_only_on_the_targeted_copy():
    decklist = game.parse_decklist_file("../data/mono_red_madness.txt")
    vocab = CardVocab([decklist], token_card_defs=(game.BLOOD_TOKEN_CARD_DEF, game.ROBOT_TOKEN_CARD_DEF))
    dart_def = game.CARD_DEFS["Lava Dart"]
    sc_me, sc_opp = PlayerState(on_the_play=True), PlayerState(on_the_play=False)
    t_a, t_b = CardInstance(dart_def), CardInstance(dart_def)
    sc_me.graveyard = [t_a, t_b]
    sc_me.battlefield = [Permanent(game.CARD_DEFS["Mountain"])]
    sc_state = GameState(on_the_play=True, players=[sc_me, sc_opp])
    # Percher is the opponent's permanent, so its trigger's controller must be seat 1 (push_to_stack records
    # state.active_idx as the controller, deciding targeted_by_mine vs targeted_by_theirs from seat 0's view).
    sc_state.active_idx = 1
    game.push_to_stack(sc_state, game.CARD_DEFS["Rooftop Percher"], lambda st, _cd: None,
                       reserves_hand_card=False, is_spell=False, targets=(("graveyard_card", t_a),))
    sc_state.active_idx = 0
    tokens = build_token_set(sc_state, 0, vocab)
    row_a = next(row for _i, row, ident in tokens if ident is t_a)
    row_b = next(row for _i, row, ident in tokens if ident is t_b)
    tgt_theirs = -SLOTS_AFTER_TARGETING - 1  # counted back from the row's end, past what _token_row appends after the pair
    assert row_a[tgt_theirs] == 1.0 and row_b[tgt_theirs] == 0.0
