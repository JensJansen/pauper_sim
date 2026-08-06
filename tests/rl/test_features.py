"""Tests for rl.features: CardVocab indexing/persistence and build_token_set tokenization."""
import os
import shutil
import tempfile

import pytest

import game
from game.state import CardInstance, GameState, Permanent, PlayerState
from rl.features import (
    CARD_TYPES,
    MANA_PIP_CAP,
    POWER_TOUGHNESS_CAP,
    STATIC_FEATURE_DIM,
    TOKEN_FEATURE_DIM,
    ZONES,
    CardVocab,
    build_token_set,
    cached_static_card_features,
)


def _base_vocab():
    decklist_a = game.parse_decklist_file("../data/mono_red_madness.txt")
    decklist_b = game.parse_decklist_file("../data/rakdos_madness.txt")
    token_defs = (game.BLOOD_TOKEN_CARD_DEF,)
    vocab = CardVocab([decklist_a, decklist_b], token_card_defs=token_defs)
    return decklist_a, decklist_b, vocab


@pytest.mark.slow
def test_card_vocab_basic_indexing():
    decklist_a, decklist_b, vocab = _base_vocab()
    all_names = sorted({name for name, *_r in decklist_a} | {name for name, *_r in decklist_b})
    assert len(all_names) == 20, f"expected 20 distinct cards across both decks, got {len(all_names)}"
    assert vocab.size == 22, f"expected vocab size 22 (20 cards + Blood token + padding sentinel), got {vocab.size}"
    assert vocab.index("Lightning Bolt") != 0
    assert vocab.index("Nonexistent Card") == 0, "unknown card must map to the padding/unknown sentinel"
    assert vocab.index("Blood") != 0, "the Blood token must get a real vocabulary index, not fall back to unknown"
    # Every real card (+ token) gets a UNIQUE index, never colliding with another real entry or the sentinel.
    indices = [vocab.index(n) for n in all_names] + [vocab.index("Blood")]
    assert len(set(indices)) == len(indices), "card/token indices must be unique"
    assert 0 not in indices, "no real card/token may be assigned the padding sentinel's index"


@pytest.mark.slow
def test_static_card_features_spell_and_creature():
    _decklist_a, _decklist_b, vocab = _base_vocab()
    feat = cached_static_card_features("Lightning Bolt", vocab)
    assert len(feat) == STATIC_FEATURE_DIM
    # Lightning Bolt costs {R: 1} -- exactly one nonzero mana-cost dim (R), rest of cost zero, not a creature (P/T both 0).
    r_idx = game.POOL_COLORS.index("R")
    assert feat[r_idx] == 1 / MANA_PIP_CAP
    assert all(v == 0.0 for i, v in enumerate(feat[:len(game.POOL_COLORS) + 1]) if i != r_idx)
    pt_start = len(game.POOL_COLORS) + 1 + len(CARD_TYPES)
    assert feat[pt_start] == 0.0 and feat[pt_start + 1] == 0.0, "Lightning Bolt is not a creature -- base P/T must be 0"

    feat_creature = cached_static_card_features("Guttersnipe", vocab)
    assert feat_creature[pt_start] == 2 / POWER_TOUGHNESS_CAP  # Guttersnipe is a 2/2
    assert feat_creature[pt_start + 1] == 2 / POWER_TOUGHNESS_CAP


@pytest.mark.slow
def test_static_card_features_land_no_crash():
    _decklist_a, _decklist_b, vocab = _base_vocab()
    # Mountain (a land, cast_cost=None) must not crash on cost.get(...) -- the `or {}` guard.
    feat_land = cached_static_card_features("Mountain", vocab)
    assert len(feat_land) == STATIC_FEATURE_DIM
    assert all(v == 0.0 for v in feat_land[:len(game.POOL_COLORS) + 1])
    type_start = len(game.POOL_COLORS) + 1
    land_type_idx = type_start + CARD_TYPES.index(game.CardType.LAND)
    assert feat_land[land_type_idx] == 1.0


@pytest.mark.slow
def test_static_card_features_token_resolves_via_vocab():
    _decklist_a, _decklist_b, vocab = _base_vocab()
    # Blood (a token, NOT in game.CARD_DEFS) -- must resolve via the vocab's
    # own merged card_def_by_name, not crash with a KeyError.
    feat_blood = cached_static_card_features("Blood", vocab)
    assert len(feat_blood) == STATIC_FEATURE_DIM
    type_start = len(game.POOL_COLORS) + 1
    artifact_type_idx = type_start + CARD_TYPES.index(game.CardType.ARTIFACT)
    assert feat_blood[artifact_type_idx] == 1.0


@pytest.mark.slow
def test_card_vocab_shared_card_same_index_across_decks():
    decklist_a, decklist_b, vocab = _base_vocab()
    # A card shared between both decks gets the SAME index/features regardless of which deck's list it's looked up through.
    assert vocab.index("Lightning Bolt") == vocab.name_to_index["Lightning Bolt"]
    shared_names = ({n for n, *_r in decklist_a} & {n for n, *_r in decklist_b})
    assert "Lightning Bolt" in shared_names and len(shared_names) == 8


@pytest.mark.slow
def test_card_vocab_dfc_back_face_alias():
    _decklist_a, _decklist_b, vocab = _base_vocab()
    # DFC back face (Delver of Secrets -> Insectile Aberration): the back name must
    # resolve to a real CardDef (no KeyError when a transformed permanent is observed)
    # and ALIAS to its front face's vocab index, WITHOUT changing vocab.size (the
    # shared embedding table / checkpoints stay valid).
    assert vocab.card_def("Insectile Aberration").name == "Insectile Aberration"  # registered, resolves
    dfc_vocab = CardVocab([game.parse_decklist_file("../data/mono_blue_terror.txt")])
    assert dfc_vocab.index("Delver of Secrets") != 0  # real decklist card
    assert dfc_vocab.index("Insectile Aberration") == dfc_vocab.index("Delver of Secrets")  # aliased to the front
    _size_no_dfc = max(v for k, v in dfc_vocab.name_to_index.items() if k != "Insectile Aberration") + 1
    assert dfc_vocab.size == _size_no_dfc, "the back-face alias must not grow vocab.size"


@pytest.mark.slow
def test_build_token_set_both_seats_all_zones_and_targeting():
    # build_token_set, against a hand-built two-seat state covering every
    # public zone -- specifically constructed with seat 1 (not seat 0) as
    # "my_seat_idx" for part of this check, since a bug that only shows up
    # when my_seat_idx=1 (using state.players[0] unconditionally instead of
    # the permanent's own true owner_idx) would pass a seat-0-only test.
    _decklist_a, _decklist_b, vocab = _base_vocab()
    seat0 = PlayerState(on_the_play=True)
    seat0.battlefield = [Permanent(game.CARD_DEFS["Guttersnipe"]), Permanent(game.CARD_DEFS["Mountain"], tapped=True)]
    seat0.graveyard = [CardInstance(game.CARD_DEFS["Lightning Bolt"])]  # graveyard holds CardInstances now
    seat1 = PlayerState(on_the_play=False)
    seat1.battlefield = [Permanent(game.CARD_DEFS["Kitchen Imp"])]
    seat1.exile = [(game.CARD_DEFS["Faithless Looting"], 3)]
    fs_state = GameState(on_the_play=True, players=[seat0, seat1])
    # Three stack entries exercising all three OBJECT target kinds
    # push_to_stack's targets= can carry (the fourth, "player", has no token
    # to attach to -- see rl.agent._scalar_features's own self-check).
    # Card-to-target pairings here are mechanically synthetic (not real
    # card text) -- this only exercises the plumbing, not card legality:
    # Fiery Temper (controller=1, "theirs" from seat0's own perspective)
    # targets seat0's own Guttersnipe (a "creature"/permanent target);
    # Lightning Bolt (controller=0) targets seat0's graveyard Lightning
    # Bolt CardInstance (a "graveyard_card" target); Grab the Prize
    # (controller=1) targets the Lightning Bolt ENTRY itself (a
    # "stack_entry" target, Counterspell-shaped).
    bolt_entry = {"card_def": game.CARD_DEFS["Lightning Bolt"], "resolve": None, "controller": 0,
                  "targets": (("graveyard_card", seat0.graveyard[0]),)}
    fs_state.stack = [
        {"card_def": game.CARD_DEFS["Fiery Temper"], "resolve": None, "controller": 1,
         "targets": (("creature", seat0.battlefield[0]),)},
        bolt_entry,
        {"card_def": game.CARD_DEFS["Grab the Prize"], "resolve": None, "controller": 1,
         "targets": (("stack_entry", bolt_entry),)},
    ]
    # seat 0's Guttersnipe blocks seat 1's Kitchen Imp -- exercises the
    # blocked_as_attacker/committed_as_blocker dynamic features on both sides.
    attacker, blocker = seat1.battlefield[0], seat0.battlefield[0]
    seat1.attackers = [attacker]
    seat1.blocked_by = {attacker: [blocker]}

    for my_seat in (0, 1):
        tokens = build_token_set(fs_state, my_seat, vocab)
        # 2 (seat0 battlefield) + 1 (seat0 graveyard) + 1 (seat1 battlefield)
        # + 1 (seat1 exile) + 3 (stack) = 8 tokens, regardless of which
        # seat's own perspective this is built from.
        assert len(tokens) == 8, f"expected 8 tokens for my_seat={my_seat}, got {len(tokens)}"
        for vocab_idx, row, identity in tokens:
            assert isinstance(vocab_idx, int) and vocab_idx != 0  # every token here is a real, known card
            assert len(row) == TOKEN_FEATURE_DIM
        battlefield_tokens = [(idx, row, ident) for idx, row, ident in tokens if isinstance(ident, Permanent)]
        assert len(battlefield_tokens) == 3, "3 battlefield permanents (2 seat0 + 1 seat1) must carry a Permanent identity"
        graveyard_tokens = [(idx, row, ident) for idx, row, ident in tokens
                            if isinstance(ident, CardInstance) and not isinstance(ident, Permanent)]
        assert len(graveyard_tokens) == 1 and graveyard_tokens[0][2].name == "Lightning Bolt", \
            "the graveyard card token must carry its exact CardInstance as pointer handle"
        assert graveyard_tokens[0][2] is seat0.graveyard[0], "graveyard token identity must be the live CardInstance"
        inert_tokens = [(idx, row, ident) for idx, row, ident in tokens if ident is None]
        assert len(inert_tokens) == 1, "only exile has no pointer-addressable resolution -> identity None"
        stack_tokens = [(idx, row, ident) for idx, row, ident in tokens if isinstance(ident, dict)]
        assert len(stack_tokens) == 3, "every stack token must carry its exact stack-entry dict as pointer handle (choose_stack_target)"

        # Guttersnipe (seat 0's own permanent) -- side flag must match
        # whether seat 0 IS my_seat, not be hardcoded to "mine" regardless.
        # Matched by Permanent identity too, not just name: a second,
        # unrelated "Guttersnipe"-named token would never collide with this,
        # but being explicit keeps the lookup robust regardless of token order.
        guttersnipe_row = next(row for idx, row, ident in tokens
                                if idx == vocab.index("Guttersnipe") and isinstance(ident, Permanent))
        guttersnipe_identity = next(ident for idx, row, ident in tokens
                                     if idx == vocab.index("Guttersnipe") and isinstance(ident, Permanent))
        assert guttersnipe_identity is seat0.battlefield[0], "battlefield token identity must be the live Permanent object"
        expected_side_flag = 1.0 if my_seat == 0 else 0.0
        assert guttersnipe_row[-1] == expected_side_flag, (
            f"my_seat={my_seat}: Guttersnipe's side flag should be {expected_side_flag}, got {guttersnipe_row[-1]}"
        )
        # Dynamic-slot offsets, counted back from the row's END (same idiom
        # the side-flag check above already relies on): side(-1), zone
        # one-hot(-2..-5), targeted_by_theirs(-6), targeted_by_mine(-7),
        # committed_blocker(-8) -- see _token_row's own row-assembly order.
        targeted_theirs_slot = -(len(ZONES) + 1) - 1
        targeted_mine_slot = -(len(ZONES) + 1) - 2
        committed_blocker_slot = -(len(ZONES) + 1) - 3
        # committed_as_blocker must be set on Guttersnipe regardless of
        # my_seat: without reading the permanent's own true owner_idx (not
        # my_seat_idx), blocked_by lookups would silently use the wrong
        # seat's dict once my_seat_idx=1.
        assert guttersnipe_row[committed_blocker_slot] == 1.0, (
            f"my_seat={my_seat}: Guttersnipe should show committed_as_blocker=1.0, got "
            f"{guttersnipe_row[committed_blocker_slot]}"
        )

        # Guttersnipe is targeted by Fiery Temper, controller=1 -- so it
        # shows targeted_by_mine when my_seat==1, targeted_by_theirs when
        # my_seat==0, regardless of whose permanent it actually is (the bit
        # tracks the PERSPECTIVE seat vs. the targeting entry's controller,
        # never the target's own owner).
        assert guttersnipe_row[targeted_mine_slot] == (1.0 if my_seat == 1 else 0.0), (
            f"my_seat={my_seat}: Guttersnipe targeted_by_mine mismatch: {guttersnipe_row[targeted_mine_slot]}"
        )
        assert guttersnipe_row[targeted_theirs_slot] == (1.0 if my_seat == 0 else 0.0), (
            f"my_seat={my_seat}: Guttersnipe targeted_by_theirs mismatch: {guttersnipe_row[targeted_theirs_slot]}"
        )
        # An untargeted permanent (Mountain never appears in any stack
        # entry's targets) must show 0/0 on both bits.
        mountain_row = next(row for idx, row, ident in tokens if idx == vocab.index("Mountain"))
        assert mountain_row[targeted_mine_slot] == 0.0 and mountain_row[targeted_theirs_slot] == 0.0

        # The graveyard Lightning Bolt CardInstance is targeted by the
        # Lightning Bolt STACK ENTRY, controller=0 -- a "graveyard_card"
        # target, matched by the same id()-keyed lookup as a battlefield
        # "creature" target.
        gy_row = graveyard_tokens[0][1]
        assert gy_row[targeted_mine_slot] == (1.0 if my_seat == 0 else 0.0)
        assert gy_row[targeted_theirs_slot] == (1.0 if my_seat == 1 else 0.0)

        # The Lightning Bolt STACK TOKEN is itself targeted by the Grab the
        # Prize entry, controller=1 -- a "stack_entry" target (Counterspell-
        # shaped): a spell/ability on the stack can be the target, not just
        # a permanent or graveyard card, and the bit lands on that stack
        # token's own row. Disambiguated from the graveyard Lightning Bolt
        # CardInstance by identity TYPE (dict = stack entry, pointer-
        # addressable for choose_stack_target -- see rl.action_bridge).
        bolt_row = next(row for idx, row, ident in tokens if idx == vocab.index("Lightning Bolt") and isinstance(ident, dict))
        assert bolt_row[targeted_mine_slot] == (1.0 if my_seat == 1 else 0.0)
        assert bolt_row[targeted_theirs_slot] == (1.0 if my_seat == 0 else 0.0)

        # The Fiery Temper entry's controller is seat 1 -- side flag must
        # track that controller, not my_seat, for the "is_mine" semantics to
        # be correct from whichever seat's own perspective this is built for.
        fiery_row = next(row for idx, row, ident in tokens if idx == vocab.index("Fiery Temper"))
        assert fiery_row[-1] == (1.0 if my_seat == 1 else 0.0)


@pytest.mark.slow
def test_own_hand_tokenized_opponent_hidden():
    # Own hand is tokenized (not hidden information from myself); the
    # OPPONENT's hand stays hidden -- the hidden-information guarantee holds
    # for the ONE zone that changed (rl.features.build_token_set's own
    # docstring).
    _decklist_a, _decklist_b, vocab = _base_vocab()
    me = PlayerState(on_the_play=True)
    opp = PlayerState(on_the_play=False)
    me.hand = [game.CARD_DEFS["Lightning Bolt"], game.CARD_DEFS["Lightning Bolt"], game.CARD_DEFS["Mountain"]]
    opp.hand = [game.CARD_DEFS["Mountain"], game.CARD_DEFS["Mountain"]]
    state = GameState(on_the_play=True, players=[me, opp])

    # "hand" is ZONES' own last entry, so its one-hot slot sits at row[-2]
    # (the slot immediately before the side flag at row[-1]) -- same
    # counted-from-the-end idiom this file's own big tokenization test uses.
    assert ZONES[-1] == "hand"
    for seat, hand_size in ((0, 3), (1, 2)):
        tokens = build_token_set(state, seat, vocab)
        hand_rows = [row for _idx, row, _ident in tokens if row[-2] == 1.0]
        assert len(hand_rows) == hand_size, f"seat {seat} must see exactly its own {hand_size}-card hand, got {len(hand_rows)}"
        assert all(row[-1] == 1.0 for row in hand_rows), "every hand token must carry the 'mine' side flag"


@pytest.mark.slow
def test_hand_tokens_identity_none():
    # Own-hand tokens carry identity=None -- not pointer-addressable (see
    # build_token_set's own docstring on the shared/interned CardDef risk).
    _decklist_a, _decklist_b, vocab = _base_vocab()
    me = PlayerState(on_the_play=True)
    me.hand = [game.CARD_DEFS["Lightning Bolt"]]
    state = GameState(on_the_play=True, players=[me, PlayerState(on_the_play=False)])
    tokens = build_token_set(state, 0, vocab)
    matches = [(idx, row, ident) for idx, row, ident in tokens if row[-2] == 1.0]  # row[-2] = "hand" zone slot
    assert len(matches) == 1
    assert matches[0][2] is None, "an own-hand token must carry identity=None"


@pytest.mark.slow
def test_hand_tokens_cheap_and_full_pass_match():
    # The two build_token_set passes _build_decision runs (include_rows=False
    # then True) must return the same token count in the same order --
    # _padded_full_mask's pointer mask is positional (rl.agent's own
    # docstring). Hand tokens must be emitted on BOTH passes; only the row is
    # skipped on the cheap one.
    _decklist_a, _decklist_b, vocab = _base_vocab()
    me = PlayerState(on_the_play=True)
    me.hand = [game.CARD_DEFS["Lightning Bolt"], game.CARD_DEFS["Mountain"]]
    state = GameState(on_the_play=True, players=[me, PlayerState(on_the_play=False)])
    cheap = build_token_set(state, 0, vocab, include_rows=False)
    full = build_token_set(state, 0, vocab)
    assert len(cheap) == len(full)
    assert [ident for _i, _r, ident in cheap] == [ident for _i, _r, ident in full]
    assert all(row is None for _i, row, _ident in cheap)
    assert all(row is not None for _i, row, _ident in full)


@pytest.mark.slow
def test_choose_graveyard_card_over_own_hand_not_double_tokenized():
    # When the choose_graveyard_card pick is over MY OWN hand (not the
    # opponent's -- test_action_bridge.py's test_choose_graveyard_card_hand_
    # reveal already covers that case), the always-on own-hand loop must
    # yield to the reveal branch, which gives those cards a REAL,
    # pointer-addressable identity -- not tokenize them a second time with
    # identity=None.
    from rl.action_bridge import pointer_legal_mask

    me = PlayerState(on_the_play=True)
    me.hand = [game.CARD_DEFS["Lightning Bolt"], game.CARD_DEFS["Mountain"]]
    state = GameState(on_the_play=True, players=[me, PlayerState(on_the_play=False)])
    picked = []
    game.begin_choose_graveyard_card(
        state, predicate=lambda c: c.card_type != game.CardType.LAND,
        on_complete=lambda st, chosen: picked.append(chosen), graveyard=me.hand,
    )
    _decklist_a, _decklist_b, vocab = _base_vocab()
    tokens = build_token_set(state, 0, vocab)
    hand_tokens = [(idx, row, ident) for idx, row, ident in tokens if row[-2] == 1.0]  # row[-2] = "hand" zone slot
    assert len(hand_tokens) == 2, "my hand must be tokenized exactly once (via the reveal branch), not twice"
    assert {ident.name for _idx, _row, ident in hand_tokens} == {"Lightning Bolt", "Mountain"}, (
        "the reveal branch's own real CardDef identity must be what's present, not an inert None duplicate"
    )
    ids = [ident for _i, _r, ident in tokens]
    legal = [ref for ref, ok in zip(ids, pointer_legal_mask(state, ids)) if ok]
    assert legal == [me.hand[0]], "only the nonland hand card must be a legal pointer target"


@pytest.mark.slow
def test_cost_reduction_delta_feature():
    # cost_reduction_delta: nonzero for a hand card with a LIVE
    # registry cost_reduction spec (Tolarian Terror, cheaper per instant/
    # sorcery in the graveyard), zero for one with no such spec.
    from rl.agent import _hand_cost_reduction_deltas

    decklist = game.parse_decklist_file("../data/mono_blue_terror.txt")
    vocab = CardVocab([decklist])
    me = PlayerState(on_the_play=True)
    me.hand = [game.CARD_DEFS["Tolarian Terror"], game.CARD_DEFS["Mental Note"]]
    me.graveyard = [CardInstance(game.CARD_DEFS["Counterspell"]), CardInstance(game.CARD_DEFS["Brainstorm"])]
    state = GameState(on_the_play=True, players=[me, PlayerState(on_the_play=False)])

    hand_cost_reduction = _hand_cost_reduction_deltas(state, 0)
    tokens = build_token_set(state, 0, vocab, hand_cost_reduction=hand_cost_reduction)
    delta_idx = STATIC_FEATURE_DIM
    terror_row = next(row for idx, row, _ident in tokens if idx == vocab.index("Tolarian Terror"))
    note_row = next(row for idx, row, _ident in tokens if idx == vocab.index("Mental Note"))
    assert terror_row[delta_idx] == pytest.approx(2 / MANA_PIP_CAP), (
        "2 instants/sorceries in the graveyard must reduce Tolarian Terror's generic cost by 2"
    )
    assert note_row[delta_idx] == 0.0, "a card with no cost_reduction spec must show zero delta"

    # No reduction at all when nothing's in the graveyard.
    empty_state = GameState(on_the_play=True, players=[PlayerState(on_the_play=True), PlayerState(on_the_play=False)])
    empty_state.players[0].hand = [game.CARD_DEFS["Tolarian Terror"]]
    empty_deltas = _hand_cost_reduction_deltas(empty_state, 0)
    assert empty_deltas == {}


@pytest.mark.slow
def test_card_vocab_persistence_append_only():
    # CardVocab persistence: append-only across separate construction calls
    # -- existing names must NEVER change index once a 3rd deck introduces
    # new names, and vocab.size must reflect the FULL persisted vocabulary
    # even when a later call only passes a SUBSET of decklists.
    decklist_a, _decklist_b, _vocab = _base_vocab()
    tmp_dir = tempfile.mkdtemp()
    try:
        vocab_path = os.path.join(tmp_dir, "vocab.json")
        vocab1 = CardVocab([decklist_a], vocab_path=vocab_path)
        original_indices = dict(vocab1.name_to_index)

        decklist_c = game.parse_decklist_file("../data/rakdos_madness.txt")  # stand-in "3rd deck" with new names
        vocab2 = CardVocab([decklist_a, decklist_c], vocab_path=vocab_path)
        for name, idx in original_indices.items():
            assert vocab2.name_to_index[name] == idx, f"persisted vocab reassigned {name!r}'s index -- would corrupt every existing checkpoint's embedding table"
        assert vocab2.size >= vocab1.size, "vocab must only grow, never shrink, across persisted calls"

        # A later call touching only the ORIGINAL (smaller) decklist must
        # still report the FULL persisted vocab size, not shrink back down
        # -- the embedding table has to stay valid for checkpoints trained
        # against the larger roster too.
        vocab3 = CardVocab([decklist_a], vocab_path=vocab_path)
        assert vocab3.size == vocab2.size
        for name, idx in vocab2.name_to_index.items():
            assert vocab3.name_to_index[name] == idx
    finally:
        shutil.rmtree(tmp_dir)
