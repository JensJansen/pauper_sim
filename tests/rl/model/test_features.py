"""Tests for rl.model.features: CardVocab indexing/persistence and build_token_set tokenization."""
import numpy as np
import os
import shutil
import tempfile

import pytest

import game
from game.state import CardInstance, GameState, Permanent, PlayerState
from rl.model.features import (
    CARD_TYPES,
    COUNTER_VOCAB,
    KEYWORD_VOCAB,
    SLOTS_AFTER_TARGETING,
    SUBTYPE_VOCAB,
    MANA_PIP_CAP,
    POWER_TOUGHNESS_CAP,
    DYNAMIC_FEATURE_DIM,
    STATIC_FEATURE_DIM,
    TOKEN_FEATURE_DIM,
    ZONES,
    CardVocab,
    build_token_set,
    cached_static_card_features,
    static_card_features,
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
    # Every real card/token gets a unique index, never colliding with another entry or the sentinel.
    indices = [vocab.index(n) for n in all_names] + [vocab.index("Blood")]
    assert len(set(indices)) == len(indices), "card/token indices must be unique"
    assert 0 not in indices, "no real card/token may be assigned the padding sentinel's index"


@pytest.mark.slow
def test_static_card_features_spell_and_creature():
    _decklist_a, _decklist_b, vocab = _base_vocab()
    feat = cached_static_card_features("Lightning Bolt", vocab)
    assert len(feat) == STATIC_FEATURE_DIM
    # Lightning Bolt costs {R: 1}: one nonzero mana-cost dim, not a creature (P/T both 0).
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
    # Mountain (cast_cost=None) must not crash on cost.get(...) -- the `or {}` guard.
    feat_land = cached_static_card_features("Mountain", vocab)
    assert len(feat_land) == STATIC_FEATURE_DIM
    assert all(v == 0.0 for v in feat_land[:len(game.POOL_COLORS) + 1])
    type_start = len(game.POOL_COLORS) + 1
    land_type_idx = type_start + CARD_TYPES.index(game.CardType.LAND)
    assert feat_land[land_type_idx] == 1.0


@pytest.mark.slow
def test_static_card_features_token_resolves_via_vocab():
    _decklist_a, _decklist_b, vocab = _base_vocab()
    # Blood (not in game.CARD_DEFS) must resolve via the vocab's merged card_def_by_name.
    feat_blood = cached_static_card_features("Blood", vocab)
    assert len(feat_blood) == STATIC_FEATURE_DIM
    type_start = len(game.POOL_COLORS) + 1
    artifact_type_idx = type_start + CARD_TYPES.index(game.CardType.ARTIFACT)
    assert feat_blood[artifact_type_idx] == 1.0


@pytest.mark.slow
def test_card_vocab_shared_card_same_index_across_decks():
    decklist_a, decklist_b, vocab = _base_vocab()
    # A card shared between both decks gets the same index regardless of which deck's list it's looked up through.
    assert vocab.index("Lightning Bolt") == vocab.name_to_index["Lightning Bolt"]
    shared_names = ({n for n, *_r in decklist_a} & {n for n, *_r in decklist_b})
    assert "Lightning Bolt" in shared_names and len(shared_names) == 8


@pytest.mark.slow
def test_card_vocab_dfc_back_face_alias():
    _decklist_a, _decklist_b, vocab = _base_vocab()
    # DFC back face (Delver of Secrets -> Insectile Aberration): the back name
    # must resolve to a real CardDef and alias to its front face's vocab
    # index without changing vocab.size.
    assert vocab.card_def("Insectile Aberration").name == "Insectile Aberration"  # registered, resolves
    dfc_vocab = CardVocab([game.parse_decklist_file("../data/mono_blue_terror.txt")])
    assert dfc_vocab.index("Delver of Secrets") != 0  # real decklist card
    assert dfc_vocab.index("Insectile Aberration") == dfc_vocab.index("Delver of Secrets")  # aliased to the front
    _size_no_dfc = max(v for k, v in dfc_vocab.name_to_index.items() if k != "Insectile Aberration") + 1
    assert dfc_vocab.size == _size_no_dfc, "the back-face alias must not grow vocab.size"


@pytest.mark.slow
def test_build_token_set_both_seats_all_zones_and_targeting():
    # Covers every public zone, and runs from both seats' perspective (a bug
    # using state.players[0] unconditionally instead of a permanent's true
    # owner_idx would pass a seat-0-only test).
    _decklist_a, _decklist_b, vocab = _base_vocab()
    seat0 = PlayerState(on_the_play=True)
    seat0.battlefield = [Permanent(game.CARD_DEFS["Guttersnipe"]), Permanent(game.CARD_DEFS["Mountain"], tapped=True)]
    seat0.graveyard = [CardInstance(game.CARD_DEFS["Lightning Bolt"])]  # graveyard holds CardInstances now
    seat1 = PlayerState(on_the_play=False)
    seat1.battlefield = [Permanent(game.CARD_DEFS["Kitchen Imp"])]
    seat1.exile = [(game.CARD_DEFS["Faithless Looting"], 3)]
    fs_state = GameState(on_the_play=True, players=[seat0, seat1])
    # Three stack entries exercising the three object target kinds (the
    # fourth, "player", has no token). Pairings are mechanically synthetic,
    # not real card text -- only exercises the plumbing: Fiery Temper
    # (controller=1) targets seat0's Guttersnipe (permanent target);
    # Lightning Bolt (controller=0) targets seat0's graveyard Lightning Bolt
    # CardInstance (graveyard_card target); Grab the Prize (controller=1)
    # targets the Lightning Bolt entry itself (stack_entry target).
    bolt_entry = {"card_def": game.CARD_DEFS["Lightning Bolt"], "resolve": None, "controller": 0,
                  "targets": (("graveyard_card", seat0.graveyard[0]),)}
    fs_state.stack = [
        {"card_def": game.CARD_DEFS["Fiery Temper"], "resolve": None, "controller": 1,
         "targets": (("creature", seat0.battlefield[0]),)},
        bolt_entry,
        {"card_def": game.CARD_DEFS["Grab the Prize"], "resolve": None, "controller": 1,
         "targets": (("stack_entry", bolt_entry),)},
    ]
    # seat 0's Guttersnipe blocks seat 1's Kitchen Imp: exercises blocked_as_attacker/committed_as_blocker.
    attacker, blocker = seat1.battlefield[0], seat0.battlefield[0]
    seat1.attackers = [attacker]
    seat1.blocked_by = {attacker: [blocker]}

    for my_seat in (0, 1):
        tokens = build_token_set(fs_state, my_seat, vocab)
        # 2 (seat0 battlefield) + 1 (seat0 graveyard) + 1 (seat1 battlefield) + 1 (seat1 exile) + 3 (stack) = 8
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

        # Guttersnipe (seat 0's permanent): side flag must match whether
        # seat 0 IS my_seat, matched by Permanent identity, not just name.
        guttersnipe_row = next(row for idx, row, ident in tokens
                                if idx == vocab.index("Guttersnipe") and isinstance(ident, Permanent))
        guttersnipe_identity = next(ident for idx, row, ident in tokens
                                     if idx == vocab.index("Guttersnipe") and isinstance(ident, Permanent))
        assert guttersnipe_identity is seat0.battlefield[0], "battlefield token identity must be the live Permanent object"
        expected_side_flag = 1.0 if my_seat == 0 else 0.0
        assert guttersnipe_row[-1] == expected_side_flag, (
            f"my_seat={my_seat}: Guttersnipe's side flag should be {expected_side_flag}, got {guttersnipe_row[-1]}"
        )
        # Dynamic-slot offsets, counted back from the row's end via SLOTS_AFTER_TARGETING.
        targeted_theirs_slot = -SLOTS_AFTER_TARGETING - 1
        targeted_mine_slot = -SLOTS_AFTER_TARGETING - 2
        # Directly before targeted_by_mine sit the four per-permanent bits
        # (is_attached, auras_attached, summoning_sick, is_attacking).
        committed_blocker_slot = targeted_mine_slot - 5
        # Must be set on Guttersnipe regardless of my_seat: needs the
        # permanent's true owner_idx, not my_seat_idx, for blocked_by lookups.
        assert guttersnipe_row[committed_blocker_slot] == 1.0, (
            f"my_seat={my_seat}: Guttersnipe should show committed_as_blocker=1.0, got "
            f"{guttersnipe_row[committed_blocker_slot]}"
        )

        # Guttersnipe is targeted by Fiery Temper (controller=1): shows
        # targeted_by_mine when my_seat==1, targeted_by_theirs when
        # my_seat==0 -- the bit tracks perspective seat vs. controller, not the target's owner.
        assert guttersnipe_row[targeted_mine_slot] == (1.0 if my_seat == 1 else 0.0), (
            f"my_seat={my_seat}: Guttersnipe targeted_by_mine mismatch: {guttersnipe_row[targeted_mine_slot]}"
        )
        assert guttersnipe_row[targeted_theirs_slot] == (1.0 if my_seat == 0 else 0.0), (
            f"my_seat={my_seat}: Guttersnipe targeted_by_theirs mismatch: {guttersnipe_row[targeted_theirs_slot]}"
        )
        # An untargeted permanent (Mountain) must show 0/0 on both bits.
        mountain_row = next(row for idx, row, ident in tokens if idx == vocab.index("Mountain"))
        assert mountain_row[targeted_mine_slot] == 0.0 and mountain_row[targeted_theirs_slot] == 0.0

        # The graveyard Lightning Bolt CardInstance is targeted by the
        # Lightning Bolt stack entry (controller=0, graveyard_card target).
        gy_row = graveyard_tokens[0][1]
        assert gy_row[targeted_mine_slot] == (1.0 if my_seat == 0 else 0.0)
        assert gy_row[targeted_theirs_slot] == (1.0 if my_seat == 1 else 0.0)

        # The Lightning Bolt stack token is itself targeted by Grab the
        # Prize (controller=1, stack_entry target). Disambiguated from the
        # graveyard CardInstance by identity type (dict = stack entry).
        bolt_row = next(row for idx, row, ident in tokens if idx == vocab.index("Lightning Bolt") and isinstance(ident, dict))
        assert bolt_row[targeted_mine_slot] == (1.0 if my_seat == 1 else 0.0)
        assert bolt_row[targeted_theirs_slot] == (1.0 if my_seat == 0 else 0.0)

        # Fiery Temper's controller is seat 1: side flag tracks the controller, not my_seat.
        fiery_row = next(row for idx, row, ident in tokens if idx == vocab.index("Fiery Temper"))
        assert fiery_row[-1] == (1.0 if my_seat == 1 else 0.0)


@pytest.mark.slow
def test_own_hand_tokenized_opponent_hidden():
    # Own hand is tokenized; the opponent's hand stays hidden.
    _decklist_a, _decklist_b, vocab = _base_vocab()
    me = PlayerState(on_the_play=True)
    opp = PlayerState(on_the_play=False)
    me.hand = [game.CARD_DEFS["Lightning Bolt"], game.CARD_DEFS["Lightning Bolt"], game.CARD_DEFS["Mountain"]]
    opp.hand = [game.CARD_DEFS["Mountain"], game.CARD_DEFS["Mountain"]]
    state = GameState(on_the_play=True, players=[me, opp])

    # The zone one-hot block sits immediately before the side flag at
    # row[-1]; derived from ZONES rather than a hardcoded offset.
    hand_slot = -(len(ZONES) - ZONES.index("hand")) - 1
    for seat, hand_size in ((0, 3), (1, 2)):
        tokens = build_token_set(state, seat, vocab)
        hand_rows = [row for _idx, row, _ident in tokens if row[hand_slot] == 1.0]
        assert len(hand_rows) == hand_size, f"seat {seat} must see exactly its own {hand_size}-card hand, got {len(hand_rows)}"
        assert all(row[-1] == 1.0 for row in hand_rows), "every hand token must carry the 'mine' side flag"


@pytest.mark.slow
def test_hand_tokens_identity_none():
    # Own-hand tokens carry identity=None: not pointer-addressable.
    _decklist_a, _decklist_b, vocab = _base_vocab()
    me = PlayerState(on_the_play=True)
    me.hand = [game.CARD_DEFS["Lightning Bolt"]]
    state = GameState(on_the_play=True, players=[me, PlayerState(on_the_play=False)])
    tokens = build_token_set(state, 0, vocab)
    hand_slot = -(len(ZONES) - ZONES.index("hand")) - 1
    matches = [(idx, row, ident) for idx, row, ident in tokens if row[hand_slot] == 1.0]
    assert len(matches) == 1
    assert matches[0][2] is None, "an own-hand token must carry identity=None"


@pytest.mark.slow
def test_hand_tokens_cheap_and_full_pass_match():
    # The two build_token_set passes _build_decision runs (include_rows=False
    # then True) must return the same token count in the same order (the
    # pointer mask is positional). Hand tokens must be emitted on both
    # passes; only the row is skipped on the cheap one.
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
    # When the choose_graveyard_card pick is over my own hand, the always-on
    # own-hand loop must yield to the reveal branch (real pointer-addressable
    # identity), not tokenize the cards a second time with identity=None.
    from rl.decision.action_bridge import pointer_legal_mask

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
    hand_slot = -(len(ZONES) - ZONES.index("hand")) - 1
    hand_tokens = [(idx, row, ident) for idx, row, ident in tokens if row[hand_slot] == 1.0]
    assert len(hand_tokens) == 2, "my hand must be tokenized exactly once (via the reveal branch), not twice"
    assert {ident.name for _idx, _row, ident in hand_tokens} == {"Lightning Bolt", "Mountain"}, (
        "the reveal branch's own real CardDef identity must be what's present, not an inert None duplicate"
    )
    ids = [ident for _i, _r, ident in tokens]
    legal = [ref for ref, ok in zip(ids, pointer_legal_mask(state, ids)) if ok]
    assert legal == [me.hand[0]], "only the nonland hand card must be a legal pointer target"


@pytest.mark.slow
def test_cost_reduction_delta_feature():
    # cost_reduction_delta: nonzero for a hand card with a live registry
    # cost_reduction spec (Tolarian Terror), zero for one with no such spec.
    from rl.decision.agent import _hand_cost_reduction_deltas

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

    # No reduction when nothing's in the graveyard.
    empty_state = GameState(on_the_play=True, players=[PlayerState(on_the_play=True), PlayerState(on_the_play=False)])
    empty_state.players[0].hand = [game.CARD_DEFS["Tolarian Terror"]]
    empty_deltas = _hand_cost_reduction_deltas(empty_state, 0)
    assert empty_deltas == {}


@pytest.mark.slow
def test_card_vocab_persistence_append_only():
    # CardVocab persistence is append-only: existing names must never change
    # index once a new deck introduces new names, and vocab.size must
    # reflect the full persisted vocabulary even for a later call that only
    # passes a subset of decklists.
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

        # A call touching only the smaller decklist must still report the
        # full persisted vocab size.
        vocab3 = CardVocab([decklist_a], vocab_path=vocab_path)
        assert vocab3.size == vocab2.size
        for name, idx in vocab2.name_to_index.items():
            assert vocab3.name_to_index[name] == idx
    finally:
        shutil.rmtree(tmp_dir)


@pytest.mark.slow
def test_shuffle_library_is_the_only_place_a_library_gets_shuffled():
    """game.effects.shared.shuffle_library is the single choke point for
    shuffling a library. Structural, not behavioural: no module may call
    rng.shuffle on a library outside shuffle_library."""
    import inspect
    import game.effects.shared as shared_mod
    import game.resolution.handlers_library as hl
    import game.resolution.handlers_mulligan as hm
    import game.effects.undercity as uc
    import game.catalog.red_cards as red
    import game.catalog.green_cards as grn
    import game.catalog.colorless_cards as col

    for mod in (shared_mod, hl, hm, uc, red, grn, col):
        src = inspect.getsource(mod)
        offenders = [ln.strip() for ln in src.splitlines()
                     if "rng.shuffle(" in ln and "library" in ln and "def shuffle_library" not in ln]
        allowed = 1 if mod is shared_mod else 0  # shared.py contains the one real shuffle, inside shuffle_library
        assert len(offenders) == allowed, f"{mod.__name__} shuffles a library outside the choke point: {offenders}"


@pytest.mark.slow
def test_an_unblocked_attacker_is_distinguishable_from_a_creature_at_home():
    """combat's damage step reads `unblocked = [p for p in attackers if p not
    in blocked_by]`, so an unblocked attacker never appears in blocked_by at
    all, and blocked_as_attacker alone can't tell it apart from a creature
    sitting at home. Also pins summoning sickness, which gates both
    attacking (CR 302.6) and tapping for mana (Llanowar Elves)."""
    _decklist_a, _decklist_b, vocab = _base_vocab()
    seat0 = PlayerState(on_the_play=True)
    attacking, at_home = Permanent(game.CARD_DEFS["Kitchen Imp"]), Permanent(game.CARD_DEFS["Kitchen Imp"])
    attacking.summoning_sick = at_home.summoning_sick = False
    seat0.battlefield = [attacking, at_home]
    seat1 = PlayerState(on_the_play=False)
    sick = Permanent(game.CARD_DEFS["Guttersnipe"])  # entered this turn -> can't attack, can't tap for mana
    seat1.battlefield = [sick]
    state = GameState(on_the_play=True, players=[seat0, seat1])
    seat0.attackers = [attacking]  # declared, and NOT blocked -> absent from blocked_by

    rows = {ident: row for _idx, row, ident in build_token_set(state, 0, vocab)}
    # np.array_equal, not != : feature rows are float32 arrays, and `!=` on an
    # array is elementwise, so bare truthiness raises.
    assert not np.array_equal(rows[attacking], rows[at_home]), (
        "an unblocked attacker must not be feature-identical to an identical creature that stayed home"
    )
    # Same two creatures from the defender's perspective: the bit tracks the
    # permanent's controller, not whoever is looking at it.
    rows_defender = {ident: row for _idx, row, ident in build_token_set(state, 1, vocab)}
    assert not np.array_equal(rows_defender[attacking], rows_defender[at_home]), (
        "the defender is the seat that most needs to see which creatures are attacking"
    )
    fresh_rows = build_token_set(state, 1, vocab)
    sick_row = next(row for _i, row, ident in fresh_rows if ident is sick)
    seat1.battlefield[0].summoning_sick = False
    ready_row = next(row for _i, row, ident in build_token_set(state, 1, vocab) if ident is sick)
    assert not np.array_equal(sick_row, ready_row), (
        "summoning sickness must be visible: it gates both attacking and tapping for mana"
    )


@pytest.mark.slow
def test_derived_card_features_separate_cards_that_used_to_collide():
    """The registry/extra-derived feature blocks separate cards that printed
    stats alone leave identical. Pinned as properties rather than a headline
    collision count, which every added card would churn: the pairs below
    each collide for a different reason, together covering every derived
    block."""
    features = {name: tuple(static_card_features(cd)) for name, cd in game.CARD_DEFS.items()}

    # mana colors -- basic lands would otherwise be indistinguishable
    assert features["Forest"] != features["Mountain"]
    # enters_tapped + an ETB trigger, vs a plain untapped source
    assert features["Bojuka Bog"] != features["Great Furnace"]
    # extra["artifact"] -- load-bearing, since affinity counts artifact lands
    assert features["Island"] != features["Seat of the Synod"]
    assert features["Mountain"] != features["Great Furnace"]
    # pending_kinds -- same cost, same type, entirely different decisions
    assert features["Brainstorm"] != features["Dispel"]
    # count_all/count mana carry their color at index 1, like "fixed" does.
    # Reading those kinds as colorless is the bug this asserts against:
    # Priest of Titania and Overgrown Battlement both tap for {G}.
    mana_start = len(game.POOL_COLORS) + 1 + len(CARD_TYPES) + 2 + len(KEYWORD_VOCAB)
    priest = static_card_features(game.CARD_DEFS["Priest of Titania"])
    assert priest[mana_start] == 1.0, "Priest of Titania produces mana"
    assert priest[mana_start + 1 + game.POOL_COLORS.index("G")] == 1.0, "and it produces GREEN, not colorless"
    assert priest[mana_start + 1 + game.POOL_COLORS.index("C")] == 0.0
    assert priest[mana_start + 1 + len(game.POOL_COLORS)] == 1.0, "count_all is board-scaled"
    assert static_card_features(game.CARD_DEFS["Forest"])[mana_start + 1 + len(game.POOL_COLORS)] == 0.0,         "a plain fixed source is not board-scaled"
    # subtypes -- the elves engine keys off creature type
    elf, non_elf = game.CARD_DEFS["Llanowar Elves"], game.CARD_DEFS["Kitchen Imp"]
    elf_slots = [i for i, t in enumerate(SUBTYPE_VOCAB) if t == "Elf"]
    assert elf_slots, "SUBTYPE_VOCAB must carry Elf"
    offset = STATIC_FEATURE_DIM - len(SUBTYPE_VOCAB)
    assert static_card_features(elf)[offset + elf_slots[0]] == 1.0
    assert static_card_features(non_elf)[offset + elf_slots[0]] == 0.0

    distinct = len(set(features.values()))
    assert distinct >= 0.85 * len(features), (
        f"only {distinct}/{len(features)} cards have a distinct structured vector -- the derived blocks "
        f"regressed (this was 78/141 before they existed, and 131/141 after)"
    )


@pytest.mark.slow
def test_a_scrying_agent_can_see_the_cards_it_is_sorting():
    """begin_scry_surveil parks the revealed cards in pending["remaining"],
    off any real zone, while scry_surveil_options offers a bare
    ["keep", "dispose"] with no card name -- the "revealed" zone tokenizes
    them so the agent can see what it's sorting."""
    decklist = game.parse_decklist_file("../data/elves.txt")
    vocab = CardVocab([decklist])
    seat0, seat1 = PlayerState(on_the_play=True), PlayerState(on_the_play=False)
    state = GameState(on_the_play=True, players=[seat0, seat1])
    seat0.library = [game.CARD_DEFS["Llanowar Elves"], game.CARD_DEFS["Forest"],
                     game.CARD_DEFS["Priest of Titania"]]
    game.begin_scry_surveil(state, "scry", 2, lambda s: None)

    zone_start = STATIC_FEATURE_DIM + DYNAMIC_FEATURE_DIM - len(ZONES) - 1
    focus_slot = -SLOTS_AFTER_TARGETING
    by_name = {}
    for idx, row, _ident in build_token_set(state, 0, vocab):
        zone = ZONES[[row[zone_start + i] for i in range(len(ZONES))].index(1.0)]
        by_name[next(n for n, i in vocab.name_to_index.items() if i == idx)] = (zone, row[focus_slot])

    assert set(by_name) == {"Llanowar Elves", "Forest"}, (
        f"both revealed cards must be observable while sorting them, got {sorted(by_name)}"
    )
    assert all(zone == "revealed" for zone, _f in by_name.values())
    # remaining[0] is the card THIS keep/dispose applies to; the other is
    # seen but not yet reached.
    assert by_name["Llanowar Elves"][1] == 1.0, "the card under decision must carry decision_focus"
    assert by_name["Forest"][1] == 0.0, "a not-yet-reached card must not"

    # The opponent of a scrying player learns nothing (gated on active_idx).
    assert build_token_set(state, 1, vocab) == [], "a scry must reveal nothing to the opponent"


@pytest.mark.slow
def test_a_tucking_agent_can_see_the_card_it_is_tucking():
    """begin_tuck_to_library (Deem Inferior) holds the bounced card in
    pending["tuck_card"], off the battlefield and not yet in the library;
    its action rows ("Tuck: 2nd from top" / "Tuck: bottom") name a position,
    not a card, so the "revealed" zone tokenizes the card itself."""
    decklist = game.parse_decklist_file("../data/elves.txt")
    vocab = CardVocab([decklist])
    seat0, seat1 = PlayerState(on_the_play=True), PlayerState(on_the_play=False)
    state = GameState(on_the_play=True, players=[seat0, seat1])
    game.begin_tuck_to_library(state, game.CARD_DEFS["Priest of Titania"], owner_idx=0)

    tokens = build_token_set(state, 0, vocab)
    assert len(tokens) == 1, f"the tucked card must be observable, got {len(tokens)} tokens"
    assert tokens[0][0] == vocab.index("Priest of Titania")
    assert tokens[0][1][-SLOTS_AFTER_TARGETING] == 1.0, "the single tuck card is always the decision focus"
