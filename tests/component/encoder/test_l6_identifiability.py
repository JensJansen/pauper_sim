"""Layer 6 -- identifiability. The encoding must separate positions that differ in a fact that matters for play, and
must NOT separate positions that are the same game position (slot numbers, list order, which of two identical cards
is which, hidden information, which seat you happen to be). All through the real tokenizer + a fixed encoder."""
import pytest

from support import scenario as sc

BASE = {
    "s0": sc.seat(bf=["Guttersnipe", "Mountain", "Mountain"], hand=["Lightning Bolt", "Counterspell"], gy=["Faithless Looting"]),
    "s1": sc.seat(bf=["Kitchen Imp", "Island"], hand=["Balustrade Spy"]),
}


@pytest.fixture(scope="module")
def world():
    vocab = sc.make_vocab()
    return vocab, sc.make_encoder(vocab)


def enc_of(world, args, seat=0):
    vocab, enc = world
    return sc.encode(sc.build(**args), seat, enc, vocab)


def variant(**changes):
    """BASE with seat/stack/combat overrides; seat dicts are merged key by key."""
    args = {k: (dict(v) if isinstance(v, dict) else v) for k, v in BASE.items()}
    for key, val in changes.items():
        if key in ("s0", "s1"):
            args[key] = {**args[key], **val}
        else:
            args[key] = val
    return args


# ---- facts that matter: the encoding must change --------------------------------------------------------------------

MATTERS = {
    "my creature taps": {"s0": {"bf": [("Guttersnipe", {"tapped": True}), "Mountain", "Mountain"]}},
    "my creature becomes summoning sick": {"s0": {"bf": [("Guttersnipe", {"sick": True}), "Mountain", "Mountain"]}},
    "my creature gets a +1/+1 counter": {"s0": {"bf": [("Guttersnipe", {"counters": {"+1/+1": 1}}), "Mountain", "Mountain"]}},
    "my creature has damage marked": {"s0": {"bf": [("Guttersnipe", {"damage": 1}), "Mountain", "Mountain"]}},
    "their creature taps": {"s1": {"bf": [("Kitchen Imp", {"tapped": True}), "Island"]}},
    "their creature gets a counter": {"s1": {"bf": [("Kitchen Imp", {"counters": {"+1/+1": 2}}), "Island"]}},
    "my creature attacks": {"attackers": {0: [0]}},
    "their creature attacks": {"attackers": {1: [0]}},
    "their attacker is blocked": {"attackers": {1: [0]}, "blocked": {1: {0: [(0, 0)]}}},
    "I have one more land": {"s0": {"bf": ["Guttersnipe", "Mountain", "Mountain", "Mountain"]}},
    "they have one more land": {"s1": {"bf": ["Kitchen Imp", "Island", "Island"]}},
    "a different creature of mine": {"s0": {"bf": ["Balustrade Spy", "Mountain", "Mountain"]}},
    "a card in my graveyard": {"s0": {"gy": ["Faithless Looting", "Lightning Bolt"]}},
    "a card in their graveyard": {"s1": {"gy": ["Counterspell"]}},
    "a card in their exile": {"s1": {"ex": ["Lightning Bolt"]}},
    "a card in my exile": {"s0": {"ex": ["Lightning Bolt"]}},
    "a different card in my hand": {"s0": {"hand": ["Lightning Bolt", "Faithless Looting"]}},
    "one fewer card in my hand": {"s0": {"hand": ["Lightning Bolt"]}},
    "a spell of theirs on the stack": {"stack": [("Counterspell", 1, None)]},
    "a spell of mine on the stack": {"stack": [("Counterspell", 0, None)]},
    "their spell targets my creature": {"stack": [("Lightning Bolt", 1, ("bf", 0, 0))]},
    "my spell targets their creature": {"stack": [("Lightning Bolt", 0, ("bf", 1, 0))]},
    "their spell targets a card in my graveyard": {"stack": [("Counterspell", 1, ("gy", 0, 0))]},
    "an aura on my creature": {"s0": {"bf": ["Guttersnipe", "Mountain", "Mountain", ("Rancor", {"aura_on": (0, 0)})]}},
}


@pytest.mark.parametrize("what", sorted(MATTERS))
def test_changing_a_relevant_fact_changes_the_encoding(world, what):
    assert sc.differs(enc_of(world, BASE), enc_of(world, variant(**MATTERS[what]))), f"'{what}' is invisible to the encoder"


def test_which_creature_an_aura_is_on_is_visible(world):
    base = {"s0": sc.seat(bf=["Guttersnipe", "Kitchen Imp", ("Rancor", {"aura_on": (0, 0)})]), "s1": sc.seat()}
    other = {"s0": sc.seat(bf=["Guttersnipe", "Kitchen Imp", ("Rancor", {"aura_on": (0, 1)})]), "s1": sc.seat()}
    assert sc.differs(enc_of(world, base), enc_of(world, other))


def test_different_auras_on_the_same_creature_are_distinguishable(world):
    a = {"s0": sc.seat(bf=["Guttersnipe", ("Rancor", {"aura_on": (0, 0)})]), "s1": sc.seat()}
    b = {"s0": sc.seat(bf=["Guttersnipe", ("Ancestral Mask", {"aura_on": (0, 0)})]), "s1": sc.seat()}
    assert sc.differs(enc_of(world, a), enc_of(world, b))


# ---- facts that do NOT matter: the encoding must be identical ---------------------------------------------------------

SAME = {
    "slot numbers relabelled": {"s0": {"bf": [("Guttersnipe", {"slot": 7}), ("Mountain", {"slot": 3}), ("Mountain", {"slot": 9})]}},
    "battlefield order reversed": {"s0": {"bf": ["Mountain", "Mountain", "Guttersnipe"]}},
    "their battlefield order reversed": {"s1": {"bf": ["Island", "Kitchen Imp"]}},
    "my hand order reversed": {"s0": {"hand": ["Counterspell", "Lightning Bolt"]}},
    "their hidden hand is different": {"s1": {"hand": ["Lightning Bolt"]}},
    "their hidden hand is larger": {"s1": {"hand": ["Balustrade Spy", "Island", "Island"]}},
}


@pytest.mark.parametrize("what", sorted(SAME))
def test_changing_an_irrelevant_fact_does_not_change_the_encoding(world, what):
    assert sc.same(enc_of(world, BASE), enc_of(world, variant(**SAME[what]))), f"'{what}' changed the encoding"


def test_two_identical_untapped_lands_are_interchangeable(world):
    a = variant(s0={"bf": ["Guttersnipe", ("Mountain", {"tapped": True}), "Mountain"]})
    b = variant(s0={"bf": ["Guttersnipe", "Mountain", ("Mountain", {"tapped": True})]})
    assert sc.same(enc_of(world, a), enc_of(world, b))


# ---- seat symmetry: the position seen from the other chair --------------------------------------------------------------

BUSY = sc.BUSY  # a busy asymmetric position, shared with the golden pin


def test_a_position_looks_the_same_from_either_seat_once_the_seats_are_swapped(world):
    vocab, enc = world
    a = sc.encode(sc.build(**BUSY), 0, enc, vocab)
    b = sc.encode(sc.build(**sc.mirror(BUSY)), 1, enc, vocab)
    assert sc.same(a, b, tol=1e-5), "the mirror-image position encodes differently"


def test_the_two_seats_see_different_things_in_an_asymmetric_position(world):
    vocab, enc = world
    state = sc.build(**BUSY)
    assert sc.differs(sc.encode(state, 0, enc, vocab), sc.encode(state, 1, enc, vocab))


# ---- COUNTING (was a defect found by this suite; fixed) -------------------------------------------------------------------
# The encoder used to be unable to COUNT identical cards: n copies encoded exactly like one (to ~1e-7), because
# self-attention and the attention pooling are weighted averages and duplicate tokens have identical rows. Fixed three
# ways: per-token copy-count columns, a sum-pooled half in every summary, and exact zone/type counts in the scalar
# block. test_l11_counting.py holds the exhaustive checks; the cases below stay as the original reproduction.
COPIES = {
    "lands I control": lambda n: {"s0": sc.seat(bf=["Mountain"] * n), "s1": sc.seat()},
    "copies of a spell in my hand": lambda n: {"s0": sc.seat(hand=["Lightning Bolt"] * n), "s1": sc.seat()},
    "cards in their graveyard": lambda n: {"s0": sc.seat(), "s1": sc.seat(gy=["Counterspell"] * n)},
    "their lands": lambda n: {"s0": sc.seat(), "s1": sc.seat(bf=["Island"] * n)},
}


@pytest.mark.parametrize("what", sorted(COPIES))
def test_n_identical_cards_are_distinguishable_from_one(world, what):
    one, many = enc_of(world, COPIES[what](1)), enc_of(world, COPIES[what](5))
    assert sc.differs(one, many), f"5 identical {what} encode exactly like 1"


@pytest.mark.parametrize("what", sorted(COPIES))
def test_but_a_change_in_the_mix_of_cards_is_visible(world, what):
    """What DOES leak: adding a different card next to the copies moves the average."""
    base = COPIES[what](3)
    other = {k: dict(v, **{("bf" if "lands" in what else "gy" if "graveyard" in what else "hand"):
                           v[("bf" if "lands" in what else "gy" if "graveyard" in what else "hand")] + ["Faithless Looting"]})
             for k, v in base.items()}
    assert sc.differs(enc_of(world, base), enc_of(world, other))
