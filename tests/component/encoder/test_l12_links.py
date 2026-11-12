"""Layer 12 -- links. A token set has no edges, so who-targets-what, who-blocks-whom and what-enchants-what used to be
invisible: two spells on the stack with their targets swapped, or two blockers with their assignments swapped, encoded
identically. Each relationship now gets a slot shared by its two endpoint tokens (see rl.model.features, LINKS)."""
import numpy as np
import pytest

from support import scenario as sc
from rl.model import features as F

SRC = F.LINK_START
DST = F.LINK_START + F.LINK_SLOTS
PLAYER = F.LINK_START + 2 * F.LINK_SLOTS


@pytest.fixture(scope="module")
def world():
    vocab = sc.make_vocab()
    return vocab, sc.make_encoder(vocab)


def rows_of(args, seat, vocab):
    return F.build_token_set(sc.build(**args), seat, vocab)


def slots(row, which):
    base = SRC if which == "src" else DST
    return {i for i in range(F.LINK_SLOTS) if row[base + i] == 1.0}


def row_named(tokens, vocab, name, mine=None):
    return next(r for i, r, _x in tokens if i == vocab.index(name) and (mine is None or r[-1] == float(mine)))


# ---- which spell targets which creature -------------------------------------------------------------------------------------

SEATS = {"s0": sc.seat(bf=["Guttersnipe", "Kitchen Imp"]), "s1": sc.seat()}


def two_spells(first_target, second_target):
    return {**SEATS, "stack": [("Lightning Bolt", 1, ("bf", 0, first_target)), ("Galvanic Blast", 1, ("bf", 0, second_target))]}


def test_which_spell_targets_which_creature_is_visible(world):
    vocab, enc = world
    a = sc.encode(sc.build(**two_spells(0, 1)), 0, enc, vocab)  # Bolt -> Guttersnipe, Blast -> Imp
    b = sc.encode(sc.build(**two_spells(1, 0)), 0, enc, vocab)  # Bolt -> Imp, Blast -> Guttersnipe
    assert sc.differs(a, b), "swapping which spell targets which creature changed nothing the encoder can see"


def test_a_spell_and_its_target_share_a_slot_and_nobody_else_does(world):
    vocab, _enc = world
    tokens = rows_of(two_spells(0, 1), 0, vocab)
    bolt, blast = row_named(tokens, vocab, "Lightning Bolt"), row_named(tokens, vocab, "Galvanic Blast")
    gutter, imp = row_named(tokens, vocab, "Guttersnipe"), row_named(tokens, vocab, "Kitchen Imp")
    assert len(slots(bolt, "src")) == 1 and len(slots(blast, "src")) == 1
    assert slots(bolt, "src") == slots(gutter, "dst") and slots(blast, "src") == slots(imp, "dst")
    assert slots(bolt, "src") != slots(blast, "src")
    assert not slots(gutter, "src") and not slots(bolt, "dst"), "slots carry direction: source and destination are separate"


def test_a_spell_aimed_at_a_player_says_whose(world):
    vocab, enc = world
    at_me = {**SEATS, "stack": [("Lightning Bolt", 1, ("player", 0))]}
    at_them = {**SEATS, "stack": [("Lightning Bolt", 1, ("player", 1))]}
    assert sc.differs(sc.encode(sc.build(**at_me), 0, enc, vocab), sc.encode(sc.build(**at_them), 0, enc, vocab))
    row = row_named(rows_of(at_me, 0, vocab), vocab, "Lightning Bolt")
    assert row[PLAYER] == 1.0 and row[PLAYER + 1] == 0.0
    row = row_named(rows_of(at_me, 1, vocab), vocab, "Lightning Bolt")
    assert row[PLAYER] == 0.0 and row[PLAYER + 1] == 1.0, "the same spell reads 'targets the opponent' from the other chair"


# ---- which blocker blocks which attacker -------------------------------------------------------------------------------------

BLOCK = {"s0": sc.seat(bf=["Balustrade Spy", "Guttersnipe"]), "s1": sc.seat(bf=["Kitchen Imp", "Guttersnipe"]), "attackers": {1: [0, 1]}}


def test_which_blocker_blocks_which_attacker_is_visible(world):
    vocab, enc = world
    a = sc.encode(sc.build(**{**BLOCK, "blocked": {1: {0: [(0, 0)], 1: [(0, 1)]}}}), 0, enc, vocab)  # Spy->Imp, Gutter->Gutter
    b = sc.encode(sc.build(**{**BLOCK, "blocked": {1: {0: [(0, 1)], 1: [(0, 0)]}}}), 0, enc, vocab)  # Gutter->Imp, Spy->Gutter
    assert sc.differs(a, b), "swapping the blocking assignments changed nothing the encoder can see"


def test_a_gang_block_links_every_blocker_to_the_attacker(world):
    vocab, _enc = world
    tokens = rows_of({**BLOCK, "blocked": {1: {0: [(0, 0), (0, 1)]}}}, 0, vocab)
    imp = row_named(tokens, vocab, "Kitchen Imp")
    spy, gutter = row_named(tokens, vocab, "Balustrade Spy", mine=True), row_named(tokens, vocab, "Guttersnipe", mine=True)
    assert len(slots(imp, "dst")) == 2, "an attacker blocked twice is the destination of two edges"
    assert len(slots(spy, "src")) == 1 and len(slots(gutter, "src")) == 1 and slots(spy, "src") != slots(gutter, "src")
    assert slots(spy, "src") | slots(gutter, "src") == slots(imp, "dst")


# ---- auras --------------------------------------------------------------------------------------------------------------------

def test_an_aura_and_its_host_share_a_slot(world):
    vocab, _enc = world
    args = {"s0": sc.seat(bf=["Guttersnipe", "Kitchen Imp", ("Rancor", {"aura_on": (0, 1)})]), "s1": sc.seat()}
    tokens = rows_of(args, 0, vocab)
    aura, gutter, imp = (row_named(tokens, vocab, n) for n in ("Rancor", "Guttersnipe", "Kitchen Imp"))
    assert len(slots(aura, "src")) == 1 and slots(aura, "src") == slots(imp, "dst") and not slots(gutter, "dst")


def test_two_different_auras_swapped_between_two_equal_hosts_is_visible(world):
    """Two creatures of identical size, two different auras: which aura is on which creature."""
    vocab, enc = world
    base = {"s0": sc.seat(bf=["Guttersnipe", "Kitchen Imp", ("Rancor", {"aura_on": (0, 0)}), ("Ancestral Mask", {"aura_on": (0, 1)})]), "s1": sc.seat()}
    swapped = {"s0": sc.seat(bf=["Guttersnipe", "Kitchen Imp", ("Rancor", {"aura_on": (0, 1)}), ("Ancestral Mask", {"aura_on": (0, 0)})]), "s1": sc.seat()}
    assert sc.differs(sc.encode(sc.build(**base), 0, enc, vocab), sc.encode(sc.build(**swapped), 0, enc, vocab))


# ---- canonical numbering ------------------------------------------------------------------------------------------------------

def test_link_numbering_does_not_depend_on_token_order_or_viewpoint(world):
    vocab, enc = world
    args = two_spells(0, 1)
    reordered = {**args, "s0": sc.seat(bf=["Kitchen Imp", "Guttersnipe"]),
                 "stack": [("Galvanic Blast", 1, ("bf", 0, 0)), ("Lightning Bolt", 1, ("bf", 0, 1))]}
    assert sc.same(sc.encode(sc.build(**args), 0, enc, vocab), sc.encode(sc.build(**reordered), 0, enc, vocab))
    a = sc.encode(sc.build(**args), 0, enc, vocab)
    b = sc.encode(sc.build(**sc.mirror(args)), 1, enc, vocab)
    assert sc.same(a, b), "the position seen from the other chair (seats swapped) must encode identically, links included"


def test_every_edge_gets_exactly_one_source_and_one_destination_bit(world):
    vocab, _enc = world
    args = {**BLOCK, "blocked": {1: {0: [(0, 0)], 1: [(0, 1)]}}, "stack": [("Lightning Bolt", 1, ("bf", 0, 0))]}
    tokens = rows_of(args, 0, vocab)
    assert sum(r[SRC:SRC + F.LINK_SLOTS].sum() for _i, r, _x in tokens) == 3  # 2 blocks + 1 spell target
    assert sum(r[DST:DST + F.LINK_SLOTS].sum() for _i, r, _x in tokens) == 3


def test_more_edges_than_slots_saturate_without_breaking(world):
    vocab, enc = world
    n = F.LINK_SLOTS + 4
    args = {"s0": sc.seat(bf=["Guttersnipe"] * n), "s1": sc.seat(bf=["Kitchen Imp"] * n), "attackers": {1: list(range(n))},
            "blocked": {1: {i: [(0, i)] for i in range(n)}}}
    tokens = rows_of(args, 0, vocab)
    rows = np.asarray([r for _i, r, _x in tokens])
    assert rows.min() >= 0.0 and rows.max() <= 1.0
    assert rows[:, DST + F.LINK_SLOTS - 1].sum() >= 5, "the overflow edges share the last slot"
    sc.encode(sc.build(**args), 0, enc, vocab)


def _bag(tokens):
    return sorted((int(i), tuple(np.round(r, 6))) for i, r, _x in tokens)


def test_identical_edges_on_both_sides_are_numbered_by_content_not_by_token_order(world):
    """Regression: when both players have an identical aura on an identical creature, the two edges tie on
    perspective-free content, and the tie used to be broken by token order (so which of them got slot 0 depended on
    list order, and my aura and theirs are NOT interchangeable)."""
    vocab, _enc = world
    one = {"s0": sc.seat(bf=["Guttersnipe", ("Rancor", {"aura_on": (0, 0)})]),
           "s1": sc.seat(bf=["Guttersnipe", ("Rancor", {"aura_on": (1, 0)})])}
    other = {"s0": sc.seat(bf=[("Rancor", {"aura_on": (0, 1)}), "Guttersnipe"]),
             "s1": sc.seat(bf=[("Rancor", {"aura_on": (1, 1)}), "Guttersnipe"])}
    for seat in (0, 1):
        assert _bag(rows_of(one, seat, vocab)) == _bag(rows_of(other, seat, vocab)), f"seat {seat}: list order changed the encoding"


def test_edges_that_differ_only_in_what_targets_their_endpoints_are_still_ordered_by_content(world):
    """Two identical enchanted creatures, only one of them targeted by a spell: which aura gets which slot must follow
    the creatures' content, not their order on the battlefield."""
    vocab, _enc = world
    def position(order):
        bf = [("Guttersnipe", {}), ("Guttersnipe", {})]
        auras = [("Rancor", {"aura_on": (0, order[0])}), ("Rancor", {"aura_on": (0, order[1])})]
        return {"s0": sc.seat(bf=bf + auras), "s1": sc.seat(), "stack": [("Lightning Bolt", 1, ("bf", 0, 0))]}
    assert _bag(rows_of(position((0, 1)), 0, vocab)) == _bag(rows_of(position((1, 0)), 0, vocab))
