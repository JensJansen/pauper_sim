"""Layer 3 -- static card features and the vocabulary. For EVERY card the engine knows (plus every token def) the
static vector is re-derived field by field from the card definition and registry by separate code, and the vocab's
contract (stable, non-zero, collision-free indices) is pinned. No game is played: fast tier."""
import collections

import numpy as np
import pytest

import game
from rl.model import features as F
from rl.model.card_behavior import (
    BEHAVIOR_FEATURE_DIM, CATEGORICAL_VOCAB, CONSTANT_VOCAB, IDENTIFIER_VOCAB, behavior_features,
)
from rl.roster import TOKEN_DEFS

ALL_DEFS = list(game.CARD_DEFS.values()) + list(TOKEN_DEFS)

# Cards whose COMPLETE static vector (printed stats + registry keys + behavior signature) is identical. Each pair was
# checked by reading both cards' definitions and code: they are the same card as implemented here (same cost, same
# stats and keywords, same cast function; the two lands have the same subtypes, mana and entry rules). Anything else
# colliding is a new gap and fails this test.
KNOWN_STATIC_COLLISIONS = {
    frozenset({"Fyndhorn Elves", "Llanowar Elves"}),
    frozenset({"Contaminated Aquifer", "Ice Tunnel"}),
}
BASE_DIM = F.STATIC_FEATURE_DIM - BEHAVIOR_FEATURE_DIM  # everything before the behavior block


def _expected_static(cd):
    """The static vector, built block by block from the card's own data."""
    spec = game.EFFECT_REGISTRY.get(cd.effect_id, {})
    cost = cd.cast_cost or {}
    cap = max([n for c in game.CARD_DEFS.values() for n in (c.cast_cost or {}).values()] + [1])
    out = [min(cost.get(c, 0), cap) / cap for c in game.POOL_COLORS] + [min(cost.get("generic", 0), cap) / cap]
    out += [float(cd.card_type == t) for t in game.CardType]
    out += [min(cd.extra.get("power", 0), 10) / 10, min(cd.extra.get("toughness", 0), 10) / 10]
    kws = set(spec.get("keywords", ()))
    all_kws = sorted({k for s in game.EFFECT_REGISTRY.values() for k in s.get("keywords", ())})
    out += [float(k in kws) for k in all_kws]
    mana = spec.get("mana")
    colors = set()
    if mana is not None:
        colors = {mana[1]} if mana[0] in ("fixed", "count", "count_all") else set(mana[1]) if mana[0] in ("flexible", "fixed_multi") else {"C"}
    out += [float(mana is not None)] + [float(c in colors) for c in game.POOL_COLORS]
    out += [float(mana is not None and mana[0] in ("count", "count_all", "tron")), float(bool(spec.get("enters_tapped"))),
            float("filter_mana" in spec)]
    out += [float(k in spec) for k in sorted({k for s in game.EFFECT_REGISTRY.values() for k in s})]
    kinds = set(spec.get("pending_kinds", ()))
    out += [float(k in kinds) for k in sorted({k for s in game.EFFECT_REGISTRY.values() for k in s.get("pending_kinds", ())})]
    out += [float(bool(cd.extra.get(fl))) for fl in ("artifact", "basic", "defender", "indestructible", "devoid")]
    subs = cd.extra.get("subtypes", ())
    out += [float(s in subs) for s in sorted({s for c in game.CARD_DEFS.values() for s in c.extra.get("subtypes", ())})]
    return np.asarray(out, dtype=np.float32)


def test_static_dimension_is_the_sum_of_its_blocks():
    dims = [len(game.POOL_COLORS) + 1, len(F.CARD_TYPES), 2, len(F.KEYWORD_VOCAB), F.MANA_FEATURE_DIM,
            len(F.SPEC_KEY_VOCAB), len(F.PENDING_KIND_VOCAB), len(F.EXTRA_FLAG_VOCAB), len(F.SUBTYPE_VOCAB)]
    assert sum(dims) + BEHAVIOR_FEATURE_DIM == F.STATIC_FEATURE_DIM
    assert len(IDENTIFIER_VOCAB) + len(CONSTANT_VOCAB) + len(CATEGORICAL_VOCAB) == BEHAVIOR_FEATURE_DIM


@pytest.mark.parametrize("cd", ALL_DEFS, ids=lambda cd: cd.name)
def test_static_vector_matches_an_independent_derivation(cd):
    got = np.asarray(F.static_card_features(cd), dtype=np.float32)
    want = _expected_static(cd)
    assert got.shape == (F.STATIC_FEATURE_DIM,)
    assert np.isfinite(got).all() and got.min() >= 0.0 and got.max() <= 1.0
    assert list(got[BASE_DIM:]) == behavior_features(cd), "the behavior block is exactly behavior_features()"
    got = got[:BASE_DIM]
    bad = np.flatnonzero(got != want)
    assert not bad.size, f"{cd.name}: columns {bad.tolist()} differ (got {got[bad]}, expected {want[bad]})"


def test_static_features_are_deterministic_and_the_cache_is_coherent():
    vocab = F.CardVocab([[(n, 1)] for n in ("Lightning Bolt", "Guttersnipe", "Mountain")])
    for name in ("Lightning Bolt", "Guttersnipe", "Mountain"):
        a = F.static_card_features(vocab.card_def(name))
        assert a == F.static_card_features(vocab.card_def(name))
        assert list(F.cached_static_card_features(name, vocab)) == list(a)
        assert list(F.cached_static_card_features(name, vocab)) == list(a)  # second hit comes from the cache


def test_static_collisions_are_exactly_the_verified_identical_pairs():
    groups = collections.defaultdict(set)
    for cd in game.CARD_DEFS.values():
        groups[tuple(F.static_card_features(cd))].add(cd.name)
    found = {frozenset(v) for v in groups.values() if len(v) > 1}
    assert found == KNOWN_STATIC_COLLISIONS, (f"new collisions: {found - KNOWN_STATIC_COLLISIONS}; "
                                              f"resolved ones: {KNOWN_STATIC_COLLISIONS - found}")


def test_vocab_gives_every_decklist_card_a_unique_nonzero_index():
    import glob
    lists = [game.parse_decklist_file(p) for p in sorted(glob.glob("../data/*.txt"))]
    vocab = F.CardVocab(lists, token_card_defs=TOKEN_DEFS)
    names = {n for dl in lists for n, *_ in dl} | {cd.name for cd in TOKEN_DEFS}
    idx = {n: vocab.index(n) for n in names}
    assert all(i >= 1 for i in idx.values())
    assert len(set(idx.values())) == len(idx), "two cards share a vocab index"
    assert vocab.index("definitely not a card") == 0
    assert vocab.size == max(vocab.name_to_index.values()) + 1


def test_vocab_is_independent_of_decklist_order_and_duplicates():
    import glob
    lists = [game.parse_decklist_file(p) for p in sorted(glob.glob("../data/*.txt"))]
    a = F.CardVocab(lists, token_card_defs=TOKEN_DEFS)
    b = F.CardVocab(list(reversed(lists)) + lists, token_card_defs=TOKEN_DEFS)
    assert a.name_to_index == b.name_to_index


def test_every_token_and_back_face_resolves_to_a_card_def():
    import glob
    lists = [game.parse_decklist_file(p) for p in sorted(glob.glob("../data/*.txt"))]
    vocab = F.CardVocab(lists, token_card_defs=TOKEN_DEFS)
    for cd in TOKEN_DEFS:
        assert vocab.card_def(cd.name) is cd and vocab.index(cd.name) >= 1
    assert vocab.index("Delver of Secrets") >= 1, "Delver is expected in some roster deck; pick another DFC if not"
    assert vocab.index("Insectile Aberration") == vocab.index("Delver of Secrets")
    assert vocab.card_def("Insectile Aberration").name == "Insectile Aberration"


# ---- the behavior block: what a card's code DOES ------------------------------------------------------------------------

def _ids_and_consts(name):
    f = behavior_features(game.CARD_DEFS[name])
    ids = {IDENTIFIER_VOCAB[i] for i, x in enumerate(f[:len(IDENTIFIER_VOCAB)]) if x}
    consts = {c for c, x in zip(CONSTANT_VOCAB, f[len(IDENTIFIER_VOCAB):len(IDENTIFIER_VOCAB) + len(CONSTANT_VOCAB)]) if x}
    return ids, consts


@pytest.mark.parametrize("a,b", [
    ("Cast Down", "Unexpected Fangs"), ("Dispel", "Mental Note"), ("Galvanic Blast", "Lightning Bolt"),
    ("Fanatical Offering", "Reckoner's Bargain"), ("Cartouche of Solidarity", "Ethereal Armor"),
    ("Urza's Mine", "Urza's Tower"), ("Urza's Mine", "Urza's Power Plant"), ("Urza's Tower", "Urza's Power Plant"),
])
def test_cards_that_used_to_collide_are_now_told_apart_by_their_behavior(a, b):
    assert behavior_features(game.CARD_DEFS[a]) != behavior_features(game.CARD_DEFS[b]), f"{a} and {b} look identical"


def test_the_behavior_block_says_what_the_card_does():
    """Anchors on cards whose effect is stated in their own docstring."""
    ids, consts = _ids_and_consts("Lightning Bolt")
    assert 3 in consts and "begin_choose_any_target" in ids, "Lightning Bolt: 3 damage to any target"
    assert {2, 4} <= _ids_and_consts("Galvanic Blast")[1], "Galvanic Blast: 2 damage, 4 with metalcraft"
    ids, consts = _ids_and_consts("Mental Note")
    assert {"mill", "draw"} <= ids and 2 in consts, "Mental Note: mill two, draw a card"
    assert "mill" not in _ids_and_consts("Dispel")[0], "Dispel is a counterspell, not a mill"
    tron = {n: [v for (k, v), x in zip(CATEGORICAL_VOCAB, behavior_features(game.CARD_DEFS[n])[-len(CATEGORICAL_VOCAB):]) if x]
            for n in ("Urza's Mine", "Urza's Tower", "Urza's Power Plant")}
    assert tron == {"Urza's Mine": ["Mine"], "Urza's Tower": ["Tower"], "Urza's Power Plant": ["Power Plant"]}


def test_every_spell_has_a_behavior_signature():
    """An instant or sorcery's whole effect lives in code, so its behavior block must never be blank. (Permanents may
    legitimately read blank: their only code is the generic 'cast a permanent' helper, shared by most of the pool, so
    boilerplate by construction; what sets them apart -- haste, madness, an ETB trigger, mana abilities -- is carried by
    the registry-key, keyword and mana blocks.)"""
    blank = [cd.name for cd in game.CARD_DEFS.values()
             if cd.card_type in (game.CardType.INSTANT, game.CardType.SORCERY)
             and not any(behavior_features(cd)[:len(IDENTIFIER_VOCAB) + len(CONSTANT_VOCAB)])]
    assert not blank, f"spells with an empty behavior signature: {blank}"


def test_behavior_vocabulary_is_deterministic_and_deduplicated():
    import subprocess
    import sys
    code = "from rl.model.card_behavior import IDENTIFIER_VOCAB as I, CONSTANT_VOCAB as C, CATEGORICAL_VOCAB as K; print(hash(I) if False else '|'.join(I), C, K)"
    a = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True).stdout
    b = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True).stdout
    assert a == b and a.strip()
    assert len(set(IDENTIFIER_VOCAB)) == len(IDENTIFIER_VOCAB)
