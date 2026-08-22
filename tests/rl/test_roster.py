"""Tests for rl.roster.build_pool: shared vocab/deck-ctx construction from the
league roster.

Relies on this directory's conftest.py chdir'ing to src/ for the duration
of each test -- build_pool()'s defaults (DECK_MANIFEST, VOCAB_PATH) and
rl.roster._load_roster's own per-deck path construction are all written
relative to src/.
"""
import json
import os
import tempfile

import pytest

from rl.roster import VOCAB_PATH, build_pool


@pytest.mark.slow
def test_build_pool_basic():
    decklists, vocab, deck_ctxs, fixed_tables = build_pool()
    # Roster-agnostic (reads whatever data/league_decks.json currently lists)
    # -- must include at least the two madness decks that are always present.
    assert {"mono_red_madness", "rakdos_madness"} <= set(decklists), "the two madness decks must always be in the roster"
    assert all(ctx[0] is vocab for ctx in deck_ctxs.values()), "every deck must share the SAME vocab instance"
    assert all(len(t) > 0 for t in fixed_tables.values()), "every deck must build a non-empty fixed action table"


@pytest.mark.slow
def test_build_pool_single_deck_roster_preserves_vocab_indices():
    decklists, vocab, deck_ctxs, fixed_tables = build_pool()

    # Roster is data-driven: a manifest naming just ONE deck must still work
    # standalone, and must NOT reassign that deck's existing (persisted)
    # vocab indices.
    tmp_manifest = os.path.join(tempfile.mkdtemp(), "one_deck.json")
    with open(tmp_manifest, "w") as f:
        json.dump({"mono_red_madness": "mono_red_madness.txt"}, f)
    decklists1, vocab1, _ctxs1, _tables1 = build_pool(manifest_path=tmp_manifest, vocab_path=VOCAB_PATH)
    assert set(decklists1) == {"mono_red_madness"}
    for name, idx in vocab.name_to_index.items():
        if name in vocab1.name_to_index:
            assert vocab1.name_to_index[name] == idx, f"single-deck roster reassigned {name!r}'s persisted index"
