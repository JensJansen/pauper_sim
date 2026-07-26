"""Shared deck-pool setup for every token/attention driver script -- ONE
CardVocab built from the WHOLE league roster, so every script's embedding
table indices line up with whatever shared stack / league checkpoints get
loaded, regardless of which single deck a given script is actually
training right now. Getting this vocab construction out of sync between
scripts would silently misalign the embedding table (same index, different
card, no error -- just wrong).

Deck roster lives in data/league_decks.json (deck name -> decklist
filename under data/), not hardcoded here -- adding deck #N is a data
change, not a code change. vocab.json (checkpoints/vocab.json) persists
the name->index mapping across separate runs and roster growth
(CardVocab's own append-only guarantee -- see its docstring): adding a
new deck to the manifest never reassigns an existing card's index, so
old checkpoints' embedding tables stay valid prefixes of any larger one."""

import json

import game
from token_action_bridge import build_fixed_action_table
from token_features import CardVocab

# ALL four token CardDefs the engine can ever put on a battlefield (Blood/
# Robot from the madness decks, Eldrazi Spawn from Malevolent Rumble,
# Warrior from Cartouche of Solidarity -- both in boggles) -- registered
# unconditionally regardless of which deck creates which. A token def that
# a given roster never actually spawns is harmless (it just reserves a
# vocab index); a MISSING one is a hard KeyError the moment a real game
# creates that token (confirmed the hard way with Blood/Robot earlier), so
# the complete set is the safe default, not a per-roster audit.
TOKEN_DEFS = (
    game.BLOOD_TOKEN_CARD_DEF, game.ROBOT_TOKEN_CARD_DEF,
    game.WARRIOR_TOKEN_CARD_DEF, game.ELDRAZI_SPAWN_TOKEN_CARD_DEF,
    game.FOOD_TOKEN_CARD_DEF,
)
DECK_MANIFEST = "../data/league_decks.json"
VOCAB_PATH = "../checkpoints/vocab.json"


def _load_roster(manifest_path):
    with open(manifest_path) as f:
        roster = json.load(f)
    return {name: f"../data/{filename}" for name, filename in roster.items()}


def build_pool(manifest_path=DECK_MANIFEST, vocab_path=VOCAB_PATH):
    """Returns (decklists, vocab, deck_ctxs, fixed_tables) -- all dicts
    keyed by deck name, plus the one shared (persisted, append-only) vocab.
    deck_ctxs[name] = (vocab, fixed_table, pending_kinds), the exact tuple
    token_train._seat_step expects."""
    deck_files = _load_roster(manifest_path)
    decklists = {name: game.parse_decklist_file(path) for name, path in deck_files.items()}
    vocab = CardVocab(list(decklists.values()), token_card_defs=TOKEN_DEFS, vocab_path=vocab_path)

    # Every card name anywhere in the league -- passed to each deck's fixed
    # action table as extra_choosable_names so a "Choose: X" action exists
    # for an OPPONENT's graveyard card too (Relic of Progenitus' exile can
    # target any player; without this, exiling from a cross-deck opponent's
    # graveyard softlocks with an empty action mask -- see build_action_
    # table's own extra_choosable_names docstring). The whole roster, not a
    # specific opponent: a trained model's action space stays fixed across
    # every matchup the league can produce.
    all_league_names = sorted({name for dl in decklists.values() for name, *_rest in dl})

    fixed_tables, deck_ctxs = {}, {}
    for name, decklist in decklists.items():
        pending_kinds = game.derive_pending_kinds(decklist)
        fixed_table = build_fixed_action_table(
            decklist, token_card_defs=TOKEN_DEFS, pending_kinds=pending_kinds,
            extra_choosable_names=all_league_names,
        )
        fixed_tables[name] = fixed_table
        deck_ctxs[name] = (vocab, fixed_table, pending_kinds)
    return decklists, vocab, deck_ctxs, fixed_tables


if __name__ == "__main__":
    # ponytail self-check: run via `python token_pool.py` from src/.
    decklists, vocab, deck_ctxs, fixed_tables = build_pool()
    # Roster-agnostic (reads whatever data/league_decks.json currently lists)
    # -- must include at least the two madness decks that are always present.
    assert {"mono_red_madness", "rakdos_madness"} <= set(decklists), "the two madness decks must always be in the roster"
    assert all(ctx[0] is vocab for ctx in deck_ctxs.values()), "every deck must share the SAME vocab instance"
    assert all(len(t) > 0 for t in fixed_tables.values()), "every deck must build a non-empty fixed action table"

    # Roster is data-driven: a manifest naming just ONE deck must still work
    # standalone, and must NOT reassign that deck's existing (persisted)
    # vocab indices.
    import os
    import tempfile

    tmp_manifest = os.path.join(tempfile.mkdtemp(), "one_deck.json")
    with open(tmp_manifest, "w") as f:
        json.dump({"mono_red_madness": "mono_red_madness.txt"}, f)
    decklists1, vocab1, _ctxs1, _tables1 = build_pool(manifest_path=tmp_manifest, vocab_path=VOCAB_PATH)
    assert set(decklists1) == {"mono_red_madness"}
    for name, idx in vocab.name_to_index.items():
        if name in vocab1.name_to_index:
            assert vocab1.name_to_index[name] == idx, f"single-deck roster reassigned {name!r}'s persisted index"

    print(f"token_pool.py self-check: OK (vocab_size={vocab.size}, "
          f"fixed_table_sizes={[len(t) for t in fixed_tables.values()]})")
