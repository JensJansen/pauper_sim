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
from rl.action_bridge import build_fixed_action_table
from rl.features import CardVocab

# EVERY token/pseudo-card the engine can put on a battlefield or the stack,
# across the whole 11-deck league -- registered unconditionally regardless of
# which deck creates which. A token def a given roster never spawns is harmless
# (it just reserves a vocab index); a MISSING one is a hard KeyError the moment
# a real game creates that token, so the complete set is the safe default, not
# a per-roster audit. INITIATIVE_MARKER_CARD is the Undercity venture
# triggered ability's pseudo-card (it rides the stack, and appears in
# order_triggers) -- same "must be vocab-known + choosable" category as a
# token.
TOKEN_DEFS = (
    game.BLOOD_TOKEN_CARD_DEF, game.ROBOT_TOKEN_CARD_DEF, game.WARRIOR_TOKEN_CARD_DEF,
    game.ELDRAZI_SPAWN_TOKEN_CARD_DEF, game.FOOD_TOKEN_CARD_DEF, game.CLUE_TOKEN_CARD_DEF,
    game.HUMAN_SOLDIER_TOKEN_CARD_DEF, game.TREASURE_TOKEN_CARD_DEF, game.BIRD_ILLUSION_TOKEN_CARD_DEF,
    game.SAMURAI_TOKEN_CARD_DEF, game.MAP_TOKEN_CARD_DEF, game.SKELETON_TOKEN_CARD_DEF,
    game.INITIATIVE_MARKER_CARD,
)
DECK_MANIFEST = "../data/league_decks.json"
VOCAB_PATH = "../checkpoints/vocab.json"


def _load_roster(manifest_path):
    with open(manifest_path) as f:
        roster = json.load(f)
    return {name: f"../data/{filename}" for name, filename in roster.items()}


def build_pool(manifest_path=DECK_MANIFEST, vocab_path=VOCAB_PATH, token_defs=TOKEN_DEFS, union_pending=True):
    """Returns (decklists, vocab, deck_ctxs, fixed_tables) -- all dicts
    keyed by deck name, plus the one shared (persisted, append-only) vocab.
    deck_ctxs[name] = (vocab, fixed_table, pending_kinds), the exact tuple
    rl.agent._seat_step expects.

    token_defs: the token/pseudo-card CardDefs to reserve vocab indices +
    choosable-name actions for (defaults to the league's TOKEN_DEFS). A pool
    that runs cards making other tokens -- or the Undercity initiative marker,
    a pseudo-card that appears on the stack -- must pass the fuller set.

    union_pending (default True): every deck's fixed table is built from the
    UNION of all decks' pending-resolution kinds, not just its own. A 2-player
    game can hand EITHER player a resolution the OTHER deck created -- pay_unless
    from a counter/Ward/Chain-Lightning rider, the venture kinds once the
    initiative is stolen, choose_any_target for a spell copy -- so a pool with
    those cross-player cards needs every table able to answer them or it
    softlocks (confirmed via cross-deck smoke). Default True now that the league
    roster includes those cards (dmir_terror/mono_blue_terror Ward, mono_red_
    rally Chain Lightning, elves initiative); pass False only for a hand-picked
    pool provably free of cross-player resolutions."""
    deck_files = _load_roster(manifest_path)
    decklists = {name: game.parse_decklist_file(path) for name, path in deck_files.items()}
    vocab = CardVocab(list(decklists.values()), token_card_defs=token_defs, vocab_path=vocab_path)

    # No extra_choosable_names: cross-deck OPPONENT-zone picks -- a graveyard
    # card (Relic of Progenitus) or a revealed hand card (Mesmeric Fiend) --
    # are reached by POINTING at that card's token (rl.action_bridge's
    # choose_graveyard_card pointer path; the revealed hand is faithfully
    # tokenized for the pick, see rl.features), not by a whole-league
    # "Choose: X" fixed row per card name. Each deck's fixed table stays
    # scoped to its own cards and does not grow with the roster.
    union_kinds = None
    if union_pending:
        union_kinds = tuple(sorted(set().union(*(set(game.derive_pending_kinds(dl)) for dl in decklists.values()))))

    fixed_tables, deck_ctxs = {}, {}
    for name, decklist in decklists.items():
        pending_kinds = union_kinds if union_pending else game.derive_pending_kinds(decklist)
        fixed_table = build_fixed_action_table(
            decklist, token_card_defs=token_defs, pending_kinds=pending_kinds,
        )
        fixed_tables[name] = fixed_table
        deck_ctxs[name] = (vocab, fixed_table, pending_kinds)
    return decklists, vocab, deck_ctxs, fixed_tables
