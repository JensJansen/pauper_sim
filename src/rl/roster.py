"""Shared deck-pool setup for every token/attention driver script: ONE
CardVocab built from the whole league roster, so embedding indices line up
across scripts regardless of which single deck is training.

Deck roster lives in data/league_decks.json (deck name -> decklist
filename), not hardcoded. vocab.json persists the name->index mapping
across runs (CardVocab's append-only guarantee): adding a deck never
reassigns an existing card's index."""

import json

import game
from rl.decision.action_bridge import build_fixed_action_table
from rl.model.features import CardVocab

# Every token/pseudo-card the engine can put on a battlefield or the stack,
# registered unconditionally regardless of which deck creates which. A token
# def a roster never spawns just reserves a vocab index; a MISSING one is a
# hard KeyError. INITIATIVE_MARKER_CARD is Undercity's venture pseudo-card
# (rides the stack, appears in order_triggers).
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


def build_pool(manifest_path=DECK_MANIFEST, vocab_path=VOCAB_PATH, token_defs=TOKEN_DEFS):
    """Returns (decklists, vocab, deck_ctxs, fixed_tables), all dicts keyed
    by deck name, plus the one shared vocab. deck_ctxs[name] = (vocab,
    fixed_table), the tuple rl.decision.agent._seat_step expects.

    token_defs: token/pseudo-card CardDefs to reserve vocab indices +
    choosable-name actions for. A pool running cards that make other tokens
    (or the Undercity initiative marker) must pass the fuller set."""
    deck_files = _load_roster(manifest_path)
    decklists = {name: game.parse_decklist_file(path) for name, path in deck_files.items()}
    vocab = CardVocab(list(decklists.values()), token_card_defs=token_defs, vocab_path=vocab_path)

    # Cross-deck opponent-zone picks (a graveyard card, a revealed hand card)
    # are reached by POINTING at that card's token (action_bridge's
    # choose_graveyard_card pointer path), not a per-card fixed row -- so
    # each deck's fixed table stays scoped to its own cards.
    #
    # drl_env.build_action_table splits pending_kinds itself: every kind
    # confirmed cross-player is unconditional in every deck's table; every
    # self-only kind reads that decklist's own derive_pending_kinds. So no
    # pending_kinds union is needed here either.
    fixed_tables, deck_ctxs = {}, {}
    for name, decklist in decklists.items():
        fixed_table = build_fixed_action_table(decklist, token_card_defs=token_defs)
        fixed_tables[name] = fixed_table
        deck_ctxs[name] = (vocab, fixed_table)
    return decklists, vocab, deck_ctxs, fixed_tables
