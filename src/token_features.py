"""Per-card, per-token feature extraction for the attention-based observation
encoding (docs discussed on the attention-opponent-encoding branch -- full
per-card fidelity instead of drl_env.py's fixed per-(name, slot) observation
slots). Two independent pieces per token, deliberately kept separate:

1. A card VOCABULARY INDEX (this module's CardVocab) -- looked up by the
   consuming network into a learned nn.Embedding, so the network can tell
   two cards with identical stats apart. Not computed here; this module only
   assigns stable, deterministic indices.
2. A STATIC, hand-authored structured feature vector (mana cost, card type,
   base power/toughness, keywords) -- deterministic, no learning, always
   available even for a card the embedding table has never trained on. This
   is what "Learning With Generalised Card Representations for Magic: The
   Gathering" (arxiv 2407.05879) found actually drives generalization to
   unseen cards; the identity embedding alone does not.

Dynamic per-instance state (tapped, damage, blocked-as-attacker, etc.) is a
separate, per-Permanent concern -- see dynamic_permanent_features below,
combined with the static vector by the tokenizer (not built yet) into one
per-token feature row.
"""

import json
import os

import game

# Full keyword vocabulary across every card in the game (scanned from the
# registry itself, not hand-maintained -- see this module's own self-check
# for how it's derived) rather than hand-listing keywords, so a keyword
# added to a future card is automatically covered without editing this file.
KEYWORD_VOCAB = tuple(sorted({
    kw for spec in game.EFFECT_REGISTRY.values() if "keywords" in spec for kw in spec["keywords"]
}))

CARD_TYPES = tuple(game.CardType)

# Same fixed-cap-then-normalize idiom drl_env.py already uses throughout
# (POOL_CAP, PER_CREATURE_POWER_CAP, ...) -- kept consistent rather than
# inventing a new normalization convention for this module.
MANA_PIP_CAP = 6
POWER_TOUGHNESS_CAP = 10

STATIC_FEATURE_DIM = len(game.POOL_COLORS) + 1 + len(CARD_TYPES) + 2 + len(KEYWORD_VOCAB)


class CardVocab:
    """name -> stable integer index, built once from the union of every
    decklist AND every token CardDef a training run needs to recognize.
    Index 0 is reserved as a padding/unknown sentinel (never assigned to a
    real card) so a padded token slot's embedding lookup is always valid
    and distinguishable from every real card.

    token_card_defs: real CardDef objects (e.g. game.BLOOD_TOKEN_CARD_DEF),
    same objects the runner's own TOKEN_CARD_DEFS_BY_NAME resolves configs
    through -- required because a token (Blood, Robot, ...) can appear on
    the battlefield mid-game but is NEVER a game.CARD_DEFS entry (confirmed
    the hard way: the first version of this module assumed every
    battlefield permanent's name resolves via game.CARD_DEFS[name] and
    crashed with a KeyError the first time a mono_red_madness game actually
    created a Blood token). drl_env.py's own fixed-slot observation just
    drops tokens from the encoding entirely (its own documented "a token
    has no observation representation at all" choice) -- reproducing that
    gap here would be a real, avoidable fidelity loss this migration exists
    to remove, not a corner worth cutting again."""

    def __init__(self, decklists, token_card_defs=(), vocab_path=None):
        """vocab_path: optional JSON file persisting name -> index across
        separate runs/deck rosters (the league's own use -- see
        token_pool.py). Append-only: an index a persisted file already gave
        a name is NEVER reassigned, since that would silently invalidate
        every existing checkpoint's embedding table the moment a new deck
        introduces a new card that happens to sort earlier. New names get
        the next free indices (stable sorted order among themselves, so
        repeated runs with the same new roster are reproducible); vocab.size
        reflects the FULL persisted vocabulary, not just this call's own
        decklists, so a shared embedding table stays valid for any deck
        roster this file has ever seen, not only the current one. Omitting
        vocab_path keeps the old behavior (fresh, non-persisted, this call's
        decklists only) for every existing non-league caller/test."""
        self.card_def_by_name = dict(game.CARD_DEFS)
        for card_def in token_card_defs:
            self.card_def_by_name[card_def.name] = card_def
        needed_names = {name for decklist in decklists for name, *_rest in decklist} | {cd.name for cd in token_card_defs}

        self.name_to_index = {}
        if vocab_path is not None and os.path.exists(vocab_path):
            with open(vocab_path) as f:
                self.name_to_index = json.load(f)

        new_names = sorted(needed_names - set(self.name_to_index))
        next_index = max(self.name_to_index.values(), default=0) + 1
        for name in new_names:
            self.name_to_index[name] = next_index
            next_index += 1
        self.size = max(self.name_to_index.values(), default=0) + 1  # +1 for the padding/unknown sentinel at index 0

        if vocab_path is not None and new_names:
            os.makedirs(os.path.dirname(vocab_path) or ".", exist_ok=True)
            with open(vocab_path, "w") as f:
                json.dump(self.name_to_index, f, indent=2, sort_keys=True)

    def index(self, name):
        return self.name_to_index.get(name, 0)

    def card_def(self, name):
        return self.card_def_by_name[name]


def static_card_features(card_def):
    """Deterministic structured feature vector for `card_def` (a real
    CardDef object, not a name -- see CardVocab.card_def, the one place
    that resolves a name to its CardDef, covering both game.CARD_DEFS
    entries and token CardDefs uniformly) -- mana cost (per-color pip
    counts + generic, normalized), card type (one-hot), base power/
    toughness (normalized, 0 for non-creatures), keywords (multi-hot).
    Same for every copy of a card and every game state; never touches
    game state, so cacheable per name (see _STATIC_FEATURE_CACHE)."""
    cost = card_def.cast_cost or {}
    out = []
    for color in game.POOL_COLORS:
        out.append(min(cost.get(color, 0), MANA_PIP_CAP) / MANA_PIP_CAP)
    out.append(min(cost.get("generic", 0), MANA_PIP_CAP) / MANA_PIP_CAP)
    for card_type in CARD_TYPES:
        out.append(1.0 if card_def.card_type == card_type else 0.0)
    out.append(min(card_def.extra.get("power", 0), POWER_TOUGHNESS_CAP) / POWER_TOUGHNESS_CAP)
    out.append(min(card_def.extra.get("toughness", 0), POWER_TOUGHNESS_CAP) / POWER_TOUGHNESS_CAP)
    card_keywords = game.EFFECT_REGISTRY.get(card_def.effect_id, {}).get("keywords", ())
    for kw in KEYWORD_VOCAB:
        out.append(1.0 if kw in card_keywords else 0.0)
    return out


_STATIC_FEATURE_CACHE = {}  # name -> static_card_features(vocab.card_def(name)), same "keyed by content, computed once" pattern as drl_env._CARD_LOOKUP_CACHE


def cached_static_card_features(name, vocab):
    cached = _STATIC_FEATURE_CACHE.get(name)
    if cached is None:
        cached = static_card_features(vocab.card_def(name))
        _STATIC_FEATURE_CACHE[name] = cached
    return cached


# ---------------------------------------------------------------------------
# Tokenization: turn each PUBLIC zone (battlefield/graveyard/stack/exile,
# both seats) into a variable-length list of per-token feature rows, instead
# of drl_env.py's fixed per-(name, slot) observation slots. Deliberately
# does NOT touch hand or library -- those stay hidden (aggregate count only,
# same as today's _opponent_aggregate_features), preserving the existing
# hidden-information guarantee. See this module's own docstring for the
# static/dynamic split; DYNAMIC_FEATURE_DIM below is this function's own
# per-instance half.
# ---------------------------------------------------------------------------

ZONES = ("battlefield", "graveyard", "stack", "exile")
# untapped, tapped, effective_power, effective_toughness, blocked_as_attacker,
# committed_as_blocker, zone one-hot (4), side flag (1) -- see _token_row's
# own inline comments for what each slot means and why.
DYNAMIC_FEATURE_DIM = 6 + len(ZONES) + 1
TOKEN_FEATURE_DIM = STATIC_FEATURE_DIM + DYNAMIC_FEATURE_DIM

PER_CREATURE_POWER_CAP = 20  # matches drl_env.PER_CREATURE_POWER_CAP -- same cap, same reasoning
PER_CREATURE_TOUGHNESS_CAP = 20


def _token_row(name, zone, is_mine, vocab, permanent=None, owner_idx=None, enchanting_auras=None, state=None):
    """One token's full feature row: static card identity/stats (always
    present) + dynamic per-instance state (mostly zero outside battlefield,
    since graveyard/stack/exile cards aren't permanents with tapped/combat
    state -- same "inert, not padded-away" convention drl_env.py's own
    creature-slot block uses for an absent slot).

    owner_idx: this permanent's OWN true seat (0 or 1) -- NOT derivable from
    is_mine alone, since is_mine is relative to whichever seat's own
    perspective build_token_set is building for right now, while blocked_by
    lookups need the permanent's actual owning seat regardless of
    perspective (same distinction drl_env._creature_slot_block's own
    owner_idx parameter draws)."""
    row = list(cached_static_card_features(name, vocab))
    untapped = tapped = eff_power = eff_toughness = blocked_attacker = committed_blocker = 0.0
    if permanent is not None:
        untapped = 0.0 if permanent.tapped else 1.0
        tapped = 1.0 if permanent.tapped else 0.0
        own_blocked_by = state.players[owner_idx].blocked_by
        # gang-blocking: blocked_by VALUES are lists of blockers now, so
        # flatten to the flat set of committed-blocker permanents.
        other_committed_blockers = {b for bs in state.players[1 - owner_idx].blocked_by.values() for b in bs}
        eff_power = min(game.permanent_power(state, permanent, enchanting_auras=enchanting_auras),
                         PER_CREATURE_POWER_CAP) / PER_CREATURE_POWER_CAP
        remaining_t = max(game.permanent_toughness(state, permanent, enchanting_auras=enchanting_auras)
                           - permanent.damage_marked, 0)
        eff_toughness = min(remaining_t, PER_CREATURE_TOUGHNESS_CAP) / PER_CREATURE_TOUGHNESS_CAP
        blocked_attacker = 1.0 if permanent in own_blocked_by else 0.0
        committed_blocker = 1.0 if permanent in other_committed_blockers else 0.0
    row += [untapped, tapped, eff_power, eff_toughness, blocked_attacker, committed_blocker]
    row += [1.0 if zone == z else 0.0 for z in ZONES]
    row.append(1.0 if is_mine else 0.0)
    assert len(row) == TOKEN_FEATURE_DIM
    return row


def build_token_set(state, my_seat_idx, vocab):
    """Every public-zone card for BOTH seats, as a flat list of (vocab_index,
    feature_row, identity) triples -- one shared token set, side-flagged
    rather than two separately-encoded halves, so a joint Set Transformer
    can let tokens from both sides attend to each other (docs discussion:
    relative valuations depend on cross-side context, e.g. an attacker's
    real threat level depends on what can block it). Order within the
    returned list is NOT meaningful (a permutation-invariant encoder
    consumes it) but IS deterministic given the same state, for
    reproducibility.

    identity: the live Permanent object for a battlefield token, else None
    -- the pointer-network action head (token_deck.py) matches a legal
    target (game.choose_permanent_options et al. already return/operate on
    these same Permanent objects, id()-based, same convention this engine
    already uses throughout -- e.g. enchanting_by_target below) back to
    "which row of this token batch is that" via this field. Only battlefield
    tokens ever need it: every pointer-scored decision this branch builds
    (Attack, Assign Blocker, Choose target, Choose opponent's) targets a
    permanent, never a graveyard/exile/stack card (those go through
    separate, non-pointer resolution kinds -- choose_graveyard_card, etc.
    -- untouched by this migration).

    Hand and library are deliberately excluded -- those stay hidden
    (aggregate count only), matching drl_env._opponent_aggregate_features'
    own "only hand/library CONTENTS are hidden" rule. Violating that here
    would leak hidden information the rest of this engine carefully
    protects."""
    opponent_seat_idx = 1 - my_seat_idx
    enchanting_by_target = {}
    for player in state.players:
        for aura in player.battlefield:
            target = aura.flags.get("enchanting")
            if target is not None:
                enchanting_by_target.setdefault(id(target), []).append(aura)

    tokens = []
    for seat_idx in (my_seat_idx, opponent_seat_idx):
        is_mine = seat_idx == my_seat_idx
        player = state.players[seat_idx]
        for p in player.battlefield:
            auras = enchanting_by_target.get(id(p), ())
            tokens.append((vocab.index(p.card_def.name),
                            _token_row(p.card_def.name, "battlefield", is_mine, vocab, permanent=p, owner_idx=seat_idx,
                                       enchanting_auras=auras, state=state),
                            p))
        for card_def in player.graveyard:
            tokens.append((vocab.index(card_def.name), _token_row(card_def.name, "graveyard", is_mine, vocab), None))
        for card_def, _plotted_turn in player.exile:
            tokens.append((vocab.index(card_def.name), _token_row(card_def.name, "exile", is_mine, vocab), None))
    for entry in state.stack:
        is_mine = entry["controller"] == my_seat_idx
        tokens.append((vocab.index(entry["card_def"].name),
                        _token_row(entry["card_def"].name, "stack", is_mine, vocab), None))
    return tokens


if __name__ == "__main__":
    # ponytail self-check: run via `python token_features.py` from src/.
    import terminated  # noqa: F401 -- confirms the module still loads alongside this self-check, same convention the other self-checks here use
    from game.state import GameState, PlayerState, Permanent

    decklist_a = game.parse_decklist_file("../data/mono_red_madness.txt")
    decklist_b = game.parse_decklist_file("../data/rakdos_madness.txt")
    token_defs = (game.BLOOD_TOKEN_CARD_DEF,)
    vocab = CardVocab([decklist_a, decklist_b], token_card_defs=token_defs)

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

    # Mountain (a land, cast_cost=None) must not crash on cost.get(...) -- the `or {}` guard.
    feat_land = cached_static_card_features("Mountain", vocab)
    assert len(feat_land) == STATIC_FEATURE_DIM
    assert all(v == 0.0 for v in feat_land[:len(game.POOL_COLORS) + 1])
    type_start = len(game.POOL_COLORS) + 1
    land_type_idx = type_start + CARD_TYPES.index(game.CardType.LAND)
    assert feat_land[land_type_idx] == 1.0

    # Blood (a token, NOT in game.CARD_DEFS) -- must resolve via the vocab's
    # own merged card_def_by_name, not crash with a KeyError (confirmed the
    # hard way: this is exactly what broke the first version of this module
    # the moment a real mono_red_madness game actually created one).
    feat_blood = cached_static_card_features("Blood", vocab)
    assert len(feat_blood) == STATIC_FEATURE_DIM
    artifact_type_idx = type_start + CARD_TYPES.index(game.CardType.ARTIFACT)
    assert feat_blood[artifact_type_idx] == 1.0

    # A card shared between both decks gets the SAME index/features regardless of which deck's list it's looked up through.
    assert vocab.index("Lightning Bolt") == vocab.name_to_index["Lightning Bolt"]
    shared_names = ({n for n, *_r in decklist_a} & {n for n, *_r in decklist_b})
    assert "Lightning Bolt" in shared_names and len(shared_names) == 8

    print(f"token_features.py self-check: OK (vocab_size={vocab.size}, static_feature_dim={STATIC_FEATURE_DIM}, "
          f"keywords={KEYWORD_VOCAB})")

    # build_token_set, against a hand-built two-seat state covering every
    # public zone -- specifically constructed with seat 1 (not seat 0) as
    # "my_seat_idx" for part of this check, since a bug that only shows up
    # when my_seat_idx=1 (using state.players[0] unconditionally instead of
    # the permanent's own true owner_idx) would pass a seat-0-only test.
    seat0 = PlayerState(on_the_play=True)
    seat0.battlefield = [Permanent(game.CARD_DEFS["Guttersnipe"]), Permanent(game.CARD_DEFS["Mountain"], tapped=True)]
    seat0.graveyard = [game.CARD_DEFS["Lightning Bolt"]]
    seat1 = PlayerState(on_the_play=False)
    seat1.battlefield = [Permanent(game.CARD_DEFS["Kitchen Imp"])]
    seat1.exile = [(game.CARD_DEFS["Faithless Looting"], 3)]
    fs_state = GameState(on_the_play=True, players=[seat0, seat1])
    fs_state.stack = [{"card_def": game.CARD_DEFS["Fiery Temper"], "resolve": None, "controller": 1}]
    # seat 0's Guttersnipe blocks seat 1's Kitchen Imp -- exercises the
    # blocked_as_attacker/committed_as_blocker dynamic features on both sides.
    attacker, blocker = seat1.battlefield[0], seat0.battlefield[0]
    seat1.attackers = [attacker]
    seat1.blocked_by = {attacker: [blocker]}

    for my_seat in (0, 1):
        tokens = build_token_set(fs_state, my_seat, vocab)
        # 2 (seat0 battlefield) + 1 (seat0 graveyard) + 1 (seat1 battlefield)
        # + 1 (seat1 exile) + 1 (shared stack) = 6 tokens, regardless of
        # which seat's own perspective this is built from.
        assert len(tokens) == 6, f"expected 6 tokens for my_seat={my_seat}, got {len(tokens)}"
        for vocab_idx, row, identity in tokens:
            assert isinstance(vocab_idx, int) and vocab_idx != 0  # every token here is a real, known card
            assert len(row) == TOKEN_FEATURE_DIM
        battlefield_tokens = [(idx, row, ident) for idx, row, ident in tokens if ident is not None]
        assert len(battlefield_tokens) == 3, "3 battlefield permanents (2 seat0 + 1 seat1) must carry identity"
        non_battlefield_tokens = [(idx, row, ident) for idx, row, ident in tokens if ident is None]
        assert len(non_battlefield_tokens) == 3, "graveyard/exile/stack tokens must NOT carry a permanent identity"

        # Guttersnipe (seat 0's own permanent) -- side flag must match
        # whether seat 0 IS my_seat, not be hardcoded to "mine" regardless.
        guttersnipe_row = next(row for idx, row, ident in tokens if idx == vocab.index("Guttersnipe"))
        guttersnipe_identity = next(ident for idx, row, ident in tokens if idx == vocab.index("Guttersnipe"))
        assert guttersnipe_identity is seat0.battlefield[0], "battlefield token identity must be the live Permanent object"
        expected_side_flag = 1.0 if my_seat == 0 else 0.0
        assert guttersnipe_row[-1] == expected_side_flag, (
            f"my_seat={my_seat}: Guttersnipe's side flag should be {expected_side_flag}, got {guttersnipe_row[-1]}"
        )
        # committed_as_blocker (the 6th-from-last dynamic slot, before the
        # 4 zone one-hot slots + 1 side-flag slot) must be set on Guttersnipe
        # regardless of my_seat -- this is exactly the bug the owner_idx fix
        # corrects: without it, blocked_by lookups silently used the wrong
        # seat's dict once my_seat_idx=1.
        committed_blocker_slot = -(len(ZONES) + 1) - 1
        assert guttersnipe_row[committed_blocker_slot] == 1.0, (
            f"my_seat={my_seat}: Guttersnipe should show committed_as_blocker=1.0, got "
            f"{guttersnipe_row[committed_blocker_slot]}"
        )

        # The stack entry's controller is seat 1 -- side flag must track
        # that controller, not my_seat, for the "is_mine" semantics to be
        # correct from whichever seat's own perspective this is built for.
        stack_row = next(row for idx, row, ident in tokens if idx == vocab.index("Fiery Temper"))
        assert stack_row[-1] == (1.0 if my_seat == 1 else 0.0)

    print("token_features.py build_token_set self-check: OK (both seats' own perspective, all 4 public zones)")

    # CardVocab persistence: append-only across separate construction calls
    # -- existing names must NEVER change index once a 3rd deck introduces
    # new names, and vocab.size must reflect the FULL persisted vocabulary
    # even when a later call only passes a SUBSET of decklists.
    import shutil
    import tempfile

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

    print("token_features.py CardVocab persistence self-check: OK (append-only across separate construction calls)")
