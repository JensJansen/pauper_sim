"""Per-card, per-token feature extraction for the attention-based observation
encoding (docs discussed on the attention-opponent-encoding branch -- full
per-card fidelity instead of the old fixed per-(name, slot) observation
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

Dynamic per-instance state (tapped, damage, blocked-as-attacker, currently
targeted by something on the stack, etc.) is a separate, per-token concern --
see _token_row below, combined with the static vector into one per-token
feature row by build_token_set (the tokenizer), which also assigns each
token's zone/side and (for battlefield/graveyard/stack tokens) its pointer-
addressable identity.
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

# Fixed-cap-then-normalize idiom: clamp a raw value to a cap, then divide to
# [0,1], rather than inventing a new normalization convention per feature.
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
    created a Blood token). The old fixed-slot observation just dropped
    tokens from the encoding entirely (a token had no observation
    representation at all) -- reproducing that gap here would be a real,
    avoidable fidelity loss this token encoding exists to remove, not a
    corner worth cutting again."""

    def __init__(self, decklists, token_card_defs=(), vocab_path=None):
        """vocab_path: optional JSON file persisting name -> index across
        separate runs/deck rosters (the league's own use -- see
        rl.pool). Append-only: an index a persisted file already gave
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

        # Double-faced back faces (Delver of Secrets -> Insectile Aberration): a
        # transformed permanent's card_def is swapped to its back face
        # (game.resolution.execute_may_transform), so features must resolve the
        # back name too. Register each back CardDef and ALIAS its vocab index to
        # its FRONT face's -- the shared card-embedding table (and every trained
        # checkpoint) keeps its exact size, and the agent perceives the flipped
        # creature as its front identity plus the already-transform-aware live
        # power/toughness (_token_row's dynamic half). Done AFTER the persist
        # write above so aliases never enter the persisted vocab or grow self.size.
        for front_def in list(self.card_def_by_name.values()):
            spec = game.EFFECT_REGISTRY.get(front_def.effect_id, {}).get("transform")
            if not spec or "card_def" not in spec:
                continue
            back = spec["card_def"]
            self.card_def_by_name[back.name] = back
            self.name_to_index.setdefault(back.name, self.name_to_index.get(front_def.name, 0))

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
# of the old fixed per-(name, slot) observation slots. Deliberately
# does NOT touch hand or library -- those stay hidden (aggregate count only),
# preserving the existing
# hidden-information guarantee. See this module's own docstring for the
# static/dynamic split; DYNAMIC_FEATURE_DIM below is this function's own
# per-instance half.
# ---------------------------------------------------------------------------

ZONES = ("battlefield", "graveyard", "stack", "exile")
# untapped, tapped, effective_power, effective_toughness, blocked_as_attacker,
# committed_as_blocker, targeted_by_mine, targeted_by_theirs, zone one-hot
# (4), side flag (1) -- see _token_row's own inline comments for what each
# slot means and why.
DYNAMIC_FEATURE_DIM = 8 + len(ZONES) + 1
TOKEN_FEATURE_DIM = STATIC_FEATURE_DIM + DYNAMIC_FEATURE_DIM

PER_CREATURE_POWER_CAP = 20  # clamp before normalizing to [0,1]; 20 covers this card subset's creatures
PER_CREATURE_TOUGHNESS_CAP = 20


def _token_row(name, zone, is_mine, vocab, permanent=None, owner_idx=None, enchanting_auras=None, state=None,
                targeted_by_mine=False, targeted_by_theirs=False):
    """One token's full feature row: static card identity/stats (always
    present) + dynamic per-instance state (mostly zero outside battlefield,
    since graveyard/stack/exile cards aren't permanents with tapped/combat
    state -- inert values, not padded away).

    owner_idx: this permanent's OWN true seat (0 or 1) -- NOT derivable from
    is_mine alone, since is_mine is relative to whichever seat's own
    perspective build_token_set is building for right now, while blocked_by
    lookups need the permanent's actual owning seat regardless of
    perspective.

    targeted_by_mine/targeted_by_theirs: is this token's own object CURRENTLY
    a declared target of something on the stack controlled by the perspective
    seat / by its opponent (see _stack_target_map) -- the target as DECLARED,
    not whether it would still resolve (real Magic: a target stays publicly
    known even after it becomes illegal and the spell later fizzles; that
    legality re-check is a separate, resolution-time concern this row doesn't
    make). Applies uniformly to battlefield permanents, graveyard cards, and
    stack entries themselves (a spell can target another spell on the stack,
    e.g. Counterspell) -- any zone whose tokens carry a stable per-instance
    identity a captured target can reference. Defaults False for zones
    nothing in this pool ever targets (exile, revealed hand)."""
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
    row += [1.0 if targeted_by_mine else 0.0, 1.0 if targeted_by_theirs else 0.0]
    row += [1.0 if zone == z else 0.0 for z in ZONES]
    row.append(1.0 if is_mine else 0.0)
    assert len(row) == TOKEN_FEATURE_DIM
    return row


def _stack_target_map(state):
    """Every CURRENT stack entry's declared targets (game.effects.stack.
    push_to_stack's own targets= tuple), split by whether the target is an
    OBJECT or a PLAYER:

    obj_controllers: id(target_object) -> the set of controller seat indices
    whose stack entry targets that object. Covers permanent ("creature"),
    graveyard-card, and stack-entry targets uniformly -- all three are
    addressed by real object identity, matching how build_token_set already
    carries each token's own identity, so one id()-keyed lookup serves every
    kind without special-casing any of them.

    player_controllers: player_idx -> the set of controller seat indices
    whose stack entry targets that PLAYER. Kept separate since no token
    exists for "the player" as an object (build_token_set never tokenizes
    one) -- rl.agent._scalar_features reads this half directly instead.

    Keyed by id(), never by the object itself: a raw stack-entry dict (a
    "stack_entry" target) is unhashable, so id()-as-int is the one
    representation safe to put in a dict across all four target kinds."""
    obj_controllers = {}
    player_controllers = {0: set(), 1: set()}
    for entry in state.stack:
        controller = entry["controller"]
        for kind, obj in entry.get("targets", ()):
            if kind == "player":
                player_controllers[obj].add(controller)
            else:
                obj_controllers.setdefault(id(obj), set()).add(controller)
    return obj_controllers, player_controllers


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

    identity: the live Permanent for a battlefield token, the exact CardInstance
    for a graveyard card (Permanent subclasses CardInstance, so battlefield vs
    graveyard is told apart by isinstance(., Permanent) first), the raw stack-
    entry dict for a stack token, the CardDef for a revealed hand card
    (DEFERRED -- hand still holds CardDefs), else None (exile only). The
    pointer-network action head (rl.deck) matches a legal target back to
    "which row of this token batch is that" via this field -- a Permanent for
    the four battlefield targeting kinds (Attack, Assign Blocker, Choose
    target, Choose opponent's), the CardInstance/CardDef object for choose_
    graveyard_card, and the stack-entry dict for choose_stack_target (matched
    by OBJECT IDENTITY in every case -- id()-keyed for choose_graveyard_card/
    choose_cast_copy/choose_stack_target specifically, since a stack entry is
    an unhashable dict, see rl.action_bridge -- so two same-named graveyard
    copies or simultaneous same-named spells are each individually
    addressable, and an opponent's graveyard/stack entry is reachable, which
    is why the per-name "Choose: X" fixed rows for either no longer exist).
    Exile cards have no pointer-addressable resolution, so None.

    Every token also carries two dynamic targeted-by-mine/targeted-by-theirs
    bits (see _token_row, _stack_target_map) reflecting whatever the CURRENT
    contents of state.stack declare as their targets -- a battlefield
    permanent, a graveyard card, or a stack entry itself (Counterspell
    targets a spell). A player-targeted burn spell has no token to carry that
    bit on; rl.agent._scalar_features surfaces it as a scalar instead.

    Hand and library are deliberately excluded from THIS token set -- those
    stay hidden here, per the "only hand/library CONTENTS are hidden" rule
    (violating that would leak hidden information the rest of this engine
    carefully protects). Their aggregate SIZE is not hidden in real Magic
    (either player can count a library or a hand) and is surfaced instead as
    a scalar, not a token -- rl.agent._scalar_features, not here. The ONE
    faithful exception to the content-hiding rule is a hand a card's own
    effect reveals (Mesmeric Fiend) -- tokenized only for the duration of
    that choose_graveyard_card pick, exactly what the real reveal shows the
    caster (see the hand-reveal block at the end of this function)."""
    opponent_seat_idx = 1 - my_seat_idx
    enchanting_by_target = {}
    for player in state.players:
        for aura in player.battlefield:
            target = aura.flags.get("enchanting")
            if target is not None:
                enchanting_by_target.setdefault(id(target), []).append(aura)

    obj_controllers, _player_controllers = _stack_target_map(state)  # player half is rl.agent._scalar_features's job

    def _targeted(obj):
        controllers = obj_controllers.get(id(obj), ())
        return my_seat_idx in controllers, opponent_seat_idx in controllers

    tokens = []
    for seat_idx in (my_seat_idx, opponent_seat_idx):
        is_mine = seat_idx == my_seat_idx
        player = state.players[seat_idx]
        for p in player.battlefield:
            auras = enchanting_by_target.get(id(p), ())
            tm, tt = _targeted(p)
            tokens.append((vocab.index(p.card_def.name),
                            _token_row(p.card_def.name, "battlefield", is_mine, vocab, permanent=p, owner_idx=seat_idx,
                                       enchanting_auras=auras, state=state, targeted_by_mine=tm, targeted_by_theirs=tt),
                            p))
        for inst in player.graveyard:
            # identity = the exact CardInstance, so two same-named graveyard cards
            # are DISTINCT pointer targets (and a flickered/returned card, being a
            # new instance, is a new token) -- MTG 400.7.
            tm, tt = _targeted(inst)
            tokens.append((vocab.index(inst.name),
                            _token_row(inst.name, "graveyard", is_mine, vocab, targeted_by_mine=tm, targeted_by_theirs=tt),
                            inst))
        for card_def, _plotted_turn in player.exile:
            tokens.append((vocab.index(card_def.name), _token_row(card_def.name, "exile", is_mine, vocab), None))
    for entry in state.stack:
        is_mine = entry["controller"] == my_seat_idx
        tm, tt = _targeted(entry)  # a spell/ability can itself be targeted, e.g. Counterspell
        tokens.append((vocab.index(entry["card_def"].name),
                        _token_row(entry["card_def"].name, "stack", is_mine, vocab, targeted_by_mine=tm, targeted_by_theirs=tt),
                        entry))  # pointer-addressable: choose_stack_target (Counterspell/Dispel/Spell Pierce) picks this exact entry

    # Faithful hand reveal: choose_graveyard_card is a generic pick, and
    # Mesmeric Fiend reuses it to exile a nonland card from the OPPONENT's hand
    # (black_cards.py passes graveyard=<a player's hand>). Real MTG reveals that
    # hand to the caster, so when the pick is over a player's hand, tokenize it
    # (identity = the CardDef object -- hand is DEFERRED; zone "hand" -> all-zero
    # zone one-hot since hand is not a public ZONE) so the pointer can address it
    # -- which is what lets both the
    # graveyard and the hand cross-player picks be pointer-scored with no
    # whole-league fixed "Choose: X" rows. Graveyard/combined-graveyard picks
    # (Relic, Pulse) add nothing here: their cards are already tokenized above.
    pending = state.pending_resolution
    if pending is not None and pending["kind"] == "choose_graveyard_card":
        hand_owner = next((i for i, pl in enumerate(state.players) if pending["graveyard"] is pl.hand), None)
        if hand_owner is not None:
            for card_def in pending["graveyard"]:
                # DEFERRED hand: still CardDefs (interned), so identity = the CardDef.
                # Two same-named nonland hand cards are indistinguishable until hand
                # instances land -- acceptable, no pool card reveals such a pair.
                tokens.append((vocab.index(card_def.name),
                                _token_row(card_def.name, "hand", hand_owner == my_seat_idx, vocab),
                                card_def))
    return tokens
