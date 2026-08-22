"""Per-card, per-token feature extraction for the attention-based observation
encoding: full per-card fidelity, not a fixed per-(name, slot) scheme. Two
independent pieces per token:

1. A card VOCABULARY INDEX (CardVocab), looked up by the consuming network
   into a learned nn.Embedding, so it can tell two cards with identical
   stats apart.
2. A STATIC structured feature vector (mana cost, card type, base power/
   toughness, keywords, plus what the card DOES -- mana production, effect
   capabilities, decision kinds it creates, subtypes), deterministic, no
   learning, always available even for a card the embedding table has never
   trained on. Everything past the printed-stats block is derived from
   EFFECT_REGISTRY and CardDef.extra, so a new card is covered the moment
   it's registered. Remaining feature collisions are genuine near-
   functional-duplicate pairs (Llanowar/Fyndhorn Elves).

   ponytail: derived presence/absence only. A hand-authored semantic vector
   per card would separate the residual duplicate pairs -- add if that's
   shown to cost real play strength.

Dynamic per-instance state (tapped, damage, blocked-as-attacker, targeted by
something on the stack, etc.) is a separate per-token concern -- see
_token_row, combined with the static vector into one row by build_token_set.
"""

import json
import os

import numpy as np

import game

# Keyword vocabulary, scanned from the registry rather than hand-listed, so a
# keyword added to a future card is covered without editing this file.
KEYWORD_VOCAB = tuple(sorted({
    kw for spec in game.EFFECT_REGISTRY.values() if "keywords" in spec for kw in spec["keywords"]
}))

CARD_TYPES = tuple(game.CardType)

# --- derived card-behavior vocabularies (see module docstring) -------------
# Scanned from the registry/catalog, so a card introducing a new capability,
# decision kind or subtype widens the feature vector automatically -- which
# changes STATIC_FEATURE_DIM and invalidates existing checkpoints.

# Every spec key any card's registry entry carries ("mana", "etb_trigger",
# "flashback", "cost_reduction", ...) as a presence multi-hot.
SPEC_KEY_VOCAB = tuple(sorted({key for spec in game.EFFECT_REGISTRY.values() for key in spec}))

# Pending-resolution kinds a card can create -- its behavioral signature
# ("choose_any_target" reads as removal, "search_fetch" as a tutor, etc.).
PENDING_KIND_VOCAB = tuple(sorted({
    kind for spec in game.EFFECT_REGISTRY.values() for kind in spec.get("pending_kinds", ())
}))

# Creature/land subtypes. Load-bearing: Priest of Titania taps for {G} per
# ELF, so the elves deck's core engine is invisible without this.
SUBTYPE_VOCAB = tuple(sorted({
    subtype for card_def in game.CARD_DEFS.values() for subtype in card_def.extra.get("subtypes", ())
}))

# CardDef.extra flags that change how a card plays. "artifact" is load-
# bearing: affinity_reduction counts artifacts (including artifact lands),
# so without it Island and Seat of the Synod look identical to the encoder.
# Other extra keys are per-card ability costs, already covered as registry
# spec keys.
EXTRA_FLAG_VOCAB = ("artifact", "basic", "defender", "indestructible", "devoid")

# Mana specs whose amount depends on the board rather than being fixed
# ("count"/"count_all" scale with a creature count, "tron" with Urza pieces).
_BOARD_SCALED_MANA = ("count", "count_all", "tron")

MANA_PIP_CAP = 6
POWER_TOUGHNESS_CAP = 10

MANA_FEATURE_DIM = 1 + len(game.POOL_COLORS) + 3
STATIC_FEATURE_DIM = (
    len(game.POOL_COLORS) + 1 + len(CARD_TYPES) + 2 + len(KEYWORD_VOCAB)
    + MANA_FEATURE_DIM + len(SPEC_KEY_VOCAB) + len(PENDING_KIND_VOCAB)
    + len(EXTRA_FLAG_VOCAB) + len(SUBTYPE_VOCAB)
)


def _mana_features(spec):
    """What mana this card's mana ability produces, from the registry's
    `mana` spec: produces-mana at all, which colors, whether the amount is
    board-scaled, whether it enters tapped, whether it can filter.

    The color lives at a different place per spec kind: ("fixed", "W") and
    ("count", "G", predicate) each name one color, ("flexible", {...}) and
    ("fixed_multi", ("B","R")) name a set, ("tron",) carries no payload (Urza
    lands produce {C})."""
    mana = spec.get("mana")
    colors = ()
    if mana is not None:
        kind = mana[0]
        if kind in ("fixed", "count", "count_all"):
            colors = (mana[1],)
        elif kind in ("flexible", "fixed_multi"):
            colors = tuple(mana[1])
        else:  # "tron": colorless
            colors = ("C",)
    return (
        [1.0 if mana is not None else 0.0]
        + [1.0 if color in colors else 0.0 for color in game.POOL_COLORS]
        + [
            1.0 if mana is not None and mana[0] in _BOARD_SCALED_MANA else 0.0,
            1.0 if spec.get("enters_tapped") else 0.0,
            1.0 if "filter_mana" in spec else 0.0,
        ]
    )


class CardVocab:
    """name -> stable integer index, built once from the union of every
    decklist and every token CardDef a training run needs to recognize.
    Index 0 is reserved as a padding/unknown sentinel, never assigned to a
    real card.

    token_card_defs: real CardDef objects (e.g. game.BLOOD_TOKEN_CARD_DEF)
    for tokens (Blood, Robot, ...) that can appear on the battlefield but are
    never a game.CARD_DEFS entry."""

    def __init__(self, decklists, token_card_defs=(), vocab_path=None):
        """vocab_path: optional JSON file persisting name -> index across
        separate runs/deck rosters. Append-only: an index already given to a
        name is never reassigned. New names get the next free indices in
        stable sorted order. vocab.size reflects the full persisted
        vocabulary, not just this call's decklists. Omitting vocab_path
        gives fresh, non-persisted behavior."""
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
        # transformed permanent's card_def swaps to its back face
        # (game.resolution.execute_may_transform), so features must resolve
        # that name too. Register each back CardDef and alias its vocab index
        # to its front face's, so the embedding table size is unaffected and
        # the agent perceives the flipped creature as its front identity plus
        # live power/toughness. Done after the persist write so aliases never
        # enter the persisted vocab.
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
    CardDef, not a name -- see CardVocab.card_def). Two halves: printed
    stats (mana cost, card type, base power/toughness, keywords) then what
    the card does (mana production, effect-capability multi-hot, decision
    kinds, extra flags, subtypes). Same for every copy of a card; never
    touches game state, so cacheable per name (_STATIC_FEATURE_CACHE)."""
    cost = card_def.cast_cost or {}
    spec = game.EFFECT_REGISTRY.get(card_def.effect_id, {})
    out = []
    for color in game.POOL_COLORS:
        out.append(min(cost.get(color, 0), MANA_PIP_CAP) / MANA_PIP_CAP)
    out.append(min(cost.get("generic", 0), MANA_PIP_CAP) / MANA_PIP_CAP)
    for card_type in CARD_TYPES:
        out.append(1.0 if card_def.card_type == card_type else 0.0)
    out.append(min(card_def.extra.get("power", 0), POWER_TOUGHNESS_CAP) / POWER_TOUGHNESS_CAP)
    out.append(min(card_def.extra.get("toughness", 0), POWER_TOUGHNESS_CAP) / POWER_TOUGHNESS_CAP)
    card_keywords = spec.get("keywords", ())
    for kw in KEYWORD_VOCAB:
        out.append(1.0 if kw in card_keywords else 0.0)

    out += _mana_features(spec)
    for key in SPEC_KEY_VOCAB:
        out.append(1.0 if key in spec else 0.0)
    card_kinds = spec.get("pending_kinds", ())
    for kind in PENDING_KIND_VOCAB:
        out.append(1.0 if kind in card_kinds else 0.0)
    for flag in EXTRA_FLAG_VOCAB:
        out.append(1.0 if card_def.extra.get(flag) else 0.0)
    subtypes = card_def.extra.get("subtypes", ())
    for subtype in SUBTYPE_VOCAB:
        out.append(1.0 if subtype in subtypes else 0.0)
    assert len(out) == STATIC_FEATURE_DIM
    return out


_STATIC_FEATURE_CACHE = {}  # name -> static_card_features(vocab.card_def(name))


def cached_static_card_features(name, vocab):
    cached = _STATIC_FEATURE_CACHE.get(name)
    if cached is None:
        cached = static_card_features(vocab.card_def(name))
        _STATIC_FEATURE_CACHE[name] = cached
    return cached


# ---------------------------------------------------------------------------
# Tokenization: turn each public zone (battlefield/graveyard/stack/exile,
# both seats) plus my own hand into a variable-length list of per-token
# feature rows. My own hand is not hidden information from myself; the
# opponent's hand and both libraries stay hidden (aggregate count only). See
# module docstring for the static/dynamic split; DYNAMIC_FEATURE_DIM below
# is this function's per-instance half.
# ---------------------------------------------------------------------------

# "revealed": cards a pending resolution is currently holding -- looked at by
# the deciding player but sitting in no real zone while the decision runs
# (begin_scry_surveil parks them in pending["remaining"]; begin_tuck_to_library
# holds the bounced card in pending["tuck_card"]). Tokenized for the deciding
# seat only (build_token_set's active_idx gate).
ZONES = ("battlefield", "graveyard", "stack", "exile", "hand", "revealed")
# Counter kinds any card in the pool can put on a permanent. Hand-listed
# (counters are set by card code, not scannable). _counter_features carries a
# catch-all bit for any kind absent from this tuple. +1/+1 and -0/-1 are also
# reflected in effective power/toughness but counted here too since the raw
# counters are separately visible. "charge"/"lifelink" reach the network no other way.
COUNTER_VOCAB = ("+1/+1", "-0/-1", "charge", "lifelink")
COUNTER_CAP = 8  # covers Everflowing Chalice's animate threshold of 7 charge counters
AURA_COUNT_CAP = 3

# cost_reduction_delta (1, own-hand tokens only), untapped, tapped,
# effective_power, effective_toughness, blocked_as_attacker,
# committed_as_blocker, is_attacking, summoning_sick, auras_attached,
# is_attached, targeted_by_mine, targeted_by_theirs, three "revealed"
# disposition bits (decision_focus / revealed_kept / revealed_disposed),
# counters (len(COUNTER_VOCAB) + 1 catch-all), zone one-hot, side flag --
# see _token_row for what each slot means.
HAND_FEATURE_DIM = 1
DYNAMIC_FEATURE_DIM = HAND_FEATURE_DIM + 12 + 3 + len(COUNTER_VOCAB) + 1 + len(ZONES) + 1

# Slot count AFTER the targeted_by_mine/targeted_by_theirs pair (revealed-
# disposition bits + counters block + zone one-hot + side flag). Two test
# modules address the targeting bits by counting back from the row's end, so
# anything inserted after the targeting pair must be added here too.
SLOTS_AFTER_TARGETING = 3 + len(COUNTER_VOCAB) + 1 + len(ZONES) + 1


def _counter_features(permanent):
    """Per-counter-kind counts (normalized) plus one catch-all bit for a kind
    outside COUNTER_VOCAB. All zero for a non-battlefield token."""
    counters = {} if permanent is None else permanent.counters
    out = [min(counters.get(kind, 0), COUNTER_CAP) / COUNTER_CAP for kind in COUNTER_VOCAB]
    out.append(1.0 if any(k not in COUNTER_VOCAB and n for k, n in counters.items()) else 0.0)
    return out
TOKEN_FEATURE_DIM = STATIC_FEATURE_DIM + DYNAMIC_FEATURE_DIM

PER_CREATURE_POWER_CAP = 20  # clamp before normalizing to [0,1]; 20 covers this card subset's creatures
PER_CREATURE_TOUGHNESS_CAP = 20


def _token_row(name, zone, is_mine, vocab, permanent=None, owner_idx=None, enchanting_auras=None, state=None,
                targeted_by_mine=False, targeted_by_theirs=False, cost_reduction_delta=0.0,
                decision_focus=False, revealed_kept=False, revealed_disposed=False):
    """One token's full feature row: static card identity/stats + dynamic
    per-instance state (mostly zero outside battlefield).

    owner_idx: this permanent's true seat (0 or 1) -- not derivable from
    is_mine, which is relative to whichever seat's perspective is being
    built; blocked_by lookups need the actual owning seat.

    targeted_by_mine/targeted_by_theirs: is this token's object currently a
    declared target of something on the stack, by the perspective seat /
    its opponent (see _stack_target_map), as declared rather than whether it
    will still resolve. Applies to battlefield permanents, graveyard cards,
    and stack entries (a spell can target another spell). False for zones
    nothing in this pool targets (exile, revealed hand).

    cost_reduction_delta: (printed_generic - effective_generic) / MANA_PIP_CAP
    for an own-hand token with a live registry "cost_reduction" spec
    (rl.decision.agent._hand_cost_reduction_deltas computes it), zero
    otherwise. Reduces only generic pips, so this plus the static block's
    printed cost reconstructs a card's true current cost.

    decision_focus/revealed_kept/revealed_disposed: only set on a
    "revealed"-zone token. A scry/surveil reveals n cards and walks them one
    at a time; these mark which card the current keep/dispose applies to
    vs. already-sorted vs. not-yet-reached. Mutually exclusive."""
    row = list(cached_static_card_features(name, vocab))
    row.append(cost_reduction_delta)
    untapped = tapped = eff_power = eff_toughness = blocked_attacker = committed_blocker = 0.0
    attacking = sick = auras_attached = is_attached = 0.0
    if permanent is not None:
        untapped = 0.0 if permanent.tapped else 1.0
        tapped = 1.0 if permanent.tapped else 0.0
        own_blocked_by = state.players[owner_idx].blocked_by
        # gang-blocking: blocked_by values are lists of blockers; flatten to a flat set.
        other_committed_blockers = {b for bs in state.players[1 - owner_idx].blocked_by.values() for b in bs}
        eff_power = min(game.permanent_power(state, permanent, enchanting_auras=enchanting_auras),
                         PER_CREATURE_POWER_CAP) / PER_CREATURE_POWER_CAP
        remaining_t = max(game.permanent_toughness(state, permanent, enchanting_auras=enchanting_auras)
                           - permanent.damage_marked, 0)
        eff_toughness = min(remaining_t, PER_CREATURE_TOUGHNESS_CAP) / PER_CREATURE_TOUGHNESS_CAP
        blocked_attacker = 1.0 if permanent in own_blocked_by else 0.0
        committed_blocker = 1.0 if permanent in other_committed_blockers else 0.0
        # NOT implied by blocked_as_attacker: an unblocked attacker is absent
        # from blocked_by entirely, so without this bit a creature dealing
        # lethal damage looked identical to one sitting at home. attackers is
        # keyed by the attacking player's PlayerState (owner_idx), not the
        # perspective seat.
        attacking = 1.0 if permanent in state.players[owner_idx].attackers else 0.0
        sick = 1.0 if permanent.summoning_sick else 0.0  # gates attacking (CR 302.6) and tapping for mana
        # ponytail: two scalars (count + attached?), not the real token-to-
        # token edge -- can't say WHICH creature a given Aura is on. Add a
        # pointer edge if a card in this pool ever needs that distinction.
        auras_attached = min(len(enchanting_auras or ()), AURA_COUNT_CAP) / AURA_COUNT_CAP
        is_attached = 1.0 if permanent.flags.get("enchanting") is not None else 0.0
    row += [untapped, tapped, eff_power, eff_toughness, blocked_attacker, committed_blocker,
            attacking, sick, auras_attached, is_attached]
    row += [1.0 if targeted_by_mine else 0.0, 1.0 if targeted_by_theirs else 0.0]
    row += [1.0 if decision_focus else 0.0, 1.0 if revealed_kept else 0.0, 1.0 if revealed_disposed else 0.0]
    row += _counter_features(permanent)
    row += [1.0 if zone == z else 0.0 for z in ZONES]
    row.append(1.0 if is_mine else 0.0)
    assert len(row) == TOKEN_FEATURE_DIM
    # float32 array, not the built-up list: one array per token instead of
    # TOKEN_FEATURE_DIM separate float objects keeps pickling/IPC cheap.
    return np.asarray(row, dtype=np.float32)


def _stack_target_map(state):
    """Every current stack entry's declared targets, split by object vs
    player:

    obj_controllers: id(target_object) -> set of controller seat indices
    targeting that object. Covers permanent, graveyard-card, and stack-entry
    targets uniformly via one id()-keyed lookup.

    player_controllers: player_idx -> set of controller seat indices
    targeting that player. Kept separate since no token exists for "the
    player" as an object -- rl.decision.agent._scalar_features reads this
    half directly.

    Keyed by id(), not the object itself: a raw stack-entry dict is
    unhashable, so id()-as-int is the one representation safe across all
    target kinds."""
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


def build_token_set(state, my_seat_idx, vocab, include_rows=True, hand_cost_reduction=None):
    """Every public-zone card for both seats, plus my own hand, as a flat
    list of (vocab_index, feature_row, identity) triples -- one shared,
    side-flagged token set so a joint Set Transformer can let tokens from
    both sides attend to each other. Order is not meaningful but is
    deterministic given the same state.

    include_rows=False skips _token_row entirely (feature_row is None);
    identities/vocab_index stay real and complete. Used by
    rl.decision.agent._build_decision's forced-decision check, which only
    needs identities for legality and never runs the network.

    hand_cost_reduction: optional {card name: cost_reduction_delta} for my
    hand only. Defaults to 0.0 for any name absent from the dict.

    identity: the live Permanent for a battlefield token, the exact
    CardInstance for a graveyard card, the raw stack-entry dict for a stack
    token, the CardDef for a revealed opponent hand card (see the hand-reveal
    block below), else None (exile, and every own-hand token -- a shared/
    interned CardDef identity there would risk a Mesmeric-Fiend-style reveal
    matching one of my own identically-named hand cards). The pointer-
    network action head (rl.model.deck) matches a legal target back to its
    row via this field, always by object identity (id()-keyed for
    choose_graveyard_card/choose_cast_copy/choose_stack_target, since a
    stack entry is an unhashable dict), so same-named graveyard copies or
    simultaneous same-named spells are each individually addressable.

    Every token also carries dynamic targeted-by-mine/targeted-by-theirs
    bits (_token_row, _stack_target_map). A player-targeted burn spell has
    no token to carry that bit; rl.decision.agent._scalar_features surfaces
    it as a scalar instead. Own-hand tokens never carry either bit.

    The opponent's hand and both libraries are excluded and stay hidden;
    their aggregate size is a scalar instead. Two exceptions: a hand a
    card's effect reveals (Mesmeric Fiend) is tokenized only for the
    duration of that choose_graveyard_card pick, and my own hand, tokenized
    unconditionally above."""
    opponent_seat_idx = 1 - my_seat_idx
    enchanting_by_target = game.enchanting_by_target(state) if include_rows else {}  # scanned once, reused per token

    obj_controllers, _player_controllers = _stack_target_map(state) if include_rows else ({}, None)  # player half handled by rl.decision.agent._scalar_features

    def _targeted(obj):
        controllers = obj_controllers.get(id(obj), ())
        return my_seat_idx in controllers, opponent_seat_idx in controllers

    # The hand-reveal block below tokenizes my hand (with pointer-addressable
    # CardDef identities) when a choose_graveyard_card pending is over it --
    # skip it in the own-hand loop below to avoid double-tokenizing.
    pending_over_my_hand = (
        state.pending_resolution is not None
        and state.pending_resolution["kind"] == "choose_graveyard_card"
        and state.pending_resolution["graveyard"] is state.players[my_seat_idx].hand
    )

    tokens = []
    for seat_idx in (my_seat_idx, opponent_seat_idx):
        is_mine = seat_idx == my_seat_idx
        player = state.players[seat_idx]
        for p in player.battlefield:
            if not include_rows:
                tokens.append((vocab.index(p.card_def.name), None, p))
                continue
            auras = enchanting_by_target.get(id(p), ())
            tm, tt = _targeted(p)
            tokens.append((vocab.index(p.card_def.name),
                            _token_row(p.card_def.name, "battlefield", is_mine, vocab, permanent=p, owner_idx=seat_idx,
                                       enchanting_auras=auras, state=state, targeted_by_mine=tm, targeted_by_theirs=tt),
                            p))
        for inst in player.graveyard:
            # identity = the exact CardInstance, so two same-named graveyard
            # cards are distinct pointer targets (MTG 400.7).
            if not include_rows:
                tokens.append((vocab.index(inst.name), None, inst))
                continue
            tm, tt = _targeted(inst)
            tokens.append((vocab.index(inst.name),
                            _token_row(inst.name, "graveyard", is_mine, vocab, targeted_by_mine=tm, targeted_by_theirs=tt),
                            inst))
        for card_def, _plotted_turn in player.exile:
            row = _token_row(card_def.name, "exile", is_mine, vocab) if include_rows else None
            tokens.append((vocab.index(card_def.name), row, None))
        if is_mine and not pending_over_my_hand:
            # My own hand: never hidden from myself, tokenized unconditionally.
            # identity=None throughout -- see docstring for the shared/interned
            # CardDef object risk.
            for card_def in player.hand:
                if not include_rows:
                    tokens.append((vocab.index(card_def.name), None, None))
                    continue
                delta = (hand_cost_reduction or {}).get(card_def.name, 0.0)
                row = _token_row(card_def.name, "hand", True, vocab, cost_reduction_delta=delta)
                tokens.append((vocab.index(card_def.name), row, None))
    for entry in state.stack:
        is_mine = entry["controller"] == my_seat_idx
        if not include_rows:
            tokens.append((vocab.index(entry["card_def"].name), None, entry))
            continue
        tm, tt = _targeted(entry)  # a spell/ability can itself be targeted, e.g. Counterspell
        tokens.append((vocab.index(entry["card_def"].name),
                        _token_row(entry["card_def"].name, "stack", is_mine, vocab, targeted_by_mine=tm, targeted_by_theirs=tt),
                        entry))  # pointer-addressable: choose_stack_target picks this exact entry

    # Mesmeric Fiend reuses choose_graveyard_card to exile a nonland card
    # from the opponent's hand, and real MTG reveals that hand to the
    # caster -- so tokenize it here with a real CardDef identity (skipped
    # above via pending_over_my_hand if the pick is over my own hand).
    # Cards a pending resolution is holding outside every real zone -- see
    # ZONES' "revealed" comment. Gated on active_idx so the opponent's
    # observation shows none of this.
    pending = state.pending_resolution
    if pending is not None and state.active_idx == my_seat_idx:
        # (cards, focus, kept, disposed) per pending kind. ponder has no focus
        # since it places by name, any remaining card.
        groups = ()
        if pending["kind"] in ("scry", "surveil"):
            remaining = pending["remaining"]
            groups = ((remaining[:1], True, False, False), (remaining[1:], False, False, False),
                      (pending["kept"], False, True, False), (pending["disposed"], False, False, True))
        elif pending["kind"] == "ponder":
            groups = ((pending["remaining"], False, False, False), (pending["ordered"] or (), False, True, False))
        elif pending["kind"] == "tuck_position":
            groups = (((pending["tuck_card"],), True, False, False),)
        for cards, focus, kept, disposed in groups:
            for card_def in cards:
                # identity=None: interned library CardDefs, and the actions
                # here (keep/dispose, Tuck: <position>) aren't pointers.
                row = _token_row(card_def.name, "revealed", True, vocab, decision_focus=focus,
                                 revealed_kept=kept, revealed_disposed=disposed) if include_rows else None
                tokens.append((vocab.index(card_def.name), row, None))

    if pending is not None and pending["kind"] == "choose_graveyard_card":
        hand_owner = next((i for i, pl in enumerate(state.players) if pending["graveyard"] is pl.hand), None)
        if hand_owner is not None:
            for card_def in pending["graveyard"]:
                # Hand cards are still CardDefs (interned), so identity = the
                # CardDef; two same-named nonland hand cards are
                # indistinguishable, acceptable since no pool card reveals such a pair.
                row = _token_row(card_def.name, "hand", hand_owner == my_seat_idx, vocab) if include_rows else None
                tokens.append((vocab.index(card_def.name), row, card_def))
    return tokens
