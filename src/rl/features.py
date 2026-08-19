"""Per-card, per-token feature extraction for the attention-based observation
encoding: full per-card fidelity, not a fixed per-(name, slot) observation
scheme. Two independent pieces per token, deliberately kept separate:

1. A card VOCABULARY INDEX (this module's CardVocab) -- looked up by the
   consuming network into a learned nn.Embedding, so the network can tell
   two cards with identical stats apart. Not computed here; this module only
   assigns stable, deterministic indices.
2. A STATIC structured feature vector (mana cost, card type, base power/
   toughness, keywords, plus what the card DOES -- mana production, effect
   capabilities, the decision kinds it creates, subtypes) -- deterministic,
   no learning, always available even for a card the embedding table has
   never trained on. This is what "Learning With Generalised Card
   Representations for Magic: The Gathering" (arxiv 2407.05879) found
   actually drives generalization to unseen cards; the identity embedding
   alone does not.

   Everything past the printed-stats block is DERIVED from EFFECT_REGISTRY
   and CardDef.extra rather than hand-authored per card, so a new card is
   covered the moment it is registered -- no parallel table to keep in sync.
   Before those blocks existed the structured half collapsed 141 cards onto
   78 distinct vectors (all 26 lands shared ONE vector, as did
   Lightning Bolt/Galvanic Blast/Lava Dart); it is now 131 of 141, and every
   remaining collision is a genuine near-functional-duplicate pair
   (Llanowar/Fyndhorn Elves, Gladecover Scout/Slippery Bogle).

   ponytail: derived presence/absence is the CHEAP end of this. The richer
   version is a hand-authored semantic vector per card -- effect class
   (damage/draw/tutor/sacrifice/bounce), magnitude, and what it targets --
   which would separate the residual pairs and give the encoder a real
   notion of "these two cards do the same KIND of thing". Add it when the
   residual pairs are shown to cost real play strength; it is 141 hand-
   written entries that rot as the catalog changes, which is exactly why it
   is not the starting point.

Dynamic per-instance state (tapped, damage, blocked-as-attacker, currently
targeted by something on the stack, etc.) is a separate, per-token concern --
see _token_row below, combined with the static vector into one per-token
feature row by build_token_set (the tokenizer), which also assigns each
token's zone/side and (for battlefield/graveyard/stack tokens) its pointer-
addressable identity.
"""

import json
import os

import numpy as np

import game

# Full keyword vocabulary across every card in the game (scanned from the
# registry itself, not hand-maintained -- see this module's own self-check
# for how it's derived) rather than hand-listing keywords, so a keyword
# added to a future card is automatically covered without editing this file.
KEYWORD_VOCAB = tuple(sorted({
    kw for spec in game.EFFECT_REGISTRY.values() if "keywords" in spec for kw in spec["keywords"]
}))

CARD_TYPES = tuple(game.CardType)

# --- derived card-behavior vocabularies (see this module's docstring) ------
# All three are scanned from the registry/catalog rather than hand-listed, so
# adding a card that introduces a new capability, decision kind or creature
# type widens the feature vector automatically. That DOES change
# STATIC_FEATURE_DIM -- i.e. adding such a card invalidates existing
# checkpoints, exactly as adding a new keyword already did.

# Every spec key any card's registry entry carries ("mana", "etb_trigger",
# "flashback", "cost_reduction", ...) as a presence multi-hot: a coarse but
# free description of what machinery a card has. Keys held by a single card
# are kept rather than thresholded away -- they cost one dim, and a threshold
# would be one more constant to justify and to drift.
SPEC_KEY_VOCAB = tuple(sorted({key for spec in game.EFFECT_REGISTRY.values() for key in spec}))

# The pending-resolution kinds a card can create -- its BEHAVIORAL signature.
# "choose_any_target" reads as removal/burn, "search_fetch" as a tutor,
# "discard" as disruption; cards that play alike share kinds even when their
# printed stats do not.
PENDING_KIND_VOCAB = tuple(sorted({
    kind for spec in game.EFFECT_REGISTRY.values() for kind in spec.get("pending_kinds", ())
}))

# Creature/land subtypes. Load-bearing, not decoration: Priest of Titania taps
# for {G} per ELF, so the elves deck's core engine is invisible without this.
SUBTYPE_VOCAB = tuple(sorted({
    subtype for card_def in game.CARD_DEFS.values() for subtype in card_def.extra.get("subtypes", ())
}))

# CardDef.extra flags that change how a card plays. "artifact" is the
# load-bearing one -- affinity_reduction counts artifacts (artifact LANDS
# included), so without it Island and Seat of the Synod were the same card to
# the encoder and the agent could not see the count driving its own cost
# reductions. The remaining extra keys are per-card ability COSTS
# (sac_ability_cost, cycling_cost, ...), already covered as registry spec
# keys, so they are not repeated here.
EXTRA_FLAG_VOCAB = ("artifact", "basic", "defender", "indestructible", "devoid")

# Mana specs whose AMOUNT depends on the board rather than being fixed
# ("count"/"count_all" scale with a creature count, "tron" with how many Urza
# pieces are assembled) -- one bit, since the amount itself is a live game-
# state fact the dynamic half cannot read off a CardDef anyway.
_BOARD_SCALED_MANA = ("count", "count_all", "tron")

# Fixed-cap-then-normalize idiom: clamp a raw value to a cap, then divide to
# [0,1], rather than inventing a new normalization convention per feature.
MANA_PIP_CAP = 6
POWER_TOUGHNESS_CAP = 10

MANA_FEATURE_DIM = 1 + len(game.POOL_COLORS) + 3
STATIC_FEATURE_DIM = (
    len(game.POOL_COLORS) + 1 + len(CARD_TYPES) + 2 + len(KEYWORD_VOCAB)
    + MANA_FEATURE_DIM + len(SPEC_KEY_VOCAB) + len(PENDING_KIND_VOCAB)
    + len(EXTRA_FLAG_VOCAB) + len(SUBTYPE_VOCAB)
)


def _mana_features(spec):
    """What mana this card's own mana ability produces, from the registry's
    `mana` spec -- produces-mana at all, which colors, whether the AMOUNT is
    board-scaled, whether it enters tapped, whether it can filter.

    The color lives at a different place per spec kind, so each is read
    explicitly rather than by a shared index: ("fixed", "W") and
    ("count", "G", predicate) both name ONE color, ("flexible", {...}) and
    ("fixed_multi", ("B","R")) name a set, and ("tron",) carries no payload
    at all (the Urza lands produce {C}). Reading `count`/`count_all` as
    colorless would be wrong in a way nothing would catch -- Priest of
    Titania and Overgrown Battlement both tap for {G}."""
    mana = spec.get("mana")
    colors = ()
    if mana is not None:
        kind = mana[0]
        if kind in ("fixed", "count", "count_all"):
            colors = (mana[1],)
        elif kind in ("flexible", "fixed_multi"):
            colors = tuple(mana[1])
        else:  # "tron" -- no payload; the Urza lands produce colorless
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
    decklist AND every token CardDef a training run needs to recognize.
    Index 0 is reserved as a padding/unknown sentinel (never assigned to a
    real card) so a padded token slot's embedding lookup is always valid
    and distinguishable from every real card.

    token_card_defs: real CardDef objects (e.g. game.BLOOD_TOKEN_CARD_DEF),
    same objects the runner's own TOKEN_CARD_DEFS_BY_NAME resolves configs
    through -- required because a token (Blood, Robot, ...) can appear on
    the battlefield mid-game but is NEVER a game.CARD_DEFS entry. Omitting a
    token from token_card_defs would leave it with no observation
    representation at all -- a real, avoidable fidelity loss this token
    encoding exists to avoid."""

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
        vocab_path gives fresh, non-persisted behavior (this call's decklists
        only), the default for every non-league caller/test."""
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
    entries and token CardDefs uniformly).

    Two halves. PRINTED STATS: mana cost (per-color pip counts + generic,
    normalized), card type (one-hot), base power/toughness (normalized, 0 for
    non-creatures), keywords (multi-hot). Then WHAT THE CARD DOES, derived
    from its registry entry and CardDef.extra: mana production
    (_mana_features), effect-capability multi-hot (SPEC_KEY_VOCAB), the
    decision kinds it creates (PENDING_KIND_VOCAB), extra flags
    (EXTRA_FLAG_VOCAB) and subtypes (SUBTYPE_VOCAB).

    Same for every copy of a card and every game state; never touches game
    state, so cacheable per name (see _STATIC_FEATURE_CACHE)."""
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


_STATIC_FEATURE_CACHE = {}  # name -> static_card_features(vocab.card_def(name)), same "keyed by content, computed once" pattern as drl_env._CARD_LOOKUP_CACHE


def cached_static_card_features(name, vocab):
    cached = _STATIC_FEATURE_CACHE.get(name)
    if cached is None:
        cached = static_card_features(vocab.card_def(name))
        _STATIC_FEATURE_CACHE[name] = cached
    return cached


# ---------------------------------------------------------------------------
# Tokenization: turn each PUBLIC zone (battlefield/graveyard/stack/exile,
# both seats) PLUS my own hand into a variable-length list of per-token
# feature rows. Own hand is not hidden information from myself, so
# tokenizing it doesn't violate the hidden-information guarantee -- the
# OPPONENT's hand and both libraries still stay hidden (aggregate count
# only). See this module's own docstring for the static/dynamic split;
# DYNAMIC_FEATURE_DIM below is this function's own per-instance half.
# ---------------------------------------------------------------------------

# There used to be a "known_top" zone here: the cards a player had seen placed
# on top of their own library (Brainstorm), persisted across turns and
# re-tokenized every frame. It was removed with the recurrent policy
# (2026-08-17) because it is a COMPUTED, REMEMBERED FACT rather than something
# observed -- the agent is meant to learn from what it sees, and it does see
# the placement: those cards leave its own tokenized hand one at a time. The
# sequence carries the information even though no single frame does, which is
# exactly what the GRU is there to integrate.
# "revealed": cards a pending resolution is currently HOLDING -- looked at by
# the deciding player but sitting in no real zone while the decision runs.
# begin_scry_surveil does `del state.library[:n]` and parks the cards in
# pending["remaining"]; begin_tuck_to_library holds the bounced card in
# pending["tuck_card"]. Neither is in a library, hand, graveyard or on the
# battlefield, and neither action row names the card ("keep"/"dispose",
# "Tuck: 2nd from top"/"Tuck: bottom"), so before this zone existed the agent
# chose BLIND -- it could not see what it was bottoming or where it was
# tucking. Every other pending exposes its cards by name through the action
# mask (search_fetch, ponder, discard, choose_graveyard_card), which is why
# only these two were affected. Tokenized for the DECIDING seat only (see
# build_token_set's own active_idx gate), so this reveals nothing the
# opponent's own scry would reveal to us.
ZONES = ("battlefield", "graveyard", "stack", "exile", "hand", "revealed")
# Counter kinds any card in the pool can put on a permanent. Hand-listed,
# unlike the derived vocabularies above: counters are set by card CODE
# (`permanent.counters["+1/+1"] = ...` inside a resolve lambda), so there is
# nothing to scan. _counter_features carries a catch-all bit for any kind
# absent from this tuple, so introducing one that nobody adds here degrades to
# "this permanent has some counter" rather than becoming silently invisible.
#
# +1/+1 and -0/-1 already reach the network through effective power/toughness;
# they are ALSO counted here because the counters are separately visible facts
# a real player uses (a 1/1 with two +1/+1 counters and a vanilla 3/3 answer
# bounce, copy and -X/-X effects differently). "charge" and "lifelink" reach
# the network through no other path at all.
COUNTER_VOCAB = ("+1/+1", "-0/-1", "charge", "lifelink")
COUNTER_CAP = 8  # covers Everflowing Chalice's animate threshold of 7 charge counters
AURA_COUNT_CAP = 3

# cost_reduction_delta (1, own-hand tokens only -- see _token_row), untapped,
# tapped, effective_power, effective_toughness, blocked_as_attacker,
# committed_as_blocker, is_attacking, summoning_sick, auras_attached,
# is_attached, targeted_by_mine, targeted_by_theirs, the three "revealed"
# disposition bits (decision_focus / revealed_kept / revealed_disposed),
# counters (len(COUNTER_VOCAB) + 1 catch-all), zone one-hot (len(ZONES)), side
# flag (1) -- see _token_row's own inline comments for what each slot means.
HAND_FEATURE_DIM = 1
DYNAMIC_FEATURE_DIM = HAND_FEATURE_DIM + 12 + 3 + len(COUNTER_VOCAB) + 1 + len(ZONES) + 1

# How many slots sit AFTER the targeted_by_mine/targeted_by_theirs pair: the
# three revealed-disposition bits, the counters block, the zone one-hot and the
# side flag. Exported because two test modules address the targeting bits (and
# the per-permanent bits before them) by counting back from the row's END, and
# that offset silently shifted every time a block was inserted ahead of it --
# three times in one session. Anything inserted before the targeting pair must
# leave this alone; anything inserted after it must be added here.
SLOTS_AFTER_TARGETING = 3 + len(COUNTER_VOCAB) + 1 + len(ZONES) + 1


def _counter_features(permanent):
    """Per-counter-kind counts (normalized) plus one catch-all bit for a kind
    outside COUNTER_VOCAB -- see that tuple's own comment. All zero for a
    non-battlefield token, which has no counters."""
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
    nothing in this pool ever targets (exile, revealed hand).

    cost_reduction_delta: (printed_generic - effective_generic) / MANA_PIP_CAP
    for an own-hand token whose card has a live registry "cost_reduction"
    spec (drl_env._effective_cast_cost, computed by the caller -- see
    rl.agent._hand_cost_reduction_deltas) -- zero for every other token.
    That helper reduces ONLY generic pips, never colored ones, so this one
    number plus the static block's printed cost fully reconstructs a card's
    TRUE current cost (affinity, Tolarian Terror's graveyard count, Deem
    Inferior's cards-drawn count, ...) instead of always showing the
    printed, possibly-stale one.

    decision_focus/revealed_kept/revealed_disposed: only ever set on a
    "revealed"-zone token (see ZONES). A scry/surveil reveals n cards and then
    walks them one at a time, so the agent has to be able to tell WHICH card
    the current keep/dispose applies to (decision_focus) apart from the ones
    it has already sorted (revealed_kept / revealed_disposed) and the ones it
    has seen but not yet reached (all three False). Mutually exclusive by
    construction. A tuck_position card is always the focus -- there is only
    one."""
    row = list(cached_static_card_features(name, vocab))
    row.append(cost_reduction_delta)
    untapped = tapped = eff_power = eff_toughness = blocked_attacker = committed_blocker = 0.0
    attacking = sick = auras_attached = is_attached = 0.0
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
        # is_attacking is NOT implied by blocked_as_attacker above: an
        # UNBLOCKED attacker is absent from blocked_by entirely (combat's own
        # `unblocked = [p for p in attackers if p not in blocked_by]`), so
        # without this bit the creature about to deal lethal damage was
        # feature-identical to one sitting at home. attackers is keyed by the
        # ATTACKING player's own PlayerState, so it reads off owner_idx --
        # the permanent's true seat -- not the perspective seat.
        attacking = 1.0 if permanent in state.players[owner_idx].attackers else 0.0
        # Gates both attacking (CR 302.6) and tapping for mana (Llanowar
        # Elves), so it is not combat-only information.
        sick = 1.0 if permanent.summoning_sick else 0.0
        # How many Auras this permanent is carrying / whether this permanent
        # IS an attached Aura. ponytail: two scalars, not the real relational
        # edge -- the network learns "killing this creature is a 2-for-1" and
        # "this Aura is live", but not WHICH creature a given Aura is on.
        # Encoding the edge itself needs a token-to-token pointer the
        # SetTransformer has no channel for today; add that when a card makes
        # the distinction matter (nothing in this pool moves an Aura).
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
    # Returned as a float32 ARRAY, not the Python list it was built as. The row
    # is built by list concatenation because that is the readable way to
    # assemble it, but it is stored in a RolloutBuffer and then PICKLED to the
    # parent process once per deck-round. As a list it is TOKEN_FEATURE_DIM
    # separate float objects; measured 2026-08-19, the parent was materialising
    # ~16.7 MILLION ~32-byte objects per deck-round in Connection.recv's
    # unpickle path, and transiently holding 9-13GB. One array per token
    # collapses that by ~TOKEN_FEATURE_DIM.
    #
    # float32 (not float64) because pad_token_batch's features buffer is
    # float32 already, so this only moves that same narrowing earlier -- the
    # values the network sees are unchanged.
    return np.asarray(row, dtype=np.float32)


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


def build_token_set(state, my_seat_idx, vocab, include_rows=True, hand_cost_reduction=None):
    """Every public-zone card for BOTH seats, plus MY OWN hand, as a flat
    list of (vocab_index, feature_row, identity) triples -- one shared token
    set, side-flagged rather than two separately-encoded halves, so a joint
    Set Transformer can let tokens from both sides attend to each other
    (relative valuations depend on cross-side context, e.g. an attacker's
    real threat level depends on what can block it). Order within the
    returned list is NOT meaningful (a permutation-invariant encoder
    consumes it) but IS deterministic given the same state, for
    reproducibility.

    include_rows=False skips _token_row entirely (feature_row is None in every
    triple) -- identities (and vocab_index) are still real and complete, only
    the expensive per-token dynamic computation (permanent_power/toughness,
    blocked-by set construction) is dropped. For rl.agent._build_decision's
    forced-decision check: identities are all it needs to determine legality
    (pointer_legal_mask) and a forced decision never runs the network, so the
    feature rows would otherwise be computed and immediately discarded unused.

    hand_cost_reduction: optional {card name: cost_reduction_delta} (see
    _token_row) for MY hand only -- rl.agent._hand_cost_reduction_deltas
    builds it, only when a forward pass will actually run (the cheap
    include_rows=False pass never needs it, its hand rows are None anyway).
    Defaults to 0.0 for every name absent from the dict, so a caller that
    doesn't pass it at all (every existing non-agent caller, e.g. this
    module's own tests) just gets a card's printed cost with no reduction.

    identity: the live Permanent for a battlefield token, the exact CardInstance
    for a graveyard card (Permanent subclasses CardInstance, so battlefield vs
    graveyard is told apart by isinstance(., Permanent) first), the raw stack-
    entry dict for a stack token, the CardDef for a revealed OPPONENT hand
    card (DEFERRED -- hand still holds CardDefs; see the hand-reveal block
    below), else None (exile, and every one of MY OWN hand tokens -- nothing
    legitimately targets my own hand through the pointer head, and a
    CardDef's identity is shared/interned across both players, so giving my
    hand tokens a real CardDef identity would risk a Mesmeric-Fiend-style
    reveal of the OPPONENT's hand accidentally matching one of MY identically-
    named hand cards by object identity in rl.action_bridge.pointer_legal_
    mask's id()-keyed _ID_MATCHED_KINDS branch). The pointer-network action
    head (rl.deck) matches a legal target back to "which row of this token
    batch is that" via this field -- a Permanent for the four battlefield
    targeting kinds (Attack, Assign Blocker, Choose target, Choose
    opponent's), the CardInstance/CardDef object for choose_graveyard_card,
    and the stack-entry dict for choose_stack_target (matched by OBJECT
    IDENTITY in every case -- id()-keyed for choose_graveyard_card/
    choose_cast_copy/choose_stack_target specifically, since a stack entry is
    an unhashable dict, see rl.action_bridge -- so two same-named graveyard
    copies or simultaneous same-named spells are each individually
    addressable, and an opponent's graveyard/stack entry is reachable, which
    is why no per-name "Choose: X" fixed rows exist for either.

    Every token also carries two dynamic targeted-by-mine/targeted-by-theirs
    bits (see _token_row, _stack_target_map) reflecting whatever the CURRENT
    contents of state.stack declare as their targets -- a battlefield
    permanent, a graveyard card, or a stack entry itself (Counterspell
    targets a spell). A player-targeted burn spell has no token to carry that
    bit on; rl.agent._scalar_features surfaces it as a scalar instead. My own
    hand tokens never carry either bit (True): nothing in this pool targets a
    card sitting in hand.

    The OPPONENT's hand and BOTH libraries are deliberately excluded from
    this token set -- those stay hidden here, per the "only hand/library
    CONTENTS are hidden" rule (violating that would leak hidden information
    the rest of this engine carefully protects). Their aggregate SIZE is not
    hidden in real Magic (either player can count a library or a hand) and is
    surfaced instead as a scalar, not a token -- rl.agent._scalar_features,
    not here. Two faithful exceptions to the opponent-hand-hiding rule: a
    hand a card's own effect reveals (Mesmeric Fiend) is tokenized only for
    the duration of that choose_graveyard_card pick, exactly what the real
    reveal shows the caster (see the hand-reveal block at the end of this
    function); and MY OWN hand, tokenized unconditionally above, which was
    never hidden information from myself to begin with."""
    opponent_seat_idx = 1 - my_seat_idx
    # Scan taken once and reused for every token built below -- see
    # game.enchanting_by_target's own docstring for the shared caution about
    # why this snapshot is safe today.
    enchanting_by_target = game.enchanting_by_target(state) if include_rows else {}

    obj_controllers, _player_controllers = _stack_target_map(state) if include_rows else ({}, None)  # player half is rl.agent._scalar_features's job; skip entirely when rows aren't needed

    def _targeted(obj):
        controllers = obj_controllers.get(id(obj), ())
        return my_seat_idx in controllers, opponent_seat_idx in controllers

    # The hand-reveal block below already tokenizes MY hand (with real,
    # pointer-addressable CardDef identities) when a choose_graveyard_card
    # pending is over it -- the own-hand loop below must skip in that one
    # case, or my hand gets double-tokenized (once inert here, once
    # addressable there).
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
            # identity = the exact CardInstance, so two same-named graveyard cards
            # are DISTINCT pointer targets (and a flickered/returned card, being a
            # new instance, is a new token) -- MTG 400.7.
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
            # MY OWN hand: never hidden from myself, so tokenized
            # unconditionally (unlike the opponent's, which stays hidden
            # except the Mesmeric-Fiend reveal branch below). identity=None
            # throughout -- see this function's own docstring for why (the
            # shared/interned CardDef object risk).
            for card_def in player.hand:
                if not include_rows:
                    tokens.append((vocab.index(card_def.name), None, None))
                    continue
                delta = (hand_cost_reduction or {}).get(card_def.name, 0.0)
                row = _token_row(card_def.name, "hand", True, vocab, cost_reduction_delta=delta)
                tokens.append((vocab.index(card_def.name), row, None))
    for entry in state.stack:
        controller = entry["controller"]
        for kind, obj in entry.get("targets", ()):
            if kind == "player":
                player_controllers[obj].add(controller)
            else:
                obj_controllers.setdefault(id(obj), set()).add(controller)
    return obj_controllers, player_controllers


def build_token_set(state, my_seat_idx, vocab, include_rows=True, hand_cost_reduction=None):
    """Every public-zone card for BOTH seats, plus MY OWN hand, as a flat
    list of (vocab_index, feature_row, identity) triples -- one shared token
    set, side-flagged rather than two separately-encoded halves, so a joint
    Set Transformer can let tokens from both sides attend to each other
    (relative valuations depend on cross-side context, e.g. an attacker's
    real threat level depends on what can block it). Order within the
    returned list is NOT meaningful (a permutation-invariant encoder
    consumes it) but IS deterministic given the same state, for
    reproducibility.

    include_rows=False skips _token_row entirely (feature_row is None in every
    triple) -- identities (and vocab_index) are still real and complete, only
    the expensive per-token dynamic computation (permanent_power/toughness,
    blocked-by set construction) is dropped. For rl.agent._build_decision's
    forced-decision check: identities are all it needs to determine legality
    (pointer_legal_mask) and a forced decision never runs the network, so the
    feature rows would otherwise be computed and immediately discarded unused.

    hand_cost_reduction: optional {card name: cost_reduction_delta} (see
    _token_row) for MY hand only -- rl.agent._hand_cost_reduction_deltas
    builds it, only when a forward pass will actually run (the cheap
    include_rows=False pass never needs it, its hand rows are None anyway).
    Defaults to 0.0 for every name absent from the dict, so a caller that
    doesn't pass it at all (every existing non-agent caller, e.g. this
    module's own tests) just gets a card's printed cost with no reduction.

    identity: the live Permanent for a battlefield token, the exact CardInstance
    for a graveyard card (Permanent subclasses CardInstance, so battlefield vs
    graveyard is told apart by isinstance(., Permanent) first), the raw stack-
    entry dict for a stack token, the CardDef for a revealed OPPONENT hand
    card (DEFERRED -- hand still holds CardDefs; see the hand-reveal block
    below), else None (exile, and every one of MY OWN hand tokens -- nothing
    legitimately targets my own hand through the pointer head, and a
    CardDef's identity is shared/interned across both players, so giving my
    hand tokens a real CardDef identity would risk a Mesmeric-Fiend-style
    reveal of the OPPONENT's hand accidentally matching one of MY identically-
    named hand cards by object identity in rl.action_bridge.pointer_legal_
    mask's id()-keyed _ID_MATCHED_KINDS branch). The pointer-network action
    head (rl.deck) matches a legal target back to "which row of this token
    batch is that" via this field -- a Permanent for the four battlefield
    targeting kinds (Attack, Assign Blocker, Choose target, Choose
    opponent's), the CardInstance/CardDef object for choose_graveyard_card,
    and the stack-entry dict for choose_stack_target (matched by OBJECT
    IDENTITY in every case -- id()-keyed for choose_graveyard_card/
    choose_cast_copy/choose_stack_target specifically, since a stack entry is
    an unhashable dict, see rl.action_bridge -- so two same-named graveyard
    copies or simultaneous same-named spells are each individually
    addressable, and an opponent's graveyard/stack entry is reachable, which
    is why no per-name "Choose: X" fixed rows exist for either.

    Every token also carries two dynamic targeted-by-mine/targeted-by-theirs
    bits (see _token_row, _stack_target_map) reflecting whatever the CURRENT
    contents of state.stack declare as their targets -- a battlefield
    permanent, a graveyard card, or a stack entry itself (Counterspell
    targets a spell). A player-targeted burn spell has no token to carry that
    bit on; rl.agent._scalar_features surfaces it as a scalar instead. My own
    hand tokens never carry either bit (True): nothing in this pool targets a
    card sitting in hand.

    The OPPONENT's hand and BOTH libraries are deliberately excluded from
    this token set -- those stay hidden here, per the "only hand/library
    CONTENTS are hidden" rule (violating that would leak hidden information
    the rest of this engine carefully protects). Their aggregate SIZE is not
    hidden in real Magic (either player can count a library or a hand) and is
    surfaced instead as a scalar, not a token -- rl.agent._scalar_features,
    not here. Two faithful exceptions to the opponent-hand-hiding rule: a
    hand a card's own effect reveals (Mesmeric Fiend) is tokenized only for
    the duration of that choose_graveyard_card pick, exactly what the real
    reveal shows the caster (see the hand-reveal block at the end of this
    function); and MY OWN hand, tokenized unconditionally above, which was
    never hidden information from myself to begin with."""
    opponent_seat_idx = 1 - my_seat_idx
    # Scan taken once and reused for every token built below -- see
    # game.enchanting_by_target's own docstring for the shared caution about
    # why this snapshot is safe today.
    enchanting_by_target = game.enchanting_by_target(state) if include_rows else {}

    obj_controllers, _player_controllers = _stack_target_map(state) if include_rows else ({}, None)  # player half is rl.agent._scalar_features's job; skip entirely when rows aren't needed

    def _targeted(obj):
        controllers = obj_controllers.get(id(obj), ())
        return my_seat_idx in controllers, opponent_seat_idx in controllers

    # The hand-reveal block below already tokenizes MY hand (with real,
    # pointer-addressable CardDef identities) when a choose_graveyard_card
    # pending is over it -- the own-hand loop below must skip in that one
    # case, or my hand gets double-tokenized (once inert here, once
    # addressable there).
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
            # identity = the exact CardInstance, so two same-named graveyard cards
            # are DISTINCT pointer targets (and a flickered/returned card, being a
            # new instance, is a new token) -- MTG 400.7.
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
            # MY OWN hand: never hidden from myself, so tokenized
            # unconditionally (unlike the opponent's, which stays hidden
            # except the Mesmeric-Fiend reveal branch below). identity=None
            # throughout -- see this function's own docstring for why (the
            # shared/interned CardDef object risk).
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
                        entry))  # pointer-addressable: choose_stack_target (Counterspell/Dispel/Spell Pierce) picks this exact entry

    # Faithful hand reveal: choose_graveyard_card is a generic pick, and
    # Mesmeric Fiend reuses it to exile a nonland card from the OPPONENT's hand
    # (black_cards.py passes graveyard=<a player's hand>). Real MTG reveals that
    # hand to the caster, so when the pick is over a player's hand, tokenize it
    # (identity = the CardDef object -- hand is DEFERRED; a real "hand" zone
    # one-hot now that ZONES includes it) so the pointer can address it -- which
    # is what lets both the
    # graveyard and the hand cross-player picks be pointer-scored with no
    # whole-league fixed "Choose: X" rows. Graveyard/combined-graveyard picks
    # (Relic, Pulse) add nothing here: their cards are already tokenized above.
    # If the pick is over MY OWN hand, the own-hand loop above already skipped
    # it (pending_over_my_hand) specifically so this is the only place it gets
    # tokenized then -- with a real, pointer-addressable CardDef identity
    # instead of the inert None the own-hand loop would have given it.
    # Cards a pending resolution is HOLDING outside every real zone -- see
    # ZONES' own "revealed" comment for why these were invisible and which
    # decisions that made blind. Gated on active_idx: scry/surveil/ponder all
    # read state.library (an _active_player_property) so the holder is always
    # the active player, and tuck_position flips active_idx to the card's
    # owner before opening. Building the OTHER seat's observation therefore
    # shows none of this, which is the hidden-information guarantee.
    pending = state.pending_resolution
    if pending is not None and state.active_idx == my_seat_idx:
        # (cards, focus, kept, disposed) per pending kind. ponder has no focus:
        # it places by NAME, any remaining card, so no single one is "current".
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
                # identity=None, same reasoning as own-hand tokens: these are
                # interned library CardDefs and nothing points at them (the
                # actions are keep/dispose and Tuck: <position>, not pointers).
                row = _token_row(card_def.name, "revealed", True, vocab, decision_focus=focus,
                                 revealed_kept=kept, revealed_disposed=disposed) if include_rows else None
                tokens.append((vocab.index(card_def.name), row, None))

    if pending is not None and pending["kind"] == "choose_graveyard_card":
        hand_owner = next((i for i, pl in enumerate(state.players) if pending["graveyard"] is pl.hand), None)
        if hand_owner is not None:
            for card_def in pending["graveyard"]:
                # DEFERRED hand: still CardDefs (interned), so identity = the CardDef.
                # Two same-named nonland hand cards are indistinguishable until hand
                # instances land -- acceptable, no pool card reveals such a pair.
                row = _token_row(card_def.name, "hand", hand_owner == my_seat_idx, vocab) if include_rows else None
                tokens.append((vocab.index(card_def.name), row, card_def))
    return tokens
