"""Gymnasium environment adapter wrapping game.py's simulator.

Not one of the 4 independent pieces (see DRL_PLAN.md) -- this is assembly
logic that combines a simulator (game.py, imported and never modified)
with an injected reward function (rewards.py's contract) into a
Gym-compatible interface a DRL model can train against.
"""

import os
import random

import numpy as np
import gymnasium
from gymnasium import spaces

import game
import rewards  # only for resource_quality_components/its two caps -- see _opponent_aggregate_features

# ---------------------------------------------------------------------------
# D2.1 -- Card indexing and observation builder (MULTI_DECK_PLAN.md Phase
# M4f: build_observation takes a decklist explicitly, sized to its
# distinct-card count, instead of a hardcoded 90-dim vector.
# ---------------------------------------------------------------------------

# Every deck needs "none" (nothing pending), "pay_cost" (mana.py is
# universal), and -- since game.effects.state_based.cleanup_step
# (docs/COMBAT_PLAN.md) can discard down to hand size at the end of
# ANY deck's turn, whether or not any of its own cards ever discard --
# "discard" too. Kept as the baseline here, not per-deck, since no deck
# could ever function without them.
BASELINE_PENDING_KINDS = ("none", "pay_cost", "discard")


def _all_pending_kinds(pending_kinds):
    """BASELINE_PENDING_KINDS plus this deck's own extra kinds
    (game.registry.derive_pending_kinds), deduplicated: a deck whose own
    cards independently declare "discard" (Faithless Looting, Highway
    Robbery, Grab the Prize, Melded Moxite) would otherwise double-count
    it now that it's also a baseline kind -- harmless to the one-hot's
    own correctness (either duplicate index reads the same true/false
    answer) but would silently disagree with observation_dim_for's own
    length depending on which deck happens to overlap, which is exactly
    the kind of bug a shared helper (used by both observation_dim_for and
    build_observation) forecloses instead of trusting two independent
    formulas to agree."""
    return BASELINE_PENDING_KINDS + tuple(k for k in pending_kinds if k not in BASELINE_PENDING_KINDS)

# Floating mana pool observation cap: a single overtapping action rarely
# floats more than a handful of pips per color (e.g. Tron's 3 lands online
# producing up to 7 colorless in one turn) -- same fixed-cap-then-normalize
# pattern rewards.resource_quality already uses for available_mana.
POOL_CAP = 8

# Stack-depth observation cap: with no opponent contributing to it, a stack
# in this solitaire engine will rarely hold more than a couple of spells at
# once (each entry is one model action) -- same fixed-cap-then-normalize
# pattern as POOL_CAP. Distinguishes "2 different spells stacked" from "2
# copies of the same spell stacked" (stack_counts alone can't -- both look
# identical there), without a full per-slot encoding of stack order, which
# would either bloat the observation (slots x card_names) or silently
# truncate a deeper stack.
STACK_DEPTH_CAP = 8

# Per-creature power observation cap (docs/COMBAT_PLAN.md's per-slot
# creature block, below): stacking every boggles power-Aura on one
# creature can reach well above a typical vanilla's power, but 20 is
# already generous headroom -- same fixed-cap-then-normalize pattern as
# POOL_CAP/STACK_DEPTH_CAP.
PER_CREATURE_POWER_CAP = 20
# Same reasoning as PER_CREATURE_POWER_CAP -- the Auras that buff toughness
# here are +X/+X (symmetric with power), so the same magnitude/cap applies.
PER_CREATURE_TOUGHNESS_CAP = 20


def observation_dim_for(decklist, pending_kinds):
    """Shared by DeckEnv.__init__ and harness.py's load() so a deck's
    observation dimension is computed identically in both places.
    pending_kinds is that deck's own extra kinds beyond the universal
    baseline (see game.registry.derive_pending_kinds) -- keeps a deck's
    dimension from moving every time an unrelated deck gains a new
    pending-resolution kind. The Phase one-hot (game.turn.Phase) is fixed
    at the same 9 dims for every deck, unlike pending_kinds -- a
    combat_enabled=False deck's phases (game.turn.MINIMAL_PHASES) are a
    subset of the full sequence, not a different enum, so those bits are
    just always 0 for it rather than the dimension itself shrinking.

    The `* 4` block is hand/battlefield-untapped/battlefield-tapped/
    remaining-elsewhere counts (see build_observation); `+ 2` more per-name
    blocks (stack_counts, stack_top's one-hot-plus-none) plus `+ 1` scalar
    (stack_depth) cover state.stack -- see build_observation's own stack
    section for what each holds. `+ 2` more scalars: turn_number and "am I
    the turn player right now" (docs/PRIORITY_PLAN.md item 5 -- state.
    active_idx == state.turn_player_idx, a genuinely new fact once most
    reactive priority windows share pending_kind == "none" with an ordinary
    proactive decision). The final term is the per-(creature name, slot)
    block (docs/COMBAT_PLAN.md's permanent-identity design, extended by
    docs/PRIORITY_PLAN.md item 5) -- 6 values (untapped-and-present/
    tapped-and-present/power/remaining-toughness/blocked-as-attacker/
    committed-as-blocker) per slot, `quantity` slots per creature name, real
    decklist creatures only (a token has no observation representation at
    all, same pre-existing precedent as every other block here)."""
    num_names = len({name for name, *_rest in decklist})
    creature_slot_dim = sum(
        qty * 6 for name, qty, *_rest in decklist if game.CARD_DEFS[name].card_type == game.CardType.CREATURE
    )
    return (num_names * 4 + 2 + len(_all_pending_kinds(pending_kinds))
            + len(game.POOL_COLORS) + len(game.turn.Phase)
            + num_names + (num_names + 1) + 1 + 1
            + creature_slot_dim)


_CARD_LOOKUP_CACHE = {}  # tuple(decklist) -> (card_names, card_copies). Keyed by CONTENT, not id(decklist): a decklist is no longer a fixed module-level constant (game.parse_decklist_file returns a fresh list every call), and id() can be silently reused once a prior short-lived decklist list is garbage-collected -- an identity-keyed cache would then return a completely different, stale decklist's (card_names, card_copies) for it. A tuple of the (name, qty) pairs is hashable and correctly reflects "same decklist contents," as a bonus also giving real cache hits across separately-parsed-but-identical decklists.


def _card_lookup(decklist):
    key = tuple(decklist)
    cached = _CARD_LOOKUP_CACHE.get(key)
    if cached is None:
        cached = (
            sorted({name for name, *_rest in decklist}),
            {name: qty for name, qty, *_rest in decklist},
        )
        _CARD_LOOKUP_CACHE[key] = cached
    return cached


def build_observation(state, decklist, horizon, pending_kinds):
    card_names, card_copies = _card_lookup(decklist)
    all_kinds = _all_pending_kinds(pending_kinds)
    creature_names = [name for name in card_names if game.CARD_DEFS[name].card_type == game.CardType.CREATURE]
    creature_slot_dim = sum(card_copies[name] * 6 for name in creature_names)
    dim = (len(card_names) * 4 + 2 + len(all_kinds) + len(game.POOL_COLORS) + len(game.turn.Phase)
           + len(card_names) + (len(card_names) + 1) + 1 + 1
           + creature_slot_dim)
    obs = np.zeros(dim, dtype=np.float32)

    hand_counts = {name: 0 for name in card_names}
    for card_def in state.hand:
        hand_counts[card_def.name] += 1

    bf_untapped = {name: 0 for name in card_names}
    bf_tapped = {name: 0 for name in card_names}
    for p in state.battlefield:
        if p.card_def.name not in bf_untapped:
            continue  # a token (Blood, Robot, ...) -- never a decklist member, so no observation slot exists for it (hand/graveyard never hold one, only battlefield -- see game/effects/tokens.py's own docstrings)
        if p.tapped:
            bf_tapped[p.card_def.name] += 1
        else:
            bf_untapped[p.card_def.name] += 1

    graveyard_counts = {name: 0 for name in card_names}
    for card_def in state.graveyard:
        graveyard_counts[card_def.name] += 1

    stack_counts = {name: 0 for name in card_names}
    for entry in state.stack:
        name = entry["card_def"].name
        if name in stack_counts:  # a token can never be cast, so never on the stack -- no observation slot needed
            stack_counts[name] += 1

    i = 0
    for name in card_names:
        # A copy already pushed onto the stack (paid for, awaiting
        # resolution -- see push_to_stack/_hand_count_available) is still
        # physically in state.hand (its resolve hasn't removed it yet) but
        # is no longer "available" -- match the action mask's own notion
        # of hand availability rather than double-counting it here and
        # again in the stack block below.
        obs[i] = max(hand_counts[name] - stack_counts[name], 0) / card_copies[name]
        i += 1
    for name in card_names:
        obs[i] = bf_untapped[name] / card_copies[name]
        i += 1
        obs[i] = bf_tapped[name] / card_copies[name]
        i += 1
    for name in card_names:
        remaining = card_copies[name] - hand_counts[name] - bf_untapped[name] - bf_tapped[name] - graveyard_counts[name]
        obs[i] = remaining / card_copies[name]
        i += 1
    obs[i] = state.turn_number / horizon
    i += 1
    obs[i] = 1.0 if state.lands_played_this_turn > 0 else 0.0
    i += 1

    # "Am I the turn player right now" (docs/PRIORITY_PLAN.md item 5) --
    # most reactive priority windows (responding to an opponent's spell,
    # blocking their attack) share pending_kind == "none" with an ordinary
    # proactive decision, so without this the model can't otherwise tell
    # the two apart the way it already can for the one narrower case
    # ("declare_blockers" has its own pending_kind).
    obs[i] = 1.0 if state.active_idx == state.turn_player_idx else 0.0
    i += 1

    # Which kind of pending resolution (if any) is active right now -- the
    # only signal in the observation itself that a decision like "which
    # tap source" or "keep or dispose" is underway; the action mask alone
    # tells the model *what's* legal but not *why* (MaskablePPO's network
    # never sees the mask as an input feature, only the observation).
    pending_kind = state.pending_resolution["kind"] if state.pending_resolution is not None else "none"
    for kind in all_kinds:
        obs[i] = 1.0 if kind == pending_kind else 0.0
        i += 1

    # Floating mana pool, one dim per color -- otherwise the model has
    # legal "Spend X from pool" actions it can't see the reason for at all
    # (the action mask says what's legal, never why).
    for color in game.POOL_COLORS:
        obs[i] = min(state.mana_pool.get(color, 0), POOL_CAP) / POOL_CAP
        i += 1

    # Which phase (game.turn.Phase) the turn is currently in -- same
    # one-hot idiom as pending_kind above, fixed 9 dims for every deck
    # (see observation_dim_for's own docstring). Without this the model
    # can't tell DECLARE_BLOCKERS apart from MAIN1, etc. -- the action
    # mask alone says what's legal, never where in the turn it is.
    for phase in game.turn.Phase:
        obs[i] = 1.0 if phase == state.phase else 0.0
        i += 1

    # state.stack (stack_counts computed above, alongside the other zone
    # tallies): every spell fully paid for but not yet resolved (see
    # game.push_to_stack) -- the model needs to see this to know what a
    # "Pass" would resolve next and what else is still queued up behind it.
    for name in card_names:
        obs[i] = stack_counts[name] / card_copies[name]
        i += 1

    # Top-of-stack one-hot, plus an explicit "none" slot -- same idiom as
    # pending_kind above (BASELINE_PENDING_KINDS' own "none" entry), not an
    # implicit all-zero. The one order-dependent fact that matters: what
    # resolves next if the model passes; stack_counts alone is an
    # unordered multiset and can't say that.
    top_name = state.stack[-1]["card_def"].name if state.stack else None
    for name in card_names:
        obs[i] = 1.0 if name == top_name else 0.0
        i += 1
    obs[i] = 1.0 if top_name is None else 0.0  # "none" slot
    i += 1

    # Stack depth, capped-then-normalized same as the mana pool above --
    # distinguishes "2 different spells stacked" from "2 copies of the same
    # spell stacked" (stack_counts alone can't tell those apart).
    obs[i] = min(len(state.stack), STACK_DEPTH_CAP) / STACK_DEPTH_CAP
    i += 1

    # Per-(creature name, slot) block -- see _creature_slot_block's own
    # docstring for what each of its 6 values means. My own battlefield:
    # owner_idx is state.active_idx, same active-relative convention as
    # every other zone this function reads (state.hand/state.battlefield/
    # ...).
    obs[i:i + creature_slot_dim] = _creature_slot_block(state, state.active_idx, creature_names, card_copies)
    i += creature_slot_dim

    return obs


def _creature_slot_block(state, owner_idx, creature_names, creature_copies):
    """Per-(creature name, slot) block (docs/COMBAT_PLAN.md's permanent-
    identity design, extended by docs/PRIORITY_PLAN.md item 5) for
    state.players[owner_idx]'s own battlefield -- the aggregate
    untapped/tapped counts elsewhere say "how many," never "which specific
    one" -- the model needs the latter to actually use "Attack: <name>
    (slot k)"'s own per-slot addressing (e.g. to tell an Aura-enchanted
    copy's higher power apart from a plain copy of the same name). Real
    decklist creatures only -- a token has no observation representation
    anywhere in this codebase (see hand/battlefield/stack counts
    elsewhere, all keyed by card_names).

    6 values per slot: untapped-and-present, tapped-and-present, power,
    remaining toughness (permanent_toughness - damage_marked, capped/
    normalized same as power), blocked-as-attacker (this permanent's own
    key-presence in state.players[owner_idx].blocked_by -- one of
    owner_idx's own declared attackers that a block just absorbed), and
    committed-as-blocker (this permanent's own value-presence in
    state.players[1 - owner_idx].blocked_by -- the symmetric signal: one
    of owner_idx's OWN creatures already assigned to block one of the
    OTHER side's attackers this combat; blocked_by is always keyed by the
    ATTACKING player's own permanents, so this is the other player's dict,
    not owner_idx's own). Always indexes state.players[owner_idx] directly
    (never the active-relative state.battlefield/state.blocked_by
    proxies) so this works identically whether owner_idx is "my own" seat
    or the opponent's, regardless of which one currently holds priority --
    shared by build_observation (owner_idx = state.active_idx) and
    _opponent_creature_block (owner_idx = the other seat)."""
    battlefield = state.players[owner_idx].battlefield
    own_blocked_by = state.players[owner_idx].blocked_by
    # 1-player mode (DeckEnv) has no "other" player at all to index --
    # blocked_by can never be populated there anyway (declare_blocker_
    # assignment only ever runs in a real 2-player game), so this is
    # always empty in that case, same as own_blocked_by above.
    other_blocked_by_values = state.players[1 - owner_idx].blocked_by.values() if len(state.players) > 1 else ()
    dim = sum(creature_copies[name] for name in creature_names) * 6
    out = np.zeros(dim, dtype=np.float32)
    i = 0
    for name in creature_names:
        by_slot = {p.slot: p for p in battlefield if p.card_def.name == name}
        for slot in range(1, creature_copies[name] + 1):
            p = by_slot.get(slot)
            if p is not None:
                out[i] = 0.0 if p.tapped else 1.0
                out[i + 1] = 1.0 if p.tapped else 0.0
                out[i + 2] = min(game.permanent_power(state, p), PER_CREATURE_POWER_CAP) / PER_CREATURE_POWER_CAP
                remaining = max(game.permanent_toughness(state, p) - p.damage_marked, 0)
                out[i + 3] = min(remaining, PER_CREATURE_TOUGHNESS_CAP) / PER_CREATURE_TOUGHNESS_CAP
                out[i + 4] = 1.0 if p in own_blocked_by else 0.0
                out[i + 5] = 1.0 if p in other_blocked_by_values else 0.0
            i += 6
    return out


# ---------------------------------------------------------------------------
# D2.2 -- Action table (MULTI_DECK_PLAN.md Phase M4e: generated from a
# decklist + game.EFFECT_REGISTRY instead of hand-typed -- this, plus the
# pending-resolution machinery in game.py, is what makes a deck built
# entirely from already-implemented cards need zero new code here.
#
# Categories, in table order:
#   A. Play land: <name>            -- one per distinct land name
#   B. Cast <name>                  -- one per card with a registry "cast" entry
#   C. Activate <name> (<ability>)  -- one per registered activated ability
#   D. Forestcycle <name>           -- one per registry "forestcycle" entry
#   E. Pass
#   F. Choose: <name>               -- shared across every pending-resolution
#      kind that picks a plain card name (paying with a fixed/Tron mana
#      source, search_fetch, ancient_stirrings, and scry/surveil's ordering
#      phase), dispatched by pending_resolution["kind"]
#   G. Choose: <name> as <color>    -- flexible/filter mana sources during
#      a pay_cost resolution specifically (the only kind needing a color)
#   H. Keep / Dispose (scry/surveil)
#   I. Decline (Ancient Stirrings)
#   J. Abandon payment -- cancels a pending pay_cost resolution outright,
#      untapping everything tapped so far. Without this, tapping a
#      flexible/filter source for the wrong color could strand a game
#      with an unpayable remaining cost and zero legal actions -- see
#      game.abandon_pay_cost's docstring.
#   K. Choose target: <name> (slot k) -- exact-(name, slot)-addressed, the
#      "choose_permanent" resolution's own actions (Aura enchant-targets,
#      Crop Rotation's sacrifice cost, land bounce) -- NOT category F,
#      unlike before: two same-named permanents stop being interchangeable
#      the instant an Aura attaches to only one of them, and cast_aura's
#      cast-time-target/resolve-time-fizzle contract depends on knowing
#      exactly which physical permanent was chosen (docs/MULTIPLAYER_GAPS.md
#      "Permanent identity").
#
# spy_combo deck additions: B also covers Winding Way's modal cast (2
# actions, one per mode), Land Grant's free alt-cost, and Dread Return's
# Flashback (cast from the graveyard); C also covers non-mana activated
# abilities (Quirion Ranger); F/H also cover select_to_hand's own
# Keep/Bottom pair and its ordering phase (Lead the Stampede) and an
# optional search's Decline (Gatecreeper Vine) alongside Ancient
# Stirrings'.
# ---------------------------------------------------------------------------

def _cast_speed(card_def, spec):
    """The game.turn.Speed a cast-like action (cast/cast_modes/alt_cast/
    flashback/plot -- each derived independently, once per action, in
    build_action_table) resolves to: an explicit "speed" key in that
    specific spec dict if a card ever needs to override it, else
    Speed.INSTANT for an actual CardType.INSTANT card (its type line
    already implies instant speed -- no per-card tag needed for the
    common case), else Speed.SORCERY -- the default for every creature/
    artifact/enchantment/sorcery/land absent a Flash-like exception, per
    real Magic's own casting-speed rule. Flashback/Plot deliberately have
    no override in this cube today, so they fall through to the same
    CardType-derived answer the card's normal cast would -- correct per
    real Magic (Flashback/Plot follow the same timing as the card
    itself), not just a convenient default."""
    override = spec.get("speed")
    if override is not None:
        return override
    if card_def.card_type == game.CardType.INSTANT:
        return game.turn.Speed.INSTANT
    return game.turn.Speed.SORCERY


def _land_drop_legal(name):
    def legal(state):
        return (
            state.pending_resolution is None
            # Real Magic: playing a land is always sorcery-speed (no
            # per-card override exists in this cube) -- speed_legal's own
            # Speed.SORCERY branch already requires state.active_idx ==
            # state.turn_player_idx (docs/PRIORITY_PLAN.md), so this alone
            # already refuses a land drop offered to the non-turn player
            # during a priority window, with no separate check needed here.
            and game.turn.speed_legal(state, game.turn.Speed.SORCERY)
            and state.lands_played_this_turn == 0
            and any(c.name == name for c in state.hand)
        )
    return legal


def _land_drop_execute(name):
    def execute(state):
        game.play_land_from_hand(state, game.CARD_DEFS[name])
    return execute


def _hand_count_available(state, name):
    """How many copies of `name` in state.hand are actually still castable
    right now. A cast-like resolve function only removes its card from hand
    when it finally RUNS -- which, since push_to_stack (game.effects.stack)
    defers it, can be well after the cost is paid -- so a copy already
    pushed onto state.stack (paid for, awaiting resolution) is still
    physically present in state.hand but must not count as available: an
    action mask that let the model "cast" that same physical copy a second
    time would push a second stack entry referencing it, and crash once
    both entries eventually try to remove it from hand. Sorcery-speed cards
    are already safe from this via speed_legal's own stack-emptiness check
    (nothing sorcery-speed is ever legal again once anything -- even
    itself -- sits unresolved on the stack); this only actually matters for
    Speed.INSTANT cards, which the stack never blocks re-casting of, but is
    correct (a no-op) to apply uniformly rather than special-casing speed
    here too."""
    hand_count = sum(1 for c in state.hand if c.name == name)
    stacked_count = sum(1 for entry in state.stack if entry["card_def"].name == name)
    return hand_count - stacked_count


def _cast_legal(name, extra_legal, speed):
    def legal(state):
        if state.pending_resolution is not None:
            return False
        if not game.turn.speed_legal(state, speed):
            return False
        if _hand_count_available(state, name) <= 0:
            return False
        card_def = game.CARD_DEFS[name]
        if game.plan_payment(state, card_def.cast_cost) is None:
            return False
        return extra_legal is None or extra_legal(state)
    return legal


def _cast_execute(name, resolve):
    def execute(state):
        card_def = game.CARD_DEFS[name]
        # Fires the instant this cast is announced, before mana is even
        # tapped -- matches real timing (a "whenever you cast" trigger
        # goes on the stack immediately, ahead of paying the cost).
        # MADNESS_DECKS_PLAN.md item 11 (Guttersnipe); every cast path
        # (this one, alt_cast, flashback, plot-from-exile below) fires it
        # identically. Deliberately NOT deferred by the spell stack below --
        # a documented pre-existing simplification, out of this feature's
        # scope.
        game.on_cast_trigger(state, card_def)
        # Once mana is fully paid, the spell is "cast" but not yet resolved
        # -- push it onto state.stack (game.push_to_stack) instead of
        # resolving immediately, so the model can respond with another
        # instant-speed action first. Something (a "Pass" -- see
        # game.turn._run_turn_gen) has to actually resolve it later.
        game.begin_pay_cost(state, card_def.cast_cost, on_complete=lambda s: game.push_to_stack(s, card_def, resolve))
    return execute


def _precast_choice_execute(name, resolve):
    """Cast-like execute for a card whose own `resolve` needs to settle
    something -- a real target (cast_aura's "enchant target creature"), or
    an additional cost (cast_crop_rotation's "sacrifice a land") -- BEFORE
    the spell is fully cast, not once it resolves off the stack. Real MTG:
    both targets and additional costs are locked in as part of casting the
    spell, never deferred to resolution; only the spell's own EFFECT waits
    on the stack. Unlike _cast_execute, `resolve` is called directly as
    pay_cost's on_complete and is responsible for its own game.push_to_stack
    call (having already run whatever precast resolution it needs -- see
    cast_aura/cast_crop_rotation's own docstrings for each one's exact
    contract) instead of this function pushing to the stack generically on
    its behalf. Selected via each registry cast/cast_modes spec's own
    "precast_choice": True flag (build_action_table)."""
    def execute(state):
        card_def = game.CARD_DEFS[name]
        game.on_cast_trigger(state, card_def)  # same timing as _cast_execute -- see its own comment
        game.begin_pay_cost(state, card_def.cast_cost, on_complete=lambda s: resolve(s, card_def))
    return execute


def _activate_legal(name, cost_key, speed):
    def legal(state):
        if state.pending_resolution is not None:
            return False
        if not game.turn.speed_legal(state, speed):
            return False
        p = next((p for p in state.battlefield if p.card_def.name == name and not p.tapped), None)
        return p is not None and game.plan_payment(state, p.card_def.extra[cost_key]) is not None
    return legal


def _activate_execute(name, cost_key, resolve):
    def execute(state):
        p = next(p for p in state.battlefield if p.card_def.name == name and not p.tapped)
        cost = p.card_def.extra[cost_key]
        game.begin_pay_cost(state, cost, on_complete=lambda s: resolve(s, p))
    return execute


def _forestcycle_legal(name, cost_key):
    def legal(state):
        if state.pending_resolution is not None:
            return False
        if not any(c.name == name for c in state.hand):
            return False
        card_def = game.CARD_DEFS[name]
        return game.plan_payment(state, card_def.extra[cost_key]) is not None
    return legal


def _forestcycle_execute(name, cost_key, resolve):
    def execute(state):
        card_def = game.CARD_DEFS[name]
        game.begin_pay_cost(state, card_def.extra[cost_key], on_complete=lambda s: resolve(s, card_def))
    return execute


def _pass_legal(state):
    return state.pending_resolution is None


def _pass_execute(state):
    pass  # handled by DeckEnv.step() itself, not via this table


def _choose_name_options(state):
    """Plain (uncolored) 'Choose: X' names currently legal, given whatever
    kind of pending resolution -- if any -- is active. "choose_permanent"
    is NOT handled here -- see _choose_permanent_legal/_choose_permanent_
    execute below: it needs exact (name, slot) addressing (docs/
    MULTIPLAYER_GAPS.md's "Permanent identity"), same as
    "choose_opponent_permanent" already gets, not this generic by-name
    dispatch."""
    pending = state.pending_resolution
    if pending is None:
        return []
    kind = pending["kind"]
    if kind == "pay_cost":
        return [n for n, c, f in _cached_tap_cost_options(state) if c is None and not f]
    if kind == "search_fetch":
        return game.search_fetch_options(state)
    if kind == "choose_graveyard_card":
        return game.choose_graveyard_card_options(state)
    if kind == "sacrifice":
        return game.sacrifice_options(state)
    if kind == "discard":
        return game.discard_options(state)
    if kind == "ancient_stirrings":
        return [n for n in game.ancient_stirrings_options(state) if n != "decline"]
    if kind == "malevolent_rumble":
        return [n for n in game.malevolent_rumble_options(state) if n != "decline"]
    if kind in ("scry", "surveil") and pending["ordered"] is not None:
        return game.scry_surveil_options(state)
    if kind == "select_to_hand" and pending["ordered"] is not None:
        return game.select_to_hand_options(state)  # ordering phase only -- "keep"/"bottom" are their own actions
    if kind == "order_triggers":
        return game.order_triggers_options(state)  # docs/PRIORITY_PLAN.md item 1
    return []


def _choose_name_legal(name):
    def legal(state):
        return name in _choose_name_options(state)
    return legal


def _choose_name_execute(name):
    def execute(state):
        kind = state.pending_resolution["kind"]
        if kind == "pay_cost":
            game.execute_tap_cost_option(state, name, None, False)
        elif kind == "search_fetch":
            game.execute_search_fetch_option(state, name)
        elif kind == "choose_graveyard_card":
            game.execute_choose_graveyard_card_option(state, name)
        elif kind == "sacrifice":
            game.execute_sacrifice_option(state, name)
        elif kind == "discard":
            game.execute_discard_option(state, name)
        elif kind == "ancient_stirrings":
            game.execute_ancient_stirrings_option(state, name)
        elif kind == "malevolent_rumble":
            game.execute_malevolent_rumble_option(state, name)
        elif kind == "select_to_hand":
            game.execute_select_to_hand_option(state, name)  # ordering phase only
        elif kind == "order_triggers":
            game.execute_order_triggers_option(state, name)  # docs/PRIORITY_PLAN.md item 1
        else:  # scry / surveil, ordering phase
            game.execute_scry_surveil_option(state, name)
    return execute


def _choose_name_color_options(state):
    """(name, color) pairs currently legal via tap_cost_options's
    flexible/filter entries -- the only pending-resolution kind that ever
    needs a color qualifier."""
    pending = state.pending_resolution
    if pending is None or pending["kind"] != "pay_cost":
        return []
    return [(n, c) for n, c, _f in _cached_tap_cost_options(state) if c is not None]


def _choose_name_color_legal(name, color):
    def legal(state):
        return (name, color) in _choose_name_color_options(state)
    return legal


def _choose_name_color_execute(name, color):
    def execute(state):
        is_filter = next(f for n, c, f in game.tap_cost_options(state) if n == name and c == color)
        game.execute_tap_cost_option(state, name, color, is_filter)
    return execute


def _attack_legal(name, slot):
    """Legal only during Phase.DECLARE_ATTACKERS, and only for the true
    turn owner (state.active_idx == state.turn_player_idx,
    docs/PRIORITY_PLAN.md) -- declaring an attacker is a turn-based
    special action, not a priority action, so the non-turn player must
    never be allowed to declare one just because state.phase (a single
    shared field describing the TURN's phase) happens to match during
    their own priority window. And only if the specific physical
    permanent occupying this (name, slot) -- docs/COMBAT_PLAN.md's
    permanent-identity design -- is currently attack-eligible
    (game.creature_attack_eligible): untapped, and not summoning sick
    unless it has haste. Attacking stays fully optional: a model can leave
    any subset of eligible creatures back, Pass with zero attackers
    declared is still legal (same as always -- state.attackers simply
    starts, and can stay, empty for this turn)."""
    def legal(state):
        if state.phase is not game.turn.Phase.DECLARE_ATTACKERS:
            return False
        if state.active_idx != state.turn_player_idx:
            return False
        return any(
            p.card_def.name == name and p.slot == slot and game.creature_attack_eligible(state, p)
            for p in state.battlefield
        )
    return legal


def _attack_execute(name, slot):
    """Declares the specific physical permanent occupying this (name,
    slot) as an attacker -- unlike the old arbitrary-pick-by-name
    behavior, this lets a model distinguish an Aura-enchanted copy
    (different effective power) from a plain one of the same name."""
    def execute(state):
        permanent = next(
            p for p in state.battlefield
            if p.card_def.name == name and p.slot == slot and game.creature_attack_eligible(state, p)
        )
        game.declare_attacker(state, permanent)
    return execute


def _choose_permanent_legal(name, slot):
    """The "choose_permanent" resolution's action-table half (Aura
    enchant-targets, Crop Rotation's sacrifice cost, land bounce) -- legal
    only while that kind is pending and (name, slot) is one of its own
    current options. Exact (name, slot) addressing, same reason
    _choose_opponent_permanent_legal below needs it (docs/
    MULTIPLAYER_GAPS.md's "Permanent identity") -- a plain by-name "Choose:
    X" can't tell two same-named permanents apart, and cast_aura's whole
    fizzle-on-invalid-target contract depends on knowing exactly which one
    was chosen."""
    def legal(state):
        pending = state.pending_resolution
        return (
            pending is not None and pending["kind"] == "choose_permanent"
            and (name, slot) in game.choose_permanent_options(state)
        )
    return legal


def _choose_permanent_execute(name, slot):
    def execute(state):
        game.execute_choose_permanent_option(state, name, slot)
    return execute


def _choose_opponent_permanent_legal(name, slot):
    """The general cross-player targeting primitive's action-table half
    (docs/COMBAT_PLAN.md) -- legal only while a "choose_opponent_permanent"
    resolution is pending and (name, slot) is one of its own current
    options. Only ever correct when the referencing side is already the
    active perspective (game.begin_choose_opponent_permanent's own
    docstring) -- blocking's own defender-decision channel is what
    guarantees that, not this function."""
    def legal(state):
        pending = state.pending_resolution
        return (
            pending is not None and pending["kind"] == "choose_opponent_permanent"
            and (name, slot) in game.choose_opponent_permanent_options(state)
        )
    return legal


def _choose_opponent_permanent_execute(name, slot):
    def execute(state):
        game.execute_choose_opponent_permanent_option(state, name, slot)
    return execute


def _assign_blocker_legal(name, slot):
    """One "Assign Blocker: <name> (slot j)" action (docs/COMBAT_PLAN.md's
    blocking design) -- legal only while a "declare_blockers" resolution
    is pending (game.turn._declare_blockers_gen has already flipped
    state.active_idx to the defender by the time this is ever checked)
    and the specific physical permanent at this (name, slot) is currently
    block-eligible (game.creature_block_eligible): untapped, not already
    assigned to block something else this combat. Unlike attacking,
    neither summoning sickness nor Defender excludes a blocker -- see
    creature_block_eligible's own docstring for why."""
    def legal(state):
        pending = state.pending_resolution
        if pending is None or pending["kind"] != "declare_blockers":
            return False
        return any(
            p.card_def.name == name and p.slot == slot and game.creature_block_eligible(state, p)
            for p in state.battlefield
        )
    return legal


def _assign_blocker_execute(name, slot):
    """Parks the specific physical permanent at this (name, slot) as a
    blocker, then hands off to game.declare_blocker_assignment, which
    nests a cross-player choose_opponent_permanent sub-resolution to pick
    which of the attacker's declared, not-yet-blocked attackers it
    blocks -- restricted by extra_predicate to attackers this specific
    blocker is actually allowed to block: flying's own restriction
    (docs/COMBAT_PLAN.md step 7) means an attacker with flying can only be
    chosen here if `blocker` itself also has flying (game.has_keyword --
    resolution.py can't compute this itself, see declare_blocker_
    assignment's own docstring for why the predicate has to come from
    here instead). Once that completes, re-opens begin_declare_blockers
    (via the captured outer on_complete) so the defender can assign
    another blocker or choose Done -- same nested-callback shape
    execute_madness_cast already uses for its own multi-step chain."""
    def execute(state):
        blocker = next(
            p for p in state.battlefield
            if p.card_def.name == name and p.slot == slot and game.creature_block_eligible(state, p)
        )
        outer_on_complete = state.pending_resolution["on_complete"]

        def _blockable_by(attacker):
            return not game.has_keyword(state, attacker, "flying") or game.has_keyword(state, blocker, "flying")

        game.declare_blocker_assignment(
            state, blocker, on_complete=lambda s: game.begin_declare_blockers(s, outer_on_complete),
            extra_predicate=_blockable_by,
        )
    return execute


def _done_blocking_legal(state):
    pending = state.pending_resolution
    return pending is not None and pending["kind"] == "declare_blockers"


def _done_blocking_execute(state):
    game.complete_resolution(state)


def _pool_spend_legal(color):
    def legal(state):
        return (
            state.pending_resolution is not None
            and state.pending_resolution["kind"] == "pay_cost"
            and color in game.pool_spend_options(state)
        )
    return legal


def _pool_spend_execute(color):
    def execute(state):
        game.execute_pool_spend(state, color)
    return execute


def _keep_dispose_legal(state):
    pending = state.pending_resolution
    return pending is not None and pending["kind"] in ("scry", "surveil") and bool(pending["remaining"])


def _keep_execute(state):
    game.execute_scry_surveil_option(state, "keep")


def _dispose_execute(state):
    game.execute_scry_surveil_option(state, "dispose")


def _decline_legal(state):
    pending = state.pending_resolution
    return pending is not None and pending["kind"] == "ancient_stirrings"


def _decline_execute(state):
    game.execute_ancient_stirrings_option(state, "decline")


def _decline_malevolent_rumble_legal(state):
    pending = state.pending_resolution
    return pending is not None and pending["kind"] == "malevolent_rumble"


def _decline_malevolent_rumble_execute(state):
    game.execute_malevolent_rumble_option(state, "decline")


def _abandon_payment_legal(state):
    pending = state.pending_resolution
    return pending is not None and pending["kind"] == "pay_cost"


def _abandon_payment_execute(state):
    game.abandon_pay_cost(state)


# ---------------------------------------------------------------------------
# spy_combo deck additions: select_to_hand's own fixed actions (Lead the
# Stampede), an optional-search decline, non-mana activated abilities
# (Quirion Ranger), Land Grant's free alt-cost, Dread Return's Flashback,
# and Winding Way's modal cast. None of these fire for Tron cards -- each
# is gated on a registry key no Tron EffectId sets.
# ---------------------------------------------------------------------------

def _select_to_hand_keep_legal(state):
    pending = state.pending_resolution
    return (
        pending is not None and pending["kind"] == "select_to_hand"
        and bool(pending["remaining"]) and pending["eligible"](pending["remaining"][0])
    )


def _select_to_hand_bottom_legal(state):
    pending = state.pending_resolution
    return pending is not None and pending["kind"] == "select_to_hand" and bool(pending["remaining"])


def _select_to_hand_keep_execute(state):
    game.execute_select_to_hand_option(state, "keep")


def _select_to_hand_bottom_execute(state):
    game.execute_select_to_hand_option(state, "bottom")


def _decline_search_legal(state):
    pending = state.pending_resolution
    return (
        pending is not None and pending["kind"] == "search_fetch" and pending.get("optional")
        and bool(game.search_fetch_options(state))
    )


def _decline_search_execute(state):
    game.execute_search_fetch_decline(state)


def _decline_discard_legal(state):
    pending = state.pending_resolution
    return (
        pending is not None and pending["kind"] == "discard" and pending.get("optional")
        and bool(game.discard_options(state))
    )


def _decline_discard_execute(state):
    game.execute_discard_decline(state)


def _madness_cast_legal(state):
    """Legal only if the model can actually afford the exiled card's
    madness cost right now -- same "guaranteed payable, not a maybe"
    contract every other alternate cast path here already follows."""
    pending = state.pending_resolution
    if pending is None or pending["kind"] != "madness_decision":
        return False
    madness_spec = game.EFFECT_REGISTRY[pending["card_def"].effect_id]["madness"]
    return game.plan_payment(state, madness_spec["cost"]) is not None


def _madness_cast_execute(state):
    game.execute_madness_cast(state)


def _madness_decline_legal(state):
    pending = state.pending_resolution
    return pending is not None and pending["kind"] == "madness_decision"


def _madness_decline_execute(state):
    game.execute_madness_decline(state)


def _activate_no_cost_legal(name, ability_legal, speed):
    """Non-mana activated-ability cost (Quirion Ranger's Forest bounce):
    no {T}-of-self assumption, unlike _activate_legal -- the ability's own
    legal(state, permanent) captures its whole cost precondition."""
    def legal(state):
        if state.pending_resolution is not None:
            return False
        if not game.turn.speed_legal(state, speed):
            return False
        p = next((p for p in state.battlefield if p.card_def.name == name), None)
        return p is not None and ability_legal(state, p)
    return legal


def _activate_no_cost_execute(name, resolve):
    def execute(state):
        p = next(p for p in state.battlefield if p.card_def.name == name)
        resolve(state, p)
    return execute


def _alt_cast_legal(name, extra_legal, speed):
    """Land Grant's free alt-cost: no mana payment at all, just the
    card's own extra_legal predicate (0 lands in hand)."""
    def legal(state):
        if state.pending_resolution is not None:
            return False
        if not game.turn.speed_legal(state, speed):
            return False
        if not any(c.name == name for c in state.hand):
            return False
        return extra_legal(state)
    return legal


def _alt_cast_execute(name, resolve):
    """No generic engine-level cost mechanism for an alt cost (unlike mana's
    begin_pay_cost) -- so, same as _flashback_execute, this calls resolve
    immediately and leaves deferring-onto-the-stack entirely up to resolve
    itself. Alt-cost shapes vary: Land Grant's is free (nothing to pay, so
    its own resolve pushes right away, same as a free Flashback), Fireblast's
    is a real alternate cost (sacrifice 2 Mountains) that must actually be
    paid -- via its own resolution -- before ITS effect gets pushed. Pushing
    generically here, before resolve even runs, would defer Fireblast's own
    cost-payment along with its effect, which is wrong: the cost must be
    paid before anything is fully paid for and put on the stack."""
    def execute(state):
        card_def = game.CARD_DEFS[name]
        game.on_cast_trigger(state, card_def)  # item 11 -- see _cast_execute
        resolve(state, card_def)
    return execute


def _flashback_legal(name, ability_legal, speed):
    """Dread Return's Flashback: cast from the graveyard, not hand. Real
    Magic: Flashback follows the same timing as the card itself, not its
    own independent rule -- speed is the same value the card's normal
    cast derived, not a separate default."""
    def legal(state):
        if state.pending_resolution is not None:
            return False
        if not game.turn.speed_legal(state, speed):
            return False
        if not any(c.name == name for c in state.graveyard):
            return False
        return ability_legal(state)
    return legal


def _flashback_execute(name, resolve):
    def execute(state):
        card_def = game.CARD_DEFS[name]
        game.on_cast_trigger(state, card_def)  # item 11 -- see _cast_execute
        resolve(state, card_def)
    return execute


def _plot_legal(name, cost, speed):
    """Plot {cost}: pay it and exile this card from hand (no board
    presence yet) -- legal exactly like a normal cast, just against the
    plot cost instead of card_def.cast_cost. Real Magic: Plot's own
    reminder text is "any time you could cast this card" -- same speed as
    the card's normal cast, not a separate timing rule; the later free
    cast from exile (_cast_from_exile_legal) uses the same speed too."""
    def legal(state):
        if state.pending_resolution is not None:
            return False
        if not game.turn.speed_legal(state, speed):
            return False
        if not any(c.name == name for c in state.hand):
            return False
        return game.plan_payment(state, cost) is not None
    return legal


def _plot_execute(name, cost, resolve):
    def execute(state):
        card_def = game.CARD_DEFS[name]
        # Plotting itself isn't casting the spell (it's exiled, not
        # resolved) -- no on_cast_trigger here; that fires from
        # _cast_from_exile_execute below, once it's actually cast.
        game.begin_pay_cost(state, cost, on_complete=lambda s: resolve(s, card_def))
    return execute


def _cast_from_exile_legal(name, extra_legal, speed):
    """Plot's second half: cast a previously-plotted copy, without paying
    its mana cost, on any turn after the one it was plotted on. speed:
    same value _plot_legal used -- see that function's own docstring.

    extra_legal: Plot only waives the MANA cost, not any other cost a
    card's normal "cast" spec gates on (e.g. Highway Robbery's own
    "discard a card" additional cost still needs a card in hand to
    discard) -- reuses the same cast_spec["extra_legal"] the normal cast
    path already checks, so a card needing both never looks payable when
    it secretly isn't. None (every existing Plot card so far) means no
    such gate, unaffected."""
    def legal(state):
        if state.pending_resolution is not None:
            return False
        if not game.turn.speed_legal(state, speed):
            return False
        has_plotted = any(
            c.name == name and stamp is not None and stamp < state.turn_number
            for c, stamp in state.exile
        )
        if not has_plotted:
            return False
        return extra_legal is None or extra_legal(state)
    return legal


def _cast_from_exile_execute(name, resolve):
    def execute(state):
        card_def = game.CARD_DEFS[name]
        entry = next(
            e for e in state.exile
            if e[0].name == name and e[1] is not None and e[1] < state.turn_number
        )
        state.exile.remove(entry)
        game.on_cast_trigger(state, card_def)  # item 11 -- see _cast_execute
        # Plot's whole point is that the cost was already paid earlier
        # (when plotted) -- already "fully paid for" now, so push
        # immediately instead of resolving now (see _cast_execute's own
        # stack comment).
        game.push_to_stack(state, card_def, resolve)
    return execute


def build_action_table(decklist, registry, token_card_defs=(), pending_kinds=(),
                        opponent_decklist=None, opponent_token_card_defs=()):
    """opponent_decklist/opponent_token_card_defs: the OTHER side's own
    decklist/tokens (docs/COMBAT_PLAN.md's cross-player targeting
    primitive) -- None/() for every 1-player deck (there's no real
    opponent battlefield to reference at all), matching combat_enabled=False
    decks never seeing "Attack: X" become legal. Only ever given by
    TwoPlayerDeckEnv, which already has this data on hand for its own
    opponent_actions table.

    token_card_defs: every token CardDef this deck's own cards can
    create at runtime (Blood, Robot, Warrior, Eldrazi Spawn --
    docs/MADNESS_DECKS_PLAN.md item 8), e.g. (game.BLOOD_TOKEN_CARD_DEF,).
    Tokens are never decklist entries (no quantity, not in game.CARD_DEFS),
    so they can't flow through distinct_names/game.CARD_DEFS[name] the way
    every other action here does -- casting/land-drop/Flashback/etc. stay
    decklist-only, a token is never cast or played as a land.

    Two independent things read this list, for two different reasons: the
    activated-abilities loop below (a token's own ability, e.g. Blood's
    sac-for-a-card or Eldrazi Spawn's sac-for-{C}, needs an action to
    exist at all), and the choosable_names set that both the "Choose: X"
    and "Choose target: X (slot k)" name lists build from (a token can be
    a perfectly legal choose_permanent/sacrifice/discard choice -- e.g. any
    creature-enchanting Aura can enchant a token creature -- despite never
    appearing in the decklist; list a token here even if it has no
    activated ability of its own, like Warrior, so it stays a legal choice
    once it's on the battlefield). Defaults to () so every existing call
    site (Tron, spy_combo -- neither creates tokens) is unaffected.

    pending_kinds: this deck's own extra pending-resolution kinds beyond
    the universal baseline (pay_cost) -- see game.registry.
    derive_pending_kinds -- gates which of the fixed kind-specific actions
    below (Keep/Dispose scry-surveil, Decline Ancient Stirrings, etc.)
    actually get added, so a deck's action table never grows because of a
    pending kind only some other deck can reach."""
    distinct_names = sorted({name for name, *_rest in decklist})
    land_names = sorted({
        name for name in distinct_names if game.CARD_DEFS[name].card_type == game.CardType.LAND
    })

    actions = []

    for name in land_names:
        actions.append((f"Play land: {name}", _land_drop_legal(name), _land_drop_execute(name)))

    for name in distinct_names:
        card_spec = registry.get(game.CARD_DEFS[name].effect_id, {})
        cast_spec = card_spec.get("cast")
        if cast_spec is not None:
            # "precast_choice": True (Auras' real targets, Crop Rotation's
            # sacrifice-as-a-cost) -- resolve must run immediately once paid
            # and manage its own push_to_stack, instead of the generic
            # auto-push _cast_execute does (see _precast_choice_execute's
            # own docstring for exactly why).
            cast_execute_fn = _precast_choice_execute if cast_spec.get("precast_choice") else _cast_execute
            actions.append((
                f"Cast {name}",
                _cast_legal(name, cast_spec.get("extra_legal"), _cast_speed(game.CARD_DEFS[name], cast_spec)),
                cast_execute_fn(name, cast_spec["resolve"]),
            ))
        # Winding Way: a modal cast (choose creature or land) instead of a
        # single "cast" entry -- one action per mode.
        cast_modes = card_spec.get("cast_modes")
        if cast_modes is not None:
            for mode_name, mode_spec in cast_modes.items():
                mode_execute_fn = _precast_choice_execute if mode_spec.get("precast_choice") else _cast_execute
                actions.append((
                    f"Cast {name} (choose {mode_name})",
                    _cast_legal(name, mode_spec.get("extra_legal"), _cast_speed(game.CARD_DEFS[name], mode_spec)),
                    mode_execute_fn(name, mode_spec["resolve"]),
                ))
        # Land Grant: a second, free cast path alongside the normal one.
        alt_cast = card_spec.get("alt_cast")
        if alt_cast is not None:
            actions.append((
                f"Cast {name} (free)",
                _alt_cast_legal(name, alt_cast["extra_legal"], _cast_speed(game.CARD_DEFS[name], alt_cast)),
                _alt_cast_execute(name, alt_cast["resolve"]),
            ))
        # Dread Return: Flashback casts from the graveyard, not hand.
        flashback = card_spec.get("flashback")
        if flashback is not None:
            actions.append((
                f"Flashback {name}",
                _flashback_legal(name, flashback["legal"], _cast_speed(game.CARD_DEFS[name], flashback)),
                _flashback_execute(name, flashback["resolve"]),
            ))
        # Highway Robbery: Plot -- pay its plot cost to exile it now,
        # cast it for free from exile on any later turn. The cast-from-
        # exile half reuses this same card's normal "cast" resolve (the
        # real spell effect is identical either way, only how the cost
        # was paid differs) -- so a "plot" entry only makes sense
        # alongside a "cast" entry, never alone.
        plot = card_spec.get("plot")
        if plot is not None:
            plot_speed = _cast_speed(game.CARD_DEFS[name], plot)  # same speed governs both actions below -- Plot's own reminder text ("any time you could cast this card") is one timing rule, not two
            actions.append((
                f"Plot {name}",
                _plot_legal(name, plot["cost"], plot_speed),
                _plot_execute(name, plot["cost"], plot["resolve"]),
            ))
            # cast_from_exile_resolve: optional override for cards whose
            # normal "cast" resolve does state.hand.remove(card_def) (the
            # universal convention for every cast resolve in this codebase)
            # -- wrong once the card already left exile, never hand, by
            # the time this runs (Highway Robbery's own real-world case;
            # every existing Plot self-check's resolve happens to be a
            # no-op, which is why this distinction never mattered before).
            # Falls back to cast_spec["resolve"] unchanged for any card
            # whose resolve doesn't care either way.
            actions.append((
                f"Cast {name} (plotted)",
                _cast_from_exile_legal(name, cast_spec.get("extra_legal"), plot_speed),
                _cast_from_exile_execute(name, plot.get("cast_from_exile_resolve", cast_spec["resolve"])),
            ))

    activatable = [(name, game.CARD_DEFS[name].effect_id) for name in distinct_names]
    activatable += [(cd.name, cd.effect_id) for cd in token_card_defs]
    for name, effect_id in activatable:
        abilities = registry.get(effect_id, {}).get("activated_abilities", {})
        for ability_name, spec in abilities.items():
            # Real Magic's own default for activated (non-mana) abilities is
            # the opposite of a spell's: any time, unless the card says
            # "activate only as a sorcery" -- an explicit "speed" key in the
            # ability's own spec is that opt-in override; every existing
            # ability (Blood, Candy Trail, Expedition Map, Bonders'
            # Ornament, Quirion Ranger, Barrels) has none, so all keep
            # working in every phase exactly as before this feature existed.
            speed = spec.get("speed", game.turn.Speed.INSTANT)
            if "cost_key" in spec:
                actions.append((
                    f"Activate {name} ({ability_name})",
                    _activate_legal(name, spec["cost_key"], speed),
                    _activate_execute(name, spec["cost_key"], spec["resolve"]),
                ))
            else:
                # Non-mana cost (Quirion Ranger: return a Forest to hand).
                actions.append((
                    f"Activate {name} ({ability_name})",
                    _activate_no_cost_legal(name, spec["legal"], speed),
                    _activate_no_cost_execute(name, spec["resolve"]),
                ))

    for name in distinct_names:
        fc_spec = registry.get(game.CARD_DEFS[name].effect_id, {}).get("forestcycle")
        if fc_spec is not None:
            actions.append((
                f"Forestcycle {name}",
                _forestcycle_legal(name, fc_spec["cost_key"]),
                _forestcycle_execute(name, fc_spec["cost_key"], fc_spec["resolve"]),
            ))

    actions.append(("Pass", _pass_legal, _pass_execute))

    # "Choose: X" needs to cover every name a sacrifice/discard/search_fetch/
    # etc. resolution could ever offer -- not just decklist names. (NOT
    # choose_permanent -- that's the "Choose target: X (slot k)" block
    # below, exact-(name, slot) addressed.) A token (e.g. boggles' Eldrazi
    # Spawn) is a perfectly legal sacrifice/discard choice despite never
    # appearing in CARD_DEFS/the decklist; omitting token names here left
    # exactly that case legal-to-create but impossible-to-choose once a
    # token was the only eligible option, softlocking the game.
    choosable_names = sorted(set(distinct_names) | {cd.name for cd in token_card_defs})
    for name in choosable_names:
        actions.append((f"Choose: {name}", _choose_name_legal(name), _choose_name_execute(name)))

    # "Attack: X (slot k)" -- one per (creature name, slot) pair
    # (docs/COMBAT_PLAN.md's permanent-identity design), legal only during
    # Phase.DECLARE_ATTACKERS (see _attack_legal). k runs 1..that card's
    # own decklist quantity for a real card -- the pooled slot scheme
    # means this is a hard, correct bound even through repeated bounce/
    # blink, since only that many physical copies can ever be
    # simultaneously alive. A token has no decklist quantity to read, so
    # k instead runs 1..TOKEN_LIMIT -- a shared pool across every token
    # name combined, so any single name could in principle claim all of
    # it, and each name's own registered range has to cover that worst
    # case independently. A deck whose own phase sequence never includes
    # DECLARE_ATTACKERS (combat_enabled=False) simply never sees any of
    # these become legal -- same "phase not in this deck's own sequence"
    # degrade every other phase-gated action already relies on.
    card_type_by_name = {name: game.CARD_DEFS[name].card_type for name in distinct_names}
    card_type_by_name.update({cd.name: cd.card_type for cd in token_card_defs})
    qty_by_name = {name: qty for name, qty, *_rest in decklist}

    # "Choose target: X (slot k)" -- the "choose_permanent" resolution's own
    # exact-(name, slot) addressed actions (Aura enchant-targets, Crop
    # Rotation's sacrifice cost, land bounce), same shape/reasoning as
    # "Choose opponent's: X (slot k)" below just scoped to THIS side's own
    # battlefield. Registered for every choosable name, not just creatures
    # (unlike "Attack:"/"Assign Blocker:" below) -- Utopia Sprawl/Abundant
    # Growth target lands, not creatures -- and legal() gates precisely at
    # runtime against whichever predicate the actual pending choose_permanent
    # resolution holds, same "pre-register broadly, mask precisely" pattern
    # "Choose: X as color" below already uses.
    for name in choosable_names:
        max_slot = qty_by_name.get(name, game.TOKEN_LIMIT)
        for slot in range(1, max_slot + 1):
            actions.append((
                f"Choose target: {name} (slot {slot})",
                _choose_permanent_legal(name, slot),
                _choose_permanent_execute(name, slot),
            ))

    attackable_names = sorted(name for name in choosable_names if card_type_by_name[name] == game.CardType.CREATURE)
    for name in attackable_names:
        max_slot = qty_by_name.get(name, game.TOKEN_LIMIT)
        for slot in range(1, max_slot + 1):
            actions.append((
                f"Attack: {name} (slot {slot})",
                _attack_legal(name, slot),
                _attack_execute(name, slot),
            ))

    # "Assign Blocker: X (slot j)" -- same own-creature (name, slot)
    # addressing as "Attack: X (slot k)" above, since blocking is a
    # decision about this player's OWN creatures (docs/COMBAT_PLAN.md),
    # just legal at a different point (once _declare_blockers_gen has
    # flipped state.active_idx to the defender and a "declare_blockers"
    # resolution is pending -- see _assign_blocker_legal). "Done blocking"
    # is the explicit action that closes the consult, same "Done" precedent
    # as scry/surveil's own keep-then-order decomposition.
    for name in attackable_names:
        max_slot = qty_by_name.get(name, game.TOKEN_LIMIT)
        for slot in range(1, max_slot + 1):
            actions.append((
                f"Assign Blocker: {name} (slot {slot})",
                _assign_blocker_legal(name, slot),
                _assign_blocker_execute(name, slot),
            ))
    actions.append(("Done blocking", _done_blocking_legal, _done_blocking_execute))

    # "Choose opponent's: X (slot k)" -- the general cross-player
    # targeting primitive (docs/COMBAT_PLAN.md), one per (opponent
    # creature name, slot), built from the OPPONENT's own decklist/tokens
    # instead of this side's own -- blocking's first consumer. Same
    # quantity-or-TOKEN_LIMIT bound as the attack registration above, just
    # applied to the other side's card pool. None/() (the default for
    # every 1-player deck) registers nothing at all -- there's no real
    # opponent battlefield to ever reference in that mode.
    if opponent_decklist is not None:
        opponent_distinct_names = sorted({name for name, *_rest in opponent_decklist})
        opponent_card_type_by_name = {name: game.CARD_DEFS[name].card_type for name in opponent_distinct_names}
        opponent_card_type_by_name.update({cd.name: cd.card_type for cd in opponent_token_card_defs})
        opponent_qty_by_name = {name: qty for name, qty, *_rest in opponent_decklist}
        opponent_choosable_names = sorted(
            set(opponent_distinct_names) | {cd.name for cd in opponent_token_card_defs}
        )
        opponent_targetable_names = sorted(
            name for name in opponent_choosable_names
            if opponent_card_type_by_name[name] == game.CardType.CREATURE
        )
        for name in opponent_targetable_names:
            max_slot = opponent_qty_by_name.get(name, game.TOKEN_LIMIT)
            for slot in range(1, max_slot + 1):
                actions.append((
                    f"Choose opponent's: {name} (slot {slot})",
                    _choose_opponent_permanent_legal(name, slot),
                    _choose_opponent_permanent_execute(name, slot),
                ))

    # Abundant Growth's own grant: a runtime, per-instance fact (which
    # specific land, if any, ends up enchanted) that can't be known when
    # this table is built, before any game state exists -- so every land
    # name gets a "Choose: X as color" slot for every color ANY card in
    # this decklist can ever grant, pre-registered here and masked
    # legal/illegal at runtime by mana.tap_cost_options actually seeing
    # (or not seeing) an attached grant.
    grantable_colors = set()
    for name in distinct_names:
        grantable_colors |= registry.get(game.CARD_DEFS[name].effect_id, {}).get("grants_mana_colors", set())

    for name in distinct_names:
        spec = registry.get(game.CARD_DEFS[name].effect_id, {})
        colors = set()
        mana = spec.get("mana")
        if mana is not None and mana[0] == "flexible":
            colors |= mana[1]
        filter_mana = spec.get("filter_mana")
        if filter_mana is not None:
            colors |= filter_mana["colors"]
        if game.CARD_DEFS[name].card_type == game.CardType.LAND:
            colors |= grantable_colors
        for color in sorted(colors):
            actions.append((
                f"Choose: {name} as {color}",
                _choose_name_color_legal(name, color),
                _choose_name_color_execute(name, color),
            ))

    for color in game.POOL_COLORS:
        actions.append((
            f"Spend {color} from pool",
            _pool_spend_legal(color),
            _pool_spend_execute(color),
        ))

    if "scry" in pending_kinds or "surveil" in pending_kinds:
        actions.append(("Keep (scry/surveil)", _keep_dispose_legal, _keep_execute))
        actions.append(("Dispose (scry/surveil)", _keep_dispose_legal, _dispose_execute))
    if "ancient_stirrings" in pending_kinds:
        actions.append(("Decline (Ancient Stirrings)", _decline_legal, _decline_execute))
    if "malevolent_rumble" in pending_kinds:
        actions.append(("Decline (Malevolent Rumble)", _decline_malevolent_rumble_legal, _decline_malevolent_rumble_execute))
    if "select_to_hand" in pending_kinds:
        actions.append(("Keep (select to hand)", _select_to_hand_keep_legal, _select_to_hand_keep_execute))
        actions.append(("Bottom (select to hand)", _select_to_hand_bottom_legal, _select_to_hand_bottom_execute))
    if "search_fetch" in pending_kinds:
        # Gated on "search_fetch" membership alone, not per-deck optionality
        # (Tron's own search_fetch uses are never optional=True, so this
        # stays present-but-permanently-illegal for Tron -- same as it was
        # unconditionally before this change; both current decks already
        # share "search_fetch" either way, so this isn't a growth vector).
        actions.append(("Decline (search)", _decline_search_legal, _decline_search_execute))
    actions.append(("Abandon payment", _abandon_payment_legal, _abandon_payment_execute))  # pay_cost is baseline, always present
    if "discard" in pending_kinds:
        actions.append(("Decline (discard)", _decline_discard_legal, _decline_discard_execute))
    if "madness_decision" in pending_kinds:
        actions.append(("Cast (madness)", _madness_cast_legal, _madness_cast_execute))
        actions.append(("Decline (madness)", _madness_decline_legal, _madness_decline_execute))

    return tuple(actions)


_tap_cost_options_cache = None  # (state, result) -- valid only for the duration of one legal_action_mask sweep, see _cached_tap_cost_options


def _cached_tap_cost_options(state):
    """Memoizes game.tap_cost_options(state) for the exact duration of one
    legal_action_mask sweep (docs/PRIORITY_PLAN.md item 6 -- profiled,
    not guessed: mana.tap_cost_options was called 480,942 times against
    only 78,969 mask builds, ~6x more than needed and ~10% of total
    training time by itself). _choose_name_legal/_choose_name_color_legal
    (the "Choose: X"/"Choose: X as color" mana-source actions) each
    independently call this from scratch, once per candidate name/color,
    so one sweep recomputes the identical list several times over.
    Provably safe to cache for exactly this scope: a legal_action_mask
    sweep only ever calls legal_fns, never an execute_* function, so state
    can't change mid-sweep -- legal_action_mask resets this cache before
    and after its own sweep (see there), so nothing outside a sweep (an
    actual execute_fn call, a later sweep against mutated state) can ever
    see a stale hit."""
    global _tap_cost_options_cache
    if _tap_cost_options_cache is None or _tap_cost_options_cache[0] is not state:
        _tap_cost_options_cache = (state, game.tap_cost_options(state))
    return _tap_cost_options_cache[1]


def legal_action_mask(state, actions):
    """Stateless: usable both by DeckEnv.action_masks() and by
    harness.evaluate(), which plays games directly through game.run_game,
    not through env.step (see DRL_CHECKLIST.md's D6 implementation note).
    `actions` is any table built by build_action_table -- every deck's own
    table, none privileged as a default (a caller with its own decklist
    always has its own table to pass, e.g. harness.py's self.actions).

    Resets _tap_cost_options_cache before AND after the sweep itself
    (not just before): guarantees the cache can never leak past this
    call's own scope into a later execute_fn call or an unrelated sweep
    against a different/mutated state, even though nothing in the current
    single-threaded, synchronous call pattern would actually trigger that
    -- belt-and-suspenders for a module-level global, not load-bearing."""
    global _tap_cost_options_cache
    _tap_cost_options_cache = None
    try:
        return np.array([legal_fn(state) for _, legal_fn, _ in actions], dtype=bool)
    finally:
        _tap_cost_options_cache = None


# ---------------------------------------------------------------------------
# D2.3 / D2.4 -- DeckEnv
# ---------------------------------------------------------------------------

def _fast_forward(gen, state, my_seat_idx, other_seat_choose_action):
    """Feeds other_seat_choose_action(state)'s answers into `gen` for as
    long as someone other than my_seat_idx currently holds priority
    (docs/PRIORITY_PLAN.md's turn-owner/priority-holder split --
    state.active_idx, not turn_player_idx, is what a generic yield
    protocol needs to dispatch on). my_seat_idx=None (DeckEnv's own
    1-player case) never loops at all -- there's no opponent to consult,
    and _run_turn_gen itself never flips active_idx away from the sole
    player in a 1-player game. Raises StopIteration exactly like a bare
    gen.send would if the turn ends mid-loop -- callers already catch
    that around this."""
    while my_seat_idx is not None and state.active_idx != my_seat_idx:
        gen.send(other_seat_choose_action(state))


def _start_turn(state, combat_enabled, my_seat_idx=None, other_seat_choose_action=None):
    """Primes game.turn._run_turn_gen -- the same turn-loop generator
    run_turn drives synchronously -- up to the first yield where
    my_seat_idx itself actually holds priority, and returns it. DeckEnv/
    TwoPlayerDeckEnv.step() then send one action (or None for Pass) at a
    time into the returned generator, one per gym step() call, instead of
    running any phase's loop themselves.

    my_seat_idx/other_seat_choose_action: None for DeckEnv (1-player, no
    real opponent to consult -- see _fast_forward). TwoPlayerDeckEnv
    passes self.my_seat_idx/self._opponent_choose_action, so every
    opponent decision that comes up DURING MY OWN turn (responding to one
    of my spells with an instant, blocking one of my attacks -- now just
    an ordinary active_idx flip inside the general priority round, not a
    separate mechanism) is fast-forwarded through internally instead of
    ever reaching the external gym step() caller, who should only ever be
    asked for MY OWN decisions.

    Returns None if the turn already ended before priority ever came back
    to my_seat_idx -- unreachable in practice on Phase.UNTAP's own effect
    (it never draws or sets turn_won), but a trigger left queued from the
    previous turn (a Sneaky-Snacker-style automatic return) drains here,
    before the first yield, and could itself set turn_won via
    enters_battlefield's terminated_fn check. See DeckEnv.step()'s
    turn_ended handling for how a None generator is handled (there is
    nothing to send to an already-exhausted one; state.turn_won is already
    set by the time this returns None, so the next done computation
    catches it)."""
    gen = game.turn._run_turn_gen(state, combat_enabled=combat_enabled)
    try:
        next(gen)
        _fast_forward(gen, state, my_seat_idx, other_seat_choose_action)
    except StopIteration:
        return None
    return gen


def _substitute_and_resolve(state, actions, pass_action, mask, action):
    """Shared tail of _resolve_step_action (DeckEnv.step/TwoPlayerDeckEnv.
    step) and model_choose_action: substitute the first currently-legal
    action if `action` turns out illegal (works for any SB3 algorithm,
    maskable or not). MUST NOT assume PASS_ACTION specifically is always
    a safe substitute (MULTI_DECK_PLAN.md Phase M4e): Pass is illegal
    whenever a resolution is pending, and blindly "passing" in that state
    would abandon the resolution mid-flight and desync
    state.pending_resolution from the turn loop. The legal set is never
    empty: with no resolution pending, Pass itself is always legal; with
    one pending, every kind guarantees at least one option (pay_cost by
    construction, search_fetch/choose_permanent's empty case auto-fizzles
    instead of leaving a stuck resolution, ancient_stirrings always
    offers "decline", scry/surveil always offers keep/dispose or a
    nonempty ordering set).

    Returns the zero-arg callable (or None for Pass) the turn generator's
    own send() protocol expects."""
    if not mask[action]:
        legal_indices = np.flatnonzero(mask)
        if legal_indices.size == 0:
            # Every kind above is supposed to guarantee at least one legal
            # action -- this should be unreachable. Fail loudly with the
            # pending-resolution context needed to find which kind's own
            # guarantee actually broke, rather than a bare, contextless
            # IndexError (same "fail loudly, not silently" precedent as
            # harness.py's load() mismatch check).
            pending = state.pending_resolution
            raise RuntimeError(
                "legal_action_mask is entirely empty -- no action, not even Pass, is legal. "
                f"pending_resolution={pending!r} phase={getattr(state, 'phase', None)!r} "
                f"active_idx={getattr(state, 'active_idx', None)!r} "
                f"turn_player_idx={getattr(state, 'turn_player_idx', None)!r} "
                f"turn_number={getattr(state, 'turn_number', None)!r} "
                f"hand={[c.name for c in state.hand]!r} "
                f"battlefield={[(p.card_def.name, p.slot, p.tapped) for p in state.battlefield]!r} "
                f"stack={[e['card_def'].name for e in state.stack]!r}"
            )
        action = int(legal_indices[0])
    if action == pass_action:
        return None
    _, _, execute_fn = actions[action]
    return lambda execute_fn=execute_fn: execute_fn(state)


def _resolve_step_action(env, action):
    """Shared by DeckEnv.step/TwoPlayerDeckEnv.step (both duck-typed with
    the same _cached_mask/state/actions/pass_action attributes): reuse
    action_masks()'s cached mask if still fresh -- MaskablePPO's rollout
    loop (sb3_contrib ppo_mask.collect_rollouts) always calls
    action_masks() immediately before step(), with no state mutation in
    between (just a policy forward pass), so recomputing here would be
    pure duplicate work. Reused once, then cleared so any other caller
    (or an out-of-order call) falls back to a fresh computation instead
    of a stale one. See _substitute_and_resolve for the rest."""
    if env._cached_mask is not None:
        mask, env._cached_mask = env._cached_mask, None
    else:
        mask = legal_action_mask(env.state, env.actions)
    return _substitute_and_resolve(env.state, env.actions, env.pass_action, mask, action)


class DeckEnv(gymnasium.Env):
    # Deck-parameterized (MULTI_DECK_PLAN.md Phase M4/M7): no deck gets a
    # default here (not even Tron) -- decklist/terminated_fn/pending_kinds
    # are always the caller's own (e.g. game.parse_decklist_file(...),
    # terminated.tron_terminated, game.derive_pending_kinds(decklist)).
    # Its own action table and observation dim are built fresh per
    # instance, never read from any module-level global.
    def __init__(self, reward_fn, decklist, terminated_fn, pending_kinds,
                 horizon=6, on_the_play=True, seed=None, combat_enabled=False, token_card_defs=()):
        super().__init__()
        self.reward_fn = reward_fn
        self.decklist = decklist
        self.terminated_fn = terminated_fn
        self.horizon = horizon
        self.on_the_play = on_the_play
        self.pending_kinds = pending_kinds
        # rakdos madness / mono red madness / boggles only (default off,
        # same as every other deck-specific knob here) -- combat itself is
        # still fully automatic, no attack/block decision (see
        # game.turn._run_turn_gen's own combat_enabled docstring), so this
        # adds no action-table entries. It does change which phases the
        # turn generator visits (game.turn.FULL_PHASES vs MINIMAL_PHASES),
        # which the Phase one-hot in build_observation reflects.
        self.combat_enabled = combat_enabled
        # Tokens an in-play card can create at runtime (Blood/Robot/Warrior/
        # Eldrazi Spawn) whose own activated ability, if any, needs an
        # action-table entry -- see build_action_table's own token_card_defs
        # docstring. Defaults to () so every existing caller (none of which
        # currently pass this) is unaffected.
        self.token_card_defs = token_card_defs
        self._rng = random.Random(seed)
        self.state = None
        self._turn_gen = None  # set by reset(); the live game.turn._run_turn_gen for the current turn
        self.actions = build_action_table(
            decklist, game.EFFECT_REGISTRY, token_card_defs=token_card_defs, pending_kinds=pending_kinds,
        )
        self.pass_action = next(i for i, (name, _legal, _execute) in enumerate(self.actions) if name == "Pass")
        self.observation_dim = observation_dim_for(decklist, pending_kinds)
        self.observation_space = spaces.Box(low=0.0, high=1.0, shape=(self.observation_dim,), dtype=np.float32)
        self.action_space = spaces.Discrete(len(self.actions))
        self._cached_mask = None  # set by action_masks(), consumed once by the next step() -- see there

    def action_masks(self):
        if self.state is None:
            raise RuntimeError("action_masks() called before reset()")
        self._cached_mask = legal_action_mask(self.state, self.actions)
        return self._cached_mask

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        if seed is not None:
            self._rng = random.Random(seed)
        self.state = game.new_game_state(self.decklist, self.terminated_fn, self.on_the_play, self._rng)
        self._turn_gen = _start_turn(self.state, self.combat_enabled)
        self._cached_mask = None
        return build_observation(self.state, self.decklist, self.horizon, self.pending_kinds), {}

    def step(self, action):
        # Pass sends None -- game.turn._run_turn_gen breaks out of the
        # CURRENT phase's action loop and advances to the next phase (not
        # necessarily the end of the turn -- there are 4-9 phases now, see
        # game.turn.Phase). A real action sends the zero-arg callable
        # itself, for the generator to call (it performs the mana
        # payment/effect and drains the trigger queue, exactly as
        # run_turn's driver does for the harness/eval path) -- this can
        # also end the turn early if the action itself sets state.turn_won
        # (mid-phase, not just at a phase boundary). Either way,
        # StopIteration means every phase has now run its course (the
        # generator fell off the end after Phase.END, or turn_won got set
        # partway through and short-circuited the rest) -- that's the only
        # signal that a whole turn (not just a phase) just finished.
        to_send = _resolve_step_action(self, action)

        game_over = False
        turn_ended = True
        if self._turn_gen is not None:
            try:
                self._turn_gen.send(to_send)
                turn_ended = False
            except StopIteration:
                turn_ended = True

        if turn_ended:
            if self.state.turn_won is not None or self.state.decked_out:
                pass  # nothing else to do this step, done computed below
            elif self.state.turn_number < self.horizon:
                self._turn_gen = _start_turn(self.state, self.combat_enabled)
            else:
                game_over = True  # just finished the final turn -- no more turns left

        done = self.state.turn_won is not None or game_over or self.state.decked_out
        reward = self.reward_fn(self.state, done, self.horizon)
        obs = build_observation(self.state, self.decklist, self.horizon, self.pending_kinds)
        return obs, reward, done, False, {}


# ---------------------------------------------------------------------------
# Two-player training (docs/MULTIPLAYER_ENGINE_PLAN.md's engine layer, wired
# up here as "opponent-as-environment": each side gets an ordinary
# single-agent Gym env whose transition function happens to include the
# opponent's whole turn, played out by a live reference to the OTHER side's
# own SB3 model -- see harness.train_two_player for how two of these are
# built and trained against each other.
# ---------------------------------------------------------------------------

def _lost(state, seat_idx):
    """True once someone has won and it wasn't seat_idx -- the one thing
    every existing 1-player reward_fn (rewards.py) can't tell on its own:
    state.turn_won/turn_number don't say WHO won, only that the game
    ended. A win (state.winner == seat_idx) or "nobody yet" (state.winner
    is None, including the still-in-progress case) both fall through
    unchanged to whatever the wrapped reward_fn would already compute --
    only an actual loss needs to be forced to 0 here."""
    return state.winner is not None and state.winner != seat_idx


def model_choose_action(state, obs, model, actions, pass_action, deterministic=False):
    """One SB3 model's choice, right now, for whichever player currently
    has priority (state.active_idx) -- matches game.turn.run_turn's own
    choose_action(state) contract (None = Pass, else a zero-arg callable),
    same illegal-action fallback rule as harness.TrainingHarness.evaluate's
    inline version and DeckEnv.step (MULTI_DECK_PLAN.md Phase M4e:
    substitute the first currently-legal action, never assume Pass is
    safe). Shared by TwoPlayerDeckEnv (driving the opponent's own turns
    during training) and harness.evaluate_two_player (driving BOTH sides
    at real eval time).

    Takes obs pre-built, rather than building it itself: 1-player and
    2-player observations have different shapes
    (build_observation vs. build_two_player_observation), and only the
    CALLER knows which one the model it's about to call was actually
    trained against -- this function only ever needs to turn "some
    observation" into "an action," never decide which builder applies.

    Auto-substitutes Pass without ever calling model.predict() when the
    mask shows nothing else legal (docs/PRIORITY_PLAN.md item 6 --
    profiled, not guessed: SB3/torch's own predict()/forward-pass/
    distribution-construction is ~40-45% of total training time, the
    single biggest slice, and the priority rewrite calls this function far
    more often per turn than the old declare-blockers-only consult did).
    Provably identical outcome, not a behavior change: whatever the model
    predicted, _substitute_and_resolve would substitute the first legal
    action the instant that prediction wasn't Pass -- and Pass is the
    ONLY legal action here by construction, so the substituted result is
    always Pass regardless of what predict() would have returned."""
    mask = legal_action_mask(state, actions)
    if mask.sum() == 1 and mask[pass_action]:
        return None
    try:
        action, _ = model.predict(obs, action_masks=mask, deterministic=deterministic)
    except TypeError:
        action, _ = model.predict(obs, deterministic=deterministic)
    return _substitute_and_resolve(state, actions, pass_action, mask, int(action))


# Opponent-visibility observation blocks (MULTIPLAYER_GAPS.md's "model's
# observation vector has zero visibility into the opponent" item).
# Everything here is either genuinely public in real Magic (life total,
# battlefield, graveyard, hand/library SIZE -- only hand/library CONTENTS
# are hidden) or, for the one exact-identity exception below, the zone
# where identity plausibly changes tactics now that combat damage is
# real. Aggregate figures are decklist-agnostic by construction -- a
# fixed-size block regardless of which two decklists are paired.
OPPONENT_HAND_SIZE_CAP = 10
OPPONENT_GRAVEYARD_CAP = 15
OPPONENT_BOARD_POWER_CAP = 20
OPPONENT_AGGREGATE_DIM = 8  # my life_total, opponent life_total/hand_size/library_size/non_land_permanents/available_mana/graveyard_size/board_power


def _for_player(state, player_idx, fn):
    """Runs fn(state) with state.active_idx temporarily set to player_idx,
    then restores it -- lets existing active-player-proxied logic
    (rewards.resource_quality_components, game.permanent_power's own
    aura-enchanting search, mana.py's Tron-awareness via state.battlefield,
    ...) be reused for a NON-active player (here: whichever seat's
    OPPONENT this observation is being built for) instead of a second,
    parallel implementation of any of it. Safe even though state.stack/
    pending_resolution are shared, not per-player -- this is only ever
    called between turns (never during a resolution), and every property
    this flip actually affects (hand/battlefield/graveyard/library/
    mana_pool/etc.) is genuinely per-player."""
    original = state.active_idx
    state.active_idx = player_idx
    try:
        return fn(state)
    finally:
        state.active_idx = original


def _opponent_aggregate_features(state, seat_idx, opponent_total_cards):
    """Fixed-size (OPPONENT_AGGREGATE_DIM) block, from seat_idx's own point
    of view -- see this section's own header comment for what's public vs.
    hidden here. available_mana/non_land_permanents reuse rewards.
    resource_quality_components AS-IS (via _for_player) rather than
    recomputing them by hand, so an opponent deck with e.g. Tron-style
    variable mana sources scores correctly with zero duplicated logic.
    opponent_total_cards (the OTHER seat's own total decklist card count)
    is the library-size normalizer -- more principled than an arbitrary
    constant cap, and self-scales for a smaller or larger opponent deck."""
    opponent_idx = 1 - seat_idx
    opponent = state.players[opponent_idx]
    components = _for_player(state, opponent_idx, rewards.resource_quality_components)
    board_power = _for_player(
        state, opponent_idx, lambda s: sum(game.permanent_power(s, p) for p in opponent.battlefield),
    )
    return np.array([
        max(state.players[seat_idx].life_total, 0) / game.state.STARTING_LIFE,
        max(opponent.life_total, 0) / game.state.STARTING_LIFE,
        min(len(opponent.hand), OPPONENT_HAND_SIZE_CAP) / OPPONENT_HAND_SIZE_CAP,
        min(len(opponent.library), opponent_total_cards) / max(opponent_total_cards, 1),
        min(components["non_land_permanents"], rewards.NON_LAND_PERMANENTS_CAP) / rewards.NON_LAND_PERMANENTS_CAP,
        min(components["available_mana"], rewards.AVAILABLE_MANA_CAP) / rewards.AVAILABLE_MANA_CAP,
        min(len(opponent.graveyard), OPPONENT_GRAVEYARD_CAP) / OPPONENT_GRAVEYARD_CAP,
        min(board_power, OPPONENT_BOARD_POWER_CAP) / OPPONENT_BOARD_POWER_CAP,
    ], dtype=np.float32)


def creature_names_and_copies(decklist):
    """Distinct CREATURE names in `decklist`, and their copy counts --
    creatures only (not the whole card pool): the one zone where exact
    identity plausibly changes tactics (a 1/1 vs. a 5/5 changes whether
    racing is correct) now that combat damage is real. Every other
    permanent type stays aggregate-only (already counted in
    _opponent_aggregate_features' own non_land_permanents). Not opponent-
    specific despite the usual call site -- build_two_player_observation
    calls this for BOTH sides (each seat's own creatures are the other
    seat's "opponent creature block")."""
    creature_entries = [
        (name, qty) for name, qty, *_rest in decklist
        if game.CARD_DEFS[name].card_type == game.CardType.CREATURE
    ]
    names = sorted({name for name, _qty in creature_entries})
    copies = dict(creature_entries)
    return names, copies


def _opponent_creature_block(state, opponent_idx, creature_names, creature_copies):
    """The opponent's creatures, at the SAME per-slot fidelity as "my own"
    battlefield (docs/PRIORITY_PLAN.md item 5 -- confirmed gap: this used
    to be a 2-value-per-name aggregate, coarser than what my own side
    already got, and real Magic never hides battlefield state). Just
    _creature_slot_block pointed at the opponent's own seat -- see its
    docstring for what each of the 6 per-slot values means."""
    return _creature_slot_block(state, opponent_idx, creature_names, creature_copies)


def _opponent_stack_block(state, opponent_card_names, opponent_card_copies):
    """Full stack visibility for the OTHER side's own cards (docs/
    PRIORITY_PLAN.md item 5 -- confirmed gap: build_observation's own
    stack section, in build_two_player_observation's `base` block below,
    is keyed only by the CALLING side's own card_names, so any card the
    opponent casts silently vanishes from it -- stack_counts[name] += 1
    only fires `if name in stack_counts`, and the top-of-stack one-hot
    goes all-zero, not even "none", when the top entry is theirs). Same
    stack_counts + top-one-hot-plus-none idiom as build_observation's own
    stack section, just keyed by the other side's card pool (every card
    type, not just creatures -- confirmed Option A: extend the existing
    per-pairing name-indexed pattern rather than a decklist-agnostic
    embedding). Together, the two blocks identify the stack's contents/top
    regardless of whose card is on it."""
    stack_counts = {name: 0 for name in opponent_card_names}
    for entry in state.stack:
        name = entry["card_def"].name
        if name in stack_counts:
            stack_counts[name] += 1
    out = np.zeros(len(opponent_card_names) * 2 + 1, dtype=np.float32)
    i = 0
    for name in opponent_card_names:
        out[i] = stack_counts[name] / opponent_card_copies[name]
        i += 1
    top_name = state.stack[-1]["card_def"].name if state.stack else None
    for name in opponent_card_names:
        out[i] = 1.0 if name == top_name else 0.0
        i += 1
    out[i] = 1.0 if top_name is None else 0.0  # "none" slot
    return out


def two_player_observation_dim(decklist, pending_kinds, opponent_decklist):
    """observation_dim_for's 2-player counterpart -- shared by
    TwoPlayerDeckEnv and harness.py so a 2p observation's dimension is
    computed identically in both places (same reason observation_dim_for
    itself is shared between DeckEnv and harness.py). Only called once per
    env/harness construction (never per-step), so deriving names/copies
    fresh from opponent_decklist here (rather than requiring the caller to
    pre-compute and pass them, the way build_two_player_observation itself
    still does for performance) is the simpler contract with no real cost."""
    opponent_card_names, opponent_card_copies = _card_lookup(opponent_decklist)
    opponent_creature_names, opponent_creature_copies = creature_names_and_copies(opponent_decklist)
    creature_slot_total = sum(opponent_creature_copies[name] for name in opponent_creature_names)
    return (observation_dim_for(decklist, pending_kinds) + OPPONENT_AGGREGATE_DIM
            + creature_slot_total * 6 + len(opponent_card_names) * 2 + 1)


def build_two_player_observation(state, seat_idx, decklist, horizon, pending_kinds,
                                  opponent_total_cards, opponent_creature_names, opponent_creature_copies,
                                  opponent_card_names, opponent_card_copies):
    """The full 2-player observation for `seat_idx`, from ITS OWN point of
    view: build_observation's existing 1-player-shaped block (exactly what
    DeckEnv would build for this decklist/state) plus the opponent-
    visibility blocks (MULTIPLAYER_GAPS.md, extended by docs/PRIORITY_
    PLAN.md item 5) concatenated on top. Shared by TwoPlayerDeckEnv (both
    its own step()/reset() and driving the opponent's own turns) and
    harness.evaluate_two_player (driving BOTH sides at real eval time), so
    every 2p decision is scored against exactly the observation shape its
    own model actually trained on -- model_choose_action itself is
    agnostic to which builder produced its obs argument, so getting this
    part right is what actually matters."""
    base = build_observation(state, decklist, horizon, pending_kinds)
    aggregate = _opponent_aggregate_features(state, seat_idx, opponent_total_cards)
    creatures = _opponent_creature_block(state, 1 - seat_idx, opponent_creature_names, opponent_creature_copies)
    stack = _opponent_stack_block(state, opponent_card_names, opponent_card_copies)
    return np.concatenate([base, aggregate, creatures, stack])


class TwoPlayerDeckEnv(gymnasium.Env):
    """Gym env for ONE seat of a real 2-player game. The opponent's entire
    turn is played out synchronously, inside reset()/step(), by
    self.opponent_model -- a plain mutable attribute (not a constructor
    arg: the opponent's own SB3 model object doesn't exist yet when this
    env is built, see TrainingHarness.__init__/set_opponent_model). It is
    the SAME object the opponent's own TrainingHarness is training, so the
    opponent gets stronger over the course of training with no
    re-wiring needed between bursts -- this is the confirmed
    "opponent-as-environment" design (see docs/MULTIPLAYER_ENGINE_PLAN.md
    for the underlying engine this drives: game.new_multiplayer_game_state/
    game.turn.run_turn).

    combat_enabled is always True here: the config-level trigger for this
    whole mode is "2 decklists present" (run.py), which per that same
    instruction always means every step -- including combat, the only path
    to a life_total win -- is enabled; a 2-player game with combat off
    would leave that whole win condition permanently unreachable.

    horizon: unlike game.turn.run_multiplayer_game's own uncapped design
    (real games there are guaranteed to terminate via deck-out alone),
    this env always takes a plain int -- the same field 1-player configs
    already require -- used both as build_observation's normalization
    denominator (which needs a number, not None) and as a generous safety
    cap. ponytail: simplest option that reuses the existing config
    shape/DeckEnv convention rather than teaching build_observation to
    handle an uncapped horizon; deck-out ends a real game well before any
    reasonable cap in practice.

    shaping_weight/shaping_gamma (MULTIPLAYER_GAPS.md's "reward functions
    are still 1p-shaped" item): an OPT-IN (default 0.0 -- no behavior
    change unless a config asks for it), potential-based dense reward on
    top of whatever reward_fn already returns -- see _shaping_potential's
    own docstring for why this specific shape (Ng et al. 1999) is what
    lets "reward every damage-dealing step" and "still let a win end up
    reinforced more than a loss" coexist, via the training algorithm's own
    advantage estimation rather than anything hand-coded here.
    """

    def __init__(self, reward_fn, decklist, terminated_fn, pending_kinds,
                 opponent_decklist, opponent_terminated_fn, opponent_pending_kinds,
                 my_seat_idx=0, horizon=40, on_the_play=True, seed=None,
                 token_card_defs=(), opponent_token_card_defs=(), opponent_deterministic=False,
                 shaping_weight=0.0, shaping_gamma=0.99):
        super().__init__()
        self.reward_fn = reward_fn
        self.decklist = decklist
        self.terminated_fn = terminated_fn
        self.pending_kinds = pending_kinds
        self.opponent_decklist = opponent_decklist
        self.opponent_terminated_fn = opponent_terminated_fn
        self.opponent_pending_kinds = opponent_pending_kinds
        self.my_seat_idx = my_seat_idx
        self.horizon = horizon
        self.on_the_play = on_the_play  # whether MY seat takes the very first turn
        self.token_card_defs = token_card_defs
        self.opponent_token_card_defs = opponent_token_card_defs
        self.shaping_weight = shaping_weight
        # Matches the actual PPO discount (TrainingHarness passes
        # model_kwargs.get("gamma", 0.99) here) rather than an independent
        # value -- a shaping gamma that disagreed with the model's own
        # discounting would be a subtly different (and unvalidated) notion
        # of "how much the future matters" than the one the model is
        # actually being trained under.
        self.shaping_gamma = shaping_gamma
        # Sampled, not argmaxed, during training rollouts (matches what the
        # opponent's own on-policy training assumes about its action
        # distribution) -- harness.evaluate_two_player passes True instead,
        # same deterministic-at-eval-time convention as the 1-player path.
        # Also governs _own_choose_action (MY OWN model, driven internally
        # by _play_opponent_turn's own dispatching closure whenever I get
        # asked something during the opponent's turn) -- one flag covers
        # both, since either one only ever runs as a simulated decision
        # inside step(), never through the external predict() call the
        # real training/eval loop actually scores.
        self.opponent_deterministic = opponent_deterministic
        self._rng = random.Random(seed)
        self.state = None
        self._turn_gen = None

        # Set post-construction by TrainingHarness.set_opponent_model, once
        # the opponent's own model actually exists. None only briefly (or
        # in a hand-written self-check) -- an opponent with no model just
        # always Passes.
        self.opponent_model = None

        # Set post-construction the same way (TrainingHarness.set_own_model)
        # -- MY OWN model, needed only so _play_opponent_turn can simulate
        # any decision of mine (blocking, responding with an instant) that
        # comes up reactively while the OPPONENT's whole turn is being
        # driven synchronously via game.turn.run_turn (the external SB3
        # caller can't be consulted mid-step()). Confirmed design: passing
        # both models in is fine as long as each stays strictly paired with
        # its own seat's observation (_own_choose_action always builds from
        # self.my_seat_idx, _opponent_choose_action always from the other
        # seat) -- see both methods below. None only briefly, same
        # always-Pass fallback as opponent_model.
        self.own_model = None

        # opponent_decklist/opponent_token_card_defs on BOTH tables below
        # (each pointed at the OTHER side's cards): blocking needs it on
        # both, not just the observation-visibility blocks further down --
        # once either side assigns a blocker (declare_blocker_assignment),
        # completing that choice needs a "Choose opponent's: <name> (slot
        # k)" action addressing the ATTACKER's battlefield, and the
        # attacker is whichever side these actions DON'T otherwise belong
        # to. Without this, that nested resolution has zero legal actions
        # the instant a real blocking consult reaches it -- see
        # docs/COMBAT_PLAN.md's account of this bug, caught by a live
        # smoke test, not any of the narrower unit self-checks below (each
        # of which builds its own action table by hand, correctly, in
        # isolation).
        self.actions = build_action_table(
            decklist, game.EFFECT_REGISTRY, token_card_defs=token_card_defs, pending_kinds=pending_kinds,
            opponent_decklist=opponent_decklist, opponent_token_card_defs=opponent_token_card_defs,
        )
        self.pass_action = next(i for i, (name, _legal, _execute) in enumerate(self.actions) if name == "Pass")
        self.action_space = spaces.Discrete(len(self.actions))

        self.opponent_actions = build_action_table(
            opponent_decklist, game.EFFECT_REGISTRY, token_card_defs=opponent_token_card_defs,
            pending_kinds=opponent_pending_kinds,
            opponent_decklist=decklist, opponent_token_card_defs=token_card_defs,
        )
        self.opponent_pass_action = next(
            i for i, (name, _legal, _execute) in enumerate(self.opponent_actions) if name == "Pass"
        )

        # Opponent-visibility observation blocks (MULTIPLAYER_GAPS.md) --
        # computed for BOTH sides: opponent_* describes the actual
        # opponent (needed for MY OWN observation, _build_observation
        # below), my_* describes MY OWN decklist from the opponent's point
        # of view (needed to drive the opponent's OWN turn in
        # _opponent_choose_action -- their model was trained against a
        # full 2p observation too, not the plain 1p build_observation).
        self.opponent_total_cards = sum(qty for _name, qty, *_rest in opponent_decklist)
        self.opponent_creature_names, self.opponent_creature_copies = creature_names_and_copies(opponent_decklist)
        self.opponent_card_names, self.opponent_card_copies = _card_lookup(opponent_decklist)
        self.my_total_cards = sum(qty for _name, qty, *_rest in decklist)
        self.my_creature_names, self.my_creature_copies = creature_names_and_copies(decklist)
        self.my_card_names, self.my_card_copies = _card_lookup(decklist)

        self.observation_dim = two_player_observation_dim(decklist, pending_kinds, opponent_decklist)
        self.observation_space = spaces.Box(low=0.0, high=1.0, shape=(self.observation_dim,), dtype=np.float32)

        self._cached_mask = None

    def _build_observation(self):
        return build_two_player_observation(
            self.state, self.my_seat_idx, self.decklist, self.horizon, self.pending_kinds,
            self.opponent_total_cards, self.opponent_creature_names, self.opponent_creature_copies,
            self.opponent_card_names, self.opponent_card_copies,
        )

    def _shaping_potential(self):
        """Phi(state): normalized opponent life ALREADY LOST, from MY seat's
        own point of view -- 0.0 at opponent's starting life, rising to
        1.0 once they're dead. Reads state.players[...] directly by index
        (never state.opponent, whose meaning depends on whichever seat is
        CURRENTLY active -- this can be either seat mid-step(), while
        driving the opponent's own turn).

        Deliberately tracks only the opponent's life, not mine (per
        discussion): the reward this shapes is "did I just deal damage,"
        not "am I also in danger" -- that second concern is already fully
        covered by the sparse terminal reward (_lost() forces it to 0 on
        an actual loss), so it doesn't need its own dense term too.

        Used only as a DELTA (gamma*Phi(after) - Phi(before), in step()),
        never as a reward on its own -- an absolute per-step value here
        would reward simply BEING ahead, every single step, which
        incentivizes stalling out a winning position instead of ending
        it. The delta form telescopes to Phi(final) - Phi(initial) across
        a whole episode regardless of how many steps it takes, so there's
        no reward for prolonging a lead, only for extending it -- and per
        Ng, Harada & Russell (1999), a potential-based term shaped this
        way never changes which policy is optimal, only how quickly
        training finds it. Which policy actually gets reinforced more --
        a damage-dealing action in a game that goes on to be won, vs. the
        same action in one that's lost -- isn't decided here at all: it
        falls out of the training algorithm's own advantage estimation
        (the return following a winning trajectory is higher than the
        return following a losing one), the same way it would with no
        shaping at all. See MULTIPLAYER_GAPS.md for the fuller reasoning."""
        opponent_life = max(self.state.players[1 - self.my_seat_idx].life_total, 0)
        return (game.state.STARTING_LIFE - opponent_life) / game.state.STARTING_LIFE

    def _opponent_choose_action(self, state):
        """The opponent's own move -- whether it's structurally their own
        turn (_play_opponent_turn's dispatching closure), or a reactive
        decision of theirs during MY OWN turn (blocking one of my attacks,
        responding to one of my spells -- docs/PRIORITY_PLAN.md's general
        priority round flips state.active_idx to them for exactly as long
        as that takes, then flips it back; _start_turn/_fast_forward feeds
        this method's answers in for the duration). Always builds from
        opponent_seat_idx specifically, never state.active_idx directly --
        the hidden-information guarantee this whole opponent/own split
        depends on."""
        opponent_seat_idx = 1 - self.my_seat_idx
        if self.opponent_model is None:
            # Before set_opponent_model has ever run: Pass if legal (the
            # ordinary case, every non-blocking call site), else the first
            # legal action -- same "never assume Pass is safe" substitution
            # model_choose_action itself applies, needed now that this
            # method can also be invoked mid-"declare_blockers" (Pass is
            # illegal there).
            mask = legal_action_mask(state, self.opponent_actions)
            return _substitute_and_resolve(
                state, self.opponent_actions, self.opponent_pass_action, mask, self.opponent_pass_action,
            )
        obs = build_two_player_observation(
            state, opponent_seat_idx, self.opponent_decklist, self.horizon, self.opponent_pending_kinds,
            self.my_total_cards, self.my_creature_names, self.my_creature_copies,
            self.my_card_names, self.my_card_copies,
        )
        return model_choose_action(
            state, obs, self.opponent_model, self.opponent_actions, self.opponent_pass_action,
            deterministic=self.opponent_deterministic,
        )

    def _own_choose_action(self, state):
        """MY OWN move, driven internally during the OPPONENT's whole turn
        (_play_opponent_turn's dispatching closure calls this whenever
        state.active_idx is my_seat_idx -- blocking one of their attacks,
        responding to one of their spells -- since game.turn.run_turn has
        no other way to reach the external SB3 caller mid-step()). Mirrors
        _opponent_choose_action exactly, just for the other seat:
        always builds from self.my_seat_idx, uses self.own_model/
        self.actions/self.pass_action/self.decklist/self.pending_kinds --
        never anything from the opponent's own side, which is the strict
        seat/model pairing the hidden-information guarantee depends on."""
        if self.own_model is None:
            mask = legal_action_mask(state, self.actions)
            return _substitute_and_resolve(state, self.actions, self.pass_action, mask, self.pass_action)
        obs = build_two_player_observation(
            state, self.my_seat_idx, self.decklist, self.horizon, self.pending_kinds,
            self.opponent_total_cards, self.opponent_creature_names, self.opponent_creature_copies,
            self.opponent_card_names, self.opponent_card_copies,
        )
        return model_choose_action(
            state, obs, self.own_model, self.actions, self.pass_action, deterministic=self.opponent_deterministic,
        )

    def _game_over(self):
        return self.state.turn_won is not None or self.state.turn_number >= self.horizon

    def _play_opponent_turn(self):
        """Runs exactly one opponent turn via game.turn.run_turn (the same
        driver harness.evaluate/generate_regression_snapshot.py already
        use -- state.active_idx is already theirs, set by the caller just
        before this runs), then flips state.active_idx back to my seat if
        the game continues. With exactly 2 players, turns strictly
        alternate -- there is never a second consecutive opponent turn to
        loop for. Mirrors game.turn.run_multiplayer_game's own lazy-flip
        convention (flip right before the NEXT turn starts, not right
        after the current one ends), since nothing else here plays that
        role now that run_multiplayer_game itself isn't driving this
        game (this env hands turns back and forth one at a time instead,
        to interleave the model's own step() calls).

        run_turn's own choose_action(state) is completely agnostic to who
        answers each yield (docs/PRIORITY_PLAN.md) -- this dispatching
        closure is what decides: whenever state.active_idx is my_seat_idx
        (I've been asked something reactively -- blocking their attack,
        responding to their spell), it's my own move (_own_choose_action,
        my own model, simulated internally since the external SB3 caller
        can't be reached again mid-step()); otherwise it's really the
        opponent's own decision (_opponent_choose_action)."""
        def choose_action(state):
            if state.active_idx == self.my_seat_idx:
                return self._own_choose_action(state)
            return self._opponent_choose_action(state)

        game.turn.run_turn(self.state, choose_action, combat_enabled=True)
        if not self._game_over():
            self.state.active_idx = self.my_seat_idx

    def action_masks(self):
        if self.state is None:
            raise RuntimeError("action_masks() called before reset()")
        self._cached_mask = legal_action_mask(self.state, self.actions)
        return self._cached_mask

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        if seed is not None:
            self._rng = random.Random(seed)

        decklists = [None, None]
        terminated_fns = [None, None]
        decklists[self.my_seat_idx] = self.decklist
        decklists[1 - self.my_seat_idx] = self.opponent_decklist
        terminated_fns[self.my_seat_idx] = self.terminated_fn
        terminated_fns[1 - self.my_seat_idx] = self.opponent_terminated_fn
        starting_idx = self.my_seat_idx if self.on_the_play else 1 - self.my_seat_idx

        self.state = game.new_multiplayer_game_state(decklists, terminated_fns, starting_idx, self._rng)
        if self.state.active_idx != self.my_seat_idx:
            self._play_opponent_turn()
        self._cached_mask = None
        self._turn_gen = None if self._game_over() else _start_turn(
            self.state, combat_enabled=True,
            my_seat_idx=self.my_seat_idx, other_seat_choose_action=self._opponent_choose_action,
        )
        return self._build_observation(), {}

    def step(self, action):
        # Captured before anything below mutates self.state -- the "before"
        # half of this step's potential-based shaping delta (see
        # _shaping_potential's own docstring). Skipped entirely at
        # shaping_weight=0.0 (the default): dead weight otherwise, and
        # every existing 2p config predates this feature.
        potential_before = self._shaping_potential() if self.shaping_weight else 0.0

        to_send = _resolve_step_action(self, action)

        turn_ended = True
        if self._turn_gen is not None:
            try:
                self._turn_gen.send(to_send)
                # My own action might have been a Pass, flipping priority to
                # the opponent (docs/PRIORITY_PLAN.md) -- fast-forward
                # through however much of their reaction that provokes
                # (declining/responding, etc.) before handing control back.
                _fast_forward(self._turn_gen, self.state, self.my_seat_idx, self._opponent_choose_action)
                turn_ended = False
            except StopIteration:
                turn_ended = True

        if turn_ended:
            if not self._game_over():
                self.state.active_idx = 1 - self.my_seat_idx  # flip to the opponent for their turn
                self._play_opponent_turn()
            self._turn_gen = None if self._game_over() else _start_turn(
                self.state, combat_enabled=True,
                my_seat_idx=self.my_seat_idx, other_seat_choose_action=self._opponent_choose_action,
            )

        done = self._game_over()
        reward = 0.0 if _lost(self.state, self.my_seat_idx) else self.reward_fn(self.state, done, self.horizon)
        if self.shaping_weight:
            # Unconditional -- NOT gated on _lost() above: the dense term
            # rewards the local act of dealing damage regardless of this
            # particular game's eventual outcome (see _shaping_potential),
            # while the sparse reward_fn term right above it stays exactly
            # as outcome-gated as it always was.
            potential_after = self._shaping_potential()
            reward += self.shaping_weight * (self.shaping_gamma * potential_after - potential_before)
        obs = self._build_observation()
        return obs, reward, done, False, {}


if __name__ == "__main__":
    # ponytail self-check: no pytest in this project, mirrors the
    # assert-based demo convention -- run via `python drl_env.py` from
    # src/. Exercises Plot (MADNESS_DECKS_PLAN.md item 4) and the on-cast
    # trigger hook (item 11) through the REAL _plot_legal/_plot_execute/
    # _cast_from_exile_legal/_cast_from_exile_execute functions -- not a
    # parallel reimplementation. No real Plot/Guttersnipe card exists yet
    # (deck assembly out of scope), so this temporarily injects into the
    # global game.CARD_DEFS/game.EFFECT_REGISTRY, saving/restoring both.
    from game.cards import CardDef, CardType, EffectId
    from game.state import GameState, Permanent, PlayerState

    _card_defs_backup = dict(game.CARD_DEFS)
    _filler_backup = game.EFFECT_REGISTRY[EffectId.FILLER]
    _generous_ent_backup = game.EFFECT_REGISTRY[EffectId.GENEROUS_ENT]

    PLOT_COST = {"generic": 1, "B": 1}  # {B}, not {R} -- EffectId.SWAMP is a real, already-correctly-wired
    # mana source (registry.py's derived views like SIMPLE_MANA_SOURCE_EFFECTS/_FIXED_SOURCE_COLOR are built
    # once at import time; injecting a fake "mana" spec onto FILLER here wouldn't be reflected in them, so the
    # legality pre-check (plan_payment) would wrongly see no valid source -- reusing a real fixed-color land
    # sidesteps that entirely rather than also having to patch the derived views to match).

    on_cast_calls = []
    plot_spell = CardDef("Fake Plot Spell", CardType.SORCERY, PLOT_COST, EffectId.FILLER)
    game.CARD_DEFS["Fake Plot Spell"] = plot_spell
    game.EFFECT_REGISTRY[EffectId.FILLER] = {
        "cast": {"resolve": lambda s, c: None},
        "plot": {"cost": PLOT_COST, "resolve": lambda s, c: (s.hand.remove(c), s.exile.append((c, s.turn_number)))},
    }
    # Guttersnipe stand-in: a permanent whose registry entry has an
    # "on_cast" trigger -- borrows EffectId.GENEROUS_ENT for the duration.
    game.EFFECT_REGISTRY[EffectId.GENEROUS_ENT] = {
        "on_cast": lambda s, permanent: on_cast_calls.append(permanent.card_def.name),
    }
    try:
        state = GameState(on_the_play=True)
        state.phase = game.turn.Phase.MAIN1  # Plot Speed defaults to SORCERY (a CardType.SORCERY card, no override) -- needs a sorcery-speed phase to be legal at all now
        state.hand = [plot_spell]
        state.battlefield = [
            Permanent(CardDef("Swamp", CardType.LAND, None, EffectId.SWAMP)),
            Permanent(CardDef("Swamp", CardType.LAND, None, EffectId.SWAMP)),
            Permanent(CardDef("Guttersnipe-ish", CardType.CREATURE, None, EffectId.GENEROUS_ENT)),
        ]

        # Plot it: pay {1}{B}, exile with this turn's stamp. Both Swamps
        # are needed (1 generic + 1 B); pay_cost is always interactive
        # regardless of what the legality pre-check found, so this taps
        # them one at a time.
        assert _plot_legal("Fake Plot Spell", PLOT_COST, game.turn.Speed.SORCERY)(state)
        _plot_execute("Fake Plot Spell", PLOT_COST, game.EFFECT_REGISTRY[EffectId.FILLER]["plot"]["resolve"])(state)
        assert state.pending_resolution["kind"] == "pay_cost"
        while state.pending_resolution is not None:
            tap_opts = game.tap_cost_options(state)
            if tap_opts:
                name, _color, is_filter = tap_opts[0]
                game.execute_tap_cost_option(state, name, None, is_filter)
            else:
                # Both Swamps produce only B -- the 2nd tap's B floats
                # into the pool instead of auto-filling the outstanding
                # {generic:1} pip (this engine deliberately never
                # auto-spends floated mana toward generic -- mana.py's
                # own documented design). Spend it explicitly.
                game.execute_pool_spend(state, game.pool_spend_options(state)[0])
        assert state.pending_resolution is None
        assert state.hand == []
        assert [c.name for c, _stamp in state.exile] == ["Fake Plot Spell"]
        assert on_cast_calls == []  # plotting itself never fires on_cast -- it isn't casting the spell

        # Same turn: not castable yet ("on a later turn").
        assert not _cast_from_exile_legal("Fake Plot Spell", None, game.turn.Speed.SORCERY)(state)

        # A later turn: castable for free, fires on_cast_trigger (Guttersnipe).
        state.turn_number += 1
        assert _cast_from_exile_legal("Fake Plot Spell", None, game.turn.Speed.SORCERY)(state)
        _cast_from_exile_execute("Fake Plot Spell", game.EFFECT_REGISTRY[EffectId.FILLER]["cast"]["resolve"])(state)
        assert state.exile == []
        assert on_cast_calls == ["Guttersnipe-ish"]

        # extra_legal gate on the cast-from-exile path (Highway Robbery's
        # own need: Plot waives the mana cost, not other costs a normal
        # cast's extra_legal already checks). Re-plot, then simulate an
        # extra_legal that's never satisfiable.
        game.EFFECT_REGISTRY[EffectId.FILLER] = {
            "cast": {"resolve": lambda s, c: None, "extra_legal": lambda s: False},
            "plot": {"cost": PLOT_COST, "resolve": lambda s, c: (s.hand.remove(c), s.exile.append((c, s.turn_number)))},
        }
        state = GameState(on_the_play=True)
        state.phase = game.turn.Phase.MAIN1
        state.hand = [plot_spell]
        state.battlefield = [
            Permanent(CardDef("Swamp", CardType.LAND, None, EffectId.SWAMP)),
            Permanent(CardDef("Swamp", CardType.LAND, None, EffectId.SWAMP)),
        ]
        _plot_execute("Fake Plot Spell", PLOT_COST, game.EFFECT_REGISTRY[EffectId.FILLER]["plot"]["resolve"])(state)
        while state.pending_resolution is not None:
            tap_opts = game.tap_cost_options(state)
            if tap_opts:
                name, _color, is_filter = tap_opts[0]
                game.execute_tap_cost_option(state, name, None, is_filter)
            else:
                game.execute_pool_spend(state, game.pool_spend_options(state)[0])
        state.turn_number += 1
        assert not _cast_from_exile_legal("Fake Plot Spell", game.EFFECT_REGISTRY[EffectId.FILLER]["cast"]["extra_legal"], game.turn.Speed.SORCERY)(state)
    finally:
        game.CARD_DEFS.clear()
        game.CARD_DEFS.update(_card_defs_backup)
        game.EFFECT_REGISTRY[EffectId.FILLER] = _filler_backup
        game.EFFECT_REGISTRY[EffectId.GENEROUS_ENT] = _generous_ent_backup

    print("drl_env.py Plot + on-cast-trigger self-check: OK")

    # Tokens (item 8): build_action_table's token_card_defs param is what
    # actually makes "Activate Blood (sac)" exist as an action at all --
    # "Blood" is never a decklist name, so the plain distinct_names-driven
    # loop alone (used by every other activated ability) can't find it.
    empty_decklist = []
    no_token_actions = build_action_table(empty_decklist, game.EFFECT_REGISTRY)
    assert not any("Blood" in nm for nm, _l, _e in no_token_actions)  # opt-in: omitted => absent, zero effect on existing decks

    token_actions = build_action_table(empty_decklist, game.EFFECT_REGISTRY, token_card_defs=(game.BLOOD_TOKEN_CARD_DEF,))
    activate_name, activate_legal, activate_execute = next(
        (nm, lg, ex) for nm, lg, ex in token_actions if nm == "Activate Blood (sac)"
    )

    state = GameState(on_the_play=True)
    game.create_token(state, game.BLOOD_TOKEN_CARD_DEF)
    state.battlefield.append(Permanent(CardDef("Swamp", CardType.LAND, None, EffectId.SWAMP)))
    state.hand = [CardDef("Card To Discard", CardType.SORCERY, {}, None)]
    state.library = [CardDef("Library Card", CardType.SORCERY, {}, None)]

    assert activate_legal(state)
    activate_execute(state)  # pays {1} via the real begin_pay_cost path, same as every other cost_key ability
    assert state.pending_resolution["kind"] == "pay_cost"
    tap_name, tap_color, tap_filter = game.tap_cost_options(state)[0]
    game.execute_tap_cost_option(state, tap_name, tap_color, tap_filter)
    if state.pending_resolution is not None:
        # Swamp produces B, which floats into the pool instead of
        # auto-filling the {generic:1} need -- same lesson as the Plot
        # check above. Spend it explicitly.
        game.execute_pool_spend(state, game.pool_spend_options(state)[0])

    assert state.pending_resolution["kind"] == "discard"  # Blood's own effect: discard a card
    game.execute_discard_option(state, "Card To Discard")
    assert state.pending_resolution is None
    assert [p.card_def.name for p in state.battlefield] == ["Swamp"]  # Blood is gone, never added to any zone
    assert [c.name for c in state.hand] == ["Library Card"]  # discarded one, drew one

    print("drl_env.py tokens self-check: OK")

    # Combat gating: DeckEnv's wiring of it across the phase sequence
    # (creature_attack_eligible/declare_attacker/combat_damage_step are
    # already self-checked directly in game/effects/combat.py; this exercises
    # the "Attack: <name>" action drl_env itself adds -- MULTIPLAYER_GAPS.md's
    # manual attack declaration). Passing until DECLARE_ATTACKERS, then
    # issuing "Attack: Generous Ent" explicitly (attacking is no longer
    # automatic) walks through every phase (UNTAP/UPKEEP/DRAW/MAIN1/
    # DECLARE_ATTACKERS/DECLARE_BLOCKERS/COMBAT_DAMAGE/MAIN2/END) rather
    # than hardcoding how many Pass steps combat is now behind. Unlike the
    # tokens check above, this can't use empty_decklist -- build_observation
    # indexes battlefield permanents by name against the decklist's own
    # card set, so a synthetic name absent from every decklist would
    # KeyError. Reuses a real Tron creature (Generous Ent) instead,
    # temporarily tagging its real CardDef with "power" (same save/restore
    # convention as every other real-card-borrowing check in this file).
    combat_terminated = lambda s: s.damage_dealt >= 3
    tron_decklist = game.parse_decklist_file(os.path.join(os.path.dirname(__file__), "..", "data", "monster_tron.txt"))
    tron_pending_kinds = game.derive_pending_kinds(tron_decklist)
    ent_card_def = game.CARD_DEFS["Generous Ent"]
    had_power = "power" in ent_card_def.extra
    original_power = ent_card_def.extra.get("power")
    ent_card_def.extra["power"] = 3
    try:
        env = DeckEnv(
            lambda *a: 0.0, decklist=tron_decklist, terminated_fn=combat_terminated,
            pending_kinds=tron_pending_kinds, horizon=5, combat_enabled=True,
        )
        env.reset()
        attacker = Permanent(ent_card_def)
        attacker.summoning_sick = False
        env.state.battlefield = [attacker]

        attack_idx = next(i for i, (nm, _l, _e) in enumerate(env.actions) if nm == "Attack: Generous Ent (slot 1)")
        done = False
        for _ in range(20):  # generous bound -- one turn's full phase sequence is 9 Pass steps at most
            if env.state.phase == game.turn.Phase.DECLARE_ATTACKERS and env.action_masks()[attack_idx]:
                _obs, _reward, done, _truncated, _info = env.step(attack_idx)
            else:
                assert env.action_masks()[env.pass_action]
                _obs, _reward, done, _truncated, _info = env.step(env.pass_action)
            if done:
                break
        assert attacker.tapped  # declared as an attacker via the explicit "Attack: Generous Ent" action, tapped there (not at damage)
        assert env.state.damage_dealt == 3
        assert env.state.turn_won == env.state.turn_number  # combat_damage_step's own terminated_fn check caught it, same as enters_battlefield's
        assert done

        # Same setup, but combat_enabled left at its default (False,
        # matching Tron/spy_combo) -- confirms combat (and the "Attack: X"
        # action existing at all) is opt-in, not accidentally on.
        env2 = DeckEnv(
            lambda *a: 0.0, decklist=tron_decklist, terminated_fn=combat_terminated,
            pending_kinds=tron_pending_kinds, horizon=5,
        )
        env2.reset()
        attacker2 = Permanent(ent_card_def)
        attacker2.summoning_sick = False
        env2.state.battlefield = [attacker2]
        assert not env2.action_masks()[attack_idx]  # never legal -- this deck's phase sequence has no DECLARE_ATTACKERS at all
        env2.step(env2.pass_action)
        assert not attacker2.tapped
        assert env2.state.damage_dealt == 0
        assert env2.state.turn_won is None
    finally:
        if had_power:
            ent_card_def.extra["power"] = original_power
        else:
            del ent_card_def.extra["power"]

    print("drl_env.py combat self-check: OK")

    # Cross-player targeting (docs/COMBAT_PLAN.md): build_action_table's
    # opponent_decklist/opponent_token_card_defs params register "Choose
    # opponent's: X (slot k)" actions from the OTHER side's own card pool
    # -- blocking's first consumer, but exercised standalone here since
    # blocking itself isn't built yet. Boggles on both sides -- what's
    # under test is MY OWN action table's opponent-facing entries, not
    # anything about my own cards, so a real decklist with real creature
    # quantities on both sides is all this needs.
    boggles_decklist = game.parse_decklist_file(os.path.join(os.path.dirname(__file__), "..", "data", "boggles.txt"))

    no_opponent_actions = build_action_table(boggles_decklist, game.EFFECT_REGISTRY)
    assert not any(nm.startswith("Choose opponent's:") for nm, _l, _e in no_opponent_actions)  # 1p mode: never registered at all

    my_actions = build_action_table(boggles_decklist, game.EFFECT_REGISTRY, opponent_decklist=boggles_decklist)
    bogle_slot_actions = [nm for nm, _l, _e in my_actions if nm.startswith("Choose opponent's: Slippery Bogle")]
    assert bogle_slot_actions == [f"Choose opponent's: Slippery Bogle (slot {k})" for k in range(1, 5)]  # boggles.txt: 4 copies
    assert not any(nm.startswith("Choose opponent's: Forest") for nm, _l, _e in my_actions)  # a land, never a targetable creature

    def _midx(action_name):
        return next(i for i, (nm, _l, _e) in enumerate(my_actions) if nm == action_name)

    target_slot_2 = _midx("Choose opponent's: Slippery Bogle (slot 2)")
    target_slot_1 = _midx("Choose opponent's: Slippery Bogle (slot 1)")

    attacker_bogle_1 = Permanent(game.CARD_DEFS["Slippery Bogle"])
    attacker_bogle_2 = Permanent(game.CARD_DEFS["Slippery Bogle"])
    attacker_bogle_2.slot = 2
    state = GameState(on_the_play=True, players=[PlayerState(True), PlayerState(False)])
    state.players[0].battlefield = [attacker_bogle_1, attacker_bogle_2]
    state.active_idx = 1  # simulating the defender's own already-flipped perspective (see game.begin_choose_opponent_permanent's own docstring)

    _, legal_slot_2, execute_slot_2 = my_actions[target_slot_2]
    _, legal_slot_1, _ = my_actions[target_slot_1]
    assert not legal_slot_2(state) and not legal_slot_1(state)  # nothing pending yet

    completed = []
    game.begin_choose_opponent_permanent(
        state, lambda p: p.card_def.card_type == game.CardType.CREATURE, lambda s, choice: completed.append(choice),
    )
    assert legal_slot_1(state) and legal_slot_2(state)
    execute_slot_2(state)
    assert completed == [("Slippery Bogle", 2)]  # the specific slot targeted, not an arbitrary same-named match
    assert not legal_slot_2(state)  # resolution is complete, nothing pending anymore

    print("drl_env.py cross-player targeting self-check: OK")

    # Turn-owner / priority-holder split (docs/PRIORITY_PLAN.md item 0):
    # _land_drop_legal (via speed_legal) and _attack_legal must both
    # refuse the non-turn player even when state.phase/state.
    # lands_played_this_turn/their own eligible creature would otherwise
    # look legal -- simulates a priority consult (active_idx flipped away
    # from turn_player_idx) without needing the full priority round built
    # yet.
    play_forest_idx = _midx("Play land: Forest")
    attack_bogle_idx = _midx("Attack: Slippery Bogle (slot 1)")
    _, play_forest_legal, _ = my_actions[play_forest_idx]
    _, attack_bogle_legal, _ = my_actions[attack_bogle_idx]

    turn_owner_state = GameState(on_the_play=True, players=[PlayerState(True), PlayerState(False)])
    turn_owner_state.turn_player_idx = 0
    turn_owner_state.active_idx = 0
    turn_owner_state.phase = game.turn.Phase.DECLARE_ATTACKERS
    attacking_bogle = Permanent(game.CARD_DEFS["Slippery Bogle"])
    attacking_bogle.summoning_sick = False
    turn_owner_state.players[0].hand = [game.CARD_DEFS["Forest"]]
    turn_owner_state.players[0].battlefield = [attacking_bogle]
    assert attack_bogle_legal(turn_owner_state)  # the turn player's own creature, their own DECLARE_ATTACKERS -- legal

    turn_owner_state.phase = game.turn.Phase.MAIN1
    assert play_forest_legal(turn_owner_state)  # the turn player's own MAIN1, land in hand, none played yet -- legal

    turn_owner_state.active_idx = 1  # simulating a priority consult of the OTHER player
    turn_owner_state.players[1].hand = [game.CARD_DEFS["Forest"]]  # even with their OWN land available
    assert not play_forest_legal(turn_owner_state)  # refused -- not their turn, regardless of their own hand/lands_played_this_turn

    turn_owner_state.phase = game.turn.Phase.DECLARE_ATTACKERS
    non_turn_bogle = Permanent(game.CARD_DEFS["Slippery Bogle"])
    non_turn_bogle.summoning_sick = False
    turn_owner_state.players[1].battlefield = [non_turn_bogle]  # even with their OWN eligible creature at the same (name, slot)
    assert not attack_bogle_legal(turn_owner_state)  # refused -- declaring attackers is the turn player's own special action

    print("drl_env.py turn-owner (land drop / declare attacker) self-check: OK")

    # "Am I the turn player right now" observation signal (docs/PRIORITY_
    # PLAN.md item 5): most reactive priority windows share pending_kind
    # == "none" with an ordinary proactive decision, so this is the only
    # thing in the observation that actually distinguishes them.
    turn_owner_pending_kinds = game.derive_pending_kinds(boggles_decklist)
    turn_owner_n = len(_card_lookup(boggles_decklist)[0])
    turn_owner_state.active_idx = turn_owner_state.turn_player_idx = 0
    obs_mine = build_observation(turn_owner_state, boggles_decklist, horizon=10, pending_kinds=turn_owner_pending_kinds)
    assert obs_mine[turn_owner_n * 4 + 2] == 1.0  # I hold priority AND it's my own turn
    turn_owner_state.active_idx = 1  # a reactive consult -- still turn_player_idx == 0's own turn
    obs_reactive = build_observation(
        turn_owner_state, boggles_decklist, horizon=10, pending_kinds=turn_owner_pending_kinds,
    )
    assert obs_reactive[turn_owner_n * 4 + 2] == 0.0

    # Blocking (docs/COMBAT_PLAN.md): build_action_table's "Assign Blocker:
    # <name> (slot j)" / "Done blocking" entries, end to end through the
    # REAL production functions (_assign_blocker_legal/_execute,
    # _done_blocking_legal/_execute) -- not a parallel reimplementation.
    # Two attacking Slippery Bogles (real power=1 stats), one defending
    # Slippery Bogle blocks only ONE of them.
    boggles_pending_kinds = game.derive_pending_kinds(boggles_decklist)
    assign_slot_1 = _midx("Assign Blocker: Slippery Bogle (slot 1)")
    done_blocking_idx = _midx("Done blocking")
    _, assign_legal, assign_execute = my_actions[assign_slot_1]
    _, done_legal, done_execute = my_actions[done_blocking_idx]

    atk_bogle_1 = Permanent(game.CARD_DEFS["Slippery Bogle"])
    atk_bogle_2 = Permanent(game.CARD_DEFS["Slippery Bogle"])
    atk_bogle_2.slot = 2
    defender_bogle = Permanent(game.CARD_DEFS["Slippery Bogle"])  # slot 1 by default
    state = GameState(on_the_play=True, players=[PlayerState(True), PlayerState(False)])
    state.players[0].battlefield = [atk_bogle_1, atk_bogle_2]
    state.players[0].attackers = [atk_bogle_1, atk_bogle_2]
    atk_bogle_1.tapped = True
    atk_bogle_2.tapped = True  # declare_attacker's own effect -- simulated directly, attacking itself isn't under test here
    state.players[1].battlefield = [defender_bogle]
    state.active_idx = 1  # simulating _declare_blockers_gen's own flip to the defender

    assert not assign_legal(state) and not done_legal(state)  # nothing pending yet

    completed = []
    game.begin_declare_blockers(state, on_complete=lambda s: completed.append(True))
    assert completed == []
    assert assign_legal(state) and done_legal(state)

    assign_execute(state)  # "Assign Blocker: Slippery Bogle (slot 1)" -- parks defender_bogle as a blocker
    assert state.pending_resolution["kind"] == "choose_opponent_permanent"
    target_slot_1 = _midx("Choose opponent's: Slippery Bogle (slot 1)")
    _, target_legal, target_execute = my_actions[target_slot_1]
    assert target_legal(state)
    target_execute(state)  # assigns it to block atk_bogle_1 specifically, not atk_bogle_2

    # Re-opened (drl_env._assign_blocker_execute's own nested on_complete):
    # defender_bogle is now spoken for, so it's no longer offered again --
    # confirms creature_block_eligible actually gates the SAME action a
    # second time, not just the first.
    assert state.pending_resolution["kind"] == "declare_blockers"
    assert not assign_legal(state)
    assert done_legal(state)
    done_execute(state)  # "Done blocking" -- atk_bogle_2 goes unblocked
    assert completed == [True]
    assert state.pending_resolution is None
    assert state.players[0].blocked_by == {atk_bogle_1: defender_bogle}

    # Attacker-side observation (build_observation's per-creature block,
    # 5th value): from the attacker's own perspective, atk_bogle_1 reads
    # blocked=1.0, atk_bogle_2 (still unblocked) reads blocked=0.0.
    def _creature_slot_values(obs, decklist, pending_kinds, name, slot):
        card_names, card_copies = _card_lookup(decklist)
        creature_names_ordered = [n for n in card_names if game.CARD_DEFS[n].card_type == game.CardType.CREATURE]
        idx = (len(card_names) * 4 + 2 + len(_all_pending_kinds(pending_kinds))
               + len(game.POOL_COLORS) + len(game.turn.Phase)
               + len(card_names) + (len(card_names) + 1) + 1 + 1)
        for n in creature_names_ordered:
            if n == name:
                return obs[idx + 6 * (slot - 1):idx + 6 * slot]
            idx += 6 * card_copies[n]
        raise ValueError(name)

    state.active_idx = 0  # back to the attacker's own perspective
    obs = build_observation(state, boggles_decklist, horizon=10, pending_kinds=boggles_pending_kinds)
    untapped_1, tapped_1, power_1, _toughness_1, blocked_1, committed_1 = _creature_slot_values(
        obs, boggles_decklist, boggles_pending_kinds, "Slippery Bogle", 1,
    )
    _untapped_2, _tapped_2, _power_2, _toughness_2, blocked_2, committed_2 = _creature_slot_values(
        obs, boggles_decklist, boggles_pending_kinds, "Slippery Bogle", 2,
    )
    assert tapped_1 == 1.0 and untapped_1 == 0.0  # declare_attacker's own tap, simulated above
    assert power_1 == 1 / PER_CREATURE_POWER_CAP
    assert blocked_1 == 1.0  # atk_bogle_1 -- the one actually blocked
    assert blocked_2 == 0.0  # atk_bogle_2 -- declared, but nothing blocked it
    assert committed_1 == 0.0 and committed_2 == 0.0  # neither is a defender's own committed blocker here

    print("drl_env.py blocking self-check: OK")

    # Flying (docs/COMBAT_PLAN.md step 7): _assign_blocker_execute's own
    # extra_predicate (game.has_keyword), end to end through the REAL
    # action table -- Silhana Ledgewalker (real "can't be blocked except
    # by creatures with flying," modeled as the "flying" keyword) can only
    # be blocked by a creature that itself has flying (Kitchen Imp, real
    # flying) -- a plain Slippery Bogle is otherwise a perfectly legal
    # (untapped) blocker, but can never be assigned to THIS specific
    # attacker. Mixes cards from different real color catalogs
    # (green/multicolor + black) purely to exercise the engine mechanism
    # -- not a claim either card is ever actually run together in a real
    # deck.
    flying_decklist = [("Silhana Ledgewalker", 2), ("Slippery Bogle", 2), ("Kitchen Imp", 2)]
    flying_actions = build_action_table(flying_decklist, game.EFFECT_REGISTRY, opponent_decklist=flying_decklist)

    def _fidx(action_name):
        return next(i for i, (nm, _l, _e) in enumerate(flying_actions) if nm == action_name)

    _, bogle_legal, bogle_execute = flying_actions[_fidx("Assign Blocker: Slippery Bogle (slot 1)")]
    _, imp_legal, imp_execute = flying_actions[_fidx("Assign Blocker: Kitchen Imp (slot 1)")]

    attacking_ledgewalker = Permanent(game.CARD_DEFS["Silhana Ledgewalker"])
    attacking_ledgewalker.tapped = True  # already attacked
    defending_bogle = Permanent(game.CARD_DEFS["Slippery Bogle"])
    defending_imp = Permanent(game.CARD_DEFS["Kitchen Imp"])
    state = GameState(on_the_play=True, players=[PlayerState(True), PlayerState(False)])
    state.players[0].battlefield = [attacking_ledgewalker]
    state.players[0].attackers = [attacking_ledgewalker]
    state.players[1].battlefield = [defending_bogle, defending_imp]
    state.active_idx = 1  # simulating _declare_blockers_gen's own flip to the defender

    game.begin_declare_blockers(state, on_complete=lambda s: None)
    assert bogle_legal(state) and imp_legal(state)  # both otherwise-eligible blockers (untapped, unused)

    bogle_execute(state)  # parks the Bogle -- but it can't legally block a flyer, so this fizzles
    assert state.pending_resolution["kind"] == "declare_blockers"  # re-opened, nothing left pending
    assert state.players[0].blocked_by == {}  # nothing assigned -- the Bogle was never a legal choice for this attacker

    imp_execute(state)  # Kitchen Imp HAS flying -- opens a real nested choice
    assert state.pending_resolution["kind"] == "choose_opponent_permanent"
    assert game.choose_opponent_permanent_options(state) == [("Silhana Ledgewalker", 1)]
    game.execute_choose_opponent_permanent_option(state, "Silhana Ledgewalker", 1)
    assert state.players[0].blocked_by == {attacking_ledgewalker: defending_imp}

    print("drl_env.py flying self-check: OK")

    # Targeting (real MTG rule, per drl_env._precast_choice_execute /
    # game.effects.casting.cast_aura's own docstrings): a target is chosen once,
    # at cast time, exact (name, slot) addressed -- not just by name -- and
    # re-validated by identity only once the spell resolves off the stack.
    # End to end through the REAL action table: "Cast Rancor" pays its {G}
    # cost, then (precast_choice, not deferred) immediately opens
    # choose_permanent with BOTH Slippery Bogles offered by their own
    # distinct slot; "Choose target: Slippery Bogle (slot 2)" picks the
    # specific one, which pushes to the stack (not yet attached, still in
    # hand); resolving the stack attaches it to exactly the one chosen, not
    # an arbitrary same-named match.
    targeting_decklist = [("Slippery Bogle", 2), ("Rancor", 2), ("Forest", 10)]
    targeting_actions = build_action_table(targeting_decklist, game.EFFECT_REGISTRY)

    def _gidx(action_name):
        return next(i for i, (nm, _l, _e) in enumerate(targeting_actions) if nm == action_name)

    bogle_1 = Permanent(game.CARD_DEFS["Slippery Bogle"])
    bogle_2 = Permanent(game.CARD_DEFS["Slippery Bogle"])
    bogle_2.slot = 2
    forest = Permanent(game.CARD_DEFS["Forest"])
    rancor_card = game.CARD_DEFS["Rancor"]
    state = GameState(on_the_play=True)
    state.phase = game.turn.Phase.MAIN1  # sorcery-speed cast requires this -- GameState defaults phase=None
    state.battlefield = [bogle_1, bogle_2, forest]
    state.hand = [rancor_card]

    _, cast_rancor_legal, cast_rancor_execute = targeting_actions[_gidx("Cast Rancor")]
    assert cast_rancor_legal(state)
    cast_rancor_execute(state)
    assert state.pending_resolution["kind"] == "pay_cost"
    targeting_actions[_gidx("Choose: Forest")][2](state)  # tap the Forest -- floats {G}
    targeting_actions[_gidx("Spend G from pool")][2](state)  # pays Rancor's {G}

    # Cost fully paid -- precast_choice means cast_aura runs its target
    # choice IMMEDIATELY here, NOT deferred to when this eventually pops
    # off the stack (that's the whole point of this redesign).
    assert state.pending_resolution["kind"] == "choose_permanent"
    assert set(game.choose_permanent_options(state)) == {("Slippery Bogle", 1), ("Slippery Bogle", 2)}

    _, choose_slot_2_legal, choose_slot_2_execute = targeting_actions[_gidx("Choose target: Slippery Bogle (slot 2)")]
    assert choose_slot_2_legal(state)
    choose_slot_2_execute(state)
    # Target chosen -- pushed to the stack, not yet attached (still
    # physically in hand, same "still in hand while on stack" convention
    # every other cast path here follows).
    assert state.pending_resolution is None
    assert state.hand == [rancor_card] and len(state.stack) == 1

    game.resolve_top_of_stack(state)
    assert state.hand == []
    rancor_permanent = next(p for p in state.battlefield if p.card_def.name == "Rancor")
    assert rancor_permanent.flags["enchanting"] is bogle_2  # the SPECIFIC one chosen -- not bogle_1, despite the identical name

    print("drl_env.py Aura targeting (exact slot addressing) self-check: OK")

    # Fizzle, same end-to-end path: the exact chosen permanent (bogle_1
    # this time) is gone by the time the cast resolves -- the whole spell
    # fails, no effect, straight to the graveyard, never attaches.
    state = GameState(on_the_play=True)
    state.phase = game.turn.Phase.MAIN1
    bogle_1 = Permanent(game.CARD_DEFS["Slippery Bogle"])
    forest = Permanent(game.CARD_DEFS["Forest"])
    state.battlefield = [bogle_1, forest]
    state.hand = [rancor_card]

    targeting_actions[_gidx("Cast Rancor")][2](state)
    targeting_actions[_gidx("Choose: Forest")][2](state)
    targeting_actions[_gidx("Spend G from pool")][2](state)
    targeting_actions[_gidx("Choose target: Slippery Bogle (slot 1)")][2](state)
    assert len(state.stack) == 1
    state.battlefield.remove(bogle_1)  # dies before the cast resolves

    game.resolve_top_of_stack(state)
    assert state.hand == []
    assert rancor_card in state.graveyard
    assert not any(p.card_def.name == "Rancor" for p in state.battlefield)

    print("drl_env.py Aura target-fizzle (end to end) self-check: OK")

    # Stack: casting a spell defers its effect (state.stack) instead of
    # resolving immediately; "Pass" resolves one entry at a time (LIFO)
    # instead of advancing the phase while the stack is non-empty; sorcery
    # speed (and playing a land) is illegal while anything sits on it.
    # Uses the real mono_red_madness decklist (Lightning Bolt is a plain
    # instant cast; Fireblast's alt-cost is the sacrifice-gated case).
    mono_red_decklist = game.parse_decklist_file(
        os.path.join(os.path.dirname(__file__), "..", "data", "mono_red_madness.txt")
    )
    mono_red_pending_kinds = game.derive_pending_kinds(mono_red_decklist)
    card_names, card_copies = _card_lookup(mono_red_decklist)

    env = DeckEnv(
        lambda *a: 0.0, decklist=mono_red_decklist, terminated_fn=lambda s: False,
        pending_kinds=mono_red_pending_kinds, horizon=5,
    )
    env.reset()
    # Untap never opens a priority round at all (rule 4), so reset()
    # itself already lands on the first phase that does: DRAW. One Pass
    # (a 1-player priority round ends after a single pass -- len(players)
    # == 1) advances it to MAIN1.
    assert env.state.phase == game.turn.Phase.DRAW
    env.step(env.pass_action)  # DRAW -> MAIN1
    assert env.state.phase == game.turn.Phase.MAIN1

    def _idx(action_name):
        return next(i for i, (nm, _l, _e) in enumerate(env.actions) if nm == action_name)

    cast_bolt, choose_mountain, play_mountain = _idx("Cast Lightning Bolt"), _idx("Choose: Mountain"), _idx("Play land: Mountain")

    bolt = game.CARD_DEFS["Lightning Bolt"]
    mountain = game.CARD_DEFS["Mountain"]
    env.state.hand = [bolt, bolt, mountain]
    env.state.battlefield = [Permanent(mountain), Permanent(mountain)]
    env.state.graveyard = []
    env.state.damage_dealt = 0

    spend_r = _idx("Spend R from pool")

    # Cast bolt #1: mana paid -> pushed to the stack, NOT resolved yet.
    # Pool-only model (MANA_POOL_PLAN.md): tapping the Mountain only
    # floats {R} into the pool -- an explicit spend actually pays it.
    assert env.action_masks()[cast_bolt]
    env.step(cast_bolt)
    assert env.state.pending_resolution["kind"] == "pay_cost"
    assert env.action_masks()[choose_mountain]
    env.step(choose_mountain)
    assert env.state.pending_resolution is not None
    assert env.action_masks()[spend_r]
    env.step(spend_r)
    assert env.state.pending_resolution is None
    assert len(env.state.stack) == 1 and env.state.stack[0]["card_def"] is bolt
    assert env.state.damage_dealt == 0  # not resolved yet
    assert [c.name for c in env.state.hand] == ["Lightning Bolt", "Lightning Bolt", "Mountain"]  # hand.remove is deferred to actual resolution

    # Sorcery speed (and a land drop) is illegal while the stack is non-empty.
    assert not env.action_masks()[play_mountain]

    # Observation reflects the stack: the already-stacked copy no longer
    # reads as "available" in the hand block (matches the action mask,
    # which is exactly why _hand_count_available exists), and shows up in
    # the new stack_counts/stack_top blocks instead.
    obs = build_observation(env.state, mono_red_decklist, env.horizon, mono_red_pending_kinds)
    assert len(obs) == env.observation_dim == observation_dim_for(mono_red_decklist, mono_red_pending_kinds)
    n = len(card_names)
    all_kinds_len = len(_all_pending_kinds(mono_red_pending_kinds))
    # +3 scalars ahead of the pending_kind block: turn_number,
    # lands_played_this_turn, and "am I the turn player" (docs/PRIORITY_
    # PLAN.md item 5).
    stack_counts_start = n * 4 + 3 + all_kinds_len + len(game.POOL_COLORS) + len(game.turn.Phase)
    stack_top_start = stack_counts_start + n
    stack_none_idx = stack_top_start + n
    stack_depth_idx = stack_none_idx + 1
    creature_slot_dim = sum(
        card_copies[name] * 6 for name in card_names if game.CARD_DEFS[name].card_type == game.CardType.CREATURE
    )
    assert stack_depth_idx == len(obs) - 1 - creature_slot_dim
    bolt_i = card_names.index("Lightning Bolt")
    assert abs(obs[bolt_i] - 1 / card_copies["Lightning Bolt"]) < 1e-6  # hand block: only 1 of 2 physical bolts still "available" (1 already on the stack)
    assert abs(obs[stack_counts_start + bolt_i] - 1 / card_copies["Lightning Bolt"]) < 1e-6
    assert obs[stack_top_start + bolt_i] == 1.0  # Lightning Bolt is the (only, so top) stack entry
    assert obs[stack_none_idx] == 0.0
    assert abs(obs[stack_depth_idx] - 1 / STACK_DEPTH_CAP) < 1e-6

    # Cast bolt #2 (the 2nd physical copy -- still legal even though the
    # first copy is still phantom-present in state.hand, since
    # _hand_count_available correctly discounts the one already stacked).
    assert env.action_masks()[cast_bolt]
    env.step(cast_bolt)
    env.step(choose_mountain)
    env.step(spend_r)
    assert env.state.pending_resolution is None
    assert len(env.state.stack) == 2
    # A third cast is illegal now: both physical copies are already stacked
    # (0 available), independent of mana (both Mountains are also tapped).
    assert not env.action_masks()[cast_bolt]

    # "Pass" while the stack is non-empty resolves exactly one entry (LIFO)
    # instead of advancing the phase.
    env.step(env.pass_action)
    assert env.state.phase == game.turn.Phase.MAIN1  # did NOT advance
    assert len(env.state.stack) == 1
    assert env.state.damage_dealt == 3
    assert [c.name for c in env.state.hand] == ["Lightning Bolt", "Mountain"]  # exactly one bolt's own resolve has now run

    env.step(env.pass_action)
    assert env.state.phase == game.turn.Phase.MAIN1  # still didn't advance -- stack was non-empty at the start of this Pass too
    assert len(env.state.stack) == 0
    assert env.state.damage_dealt == 6
    assert [c.name for c in env.state.hand] == ["Mountain"]
    assert env.action_masks()[play_mountain]  # legal again now that the stack is empty

    env.step(env.pass_action)
    assert env.state.phase != game.turn.Phase.MAIN1  # empty stack -- Pass finally advances the phase

    # Fireblast's alt-cost (sacrifice 2 Mountains instead of paying mana):
    # the stack must stay EMPTY for the whole sacrifice resolution -- only
    # once the alt cost is actually paid does the (pure damage) effect get
    # pushed, confirming the cost/effect split (game.catalog.red_cards.
    # cast_fireblast_alt) actually defers only the effect, not the cost.
    # Still mid-turn (now in END, per the phase-advance just above) --
    # irrelevant here since Fireblast's alt_cast is Speed.INSTANT, legal in
    # any phase; the stack-push/resolve mechanics being exercised are the
    # same in every phase's action loop, not phase-specific.
    fireblast = game.CARD_DEFS["Fireblast"]
    env.state.hand = [fireblast]
    env.state.battlefield = [Permanent(mountain), Permanent(mountain)]
    env.state.graveyard = []
    env.state.damage_dealt = 0

    cast_fireblast_alt = _idx("Cast Fireblast (free)")
    assert env.action_masks()[cast_fireblast_alt]
    env.step(cast_fireblast_alt)
    assert env.state.pending_resolution["kind"] == "sacrifice"
    assert env.state.stack == []  # cost not yet paid -- nothing pushed
    assert env.state.hand == []  # already left hand the instant Fireblast's alt cost began (real-rules cast timing)
    assert [c.name for c in env.state.graveyard] == ["Fireblast"]

    env.step(choose_mountain)  # sacrifice Mountain #1 of 2 -- still not fully paid
    assert env.state.pending_resolution["kind"] == "sacrifice"
    assert env.state.stack == []

    env.step(choose_mountain)  # sacrifice Mountain #2 -- alt cost now fully paid
    assert env.state.pending_resolution is None
    assert len(env.state.stack) == 1 and env.state.stack[0]["card_def"] is fireblast
    assert env.state.damage_dealt == 0  # not resolved yet

    env.step(env.pass_action)
    assert env.state.stack == []
    assert env.state.damage_dealt == 4

    print("drl_env.py stack self-check: OK")

    # -- TwoPlayerDeckEnv: opponent-as-environment, driven through the real
    # Gym step() interface (not game.turn.run_multiplayer_game directly --
    # that's already self-checked in turn.py). Player 0 (Mountain +
    # Lightning Bolt, same real cards turn.py's own 2p self-check uses)
    # races a pure-Mountain punching-bag opponent to 0 life_total, entirely
    # through env.step(action_index) calls -- proves reset()/step() alternate
    # active_idx correctly, drive the opponent's whole turn via a stand-in
    # "model" (only .predict matters, no real SB3 dependency needed here),
    # and zero the loser's reward via _lost.
    import rewards

    class _AlwaysPassModel:
        """Stand-in for an SB3 model: always Passes (mirrors turn.py's own
        _burn_policy's punching-bag opponent, which always returns None) --
        model_choose_action's own fallback substitutes a legal action if
        Pass itself somehow isn't (never the case for a player with no
        lands and no hand ever gaining mana)."""

        def __init__(self, pass_action):
            self.pass_action = pass_action

        def predict(self, obs, action_masks=None, deterministic=False):
            return self.pass_action, None

    assert _lost(type("S", (), {"winner": 1})(), 0) is True
    assert _lost(type("S", (), {"winner": 0})(), 0) is False
    assert _lost(type("S", (), {"winner": None})(), 0) is False

    two_p_my_decklist = [("Mountain", 20), ("Lightning Bolt", 10)]
    two_p_opp_decklist = [("Mountain", 20)]
    two_p_my_pending = game.derive_pending_kinds(two_p_my_decklist)
    two_p_opp_pending = game.derive_pending_kinds(two_p_opp_decklist)

    two_p_env = TwoPlayerDeckEnv(
        rewards.strict_binary_reward, decklist=two_p_my_decklist, terminated_fn=lambda s: False,
        pending_kinds=two_p_my_pending, opponent_decklist=two_p_opp_decklist,
        opponent_terminated_fn=lambda s: False, opponent_pending_kinds=two_p_opp_pending,
        my_seat_idx=0, horizon=60, on_the_play=True, seed=0,
    )
    two_p_env.opponent_model = _AlwaysPassModel(two_p_env.opponent_pass_action)
    two_p_env.reset()
    assert two_p_env.state.active_idx == 0  # on_the_play -- I go first

    def _tidx(name):
        return next(i for i, (nm, _l, _e) in enumerate(two_p_env.actions) if nm == name)

    two_p_play_mountain, two_p_cast_bolt, two_p_choose_mountain = (
        _tidx("Play land: Mountain"), _tidx("Cast Lightning Bolt"), _tidx("Choose: Mountain"),
    )

    two_p_done = False
    two_p_reward = None
    for _ in range(800):  # generous bound -- see turn.py's own real-game self-check for the same lethal-well-before-any-cap expectation
        two_p_mask = two_p_env.action_masks()
        if two_p_mask[two_p_play_mountain] and two_p_env.state.lands_played_this_turn == 0:
            two_p_action = two_p_play_mountain
        elif (two_p_env.state.pending_resolution is not None
              and two_p_env.state.pending_resolution["kind"] == "pay_cost"):
            two_p_action = two_p_choose_mountain
        elif two_p_mask[two_p_cast_bolt]:
            two_p_action = two_p_cast_bolt
        else:
            two_p_action = two_p_env.pass_action
        _obs, two_p_reward, two_p_done, _truncated, _info = two_p_env.step(two_p_action)
        if two_p_done:
            break

    assert two_p_done
    assert two_p_env.state.winner == 0  # I (Bolt deck) won
    assert two_p_env.state.players[1].life_total <= 0
    assert two_p_reward > 0.0  # strict_binary_reward's real value, NOT zeroed -- I'm the winner, _lost(state, 0) is False
    print(
        f"drl_env.py TwoPlayerDeckEnv self-check: OK (opponent dead on turn {two_p_env.state.turn_number}, "
        f"life_total={two_p_env.state.players[1].life_total}, reward={two_p_reward:.4f})"
    )

    # Opponent-visibility observation blocks (MULTIPLAYER_GAPS.md): the env
    # self-check above never puts a creature in the punching-bag opponent's
    # deck, so it alone never exercises the non-empty creature-block case --
    # a direct, synthetic check of _opponent_creature_block/
    # _opponent_aggregate_features instead. Confirms two_p_env's own
    # observation_dim actually grew by the right amount too.
    assert two_p_env.opponent_creature_names == []  # two_p_opp_decklist (Mountain only) has no creatures
    assert two_p_env.observation_dim == two_player_observation_dim(two_p_my_decklist, two_p_my_pending, two_p_opp_decklist)

    opp_creature_decklist = [("Mountain", 4), ("Generous Ent", 2)]  # a real Tron creature -- same one this file's own combat self-check borrows
    opp_creature_names, opp_creature_copies = creature_names_and_copies(opp_creature_decklist)
    assert opp_creature_names == ["Generous Ent"] and opp_creature_copies == {"Generous Ent": 2}

    ent_def = game.CARD_DEFS["Generous Ent"]
    had_power = "power" in ent_def.extra
    original_power = ent_def.extra.get("power")
    ent_def.extra["power"] = 3
    try:
        synth_state = GameState(
            on_the_play=True, players=[PlayerState(on_the_play=True), PlayerState(on_the_play=False)],
        )
        # docs/PRIORITY_PLAN.md item 5: the opponent's creatures now get the
        # SAME 6-value-per-slot fidelity as my own side (untapped, tapped,
        # power, remaining toughness, blocked-as-attacker, committed-as-
        # blocker) -- distinct slots (Permanent defaults to slot 1) so both
        # physical copies actually show up, same precedent as the
        # attacker-side per-slot self-checks elsewhere in this file.
        ent_1 = Permanent(ent_def, tapped=False)
        ent_2 = Permanent(ent_def, tapped=True)
        ent_2.slot = 2
        ent_2.damage_marked = 2  # toughness 5 - 2 damage = 3 remaining
        fake_attacker = Permanent(game.CARD_DEFS["Mountain"])  # stands in for one of MY attackers -- only used as a blocked_by key/value here
        synth_state.players[1].battlefield = [ent_1, ent_2]
        synth_state.players[1].blocked_by = {ent_1: fake_attacker}  # ent_1 attacked me and got blocked
        synth_state.players[0].blocked_by = {fake_attacker: ent_2}  # ent_2 is already committed blocking one of MY attackers

        creature_block = _opponent_creature_block(synth_state, 1, opp_creature_names, opp_creature_copies)
        assert creature_block.shape == (12,)  # 2 slots * 6 values
        # slot 1 (ent_1): untapped, power 3, toughness 5 (undamaged), blocked-as-attacker, not committed
        assert abs(creature_block[0] - 1.0) < 1e-6
        assert abs(creature_block[1] - 0.0) < 1e-6
        assert abs(creature_block[2] - 3 / PER_CREATURE_POWER_CAP) < 1e-6
        assert abs(creature_block[3] - 5 / PER_CREATURE_TOUGHNESS_CAP) < 1e-6
        assert creature_block[4] == 1.0
        assert creature_block[5] == 0.0
        # slot 2 (ent_2): tapped, power 3, remaining toughness 3 (5 - 2 damage), not blocked, committed-as-blocker
        assert abs(creature_block[6] - 0.0) < 1e-6
        assert abs(creature_block[7] - 1.0) < 1e-6
        assert abs(creature_block[8] - 3 / PER_CREATURE_POWER_CAP) < 1e-6
        assert abs(creature_block[9] - 3 / PER_CREATURE_TOUGHNESS_CAP) < 1e-6
        assert creature_block[10] == 0.0
        assert creature_block[11] == 1.0

        synth_state.players[1].blocked_by = {}
        synth_state.players[0].blocked_by = {}
        synth_state.players[0].life_total = 20
        synth_state.players[1].life_total = 15
        synth_state.players[1].hand = [CardDef("X", CardType.SORCERY, {}, None)] * 3
        synth_state.players[1].library = [CardDef("Y", CardType.LAND, None, None)] * 10
        synth_state.players[1].graveyard = [CardDef("Z", CardType.SORCERY, {}, None)] * 2

        agg = _opponent_aggregate_features(synth_state, seat_idx=0, opponent_total_cards=20)
        assert agg.shape == (OPPONENT_AGGREGATE_DIM,)
        assert abs(agg[0] - 1.0) < 1e-6                                    # my own life_total: 20/20
        assert abs(agg[1] - 0.75) < 1e-6                                   # opponent life_total: 15/20
        assert abs(agg[2] - 3 / OPPONENT_HAND_SIZE_CAP) < 1e-6             # opponent hand size
        assert abs(agg[3] - 10 / 20) < 1e-6                                # opponent library size / total deck size
        assert abs(agg[4] - 2 / rewards.NON_LAND_PERMANENTS_CAP) < 1e-6    # 2 Generous Ents, both non-land
        assert agg[5] == 0.0                                               # no mana-producing permanents on this board
        assert abs(agg[6] - 2 / OPPONENT_GRAVEYARD_CAP) < 1e-6             # opponent graveyard size
        assert abs(agg[7] - 6 / OPPONENT_BOARD_POWER_CAP) < 1e-6           # 2 Generous Ents * power 3 = 6
        # state.active_idx must be restored, not left pointed at the
        # opponent -- _for_player's whole contract (see its own docstring).
        assert synth_state.active_idx == 0
    finally:
        if had_power:
            ent_def.extra["power"] = original_power
        else:
            del ent_def.extra["power"]

    print("drl_env.py opponent-visibility observation self-check: OK")

    # Stack visibility, confirmed real gap fixed (docs/PRIORITY_PLAN.md
    # item 5): build_observation's own stack section is keyed only by the
    # CALLING side's own card_names -- Generous Ent isn't in
    # two_p_my_decklist (Mountain + Lightning Bolt) at all, so a card of
    # that name sitting on the stack would silently vanish from my own
    # observation. _opponent_stack_block (keyed by the OTHER side's card
    # pool) is what actually reveals it.
    stack_state = GameState(
        on_the_play=True, players=[PlayerState(on_the_play=True), PlayerState(on_the_play=False)],
    )
    stack_state.stack = [{"card_def": ent_def, "resolve": lambda s, cd: None, "controller": 1}]
    my_card_names, _my_card_copies = _card_lookup(two_p_my_decklist)
    assert "Generous Ent" not in my_card_names  # confirms this is genuinely invisible to build_observation's own stack section
    opp_card_names, opp_card_copies = _card_lookup(opp_creature_decklist)
    stack_block = _opponent_stack_block(stack_state, opp_card_names, opp_card_copies)
    ent_stack_i = opp_card_names.index("Generous Ent")
    assert abs(stack_block[ent_stack_i] - 1 / opp_card_copies["Generous Ent"]) < 1e-6
    top_start = len(opp_card_names)
    assert stack_block[top_start + ent_stack_i] == 1.0  # Generous Ent is the (only) top-of-stack entry
    assert stack_block[top_start + len(opp_card_names)] == 0.0  # "none" slot -- stack is non-empty

    print("drl_env.py opponent stack-visibility self-check: OK")

    # Hidden information during blocking (docs/COMBAT_PLAN.md's own design
    # discussion: "this would reveal the information of the attacker's hand
    # to the defender no?"): _opponent_choose_action/_own_choose_action are
    # each invoked directly here (bypassing _declare_blockers_gen's own
    # flip -- state.active_idx is set by hand below, exactly as that
    # function would leave it) with a spy "model" that records the obs
    # array it was actually handed. Both decklists are IDENTICAL on
    # purpose: same card_names ordering on both sides means the SAME
    # observation slot means "how many Lightning Bolts in MY hand" no
    # matter which seat is active when the obs is built -- so a real leak
    # (reading the wrong seat's hand) shows up as the WRONG side's known,
    # distinct hand count landing in that slot, not just a shape mismatch.
    hidden_decklist = [("Mountain", 10), ("Lightning Bolt", 5), ("Generous Ent", 3)]
    hidden_pending = game.derive_pending_kinds(hidden_decklist)
    lightning_bolt_idx = sorted({name for name, *_r in hidden_decklist}).index("Lightning Bolt")

    class _SpyModel:
        """Records every obs it's ever handed, always picks "Done blocking"
        (always legal whenever a declare_blockers resolution is pending --
        _done_blocking_legal) so the resolution actually closes instead of
        looping forever."""

        def __init__(self, actions):
            self.captured_obs = []
            self.done_idx = next(i for i, (nm, _l, _e) in enumerate(actions) if nm == "Done blocking")

        def predict(self, obs, action_masks=None, deterministic=False):
            self.captured_obs.append(obs)
            return self.done_idx, None

    hidden_env = TwoPlayerDeckEnv(
        rewards.strict_binary_reward, decklist=hidden_decklist, terminated_fn=lambda s: False,
        pending_kinds=hidden_pending, opponent_decklist=hidden_decklist,
        opponent_terminated_fn=lambda s: False, opponent_pending_kinds=hidden_pending,
        my_seat_idx=0, horizon=60, on_the_play=True, seed=3,
    )
    opponent_spy = _SpyModel(hidden_env.opponent_actions)
    own_spy = _SpyModel(hidden_env.actions)
    hidden_env.opponent_model = opponent_spy
    hidden_env.own_model = own_spy

    my_ent = Permanent(game.CARD_DEFS["Generous Ent"])
    opp_ent = Permanent(game.CARD_DEFS["Generous Ent"])
    hidden_state = GameState(
        on_the_play=True, players=[PlayerState(on_the_play=True), PlayerState(on_the_play=False)],
    )
    hidden_state.players[0].hand = [game.CARD_DEFS["Lightning Bolt"]] * 3 + [game.CARD_DEFS["Mountain"]]
    hidden_state.players[1].hand = [game.CARD_DEFS["Lightning Bolt"]] * 1 + [game.CARD_DEFS["Mountain"]] * 2
    hidden_state.players[0].battlefield = [my_ent]
    hidden_state.players[1].battlefield = [opp_ent]
    hidden_env.state = hidden_state

    # I attack, the opponent defends -- _declare_blockers_gen would flip
    # active_idx to 1 (the opponent's own seat) before ever consulting
    # them; _opponent_choose_action must build ITS obs from seat 1, never
    # seat 0 (mine).
    hidden_state.players[0].attackers = [my_ent]
    hidden_state.active_idx = 1
    game.begin_declare_blockers(hidden_state, on_complete=lambda s: None)
    hidden_env._opponent_choose_action(hidden_state)()
    opponent_obs = opponent_spy.captured_obs[-1]
    assert opponent_obs[lightning_bolt_idx] == 1 / 5  # the OPPONENT's own 1 Bolt -- never my 3
    assert hidden_state.pending_resolution is None  # "Done blocking" actually closed it

    # The opponent attacks, I defend -- symmetric case, active_idx flipped
    # to 0 (my own seat); _own_choose_action must build from seat 0, never
    # seat 1 (the opponent's).
    hidden_state.players[0].attackers = []
    hidden_state.players[1].attackers = [opp_ent]
    hidden_state.active_idx = 0
    game.begin_declare_blockers(hidden_state, on_complete=lambda s: None)
    hidden_env._own_choose_action(hidden_state)()
    own_obs = own_spy.captured_obs[-1]
    assert own_obs[lightning_bolt_idx] == 3 / 5  # MY OWN 3 Bolts -- never the opponent's 1
    assert hidden_state.pending_resolution is None

    print("drl_env.py hidden-information (blocking) self-check: OK")

    # Potential-based dense shaping (MULTIPLAYER_GAPS.md's "reward
    # functions are still 1p-shaped" item): shaping_weight=1.0 and a
    # reward_fn that always returns 0.0 isolate the shaping term entirely,
    # so its value can be checked exactly against the same Phi formula
    # _shaping_potential's own docstring documents, rather than trusting
    # the wiring by inspection alone.
    shaping_env = TwoPlayerDeckEnv(
        lambda *a: 0.0, decklist=two_p_my_decklist, terminated_fn=lambda s: False,
        pending_kinds=two_p_my_pending, opponent_decklist=two_p_opp_decklist,
        opponent_terminated_fn=lambda s: False, opponent_pending_kinds=two_p_opp_pending,
        my_seat_idx=0, horizon=60, on_the_play=True, seed=2, shaping_weight=1.0, shaping_gamma=0.99,
    )
    shaping_env.opponent_model = _AlwaysPassModel(shaping_env.opponent_pass_action)
    shaping_env.reset()
    assert shaping_env.state.active_idx == 0

    def _sidx(name):
        return next(i for i, (nm, _l, _e) in enumerate(shaping_env.actions) if nm == name)

    s_cast_bolt, s_choose_mountain = _sidx("Cast Lightning Bolt"), _sidx("Choose: Mountain")
    s_spend_r = _sidx("Spend R from pool")

    bolt_def = game.CARD_DEFS["Lightning Bolt"]
    mountain_def = game.CARD_DEFS["Mountain"]
    shaping_env.state.hand = [bolt_def]
    shaping_env.state.battlefield = [Permanent(mountain_def)]  # Bolt costs just {R}

    # Casting (announce + begin paying), tapping the land (which only
    # floats {R} into the pool -- MANA_POOL_PLAN.md), and spending it move
    # no life total at all -- Bolt's effect hasn't resolved yet (still on
    # the stack) -- so the isolated shaping term must be exactly 0 for all three.
    assert shaping_env.action_masks()[s_cast_bolt]
    _obs, cast_reward, _done, _truncated, _info = shaping_env.step(s_cast_bolt)
    assert shaping_env.state.pending_resolution["kind"] == "pay_cost"
    assert cast_reward == 0.0
    _obs, tap_reward, _done, _truncated, _info = shaping_env.step(s_choose_mountain)
    assert shaping_env.state.pending_resolution is not None
    assert tap_reward == 0.0
    _obs, spend_reward, _done, _truncated, _info = shaping_env.step(s_spend_r)
    assert shaping_env.state.pending_resolution is None
    assert len(shaping_env.state.stack) == 1  # paid for, not yet resolved
    assert spend_reward == 0.0

    # Pass resolves the stack (LIFO) -- Bolt's effect actually lands now,
    # dealing 3 damage to the opponent. This is the one step Phi actually
    # moves: Phi_before = 0.0 (opponent at full 20 life), Phi_after =
    # (20-17)/20 = 0.15.
    assert shaping_env.state.players[1].life_total == 20
    _obs, resolve_reward, _done, _truncated, _info = shaping_env.step(shaping_env.pass_action)
    assert shaping_env.state.players[1].life_total == 17
    expected_shaping = 1.0 * (0.99 * (3 / game.state.STARTING_LIFE) - 0.0)
    assert abs(resolve_reward - expected_shaping) < 1e-6

    print(f"drl_env.py potential-based shaping self-check: OK (isolated shaping reward={resolve_reward:.4f})")

    # tap_cost_options memoization never returns a stale answer (docs/
    # PRIORITY_PLAN.md item 6): build a pay_cost resolution with exactly 1
    # untapped Mountain, sweep the mask (populating the cache -- "Choose:
    # Mountain" legal), tap it (a real mutation -- zero untapped sources
    # left, so tap_cost_options itself now returns empty), then sweep
    # again -- the second sweep must see the mutation, not the first
    # sweep's cached answer, proving the cache doesn't leak across
    # separate legal_action_mask calls.
    perf_decklist = [("Mountain", 10), ("Lightning Bolt", 5)]
    perf_pending = game.derive_pending_kinds(perf_decklist)
    perf_actions = build_action_table(perf_decklist, game.EFFECT_REGISTRY, pending_kinds=perf_pending)
    perf_choose_mountain = next(i for i, (nm, _l, _e) in enumerate(perf_actions) if nm == "Choose: Mountain")
    perf_state = GameState(on_the_play=True, players=[PlayerState(True)])
    perf_state.hand = [game.CARD_DEFS["Lightning Bolt"]]
    perf_state.battlefield = [Permanent(game.CARD_DEFS["Mountain"])]
    game.begin_pay_cost(perf_state, {"R": 1}, on_complete=lambda s: None)
    assert legal_action_mask(perf_state, perf_actions)[perf_choose_mountain]
    assert _tap_cost_options_cache is None  # cleared again once the sweep itself returns
    game.execute_tap_cost_option(perf_state, "Mountain", None, False)  # taps the only Mountain -- 0 untapped sources left
    assert game.tap_cost_options(perf_state) == []  # ground truth: nothing left to tap
    assert not legal_action_mask(perf_state, perf_actions)[perf_choose_mountain]  # would be wrongly True if the first sweep's stale cache leaked through

    print("drl_env.py tap_cost_options cache self-check: OK")

    # Auto-pass without ever calling model.predict() (docs/PRIORITY_
    # PLAN.md item 6): a spy model that raises if predict() is ever
    # actually called, exercised against a state where Pass is the ONLY
    # legal action (nothing in hand, nothing on the battlefield, no
    # resolution pending).
    class _PredictShouldNotBeCalled:
        def predict(self, obs, action_masks=None, deterministic=False):
            raise AssertionError("model_choose_action must not call predict() when only Pass is legal")

    autopass_decklist = [("Mountain", 10)]
    autopass_actions = build_action_table(autopass_decklist, game.EFFECT_REGISTRY, pending_kinds=())
    autopass_pass_action = next(i for i, (nm, _l, _e) in enumerate(autopass_actions) if nm == "Pass")
    autopass_state = GameState(on_the_play=True, players=[PlayerState(True)])
    autopass_state.hand = []
    result = model_choose_action(
        autopass_state, obs=None, model=_PredictShouldNotBeCalled(),
        actions=autopass_actions, pass_action=autopass_pass_action,
    )
    assert result is None  # Pass -- and no exception, confirming predict() was genuinely never invoked

    print("drl_env.py model_choose_action auto-pass self-check: OK")
