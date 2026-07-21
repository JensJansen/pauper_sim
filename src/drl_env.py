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

# ---------------------------------------------------------------------------
# D2.1 -- Card indexing and observation builder (MULTI_DECK_PLAN.md Phase
# M4f: build_observation takes a decklist explicitly, sized to its
# distinct-card count, instead of a hardcoded 90-dim vector.
# ---------------------------------------------------------------------------

# Every deck needs "none" (nothing pending) and "pay_cost" (mana.py is
# universal) -- everything else is deck-specific (each card's own
# registry entry declares which kinds it needs -- see
# game.registry.derive_pending_kinds). Kept as the baseline here, not
# per-deck, since no deck could ever function without them.
BASELINE_PENDING_KINDS = ("none", "pay_cost")

# Floating mana pool observation cap: a single overtapping action rarely
# floats more than a handful of pips per color (e.g. Tron's 3 lands online
# producing up to 7 colorless in one turn) -- same fixed-cap-then-normalize
# pattern rewards.resource_quality already uses for available_mana.
POOL_CAP = 8


def observation_dim_for(decklist, pending_kinds):
    """Shared by DeckEnv.__init__ and harness.py's load() so a deck's
    observation dimension is computed identically in both places.
    pending_kinds is that deck's own extra kinds beyond the universal
    baseline (see game.registry.derive_pending_kinds) -- keeps a deck's
    dimension from moving every time an unrelated deck gains a new
    pending-resolution kind."""
    return len({name for name, *_rest in decklist}) * 4 + 2 + len(BASELINE_PENDING_KINDS) + len(pending_kinds) + len(game.POOL_COLORS)


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
    all_kinds = BASELINE_PENDING_KINDS + pending_kinds
    dim = len(card_names) * 4 + 2 + len(all_kinds) + len(game.POOL_COLORS)
    obs = np.zeros(dim, dtype=np.float32)

    hand_counts = {name: 0 for name in card_names}
    for card_def in state.hand:
        hand_counts[card_def.name] += 1

    bf_untapped = {name: 0 for name in card_names}
    bf_tapped = {name: 0 for name in card_names}
    for p in state.battlefield:
        if p.card_def.name not in bf_untapped:
            continue  # a token (Blood, Robot, ...) -- never a decklist member, so no observation slot exists for it (hand/graveyard never hold one, only battlefield -- see effects_common.py's token docstrings)
        if p.tapped:
            bf_tapped[p.card_def.name] += 1
        else:
            bf_untapped[p.card_def.name] += 1

    graveyard_counts = {name: 0 for name in card_names}
    for card_def in state.graveyard:
        graveyard_counts[card_def.name] += 1

    i = 0
    for name in card_names:
        obs[i] = hand_counts[name] / card_copies[name]
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

    return obs


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
#      source, search_fetch, choose_permanent, ancient_stirrings, and
#      scry/surveil's ordering phase), dispatched by pending_resolution["kind"]
#   G. Choose: <name> as <color>    -- flexible/filter mana sources during
#      a pay_cost resolution specifically (the only kind needing a color)
#   H. Keep / Dispose (scry/surveil)
#   I. Decline (Ancient Stirrings)
#   J. Abandon payment -- cancels a pending pay_cost resolution outright,
#      untapping everything tapped so far. Without this, tapping a
#      flexible/filter source for the wrong color could strand a game
#      with an unpayable remaining cost and zero legal actions -- see
#      game.abandon_pay_cost's docstring.
#
# spy_combo deck additions: B also covers Winding Way's modal cast (2
# actions, one per mode), Land Grant's free alt-cost, and Dread Return's
# Flashback (cast from the graveyard); C also covers non-mana activated
# abilities (Quirion Ranger); F/H also cover select_to_hand's own
# Keep/Bottom pair and its ordering phase (Lead the Stampede) and an
# optional search's Decline (Gatecreeper Vine) alongside Ancient
# Stirrings'.
# ---------------------------------------------------------------------------

def _land_drop_legal(name):
    def legal(state):
        return (
            state.pending_resolution is None
            and state.lands_played_this_turn == 0
            and any(c.name == name for c in state.hand)
        )
    return legal


def _land_drop_execute(name):
    def execute(state):
        game.play_land_from_hand(state, game.CARD_DEFS[name])
    return execute


def _cast_legal(name, extra_legal):
    def legal(state):
        if state.pending_resolution is not None:
            return False
        if not any(c.name == name for c in state.hand):
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
        # identically.
        game.on_cast_trigger(state, card_def)
        game.begin_pay_cost(state, card_def.cast_cost, on_complete=lambda s: resolve(s, card_def))
    return execute


def _activate_legal(name, cost_key):
    def legal(state):
        if state.pending_resolution is not None:
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
    kind of pending resolution -- if any -- is active."""
    pending = state.pending_resolution
    if pending is None:
        return []
    kind = pending["kind"]
    if kind == "pay_cost":
        return [n for n, c, f in game.tap_cost_options(state) if c is None and not f]
    if kind == "search_fetch":
        return game.search_fetch_options(state)
    if kind == "choose_permanent":
        return game.choose_permanent_options(state)
    if kind == "choose_graveyard_card":
        return game.choose_graveyard_card_options(state)
    if kind == "sacrifice":
        return game.sacrifice_options(state)
    if kind == "discard":
        return game.discard_options(state)
    if kind == "ancient_stirrings":
        return [n for n in game.ancient_stirrings_options(state) if n != "decline"]
    if kind in ("scry", "surveil") and pending["ordered"] is not None:
        return game.scry_surveil_options(state)
    if kind == "select_to_hand" and pending["ordered"] is not None:
        return game.select_to_hand_options(state)  # ordering phase only -- "keep"/"bottom" are their own actions
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
        elif kind == "choose_permanent":
            game.execute_choose_permanent_option(state, name)
        elif kind == "choose_graveyard_card":
            game.execute_choose_graveyard_card_option(state, name)
        elif kind == "sacrifice":
            game.execute_sacrifice_option(state, name)
        elif kind == "discard":
            game.execute_discard_option(state, name)
        elif kind == "ancient_stirrings":
            game.execute_ancient_stirrings_option(state, name)
        elif kind == "select_to_hand":
            game.execute_select_to_hand_option(state, name)  # ordering phase only
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
    return [(n, c) for n, c, _f in game.tap_cost_options(state) if c is not None]


def _choose_name_color_legal(name, color):
    def legal(state):
        return (name, color) in _choose_name_color_options(state)
    return legal


def _choose_name_color_execute(name, color):
    def execute(state):
        is_filter = next(f for n, c, f in game.tap_cost_options(state) if n == name and c == color)
        game.execute_tap_cost_option(state, name, color, is_filter)
    return execute


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


def _activate_no_cost_legal(name, ability_legal):
    """Non-mana activated-ability cost (Quirion Ranger's Forest bounce):
    no {T}-of-self assumption, unlike _activate_legal -- the ability's own
    legal(state, permanent) captures its whole cost precondition."""
    def legal(state):
        if state.pending_resolution is not None:
            return False
        p = next((p for p in state.battlefield if p.card_def.name == name), None)
        return p is not None and ability_legal(state, p)
    return legal


def _activate_no_cost_execute(name, resolve):
    def execute(state):
        p = next(p for p in state.battlefield if p.card_def.name == name)
        resolve(state, p)
    return execute


def _alt_cast_legal(name, extra_legal):
    """Land Grant's free alt-cost: no mana payment at all, just the
    card's own extra_legal predicate (0 lands in hand)."""
    def legal(state):
        if state.pending_resolution is not None:
            return False
        if not any(c.name == name for c in state.hand):
            return False
        return extra_legal(state)
    return legal


def _alt_cast_execute(name, resolve):
    def execute(state):
        card_def = game.CARD_DEFS[name]
        game.on_cast_trigger(state, card_def)  # item 11 -- see _cast_execute
        resolve(state, card_def)
    return execute


def _flashback_legal(name, ability_legal):
    """Dread Return's Flashback: cast from the graveyard, not hand."""
    def legal(state):
        if state.pending_resolution is not None:
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


def _plot_legal(name, cost):
    """Plot {cost}: pay it and exile this card from hand (no board
    presence yet) -- legal exactly like a normal cast, just against the
    plot cost instead of card_def.cast_cost."""
    def legal(state):
        if state.pending_resolution is not None:
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


def _cast_from_exile_legal(name, extra_legal):
    """Plot's second half: cast a previously-plotted copy, without paying
    its mana cost, on any turn after the one it was plotted on.

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
        resolve(state, card_def)
    return execute


def build_action_table(decklist, registry, token_card_defs=(), pending_kinds=()):
    """token_card_defs: activatable tokens a deck's own cards can create
    at runtime (Blood, Robot -- docs/MADNESS_DECKS_PLAN.md item 8), e.g.
    (game.BLOOD_TOKEN_CARD_DEF,). Tokens are never decklist entries (no
    quantity, not in game.CARD_DEFS), so they can't flow through
    distinct_names/game.CARD_DEFS[name] the way every other action here
    does -- only the activated-abilities loop below needs to know about
    them at all; casting/land-drop/Flashback/etc. stay decklist-only, a
    token is never cast or played as a land. Defaults to () so every
    existing call site (Tron, spy_combo -- neither creates tokens) is
    unaffected.

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
            actions.append((
                f"Cast {name}",
                _cast_legal(name, cast_spec.get("extra_legal")),
                _cast_execute(name, cast_spec["resolve"]),
            ))
        # Winding Way: a modal cast (choose creature or land) instead of a
        # single "cast" entry -- one action per mode.
        cast_modes = card_spec.get("cast_modes")
        if cast_modes is not None:
            for mode_name, mode_spec in cast_modes.items():
                actions.append((
                    f"Cast {name} (choose {mode_name})",
                    _cast_legal(name, mode_spec.get("extra_legal")),
                    _cast_execute(name, mode_spec["resolve"]),
                ))
        # Land Grant: a second, free cast path alongside the normal one.
        alt_cast = card_spec.get("alt_cast")
        if alt_cast is not None:
            actions.append((
                f"Cast {name} (free)",
                _alt_cast_legal(name, alt_cast["extra_legal"]),
                _alt_cast_execute(name, alt_cast["resolve"]),
            ))
        # Dread Return: Flashback casts from the graveyard, not hand.
        flashback = card_spec.get("flashback")
        if flashback is not None:
            actions.append((
                f"Flashback {name}",
                _flashback_legal(name, flashback["legal"]),
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
            actions.append((
                f"Plot {name}",
                _plot_legal(name, plot["cost"]),
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
                _cast_from_exile_legal(name, cast_spec.get("extra_legal")),
                _cast_from_exile_execute(name, plot.get("cast_from_exile_resolve", cast_spec["resolve"])),
            ))

    activatable = [(name, game.CARD_DEFS[name].effect_id) for name in distinct_names]
    activatable += [(cd.name, cd.effect_id) for cd in token_card_defs]
    for name, effect_id in activatable:
        abilities = registry.get(effect_id, {}).get("activated_abilities", {})
        for ability_name, spec in abilities.items():
            if "cost_key" in spec:
                actions.append((
                    f"Activate {name} ({ability_name})",
                    _activate_legal(name, spec["cost_key"]),
                    _activate_execute(name, spec["cost_key"], spec["resolve"]),
                ))
            else:
                # Non-mana cost (Quirion Ranger: return a Forest to hand).
                actions.append((
                    f"Activate {name} ({ability_name})",
                    _activate_no_cost_legal(name, spec["legal"]),
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

    for name in distinct_names:
        actions.append((f"Choose: {name}", _choose_name_legal(name), _choose_name_execute(name)))

    for name in distinct_names:
        spec = registry.get(game.CARD_DEFS[name].effect_id, {})
        colors = set()
        mana = spec.get("mana")
        if mana is not None and mana[0] == "flexible":
            colors |= mana[1]
        filter_mana = spec.get("filter_mana")
        if filter_mana is not None:
            colors |= filter_mana["colors"]
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


def legal_action_mask(state, actions):
    """Stateless: usable both by DeckEnv.action_masks() and by
    harness.evaluate(), which plays games directly through game.run_game,
    not through env.step (see DRL_CHECKLIST.md's D6 implementation note).
    `actions` is any table built by build_action_table -- every deck's own
    table, none privileged as a default (a caller with its own decklist
    always has its own table to pass, e.g. harness.py's self.actions)."""
    return np.array([legal_fn(state) for _, legal_fn, _ in actions], dtype=bool)


# ---------------------------------------------------------------------------
# D2.3 / D2.4 -- DeckEnv
# ---------------------------------------------------------------------------

def _start_turn(state):
    """Exactly game.run_turn's preamble (composing its public primitives in
    the same order), stopping short of running a main-phase loop -- that
    loop is what DeckEnv.step() replaces, one action at a time."""
    state.turn_number += 1
    state.lands_played_this_turn = 0
    state.cards_drawn_this_turn = 0
    game.untap_step(state)
    game.draw_step(state)
    game.drain_trigger_queue(state)  # e.g. Sneaky Snacker's return, if the turn's own draw queued it (item 7); no-op otherwise, safe to call unconditionally


class DeckEnv(gymnasium.Env):
    # Deck-parameterized (MULTI_DECK_PLAN.md Phase M4/M7): no deck gets a
    # default here (not even Tron) -- decklist/terminated_fn/pending_kinds
    # are always the caller's own (e.g. game.parse_decklist_file(...),
    # terminated.tron_terminated, game.derive_pending_kinds(decklist)).
    # Its own action table and observation dim are built fresh per
    # instance, never read from any module-level global.
    def __init__(self, reward_fn, decklist, terminated_fn, pending_kinds,
                 horizon=6, on_the_play=True, seed=None, combat_enabled=False):
        super().__init__()
        self.reward_fn = reward_fn
        self.decklist = decklist
        self.terminated_fn = terminated_fn
        self.horizon = horizon
        self.on_the_play = on_the_play
        self.pending_kinds = pending_kinds
        # rakdos madness / mono red madness only (default off, same as every
        # other deck-specific knob here) -- combat is fully automatic (see
        # game.turn.run_turn's own combat_enabled docstring), so this adds
        # no action-table entries and no observation dims, just a call at
        # the same "Pass ends the main phase" point below.
        self.combat_enabled = combat_enabled
        self._rng = random.Random(seed)
        self.state = None
        self.actions = build_action_table(decklist, game.EFFECT_REGISTRY, pending_kinds=pending_kinds)
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
        _start_turn(self.state)
        self._cached_mask = None
        return build_observation(self.state, self.decklist, self.horizon, self.pending_kinds), {}

    def step(self, action):
        # MaskablePPO's rollout loop (sb3_contrib ppo_mask.collect_rollouts)
        # always calls action_masks() immediately before step(), with no
        # state mutation in between (just a policy forward pass) -- so
        # recomputing the same mask here would be pure duplicate work.
        # Reused once, then cleared so any other caller (or an out-of-order
        # call) falls back to a fresh computation instead of a stale one.
        if self._cached_mask is not None:
            mask, self._cached_mask = self._cached_mask, None
        else:
            mask = legal_action_mask(self.state, self.actions)
        if not mask[action]:
            # Illegal action -> substitute the first currently-legal one
            # (works for any SB3 algorithm, maskable or not). MUST NOT
            # assume PASS_ACTION specifically is always a safe substitute
            # (MULTI_DECK_PLAN.md Phase M4e): Pass is illegal whenever a
            # resolution is pending, and blindly "passing" in that state
            # would abandon the resolution mid-flight and desync
            # state.pending_resolution from the turn loop. legal_indices
            # is never empty: with no resolution pending, Pass itself is
            # always legal; with one pending, every kind guarantees at
            # least one option (pay_cost by construction, search_fetch/
            # choose_permanent's empty case auto-fizzles instead of
            # leaving a stuck resolution, ancient_stirrings always offers
            # "decline", scry/surveil always offers keep/dispose or a
            # nonempty ordering set).
            legal_indices = np.flatnonzero(mask)
            action = int(legal_indices[0])

        game_over = False
        if action == self.pass_action:
            if self.combat_enabled:
                # Runs once, right as the main phase ends -- a creature
                # tapped earlier this turn for something else is how the
                # model "held it back" instead of attacking; may itself
                # set state.turn_won (a second way to reach a damage_dealt
                # win condition, alongside a permanent entering).
                game.combat_step(self.state)
            if self.state.turn_won is not None:
                pass  # combat just ended it -- nothing else to do this step, done computed below
            elif self.state.turn_number < self.horizon:
                _start_turn(self.state)
            else:
                game_over = True  # just passed during the final turn -- no more turns left
        else:
            _, _, execute_fn = self.actions[action]
            execute_fn(self.state)
            # Drains any queued Madness decision / (later) Sneaky Snacker
            # return -- only takes effect once execute_fn's own action is
            # fully resolved (pending_resolution back to None), never
            # mid-resolution. See docs/MADNESS_DECKS_PLAN.md items 1/3/7.
            game.drain_trigger_queue(self.state)

        done = self.state.turn_won is not None or game_over or self.state.decked_out
        reward = self.reward_fn(self.state, done, self.horizon)
        obs = build_observation(self.state, self.decklist, self.horizon, self.pending_kinds)
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
    from game.state import GameState, Permanent

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
        assert _plot_legal("Fake Plot Spell", PLOT_COST)(state)
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
        assert not _cast_from_exile_legal("Fake Plot Spell", None)(state)

        # A later turn: castable for free, fires on_cast_trigger (Guttersnipe).
        state.turn_number += 1
        assert _cast_from_exile_legal("Fake Plot Spell", None)(state)
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
        assert not _cast_from_exile_legal("Fake Plot Spell", game.EFFECT_REGISTRY[EffectId.FILLER]["cast"]["extra_legal"])(state)
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

    # Combat gating: fully automatic, no new action-table entries or
    # observation dims (see game.turn.run_turn's own combat_enabled
    # docstring) -- this exercises DeckEnv.step's wiring of it at the
    # "Pass" boundary specifically (combat_step itself is already
    # self-checked directly in effects_common.py). Unlike the tokens check
    # above, this can't use empty_decklist -- build_observation indexes
    # battlefield permanents by name against the decklist's own card set,
    # so a synthetic name absent from every decklist would KeyError.
    # Reuses a real Tron creature (Generous Ent) instead, temporarily
    # tagging its real CardDef with "power" (same save/restore convention
    # as every other real-card-borrowing check in this file).
    combat_terminated = lambda s: s.damage_dealt >= 3
    tron_decklist = game.parse_decklist_file(os.path.join(os.path.dirname(__file__), "..", "data", "monster_tron.txt"))
    tron_pending_kinds = game.derive_pending_kinds(tron_decklist)
    ent_card_def = game.CARD_DEFS["Generous Ent"]
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

        assert env.action_masks()[env.pass_action]
        _obs, _reward, done, _truncated, _info = env.step(env.pass_action)
        assert attacker.tapped  # combat ran on Pass, before any next-turn untap
        assert env.state.damage_dealt == 3
        assert env.state.turn_won == env.state.turn_number  # combat_step's own terminated_fn check caught it, same as enters_battlefield's
        assert done

        # Same setup, but combat_enabled left at its default (False,
        # matching Tron/spy_combo) -- confirms combat is opt-in, not
        # accidentally on.
        env2 = DeckEnv(
            lambda *a: 0.0, decklist=tron_decklist, terminated_fn=combat_terminated,
            pending_kinds=tron_pending_kinds, horizon=5,
        )
        env2.reset()
        attacker2 = Permanent(ent_card_def)
        attacker2.summoning_sick = False
        env2.state.battlefield = [attacker2]
        env2.step(env2.pass_action)
        assert not attacker2.tapped
        assert env2.state.damage_dealt == 0
        assert env2.state.turn_won is None
    finally:
        del ent_card_def.extra["power"]

    print("drl_env.py combat self-check: OK")
