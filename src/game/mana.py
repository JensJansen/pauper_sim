"""Mana system (float-first): what a permanent produces, activating a mana
ability to float mana into the pool (activate_mana_source), spending that pool
to pay a cost (begin_pay_cost + execute_pool_spend), and pure pool-affordability
legality checks (pool_can_pay / plan_payment).

References registry.EFFECT_REGISTRY + its derived views only inside function
bodies (the lazy-lookup convention every submodule here uses -- keeps mana.py
import-order-safe from anywhere)."""

from . import registry
from .cards import CardType, EffectId
from .resolution import begin_resolution, complete_resolution


def _has_haste(state, permanent):
    """Haste from either source: a registry "haste": True (Kitchen Imp) or a
    granted/static "haste" keyword (Goblin Tomb Raider). Lazy stats import --
    stats has no need for mana, but keep the load-order convention."""
    if registry.EFFECT_REGISTRY.get(permanent.card_def.effect_id, {}).get("haste", False):
        return True
    from .effects.stats import creature_keywords
    return "haste" in creature_keywords(state, permanent)


def tap_summoning_locked(state, permanent):
    """A CREATURE ability with the {T} symbol in its cost can't be activated
    while summoning sick (real Magic 302.6: needs continuous control since your
    most recent turn began), unless the creature has haste. Non-creatures are
    never summoning sick. Shared by the mana system (creature mana dorks) and
    the non-mana {T} activated abilities (Wellwisher, Timberwatch Elf).

    EXCEPTION -- a mana ability with NO {T} in its real cost is NOT gated:
    Wall of Roots ("{0}: Put a -0/-1 counter on this: Add {G}", marked
    "mana_no_tap") can produce mana the very turn it enters, exactly because
    its ability has no tap symbol. 302.6 restricts only {T}/{Q} abilities."""
    if permanent.card_type != CardType.CREATURE:
        return False
    if registry.EFFECT_REGISTRY.get(permanent.card_def.effect_id, {}).get("mana_no_tap", False):
        return False  # no {T} in the cost -> summoning sickness doesn't apply
    return permanent.summoning_sick and not _has_haste(state, permanent)


TRON_TYPES = {"Mine", "Power Plant", "Tower"}
COLORS = ("W", "U", "B", "R", "G")
POOL_COLORS = COLORS + ("C",)  # every symbol a mana source can actually produce


def _cost_satisfied(remaining):
    return not any(v > 0 for v in remaining.values())


def controls_all_tron_types(state):
    present = {
        p.card_def.extra["tron_type"]
        for p in state.battlefield
        if p.card_def.effect_id == EffectId.TRON_LAND
    }
    return TRON_TYPES.issubset(present)


_enchanting_cache = None  # (state, {id(permanent): [auras enchanting it]}) -- see _enchanting/reset_mana_cache


def _enchanting(state, permanent):
    """Every Aura on state.battlefield currently enchanting `permanent` --
    the shared scan _bonus_mana_symbols/_granted_mana_colors below both
    used to redo independently, once per call. mana_ability_options and
    mana_output (via _bonus_mana_symbols/_granted_mana_colors) are called
    across many candidate permanents per drl_env.legal_action_mask sweep,
    and none of that battlefield scan depends on which permanent is being
    tested -- so it was identical, wasted work repeated every call. Cached
    per state object, same scope as _cached_battlefield_lookup/
    _cached_mana_ability_options in drl_env: safe because a
    legal_action_mask sweep only ever calls legal_fns, never an execute_fn,
    so state can't mutate mid-sweep. Reset by drl_env.legal_action_mask
    before AND after its own sweep (see reset_mana_cache below) -- not by
    watching for mutation, so it must never be trusted to outlive one sweep
    on its own."""
    global _enchanting_cache
    if _enchanting_cache is None or _enchanting_cache[0] is not state:
        by_target = {}
        for aura in state.battlefield:
            target = aura.flags.get("enchanting")
            if target is not None:
                by_target.setdefault(id(target), []).append(aura)
        _enchanting_cache = (state, by_target)
    return _enchanting_cache[1].get(id(permanent), ())


def reset_mana_cache():
    """Called by drl_env.legal_action_mask before and after each sweep --
    see _enchanting's own docstring for why this can't self-invalidate."""
    global _enchanting_cache
    _enchanting_cache = None


def _bonus_mana_symbols(state, permanent):
    """Utopia Sprawl's own mechanic: an Aura enchanting this permanent with
    a "bonus_mana_color" flag (chosen once, at cast time) adds that
    color's symbol every time this permanent is tapped for mana --
    automatic, always on top of whatever ability was actually used
    (contrast _granted_mana_colors below, Abundant Growth's genuinely
    competing ability -- real Utopia Sprawl triggers on the land being
    "tapped for mana" at all, not specifically its own native ability)."""
    return [
        aura.flags["bonus_mana_color"]
        for aura in _enchanting(state, permanent)
        if "bonus_mana_color" in aura.flags
    ]


def _granted_mana_colors(state, permanent):
    """Colors a competing granted ability (Abundant Growth's "{T}: Add one
    mana of any color") lets this permanent tap for, in ADDITION to (not
    replacing) its own native ability -- the model chooses one or the
    other each time it taps, unlike _bonus_mana_symbols above. Union
    across every Aura enchanting this permanent, in case more than one
    ever grants colors to the same land."""
    granted = set()
    for aura in _enchanting(state, permanent):
        granted |= aura.flags.get("bonus_mana_colors", set())
    return granted


def mana_output(permanent, state, color_choice=None):
    """Mana symbols this permanent would produce if tapped for its plain
    mana ability right now. Raises if effect_id isn't a simple source or if
    a required/forbidden color_choice is missing/invalid.

    A granted color (_granted_mana_colors) is checked first and, if
    matched, short-circuits the registry-driven dispatch below entirely --
    it's a genuinely separate ability from the permanent's own, not a
    variant of it, so none of the per-kind color_choice validation below
    applies to it."""
    effect = permanent.card_def.effect_id
    spec = registry.EFFECT_REGISTRY.get(effect, {}).get("mana")
    if spec is None:
        raise ValueError(f"{permanent.card_def.name} is not a simple mana source")
    if color_choice is not None and color_choice in _granted_mana_colors(state, permanent):
        return [color_choice] + _bonus_mana_symbols(state, permanent)
    kind = spec[0]
    if kind == "tron":
        if not controls_all_tron_types(state):
            output = ["C"]
        else:
            # Urza's Tower doubles to {C}{C}{C} when online; Mine/Power
            # Plant double to {C}{C} -- the three Tron lands aren't
            # interchangeable here despite sharing the same effect_id/kind.
            output = ["C", "C", "C"] if permanent.card_def.extra["tron_type"] == "Tower" else ["C", "C"]
    elif kind == "fixed":
        if color_choice is not None:
            raise ValueError(f"{permanent.card_def.name} has no color choice")
        output = [spec[1]]
    elif kind == "fixed_multi":
        # ("fixed_multi", (symbol, symbol, ...)): Rakdos Carnarium's
        # {T}: Add {B}{R} -- both symbols from one tap, not a choice of
        # one.
        if color_choice is not None:
            raise ValueError(f"{permanent.card_def.name} has no color choice")
        output = list(spec[1])
    elif kind == "flexible":
        choices = spec[1]
        if color_choice not in choices:
            raise ValueError(f"{permanent.card_def.name} cannot produce {color_choice}")
        output = [color_choice]
    elif kind == "count":
        # ("count", symbol, predicate): Overgrown Battlement -- one symbol
        # per battlefield permanent matching predicate (itself included).
        if color_choice is not None:
            raise ValueError(f"{permanent.card_def.name} has no color choice")
        symbol, predicate = spec[1], spec[2]
        output = [symbol] * sum(1 for p in state.battlefield if predicate(p))
    elif kind == "count_all":
        # ("count_all", symbol, predicate): like "count", but "for each X on
        # THE BATTLEFIELD" (both players' battlefields) rather than "you
        # control" -- Priest of Titania ("Add {G} for each Elf on the
        # battlefield").
        if color_choice is not None:
            raise ValueError(f"{permanent.card_def.name} has no color choice")
        symbol, predicate = spec[1], spec[2]
        output = [symbol] * sum(1 for pl in state.players for p in pl.battlefield if predicate(p))
    else:
        raise ValueError(f"{permanent.card_def.name} is not a simple mana source")
    return output + _bonus_mana_symbols(state, permanent)


def begin_pay_cost(state, cost, on_complete):
    """Float-first mana payment: entered only when `cost` is already affordable
    from the floating pool (callers gate on plan_payment/pool_can_pay first, and
    the agent floats its mana via mana abilities BEFORE casting). The model then
    spends one floating pip at a time (see pool_spend_options/execute_pool_spend)
    until `cost` is covered -- no source is ever tapped during payment.

    A cost already fully covered before any spend (e.g. Lotus Petal's empty {}
    cast cost) completes immediately -- the same check execute_pool_spend runs
    after each spend, run here too for the zero-input case. Without it the
    resolution would open with nothing left to pay and nothing able to close
    it, softlocking the cast instead of resolving it."""
    begin_resolution(state, "pay_cost", on_complete, remaining=dict(cost))
    if _cost_satisfied(state.pending_resolution["remaining"]):
        complete_resolution(state)


def mana_ability_options(state):
    """Float-first: every (name, color_choice) mana ability the active player
    can activate RIGHT NOW via this GENERIC (name, color) shape, one per
    distinct source name -- a top-level action legal in ANY priority window,
    even mid-resolution of anything else (605.1a/605.3b: a mana ability never
    uses the stack and doesn't require priority to activate). Lists only what
    CAN produce mana now (untapped/available, not summoning-locked, extra cost
    payable), leaving spending it for a later cast/payment step. Filters
    (filter_mana) are a separate pool->pool conversion, not listed here.
    Sources whose activation needs an extra CHOICE (mana_extra_choose --
    Saruli Caretaker's "tap another creature") are also excluded: that choice
    must be enumerated atomically alongside it (drl_env's own dedicated
    action), not exposed via this plain (name, color) shape."""
    options, seen = [], set()

    def _add(key):
        if key not in seen:
            seen.add(key)
            options.append(key)

    for p in state.battlefield:
        if p.tapped or tap_summoning_locked(state, p):
            continue
        spec = registry.EFFECT_REGISTRY.get(p.card_def.effect_id, {}).get("mana")
        if spec is None:
            continue
        if registry.EFFECT_REGISTRY.get(p.card_def.effect_id, {}).get("mana_extra_choose") is not None:
            continue
        extra = registry.EFFECT_REGISTRY.get(p.card_def.effect_id, {}).get("mana_extra_available")
        if extra is not None and not extra(state, p):
            continue
        kind = spec[0]
        if kind in ("fixed", "fixed_multi", "tron", "count", "count_all"):
            _add((p.card_def.name, None))
        elif kind == "flexible":
            for color in spec[1]:
                _add((p.card_def.name, color))
    # Abundant Growth's granted colors -- a runtime per-instance fact (every
    # Forest shares one effect_id), so it can't fold into the spec branch above.
    for p in state.battlefield:
        if p.tapped:
            continue
        for color in _granted_mana_colors(state, p):
            _add((p.card_def.name, color))
    return options


def activate_mana_source(state, permanent, color_choice=None):
    """Float-first: activate `permanent`'s mana ability immediately -- tap it
    (unless mana_no_tap), add its produced symbols to the pool, run any on_tap
    side effect (Lotus Petal/Treasure self-sac, Wall of Roots counter). No
    pending resolution: a mana ability resolves at once and never uses the
    stack. (Saruli's extra "tap another creature" cost is an agent choice
    opened by the drl_env mana-ability action, not an on_tap side effect.)"""
    no_tap = registry.EFFECT_REGISTRY.get(permanent.card_def.effect_id, {}).get("mana_no_tap", False)
    if not no_tap:
        permanent.tapped = True
    produced = mana_output(permanent, state, color_choice)
    for symbol in produced:
        state.mana_pool[symbol] = state.mana_pool.get(symbol, 0) + 1
    state.log_event("mana_tap", permanent=(permanent.card_def.name, permanent.slot),
                    mode="no_tap" if no_tap else "normal", produced=produced)
    on_tap = registry.EFFECT_REGISTRY.get(permanent.card_def.effect_id, {}).get("on_tap")
    if on_tap is not None:
        on_tap(state, permanent)


def pool_spend_options(state):
    """While a pay_cost resolution is pending: every floating-pool color
    with a nonzero balance that would still make progress on the remaining
    cost right now -- matching a live colored need, or (any color) an
    outstanding generic need."""
    pending = state.pending_resolution
    remaining = pending["remaining"]
    if _cost_satisfied(remaining):
        return []
    generic_needed = remaining.get("generic", 0) > 0
    return sorted(
        color for color in POOL_COLORS
        if state.mana_pool.get(color, 0) > 0 and (remaining.get(color, 0) > 0 or generic_needed)
    )


def execute_pool_spend(state, color):
    """Spend one floating pip of `color` toward the pending cost: its own
    matching colored need first, else outstanding generic -- which color to
    preserve for a possible same-phase second spell is the agent's decision,
    made explicitly here."""
    pending = state.pending_resolution
    pool = state.mana_pool
    pool[color] -= 1
    if pool[color] <= 0:
        del pool[color]
    # Marks this phase as having paid for SOMETHING -- one of the three
    # exemptions game.turn._empty_mana_pools checks before ever counting a
    # later burn as a mistake (see PlayerState.cost_paid_this_phase).
    state.cost_paid_this_phase = True

    remaining = pending["remaining"]
    need = remaining.get(color, 0)
    if need > 0:
        remaining[color] = need - 1
        toward = "colored"
    else:
        remaining["generic"] = max(0, remaining.get("generic", 0) - 1)
        toward = "generic"
    state.log_event("mana_spend", color=color, toward=toward)

    if _cost_satisfied(remaining):
        complete_resolution(state)


def _reduce_cost_by_pool(pool, cost):
    """Pure: how much of `cost` the floating pool alone could already cover
    (own-color pips first, then any leftover pool color against generic),
    returned as the remaining cost still needing a tap. Used only for
    legality (plan_payment/_cast_legal etc.) -- actually spending pool mana
    during a real payment is always the model's own execute_pool_spend
    action, never decided here."""
    remaining = dict(cost)
    spare = dict(pool)
    for color in POOL_COLORS:
        need = remaining.get(color, 0)
        have = spare.get(color, 0)
        used = min(need, have)
        if used:
            remaining[color] = need - used
            spare[color] = have - used
    leftover = sum(spare.values())
    generic_needed = remaining.get("generic", 0)
    remaining["generic"] = max(0, generic_needed - leftover)
    return remaining


def pool_can_pay(pool, cost):
    """True iff the floating `pool` alone already covers `cost` -- specific
    pips (W/U/B/R/G/C) by their own symbol, generic by any leftover. The
    float-first legality check: under float-first the agent produces its mana
    into the pool via mana abilities BEFORE casting, so "can I afford this?"
    is a pure, exact question about concrete mana -- no source-tapping solver
    (plan_payment) needed. Pure (no mutation)."""
    return _cost_satisfied(_reduce_cost_by_pool(pool, cost))


def plan_payment(state, cost):
    """Float-first legality shim: a spell/ability is castable iff the floating
    pool already covers `cost` (pool_can_pay) -- the agent produces its mana via
    mana abilities BEFORE casting, so affordability is a pure, exact pool
    question. Returns a truthy sentinel when payable, else None, so every
    existing `plan_payment(...) is not None` legality gate reads the pool with no
    call-site edits. A filter (Barrels/Conduit Pylons) is now an explicit
    pool->pool conversion the agent runs itself, not a payment-time fallback.
    Pure (no mutation). (Thin alias for pool_can_pay kept only to spare its many
    `is not None` call sites a rename.)"""
    return True if pool_can_pay(state.mana_pool, cost) else None
