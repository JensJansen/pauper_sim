"""Mana system (cast-then-pay, real rule 601.2): what a permanent produces,
activating a mana ability to float mana into the pool (activate_mana_source),
spending that pool to pay a cost (begin_pay_cost + execute_pool_spend), and the
affordability check every cast/activate legality gate runs (plan_payment, over
available_mana_units + can_pay).

ORDERING. Real Magic announces a spell, settles modes/targets/X, determines the
total cost (601.2e), and only THEN activates mana abilities (601.2f) and pays
(601.2g). This engine used to run 601.2f FIRST -- "float-first": a cast was
illegal unless the floating pool already covered it, and no source was ever
tapped during payment. That is the reverse of the real sequence and it forced
the agent to commit to a mana configuration before knowing what it was paying
for. Cast-then-pay (2026-08-17) restores the real order: a cast is legal when
the pool PLUS still-untapped sources could cover the cost, and the sources are
tapped during the payment itself.

THE STRANDING INVARIANT. There is no "Abandon payment" action, so a payment the
agent cannot finish leaves it with no legal action at all (an all-False mask,
a hard error). Two things uphold the invariant that a begun payment can always
be finished:
  1. plan_payment is EXACT -- see can_pay. A false positive would strand;
     a false negative would hide a legal cast, which is its own faithfulness
     bug, so "conservative" is not a safe default here either.
  2. every action taken while a payment is open is legal only if the payment
     is still completable afterwards (drl_env's own gates call back into
     can_pay for this). Tapping a source only ever adds mana, but a FILTER
     taps its own source (Conduit Pylons is both a filter and a {C} source)
     and Saruli Caretaker's cost taps another creature that may itself be a
     mana source, so neither is automatically safe.

References registry.EFFECT_REGISTRY + its derived views only inside function
bodies (the lazy-lookup convention every submodule here uses -- keeps mana.py
import-order-safe from anywhere)."""

from itertools import combinations

from . import registry
from .cards import CardType, EffectId
from .resolution import begin_resolution, complete_resolution


def _has_haste(state, permanent):
    """Delegates to stats.has_haste, the one canonical haste check shared
    with combat.py's creature_attack_eligible -- see its docstring. Lazy
    stats import: stats has no need for mana, but keep the load-order
    convention."""
    from .effects.stats import has_haste
    return has_haste(state, permanent)


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
    """CR 601.2f/601.2g: the payment window. Entered once `cost` is known to be
    payable (callers gate on plan_payment first), which under cast-then-pay
    means from the floating pool PLUS whatever is still tappable -- so this is
    where the agent both activates mana abilities and spends the pool, one
    action at a time, until `cost` is covered.

    `announced` records the cost as it stood when the payment opened, next to
    the `remaining` that shrinks as it is paid. Nothing in the payment reads it;
    it exists so that if a payment ever DOES strand, the bug report can say
    whether plan_payment was wrong to allow the announcement or whether supply
    was consumed afterwards -- two different bugs that look identical from
    `remaining` alone (see rl.agent._raise_all_false).

    A cost already fully covered before any spend (e.g. Lotus Petal's empty {}
    cast cost) completes immediately -- the same check execute_pool_spend runs
    after each spend, run here too for the zero-input case. Without it the
    resolution would open with nothing left to pay and nothing able to close
    it, softlocking the cast instead of resolving it."""
    # The STRANDING INVARIANT, asserted at the one place it can still be
    # cheaply localised. Every caller is supposed to have gated on plan_payment
    # first; if one has not, the failure otherwise surfaces several actions
    # later as an all-False mask, with the traceback pointing at the agent
    # rather than at the cast path that skipped its check. Failing here names
    # the guilty call site directly in the traceback.
    assert can_pay(available_mana_units(state), cost), (
        f"begin_pay_cost opened an unpayable cost {dict(cost)} -- available units "
        f"{sorted(''.join(sorted(u)) for u in available_mana_units(state))}, pool "
        f"{dict(state.mana_pool)}. This caller did not gate on plan_payment, or gated on a "
        f"DIFFERENT cost than it went on to charge."
    )
    begin_resolution(state, "pay_cost", on_complete, remaining=dict(cost), announced=dict(cost))
    if _cost_satisfied(state.pending_resolution["remaining"]):
        complete_resolution(state)


def tappable_for_mana(state, permanent):
    """Can `permanent` be tapped for mana RIGHT NOW? Untapped, not
    summoning-locked (302.6), and its own extra availability condition (Wall of
    Roots' once-per-turn) satisfied.

    THE one place these gates live. mana_ability_options (which abilities are
    offered) and source_mana_units (how much mana is available to pay with) both
    ask it, so the action space and the affordability check cannot drift into
    disagreeing about what is usable -- they previously re-implemented the same
    four conditions side by side, with the guarantee stated only in a comment.

    Excludes mana_extra_choose (Saruli Caretaker's "tap another creature"),
    whose activation needs a choice enumerated atomically alongside it rather
    than through either caller's plain shape -- and which source_mana_units must
    exclude anyway, since its cost CONSUMES another source (see there)."""
    if permanent.tapped or tap_summoning_locked(state, permanent):
        return False
    entry = registry.EFFECT_REGISTRY.get(permanent.card_def.effect_id, {})
    if entry.get("mana_extra_choose") is not None:
        return False
    extra = entry.get("mana_extra_available")
    return extra is None or extra(state, permanent)


def mana_ability_options(state):
    """Every (name, color_choice) mana ability the active player can activate
    RIGHT NOW via this GENERIC (name, color) shape, one per
    distinct source name -- a top-level action legal in ANY priority window,
    even mid-resolution of anything else (605.1a/605.3b: a mana ability never
    uses the stack and doesn't require priority to activate). Lists only what
    CAN produce mana now (tappable_for_mana), leaving spending it for a later
    cast/payment step. Filters (filter_mana) are a separate pool->pool
    conversion, not listed here."""
    options, seen = [], set()

    def _add(key):
        if key not in seen:
            seen.add(key)
            options.append(key)

    for p in state.battlefield:
        if not tappable_for_mana(state, p):
            continue
        spec = registry.EFFECT_REGISTRY.get(p.card_def.effect_id, {}).get("mana")
        kind = spec[0] if spec is not None else None
        if kind in ("fixed", "fixed_multi", "tron", "count", "count_all"):
            _add((p.card_def.name, None))
        elif kind == "flexible":
            for color in spec[1]:
                _add((p.card_def.name, color))
        # Abundant Growth's granted colors -- a runtime per-instance fact (every
        # Forest shares one effect_id), so it can't fold into the spec branch
        # above. Folded into this loop rather than a second pass over the
        # battlefield: the second pass only checked `tapped`, so it could offer
        # a granted color on a summoning-locked source the first pass had
        # already refused. Unreachable today (Abundant Growth enchants lands,
        # and tap_summoning_locked is False for anything non-creature), which is
        # why this is a consistency fix rather than a behavior change.
        for color in _granted_mana_colors(state, p):
            _add((p.card_def.name, color))
    return options


def activate_mana_source(state, permanent, color_choice=None):
    """Activate `permanent`'s mana ability immediately -- tap it
    (unless mana_no_tap), add its produced symbols to the pool (float_mana,
    which also tags single-pip production -- see its own docstring), run any
    on_tap side effect (Lotus Petal/Treasure self-sac, Wall of Roots
    counter). No pending resolution: a mana ability resolves at once and
    never uses the stack. (Saruli's extra "tap another creature" cost is an
    agent choice opened by the drl_env mana-ability action, not an on_tap
    side effect.)"""
    no_tap = registry.EFFECT_REGISTRY.get(permanent.card_def.effect_id, {}).get("mana_no_tap", False)
    if not no_tap:
        permanent.tapped = True
    produced = mana_output(permanent, state, color_choice)
    float_mana(state, produced)
    state.log_event("mana_tap", permanent=(permanent.card_def.name, permanent.slot),
                    mode="no_tap" if no_tap else "normal", produced=produced)
    on_tap = registry.EFFECT_REGISTRY.get(permanent.card_def.effect_id, {}).get("on_tap")
    if on_tap is not None:
        on_tap(state, permanent)


def float_mana(state, symbols, taggable=True):
    """Adds `symbols` (the list a mana-producing EVENT just produced, e.g.
    ["G"] or ["G", "G", "G"]) to state.mana_pool, one add per symbol.

    Also TAGS the floated pip in state.mana_pool_single_pip -- the
    "single-pip" tag rule is dynamic and per-event: an event is single-pip
    iff it added EXACTLY 1 symbol to the pool (len(symbols) == 1),
    regardless of the source's registry mana `kind`. A plain land or
    Llanowar Elves always qualifies; a 2+-symbol event (Rakdos Carnarium,
    Priest of Titania with other Elves out, Utopia Sprawl's automatic
    bonus, an online Tron land) never does. See PlayerState.
    mana_pool_single_pip's own docstring for the full rationale.

    taggable=False is the ONE explicit exception to that rule: forces no
    tag even for a 1-symbol event -- used only by a mana filter's output
    pip (drl_env._actions_mana._filter_mana_execute), which always
    produces exactly 1 symbol but is a deliberate pool->pool conversion,
    not reflexive tapping, and must never be punished as such."""
    pool = state.mana_pool
    for symbol in symbols:
        pool[symbol] = pool.get(symbol, 0) + 1
    if taggable and len(symbols) == 1:
        single = state.mana_pool_single_pip
        color = symbols[0]
        single[color] = single.get(color, 0) + 1


def discount_departing_source(state, permanent, owner_idx):
    """`permanent` just became newly available for a fresh tap this phase
    without actually being tapped for it -- either it left the battlefield
    via sacrifice, or an effect just untapped it after it had already been
    tapped. Tapping it for its own mana ability right before/around that
    event would have been strictly free value (there's no future turn where
    holding the mana back could have mattered), so any single-pip-tagged
    mana of a color it can produce is retroactively excused from burn --
    exactly like a mana filter's own output pip (float_mana's taggable=False
    case above).

    mana_pool_single_pip carries no per-source identity (see its own
    docstring), so this can't target the exact pip `permanent` itself
    produced -- it discounts whichever of its candidate colors currently has
    the most tagged mana floating (state.rng breaks a tie, over colors
    sorted first so the tie-break depends only on state.rng, not on a
    set's hash-randomized iteration order), the same fungible-pip treatment
    spend_one_pip already gives same-color pips.

    Only the "fixed" and "flexible" mana-spec kinds qualify -- every kind
    that always produces exactly one symbol per tap, matching float_mana's
    own single-pip tag rule. fixed_multi/tron/count/count_all are out of
    scope (almost never tagged in the first place)."""
    spec = registry.EFFECT_REGISTRY.get(permanent.card_def.effect_id, {}).get("mana")
    if spec is None or spec[0] not in ("fixed", "flexible"):
        return
    candidates = [spec[1]] if spec[0] == "fixed" else sorted(spec[1])
    single = state.players[owner_idx].mana_pool_single_pip
    positive = [c for c in candidates if single.get(c, 0) > 0]
    if not positive:
        return
    best = max(single[c] for c in positive)
    tied = [c for c in positive if single[c] == best]
    color = tied[0] if len(tied) == 1 else state.rng.choice(tied)
    single[color] -= 1
    if single[color] <= 0:
        del single[color]


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


def spend_one_pip(state, color):
    """Decrements one floating pip of `color` off state.mana_pool --
    applying the spend-order convention: an UNTAGGED (multi-pip-burst-
    sourced) pip of this color is always consumed BEFORE a TAGGED
    (single-pip-sourced) one. state.mana_pool_single_pip only shrinks once
    no untagged pip of this color remains. This is what lets a burst
    source's own unavoidable excess (Priest of Titania, Overgrown
    Battlement) absorb blame for a burnt leftover ahead of a genuinely
    avoidable single-pip tap of the same color (e.g. tapping 2 unneeded
    Llanowar Elves alongside a Priest burst that alone would have covered
    the spend) -- the burst mana is always "spent" first by this
    convention, so it's never what's left over to be counted as burnt.

    Same precondition the two-line decrement this replaces always had:
    `color` must already have a positive balance in state.mana_pool --
    callers only ever reach this after their own legality gate
    (pool_spend_options / drl_env's _filter_mana_legal) already confirmed
    that."""
    pool = state.mana_pool
    single = state.mana_pool_single_pip
    total = pool[color]
    tagged = single.get(color, 0)
    if tagged >= total:  # no untagged pip of this color remains
        single[color] = tagged - 1
        if single[color] <= 0:
            del single[color]
    pool[color] = total - 1
    if pool[color] <= 0:
        del pool[color]


def execute_pool_spend(state, color):
    """Spend one floating pip of `color` toward the pending cost: its own
    matching colored need first, else outstanding generic -- which color to
    preserve for a possible same-phase second spell is the agent's decision,
    made explicitly here."""
    pending = state.pending_resolution
    spend_one_pip(state, color)
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


def _pool_units(pool):
    """A floating pool as can_pay units: one singleton entry per pip."""
    return [frozenset({color}) for color, n in pool.items() for _ in range(n)]


def pool_can_pay(pool, cost):
    """True iff the floating `pool` ALONE already covers `cost` -- specific
    pips (W/U/B/R/G/C) by their own symbol, generic by any leftover.

    Not the affordability gate: that is plan_payment, which also counts what is
    still tappable. This is the narrower "could I pay without touching another
    source?" question, kept because it is genuinely a different one.

    Answered by can_pay over pool-only units rather than by its own cost-
    reduction walk. That walk (own-color pips first, then leftovers against
    generic) was a second, independent implementation of the same feasibility
    question, and on singleton units can_pay reduces to exactly it -- Hall's
    condition per colour collapses to "enough pips of each colour", and the
    count check covers generic. One algorithm, not two that can drift.
    Pure (no mutation)."""
    return can_pay(_pool_units(pool), cost)


def source_mana_units(state, permanent):
    """The mana units `permanent` could still contribute, or () if it can't be
    tapped for mana right now. A "unit" is ONE mana symbol, represented as the
    frozenset of colors that symbol could be -- singleton for anything with a
    determined output, wider for a color CHOICE.

    Shares tappable_for_mana with mana_ability_options, so the affordability
    check and the list of abilities actually offered cannot disagree about what
    is usable.

    That shared gate excludes mana_extra_choose (Saruli Caretaker), and here the
    exclusion is load-bearing rather than incidental: Saruli's cost taps ANOTHER
    creature, which in spy_combo is
    usually itself a mana source (Overgrown Battlement, Wall of Roots, Lotus
    Petal). Counting Saruli as free supply would say two units are available
    where only one is -- Saruli CONSUMES the other creature -- and an
    overcount strands the payment. Deliberately conservative: it can hide a
    line where Saruli upgrades a mana creature's fixed color into any color
    (owner-accepted 2026-08-17, recorded as a known false negative). The agent
    can still float Saruli by hand in its own main phase and then cast."""
    if not tappable_for_mana(state, permanent):
        return ()
    spec = registry.EFFECT_REGISTRY.get(permanent.card_def.effect_id, {}).get("mana")
    granted = _granted_mana_colors(state, permanent)
    if spec is None and not granted:
        return ()

    # Utopia Sprawl's bonus rides along with ANY tap of this permanent, so it
    # is added to whatever the native ability produced, never chosen against.
    bonus = [frozenset({c}) for c in _bonus_mana_symbols(state, permanent)]

    if spec is None:  # a grant with no native mana ability of its own
        native = [granted]
    elif spec[0] == "flexible":
        # One symbol either way, so the native choices and any granted color
        # collapse into one unit exactly -- no information is lost.
        native = [set(spec[1]) | granted]
    else:
        # Every other kind produces a DETERMINED multiset (fixed/fixed_multi/
        # tron/count/count_all). Ask mana_output rather than approximating, so
        # Rakdos Carnarium's two DIFFERENT symbols both count and an online Tron
        # land's three do -- the pre-float-first solver's fixed-at-1 undercount
        # is exactly the bug that costs. mana_output already appends the bonus,
        # so strip it back off and re-add it once at the end.
        full = mana_output(permanent, state)
        symbols = full[:len(full) - len(bonus)] if bonus else full
        if granted and len(symbols) == 1:
            native = [set(symbols) | granted]  # one symbol either way -- still exact
        else:
            # No grant, or a grant on a MULTI-symbol source. The latter is a
            # genuine either/or -- all the native symbols, OR one symbol of a
            # granted color, never a mix -- and a flat unit list cannot express
            # that. Widening one native unit would claim both at once (Tron
            # Tower reading as "{C}{C}{C}, one of which may be {G}"), which
            # OVERSTATES supply and would strand a payment, so the grant is
            # dropped instead: never wrong about the count, and it only hides
            # the line where the grant's color was the point. Unreachable in
            # the current catalog anyway -- Abundant Growth enchants a land,
            # and every land here is single-symbol.
            native = [{s} for s in symbols]
    return [frozenset(colors) for colors in native] + bonus


def available_mana_units(state):
    """Every mana unit the ACTIVE player could still put toward a cost right
    now: one per floating pool pip, plus whatever each untapped, available
    source would produce (source_mana_units). Each unit is one symbol, as the
    frozenset of colors it could be.

    A pool pip and a not-yet-tapped source symbol are interchangeable for
    affordability, which is the whole point of the representation -- it is why
    cast-then-pay needs no separate "pool" and "sources" reasoning anywhere.

    DELIBERATELY NOT CACHED, unlike _enchanting/_cached_mana_ability_options.
    Those are safe to memoize per state object because their only callers are
    legal_action_mask sweeps, which never mutate state mid-sweep and which reset
    the caches on both sides. plan_payment is NOT sweep-only -- cards call it
    directly (red_cards' _makeshift_munitions_legal) and so do tests -- so a
    state-keyed cache populated outside a sweep is never invalidated and goes
    stale the moment anything taps. That was tried and it silently reported a
    just-floated pool as still empty. Measured at ~1.5x the sweep cost of the
    pool-only check it replaces, which is not worth a correctness hazard.
    Pure (no mutation)."""
    units = _pool_units(state.mana_pool)
    for p in state.battlefield:
        units.extend(source_mana_units(state, p))
    return units


def can_pay(units, cost):
    """EXACT: could `cost` be paid using `units` (available_mana_units' shape)?

    Two conditions, and together they are both necessary and sufficient:
      1. there are at least as many units as pips demanded in total -- generic
         accepts any color, so for it only the COUNT matters;
      2. Hall's condition on the colored half: for every non-empty subset S of
         the colors actually demanded, the units able to produce some color in
         S are at least the pips demanded across S.
    (2) alone permits a colored assignment to exist; (1) then guarantees enough
    units are left over to cover generic. Neither implies the other:
    {G:2} against [{G,R}, {R}] fails (2) while passing (1); {G:1, generic:1}
    against [{G}] fails (1) while passing (2).

    Exactness matters in BOTH directions here, which is why this is Hall's
    condition and not the greedy tap-solver this engine used before float-first:
    a false positive lets a payment begin that cannot finish (all-False mask,
    hard error -- there is no Abandon payment), and a false negative hides a
    legal cast from the agent, which is a rules-faithfulness bug in its own
    right. At most six colors can be demanded, so the subset sweep is at most 63
    iterations and in real costs (one to three colors) at most seven.

    Over-production is free: tapping a source the payment doesn't need only
    floats extra mana, so every unit counts as available with no "should I tap
    it" decision folded in. That decision stays entirely the agent's.
    Pure (no mutation)."""
    need = {color: cost.get(color, 0) for color in POOL_COLORS if cost.get(color, 0) > 0}
    if len(units) < sum(need.values()) + cost.get("generic", 0):
        return False
    demanded = list(need)
    for size in range(1, len(demanded) + 1):
        for subset in combinations(demanded, size):
            wanted = frozenset(subset)
            if sum(need[c] for c in subset) > sum(1 for u in units if u & wanted):
                return False
    return True


PAYMENT_PENDING_KINDS = frozenset({"pay_cost", "pay_unless"})

# The steps of an in-flight cast that come BEFORE 601.2f. Real Magic activates
# mana abilities at 601.2f -- after modes and X (601.2b), targets (601.2c) and
# the total cost (601.2e) are settled, and immediately before paying (601.2g).
# It does NOT let a player tap for mana partway through choosing them. So a mana
# ability is illegal while one of these is open, and the payment window that
# follows is where the mana is actually produced.
#
# This is a faithfulness FIX, not a restriction: the engine used to allow mana
# abilities here (gate-free, "legal in ANY priority window"), which is right for
# a priority window but wrong mid-cast, since no player receives priority during
# 601.2. It also closes a stranding hole cast-then-pay would otherwise open --
# the choices below re-check affordability per option, so a tap taken between
# announcing and paying could leave EVERY option unpayable and the mask empty
# (found live: Gurmag Angler's delve step with the black source tapped for blue
# after announcing, every "Delve N" button illegal).
_CASTING_STEP_PENDING_KINDS = frozenset({
    "choose_cast_mode", "choose_cast_x", "choose_delve_amount", "choose_cast_copy",
})


def mid_cast(state):
    """Is a cast in flight and not yet at 601.2f? No mana ability may be
    activated while this holds.

    Two sources, because the answer is not always derivable from the pending
    KIND. The kinds above only ever occur mid-cast. choose_graveyard_card does
    not -- it is equally a resolution-time choice (Masked Vandal's ETB, Relic of
    Progenitus), where mana abilities are perfectly legal -- so delve's exile
    stamps `mid_cast` on its own pendings instead
    (resolution.begin_exile_n_from_graveyard)."""
    pending = state.pending_resolution
    return pending is not None and (
        pending["kind"] in _CASTING_STEP_PENDING_KINDS or bool(pending.get("mid_cast"))
    )


def payment_in_progress(state):
    """Is the active player being asked for mana right now? Real Magic lets a
    player activate a mana ability whenever they have priority, whenever they
    are casting/activating something that needs a mana payment, AND whenever a
    rule or effect asks for a mana payment (605.3a). This is the last two cases,
    and it is what makes payment-time tapping legal in EVERY phase while
    SPECULATIVE floating is restricted to the active player's own main phase
    (see drl_env._actions_mana._mana_timing_legal for that restriction and its
    AUTHORIZED SIMPLIFICATION marker).

    "pay_unless" (Ward, Spell Pierce -- "counter it unless you pay {2}") is a
    payment window, not merely a yes/no prompt: 605.3a's third case is exactly
    "a rule or effect asks for a mana payment", so the payer may tap for it
    right then. It carries no stranding risk of its own -- declining stays legal
    throughout -- so outstanding_cost deliberately does NOT report it; only the
    timing gate cares. begin_pay_unless flips active_idx to the payer first, so
    state.battlefield here is already THEIR board."""
    pending = state.pending_resolution
    return pending is not None and pending["kind"] in PAYMENT_PENDING_KINDS


def outstanding_cost(state):
    """The cost an open payment still owes, or None if none is open. Whatever
    this returns must stay payable, so it is what every mid-payment legality
    gate checks against (see this module's own STRANDING INVARIANT).

    Only "pay_cost". The gap between committing to a cast and the payment
    actually opening (choosing modes, X, a delve amount, which graveyard copy)
    used to need protecting here too, because mana abilities were legal across
    it; mid_cast now makes them illegal there outright, per 601.2f, so nothing
    can consume supply in that gap for this to defend."""
    pending = state.pending_resolution
    if pending is not None and pending["kind"] == "pay_cost":
        return pending["remaining"]
    return None


def units_after(state, tapped=(), spent=(), produced=()):
    """available_mana_units as it WOULD be after a hypothetical mana action:
    every permanent in `tapped` loses its still-available units, every color in
    `spent` loses one pool pip, and each entry in `produced` (an iterable of
    colors) adds one unit that could be any of them.

    One helper for all three mid-payment actions, because all three change the
    unit list and only the bookkeeping differs:
      tap a source   tapped=[source], produced=[{s} for s in mana_output(...)]
      mana filter    tapped=[source], spent=[input], produced=[output_colors]
      Saruli's cost  tapped=[the creature it taps], produced=[COLORS]
    A tap looks like it could never strand a payment, since it only adds mana --
    but a source with a color CHOICE counts while untapped as one unit that
    could be ANY of its colors, and tapping collapses it to one concrete color.
    That narrowing stranded a real game: Jagged Barrens ({B} or {R}) counted
    toward an {R} cost, then tapped for {B}, leaving {R} owed and no red source.

    Removal is by VALUE (frozensets compare equal), so it does not matter
    whether the entry dropped for a spent pip came from the pool or from a
    source that produces the same color -- they are interchangeable to can_pay
    by construction. Pure (no mutation)."""
    units = list(available_mana_units(state))
    for permanent in tapped:
        for unit in source_mana_units(state, permanent):
            units.remove(unit)
    for color in spent:
        units.remove(frozenset({color}))
    units.extend(frozenset(colors) for colors in produced)
    return units


def payment_survives(state, units):
    """Could an in-flight payment still be finished from `units`? True when no
    payment is in flight at all, so callers can ask unconditionally.

    This is the second half of the STRANDING INVARIANT at the top of this
    module: every action available while a payment is open is gated on it, so
    the agent can never reach a state where it owes mana and has no legal way to
    produce it. Pure (no mutation)."""
    owed = outstanding_cost(state)
    return owed is None or can_pay(units, owed)


def plan_payment(state, cost):
    """The one affordability gate every cast/activate legality check calls:
    could `cost` be paid right now from the floating pool PLUS whatever is still
    tappable (available_mana_units + can_pay)? Returns a truthy sentinel when
    payable, else None, so every existing `plan_payment(...) is not None` call
    site reads the new semantics with no edit.

    Legality only. It decides WHETHER a cost can be paid, never HOW -- which
    sources to tap, in what order, and which color each flexible one makes are
    all the agent's own actions during the payment. That separation is the point:
    the pre-float-first version of this function returned an actual tap plan and
    execute_payment applied it, which is exactly how the engine used to take the
    mana decision away from the agent. Pure (no mutation)."""
    return True if can_pay(available_mana_units(state), cost) else None
