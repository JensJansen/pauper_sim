"""Mana system (cast-then-pay, real rule 601.2): what a permanent produces,
activating a mana ability to float mana into the pool (activate_mana_source),
spending that pool to pay a cost (begin_pay_cost + execute_pool_spend), and the
affordability check every cast/activate legality gate runs (plan_payment, over
available_mana_units + can_pay).

ORDERING. Real Magic settles modes/targets/X and the total cost (601.2e)
before activating mana abilities (601.2f) and paying (601.2g): a cast is legal
when the pool PLUS still-untapped sources could cover the cost, and sources
are tapped during the payment itself.

THE STRANDING INVARIANT. There is no "Abandon payment" action, so a payment
that cannot finish leaves no legal action at all (an all-False mask, a hard
error). Two things uphold the invariant that a begun payment can always be
finished:
  1. plan_payment is EXACT -- see can_pay.
  2. every action taken while a payment is open is legal only if the payment
     is still completable afterwards (drl_env's own gates call back into
     can_pay for this).

References registry.EFFECT_REGISTRY only inside function bodies -- the
lazy-lookup convention every submodule here uses to stay import-order-safe."""

from itertools import combinations

from . import registry
from .cards import CardType, EffectId
from .resolution import begin_resolution, complete_resolution


def _has_haste(state, permanent):
    """Delegates to stats.has_haste (shared with combat.creature_attack_eligible).
    Lazy import to keep mana.py's load-order convention."""
    from .effects.stats import has_haste
    return has_haste(state, permanent)


def tap_summoning_locked(state, permanent):
    """A CREATURE ability with {T} in its cost can't be activated while
    summoning sick (302.6), unless it has haste. Non-creatures are never
    summoning sick.

    Exception: an ability with no {T} in its cost (mana_no_tap, e.g. Wall of
    Roots) is not gated -- 302.6 restricts only {T}/{Q} abilities."""
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
    """Every Aura on state.battlefield currently enchanting `permanent`.
    Cached per state object, like drl_env's own sweep caches: safe only
    because a legal_action_mask sweep never mutates state mid-sweep, and
    only because reset_mana_cache runs before AND after each sweep -- never
    trust this to outlive one sweep on its own."""
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
    """Utopia Sprawl's mechanic: an Aura with a "bonus_mana_color" flag adds
    that color's symbol on every tap of this permanent, automatic and on
    top of whatever ability was used (contrast _granted_mana_colors,
    Abundant Growth's competing ability)."""
    return [
        aura.flags["bonus_mana_color"]
        for aura in _enchanting(state, permanent)
        if "bonus_mana_color" in aura.flags
    ]


def _granted_mana_colors(state, permanent):
    """Colors a competing granted ability (Abundant Growth's "{T}: Add one
    mana of any color") lets this permanent tap for, IN ADDITION to its
    native ability -- the agent picks one or the other each tap, unlike
    _bonus_mana_symbols above."""
    granted = set()
    for aura in _enchanting(state, permanent):
        granted |= aura.flags.get("bonus_mana_colors", set())
    return granted


def mana_output(permanent, state, color_choice=None, exclude=()):
    """Mana symbols this permanent would produce if tapped for its plain
    mana ability right now. Raises if effect_id isn't a simple source or a
    required/forbidden color_choice is missing/invalid.

    A granted color (_granted_mana_colors) is checked first and, if
    matched, short-circuits the registry-driven dispatch below -- a
    separate ability, not a variant of the native one.

    exclude: permanents to treat as already gone from the battlefield when
    evaluating a "count"/"count_all" predicate (e.g. Overgrown Battlement's
    "per Wall you control") -- see units_after's own docstring for why this
    exists: a hypothetical tap can itself remove a DIFFERENT, still-untapped
    source's own predicate match, which a plain snapshot can't see."""
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
            # Tower doubles to {C}{C}{C} when online; Mine/Power Plant double to {C}{C}.
            output = ["C", "C", "C"] if permanent.card_def.extra["tron_type"] == "Tower" else ["C", "C"]
    elif kind == "fixed":
        if color_choice is not None:
            raise ValueError(f"{permanent.card_def.name} has no color choice")
        output = [spec[1]]
    elif kind == "fixed_multi":
        # ("fixed_multi", (symbol, ...)): all symbols from one tap (Rakdos Carnarium).
        if color_choice is not None:
            raise ValueError(f"{permanent.card_def.name} has no color choice")
        output = list(spec[1])
    elif kind == "flexible":
        choices = spec[1]
        if color_choice not in choices:
            raise ValueError(f"{permanent.card_def.name} cannot produce {color_choice}")
        output = [color_choice]
    elif kind == "count":
        # ("count", symbol, predicate): one symbol per matching permanent you control (Overgrown Battlement).
        if color_choice is not None:
            raise ValueError(f"{permanent.card_def.name} has no color choice")
        symbol, predicate = spec[1], spec[2]
        output = [symbol] * sum(1 for p in state.battlefield if p not in exclude and predicate(p))
    elif kind == "count_all":
        # ("count_all", symbol, predicate): like "count" but over BOTH
        # players' battlefields (Priest of Titania).
        if color_choice is not None:
            raise ValueError(f"{permanent.card_def.name} has no color choice")
        symbol, predicate = spec[1], spec[2]
        output = [symbol] * sum(1 for pl in state.players for p in pl.battlefield if p not in exclude and predicate(p))
    else:
        raise ValueError(f"{permanent.card_def.name} is not a simple mana source")
    return output + _bonus_mana_symbols(state, permanent)


def begin_pay_cost(state, cost, on_complete):
    """CR 601.2f/601.2g: the payment window, entered once `cost` is known
    payable (callers gate on plan_payment first) -- the agent activates
    mana abilities and spends the pool, one action at a time, until `cost`
    is covered.

    `announced` records the cost as it stood when the payment opened,
    alongside `remaining` which shrinks as it's paid -- lets a strand bug
    report distinguish a bad plan_payment from supply consumed afterwards.

    A cost already fully covered before any spend (e.g. Lotus Petal's
    empty {} cast cost) completes immediately."""
    # The STRANDING INVARIANT (see module docstring), checked here so a
    # caller that skipped plan_payment fails with the guilty call site in
    # the traceback, not several actions later as an opaque all-False mask.
    # An explicit raise, not `assert`, since assert no-ops under -O.
    if not can_pay(available_mana_units(state), cost):
        raise AssertionError(
            f"begin_pay_cost opened an unpayable cost {dict(cost)} -- available units "
            f"{sorted(''.join(sorted(u)) for u in available_mana_units(state))}, pool "
            f"{dict(state.mana_pool)}. This caller did not gate on plan_payment, or gated on a "
            f"DIFFERENT cost than it went on to charge."
        )
    begin_resolution(state, "pay_cost", on_complete, remaining=dict(cost), announced=dict(cost))
    if _cost_satisfied(state.pending_resolution["remaining"]):
        complete_resolution(state)


def mana_tap_gate_open(state, permanent):
    """Is `permanent` free of the tap/summoning-sickness half of mana-source
    availability RIGHT NOW? Untapped -- UNLESS mana_no_tap (e.g. Wall of
    Roots: no {T} in the ability's own cost, so its OWN tapped state never
    gates it, same exception tap_summoning_locked already carries for the
    identical reason) -- and not summoning-locked (302.6).

    THE ONE PLACE this specific condition lives, full stop -- not "the one
    place for plain mana sources, duplicated elsewhere for Saruli-shaped
    ones." Until 2026-08-23 this exact check was copy-pasted here AND in
    drl_env._actions_mana._find_mana_extra_source (Saruli Caretaker's own
    source lookup, which can't call tappable_for_mana below because that
    function deliberately excludes mana_extra_choose sources for an
    unrelated reason -- see its docstring). The copies drifted: this one
    gained the mana_no_tap exception, the other didn't, and mid-payment
    that mismatch stranded a real cast of Lotleth Giant when Saruli tapped
    Wall of Roots as its cost (an all-False mask, a hard crash in
    production league training) -- fixing tappable_for_mana ALONE had
    already made the promise, but nothing enforced the duplicate honoring
    it. A shared function makes that class of drift impossible rather than
    merely fixed once: there is now nowhere else this condition could be
    re-typed out of sync. Do not inline this check anywhere else -- import
    and call this."""
    entry = registry.EFFECT_REGISTRY.get(permanent.card_def.effect_id, {})
    no_tap = entry.get("mana_no_tap", False)
    return not ((permanent.tapped and not no_tap) or tap_summoning_locked(state, permanent))


def tappable_for_mana(state, permanent):
    """Can `permanent` be tapped for mana RIGHT NOW? mana_tap_gate_open,
    plus its extra availability condition (Wall of Roots' once-per-turn,
    tracked independently via permanent.flags, not permanent.tapped)
    satisfied.

    Both mana_ability_options and source_mana_units call this, so they
    can't disagree about what's usable. Excludes mana_extra_choose (Saruli
    Caretaker), whose activation needs a choice enumerated atomically
    alongside it -- see mana_tap_gate_open for where THAT source instead
    gets its tap/sickness check from."""
    entry = registry.EFFECT_REGISTRY.get(permanent.card_def.effect_id, {})
    if not mana_tap_gate_open(state, permanent):
        return False
    if entry.get("mana_extra_choose") is not None:
        return False
    extra = entry.get("mana_extra_available")
    return extra is None or extra(state, permanent)


def mana_ability_options(state):
    """Every (name, color_choice) mana ability the active player can
    activate RIGHT NOW, one per distinct source name -- legal in any
    priority window, even mid-resolution (605.1a/605.3b). Lists only what
    CAN produce mana now (tappable_for_mana); spending happens separately.
    Mana filters are a separate pool->pool conversion, not listed here."""
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
        # Abundant Growth's granted colors are a per-instance runtime fact
        # (every Forest shares one effect_id), so they're offered here
        # rather than via the spec branch above.
        for color in _granted_mana_colors(state, p):
            _add((p.card_def.name, color))
    return options


def activate_mana_source(state, permanent, color_choice=None):
    """Activate `permanent`'s mana ability immediately: tap it (unless
    mana_no_tap), add its produced symbols to the pool (float_mana), and
    run any on_tap side effect (Lotus Petal/Treasure self-sac, Wall of
    Roots counter). No pending resolution -- a mana ability resolves at
    once, never using the stack."""
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
    """Adds `symbols` (a mana-producing EVENT's output, e.g. ["G"] or
    ["G", "G", "G"]) to state.mana_pool, one add per symbol.

    Also tags the floated pip in state.mana_pool_single_pip when the event
    added EXACTLY 1 symbol (len(symbols) == 1) -- dynamic per-event, not a
    static per-source-kind rule. See PlayerState.mana_pool_single_pip for
    the full tag rationale.

    taggable=False forces no tag even for a 1-symbol event -- used only by
    a mana filter's output pip, a deliberate pool->pool conversion that
    must never be punished as reflexive tapping."""
    pool = state.mana_pool
    for symbol in symbols:
        pool[symbol] = pool.get(symbol, 0) + 1
    if taggable and len(symbols) == 1:
        single = state.mana_pool_single_pip
        color = symbols[0]
        single[color] = single.get(color, 0) + 1


def discount_departing_source(state, permanent, owner_idx):
    """`permanent` just became available for a fresh tap this phase without
    being tapped for it -- sacrificed, or untapped by an effect after
    already being tapped. Tapping it right then would have been free
    value, so any single-pip-tagged mana of a color it can produce is
    retroactively excused from burn.

    mana_pool_single_pip has no per-source identity, so this discounts
    whichever candidate color currently has the most tagged mana floating
    (state.rng breaks ties over sorted colors).

    Only "fixed" and "flexible" mana-spec kinds qualify -- the kinds that
    always produce exactly one symbol per tap."""
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
    """Decrements one floating pip of `color`, applying the spend-order
    convention: an UNTAGGED (multi-pip-burst-sourced) pip is always spent
    BEFORE a TAGGED (single-pip-sourced) one, so a burst source's own
    unavoidable excess absorbs blame ahead of a genuinely avoidable
    single-pip tap.

    Precondition: `color` must already have a positive balance in
    state.mana_pool -- callers gate on pool_spend_options first."""
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
    # One of three exemptions game.turn._empty_mana_pools checks before tallying a burn as a mistake.
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
    pips by their own symbol, generic by any leftover. Narrower than
    plan_payment (which also counts what's still tappable).

    Answered by can_pay over pool-only units, not a separate reduction
    walk -- one algorithm, not two that can drift. Pure (no mutation)."""
    return can_pay(_pool_units(pool), cost)


def source_mana_units(state, permanent, exclude=()):
    """The mana units `permanent` could still contribute, or () if it can't
    be tapped for mana right now. A "unit" is one mana symbol, as the
    frozenset of colors it could be -- singleton for a determined output,
    wider for a color CHOICE.

    exclude: forwarded to mana_output -- see its own docstring.

    Shares tappable_for_mana with mana_ability_options, so the two can't
    disagree about what's usable.

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

    # Utopia Sprawl's bonus rides along with any tap, added to the native output.
    bonus = [frozenset({c}) for c in _bonus_mana_symbols(state, permanent)]

    if spec is None:  # a grant with no native mana ability of its own
        native = [granted]
    elif spec[0] == "flexible":
        # One symbol either way -- native choices and granted color collapse into one unit.
        native = [set(spec[1]) | granted]
    else:
        # Every other kind produces a DETERMINED multiset (fixed/fixed_multi/
        # tron/count/count_all) -- ask mana_output rather than approximating.
        # It already appends the bonus, so strip it off and re-add once below.
        full = mana_output(permanent, state, exclude=exclude)
        symbols = full[:len(full) - len(bonus)] if bonus else full
        if granted and len(symbols) == 1:
            native = [set(symbols) | granted]  # one symbol either way -- still exact
        else:
            # No grant, or a grant on a MULTI-symbol source: an either/or a flat
            # unit list can't express, so the grant is dropped to avoid
            # OVERSTATING supply (which would strand a payment).
            native = [{s} for s in symbols]
    return [frozenset(colors) for colors in native] + bonus


def available_mana_units(state, exclude=()):
    """Every mana unit the ACTIVE player could still put toward a cost right
    now: one per floating pool pip, plus whatever each untapped, available
    source would produce (source_mana_units).

    exclude: permanents to treat as already gone from the battlefield --
    both their own contribution AND any OTHER source's "count"/"count_all"
    predicate match (see units_after's own docstring for why this exists).
    Forwarded into source_mana_units for every remaining permanent, not
    just used to skip iterating the excluded ones, since a count-based
    source's own predicate scan reads state.battlefield directly and has
    no other way to learn about the exclusion.

    Deliberately NOT cached, unlike _enchanting: plan_payment is called
    directly by cards and tests, not just sweep legal_fns, so a state-keyed
    cache goes stale the moment anything taps between calls. Pure (no
    mutation)."""
    units = _pool_units(state.mana_pool)
    for p in state.battlefield:
        if p in exclude:
            continue
        units.extend(source_mana_units(state, p, exclude=exclude))
    return units


def can_pay(units, cost):
    """EXACT: could `cost` be paid using `units` (available_mana_units' shape)?

    Two conditions, both necessary and sufficient:
      1. total unit count >= total pips demanded (generic accepts any
         color, so only the count matters for it);
      2. Hall's condition on the colored half: for every non-empty subset S
         of demanded colors, units able to produce some color in S are at
         least the pips demanded across S.
    Neither implies the other: {G:2} against [{G,R}, {R}] fails (2) while
    passing (1); {G:1, generic:1} against [{G}] fails (1) while passing (2).

    Exactness matters in both directions: a false positive lets a payment
    begin that cannot finish (all-False mask, hard error); a false negative
    hides a legal cast. At most six colors can be demanded, so the subset
    sweep is at most 63 iterations, far fewer in practice.

    Over-production is free: every unit counts as available with no "should
    I tap it" decision folded in -- that stays the agent's own choice.
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

# The steps of an in-flight cast that come BEFORE 601.2f (mana abilities
# activate at 601.2f, after modes/X/targets/cost are settled, immediately
# before paying at 601.2g). A mana ability is illegal while one of these is
# open, since real Magic gives no priority mid-cast (601.2) -- and it closes
# a stranding hole: a tap taken between announcing and paying could leave
# every remaining option unpayable and the mask empty.
_CASTING_STEP_PENDING_KINDS = frozenset({
    "choose_cast_mode", "choose_cast_x", "choose_delve_amount", "choose_cast_copy",
})


def mid_cast(state):
    """Is a cast in flight and not yet at 601.2f? No mana ability may be
    activated while this holds.

    Two sources: the kinds above only ever occur mid-cast; choose_graveyard_card
    doesn't (it's also a resolution-time choice, e.g. Relic of Progenitus, where
    mana abilities are legal), so delve's exile stamps `mid_cast` on its own
    pending instead (resolution.begin_exile_n_from_graveyard)."""
    pending = state.pending_resolution
    return pending is not None and (
        pending["kind"] in _CASTING_STEP_PENDING_KINDS or bool(pending.get("mid_cast"))
    )


def payment_in_progress(state):
    """Is the active player being asked for mana right now (605.3a's last
    two cases)? Payment-time tapping is legal in EVERY phase, unlike
    speculative floating (restricted to the active player's own main
    phase -- see drl_env._actions_mana._mana_timing_legal).

    "pay_unless" (Ward, Spell Pierce) is a real payment window, not just a
    yes/no prompt -- the payer may tap for it. It carries no stranding risk
    (declining stays legal), so outstanding_cost deliberately does not
    report it."""
    pending = state.pending_resolution
    return pending is not None and pending["kind"] in PAYMENT_PENDING_KINDS


def outstanding_cost(state):
    """The cost an open payment still owes, or None if none is open. What
    every mid-payment legality gate checks against (the STRANDING
    INVARIANT).

    Only "pay_cost" -- mid_cast makes mana abilities illegal during the
    earlier casting-step choices (601.2f), so nothing can consume supply
    in that gap for this to defend."""
    pending = state.pending_resolution
    if pending is not None and pending["kind"] == "pay_cost":
        return pending["remaining"]
    return None


def _mana_tap_removes_self(state, permanent):
    """Would activating `permanent`'s own mana ability right now cause it to
    leave the battlefield -- for ANY reason (toughness reaching 0, an
    unconditional sacrifice cost, etc.), not just "is this tap lethal"?
    Reads a per-card registry predicate (registry.EFFECT_REGISTRY[effect_id]
    ["mana_tap_removes_self"]), the same declarative-per-card pattern
    mana_no_tap/mana_extra_available/on_tap already use, so a future card
    with a similarly self-removing mana ability declares its own version
    rather than this needing to special-case a card by name. False (no
    removal) for every card without the key -- today, every card except
    Wall of Roots (whose own -0/-1 counter can bring it to lethal toughness;
    see its registry entry). See units_after's own docstring for why this
    matters: a hypothetical tap that removes the source can ALSO devalue a
    different, still-untapped "count"-based source (Overgrown Battlement,
    Priest of Titania) whose own output depends on this permanent's
    continued presence -- a real production strand, 2026-08-25 (Lotleth
    Giant, Wall of Roots' 5th activation killing it mid-payment while
    Overgrown Battlement's still-uncounted 2nd Wall was still needed)."""
    spec = registry.EFFECT_REGISTRY.get(permanent.card_def.effect_id, {}).get("mana_tap_removes_self")
    return spec is not None and spec(state, permanent)


def units_after(state, tapped=(), spent=(), produced=(), own_ability_tap=None):
    """available_mana_units as it WOULD be after a hypothetical mana action:
    every permanent in `tapped` loses its still-available units, every color
    in `spent` loses one pool pip, and each entry in `produced` adds one
    unit that could be any of those colors.

    One helper for all three mid-payment actions (tap a source, a mana
    filter, Saruli's cost) -- only the bookkeeping differs:
      tap a source   tapped=[source], produced=[{s} for s in mana_output(...)],
                     own_ability_tap=source
      mana filter    tapped=[source], spent=[input], produced=[output_colors]
      Saruli's cost  tapped=[the creature it taps], produced=[COLORS]

    A tap can strand a payment: a source with a color CHOICE counts while
    untapped as one unit that could be ANY of its colors, and tapping
    collapses it to one concrete color (Jagged Barrens counted toward {R},
    then tapped for {B}, leaving {R} owed with no red source left).

    own_ability_tap: pass the permanent ONLY when this hypothetical action
    is that permanent's OWN mana ability actually being activated (i.e. the
    caller will go on to call activate_mana_source, which is what actually
    runs a card's on_tap side effect) -- never for the other two shapes
    above (a mana filter's execute sets p.tapped directly and a Saruli-cost
    tap calls tap_for_cost, neither ever runs on_tap, so a permanent merely
    APPEARING in `tapped` there must never be treated as possibly dying: a
    nearly-dead Wall of Roots being tapped as Saruli's unrelated cost does
    NOT accumulate a -0/-1 counter and must not be excluded here). When
    _mana_tap_removes_self(state, own_ability_tap) is true, the baseline
    snapshot is computed with that permanent already excluded from the
    battlefield (via available_mana_units' own exclude param) instead of
    merely patched out afterward -- so any OTHER still-untapped "count"-
    based source's own value is correctly recomputed as if it were already
    gone, not left at its stale pre-tap value.

    Removal is by VALUE, so an entry from the pool and one from a same-color
    source are interchangeable. Pure (no mutation)."""
    dying = own_ability_tap is not None and _mana_tap_removes_self(state, own_ability_tap)
    excl = {own_ability_tap} if dying else ()
    units = list(available_mana_units(state, exclude=excl))
    for permanent in tapped:
        if dying and permanent is own_ability_tap:
            continue  # already excluded from the baseline above -- don't double-subtract
        for unit in source_mana_units(state, permanent, exclude=excl):
            units.remove(unit)
    for color in spent:
        units.remove(frozenset({color}))
    units.extend(frozenset(colors) for colors in produced)
    return units


def payment_survives(state, units):
    """Could an in-flight payment still be finished from `units`? True when
    no payment is in flight, so callers can ask unconditionally.

    Second half of the STRANDING INVARIANT: every action while a payment is
    open is gated on this. Pure (no mutation)."""
    owed = outstanding_cost(state)
    return owed is None or can_pay(units, owed)


def plan_payment(state, cost):
    """The one affordability gate every cast/activate legality check calls:
    could `cost` be paid right now from the pool plus whatever is still
    tappable (available_mana_units + can_pay)? Returns a truthy sentinel
    when payable, else None.

    Legality only -- decides WHETHER a cost can be paid, never HOW. Which
    sources to tap, in what order, and which color each flexible one makes
    stay entirely the agent's own actions during the payment. Pure (no
    mutation)."""
    return True if can_pay(available_mana_units(state), cost) else None
