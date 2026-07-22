"""Mana system: what a permanent produces, paying a cost (batch and
interactive/one-tap-at-a-time), and pure payment planning/legality checks.

References game.registry's EFFECT_REGISTRY and its derived views
(SIMPLE_MANA_SOURCE_EFFECTS, _FLEXIBLE_SOURCE_CHOICES) only from inside
function bodies, via `registry.NAME` -- never as a bare name imported at
module load time. registry.py imports the deck modules, which don't
import mana.py directly, so this isn't strictly part of that cycle today,
but keeping the same lazy-lookup convention here too means mana.py stays
safe to import from anywhere without caring about ordering.
"""

from . import registry
from .cards import EffectId
from .resolution import begin_resolution, complete_resolution

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
        for aura in state.battlefield
        if aura.flags.get("enchanting") is permanent and "bonus_mana_color" in aura.flags
    ]


def _granted_mana_colors(state, permanent):
    """Colors a competing granted ability (Abundant Growth's "{T}: Add one
    mana of any color") lets this permanent tap for, in ADDITION to (not
    replacing) its own native ability -- the model chooses one or the
    other each time it taps, unlike _bonus_mana_symbols above. Union
    across every Aura enchanting this permanent, in case more than one
    ever grants colors to the same land."""
    granted = set()
    for aura in state.battlefield:
        if aura.flags.get("enchanting") is permanent:
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
        # one (docs/MADNESS_DECKS_PLAN.md item 9).
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
    else:
        raise ValueError(f"{permanent.card_def.name} is not a simple mana source")
    return output + _bonus_mana_symbols(state, permanent)


def pay_cost(state, cost, tap_choices):
    """Execute an already-decided payment (choose_taps_for_cost/
    plan_payment decide; this only executes).

    tap_choices: list of (permanent, color_choice_or_None). Must be untapped
    permanents on state.battlefield with a simple mana-source effect. Raises
    ValueError if the plan is invalid or doesn't cover `cost`; on success,
    taps every permanent in tap_choices and returns.
    """
    for permanent, _color in tap_choices:
        if permanent not in state.battlefield:
            raise ValueError(f"{permanent.card_def.name} is not on the battlefield")
        if permanent.tapped:
            raise ValueError(f"{permanent.card_def.name} is already tapped")
        if permanent.card_def.effect_id not in registry.SIMPLE_MANA_SOURCE_EFFECTS:
            raise ValueError(f"{permanent.card_def.name} is not a simple mana source")
    if len({id(p) for p, _c in tap_choices}) != len(tap_choices):
        raise ValueError("tap_choices repeats the same permanent")

    produced = []
    for permanent, color_choice in tap_choices:
        produced.extend(mana_output(permanent, state, color_choice))

    remaining = dict(cost)
    leftover = 0
    for symbol in produced:
        need = remaining.get(symbol, 0)
        if need > 0:
            remaining[symbol] = need - 1
        else:
            leftover += 1
    generic_needed = remaining.get("generic", 0)
    remaining["generic"] = max(0, generic_needed - leftover)
    if not _cost_satisfied(remaining):
        raise ValueError(f"tap_choices do not cover cost {cost} (produced {produced})")

    for permanent, _color in tap_choices:
        permanent.tapped = True


def choose_taps_for_cost(state, cost):
    """Default source-selection: prefer non-Tron mana for costs. Returns a
    tap_choices list usable by pay_cost, or None if the cost can't
    currently be paid from simple mana sources.

    Legality-only, like the rest of plan_payment's machinery -- nothing
    here ever actually taps a real permanent for real gameplay (every
    live cast always pays interactively via begin_pay_cost/
    tap_cost_options; execute_payment, the only caller that would apply
    this function's return value for real, has no caller of its own
    anywhere in this codebase). That's what makes tapping every eligible
    non-flexible source unconditionally (pass 1 below) safe: there's no
    real permanent state to waste.

    Two passes, mirroring pay_cost's own per-symbol bookkeeping exactly
    (own live need first, any leftover toward generic -- _apply below),
    so `remaining` reaching all-zero here is a provably accurate
    predictor of what pay_cost will actually accept later:
      1. Every non-flexible source (fixed/fixed_multi/tron/count) taps
         unconditionally, crediting its REAL output via mana_output(p,
         state) -- not a single-symbol-per-effect_id approximation. This
         is what makes a "fixed_multi" source's two different symbols
         both count (Rakdos Carnarium's {B}{R}), and, as a side effect,
         also correctly credits "count" sources' full variable total
         (Overgrown Battlement) instead of the old fixed-at-1 undercount
         (docs/MADNESS_DECKS_PLAN.md item 9).
      2. Flexible sources tap one at a time, each choosing whichever
         color is still live (falling back to an arbitrary choice once
         only generic remains), same preference order the original
         single-pass version used.
    """
    untapped = [
        p for p in state.battlefield
        if not p.tapped and p.card_def.effect_id in registry.SIMPLE_MANA_SOURCE_EFFECTS
    ]
    untapped.sort(key=lambda p: p.card_def.effect_id == EffectId.TRON_LAND)

    chosen = []
    remaining = dict(cost)
    pool = list(untapped)

    def _apply(produced_symbols):
        leftover = 0
        for symbol in produced_symbols:
            need = remaining.get(symbol, 0)
            if need > 0:
                remaining[symbol] = need - 1
            else:
                leftover += 1
        if leftover:
            remaining["generic"] = max(0, remaining.get("generic", 0) - leftover)

    for p in list(pool):
        if _cost_satisfied(remaining):
            break
        if p.card_def.effect_id in registry._FLEXIBLE_SOURCE_CHOICES:
            continue  # pass 2
        if _granted_mana_colors(state, p):
            continue  # pass 2.5 -- might be better served by its grant than its own native color
        _apply(mana_output(p, state))
        chosen.append((p, None))
        pool.remove(p)

    for p in list(pool):
        if _cost_satisfied(remaining):
            break
        effect = p.card_def.effect_id
        if effect not in registry._FLEXIBLE_SOURCE_CHOICES:
            continue
        choices = registry._FLEXIBLE_SOURCE_CHOICES[effect]
        color_choice = next((c for c in choices if remaining.get(c, 0) > 0), next(iter(choices)))
        _apply(mana_output(p, state, color_choice))
        chosen.append((p, color_choice))
        pool.remove(p)

    # Pass 2.5: permanents carrying a competing granted ability (Abundant
    # Growth) -- a runtime, per-instance fact registry._FLEXIBLE_SOURCE_CHOICES
    # can't see (that view is built once from EFFECT_REGISTRY, keyed by
    # effect_id, so it can never reflect which specific permanent happens
    # to be enchanted this game). Choose whichever of {native color,
    # granted colors} is still useful, same "arbitrary once only generic
    # remains" fallback pass 2 uses above -- native kind is assumed
    # "fixed" here since every real card that can receive this grant
    # today is a basic land.
    for p in list(pool):
        if _cost_satisfied(remaining):
            break
        granted = _granted_mana_colors(state, p)
        if not granted:
            continue
        native_spec = registry.EFFECT_REGISTRY[p.card_def.effect_id]["mana"]
        native_choices = {native_spec[1]} if native_spec[0] == "fixed" else set()
        candidates = native_choices | granted
        color_choice = next((c for c in candidates if remaining.get(c, 0) > 0), None)
        if color_choice is None or color_choice in native_choices:
            _apply(mana_output(p, state))
            chosen.append((p, None))
        else:
            _apply(mana_output(p, state, color_choice))
            chosen.append((p, color_choice))
        pool.remove(p)

    if not _cost_satisfied(remaining):
        return None
    return chosen


def begin_pay_cost(state, cost, on_complete):
    """Interactive mana payment: the model taps one source at a time (see
    tap_cost_options/execute_tap_cost_option) until `cost` is fully
    covered, instead of an automatic solver picking every tap at once the
    way choose_taps_for_cost/pay_cost still do (kept for pure legality
    checks -- plan_payment(state, cost) is not None -- since "can this be
    paid at all" is a feasibility question, not a strategic one).

    pool_delta tracks this payment attempt's net effect on state.mana_pool
    (+1 per symbol floated by a tap here, -1 per symbol spent from the pool
    here) so abandon_pay_cost can undo exactly this attempt's contribution
    without disturbing mana that was already floating before it began.

    A cost already fully covered before any tap/spend (e.g. Lotus Petal's
    empty {} cast cost) completes immediately -- same check
    execute_tap_cost_option/execute_pool_spend each run after applying a
    tap/spend, just also run here for the zero-input case, which neither
    of those is ever called for. Without this, the resolution opens with
    nothing left to pay and nothing able to close it, leaving "Abandon
    payment" as the only legal action -- softlocking the cast forever
    instead of resolving it."""
    begin_resolution(state, "pay_cost", on_complete, remaining=dict(cost), tapped=[], pool_delta={})
    if _cost_satisfied(state.pending_resolution["remaining"]):
        complete_resolution(state)


def _float_produced_mana(pool, pool_delta, produced_symbols):
    """Every tap's output floats into the mana pool, unconditionally -- a
    tap never directly pays any part of a cost, including generic. Paying
    a cost, generic included, is always a separate explicit
    execute_pool_spend action from here on (MANA_POOL_PLAN.md)."""
    for symbol in produced_symbols:
        pool[symbol] = pool.get(symbol, 0) + 1
        pool_delta[symbol] = pool_delta.get(symbol, 0) + 1


def tap_cost_options(state):
    """While a pay_cost resolution is pending: every (name, color_choice,
    is_filter) tap option still available, one per distinct source *name*
    (not per physical permanent -- same-named untapped sources are
    interchangeable, so this stays a small bounded list regardless of how
    many copies are in play). color_choice is None for fixed-color/Tron
    sources; is_filter marks Barrels/Conduit Pylons used in their
    colored-pip filter mode -- offered for any of the 5 colors, same as a
    flexible source, never gated on whether that exact color still has a
    live need: every tap here only ever floats to the pool for a later,
    separate spend decision (execute_tap_cost_option/execute_pool_spend),
    so there's no way for an untimely tap to be "wasted."""
    pending = state.pending_resolution
    remaining = pending["remaining"]
    if _cost_satisfied(remaining):
        return []

    tapped_ids = {id(p) for p, _is_filter in pending["tapped"]}
    options = []
    seen = set()

    for p in state.battlefield:
        if p.tapped or id(p) in tapped_ids:
            continue
        effect = p.card_def.effect_id
        spec = registry.EFFECT_REGISTRY.get(effect, {}).get("mana")
        if spec is not None:
            # Saruli Caretaker: not offered as a mana source unless its own
            # extra cost (tap another untapped creature) is currently payable.
            extra_available = registry.EFFECT_REGISTRY.get(effect, {}).get("mana_extra_available")
            if extra_available is not None and not extra_available(state, p):
                continue
            kind = spec[0]
            if kind in ("fixed", "fixed_multi", "tron", "count"):
                key = (p.card_def.name, None, False)
                if key not in seen:
                    seen.add(key)
                    options.append(key)
            elif kind == "flexible":
                for color in spec[1]:
                    key = (p.card_def.name, color, False)
                    if key not in seen:
                        seen.add(key)
                        options.append(key)

    # Abundant Growth's granted colors -- a runtime, per-instance fact, so
    # this can't fold into the per-effect_id "flexible" branch above (every
    # Forest shares one effect_id regardless of which specific copy, if
    # any, is enchanted).
    for p in state.battlefield:
        if p.tapped or id(p) in tapped_ids:
            continue
        for color in _granted_mana_colors(state, p):
            key = (p.card_def.name, color, False)
            if key not in seen:
                seen.add(key)
                options.append(key)

    for p in state.battlefield:
        if id(p) in tapped_ids:
            continue
        effect = p.card_def.effect_id
        if registry.EFFECT_REGISTRY.get(effect, {}).get("filter_mana") is None:
            continue
        already_used = (
            p.flags.get("used_this_turn", False) if effect == EffectId.BARRELS_OF_BLASTING_JELLY
            else p.tapped
        )
        if already_used:
            continue
        for color in COLORS:
            key = (p.card_def.name, color, True)
            if key not in seen:
                seen.add(key)
                options.append(key)

    return options


def execute_tap_cost_option(state, name, color_choice, is_filter):
    pending = state.pending_resolution
    tapped_ids = {id(p) for p, _is_filter in pending["tapped"]}

    def _available(p):
        if id(p) in tapped_ids or p.card_def.name != name:
            return False
        if is_filter:
            return not (
                p.flags.get("used_this_turn", False) if p.card_def.effect_id == EffectId.BARRELS_OF_BLASTING_JELLY
                else p.tapped
            )
        return not p.tapped

    matching = [p for p in state.battlefield if _available(p)]
    if is_filter:
        permanent = matching[0]
    else:
        # Same-named untapped sources are normally fully interchangeable in
        # this engine -- an attached bonus/granted-mana Aura breaks that for
        # the first time, since not every same-named permanent necessarily
        # produces the same output anymore. Prefer whichever one produces
        # the most mana right now (e.g. a Utopia-Sprawl-enchanted Forest
        # over a plain one, or specifically the Abundant-Growth-enchanted
        # one when color_choice is a color only it grants -- mana_output
        # raises for a same-named permanent that can't actually produce
        # this particular color_choice, so that's treated as "never
        # preferred," not a crash): strictly at least as good as picking
        # the first match arbitrarily, never wrong.
        def _output_len(p):
            try:
                return len(mana_output(p, state, color_choice))
            except ValueError:
                return -1

        permanent = max(matching, key=_output_len)
    pending["tapped"].append((permanent, is_filter))  # is_filter recorded so abandon_pay_cost can reverse it correctly

    if is_filter:
        if permanent.card_def.effect_id == EffectId.BARRELS_OF_BLASTING_JELLY:
            permanent.flags["used_this_turn"] = True
        else:
            permanent.tapped = True
        # The filter ability itself costs {1} (real cards: "{1}: Add one
        # mana of any color" / "{1}, T: Add one mana of any color") -- a
        # pure color-fix, never a net mana gain -- represented as an extra
        # generic now owed in this resolution, paid the same explicit
        # execute_pool_spend way as everything else.
        pending["remaining"]["generic"] = pending["remaining"].get("generic", 0) + 1
        _float_produced_mana(state.mana_pool, pending["pool_delta"], [color_choice])
    else:
        permanent.tapped = True
        produced = mana_output(permanent, state, color_choice)
        _float_produced_mana(state.mana_pool, pending["pool_delta"], produced)
        # spy_combo: Lotus Petal sacrifices itself, Saruli Caretaker also
        # taps another creature, Wall of Roots may die on its 5th use --
        # each an optional per-effect side effect of a normal tap, mirrored
        # by on_tap_undo in abandon_pay_cost below.
        on_tap = registry.EFFECT_REGISTRY.get(permanent.card_def.effect_id, {}).get("on_tap")
        if on_tap is not None:
            on_tap(state, permanent)

    # A tap never directly pays any part of `remaining` anymore -- only
    # execute_pool_spend can complete this resolution now.


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
    matching colored need first, else outstanding generic -- same
    preference order a fresh tap uses, just decided explicitly here rather
    than automatically."""
    pending = state.pending_resolution
    pool = state.mana_pool
    pool[color] -= 1
    if pool[color] <= 0:
        del pool[color]
    pending["pool_delta"][color] = pending["pool_delta"].get(color, 0) - 1

    remaining = pending["remaining"]
    need = remaining.get(color, 0)
    if need > 0:
        remaining[color] = need - 1
    else:
        remaining["generic"] = max(0, remaining.get("generic", 0) - 1)

    if _cost_satisfied(remaining):
        complete_resolution(state)


def abandon_pay_cost(state):
    """Reverses every tap made so far in a pending pay_cost resolution --
    plus this attempt's entire net effect on the floating mana pool (taps
    that floated mana here, pool spends made here) -- and cancels it
    outright, no on_complete call, as if the action that began paying this
    cost was never chosen. Without this, tapping a flexible/filter source
    for a color that turns out not to be needed could strand the game with
    an unpayable remaining cost and zero legal actions -- real Magic's
    actual rule is that being unable to complete a cost undoes the whole
    action, not that every choice leading there must be prevented in
    advance. Safe to call any time a pay_cost resolution is pending: the
    action that triggered payment never touches hand/battlefield/graveyard
    until its own on_complete fires, so undoing the taps and pool_delta
    alone is a complete, correct undo."""
    pending = state.pending_resolution
    for permanent, is_filter in pending["tapped"]:
        if is_filter and permanent.card_def.effect_id == EffectId.BARRELS_OF_BLASTING_JELLY:
            permanent.flags["used_this_turn"] = False
        else:
            permanent.tapped = False
            if not is_filter:
                on_tap_undo = registry.EFFECT_REGISTRY.get(permanent.card_def.effect_id, {}).get("on_tap_undo")
                if on_tap_undo is not None:
                    on_tap_undo(state, permanent)
    for color, delta in pending["pool_delta"].items():
        remaining_amount = state.mana_pool.get(color, 0) - delta
        if remaining_amount > 0:
            state.mana_pool[color] = remaining_amount
        else:
            state.mana_pool.pop(color, None)
    state.pending_resolution = None


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


def plan_payment(state, cost):
    """Pure (no mutation): decide how `cost` could be paid right now, via
    simple sources or, for a single missing colored pip, the Barrels of
    Blasting Jelly / Conduit Pylons mana-filter fallback. Returns an opaque
    plan for execute_payment, or None if unpayable right now.

    First folds in whatever the floating pool alone could already cover
    (_reduce_cost_by_pool) -- without this, a cost payable via already-
    banked mana but no remaining untapped sources would wrongly read as
    illegal, hiding a real action from the model."""
    cost = _reduce_cost_by_pool(state.mana_pool, cost)
    taps = choose_taps_for_cost(state, cost)
    if taps is not None:
        return ("simple", cost, taps)

    needed_colors = [c for c in COLORS if cost.get(c, 0) > 0]
    if len(needed_colors) == 1 and cost[needed_colors[0]] == 1:
        filtered_cost = {"generic": cost.get("generic", 0) + 1}
        filtered_taps = choose_taps_for_cost(state, filtered_cost)
        if filtered_taps is not None:
            used = {id(p) for p, _c in filtered_taps}
            barrels = next(
                (p for p in state.battlefield
                 if p.card_def.effect_id == EffectId.BARRELS_OF_BLASTING_JELLY
                 and not p.flags.get("used_this_turn", False)),
                None,
            )
            if barrels is not None:
                return ("filter", filtered_cost, filtered_taps, barrels)
            pylons = next(
                (p for p in state.battlefield
                 if p.card_def.effect_id == EffectId.CONDUIT_PYLONS
                 and not p.tapped and id(p) not in used),
                None,
            )
            if pylons is not None:
                return ("filter", filtered_cost, filtered_taps, pylons)
    return None


def execute_payment(state, plan):
    kind = plan[0]
    if kind == "simple":
        _, cost, taps = plan
        pay_cost(state, cost, taps)
        return
    _, filtered_cost, taps, filterer = plan
    pay_cost(state, filtered_cost, taps)
    if filterer.card_def.effect_id == EffectId.BARRELS_OF_BLASTING_JELLY:
        filterer.flags["used_this_turn"] = True
    else:
        filterer.tapped = True


if __name__ == "__main__":
    # ponytail self-check: no pytest in this project, mirrors the
    # assert-based demo convention -- run via `python -m game.mana` from
    # src/. Exercises choose_taps_for_cost's rewrite (MADNESS_DECKS_PLAN.md
    # item 9): a fake fixed_multi source (no real dual-symbol card exists
    # yet -- deck assembly out of scope), plus the REAL Overgrown
    # Battlement card to prove the pre-existing count-source undercount is
    # actually fixed, not just theoretically.
    from . import registry as _registry
    from .cards import CardDef, CardType, EffectId as _EffectId
    from .state import GameState, Permanent

    # fixed_multi: one tap of a Rakdos-Carnarium-like source covers both
    # an outstanding B need and an outstanding R need at once -- the exact
    # case a single-symbol-per-source approximation couldn't see at all.
    _filler_backup = _registry.EFFECT_REGISTRY[_EffectId.FILLER]
    _registry.EFFECT_REGISTRY[_EffectId.FILLER] = {"mana": ("fixed_multi", ("B", "R"))}
    # SIMPLE_MANA_SOURCE_EFFECTS is also a derived view frozen once at
    # import time (registry.py) -- unlike EFFECT_REGISTRY itself, patching
    # it back to reflect FILLER's fake "mana" spec needs its own explicit
    # (temporary) update, same lesson drl_env.py's Plot self-check hit
    # first for _FIXED_SOURCE_COLOR before that view got deleted outright.
    _was_simple_source = _EffectId.FILLER in _registry.SIMPLE_MANA_SOURCE_EFFECTS
    _registry.SIMPLE_MANA_SOURCE_EFFECTS.add(_EffectId.FILLER)
    try:
        state = GameState(on_the_play=True)
        state.battlefield = [Permanent(CardDef("Carnarium-ish", CardType.LAND, None, _EffectId.FILLER))]
        assert mana_output(state.battlefield[0], state) == ["B", "R"]
        taps = choose_taps_for_cost(state, {"B": 1, "R": 1})
        assert taps is not None and len(taps) == 1  # a single tap covers both pips
        assert plan_payment(state, {"B": 1, "R": 1}) is not None
    finally:
        _registry.EFFECT_REGISTRY[_EffectId.FILLER] = _filler_backup
        if not _was_simple_source:
            _registry.SIMPLE_MANA_SOURCE_EFFECTS.discard(_EffectId.FILLER)

    # Overgrown Battlement (real card, "count" kind -- one G per Defender
    # you control, itself included): 3 Defenders on the battlefield means
    # ONE tap of Battlement alone produces 3 G. The pre-rewrite solver
    # credited count sources as if they always produced exactly 1,
    # regardless of the real total -- confirm that undercount is gone.
    state = GameState(on_the_play=True)
    state.battlefield = [
        Permanent(CardDef("Overgrown Battlement", CardType.CREATURE, {"G": 1}, _EffectId.OVERGROWN_BATTLEMENT, defender=True)),
        Permanent(CardDef("Wall of Roots", CardType.CREATURE, {"generic": 1, "G": 1}, _EffectId.WALL_OF_ROOTS, defender=True)),
        Permanent(CardDef("Wall of Roots", CardType.CREATURE, {"generic": 1, "G": 1}, _EffectId.WALL_OF_ROOTS, defender=True)),
    ]
    battlement = state.battlefield[0]
    assert mana_output(battlement, state) == ["G", "G", "G"]  # 3 Defenders, itself included
    # Untapped, 0 other sources -- old code would've credited Battlement
    # for only 1 G here (undercount) and returned None for a 3-G cost.
    taps = choose_taps_for_cost(state, {"G": 3})
    assert taps == [(battlement, None)]
    assert plan_payment(state, {"G": 3}) is not None

    # begin_pay_cost: a cost already fully covered before any tap/spend
    # (e.g. Lotus Petal's empty {} cast cost) must complete immediately --
    # historically it opened a pending_resolution nothing could ever close
    # except Abandon payment, softlocking the cast forever.
    state = GameState(on_the_play=True)
    _resolved = []
    begin_pay_cost(state, {}, on_complete=lambda s: _resolved.append(True))
    assert state.pending_resolution is None
    assert _resolved == [True]

    print("mana.py solver self-check: OK")

    # Boggles' two mana-fixing Auras need genuinely different treatment:
    # Utopia Sprawl's bonus is automatic (always on top of the land's own
    # output, no extra choice), Abundant Growth's is a competing ability
    # (the model picks native or granted each tap) -- see mana_output's
    # own module comments. Both exercised directly against a real Forest/
    # Plains, using synthetic Aura permanents (real Utopia Sprawl/Abundant
    # Growth CardDefs, just not attached via the real cast_aura flow).
    state = GameState(on_the_play=True)
    forest = Permanent(CardDef("Forest", CardType.LAND, None, _EffectId.FOREST))
    utopia_sprawl = Permanent(CardDef("Utopia Sprawl", CardType.ENCHANTMENT, {"G": 1}, _EffectId.UTOPIA_SPRAWL))
    utopia_sprawl.flags["enchanting"] = forest
    utopia_sprawl.flags["bonus_mana_color"] = "W"
    state.battlefield = [forest, utopia_sprawl]

    assert mana_output(forest, state) == ["G", "W"]  # native G, plus Utopia Sprawl's automatic bonus
    begin_pay_cost(state, {"W": 1}, on_complete=lambda s: None)
    assert tap_cost_options(state) == [("Forest", None, False)]  # no color choice needed -- the bonus is automatic
    execute_tap_cost_option(state, "Forest", None, False)
    # Pool-only model (MANA_POOL_PLAN.md): a tap only ever floats its
    # output -- both G and W here -- into the pool; it never directly
    # pays a cost, even a color that happens to match. The resolution
    # stays pending until an explicit pool spend actually pays the {W: 1}.
    assert state.pending_resolution is not None
    assert state.mana_pool == {"G": 1, "W": 1}
    execute_pool_spend(state, "W")
    assert state.pending_resolution is None  # {W: 1} now fully paid, via the explicit spend
    assert state.mana_pool.get("G", 0) == 1  # the native G stays floating, unneeded by this cost

    # Abundant Growth: Plains gets a genuinely competing "any of {G, W}"
    # ability -- both its own native W and the grant stay usable.
    state = GameState(on_the_play=True)
    plains = Permanent(CardDef("Plains", CardType.LAND, None, _EffectId.PLAINS))
    abundant_growth = Permanent(CardDef("Abundant Growth", CardType.ENCHANTMENT, {"G": 1}, _EffectId.ABUNDANT_GROWTH))
    abundant_growth.flags["enchanting"] = plains
    abundant_growth.flags["bonus_mana_colors"] = {"G", "W"}
    state.battlefield = [plains, abundant_growth]

    assert mana_output(plains, state) == ["W"]  # native, no color_choice
    assert mana_output(plains, state, "G") == ["G"]  # via the grant
    # A {G: 1} cost is payable only via the grant (Plains' own native
    # color is W) -- confirms choose_taps_for_cost/plan_payment (the
    # upfront legality gate every "Cast X" check uses) sees the grant too,
    # not just the interactive tap_cost_options path.
    assert plan_payment(state, {"G": 1}) is not None
    begin_pay_cost(state, {"G": 1}, on_complete=lambda s: None)
    assert ("Plains", "G", False) in tap_cost_options(state)
    execute_tap_cost_option(state, "Plains", "G", False)
    assert state.pending_resolution is not None and state.mana_pool == {"G": 1}  # floated, not yet spent
    execute_pool_spend(state, "G")
    assert state.pending_resolution is None  # {G: 1} now fully covered via the grant

    # execute_tap_cost_option must pick the ENCHANTED Plains specifically
    # when tapping for the granted color, even with an identical-by-name
    # plain Plains also in play -- same-named sources are normally fully
    # interchangeable in this engine; a granted-mana Aura breaks that for
    # the first time (this is the exact bug a full-decklist smoke test
    # caught: picking an arbitrary same-named Plains raised "has no color
    # choice" whenever it happened to pick the unenchanted one).
    state = GameState(on_the_play=True)
    plain_plains = Permanent(CardDef("Plains", CardType.LAND, None, _EffectId.PLAINS))
    grant_plains = Permanent(CardDef("Plains", CardType.LAND, None, _EffectId.PLAINS))
    abundant_growth2 = CardDef("Abundant Growth", CardType.ENCHANTMENT, {"G": 1}, _EffectId.ABUNDANT_GROWTH)
    abundant_growth2 = Permanent(abundant_growth2)
    abundant_growth2.flags["enchanting"] = grant_plains
    abundant_growth2.flags["bonus_mana_colors"] = {"G", "W"}
    state.battlefield = [plain_plains, grant_plains, abundant_growth2]

    begin_pay_cost(state, {"G": 1}, on_complete=lambda s: None)
    execute_tap_cost_option(state, "Plains", "G", False)
    assert grant_plains.tapped and not plain_plains.tapped

    print("mana.py Aura self-check: OK")

    # Pool-only model, filter mana (MANA_POOL_PLAN.md): Conduit Pylons'
    # colored-pip filter mode used to be offered only for the single
    # color matching exactly one outstanding pip of quantity 1. Now that
    # every tap only ever floats to the pool (never wasted by tapping
    # "too early"), it's offered for any of the 5 colors regardless of
    # what the remaining cost actually looks like -- exercised here
    # against a cost with two different colored needs, neither of
    # quantity 1, which the old eligibility rule would have rejected
    # entirely (zero filter options offered at all).
    state = GameState(on_the_play=True)
    pylons = Permanent(CardDef("Conduit Pylons", CardType.LAND, None, _EffectId.CONDUIT_PYLONS))
    state.battlefield = [pylons]
    begin_pay_cost(state, {"B": 2, "R": 2}, on_complete=lambda s: None)
    options = tap_cost_options(state)
    assert sorted(o for o in options if o[2]) == [("Conduit Pylons", c, True) for c in sorted(COLORS)]
    execute_tap_cost_option(state, "Conduit Pylons", "U", True)
    # Its own {1} activation cost is now owed on top of the original
    # cost, tracked the same explicit way as everything else -- not paid
    # automatically just because a color happened to float.
    assert state.pending_resolution["remaining"]["generic"] == 1
    assert state.mana_pool == {"U": 1}

    print("mana.py filter-mana self-check: OK")
