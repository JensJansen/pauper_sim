"""Mana production: native/flexible tap abilities, Saruli-Caretaker-shaped
extra-cost ("tap another creature") sources, mana filters (Conduit Pylons/
Barrels of Blasting Jelly), and Chromatic Star's choose_mana_color.

None of these carry a _pending_gate (605.1a/605.3b -- a mana ability never uses
the stack and doesn't require priority), except the shared choose_color
mana_subdecision buttons and choose_mana_color, which are pending-kind-gated.
That is what makes CR 601.2f work: a mana ability stays legal during an open
payment, which under cast-then-pay is the normal way mana gets produced.

Two rules cut across every entry point here, both added with cast-then-pay
(2026-08-17):
  _mana_timing_legal   WHEN a mana ability may be activated at all -- during a
                       payment (faithful, any phase), or speculatively in the
                       active player's own main phase (the one AUTHORIZED
                       SIMPLIFICATION). Never mid-cast before 601.2f.
  payment_survives     WHETHER this specific activation would leave an open
                       payment finishable. See game.mana's STRANDING INVARIANT:
                       tapping is not automatically safe, because a color CHOICE
                       collapses to one concrete color.

legal(state)/execute(state) factory pairs build_action_table
(drl_env._actions_table) calls once per matching source."""

import game

_mana_ability_options_cache = None  # (state, result) -- one legal_action_mask sweep, same lifecycle as the combat-side battlefield-lookup cache in _actions_combat


def _cached_mana_ability_options(state):
    """Memoizes game.mana_ability_options(state) for one legal_action_mask
    sweep -- every "Tap X for <color>" row's legal() calls it, so a sweep would
    otherwise recompute the identical battlefield scan once per mana row."""
    global _mana_ability_options_cache
    if _mana_ability_options_cache is None or _mana_ability_options_cache[0] is not state:
        _mana_ability_options_cache = (state, game.mana_ability_options(state))
    return _mana_ability_options_cache[1]


def _mana_timing_legal(state):
    """WHEN a mana ability may be activated at all. No pending-resolution gate
    (605.1a/605.3b: a mana ability never uses the stack and doesn't require
    priority), so this is purely a phase/payment question.

    Two windows:
      1. a payment is open (game.payment_in_progress) -- CR 601.2f/605.3a,
         fully faithful, legal in EVERY phase. This is the window cast-then-pay
         runs in: announce, then tap for what the cost turned out to be.
      2. SPECULATIVE floating, with nothing to pay for.

    # AUTHORIZED SIMPLIFICATION (owner-approved 2026-08-17): window 2 is
    # restricted to the active player's own main phase. Real Magic allows
    # floating in any step or phase in which you have priority. The restriction
    # costs very little in practice -- holding mana up is leaving lands
    # UNTAPPED, not floating, and paying for an instant on the opponent's turn
    # goes through window 1 -- so what it actually removes is floating for a
    # mana ability's SIDE EFFECT (Lotus Petal's sacrifice, Wall of Roots'
    # counter) outside a main phase.
    #
    # This subsumes the old `not state.in_cleanup` check that every mana entry
    # point used to carry (its own separate authorized simplification, added
    # 2026-08-10 so floated mana could not dodge the burn signal in cleanup):
    # cleanup is not a main phase, and no payment is open there, so both
    # windows already exclude it and the special case is gone.

    Neither window is open mid-cast, before 601.2f -- see
    game.mana.CASTING_STEP_PENDING_KINDS. That case has to be refused ahead of
    the main-phase test rather than folded into it, because a cast announced in
    a main phase is still in a main phase while its modes/X/delve/copy are being
    chosen."""
    pending = state.pending_resolution
    if pending is not None and (pending["kind"] in game.CASTING_STEP_PENDING_KINDS
                                or pending.get("mid_cast")):
        return False
    return (
        game.payment_in_progress(state)
        or (state.phase in game.turn.SORCERY_SPEED_PHASES and state.active_idx == state.turn_player_idx)
    )


def _mana_ability_legal(name, color):
    """Legal iff a source named `name` can produce `color` right now
    (game.mana_ability_options; color=None for fixed/tron/count sources), the
    timing window allows it (_mana_timing_legal), and tapping it that way leaves
    an open payment still finishable (game.payment_survives).

    No `_pending_gate` attribute, so legal_action_mask always calls this
    regardless of what is pending -- which is what makes CR 601.2f work: a mana
    ability is legal mid-payment, and under cast-then-pay that is the normal
    way mana gets produced."""
    def legal(state):
        if not _mana_timing_legal(state) or (name, color) not in _cached_mana_ability_options(state):
            return False
        source = _cached_mana_source(state, name, color)
        if source is None:
            return False
        produced = [frozenset({symbol}) for symbol in game.mana_output(source, state, color)]
        return game.payment_survives(state, game.units_after(state, tapped=[source], produced=produced))
    return legal


def _find_mana_source(state, name, color):
    """The specific untapped, available permanent named `name` that can produce
    `color` now -- same-named sources are fungible, but `color` narrows to the
    one flexible/granted source that can make it (e.g. an Abundant-Growth land).
    Mirrors mana_ability_options' per-permanent gates (tap-lock, extra cost)."""
    for p in state.battlefield:
        if p.card_def.name != name or p.tapped or game.tap_summoning_locked(state, p):
            continue
        extra = game.EFFECT_REGISTRY.get(p.card_def.effect_id, {}).get("mana_extra_available")
        if extra is not None and not extra(state, p):
            continue
        try:
            game.mana_output(p, state, color)  # raises if this p can't produce `color`
        except ValueError:
            continue
        return p
    return None


_mana_source_cache = None  # (state, {(name, color): Permanent or None}) -- one legal_action_mask sweep, same lifecycle as the combat-side battlefield-lookup cache in _actions_combat


def _cached_mana_source(state, name, color):
    """Memoizes _find_mana_source(state, name, color) per (name, color) for one
    legal_action_mask sweep -- every extra-tap row sharing a (name, color) pair
    (one per target name/slot) would otherwise re-scan state.battlefield from
    scratch; same "profiled, not guessed" caching _cached_battlefield_lookup/
    _cached_mana_ability_options already apply to the analogous per-row scans
    above."""
    global _mana_source_cache
    if _mana_source_cache is None or _mana_source_cache[0] is not state:
        _mana_source_cache = (state, {})
    cache = _mana_source_cache[1]
    key = (name, color)
    if key not in cache:
        cache[key] = _find_mana_source(state, name, color)
    return cache[key]


def _mana_ability_execute(name, color):
    def execute(state):
        p = _find_mana_source(state, name, color)
        game.activate_mana_source(state, p, color)
    return execute


def _find_mana_extra_source(state, name):
    """Same as _find_mana_source, minus the color-producibility check --
    for a mana_extra_choose source (Saruli Caretaker), the color isn't
    chosen until the SECOND stage of its mana_subdecision (601.2f-shaped:
    the cost -- tapping a creature -- is paid before the color-choice
    effect resolves), so there's no color to check availability against
    yet at this point. Only mana_extra_choose currently exists on one card
    with no per-instance color restriction, so no generic color-dependent
    availability case is being dropped here -- see _mana_subdecision_color_
    legal for where that check properly lives instead, against the
    already-resolved source."""
    for p in state.battlefield:
        if p.card_def.name != name or p.tapped or game.tap_summoning_locked(state, p):
            continue
        extra = game.EFFECT_REGISTRY.get(p.card_def.effect_id, {}).get("mana_extra_available")
        if extra is not None and not extra(state, p):
            continue
        return p
    return None


def _mana_extra_choose_legal(name):
    """Saruli-Caretaker-shaped mana ability whose additional cost is tapping
    ANOTHER untapped creature (registry "mana_extra_choose") -- a COST
    CHOICE (602.5g), decided as the FIRST stage of a mana_subdecision (see
    game.resolution.begin_mana_subdecision), not enumerated as a fixed-table
    row anymore. No pending-resolution gate (same reasoning
    _mana_ability_legal's own docstring gives -- a mana ability is legal in
    ANY priority window, even mid-resolution of anything else, 605.1a/
    605.3b) -- gate-free is load-bearing here specifically (confirmed:
    Quirion Ranger + an opposing Ward creature reaches this exact window in
    real league play, see the self-check below). Legal iff a source of this
    name is currently untapped/available AND at least one OTHER creature
    satisfies the source's own mana_extra_choose predicate. Looks up
    extra_pred off the resolved permanent's own card_def.effect_id (not a
    closed-over table-build-time value) -- same reasoning
    _mana_extra_choose_execute gives, handles a token-sourced name
    identically to a real decklist name. Timing follows the shared
    _mana_timing_legal rule, same as every other mana entry point.

    Mid-payment a candidate only counts if tapping IT specifically leaves the
    payment finishable (mana_extra_choose_target_safe). Checking that HERE, and
    not only at the target-choice step, is load-bearing: the choose_target stage
    takes EXCLUSIVE priority (legal_action_mask suppresses every other action
    while a mana_subdecision is open), so activating with no safe target left
    would produce an all-False mask with no way back -- the exact failure this
    gate exists to prevent. Any-not-all: one safe candidate is enough to
    activate, and the target step then offers precisely those."""
    def legal(state):
        if not _mana_timing_legal(state):
            return False
        p = _find_mana_extra_source(state, name)
        if p is None:
            return False
        extra_pred = game.EFFECT_REGISTRY.get(p.card_def.effect_id, {}).get("mana_extra_choose")
        if extra_pred is None:
            return False
        return any(
            q is not p and not q.tapped and extra_pred(q) and mana_extra_choose_target_safe(state, q)
            for q in state.battlefield
        )
    return legal


def mana_extra_choose_target_safe(state, target):
    """Would tapping `target` as a mana_extra_choose source's additional cost
    (Saruli Caretaker's "tap another untapped creature you control") leave an
    open payment still finishable?

    Net effect on supply: `target` loses whatever it could still have produced,
    and one pip of any color arrives once the color stage resolves. Saruli
    itself contributes nothing to remove -- game.source_mana_units deliberately
    excludes mana_extra_choose sources from the count, for exactly the reason
    this function has to exist.

    So it is a SWAP, not a gain, and in spy_combo the creature tapped is usually
    itself a mana source (Overgrown Battlement, Wall of Roots, Lotus Petal).
    Tapping Overgrown Battlement with several defenders out trades several green
    pips for one, which is a real loss and can strand a payment. Shared with
    rl.action_bridge, which masks the target choice with this same predicate so
    the two can never disagree about which creatures are on offer."""
    return game.payment_survives(state, game.units_after(
        state, tapped=[target], produced=[game.COLORS]))


def _mana_extra_choose_execute(name):
    """Resolves the specific source permanent NOW (not deferred) -- the
    pointer-routed target-choice step needs to identity-exclude THIS exact
    copy, not just any creature named `name`, when multiple same-named
    sources exist (two Saruli Caretakers: tapping one to pay for the
    other's ability is legal; tapping itself is not). extra_pred is looked
    up off the resolved permanent's own card_def.effect_id, not
    game.CARD_DEFS[name] -- a token-sourced name (extra_tap_source_names'
    own token_card_defs union below) has no CARD_DEFS entry at all."""
    def execute(state):
        p = _find_mana_extra_source(state, name)
        extra_pred = game.EFFECT_REGISTRY[p.card_def.effect_id]["mana_extra_choose"]
        game.begin_mana_subdecision(state, p, extra_pred)
    return execute


def _mana_subdecision_color_legal(color):
    """Shared "Produce <color>" button, reused by every gate-free mana
    ability with a final choose-a-color step (Saruli Caretaker;
    filter_mana cards) -- legal only mid the choose_color stage of a
    mana_subdecision, and only if whichever ability opened it says this
    color is currently offerable (state.mana_subdecision["can_produce"],
    bound as a closure by that opener -- see game.resolution.
    begin_mana_color_choice's own docstring for what each real caller
    binds). Generic: this function has no idea which ability is asking.

    Mid-payment it must additionally not strand that payment. This is the LAST
    gate in the chain and the one that actually pins the color down: everything
    upstream (the filter's activation, Saruli's activation and target) had to
    reason optimistically about a color the agent had not chosen yet, crediting
    one pip of ANY of the offered colors. Choosing here is where that optimism
    is cashed in, so choosing a color the remaining cost cannot use is refused
    rather than allowed to strand -- the same shape as tapping a dual land for
    the wrong half, which is how this class of bug was first found."""
    def legal(state):
        if not state.active_mana_subdecision["can_produce"](state, color):
            return False
        return game.payment_survives(state, game.units_after(state, produced=[{color}]))
    legal._mana_subdecision_gate = "choose_color"
    return legal


def _mana_subdecision_color_execute(color):
    def execute(state):
        game.execute_mana_subdecision_color(state, color)
    return execute


def _choose_mana_color_legal(color):
    def legal(state):
        pending = state.pending_resolution
        return pending is not None and pending["kind"] == "choose_mana_color"
    legal._pending_gate = frozenset({"choose_mana_color"})
    return legal


def _choose_mana_color_execute(color):
    return lambda state: game.execute_choose_mana_color(state, color)


def _find_filter_source(state, name):
    """An unused mana filter named `name` -- Conduit Pylons (gated by tapped),
    Barrels of Blasting Jelly (gated by its own once-per-turn used_this_turn)."""
    for p in state.battlefield:
        if p.card_def.name != name or game.EFFECT_REGISTRY.get(p.card_def.effect_id, {}).get("filter_mana") is None:
            continue
        used = p.flags.get("used_this_turn", False) if name == "Barrels of Blasting Jelly" else p.tapped
        if not used:
            return p
    return None


_filter_source_cache = None  # (state, {name: Permanent or None}) -- one legal_action_mask sweep, same lifecycle as the combat-side battlefield-lookup cache in _actions_combat


def _cached_filter_source(state, name):
    """Memoizes _find_filter_source(state, name) per source name for one
    legal_action_mask sweep -- every (output_color, input_color) row for the
    same filter source (up to len(POOL_COLORS)**2 of them) would otherwise
    re-scan state.battlefield from scratch."""
    global _filter_source_cache
    if _filter_source_cache is None or _filter_source_cache[0] is not state:
        _filter_source_cache = (state, {})
    cache = _filter_source_cache[1]
    if name not in cache:
        cache[name] = _find_filter_source(state, name)
    return cache[name]


def _filter_mana_legal(name, input_color):
    """A mana filter's OWN activation cost -- "{1}[, {T}]: add one mana of
    any color", the {1} half -- paid immediately as a flat fixed-table
    action, one row per (source, input_color): which floating pip pays the
    {1}. The output half (which color comes out) is a separate, later
    choice via the shared choose_color mana_subdecision stage (see
    _filter_mana_execute) -- reusing Saruli Caretaker's own machinery,
    since "offer a small set of colors, then produce the chosen one" is
    identical between the two abilities; only how each PAYS to get there
    differs. Never a nested pay_cost for the {1} itself (which would risk
    clobbering whatever pending_resolution is already open --
    state.pending_resolution is a single slot, not a stack; same reasoning
    state.mana_subdecision's own docstring gives) -- no pending-resolution
    gate, per 605.1a. Legal iff the timing window allows a mana ability at
    all (_mana_timing_legal), an unused filter named `name` exists, the pool
    already holds a floating `input_color` pip (any color, including
    colorless, can pay a generic cost), and converting that pip away would
    not strand an open payment (_filter_would_strand_payment). Uses the
    sweep-scoped _cached_filter_source cache, not a fresh scan -- this
    legal() runs once per (source, input_color) row, POOL_COLORS-many per
    source now instead of POOL_COLORS x len(colors).

    A filter needs a floating pip to convert, so outside a payment it is
    reachable only in a main phase anyway -- the timing gate is what makes
    that a stated rule rather than an accident of when the pool is non-empty."""
    def legal(state):
        if not _mana_timing_legal(state) or _cached_filter_source(state, name) is None \
                or state.mana_pool.get(input_color, 0) <= 0:
            return False
        return not _filter_would_strand_payment(state, name, input_color)
    return legal


def _filter_would_strand_payment(state, name, input_color):
    """Would converting one `input_color` pip away right now make an ALREADY-
    BEGUN payment -- or one about to begin the instant the current choice
    resolves -- impossible to finish?

    The engine's standing guarantee is that a payment, once begun, can always be
    completed -- there is no "Abandon payment" action (it was deliberately
    removed, having existed to escape exactly this), so a payment the agent
    cannot finish is unescapable BY CONSTRUCTION: every cast/activate action is illegal while a
    pending is open, Pass is illegal, and the only remaining actions are
    spending pool mana and tapping sources. That guarantee was never enforced
    for filters, and a real pretrain run found the hole (monster_tron, turn 10:
    tap Forest for the only {G}, cast Crop Rotation for {G}, then filter that
    same {G} into {U} -- remaining {'G': 1}, no untapped green source, all-False
    action mask, RuntimeError).

    A SECOND, later hole in the same guarantee: choose_cast_copy (WHICH
    graveyard copy to cast/activate, MTG 601.2a) sits BETWEEN the moment a
    flashback/escape/graveyard-ability action is chosen (its own legality
    already required plan_payment to confirm the cost was payable) and the
    begin_pay_cost that actually opens once a copy is picked -- and mana
    abilities/filters stay legal "in ANY priority window" (605.1a/605.3b)
    during that gap too, same as mid-pay_cost. So filtering away the very pip
    the not-yet-open payment is guaranteed to need is exactly as breaking here
    as it is once pay_cost has already begun -- confirmed live (monster_tron,
    turn 44: 2 Bramble Wurms in the graveyard, activated the {2}{G} graveyard
    ability, filtered the floating {G} away while still choosing WHICH copy;
    pay_cost then opened already unpayable, all-False mask). Both windows are
    covered because game.outstanding_cost reports both.

    THREE things change, and the version of this check written for float-first
    only modelled one of them. It reasoned that a filter is one pip in, one pip
    out, so the pool's SIZE is invariant and only a COLORED requirement could
    break -- true of the pool, but the filter also TAPS ITS OWN SOURCE, and
    Conduit Pylons is both a filter and a plain {C} mana source. Under
    cast-then-pay an untapped Pylons is counted toward affordability, so
    filtering with it silently deletes a unit the cast was allowed on:
    pool {U}, owing {2}, Pylons the only other source -> plan_payment says
    payable (2 units), filter it and only 1 unit remains. The old check looked
    at `remaining["U"]`, saw 0, and allowed exactly that.

    So ask the real question instead of a proxy for it: rebuild the unit list as
    it would be after the conversion -- input pip spent, filter source tapped,
    one pip of the filter's own output colors added (which color is the agent's
    later choice, gated in turn by _mana_subdecision_color_legal) -- and check
    whether the payment still survives. Shorter than the reasoning it replaces,
    exact in both directions, and it covers generic requirements too."""
    source = _cached_filter_source(state, name)
    if source is None:
        return False
    output_colors = game.EFFECT_REGISTRY[source.card_def.effect_id]["filter_mana"]["colors"]
    return not game.payment_survives(state, game.units_after(
        state, tapped=[source], spent=[input_color], produced=[output_colors]))


def _filter_mana_execute(name, input_color):
    """Pays the {1} immediately (taps/flags the resolved source, spends the
    chosen input pip -- exactly what the pre-split atomic execute did for
    this half, unchanged), then opens the SHARED choose_color
    mana_subdecision stage (game.begin_mana_color_choice) for the output
    half -- can_produce/on_choose_color read the resolved source's own
    "filter_mana" spec fresh (not a closed-over table-build-time value),
    same reasoning _mana_extra_choose_execute's own docstring gives for a
    token-sourced name."""
    def execute(state):
        p = _find_filter_source(state, name)
        if name == "Barrels of Blasting Jelly":
            p.flags["used_this_turn"] = True
        else:
            p.tapped = True
        game.spend_one_pip(state, input_color)
        state.log_event("mana_spend", color=input_color, toward="filter")

        colors = game.EFFECT_REGISTRY[p.card_def.effect_id]["filter_mana"]["colors"]

        def can_produce(state, color):
            return color in colors

        def on_choose_color(state, color):
            # taggable=False: a filter's output is a deliberate pool->pool
            # conversion, never reflexive tapping -- must not be tagged as
            # single-pip production even though it's exactly 1 symbol. See
            # PlayerState.mana_pool_single_pip's own docstring.
            #
            # KNOWN, OWNER-APPROVED GAP (2026-08): because game.spend_one_pip
            # (called on input_color above) drains an UNTAGGED pip of a color
            # before a TAGGED one, a filter can "launder" an already-TAGGED
            # (avoidable) pip into an untagged one -- pay the {1} with a
            # tagged pip (only one floating of that color), then choose the
            # SAME output color here -- for zero pool-size cost, since
            # nothing stops choosing the input color as the output color.
            # That pip's mana-burn tag is erased even though it traces back
            # to an ordinary, avoidable tap. Deliberately left as designed:
            # filters are once-per-turn per source, so the exploitable
            # surface is small, and reward-shaping input here doesn't have
            # to be airtight, only directionally correct. Revisit only if
            # training data shows filter-heavy decks (monster_tron) training
            # differently than their non-filter peers.
            game.float_mana(state, [color], taggable=False)
            state.log_event("mana_tap", permanent=(name, p.slot), mode="filter", produced=[color])

        game.begin_mana_color_choice(state, can_produce, on_choose_color)
    return execute


__all__ = [
    '_find_mana_extra_source',
    '_mana_extra_choose_legal',
    '_mana_extra_choose_execute',
    '_mana_subdecision_color_legal',
    '_mana_subdecision_color_execute',
    '_choose_mana_color_legal',
    '_choose_mana_color_execute',
]
