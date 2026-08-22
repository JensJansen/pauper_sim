"""Mana production: native/flexible tap abilities, Saruli-Caretaker-shaped
extra-cost ("tap another creature") sources, mana filters (Conduit Pylons/
Barrels of Blasting Jelly), and Chromatic Star's choose_mana_color.

None of these carry a _pending_gate (605.1a/605.3b: a mana ability never
uses the stack or requires priority), except the shared choose_color
mana_subdecision buttons and choose_mana_color, which are pending-kind-
gated. That is what makes CR 601.2f work: a mana ability stays legal
during an open payment, the normal way mana gets produced under
cast-then-pay.

Two rules cut across every entry point here:
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
    sweep, since every "Tap X for <color>" row's legal() calls it."""
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

    Neither window is open mid-cast, before 601.2f (game.mid_cast). That case
    has to be refused ahead of the main-phase test rather than folded into it,
    because a cast announced in a main phase is still in a main phase while its
    modes/X/delve/copy are being chosen."""
    if game.mid_cast(state):
        return False
    return (
        game.payment_in_progress(state)
        or (state.phase in game.turn.SORCERY_SPEED_PHASES and state.active_idx == state.turn_player_idx)
    )


def _mana_ability_legal(name, color):
    """Legal iff a source named `name` can produce `color` right now
    (game.mana_ability_options; color=None for fixed/tron/count sources),
    the timing window allows it (_mana_timing_legal), and tapping it leaves
    an open payment still finishable (game.payment_survives).

    No `_pending_gate` attribute, so legal_action_mask always calls this
    regardless of what is pending, which is what makes a mana ability
    legal mid-payment (CR 601.2f)."""
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
    """The untapped, available permanent named `name` that can produce
    `color` now (`color` narrows to the flexible/granted source that can
    make it, e.g. an Abundant Growth land). Mirrors mana_ability_options'
    per-permanent gates (tap-lock, extra cost)."""
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
    """Memoizes _find_mana_source per (name, color) for one legal_action_mask
    sweep, since every row sharing a (name, color) pair would otherwise
    re-scan state.battlefield from scratch."""
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
    """Same as _find_mana_source, minus the color-producibility check: for
    a mana_extra_choose source (Saruli Caretaker), the color isn't chosen
    until the second stage of its mana_subdecision, so there's no color to
    check yet. See _mana_subdecision_color_legal for where that check
    lives instead."""
    for p in state.battlefield:
        if p.card_def.name != name or p.tapped or game.tap_summoning_locked(state, p):
            continue
        extra = game.EFFECT_REGISTRY.get(p.card_def.effect_id, {}).get("mana_extra_available")
        if extra is not None and not extra(state, p):
            continue
        return p
    return None


def _mana_extra_choose_legal(name):
    """Saruli-Caretaker-shaped mana ability whose additional cost is
    tapping another untapped creature (registry "mana_extra_choose") -- a
    cost choice (602.5g), decided as the first stage of a
    mana_subdecision. No pending-resolution gate: a mana ability is legal
    in any priority window (605.1a/605.3b). Legal iff a source of this
    name is untapped/available and at least one other creature satisfies
    the source's mana_extra_choose predicate. Timing follows the shared
    _mana_timing_legal rule.

    Mid-payment a candidate only counts if tapping it specifically leaves
    the payment finishable (mana_extra_choose_target_safe) -- checked here,
    not only at the target-choice step, because that step takes exclusive
    priority and offers no way back if it finds zero safe targets. One
    safe candidate is enough to activate."""
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
    """Would tapping `target` as a mana_extra_choose source's additional
    cost (e.g. Saruli Caretaker's "tap another untapped creature") leave an
    open payment still finishable? `target` loses whatever it could still
    have produced, and one pip of any color arrives once the color stage
    resolves -- a swap, not a gain, which is a real loss if `target` is
    itself a mana source (e.g. Overgrown Battlement). Shared with
    rl.decision.action_bridge, which masks the target choice with this same
    predicate."""
    return game.payment_survives(state, game.units_after(
        state, tapped=[target], produced=[game.COLORS]))


def _mana_extra_choose_execute(name):
    """Resolves the specific source permanent now (not deferred): the
    target-choice step must identity-exclude this exact copy, not just any
    creature named `name` (two Saruli Caretakers: tapping the other is
    legal, tapping itself is not). extra_pred is looked up off the
    resolved permanent's card_def.effect_id, not game.CARD_DEFS[name],
    since a token-sourced name has no CARD_DEFS entry."""
    def execute(state):
        p = _find_mana_extra_source(state, name)
        extra_pred = game.EFFECT_REGISTRY[p.card_def.effect_id]["mana_extra_choose"]
        game.begin_mana_subdecision(state, p, extra_pred)
    return execute


def _mana_subdecision_color_legal(color):
    """Shared "Produce <color>" button, reused by every gate-free mana
    ability with a final choose-a-color step (Saruli Caretaker;
    filter_mana cards). Legal only mid the choose_color stage of a
    mana_subdecision, and only if the ability that opened it says this
    color is offerable (state.mana_subdecision["can_produce"]).

    Mid-payment it must additionally not strand that payment: everything
    upstream reasoned optimistically about a color not yet chosen,
    crediting one pip of any offered color, so choosing a color the
    remaining cost cannot use is refused here rather than allowed to
    strand."""
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
    """Memoizes _find_filter_source per source name for one
    legal_action_mask sweep, since every (output_color, input_color) row
    for the same source would otherwise re-scan state.battlefield."""
    global _filter_source_cache
    if _filter_source_cache is None or _filter_source_cache[0] is not state:
        _filter_source_cache = (state, {})
    cache = _filter_source_cache[1]
    if name not in cache:
        cache[name] = _find_filter_source(state, name)
    return cache[name]


def _filter_mana_legal(name, input_color):
    """A mana filter's own activation cost -- "{1}[, {T}]: add one mana of
    any color", the {1} half -- paid immediately, one row per (source,
    input_color). The output half is a separate later choice via the
    shared choose_color mana_subdecision stage (_filter_mana_execute),
    reusing Saruli Caretaker's own machinery. No pending-resolution gate
    (605.1a); never a nested pay_cost for the {1}, since
    state.pending_resolution is a single slot, not a stack. Legal iff the
    timing window allows a mana ability (_mana_timing_legal), an unused
    filter named `name` exists, the pool holds a floating `input_color`
    pip, and converting it away would not strand an open payment
    (_filter_would_strand_payment)."""
    def legal(state):
        if not _mana_timing_legal(state) or _cached_filter_source(state, name) is None \
                or state.mana_pool.get(input_color, 0) <= 0:
            return False
        return not _filter_would_strand_payment(state, name, input_color)
    return legal


def _filter_would_strand_payment(state, name, input_color):
    """Would converting one `input_color` pip away right now make an
    already-begun payment (or one about to begin) impossible to finish?

    A filter changes three things -- the input pip is spent, the source is
    tapped, and one pip of an output color arrives -- so this rebuilds the
    unit list as it would be afterward and asks game.payment_survives,
    rather than reasoning about pool size alone. The source-tapping part
    matters because a filter source (e.g. Conduit Pylons) can also be a
    plain mana source counted toward affordability: filtering with it can
    delete a unit the cast was relying on even though the pool's total size
    is unchanged."""
    source = _cached_filter_source(state, name)
    if source is None:
        return False
    output_colors = game.EFFECT_REGISTRY[source.card_def.effect_id]["filter_mana"]["colors"]
    return not game.payment_survives(state, game.units_after(
        state, tapped=[source], spent=[input_color], produced=[output_colors]))


def _filter_mana_execute(name, input_color):
    """Pays the {1} immediately (taps/flags the resolved source, spends the
    chosen input pip), then opens the shared choose_color mana_subdecision
    stage (game.begin_mana_color_choice) for the output half. can_produce/
    on_choose_color read the resolved source's "filter_mana" spec fresh,
    not a closed-over table-build-time value."""
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
