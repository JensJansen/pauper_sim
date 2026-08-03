"""The token/attention DRL action layer: per-action legal/execute/options
factories, build_action_table (assembles the flat fixed action table), and
legal_action_mask (+ its sweep-scoped caches). Split out of drl_env/__init__
unchanged; re-exported there so `drl_env.X` keeps working."""

import numpy as np

import game


# ---------------------------------------------------------------------------
# Action table: generated from a decklist + game.EFFECT_REGISTRY instead of
# hand-typed -- this, plus the pending-resolution machinery in the game
# engine, is what makes a deck built entirely from already-implemented cards
# need zero new code here.
#
# Categories, in table order:
#   A. Play land: <name>            -- one per distinct land name
#   A2. Tap <name> [for <color>]    -- float-first mana abilities: one no-color
#      row per fixed/Tron/count source, one row per producible color for a
#      flexible/granted source. Legal in ANY priority window, even
#      mid-resolution of anything else (605.1a/605.3b -- no pending-resolution
#      gate at all), resolves immediately (never the stack) -- masked by
#      game.mana_ability_options. "Tap <name>" (Saruli Caretaker): same
#      gate-free category, but its extra cost (tap ANOTHER creature -- a cost
#      choice, 602.5g, not a target) opens a mana_subdecision (a SEPARATE
#      state field from pending_resolution, so this can still open mid-
#      resolution of anything else without clobbering it) instead of
#      resolving in one shot -- see state.mana_subdecision's own docstring
#      and _mana_extra_choose_legal's own docstring for why.
#      "Filter <name>, paying <input_color>" (Conduit Pylons/Barrels): pays
#      the {1} activation cost immediately as a flat fixed-table row, then
#      opens the shared choose_color mana_subdecision stage (see "Tap
#      <name>"/Saruli's own note above and _filter_mana_execute's own
#      docstring) to pick the output color via the "Produce <color>"
#      buttons -- no nested pay_cost either, same reasoning.
#   B. Cast <name>                  -- one per card with a registry "cast" entry
#   C. Activate <name> (<ability>)  -- one per registered activated ability
#   D. Forestcycle <name>           -- one per registry "forestcycle" entry
#   E. Pass
#   F. Choose: <name>               -- shared across every pending-resolution
#      kind that picks a plain card name (search_fetch, ancient_stirrings,
#      discard, and scry/surveil's ordering phase), dispatched by
#      pending_resolution["kind"]. Paying a cost never appears here -- once a
#      cost is announced, the only legal actions are "Spend <color> from
#      pool" (below), spending mana already floated via a Tap action.
#      NOT sacrifice (see category K) -- a battlefield permanent is never
#      just a name, unlike a hand/library/graveyard card.
#   H. Keep / Dispose (scry/surveil)
#   I. Decline (Ancient Stirrings)
#   K. Choose target: <name> (slot k) -- exact-(name, slot)-addressed, the
#      "choose_permanent" resolution's own actions (Aura enchant-targets,
#      Crop Rotation's sacrifice cost, land bounce, and now every generic
#      sacrifice cost -- begin_sacrifice/Highway Robbery's own sacrifice
#      trigger, both chained through this same primitive) -- NOT category F:
#      two same-named permanents stop being interchangeable the instant an
#      Aura attaches to only one of them (or one has a counter, is tapped,
#      ...), and cast_aura's cast-time-target/resolve-time-fizzle contract
#      (same as knowing exactly which physical permanent was sacrificed)
#      depends on knowing exactly which physical permanent was chosen.
#
# spy_combo deck additions: B also covers Winding Way's modal cast (2
# actions, one per mode), Land Grant's free alt-cost, Dread Return's
# Flashback (cast from the graveyard), and Nyxborn Hydra's own
# "x_cast_modes" (one action per (mode, X) pair -- its normal creature cast
# and Bestow, each its own cost distinct from card_def.cast_cost, see
# _x_cast_legal/_x_cast_execute/_x_precast_choice_execute); C also covers
# non-mana activated abilities (Quirion Ranger, Pinnacle Kill-Ship's own
# Station); F/H also cover select_to_hand's own Keep/Bottom pair and its
# ordering phase (Lead the Stampede) and an optional search's Decline
# (Gatecreeper Vine) alongside Ancient Stirrings'; K also covers Pinnacle
# Kill-Ship's own opponent-facing ETB target (choose_opponent_permanent,
# same category as blocking's own cross-player targeting -- correctly a
# no-op in every current 1-player Tron config, since the underlying
# resolution auto-completes with no legal target).
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


_GATE_NO_PENDING = object()  # sentinel: this closure's own first check is "state.pending_resolution is None" -- see legal_action_mask's own docstring

# Shared button-count caps for the generic choose_cast_mode/choose_cast_x/
# choose_delve_amount sub-decisions (build_action_table's own cast_modes/
# x_cast_modes/delve loops below) -- ponytail: sized to the current catalog's
# own max (Utopia Sprawl's 5 modes, Nyxborn Hydra's X=0..10, Gurmag Angler's
# delve 0..6), not a speculative ceiling. plan_payment/index-bounds masking
# already keeps a card with a smaller max from ever offering more than its
# own range, so raising these later for a bigger future card is the only
# thing a new card can ever require here -- see each button's own legal()
# below for the mask that makes over-provisioning safe.
_CAST_MODE_BUTTON_MAX = 5
_CAST_X_BUTTON_MAX = 10
_DELVE_BUTTON_MAX = 6


def _land_drop_legal(name):
    def legal(state):
        return (
            state.pending_resolution is None
            # Real Magic: playing a land is always sorcery-speed (no
            # per-card override exists in this cube) -- speed_legal's own
            # Speed.SORCERY branch already requires state.active_idx ==
            # state.turn_player_idx, so this alone
            # already refuses a land drop offered to the non-turn player
            # during a priority window, with no separate check needed here.
            and game.turn.speed_legal(state, game.turn.Speed.SORCERY)
            and state.lands_played_this_turn == 0
            and any(c.name == name for c in state.hand)
        )
    legal._pending_gate = _GATE_NO_PENDING
    return legal


def _land_drop_execute(name):
    def execute(state):
        game.play_land_from_hand(state, game.CARD_DEFS[name])
    return execute


def _hand_count_available(state, name):
    """How many copies of `name` in state.hand are castable right now -- just
    the count in hand. A spell LEAVES hand the instant it is put on the stack
    (game.effects.stack.push_to_stack removes it at cast), so a copy already
    paid-for and awaiting resolution is simply not in state.hand and can't be
    re-cast -- the card physically leaving hand IS the re-cast guard, so a
    plain hand tally is all this needs."""
    return sum(1 for c in state.hand if c.name == name)


def _effective_cast_cost(state, card_def):
    """A card's cast cost after any registry "cost_reduction" -- a
    lambda(state) -> int (affinity = # artifacts you control; the graveyard
    instant/sorcery count for Tolarian Terror / Cryptic Serpent; cards drawn
    this turn for Deem Inferior). The reduction lowers ONLY the generic pips,
    floored at 0 -- colored pips are never reduced (real "costs {N} less").
    A card with no such spec (every existing card, and every G1-G6 card) pays
    card_def.cast_cost unchanged, so this is a transparent no-op for them --
    the single reason it's safe to route the whole cast path through it."""
    cost = card_def.cast_cost
    spec = game.EFFECT_REGISTRY.get(card_def.effect_id, {}).get("cost_reduction")
    if spec is None or cost is None:
        return cost
    reduction = spec(state)
    if reduction <= 0:
        return cost
    reduced = dict(cost)
    reduced["generic"] = max(0, reduced.get("generic", 0) - reduction)
    return reduced


def _cast_legal(name, extra_legal, speed):
    def legal(state):
        if state.pending_resolution is not None:
            return False
        if not game.turn.speed_legal(state, speed):
            return False
        if _hand_count_available(state, name) <= 0:
            return False
        card_def = game.CARD_DEFS[name]
        if game.plan_payment(state, _effective_cast_cost(state, card_def)) is None:
            return False
        return extra_legal is None or extra_legal(state)
    legal._pending_gate = _GATE_NO_PENDING
    return legal


def _cast_execute(name, resolve):
    def execute(state):
        card_def = game.CARD_DEFS[name]
        def _after_pay(s):
            # Fires only once mana is actually paid -- NOT the instant this
            # cast is announced. Real MTG (601.2i): a spell isn't "cast" until
            # its cost is paid, so a "whenever you cast" trigger (e.g.
            # Guttersnipe) must never fire before that. Every cast path (this one,
            # alt_cast, flashback, plot-from-exile below) fires it
            # identically once its own cost is similarly locked in.
            game.on_cast_trigger(s, card_def)
            # Once mana is fully paid, the spell is "cast" but not yet
            # resolved -- push it onto state.stack (game.push_to_stack)
            # instead of resolving immediately, so the model can respond
            # with another instant-speed action first. Something (a
            # "Pass" -- see game.turn._run_turn_gen) has to actually
            # resolve it later.
            game.push_to_stack(s, card_def, resolve)
        game.begin_pay_cost(state, _effective_cast_cost(state, card_def), on_complete=_after_pay)
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
        def _after_pay(s):
            game.on_cast_trigger(s, card_def)  # only once mana is irreversibly paid -- see _cast_execute's own comment
            resolve(s, card_def)
        game.begin_pay_cost(state, _effective_cast_cost(state, card_def), on_complete=_after_pay)
    return execute


def _x_cast_legal(name, cost, extra_legal, speed):
    """Like _cast_legal, but against an explicit `cost` instead of
    card_def.cast_cost -- one X value's own concrete cost (an
    "x_cast_modes" registry entry's own per-mode base cost plus that X's
    own generic, build_action_table's own loop below), same "a real cost
    distinct from card_def.cast_cost" shape _plot_legal/_omen_cast_legal
    already use for their own alternate costs, not a param bolted onto
    _cast_legal itself."""
    def legal(state):
        if state.pending_resolution is not None:
            return False
        if not game.turn.speed_legal(state, speed):
            return False
        if _hand_count_available(state, name) <= 0:
            return False
        if game.plan_payment(state, cost) is None:
            return False
        return extra_legal is None or extra_legal(state)
    legal._pending_gate = _GATE_NO_PENDING
    return legal


def _x_cast_execute(name, cost, resolve):
    """Same shape as _cast_execute, against an explicit `cost`."""
    def execute(state):
        card_def = game.CARD_DEFS[name]
        def _after_pay(s):
            game.on_cast_trigger(s, card_def)
            game.push_to_stack(s, card_def, resolve)
        game.begin_pay_cost(state, cost, on_complete=_after_pay)
    return execute


def _x_precast_choice_execute(name, cost, resolve):
    """Same shape as _precast_choice_execute, against an explicit `cost` --
    Nyxborn Hydra's own Bestow mode needs both: a real target chosen before
    the stack (cast_aura, same as Rancor/Ancestral Mask/Ethereal Armor) AND
    its own X-dependent cost distinct from the card's normal cast_cost."""
    def execute(state):
        card_def = game.CARD_DEFS[name]
        def _after_pay(s):
            game.on_cast_trigger(s, card_def)
            resolve(s, card_def)
        game.begin_pay_cost(state, cost, on_complete=_after_pay)
    return execute


def _delve_reduced_cost(card_def, n):
    """Delve pays {1} of the generic cost per graveyard card exiled -- reduce
    the generic pips by n, floored at 0 (colored pips untouched)."""
    cost = dict(card_def.cast_cost)
    cost["generic"] = max(0, cost.get("generic", 0) - n)
    return cost


def _delve_legal(name, max_n, speed):
    """Cast <name> is legal iff at least one delve amount 0..max_n is
    currently affordable given the current graveyard size -- aggregated
    across the range (one "Cast <name>" row, not one per n); the shared
    "Delve 0".."Delve 6" buttons (choose_delve_amount below) mask the
    unaffordable amounts next, same plan_payment check just re-encoded."""
    def legal(state):
        if state.pending_resolution is not None:
            return False
        if not game.turn.speed_legal(state, speed):
            return False
        if _hand_count_available(state, name) <= 0:
            return False
        card_def = game.CARD_DEFS[name]
        for n in range(min(max_n, len(state.graveyard)) + 1):
            if game.plan_payment(state, _delve_reduced_cost(card_def, n)) is not None:
                return True
        return False
    legal._pending_gate = _GATE_NO_PENDING
    return legal


def _delve_execute(name, max_n, resolve):
    """601.2f/702.66: the delve amount is chosen before the exile sub-cost
    opens and the reduced cost is calculated -- begin_choose_delve_amount
    (shared "Delve 0".."Delve 6" buttons) first, THEN exile that many
    graveyard cards (the model chooses which -- begin_exile_n_from_
    graveyard, unchanged), THEN pay the {generic-n} remainder, then cast
    normally. The exile is a cost, paid first and irreversible; the mana
    payment that follows is a pure pool spend (float-first: no undo, see the
    "no Abandon payment" note in build_action_table below)."""
    def execute(state):
        card_def = game.CARD_DEFS[name]

        def _after_n(state, n):
            def _after_exile(s):
                def _after_pay(s2):
                    game.on_cast_trigger(s2, card_def)
                    game.push_to_stack(s2, card_def, resolve)
                game.begin_pay_cost(s, _delve_reduced_cost(card_def, n), on_complete=_after_pay)
            game.begin_exile_n_from_graveyard(state, n, _after_exile)

        game.begin_choose_delve_amount(state, card_def, max_n, _after_n)
    return execute


def _choose_delve_amount_legal(n):
    """Shared "Delve n" button (0..n..max_n), reused by every delve card --
    legal only mid a choose_delve_amount resolution, within this specific
    card's own max_n, with enough graveyard cards, and only if the resulting
    reduced cost is actually affordable (the exact per-value masking a flat
    per-row enumeration already did, just re-encoded)."""
    def legal(state):
        pending = state.pending_resolution
        if pending is None or pending["kind"] != "choose_delve_amount":
            return False
        if n > pending["max_n"]:
            return False
        if len(state.graveyard) < n:
            return False
        return game.plan_payment(state, _delve_reduced_cost(pending["card_def"], n)) is not None
    legal._pending_gate = frozenset({"choose_delve_amount"})
    return legal


def _choose_delve_amount_execute(n):
    def execute(state):
        game.execute_choose_delve_amount_option(state, n)
    return execute


def _modal_legal(name, mode_items, speed):
    """Cast <name> is legal iff at least one of its modes is currently
    produceable -- extra_legal (if any) passes AND its own cost (mode
    override, else the card's cast_cost via _effective_cast_cost) is
    affordable. One "Cast <name>" row, not one per mode; the aggregate
    check mirrors today's per-row plan_payment masking, now aggregated
    across modes instead of enumerated per row."""
    def legal(state):
        if state.pending_resolution is not None:
            return False
        if not game.turn.speed_legal(state, speed):
            return False
        if _hand_count_available(state, name) <= 0:
            return False
        card_def = game.CARD_DEFS[name]
        for _mode_name, mode_spec in mode_items:
            extra_legal = mode_spec.get("extra_legal")
            if extra_legal is not None and not extra_legal(state):
                continue
            cost = mode_spec.get("cost")
            if cost is None:
                cost = _effective_cast_cost(state, card_def)
            if game.plan_payment(state, cost) is not None:
                return True
        return False
    legal._pending_gate = _GATE_NO_PENDING
    return legal


def _modal_execute(name, mode_items):
    """601.2b: mode chosen before the cost is calculated -- opens the
    generic choose_cast_mode resolution (shared "Mode 1".."Mode 5" buttons),
    then -- once a mode is picked -- computes THAT mode's own cost and
    proceeds exactly as a plain/cost-overridden cast would (begin_pay_cost
    -> push_to_stack, or the precast_choice variant if this mode needs
    one -- e.g. Utopia Sprawl's own "enchant target Forest")."""
    def execute(state):
        card_def = game.CARD_DEFS[name]

        def _afford_check(mode_spec):
            cost = mode_spec.get("cost")
            def check(state, cost=cost, card_def=card_def):
                c = cost if cost is not None else _effective_cast_cost(state, card_def)
                return game.plan_payment(state, c) is not None
            return check

        modes_for_gate = tuple((spec.get("extra_legal"), _afford_check(spec)) for _n, spec in mode_items)

        def _after_mode(state, mode_index):
            _mode_name, mode_spec = mode_items[mode_index]
            cost = mode_spec.get("cost")
            if cost is None:
                cost = _effective_cast_cost(state, card_def)
            resolve = mode_spec["resolve"]

            def _after_pay(s):
                game.on_cast_trigger(s, card_def)
                if mode_spec.get("precast_choice"):
                    resolve(s, card_def)
                else:
                    game.push_to_stack(s, card_def, resolve)
            game.begin_pay_cost(state, cost, on_complete=_after_pay)

        game.begin_choose_cast_mode(state, card_def, modes_for_gate, _after_mode)
    return execute


def _x_modal_legal(name, mode_items, speed):
    """Cast <name> is legal iff at least one (mode, X) pair is currently
    affordable, X=0 upward per mode's own base cost -- aggregated across
    modes AND the X range, mirroring today's per-(mode,X)-row masking, now
    behind one "Cast <name>" row."""
    def legal(state):
        if state.pending_resolution is not None:
            return False
        if not game.turn.speed_legal(state, speed):
            return False
        if _hand_count_available(state, name) <= 0:
            return False
        for _mode_name, mode_spec in mode_items:
            extra_legal = mode_spec.get("extra_legal")
            if extra_legal is not None and not extra_legal(state):
                continue
            base_cost = mode_spec["cost"]
            for x in range(mode_spec["max_x"] + 1):
                cost = dict(base_cost)
                cost["generic"] = cost.get("generic", 0) + x
                if game.plan_payment(state, cost) is not None:
                    return True
        return False
    legal._pending_gate = _GATE_NO_PENDING
    return legal


def _x_modal_execute(name, mode_items):
    """601.2b then 601.2f: mode chosen first, X chosen second, both before
    the total cost is calculated/paid -- matches real sequencing (mode is
    part of announcing the spell; X is determined as part of costing it,
    strictly afterward). Nyxborn Hydra's Bestow mode still needs a real Aura
    target (precast_choice), settled after payment, same as a plain cast --
    make_resolve(x) (green_cards.cast_nyxborn_hydra_creature/_bestow) is
    called once X is known, at execute time rather than table-build time."""
    def execute(state):
        card_def = game.CARD_DEFS[name]

        def _x_afford_check(mode_spec):
            base_cost, max_x = mode_spec["cost"], mode_spec["max_x"]
            def check(state, base_cost=base_cost, max_x=max_x):
                for x in range(max_x + 1):
                    cost = dict(base_cost)
                    cost["generic"] = cost.get("generic", 0) + x
                    if game.plan_payment(state, cost) is not None:
                        return True
                return False
            return check

        modes_for_gate = tuple((spec.get("extra_legal"), _x_afford_check(spec)) for _n, spec in mode_items)

        def _after_mode(state, mode_index):
            _mode_name, mode_spec = mode_items[mode_index]
            base_cost, max_x = mode_spec["cost"], mode_spec["max_x"]
            make_resolve = mode_spec["resolve"]

            def _after_x(state, x):
                cost = dict(base_cost)
                cost["generic"] = cost.get("generic", 0) + x
                resolve = make_resolve(x)

                def _after_pay(s):
                    game.on_cast_trigger(s, card_def)
                    if mode_spec.get("precast_choice"):
                        resolve(s, card_def)
                    else:
                        game.push_to_stack(s, card_def, resolve)
                game.begin_pay_cost(state, cost, on_complete=_after_pay)

            game.begin_choose_cast_x(state, base_cost, max_x, _after_x)

        game.begin_choose_cast_mode(state, card_def, modes_for_gate, _after_mode)
    return execute


def _choose_cast_mode_legal(mode_index):
    """Shared "Mode n" button (1..5, drl_env's own numbering -- 0-based
    internally), reused by every cast_modes/x_cast_modes card -- legal only
    mid a choose_cast_mode resolution, within THIS card's own mode count,
    and only if that specific mode's own extra_legal/affordability check
    (closed over per-mode at "Cast <name>" execute time) currently passes."""
    def legal(state):
        pending = state.pending_resolution
        if pending is None or pending["kind"] != "choose_cast_mode":
            return False
        modes = pending["modes"]
        if mode_index >= len(modes):
            return False
        extra_legal, afford_check = modes[mode_index]
        if extra_legal is not None and not extra_legal(state):
            return False
        return afford_check(state)
    legal._pending_gate = frozenset({"choose_cast_mode"})
    return legal


def _choose_cast_mode_execute(mode_index):
    def execute(state):
        game.execute_choose_cast_mode_option(state, mode_index)
    return execute


def _choose_cast_x_legal(x):
    """Shared "X=n" button (0..10), reused by every x_cast_modes card --
    legal only mid a choose_cast_x resolution, within this mode's own
    max_x, and only if base_cost+x's generic is actually affordable (the
    exact per-value masking a flat per-row enumeration already did, just
    re-encoded)."""
    def legal(state):
        pending = state.pending_resolution
        if pending is None or pending["kind"] != "choose_cast_x":
            return False
        if x > pending["max_x"]:
            return False
        cost = dict(pending["base_cost"])
        cost["generic"] = cost.get("generic", 0) + x
        return game.plan_payment(state, cost) is not None
    legal._pending_gate = frozenset({"choose_cast_x"})
    return legal


def _choose_cast_x_execute(x):
    def execute(state):
        game.execute_choose_cast_x_option(state, x)
    return execute


def _tuck_position_legal(state):
    pending = state.pending_resolution
    return pending is not None and pending["kind"] == "tuck_position"


_tuck_position_legal._pending_gate = frozenset({"tuck_position"})


def _activate_legal(name, cost_key, speed, extra_legal=None):
    """extra_legal (optional): an additional state-only predicate beyond
    "an untapped copy exists and its cost_key cost is payable" -- for an
    ability whose OWN targets/effect impose a further precondition beyond
    cost (Barrels of Blasting Jelly's "{5}, T, Sacrifice: 5 damage to
    TARGET creature" can't be activated with zero legal creature targets on
    board, 602.2b/601.2c), same role "extra_legal" plays on a "cast" spec."""
    def legal(state):
        if state.pending_resolution is not None:
            return False
        if not game.turn.speed_legal(state, speed):
            return False
        p = next((p for p in state.battlefield if p.card_def.name == name and not p.tapped), None)
        if p is None or game.plan_payment(state, p.card_def.extra[cost_key]) is None:
            return False
        return extra_legal is None or extra_legal(state)
    legal._pending_gate = _GATE_NO_PENDING
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
    legal._pending_gate = _GATE_NO_PENDING
    return legal


def _forestcycle_execute(name, cost_key, resolve):
    def execute(state):
        card_def = game.CARD_DEFS[name]
        game.begin_pay_cost(state, card_def.extra[cost_key], on_complete=lambda s: resolve(s, card_def))
    return execute


def _graveyard_ability_legal(name, cost_key):
    """Bramble Wurm's own "{2}{G}, Exile this card from your graveyard:
    gain 5 life" -- same hand-zone/cost-key/card_def shape as
    _forestcycle_legal above, just sourced from state.graveyard instead of
    state.hand (a graveyard activated ability, unlike Flashback, never
    recasts the spell -- resolve just runs the ability directly, no
    push_to_stack, matching every other activated ability in this engine).
    No speed gate: every existing activated ability defaults to "any
    time" absent an explicit override (see build_action_table's own
    activated_abilities loop), and this one has none."""
    def legal(state):
        if state.pending_resolution is not None:
            return False
        if not any(c.name == name for c in state.graveyard):
            return False
        card_def = game.CARD_DEFS[name]
        return game.plan_payment(state, card_def.extra[cost_key]) is not None
    legal._pending_gate = _GATE_NO_PENDING
    return legal


def _graveyard_ability_execute(name, cost_key, resolve):
    """resolve receives the graveyard CardInstance whose ability this is (NOT
    the interned CardDef) -- see _graveyard_instance. The cost itself is read
    off game.CARD_DEFS[name]: a cost is a type-level rules property, identical
    for every copy, so the canonical def is the right source for it."""
    def execute(state):
        cost = game.CARD_DEFS[name].extra[cost_key]

        def _proceed(state, inst):
            # Copy chosen first, then the cost is paid (602.2/601.2a order), then
            # the ability's own resolve exiles that exact instance as its cost.
            game.begin_pay_cost(state, cost, on_complete=lambda s, inst=inst: resolve(s, inst))

        _with_chosen_copy(state, name, _proceed, reserved_cost=cost)
    return execute


def _pass_legal(state):
    if state.pending_resolution is not None:
        return False
    # Goad (Undercity Arena): a goaded creature that CAN attack must be
    # declared -- its controller may not end their own declare-attackers step
    # (Pass) while one is still able and undeclared. game.has_unfulfilled_goad
    # only ever returns True during DECLARE_ATTACKERS for the turn player, so
    # this is a no-op everywhere else.
    if state.phase is game.turn.Phase.DECLARE_ATTACKERS and game.has_unfulfilled_goad(state):
        return False
    return True


_pass_legal._pending_gate = _GATE_NO_PENDING


def _pass_execute(state):
    pass  # no-op: a Pass is signalled by choose_action returning None (the game loop then advances), never by invoking this execute fn


# The complete set of pending kinds _choose_name_options ever dispatches --
# the SOLE authoritative list (both _choose_name_legal's gate and
# legal_action_mask's own coverage guard below read this SAME constant, so
# there is exactly one place that can ever drift out of sync with the
# dispatch table above/below it).
#
# STRUCTURAL INVARIANT, not a convention to remember: every kind in this set
# MUST have its candidate names confined to the DECIDING PLAYER'S OWN cards
# (their hand/library/battlefield/graveyard/trigger_queue -- all already
# active-player-proxied). A "Choose: X" row only ever exists for a name in
# the ASKING player's own deck (build_action_table's choosable_names, built
# once per deck) -- so a kind whose candidates could ever include another
# player's card CANNOT be represented here, no matter how many rows exist.
# choose_graveyard_card and choose_stack_target both learned this the hard
# way (Relic of Progenitus/Mesmeric Fiend reach the opponent's graveyard/
# hand; a spell to counter is very often the opponent's) and were migrated
# to POINTER addressing instead (rl.action_bridge) -- pointer scoring reads
# live token identity, never a per-deck name table, so it needs no such
# guarantee. A future kind belongs in this set ONLY if it can never read
# anything but the acting player's own zones; otherwise it must be a pointer
# target. This is not merely documented: legal_action_mask's own coverage
# check below FAILS LOUDLY, the first time any game ever exercises it, if a
# kind in this set ever produces a candidate this constant's own promise
# doesn't cover -- so a future violation cannot ship unnoticed the way this
# one did.
_CHOOSE_NAME_PENDING_KINDS = frozenset({
    "search_fetch", "throne_reveal", "discard",
    "discard_or_sacrifice", "ancient_stirrings", "malevolent_rumble", "scry", "surveil",
    "select_to_hand", "order_triggers", "put_on_top", "ponder",
})


def _choose_name_options(state):
    """Plain (uncolored) 'Choose: X' names currently legal, given whatever
    kind of pending resolution -- if any -- is active. "choose_permanent"
    is NOT handled here -- see _choose_permanent_legal/_choose_permanent_
    execute below: it needs exact (name, slot) addressing (docs/
    "Permanent identity"), same as
    "choose_opponent_permanent" already gets, not this generic by-name
    dispatch.

    Every kind handled here must be in _CHOOSE_NAME_PENDING_KINDS -- see
    that constant's own docstring for the invariant it (and this function)
    are required to uphold."""
    pending = state.pending_resolution
    if pending is None:
        return []
    kind = pending["kind"]
    # Float-first: no tap-during-payment. Mana is produced by top-level mana
    # abilities BEFORE casting; paying a cost only ever spends floated pool mana
    # (_pool_spend), never taps a source here. So "pay_cost" is absent from this
    # by-name dispatch.
    if kind == "search_fetch":
        return game.search_fetch_options(state)
    if kind == "throne_reveal":  # Undercity Throne: pick a creature card from the revealed top 10
        return game.throne_reveal_options(state)
    # choose_graveyard_card, choose_stack_target, and choose_permanent
    # (which now also covers every generic sacrifice) are deliberately
    # absent: all are POINTER targets (rl.action_bridge), not by-name fixed
    # actions -- the chosen card/stack-entry/permanent is picked by pointing
    # at its token, so an opponent's cards are reachable (and a battlefield
    # permanent is addressed exactly, not by a fungible name) without a
    # whole-league "Choose: X" fixed row per card name.
    if kind == "discard":
        return game.discard_options(state)
    if kind == "discard_or_sacrifice":
        # Only the DISCARD half reuses this generic "Choose: X" dispatch
        # (bare hand-card names, same as plain "discard") -- the sacrifice
        # half is a single trigger action that opens its own nested
        # choose_permanent pointer choice instead (see
        # _discard_or_sacrifice_trigger_sacrifice_legal's own docstring),
        # precisely to avoid ambiguity if a hand card and a battlefield land
        # ever share a name (e.g. a Mountain in hand while Mountains are
        # also in play) -- two different action shapes, never one bare name
        # that could mean either.
        return game.discard_or_sacrifice_discard_options(state)
    if kind == "ancient_stirrings":
        return [n for n in game.ancient_stirrings_options(state) if n != "decline"]
    if kind == "malevolent_rumble":
        return [n for n in game.malevolent_rumble_options(state) if n != "decline"]
    if kind in ("scry", "surveil") and pending["ordered"] is not None:
        return game.scry_surveil_options(state)
    if kind == "select_to_hand" and pending["ordered"] is not None:
        return game.select_to_hand_options(state)  # ordering phase only -- "keep"/"bottom" are their own actions
    if kind == "order_triggers":
        return game.order_triggers_options(state)
    if kind == "put_on_top":  # Brainstorm: which hand card to put on top next
        return game.put_on_top_options(state)
    if kind == "ponder":  # Ponder: which revealed card to place on top next ("Shuffle (Ponder)" is its own action)
        return game.ponder_options(state)
    return []


def _choose_name_legal(name):
    def legal(state):
        return name in _choose_name_options(state)
    legal._pending_gate = _CHOOSE_NAME_PENDING_KINDS
    return legal


def _choose_name_execute(name):
    def execute(state):
        kind = state.pending_resolution["kind"]
        if kind == "search_fetch":
            game.execute_search_fetch_option(state, name)
        elif kind == "throne_reveal":
            game.execute_throne_reveal_option(state, name)
        elif kind == "discard":
            game.execute_discard_option(state, name)
        elif kind == "discard_or_sacrifice":
            game.execute_discard_or_sacrifice_discard(state, name)
        elif kind == "ancient_stirrings":
            game.execute_ancient_stirrings_option(state, name)
        elif kind == "malevolent_rumble":
            game.execute_malevolent_rumble_option(state, name)
        elif kind == "select_to_hand":
            game.execute_select_to_hand_option(state, name)  # ordering phase only
        elif kind == "order_triggers":
            game.execute_order_triggers_option(state, name)
        elif kind == "put_on_top":
            game.execute_put_on_top_option(state, name)
        elif kind == "ponder":
            game.execute_ponder_option(state, name)
        else:  # scry / surveil, ordering phase
            game.execute_scry_surveil_option(state, name)
    return execute


def _attack_legal(name, slot):
    """Legal only during Phase.DECLARE_ATTACKERS, and only for the true
    turn owner (state.active_idx == state.turn_player_idx,
   ) -- declaring an attacker is a turn-based
    special action, not a priority action, so the non-turn player must
    never be allowed to declare one just because state.phase (a single
    shared field describing the TURN's phase) happens to match during
    their own priority window. And only if the specific physical
    permanent occupying this (name, slot)
    permanent-identity design -- is currently attack-eligible
    (game.creature_attack_eligible): untapped, and not summoning sick
    unless it has haste. Attacking stays fully optional: a model can leave
    any subset of eligible creatures back, Pass with zero attackers
    declared is still legal (same as always -- state.attackers simply
    starts, and can stay, empty for this turn).

    Gated _GATE_NO_PENDING (owner-confirmed): a genuine DECLARE_ATTACKERS
    window never coexists with an open pending_resolution -- anything an
    attack declaration itself triggers (e.g. "on attacks" abilities) gets
    its own later priority window, before declare-blockers, not during
    declare-attackers. So `pending is not None` already rules this out;
    the phase/turn-owner checks below still gate the (much more common)
    case where pending IS None but it isn't a legal attack window."""
    def legal(state):
        if state.phase is not game.turn.Phase.DECLARE_ATTACKERS:
            return False
        if state.active_idx != state.turn_player_idx:
            return False
        p = _cached_battlefield_lookup(state).get((name, slot))
        return p is not None and game.creature_attack_eligible(state, p)
    legal._pending_gate = _GATE_NO_PENDING
    return legal


def _attack_execute(name, slot):
    """Declares the specific physical permanent occupying this (name,
    slot) as an attacker -- exact-slot addressing lets a model distinguish
    an Aura-enchanted copy (different effective power) from a plain one of
    the same name."""
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
    "Permanent identity") -- a plain by-name "Choose:
    X" can't tell two same-named permanents apart, and cast_aura's whole
    fizzle-on-invalid-target contract depends on knowing exactly which one
    was chosen."""
    def legal(state):
        pending = state.pending_resolution
        return (
            pending is not None and pending["kind"] == "choose_permanent"
            and (name, slot) in game.choose_permanent_options(state)
        )
    legal._pending_gate = frozenset({"choose_permanent"})
    return legal


def _choose_permanent_execute(name, slot):
    def execute(state):
        game.execute_choose_permanent_option(state, name, slot)
    return execute


def _choose_opponent_permanent_legal(name, slot):
    """The general cross-player targeting primitive's action-table half
    -- legal only while a "choose_opponent_permanent"
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
    legal._pending_gate = frozenset({"choose_opponent_permanent"})
    return legal


def _choose_opponent_permanent_execute(name, slot):
    def execute(state):
        game.execute_choose_opponent_permanent_option(state, name, slot)
    return execute


def _assign_blocker_legal(name, slot):
    """One "Assign Blocker: <name> (slot j)" action -- legal only while a "declare_blockers" resolution
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
        p = _cached_battlefield_lookup(state).get((name, slot))
        return p is not None and game.creature_block_eligible(state, p)
    legal._pending_gate = frozenset({"declare_blockers"})
    return legal


def _assign_blocker_execute(name, slot):
    """Parks the specific physical permanent at this (name, slot) as a
    blocker, then hands off to game.declare_blocker_assignment, which
    nests a cross-player choose_opponent_permanent sub-resolution to pick
    which of the attacker's declared, not-yet-blocked attackers it
    blocks -- restricted by extra_predicate to attackers this specific
    blocker is actually allowed to block: flying's own restriction
    means an attacker with flying can only be
    chosen here if `blocker` itself also has flying (game.has_keyword --
    resolution can't compute this itself, see declare_blocker_
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
    if pending is None or pending["kind"] != "declare_blockers":
        return False
    # Menace (509.1c): can't FINISH a block declaration that leaves a menace
    # attacker blocked by exactly one creature -- the defender must add a
    # second blocker. No undo available (by design -- see
    # game.menace_block_incomplete's own docstring): if no second blocker is
    # available, this stays illegal until the phase's action cap forces
    # completion and combat.enforce_menace drops the illegal lone block.
    return not game.menace_block_incomplete(state)


_done_blocking_legal._pending_gate = frozenset({"declare_blockers"})


def _done_blocking_execute(state):
    game.complete_resolution(state)


def _assign_damage_to_opponent_legal(state):
    """The trample "assign this combat-damage point to the defending player"
    action -- legal only during an assign_combat_damage resolution whose
    attacker HAS trample and still has points to assign (the blockers are
    the pointer half of this decision; this fixed action is the player
    half). NOT a targeting-prefixed name, so build_fixed_action_table keeps
    it in the fixed table rather than stripping it to the pointer scheme."""
    pending = state.pending_resolution
    return (
        pending is not None and pending["kind"] == "assign_combat_damage"
        and pending["has_trample"] and pending["remaining"] > 0
    )


_assign_damage_to_opponent_legal._pending_gate = frozenset({"assign_combat_damage"})


def _assign_damage_to_opponent_execute(state):
    game.execute_assign_combat_damage_to_player(state)


def _pool_spend_legal(color):
    def legal(state):
        return (
            state.pending_resolution is not None
            and state.pending_resolution["kind"] == "pay_cost"
            and color in game.pool_spend_options(state)
        )
    legal._pending_gate = frozenset({"pay_cost"})
    return legal


def _pool_spend_execute(color):
    def execute(state):
        game.execute_pool_spend(state, color)
    return execute


def _keep_dispose_legal(state):
    pending = state.pending_resolution
    return pending is not None and pending["kind"] in ("scry", "surveil") and bool(pending["remaining"])


_keep_dispose_legal._pending_gate = frozenset({"scry", "surveil"})


def _keep_execute(state):
    game.execute_scry_surveil_option(state, "keep")


def _dispose_execute(state):
    game.execute_scry_surveil_option(state, "dispose")


# NOTE: nothing in this action table references pregame mulligan decisions --
# the MulliganNet (rl.mulligan) owns the pregame phase instead (see the
# pregame-mulligan note further down, near the universal decision rows). The
# engine's own mulligan (game.execute_mulligan_keep/take, game.turn.
# run_mulligan_phase) is unaffected.


def _decline_legal(state):
    pending = state.pending_resolution
    return pending is not None and pending["kind"] == "ancient_stirrings"


_decline_legal._pending_gate = frozenset({"ancient_stirrings"})


def _decline_execute(state):
    game.execute_ancient_stirrings_option(state, "decline")


def _decline_malevolent_rumble_legal(state):
    pending = state.pending_resolution
    return pending is not None and pending["kind"] == "malevolent_rumble"


_decline_malevolent_rumble_legal._pending_gate = frozenset({"malevolent_rumble"})


def _decline_malevolent_rumble_execute(state):
    game.execute_malevolent_rumble_option(state, "decline")




def _ponder_shuffle_legal(state):
    """Ponder's "you may shuffle" -- an alternative to ordering the revealed
    cards, so legal only BEFORE any card has been placed on top (ordered
    still empty). Once ordering has begun, that choice is made."""
    pending = state.pending_resolution
    return pending is not None and pending["kind"] == "ponder" and not pending["ordered"]


_ponder_shuffle_legal._pending_gate = frozenset({"ponder"})


def _ponder_shuffle_execute(state):
    game.execute_ponder_shuffle(state)


def _pay_unless_pay_legal(state):
    """"Pay {N}" for the Spell Pierce / Ward rider -- legal only while a
    pay_unless resolution is open AND the payer can actually afford the {N}
    (active_idx is already flipped to the payer, so plan_payment reads THEIR
    sources)."""
    pending = state.pending_resolution
    if pending is None or pending["kind"] != "pay_unless":
        return False
    return game.plan_payment(state, pending["cost"]) is not None


_pay_unless_pay_legal._pending_gate = frozenset({"pay_unless"})


def _pay_unless_pay_execute(state):
    game.pay_unless_pay(state)


def _pay_unless_decline_legal(state):
    pending = state.pending_resolution
    return pending is not None and pending["kind"] == "pay_unless"


_pay_unless_decline_legal._pending_gate = frozenset({"pay_unless"})


def _pay_unless_decline_execute(state):
    game.pay_unless_decline(state)


def _may_transform_legal(state):
    pending = state.pending_resolution
    return pending is not None and pending["kind"] == "may_transform"


_may_transform_legal._pending_gate = frozenset({"may_transform"})


def _may_copy_legal(state):
    pending = state.pending_resolution
    return pending is not None and pending["kind"] == "may_copy"


_may_copy_legal._pending_gate = frozenset({"may_copy"})


def _may_cast_legal(state):
    pending = state.pending_resolution
    return pending is not None and pending["kind"] == "may_cast"


_may_cast_legal._pending_gate = frozenset({"may_cast"})


def _choose_room_legal(room):
    def legal(state):
        pending = state.pending_resolution
        return pending is not None and pending["kind"] == "choose_room" and room in pending["options"]
    legal._pending_gate = frozenset({"choose_room"})
    return legal


def _choose_room_execute(room):
    return lambda state: game.execute_choose_room_option(state, room)


def _choose_mana_color_legal(color):
    def legal(state):
        pending = state.pending_resolution
        return pending is not None and pending["kind"] == "choose_mana_color"
    legal._pending_gate = frozenset({"choose_mana_color"})
    return legal


def _choose_mana_color_execute(color):
    return lambda state: game.execute_choose_mana_color(state, color)


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


_select_to_hand_keep_legal._pending_gate = frozenset({"select_to_hand"})


def _select_to_hand_bottom_legal(state):
    pending = state.pending_resolution
    return pending is not None and pending["kind"] == "select_to_hand" and bool(pending["remaining"])


_select_to_hand_bottom_legal._pending_gate = frozenset({"select_to_hand"})


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


_decline_search_legal._pending_gate = frozenset({"search_fetch"})


def _decline_search_execute(state):
    game.execute_search_fetch_decline(state)


def _decline_graveyard_card_legal(state):
    """Only for an OPTIONAL choose_graveyard_card (Masked Vandal's "you may
    exile a creature card from your graveyard") with real options to decline
    -- gated on pending["optional"] so it never appears for Dread Return /
    Relic's own MANDATORY graveyard picks, which share the same kind."""
    pending = state.pending_resolution
    return (
        pending is not None and pending["kind"] == "choose_graveyard_card" and pending.get("optional")
        and bool(game.choose_graveyard_card_options(state))
    )


_decline_graveyard_card_legal._pending_gate = frozenset({"choose_graveyard_card"})


def _decline_graveyard_card_execute(state):
    game.execute_choose_graveyard_card_decline(state)


def _decline_discard_legal(state):
    pending = state.pending_resolution
    return (
        pending is not None and pending["kind"] == "discard" and pending.get("optional")
        and bool(game.discard_options(state))
    )


_decline_discard_legal._pending_gate = frozenset({"discard"})


def _decline_discard_execute(state):
    game.execute_discard_decline(state)


def _target_self_legal(state):
    pending = state.pending_resolution
    return pending is not None and pending["kind"] == "choose_target_player"


_target_self_legal._pending_gate = frozenset({"choose_target_player"})


def _target_self_execute(state):
    game.execute_choose_target_player_option(state, state.active_idx)


def _target_opponent_legal(state):
    """Legal only once a real second PlayerState exists -- "target
    player" genuinely offers a choice the instant one does (Relic of
    Progenitus' own repeatable exile ability), same "only legal with a
    real opponent" gate deal_damage_to_opponent's own 2-player branch
    already uses elsewhere."""
    pending = state.pending_resolution
    return pending is not None and pending["kind"] == "choose_target_player" and len(state.players) > 1


_target_opponent_legal._pending_gate = frozenset({"choose_target_player"})


def _target_opponent_execute(state):
    game.execute_choose_target_player_option(state, 1 - state.active_idx)


def _target_any_self_legal(state):
    """The "any target" player half (real Magic: a player is always a legal
    "any target", including yourself -- Lightning Bolt to your own face is
    legal, if rarely wise). Only offered when the pending choose_any_target
    allows players (a "target creature"-only choice sets allow_players=False
    and this stays masked). The creature half of the same choice rides the
    identity pointer scheme (rl.action_bridge), not a fixed action."""
    pending = state.pending_resolution
    return pending is not None and pending["kind"] == "choose_any_target" and pending["allow_players"]


_target_any_self_legal._pending_gate = frozenset({"choose_any_target"})


def _target_any_self_execute(state):
    game.execute_choose_any_target_player(state, state.active_idx)


def _target_any_opponent_legal(state):
    pending = state.pending_resolution
    return (
        pending is not None and pending["kind"] == "choose_any_target"
        and pending["allow_players"] and len(state.players) > 1
    )


_target_any_opponent_legal._pending_gate = frozenset({"choose_any_target"})


def _target_any_opponent_execute(state):
    game.execute_choose_any_target_player(state, 1 - state.active_idx)


def _target_any_decline_legal(state):
    """Decline an "up to one target" (optional) choose_any_target -- e.g.
    Pinnacle Kill-Ship's ETB choosing zero targets. Only legal when the
    pending was begun optional=True."""
    pending = state.pending_resolution
    return pending is not None and pending["kind"] == "choose_any_target" and pending.get("optional", False)


_target_any_decline_legal._pending_gate = frozenset({"choose_any_target"})


def _target_any_decline_execute(state):
    game.execute_choose_any_target_decline(state)


def _discard_or_sacrifice_trigger_sacrifice_legal(state):
    """The SACRIFICE half of Highway Robbery's own "discard a card or
    sacrifice a land" -- ONE trigger action (not one per land name):
    picking it opens a nested choose_permanent sub-decision for WHICH exact
    land pays the cost (game.execute_discard_or_sacrifice_trigger_
    sacrifice), giving the model the same real per-instance choice
    begin_sacrifice's own predicate-driven picks get (see that function's
    own docstring for why first-same-name-match isn't good enough --
    battlefield permanents aren't fungible the way hand/library cards are),
    instead of a name per eligible land the way the DISCARD half's "Choose:
    X" rows work. Legal only while discard_or_sacrifice is pending and at
    least one eligible permanent exists."""
    pending = state.pending_resolution
    return (
        pending is not None and pending["kind"] == "discard_or_sacrifice"
        and game.discard_or_sacrifice_can_sacrifice(state)
    )


_discard_or_sacrifice_trigger_sacrifice_legal._pending_gate = frozenset({"discard_or_sacrifice"})


def _discard_or_sacrifice_trigger_sacrifice_execute(state):
    game.execute_discard_or_sacrifice_trigger_sacrifice(state)


def _decline_discard_or_sacrifice_legal(state):
    pending = state.pending_resolution
    return pending is not None and pending["kind"] == "discard_or_sacrifice"


_decline_discard_or_sacrifice_legal._pending_gate = frozenset({"discard_or_sacrifice"})


def _decline_discard_or_sacrifice_execute(state):
    game.execute_discard_or_sacrifice_decline(state)


def _madness_cast_legal(state):
    """Legal only if the model can actually afford the exiled card's
    madness cost right now -- same "guaranteed payable, not a maybe"
    contract every other alternate cast path here already follows."""
    pending = state.pending_resolution
    if pending is None or pending["kind"] != "madness_decision":
        return False
    madness_spec = game.EFFECT_REGISTRY[pending["card_def"].effect_id]["madness"]
    return game.plan_payment(state, madness_spec["cost"]) is not None


_madness_cast_legal._pending_gate = frozenset({"madness_decision"})


def _madness_cast_execute(state):
    game.execute_madness_cast(state)


def _madness_decline_legal(state):
    pending = state.pending_resolution
    return pending is not None and pending["kind"] == "madness_decision"


_madness_decline_legal._pending_gate = frozenset({"madness_decision"})


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
    legal._pending_gate = _GATE_NO_PENDING
    return legal


def _activate_no_cost_execute(name, resolve):
    def execute(state):
        p = next(p for p in state.battlefield if p.card_def.name == name)
        resolve(state, p)
    return execute


def _alt_cast_legal(name, extra_legal, speed):
    """Land Grant's free alt-cost: no mana payment at all, just the
    card's own extra_legal predicate (0 lands in hand).

    Availability must go through _hand_count_available, not a bare
    "any copy in hand" check -- confirmed live via mono_red_madness_mirror
    training: a bare existence check let Fireblast's alt-cost (sacrifice 2
    Mountains) be cast a second time while the first cast's own copy was
    still physically in hand but already reserved on the stack (removal
    deferred to its own resolve, same as every cast-like path -- see
    push_to_stack's docstring), pushing a second stack entry for the same
    physical card. cast_fireblast_alt's own discard_from_hand_to_graveyard
    then ate that shared copy immediately (its eager, non-deferred
    hand-removal), so when the FIRST cast's stack entry finally resolved,
    its own discard_from_hand_to_graveyard found no copy left -- the
    "should be unreachable" RuntimeError."""
    def legal(state):
        if state.pending_resolution is not None:
            return False
        if not game.turn.speed_legal(state, speed):
            return False
        if _hand_count_available(state, name) <= 0:
            return False
        return extra_legal(state)
    legal._pending_gate = _GATE_NO_PENDING
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


def _flashback_legal(name, ability_legal, speed, cost=None):
    """Dread Return's Flashback: cast from the graveyard, not hand. Real
    Magic: Flashback follows the same timing as the card itself, not its
    own independent rule -- speed is the same value the card's normal
    cast derived, not a separate default.

    cost (optional): a MANA cost dict for a flashback whose flashback cost
    includes mana (Deep Analysis' {1}{U}, Faithless Looting's {2}{R}). When
    present, its affordability is checked here (plan_payment) exactly like a
    normal cast; the truly free/sacrifice-only flashbacks (Lava Dart, Dread
    Return) leave it None and pay entirely inside their own resolve. Any
    NON-mana additional cost (Deep Analysis' "Pay 3 life") is gated by
    ability_legal instead (it can't be expressed as a mana dict)."""
    def legal(state):
        if state.pending_resolution is not None:
            return False
        if not game.turn.speed_legal(state, speed):
            return False
        if not any(c.name == name for c in state.graveyard):
            return False
        if cost is not None and game.plan_payment(state, cost) is None:
            return False
        return ability_legal(state)
    legal._pending_gate = _GATE_NO_PENDING
    return legal


def _graveyard_instance(state, name):
    """THE type->instance boundary for every graveyard-sourced action.

    The action table is built from card NAMES (it must be -- a deck's action
    space is fixed-width), so an execute closure only ever holds a name and has
    to recover the real game object. For a HAND cast that's free: hand holds
    CardDefs and game.CARD_DEFS[name] IS the object in hand, so type-identity
    and object-identity coincide. For a GRAVEYARD cast they don't -- the
    graveyard holds per-object CardInstances (see plans/object-identity-zone-
    model.md), and game.CARD_DEFS[name] is the interned, one-per-name rules
    definition, never identity-equal to any instance.

    Passing that CardDef onward is what forced every graveyard-cast resolve to
    re-derive the instance itself: six did it by name (via the old
    casting.remove_graveyard_card, or hand-rolled), and the seventh
    (flashback_deep_analysis) forgot and crashed a real pretrain run with
    `ValueError: list.remove(x): x not in list`. Resolving it HERE, once, is
    what lets every downstream removal/capture be a true identity operation.

    Picks the first same-named instance. MTG 400.7 makes same-named graveyard
    cards interchangeable for "a card with this name leaves the graveyard", so
    any one is legal today. Real Magic does let the PLAYER choose which copy,
    and that choice becomes observable when another object references one
    specific copy (a Rooftop Percher target locked on copy A while copy B is
    the one flashed back). Routing that choice through the pointer head is a
    deliberate OPEN ITEM for the owner, not settled here."""
    inst = next((c for c in state.graveyard if c.name == name), None)
    if inst is None:
        # Fail loudly with context, not a bare StopIteration -- same precedent
        # as game.effects.shared.discard_from_hand_to_graveyard's own guard.
        # Unreachable via a legal action (_flashback_legal/_graveyard_ability_legal
        # both require a same-named graveyard card), so this means a caller's own
        # guarantee broke.
        raise RuntimeError(
            f"_graveyard_instance: no {name!r} in graveyard. "
            f"active_idx={getattr(state, 'active_idx', None)!r} "
            f"turn_number={getattr(state, 'turn_number', None)!r} "
            f"graveyard={[c.name for c in state.graveyard]!r}"
        )
    return inst


def _with_chosen_copy(state, name, proceed, reserved_cost=None):
    """Run `proceed(state, inst)` on the graveyard copy of `name` being cast.

    With 2+ same-named copies this is a REAL agent choice (MTG 601.2a -- the
    object being cast is chosen at announcement), so it opens a pointer-only
    choose_cast_copy pending and continues from its on_complete; the choice is
    made BEFORE any cost is paid, which is both the faithful order and what
    keeps the single pending_resolution slot free for the payment that follows.
    With exactly one copy there is no choice to make, so it proceeds inline:
    this is not a simplification, just the absence of a decision (the harness
    would auto-resolve a one-option pending anyway; skipping it avoids a
    pointless token-set build + mask sweep on the common path).

    reserved_cost: the mana cost `proceed` will pay via begin_pay_cost once a
    copy is chosen (None if there is none, e.g. a life-only flashback cost) --
    threaded straight through to begin_choose_cast_copy, whose own docstring
    explains why a not-yet-open payment still needs strand protection during
    this choice. Irrelevant on the single-copy fast path: nothing else gets a
    turn to filter mana away between this call and begin_pay_cost when there's
    no intervening pending resolution at all."""
    copies = [c for c in state.graveyard if c.name == name]
    if len(copies) <= 1:
        proceed(state, _graveyard_instance(state, name))
        return
    game.begin_choose_cast_copy(state, name, on_complete=proceed, reserved_cost=reserved_cost)


def _flashback_execute(name, resolve, cost=None):
    """resolve receives the graveyard CardInstance being cast (NOT the interned
    CardDef) -- see _graveyard_instance. Every flashback/escape resolve removes
    that exact object from the graveyard by identity. WHICH copy, when the
    graveyard holds several, is the agent's own choice (_with_chosen_copy)."""
    def execute(state):
        def _proceed(state, inst):
            if cost is None:
                game.on_cast_trigger(state, inst)  # item 11 -- see _cast_execute
                resolve(state, inst)
                return
            # Mana flashback cost: pay it AFTER the copy is chosen (601.2a
            # announce-then-pay), then fire the on-cast trigger and run the
            # resolve, which pays any further additional cost (life) and pushes
            # the effect.
            def _after_pay(state, inst=inst):
                game.on_cast_trigger(state, inst)
                resolve(state, inst)
            game.begin_pay_cost(state, cost, on_complete=_after_pay)

        _with_chosen_copy(state, name, _proceed, reserved_cost=cost)
    return execute


def _impulse_entry(state, name):
    """The topmost still-unexpired impulse entry (card_def, deadline) for
    `name`, or None. Expired entries are pruned at untap, but this also
    re-checks the deadline defensively."""
    for cd, deadline in reversed(state.impulse):
        if cd.name == name and state.turn_number <= deadline:
            return (cd, deadline)
    return None


def _play_impulse_land_legal(name):
    def legal(state):
        if state.pending_resolution is not None:
            return False
        if _impulse_entry(state, name) is None:
            return False
        if not game.turn.speed_legal(state, game.turn.Speed.SORCERY):  # playing a land is sorcery-speed, your turn
            return False
        return state.lands_played_this_turn == 0
    legal._pending_gate = _GATE_NO_PENDING
    return legal


def _play_impulse_land_execute(name):
    def execute(state):
        entry = _impulse_entry(state, name)
        state.impulse.remove(entry)
        state.hand.append(entry[0])  # source it via hand so play_land_from_hand works (Cascade-style insertion)
        game.play_land_from_hand(state, entry[0])
    return execute


def _play_impulse_cast_legal(name, cost, extra_legal, speed):
    def legal(state):
        if state.pending_resolution is not None:
            return False
        if _impulse_entry(state, name) is None:
            return False
        if not game.turn.speed_legal(state, speed):
            return False
        if game.plan_payment(state, cost) is None:
            return False
        return extra_legal is None or extra_legal(state)
    legal._pending_gate = _GATE_NO_PENDING
    return legal


def _play_impulse_cast_execute(name, cost, resolve, precast):
    """Cast an impulse-exiled spell for `cost` (its normal cost -- impulse,
    unlike Plot, is NOT free). Mirrors _cast_execute/_precast_choice_execute,
    but the card is removed from the impulse zone only in _after_pay (once
    mana is actually paid) -- so a not-yet-paid cast leaves it in impulse,
    no leak. Then it's inserted into hand (Cascade-style, so the card's own
    resolve, written for a hand cast, finds and removes it) and either pushed
    to the stack (non-precast) or resolved directly (precast, which pushes
    itself)."""
    def execute(state):
        entry = _impulse_entry(state, name)

        def _after_pay(s):
            if entry in s.impulse:
                s.impulse.remove(entry)
            s.hand.append(entry[0])
            game.on_cast_trigger(s, entry[0])
            if precast:
                resolve(s, entry[0])
            else:
                game.push_to_stack(s, entry[0], resolve)

        game.begin_pay_cost(state, cost, on_complete=_after_pay)
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
    legal._pending_gate = _GATE_NO_PENDING
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
    legal._pending_gate = _GATE_NO_PENDING
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
        game.push_to_stack(state, card_def, resolve, reserves_hand_card=False)
    return execute


def _omen_cast_legal(hand_name, cost, speed):
    """Sagu Wildling's Omen: real Scryfall reminder text is "(Also shuffle
    this card.)" attached to Roost Seek's own library search -- unlike
    real Adventure, an Omen card does NOT exile itself for a later free-
    zone cast; the resolved sorcery is shuffled directly into the LIBRARY
    (cast_roost_seek), and the real creature half only ever becomes
    castable again once the same physical card is redrawn into HAND, same
    as any ordinary card. So this is really just "the same hand card, a
    second cast option with its own distinct cost" -- checked against
    state.hand, not state.exile. hand_name is the SORCERY side's own
    registered name (the only one ever physically in a zone) -- the
    creature side is a distinct CardDef, never separately registered in
    game.CARD_DEFS (see the "omen" registry spec's own "card_def" key)."""
    def legal(state):
        if state.pending_resolution is not None:
            return False
        if not game.turn.speed_legal(state, speed):
            return False
        if not any(c.name == hand_name for c in state.hand):
            return False
        return game.plan_payment(state, cost) is not None
    legal._pending_gate = _GATE_NO_PENDING
    return legal


def _omen_cast_execute(creature_card_def, cost, resolve):
    """Same begin_pay_cost -> push_to_stack shape as a normal hand cast
    (_cast_execute), just for `creature_card_def` (the distinct creature
    CardDef) instead of game.CARD_DEFS[name]. reserves_hand_card defaults
    True here (unlike Plot/Flashback's own exile/graveyard-sourced pushes)
    -- the physical card genuinely IS still sitting in the caster's hand,
    unresolved, while this is paid for; _hand_count_available matches
    stack entries by NAME, so pushing creature_card_def (same display name
    as whatever's in hand) still correctly reserves it, blocking the
    sorcery-mode cast of the same physical copy in the meantime. `resolve`
    is responsible for removing the matching hand card itself (by NAME,
    not identity -- the object actually sitting in state.hand is the
    sorcery side's own CardDef, a different object from creature_card_def
    despite sharing a display name), same "resolve does its own zone
    removal" convention every other cast path here follows."""
    def execute(state):
        game.on_cast_trigger(state, creature_card_def)  # no-op for a CREATURE card_def (on_cast_trigger only fires for INSTANT/SORCERY) -- called anyway for the same hygiene every other cast path here has

        def _after_pay(s):
            # The physical hand card LEAVES hand at cast, like every other cast
            # (game.push_to_stack) -- never re-entering hand. It shares
            # creature_card_def's display name but is a DIFFERENT object (the
            # hand card is the sorcery/normal side), so push_to_stack's own
            # identity-based removal below misses it; remove it here by name.
            # That is what makes the OTHER mode uncastable while this copy is on
            # the stack, now that the card physically leaving hand is the sole
            # re-cast guard (_hand_count_available is a plain hand tally).
            hand_card = next((c for c in s.hand if c.name == creature_card_def.name), None)
            if hand_card is not None:
                s.hand.remove(hand_card)
            game.push_to_stack(s, creature_card_def, resolve)
        game.begin_pay_cost(state, cost, on_complete=_after_pay)
    return execute


def build_action_table(decklist, registry, token_card_defs=(),
                        opponent_decklist=None, opponent_token_card_defs=(), extra_choosable_names=()):
    """opponent_decklist/opponent_token_card_defs: the OTHER side's own
    decklist/tokens -- None/() for every 1-player deck (there's no real
    opponent battlefield to reference at all), matching combat_enabled=False
    decks never seeing "Attack: X" become legal. The token/pointer pipeline
    passes None here (cross-player targeting moved to the pointer head, not
    a fixed opponent action table) -- kept for a fixed "Choose opponent's:
    X" table if one is ever wanted again.

    token_card_defs: every token CardDef this deck's own cards can
    create at runtime (Blood, Robot, Warrior, Eldrazi Spawn --
   ), e.g. (game.BLOOD_TOKEN_CARD_DEF,).
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

    No `pending_kinds` parameter: whether a resolution kind can ever cross
    players is a fixed fact about the ENGINE (which cards target an
    opponent, which don't), not something a caller should have to compute
    and thread through -- see own_pending_kinds just below and the
    "UNIVERSAL DECISION ROWS" block further down for the two-way split this
    function makes on its own."""
    distinct_names = sorted({name for name, *_rest in decklist})
    # THIS decklist's own derived kinds. Used below for every kind confirmed
    # to be self-only (impulse, discard_or_sacrifice, may_transform,
    # may_cast, madness_decision, ancient_stirrings, malevolent_rumble,
    # select_to_hand, choose_mana_color -- each has exactly one call site in
    # the whole codebase, and it's always the deciding player's own card),
    # so those rows don't inflate a deck's table for a card it doesn't run.
    # The confirmed-cross-player kinds (pay_unless, tuck_position, may_copy,
    # choose_any_target, choose_room, choose_target_player, scry/
    # search_fetch's Decline, discard/graveyard-decline) don't consult this
    # at all -- they're unconditional in the "UNIVERSAL DECISION ROWS" block
    # further down, since an OPPONENT's card can pose those questions too.
    own_pending_kinds = game.derive_pending_kinds(decklist)
    land_names = sorted({
        name for name in distinct_names if game.CARD_DEFS[name].card_type == game.CardType.LAND
    })
    # Hoisted here (rather than just before their own first use, below) so the
    # extra-cost mana-ability loop can also use them: card_type_by_name/
    # qty_by_name index THIS side's own battlefield permanents (an opponent's
    # card is never a legal choose_permanent/attack/mana-extra-cost target of
    # mine); choosable_names is every name (real or token) this side could
    # ever have on its own battlefield.
    card_type_by_name = {name: game.CARD_DEFS[name].card_type for name in distinct_names}
    card_type_by_name.update({cd.name: cd.card_type for cd in token_card_defs})
    qty_by_name = {name: qty for name, qty, *_rest in decklist}
    # DFC back faces (Delver of Secrets -> Insectile Aberration): a transformed
    # permanent's card_def swaps to its back face (game.resolution.execute_may_
    # transform), so any BY-NAME resolution (order_triggers, sacrifice, discard,
    # search_fetch, ...) can end up needing to reference that back-face name --
    # omitting it here is the exact same softlock class as the token case just
    # above (a legal, on-battlefield object with no "Choose: X" row able to name
    # it), just for DFCs instead of tokens. Confirmed the hard way: a real
    # pretrain run hit "all-False action mask for pending kind 'order_triggers'"
    # the first time a transformed Delver's upkeep trigger needed ordering
    # against another simultaneous trigger. Discovered the same way
    # rl.features.CardVocab already discovers back faces: scan each front
    # face's own registry "transform" spec.
    back_face_defs = [
        spec["card_def"] for name in distinct_names
        for spec in [registry.get(game.CARD_DEFS[name].effect_id, {}).get("transform")]
        if spec is not None and "card_def" in spec
    ]
    card_type_by_name.update({cd.name: cd.card_type for cd in back_face_defs})
    choosable_names = sorted(set(distinct_names) | {cd.name for cd in token_card_defs} | {cd.name for cd in back_face_defs})

    actions = []

    for name in land_names:
        actions.append((f"Play land: {name}", _land_drop_legal(name), _land_drop_execute(name)))

    # Float-first mana abilities: "Tap X [for <color>]" produces mana into the
    # pool in ANY priority window, even mid-resolution of anything else
    # (605.1a/605.3b -- a mana ability never uses the stack and doesn't
    # require priority; see _mana_ability_legal's own docstring for why no
    # pending-resolution gate is needed at all). Row count is DERIVED per
    # source rather than a blanket "no-color + all 6 pool colors" for every
    # name: the registry's own "mana" spec kind already says, statically,
    # which colors that source's NATIVE ability could ever offer --
    # game.mana_ability_options only ever emits (name, None) for a fixed/
    # fixed_multi/tron/count/count_all source and (name, color) for color in
    # its own spec[1] for a flexible one -- so a row neither of those can
    # ever produce is a permanently-dead output neuron. "Tap X for C" is
    # never one of the rows generated below: colorless is only ever produced
    # via the no-color row (a fixed/tron/count source's own single output),
    # never a color CHOICE, for any source or grant this engine has.
    #
    # LANDS are the one case a per-source static lookup alone would miss:
    # Abundant Growth's cast_aura targets "any land, either side"
    # (choose_any_target) and grants a color via game.mana._granted_mana_
    # colors ON TOP OF the land's own native ability -- a runtime fact no
    # single card's "mana" spec captures. Any land in ANY league deck can
    # end up enchanted (this deck's own, or an opponent's, since Abundant
    # Growth can target either side), so every land-type source keeps a row
    # for every color the FULL registry ever declares grantable
    # ("grants_mana_colors", e.g. Abundant Growth's own entry), unioned with
    # its own native colors -- this is the one kind-of-decision in this
    # function that DOES need to look past just this decklist's own cards.
    grantable_mana_colors = set().union(*(spec.get("grants_mana_colors", set()) for spec in registry.values()))
    mana_source_effect_id = {}
    for name in distinct_names:
        effect_id = game.CARD_DEFS[name].effect_id
        spec = registry.get(effect_id, {})
        if "mana" in spec and "mana_extra_choose" not in spec:
            mana_source_effect_id[name] = effect_id
    for cd in token_card_defs:
        spec = registry.get(cd.effect_id, {})
        if "mana" in spec and "mana_extra_choose" not in spec:
            mana_source_effect_id[cd.name] = cd.effect_id
    for name in sorted(mana_source_effect_id):
        kind, *rest = registry[mana_source_effect_id[name]]["mana"]
        if kind != "flexible":
            # flexible is the ONLY kind whose native ability never offers
            # (name, None) -- see mana_ability_options's own kind grouping.
            actions.append((f"Tap {name}", _mana_ability_legal(name, None), _mana_ability_execute(name, None)))
        colors = set(rest[0]) if kind == "flexible" else set()
        if card_type_by_name.get(name) == game.CardType.LAND:
            colors |= grantable_mana_colors
        for color in game.COLORS:
            if color in colors:
                actions.append((
                    f"Tap {name} for {color}",
                    _mana_ability_legal(name, color),
                    _mana_ability_execute(name, color),
                ))

    # Saruli Caretaker-shaped mana abilities: an extra cost that taps ANOTHER
    # untapped creature (mana_extra_choose) -- a COST CHOICE (602.5g), not a
    # target. ONE gate-free "Tap <name>" row per source name (same shape as
    # an ordinary mana ability, mana_source_names above); its execute opens
    # a mana_subdecision (game.begin_mana_subdecision) -- choose which
    # creature to tap (pointer-routed, rl.action_bridge, no fixed-table row
    # needed), THEN choose a color (the shared "Produce <color>" buttons
    # below) -- matching real sequencing (602.5g cost paid before the
    # ability's own color-choice effect resolves) while staying gate-free
    # (605.1a/605.3b): mana_subdecision is a field SEPARATE from
    # pending_resolution specifically so this can still open mid-resolution
    # of anything else without clobbering it -- see state.mana_subdecision's
    # own docstring. needs_mana_subdecision_color_buttons is shared with the
    # mana-filter block below -- either mechanic reaches the SAME
    # choose_color stage (game.begin_mana_color_choice), so a deck needs the
    # buttons if EITHER is present, even with only one of the two.
    needs_mana_subdecision_color_buttons = False
    extra_tap_source_names = sorted(
        {name for name in distinct_names if "mana_extra_choose" in registry.get(game.CARD_DEFS[name].effect_id, {})}
        | {cd.name for cd in token_card_defs if "mana_extra_choose" in registry.get(cd.effect_id, {})}
    )
    for name in extra_tap_source_names:
        actions.append((f"Tap {name}", _mana_extra_choose_legal(name), _mana_extra_choose_execute(name)))
        needs_mana_subdecision_color_buttons = True

    # Mana filters (Conduit Pylons / Barrels of Blasting Jelly): "{1}: add one
    # mana of any color" -- the {1} half is a flat fixed-table row per
    # (source, input_color): which floating pip pays it, paid immediately.
    # input_color ranges over every pool color (any floating pip, incl.
    # colorless, can pay a generic {1}). The output half (which color comes
    # out) is a SEPARATE, later choice via the shared choose_color
    # mana_subdecision stage above -- reusing Saruli's own "Produce <color>"
    # buttons rather than a second per-row dimension (output_color x
    # input_color used to be a real cross product here; see
    # _filter_mana_execute). Never a nested pay_cost for the {1} itself:
    # folding it into one would open a second pending resolution
    # mid-activation, at risk of clobbering whatever's already open.
    filter_source_names = sorted({n for n in distinct_names if "filter_mana" in registry.get(game.CARD_DEFS[n].effect_id, {})})
    for name in filter_source_names:
        for input_color in game.POOL_COLORS:
            actions.append((
                f"Filter {name}, paying {input_color}",
                _filter_mana_legal(name, input_color),
                _filter_mana_execute(name, input_color),
            ))
    if filter_source_names:
        needs_mana_subdecision_color_buttons = True

    # Tracked so the generic choose_cast_mode/choose_cast_x/choose_delve_amount
    # buttons below only get added to a deck's table that actually has a card
    # using that shape -- these are MY OWN casting decisions (never posed by
    # an opponent's card, unlike the pay_unless/tuck_position-style universal
    # rows further down), so a deck with none of these cards never needs them.
    needs_cast_mode_buttons = False
    needs_cast_x_buttons = False
    needs_delve_buttons = False

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
        # Winding Way/Utopia Sprawl/Goblin Bushwhacker: a modal cast -- ONE
        # "Cast <name>" row (not one per mode); which mode is a generic
        # choose_cast_mode sub-decision (601.2b: chosen before the cost is
        # calculated), shared "Mode 1".."Mode 5" buttons reused by every
        # modal card instead of a per-card, per-mode fixed row.
        cast_modes = card_spec.get("cast_modes")
        if cast_modes is not None:
            mode_items = list(cast_modes.items())
            speed = _cast_speed(game.CARD_DEFS[name], mode_items[0][1])
            actions.append((f"Cast {name}", _modal_legal(name, mode_items, speed), _modal_execute(name, mode_items)))
            needs_cast_mode_buttons = True
        # Nyxborn Hydra: X-cost modes (its own normal creature cast AND
        # Bestow, each with a different base cost) -- ONE "Cast <name>" row;
        # mode chosen first (601.2b), X chosen second (601.2f), both generic
        # sub-decisions (shared "Mode 1".."Mode 5" then "X=0".."X=10"
        # buttons) instead of one fixed row per (mode, X) pair. Each mode's
        # own "resolve" is still a function OF x returning the (state,
        # card_def) resolve itself (green_cards.cast_nyxborn_hydra_creature/
        # cast_nyxborn_hydra_bestow) -- called once X is actually chosen
        # (_x_modal_execute), not baked in at table-build time anymore.
        x_cast_modes = card_spec.get("x_cast_modes")
        if x_cast_modes is not None:
            mode_items = list(x_cast_modes.items())
            speed = _cast_speed(game.CARD_DEFS[name], mode_items[0][1])
            actions.append((f"Cast {name}", _x_modal_legal(name, mode_items, speed), _x_modal_execute(name, mode_items)))
            needs_cast_mode_buttons = True
            needs_cast_x_buttons = True
        # Gurmag Angler: Delve -- ONE "Cast <name>" row; the delve amount is
        # a generic choose_delve_amount sub-decision (702.66: chosen before
        # the exile sub-cost opens and the reduced cost is calculated),
        # shared "Delve 0".."Delve 6" buttons instead of one fixed row per N.
        delve = card_spec.get("delve")
        if delve is not None:
            delve_speed = _cast_speed(game.CARD_DEFS[name], delve)
            actions.append((
                f"Cast {name}",
                _delve_legal(name, delve["max"], delve_speed),
                _delve_execute(name, delve["max"], delve["resolve"]),
            ))
            needs_delve_buttons = True
        # Land Grant: a second, free cast path alongside the normal one.
        alt_cast = card_spec.get("alt_cast")
        if alt_cast is not None:
            actions.append((
                f"Cast {name} (free)",
                _alt_cast_legal(name, alt_cast["extra_legal"], _cast_speed(game.CARD_DEFS[name], alt_cast)),
                _alt_cast_execute(name, alt_cast["resolve"]),
            ))
        # Dread Return: Flashback casts from the graveyard, not hand. Escape
        # (Sleep of the Dead) is the same graveyard-cast machinery with a mana
        # cost + its own additional cost handled inside the resolve -- shares
        # _flashback_legal/_execute, only the action label differs.
        for gy_cast_key, gy_cast_label in (("flashback", "Flashback"), ("escape", "Escape")):
            gy_cast = card_spec.get(gy_cast_key)
            if gy_cast is not None:
                gc_cost = gy_cast.get("cost")  # mana cost dict, if the cost includes mana (Deep Analysis, Sleep of the Dead)
                actions.append((
                    f"{gy_cast_label} {name}",
                    _flashback_legal(name, gy_cast["legal"], _cast_speed(game.CARD_DEFS[name], gy_cast), gc_cost),
                    _flashback_execute(name, gy_cast["resolve"], gc_cost),
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
        # Sagu Wildling: Omen -- cast_roost_seek (this same "name"'s own
        # "cast" resolve above) shuffles the card into the LIBRARY instead
        # of graveyarding OR exiling it (real Adventure's own exile
        # doesn't apply to Omen -- see _omen_cast_legal's own docstring).
        # This second action is just an ordinary-looking second cast
        # option for the same hand card, its own real cost, offered
        # whenever a same-named card is back in hand (redrawn after being
        # shuffled in) -- never shares a resolve with the sorcery mode,
        # the creature side is a wholly different spell.
        omen = card_spec.get("omen")
        if omen is not None:
            omen_speed = _cast_speed(omen["card_def"], omen)
            actions.append((
                f"Cast {omen['card_def'].name} (omen)",
                _omen_cast_legal(name, omen["cost"], omen_speed),
                _omen_cast_execute(omen["card_def"], omen["cost"], omen["resolve"]),
            ))
        # Boulderbranch Golem: Prototype -- a second cast option for the same
        # hand card, its own cheaper cost, producing a DIFFERENT CardDef (the
        # smaller 3/3 with its own ETB). Structurally identical to Omen ("cast
        # this hand card for an alternate cost as a different creature"), so it
        # reuses the same _omen_cast_legal/_omen_cast_execute helpers -- only
        # the resolve differs (no library shuffle; the prototype creature just
        # enters). Real reminder text: "You may cast this spell with different
        # mana cost, color, and size. It keeps its abilities and types."
        prototype = card_spec.get("prototype")
        if prototype is not None:
            proto_speed = _cast_speed(prototype["card_def"], prototype)
            actions.append((
                f"Cast {name} (prototype)",
                _omen_cast_legal(name, prototype["cost"], proto_speed),
                _omen_cast_execute(prototype["card_def"], prototype["cost"], prototype["resolve"]),
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
            # working in every phase, unrestricted by speed.
            speed = spec.get("speed", game.turn.Speed.INSTANT)
            if "cost_key" in spec:
                actions.append((
                    f"Activate {name} ({ability_name})",
                    _activate_legal(name, spec["cost_key"], speed, spec.get("extra_legal")),
                    _activate_execute(name, spec["cost_key"], spec["resolve"]),
                ))
            else:
                # Non-mana cost (Quirion Ranger: return a Forest to hand).
                actions.append((
                    f"Activate {name} ({ability_name})",
                    _activate_no_cost_legal(name, spec["legal"], speed),
                    _activate_no_cost_execute(name, spec["resolve"]),
                ))

    # "Discard this card from hand: <do something>" cycling-family actions.
    # Both keys share the identical hand-zone/cost-key/resolve plumbing
    # (_forestcycle_legal/_execute) -- they differ only in the action label:
    #   "forestcycle" -- basic-land-to-hand search (Generous Ent, Ash Barrens)
    #   "cycle"       -- plain Cycling (discard, draw) and typed cycling like
    #                    Islandcycling (Lorien Revealed) / Twisted Landscape
    for name in distinct_names:
        for spec_key, label in (("forestcycle", "Forestcycle"), ("cycle", "Cycle")):
            cyc_spec = registry.get(game.CARD_DEFS[name].effect_id, {}).get(spec_key)
            if cyc_spec is not None:
                actions.append((
                    f"{label} {name}",
                    _forestcycle_legal(name, cyc_spec["cost_key"]),
                    _forestcycle_execute(name, cyc_spec["cost_key"], cyc_spec["resolve"]),
                ))

    # Impulse: "you may play the exiled cards" (Reckless Impulse / Experimental
    # Synthesizer / Clockwork Percussionist). Only emitted for a deck that can
    # actually impulse (its impulse-source card declares pending_kinds
    # {"impulse"}), so decks without one never carry these mostly-illegal
    # actions. One action per deck card name: a land play, or a cast per the
    # card's own cast/cast_modes spec (paying its NORMAL cost -- impulse, unlike
    # Plot, is not free; x_cast_modes cards, none in the impulse decks, aren't
    # offered from impulse).
    #
    # Gated on THIS decklist's own derived kinds -- unlike pay_unless/
    # tuck_position (genuinely posed by an OPPONENT's card, unconditional in
    # the "UNIVERSAL DECISION ROWS" block further down), impulse is never
    # cross-player: state.impulse is always the ACTIVE player's own zone
    # (game/state.py's _active_player_property) and every impulse source
    # only ever exiles from its own controller's library (shared.
    # impulse_exile's own docstring). A deck without an impulse card of its
    # own never has "impulse" in own_pending_kinds, so it never carries a
    # "Play from exile: X" row per own land/castable card for nothing --
    # permanently illegal, since no card of theirs ever opens the zone.
    if "impulse" in own_pending_kinds:
        for name in distinct_names:
            card_def = game.CARD_DEFS[name]
            if card_def.card_type == game.CardType.LAND:
                actions.append((f"Play from exile: {name}", _play_impulse_land_legal(name), _play_impulse_land_execute(name)))
                continue
            spec = registry.get(card_def.effect_id, {})
            cast_spec = spec.get("cast")
            if cast_spec is not None:
                speed = _cast_speed(card_def, cast_spec)
                actions.append((
                    f"Play from exile: {name}",
                    _play_impulse_cast_legal(name, card_def.cast_cost, cast_spec.get("extra_legal"), speed),
                    _play_impulse_cast_execute(name, card_def.cast_cost, cast_spec["resolve"], cast_spec.get("precast_choice", False)),
                ))
            for mode_name, mode_spec in (spec.get("cast_modes") or {}).items():
                mode_cost = mode_spec.get("cost", card_def.cast_cost)
                speed = _cast_speed(card_def, mode_spec)
                actions.append((
                    f"Play from exile: {name} ({mode_name})",
                    _play_impulse_cast_legal(name, mode_cost, mode_spec.get("extra_legal"), speed),
                    _play_impulse_cast_execute(name, mode_cost, mode_spec["resolve"], mode_spec.get("precast_choice", False)),
                ))

    # Bramble Wurm: an activated ability usable from the graveyard, not
    # the battlefield (unlike every "activated_abilities" entry above) or
    # hand (unlike Forestcycle) -- its own "graveyard_ability" registry key.
    for name in distinct_names:
        gy_spec = registry.get(game.CARD_DEFS[name].effect_id, {}).get("graveyard_ability")
        if gy_spec is not None:
            actions.append((
                f"Activate {name} (graveyard)",
                _graveyard_ability_legal(name, gy_spec["cost_key"]),
                _graveyard_ability_execute(name, gy_spec["cost_key"], gy_spec["resolve"]),
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
    # extra_choosable_names: card names that can be a "Choose: X" option despite
    # not being in THIS deck (nor its tokens) -- e.g. an opponent's graveyard
    # cards. The league (rl.pool) passes none: choose_graveyard_card is a
    # POINTER target (rl.action_bridge), so an opponent's graveyard is reached
    # by pointing at its token, not a whole-league "Choose: X" row per card
    # name. Kept as a general knob, used by
    # scripts/migrate_pointer_graveyard.py to reconstruct the pre-pointer
    # table; defaults to none. choosable_names
    # itself (not just extra_choosable_names) also drives "Choose target: X
    # (slot k))" and "Attack: X (slot k)" below -- both strictly THIS side's
    # own battlefield permanents.
    choose_by_name = sorted(set(choosable_names) | set(extra_choosable_names))
    for name in choose_by_name:
        actions.append((f"Choose: {name}", _choose_name_legal(name), _choose_name_execute(name)))

    # "Attack: X (slot k)" -- one per (creature name, slot) pair
    #, legal only during
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
    # decision about this player's OWN creatures,
    # just legal at a different point (once _declare_blockers_gen has
    # flipped state.active_idx to the defender and a "declare_blockers"
    # resolution is pending -- see _assign_blocker_legal). "Done blocking"
    # is the explicit action that closes the consult, same "Done" precedent
    # as scry/surveil's own keep-then-order decomposition. Deliberately NO
    # AUTHORIZED SIMPLIFICATION (owner, 2026-07-31): no undo (no "Unassign
    # Blocker") -- once committed, a blocker stays committed until Done. Real
    # Magic's declare-blockers is a single simultaneous action (509.2); this
    # engine linearizes it into one-creature-at-a-time picks with no way
    # back, by design, across the whole engine (no action anywhere lets the
    # agent reconsider/reverse an earlier commitment -- see
    # todo/no_undo_policy.md for the broader rationale).
    # An "Unassign Blocker" action would let assign/unassign cycle
    # indefinitely with turn_number never advancing (turn.py's own
    # PRIORITY_ROUND_ACTION_CAP docstring: tens of thousands of iterations in
    # one boggles_mirror evaluation) -- the no-undo rule is what forecloses
    # that pathology.
    for name in attackable_names:
        max_slot = qty_by_name.get(name, game.TOKEN_LIMIT)
        for slot in range(1, max_slot + 1):
            actions.append((
                f"Assign Blocker: {name} (slot {slot})",
                _assign_blocker_legal(name, slot),
                _assign_blocker_execute(name, slot),
            ))
    actions.append(("Done blocking", _done_blocking_legal, _done_blocking_execute))
    # Trample's "assign a combat-damage point to the defending player" half
    # of a gang-blocking damage assignment (the blocker half is the pointer
    # scheme). One fixed action, runtime-gated to a trampling attacker mid-
    # assign_combat_damage -- masked illegal otherwise.
    actions.append((
        "Assign combat damage to opponent", _assign_damage_to_opponent_legal, _assign_damage_to_opponent_execute,
    ))

    # "Choose opponent's: X (slot k)" -- the general cross-player
    # targeting primitive, one per (opponent name, slot), built from the
    # OPPONENT's own decklist/tokens instead of this side's own. Registered
    # for every opponent choosable name, not just creatures -- same
    # "pre-register broadly, mask precisely" pattern as "Choose target: X"
    # above: blocking's predicate only ever matches attackers (creatures),
    # but Masked Vandal's ETB (green_cards.py) targets an opponent artifact
    # or enchantment, so a creature-only filter here left that a legal
    # target with no matching action row -- an all-False mask crash the
    # instant an opponent controlled a targetable artifact/enchantment.
    # Same quantity-or-TOKEN_LIMIT bound as the attack registration above,
    # just applied to the other side's card pool. None/() (the default for
    # every 1-player deck) registers nothing at all -- there's no real
    # opponent battlefield to ever reference in that mode.
    if opponent_decklist is not None:
        opponent_distinct_names = sorted({name for name, *_rest in opponent_decklist})
        opponent_qty_by_name = {name: qty for name, qty, *_rest in opponent_decklist}
        opponent_choosable_names = sorted(
            set(opponent_distinct_names) | {cd.name for cd in opponent_token_card_defs}
        )
        for name in opponent_choosable_names:
            max_slot = opponent_qty_by_name.get(name, game.TOKEN_LIMIT)
            for slot in range(1, max_slot + 1):
                actions.append((
                    f"Choose opponent's: {name} (slot {slot})",
                    _choose_opponent_permanent_legal(name, slot),
                    _choose_opponent_permanent_execute(name, slot),
                ))

    # Float-first: no "Choose: X as color" tap-during-payment rows. A flexible
    # or granted source's color is now chosen at TAP time (the "Tap X for
    # <color>" mana-ability rows above float directly into the pool); paying a
    # cost only ever spends floated pool mana (below).
    for color in game.POOL_COLORS:
        actions.append((
            f"Spend {color} from pool",
            _pool_spend_legal(color),
            _pool_spend_execute(color),
        ))

    # ---- UNIVERSAL DECISION ROWS (deliberately NOT gated on any pending kind) ----
    # Every fixed row below answers a QUESTION the engine can pose, and each is a
    # small constant set of buttons (never a per-card-name loop). They are added
    # to EVERY deck's table unconditionally, because a decision is not always
    # answered by the player whose deck produced it -- an opponent's card can pose
    # a question that YOU must answer:
    #   pay_unless          Spell Pierce/Ward/Nihil Spellbomb/Chain Lightning --
    #                       the PAYER is the spell's controller, i.e. the opponent.
    #   tuck_position       Deem Inferior -- the tucked permanent's OWNER chooses.
    #   may_copy            Chain Lightning -- the affected player may copy.
    #   choose_any_target   a Chain Lightning COPIER may choose a new target.
    #   choose_target_player/scry/search_fetch  Undercity's own Trap/Lost Well/
    #                       Secret Entrance rooms -- initiative can be STOLEN
    #                       (combat damage), so a deck with none of these kinds
    #                       among its own cards can still be the one venturing
    #                       and facing them; search_fetch's Decline is ALSO
    #                       independently reachable via Cleansing Wildfire (the
    #                       DESTROYED land's controller searches their own
    #                       library, not the caster).
    #   discard-decline, choose_graveyard_card-decline  same shape (Refurbished
    #                       Familiar / Relic of Progenitus target the opponent).
    # `derive_pending_kinds` reads a deck's OWN cards, so gating these on it left
    # the answering seat with no row for the question and an all-False action mask
    # -- a hard crash (real: monster_tron/mono_blue_terror league play; an
    # empirical cross-deck audit reproduced pay_unless and tuck_position and
    # flagged choose_graveyard_card/discard/surveil as latent). Rather than
    # maintain a hand-audited list of "which kinds may cross" -- which silently
    # rots the next time a card hands a choice to the opponent -- every constant
    # decision-button set whose kind COULD ever cross players is always present
    # and runtime-gated illegal, exactly the "present-but-permanently-illegal"
    # footing "Decline (graveyard)" and "Decline (search)" already documented
    # for themselves.
    #
    # Gated below on own_pending_kinds instead (confirmed, not assumed, self-
    # only -- each has exactly one call site and it's always the deciding
    # player's own card): may_transform (Delver of Secrets' own upkeep --
    # mono_blue_terror only), may_cast (Cascade -- monster_tron only),
    # choose_mana_color (Chromatic Star -- grixis_affinity only),
    # ancient_stirrings/malevolent_rumble/select_to_hand (each one specific
    # card's own search/rummage), madness_decision (madness always triggers
    # off YOUR OWN discarded card, whoever forced the discard), and
    # discard_or_sacrifice (Highway Robbery's sacrifice trigger is the
    # CASTER's own optional cost, never posed to an opponent -- also true of
    # `impulse` just above, for the identical reason). Making any of these
    # universal would have repeated the exact bloat impulse's own history
    # already illustrates: real rows for a card the deck never runs.
    actions.append(("Keep (scry/surveil)", _keep_dispose_legal, _keep_execute))
    actions.append(("Dispose (scry/surveil)", _keep_dispose_legal, _dispose_execute))
    # Self-only (Ancient Stirrings/Malevolent Rumble are each one specific
    # card's own search/rummage) -- see this block's own header comment.
    if "ancient_stirrings" in own_pending_kinds:
        actions.append(("Decline (Ancient Stirrings)", _decline_legal, _decline_execute))
    if "malevolent_rumble" in own_pending_kinds:
        actions.append(("Decline (Malevolent Rumble)", _decline_malevolent_rumble_legal, _decline_malevolent_rumble_execute))
    actions.append(("Shuffle (Ponder)", _ponder_shuffle_legal, _ponder_shuffle_execute))
    actions.append(("Pay (unless)", _pay_unless_pay_legal, _pay_unless_pay_execute))
    actions.append(("Don't pay (unless)", _pay_unless_decline_legal, _pay_unless_decline_execute))
    actions.append(("Tuck: 2nd from top", _tuck_position_legal, lambda state: game.execute_tuck_position(state, "top2")))
    actions.append(("Tuck: bottom", _tuck_position_legal, lambda state: game.execute_tuck_position(state, "bottom")))
    # Self-only (Delver of Secrets' own upkeep trigger -- only its own
    # controller ever faces this) -- see this block's own header comment.
    if "may_transform" in own_pending_kinds:
        actions.append(("Transform", _may_transform_legal, lambda state: game.execute_may_transform(state, True)))
        actions.append(("Don't transform", _may_transform_legal, lambda state: game.execute_may_transform(state, False)))
    actions.append(("Copy spell", _may_copy_legal, lambda state: game.execute_may_copy(state, True)))
    actions.append(("Don't copy spell", _may_copy_legal, lambda state: game.execute_may_copy(state, False)))
    # Self-only (Cascade's may-cast of the hit is always the CASCADING
    # player's own choice) -- see this block's own header comment.
    if "may_cast" in own_pending_kinds:
        actions.append(("Cast (may)", _may_cast_legal, lambda state: game.execute_may_cast(state, True)))
        actions.append(("Decline (may)", _may_cast_legal, lambda state: game.execute_may_cast(state, False)))
    # Self-only (Lead the Stampede's own reveal-and-pick) -- see this block's
    # own header comment.
    if "select_to_hand" in own_pending_kinds:
        actions.append(("Keep (select to hand)", _select_to_hand_keep_legal, _select_to_hand_keep_execute))
        actions.append(("Bottom (select to hand)", _select_to_hand_bottom_legal, _select_to_hand_bottom_execute))
    actions.append(("Decline (search)", _decline_search_legal, _decline_search_execute))
    # Only ever legal for an OPTIONAL choose_graveyard_card (Masked Vandal's "you
    # may exile a creature from your graveyard"); the legal_fn itself gates on
    # pending["optional"], so it is present-but-permanently-illegal for decks
    # whose only graveyard picks are mandatory (Dread Return, Relic). Universal
    # per the block above -- Relic of Progenitus already poses its graveyard pick
    # to the TARGETED player, so the decline row has to exist in their table too
    # the day any cross-player graveyard pick becomes optional.
    actions.append(("Decline (graveyard)", _decline_graveyard_card_legal, _decline_graveyard_card_execute))
    # Float-first: NO "Abandon payment" -- there is no undo. Paying a cost is
    # just spending already-floated pool mana (the _pool_spend rows below), and
    # affordability was checked exactly (game.mana.pool_can_pay) before the
    # payment began, so spending alone can never strand. With no such action to
    # take, there is no tap/untap churn cycle for a dense reward penalty to
    # ever need to fight.
    #
    # That "cannot strand" guarantee is NOT self-enforcing, though: because
    # there is no undo, anything that DESTROYS floating mana mid-payment can
    # make an already-begun payment impossible to finish, and with every cast/
    # activate action and Pass illegal during a pending, the agent is left with
    # no legal action at all (an all-False mask). A mana FILTER is exactly such
    # an action -- pool->pool conversion, legal mid-payment -- and a real
    # pretrain run hit it (monster_tron: filter away the only {G} while owing
    # {G}). _filter_would_strand_payment now upholds the guarantee for that
    # path; any FUTURE action that consumes floating mana must do the same.
    # NOTE: the pregame mulligan actions ("Keep hand" / "Mulligan") and the
    # "mulligan_bottom" branch of the generic "Choose: X" action have no rows in
    # this table. The per-deck MulliganNet (rl.mulligan) owns every pregame
    # decision -- rl.agent.SeatAgent intercepts the pregame pending kinds before
    # the main net's forward -- so the main policy's action space contains ZERO
    # pregame actions and a game can never fall back to a fixed-table mulligan.
    # The _mulligan_*_legal/_execute helpers remain exported for the engine's own
    # use, not wired into any table.
    # Refurbished Familiar already makes the OPPONENT discard from their own hand
    # (the "Choose: X" hand-name rows are their own, so those self-serve) -- but
    # the decline row is a constant button, so it is universal per the block above.
    actions.append(("Decline (discard)", _decline_discard_legal, _decline_discard_execute))
    if "discard_or_sacrifice" in own_pending_kinds:
        # Self-only: Highway Robbery's "discard a card or sacrifice a land"
        # is the CASTER's own optional cost, never posed to an opponent --
        # see this block's own header comment. The DISCARD half reuses the
        # generic "Choose: X" action built above (bare hand-card names); the
        # SACRIFICE half is a single trigger action that opens its own
        # nested choose_permanent pointer choice for WHICH exact permanent
        # to sacrifice -- see _discard_or_sacrifice_trigger_sacrifice_legal's
        # own docstring for why this isn't a per-land-name loop anymore.
        actions.append((
            "Sacrifice a permanent (discard or sacrifice)",
            _discard_or_sacrifice_trigger_sacrifice_legal,
            _discard_or_sacrifice_trigger_sacrifice_execute,
        ))
        # Same self-only gate as the trigger action just above -- a decline
        # button for a resolution kind that can never open has nothing to
        # decline.
        actions.append((
            "Decline (discard or sacrifice)",
            _decline_discard_or_sacrifice_legal,
            _decline_discard_or_sacrifice_execute,
        ))
    # Self-only (madness always triggers off YOUR OWN discarded card,
    # whoever forced the discard) -- see this block's own header comment.
    if "madness_decision" in own_pending_kinds:
        actions.append(("Cast (madness)", _madness_cast_legal, _madness_cast_execute))
        actions.append(("Decline (madness)", _madness_decline_legal, _madness_decline_execute))
    # "Target: yourself" is always legal the instant this pending kind is reached
    # (a real Magic legality fact -- "target player" never excludes its own
    # caster), even alone in a 1-player game; "Target: opponent" only becomes
    # legal once a real second PlayerState exists. Two fixed actions, not a
    # per-name loop -- there are only ever at most 2 possible players.
    actions.append(("Target: yourself", _target_self_legal, _target_self_execute))
    actions.append(("Target: opponent", _target_opponent_legal, _target_opponent_execute))
    # The player half of an "any target" choice (Lightning Bolt etc.) -- same
    # shape/reasoning as the choose_target_player pair above. The creature half
    # (either battlefield) rides the identity pointer scheme (rl.action_bridge),
    # not fixed actions. Universal because a Chain Lightning COPIER -- the
    # affected player, i.e. the opponent -- may choose a new target for the copy.
    actions.append(("Target any: yourself", _target_any_self_legal, _target_any_self_execute))
    actions.append(("Target any: opponent", _target_any_opponent_legal, _target_any_opponent_execute))
    actions.append(("Choose no target", _target_any_decline_legal, _target_any_decline_execute))  # "up to one" decline
    # Undercity venture (which next room -- a <=2-way branch): a tiny CONSTANT
    # set (ROOM_NAMES), not a per-deck-card loop, so it's universal -- initiative
    # can pass between players (combat damage), so a deck with no Undercity card
    # of its own can still be the one venturing.
    for room in game.ROOM_NAMES:
        actions.append((f"Enter room: {room}", _choose_room_legal(room), _choose_room_execute(room)))
    # Chromatic Star's "add one mana of any color": self-only (the activating
    # player's own choice, single call site) -- see this block's own header
    # comment. Gated, unlike the room set above.
    if "choose_mana_color" in own_pending_kinds:
        for color in game.COLORS:
            actions.append((f"Add mana: {color}", _choose_mana_color_legal(color), _choose_mana_color_execute(color)))

    # choose_cast_mode/choose_cast_x/choose_delve_amount: shared, per-deck-
    # conditional button sets for the generic mode/X/delve-amount sub-
    # decisions the cast_modes/x_cast_modes/delve loops above open -- unlike
    # the universal rows just above, these are never posed by an OPPONENT's
    # card (they only ever fire mid this deck's own casting), so they're
    # gated on this deck actually having a card of that shape, same
    # reasoning as the deck-gated impulse/discard_or_sacrifice rows.
    if needs_cast_mode_buttons:
        for i in range(_CAST_MODE_BUTTON_MAX):
            actions.append((f"Mode {i + 1}", _choose_cast_mode_legal(i), _choose_cast_mode_execute(i)))
    if needs_cast_x_buttons:
        for x in range(_CAST_X_BUTTON_MAX + 1):
            actions.append((f"X={x}", _choose_cast_x_legal(x), _choose_cast_x_execute(x)))
    if needs_delve_buttons:
        for n in range(_DELVE_BUTTON_MAX + 1):
            actions.append((f"Delve {n}", _choose_delve_amount_legal(n), _choose_delve_amount_execute(n)))

    # Shared "Produce <color>" buttons for the choose_color stage of a
    # mana_subdecision -- reused by BOTH Saruli-shaped mana_extra_choose
    # sources and filter_mana sources (whichever one is present sets
    # needs_mana_subdecision_color_buttons above); never posed by an
    # opponent's card, only ever opened by this deck's own source.
    if needs_mana_subdecision_color_buttons:
        for color in game.COLORS:
            actions.append((
                f"Produce {color}", _mana_subdecision_color_legal(color), _mana_subdecision_color_execute(color),
            ))

    return tuple(actions)


_battlefield_lookup_cache = None  # (state, {(name, slot): Permanent}) -- valid only for the duration of one legal_action_mask sweep, same lifecycle as _mana_ability_options_cache below


def _cached_battlefield_lookup(state):
    """Sweep-scoped {(name, slot): Permanent} lookup for state.battlefield --
    same "profiled, not guessed" caching pattern as _cached_mana_ability_options
    just below: _attack_legal/
    _assign_blocker_legal each independently scanned the WHOLE battlefield
    with any(...) to find one specific (name, slot), once per action-table
    entry -- for a deck with many creature copies (boggles' Auras/tokens)
    that's O(action_table_size x battlefield_size) repeated work every
    sweep (profiled: 2 closures alone accounted for ~3.4M calls across a
    single 8192-step training burst). Building this dict once per sweep
    turns each of those checks into an O(1) lookup. Safe for the same
    reason _cached_mana_ability_options is: a legal_action_mask sweep only
    ever calls legal_fns, never an execute_* function, so state can't change
    mid-sweep. (name, slot) is a safe dict key here because it is unique
    per side -- state.battlefield is always ONE side's own,
    active-relative zone (see this module's other active-relative
    docstrings), never two players' permanents mixed in one sweep."""
    global _battlefield_lookup_cache
    if _battlefield_lookup_cache is None or _battlefield_lookup_cache[0] is not state:
        _battlefield_lookup_cache = (state, {(p.card_def.name, p.slot): p for p in state.battlefield})
    return _battlefield_lookup_cache[1]




_mana_ability_options_cache = None  # (state, result) -- one legal_action_mask sweep, same lifecycle as _battlefield_lookup_cache above


def _cached_mana_ability_options(state):
    """Memoizes game.mana_ability_options(state) for one legal_action_mask
    sweep -- every "Tap X for <color>" row's legal() calls it, so a sweep would
    otherwise recompute the identical battlefield scan once per mana row."""
    global _mana_ability_options_cache
    if _mana_ability_options_cache is None or _mana_ability_options_cache[0] is not state:
        _mana_ability_options_cache = (state, game.mana_ability_options(state))
    return _mana_ability_options_cache[1]


def _mana_ability_legal(name, color):
    """Float-first: a mana ability is legal in ANY priority window, even
    mid-resolution of anything else (605.1a/605.3b -- it never uses the
    stack and doesn't require priority to activate), so this has no
    pending-resolution gate at all -- no `_pending_gate` attribute means
    legal_action_mask always calls it, regardless of what's pending. Legal
    iff a source named `name` can produce `color` right now
    (game.mana_ability_options; color=None for fixed/tron/count sources)."""
    def legal(state):
        return (name, color) in _cached_mana_ability_options(state)
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


_mana_source_cache = None  # (state, {(name, color): Permanent or None}) -- one legal_action_mask sweep, same lifecycle as _battlefield_lookup_cache above


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
    identically to a real decklist name."""
    def legal(state):
        p = _find_mana_extra_source(state, name)
        if p is None:
            return False
        extra_pred = game.EFFECT_REGISTRY.get(p.card_def.effect_id, {}).get("mana_extra_choose")
        if extra_pred is None:
            return False
        return any(
            q is not p and not q.tapped and extra_pred(q)
            for q in state.battlefield
        )
    return legal


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
    binds). Generic: this function has no idea which ability is asking."""
    def legal(state):
        return state.mana_subdecision["can_produce"](state, color)
    legal._mana_subdecision_gate = "choose_color"
    return legal


def _mana_subdecision_color_execute(color):
    def execute(state):
        game.execute_mana_subdecision_color(state, color)
    return execute


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


_filter_source_cache = None  # (state, {name: Permanent or None}) -- one legal_action_mask sweep, same lifecycle as _battlefield_lookup_cache above


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
    gate, per 605.1a. Legal iff an unused filter named `name` exists and
    the pool already holds a floating `input_color` pip (any color,
    including colorless, can pay a generic cost) -- AND, mid-payment,
    converting that pip away must not strand the payment (see
    _filter_would_strand_payment). Uses the sweep-scoped
    _cached_filter_source cache, not a fresh scan -- this legal() runs once
    per (source, input_color) row, POOL_COLORS-many per source now instead
    of POOL_COLORS x len(colors)."""
    def legal(state):
        if _cached_filter_source(state, name) is None or state.mana_pool.get(input_color, 0) <= 0:
            return False
        return not _filter_would_strand_payment(state, input_color)
    return legal


def _filter_would_strand_payment(state, input_color):
    """Would converting one `input_color` pip away right now make an ALREADY-
    BEGUN payment -- or one about to begin the instant the current choice
    resolves -- impossible to finish?

    Float-first's own design guarantee is that a payment, once begun, can always
    be completed -- it deliberately removed the "Abandon payment" action (which
    existed to escape exactly this), so a payment the agent cannot finish is
    unescapable BY CONSTRUCTION: every cast/activate action is illegal while a
    pending is open, Pass is illegal, and the only remaining actions are
    spending pool mana and tapping sources. That guarantee was never enforced
    for filters, and a real pretrain run found the hole (monster_tron, turn 10:
    tap Forest for the only {G}, cast Crop Rotation for {G}, then filter that
    same {G} into {U} -- remaining {'G': 1}, no untapped green source, all-False
    action mask, RuntimeError).

    A SECOND, later hole in the same guarantee: choose_cast_copy (WHICH
    graveyard copy to cast/activate, MTG 601.2a) sits BETWEEN the moment a
    flashback/escape/graveyard-ability action is chosen (its own legality
    already required plan_payment to confirm the pool covers the cost) and the
    begin_pay_cost that actually opens once a copy is picked -- and mana
    abilities/filters stay legal "in ANY priority window" (605.1a/605.3b)
    during that gap too, same as mid-pay_cost. So filtering away the very pip
    the not-yet-open payment is guaranteed to need is exactly as breaking here
    as it is once pay_cost has already begun -- confirmed live (monster_tron,
    turn 44: 2 Bramble Wurms in the graveyard, activated the {2}{G} graveyard
    ability, filtered the floating {G} away while still choosing WHICH copy;
    pay_cost then opened already unpayable, all-False mask). game.resolution.
    handlers.begin_choose_cast_copy's own docstring covers why only this one
    pointer pending needs this (reserved_cost, stashed on the pending by
    whichever registry-driven execute closure already knows the upcoming cost
    in full -- drl_env._actions._graveyard_ability_execute/_flashback_execute).

    Only COLORED requirements can break. A filter is one pip in, one pip out, so
    the pool's SIZE is invariant and any outstanding generic requirement stays
    exactly as payable as before (any color pays generic) -- so this checks only
    whether the pool would still hold enough `input_color` for what the cost
    specifically demands in that color. Exact, not a heuristic: it permits every
    conversion that leaves the payment completable from the pool (including
    converting a spare copy of a needed color) and rejects only those that
    provably break it. Pool-only, matching game.mana.pool_can_pay -- the same
    domain float-first already uses for affordability everywhere else.

    A conservative edge, deliberately accepted: it does not additionally reason
    about mana still tappable from untapped sources, so a line like "convert my
    only G to U, then tap a second Forest for G" is refused. That line is never
    NEEDED -- pool_can_pay already required the full cost to be floating when
    the payment began (or, for the choose_cast_copy case, when the flashback/
    activate action was chosen), so converting away a still-demanded pip can
    only ever reduce sufficiency."""
    pending = state.pending_resolution
    if pending is None:
        return False
    if pending["kind"] == "pay_cost":
        still_needed = pending["remaining"].get(input_color, 0)
    elif pending["kind"] == "choose_cast_copy" and pending.get("reserved_cost") is not None:
        still_needed = pending["reserved_cost"].get(input_color, 0)
    else:
        return False
    return state.mana_pool.get(input_color, 0) - 1 < still_needed


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
        state.mana_pool[input_color] -= 1
        if state.mana_pool[input_color] <= 0:
            del state.mana_pool[input_color]
        state.log_event("mana_spend", color=input_color, toward="filter")

        colors = game.EFFECT_REGISTRY[p.card_def.effect_id]["filter_mana"]["colors"]

        def can_produce(state, color):
            return color in colors

        def on_choose_color(state, color):
            state.mana_pool[color] = state.mana_pool.get(color, 0) + 1
            state.log_event("mana_tap", permanent=(name, p.slot), mode="filter", produced=[color])

        game.begin_mana_color_choice(state, can_produce, on_choose_color)
    return execute


def _validate_choose_name_coverage(state, actions):
    """The durable, always-on half of the by-name/cross-player guardrail
    (see _CHOOSE_NAME_PENDING_KINDS's own docstring for the invariant this
    enforces). Runs on every legal_action_mask sweep, but is a cheap no-op
    unless the current pending is actually one of that constant's kinds --
    which is already a small minority of decisions.

    Cross-checks _choose_name_options' own OUTPUT against what this deck's
    table can actually represent (every "Choose: X" row already built into
    `actions`, read directly off it -- no separate choosable_names threading
    needed). If a candidate has no matching row, that candidate is
    UNREACHABLE: not merely "illegal right now" (any legal_fn can say that),
    but a card the agent has NO WAY to ever select for this decision, which
    means either every candidate is unreachable (an eventual all-False
    crash, the failure mode that surfaced this bug) or only SOME are (a
    silent, still-live correctness bug -- the agent quietly loses the
    ability to choose a specific legal option, with no crash at all to flag
    it). Raising immediately, the first time ANY game -- a self-check,
    random self-play, or real training -- exercises the violation, is what
    makes this a property of the code instead of a fact someone has to
    remember: no comment to read, no separate audit script to run and act
    on. Confirmed reachable in real cross-deck league play: choose_
    stack_target asked mono_blue_terror to name a spell dmir_terror cast,
    which mono_blue_terror's own table (built only from its own decklist)
    had no row for."""
    pending = state.pending_resolution
    if pending is None or pending["kind"] not in _CHOOSE_NAME_PENDING_KINDS:
        return
    candidates = _choose_name_options(state)
    if not candidates:
        return
    table_names = {name[len("Choose: "):] for name, _legal, _execute in actions if name.startswith("Choose: ")}
    missing = set(candidates) - table_names
    if missing:
        raise RuntimeError(
            f"_choose_name_options returned {sorted(missing)} for pending kind {pending['kind']!r}, but this "
            f"deck's own action table has no 'Choose: X' row for {'it' if len(missing) == 1 else 'them'}. "
            f"A by-name pending kind can only ever be answered from names in the DECIDING PLAYER's own deck "
            f"(build_action_table's choosable_names) -- {pending['kind']!r} can apparently produce a candidate "
            f"outside that (most likely another player's card), which means it must be POINTER-addressed "
            f"(see rl.action_bridge; choose_graveyard_card and choose_stack_target are the established pattern), "
            f"not left in _CHOOSE_NAME_PENDING_KINDS."
        )


_mana_subdecision_rows_cache = {}  # id(actions) -> (actions, [(idx, mana_subdecision_gate), ...] for entries stamped with one)
_MANA_SUBDECISION_ROWS_CACHE_CAP = 32  # ponytail: FIFO-evicted, not reset per sweep like its siblings (this cache must survive across sweeps -- see below) -- bounds memory when a caller (e.g. a persistent multiprocessing worker) rebuilds fresh `actions` tables across many calls instead of reusing one per deck for the session


def _mana_subdecision_rows(actions):
    """The (idx, _mana_subdecision_gate) pairs for entries carrying that
    stamp, cached once per distinct `actions` table (see legal_action_mask's
    own docstring for the id()-cache lifecycle reasoning -- identical to
    the removed _action_gates' -- this is the ONE gate still worth caching:
    _mana_subdecision_color_legal is unsafe to call unconditionally (see
    its own body -- it dereferences state.mana_subdecision["can_produce"]
    with no None-guard, so calling it with no subdecision open raises
    TypeError, not a graceful False), so legal_action_mask MUST know which
    few indices those are without calling them first to find out. Empty for
    most decks -- populated only by a Saruli-Caretaker-shaped
    mana_extra_choose card or a filter_mana card (Conduit Pylons/Barrels of
    Blasting Jelly)."""
    cached = _mana_subdecision_rows_cache.get(id(actions))
    if cached is not None and cached[0] is actions:
        return cached[1]
    rows = [(idx, legal_fn._mana_subdecision_gate) for idx, (_name, legal_fn, _execute) in enumerate(actions)
            if getattr(legal_fn, "_mana_subdecision_gate", None) is not None]
    if len(_mana_subdecision_rows_cache) >= _MANA_SUBDECISION_ROWS_CACHE_CAP:
        _mana_subdecision_rows_cache.pop(next(iter(_mana_subdecision_rows_cache)))
    _mana_subdecision_rows_cache[id(actions)] = (actions, rows)
    return rows


def legal_action_mask(state, actions):
    """Stateless -- takes the action table explicitly, so any caller (the
    token pipeline's own _seat_step, a direct game-loop driver, ...)
    can use it. `actions` is any table built by build_action_table -- every
    deck's own table, none privileged as a default (a caller with its own
    decklist always has its own table to pass).

    Calls every closure directly -- no pending_resolution-based category
    gating. A gating layer (skip a closure via a cheap pre-check on
    state.pending_resolution["kind"], mirroring each closure's own first-
    line check -- see every `._pending_gate =` stamp still sitting on the
    closures below) was tried, INCLUDING a per-table cache of the gate
    lookups (avoiding a fresh getattr every sweep), and measured SLOWER
    than calling everything unconditionally -- consistently, ~15-25%,
    across every pending kind that occurs in real play (controlled, order-
    randomized timing over 4000 real captured decisions, comparing on the
    SAME states in interleaved repeats to rule out cache-warmup bias).
    Most `_X_legal` closures are already cheap enough (one or two
    attribute/dict checks) that the overhead of DECIDING whether to call
    them costs as much or more than just calling them. The `._pending_gate`
    stamps are left in place as documentation of each closure's real
    legality precondition (and in case a more selective future approach --
    gating only the handful of genuinely expensive closures instead of all
    ~300 -- turns out to be worth it); this sweep no longer reads them.

    state.mana_subdecision is NOT the same kind of optional speed layer,
    and is NOT dropped: it's the ONLY thing that makes every OTHER action
    illegal while a gate-free mana ability's own multi-step choice (Saruli
    Caretaker: tap another creature, then choose a color) is open -- no
    individual `_X_legal` closure checks state.mana_subdecision itself
    except `_mana_subdecision_color_legal` (gated to its own "choose_color"
    stage, and unsafe to call at all when no subdecision is open -- see
    _mana_subdecision_rows' own docstring). Skipping this check would
    silently make every other action look legal mid-subdecision (or crash
    on the "Produce <color>" rows specifically); see
    test_saruli_caretaker_two_stage_mana_subdecision for the regression
    coverage. Three cases, cheapest first:
      1. This deck's table has no mana_subdecision-gated row at all (most
         decks) -- every closure is safe to call unconditionally, so this
         degenerates to exactly the same zero-overhead sweep as case 1
         above, no per-entry check of any kind.
      2. The table has such a row, but none is open right now -- everything
         else is still safe to call directly; only the (few, known-by-index)
         subdecision rows are forced False without being called.
      3. A subdecision is genuinely open -- exclusive priority, suppress
         everything except the matching-stage subdecision closures, mirroring
         how activating a mana ability is atomic from everyone else's view
         in real Magic. See state.mana_subdecision's own docstring for why
         this can't just be another pending_resolution kind (a single slot,
         not a stack -- a second pending would clobber whatever's already
         open).

    Resets _battlefield_lookup_cache, _mana_ability_options_cache,
    _mana_source_cache, _filter_source_cache, and game.mana's own
    _enchanting_cache (game.reset_mana_cache) before AND after the sweep
    itself (not just before): guarantees none of these caches can ever leak
    past this call's own scope into a later execute_fn call or an unrelated
    sweep against a different/mutated state, even though nothing in the
    current single-threaded, synchronous call pattern would actually trigger
    that -- belt-and-suspenders for a module-level global, not load-bearing.
    mana.py's own cache is reset from here, not self-invalidating there, for
    the same reason the others aren't: see game.mana._enchanting's own
    docstring.

    Mask is built as a plain list and converted to a numpy array ONCE at
    the end -- indexed numpy writes in a tight Python loop carry real
    per-call overhead, the same lesson rl.arch.pad_token_batch's own
    docstring already applies to token tensors."""
    global _battlefield_lookup_cache, _mana_ability_options_cache, _mana_source_cache, _filter_source_cache
    _battlefield_lookup_cache = None
    _mana_ability_options_cache = None
    _mana_source_cache = None
    _filter_source_cache = None
    game.reset_mana_cache()
    mana_sub = state.mana_subdecision
    sub_rows = _mana_subdecision_rows(actions)
    try:
        if not sub_rows:
            mask = [legal_fn(state) for _name, legal_fn, _execute in actions]
        elif mana_sub is None:
            sub_indices = {idx for idx, _gate in sub_rows}
            mask = [False if idx in sub_indices else legal_fn(state)
                    for idx, (_name, legal_fn, _execute) in enumerate(actions)]
        else:
            mana_sub_stage = mana_sub["stage"]
            mask = [False] * len(actions)
            for idx, gate in sub_rows:
                if gate == mana_sub_stage:
                    mask[idx] = actions[idx][1](state)
        _validate_choose_name_coverage(state, actions)
        return np.asarray(mask, dtype=bool)
    finally:
        _battlefield_lookup_cache = None
        _mana_ability_options_cache = None
        _mana_source_cache = None
        _filter_source_cache = None
        game.reset_mana_cache()


__all__ = [
    '_cast_speed',
    '_GATE_NO_PENDING',
    '_land_drop_legal',
    '_land_drop_execute',
    '_hand_count_available',
    '_effective_cast_cost',
    '_cast_legal',
    '_cast_execute',
    '_precast_choice_execute',
    '_x_cast_legal',
    '_x_cast_execute',
    '_x_precast_choice_execute',
    '_delve_reduced_cost',
    '_delve_legal',
    '_delve_execute',
    '_choose_delve_amount_legal',
    '_choose_delve_amount_execute',
    '_modal_legal',
    '_modal_execute',
    '_x_modal_legal',
    '_x_modal_execute',
    '_choose_cast_mode_legal',
    '_choose_cast_mode_execute',
    '_choose_cast_x_legal',
    '_choose_cast_x_execute',
    '_tuck_position_legal',
    '_activate_legal',
    '_activate_execute',
    '_forestcycle_legal',
    '_forestcycle_execute',
    '_graveyard_ability_legal',
    '_graveyard_ability_execute',
    '_pass_legal',
    '_pass_execute',
    '_choose_name_options',
    '_choose_name_legal',
    '_choose_name_execute',
    '_attack_legal',
    '_attack_execute',
    '_choose_permanent_legal',
    '_choose_permanent_execute',
    '_choose_opponent_permanent_legal',
    '_choose_opponent_permanent_execute',
    '_assign_blocker_legal',
    '_assign_blocker_execute',
    '_done_blocking_legal',
    '_done_blocking_execute',
    '_assign_damage_to_opponent_legal',
    '_assign_damage_to_opponent_execute',
    '_pool_spend_legal',
    '_pool_spend_execute',
    '_keep_dispose_legal',
    '_keep_execute',
    '_dispose_execute',
    '_decline_legal',
    '_decline_execute',
    '_decline_malevolent_rumble_legal',
    '_decline_malevolent_rumble_execute',
    '_ponder_shuffle_legal',
    '_ponder_shuffle_execute',
    '_pay_unless_pay_legal',
    '_pay_unless_pay_execute',
    '_pay_unless_decline_legal',
    '_pay_unless_decline_execute',
    '_may_transform_legal',
    '_may_copy_legal',
    '_may_cast_legal',
    '_choose_room_legal',
    '_choose_room_execute',
    '_choose_mana_color_legal',
    '_choose_mana_color_execute',
    '_find_mana_extra_source',
    '_mana_extra_choose_legal',
    '_mana_extra_choose_execute',
    '_mana_subdecision_color_legal',
    '_mana_subdecision_color_execute',
    '_select_to_hand_keep_legal',
    '_select_to_hand_bottom_legal',
    '_select_to_hand_keep_execute',
    '_select_to_hand_bottom_execute',
    '_decline_search_legal',
    '_decline_search_execute',
    '_decline_graveyard_card_legal',
    '_decline_graveyard_card_execute',
    '_decline_discard_legal',
    '_decline_discard_execute',
    '_target_self_legal',
    '_target_self_execute',
    '_target_opponent_legal',
    '_target_opponent_execute',
    '_target_any_self_legal',
    '_target_any_self_execute',
    '_target_any_opponent_legal',
    '_target_any_opponent_execute',
    '_target_any_decline_legal',
    '_target_any_decline_execute',
    '_discard_or_sacrifice_trigger_sacrifice_legal',
    '_discard_or_sacrifice_trigger_sacrifice_execute',
    '_decline_discard_or_sacrifice_legal',
    '_decline_discard_or_sacrifice_execute',
    '_madness_cast_legal',
    '_madness_cast_execute',
    '_madness_decline_legal',
    '_madness_decline_execute',
    '_activate_no_cost_legal',
    '_activate_no_cost_execute',
    '_alt_cast_legal',
    '_alt_cast_execute',
    '_flashback_legal',
    '_flashback_execute',
    '_impulse_entry',
    '_play_impulse_land_legal',
    '_play_impulse_land_execute',
    '_play_impulse_cast_legal',
    '_play_impulse_cast_execute',
    '_plot_legal',
    '_plot_execute',
    '_cast_from_exile_legal',
    '_cast_from_exile_execute',
    '_omen_cast_legal',
    '_omen_cast_execute',
    'build_action_table',
    '_battlefield_lookup_cache',
    '_cached_battlefield_lookup',
    'legal_action_mask',
]
