"""Casting/activating/playing a card from a zone: table categories A (Play
land), B's plain/modal/X-cost/Delve hand cast, C (Activate), and D
(Forestcycle/Cycle) from drl_env._actions_table's own module docstring, plus
impulse ("play from exile"). The rest of category B -- alt-cost/Flashback/
Escape/Plot/Omen/Prototype, everything that casts from a NON-hand zone or
for a non-default cost -- lives in the sibling drl_env._actions_cast_altzone
(see its own module docstring for why it's split out). Each is a
legal(state)/execute(state) factory pair build_action_table (drl_env.
_actions_table) calls once per matching card."""

import game

from ._actions_cast_altzone import _with_chosen_copy
from ._actions_common import _GATE_NO_PENDING, _hand_count_available

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
    distinct from card_def.cast_cost" shape _actions_cast_altzone._plot_legal/
    _omen_cast_legal already use for their own alternate costs, not a param
    bolted onto _cast_legal itself."""
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
    the interned CardDef) -- see _actions_cast_altzone._graveyard_instance.
    The cost itself is read
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


__all__ = [
    '_cast_speed',
    '_CAST_MODE_BUTTON_MAX',
    '_CAST_X_BUTTON_MAX',
    '_DELVE_BUTTON_MAX',
    '_land_drop_legal',
    '_land_drop_execute',
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
    '_activate_legal',
    '_activate_execute',
    '_forestcycle_legal',
    '_forestcycle_execute',
    '_graveyard_ability_legal',
    '_graveyard_ability_execute',
    '_activate_no_cost_legal',
    '_activate_no_cost_execute',
    '_impulse_entry',
    '_play_impulse_land_legal',
    '_play_impulse_land_execute',
    '_play_impulse_cast_legal',
    '_play_impulse_cast_execute',
]
