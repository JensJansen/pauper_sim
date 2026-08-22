"""Casting/activating/playing a card from a zone: play land, plain/modal/
X-cost/Delve hand cast, activate, forestcycle/cycle, and impulse ("play
from exile"). Casting from a non-hand zone or for a non-default cost
(alt-cost/Flashback/Escape/Plot/Omen/Prototype) lives in the sibling
_actions_cast_altzone. Each function here is a legal(state)/execute(state)
factory pair that build_action_table calls once per matching card."""

import game

from ._actions_cast_altzone import _with_chosen_copy
from ._actions_common import _GATE_NO_PENDING, _hand_count_available

# Shared button-count caps for choose_cast_mode/choose_cast_x/
# choose_delve_amount (sized to the current catalog's max: 5 modes, X=0..10,
# delve 0..6). Raise if a future card needs more; per-button legal() masks
# unaffordable/out-of-range values regardless of the cap.
_CAST_MODE_BUTTON_MAX = 5
_CAST_X_BUTTON_MAX = 10
_DELVE_BUTTON_MAX = 6


def _cast_speed(card_def, spec):
    """The game.turn.Speed for a cast-like action: an explicit "speed" key
    in spec if given, else Speed.INSTANT for a CardType.INSTANT card, else
    Speed.SORCERY."""
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
            # Playing a land is sorcery-speed; speed_legal's SORCERY branch
            # also requires active_idx == turn_player_idx.
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
    """Cast cost after any registry "cost_reduction" (a lambda(state) -> int,
    e.g. affinity, graveyard instant/sorcery count, cards drawn this turn).
    Lowers only the generic pips, floored at 0; colored pips are never
    reduced. Cards without a cost_reduction spec pay cast_cost unchanged."""
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
            # 601.2i: not "cast" until cost is paid, so a cast trigger fires here.
            game.on_cast_trigger(s, card_def)
            # Cast but not yet resolved -- push onto the stack rather than
            # resolving immediately, so priority can pass first.
            game.push_to_stack(s, card_def, resolve)
        game.begin_pay_cost(state, _effective_cast_cost(state, card_def), on_complete=_after_pay)
    return execute


def _precast_choice_execute(name, resolve):
    """Cast-like execute for a card whose `resolve` must settle a target
    (cast_aura) or additional cost (cast_crop_rotation) before the spell is
    fully cast, not on resolution. Unlike _cast_execute, `resolve` is called
    directly as pay_cost's on_complete and pushes to the stack itself.
    Selected via a cast/cast_modes spec's "precast_choice": True flag."""
    def execute(state):
        card_def = game.CARD_DEFS[name]
        def _after_pay(s):
            game.on_cast_trigger(s, card_def)
            resolve(s, card_def)
        game.begin_pay_cost(state, _effective_cast_cost(state, card_def), on_complete=_after_pay)
    return execute


def _x_cast_legal(name, cost, extra_legal, speed):
    """Like _cast_legal, but against an explicit `cost` (an x_cast_modes
    mode's base cost plus a given X's generic) instead of card_def.cast_cost."""
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
    """Same shape as _precast_choice_execute, against an explicit `cost`
    (e.g. Nyxborn Hydra's Bestow mode: a precast target plus an X-dependent
    cost)."""
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
    """Legal iff at least one delve amount 0..max_n is affordable given the
    current graveyard size. One "Cast <name>" row; the shared "Delve n"
    buttons mask unaffordable amounts."""
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
    """601.2f/702.66: choose the delve amount, exile that many graveyard
    cards (model chooses which) as a cost, pay the reduced remainder, then
    cast normally. Both the amount choice and the exile happen before
    601.2f, so mana abilities are illegal during them (mid_cast=True)."""
    def execute(state):
        card_def = game.CARD_DEFS[name]

        def _after_n(state, n):
            def _after_exile(s):
                def _after_pay(s2):
                    game.on_cast_trigger(s2, card_def)
                    game.push_to_stack(s2, card_def, resolve)
                game.begin_pay_cost(s, _delve_reduced_cost(card_def, n), on_complete=_after_pay)
            game.begin_exile_n_from_graveyard(state, n, _after_exile, mid_cast=True)

        game.begin_choose_delve_amount(state, card_def, max_n, _after_n)
    return execute


def _choose_delve_amount_legal(n):
    """Shared "Delve n" button. Legal only mid a choose_delve_amount
    resolution, within the card's max_n, with enough graveyard cards, and
    only if the resulting reduced cost is affordable."""
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
    """Legal iff at least one mode is currently produceable: its
    extra_legal (if any) passes and its cost (mode override, else
    _effective_cast_cost) is affordable. One "Cast <name>" row, not one
    per mode."""
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
    """601.2b: mode chosen before cost is calculated. Opens
    choose_cast_mode; once a mode is picked, computes its cost and proceeds
    as a plain cast (begin_pay_cost -> push_to_stack), or via
    precast_choice if the mode needs one (e.g. Utopia Sprawl)."""
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
    """Legal iff at least one (mode, X) pair is affordable, X=0 upward per
    mode's base cost."""
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
    cost is paid. make_resolve(x) is called once X is known, at execute
    time rather than table-build time."""
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
    """Shared "Mode n" button. Legal only mid a choose_cast_mode
    resolution, within the card's mode count, and only if that mode's
    extra_legal/affordability check passes."""
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
    """Shared "X=n" button. Legal only mid a choose_cast_x resolution,
    within the mode's max_x, and only if base_cost+x's generic is
    affordable."""
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
    "an untapped copy exists and its cost_key cost is payable" -- e.g. an
    ability requiring a legal target (602.2b/601.2c)."""
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
    """Graveyard activated ability (e.g. Bramble Wurm's exile-for-life). Same
    shape as _forestcycle_legal but sourced from state.graveyard; unlike
    Flashback it resolves directly, no push_to_stack. No speed gate --
    activated abilities default to "any time"."""
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
    """resolve receives the graveyard CardInstance whose ability this is
    (not the interned CardDef). The cost is read off game.CARD_DEFS[name]
    since it's identical for every copy."""
    def execute(state):
        cost = game.CARD_DEFS[name].extra[cost_key]

        def _proceed(state, inst):
            # Copy chosen, then cost paid (602.2/601.2a order), then resolve.
            game.begin_pay_cost(state, cost, on_complete=lambda s, inst=inst: resolve(s, inst))

        _with_chosen_copy(state, name, _proceed)
    return execute


def _activate_no_cost_legal(name, ability_legal, speed):
    """Non-mana activated-ability cost (e.g. Quirion Ranger's Forest
    bounce): no {T}-of-self assumption; ability_legal(state, permanent)
    captures the whole cost precondition."""
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
    """Topmost still-unexpired impulse entry (card_def, deadline) for
    `name`, or None."""
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
        if not game.turn.speed_legal(state, game.turn.Speed.SORCERY):
            return False
        return state.lands_played_this_turn == 0
    legal._pending_gate = _GATE_NO_PENDING
    return legal


def _play_impulse_land_execute(name):
    def execute(state):
        entry = _impulse_entry(state, name)
        state.impulse.remove(entry)
        state.hand.append(entry[0])  # source via hand so play_land_from_hand works
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
    unlike Plot, is not free). The card leaves the impulse zone only once
    mana is paid, then is inserted into hand so the card's own resolve
    (written for a hand cast) finds and removes it, then either pushed to
    the stack or resolved directly if precast."""
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
