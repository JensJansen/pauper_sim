"""Casting from a non-hand zone or for a non-default cost: free alt-costs
(Land Grant), Flashback/Escape (graveyard), Plot (exile), and Omen/
Prototype (a second cast option for the same hand card, its own cost).
Split out of drl_env._actions_cast, which covers plain hand casts plus
play-land/activate/forestcycle/impulse. Each function is a legal(state)/
execute(state) factory pair build_action_table calls once per matching
card."""

import game

from ._actions_common import _GATE_NO_PENDING, _hand_count_available


def _alt_cast_legal(name, extra_legal, speed):
    """Free alt-cost (e.g. Land Grant): no mana payment, just extra_legal.

    Must use _hand_count_available rather than a bare "any copy in hand"
    check: a copy already cast and awaiting resolution on the stack is
    still physically in hand (removal is deferred to its own resolve), so
    a bare existence check would let it be re-cast."""
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
    """No generic engine-level cost mechanism for an alt cost, so this calls
    resolve immediately and leaves paying the cost and pushing to the
    stack entirely up to resolve itself (cost varies per card: some are
    free, some like Fireblast's sacrifice cost must be paid before the
    effect is pushed)."""
    def execute(state):
        card_def = game.CARD_DEFS[name]
        game.on_cast_trigger(state, card_def)
        resolve(state, card_def)
    return execute


def _flashback_legal(name, ability_legal, speed, cost=None):
    """Flashback: cast from the graveyard, not hand. speed matches the
    card's normal cast (Flashback follows the same timing as the card).

    cost (optional): a mana cost dict for a flashback with a mana
    component (e.g. Deep Analysis' {1}{U}), checked via plan_payment like a
    normal cast. Free/sacrifice-only flashbacks leave it None and pay
    entirely inside resolve. Non-mana additional costs (e.g. "pay 3 life")
    are gated by ability_legal instead."""
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
    """Recovers the real graveyard CardInstance for `name`. Execute closures
    only hold a card name (the action table is name-keyed), and unlike hand
    casts (where game.CARD_DEFS[name] IS the object in hand), the graveyard
    holds per-object CardInstances distinct from the interned CardDef -- so
    this is the one place that resolves name to instance.

    Picks the first same-named instance -- correct only when at most one
    copy exists (MTG 400.7: same-named graveyard cards are otherwise
    interchangeable). When 2+ copies exist and the choice is observable,
    the caller _with_chosen_copy asks the player instead and only calls
    here for the no-choice case."""
    inst = next((c for c in state.graveyard if c.name == name), None)
    if inst is None:
        # Should be unreachable: callers already require a same-named graveyard card.
        raise RuntimeError(
            f"_graveyard_instance: no {name!r} in graveyard. "
            f"active_idx={getattr(state, 'active_idx', None)!r} "
            f"turn_number={getattr(state, 'turn_number', None)!r} "
            f"graveyard={[c.name for c in state.graveyard]!r}"
        )
    return inst


def _with_chosen_copy(state, name, proceed):
    """Run `proceed(state, inst)` on the graveyard copy of `name` being cast.

    With 2+ same-named copies, which object is chosen is a real agent
    choice (MTG 601.2a), made before any cost is paid: opens a
    choose_cast_copy pending and continues from its on_complete. With
    exactly one copy it proceeds inline, no choice to make.

    Also called from _actions_cast._graveyard_ability_execute for the same
    identity-recovery need."""
    copies = [c for c in state.graveyard if c.name == name]
    if len(copies) <= 1:
        proceed(state, _graveyard_instance(state, name))
        return
    game.begin_choose_cast_copy(state, name, on_complete=proceed)


def _flashback_execute(name, resolve, cost=None):
    """resolve receives the graveyard CardInstance being cast (not the
    interned CardDef). WHICH copy, when several exist, is the agent's
    choice (_with_chosen_copy)."""
    def execute(state):
        def _proceed(state, inst):
            if cost is None:
                game.on_cast_trigger(state, inst)
                resolve(state, inst)
                return
            # Mana cost paid after the copy is chosen (601.2a announce-then-pay).
            def _after_pay(state, inst=inst):
                game.on_cast_trigger(state, inst)
                resolve(state, inst)
            game.begin_pay_cost(state, cost, on_complete=_after_pay)

        _with_chosen_copy(state, name, _proceed)
    return execute


def _plot_legal(name, cost, speed):
    """Plot {cost}: pay it and exile this card from hand (no board presence
    yet) -- legal like a normal cast, against the plot cost instead of
    cast_cost. speed matches the card's normal cast, per Plot's own
    reminder text ("any time you could cast this card")."""
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
        # Plotting isn't casting the spell -- on_cast_trigger fires later, from
        # _cast_from_exile_execute, once it's actually cast.
        game.begin_pay_cost(state, cost, on_complete=lambda s: resolve(s, card_def))
    return execute


def _cast_from_exile_legal(name, extra_legal, speed):
    """Plot's second half: cast a previously-plotted copy without paying
    its mana cost, on any turn after the one it was plotted on. speed
    matches _plot_legal.

    extra_legal: Plot only waives the mana cost, not any other cost a
    card's normal "cast" spec gates on (e.g. a discard additional cost);
    reuses the same cast_spec["extra_legal"] the normal cast path checks."""
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
        game.on_cast_trigger(state, card_def)
        # Cost was already paid when plotted, so push straight to the stack.
        game.push_to_stack(state, card_def, resolve, reserves_hand_card=False)
    return execute


def _omen_cast_legal(hand_name, cost, speed):
    """Omen (e.g. Sagu Wildling): unlike Adventure, the resolved sorcery
    shuffles back into the library rather than exiling for a later cast, so
    the creature side only becomes castable again once redrawn to hand.
    This is really just "the same hand card, a second cast option with its
    own cost" -- checked against state.hand. hand_name is the sorcery
    side's registered name; the creature side is a distinct CardDef never
    separately registered in game.CARD_DEFS."""
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
    """Same begin_pay_cost -> push_to_stack shape as a normal hand cast, for
    `creature_card_def` instead of game.CARD_DEFS[name]. The hand card is a
    different object from creature_card_def despite sharing a display name
    (it's the sorcery side's CardDef), so push_to_stack's identity-based
    removal misses it; it's removed here by name instead, which is what
    blocks casting the other mode of the same physical copy meanwhile."""
    def execute(state):
        game.on_cast_trigger(state, creature_card_def)  # no-op for a creature card_def

        def _after_pay(s):
            hand_card = next((c for c in s.hand if c.name == creature_card_def.name), None)
            if hand_card is not None:
                s.hand.remove(hand_card)
            game.push_to_stack(s, creature_card_def, resolve)
        game.begin_pay_cost(state, cost, on_complete=_after_pay)
    return execute


__all__ = [
    '_alt_cast_legal',
    '_alt_cast_execute',
    '_flashback_legal',
    '_graveyard_instance',
    '_with_chosen_copy',
    '_flashback_execute',
    '_plot_legal',
    '_plot_execute',
    '_cast_from_exile_legal',
    '_cast_from_exile_execute',
    '_omen_cast_legal',
    '_omen_cast_execute',
]
