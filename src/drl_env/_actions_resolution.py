"""Generic pending-resolution dispatch: Pass, the shared by-name "Choose: X"
dispatch (search_fetch/discard/scry/ponder/...), exact-(name, slot)
permanent targeting (own and cross-player), pool-mana spending, and every
small universal decision row (pay_unless, tuck_position, may_transform/
copy/cast, choose_room, target player/any-target, madness, discard-or-
sacrifice, and each kind's optional Decline). Each just answers one
pending_resolution kind; none is cast/combat/mana specific. legal(state)/
execute(state) factory pairs (or bare functions) build_action_table wires
into every deck's table, most unconditionally."""

import game

from ._actions_common import _GATE_NO_PENDING


def _pass_legal(state):
    if state.pending_resolution is not None:
        return False
    # Goad: a goaded creature that can attack must be declared before Pass ends
    # declare-attackers.
    if state.phase is game.turn.Phase.DECLARE_ATTACKERS and game.has_unfulfilled_goad(state):
        return False
    return True


_pass_legal._pending_gate = _GATE_NO_PENDING


def _pass_execute(state):
    pass  # no-op: Pass is signalled by choose_action returning None, never by calling this


def _tuck_position_legal(state):
    pending = state.pending_resolution
    return pending is not None and pending["kind"] == "tuck_position"


_tuck_position_legal._pending_gate = frozenset({"tuck_position"})


# The complete set of pending kinds _choose_name_options ever dispatches --
# the sole authoritative list (_choose_name_legal's gate and
# legal_action_mask's coverage guard both read this same constant).
#
# INVARIANT: every kind in this set must have its candidate names confined
# to the deciding player's own cards (hand/library/battlefield/graveyard/
# trigger_queue). A "Choose: X" row only ever exists for a name in the
# asking player's own deck, so a kind whose candidates could include
# another player's card must use pointer addressing instead
# (rl.decision.action_bridge), not this by-name dispatch. legal_action_mask's
# coverage check fails loudly if a kind in this set ever produces a
# candidate outside that promise.
_CHOOSE_NAME_PENDING_KINDS = frozenset({
    "search_fetch", "throne_reveal", "discard",
    "discard_or_sacrifice", "ancient_stirrings", "malevolent_rumble", "scry", "surveil",
    "select_to_hand", "order_triggers", "put_on_top", "ponder",
})


def _choose_name_options(state):
    """Plain (uncolored) 'Choose: X' names currently legal for whatever
    pending resolution is active. "choose_permanent" is not handled here
    (see _choose_permanent_legal/_choose_permanent_execute below): it needs
    exact (name, slot) addressing, not this by-name dispatch.

    Every kind handled here must be in _CHOOSE_NAME_PENDING_KINDS."""
    pending = state.pending_resolution
    if pending is None:
        return []
    kind = pending["kind"]
    # "pay_cost" is absent on purpose: producing mana is the "Tap X" mana-ability
    # rows, spending it is "Spend <color> from pool" -- neither is a named option.
    if kind == "search_fetch":
        return game.search_fetch_options(state)
    if kind == "throne_reveal":  # Undercity Throne: pick a creature from the revealed top 10
        return game.throne_reveal_options(state)
    # choose_graveyard_card, choose_stack_target, and choose_permanent are
    # deliberately absent: all are pointer targets (rl.decision.action_bridge),
    # since their candidates can include an opponent's cards or need exact
    # permanent addressing.
    if kind == "discard":
        return game.discard_options(state)
    if kind == "discard_or_sacrifice":
        # Only the discard half reuses this dispatch; the sacrifice half opens
        # its own nested choose_permanent pointer choice instead, to avoid
        # ambiguity when a hand card and a battlefield permanent share a name.
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


def _choose_permanent_legal(name, slot):
    """The "choose_permanent" resolution's action-table half (Aura
    enchant-targets, Crop Rotation's sacrifice cost, land bounce). Legal
    only while that kind is pending and (name, slot) is a current option.
    Exact (name, slot) addressing: a plain by-name "Choose: X" can't tell
    two same-named permanents apart, and cast_aura's fizzle-on-invalid-
    target contract depends on knowing exactly which one was chosen."""
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
    """Cross-player targeting primitive's action-table half. Legal only
    while a "choose_opponent_permanent" resolution is pending and
    (name, slot) is a current option."""
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


# Pregame mulligan decisions are not in this action table -- MulliganNet
# (rl.model.mulligan) owns the pregame phase instead.


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
    """Ponder's "you may shuffle" -- legal only before any card has been
    placed on top (ordered still empty)."""
    pending = state.pending_resolution
    return pending is not None and pending["kind"] == "ponder" and not pending["ordered"]


_ponder_shuffle_legal._pending_gate = frozenset({"ponder"})


def _ponder_shuffle_execute(state):
    game.execute_ponder_shuffle(state)


def _pay_unless_pay_legal(state):
    """"Pay {N}" for the Spell Pierce / Ward rider -- legal only while a
    pay_unless resolution is open and the payer can afford the {N}
    (active_idx is already flipped to the payer)."""
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
    """Only for an optional choose_graveyard_card (e.g. Masked Vandal) with
    real options to decline -- gated on pending["optional"] so it never
    appears for a mandatory graveyard pick sharing the same kind."""
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
    """Legal only once a real second PlayerState exists."""
    pending = state.pending_resolution
    return pending is not None and pending["kind"] == "choose_target_player" and len(state.players) > 1


_target_opponent_legal._pending_gate = frozenset({"choose_target_player"})


def _target_opponent_execute(state):
    game.execute_choose_target_player_option(state, 1 - state.active_idx)


def _target_any_self_legal(state):
    """The "any target" player half (a player is always a legal any
    target, including yourself). Only offered when the pending
    choose_any_target allows players. The creature half of the same choice
    rides the identity pointer scheme (rl.decision.action_bridge), not a
    fixed action."""
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
    """Decline an optional "up to one target" choose_any_target. Only
    legal when the pending was begun optional=True."""
    pending = state.pending_resolution
    return pending is not None and pending["kind"] == "choose_any_target" and pending.get("optional", False)


_target_any_decline_legal._pending_gate = frozenset({"choose_any_target"})


def _target_any_decline_execute(state):
    game.execute_choose_any_target_decline(state)


def _discard_or_sacrifice_trigger_sacrifice_legal(state):
    """The sacrifice half of "discard a card or sacrifice a land" -- one
    trigger action (not one per land name); picking it opens a nested
    choose_permanent sub-decision for which exact land pays the cost.
    Legal only while discard_or_sacrifice is pending and at least one
    eligible permanent exists."""
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
    """Legal only if the model can afford the exiled card's madness cost
    right now."""
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


__all__ = [
    '_tuck_position_legal',
    '_pass_legal',
    '_pass_execute',
    '_choose_name_options',
    '_choose_name_legal',
    '_choose_name_execute',
    '_choose_permanent_legal',
    '_choose_permanent_execute',
    '_choose_opponent_permanent_legal',
    '_choose_opponent_permanent_execute',
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
]
