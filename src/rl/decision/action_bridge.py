"""Bridges rl.model.deck's combined (fixed-table + pointer-head) action space
to the game engine. Two halves:

1. build_fixed_action_table: drl_env.build_action_table filtered to drop the
   four targeting categories (Attack, Assign Blocker, Choose target, Choose
   opponent's), which move to the pointer head below.

2. pointer_legal_mask / execute_pointer_choice: legality and execution for
   the targeting half, reusing the engine's existing predicates
   (game.creature_attack_eligible, game.creature_block_eligible,
   game.choose_permanent_options, game.choose_opponent_permanent_options) and
   execute closures (drl_env._attack_execute/_assign_blocker_execute/
   _choose_permanent_execute/_choose_opponent_permanent_execute), re-addressed
   by token position instead of by (name, slot) table lookup."""

import game
import drl_env

_TARGETING_PREFIXES = ("Choose target: ", "Attack: ", "Assign Blocker: ", "Choose opponent's: ")

# Pending kinds whose pointer targets are matched by object identity (id()),
# not (name, slot): duplicate same-named copies must stay independently
# addressable (choose_cast_copy: MTG 601.2a; choose_graveyard_card: either
# player's graveyard; choose_stack_target: often the opponent's spell, and its
# own option set holds unhashable dicts anyway). Maps kind -> (options-fetcher,
# executor) names.
_ID_MATCHED_KINDS = {
    "choose_cast_copy": ("choose_cast_copy_options", "execute_choose_cast_copy_option"),
    "choose_graveyard_card": ("choose_graveyard_card_options", "execute_choose_graveyard_card_option"),
    "choose_stack_target": ("choose_stack_target_options", "execute_choose_stack_target_option"),
}


def build_fixed_action_table(decklist, token_card_defs=(), extra_choosable_names=()):
    """Every non-targeting action for this decklist (Play land, Cast,
    Activate, Forestcycle, Pass, "Choose: X" resolution picks, "Choose: X as
    color", Keep/Dispose, Decline, Abandon payment, mulligan actions, "Done
    blocking"). Always calls with opponent_decklist=None since this table
    drops "Choose opponent's: ..." rows regardless."""
    full_table = drl_env.build_action_table(
        decklist, game.EFFECT_REGISTRY, token_card_defs=token_card_defs,
        opponent_decklist=None, extra_choosable_names=extra_choosable_names,
    )
    return [
        (name, legal_fn, execute_fn) for name, legal_fn, execute_fn in full_table
        if not name.startswith(_TARGETING_PREFIXES)
    ]


def pointer_legal_mask(state, identities_row):
    """identities_row: one batch element's per-token-position list of
    (Permanent, graveyard CardInstance, revealed-hand CardDef, or None)
    (pad_token_batch's own per-row output). Returns a same-length bool list
    marking legal pointer targets for whichever ONE targeting category
    currently applies (at most one ever does).

    Checked before state.pending_resolution: a mana_subdecision can be open
    while a pending_resolution is also open, and its own choose_target stage
    is itself a pointer choice that takes priority over the pending's."""
    mana_sub = state.active_mana_subdecision
    if mana_sub is not None:
        mask = [False] * len(identities_row)
        if mana_sub["stage"] == "choose_target":
            # Excludes any tap that would leave the payment unpayable.
            from drl_env._actions_mana import mana_extra_choose_target_safe
            source, predicate = mana_sub["source"], mana_sub["target_predicate"]
            for i, p in enumerate(identities_row):
                if (p is not None and p in state.battlefield and p is not source
                        and not p.tapped and predicate(p)
                        and mana_extra_choose_target_safe(state, p)):
                    mask[i] = True
        return mask  # choose_color stage is a fixed button, not a pointer choice

    pending = state.pending_resolution
    mask = [False] * len(identities_row)

    if (state.phase is game.turn.Phase.DECLARE_ATTACKERS and state.active_idx == state.turn_player_idx
            and pending is None):
        for i, p in enumerate(identities_row):
            if p is not None and p in state.battlefield and game.creature_attack_eligible(state, p):
                mask[i] = True
        return mask

    if pending is not None and pending["kind"] == "declare_blockers":
        for i, p in enumerate(identities_row):
            if p is not None and p in state.battlefield and game.creature_block_eligible(state, p):
                mask[i] = True
        return mask

    if pending is not None and pending["kind"] == "choose_permanent":
        # (name, slot) isn't globally unique -- slot numbers restart per
        # player -- so membership in state.battlefield (my own board) must
        # also be checked to avoid colliding with the opponent's same-named
        # permanent.
        legal_pairs = set(game.choose_permanent_options(state))
        for i, p in enumerate(identities_row):
            if p is not None and p in state.battlefield and (p.card_def.name, p.slot) in legal_pairs:
                mask[i] = True
        return mask

    if pending is not None and pending["kind"] == "choose_opponent_permanent":
        # Same (name, slot) collision, opposite side: membership is checked
        # against state.opponent.battlefield instead.
        legal_pairs = set(game.choose_opponent_permanent_options(state))
        for i, p in enumerate(identities_row):
            if p is not None and p in state.opponent.battlefield and (p.card_def.name, p.slot) in legal_pairs:
                mask[i] = True
        return mask

    if pending is not None and pending["kind"] == "assign_combat_damage":
        # Legal blockers are those still under their own lethal_by_blocker
        # cap (no overkill). Matched by identity, not (name, slot), since a
        # same-named creature can exist on either side. Trample's "assign to
        # the player" is a forced outcome once every blocker is capped, not
        # a choice offered here.
        blockers = pending["blockers"]
        lethal_by_blocker = pending["lethal_by_blocker"]
        amounts = pending["amounts"]
        for i, p in enumerate(identities_row):
            if p is not None and p in blockers and amounts.get(p, 0) < lethal_by_blocker[p]:
                mask[i] = True
        return mask

    if pending is not None and pending["kind"] == "choose_any_target":
        # "Any target" creature half (a creature on either battlefield).
        # Matched by (side, name, slot) so same-named creatures on each side
        # stay distinct. The player-target half is a fixed action, not a
        # pointer target.
        legal_triples = set(game.choose_any_target_creature_options(state))
        for i, p in enumerate(identities_row):
            if p is None:
                continue
            if p in state.players[0].battlefield:
                side = 0
            elif len(state.players) > 1 and p in state.players[1].battlefield:
                side = 1
            else:
                continue
            if (side, p.card_def.name, p.slot) in legal_triples:
                mask[i] = True
        return mask

    if pending is not None and pending["kind"] in _ID_MATCHED_KINDS:
        # id()-keyed, not `ref in set(...)`: identities_row can hold
        # stack-entry dicts here, which are unhashable.
        options_fn = getattr(game, _ID_MATCHED_KINDS[pending["kind"]][0])
        legal_ids = {id(o) for o in options_fn(state)}
        for i, ref in enumerate(identities_row):
            if ref is not None and id(ref) in legal_ids:
                mask[i] = True
        return mask

    return mask  # no targeting category applies -- pointer half entirely illegal


def any_pointer_legal(state):
    """Cheap check for whether any pointer category applies right now,
    without building the token list -- lets a caller skip mask work when
    it's not relevant."""
    mana_sub = state.active_mana_subdecision
    if mana_sub is not None:
        return mana_sub["stage"] == "choose_target"
    pending = state.pending_resolution
    if pending is None:
        return state.phase is game.turn.Phase.DECLARE_ATTACKERS and state.active_idx == state.turn_player_idx
    return pending["kind"] in (
        "declare_blockers", "choose_permanent", "choose_opponent_permanent", "assign_combat_damage",
        "choose_any_target", "choose_graveyard_card", "choose_cast_copy", "choose_stack_target",
    )


def pointer_kind(state):
    """Which one pointer targeting category applies right now, if any -- same
    dispatch order as pointer_legal_mask, returning its name instead of a
    mask. Used only for decision-weight logging (the replay viewer's
    label)."""
    mana_sub = state.active_mana_subdecision
    if mana_sub is not None:
        return "mana_subdecision" if mana_sub["stage"] == "choose_target" else None
    pending = state.pending_resolution
    if pending is None:
        if state.phase is game.turn.Phase.DECLARE_ATTACKERS and state.active_idx == state.turn_player_idx:
            return "declare_attackers"
        return None
    if pending["kind"] in (
        "declare_blockers", "choose_permanent", "choose_opponent_permanent", "assign_combat_damage",
        "choose_any_target",
    ) or pending["kind"] in _ID_MATCHED_KINDS:
        return pending["kind"]
    return None


def execute_pointer_choice(state, chosen):
    """Dispatches to the engine's existing execute paths for whichever
    targeting category is pending. `chosen` is the exact object for the
    identity-matched kinds (graveyard CardInstance, revealed-hand CardDef, or
    stack-entry dict); otherwise the live Permanent, addressed by
    (name, slot)."""
    mana_sub = state.active_mana_subdecision
    if mana_sub is not None and mana_sub["stage"] == "choose_target":
        # `chosen` is the Permanent to tap, passed through directly (no
        # shared (name, slot) execute closure exists for this case).
        return game.execute_mana_subdecision_target(state, chosen)
    pending = state.pending_resolution
    if pending is not None and pending["kind"] in _ID_MATCHED_KINDS:
        # chosen is the exact object selected by identity (see _ID_MATCHED_KINDS).
        return getattr(game, _ID_MATCHED_KINDS[pending["kind"]][1])(state, chosen)
    name, slot = chosen.card_def.name, chosen.slot
    if pending is None:
        return drl_env._attack_execute(name, slot)(state)
    if pending["kind"] == "declare_blockers":
        return drl_env._assign_blocker_execute(name, slot)(state)
    if pending["kind"] == "choose_permanent":
        return drl_env._choose_permanent_execute(name, slot)(state)
    if pending["kind"] == "choose_opponent_permanent":
        return drl_env._choose_opponent_permanent_execute(name, slot)(state)
    if pending["kind"] == "assign_combat_damage":
        return game.execute_assign_combat_damage_option(state, name, slot)  # +1 damage point to this blocker
    if pending["kind"] == "choose_any_target":
        side = 0 if chosen in state.players[0].battlefield else 1  # which battlefield the chosen creature is on
        return game.execute_choose_any_target_creature(state, side, name, slot)
    raise ValueError(f"execute_pointer_choice: no pointer category applies for pending kind {pending['kind']!r}")
