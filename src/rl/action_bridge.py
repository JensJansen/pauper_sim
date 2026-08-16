"""Bridges rl.deck's combined (fixed-table + pointer-head) action
space to the REAL game engine -- the piece that determines whether any of
this can actually drive a game. Two halves:

1. build_fixed_action_table: today's drl_env.build_action_table, filtered
   to drop the four "(name) (slot k)"-addressed targeting categories
   (Attack, Assign Blocker, Choose target, Choose opponent's) -- those move
   to the pointer head below. Deliberately reuses build_action_table rather
   than reimplementing the non-targeting half; only the targeting slice of
   this codebase's action representation is changing.

2. pointer_legal_mask / execute_pointer_choice: legality and execution for
   the targeting half, reusing the EXACT existing predicates
   (game.creature_attack_eligible, game.creature_block_eligible,
   game.choose_permanent_options, game.choose_opponent_permanent_options)
   and the EXACT existing execute closures
   (drl_env._attack_execute/_assign_blocker_execute/_choose_permanent_
   execute/_choose_opponent_permanent_execute) -- combat/resolution
   mechanics are not reimplemented here, only re-addressed by token
   position instead of by (name, slot) table lookup."""

import game
import drl_env

_TARGETING_PREFIXES = ("Choose target: ", "Attack: ", "Assign Blocker: ", "Choose opponent's: ")

# Pending kinds whose pointer targets are matched by OBJECT IDENTITY
# (id()-keyed, never `ref in <set(...)>`, since choose_stack_target's own
# option set holds unhashable stack-entry dicts) rather than by (name, slot):
# each is a case where two same-named/same-shaped copies must stay
# independently addressable -- choose_cast_copy (WHICH graveyard copy is cast,
# MTG 601.2a: casting the one a Rooftop Percher trigger targets saves it from
# exile, casting the other doesn't), choose_graveyard_card (WHICH graveyard
# copy leaves the yard -- no whole-league "Choose: X" fixed row exists, and
# both seats' graveyards, e.g. Relic of Progenitus, are reachable), and
# choose_stack_target (WHICH spell on the stack to counter, very often the
# OPPONENT's -- confirmed live via an all-False mask in real cross-deck league
# play; two simultaneous same-named spells must stay independently
# addressable too). Maps kind -> (options-fetcher, executor) names.
_ID_MATCHED_KINDS = {
    "choose_cast_copy": ("choose_cast_copy_options", "execute_choose_cast_copy_option"),
    "choose_graveyard_card": ("choose_graveyard_card_options", "execute_choose_graveyard_card_option"),
    "choose_stack_target": ("choose_stack_target_options", "execute_choose_stack_target_option"),
}


def build_fixed_action_table(decklist, token_card_defs=(), extra_choosable_names=()):
    """Every non-targeting action for this decklist (Play land, Cast,
    Activate, Forestcycle, Pass, "Choose: X" resolution picks, "Choose: X
    as color", Keep/Dispose, Decline, Abandon payment, mulligan actions,
    "Done blocking") -- legitimately fixed-shape, since a trained model's
    own decklist never changes at inference time under this design.
    opponent_decklist=None always: the ONLY thing that constructor arg
    controls is whether "Choose opponent's: ..." entries get added at all
    (drl_env.build_action_table's own "if opponent_decklist is not None"
    gate), and this table drops that category regardless -- passing the
    real opponent's decklist here would build entries this function then
    immediately throws away."""
    full_table = drl_env.build_action_table(
        decklist, game.EFFECT_REGISTRY, token_card_defs=token_card_defs,
        opponent_decklist=None, extra_choosable_names=extra_choosable_names,
    )
    return [
        (name, legal_fn, execute_fn) for name, legal_fn, execute_fn in full_table
        if not name.startswith(_TARGETING_PREFIXES)
    ]


def pointer_legal_mask(state, identities_row):
    """identities_row: one batch element's own list of (Permanent, graveyard
    CardInstance, revealed-hand CardDef, or None) per token position
    (pad_token_batch's own per-row output). Returns a
    same-length list of bools -- which positions are a legal pointer target
    for whichever ONE targeting category actually applies right now (at most
    one ever does, by construction of this engine's own turn/resolution state
    machine -- see each branch's own gate below, mirroring drl_env's own
    _attack_legal/_assign_blocker_legal/_choose_permanent_legal/etc. exactly).

    Checked BEFORE state.pending_resolution, taking exclusive priority over
    it: a mana_subdecision (Saruli Caretaker's own "tap another creature,
    then choose a color") can be open WHILE a pending_resolution is also
    open (that's the entire reason it's a separate field -- see
    state.mana_subdecision's own docstring), and its own choose_target stage
    is itself a pointer choice, so it needs to win the dispatch here, not
    fall through to whatever pending_resolution's own branch would offer."""
    mana_sub = state.active_mana_subdecision
    if mana_sub is not None:
        mask = [False] * len(identities_row)
        if mana_sub["stage"] == "choose_target":
            source, predicate = mana_sub["source"], mana_sub["target_predicate"]
            for i, p in enumerate(identities_row):
                if (p is not None and p in state.battlefield and p is not source
                        and not p.tapped and predicate(p)):
                    mask[i] = True
        return mask  # choose_color stage is a fixed button, not a pointer choice -- all-False here

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
        # (name, slot) alone is NOT globally unique -- slot numbers restart
        # per player (game/state.py's Permanent defaults slot=1, reassigned
        # independently per side in casting.py), so a same-named permanent
        # on the OPPONENT's board with the same slot number collides here
        # unless membership in MY OWN state.battlefield is checked too --
        # exactly the check the attack/block branches above already make.
        # choose_permanent_options only ever enumerates state.battlefield
        # (my own board -- see its own docstring), so that's the correct
        # domain to require membership in.
        legal_pairs = set(game.choose_permanent_options(state))
        for i, p in enumerate(identities_row):
            if p is not None and p in state.battlefield and (p.card_def.name, p.slot) in legal_pairs:
                mask[i] = True
        return mask

    if pending is not None and pending["kind"] == "choose_opponent_permanent":
        # Same collision, opposite side: choose_opponent_permanent_options
        # only ever enumerates state.opponent.battlefield, so membership
        # must be checked against THAT list, not mine.
        legal_pairs = set(game.choose_opponent_permanent_options(state))
        for i, p in enumerate(identities_row):
            if p is not None and p in state.opponent.battlefield and (p.card_def.name, p.slot) in legal_pairs:
                mask[i] = True
        return mask

    if pending is not None and pending["kind"] == "assign_combat_damage":
        # The attacker assigns its next combat-damage point to one of its
        # blockers (gang-blocking). Match by IDENTITY against the pending's
        # own blocker list -- NOT (name, slot), which collides across sides
        # (a same-named creature on the attacker's own board) -- so only the
        # actual blockers are offered. "Assign to the player" (trample) is a
        # separate FIXED action, not a pointer target.
        blockers = pending["blockers"]
        for i, p in enumerate(identities_row):
            if p is not None and p in blockers:
                mask[i] = True
        return mask

    if pending is not None and pending["kind"] == "choose_any_target":
        # "Any target" creature half -- a creature on EITHER battlefield
        # (Lightning Bolt). Match by (side, name, slot) where side is which
        # player controls the permanent, so a same-named creature on each
        # side stays distinct (the same cross-side (name,slot) collision the
        # choose_permanent branches guard). The identity p tells us its side
        # directly. The player half is fixed actions, not pointer targets.
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
        # id()-KEYED, not `ref in set(...)`: identities_row can carry graveyard
        # CardInstances, revealed-hand CardDefs, or STACK-ENTRY dicts depending
        # on which of these three kinds is pending, and a dict is unhashable --
        # `in <set>` would crash outright the instant one reached this branch.
        # id() sidesteps that for every object type, no exceptions, the same
        # reasoning rl.features._stack_target_map already established. See
        # _ID_MATCHED_KINDS' own docstring-comment above for why each of these
        # three kinds needs identity (not name/slot) matching in the first place.
        options_fn = getattr(game, _ID_MATCHED_KINDS[pending["kind"]][0])
        legal_ids = {id(o) for o in options_fn(state)}
        for i, ref in enumerate(identities_row):
            if ref is not None and id(ref) in legal_ids:
                mask[i] = True
        return mask

    return mask  # no targeting category applies right now -- pointer half is entirely illegal, only the fixed table matters


def any_pointer_legal(state):
    """Cheap pre-check (no token list needed) for whether ANY pointer
    category applies at all right now -- lets a caller skip building
    identities/mask work entirely on the (common) steps where it's not
    even relevant, same "cheap gate before the expensive check" shape
    drl_env's own _pending_gate convention already uses throughout."""
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
    """Which ONE pointer targeting category applies right now, if any -- same
    two-tier dispatch as pointer_legal_mask (mana_subdecision checked first,
    exactly as that function's own docstring explains), just returning its
    name instead of a legality mask. For decision-weight logging only
    (rl.agent/rl.mulligan); execution and legality never need this, only the
    display label a replay viewer attaches to a pointer candidate."""
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
    """Dispatches to the EXACT existing engine/drl_env execute paths (never
    reimplemented here) for whichever targeting category is currently pending.
    `chosen` is the exact object for choose_graveyard_card (a graveyard
    CardInstance, or a revealed-hand CardDef -- executed by object identity),
    for choose_cast_copy (the graveyard CardInstance being cast), and for
    choose_stack_target (the exact stack-entry dict to counter) -- executed by
    object identity in all three cases; else the live Permanent for a
    battlefield target (addressed by its own (name, slot), exactly what those
    closures already expect) -- just sourced from a pointer-head selection
    instead of a fixed-table lookup."""
    mana_sub = state.active_mana_subdecision
    if mana_sub is not None and mana_sub["stage"] == "choose_target":
        # `chosen` is the exact Permanent to tap -- passed through directly
        # (same identity-object shape _ID_MATCHED_KINDS below already uses),
        # not round-tripped through (name, slot): there's no shared
        # (name, slot)-keyed execute closure to reuse here the way the
        # battlefield-target branches below do.
        return game.execute_mana_subdecision_target(state, chosen)
    pending = state.pending_resolution
    if pending is not None and pending["kind"] in _ID_MATCHED_KINDS:
        # chosen is the exact object (graveyard instance, revealed-hand
        # CardDef, or stack-entry dict) selected by object identity -- see
        # _ID_MATCHED_KINDS' own docstring-comment for why each of these
        # kinds is pointer-only and identity-matched.
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
