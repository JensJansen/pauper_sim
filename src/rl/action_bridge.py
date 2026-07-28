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


def build_fixed_action_table(decklist, token_card_defs=(), pending_kinds=(), extra_choosable_names=()):
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
        decklist, game.EFFECT_REGISTRY, token_card_defs=token_card_defs, pending_kinds=pending_kinds,
        opponent_decklist=None, extra_choosable_names=extra_choosable_names,
    )
    return [
        (name, legal_fn, execute_fn) for name, legal_fn, execute_fn in full_table
        if not name.startswith(_TARGETING_PREFIXES)
    ]


def pointer_legal_mask(state, identities_row):
    """identities_row: one batch element's own list of (Permanent or None)
    per token position (pad_token_batch's own per-row output). Returns a
    same-length list of bools -- which positions are a legal pointer target
    for whichever ONE of the four targeting categories actually applies
    right now (at most one ever does, by construction of this engine's own
    turn/resolution state machine -- see each branch's own gate below,
    mirroring drl_env's own _attack_legal/_assign_blocker_legal/
    _choose_permanent_legal/_choose_opponent_permanent_legal exactly)."""
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

    return mask  # no targeting category applies right now -- pointer half is entirely illegal, only the fixed table matters


def any_pointer_legal(state):
    """Cheap pre-check (no token list needed) for whether ANY pointer
    category applies at all right now -- lets a caller skip building
    identities/mask work entirely on the (common) steps where it's not
    even relevant, same "cheap gate before the expensive check" shape
    drl_env's own _pending_gate convention already uses throughout."""
    pending = state.pending_resolution
    if pending is None:
        return state.phase is game.turn.Phase.DECLARE_ATTACKERS and state.active_idx == state.turn_player_idx
    return pending["kind"] in (
        "declare_blockers", "choose_permanent", "choose_opponent_permanent", "assign_combat_damage", "choose_any_target",
    )


def execute_pointer_choice(state, permanent):
    """Dispatches to the EXACT existing drl_env execute closures (never
    reimplemented here) for whichever targeting category is currently
    pending, addressed by the chosen permanent's own (name, slot) --
    exactly what those closures already expect, just sourced from a
    pointer-head selection instead of a fixed-table lookup."""
    name, slot = permanent.card_def.name, permanent.slot
    pending = state.pending_resolution
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
        side = 0 if permanent in state.players[0].battlefield else 1  # which battlefield the chosen creature is on
        return game.execute_choose_any_target_creature(state, side, name, slot)
    raise ValueError(f"execute_pointer_choice: no pointer category applies for pending kind {pending['kind']!r}")


if __name__ == "__main__":
    # ponytail self-check: run via `python -m rl.action_bridge` from src/.
    decklist = game.parse_decklist_file("../data/mono_red_madness.txt")
    pending_kinds = game.derive_pending_kinds(decklist)
    # mono_red_madness creates BOTH of these mid-game (Blood from madness
    # discards, Robot from Melded Moxite's own sacrifice ability) -- note
    # the deck config's own "token_card_defs" only
    # lists "Blood", missing "Robot" (confirmed the hard way: a real game
    # creates one). Out of scope to fix that config from this branch, but
    # this module needs the complete, correct set regardless.
    token_defs = (game.BLOOD_TOKEN_CARD_DEF, game.ROBOT_TOKEN_CARD_DEF)
    fixed_table = build_fixed_action_table(decklist, token_card_defs=token_defs, pending_kinds=pending_kinds)
    fixed_names = [name for name, _l, _e in fixed_table]
    assert not any(name.startswith(_TARGETING_PREFIXES) for name in fixed_names), (
        "fixed table must contain zero targeting actions"
    )
    assert "Pass" in fixed_names
    assert any(name.startswith("Play land:") for name in fixed_names)
    assert any(name.startswith("Cast ") for name in fixed_names)
    print(f"rl.action_bridge build_fixed_action_table self-check: OK ({len(fixed_table)} fixed actions)")

    # pointer_legal_mask / execute_pointer_choice against a REAL 2-PLAYER
    # game (build_token_set is inherently 2-player -- it always indexes an
    # opponent seat, so a 1-player game state doesn't fit its
    # contract at all; confirmed the hard way, see this module's own git
    # history for the IndexError that caught it). Mirror match, one shared
    # choose_action dispatching on state.active_idx. Always prefers a pointer action when
    # one is legal (Attack/Assign Blocker/Choose opponent's), else falls
    # back to the fixed table's own "prefer Pass, else first legal" rule --
    # exercises attacking, blocking, AND the cross-player "Choose
    # opponent's" consult nested inside blocking, across real turns.
    import random
    from rl.features import CardVocab, build_token_set

    vocab = CardVocab([decklist], token_card_defs=token_defs)
    pass_action = next(i for i, (name, _l, _e) in enumerate(fixed_table) if name == "Pass")
    saw_attack, saw_block, saw_choose_opponent = [False], [False], [False]

    def make_choose_action(rng):
        def choose_action(state):
            seat = state.active_idx
            pend = state.pending_resolution
            if pend is not None and pend["kind"] in ("mulligan_decision", "mulligan_bottom"):
                # Pregame is no longer in the fixed table (owned by the mulligan model
                # in real play -- the harness refactor). Keep every hand here so
                # this self-check's random fixed/pointer play still reaches real turns.
                return lambda state=state: game.execute_mulligan_keep(state)
            if any_pointer_legal(state):
                tokens = build_token_set(state, seat, vocab)
                identities_row = [ident for _idx, _row, ident in tokens]
                mask = pointer_legal_mask(state, identities_row)
                legal = [p for p, ok in zip(identities_row, mask) if ok]
                if legal:
                    chosen = rng.choice(legal)  # randomize which target too, not always the first -- more varied lines
                    pending = state.pending_resolution
                    if pending is None:
                        saw_attack[0] = True
                    elif pending["kind"] == "declare_blockers":
                        saw_block[0] = True
                    elif pending["kind"] == "choose_opponent_permanent":
                        saw_choose_opponent[0] = True
                    return lambda state=state, chosen=chosen: execute_pointer_choice(state, chosen)

            # Prefer any NON-Pass legal action (randomly) so the game
            # actually develops a board -- always preferring Pass would
            # mean lands/creatures never get played at all, and combat
            # would never happen for this self-check to exercise. Pass is
            # the fallback of last resort, not the default preference
            # (that inversion IS a bug this test already caught once, see
            # git history: the first version always-preferred Pass and
            # never once reached combat in 30 turns).
            #
            # Legality via ONE drl_env.legal_action_mask sweep, never by
            # calling each action's own legal_fn independently -- also
            # confirmed the hard way: _tap_cost_options_cache is explicitly
            # documented (drl_env) as "valid only for the duration of
            # one legal_action_mask sweep," and calling legal_fn ad hoc,
            # action by action, outside that sweep produced a real crash
            # (max() on an empty sequence inside execute_tap_cost_option)
            # from exactly that staleness.
            mask = drl_env.legal_action_mask(state, fixed_table)
            non_pass_indices = [i for i, ok in enumerate(mask) if ok and i != pass_action]
            if non_pass_indices:
                i = rng.choice(non_pass_indices)
                execute_fn = fixed_table[i][2]
                return lambda state=state, execute_fn=execute_fn: execute_fn(state)
            if mask[pass_action]:
                return None
            raise AssertionError("no fixed action legal and no pointer target legal -- the model would be stuck")
        return choose_action

    # Random unguided play doesn't reliably reach every combat sub-case in
    # any ONE game (declare_blockers only yields a real decision if attacks
    # were declared AND the defender happens to have an eligible blocker
    # alive at that moment) -- confirmed the hard way: a single 30-turn
    # game reached Attack but never Assign-Blocker. Retrying across
    # several independent games is the robust fix, not enlarging one game
    # and hoping.
    for seed in range(10):
        if saw_attack[0] and saw_block[0] and saw_choose_opponent[0]:
            break
        game.run_multiplayer_game(
            decklists=[decklist, decklist],
            rng=random.Random(seed), starting_player_idx=seed % 2, choose_action=make_choose_action(random.Random(seed)),
            horizon=40, combat_enabled=True,
        )
    assert saw_attack[0], "expected at least one real declare-attackers window across 10 real 2p games"
    assert saw_block[0], "expected at least one real declare-blockers window across 10 real 2p games"
    assert saw_choose_opponent[0], "expected at least one real cross-player 'Choose opponent's' consult across 10 real 2p games"
    print("rl.action_bridge pointer_legal_mask/execute_pointer_choice self-check: OK "
          "(real 2-player game -- attack, block, and cross-player targeting all exercised)")
