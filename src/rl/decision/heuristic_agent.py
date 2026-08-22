"""A hand-authored, non-learned opponent -- the gauntlet mechanism's tier-1
member (see README's Gauntlet section).

Split out of rl/decision/agent.py 2026-08-17: agent.py is the LEARNED path (feature
extraction, the policy's decision plumbing, SeatAgent), and a rules-based
opponent that merely reuses that plumbing does not belong in the same file.
The dependency is one-way -- this module imports rl.decision.agent, never the reverse --
so nothing in the trained path can come to depend on the heuristic.
"""

import numpy as np

import game
from rl.decision.agent import (PREGAME_KINDS, AlwaysKeep, DecisionResult, _build_decision,
                                _executor_for, _is_pass)


def _card_name_from_cast_action(name):
    """"Cast X" / "Cast X (free)" / "Cast X (plotted)" / "Cast X (omen)" /
    "Cast X (prototype)" -> X, matching drl_env.build_action_table's own
    f"Cast {name}" / f"Cast {name} ({suffix})" naming. Only ever called on a
    name already known to start with "Cast " -- a parenthesized suffix is the
    only other thing that can follow the card name in that scheme."""
    rest = name[len("Cast "):]
    paren = rest.find(" (")
    return rest[:paren] if paren != -1 else rest


def _cmc(card_def):
    return sum(card_def.cast_cost.values()) if card_def.cast_cost else 0


def _fixed_tier(name):
    """Rough priority for a legal fixed-table action when no pointer rule
    applies: a land drop first, then a cast (ties broken by CMC below --
    "cast the highest-cost thing you can afford"; legality already means
    affordable), then anything else fixed (Activate, Forestcycle, Decline,
    Abandon payment, ...), Pass only as an actual last resort -- the owner's
    own rule order ends "...; else pass"."""
    if name.startswith("Play land: "):
        return 3
    if name.startswith("Cast "):
        return 2
    if name == "Pass":
        return 0
    return 1


def _best_fixed_index(fixed_table, fixed_legal):
    def key(i):
        name = fixed_table[i][0]
        cmc = _cmc(game.CARD_DEFS[_card_name_from_cast_action(name)]) if name.startswith("Cast ") else 0
        return (_fixed_tier(name), cmc)
    return max(fixed_legal, key=key)


def _fixed_index_named(fixed_table, fixed_legal, name):
    for i in fixed_legal:
        if fixed_table[i][0] == name:
            return i
    return None


def _could_be_blocked_by(state, attacker, blocker):
    """The one block restriction this engine encodes beyond raw eligibility
    (declare_blocker_assignment's own extra_predicate; see action_bridge.py's
    execute_pointer_choice for the same check on the real assignment path):
    a flying attacker can only be blocked by a flying blocker."""
    return not game.has_keyword(state, attacker, "flying") or game.has_keyword(state, blocker, "flying")


def _attack_is_worthwhile(state, attacker):
    """Owner's rule: attack if the creature is SAFE (nothing can kill it).
    If not safe, attack only if the CHEAPEST creature that can profitably
    kill it costs at least as much mana as the attacker itself -- an
    equal-or-worse-for-them trade is fine, a strictly cheaper answer is not."""
    my_toughness = game.permanent_toughness(state, attacker)
    threats = [
        p for p in state.opponent.battlefield
        if p.card_def.card_type == game.CardType.CREATURE and not p.tapped
        and _could_be_blocked_by(state, attacker, p)
        and game.permanent_power(state, p) >= my_toughness
    ]
    if not threats:
        return True
    my_cmc = _cmc(attacker.card_def)
    cheapest_threat_cmc = min(_cmc(p.card_def) for p in threats)
    return cheapest_threat_cmc >= my_cmc


def _blocker_worth_assigning(state, blocker):
    """"Block to kill when possible" -- does this potential blocker
    (state.battlefield, already block-eligible per the engine's own mask)
    have enough power to kill at least one of the opponent's currently
    declared attackers. Deliberately rough: doesn't re-derive already-
    assigned/gang-block state itself -- the engine's own legal-action mask
    is what actually restricts which blocker/attacker pairings are offered,
    so picking a legal-but-redundant blocker here is at worst a wasted
    action, never an illegal or crashing one."""
    return any(
        _could_be_blocked_by(state, attacker, blocker)
        and game.permanent_power(state, blocker) >= game.permanent_toughness(state, attacker)
        for attacker in state.opponent.attackers
    )


def _heuristic_action_index(state, dec):
    """Picks one legal index (fixed or pointer half) per the owner's rough
    rule set: play a land if possible; else cast the highest-cost thing
    affordable; attack a creature only if safe or a fair-or-better trade;
    block to kill when possible; else pass. Any pointer category outside
    attack/block/"which attacker does my just-assigned blocker hit" (a
    resolving spell's own target choice, a sacrifice cost, ...) has no rule
    from the owner -- falls back to the first legal option, deterministic
    rather than guessed-at, since this bot was only asked to cover general
    play principles, not reimplement optimal targeting for every card."""
    legal = np.flatnonzero(dec.full_mask)
    n_fixed = dec.n_fixed
    fixed_legal = [i for i in legal if i < n_fixed]
    pointer_legal = [i for i in legal if i >= n_fixed]
    pending = state.pending_resolution

    if (pending is None and pointer_legal and state.phase is game.turn.Phase.DECLARE_ATTACKERS
            and state.active_idx == state.turn_player_idx):
        worthwhile = [i for i in pointer_legal if _attack_is_worthwhile(state, dec.identities[i - n_fixed])]
        if worthwhile:
            return max(worthwhile, key=lambda i: game.permanent_power(state, dec.identities[i - n_fixed]))
        named = _fixed_index_named(dec.fixed_table, fixed_legal, "Pass")
        if named is not None:
            return named

    elif pending is not None and pending["kind"] == "declare_blockers" and pointer_legal:
        worthwhile = [i for i in pointer_legal if _blocker_worth_assigning(state, dec.identities[i - n_fixed])]
        if worthwhile:
            return max(worthwhile, key=lambda i: game.permanent_power(state, dec.identities[i - n_fixed]))
        named = _fixed_index_named(dec.fixed_table, fixed_legal, "Done blocking")
        if named is not None:
            return named

    elif pending is not None and pending["kind"] == "choose_opponent_permanent" and pointer_legal:
        # The common real use of this primitive here is "which of my just-
        # assigned blocker's legal attackers does it block" -- prefer the
        # biggest threat among whatever's actually offered (already filtered
        # to what THIS specific blocker may legally block).
        return max(pointer_legal, key=lambda i: game.permanent_power(state, dec.identities[i - n_fixed]))

    if fixed_legal:
        return _best_fixed_index(dec.fixed_table, fixed_legal)
    return int(legal[0])  # a pure pointer decision with no fixed fallback at all -- first legal, deterministic


class HeuristicAgent:
    """A hand-authored, non-learned opponent -- the gauntlet mechanism's
    tier-1 member (see README's Gauntlet section). Reuses the SAME legal-
    action machinery a trained SeatAgent's main policy does (_build_decision,
    _executor_for) but scores among the legal choices by the owner's own
    rough, general-principle rules instead of a trained policy: a fixed,
    non-self-play reference for catching population-wide blind spots a
    hand-authored, never-adapting strategy wouldn't share. Pregame (mulligan)
    delegates to AlwaysKeep -- no heuristic pregame logic was asked for."""

    def __init__(self, deck_ctx):
        self.deck_ctx = deck_ctx
        self.mulligan = AlwaysKeep()

    def decide(self, state, seat, horizon, device, greedy=False):
        pend = state.pending_resolution
        if pend is not None and pend["kind"] in PREGAME_KINDS:
            return DecisionResult(self.mulligan.decide(state), None, None, is_pass=False)
        dec = _build_decision(state, seat, self.deck_ctx, horizon)
        action_idx = dec.sole_action if dec.sole_action is not None else _heuristic_action_index(state, dec)
        executor = _executor_for(state, action_idx, dec.fixed_table, dec.identities)
        return DecisionResult(executor, None, None, _is_pass(action_idx, dec.fixed_table))
