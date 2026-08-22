"""A hand-authored, non-learned opponent -- the gauntlet mechanism's tier-1
member (see README's Gauntlet section).

Imports rl.decision.agent for its decision plumbing (_build_decision,
_executor_for, _is_pass, AlwaysKeep); the dependency is one-way, so nothing
in the trained path depends on the heuristic."""

import numpy as np

import game
from rl.decision.agent import (PREGAME_KINDS, AlwaysKeep, DecisionResult, _build_decision,
                                _executor_for, _is_pass)


def _card_name_from_cast_action(name):
    """"Cast X" / "Cast X (free)" / "Cast X (plotted)" / ... -> X, matching
    drl_env.build_action_table's naming. Only called on a name already known
    to start with "Cast "."""
    rest = name[len("Cast "):]
    paren = rest.find(" (")
    return rest[:paren] if paren != -1 else rest


def _cmc(card_def):
    return sum(card_def.cast_cost.values()) if card_def.cast_cost else 0


def _fixed_tier(name):
    """Rough priority for a legal fixed-table action when no pointer rule
    applies: land drop first, then a cast (ties broken by CMC -- "cast the
    highest-cost thing affordable"), then any other fixed action, Pass as
    last resort."""
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
    """A flying attacker can only be blocked by a flying blocker (the one
    block restriction beyond raw eligibility)."""
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
    """"Block to kill when possible": does this already block-eligible
    blocker have enough power to kill at least one of the opponent's
    declared attackers. Deliberately rough -- picking a legal-but-redundant
    blocker is at worst wasted, never illegal, since the engine's own mask
    restricts which pairings are offered."""
    return any(
        _could_be_blocked_by(state, attacker, blocker)
        and game.permanent_power(state, blocker) >= game.permanent_toughness(state, attacker)
        for attacker in state.opponent.attackers
    )


def _heuristic_action_index(state, dec):
    """Picks one legal index per the owner's rough rule set: play a land if
    possible; else cast the highest-cost thing affordable; attack only if
    safe or a fair-or-better trade; block to kill when possible; else pass.
    Any pointer category with no owner rule (a resolving spell's own target,
    a sacrifice cost, ...) falls back to the first legal option."""
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
        # Usually "which attacker does my just-assigned blocker hit" --
        # prefer the biggest threat among what's offered.
        return max(pointer_legal, key=lambda i: game.permanent_power(state, dec.identities[i - n_fixed]))

    if fixed_legal:
        return _best_fixed_index(dec.fixed_table, fixed_legal)
    return int(legal[0])  # a pure pointer decision with no fixed fallback at all -- first legal, deterministic


class HeuristicAgent:
    """A hand-authored, non-learned opponent -- the gauntlet mechanism's
    tier-1 member (see README's Gauntlet section). Reuses SeatAgent's
    legal-action machinery (_build_decision, _executor_for) but scores among
    legal choices by fixed, general-principle rules instead of a trained
    policy. Pregame (mulligan) delegates to AlwaysKeep."""

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
