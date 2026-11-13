"""Independent legality oracle for the fixed action table.

`oracle_labels(state)` returns the set of table-row LABELS ("Cast Lotus Petal", "Tap Forest for G", "Pass", ...) that
are legal for the player who holds priority, derived from the rules and the card data (registry specs, costs, zones),
NOT from drl_env's legal() closures or game.plan_payment. The label vocabulary is the one convention shared with the
table; everything else is restated here.

What is deliberately independent
  * timing (sorcery / your-turn / instant), land drop, one-copy-per-hand-card cast guard;
  * affordability: a brute-force search over what each mana source can produce (`payable`), not Hall's theorem;
  * the per-card source model (`source_alternatives`), including either/or ability grants that the engine drops;
  * the strand rule while a payment is open (a tap is legal only if the payment can still finish).

What is shared with the engine (listed so nobody mistakes the oracle for more than it is)
  * per-card DATA read from the registry (costs, speeds, `extra_legal`/`legal` predicates, cost-reduction lambdas);
  * game.effects.stats.has_haste, game.has_unfulfilled_goad, and the pending-kind option functions used for "Choose: X";
  * the label strings themselves.

Rules that differ from the real game on purpose are listed in AUTHORIZED / ASSUMPTIONS below. The oracle follows the
authorized ones (so it does not report them) and reports everything else.
"""
import re

import game
from game.cards import CardType, EffectId

REG = game.EFFECT_REGISTRY
COLORS = ("W", "U", "B", "R", "G")
POOL = COLORS + ("C",)

# Deviations the owner has authorized. The oracle follows these on purpose; each is a line the owner can strike.
AUTHORIZED = {
    "speculative-float-main-phase-only":
        "Floating mana with nothing to pay for is legal only in the active player's own main phase "
        "(owner-approved 2026-08-17, drl_env/_actions_mana.py::_mana_timing_legal).",
    "saruli-supply-excluded":
        "A mana_extra_choose source (Saruli Caretaker) is never counted as free supply when judging affordability "
        "(owner-accepted 2026-08-17, game/mana.py::source_mana_units).",
    "strand-rule":
        "While a payment is open, a mana ability is offered only if the remaining cost can still be paid afterwards "
        "(game/mana.py STRANDING INVARIANT). Real Magic lets you tap for the wrong color and rewinds (732) "
        "(owner-accepted 2026-11-12).",
    "no-undo-blocking":
        "Declare-blockers is linearized, one blocker at a time, with no undo (owner 2026-07-31).",
}
# Engine behaviours the oracle mirrors but that are NOT marked authorized in the source. Each is a question for the
# owner; the oracle follows the engine here so it does not drown real findings in known noise.
ASSUMPTIONS = {}  # engine behaviours mirrored but not authorized: none open (the strand rule is authorized, see above)

# Findings: engine behaviours that differ from the real rules and are not authorized. Each is a MODE the oracle can switch
# on to mimic the engine, so a disagreement can be attributed to a named, owner-visible question rather than hidden.
# None are open. Fixed 2026-11-12 by owner decision: F-01 cross-controller Auras, F-02 same-name copies, F-03 a granted
# ability on a multi-symbol source dropped from affordability (game.mana.EitherOr). The ledger lives in
# tests/component/action_space/test_a2_oracle.py.
MODES = frozenset()

COVERED = {"pass", "play_land", "cast", "gy_cast", "plot", "activate", "cycle", "impulse", "tap", "spend"}


def family(label, extra_tap_names=()):
    if label == "Pass":
        return "pass"
    if label.startswith("Play land:"):
        return "play_land"
    if label.startswith("Play from exile:"):
        return "impulse"
    if label.startswith(("Flashback ", "Escape ")):
        return "gy_cast"
    if label.startswith("Plot "):
        return "plot"
    if label.startswith("Cast ("):
        return "decision"  # "Cast (madness)", "Cast (may)": answers to a pending question, not a spell cast
    if label.startswith("Cast "):
        return "cast"
    if label.startswith("Activate "):
        return "activate"
    if label.startswith(("Forestcycle ", "Cycle ")):
        return "cycle"
    if label.startswith("Spend "):
        return "spend"
    if label.startswith("Tap "):
        name = re.sub(r" for [WUBRGC]$", "", label[4:])
        return "tap_extra" if name in extra_tap_names else "tap"
    if label.startswith("Filter "):
        return "filter"
    if label.startswith("Produce "):
        return "produce"
    if label.startswith(("Mode ", "X=", "Delve ")):
        return "cast_button"
    if label.startswith("Choose: "):
        return "choose_name"
    return "decision"


# ---- affordability: brute force over what every source can produce ---------------------------------------------------

def _cost_needs(cost):
    need = {c: cost.get(c, 0) for c in POOL if cost.get(c, 0) > 0}
    return need, cost.get("generic", 0)


def payable(pool, sources, cost):
    """Can `cost` be paid from the floating `pool` (list of colour symbols) plus `sources` (a list; each entry is a list of
    ALTERNATIVES, each alternative a list of symbols produced together by that source)? Exact: it tries every
    alternative of every source, applying each produced symbol to its own coloured need first (never worse than
    spending it on generic), and asks whether some combination zeroes the cost."""
    need, generic = _cost_needs(cost)
    colors = tuple(need)

    def apply(state, symbols):
        left, gen = list(state[0]), state[1]
        for s in symbols:
            if s in colors and left[colors.index(s)] > 0:
                left[colors.index(s)] -= 1
            elif gen > 0:
                gen -= 1
        return (tuple(left), gen)

    frontier = {apply((tuple(need[c] for c in colors), generic), pool)}
    for alts in sources:
        frontier = {apply(st, alt) for st in frontier for alt in alts}
        if any(not any(st[0]) and st[1] == 0 for st in frontier):
            return True
    return any(not any(st[0]) and st[1] == 0 for st in frontier)


def _me(state):
    return state.players[state.active_idx]


def _enchanting_auras(state, p):
    return [a for pl in state.players for a in pl.battlefield if a.flags.get("enchanting") is p]


def _tron_online(me):
    present = {q.card_def.extra["tron_type"] for q in me.battlefield if q.card_def.effect_id == EffectId.TRON_LAND}
    return {"Mine", "Power Plant", "Tower"} <= present


def source_gate_open(state, p):
    """Untapped (unless its ability has no {T}) and not summoning-sick (for a creature whose ability taps)."""
    entry = REG.get(p.card_def.effect_id, {})
    no_tap = entry.get("mana_no_tap", False)
    if p.tapped and not no_tap:
        return False
    if p.card_type == CardType.CREATURE and not no_tap and p.summoning_sick:
        from game.effects.stats import has_haste
        if not has_haste(state, p):
            return False
    return True


def source_alternatives(state, p, exclude=()):
    """What `p` could add if tapped for mana right now, as a list of alternatives (see `payable`), or None when it
    cannot be tapped. Follows the card data: native ability, plus either/or grants from Auras, plus bonus symbols that
    ride along with every tap."""
    entry = REG.get(p.card_def.effect_id, {})
    if entry.get("mana_extra_choose") is not None:  # AUTHORIZED saruli-supply-excluded
        return None
    if not source_gate_open(state, p):
        return None
    extra = entry.get("mana_extra_available")
    if extra is not None and not extra(state, p):
        return None
    auras = _enchanting_auras(state, p)
    bonus = [a.flags["bonus_mana_color"] for a in auras if "bonus_mana_color" in a.flags]
    granted = set()
    for a in auras:
        granted |= a.flags.get("bonus_mana_colors", set())
    spec = entry.get("mana")
    me = _me(state)
    alts = []
    if spec is not None:
        kind = spec[0]
        if kind == "fixed":
            alts.append([spec[1]])
        elif kind == "fixed_multi":
            alts.append(list(spec[1]))
        elif kind == "flexible":
            alts.extend([c] for c in spec[1])
        elif kind == "tron":
            alts.append(["C"] * (3 if p.card_def.extra["tron_type"] == "Tower" else 2) if _tron_online(me) else ["C"])
        elif kind == "count":
            alts.append([spec[1]] * sum(1 for q in me.battlefield if q not in exclude and spec[2](q)))
        elif kind == "count_all":
            alts.append([spec[1]] * sum(1 for pl in state.players for q in pl.battlefield if q not in exclude and spec[2](q)))
    alts.extend([c] for c in sorted(granted))
    if not alts:
        return None
    return [alt + bonus for alt in alts]


def _pool_symbols(me):
    return [c for c, n in me.mana_pool.items() for _ in range(n)]


def _all_sources(state, exclude=(), skip=()):
    me = _me(state)
    out = []
    for p in me.battlefield:
        if p in skip or p in exclude:
            continue
        alts = source_alternatives(state, p, exclude=exclude)
        if alts is not None:
            out.append(alts)
    return out


def can_afford(state, cost, modes=()):
    """Can the active player pay `cost`? `modes`: engine deviations to mimic (see MODES); default is the real rule."""
    global MODES
    prev, MODES = MODES, frozenset(modes)
    try:
        return payable(_pool_symbols(_me(state)), _all_sources(state), cost)
    finally:
        MODES = prev


# ---- timing -------------------------------------------------------------------------------------------------------

def _speed_name(speed):
    return speed.value if hasattr(speed, "value") else str(speed)


def timing_ok(state, speed):
    s = _speed_name(speed)
    mine = state.active_idx == state.turn_player_idx
    if s == "instant":
        return True
    if s == "your_turn":
        return mine
    return mine and state.phase is not None and state.phase.value in ("main1", "main2") and not state.stack


def default_speed(card_def, spec):
    if spec.get("speed") is not None:
        return spec["speed"]
    return game.turn.Speed.INSTANT if card_def.card_type == CardType.INSTANT else game.turn.Speed.SORCERY


def effective_cost(state, card_def):
    """Cast cost after the card's own generic-cost reduction (coloured pips are never reduced)."""
    cost = card_def.cast_cost
    red = REG.get(card_def.effect_id, {}).get("cost_reduction")
    if red is None or cost is None:
        return cost
    n = red(state)
    if n <= 0:
        return cost
    out = dict(cost)
    out["generic"] = max(0, out.get("generic", 0) - n)
    return out


# ---- the oracle ----------------------------------------------------------------------------------------------------

def oracle_labels(state, modes=()):
    """Legal labels, over the covered families, for the seat holding priority. `modes` (see MODES) makes the oracle mimic
    a known engine deviation; the default is the real rule."""
    global MODES
    prev, MODES = MODES, frozenset(modes)
    try:
        return _oracle_labels(state)
    finally:
        MODES = prev


def _oracle_labels(state):
    if state.active_mana_subdecision is not None:
        return set()  # exclusive priority: only the sub-decision's own buttons (family 'produce', not covered here)
    me = _me(state)
    pend = state.pending_resolution
    out = set()
    hand = list(me.hand)
    hand_names = [c.name for c in hand]

    # --- mana abilities and pool spending: usable under a payment, or speculatively in own main phase -------------
    mid_cast = pend is not None and (pend["kind"] in ("choose_cast_mode", "choose_cast_x", "choose_delve_amount",
                                                      "choose_cast_copy") or bool(pend.get("mid_cast")))
    paying = pend is not None and pend["kind"] in ("pay_cost", "pay_unless")
    own_main = (state.phase is not None and state.phase.value in ("main1", "main2")
                and state.active_idx == state.turn_player_idx)
    if not mid_cast and (paying or own_main):
        out |= _tap_labels(state, me, pend)
    if pend is not None and pend["kind"] == "pay_cost":
        rem = pend["remaining"]
        if any(v > 0 for v in rem.values()):
            gen = rem.get("generic", 0) > 0
            out |= {f"Spend {c} from pool" for c in POOL if me.mana_pool.get(c, 0) > 0 and (rem.get(c, 0) > 0 or gen)}

    if pend is not None:
        return out  # everything below is a top-level action: legal only with nothing pending

    if not (state.phase is game.turn.Phase.DECLARE_ATTACKERS and game.has_unfulfilled_goad(state)):
        out.add("Pass")

    # --- lands ------------------------------------------------------------------------------------------------------
    if timing_ok(state, game.turn.Speed.SORCERY) and state.lands_played_this_turn == 0:
        out |= {f"Play land: {n}" for n in set(hand_names) if game.CARD_DEFS[n].card_type == CardType.LAND}

    # --- casting and other hand/graveyard/exile actions -------------------------------------------------------------
    for name in sorted(set(hand_names)):
        out |= _hand_card_labels(state, me, name)
    for name in sorted({c.name for c in me.graveyard}):
        out |= _graveyard_labels(state, me, name)
    out |= _exile_labels(state, me)
    out |= _battlefield_ability_labels(state, me)
    return out


def _cast_spec_labels(state, name, spec_owner_name=None):
    """Labels for casting `name` from HAND via its normal registry shapes."""
    card_def = game.CARD_DEFS[name]
    spec = REG.get(card_def.effect_id, {})
    out = set()
    cast = spec.get("cast")
    if cast is not None:
        speed = default_speed(card_def, cast)
        if (timing_ok(state, speed) and can_afford(state, effective_cost(state, card_def))
                and (cast.get("extra_legal") is None or cast["extra_legal"](state))):
            out.add(f"Cast {name}")
    modes = spec.get("cast_modes")
    if modes is not None:
        items = list(modes.items())
        speed = default_speed(card_def, items[0][1])
        if timing_ok(state, speed):
            for _n, m in items:
                if m.get("extra_legal") is not None and not m["extra_legal"](state):
                    continue
                cost = m.get("cost")
                if cost is None:
                    cost = effective_cost(state, card_def)
                if can_afford(state, cost):
                    out.add(f"Cast {name}")
                    break
    xmodes = spec.get("x_cast_modes")
    if xmodes is not None:
        items = list(xmodes.items())
        speed = default_speed(card_def, items[0][1])
        if timing_ok(state, speed):
            for _n, m in items:
                if m.get("extra_legal") is not None and not m["extra_legal"](state):
                    continue
                for x in range(m["max_x"] + 1):
                    c = dict(m["cost"])
                    c["generic"] = c.get("generic", 0) + x
                    if can_afford(state, c):
                        out.add(f"Cast {name}")
                        break
                else:
                    continue
                break
    delve = spec.get("delve")
    if delve is not None and timing_ok(state, default_speed(card_def, delve)):
        gy = len(_me(state).graveyard)
        for n in range(min(delve["max"], gy) + 1):
            c = dict(card_def.cast_cost)
            c["generic"] = max(0, c.get("generic", 0) - n)
            if can_afford(state, c):
                out.add(f"Cast {name}")
                break
    alt = spec.get("alt_cast")
    if alt is not None and timing_ok(state, default_speed(card_def, alt)) and alt["extra_legal"](state):
        out.add(f"Cast {name} (free)")
    plot = spec.get("plot")
    if plot is not None and timing_ok(state, default_speed(card_def, plot)) and can_afford(state, plot["cost"]):
        out.add(f"Plot {name}")
    omen = spec.get("omen")
    if omen is not None and timing_ok(state, default_speed(omen["card_def"], omen)) and can_afford(state, omen["cost"]):
        out.add(f"Cast {omen['card_def'].name} (omen)")
    proto = spec.get("prototype")
    if proto is not None and timing_ok(state, default_speed(proto["card_def"], proto)) and can_afford(state, proto["cost"]):
        out.add(f"Cast {proto['card_def'].name} (prototype)")
    return out


def _hand_card_labels(state, me, name):
    out = _cast_spec_labels(state, name)
    card_def = game.CARD_DEFS[name]
    spec = REG.get(card_def.effect_id, {})
    for key, label in (("forestcycle", "Forestcycle"), ("cycle", "Cycle")):
        cyc = spec.get(key)
        if cyc is not None and can_afford(state, card_def.extra[cyc["cost_key"]]):
            out.add(f"{label} {name}")
    return out


def _graveyard_labels(state, me, name):
    out = set()
    card_def = game.CARD_DEFS[name]
    spec = REG.get(card_def.effect_id, {})
    for key, label in (("flashback", "Flashback"), ("escape", "Escape")):
        gy = spec.get(key)
        if gy is None or not timing_ok(state, default_speed(card_def, gy)):
            continue
        cost = gy.get("cost")
        if cost is not None and not can_afford(state, cost):
            continue
        if gy["legal"](state):
            out.add(f"{label} {name}")
    ga = spec.get("graveyard_ability")
    if ga is not None and can_afford(state, card_def.extra[ga["cost_key"]]):
        out.add(f"Activate {name} (graveyard)")
    return out


def _exile_labels(state, me):
    out = set()
    # plotted copies: cast free on a later turn
    for c, stamp in me.exile:
        if stamp is None or not stamp < state.turn_number:
            continue
        spec = REG.get(c.effect_id, {})
        plot, cast = spec.get("plot"), spec.get("cast")
        if plot is None or cast is None:
            continue
        if timing_ok(state, default_speed(c, plot)) and (cast.get("extra_legal") is None or cast["extra_legal"](state)):
            out.add(f"Cast {c.name} (plotted)")
    # impulse: "you may play the exiled cards" until the deadline
    live = {cd.name: cd for cd, deadline in me.impulse if state.turn_number <= deadline}
    for name, cd in live.items():
        spec = REG.get(cd.effect_id, {})
        if cd.card_type == CardType.LAND:
            if timing_ok(state, game.turn.Speed.SORCERY) and state.lands_played_this_turn == 0:
                out.add(f"Play from exile: {name}")
            continue
        cast = spec.get("cast")
        if cast is not None and timing_ok(state, default_speed(cd, cast)) \
                and can_afford(state, cd.cast_cost) and (cast.get("extra_legal") is None or cast["extra_legal"](state)):
            out.add(f"Play from exile: {name}")
        for mode, m in (spec.get("cast_modes") or {}).items():
            if timing_ok(state, default_speed(cd, m)) and can_afford(state, m.get("cost", cd.cast_cost)) \
                    and (m.get("extra_legal") is None or m["extra_legal"](state)):
                out.add(f"Play from exile: {name} ({mode})")
    return out


def _can_pay_tap_cost(state, p):
    """{T} in an activation's cost: the permanent must be untapped and, if a creature, not summoning-sick (302.6)."""
    if p.tapped:
        return False
    if p.card_type == CardType.CREATURE and p.summoning_sick:
        from game.effects.stats import has_haste
        return has_haste(state, p)
    return True


def _battlefield_ability_labels(state, me):
    """Activated abilities. Whether an ability's cost includes {T} is the registry's per-ability "taps" key (card data); the
    player may activate with ANY eligible copy."""
    out = set()
    by_name = {}
    for p in me.battlefield:
        by_name.setdefault(p.card_def.name, []).append(p)
    for name, perms in by_name.items():
        for ability, spec in REG.get(perms[0].card_def.effect_id, {}).get("activated_abilities", {}).items():
            if not timing_ok(state, spec.get("speed", game.turn.Speed.INSTANT)):
                continue
            eligible = [p for p in perms if not spec["taps"] or _can_pay_tap_cost(state, p)]
            if "cost_key" in spec:
                ok = any(can_afford(state, p.card_def.extra[spec["cost_key"]]) for p in eligible)
                ok = ok and (spec.get("extra_legal") is None or spec["extra_legal"](state))
            else:
                ok = any(spec["legal"](state, p) for p in eligible)
            if ok:
                out.add(f"Activate {name} ({ability})")
    return out


def _tap_labels(state, me, pend):
    """Every "Tap X [for c]" that is legal now. A tap is a mana ability: legal in any priority window, subject to the
    timing rule applied by the caller and, while a payment is open, to the strand rule. A label is legal if ANY
    permanent of that name can make that choice (the player picks which)."""
    remaining = pend["remaining"] if pend is not None and pend["kind"] == "pay_cost" else None
    pool = _pool_symbols(me)
    candidates = {}  # label -> [(permanent, produced symbols)] in battlefield order
    for p in me.battlefield:
        entry = REG.get(p.card_def.effect_id, {})
        if entry.get("mana_extra_choose") is not None:
            continue
        if source_alternatives(state, p) is None:
            continue
        spec = entry.get("mana")
        auras = _enchanting_auras(state, p)
        bonus = [a.flags["bonus_mana_color"] for a in auras if "bonus_mana_color" in a.flags]
        granted = set()
        for a in auras:
            granted |= a.flags.get("bonus_mana_colors", set())
        name = p.card_def.name
        if spec is not None:
            if spec[0] == "flexible":
                for c in spec[1]:
                    candidates.setdefault(f"Tap {name} for {c}", []).append((p, [c] + bonus))
            else:
                candidates.setdefault(f"Tap {name}", []).append((p, _native_output(state, me, p, spec) + bonus))
        for c in sorted(granted):
            candidates.setdefault(f"Tap {name} for {c}", []).append((p, [c] + bonus))
    out = set()
    for label, cands in candidates.items():
        for p, produced in cands:
            if remaining is None:
                out.add(label)
                break
            entry = REG.get(p.card_def.effect_id, {})
            dying = entry.get("mana_tap_removes_self") is not None and entry["mana_tap_removes_self"](state, p)
            rest = _all_sources(state, exclude={p} if dying else set(), skip={p})
            if payable(pool + produced, rest, remaining):
                out.add(label)
                break
    return out


def _native_output(state, me, p, spec):
    kind = spec[0]
    if kind == "fixed":
        return [spec[1]]
    if kind == "fixed_multi":
        return list(spec[1])
    if kind == "tron":
        return (["C"] * (3 if p.card_def.extra["tron_type"] == "Tower" else 2)) if _tron_online(me) else ["C"]
    if kind == "count":
        return [spec[1]] * sum(1 for q in me.battlefield if spec[2](q))
    if kind == "count_all":
        return [spec[1]] * sum(1 for pl in state.players for q in pl.battlefield if spec[2](q))
    raise ValueError(kind)
