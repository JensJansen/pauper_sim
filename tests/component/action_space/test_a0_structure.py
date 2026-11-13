"""Layer A0 -- static structure of the action space, no games played.

  * every table has unique labels, and the production table is the full table minus exactly the pointer-routed rows;
  * every card's registry spec produces exactly the rows it should (the expected set is derived HERE, from the registry,
    not from the table builder -- so the order pin in test_action_table_order pins the right thing);
  * every pending-resolution kind the engine can open has an answer path (a table row, a pointer kind, or the
    MulliganNet), and the hand-kept pointer dispatches agree.
"""
import ast
from types import SimpleNamespace

import pytest

import drl_env
import game
from rl.decision import action_bridge as AB
from rl.roster import TOKEN_DEFS, build_pool
from support.paths import SRC

PREGAME = {"mulligan_decision", "mulligan_bottom"}  # owned by MulliganNet, deliberately not in the table
# Kinds begun through a helper that takes the kind as a variable (begin_scry_surveil), so the AST scan cannot read them.
VARIABLE_KINDS = {"scry", "surveil"}
# Kinds opened by code that builds the pending dict without a begin_resolution literal.
OPENED_ELSEWHERE = {"choose_any_target", "choose_cast_copy"}


@pytest.fixture(scope="module")
def pool():
    mp = pytest.MonkeyPatch()
    mp.chdir(SRC)  # rl.roster loads "../data/..." relative to src/
    try:
        yield build_pool()
    finally:
        mp.undo()


def test_labels_are_unique_in_every_roster_table(pool):
    _decks, _vocab, _ctxs, tables = pool
    for deck, table in tables.items():
        labels = [row[0] for row in table]
        dup = sorted({lab for lab in labels if labels.count(lab) > 1})
        assert not dup, f"{deck}: duplicate row labels {dup[:5]}"


def test_production_table_is_the_full_table_minus_pointer_rows(pool):
    decks = pool[0]
    for deck, decklist in decks.items():
        full = drl_env.build_action_table(decklist, game.EFFECT_REGISTRY, token_card_defs=tuple(TOKEN_DEFS))
        kept = AB.build_fixed_action_table(decklist, token_card_defs=tuple(TOKEN_DEFS))
        dropped = [r[0] for r in full if r[0].startswith(AB._TARGETING_PREFIXES)]
        assert [r[0] for r in kept] == [r[0] for r in full if not r[0].startswith(AB._TARGETING_PREFIXES)], deck
        assert dropped, f"{deck}: no targeting rows at all -- the prefix list no longer matches the table"


# ---- registry -> rows ------------------------------------------------------------------------------------------------

def expected_rows(name):
    """Row labels card `name` must contribute, derived only from its registry spec."""
    card_def = game.CARD_DEFS[name]
    spec = game.EFFECT_REGISTRY.get(card_def.effect_id, {})
    out = set()
    if card_def.card_type == game.CardType.LAND:
        out.add(f"Play land: {name}")
    if "cast" in spec or "cast_modes" in spec or "x_cast_modes" in spec or "delve" in spec:
        out.add(f"Cast {name}")
    if "alt_cast" in spec:
        out.add(f"Cast {name} (free)")
    for key, label in (("flashback", "Flashback"), ("escape", "Escape")):
        if key in spec:
            out.add(f"{label} {name}")
    if "plot" in spec:
        out |= {f"Plot {name}", f"Cast {name} (plotted)"}
    if "omen" in spec:
        out.add(f"Cast {spec['omen']['card_def'].name} (omen)")
    if "prototype" in spec:
        out.add(f"Cast {spec['prototype']['card_def'].name} (prototype)")
    if "cycle" in spec:
        out.add(f"Cycle {name}")
    if "forestcycle" in spec:
        out.add(f"Forestcycle {name}")
    if "graveyard_ability" in spec:
        out.add(f"Activate {name} (graveyard)")
    for ability in spec.get("activated_abilities", {}):
        out.add(f"Activate {name} ({ability})")
    if "mana_extra_choose" in spec:
        out.add(f"Tap {name}")
    elif "mana" in spec:
        kind = spec["mana"][0]
        if kind != "flexible":
            out.add(f"Tap {name}")
        else:
            out |= {f"Tap {name} for {c}" for c in spec["mana"][1]}
    if card_def.card_type == game.CardType.LAND:
        grantable = set().union(*(s.get("grants_mana_colors", set()) for s in game.EFFECT_REGISTRY.values()))
        out |= {f"Tap {name} for {c}" for c in grantable}
    if "filter_mana" in spec:
        out |= {f"Filter {name}, paying {c}" for c in game.POOL_COLORS}
    return out


_DERIVED_FAMILIES = ("Play land:", "Cast ", "Flashback ", "Escape ", "Plot ", "Cycle ", "Forestcycle ", "Activate ", "Tap ",
                     "Filter ")


def _impulse_rows(name):
    card_def = game.CARD_DEFS[name]
    spec = game.EFFECT_REGISTRY.get(card_def.effect_id, {})
    if card_def.card_type == game.CardType.LAND:
        return {f"Play from exile: {name}"}
    out = set()
    if "cast" in spec:
        out.add(f"Play from exile: {name}")
    out |= {f"Play from exile: {name} ({m})" for m in (spec.get("cast_modes") or {})}
    return out


def test_every_card_alone_gets_exactly_the_rows_its_registry_spec_implies():
    baseline = {r[0] for r in AB.build_fixed_action_table([("Forest", 1)], token_card_defs=tuple(TOKEN_DEFS))}
    problems = []
    for name in sorted(game.CARD_DEFS):
        table = {r[0] for r in AB.build_fixed_action_table([(name, 2)], token_card_defs=tuple(TOKEN_DEFS))}
        want = expected_rows(name)
        impulse = "impulse" in game.derive_pending_kinds([(name, 2)])
        if impulse:
            want |= _impulse_rows(name)
        missing = want - table
        derived = {lab for lab in table if lab.startswith(_DERIVED_FAMILIES) and not lab.startswith("Cast (")
                   and lab not in baseline}  # "Cast (madness)" / "Cast (may)" answer a pending question
        orphan = derived - want
        if missing or orphan:
            problems.append(f"{name}: missing {sorted(missing)[:4]} orphan {sorted(orphan)[:4]}")
    assert not problems, "table rows and registry specs disagree:\n  " + "\n  ".join(problems)


# ---- pending kinds ---------------------------------------------------------------------------------------------------

def _literal_kinds():
    kinds = set()
    for path in (SRC / "game").rglob("*.py"):
        if "__pycache__" in path.parts:
            continue
        for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
            is_begin = isinstance(node, ast.Call) and getattr(node.func, "id", getattr(node.func, "attr", None)) == "begin_resolution"
            if is_begin and len(node.args) >= 2 and isinstance(node.args[1], ast.Constant) and isinstance(node.args[1].value, str):
                kinds.add(node.args[1].value)
    return kinds


def _answer_paths(pool):
    """(kinds some table row is gated on, kinds the pointer head answers), from the table's own gate stamps."""
    decks = pool[0]
    gates = set()
    for decklist in decks.values():
        full = drl_env.build_action_table(decklist, game.EFFECT_REGISTRY, token_card_defs=tuple(TOKEN_DEFS),
                                          opponent_decklist=decklist, opponent_token_card_defs=tuple(TOKEN_DEFS))
        for _label, legal, _ex in full:
            gate = getattr(legal, "_pending_gate", None)
            if isinstance(gate, (set, frozenset)):
                gates |= set(gate)
    pointer = {"declare_blockers", "choose_permanent", "choose_opponent_permanent", "assign_combat_damage",
               "choose_any_target"} | set(AB._ID_MATCHED_KINDS)
    return gates, pointer


def test_every_pending_kind_has_an_answer_path(pool):
    kinds = _literal_kinds() | VARIABLE_KINDS | OPENED_ELSEWHERE
    gates, pointer = _answer_paths(pool)
    unanswered = sorted(k for k in kinds - PREGAME - {"pay_unless"} if k not in gates and k not in pointer)
    assert not unanswered, (f"pending kinds with no table row gated on them and no pointer kind: {unanswered}. "
                            f"A decision of that kind would leave the mask all-False.")
    assert "pay_unless" in gates and "pay_cost" in gates  # answered by Pay/Don't pay and by Tap + Spend rows


def test_the_kind_ledger_has_no_stale_entries(pool):
    """Every kind a row is gated on or the pointer head handles must still be opened somewhere in the engine."""
    kinds = _literal_kinds() | VARIABLE_KINDS | PREGAME | OPENED_ELSEWHERE
    gates, pointer = _answer_paths(pool)
    stale = sorted((gates | pointer) - kinds)
    assert not stale, f"rows/pointer handle kinds the engine never opens: {stale}"


def test_pointer_dispatches_agree_on_every_kind():
    """any_pointer_legal and pointer_kind are two hand-kept copies of one dispatch. For every kind, 'is a pointer
    decision' must be the same answer from both."""
    for kind in sorted(_literal_kinds() | VARIABLE_KINDS | OPENED_ELSEWHERE):
        state = SimpleNamespace(active_mana_subdecision=None, pending_resolution={"kind": kind},
                                phase=game.turn.Phase.MAIN1, active_idx=0, turn_player_idx=0)
        assert AB.any_pointer_legal(state) == (AB.pointer_kind(state) is not None), kind
    for phase in game.turn.Phase:
        state = SimpleNamespace(active_mana_subdecision=None, pending_resolution=None, phase=phase,
                                active_idx=0, turn_player_idx=0)
        assert AB.any_pointer_legal(state) == (AB.pointer_kind(state) is not None), phase
    for stage, expect in (("choose_target", True), ("choose_color", False)):
        state = SimpleNamespace(active_mana_subdecision={"stage": stage}, pending_resolution=None,
                                phase=game.turn.Phase.MAIN1, active_idx=0, turn_player_idx=0)
        assert AB.any_pointer_legal(state) is expect and (AB.pointer_kind(state) is not None) is expect, stage
