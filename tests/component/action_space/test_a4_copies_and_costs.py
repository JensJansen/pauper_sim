"""Layer A4 -- which copy, and what an activation costs. Regression and cross-check tests for three fixes (2026-11-12):

  F-01  an Aura controlled by one player on the OTHER player's land still works for the land's controller;
  F-02  when several same-named permanents could tap / activate and they are NOT interchangeable (different Auras,
        counters, ...), the player chooses which (a choose_source mana sub-decision for taps, a choose_permanent
        decision for activations); when they are interchangeable nothing is asked;
  cost  an activated ability's {T} is declared per ability ("taps" in the registry), so Clue / Reckless Lackey / Relic's
        draw work on a tapped permanent and a {T} ability does not -- and the declaration is checked against what each
        ability's resolve actually does.
"""
import pytest

import drl_env
import game
from game.state import Permanent
from rl.decision import action_bridge as AB
from rl.roster import TOKEN_DEFS
from support import scenario as S

LANDS = [("Forest", 4), ("Plains", 4), ("Mountain", 4), ("Island", 4), ("Utopia Sprawl", 1), ("Abundant Growth", 1)]


def _table(decklist):
    return AB.build_fixed_action_table(decklist, token_card_defs=tuple(TOKEN_DEFS))


def _row(table, label):
    return next(i for i, r in enumerate(table) if r[0] == label)


def _board(me_bf, opp_bf=(), me_hand=(), phase=game.turn.Phase.MAIN1):
    """Seat 1 is the active player and owns the turn; seat 0 is the opponent."""
    state = S.build(S.seat(bf=opp_bf), S.seat(bf=me_bf, hand=me_hand))
    state.active_idx = state.turn_player_idx = 1
    state.phase = phase
    return state


def _mask(state, table):
    return drl_env.legal_action_mask(state, table)


# ---- F-01: Auras across the table ---------------------------------------------------------------------------------

def test_f01_utopia_sprawl_controlled_by_the_opponent_still_adds_its_bonus_mana():
    table = _table(LANDS)
    state = _board(["Forest"], opp_bf=[("Utopia Sprawl", {"aura_on": (1, 0)})])
    state.players[0].battlefield[0].flags["bonus_mana_color"] = "R"
    assert sorted(sorted(u) for u in game.available_mana_units(state)) == [["G"], ["R"]]
    assert _mask(state, table)[_row(table, "Tap Forest")]
    table[_row(table, "Tap Forest")][2](state)
    assert dict(state.mana_pool) == {"G": 1, "R": 1}


def test_f01_abundant_growth_controlled_by_the_opponent_grants_its_any_colour_ability():
    table = _table(LANDS)
    state = _board(["Plains"], opp_bf=[("Abundant Growth", {"aura_on": (1, 0)})])
    state.players[0].battlefield[0].flags["bonus_mana_colors"] = set(game.COLORS)
    assert ("Plains", "U") in game.mana_ability_options(state)
    assert _mask(state, table)[_row(table, "Tap Plains for U")]
    table[_row(table, "Tap Plains for U")][2](state)
    assert dict(state.mana_pool) == {"U": 1}


# ---- F-02: choosing the copy ---------------------------------------------------------------------------------------

def _two_forests_one_with_sprawl():
    state = _board(["Forest", "Forest"], opp_bf=[("Utopia Sprawl", {"aura_on": (1, 1)})])
    state.players[0].battlefield[0].flags["bonus_mana_color"] = "R"
    return state


def test_f02_interchangeable_copies_are_not_asked():
    table = _table(LANDS)
    state = _board(["Forest", "Forest"])
    table[_row(table, "Tap Forest")][2](state)
    assert state.mana_subdecision is None and dict(state.mana_pool) == {"G": 1}
    assert [p.tapped for p in state.players[1].battlefield] == [True, False]


def test_f02_different_copies_open_a_choose_source_decision_and_the_choice_matters():
    table = _table(LANDS)
    state = _two_forests_one_with_sprawl()
    plain, sprawled = state.players[1].battlefield
    assert _mask(state, table)[_row(table, "Tap Forest")]
    table[_row(table, "Tap Forest")][2](state)
    sub = state.mana_subdecision
    assert sub["stage"] == "choose_source" and {id(p) for p in sub["candidates"]} == {id(plain), id(sprawled)}
    assert not _mask(state, table).any(), "while the choice is open, only the pointer half may act"
    assert AB.any_pointer_legal(state) and AB.pointer_kind(state) == "mana_subdecision"
    identities = [*state.players[1].battlefield, *state.players[0].battlefield]
    assert AB.pointer_legal_mask(state, identities) == [True, True, False]
    AB.execute_pointer_choice(state, sprawled)
    assert state.mana_subdecision is None
    assert dict(state.mana_pool) == {"G": 1, "R": 1} and sprawled.tapped and not plain.tapped


def test_f02_picking_the_plain_forest_gives_plain_mana():
    table = _table(LANDS)
    state = _two_forests_one_with_sprawl()
    plain, sprawled = state.players[1].battlefield
    table[_row(table, "Tap Forest")][2](state)
    AB.execute_pointer_choice(state, plain)
    assert dict(state.mana_pool) == {"G": 1} and plain.tapped and not sprawled.tapped


def test_f02_the_choice_is_also_offered_in_the_middle_of_a_payment():
    table = _table(LANDS)
    state = _two_forests_one_with_sprawl()
    state.pending_resolution = {"kind": "pay_cost", "remaining": {"R": 1}, "announced": {"R": 1}, "on_complete": lambda s: None}
    assert _mask(state, table)[_row(table, "Tap Forest")]
    table[_row(table, "Tap Forest")][2](state)
    assert state.mana_subdecision["stage"] == "choose_source"
    assert state.pending_resolution["kind"] == "pay_cost", "the open payment is untouched by the sub-decision"


def test_f02_a_tap_that_would_strand_the_payment_is_not_a_candidate():
    """Remaining {R}: the plain Forest cannot make R and tapping it removes no R source, but with a single R-capable copy
    and a second payment need the strand rule must exclude the copy whose tap wastes the supply."""
    table = _table(LANDS)
    state = _two_forests_one_with_sprawl()
    plain, sprawled = state.players[1].battlefield
    state.pending_resolution = {"kind": "pay_cost", "remaining": {"R": 1, "G": 1}, "announced": {"R": 1, "G": 1},
                                "on_complete": lambda s: None}
    # tapping the plain Forest (G) leaves the Sprawl Forest's G+R to pay R: fine; tapping the Sprawl Forest pays both: fine.
    table[_row(table, "Tap Forest")][2](state)
    assert {id(p) for p in state.mana_subdecision["candidates"]} == {id(plain), id(sprawled)}
    # now make the plain Forest useless: remaining {R:2} cannot be paid at all, so no copy may be offered
    state2 = _two_forests_one_with_sprawl()
    state2.pending_resolution = {"kind": "pay_cost", "remaining": {"R": 2}, "announced": {"R": 2}, "on_complete": lambda s: None}
    assert not _mask(state2, table)[_row(table, "Tap Forest")]


# ---- activations: which copy -----------------------------------------------------------------------------------------

def _kill_ship_deck():
    return [("Pinnacle Kill-Ship", 2), ("Forest", 4)]


def test_f02_station_is_legal_through_the_copy_that_can_pay_even_if_the_first_cannot():
    """Two Kill-Ships; the first is an animated creature with no OTHER untapped creature to tap, the second can tap the first."""
    table = _table(_kill_ship_deck())
    state = _board(["Pinnacle Kill-Ship", "Pinnacle Kill-Ship"])
    first, second = state.players[1].battlefield
    first.type_override = game.CardType.CREATURE
    label = "Activate Pinnacle Kill-Ship (station)"
    assert _mask(state, table)[_row(table, label)]
    table[_row(table, label)][2](state)
    pending = state.pending_resolution
    assert pending is not None and pending["kind"] == "choose_permanent"
    assert game.choose_permanent_options(state) == [("Pinnacle Kill-Ship", first.slot)]


def test_f02_non_interchangeable_activation_copies_ask_which_one():
    table = _table(_kill_ship_deck())
    state = _board(["Pinnacle Kill-Ship", "Pinnacle Kill-Ship", "Forest"])
    a, b, forest = state.players[1].battlefield
    a.counters["charge"] = 3
    forest.card_type_override = None
    a.type_override = b.type_override = game.CardType.CREATURE  # both animated: each can tap the other
    label = "Activate Pinnacle Kill-Ship (station)"
    table[_row(table, label)][2](state)
    pending = state.pending_resolution
    assert pending["kind"] == "choose_permanent"
    assert set(game.choose_permanent_options(state)) == {("Pinnacle Kill-Ship", a.slot), ("Pinnacle Kill-Ship", b.slot)}
    mask = AB.pointer_legal_mask(state, list(state.players[1].battlefield))
    assert mask == [True, True, False]


# ---- cost: {T} is per ability ------------------------------------------------------------------------------------------

COST_DECK = [("Mountain", 4), ("Reckless Lackey", 1), ("Bonder's Ornament", 1), ("Wellwisher", 1)]


def _pool(state, n=8):
    for c in game.POOL_COLORS:
        state.players[1].mana_pool[c] = n


def test_an_ability_without_tap_in_its_cost_works_on_a_tapped_permanent():
    table = _table(COST_DECK)
    state = _board([("Reckless Lackey", {"tapped": True})])  # attacked last turn, never untapped
    _pool(state)
    assert _mask(state, table)[_row(table, "Activate Reckless Lackey (sac)")]


def test_an_ability_with_tap_in_its_cost_does_not_work_on_a_tapped_permanent():
    table = _table(COST_DECK)
    state = _board([("Bonder's Ornament", {"tapped": True})])
    _pool(state)
    assert not _mask(state, table)[_row(table, "Activate Bonder's Ornament (draw)")]
    untapped = _board(["Bonder's Ornament"])
    _pool(untapped)
    assert _mask(untapped, table)[_row(table, "Activate Bonder's Ornament (draw)")]


def test_a_summoning_sick_creature_cannot_pay_tap_but_can_pay_a_sacrifice_cost():
    table = _table(COST_DECK)
    state = _board([("Wellwisher", {"sick": True}), ("Reckless Lackey", {"sick": True})])
    _pool(state)
    mask = _mask(state, table)
    assert not mask[_row(table, "Activate Wellwisher (lifegain)")]
    assert mask[_row(table, "Activate Reckless Lackey (sac)")]


def test_clue_token_works_tapped():
    table = _table(COST_DECK)
    clue = next(cd for cd in TOKEN_DEFS if cd.name == "Clue")
    state = _board([])
    p = Permanent(clue, tapped=True)
    p.summoning_sick = False
    state.players[1].battlefield = [p]
    _pool(state)
    assert _mask(state, table)[_row(table, "Activate Clue (sac)")]


# ---- the declaration is checked against what each resolve really does ---------------------------------------------------

def _abilities():
    out = []
    for _name, cd in sorted(game.CARD_DEFS.items()):
        for ab, spec in game.EFFECT_REGISTRY.get(cd.effect_id, {}).get("activated_abilities", {}).items():
            out.append((cd, ab, spec))
    for cd in TOKEN_DEFS:
        for ab, spec in game.EFFECT_REGISTRY.get(cd.effect_id, {}).get("activated_abilities", {}).items():
            out.append((cd, ab, spec))
    return out


@pytest.mark.parametrize("cd,ability,spec", _abilities(), ids=lambda v: getattr(v, "name", v if isinstance(v, str) else ""))
def test_every_activated_ability_declares_whether_its_cost_taps(cd, ability, spec):
    assert isinstance(spec.get("taps"), bool), f"{cd.name} ({ability}) must declare taps: True/False in its registry spec"


def _drive_activation(cd, ability, spec):
    """Activate `cd`'s ability with ample mana and report whether paying its cost tapped the permanent (or removed it)."""
    seen = {}
    real = spec["resolve"]

    def spy(state, permanent):
        real(state, permanent)
        seen["tapped"] = permanent.tapped or permanent not in state.players[1].battlefield
        seen["gone"] = permanent not in state.players[1].battlefield

    spec["resolve"] = spy
    try:
        deck = [(cd.name, 2)] if cd.name in game.CARD_DEFS else [("Forest", 4)]
        table = _table(deck + [("Forest", 4)])
        state = _board([(cd.name, {})] if cd.name in game.CARD_DEFS else [], opp_bf=["Llanowar Elves"])
        me = state.players[1]
        if cd.name not in game.CARD_DEFS:
            p = Permanent(cd)
            p.summoning_sick = False
            me.battlefield.append(p)
        me.battlefield += [Permanent(game.CARD_DEFS["Forest"]) for _ in range(2)]
        me.battlefield += [Permanent(game.CARD_DEFS["Llanowar Elves"])]
        for q in me.battlefield:
            q.summoning_sick = False
        me.library = [game.CARD_DEFS["Forest"]] * 20
        me.hand = [game.CARD_DEFS["Forest"]] * 3
        me.graveyard = []
        state.players[0].graveyard = []
        _pool(state, 10)
        label = f"Activate {cd.name} ({ability})"
        idx = next((i for i, r in enumerate(table) if r[0] == label), None)
        if idx is None or not _mask(state, table)[idx]:
            return None
        table[idx][2](state)
        for _ in range(40):  # answer whatever the cost asks (spend mana, pick a permanent, ...) until the resolve ran
            if "tapped" in seen or state.pending_resolution is None:
                break
            mask = _mask(state, table)
            ids = list(state.players[1].battlefield)
            if AB.any_pointer_legal(state):
                ptr = AB.pointer_legal_mask(state, ids)
                if any(ptr):
                    AB.execute_pointer_choice(state, ids[ptr.index(True)])
                    continue
            legal = [i for i, ok in enumerate(mask) if ok]
            if not legal:
                break
            table[legal[0]][2](state)
        return seen.get("tapped"), seen.get("gone")
    finally:
        spec["resolve"] = real


@pytest.mark.parametrize("cd,ability,spec", [t for t in _abilities() if "cost_key" in t[2]],
                         ids=lambda v: getattr(v, "name", v if isinstance(v, str) else ""))
def test_a_cost_key_abilitys_tap_declaration_matches_what_its_resolve_does(cd, ability, spec):
    result = _drive_activation(cd, ability, spec)
    if result is None or result[0] is None:
        pytest.skip(f"{cd.name} ({ability}): the generic scenario cannot activate it")
    tapped_or_gone, gone = result
    if spec["taps"]:
        assert tapped_or_gone, f"{cd.name} ({ability}) declares taps=True but its cost left the permanent untapped"
    else:
        assert not tapped_or_gone or gone, f"{cd.name} ({ability}) declares taps=False but its cost tapped the permanent"


def test_f02_the_copy_whose_tap_would_strand_the_payment_is_never_offered():
    """Owing {U}: the Forest carrying Abundant Growth is the only U source, so tapping it for its native G would strand
    the payment; the plain Forest is the only legal choice, so nothing is asked and the plain one is tapped."""
    table = _table(LANDS)
    state = _board(["Forest", "Forest"], opp_bf=[("Abundant Growth", {"aura_on": (1, 1)})])
    state.players[0].battlefield[0].flags["bonus_mana_colors"] = set(game.COLORS)
    plain, grown = state.players[1].battlefield
    state.pending_resolution = {"kind": "pay_cost", "remaining": {"U": 1}, "announced": {"U": 1}, "on_complete": lambda s: None}
    assert _mask(state, table)[_row(table, "Tap Forest")]
    table[_row(table, "Tap Forest")][2](state)
    assert state.mana_subdecision is None
    assert plain.tapped and not grown.tapped and dict(state.mana_pool) == {"G": 1}


# ---- F-03: a granted ability on a source that makes several mana -------------------------------------------------------

def _carnarium_with_growth(opp_controls_aura=False):
    """Rakdos Carnarium ({B}{R} when tapped) enchanted with Abundant Growth (any color, one mana)."""
    growth = ("Abundant Growth", {"aura_on": (1, 0)})
    state = _board(["Rakdos Carnarium"], opp_bf=[growth]) if opp_controls_aura else _board(["Rakdos Carnarium", growth[0]])
    if not opp_controls_aura:
        state.players[1].battlefield[1].flags["enchanting"] = state.players[1].battlefield[0]
    for pl in state.players:
        for p in pl.battlefield:
            if p.card_def.name == "Abundant Growth":
                p.flags["bonus_mana_colors"] = set(game.COLORS)
    return state


def test_f03_a_multi_mana_source_with_a_granted_ability_can_pay_for_either_way():
    state = _carnarium_with_growth()
    # its own {B}{R}, OR one mana of any color: {U} is payable (granted), {B}{R} is payable (native), {B}{U} is not
    assert game.plan_payment(state, {"U": 1}) is not None
    assert game.plan_payment(state, {"B": 1, "R": 1}) is not None
    assert game.plan_payment(state, {"generic": 2}) is not None
    assert game.plan_payment(state, {"B": 1, "U": 1}) is None
    assert game.plan_payment(state, {"generic": 3}) is None


def test_f03_the_same_holds_when_the_opponent_controls_the_aura():
    state = _carnarium_with_growth(opp_controls_aura=True)
    assert game.plan_payment(state, {"G": 1}) is not None
    assert game.plan_payment(state, {"B": 1, "G": 1}) is None


def test_f03_a_spell_that_needs_the_granted_color_is_castable_and_one_that_needs_two_is_not():
    table = _table([("Brainstorm", 4), ("Counterspell", 4), ("Rakdos Carnarium", 1), ("Abundant Growth", 1)])
    state = _carnarium_with_growth()
    state.players[1].hand = [game.CARD_DEFS["Brainstorm"], game.CARD_DEFS["Counterspell"]]
    mask = _mask(state, table)
    assert mask[_row(table, "Cast Brainstorm")], "{U} is payable through the granted ability"
    assert not mask[_row(table, "Cast Counterspell")], "{U}{U} needs two blue; the granted ability makes only one mana"


def test_f02_legality_looks_at_every_copy_not_just_the_first():
    """The FIRST Forest (carrying Abundant Growth, the only U source) would strand the payment if tapped for its native G;
    the SECOND, plain Forest is fine. 'Tap Forest' must still be legal."""
    table = _table(LANDS)
    state = _board(["Forest", "Forest"], opp_bf=[("Abundant Growth", {"aura_on": (1, 0)})])
    state.players[0].battlefield[0].flags["bonus_mana_colors"] = set(game.COLORS)
    grown, plain = state.players[1].battlefield
    state.pending_resolution = {"kind": "pay_cost", "remaining": {"U": 1}, "announced": {"U": 1}, "on_complete": lambda s: None}
    assert _mask(state, table)[_row(table, "Tap Forest")]
    table[_row(table, "Tap Forest")][2](state)
    assert plain.tapped and not grown.tapped
