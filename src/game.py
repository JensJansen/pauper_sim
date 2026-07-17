"""Tron Assembly Simulator.

Implements TRON_SIMULATOR_PLAN.md / TRON_SIMULATOR_CHECKLIST.md, phase by phase.
"""

from enum import Enum, auto


class EffectId(Enum):
    TRON_LAND = auto()
    FOREST = auto()
    WOODED_RIDGELINE = auto()
    BOJUKA_BOG = auto()
    TOCASIA_DIG_SITE = auto()
    CONDUIT_PYLONS = auto()
    EXPEDITION_MAP = auto()
    CROP_ROTATION = auto()
    ANCIENT_STIRRINGS = auto()
    BONDERS_ORNAMENT = auto()
    CANDY_TRAIL = auto()
    BARRELS_OF_BLASTING_JELLY = auto()
    RELIC_OF_PROGENITUS = auto()
    GENEROUS_ENT = auto()
    FILLER = auto()


class CardType(Enum):
    LAND = auto()
    ARTIFACT = auto()
    SORCERY = auto()
    INSTANT = auto()
    CREATURE = auto()
    FILLER = auto()


class CardDef:
    """Definition shared by every physical copy of a named card (Phase 0)."""

    def __init__(self, name, card_type, cast_cost, effect_id, **extra):
        self.name = name
        self.card_type = card_type
        self.cast_cost = cast_cost  # dict like {"generic": 1, "G": 1}, or None
        self.effect_id = effect_id
        self.extra = extra  # e.g. tron_type="Mine" for the three Tron lands

    def __repr__(self):
        return f"CardDef({self.name!r})"


# (name, quantity, card_type, cast_cost, effect_id, extra kwargs)
# MULTI_DECK_PLAN.md Phase M2: renamed from the bare DECKLIST -- this is
# Tron's decklist specifically, no longer the only one that will ever
# exist, passed explicitly wherever a game needs it rather than read as
# an implicit global.
TRON_DECKLIST = [
    # --- Tron lands (12) ---
    ("Urza's Mine", 4, CardType.LAND, None, EffectId.TRON_LAND, {"tron_type": "Mine"}),
    ("Urza's Power Plant", 4, CardType.LAND, None, EffectId.TRON_LAND, {"tron_type": "Power Plant"}),
    ("Urza's Tower", 4, CardType.LAND, None, EffectId.TRON_LAND, {"tron_type": "Tower"}),

    # --- Other real lands (6) ---
    ("Forest", 2, CardType.LAND, None, EffectId.FOREST, {}),
    ("Wooded Ridgeline", 1, CardType.LAND, None, EffectId.WOODED_RIDGELINE, {}),
    ("Bojuka Bog", 1, CardType.LAND, None, EffectId.BOJUKA_BOG, {}),
    ("Tocasia's Dig Site", 1, CardType.LAND, None, EffectId.TOCASIA_DIG_SITE,
     {"surveil_ability_cost": {"generic": 3}}),
    ("Conduit Pylons", 1, CardType.LAND, None, EffectId.CONDUIT_PYLONS, {}),

    # --- Other relevant nonland cards (24) ---
    ("Expedition Map", 4, CardType.ARTIFACT, {"generic": 1}, EffectId.EXPEDITION_MAP,
     {"ability_cost": {"generic": 2}}),
    ("Crop Rotation", 2, CardType.INSTANT, {"G": 1}, EffectId.CROP_ROTATION, {}),
    ("Ancient Stirrings", 4, CardType.SORCERY, {"G": 1}, EffectId.ANCIENT_STIRRINGS, {}),
    ("Bonder's Ornament", 4, CardType.ARTIFACT, {"generic": 3}, EffectId.BONDERS_ORNAMENT,
     {"draw_ability_cost": {"generic": 4}}),
    ("Candy Trail", 4, CardType.ARTIFACT, {"generic": 1}, EffectId.CANDY_TRAIL,
     {"sac_ability_cost": {"generic": 2}}),
    ("Barrels of Blasting Jelly", 2, CardType.ARTIFACT, {"generic": 1}, EffectId.BARRELS_OF_BLASTING_JELLY,
     {"mana_ability_cost": {"generic": 1}}),
    ("Relic of Progenitus", 2, CardType.ARTIFACT, {"generic": 1}, EffectId.RELIC_OF_PROGENITUS,
     {"draw_ability_cost": {"generic": 1}}),
    ("Generous Ent", 2, CardType.CREATURE, {"generic": 5, "G": 1}, EffectId.GENEROUS_ENT,
     {"forestcycling_cost": {"generic": 1}}),

    # --- Filler (18) --- real cast costs are irrelevant: filler is never cast (see plan).
    ("Rooftop Percher", 2, CardType.FILLER, None, EffectId.FILLER, {}),
    ("Boulderbranch Golem", 2, CardType.FILLER, None, EffectId.FILLER, {}),
    ("Maelstrom Colossus", 4, CardType.FILLER, None, EffectId.FILLER, {}),
    ("Bramble Wurm", 4, CardType.FILLER, None, EffectId.FILLER, {}),
    ("Pinnacle Kill-Ship", 4, CardType.FILLER, None, EffectId.FILLER, {}),
    ("Breath Weapon", 2, CardType.FILLER, None, EffectId.FILLER, {}),
]


def build_card_defs(decklist):
    """One CardDef per distinct name (quantity handled separately at
    deck-construction time). Takes a decklist explicitly so a future
    second deck can extend CARD_DEFS the same way, without changing this
    function -- deferred until a second deck actually exists (see
    MULTI_DECK_PLAN.md's explicit "out of scope" note)."""
    return {
        name: CardDef(name, card_type, cast_cost, effect_id, **extra)
        for name, _qty, card_type, cast_cost, effect_id, extra in decklist
    }


# Shared across every deck, not deck-scoped (MULTI_DECK_PLAN.md Phase M2
# decision): a card's definition is fixed metadata, identical no matter
# which deck it's played in, so one global name->CardDef lookup can grow
# to cover every implemented card as more decks are added, rather than
# each deck needing its own copy.
CARD_DEFS = build_card_defs(TRON_DECKLIST)


def _phase0_sanity_check():
    total = sum(qty for _name, qty, *_rest in TRON_DECKLIST)
    assert total == 60, f"decklist quantities sum to {total}, expected 60"
    assert len(TRON_DECKLIST) == len(CARD_DEFS), "duplicate card name in TRON_DECKLIST"



# ---------------------------------------------------------------------------
# Phase 1 — Zone & state model
# ---------------------------------------------------------------------------

import random


class Permanent:
    """A specific physical card sitting on the battlefield (Phase 1)."""

    def __init__(self, card_def, tapped=False):
        self.card_def = card_def
        self.tapped = tapped
        # Generic per-permanent runtime flags, e.g. "used_mana_filter_this_turn"
        # for Barrels of Blasting Jelly's once-per-turn ability (Phase 3).
        self.flags = {}

    def __repr__(self):
        return f"Permanent({self.card_def.name!r}, tapped={self.tapped})"


class GameState:
    """All zones plus turn/metric bookkeeping for one simulated game (Phase 1).

    No mana_pool field: Phase 2 pays costs by tapping specific permanents
    directly against a cost (pay-as-you-go), rather than floating mana into
    an abstract pool first. See Phase 2 notes.
    """

    def __init__(self, on_the_play, rng=None, terminated_fn=None):
        self.library = []       # ordered list[CardDef], index 0 = top of deck
        self.hand = []          # list[CardDef]
        self.battlefield = []   # list[Permanent]
        self.graveyard = []     # list[CardDef] -- bookkeeping only, never read back

        self.lands_played_this_turn = 0
        self.turn_number = 0
        self.on_the_play = on_the_play
        self.rng = rng or random.Random()

        # MULTI_DECK_PLAN.md Phase M3: terminated_fn(state) -> bool is the
        # per-deck injected win condition, checked generically in
        # enters_battlefield (below) rather than hardcoded to Tron lands.
        # turn_won replaces turn_assembled/turn_online -- a single turn
        # number, set once, the first time terminated_fn returns True.
        # There's no more "online" second tier here: a deck that wants
        # that distinction expresses it as a scoring function instead (see
        # rewards.py), not as tracked GameState. Defaults to None so
        # hand-built test states (which set turn_won directly, bypassing
        # this mechanism) don't need to supply one; real games always pass
        # one explicitly via new_game_state.
        self.terminated_fn = terminated_fn
        self.turn_won = None

        # MULTI_DECK_PLAN.md Phase M4: None, or a dict describing an
        # in-progress multi-step decision (paying a cost one tap at a
        # time, resolving a scry/surveil, choosing a search target, ...)
        # that must be fully resolved -- via repeated calls into whatever
        # that resolution's own option/execute functions are -- before any
        # other action becomes legal again. See begin_resolution/
        # complete_resolution below.
        self.pending_resolution = None

    def draw(self, n=1):
        for _ in range(n):
            if not self.library:
                raise RuntimeError("cannot draw from an empty library")
            self.hand.append(self.library.pop(0))


def build_shuffled_library(decklist, rng):
    """Expand a decklist's quantities into CardDef refs and shuffle (Phase
    1/6). CARD_DEFS stays the shared global lookup (see the Phase M2 note
    above); only which decklist's quantities to expand is parameterized."""
    library = []
    for name, qty, *_rest in decklist:
        library.extend([CARD_DEFS[name]] * qty)
    rng.shuffle(library)
    return library


def new_game_state(decklist, terminated_fn, on_the_play, rng):
    state = GameState(on_the_play, rng=rng, terminated_fn=terminated_fn)
    state.library = build_shuffled_library(decklist, rng)
    state.draw(7)
    return state


def _phase1_sanity_check():
    rng = random.Random(0)
    state = new_game_state(TRON_DECKLIST, tron_terminated, on_the_play=True, rng=rng)
    assert len(state.library) == 53, len(state.library)
    assert len(state.hand) == 7, len(state.hand)
    assert len(state.battlefield) == 0
    assert len(state.graveyard) == 0


# ---------------------------------------------------------------------------
# Phase 2 — Mana system
# ---------------------------------------------------------------------------

TRON_TYPES = {"Mine", "Power Plant", "Tower"}
COLORS = ("W", "U", "B", "R", "G")

# SIMPLE_MANA_SOURCE_EFFECTS / _FIXED_SOURCE_COLOR / _FLEXIBLE_SOURCE_CHOICES
# used to be hand-authored dicts here. As of MULTI_DECK_PLAN.md Phase M1
# they're derived views over EFFECT_REGISTRY (defined at the end of the
# "Phase 3 -- Card effects" section below, since it also needs to reference
# card-effect resolve functions -- surveil, scry, the activate_* functions --
# that aren't defined yet at this point in the file). Referencing them here,
# in functions defined before that point (mana_output, pay_cost,
# choose_taps_for_cost), is safe: Python resolves a name inside a function
# body when that function is *called*, not when it's defined, and nothing
# calls these functions until the whole module has finished loading.


def controls_all_tron_types(state):
    present = {
        p.card_def.extra["tron_type"]
        for p in state.battlefield
        if p.card_def.effect_id == EffectId.TRON_LAND
    }
    return TRON_TYPES.issubset(present)


def tron_terminated(state):
    """Tron's terminated_fn (MULTI_DECK_PLAN.md Phase M3): the win
    condition is exactly controls_all_tron_types -- a thin, named wrapper
    matching the terminated_fn(state) -> bool contract every deck's own
    win condition implements."""
    return controls_all_tron_types(state)


def mana_output(permanent, state, color_choice=None):
    """Mana symbols this permanent would produce if tapped for its plain
    mana ability right now. Raises if effect_id isn't a simple source or if
    a required/forbidden color_choice is missing/invalid."""
    effect = permanent.card_def.effect_id
    spec = EFFECT_REGISTRY.get(effect, {}).get("mana")
    if spec is None:
        raise ValueError(f"{permanent.card_def.name} is not a simple mana source")
    kind = spec[0]
    if kind == "tron":
        return ["C", "C"] if controls_all_tron_types(state) else ["C"]
    if kind == "fixed":
        if color_choice is not None:
            raise ValueError(f"{permanent.card_def.name} has no color choice")
        return [spec[1]]
    if kind == "flexible":
        choices = spec[1]
        if color_choice not in choices:
            raise ValueError(f"{permanent.card_def.name} cannot produce {color_choice}")
        return [color_choice]
    raise ValueError(f"{permanent.card_def.name} is not a simple mana source")


def pay_cost(state, cost, tap_choices):
    """Execute an already-decided payment (Phase 5 chooses; this only executes).

    tap_choices: list of (permanent, color_choice_or_None). Must be untapped
    permanents on state.battlefield with a simple mana-source effect. Raises
    ValueError if the plan is invalid or doesn't cover `cost`; on success,
    taps every permanent in tap_choices and returns.
    """
    for permanent, _color in tap_choices:
        if permanent not in state.battlefield:
            raise ValueError(f"{permanent.card_def.name} is not on the battlefield")
        if permanent.tapped:
            raise ValueError(f"{permanent.card_def.name} is already tapped")
        if permanent.card_def.effect_id not in SIMPLE_MANA_SOURCE_EFFECTS:
            raise ValueError(f"{permanent.card_def.name} is not a simple mana source")
    if len({id(p) for p, _c in tap_choices}) != len(tap_choices):
        raise ValueError("tap_choices repeats the same permanent")

    produced = []
    for permanent, color_choice in tap_choices:
        produced.extend(mana_output(permanent, state, color_choice))

    remaining = dict(cost)
    leftover_generic = 0
    for symbol in produced:
        need = remaining.get(symbol, 0)
        if need > 0:
            remaining[symbol] = need - 1
        else:
            leftover_generic += 1
    generic_needed = remaining.get("generic", 0)
    remaining["generic"] = max(0, generic_needed - leftover_generic)
    if any(v > 0 for v in remaining.values()):
        raise ValueError(f"tap_choices do not cover cost {cost} (produced {produced})")

    for permanent, _color in tap_choices:
        permanent.tapped = True


def choose_taps_for_cost(state, cost):
    """Default source-selection: prefer non-Tron mana for costs (per plan's
    greedy policy rule). Returns a tap_choices list usable by pay_cost, or
    None if the cost can't currently be paid from simple mana sources."""
    untapped = [
        p for p in state.battlefield
        if not p.tapped and p.card_def.effect_id in SIMPLE_MANA_SOURCE_EFFECTS
    ]
    untapped.sort(key=lambda p: p.card_def.effect_id == EffectId.TRON_LAND)

    chosen = []
    pool = list(untapped)

    for color in COLORS:
        need = cost.get(color, 0)
        for _ in range(need):
            pick = None
            for p in pool:
                effect = p.card_def.effect_id
                if effect in _FIXED_SOURCE_COLOR and _FIXED_SOURCE_COLOR[effect] == color:
                    pick = (p, None)
                    break
                if effect in _FLEXIBLE_SOURCE_CHOICES and color in _FLEXIBLE_SOURCE_CHOICES[effect]:
                    pick = (p, color)
                    break
            if pick is None:
                return None
            chosen.append(pick)
            pool.remove(pick[0])

    generic_needed = cost.get("generic", 0)
    while generic_needed > 0 and pool:
        p = pool.pop(0)
        effect = p.card_def.effect_id
        if effect in _FLEXIBLE_SOURCE_CHOICES:
            color_choice = next(iter(_FLEXIBLE_SOURCE_CHOICES[effect]))
            produced = mana_output(p, state, color_choice)
            chosen.append((p, color_choice))
        else:
            produced = mana_output(p, state, None)
            chosen.append((p, None))
        generic_needed -= len(produced)

    if generic_needed > 0:
        return None
    return chosen


# ---------------------------------------------------------------------------
# Pending resolution (MULTI_DECK_PLAN.md Phase M4): a decision point that
# takes more than one action to fully resolve -- paying a cost one tap at
# a time, walking a scry/surveil, choosing a search target -- because the
# model, not an automatic solver, makes every one of these choices now.
# Generic core (begin/complete) plus one concrete kind (pay_cost) live
# here, in the mana section, since payment is by far the most common use;
# scry/surveil/search/Ancient Stirrings's own kinds are defined near their
# own effects further down.
# ---------------------------------------------------------------------------

def begin_resolution(state, kind, on_complete, **fields):
    """Start a pending resolution. on_complete(state) runs once it's fully
    resolved (via repeated calls into that kind's own option/execute
    functions) -- it may itself begin a further resolution, so multi-step
    effects (e.g. cast Ancient Stirrings: pay its cost, then choose which
    revealed card to take) chain naturally through nested callbacks rather
    than needing a single monolithic resolution type."""
    state.pending_resolution = {"kind": kind, "on_complete": on_complete, **fields}


def complete_resolution(state, *args):
    """*args is an optional payload for kinds whose completion carries a
    result the caller needs (e.g. search_fetch's chosen card name) --
    on_complete(state) for kinds that don't (e.g. pay_cost)."""
    on_complete = state.pending_resolution["on_complete"]
    state.pending_resolution = None
    on_complete(state, *args)


def begin_pay_cost(state, cost, on_complete):
    """Interactive mana payment: the model taps one source at a time (see
    tap_cost_options/execute_tap_cost_option) until `cost` is fully
    covered, instead of an automatic solver picking every tap at once the
    way choose_taps_for_cost/pay_cost still do (kept for pure legality
    checks -- plan_payment(state, cost) is not None -- since "can this be
    paid at all" is a feasibility question, not a strategic one)."""
    begin_resolution(state, "pay_cost", on_complete, remaining=dict(cost), tapped=[])


def _apply_tap_to_remaining(remaining, produced_symbols):
    """Same math as pay_cost's batch version (Phase 2), applied to one
    tap's output at a time: symbols matching an outstanding need reduce
    it; unmatched symbols spill into reducing outstanding generic."""
    leftover_generic = 0
    for symbol in produced_symbols:
        need = remaining.get(symbol, 0)
        if need > 0:
            remaining[symbol] = need - 1
        else:
            leftover_generic += 1
    generic_needed = remaining.get("generic", 0)
    remaining["generic"] = max(0, generic_needed - leftover_generic)


def _filter_mana_eligible(remaining):
    """Barrels of Blasting Jelly / Conduit Pylons' colored-pip filter mode
    is only ever useful for exactly one outstanding colored pip of exactly
    quantity 1 -- the same narrow scope plan_payment already enforces for
    its own legality check, preserved exactly here for the interactive
    version. Returns that one color, or None if the condition isn't met."""
    needed_colors = [c for c in COLORS if remaining.get(c, 0) > 0]
    if len(needed_colors) == 1 and remaining[needed_colors[0]] == 1:
        return needed_colors[0]
    return None


def tap_cost_options(state):
    """While a pay_cost resolution is pending: every (name, color_choice,
    is_filter) option that would make progress on the remaining cost right
    now, one per distinct source *name* (not per physical permanent --
    same-named untapped sources are interchangeable, so this stays a small
    bounded list regardless of how many copies are in play). color_choice
    is None for fixed-color/Tron sources; is_filter marks Barrels/Conduit
    Pylons used in their colored-pip filter mode."""
    pending = state.pending_resolution
    remaining = pending["remaining"]
    if not any(v > 0 for v in remaining.values()):
        return []

    tapped_ids = {id(p) for p, _is_filter in pending["tapped"]}
    options = []
    seen = set()

    for p in state.battlefield:
        if p.tapped or id(p) in tapped_ids:
            continue
        effect = p.card_def.effect_id
        spec = EFFECT_REGISTRY.get(effect, {}).get("mana")
        if spec is not None:
            kind = spec[0]
            if kind in ("fixed", "tron"):
                key = (p.card_def.name, None, False)
                if key not in seen:
                    seen.add(key)
                    options.append(key)
            elif kind == "flexible":
                for color in spec[1]:
                    key = (p.card_def.name, color, False)
                    if key not in seen:
                        seen.add(key)
                        options.append(key)

    filter_color = _filter_mana_eligible(remaining)
    if filter_color is not None:
        for p in state.battlefield:
            if id(p) in tapped_ids:
                continue
            effect = p.card_def.effect_id
            if EFFECT_REGISTRY.get(effect, {}).get("filter_mana") is None:
                continue
            already_used = (
                p.flags.get("used_this_turn", False) if effect == EffectId.BARRELS_OF_BLASTING_JELLY
                else p.tapped
            )
            if already_used:
                continue
            key = (p.card_def.name, filter_color, True)
            if key not in seen:
                seen.add(key)
                options.append(key)

    return options


def execute_tap_cost_option(state, name, color_choice, is_filter):
    pending = state.pending_resolution
    tapped_ids = {id(p) for p, _is_filter in pending["tapped"]}

    def _available(p):
        if id(p) in tapped_ids or p.card_def.name != name:
            return False
        if is_filter:
            return not (
                p.flags.get("used_this_turn", False) if p.card_def.effect_id == EffectId.BARRELS_OF_BLASTING_JELLY
                else p.tapped
            )
        return not p.tapped

    permanent = next(p for p in state.battlefield if _available(p))
    pending["tapped"].append((permanent, is_filter))  # is_filter recorded so abandon_pay_cost can reverse it correctly

    if is_filter:
        if permanent.card_def.effect_id == EffectId.BARRELS_OF_BLASTING_JELLY:
            permanent.flags["used_this_turn"] = True
        else:
            permanent.tapped = True
        pending["remaining"][color_choice] = max(0, pending["remaining"].get(color_choice, 0) - 1)
        # The filter ability itself costs {1} (real cards: "{1}: Add one
        # mana of any color" / "{1}, T: Add one mana of any color") -- a
        # pure color-fix, never a net mana gain. plan_payment's legality
        # check already folds this into filtered_cost; this is the same
        # charge applied one tap at a time.
        pending["remaining"]["generic"] = pending["remaining"].get("generic", 0) + 1
    else:
        permanent.tapped = True
        produced = mana_output(permanent, state, color_choice)
        _apply_tap_to_remaining(pending["remaining"], produced)

    if not any(v > 0 for v in pending["remaining"].values()):
        complete_resolution(state)


def abandon_pay_cost(state):
    """Reverses every tap made so far in a pending pay_cost resolution and
    cancels it outright -- no on_complete call, as if the action that
    began paying this cost was never chosen (MULTI_DECK_PLAN.md Phase
    M4e). Without this, tapping a flexible/filter source for a color that
    turns out not to be needed could strand the game with an unpayable
    remaining cost and zero legal actions -- real Magic's actual rule is
    that being unable to complete a cost undoes the whole action, not
    that every choice leading there must be prevented in advance. Safe to
    call any time a pay_cost resolution is pending: the action that
    triggered payment never touches hand/battlefield/graveyard until its
    own on_complete fires (see _cast_execute/_activate_execute/
    _forestcycle_execute in tron_env.py), so undoing the taps alone is a
    complete, correct undo."""
    pending = state.pending_resolution
    for permanent, is_filter in pending["tapped"]:
        if is_filter and permanent.card_def.effect_id == EffectId.BARRELS_OF_BLASTING_JELLY:
            permanent.flags["used_this_turn"] = False
        else:
            permanent.tapped = False
    state.pending_resolution = None


def _phase_m4a_sanity_check():
    """MULTI_DECK_PLAN.md Phase M4a: interactive mana payment, dedicated
    coverage since it replaces a previously-automatic solver."""
    done = []

    # Multi-tap generic payment: 2 Forests, one tap at a time.
    state = GameState(on_the_play=True)
    f1, f2 = Permanent(CARD_DEFS["Forest"]), Permanent(CARD_DEFS["Forest"])
    state.battlefield = [f1, f2]
    begin_pay_cost(state, {"generic": 2}, on_complete=lambda s: done.append("generic"))
    assert tap_cost_options(state) == [("Forest", None, False)]
    execute_tap_cost_option(state, "Forest", None, False)
    assert state.pending_resolution is not None, "1 of 2 taps done -- not resolved yet"
    assert tap_cost_options(state) == [("Forest", None, False)], "the other Forest is still offered"
    execute_tap_cost_option(state, "Forest", None, False)
    assert state.pending_resolution is None, "2nd tap covers the cost -- auto-completes"
    assert done == ["generic"]
    assert f1.tapped and f2.tapped

    # Colored pip (Forest for G) + generic (Urza's Mine, only 1 Tron type
    # in play so it produces a single C, not the doubled amount).
    state2 = GameState(on_the_play=True)
    forest = Permanent(CARD_DEFS["Forest"])
    mine = Permanent(CARD_DEFS["Urza's Mine"])
    state2.battlefield = [forest, mine]
    begin_pay_cost(state2, {"G": 1, "generic": 1}, on_complete=lambda s: done.append("colored"))
    execute_tap_cost_option(state2, "Forest", None, False)
    assert state2.pending_resolution is not None
    assert tap_cost_options(state2) == [("Urza's Mine", None, False)], "G is covered -- only the generic option remains"
    execute_tap_cost_option(state2, "Urza's Mine", None, False)
    assert state2.pending_resolution is None
    assert done == ["generic", "colored"]

    # Tron doubling: all 3 types in play -> a single Tron land tap covers
    # a 2-generic cost in one tap, not two.
    state3 = GameState(on_the_play=True)
    state3.battlefield = [
        Permanent(CARD_DEFS["Urza's Mine"]),
        Permanent(CARD_DEFS["Urza's Power Plant"]),
        Permanent(CARD_DEFS["Urza's Tower"]),
    ]
    begin_pay_cost(state3, {"generic": 2}, on_complete=lambda s: done.append("tron_double"))
    execute_tap_cost_option(state3, "Urza's Mine", None, False)
    assert state3.pending_resolution is None, "one Tron land taps for 2 C when all three types are controlled"
    assert done[-1] == "tron_double"

    # Flexible source color choice: Wooded Ridgeline can pay R or G.
    state4 = GameState(on_the_play=True)
    ridgeline = Permanent(CARD_DEFS["Wooded Ridgeline"])
    state4.battlefield = [ridgeline]
    begin_pay_cost(state4, {"R": 1}, on_complete=lambda s: done.append("flexible"))
    options4 = tap_cost_options(state4)
    assert set(options4) == {("Wooded Ridgeline", "R", False), ("Wooded Ridgeline", "G", False)}
    execute_tap_cost_option(state4, "Wooded Ridgeline", "R", False)
    assert state4.pending_resolution is None
    assert done[-1] == "flexible"

    # Barrels of Blasting Jelly's colored-pip filter mode: no direct G
    # source, only Barrels (filter) + a Forest to cover both the {G} and
    # the filter ability's own {1} (real card: "{1}: Add one mana of any
    # color" -- a pure color-fix, never a net mana gain).
    state5 = GameState(on_the_play=True)
    barrels = Permanent(CARD_DEFS["Barrels of Blasting Jelly"])
    forest5 = Permanent(CARD_DEFS["Forest"])
    state5.battlefield = [barrels, forest5]
    begin_pay_cost(state5, {"G": 1}, on_complete=lambda s: done.append("filter"))
    assert ("Barrels of Blasting Jelly", "G", True) in tap_cost_options(state5)
    execute_tap_cost_option(state5, "Barrels of Blasting Jelly", "G", True)
    assert barrels.flags.get("used_this_turn") is True, "Barrels uses a flag, not tapped, to mark itself spent"
    assert barrels.tapped is False
    assert tap_cost_options(state5) == [("Forest", None, False)], "G is covered, but filtering it cost {1} -- only the generic option remains"
    execute_tap_cost_option(state5, "Forest", None, False)
    assert state5.pending_resolution is None
    assert done[-1] == "filter"

    # Abandon payment: tapping Bonder's Ornament for the wrong color (W,
    # not the needed G) would otherwise strand the game -- Forest is still
    # untapped, but the model already spent Bonder's Ornament on nothing
    # useful. Abandoning must fully reverse it (both the simple tap and,
    # separately below, a filter-mode tap), not just clear the resolution.
    state6 = GameState(on_the_play=True)
    ornament = Permanent(CARD_DEFS["Bonder's Ornament"])
    forest6 = Permanent(CARD_DEFS["Forest"])
    state6.battlefield = [ornament, forest6]
    begin_pay_cost(state6, {"G": 1}, on_complete=lambda s: done.append("should not fire"))
    execute_tap_cost_option(state6, "Bonder's Ornament", "W", False)
    assert ornament.tapped is True
    assert state6.pending_resolution is not None, "G still unmet -- Forest alone would have covered it, but W was chosen instead"
    abandon_pay_cost(state6)
    assert state6.pending_resolution is None
    assert ornament.tapped is False, "abandon must untap what was already tapped"
    assert forest6.tapped is False
    assert done == ["generic", "colored", "tron_double", "flexible", "filter"], "on_complete must never fire on an abandoned payment"

    # Same, but the tap-so-far was a filter-mode one (a flag, not .tapped)
    # -- abandon must clear the flag, not try to untap a permanent that
    # was never tapped in the first place.
    state7 = GameState(on_the_play=True)
    barrels7 = Permanent(CARD_DEFS["Barrels of Blasting Jelly"])
    state7.battlefield = [barrels7]
    begin_pay_cost(state7, {"G": 1}, on_complete=lambda s: done.append("should not fire"))
    execute_tap_cost_option(state7, "Barrels of Blasting Jelly", "W", True)  # wrong color on purpose
    assert barrels7.flags.get("used_this_turn") is True
    abandon_pay_cost(state7)
    assert state7.pending_resolution is None
    assert barrels7.flags.get("used_this_turn") is False
    assert barrels7.tapped is False


def _phase2_sanity_check():
    rng = random.Random(0)
    state = GameState(on_the_play=True)
    forest = Permanent(CARD_DEFS["Forest"])
    mine = Permanent(CARD_DEFS["Urza's Mine"])
    state.battlefield = [mine, forest]

    taps = choose_taps_for_cost(state, {"G": 1})
    assert taps is not None
    assert [p.card_def.name for p, _c in taps] == ["Forest"], taps

    pay_cost(state, {"G": 1}, taps)
    assert forest.tapped is True
    assert mine.tapped is False


# ---------------------------------------------------------------------------
# Phase 3 — Card effects
# ---------------------------------------------------------------------------

# ENTERS_TAPPED_EFFECTS is derived from EFFECT_REGISTRY, defined near the
# end of this section (see the Phase M1 note there) -- not redefined here.
#
# TRON_TYPE_PRIORITY / missing_tron_types (a Tron-priority-ranking helper
# for the deleted heuristic and its is_priority_land/_rank_priority
# support functions) were removed alongside them in Phase M5 -- nothing
# else ever called them.


def is_noncreature_colorless(card_def):
    if card_def.card_type in (CardType.CREATURE, CardType.FILLER):
        return False
    if card_def.cast_cost is None:
        return True  # a land -- no mana cost, therefore colorless
    return not any(k in COLORS for k in card_def.cast_cost)


def enters_battlefield(state, card_def):
    """Move a CardDef onto the battlefield as a new Permanent, applying its
    enters-tapped default and ETB trigger (via EFFECT_REGISTRY, Phase 3),
    then check state.terminated_fn (MULTI_DECK_PLAN.md Phase M3) since a
    permanent entering is the only way any win condition discussed so far
    can newly become true. Caller has already removed card_def from its
    previous zone (hand/library)."""
    spec = EFFECT_REGISTRY.get(card_def.effect_id, {})
    tapped = spec.get("enters_tapped", False)
    permanent = Permanent(card_def, tapped=tapped)
    state.battlefield.append(permanent)

    etb_trigger = spec.get("etb_trigger")
    if etb_trigger is not None:
        etb_trigger(state)
    # Bojuka Bog's "exile target player's graveyard" ETB is a documented
    # no-op: no opposing graveyard exists in this solitaire simulator.

    if state.terminated_fn is not None and state.turn_won is None and state.terminated_fn(state):
        state.turn_won = state.turn_number

    return permanent


def play_land_from_hand(state, card_def):
    state.hand.remove(card_def)
    state.lands_played_this_turn += 1
    return enters_battlefield(state, card_def)


def cast_permanent_from_hand(state, card_def):
    """Artifacts with no additional cost beyond mana and no target choices:
    Expedition Map, Bonder's Ornament, Candy Trail, Barrels of Blasting
    Jelly, Relic of Progenitus. Mana cost is paid by the caller first."""
    state.hand.remove(card_def)
    return enters_battlefield(state, card_def)


def find_and_remove_by_name(state, name):
    """Search state.library for the first card matching `name`, remove and
    return it (or None if absent). Does not shuffle -- callers shuffle per
    their own card's rules. Replaces find_and_remove_priority_land's
    auto-pick (MULTI_DECK_PLAN.md Phase M4): which name to search for is
    now the model's choice, made via a search_fetch pending resolution
    (below) before this ever runs, not decided here."""
    for i, c in enumerate(state.library):
        if c.name == name:
            return state.library.pop(i)
    return None


def begin_search_fetch(state, predicate, on_complete):
    """Pending resolution (MULTI_DECK_PLAN.md Phase M4): the model picks
    ONE library card by name, among distinct names currently matching
    `predicate`, to fetch -- one action per matching name (search_fetch_
    options), not a full reveal (search effects look at the whole library,
    already-known information by elimination, not a scry-style reveal of
    previously-hidden cards). on_complete(state, chosen_name) runs once
    decided. If nothing in the library matches right now (legality only
    guarantees the *cost* was payable, not that a target still exists --
    e.g. every land could already be drawn), fizzles immediately with
    chosen_name=None instead of leaving a resolution with zero legal
    options, matching the old find_and_remove_priority_land's graceful
    "nothing found" behavior."""
    begin_resolution(state, "search_fetch", on_complete, predicate=predicate)
    if not search_fetch_options(state):
        complete_resolution(state, None)


def search_fetch_options(state):
    predicate = state.pending_resolution["predicate"]
    return sorted({c.name for c in state.library if predicate(c)})


def execute_search_fetch_option(state, name):
    complete_resolution(state, name)


def begin_choose_permanent(state, predicate, on_complete):
    """Pending resolution (MULTI_DECK_PLAN.md Phase M4): the model picks
    ONE of its own battlefield permanents, by name, among those matching
    `predicate` -- e.g. Crop Rotation's sacrifice target. Same
    fungible-by-name simplification as tap_cost_options: which physical
    copy doesn't matter, only which name. on_complete(state, chosen_name)
    runs once decided. Same empty-options safety net as begin_search_fetch
    -- fizzles immediately with chosen_name=None if nothing matches."""
    begin_resolution(state, "choose_permanent", on_complete, predicate=predicate)
    if not choose_permanent_options(state):
        complete_resolution(state, None)


def choose_permanent_options(state):
    predicate = state.pending_resolution["predicate"]
    return sorted({p.card_def.name for p in state.battlefield if predicate(p)})


def execute_choose_permanent_option(state, name):
    complete_resolution(state, name)


def begin_scry_surveil(state, kind, n, on_complete):
    """Pending resolution (MULTI_DECK_PLAN.md Phase M4c): reveal the top n
    library cards; the model decides keep-on-top or dispose for each one
    in turn (scry_surveil_options/execute_scry_surveil_option below),
    then -- if 2+ were kept -- the order to put them back in. Kept cards
    return to the library top in that model-chosen order; disposed cards
    go to the library bottom in random order (kind="scry") or the
    graveyard (kind="surveil") -- their order was never a model decision
    even before this refactor, since nothing here ever reads it again."""
    revealed = state.library[:n]
    del state.library[:n]
    begin_resolution(state, kind, on_complete, remaining=revealed, kept=[], disposed=[], ordered=None)


def scry_surveil_options(state):
    """While deciding (remaining non-empty): keep or dispose the current
    (front of remaining) card. While ordering (remaining empty, 2+ kept,
    not yet all placed): one option per distinct name still waiting to be
    placed on top."""
    pending = state.pending_resolution
    if pending["remaining"]:
        return ["keep", "dispose"]
    if pending["ordered"] is not None:
        return sorted({c.name for c in pending["kept"]})
    return []


def _finish_scry_surveil(state):
    pending = state.pending_resolution
    kept_final = pending["ordered"] if pending["ordered"] is not None else pending["kept"]
    disposed = pending["disposed"]
    state.library[0:0] = kept_final
    if pending["kind"] == "scry":
        state.rng.shuffle(disposed)
        state.library.extend(disposed)
    else:  # surveil
        state.graveyard.extend(disposed)
    complete_resolution(state)


def execute_scry_surveil_option(state, option):
    pending = state.pending_resolution
    if pending["remaining"]:
        card = pending["remaining"].pop(0)
        (pending["kept"] if option == "keep" else pending["disposed"]).append(card)
        if pending["remaining"]:
            return  # more cards still to decide
        if len(pending["kept"]) <= 1:
            _finish_scry_surveil(state)  # 0 or 1 kept -- no ordering choice to make
        else:
            pending["ordered"] = []  # 2+ kept -- enter the ordering phase
        return

    # Ordering phase: option is the name of the next card to place on top.
    idx = next(i for i, c in enumerate(pending["kept"]) if c.name == option)
    pending["ordered"].append(pending["kept"].pop(idx))
    if not pending["kept"]:
        _finish_scry_surveil(state)


def scry(state, n):
    """Scry n (Candy Trail's ETB): see begin_scry_surveil."""
    begin_scry_surveil(state, "scry", n, on_complete=lambda s: None)


def surveil(state, n):
    """Surveil n (Conduit Pylons' ETB, Tocasia's Dig Site's ability): see
    begin_scry_surveil."""
    begin_scry_surveil(state, "surveil", n, on_complete=lambda s: None)


def activate_expedition_map(state, permanent):
    """{2}, T, Sacrifice: search library for a land -- the model's choice
    (MULTI_DECK_PLAN.md Phase M4b: begins a search_fetch pending
    resolution instead of auto-picking via the old priority rule). Caller
    has already paid the {1} cost."""
    state.battlefield.remove(permanent)
    state.graveyard.append(permanent.card_def)

    def _on_chosen(state, land_name):
        found = find_and_remove_by_name(state, land_name)
        state.rng.shuffle(state.library)
        if found:
            state.hand.append(found)

    begin_search_fetch(state, lambda c: c.card_type == CardType.LAND, _on_chosen)


def cast_crop_rotation(state, card_def):
    """{G}, sacrifice a land: search library for a land, put it directly
    onto the battlefield (its own normal tapped/ETB rules apply), shuffle.
    Both the sacrifice target and the fetch target are the model's choice
    (MULTI_DECK_PLAN.md Phase M4b: begins a choose_permanent resolution
    for the sacrifice, chaining into a search_fetch resolution for the
    fetch, instead of the old auto-picked-fodder + priority-rule
    combination -- note the signature dropped the old land_to_sacrifice
    parameter, since it's no longer decided by the caller). Caller has
    already paid the {G} cost."""
    state.hand.remove(card_def)
    state.graveyard.append(card_def)

    def _on_sac_chosen(state, sac_name):
        if sac_name is None:
            return  # begin_choose_permanent found no valid sacrifice target -- fizzle, per Crop Rotation's cast legality this shouldn't happen, but don't crash if it somehow does
        sac_permanent = next(p for p in state.battlefield if p.card_def.name == sac_name)
        state.battlefield.remove(sac_permanent)
        state.graveyard.append(sac_permanent.card_def)

        def _on_fetch_chosen(state, land_name):
            found = find_and_remove_by_name(state, land_name)
            state.rng.shuffle(state.library)
            if found:
                enters_battlefield(state, found)

        begin_search_fetch(state, lambda c: c.card_type == CardType.LAND, _on_fetch_chosen)

    begin_choose_permanent(
        state,
        lambda p: p.card_def.card_type == CardType.LAND and p.card_def.effect_id != EffectId.TRON_LAND,
        _on_sac_chosen,
    )


def begin_ancient_stirrings(state, revealed, on_complete):
    """Pending resolution (MULTI_DECK_PLAN.md Phase M4d): the model picks
    at most one noncreature-colorless card among `revealed` to take, or
    declines -- a single decision, not a sequential walk like scry/surveil
    (Ancient Stirrings only ever takes one, if any). on_complete(state,
    chosen_card_or_None) runs once decided."""
    begin_resolution(state, "ancient_stirrings", on_complete, revealed=revealed)


def ancient_stirrings_options(state):
    revealed = state.pending_resolution["revealed"]
    eligible_names = sorted({c.name for c in revealed if is_noncreature_colorless(c)})
    return eligible_names + ["decline"]


def execute_ancient_stirrings_option(state, option):
    revealed = state.pending_resolution["revealed"]
    if option == "decline":
        chosen = None
    else:
        idx = next(i for i, c in enumerate(revealed) if c.name == option)
        chosen = revealed.pop(idx)
    state.rng.shuffle(revealed)  # whatever's left (all of it, if declined) goes to the bottom
    state.library.extend(revealed)
    complete_resolution(state, chosen)


def cast_ancient_stirrings(state, card_def):
    """{G}: look at top 5, may take one noncreature colorless card to hand
    -- the model's choice among eligible ones, or decline
    (MULTI_DECK_PLAN.md Phase M4d) -- rest to bottom in random order."""
    state.hand.remove(card_def)
    state.graveyard.append(card_def)
    top = state.library[:5]
    del state.library[:5]

    def _on_chosen(state, chosen):
        if chosen is not None:
            state.hand.append(chosen)

    begin_ancient_stirrings(state, top, _on_chosen)


def forestcycle_generous_ent(state, card_def):
    """{1}, discard this card from hand: search library for a Forest, put
    into hand, shuffle. Only one possible target name, so this resolves
    immediately -- no model choice/pending resolution needed, unlike
    Expedition Map/Crop Rotation's any-land search (MULTI_DECK_PLAN.md
    Phase M4b)."""
    state.hand.remove(card_def)
    state.graveyard.append(card_def)
    found = find_and_remove_by_name(state, "Forest")
    state.rng.shuffle(state.library)
    if found:
        state.hand.append(found)


def activate_bonders_ornament_draw(state, permanent):
    """{4}, T: draw a card (shares the tap cost with its plain mana ability)."""
    permanent.tapped = True
    state.draw(1)


def activate_candy_trail_sac(state, permanent):
    """{2}, T, Sacrifice: draw a card (lifegain omitted, see plan)."""
    state.battlefield.remove(permanent)
    state.graveyard.append(permanent.card_def)
    state.draw(1)


def activate_relic_of_progenitus(state, permanent):
    """{1}, Exile this artifact: draw a card (graveyard-exile tap ability
    omitted -- no-op with no opposing graveyard, see plan)."""
    state.battlefield.remove(permanent)  # exiled, not graveyard; exile is untracked
    state.draw(1)


def activate_tocasia_dig_site_surveil(state, permanent):
    """{3}, T: Surveil 1 (shares the tap cost with its plain {T}: Add {C})."""
    permanent.tapped = True
    surveil(state, 1)


# ---------------------------------------------------------------------------
# Effect registry (MULTI_DECK_PLAN.md Phase M1) -- one place per EffectId
# describing its mana output (if any), whether it enters tapped, its ETB
# trigger (if any), and its activated abilities (cost + resolve function).
# This is what makes a card reusable by a future deck without new code: an
# EffectId present here is "already implemented." Deck data (DECKLIST)
# still only describes Tron as of this phase -- nothing about the action
# table changes yet (that's Phase M4).
#
# Defined here, after every resolve function it references (surveil, scry,
# the activate_* functions) already exists by name, purely for
# readability -- functions defined earlier in this module (mana_output,
# enters_battlefield, Phase 2's pay_cost/choose_taps_for_cost) reference
# EFFECT_REGISTRY and the derived globals below safely regardless of
# textual order, since Python resolves a name inside a function body when
# that function is *called*, not when it's defined.
#
# "mana" shapes: ("tron",) -- Tron's controls-all-three-doubling rule;
# ("fixed", symbol) -- always produces that one symbol; ("flexible",
# {symbols}) -- caller chooses one of several. "filter_mana": {"colors":
# {...}} (MULTI_DECK_PLAN.md Phase M4) marks Barrels of Blasting Jelly's
# and Conduit Pylons' colored-pip filter ability (as opposed to Conduit
# Pylons' plain {T}: Add {C}, which IS a "fixed" mana source below) --
# offered by tap_cost_options only when exactly one colored pip of
# quantity 1 remains outstanding, same narrow scope plan_payment's own
# legality check already enforced.
#
# "cast": {"resolve": fn(state, card_def), "extra_legal": fn(state) ->
# bool (optional)} (MULTI_DECK_PLAN.md Phase M4e) -- present only on
# castable nonland cards, tells the generic action-table builder how to
# resolve casting this card once its cost is paid, and any rules
# precondition beyond cost/being-in-hand (e.g. Crop Rotation needs a
# non-Tron land to sacrifice). "forestcycle": {"cost_key", "resolve"} is
# the same idea for Generous Ent's alternate from-hand cost.
EFFECT_REGISTRY = {
    EffectId.TRON_LAND: {
        "mana": ("tron",),
    },
    EffectId.FOREST: {
        "mana": ("fixed", "G"),
    },
    EffectId.WOODED_RIDGELINE: {
        "mana": ("flexible", {"R", "G"}),
        "enters_tapped": True,
    },
    EffectId.BOJUKA_BOG: {
        "mana": ("fixed", "B"),
        "enters_tapped": True,
    },
    EffectId.TOCASIA_DIG_SITE: {
        "mana": ("fixed", "C"),
        "activated_abilities": {
            "surveil": {
                "cost_key": "surveil_ability_cost",
                "resolve": lambda state, permanent: activate_tocasia_dig_site_surveil(state, permanent),
            },
        },
    },
    EffectId.CONDUIT_PYLONS: {
        "mana": ("fixed", "C"),
        "etb_trigger": lambda state: surveil(state, 1),
        "filter_mana": {"colors": set(COLORS)},
    },
    EffectId.EXPEDITION_MAP: {
        "cast": {"resolve": lambda state, card_def: cast_permanent_from_hand(state, card_def)},
        "activated_abilities": {
            "activate": {
                "cost_key": "ability_cost",
                "resolve": lambda state, permanent: activate_expedition_map(state, permanent),
            },
        },
    },
    EffectId.CROP_ROTATION: {
        "cast": {
            "resolve": lambda state, card_def: cast_crop_rotation(state, card_def),
            "extra_legal": lambda state: any(
                p.card_def.card_type == CardType.LAND and p.card_def.effect_id != EffectId.TRON_LAND
                for p in state.battlefield
            ),
        },
    },
    EffectId.ANCIENT_STIRRINGS: {
        "cast": {"resolve": lambda state, card_def: cast_ancient_stirrings(state, card_def)},
    },
    EffectId.BONDERS_ORNAMENT: {
        "mana": ("flexible", set(COLORS)),
        "cast": {"resolve": lambda state, card_def: cast_permanent_from_hand(state, card_def)},
        "activated_abilities": {
            "draw": {
                "cost_key": "draw_ability_cost",
                "resolve": lambda state, permanent: activate_bonders_ornament_draw(state, permanent),
            },
        },
    },
    EffectId.CANDY_TRAIL: {
        "etb_trigger": lambda state: scry(state, 2),
        "cast": {"resolve": lambda state, card_def: cast_permanent_from_hand(state, card_def)},
        "activated_abilities": {
            "sac": {
                "cost_key": "sac_ability_cost",
                "resolve": lambda state, permanent: activate_candy_trail_sac(state, permanent),
            },
        },
    },
    EffectId.BARRELS_OF_BLASTING_JELLY: {
        "cast": {"resolve": lambda state, card_def: cast_permanent_from_hand(state, card_def)},
        "filter_mana": {"colors": set(COLORS)},
    },
    EffectId.RELIC_OF_PROGENITUS: {
        "cast": {"resolve": lambda state, card_def: cast_permanent_from_hand(state, card_def)},
        "activated_abilities": {
            "draw": {
                "cost_key": "draw_ability_cost",
                "resolve": lambda state, permanent: activate_relic_of_progenitus(state, permanent),
            },
        },
    },
    EffectId.GENEROUS_ENT: {
        # Never hard-cast in this deck (no "cast" key) -- only forestcycled.
        "forestcycle": {
            "cost_key": "forestcycling_cost",
            "resolve": lambda state, card_def: forestcycle_generous_ent(state, card_def),
        },
    },
    EffectId.FILLER: {},
}

# Derived views, kept as module-level names for backward compatibility with
# every existing caller (game.py's own Phase 2 mana functions and Phase 5
# heuristic, plus rewards.py's resource_quality_components) -- these used
# to be the hand-authored source of truth; EFFECT_REGISTRY is now.
SIMPLE_MANA_SOURCE_EFFECTS = {
    effect_id for effect_id, spec in EFFECT_REGISTRY.items() if spec.get("mana") is not None
}
_FIXED_SOURCE_COLOR = {
    effect_id: spec["mana"][1]
    for effect_id, spec in EFFECT_REGISTRY.items()
    if spec.get("mana", (None,))[0] == "fixed"
}
_FLEXIBLE_SOURCE_CHOICES = {
    effect_id: spec["mana"][1]
    for effect_id, spec in EFFECT_REGISTRY.items()
    if spec.get("mana", (None,))[0] == "flexible"
}
ENTERS_TAPPED_EFFECTS = {
    effect_id for effect_id, spec in EFFECT_REGISTRY.items() if spec.get("enters_tapped")
}


def _phase3_sanity_check():
    rng = random.Random(0)

    # 3rd Tron land ETBs -> turn_won is set immediately, regardless of the
    # other two's tapped status (MULTI_DECK_PLAN.md Phase M3: there's only
    # one termination condition now, no "online" second tier that tapped
    # status used to be able to delay -- see rewards.py for the
    # Tron-specific "was it also fully untapped" scoring function that
    # replaces that distinction).
    state = GameState(on_the_play=True, rng=rng, terminated_fn=tron_terminated)
    state.turn_number = 3
    state.battlefield = [
        Permanent(CARD_DEFS["Urza's Mine"], tapped=True),
        Permanent(CARD_DEFS["Urza's Power Plant"]),
    ]
    enters_battlefield(state, CARD_DEFS["Urza's Tower"])
    assert state.turn_won == 3, state.turn_won

    # Expedition Map: search offers every land in the library; the model
    # explicitly chooses which to fetch (MULTI_DECK_PLAN.md Phase M4b --
    # no more auto-priority pick).
    state3 = GameState(on_the_play=True, rng=rng)
    state3.turn_number = 2
    state3.battlefield = [Permanent(CARD_DEFS["Urza's Mine"])]
    state3.library = [CARD_DEFS["Rooftop Percher"], CARD_DEFS["Urza's Tower"], CARD_DEFS["Forest"]]
    map_permanent = Permanent(CARD_DEFS["Expedition Map"])
    state3.battlefield.append(map_permanent)
    activate_expedition_map(state3, map_permanent)
    assert state3.pending_resolution["kind"] == "search_fetch"
    assert search_fetch_options(state3) == ["Forest", "Urza's Tower"]
    execute_search_fetch_option(state3, "Urza's Tower")
    assert state3.hand == [CARD_DEFS["Urza's Tower"]], state3.hand
    assert len(state3.library) == 2
    assert map_permanent not in state3.battlefield
    assert state3.pending_resolution is None

    # Ancient Stirrings: offers every noncreature-colorless card among the
    # top 5 plus "decline"; the model's explicit choice determines what's
    # taken (MULTI_DECK_PLAN.md Phase M4d -- no more auto-priority pick).
    state4 = GameState(on_the_play=True, rng=rng)
    state4.turn_number = 2
    state4.battlefield = [Permanent(CARD_DEFS["Urza's Mine"])]
    state4.hand = [CARD_DEFS["Ancient Stirrings"]]
    state4.library = [
        CARD_DEFS["Rooftop Percher"], CARD_DEFS["Bramble Wurm"], CARD_DEFS["Urza's Power Plant"],
        CARD_DEFS["Maelstrom Colossus"], CARD_DEFS["Pinnacle Kill-Ship"],
        CARD_DEFS["Forest"],
    ]
    cast_ancient_stirrings(state4, CARD_DEFS["Ancient Stirrings"])
    assert state4.pending_resolution["kind"] == "ancient_stirrings"
    assert ancient_stirrings_options(state4) == ["Urza's Power Plant", "decline"], "the other 4 are creatures/filler, ineligible"
    execute_ancient_stirrings_option(state4, "Urza's Power Plant")
    assert state4.pending_resolution is None
    assert state4.hand == [CARD_DEFS["Urza's Power Plant"]], state4.hand
    assert len(state4.library) == 5
    assert CARD_DEFS["Urza's Power Plant"] not in state4.library


def _phase_m4b_sanity_check():
    """MULTI_DECK_PLAN.md Phase M4b: search effects as model-chosen
    fetch-by-name, Crop Rotation's sacrifice target as a model choice."""

    # Crop Rotation: two decisions in sequence -- which land to sacrifice,
    # then which land to fetch -- both the model's choice now.
    state = GameState(on_the_play=True)
    crop_rotation = CARD_DEFS["Crop Rotation"]
    state.hand = [crop_rotation]
    bog = Permanent(CARD_DEFS["Bojuka Bog"])
    forest_bf = Permanent(CARD_DEFS["Forest"])
    mine = Permanent(CARD_DEFS["Urza's Mine"])  # a Tron land -- must never be offered as sac fodder
    state.battlefield = [bog, forest_bf, mine]
    state.library = [CARD_DEFS["Urza's Tower"], CARD_DEFS["Rooftop Percher"]]

    cast_crop_rotation(state, crop_rotation)
    assert state.hand == [] and crop_rotation in state.graveyard
    assert state.pending_resolution["kind"] == "choose_permanent"
    assert choose_permanent_options(state) == ["Bojuka Bog", "Forest"], "Urza's Mine (a Tron land) must never be offered"

    execute_choose_permanent_option(state, "Bojuka Bog")
    assert bog not in state.battlefield and bog.card_def in state.graveyard
    assert state.pending_resolution["kind"] == "search_fetch", "chains straight into the fetch decision"
    assert search_fetch_options(state) == ["Urza's Tower"]

    execute_search_fetch_option(state, "Urza's Tower")
    assert state.pending_resolution is None
    assert any(p.card_def.name == "Urza's Tower" for p in state.battlefield)
    assert len(state.library) == 1

    # Forestcycle Generous Ent: only one possible target (Forest), so it
    # resolves immediately -- no pending resolution at all.
    state2 = GameState(on_the_play=True)
    ent = CARD_DEFS["Generous Ent"]
    state2.hand = [ent]
    state2.library = [CARD_DEFS["Rooftop Percher"], CARD_DEFS["Forest"], CARD_DEFS["Bramble Wurm"]]
    forestcycle_generous_ent(state2, ent)
    assert state2.pending_resolution is None
    assert state2.hand == [CARD_DEFS["Forest"]], state2.hand
    assert len(state2.library) == 2


def _phase_m4c_sanity_check():
    """MULTI_DECK_PLAN.md Phase M4c: scry/surveil as a pending resolution
    -- keep/dispose per card, then order the kept ones, all model-chosen."""

    # Scry 2, both kept: the model's chosen order wins, not original
    # library order -- proves ordering is real, not a no-op.
    state = GameState(on_the_play=True)
    state.library = [
        CARD_DEFS["Forest"], CARD_DEFS["Urza's Mine"],
        CARD_DEFS["Rooftop Percher"], CARD_DEFS["Bramble Wurm"],
    ]
    scry(state, 2)
    assert state.pending_resolution["kind"] == "scry"
    assert scry_surveil_options(state) == ["keep", "dispose"]
    execute_scry_surveil_option(state, "keep")   # Forest
    execute_scry_surveil_option(state, "keep")   # Urza's Mine
    assert sorted(scry_surveil_options(state)) == ["Forest", "Urza's Mine"], "2 kept -- ordering phase"
    execute_scry_surveil_option(state, "Urza's Mine")  # place 2nd-revealed card first
    execute_scry_surveil_option(state, "Forest")
    assert state.pending_resolution is None
    assert [c.name for c in state.library] == ["Urza's Mine", "Forest", "Rooftop Percher", "Bramble Wurm"]

    # Scry 2, one kept + one disposed: disposed goes to the bottom, no
    # ordering step needed for a single kept card.
    state2 = GameState(on_the_play=True)
    state2.library = [
        CARD_DEFS["Forest"], CARD_DEFS["Rooftop Percher"],
        CARD_DEFS["Bramble Wurm"], CARD_DEFS["Maelstrom Colossus"],
    ]
    scry(state2, 2)
    execute_scry_surveil_option(state2, "dispose")  # Forest -> bottom
    execute_scry_surveil_option(state2, "keep")      # Rooftop Percher -> stays on top
    assert state2.pending_resolution is None, "only 1 kept -- no ordering phase"
    assert [c.name for c in state2.library] == ["Rooftop Percher", "Bramble Wurm", "Maelstrom Colossus", "Forest"]

    # Surveil 1, disposed -> graveyard, not the library bottom.
    state3 = GameState(on_the_play=True)
    state3.library = [CARD_DEFS["Forest"], CARD_DEFS["Rooftop Percher"]]
    surveil(state3, 1)
    assert state3.pending_resolution["kind"] == "surveil"
    execute_scry_surveil_option(state3, "dispose")
    assert state3.pending_resolution is None
    assert [c.name for c in state3.library] == ["Rooftop Percher"]
    assert [c.name for c in state3.graveyard] == ["Forest"]

    # Surveil 1, kept -> stays on top, graveyard untouched.
    state4 = GameState(on_the_play=True)
    state4.library = [CARD_DEFS["Forest"], CARD_DEFS["Rooftop Percher"]]
    surveil(state4, 1)
    execute_scry_surveil_option(state4, "keep")
    assert state4.pending_resolution is None
    assert [c.name for c in state4.library] == ["Forest", "Rooftop Percher"]
    assert state4.graveyard == []


def _phase_m4d_sanity_check():
    """MULTI_DECK_PLAN.md Phase M4d: Ancient Stirrings' take-one-or-decline
    as a single-step pending resolution -- the decline path specifically
    (the "take a card" path is already covered in _phase3_sanity_check)."""
    state = GameState(on_the_play=True)
    stirrings = CARD_DEFS["Ancient Stirrings"]
    state.hand = [stirrings]
    state.library = [
        CARD_DEFS["Urza's Power Plant"], CARD_DEFS["Rooftop Percher"], CARD_DEFS["Bramble Wurm"],
        CARD_DEFS["Maelstrom Colossus"], CARD_DEFS["Pinnacle Kill-Ship"], CARD_DEFS["Forest"],
    ]
    cast_ancient_stirrings(state, stirrings)
    assert ancient_stirrings_options(state) == ["Urza's Power Plant", "decline"]
    execute_ancient_stirrings_option(state, "decline")
    assert state.pending_resolution is None
    assert stirrings not in state.hand
    assert len(state.library) == 6, "all 5 revealed cards go to the bottom when nothing's taken"
    assert sorted(c.name for c in state.library[-5:]) == sorted([
        "Urza's Power Plant", "Rooftop Percher", "Bramble Wurm", "Maelstrom Colossus", "Pinnacle Kill-Ship",
    ])
    assert state.library[0].name == "Forest", "the 6th, never-revealed card is still on top"


# ---------------------------------------------------------------------------
# Phase 4 — Turn loop
# ---------------------------------------------------------------------------

MAX_MAIN_PHASE_ACTIONS = 200  # guard against an infinite policy loop, not expected --
# bumped from 50 (MULTI_DECK_PLAN.md Phase M4e): a single "logical" action
# (cast a spell, activate an ability) now costs multiple loop iterations
# to fully resolve (one per mana tap, plus any search/scry/take decisions),
# where it used to cost exactly one.


def untap_step(state):
    for permanent in state.battlefield:
        permanent.tapped = False
        permanent.flags.pop("used_this_turn", None)  # Barrels of Blasting Jelly


def draw_step(state):
    if state.turn_number == 1 and state.on_the_play:
        return
    state.draw(1)


def run_turn(state, choose_action):
    """One full turn. `choose_action(state)` (Phase 5) returns either None
    ("pass," end the main phase) or a zero-arg callable that performs one
    complete action (mana payment + effect) when invoked."""
    state.turn_number += 1
    state.lands_played_this_turn = 0
    untap_step(state)
    draw_step(state)

    for _ in range(MAX_MAIN_PHASE_ACTIONS):
        if state.turn_won is not None:
            break  # already fixed; nothing left to observe
        action = choose_action(state)
        if action is None:
            break
        action()


def run_game(decklist, terminated_fn, rng, on_the_play, horizon, choose_action):
    state = new_game_state(decklist, terminated_fn, on_the_play, rng)
    while state.turn_number < horizon and state.turn_won is None:
        run_turn(state, choose_action)
    return state


def _phase4_sanity_check():
    rng = random.Random(0)
    state = run_game(TRON_DECKLIST, tron_terminated, rng, on_the_play=True, horizon=6, choose_action=lambda s: None)
    assert state.turn_number == 6, state.turn_number
    assert state.turn_won is None
    assert len(state.hand) == 7 + 5, len(state.hand)  # turn 1 no draw (on the play) + 5 more turns
    assert len(state.library) == 60 - 7 - 5, len(state.library)



# ---------------------------------------------------------------------------
# Phases 5/6 (the fixed greedy heuristic and its Monte Carlo driver) were
# removed outright in MULTI_DECK_PLAN.md Phase M5: every deck is always
# played by a DRL model now (see MULTI_DECK_PLAN.md's "no hand-coded
# heuristics anywhere" decision), and the deleted functions' own baseline
# numbers live on in DRL_PLAN.md/README.md as a historical comparison
# point, same treatment the reverted lookahead-search experiment got.
# Phase numbering below keeps its original gaps rather than renumbering,
# for the same reason.
# ---------------------------------------------------------------------------

def plan_payment(state, cost):
    """Pure (no mutation): decide how `cost` could be paid right now, via
    simple sources (Phase 2) or, for a single missing colored pip, the
    Barrels of Blasting Jelly / Conduit Pylons mana-filter fallback. Returns
    an opaque plan for execute_payment, or None if unpayable right now."""
    taps = choose_taps_for_cost(state, cost)
    if taps is not None:
        return ("simple", cost, taps)

    needed_colors = [c for c in COLORS if cost.get(c, 0) > 0]
    if len(needed_colors) == 1 and cost[needed_colors[0]] == 1:
        filtered_cost = {"generic": cost.get("generic", 0) + 1}
        filtered_taps = choose_taps_for_cost(state, filtered_cost)
        if filtered_taps is not None:
            used = {id(p) for p, _c in filtered_taps}
            barrels = next(
                (p for p in state.battlefield
                 if p.card_def.effect_id == EffectId.BARRELS_OF_BLASTING_JELLY
                 and not p.flags.get("used_this_turn", False)),
                None,
            )
            if barrels is not None:
                return ("filter", filtered_cost, filtered_taps, barrels)
            pylons = next(
                (p for p in state.battlefield
                 if p.card_def.effect_id == EffectId.CONDUIT_PYLONS
                 and not p.tapped and id(p) not in used),
                None,
            )
            if pylons is not None:
                return ("filter", filtered_cost, filtered_taps, pylons)
    return None


def execute_payment(state, plan):
    kind = plan[0]
    if kind == "simple":
        _, cost, taps = plan
        pay_cost(state, cost, taps)
        return
    _, filtered_cost, taps, filterer = plan
    pay_cost(state, filtered_cost, taps)
    if filterer.card_def.effect_id == EffectId.BARRELS_OF_BLASTING_JELLY:
        filterer.flags["used_this_turn"] = True
    else:
        filterer.tapped = True


# ---------------------------------------------------------------------------
# Phase 7 — Aggregation & output
# ---------------------------------------------------------------------------

def _mean(xs):
    return sum(xs) / len(xs) if xs else None


def _median(xs):
    if not xs:
        return None
    s = sorted(xs)
    mid = len(s) // 2
    if len(s) % 2 == 1:
        return s[mid]
    return (s[mid - 1] + s[mid]) / 2


def aggregate_results(results, horizon):
    """results: list of (terminated_turn_or_None, scores) pairs -- scores
    a fixed-length list, score 1 (the training reward) always first
    (MULTI_DECK_PLAN.md Phase M6, replacing the old Tron-specific
    (turn_assembled, turn_online) shape with one generic termination-rate
    column plus a mean/median per configured score). Score means/medians
    are computed across every game, not just terminated ones -- a failed
    game's scores are already zeroed by finalize_scores, so including them
    correctly drags the mean down to reflect overall policy quality, not
    just "how good are the wins." Terminated-turn mean/median has no such
    reading for a failure (there's no turn to average), so that one
    excludes them, same as the old assembled/online means always did."""
    n = len(results)
    rows = []
    for turn in range(1, horizon + 1):
        terminated_pct = 100 * sum(1 for t, _s in results if t is not None and t <= turn) / n
        rows.append((turn, terminated_pct))

    terminated_turns = [t for t, _s in results if t is not None]
    num_scores = len(results[0][1]) if results else 0
    score_summaries = [
        {
            "mean": _mean([s[i] for _t, s in results]),
            "median": _median([s[i] for _t, s in results]),
        }
        for i in range(num_scores)
    ]

    summary = {
        "terminated_mean_turn": _mean(terminated_turns),
        "terminated_median_turn": _median(terminated_turns),
        "never_pct": 100 * (n - len(terminated_turns)) / n,
        "scores": score_summaries,
    }
    return rows, summary


def _fmt(value, spec="{:.2f}"):
    return spec.format(value) if value is not None else "n/a"


def print_report(results, horizon):
    rows, summary = aggregate_results(results, horizon)
    print(f"{'Turn':>4}  {'Terminated %':>13}")
    for turn, terminated_pct in rows:
        print(f"{turn:>4}  {terminated_pct:>12.1f}%")
    print()
    print(f"Terminated: mean turn {_fmt(summary['terminated_mean_turn'])}, "
          f"median {_fmt(summary['terminated_median_turn'], '{:g}')}, "
          f"never by horizon: {summary['never_pct']:.1f}%")
    for i, s in enumerate(summary["scores"]):
        print(f"Score {i + 1}: mean {_fmt(s['mean'])}, median {_fmt(s['median'])}")


def _phase7_sanity_check():
    results = [(1, [10.0, 100.0]), (2, [20.0, 90.0]), (None, [0.0, 0.0]), (3, [30.0, 80.0])]
    rows, summary = aggregate_results(results, horizon=4)
    assert rows == [(1, 25.0), (2, 50.0), (3, 75.0), (4, 75.0)], rows
    assert summary["terminated_mean_turn"] == 2.0
    assert summary["terminated_median_turn"] == 2
    assert summary["never_pct"] == 25.0
    assert summary["scores"][0]["mean"] == 15.0, summary["scores"][0]
    assert summary["scores"][0]["median"] == 15.0, summary["scores"][0]
    assert summary["scores"][1]["mean"] == 67.5, summary["scores"][1]
    assert summary["scores"][1]["median"] == 85.0, summary["scores"][1]



# ---------------------------------------------------------------------------
# Phase 8 — Final sanity pass
# ---------------------------------------------------------------------------

def _phase8_hand_fed_scenario():
    """Hand-feed Mine/Power Plant/Tower into the opening hand with nothing
    else relevant: playing one per turn -- via direct hand-authored
    actions, not a heuristic (MULTI_DECK_PLAN.md Phase M5: policy_
    choose_action is gone) -- assembles Tron on turn 3."""
    def play_named_land(state, name):
        def choose_action(s):
            if s.lands_played_this_turn > 0:
                return None
            card_def = next(c for c in s.hand if c.name == name)
            return lambda: play_land_from_hand(s, card_def)
        run_turn(state, choose_action)

    rng = random.Random(0)
    state = GameState(on_the_play=True, rng=rng, terminated_fn=tron_terminated)
    state.hand = [
        CARD_DEFS["Urza's Mine"], CARD_DEFS["Urza's Power Plant"], CARD_DEFS["Urza's Tower"],
        CARD_DEFS["Rooftop Percher"], CARD_DEFS["Rooftop Percher"],
        CARD_DEFS["Bramble Wurm"], CARD_DEFS["Bramble Wurm"],
    ]
    state.library = [CARD_DEFS["Rooftop Percher"]] * 53

    play_named_land(state, "Urza's Mine")            # turn 1
    play_named_land(state, "Urza's Power Plant")     # turn 2
    play_named_land(state, "Urza's Tower")           # turn 3
    assert state.turn_won == 3, state.turn_won


def _phase8_sanity_check():
    _phase0_sanity_check()          # deck totals to 60
    _phase1_sanity_check()          # opening hand is 7
    _phase8_hand_fed_scenario()     # deterministic turn-3 assembly


if __name__ == "__main__":
    _phase0_sanity_check()
    print(f"Phase 0 OK: {len(TRON_DECKLIST)} distinct cards, 60 total copies.")

    _phase1_sanity_check()
    print("Phase 1 OK: shuffled 60-card library, drew 7 -> zone counts 53/7/0/0.")

    _phase2_sanity_check()
    print("Phase 2 OK: paying {G} with Forest+Mine untapped taps the Forest, not the Mine.")

    _phase3_sanity_check()
    print("Phase 3 OK: Metric A/B computation, Expedition Map, and Ancient Stirrings behave correctly.")

    _phase4_sanity_check()
    print("Phase 4 OK: an always-pass policy runs 6 turns, draws correctly, never assembles Tron.")

    # Phases 5/6 (the greedy heuristic and its Monte Carlo driver) were
    # removed in MULTI_DECK_PLAN.md Phase M5 -- see the comment where they
    # used to live, just above plan_payment.

    _phase7_sanity_check()
    print("Phase 7 OK: aggregation math matches a hand-computed example.")

    _phase8_sanity_check()
    print("Phase 8 OK: hand-fed Mine/PP/Tower assembles turn 3 via direct hand-authored actions.")

    _phase_m4a_sanity_check()
    print("Phase M4a OK: interactive mana payment (multi-tap generic, colored pip, Tron doubling, "
          "flexible color choice, Barrels/Pylons filter mode, abandon-payment) all behave correctly.")

    _phase_m4b_sanity_check()
    print("Phase M4b OK: search effects as model-chosen fetch-by-name, Crop Rotation's sacrifice "
          "target as a model choice.")

    _phase_m4c_sanity_check()
    print("Phase M4c OK: scry/surveil keep/dispose/order pending resolution, disposed destinations "
          "(bottom vs graveyard) all correct.")

    _phase_m4d_sanity_check()
    print("Phase M4d OK: Ancient Stirrings decline path -- all 5 revealed cards correctly go to the "
          "bottom, 6th unrevealed card stays on top.")
