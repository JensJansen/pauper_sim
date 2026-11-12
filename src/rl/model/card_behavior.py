"""What a card DOES, derived automatically from the engine's own code -- the part of a card's static feature vector that
printed stats and registry keys cannot see.

The base static vector (rl.model.features) knows a card's cost, type, P/T, keywords and which top-level registry keys it
has. It cannot see what a `resolve` actually does or how big its numbers are, so Cast Down (destroy a creature) and
Unexpected Fangs (a counter), Lightning Bolt (3 damage) and Galvanic Blast (2, or 4 with metalcraft), Dispel and Mental
Note all collapsed to the same vector. This module reads the card's code instead:

  identifiers   the engine primitives the card's code reaches (function and attribute names such as `mill`, `draw`,
                `_cast_burn_any_target`, `card_type`), following helper calls a few levels deep. Kept only when 2..MAX_DF
                cards reach the name: one card alone is that card's identity (the embedding's job), a name nearly every
                card touches says nothing. Columns with identical card sets are merged.
  constants     the small integers in the card's OWN code (the amount in "deals 3 damage", "mill two"), so magnitudes
                separate cards that call the same helper.
  categoricals  every string-valued entry of CardDef.extra (e.g. tron_type = Mine / Tower / Power Plant), one-hot.

All three vocabularies are scanned from the registry, like the rest of rl.model.features: a new card is described the
moment it is registered (and, as with every widening of the feature vector, existing checkpoints stop loading).
Deterministic: sorted everywhere, no randomness."""
import functools
import types

import game

MAX_DEPTH = 3     # helper-call levels followed from a card's own functions
MAX_DF = 40       # an identifier reached by more cards than this is boilerplate
MIN_DF = 2        # ... and one reached by a single card is just that card's identity
MAX_CONSTANT = 12  # small integers only: amounts, counts, thresholds


def _callables(value, out):
    """Every function reachable from one registry spec value (functions, partials, containers of either)."""
    if isinstance(value, types.FunctionType):
        out.append(value)
    elif isinstance(value, functools.partial):
        out.append(value.func)
        for a in value.args:
            _callables(a, out)
    elif isinstance(value, dict):
        for v in value.values():
            _callables(v, out)
    elif isinstance(value, (tuple, list, set, frozenset)):
        for v in value:
            _callables(v, out)


def _engine_function(obj):
    return isinstance(obj, types.FunctionType) and obj.__module__.startswith("game")


def _walk(fn, depth, reached):
    """Adds fn and every engine function it calls (to MAX_DEPTH) to `reached` ({function: shallowest depth})."""
    if fn in reached and reached[fn] <= depth:
        return
    reached[fn] = depth
    if depth >= MAX_DEPTH:
        return
    codes = [fn.__code__]
    while codes:
        code = codes.pop()
        for const in code.co_consts:
            if isinstance(const, types.CodeType):
                codes.append(const)
        for name in code.co_names:
            target = fn.__globals__.get(name)
            if _engine_function(target):
                _walk(target, depth + 1, reached)
    for cell in fn.__closure__ or ():
        try:
            content = cell.cell_contents
        except ValueError:
            continue
        if _engine_function(content):
            _walk(content, depth + 1, reached)


def _code_parts(fn):
    """(identifiers, small integer constants) of fn's code including nested lambdas/inner functions."""
    names, consts = set(), set()
    codes = [fn.__code__]
    while codes:
        code = codes.pop()
        for const in code.co_consts:
            if isinstance(const, types.CodeType):
                codes.append(const)
            elif isinstance(const, int) and not isinstance(const, bool) and 0 <= const <= MAX_CONSTANT:
                consts.add(const)
        names.update(code.co_names)
    for cell in fn.__closure__ or ():
        try:
            content = cell.cell_contents
        except ValueError:
            continue
        if isinstance(content, int) and not isinstance(content, bool) and 0 <= content <= MAX_CONSTANT:
            consts.add(content)
    return names, consts


def _reach(card_def):
    spec = game.EFFECT_REGISTRY.get(card_def.effect_id, {})
    roots = []
    for value in spec.values():
        _callables(value, roots)
    reached = {}
    for fn in roots:
        _walk(fn, 0, reached)
    return reached


def _card_names_and_constants(reached, reach_count):
    """Identifiers from every reached function; constants only from the card's OWN code: its spec callables and the
    function they call directly (depth <= 1), plus any helper reached by this card alone."""
    names, consts = set(), set()
    for fn, depth in reached.items():
        n, c = _code_parts(fn)
        names |= n
        if depth <= 1 or reach_count[fn] == 1:
            consts |= c
    return names, consts


def _build():
    defs = list(game.CARD_DEFS.values())
    reaches = {cd.name: _reach(cd) for cd in defs}
    reach_count = {}
    for reached in reaches.values():
        for fn in reached:
            reach_count[fn] = reach_count.get(fn, 0) + 1
    sig = {name: _card_names_and_constants(reached, reach_count) for name, reached in reaches.items()}

    df = {}
    for names, _consts in sig.values():
        for n in names:
            df[n] = df.get(n, 0) + 1
    columns = {}  # frozenset of cards -> representative identifier (merge columns that always co-occur)
    for ident in sorted(n for n, c in df.items() if MIN_DF <= c <= MAX_DF):
        holders = frozenset(name for name, (names, _c) in sig.items() if ident in names)
        columns.setdefault(holders, ident)
    identifiers = tuple(sorted(columns.values()))
    constants = tuple(sorted({k for _names, consts in sig.values() for k in consts}))
    categoricals = tuple(sorted({(k, v) for cd in defs for k, v in cd.extra.items() if isinstance(v, str)}))
    return reach_count, identifiers, constants, categoricals


_REACH_COUNT, IDENTIFIER_VOCAB, CONSTANT_VOCAB, CATEGORICAL_VOCAB = _build()
# every raw identifier that maps onto a kept column: all names sharing the representative's card set
_IDENT_SET = frozenset(IDENTIFIER_VOCAB)
BEHAVIOR_FEATURE_DIM = len(IDENTIFIER_VOCAB) + len(CONSTANT_VOCAB) + len(CATEGORICAL_VOCAB)
_CACHE = {}


def behavior_features(card_def):
    """[BEHAVIOR_FEATURE_DIM] floats: identifier multi-hot, constant multi-hot, categorical one-hot. Works for any
    CardDef (tokens and back faces included); names outside the vocabulary are simply not represented."""
    cached = _CACHE.get(card_def.name)
    if cached is not None:
        return cached
    reached = _reach(card_def)
    counts = {fn: _REACH_COUNT.get(fn, 0) + (0 if fn in _REACH_COUNT else 1) for fn in reached}
    names, consts = _card_names_and_constants(reached, counts)
    # a name stands for its merged column if it IS the representative, or shares the representative's holders; since
    # merged names always co-occur, checking the representative alone is exact for CARD_DEFS cards
    out = [1.0 if ident in names else 0.0 for ident in IDENTIFIER_VOCAB]
    out += [1.0 if k in consts else 0.0 for k in CONSTANT_VOCAB]
    out += [1.0 if card_def.extra.get(key) == value else 0.0 for key, value in CATEGORICAL_VOCAB]
    _CACHE[card_def.name] = out
    return out
