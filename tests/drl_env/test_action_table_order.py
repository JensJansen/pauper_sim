"""Guards the ORDER and CONTENT of the fixed action table.

Each deck's policy has an action head sized and indexed by its table
(DeckNetwork.non_targeting_head: one logit per row). A trained checkpoint is
therefore only meaningful against the exact table it was trained on: reorder
or relabel rows without changing the length and every old checkpoint still
loads, silently pointing each learned logit at a different action. Nothing
else in the suite would notice.

So this file pins, against tests/drl_env/action_table_snapshot.json:
  1. the production (pointer-filtered) table's labels, in order, per roster deck;
  2. a digest per decklist variant of the FULL table's row structure -- every
     row's label plus its legal/execute function identity, captured closure
     values, defaults and attributes -- over the roster decks (plain, with
     tokens, with an opponent, with extra choosable names) and every catalog
     card alone, so a branch no roster deck reaches is still covered;
  3. that the head size of each saved main-league checkpoint on disk equals
     its deck's current table length (skipped where no checkpoints exist).

A FAILURE HERE MEANS: either you changed the table by accident, or you
intended to and every checkpoint trained on the old table is invalidated.
Only in the second case, regenerate the snapshot:
    python tests/drl_env/test_action_table_order.py --regenerate
and retrain (or migrate) the affected decks.
"""
import hashlib
import json
import os
import sys
import types
from pathlib import Path

import pytest

_REPO = Path(__file__).resolve().parents[2]
_SRC = _REPO / "src"
SNAPSHOT = Path(__file__).with_name("action_table_snapshot.json")
LEAGUE = "main-league"

pytestmark = pytest.mark.slow


def _norm(v, depth=0):
    """Id-free description of a closure cell / default / attribute value."""
    if depth > 4:
        return "<deep>"
    if isinstance(v, types.FunctionType):
        return f"fn:{_fn_name(v)}"
    if isinstance(v, (types.BuiltinFunctionType, types.MethodType)):
        return f"fn:{getattr(v, '__module__', '?')}.{getattr(v, '__qualname__', '?')}"
    if isinstance(v, (int, float, str, bool, type(None))):
        return repr(v)
    if isinstance(v, (tuple, list)):
        return "[" + ",".join(_norm(x, depth + 1) for x in v) + "]"
    if isinstance(v, (set, frozenset)):
        return "{" + ",".join(sorted(_norm(x, depth + 1) for x in v)) + "}"
    if isinstance(v, dict):
        items = sorted(v.items(), key=lambda kv: str(kv[0]))
        return "{" + ",".join(f"{_norm(k, depth + 1)}:{_norm(x, depth + 1)}" for k, x in items) + "}"
    name = getattr(v, "name", None)
    if isinstance(name, str):
        return f"{type(v).__name__}:{name}"
    return f"<{type(v).__name__}>"


def _fn_name(f):
    """A function's identity. A lambda is named by what it does (the names and constants its code
    uses), not by the function it happens to be defined in, so moving it between helpers is not a
    change but editing it is."""
    if f.__name__ == "<lambda>":
        consts = [c for c in f.__code__.co_consts if isinstance(c, (str, int, float, bool, type(None)))]
        return f"{f.__module__}.<lambda>{f.__code__.co_names}{consts}"
    return f"{f.__module__}.{f.__qualname__}"


def _fn_fp(f):
    return {"fn": _fn_name(f),
            "closure": [_norm(c.cell_contents) for c in (f.__closure__ or ())],
            "defaults": _norm(f.__defaults__),
            "attrs": {k: _norm(v) for k, v in sorted(vars(f).items())}}


def _digest(actions):
    rows = [[label, _fn_fp(legal), _fn_fp(execute)] for label, legal, execute in actions]
    return hashlib.sha256(json.dumps(rows, sort_keys=True).encode()).hexdigest()[:16]


def compute_snapshot():
    """Builds the current labels + digests. Needs cwd == src/ (rl.roster's relative data paths)."""
    import game
    import drl_env
    from rl.roster import TOKEN_DEFS, build_pool

    decklists, _vocab, _ctxs, fixed_tables = build_pool()
    names = sorted(decklists)
    labels = {n: [row[0] for row in fixed_tables[n]] for n in names}
    digests = {}
    for i, n in enumerate(names):
        other = decklists[names[(i + 3) % len(names)]]
        variants = {
            f"roster:{n}": {"decklist": decklists[n]},
            f"roster+tokens:{n}": {"decklist": decklists[n], "token_card_defs": tuple(TOKEN_DEFS)},
            f"roster+opponent:{n}": {"decklist": decklists[n], "opponent_decklist": other,
                                     "opponent_token_card_defs": tuple(TOKEN_DEFS)},
            f"roster+extra:{n}": {"decklist": decklists[n], "extra_choosable_names": ("Zzz Extra", "Aaa Extra")},
        }
        for key, kw in variants.items():
            digests[key] = _digest(drl_env.build_action_table(kw.pop("decklist"), game.EFFECT_REGISTRY, **kw))
    for card in sorted(game.CARD_DEFS):
        digests[f"single:{card}"] = _digest(
            drl_env.build_action_table([(card, 2)], game.EFFECT_REGISTRY, token_card_defs=tuple(TOKEN_DEFS)))
    return {"labels": labels, "digests": digests}


@pytest.fixture(scope="module")
def current():
    mp = pytest.MonkeyPatch()
    mp.chdir(_SRC)  # rl.roster loads "../data/..." relative to src/
    try:
        yield compute_snapshot()
    finally:
        mp.undo()


@pytest.fixture(scope="module")
def pinned():
    return json.loads(SNAPSHOT.read_text(encoding="utf-8"))


_REGEN = "If this change is deliberate it invalidates every checkpoint trained on the old table; regenerate with `python tests/drl_env/test_action_table_order.py --regenerate`."


def test_production_table_labels_keep_their_order(current, pinned):
    problems = []
    for deck, expected in pinned["labels"].items():
        got = current["labels"].get(deck)
        if got == expected:
            continue
        if got is None:
            problems.append(f"{deck}: deck no longer in the roster")
            continue
        first = next((i for i, (a, b) in enumerate(zip(expected, got)) if a != b), min(len(expected), len(got)))
        problems.append(f"{deck}: table length {len(expected)} -> {len(got)}, first difference at row {first} "
                        f"({expected[first] if first < len(expected) else None!r} -> {got[first] if first < len(got) else None!r})")
    assert not problems, "action table order/labels changed:\n  " + "\n  ".join(problems) + "\n" + _REGEN


def test_full_table_structure_is_unchanged(current, pinned):
    changed = sorted(k for k, v in pinned["digests"].items() if current["digests"].get(k) != v)
    assert not changed, (f"{len(changed)} table variant(s) changed structure, e.g. {changed[:5]}\n" + _REGEN)


def test_head_size_of_saved_checkpoints_matches_the_table(current):
    import torch
    from repo_paths import CHECKPOINTS_DIR

    league = Path(CHECKPOINTS_DIR) / LEAGUE
    checked = 0
    for deck, labels in current["labels"].items():
        path = league / deck / "live.pt"
        if not path.exists():
            continue
        saved = torch.load(path, map_location="cpu", weights_only=False)["net"]
        assert saved["non_targeting_head.weight"].shape[0] == len(labels), (
            f"{deck}: checkpoint head has {saved['non_targeting_head.weight'].shape[0]} actions but the table now "
            f"has {len(labels)} -- the saved policy no longer matches this table")
        checked += 1
    if not checked:
        pytest.skip(f"no {LEAGUE} checkpoints on disk")


if __name__ == "__main__":
    if "--regenerate" not in sys.argv:
        raise SystemExit(__doc__)
    sys.path.insert(0, str(_SRC))
    os.chdir(_SRC)
    snap = compute_snapshot()
    SNAPSHOT.write_text(json.dumps(snap, indent=1, sort_keys=True) + "\n", encoding="utf-8")
    print(f"wrote {SNAPSHOT}: {len(snap['labels'])} decks, {len(snap['digests'])} digests")
