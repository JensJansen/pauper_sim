"""Layer A1 -- purity and determinism of the legality computation, on live decisions from real games.

Asking "what is legal?" must be a pure function of the position. For every sampled decision, on the LIVE state:

  * computing the mask (fixed table sweep AND pointer half) leaves a deep fingerprint of the state unchanged -- this
    includes the rng position, so a mask that consumed randomness would fail;
  * computing it twice, and on a deepcopy of the state, gives the same answer;
  * the sweep (`legal_action_mask`, with its sweep-scoped caches) equals evaluating every row's closure one at a time;
  * a bigger table (opponent rows, extra choosable names) agrees with the production table on every shared label.
"""
import copy
import os

import numpy as np
import pytest

import drl_env
import game
from drl_env import _actions_mana
from rl.decision import action_bridge as AB
from rl.roster import TOKEN_DEFS
from support import action_corpus as AC
from support.fingerprint import state_fingerprint
from support.paths import SRC

EVERY = int(os.environ.get("ACTION_PURITY_EVERY", "6"))  # fingerprint every Nth decision (the walk is the expensive part)


def _row_by_row(state, table):
    """Each row's legal() called alone, caches reset between calls, as a legal_action_mask-independent evaluation.
    Rows stamped with _mana_subdecision_gate are not total functions (they dereference the open sub-decision with no
    None guard, by documented design), so they are skipped -- exactly as the sweep skips them -- while none is open."""
    out = []
    for _label, legal, _ex in table:
        if getattr(legal, "_mana_subdecision_gate", None) is not None:
            out.append(False)
            continue
        _actions_mana._mana_ability_options_cache = None
        _actions_mana._mana_source_cache = None
        _actions_mana._filter_source_cache = None
        game.reset_mana_cache()
        out.append(bool(legal(state)))
    _actions_mana._mana_ability_options_cache = None
    _actions_mana._mana_source_cache = None
    _actions_mana._filter_source_cache = None
    game.reset_mana_cache()
    return out


@pytest.fixture(scope="module")
def purity_run():
    prev = os.getcwd()
    os.chdir(SRC)
    fails = {"mutates": [], "unstable": [], "copy_differs": [], "sweep_vs_rows": [], "variant_differs": []}
    n = {"decisions": 0, "fingerprinted": 0, "rows_checked": 0, "variants": 0}
    big_tables = {}
    decklists = None

    def on_decision(state, seat, deck, dec, legal, ctxs, tables):
        nonlocal decklists
        n["decisions"] += 1
        if n["decisions"] % EVERY:
            return
        if decklists is None:
            from rl.roster import build_pool
            decklists = build_pool()[0]
        table = dec.fixed_table
        before = state_fingerprint(state)
        m1 = drl_env.legal_action_mask(state, table)
        if AB.any_pointer_legal(state):
            AB.pointer_legal_mask(state, dec.identities)  # the pointer half is part of "computing the mask"
        after = state_fingerprint(state)
        n["fingerprinted"] += 1
        ctx = AC.context(state, seat, deck)
        if before != after:
            fails["mutates"].append(ctx)
        m2 = drl_env.legal_action_mask(state, table)
        if not np.array_equal(m1, m2):
            fails["unstable"].append(ctx)
        clone = copy.deepcopy(state)
        if not np.array_equal(drl_env.legal_action_mask(clone, table), m1):
            fails["copy_differs"].append(ctx)
        if state.active_mana_subdecision is None:
            rows = _row_by_row(state, table)
            n["rows_checked"] += len(rows)
            if rows != [bool(x) for x in m1]:
                bad = [table[i][0] for i, (a, b) in enumerate(zip(rows, m1)) if a != bool(b)]
                fails["sweep_vs_rows"].append({"rows": bad[:5], **ctx})
        key = deck
        if key not in big_tables:
            other = decklists[sorted(decklists)[(sorted(decklists).index(deck) + 3) % len(decklists)]]
            big_tables[key] = drl_env.build_action_table(
                decklists[deck], game.EFFECT_REGISTRY, token_card_defs=tuple(TOKEN_DEFS), opponent_decklist=other,
                opponent_token_card_defs=tuple(TOKEN_DEFS), extra_choosable_names=("Zzz Extra",))
        big = big_tables[key]
        bm = drl_env.legal_action_mask(state, big)
        by_label = {big[i][0]: bool(bm[i]) for i in range(len(big))}
        n["variants"] += 1
        diff = [table[i][0] for i in range(len(table)) if table[i][0] in by_label and by_label[table[i][0]] != bool(m1[i])]
        if diff:
            fails["variant_differs"].append({"rows": diff[:5], **ctx})

    AC.IMPURE.clear()
    AC.PURITY_EVERY = EVERY
    try:
        played, errors = AC.play_games(on_decision, games_per_pair=1, seed=11)
    finally:
        AC.PURITY_EVERY = 0
        os.chdir(prev)
    fails["mutates"] += AC.IMPURE  # the first computation of each decision's mask, observed from inside the driver
    return fails, n, errors


def _show(items):
    return "\n".join(str(x)[:400] for x in items[:3])


def test_run_was_substantial(purity_run):
    _fails, n, errors = purity_run
    assert not errors, errors[:3]
    assert n["fingerprinted"] >= 3000 and n["rows_checked"] > 300_000, n


def test_computing_the_mask_does_not_change_the_state(purity_run):
    assert not purity_run[0]["mutates"], _show(purity_run[0]["mutates"])


def test_the_mask_is_stable_across_calls(purity_run):
    assert not purity_run[0]["unstable"], _show(purity_run[0]["unstable"])


def test_a_deepcopy_of_the_state_has_the_same_mask(purity_run):
    assert not purity_run[0]["copy_differs"], _show(purity_run[0]["copy_differs"])


def test_the_sweep_equals_evaluating_every_row_alone(purity_run):
    assert not purity_run[0]["sweep_vs_rows"], _show(purity_run[0]["sweep_vs_rows"])


def test_a_bigger_table_agrees_on_every_shared_row(purity_run):
    assert not purity_run[0]["variant_differs"], _show(purity_run[0]["variant_differs"])
