"""Layer 1 -- independent oracle. Every token and every scalar of every real decision is recomputed from raw engine
state by encoder_corpus.oracle_tokens / oracle_scalar (a separate statement of the spec) and must agree exactly.
Comparison is as a MULTISET -- token order carries no meaning -- so a missing, extra or wrong token is all caught."""
import collections

import numpy as np

from support import corpus as ec
from rl.model import features as F


def _bag(idx, rows):
    return collections.Counter((int(i), tuple(np.round(r, 5))) for i, r in zip(idx, rows))


def _explain(f, k):
    got, want = _bag(f.vocab_idx, f.rows), _bag(f.expected_vocab_idx, f.expected_rows)
    lines = [f"frame {k} (pending={f.pending_kind}, phase={f.phase}, seat={f.seat}): "
             f"{len(f.rows)} tokens vs {len(f.expected_rows)} expected"]
    extra, missing = list(got - want), list(want - got)
    for tok in extra[:3]:
        near = next((w for w in missing if w[0] == tok[0]), None)
        if near is None:
            lines.append(f"  unexpected token, vocab {tok[0]}")
            continue
        diff = [(ec.DYN_COLUMNS[c - F.STATIC_FEATURE_DIM] if c >= F.STATIC_FEATURE_DIM else f"static[{c}]", a, b)
                for c, (a, b) in enumerate(zip(tok[1], near[1])) if a != b]
        lines.append(f"  vocab {tok[0]}: (got, expected) differ in {diff[:6]}")
    for tok in missing[:3]:
        if not any(g[0] == tok[0] for g in extra):
            lines.append(f"  missing token, vocab {tok[0]}")
    return "\n".join(lines)


def test_token_set_matches_the_oracle_on_every_decision(frames):
    bad = [k for k, f in enumerate(frames)
           if _bag(f.vocab_idx, f.rows) != _bag(f.expected_vocab_idx, f.expected_rows)]
    assert not bad, (f"{len(bad)}/{len(frames)} frames disagree with the oracle:\n"
                     + "\n".join(_explain(frames[k], k) for k in bad[:5]))


def test_scalar_features_match_the_oracle_on_every_decision(frames):
    bad = [k for k, f in enumerate(frames) if not np.allclose(f.scalar, f.expected_scalar, atol=1e-6)]
    if bad:
        f = frames[bad[0]]
        cols = np.flatnonzero(~np.isclose(f.scalar, f.expected_scalar, atol=1e-6))
        raise AssertionError(f"{len(bad)} frames disagree; first is {bad[0]}, scalar columns {cols.tolist()}")


# ---- coverage gate: the corpus must actually contain what the checks above claim to cover ------------------------------

def test_corpus_exercises_every_zone_and_the_interesting_columns(frames):
    seen = collections.Counter()
    for f in frames:
        for c in ec.DYN_COLUMNS:
            seen[c] += int((f.rows[:, ec.COL[c]] != 0).sum())
    must_have = (["untapped", "tapped", "eff_power", "eff_toughness", "attacking", "summoning_sick", "blocked_attacker",
                  "committed_blocker", "targeted_by_mine", "targeted_by_theirs", "revealed_focus", "revealed_kept",
                  "cost_reduction"] + [f"zone[{z}]" for z in ec.ZONE_ORDER])
    missing = [c for c in must_have if not seen[c]]
    assert not missing, f"corpus never exercised: {missing}"


def test_corpus_covers_both_seats_and_many_games(frames):
    assert {f.seat for f in frames} == {0, 1}
    assert len({f.game_id for f in frames}) >= 10


def test_corpus_covers_the_main_pending_kinds(frames):
    seen = {f.pending_kind for f in frames}
    expected = {None, "pay_cost", "choose_any_target", "declare_blockers", "choose_permanent", "discard", "search_fetch",
                "choose_graveyard_card", "scry", "ponder", "assign_combat_damage", "choose_opponent_permanent"}
    assert expected <= seen, f"corpus never reached pending kinds: {expected - seen}"


def test_corpus_covers_the_main_pointer_categories(frames):
    seen = {f.pointer_kind for f in frames} - {None}
    expected = {"declare_attackers", "declare_blockers", "choose_permanent", "choose_opponent_permanent",
                "choose_any_target", "assign_combat_damage", "choose_graveyard_card"}
    assert expected <= seen, f"pointer categories never exercised: {expected - seen}"
