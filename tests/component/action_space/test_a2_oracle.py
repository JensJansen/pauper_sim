"""Layer A2 -- the legality mask against an independent oracle, over real decisions from every ordered deck pair.

At each decision of random-legal self-play (`support/action_corpus.py`) the set of fixed-table rows the production mask
allows is compared with `support.action_oracle.oracle_labels`, written from the rules and the card data and sharing none
of drl_env's legality code. Compared two ways, over the families the oracle covers (pass, land, cast and its alternative
shapes, flashback/escape/plot, activate, cycle, impulse, mana taps, pool spend):

  unsound          the mask offers an action the oracle says is illegal
  incomplete       the oracle says an action is legal, the table has a row for it, the mask refuses it
  unrepresentable  the oracle says an action is legal and the table has no row that could express it

A disagreement is either explained by a FINDING (see FINDINGS below: a known engine behaviour that differs from the real
rules, awaiting the owner) or it is a failure. Findings are not hidden: this file lists them, counts them, and fails when
one stops occurring (so a fix cannot leave a stale entry behind) or a new one appears.
"""
from support import action_oracle as O

# Open questions for the owner. Status is "open" until the owner either authorizes the behaviour (then it moves to
# support/action_oracle.AUTHORIZED, the engine code gets an AUTHORIZED SIMPLIFICATION marker and the oracle mirrors it) or
# asks for it to be fixed (then the engine changes and the entry is deleted).
FINDINGS = {}
# Resolved findings (kept so the history is visible; each has a regression test in test_a4_copies_and_costs.py):
#   F-01  an Aura on the OTHER player's land did nothing for the land's controller (game/mana.py::_enchanting looked only
#         at the active player's battlefield).                      FIXED 2026-11-12, owner decision.
#   F-02  same-named permanents were indistinguishable: Tap X / Activate X acted on the first copy only. Now the player
#         picks among NON-interchangeable copies (choose_source mana sub-decision / choose_permanent). FIXED 2026-11-12.
#   F-03  a granted ability on a multi-symbol source was dropped from affordability (game.mana.source_mana_units).
#         FIXED 2026-11-12 (EitherOr supply unit; owner decision).
#   (cost-key abilities required an untapped copy even without {T} in the cost -> per-ability "taps" key, FIXED 2026-11-12.)


def test_no_oracle_errors_and_no_game_crashes(action_corpus):
    assert not action_corpus.errors, action_corpus.errors[:3]
    assert not action_corpus.oracle_errors, action_corpus.oracle_errors[:3]


def test_the_mask_never_offers_an_action_the_oracle_calls_illegal(action_corpus):
    assert action_corpus.n_unsound == 0, f"{action_corpus.n_unsound} unexplained, e.g.:\n" + "\n".join(
        str(x)[:500] for x in action_corpus.unsound[:3])


def test_the_mask_never_refuses_an_action_the_oracle_calls_legal(action_corpus):
    assert action_corpus.n_incomplete == 0, f"{action_corpus.n_incomplete} unexplained, e.g.:\n" + "\n".join(
        str(x)[:500] for x in action_corpus.incomplete[:3])


def test_every_oracle_legal_action_has_a_row_to_express_it(action_corpus):
    assert action_corpus.n_unrepresentable == 0, f"{action_corpus.n_unrepresentable} unexplained, e.g.:\n" + "\n".join(
        str(x)[:500] for x in action_corpus.unrepresentable[:3])


def test_known_findings_are_still_observed_and_nothing_new_appeared(action_corpus):
    seen = {f for (_kind, f) in action_corpus.by_finding}
    assert seen <= set(FINDINGS), f"disagreements attributed to unknown findings: {seen - set(FINDINGS)}"
    gone = sorted(f for f in FINDINGS if f not in seen)
    assert not gone, (f"findings {gone} no longer occur in the corpus -- if the engine was fixed or the behaviour "
                      f"authorized, remove the entry (and move an authorization to action_oracle.AUTHORIZED)")


def test_the_corpus_is_big_enough_to_mean_something(action_corpus):
    c = action_corpus
    assert c.games >= 121 and c.decisions >= 250 * c.games, (c.decisions, c.games)  # every ordered deck pair, ~340 decisions each
    for fam in sorted(O.COVERED):
        assert c.family_legal[fam] >= 25, f"oracle family {fam!r} was legal only {c.family_legal[fam]} times -- too thin to trust"


def test_report_uncovered_families(action_corpus, capsys):
    """Not an assertion: states plainly what the oracle does NOT yet judge, so a green run is not read as more than it is."""
    with capsys.disabled():
        print("\n[A2] oracle-covered legal rows:", dict(action_corpus.family_legal))
        print("[A2] legal rows in families the oracle does not judge yet:", dict(action_corpus.uncovered_legal))
        print("[A2] explained by open findings:", dict(action_corpus.by_finding))
