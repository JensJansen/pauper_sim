"""Random-legal self-play for the action-space suite.

`play_games` drives real games (every ordered deck pair) with a pluggable policy over the mask the production code
builds (`rl.decision.agent._build_decision`: fixed table + pointer half), calling `on_decision` at every non-pregame
decision on the LIVE state. `build_corpus` uses it to compare the mask with the independent oracle
(`support.action_oracle`) and to count which rows / pending kinds / pointer kinds ever became legal.

Everything kept is plain data (labels, counters, short context dicts), never live game objects, so the result pickles.
"""
import os
import random
from collections import Counter, defaultdict
from dataclasses import dataclass, field

import numpy as np

import game
import rl.decision.agent as A
from rl.decision.action_bridge import pointer_kind
from rl.roster import build_pool
from support import action_oracle as O
from support.fingerprint import state_fingerprint
from support.paths import REPO, SRC  # noqa: F401

HORIZON = 14
MAX_DIFF_FRAMES = 400  # the first few hundred disagreements are plenty to triage; the counters stay exact


@dataclass
class ActionCorpus:
    decisions: int = 0
    games: int = 0
    errors: list = field(default_factory=list)          # (deck_a, deck_b, game_idx, repr) -- an exception out of a game
    by_kind: Counter = field(default_factory=Counter)    # pending kind (or "none") -> decisions
    by_pointer: Counter = field(default_factory=Counter)
    legal_rows: dict = field(default_factory=lambda: defaultdict(Counter))  # deck -> row label -> times legal
    table_labels: dict = field(default_factory=dict)     # deck -> tuple of fixed-table labels, in order
    family_legal: Counter = field(default_factory=Counter)  # family -> times a row of it was legal
    uncovered_legal: Counter = field(default_factory=Counter)  # family the oracle does not cover -> times legal
    unsound: list = field(default_factory=list)          # mask offered a covered label the oracle says is illegal
    incomplete: list = field(default_factory=list)       # oracle says legal, table has the row, mask says no
    unrepresentable: list = field(default_factory=list)  # oracle says legal, the table has no such row at all
    oracle_errors: list = field(default_factory=list)
    by_finding: Counter = field(default_factory=Counter)  # (kind, finding id) -> disagreements explained by a known finding
    finding_samples: dict = field(default_factory=dict)
    n_unsound: int = 0
    n_incomplete: int = 0
    n_unrepresentable: int = 0


def context(state, seat, deck):
    """A short, printable picture of the position, for triage."""
    me = state.players[state.active_idx]
    pend = state.pending_resolution
    return {
        "deck": deck, "seat": seat, "turn": state.turn_number, "phase": state.phase.value if state.phase else None,
        "active": state.active_idx, "turn_player": state.turn_player_idx,
        "pending": None if pend is None else {k: (v if isinstance(v, (str, int, bool, type(None))) else repr(v)[:80])
                                              for k, v in pend.items() if k != "on_complete"},
        "hand": sorted(c.name for c in me.hand),
        "battlefield": sorted((p.card_def.name, "T" if p.tapped else "u", "sick" if p.summoning_sick else "")
                              for p in me.battlefield),
        "graveyard": sorted(c.name for c in me.graveyard), "pool": dict(me.mana_pool),
        "stack": len(state.stack), "lands_played": me.lands_played_this_turn,
    }


def uniform_policy(legal, dec, state, rng):
    return int(rng.choice(list(legal)))


class Game:
    """Identity of one game of the run: enough to replay it exactly (the engine is seeded and deterministic)."""

    def __init__(self, a, b, gi, pi, seed, pool):
        self.a, self.b, self.gi, self.pi, self.seed, self.pool = a, b, gi, pi, seed, pool
        self.trace = []   # chosen full-mask index per non-pregame decision, in order
        self.count = 0    # number of non-pregame decisions seen so far (the index of the CURRENT decision inside on_decision)

    @property
    def seats(self):
        return (self.a, self.b)


CURRENT = None  # the Game being played; read by hooks that need to replay-fork


PURITY_EVERY = 0   # >0: fingerprint the position before and after the production mask is built, every Nth decision
IMPURE = []        # positions where building the mask changed the state (the first computation is the one that matters:
                   # later calls are idempotent, so a mutating mask would otherwise be invisible to a before/after check)

_KEEPALIVE = []  # suspended game generators of replay forks: closing one would run its `finally` blocks on the fork's state


def play_one(game_id, on_decision, policy=uniform_policy, horizon=HORIZON, stop_at=None, forced=None):
    """One game, driven through game.game_coroutine. `forced`: replay these chosen indices instead of consulting the
    policy. `stop_at`: return (state, seat, deck, dec) at that decision WITHOUT acting and without unwinding the game
    generator (unwinding would run its `finally` blocks and mutate the position -- e.g. restore active_idx). The mask is
    the production one (`_build_decision`)."""
    decklists, _vocab, ctxs, tables = game_id.pool
    a, b, gi, pi, seed = game_id.a, game_id.b, game_id.gi, game_id.pi, game_id.seed
    rng = random.Random(seed * 1_000_003 + gi * 10_007 + pi)
    pick = random.Random(seed * 7_919 + gi * 31 + pi)
    mull = A.RandomMulligan(random.Random(pi + gi))
    seats = (a, b)
    n = [0]
    stopped = []

    def choose(state):
        seat = state.active_idx
        pend = state.pending_resolution
        if pend is not None and pend["kind"] in A.PREGAME_KINDS:
            return mull.decide(state)
        deck = seats[seat]
        watch = PURITY_EVERY and forced is None and (n[0] + pi) % PURITY_EVERY == 0
        before = state_fingerprint(state) if watch else None
        dec = A._build_decision(state, seat, ctxs[deck], 10 ** 9)  # builds the production mask for this decision
        if watch and state_fingerprint(state) != before:
            IMPURE.append(context(state, seat, deck))
        legal = np.flatnonzero(dec.full_mask)
        if legal.size == 0:
            A._raise_all_false(state, seat)
        if stop_at is not None and n[0] == stop_at:
            stopped.append((state, seat, deck, dec))
            return None
        if forced is None:
            game_id.count = n[0]
        if on_decision is not None:
            on_decision(state, seat, deck, dec, legal, ctxs, tables)
        i = forced[n[0]] if forced is not None else policy(legal, dec, state, pick)
        if forced is None:
            game_id.trace.append(int(i))
        n[0] += 1
        if i < dec.n_fixed:
            if dec.fixed_table[i][0] == "Pass":
                return None
            return lambda: dec.fixed_table[i][2](state)
        chosen = dec.identities[i - dec.n_fixed]
        return lambda: A.execute_pointer_choice(state, chosen)

    state = game.new_multiplayer_game_state([decklists[a], decklists[b]], (gi + pi) % 2, rng)
    gen = game.turn.game_coroutine(state, horizon=horizon, combat_enabled=True)
    try:
        req = next(gen)
        while True:
            action = choose(req)
            if stopped:
                _KEEPALIVE.append(gen)
                del _KEEPALIVE[:-8]
                return stopped[0]
            req = gen.send(action)
    except StopIteration:
        return None


def replay_fork(game_id, k):
    """A fresh live copy of the position at decision k of `game_id`, by re-running the game along its recorded choices.
    Returns (state, seat, deck, dec). Exact, because the engine and every rng in the run are seeded."""
    out = play_one(game_id, None, forced=game_id.trace[:k], stop_at=k)
    if out is None:
        raise RuntimeError(f"replay of {game_id.a} vs {game_id.b} ended before decision {k}")
    return out


def play_games(on_decision, games_per_pair=1, decks=None, seed=0, policy=uniform_policy, horizon=HORIZON, pairs=None):
    """Plays the games; calls on_decision(state, seat, deck, dec, legal_idx, ctxs, tables) at each non-pregame
    decision BEFORE the chosen action runs (`CURRENT` names the game, its trace so far and the decision index).
    Returns (games_played, errors)."""
    global CURRENT
    pool = build_pool()
    decklists = pool[0]
    names = sorted(decklists) if decks is None else list(decks)
    pairs = pairs if pairs is not None else [(a, b) for a in names for b in names]
    errors = []
    played = 0
    for gi in range(games_per_pair):
        for pi, (a, b) in enumerate(pairs):
            CURRENT = Game(a, b, gi, pi, seed, pool)
            try:
                play_one(CURRENT, on_decision, policy=policy, horizon=horizon)
            except Exception as e:  # noqa: BLE001  -- recorded, the suite decides what to do with it
                errors.append((a, b, gi, repr(e)[:400]))
            played += 1
    CURRENT = None
    return played, errors


def diff_against_oracle(state, table_labels, mask_labels):
    """Compare a mask (as a set of fixed-table labels) with the oracle. Returns a list of
    (kind, label, finding_or_None) for every disagreement over the oracle-covered families; kind is one of
    unsound / incomplete / unrepresentable. `table_labels`: every label the table has."""
    extra_tap = _extra_tap_names()
    orc = O.oracle_labels(state)
    table = set(table_labels)
    mask_cov = {lab for lab in mask_labels if O.family(lab, extra_tap) in O.COVERED}
    if mask_cov == (orc & table) and not (orc - table):
        return []
    finding = _attribute(state, mask_cov)
    out = [("unsound", lab, finding.get(lab)) for lab in sorted(mask_cov - orc)]
    out += [("incomplete" if lab in table else "unrepresentable", lab, finding.get(lab)) for lab in sorted(orc - mask_cov)]
    return out


def build_corpus(games_per_pair=1, decks=None, seed=0):
    corpus = ActionCorpus()

    def on_decision(state, seat, deck, dec, legal, ctxs, tables):
        corpus.decisions += 1
        pend = state.pending_resolution
        kind = pend["kind"] if pend else "none"
        corpus.by_kind[kind] += 1
        labels = corpus.table_labels.setdefault(deck, tuple(r[0] for r in dec.fixed_table))
        fixed = [int(i) for i in legal if i < dec.n_fixed]
        mask_labels = {labels[i] for i in fixed}
        for lab in mask_labels:
            corpus.legal_rows[deck][lab] += 1
        extra_tap = _extra_tap_names()
        for lab in mask_labels:
            fam = O.family(lab, extra_tap)
            (corpus.family_legal if fam in O.COVERED else corpus.uncovered_legal)[fam] += 1
        if len(legal) > len(fixed):
            corpus.by_pointer[pointer_kind(state)] += 1
        try:
            diffs = diff_against_oracle(state, labels, mask_labels)
        except Exception as e:  # noqa: BLE001
            corpus.oracle_errors.append({"error": repr(e)[:300], **context(state, seat, deck)})
            return
        for kind, lab, finding in diffs:
            _record(corpus, kind, lab, finding, state, seat, deck)

    corpus.games, corpus.errors = play_games(on_decision, games_per_pair=games_per_pair, decks=decks, seed=seed)
    return corpus


def _attribute(state, mask_cov):
    """For every label on which the mask and the real-rules oracle disagree, the first known finding whose mimicry makes
    the oracle agree on that label (None = unexplained)."""
    real = O.oracle_labels(state)
    out = {}
    for finding, modes in ():  # no engine deviation is currently mimicked (see action_oracle.MODES); add (id, modes) pairs here
        mimic = O.oracle_labels(state, modes)
        for lab in (mask_cov ^ real):
            if lab not in out and ((lab in mask_cov) == (lab in mimic)):
                out[lab] = finding
    return out


def _record(corpus, kind, label, finding, state, seat, deck):
    if finding is not None:
        corpus.by_finding[(kind, finding)] += 1
        samples = corpus.finding_samples.setdefault(finding, [])
        if len(samples) < 6:
            samples.append({"kind": kind, "label": label, **context(state, seat, deck)})
        return
    setattr(corpus, f"n_{kind}", getattr(corpus, f"n_{kind}") + 1)
    bucket = getattr(corpus, kind)
    if len(bucket) < MAX_DIFF_FRAMES:
        bucket.append({"label": label, **context(state, seat, deck)})


_EXTRA_TAP = None


def _extra_tap_names():
    """Names of Saruli-shaped mana sources (their "Tap X" row opens a sub-decision), per the registry."""
    global _EXTRA_TAP
    if _EXTRA_TAP is None:
        _EXTRA_TAP = {cd.name for cd in game.CARD_DEFS.values()
                      if "mana_extra_choose" in game.EFFECT_REGISTRY.get(cd.effect_id, {})}
    return _EXTRA_TAP


def corpus_size():
    return int(os.environ.get("ACTION_CORPUS_GAMES", "2"))
