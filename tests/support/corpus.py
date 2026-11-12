"""Corpus of real decisions for the encoder test suite, plus the INDEPENDENT oracle that says what every token should
contain.

A `Frame` is one decision point seen from one seat. Everything a test needs is captured at decision time as plain
numpy / tuples (the live game objects mutate afterwards, so no Frame keeps a reference to one):

  rows / vocab_idx / side      the token set rl.model.features.build_token_set actually produced
  expected_rows                the same set re-derived here, from raw engine state, by code that shares nothing with
                               features.py except the engine's own stat functions (see `oracle_tokens`)
  descriptors                  per token, what its identity handle really is (zone, name, owner) -- for alignment tests
  scalar / expected_scalar     rl.decision.agent._scalar_features vs. its oracle
  full_mask / n_fixed          the legal-action mask the agent used
  hidden_ok / symmetry_ok / slot_ok
                               results of metamorphic checks run on the LIVE state (they need to mutate it and put it
                               back, which can't be done later): hidden information doesn't reach the tokens, the
                               other seat's view is the mirror image, relabelling `slot` numbers changes nothing.

The column layout below is a deliberately separate statement of the spec in features.py's comments: if someone
reorders a column, this file disagrees loudly instead of agreeing with itself."""
import os
import random
from dataclasses import dataclass, field

import numpy as np
import torch

import game
import rl.decision.agent as agent_mod
from game.state import CardInstance, Permanent
from rl.decision.action_bridge import any_pointer_legal, pointer_kind
from rl.decision.agent import AlwaysKeep, SeatAgent, _hand_cost_reduction_deltas, _scalar_features
from rl.league.league_runner import build_deck_net
from rl.model import features as F
from rl.rewards import flat_win_loss_reward
from rl.roster import build_pool
from rl.training.train import _constant_pairing, collect_rollout
from support.paths import REPO, SRC  # noqa: F401


# ---- the spec, restated independently of features.py -------------------------------------------------------------
ZONE_ORDER = ("battlefield", "graveyard", "stack", "exile", "hand", "revealed")
COUNTER_KINDS = ("+1/+1", "-0/-1", "charge", "lifelink")
COPIES_CAP = 20
LINK_SLOTS = 12
COUNT_CAP = 60
COUNTER_CAP = 8
AURA_CAP = 3
PT_CAP = 20
PIP_CAP = 6
DYN_COLUMNS = (
    ["cost_reduction", "copies_same_card", "copies_same_state"]
    + [f"link_src[{i}]" for i in range(12)] + [f"link_dst[{i}]" for i in range(12)]
    + ["targets_player_mine", "targets_player_theirs"]
    + ["untapped", "tapped", "eff_power", "eff_toughness", "blocked_attacker", "committed_blocker",
     "attacking", "summoning_sick", "auras_attached", "is_attached", "targeted_by_mine", "targeted_by_theirs",
     "revealed_focus", "revealed_kept", "revealed_disposed"]
    + [f"counter[{k}]" for k in COUNTER_KINDS] + ["counter[other]"]
    + [f"zone[{z}]" for z in ZONE_ORDER] + ["side_mine"]
)
N_DYN = len(DYN_COLUMNS)
assert N_DYN == F.DYNAMIC_FEATURE_DIM, "features.py grew/lost a dynamic column; update DYN_COLUMNS deliberately"
assert F.STATIC_FEATURE_DIM + N_DYN == F.TOKEN_FEATURE_DIM
COL = {name: F.STATIC_FEATURE_DIM + i for i, name in enumerate(DYN_COLUMNS)}


@dataclass
class Frame:
    game_id: int
    seat: int
    pending_kind: object
    phase: str
    forced: bool
    n_fixed: int
    vocab_idx: np.ndarray            # [n]
    rows: np.ndarray                 # [n, TOKEN_FEATURE_DIM]
    expected_vocab_idx: np.ndarray   # [n]  (same order as `expected_rows`; ORDER differs from `rows`, compare as sets)
    expected_rows: np.ndarray
    descriptors: list                # per actual token: (identity_kind, zone, name, owner) or None
    scalar: np.ndarray
    expected_scalar: np.ndarray
    full_mask: np.ndarray            # [n_fixed + n]
    pointer_kind: object
    engine_option_count: object     # how many pointer targets the engine says exist, or None if not a pointer decision
    hidden_ok: object = None        # None = not checked on this frame
    symmetry_ok: object = None
    slot_ok: object = None
    agent_aligned: object = None    # the agent's own dec.tokens/dec.identities == the tokenizer's, same order (None if forced)
    failures: list = field(default_factory=list)


# ---- independent oracle -------------------------------------------------------------------------------------------

def _controllers_targeting(state, obj):
    return {e["controller"] for e in state.stack for kind, o in e.get("targets", ()) if kind != "player" and o is obj}


def _oracle_row(state, seat, vocab, name, zone, is_mine, perm=None, owner=None, tm=False, tt=False,
                cost_delta=0.0, focus=False, kept=False, disposed=False):
    row = list(F.static_card_features(vocab.card_def(name)))  # fresh, bypassing the cache
    dyn = dict.fromkeys(DYN_COLUMNS, 0.0)
    dyn["cost_reduction"] = cost_delta
    if perm is not None:
        auras = sum(1 for pl in state.players for a in pl.battlefield if a.flags.get("enchanting") is perm)
        remaining = max(game.permanent_toughness(state, perm) - perm.damage_marked, 0)
        dyn["untapped"], dyn["tapped"] = (0.0, 1.0) if perm.tapped else (1.0, 0.0)
        dyn["eff_power"] = max(min(game.permanent_power(state, perm), PT_CAP), -PT_CAP) / PT_CAP  # signed
        dyn["eff_toughness"] = min(remaining, PT_CAP) / PT_CAP
        dyn["blocked_attacker"] = float(any(a is perm for a in state.players[owner].blocked_by))
        dyn["committed_blocker"] = float(any(b is perm for bs in state.players[1 - owner].blocked_by.values() for b in bs))
        dyn["attacking"] = float(any(a is perm for a in state.players[owner].attackers))
        dyn["summoning_sick"] = float(perm.summoning_sick)
        dyn["auras_attached"] = min(auras, AURA_CAP) / AURA_CAP
        dyn["is_attached"] = float(perm.flags.get("enchanting") is not None)
        other = False
        for k, n in perm.counters.items():
            if k in COUNTER_KINDS:
                dyn[f"counter[{k}]"] = min(n, COUNTER_CAP) / COUNTER_CAP
            elif n:
                other = True
        dyn["counter[other]"] = float(other)
    dyn["targeted_by_mine"], dyn["targeted_by_theirs"] = float(tm), float(tt)
    dyn["revealed_focus"], dyn["revealed_kept"], dyn["revealed_disposed"] = float(focus), float(kept), float(disposed)
    dyn[f"zone[{zone}]"] = 1.0
    dyn["side_mine"] = float(is_mine)
    return np.asarray(row + [dyn[c] for c in DYN_COLUMNS], dtype=np.float32)


def oracle_tokens(state, seat, vocab):
    """[(vocab_idx, row)] -- what the token set for `seat` SHOULD be, derived straight from the engine's zones."""
    opp = 1 - seat
    deltas = _hand_cost_reduction_deltas(state, seat)
    pend = state.pending_resolution
    out = []
    names = []  # the real card name of each oracle token, in step with `out` (a vocab index can alias two faces)
    objs = []   # the engine object each token stands for (Permanent / CardInstance / stack entry), or None

    def add(name, *a, obj=None, **k):
        names.append(name)
        objs.append(obj)
        out.append((vocab.index(name), _oracle_row(state, seat, vocab, name, *a, **k)))

    hand_pending_owner = None
    if pend is not None and pend["kind"] == "choose_graveyard_card":
        hand_pending_owner = next((i for i, pl in enumerate(state.players) if pend["graveyard"] is pl.hand), None)

    for s in (seat, opp):
        mine = s == seat
        pl = state.players[s]
        for p in pl.battlefield:
            tm, tt = (lambda c: (seat in c, opp in c))(_controllers_targeting(state, p))
            add(p.card_def.name, "battlefield", mine, perm=p, owner=s, tm=tm, tt=tt, obj=p)
        for inst in pl.graveyard:
            tm, tt = (lambda c: (seat in c, opp in c))(_controllers_targeting(state, inst))
            add(inst.name, "graveyard", mine, tm=tm, tt=tt, obj=inst)
        for card_def, _t in pl.exile:
            add(card_def.name, "exile", mine)
        if mine and hand_pending_owner != seat:
            for cd in pl.hand:
                add(cd.name, "hand", True, cost_delta=deltas.get(cd.name, 0.0))
    for e in state.stack:
        tm, tt = (lambda c: (seat in c, opp in c))(_controllers_targeting(state, e))
        add(e["card_def"].name, "stack", e["controller"] == seat, tm=tm, tt=tt, obj=e)
    if pend is not None and state.active_idx == seat:
        groups = ()
        if pend["kind"] in ("scry", "surveil"):
            rem = pend["remaining"]
            groups = ((rem[:1], {"focus": True}), (rem[1:], {}), (pend["kept"], {"kept": True}),
                      (pend["disposed"], {"disposed": True}))
        elif pend["kind"] == "ponder":
            groups = ((pend["remaining"], {}), (pend["ordered"] or (), {"kept": True}))
        elif pend["kind"] == "tuck_position":
            groups = (((pend["tuck_card"],), {"focus": True}),)
        for cards, flags in groups:
            for cd in cards:
                add(cd.name, "revealed", True, **flags)
    if hand_pending_owner is not None:
        for cd in pend["graveyard"]:
            add(cd.name, "hand", hand_pending_owner == seat,
                cost_delta=0.0)  # features passes no cost_reduction for a revealed hand
    return _with_links(_with_multiplicity(out, names), objs, state, seat)


def _with_links(tokens, objs, state, seat):
    """Link slots and player-target flags, from the engine state: aura -> host, stack entry -> target, blocker ->
    attacker. Edges are numbered by sorting on (kind, source content, destination content) where content is the oracle
    row with everything viewpoint-dependent (side, targeted-by, player-target flags, links) blanked, then on the rows
    as they are (links blanked); slots saturate at LINK_SLOTS - 1. Written from the spec in features.py's LINKS comment, sharing no code with it."""
    at = {id(o): i for i, o in enumerate(objs) if o is not None}
    edges, player = [], {}
    for pl in state.players:
        for perm in pl.battlefield:
            host = perm.flags.get("enchanting")
            if host is not None and id(perm) in at and id(host) in at:
                edges.append((0, at[id(perm)], at[id(host)]))
    for e in state.stack:
        for kind, tgt in e.get("targets", ()):
            if kind == "player":
                player.setdefault(at[id(e)], set()).add("mine" if tgt == seat else "theirs")
            elif id(tgt) in at:
                edges.append((1, at[id(e)], at[id(tgt)]))
    for owner in state.players:
        for attacker, blockers in owner.blocked_by.items():
            for blocker in blockers:
                if id(blocker) in at and id(attacker) in at:
                    edges.append((2, at[id(blocker)], at[id(attacker)]))
    rows = [np.array(r, copy=True) for _i, r in tokens]
    blank = [COL["side_mine"], COL["targeted_by_mine"], COL["targeted_by_theirs"], COL["targets_player_mine"],
             COL["targets_player_theirs"]] + [COL[f"link_src[{k}]"] for k in range(LINK_SLOTS)]         + [COL[f"link_dst[{k}]"] for k in range(LINK_SLOTS)]

    links = [COL[f"link_src[{k}]"] for k in range(LINK_SLOTS)] + [COL[f"link_dst[{k}]"] for k in range(LINK_SLOTS)]

    def content(i, neutral):
        r = rows[i].copy()
        r[blank if neutral else links] = 0.0  # perspective blanked for the primary key, the rows as they are for the secondary
        return r.tobytes()

    for rank, (_key, s, d) in enumerate(sorted(((k, content(s, True), content(d, True), content(s, False), content(d, False)), s, d)
                                               for k, s, d in edges)):
        slot = min(rank, LINK_SLOTS - 1)
        rows[s][COL[f"link_src[{slot}]"]] = 1.0
        rows[d][COL[f"link_dst[{slot}]"]] = 1.0
    for i, who in player.items():
        rows[i][COL["targets_player_mine"]] = float("mine" in who)
        rows[i][COL["targets_player_theirs"]] = float("theirs" in who)
    return [(idx, r) for (idx, _old), r in zip(tokens, rows)]


def _with_multiplicity(tokens, card_names):
    """Fills copies_same_card / copies_same_state, counted here by real card name (not by vocab index / static bytes as
    the implementation does) and by whole-row equality, as log(n)/log(COPIES_CAP)."""
    import math
    c_card, c_state = COL["copies_same_card"], COL["copies_same_state"]
    zone_side = [COL[f"zone[{z}]"] for z in ZONE_ORDER] + [COL["side_mine"]]
    names = [(nm, tuple(row[zone_side])) for nm, (_idx, row) in zip(card_names, tokens)]
    full = [(idx, tuple(row)) for idx, row in tokens]  # multiplicity columns are still 0 at this point
    out = []
    for (idx, base_row), nz, fz in zip(tokens, names, full):
        row = np.array(base_row, copy=True)
        row[c_card] = math.log(sum(1 for t in names if t == nz)) / math.log(COPIES_CAP)
        row[c_state] = math.log(sum(1 for t in full if t == fz)) / math.log(COPIES_CAP)
        out.append((idx, row))
    return out


def oracle_scalar(state, seat, horizon):
    me, other = state.players[seat], state.players[1 - seat]
    pc = {0: False, 1: False}
    for e in state.stack:
        for kind, obj in e.get("targets", ()):
            if kind == "player":
                pc[obj] = True
    out = [min(state.turn_number / horizon, 1.0), float(me.lands_played_this_turn > 0), min(me.mulligans_taken, 7) / 7,
           float(seat == state.turn_player_idx)]
    out += [min(me.mana_pool.get(c, 0), 8) / 8 for c in game.POOL_COLORS]
    out += [min(other.mana_pool.get(c, 0), 8) / 8 for c in game.POOL_COLORS]
    out += [float(ph == state.phase) for ph in game.turn.Phase]
    out += [min(max(me.life_total, 0), 200) / 200, min(max(other.life_total, 0), 200) / 200]
    out += [min(len(me.library), 60) / 60, min(len(other.library), 60) / 60, min(len(other.hand), 12) / 12]
    out += [float(pc[seat]), float(pc[1 - seat]), float(me.on_the_play), min(other.mulligans_taken, 7) / 7,
            min(other.cleanup_discard_turns, 7) / 7]
    import math

    def cnt(n):
        return math.log(1 + n) / math.log(1 + COUNT_CAP)

    for who in (seat, 1 - seat):  # exact counts: by type, untapped by type, graveyard, exile, stack entries it controls
        pl = state.players[who]
        out += [cnt(len([p for p in pl.battlefield if p.card_type == t])) for t in game.CardType]
        out += [cnt(len([p for p in pl.battlefield if p.card_type == t and not p.tapped])) for t in game.CardType]
        out += [cnt(len(pl.graveyard)), cnt(len(pl.exile)), cnt(len([e for e in state.stack if e["controller"] == who]))]
    out.append(cnt(len(me.hand)))
    return np.asarray(out, dtype=np.float32)


def describe(state, ident):
    """What a token's identity handle really is: (handle kind, zone, name, owner seat) or None for no handle."""
    if ident is None:
        return None
    if isinstance(ident, dict):
        return ("stack_entry", "stack", ident["card_def"].name, ident["controller"])
    if isinstance(ident, Permanent):
        for i, pl in enumerate(state.players):
            if any(ident is p for p in pl.battlefield):
                return ("permanent", "battlefield", ident.card_def.name, i)
        return ("permanent", "battlefield", ident.card_def.name, None)
    if isinstance(ident, CardInstance):
        for i, pl in enumerate(state.players):
            if any(ident is g for g in pl.graveyard):
                return ("card_instance", "graveyard", ident.name, i)
        return ("card_instance", "graveyard", ident.name, None)
    for i, pl in enumerate(state.players):  # a CardDef: a revealed hand card
        if any(ident is c for c in pl.hand):
            return ("card_def", "hand", ident.name, i)
    return ("card_def", "hand", getattr(ident, "name", None), None)


def engine_option_count(state, identities):
    """How many pointer targets the ENGINE reports for the current decision (None if not a pointer decision)."""
    mana_sub = state.active_mana_subdecision
    pend = state.pending_resolution
    if mana_sub is not None:
        return None  # legality there is a per-permanent predicate with no option list; checked by mask only
    if pend is None:
        if state.phase is game.turn.Phase.DECLARE_ATTACKERS and state.active_idx == state.turn_player_idx:
            return sum(1 for p in state.battlefield if game.creature_attack_eligible(state, p))
        return None
    k = pend["kind"]
    if k == "declare_blockers":
        return sum(1 for p in state.battlefield if game.creature_block_eligible(state, p))
    if k == "choose_permanent":
        return len(set(game.choose_permanent_options(state)))
    if k == "choose_opponent_permanent":
        return len(set(game.choose_opponent_permanent_options(state)))
    if k == "choose_any_target":
        return len(set(game.choose_any_target_creature_options(state)))
    if k == "choose_cast_copy":
        return len(game.choose_cast_copy_options(state))
    if k == "choose_graveyard_card":
        return len(game.choose_graveyard_card_options(state))
    if k == "choose_stack_target":
        return len(game.choose_stack_target_options(state))
    return None


LAST_SYMMETRY_FAILURE = []  # filled by metamorphic_checks when the mirror comparison fails (debugging aid)


# ---- metamorphic checks on the live state ---------------------------------------------------------------------------

def _signature(tokens, scalar):
    rows = sorted((int(i), tuple(np.round(r, 6))) for i, r, _ident in tokens)
    return rows, tuple(np.round(scalar, 6))


def _flip(row):
    """The same token seen from the other seat: side flag flipped, targeted-by-mine/theirs swapped."""
    r = np.array(row, copy=True)
    r[COL["side_mine"]] = 1.0 - r[COL["side_mine"]]
    r[COL["targeted_by_mine"]], r[COL["targeted_by_theirs"]] = row[COL["targeted_by_theirs"]], row[COL["targeted_by_mine"]]
    r[COL["targets_player_mine"]], r[COL["targets_player_theirs"]] = row[COL["targets_player_theirs"]], row[COL["targets_player_mine"]]
    return r


def _canon_links(row):
    """The row with its link SLOT numbers replaced by how many source / destination slots it carries. Two views of the
    same position number edges identically except where edges differ only by perspective (e.g. identical auras on
    identical creatures on both sides); there the numbering is a relabelling, so a mirror comparison must not depend on
    it. The edges themselves are checked exactly by the oracle (test_l1) and test_l12_links."""
    r = np.array(row, copy=True)
    n_src = r[COL["link_src[0]"]:COL["link_src[0]"] + LINK_SLOTS].sum()
    n_dst = r[COL["link_dst[0]"]:COL["link_dst[0]"] + LINK_SLOTS].sum()
    r[COL["link_src[0]"]:COL["link_dst[0]"] + LINK_SLOTS] = 0.0
    r[COL["link_src[0]"]], r[COL["link_dst[0]"]] = n_src, n_dst
    return r


def metamorphic_checks(state, seat, vocab, horizon, base_tokens, base_scalar):
    """(hidden_ok, symmetry_ok, slot_ok), each a bool; the state is restored exactly before returning."""
    opp = state.players[1 - seat]
    # 1. hidden information: reordering the opponent's hand and both libraries must change nothing observable
    saved = [(pl.hand, list(pl.hand), pl.library, list(pl.library)) for pl in state.players]
    rng = random.Random(len(base_tokens))
    try:
        for pl in state.players:
            rng.shuffle(pl.library)
        rng.shuffle(opp.hand)
        after = build_token_set_(state, seat, vocab)
        after_scalar = np.asarray(_scalar_features(state, seat, horizon), dtype=np.float32)
        hidden_ok = _signature(after, after_scalar) == _signature(base_tokens, base_scalar)
    finally:
        for lst, lst_copy, lib, lib_copy in saved:
            lst[:] = lst_copy
            lib[:] = lib_copy
    # 2. slot numbers are an addressing detail, never an input
    perms = [p for pl in state.players for p in pl.battlefield]
    old_slots = [p.slot for p in perms]
    try:
        for p in perms:
            p.slot += 100
        slot_ok = _signature(build_token_set_(state, seat, vocab), base_scalar) == _signature(base_tokens, base_scalar)
    finally:
        for p, s in zip(perms, old_slots):
            p.slot = s
    # 3. seat symmetry: the other seat's view of the SHARED tokens is this view with the side flag / targeting flipped
    other_view = build_token_set_(state, 1 - seat, vocab)
    skip_zones = (COL["zone[hand]"], COL["zone[revealed]"])
    def keep(r):
        return all(r[c] == 0.0 for c in skip_zones)
    mine = sorted((int(i), tuple(np.round(_canon_links(_flip(r)), 6))) for i, r, _x in base_tokens if keep(r))
    theirs = sorted((int(i), tuple(np.round(_canon_links(r), 6))) for i, r, _x in other_view if keep(r))
    symmetry_ok = mine == theirs
    if not symmetry_ok:  # keep a readable account of what differs, for the failure message
        only_mine = [t for t in mine if t not in theirs]
        only_theirs = [t for t in theirs if t not in mine]
        detail = []
        for (i, row_a), (_j, row_b) in zip(only_mine, only_theirs):
            cols = [(DYN_COLUMNS[c - F.STATIC_FEATURE_DIM] if c >= F.STATIC_FEATURE_DIM else f"static[{c}]", x, y)
                    for c, (x, y) in enumerate(zip(row_a, row_b)) if x != y]
            zones = [z for z in ZONE_ORDER if row_a[COL[f"zone[{z}]"]] == 1.0]
            name = next((n for n, ix in vocab.name_to_index.items() if ix == i), i)
            detail.append((name, zones, cols[:6]))
        LAST_SYMMETRY_FAILURE.append((len(mine), len(theirs), detail[:3]))
    return hidden_ok, symmetry_ok, slot_ok


def build_token_set_(state, seat, vocab):
    return F.build_token_set(state, seat, vocab, hand_cost_reduction=_hand_cost_reduction_deltas(state, seat))


# ---- capture ---------------------------------------------------------------------------------------------------------

def capture(state, seat, deck_ctx, horizon, dec, game_id, metamorphic):
    vocab, _table = deck_ctx
    tokens = build_token_set_(state, seat, vocab)
    scalar = np.asarray(_scalar_features(state, seat, horizon), dtype=np.float32)
    exp = oracle_tokens(state, seat, vocab)
    identities = [ident for _i, _r, ident in tokens]
    pk = pointer_kind(state)
    f = Frame(
        game_id=game_id, seat=seat, pending_kind=state.pending_resolution["kind"] if state.pending_resolution else None,
        phase=state.phase.name, forced=dec.sole_action is not None, n_fixed=dec.n_fixed,
        vocab_idx=np.asarray([i for i, _r, _x in tokens], dtype=np.int64),
        rows=np.asarray([r for _i, r, _x in tokens], dtype=np.float32).reshape(len(tokens), F.TOKEN_FEATURE_DIM),
        expected_vocab_idx=np.asarray([i for i, _r in exp], dtype=np.int64),
        expected_rows=np.asarray([r for _i, r in exp], dtype=np.float32).reshape(len(exp), F.TOKEN_FEATURE_DIM),
        descriptors=[describe(state, x) for x in identities],
        scalar=scalar, expected_scalar=oracle_scalar(state, seat, horizon),
        full_mask=np.asarray(dec.full_mask, dtype=bool), pointer_kind=pk,
        engine_option_count=engine_option_count(state, identities) if any_pointer_legal(state) else None,
    )
    if dec.tokens is not None:
        f.agent_aligned = (len(dec.tokens) == len(tokens) and len(dec.identities) == len(tokens)
                           and all(a[0] == b[0] and np.array_equal(a[1], b[1]) for a, b in zip(dec.tokens, tokens))
                           and all(x is y for x, y in zip(dec.identities, identities)))
    if metamorphic:
        f.hidden_ok, f.symmetry_ok, f.slot_ok = metamorphic_checks(state, seat, vocab, horizon, tokens, scalar)
    return f


# Probability of keeping a decision, by pending kind: the mundane ones (paying a cost, an ordinary priority window) are
# thinned so rare decisions (targets, blocks, scry, ...) make up a useful share of the corpus.
KEEP_PROB = {"pay_cost": 0.2, None: 0.5}
FORCED_KEEP_PROB = 0.2
FRAMES_PER_GAME = 160


def build_corpus(games_per_pair=1, horizon=60, seed=0, metamorphic_every=3, deck_names=None):
    """Plays real games between every pair of decks (mirrors included) with randomly initialised networks and records
    a Frame for a thinned, stratified sample of the decisions (see KEEP_PROB), at most FRAMES_PER_GAME per game.
    Deterministic for a given seed."""
    prev_cwd = os.getcwd()
    os.chdir(SRC)  # rl.roster's manifest/vocab paths are relative to src/
    try:
        decklists, vocab, deck_ctxs, tables = build_pool(vocab_path=None)
        env_decks = os.environ.get("ENCODER_CORPUS_DECKS")  # comma list: a smaller corpus for the mutation sweep
        names = list(deck_names or (env_decks.split(",") if env_decks else sorted(decklists)))
        torch.manual_seed(seed)
        enc = {"d_model": 32, "n_heads": 2, "n_layers": 2, "dim_feedforward": 64}
        agents = {}
        for n in names:
            net = build_deck_net(vocab.size, len(tables[n]), trunk_hidden=(32, 32), encoder=enc)
            net.eval()
            agents[n] = SeatAgent(net, AlwaysKeep(), deck_ctxs[n])

        frames, counter, rng = [], {"i": 0, "game": 0, "in_game": 0}, random.Random(seed)
        real_build = agent_mod._build_decision

        def recording_build(state, seat, deck_ctx, hz):
            dec = real_build(state, seat, deck_ctx, hz)
            counter["i"] += 1
            kind = state.pending_resolution["kind"] if state.pending_resolution else None
            keep = FORCED_KEEP_PROB if dec.sole_action is not None else KEEP_PROB.get(kind, 1.0)
            if counter["in_game"] < FRAMES_PER_GAME and rng.random() < keep:
                counter["in_game"] += 1
                meta = counter["i"] % metamorphic_every == 0
                frames.append(capture(state, seat, deck_ctx, hz, dec, counter["game"], meta))
            return dec

        agent_mod._build_decision = recording_build
        try:
            pairs = [(a, b) for i, a in enumerate(names) for b in names[i:]]
            for a, b in pairs:
                for _g in range(games_per_pair):
                    counter["game"] += 1
                    counter["in_game"] = 0
                    pairing = _constant_pairing([agents[a], agents[b]], [decklists[a], decklists[b]],
                                                [flat_win_loss_reward()] * 2, ["m", "m"])
                    collect_rollout(pairing, 1, horizon=horizon, rng=random.Random(seed * 1000 + counter["game"]), device="cpu")
        finally:
            agent_mod._build_decision = real_build
        return frames, vocab, agents
    finally:
        os.chdir(prev_cwd)
