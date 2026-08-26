"""Shared helpers for train_mulligan.py (the mulligan-net retrain script,
opponent-mode twin or self-mirror): net loading, land-count audit, and the
probe-hand trajectory tracker. Not a script itself; the caller does its own
sys.path insert before importing this.
"""
import math
import os
import random

import torch

import game
from game.state import GameState, PlayerState
from rl import checkpoint as ckpt_io
from rl.model.arch import pad_token_batch
from rl.model.features import build_token_set
from rl.league.league_runner import build_deck_net


def load_frozen_nets(league_dir, deck_names, vocab, fixed_tables):
    """Every deck's live net from one league directory, eval mode,
    requires_grad=False -- structurally incapable of training."""
    nets = {}
    for name in deck_names:
        live_path = f"{league_dir}/{name}/live.pt"
        if not os.path.exists(live_path):
            raise SystemExit(f"no live.pt for {name} in {league_dir}")
        net = build_deck_net(vocab.size, len(fixed_tables[name]),
                             ckpt_io.trunk_hidden_from_deck_checkpoint(live_path))
        ckpt_io.load_deck_checkpoint(live_path, net)
        net.eval()
        for p in net.parameters():
            p.requires_grad = False
        nets[name] = net
    return nets


_UNBUCKETED = object()  # sentinel key used internally when deck_by_game_seat is None


def _binary_entropy_bits(p):
    """Shannon entropy of a Bernoulli(p) keep/mulligan choice, in bits (0 at
    p=0 or 1, max 1.0 at p=0.5 -- a flat, collapse-agnostic scale for
    tracking a policy sharpening toward "always keep" over a training
    run). 0*log2(0) := 0 by convention."""
    def term(x):
        return -x * math.log2(x) if x > 0 else 0.0
    return term(p) + term(1 - p)


def audit_land_counts(game_logs, deck_by_game_seat=None):
    """Reconstructs each game's hand at every keep/mulligan decision from its
    event log (zone_move/draw, /mulligan_take, /mulligan_bottom -- there is
    no separate "hand contents" event, the draw events are the hand) and
    buckets by land count. Only meaningful for an arm whose pregame decider
    is a real MulliganNet -- AlwaysKeep/RandomMulligan never emit the
    decision_weights events this reads.

    deck_by_game_seat (optional): a list parallel to game_logs, each entry
    {seat: deck_name} attributing that game's seats to a deck. A seat absent
    from the dict is excluded entirely -- the caller's way to filter out a
    seat it doesn't want counted (e.g. an opponent-league seat in a
    cross-league game: validation.mulligan_audit passes an entry only for
    the primary-controlled seat). When given, returns
    {deck_name: {land_count: {...}}} instead of the flat {land_count: {...}}
    a bare game_logs call still returns, unchanged from before this parameter
    existed."""
    by_deck = {}
    for i, events in enumerate(game_logs):
        seat_deck = deck_by_game_seat[i] if deck_by_game_seat is not None else None
        hand = {0: [], 1: []}
        winner = None
        decisions = []  # (deck_name, seat, land_count, chosen, p_keep)
        for ev in events:
            k, seat = ev.get("kind"), ev.get("active_idx")
            if k == "zone_move" and ev.get("reason") == "draw" and ev.get("to_zone") == "hand":
                hand[seat].extend(ev["cards"])
            elif k == "zone_move" and ev.get("reason") == "mulligan_take":
                hand[seat] = []
            elif k == "zone_move" and ev.get("reason") == "mulligan_bottom":
                name = ev.get("card")
                if name in hand[seat]:
                    hand[seat].remove(name)
            elif k == "decision_weights" and ev.get("network") == "mulligan_keep":
                if seat_deck is not None and seat not in seat_deck:
                    continue  # this seat is excluded by the caller
                deck_name = seat_deck[seat] if seat_deck is not None else _UNBUCKETED
                chosen = ev["chosen_index"]
                p_keep = next(c["probability"] for c in ev["candidates"] if c["fixed_label"] == "Keep")
                lc = sum(1 for c in hand[seat] if game.CARD_DEFS[c].card_type.name == "LAND")
                decisions.append((deck_name, seat, lc, chosen, p_keep))
            elif k == "game_over":
                winner = ev.get("winner")
        last_by_seat = {}  # seat -> (deck_name, land_count)
        for deck_name, seat, lc, chosen, p_keep in decisions:
            by_lc = by_deck.setdefault(deck_name, {})
            d = by_lc.setdefault(lc, {"kept": 0, "mulliganed": 0, "keep_probs": [], "entropy_bits": [],
                                       "wins": 0, "losses": 0})
            d["entropy_bits"].append(_binary_entropy_bits(p_keep))  # every decision, kept or not
            if chosen == 0:
                d["kept"] += 1
                d["keep_probs"].append(p_keep)
                last_by_seat[seat] = (deck_name, lc)
            else:
                d["mulliganed"] += 1
        for seat, (deck_name, lc) in last_by_seat.items():
            if winner is None:
                continue
            by_deck[deck_name][lc]["wins" if winner == seat else "losses"] += 1
    if deck_by_game_seat is None:
        return by_deck.get(_UNBUCKETED, {})
    return by_deck


def _hand_tokens(card_names, vocab):
    """One 7-card hand -> the build_token_set tokens decide()/update() would
    see for it (pregame: every zone but this hand is empty)."""
    p0, p1 = PlayerState(on_the_play=True), PlayerState(on_the_play=False)
    p0.hand = [game.CARD_DEFS[n] for n in card_names]
    state = GameState(on_the_play=True, players=[p0, p1], event_log=None)
    state.active_idx = 0
    return build_token_set(state, 0, vocab)


def build_probe_hands(decklist, vocab):
    """A small, fixed set of synthetic opening hands for one deck -- every
    land count 0 through 7, built from real cards in that deck's own list
    (cycling through the land/nonland pools if fewer than 7 distinct names
    exist) -- through the same build_token_set pipeline decide()/update()
    use, so what gets probed is exactly what the net sees.

    Fixed rather than sampled: evaluated at the same hands every time
    training pauses to check in, so a change in P(mulligan) reflects the net
    moving, not a different hand being asked about. See
    build_probe_hands_sampled for a multi-hand-per-count variant that
    reflects true deck composition -- this one just wants a stable target."""
    names = sorted({n for n, *_ in decklist})
    lands = [n for n in names if game.CARD_DEFS[n].card_type.name == "LAND"]
    nonlands = [n for n in names if game.CARD_DEFS[n].card_type.name != "LAND"]
    assert lands and nonlands, "a deck with no lands or no spells can't build these probes"

    def take(pool, k):
        return [pool[i % len(pool)] for i in range(k)]

    return {f"{n}_land": _hand_tokens(take(lands, n) + take(nonlands, 7 - n), vocab) for n in range(8)}


def build_probe_hands_sampled(decklist, vocab, land_counts=range(8), n_variants=6, seed=0):
    """n_variants distinct synthetic hands per land count in land_counts,
    each sampled without replacement from the real card multiplicities in
    decklist (an actual possible 7-card draw), unlike build_probe_hands'
    single deterministic hand per count. Seeded rather than random, so the
    same call always returns the same hands -- comparable across cadence
    points during a run -- while averaging out any one hand's idiosyncrasy
    and reaching land counts (6, 7; often 0 too) natural self-play rarely or
    never draws. Returns {land_count: [tokens, ...]}.

    A land_count this deck could never actually draw (lc lands sampled
    without replacement needs lc <= len(lands), and the other 7-lc slots
    need 7-lc <= len(spells)) is skipped rather than raising or padding with
    a fake repeat -- e.g. a low-land deck like spy_combo (4 real Lands, the
    rest mana creatures) can't ever be dealt a 5+-land hand, so probing for
    one isn't a rare case worth forcing, it's asking about a hand that
    can't exist. Returns whatever subset of land_counts is achievable;
    probe_land_count_stats' own {lc: ...} dict comprehension already only
    walks the keys actually present."""
    rng = random.Random(seed)
    lands = [n for n, c, *_ in decklist for _ in range(c) if game.CARD_DEFS[n].card_type.name == "LAND"]
    spells = [n for n, c, *_ in decklist for _ in range(c) if game.CARD_DEFS[n].card_type.name != "LAND"]
    assert lands and spells, "a deck with no lands or no spells can't build these probes"
    out = {}
    for lc in land_counts:
        if lc > len(lands) or 7 - lc > len(spells):
            continue
        hands = []
        for _ in range(n_variants):
            hand = rng.sample(lands, lc) + rng.sample(spells, 7 - lc)
            rng.shuffle(hand)
            hands.append(_hand_tokens(hand, vocab))
        out[lc] = hands
    return out


def probe_land_count_stats(net, probes_by_lc, scalars=(0.0, 1.0)):
    """{land_count: {p_mulligan_mean, p_mulligan_spread, entropy_bits_mean,
    n}} for hands built by build_probe_hands_sampled -- the sculpted-hand
    counterpart to audit_land_counts' natural-game numbers: exact land
    counts on demand instead of whatever self-play happened to draw. `net`
    is the MulliganNet (matches probe_p_mulligan). Read-only, safe to call
    mid-training."""
    sc = torch.tensor([list(scalars)], dtype=torch.float32)
    out = {}
    with torch.inference_mode():
        for lc, hands in probes_by_lc.items():
            ps = []
            for tokens in hands:
                vocab_idx, features, key_padding_mask, _identities = pad_token_batch([tokens])
                side_flag = features[:, :, -1]
                mine_summary, _token_reps = net.encode(vocab_idx, features, key_padding_mask, side_flag)
                logits, _value = net.decision(mine_summary, sc)
                ps.append(torch.softmax(logits, dim=-1)[0, 1].item())
            out[lc] = {
                "p_mulligan_mean": sum(ps) / len(ps),
                "p_mulligan_spread": max(ps) - min(ps),
                "entropy_bits_mean": sum(_binary_entropy_bits(p) for p in ps) / len(ps),
                "n": len(ps),
            }
    return out


def probe_p_mulligan(net, probes, scalars=(0.0, 1.0)):
    """{probe name: P(mulligan)} for the net's current weights -- read-only,
    safe to call mid-training without disturbing the optimizer's graph.
    scalars=(mulligans_taken/HAND, on_the_play): fixed at (0, on-the-play)
    for every probe, matching a real opening-hand decision."""
    out = {}
    sc = torch.tensor([list(scalars)], dtype=torch.float32)
    with torch.inference_mode():
        for name, tokens in probes.items():
            vocab_idx, features, key_padding_mask, _identities = pad_token_batch([tokens])
            side_flag = features[:, :, -1]
            mine_summary, _token_reps = net.encode(vocab_idx, features, key_padding_mask, side_flag)
            logits, _value = net.decision(mine_summary, sc)
            out[name] = torch.softmax(logits, dim=-1)[0, 1].item()
    return out


def print_land_audit(by_lc):
    if not by_lc:
        print("    land-count audit: no mulligan_keep decision_weights events found")
        return
    print("    land-count audit (trained MulliganNet, greedy eval):")
    for lc in sorted(by_lc):
        d = by_lc[lc]
        avg_p = sum(d["keep_probs"]) / len(d["keep_probs"]) if d["keep_probs"] else float("nan")
        avg_ent = sum(d["entropy_bits"]) / len(d["entropy_bits"]) if d["entropy_bits"] else float("nan")
        wl = d["wins"] + d["losses"]
        wr = d["wins"] / wl if wl else float("nan")
        print(f"      lands={lc}: kept={d['kept']:3d} mulliganed={d['mulliganed']:3d} "
              f"avg_P(keep|kept)={avg_p:.3f} avg_entropy_bits={avg_ent:.3f} "
              f"win_rate_after_keep={wr:.3f} (n={wl})")
