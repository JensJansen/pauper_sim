"""Shared helpers for the mulligan-net retrain scripts (train_mulligan_vs_twin.py,
train_mulligan_self_mirror.py): net loading, land-count audit, and the
probe-hand trajectory tracker. Not a script itself; each caller does its own
sys.path insert before importing this.
"""
import os

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


def audit_land_counts(game_logs):
    """Reconstructs each game's hand at every keep/mulligan decision from its
    event log (zone_move/draw, /mulligan_take, /mulligan_bottom -- there is
    no separate "hand contents" event, the draw events are the hand) and
    buckets by land count. Only meaningful for an arm whose pregame decider
    is a real MulliganNet -- AlwaysKeep/RandomMulligan never emit the
    decision_weights events this reads."""
    by_lc = {}
    for events in game_logs:
        hand = {0: [], 1: []}
        winner = None
        decisions = []  # (seat, land_count, chosen, p_keep)
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
                chosen = ev["chosen_index"]
                p_keep = next(c["probability"] for c in ev["candidates"] if c["fixed_label"] == "Keep")
                lc = sum(1 for c in hand[seat] if game.CARD_DEFS[c].card_type.name == "LAND")
                decisions.append((seat, lc, chosen, p_keep))
            elif k == "game_over":
                winner = ev.get("winner")
        last_by_seat = {}
        for seat, lc, chosen, p_keep in decisions:
            d = by_lc.setdefault(lc, {"kept": 0, "mulliganed": 0, "keep_probs": [], "wins": 0, "losses": 0})
            if chosen == 0:
                d["kept"] += 1
                d["keep_probs"].append(p_keep)
                last_by_seat[seat] = lc
            else:
                d["mulliganed"] += 1
        for seat, lc in last_by_seat.items():
            if winner is None:
                continue
            by_lc[lc]["wins" if winner == seat else "losses"] += 1
    return by_lc


def build_probe_hands(decklist, vocab):
    """A small, fixed set of synthetic opening hands for one deck -- every
    land count 0 through 7, built from real cards in that deck's own list
    (cycling through the land/nonland pools if fewer than 7 distinct names
    exist) -- through the same build_token_set pipeline decide()/update()
    use, so what gets probed is exactly what the net sees.

    Fixed rather than sampled: evaluated at the same hands every time
    training pauses to check in, so a change in P(mulligan) reflects the net
    moving, not a different hand being asked about."""
    names = sorted({n for n, *_ in decklist})
    lands = [n for n in names if game.CARD_DEFS[n].card_type.name == "LAND"]
    nonlands = [n for n in names if game.CARD_DEFS[n].card_type.name != "LAND"]
    assert lands and nonlands, "a deck with no lands or no spells can't build these probes"

    def take(pool, k):
        return [pool[i % len(pool)] for i in range(k)]

    def hand_tokens(card_names):
        p0, p1 = PlayerState(on_the_play=True), PlayerState(on_the_play=False)
        p0.hand = [game.CARD_DEFS[n] for n in card_names]
        state = GameState(on_the_play=True, players=[p0, p1], event_log=None)
        state.active_idx = 0
        return build_token_set(state, 0, vocab)

    return {f"{n}_land": hand_tokens(take(lands, n) + take(nonlands, 7 - n)) for n in range(8)}


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
        wl = d["wins"] + d["losses"]
        wr = d["wins"] / wl if wl else float("nan")
        print(f"      lands={lc}: kept={d['kept']:3d} mulliganed={d['mulliganed']:3d} "
              f"avg_P(keep|kept)={avg_p:.3f} win_rate_after_keep={wr:.3f} (n={wl})")
