"""Policy-movement probe: does an agent's decision-making change as it trains?

A probe set is a fixed collection of REFERENCE GAMES of one deck (fixed seeds, played by frozen agents of several
ages against frozen opponents of several ages), stored as the network inputs the training buffer already records per
decision (token set, scalar features, legal-action mask) plus the action that was taken. Any checkpoint of that deck
is then run over the stored games with the stored actions fed back in (teacher forcing: the deck network has a GRU,
so a decision depends on the history before it -- every checkpoint sees the SAME history), and its action
distribution is read off at the probe decisions: those from the middle of the game (a developed board, a real
choice), not the forced or opening moves. Only the first decision at each state matters, so the opponent's replies
to a checkpoint's choices never enter: the states are frozen.

Compare two checkpoints by how far apart their distributions are over the probe decisions (mean Jensen-Shannon
divergence, top-1 agreement). Movement is not progress -- read it next to the strength curves (retro_sweep.py): a
flat strength curve with a still-moving policy is churning, with a frozen one it is stuck. The scale needs a floor
and a ceiling: two snapshots one snapshot-interval apart (how much it moves in a normal step) and a randomly
initialised net (how far apart unrelated policies are)."""
import gzip
import hashlib
import pickle

import numpy as np
import torch

from rl.model.arch import pad_token_batch

MID_TURN = 5       # state.turn_number at the decision (the engine counts both players' turns)
MID_PERMANENTS = 6  # permanents on both battlefields together
MIN_LEGAL = 3      # fewer legal actions is a forced move or a coin flip with nothing to read


def _selected(meta, mask):
    turn, permanents = meta
    return turn >= MID_TURN and permanents >= MID_PERMANENTS and int(np.count_nonzero(mask)) >= MIN_LEGAL


def record_probe_set(deck, agents_by_deck, decklists, deck_ctxs, n_games, seed, horizon):
    """Plays `n_games` reference games of `deck` and returns the probe set:
    {"deck", "episodes": [{"tokens", "scalar", "mask", "action", "probe": [bool per decision]}]}.

    agents_by_deck: {(deck_name, age_label): factory() -> a fresh SeatAgent} for every deck/age that may appear.
    Each game picks the probe deck's age, the opponent deck and its age, and the seat, from the seeded rng, so the
    set is a pure function of (agents, seed)."""
    import random
    from rl.decision.agent import SeatAgent
    from rl.rewards import flat_win_loss_reward
    from rl.training.train import collect_rollout

    rng = random.Random(seed)
    ages = sorted({age for _, age in agents_by_deck})
    decks = sorted({d for d, _ in agents_by_deck})
    meta = []  # (turn, permanents) per recorded decision, in buffer order

    class Recorder(SeatAgent):
        def decide(self, state, seat, horizon, device, greedy=False):
            result = super().decide(state, seat, horizon, device, greedy=greedy)
            if result.ppo_entry is not None:
                meta.append((state.turn_number, sum(len(p.battlefield) for p in state.players)))
            return result

    reward_fn = flat_win_loss_reward()

    def pairing(game_rng):
        own = agents_by_deck[(deck, game_rng.choice(ages))]()
        opp_deck = game_rng.choice(decks)
        opp = agents_by_deck[(opp_deck, game_rng.choice(ages))]()
        probe_agent = Recorder(own.main, own.mulligan, own.deck_ctx)
        seat = game_rng.randint(0, 1)
        agents, lists, rewards, record_as = [None, None], [None, None], [None, None], [None, None]
        agents[seat], lists[seat], rewards[seat], record_as[seat] = probe_agent, decklists[deck], reward_fn, deck
        agents[1 - seat], lists[1 - seat] = opp, decklists[opp_deck]
        return agents, lists, rewards, record_as

    buffers, _mull, _played = collect_rollout(pairing, n_games, horizon, rng, device="cpu", record=True, greedy=False)
    buf = buffers[deck]
    assert len(meta) == len(buf), f"recorder saw {len(meta)} decisions, buffer holds {len(buf)}"
    episodes, start = [], 0
    for i in range(len(buf)):
        if buf.done[i]:
            sl = slice(start, i + 1)
            # The identity in each (vocab index, feature row, identity) token points into the live engine state
            # (unpicklable, and only the pointer head's action lookup uses it): the probe keeps the other two.
            tokens = [[(idx, row, None) for idx, row, _identity in tl] for tl in buf.token_lists[sl]]
            episodes.append({"tokens": tokens, "scalar": buf.scalar[sl], "mask": buf.mask[sl],
                             "action": buf.action[sl],
                             "probe": [_selected(meta[j], buf.mask[j]) for j in range(start, i + 1)]})
            start = i + 1
    return {"deck": deck, "episodes": episodes, "seed": seed, "n_games": n_games}


def _trim(episodes):
    """Drops episodes with no probe decision and everything after an episode's last one (the replay never reads it)."""
    out = []
    for e in episodes:
        last = max((i for i, s in enumerate(e["probe"]) if s), default=-1) + 1
        if last:
            out.append({k: v[:last] for k, v in e.items()})
    return out


def table_fingerprint(fixed_table):
    """Hash of a deck's fixed action table's labels, in order: the probe's stored action indices only mean
    anything against the same table."""
    return hashlib.sha256("\n".join(row[0] for row in fixed_table).encode()).hexdigest()


def save_probe(path, probe, vocab, fixed_table):
    """Writes the probe set gzip-compressed (~3 MB for 80 games). Card vocabulary indices are replaced by card names
    (vocab.json is append-only, so its indices depend on the order decks were first seen and are not stable across
    a rebuilt vocab); load_probe maps them back through whatever vocab it is given."""
    by_index = {i: name for name, i in vocab.name_to_index.items()}
    episodes = []
    for e in _trim(probe["episodes"]):
        tokens = [[(by_index.get(idx, idx), row, None) for idx, row, _identity in step] for step in e["tokens"]]
        episodes.append({**e, "tokens": tokens})
    saved = {**probe, "episodes": episodes, "table_sha": table_fingerprint(fixed_table)}
    with gzip.open(path, "wb", compresslevel=5) as f:
        pickle.dump(saved, f, protocol=4)


def load_probe(path, vocab, fixed_table):
    """Reads a save_probe file for the given vocab and the deck's current fixed action table. Raises ValueError if the
    table's labels changed since it was recorded (the stored actions would mean something else) or a stored card is
    missing from the vocab."""
    with gzip.open(path, "rb") as f:
        probe = pickle.load(f)
    if probe["table_sha"] != table_fingerprint(fixed_table):
        raise ValueError(f"{path}: the deck's action table changed since the probe set was recorded")
    for e in probe["episodes"]:
        e["tokens"] = [[(vocab.name_to_index[name] if isinstance(name, str) else name, row, None)
                        for name, row, _identity in step] for step in e["tokens"]]
    return probe


def probe_distributions(net, probe, device="cpu", batch_episodes=16):
    """The net's action distribution at every probe decision: a list of 1-D float arrays, one per probe decision in
    (episode, step) order, each over that decision's legal actions in ascending index order (so two checkpoints'
    vectors line up element for element)."""
    net.eval()
    n_fixed = net.non_targeting_head.out_features
    out = []
    eps = probe["episodes"]
    with torch.no_grad():
        for b0 in range(0, len(eps), batch_episodes):
            chunk = eps[b0:b0 + batch_episodes]
            max_steps = max(len(e["action"]) for e in chunk)
            flat_tokens, flat_scalar, flat_mask, prev = [], [], [], []
            for e in chunk:
                n = len(e["action"])
                idx = list(range(n)) + [n - 1] * (max_steps - n)  # ragged episodes padded by repeating the last step
                flat_tokens += [e["tokens"][i] for i in idx]
                flat_scalar += [e["scalar"][i] for i in idx]
                flat_mask += [e["mask"][i] for i in idx]
                prev += [-1] + [e["action"][i] for i in idx[:-1]]
            vocab_idx, features, pad, _ids = pad_token_batch(flat_tokens, device=device)
            mask = torch.zeros((len(flat_mask), n_fixed + vocab_idx.shape[1]), dtype=torch.bool)
            for i, stored in enumerate(flat_mask):
                mask[i, :len(stored)] = torch.as_tensor(np.asarray(stored, dtype=bool))
            mine, theirs, reps = net.encoder(vocab_idx, features, pad, features[:, :, -1])
            scalar = torch.as_tensor(np.array(flat_scalar), dtype=torch.float32, device=device)
            logits, _value, _h = net(mine, theirs, scalar, reps, mask[:, n_fixed:], seq=(len(chunk), max_steps),
                                     prev_action=net.prev_action_symbols(prev, device))
            probs = torch.softmax(logits.masked_fill(~mask, -1e8), dim=-1)
            for row, e in enumerate(chunk):
                for step, selected in enumerate(e["probe"]):
                    if selected:
                        i = row * max_steps + step
                        out.append(probs[i][mask[i]].numpy().astype(np.float64))
    return out


def js_divergence(p, q):
    """Jensen-Shannon divergence (nats, 0..ln 2) of two probability vectors over the same support."""
    m = 0.5 * (p + q)

    def kl(a, b):
        nz = a > 0
        return float(np.sum(a[nz] * np.log(a[nz] / b[nz])))
    return 0.5 * kl(p, m) + 0.5 * kl(q, m)


def compare(dists_a, dists_b):
    """(mean JS divergence, top-1 agreement) between two checkpoints over the same probe decisions."""
    assert len(dists_a) == len(dists_b)
    js = np.mean([js_divergence(p, q) for p, q in zip(dists_a, dists_b)])
    agree = np.mean([int(np.argmax(p) == np.argmax(q)) for p, q in zip(dists_a, dists_b)])
    return float(js), float(agree)


def mean_entropy(dists):
    return float(np.mean([-np.sum(p[p > 0] * np.log(p[p > 0])) for p in dists]))
