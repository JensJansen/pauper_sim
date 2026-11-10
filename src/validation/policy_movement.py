"""Check: policy_movement -- how far each deck's CURRENT policy has moved from its own past policies, measured on a
fixed suite of mid-game decisions (evaluation/probe.py has the what and the why; the suite is recorded once with
`python -m evaluation probe record` and kept under ctx.probe_states_dir, e.g. data/probe_states/small_league/).

Per deck: mean Jensen-Shannon divergence (nats; 0 identical .. ln 2 disjoint) and top-1 agreement between the current
policy and its snapshots at the strength_history ladder ages, plus the policy's entropy and two scale references --
the newest two snapshots (one snapshot interval apart: a normal step) and a randomly initialised net against the
current policy (unrelated policies). Skipped for a deck with no probe file, and entirely when probe_states_dir is
unset. Movement is not progress: read it with strength_history (progress_indicator does)."""
import os

import torch

from . import _common
from .strength_history import DEFAULT_LADDER

NAME = "policy_movement"


def _deck_worker(league, deck, probe_path, anchors, last_two, own_threads):
    from evaluation import play
    from evaluation.probe import compare, load_probe, mean_entropy, probe_distributions
    from rl.league import league_runner

    prev_threads = torch.get_num_threads()
    if own_threads:
        torch.set_num_threads(1)
    try:
        _dl, vocab, _ctxs, tables = play._pool()
        probe = load_probe(probe_path, vocab, tables[deck])

        def dists(snapshot):
            return probe_distributions(play._load(league, deck, snapshot)[0], probe)

        current = dists(None)
        if not current:
            return None  # a probe file with no probe decisions: nothing to measure
        rows = []
        for games, sid in anchors:
            js, top1 = compare(current, dists(sid))
            rows.append({"anchor_games": games, "js": js, "top1": top1})
        adjacent = None
        if last_two is not None:
            adjacent = dict(zip(("js", "top1"), compare(dists(last_two[0]), dists(last_two[1]))))
        torch.manual_seed(0)
        random_net = league_runner.build_deck_net(vocab.size, len(tables[deck]))
        random_js, random_top1 = compare(probe_distributions(random_net, probe), current)
        return {"deck": deck, "n_probe_decisions": len(current), "entropy": mean_entropy(current), "anchors": rows,
                "adjacent": adjacent, "random": {"js": random_js, "top1": random_top1}}
    finally:
        torch.set_num_threads(prev_threads)


def run(ctx):
    from evaluation import history
    if not ctx.probe_states_dir or ctx.games_per_snapshot is None:
        return {"skipped": "probe_states_dir / games_per_snapshot not set"}
    league = ctx.primary_league_dir
    ladder = ctx.history_ladder or DEFAULT_LADDER
    jobs = {}
    for d in ctx.train_decks:
        path = os.path.join(ctx.probe_states_dir, f"probe_{d}.pkl.gz")
        if not os.path.exists(path):
            continue
        ids = history.snapshot_ids(league, d)
        anchors = history.ladder_points(league, d, ctx.cumulative_games, ladder, ctx.games_per_snapshot)
        jobs[d] = (league, d, path, anchors, tuple(ids[-2:]) if len(ids) >= 2 else None)
    if not jobs:
        return {"skipped": "no probe file for any training deck"}
    if ctx.executor is None or ctx.n_workers <= 1:
        per_deck = {d: _deck_worker(*job, False) for d, job in jobs.items()}
    else:
        futures = {d: ctx.executor.submit(_deck_worker, *job, True) for d, job in jobs.items()}
        per_deck = {d: f.result() for d, f in futures.items()}
    per_deck = {d: r for d, r in per_deck.items() if r is not None}
    if not per_deck:
        return {"skipped": "no probe file held a probe decision"}

    payload = {"check": NAME, "primary_league": ctx.primary_league_name, "cumulative_games": ctx.cumulative_games,
               "decks": per_deck}
    write_path = _common.write_league_json(ctx, NAME, payload)
    for d, r in per_deck.items():
        for a in r["anchors"]:
            _common.append_metric(ctx, kind=NAME, deck=d, anchor_games=a["anchor_games"], js=a["js"], top1=a["top1"])
        _common.append_metric(ctx, kind="policy_scale", deck=d, entropy=r["entropy"],
                              adjacent_js=r["adjacent"]["js"] if r["adjacent"] else None, random_js=r["random"]["js"])
    ctx.extras[NAME] = per_deck
    return {"decks": len(per_deck), "wrote": write_path}
