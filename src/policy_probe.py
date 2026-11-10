"""Policy-movement probe over a league's archived snapshots (see evaluation/probe.py for what it measures).

  python policy_probe.py record --league DIR --probes OUTDIR [--decks a,b] [--games 60] [--seed 3]
                                [--ref-ages 10000,40000,final] [--workers 6]
      Plays fixed-seed reference games of each deck (the deck at each reference age against random decks at random
      reference ages) and stores them as probe_<deck>.pkl.gz in OUTDIR. Done once; the same set is used for every
      checkpoint, forever.
  python policy_probe.py run --league DIR --probes OUTDIR [--ages 1000,...,final] [--out FILE.json] [--workers 6]
      For each deck and age: the policy's action distribution over the probe decisions, then its distance (mean
      Jensen-Shannon divergence, nats; top-1 agreement) to the previous age and to the final policy, and its entropy.
      Scale references per deck: the same measure between two snapshots ONE snapshot interval apart (the normal
      step), and between a randomly initialised net and the final policy (unrelated policies).
  python policy_probe.py report --in FILE.json   re-print a saved run.
"""
import argparse
import json
import os
import sys
from concurrent.futures import ProcessPoolExecutor

from repo_paths import SRC_DIR

DEFAULT_AGES = "1000,2000,3000,5000,7500,10000,15000,20000,25000,30000,40000,50000,60000,70000,80000,90000,final"
ADJACENT_AT = (20000, 60000, 90000)  # where the one-snapshot-apart baseline is measured


def _agent_factory(league, deck, snapshot):
    from evaluation import play
    from rl.decision.agent import SeatAgent

    def make():
        net, mnet = play._load(league, deck, snapshot)
        return SeatAgent(net, mnet, play._pool()[2][deck])
    return make


def _record_deck(job):
    import torch
    league, deck, decks, ref_ages, n_games, seed, outdir, games_per_snapshot = job
    from evaluation import history, play
    from rl.league import league_runner
    from evaluation.probe import record_probe_set, save_probe

    torch.set_num_threads(1)
    torch.manual_seed(seed)
    decklists, vocab, deck_ctxs, tables = play._pool()
    factories = {}
    for d in decks:
        for label, sid, _games in history.at_ages(league, d, ref_ages, games_per_snapshot):
            factories[(d, label)] = _agent_factory(league, d, sid)
    probe = record_probe_set(deck, factories, decklists, deck_ctxs, n_games, seed, league_runner.HORIZON)
    path = os.path.join(outdir, f"probe_{deck}.pkl.gz")
    n_probe = sum(sum(e["probe"]) for e in probe["episodes"])
    save_probe(path, probe, vocab, tables[deck])
    return deck, len(probe["episodes"]), sum(len(e["action"]) for e in probe["episodes"]), n_probe


def _run_deck(job):
    import torch
    league, deck, probes_dir, ages, games_per_snapshot = job
    from evaluation import history, play
    from evaluation.probe import compare, load_probe, mean_entropy, probe_distributions
    from rl.league import league_runner

    torch.set_num_threads(1)
    _decklists, vocab, _ctxs, tables = play._pool()
    probe = load_probe(os.path.join(probes_dir, f"probe_{deck}.pkl.gz"), vocab, tables[deck])

    def dists(snapshot):
        net, _mnet = play._load(league, deck, snapshot)
        return probe_distributions(net, probe)

    pts = history.at_ages(league, deck, ages, games_per_snapshot)
    final_sid = history.at_ages(league, deck, ["final"], games_per_snapshot)[0][1]
    final = dists(final_sid)
    rows, prev = [], None
    for _label, sid, games in pts:
        d = final if sid == final_sid else dists(sid)
        row = {"age": games, "entropy": mean_entropy(d)}
        row["js_final"], row["top1_final"] = compare(d, final)
        row["js_prev"], row["top1_prev"] = compare(d, prev) if prev is not None else (None, None)
        rows.append(row)
        prev = d
    adjacent = []
    for target in ADJACENT_AT:
        sid = history.at_ages(league, deck, [target], games_per_snapshot)[0][1]
        if sid + 1 <= final_sid:
            adjacent.append({"age": history.games_at(sid, games_per_snapshot),
                             **dict(zip(("js", "top1"), compare(dists(sid), dists(sid + 1))))})
    torch.manual_seed(0)
    random_net = league_runner.build_deck_net(vocab.size, len(tables[deck]))
    rnd_js, rnd_top1 = compare(probe_distributions(random_net, probe), final)
    return {"deck": deck, "n_probe_decisions": len(final), "ages": rows, "adjacent": adjacent,
            "random_vs_final": {"js": rnd_js, "top1": rnd_top1}, "final_entropy": mean_entropy(final)}


def report(result):
    import numpy as np
    decks = result["decks"]
    ages = [r["age"] for r in decks[0]["ages"]]

    def table(title, key, fmt):
        print(f"\n{title}")
        print(f"{'deck':20s}" + "".join(f"{a:>8d}" for a in ages))
        for d in decks:
            print(f"{d['deck']:20s}" + "".join(f"{fmt(r[key]):>8s}" if r[key] is not None else f"{'':>8s}"
                                                for r in d["ages"]))
        vals = [[r[key] for r in d["ages"]] for d in decks]
        print(f"{'MEAN':20s}" + "".join(
            f"{fmt(float(np.mean([v[i] for v in vals if v[i] is not None]))):>8s}" if vals[0][i] is not None else f"{'':>8s}"
            for i in range(len(ages))))

    n = [d["n_probe_decisions"] for d in decks]
    print(f"probe decisions per deck: min {min(n)}, max {max(n)}")
    table("DISTANCE TO THE FINAL POLICY (mean JS divergence, nats; 0 = identical, ln2=0.693 = disjoint)", "js_final",
          lambda v: f"{v:.3f}")
    table("TOP-1 AGREEMENT WITH THE FINAL POLICY (share of probe decisions with the same favourite action)", "top1_final",
          lambda v: f"{v:.2f}")
    table("DISTANCE TO THE PREVIOUS AGE LISTED (JS)", "js_prev", lambda v: f"{v:.3f}")
    table("POLICY ENTROPY (nats, mean over probe decisions)", "entropy", lambda v: f"{v:.2f}")
    adj = [a["js"] for d in decks for a in d["adjacent"]]
    adj1 = [a["top1"] for d in decks for a in d["adjacent"]]
    print(f"\nSCALE: JS between snapshots ONE interval (180 games) apart, mean over decks and ages "
          f"{', '.join(map(str, ADJACENT_AT))}: {np.mean(adj):.4f}  (top-1 agreement {np.mean(adj1):.2f})")
    print(f"SCALE: JS between a randomly initialised net and the final policy, mean over decks: "
          f"{np.mean([d['random_vs_final']['js'] for d in decks]):.3f}  (top-1 agreement "
          f"{np.mean([d['random_vs_final']['top1'] for d in decks]):.2f})")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("mode", choices=["record", "run", "report"])
    ap.add_argument("--league")
    ap.add_argument("--probes")
    ap.add_argument("--decks")
    ap.add_argument("--games", type=int, default=60)
    ap.add_argument("--seed", type=int, default=3)
    ap.add_argument("--ref-ages", default="10000,40000,final")
    ap.add_argument("--ages", default=DEFAULT_AGES)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--games-per-snapshot", type=int, default=180)
    ap.add_argument("--out")
    ap.add_argument("--in", dest="infile")
    args = ap.parse_args(argv)

    league = os.path.abspath(args.league) if args.league else None
    probes = os.path.abspath(args.probes) if args.probes else None
    out = os.path.abspath(args.out) if args.out else None
    infile = os.path.abspath(args.infile) if args.infile else None
    os.chdir(SRC_DIR)  # build_pool's data paths are relative to src/
    if str(SRC_DIR) not in sys.path:
        sys.path.insert(0, str(SRC_DIR))
    if args.mode == "report":
        with open(infile) as f:
            report(json.load(f))
        return

    from evaluation import history
    from evaluation.matchups import available_decks

    decks = args.decks.split(",") if args.decks else available_decks(league)
    for d in decks:
        history.verify(league, d, args.games_per_snapshot)
    ages = [a if a == "final" else int(a) for a in args.ages.split(",")]
    if args.mode == "record":
        os.makedirs(probes, exist_ok=True)
        ref_ages = [a if a == "final" else int(a) for a in args.ref_ages.split(",")]
        jobs = [(league, d, decks, ref_ages, args.games, args.seed + i, probes, args.games_per_snapshot)
                for i, d in enumerate(decks)]
        fn = _record_deck
    else:
        jobs = [(league, d, probes, ages, args.games_per_snapshot) for d in decks]
        fn = _run_deck
    if args.workers > 1:
        with ProcessPoolExecutor(max_workers=args.workers) as ex:
            results = list(ex.map(fn, jobs))
    else:
        results = [fn(j) for j in jobs]
    if args.mode == "record":
        for deck, n_eps, n_dec, n_probe in results:
            print(f"{deck:20s} {n_eps} games, {n_dec} decisions, {n_probe} probe decisions", flush=True)
        return
    result = {"league": league, "decks": results}
    if out:
        with open(out, "w") as f:
            json.dump(result, f)
        print(f"wrote {out}", flush=True)
    report(result)


if __name__ == "__main__":
    main()
