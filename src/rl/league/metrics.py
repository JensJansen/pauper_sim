"""checkpoints/<league>/metrics.jsonl: the one append-only record of a run, and the one place that defines what is in it.

Every line is a JSON object with a `kind`. Training writes `session_start`, `ppo` and `mulligan` (rl/league/
league_runner.py); each validation check writes the kind(s) named after it (validation/<check>.py). `append`
is the single writer (it also mirrors the line to the web app, see webapp_publish/mirror.py) and `read` the
single reader. KINDS below is the schema: kind -> (who writes it, what its fields mean). A kind not listed here is
still written (telemetry must never stop training) but prints a warning once, and tests/unit/rl/league/test_metrics.py
fails if code emits a kind that is not registered or the README's metrics reference omits one.

Fields shared by every kind: `kind`, `cumulative_games` (games/deck trained when it was written). Training kinds also
carry `session` (this process's session number) and `iteration` (within the session). Readers must use .get: the file
is append-only across many schema revisions."""
import json
import os

from webapp_publish.mirror import mirror_metrics_line

METRICS_FILE = "metrics.jsonl"

# kind -> {"writer": who emits it, "fields": {name: meaning}, "retired": optional note}
KINDS = {
    "session_start": {
        "writer": "league_runner._run_session",
        "fields": {"session": "session number", "reward_fn": "reward function name", "roster": "all decks in the pool",
                   "train_decks": "decks being trained", "n_iterations": "iterations this session",
                   "games_per_iteration": "games/deck per iteration", "snapshot_every": "iterations between snapshots",
                   "checkpoint_rate": "share of games against a frozen snapshot", "pfsp": "PFSP opponent sampling",
                   "n_workers": "collection workers", "horizon": "turn horizon", "pfsp_power": "PFSP exponent",
                   "stratify_0land_pct": "forced 0-land hand rate", "overlap_collect": "collect/update overlapped",
                   "lr / mulligan_lr / gamma / gae_lambda / target_kl / n_epochs / adv_norm_floor / ent_coef":
                       "PPO hyper-parameters in force",
                   "seq_batch_size_start / seq_batch_size_cap": "minibatch ramp"},
    },
    "ppo": {
        "writer": "league_runner._run_session (per deck per iteration)",
        "fields": {"deck": "deck", "games": "games collected", "buffer_size": "transitions in the update",
                   "batch_size": "episodes per minibatch", "salvaged": "transitions salvaged from other decks' games",
                   "ent_coef": "entropy coefficient", "policy_loss": "", "value_loss": "",
                   "entropy": "mean policy entropy (nats)", "approx_kl": "k3 estimate after the update",
                   "clip_fraction": "share of ratios clipped", "epochs_run": "epochs before target_kl stopped it",
                   "explained_variance": "critic fit before the update", "adv_std": "raw advantage spread",
                   "target_kl": "the early-stop threshold in force"},
    },
    "mulligan": {
        "writer": "league_runner._run_session (per deck per REINFORCE flush)",
        "fields": {"deck": "deck", "n": "mulligan decisions in the flush", "loss": "REINFORCE loss"},
    },
    "mulligan_audit": {
        "writer": "validation.mulligan_audit",
        "fields": {"deck": "deck or ALL", "decisions": "mulligan decisions audited", "mulligan_rate": "share mulliganed",
                   "by_land_count": "keep/mulligan by land count (natural games)",
                   "probe_hands": "P(mulligan) on synthetic hands of every land count"},
    },
    "primary_vs_primary_round_robin": {
        "writer": "validation.round_robin_primary",
        "fields": {"deck": "deck", "games": "games vs OTHER decks (mirrors excluded since 2026-11-10; older lines "
                                           "counted each mirror game as a win)", "wins": "", "win_rate": ""},
    },
    "primary_vs_training_round_robin": {
        "writer": "validation.round_robin_training",
        "fields": {"deck": "deck", "games": "games vs the independently trained twin league", "wins": "",
                   "win_rate": "", "mana_burnt_by_deck": "diagnostic, on this kind only"},
    },
    "strength_history": {
        "writer": "validation.strength_history (own)",
        "fields": {"deck": "deck", "anchor_games": "age (games/deck) of the deck's own snapshot played",
                   "games": "games in the cell", "score": "current agent's score against that snapshot"},
    },
    "strength_field": {
        "writer": "validation.strength_history (field)",
        "fields": {"deck": "OPPONENT deck B", "anchor_games": "B's snapshot age",
                   "games": "games, pooled over the other decks' current agents",
                   "score": "their pooled score against B@age"},
    },
    "policy_movement": {
        "writer": "validation.policy_movement",
        "fields": {"deck": "deck", "anchor_games": "age of the snapshot compared with",
                   "js": "mean Jensen-Shannon divergence, current vs that snapshot, on the probe suite (nats)",
                   "top1": "share of probe decisions with the same favourite action"},
    },
    "policy_scale": {
        "writer": "validation.policy_movement",
        "fields": {"deck": "deck", "entropy": "current policy entropy on the probe suite",
                   "adjacent_js": "JS between the newest two snapshots (one interval apart)",
                   "random_js": "JS between a random net and the current policy"},
    },
    "progress_indicator": {
        "writer": "validation.progress_indicator",
        "fields": {"deck": "ALL", "label": "IMPROVING / CHURNING / STUCK / FLAT / WARMUP (provisional thresholds)",
                   "score": "pooled score of current vs ~0.6x-age selves", "lo / hi": "95% interval",
                   "games": "games behind the score", "policy_ratio": "policy distance as a fraction of the random distance"},
    },
    "vs_history": {
        "writer": "(retired 2026-11-10; legacy records only)",
        "retired": True,
        "fields": {"deck": "deck", "label": "archive_oldest | active_oldest", "snapshot_id": "", "games": "",
                   "live_wins": "", "win_rate": ""},
    },
}

_warned = set()


def append(league_dir, **fields):
    """Appends one line (fields must include `kind`) to <league_dir>/metrics.jsonl and mirrors it to the web app.
    Does not create league_dir -- the caller's LeaguePool construction (training) or validation._common
    (checks) already does."""
    kind = fields.get("kind")
    if kind not in KINDS and kind not in _warned:
        _warned.add(kind)
        print(f"metrics: kind {kind!r} is not in rl.league.metrics.KINDS -- add it (written anyway)", flush=True)
    with open(os.path.join(league_dir, METRICS_FILE), "a") as f:
        f.write(json.dumps(fields) + "\n")
    mirror_metrics_line(league_dir, fields)  # best-effort webapp copy, see webapp_publish/mirror.py


def read(league_dir, kinds=None):
    """Yields every record in <league_dir>/metrics.jsonl (optionally only those whose kind is in `kinds`), in the
    order written; a blank or truncated line (an interrupted append) is skipped."""
    path = os.path.join(league_dir, METRICS_FILE)
    if not os.path.exists(path):
        return
    with open(path) as f:
        for line in f:
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            if kinds is None or record.get("kind") in kinds:
                yield record
