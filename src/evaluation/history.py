"""A league directory's training history: every snapshot a deck ever took, addressable by games trained.

A deck's snapshots (`snapshot_<id>.pt`, the active window in <deck>/ plus everything evicted to <deck>/archive/) are
the whole frozen agent -- deck net plus its era-matched mulligan net -- taken every `games_per_snapshot` games per
deck (snapshot_every iterations x games_per_iteration; 3 x 60 = 180 for main-league, constant across all of its
sessions). Snapshot k was therefore taken at (k + 1) * games_per_snapshot games/deck, which lets a run be evaluated
retroactively at any age with no extra storage."""
import json
import os

GAMES_PER_SNAPSHOT = 180


def _ids(directory):
    if not os.path.isdir(directory):
        return []
    return [int(fn[len("snapshot_"):-len(".pt")]) for fn in os.listdir(directory)
            if fn.startswith("snapshot_") and fn.endswith(".pt")]


def snapshot_ids(league_dir, deck):
    """Every snapshot id the deck has, active window and archive together, sorted."""
    deck_dir = os.path.join(league_dir, deck)
    return sorted(_ids(deck_dir) + _ids(os.path.join(deck_dir, "archive")))


def snapshot_path(league_dir, deck, snapshot):
    for sub in ("", "archive"):
        path = os.path.join(league_dir, deck, sub, f"snapshot_{snapshot}.pt")
        if os.path.exists(path):
            return path
    raise FileNotFoundError(f"{deck}: no snapshot_{snapshot}.pt under {league_dir}")


def games_at(snapshot, games_per_snapshot=GAMES_PER_SNAPSHOT):
    """Games/deck trained when snapshot `snapshot` was taken."""
    return (snapshot + 1) * games_per_snapshot


def verify(league_dir, deck, games_per_snapshot=GAMES_PER_SNAPSHOT):
    """Raises unless the deck's snapshots are exactly 0..n-1 and the last one sits at the league's recorded
    games/deck (progress.json) -- i.e. unless the (id -> games) mapping above is true for this directory.
    Returns the ids."""
    ids = snapshot_ids(league_dir, deck)
    if ids != list(range(len(ids))):
        raise ValueError(f"{deck}: snapshot ids are not contiguous from 0 ({ids[:3]}...{ids[-3:]})")
    with open(os.path.join(league_dir, "progress.json")) as f:
        trained = json.load(f)["cumulative_games_per_deck"]
    if games_at(ids[-1], games_per_snapshot) != trained:
        raise ValueError(f"{deck}: last snapshot is at {games_at(ids[-1], games_per_snapshot)} games but progress.json "
                         f"says {trained}; pass the right games_per_snapshot")
    return ids


def at_ages(league_dir, deck, ages, games_per_snapshot=GAMES_PER_SNAPSHOT):
    """[(label, snapshot id, games at that snapshot)] for each requested age -- the snapshot nearest the age, in
    games/deck; the string "final" is the deck's newest snapshot. Duplicate snapshots are dropped."""
    ids = snapshot_ids(league_dir, deck)
    out, seen = [], set()
    for age in ages:
        sid = ids[-1] if age == "final" else min(ids, key=lambda i: abs(games_at(i, games_per_snapshot) - age))
        if sid not in seen:
            seen.add(sid)
            out.append((str(age), sid, games_at(sid, games_per_snapshot)))
    return out


def ladder_points(league_dir, deck, current_games, ladder, games_per_snapshot=GAMES_PER_SNAPSHOT):
    """[(games at the snapshot, snapshot id)] -- the deck's snapshot nearest each rung of `ladder` (games/deck),
    keeping only rungs at least two snapshot intervals older than `current_games` (anything closer is the agent
    against itself) and dropping duplicates. For a periodic check: the anchors an agent at `current_games` is
    measured against."""
    out, seen = [], set()
    for rung in ladder:
        if rung > current_games - 2 * games_per_snapshot:
            break
        (_label, sid, games), = at_ages(league_dir, deck, [rung], games_per_snapshot)
        if sid not in seen and games <= current_games - 2 * games_per_snapshot:
            seen.add(sid)
            out.append((games, sid))
    return out
