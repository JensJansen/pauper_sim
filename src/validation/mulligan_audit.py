"""Check: mulligan_audit -- % of hands kept vs. mulliganed by land count
(0-7), primary league only, per deck plus a league-wide rollup.

Does NOT play any games itself: it's a post-hoc analysis of the games
round_robin_primary and round_robin_training already played this cadence
point (ctx.collected_game_logs / ctx.collected_deck_league) -- registry
order in validation/__init__.py puts this check after both, so that data
already exists by the time it runs. A non-primary seat in a cross-league
game (the training-league side) is excluded by construction: only entries
present in the per-game seat->deck map count.

Extends analysis.mulligan_retrain._mulligan_common.audit_land_counts with
per-deck attribution rather than reimplementing the land-count/hand
reconstruction logic.
"""
from analysis.mulligan_retrain._mulligan_common import audit_land_counts

from . import _common

NAME = "mulligan_audit"


def _rate_table(by_lc):
    """{land_count: {...}} -> the same, plus a keep_rate and avg_p_keep,
    with the non-serializable keep_probs list dropped (matches
    train_mulligan.py's own precedent for writing this out to JSON)."""
    table = {}
    for lc in sorted(by_lc):
        d = by_lc[lc]
        decided = d["kept"] + d["mulliganed"]
        avg_p = sum(d["keep_probs"]) / len(d["keep_probs"]) if d["keep_probs"] else None
        wl = d["wins"] + d["losses"]
        table[str(lc)] = {
            "kept": d["kept"], "mulliganed": d["mulliganed"],
            "keep_rate": d["kept"] / decided if decided else None,
            "avg_p_keep_when_kept": avg_p,
            "win_rate_after_keep": d["wins"] / wl if wl else None,
            "wins": d["wins"], "losses": d["losses"],
        }
    return table


def run(ctx):
    if not ctx.collected_game_logs:
        return {"skipped": "no games collected this cadence point (round-robin checks ran first?)"}

    deck_by_game_seat = [
        {seat: deck for seat, (league, deck) in game_map.items() if league == "primary"}
        for game_map in ctx.collected_deck_league
    ]
    by_deck = audit_land_counts(ctx.collected_game_logs, deck_by_game_seat)

    per_deck_tables = {}
    for deck in ctx.train_decks:
        table = _rate_table(by_deck.get(deck, {}))
        per_deck_tables[deck] = table
        payload = {"check": NAME, "primary_league": ctx.primary_league_name, "deck": deck,
                  "cumulative_games": ctx.cumulative_games, "by_land_count": table}
        _common.write_deck_json(ctx, deck, NAME, payload)

        decided_total = sum(d["kept"] + d["mulliganed"] for d in by_deck.get(deck, {}).values())
        mulliganed_total = sum(d["mulliganed"] for d in by_deck.get(deck, {}).values())
        _common.append_metric(ctx, kind=NAME, deck=deck,
                              decisions=decided_total, mulligan_rate=mulliganed_total / decided_total
                              if decided_total else None, by_land_count=table)

    league_payload = {"check": NAME, "primary_league": ctx.primary_league_name,
                      "cumulative_games": ctx.cumulative_games, "decks": per_deck_tables}
    write_path = _common.write_league_json(ctx, NAME, league_payload)

    return {"decks": len(per_deck_tables), "wrote": write_path}
