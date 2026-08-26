"""Check: mulligan_audit -- % of hands kept vs. mulliganed by land count
(0-7), primary league only, per deck plus a league-wide rollup.

Two data sources, both per deck:

  1. NATURAL-GAME AUDIT (by_land_count). Post-hoc analysis of the games
     round_robin_primary and round_robin_training already played this
     cadence point (ctx.collected_game_logs / ctx.collected_deck_league) --
     registry order in validation/__init__.py puts this check after both,
     so that data already exists by the time it runs. A non-primary seat in
     a cross-league game (the training-league side) is excluded by
     construction: only entries present in the per-game seat->deck map
     count. Whatever land counts self-play happened to draw -- 0- and
     7-land hands are rare or absent, so those buckets are thin or missing.
  2. SCULPTED-HAND PROBE (probe_hands). Loads the deck's current live +
     mulligan weights and queries fixed, seeded synthetic hands covering
     every land count 0-7 on demand (analysis.mulligan_retrain
     ._mulligan_common.build_probe_hands_sampled/probe_land_count_stats) --
     fills in exactly the buckets (1) can't reliably reach, and reports the
     P(mulligan) it lands on regardless of what self-play happened to draw.

Both report an entropy figure (entropy_bits, 0-1, of the keep/mulligan
decision) alongside keep/mulligan rate -- a policy that always keeps has
collapsed to ~0 bits regardless of whether the keep_rate itself still looks
land-sensitive on average.

Extends analysis.mulligan_retrain._mulligan_common.audit_land_counts with
per-deck attribution rather than reimplementing the land-count/hand
reconstruction logic.
"""
from analysis.mulligan_retrain._mulligan_common import (
    audit_land_counts, build_probe_hands_sampled, probe_land_count_stats,
)

from . import _common

NAME = "mulligan_audit"
PROBE_VARIANTS_PER_LAND_COUNT = 6  # sculpted hands per land count; see build_probe_hands_sampled


def _rate_table(by_lc):
    """{land_count: {...}} -> the same, plus a keep_rate, avg_p_keep, and
    avg_entropy_bits, with the non-serializable keep_probs/entropy_bits
    lists dropped (matches train_mulligan.py's own precedent for writing
    this out to JSON)."""
    table = {}
    for lc in sorted(by_lc):
        d = by_lc[lc]
        decided = d["kept"] + d["mulliganed"]
        avg_p = sum(d["keep_probs"]) / len(d["keep_probs"]) if d["keep_probs"] else None
        avg_ent = sum(d["entropy_bits"]) / len(d["entropy_bits"]) if d["entropy_bits"] else None
        wl = d["wins"] + d["losses"]
        table[str(lc)] = {
            "kept": d["kept"], "mulliganed": d["mulliganed"],
            "keep_rate": d["kept"] / decided if decided else None,
            "avg_p_keep_when_kept": avg_p,
            "avg_entropy_bits": avg_ent,
            "win_rate_after_keep": d["wins"] / wl if wl else None,
            "wins": d["wins"], "losses": d["losses"],
        }
    return table


def _probe_table(mnet, decklist, vocab):
    """{land_count: {...}} sculpted-hand probe stats -- see module docstring."""
    probes = build_probe_hands_sampled(decklist, vocab, n_variants=PROBE_VARIANTS_PER_LAND_COUNT)
    stats = probe_land_count_stats(mnet, probes)
    return {str(lc): stats[lc] for lc in sorted(stats)}


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
        _net, mnet = _common.load_deck_net(ctx.primary_league_dir, deck, ctx.vocab, ctx.fixed_tables[deck])
        probe_table = _probe_table(mnet, ctx.decklists[deck], ctx.vocab)
        per_deck_tables[deck] = {"by_land_count": table, "probe_hands": probe_table}
        payload = {"check": NAME, "primary_league": ctx.primary_league_name, "deck": deck,
                  "cumulative_games": ctx.cumulative_games, "by_land_count": table, "probe_hands": probe_table}
        _common.write_deck_json(ctx, deck, NAME, payload)

        decided_total = sum(d["kept"] + d["mulliganed"] for d in by_deck.get(deck, {}).values())
        mulliganed_total = sum(d["mulliganed"] for d in by_deck.get(deck, {}).values())
        # by_land_count/probe_hands deliberately excluded here -- they're
        # already written in full above (write_deck_json) and in the league
        # rollup below (write_league_json); metrics.jsonl stays a compact
        # trend-line record, matching every other check's append_metric call
        # (vs_history, both round robins).
        _common.append_metric(ctx, kind=NAME, deck=deck,
                              decisions=decided_total, mulligan_rate=mulliganed_total / decided_total
                              if decided_total else None)

    league_payload = {"check": NAME, "primary_league": ctx.primary_league_name,
                      "cumulative_games": ctx.cumulative_games, "decks": per_deck_tables}
    write_path = _common.write_league_json(ctx, NAME, league_payload)

    return {"decks": len(per_deck_tables), "wrote": write_path}
