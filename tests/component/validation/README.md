# Component: validation checks (`src/validation/`)

**Contract.** Each periodic check plays real games with the real engine and networks and writes correct, mirrored records
(`metrics.jsonl`, per-check JSON, replay files) without ever stopping training.

Covered today: the primary/training round robins (including the parallel path), the review replays, the mulligan audit.
Pure-logic checks (registry failure isolation, history-check decision rules, output plumbing) are unit tests in `tests/unit/validation/`.

**Cost.** ~2 minutes. **Planted bugs.** None yet.

**Gaps (next work).** Win-rate and score arithmetic against hand-computed tables; interval/label thresholds at their boundaries;
mirrored copies in the web app equal the originals.
