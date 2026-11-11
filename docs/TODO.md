# TODO

## Investigate entropy-driven entropy_coeff as a mechanism to force changes based on agent confidence

Currently `ent_coef` (the PPO entropy-bonus weight) is computed once per
iteration from a single global schedule (`ent_coef_schedule`, keyed only on
cumulative games trained) and applied identically to all 11 decks -- see
`league_runner.py`'s per-iteration `ent_coef` computation. It is blind to
each deck's actual policy entropy.

Real per-deck policy entropy (already computed in `ppo_update` and logged to
`metrics.jsonl` per deck, just unused for control) varies substantially
across decks -- e.g. recently `mono_red_madness` ~0.20 (confident/peaked)
vs. `monster_tron` ~0.56 (closer to uniform/uncertain). The idea: adapt each
deck's own `ent_coef` toward a target entropy level (SAC-style automatic
temperature tuning) instead of one global anneal, so a deck's exploration
pressure reflects its own current confidence rather than total games trained
league-wide.

Open concern, not yet resolved: measured iteration-to-iteration noise in
per-deck entropy is 14-29% of each deck's own mean entropy, which is large
enough that a naive controller reacting to the raw per-iteration value would
likely be chasing noise rather than real drift in confidence. Any adaptive
scheme probably needs a smoothed (e.g. EMA) entropy input rather than the
raw per-iteration reading -- untested.

## Agent managed sideboarding

## Deck dropin training protocol

## Game playing harness to compete with agents
