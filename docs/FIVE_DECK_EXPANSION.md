# Five-Deck Expansion + Full Retrain Checklist

Goal: expand the league roster from 2 decks (mono_red_madness, rakdos_madness)
to all 5 known decks (+ spy_combo, boggles, monster_tron), for both training
and the opponent pool, and restart training cleanly on the larger vocabulary.

**Status: NOT started. Do not begin until explicitly told to.** This is a
plan only. The two config edits below are currently REVERTED to the 2-deck
state so a madness-vs-madness match stays runnable against the existing
trained weights.

## Why a full retrain is unavoidable

The frozen shared perception stack (`checkpoints/shared_stack_frozen.pt`) has
an `nn.Embedding(vocab_size, d_model)` table sized for exactly the **23-card
vocab** of the current 2 decks + 2 tokens (Blood, Robot). Adding 3 decks (and
2 more token types) grows the vocab well past 23. A frozen embedding table's
row count cannot be extended after creation, and `run_league.py`/
`load_frozen_stack` asserts `ckpt["vocab_size"] == vocab.size` and refuses to
run on a mismatch. Therefore the shared stack must be re-pretrained (Phase 4)
over the full 5-deck vocab, and every DeckNetwork trained against the old
stack's outputs (`checkpoints/league/*/live.pt`, the 2000-game result) becomes
incompatible and must be retrained from scratch on the new stack.

Nothing is deleted -- old checkpoints get **archived**, not discarded.

## Checklist

### 1. Re-apply the 5-deck wiring (currently reverted)
- [ ] `src/token_pool.py`: `TOKEN_DEFS` -> all 4 token defs
      `(game.BLOOD_TOKEN_CARD_DEF, game.ROBOT_TOKEN_CARD_DEF,
        game.WARRIOR_TOKEN_CARD_DEF, game.ELDRAZI_SPAWN_TOKEN_CARD_DEF)`
      (Warrior <- Cartouche of Solidarity, Eldrazi Spawn <- Malevolent Rumble;
      both in boggles. Missing token def == hard KeyError mid-game.)
- [ ] `data/league_decks.json`: all 5 decks
      (mono_red_madness, rakdos_madness, spy_combo, boggles, monster_tron).

### 2. Archive the old (2-deck) artifacts -- do NOT delete
- [ ] Move to `checkpoints/archive_2deck/`:
      `shared_stack_frozen.pt`, `pretrain_shared_stack.pt`, `vocab.json`,
      the whole `league/` subtree (both decks' `live.pt`, all `snapshot_*.pt`,
      `session.txt`).
- [ ] After archiving, `checkpoints/vocab.json` must be absent so it rebuilds
      fresh from all 5 decks (append-safe, but a clean rebuild avoids carrying
      the old 23-card ordering; either is correct since it's a full reset).

### 3. Smoke test the 3 NEW decks in 2-player BEFORE the long Phase 4
- [ ] Run a handful of real 2-player games with spy_combo, boggles, and
      monster_tron (mirror and cross), confirming each runs to completion
      without a crash. This is where any 2-player-only card bug surfaces
      (cf. the discard/CardDef bug that only appeared in a real madness game).
      monster_tron is the risk case -- a slow ramp deck; confirm it can
      actually reach a win/loss within `horizon=120`, not just durdle.
- [ ] Watch specifically for: unregistered-token KeyErrors, non-serializable
      objects if logging, and stalls (same action repeated pointlessly).

### 4. Re-run Phase 4 (pretrain + freeze the shared stack) over all 5 decks
- [ ] `python run_pretrain.py <iterations> <games_per_iter> --freeze`
      Start small (validate no crash), then scale toward ~1hr batches, same
      escalation discipline used the first time. Produces a NEW
      `shared_stack_frozen.pt` with the full 5-deck vocab.
- [ ] Confirm the new frozen `vocab_size` matches `build_pool()`'s vocab.size.

### 5. Re-run league training over all 5 decks
- [ ] `python run_league.py --n-iterations N --games-per-iteration 6
        --snapshot-every 15 --n-workers 6`
      Fresh league (no resumable `live.pt` yet -> all decks start from the new
      frozen stack). Start small, watch health, scale up.
- [ ] Sanity: each deck should register snapshots on schedule and produce
      finite losses; monster_tron's reward signal in particular should be
      non-trivial (if it never wins, revisit whether it belongs in the pool).

## Settled decisions carried in from benchmarking (already done, for context)
- ppo_update stays on **CPU**: the batch-size sweep found GPU ~4x slower at
  every batch size (128-4096) with no crossover -- the model is too small
  (d_model=64, 2 layers, ~200-250K params) for GPU throughput to beat its
  fixed per-minibatch overhead. `run_league.py --gpu-threshold` should default
  to disabled (always CPU); the batch-size *schedule* (small->large) is kept
  for its training-dynamics benefit, the device-switch half is not used.
- Rollout collection is parallelized across **6 worker processes** (~3.2-3.5x,
  plateaus at physical core count).
