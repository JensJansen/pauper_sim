# Sac-cost fizzle crash: candidate causes (hypotheses only, nothing here has been run or changed)

The crash: `_sac_artifact_or_creature._on_chosen` ([black_cards.py:363](src/game/catalog/black_cards.py#L363)) receives
`choice=None` and dies on `name, slot = choice`. `begin_choose_permanent`
([handlers_targeting.py:18](src/game/resolution/handlers_targeting.py#L18)) completes with `None` when no permanent matches the
predicate, i.e. **at the moment the sacrifice is chosen, the caster controls no creature or artifact**, although the cast
was allowed to begin. Seen in production on 2026-08-27 (Insight), ~08-31 (Fanatical Offering), and 2026-11-10 (Insight, in
the L1 arm). The question is only: *how can a cast that passed `extra_legal` reach its sacrifice with no fodder?*

Ground rules for reading this: each hypothesis says why the code *might* allow it, what the evidence for/against is, and what
would confirm or refute it. "Rank" is my plausibility estimate after reading the code, not a finding.

## What the evidence actually is

`logs/sac_fizzle_diagnostics.jsonl` has 16 records, but **15 are noise**: identical single-player records (one player, two tapped
Swamps, empty hand). They are the unit test `test_sac_cost_fodder_dying_mid_payment_crashes_the_resolve`
([test_black_cards.py:757](tests/game/catalog/test_black_cards.py#L757)) writing the dump each time it deliberately provokes the
crash; that test pollutes the production log. (Correction to my earlier note, which counted 3 production dumps: there is **one**
real record, from the 2026-11-10 L1 crash. The dump only exists since 09-01, so the two earlier crashes left none.)

The one real record (caster = player 1, `jund_wildfire`, the only deck containing Eviscerator's Insight):

| field | value | reading |
|---|---|---|
| `active_idx` = `turn_player_idx` = 1, phase `UPKEEP` | the caster holds priority in their own upkeep | instant cast in upkeep; not an opponent-priority case |
| `casting_depth` = 1, `stack` = [] | exactly one cast in flight, nothing pushed | no depth leak, no earlier cast sitting on the stack |
| caster battlefield | 5 lands only (Swamp#1 tapped, Forest#1 tapped, Twisted Landscape, Swamp#2, Swamp#3 untapped) | **zero non-land permanents** at sacrifice time |
| caster `mana_pool` | `{G: 1}` | **a leftover pip after the {1}{B} cost was fully paid** |
| caster hand | `[Eviscerator's Insight]` | consistent: the card leaves hand only at `push_to_stack` |
| opponent | Nyxborn Hydra (4 counters, tapped), Priest of Titania | irrelevant to the caster's fodder (see H7) |

Two clues to explain together: **(1) no fodder** and **(2) a stray floating G**. Cost {1}{B} is two pips, two lands tapped
produce two pips (B from Swamp, G from Forest), and both would be spent: nothing should float. A leftover G means either a
third pip entered the pool (from a source no longer on the battlefield: nothing tapped), or one pip was already floating
before the cast. The deck's registry scan shows **no mana-producing non-land permanent** (Treasure/Lotus Petal/Wall of Roots do not exist in
`jund_wildfire`), which constrains H3/H4 below.

## Structural precondition (true for every hypothesis)

**H0, the legality check is a one-time snapshot, and the sacrifice is chosen last.** `_cast_legal`
([_actions_cast.py:62](src/drl_env/_actions_cast.py#L62)) tests `extra_legal` (`_has_sac_fodder`, black_cards.py:303) once, when the
cast is announced. `_precast_choice_execute` ([_actions_cast.py:104](src/drl_env/_actions_cast.py#L104)) then pays mana first and
only afterwards, in `_after_pay`, calls `cast_*` -> `begin_choose_permanent`. The additional cost (CR 601.2h pays costs as one
step, and a sacrificed permanent can't also have paid a mana ability) is therefore *not* reserved at announcement: anything
that removes the fodder between the two points breaks the cast. Nothing re-validates at `_after_pay`, and `_on_chosen` has no
`None` branch (its own contract, handlers_targeting.py:15, says it can fizzle to `None`). All of H1-H9 are ways to *remove* the
fodder in that window, or to enter it without fodder.

## Hypotheses

### H1 (rank: high). Fodder dies inside the window through a path the `casting_depth` guard does not cover
The 08-29 fix suppresses `check_state_based_actions`/`refizzle_if_now_targetless` only at the top of
`_run_priority_round_gen`'s loop ([turn.py:386](src/game/turn.py#L386)) while `casting_depth > 0`. Other callers exist
(`check_state_based_actions(` at combat.py:371/388, state_based.py:233, red_cards.py:107/281/348/538/550, colorless_cards.py:444/523,
green_cards.py:654, multicolor_cards.py:134). Most sit inside activations that require `pending_resolution is None`, so they
cannot run during `pay_cost`; but any of them reached *transitively* from a payment action (mana ability side effect, `float_mana`,
a `discount_departing_source` path) would still claim a creature. Against: the dump's `phase` is UPKEEP with no combat; no
payment action I read calls an SBA directly. **Test:** instrument `check_state_based_actions` to record
`(caller stack, casting_depth)` whenever `casting_depth > 0` and any permanent leaves the battlefield.

### H2 (rank: high). `casting_depth` is wrong at the moment it matters (premature decrement or lost increment)
`push_to_stack` decrements `casting_depth` for *every* push, including promoted triggers and ability pushes
([stack.py:65](src/game/effects/stack.py#L65)); `_run_priority_round_gen` also zeroes it at the top of every round
([turn.py:384](src/game/turn.py#L384)). If another push (a trigger that promotes while the cast is open, e.g. a trigger
queued by an earlier action and promoted when `pending_resolution` momentarily becomes `None` between `pay_cost` and the
`choose_permanent`) lowers the counter to 0 *before* the last pip, the loop's next iteration runs a full SBA/refizzle pass
mid-cast. The dump shows depth 1 at the failure, which does not rule this out (a later increment, or the counter being
re-raised, would mask it). A mid-cast SBA would also explain "no fodder" without any exotic card. **Test:** assert
`casting_depth > 0` on every loop iteration while a cast's `pay_cost`/follow-up pending is open; log each
increment/decrement with its caller.

### H3 (rank: medium-low for this deck). A mana ability consumes the only fodder during payment
`activate_mana_source` runs `on_tap` side effects: Treasure/Lotus Petal call `sacrifice_to_graveyard`
(colorless_cards.py:294/299). During `pay_cost`, tapping such a source is legal and nothing knows the pending sacrifice needs that
permanent (CR: one permanent cannot pay both costs). This gives exactly the "extra stray pip + no fodder" signature
(three pips produced, two spent, the sac source gone). Against: a registry scan of `jund_wildfire` finds no Treasure/Petal/Wall
source in the 60 cards, and Treasure is only created by Reckless Lackey (not in the deck). It remains the *signature-perfect*
mechanism, so it should be checked against the exact deck of the crashed game (the dump records no decklist for the opponent
side or any token created this game) and against Fanatical Offering/Reckoner's Bargain in any other deck. **Test:** grep every
deck for `Treasure`, `Lotus Petal`, `Wall of Roots`; replay with a pip-by-pip event log.

### H4 (rank: medium-low). Eldrazi Spawn sacrifice floats mana outside the payment window and pre-loads the pool
`activate_eldrazi_spawn_sac` ([tokens.py:50](src/game/effects/tokens.py#L50)) is an *activated ability* (not a mana ability),
legal at instant speed in any phase when nothing is pending, which bypasses the owner-approved "floating only in your own main
phase" restriction ([_actions_mana.py:46](src/drl_env/_actions_mana.py#L46)). Writhing Chrysalis (4 copies) makes the Spawn
tokens, which are the deck's main fodder. Sequence that fits both clues: Spawn A sacrificed for {C} (floats), the agent then
announces Insight with the remaining fodder, taps lands anyway, spends; the cast completes... but then the *remaining* fodder
should still exist, so this alone does not explain "no fodder". It would, however, explain the stray pip, and combined with
H1/H2 (a Chrysalis `on_sacrifice` trigger or an SBA wiping the last Spawn) it is a plausible full story. Also worth noting: this
is a rules-faithfulness question in its own right (floating mana outside main phase through the activated-ability path).

### H5 (rank: medium). The action executed was not legal in the state it ran in (stale or mismatched mask)
`_precast_choice_execute` and every other `execute` re-check nothing; they trust the mask. If the chosen action index came from
a mask built on a different state (batched decisions across interleaved games, an agent caching its mask across a game
transition, or a `SeatAgent` seat mix-up), the cast would start with no fodder at all. The "nothing re-validated" design makes a
single stale index fatal. Evidence for: the dump's *final* state is simply "cast with no fodder"; nothing needs to have
been removed. Against: the pool clue (G:1) is unexplained by it, and masks are rebuilt every decision. **Test:** assert
`legal(state)` of the chosen row immediately before `execute` in `agent.py:154`, and dump on mismatch.

### H6 (rank: medium-low). Predicate/accounting divergence between "has fodder" and "choose fodder"
`_has_sac_fodder` and the `choose_permanent` predicate are written twice (black_cards.py:304 and :373) with the same condition
(`CardType.CREATURE or is_artifact`). They agree today, but they read `card_type` at different times; anything that changes a
permanent's type (animation, type-changing effect, a DFC front/back distinction via `departing_card_def`) between the two would
split them. Against: no such effect is in this deck, and for the sole real record there is no non-land permanent at all, so a
predicate quirk cannot explain it. Low priority but cheap to fold into one function.

### H7 (rank: low). Wrong-player view (all state accessors are "the active player")
`state.battlefield`, `state.hand`, `state.mana_pool` are `_active_player_property` views (state.py:496). `active_idx` flips on
every pass and is restored after resolutions (turn.py:411-424). If `active_idx` changed between announce and sacrifice, the
sacrifice would look at the other player's board (which has creatures, so it would not fizzle) *or* the announce was checked
on the other board. In the one record `active_idx == turn_player_idx == 1` and the hand belongs to player 1, so the
announce was on the caster's board; a flip in the window would also have given a *non-empty* board here. So this explains neither clue.

### H8 (rank: low). A Chrysalis-style `on_sacrifice` or Gixian trigger reorders the flow
Tokens/triggers created by the first sacrifice are queued (`fire_sacrifice_triggers`) and promoted only when pending is
`None`. If the cast ever ran the sacrifice twice (the same `_after_pay` firing twice, or `cast_*` invoked once per copy of a
cast), the second pass would find the fodder gone. No double invocation was found in `_cast_execute`/`complete_resolution`
(on_complete runs once; `pending_resolution` is cleared first), so this needs a duplicated caller I haven't found.

### H9 (rank: low). Cached mana/availability state leaks between interleaved games in a worker
`_mana_ability_options_cache` and `game.mana`'s caches are module globals keyed by `state is`
([_actions_mana.py:27](src/drl_env/_actions_mana.py#L27)); `legal_action_mask` clears them, but `execute` functions run between
sweeps, and `begin_pay_cost` calls `can_pay(available_mana_units(state), ...)`. A stale per-state cache could mis-describe
which sources are available (stray pip, a payment plan that includes a source that has since left). It cannot by itself remove
fodder, but it could explain H3-like pips without a real source, and it is process-global in the workers.

### H10 (rank: low). Flashback variant
`flashback_eviscerators_insight` moves the card out of the graveyard *before* choosing the sacrifice
(black_cards.py:417), and its `legal` is the same snapshot check. Same structural hole (H0); not this crash (the dump's source
is the hard cast), but a latent twin that any fix should cover.

### H11 (rank: context only). Why it surfaced in the L1 arm and not S
Both arms run the same engine and the rollout code is identical; overlap changes only the policy weights the workers use (one
update stale). A crash bug that needs a specific play pattern (tapping a particular way, casting in upkeep, holding
the last fodder) is policy-dependent, so a different trajectory can hit it in 6 iterations or never in 167. Nothing about
`overlap_collect` can reach `_on_chosen`. (This is also why the retry-with-a-bumped-seed policy is a sound workaround.)

## Suggested confirmation plan (for after the A/B finishes; nothing done yet)

1. **Replay harness first.** Re-create the crash: run `jund_wildfire` vs the recorded opponent with the same seeds in a loop until
   it recurs, with a *richer* dump: the last ~50 `state.log_event` records, the announce-time board and pool, the acting seat,
   the chosen action name, and `casting_depth` at announce / at each pip. The current dump lacks all of that, which is why four
   occurrences are still unexplained.
2. **Instrument, don't guess:** (a) assert `legal(state)` right before `execute`; (b) log every `casting_depth` change with caller;
   (c) wrap `check_state_based_actions` to record any removal while a `pay_cost`/`choose_permanent` is open.
3. **Decide H0's fix separately from the cause:** reserve/verify the additional cost at `_after_pay` (re-check `extra_legal` and
   refuse the cast), or pay the sacrifice at announcement per CR 601.2h. This is an engine rules decision for the owner.
4. **Clean the evidence:** make the unit test not write to the production `logs/` file (or point the dump at a tmp path under
   pytest), so a real record is never buried in test output.

Most informative single experiment: step 2(c) plus the richer dump in step 1. It separates H1/H2 (an SBA or counter bug), H3/H4
(a mana path consumes the fodder), and H5 (nothing was removed; the cast should never have been legal) in one run.
