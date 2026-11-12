# Component: encoder

**Contract.** The observation the policy sees (`rl/model/features.py` tokenizer + `card_behavior.py` + `rl/model/arch.py`
`SetTransformer`, plus the scalar block in `rl/decision/agent.py`) must (1) reflect the true game state, (2) hide what
must stay hidden, (3) be invariant to everything that is not part of the position (token order, padding, slot numbers,
which chair you sit in), and (4) carry every fact a player needs to decide — including counts and relationships.

| File | What it establishes |
|---|---|
| `test_l0_contract.py` | shapes, finiteness, ranges (everything in [0,1] except the signed `eff_power`), one zone per token, hidden-information rules, boundary values for life and cost reduction |
| `test_l1_oracle.py` | every token and scalar of ~10k real decisions equals an **independent** re-derivation from raw engine state (`support/corpus.py`); plus a coverage gate on the corpus |
| `test_l2_hidden_symmetry.py` | metamorphic, on the live state: reordering hidden info changes nothing; the other seat's view is the mirror image; slot numbers are not an input |
| `test_l3_static_and_vocab.py` | every card's static vector vs a field-by-field derivation; the behavior block says what a card does; the verified-identical collisions; vocab stability |
| `test_l4_encoder_math.py` | permutation equivariance, padding (incl. garbage), batch-vs-single in train and eval, empty sides, all-padding rows, pooling masks, gradients — synthetic and on real states |
| `test_l5_sensitivity.py` | no dead inputs: every non-side feature column and the card index moves the output |
| `test_l6_identifiability.py` | hand-built boards: relevant facts change the encoding, irrelevant ones don't; seat symmetry |
| `test_l7_pointer_alignment.py` | token *i*'s row describes the object its handle points at; legal pointer counts equal the engine's; agent token/identity lists aligned |
| `test_l9_probes.py` | efficacy: a fresh encoder learns to decode board facts and to point at the strongest permanent (~3 min) |
| `test_l10_golden.py` | pins feature widths, every static vector, and the encoder's output on a fixed position (`ENCODER_UPDATE_GOLDEN=1` regenerates `golden.json` — owner sign-off) |
| `test_l11_counting.py` | n identical cards encode differently at every level (token copy counts, both summary halves, exact-count scalars) |
| `test_l12_links.py` | which spell targets which creature / which blocker blocks which attacker / which aura is on which host; canonical, order-independent slot numbering |
| `test_token_order_invariance.py` | the whole `DeckNetwork` is order- and padding-invariant on real decisions |

Equivalence of rollout and update, which depends on the encoder, lives in `tests/component/training/`.

**Cost.** ~3 minutes (corpus build ~100 s, cached under `tests/.cache/` keyed by a hash of the code it depends on).
Knobs: `ENCODER_CORPUS_GAMES`, `ENCODER_CORPUS_DECKS`, `ENCODER_CORPUS_NOCACHE`; probes: `ENCODER_PROBE_SEED/EPOCHS/ABLATE`.

**Planted bugs.** `mutations.py` — 48 of them. `python tests/support/mutation_sweep.py encoder` (~40 minutes; `--baseline`
first confirms the harness). Currently 48/48 caught.

**Known gaps and open items.**
- The corpus uses randomly initialised networks, so trained-agent state distributions are not exercised.
- The oracle reuses the engine's own power/toughness functions (it checks the encoder's plumbing, not the engine's stat rules).
- Probe accuracy for the pointer task is ~0.93 against ~0.96 with the new feature groups zeroed: a small, unexplained cost on a
  ~1,500-state probe; watch it in real training.
- Static vectors still collide for Llanowar/Fyndhorn Elves and Contaminated Aquifer/Ice Tunnel (identical as implemented).
