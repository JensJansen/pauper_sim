# Component: action space -- the policy repo's half

The action space itself (`drl_env`: the table, the legal-action mask, `action_bridge`) is tested in the **engine
submodule**: `engine/tests/component/action_space/` (structure, purity, an independent rules oracle, execution on forks,
mana and combat searches, hand-built regressions; planted-bug list in its `mutations.py`). Run it from `engine/`.

What stays here is what only makes sense with the trained policy and its checkpoints:

| File | What it establishes |
|---|---|
| `test_action_table_order.py` | pins labels and row structure of every league deck's table against `action_table_snapshot.json`, and the action-head size of the committed checkpoint. A change invalidates every saved checkpoint (a policy head has one logit per row), so regenerating the snapshot needs the owner's sign-off. Needs `data/` and `checkpoints/`, so it cannot live in the engine. |
| `test_engine_contract.py` | `rl.decision.action_bridge` is the engine's bridge (same objects); `drl_env.action_bridge.pointer_identities` (the engine's list of addressable objects) equals the identity tokens of `rl.model.features.build_token_set` on every decision of real games. If they diverge, a pointer the policy can see is not one the engine offers (or the reverse). |

Related, in other components: the encoder's pointer-alignment layer (`tests/component/encoder`, L7) proves token position
and identity line up; the Rooftop Percher `targeted_by_theirs` bit is `tests/unit/rl/model/test_token_targeting.py`.
