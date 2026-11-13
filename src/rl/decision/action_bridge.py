"""The fixed-table / pointer action bridge now lives in the engine submodule as drl_env.action_bridge.

Re-exported here so the policy stack's existing `rl.decision.action_bridge` imports keep working unchanged. Edit the
engine copy (engine/src/drl_env/action_bridge.py), never this file."""
from drl_env.action_bridge import *  # noqa: F403
from drl_env.action_bridge import pointer_identities  # noqa: F401  (re-exported explicitly: the contract test uses it)
