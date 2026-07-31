"""Action-table builder and game-state helpers for the token/attention DRL
policy -- not an environment or a model, just the assembly logic between the
game engine and the training loop.

build_action_table turns a decklist + game.EFFECT_REGISTRY into a flat action
table the token pipeline drives: for each action, a human-readable label, a
legal(state) predicate, and an execute(state) that applies it to the engine.
A deck built entirely from already-implemented cards needs zero new code here.
Also provides the per-seat helpers the training loop (rl.train) reuses
(_for_player, _lost, _hand_count_available, ...). Reward functions (rl.rewards's
contract) are injected separately by that loop, never here.
"""

from ._actions import *
from ._seat import _for_player, _lost
