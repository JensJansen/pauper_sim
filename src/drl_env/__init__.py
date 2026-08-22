"""Action-table builder and game-state helpers for the token/attention DRL
policy -- not an environment or a model, just the assembly logic between the
game engine and the training loop.

build_action_table turns a decklist + game.EFFECT_REGISTRY into a flat action
table the token pipeline drives: for each action, a human-readable label, a
legal(state) predicate, and an execute(state) that applies it to the engine.
A deck built entirely from already-implemented cards needs zero new code here.
Also provides the per-seat helpers the training loop (rl.training.train) reuses
(_for_player, _lost, _hand_count_available, ...). Reward functions (rl.rewards's
contract) are injected separately by that loop, never here.

The action-table machinery itself is split by category -- _actions_common
(shared sentinel + _hand_count_available), _actions_cast (play land / plain
cast, incl. modal/X-cost/Delve / activate / forestcycle / impulse),
_actions_cast_altzone (casting from a non-hand zone or non-default cost:
alt-cost/Flashback/Escape/Plot/Omen/Prototype -- split out of _actions_cast,
see its own module docstring), _actions_combat (attack/block/damage-assign),
_actions_resolution (generic pending-kind dispatch: Pass, Choose:, targeting,
every small universal decision row), _actions_mana (mana abilities/filters),
_actions_table (build_action_table + legal_action_mask, which touch every
category) -- all re-exported flat here so `drl_env.X` keeps working
regardless of which submodule X actually lives in.
"""

from ._actions_common import *
from ._actions_cast import *
from ._actions_cast_altzone import *
from ._actions_combat import *
from ._actions_resolution import *
from ._actions_mana import *
from ._actions_table import *
from ._seat import _for_player, _lost
