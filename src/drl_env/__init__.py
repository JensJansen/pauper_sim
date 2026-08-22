"""Action-table builder and game-state helpers for the DRL policy: the
assembly logic between the game engine and the training loop.

build_action_table turns a decklist + game.EFFECT_REGISTRY into a flat
action table: each row has a label, a legal(state) predicate, and an
execute(state). Also exposes the per-seat helpers used by rl.training.train
(_for_player, _lost, _hand_count_available). Reward functions are injected
by that loop, not defined here.

Split by category across submodules (_actions_common, _actions_cast,
_actions_cast_altzone, _actions_combat, _actions_resolution, _actions_mana,
_actions_table) and re-exported flat here so `drl_env.X` works regardless
of which submodule defines X.
"""

from ._actions_common import *
from ._actions_cast import *
from ._actions_cast_altzone import *
from ._actions_combat import *
from ._actions_resolution import *
from ._actions_mana import *
from ._actions_table import *
from ._seat import _for_player, _lost
