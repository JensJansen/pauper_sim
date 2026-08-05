"""Pending resolution: a decision that takes more than one action to resolve
-- paying a cost one tap at a time, walking a scry/surveil, choosing a search
target -- because the MODEL, not an automatic solver, makes each choice.

A package split into _core (the begin/complete state machine) and one
handlers_<category> module per resolution kind (targeting, combat, casting,
library, mulligan, triggers), all re-exported flat here so every
`from game.resolution import X` works regardless of which submodule X
actually lives in."""

from ._core import *
from .handlers_targeting import *
from .handlers_combat import *
from .handlers_casting import *
from .handlers_library import *
from .handlers_mulligan import *
from .handlers_triggers import *
# `import *` skips underscore-prefixed names, but game.effects.madness_and_plot
# reaches this one directly as resolution._remove_one_from_exile -- re-export
# it explicitly too.
from ._core import _loggable
from .handlers_casting import _remove_one_from_exile
