"""Pending resolution: a decision that takes more than one action to resolve
(paying a cost one tap at a time, walking a scry, choosing a search target)
because the model, not an automatic solver, makes each choice.

Split into _core (the begin/complete state machine) and one
handlers_<category> module per resolution kind, all re-exported flat here so
`from game.resolution import X` works regardless of submodule."""

from ._core import *
from .handlers_targeting import *
from .handlers_combat import *
from .handlers_casting import *
from .handlers_library import *
from .handlers_mulligan import *
from .handlers_triggers import *
# `import *` skips underscore-prefixed names; re-export these explicitly
# since callers reach them directly.
from ._core import _loggable
from .handlers_casting import _remove_one_from_exile
