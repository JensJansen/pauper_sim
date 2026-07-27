"""Pending resolution: a decision that takes more than one action to resolve
-- paying a cost one tap at a time, walking a scry/surveil, choosing a search
target -- because the MODEL, not an automatic solver, makes each choice.

Was a single resolution.py; now a package split into _core (the
begin/complete state machine) and handlers (every concrete resolution kind),
both re-exported flat here so every `from game.resolution import X` still works."""

from ._core import *
from .handlers import *
# underscore-prefixed internals `import *` skips, but the old single-module
# resolution.py exposed them as attributes (e.g. resolution._remove_one_from_exile,
# reached from game.effects.madness_and_plot) -- re-export to match that namespace.
from ._core import _loggable
from .handlers import (
    _ANY_COLORS,
    _available_hand_names,
    _discard_one,
    _finish_assign_combat_damage,
    _finish_put_on_top,
    _finish_scry_surveil,
    _finish_throne,
    _remove_one_from_exile,
)
