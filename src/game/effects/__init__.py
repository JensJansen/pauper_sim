"""Generic effect plumbing every color catalog's cast/activate functions
call into, split by responsibility: stats.py (Aura/keyword/power/toughness),
combat.py, state_based.py (SBA + cleanup), win_check.py, casting.py
(battlefield entry + Aura casting), tokens.py, stack.py + triggers.py (the
priority stack and trigger queue), madness_and_plot.py (mana+resolution
bridge for those two mechanics). No per-card entries live here -- those are
in game/catalog/.

Nothing is re-exported: callers import directly from the owning submodule.
"""
