"""Generic effect plumbing every color catalog's own cast/activate
functions call into -- split by responsibility (see each submodule's own
docstring): stats.py (Aura/keyword/power/toughness), combat.py,
state_based.py (SBA + cleanup), win_check.py, casting.py (battlefield
entry + Aura casting), tokens.py, stack.py + triggers.py (the priority
stack and trigger queue), and madness_and_plot.py (the mana+resolution
bridge those two mechanics need). No per-card catalog entries live here --
every real card lives exactly once in its own color file under
game/catalog/ (DECK_REGISTRY_REFRESH_PLAN.md).

Nothing is re-exported here -- callers import directly from the owning
submodule (e.g. `from .effects.combat import combat_damage_step`), same as
game/__init__.py does. Formerly one 1573-line game/effects_common.py.
"""
