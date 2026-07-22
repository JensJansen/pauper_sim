"""The priority stack itself: push, pop-and-resolve, and the one cast-time
trigger hook. Deliberately dumb -- these three functions never inspect
what `resolve` does, so this module has no dependency on casting.py even
though casting.cast_aura is one of its callers. triggers.py (queued
triggers -> stack entries) sits ABOVE this module instead, since
_trigger_resolve's "automatic" branch needs casting.enters_battlefield --
putting that here would recreate a real cycle (casting needs push_to_stack
from here; here would need enters_battlefield from casting).

References game.registry.EFFECT_REGISTRY only from inside function
bodies, via `registry.EFFECT_REGISTRY` -- see game/registry.py's own
module docstring for why."""

from .. import registry
from ..cards import CardType


def on_cast_trigger(state, card_def):
    """Fires at cast time, before the spell's own resolve runs -- matches
    real Magic timing (the triggered ability goes on the stack above the
    spell it triggered off, so it happens first). Every cast path (normal
    cast, alt_cast, Flashback, and Plot's cast-from-exile) calls this
    identically, from inside drl_env.py's own per-path wrapper functions
    -- so a card like Guttersnipe ("whenever you cast an instant or
    sorcery...") fires the same regardless of which path cast the
    triggering spell. See docs/MADNESS_DECKS_PLAN.md item 11."""
    if card_def.card_type not in (CardType.INSTANT, CardType.SORCERY):
        return
    for permanent in state.battlefield:
        trigger = registry.EFFECT_REGISTRY.get(permanent.card_def.effect_id, {}).get("on_cast")
        if trigger is not None:
            trigger(state, permanent)


def push_to_stack(state, card_def, resolve):
    """A spell is fully paid for (mana or an alternate cost) but not yet
    resolved -- defer `resolve(state, card_def)` onto state.stack instead of
    calling it now, giving the model a chance to respond (cast another
    instant-speed spell) before it resolves. Every cast-like top-level
    action (normal cast, cast_modes, alt_cast, Flashback, Plot's cast-from-
    exile, Madness) pushes here once its own cost-payment is fully done --
    never before (a card whose alt cost is itself a resolution, e.g.
    Fireblast's sacrifice-2-Mountains, must push only from that
    resolution's own on_complete, not from inside the sacrifice itself).

    A pushed card's own hand/graveyard/exile removal still happens inside
    `resolve` itself (unchanged from before the stack existed), not here --
    so a card sitting on the stack, paid for but unresolved, is still
    physically present in whatever zone it came from until it actually
    resolves. Two places correct for that instead of treating it as
    "available": drl_env._hand_count_available (cast legality -- also
    already a non-issue for every Flashback/Plot/Madness path, which each
    remove from their own zone before ever reaching a resolution that
    could push) and resolution.discard_options (an instant-speed activated
    ability, e.g. Blood's sac-for-a-card, isn't blocked by a non-empty
    stack the way a sorcery-speed cast is, so it can still be offered a
    card that's actually already spoken for by an unresolved stack entry).

    Records state.active_idx as this entry's own controller (docs/
    PRIORITY_PLAN.md): a real priority round can flip active_idx through
    both players (whoever's currently deciding to act/pass) between now
    and whenever this entry actually resolves, but state.hand/graveyard/
    battlefield (state.py's own active_idx-proxy) must still resolve
    against the CASTER's zones, not whoever last happened to hold
    priority -- resolve_top_of_stack restores it below."""
    state.stack.append({"card_def": card_def, "resolve": resolve, "controller": state.active_idx})


def resolve_top_of_stack(state):
    """Pop and resolve the most recently pushed spell -- LIFO, no
    reordering action needed (real Magic's own stack order). Called once
    per "Pass" while state.stack is non-empty (game.turn._run_turn_gen),
    never automatically -- the model must explicitly let it happen instead
    of casting something else in response.

    Restores active_idx to this entry's own controller (push_to_stack)
    before resolving: by the time all players have passed in a row,
    active_idx may be sitting on whoever passed last, not the original
    caster (docs/PRIORITY_PLAN.md) -- resolve must run from the
    controller's own zone perspective regardless."""
    entry = state.stack.pop()
    state.active_idx = entry["controller"]
    entry["resolve"](state, entry["card_def"])


if __name__ == "__main__":
    # ponytail self-check: run via `python -m game.effects.stack` from src/.
    from ..cards import CardDef, CardType, EffectId
    from ..state import GameState

    state = GameState(on_the_play=True)
    resolved = []
    card_def = CardDef("Fake Spell", CardType.SORCERY, {}, EffectId.FILLER)
    push_to_stack(state, card_def, lambda s, c: resolved.append(c.name))
    assert len(state.stack) == 1 and resolved == []
    resolve_top_of_stack(state)
    assert state.stack == [] and resolved == ["Fake Spell"]

    # controller restoration: pushed while active_idx=1, resolved while
    # active_idx has since moved to 0 -- resolve must still see active_idx=1.
    from ..state import PlayerState
    state2 = GameState(on_the_play=True, players=[PlayerState(True), PlayerState(False)])
    state2.active_idx = 1
    seen_active_idx = []
    push_to_stack(state2, card_def, lambda s, c: seen_active_idx.append(s.active_idx))
    state2.active_idx = 0
    resolve_top_of_stack(state2)
    assert seen_active_idx == [1] and state2.active_idx == 1

    # on_cast_trigger: only fires for INSTANT/SORCERY, only for permanents
    # whose registry entry actually has an "on_cast" hook.
    calls = []
    _filler_backup = registry.EFFECT_REGISTRY[EffectId.FILLER]
    registry.EFFECT_REGISTRY[EffectId.FILLER] = {"on_cast": lambda s, p: calls.append(p.card_def.name)}
    try:
        from ..state import Permanent
        state3 = GameState(on_the_play=True)
        state3.battlefield = [Permanent(CardDef("Guttersnipe-like", CardType.CREATURE, None, EffectId.FILLER))]
        on_cast_trigger(state3, CardDef("A Sorcery", CardType.SORCERY, {}, None))
        assert calls == ["Guttersnipe-like"]
        on_cast_trigger(state3, CardDef("A Land", CardType.LAND, None, None))
        assert calls == ["Guttersnipe-like"]  # lands don't trigger on-cast hooks
    finally:
        registry.EFFECT_REGISTRY[EffectId.FILLER] = _filler_backup

    print("stack.py self-check: OK")
