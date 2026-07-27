"""The priority stack: push, pop-and-resolve, and the one cast-time trigger
hook. Deliberately dumb -- these functions never inspect what `resolve` does,
so this module doesn't depend on casting.py (even though cast_aura calls
push_to_stack). triggers.py sits ABOVE it instead: _trigger_resolve's
"automatic" branch needs casting.enters_battlefield, which put here would
recreate a cycle.

References registry.EFFECT_REGISTRY only inside function bodies -- see
game/registry.py's docstring for why."""

from .. import registry
from ..cards import CardType


def on_cast_trigger(state, card_def):
    """A "whenever you cast an instant/sorcery" ability (Guttersnipe). Real
    Magic 603.3: a triggered ability doesn't take effect when it triggers -- it
    goes on the stack at the next priority point, ABOVE the triggering spell,
    with a response window. So this only QUEUES it (state.trigger_queue);
    game.turn's priority round promotes it once the cast is fully done (never
    mid-cast, or it'd land under a spell not yet pushed). Every cast path
    (normal / alt_cast / Flashback / Plot) calls this identically."""
    if card_def.card_type not in (CardType.INSTANT, CardType.SORCERY):
        return
    for permanent in state.battlefield:
        trigger = registry.EFFECT_REGISTRY.get(permanent.card_def.effect_id, {}).get("on_cast")
        if trigger is not None:
            state.log_event(
                "trigger_fired", source=(permanent.card_def.name, permanent.slot), trigger_kind="on_cast",
                triggering_card=card_def.name,
            )
            state.trigger_queue.append({"type": "cast_trigger", "card_def": permanent.card_def, "permanent": permanent})


def push_to_stack(state, card_def, resolve, reserves_hand_card=True, is_spell=True):
    """Defer `resolve(state, card_def)` onto state.stack instead of running it
    now, giving both players a priority window before it resolves. Pushed only
    once the spell's cost is fully paid -- never mid-payment (an alt cost that
    is itself a resolution, e.g. Fireblast's sacrifice-2-Mountains, pushes from
    that resolution's own on_complete).

    A pushed card stays physically in its origin zone until `resolve` moves it
    (resolve does its own hand/graveyard/exile removal, not here) -- so a paid-
    but-unresolved card is still "in hand." Two places must NOT count it as
    available: drl_env._hand_count_available (cast legality) and
    resolution.discard_options (instant-speed discard) -- both subtract
    same-named stack entries still reserving a hand card.

    reserves_hand_card=False when the card is NOT awaiting removal from the
    caster's hand: Flashback/reanimate (already out of the graveyard),
    Plot/Adventure/Madness cast-from-exile (already out of exile), an alt cost
    that discards eagerly (Fireblast, Crop Rotation), or a promoted trigger.
    (Confirmed live: two same-named Madness cards discarded back-to-back --
    default True miscounted the first's promoted entry as reserving the second,
    still-in-hand copy, leaving zero legal actions.)

    Records active_idx as the entry's controller: a priority round can flip
    active_idx before this resolves, but resolve must run against the CASTER's
    zones (state.py's active_idx proxy) -- resolve_top_of_stack restores it."""
    state.stack.append({
        "card_def": card_def, "resolve": resolve, "controller": state.active_idx,
        "reserves_hand_card": reserves_hand_card, "is_spell": is_spell,
    })
    state.log_event("zone_move", card=card_def.name, to_zone="stack", controller=state.active_idx)


def push_ability_to_stack(state, source_card_def, effect):
    """Put a NON-MANA ability's effect on the stack (real Magic 605: activated/
    triggered abilities use the stack + a priority window; only mana abilities
    skip it). `source_card_def` labels the entry; `effect(state)` runs on
    resolution. Costs and targets are already paid/chosen before this push.
    reserves_hand_card=False (source is on the battlefield, not a hand card);
    is_spell=False (an ability isn't a legal Counterspell/Dispel/Spell Pierce
    target)."""
    push_to_stack(state, source_card_def, lambda st, cd: effect(st), reserves_hand_card=False, is_spell=False)


def counter_spell(state, entry):
    """Counter a spell: remove its entry from the stack so it never resolves,
    and send its card to the right zone. A normally-cast spell
    (reserves_hand_card -- the card is still physically in its controller's
    hand while on the stack, see push_to_stack) goes to that controller's
    graveyard (real Magic: a countered spell goes to its owner's graveyard).
    A spell cast from elsewhere (flashback/madness/plot/free-alt,
    reserves_hand_card=False) has already left its origin zone at cast time,
    so there's nothing to move -- it simply ceases (a flashback spell would
    be exiled either way; a self-discarding spell like Crop Rotation is
    already in the graveyard, which is exactly where a countered spell goes).
    No-op if the entry has already left the stack (already countered/resolved
    in response). Returns True iff it actually countered something."""
    if entry not in state.stack:
        return False
    state.stack.remove(entry)
    cd = entry["card_def"]
    if entry.get("reserves_hand_card"):
        controller = state.players[entry["controller"]]
        if cd in controller.hand:
            controller.hand.remove(cd)
        controller.graveyard.append(cd)
    state.log_event("countered", card=cd.name, controller=entry["controller"])
    return True


def resolve_top_of_stack(state):
    """Pop and resolve the most recently pushed spell -- LIFO, no
    reordering action needed (real Magic's own stack order). Called once
    per "Pass" while state.stack is non-empty (game.turn._run_turn_gen),
    never automatically -- the model must explicitly let it happen instead
    of casting something else in response.

    Restores active_idx to this entry's own controller (push_to_stack)
    before resolving: by the time all players have passed in a row,
    active_idx may be sitting on whoever passed last, not the original
    caster -- resolve must run from the
    controller's own zone perspective regardless."""
    entry = state.stack.pop()
    state.active_idx = entry["controller"]
    state.log_event("zone_move", card=entry["card_def"].name, from_zone="stack", reason="resolve")
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

    # on_cast_trigger: only QUEUES a trigger (faithful timing -- the ability
    # goes on the stack at the next priority point, not inline), for
    # INSTANT/SORCERY casts, only for permanents whose registry entry
    # actually has an "on_cast" hook. The effect fires only once that queued
    # trigger is promoted to the stack and resolved.
    calls = []
    _filler_backup = registry.EFFECT_REGISTRY[EffectId.FILLER]
    registry.EFFECT_REGISTRY[EffectId.FILLER] = {"on_cast": lambda s, p: calls.append(p.card_def.name)}
    try:
        from ..state import Permanent
        from .triggers import promote_triggers_to_stack
        state3 = GameState(on_the_play=True)
        state3.battlefield = [Permanent(CardDef("Guttersnipe-like", CardType.CREATURE, None, EffectId.FILLER))]
        on_cast_trigger(state3, CardDef("A Sorcery", CardType.SORCERY, {}, None))
        assert calls == []  # not fired inline -- only queued
        assert [e["type"] for e in state3.trigger_queue] == ["cast_trigger"]
        promote_triggers_to_stack(state3)  # game.turn's priority round does this at a priority point
        resolve_top_of_stack(state3)  # a "Pass" resolves it in real play
        assert calls == ["Guttersnipe-like"]  # effect fires only on resolution
        on_cast_trigger(state3, CardDef("A Land", CardType.LAND, None, None))
        assert state3.trigger_queue == [] and calls == ["Guttersnipe-like"]  # lands don't trigger on-cast hooks
    finally:
        registry.EFFECT_REGISTRY[EffectId.FILLER] = _filler_backup

    print("stack.py self-check: OK")
