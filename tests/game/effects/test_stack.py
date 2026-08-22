"""The priority stack: push, pop-and-resolve, and the one cast-time trigger
hook (stack.py)."""
from game import registry
from game.cards import CardDef, CardType, EffectId
from game.effects.stack import counter_spell, on_cast_trigger, push_ability_to_stack, push_to_stack, resolve_top_of_stack
from game.state import GameState, PlayerState, Permanent


def test_push_and_resolve_basic():
    state = GameState(on_the_play=True)
    resolved = []
    card_def = CardDef("Fake Spell", CardType.SORCERY, {}, EffectId.FILLER)
    push_to_stack(state, card_def, lambda s, c: resolved.append(c.name))
    assert len(state.stack) == 1 and resolved == []
    resolve_top_of_stack(state)
    assert state.stack == [] and resolved == ["Fake Spell"]


def test_reserved_hand_card_lifecycle():
    # a cast spell must leave hand at cast, stay out of hand during its own
    # resolution, then end up in graveyard as a fresh instance (move_card)
    state_r = GameState(on_the_play=True)
    spell = CardDef("Reserved Spell", CardType.INSTANT, {}, EffectId.FILLER)
    state_r.hand.append(spell)
    seen = {}

    def _spell_resolve(s, c):
        seen["in_hand_during_resolve"] = c in s.hand
        s.move_card(c, s.graveyard)

    push_to_stack(state_r, spell, _spell_resolve)
    assert spell not in state_r.hand, "a cast spell must leave hand AT CAST (else it stays re-castable off the stack)"
    assert len(state_r.stack) == 1 and state_r.stack[0]["card_def"] is spell
    resolve_top_of_stack(state_r)
    assert seen["in_hand_during_resolve"] is False, "a resolving spell must NEVER be in hand during its own resolution"
    assert spell not in state_r.hand and [g.name for g in state_r.graveyard] == ["Reserved Spell"], "a resolved spell ends in graveyard, not hand"


def test_countered_reserved_hand_card_goes_to_graveyard_once():
    state_c = GameState(on_the_play=True)
    spell_c = CardDef("Countered Spell", CardType.INSTANT, {}, EffectId.FILLER)
    state_c.hand.append(spell_c)
    push_to_stack(state_c, spell_c, lambda s, c: None)
    assert spell_c not in state_c.hand
    assert counter_spell(state_c, state_c.stack[0]) is True
    assert [g.name for g in state_c.graveyard] == ["Countered Spell"] and spell_c not in state_c.hand and state_c.stack == []


def test_controller_restored_at_resolution():
    # pushed while active_idx=1; resolve must still see active_idx=1 even
    # though active_idx has since moved to 0
    state2 = GameState(on_the_play=True, players=[PlayerState(True), PlayerState(False)])
    state2.active_idx = 1
    seen_active_idx = []
    card_def = CardDef("Fake Spell", CardType.SORCERY, {}, EffectId.FILLER)
    push_to_stack(state2, card_def, lambda s, c: seen_active_idx.append(s.active_idx))
    state2.active_idx = 0
    resolve_top_of_stack(state2)
    assert seen_active_idx == [1] and state2.active_idx == 1


def test_on_cast_trigger_queues_and_fires_only_on_resolution():
    # on_cast_trigger only queues a trigger for INSTANT/SORCERY casts, for
    # permanents whose registry entry has an "on_cast" hook; the effect
    # fires only once that trigger is promoted to the stack and resolved.
    from game.effects.triggers import promote_triggers_to_stack

    calls = []
    _filler_backup = registry.EFFECT_REGISTRY[EffectId.FILLER]
    registry.EFFECT_REGISTRY[EffectId.FILLER] = {"on_cast": lambda s, p: calls.append(p.card_def.name)}
    try:
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


def test_ability_on_stack_never_logged_as_a_card_zone_move():
    # an ability going on/off the stack moves no card, so it must emit no
    # zone_move event and must be marked is_spell=False; a real spell still
    # logs hand->stack then its stack resolution.
    log = []
    logged = GameState(on_the_play=True, event_log=log)
    enchantment = CardDef("An Enchantment", CardType.ENCHANTMENT, {}, EffectId.FILLER)
    push_ability_to_stack(logged, enchantment, lambda st: None)  # this enchantment's activated ability
    assert logged.stack[-1]["is_spell"] is False
    resolve_top_of_stack(logged)
    assert not any(e["kind"] == "zone_move" and e.get("card") == "An Enchantment" for e in log), \
        "an ability must not be logged as a card moving to/from the stack"
    spell = CardDef("A Spell", CardType.INSTANT, {}, EffectId.FILLER)
    logged.hand.append(spell)
    push_to_stack(logged, spell, lambda s, c: None)
    resolve_top_of_stack(logged)
    spell_moves = [e for e in log if e["kind"] == "zone_move" and e.get("card") == "A Spell"]
    assert any(e.get("from_zone") == "hand" and e.get("to_zone") == "stack" for e in spell_moves), \
        "a spell cast from hand must log hand->stack"
    assert any(e.get("from_zone") == "stack" and e.get("reason") == "resolve" for e in spell_moves), \
        "a spell must log its stack resolution"
