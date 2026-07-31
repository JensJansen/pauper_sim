"""Migrated from game/effects/stack.py's __main__ ponytail self-check."""
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
    # reserved-hand-card lifecycle (regression: the reserved-stack-card
    # double-cast bug). A normally-cast spell must LEAVE hand at cast -- else it
    # stays castable off the stack and a second copy's resolve finds it already
    # gone (a real crash, Lorien Revealed ~turn 39 in league training). Cast it,
    # confirm it left hand AT CAST, then resolve: resolve_top_of_stack
    # transiently restores it so the unchanged resolve moves it to graveyard.
    state_r = GameState(on_the_play=True)
    spell = CardDef("Reserved Spell", CardType.INSTANT, {}, EffectId.FILLER)
    state_r.hand.append(spell)
    seen = {}

    def _spell_resolve(s, c):
        seen["in_hand_during_resolve"] = c in s.hand  # MUST be False -- never back in hand
        s.move_card(c, s.graveyard)  # off hand since cast: stack -> graveyard (fresh instance)

    push_to_stack(state_r, spell, _spell_resolve)
    assert spell not in state_r.hand, "a cast spell must leave hand AT CAST (else it stays re-castable off the stack)"
    assert len(state_r.stack) == 1 and state_r.stack[0]["card_def"] is spell
    resolve_top_of_stack(state_r)
    assert seen["in_hand_during_resolve"] is False, "a resolving spell must NEVER be in hand during its own resolution"
    # ends in graveyard as a FRESH instance (move_card), not the hand CardDef itself.
    assert spell not in state_r.hand and [g.name for g in state_r.graveyard] == ["Reserved Spell"], "a resolved spell ends in graveyard, not hand"


def test_countered_reserved_hand_card_goes_to_graveyard_once():
    # A countered reserved-hand-card spell (already off hand at cast) still goes
    # to its controller's graveyard exactly once, never stranded in hand.
    state_c = GameState(on_the_play=True)
    spell_c = CardDef("Countered Spell", CardType.INSTANT, {}, EffectId.FILLER)
    state_c.hand.append(spell_c)
    push_to_stack(state_c, spell_c, lambda s, c: None)
    assert spell_c not in state_c.hand
    assert counter_spell(state_c, state_c.stack[0]) is True
    assert [g.name for g in state_c.graveyard] == ["Countered Spell"] and spell_c not in state_c.hand and state_c.stack == []


def test_controller_restored_at_resolution():
    # controller restoration: pushed while active_idx=1, resolved while
    # active_idx has since moved to 0 -- resolve must still see active_idx=1.
    state2 = GameState(on_the_play=True, players=[PlayerState(True), PlayerState(False)])
    state2.active_idx = 1
    seen_active_idx = []
    card_def = CardDef("Fake Spell", CardType.SORCERY, {}, EffectId.FILLER)
    push_to_stack(state2, card_def, lambda s, c: seen_active_idx.append(s.active_idx))
    state2.active_idx = 0
    resolve_top_of_stack(state2)
    assert seen_active_idx == [1] and state2.active_idx == 1


def test_on_cast_trigger_queues_and_fires_only_on_resolution():
    # on_cast_trigger: only QUEUES a trigger (faithful timing -- the ability
    # goes on the stack at the next priority point, not inline), for
    # INSTANT/SORCERY casts, only for permanents whose registry entry
    # actually has an "on_cast" hook. The effect fires only once that queued
    # trigger is promoted to the stack and resolved.
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
    # Regression (ability-logged-as-a-card-on-the-stack bug): an activated/
    # triggered ability going on -- or resolving off -- the stack moves no card,
    # so it must emit NO card zone_move. Emitting one made the replay converter
    # mint a phantom library copy per activation (Makeshift Munitions,
    # Krark-Clan Shaman, ...), inflating the shown deck size; and marking such an
    # ability is_spell=True would also make it a wrongly-legal Counterspell
    # target. A real spell still logs hand->stack then a stack resolution.
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
