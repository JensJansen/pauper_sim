"""Migrated from game/effects/tokens.py's __main__ ponytail self-check."""
from game import registry, resolution
from game.cards import CardDef, CardType, EffectId
from game.effects.stack import resolve_top_of_stack
from game.effects.tokens import (
    BLOOD_TOKEN_CARD_DEF,
    ELDRAZI_SPAWN_TOKEN_CARD_DEF,
    ROBOT_TOKEN_CARD_DEF,
    WARRIOR_TOKEN_CARD_DEF,
    activate_blood_sac,
    create_token,
)
from game.state import GameState, Permanent


def test_create_token_and_blood_sac():
    state = GameState(on_the_play=True)
    create_token(state, ROBOT_TOKEN_CARD_DEF, tapped=True)
    create_token(state, BLOOD_TOKEN_CARD_DEF)  # untapped by default
    assert [(p.card_def.name, p.tapped) for p in state.battlefield] == [("Robot", True), ("Blood", False)]

    _filler_backup = registry.EFFECT_REGISTRY[EffectId.FILLER]
    registry.EFFECT_REGISTRY[EffectId.FILLER] = {"madness": {"cost": {}, "resolve": lambda s, c: None}}
    try:
        blood = next(p for p in state.battlefield if p.card_def.name == "Blood")
        other_card = CardDef("Fake Madness Card", CardType.SORCERY, {}, EffectId.FILLER)
        state.hand = [other_card]
        state.library = [CardDef("Library Card", CardType.SORCERY, {}, None)]

        drawn_before = len(state.hand)
        activate_blood_sac(state, blood)  # cost payment ({1} mana) is drl_env's concern, not this function's
        assert state.pending_resolution["kind"] == "discard"
        assert resolution.discard_options(state) == ["Fake Madness Card"]
        resolution.execute_discard_option(state, "Fake Madness Card")

        # Sacrificed: gone, never added to any zone (a token ceases to
        # exist, unlike a real card being discarded/sacrificed).
        assert [p.card_def.name for p in state.battlefield] == ["Robot"]
        assert state.graveyard == []
        assert len(state.trigger_queue) == 1 and state.trigger_queue[0]["kind"] == "madness"
        # The DRAW is the ability's effect -- now on the stack (faithful
        # timing), not fired inline the instant the costs were paid. Nothing
        # drawn until it resolves off the stack.
        assert len(state.stack) == 1 and len(state.hand) == 0
        resolve_top_of_stack(state)
        # Draw fired on resolution -- net hand size unchanged vs. before the
        # activation (lost the discarded card, gained one drawn).
        assert len(state.hand) == drawn_before
    finally:
        registry.EFFECT_REGISTRY[EffectId.FILLER] = _filler_backup


def test_token_limit_enforcement():
    # TOKEN_LIMIT: a shared pool across every token name, not per-name --
    # 19 Robots already in play leaves room for exactly 1 more token of
    # ANY kind (a Warrior here), then nothing at all, not even a
    # different name.
    state = GameState(on_the_play=True)
    state.battlefield = [Permanent(ROBOT_TOKEN_CARD_DEF) for _ in range(19)]
    warrior = create_token(state, WARRIOR_TOKEN_CARD_DEF)
    assert warrior is not None and warrior in state.battlefield
    assert len(state.battlefield) == 20
    overflow = create_token(state, ELDRAZI_SPAWN_TOKEN_CARD_DEF)
    assert overflow is None
    assert len(state.battlefield) == 20  # never added -- not even a phantom entry
    assert not any(p.card_def.name == "Eldrazi Spawn" for p in state.battlefield)

    # "Fails outright" also means no ETB trigger fires for the rejected token.
    etb_calls = []
    _filler_backup = registry.EFFECT_REGISTRY[EffectId.FILLER]
    registry.EFFECT_REGISTRY[EffectId.FILLER] = {"etb_trigger": lambda s, permanent: etb_calls.append(True)}
    try:
        fake_token = CardDef("Fake Token", CardType.CREATURE, None, EffectId.FILLER)
        result = create_token(state, fake_token)
        assert result is None
        assert etb_calls == []
    finally:
        registry.EFFECT_REGISTRY[EffectId.FILLER] = _filler_backup
