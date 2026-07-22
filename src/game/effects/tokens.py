"""Token creation, the two token-specific activated abilities, and the
token CardDefs themselves -- Melded Moxite's Robot, Voldaren Epicure/
Vampire's Kiss's Blood, Cartouche of Solidarity's Warrior, Malevolent
Rumble's Eldrazi Spawn (docs/MADNESS_DECKS_PLAN.md item 8). Builds on
casting.enters_battlefield for the actual battlefield-entry mechanics --
a token entering is exactly as real as any other permanent from there on;
only its creation (no hand/library removal beforehand) differs."""

from . import casting
from .. import registry, resolution
from ..cards import CardDef, CardType, EffectId

TOKEN_LIMIT = 20  # shared across every token name, not per-name -- see docs/COMBAT_PLAN.md


def create_token(state, card_def, tapped=False):
    """A token permanent, not backed by any library/CARD_DEFS card.
    Reuses casting.enters_battlefield's full battlefield-entry path (ETB
    dispatch, terminated_fn check) unchanged -- only its creation (no
    hand/library removal beforehand) is different. tapped=True covers
    "Create a TAPPED 2/2 Robot" (Melded Moxite's own wording); Blood
    tokens enter untapped.

    TOKEN_LIMIT caps how many tokens (any name, combined -- an Eldrazi
    Spawn and a Warrior count the same toward this one shared pool) this
    player can have on the battlefield at once (docs/COMBAT_PLAN.md's
    permanent-identity discussion: no per-card token-production math,
    just one flat, generous ceiling). Beyond it, creation fails outright
    -- returns None, never touches the battlefield, never fires an ETB
    trigger, as if it was never attempted at all. No real deck comes
    remotely close today; this exists for whatever degenerate future
    token engine might."""
    token_count = sum(1 for p in state.battlefield if p.card_def.name not in registry.CARD_DEFS)
    if token_count >= TOKEN_LIMIT:
        return None
    return casting.enters_battlefield(state, card_def, force_tapped=tapped)


def activate_blood_sac(state, permanent):
    """Blood's {1}, {T}, Discard a card, Sacrifice this token: Draw a
    card. The {1} mana and the untapped precondition are both already
    handled generically by drl_env.py's cost_key-based activated-ability
    wiring (same as Candy Trail's own sac ability, which has no {T}
    symbol at all yet gets the identical untapped check for free today)
    -- this only covers what's specific to Blood: sacrifice (a token
    ceases to exist once it leaves the battlefield, per real Magic's own
    state-based action -- never added to the graveyard, unlike a real
    card), then discard a card (reusing resolution.begin_discard
    directly, which is what makes Madness-awareness automatic for
    whatever gets discarded this way), then draw."""
    state.battlefield.remove(permanent)
    resolution.begin_discard(state, 1, optional=False, on_complete=lambda s, _cards: s.draw(1))


def activate_eldrazi_spawn_sac(state, permanent):
    """Malevolent Rumble's Eldrazi Spawn token: "Sacrifice this creature:
    Add {C}." No {T} in the real cost -- unlike every other mana source
    in this engine, this doesn't tap (so summoning sickness never gates
    it) and isn't offered through mana.py's tap-based machinery at all.
    Modeled as a standalone no-mana-cost activated ability (same shape
    Quirion Ranger's Forest-bounce already uses) whose only effect is
    floating {C} directly into the mana pool -- reusing state.mana_pool's
    existing "produced now, spent later via a separate action" mechanism
    unchanged, since a sacrifice isn't a tap this engine's interactive
    pay_cost loop has any other way to represent."""
    state.battlefield.remove(permanent)
    state.mana_pool["C"] = state.mana_pool.get("C", 0) + 1


BLOOD_TOKEN_CARD_DEF = CardDef("Blood", CardType.ARTIFACT, None, EffectId.BLOOD_TOKEN, sac_ability_cost={"generic": 1})
ROBOT_TOKEN_CARD_DEF = CardDef("Robot", CardType.CREATURE, None, EffectId.ROBOT_TOKEN, power=2, toughness=2)  # 2/2
WARRIOR_TOKEN_CARD_DEF = CardDef("Warrior", CardType.CREATURE, None, EffectId.WARRIOR_TOKEN, power=1, toughness=1)  # 1/1; vigilance -- see EffectId.WARRIOR_TOKEN's own registry entry (white_cards.py)
ELDRAZI_SPAWN_TOKEN_CARD_DEF = CardDef("Eldrazi Spawn", CardType.CREATURE, None, EffectId.ELDRAZI_SPAWN_TOKEN, power=0, toughness=1)  # 0/1


if __name__ == "__main__":
    # ponytail self-check: run via `python -m game.effects.tokens` from
    # src/. create_token + Blood's sac ability + TOKEN_LIMIT enforcement
    # (including the "fails outright, no ETB fires" guarantee).
    from ..state import GameState, Permanent

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
        activate_blood_sac(state, blood)  # cost payment ({1} mana) is drl_env.py's concern, not this function's
        assert state.pending_resolution["kind"] == "discard"
        assert resolution.discard_options(state) == ["Fake Madness Card"]
        resolution.execute_discard_option(state, "Fake Madness Card")

        # Sacrificed: gone, never added to any zone (a token ceases to
        # exist, unlike a real card being discarded/sacrificed).
        assert [p.card_def.name for p in state.battlefield] == ["Robot"]
        assert state.graveyard == []
        assert len(state.trigger_queue) == 1 and state.trigger_queue[0]["kind"] == "madness"
        # Draw fires via begin_discard's own on_complete regardless of the
        # queued trigger -- net hand size unchanged (lost the discarded
        # card, gained one drawn).
        assert len(state.hand) == drawn_before
    finally:
        registry.EFFECT_REGISTRY[EffectId.FILLER] = _filler_backup

    print("tokens.py create_token + Blood-sac self-check: OK")

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
    registry.EFFECT_REGISTRY[EffectId.FILLER] = {"etb_trigger": lambda s: etb_calls.append(True)}
    try:
        fake_token = CardDef("Fake Token", CardType.CREATURE, None, EffectId.FILLER)
        result = create_token(state, fake_token)
        assert result is None
        assert etb_calls == []
    finally:
        registry.EFFECT_REGISTRY[EffectId.FILLER] = _filler_backup

    print("tokens.py TOKEN_LIMIT self-check: OK")
