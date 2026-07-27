"""Token creation, the token-specific activated abilities (Blood/Food/Clue/Map
sac abilities, Eldrazi Spawn's sac-for-{C}), and the token CardDefs themselves.
Builds on casting.enters_battlefield -- a token entering is as real as any
permanent; only its creation (no prior hand/library removal) differs."""

from . import casting
from .stack import push_ability_to_stack
from .win_check import gain_life
from .. import registry, resolution
from ..cards import CardDef, CardType, EffectId

TOKEN_LIMIT = 20  # shared across every token name, not per-name


def create_token(state, card_def, tapped=False):
    """A token permanent, not backed by any library/CARD_DEFS card.
    Reuses casting.enters_battlefield's full battlefield-entry path (ETB
    dispatch, end-of-game check) unchanged -- only its creation (no
    hand/library removal beforehand) is different. tapped=True covers
    "Create a TAPPED 2/2 Robot" (Melded Moxite's own wording); Blood
    tokens enter untapped.

    TOKEN_LIMIT caps how many tokens (any name, combined -- an Eldrazi
    Spawn and a Warrior count the same toward this one shared pool) this
    player can have on the battlefield at once. Beyond it, creation fails outright
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
    whatever gets discarded this way), then draw.

    Faithful timing: the sacrifice and the discard
    are both COSTS, paid immediately on activation; only the DRAW is the
    ability's effect, so it goes on the stack (push_ability_to_stack) once
    both costs are paid -- both players get a priority window before it
    resolves, and a same-name copy still discarded to pay this cost is
    reserved correctly (push_ability_to_stack passes reserves_hand_card=
    False; the discard already left hand at cost time)."""
    state.battlefield.remove(permanent)
    state.log_event(
        "zone_move", permanent=(permanent.card_def.name, permanent.slot), from_zone="battlefield",
        to_zone="ceases_to_exist", reason="sacrifice",
    )
    resolution.begin_discard(
        state, 1, optional=False,
        on_complete=lambda s, _cards: push_ability_to_stack(s, permanent.card_def, lambda st: st.draw(1)),
    )


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
    state.log_event(
        "zone_move", permanent=(permanent.card_def.name, permanent.slot), from_zone="battlefield",
        to_zone="ceases_to_exist", reason="sacrifice",
    )
    state.mana_pool["C"] = state.mana_pool.get("C", 0) + 1
    state.log_event("mana_tap", permanent=(permanent.card_def.name, permanent.slot), mode="sac_ability", produced=["C"])
    from .shared import fire_sacrifice_triggers
    fire_sacrifice_triggers(state, state.active_idx, permanent.card_def)  # Writhing Chrysalis: "whenever you sacrifice another Eldrazi"


def activate_clue_sac(state, permanent):
    """Clue's "{2}, Sacrifice this artifact: Draw a card" (Investigate makes
    one -- Toxin Analysis). The {2} mana + untapped precondition come from
    drl_env's cost_key wiring (same as Food/Blood); real Clue has no {T} in
    its cost, but a Clue is never tapped anyway, so the generic untapped
    check is harmless. Sacrifice a TOKEN (ceases to exist, never a graveyard
    trip), then the draw is the effect and goes on the stack, resolving after
    a priority window."""
    state.battlefield.remove(permanent)
    state.log_event(
        "zone_move", permanent=(permanent.card_def.name, permanent.slot), from_zone="battlefield",
        to_zone="ceases_to_exist", reason="sacrifice",
    )
    push_ability_to_stack(state, permanent.card_def, lambda st: st.draw(1))


def activate_food_sac(state, permanent):
    """Food's "{2}, {T}, Sacrifice this token: You gain 3 life" (Generous
    Ent's ETB makes one). The {2} mana and the untapped precondition are
    handled generically by drl_env.py's cost_key-based activated-ability
    wiring (same as Candy Trail's own sac ability). Sacrifice a TOKEN (ceases
    to exist, never a graveyard trip -- unlike Candy Trail, a real card), then
    the gain-3 is the effect and goes on the stack (push_ability_to_stack),
    resolving after a priority window."""
    state.battlefield.remove(permanent)
    state.log_event(
        "zone_move", permanent=(permanent.card_def.name, permanent.slot), from_zone="battlefield",
        to_zone="ceases_to_exist", reason="sacrifice",
    )
    push_ability_to_stack(state, permanent.card_def, lambda st: gain_life(st, 3))


BLOOD_TOKEN_CARD_DEF = CardDef("Blood", CardType.ARTIFACT, None, EffectId.BLOOD_TOKEN, sac_ability_cost={"generic": 1})
ROBOT_TOKEN_CARD_DEF = CardDef("Robot", CardType.CREATURE, None, EffectId.ROBOT_TOKEN, power=2, toughness=2)  # 2/2
WARRIOR_TOKEN_CARD_DEF = CardDef("Warrior", CardType.CREATURE, None, EffectId.WARRIOR_TOKEN, power=1, toughness=1)  # 1/1; vigilance -- see EffectId.WARRIOR_TOKEN's own registry entry (white_cards.py)
ELDRAZI_SPAWN_TOKEN_CARD_DEF = CardDef("Eldrazi Spawn", CardType.CREATURE, None, EffectId.ELDRAZI_SPAWN_TOKEN, power=0, toughness=1, subtypes=("Eldrazi",))  # 0/1; Eldrazi -- for Writhing Chrysalis
FOOD_TOKEN_CARD_DEF = CardDef("Food", CardType.ARTIFACT, None, EffectId.FOOD_TOKEN, sac_ability_cost={"generic": 2})  # {2},{T},Sac: gain 3
CLUE_TOKEN_CARD_DEF = CardDef("Clue", CardType.ARTIFACT, None, EffectId.CLUE_TOKEN, sac_ability_cost={"generic": 2})  # {2},Sac: draw a card (Investigate)
# --- G6 tokens ---
# 1/1 white Human Soldier (Rally at the Hornburg). Vanilla -- no registry
# entry; the Human/Soldier subtypes are what "Humans you control" reads.
HUMAN_SOLDIER_TOKEN_CARD_DEF = CardDef("Human Soldier", CardType.CREATURE, None, EffectId.HUMAN_SOLDIER_TOKEN, power=1, toughness=1, subtypes=("Human", "Soldier"))
# Treasure: "{T}, Sacrifice this artifact: Add one mana of any color." Same
# consumed-on-tap flexible mana source as Lotus Petal (registry + on_tap in
# colorless_cards.py).
TREASURE_TOKEN_CARD_DEF = CardDef("Treasure", CardType.ARTIFACT, None, EffectId.TREASURE_TOKEN)
# 1/1 blue Bird Illusion with flying (Murmuring Mystic). Registry (flying) in blue_cards.py.
BIRD_ILLUSION_TOKEN_CARD_DEF = CardDef("Bird Illusion", CardType.CREATURE, None, EffectId.BIRD_ILLUSION_TOKEN, power=1, toughness=1, subtypes=("Bird", "Illusion"))
# --- G8 tokens ---
# 2/2 white Samurai with vigilance (Experimental Synthesizer). Registry (vigilance) in white_cards.py.
SAMURAI_TOKEN_CARD_DEF = CardDef("Samurai", CardType.CREATURE, None, EffectId.SAMURAI_TOKEN, power=2, toughness=2, subtypes=("Samurai",))
# Map (Fanatical Offering): "{1}, {T}, Sacrifice this token: Target creature
# you control explores. Activate only as a sorcery." Registry in colorless_cards.py.
MAP_TOKEN_CARD_DEF = CardDef("Map", CardType.ARTIFACT, None, EffectId.MAP_TOKEN, ability_cost={"generic": 1})
# --- G12 tokens ---
# 4/1 black Skeleton with menace (Undercity Catacombs room). Registry (menace
# keyword) in green_cards.py, co-located with the rest of the initiative subsystem.
SKELETON_TOKEN_CARD_DEF = CardDef("Skeleton", CardType.CREATURE, None, EffectId.SKELETON_TOKEN, power=4, toughness=1, subtypes=("Skeleton",))


def activate_map_sac(state, permanent):
    """{1}, {T}, Sacrifice this token: a target creature you control explores.
    The {1} + untapped precondition come from the cost_key wiring; sacrifice
    the Map (a token -- ceases) as a cost, choose the target creature at
    activation, and the explore waits on the stack (a priority window), then
    resolves -- fizzling if that creature has since left."""
    state.battlefield.remove(permanent)
    state.log_event(
        "zone_move", permanent=(permanent.card_def.name, permanent.slot), from_zone="battlefield",
        to_zone="ceases_to_exist", reason="sacrifice",
    )

    def _on_chosen(state, choice):
        if choice is None:
            return
        name, slot = choice
        captured = next((p for p in state.battlefield if p.card_def.name == name and p.slot == slot), None)

        def _effect(st):
            if captured is not None and any(captured is p for p in st.battlefield):
                resolution.explore(st, captured)

        push_ability_to_stack(state, permanent.card_def, _effect)

    resolution.begin_choose_permanent(state, lambda p: p.card_type == CardType.CREATURE, _on_chosen)


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
        # The DRAW is the ability's effect -- now on the stack (faithful
        # timing), not fired inline the instant the costs were paid. Nothing
        # drawn until it resolves off the stack.
        from .stack import resolve_top_of_stack
        assert len(state.stack) == 1 and len(state.hand) == 0
        resolve_top_of_stack(state)
        # Draw fired on resolution -- net hand size unchanged vs. before the
        # activation (lost the discarded card, gained one drawn).
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
    registry.EFFECT_REGISTRY[EffectId.FILLER] = {"etb_trigger": lambda s, permanent: etb_calls.append(True)}
    try:
        fake_token = CardDef("Fake Token", CardType.CREATURE, None, EffectId.FILLER)
        result = create_token(state, fake_token)
        assert result is None
        assert etb_calls == []
    finally:
        registry.EFFECT_REGISTRY[EffectId.FILLER] = _filler_backup

    print("tokens.py TOKEN_LIMIT self-check: OK")
