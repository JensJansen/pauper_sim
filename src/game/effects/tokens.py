"""Token creation, the token-specific activated abilities (Blood/Food/Clue/Map
sac abilities, Eldrazi Spawn's sac-for-{C}), and the token CardDefs themselves.
Builds on casting.enters_battlefield -- a token entering is as real as any
permanent; only its creation (no prior hand/library removal) differs."""

from . import casting
from .stack import push_ability_to_stack
from .state_based import is_token, sacrifice_to_graveyard
from .win_check import gain_life
from .. import resolution
from ..cards import CardDef, CardType, EffectId
from ..mana import float_mana

TOKEN_LIMIT = 20  # shared across every token name, not per-name


def create_token(state, card_def, tapped=False):
    """A token permanent, not backed by any library/CARD_DEFS card. Reuses
    casting.enters_battlefield's battlefield-entry path unchanged -- only
    creation (no hand/library removal) differs. tapped=True covers "Create
    a tapped 2/2 Robot" (Melded Moxite).

    TOKEN_LIMIT caps how many tokens (any name, combined) this player can
    have on the battlefield at once. Beyond it, creation fails outright --
    returns None, no battlefield touch, no ETB trigger."""
    token_count = sum(1 for p in state.battlefield if is_token(p.card_def.name))
    if token_count >= TOKEN_LIMIT:
        return None
    return casting.enters_battlefield(state, card_def, force_tapped=tapped)


def activate_blood_sac(state, permanent):
    """Blood's {1}, {T}, Discard a card, Sacrifice this token: Draw a
    card. The {1} mana and untapped precondition are handled generically
    by drl_env's cost_key-based activated-ability wiring; this only covers
    what's specific to Blood: sacrifice (a token, ceases to exist), then
    discard (reusing resolution.begin_discard, which makes Madness-
    awareness automatic), then draw.

    Sacrifice and discard are both costs, paid immediately; only the draw
    is the ability's effect, so it goes on the stack once both costs are
    paid."""
    sacrifice_to_graveyard(state, permanent)  # dies-trigger queued (Gixian Infiltrator)
    resolution.begin_discard(
        state, 1, optional=False,
        on_complete=lambda s, _cards: push_ability_to_stack(s, permanent.card_def, lambda st: st.draw(1)),
    )


def activate_eldrazi_spawn_sac(state, permanent):
    """Malevolent Rumble's Eldrazi Spawn token: "Sacrifice this creature:
    Add {C}." No {T} in the real cost, so summoning sickness never gates
    it and it isn't offered through mana.py's tap-based machinery. Modeled
    as a standalone no-mana-cost activated ability whose only effect is
    floating {C} directly into the mana pool.

    taggable=False: this token has no registry "mana" spec, so
    mana.discount_departing_source's sacrifice-time lookup can't find it
    the way it finds a land -- forced untagged at the source instead."""
    # Mana floated before the sacrifice so it's on the pool ahead of the
    # dies-trigger's queueing.
    float_mana(state, ["C"], taggable=False)
    state.log_event("mana_tap", permanent=(permanent.card_def.name, permanent.slot), mode="sac_ability", produced=["C"])
    sacrifice_to_graveyard(state, permanent)  # dies-trigger queued (Writhing Chrysalis)


def activate_clue_sac(state, permanent):
    """Clue's "{2}, Sacrifice this artifact: Draw a card" (Investigate makes
    one). The {2} mana + untapped precondition come from drl_env's cost_key
    wiring. Sacrifice a token (ceases to exist), then the draw is the
    effect and goes on the stack."""
    sacrifice_to_graveyard(state, permanent)  # dies-trigger queued (Gixian Infiltrator)
    push_ability_to_stack(state, permanent.card_def, lambda st: st.draw(1))


def activate_food_sac(state, permanent):
    """Food's "{2}, {T}, Sacrifice this token: You gain 3 life" (Generous
    Ent's ETB makes one). Sacrifice a token (ceases to exist), then the
    gain-3 is the effect and goes on the stack."""
    sacrifice_to_graveyard(state, permanent)  # dies-trigger queued (Gixian Infiltrator)
    push_ability_to_stack(state, permanent.card_def, lambda st: gain_life(st, 3))


BLOOD_TOKEN_CARD_DEF = CardDef("Blood", CardType.ARTIFACT, None, EffectId.BLOOD_TOKEN, sac_ability_cost={"generic": 1})
ROBOT_TOKEN_CARD_DEF = CardDef("Robot", CardType.CREATURE, None, EffectId.ROBOT_TOKEN, power=2, toughness=2)  # 2/2
WARRIOR_TOKEN_CARD_DEF = CardDef("Warrior", CardType.CREATURE, None, EffectId.WARRIOR_TOKEN, power=1, toughness=1)  # 1/1; vigilance -- see EffectId.WARRIOR_TOKEN's own registry entry (white_cards.py)
ELDRAZI_SPAWN_TOKEN_CARD_DEF = CardDef("Eldrazi Spawn", CardType.CREATURE, None, EffectId.ELDRAZI_SPAWN_TOKEN, power=0, toughness=1, subtypes=("Eldrazi",))  # 0/1; Eldrazi -- for Writhing Chrysalis
FOOD_TOKEN_CARD_DEF = CardDef("Food", CardType.ARTIFACT, None, EffectId.FOOD_TOKEN, sac_ability_cost={"generic": 2})  # {2},{T},Sac: gain 3
CLUE_TOKEN_CARD_DEF = CardDef("Clue", CardType.ARTIFACT, None, EffectId.CLUE_TOKEN, sac_ability_cost={"generic": 2})  # {2},Sac: draw a card (Investigate)
# --- G6 tokens ---
HUMAN_SOLDIER_TOKEN_CARD_DEF = CardDef("Human Soldier", CardType.CREATURE, None, EffectId.HUMAN_SOLDIER_TOKEN, power=1, toughness=1, subtypes=("Human", "Soldier"))  # Rally at the Hornburg, vanilla
TREASURE_TOKEN_CARD_DEF = CardDef("Treasure", CardType.ARTIFACT, None, EffectId.TREASURE_TOKEN)  # {T},Sac: add one mana of any color
BIRD_ILLUSION_TOKEN_CARD_DEF = CardDef("Bird Illusion", CardType.CREATURE, None, EffectId.BIRD_ILLUSION_TOKEN, power=1, toughness=1, subtypes=("Bird", "Illusion"))  # flying, Murmuring Mystic
# --- G8 tokens ---
SAMURAI_TOKEN_CARD_DEF = CardDef("Samurai", CardType.CREATURE, None, EffectId.SAMURAI_TOKEN, power=2, toughness=2, subtypes=("Samurai",))  # vigilance, Experimental Synthesizer
MAP_TOKEN_CARD_DEF = CardDef("Map", CardType.ARTIFACT, None, EffectId.MAP_TOKEN, ability_cost={"generic": 1})  # {1},{T},Sac: target creature explores
# --- G12 tokens ---
SKELETON_TOKEN_CARD_DEF = CardDef("Skeleton", CardType.CREATURE, None, EffectId.SKELETON_TOKEN, power=4, toughness=1, subtypes=("Skeleton",))  # menace, Undercity Catacombs


def activate_map_sac(state, permanent):
    """{1}, {T}, Sacrifice this token: a target creature you control
    explores. Sacrifice the Map as a cost, choose the target at activation,
    and the explore waits on the stack, fizzling if the target has left."""
    sacrifice_to_graveyard(state, permanent)  # dies-trigger queued (Gixian Infiltrator)

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
