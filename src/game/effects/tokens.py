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
    token_count = sum(1 for p in state.battlefield if is_token(p.card_def.name))
    if token_count >= TOKEN_LIMIT:
        return None
    return casting.enters_battlefield(state, card_def, force_tapped=tapped)


def activate_blood_sac(state, permanent):
    """Blood's {1}, {T}, Discard a card, Sacrifice this token: Draw a
    card. The {1} mana and the untapped precondition are both already
    handled generically by drl_env's cost_key-based activated-ability
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
    sacrifice_to_graveyard(state, permanent)  # ceases to exist; queues the dies-trigger (Gixian Infiltrator)
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
    pay_cost loop has any other way to represent.

    taggable=False: this token has no registry "mana" spec (its production
    is this bespoke function, not registry-driven), so mana.
    discount_departing_source's generic sacrifice-time lookup can't find it
    the way it finds a land. The float and the sacrifice are the same atomic
    action anyway -- there's no "tap now, sacrifice later" window -- so it's
    forced untagged at the source instead, same as a mana filter's output
    pip (mana.float_mana's own taggable=False case)."""
    # Mana is floated (and logged) BEFORE the sacrifice so it's on the pool
    # ahead of the dies-trigger's queueing -- sacrifice_to_graveyard bundles
    # the zone-move log and the trigger fire into one call, with no seam to
    # insert this between them the way the old hand-rolled body could.
    float_mana(state, ["C"], taggable=False)
    state.log_event("mana_tap", permanent=(permanent.card_def.name, permanent.slot), mode="sac_ability", produced=["C"])
    sacrifice_to_graveyard(state, permanent)  # ceases to exist; queues the dies-trigger (Writhing Chrysalis: "whenever you sacrifice another Eldrazi")


def activate_clue_sac(state, permanent):
    """Clue's "{2}, Sacrifice this artifact: Draw a card" (Investigate makes
    one -- Toxin Analysis). The {2} mana + untapped precondition come from
    drl_env's cost_key wiring (same as Food/Blood); real Clue has no {T} in
    its cost, but a Clue is never tapped anyway, so the generic untapped
    check is harmless. Sacrifice a TOKEN (ceases to exist, never a graveyard
    trip), then the draw is the effect and goes on the stack, resolving after
    a priority window."""
    sacrifice_to_graveyard(state, permanent)  # ceases to exist; queues the dies-trigger (Gixian Infiltrator)
    push_ability_to_stack(state, permanent.card_def, lambda st: st.draw(1))


def activate_food_sac(state, permanent):
    """Food's "{2}, {T}, Sacrifice this token: You gain 3 life" (Generous
    Ent's ETB makes one). The {2} mana and the untapped precondition are
    handled generically by drl_env's cost_key-based activated-ability
    wiring (same as Candy Trail's own sac ability). Sacrifice a TOKEN (ceases
    to exist, never a graveyard trip -- unlike Candy Trail, a real card), then
    the gain-3 is the effect and goes on the stack (push_ability_to_stack),
    resolving after a priority window."""
    sacrifice_to_graveyard(state, permanent)  # ceases to exist; queues the dies-trigger (Gixian Infiltrator)
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
    sacrifice_to_graveyard(state, permanent)  # ceases to exist; queues the dies-trigger (Gixian Infiltrator)

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
