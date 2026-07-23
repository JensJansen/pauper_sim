"""Black-identity card catalog: every card whose real mana cost is
mono-black (or, for lands with no cost, whose only mana output is black).
Every card's cost/type/oracle-text below is a direct Scryfall pull,
except creature power/toughness, which is a design choice, not Scryfall
data. Real Jagged Barrens/End the Festivities/Vampire's Kiss/Voldaren
Epicure/Alms of the Vein reference "each opponent"/"target opponent" --
all of these route through win_check.deal_damage_to_opponent, which
hits a real per-player life_total in a 2-player game and the historical
state.damage_dealt counter otherwise (docs/MULTIPLAYER_ENGINE_PLAN.md)."""

from .. import resolution
from ..cards import CardDef, CardType, EffectId
from ..effects.casting import cast_permanent_from_hand, enters_battlefield
from ..effects.shared import discard_from_hand_to_graveyard
from ..effects.stack import push_to_stack
from ..effects.tokens import BLOOD_TOKEN_CARD_DEF, create_token
from ..effects.win_check import deal_damage_to_opponent

BLACK_CARD_CATALOG = {
    "Swamp": CardDef("Swamp", CardType.LAND, None, EffectId.SWAMP),
    "Bojuka Bog": CardDef("Bojuka Bog", CardType.LAND, None, EffectId.BOJUKA_BOG),
    "Balustrade Spy": CardDef(
        "Balustrade Spy", CardType.CREATURE, {"generic": 3, "B": 1}, EffectId.BALUSTRADE_SPY, power=2, toughness=2,
    ),
    "Lotleth Giant": CardDef(
        "Lotleth Giant", CardType.CREATURE, {"generic": 6, "B": 1}, EffectId.LOTLETH_GIANT, power=5, toughness=5,
    ),
    # No ability -- vanilla 1/1 for {1}{B} (opponent hand-disruption isn't
    # modeled; see design discussion).
    "Mesmeric Fiend": CardDef(
        "Mesmeric Fiend", CardType.CREATURE, {"generic": 1, "B": 1}, EffectId.MESMERIC_FIEND, power=1, toughness=1,
    ),
    "Dread Return": CardDef("Dread Return", CardType.SORCERY, {"generic": 2, "B": 2}, EffectId.DREAD_RETURN),
    "Kitchen Imp": CardDef(
        "Kitchen Imp", CardType.CREATURE, {"generic": 3, "B": 1}, EffectId.KITCHEN_IMP, power=2, toughness=2,
    ),
    "Vampire's Kiss": CardDef("Vampire's Kiss", CardType.SORCERY, {"generic": 1, "B": 1}, EffectId.VAMPIRES_KISS),
    "Alms of the Vein": CardDef("Alms of the Vein", CardType.SORCERY, {"generic": 2, "B": 1}, EffectId.ALMS_OF_THE_VEIN),
}


def mill_until_land(state):
    """Balustrade Spy's ETB: reveal from the top until a land card, milling
    everything revealed (including the land) to the graveyard. No model
    choice, so a plain loop, not a pending resolution. If the library
    empties before a land turns up, everything left mills and the library
    simply ends up empty -- this deck's own combo enabler. draw() (not
    this function) is what detects and flags actually running out, on
    whatever later draw attempts to pull from the now-empty library."""
    while state.library:
        card = state.library.pop(0)
        state.graveyard.append(card)
        if card.card_type == CardType.LAND:
            break


def lotleth_giant_etb(state):
    """Undergrowth ETB: 1 damage to the opponent per creature card in your
    graveyard."""
    creature_count = sum(1 for c in state.graveyard if c.card_type == CardType.CREATURE)
    deal_damage_to_opponent(state, creature_count)


def cast_dread_return(state, card_def):
    """{2}{B}{B}: return target creature card from your graveyard to the
    battlefield. This card is already in the graveyard by the time the
    reanimation choice begins (below), so -- being a sorcery, not a
    creature card -- it's correctly never offered as its own target."""
    discard_from_hand_to_graveyard(state, card_def)

    def _on_chosen(state, name):
        if name is None:
            return
        found = next(c for c in state.graveyard if c.name == name)
        state.graveyard.remove(found)
        enters_battlefield(state, found)

    resolution.begin_choose_graveyard_card(state, lambda c: c.card_type == CardType.CREATURE, _on_chosen)


def flashback_dread_return(state, card_def):
    """Flashback -- Sacrifice three creatures: cast from the graveyard
    instead of paying {2}{B}{B}. Same reanimation effect as the hard-cast
    above, deferred onto the stack (push_to_stack) only once the
    sacrifice -- Flashback's alternate cost -- is actually paid; the
    reanimation itself doesn't happen until something resolves the stack.
    Newly-sacrificed creatures land in the graveyard before the
    reanimation choice begins, so they're correctly eligible targets for
    this same casting -- a real rules interaction, not a bug. The card
    itself never returns to the graveyard afterward (exiled, per its own
    text) -- reusing the existing "exile is untracked" precedent (Relic of
    Progenitus) rather than adding a real exile zone."""
    state.graveyard.remove(card_def)  # leaves the graveyard the moment Flashback is chosen, same as any other cast

    def _reanimate(state, card_def):
        def _on_chosen(state, name):
            if name is None:
                return
            found = next(c for c in state.graveyard if c.name == name)
            state.graveyard.remove(found)
            enters_battlefield(state, found)

        resolution.begin_choose_graveyard_card(state, lambda c: c.card_type == CardType.CREATURE, _on_chosen)

    def _on_sacrificed(state, ok):
        if not ok:
            return  # the environment's own Flashback legality check guarantees this can't happen
        push_to_stack(state, card_def, _reanimate, reserves_hand_card=False)

    resolution.begin_sacrifice(state, lambda p: p.card_def.card_type == CardType.CREATURE, 3, _on_sacrificed)


def madness_kitchen_imp(state, card_def):
    """Kitchen Imp -- Flying, haste. Madness {B}. No ETB at all (real
    Oracle text has no triggered ability beyond Madness itself). Madness
    resolve for a creature: execute_madness_cast has already pulled the
    card out of exile, so this just needs the normal battlefield-entry
    path -- never touches hand, unlike a normal cast."""
    enters_battlefield(state, card_def)


def cast_vampires_kiss(state, card_def):
    """Target player loses 2 life and you gain 2 life. Create two Blood
    tokens. No Madness on this one (only Fiery Temper/Alms of the Vein
    have it)."""
    discard_from_hand_to_graveyard(state, card_def)
    deal_damage_to_opponent(state, 2)
    create_token(state, BLOOD_TOKEN_CARD_DEF)
    create_token(state, BLOOD_TOKEN_CARD_DEF)


def _alms_of_the_vein_damage(state):
    deal_damage_to_opponent(state, 3)


def cast_alms_of_the_vein(state, card_def):
    """Target opponent loses 3 life and you gain 3 life. Madness {B}."""
    discard_from_hand_to_graveyard(state, card_def)
    _alms_of_the_vein_damage(state)


def madness_alms_of_the_vein(state, card_def):
    state.graveyard.append(card_def)
    _alms_of_the_vein_damage(state)


BLACK_EFFECT_REGISTRY = {
    EffectId.SWAMP: {
        "mana": ("fixed", "B"),
    },
    EffectId.BOJUKA_BOG: {
        "mana": ("fixed", "B"),
        "enters_tapped": True,
    },
    EffectId.BALUSTRADE_SPY: {
        "cast": {"resolve": lambda state, card_def: cast_permanent_from_hand(state, card_def)},
        "etb_trigger": lambda state: mill_until_land(state),
    },
    EffectId.LOTLETH_GIANT: {
        "cast": {"resolve": lambda state, card_def: cast_permanent_from_hand(state, card_def)},
        "etb_trigger": lambda state: lotleth_giant_etb(state),
    },
    EffectId.MESMERIC_FIEND: {
        "cast": {"resolve": lambda state, card_def: cast_permanent_from_hand(state, card_def)},
    },
    EffectId.DREAD_RETURN: {
        "cast": {
            "resolve": lambda state, card_def: cast_dread_return(state, card_def),
            "extra_legal": lambda state: any(c.card_type == CardType.CREATURE for c in state.graveyard),
        },
        "flashback": {
            "legal": lambda state: sum(1 for p in state.battlefield if p.card_def.card_type == CardType.CREATURE) >= 3,
            "resolve": lambda state, card_def: flashback_dread_return(state, card_def),
        },
        "pending_kinds": {"choose_graveyard_card", "sacrifice"},
    },
    EffectId.KITCHEN_IMP: {
        # Real text: Flying, haste (docs/COMBAT_PLAN.md step 7 -- flying
        # means it can only be blocked by a creature that itself has
        # flying, drl_env._assign_blocker_execute's own extra_predicate).
        "cast": {"resolve": lambda state, card_def: cast_permanent_from_hand(state, card_def)},
        "haste": True,
        "keywords": {"flying"},
        "madness": {"cost": {"B": 1}, "resolve": lambda state, card_def: madness_kitchen_imp(state, card_def)},
        # order_triggers (docs/PRIORITY_PLAN.md item 1): reachable the
        # instant 2+ Madness cards get discarded at once (Faithless
        # Looting's discard-2) -- both trigger simultaneously and need a
        # real placement-order choice.
        "pending_kinds": {"madness_decision", "order_triggers"},
    },
    EffectId.VAMPIRES_KISS: {
        "cast": {"resolve": lambda state, card_def: cast_vampires_kiss(state, card_def)},
    },
    EffectId.ALMS_OF_THE_VEIN: {
        "cast": {"resolve": lambda state, card_def: cast_alms_of_the_vein(state, card_def)},
        "madness": {"cost": {"B": 1}, "resolve": lambda state, card_def: madness_alms_of_the_vein(state, card_def)},
        "pending_kinds": {"madness_decision", "order_triggers"},  # see EffectId.KITCHEN_IMP's own comment
    },
}
