"""Red-identity card catalog: every card whose real mana cost is
mono-red (or, for lands with no cost, whose only mana output is red).
Every card's cost/type/oracle-text below is a direct Scryfall pull,
except creature power/toughness, which is a design choice, not Scryfall
data. Breath Weapon (Tron filler, real cost {2}{R}) files here rather
than colorless_cards.py -- verified via Scryfall, not guessed."""

from .. import resolution
from ..cards import CardDef, CardType, EffectId
from ..effects_common import (
    BLOOD_TOKEN_CARD_DEF,
    ROBOT_TOKEN_CARD_DEF,
    cast_permanent_from_hand,
    create_token,
    deal_damage_to_opponent,
    discard_from_hand_to_graveyard,
    plot_to_exile,
    push_to_stack,
)

RED_CARD_CATALOG = {
    "Mountain": CardDef("Mountain", CardType.LAND, None, EffectId.MOUNTAIN),
    "Voldaren Epicure": CardDef(
        "Voldaren Epicure", CardType.CREATURE, {"R": 1}, EffectId.VOLDAREN_EPICURE, power=1, toughness=1,
    ),
    "Lightning Bolt": CardDef("Lightning Bolt", CardType.INSTANT, {"R": 1}, EffectId.LIGHTNING_BOLT),
    "Fiery Temper": CardDef("Fiery Temper", CardType.INSTANT, {"generic": 1, "R": 2}, EffectId.FIERY_TEMPER),
    "Faithless Looting": CardDef("Faithless Looting", CardType.SORCERY, {"R": 1}, EffectId.FAITHLESS_LOOTING),
    "Highway Robbery": CardDef("Highway Robbery", CardType.SORCERY, {"generic": 1, "R": 1}, EffectId.HIGHWAY_ROBBERY),
    "Grab the Prize": CardDef("Grab the Prize", CardType.SORCERY, {"generic": 1, "R": 1}, EffectId.GRAB_THE_PRIZE),
    "Melded Moxite": CardDef(
        "Melded Moxite", CardType.ARTIFACT, {"generic": 1, "R": 1}, EffectId.MELDED_MOXITE,
        sac_ability_cost={"generic": 3},
    ),
    "Fireblast": CardDef("Fireblast", CardType.INSTANT, {"generic": 4, "R": 2}, EffectId.FIREBLAST),
    # power was previously 0 (an unexplained placeholder from before combat
    # was real) -- corrected to Guttersnipe's real printed 2/2
    # (docs/COMBAT_PLAN.md's full-stats pass).
    "Guttersnipe": CardDef(
        "Guttersnipe", CardType.CREATURE, {"generic": 2, "R": 1}, EffectId.GUTTERSNIPE, power=2, toughness=2,
    ),
    "Lava Dart": CardDef("Lava Dart", CardType.INSTANT, {"R": 1}, EffectId.LAVA_DART),
    "End the Festivities": CardDef("End the Festivities", CardType.SORCERY, {"R": 1}, EffectId.END_THE_FESTIVITIES),
    # Real cast cost is irrelevant here: filler is never cast (Tron deck bulk).
    "Breath Weapon": CardDef("Breath Weapon", CardType.FILLER, None, EffectId.FILLER),
}


def voldaren_epicure_etb(state):
    """Oracle: "When this creature enters, it deals 1 damage to each
    opponent. Create a Blood token." """
    deal_damage_to_opponent(state, 1)
    create_token(state, BLOOD_TOKEN_CARD_DEF)


def cast_lightning_bolt(state, card_def):
    """{R}: deals 3 damage to any target -- targeting is simplified to
    "the opponent" (every other burn effect's own precedent; see
    deal_damage_to_opponent)."""
    discard_from_hand_to_graveyard(state, card_def)
    deal_damage_to_opponent(state, 3)


def _fiery_temper_damage(state):
    deal_damage_to_opponent(state, 3)


def cast_fiery_temper(state, card_def):
    discard_from_hand_to_graveyard(state, card_def)
    _fiery_temper_damage(state)


def madness_fiery_temper(state, card_def):
    """Madness resolve: by the time this runs, execute_madness_cast has
    already pulled the card out of exile -- never touch hand here (it
    isn't there), just the effect, then to the graveyard like any
    resolved spell."""
    state.graveyard.append(card_def)
    _fiery_temper_damage(state)


def faithless_looting_discard(state):
    """Draw two, then discard two -- shared by the normal cast and
    Flashback below (identical effect, only how the cost was paid
    differs)."""
    state.draw(2)
    resolution.begin_discard(state, 2, optional=False, on_complete=lambda s, _cards: None)


def cast_faithless_looting(state, card_def):
    discard_from_hand_to_graveyard(state, card_def)
    faithless_looting_discard(state)


def flashback_faithless_looting(state, card_def):
    """No alternate cost of its own (unlike Dread Return/Lava Dart's
    sacrifice) -- so, same as Land Grant's free alt_cast, the effect is
    already "fully paid for" the instant Flashback is chosen and pushes
    onto the stack immediately, not gated behind any further resolution."""
    state.graveyard.remove(card_def)  # leaves the graveyard the moment Flashback is chosen -- exiled after, untracked (Dread Return's own Flashback precedent)
    push_to_stack(state, card_def, lambda st, cd: faithless_looting_discard(st))


def _highway_robbery_effect(state):
    """Oracle: "You may discard a card or sacrifice a land. If you do,
    draw two cards." Simplified to the discard half only -- real card
    also allows sacrificing a land as the alternative optional cost;
    dropped rather than modeling a per-cast choice of cost type on top of
    Plot. Genuinely optional (not an additional cost) -- unlike Grab the
    Prize, casting this never requires a card in hand at all."""
    resolution.begin_discard(state, 1, optional=True, on_complete=lambda s, cards: s.draw(2) if cards else None)


def cast_highway_robbery(state, card_def):
    discard_from_hand_to_graveyard(state, card_def)
    _highway_robbery_effect(state)


def cast_highway_robbery_from_exile(state, card_def):
    """Plot's cast-from-exile resolve. By the time this runs, the card
    already left exile, never hand -- unlike cast_highway_robbery above,
    this never touches state.hand."""
    state.graveyard.append(card_def)
    _highway_robbery_effect(state)


def _grab_the_prize_extra_legal(state):
    """As an additional cost, discard a card -- needs a card in hand
    besides the one being cast."""
    return len(state.hand) >= 2


def _grab_the_prize_effect(state, discarded_cards):
    """Oracle: "Draw two cards. If the discarded card wasn't a land card,
    Grab the Prize deals 2 damage to each opponent." discarded_cards is
    always exactly 1 card here (mandatory n=1 discard, guaranteed payable
    by extra_legal above).

    This discard is a real-rules additional cost, but -- unlike Fireblast/
    Lava Dart/Dread Return's sacrifice alt costs -- it happens after the
    spell's own mana cost is already paid via the normal begin_pay_cost
    path, so the whole cast_grab_the_prize call (discard included) is what
    gets pushed onto the stack as one deferred unit, not split further. No
    observable difference in this solitaire sim: nothing can respond to or
    depend on the timing of an in-hand discard choice."""
    state.draw(2)
    if discarded_cards and discarded_cards[0].card_type != CardType.LAND:
        deal_damage_to_opponent(state, 2)


def cast_grab_the_prize(state, card_def):
    discard_from_hand_to_graveyard(state, card_def)
    resolution.begin_discard(state, 1, optional=False, on_complete=lambda s, cards: _grab_the_prize_effect(s, cards))


def melded_moxite_etb(state):
    """When this artifact enters, you may discard a card. If you do, draw
    two cards."""
    resolution.begin_discard(state, 1, optional=True, on_complete=lambda s, cards: s.draw(2) if cards else None)


def activate_melded_moxite_sac(state, permanent):
    """{3}, Sacrifice this artifact: create a tapped 2/2 colorless Robot
    artifact creature token (the same shared ROBOT_TOKEN_CARD_DEF)."""
    state.battlefield.remove(permanent)
    create_token(state, ROBOT_TOKEN_CARD_DEF, tapped=True)


def guttersnipe_on_cast(state, permanent):
    """Whenever you cast an instant or sorcery spell, deals 2 damage to
    each opponent -- fires via the generic on_cast_trigger chokepoint,
    identically for every cast path (normal, Flashback, Madness, Plot)
    already wired through it."""
    deal_damage_to_opponent(state, 2)


def _fireblast_damage(state):
    deal_damage_to_opponent(state, 4)


def cast_fireblast(state, card_def):
    discard_from_hand_to_graveyard(state, card_def)
    _fireblast_damage(state)


def _fireblast_alt_extra_legal(state):
    return sum(1 for p in state.battlefield if p.card_def.name == "Mountain") >= 2


def cast_fireblast_alt(state, card_def):
    """You may sacrifice two Mountains rather than pay this spell's mana
    cost. Same effect as the hard-cast above, deferred onto the stack
    (push_to_stack) only once the sacrifice -- this alt cost -- is
    actually paid; the damage itself waits for the stack to resolve."""
    discard_from_hand_to_graveyard(state, card_def)
    resolution.begin_sacrifice(
        state, lambda p: p.card_def.name == "Mountain", 2,
        on_complete=lambda s, ok: push_to_stack(s, card_def, lambda st, cd: _fireblast_damage(st)),
    )


def _lava_dart_damage(state):
    deal_damage_to_opponent(state, 1)


def cast_lava_dart(state, card_def):
    discard_from_hand_to_graveyard(state, card_def)
    _lava_dart_damage(state)


def flashback_lava_dart(state, card_def):
    """Flashback -- Sacrifice a Mountain: no mana component at all, same
    shape as Dread Return's Flashback but a land instead of 3 creatures --
    same deferred-onto-the-stack treatment as Fireblast's alt cost above."""
    state.graveyard.remove(card_def)  # leaves the graveyard the moment Flashback is chosen -- exiled after, untracked (Dread Return's own Flashback precedent)
    resolution.begin_sacrifice(
        state, lambda p: p.card_def.name == "Mountain", 1,
        on_complete=lambda s, ok: push_to_stack(s, card_def, lambda st, cd: _lava_dart_damage(st)),
    )


def cast_end_the_festivities(state, card_def):
    """Deals 1 damage to each opponent and each creature and planeswalker
    they control -- no opposing board modeled, so just the 1 damage."""
    discard_from_hand_to_graveyard(state, card_def)
    deal_damage_to_opponent(state, 1)


RED_EFFECT_REGISTRY = {
    EffectId.MOUNTAIN: {
        "mana": ("fixed", "R"),
    },
    EffectId.VOLDAREN_EPICURE: {
        "cast": {"resolve": lambda state, card_def: cast_permanent_from_hand(state, card_def)},
        "etb_trigger": lambda state: voldaren_epicure_etb(state),
    },
    EffectId.LIGHTNING_BOLT: {
        "cast": {"resolve": lambda state, card_def: cast_lightning_bolt(state, card_def)},
    },
    EffectId.FIERY_TEMPER: {
        "cast": {"resolve": lambda state, card_def: cast_fiery_temper(state, card_def)},
        "madness": {"cost": {"R": 1}, "resolve": lambda state, card_def: madness_fiery_temper(state, card_def)},
        # order_triggers (docs/PRIORITY_PLAN.md item 1): reachable the
        # instant 2+ Madness cards get discarded at once -- Faithless
        # Looting's own discard-2, right below, is exactly that source.
        "pending_kinds": {"madness_decision", "order_triggers"},
    },
    EffectId.FAITHLESS_LOOTING: {
        "cast": {"resolve": lambda state, card_def: cast_faithless_looting(state, card_def)},
        "flashback": {
            "legal": lambda state: True,
            "resolve": lambda state, card_def: flashback_faithless_looting(state, card_def),
        },
        "pending_kinds": {"discard"},
    },
    # Real Highway Robbery also allows sacrificing a land instead of
    # discarding; simplified to discard-only here rather than modeling a
    # per-cast choice of cost type on top of Plot (see cast_highway_robbery).
    # Genuinely optional, no extra_legal gate -- always castable.
    EffectId.HIGHWAY_ROBBERY: {
        "cast": {"resolve": lambda state, card_def: cast_highway_robbery(state, card_def)},
        "plot": {
            "cost": {"generic": 1, "R": 1},
            "resolve": lambda state, card_def: plot_to_exile(state, card_def),
            "cast_from_exile_resolve": lambda state, card_def: cast_highway_robbery_from_exile(state, card_def),
        },
        "pending_kinds": {"discard"},
    },
    EffectId.GRAB_THE_PRIZE: {
        "cast": {
            "resolve": lambda state, card_def: cast_grab_the_prize(state, card_def),
            "extra_legal": lambda state: _grab_the_prize_extra_legal(state),
        },
        "pending_kinds": {"discard"},
    },
    EffectId.MELDED_MOXITE: {
        "cast": {"resolve": lambda state, card_def: cast_permanent_from_hand(state, card_def)},
        "etb_trigger": lambda state: melded_moxite_etb(state),
        "activated_abilities": {
            "sac": {
                "cost_key": "sac_ability_cost",
                "resolve": lambda state, permanent: activate_melded_moxite_sac(state, permanent),
            },
        },
        "pending_kinds": {"discard"},
    },
    EffectId.GUTTERSNIPE: {
        "cast": {"resolve": lambda state, card_def: cast_permanent_from_hand(state, card_def)},
        "on_cast": lambda state, permanent: guttersnipe_on_cast(state, permanent),
    },
    EffectId.FIREBLAST: {
        "cast": {"resolve": lambda state, card_def: cast_fireblast(state, card_def)},
        "alt_cast": {
            "extra_legal": lambda state: _fireblast_alt_extra_legal(state),
            "resolve": lambda state, card_def: cast_fireblast_alt(state, card_def),
        },
        "pending_kinds": {"sacrifice"},
    },
    EffectId.LAVA_DART: {
        "cast": {"resolve": lambda state, card_def: cast_lava_dart(state, card_def)},
        "flashback": {
            "legal": lambda state: any(p.card_def.name == "Mountain" for p in state.battlefield),
            "resolve": lambda state, card_def: flashback_lava_dart(state, card_def),
        },
        "pending_kinds": {"sacrifice"},
    },
    EffectId.END_THE_FESTIVITIES: {
        "cast": {"resolve": lambda state, card_def: cast_end_the_festivities(state, card_def)},
    },
    # Breath Weapon (filler): no entry needed -- EffectId.FILLER's single
    # canonical {} registry entry lives in colorless_cards.py; every
    # reader consults it via EFFECT_REGISTRY.get(effect_id, {}), which
    # already defaults missing keys to {} the same way.
}
