"""Red-identity card catalog: every card whose real mana cost is
mono-red (or, for lands with no cost, whose only mana output is red).
Every card's cost/type/oracle-text below is a direct Scryfall pull,
except creature power/toughness, which is a design choice, not Scryfall
data. Breath Weapon (Tron filler, real cost {2}{R}) files here rather
than colorless_cards.py -- verified via Scryfall, not guessed; its real
"non-Dragon" filter is dropped in cast_breath_weapon below (no card in
this entire catalog is ever a Dragon -- a checked invariant, not a
guess), so it's implemented as a real, symmetric "2 damage to every
creature in play" board wipe, this deck's own creatures included."""

from .. import resolution
from ..cards import CardDef, CardType, EffectId
from ..effects.casting import cast_permanent_from_hand
from ..effects.madness_and_plot import plot_to_exile
from ..effects.shared import discard_from_hand_to_graveyard
from ..effects.stack import push_to_stack
from ..effects.state_based import check_state_based_actions
from ..effects.tokens import BLOOD_TOKEN_CARD_DEF, ROBOT_TOKEN_CARD_DEF, create_token
from ..effects.win_check import deal_damage_to_opponent

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
    "Breath Weapon": CardDef("Breath Weapon", CardType.INSTANT, {"generic": 2, "R": 1}, EffectId.BREATH_WEAPON),
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
    push_to_stack(state, card_def, lambda st, cd: faithless_looting_discard(st), reserves_hand_card=False)


def _highway_robbery_effect(state):
    """Oracle: "You may discard a card or sacrifice a land. If you do,
    draw two cards." Both cost options offered as one optional decision
    (resolution.begin_discard_or_sacrifice) -- genuinely optional (not an
    additional cost, unlike Grab the Prize), so casting this never
    requires a card in hand OR a land in play. Shared unchanged by both
    the normal cast and Plot's cast-from-exile below: real Plot lets you
    cast the card later "as you could normally cast it," which means this
    same may-discard-or-sacrifice choice is made fresh at THAT time too,
    not locked in when it was plotted."""
    resolution.begin_discard_or_sacrifice(
        state, lambda p: p.card_def.card_type == CardType.LAND,
        on_complete=lambda s, paid: s.draw(2) if paid else None,
    )


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
        on_complete=lambda s, ok: push_to_stack(s, card_def, lambda st, cd: _fireblast_damage(st), reserves_hand_card=False),
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
        on_complete=lambda s, ok: push_to_stack(s, card_def, lambda st, cd: _lava_dart_damage(st), reserves_hand_card=False),
    )


def cast_end_the_festivities(state, card_def):
    """Deals 1 damage to each opponent and each creature and planeswalker
    they control -- no opposing board modeled, so just the 1 damage."""
    discard_from_hand_to_graveyard(state, card_def)
    deal_damage_to_opponent(state, 1)


def cast_breath_weapon(state, card_def):
    """Real text: deals 2 damage to each NON-DRAGON creature. No card in
    this catalog is ever a Dragon (creature subtype isn't tracked at all
    here -- nothing needs it anywhere else), so that filter is always
    satisfied: this hits every creature currently in play, on either
    player's battlefield, a real symmetric board wipe (this deck's own
    creatures included, exactly like the real card)."""
    discard_from_hand_to_graveyard(state, card_def)
    for player in state.players:
        for permanent in player.battlefield:
            if permanent.card_def.card_type == CardType.CREATURE:
                permanent.damage_marked += 2
    check_state_based_actions(state)


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
    # Genuinely optional, no extra_legal gate -- always castable, even
    # with an empty hand and no land in play (see _highway_robbery_effect).
    EffectId.HIGHWAY_ROBBERY: {
        "cast": {"resolve": lambda state, card_def: cast_highway_robbery(state, card_def)},
        "plot": {
            "cost": {"generic": 1, "R": 1},
            "resolve": lambda state, card_def: plot_to_exile(state, card_def),
            "cast_from_exile_resolve": lambda state, card_def: cast_highway_robbery_from_exile(state, card_def),
        },
        "pending_kinds": {"discard_or_sacrifice"},
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
    EffectId.BREATH_WEAPON: {
        "cast": {"resolve": lambda state, card_def: cast_breath_weapon(state, card_def)},
    },
}


if __name__ == "__main__":
    # ponytail self-check: run via `python -m game.catalog.red_cards` from
    # src/. No pre-existing self-check block in this file to extend --
    # scoped narrowly to cast_breath_weapon, the one genuinely new piece of
    # logic added here (a symmetric board wipe across BOTH players'
    # battlefields, unlike every other burn spell in this file, which only
    # ever touches the opponent's life total).
    from ..state import GameState, Permanent, PlayerState

    state = GameState(on_the_play=True, players=[PlayerState(True), PlayerState(False)])
    breath_weapon = CardDef("Breath Weapon", CardType.INSTANT, {"generic": 2, "R": 1}, EffectId.BREATH_WEAPON)
    state.hand = [breath_weapon]
    mine_dies = Permanent(CardDef("Mine (dies)", CardType.CREATURE, None, EffectId.FILLER, power=1, toughness=2))
    mine_survives = Permanent(CardDef("Mine (survives)", CardType.CREATURE, None, EffectId.FILLER, power=1, toughness=3))
    theirs_dies = Permanent(CardDef("Theirs (dies)", CardType.CREATURE, None, EffectId.FILLER, power=1, toughness=1))
    not_a_creature = Permanent(CardDef("Some Land", CardType.LAND, None, EffectId.FILLER))
    state.players[0].battlefield = [mine_dies, mine_survives, not_a_creature]
    state.players[1].battlefield = [theirs_dies]

    cast_breath_weapon(state, breath_weapon)
    assert state.hand == [] and breath_weapon in state.graveyard
    assert mine_dies not in state.players[0].battlefield  # 2 damage >= 2 toughness -- this deck's own creature dies too
    assert mine_survives in state.players[0].battlefield and mine_survives.damage_marked == 2
    assert theirs_dies not in state.players[1].battlefield
    assert not_a_creature in state.players[0].battlefield  # a land is never a valid target

    print("red_cards.py Breath Weapon self-check: OK")

    # Highway Robbery: "discard a card or sacrifice a land. If you do,
    # draw two cards" -- both cost options, plus decline (no draw).

    # Discard path, discarding a Madness card: Madness's own
    # exile-not-graveyard replacement effect fires regardless of WHY the
    # card was discarded (an optional cost here, not Faithless Looting's
    # own discard-2 effect).
    state = GameState(on_the_play=True)
    hr = CardDef("Highway Robbery", CardType.SORCERY, {"generic": 1, "R": 1}, EffectId.HIGHWAY_ROBBERY)
    fiery_temper = CardDef("Fiery Temper", CardType.INSTANT, {"generic": 1, "R": 2}, EffectId.FIERY_TEMPER)
    state.hand = [hr, fiery_temper]
    state.library = [CardDef(f"Filler {i}", CardType.LAND, None, EffectId.MOUNTAIN) for i in range(2)]
    cast_highway_robbery(state, hr)
    assert state.pending_resolution["kind"] == "discard_or_sacrifice"
    resolution.execute_discard_or_sacrifice_option(state, "discard", "Fiery Temper")
    assert len(state.hand) == 2  # drew 2
    assert state.exile and state.exile[0][0].name == "Fiery Temper"  # exiled, not graveyarded -- Madness
    assert state.trigger_queue and state.trigger_queue[0]["kind"] == "madness"

    # Sacrifice-a-land path -- the alternative cost the old implementation
    # dropped entirely.
    state2 = GameState(on_the_play=True)
    hr2 = CardDef("Highway Robbery", CardType.SORCERY, {"generic": 1, "R": 1}, EffectId.HIGHWAY_ROBBERY)
    mountain = Permanent(CardDef("Mountain", CardType.LAND, None, EffectId.MOUNTAIN))
    state2.hand = [hr2]
    state2.battlefield = [mountain]
    state2.library = [CardDef(f"Filler {i}", CardType.LAND, None, EffectId.MOUNTAIN) for i in range(2)]
    cast_highway_robbery(state2, hr2)
    resolution.execute_discard_or_sacrifice_option(state2, "sacrifice", "Mountain")
    assert state2.battlefield == []
    assert sorted(c.name for c in state2.graveyard) == ["Highway Robbery", "Mountain"]
    assert len(state2.hand) == 2

    # Decline -- genuinely optional even with something payable on hand
    # (a spare land AND a spare card), no draw either way.
    state3 = GameState(on_the_play=True)
    hr3 = CardDef("Highway Robbery", CardType.SORCERY, {"generic": 1, "R": 1}, EffectId.HIGHWAY_ROBBERY)
    spare_card = CardDef("Lightning Bolt", CardType.INSTANT, {"R": 1}, EffectId.LIGHTNING_BOLT)
    spare_land = Permanent(CardDef("Mountain", CardType.LAND, None, EffectId.MOUNTAIN))
    state3.hand = [hr3, spare_card]
    state3.battlefield = [spare_land]
    cast_highway_robbery(state3, hr3)
    assert state3.pending_resolution["kind"] == "discard_or_sacrifice"  # genuinely offered, not auto-completed
    resolution.execute_discard_or_sacrifice_decline(state3)
    assert [c.name for c in state3.hand] == ["Lightning Bolt"]  # untouched, no draw
    assert spare_land in state3.battlefield  # untouched
    assert state3.pending_resolution is None

    print("red_cards.py Highway Robbery self-check: OK")
