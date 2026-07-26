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
from ..effects.casting import _log_target_fizzle, capture_any_target, cast_permanent_from_hand, target_still_legal
from ..effects.madness_and_plot import plot_to_exile
from ..effects.shared import discard_from_hand_to_graveyard
from ..effects.stack import push_ability_to_stack, push_to_stack
from ..effects.state_based import check_state_based_actions
from ..effects.stats import can_be_targeted
from ..effects.tokens import BLOOD_TOKEN_CARD_DEF, ROBOT_TOKEN_CARD_DEF, create_token
from ..effects.win_check import deal_damage_to_opponent, deal_damage_to_player

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
    #.
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


def _resolve_burn_damage(state, captured, amount, card_def):
    """Apply a burn spell's `amount` damage to its locked any-target when it
    resolves off the stack. Real Magic 608.2b: if the target is now illegal
    (a creature that has left the battlefield) the spell fizzles -- no
    damage, already in the graveyard. A player target never becomes illegal.
    Shared by every "N damage to any target" burn below; the card is already
    hand->graveyard by the time this runs (its own resolve did that first,
    same as any resolving spell)."""
    if not target_still_legal(state, captured):
        perm = captured[1]  # a creature target that's gone -- (name, slot) for the fizzle log
        _log_target_fizzle(state, card_def, (perm.card_def.name, perm.slot))
        return
    if captured[0] == "player":
        deal_damage_to_player(state, captured[1], amount)
    else:  # ("creature", permanent)
        captured[1].damage_marked += amount
        check_state_based_actions(state)


def _burn_choose_target_and_push(state, card_def, amount, to_graveyard, reserves_hand_card=True):
    """Shared faithful-burn cast tail for every "deal `amount` damage to any
    target" spell (Lightning Bolt, Fiery Temper, Fireblast, Lava Dart). The
    target (any creature on either battlefield -- hexproof/shroud aware -- or
    either player, yourself legal) is chosen AS THE SPELL IS CAST and locked
    onto the stack via capture_any_target; the effect waits on the stack and,
    on resolution, hits that exact target or fizzles (a creature target gone
    by then, rule 608.2b). Each cast MODE supplies how its card reaches the
    graveyard on resolution -- `to_graveyard(state, card_def)` -- and whether
    a same-named hand copy is still spoken for (`reserves_hand_card`): a
    normal hand cast discards from hand and reserves; madness/flashback/alt
    have already moved the card out of its prior zone by cast time."""
    def _on_target(state, target):
        captured = capture_any_target(state, target)

        def _resolve(state, card_def):
            to_graveyard(state, card_def)
            _resolve_burn_damage(state, captured, amount, card_def)

        push_to_stack(state, card_def, _resolve, reserves_hand_card=reserves_hand_card)

    # "any target" creature candidates exclude what the caster can't legally
    # target: shroud (anyone) and opponent-controlled hexproof (can_be_targeted).
    resolution.begin_choose_any_target(
        state,
        lambda p: p.card_type == CardType.CREATURE and can_be_targeted(state, p, state.active_idx),
        _on_target,
    )


def _cast_burn_any_target(state, card_def, amount):
    """Normal hand-cast of an 'any target' burn: card goes hand -> graveyard
    on resolution (the 'still in hand while on the stack' convention)."""
    _burn_choose_target_and_push(state, card_def, amount, discard_from_hand_to_graveyard)


def cast_lightning_bolt(state, card_def):
    """{R}: Lightning Bolt deals 3 damage to any target."""
    _cast_burn_any_target(state, card_def, 3)


def cast_fiery_temper(state, card_def):
    """{1}{R}{R}: Fiery Temper deals 3 damage to any target."""
    _cast_burn_any_target(state, card_def, 3)


def madness_fiery_temper(state, card_def):
    """Madness {R}: same "3 damage to any target", cast from exile. By the
    time this runs (precast_choice, so execute_madness_cast calls it directly
    rather than pushing it) the card is already out of exile -- it appends
    itself to the graveyard on resolution, never touches hand. Target is
    locked here as the spell is put on the stack, same as a hard cast."""
    _burn_choose_target_and_push(state, card_def, 3, lambda s, c: s.graveyard.append(c), reserves_hand_card=False)


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
    artifact creature token (the same shared ROBOT_TOKEN_CARD_DEF).

    Faithful timing: the sacrifice is a COST, paid
    now on activation; creating the token is the effect, so it goes on the
    stack (push_ability_to_stack) and resolves after a priority window.

    Melded Moxite is a real (nontoken) card, so sacrificing it puts it in
    the graveyard (real Magic 701.17 -- unlike Blood/Eldrazi Spawn tokens,
    which cease to exist), same as Candy Trail's own sac ability."""
    state.battlefield.remove(permanent)
    state.graveyard.append(permanent.card_def)
    state.log_event(
        "zone_move", permanent=(permanent.card_def.name, permanent.slot), from_zone="battlefield",
        to_zone="graveyard", reason="sacrifice",
    )
    push_ability_to_stack(state, permanent.card_def, lambda st: create_token(st, ROBOT_TOKEN_CARD_DEF, tapped=True))


def guttersnipe_on_cast(state, permanent):
    """Whenever you cast an instant or sorcery spell, deals 2 damage to
    each opponent -- fires via the generic on_cast_trigger chokepoint,
    identically for every cast path (normal, Flashback, Madness, Plot)
    already wired through it."""
    deal_damage_to_opponent(state, 2)


def cast_fireblast(state, card_def):
    """{4}{R}{R}: Fireblast deals 4 damage to any target."""
    _cast_burn_any_target(state, card_def, 4)


def _fireblast_alt_extra_legal(state):
    return sum(1 for p in state.battlefield if p.card_def.name == "Mountain") >= 2


def cast_fireblast_alt(state, card_def):
    """You may sacrifice two Mountains rather than pay this spell's mana
    cost. Same "4 damage to any target" as the hard-cast; once the sacrifice
    -- this alt cost -- is paid, the target is chosen and the effect pushed
    onto the stack. The card goes to the graveyard as it's cast (eager, kept
    from before this became targeted -- see drl_env's hand-count note), so
    its stack resolve does no further zone move."""
    discard_from_hand_to_graveyard(state, card_def)
    resolution.begin_sacrifice(
        state, lambda p: p.card_def.name == "Mountain", 2,
        on_complete=lambda s, ok: _burn_choose_target_and_push(s, card_def, 4, lambda st, cd: None, reserves_hand_card=False),
    )


def cast_lava_dart(state, card_def):
    """{R}: Lava Dart deals 1 damage to any target."""
    _cast_burn_any_target(state, card_def, 1)


def flashback_lava_dart(state, card_def):
    """Flashback -- Sacrifice a Mountain: no mana component at all. Same "1
    damage to any target"; once the sacrifice is paid, the target is chosen
    and the effect pushed onto the stack. The card left the graveyard when
    Flashback was chosen and is exiled after (untracked) -- so its stack
    resolve does no zone move."""
    state.graveyard.remove(card_def)  # leaves the graveyard the moment Flashback is chosen -- exiled after, untracked (Dread Return's own Flashback precedent)
    resolution.begin_sacrifice(
        state, lambda p: p.card_def.name == "Mountain", 1,
        on_complete=lambda s, ok: _burn_choose_target_and_push(s, card_def, 1, lambda st, cd: None, reserves_hand_card=False),
    )


def cast_end_the_festivities(state, card_def):
    """Real text: End the Festivities deals 1 damage to EACH CREATURE -- a
    symmetric 1-damage board sweep hitting every creature on either
    battlefield (this deck's own included), same shape as cast_breath_weapon's
    2-damage wipe. It does NOT hit players (the prior "1 to the opponent's
    face" was a misread of the card)."""
    discard_from_hand_to_graveyard(state, card_def)
    for player in state.players:
        for permanent in player.battlefield:
            if permanent.card_type == CardType.CREATURE:
                permanent.damage_marked += 1
    check_state_based_actions(state)


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
            if permanent.card_type == CardType.CREATURE:
                permanent.damage_marked += 2
    check_state_based_actions(state)


RED_EFFECT_REGISTRY = {
    EffectId.MOUNTAIN: {
        "mana": ("fixed", "R"),
    },
    EffectId.VOLDAREN_EPICURE: {
        "cast": {"resolve": lambda state, card_def: cast_permanent_from_hand(state, card_def)},
        "etb_trigger": lambda state, permanent: voldaren_epicure_etb(state),
    },
    EffectId.LIGHTNING_BOLT: {
        # precast_choice: "any target" is locked in as the spell is cast
        # (drl_env._precast_choice_execute), not at resolution -- real Magic.
        "cast": {"resolve": lambda state, card_def: cast_lightning_bolt(state, card_def), "precast_choice": True},
        "pending_kinds": {"choose_any_target"},
    },
    EffectId.FIERY_TEMPER: {
        # precast_choice on BOTH modes: "any target" is locked as the spell
        # is put on the stack (madness routes through execute_madness_cast's
        # own precast_choice branch), never at resolution.
        "cast": {"resolve": lambda state, card_def: cast_fiery_temper(state, card_def), "precast_choice": True},
        "madness": {
            "cost": {"R": 1}, "resolve": lambda state, card_def: madness_fiery_temper(state, card_def),
            "precast_choice": True,
        },
        # order_triggers: reachable the
        # instant 2+ Madness cards get discarded at once -- Faithless
        # Looting's own discard-2, right below, is exactly that source.
        "pending_kinds": {"madness_decision", "order_triggers", "choose_any_target"},
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
        "etb_trigger": lambda state, permanent: melded_moxite_etb(state),
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
        # Hard cast is precast_choice (target locked at cast); the alt_cast
        # resolve chooses its own target after the sacrifice cost and pushes
        # itself, so it needs no precast flag -- just the pending kind.
        "cast": {"resolve": lambda state, card_def: cast_fireblast(state, card_def), "precast_choice": True},
        "alt_cast": {
            "extra_legal": lambda state: _fireblast_alt_extra_legal(state),
            "resolve": lambda state, card_def: cast_fireblast_alt(state, card_def),
        },
        "pending_kinds": {"sacrifice", "choose_any_target"},
    },
    EffectId.LAVA_DART: {
        "cast": {"resolve": lambda state, card_def: cast_lava_dart(state, card_def), "precast_choice": True},
        "flashback": {
            "legal": lambda state: any(p.card_def.name == "Mountain" for p in state.battlefield),
            "resolve": lambda state, card_def: flashback_lava_dart(state, card_def),
        },
        "pending_kinds": {"sacrifice", "choose_any_target"},
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

    # Lightning Bolt: faithful "3 damage to any target" -- target locked at
    # cast (capture_any_target), applied or fizzled at resolution. Creature
    # on either side, either player (self is legal), and the 608.2b fizzle
    # when the chosen creature leaves the battlefield before it resolves.
    import contextlib
    import io

    from .. import resolution as _res
    from ..effects.stack import resolve_top_of_stack

    bolt = CardDef("Lightning Bolt", CardType.INSTANT, {"R": 1}, EffectId.LIGHTNING_BOLT)

    def _fresh_bolt_state():
        s = GameState(on_the_play=True, players=[PlayerState(True), PlayerState(False)])
        s.hand = [bolt]
        return s

    # (a) opponent creature -> 3 damage marked on that exact creature
    state = _fresh_bolt_state()
    opp_creature = Permanent(CardDef("Grizzly Bears", CardType.CREATURE, None, EffectId.FILLER, power=2, toughness=4))
    opp_creature.slot = 1
    state.players[1].battlefield = [opp_creature]
    cast_lightning_bolt(state, bolt)  # begins choose_any_target (precast)
    _res.execute_choose_any_target_creature(state, 1, "Grizzly Bears", 1)  # lock the opponent's creature
    assert state.hand == [bolt]  # still in hand while paid-but-unresolved on the stack
    resolve_top_of_stack(state)
    assert opp_creature.damage_marked == 3 and bolt in state.graveyard

    # (b) opponent player -> 3 to the face (20 -> 17)
    state = _fresh_bolt_state()
    cast_lightning_bolt(state, bolt)
    _res.execute_choose_any_target_player(state, 1)
    resolve_top_of_stack(state)
    assert state.players[1].life_total == 17  # 20 - 3 to the opponent's face

    # (c) yourself -> legal, 3 to own face (20 -> 17), pure self-damage
    state = _fresh_bolt_state()
    cast_lightning_bolt(state, bolt)
    _res.execute_choose_any_target_player(state, 0)
    resolve_top_of_stack(state)
    assert state.players[0].life_total == 17  # 3 to your OWN face -- pure self-damage, never opponent

    # (d) FIZZLE: chosen creature gone before resolution -> no effect
    state = _fresh_bolt_state()
    doomed = Permanent(CardDef("Grizzly Bears", CardType.CREATURE, None, EffectId.FILLER, power=2, toughness=4))
    doomed.slot = 1
    state.players[1].battlefield = [doomed]
    cast_lightning_bolt(state, bolt)
    _res.execute_choose_any_target_creature(state, 1, "Grizzly Bears", 1)
    state.players[1].battlefield = []  # target leaves before the bolt resolves
    fizzle_log = io.StringIO()
    with contextlib.redirect_stdout(fizzle_log):
        resolve_top_of_stack(state)
    assert "fizzle" in fizzle_log.getvalue().lower()
    assert bolt in state.graveyard and state.players[1].life_total == 20  # nothing happened

    # (e) hexproof: an OPPONENT'S hexproof creature is NOT a legal Bolt
    # target, but the caster's OWN hexproof creature still is (boggles' whole
    # point -- opponents can't burn its bogles).
    from .. import registry as _reg
    _fb = _reg.EFFECT_REGISTRY[EffectId.FILLER]
    try:
        _reg.EFFECT_REGISTRY[EffectId.FILLER] = {"keywords": {"hexproof"}}
        state = _fresh_bolt_state()
        opp_hex = Permanent(CardDef("Hex Bear", CardType.CREATURE, None, EffectId.FILLER, power=1, toughness=1))
        mine_hex = Permanent(CardDef("Hex Bear", CardType.CREATURE, None, EffectId.FILLER, power=1, toughness=1))
        opp_hex.slot = mine_hex.slot = 1
        state.players[1].battlefield = [opp_hex]  # opponent's -- untargetable by me
        state.players[0].battlefield = [mine_hex]  # my own -- still targetable by me
        cast_lightning_bolt(state, bolt)
        creature_opts = _res.choose_any_target_creature_options(state)
        assert (0, "Hex Bear", 1) in creature_opts  # my own hexproof creature is fair game to me
        assert (1, "Hex Bear", 1) not in creature_opts  # the opponent's is not
    finally:
        _reg.EFFECT_REGISTRY[EffectId.FILLER] = _fb

    print("red_cards.py Lightning Bolt any-target self-check: OK")
