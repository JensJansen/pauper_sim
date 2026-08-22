"""Red-identity card catalog: cards whose real mana cost is mono-red (or, for
a cost-less land, whose only mana output is red). cast_breath_weapon excludes
Dragons, matching green's Avenging Hunter (a Dragon) in this card pool."""

from .. import resolution
from ..cards import CardDef, CardType, EffectId, card_subtypes, is_artifact
from ..effects.casting import (
    _log_target_fizzle, capture_any_target, cast_permanent_from_hand, enters_battlefield,
    target_still_legal,
)
from ..effects.madness_and_plot import plot_to_exile
from ..effects.shared import (
    discard_from_hand_to_graveyard, find_and_remove_by_name, impulse_exile,
 shuffle_library,
)
from ..effects.stack import push_ability_to_stack, push_to_stack
from ..effects.state_based import check_state_based_actions, destroy_permanent, sacrifice_to_graveyard
from ..effects.stats import can_be_targeted, controller_idx, has_keyword
from ..effects.tokens import (
    BLOOD_TOKEN_CARD_DEF, HUMAN_SOLDIER_TOKEN_CARD_DEF, ROBOT_TOKEN_CARD_DEF, SAMURAI_TOKEN_CARD_DEF,
    TREASURE_TOKEN_CARD_DEF, create_token,
)
from ..mana import begin_pay_cost, float_mana, plan_payment
from ..turn import Speed
from ..effects.win_check import deal_damage_to_opponent, deal_damage_to_player

RED_CARD_CATALOG = {
    "Mountain": CardDef("Mountain", CardType.LAND, None, EffectId.MOUNTAIN, basic=True, subtypes=("Mountain",)),
    # Also an artifact: affinity/metalcraft/artifact-sac read extra["artifact"].
    "Great Furnace": CardDef("Great Furnace", CardType.LAND, None, EffectId.GREAT_FURNACE, artifact=True),
    "Voldaren Epicure": CardDef(
        "Voldaren Epicure", CardType.CREATURE, {"R": 1}, EffectId.VOLDAREN_EPICURE, power=1, toughness=1,
        subtypes=("Human", "Vampire"),  # Human -- for Rally at the Hornburg's "Humans you control gain haste"
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
    "Guttersnipe": CardDef(
        "Guttersnipe", CardType.CREATURE, {"generic": 2, "R": 1}, EffectId.GUTTERSNIPE, power=2, toughness=2,
    ),
    "Lava Dart": CardDef("Lava Dart", CardType.INSTANT, {"R": 1}, EffectId.LAVA_DART),
    "End the Festivities": CardDef("End the Festivities", CardType.SORCERY, {"R": 1}, EffectId.END_THE_FESTIVITIES),
    "Breath Weapon": CardDef("Breath Weapon", CardType.INSTANT, {"generic": 2, "R": 1}, EffectId.BREATH_WEAPON),

    # --- G3: jund_wildfire ---
    "Cleansing Wildfire": CardDef("Cleansing Wildfire", CardType.SORCERY, {"generic": 1, "R": 1}, EffectId.CLEANSING_WILDFIRE),

    # --- G5: mono_red_rally ---
    "Reckless Impulse": CardDef("Reckless Impulse", CardType.SORCERY, {"generic": 1, "R": 1}, EffectId.RECKLESS_IMPULSE),
    "Goblin Bushwhacker": CardDef(
        "Goblin Bushwhacker", CardType.CREATURE, {"R": 1}, EffectId.GOBLIN_BUSHWHACKER, power=1, toughness=1,
    ),

    # --- G6: mono_red_rally ---
    "Rally at the Hornburg": CardDef("Rally at the Hornburg", CardType.SORCERY, {"generic": 1, "R": 1}, EffectId.RALLY_AT_THE_HORNBURG),
    "Reckless Lackey": CardDef(
        "Reckless Lackey", CardType.CREATURE, {"R": 1}, EffectId.RECKLESS_LACKEY, power=1, toughness=2,
        sac_ability_cost={"generic": 2, "R": 1},
    ),
    "Goblin Tomb Raider": CardDef(
        "Goblin Tomb Raider", CardType.CREATURE, {"R": 1}, EffectId.GOBLIN_TOMB_RAIDER, power=1, toughness=2,
    ),
    # AUTHORIZED SIMPLIFICATION (owner, 2026-07-31): {R/G}{R/G} -- modeled
    # {R}{R}: the green half is unreachable in mono_red_rally (no green
    # source). Same deviation as Slippery Bogle's {G/U}->{G}
    # (multicolor_cards.py's own module docstring; real cost per Scryfall).
    "Burning-Tree Emissary": CardDef(
        "Burning-Tree Emissary", CardType.CREATURE, {"R": 2}, EffectId.BURNING_TREE_EMISSARY, power=2, toughness=2,
        subtypes=("Human", "Shaman"),  # Human -- for Rally at the Hornburg
    ),

    # --- G7: grixis_affinity / mono_red_rally ---
    "Galvanic Blast": CardDef("Galvanic Blast", CardType.INSTANT, {"R": 1}, EffectId.GALVANIC_BLAST),

    # --- G11: mono_red_rally ---
    "Chain Lightning": CardDef("Chain Lightning", CardType.SORCERY, {"R": 1}, EffectId.CHAIN_LIGHTNING),

    # --- G8: sac outlets / artifact engines (grixis / jund / mono_red) ---
    "Krark-Clan Shaman": CardDef("Krark-Clan Shaman", CardType.CREATURE, {"R": 1}, EffectId.KRARK_CLAN_SHAMAN, power=1, toughness=1),
    "Makeshift Munitions": CardDef("Makeshift Munitions", CardType.ENCHANTMENT, {"generic": 1, "R": 1}, EffectId.MAKESHIFT_MUNITIONS),
    "Experimental Synthesizer": CardDef("Experimental Synthesizer", CardType.ARTIFACT, {"R": 1}, EffectId.EXPERIMENTAL_SYNTHESIZER, sac_ability_cost={"generic": 2, "R": 1}),
    "Clockwork Percussionist": CardDef("Clockwork Percussionist", CardType.CREATURE, {"R": 1}, EffectId.CLOCKWORK_PERCUSSIONIST, power=1, toughness=1, artifact=True),
}


def _controls_artifact(state):
    return any(is_artifact(p.card_def) for p in state.battlefield)


def krark_clan_shaman_activate(state, permanent):
    """Sacrifice an artifact: deal 1 damage to each non-flying creature on
    either battlefield, itself included. Sacrifice is the cost; the sweep
    resolves on the stack."""
    def _on_sac(state, _ok):
        def _effect(st):
            for player in st.players:
                for p in player.battlefield:
                    if p.card_type == CardType.CREATURE and not has_keyword(st, p, "flying"):
                        p.damage_marked += 1
            check_state_based_actions(st)
        push_ability_to_stack(state, permanent.card_def, _effect)

    resolution.begin_sacrifice(state, lambda p: is_artifact(p.card_def), 1, _on_sac)


def _makeshift_munitions_legal(state, permanent):
    # {1} affordable AND an artifact or creature to sacrifice.
    if plan_payment(state, {"generic": 1}) is None:
        return False
    return any(p.card_type == CardType.CREATURE or is_artifact(p.card_def) for p in state.battlefield)


def makeshift_munitions_activate(state, permanent):
    """{1}, Sacrifice an artifact or creature: deal 1 damage to any target.
    Mana is paid, then the sacrifice, then the damage locks its target and
    resolves on the stack."""
    def _after_pay(st):
        def _on_sac(st2, _ok):
            _burn_choose_target_and_push(st2, permanent.card_def, 1, lambda s, c: None, reserves_hand_card=False, is_spell=False)
        resolution.begin_sacrifice(st, lambda p: p.card_type == CardType.CREATURE or is_artifact(p.card_def), 1, _on_sac)

    begin_pay_cost(state, {"generic": 1}, on_complete=_after_pay)


def experimental_synthesizer_sac(state, permanent):
    """{2}{R}, Sacrifice this artifact: create a 2/2 white Samurai with
    vigilance. sacrifice_to_graveyard also fires the card's own ltb_trigger
    (impulse-exile)."""
    sacrifice_to_graveyard(state, permanent)
    push_ability_to_stack(state, permanent.card_def, lambda st: create_token(st, SAMURAI_TOKEN_CARD_DEF))


def _galvanic_blast_amount(state):
    """Metalcraft: 4 damage if you control 3+ artifacts, else 2. Evaluated at
    resolution, when active_idx is back on the controller."""
    return 4 if sum(1 for p in state.battlefield if is_artifact(p.card_def)) >= 3 else 2


def cast_galvanic_blast(state, card_def):
    """{R}: deals 2 damage to any target, 4 with Metalcraft."""
    _cast_burn_any_target(state, card_def, _galvanic_blast_amount)


def cast_rally_at_the_hornburg(state, card_def):
    """{1}{R}: create two 1/1 white Human Soldier tokens; Humans you control
    gain haste until end of turn."""
    discard_from_hand_to_graveyard(state, card_def)
    create_token(state, HUMAN_SOLDIER_TOKEN_CARD_DEF)
    create_token(state, HUMAN_SOLDIER_TOKEN_CARD_DEF)
    for p in state.battlefield:
        if p.card_type == CardType.CREATURE and "Human" in card_subtypes(p.card_def):
            p.temp_keywords = p.temp_keywords | {"haste"}


def activate_reckless_lackey_sac(state, permanent):
    """{2}{R}, Sacrifice this creature: draw a card and create a Treasure
    token. No {T} in the cost, so this stays legal after Reckless Lackey has
    already attacked or blocked; sacrifice_to_graveyard also removes it from
    state.attackers/blocked_by."""
    sacrifice_to_graveyard(state, permanent)  # queues the dies-trigger (Gixian Infiltrator)

    def _effect(st):
        st.draw(1)
        create_token(st, TREASURE_TOKEN_CARD_DEF)

    push_ability_to_stack(state, permanent.card_def, _effect)


def _goblin_tomb_raider_controls_artifact(state, permanent):
    idx = controller_idx(state, permanent)
    if idx is None:
        return False
    return any(is_artifact(p.card_def) for p in state.players[idx].battlefield)


def burning_tree_emissary_etb(state):
    """Real card adds {R}{G}. AUTHORIZED SIMPLIFICATION (owner, 2026-08-02):
    adds {R}{R} instead -- mono_red_rally has no green source/sink for the G."""
    float_mana(state, ["R", "R"])
    state.log_event("mana_tap", permanent=("Burning-Tree Emissary", None), mode="etb", produced=["R", "R"])


def cast_reckless_impulse(state, card_def):
    """{1}{R}: exile the top two cards of your library; you may play them
    until the end of your next turn."""
    discard_from_hand_to_graveyard(state, card_def)
    impulse_exile(state, 2, until_next_turn=True)


def _goblin_bushwhacker_kicked(state, card_def):
    permanent = cast_permanent_from_hand(state, card_def)
    permanent.flags["kicked"] = True  # read by the ETB below


def goblin_bushwhacker_etb(state, permanent):
    """If kicked, creatures you control get +1/+0 and gain haste until end of
    turn, itself included."""
    if not permanent.flags.get("kicked"):
        return
    for p in state.battlefield:
        if p.card_type == CardType.CREATURE:
            p.temp_power += 1
            p.temp_keywords = p.temp_keywords | {"haste"}


def _is_basic_land(card_def):
    return card_def.extra.get("basic", False)


def cast_cleansing_wildfire(state, card_def):
    """{1}{R}: destroy target land (any land, either battlefield). Its
    controller may search their library for a basic, put it onto the
    battlefield tapped, and shuffle; the caster draws a card. Search and
    draw happen even if the land survives (indestructible); both are
    skipped only if the target is gone by resolution."""
    idx = state.active_idx

    def _on_target(state, descriptor):
        captured = capture_any_target(state, descriptor)

        def _resolve(state, card_def):
            discard_from_hand_to_graveyard(state, card_def)
            if captured is None or not target_still_legal(state, captured):
                where = (captured[1].card_def.name, captured[1].slot) if captured is not None else None
                _log_target_fizzle(state, card_def, where)
                return
            land = captured[1]
            caster = state.active_idx
            controller = controller_idx(state, land)  # the destroyed land's controller does the search
            destroy_permanent(state, land)  # no-op if indestructible

            def _after_search(state, fetched_name):
                found = find_and_remove_by_name(state, fetched_name) if fetched_name is not None else None
                shuffle_library(state)
                if found is not None:
                    enters_battlefield(state, found, force_tapped=True)
                state.active_idx = caster  # restore so the caster draws
                state.draw(1)

            state.active_idx = controller  # search runs as the land's controller
            resolution.begin_search_fetch(state, _is_basic_land, _after_search, optional=True)

        push_to_stack(state, card_def, _resolve, targets=() if captured is None else (captured,))

    resolution.begin_choose_any_target(
        state,
        lambda p: p.card_type == CardType.LAND and can_be_targeted(state, p, idx),
        _on_target, allow_players=False,
    )


def voldaren_epicure_etb(state):
    """When this creature enters: deal 1 damage to each opponent, create a
    Blood token."""
    deal_damage_to_opponent(state, 1)
    create_token(state, BLOOD_TOKEN_CARD_DEF)


def _resolve_burn_damage(state, captured, amount, card_def):
    """Apply a burn spell's `amount` damage to its locked any-target at
    resolution. A creature target that has left the battlefield fizzles the
    spell (608.2b); a player target is always legal. Shared by every burn
    below."""
    if not target_still_legal(state, captured):
        perm = captured[1]
        _log_target_fizzle(state, card_def, (perm.card_def.name, perm.slot))
        return
    # amount may be a callable(state) -> int (e.g. Galvanic Blast's Metalcraft).
    amt = amount(state) if callable(amount) else amount
    if captured[0] == "player":
        deal_damage_to_player(state, captured[1], amt)
    else:  # ("creature", permanent)
        captured[1].damage_marked += amt
        check_state_based_actions(state)


def _burn_choose_target_and_push(state, card_def, amount, to_graveyard, reserves_hand_card=True, is_spell=True, exiles_on_resolve=False):
    """Shared tail for every "deal `amount` damage to any target" effect --
    burn spells (is_spell=True) and Makeshift Munitions' ability
    (is_spell=False). Target (a creature on either battlefield, or either
    player) is chosen and locked as the spell/ability goes on the stack;
    `to_graveyard` handles how the source card leaves its zone on
    resolution, a no-op for the ability."""
    def _on_target(state, target):
        captured = capture_any_target(state, target)

        def _resolve(state, card_def):
            to_graveyard(state, card_def)
            _resolve_burn_damage(state, captured, amount, card_def)

        push_to_stack(state, card_def, _resolve, reserves_hand_card=reserves_hand_card, is_spell=is_spell,
                      exiles_on_resolve=exiles_on_resolve, targets=() if captured is None else (captured,))

    # excludes what the caster can't legally target: shroud, opponent hexproof.
    resolution.begin_choose_any_target(
        state,
        lambda p: p.card_type == CardType.CREATURE and can_be_targeted(state, p, state.active_idx),
        _on_target,
    )


def _cast_burn_any_target(state, card_def, amount):
    """Normal hand-cast of an "any target" burn: card goes hand -> graveyard
    on resolution."""
    _burn_choose_target_and_push(state, card_def, amount, discard_from_hand_to_graveyard)


def cast_lightning_bolt(state, card_def):
    """{R}: Lightning Bolt deals 3 damage to any target."""
    _cast_burn_any_target(state, card_def, 3)


def cast_fiery_temper(state, card_def):
    """{1}{R}{R}: Fiery Temper deals 3 damage to any target."""
    _cast_burn_any_target(state, card_def, 3)


def madness_fiery_temper(state, card_def):
    """Madness {R}: same "3 damage to any target", cast from exile. The card
    never touches hand; it goes straight to the graveyard on resolution."""
    _burn_choose_target_and_push(state, card_def, 3, lambda s, c: s.move_card(c, s.graveyard), reserves_hand_card=False)


def _chain_lightning_resolve_tail(state, captured, card_def):
    """Chain Lightning's effect on resolution (spell or copy): deal 3 to the
    locked any-target, then the copy rider -- the affected player (the target
    player, or a creature target's controller, captured before the damage can
    remove it) may pay {R}{R}; if they do, they may separately choose to copy
    the spell with a new target."""
    controller = state.active_idx
    if not target_still_legal(state, captured):
        perm = captured[1]
        _log_target_fizzle(state, card_def, (perm.card_def.name, perm.slot))
        return
    if captured[0] == "player":
        affected_idx = captured[1]
        deal_damage_to_player(state, affected_idx, 3)
    else:  # ("creature", permanent)
        affected_idx = controller_idx(state, captured[1])  # captured before SBA can remove it
        captured[1].damage_marked += 3
        check_state_based_actions(state)

    def _on_pay_result(state, paid):
        if not paid:
            return
        state.active_idx = affected_idx  # payer/copier owns the may-copy + new-target choices

        def _on_copy_decision(state, do_copy):
            if do_copy:
                _chain_lightning_make_copy(state, card_def, affected_idx, restore_idx=controller)
            else:
                state.active_idx = controller

        resolution.begin_may_copy(state, _on_copy_decision)

    resolution.begin_pay_unless(state, affected_idx, {"R": 2}, _on_pay_result)


def _chain_lightning_make_copy(state, card_def, copier_idx, restore_idx):
    """The copier puts a copy of Chain Lightning on the stack, controlled by
    them, and may choose a new target for it. A copy touches no zone and
    reserves no hand card; it carries the same resolve tail, so its own copy
    rider can fire recursively."""
    state.active_idx = copier_idx

    def _on_target(state, target):
        captured = capture_any_target(state, target)
        push_to_stack(
            state, card_def, lambda s, cd: _chain_lightning_resolve_tail(s, captured, cd),
            reserves_hand_card=False,
            targets=() if captured is None else (captured,),
        )
        state.active_idx = restore_idx

    resolution.begin_choose_any_target(
        state,
        lambda p: p.card_type == CardType.CREATURE and can_be_targeted(state, p, copier_idx),
        _on_target,
    )


def cast_chain_lightning(state, card_def):
    """{R} sorcery: deals 3 damage to any target, then the copy rider (see
    _chain_lightning_resolve_tail)."""
    def _on_target(state, target):
        captured = capture_any_target(state, target)

        def _resolve(state, card_def):
            discard_from_hand_to_graveyard(state, card_def)
            _chain_lightning_resolve_tail(state, captured, card_def)

        push_to_stack(state, card_def, _resolve, targets=() if captured is None else (captured,))

    resolution.begin_choose_any_target(
        state,
        lambda p: p.card_type == CardType.CREATURE and can_be_targeted(state, p, state.active_idx),
        _on_target,
    )


def faithless_looting_discard(state):
    """Draw two, then discard two -- shared by the normal cast and Flashback
    below."""
    state.draw(2)
    resolution.begin_discard(state, 2, optional=False, on_complete=lambda s, _cards: None)


def cast_faithless_looting(state, card_def):
    discard_from_hand_to_graveyard(state, card_def)
    faithless_looting_discard(state)


def flashback_faithless_looting(state, inst):
    """Flashback cost is {2}{R}, already paid by the generic flashback cost
    path before this resolve runs, so it pushes onto the stack immediately.

    inst: the exact graveyard CardInstance being flashed back -- see
    black_cards.flashback_dread_return."""
    state.graveyard.remove(inst)  # exiled on resolution (702.34)
    push_to_stack(state, inst, lambda st, cd: faithless_looting_discard(st), reserves_hand_card=False, exiles_on_resolve=True)


def _highway_robbery_effect(state):
    """You may discard a card or sacrifice a land; if you do, draw two cards.
    Genuinely optional, so casting never requires a card in hand or a land in
    play. Shared by the normal cast and Plot's cast-from-exile: the choice is
    made fresh at cast time, not locked in when plotted."""
    resolution.begin_discard_or_sacrifice(
        state, lambda p: p.card_def.card_type == CardType.LAND,
        on_complete=lambda s, paid: s.draw(2) if paid else None,
    )


def cast_highway_robbery(state, card_def):
    discard_from_hand_to_graveyard(state, card_def)
    _highway_robbery_effect(state)


def cast_highway_robbery_from_exile(state, card_def):
    """Plot's cast-from-exile resolve; the card never touches state.hand."""
    state.move_card(card_def, state.graveyard)
    _highway_robbery_effect(state)


def _grab_the_prize_extra_legal(state):
    """Additional-cost discard needs a card in hand besides the one cast."""
    return len(state.hand) >= 2


def _grab_the_prize_effect(state, discarded_cards):
    """Draw two cards. If the discarded card wasn't a land, deal 2 damage to
    each opponent. discarded_cards is always exactly 1 card (mandatory
    discard, guaranteed payable by extra_legal above)."""
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
    token. Sacrifice is the cost, paid now; the token creation is the effect,
    resolving on the stack. Melded Moxite is a real card, so it goes to the
    graveyard (701.17), unlike a token which would cease to exist."""
    sacrifice_to_graveyard(state, permanent)  # queues the dies-trigger (Gixian Infiltrator)
    push_ability_to_stack(state, permanent.card_def, lambda st: create_token(st, ROBOT_TOKEN_CARD_DEF, tapped=True))


def guttersnipe_on_cast(state, permanent):
    """Whenever you cast an instant or sorcery, deal 2 damage to each
    opponent."""
    deal_damage_to_opponent(state, 2)


def cast_fireblast(state, card_def):
    """{4}{R}{R}: Fireblast deals 4 damage to any target."""
    _cast_burn_any_target(state, card_def, 4)


def _fireblast_alt_extra_legal(state):
    return sum(1 for p in state.battlefield if p.card_def.name == "Mountain") >= 2


def cast_fireblast_alt(state, card_def):
    """You may sacrifice two Mountains rather than pay this spell's mana
    cost. Same "4 damage to any target" as the hard cast, once the sacrifice
    is paid."""
    discard_from_hand_to_graveyard(state, card_def)
    resolution.begin_sacrifice(
        state, lambda p: p.card_def.name == "Mountain", 2,
        on_complete=lambda s, ok: _burn_choose_target_and_push(s, card_def, 4, lambda st, cd: None, reserves_hand_card=False),
    )


def cast_lava_dart(state, card_def):
    """{R}: Lava Dart deals 1 damage to any target."""
    _cast_burn_any_target(state, card_def, 1)


def flashback_lava_dart(state, inst):
    """Flashback -- Sacrifice a Mountain, no mana cost. Same "1 damage to any
    target"; the card is exiled (not returned to graveyard) on resolution.

    inst: the exact graveyard CardInstance being flashed back -- see
    black_cards.flashback_dread_return."""
    state.graveyard.remove(inst)  # leaves the graveyard the moment Flashback is chosen; exiled on resolution
    resolution.begin_sacrifice(
        state, lambda p: p.card_def.name == "Mountain", 1,
        on_complete=lambda s, ok: _burn_choose_target_and_push(s, inst, 1, lambda st, cd: None, reserves_hand_card=False, exiles_on_resolve=True),
    )


def cast_end_the_festivities(state, card_def):
    """{R}: deal 1 damage to each opponent and each creature they control.
    Not symmetric -- this deck's own creatures are untouched."""
    discard_from_hand_to_graveyard(state, card_def)
    deal_damage_to_opponent(state, 1)
    for permanent in state.opponent.battlefield:
        if permanent.card_type == CardType.CREATURE:
            permanent.damage_marked += 1
    check_state_based_actions(state)


def cast_breath_weapon(state, card_def):
    """Deal 2 damage to each non-Dragon creature on either battlefield,
    including this deck's own. Green's Avenging Hunter is a Dragon, so the
    exclusion is live in this pool."""
    discard_from_hand_to_graveyard(state, card_def)
    for player in state.players:
        for permanent in player.battlefield:
            if permanent.card_type == CardType.CREATURE and "Dragon" not in card_subtypes(permanent.card_def):
                permanent.damage_marked += 2
    check_state_based_actions(state)


RED_EFFECT_REGISTRY = {
    EffectId.MOUNTAIN: {
        "mana": ("fixed", "R"),
    },
    EffectId.GREAT_FURNACE: {
        "mana": ("fixed", "R"),
    },
    EffectId.CLEANSING_WILDFIRE: {
        "cast": {
            "resolve": lambda state, card_def: cast_cleansing_wildfire(state, card_def),
            # needs >=1 targetable land on either battlefield
            "extra_legal": lambda state: any(
                p.card_type == CardType.LAND and can_be_targeted(state, p, state.active_idx)
                for pl in state.players for p in pl.battlefield),
            "precast_choice": True,  # target land chosen at cast
        },
        "pending_kinds": {"choose_any_target", "search_fetch"},
    },
    EffectId.RECKLESS_IMPULSE: {
        "cast": {"resolve": lambda state, card_def: cast_reckless_impulse(state, card_def)},
        "pending_kinds": {"impulse"},  # marks the deck as impulse-capable -> "Play from exile: X" actions
    },
    EffectId.RALLY_AT_THE_HORNBURG: {
        "cast": {"resolve": lambda state, card_def: cast_rally_at_the_hornburg(state, card_def)},
    },
    EffectId.RECKLESS_LACKEY: {
        "cast": {"resolve": lambda state, card_def: cast_permanent_from_hand(state, card_def)},
        "keywords": {"first_strike", "haste"},
        "activated_abilities": {
            "sac": {
                "cost_key": "sac_ability_cost",
                "resolve": lambda state, permanent: activate_reckless_lackey_sac(state, permanent),
            },
        },
    },
    EffectId.GOBLIN_TOMB_RAIDER: {
        "cast": {"resolve": lambda state, card_def: cast_permanent_from_hand(state, card_def)},
        # gets +1/+0 and haste while you control an artifact
        "static_self": {
            "condition": lambda state, permanent: _goblin_tomb_raider_controls_artifact(state, permanent),
            "power": 1,
            "keywords": {"haste"},
        },
    },
    EffectId.BURNING_TREE_EMISSARY: {
        "cast": {"resolve": lambda state, card_def: cast_permanent_from_hand(state, card_def)},
        "etb_trigger": lambda state, permanent: burning_tree_emissary_etb(state),
    },
    EffectId.GALVANIC_BLAST: {
        "cast": {"resolve": lambda state, card_def: cast_galvanic_blast(state, card_def), "precast_choice": True},
        "pending_kinds": {"choose_any_target"},
    },
    EffectId.KRARK_CLAN_SHAMAN: {
        "cast": {"resolve": lambda state, card_def: cast_permanent_from_hand(state, card_def)},
        "activated_abilities": {
            "sweep": {  # non-mana cost (sacrifice an artifact) -- legal/resolve path
                "legal": lambda state, permanent: _controls_artifact(state),
                "resolve": lambda state, permanent: krark_clan_shaman_activate(state, permanent),
            },
        },
        "pending_kinds": {"choose_permanent"},
    },
    EffectId.MAKESHIFT_MUNITIONS: {
        "cast": {"resolve": lambda state, card_def: cast_permanent_from_hand(state, card_def)},
        "activated_abilities": {
            "ping": {  # {1} + sacrifice -- paid inside the resolve, so legal/resolve path
                "legal": lambda state, permanent: _makeshift_munitions_legal(state, permanent),
                "resolve": lambda state, permanent: makeshift_munitions_activate(state, permanent),
            },
        },
        "pending_kinds": {"choose_permanent", "choose_any_target"},
    },
    EffectId.EXPERIMENTAL_SYNTHESIZER: {
        "cast": {"resolve": lambda state, card_def: cast_permanent_from_hand(state, card_def)},
        # ETB and LTB both impulse-exile the top card (playable until end of turn).
        "etb_trigger": lambda state, permanent: impulse_exile(state, 1, until_next_turn=False),
        "ltb_trigger": lambda state, permanent: impulse_exile(state, 1, until_next_turn=False),
        "activated_abilities": {
            "make_samurai": {
                "cost_key": "sac_ability_cost",
                "speed": Speed.SORCERY,  # "Activate only as a sorcery"
                "resolve": lambda state, permanent: experimental_synthesizer_sac(state, permanent),
            },
        },
        "pending_kinds": {"impulse"},  # marks the deck impulse-capable
    },
    EffectId.CLOCKWORK_PERCUSSIONIST: {
        "cast": {"resolve": lambda state, card_def: cast_permanent_from_hand(state, card_def)},
        "keywords": {"haste"},
        # "When this creature dies, ..." -- a battlefield->graveyard leave, so ltb_trigger.
        "ltb_trigger": lambda state, permanent: impulse_exile(state, 1, until_next_turn=True),
        "pending_kinds": {"impulse"},
    },
    EffectId.GOBLIN_BUSHWHACKER: {
        "cast_modes": {
            # kicked = {R}{R}, flags the ETB to pump the team
            "unkicked": {"resolve": lambda state, card_def: cast_permanent_from_hand(state, card_def)},
            "kicked": {"cost": {"R": 2}, "resolve": lambda state, card_def: _goblin_bushwhacker_kicked(state, card_def)},
        },
        "etb_trigger": lambda state, permanent: goblin_bushwhacker_etb(state, permanent),
    },
    EffectId.VOLDAREN_EPICURE: {
        "cast": {"resolve": lambda state, card_def: cast_permanent_from_hand(state, card_def)},
        "etb_trigger": lambda state, permanent: voldaren_epicure_etb(state),
    },
    EffectId.LIGHTNING_BOLT: {
        # precast_choice: target locked as the spell is cast, not at resolution
        "cast": {"resolve": lambda state, card_def: cast_lightning_bolt(state, card_def), "precast_choice": True},
        "pending_kinds": {"choose_any_target"},
    },
    EffectId.CHAIN_LIGHTNING: {
        # copy rider adds pay_unless + a second choose_any_target for the copier
        "cast": {"resolve": lambda state, card_def: cast_chain_lightning(state, card_def), "precast_choice": True},
        "pending_kinds": {"choose_any_target", "pay_unless", "may_copy"},
    },
    EffectId.FIERY_TEMPER: {
        "cast": {"resolve": lambda state, card_def: cast_fiery_temper(state, card_def), "precast_choice": True},
        "madness": {
            "cost": {"R": 1}, "resolve": lambda state, card_def: madness_fiery_temper(state, card_def),
            "precast_choice": True,
        },
        # order_triggers: reachable when 2+ Madness cards discard at once,
        # e.g. via Faithless Looting's discard-2 below.
        "pending_kinds": {"madness_decision", "order_triggers", "choose_any_target"},
    },
    EffectId.FAITHLESS_LOOTING: {
        "cast": {"resolve": lambda state, card_def: cast_faithless_looting(state, card_def)},
        "flashback": {
            "cost": {"generic": 2, "R": 1},  # real Flashback cost -- paid by the generic mana-flashback path
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
        # alt_cast chooses its own target after the sacrifice cost, so it
        # needs no precast_choice flag, just the pending kind.
        "cast": {"resolve": lambda state, card_def: cast_fireblast(state, card_def), "precast_choice": True},
        "alt_cast": {
            "extra_legal": lambda state: _fireblast_alt_extra_legal(state),
            "resolve": lambda state, card_def: cast_fireblast_alt(state, card_def),
        },
        "pending_kinds": {"choose_permanent", "choose_any_target"},
    },
    EffectId.LAVA_DART: {
        "cast": {"resolve": lambda state, card_def: cast_lava_dart(state, card_def), "precast_choice": True},
        "flashback": {
            "legal": lambda state: any(p.card_def.name == "Mountain" for p in state.battlefield),
            "resolve": lambda state, card_def: flashback_lava_dart(state, card_def),
        },
        "pending_kinds": {"choose_permanent", "choose_any_target"},
    },
    EffectId.END_THE_FESTIVITIES: {
        "cast": {"resolve": lambda state, card_def: cast_end_the_festivities(state, card_def)},
    },
    EffectId.BREATH_WEAPON: {
        "cast": {"resolve": lambda state, card_def: cast_breath_weapon(state, card_def)},
    },
}

