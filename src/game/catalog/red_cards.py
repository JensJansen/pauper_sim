"""Red-identity card catalog: every card whose real mana cost is mono-red (or,
for a cost-less land, whose only mana output is red). Costs/types/oracle text
are direct Scryfall pulls; creature power/toughness is a design choice. Breath
Weapon (real {2}{R}) files here despite being Tron filler; its "non-Dragon"
filter is enforced against green's Avenging Hunter (a Dragon), so
cast_breath_weapon is NOT a purely symmetric wipe -- Dragons are excluded."""

from .. import resolution
from ..cards import CardDef, CardType, EffectId, card_subtypes, is_artifact
from ..effects.casting import (
    _log_target_fizzle, capture_any_target, cast_permanent_from_hand, enters_battlefield,
    target_still_legal,
)
from ..effects.madness_and_plot import plot_to_exile
from ..effects.shared import (
    discard_from_hand_to_graveyard, find_and_remove_by_name, fire_sacrifice_triggers, impulse_exile,
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
    # Artifact land: played as a land, but also an artifact (affinity/
    # metalcraft/artifact-sac read extra["artifact"]).
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
    """Sacrifice an artifact: this creature deals 1 damage to each creature
    without flying (both battlefields, itself included). The sacrifice is the
    cost; the sweep is the effect (on the stack)."""
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
    """{1}, Sacrifice an artifact or creature: this enchantment deals 1 damage
    to any target. Pay {1}, then sacrifice (the cost), then the 1 damage waits
    on the stack for its locked target."""
    def _after_pay(st):
        def _on_sac(st2, _ok):
            _burn_choose_target_and_push(st2, permanent.card_def, 1, lambda s, c: None, reserves_hand_card=False, is_spell=False)
        resolution.begin_sacrifice(st, lambda p: p.card_type == CardType.CREATURE or is_artifact(p.card_def), 1, _on_sac)

    begin_pay_cost(state, {"generic": 1}, on_complete=_after_pay)


def experimental_synthesizer_sac(state, permanent):
    """{2}{R}, Sacrifice this artifact: create a 2/2 white Samurai with
    vigilance. Sacrificing also fires its own "or leaves the battlefield"
    impulse (its ltb_trigger, via sacrifice_to_graveyard)."""
    sacrifice_to_graveyard(state, permanent)  # queues the LTB impulse
    push_ability_to_stack(state, permanent.card_def, lambda st: create_token(st, SAMURAI_TOKEN_CARD_DEF))


def _galvanic_blast_amount(state):
    """Metalcraft: 4 damage if you control 3+ artifacts, else 2. Evaluated at
    resolution (state.battlefield is the caster's own, active_idx restored to
    the controller by resolve_top_of_stack)."""
    return 4 if sum(1 for p in state.battlefield if is_artifact(p.card_def)) >= 3 else 2


def cast_galvanic_blast(state, card_def):
    """{R}: Galvanic Blast deals 2 damage to any target -- 4 with Metalcraft.
    Reuses the shared any-target burn tail with a callable amount (the
    Metalcraft check runs at resolution)."""
    _cast_burn_any_target(state, card_def, _galvanic_blast_amount)


def cast_rally_at_the_hornburg(state, card_def):
    """{1}{R}: Create two 1/1 white Human Soldier tokens; Humans you control
    gain haste until end of turn (temp keyword, cleared at cleanup)."""
    discard_from_hand_to_graveyard(state, card_def)
    create_token(state, HUMAN_SOLDIER_TOKEN_CARD_DEF)
    create_token(state, HUMAN_SOLDIER_TOKEN_CARD_DEF)
    for p in state.battlefield:
        if p.card_type == CardType.CREATURE and "Human" in card_subtypes(p.card_def):
            p.temp_keywords = p.temp_keywords | {"haste"}


def activate_reckless_lackey_sac(state, permanent):
    """{2}{R}, Sacrifice this creature: Draw a card and create a Treasure
    token. The {2}{R} + untapped precondition come from the generic cost_key
    wiring; sacrifice (a real card -> graveyard) is a cost paid now, and the
    draw + Treasure are the effect (on the stack, after a priority window)."""
    state.battlefield.remove(permanent)
    state.move_card(permanent.card_def, state.graveyard)
    state.log_event(
        "zone_move", permanent=(permanent.card_def.name, permanent.slot), from_zone="battlefield",
        to_zone="graveyard", reason="sacrifice",
    )
    fire_sacrifice_triggers(state, state.active_idx, permanent.card_def)  # Gixian Infiltrator

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
    """{1}{R}: Exile the top two cards of your library; until the end of your
    NEXT turn, you may play them (shared.impulse_exile -> the impulse zone;
    the "Play from exile: X" actions cast/play them, paying normal costs)."""
    discard_from_hand_to_graveyard(state, card_def)
    impulse_exile(state, 2, until_next_turn=True)


def _goblin_bushwhacker_kicked(state, card_def):
    permanent = cast_permanent_from_hand(state, card_def)
    permanent.flags["kicked"] = True  # read by the ETB below


def goblin_bushwhacker_etb(state, permanent):
    """"When this creature enters, if it was kicked, creatures you control get
    +1/+0 and gain haste until end of turn." Until-EOT team pump via the G3
    temp-modifier machinery (temp_power / temp_keywords, cleared at cleanup).
    Includes Bushwhacker itself, and the granted haste lets the team attack
    this turn."""
    if not permanent.flags.get("kicked"):
        return
    for p in state.battlefield:
        if p.card_type == CardType.CREATURE:
            p.temp_power += 1
            p.temp_keywords = p.temp_keywords | {"haste"}


def _is_basic_land(card_def):
    return card_def.extra.get("basic", False)


def cast_cleansing_wildfire(state, card_def):
    """{1}{R}: "Destroy target land. Its controller may search their library
    for a basic land card, put it onto the battlefield tapped, then shuffle.
    Draw a card." Target ANY land on either battlefield (locked at cast). The
    DESTROYED land's CONTROLLER (not necessarily the caster) does the optional
    search into THEIR OWN library, putting the basic onto THEIR battlefield
    tapped and shuffling THEIR library; the CASTER draws.

    The search and the draw happen regardless of whether the destroy actually
    succeeded (an indestructible land survives, but the controller still
    searches) -- both skipped only if the target land is gone by resolution
    (target illegal -> the spell does nothing, 608.2b)."""
    idx = state.active_idx

    def _on_target(state, descriptor):
        captured = capture_any_target(state, descriptor)

        def _resolve(state, card_def):
            discard_from_hand_to_graveyard(state, card_def)  # the sorcery itself -> caster's graveyard
            if captured is None or not target_still_legal(state, captured):
                where = (captured[1].card_def.name, captured[1].slot) if captured is not None else None
                _log_target_fizzle(state, card_def, where)  # target gone -> no destroy, no search, no draw
                return
            land = captured[1]
            caster = state.active_idx  # controller of the resolving spell
            controller = controller_idx(state, land)  # the DESTROYED land's controller does the search
            destroy_permanent(state, land)  # no-op if indestructible; search/draw still happen

            def _after_search(state, fetched_name):
                found = find_and_remove_by_name(state, fetched_name) if fetched_name is not None else None
                state.rng.shuffle(state.library)  # the controller's library (active_idx == controller here)
                if found is not None:
                    enters_battlefield(state, found, force_tapped=True)  # onto the controller's battlefield, tapped
                state.active_idx = caster  # restore before the CASTER draws
                state.draw(1)  # "Draw a card" (the caster) -- whether or not a basic was fetched

            state.active_idx = controller  # the land's controller searches THEIR own library
            resolution.begin_search_fetch(state, _is_basic_land, _after_search, optional=True)  # "MAY search"

        push_to_stack(state, card_def, _resolve, targets=() if captured is None else (captured,))

    resolution.begin_choose_any_target(
        state,
        lambda p: p.card_type == CardType.LAND and can_be_targeted(state, p, idx),
        _on_target, allow_players=False,
    )


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
    # `amount` may be a callable(state) -> int for a burn whose damage is
    # decided at resolution (Galvanic Blast's Metalcraft: 4 if you control 3+
    # artifacts, else 2), not a fixed int (Lightning Bolt).
    amt = amount(state) if callable(amount) else amount
    if captured[0] == "player":
        deal_damage_to_player(state, captured[1], amt)
    else:  # ("creature", permanent)
        captured[1].damage_marked += amt
        check_state_based_actions(state)


def _burn_choose_target_and_push(state, card_def, amount, to_graveyard, reserves_hand_card=True, is_spell=True, exiles_on_resolve=False):
    """Shared faithful-burn tail for every "deal `amount` damage to any target"
    effect. Used both by burn SPELLS (Lightning Bolt, Fiery Temper, Fireblast,
    Lava Dart -- is_spell=True) and by Makeshift Munitions' activated ABILITY
    (is_spell=False -- an ability is not a Counterspell/Spell Pierce target, and
    its source enchantment never leaves the battlefield). The target (any
    creature on either battlefield -- hexproof/shroud aware -- or either player,
    yourself legal) is chosen AS THE SPELL/ABILITY IS PUT ON THE STACK and locked
    via capture_any_target; the effect waits on the stack and, on resolution,
    hits that exact target or fizzles (a creature target gone by then, rule
    608.2b). Each MODE supplies how its card reaches the graveyard on resolution
    -- `to_graveyard(state, card_def)` (a no-op for the ability, whose source
    stays put) -- and whether a same-named hand copy is still spoken for
    (`reserves_hand_card`): a normal hand cast discards from hand and reserves;
    madness/flashback/alt and the ability have already moved (or never had) the
    card out of its prior zone by push time."""
    def _on_target(state, target):
        captured = capture_any_target(state, target)

        def _resolve(state, card_def):
            to_graveyard(state, card_def)
            _resolve_burn_damage(state, captured, amount, card_def)

        push_to_stack(state, card_def, _resolve, reserves_hand_card=reserves_hand_card, is_spell=is_spell,
                      exiles_on_resolve=exiles_on_resolve, targets=() if captured is None else (captured,))

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
    _burn_choose_target_and_push(state, card_def, 3, lambda s, c: s.move_card(c, s.graveyard), reserves_hand_card=False)


def _chain_lightning_resolve_tail(state, captured, card_def):
    """Chain Lightning's own effect once it (or a copy) resolves off the
    stack: deal 3 to the locked any-target, then the copy rider. Real Magic
    608.2b fizzle if a creature target has left; a player target is always
    legal. "Then that player or that permanent's controller may pay {R}{R}. If
    the player does, they may copy this spell and may choose a new target for
    that copy" -- the affected player is the target player, or (for a creature
    target) that creature's controller, captured BEFORE the 3 damage can kill
    and remove it (last-known controller, matching the resolving instruction).

    Two INDEPENDENT "may"s, faithfully separate: (1) the affected player may
    pay {R}{R} (begin_pay_unless); (2) IF they paid, they may copy (begin_may_
    copy) -- and only then choose a new target for the copy. active_idx is the
    resolving spell's controller here; the payer/copier's own two decisions run
    under active_idx flipped to them, restored to the controller once the whole
    rider settles (the begin_pay_unless convention, extended across the copy)."""
    controller = state.active_idx  # the resolving spell's/copy's own controller
    if not target_still_legal(state, captured):
        perm = captured[1]
        _log_target_fizzle(state, card_def, (perm.card_def.name, perm.slot))
        return
    if captured[0] == "player":
        affected_idx = captured[1]
        deal_damage_to_player(state, affected_idx, 3)
    else:  # ("creature", permanent)
        affected_idx = controller_idx(state, captured[1])  # capture before SBA can remove it
        captured[1].damage_marked += 3
        check_state_based_actions(state)

    def _on_pay_result(state, paid):
        if not paid:
            return  # first "may" declined (or unaffordable) -> no copy; active_idx already restored to controller
        # {R}{R} paid. The SECOND, independent "may": the payer may copy.
        state.active_idx = affected_idx  # the payer/copier owns the may-copy + new-target choices

        def _on_copy_decision(state, do_copy):
            if do_copy:
                _chain_lightning_make_copy(state, card_def, affected_idx, restore_idx=controller)
            else:
                state.active_idx = controller  # paid but declined to copy (a legal, if rare, line)

        resolution.begin_may_copy(state, _on_copy_decision)

    resolution.begin_pay_unless(state, affected_idx, {"R": 2}, _on_pay_result)


def _chain_lightning_make_copy(state, card_def, copier_idx, restore_idx):
    """The copier puts a COPY of Chain Lightning on the stack -- controlled by
    them, so its own rider's payment/perspective are theirs -- and "may choose
    a new target for that copy" (begin_choose_any_target from the copier's
    perspective; keeping the same target is allowed). A copy is not a card: it
    touches no zone and reserves no hand card, and simply ceases to exist after
    resolving. The copy carries the identical resolve tail, so its own copy
    rider fires recursively (mana-bounded). active_idx is set to the copier for
    the target choice, then restored to restore_idx (the resolving spell's
    controller) once the copy is on the stack."""
    state.active_idx = copier_idx

    def _on_target(state, target):
        captured = capture_any_target(state, target)
        push_to_stack(
            state, card_def, lambda s, cd: _chain_lightning_resolve_tail(s, captured, cd),
            reserves_hand_card=False,  # a copy, not the physical card
            targets=() if captured is None else (captured,),
        )
        state.active_idx = restore_idx  # copy pushed with copier as controller; restore

    resolution.begin_choose_any_target(
        state,
        lambda p: p.card_type == CardType.CREATURE and can_be_targeted(state, p, copier_idx),
        _on_target,
    )


def cast_chain_lightning(state, card_def):
    """{R} sorcery: Chain Lightning deals 3 damage to any target; then the copy
    rider (see _chain_lightning_resolve_tail). Target locked as the spell is
    put on the stack (precast_choice), like every other any-target burn; the
    card goes hand -> graveyard on resolution, ahead of the damage."""
    def _on_target(state, target):
        captured = capture_any_target(state, target)

        def _resolve(state, card_def):
            discard_from_hand_to_graveyard(state, card_def)  # itself -> graveyard first
            _chain_lightning_resolve_tail(state, captured, card_def)

        push_to_stack(state, card_def, _resolve, targets=() if captured is None else (captured,))

    resolution.begin_choose_any_target(
        state,
        lambda p: p.card_type == CardType.CREATURE and can_be_targeted(state, p, state.active_idx),
        _on_target,
    )


def faithless_looting_discard(state):
    """Draw two, then discard two -- shared by the normal cast and
    Flashback below (identical effect, only how the cost was paid
    differs)."""
    state.draw(2)
    resolution.begin_discard(state, 2, optional=False, on_complete=lambda s, _cards: None)


def cast_faithless_looting(state, card_def):
    discard_from_hand_to_graveyard(state, card_def)
    faithless_looting_discard(state)


def flashback_faithless_looting(state, inst):
    """Real Flashback cost is {2}{R} (unlike Dread Return/Lava Dart's free
    sacrifice-only Flashback) -- the mana is already paid by the generic
    flashback cost path (drl_env._actions_cast_altzone._flashback_execute) before this
    resolve ever runs, so by the time we get here it's "fully paid for" and
    pushes onto the stack immediately, not gated behind any further
    resolution.

    inst: the exact graveyard CardInstance being flashed back -- see
    black_cards.flashback_dread_return."""
    state.graveyard.remove(inst)  # leaves the graveyard the moment Flashback is chosen; exiled on resolution (702.34)
    push_to_stack(state, inst, lambda st, cd: faithless_looting_discard(st), reserves_hand_card=False, exiles_on_resolve=True)


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
    state.move_card(card_def, state.graveyard)
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
    state.move_card(permanent.card_def, state.graveyard)
    state.log_event(
        "zone_move", permanent=(permanent.card_def.name, permanent.slot), from_zone="battlefield",
        to_zone="graveyard", reason="sacrifice",
    )
    fire_sacrifice_triggers(state, state.active_idx, permanent.card_def)  # Gixian Infiltrator
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


def flashback_lava_dart(state, inst):
    """Flashback -- Sacrifice a Mountain: no mana component at all. Same "1
    damage to any target"; once the sacrifice is paid, the target is chosen
    and the effect pushed onto the stack. The card leaves the graveyard when
    Flashback is chosen and is EXILED as it resolves (exiles_on_resolve) --
    tracked in the exile zone, not the graveyard (702.34).

    inst: the exact graveyard CardInstance being flashed back -- see
    black_cards.flashback_dread_return."""
    state.graveyard.remove(inst)  # leaves the graveyard the moment Flashback is chosen; exiled on resolution
    resolution.begin_sacrifice(
        state, lambda p: p.card_def.name == "Mountain", 1,
        on_complete=lambda s, ok: _burn_choose_target_and_push(s, inst, 1, lambda st, cd: None, reserves_hand_card=False, exiles_on_resolve=True),
    )


def cast_end_the_festivities(state, card_def):
    """Real text (Innistrad: Crimson Vow, {R}): "End the Festivities deals 1
    damage to each opponent and each creature and planeswalker they
    control." NOT symmetric: only the OPPONENT's face and the OPPONENT's own
    board take damage -- this deck's own creatures are untouched. (No deck's
    pool has planeswalkers, so that clause never applies here.)"""
    discard_from_hand_to_graveyard(state, card_def)
    deal_damage_to_opponent(state, 1)
    for permanent in state.opponent.battlefield:
        if permanent.card_type == CardType.CREATURE:
            permanent.damage_marked += 1
    check_state_based_actions(state)


def cast_breath_weapon(state, card_def):
    """Real text: deals 2 damage to each NON-DRAGON creature. Green's
    Avenging Hunter (subtypes=("Dragon", "Ranger")) is a Dragon in this
    pool, so the filter is live, not vacuous -- it must be excluded on
    either battlefield, same symmetric-wipe shape otherwise (this deck's
    own creatures included, exactly like the real card)."""
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
            # A targeted spell can't be cast with no legal target: needs >=1
            # land on EITHER battlefield this player can target.
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
        # "As long as you control an artifact, this creature gets +1/+0 and
        # has haste." A conditional static self-boost (stats._static_self).
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
            # Unkicked: {R} (card_def.cast_cost, no per-mode override). Kicked:
            # {R}{R} (Kicker {R}), flagging the ETB to pump the team.
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
        # precast_choice: "any target" is locked in as the spell is cast
        # (drl_env._precast_choice_execute), not at resolution -- real Magic.
        "cast": {"resolve": lambda state, card_def: cast_lightning_bolt(state, card_def), "precast_choice": True},
        "pending_kinds": {"choose_any_target"},
    },
    EffectId.CHAIN_LIGHTNING: {
        # precast_choice: original target locked at cast, same as the burn above.
        # The copy rider adds pay_unless (the {R}{R} may-pay) + another
        # choose_any_target (the copier's new target); both drl_env-driven.
        "cast": {"resolve": lambda state, card_def: cast_chain_lightning(state, card_def), "precast_choice": True},
        "pending_kinds": {"choose_any_target", "pay_unless", "may_copy"},
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
        # Hard cast is precast_choice (target locked at cast); the alt_cast
        # resolve chooses its own target after the sacrifice cost and pushes
        # itself, so it needs no precast flag -- just the pending kind.
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

