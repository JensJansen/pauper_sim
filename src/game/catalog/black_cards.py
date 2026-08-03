"""Black-identity card catalog: every card whose real mana cost is
mono-black (or, for lands with no cost, whose only mana output is black).
Every card's cost/type/oracle-text below is a direct Scryfall pull,
except creature power/toughness, which is a design choice, not Scryfall
data. Real End the Festivities/Voldaren Epicure reference "each opponent"
(non-targeted -- these route through win_check.deal_damage_to_opponent,
which hits the opponent's real per-player life_total). Vampire's Kiss/Alms
of the Vein/Mesmeric Fiend/Balustrade Spy are genuinely TARGETED ("target
player"/"target opponent"): each captures its target (a real
begin_choose_target_player choice, or -- for the opponent-restricted cards
-- the opponent's index directly, since no player-hexproof/protection/
removal-from-game mechanic exists anywhere in this engine to make them an
illegal choice) at cast/activation or ETB-promotion time, same convention
as Bojuka Bog's own "exile target player's graveyard"."""

from .. import resolution
from ..cards import CardDef, CardType, EffectId
from ..effects.casting import (
    _log_target_fizzle, cast_permanent_from_hand, cast_targeting_creature, enters_battlefield, has_creature_target,
)
from ..effects.shared import affinity_reduction, card_colors, card_subtypes, discard_from_hand_to_graveyard, is_artifact
from ..effects.stack import push_to_stack
from ..effects.state_based import destroy_permanent, sacrifice_to_graveyard
from ..effects.tokens import BLOOD_TOKEN_CARD_DEF, CLUE_TOKEN_CARD_DEF, MAP_TOKEN_CARD_DEF, create_token
from ..effects.win_check import deal_damage_to_opponent, deal_damage_to_player, gain_life, lose_life

BLACK_CARD_CATALOG = {
    "Swamp": CardDef("Swamp", CardType.LAND, None, EffectId.SWAMP, basic=True, subtypes=("Swamp",)),
    # Artifact land: played as a land, but also an artifact (affinity/
    # metalcraft/artifact-sac read extra["artifact"]).
    "Vault of Whispers": CardDef("Vault of Whispers", CardType.LAND, None, EffectId.VAULT_OF_WHISPERS, artifact=True),
    "Bojuka Bog": CardDef("Bojuka Bog", CardType.LAND, None, EffectId.BOJUKA_BOG),
    "Balustrade Spy": CardDef(
        "Balustrade Spy", CardType.CREATURE, {"generic": 3, "B": 1}, EffectId.BALUSTRADE_SPY, power=2, toughness=2,
    ),
    "Lotleth Giant": CardDef(
        "Lotleth Giant", CardType.CREATURE, {"generic": 6, "B": 1}, EffectId.LOTLETH_GIANT, power=5, toughness=5,
    ),
    # ETB exiles a nonland from target opponent's hand (tracked); its LTB
    # returns that card when the Fiend leaves the battlefield -- the first
    # leaves-the-battlefield trigger in the engine (mesmeric_fiend_etb/_ltb).
    "Mesmeric Fiend": CardDef(
        "Mesmeric Fiend", CardType.CREATURE, {"generic": 1, "B": 1}, EffectId.MESMERIC_FIEND, power=1, toughness=1,
    ),
    "Dread Return": CardDef("Dread Return", CardType.SORCERY, {"generic": 2, "B": 2}, EffectId.DREAD_RETURN),
    "Kitchen Imp": CardDef(
        "Kitchen Imp", CardType.CREATURE, {"generic": 3, "B": 1}, EffectId.KITCHEN_IMP, power=2, toughness=2,
    ),
    "Vampire's Kiss": CardDef("Vampire's Kiss", CardType.SORCERY, {"generic": 1, "B": 1}, EffectId.VAMPIRES_KISS),
    "Alms of the Vein": CardDef("Alms of the Vein", CardType.SORCERY, {"generic": 2, "B": 1}, EffectId.ALMS_OF_THE_VEIN),

    # --- G7: grixis_affinity / jund_wildfire ---
    "Refurbished Familiar": CardDef("Refurbished Familiar", CardType.CREATURE, {"generic": 3, "B": 1}, EffectId.REFURBISHED_FAMILIAR, power=2, toughness=1, artifact=True),
    "Gurmag Angler": CardDef("Gurmag Angler", CardType.CREATURE, {"generic": 6, "B": 1}, EffectId.GURMAG_ANGLER, power=5, toughness=5),

    # --- G8: sac engines (grixis_affinity / jund_wildfire) ---
    "Blood Fountain": CardDef("Blood Fountain", CardType.ARTIFACT, {"B": 1}, EffectId.BLOOD_FOUNTAIN, sac_ability_cost={"generic": 3, "B": 1}),
    "Gixian Infiltrator": CardDef("Gixian Infiltrator", CardType.CREATURE, {"generic": 1, "B": 1}, EffectId.GIXIAN_INFILTRATOR, power=2, toughness=1),
    "Fanatical Offering": CardDef("Fanatical Offering", CardType.INSTANT, {"generic": 1, "B": 1}, EffectId.FANATICAL_OFFERING),
    "Eviscerator's Insight": CardDef("Eviscerator's Insight", CardType.INSTANT, {"generic": 1, "B": 1}, EffectId.EVISCERATORS_INSIGHT),
    "Reckoner's Bargain": CardDef("Reckoner's Bargain", CardType.INSTANT, {"generic": 1, "B": 1}, EffectId.RECKONERS_BARGAIN),

    # --- G3 removal & tricks (mono-black) ---
    "Cast Down": CardDef("Cast Down", CardType.INSTANT, {"generic": 1, "B": 1}, EffectId.CAST_DOWN),
    "Snuff Out": CardDef("Snuff Out", CardType.INSTANT, {"generic": 3, "B": 1}, EffectId.SNUFF_OUT),
    "Unexpected Fangs": CardDef("Unexpected Fangs", CardType.INSTANT, {"generic": 1, "B": 1}, EffectId.UNEXPECTED_FANGS),
    "Toxin Analysis": CardDef("Toxin Analysis", CardType.INSTANT, {"B": 1}, EffectId.TOXIN_ANALYSIS),
}


def _nonblack(permanent):
    """Snuff Out's "nonblack creature" target restriction -- colorless (a
    Devoid creature like Writhing Chrysalis) counts as nonblack."""
    return "B" not in card_colors(permanent.card_def)


def _destroy_on_resolve(state, permanent):
    destroy_permanent(state, permanent)


def cast_cast_down(state, card_def):
    """{1}{B}: Destroy target nonlegendary creature. No legendary creature
    exists in this pool, so "nonlegendary" is every creature (default
    eligible)."""
    cast_targeting_creature(state, card_def, _destroy_on_resolve)


def cast_snuff_out(state, card_def):
    """{3}{B} (or the 4-life alt cost below): Destroy target nonblack
    creature. It can't be regenerated -- a no-op, since no card in this
    engine ever grants regeneration."""
    cast_targeting_creature(state, card_def, _destroy_on_resolve, eligible=_nonblack)


def _controls_a_swamp(state):
    # "control a Swamp" -- any land with the Swamp subtype (basic Swamp, and
    # the Island Swamp duals Contaminated Aquifer / Ice Tunnel).
    return any("Swamp" in card_subtypes(p.card_def) for p in state.battlefield)


def snuff_out_alt_legal(state):
    """The 4-life alt cost: legal only if you control a Swamp, have >= 4 life
    to pay, and there's a legal nonblack creature to target (a targeted spell
    needs a legal target regardless of how it's paid for)."""
    return _controls_a_swamp(state) and state.life_total >= 4 and has_creature_target(state, _nonblack)


def cast_snuff_out_alt(state, card_def):
    """Alt cost: pay 4 life instead of {3}{B} (real "you may pay 4 life
    rather than pay this spell's mana cost"). Pay the life now (a cost, at
    cast time), then the spell targets + resolves exactly like the normal
    cast -- same cast_snuff_out body, which opens the target choice and
    pushes the destroy onto the stack."""
    lose_life(state, 4, reason="snuff_out_alt")
    cast_snuff_out(state, card_def)


def _unexpected_fangs_resolve(state, permanent):
    """Put a +1/+1 counter AND a lifelink counter on the target. The lifelink
    counter grants lifelink (stats.creature_keywords reads counters["lifelink"])."""
    permanent.counters["+1/+1"] = permanent.counters.get("+1/+1", 0) + 1
    permanent.counters["lifelink"] = permanent.counters.get("lifelink", 0) + 1


def cast_unexpected_fangs(state, card_def):
    cast_targeting_creature(state, card_def, _unexpected_fangs_resolve)


def _toxin_analysis_resolve(state, permanent):
    """Target creature gains deathtouch and lifelink until end of turn
    (temp_keywords, cleared at cleanup_step); then Investigate (a Clue
    token). Investigate is inside the target-legal branch: if the only
    target is gone, the spell doesn't resolve at all, so no Clue is made."""
    permanent.temp_keywords |= {"deathtouch", "lifelink"}
    create_token(state, CLUE_TOKEN_CARD_DEF)


def cast_toxin_analysis(state, card_def):
    cast_targeting_creature(state, card_def, _toxin_analysis_resolve)


def mill_until_land(state, permanent):
    """Balustrade Spy's ETB: target player reveals cards from the top of
    THEIR library until they reveal a land card, then mills everything
    revealed (the land included) to THEIR graveyard. Real "target player"
    choice (begin_choose_target_player), same shape as Bojuka Bog's own ETB
    just above -- yourself is always legal, the opponent becomes a second
    option once one exists. Target-at-promotion (etb_targets: True) --
    never fizzles (a player is always a legal target), but surfaces the
    choice one priority window earlier, same rationale as Bojuka Bog. If
    the target's library empties before a land turns up, everything left
    mills and the library simply ends up empty -- draw() (not this
    function) is what detects and flags actually running out."""
    def _on_player_chosen(state, idx):
        def _resolve(state, card_def):
            target = state.players[idx]
            milled = []
            while target.library:
                card = target.library.pop(0)
                state.move_card(card, target.graveyard)
                milled.append(card.name)
                if card.card_type == CardType.LAND:
                    break
            state.log_event("balustrade_spy_mill", target_player_idx=idx, milled=milled)

        push_to_stack(state, permanent.card_def, _resolve, reserves_hand_card=False, is_spell=False)

    resolution.begin_choose_target_player(state, _on_player_chosen)


def lotleth_giant_etb(state):
    """Undergrowth ETB: 1 damage to the opponent per creature card in your
    graveyard."""
    creature_count = sum(1 for c in state.graveyard if c.card_type == CardType.CREATURE)
    deal_damage_to_opponent(state, creature_count)


def bojuka_bog_etb(state, permanent):
    """When Bojuka Bog enters, exile target player's graveyard. "Exile a
    graveyard" just empties it here -- exile is untracked, same convention
    as Relic of Progenitus' own graveyard-exile (game.catalog.colorless_
    cards). A REAL target-player choice (begin_choose_target_player), same
    as Relic's exile ability: "yourself" is always legal (true even alone in
    a 1-player game), the opponent becomes a second option once one exists,
    and the model picks explicitly.

    Target-at-promotion (project_targeted_triggered_abilities): the target
    player is chosen as the ability goes on the stack, and the exile happens at
    resolution. "Target player" is always legal (no way to make a player an
    illegal target), so this never fizzles -- but choosing at promotion still
    surfaces the decision one priority window earlier, which can inform play,
    so it's modeled faithfully rather than collapsed to resolution."""
    def _on_player_chosen(state, idx):
        def _resolve(state, card_def):
            target = state.players[idx]
            exiled = [c.name for c in target.graveyard]
            target.graveyard.clear()
            state.log_event("graveyard_exiled", target_player_idx=idx, exiled=exiled)

        push_to_stack(state, permanent.card_def, _resolve, reserves_hand_card=False, is_spell=False)

    resolution.begin_choose_target_player(state, _on_player_chosen)


def mesmeric_fiend_etb(state, permanent):
    """When Mesmeric Fiend enters, target opponent reveals their hand and you
    choose a nonland card from it. Exile that card -- TRACKED, linked to THIS
    exact Fiend on its own flags; the matching "when this creature leaves the
    battlefield, return the exiled card to its owner's hand" is mesmeric_
    fiend_ltb below.

    Real "target opponent": the opponent is the only ever-legal candidate in
    this strictly-2-player engine (no player-hexproof/protection/removal-
    from-game mechanic exists anywhere here to make them illegal -- same
    vacuous-restriction class as Cast Down's "nonlegendary"/Snuff Out's
    "can't be regenerated"), so the target is captured directly rather than
    routed through begin_choose_target_player, which would wrongly also
    offer "yourself" (real "target opponent" never does). Needs a real
    opponent (2-player) -- no legal target in a 1-player config, so the ETB
    does nothing there.

    Target-at-promotion (etb_targets: True, 603.3d): this whole function runs
    AT PROMOTION (registry.etb_trigger invoked directly by triggers.
    _place_trigger_groups for a targeting ETB), with active_idx already set
    to the Fiend's controller -- so the opponent is captured here, and the
    reveal/exile effect (including which nonland card gets picked, a modal
    choice, not itself a target) is deferred onto the stack via push_to_stack,
    same shape as Bojuka Bog's own ETB."""
    if len(state.players) < 2:
        return
    opponent_idx = 1 - state.active_idx
    opponent = state.players[opponent_idx]

    def _resolve(state, card_def):
        def _on_chosen(state, chosen):
            if chosen is None:
                return  # no nonland card in the opponent's hand
            card = chosen  # the exact hand card (hand is DEFERRED -- still CardDefs, interned)
            opponent.hand.remove(card)
            # Tracked exile, linked to this exact Fiend -- returned by its LTB.
            permanent.flags["mesmeric_exiled"] = (card, opponent_idx)
            state.log_event(
                "zone_move", card=card.name, from_zone="hand", to_zone="exile_mesmeric",
                owner_idx=opponent_idx, source=(permanent.card_def.name, permanent.slot),
            )

        resolution.begin_choose_graveyard_card(
            state, lambda c: c.card_type != CardType.LAND, _on_chosen, graveyard=opponent.hand,
        )

    push_to_stack(state, permanent.card_def, _resolve, reserves_hand_card=False, is_spell=False)


def mesmeric_fiend_ltb(state, permanent):
    """When Mesmeric Fiend leaves the battlefield, return the exiled card to
    its owner's hand. Reads the linkage mesmeric_fiend_etb stored on this exact
    permanent -- the object survives leaving the battlefield (state_based.
    _queue_leave_triggers / sacrifice_to_graveyard carry it here). A no-op if
    nothing was exiled (an empty/all-land opponent hand at ETB, or a 1-player
    game where the ETB never fired)."""
    exiled = permanent.flags.pop("mesmeric_exiled", None)
    if exiled is None:
        return
    card, owner_idx = exiled
    state.players[owner_idx].hand.append(card)
    state.log_event(
        "zone_move", card=card.name, from_zone="exile_mesmeric", to_zone="hand", owner_idx=owner_idx,
        reason="mesmeric_leaves",
    )


def _dread_return_choose_and_push(state, card_def, to_graveyard, reserves_hand_card, exiles_on_resolve=False):
    """Choose the reanimation target -- a creature card in your graveyard --
    as Dread Return is put on the stack, lock it, and push the reanimation
    resolve. On resolution the chosen card returns from the graveyard to the
    battlefield; Dread Return FIZZLES if that card has left the graveyard by
    then (608.2b -- reachable via opponent graveyard hate, e.g. Relic of
    Progenitus exiling graveyards). Dread Return itself, a sorcery, is never a
    legal creature target. `to_graveyard`/`reserves_hand_card` say how the
    Dread Return card reaches the graveyard on resolution: from hand (hard
    cast) or a no-op (Flashback -- exiled, untracked).

    Target is locked at cast BY OBJECT IDENTITY: the exact chosen graveyard
    instance is captured, and the fizzle check at resolution is "is that exact
    instance still in the graveyard" -- so two same-named copies are now
    distinct (one can leave while the other stays), per real MTG 400.7/608.2b."""
    def _on_chosen(state, chosen):
        captured = chosen  # the exact graveyard instance, locked at cast

        def _resolve(state, card_def):
            to_graveyard(state, card_def)
            if captured is None or captured not in state.graveyard:
                _log_target_fizzle(state, card_def, (captured.name, "graveyard") if captured is not None else None)
                return
            state.graveyard.remove(captured)
            enters_battlefield(state, captured.card_def, from_zone="graveyard")

        push_to_stack(state, card_def, _resolve, reserves_hand_card=reserves_hand_card, exiles_on_resolve=exiles_on_resolve)

    resolution.begin_choose_graveyard_card(state, lambda c: c.card_type == CardType.CREATURE, _on_chosen)


def cast_dread_return(state, card_def):
    """{2}{B}{B}: return target creature card from your graveyard to the
    battlefield. precast_choice -- the target is locked as the spell is cast
    (real Magic), the reanimation waits on the stack, and Dread Return itself
    goes to the graveyard when it resolves."""
    _dread_return_choose_and_push(state, card_def, to_graveyard=discard_from_hand_to_graveyard, reserves_hand_card=True)


def flashback_dread_return(state, inst):
    """Flashback -- Sacrifice three creatures instead of {2}{B}{B}. Same
    reanimation; the target is chosen as the spell is put on the stack (after
    the sacrifice cost is paid), not at resolution. The newly sacrificed
    creatures are in the graveyard by then, so they're eligible targets (a
    real interaction). Dread Return is exiled afterward (untracked, per its
    own text), so its resolve makes no further zone move for itself.

    inst: the exact graveyard CardInstance being flashed back (resolved once at
    the action boundary, drl_env._actions._graveyard_instance) -- removed by
    object identity, never by a name re-lookup."""
    state.graveyard.remove(inst)  # leaves the graveyard the moment Flashback is chosen; exiled after (untracked)
    resolution.begin_sacrifice(
        state, lambda p: p.card_type == CardType.CREATURE, 3,
        on_complete=lambda s, ok: _dread_return_choose_and_push(
            s, inst, to_graveyard=lambda st, cd: None, reserves_hand_card=False, exiles_on_resolve=True,
        ) if ok else None,
    )


def madness_kitchen_imp(state, card_def):
    """Kitchen Imp -- Flying, haste. Madness {B}. No ETB at all (real
    Oracle text has no triggered ability beyond Madness itself). Madness
    resolve for a creature: execute_madness_cast has already pulled the
    card out of exile, so this just needs the normal battlefield-entry
    path -- never touches hand, unlike a normal cast."""
    enters_battlefield(state, card_def)


def _vampires_kiss_resolve(state, player_idx):
    deal_damage_to_player(state, player_idx, 2)
    gain_life(state, 2)
    create_token(state, BLOOD_TOKEN_CARD_DEF)
    create_token(state, BLOOD_TOKEN_CARD_DEF)


def cast_vampires_kiss(state, card_def):
    """{1}{B}: Target player loses 2 life and you gain 2 life. Create two
    Blood tokens. No Madness on this one (only Fiery Temper/Alms of the Vein
    have it). Real "target player" -- ANY player, including yourself
    (unusual but legal, unlike Alms of the Vein's opponent-restricted
    version) -- a genuine begin_choose_target_player choice, locked at CAST
    (precast_choice), not resolution; "target player" is always legal (at
    minimum, yourself), so this never fizzles."""
    def _on_player_chosen(state, idx):
        def _resolve(state, card_def):
            discard_from_hand_to_graveyard(state, card_def)
            _vampires_kiss_resolve(state, idx)

        push_to_stack(state, card_def, _resolve)

    resolution.begin_choose_target_player(state, _on_player_chosen)


def _alms_of_the_vein_resolve(state, opponent_idx):
    deal_damage_to_player(state, opponent_idx, 3)
    gain_life(state, 3)


def cast_alms_of_the_vein(state, card_def):
    """{2}{B}: Target opponent loses 3 life and you gain 3 life. Madness
    {B}. Real "target opponent": the opponent is the only ever-legal
    candidate in this strictly-2-player engine (see mesmeric_fiend_etb's own
    docstring for why that's captured directly rather than through
    begin_choose_target_player), locked at CAST (precast_choice) rather than
    resolution. The registry's own extra_legal gates casting on an opponent
    actually existing (a mandatory single target needs one to be legal at
    all, 601.2c)."""
    opponent_idx = 1 - state.active_idx

    def _resolve(state, card_def):
        discard_from_hand_to_graveyard(state, card_def)
        _alms_of_the_vein_resolve(state, opponent_idx)

    push_to_stack(state, card_def, _resolve)


def madness_alms_of_the_vein(state, card_def):
    """Madness {B}: same target-opponent capture, from exile -- never
    touches hand (see red_cards.madness_fiery_temper's own convention). No
    extra_legal gate on the madness path itself (offered whenever the card
    is discarded, regardless of opponent count, matching this engine's
    existing "no legal opponent -> no-op" tolerance for the rare 1-player
    edge case, same as mesmeric_fiend_etb's own guard)."""
    opponent_idx = 1 - state.active_idx

    def _resolve(state, card_def):
        state.move_card(card_def, state.graveyard)
        if len(state.players) > 1:
            _alms_of_the_vein_resolve(state, opponent_idx)

    push_to_stack(state, card_def, _resolve, reserves_hand_card=False)


def _has_sac_fodder(state):
    return any(p.card_type == CardType.CREATURE or is_artifact(p.card_def) for p in state.battlefield)


def _mana_value(card_def):
    return sum(card_def.cast_cost.values()) if card_def.cast_cost else 0


def _sac_artifact_or_creature(state, on_sacrificed):
    """Choose an artifact or creature you control and sacrifice it (the
    additional cost shared by Fanatical Offering / Eviscerator's Insight /
    Reckoner's Bargain), then on_sacrificed(state, sacrificed_card_def)."""
    def _on_chosen(state, choice):
        name, slot = choice
        perm = next(p for p in state.battlefield if p.card_def.name == name and p.slot == slot)
        sacced = perm.card_def
        sacrifice_to_graveyard(state, perm)  # cost paid now -> fires dies + sacrifice triggers
        on_sacrificed(state, sacced)

    resolution.begin_choose_permanent(
        state, lambda p: p.card_type == CardType.CREATURE or is_artifact(p.card_def), _on_chosen,
    )


def cast_fanatical_offering(state, card_def):
    """{1}{B}, sacrifice an artifact or creature: Draw two cards and create a
    Map token."""
    def _after_sac(state, _sacced):
        def _resolve(st, cd):
            discard_from_hand_to_graveyard(st, cd)
            st.draw(2)
            create_token(st, MAP_TOKEN_CARD_DEF)
        push_to_stack(state, card_def, _resolve)

    _sac_artifact_or_creature(state, _after_sac)


def cast_reckoners_bargain(state, card_def):
    """{1}{B}, sacrifice an artifact or creature: gain life equal to the
    sacrificed permanent's mana value, draw two."""
    def _after_sac(state, sacced):
        mv = _mana_value(sacced)

        def _resolve(st, cd):
            discard_from_hand_to_graveyard(st, cd)
            gain_life(st, mv)
            st.draw(2)
        push_to_stack(state, card_def, _resolve)

    _sac_artifact_or_creature(state, _after_sac)


def cast_eviscerators_insight(state, card_def):
    """{1}{B}, sacrifice an artifact or creature: Draw two cards."""
    def _after_sac(state, _sacced):
        def _resolve(st, cd):
            discard_from_hand_to_graveyard(st, cd)
            st.draw(2)
        push_to_stack(state, card_def, _resolve)

    _sac_artifact_or_creature(state, _after_sac)


def flashback_eviscerators_insight(state, inst):
    """Flashback {4}{B} (mana paid by the graveyard-cast machinery) + the same
    sacrifice-an-artifact-or-creature additional cost: draw two, then exile.

    inst: the exact graveyard CardInstance being flashed back -- see
    flashback_dread_return."""
    state.graveyard.remove(inst)  # leaves gy; exiled after resolution

    def _after_sac(state, _sacced):
        push_to_stack(state, inst, lambda st, cd: st.draw(2), reserves_hand_card=False, exiles_on_resolve=True)

    _sac_artifact_or_creature(state, _after_sac)


def blood_fountain_return(state, permanent):
    """{3}{B}, {T}, Sacrifice: Return up to two target creature cards from your
    graveyard to your hand. (Blood Fountain has no dies-trigger of its own.)

    The {3}{B} is paid before this resolve fires (drl_env._activate_execute's
    own begin_pay_cost call); {T} and Sacrifice are paid here, in that order,
    same as any real activation -- tapping first, THEN sacrificing (tapping a
    permanent that's about to leave the battlefield is otherwise inert, but
    modeling both cost components keeps this faithful to the real cost).

    Faithful targeting (matching Rooftop Percher's own "up to two target
    cards from graveyards" shape): the up-to-2 targets are chosen NOW, as
    the ability is activated -- BEFORE it goes on the stack -- captured by
    object identity, and the return fizzles per-target at resolution:
    returning every still-present target, doing nothing only if ALL chosen
    targets have already left the graveyard by then (608.2c)."""
    permanent.tapped = True  # cost ({T})
    sacrifice_to_graveyard(state, permanent)  # cost (Sacrifice)

    def _on_targets(state, chosen):
        captured = list(chosen)  # exact graveyard instances, locked as the ability is put on the stack

        def _resolve(state, card_def):
            survivors = [inst for inst in captured if inst in state.graveyard]
            if captured and not survivors:
                _log_target_fizzle(state, card_def, None)  # every target left the graveyard -> nothing returns
                return
            for inst in survivors:
                _return_creature_from_graveyard(state, inst)

        push_to_stack(
            state, permanent.card_def, _resolve, reserves_hand_card=False, is_spell=False,
            targets=tuple(("graveyard_card", inst) for inst in captured),
        )

    resolution.begin_choose_up_to_graveyard(state, lambda c: c.card_type == CardType.CREATURE, 2, _on_targets)


def _return_creature_from_graveyard(state, chosen):
    # chosen: the exact graveyard instance (or None if the optional pick declined).
    if chosen is not None:
        state.graveyard.remove(chosen)
        state.hand.append(chosen.card_def)  # hand is DEFERRED -- CardDefs
        state.log_event("zone_move", card=chosen.name, from_zone="graveyard", to_zone="hand", reason="blood_fountain")


def refurbished_familiar_etb(state):
    """"When this creature enters, each opponent discards a card. For each
    opponent who can't, you draw a card." 2-player: the one opponent discards
    a card of THEIR choice (active_idx flipped to them for the discard) if
    their hand is non-empty; otherwise (they can't) the caster draws one.
    No opponent (1-player) -> nothing."""
    if len(state.players) < 2:
        return
    caster = state.active_idx
    opp_idx = 1 - caster
    if not state.players[opp_idx].hand:
        state.draw(1)  # opponent can't discard -> you draw
        return

    def _restore(s, _discarded):
        s.active_idx = caster

    state.active_idx = opp_idx
    resolution.begin_discard(state, 1, optional=False, on_complete=_restore)


BLACK_EFFECT_REGISTRY = {
    EffectId.SWAMP: {
        "mana": ("fixed", "B"),
    },
    EffectId.VAULT_OF_WHISPERS: {
        "mana": ("fixed", "B"),
    },
    EffectId.REFURBISHED_FAMILIAR: {
        "cast": {"resolve": lambda state, card_def: cast_permanent_from_hand(state, card_def)},
        "cost_reduction": affinity_reduction,  # Affinity for artifacts
        "keywords": {"flying"},
        "etb_trigger": lambda state, permanent: refurbished_familiar_etb(state),
        "pending_kinds": {"discard"},
    },
    EffectId.GURMAG_ANGLER: {
        # Delve only -- "Cast Gurmag Angler", then a generic "Delve N" (N in
        # 0..6; delve 0 is the plain {6}{B} cast) choice exiles N graveyard
        # cards to pay {N} of the generic (drl_env delve loop).
        "delve": {"max": 6, "resolve": lambda state, card_def: cast_permanent_from_hand(state, card_def)},
        "pending_kinds": {"choose_graveyard_card"},
    },
    EffectId.BLOOD_FOUNTAIN: {
        "cast": {"resolve": lambda state, card_def: cast_permanent_from_hand(state, card_def)},
        "etb_trigger": lambda state, permanent: create_token(state, BLOOD_TOKEN_CARD_DEF),
        "activated_abilities": {
            "return": {"cost_key": "sac_ability_cost", "resolve": lambda state, permanent: blood_fountain_return(state, permanent)},
        },
        "pending_kinds": {"choose_graveyard_card"},
    },
    EffectId.GIXIAN_INFILTRATOR: {
        "cast": {"resolve": lambda state, card_def: cast_permanent_from_hand(state, card_def)},
        # "Whenever you sacrifice another permanent, put a +1/+1 counter on
        # this creature." Fired by shared.fire_sacrifice_triggers from every
        # sacrifice path (it only iterates the sacrificer's OTHER battlefield
        # permanents, so "another" is automatic).
        "on_sacrifice": lambda state, permanent, sacrificed_card_def: permanent.counters.__setitem__(
            "+1/+1", permanent.counters.get("+1/+1", 0) + 1,
        ),
    },
    EffectId.FANATICAL_OFFERING: {
        "cast": {
            "resolve": lambda state, card_def: cast_fanatical_offering(state, card_def),
            "extra_legal": lambda state: _has_sac_fodder(state),
            "precast_choice": True,  # the sacrifice (additional cost) is paid as it's cast
        },
        "pending_kinds": {"choose_permanent"},
    },
    EffectId.RECKONERS_BARGAIN: {
        "cast": {
            "resolve": lambda state, card_def: cast_reckoners_bargain(state, card_def),
            "extra_legal": lambda state: _has_sac_fodder(state),
            "precast_choice": True,
        },
        "pending_kinds": {"choose_permanent"},
    },
    EffectId.EVISCERATORS_INSIGHT: {
        "cast": {
            "resolve": lambda state, card_def: cast_eviscerators_insight(state, card_def),
            "extra_legal": lambda state: _has_sac_fodder(state),
            "precast_choice": True,
        },
        "flashback": {
            "cost": {"generic": 4, "B": 1},
            "legal": lambda state: _has_sac_fodder(state),  # the sac additional cost must be payable
            "resolve": lambda state, card_def: flashback_eviscerators_insight(state, card_def),
        },
        "pending_kinds": {"choose_permanent"},
    },
    EffectId.CAST_DOWN: {
        "cast": {
            "resolve": lambda state, card_def: cast_cast_down(state, card_def),
            "extra_legal": lambda state: has_creature_target(state),
            "precast_choice": True,  # target locked at cast
        },
        "pending_kinds": {"choose_any_target"},
    },
    EffectId.SNUFF_OUT: {
        "cast": {
            "resolve": lambda state, card_def: cast_snuff_out(state, card_def),
            "extra_legal": lambda state: has_creature_target(state, _nonblack),
            "precast_choice": True,
        },
        "alt_cast": {  # "you may pay 4 life rather than pay this spell's mana cost" (needs a Swamp)
            "extra_legal": lambda state: snuff_out_alt_legal(state),
            "resolve": lambda state, card_def: cast_snuff_out_alt(state, card_def),
        },
        "pending_kinds": {"choose_any_target"},
    },
    EffectId.UNEXPECTED_FANGS: {
        "cast": {
            "resolve": lambda state, card_def: cast_unexpected_fangs(state, card_def),
            "extra_legal": lambda state: has_creature_target(state),
            "precast_choice": True,
        },
        "pending_kinds": {"choose_any_target"},
    },
    EffectId.TOXIN_ANALYSIS: {
        "cast": {
            "resolve": lambda state, card_def: cast_toxin_analysis(state, card_def),
            "extra_legal": lambda state: has_creature_target(state),
            "precast_choice": True,
        },
        "pending_kinds": {"choose_any_target"},
    },
    EffectId.BOJUKA_BOG: {
        "mana": ("fixed", "B"),
        "enters_tapped": True,
        "etb_trigger": lambda state, permanent: bojuka_bog_etb(state, permanent),
        "etb_targets": True,  # target player chosen at promotion (never fizzles, but surfaces the choice early)
        "pending_kinds": {"choose_target_player"},
    },
    EffectId.BALUSTRADE_SPY: {
        "cast": {"resolve": lambda state, card_def: cast_permanent_from_hand(state, card_def)},
        "etb_trigger": lambda state, permanent: mill_until_land(state, permanent),
        "etb_targets": True,  # target player chosen at promotion (never fizzles, but surfaces the choice early)
        "pending_kinds": {"choose_target_player"},
        "keywords": {"flying"},  # real Balustrade Spy is 2/3 Flying (P/T is this file's own design choice per the module docstring; Flying is not)
    },
    EffectId.LOTLETH_GIANT: {
        "cast": {"resolve": lambda state, card_def: cast_permanent_from_hand(state, card_def)},
        "etb_trigger": lambda state, permanent: lotleth_giant_etb(state),
    },
    EffectId.MESMERIC_FIEND: {
        "cast": {"resolve": lambda state, card_def: cast_permanent_from_hand(state, card_def)},
        # ETB: exile a nonland from target opponent's hand (tracked, linked to
        # this Fiend); LTB: return it to its owner's hand.
        "etb_trigger": lambda state, permanent: mesmeric_fiend_etb(state, permanent),
        "etb_targets": True,  # target opponent captured at promotion (603.3d); the only ever-legal candidate here
        "ltb_trigger": lambda state, permanent: mesmeric_fiend_ltb(state, permanent),
        "pending_kinds": {"choose_graveyard_card"},
    },
    EffectId.DREAD_RETURN: {
        "cast": {
            "resolve": lambda state, card_def: cast_dread_return(state, card_def),
            "extra_legal": lambda state: any(c.card_type == CardType.CREATURE for c in state.graveyard),
            "precast_choice": True,  # target locked at cast (real Magic), reanimation waits on the stack
        },
        "flashback": {
            "legal": lambda state: sum(1 for p in state.battlefield if p.card_type == CardType.CREATURE) >= 3,
            "resolve": lambda state, card_def: flashback_dread_return(state, card_def),
        },
        "pending_kinds": {"choose_graveyard_card", "choose_permanent"},
    },
    EffectId.KITCHEN_IMP: {
        # Real text: Flying, haste.
        "cast": {"resolve": lambda state, card_def: cast_permanent_from_hand(state, card_def)},
        "haste": True,
        "keywords": {"flying"},
        "madness": {"cost": {"B": 1}, "resolve": lambda state, card_def: madness_kitchen_imp(state, card_def)},
        # order_triggers: reachable the
        # instant 2+ Madness cards get discarded at once (Faithless
        # Looting's discard-2) -- both trigger simultaneously and need a
        # real placement-order choice.
        "pending_kinds": {"madness_decision", "order_triggers"},
    },
    EffectId.VAMPIRES_KISS: {
        "cast": {
            "resolve": lambda state, card_def: cast_vampires_kiss(state, card_def),
            "precast_choice": True,  # target player locked at cast
        },
        "pending_kinds": {"choose_target_player"},
    },
    EffectId.ALMS_OF_THE_VEIN: {
        "cast": {
            "resolve": lambda state, card_def: cast_alms_of_the_vein(state, card_def),
            "extra_legal": lambda state: len(state.players) > 1,  # a mandatory "target opponent" needs one to exist
            "precast_choice": True,  # target opponent locked at cast
        },
        "madness": {"cost": {"B": 1}, "resolve": lambda state, card_def: madness_alms_of_the_vein(state, card_def)},
        "pending_kinds": {"madness_decision", "order_triggers"},  # see EffectId.KITCHEN_IMP's own comment
    },
}
