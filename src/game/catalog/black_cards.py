"""Black-identity card catalog: every card whose real mana cost is
mono-black (or, for lands with no cost, whose only mana output is black).
Every card's cost/type/oracle-text below is a direct Scryfall pull,
except creature power/toughness, which is a design choice, not Scryfall
data. Real Jagged Barrens/End the Festivities/Vampire's Kiss/Voldaren
Epicure/Alms of the Vein reference "each opponent"/"target opponent" --
all of these route through win_check.deal_damage_to_opponent, which hits
the opponent's real per-player life_total."""

from .. import resolution
from ..cards import CardDef, CardType, EffectId
from ..effects.casting import (
    _log_target_fizzle, cast_permanent_from_hand, cast_targeting_creature, enters_battlefield, has_creature_target,
)
from ..effects.shared import affinity_reduction, card_colors, card_subtypes, discard_from_hand_to_graveyard, is_artifact
from ..effects.stack import push_ability_to_stack, push_to_stack
from ..effects.state_based import destroy_permanent, sacrifice_to_graveyard
from ..effects.tokens import BLOOD_TOKEN_CARD_DEF, CLUE_TOKEN_CARD_DEF, MAP_TOKEN_CARD_DEF, create_token
from ..effects.win_check import deal_damage_to_opponent, gain_life, lose_life

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
    creature. It can't be regenerated (a no-op -- regeneration isn't
    modeled)."""
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


def bojuka_bog_etb(state):
    """When Bojuka Bog enters, exile target player's graveyard. "Exile a
    graveyard" just empties it here -- exile is untracked, same convention
    as Relic of Progenitus' own graveyard-exile (game.catalog.colorless_
    cards). A REAL target-player choice (begin_choose_target_player), same
    as Relic's exile ability: "yourself" is always legal (true even alone in
    a 1-player game), the opponent becomes a second option once one exists,
    and the model picks explicitly.

    Runs as this ETB triggered ability RESOLVES off the stack (ETBs now go
    through the trigger queue -- casting.enters_battlefield). The target is
    chosen here at resolution rather than locked at stack-placement: "target
    player" is always legal (no way to make a player an illegal target), and
    nothing in this pool manipulates a graveyard at instant speed in response
    to the trigger, so the two orderings are outcome-identical -- no
    observable difference, same reasoning begin_choose_graveyard_card's own
    docstring already applies to Relic's cross-player pick."""
    def _on_player_chosen(state, idx):
        target = state.players[idx]
        exiled = [c.name for c in target.graveyard]
        target.graveyard.clear()
        state.log_event("graveyard_exiled", target_player_idx=idx, exiled=exiled)

    resolution.begin_choose_target_player(state, _on_player_chosen)


def mesmeric_fiend_etb(state, permanent):
    """When Mesmeric Fiend enters, target opponent reveals their hand and you
    choose a nonland card from it. Exile that card -- TRACKED, linked to THIS
    exact Fiend on its own flags; the matching "when this creature leaves the
    battlefield, return the exiled card to its owner's hand" is mesmeric_
    fiend_ltb below.

    Needs a real opponent (2-player) -- "target opponent" has no legal target
    in a 1-player config, so the ETB does nothing there. The nonland card is
    picked from the opponent's hand by reusing begin_choose_graveyard_card (a
    generic "choose one card by name from this list matching a predicate",
    despite its graveyard-flavored name) with the opponent's hand as the list.
    Runs as this ETB resolves off the stack (trigger queue), choosing at
    resolution -- the same ETB convention Pinnacle Kill-Ship / Masked Vandal
    already follow."""
    if len(state.players) < 2:
        return
    opponent = state.opponent
    opponent_idx = state.players.index(opponent)

    def _on_chosen(state, name):
        if name is None:
            return  # no nonland card in the opponent's hand
        card = next(c for c in opponent.hand if c.name == name)
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


def mesmeric_fiend_ltb(state, permanent):
    """When Mesmeric Fiend leaves the battlefield, return the exiled card to
    its owner's hand. Reads the linkage mesmeric_fiend_etb stored on this exact
    permanent -- the object survives leaving the battlefield (state_based.
    _queue_leave_triggers / resolution.execute_sacrifice_option carry it here).
    A no-op if nothing was exiled (an empty/all-land opponent hand at ETB, or a
    1-player game where the ETB never fired)."""
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

    Duplicate-target note: graveyard cards are shared CardDef objects, so two
    copies of the same creature card are indistinguishable -- the fizzle
    check is "no copy of this card remains", not per-physical-copy identity
    (which the shared-CardDef graveyard cannot represent)."""
    def _on_chosen(state, name):
        captured = next((c for c in state.graveyard if c.name == name), None) if name is not None else None

        def _resolve(state, card_def):
            to_graveyard(state, card_def)
            if captured is None or captured not in state.graveyard:
                _log_target_fizzle(state, card_def, (name, "graveyard") if name is not None else None)
                return
            state.graveyard.remove(captured)
            enters_battlefield(state, captured, from_zone="graveyard")

        push_to_stack(state, card_def, _resolve, reserves_hand_card=reserves_hand_card, exiles_on_resolve=exiles_on_resolve)

    resolution.begin_choose_graveyard_card(state, lambda c: c.card_type == CardType.CREATURE, _on_chosen)


def cast_dread_return(state, card_def):
    """{2}{B}{B}: return target creature card from your graveyard to the
    battlefield. precast_choice -- the target is locked as the spell is cast
    (real Magic), the reanimation waits on the stack, and Dread Return itself
    goes to the graveyard when it resolves."""
    _dread_return_choose_and_push(state, card_def, to_graveyard=discard_from_hand_to_graveyard, reserves_hand_card=True)


def flashback_dread_return(state, card_def):
    """Flashback -- Sacrifice three creatures instead of {2}{B}{B}. Same
    reanimation; the target is chosen as the spell is put on the stack (after
    the sacrifice cost is paid), not at resolution. The newly sacrificed
    creatures are in the graveyard by then, so they're eligible targets (a
    real interaction). Dread Return is exiled afterward (untracked, per its
    own text), so its resolve makes no further zone move for itself."""
    state.graveyard.remove(card_def)  # leaves the graveyard the moment Flashback is chosen; exiled after (untracked)
    resolution.begin_sacrifice(
        state, lambda p: p.card_type == CardType.CREATURE, 3,
        on_complete=lambda s, ok: _dread_return_choose_and_push(
            s, card_def, to_graveyard=lambda st, cd: None, reserves_hand_card=False, exiles_on_resolve=True,
        ) if ok else None,
    )


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


def flashback_eviscerators_insight(state, card_def):
    """Flashback {4}{B} (mana paid by the graveyard-cast machinery) + the same
    sacrifice-an-artifact-or-creature additional cost: draw two, then exile."""
    state.graveyard.remove(card_def)  # leaves gy; exiled after resolution

    def _after_sac(state, _sacced):
        push_to_stack(state, card_def, lambda st, cd: st.draw(2), reserves_hand_card=False, exiles_on_resolve=True)

    _sac_artifact_or_creature(state, _after_sac)


def blood_fountain_return(state, permanent):
    """{3}{B}, {T}, Sacrifice: Return up to two target creature cards from your
    graveyard to your hand. (Blood Fountain has no dies-trigger of its own.)"""
    sacrifice_to_graveyard(state, permanent)  # cost

    def _effect(st):
        def _first(st, name1):
            if name1 is None:
                return
            _return_creature_from_graveyard(st, name1)
            resolution.begin_choose_graveyard_card(
                st, lambda c: c.card_type == CardType.CREATURE,
                lambda st2, name2: _return_creature_from_graveyard(st2, name2) if name2 else None, optional=True,
            )
        resolution.begin_choose_graveyard_card(st, lambda c: c.card_type == CardType.CREATURE, _first, optional=True)

    push_ability_to_stack(state, permanent.card_def, _effect)


def _return_creature_from_graveyard(state, name):
    found = next((c for c in state.graveyard if c.name == name), None)
    if found is not None:
        state.graveyard.remove(found)
        state.hand.append(found)
        state.log_event("zone_move", card=found.name, from_zone="graveyard", to_zone="hand", reason="blood_fountain")


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
        # Delve only -- "Cast Gurmag Angler (delve N)" for N in 0..6 (delve 0
        # is the plain {6}{B} cast). Each exiles N graveyard cards to pay {N}
        # of the generic (drl_env delve loop).
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
        "etb_trigger": lambda state, permanent: bojuka_bog_etb(state),
        "pending_kinds": {"choose_target_player"},
    },
    EffectId.BALUSTRADE_SPY: {
        "cast": {"resolve": lambda state, card_def: cast_permanent_from_hand(state, card_def)},
        "etb_trigger": lambda state, permanent: mill_until_land(state),
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
        "pending_kinds": {"choose_graveyard_card", "sacrifice"},
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
        "cast": {"resolve": lambda state, card_def: cast_vampires_kiss(state, card_def)},
    },
    EffectId.ALMS_OF_THE_VEIN: {
        "cast": {"resolve": lambda state, card_def: cast_alms_of_the_vein(state, card_def)},
        "madness": {"cost": {"B": 1}, "resolve": lambda state, card_def: madness_alms_of_the_vein(state, card_def)},
        "pending_kinds": {"madness_decision", "order_triggers"},  # see EffectId.KITCHEN_IMP's own comment
    },
}


if __name__ == "__main__":
    # ponytail self-check (run via `python -m game.catalog.black_cards` from
    # src/): Dread Return -- target locked at cast, effect on the stack, and
    # the 608.2b fizzle when the chosen creature card leaves the graveyard
    # before it resolves (reachable via opponent graveyard hate).
    import contextlib
    import io

    from ..effects.stack import resolve_top_of_stack
    from ..state import GameState, PlayerState

    dr = CardDef("Dread Return", CardType.SORCERY, {"generic": 2, "B": 2}, EffectId.DREAD_RETURN)
    grizzly = CardDef("Grizzly Bears", CardType.CREATURE, {"G": 1}, EffectId.FILLER, power=2, toughness=2)

    # (a) hard cast reanimates the chosen creature card
    state = GameState(on_the_play=True, players=[PlayerState(True), PlayerState(False)])
    state.hand = [dr]
    state.graveyard = [grizzly]
    cast_dread_return(state, dr)  # precast: begins the graveyard-target choice
    assert state.pending_resolution["kind"] == "choose_graveyard_card"
    resolution.execute_choose_graveyard_card_option(state, "Grizzly Bears")
    assert state.hand == [] and len(state.stack) == 1  # Dread Return LEFT hand at cast, now on the stack
    resolve_top_of_stack(state)
    assert dr in state.graveyard  # Dread Return resolved -> graveyard
    assert any(p.card_def is grizzly for p in state.battlefield) and grizzly not in state.graveyard  # reanimated

    # (b) fizzle: the chosen card leaves the graveyard before resolution
    state = GameState(on_the_play=True, players=[PlayerState(True), PlayerState(False)])
    state.hand = [dr]
    state.graveyard = [grizzly]
    cast_dread_return(state, dr)
    resolution.execute_choose_graveyard_card_option(state, "Grizzly Bears")
    state.graveyard.remove(grizzly)  # exiled by graveyard hate before Dread Return resolves
    _log = io.StringIO()
    with contextlib.redirect_stdout(_log):
        resolve_top_of_stack(state)
    assert "fizzle" in _log.getvalue().lower()
    assert dr in state.graveyard  # Dread Return still goes to the graveyard
    assert not any(p.card_def is grizzly for p in state.battlefield)  # nothing reanimated

    print("black_cards.py Dread Return target-at-cast + fizzle self-check: OK")

    # Bojuka Bog: enters tapped, and its ETB "exile target player's
    # graveyard" is a real target-player choice resolved off the stack (ETBs
    # go through the trigger queue now). 2-player so targeting the OPPONENT
    # empties THEIR graveyard, never the active player's own.
    from ..effects.triggers import promote_triggers_to_stack
    from ..resolution import execute_choose_target_player_option

    state = GameState(on_the_play=True, players=[PlayerState(True), PlayerState(False)])
    state.players[0].graveyard = [CardDef("Mine", CardType.CREATURE, None, EffectId.FILLER)]
    state.players[1].graveyard = [CardDef("Theirs", CardType.CREATURE, None, EffectId.FILLER)]
    bog = CardDef("Bojuka Bog", CardType.LAND, None, EffectId.BOJUKA_BOG)
    enters_battlefield(state, bog, from_zone="hand")
    bog_perm = next(p for p in state.battlefield if p.card_def.name == "Bojuka Bog")
    assert bog_perm.tapped  # enters tapped
    # ETB queued (faithful timing), not run inline -- promote + resolve opens the target choice.
    assert state.pending_resolution is None
    assert [e["type"] for e in state.trigger_queue] == ["etb"]
    promote_triggers_to_stack(state)
    resolve_top_of_stack(state)
    assert state.pending_resolution["kind"] == "choose_target_player"
    execute_choose_target_player_option(state, 1)  # target the opponent
    assert state.players[1].graveyard == []  # their graveyard exiled
    assert [c.name for c in state.players[0].graveyard] == ["Mine"]  # own graveyard untouched

    print("black_cards.py Bojuka Bog ETB (exile target player's graveyard) self-check: OK")

    # Mesmeric Fiend: ETB exiles a nonland from target opponent's hand
    # (tracked, linked to this Fiend); its leaves-the-battlefield trigger --
    # the engine's first -- returns that card when the Fiend leaves, whether
    # it DIES (state-based) or is SACRIFICED (resolution.execute_sacrifice_
    # option), the two ways a creature leaves in this pool.
    from ..effects.state_based import check_state_based_actions
    from ..resolution import (
        begin_sacrifice, choose_graveyard_card_options, execute_choose_graveyard_card_option, execute_sacrifice_option,
    )

    fiend_def = CardDef("Mesmeric Fiend", CardType.CREATURE, {"generic": 1, "B": 1}, EffectId.MESMERIC_FIEND, power=1, toughness=1)

    def _enter_fiend_and_exile():
        st = GameState(on_the_play=True, players=[PlayerState(True), PlayerState(False)])
        a_spell = CardDef("Their Spell", CardType.SORCERY, {"B": 1}, EffectId.FILLER)
        a_land = CardDef("Their Land", CardType.LAND, None, EffectId.SWAMP)
        st.players[1].hand = [a_spell, a_land]
        fiend = enters_battlefield(st, fiend_def, from_zone="hand")
        assert [e["type"] for e in st.trigger_queue] == ["etb"]
        promote_triggers_to_stack(st)
        resolve_top_of_stack(st)  # ETB resolves -> choose a nonland from the opponent's hand
        assert st.pending_resolution["kind"] == "choose_graveyard_card"
        assert choose_graveyard_card_options(st) == ["Their Spell"]  # the LAND is excluded (nonland only)
        execute_choose_graveyard_card_option(st, "Their Spell")
        assert [c.name for c in st.players[1].hand] == ["Their Land"]  # nonland exiled from their hand
        assert fiend.flags["mesmeric_exiled"][0] is a_spell  # tracked, linked to this Fiend
        return st, fiend, a_spell

    # (a) the Fiend DIES -> LTB returns the exiled card to its owner's hand.
    state, fiend, a_spell = _enter_fiend_and_exile()
    fiend.damage_marked = fiend_def.extra["toughness"]  # lethal
    check_state_based_actions(state)
    assert fiend not in state.players[0].battlefield  # dead
    assert [e["type"] for e in state.trigger_queue] == ["ltb"]  # LTB queued on leave
    promote_triggers_to_stack(state)
    resolve_top_of_stack(state)
    assert a_spell in state.players[1].hand  # returned to its OWNER's hand
    assert "mesmeric_exiled" not in fiend.flags  # linkage consumed

    # (b) the Fiend is SACRIFICED (e.g. to Dread Return's Flashback) -> same LTB.
    state, fiend, a_spell = _enter_fiend_and_exile()
    begin_sacrifice(state, lambda p: p.card_def.name == "Mesmeric Fiend", 1, on_complete=lambda s, ok: None)
    execute_sacrifice_option(state, "Mesmeric Fiend")
    assert fiend not in state.players[0].battlefield
    assert [e["type"] for e in state.trigger_queue] == ["ltb"]
    promote_triggers_to_stack(state)
    resolve_top_of_stack(state)
    assert a_spell in state.players[1].hand

    # (c) 1-player (no opponent to target): ETB does nothing, no exile.
    solo = GameState(on_the_play=True)
    solo_fiend = enters_battlefield(solo, fiend_def, from_zone="hand")
    promote_triggers_to_stack(solo)
    resolve_top_of_stack(solo)
    assert solo.pending_resolution is None and "mesmeric_exiled" not in solo_fiend.flags

    print("black_cards.py Mesmeric Fiend ETB/LTB self-check: OK")

    # --- G3 removal & tricks ---
    from .. import registry
    from ..effects.state_based import cleanup_step
    from ..effects.stats import has_keyword, lifelink_count, permanent_power, permanent_toughness
    from ..state import Permanent

    def _g3_two():
        return GameState(on_the_play=True, players=[PlayerState(True), PlayerState(False)])

    # Cast Down destroys any creature; the spell fizzles if the target's gone.
    state = _g3_two()
    victim = Permanent(CardDef("Kitchen Imp", CardType.CREATURE, {"generic": 3, "B": 1}, EffectId.KITCHEN_IMP, power=2, toughness=2))
    victim.slot = 1
    state.players[1].battlefield = [victim]
    state.players[0].hand = [registry.CARD_DEFS["Cast Down"]]
    cast_cast_down(state, registry.CARD_DEFS["Cast Down"])
    resolution.execute_choose_any_target_creature(state, 1, "Kitchen Imp", 1)
    resolve_top_of_stack(state)
    assert victim not in state.players[1].battlefield
    print("black_cards.py Cast Down self-check: OK")

    # Snuff Out: nonblack only, and the 4-life alt cost (needs a Swamp).
    assert _nonblack(Permanent(CardDef("R Creature", CardType.CREATURE, {"R": 1}, EffectId.FILLER, power=1, toughness=1)))
    assert not _nonblack(Permanent(CardDef("B Creature", CardType.CREATURE, {"B": 1}, EffectId.FILLER, power=1, toughness=1)))
    state = _g3_two()
    swamp = Permanent(CardDef("Swamp", CardType.LAND, None, EffectId.SWAMP, basic=True, subtypes=("Swamp",)))
    swamp.slot = 1
    tgt = Permanent(CardDef("R Creature", CardType.CREATURE, {"R": 1}, EffectId.FILLER, power=1, toughness=1))
    tgt.slot = 1
    state.players[0].battlefield = [swamp]
    state.players[1].battlefield = [tgt]
    state.players[0].hand = [registry.CARD_DEFS["Snuff Out"]]
    state.players[0].life_total = 20
    assert snuff_out_alt_legal(state)
    cast_snuff_out_alt(state, registry.CARD_DEFS["Snuff Out"])
    assert state.players[0].life_total == 16  # paid 4 life
    resolution.execute_choose_any_target_creature(state, 1, "R Creature", 1)
    resolve_top_of_stack(state)
    assert tgt not in state.players[1].battlefield
    print("black_cards.py Snuff Out (nonblack + 4-life alt) self-check: OK")

    # Unexpected Fangs: +1/+1 counter + lifelink counter (grants lifelink).
    state = _g3_two()
    c = Permanent(CardDef("Bear", CardType.CREATURE, {"G": 1}, EffectId.FILLER, power=1, toughness=1))
    c.slot = 1
    state.players[0].battlefield = [c]
    state.players[0].hand = [registry.CARD_DEFS["Unexpected Fangs"]]
    cast_unexpected_fangs(state, registry.CARD_DEFS["Unexpected Fangs"])
    resolution.execute_choose_any_target_creature(state, 0, "Bear", 1)
    resolve_top_of_stack(state)
    assert permanent_power(state, c) == 2 and permanent_toughness(state, c) == 2
    assert has_keyword(state, c, "lifelink") and lifelink_count(state, c) == 1
    print("black_cards.py Unexpected Fangs self-check: OK")

    # Toxin Analysis: grants deathtouch+lifelink until EOT + a Clue; cleanup
    # clears the temp keywords.
    state = _g3_two()
    c = Permanent(CardDef("Bear", CardType.CREATURE, {"G": 1}, EffectId.FILLER, power=1, toughness=1))
    c.slot = 1
    state.players[0].battlefield = [c]
    state.players[0].hand = [registry.CARD_DEFS["Toxin Analysis"]]
    cast_toxin_analysis(state, registry.CARD_DEFS["Toxin Analysis"])
    resolution.execute_choose_any_target_creature(state, 0, "Bear", 1)
    resolve_top_of_stack(state)
    assert has_keyword(state, c, "deathtouch") and has_keyword(state, c, "lifelink")
    assert any(p.card_def.name == "Clue" for p in state.players[0].battlefield)
    cleanup_step(state)
    assert not has_keyword(state, c, "deathtouch") and not has_keyword(state, c, "lifelink")
    print("black_cards.py Toxin Analysis (deathtouch/lifelink + Clue) self-check: OK")

    # --- G7 ---
    import drl_env

    # Refurbished Familiar affinity ({3}{B}, -1 per artifact you control).
    state = GameState(on_the_play=True)
    state.battlefield = [Permanent(registry.CARD_DEFS["Great Furnace"]), Permanent(registry.CARD_DEFS["Great Furnace"])]
    eff = drl_env._effective_cast_cost(state, registry.CARD_DEFS["Refurbished Familiar"])
    assert eff["generic"] == 1, eff  # 3 - 2 artifacts
    # ETB: the opponent discards a card of their choice.
    state = GameState(on_the_play=True, players=[PlayerState(True), PlayerState(False)])
    state.active_idx = 0
    state.players[1].hand = [CardDef("Opp1", CardType.LAND, None, EffectId.SWAMP), CardDef("Opp2", CardType.LAND, None, EffectId.SWAMP)]
    refurbished_familiar_etb(state)
    assert state.pending_resolution["kind"] == "discard" and state.active_idx == 1
    resolution.execute_discard_option(state, "Opp1")
    assert len(state.players[1].hand) == 1 and state.active_idx == 0
    # ETB with the opponent's hand empty: you draw instead.
    state = GameState(on_the_play=True, players=[PlayerState(True), PlayerState(False)])
    state.active_idx = 0
    state.players[1].hand = []
    state.players[0].library = [CardDef("Top", CardType.LAND, None, EffectId.SWAMP)]
    refurbished_familiar_etb(state)
    assert len(state.players[0].hand) == 1  # drew (opponent couldn't discard)
    print("black_cards.py Refurbished Familiar (affinity + ETB) self-check: OK")

    # Gurmag Angler Delve: delve N exiles N graveyard cards + pays {6-N}{B}.
    from ..cards import CardDef as _CD
    dl = [("Gurmag Angler", 4), ("Swamp", 8)]
    byname = {a[0]: (a[1], a[2]) for a in drl_env.build_action_table(dl, registry.EFFECT_REGISTRY, pending_kinds=registry.derive_pending_kinds(dl))}
    from ..turn import Phase as _Phase
    state = GameState(on_the_play=True)
    state.phase = _Phase.MAIN1
    state.turn_player_idx = 0
    state.active_idx = 0
    state.hand = [registry.CARD_DEFS["Gurmag Angler"]]
    state.graveyard = [_CD("g1", CardType.INSTANT, {"U": 1}, EffectId.FILLER), _CD("g2", CardType.INSTANT, {"U": 1}, EffectId.FILLER)]
    state.battlefield = [Permanent(registry.CARD_DEFS["Swamp"]) for _ in range(5)]  # {6}{B} minus delve 2 = {4}{B} = 5 mana
    legal, execute = byname["Cast Gurmag Angler (delve 2)"]
    assert legal(state)
    execute(state)
    from ..resolution import execute_choose_graveyard_card_option as _egc
    _egc(state, "g1")
    _egc(state, "g2")
    from ..mana import execute_pool_spend, pool_spend_options, execute_tap_cost_option, tap_cost_options
    _guard = 0
    while state.pending_resolution is not None and state.pending_resolution["kind"] == "pay_cost":
        _guard += 1
        assert _guard < 30
        taps = tap_cost_options(state)
        if taps:
            n, cc, f = taps[0]
            execute_tap_cost_option(state, n, cc, f)
        else:
            execute_pool_spend(state, pool_spend_options(state)[0])
    resolve_top_of_stack(state)
    assert any(p.card_def.name == "Gurmag Angler" for p in state.battlefield) and state.graveyard == []
    print("black_cards.py Gurmag Angler (Delve) self-check: OK")

    # --- G8 sac engines ---
    from ..effects.state_based import sacrifice_to_graveyard as _sac
    from ..effects.triggers import promote_triggers_to_stack as _prom
    from ..resolution import execute_choose_permanent_option as _ecp

    def _drive8(s):
        _prom(s)
        while s.stack:
            resolve_top_of_stack(s)

    # Gixian Infiltrator: +1/+1 whenever you sacrifice another permanent.
    state = GameState(on_the_play=True)
    gix = Permanent(registry.CARD_DEFS["Gixian Infiltrator"])
    gix.slot = 1
    fodder = Permanent(registry.CARD_DEFS["Ichor Wellspring"])
    fodder.slot = 1
    state.battlefield = [gix, fodder]
    state.library = [CardDef("x", CardType.LAND, None, EffectId.SWAMP, basic=True)]
    _sac(state, fodder)
    assert gix.counters.get("+1/+1") == 1
    print("black_cards.py Gixian Infiltrator (sac -> +1/+1) self-check: OK")

    # Fanatical Offering: sac an artifact/creature -> draw 2 + Map token.
    state = GameState(on_the_play=True)
    state.hand = [registry.CARD_DEFS["Fanatical Offering"]]
    state.library = [CardDef(f"n{i}", CardType.LAND, None, EffectId.SWAMP, basic=True) for i in range(5)]
    fodder = Permanent(CardDef("Fodder", CardType.CREATURE, {"B": 1}, EffectId.FILLER, power=1, toughness=1))
    fodder.slot = 1
    state.battlefield = [fodder]
    cast_fanatical_offering(state, registry.CARD_DEFS["Fanatical Offering"])
    _ecp(state, "Fodder", 1)
    _drive8(state)
    assert any(p.card_def.name == "Map" for p in state.battlefield) and len(state.hand) == 2
    print("black_cards.py Fanatical Offering (sac -> draw 2 + Map) self-check: OK")

    # Reckoner's Bargain: gain life = sacrificed permanent's mana value + draw 2.
    state = GameState(on_the_play=True)
    state.hand = [registry.CARD_DEFS["Reckoner's Bargain"]]
    state.library = [CardDef(f"m{i}", CardType.LAND, None, EffectId.SWAMP, basic=True) for i in range(5)]
    big = Permanent(CardDef("Big", CardType.CREATURE, {"generic": 4, "B": 1}, EffectId.FILLER, power=3, toughness=3))  # MV 5
    big.slot = 1
    state.battlefield = [big]
    cast_reckoners_bargain(state, registry.CARD_DEFS["Reckoner's Bargain"])
    _ecp(state, "Big", 1)
    _drive8(state)
    assert state.life_total == 25 and len(state.hand) == 2  # +5 MV, drew 2
    print("black_cards.py Reckoner's Bargain (gain MV + draw 2) self-check: OK")
