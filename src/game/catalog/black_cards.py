"""Black-identity card catalog: BLACK_CARD_CATALOG (name -> CardDef) and
BLACK_EFFECT_REGISTRY (EffectId -> spec), unioned into game.CARD_DEFS /
EFFECT_REGISTRY by game/registry.py. Cost/type/oracle text is from
Scryfall; power/toughness is a design choice.

"Each opponent" effects (End the Festivities, Voldaren Epicure) are
non-targeted, via win_check.deal_damage_to_opponent. Cards with "target
player"/"target opponent" (Vampire's Kiss, Alms of the Vein, Mesmeric
Fiend, Balustrade Spy) capture a real target -- via
begin_choose_target_player, or the opponent's index directly when only
the opponent is a legal target -- at cast/activation or ETB-promotion
time."""

from .. import resolution
from ..cards import CardDef, CardType, EffectId, card_colors, card_subtypes, is_artifact
from ..effects.casting import (
    _log_target_fizzle, cast_permanent_from_hand, cast_targeting_creature, enters_battlefield, has_creature_target,
)
from ..effects.shared import affinity_reduction, discard_from_hand_to_graveyard, tap_for_cost
from ..effects.stack import push_to_stack
from ..effects.state_based import destroy_permanent, sacrifice_to_graveyard
from ..effects.tokens import BLOOD_TOKEN_CARD_DEF, CLUE_TOKEN_CARD_DEF, MAP_TOKEN_CARD_DEF, create_token
from ..effects.win_check import deal_damage_to_opponent, deal_damage_to_player, gain_life, lose_life

BLACK_CARD_CATALOG = {
    "Swamp": CardDef("Swamp", CardType.LAND, None, EffectId.SWAMP, basic=True, subtypes=("Swamp",)),
    "Vault of Whispers": CardDef("Vault of Whispers", CardType.LAND, None, EffectId.VAULT_OF_WHISPERS, artifact=True),  # land + artifact
    "Bojuka Bog": CardDef("Bojuka Bog", CardType.LAND, None, EffectId.BOJUKA_BOG),
    "Balustrade Spy": CardDef(
        "Balustrade Spy", CardType.CREATURE, {"generic": 3, "B": 1}, EffectId.BALUSTRADE_SPY, power=2, toughness=2,
    ),
    "Lotleth Giant": CardDef(
        "Lotleth Giant", CardType.CREATURE, {"generic": 6, "B": 1}, EffectId.LOTLETH_GIANT, power=5, toughness=5,
    ),
    # ETB exiles a nonland from target opponent's hand; LTB returns it (mesmeric_fiend_etb/_ltb).
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
    """Snuff Out's "nonblack creature" restriction; colorless counts as nonblack."""
    return "B" not in card_colors(permanent.card_def)


def _destroy_on_resolve(state, permanent):
    destroy_permanent(state, permanent)


def cast_cast_down(state, card_def):
    """{1}{B}: Destroy target nonlegendary creature."""
    cast_targeting_creature(state, card_def, _destroy_on_resolve)


def cast_snuff_out(state, card_def):
    """{3}{B} (or the 4-life alt cost below): Destroy target nonblack creature."""
    cast_targeting_creature(state, card_def, _destroy_on_resolve, eligible=_nonblack)


def _controls_a_swamp(state):
    return any("Swamp" in card_subtypes(p.card_def) for p in state.battlefield)


def snuff_out_alt_legal(state):
    """4-life alt cost is legal only with a Swamp, >= 4 life, and a legal nonblack target."""
    return _controls_a_swamp(state) and state.life_total >= 4 and has_creature_target(state, _nonblack)


def cast_snuff_out_alt(state, card_def):
    """Pay 4 life instead of {3}{B}, then target and resolve like the normal cast."""
    lose_life(state, 4, reason="snuff_out_alt")
    cast_snuff_out(state, card_def)


def _unexpected_fangs_resolve(state, permanent):
    """Puts a +1/+1 counter and a lifelink counter on the target."""
    permanent.counters["+1/+1"] = permanent.counters.get("+1/+1", 0) + 1
    permanent.counters["lifelink"] = permanent.counters.get("lifelink", 0) + 1


def cast_unexpected_fangs(state, card_def):
    cast_targeting_creature(state, card_def, _unexpected_fangs_resolve)


def _toxin_analysis_resolve(state, permanent):
    """Target creature gains deathtouch and lifelink until end of turn, then Investigate."""
    permanent.temp_keywords |= {"deathtouch", "lifelink"}
    create_token(state, CLUE_TOKEN_CARD_DEF)


def cast_toxin_analysis(state, card_def):
    cast_targeting_creature(state, card_def, _toxin_analysis_resolve)


def mill_until_land(state, permanent):
    """Balustrade Spy's ETB: target player mills cards off their library
    until a land is milled (target chosen at promotion)."""
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
    """Undergrowth ETB: 1 damage to the opponent per creature card in your graveyard."""
    creature_count = sum(1 for c in state.graveyard if c.card_type == CardType.CREATURE)
    deal_damage_to_opponent(state, creature_count)


def bojuka_bog_etb(state, permanent):
    """When Bojuka Bog enters, exile target player's graveyard (untracked
    exile). Target is chosen at promotion; "yourself" is always a legal
    choice, the opponent is a second option once one exists."""
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
    choose a nonland card from it to exile (tracked, linked to this Fiend;
    returned by mesmeric_fiend_ltb). The opponent is the only legal target
    (captured directly, not via begin_choose_target_player, since "target
    opponent" never includes yourself). No-op with no opponent."""
    if len(state.players) < 2:
        return
    opponent_idx = 1 - state.active_idx
    opponent = state.players[opponent_idx]

    def _resolve(state, card_def):
        def _on_chosen(state, chosen):
            if chosen is None:
                return  # no nonland card in hand
            card = chosen
            opponent.hand.remove(card)
            permanent.flags["mesmeric_exiled"] = (card, opponent_idx)  # returned by mesmeric_fiend_ltb
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
    its owner's hand. No-op if nothing was exiled."""
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
    """Choose the reanimation target (a graveyard creature card) as Dread
    Return is cast, then push the reanimation resolve. Fizzles if the
    target has left the graveyard by resolution, tracked by object
    identity so same-named copies are distinct. `to_graveyard`/
    `reserves_hand_card` control how Dread Return itself reaches the
    graveyard on resolution: from hand, or a no-op for Flashback."""
    def _on_chosen(state, chosen):
        captured = chosen  # exact graveyard instance, locked at cast

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
    """{2}{B}{B}: return target creature card from your graveyard to the battlefield."""
    _dread_return_choose_and_push(state, card_def, to_graveyard=discard_from_hand_to_graveyard, reserves_hand_card=True)


def flashback_dread_return(state, inst):
    """Flashback: sacrifice three creatures instead of {2}{B}{B}. Target is
    chosen after the sacrifice, so newly sacrificed creatures are eligible.
    Dread Return exiles afterward (untracked).

    inst: the exact graveyard CardInstance being flashed back, removed by
    object identity."""
    state.graveyard.remove(inst)  # exiled after resolution
    resolution.begin_sacrifice(
        state, lambda p: p.card_type == CardType.CREATURE, 3,
        on_complete=lambda s, ok: _dread_return_choose_and_push(
            s, inst, to_graveyard=lambda st, cd: None, reserves_hand_card=False, exiles_on_resolve=True,
        ) if ok else None,
    )


def madness_kitchen_imp(state, card_def):
    """Madness resolve: normal battlefield entry from exile, no hand step."""
    enters_battlefield(state, card_def)


def _vampires_kiss_resolve(state, player_idx):
    deal_damage_to_player(state, player_idx, 2)
    gain_life(state, 2)
    create_token(state, BLOOD_TOKEN_CARD_DEF)
    create_token(state, BLOOD_TOKEN_CARD_DEF)


def cast_vampires_kiss(state, card_def):
    """{1}{B}: Target player (any player, including yourself) loses 2 life,
    you gain 2 life, create two Blood tokens. Target locked at cast."""
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
    """{2}{B}: Target opponent loses 3 life and you gain 3 life. Madness {B}.
    Opponent captured directly (the only legal target), locked at cast."""
    opponent_idx = 1 - state.active_idx

    def _resolve(state, card_def):
        discard_from_hand_to_graveyard(state, card_def)
        _alms_of_the_vein_resolve(state, opponent_idx)

    push_to_stack(state, card_def, _resolve)


def madness_alms_of_the_vein(state, card_def):
    """Madness {B}: same effect, cast from exile. No-op if no opponent exists."""
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
    """Flashback {4}{B}, sacrifice an artifact or creature: draw two, then exile."""
    state.graveyard.remove(inst)  # exiled after resolution

    def _after_sac(state, _sacced):
        push_to_stack(state, inst, lambda st, cd: st.draw(2), reserves_hand_card=False, exiles_on_resolve=True)

    _sac_artifact_or_creature(state, _after_sac)


def blood_fountain_return(state, permanent):
    """{3}{B}, {T}, Sacrifice: Return up to two target creature cards from
    your graveyard to your hand. Targets are chosen as the ability
    activates, and return fizzles per-target if a target has since left
    the graveyard."""
    tap_for_cost(state, permanent)
    sacrifice_to_graveyard(state, permanent)

    def _on_targets(state, chosen):
        captured = list(chosen)  # exact graveyard instances, locked at activation

        def _resolve(state, card_def):
            survivors = [inst for inst in captured if inst in state.graveyard]
            if captured and not survivors:
                _log_target_fizzle(state, card_def, None)
                return
            for inst in survivors:
                _return_creature_from_graveyard(state, inst)

        push_to_stack(
            state, permanent.card_def, _resolve, reserves_hand_card=False, is_spell=False,
            targets=tuple(("graveyard_card", inst) for inst in captured),
        )

    resolution.begin_choose_up_to_graveyard(state, lambda c: c.card_type == CardType.CREATURE, 2, _on_targets)


def _return_creature_from_graveyard(state, chosen):
    if chosen is not None:
        state.graveyard.remove(chosen)
        state.hand.append(chosen.card_def)
        state.log_event("zone_move", card=chosen.name, from_zone="graveyard", to_zone="hand", reason="blood_fountain")


def refurbished_familiar_etb(state):
    """ETB: each opponent discards a card; if they can't, you draw a card."""
    if len(state.players) < 2:
        return
    caster = state.active_idx
    opp_idx = 1 - caster
    if not state.players[opp_idx].hand:
        state.draw(1)
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
        "cost_reduction": affinity_reduction,  # affinity for artifacts
        "keywords": {"flying"},
        "etb_trigger": lambda state, permanent: refurbished_familiar_etb(state),
        "pending_kinds": {"discard"},
    },
    EffectId.GURMAG_ANGLER: {
        # Delve N (0..6) exiles N graveyard cards to pay {N} of the generic cost.
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
        # Whenever you sacrifice another permanent, put a +1/+1 counter on this creature.
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
        "alt_cast": {  # pay 4 life instead of mana cost (needs a Swamp)
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
        "etb_targets": True,  # target player chosen at promotion
        "pending_kinds": {"choose_target_player"},
    },
    EffectId.BALUSTRADE_SPY: {
        "cast": {"resolve": lambda state, card_def: cast_permanent_from_hand(state, card_def)},
        "etb_trigger": lambda state, permanent: mill_until_land(state, permanent),
        "etb_targets": True,  # target player chosen at promotion
        "pending_kinds": {"choose_target_player"},
        "keywords": {"flying"},  # 2/3 Flying
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
        "etb_targets": True,  # target opponent captured at promotion; only legal candidate
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
        "cast": {"resolve": lambda state, card_def: cast_permanent_from_hand(state, card_def)},
        "haste": True,
        "keywords": {"flying"},
        "madness": {"cost": {"B": 1}, "resolve": lambda state, card_def: madness_kitchen_imp(state, card_def)},
        "pending_kinds": {"madness_decision", "order_triggers"},  # order_triggers: 2+ Madness cards discarded at once
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
