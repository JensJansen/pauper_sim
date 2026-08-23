"""Green-identity card catalog: cards whose real mana cost is mono-green (or,
for a land, whose only mana output is green). Costs/types/oracle text are
Scryfall data; creature power/toughness is a design choice. Same
GREEN_CARD_CATALOG / GREEN_EFFECT_REGISTRY shape as every color file, unioned
by game/registry.py."""

from .. import resolution
from ..cards import CardDef, CardType, EffectId, card_subtypes
from ..effects.casting import (
    _log_target_fizzle, capture_any_target, cast_aura, cast_permanent_from_hand, enters_battlefield,
    target_still_legal,
)
from ..effects.shared import (
    any_creature_on_either_battlefield, any_land_on_either_battlefield, discard_from_hand_to_graveyard,
    find_and_remove_by_name, find_to_hand,
 shuffle_library, set_tapped, tap_for_cost,
)
from ..effects.stack import push_ability_to_stack, push_to_stack
from ..effects.state_based import check_state_based_actions
from ..effects.stats import can_be_targeted, controller_idx, enchantment_count, has_keyword, permanent_power, permanent_toughness
from ..effects.tokens import (
    ELDRAZI_SPAWN_TOKEN_CARD_DEF, FOOD_TOKEN_CARD_DEF, activate_eldrazi_spawn_sac, activate_food_sac, create_token,
)
from ..effects import undercity
from ..effects.win_check import deal_damage_to_opponent, gain_life
from ..mana import COLORS, discount_departing_source, tap_summoning_locked

GREEN_CARD_CATALOG = {
    "Forest": CardDef("Forest", CardType.LAND, None, EffectId.FOREST, basic=True, subtypes=("Forest",)),
    # Enters tapped unless you control 3+ other Forests; untapped entry
    # creates a Food token. Has the Forest subtype itself.
    "Gingerbread Cabin": CardDef(
        "Gingerbread Cabin", CardType.LAND, None, EffectId.GINGERBREAD_CABIN, subtypes=("Forest",),
    ),
    "Generous Ent": CardDef(
        "Generous Ent", CardType.CREATURE, {"generic": 5, "G": 1}, EffectId.GENEROUS_ENT,
        forestcycling_cost={"generic": 1}, power=5, toughness=5,
    ),
    "Masked Vandal": CardDef(
        "Masked Vandal", CardType.CREATURE, {"generic": 1, "G": 1}, EffectId.MASKED_VANDAL, power=1, toughness=3,
        # AUTHORIZED SIMPLIFICATION: real Masked Vandal is a Changeling (every
        # creature type simultaneously); narrowed here to just "Elf", the only
        # creature type any card in this pool ever counts (Priest of Titania /
        # Wellwisher / Timberwatch Elf) -- approved 2026-08-03 (repo owner).
        subtypes=("Elf",),
    ),
    "Saruli Caretaker": CardDef(
        "Saruli Caretaker", CardType.CREATURE, {"G": 1}, EffectId.SARULI_CARETAKER, defender=True, power=1, toughness=1,
    ),
    "Overgrown Battlement": CardDef(
        "Overgrown Battlement", CardType.CREATURE, {"generic": 1, "G": 1}, EffectId.OVERGROWN_BATTLEMENT, defender=True,
        power=0, toughness=4,
    ),
    "Wall of Roots": CardDef(
        "Wall of Roots", CardType.CREATURE, {"generic": 1, "G": 1}, EffectId.WALL_OF_ROOTS, defender=True,
        power=0, toughness=5,
    ),
    "Sagu Wildling": CardDef("Sagu Wildling", CardType.SORCERY, {"G": 1}, EffectId.ROOST_SEEK),
    "Gatecreeper Vine": CardDef(
        "Gatecreeper Vine", CardType.CREATURE, {"generic": 1, "G": 1}, EffectId.GATECREEPER_VINE, defender=True,
        power=0, toughness=3,
    ),
    "Nyxborn Hydra": CardDef("Nyxborn Hydra", CardType.CREATURE, {"G": 1}, EffectId.NYXBORN_HYDRA, power=0, toughness=1),
    "Quirion Ranger": CardDef("Quirion Ranger", CardType.CREATURE, {"G": 1}, EffectId.QUIRION_RANGER, power=1, toughness=1, subtypes=("Elf",)),
    "Winding Way": CardDef("Winding Way", CardType.SORCERY, {"generic": 1, "G": 1}, EffectId.WINDING_WAY),
    "Lead the Stampede": CardDef("Lead the Stampede", CardType.SORCERY, {"generic": 2, "G": 1}, EffectId.LEAD_THE_STAMPEDE),
    "Land Grant": CardDef("Land Grant", CardType.SORCERY, {"generic": 1, "G": 1}, EffectId.LAND_GRANT),
    "Crop Rotation": CardDef("Crop Rotation", CardType.INSTANT, {"G": 1}, EffectId.CROP_ROTATION),
    "Ancient Stirrings": CardDef("Ancient Stirrings", CardType.SORCERY, {"G": 1}, EffectId.ANCIENT_STIRRINGS),
    "Bramble Wurm": CardDef(
        "Bramble Wurm", CardType.CREATURE, {"generic": 6, "G": 1}, EffectId.BRAMBLE_WURM, power=7, toughness=6,
        gy_ability_cost={"generic": 2, "G": 1},
    ),

    # --- boggles deck ---
    "Gladecover Scout": CardDef(
        "Gladecover Scout", CardType.CREATURE, {"G": 1}, EffectId.GLADECOVER_SCOUT, power=1, toughness=1,
    ),
    "Silhana Ledgewalker": CardDef(
        "Silhana Ledgewalker", CardType.CREATURE, {"generic": 1, "G": 1}, EffectId.SILHANA_LEDGEWALKER,
        power=1, toughness=1,
    ),
    "Rancor": CardDef("Rancor", CardType.ENCHANTMENT, {"G": 1}, EffectId.RANCOR),
    "Ancestral Mask": CardDef("Ancestral Mask", CardType.ENCHANTMENT, {"generic": 2, "G": 1}, EffectId.ANCESTRAL_MASK),
    "Utopia Sprawl": CardDef("Utopia Sprawl", CardType.ENCHANTMENT, {"G": 1}, EffectId.UTOPIA_SPRAWL),
    "Abundant Growth": CardDef("Abundant Growth", CardType.ENCHANTMENT, {"G": 1}, EffectId.ABUNDANT_GROWTH),
    "Malevolent Rumble": CardDef(
        "Malevolent Rumble", CardType.SORCERY, {"generic": 1, "G": 1}, EffectId.MALEVOLENT_RUMBLE,
    ),
    # One-sided fight: your creature deals its power to an opponent's
    # creature (trample overflow to its controller). Requires 2 players.
    "Ram Through": CardDef("Ram Through", CardType.INSTANT, {"generic": 1, "G": 1}, EffectId.RAM_THROUGH),

    # --- G3: jund_wildfire ---
    "Pulse of Murasa": CardDef("Pulse of Murasa", CardType.INSTANT, {"generic": 2, "G": 1}, EffectId.PULSE_OF_MURASA),

    # --- G9: elves ---
    "Llanowar Elves": CardDef("Llanowar Elves", CardType.CREATURE, {"G": 1}, EffectId.LLANOWAR_ELVES, power=1, toughness=1, subtypes=("Elf", "Druid")),
    "Fyndhorn Elves": CardDef("Fyndhorn Elves", CardType.CREATURE, {"G": 1}, EffectId.FYNDHORN_ELVES, power=1, toughness=1, subtypes=("Elf", "Druid")),
    "Priest of Titania": CardDef("Priest of Titania", CardType.CREATURE, {"generic": 1, "G": 1}, EffectId.PRIEST_OF_TITANIA, power=1, toughness=1, subtypes=("Elf", "Druid")),
    "Wellwisher": CardDef("Wellwisher", CardType.CREATURE, {"generic": 1, "G": 1}, EffectId.WELLWISHER, power=1, toughness=1, subtypes=("Elf",)),
    "Timberwatch Elf": CardDef("Timberwatch Elf", CardType.CREATURE, {"generic": 2, "G": 1}, EffectId.TIMBERWATCH_ELF, power=1, toughness=2, subtypes=("Elf",)),

    # --- G12: initiative (elves) ---
    "Avenging Hunter": CardDef("Avenging Hunter", CardType.CREATURE, {"generic": 4, "G": 1}, EffectId.AVENGING_HUNTER, power=5, toughness=4, subtypes=("Dragon", "Ranger")),
}


def _is_elf(permanent):
    return "Elf" in card_subtypes(permanent.card_def)


def _count_elves(state):
    """Elves on both players' battlefields (Priest of Titania / Wellwisher /
    Timberwatch Elf's shared count)."""
    return sum(1 for player in state.players for p in player.battlefield if _is_elf(p))


def wellwisher_activate(state, permanent):
    """{T}: You gain 1 life for each Elf on the battlefield (counted at
    resolution)."""
    tap_for_cost(state, permanent)
    push_ability_to_stack(state, permanent.card_def, lambda st: gain_life(st, _count_elves(st)))


def timberwatch_elf_activate(state, permanent):
    """{T}: Target creature gets +X/+X until end of turn, X = # Elves on the
    battlefield (counted at resolution). Target locked at activation."""
    tap_for_cost(state, permanent)
    idx = state.active_idx

    def _on_target(state, descriptor):
        captured = capture_any_target(state, descriptor)

        def _resolve(st, cd):
            if captured is None or not target_still_legal(st, captured):
                _log_target_fizzle(st, cd, None)
                return
            x = _count_elves(st)
            captured[1].temp_power += x
            captured[1].temp_toughness += x
            st.log_event("pump", target=(captured[1].card_def.name, captured[1].slot), amount=x)

        push_to_stack(state, permanent.card_def, _resolve, reserves_hand_card=False, is_spell=False,
                      targets=() if captured is None else (captured,))

    resolution.begin_choose_any_target(
        state, lambda p: p.card_type == CardType.CREATURE and can_be_targeted(state, p, idx),
        _on_target, allow_players=False,
    )

# Sagu Wildling's creature half (the Omen's payoff) -- a distinct CardDef
# from the "Sagu Wildling" sorcery entry above. Not registered in
# GREEN_CARD_CATALOG itself; only reached via ROOST_SEEK's "omen" registry
# spec (see game.effects.tokens for the same pattern).
SAGU_WILDLING_CREATURE_CARD_DEF = CardDef(
    "Sagu Wildling", CardType.CREATURE, {"generic": 4, "G": 1}, EffectId.SAGU_WILDLING, power=3, toughness=3,
)


def _is_defender(permanent):
    return permanent.card_def.extra.get("defender", False)


def forestcycle_generous_ent(state, card_def):
    """{1}, discard from hand: search library for a Forest to hand, shuffle."""
    discard_from_hand_to_graveyard(state, card_def)
    find_to_hand(state, "Forest")


def _wall_of_roots_mana_available(state, permanent):
    """Enforces "Activate only once each turn" (real card has no {T} cost;
    the registry's mana_no_tap flag skips the tap itself)."""
    return not permanent.flags.get("used_this_turn", False)


def _wall_of_roots_on_tap(state, permanent):
    """Adds a -0/-1 counter; death from toughness reaching 0 is handled by
    the ordinary state-based-action check, not bespoke logic here."""
    permanent.flags["used_this_turn"] = True
    permanent.counters["-0/-1"] = permanent.counters.get("-0/-1", 0) + 1


def cast_roost_seek(state, card_def):
    """Sagu Wildling's Omen sorcery half: {G}, search library for any basic
    land (no color restriction) to hand, shuffle.

    Omen doesn't exile itself like an ordinary sorcery -- it shuffles back
    into the library as part of this same search. The creature half becomes
    castable again once this card is redrawn (see EffectId.ROOST_SEEK's
    "omen" registry spec)."""
    if card_def in state.hand:
        state.hand.remove(card_def)  # already removed at cast normally; tolerant of direct calls
    state.library.append(card_def)  # Omen: goes to the library, not the graveyard
    resolution.begin_search_fetch(state, lambda c: c.card_type == CardType.LAND and c.extra.get("basic", False), find_to_hand)


def cast_sagu_wildling_creature(state, card_def):
    """Sagu Wildling's creature half: cast once the card is redrawn (see
    cast_roost_seek), via the "omen" registry spec. {4}{G} 3/3 Flying, ETB
    gain 3 life.

    `card_def` is SAGU_WILDLING_CREATURE_CARD_DEF, a different object from
    the sorcery-side CardDef sitting in state.hand (same display name) --
    so the hand card must be found by name, not identity."""
    hand_card = next((c for c in state.hand if c.name == "Sagu Wildling"), None)
    if hand_card is not None:
        state.hand.remove(hand_card)  # already removed at cast normally; tolerant of a direct call
    enters_battlefield(state, card_def)


def gatecreeper_vine_etb(state):
    """ETB: may search a basic land or Gate card to hand, optional even with
    a legal target. No Gate cards exist in this pool yet; the predicate stays
    correct if one is added."""
    resolution.begin_search_fetch(
        state, lambda c: c.card_type == CardType.LAND and (c.extra.get("basic", False) or "Gate" in card_subtypes(c)),
        find_to_hand, optional=True,
    )


NYXBORN_HYDRA_MAX_X = 10  # ponytail: bounds the action table only; plan_payment masks unaffordable X. Raise if this ever binds in a real game.


def cast_nyxborn_hydra_creature(x):
    """{X}{G} cast: ETB with X +1/+1 counters on the 0/1 base. Returns the
    (state, card_def) resolve callable that build_action_table's x_cast_modes
    loop expects, one closure per X value."""
    def resolve(state, card_def):
        permanent = cast_permanent_from_hand(state, card_def)
        if x:
            permanent.counters["+1/+1"] = x
    return resolve


def cast_nyxborn_hydra_bestow(x):
    """Bestow: {G}{G}{X}, enchant target creature (any creature, either side
    -- no "you control" restriction). Cast as an Aura via cast_aura, with a
    precast target choice.

    on_attached sets type_override=ENCHANTMENT (so combat eligibility,
    state-based death, and "another creature" checks treat it as NOT a
    creature while attached) and X +1/+1 counters, which pt_bonus/
    toughness_bonus read back dynamically to size the bonus. Falling off is
    handled by state_based._destroy_creature's becomes_creature_when_orphaned
    branch, not here.

    no_target_fallback (702.103e): if there's no legal creature to enchant
    at resolution, the spell does NOT fizzle to the graveyard like an
    ordinary Aura -- it enters the battlefield as a creature with its X
    +1/+1 counters instead, reusing cast_nyxborn_hydra_creature directly."""
    def resolve(state, card_def):
        def _on_attached(state, aura):
            aura.type_override = CardType.ENCHANTMENT
            if x:
                aura.counters["+1/+1"] = x
        cast_aura(state, card_def, lambda p: p.card_type == CardType.CREATURE, on_attached=_on_attached,
                  no_target_fallback=cast_nyxborn_hydra_creature(x))
    return resolve


def cast_land_grant(state, card_def):
    """Search library for a Forest, put to hand, shuffle. The pure effect,
    shared by the normal {1}{G} cast (already deferred by drl_env.
    _cast_execute's mana push) and the free alt-cost cast below (which
    pushes itself) -- they differ only in cost/stack handling, not the
    effect."""
    discard_from_hand_to_graveyard(state, card_def)
    find_to_hand(state, "Forest")


def cast_land_grant_alt(state, card_def):
    """Free alt-cost path: nothing to pay, so pushes itself to the stack
    immediately (drl_env._alt_cast_execute calls this synchronously) --
    same treatment as Faithless Looting's free Flashback."""
    push_to_stack(state, card_def, cast_land_grant)


def land_grant_alt_cost_legal(state):
    """Free alt-cost ("reveal your hand" instead of paying) is legal only
    with no land cards in hand. Revealing has no simulator-visible effect
    (no opponent to show it to)."""
    return not any(c.card_type == CardType.LAND for c in state.hand)


def quirion_ranger_untap_legal(state, permanent):
    """Legal once per turn (used_this_turn, reset every untap_step) with a
    Forest controlled. No {T} in the real cost, so this permanent's own
    tapped state doesn't matter."""
    if permanent.flags.get("used_this_turn", False):
        return False
    return any(p.card_def.name == "Forest" for p in state.battlefield)


def quirion_ranger_untap_resolve(state, permanent):
    """"Return a Forest you control to hand: untap target creature." Which
    Forest pays the cost is the player's own choice among every eligible one
    (602.5g) -- Forests stop being fungible once one carries something like
    Utopia Sprawl. Once the chosen Forest is bounced (the cost, paid via
    resolution.begin_choose_permanent), the untap is an activated ability
    on the stack: its target (any creature, either battlefield,
    hexproof-aware) is locked at activation and fizzles if that creature
    has left the battlefield by then (608.2b) or is illegal. If the target
    was tapped, also discounts whatever mana it could have produced
    (mana.discount_departing_source)."""
    permanent.flags["used_this_turn"] = True

    def _on_forest_chosen(state, choice):
        name, slot = choice
        forest = next(p for p in state.battlefield if p.card_def.name == name and p.slot == slot)
        state.battlefield.remove(forest)
        state.hand.append(forest.card_def)
        state.log_event(
            "zone_move", permanent=(forest.card_def.name, forest.slot), from_zone="battlefield", to_zone="hand",
            reason="quirion_ranger_bounce",
        )

        def _on_target(state, target_descriptor):
            captured = capture_any_target(state, target_descriptor)

            def _resolve(state, card_def):
                if captured is None or not target_still_legal(state, captured):
                    where = (captured[1].card_def.name, captured[1].slot) if captured is not None else None
                    _log_target_fizzle(state, card_def, where)
                    return
                target = captured[1]
                was_tapped = target.tapped
                if was_tapped:
                    # Free untap -- excuse any floating mana this source could have produced.
                    discount_departing_source(state, target, controller_idx(state, target))
                set_tapped(state, target, False, reason="quirion_ranger")

            push_to_stack(state, permanent.card_def, _resolve, reserves_hand_card=False, is_spell=False,  # activated ability -- not a spell
                          targets=() if captured is None else (captured,))

        resolution.begin_choose_any_target(
            state,
            lambda p: p.card_type == CardType.CREATURE and can_be_targeted(state, p, state.active_idx),
            _on_target,
            allow_players=False,
        )

    resolution.begin_choose_permanent(state, lambda p: p.card_def.name == "Forest", _on_forest_chosen)


def _cast_winding_way(state, card_def, chosen_type):
    """Choose creature or land at cast (two action-table entries). Reveal
    top 4; matches go to hand, rest to graveyard."""
    discard_from_hand_to_graveyard(state, card_def)
    revealed = state.library[:4]
    del state.library[:4]
    for card in revealed:
        if card.card_type == chosen_type:
            state.hand.append(card)
        else:
            state.move_card(card, state.graveyard)


def cast_winding_way_creature(state, card_def):
    _cast_winding_way(state, card_def, CardType.CREATURE)


def cast_winding_way_land(state, card_def):
    _cast_winding_way(state, card_def, CardType.LAND)


def begin_select_to_hand(state, n, eligible_predicate, on_complete):
    """Lead the Stampede: reveal top n; model keeps (if eligible) or bottoms
    each card in turn, then orders any 2+ bottomed cards. Mirrors
    resolution.begin_scry_surveil's shape, except kept cards go to hand and
    only eligible cards may be kept."""
    revealed = state.library[:n]
    del state.library[:n]
    resolution.begin_resolution(
        state, "select_to_hand", on_complete,
        remaining=revealed, eligible=eligible_predicate, kept=[], disposed=[], ordered=None,
    )
    if not revealed:
        resolution.complete_resolution(state)  # empty library -- nothing to decide, complete immediately


def select_to_hand_options(state):
    """Deciding: keep (if the front card is eligible) or bottom. Ordering
    (remaining empty, 2+ disposed): one option per distinct name still
    waiting to be bottomed."""
    pending = state.pending_resolution
    if pending["remaining"]:
        front = pending["remaining"][0]
        return ["keep", "bottom"] if pending["eligible"](front) else ["bottom"]
    if pending["ordered"] is not None:
        return sorted({c.name for c in pending["disposed"]})
    return []


def _finish_select_to_hand(state):
    pending = state.pending_resolution
    state.hand.extend(pending["kept"])
    disposed_final = pending["ordered"] if pending["ordered"] is not None else pending["disposed"]
    state.library.extend(disposed_final)
    resolution.complete_resolution(state)


def execute_select_to_hand_option(state, option):
    pending = state.pending_resolution
    if pending["remaining"]:
        card = pending["remaining"].pop(0)
        (pending["kept"] if option == "keep" else pending["disposed"]).append(card)
        if pending["remaining"]:
            return  # more cards still to decide
        if len(pending["disposed"]) <= 1:
            _finish_select_to_hand(state)  # 0 or 1 bottomed -- no ordering choice to make
        else:
            pending["ordered"] = []  # 2+ bottomed -- enter the ordering phase
        return

    # Ordering phase: option is the name of the next card to bottom.
    idx = next(i for i, c in enumerate(pending["disposed"]) if c.name == option)
    pending["ordered"].append(pending["disposed"].pop(idx))
    if not pending["disposed"]:
        _finish_select_to_hand(state)


def cast_lead_the_stampede(state, card_def):
    """{2}{G}: look at top 5, may reveal any number of creatures to hand,
    rest to the bottom in any order."""
    discard_from_hand_to_graveyard(state, card_def)
    begin_select_to_hand(state, 5, lambda c: c.card_type == CardType.CREATURE, on_complete=lambda s: None)


def is_noncreature_colorless(card_def):
    if card_def.card_type in (CardType.CREATURE, CardType.FILLER):
        return False
    if card_def.cast_cost is None:
        return True  # a land -- no mana cost, therefore colorless
    return not any(k in COLORS for k in card_def.cast_cost)


def cast_crop_rotation(state, card_def):
    """{G}, sacrifice a land: search library for a land, put it onto the
    battlefield (its own tapped/ETB rules apply), shuffle.

    The sacrifice is an additional cost, paid before the spell is fully
    cast -- this function runs directly as pay_cost's on_complete, not
    deferred onto the stack. Only the search (the spell's actual effect)
    is pushed to the stack."""
    discard_from_hand_to_graveyard(state, card_def)

    def _on_sacrificed(state, _ok):
        def _resolve(state, card_def):
            def _on_fetch_chosen(state, land_name):
                found = find_and_remove_by_name(state, land_name)
                shuffle_library(state)
                if found:
                    enters_battlefield(state, found)

            resolution.begin_search_fetch(state, lambda c: c.card_type == CardType.LAND, _on_fetch_chosen)

        push_to_stack(state, card_def, _resolve, reserves_hand_card=False)

    resolution.begin_sacrifice(
        state,
        lambda p: p.card_def.card_type == CardType.LAND and p.card_def.effect_id != EffectId.TRON_LAND,
        1,
        _on_sacrificed,
    )


def begin_ancient_stirrings(state, revealed, on_complete):
    """Model picks at most one noncreature-colorless card from `revealed`,
    or declines -- a single decision, not a sequential walk like scry/
    surveil. on_complete(state, chosen_card_or_None) runs once decided."""
    resolution.begin_resolution(state, "ancient_stirrings", on_complete, revealed=revealed)


def ancient_stirrings_options(state):
    revealed = state.pending_resolution["revealed"]
    eligible_names = sorted({c.name for c in revealed if is_noncreature_colorless(c)})
    return eligible_names + ["decline"]


def execute_ancient_stirrings_option(state, option):
    revealed = state.pending_resolution["revealed"]
    if option == "decline":
        chosen = None
    else:
        idx = next(i for i, c in enumerate(revealed) if c.name == option)
        chosen = revealed.pop(idx)
    state.rng.shuffle(revealed)  # the rest (all of it, if declined) goes to the bottom
    state.library.extend(revealed)
    resolution.complete_resolution(state, chosen)


def cast_ancient_stirrings(state, card_def):
    """{G}: look at top 5, may take one noncreature colorless card to hand,
    rest to bottom in random order."""
    discard_from_hand_to_graveyard(state, card_def)
    top = state.library[:5]
    del state.library[:5]

    def _on_chosen(state, chosen):
        if chosen is not None:
            state.hand.append(chosen)

    begin_ancient_stirrings(state, top, _on_chosen)


def cast_rancor(state, card_def):
    cast_aura(state, card_def, lambda p: p.card_type == CardType.CREATURE)


def cast_ancestral_mask(state, card_def):
    cast_aura(state, card_def, lambda p: p.card_type == CardType.CREATURE)


def _utopia_sprawl_attach(color):
    def on_attached(state, aura):
        aura.flags["bonus_mana_color"] = color
    return on_attached


def cast_utopia_sprawl(state, card_def, color):
    """Enchants a Forest specifically, not any land. The chosen color is
    recorded on the Aura's flags; mana.py's _bonus_mana_symbols reads it to
    add that color alongside the Forest's own G whenever it's tapped."""
    cast_aura(state, card_def, lambda p: p.card_def.name == "Forest", on_attached=_utopia_sprawl_attach(color))


def abundant_growth_attach(state, aura):
    state.draw(1)
    aura.flags["bonus_mana_colors"] = set(COLORS)  # enchanted land taps for any of the 5 colors


def cast_abundant_growth(state, card_def):
    cast_aura(state, card_def, lambda p: p.card_def.card_type == CardType.LAND, on_attached=abundant_growth_attach)


def begin_malevolent_rumble(state, revealed, on_complete):
    """Reveal top 4; model may take one permanent card to hand, rest to
    graveyard. Its own resolution kind, distinct from ancient_stirrings'
    similar-looking shape (the disposal zone differs: graveyard here,
    library bottom there)."""
    resolution.begin_resolution(state, "malevolent_rumble", on_complete, revealed=revealed)


_PERMANENT_CARD_TYPES = (CardType.LAND, CardType.ARTIFACT, CardType.CREATURE, CardType.ENCHANTMENT)


def malevolent_rumble_options(state):
    revealed = state.pending_resolution["revealed"]
    eligible_names = sorted({c.name for c in revealed if c.card_type in _PERMANENT_CARD_TYPES})
    return eligible_names + ["decline"]


def execute_malevolent_rumble_option(state, option):
    revealed = state.pending_resolution["revealed"]
    if option == "decline":
        chosen = None
    else:
        idx = next(i for i, c in enumerate(revealed) if c.name == option)
        chosen = revealed.pop(idx)
    for c in revealed:  # order is never read again
        state.move_card(c, state.graveyard)
    resolution.complete_resolution(state, chosen)


def cast_malevolent_rumble(state, card_def):
    """{1}{G}: reveal top 4, may take one permanent card to hand, rest to
    graveyard, create a 0/1 Eldrazi Spawn token. No Madness (real card has
    none)."""
    discard_from_hand_to_graveyard(state, card_def)
    create_token(state, ELDRAZI_SPAWN_TOKEN_CARD_DEF)
    top = state.library[:4]
    del state.library[:4]

    def _on_chosen(state, chosen):
        if chosen is not None:
            state.hand.append(chosen)

    begin_malevolent_rumble(state, top, _on_chosen)


def activate_bramble_wurm_gy(state, inst):
    """{2}{G}, exile this card from your graveyard: gain 5 life. Exiling
    from the graveyard is a cost, paid now on activation; the life gain is
    the effect and goes on the stack.

    inst: the exact graveyard CardInstance this ability belongs to -- see
    black_cards.flashback_dread_return."""
    state.graveyard.remove(inst)
    push_ability_to_stack(state, inst, lambda st: gain_life(st, 5))


def _ram_through_extra_legal(state):
    """Legal only with both a creature you control and one you don't to
    target. No-op in a 1-player config (no opponent creature)."""
    if len(state.players) < 2:
        return False
    idx = state.active_idx
    mine = any(p.card_type == CardType.CREATURE and can_be_targeted(state, p, idx) for p in state.battlefield)
    theirs = any(p.card_type == CardType.CREATURE and can_be_targeted(state, p, idx) for p in state.opponent.battlefield)
    return mine and theirs


def cast_ram_through(state, card_def):
    """{1}{G} instant: target creature you control deals damage equal to its
    power to target creature you don't control; trample overflow goes to
    that creature's controller instead.

    Both targets locked at cast, rechecked at resolution -- the spell
    fizzles if either has left the battlefield or become illegal by then
    (608.2c). Damage is marked like combat damage and lethality resolved
    via the shared state-based-action check; trample overflow uses the same
    deal_damage_to_opponent path as combat."""
    idx = state.active_idx

    def _on_source_chosen(state, source_descriptor):
        cap_source = capture_any_target(state, source_descriptor)

        def _on_target_chosen(state, target_descriptor):
            cap_target = capture_any_target(state, target_descriptor)

            def _resolve(state, card_def):
                discard_from_hand_to_graveyard(state, card_def)  # Ram Through itself -> graveyard
                if (cap_source is None or cap_target is None
                        or not target_still_legal(state, cap_source) or not target_still_legal(state, cap_target)):
                    where = (cap_target[1].card_def.name, cap_target[1].slot) if cap_target is not None else None
                    _log_target_fizzle(state, card_def, where)
                    return
                source, target = cap_source[1], cap_target[1]
                power = permanent_power(state, source)
                if power <= 0:
                    return  # deals 0 damage -- nothing happens
                if has_keyword(state, source, "trample"):
                    lethal = max(permanent_toughness(state, target) - target.damage_marked, 0)
                    to_creature = min(power, lethal)
                    to_controller = power - to_creature
                else:
                    to_creature, to_controller = power, 0
                target.damage_marked += to_creature
                state.log_event(
                    "fight_damage", source=(source.card_def.name, source.slot),
                    target=(target.card_def.name, target.slot), amount=to_creature,
                    trample_excess_to_controller=to_controller,
                )
                if to_controller > 0:
                    deal_damage_to_opponent(state, to_controller)  # "that creature's controller" == the opponent
                check_state_based_actions(state)  # kills the target if the damage was lethal

            push_to_stack(state, card_def, _resolve, targets=tuple(t for t in (cap_source, cap_target) if t is not None))

        # Second target: a creature you don't control, hexproof/shroud-aware.
        resolution.begin_choose_any_target(
            state,
            lambda p: p not in state.battlefield and p.card_type == CardType.CREATURE and can_be_targeted(state, p, idx),
            _on_target_chosen,
            allow_players=False,
        )

    # First target: a creature you control. Shroud blocks even your own
    # creature; hexproof only stops opponents.
    resolution.begin_choose_any_target(
        state,
        lambda p: p in state.battlefield and p.card_type == CardType.CREATURE and can_be_targeted(state, p, idx),
        _on_source_chosen,
        allow_players=False,
    )


def masked_vandal_etb(state, permanent):
    """When Masked Vandal enters, you may exile a creature card from your
    graveyard. If you do, exile target artifact or enchantment an opponent
    controls. (Changeling is a no-op -- no tribal-synergy card exists in
    this pool.)

    The target (opponent artifact/enchantment) is chosen and captured at
    trigger time; with no legal target the ability isn't put on the stack
    (603.3c). At resolution: the "may exile" choice, and if taken, exile
    the captured target if still legal (else fizzle, 608.2b). Declining
    the graveyard exile is not itself a fizzle."""
    if len(state.players) < 2:
        return

    def _targetable(p):
        return p.card_type in (CardType.ARTIFACT, CardType.ENCHANTMENT) and can_be_targeted(state, p, state.active_idx)

    if not any(_targetable(p) for p in state.opponent.battlefield):
        return  # no legal target -> not put on the stack (603.3c)

    def _on_target(state, choice):
        if choice is None:
            return  # unreachable: legality confirmed above
        tname, tslot = choice
        target = next(p for p in state.opponent.battlefield if p.card_def.name == tname and p.slot == tslot)  # captured at promotion

        def _resolve(state, card_def):
            def _after_gy(state, chosen):
                if chosen is None:
                    return  # declined -- "if you do" doesn't happen
                state.graveyard.remove(chosen)  # exile the chosen creature (untracked)
                state.log_event("zone_move", card=chosen.name, from_zone="graveyard", to_zone="exile_untracked", reason="masked_vandal")
                # exile the captured target if still legal (608.2b)
                owner = next((pl for pl in state.players if target in pl.battlefield), None)
                if owner is None:
                    _log_target_fizzle(state, card_def, (target.card_def.name, target.slot))
                    return
                owner.battlefield.remove(target)  # exiled, untracked
                state.log_event(
                    "zone_move", permanent=(target.card_def.name, target.slot), from_zone="battlefield",
                    to_zone="exile_untracked", reason="masked_vandal",
                )

            resolution.begin_choose_graveyard_card(state, lambda c: c.card_type == CardType.CREATURE, _after_gy, optional=True)

        push_to_stack(state, permanent.card_def, _resolve, reserves_hand_card=False, is_spell=False)  # ETB effect -- not a spell

    resolution.begin_choose_opponent_permanent(state, _targetable, _on_target)


def _gingerbread_cabin_enters_tapped(state):
    """Enters tapped unless controlling 3+ other Forests. Counts
    Forest-subtype permanents already on the battlefield (this Cabin isn't
    added yet, so all counted are genuinely "other")."""
    forests = sum(1 for p in state.battlefield if "Forest" in card_subtypes(p.card_def))
    return forests < 3


def gingerbread_cabin_etb(state, permanent):
    """Untapped entry creates a Food token. Reads flags["entered_tapped"]
    (set by enters_battlefield) rather than current tapped state, which a
    later mana tap could flip before this ETB resolves."""
    if not permanent.flags.get("entered_tapped", False):
        create_token(state, FOOD_TOKEN_CARD_DEF)


def _pulse_of_murasa_eligible(card_def):
    return card_def.card_type in (CardType.CREATURE, CardType.LAND)


def cast_pulse_of_murasa(state, card_def):
    """{2}{G}: return target creature or land card from a graveyard to its
    owner's hand; gain 6 life. Target chosen at cast from either player's
    graveyard, rechecked at resolution -- fizzles (no lifegain) if it's
    since left (608.2c). The card returns to its owner's hand.

    Same-named cards present in both graveyards resolve caster's-graveyard-
    first (this engine's fungible-by-name convention)."""
    caster_idx = state.active_idx

    def _on_chosen(state, chosen):
        def _resolve(state, card_def):
            discard_from_hand_to_graveyard(state, card_def)  # Pulse itself -> caster's graveyard
            # fizzle (608.2b) if the locked instance has since left the graveyard
            owner_idx = (next((i for i, pl in enumerate(state.players) if chosen in pl.graveyard), None)
                         if chosen is not None else None)
            if owner_idx is None:
                _log_target_fizzle(state, card_def, None)  # target left the graveyard -> fizzle, no lifegain
                return
            state.players[owner_idx].graveyard.remove(chosen)
            state.players[owner_idx].hand.append(chosen.card_def)  # to its OWNER's hand (hand DEFERRED -- CardDef)
            state.log_event("zone_move", card=chosen.name, from_zone="graveyard", to_zone="hand", reason="pulse_of_murasa")
            gain_life(state, 6)  # caster (active_idx at resolution) gains

        push_to_stack(state, card_def, _resolve)

    combined = [c for pl in state.players for c in pl.graveyard]  # A graveyard = either player's
    resolution.begin_choose_graveyard_card(state, _pulse_of_murasa_eligible, _on_chosen, graveyard=combined)


GREEN_EFFECT_REGISTRY = {
    EffectId.FOREST: {
        "mana": ("fixed", "G"),
    },
    EffectId.PULSE_OF_MURASA: {
        "cast": {
            "resolve": lambda state, card_def: cast_pulse_of_murasa(state, card_def),
            "extra_legal": lambda state: any(
                _pulse_of_murasa_eligible(c) for pl in state.players for c in pl.graveyard),  # A graveyard, either player
            "precast_choice": True,  # target graveyard card locked at cast
        },
        "pending_kinds": {"choose_graveyard_card"},
    },
    # --- G9: elves ---
    EffectId.LLANOWAR_ELVES: {
        "cast": {"resolve": lambda state, card_def: cast_permanent_from_hand(state, card_def)},
        "mana": ("fixed", "G"),  # {T}: Add {G}
    },
    EffectId.FYNDHORN_ELVES: {
        "cast": {"resolve": lambda state, card_def: cast_permanent_from_hand(state, card_def)},
        "mana": ("fixed", "G"),  # {T}: Add {G}
    },
    EffectId.PRIEST_OF_TITANIA: {
        "cast": {"resolve": lambda state, card_def: cast_permanent_from_hand(state, card_def)},
        "mana": ("count_all", "G", _is_elf),  # {T}: Add {G} for each Elf on the battlefield (both sides)
    },
    EffectId.WELLWISHER: {
        "cast": {"resolve": lambda state, card_def: cast_permanent_from_hand(state, card_def)},
        "activated_abilities": {
            "lifegain": {  # {T}: gain 1 life per Elf
                "legal": lambda state, permanent: not permanent.tapped and not tap_summoning_locked(state, permanent),  # summoning-sickness gated (302.6)
                "resolve": lambda state, permanent: wellwisher_activate(state, permanent),
            },
        },
    },
    EffectId.TIMBERWATCH_ELF: {
        "cast": {"resolve": lambda state, card_def: cast_permanent_from_hand(state, card_def)},
        "activated_abilities": {
            "pump": {  # {T}: target creature +X/+X (X = # Elves) until EOT
                "legal": lambda state, permanent: not permanent.tapped and not tap_summoning_locked(state, permanent),  # summoning-sickness gated (302.6)
                "resolve": lambda state, permanent: timberwatch_elf_activate(state, permanent),
            },
        },
        "pending_kinds": {"choose_any_target"},
    },
    EffectId.GINGERBREAD_CABIN: {
        "mana": ("fixed", "G"),
        "enters_tapped": _gingerbread_cabin_enters_tapped,  # callable(state) -- see enters_battlefield
        "etb_trigger": lambda state, permanent: gingerbread_cabin_etb(state, permanent),
    },
    EffectId.GENEROUS_ENT: {
        # Reach + ETB creates a Food token, alongside its Forestcycling {1}.
        # Power/toughness is this catalog's own design-choice value.
        "cast": {"resolve": lambda state, card_def: cast_permanent_from_hand(state, card_def)},
        "keywords": {"reach"},
        "etb_trigger": lambda state, permanent: create_token(state, FOOD_TOKEN_CARD_DEF),
        "forestcycle": {
            "cost_key": "forestcycling_cost",
            "resolve": lambda state, card_def: forestcycle_generous_ent(state, card_def),
        },
    },
    EffectId.FOOD_TOKEN: {
        # "{2}, {T}, Sacrifice: gain 3 life." Only ever created by Generous
        # Ent's ETB.
        "activated_abilities": {
            "sac": {
                "cost_key": "sac_ability_cost",
                "resolve": lambda state, permanent: activate_food_sac(state, permanent),
            },
        },
    },
    EffectId.MASKED_VANDAL: {
        "cast": {"resolve": lambda state, card_def: cast_permanent_from_hand(state, card_def)},
        # "you may exile a creature card from your graveyard. If you do,
        # exile target artifact or enchantment an opponent controls."
        "etb_trigger": lambda state, permanent: masked_vandal_etb(state, permanent),
        "etb_targets": True,  # target chosen at promotion; the "you may exile a creature" + fizzle at resolution
        "pending_kinds": {"choose_graveyard_card", "choose_opponent_permanent"},
    },
    EffectId.SARULI_CARETAKER: {
        "cast": {"resolve": lambda state, card_def: cast_permanent_from_hand(state, card_def)},
        "mana": ("flexible", set(COLORS)),
        # CONFIRMED WITH REPO OWNER (2026-08-23, after repeat confusion on this
        # exact point across multiple sessions): the full, correct activation
        # cost is "{T}, Tap an untapped creature you control: Add one mana of
        # any color" -- TWO tap costs. Saruli DOES tap itself; no
        # "mana_no_tap" entry here is correct and must stay that way. Do not
        # "fix" this again.
        #
        # The untapped-creature half is an extra cost choice (602.5g), so the
        # agent picks which. Two sequential steps (tap choice, then color
        # choice) via a mana_subdecision (drl_env._mana_extra_choose_legal/
        # _execute) -- separate from pending_resolution so it stays legal in
        # any priority window, even mid-resolution of something else
        # (605.1a/605.3b). See state.mana_subdecision's own docstring.
        "mana_extra_choose": lambda p: p.card_type == CardType.CREATURE,
    },
    EffectId.OVERGROWN_BATTLEMENT: {
        "cast": {"resolve": lambda state, card_def: cast_permanent_from_hand(state, card_def)},
        "mana": ("count", "G", _is_defender),
    },
    EffectId.WALL_OF_ROOTS: {
        "cast": {"resolve": lambda state, card_def: cast_permanent_from_hand(state, card_def)},
        "mana": ("fixed", "G"),
        "mana_no_tap": True,
        "mana_extra_available": lambda state, permanent: _wall_of_roots_mana_available(state, permanent),
        "on_tap": lambda state, permanent: _wall_of_roots_on_tap(state, permanent),
    },
    EffectId.ROOST_SEEK: {
        "cast": {"resolve": lambda state, card_def: cast_roost_seek(state, card_def)},
        "omen": {
            "card_def": SAGU_WILDLING_CREATURE_CARD_DEF,
            "cost": {"generic": 4, "G": 1},
            "resolve": lambda state, card_def: cast_sagu_wildling_creature(state, card_def),
        },
        "pending_kinds": {"search_fetch"},
    },
    EffectId.SAGU_WILDLING: {
        "etb_trigger": lambda state, permanent: gain_life(state, 3),
        "keywords": {"flying"},
    },
    EffectId.GATECREEPER_VINE: {
        "cast": {"resolve": lambda state, card_def: cast_permanent_from_hand(state, card_def)},
        "etb_trigger": lambda state, permanent: gatecreeper_vine_etb(state),
        "pending_kinds": {"search_fetch"},
    },
    EffectId.NYXBORN_HYDRA: {
        # Two cast modes, one action per X (0..NYXBORN_HYDRA_MAX_X) via
        # build_action_table's x_cast_modes loop.
        "x_cast_modes": {
            "creature": {"cost": {"G": 1}, "max_x": NYXBORN_HYDRA_MAX_X, "resolve": cast_nyxborn_hydra_creature},
            "bestow": {
                "cost": {"G": 2}, "max_x": NYXBORN_HYDRA_MAX_X, "precast_choice": True,
                "extra_legal": lambda state: any_creature_on_either_battlefield(state),
                "resolve": cast_nyxborn_hydra_bestow,
            },
        },
        # Read back only while Bestowed -- the bonus is this permanent's own
        # +1/+1 counters, not a fixed constant like Rancor's +2.
        "pt_bonus": lambda state, aura: aura.counters.get("+1/+1", 0),
        "toughness_bonus": lambda state, aura: aura.counters.get("+1/+1", 0),
        # Bestow fall-off: stays on the battlefield and becomes a creature
        # again, per state_based._destroy_creature's third branch.
        "becomes_creature_when_orphaned": True,
    },
    EffectId.QUIRION_RANGER: {
        "cast": {"resolve": lambda state, card_def: cast_permanent_from_hand(state, card_def)},
        "activated_abilities": {
            "untap": {
                "legal": lambda state, permanent: quirion_ranger_untap_legal(state, permanent),
                "resolve": lambda state, permanent: quirion_ranger_untap_resolve(state, permanent),
            },
        },
        # Forest choice is choose_permanent; the untap target (any creature,
        # either side) is choose_any_target.
        "pending_kinds": {"choose_permanent", "choose_any_target"},
    },
    EffectId.RAM_THROUGH: {
        "cast": {
            "resolve": lambda state, card_def: cast_ram_through(state, card_def),
            "extra_legal": lambda state: _ram_through_extra_legal(state),
            "precast_choice": True,  # both targets locked at cast; the one-sided fight waits on the stack
        },
        "pending_kinds": {"choose_any_target"},
    },
    EffectId.WINDING_WAY: {
        "cast_modes": {
            "creature": {"resolve": lambda state, card_def: cast_winding_way_creature(state, card_def)},
            "land": {"resolve": lambda state, card_def: cast_winding_way_land(state, card_def)},
        },
    },
    EffectId.LEAD_THE_STAMPEDE: {
        "cast": {"resolve": lambda state, card_def: cast_lead_the_stampede(state, card_def)},
        "pending_kinds": {"select_to_hand"},
    },
    EffectId.LAND_GRANT: {
        "cast": {"resolve": lambda state, card_def: cast_land_grant(state, card_def)},
        "alt_cast": {
            "extra_legal": lambda state: land_grant_alt_cost_legal(state),
            "resolve": lambda state, card_def: cast_land_grant_alt(state, card_def),
        },
    },
    EffectId.CROP_ROTATION: {
        "cast": {
            "resolve": lambda state, card_def: cast_crop_rotation(state, card_def),
            "extra_legal": lambda state: any(
                p.card_def.card_type == CardType.LAND and p.card_def.effect_id != EffectId.TRON_LAND
                for p in state.battlefield
            ),
            # NOT targeted -- "sacrifice a land" is an additional cost, paid
            # alongside the {G} mana cost before the spell is fully cast.
            # Reuses precast_choice routing because it needs "resolve
            # immediately, manage its own push_to_stack," not because this
            # is a target.
            "precast_choice": True,
        },
        "pending_kinds": {"choose_permanent", "search_fetch"},
    },
    EffectId.ANCIENT_STIRRINGS: {
        "cast": {"resolve": lambda state, card_def: cast_ancient_stirrings(state, card_def)},
        "pending_kinds": {"ancient_stirrings"},
    },
    EffectId.BRAMBLE_WURM: {
        "cast": {"resolve": lambda state, card_def: cast_permanent_from_hand(state, card_def)},
        "etb_trigger": lambda state, permanent: gain_life(state, 5),
        # Trample + Reach; Reach lets it block flyers (combat.can_block).
        "keywords": {"trample", "reach"},
        "graveyard_ability": {
            "cost_key": "gy_ability_cost",
            "resolve": lambda state, card_def: activate_bramble_wurm_gy(state, card_def),
        },
    },

    # --- boggles deck ---
    EffectId.GLADECOVER_SCOUT: {
        # Hexproof is a real targeting restriction (stats.can_be_targeted),
        # same as Slippery Bogle.
        "cast": {"resolve": lambda state, card_def: cast_permanent_from_hand(state, card_def)},
        "keywords": {"hexproof"},
    },
    EffectId.SILHANA_LEDGEWALKER: {
        # Hexproof + "can't be blocked except by flying" (an attacker-side
        # restriction, combat.can_block) -- distinct from real flying, so
        # Silhana can't block flyers itself.
        "cast": {"resolve": lambda state, card_def: cast_permanent_from_hand(state, card_def)},
        "keywords": {"cant_be_blocked_except_by_flying", "hexproof"},
    },
    EffectId.RANCOR: {
        # Also grants trample and returns Rancor to hand instead of the
        # graveyard when it leaves the battlefield -- returns_to_hand_when_
        # orphaned, see effects.state_based._destroy_creature.
        # +2/+0, power only -- no toughness_bonus (unlike the symmetric
        # +X/+X Auras: Ancestral Mask/Ethereal Armor/Cartouche of
        # Solidarity/Armadillo Cloak).
        "cast": {
            "resolve": lambda state, card_def: cast_rancor(state, card_def),
            "extra_legal": lambda state: any_creature_on_either_battlefield(state),
            "precast_choice": True,  # enchant target creature -- chosen before the stack
        },
        "pending_kinds": {"choose_any_target"},  # Aura: any creature, either side, hexproof-aware
        "pt_bonus": lambda state, aura: 2,
        "returns_to_hand_when_orphaned": True,
        "keywords": {"trample"},
    },
    EffectId.ANCESTRAL_MASK: {
        # +2/+2 per OTHER enchantment you control (excludes itself, unlike
        # Ethereal Armor).
        "cast": {
            "resolve": lambda state, card_def: cast_ancestral_mask(state, card_def),
            "extra_legal": lambda state: any_creature_on_either_battlefield(state),
            "precast_choice": True,  # enchant target creature -- chosen before the stack
        },
        "pending_kinds": {"choose_any_target"},  # Aura: any creature, either side, hexproof-aware
        "pt_bonus": lambda state, aura: 2 * (enchantment_count(state, aura) - 1),
        "toughness_bonus": lambda state, aura: 2 * (enchantment_count(state, aura) - 1),
    },
    EffectId.UTOPIA_SPRAWL: {
        # One cast mode per color (WUBRG); the chosen color becomes the
        # bonus mana the enchanted Forest adds. Forest target chosen at cast.
        "cast_modes": {
            "green": {
                "resolve": lambda state, card_def: cast_utopia_sprawl(state, card_def, "G"),
                "extra_legal": lambda state: any(p.card_def.name == "Forest" for p in state.battlefield),
                "precast_choice": True,  # enchant target Forest -- chosen before the stack
            },
            "white": {
                "resolve": lambda state, card_def: cast_utopia_sprawl(state, card_def, "W"),
                "extra_legal": lambda state: any(p.card_def.name == "Forest" for p in state.battlefield),
                "precast_choice": True,
            },
            "blue": {
                "resolve": lambda state, card_def: cast_utopia_sprawl(state, card_def, "U"),
                "extra_legal": lambda state: any(p.card_def.name == "Forest" for p in state.battlefield),
                "precast_choice": True,
            },
            "black": {
                "resolve": lambda state, card_def: cast_utopia_sprawl(state, card_def, "B"),
                "extra_legal": lambda state: any(p.card_def.name == "Forest" for p in state.battlefield),
                "precast_choice": True,
            },
            "red": {
                "resolve": lambda state, card_def: cast_utopia_sprawl(state, card_def, "R"),
                "extra_legal": lambda state: any(p.card_def.name == "Forest" for p in state.battlefield),
                "precast_choice": True,
            },
        },
        "pending_kinds": {"choose_any_target"},  # Aura: any Forest, either side, hexproof/shroud aware
    },
    EffectId.ABUNDANT_GROWTH: {
        "cast": {
            "resolve": lambda state, card_def: cast_abundant_growth(state, card_def),
            # any_land_on_either_battlefield, not just the caster's own: the
            # target predicate is "any land, either side" (real "Enchant
            # land" has no "you control" restriction) -- a land-screwed
            # caster with only an opponent's land still has a legal target
            # (601.2c).
            "extra_legal": lambda state: any_land_on_either_battlefield(state),
            "precast_choice": True,  # enchant target land -- chosen before the stack
        },
        "pending_kinds": {"choose_any_target"},  # Aura: any land, either side, hexproof/shroud aware
        # Static fact for build_action_table's pre-registration (which
        # "Tap <land> for <color>" rows exist, before any game state does)
        # -- kept in sync by hand with abundant_growth_attach's runtime
        # value (all five colors).
        "grants_mana_colors": set(COLORS),
    },
    EffectId.MALEVOLENT_RUMBLE: {
        "cast": {"resolve": lambda state, card_def: cast_malevolent_rumble(state, card_def)},
        "pending_kinds": {"malevolent_rumble"},
    },
    EffectId.ELDRAZI_SPAWN_TOKEN: {
        "activated_abilities": {
            "sac": {
                "legal": lambda state, permanent: True,
                "resolve": lambda state, permanent: activate_eldrazi_spawn_sac(state, permanent),
            },
        },
    },
    # --- G12: initiative / Undercity ---
    EffectId.AVENGING_HUNTER: {
        "cast": {"resolve": lambda state, card_def: cast_permanent_from_hand(state, card_def)},
        "keywords": {"trample"},
        # Controller takes the initiative on ETB; take_initiative also queues the venture.
        "etb_trigger": lambda state, permanent: undercity.take_initiative(state, state.active_idx),
        # Every resolution kind an Undercity venture can open, so
        # build_action_table pre-registers matching actions.
        "pending_kinds": {"choose_room", "throne_reveal", "search_fetch", "scry", "choose_any_target", "choose_target_player"},
    },
    EffectId.SKELETON_TOKEN: {"keywords": {"menace"}},  # 4/1 Undercity Catacombs Skeleton
}

