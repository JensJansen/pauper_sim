"""Green-identity card catalog: every card whose real mana cost is mono-green
(or, for a cost-less land, whose only mana output is green). Costs/types/oracle
text are direct Scryfall pulls; creature power/toughness is a design choice.
Bramble Wurm (real {6}{G}) files here rather than colorless_cards despite being
Tron filler. Same GREEN_CARD_CATALOG / GREEN_EFFECT_REGISTRY shape as every
color file, unioned by game/registry.py (per-card mechanics -- Bramble Wurm's
graveyard ability, Sagu Wildling's Omen, the defender flag -- are documented at
their own registry entries below)."""

from .. import resolution
from ..cards import CardDef, CardType, EffectId
from ..effects.casting import (
    _log_target_fizzle, capture_any_target, cast_aura, cast_permanent_from_hand, enters_battlefield,
    target_still_legal,
)
from ..effects.shared import (
    any_creature_on_battlefield, card_subtypes, discard_from_hand_to_graveyard, find_and_remove_by_name, find_to_hand,
)
from ..effects.stack import push_ability_to_stack, push_to_stack
from ..effects.state_based import check_state_based_actions
from ..effects.stats import can_be_targeted, enchantment_count, has_keyword, permanent_power, permanent_toughness
from ..effects.tokens import (
    ELDRAZI_SPAWN_TOKEN_CARD_DEF, FOOD_TOKEN_CARD_DEF, activate_eldrazi_spawn_sac, activate_food_sac, create_token,
)
from ..effects import undercity
from ..effects.win_check import deal_damage_to_opponent, gain_life
from ..mana import COLORS, tap_summoning_locked

GREEN_CARD_CATALOG = {
    "Forest": CardDef("Forest", CardType.LAND, None, EffectId.FOREST, basic=True, subtypes=("Forest",)),
    # Enters tapped unless you control 3+ OTHER Forests; when it enters
    # untapped, create a Food token. Subtype Forest -- counts itself toward
    # another Gingerbread Cabin's own "3+ other Forests" check.
    "Gingerbread Cabin": CardDef(
        "Gingerbread Cabin", CardType.LAND, None, EffectId.GINGERBREAD_CABIN, subtypes=("Forest",),
    ),
    "Generous Ent": CardDef(
        "Generous Ent", CardType.CREATURE, {"generic": 5, "G": 1}, EffectId.GENEROUS_ENT,
        forestcycling_cost={"generic": 1}, power=5, toughness=5,
    ),
    "Masked Vandal": CardDef(
        "Masked Vandal", CardType.CREATURE, {"generic": 1, "G": 1}, EffectId.MASKED_VANDAL, power=1, toughness=3,
        subtypes=("Elf",),  # real Masked Vandal is a Changeling (every creature type) -- only the Elf type matters in this pool (Priest of Titania / Wellwisher / Timberwatch Elf count it)
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
    # Real one-sided fight (cast_ram_through): target creature you control
    # deals its power to target creature you don't control, trample overflow
    # to that creature's controller. Both targets locked at cast, resolved
    # via the shared combat damage-marking + state-based death machinery.
    # Needs a real opponent creature to target, so it's only castable in a
    # 2-player game (_ram_through_extra_legal) -- a documented no-op in any
    # 1-player config, like every other opponent-requiring effect here.
    "Ram Through": CardDef("Ram Through", CardType.INSTANT, {"generic": 1, "G": 1}, EffectId.RAM_THROUGH),

    # --- G3: jund_wildfire ---
    "Pulse of Murasa": CardDef("Pulse of Murasa", CardType.INSTANT, {"generic": 2, "G": 1}, EffectId.PULSE_OF_MURASA),

    # --- G9: elves. NOTE (pre-existing engine convention): creature {T}
    # abilities here don't check summoning sickness (the mana framework never
    # has -- Overgrown Battlement/Saruli Caretaker already tap the turn they
    # enter), so these mana dorks / tap abilities inherit that uniformly. ---
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
    """Elves on THE battlefield -- both players' -- matching Priest of Titania
    / Wellwisher / Timberwatch Elf's "each Elf on the battlefield"."""
    return sum(1 for player in state.players for p in player.battlefield if _is_elf(p))


def wellwisher_activate(state, permanent):
    """{T}: You gain 1 life for each Elf on the battlefield (counted at
    resolution)."""
    permanent.tapped = True
    push_ability_to_stack(state, permanent.card_def, lambda st: gain_life(st, _count_elves(st)))


def timberwatch_elf_activate(state, permanent):
    """{T}: Target creature gets +X/+X until end of turn, X = # Elves on the
    battlefield (counted at resolution). Target locked at activation."""
    permanent.tapped = True
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

# Sagu Wildling's own creature half (Omen's real payoff) -- a distinct
# CardDef from the "Sagu Wildling" entry above (that one is the SORCERY
# side, EffectId.ROOST_SEEK, the only one ever in a decklist/
# game.CARD_DEFS by that name). Never registered in GREEN_CARD_CATALOG
# itself: it's only ever reached via ROOST_SEEK's own "omen" registry
# spec (ROOST_SEEK's own docstring), ontology matching every other token-
# like CardDef constant in this codebase (see game.effects.tokens).
SAGU_WILDLING_CREATURE_CARD_DEF = CardDef(
    "Sagu Wildling", CardType.CREATURE, {"generic": 4, "G": 1}, EffectId.SAGU_WILDLING, power=3, toughness=3,
)


def _is_defender(permanent):
    return permanent.card_def.extra.get("defender", False)


def forestcycle_generous_ent(state, card_def):
    """{1}, discard this card from hand: search library for a Forest, put
    into hand, shuffle. Only one possible target name, so this resolves
    immediately -- no model choice/pending resolution needed."""
    discard_from_hand_to_graveyard(state, card_def)
    find_to_hand(state, "Forest")


def _wall_of_roots_mana_available(state, permanent):
    """Real text has no {T} at all, just "Activate only once each turn" --
    this registry entry's own "mana_extra_available" gate is what enforces
    that limit (mana.py's "mana_no_tap": True on this same registry entry is
    what skips the tap itself), same used_this_turn flag/reset
    (turn.untap_step) Barrels of Blasting Jelly's filter ability already
    relies on."""
    return not permanent.flags.get("used_this_turn", False)


def _wall_of_roots_on_tap(state, permanent):
    """Put a -0/-1 counter on this creature -- a real counter now (Permanent.
    counters, stats.py), not a private activation count: toughness reaching
    0 (5 uses against this wall's own 5 toughness, same real number as
    before) is left entirely to the ordinary state-based-action death check
    (state_based.check_state_based_actions), same as any other creature's
    lethal damage -- no bespoke removal here."""
    permanent.flags["used_this_turn"] = True
    permanent.counters["-0/-1"] = permanent.counters.get("-0/-1", 0) + 1


def cast_roost_seek(state, card_def):
    """Sagu Wildling's Omen sorcery half: {G}, search library for a basic
    land -- two possible names here (Forest or Swamp), a real model
    choice, unlike Land Grant's single fixed target below.

    Omen's own rule, unlike every other sorcery/instant in this catalog
    AND unlike real Adventure: this does NOT exile itself. The real
    reminder text is "(Also shuffle this card.)", attached to this same
    search's own shuffle -- so the resolved sorcery goes straight into the
    LIBRARY (added here, before begin_search_fetch's own find_to_hand
    shuffles it -- one shuffle, not two, matching the single "then
    shuffle" the reminder text is modifying). The real creature half only
    becomes castable again once this same physical card is redrawn into
    hand, same as any ordinary card -- see EffectId.ROOST_SEEK's own
    "omen" registry spec / drl_env._omen_cast_legal for that second cast
    option, offered directly against state.hand, no exile involved."""
    if card_def in state.hand:
        state.hand.remove(card_def)  # normally already gone (removed at cast, push_to_stack); tolerant since this is the resolving spell
    state.library.append(card_def)  # Omen: shuffles ITSELF into the LIBRARY (stack -> library), not graveyard
    resolution.begin_search_fetch(state, lambda c: c.card_type == CardType.LAND, find_to_hand)


def cast_sagu_wildling_creature(state, card_def):
    """Sagu Wildling's real creature half: an ordinary hand cast once the
    same physical card has been redrawn (see cast_roost_seek above),
    offered via the "omen" registry spec (paid for by the normal
    begin_pay_cost path -- see drl_env._omen_cast_execute): {4}{G} 3/3
    Flying, ETB gain 3 life (EffectId.SAGU_WILDLING's own etb_trigger).
    `card_def` here is SAGU_WILDLING_CREATURE_CARD_DEF, a different object
    from whatever's actually sitting in state.hand (the sorcery side's own
    CardDef, despite sharing this display name) -- so the hand card has to
    be found by NAME, not identity."""
    hand_card = next((c for c in state.hand if c.name == "Sagu Wildling"), None)
    if hand_card is not None:
        state.hand.remove(hand_card)  # normally already removed at cast (_omen_cast_execute); tolerant for a direct call
    enters_battlefield(state, card_def)


def gatecreeper_vine_etb(state):
    """ETB: may search a basic land to hand -- optional even when a target
    exists, unlike Expedition Map/Crop Rotation's mandatory fetches."""
    resolution.begin_search_fetch(state, lambda c: c.card_type == CardType.LAND, find_to_hand, optional=True)


NYXBORN_HYDRA_MAX_X = 10  # ponytail: the action table needs a finite X range to enumerate; plan_payment already masks out any X this player can't currently afford, so this only bounds the table size, not what's ever actually playable. Raise it if a real game ever shows it binding.


def cast_nyxborn_hydra_creature(x):
    """Normal cast: {X}{G}, ETB with X +1/+1 counters on top of this card's
    own 0/1 base (a design choice, not Scryfall data -- this module's own
    docstring). Returns the (state, card_def) resolve build_action_table's
    own x_cast_modes loop expects, one closure per X value."""
    def resolve(state, card_def):
        permanent = cast_permanent_from_hand(state, card_def)
        if x:
            permanent.counters["+1/+1"] = x
    return resolve


def cast_nyxborn_hydra_bestow(x):
    """Bestow: {G}{G}{X}, enchant target creature you control -- the SAME
    physical card, cast as an Aura instead of a creature (cast_aura, the
    same infra every boggles Aura already uses; real MTG "enchant target
    creature" -- must be chosen before the stack, same "precast_choice"
    treatment Rancor/Ancestral Mask/Ethereal Armor already get, see
    drl_env._x_precast_choice_execute).

    on_attached sets type_override=ENCHANTMENT (so every "is this currently
    a creature" check -- combat eligibility, state-based death, Saruli
    Caretaker/Quirion Ranger/Dread Return's own "another creature" checks --
    correctly treats it as NOT a creature while attached, Permanent.card_type
    in state.py) and X +1/+1 counters, which pt_bonus/toughness_bonus below
    read back dynamically off THIS SAME permanent to size the bonus granted
    to whatever it enchants. Falling off (the enchanted creature dying) is
    NOT handled here -- see state_based._destroy_creature's own
    "becomes_creature_when_orphaned" branch (this EffectId's own registry
    flag below), real Bestow's fall-off rule: stays on the battlefield,
    becomes a creature again, keeps its own counters.

    no_target_fallback (real MTG 702.103e): if there is no legal creature to
    enchant AT ALL when this resolves -- never had one (begin_choose_any_
    target's own empty-candidate auto-None; confirmed reachable: paying this
    same cast's own {X} cost can tap the caster's last creature-mana-source
    to death, e.g. Wall of Roots' 5th self-damaging activation) or had one
    that died before resolving -- the spell does NOT fizzle to the graveyard
    like an ordinary Aura. It still enters the battlefield, as a creature,
    with its X +1/+1 counters, exactly the plain creature-mode cast
    (cast_nyxborn_hydra_creature) -- reused directly here rather than
    duplicated, so both paths mint an identical permanent."""
    def resolve(state, card_def):
        def _on_attached(state, aura):
            aura.type_override = CardType.ENCHANTMENT
            if x:
                aura.counters["+1/+1"] = x
        cast_aura(state, card_def, lambda p: p.card_type == CardType.CREATURE, on_attached=_on_attached,
                  no_target_fallback=cast_nyxborn_hydra_creature(x))
    return resolve


def cast_land_grant(state, card_def):
    """Search library for a Forest specifically -- single target name, so
    this resolves immediately once actually invoked, no pending resolution
    of its own. The pure effect, shared by both the normal {1}{G} cast
    (already deferred generically by drl_env._cast_execute's own mana-cost
    push, so this must NOT also push itself) and the free alt-cost cast
    below (which pushes itself, via cast_land_grant_alt, since nothing else
    would defer it) -- they differ only in how/whether the cost was paid
    and who's responsible for the stack push, never in the effect itself."""
    discard_from_hand_to_graveyard(state, card_def)
    find_to_hand(state, "Forest")


def cast_land_grant_alt(state, card_def):
    """Free alt-cost path: no cost at all, so already "fully paid for" the
    instant this is chosen -- push immediately instead of resolving now
    (drl_env._alt_cast_execute calls this synchronously, unlike the normal
    cast path, so nothing else will defer it), same treatment as Faithless
    Looting's free Flashback."""
    push_to_stack(state, card_def, cast_land_grant)


def land_grant_alt_cost_legal(state):
    """Land Grant's free alt-cost ("reveal your hand" instead of paying):
    legal only with no land cards in hand. Revealing the hand has no
    simulator-visible effect (solitaire, no opponent to show it to) --
    this predicate is the only real consequence of that clause."""
    return not any(c.card_type == CardType.LAND for c in state.hand)


def quirion_ranger_untap_legal(state, permanent):
    """Return a Forest you control to hand: untap target creature. Once
    each turn (the used_this_turn flag, reset for every permanent by
    untap_step regardless of which card set it -- same mechanism Barrels
    of Blasting Jelly's filter ability already relies on). No {T} in this
    ability's real cost, so -- unlike every other activated ability here
    -- it doesn't require this permanent itself to be untapped."""
    if permanent.flags.get("used_this_turn", False):
        return False
    return any(p.card_def.name == "Forest" for p in state.battlefield)


def quirion_ranger_untap_resolve(state, permanent):
    """"Return a Forest you control to hand: Untap target creature." Faithful
    now: the Forest bounce is the COST (paid immediately on activation), and
    the untap is a real activated ability that goes ON THE STACK -- its
    target (any creature on either battlefield, hexproof-aware) is locked
    here at activation, then the creature is untapped when the ability
    resolves off the stack, or the ability fizzles if that exact creature
    has left the battlefield by then (608.2b) or is illegal."""
    permanent.flags["used_this_turn"] = True
    forest = next(p for p in state.battlefield if p.card_def.name == "Forest")
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
            target.tapped = False
            state.log_event("untap", permanent=(target.card_def.name, target.slot), reason="quirion_ranger")

        push_to_stack(state, permanent.card_def, _resolve, reserves_hand_card=False, is_spell=False,  # activated ability -- not a spell
                      targets=() if captured is None else (captured,))

    resolution.begin_choose_any_target(
        state,
        lambda p: p.card_type == CardType.CREATURE and can_be_targeted(state, p, state.active_idx),
        _on_target,
        allow_players=False,
    )


def _cast_winding_way(state, card_def, chosen_type):
    """Choose creature or land at cast time -- two separate action-table
    entries, not a pending resolution. Reveal top 4; matches to hand, the
    rest to the graveyard. Fully deterministic given the chosen type -- no
    further model choice."""
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
    """Lead the Stampede: reveal top n; the model decides keep-to-hand
    (only if eligible_predicate matches) or bottom for each in turn, then
    -- if 2+ went to the bottom -- the order to put them there. Mirrors
    game.resolution.begin_scry_surveil's remaining/kept/disposed/ordered
    shape exactly, except "kept" lands in hand (not library top) and only
    eligible cards may be kept."""
    revealed = state.library[:n]
    del state.library[:n]
    resolution.begin_resolution(
        state, "select_to_hand", on_complete,
        remaining=revealed, eligible=eligible_predicate, kept=[], disposed=[], ordered=None,
    )
    if not revealed:
        # Library was already empty (this deck's own mill-out combo can get
        # here) -- nothing to decide, so complete immediately instead of
        # leaving a pending resolution with zero legal actions.
        resolution.complete_resolution(state)


def select_to_hand_options(state):
    """While deciding (remaining non-empty): keep (only if the front card
    is eligible) or bottom. While ordering (remaining empty, 2+ disposed,
    not yet all placed): one option per distinct name still waiting to be
    bottomed."""
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
    """{G}, sacrifice a land: search library for a land, put it directly
    onto the battlefield (its own normal tapped/ETB rules apply), shuffle.

    Real card text: "As an additional cost to cast this spell, sacrifice a
    land." -- a COST, not a target and not a resolve-time choice. Like any
    other additional cost (mana, a discard-as-a-cost), it has to be chosen
    and paid before the spell is fully cast, so it can't be interacted
    with or undone once the spell is actually on the stack -- exactly the
    same category as Fireblast/Lava Dart's own sacrifice alt-costs
    (cast_fireblast_alt above). This function runs directly as pay_cost's
    on_complete (drl_env._precast_choice_execute), right alongside the
    {G} mana payment, not deferred onto the stack itself -- only the
    SEARCH (the spell's actual effect) gets pushed to the stack and waits
    to resolve; the sacrifice, once paid here, is already done."""
    discard_from_hand_to_graveyard(state, card_def)

    def _on_sacrificed(state, _ok):
        def _resolve(state, card_def):
            def _on_fetch_chosen(state, land_name):
                found = find_and_remove_by_name(state, land_name)
                state.rng.shuffle(state.library)
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
    """The model picks at most one noncreature-colorless card among
    `revealed` to take, or declines -- a single decision, not a
    sequential walk like scry/surveil (Ancient Stirrings only ever takes
    one, if any). on_complete(state, chosen_card_or_None) runs once
    decided."""
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
    state.rng.shuffle(revealed)  # whatever's left (all of it, if declined) goes to the bottom
    state.library.extend(revealed)
    resolution.complete_resolution(state, chosen)


def cast_ancient_stirrings(state, card_def):
    """{G}: look at top 5, may take one noncreature colorless card to hand
    -- the model's choice among eligible ones, or decline -- rest to
    bottom in random order."""
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
    """Enchant Forest specifically -- the real card's own restriction, not
    "enchant land" generally (verified via Scryfall). The chosen color is
    recorded on the Aura's own flags; mana.py's own _bonus_mana_symbols
    reads it to add that color automatically alongside the Forest's own G
    every time it's tapped -- not a competing choice, unlike Abundant
    Growth below (see mana.py's module comments for why these two need
    genuinely different treatment despite reading like twins)."""
    cast_aura(state, card_def, lambda p: p.card_def.name == "Forest", on_attached=_utopia_sprawl_attach(color))


def abundant_growth_attach(state, aura):
    state.draw(1)
    # Real card: enchanted land taps for one mana of ANY color -- all 5
    # (WUBRG), not colorless. mana.py reads bonus_mana_colors as the choice.
    aura.flags["bonus_mana_colors"] = set(COLORS)


def cast_abundant_growth(state, card_def):
    cast_aura(state, card_def, lambda p: p.card_def.card_type == CardType.LAND, on_attached=abundant_growth_attach)


def begin_malevolent_rumble(state, revealed, on_complete):
    """Reveal the top four cards of your library. You may put a permanent
    card from among them into your hand. Put the rest into your
    graveyard. Its own dedicated resolution kind, not a reuse of
    ancient_stirrings' take-one-or-decline shape above: despite looking
    similar, the real disposal zone differs (graveyard here, library
    bottom there), so once you know Malevolent Rumble's real text they
    aren't actually the same primitive."""
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
    for c in revealed:  # the rest -- order is never read again
        state.move_card(c, state.graveyard)
    resolution.complete_resolution(state, chosen)


def cast_malevolent_rumble(state, card_def):
    """{1}{G}: reveal top 4, may take one permanent card to hand, rest to
    graveyard, create a 0/1 Eldrazi Spawn token ("Sacrifice this
    creature: Add {C}."). No Madness -- real card has none (verified via
    Scryfall; an earlier draft of this plan wrongly assumed it did)."""
    discard_from_hand_to_graveyard(state, card_def)
    create_token(state, ELDRAZI_SPAWN_TOKEN_CARD_DEF)
    top = state.library[:4]
    del state.library[:4]

    def _on_chosen(state, chosen):
        if chosen is not None:
            state.hand.append(chosen)

    begin_malevolent_rumble(state, top, _on_chosen)


def activate_bramble_wurm_gy(state, inst):
    """{2}{G}, Exile this card from your graveyard: gain 5 life. Removed
    from the graveyard only -- exile itself is untracked, same convention
    as Relic of Progenitus' own graveyard-exile ability
    (game.catalog.colorless_cards).

    Faithful timing: exiling from the graveyard is
    a COST, paid now on activation; gaining 5 life is the effect, so it
    goes on the stack (push_ability_to_stack) and resolves after a priority
    window.

    inst: the exact graveyard CardInstance whose ability this is -- see
    black_cards.flashback_dread_return."""
    state.graveyard.remove(inst)
    push_ability_to_stack(state, inst, lambda st: gain_life(st, 5))


def _ram_through_extra_legal(state):
    """Castable only with BOTH a legal "creature you control" and a legal
    "creature you don't control" available to target -- real Magic won't let
    you cast a spell without a legal choice for each of its targets. A
    1-player config has no opponent creature, so Ram Through is simply never
    castable there (documented no-op, same as every other opponent-requiring
    effect in this engine)."""
    if len(state.players) < 2:
        return False
    idx = state.active_idx
    mine = any(p.card_type == CardType.CREATURE and can_be_targeted(state, p, idx) for p in state.battlefield)
    theirs = any(p.card_type == CardType.CREATURE and can_be_targeted(state, p, idx) for p in state.opponent.battlefield)
    return mine and theirs


def cast_ram_through(state, card_def):
    """{1}{G} Instant: target creature you control deals damage equal to its
    power to target creature you don't control; if your creature has trample,
    the excess (beyond lethal) is dealt to that creature's controller instead.

    Two targets, both chosen and locked as the spell is cast (real Magic --
    precast_choice), then rechecked by object identity on resolution: if
    EITHER the source or the target creature has left the battlefield (or
    otherwise become illegal) by then, the spell fizzles (608.2c -- source
    gone means nothing to deal damage, target gone means nothing to deal it
    to; either way, no damage). The damage is marked like combat damage and
    lethality is resolved by the shared state-based-action check, so the
    target dies exactly when combat would kill it; trample overflow spills to
    its controller through the same deal_damage_to_opponent path combat
    uses."""
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

        # Second target: a creature you DON'T control (opponent's side),
        # hexproof/shroud-aware.
        resolution.begin_choose_any_target(
            state,
            lambda p: p not in state.battlefield and p.card_type == CardType.CREATURE and can_be_targeted(state, p, idx),
            _on_target_chosen,
            allow_players=False,
        )

    # First target: a creature you control (your own side). Shroud-aware
    # (your own shroud creature can't be targeted even by you); hexproof only
    # stops opponents, so your own hexproof creature stays a legal choice.
    resolution.begin_choose_any_target(
        state,
        lambda p: p in state.battlefield and p.card_type == CardType.CREATURE and can_be_targeted(state, p, idx),
        _on_source_chosen,
        allow_players=False,
    )


def masked_vandal_etb(state, permanent):
    """When Masked Vandal enters, you may exile a creature card from your
    graveyard. If you do, exile target artifact or enchantment an opponent
    controls. (Changeling is a no-op -- no tribal-synergy card exists in this
    pool, same documented drop as Rooftop Percher's Changeling.)

    Target-at-promotion (project_targeted_triggered_abilities): the target
    (opponent artifact/enchantment) is chosen and captured by object identity as
    the ability goes on the stack. A targeted ability needs a legal target to be
    put on the stack at all (603.3c) -- with none (a 1-player config, or an
    opponent with only hexproof/shroud artifacts/enchantments), the ability does
    nothing and isn't put on the stack, so nothing is offered. At RESOLUTION: the
    "you may exile a creature card from your graveyard" choice, and IF taken,
    exile the captured target if it is still legal (else the exile fizzles,
    608.2b). Declining the graveyard exile simply makes the "if you do" not
    happen -- that is not a fizzle."""
    if len(state.players) < 2:
        return

    def _targetable(p):
        return p.card_type in (CardType.ARTIFACT, CardType.ENCHANTMENT) and can_be_targeted(state, p, state.active_idx)

    if not any(_targetable(p) for p in state.opponent.battlefield):
        return  # no legal target -> not put on the stack (603.3c)

    def _on_target(state, choice):
        if choice is None:
            return  # (safety) a legal target was confirmed above, so this shouldn't happen
        tname, tslot = choice
        target = next(p for p in state.opponent.battlefield if p.card_def.name == tname and p.slot == tslot)  # captured at promotion

        def _resolve(state, card_def):
            def _after_gy(state, chosen):
                if chosen is None:
                    return  # declined the "you may" -> the "if you do" doesn't happen (no fizzle)
                state.graveyard.remove(chosen)  # exile the chosen creature (untracked) -- paying the "if you do"
                state.log_event("zone_move", card=chosen.name, from_zone="graveyard", to_zone="exile_untracked", reason="masked_vandal")
                # Exile the captured target if it is still a legal target (608.2b, object identity).
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
    """"This land enters tapped unless you control three or more other
    Forests." Counts Forest-subtype permanents already on the battlefield --
    this Cabin isn't added until after enters_battlefield reads this, so
    they're all genuinely "other" (a basic Forest is a Forest; another
    Gingerbread Cabin is too)."""
    forests = sum(1 for p in state.battlefield if "Forest" in card_subtypes(p.card_def))
    return forests < 3


def gingerbread_cabin_etb(state, permanent):
    """"When this land enters untapped, create a Food token." Reads how it
    entered (flags["entered_tapped"], set by enters_battlefield) rather than
    its current tapped-ness, which a later tap-for-mana could have flipped
    before this queued ETB resolves."""
    if not permanent.flags.get("entered_tapped", False):
        create_token(state, FOOD_TOKEN_CARD_DEF)


def _pulse_of_murasa_eligible(card_def):
    return card_def.card_type in (CardType.CREATURE, CardType.LAND)


def cast_pulse_of_murasa(state, card_def):
    """{2}{G}: "Return target creature or land card from A graveyard to its
    owner's hand. You gain 6 life." Target chosen at cast (precast_choice) from
    EITHER player's graveyard, rechecked at resolution -- if that card has left
    the graveyard by then (graveyard hate in response), the spell has no legal
    target and does nothing, lifegain included (608.2c). The returned card goes
    to ITS OWNER's hand (the graveyard it was in); the caster gains the 6 life.

    Cross-graveyard is offered by name across both graveyards; a same-named
    card present in both resolves caster's-graveyard-first (the pre-existing
    engine-wide fungible-by-name convention for graveyard/permanent selection,
    and the only rationally-wanted target anyway -- returning an opponent's
    card to THEIR hand never helps you)."""
    caster_idx = state.active_idx

    def _on_chosen(state, chosen):
        def _resolve(state, card_def):
            discard_from_hand_to_graveyard(state, card_def)  # Pulse itself -> caster's graveyard
            # chosen: the exact graveyard instance locked at cast. Fizzle (608.2b)
            # if it has since left -- find whichever player's graveyard still holds it.
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
        "mana": ("fixed", "G"),  # {T}: Add {G} (functional twin of Llanowar Elves)
    },
    EffectId.PRIEST_OF_TITANIA: {
        "cast": {"resolve": lambda state, card_def: cast_permanent_from_hand(state, card_def)},
        "mana": ("count_all", "G", _is_elf),  # {T}: Add {G} for each Elf on the battlefield (both sides)
    },
    EffectId.WELLWISHER: {
        "cast": {"resolve": lambda state, card_def: cast_permanent_from_hand(state, card_def)},
        "activated_abilities": {
            "lifegain": {  # {T}: gain 1 life per Elf -- non-mana tap ability (legal/resolve path)
                # {T} ability -> summoning-sickness gated (302.6): untapped AND not a summoning-sick hasteless creature.
                "legal": lambda state, permanent: not permanent.tapped and not tap_summoning_locked(state, permanent),
                "resolve": lambda state, permanent: wellwisher_activate(state, permanent),
            },
        },
    },
    EffectId.TIMBERWATCH_ELF: {
        "cast": {"resolve": lambda state, card_def: cast_permanent_from_hand(state, card_def)},
        "activated_abilities": {
            "pump": {  # {T}: target creature +X/+X (X = # Elves) until EOT
                # {T} ability -> summoning-sickness gated (302.6).
                "legal": lambda state, permanent: not permanent.tapped and not tap_summoning_locked(state, permanent),
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
        # Reach + ETB "create a Food token" -- the real hard-cast, alongside
        # its Forestcycling {1}. Reach lets it block fliers (combat.can_block);
        # the Food token has its own "{2},{T},Sacrifice: gain 3 life" ability
        # (FOOD_TOKEN registry below). Power/toughness stays this catalog's own
        # design-choice value (not Scryfall's 5/7).
        "cast": {"resolve": lambda state, card_def: cast_permanent_from_hand(state, card_def)},
        "keywords": {"reach"},
        "etb_trigger": lambda state, permanent: create_token(state, FOOD_TOKEN_CARD_DEF),
        "forestcycle": {
            "cost_key": "forestcycling_cost",
            "resolve": lambda state, card_def: forestcycle_generous_ent(state, card_def),
        },
    },
    EffectId.FOOD_TOKEN: {
        # "{2}, {T}, Sacrifice this token: You gain 3 life." Never a decklist
        # card -- only ever created by Generous Ent's ETB (create_token). Its
        # {2} + untapped precondition come from the generic cost_key wiring,
        # same as Blood/Candy Trail.
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
        # The extra cost taps ANOTHER untapped creature you control -- a COST
        # CHOICE (602.5g), not a target, so the AGENT chooses which (no engine
        # auto-pick; which creature you tap is a real decision -- keep a
        # blocker vs an attacker up). Two sequential steps -- choose the
        # creature to tap, THEN choose the produced color (matches real
        # sequencing: the cost is paid before the ability's own color-choice
        # effect resolves) -- via a mana_subdecision (drl_env._mana_extra_
        # choose_legal/_execute, game.begin_mana_subdecision), never the
        # stack (605.1a), same as any other mana ability. mana_subdecision is
        # a field SEPARATE from state.pending_resolution specifically so this
        # stays legal in ANY priority window, even mid-resolution of
        # something else (605.1a/605.3b -- confirmed load-bearing: Quirion
        # Ranger targeting an opposing Ward creature opens exactly this
        # window in real league play) -- see state.mana_subdecision's own
        # docstring for the full reasoning.
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
        # Two real cast modes, one action per X (0..NYXBORN_HYDRA_MAX_X)
        # each -- drl_env.build_action_table's own "x_cast_modes" loop,
        # generic X-cost machinery parallel to cast_modes/omen/plot.
        "x_cast_modes": {
            "creature": {"cost": {"G": 1}, "max_x": NYXBORN_HYDRA_MAX_X, "resolve": cast_nyxborn_hydra_creature},
            "bestow": {
                "cost": {"G": 2}, "max_x": NYXBORN_HYDRA_MAX_X, "precast_choice": True,
                "extra_legal": lambda state: any_creature_on_battlefield(state),
                "resolve": cast_nyxborn_hydra_bestow,
            },
        },
        # Read back only while Bestowed (an ordinary creature-mode Hydra
        # never has "enchanting" set on anything, so _enchanting_auras
        # never even looks these up for it) -- the bonus IS this permanent's
        # OWN +1/+1 counters, not a fixed constant like Rancor's +2.
        "pt_bonus": lambda state, aura: aura.counters.get("+1/+1", 0),
        "toughness_bonus": lambda state, aura: aura.counters.get("+1/+1", 0),
        # Real Bestow fall-off: stays on the battlefield and becomes a
        # creature again (not graveyarded/returned to hand like an ordinary
        # orphaned Aura) -- state_based._destroy_creature's own third
        # branch.
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
        # untap target creature is now a real ability on the stack, any creature
        # either side (quirion_ranger_untap_resolve) -- hence choose_any_target.
        "pending_kinds": {"choose_any_target"},
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
            # NOT "targeted" -- real MTG text has no "target" here, "sacrifice
            # a land" is an ADDITIONAL COST (paid alongside the {G} mana cost,
            # before the spell is even fully cast), same category as
            # cast_fireblast_alt's own sacrifice cost -- see cast_crop_
            # rotation's own docstring. Reuses the same drl_env.
            # _precast_choice_execute routing as Auras' real targets purely
            # because both need "resolve runs immediately once paid, manages
            # its own push_to_stack" instead of the generic auto-push -- not
            # because this is a target.
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
        # Real text: Trample + Reach. Reach lets it block flyers
        # (combat.can_block) -- e.g. an opposing Kitchen Imp.
        "keywords": {"trample", "reach"},
        "graveyard_ability": {
            "cost_key": "gy_ability_cost",
            "resolve": lambda state, card_def: activate_bramble_wurm_gy(state, card_def),
        },
    },

    # --- boggles deck ---
    EffectId.GLADECOVER_SCOUT: {
        # Vanilla 1/1 with hexproof for {G} -- now a REAL targeting
        # restriction (stats.can_be_targeted), same as Slippery Bogle.
        "cast": {"resolve": lambda state, card_def: cast_permanent_from_hand(state, card_def)},
        "keywords": {"hexproof"},
    },
    EffectId.SILHANA_LEDGEWALKER: {
        # Real text: Hexproof + "can't be blocked except by creatures with
        # flying". The evasion is its OWN attacker-side restriction
        # ("cant_be_blocked_except_by_flying", combat.can_block) -- distinct
        # from real flying, so unlike a true flier (Kitchen Imp) Silhana can
        # NOT itself block flyers. Hexproof is the targeting restriction
        # (stats.can_be_targeted).
        "cast": {"resolve": lambda state, card_def: cast_permanent_from_hand(state, card_def)},
        "keywords": {"cant_be_blocked_except_by_flying", "hexproof"},
    },
    EffectId.RANCOR: {
        # Real text also grants trample and
        # returns Rancor to hand instead of the graveyard when it's put
        # there from the battlefield -- modeled via
        # returns_to_hand_when_orphaned now that combat death (step 6)
        # makes that reachable; see effects.state_based._destroy_creature.
        # +2/+0, power only -- no toughness_bonus (unlike Ancestral Mask/
        # Ethereal Armor/Cartouche of Solidarity/Armadillo Cloak, all
        # symmetric +X/+X) -- see permanent_toughness's own docstring.
        "cast": {
            "resolve": lambda state, card_def: cast_rancor(state, card_def),
            "extra_legal": lambda state: any_creature_on_battlefield(state),
            "precast_choice": True,  # real MTG "enchant target creature" -- must be chosen before the stack, see drl_env._precast_choice_execute
        },
        "pending_kinds": {"choose_any_target"},  # Aura: cast_aura now targets any creature (either side), hexproof-aware
        "pt_bonus": lambda state, aura: 2,
        "returns_to_hand_when_orphaned": True,
        "keywords": {"trample"},
    },
    EffectId.ANCESTRAL_MASK: {
        # Real text: +2/+2 for each OTHER enchantment you control (unlike
        # Ethereal Armor's "each enchantment," this one excludes itself --
        # verified via Scryfall).
        "cast": {
            "resolve": lambda state, card_def: cast_ancestral_mask(state, card_def),
            "extra_legal": lambda state: any_creature_on_battlefield(state),
            "precast_choice": True,  # real MTG "enchant target creature" -- must be chosen before the stack, see drl_env._precast_choice_execute
        },
        "pending_kinds": {"choose_any_target"},  # Aura: cast_aura now targets any creature (either side), hexproof-aware
        "pt_bonus": lambda state, aura: 2 * (enchantment_count(state, aura) - 1),
        "toughness_bonus": lambda state, aura: 2 * (enchantment_count(state, aura) - 1),
    },
    EffectId.UTOPIA_SPRAWL: {
        # Real "As ~ enters, choose a color" -- one cast mode per color of
        # the five (WUBRG); the chosen color is the bonus mana the enchanted
        # Forest adds. precast_choice: the Forest target is chosen at cast.
        "cast_modes": {
            "green": {
                "resolve": lambda state, card_def: cast_utopia_sprawl(state, card_def, "G"),
                "extra_legal": lambda state: any(p.card_def.name == "Forest" for p in state.battlefield),
                "precast_choice": True,  # real MTG "enchant target Forest" -- must be chosen before the stack, see drl_env._precast_choice_execute
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
        "pending_kinds": {"choose_any_target"},  # Aura: cast_aura targets any Forest (either side), hexproof/shroud aware
    },
    EffectId.ABUNDANT_GROWTH: {
        "cast": {
            "resolve": lambda state, card_def: cast_abundant_growth(state, card_def),
            "extra_legal": lambda state: any(p.card_def.card_type == CardType.LAND for p in state.battlefield),
            "precast_choice": True,  # real MTG "enchant target land" -- must be chosen before the stack, see drl_env._precast_choice_execute
        },
        "pending_kinds": {"choose_any_target"},  # Aura: cast_aura targets any land (either side), hexproof/shroud aware
        # Static fact for drl_env.build_action_table's own action-table
        # pre-registration (which land x color "Choose" actions need to
        # exist at all, before any game state does) -- kept in sync by
        # hand with abundant_growth_attach's runtime aura.flags value.
        "grants_mana_colors": {"G", "W"},
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
        # "When this creature enters, you take the initiative" -- its controller
        # (active_idx at ETB resolution). take_initiative also queues the venture.
        "etb_trigger": lambda state, permanent: undercity.take_initiative(state, state.active_idx),
        # Every resolution kind an Undercity venture can open, so
        # drl_env.build_action_table pre-registers the matching actions for any
        # deck running Avenging Hunter: room branch, the 6 rooms with choices,
        # and Throne's reveal-and-pick.
        "pending_kinds": {"choose_room", "throne_reveal", "search_fetch", "scry", "choose_any_target", "choose_target_player"},
    },
    EffectId.SKELETON_TOKEN: {"keywords": {"menace"}},  # 4/1 Undercity Catacombs Skeleton
    # Ram Through (functional blank): deliberately no entry -- see the
    # comment on its CardDef above.
}

