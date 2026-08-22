"""Colorless-identity card catalog: lands/artifacts with no colored mana
symbol in cost or output. COLORLESS_CARD_CATALOG (name -> CardDef) and
COLORLESS_EFFECT_REGISTRY (EffectId -> spec). Cost/type/oracle text is
from Scryfall; power/toughness is a design choice.

Rooftop Percher / Boulderbranch Golem / Maelstrom Colossus / Pinnacle
Kill-Ship are cast at their real cost/stats, with each card's ETB
life-gain wired for real; each one's other complicating clause is
handled as follows:
- Rooftop Percher: Changeling is a no-op (no tribal-synergy card exists
  here). Its "exile up to two target cards from graveyards" IS
  implemented (rooftop_percher_etb) -- targets chosen from either
  player's graveyard as the ability goes on the stack, exiling every
  still-present target at resolution; the +3 life is unconditional even
  if every target has left the graveyard (see that function's own
  AUTHORIZED SIMPLIFICATION note).
- Boulderbranch Golem: Prototype ({3}{G} for a 3/3) is implemented as a
  second CardDef (BOULDERBRANCH_GOLEM_PROTOTYPE_CARD_DEF) offered via
  the Omen cast machinery; its ETB life gain is baked per mode (6
  normal / 3 prototype).
- Maelstrom Colossus: Cascade is implemented for real (cast_maelstrom_
  colossus) -- it invokes the hit card's own "cast" resolve by
  temporarily inserting it into state.hand (CardDefs are interned per
  name, so existing hand-removal code finds it). extra_legal is still
  checked first; if the hit's resolve opens a further pending
  resolution, Colossus's battlefield entry chains onto that
  resolution's on_complete.
- Pinnacle Kill-Ship: Station and its ETB are both implemented for
  real. Station taps another creature you control and adds that many
  charge counters; at 7+ the permanent becomes a 7/7 flying creature
  (Permanent.type_override, read generically by every "is this a
  creature" check). The ETB is a harmless no-op with no legal target.

"mana" shapes: ("tron",) Tron's doubling rule; ("fixed", symbol) always
that symbol; ("flexible", {symbols}) caller chooses one. "filter_mana":
{"colors": {...}} marks a colored-pip filter ability -- spend one
floating pip, then choose a color -- as opposed to a plain fixed mana
source."""

from .. import registry
from ..cards import CardDef, CardType, EffectId
from ..effects.casting import (
    _log_target_fizzle, capture_any_target, cast_permanent_from_hand, enters_battlefield, has_creature_target,
    target_still_legal,
)
from ..effects.stack import push_ability_to_stack, push_to_stack
from ..effects.shared import (
    affinity_reduction, discard_from_hand_to_graveyard, find_and_remove_by_name, find_to_hand,
 shuffle_library, tap_for_cost,
)
from ..effects.state_based import check_state_based_actions, sacrifice_to_graveyard
from ..effects.stats import can_be_targeted, permanent_power
from ..effects.tokens import activate_blood_sac, activate_clue_sac, activate_map_sac
from ..effects.win_check import gain_life
from ..mana import COLORS, float_mana
from ..resolution import (
    begin_choose_any_target, begin_choose_graveyard_card, begin_choose_mana_color, begin_choose_opponent_permanent,
    begin_choose_permanent, begin_choose_target_player, begin_choose_up_to_graveyard, begin_may_cast, begin_pay_unless,
    begin_scry_surveil, begin_search_fetch, scry, surveil,
)
from ..turn import Speed

COLORLESS_CARD_CATALOG = {
    "Urza's Mine": CardDef("Urza's Mine", CardType.LAND, None, EffectId.TRON_LAND, tron_type="Mine"),
    "Urza's Power Plant": CardDef("Urza's Power Plant", CardType.LAND, None, EffectId.TRON_LAND, tron_type="Power Plant"),
    "Urza's Tower": CardDef("Urza's Tower", CardType.LAND, None, EffectId.TRON_LAND, tron_type="Tower"),
    "Tocasia's Dig Site": CardDef(
        "Tocasia's Dig Site", CardType.LAND, None, EffectId.TOCASIA_DIG_SITE,
        surveil_ability_cost={"generic": 3},
    ),
    "Conduit Pylons": CardDef("Conduit Pylons", CardType.LAND, None, EffectId.CONDUIT_PYLONS),
    "Expedition Map": CardDef(
        "Expedition Map", CardType.ARTIFACT, {"generic": 1}, EffectId.EXPEDITION_MAP, ability_cost={"generic": 2},
    ),
    "Bonder's Ornament": CardDef(
        "Bonder's Ornament", CardType.ARTIFACT, {"generic": 3}, EffectId.BONDERS_ORNAMENT,
        draw_ability_cost={"generic": 4},
    ),
    "Candy Trail": CardDef(
        "Candy Trail", CardType.ARTIFACT, {"generic": 1}, EffectId.CANDY_TRAIL, sac_ability_cost={"generic": 2},
    ),
    "Barrels of Blasting Jelly": CardDef(
        "Barrels of Blasting Jelly", CardType.ARTIFACT, {"generic": 1}, EffectId.BARRELS_OF_BLASTING_JELLY,
        blast_ability_cost={"generic": 5},
    ),
    "Relic of Progenitus": CardDef(
        "Relic of Progenitus", CardType.ARTIFACT, {"generic": 1}, EffectId.RELIC_OF_PROGENITUS,
        draw_ability_cost={"generic": 1}, graveyard_exile_ability_cost={},
    ),
    "Lotus Petal": CardDef("Lotus Petal", CardType.ARTIFACT, {}, EffectId.LOTUS_PETAL),
    "Rooftop Percher": CardDef(
        "Rooftop Percher", CardType.CREATURE, {"generic": 5}, EffectId.ROOFTOP_PERCHER, power=3, toughness=3,
    ),
    "Boulderbranch Golem": CardDef(
        "Boulderbranch Golem", CardType.CREATURE, {"generic": 7}, EffectId.BOULDERBRANCH_GOLEM, power=6, toughness=5,
    ),
    "Maelstrom Colossus": CardDef(
        "Maelstrom Colossus", CardType.CREATURE, {"generic": 8}, EffectId.MAELSTROM_COLOSSUS, power=7, toughness=7,
    ),
    "Pinnacle Kill-Ship": CardDef("Pinnacle Kill-Ship", CardType.ARTIFACT, {"generic": 7}, EffectId.PINNACLE_KILL_SHIP),

    # --- boggles deck ---
    "Ash Barrens": CardDef("Ash Barrens", CardType.LAND, None, EffectId.ASH_BARRENS, cycling_cost={"generic": 1}),

    # --- jund_wildfire ---
    # {T}: Add {C}. {T}, Sacrifice: fetch a basic Swamp/Mountain/Forest tapped.
    # Cycling {B}{R}{G}: discard, draw a card.
    "Twisted Landscape": CardDef(
        "Twisted Landscape", CardType.LAND, None, EffectId.TWISTED_LANDSCAPE,
        fetch_ability_cost={}, cycling_cost={"B": 1, "R": 1, "G": 1},
    ),

    # --- G7: grixis_affinity ---
    "Myr Enforcer": CardDef("Myr Enforcer", CardType.CREATURE, {"generic": 7}, EffectId.MYR_ENFORCER, power=4, toughness=4, artifact=True),

    # --- G8: artifact value engines (grixis_affinity / jund_wildfire) ---
    "Ichor Wellspring": CardDef("Ichor Wellspring", CardType.ARTIFACT, {"generic": 2}, EffectId.ICHOR_WELLSPRING),
    "Chromatic Star": CardDef("Chromatic Star", CardType.ARTIFACT, {"generic": 1}, EffectId.CHROMATIC_STAR, sac_ability_cost={"generic": 1}),
    "Nihil Spellbomb": CardDef("Nihil Spellbomb", CardType.ARTIFACT, {"generic": 1}, EffectId.NIHIL_SPELLBOMB, sac_ability_cost={}),
    "Lembas": CardDef("Lembas", CardType.ARTIFACT, {"generic": 2}, EffectId.LEMBAS, sac_ability_cost={"generic": 2}),
}


def ichor_wellspring_draw(state, *_a):
    """ETB and dies both "draw a card"."""
    state.draw(1)


def chromatic_star_mana(state, permanent):
    """{1}, {T}, Sacrifice: Add one mana of any color. Sacrificing fires its
    dies-trigger (draw); the activating player then chooses a color."""
    sacrifice_to_graveyard(state, permanent)  # queues the dies-trigger (draw)

    def _add_color(state, color):
        float_mana(state, [color])
        state.log_event("mana_tap", permanent=(permanent.card_def.name, permanent.slot), mode="chromatic_star", produced=[color])

    begin_choose_mana_color(state, _add_color)


def nihil_spellbomb_sac(state, permanent):
    """{T}, Sacrifice: Exile target player's graveyard."""
    sacrifice_to_graveyard(state, permanent)  # queues the dies-trigger

    def _on_player(state, idx):
        def _effect(st):
            st.players[idx].graveyard.clear()
            st.log_event("graveyard_exiled", player_idx=idx)
        push_ability_to_stack(state, permanent.card_def, _effect)

    begin_choose_target_player(state, _on_player)


def nihil_spellbomb_dies(state, permanent):
    """When put into a graveyard from the battlefield, its owner may pay {B}
    to draw a card. Owner is read from flags, since the permanent is
    already off the battlefield by resolution."""
    payer = permanent.flags["owner_idx"]

    def _on_result(state, paid):
        if paid:
            state.draw(1)

    begin_pay_unless(state, payer, {"B": 1}, _on_result)


def lembas_etb(state):
    """When Lembas enters: scry 1, then draw a card."""
    begin_scry_surveil(state, "scry", 1, on_complete=lambda s: s.draw(1))


def lembas_sac(state, permanent):
    """{2}, {T}, Sacrifice: You gain 3 life. Sacrificing fires its dies-trigger
    (shuffle back into the library)."""
    sacrifice_to_graveyard(state, permanent)  # queues the dies-trigger (shuffle into library)
    push_ability_to_stack(state, permanent.card_def, lambda st: gain_life(st, 3))


def lembas_dies(state, permanent):
    """When put into a graveyard from the battlefield, its owner shuffles it
    into their library. Tracks the exact card instance that died; a no-op
    if it's already left the graveyard (e.g. reanimated first)."""
    inst = permanent.flags.get("graveyard_instance")
    if inst is not None and inst in state.graveyard:
        state.graveyard.remove(inst)
        state.library.append(inst.card_def)  # library is DEFERRED (CardDefs)
        shuffle_library(state)
        state.log_event("zone_move", card=inst.name, from_zone="graveyard", to_zone="library", reason="lembas_shuffle")


def _twisted_landscape_fetch_eligible(card_def):
    return card_def.extra.get("basic", False) and card_def.name in ("Swamp", "Mountain", "Forest")


def activate_twisted_landscape_fetch(state, permanent):
    """{T}, Sacrifice: search library for a basic Swamp, Mountain, or Forest,
    put it onto the battlefield tapped, then shuffle."""
    sacrifice_to_graveyard(state, permanent)  # queues the dies-trigger (Gixian Infiltrator)

    def _effect(st):
        def _on_fetch(st, land_name):
            found = find_and_remove_by_name(st, land_name)
            shuffle_library(st)
            if found:
                enters_battlefield(st, found, force_tapped=True)

        begin_search_fetch(st, _twisted_landscape_fetch_eligible, _on_fetch)  # fizzles to None if no basic left

    push_ability_to_stack(state, permanent.card_def, _effect)


def cycle_twisted_landscape(state, card_def):
    """Cycling {B}{R}{G}: discard this card, draw a card."""
    discard_from_hand_to_graveyard(state, card_def)
    state.draw(1)


def activate_tocasia_dig_site_surveil(state, permanent):
    """{3}, T: Surveil 1."""
    tap_for_cost(state, permanent)
    push_ability_to_stack(state, permanent.card_def, lambda st: surveil(st, 1))


def activate_expedition_map(state, permanent):
    """{2}, T, Sacrifice: search library for a land (the model's choice)."""
    sacrifice_to_graveyard(state, permanent)  # queues the dies-trigger (Gixian Infiltrator)
    push_ability_to_stack(state, permanent.card_def, lambda st: begin_search_fetch(st, lambda c: c.card_type == CardType.LAND, find_to_hand))


def activate_bonders_ornament_draw(state, permanent):
    """{4}, T: draw a card."""
    tap_for_cost(state, permanent)
    push_ability_to_stack(state, permanent.card_def, lambda st: st.draw(1))


def activate_candy_trail_sac(state, permanent):
    """{2}, T, Sacrifice: gain 3 life and draw a card."""
    sacrifice_to_graveyard(state, permanent)  # queues the dies-trigger (Gixian Infiltrator)

    def _effect(state):
        gain_life(state, 3)
        state.draw(1)

    push_ability_to_stack(state, permanent.card_def, _effect)


def activate_relic_of_progenitus_draw(state, permanent):
    """{1}, Exile this artifact: exile all graveyards, draw a card."""
    state.battlefield.remove(permanent)  # exiled, not graveyard; exile is untracked
    state.log_event(
        "zone_move", permanent=(permanent.card_def.name, permanent.slot), from_zone="battlefield",
        to_zone="exile_untracked", reason="activate_exile_self",
    )

    def _effect(state):
        for player in state.players:
            player.graveyard.clear()
        state.log_event("graveyards_exiled")
        state.draw(1)

    push_ability_to_stack(state, permanent.card_def, _effect)


def activate_relic_of_progenitus_exile(state, permanent):
    """{T}: Target player exiles a card from their graveyard, chosen by that
    player, not the activator. Target locked at activation; active_idx
    flips to the targeted player for their forced choice, then restores."""
    tap_for_cost(state, permanent)  # {T} -- a COST, paid now on activation

    def _on_player_chosen(state, idx):
        def _effect(state):
            # flip to the targeted player so THEY pick the card; restored below
            activator = state.active_idx
            target_player = state.players[idx]

            def _on_card_chosen(state, chosen):
                state.active_idx = activator  # the targeted player's forced choice is done
                if chosen is None:
                    return
                target_player.graveyard.remove(chosen)  # the exact chosen instance; exiled (untracked)
                state.log_event(
                    "zone_move", card=chosen.name, from_zone="graveyard", to_zone="exile_untracked",
                    target_player_idx=idx,
                )

            state.active_idx = idx
            begin_choose_graveyard_card(state, lambda c: True, _on_card_chosen, graveyard=target_player.graveyard)

        push_ability_to_stack(state, permanent.card_def, _effect)

    begin_choose_target_player(state, _on_player_chosen)


def _lotus_petal_on_tap(state, permanent):
    """{T}, Sacrifice: add one mana of any color -- consumed, not just tapped."""
    sacrifice_to_graveyard(state, permanent)  # queues the dies-trigger (Gixian Infiltrator)


def _treasure_on_tap(state, permanent):
    """{T}, Sacrifice: add one mana of any color. A token, so it ceases to
    exist rather than going to the graveyard."""
    sacrifice_to_graveyard(state, permanent)  # ceases to exist; queues the dies-trigger (Gixian Infiltrator)


def _basic_land(card_def):
    return card_def.extra.get("basic", False)


def cycle_ash_barrens(state, card_def):
    """Basic landcycling {1}: discard this card, search library for a basic
    land, put it into hand, shuffle. No draw-a-card rider."""
    discard_from_hand_to_graveyard(state, card_def)
    begin_search_fetch(state, _basic_land, find_to_hand)


def _mana_value(cast_cost):
    """Total converted cost of a cast_cost dict, e.g. {"generic": 2, "G": 1} -> 3."""
    return sum(cast_cost.values())


def cast_maelstrom_colossus(state, card_def):
    """Cascade: exile cards from the top of the library until a nonland
    card with mana value less than 8 is exiled; cast it without paying its
    mana cost, then put the rest on the bottom in a random order.

    Casting the hit reuses its own registry "cast" spec unchanged, by
    temporarily inserting it into state.hand (CardDefs are interned per
    name, so existing hand-removal code finds it). Skipped if the hit has
    no "cast" spec, or its own extra_legal fails (Cascade only waives the
    mana cost, not other costs/preconditions).

    If the hit's own resolve opens a further pending resolution (e.g.
    Ancient Stirrings' take-one-or-decline), Colossus's battlefield entry
    chains onto that resolution's on_complete instead of running
    immediately, so it lands only once the cascaded card is fully done."""
    discard_from_hand_to_graveyard(state, card_def)
    exiled = []
    hit = None
    while state.library:
        card = state.library.pop(0)
        exiled.append(card)
        if card.card_type != CardType.LAND and _mana_value(card.cast_cost) < 8:
            hit = card
            break

    cast_spec = registry.EFFECT_REGISTRY.get(hit.effect_id, {}).get("cast") if hit is not None else None
    extra_legal = cast_spec.get("extra_legal") if cast_spec is not None else None
    can_cast = cast_spec is not None and (extra_legal is None or extra_legal(state))

    def _bottom(cards):
        state.rng.shuffle(cards)  # real text: "the exiled cards, on the bottom in a random order"
        state.library.extend(cards)

    def _enter_colossus(state):
        enters_battlefield(state, card_def)

    if not can_cast:
        # A whiff, a hit with no "cast" spec, or one whose extra_legal fails:
        # there is no may-cast decision -- every exiled card just bottoms.
        _bottom(exiled)
        _enter_colossus(state)
        return

    def _resolve_hit(state, do_cast):
        # Real Cascade is "you MAY cast it." Cast -> the OTHERS bottom and the
        # hit is cast for free via its own normal cast machinery. Decline -> the
        # hit bottoms with the others (it's still one of "the exiled cards").
        if not do_cast:
            _bottom(exiled)
            _enter_colossus(state)
            return
        _bottom([c for c in exiled if c is not hit])
        state.hand.append(hit)
        cast_spec["resolve"](state, hit)
        if state.pending_resolution is not None:
            pending = state.pending_resolution
            inner_on_complete = pending["on_complete"]

            def _finish_cascade(state, *args):
                inner_on_complete(state, *args)
                _enter_colossus(state)

            pending["on_complete"] = _finish_cascade
            return
        _enter_colossus(state)

    # The CASTER's own may-cast decision (Cascade is the caster's trigger -- no
    # active_idx flip). Two drl_env actions "Cast (may)" / "Decline (may)".
    begin_may_cast(state, on_complete=_resolve_hit)


def rooftop_percher_etb(state, permanent):
    """ETB: exile up to two target cards from graveyards. You gain 3 life.
    Up to two targets (from either player's graveyard) are chosen as the
    ability goes on the stack; the exile fizzles per-target at resolution,
    doing nothing only if all chosen targets have left the graveyard.

    The +3 life is a SEPARATE, non-targeted effect that ALWAYS happens at
    resolution -- even if the exile fully fizzles.
    # AUTHORIZED SIMPLIFICATION (owner, 2026-07-29): the ability never
    # wholesale-fizzles on all-targets-illegal (strict 608.2b would counter it,
    # dropping the life gain too); the life gain is unconditional here."""
    percher_def = registry.CARD_DEFS["Rooftop Percher"]
    combined = [c for pl in state.players for c in pl.graveyard]  # either player's graveyard

    def _on_targets(state, chosen):
        captured = list(chosen)  # exact graveyard instances, locked as the ability is put on the stack

        def _resolve(state, card_def):
            gain_life(state, 3)  # unconditional (owner directive)
            survivors = [inst for inst in captured if any(inst in pl.graveyard for pl in state.players)]
            if captured and not survivors:
                _log_target_fizzle(state, card_def, None)
                return
            for inst in survivors:
                owner = next(pl for pl in state.players if inst in pl.graveyard)
                owner.graveyard.remove(inst)  # exiled, untracked
                state.log_event(
                    "zone_move", card=inst.name, from_zone="graveyard", to_zone="exile_untracked", reason="rooftop_percher",
                )

        push_to_stack(state, percher_def, _resolve, reserves_hand_card=False, is_spell=False,  # ETB effect -- not a spell
                      targets=tuple(("graveyard_card", inst) for inst in captured))

    begin_choose_up_to_graveyard(state, lambda c: True, 2, _on_targets, graveyard=combined)


def pinnacle_kill_ship_etb(state):
    """ETB: deals 10 damage to up to one target creature (either side).
    Target chosen as the ability goes on the stack; fizzles if the target
    has left the battlefield by resolution."""
    kill_ship_def = registry.CARD_DEFS["Pinnacle Kill-Ship"]

    def _on_target(state, target_descriptor):
        captured = capture_any_target(state, target_descriptor)  # ("creature", perm) or None

        def _resolve(state, card_def):
            if captured is None:
                return  # "up to one": no target chosen
            if not target_still_legal(state, captured):
                _log_target_fizzle(state, card_def, (captured[1].card_def.name, captured[1].slot))
                return
            captured[1].damage_marked += 10
            check_state_based_actions(state)

        push_to_stack(state, kill_ship_def, _resolve, reserves_hand_card=False, is_spell=False,
                      targets=() if captured is None else (captured,))

    begin_choose_any_target(
        state,
        lambda p: p.card_type == CardType.CREATURE and can_be_targeted(state, p, state.active_idx),
        _on_target,
        allow_players=False,
        optional=True,  # "up to one target"
    )


def _pinnacle_kill_ship_station_legal(state, permanent):
    """Station: legal only with another untapped creature to tap."""
    return any(p is not permanent and not p.tapped and p.card_type == CardType.CREATURE for p in state.battlefield)


def _pinnacle_kill_ship_station_resolve(state, permanent):
    """Tap another creature you control; put that many charge counters on
    this permanent. At 7+ charge counters, this becomes a 7/7 flying
    creature (Permanent.type_override, flipped once, permanently)."""
    def _on_chosen(state, choice):
        if choice is None:
            return
        name, slot = choice
        tapped_creature = next(p for p in state.battlefield if p.card_def.name == name and p.slot == slot)
        tap_for_cost(state, tapped_creature, reason="pinnacle_kill_ship_station")  # a COST, paid now
        gained = permanent_power(state, tapped_creature)  # snapshotted at cost-payment time

        def _effect(state):
            permanent.counters["charge"] = permanent.counters.get("charge", 0) + gained
            animate = registry.EFFECT_REGISTRY[EffectId.PINNACLE_KILL_SHIP]["animate"]
            if permanent.counters["charge"] >= animate["threshold"]:
                permanent.type_override = CardType.CREATURE
                state.log_event("animated", permanent=(permanent.card_def.name, permanent.slot), new_type="CREATURE")

        push_ability_to_stack(state, permanent.card_def, _effect)

    begin_choose_permanent(
        state, lambda p: p is not permanent and not p.tapped and p.card_type == CardType.CREATURE, _on_chosen,
    )


# Boulderbranch Golem's Prototype {3}{G} 3/3 mode: a distinct CardDef (own
# stats + effect_id), reached only via the "prototype" registry spec, never
# registered in COLORLESS_CARD_CATALOG itself.
BOULDERBRANCH_GOLEM_PROTOTYPE_CARD_DEF = CardDef(
    "Boulderbranch Golem", CardType.CREATURE, {"generic": 3, "G": 1}, EffectId.BOULDERBRANCH_GOLEM_PROTOTYPE,
    power=3, toughness=3,
)


def cast_boulderbranch_prototype(state, card_def):
    """Cast Boulderbranch Golem for its Prototype cost as the 3/3. `card_def`
    is a different object from the CardDef in hand (same display name), so
    the hand card is found by name, not identity."""
    hand_card = next((c for c in state.hand if c.name == "Boulderbranch Golem"), None)
    if hand_card is not None:
        state.hand.remove(hand_card)  # tolerant for a direct call outside normal cast
    enters_battlefield(state, card_def)


def activate_barrels_of_blasting_jelly_burn(state, permanent):
    """{5}, {T}, Sacrifice: deals 5 damage to target creature. Target chosen
    at activation; fizzles if the target is gone by resolution."""
    sacrifice_to_graveyard(state, permanent)
    idx = state.active_idx

    def _on_target(state, target):
        captured = capture_any_target(state, target)

        def _resolve(state, card_def):
            if captured is None or not target_still_legal(state, captured):
                where = (captured[1].card_def.name, captured[1].slot) if captured is not None else None
                _log_target_fizzle(state, card_def, where)
                return
            captured[1].damage_marked += 5
            check_state_based_actions(state)

        push_to_stack(
            state, permanent.card_def, _resolve, reserves_hand_card=False, is_spell=False,
            targets=() if captured is None else (captured,),
        )

    begin_choose_any_target(
        state, lambda p: p.card_type == CardType.CREATURE and can_be_targeted(state, p, idx),
        _on_target, allow_players=False,
    )


COLORLESS_EFFECT_REGISTRY = {
    EffectId.TRON_LAND: {
        "mana": ("tron",),
    },
    EffectId.TOCASIA_DIG_SITE: {
        "mana": ("fixed", "C"),
        "activated_abilities": {
            "surveil": {
                "cost_key": "surveil_ability_cost",
                "resolve": lambda state, permanent: activate_tocasia_dig_site_surveil(state, permanent),
            },
        },
        "pending_kinds": {"surveil"},
    },
    EffectId.CONDUIT_PYLONS: {
        "mana": ("fixed", "C"),
        "etb_trigger": lambda state, permanent: surveil(state, 1),
        "filter_mana": {"colors": set(COLORS)},
        "pending_kinds": {"surveil"},
    },
    EffectId.EXPEDITION_MAP: {
        "cast": {"resolve": lambda state, card_def: cast_permanent_from_hand(state, card_def)},
        "activated_abilities": {
            "activate": {
                "cost_key": "ability_cost",
                "resolve": lambda state, permanent: activate_expedition_map(state, permanent),
            },
        },
        "pending_kinds": {"search_fetch"},
    },
    EffectId.BONDERS_ORNAMENT: {
        "mana": ("flexible", set(COLORS)),
        "cast": {"resolve": lambda state, card_def: cast_permanent_from_hand(state, card_def)},
        "activated_abilities": {
            "draw": {
                "cost_key": "draw_ability_cost",
                "resolve": lambda state, permanent: activate_bonders_ornament_draw(state, permanent),
            },
        },
    },
    EffectId.CANDY_TRAIL: {
        "etb_trigger": lambda state, permanent: scry(state, 2),
        "cast": {"resolve": lambda state, card_def: cast_permanent_from_hand(state, card_def)},
        "activated_abilities": {
            "sac": {
                "cost_key": "sac_ability_cost",
                "resolve": lambda state, permanent: activate_candy_trail_sac(state, permanent),
            },
        },
        "pending_kinds": {"scry"},
    },
    EffectId.BLOOD_TOKEN: {
        "activated_abilities": {
            "sac": {
                "cost_key": "sac_ability_cost",
                "resolve": lambda state, permanent: activate_blood_sac(state, permanent),
            },
        },
    },
    EffectId.CLUE_TOKEN: {
        # {2}, Sacrifice: Draw a card. Never a decklist card -- only created by Investigate.
        "activated_abilities": {
            "sac": {
                "cost_key": "sac_ability_cost",
                "resolve": lambda state, permanent: activate_clue_sac(state, permanent),
            },
        },
    },
    EffectId.TREASURE_TOKEN: {
        # {T}, Sacrifice: Add one mana of any color (ceases to exist; not graveyard-bound).
        "mana": ("flexible", set(COLORS)),
        "on_tap": lambda state, permanent: _treasure_on_tap(state, permanent),
    },
    EffectId.BARRELS_OF_BLASTING_JELLY: {
        "cast": {"resolve": lambda state, card_def: cast_permanent_from_hand(state, card_def)},
        "filter_mana": {"colors": set(COLORS)},
        "activated_abilities": {
            "blast": {
                "cost_key": "blast_ability_cost",
                "extra_legal": lambda state: has_creature_target(state),
                "resolve": lambda state, permanent: activate_barrels_of_blasting_jelly_burn(state, permanent),
            },
        },
        "pending_kinds": {"choose_any_target"},
    },
    EffectId.RELIC_OF_PROGENITUS: {
        "cast": {"resolve": lambda state, card_def: cast_permanent_from_hand(state, card_def)},
        "activated_abilities": {
            "draw": {
                "cost_key": "draw_ability_cost",
                "resolve": lambda state, permanent: activate_relic_of_progenitus_draw(state, permanent),
            },
            "exile": {
                "cost_key": "graveyard_exile_ability_cost",
                "resolve": lambda state, permanent: activate_relic_of_progenitus_exile(state, permanent),
            },
        },
        "pending_kinds": {"choose_target_player", "choose_graveyard_card"},
    },
    EffectId.LOTUS_PETAL: {
        "cast": {"resolve": lambda state, card_def: cast_permanent_from_hand(state, card_def)},
        "mana": ("flexible", set(COLORS)),
        "on_tap": lambda state, permanent: _lotus_petal_on_tap(state, permanent),
    },
    # Kept explicit (rather than omitted) because several self-checks reassign
    # registry.EFFECT_REGISTRY[EffectId.FILLER] via direct bracket indexing.
    EffectId.FILLER: {},
    EffectId.ROOFTOP_PERCHER: {
        "cast": {"resolve": lambda state, card_def: cast_permanent_from_hand(state, card_def)},
        "etb_trigger": lambda state, permanent: rooftop_percher_etb(state, permanent),
        "etb_targets": True,  # up-to-2 graveyard targets chosen at promotion
        "pending_kinds": {"choose_graveyard_card"},
        "keywords": {"flying"},
    },
    EffectId.BOULDERBRANCH_GOLEM: {
        "cast": {"resolve": lambda state, card_def: cast_permanent_from_hand(state, card_def)},
        "etb_trigger": lambda state, permanent: gain_life(state, 6),  # gain life equal to its power (6 in this mode)
        # Prototype {3}{G} -- a second, cheaper cast option producing the 3/3 mode.
        "prototype": {
            "card_def": BOULDERBRANCH_GOLEM_PROTOTYPE_CARD_DEF,
            "cost": {"generic": 3, "G": 1},
            "resolve": lambda state, card_def: cast_boulderbranch_prototype(state, card_def),
        },
    },
    EffectId.BOULDERBRANCH_GOLEM_PROTOTYPE: {
        # Reached only via the Prototype cast action. Gain life equal to its power (3 in this mode).
        "etb_trigger": lambda state, permanent: gain_life(state, 3),
    },
    EffectId.MAELSTROM_COLOSSUS: {
        "cast": {"resolve": lambda state, card_def: cast_maelstrom_colossus(state, card_def)},
        "pending_kinds": {"may_cast"},  # Cascade's may-cast of the hit; self-only decision
    },
    EffectId.PINNACLE_KILL_SHIP: {
        "cast": {"resolve": lambda state, card_def: cast_permanent_from_hand(state, card_def)},
        "etb_trigger": lambda state, permanent: pinnacle_kill_ship_etb(state),
        "etb_targets": True,  # "up to one target creature" chosen at promotion
        "activated_abilities": {
            "station": {
                "speed": Speed.SORCERY,  # "Station only as a sorcery"
                "legal": lambda state, permanent: _pinnacle_kill_ship_station_legal(state, permanent),
                "resolve": lambda state, permanent: _pinnacle_kill_ship_station_resolve(state, permanent),
            },
        },
        # choose_permanent = Station's tap-a-creature cost; choose_any_target = the ETB's target.
        "pending_kinds": {"choose_permanent", "choose_any_target"},
        "animate": {"counter": "charge", "threshold": 7, "power": 7, "toughness": 7, "keywords": {"flying"}},
    },

    # --- boggles deck ---
    EffectId.ASH_BARRENS: {
        "mana": ("fixed", "C"),
        "forestcycle": {
            "cost_key": "cycling_cost",
            "resolve": lambda state, card_def: cycle_ash_barrens(state, card_def),
        },
        "pending_kinds": {"search_fetch"},
    },

    # --- G7: grixis_affinity ---
    EffectId.MYR_ENFORCER: {
        "cast": {"resolve": lambda state, card_def: cast_permanent_from_hand(state, card_def)},
        "cost_reduction": affinity_reduction,  # affinity for artifacts
    },

    # --- G8: artifact value engines ---
    EffectId.ICHOR_WELLSPRING: {
        "cast": {"resolve": lambda state, card_def: cast_permanent_from_hand(state, card_def)},
        "etb_trigger": lambda state, permanent: ichor_wellspring_draw(state),  # ETB draw
        "ltb_trigger": lambda state, permanent: ichor_wellspring_draw(state),  # "put into a graveyard from the battlefield" draw
    },
    EffectId.CHROMATIC_STAR: {
        "cast": {"resolve": lambda state, card_def: cast_permanent_from_hand(state, card_def)},
        "activated_abilities": {
            "sac": {"cost_key": "sac_ability_cost", "resolve": lambda state, permanent: chromatic_star_mana(state, permanent)},
        },
        "ltb_trigger": lambda state, permanent: ichor_wellspring_draw(state),  # dies -> draw
        "pending_kinds": {"choose_mana_color"},  # "add one mana of any color" color choice
    },
    EffectId.NIHIL_SPELLBOMB: {
        "cast": {"resolve": lambda state, card_def: cast_permanent_from_hand(state, card_def)},
        "activated_abilities": {
            "sac": {"cost_key": "sac_ability_cost", "resolve": lambda state, permanent: nihil_spellbomb_sac(state, permanent)},
        },
        "ltb_trigger": lambda state, permanent: nihil_spellbomb_dies(state, permanent),  # dies -> may pay {B} draw
        "pending_kinds": {"choose_target_player", "pay_unless"},
    },
    EffectId.LEMBAS: {
        "cast": {"resolve": lambda state, card_def: cast_permanent_from_hand(state, card_def)},
        "etb_trigger": lambda state, permanent: lembas_etb(state),  # scry 1, then draw
        "activated_abilities": {
            "sac": {"cost_key": "sac_ability_cost", "resolve": lambda state, permanent: lembas_sac(state, permanent)},
        },
        "ltb_trigger": lambda state, permanent: lembas_dies(state, permanent),  # dies -> shuffle into library
        "pending_kinds": {"scry"},
    },
    EffectId.MAP_TOKEN: {
        "activated_abilities": {
            "explore": {
                "cost_key": "ability_cost",
                "speed": Speed.SORCERY,  # "Activate only as a sorcery"
                "resolve": lambda state, permanent: activate_map_sac(state, permanent),
            },
        },
        "pending_kinds": {"choose_permanent", "surveil"},  # choose the exploring creature; surveil = explore's keep/bin
    },

    # --- jund_wildfire ---
    EffectId.TWISTED_LANDSCAPE: {
        "mana": ("fixed", "C"),
        "activated_abilities": {
            "fetch": {
                "cost_key": "fetch_ability_cost",  # {} -- only {T} + sacrifice, both paid inside the resolve
                "resolve": lambda state, permanent: activate_twisted_landscape_fetch(state, permanent),
            },
        },
        "cycle": {  # Cycling {B}{R}{G}: discard, draw a card -- the generic "Cycle {name}" action
            "cost_key": "cycling_cost",
            "resolve": lambda state, card_def: cycle_twisted_landscape(state, card_def),
        },
        "pending_kinds": {"search_fetch"},
    },
}
