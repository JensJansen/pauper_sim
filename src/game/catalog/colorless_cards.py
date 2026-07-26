"""Colorless-identity card catalog: lands/artifacts with no colored mana
symbol in their cost and no fixed-color mana output (an "any color"
ability grants no specific color, matching real Magic's own
color-identity rule -- e.g. Bonder's Ornament, Tron lands). Every card's
cost/type/oracle-text below is a direct Scryfall pull, except creature
power/toughness, which is a design choice, not Scryfall data. Rooftop
Percher/Boulderbranch Golem/Maelstrom Colossus/Pinnacle Kill-Ship (Tron
filler) verified colorless via Scryfall, not guessed -- Bramble Wurm and
Breath Weapon, the other two Tron filler names, turned out to be green
and red respectively and file there instead.

Rooftop Percher/Boulderbranch Golem/Maelstrom Colossus/Pinnacle Kill-Ship
are cast at their real default cost/stats, with whichever clause is a
real ETB life-gain effect wired for real. Each one's OWN complicating
clause is a deliberate, documented drop rather than a guess:
- Rooftop Percher: Changeling (every creature type) is a no-op -- no
  tribal-synergy card exists anywhere in this catalog to care. Its own
  "exile up to two target cards from graveyards" is also dropped: in this
  solitaire sim the only legal targets are this player's own graveyard,
  and nothing rewards emptying it, so a rational cast always chooses zero
  targets anyway -- unlike Relic of Progenitus' own repeatable graveyard-
  exile ability (this file, activate_relic_of_progenitus_exile), which IS
  implemented despite the identical "no real upside" reasoning, simply
  because it was worth building as its own always-available action
  rather than an ETB-only one-off choice bundled into a bigger creature
  spell.
- Boulderbranch Golem: Prototype ({3}{G} for a 3/3 instead) is IMPLEMENTED --
  a second CardDef (BOULDERBRANCH_GOLEM_PROTOTYPE_CARD_DEF) with its own
  stats/effect_id for the same display name, offered as a "Cast X (prototype)"
  action that reuses the Omen cast machinery. Its ETB "gain life equal to its
  power" is baked per mode (6 normal / 3 prototype), since nothing in this
  pool changes the Golem's power before the ETB resolves.
- Maelstrom Colossus: Cascade is implemented for real (cast_maelstrom_
  colossus below) -- it invokes an ARBITRARY other catalog card's own
  "cast" resolve, which normally assumes the card is already in
  state.hand (discard_from_hand_to_graveyard's own not-in-hand
  RuntimeError), by temporarily inserting the hit card into state.hand
  first: CardDefs are shared/interned per name (registry.CARD_DEFS holds
  one per distinct name, not per physical copy), so every existing
  resolve's own hand-removal correctly finds and removes it, with no
  parallel "cast from library" implementation needed for any card. Its
  own extra_legal is still checked first (Cascade only waives the MANA
  cost, not other costs -- same distinction Plot's own docstring already
  draws), and if the hit's own resolve opens a further pending resolution
  of its own (Ancient Stirrings' take-one-or-decline), Maelstrom
  Colossus's own battlefield entry is chained onto that resolution's
  on_complete rather than happening immediately -- see the function's own
  docstring.
- Pinnacle Kill-Ship: Station and its ETB are both implemented for real
  (unlike the three drops above) -- see this file's own
  activate_pinnacle_kill_ship_station/pinnacle_kill_ship_etb. Station taps
  ANOTHER creature you control (a real per-creature choice, since unlike
  Saruli Caretaker's identical-cost tap the CHOSEN creature's own power
  matters) and puts that many "charge" counters (Permanent.counters,
  state.py) on this permanent; at 7+, Permanent.type_override (state.py)
  flips to CardType.CREATURE and stats._animate_spec starts reporting its
  real 7/7 flying in place of the Artifact's own 0/0 -- both read
  generically by every "is this currently a creature" check (combat
  eligibility, state-based death, Saruli/Quirion/Dread Return's own
  "another creature" checks) with no Kill-Ship-specific code at any of
  those call sites. The ETB reuses begin_choose_opponent_permanent
  (resolution.py), which already auto-no-ops with zero legal targets -- so
  it's a correct, harmless no-op in every current (1-player, no opponent)
  Tron config, and becomes a real removal spell the instant a 2-player Tron
  config exists to reach it.

"mana" shapes: ("tron",) -- Tron's controls-all-three-doubling rule;
("fixed", symbol) -- always produces that one symbol; ("flexible",
{symbols}) -- caller chooses one of several. "filter_mana": {"colors":
{...}} marks Barrels of Blasting Jelly's and Conduit Pylons' colored-pip
filter ability (as opposed to Conduit Pylons' plain {T}: Add {C}, which
IS a "fixed" mana source below) -- offered by mana.tap_cost_options for
any of the 5 colors, same as a flexible source (its own {1} activation
cost is tracked separately, see mana.execute_tap_cost_option)."""

from .. import registry
from ..cards import CardDef, CardType, EffectId
from ..effects.casting import _log_target_fizzle, capture_any_target, cast_permanent_from_hand, enters_battlefield, target_still_legal
from ..effects.stack import push_ability_to_stack, push_to_stack
from ..effects.shared import discard_from_hand_to_graveyard, find_to_hand
from ..effects.state_based import check_state_based_actions
from ..effects.stats import can_be_targeted, permanent_power
from ..effects.tokens import activate_blood_sac
from ..effects.win_check import gain_life
from ..mana import COLORS
from ..resolution import (
    begin_choose_any_target, begin_choose_graveyard_card, begin_choose_opponent_permanent, begin_choose_permanent,
    begin_choose_target_player, begin_search_fetch, scry, surveil,
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
        mana_ability_cost={"generic": 1},
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
}


def activate_tocasia_dig_site_surveil(state, permanent):
    """{3}, T: Surveil 1 (shares the tap cost with its plain {T}: Add {C}).
    Faithful timing: the tap is a COST (paid now); the surveil is the
    effect, so it goes on the stack and resolves after a priority window."""
    permanent.tapped = True
    state.log_event("tap", permanent=(permanent.card_def.name, permanent.slot), reason="activate")
    push_ability_to_stack(state, permanent.card_def, lambda st: surveil(st, 1))


def activate_expedition_map(state, permanent):
    """{2}, T, Sacrifice: search library for a land -- the model's choice.
    Caller has already paid the {1} cost. Faithful timing: the {T} and the
    sacrifice are COSTS (paid now); the search is the effect, so it goes on
    the stack and resolves (opening the search) after a priority window."""
    state.battlefield.remove(permanent)
    state.graveyard.append(permanent.card_def)
    state.log_event(
        "zone_move", permanent=(permanent.card_def.name, permanent.slot), from_zone="battlefield",
        to_zone="graveyard", reason="sacrifice",
    )
    push_ability_to_stack(state, permanent.card_def, lambda st: begin_search_fetch(st, lambda c: c.card_type == CardType.LAND, find_to_hand))


def activate_bonders_ornament_draw(state, permanent):
    """{4}, T: draw a card (shares the tap cost with its plain mana ability).
    Faithful timing: the tap is a COST (paid now); the draw is the effect,
    so it goes on the stack and resolves after a priority window."""
    permanent.tapped = True
    state.log_event("tap", permanent=(permanent.card_def.name, permanent.slot), reason="activate")
    push_ability_to_stack(state, permanent.card_def, lambda st: st.draw(1))


def activate_candy_trail_sac(state, permanent):
    """{2}, T, Sacrifice: gain 3 life and draw a card. Faithful timing: the
    {T} and the sacrifice are COSTS (paid now); gaining 3 and drawing are
    the effect, so they go on the stack and resolve after a priority
    window."""
    state.battlefield.remove(permanent)
    state.graveyard.append(permanent.card_def)
    state.log_event(
        "zone_move", permanent=(permanent.card_def.name, permanent.slot), from_zone="battlefield",
        to_zone="graveyard", reason="sacrifice",
    )

    def _effect(state):
        gain_life(state, 3)
        state.draw(1)

    push_ability_to_stack(state, permanent.card_def, _effect)


def activate_relic_of_progenitus_draw(state, permanent):
    """{1}, Exile this artifact: exile all graveyards, draw a card. "All
    graveyards" loops every PlayerState (not just the active one) --
    correct even in a hypothetical future 2-player config that plays this
    card, and costs nothing extra in the 1-player configs that actually
    do today. Exile itself untracked, same convention as this same
    artifact's own repeatable {T} ability below -- just clears each
    graveyard to nothing rather than tracking a real exile pile.

    Faithful timing: exiling this artifact is a COST (paid now); exiling
    all graveyards and drawing are the effect, so they go on the stack and
    resolve after a priority window. "Exile all graveyards" is measured at
    resolution (the effect reads state.players' graveyards then), matching
    real Magic."""
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
    """{T}: Target player exiles a card from their graveyard -- a REAL
    target-player choice (resolution.begin_choose_target_player), not a
    self-target assumed on the ability's behalf: "yourself" is always one
    legal choice (true in real Magic even alone), "the opponent" becomes
    a second one the moment a real one exists, and the model picks
    explicitly either way via drl_env's own fixed "Target: yourself"/
    "Target: opponent" actions. Whichever player is targeted, then chooses
    (simplified to the ACTIVATING player's own choice, same "no observable
    difference in this solitaire sim" precedent begin_choose_graveyard_
    card's own docstring already documents) one of THAT player's
    graveyard cards to exile -- untracked, same convention as this same
    artifact's other, one-shot exile-self ability above. Repeatable (no
    mana cost, {T} only), independent of that other ability. An empty
    target graveyard -> nothing to choose, the same empty-options safety
    net begin_choose_graveyard_card already provides.

    Faithful timing + cross-player choice:
    the {T} is a COST (paid now); the target player is chosen as the ability
    is put on the stack (targets lock at activation); only the EFFECT waits
    on the stack. When it resolves, the TARGETED player -- not the activator
    -- chooses which of their own graveyard cards to exile: active_idx is
    flipped to them for that forced choice and restored afterward. This
    replaces the old "simplified to the activating player's choice"
    approximation now that a real cross-player flip exists to do it right."""
    permanent.tapped = True  # {T} -- a COST, paid now on activation
    state.log_event("tap", permanent=(permanent.card_def.name, permanent.slot), reason="activate")

    def _on_player_chosen(state, idx):
        def _effect(state):
            # resolve_top_of_stack set active_idx to this entry's controller
            # (the activator). Flip to the targeted player so THEY pick the
            # card; the priority loop leaves active_idx alone while a pending
            # they own is open (game.turn._run_priority_round_gen), and
            # _on_card_chosen restores it once the choice is made.
            activator = state.active_idx
            target_player = state.players[idx]

            def _on_card_chosen(state, name):
                state.active_idx = activator  # the targeted player's forced choice is done
                if name is None:
                    return
                found = next(c for c in target_player.graveyard if c.name == name)
                target_player.graveyard.remove(found)  # exiled, not removed-to-nowhere; exile is untracked
                state.log_event(
                    "zone_move", card=found.name, from_zone="graveyard", to_zone="exile_untracked",
                    target_player_idx=idx,
                )

            state.active_idx = idx
            begin_choose_graveyard_card(state, lambda c: True, _on_card_chosen, graveyard=target_player.graveyard)

        push_ability_to_stack(state, permanent.card_def, _effect)

    begin_choose_target_player(state, _on_player_chosen)


def _lotus_petal_on_tap(state, permanent):
    """{T}, Sacrifice: add one mana of any color -- consumed, not just
    tapped, unlike every other mana source in this engine."""
    state.battlefield.remove(permanent)
    state.graveyard.append(permanent.card_def)
    state.log_event(
        "zone_move", permanent=(permanent.card_def.name, permanent.slot), from_zone="battlefield",
        to_zone="graveyard", reason="sacrifice",
    )


def _lotus_petal_on_tap_undo(state, permanent):
    state.graveyard.remove(permanent.card_def)
    state.battlefield.append(permanent)


def _basic_land(card_def):
    return card_def.extra.get("basic", False)


def cycle_ash_barrens(state, card_def):
    """Basic landcycling {1}: discard this card from hand, search library
    for a basic land, put it into hand, shuffle. No draw-a-card rider (a
    plain Cycling ability would have one; Basic Landcycling doesn't --
    verified via Scryfall, not guessed), and the found land goes to hand,
    not the battlefield -- this is exactly Generous Ent's own forestcycle
    shape (game.catalog.green_cards), just with a real model choice of
    WHICH basic land (this decklist runs both Forest and Plains, unlike
    Generous Ent's single fixed "Forest" target)."""
    discard_from_hand_to_graveyard(state, card_def)
    begin_search_fetch(state, _basic_land, find_to_hand)


def _mana_value(cast_cost):
    """Total converted cost of a cast_cost dict ({"generic": 2, "G": 1} ->
    3) -- every symbol, colored or generic, contributes 1 per point. None
    (a land) is never passed in here; {} (Lotus Petal) correctly gives 0."""
    return sum(cast_cost.values())


def cast_maelstrom_colossus(state, card_def):
    """Cascade: exile cards from the top of the library until a nonland
    card with mana value LESS than this card's own (8) is exiled; cast it
    without paying its mana cost, then put the rest on the bottom in a
    random order (real text: "the exiled cards," i.e. every OTHER card
    seen along the way, not the hit itself).

    "Cast it for free" reuses the hit card's own registry "cast" spec
    completely unchanged -- CardDefs are shared/interned per name
    (registry.CARD_DEFS holds one per distinct name, not per physical
    copy), so temporarily inserting it into state.hand first makes every
    existing resolve's own hand-removal (discard_from_hand_to_graveyard's
    universal convention) correctly find and remove it, with no parallel
    "cast from library" implementation needed for any card, current or
    future. Skipped (no cast, straight to the bottom with everything
    else) if the hit has no "cast" spec at all (Generous Ent's own
    "never hard-cast" precedent -- forestcycle-only cards have none) or
    its own extra_legal fails (Cascade only waives the MANA cost, not any
    other cost or precondition a card's normal cast still gates on --
    same distinction Plot's own docstring already draws for Highway
    Robbery; Crop Rotation's "sacrifice a non-Tron land" extra_legal is
    this catalog's own live example).

    If the hit's own resolve opens a further pending resolution of its
    own (Ancient Stirrings' take-one-or-decline is this catalog's only
    such "cast" spec), Maelstrom Colossus's own battlefield entry can't
    happen yet -- chained onto that resolution's own on_complete instead
    of running immediately, so it lands only once the cascaded card's
    entire effect, decisions included, is actually done (matching real
    Magic: the Cascade trigger resolves completely before Colossus, still
    the next thing up on the stack, ever does)."""
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

    # Every exiled card goes to the bottom EXCEPT the hit, and only if
    # it's actually being cast -- a whiff, a hit with no "cast" spec, or
    # one whose own extra_legal fails all leave it right here among the
    # rest (real text: "Put THE EXILED CARDS on the bottom" -- a card
    # that was never cast is still one of them).
    remaining = [c for c in exiled if not (can_cast and c is hit)]
    state.rng.shuffle(remaining)
    state.library.extend(remaining)

    def _enter_colossus(state):
        enters_battlefield(state, card_def)

    if can_cast:
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


def pinnacle_kill_ship_etb(state):
    """ETB trigger: "deals 10 damage to up to one target creature." Faithful:
    the trigger's EFFECT goes on the stack -- up to one target creature on
    EITHER battlefield (hexproof/shroud aware) is chosen as the trigger is
    put on the stack (begin_choose_any_target, optional=True for "up to
    one"), the 10 damage waits on the stack, then hits that exact creature at
    resolution, or fizzles if it has left the battlefield by then (608.2b),
    or does nothing if no target was chosen. Target selection at ETB time is
    when the trigger goes on the stack (603.3d) -- there's no state-based
    step between entering and this that could change the legal targets."""
    kill_ship_def = registry.CARD_DEFS["Pinnacle Kill-Ship"]

    def _on_target(state, target_descriptor):
        captured = capture_any_target(state, target_descriptor)  # ("creature", perm) or None (declined / no target)

        def _resolve(state, card_def):
            if captured is None:
                return  # "up to one": no target chosen -- the ability resolves doing nothing
            if not target_still_legal(state, captured):
                _log_target_fizzle(state, card_def, (captured[1].card_def.name, captured[1].slot))
                return
            captured[1].damage_marked += 10
            check_state_based_actions(state)

        push_to_stack(state, kill_ship_def, _resolve, reserves_hand_card=False)  # the ETB effect, on the stack

    begin_choose_any_target(
        state,
        lambda p: p.card_type == CardType.CREATURE and can_be_targeted(state, p, state.active_idx),
        _on_target,
        allow_players=False,
        optional=True,  # "up to one target"
    )


def _pinnacle_kill_ship_station_legal(state, permanent):
    """Station -- tap ANOTHER creature you control (no mana cost of its
    own, per its own real text). Unlike Saruli Caretaker's identical-shape
    tap cost, WHICH creature gets tapped genuinely matters here (its power
    sets how many charge counters this gets), so this can't reuse Saruli's
    own "any untapped creature, arbitrarily" simplification -- see
    _pinnacle_kill_ship_station_resolve's own real per-creature choice."""
    return any(p is not permanent and not p.tapped and p.card_type == CardType.CREATURE for p in state.battlefield)


def _pinnacle_kill_ship_station_resolve(state, permanent):
    """Real per-creature choice (unlike Saruli Caretaker's fungible-by-name
    auto-pick): the tapped creature's own power sets how many charge
    counters land here, so which one gets tapped is a real decision, same
    "genuine choice, not simplified away" treatment Quirion Ranger's own
    untap target already gets. Charge counters alone (Permanent.counters)
    are the whole mechanism -- stats._animate_spec reads them straight off
    this permanent to decide power/toughness/flying/current CardType
    (Permanent.card_type) once 7+ are reached; type_override is flipped
    here explicitly (once, the instant the threshold is first crossed --
    nothing in this card pool ever removes a charge counter, so there's no
    "un-animate" path to handle).

    Faithful timing: tapping another creature is a
    COST, paid now on activation; putting the charge counters (and the
    animate check) is the effect, so it goes on the stack and resolves after
    a priority window."""
    def _on_chosen(state, choice):
        if choice is None:
            return
        name, slot = choice
        tapped_creature = next(p for p in state.battlefield if p.card_def.name == name and p.slot == slot)
        tapped_creature.tapped = True  # tap another creature -- a COST, paid now
        state.log_event(
            "tap", permanent=(name, slot), reason="pinnacle_kill_ship_station",
            source=(permanent.card_def.name, permanent.slot),
        )
        # Charge counters gained = the tapped creature's power. Snapshotted
        # here at cost-payment time: nothing in this pool changes a
        # creature's power at instant speed, so this equals its power at
        # resolution, and stands as last-known information if that creature
        # leaves the battlefield (bounced/killed in response) before the
        # effect resolves.
        gained = permanent_power(state, tapped_creature)

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


# Boulderbranch Golem's Prototype {3}{G} 3/3 mode. A DISTINCT CardDef (same
# display name, own smaller stats + own effect_id) reached only via the
# "prototype" registry spec + drl_env's "Cast X (prototype)" action -- never
# registered in COLORLESS_CARD_CATALOG itself, same ontology as green_cards'
# SAGU_WILDLING_CREATURE_CARD_DEF. Its own effect_id carries the ETB
# "gain life equal to its power" as a fixed 3 (this mode's power) -- see the
# BOULDERBRANCH_GOLEM_PROTOTYPE registry entry below. Real reminder text:
# "You may cast this spell with different mana cost, color, and size. It keeps
# its abilities and types." (The artifact/color change is a no-op here -- the
# engine models both modes as plain CREATUREs, and no card cares about the
# green color a Prototype cast grants.)
BOULDERBRANCH_GOLEM_PROTOTYPE_CARD_DEF = CardDef(
    "Boulderbranch Golem", CardType.CREATURE, {"generic": 3, "G": 1}, EffectId.BOULDERBRANCH_GOLEM_PROTOTYPE,
    power=3, toughness=3,
)


def cast_boulderbranch_prototype(state, card_def):
    """Cast Boulderbranch Golem for its Prototype cost as the 3/3. The mana
    ({3}{G}) is already paid by drl_env._omen_cast_execute; `card_def` is
    BOULDERBRANCH_GOLEM_PROTOTYPE_CARD_DEF, a different object from the
    "Boulderbranch Golem" actually sitting in hand (the normal {7} CardDef,
    same display name) -- so the hand card is found by NAME, not identity,
    same as Sagu Wildling's own Omen creature half."""
    hand_card = next(c for c in state.hand if c.name == "Boulderbranch Golem")
    state.hand.remove(hand_card)
    enters_battlefield(state, card_def)


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
    EffectId.BARRELS_OF_BLASTING_JELLY: {
        "cast": {"resolve": lambda state, card_def: cast_permanent_from_hand(state, card_def)},
        "filter_mana": {"colors": set(COLORS)},
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
        "on_tap_undo": lambda state, permanent: _lotus_petal_on_tap_undo(state, permanent),
    },
    # EffectId.FILLER's single canonical registry entry -- every reader
    # consults it via EFFECT_REGISTRY.get(effect_id, {}), which already
    # defaults a missing key to {} the same way -- kept explicit here
    # (rather than omitted entirely) only because several
    # game/effects/*.py self-checks temporarily reassign
    # registry.EFFECT_REGISTRY[EffectId.FILLER] via direct bracket
    # indexing, which requires the key to already exist.
    EffectId.FILLER: {},
    EffectId.ROOFTOP_PERCHER: {
        "cast": {"resolve": lambda state, card_def: cast_permanent_from_hand(state, card_def)},
        "etb_trigger": lambda state, permanent: gain_life(state, 3),
        "keywords": {"flying"},
    },
    EffectId.BOULDERBRANCH_GOLEM: {
        "cast": {"resolve": lambda state, card_def: cast_permanent_from_hand(state, card_def)},
        # Real text is "gain life equal to its power." Power is fixed at 6 in
        # THIS (normal {7} 6/5) mode -- nothing in this pool changes it before
        # the ETB resolves -- so the amount is baked as 6 rather than read off
        # the permanent; the Prototype 3/3 mode bakes 3 the same way (its own
        # BOULDERBRANCH_GOLEM_PROTOTYPE effect_id below).
        "etb_trigger": lambda state, permanent: gain_life(state, 6),
        # Prototype {3}{G} -- 3/3: a second, cheaper cast option producing the
        # smaller creature (BOULDERBRANCH_GOLEM_PROTOTYPE_CARD_DEF). Reuses the
        # Omen action machinery (drl_env.build_action_table's own "prototype"
        # block + _omen_cast_legal/_omen_cast_execute).
        "prototype": {
            "card_def": BOULDERBRANCH_GOLEM_PROTOTYPE_CARD_DEF,
            "cost": {"generic": 3, "G": 1},
            "resolve": lambda state, card_def: cast_boulderbranch_prototype(state, card_def),
        },
    },
    EffectId.BOULDERBRANCH_GOLEM_PROTOTYPE: {
        # Reached only via the Prototype cast action (never a decklist/CARD_DEFS
        # entry by this effect_id). "Gain life equal to its power" = 3 in this
        # 3/3 mode, baked the same way the normal mode bakes 6 above.
        "etb_trigger": lambda state, permanent: gain_life(state, 3),
    },
    EffectId.MAELSTROM_COLOSSUS: {
        "cast": {"resolve": lambda state, card_def: cast_maelstrom_colossus(state, card_def)},
    },
    EffectId.PINNACLE_KILL_SHIP: {
        "cast": {"resolve": lambda state, card_def: cast_permanent_from_hand(state, card_def)},
        "etb_trigger": lambda state, permanent: pinnacle_kill_ship_etb(state),
        "activated_abilities": {
            "station": {
                "speed": Speed.SORCERY,  # real text: "Station only as a sorcery"
                "legal": lambda state, permanent: _pinnacle_kill_ship_station_legal(state, permanent),
                "resolve": lambda state, permanent: _pinnacle_kill_ship_station_resolve(state, permanent),
            },
        },
        # choose_permanent = Station (tap a creature YOU control -- a cost);
        # choose_any_target = the ETB "up to one target creature" (either side).
        "pending_kinds": {"choose_permanent", "choose_any_target"},
        # Read by stats._animate_spec (power/toughness/keywords once
        # animated) and _pinnacle_kill_ship_station_resolve (the threshold
        # that flips Permanent.type_override to CREATURE). Real text: 7+
        # charge counters -> an artifact creature with flying.
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
}


if __name__ == "__main__":
    # ponytail self-check: no pytest in this project, mirrors the
    # assert-based demo convention -- run via
    # `python -m game.catalog.colorless_cards` from src/.
    from ..effects.stack import resolve_top_of_stack
    from ..effects.triggers import promote_triggers_to_stack
    from ..resolution import choose_graveyard_card_options, execute_choose_graveyard_card_option
    from ..state import GameState, Permanent

    def _resolve_etb(state):
        """Test helper: drive a just-queued ETB (faithful timing queues it
        rather than firing inline) onto the stack and resolve it, the way
        game.turn's own priority round does in real play."""
        promote_triggers_to_stack(state)
        resolve_top_of_stack(state)

    # Basic landcycling {1}: discard this card from hand, search for a
    # basic land -- a real model choice between Forest and Plains (unlike
    # Generous Ent's own forestcycle, which always searches "Forest"
    # specifically), put into hand, shuffle. No draw-a-card rider (unlike
    # a plain Cycling ability) -- verified via Scryfall, not guessed.
    state = GameState(on_the_play=True)
    ash_barrens = CardDef("Ash Barrens", CardType.LAND, None, EffectId.ASH_BARRENS, cycling_cost={"generic": 1})
    state.hand = [ash_barrens]
    state.library = [
        CardDef("Forest", CardType.LAND, None, EffectId.FOREST, basic=True),
        CardDef("Plains", CardType.LAND, None, EffectId.PLAINS, basic=True),
        CardDef("Ash Barrens", CardType.LAND, None, EffectId.ASH_BARRENS, cycling_cost={"generic": 1}),  # not basic -- ineligible
    ]
    cycle_ash_barrens(state, ash_barrens)
    assert state.pending_resolution["kind"] == "search_fetch"
    from ..resolution import search_fetch_options, execute_search_fetch_option
    assert search_fetch_options(state) == ["Forest", "Plains"]  # the 2nd Ash Barrens is correctly excluded
    execute_search_fetch_option(state, "Plains")
    assert state.pending_resolution is None
    assert [c.name for c in state.hand] == ["Plains"]
    assert sorted(c.name for c in state.graveyard) == ["Ash Barrens"]  # discarded itself, not the fetched land
    assert sorted(c.name for c in state.library) == ["Ash Barrens", "Forest"]  # shuffled; the unchosen basic stays

    print("colorless_cards.py Ash Barrens self-check: OK")

    # Candy Trail's sac ability: gain 3 life AND draw a card -- both
    # halves of its real text, not just the draw.
    state = GameState(on_the_play=True)
    candy_trail = Permanent(CardDef(
        "Candy Trail", CardType.ARTIFACT, {"generic": 1}, EffectId.CANDY_TRAIL, sac_ability_cost={"generic": 2},
    ))
    state.battlefield = [candy_trail]
    state.library = [CardDef("Forest", CardType.LAND, None, EffectId.FOREST, basic=True)]
    activate_candy_trail_sac(state, candy_trail)
    assert state.battlefield == []  # sacrificed -- a cost, paid immediately on activation
    # gain 3 + draw are the ability's EFFECT -- now on the stack (faithful
    # timing), not applied the instant the cost was paid.
    assert len(state.stack) == 1 and state.life_total == 20 and state.hand == []
    resolve_top_of_stack(state)
    assert state.life_total == 23  # STARTING_LIFE (20) + 3, once the effect resolves
    assert [c.name for c in state.hand] == ["Forest"]
    print("colorless_cards.py Candy Trail self-check: OK")

    # Relic of Progenitus: two independent abilities now. The repeatable
    # {T} one is a REAL target-player choice, not a self-target assumed
    # on its behalf -- "yourself" chosen explicitly still exiles a real
    # card (Test A), and targeting a genuine opponent reaches into THEIR
    # graveyard instead (Test B), never state.graveyard (the active
    # player's own). The one-shot {1}+exile-self draw ability now also
    # clears every graveyard first (real text: "exile ALL graveyards"),
    # not just draw (Test C).
    from ..resolution import execute_choose_target_player_option

    # Test A: explicit self-target.
    state = GameState(on_the_play=True)
    relic = Permanent(CardDef(
        "Relic of Progenitus", CardType.ARTIFACT, {"generic": 1}, EffectId.RELIC_OF_PROGENITUS,
        draw_ability_cost={"generic": 1}, graveyard_exile_ability_cost={},
    ))
    state.battlefield = [relic]
    state.graveyard = [
        CardDef("Bramble Wurm", CardType.CREATURE, {"generic": 6, "G": 1}, EffectId.BRAMBLE_WURM),
        CardDef("Breath Weapon", CardType.INSTANT, {"generic": 2, "R": 1}, EffectId.BREATH_WEAPON),
    ]
    activate_relic_of_progenitus_exile(state, relic)
    assert relic.tapped  # {T} -- a cost, paid immediately on activation
    assert state.pending_resolution["kind"] == "choose_target_player"  # target chosen at activation
    execute_choose_target_player_option(state, 0)  # explicitly target yourself
    # The effect (the targeted player exiles a graveyard card) is now on the
    # stack -- it opens the graveyard-card choice only once it resolves.
    assert state.pending_resolution is None and len(state.stack) == 1
    resolve_top_of_stack(state)
    assert state.pending_resolution["kind"] == "choose_graveyard_card"
    assert choose_graveyard_card_options(state) == ["Bramble Wurm", "Breath Weapon"]
    execute_choose_graveyard_card_option(state, "Bramble Wurm")
    assert state.pending_resolution is None
    assert [c.name for c in state.graveyard] == ["Breath Weapon"]  # only the chosen one removed

    state.library = [CardDef("Forest", CardType.LAND, None, EffectId.FOREST, basic=True)]
    activate_relic_of_progenitus_draw(state, relic)
    assert state.battlefield == []  # exile-self -- a cost, paid immediately
    # "exile ALL graveyards" + draw are the effect -- on the stack now.
    assert len(state.stack) == 1 and [c.name for c in state.graveyard] == ["Breath Weapon"]
    resolve_top_of_stack(state)
    assert state.graveyard == []  # the untouched "Breath Weapon" is gone too, once resolved
    assert [c.name for c in state.hand] == ["Forest"]
    print("colorless_cards.py Relic of Progenitus (self-target + exile-all) self-check: OK")

    # Test B: a real opponent exists -- targeting them reaches into THEIR
    # graveyard, never the active player's own.
    from ..state import PlayerState

    state2 = GameState(on_the_play=True, players=[PlayerState(True), PlayerState(False)])
    relic2 = Permanent(CardDef(
        "Relic of Progenitus", CardType.ARTIFACT, {"generic": 1}, EffectId.RELIC_OF_PROGENITUS,
        draw_ability_cost={"generic": 1}, graveyard_exile_ability_cost={},
    ))
    state2.players[0].battlefield = [relic2]
    state2.players[0].graveyard = [CardDef("Mine", CardType.CREATURE, None, EffectId.FILLER)]
    state2.players[1].graveyard = [CardDef("Theirs", CardType.CREATURE, None, EffectId.FILLER)]
    activate_relic_of_progenitus_exile(state2, relic2)
    execute_choose_target_player_option(state2, 1)  # target the opponent (locked at activation)
    resolve_top_of_stack(state2)  # effect resolves -> the TARGETED player picks the card
    assert state2.active_idx == 1  # active_idx flipped to the targeted player for their choice
    assert choose_graveyard_card_options(state2) == ["Theirs"]  # THEIR graveyard, not "Mine"
    execute_choose_graveyard_card_option(state2, "Theirs")
    assert state2.active_idx == 0  # restored to the activator once the choice is made
    assert state2.players[1].graveyard == []
    assert [c.name for c in state2.players[0].graveyard] == ["Mine"]  # own graveyard untouched
    print("colorless_cards.py Relic of Progenitus (real opponent target) self-check: OK")

    # Rooftop Percher / Boulderbranch Golem: real ETB gain-life triggers,
    # the one piece of genuinely new logic these two add now that they're
    # no longer inert EffectId.FILLER entries (cast_permanent_from_hand
    # and enters_battlefield's own etb_trigger dispatch are already
    # self-checked elsewhere -- casting.py, this just confirms these two
    # cards' own specific gain amounts are wired to the right effect_id).
    state = GameState(on_the_play=True)
    percher = CardDef("Rooftop Percher", CardType.CREATURE, {"generic": 5}, EffectId.ROOFTOP_PERCHER, power=3, toughness=3)
    state.hand = [percher]
    cast_permanent_from_hand(state, percher)
    _resolve_etb(state)  # ETB gain-3 is queued now (faithful timing), resolved off the stack
    assert state.life_total == 23  # STARTING_LIFE (20) + 3

    golem = CardDef("Boulderbranch Golem", CardType.CREATURE, {"generic": 7}, EffectId.BOULDERBRANCH_GOLEM, power=6, toughness=5)
    state.hand = [golem]
    cast_permanent_from_hand(state, golem)
    _resolve_etb(state)
    assert state.life_total == 29  # +6 on top of the 23 above

    print("colorless_cards.py Tron filler creature self-check: OK")

    # Boulderbranch Golem Prototype {3}{G} -- 3/3: a second cast option (own
    # cheaper cost, smaller stats, own ETB "gain life = its power" = 3). Cast
    # the prototype from hand; the {7} hand card is consumed, a 3/3 enters,
    # and life goes up by 3 (not the normal mode's 6).
    state = GameState(on_the_play=True)
    golem_hand = CardDef("Boulderbranch Golem", CardType.CREATURE, {"generic": 7}, EffectId.BOULDERBRANCH_GOLEM, power=6, toughness=5)
    state.hand = [golem_hand]
    cast_boulderbranch_prototype(state, BOULDERBRANCH_GOLEM_PROTOTYPE_CARD_DEF)
    assert state.hand == []  # the {7} hand card was consumed by the prototype cast
    _resolve_etb(state)
    assert state.life_total == 23  # 20 + 3 (the 3/3's power), not +6
    proto = next(p for p in state.battlefield if p.card_def.name == "Boulderbranch Golem")
    assert proto.card_def is BOULDERBRANCH_GOLEM_PROTOTYPE_CARD_DEF
    from ..effects.stats import permanent_toughness as _proto_toughness
    assert permanent_power(state, proto) == 3 and _proto_toughness(state, proto) == 3

    print("colorless_cards.py Boulderbranch Golem Prototype self-check: OK")

    # Maelstrom Colossus's real Cascade -- four cases: a hit that's cast
    # for free, a whiff (nothing eligible), a hit whose own extra_legal
    # fails (skipped, same as real Magic), and a hit whose own resolve
    # opens a further pending resolution of its own (Colossus's entry
    # deferred until that completes).
    from .. import resolution
    from ..state import GameState as _GS
    _filler_backup = registry.EFFECT_REGISTRY[EffectId.FILLER]
    try:
        # Hit: a plain permanent (cast_permanent_from_hand), mana value 2
        # (< 8) -- exiled cards after it go to the bottom (this engine's
        # library is already a shuffled abstraction, so "bottom" just
        # means "back in state.library"), Colossus itself enters last.
        registry.EFFECT_REGISTRY[EffectId.FILLER] = {"cast": {"resolve": lambda state, card_def: cast_permanent_from_hand(state, card_def)}}
        state = _GS(on_the_play=True)
        colossus = CardDef("Maelstrom Colossus", CardType.CREATURE, {"generic": 8}, EffectId.MAELSTROM_COLOSSUS, power=7, toughness=7)
        a_land = CardDef("A Land", CardType.LAND, None, EffectId.FOREST, basic=True)
        hit = CardDef("Free Hit", CardType.ARTIFACT, {"generic": 2}, EffectId.FILLER)
        after = CardDef("Never Seen", CardType.ARTIFACT, {"generic": 1}, EffectId.FILLER)
        state.hand = [colossus]
        state.library = [a_land, hit, after]
        cast_maelstrom_colossus(state, colossus)
        assert state.graveyard == [colossus]
        assert sorted(p.card_def.name for p in state.battlefield) == ["Free Hit", "Maelstrom Colossus"]
        # Real Cascade only exiles cards UP TO AND INCLUDING the hit --
        # "Never Seen" was still sitting below it, never revealed/exiled
        # at all, so it stays exactly where it was; only what was
        # actually exiled alongside the hit ("A Land") gets shuffled and
        # returned, appended after it.
        assert [c.name for c in state.library] == ["Never Seen", "A Land"]
        print("colorless_cards.py Maelstrom Colossus Cascade (hit) self-check: OK")

        # Whiff: nothing eligible (everything's either a land or costs 8+)
        # -- Colossus still enters, nothing else does.
        state2 = _GS(on_the_play=True)
        colossus2 = CardDef("Maelstrom Colossus", CardType.CREATURE, {"generic": 8}, EffectId.MAELSTROM_COLOSSUS, power=7, toughness=7)
        too_expensive = CardDef("Too Expensive", CardType.ARTIFACT, {"generic": 8}, EffectId.FILLER)
        state2.hand = [colossus2]
        state2.library = [a_land, too_expensive]
        cast_maelstrom_colossus(state2, colossus2)
        assert [p.card_def.name for p in state2.battlefield] == ["Maelstrom Colossus"]
        assert sorted(c.name for c in state2.library) == ["A Land", "Too Expensive"]

        # extra_legal fails: Cascade only waives the MANA cost, not other
        # preconditions -- skipped, straight to the bottom with the rest,
        # same as a genuine whiff.
        registry.EFFECT_REGISTRY[EffectId.FILLER] = {
            "cast": {"resolve": lambda state, card_def: cast_permanent_from_hand(state, card_def), "extra_legal": lambda state: False},
        }
        state3 = _GS(on_the_play=True)
        colossus3 = CardDef("Maelstrom Colossus", CardType.CREATURE, {"generic": 8}, EffectId.MAELSTROM_COLOSSUS, power=7, toughness=7)
        ineligible_hit = CardDef("Ineligible Hit", CardType.ARTIFACT, {"generic": 2}, EffectId.FILLER)
        state3.hand = [colossus3]
        state3.library = [ineligible_hit]
        cast_maelstrom_colossus(state3, colossus3)
        assert [p.card_def.name for p in state3.battlefield] == ["Maelstrom Colossus"]
        assert [c.name for c in state3.library] == ["Ineligible Hit"]  # never cast, just shuffled back in

        # The hit's own resolve opens a further pending resolution (an
        # Ancient-Stirrings-like take-it-or-decline) -- Colossus must NOT
        # enter until that resolution is actually completed.
        entered_before_decision = []

        def _opens_pending(state, card_def):
            state.hand.remove(card_def)

            def _on_complete(state, taken):
                if taken:
                    state.graveyard.append(card_def)
                entered_before_decision.append(any(p.card_def.name == "Maelstrom Colossus" for p in state.battlefield))

            resolution.begin_resolution(state, "fake_choice", _on_complete)

        registry.EFFECT_REGISTRY[EffectId.FILLER] = {"cast": {"resolve": _opens_pending}}
        state4 = _GS(on_the_play=True)
        colossus4 = CardDef("Maelstrom Colossus", CardType.CREATURE, {"generic": 8}, EffectId.MAELSTROM_COLOSSUS, power=7, toughness=7)
        decision_hit = CardDef("Decision Hit", CardType.ARTIFACT, {"generic": 2}, EffectId.FILLER)
        state4.hand = [colossus4]
        state4.library = [decision_hit]
        cast_maelstrom_colossus(state4, colossus4)
        assert state4.pending_resolution["kind"] == "fake_choice"  # Colossus genuinely hasn't entered yet
        assert not any(p.card_def.name == "Maelstrom Colossus" for p in state4.battlefield)
        resolution.complete_resolution(state4, True)
        assert entered_before_decision == [False]  # Colossus hadn't entered DURING the decision's own on_complete either
        assert sorted(p.card_def.name for p in state4.battlefield) == ["Maelstrom Colossus"]  # decision_hit chose graveyard, not battlefield, in this fake resolve
        assert state4.pending_resolution is None

        print("colorless_cards.py Maelstrom Colossus Cascade (whiff/extra_legal/chained-resolution) self-check: OK")
    finally:
        registry.EFFECT_REGISTRY[EffectId.FILLER] = _filler_backup

    # Pinnacle Kill-Ship: Station is a REAL per-creature choice (unlike
    # Saruli Caretaker's fungible-by-name tap) -- charge counters equal to
    # whichever creature actually gets tapped, and the animate threshold
    # (Permanent.type_override/stats._animate_spec) they feed once 7+ are
    # reached.
    from ..effects.stats import has_keyword as _has_keyword
    from ..effects.stats import permanent_toughness as _permanent_toughness

    state = GameState(on_the_play=True)
    kill_ship = Permanent(CardDef("Pinnacle Kill-Ship", CardType.ARTIFACT, {"generic": 7}, EffectId.PINNACLE_KILL_SHIP))
    weak = Permanent(CardDef("Weak Tapper", CardType.CREATURE, None, EffectId.FILLER, power=3, toughness=3))
    strong = Permanent(CardDef("Strong Tapper", CardType.CREATURE, None, EffectId.FILLER, power=5, toughness=5))
    state.battlefield = [kill_ship, weak, strong]

    assert _pinnacle_kill_ship_station_legal(state, kill_ship) is True
    assert kill_ship.card_type == CardType.ARTIFACT  # not yet animated
    assert permanent_power(state, kill_ship) == 0 and _permanent_toughness(state, kill_ship) == 0

    _pinnacle_kill_ship_station_resolve(state, kill_ship)
    assert resolution.choose_permanent_options(state) == [("Strong Tapper", 1), ("Weak Tapper", 1)]  # Kill-Ship itself never offered -- "another creature"
    resolution.execute_choose_permanent_option(state, "Strong Tapper", 1)
    assert strong.tapped is True  # tapped -- a cost, paid immediately on activation
    # Placing the charge counters is the EFFECT -- on the stack now (faithful
    # timing), applied only on resolution.
    assert len(state.stack) == 1 and kill_ship.counters.get("charge", 0) == 0
    resolve_top_of_stack(state)
    assert kill_ship.counters["charge"] == 5  # the TAPPED creature's own power
    assert kill_ship.card_type == CardType.ARTIFACT  # still below the 7-counter threshold

    _pinnacle_kill_ship_station_resolve(state, kill_ship)
    resolution.execute_choose_permanent_option(state, "Weak Tapper", 1)
    resolve_top_of_stack(state)
    assert kill_ship.counters["charge"] == 8  # 5 + 3, now >= 7
    assert kill_ship.card_type == CardType.CREATURE  # animated
    assert permanent_power(state, kill_ship) == 7 and _permanent_toughness(state, kill_ship) == 7
    assert _has_keyword(state, kill_ship, "flying") is True
    assert not _pinnacle_kill_ship_station_legal(state, kill_ship)  # both other creatures already tapped

    print("colorless_cards.py Pinnacle Kill-Ship Station self-check: OK")

    # ETB "up to one target creature" -- now on the stack: 10 damage to a
    # chosen creature (either side, hexproof-aware) at resolution, or fizzle
    # if it's gone, or nothing if declined.
    import contextlib
    import io

    from .. import resolution as _res
    from ..effects.stack import resolve_top_of_stack
    from ..state import PlayerState

    # (a) target an opponent creature -> 10 damage at resolution (lethal via SBA)
    state = GameState(on_the_play=True, players=[PlayerState(True), PlayerState(False)])
    victim = Permanent(CardDef("Victim", CardType.CREATURE, None, EffectId.FILLER, power=1, toughness=3))
    victim.slot = 1
    state.players[1].battlefield = [victim]
    pinnacle_kill_ship_etb(state)
    assert state.pending_resolution["kind"] == "choose_any_target" and state.pending_resolution["optional"]
    _res.execute_choose_any_target_creature(state, 1, "Victim", 1)
    assert len(state.stack) == 1  # the ETB effect waits on the stack
    resolve_top_of_stack(state)
    assert victim not in state.players[1].battlefield  # 10 vs 3 -> lethal via SBA

    # (b) up-to-one: decline -> the ability resolves doing nothing
    state = GameState(on_the_play=True, players=[PlayerState(True), PlayerState(False)])
    bystander = Permanent(CardDef("Bystander", CardType.CREATURE, None, EffectId.FILLER, power=1, toughness=3))
    bystander.slot = 1
    state.players[1].battlefield = [bystander]
    pinnacle_kill_ship_etb(state)
    _res.execute_choose_any_target_decline(state)
    resolve_top_of_stack(state)
    assert bystander in state.players[1].battlefield and bystander.damage_marked == 0

    # (c) fizzle: chosen creature gone before the ETB resolves
    state = GameState(on_the_play=True, players=[PlayerState(True), PlayerState(False)])
    doomed = Permanent(CardDef("Doomed", CardType.CREATURE, None, EffectId.FILLER, power=1, toughness=3))
    doomed.slot = 1
    state.players[1].battlefield = [doomed]
    pinnacle_kill_ship_etb(state)
    _res.execute_choose_any_target_creature(state, 1, "Doomed", 1)
    state.players[1].battlefield = []  # exiled/bounced before resolution
    _log = io.StringIO()
    with contextlib.redirect_stdout(_log):
        resolve_top_of_stack(state)
    assert "fizzle" in _log.getvalue().lower()

    print("colorless_cards.py Pinnacle Kill-Ship ETB (up-to-one, on stack, fizzle) self-check: OK")
