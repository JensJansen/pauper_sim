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
  "exile up to two target cards from graveyards" IS implemented
  (rooftop_percher_etb below) -- a real ETB trigger, up to two targets
  chosen from EITHER player's graveyard as it goes on the stack, exiling
  every still-present target at resolution (608.2c partial fizzle); the
  +3 life is unconditional even if every target leaves the graveyard
  first (an owner-authorized simplification, see that function's own
  docstring).
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
  (resolution), which already auto-no-ops with zero legal targets -- so
  it's a correct, harmless no-op in every current (1-player, no opponent)
  Tron config, and becomes a real removal spell the instant a 2-player Tron
  config exists to reach it.

"mana" shapes: ("tron",) -- Tron's controls-all-three-doubling rule;
("fixed", symbol) -- always produces that one symbol; ("flexible",
{symbols}) -- caller chooses one of several. "filter_mana": {"colors":
{...}} marks Barrels of Blasting Jelly's and Conduit Pylons' colored-pip
filter ability (as opposed to Conduit Pylons' plain {T}: Add {C}, which
IS a "fixed" mana source below) -- a two-step pay-then-choose-color
conversion (drl_env._filter_mana_legal/_execute: spend one floating pip as
the {1} activation cost immediately, then choose which of the 5 colors to
add via the shared choose_color mana_subdecision stage), never a nested
cost payment."""

from .. import registry
from ..cards import CardDef, CardType, EffectId
from ..effects.casting import (
    _log_target_fizzle, capture_any_target, cast_permanent_from_hand, enters_battlefield, has_creature_target,
    target_still_legal,
)
from ..effects.stack import push_ability_to_stack, push_to_stack
from ..effects.shared import (
    affinity_reduction, discard_from_hand_to_graveyard, find_and_remove_by_name, find_to_hand, fire_sacrifice_triggers,
)
from ..effects.state_based import check_state_based_actions, sacrifice_to_graveyard
from ..effects.stats import can_be_targeted, permanent_power
from ..effects.tokens import activate_blood_sac, activate_clue_sac, activate_map_sac
from ..effects.win_check import gain_life
from ..mana import COLORS
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
    # {T}: Add {C}. {T}, Sacrifice this land: fetch a basic Swamp/Mountain/
    # Forest onto the battlefield tapped. Cycling {B}{R}{G} (plain cycling =
    # discard, draw a card) is wired in G2 alongside the cycling
    # generalization -- cycling_cost carried here now so it's ready.
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
    """{1}, {T}, Sacrifice this artifact: Add one mana of ANY COLOR. The {1}
    and untapped precondition come from drl_env's cost_key wiring; sacrificing
    (a mana ability, resolved immediately, no stack) fires its dies-trigger
    (draw). The activating player then chooses which of the five colors to
    add (begin_choose_mana_color) -- real color fixing, not a fixed {C}."""
    sacrifice_to_graveyard(state, permanent)  # queues the dies-trigger (draw)

    def _add_color(state, color):
        state.mana_pool[color] = state.mana_pool.get(color, 0) + 1
        state.log_event("mana_tap", permanent=(permanent.card_def.name, permanent.slot), mode="chromatic_star", produced=[color])

    begin_choose_mana_color(state, _add_color)


def nihil_spellbomb_sac(state, permanent):
    """{T}, Sacrifice: Exile target player's graveyard. The sacrifice fires
    its dies-trigger (may pay {B} draw); the exile (of the chosen player's
    whole graveyard) is the effect, on the stack."""
    sacrifice_to_graveyard(state, permanent)  # queues the dies-trigger

    def _on_player(state, idx):
        def _effect(st):
            st.players[idx].graveyard.clear()
            st.log_event("graveyard_exiled", player_idx=idx)
        push_ability_to_stack(state, permanent.card_def, _effect)

    begin_choose_target_player(state, _on_player)


def nihil_spellbomb_dies(state, permanent):
    """"When put into a graveyard from the battlefield, you may pay {B}. If you
    do, draw a card." The controller -- Nihil Spellbomb's owner, NOT
    necessarily whoever is active when this trigger resolves (a later
    priority window can flip state.active_idx before an LTB trigger actually
    resolves) -- may pay {B} for the draw. `owner_idx` is stashed on the
    permanent's own flags at removal time (state_based._destroy_creature /
    sacrifice_to_graveyard), the same real controller a later resolution
    can no longer derive from the battlefield (the permanent's already gone)."""
    payer = permanent.flags["owner_idx"]

    def _on_result(state, paid):
        if paid:
            state.draw(1)

    begin_pay_unless(state, payer, {"B": 1}, _on_result)


def lembas_etb(state):
    """When Lembas enters: scry 1, THEN draw a card (the draw chained onto the
    scry's completion so it happens after)."""
    begin_scry_surveil(state, "scry", 1, on_complete=lambda s: s.draw(1))


def lembas_sac(state, permanent):
    """{2}, {T}, Sacrifice: You gain 3 life. Sacrificing fires its dies-trigger
    (shuffle itself back into the library) -- so Lembas recurs."""
    sacrifice_to_graveyard(state, permanent)  # queues the dies-trigger (shuffle into library)
    push_ability_to_stack(state, permanent.card_def, lambda st: gain_life(st, 3))


def lembas_dies(state, permanent):
    """"When put into a graveyard from the battlefield, its owner shuffles it
    into their library." A 400.7 linked ability -- it must reference the EXACT
    card that died, not just any same-named graveyard card. The death that
    queued this trigger stashed the fresh graveyard instance on `permanent`'s
    own flags (state_based._destroy_creature / sacrifice_to_graveyard); it's
    in the owner's graveyard now (this trigger resolves for that owner ->
    state.graveyard is theirs) unless something else already moved it (e.g. a
    Dread Return reanimated it first) -- in which case there's nothing to do."""
    inst = permanent.flags.get("graveyard_instance")
    if inst is not None and inst in state.graveyard:
        state.graveyard.remove(inst)
        state.library.append(inst.card_def)  # library is DEFERRED (CardDefs)
        state.rng.shuffle(state.library)
        state.log_event("zone_move", card=inst.name, from_zone="graveyard", to_zone="library", reason="lembas_shuffle")


def _twisted_landscape_fetch_eligible(card_def):
    return card_def.extra.get("basic", False) and card_def.name in ("Swamp", "Mountain", "Forest")


def activate_twisted_landscape_fetch(state, permanent):
    """{T}, Sacrifice this land: search library for a basic Swamp, Mountain,
    or Forest, put it onto the battlefield TAPPED, then shuffle. The {T}
    (untapped requirement enforced generically by drl_env._activate_legal)
    and the sacrifice are COSTS, paid now; the search is the effect, so it
    goes on the stack and resolves after a priority window -- same shape as
    Expedition Map, except the fetch lands on the battlefield tapped
    (force_tapped) rather than in hand."""
    state.battlefield.remove(permanent)
    state.move_card(permanent.card_def, state.graveyard)
    state.log_event(
        "zone_move", permanent=(permanent.card_def.name, permanent.slot), from_zone="battlefield",
        to_zone="graveyard", reason="sacrifice",
    )
    fire_sacrifice_triggers(state, state.active_idx, permanent.card_def)  # Gixian Infiltrator

    def _effect(st):
        def _on_fetch(st, land_name):
            found = find_and_remove_by_name(st, land_name)
            st.rng.shuffle(st.library)
            if found:
                enters_battlefield(st, found, force_tapped=True)

        begin_search_fetch(st, _twisted_landscape_fetch_eligible, _on_fetch)  # mandatory (Expedition Map precedent); fizzles to None if no basic left

    push_ability_to_stack(state, permanent.card_def, _effect)


def cycle_twisted_landscape(state, card_def):
    """Cycling {B}{R}{G}: discard this card from hand, draw a card (plain
    Cycling -- has the draw rider, unlike Ash Barrens' basic Landcycling).
    Reuses the generic "cycle" hand-zone action (drl_env)."""
    discard_from_hand_to_graveyard(state, card_def)
    state.draw(1)


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
    state.move_card(permanent.card_def, state.graveyard)
    state.log_event(
        "zone_move", permanent=(permanent.card_def.name, permanent.slot), from_zone="battlefield",
        to_zone="graveyard", reason="sacrifice",
    )
    fire_sacrifice_triggers(state, state.active_idx, permanent.card_def)  # Gixian Infiltrator
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
    state.move_card(permanent.card_def, state.graveyard)
    state.log_event(
        "zone_move", permanent=(permanent.card_def.name, permanent.slot), from_zone="battlefield",
        to_zone="graveyard", reason="sacrifice",
    )
    fire_sacrifice_triggers(state, state.active_idx, permanent.card_def)  # Gixian Infiltrator

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
    "Target: opponent" actions. Whichever player is targeted then chooses one
    of THEIR OWN graveyard cards to exile -- made by THAT player themselves via
    the active_idx flip (see "Faithful timing + cross-player choice" below).
    Untracked exile, same convention as this same artifact's other, one-shot
    exile-self ability above. Repeatable (no
    mana cost, {T} only), independent of that other ability. An empty
    target graveyard -> nothing to choose, the same empty-options safety
    net begin_choose_graveyard_card already provides.

    Faithful timing + cross-player choice:
    the {T} is a COST (paid now); the target player is chosen as the ability
    is put on the stack (targets lock at activation); only the EFFECT waits
    on the stack. When it resolves, the TARGETED player -- not the activator
    -- chooses which of their own graveyard cards to exile: active_idx is
    flipped to them for that forced choice and restored afterward."""
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
    """{T}, Sacrifice: add one mana of any color -- consumed, not just
    tapped, unlike every other mana source in this engine. A real card (not
    a token), so sacrifice_to_graveyard's non-token branch sends it to its
    owner's graveyard (to_zone="graveyard") -- same destination as before."""
    sacrifice_to_graveyard(state, permanent)  # queues the dies-trigger (Gixian Infiltrator); Lotus Petal has no ltb_trigger of its own


def _treasure_on_tap(state, permanent):
    """Treasure's "{T}, Sacrifice this artifact: Add one mana of any color" --
    consumed on tap like Lotus Petal, but a TOKEN, so it ceases to exist
    (never a graveyard trip). The flexible-color output + untapped-{T}
    precondition come from the registry "mana" spec, same as Lotus Petal."""
    sacrifice_to_graveyard(state, permanent)  # ceases to exist; queues the dies-trigger (Gixian Infiltrator)


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
    """ETB: "exile up to two target cards from graveyards. You gain 3 life."
    Faithful targeting (project_targeted_triggered_abilities): the up-to-2
    graveyard targets are chosen NOW, as the ability goes on the stack
    (target-at-promotion), captured by object identity, and the exile fizzles
    per-target at resolution -- exiling every still-present target, doing nothing
    only if ALL chosen targets have left the graveyard (608.2c). Targets can come
    from EITHER player's graveyard ("from graveyardS").

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
            gain_life(state, 3)  # unconditional -- happens even if the exile fully fizzles (owner directive)
            survivors = [inst for inst in captured if any(inst in pl.graveyard for pl in state.players)]
            if captured and not survivors:
                _log_target_fizzle(state, card_def, None)  # every exile target left the graveyard -> exile does nothing
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

        push_to_stack(state, kill_ship_def, _resolve, reserves_hand_card=False, is_spell=False,  # ETB (triggered ability) -- not a spell
                      targets=() if captured is None else (captured,))

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
    hand_card = next((c for c in state.hand if c.name == "Boulderbranch Golem"), None)
    if hand_card is not None:
        state.hand.remove(hand_card)  # normally already removed at cast (_omen_cast_execute); tolerant for a direct call
    enters_battlefield(state, card_def)


def activate_barrels_of_blasting_jelly_burn(state, permanent):
    """{5}, {T}, Sacrifice this artifact: it deals 5 damage to target
    creature. The {5} mana + {T}-of-self are paid by the generic cost_key
    wiring before this ever runs (drl_env._activate_execute); the
    sacrifice is a COST, paid now, same shape as Expedition Map/Candy
    Trail's own {T}, Sacrifice abilities in this same file. The target
    creature is chosen right after -- before the ability goes on the
    stack, matching every other targeted ability here -- and the 5 damage
    waits on the stack for it, fizzling per 608.2b if that creature is
    gone by resolution. A MANDATORY "target creature" (not "any target"),
    so the registry's own extra_legal gates activation on a legal creature
    target actually existing (601.2c/602.2b)."""
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
        # "{2}, Sacrifice this artifact: Draw a card." Never a decklist card --
        # only ever created by Investigate (Toxin Analysis). Same cost_key
        # wiring as Blood/Food.
        "activated_abilities": {
            "sac": {
                "cost_key": "sac_ability_cost",
                "resolve": lambda state, permanent: activate_clue_sac(state, permanent),
            },
        },
    },
    EffectId.TREASURE_TOKEN: {
        # "{T}, Sacrifice: Add one mana of any color" -- consumed flexible
        # source, exactly Lotus Petal's shape (mana + on_tap sacrifice), only
        # this is a token so it ceases rather than going to the graveyard.
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
        "etb_trigger": lambda state, permanent: rooftop_percher_etb(state, permanent),
        "etb_targets": True,  # up-to-2 graveyard targets chosen at promotion, exile fizzles at resolution
        "pending_kinds": {"choose_graveyard_card"},  # the up-to-2 target picks (from EITHER graveyard)
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
        "pending_kinds": {"may_cast"},  # Cascade's may-cast of the hit -- self-only (the CASCADING player's own
        # choice); the HIT's own pending_kinds need no separate declaration here either, since the hit is always
        # one of THIS SAME decklist's own cards, already covered by derive_pending_kinds scanning it directly.
    },
    EffectId.PINNACLE_KILL_SHIP: {
        "cast": {"resolve": lambda state, card_def: cast_permanent_from_hand(state, card_def)},
        "etb_trigger": lambda state, permanent: pinnacle_kill_ship_etb(state),
        "etb_targets": True,  # "up to one target creature" -- chosen at promotion, fizzles at resolution (project_targeted_triggered_abilities)
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

    # --- G7: grixis_affinity ---
    EffectId.MYR_ENFORCER: {
        "cast": {"resolve": lambda state, card_def: cast_permanent_from_hand(state, card_def)},
        "cost_reduction": affinity_reduction,  # Affinity for artifacts (4/4 vanilla artifact creature)
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
