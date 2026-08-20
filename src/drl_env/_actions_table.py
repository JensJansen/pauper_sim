"""build_action_table (assembles the flat fixed action table from a decklist
+ game.EFFECT_REGISTRY) and legal_action_mask (+ its sweep-scoped coverage
check) -- the two entry points every other _actions_* category module feeds
into. Kept together, in their own file, since the table-builder touches
every category and the mask sweep resets every category's own sweep-scoped
cache; splitting either further apart from its own dependencies would just
turn this file's import block into the split.

Categories, in table order (see drl_env._actions_cast/_actions_cast_altzone/
_combat/_resolution/_mana for the category boundaries themselves):
  A. Play land: <name>            -- one per distinct land name
  A2. Tap <name> [for <color>]    -- mana abilities: one no-color
     row per fixed/Tron/count source, one row per producible color for a
     flexible/granted source. Legal in ANY priority window, even
     mid-resolution of anything else (605.1a/605.3b -- no pending-resolution
     gate at all), resolves immediately (never the stack) -- masked by
     game.mana_ability_options. "Tap <name>" (Saruli Caretaker): same
     gate-free category, but its extra cost (tap ANOTHER creature -- a cost
     choice, 602.5g, not a target) opens a mana_subdecision (a SEPARATE
     state field from pending_resolution, so this can still open mid-
     resolution of anything else without clobbering it) instead of
     resolving in one shot -- see state.mana_subdecision's own docstring
     and _mana_extra_choose_legal's own docstring for why.
     "Filter <name>, paying <input_color>" (Conduit Pylons/Barrels): pays
     the {1} activation cost immediately as a flat fixed-table row, then
     opens the shared choose_color mana_subdecision stage (see "Tap
     <name>"/Saruli's own note above and _filter_mana_execute's own
     docstring) to pick the output color via the "Produce <color>"
     buttons -- no nested pay_cost either, same reasoning.
  B. Cast <name>                  -- one per card with a registry "cast" entry
  C. Activate <name> (<ability>)  -- one per registered activated ability
  D. Forestcycle <name>           -- one per registry "forestcycle" entry
  E. Pass
  F. Choose: <name>               -- shared across every pending-resolution
     kind that picks a plain card name (search_fetch, ancient_stirrings,
     discard, and scry/surveil's ordering phase), dispatched by
     pending_resolution["kind"]. Paying a cost never appears here -- once a
     cost is announced, the only legal actions are the A2 mana abilities
     (601.2f, producing the mana) and "Spend <color> from pool" (below,
     spending it).
     NOT sacrifice (see category K) -- a battlefield permanent is never
     just a name, unlike a hand/library/graveyard card.
  H. Keep / Dispose (scry/surveil)
  I. Decline (Ancient Stirrings)
  K. Choose target: <name> (slot k) -- exact-(name, slot)-addressed, the
     "choose_permanent" resolution's own actions (Aura enchant-targets,
     Crop Rotation's sacrifice cost, land bounce, and now every generic
     sacrifice cost -- begin_sacrifice/Highway Robbery's own sacrifice
     trigger, both chained through this same primitive) -- NOT category F:
     two same-named permanents stop being interchangeable the instant an
     Aura attaches to only one of them (or one has a counter, is tapped,
     ...), and cast_aura's cast-time-target/resolve-time-fizzle contract
     (same as knowing exactly which physical permanent was sacrificed)
     depends on knowing exactly which physical permanent was chosen.

spy_combo deck additions: B also covers Winding Way's modal cast (2
actions, one per mode), Land Grant's free alt-cost, Dread Return's
Flashback (cast from the graveyard), and Nyxborn Hydra's own
"x_cast_modes" (one action per (mode, X) pair -- its normal creature cast
and Bestow, each its own cost distinct from card_def.cast_cost, see
_x_cast_legal/_x_cast_execute/_x_precast_choice_execute in drl_env.
_actions_cast); C also covers non-mana activated abilities (Quirion
Ranger, Pinnacle Kill-Ship's own Station); F/H also cover select_to_hand's
own Keep/Bottom pair and its ordering phase (Lead the Stampede) and an
optional search's Decline (Gatecreeper Vine) alongside Ancient Stirrings';
K also covers Pinnacle Kill-Ship's own opponent-facing ETB target
(choose_opponent_permanent, same category as blocking's own cross-player
targeting -- correctly a no-op in every current 1-player Tron config,
since the underlying resolution auto-completes with no legal target)."""

import numpy as np

import game

from . import _actions_combat, _actions_mana
from ._actions_cast import (
    _CAST_MODE_BUTTON_MAX,
    _CAST_X_BUTTON_MAX,
    _DELVE_BUTTON_MAX,
    _activate_execute,
    _activate_legal,
    _activate_no_cost_execute,
    _activate_no_cost_legal,
    _cast_execute,
    _cast_legal,
    _cast_speed,
    _choose_cast_mode_execute,
    _choose_cast_mode_legal,
    _choose_cast_x_execute,
    _choose_cast_x_legal,
    _choose_delve_amount_execute,
    _choose_delve_amount_legal,
    _delve_execute,
    _delve_legal,
    _forestcycle_execute,
    _forestcycle_legal,
    _graveyard_ability_execute,
    _graveyard_ability_legal,
    _land_drop_execute,
    _land_drop_legal,
    _modal_execute,
    _modal_legal,
    _play_impulse_cast_execute,
    _play_impulse_cast_legal,
    _play_impulse_land_execute,
    _play_impulse_land_legal,
    _precast_choice_execute,
    _x_modal_execute,
    _x_modal_legal,
)
from ._actions_cast_altzone import (
    _alt_cast_execute,
    _alt_cast_legal,
    _cast_from_exile_execute,
    _cast_from_exile_legal,
    _flashback_execute,
    _flashback_legal,
    _omen_cast_execute,
    _omen_cast_legal,
    _plot_execute,
    _plot_legal,
)
from ._actions_combat import (
    _assign_blocker_execute,
    _assign_blocker_legal,
    _assign_damage_to_opponent_execute,
    _assign_damage_to_opponent_legal,
    _attack_execute,
    _attack_legal,
    _done_blocking_execute,
    _done_blocking_legal,
)
from ._actions_mana import (
    _choose_mana_color_execute,
    _choose_mana_color_legal,
    _filter_mana_execute,
    _filter_mana_legal,
    _mana_ability_execute,
    _mana_ability_legal,
    _mana_extra_choose_execute,
    _mana_extra_choose_legal,
    _mana_subdecision_color_execute,
    _mana_subdecision_color_legal,
)
from ._actions_resolution import (
    _CHOOSE_NAME_PENDING_KINDS,
    _choose_name_execute,
    _choose_name_legal,
    _choose_name_options,
    _choose_opponent_permanent_execute,
    _choose_opponent_permanent_legal,
    _choose_permanent_execute,
    _choose_permanent_legal,
    _choose_room_execute,
    _choose_room_legal,
    _decline_discard_execute,
    _decline_discard_legal,
    _decline_discard_or_sacrifice_execute,
    _decline_discard_or_sacrifice_legal,
    _decline_execute,
    _decline_graveyard_card_execute,
    _decline_graveyard_card_legal,
    _decline_legal,
    _decline_malevolent_rumble_execute,
    _decline_malevolent_rumble_legal,
    _decline_search_execute,
    _decline_search_legal,
    _discard_or_sacrifice_trigger_sacrifice_execute,
    _discard_or_sacrifice_trigger_sacrifice_legal,
    _dispose_execute,
    _keep_dispose_legal,
    _keep_execute,
    _madness_cast_execute,
    _madness_cast_legal,
    _madness_decline_execute,
    _madness_decline_legal,
    _may_cast_legal,
    _may_copy_legal,
    _may_transform_legal,
    _pass_execute,
    _pass_legal,
    _pay_unless_decline_execute,
    _pay_unless_decline_legal,
    _pay_unless_pay_execute,
    _pay_unless_pay_legal,
    _ponder_shuffle_execute,
    _ponder_shuffle_legal,
    _pool_spend_execute,
    _pool_spend_legal,
    _select_to_hand_bottom_execute,
    _select_to_hand_bottom_legal,
    _select_to_hand_keep_execute,
    _select_to_hand_keep_legal,
    _target_any_decline_execute,
    _target_any_decline_legal,
    _target_any_opponent_execute,
    _target_any_opponent_legal,
    _target_any_self_execute,
    _target_any_self_legal,
    _target_opponent_execute,
    _target_opponent_legal,
    _target_self_execute,
    _target_self_legal,
    _tuck_position_legal,
)


def build_action_table(decklist, registry, token_card_defs=(),
                        opponent_decklist=None, opponent_token_card_defs=(), extra_choosable_names=()):
    """opponent_decklist/opponent_token_card_defs: the OTHER side's own
    decklist/tokens -- None/() for every 1-player deck (there's no real
    opponent battlefield to reference at all), matching combat_enabled=False
    decks never seeing "Attack: X" become legal. The token/pointer pipeline
    passes None here (cross-player targeting moved to the pointer head, not
    a fixed opponent action table) -- kept for a fixed "Choose opponent's:
    X" table if one is ever wanted again.

    token_card_defs: every token CardDef this deck's own cards can
    create at runtime (Blood, Robot, Warrior, Eldrazi Spawn --
   ), e.g. (game.BLOOD_TOKEN_CARD_DEF,).
    Tokens are never decklist entries (no quantity, not in game.CARD_DEFS),
    so they can't flow through distinct_names/game.CARD_DEFS[name] the way
    every other action here does -- casting/land-drop/Flashback/etc. stay
    decklist-only, a token is never cast or played as a land.

    Two independent things read this list, for two different reasons: the
    activated-abilities loop below (a token's own ability, e.g. Blood's
    sac-for-a-card or Eldrazi Spawn's sac-for-{C}, needs an action to
    exist at all), and the choosable_names set that both the "Choose: X"
    and "Choose target: X (slot k)" name lists build from (a token can be
    a perfectly legal choose_permanent/sacrifice/discard choice -- e.g. any
    creature-enchanting Aura can enchant a token creature -- despite never
    appearing in the decklist; list a token here even if it has no
    activated ability of its own, like Warrior, so it stays a legal choice
    once it's on the battlefield). Defaults to () so every existing call
    site (Tron, spy_combo -- neither creates tokens) is unaffected.

    No `pending_kinds` parameter: whether a resolution kind can ever cross
    players is a fixed fact about the ENGINE (which cards target an
    opponent, which don't), not something a caller should have to compute
    and thread through -- see own_pending_kinds just below and the
    "UNIVERSAL DECISION ROWS" block further down for the two-way split this
    function makes on its own."""
    distinct_names = sorted({name for name, *_rest in decklist})
    # THIS decklist's own derived kinds. Used below for every kind confirmed
    # to be self-only (impulse, discard_or_sacrifice, may_transform,
    # may_cast, madness_decision, ancient_stirrings, malevolent_rumble,
    # select_to_hand, choose_mana_color -- each has exactly one call site in
    # the whole codebase, and it's always the deciding player's own card),
    # so those rows don't inflate a deck's table for a card it doesn't run.
    # The confirmed-cross-player kinds (pay_unless, tuck_position, may_copy,
    # choose_any_target, choose_room, choose_target_player, scry/
    # search_fetch's Decline, discard/graveyard-decline) don't consult this
    # at all -- they're unconditional in the "UNIVERSAL DECISION ROWS" block
    # further down, since an OPPONENT's card can pose those questions too.
    own_pending_kinds = game.derive_pending_kinds(decklist)
    land_names = sorted({
        name for name in distinct_names if game.CARD_DEFS[name].card_type == game.CardType.LAND
    })
    # Hoisted here (rather than just before their own first use, below) so the
    # extra-cost mana-ability loop can also use them: card_type_by_name/
    # qty_by_name index THIS side's own battlefield permanents (an opponent's
    # card is never a legal choose_permanent/attack/mana-extra-cost target of
    # mine); choosable_names is every name (real or token) this side could
    # ever have on its own battlefield.
    card_type_by_name = {name: game.CARD_DEFS[name].card_type for name in distinct_names}
    card_type_by_name.update({cd.name: cd.card_type for cd in token_card_defs})
    qty_by_name = {name: qty for name, qty, *_rest in decklist}
    # DFC back faces (Delver of Secrets -> Insectile Aberration): a transformed
    # permanent's card_def swaps to its back face (game.resolution.execute_may_
    # transform), so any BY-NAME resolution (order_triggers, sacrifice, discard,
    # search_fetch, ...) can end up needing to reference that back-face name --
    # omitting it here is the exact same softlock class as the token case just
    # above (a legal, on-battlefield object with no "Choose: X" row able to name
    # it), just for DFCs instead of tokens. Confirmed the hard way: a real
    # pretrain run hit "all-False action mask for pending kind 'order_triggers'"
    # the first time a transformed Delver's upkeep trigger needed ordering
    # against another simultaneous trigger. Discovered the same way
    # rl.features.CardVocab already discovers back faces: scan each front
    # face's own registry "transform" spec.
    back_face_defs = [
        spec["card_def"] for name in distinct_names
        for spec in [registry.get(game.CARD_DEFS[name].effect_id, {}).get("transform")]
        if spec is not None and "card_def" in spec
    ]
    card_type_by_name.update({cd.name: cd.card_type for cd in back_face_defs})
    choosable_names = sorted(set(distinct_names) | {cd.name for cd in token_card_defs} | {cd.name for cd in back_face_defs})

    actions = []

    for name in land_names:
        actions.append((f"Play land: {name}", _land_drop_legal(name), _land_drop_execute(name)))

    # Mana abilities: "Tap X [for <color>]" produces mana into the
    # pool in ANY priority window, even mid-resolution of anything else
    # (605.1a/605.3b -- a mana ability never uses the stack and doesn't
    # require priority; see _mana_ability_legal's own docstring for why no
    # pending-resolution gate is needed at all). Row count is DERIVED per
    # source rather than a blanket "no-color + all 6 pool colors" for every
    # name: the registry's own "mana" spec kind already says, statically,
    # which colors that source's NATIVE ability could ever offer --
    # game.mana_ability_options only ever emits (name, None) for a fixed/
    # fixed_multi/tron/count/count_all source and (name, color) for color in
    # its own spec[1] for a flexible one -- so a row neither of those can
    # ever produce is a permanently-dead output neuron. "Tap X for C" is
    # never one of the rows generated below: colorless is only ever produced
    # via the no-color row (a fixed/tron/count source's own single output),
    # never a color CHOICE, for any source or grant this engine has.
    #
    # LANDS are the one case a per-source static lookup alone would miss:
    # Abundant Growth's cast_aura targets "any land, either side"
    # (choose_any_target) and grants a color via game.mana._granted_mana_
    # colors ON TOP OF the land's own native ability -- a runtime fact no
    # single card's "mana" spec captures. Any land in ANY league deck can
    # end up enchanted (this deck's own, or an opponent's, since Abundant
    # Growth can target either side), so every land-type source keeps a row
    # for every color the FULL registry ever declares grantable
    # ("grants_mana_colors", e.g. Abundant Growth's own entry), unioned with
    # its own native colors -- this is the one kind-of-decision in this
    # function that DOES need to look past just this decklist's own cards.
    grantable_mana_colors = set().union(*(spec.get("grants_mana_colors", set()) for spec in registry.values()))
    mana_source_effect_id = {}
    for name in distinct_names:
        effect_id = game.CARD_DEFS[name].effect_id
        spec = registry.get(effect_id, {})
        if "mana" in spec and "mana_extra_choose" not in spec:
            mana_source_effect_id[name] = effect_id
    for cd in token_card_defs:
        spec = registry.get(cd.effect_id, {})
        if "mana" in spec and "mana_extra_choose" not in spec:
            mana_source_effect_id[cd.name] = cd.effect_id
    for name in sorted(mana_source_effect_id):
        kind, *rest = registry[mana_source_effect_id[name]]["mana"]
        if kind != "flexible":
            # flexible is the ONLY kind whose native ability never offers
            # (name, None) -- see mana_ability_options's own kind grouping.
            actions.append((f"Tap {name}", _mana_ability_legal(name, None), _mana_ability_execute(name, None)))
        colors = set(rest[0]) if kind == "flexible" else set()
        if card_type_by_name.get(name) == game.CardType.LAND:
            colors |= grantable_mana_colors
        for color in game.COLORS:
            if color in colors:
                actions.append((
                    f"Tap {name} for {color}",
                    _mana_ability_legal(name, color),
                    _mana_ability_execute(name, color),
                ))

    # Saruli Caretaker-shaped mana abilities: an extra cost that taps ANOTHER
    # untapped creature (mana_extra_choose) -- a COST CHOICE (602.5g), not a
    # target. ONE gate-free "Tap <name>" row per source name (same shape as
    # an ordinary mana ability, mana_source_names above); its execute opens
    # a mana_subdecision (game.begin_mana_subdecision) -- choose which
    # creature to tap (pointer-routed, rl.action_bridge, no fixed-table row
    # needed), THEN choose a color (the shared "Produce <color>" buttons
    # below) -- matching real sequencing (602.5g cost paid before the
    # ability's own color-choice effect resolves) while staying gate-free
    # (605.1a/605.3b): mana_subdecision is a field SEPARATE from
    # pending_resolution specifically so this can still open mid-resolution
    # of anything else without clobbering it -- see state.mana_subdecision's
    # own docstring. needs_mana_subdecision_color_buttons is shared with the
    # mana-filter block below -- either mechanic reaches the SAME
    # choose_color stage (game.begin_mana_color_choice), so a deck needs the
    # buttons if EITHER is present, even with only one of the two.
    needs_mana_subdecision_color_buttons = False
    extra_tap_source_names = sorted(
        {name for name in distinct_names if "mana_extra_choose" in registry.get(game.CARD_DEFS[name].effect_id, {})}
        | {cd.name for cd in token_card_defs if "mana_extra_choose" in registry.get(cd.effect_id, {})}
    )
    for name in extra_tap_source_names:
        actions.append((f"Tap {name}", _mana_extra_choose_legal(name), _mana_extra_choose_execute(name)))
        needs_mana_subdecision_color_buttons = True

    # Mana filters (Conduit Pylons / Barrels of Blasting Jelly): "{1}: add one
    # mana of any color" -- the {1} half is a flat fixed-table row per
    # (source, input_color): which floating pip pays it, paid immediately.
    # input_color ranges over every pool color (any floating pip, incl.
    # colorless, can pay a generic {1}). The output half (which color comes
    # out) is a SEPARATE, later choice via the shared choose_color
    # mana_subdecision stage above -- reusing Saruli's own "Produce <color>"
    # buttons rather than a second per-row dimension (output_color x
    # input_color used to be a real cross product here; see
    # _filter_mana_execute). Never a nested pay_cost for the {1} itself:
    # folding it into one would open a second pending resolution
    # mid-activation, at risk of clobbering whatever's already open.
    filter_source_names = sorted({n for n in distinct_names if "filter_mana" in registry.get(game.CARD_DEFS[n].effect_id, {})})
    for name in filter_source_names:
        for input_color in game.POOL_COLORS:
            actions.append((
                f"Filter {name}, paying {input_color}",
                _filter_mana_legal(name, input_color),
                _filter_mana_execute(name, input_color),
            ))
    if filter_source_names:
        needs_mana_subdecision_color_buttons = True

    # Tracked so the generic choose_cast_mode/choose_cast_x/choose_delve_amount
    # buttons below only get added to a deck's table that actually has a card
    # using that shape -- these are MY OWN casting decisions (never posed by
    # an opponent's card, unlike the pay_unless/tuck_position-style universal
    # rows further down), so a deck with none of these cards never needs them.
    needs_cast_mode_buttons = False
    needs_cast_x_buttons = False
    needs_delve_buttons = False

    for name in distinct_names:
        card_spec = registry.get(game.CARD_DEFS[name].effect_id, {})
        cast_spec = card_spec.get("cast")
        if cast_spec is not None:
            # "precast_choice": True (Auras' real targets, Crop Rotation's
            # sacrifice-as-a-cost) -- resolve must run immediately once paid
            # and manage its own push_to_stack, instead of the generic
            # auto-push _cast_execute does (see _precast_choice_execute's
            # own docstring for exactly why).
            cast_execute_fn = _precast_choice_execute if cast_spec.get("precast_choice") else _cast_execute
            actions.append((
                f"Cast {name}",
                _cast_legal(name, cast_spec.get("extra_legal"), _cast_speed(game.CARD_DEFS[name], cast_spec)),
                cast_execute_fn(name, cast_spec["resolve"]),
            ))
        # Winding Way/Utopia Sprawl/Goblin Bushwhacker: a modal cast -- ONE
        # "Cast <name>" row (not one per mode); which mode is a generic
        # choose_cast_mode sub-decision (601.2b: chosen before the cost is
        # calculated), shared "Mode 1".."Mode 5" buttons reused by every
        # modal card instead of a per-card, per-mode fixed row.
        cast_modes = card_spec.get("cast_modes")
        if cast_modes is not None:
            mode_items = list(cast_modes.items())
            speed = _cast_speed(game.CARD_DEFS[name], mode_items[0][1])
            actions.append((f"Cast {name}", _modal_legal(name, mode_items, speed), _modal_execute(name, mode_items)))
            needs_cast_mode_buttons = True
        # Nyxborn Hydra: X-cost modes (its own normal creature cast AND
        # Bestow, each with a different base cost) -- ONE "Cast <name>" row;
        # mode chosen first (601.2b), X chosen second (601.2f), both generic
        # sub-decisions (shared "Mode 1".."Mode 5" then "X=0".."X=10"
        # buttons) instead of one fixed row per (mode, X) pair. Each mode's
        # own "resolve" is still a function OF x returning the (state,
        # card_def) resolve itself (green_cards.cast_nyxborn_hydra_creature/
        # cast_nyxborn_hydra_bestow) -- called once X is actually chosen
        # (_x_modal_execute), not baked in at table-build time anymore.
        x_cast_modes = card_spec.get("x_cast_modes")
        if x_cast_modes is not None:
            mode_items = list(x_cast_modes.items())
            speed = _cast_speed(game.CARD_DEFS[name], mode_items[0][1])
            actions.append((f"Cast {name}", _x_modal_legal(name, mode_items, speed), _x_modal_execute(name, mode_items)))
            needs_cast_mode_buttons = True
            needs_cast_x_buttons = True
        # Gurmag Angler: Delve -- ONE "Cast <name>" row; the delve amount is
        # a generic choose_delve_amount sub-decision (702.66: chosen before
        # the exile sub-cost opens and the reduced cost is calculated),
        # shared "Delve 0".."Delve 6" buttons instead of one fixed row per N.
        delve = card_spec.get("delve")
        if delve is not None:
            delve_speed = _cast_speed(game.CARD_DEFS[name], delve)
            actions.append((
                f"Cast {name}",
                _delve_legal(name, delve["max"], delve_speed),
                _delve_execute(name, delve["max"], delve["resolve"]),
            ))
            needs_delve_buttons = True
        # Land Grant: a second, free cast path alongside the normal one.
        alt_cast = card_spec.get("alt_cast")
        if alt_cast is not None:
            actions.append((
                f"Cast {name} (free)",
                _alt_cast_legal(name, alt_cast["extra_legal"], _cast_speed(game.CARD_DEFS[name], alt_cast)),
                _alt_cast_execute(name, alt_cast["resolve"]),
            ))
        # Dread Return: Flashback casts from the graveyard, not hand. Escape
        # (Sleep of the Dead) is the same graveyard-cast machinery with a mana
        # cost + its own additional cost handled inside the resolve -- shares
        # _flashback_legal/_execute, only the action label differs.
        for gy_cast_key, gy_cast_label in (("flashback", "Flashback"), ("escape", "Escape")):
            gy_cast = card_spec.get(gy_cast_key)
            if gy_cast is not None:
                gc_cost = gy_cast.get("cost")  # mana cost dict, if the cost includes mana (Deep Analysis, Sleep of the Dead)
                actions.append((
                    f"{gy_cast_label} {name}",
                    _flashback_legal(name, gy_cast["legal"], _cast_speed(game.CARD_DEFS[name], gy_cast), gc_cost),
                    _flashback_execute(name, gy_cast["resolve"], gc_cost),
                ))
        # Highway Robbery: Plot -- pay its plot cost to exile it now,
        # cast it for free from exile on any later turn. The cast-from-
        # exile half reuses this same card's normal "cast" resolve (the
        # real spell effect is identical either way, only how the cost
        # was paid differs) -- so a "plot" entry only makes sense
        # alongside a "cast" entry, never alone.
        plot = card_spec.get("plot")
        if plot is not None:
            plot_speed = _cast_speed(game.CARD_DEFS[name], plot)  # same speed governs both actions below -- Plot's own reminder text ("any time you could cast this card") is one timing rule, not two
            actions.append((
                f"Plot {name}",
                _plot_legal(name, plot["cost"], plot_speed),
                _plot_execute(name, plot["cost"], plot["resolve"]),
            ))
            # cast_from_exile_resolve: optional override for cards whose
            # normal "cast" resolve does state.hand.remove(card_def) (the
            # universal convention for every cast resolve in this codebase)
            # -- wrong once the card already left exile, never hand, by
            # the time this runs (Highway Robbery's own real-world case;
            # every existing Plot self-check's resolve happens to be a
            # no-op, which is why this distinction never mattered before).
            # Falls back to cast_spec["resolve"] unchanged for any card
            # whose resolve doesn't care either way.
            actions.append((
                f"Cast {name} (plotted)",
                _cast_from_exile_legal(name, cast_spec.get("extra_legal"), plot_speed),
                _cast_from_exile_execute(name, plot.get("cast_from_exile_resolve", cast_spec["resolve"])),
            ))
        # Sagu Wildling: Omen -- cast_roost_seek (this same "name"'s own
        # "cast" resolve above) shuffles the card into the LIBRARY instead
        # of graveyarding OR exiling it (real Adventure's own exile
        # doesn't apply to Omen -- see _omen_cast_legal's own docstring).
        # This second action is just an ordinary-looking second cast
        # option for the same hand card, its own real cost, offered
        # whenever a same-named card is back in hand (redrawn after being
        # shuffled in) -- never shares a resolve with the sorcery mode,
        # the creature side is a wholly different spell.
        omen = card_spec.get("omen")
        if omen is not None:
            omen_speed = _cast_speed(omen["card_def"], omen)
            actions.append((
                f"Cast {omen['card_def'].name} (omen)",
                _omen_cast_legal(name, omen["cost"], omen_speed),
                _omen_cast_execute(omen["card_def"], omen["cost"], omen["resolve"]),
            ))
        # Boulderbranch Golem: Prototype -- a second cast option for the same
        # hand card, its own cheaper cost, producing a DIFFERENT CardDef (the
        # smaller 3/3 with its own ETB). Structurally identical to Omen ("cast
        # this hand card for an alternate cost as a different creature"), so it
        # reuses the same _omen_cast_legal/_omen_cast_execute helpers -- only
        # the resolve differs (no library shuffle; the prototype creature just
        # enters). Real reminder text: "You may cast this spell with different
        # mana cost, color, and size. It keeps its abilities and types."
        prototype = card_spec.get("prototype")
        if prototype is not None:
            proto_speed = _cast_speed(prototype["card_def"], prototype)
            actions.append((
                f"Cast {name} (prototype)",
                _omen_cast_legal(name, prototype["cost"], proto_speed),
                _omen_cast_execute(prototype["card_def"], prototype["cost"], prototype["resolve"]),
            ))

    activatable = [(name, game.CARD_DEFS[name].effect_id) for name in distinct_names]
    activatable += [(cd.name, cd.effect_id) for cd in token_card_defs]
    for name, effect_id in activatable:
        abilities = registry.get(effect_id, {}).get("activated_abilities", {})
        for ability_name, spec in abilities.items():
            # Real Magic's own default for activated (non-mana) abilities is
            # the opposite of a spell's: any time, unless the card says
            # "activate only as a sorcery" -- an explicit "speed" key in the
            # ability's own spec is that opt-in override; every existing
            # ability (Blood, Candy Trail, Expedition Map, Bonders'
            # Ornament, Quirion Ranger, Barrels) has none, so all keep
            # working in every phase, unrestricted by speed.
            speed = spec.get("speed", game.turn.Speed.INSTANT)
            if "cost_key" in spec:
                actions.append((
                    f"Activate {name} ({ability_name})",
                    _activate_legal(name, spec["cost_key"], speed, spec.get("extra_legal")),
                    _activate_execute(name, spec["cost_key"], spec["resolve"]),
                ))
            else:
                # Non-mana cost (Quirion Ranger: return a Forest to hand).
                actions.append((
                    f"Activate {name} ({ability_name})",
                    _activate_no_cost_legal(name, spec["legal"], speed),
                    _activate_no_cost_execute(name, spec["resolve"]),
                ))

    # "Discard this card from hand: <do something>" cycling-family actions.
    # Both keys share the identical hand-zone/cost-key/resolve plumbing
    # (_forestcycle_legal/_execute) -- they differ only in the action label:
    #   "forestcycle" -- basic-land-to-hand search (Generous Ent, Ash Barrens)
    #   "cycle"       -- plain Cycling (discard, draw) and typed cycling like
    #                    Islandcycling (Lorien Revealed) / Twisted Landscape
    for name in distinct_names:
        for spec_key, label in (("forestcycle", "Forestcycle"), ("cycle", "Cycle")):
            cyc_spec = registry.get(game.CARD_DEFS[name].effect_id, {}).get(spec_key)
            if cyc_spec is not None:
                actions.append((
                    f"{label} {name}",
                    _forestcycle_legal(name, cyc_spec["cost_key"]),
                    _forestcycle_execute(name, cyc_spec["cost_key"], cyc_spec["resolve"]),
                ))

    # Impulse: "you may play the exiled cards" (Reckless Impulse / Experimental
    # Synthesizer / Clockwork Percussionist). Only emitted for a deck that can
    # actually impulse (its impulse-source card declares pending_kinds
    # {"impulse"}), so decks without one never carry these mostly-illegal
    # actions. One action per deck card name: a land play, or a cast per the
    # card's own cast/cast_modes spec (paying its NORMAL cost -- impulse, unlike
    # Plot, is not free; x_cast_modes cards, none in the impulse decks, aren't
    # offered from impulse).
    #
    # Gated on THIS decklist's own derived kinds -- unlike pay_unless/
    # tuck_position (genuinely posed by an OPPONENT's card, unconditional in
    # the "UNIVERSAL DECISION ROWS" block further down), impulse is never
    # cross-player: state.impulse is always the ACTIVE player's own zone
    # (game/state.py's _active_player_property) and every impulse source
    # only ever exiles from its own controller's library (shared.
    # impulse_exile's own docstring). A deck without an impulse card of its
    # own never has "impulse" in own_pending_kinds, so it never carries a
    # "Play from exile: X" row per own land/castable card for nothing --
    # permanently illegal, since no card of theirs ever opens the zone.
    if "impulse" in own_pending_kinds:
        for name in distinct_names:
            card_def = game.CARD_DEFS[name]
            if card_def.card_type == game.CardType.LAND:
                actions.append((f"Play from exile: {name}", _play_impulse_land_legal(name), _play_impulse_land_execute(name)))
                continue
            spec = registry.get(card_def.effect_id, {})
            cast_spec = spec.get("cast")
            if cast_spec is not None:
                speed = _cast_speed(card_def, cast_spec)
                actions.append((
                    f"Play from exile: {name}",
                    _play_impulse_cast_legal(name, card_def.cast_cost, cast_spec.get("extra_legal"), speed),
                    _play_impulse_cast_execute(name, card_def.cast_cost, cast_spec["resolve"], cast_spec.get("precast_choice", False)),
                ))
            for mode_name, mode_spec in (spec.get("cast_modes") or {}).items():
                mode_cost = mode_spec.get("cost", card_def.cast_cost)
                speed = _cast_speed(card_def, mode_spec)
                actions.append((
                    f"Play from exile: {name} ({mode_name})",
                    _play_impulse_cast_legal(name, mode_cost, mode_spec.get("extra_legal"), speed),
                    _play_impulse_cast_execute(name, mode_cost, mode_spec["resolve"], mode_spec.get("precast_choice", False)),
                ))

    # Bramble Wurm: an activated ability usable from the graveyard, not
    # the battlefield (unlike every "activated_abilities" entry above) or
    # hand (unlike Forestcycle) -- its own "graveyard_ability" registry key.
    for name in distinct_names:
        gy_spec = registry.get(game.CARD_DEFS[name].effect_id, {}).get("graveyard_ability")
        if gy_spec is not None:
            actions.append((
                f"Activate {name} (graveyard)",
                _graveyard_ability_legal(name, gy_spec["cost_key"]),
                _graveyard_ability_execute(name, gy_spec["cost_key"], gy_spec["resolve"]),
            ))

    actions.append(("Pass", _pass_legal, _pass_execute))

    # "Choose: X" needs to cover every name a sacrifice/discard/search_fetch/
    # etc. resolution could ever offer -- not just decklist names. (NOT
    # choose_permanent -- that's the "Choose target: X (slot k)" block
    # below, exact-(name, slot) addressed.) A token (e.g. boggles' Eldrazi
    # Spawn) is a perfectly legal sacrifice/discard choice despite never
    # appearing in CARD_DEFS/the decklist; omitting token names here left
    # exactly that case legal-to-create but impossible-to-choose once a
    # token was the only eligible option, softlocking the game.
    # extra_choosable_names: card names that can be a "Choose: X" option despite
    # not being in THIS deck (nor its tokens) -- e.g. an opponent's graveyard
    # cards. The league (rl.pool) passes none: choose_graveyard_card is a
    # POINTER target (rl.action_bridge), so an opponent's graveyard is reached
    # by pointing at its token, not a whole-league "Choose: X" row per card
    # name. Kept as a general knob, used by
    # scripts/migrate_pointer_graveyard.py to reconstruct the pre-pointer
    # table; defaults to none. choosable_names
    # itself (not just extra_choosable_names) also drives "Choose target: X
    # (slot k))" and "Attack: X (slot k)" below -- both strictly THIS side's
    # own battlefield permanents.
    choose_by_name = sorted(set(choosable_names) | set(extra_choosable_names))
    for name in choose_by_name:
        actions.append((f"Choose: {name}", _choose_name_legal(name), _choose_name_execute(name)))

    # "Attack: X (slot k)" -- one per (creature name, slot) pair
    #, legal only during
    # Phase.DECLARE_ATTACKERS (see _attack_legal). k runs 1..that card's
    # own decklist quantity for a real card -- the pooled slot scheme
    # means this is a hard, correct bound even through repeated bounce/
    # blink, since only that many physical copies can ever be
    # simultaneously alive. A token has no decklist quantity to read, so
    # k instead runs 1..TOKEN_LIMIT -- a shared pool across every token
    # name combined, so any single name could in principle claim all of
    # it, and each name's own registered range has to cover that worst
    # case independently. A deck whose own phase sequence never includes
    # DECLARE_ATTACKERS (combat_enabled=False) simply never sees any of
    # these become legal -- same "phase not in this deck's own sequence"
    # degrade every other phase-gated action already relies on.

    # "Choose target: X (slot k)" -- the "choose_permanent" resolution's own
    # exact-(name, slot) addressed actions (Aura enchant-targets, Crop
    # Rotation's sacrifice cost, land bounce), same shape/reasoning as
    # "Choose opponent's: X (slot k)" below just scoped to THIS side's own
    # battlefield. Registered for every choosable name, not just creatures
    # (unlike "Attack:"/"Assign Blocker:" below) -- Utopia Sprawl/Abundant
    # Growth target lands, not creatures -- and legal() gates precisely at
    # runtime against whichever predicate the actual pending choose_permanent
    # resolution holds, same "pre-register broadly, mask precisely" pattern
    # "Choose: X as color" below already uses.
    for name in choosable_names:
        max_slot = qty_by_name.get(name, game.TOKEN_LIMIT)
        for slot in range(1, max_slot + 1):
            actions.append((
                f"Choose target: {name} (slot {slot})",
                _choose_permanent_legal(name, slot),
                _choose_permanent_execute(name, slot),
            ))

    attackable_names = sorted(name for name in choosable_names if card_type_by_name[name] == game.CardType.CREATURE)
    for name in attackable_names:
        max_slot = qty_by_name.get(name, game.TOKEN_LIMIT)
        for slot in range(1, max_slot + 1):
            actions.append((
                f"Attack: {name} (slot {slot})",
                _attack_legal(name, slot),
                _attack_execute(name, slot),
            ))

    # "Assign Blocker: X (slot j)" -- same own-creature (name, slot)
    # addressing as "Attack: X (slot k)" above, since blocking is a
    # decision about this player's OWN creatures,
    # just legal at a different point (once _declare_blockers_gen has
    # flipped state.active_idx to the defender and a "declare_blockers"
    # resolution is pending -- see _assign_blocker_legal). "Done blocking"
    # is the explicit action that closes the consult, same "Done" precedent
    # as scry/surveil's own keep-then-order decomposition. Deliberately NO
    # AUTHORIZED SIMPLIFICATION (owner, 2026-07-31): no undo (no "Unassign
    # Blocker") -- once committed, a blocker stays committed until Done. Real
    # Magic's declare-blockers is a single simultaneous action (509.1); this
    # engine linearizes it into one-creature-at-a-time picks with no way
    # back, by design, across the whole engine (no action anywhere lets the
    # agent reconsider/reverse an earlier commitment -- see
    # todo/no_undo_policy.md for the broader rationale).
    # An "Unassign Blocker" action would let assign/unassign cycle
    # indefinitely with turn_number never advancing (a real, observed
    # pathology -- tens of thousands of iterations in one boggles_mirror
    # evaluation, back when a per-round action cap existed to force an end
    # to it). That cap is gone now (game/turn.py, 2026-08-19 -- no
    # iteration cap exists anywhere in the turn loop), so the no-undo rule
    # is the ONLY thing foreclosing this pathology -- there is no longer a
    # backstop if it were ever reintroduced.
    for name in attackable_names:
        max_slot = qty_by_name.get(name, game.TOKEN_LIMIT)
        for slot in range(1, max_slot + 1):
            actions.append((
                f"Assign Blocker: {name} (slot {slot})",
                _assign_blocker_legal(name, slot),
                _assign_blocker_execute(name, slot),
            ))
    actions.append(("Done blocking", _done_blocking_legal, _done_blocking_execute))
    # Trample-to-player is no longer an agent choice (game.resolution.
    # handlers_combat._autoresolve_if_no_choices_left applies it
    # automatically instead) -- this row's legal() is now permanently
    # False. Kept registered anyway, purely so the fixed table's length --
    # and every trained DeckNetwork's action-output shape -- stays stable;
    # see _assign_damage_to_opponent_legal's own docstring.
    actions.append((
        "Assign combat damage to opponent", _assign_damage_to_opponent_legal, _assign_damage_to_opponent_execute,
    ))

    # "Choose opponent's: X (slot k)" -- the general cross-player
    # targeting primitive, one per (opponent name, slot), built from the
    # OPPONENT's own decklist/tokens instead of this side's own. Registered
    # for every opponent choosable name, not just creatures -- same
    # "pre-register broadly, mask precisely" pattern as "Choose target: X"
    # above: blocking's predicate only ever matches attackers (creatures),
    # but Masked Vandal's ETB (green_cards.py) targets an opponent artifact
    # or enchantment, so a creature-only filter here left that a legal
    # target with no matching action row -- an all-False mask crash the
    # instant an opponent controlled a targetable artifact/enchantment.
    # Same quantity-or-TOKEN_LIMIT bound as the attack registration above,
    # just applied to the other side's card pool. None/() (the default for
    # every 1-player deck) registers nothing at all -- there's no real
    # opponent battlefield to ever reference in that mode.
    if opponent_decklist is not None:
        opponent_distinct_names = sorted({name for name, *_rest in opponent_decklist})
        opponent_qty_by_name = {name: qty for name, qty, *_rest in opponent_decklist}
        opponent_choosable_names = sorted(
            set(opponent_distinct_names) | {cd.name for cd in opponent_token_card_defs}
        )
        for name in opponent_choosable_names:
            max_slot = opponent_qty_by_name.get(name, game.TOKEN_LIMIT)
            for slot in range(1, max_slot + 1):
                actions.append((
                    f"Choose opponent's: {name} (slot {slot})",
                    _choose_opponent_permanent_legal(name, slot),
                    _choose_opponent_permanent_execute(name, slot),
                ))

    # No "Choose: X as color" rows: a flexible or granted source's color is
    # chosen at TAP time (the "Tap X for <color>" mana-ability rows above float
    # directly into the pool), whether that tap happens speculatively in a main
    # phase or inside a payment. Spending what is floating is the separate row
    # set below.
    for color in game.POOL_COLORS:
        actions.append((
            f"Spend {color} from pool",
            _pool_spend_legal(color),
            _pool_spend_execute(color),
        ))

    # ---- UNIVERSAL DECISION ROWS (deliberately NOT gated on any pending kind) ----
    # Every fixed row below answers a QUESTION the engine can pose, and each is a
    # small constant set of buttons (never a per-card-name loop). They are added
    # to EVERY deck's table unconditionally, because a decision is not always
    # answered by the player whose deck produced it -- an opponent's card can pose
    # a question that YOU must answer:
    #   pay_unless          Spell Pierce/Ward/Nihil Spellbomb/Chain Lightning --
    #                       the PAYER is the spell's controller, i.e. the opponent.
    #   tuck_position       Deem Inferior -- the tucked permanent's OWNER chooses.
    #   may_copy            Chain Lightning -- the affected player may copy.
    #   choose_any_target   a Chain Lightning COPIER may choose a new target.
    #   choose_target_player/scry/search_fetch  Undercity's own Trap/Lost Well/
    #                       Secret Entrance rooms -- initiative can be STOLEN
    #                       (combat damage), so a deck with none of these kinds
    #                       among its own cards can still be the one venturing
    #                       and facing them; search_fetch's Decline is ALSO
    #                       independently reachable via Cleansing Wildfire (the
    #                       DESTROYED land's controller searches their own
    #                       library, not the caster).
    #   discard-decline, choose_graveyard_card-decline  same shape (Refurbished
    #                       Familiar / Relic of Progenitus target the opponent).
    # `derive_pending_kinds` reads a deck's OWN cards, so gating these on it left
    # the answering seat with no row for the question and an all-False action mask
    # -- a hard crash (real: monster_tron/mono_blue_terror league play; an
    # empirical cross-deck audit reproduced pay_unless and tuck_position and
    # flagged choose_graveyard_card/discard/surveil as latent). Rather than
    # maintain a hand-audited list of "which kinds may cross" -- which silently
    # rots the next time a card hands a choice to the opponent -- every constant
    # decision-button set whose kind COULD ever cross players is always present
    # and runtime-gated illegal, exactly the "present-but-permanently-illegal"
    # footing "Decline (graveyard)" and "Decline (search)" already documented
    # for themselves.
    #
    # Gated below on own_pending_kinds instead (confirmed, not assumed, self-
    # only -- each has exactly one call site and it's always the deciding
    # player's own card): may_transform (Delver of Secrets' own upkeep --
    # mono_blue_terror only), may_cast (Cascade -- monster_tron only),
    # choose_mana_color (Chromatic Star -- grixis_affinity only),
    # ancient_stirrings/malevolent_rumble/select_to_hand (each one specific
    # card's own search/rummage), madness_decision (madness always triggers
    # off YOUR OWN discarded card, whoever forced the discard), and
    # discard_or_sacrifice (Highway Robbery's sacrifice trigger is the
    # CASTER's own optional cost, never posed to an opponent -- also true of
    # `impulse` just above, for the identical reason). Making any of these
    # universal would have repeated the exact bloat impulse's own history
    # already illustrates: real rows for a card the deck never runs.
    actions.append(("Keep (scry/surveil)", _keep_dispose_legal, _keep_execute))
    actions.append(("Dispose (scry/surveil)", _keep_dispose_legal, _dispose_execute))
    # Self-only (Ancient Stirrings/Malevolent Rumble are each one specific
    # card's own search/rummage) -- see this block's own header comment.
    if "ancient_stirrings" in own_pending_kinds:
        actions.append(("Decline (Ancient Stirrings)", _decline_legal, _decline_execute))
    if "malevolent_rumble" in own_pending_kinds:
        actions.append(("Decline (Malevolent Rumble)", _decline_malevolent_rumble_legal, _decline_malevolent_rumble_execute))
    actions.append(("Shuffle (Ponder)", _ponder_shuffle_legal, _ponder_shuffle_execute))
    actions.append(("Pay (unless)", _pay_unless_pay_legal, _pay_unless_pay_execute))
    actions.append(("Don't pay (unless)", _pay_unless_decline_legal, _pay_unless_decline_execute))
    actions.append(("Tuck: 2nd from top", _tuck_position_legal, lambda state: game.execute_tuck_position(state, "top2")))
    actions.append(("Tuck: bottom", _tuck_position_legal, lambda state: game.execute_tuck_position(state, "bottom")))
    # Self-only (Delver of Secrets' own upkeep trigger -- only its own
    # controller ever faces this) -- see this block's own header comment.
    if "may_transform" in own_pending_kinds:
        actions.append(("Transform", _may_transform_legal, lambda state: game.execute_may_transform(state, True)))
        actions.append(("Don't transform", _may_transform_legal, lambda state: game.execute_may_transform(state, False)))
    actions.append(("Copy spell", _may_copy_legal, lambda state: game.execute_may_copy(state, True)))
    actions.append(("Don't copy spell", _may_copy_legal, lambda state: game.execute_may_copy(state, False)))
    # Self-only (Cascade's may-cast of the hit is always the CASCADING
    # player's own choice) -- see this block's own header comment.
    if "may_cast" in own_pending_kinds:
        actions.append(("Cast (may)", _may_cast_legal, lambda state: game.execute_may_cast(state, True)))
        actions.append(("Decline (may)", _may_cast_legal, lambda state: game.execute_may_cast(state, False)))
    # Self-only (Lead the Stampede's own reveal-and-pick) -- see this block's
    # own header comment.
    if "select_to_hand" in own_pending_kinds:
        actions.append(("Keep (select to hand)", _select_to_hand_keep_legal, _select_to_hand_keep_execute))
        actions.append(("Bottom (select to hand)", _select_to_hand_bottom_legal, _select_to_hand_bottom_execute))
    actions.append(("Decline (search)", _decline_search_legal, _decline_search_execute))
    # Only ever legal for an OPTIONAL choose_graveyard_card (Masked Vandal's "you
    # may exile a creature from your graveyard"); the legal_fn itself gates on
    # pending["optional"], so it is present-but-permanently-illegal for decks
    # whose only graveyard picks are mandatory (Dread Return, Relic). Universal
    # per the block above -- Relic of Progenitus already poses its graveyard pick
    # to the TARGETED player, so the decline row has to exist in their table too
    # the day any cross-player graveyard pick becomes optional.
    actions.append(("Decline (graveyard)", _decline_graveyard_card_legal, _decline_graveyard_card_execute))
    # NO "Abandon payment" -- there is no undo. Real Magic rewinds a spell whose
    # cost cannot be paid (an illegal-action correction, not a normal game
    # action), and this engine keeps the payment unabandonable instead, so a
    # payment that cannot be finished leaves the agent with no legal action at
    # all (an all-False mask, a hard error). Every cast/activate action and Pass
    # are illegal while a pending is open, so there is nowhere else to go.
    #
    # Under cast-then-pay that guarantee needs real work, and it is NOT
    # self-enforcing. See game.mana's own STRANDING INVARIANT for the full
    # statement; the two halves are:
    #   1. game.plan_payment is EXACT (game.can_pay, Hall's condition), so a
    #      payment only ever begins when it can be finished -- and
    #      begin_pay_cost asserts that itself, so a caller that skips the gate
    #      fails there rather than several actions later here;
    #   2. every action available DURING a payment is gated on the payment
    #      surviving it (game.payment_survives). Tapping is not automatically
    #      safe: a source with a color choice counts as one unit that could be
    #      any of its colors, and tapping collapses it to one. Filters tap their
    #      own source, and Conduit Pylons is also a {C} source. Saruli's cost
    #      taps another creature that may itself be a mana source.
    # Mana abilities are additionally illegal for the whole of an in-flight cast
    # before 601.2f (game.mid_cast), which is both faithful and removes a whole
    # class of these hazards rather than guarding each one.
    #
    # Any FUTURE action that can reduce available mana mid-payment must add the
    # same payment_survives gate.
    # NOTE: the pregame mulligan actions ("Keep hand" / "Mulligan") and the
    # "mulligan_bottom" branch of the generic "Choose: X" action have no rows in
    # this table. The per-deck MulliganNet (rl.mulligan) owns every pregame
    # decision -- rl.agent.SeatAgent intercepts the pregame pending kinds before
    # the main net's forward -- so the main policy's action space contains ZERO
    # pregame actions and a game can never fall back to a fixed-table mulligan.
    # The _mulligan_*_legal/_execute helpers remain exported for the engine's own
    # use, not wired into any table.
    # Refurbished Familiar already makes the OPPONENT discard from their own hand
    # (the "Choose: X" hand-name rows are their own, so those self-serve) -- but
    # the decline row is a constant button, so it is universal per the block above.
    actions.append(("Decline (discard)", _decline_discard_legal, _decline_discard_execute))
    if "discard_or_sacrifice" in own_pending_kinds:
        # Self-only: Highway Robbery's "discard a card or sacrifice a land"
        # is the CASTER's own optional cost, never posed to an opponent --
        # see this block's own header comment. The DISCARD half reuses the
        # generic "Choose: X" action built above (bare hand-card names); the
        # SACRIFICE half is a single trigger action that opens its own
        # nested choose_permanent pointer choice for WHICH exact permanent
        # to sacrifice -- see _discard_or_sacrifice_trigger_sacrifice_legal's
        # own docstring for why this isn't a per-land-name loop anymore.
        actions.append((
            "Sacrifice a permanent (discard or sacrifice)",
            _discard_or_sacrifice_trigger_sacrifice_legal,
            _discard_or_sacrifice_trigger_sacrifice_execute,
        ))
        # Same self-only gate as the trigger action just above -- a decline
        # button for a resolution kind that can never open has nothing to
        # decline.
        actions.append((
            "Decline (discard or sacrifice)",
            _decline_discard_or_sacrifice_legal,
            _decline_discard_or_sacrifice_execute,
        ))
    # Self-only (madness always triggers off YOUR OWN discarded card,
    # whoever forced the discard) -- see this block's own header comment.
    if "madness_decision" in own_pending_kinds:
        actions.append(("Cast (madness)", _madness_cast_legal, _madness_cast_execute))
        actions.append(("Decline (madness)", _madness_decline_legal, _madness_decline_execute))
    # "Target: yourself" is always legal the instant this pending kind is reached
    # (a real Magic legality fact -- "target player" never excludes its own
    # caster), even alone in a 1-player game; "Target: opponent" only becomes
    # legal once a real second PlayerState exists. Two fixed actions, not a
    # per-name loop -- there are only ever at most 2 possible players.
    actions.append(("Target: yourself", _target_self_legal, _target_self_execute))
    actions.append(("Target: opponent", _target_opponent_legal, _target_opponent_execute))
    # The player half of an "any target" choice (Lightning Bolt etc.) -- same
    # shape/reasoning as the choose_target_player pair above. The creature half
    # (either battlefield) rides the identity pointer scheme (rl.action_bridge),
    # not fixed actions. Universal because a Chain Lightning COPIER -- the
    # affected player, i.e. the opponent -- may choose a new target for the copy.
    actions.append(("Target any: yourself", _target_any_self_legal, _target_any_self_execute))
    actions.append(("Target any: opponent", _target_any_opponent_legal, _target_any_opponent_execute))
    actions.append(("Choose no target", _target_any_decline_legal, _target_any_decline_execute))  # "up to one" decline
    # Undercity venture (which next room -- a <=2-way branch): a tiny CONSTANT
    # set (ROOM_NAMES), not a per-deck-card loop, so it's universal -- initiative
    # can pass between players (combat damage), so a deck with no Undercity card
    # of its own can still be the one venturing.
    for room in game.ROOM_NAMES:
        actions.append((f"Enter room: {room}", _choose_room_legal(room), _choose_room_execute(room)))
    # Chromatic Star's "add one mana of any color": self-only (the activating
    # player's own choice, single call site) -- see this block's own header
    # comment. Gated, unlike the room set above.
    if "choose_mana_color" in own_pending_kinds:
        for color in game.COLORS:
            actions.append((f"Add mana: {color}", _choose_mana_color_legal(color), _choose_mana_color_execute(color)))

    # choose_cast_mode/choose_cast_x/choose_delve_amount: shared, per-deck-
    # conditional button sets for the generic mode/X/delve-amount sub-
    # decisions the cast_modes/x_cast_modes/delve loops above open -- unlike
    # the universal rows just above, these are never posed by an OPPONENT's
    # card (they only ever fire mid this deck's own casting), so they're
    # gated on this deck actually having a card of that shape, same
    # reasoning as the deck-gated impulse/discard_or_sacrifice rows.
    if needs_cast_mode_buttons:
        for i in range(_CAST_MODE_BUTTON_MAX):
            actions.append((f"Mode {i + 1}", _choose_cast_mode_legal(i), _choose_cast_mode_execute(i)))
    if needs_cast_x_buttons:
        for x in range(_CAST_X_BUTTON_MAX + 1):
            actions.append((f"X={x}", _choose_cast_x_legal(x), _choose_cast_x_execute(x)))
    if needs_delve_buttons:
        for n in range(_DELVE_BUTTON_MAX + 1):
            actions.append((f"Delve {n}", _choose_delve_amount_legal(n), _choose_delve_amount_execute(n)))

    # Shared "Produce <color>" buttons for the choose_color stage of a
    # mana_subdecision -- reused by BOTH Saruli-shaped mana_extra_choose
    # sources and filter_mana sources (whichever one is present sets
    # needs_mana_subdecision_color_buttons above); never posed by an
    # opponent's card, only ever opened by this deck's own source.
    if needs_mana_subdecision_color_buttons:
        for color in game.COLORS:
            actions.append((
                f"Produce {color}", _mana_subdecision_color_legal(color), _mana_subdecision_color_execute(color),
            ))

    return tuple(actions)


def _validate_choose_name_coverage(state, actions):
    """The durable, always-on half of the by-name/cross-player guardrail
    (see _CHOOSE_NAME_PENDING_KINDS's own docstring for the invariant this
    enforces). Runs on every legal_action_mask sweep, but is a cheap no-op
    unless the current pending is actually one of that constant's kinds --
    which is already a small minority of decisions.

    Cross-checks _choose_name_options' own OUTPUT against what this deck's
    table can actually represent (every "Choose: X" row already built into
    `actions`, read directly off it -- no separate choosable_names threading
    needed). If a candidate has no matching row, that candidate is
    UNREACHABLE: not merely "illegal right now" (any legal_fn can say that),
    but a card the agent has NO WAY to ever select for this decision, which
    means either every candidate is unreachable (an eventual all-False
    crash, the failure mode that surfaced this bug) or only SOME are (a
    silent, still-live correctness bug -- the agent quietly loses the
    ability to choose a specific legal option, with no crash at all to flag
    it). Raising immediately, the first time ANY game -- a self-check,
    random self-play, or real training -- exercises the violation, is what
    makes this a property of the code instead of a fact someone has to
    remember: no comment to read, no separate audit script to run and act
    on. Confirmed reachable in real cross-deck league play: choose_
    stack_target asked mono_blue_terror to name a spell dmir_terror cast,
    which mono_blue_terror's own table (built only from its own decklist)
    had no row for."""
    pending = state.pending_resolution
    if pending is None or pending["kind"] not in _CHOOSE_NAME_PENDING_KINDS:
        return
    candidates = _choose_name_options(state)
    if not candidates:
        return
    table_names = {name[len("Choose: "):] for name, _legal, _execute in actions if name.startswith("Choose: ")}
    missing = set(candidates) - table_names
    if missing:
        raise RuntimeError(
            f"_choose_name_options returned {sorted(missing)} for pending kind {pending['kind']!r}, but this "
            f"deck's own action table has no 'Choose: X' row for {'it' if len(missing) == 1 else 'them'}. "
            f"A by-name pending kind can only ever be answered from names in the DECIDING PLAYER's own deck "
            f"(build_action_table's choosable_names) -- {pending['kind']!r} can apparently produce a candidate "
            f"outside that (most likely another player's card), which means it must be POINTER-addressed "
            f"(see rl.action_bridge; choose_graveyard_card and choose_stack_target are the established pattern), "
            f"not left in _CHOOSE_NAME_PENDING_KINDS."
        )


_mana_subdecision_rows_cache = {}  # id(actions) -> (actions, [(idx, mana_subdecision_gate), ...] for entries stamped with one)
_MANA_SUBDECISION_ROWS_CACHE_CAP = 32  # ponytail: FIFO-evicted, not reset per sweep like its siblings (this cache must survive across sweeps -- see below) -- bounds memory when a caller (e.g. a persistent multiprocessing worker) rebuilds fresh `actions` tables across many calls instead of reusing one per deck for the session


def _mana_subdecision_rows(actions):
    """The (idx, _mana_subdecision_gate) pairs for entries carrying that
    stamp, cached once per distinct `actions` table (see legal_action_mask's
    own docstring for the id()-cache lifecycle reasoning -- identical to
    the removed _action_gates' -- this is the ONE gate still worth caching:
    _mana_subdecision_color_legal is unsafe to call unconditionally (see
    its own body -- it dereferences state.mana_subdecision["can_produce"]
    with no None-guard, so calling it with no subdecision open raises
    TypeError, not a graceful False), so legal_action_mask MUST know which
    few indices those are without calling them first to find out. Empty for
    most decks -- populated only by a Saruli-Caretaker-shaped
    mana_extra_choose card or a filter_mana card (Conduit Pylons/Barrels of
    Blasting Jelly)."""
    cached = _mana_subdecision_rows_cache.get(id(actions))
    if cached is not None and cached[0] is actions:
        return cached[1]
    rows = [(idx, legal_fn._mana_subdecision_gate) for idx, (_name, legal_fn, _execute) in enumerate(actions)
            if getattr(legal_fn, "_mana_subdecision_gate", None) is not None]
    if len(_mana_subdecision_rows_cache) >= _MANA_SUBDECISION_ROWS_CACHE_CAP:
        _mana_subdecision_rows_cache.pop(next(iter(_mana_subdecision_rows_cache)))
    _mana_subdecision_rows_cache[id(actions)] = (actions, rows)
    return rows


def legal_action_mask(state, actions):
    """Stateless -- takes the action table explicitly, so any caller (the
    token pipeline's own _seat_step, a direct game-loop driver, ...)
    can use it. `actions` is any table built by build_action_table -- every
    deck's own table, none privileged as a default (a caller with its own
    decklist always has its own table to pass).

    Calls every closure directly -- no pending_resolution-based category
    gating. A gating layer (skip a closure via a cheap pre-check on
    state.pending_resolution["kind"], mirroring each closure's own first-
    line check -- see every `._pending_gate =` stamp still sitting on the
    closures below) was tried, INCLUDING a per-table cache of the gate
    lookups (avoiding a fresh getattr every sweep), and measured SLOWER
    than calling everything unconditionally -- consistently, ~15-25%,
    across every pending kind that occurs in real play (controlled, order-
    randomized timing over 4000 real captured decisions, comparing on the
    SAME states in interleaved repeats to rule out cache-warmup bias).
    Most `_X_legal` closures are already cheap enough (one or two
    attribute/dict checks) that the overhead of DECIDING whether to call
    them costs as much or more than just calling them. The `._pending_gate`
    stamps are left in place as documentation of each closure's real
    legality precondition (and in case a more selective future approach --
    gating only the handful of genuinely expensive closures instead of all
    ~300 -- turns out to be worth it); this sweep no longer reads them.

    state.mana_subdecision is NOT the same kind of optional speed layer,
    and is NOT dropped: it's the ONLY thing that makes every OTHER action
    illegal while a gate-free mana ability's own multi-step choice (Saruli
    Caretaker: tap another creature, then choose a color) is open -- no
    individual `_X_legal` closure checks state.mana_subdecision itself
    except `_mana_subdecision_color_legal` (gated to its own "choose_color"
    stage, and unsafe to call at all when no subdecision is open -- see
    _mana_subdecision_rows' own docstring). Skipping this check would
    silently make every other action look legal mid-subdecision (or crash
    on the "Produce <color>" rows specifically); see
    test_saruli_caretaker_two_stage_mana_subdecision for the regression
    coverage. Three cases, cheapest first:
      1. This deck's table has no mana_subdecision-gated row at all (most
         decks) -- every closure is safe to call unconditionally, so this
         degenerates to exactly the same zero-overhead sweep as case 1
         above, no per-entry check of any kind.
      2. The table has such a row, but none is open right now -- everything
         else is still safe to call directly; only the (few, known-by-index)
         subdecision rows are forced False without being called.
      3. A subdecision is genuinely open -- exclusive priority, suppress
         everything except the matching-stage subdecision closures, mirroring
         how activating a mana ability is atomic from everyone else's view
         in real Magic. See state.mana_subdecision's own docstring for why
         this can't just be another pending_resolution kind (a single slot,
         not a stack -- a second pending would clobber whatever's already
         open).

    Resets _actions_combat._battlefield_lookup_cache and _actions_mana's own
    _mana_ability_options_cache/_mana_source_cache/_filter_source_cache, and
    game.mana's own _enchanting_cache (game.reset_mana_cache), before AND
    after the sweep itself (not just before): guarantees none of these
    caches can ever leak past this call's own scope into a later execute_fn
    call or an unrelated sweep against a different/mutated state, even
    though nothing in the current single-threaded, synchronous call pattern
    would actually trigger that -- belt-and-suspenders for a module-level
    global, not load-bearing. mana.py's own cache is reset from here, not
    self-invalidating there, for the same reason the others aren't: see
    game.mana._enchanting's own docstring.

    Mask is built as a plain list and converted to a numpy array ONCE at
    the end -- indexed numpy writes in a tight Python loop carry real
    per-call overhead, the same lesson rl.arch.pad_token_batch's own
    docstring already applies to token tensors."""
    _actions_combat._battlefield_lookup_cache = None
    _actions_mana._mana_ability_options_cache = None
    _actions_mana._mana_source_cache = None
    _actions_mana._filter_source_cache = None
    game.reset_mana_cache()
    mana_sub = state.active_mana_subdecision
    sub_rows = _mana_subdecision_rows(actions)
    try:
        if not sub_rows:
            mask = [legal_fn(state) for _name, legal_fn, _execute in actions]
        elif mana_sub is None:
            sub_indices = {idx for idx, _gate in sub_rows}
            mask = [False if idx in sub_indices else legal_fn(state)
                    for idx, (_name, legal_fn, _execute) in enumerate(actions)]
        else:
            mana_sub_stage = mana_sub["stage"]
            mask = [False] * len(actions)
            for idx, gate in sub_rows:
                if gate == mana_sub_stage:
                    mask[idx] = actions[idx][1](state)
        _validate_choose_name_coverage(state, actions)
        return np.asarray(mask, dtype=bool)
    finally:
        _actions_combat._battlefield_lookup_cache = None
        _actions_mana._mana_ability_options_cache = None
        _actions_mana._mana_source_cache = None
        _actions_mana._filter_source_cache = None
        game.reset_mana_cache()


__all__ = [
    'build_action_table',
    'legal_action_mask',
]
