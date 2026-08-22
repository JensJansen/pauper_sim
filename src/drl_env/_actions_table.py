"""build_action_table (assembles the flat fixed action table from a
decklist + game.EFFECT_REGISTRY) and legal_action_mask (+ its sweep-scoped
coverage check) -- the two entry points every other _actions_* category
module feeds into. Kept together since the table-builder touches every
category and the mask sweep resets every category's own sweep-scoped
cache.

Categories, in table order (see _actions_cast/_actions_cast_altzone/
_combat/_resolution/_mana for the category boundaries themselves):
  A. Play land: <name>            -- one per distinct land name
  A2. Tap <name> [for <color>]    -- mana abilities: one no-color row per
     fixed/Tron/count source, one row per producible color for a
     flexible/granted source. Legal in any priority window, even
     mid-resolution of anything else (605.1a/605.3b), resolves
     immediately, never the stack. "Tap <name>" (Saruli Caretaker): same
     gate-free category, but its extra cost (tap another creature, a cost
     choice, 602.5g) opens a mana_subdecision instead of resolving in one
     shot. "Filter <name>, paying <input_color>" (Conduit Pylons/Barrels):
     pays the {1} immediately, then opens the same choose_color
     mana_subdecision stage for the output color.
  B. Cast <name>                  -- one per card with a registry "cast" entry
  C. Activate <name> (<ability>)  -- one per registered activated ability
  D. Forestcycle <name>           -- one per registry "forestcycle" entry
  E. Pass
  F. Choose: <name>               -- shared across every pending-resolution
     kind that picks a plain card name (search_fetch, ancient_stirrings,
     discard, and scry/surveil's ordering phase), dispatched by
     pending_resolution["kind"]. Paying a cost never appears here: once a
     cost is announced, the only legal actions are the A2 mana abilities
     and "Spend <color> from pool" below.
     Not sacrifice (see category K) -- a battlefield permanent is never
     just a name, unlike a hand/library/graveyard card.
  H. Keep / Dispose (scry/surveil)
  I. Decline (Ancient Stirrings)
  K. Choose target: <name> (slot k) -- exact-(name, slot)-addressed, the
     "choose_permanent" resolution's own actions (Aura enchant-targets,
     Crop Rotation's sacrifice cost, land bounce, and every generic
     sacrifice cost). Not category F: two same-named permanents stop
     being interchangeable once one is enchanted/tapped/has a counter,
     and cast_aura's fizzle-on-invalid-target contract depends on knowing
     exactly which physical permanent was chosen.

Per-deck additions: B also covers modal casts, free alt-costs, Flashback,
and x_cast_modes (one action per (mode, X) pair, each its own cost
distinct from card_def.cast_cost); C also covers non-mana activated
abilities; F/H also cover select_to_hand's Keep/Bottom pair, its ordering
phase, and an optional search's Decline; K also covers opponent-facing ETB
targets (choose_opponent_permanent, same category as blocking's
cross-player targeting)."""

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
    """opponent_decklist/opponent_token_card_defs: the other side's own
    decklist/tokens -- None/() for every 1-player deck, since there's no
    real opponent battlefield to reference. The token/pointer pipeline
    passes None (cross-player targeting moved to the pointer head); kept
    for a fixed "Choose opponent's: X" table if one is ever wanted again.

    token_card_defs: every token CardDef this deck's cards can create at
    runtime (Blood, Robot, Warrior, Eldrazi Spawn, ...). Tokens are never
    decklist entries, so they can't flow through distinct_names/
    game.CARD_DEFS[name] the way every other action here does. Feeds both
    the activated-abilities loop below (a token's own ability needs an
    action to exist) and the choosable_names set (a token can be a legal
    choose_permanent/sacrifice/discard choice despite never appearing in
    the decklist).

    No `pending_kinds` parameter: whether a resolution kind can ever cross
    players is a fixed engine fact, not something a caller computes --
    see own_pending_kinds below and the "UNIVERSAL DECISION ROWS" block."""
    distinct_names = sorted({name for name, *_rest in decklist})
    # This decklist's own derived kinds. Used below for kinds confirmed self-only
    # (impulse, discard_or_sacrifice, may_transform, may_cast, madness_decision,
    # ancient_stirrings, malevolent_rumble, select_to_hand, choose_mana_color),
    # so those rows don't inflate a deck's table for a card it doesn't run.
    # Confirmed-cross-player kinds (pay_unless, tuck_position, may_copy,
    # choose_any_target, choose_room, choose_target_player, scry/search_fetch's
    # Decline, discard/graveyard-decline) don't consult this at all -- they're
    # unconditional in the "UNIVERSAL DECISION ROWS" block further down.
    own_pending_kinds = game.derive_pending_kinds(decklist)
    land_names = sorted({
        name for name in distinct_names if game.CARD_DEFS[name].card_type == game.CardType.LAND
    })
    # Hoisted here so the extra-cost mana-ability loop can also use them:
    # card_type_by_name/qty_by_name index this side's own battlefield
    # permanents; choosable_names is every name (real or token) this side
    # could ever have on its own battlefield.
    card_type_by_name = {name: game.CARD_DEFS[name].card_type for name in distinct_names}
    card_type_by_name.update({cd.name: cd.card_type for cd in token_card_defs})
    qty_by_name = {name: qty for name, qty, *_rest in decklist}
    # DFC back faces (Delver of Secrets -> Insectile Aberration): a transformed
    # permanent's card_def swaps to its back face, so any by-name resolution
    # (order_triggers, sacrifice, discard, search_fetch, ...) can need to
    # reference that back-face name too. Discovered the same way
    # rl.model.features.CardVocab discovers back faces: scan each front face's
    # own registry "transform" spec.
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

    # Mana abilities: "Tap X [for <color>]" produces mana into the pool in
    # any priority window, even mid-resolution of anything else (605.1a/
    # 605.3b, no pending-resolution gate). Row count is derived per source:
    # the registry's "mana" spec kind statically says which colors that
    # source's native ability could ever offer, so a row neither can
    # produce is never generated. "Tap X for C" never appears: colorless is
    # only ever produced via the no-color row, never a color choice.
    #
    # Lands are the one case a per-source static lookup alone would miss:
    # Abundant Growth's cast_aura targets any land, either side, and grants
    # a color on top of the land's own native ability -- a runtime fact no
    # single card's "mana" spec captures. So every land-type source keeps a
    # row for every color the full registry ever declares grantable
    # ("grants_mana_colors"), unioned with its own native colors.
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

    # Saruli Caretaker-shaped mana abilities: an extra cost that taps another
    # untapped creature (mana_extra_choose) -- a cost choice (602.5g), not a
    # target. One gate-free "Tap <name>" row per source name; its execute
    # opens a mana_subdecision to choose which creature to tap (pointer-
    # routed), then a color (the shared "Produce <color>" buttons below),
    # matching real sequencing (cost paid before the color-choice effect
    # resolves). needs_mana_subdecision_color_buttons is shared with the
    # mana-filter block below, since both mechanics reach the same
    # choose_color stage.
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
    # (source, input_color), paid immediately. input_color ranges over every
    # pool color. The output half is a separate, later choice via the shared
    # choose_color mana_subdecision stage above, reusing the "Produce <color>"
    # buttons. Never a nested pay_cost for the {1} itself, which would risk
    # clobbering whatever pending_resolution is already open.
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
    # buttons below are only added to a deck's table that actually has a card
    # using that shape -- these are the deck's own casting decisions, never
    # posed by an opponent's card.
    needs_cast_mode_buttons = False
    needs_cast_x_buttons = False
    needs_delve_buttons = False

    for name in distinct_names:
        card_spec = registry.get(game.CARD_DEFS[name].effect_id, {})
        cast_spec = card_spec.get("cast")
        if cast_spec is not None:
            # "precast_choice": True (Auras' real targets, Crop Rotation's
            # sacrifice-as-a-cost) -- resolve runs immediately once paid and
            # manages its own push_to_stack, instead of _cast_execute's
            # generic auto-push.
            cast_execute_fn = _precast_choice_execute if cast_spec.get("precast_choice") else _cast_execute
            actions.append((
                f"Cast {name}",
                _cast_legal(name, cast_spec.get("extra_legal"), _cast_speed(game.CARD_DEFS[name], cast_spec)),
                cast_execute_fn(name, cast_spec["resolve"]),
            ))
        # A modal cast: one "Cast <name>" row (not one per mode); which mode is
        # a generic choose_cast_mode sub-decision (601.2b: chosen before cost
        # is calculated), shared "Mode 1".."Mode 5" buttons.
        cast_modes = card_spec.get("cast_modes")
        if cast_modes is not None:
            mode_items = list(cast_modes.items())
            speed = _cast_speed(game.CARD_DEFS[name], mode_items[0][1])
            actions.append((f"Cast {name}", _modal_legal(name, mode_items, speed), _modal_execute(name, mode_items)))
            needs_cast_mode_buttons = True
        # X-cost modes (e.g. a normal creature cast and Bestow, each with a
        # different base cost): one "Cast <name>" row; mode chosen first
        # (601.2b), X chosen second (601.2f), both generic sub-decisions
        # (shared "Mode n" then "X=n" buttons). Each mode's "resolve" is a
        # function of x, called once X is actually chosen (_x_modal_execute).
        x_cast_modes = card_spec.get("x_cast_modes")
        if x_cast_modes is not None:
            mode_items = list(x_cast_modes.items())
            speed = _cast_speed(game.CARD_DEFS[name], mode_items[0][1])
            actions.append((f"Cast {name}", _x_modal_legal(name, mode_items, speed), _x_modal_execute(name, mode_items)))
            needs_cast_mode_buttons = True
            needs_cast_x_buttons = True
        # Delve: one "Cast <name>" row; the delve amount is a generic
        # choose_delve_amount sub-decision (702.66: chosen before the exile
        # sub-cost opens and the reduced cost is calculated), shared
        # "Delve n" buttons.
        delve = card_spec.get("delve")
        if delve is not None:
            delve_speed = _cast_speed(game.CARD_DEFS[name], delve)
            actions.append((
                f"Cast {name}",
                _delve_legal(name, delve["max"], delve_speed),
                _delve_execute(name, delve["max"], delve["resolve"]),
            ))
            needs_delve_buttons = True
        # A second, free cast path alongside the normal one (e.g. Land Grant).
        alt_cast = card_spec.get("alt_cast")
        if alt_cast is not None:
            actions.append((
                f"Cast {name} (free)",
                _alt_cast_legal(name, alt_cast["extra_legal"], _cast_speed(game.CARD_DEFS[name], alt_cast)),
                _alt_cast_execute(name, alt_cast["resolve"]),
            ))
        # Flashback casts from the graveyard, not hand. Escape is the same
        # graveyard-cast machinery with a mana cost plus its own additional
        # cost handled inside resolve -- shares _flashback_legal/_execute,
        # only the action label differs.
        for gy_cast_key, gy_cast_label in (("flashback", "Flashback"), ("escape", "Escape")):
            gy_cast = card_spec.get(gy_cast_key)
            if gy_cast is not None:
                gc_cost = gy_cast.get("cost")  # mana cost dict, if the cost includes mana
                actions.append((
                    f"{gy_cast_label} {name}",
                    _flashback_legal(name, gy_cast["legal"], _cast_speed(game.CARD_DEFS[name], gy_cast), gc_cost),
                    _flashback_execute(name, gy_cast["resolve"], gc_cost),
                ))
        # Plot: pay its plot cost to exile it now, cast it for free from
        # exile on any later turn. The cast-from-exile half reuses this
        # card's normal "cast" resolve, so a "plot" entry only makes sense
        # alongside a "cast" entry, never alone.
        plot = card_spec.get("plot")
        if plot is not None:
            plot_speed = _cast_speed(game.CARD_DEFS[name], plot)  # same speed governs both actions below
            actions.append((
                f"Plot {name}",
                _plot_legal(name, plot["cost"], plot_speed),
                _plot_execute(name, plot["cost"], plot["resolve"]),
            ))
            # cast_from_exile_resolve: optional override for a normal "cast"
            # resolve that does state.hand.remove(card_def) -- wrong once the
            # card already left exile, never hand. Falls back to
            # cast_spec["resolve"] for a card whose resolve doesn't care.
            actions.append((
                f"Cast {name} (plotted)",
                _cast_from_exile_legal(name, cast_spec.get("extra_legal"), plot_speed),
                _cast_from_exile_execute(name, plot.get("cast_from_exile_resolve", cast_spec["resolve"])),
            ))
        # Omen: the sorcery side's resolve shuffles the card into the library
        # instead of graveyarding or exiling it (see _omen_cast_legal). This
        # is a second cast option for the same hand card, its own cost,
        # offered whenever a same-named card is back in hand -- the creature
        # side is a wholly different spell, never sharing a resolve.
        omen = card_spec.get("omen")
        if omen is not None:
            omen_speed = _cast_speed(omen["card_def"], omen)
            actions.append((
                f"Cast {omen['card_def'].name} (omen)",
                _omen_cast_legal(name, omen["cost"], omen_speed),
                _omen_cast_execute(omen["card_def"], omen["cost"], omen["resolve"]),
            ))
        # Prototype: a second cast option for the same hand card, its own
        # cheaper cost, producing a different CardDef. Structurally identical
        # to Omen, so it reuses the same _omen_cast_legal/_omen_cast_execute
        # helpers -- only the resolve differs (no library shuffle).
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
            # Default for activated (non-mana) abilities is any time, unless
            # the card says "activate only as a sorcery" -- an explicit
            # "speed" key in the ability's spec is that opt-in override.
            speed = spec.get("speed", game.turn.Speed.INSTANT)
            if "cost_key" in spec:
                actions.append((
                    f"Activate {name} ({ability_name})",
                    _activate_legal(name, spec["cost_key"], speed, spec.get("extra_legal")),
                    _activate_execute(name, spec["cost_key"], spec["resolve"]),
                ))
            else:
                # Non-mana cost (e.g. Quirion Ranger: return a Forest to hand).
                actions.append((
                    f"Activate {name} ({ability_name})",
                    _activate_no_cost_legal(name, spec["legal"], speed),
                    _activate_no_cost_execute(name, spec["resolve"]),
                ))

    # "Discard this card from hand: <do something>" cycling-family actions.
    # Both keys share identical hand-zone/cost-key/resolve plumbing
    # (_forestcycle_legal/_execute), differing only in the action label:
    #   "forestcycle" -- basic-land-to-hand search
    #   "cycle"       -- plain Cycling and typed cycling (e.g. Islandcycling)
    for name in distinct_names:
        for spec_key, label in (("forestcycle", "Forestcycle"), ("cycle", "Cycle")):
            cyc_spec = registry.get(game.CARD_DEFS[name].effect_id, {}).get(spec_key)
            if cyc_spec is not None:
                actions.append((
                    f"{label} {name}",
                    _forestcycle_legal(name, cyc_spec["cost_key"]),
                    _forestcycle_execute(name, cyc_spec["cost_key"], cyc_spec["resolve"]),
                ))

    # Impulse: "you may play the exiled cards". Only emitted for a deck that
    # can actually impulse (its source card declares pending_kinds
    # {"impulse"}), so decks without one never carry these mostly-illegal
    # actions. One action per deck card name: a land play, or a cast per the
    # card's own cast/cast_modes spec, paying its normal cost (impulse,
    # unlike Plot, is not free).
    #
    # Gated on this decklist's own derived kinds, unlike pay_unless/
    # tuck_position: impulse is never cross-player, since state.impulse is
    # always the active player's own zone and every impulse source only ever
    # exiles from its own controller's library.
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

    # An activated ability usable from the graveyard, not the battlefield
    # (unlike "activated_abilities" above) or hand (unlike Forestcycle) --
    # the "graveyard_ability" registry key.
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
    # etc. resolution could ever offer, not just decklist names (not
    # choose_permanent -- that's the "Choose target: X (slot k)" block below,
    # exact-(name, slot) addressed). A token is a perfectly legal sacrifice/
    # discard choice despite never appearing in CARD_DEFS/the decklist.
    # extra_choosable_names: card names that can be a "Choose: X" option
    # despite not being in this deck (nor its tokens), e.g. an opponent's
    # graveyard cards. The league passes none: choose_graveyard_card is a
    # pointer target (rl.decision.action_bridge). Kept as a general knob for
    # scripts/migrate_pointer_graveyard.py; defaults to none. choosable_names
    # itself also drives "Choose target: X (slot k)" and "Attack: X (slot k)"
    # below, both strictly this side's own battlefield permanents.
    choose_by_name = sorted(set(choosable_names) | set(extra_choosable_names))
    for name in choose_by_name:
        actions.append((f"Choose: {name}", _choose_name_legal(name), _choose_name_execute(name)))

    # "Attack: X (slot k)" -- one per (creature name, slot) pair, legal only
    # during Phase.DECLARE_ATTACKERS. k runs 1..decklist quantity for a real
    # card, since only that many physical copies can ever be simultaneously
    # alive. A token has no decklist quantity, so k runs 1..TOKEN_LIMIT, a
    # shared pool across every token name.

    # "Choose target: X (slot k)" -- the "choose_permanent" resolution's own
    # exact-(name, slot) addressed actions (Aura enchant-targets, Crop
    # Rotation's sacrifice cost, land bounce), scoped to this side's own
    # battlefield. Registered for every choosable name, not just creatures
    # (e.g. Utopia Sprawl targets lands), with legal() gating precisely at
    # runtime against the actual pending choose_permanent resolution.
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
    # Trample-to-player is no longer an agent choice -- automatic now (see
    # _assign_damage_to_opponent_legal). Kept registered, permanently
    # illegal, so the fixed table's length stays stable.
    actions.append((
        "Assign combat damage to opponent", _assign_damage_to_opponent_legal, _assign_damage_to_opponent_execute,
    ))

    # "Choose opponent's: X (slot k)" -- the general cross-player targeting
    # primitive, one per (opponent name, slot), built from the opponent's
    # own decklist/tokens. Registered for every opponent choosable name, not
    # just creatures (e.g. Masked Vandal's ETB targets an opponent artifact
    # or enchantment). Same quantity-or-TOKEN_LIMIT bound as the attack
    # registration above, applied to the other side's card pool. None/()
    # (the default for every 1-player deck) registers nothing at all.
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
    # chosen at tap time (the "Tap X for <color>" rows above float directly
    # into the pool). Spending what is floating is the separate row set below.
    for color in game.POOL_COLORS:
        actions.append((
            f"Spend {color} from pool",
            _pool_spend_legal(color),
            _pool_spend_execute(color),
        ))

    # ---- UNIVERSAL DECISION ROWS (deliberately NOT gated on any pending kind) ----
    # Every fixed row below answers a question the engine can pose, each a
    # small constant set of buttons, never a per-card-name loop. Added to
    # every deck's table unconditionally, because an opponent's card can pose
    # a question that you must answer:
    #   pay_unless          the payer is the spell's controller, i.e. the opponent.
    #   tuck_position       the tucked permanent's owner chooses.
    #   may_copy            the affected player may copy.
    #   choose_any_target   a copier may choose a new target.
    #   choose_target_player/scry/search_fetch  initiative can be stolen via
    #                       combat damage, so a deck can face these without
    #                       running any card of that kind itself.
    #   discard-decline, choose_graveyard_card-decline  same shape (some
    #                       cards target the opponent).
    # `derive_pending_kinds` reads a deck's own cards, so gating these on it
    # left the answering seat with no row for the question and an all-False
    # action mask. Rather than maintain a hand-audited list of "which kinds
    # may cross" (which silently rots the next time a card hands a choice to
    # the opponent), every decision-button set whose kind could ever cross
    # players is always present and runtime-gated illegal instead.
    #
    # Gated below on own_pending_kinds instead (confirmed self-only: each has
    # exactly one call site, always the deciding player's own card):
    # may_transform, may_cast, choose_mana_color, ancient_stirrings/
    # malevolent_rumble/select_to_hand, madness_decision (always triggers off
    # your own discarded card), and discard_or_sacrifice (the caster's own
    # optional cost, never posed to an opponent, same as `impulse` above).
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
    # this table. The per-deck MulliganNet (rl.model.mulligan) owns every pregame
    # decision -- rl.decision.agent.SeatAgent intercepts the pregame pending kinds before
    # the main net's forward -- so the main policy's action space contains ZERO
    # pregame actions and a game can never fall back to a fixed-table mulligan.
    # The engine's own mulligan handlers (game.execute_mulligan_keep/
    # execute_mulligan_take) are called directly by rl.decision.agent, not through this
    # table.
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
    # (either battlefield) rides the identity pointer scheme (rl.decision.action_bridge),
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
    # "Add one mana of any color": self-only (single call site). Gated,
    # unlike the room set above.
    if "choose_mana_color" in own_pending_kinds:
        for color in game.COLORS:
            actions.append((f"Add mana: {color}", _choose_mana_color_legal(color), _choose_mana_color_execute(color)))

    # choose_cast_mode/choose_cast_x/choose_delve_amount: shared, per-deck-
    # conditional button sets for the mode/X/delve-amount sub-decisions the
    # cast_modes/x_cast_modes/delve loops above open. Never posed by an
    # opponent's card, so gated on this deck actually having a card of that
    # shape.
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
    # mana_subdecision -- reused by both mana_extra_choose and filter_mana
    # sources; never posed by an opponent's card.
    if needs_mana_subdecision_color_buttons:
        for color in game.COLORS:
            actions.append((
                f"Produce {color}", _mana_subdecision_color_legal(color), _mana_subdecision_color_execute(color),
            ))

    return tuple(actions)


def _validate_choose_name_coverage(state, actions):
    """Enforces the invariant from _CHOOSE_NAME_PENDING_KINDS. Runs on every
    legal_action_mask sweep, a cheap no-op unless the current pending is
    actually one of that constant's kinds.

    Cross-checks _choose_name_options' output against every "Choose: X" row
    already built into `actions`. A candidate with no matching row is
    unreachable -- not just illegal right now, but a card the agent has no
    way to ever select for this decision, either crashing all-False or
    silently losing the ability to choose a legal option. Raises
    immediately so this is a property of the code, not a fact to remember."""
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
            f"(see rl.decision.action_bridge; choose_graveyard_card and choose_stack_target are the established pattern), "
            f"not left in _CHOOSE_NAME_PENDING_KINDS."
        )


_mana_subdecision_rows_cache = {}  # id(actions) -> (actions, [(idx, mana_subdecision_gate), ...])
_MANA_SUBDECISION_ROWS_CACHE_CAP = 32  # ponytail: FIFO-evicted; bounds memory across many rebuilt `actions` tables


def _mana_subdecision_rows(actions):
    """The (idx, _mana_subdecision_gate) pairs for entries carrying that
    stamp, cached once per distinct `actions` table. The one gate worth
    caching: _mana_subdecision_color_legal is unsafe to call unconditionally
    (it dereferences state.mana_subdecision["can_produce"] with no
    None-guard), so legal_action_mask must know which indices those are
    without calling them first. Empty for most decks."""
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
    """Stateless -- takes the action table explicitly, so any caller can use
    it. `actions` is any table built by build_action_table.

    Calls every closure directly, with no pending_resolution-based category
    gating: a gating pre-check on state.pending_resolution["kind"] (even
    cached) measured ~15-25% slower than calling everything unconditionally,
    since most `_X_legal` closures are already cheap enough that deciding
    whether to call them costs as much as calling them. The `._pending_gate`
    stamps are left on the closures as documentation only; this sweep
    doesn't read them.

    state.mana_subdecision IS still checked: it's the only thing that makes
    every other action illegal while a gate-free mana ability's own
    multi-step choice is open. No individual `_X_legal` closure checks it
    itself except `_mana_subdecision_color_legal`, which is unsafe to call
    at all when no subdecision is open. Three cases, cheapest first:
      1. This deck's table has no mana_subdecision-gated row (most decks) --
         every closure is safe to call unconditionally.
      2. The table has such a row, but none is open right now -- everything
         else still safe to call directly; only the (few, known-by-index)
         subdecision rows are forced False without being called.
      3. A subdecision is genuinely open -- exclusive priority, suppress
         everything except the matching-stage subdecision closures, mirroring
         how activating a mana ability is atomic in real Magic. Can't just be
         another pending_resolution kind since that's a single slot, not a
         stack, and a second pending would clobber whatever's already open.

    Resets _actions_combat._battlefield_lookup_cache, _actions_mana's own
    sweep-scoped caches, and game.mana's cache before and after the sweep,
    so none can leak past this call's scope into a later execute_fn call or
    an unrelated sweep.

    Mask is built as a plain list and converted to a numpy array once at
    the end, since indexed numpy writes in a tight loop carry real overhead."""
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
