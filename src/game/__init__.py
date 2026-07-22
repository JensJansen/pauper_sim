"""Multi-deck Magic-subset simulator -- package entry point.

This package replaces what used to be a single game.py file, split by
domain (cards, state, resolution, mana, effects, turn loop, reporting,
per-color card catalogs, and the registry that unions every color). A
deck is just a decklist file (data/*.txt, parsed by game.decklist)
resolved against the shared card catalog (game.CARD_DEFS) each color
catalog contributes to -- adding or reweighting a deck built entirely
from already-implemented cards never needs a code change, and neither
does reusing a card across multiple decks (DECK_REGISTRY_REFRESH_PLAN.md).

Every submodule is re-exported here flat (game.CARD_DEFS, game.GameState,
game.play_land_from_hand, ...) so every existing `import game; game.X`
caller (drl_env.py, rewards.py, harness.py, run.py,
generate_regression_snapshot.py) keeps working unchanged.

Import order matters: `from . import registry` first is what actually
triggers loading every catalog module (and, transitively, most of the
game.effects package / mana / resolution / state / cards) -- see
game/registry.py's own module docstring for why several of those modules
only reference `registry.EFFECT_REGISTRY` lazily instead of importing the
name directly, which is what makes this order-independent rather than a
real problem.
"""

import random  # noqa: F401 -- re-exported so `game.random.Random(seed)` keeps working

from . import registry
from .cards import CardDef, CardType, EffectId
from .catalog.black_cards import (
    begin_choose_graveyard_card,
    cast_dread_return,
    choose_graveyard_card_options,
    execute_choose_graveyard_card_option,
    flashback_dread_return,
    lotleth_giant_etb,
    mill_until_land,
)
from .catalog.colorless_cards import (
    activate_bonders_ornament_draw,
    activate_candy_trail_sac,
    activate_expedition_map,
    activate_relic_of_progenitus,
    activate_tocasia_dig_site_surveil,
)
from .catalog.green_cards import (
    ancient_stirrings_options,
    begin_ancient_stirrings,
    begin_select_to_hand,
    cast_ancient_stirrings,
    cast_crop_rotation,
    cast_land_grant,
    cast_lead_the_stampede,
    cast_roost_seek,
    cast_winding_way_creature,
    cast_winding_way_land,
    execute_ancient_stirrings_option,
    execute_malevolent_rumble_option,
    execute_select_to_hand_option,
    gatecreeper_vine_etb,
    is_noncreature_colorless,
    land_grant_alt_cost_legal,
    malevolent_rumble_options,
    quirion_ranger_untap_legal,
    quirion_ranger_untap_resolve,
    select_to_hand_options,
)
from .decklist import parse_decklist_file, parse_decklist_text
from .effects.casting import bounce_land_etb, cast_aura, cast_permanent_from_hand, enters_battlefield, play_land_from_hand
from .effects.combat import combat_damage_step, creature_attack_eligible, creature_block_eligible, declare_attacker, declare_attackers_step
from .effects.madness_and_plot import execute_madness_cast, plot_to_exile
from .effects.shared import find_and_remove_by_name
from .effects.stack import on_cast_trigger, push_to_stack, resolve_top_of_stack
from .effects.state_based import HAND_SIZE_LIMIT, cleanup_step
from .effects.stats import creature_keywords, enchantment_count, has_keyword, permanent_power, permanent_toughness
from .effects.tokens import (
    BLOOD_TOKEN_CARD_DEF,
    ELDRAZI_SPAWN_TOKEN_CARD_DEF,
    ROBOT_TOKEN_CARD_DEF,
    TOKEN_LIMIT,
    WARRIOR_TOKEN_CARD_DEF,
    activate_blood_sac,
    activate_eldrazi_spawn_sac,
    create_token,
)
from .effects.triggers import promote_triggers_to_stack
from .mana import (
    COLORS,
    POOL_COLORS,
    TRON_TYPES,
    abandon_pay_cost,
    begin_pay_cost,
    choose_taps_for_cost,
    controls_all_tron_types,
    execute_payment,
    execute_pool_spend,
    execute_tap_cost_option,
    mana_output,
    pay_cost,
    plan_payment,
    pool_spend_options,
    tap_cost_options,
)
from .registry import CARD_DEFS, EFFECT_REGISTRY, ENTERS_TAPPED_EFFECTS, SIMPLE_MANA_SOURCE_EFFECTS, derive_pending_kinds
from .reporting import aggregate_results, print_report
from .resolution import (
    begin_choose_opponent_permanent,
    begin_choose_permanent,
    begin_declare_blockers,
    begin_discard,
    begin_madness_decision,
    begin_resolution,
    begin_sacrifice,
    begin_order_triggers,
    begin_scry_surveil,
    begin_search_fetch,
    choose_opponent_permanent_options,
    choose_permanent_options,
    complete_resolution,
    declare_blocker_assignment,
    discard_options,
    execute_choose_opponent_permanent_option,
    execute_choose_permanent_option,
    execute_discard_decline,
    execute_discard_option,
    execute_madness_decline,
    execute_order_triggers_option,
    execute_sacrifice_option,
    execute_scry_surveil_option,
    execute_search_fetch_decline,
    execute_search_fetch_option,
    madness_decision_options,
    order_triggers_options,
    sacrifice_options,
    scry,
    scry_surveil_options,
    search_fetch_options,
    surveil,
)
from .state import GameState, Permanent, build_shuffled_library, new_game_state, new_multiplayer_game_state
from .turn import Phase, Speed, draw_step, run_game, run_multiplayer_game, run_turn, untap_step
