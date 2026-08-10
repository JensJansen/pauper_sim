"""Multi-deck Magic-subset simulator -- package entry point.

Split by domain: cards, state, resolution, mana, effects, turn loop,
per-color card catalogs, and the registry that unions every color. A
deck is just a decklist file (data/*.txt, parsed by game.decklist)
resolved against the shared card catalog (game.CARD_DEFS) each color
catalog contributes to -- adding or reweighting a deck built entirely
from already-implemented cards never needs a code change, and neither
does reusing a card across multiple decks.

Every submodule is re-exported here flat (game.CARD_DEFS, game.GameState,
game.play_land_from_hand, ...) so every existing `import game; game.X`
caller (drl_env, rl.rewards, and the rl.* training pipeline)
keeps working unchanged.

Import order matters: `from . import registry` first is what actually
triggers loading every catalog module (and, transitively, most of the
game.effects package / mana / resolution / state / cards) -- see
game/registry.py's own module docstring for why several of those modules
only reference `registry.EFFECT_REGISTRY` lazily instead of importing the
name directly, which is what makes this order-independent rather than a
real problem.
"""

from . import registry
from .cards import CardDef, CardType, EffectId
from .catalog.black_cards import (
    cast_dread_return,
    flashback_dread_return,
    lotleth_giant_etb,
    mill_until_land,
)
from .catalog.colorless_cards import (
    activate_bonders_ornament_draw,
    activate_candy_trail_sac,
    activate_expedition_map,
    activate_relic_of_progenitus_draw,
    activate_relic_of_progenitus_exile,
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
from .effects.combat import combat_damage_step, creature_attack_eligible, creature_block_eligible, declare_attacker, declare_attackers_step, enforce_menace, has_unfulfilled_goad, menace_block_incomplete
from .effects.undercity import (
    INITIATIVE_MARKER_CARD,
    ROOM_NAMES,
    apply_goad,
    begin_choose_room,
    begin_throne_reveal,
    choose_room_options,
    execute_choose_room_option,
    execute_throne_reveal_option,
    expire_until_next_turn,
    take_initiative,
    throne_reveal_options,
    venture,
)
from .effects.madness_and_plot import execute_madness_cast, plot_to_exile
from .effects.shared import find_and_remove_by_name
from .effects.stack import counter_spell, on_cast_trigger, push_ability_to_stack, push_to_stack, resolve_top_of_stack
from .effects.state_based import HAND_SIZE_LIMIT, cleanup_step
from .effects.stats import creature_keywords, enchanting_by_target, enchantment_count, has_keyword, permanent_power, permanent_toughness
from .effects.tokens import (
    BIRD_ILLUSION_TOKEN_CARD_DEF,
    BLOOD_TOKEN_CARD_DEF,
    CLUE_TOKEN_CARD_DEF,
    ELDRAZI_SPAWN_TOKEN_CARD_DEF,
    FOOD_TOKEN_CARD_DEF,
    HUMAN_SOLDIER_TOKEN_CARD_DEF,
    MAP_TOKEN_CARD_DEF,
    ROBOT_TOKEN_CARD_DEF,
    SAMURAI_TOKEN_CARD_DEF,
    SKELETON_TOKEN_CARD_DEF,
    TOKEN_LIMIT,
    TREASURE_TOKEN_CARD_DEF,
    WARRIOR_TOKEN_CARD_DEF,
    activate_blood_sac,
    activate_eldrazi_spawn_sac,
    activate_food_sac,
    create_token,
)
from .effects.triggers import promote_triggers_to_stack
from .mana import (
    COLORS,
    POOL_COLORS,
    TRON_TYPES,
    activate_mana_source,
    begin_pay_cost,
    controls_all_tron_types,
    execute_pool_spend,
    float_mana,
    mana_ability_options,
    mana_output,
    plan_payment,
    pool_can_pay,
    pool_spend_options,
    reset_mana_cache,
    spend_one_pip,
    tap_summoning_locked,
)
from .registry import CARD_DEFS, EFFECT_REGISTRY, ENTERS_TAPPED_EFFECTS, derive_pending_kinds
from .resolution import (
    assign_combat_damage_options,
    begin_assign_combat_damage,
    begin_bottom,
    begin_choose_any_target,
    begin_choose_cast_copy,
    begin_choose_cast_mode,
    begin_choose_cast_x,
    begin_choose_delve_amount,
    begin_choose_graveyard_card,
    begin_choose_opponent_permanent,
    begin_choose_mana_color,
    begin_choose_permanent,
    begin_choose_stack_target,
    begin_choose_target_player,
    begin_declare_blockers,
    begin_exile_n_from_graveyard,
    begin_mana_color_choice,
    begin_mana_subdecision,
    begin_may_cast,
    begin_may_copy,
    begin_may_transform,
    begin_pay_unless,
    begin_tuck_to_library,
    begin_discard,
    begin_discard_or_sacrifice,
    begin_madness_decision,
    begin_mulligan,
    begin_ponder,
    begin_put_on_top_from_hand,
    begin_resolution,
    begin_sacrifice,
    begin_order_triggers,
    begin_scry_surveil,
    begin_search_fetch,
    bottom_options,
    choose_any_target_creature_options,
    choose_any_target_options,
    choose_cast_copy_options,
    choose_cast_mode_options,
    choose_cast_x_options,
    choose_delve_amount_options,
    choose_graveyard_card_options,
    choose_opponent_permanent_options,
    choose_mana_color_options,
    choose_permanent_options,
    choose_stack_target_options,
    complete_resolution,
    declare_blocker_assignment,
    discard_options,
    discard_or_sacrifice_can_sacrifice,
    discard_or_sacrifice_discard_options,
    execute_bottom_option,
    execute_choose_any_target_creature,
    execute_choose_any_target_decline,
    execute_choose_any_target_player,
    execute_choose_cast_copy_option,
    execute_choose_cast_mode_option,
    execute_choose_cast_x_option,
    execute_choose_delve_amount_option,
    execute_choose_graveyard_card_decline,
    execute_choose_graveyard_card_option,
    execute_choose_opponent_permanent_option,
    execute_choose_mana_color,
    execute_choose_permanent_option,
    execute_mana_subdecision_color,
    execute_mana_subdecision_target,
    execute_choose_stack_target_option,
    execute_assign_combat_damage_option,
    execute_assign_combat_damage_to_player,
    execute_choose_target_player_option,
    execute_discard_decline,
    execute_discard_option,
    execute_discard_or_sacrifice_decline,
    execute_discard_or_sacrifice_discard,
    execute_discard_or_sacrifice_trigger_sacrifice,
    execute_madness_decline,
    execute_mulligan_keep,
    execute_mulligan_take,
    execute_order_triggers_option,
    execute_ponder_option,
    execute_ponder_shuffle,
    execute_put_on_top_option,
    execute_scry_surveil_option,
    execute_may_cast,
    execute_may_copy,
    execute_may_transform,
    execute_tuck_position,
    explore,
    execute_search_fetch_decline,
    execute_search_fetch_option,
    madness_decision_options,
    mulligan_decision_options,
    order_triggers_options,
    pay_unless_decline,
    pay_unless_pay,
    ponder_options,
    put_on_top_options,
    scry,
    scry_surveil_options,
    search_fetch_options,
    surveil,
)
from .state import GameState, Permanent, build_shuffled_library, new_multiplayer_game_state
from .turn import Phase, Speed, draw_step, game_coroutine, run_multiplayer_game, run_turn, untap_step
