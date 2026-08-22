"""Multicolor-identity card catalog: any card whose real cost or mana
output touches 2+ colors (e.g. Rakdos Carnarium's {T}: Add {B}{R}), one
shared bucket regardless of pair. Cost/type/oracle text is from Scryfall.
Sneaky Snacker's real cost is {U}{B} -- never actually cast in either
deck that plays it (no "cast" spec: discarded, then returned by its own
on_draw_count trigger), kept here as accurate catalog metadata.

AUTHORIZED SIMPLIFICATION (owner, 2026-07-31): Slippery Bogle's real cost is
the hybrid {G/U} -- multicolor for color identity purposes (a hybrid symbol
counts as both colors), same reasoning that puts it here rather than
green_cards.py despite boggles.txt never touching blue. cast_cost below is
modeled as plain {G}: boggles runs no blue mana sources at all, so the
hybrid's blue half is unreachable regardless of how it's represented, and
this engine has no general "pay with either of these colors" cost
representation to build for a single unreachable branch on one card. The
same authorized deviation covers Abandon Attachments' {1}{U/R}->{1}{U}
(blue_cards.py) and Burning-Tree Emissary's {R/G}{R/G}->{R}{R} (red_cards.py)
-- each card's own alternate color is likewise unreachable in the one deck
that plays it (real costs verified via Scryfall)."""

from .. import resolution
from ..cards import CardDef, CardType, EffectId, card_subtypes
from ..effects.casting import (
    bounce_land_etb, capture_any_target, cast_aura, cast_permanent_from_hand,
    cast_targeting_creature, has_creature_target, target_still_legal,
)
from ..effects.shared import any_creature_on_either_battlefield, discard_from_hand_to_graveyard
from ..effects.stack import push_to_stack
from ..effects.state_based import check_state_based_actions, destroy_permanent
from ..effects.stats import can_be_targeted
from ..effects.tokens import ELDRAZI_SPAWN_TOKEN_CARD_DEF, create_token
from ..effects.win_check import deal_damage_to_player

MULTICOLOR_CARD_CATALOG = {
    "Wooded Ridgeline": CardDef("Wooded Ridgeline", CardType.LAND, None, EffectId.WOODED_RIDGELINE),
    "Rakdos Carnarium": CardDef("Rakdos Carnarium", CardType.LAND, None, EffectId.RAKDOS_CARNARIUM),
    "Jagged Barrens": CardDef("Jagged Barrens", CardType.LAND, None, EffectId.JAGGED_BARRENS),
    # The four indestructible "Bridge" artifact lands: each enters tapped,
    # taps for one of two colors, and is both a land and an artifact (so it
    # counts for affinity/metalcraft and is a legal artifact-sacrifice).
    "Drossforge Bridge": CardDef(
        "Drossforge Bridge", CardType.LAND, None, EffectId.DROSSFORGE_BRIDGE, artifact=True, indestructible=True,
    ),
    "Mistvault Bridge": CardDef(
        "Mistvault Bridge", CardType.LAND, None, EffectId.MISTVAULT_BRIDGE, artifact=True, indestructible=True,
    ),
    "Silverbluff Bridge": CardDef(
        "Silverbluff Bridge", CardType.LAND, None, EffectId.SILVERBLUFF_BRIDGE, artifact=True, indestructible=True,
    ),
    "Slagwoods Bridge": CardDef(
        "Slagwoods Bridge", CardType.LAND, None, EffectId.SLAGWOODS_BRIDGE, artifact=True, indestructible=True,
    ),
    # U/B tapped duals. Not artifacts. Island subtype matters for Islandcycling.
    # Ice Tunnel is a Snow land; snow is untracked (no card in this pool cares).
    "Contaminated Aquifer": CardDef(
        "Contaminated Aquifer", CardType.LAND, None, EffectId.CONTAMINATED_AQUIFER, subtypes=("Island", "Swamp"),
    ),
    "Ice Tunnel": CardDef(
        "Ice Tunnel", CardType.LAND, None, EffectId.ICE_TUNNEL, subtypes=("Island", "Swamp"),
    ),
    "Sneaky Snacker": CardDef(
        "Sneaky Snacker", CardType.CREATURE, {"U": 1, "B": 1}, EffectId.SNEAKY_SNACKER, power=2, toughness=1,
    ),
    "Slippery Bogle": CardDef(
        "Slippery Bogle", CardType.CREATURE, {"G": 1}, EffectId.SLIPPERY_BOGLE, power=1, toughness=1,
    ),
    "Armadillo Cloak": CardDef(
        "Armadillo Cloak", CardType.ENCHANTMENT, {"generic": 1, "G": 1, "W": 1}, EffectId.ARMADILLO_CLOAK,
    ),

    # --- G3 removal & tricks (multicolor) ---
    "Terminate": CardDef("Terminate", CardType.INSTANT, {"B": 1, "R": 1}, EffectId.TERMINATE),
    "Agony Warp": CardDef("Agony Warp", CardType.INSTANT, {"U": 1, "B": 1}, EffectId.AGONY_WARP),

    # --- G8: jund_wildfire. Devoid -> colorless. Reach.
    "Writhing Chrysalis": CardDef(
        "Writhing Chrysalis", CardType.CREATURE, {"generic": 2, "R": 1, "G": 1}, EffectId.WRITHING_CHRYSALIS,
        power=2, toughness=3, devoid=True, subtypes=("Eldrazi", "Drone"),
    ),
}


def jagged_barrens_etb(state, permanent):
    """When this land enters, it deals 1 damage to target opponent. Opponent
    captured directly (the only legal target). No-op with no opponent."""
    if len(state.players) < 2:
        return
    opponent_idx = 1 - state.active_idx

    def _resolve(state, card_def):
        deal_damage_to_player(state, opponent_idx, 1)

    push_to_stack(state, permanent.card_def, _resolve, reserves_hand_card=False, is_spell=False)


def cast_writhing_chrysalis(state, card_def):
    """{2}{R}{G} Devoid: when you cast this spell, create two 0/1 Eldrazi
    Spawn, then the 2/3 enters on resolution."""
    create_token(state, ELDRAZI_SPAWN_TOKEN_CARD_DEF)
    create_token(state, ELDRAZI_SPAWN_TOKEN_CARD_DEF)
    push_to_stack(state, card_def, lambda st, cd: cast_permanent_from_hand(st, cd))


def _writhing_chrysalis_on_sacrifice(state, permanent, sacrificed_card_def):
    if "Eldrazi" in card_subtypes(sacrificed_card_def):  # "another Eldrazi"
        permanent.counters["+1/+1"] = permanent.counters.get("+1/+1", 0) + 1


def cast_terminate(state, card_def):
    """{B}{R}: Destroy target creature. It can't be regenerated -- a no-op,
    since no card in this engine ever grants regeneration."""
    cast_targeting_creature(state, card_def, lambda st, perm: destroy_permanent(st, perm))


def cast_agony_warp(state, card_def):
    """{U}{B}: Target creature gets -3/-0 until end of turn. Target creature
    gets -0/-3 until end of turn. Two independent targets, locked at cast,
    that may be the same creature. Each half applies only if its own
    target is still legal at resolution."""
    idx = state.active_idx

    def _on_first(state, desc1):
        cap1 = capture_any_target(state, desc1)

        def _on_second(state, desc2):
            cap2 = capture_any_target(state, desc2)

            def _resolve(state, card_def):
                discard_from_hand_to_graveyard(state, card_def)
                if cap1 is not None and target_still_legal(state, cap1):
                    cap1[1].temp_power -= 3  # -3/-0 until end of turn
                if cap2 is not None and target_still_legal(state, cap2):
                    cap2[1].temp_toughness -= 3  # -0/-3 until end of turn
                check_state_based_actions(state)  # 0-toughness -> dies

            push_to_stack(state, card_def, _resolve, targets=tuple(t for t in (cap1, cap2) if t is not None))

        resolution.begin_choose_any_target(
            state, lambda p: p.card_type == CardType.CREATURE and can_be_targeted(state, p, idx),
            _on_second, allow_players=False,
        )

    resolution.begin_choose_any_target(
        state, lambda p: p.card_type == CardType.CREATURE and can_be_targeted(state, p, idx),
        _on_first, allow_players=False,
    )


MULTICOLOR_EFFECT_REGISTRY = {
    EffectId.WOODED_RIDGELINE: {
        "mana": ("flexible", {"R", "G"}),
        "enters_tapped": True,
    },
    EffectId.RAKDOS_CARNARIUM: {
        "mana": ("fixed_multi", ("B", "R")),
        "enters_tapped": True,
        "etb_trigger": lambda state, permanent: bounce_land_etb(state),
        "pending_kinds": {"choose_permanent"},
    },
    EffectId.JAGGED_BARRENS: {
        "mana": ("flexible", {"B", "R"}),
        "enters_tapped": True,
        "etb_trigger": lambda state, permanent: jagged_barrens_etb(state, permanent),
        "etb_targets": True,  # target opponent captured at promotion; only legal candidate
    },
    # The four indestructible Bridge artifact lands: enter tapped, tap for one of two colors.
    EffectId.DROSSFORGE_BRIDGE: {
        "mana": ("flexible", {"B", "R"}),
        "enters_tapped": True,
    },
    EffectId.MISTVAULT_BRIDGE: {
        "mana": ("flexible", {"U", "B"}),
        "enters_tapped": True,
    },
    EffectId.SILVERBLUFF_BRIDGE: {
        "mana": ("flexible", {"U", "R"}),
        "enters_tapped": True,
    },
    EffectId.SLAGWOODS_BRIDGE: {
        "mana": ("flexible", {"R", "G"}),
        "enters_tapped": True,
    },
    EffectId.CONTAMINATED_AQUIFER: {
        "mana": ("flexible", {"U", "B"}),
        "enters_tapped": True,
    },
    EffectId.ICE_TUNNEL: {
        "mana": ("flexible", {"U", "B"}),
        "enters_tapped": True,
    },
    # Never actually cast (real cost {U}{B} is off-color for both decks that
    # play it): always discarded, then returned by its own on_draw_count trigger.
    EffectId.SNEAKY_SNACKER: {
        "keywords": {"flying"},
        "on_draw_count": {"count": 3},
        # order_triggers: 2+ copies crossing their draw-count trigger on the same draw need a placement-order choice.
        "pending_kinds": {"order_triggers"},
    },
    EffectId.SLIPPERY_BOGLE: {
        # Vanilla 1/1 with hexproof for {G} -- can't be targeted by opponents' spells/abilities.
        "cast": {"resolve": lambda state, card_def: cast_permanent_from_hand(state, card_def)},
        "keywords": {"hexproof"},
    },
    EffectId.TERMINATE: {
        "cast": {
            "resolve": lambda state, card_def: cast_terminate(state, card_def),
            "extra_legal": lambda state: has_creature_target(state),
            "precast_choice": True,
        },
        "pending_kinds": {"choose_any_target"},
    },
    EffectId.WRITHING_CHRYSALIS: {
        "cast": {
            "resolve": lambda state, card_def: cast_writhing_chrysalis(state, card_def),
            "precast_choice": True,  # the two Eldrazi Spawn are made as it's cast, then it enters
        },
        "keywords": {"reach"},
        "on_sacrifice": lambda state, permanent, sacrificed_card_def: _writhing_chrysalis_on_sacrifice(state, permanent, sacrificed_card_def),
    },
    EffectId.AGONY_WARP: {
        "cast": {
            "resolve": lambda state, card_def: cast_agony_warp(state, card_def),
            "extra_legal": lambda state: has_creature_target(state),  # targets may be the same creature
            "precast_choice": True,  # both targets locked at cast
        },
        "pending_kinds": {"choose_any_target"},
    },
    EffectId.ARMADILLO_CLOAK: {
        # Enchanted creature gets trample and +2/+2, and triggers a life gain
        # equal to damage dealt whenever it deals damage -- not real lifelink,
        # so it's its own "lifelink" key (stats.lifelink_count), summed across
        # every enchanting Aura rather than deduped, so two Cloaks trigger twice.
        "cast": {
            "resolve": lambda state, card_def: cast_aura(
                state, card_def, lambda p: p.card_type == CardType.CREATURE,
            ),
            "extra_legal": lambda state: any_creature_on_either_battlefield(state),
            "precast_choice": True,  # target chosen before the stack
        },
        "pending_kinds": {"choose_any_target"},
        "pt_bonus": lambda state, aura: 2,
        "toughness_bonus": lambda state, aura: 2,
        "keywords": {"trample"},
        "lifelink": True,
    },
}
