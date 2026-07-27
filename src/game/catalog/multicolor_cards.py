"""Multicolor-identity card catalog: any card whose real cost or mana
output touches 2+ colors (e.g. Rakdos Carnarium's {T}: Add {B}{R}), one
shared bucket regardless of which pair for now -- split into per-guild
files later only if this one gets crowded. Every card's cost/type/
oracle-text below is a direct Scryfall pull. Sneaky Snacker's real cost
is {U}{B} -- never actually cast in either deck that plays it (no "cast"
spec at all: it's discarded, then returned by its own on_draw_count
trigger), kept here only as accurate catalog metadata.

Slippery Bogle's real cost is the hybrid {G/U} -- multicolor for color
identity purposes (a hybrid symbol counts as both colors), same reasoning
that puts it here rather than green_cards.py despite boggles.txt never
touching blue. cast_cost below is modeled as plain {G}: boggles runs no
blue mana sources at all, so the hybrid's blue half is unreachable
regardless of how it's represented, and this engine has no general
"pay with either of these colors" cost representation to build for a
single unreachable branch on one card -- a deliberate simplification, not
a guess (real cost verified via Scryfall)."""

from .. import resolution
from ..cards import CardDef, CardType, EffectId
from ..effects.casting import (
    bounce_land_etb, capture_any_target, cast_aura, cast_permanent_from_hand,
    cast_targeting_creature, has_creature_target, target_still_legal,
)
from ..effects.shared import any_creature_on_battlefield, card_subtypes, discard_from_hand_to_graveyard
from ..effects.stack import push_to_stack
from ..effects.state_based import check_state_based_actions, destroy_permanent
from ..effects.stats import can_be_targeted
from ..effects.tokens import ELDRAZI_SPAWN_TOKEN_CARD_DEF, create_token
from ..effects.win_check import deal_damage_to_opponent

MULTICOLOR_CARD_CATALOG = {
    "Wooded Ridgeline": CardDef("Wooded Ridgeline", CardType.LAND, None, EffectId.WOODED_RIDGELINE),
    "Rakdos Carnarium": CardDef("Rakdos Carnarium", CardType.LAND, None, EffectId.RAKDOS_CARNARIUM),
    "Jagged Barrens": CardDef("Jagged Barrens", CardType.LAND, None, EffectId.JAGGED_BARRENS),
    # --- new decks: the four indestructible "Bridge" artifact lands
    # (grixis_affinity / jund_wildfire). Each enters tapped, taps for one of
    # two colors, and is BOTH a land and an artifact (extra["artifact"]) --
    # so it counts for affinity/metalcraft and is a legal artifact-sacrifice.
    # Indestructible (extra["indestructible"]) matters against "destroy target
    # land" (Cleansing Wildfire) -- honored once that lands in G3. ---
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
    # U/B tapped duals (dmir_terror). NOT artifacts. Subtype Island (+Swamp)
    # matters for Islandcycling's "search for an Island card" (G2). Ice
    # Tunnel is a Snow land -- snow is a no-op in this pool (no card cares),
    # so it's not tracked, only noted.
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

    # --- G8: jund_wildfire. Devoid -> colorless (extra["devoid"]); the {2}{R}{G}
    # cost is what it's paid with (green half reachable in jund, so no
    # hybrid/color simplification here). Reach.
    "Writhing Chrysalis": CardDef(
        "Writhing Chrysalis", CardType.CREATURE, {"generic": 2, "R": 1, "G": 1}, EffectId.WRITHING_CHRYSALIS,
        power=2, toughness=3, devoid=True, subtypes=("Eldrazi", "Drone"),
    ),
}


def cast_writhing_chrysalis(state, card_def):
    """{2}{R}{G} Devoid: "When you cast this spell, create two 0/1 Eldrazi
    Spawn" -- made as it's cast (before it resolves), then the 2/3 enters on
    resolution. Reach + "whenever you sacrifice another Eldrazi, +1/+1" (its
    on_sacrifice below)."""
    create_token(state, ELDRAZI_SPAWN_TOKEN_CARD_DEF)
    create_token(state, ELDRAZI_SPAWN_TOKEN_CARD_DEF)
    push_to_stack(state, card_def, lambda st, cd: cast_permanent_from_hand(st, cd))


def _writhing_chrysalis_on_sacrifice(state, permanent, sacrificed_card_def):
    if "Eldrazi" in card_subtypes(sacrificed_card_def):  # "another Eldrazi"
        permanent.counters["+1/+1"] = permanent.counters.get("+1/+1", 0) + 1


def cast_terminate(state, card_def):
    """{B}{R}: Destroy target creature. It can't be regenerated (a no-op --
    regeneration isn't modeled)."""
    cast_targeting_creature(state, card_def, lambda st, perm: destroy_permanent(st, perm))


def cast_agony_warp(state, card_def):
    """{U}{B}: "Target creature gets -3/-0 until end of turn. Target creature
    gets -0/-3 until end of turn." Two independent targets, both locked at
    cast (precast_choice) -- they MAY be the same creature (then -3/-3), so a
    single legal creature is enough to cast. On resolution each half applies
    only if its own target is still a legal creature (608.2c -- a spell does
    as much as it can); a -0/-3 that drops a creature to 0 toughness kills it
    via the state-based-action check."""
    idx = state.active_idx

    def _on_first(state, desc1):
        cap1 = capture_any_target(state, desc1)

        def _on_second(state, desc2):
            cap2 = capture_any_target(state, desc2)

            def _resolve(state, card_def):
                discard_from_hand_to_graveyard(state, card_def)
                if target_still_legal(state, cap1):
                    cap1[1].temp_power -= 3  # -3/-0 until end of turn
                if target_still_legal(state, cap2):
                    cap2[1].temp_toughness -= 3  # -0/-3 until end of turn
                check_state_based_actions(state)  # 0-toughness -> dies

            push_to_stack(state, card_def, _resolve)

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
        # Deals 1 damage to target opponent.
        "etb_trigger": lambda state, permanent: deal_damage_to_opponent(state, 1),
    },
    # --- the four indestructible Bridge artifact lands: enter tapped, tap
    # for one of two colors. Indestructibility is data-only for now
    # (extra["indestructible"]) -- honored by "destroy target land" in G3. ---
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
    # Never actually cast (real cost is {U}{B} -- off-color for both
    # rakdos_madness and mono_red_madness, by design): always discarded,
    # then returned by its own on_draw_count trigger. No "cast" spec at
    # all -- matches Generous Ent's own "never hard-cast" precedent.
    EffectId.SNEAKY_SNACKER: {
        "on_draw_count": {"count": 3},
        # order_triggers: reachable the
        # instant 2+ copies both cross their own draw-count trigger on
        # the same draw -- a real placement-order choice, not fixed
        # queue order, even though this trigger is otherwise "automatic"
        # (no cast-or-decline choice of its own).
        "pending_kinds": {"order_triggers"},
    },
    EffectId.SLIPPERY_BOGLE: {
        # Vanilla 1/1 with hexproof for {G}. Hexproof is now a REAL targeting
        # restriction (stats.can_be_targeted): once opponents can target
        # across sides (faithful burn/removal), this bogle can't be the
        # target of their spells/abilities -- the whole point of the deck.
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
            # Two targets, but they may be the SAME creature, so one legal
            # creature is enough to cast.
            "extra_legal": lambda state: has_creature_target(state),
            "precast_choice": True,  # both targets locked at cast (like Ram Through)
        },
        "pending_kinds": {"choose_any_target"},
    },
    EffectId.ARMADILLO_CLOAK: {
        # Real text: enchanted creature also gets trample (combat_damage_
        # step's own trample-aware damage assignment reads the "keywords"
        # below) and "whenever enchanted creature
        # deals damage, you gain that much life" -- a TRIGGERED ability,
        # not real lifelink, so it's its OWN "lifelink": True key
        # (stats.lifelink_count), summed across every enchanting Aura
        # rather than deduped into the boolean "keywords" set below: two
        # Cloaks on the same creature really do trigger twice, once each,
        # each for the full damage dealt (see stats.lifelink_count's own
        # docstring for why a boolean keyword would silently undercount
        # that stacking).
        "cast": {
            "resolve": lambda state, card_def: cast_aura(
                state, card_def, lambda p: p.card_type == CardType.CREATURE,
            ),
            "extra_legal": lambda state: any_creature_on_battlefield(state),
            "precast_choice": True,  # real MTG "enchant target creature" -- must be chosen before the stack, see drl_env._precast_choice_execute
        },
        "pending_kinds": {"choose_any_target"},  # Aura: cast_aura targets any creature (either side), hexproof-aware
        "pt_bonus": lambda state, aura: 2,
        "toughness_bonus": lambda state, aura: 2,  # real text is +2/+2
        "keywords": {"trample"},
        "lifelink": True,
    },
}


if __name__ == "__main__":
    # ponytail self-check (run via `python -m game.catalog.multicolor_cards`
    # from src/): G3 removal -- Terminate (destroy any creature, incl. black)
    # and Agony Warp (-3/-0 & -0/-3 until EOT, two independent targets, a
    # -0/-3 that drops a creature to 0 toughness kills it via SBA).
    from .. import registry
    from ..effects.stack import resolve_top_of_stack
    from ..effects.stats import permanent_power, permanent_toughness
    from ..state import GameState, Permanent, PlayerState

    def _two():
        return GameState(on_the_play=True, players=[PlayerState(True), PlayerState(False)])

    # Terminate destroys a black creature (any creature qualifies).
    state = _two()
    victim = Permanent(CardDef("Black Creature", CardType.CREATURE, {"B": 1}, EffectId.FILLER, power=2, toughness=2))
    victim.slot = 1
    state.players[1].battlefield = [victim]
    state.players[0].hand = [registry.CARD_DEFS["Terminate"]]
    cast_terminate(state, registry.CARD_DEFS["Terminate"])
    resolution.execute_choose_any_target_creature(state, 1, "Black Creature", 1)
    resolve_top_of_stack(state)
    assert victim not in state.players[1].battlefield
    print("multicolor_cards.py Terminate self-check: OK")

    # Agony Warp on two different targets: 5/5 -> 2/5 (survives) and 1/3 ->
    # 1/0 (dies).
    state = _two()
    big = Permanent(CardDef("Big", CardType.CREATURE, None, EffectId.FILLER, power=5, toughness=5))
    big.slot = 1
    small = Permanent(CardDef("Small", CardType.CREATURE, None, EffectId.FILLER, power=1, toughness=3))
    small.slot = 1
    state.players[1].battlefield = [big, small]
    state.players[0].hand = [registry.CARD_DEFS["Agony Warp"]]
    cast_agony_warp(state, registry.CARD_DEFS["Agony Warp"])
    resolution.execute_choose_any_target_creature(state, 1, "Big", 1)     # -3/-0
    resolution.execute_choose_any_target_creature(state, 1, "Small", 1)   # -0/-3
    resolve_top_of_stack(state)
    assert big in state.players[1].battlefield and permanent_power(state, big) == 2 and permanent_toughness(state, big) == 5
    assert small not in state.players[1].battlefield  # 3 - 3 = 0 toughness -> dead
    print("multicolor_cards.py Agony Warp (split targets, -0/-3 lethal) self-check: OK")

    # Agony Warp both halves onto the SAME creature (only one on board): -3/-3.
    state = _two()
    lone = Permanent(CardDef("Lone", CardType.CREATURE, None, EffectId.FILLER, power=4, toughness=4))
    lone.slot = 1
    state.players[1].battlefield = [lone]
    state.players[0].hand = [registry.CARD_DEFS["Agony Warp"]]
    cast_agony_warp(state, registry.CARD_DEFS["Agony Warp"])
    resolution.execute_choose_any_target_creature(state, 1, "Lone", 1)
    resolution.execute_choose_any_target_creature(state, 1, "Lone", 1)
    resolve_top_of_stack(state)
    assert lone in state.players[1].battlefield  # 4/4 -> 1/1, survives
    assert permanent_power(state, lone) == 1 and permanent_toughness(state, lone) == 1
    print("multicolor_cards.py Agony Warp (same creature -3/-3) self-check: OK")

    # --- G8: Writhing Chrysalis -- Devoid; cast makes two Eldrazi Spawn; sac
    # another Eldrazi -> +1/+1.
    from ..effects.shared import card_colors
    state = GameState(on_the_play=True)
    state.hand = [registry.CARD_DEFS["Writhing Chrysalis"]]
    cast_writhing_chrysalis(state, registry.CARD_DEFS["Writhing Chrysalis"])
    assert sum(1 for p in state.battlefield if p.card_def.name == "Eldrazi Spawn") == 2  # made at cast
    resolve_top_of_stack(state)  # Writhing enters
    wr = next(p for p in state.battlefield if p.card_def.name == "Writhing Chrysalis")
    assert card_colors(wr.card_def) == set()  # Devoid -> colorless
    spawn = next(p for p in state.battlefield if p.card_def.name == "Eldrazi Spawn")
    registry.EFFECT_REGISTRY[EffectId.ELDRAZI_SPAWN_TOKEN]["activated_abilities"]["sac"]["resolve"](state, spawn)
    assert wr.counters.get("+1/+1") == 1  # "whenever you sacrifice another Eldrazi"
    print("multicolor_cards.py Writhing Chrysalis (Devoid + Eldrazi Spawn + sac counter) self-check: OK")
