"""Blue-identity card catalog: BLUE_CARD_CATALOG (name -> CardDef) and
BLUE_EFFECT_REGISTRY (EffectId -> spec), unioned into game.CARD_DEFS /
EFFECT_REGISTRY by game/registry.py. Cost/type/oracle text is from
Scryfall.

Seat of the Synod is a land that's also an artifact, so it counts for
affinity/metalcraft and is a legal artifact-sacrifice."""

from .. import registry
from ..cards import CardDef, CardType, EffectId, card_subtypes
from ..effects.casting import (
    _log_target_fizzle, capture_any_target, cast_permanent_from_hand, cast_targeting_creature, has_creature_target,
    target_still_legal,
)
from ..effects.shared import (
    affinity_reduction, discard_from_hand_to_graveyard, find_to_hand, graveyard_instant_sorcery_count, mill,
    set_tapped,
)
from ..effects.stack import counter_spell, push_ability_to_stack, push_to_stack
from ..effects.state_based import departing_card_def, sacrifice_to_graveyard
from ..effects.combat import remove_from_combat
from ..effects.stats import can_be_targeted, controller_idx
from ..effects.tokens import BIRD_ILLUSION_TOKEN_CARD_DEF, create_token
from ..effects.win_check import lose_life
from ..mana import discount_departing_source
from ..turn import Speed
from ..resolution import (
    begin_choose_any_target, begin_choose_graveyard_card, begin_choose_stack_target, begin_choose_target_player,
    begin_discard, begin_may_transform, begin_pay_unless, begin_ponder, begin_put_on_top_from_hand, begin_search_fetch,
    begin_tuck_to_library,
)

BLUE_CARD_CATALOG = {
    "Island": CardDef("Island", CardType.LAND, None, EffectId.ISLAND, basic=True, subtypes=("Island",)),
    "Seat of the Synod": CardDef(
        "Seat of the Synod", CardType.LAND, None, EffectId.SEAT_OF_THE_SYNOD, artifact=True,
    ),
    "Mental Note": CardDef("Mental Note", CardType.INSTANT, {"U": 1}, EffectId.MENTAL_NOTE),
    "Thought Scour": CardDef("Thought Scour", CardType.INSTANT, {"U": 1}, EffectId.THOUGHT_SCOUR),
    "Lórien Revealed": CardDef(
        "Lórien Revealed", CardType.SORCERY, {"generic": 3, "U": 2}, EffectId.LORIEN_REVEALED,
        islandcycling_cost={"generic": 1},
    ),
    "Brainstorm": CardDef("Brainstorm", CardType.INSTANT, {"U": 1}, EffectId.BRAINSTORM),
    "Ponder": CardDef("Ponder", CardType.SORCERY, {"U": 1}, EffectId.PONDER),
    "Deep Analysis": CardDef("Deep Analysis", CardType.SORCERY, {"generic": 3, "U": 1}, EffectId.DEEP_ANALYSIS),
    "Counterspell": CardDef("Counterspell", CardType.INSTANT, {"U": 2}, EffectId.COUNTERSPELL),
    "Dispel": CardDef("Dispel", CardType.INSTANT, {"U": 1}, EffectId.DISPEL),
    "Spell Pierce": CardDef("Spell Pierce", CardType.INSTANT, {"U": 1}, EffectId.SPELL_PIERCE),
    # AUTHORIZED SIMPLIFICATION (owner, 2026-07-31): {1}{U/R} -- modeled
    # {1}{U}: the hybrid's red half is unreachable in dmir_terror (a U/B deck
    # with no red source). Same deviation as Slippery Bogle's {G/U}->{G}
    # (multicolor_cards.py's own module docstring; real cost per Scryfall).
    "Abandon Attachments": CardDef("Abandon Attachments", CardType.INSTANT, {"generic": 1, "U": 1}, EffectId.ABANDON_ATTACHMENTS),
    "Sleep of the Dead": CardDef("Sleep of the Dead", CardType.SORCERY, {"U": 1}, EffectId.SLEEP_OF_THE_DEAD),
    # --- G6: dmir_terror ---
    "Murmuring Mystic": CardDef("Murmuring Mystic", CardType.CREATURE, {"generic": 3, "U": 1}, EffectId.MURMURING_MYSTIC, power=1, toughness=5),
    # --- G8: grixis_affinity ---
    "Sewer-veillance Cam": CardDef("Sewer-veillance Cam", CardType.ARTIFACT, {"U": 1}, EffectId.SEWER_VEILLANCE_CAM, sac_ability_cost={"generic": 3, "U": 1}),

    # --- G7: affinity / gy-count threats ---
    "Utrom Monitor": CardDef("Utrom Monitor", CardType.CREATURE, {"generic": 4, "U": 1}, EffectId.UTROM_MONITOR, power=3, toughness=3, artifact=True),
    "Thoughtcast": CardDef("Thoughtcast", CardType.SORCERY, {"generic": 4, "U": 1}, EffectId.THOUGHTCAST),
    "Tolarian Terror": CardDef("Tolarian Terror", CardType.CREATURE, {"generic": 6, "U": 1}, EffectId.TOLARIAN_TERROR, power=5, toughness=5),
    "Cryptic Serpent": CardDef("Cryptic Serpent", CardType.CREATURE, {"generic": 5, "U": 2}, EffectId.CRYPTIC_SERPENT, power=6, toughness=5),
    "Deem Inferior": CardDef("Deem Inferior", CardType.SORCERY, {"generic": 3, "U": 1}, EffectId.DEEM_INFERIOR),

    # --- G10: transform / DFC (mono_blue_terror) ---
    "Delver of Secrets": CardDef("Delver of Secrets", CardType.CREATURE, {"U": 1}, EffectId.DELVER_OF_SECRETS, power=1, toughness=1),
}

# Back face of Delver of Secrets. Not in BLUE_CARD_CATALOG / CARD_DEFS -- only
# reachable as the card_def a transformed Delver Permanent swaps to
# (execute_may_transform). Same EffectId as the front face so registry
# lookups still resolve. A DFC reverts to its front face in every zone but
# the battlefield (state_based._departing_card_def).
INSECTILE_ABERRATION_CARD_DEF = CardDef(
    "Insectile Aberration", CardType.CREATURE, {"U": 1}, EffectId.DELVER_OF_SECRETS, power=3, toughness=2,
)


def cast_deem_inferior(state, card_def):
    """{3}{U} (costs {1} less per card drawn this turn): the owner of target
    nonland permanent puts it into their library second from the top or on
    the bottom. Target locked at cast; fizzles if the permanent has left
    the battlefield by resolution."""
    idx = state.active_idx

    def _on_target(state, descriptor):
        captured = capture_any_target(state, descriptor)

        def _resolve(state, card_def):
            discard_from_hand_to_graveyard(state, card_def)
            if captured is None or not target_still_legal(state, captured):
                where = (captured[1].card_def.name, captured[1].slot) if captured is not None else None
                _log_target_fizzle(state, card_def, where)
                return
            permanent = captured[1]
            owner_idx = controller_idx(state, permanent)
            state.players[owner_idx].battlefield.remove(permanent)
            remove_from_combat(state, permanent)  # tucking an attacker removes it from combat
            # a tucked DFC reverts to its front face
            begin_tuck_to_library(state, departing_card_def(permanent), owner_idx)

        push_to_stack(state, card_def, _resolve, targets=() if captured is None else (captured,))

    begin_choose_any_target(
        state,
        lambda p: p.card_type != CardType.LAND and can_be_targeted(state, p, idx),
        _on_target, allow_players=False,
    )


def cast_thoughtcast(state, card_def):
    """{4}{U}, Affinity for artifacts: Draw two cards."""
    discard_from_hand_to_graveyard(state, card_def)
    state.draw(2)


def cast_abandon_attachments(state, card_def):
    """{1}{U}: You may discard a card. If you do, draw two cards."""
    discard_from_hand_to_graveyard(state, card_def)

    def _on_discard(state, discarded_cards):
        if discarded_cards:
            state.draw(2)

    begin_discard(state, 1, optional=True, on_complete=_on_discard)


def _sleep_tap_skip(state, permanent):
    """Taps the target creature; it skips its controller's next untap step."""
    set_tapped(state, permanent, True, reason="sleep_of_the_dead")
    permanent.flags["skip_next_untap"] = True


def cast_sleep_of_the_dead(state, card_def):
    """{U}: Tap target creature; skip its next untap."""
    cast_targeting_creature(state, card_def, _sleep_tap_skip)


def _exile_n_other_from_graveyard(state, n, exclude, on_complete):
    """Escape's additional cost: exiles n cards from your graveyard other
    than `exclude` (by object identity), one at a time, then runs
    on_complete(state)."""
    def _step(remaining):
        if remaining == 0:
            on_complete(state)
            return

        def _chosen(state, chosen):
            state.graveyard.remove(chosen)
            state.log_event("zone_move", card=chosen.name, from_zone="graveyard", to_zone="exile_untracked", reason="escape")
            _step(remaining - 1)

        begin_choose_graveyard_card(state, lambda c: c is not exclude, _chosen)

    _step(n)


def _sleep_escape_legal(state):
    """Escape {2}{U}, exile three other graveyard cards: legal only with 3+
    other graveyard cards and a legal creature to tap."""
    if not any(c.name == "Sleep of the Dead" for c in state.graveyard):
        return False
    return len(state.graveyard) - 1 >= 3 and has_creature_target(state)


def escape_sleep_of_the_dead(state, inst):
    """Escape: {2}{U} + exile three other graveyard cards, then Sleep leaves
    the graveyard and taps a target creature (chosen at cast).

    inst: the exact graveyard CardInstance being escaped, removed by object
    identity; also the exclusion passed to _exile_n_other_from_graveyard."""
    sleep_inst = inst

    def _after_exile(state):
        state.graveyard.remove(sleep_inst)  # exiled after resolution
        idx = state.active_idx

        def _on_target(state, descriptor):
            captured = capture_any_target(state, descriptor)

            def _resolve(state, cd):
                if captured is None or not target_still_legal(state, captured):
                    _log_target_fizzle(state, cd, None)
                    return
                _sleep_tap_skip(state, captured[1])

            push_to_stack(state, sleep_inst, _resolve, reserves_hand_card=False, is_spell=True,
                          targets=() if captured is None else (captured,))

        begin_choose_any_target(
            state, lambda p: p.card_type == CardType.CREATURE and can_be_targeted(state, p, idx),
            _on_target, allow_players=False,
        )

    _exile_n_other_from_graveyard(state, 3, sleep_inst, _after_exile)


def sewer_cam_tap_or_untap(state, source_card_def):
    """Sewer-veillance Cam ETB/LTB: you may tap or untap target creature.
    Target chosen at promotion; toggles the target's tapped state
    (equivalent to picking whichever direction is meaningful), fizzling if
    the target's gone by resolution. Untapping also discounts any mana the
    target could have produced (mana.discount_departing_source)."""
    idx = state.active_idx

    def _on_target(state, descriptor):
        captured = capture_any_target(state, descriptor)

        def _resolve(st, cd):
            if captured is None or not target_still_legal(st, captured):
                return
            perm = captured[1]
            was_tapped = perm.tapped
            if was_tapped:
                discount_departing_source(st, perm, controller_idx(st, perm))
            set_tapped(st, perm, not was_tapped, reason="sewer_cam")

        push_to_stack(state, source_card_def, _resolve, reserves_hand_card=False, is_spell=False,
                      targets=() if captured is None else (captured,))

    begin_choose_any_target(
        state, lambda p: p.card_type == CardType.CREATURE and can_be_targeted(state, p, idx),
        _on_target, allow_players=False, optional=True,  # "you MAY"
    )


def sewer_cam_sac(state, permanent):
    """{3}{U}, Sacrifice this artifact: Draw two cards. Sacrificing also fires
    its own LTB (tap/untap)."""
    sacrifice_to_graveyard(state, permanent)
    push_ability_to_stack(state, permanent.card_def, lambda st: st.draw(2))


def _has_stack_spell(state, predicate):
    return any(e.get("is_spell") and predicate(e) for e in state.stack)


def _cast_counter(state, card_def, predicate, on_countered=None):
    """Shared counterspell body: choose a matching spell on the stack (locked
    at cast), then push this spell's own effect. On resolution, if the
    chosen spell is still on the stack it's countered -- or, for a rider
    (Spell Pierce), on_countered(entry) decides via begin_pay_unless."""
    def _on_target(state, entry):
        def _resolve(state, card_def):
            discard_from_hand_to_graveyard(state, card_def)
            if entry is None or entry not in state.stack:
                _log_target_fizzle(state, card_def, None)
                return
            if on_countered is None:
                counter_spell(state, entry)
            else:
                on_countered(state, entry)

        push_to_stack(state, card_def, _resolve, targets=() if entry is None else (("stack_entry", entry),))

    begin_choose_stack_target(state, predicate, _on_target)


def cast_counterspell(state, card_def):
    """{U}{U}: Counter target spell."""
    _cast_counter(state, card_def, lambda e: True)


def cast_dispel(state, card_def):
    """{U}: Counter target instant spell."""
    _cast_counter(state, card_def, lambda e: e["card_def"].card_type == CardType.INSTANT)


def _spell_pierce_countered(state, entry):
    """Unless its controller pays {2}, it's countered."""
    def _on_result(state, paid):
        if not paid:
            counter_spell(state, entry)

    begin_pay_unless(state, entry["controller"], {"generic": 2}, _on_result)


def cast_spell_pierce(state, card_def):
    """{U}: Counter target noncreature spell unless its controller pays {2}."""
    _cast_counter(
        state, card_def,
        lambda e: e["card_def"].card_type != CardType.CREATURE,
        on_countered=_spell_pierce_countered,
    )


def _is_island_card(card_def):
    """Any card with the Island land subtype."""
    return "Island" in card_subtypes(card_def)


def cast_mental_note(state, card_def):
    """{U}: Mill two cards. Draw a card."""
    discard_from_hand_to_graveyard(state, card_def)
    mill(state, 2)
    state.draw(1)


def cast_thought_scour(state, card_def):
    """{U}: Target player mills two cards. You draw a card. Target locked at cast."""
    def _on_player(state, idx):
        def _resolve(state, card_def):
            discard_from_hand_to_graveyard(state, card_def)
            mill(state, 2, idx)
            state.draw(1)  # the caster draws, regardless of who's milled

        push_to_stack(state, card_def, _resolve)

    begin_choose_target_player(state, _on_player)


def cast_lorien_revealed(state, card_def):
    """{3}{U}{U}: Draw three cards."""
    discard_from_hand_to_graveyard(state, card_def)
    state.draw(3)


def islandcycle_lorien_revealed(state, card_def):
    """Islandcycling {1}: discard this card, search library for an Island
    card, put it into hand, shuffle."""
    discard_from_hand_to_graveyard(state, card_def)
    begin_search_fetch(state, _is_island_card, find_to_hand)


def cast_brainstorm(state, card_def):
    """{U}: Draw three cards, then put two cards from hand on top of the
    library in any order."""
    discard_from_hand_to_graveyard(state, card_def)
    state.draw(3)
    begin_put_on_top_from_hand(state, 2, on_complete=lambda s: None)


def cast_ponder(state, card_def):
    """{U}: Look at the top three cards, reorder them or shuffle, then draw a card."""
    discard_from_hand_to_graveyard(state, card_def)
    begin_ponder(state, on_complete=lambda s: s.draw(1))


def _target_player_draws(state, idx, n):
    """Makes player `idx` draw n cards. Flips active_idx to them for the
    draw; on a deck-out, active_idx stays on them so the loss attributes
    correctly."""
    saved = state.active_idx
    state.active_idx = idx
    state.draw(n)
    state.active_idx = saved


def _deep_analysis_choose_and_push(state, card_def, to_graveyard, reserves_hand_card, exiles_on_resolve=False):
    """Choose the target player as Deep Analysis is cast, shared by the hard
    cast and Flashback below. `to_graveyard`/`reserves_hand_card` control how
    the card itself reaches the graveyard on resolution. Whoever is chosen
    draws two."""
    def _on_player(state, idx):
        def _resolve(state, card_def):
            to_graveyard(state, card_def)
            _target_player_draws(state, idx, 2)

        push_to_stack(state, card_def, _resolve, reserves_hand_card=reserves_hand_card, exiles_on_resolve=exiles_on_resolve)

    begin_choose_target_player(state, _on_player)


def cast_deep_analysis(state, card_def):
    """{3}{U}: Target player draws two cards. Target locked at cast."""
    _deep_analysis_choose_and_push(state, card_def, discard_from_hand_to_graveyard, reserves_hand_card=True)


def flashback_deep_analysis(state, inst):
    """Flashback {1}{U}, Pay 3 life: same effect, target locked as it's cast
    from the graveyard; exiled after resolution.

    inst: the exact graveyard CardInstance being flashed back, removed by
    object identity."""
    state.graveyard.remove(inst)
    lose_life(state, 3, reason="deep_analysis_flashback")
    _deep_analysis_choose_and_push(state, inst, lambda s, cd: None, reserves_hand_card=False, exiles_on_resolve=True)


def delver_upkeep(state, permanent):
    """Upkeep: look at the top card of your library; if it's an instant or
    sorcery, offer to transform Delver of Secrets."""
    if not state.library:
        return
    top = state.library[0]
    if top.card_type in (CardType.INSTANT, CardType.SORCERY):
        begin_may_transform(state, permanent, revealed_card=top.name)


BLUE_EFFECT_REGISTRY = {
    EffectId.ISLAND: {"mana": ("fixed", "U")},
    EffectId.SEAT_OF_THE_SYNOD: {"mana": ("fixed", "U")},
    EffectId.BRAINSTORM: {
        "cast": {"resolve": lambda state, card_def: cast_brainstorm(state, card_def)},
        "pending_kinds": {"put_on_top"},
    },
    EffectId.PONDER: {
        "cast": {"resolve": lambda state, card_def: cast_ponder(state, card_def)},
        "pending_kinds": {"ponder"},
    },
    EffectId.DEEP_ANALYSIS: {
        "cast": {
            "resolve": lambda state, card_def: cast_deep_analysis(state, card_def),
            "precast_choice": True,  # target player locked at cast
        },
        "flashback": {
            "cost": {"generic": 1, "U": 1},
            "legal": lambda state: state.life_total >= 3,  # can pay the "Pay 3 life" additional cost
            "resolve": lambda state, card_def: flashback_deep_analysis(state, card_def),
        },
        "pending_kinds": {"choose_target_player"},
    },
    EffectId.COUNTERSPELL: {
        "cast": {
            "resolve": lambda state, card_def: cast_counterspell(state, card_def),
            "extra_legal": lambda state: _has_stack_spell(state, lambda e: True),
            "precast_choice": True,  # target spell chosen at cast
        },
        # No pending_kinds: choose_stack_target is pointer-addressed (rl.decision.action_bridge),
        # not a by-name fixed action.
    },
    EffectId.DISPEL: {
        "cast": {
            "resolve": lambda state, card_def: cast_dispel(state, card_def),
            "extra_legal": lambda state: _has_stack_spell(state, lambda e: e["card_def"].card_type == CardType.INSTANT),
            "precast_choice": True,
        },
    },
    EffectId.SPELL_PIERCE: {
        "cast": {
            "resolve": lambda state, card_def: cast_spell_pierce(state, card_def),
            "extra_legal": lambda state: _has_stack_spell(state, lambda e: e["card_def"].card_type != CardType.CREATURE),
            "precast_choice": True,
        },
        "pending_kinds": {"pay_unless"},
    },
    EffectId.ABANDON_ATTACHMENTS: {
        "cast": {"resolve": lambda state, card_def: cast_abandon_attachments(state, card_def)},
        "pending_kinds": {"discard"},
    },
    EffectId.UTROM_MONITOR: {
        "cast": {"resolve": lambda state, card_def: cast_permanent_from_hand(state, card_def)},
        "cost_reduction": affinity_reduction,  # affinity for artifacts
        "keywords": {"flying"},
    },
    EffectId.THOUGHTCAST: {
        "cast": {"resolve": lambda state, card_def: cast_thoughtcast(state, card_def)},
        "cost_reduction": affinity_reduction,  # affinity for artifacts
    },
    EffectId.TOLARIAN_TERROR: {
        "cast": {"resolve": lambda state, card_def: cast_permanent_from_hand(state, card_def)},
        "cost_reduction": graveyard_instant_sorcery_count,  # {1} less per I/S in graveyard
        # Ward {2}: countering an opponent's targeting spell/ability unless they pay {2}
        # (fired via casting.capture_any_target when an opponent locks this as a target).
        "ward": {"generic": 2},
        "pending_kinds": {"pay_unless"},
    },
    EffectId.CRYPTIC_SERPENT: {
        "cast": {"resolve": lambda state, card_def: cast_permanent_from_hand(state, card_def)},
        "cost_reduction": graveyard_instant_sorcery_count,  # {1} less per I/S in graveyard
    },
    EffectId.DEEM_INFERIOR: {
        "cast": {
            "resolve": lambda state, card_def: cast_deem_inferior(state, card_def),
            # needs >=1 nonland permanent this player can legally target
            "extra_legal": lambda state: any(
                p.card_type != CardType.LAND and can_be_targeted(state, p, state.active_idx)
                for pl in state.players for p in pl.battlefield),
            "precast_choice": True,  # target locked at cast
        },
        "cost_reduction": lambda state: state.cards_drawn_this_turn,  # {1} less per card drawn this turn
        "pending_kinds": {"choose_any_target", "tuck_position"},
    },
    EffectId.DELVER_OF_SECRETS: {
        "cast": {"resolve": lambda state, card_def: cast_permanent_from_hand(state, card_def)},
        "upkeep_trigger": delver_upkeep,
        # Back face: 3/2 flying (execute_may_transform swaps the Permanent's card_def).
        "transform": {"name": "Insectile Aberration", "power": 3, "toughness": 2,
                      "keywords": {"flying"}, "card_def": INSECTILE_ABERRATION_CARD_DEF},
        "pending_kinds": {"may_transform"},
    },
    EffectId.SEWER_VEILLANCE_CAM: {
        "cast": {
            "resolve": lambda state, card_def: cast_permanent_from_hand(state, card_def),
            "speed": Speed.INSTANT,  # Flash
        },
        "etb_trigger": lambda state, permanent: sewer_cam_tap_or_untap(state, permanent.card_def),
        "etb_targets": True,  # target creature chosen at promotion
        "ltb_trigger": lambda state, permanent: sewer_cam_tap_or_untap(state, permanent.card_def),
        "ltb_targets": True,
        "activated_abilities": {
            "draw": {"cost_key": "sac_ability_cost", "resolve": lambda state, permanent: sewer_cam_sac(state, permanent)},
        },
        "pending_kinds": {"choose_any_target"},
    },
    EffectId.BIRD_ILLUSION_TOKEN: {"keywords": {"flying"}},  # 1/1 flyer made by Murmuring Mystic
    EffectId.MURMURING_MYSTIC: {
        "cast": {"resolve": lambda state, card_def: cast_permanent_from_hand(state, card_def)},
        # Whenever you cast an instant or sorcery, create a 1/1 blue Bird Illusion with flying.
        "on_cast": lambda state, permanent: create_token(state, BIRD_ILLUSION_TOKEN_CARD_DEF),
    },
    EffectId.SLEEP_OF_THE_DEAD: {
        "cast": {
            "resolve": lambda state, card_def: cast_sleep_of_the_dead(state, card_def),
            "extra_legal": lambda state: has_creature_target(state),
            "precast_choice": True,  # target creature locked at cast
        },
        "escape": {  # Escape {2}{U}, exile three other graveyard cards (drl_env's "Escape {name}" action)
            "cost": {"generic": 2, "U": 1},
            "legal": lambda state: _sleep_escape_legal(state),
            "resolve": lambda state, card_def: escape_sleep_of_the_dead(state, card_def),
        },
        "pending_kinds": {"choose_any_target", "choose_graveyard_card"},
    },
    EffectId.MENTAL_NOTE: {
        "cast": {"resolve": lambda state, card_def: cast_mental_note(state, card_def)},
    },
    EffectId.THOUGHT_SCOUR: {
        "cast": {
            "resolve": lambda state, card_def: cast_thought_scour(state, card_def),
            "precast_choice": True,  # target player locked at cast
        },
        "pending_kinds": {"choose_target_player"},
    },
    EffectId.LORIEN_REVEALED: {
        "cast": {"resolve": lambda state, card_def: cast_lorien_revealed(state, card_def)},
        "cycle": {  # Islandcycling {1} -- the generic "Cycle {name}" action (drl_env)
            "cost_key": "islandcycling_cost",
            "resolve": lambda state, card_def: islandcycle_lorien_revealed(state, card_def),
        },
        "pending_kinds": {"search_fetch"},
    },
}
