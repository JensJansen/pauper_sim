"""Blue-identity card catalog: every card whose real mana cost is
mono-blue (or, for lands with no cost, whose only mana output is blue).
Same shape as every other color file: a BLUE_CARD_CATALOG dict (name ->
CardDef) and a BLUE_EFFECT_REGISTRY dict (EffectId -> spec), unioned into
game.CARD_DEFS/EFFECT_REGISTRY by game/registry.py. Every cost/type/
oracle-text is a direct Scryfall pull.

Seat of the Synod is an ARTIFACT LAND -- played as a land (card_type LAND,
land-drop path) but also an artifact (extra["artifact"]=True), so it
counts for affinity/metalcraft and is a legal artifact-sacrifice."""

from .. import registry
from ..cards import CardDef, CardType, EffectId
from ..effects.casting import (
    _log_target_fizzle, capture_any_target, cast_permanent_from_hand, cast_targeting_creature, has_creature_target,
    target_still_legal,
)
from ..effects.shared import (
    affinity_reduction, card_subtypes, discard_from_hand_to_graveyard, find_to_hand, graveyard_instant_sorcery_count, mill,
)
from ..effects.stack import counter_spell, push_ability_to_stack, push_to_stack
from ..effects.state_based import sacrifice_to_graveyard
from ..effects.stats import can_be_targeted, controller_idx
from ..effects.tokens import BIRD_ILLUSION_TOKEN_CARD_DEF, create_token
from ..effects.win_check import lose_life
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
    # {1}{U/R} -- modeled {1}{U}: the hybrid's red half is unreachable in
    # dmir_terror (a U/B deck with no red source), same authorized
    # simplification as Slippery Bogle's {G/U}->{G} (real cost per Scryfall).
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


def cast_deem_inferior(state, card_def):
    """{3}{U} (costs {1} less per card drawn this turn): "The owner of target
    NONLAND PERMANENT puts it into their library second from the top or on the
    bottom." Target ANY nonland permanent on either battlefield (creature,
    artifact, enchantment -- hexproof/shroud aware, Ward-aware via
    capture_any_target), locked at cast; on resolution its owner tucks it (their
    choice of position), or the spell fizzles if it has left the battlefield."""
    idx = state.active_idx

    def _on_target(state, descriptor):
        captured = capture_any_target(state, descriptor)

        def _resolve(state, card_def):
            discard_from_hand_to_graveyard(state, card_def)  # Deem Inferior itself -> graveyard
            if not target_still_legal(state, captured):
                where = (captured[1].card_def.name, captured[1].slot) if captured is not None else None
                _log_target_fizzle(state, card_def, where)
                return
            permanent = captured[1]
            owner_idx = controller_idx(state, permanent)  # controller == owner (no control-changing in this pool)
            state.players[owner_idx].battlefield.remove(permanent)
            begin_tuck_to_library(state, permanent.card_def, owner_idx)

        push_to_stack(state, card_def, _resolve)

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
    """{1}{U}: You may discard a card. If you do, draw two cards. (Same
    "may discard, if you do draw" shape as Highway Robbery / Melded Moxite.)"""
    discard_from_hand_to_graveyard(state, card_def)  # the instant itself -> graveyard

    def _on_discard(state, discarded_cards):
        if discarded_cards:  # a card was actually discarded ("if you do")
            state.draw(2)

    begin_discard(state, 1, optional=True, on_complete=_on_discard)


def _sleep_tap_skip(state, permanent):
    """Tap the target creature; it doesn't untap during its controller's
    next untap step (untap_step consumes the skip_next_untap flag)."""
    permanent.tapped = True
    permanent.flags["skip_next_untap"] = True


def cast_sleep_of_the_dead(state, card_def):
    """{U}: Tap target creature; skip its next untap. Reuses the shared
    single-target-creature primitive (target locked at cast, fizzles if
    gone)."""
    cast_targeting_creature(state, card_def, _sleep_tap_skip)


def _exile_n_other_from_graveyard(state, n, exclude, on_complete):
    """Escape's additional cost: the model exiles n cards from its own
    graveyard OTHER than `exclude` (the card being escaped), one at a time
    (chained begin_choose_graveyard_card). Runs on_complete(state) once n are
    exiled. NOTE: excludes `exclude` by identity -- CardDefs are interned per
    name, so a second copy of the same card can't be used to pay (a benign,
    conservative limitation, never wrongly permissive)."""
    def _step(remaining):
        if remaining == 0:
            on_complete(state)
            return

        def _chosen(state, name):
            found = next(c for c in state.graveyard if c.name == name)
            state.graveyard.remove(found)  # exiled, untracked
            state.log_event("zone_move", card=found.name, from_zone="graveyard", to_zone="exile_untracked", reason="escape")
            _step(remaining - 1)

        begin_choose_graveyard_card(state, lambda c: c is not exclude, _chosen)

    _step(n)


def _sleep_escape_legal(state):
    """Escape {2}{U}, exile three OTHER cards from your graveyard: legal only
    with 3+ other graveyard cards to exile AND a legal creature to tap
    (a targeted spell needs a target). Mana + Sleep-in-graveyard are checked
    by the generic graveyard-cast machinery (drl_env._flashback_legal)."""
    sleep_def = registry.CARD_DEFS["Sleep of the Dead"]
    others = sum(1 for c in state.graveyard if c is not sleep_def)
    return others >= 3 and has_creature_target(state)


def escape_sleep_of_the_dead(state, card_def):
    """Escape: {2}{U} (paid by the graveyard-cast machinery) + exile three
    other graveyard cards, then Sleep leaves the graveyard (escaping; exiled
    after it resolves) and taps a target creature (chosen at cast, fizzles if
    gone)."""
    def _after_exile(state):
        state.graveyard.remove(card_def)  # escapes the graveyard; exiled after resolution (untracked)
        idx = state.active_idx

        def _on_target(state, descriptor):
            captured = capture_any_target(state, descriptor)

            def _resolve(state, cd):
                if not target_still_legal(state, captured):
                    _log_target_fizzle(state, cd, None)
                    return
                _sleep_tap_skip(state, captured[1])

            push_to_stack(state, card_def, _resolve, reserves_hand_card=False, is_spell=True)

        begin_choose_any_target(
            state, lambda p: p.card_type == CardType.CREATURE and can_be_targeted(state, p, idx),
            _on_target, allow_players=False,
        )

    _exile_n_other_from_graveyard(state, 3, card_def, _after_exile)


def sewer_cam_tap_or_untap(state, source_card_def):
    """Sewer-veillance Cam ETB/LTB: "you may tap or untap target creature."
    Optional target (either side, hexproof/shroud-aware); on resolution it
    TOGGLES the target's tapped state -- observationally identical to the
    real tap-or-untap choice, since the no-op direction (tapping a tapped or
    untapping an untapped creature) is never the meaningful pick. The effect
    waits on the stack (a triggered ability), fizzling if the target's gone."""
    idx = state.active_idx

    def _on_target(state, descriptor):
        captured = capture_any_target(state, descriptor)

        def _resolve(st, cd):
            if captured is None or not target_still_legal(st, captured):
                return
            perm = captured[1]
            perm.tapped = not perm.tapped
            st.log_event("tap_or_untap", permanent=(perm.card_def.name, perm.slot), now_tapped=perm.tapped)

        push_to_stack(state, source_card_def, _resolve, reserves_hand_card=False, is_spell=False)

    begin_choose_any_target(
        state, lambda p: p.card_type == CardType.CREATURE and can_be_targeted(state, p, idx),
        _on_target, allow_players=False, optional=True,  # "you MAY"
    )


def sewer_cam_sac(state, permanent):
    """{3}{U}, Sacrifice this artifact: Draw two cards. Sacrificing also fires
    its own LTB (tap/untap), via sacrifice_to_graveyard's ltb queue."""
    sacrifice_to_graveyard(state, permanent)  # queues the LTB tap/untap
    push_ability_to_stack(state, permanent.card_def, lambda st: st.draw(2))


def _has_stack_spell(state, predicate):
    return any(e.get("is_spell") and predicate(e) for e in state.stack)


def _cast_counter(state, card_def, predicate, on_countered=None):
    """Shared body for a counterspell: choose a matching spell on the stack
    (locked at cast, precast_choice), then push this counter spell's own
    effect. On resolution the counter spell goes to the graveyard and, if the
    chosen spell is still on the stack (else the counter fizzles, 608.2b), it
    is countered -- or, for a rider (Spell Pierce), on_countered(entry)
    decides via begin_pay_unless whether it actually gets countered."""
    def _on_target(state, entry):
        def _resolve(state, card_def):
            discard_from_hand_to_graveyard(state, card_def)  # the counter spell itself -> graveyard
            if entry is None or entry not in state.stack:
                _log_target_fizzle(state, card_def, None)  # the target spell already left the stack
                return
            if on_countered is None:
                counter_spell(state, entry)
            else:
                on_countered(state, entry)

        push_to_stack(state, card_def, _resolve)

    begin_choose_stack_target(state, predicate, _on_target)


def cast_counterspell(state, card_def):
    """{U}{U}: Counter target spell."""
    _cast_counter(state, card_def, lambda e: True)


def cast_dispel(state, card_def):
    """{U}: Counter target instant spell."""
    _cast_counter(state, card_def, lambda e: e["card_def"].card_type == CardType.INSTANT)


def _spell_pierce_countered(state, entry):
    """"unless its controller pays {2}": the spell's controller may pay {2};
    if they do, it is NOT countered."""
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
    """"an Island card" -- any card with the Island land subtype (basic
    Island, Contaminated Aquifer, Ice Tunnel), matching Islandcycling's real
    search."""
    return "Island" in card_subtypes(card_def)


def cast_mental_note(state, card_def):
    """{U}: Mill two cards. Draw a card. Both halves affect YOU (no target).
    The spell moves itself to the graveyard as it resolves, before the mill/
    draw, same convention as every other instant/sorcery here."""
    discard_from_hand_to_graveyard(state, card_def)
    mill(state, 2)
    state.draw(1)


def cast_thought_scour(state, card_def):
    """{U}: Target player mills two cards. Draw a card (you). The target
    player is chosen when this resolves rather than at cast: a player target
    can never become an illegal target (players don't leave the game here),
    so there is NO observable difference vs locking it at cast -- the same
    "no observable difference in this solitaire sim" treatment
    begin_choose_graveyard_card / Relic of Progenitus' targeted ability
    already use. Whoever is milled, the caster (active player) draws."""
    discard_from_hand_to_graveyard(state, card_def)

    def _on_player(state, idx):
        mill(state, 2, idx)
        state.draw(1)  # the caster draws -- active_idx is the controller throughout (no flip)

    begin_choose_target_player(state, _on_player)


def cast_lorien_revealed(state, card_def):
    """{3}{U}{U}: Draw three cards."""
    discard_from_hand_to_graveyard(state, card_def)
    state.draw(3)


def islandcycle_lorien_revealed(state, card_def):
    """Islandcycling {1}: discard this card from hand, search library for an
    Island card, put it into hand, shuffle. A real model choice among the
    Island-subtype cards present (basic Island / Contaminated Aquifer / Ice
    Tunnel), unlike Generous Ent's single fixed "Forest" -- same begin_search_
    fetch-to-hand shape as Ash Barrens' basic landcycling."""
    discard_from_hand_to_graveyard(state, card_def)
    begin_search_fetch(state, _is_island_card, find_to_hand)


def cast_brainstorm(state, card_def):
    """{U}: Draw three cards, then put two cards from your hand on top of your
    library in any order (begin_put_on_top_from_hand -- the model picks which
    two and their order)."""
    discard_from_hand_to_graveyard(state, card_def)
    state.draw(3)
    begin_put_on_top_from_hand(state, 2, on_complete=lambda s: None)


def cast_ponder(state, card_def):
    """{U}: Look at the top three cards, put them back in any order OR shuffle
    (begin_ponder), then draw a card (the on_complete, run either way)."""
    discard_from_hand_to_graveyard(state, card_def)
    begin_ponder(state, on_complete=lambda s: s.draw(1))


def _target_player_draws(state, idx, n):
    """Make player `idx` draw n cards. Flips active_idx to them so
    GameState.draw hits THEIR library/hand -- and, if that draw decks them
    out, LEAVES active_idx pointing at them so the turn generator's DeckedOut
    handler awards the win to the other player (winner = 1 - active_idx).
    active_idx is restored only on the normal, non-deck-out path (the restore
    line simply isn't reached if draw() raised DeckedOut)."""
    saved = state.active_idx
    state.active_idx = idx
    state.draw(n)  # DeckedOut (idx had < n cards) propagates with active_idx == idx -- correct attribution
    state.active_idx = saved


def _deep_analysis_effect(state, card_def):
    """Target player draws two cards. The target player is chosen at
    resolution rather than locked at cast -- a player can never become an
    illegal target here (players don't leave), so there's no observable
    difference, same treatment as Thought Scour / begin_choose_graveyard_
    card. Whoever is chosen draws two (cross-player via _target_player_draws)."""
    def _on_player(state, idx):
        _target_player_draws(state, idx, 2)

    begin_choose_target_player(state, _on_player)


def cast_deep_analysis(state, card_def):
    """{3}{U}: Target player draws two cards. (Normal cast -- already on the
    stack via the generic cast path, so the effect runs inline at
    resolution.)"""
    discard_from_hand_to_graveyard(state, card_def)
    _deep_analysis_effect(state, card_def)


def flashback_deep_analysis(state, card_def):
    """Flashback -- {1}{U}, Pay 3 life. The {1}{U} was already paid by the
    generic mana-flashback path (drl_env._flashback_execute's begin_pay_cost);
    pay the 3-life additional cost now, remove this card from the graveyard
    (it leaves the moment Flashback is chosen; exiled afterward -- untracked,
    Dread Return's precedent), then push the same target-player-draws-two
    effect onto the stack. Life >= 3 is enforced by the flashback's own
    legal predicate."""
    state.graveyard.remove(card_def)
    lose_life(state, 3, reason="deep_analysis_flashback")
    push_to_stack(state, card_def, _deep_analysis_effect, reserves_hand_card=False)


def delver_upkeep(state, permanent):
    """Delver of Secrets -- "At the beginning of your upkeep, look at the top
    card of your library. You may reveal that card. If an instant or sorcery
    card is revealed this way, transform Delver of Secrets."

    Looking + the may-reveal are private information with no observable game
    effect on their own, so they collapse to: if the top card is an instant or
    sorcery, offer the transform choice (begin_may_transform); otherwise
    nothing happens. Empty library -> nothing to look at."""
    if not state.library:
        return
    top = state.library[0]
    if top.card_type in (CardType.INSTANT, CardType.SORCERY):
        begin_may_transform(state, permanent)


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
        "cast": {"resolve": lambda state, card_def: cast_deep_analysis(state, card_def)},
        "flashback": {
            "cost": {"generic": 1, "U": 1},  # the {1}{U} mana half -- paid by the generic mana-flashback path
            "legal": lambda state: state.life_total >= 3,  # can pay the "Pay 3 life" additional cost
            "resolve": lambda state, card_def: flashback_deep_analysis(state, card_def),
        },
        "pending_kinds": {"choose_target_player"},
    },
    EffectId.COUNTERSPELL: {
        "cast": {
            "resolve": lambda state, card_def: cast_counterspell(state, card_def),
            "extra_legal": lambda state: _has_stack_spell(state, lambda e: True),  # a spell on the stack to counter
            "precast_choice": True,  # target spell chosen at cast (from the current stack)
        },
        "pending_kinds": {"choose_stack_target"},
    },
    EffectId.DISPEL: {
        "cast": {
            "resolve": lambda state, card_def: cast_dispel(state, card_def),
            "extra_legal": lambda state: _has_stack_spell(state, lambda e: e["card_def"].card_type == CardType.INSTANT),
            "precast_choice": True,
        },
        "pending_kinds": {"choose_stack_target"},
    },
    EffectId.SPELL_PIERCE: {
        "cast": {
            "resolve": lambda state, card_def: cast_spell_pierce(state, card_def),
            "extra_legal": lambda state: _has_stack_spell(state, lambda e: e["card_def"].card_type != CardType.CREATURE),
            "precast_choice": True,
        },
        "pending_kinds": {"choose_stack_target", "pay_unless"},  # pay_unless: the "unless controller pays {2}" rider
    },
    EffectId.ABANDON_ATTACHMENTS: {
        "cast": {"resolve": lambda state, card_def: cast_abandon_attachments(state, card_def)},
        "pending_kinds": {"discard"},
    },
    EffectId.UTROM_MONITOR: {
        "cast": {"resolve": lambda state, card_def: cast_permanent_from_hand(state, card_def)},
        "cost_reduction": affinity_reduction,  # Affinity for artifacts
        "keywords": {"flying"},
    },
    EffectId.THOUGHTCAST: {
        "cast": {"resolve": lambda state, card_def: cast_thoughtcast(state, card_def)},
        "cost_reduction": affinity_reduction,  # Affinity for artifacts
    },
    EffectId.TOLARIAN_TERROR: {
        "cast": {"resolve": lambda state, card_def: cast_permanent_from_hand(state, card_def)},
        "cost_reduction": graveyard_instant_sorcery_count,  # {1} less per I/S in graveyard
        # Ward {2}: "Whenever this becomes the target of a spell or ability an
        # opponent controls, counter it unless that player pays {2}." Fired by
        # casting.capture_any_target (the universal creature-targeting choke
        # point) when an OPPONENT locks it as a target -> a "ward" trigger
        # (game.effects.triggers) that, above the triggering spell on the
        # stack, makes that opponent pay {2} (begin_pay_unless) or the spell is
        # countered. pending_kinds "pay_unless" for the opponent's pay decision.
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
            # A targeted spell can't be cast with no legal target: needs >=1
            # nonland permanent this player can target (hexproof/shroud aware).
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
        # Back face Insectile Aberration: 3/2 flying (flag-gated in effects.stats
        # via permanent.flags["transformed"], set by execute_may_transform).
        "transform": {"power": 3, "toughness": 2, "keywords": {"flying"}},
        "pending_kinds": {"may_transform"},
    },
    EffectId.SEWER_VEILLANCE_CAM: {
        "cast": {
            "resolve": lambda state, card_def: cast_permanent_from_hand(state, card_def),
            "speed": Speed.INSTANT,  # Flash
        },
        "etb_trigger": lambda state, permanent: sewer_cam_tap_or_untap(state, permanent.card_def),
        "ltb_trigger": lambda state, permanent: sewer_cam_tap_or_untap(state, permanent.card_def),
        "activated_abilities": {
            "draw": {"cost_key": "sac_ability_cost", "resolve": lambda state, permanent: sewer_cam_sac(state, permanent)},
        },
        "pending_kinds": {"choose_any_target"},
    },
    EffectId.BIRD_ILLUSION_TOKEN: {"keywords": {"flying"}},  # 1/1 flyer made by Murmuring Mystic
    EffectId.MURMURING_MYSTIC: {
        "cast": {"resolve": lambda state, card_def: cast_permanent_from_hand(state, card_def)},
        # "Whenever you cast an instant or sorcery spell, create a 1/1 blue
        # Bird Illusion creature token with flying." Same on_cast chokepoint
        # as Guttersnipe (fires for every cast path, faithful timing).
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
        "cast": {"resolve": lambda state, card_def: cast_thought_scour(state, card_def)},
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


if __name__ == "__main__":
    # ponytail self-check: run via `python -m game.catalog.blue_cards` from src/.
    from ..resolution import execute_choose_target_player_option, execute_search_fetch_option, search_fetch_options
    from ..state import GameState, Permanent, PlayerState

    # Mental Note {U}: mill 2 (self), draw 1. The spell moves itself to the
    # graveyard as it resolves, ahead of the mill/draw.
    state = GameState(on_the_play=True)
    state.hand = [CardDef("Mental Note", CardType.INSTANT, {"U": 1}, EffectId.MENTAL_NOTE)]
    state.library = [
        CardDef("A", CardType.LAND, None, EffectId.ISLAND), CardDef("B", CardType.LAND, None, EffectId.ISLAND),
        CardDef("C", CardType.LAND, None, EffectId.ISLAND),
    ]
    cast_mental_note(state, state.hand[0])
    assert [c.name for c in state.graveyard] == ["Mental Note", "A", "B"]  # itself + 2 milled
    assert [c.name for c in state.hand] == ["C"]  # drew the 3rd
    print("blue_cards.py Mental Note self-check: OK")

    # Thought Scour {U}: target player mills 2, caster draws 1. Targeting the
    # OPPONENT mills THEIR library; the active player still draws.
    state = GameState(on_the_play=True, players=[PlayerState(True), PlayerState(False)])
    state.players[0].hand = [CardDef("Thought Scour", CardType.INSTANT, {"U": 1}, EffectId.THOUGHT_SCOUR)]
    state.players[0].library = [CardDef("MyCard", CardType.LAND, None, EffectId.ISLAND)]
    state.players[1].library = [
        CardDef("Opp1", CardType.LAND, None, EffectId.ISLAND), CardDef("Opp2", CardType.LAND, None, EffectId.ISLAND),
        CardDef("Opp3", CardType.LAND, None, EffectId.ISLAND),
    ]
    cast_thought_scour(state, state.players[0].hand[0])
    assert state.pending_resolution["kind"] == "choose_target_player"
    execute_choose_target_player_option(state, 1)  # target the opponent
    assert [c.name for c in state.players[1].graveyard] == ["Opp1", "Opp2"]  # THEIR top 2 milled
    assert [c.name for c in state.players[0].hand] == ["MyCard"]  # the caster drew, not the opponent
    assert state.active_idx == 0  # no stray active_idx flip
    print("blue_cards.py Thought Scour (target opponent) self-check: OK")

    # Lórien Revealed: cast draws three; Islandcycling searches an Island-
    # subtype card (basic Island / Contaminated Aquifer / Ice Tunnel), not a
    # Mountain.
    state = GameState(on_the_play=True)
    lorien = CardDef(
        "Lórien Revealed", CardType.SORCERY, {"generic": 3, "U": 2}, EffectId.LORIEN_REVEALED,
        islandcycling_cost={"generic": 1},
    )
    state.hand = [lorien]
    state.library = [CardDef(n, CardType.LAND, None, EffectId.ISLAND) for n in ("d1", "d2", "d3", "d4")]
    cast_lorien_revealed(state, lorien)
    assert len(state.hand) == 3  # drew 3

    state = GameState(on_the_play=True)
    state.hand = [lorien]
    state.library = [
        CardDef("Ice Tunnel", CardType.LAND, None, EffectId.ICE_TUNNEL, subtypes=("Island", "Swamp")),
        CardDef("Mountain", CardType.LAND, None, EffectId.MOUNTAIN, basic=True, subtypes=("Mountain",)),
        CardDef("Island", CardType.LAND, None, EffectId.ISLAND, basic=True, subtypes=("Island",)),
    ]
    islandcycle_lorien_revealed(state, lorien)
    assert state.pending_resolution["kind"] == "search_fetch"
    assert sorted(search_fetch_options(state)) == ["Ice Tunnel", "Island"]  # Mountain excluded (no Island subtype)
    execute_search_fetch_option(state, "Ice Tunnel")
    assert any(c.name == "Ice Tunnel" for c in state.hand)
    assert [c.name for c in state.graveyard] == ["Lórien Revealed"]  # discarded itself
    print("blue_cards.py Lórien Revealed (draw 3 + Islandcycling) self-check: OK")

    # Brainstorm {U}: draw 3, then put 2 hand cards on top in a chosen order
    # (first placed ends on top).
    from ..effects.stack import resolve_top_of_stack as _resolve_top
    from ..resolution import (
        execute_ponder_option as _ep,
        execute_ponder_shuffle as _eps,
        execute_put_on_top_option as _eput,
    )

    def _card(name, ct=CardType.LAND, eid=EffectId.ISLAND):
        return CardDef(name, ct, None, eid)

    state = GameState(on_the_play=True)
    state.hand = [CardDef("Brainstorm", CardType.INSTANT, {"U": 1}, EffectId.BRAINSTORM)]
    state.library = [_card("L1"), _card("L2"), _card("L3"), _card("L4")]
    cast_brainstorm(state, state.hand[0])
    assert state.pending_resolution["kind"] == "put_on_top"  # after drawing 3
    assert len(state.hand) == 3  # L1, L2, L3 (Brainstorm went to gy)
    _eput(state, "L3")  # L3 on top
    _eput(state, "L1")  # then L1
    assert state.pending_resolution is None
    assert [c.name for c in state.library[:2]] == ["L3", "L1"]  # first-placed on top
    assert len(state.hand) == 1 and state.hand[0].name == "L2"
    print("blue_cards.py Brainstorm self-check: OK")

    # Brainstorm can NOT put a spell that's currently on the stack (reserved,
    # still physically in the hand list) back on top of the library -- it's on
    # the stack, not in hand (real Magic). Regression for the reserved-hand-
    # card bug (Gurmag Angler mid-cast + Brainstorm + Mental Note crash).
    from ..effects.stack import push_to_stack as _push_reserved
    from ..resolution import put_on_top_options as _put_opts
    state = GameState(on_the_play=True)
    reserved_spell = CardDef("Gurmag Angler", CardType.CREATURE, {"generic": 6, "B": 1}, EffectId.FILLER, power=5, toughness=5)
    state.hand = [CardDef("Brainstorm", CardType.INSTANT, {"U": 1}, EffectId.BRAINSTORM), reserved_spell]
    _push_reserved(state, reserved_spell, lambda s, c: None)  # a spell cast in response, reserved on the stack
    state.library = [_card(n) for n in ("BL1", "BL2", "BL3", "BL4", "BL5")]
    cast_brainstorm(state, state.hand[0])
    assert state.pending_resolution["kind"] == "put_on_top"
    opts = _put_opts(state)
    assert "Gurmag Angler" not in opts, opts  # reserved on the stack -> not a legal put-on-top pick
    assert {"BL1", "BL2", "BL3"} == set(opts)  # the three genuinely-in-hand drawn cards are
    print("blue_cards.py Brainstorm reserved-card exclusion self-check: OK")

    # Ponder {U}: order the top 3, then draw the new top; OR shuffle then draw.
    state = GameState(on_the_play=True)
    state.hand = [CardDef("Ponder", CardType.SORCERY, {"U": 1}, EffectId.PONDER)]
    state.library = [_card("P1"), _card("P2"), _card("P3"), _card("P4")]
    cast_ponder(state, state.hand[0])
    assert state.pending_resolution["kind"] == "ponder"
    _ep(state, "P3")  # P3 to top (will be drawn)
    _ep(state, "P1")
    _ep(state, "P2")
    assert state.pending_resolution is None
    assert state.hand[0].name == "P3"  # drew the card placed on top
    assert [c.name for c in state.library[:2]] == ["P1", "P2"]  # rest below in chosen order

    state = GameState(on_the_play=True)
    state.hand = [CardDef("Ponder", CardType.SORCERY, {"U": 1}, EffectId.PONDER)]
    state.library = [_card(f"S{i}") for i in range(6)]
    cast_ponder(state, state.hand[0])
    _eps(state)  # shuffle instead of ordering
    assert state.pending_resolution is None and len(state.hand) == 1  # shuffled + drew 1
    print("blue_cards.py Ponder (order + shuffle) self-check: OK")

    # Deep Analysis {3}{U}: target player draws two. Flashback pays 3 life +
    # {1}{U} (mana paid by the generic path); the 3-life cost is paid here and
    # the effect goes on the stack.
    state = GameState(on_the_play=True)
    state.hand = [CardDef("Deep Analysis", CardType.SORCERY, {"generic": 3, "U": 1}, EffectId.DEEP_ANALYSIS)]
    state.library = [_card(f"D{i}") for i in range(4)]
    cast_deep_analysis(state, state.hand[0])
    assert state.pending_resolution["kind"] == "choose_target_player"
    execute_choose_target_player_option(state, 0)
    assert len(state.hand) == 2

    state = GameState(on_the_play=True)
    da = CardDef("Deep Analysis", CardType.SORCERY, {"generic": 3, "U": 1}, EffectId.DEEP_ANALYSIS)
    state.graveyard = [da]
    state.library = [_card(f"F{i}") for i in range(4)]
    state.players[0].life_total = 10
    flashback_deep_analysis(state, da)  # mana already paid upstream in real play
    assert da not in state.graveyard and state.life_total == 7 and len(state.stack) == 1
    _resolve_top(state)
    assert state.pending_resolution["kind"] == "choose_target_player"
    execute_choose_target_player_option(state, 0)
    assert len(state.hand) == 2
    print("blue_cards.py Deep Analysis (cast + flashback pay-3-life) self-check: OK")

    # --- G4 counters & permission ---
    from ..effects.stack import push_to_stack as _push
    from ..resolution import (
        execute_choose_stack_target_option, pay_unless_decline, pay_unless_pay,
    )

    def _stack_spell(st, name, controller=1):
        """Put a real spell on the stack as if `controller` cast it (its card
        physically in their hand, reserved -- what a normal cast looks like)."""
        cd = CardDef(name, CardType.INSTANT if name != "Ponder" else CardType.SORCERY, {"U": 1}, EffectId.FILLER)
        st.players[controller].hand.append(cd)
        saved = st.active_idx
        st.active_idx = controller
        _push(st, cd, lambda s, c: None, reserves_hand_card=True, is_spell=True)
        st.active_idx = saved
        return cd

    # Counterspell counters any spell -> the countered spell to its
    # controller's graveyard.
    state = GameState(on_the_play=True, players=[PlayerState(True), PlayerState(False)])
    tgt = _stack_spell(state, "Some Instant", controller=1)
    state.active_idx = 0
    _cs = CardDef("Counterspell", CardType.INSTANT, {"U": 2}, EffectId.COUNTERSPELL)
    state.players[0].hand = [_cs]  # the counter spell is in hand while on the stack (reserved), removed on resolve
    cast_counterspell(state, _cs)
    assert state.pending_resolution["kind"] == "choose_stack_target"
    execute_choose_stack_target_option(state, "Some Instant")
    _resolve_top(state)
    assert all(e["card_def"].name != "Some Instant" for e in state.stack)  # countered off the stack
    assert tgt in state.players[1].graveyard
    print("blue_cards.py Counterspell self-check: OK")

    # Dispel: an instant is a legal target; a sorcery is not.
    state = GameState(on_the_play=True, players=[PlayerState(True), PlayerState(False)])
    _stack_spell(state, "Ponder", controller=1)  # sorcery
    state.active_idx = 0
    assert not _has_stack_spell(state, lambda e: e["card_def"].card_type == CardType.INSTANT)
    _stack_spell(state, "An Instant", controller=1)
    assert _has_stack_spell(state, lambda e: e["card_def"].card_type == CardType.INSTANT)
    print("blue_cards.py Dispel legality (instant-only) self-check: OK")

    # Spell Pierce: controller declines to pay {2} -> countered; pays -> survives.
    state = GameState(on_the_play=True, players=[PlayerState(True), PlayerState(False)])
    tgt = _stack_spell(state, "Ponder", controller=1)  # noncreature
    state.active_idx = 0
    _sp = CardDef("Spell Pierce", CardType.INSTANT, {"U": 1}, EffectId.SPELL_PIERCE)
    state.players[0].hand = [_sp]
    cast_spell_pierce(state, _sp)
    execute_choose_stack_target_option(state, "Ponder")
    _resolve_top(state)
    assert state.pending_resolution["kind"] == "pay_unless" and state.active_idx == 1  # payer's decision
    pay_unless_decline(state)
    assert tgt in state.players[1].graveyard and state.active_idx == 0  # not paid -> countered, active_idx restored

    state = GameState(on_the_play=True, players=[PlayerState(True), PlayerState(False)])
    tgt = _stack_spell(state, "Ponder", controller=1)
    state.players[1].mana_pool = {"U": 2}  # can afford the {2}
    state.active_idx = 0
    _sp2 = CardDef("Spell Pierce", CardType.INSTANT, {"U": 1}, EffectId.SPELL_PIERCE)
    state.players[0].hand = [_sp2]
    cast_spell_pierce(state, _sp2)
    execute_choose_stack_target_option(state, "Ponder")
    _resolve_top(state)
    pay_unless_pay(state)
    _guard = 0
    while state.pending_resolution is not None and state.pending_resolution["kind"] == "pay_cost":
        _guard += 1
        assert _guard < 10
        from ..mana import execute_pool_spend, pool_spend_options
        execute_pool_spend(state, pool_spend_options(state)[0])
    assert tgt not in state.players[1].graveyard  # paid -> NOT countered
    assert any(e["card_def"].name == "Ponder" for e in state.stack) and state.active_idx == 0
    print("blue_cards.py Spell Pierce (decline counters / pay survives) self-check: OK")

    # --- G5 ---
    from ..resolution import execute_choose_any_target_creature, execute_choose_graveyard_card_option, execute_discard_option
    from ..turn import untap_step as _untap

    # Abandon Attachments: may discard a card; if you do, draw two.
    state = GameState(on_the_play=True)
    aa = CardDef("Abandon Attachments", CardType.INSTANT, {"generic": 1, "U": 1}, EffectId.ABANDON_ATTACHMENTS)
    state.hand = [aa, CardDef("Spare", CardType.LAND, None, EffectId.ISLAND)]
    state.library = [CardDef(n, CardType.LAND, None, EffectId.ISLAND) for n in ("x", "y", "z")]
    cast_abandon_attachments(state, aa)
    assert state.pending_resolution["kind"] == "discard"
    execute_discard_option(state, "Spare")
    assert len(state.hand) == 2  # discarded Spare, drew 2
    print("blue_cards.py Abandon Attachments self-check: OK")

    # Sleep of the Dead main cast: tap target + it skips its next untap.
    state = GameState(on_the_play=True, players=[PlayerState(True), PlayerState(False)])
    victim = Permanent(CardDef("Victim", CardType.CREATURE, None, EffectId.FILLER, power=1, toughness=1))
    victim.slot = 1
    state.players[1].battlefield = [victim]
    sd = CardDef("Sleep of the Dead", CardType.SORCERY, {"U": 1}, EffectId.SLEEP_OF_THE_DEAD)
    state.players[0].hand = [sd]
    state.active_idx = 0
    cast_sleep_of_the_dead(state, sd)
    execute_choose_any_target_creature(state, 1, "Victim", 1)
    _resolve_top(state)
    assert victim.tapped and victim.flags.get("skip_next_untap")
    state.active_idx = 1
    state.turn_number = 2
    _untap(state)  # its controller's untap -- stays tapped, skip consumed
    assert victim.tapped and not victim.flags.get("skip_next_untap")
    print("blue_cards.py Sleep of the Dead (tap + skip next untap) self-check: OK")

    # Sleep of the Dead Escape: exile 3 other graveyard cards + tap; the card
    # escapes the graveyard.
    state = GameState(on_the_play=True, players=[PlayerState(True), PlayerState(False)])
    sd = registry.CARD_DEFS["Sleep of the Dead"]
    state.players[0].graveyard = [sd, CardDef("g1", CardType.INSTANT, {"U": 1}, EffectId.FILLER),
                                  CardDef("g2", CardType.INSTANT, {"U": 1}, EffectId.FILLER),
                                  CardDef("g3", CardType.INSTANT, {"U": 1}, EffectId.FILLER)]
    victim = Permanent(CardDef("Victim", CardType.CREATURE, None, EffectId.FILLER, power=1, toughness=1))
    victim.slot = 1
    state.players[1].battlefield = [victim]
    state.active_idx = 0
    assert _sleep_escape_legal(state)
    escape_sleep_of_the_dead(state, sd)
    for nm in ("g1", "g2", "g3"):
        assert state.pending_resolution["kind"] == "choose_graveyard_card"
        execute_choose_graveyard_card_option(state, nm)
    assert state.pending_resolution["kind"] == "choose_any_target"
    execute_choose_any_target_creature(state, 1, "Victim", 1)
    _resolve_top(state)
    assert victim.tapped and victim.flags.get("skip_next_untap")
    assert sd not in state.players[0].graveyard and state.players[0].graveyard == []  # 3 exiled + escaped
    print("blue_cards.py Sleep of the Dead (Escape) self-check: OK")

    # --- G6: Murmuring Mystic -- casting an instant/sorcery makes a 1/1
    # flying Bird Illusion; a land cast does not.
    from ..effects.stack import on_cast_trigger
    from ..effects.stats import has_keyword as _hk
    from ..effects.triggers import promote_triggers_to_stack as _promote2

    state = GameState(on_the_play=True)
    mystic = Permanent(registry.CARD_DEFS["Murmuring Mystic"])
    mystic.slot = 1
    state.battlefield = [mystic]
    on_cast_trigger(state, CardDef("Some Instant", CardType.INSTANT, {"U": 1}, EffectId.FILLER))
    _promote2(state)
    while state.stack:
        _resolve_top(state)
    birds = [p for p in state.battlefield if p.card_def.name == "Bird Illusion"]
    assert len(birds) == 1 and _hk(state, birds[0], "flying")
    on_cast_trigger(state, CardDef("Some Land", CardType.LAND, None, EffectId.FILLER))
    assert not state.trigger_queue  # a land cast doesn't trigger it
    print("blue_cards.py Murmuring Mystic self-check: OK")

    # --- G8: Sewer-veillance Cam -- ETB may tap/untap (toggle) target
    # creature; {3}{U},Sac -> draw 2.
    from ..effects.casting import cast_permanent_from_hand as _cpfh
    from ..effects.triggers import promote_triggers_to_stack as _prom8
    from ..resolution import execute_choose_any_target_creature, execute_choose_any_target_decline

    def _drive8(s):
        _prom8(s)
        while s.stack:
            _resolve_top(s)

    state = GameState(on_the_play=True, players=[PlayerState(True), PlayerState(False)])
    state.active_idx = 0
    opp = Permanent(CardDef("Opp", CardType.CREATURE, None, EffectId.FILLER, power=2, toughness=2))
    opp.slot = 1
    opp.tapped = False
    state.players[1].battlefield = [opp]
    state.players[0].hand = [registry.CARD_DEFS["Sewer-veillance Cam"]]
    _cpfh(state, registry.CARD_DEFS["Sewer-veillance Cam"])
    _drive8(state)  # ETB opens the optional tap/untap target choice
    assert state.pending_resolution["kind"] == "choose_any_target"
    execute_choose_any_target_creature(state, 1, "Opp", 1)
    _drive8(state)
    assert opp.tapped  # toggled untapped -> tapped
    cam = next(p for p in state.players[0].battlefield if p.card_def.name == "Sewer-veillance Cam")
    state.players[0].library = [CardDef(f"d{i}", CardType.LAND, None, EffectId.ISLAND, basic=True) for i in range(3)]
    sewer_cam_sac(state, cam)
    _drive8(state)
    if state.pending_resolution is not None and state.pending_resolution["kind"] == "choose_any_target":
        execute_choose_any_target_decline(state)  # decline the LTB toggle
        _drive8(state)
    assert len(state.players[0].hand) == 2  # drew 2
    print("blue_cards.py Sewer-veillance Cam (toggle + sac draw 2) self-check: OK")

    # --- G7 ---
    import drl_env
    from ..resolution import execute_choose_any_target_creature as _ect, execute_tuck_position

    # Cost reduction is per the CASTER only. Tolarian Terror {6}{U}: -1 per
    # instant/sorcery in the caster's OWN graveyard; the opponent's graveyard
    # is never counted.
    state = GameState(on_the_play=True, players=[PlayerState(True), PlayerState(False)])
    state.active_idx = 0
    state.players[0].graveyard = [CardDef("A", CardType.INSTANT, {"U": 1}, EffectId.FILLER),
                                  CardDef("B", CardType.SORCERY, {"U": 1}, EffectId.FILLER),
                                  CardDef("L", CardType.LAND, None, EffectId.ISLAND)]  # 2 I/S + 1 land
    state.players[1].graveyard = [CardDef(f"opp{i}", CardType.INSTANT, {"U": 1}, EffectId.FILLER) for i in range(5)]  # must NOT count
    eff = drl_env._effective_cast_cost(state, registry.CARD_DEFS["Tolarian Terror"])
    assert eff["generic"] == 4, eff  # 6 - 2 (own I/S only), opponent's 5 ignored
    print("blue_cards.py Tolarian/Cryptic cost reduction (caster's own graveyard only) self-check: OK")

    # Deem Inferior: the target's OWNER tucks it 2nd-from-top or bottom.
    state = GameState(on_the_play=True, players=[PlayerState(True), PlayerState(False)])
    state.active_idx = 0
    victim = Permanent(CardDef("Threat", CardType.CREATURE, None, EffectId.FILLER, power=3, toughness=3))
    victim.slot = 1
    state.players[1].battlefield = [victim]
    state.players[1].library = [CardDef(f"lib{i}", CardType.LAND, None, EffectId.ISLAND) for i in range(3)]
    di = registry.CARD_DEFS["Deem Inferior"]
    state.players[0].hand = [di]
    cast_deem_inferior(state, di)
    _ect(state, 1, "Threat", 1)
    _resolve_top(state)
    assert state.pending_resolution["kind"] == "tuck_position" and state.active_idx == 1  # owner picks
    execute_tuck_position(state, "bottom")
    assert victim not in state.players[1].battlefield
    assert state.players[1].library[-1].name == "Threat" and state.active_idx == 0  # bottom, active restored

    # Deem Inferior targets ANY nonland permanent -- here an opponent's ARTIFACT
    # (not a creature) -- and a land is NOT a legal target.
    state = GameState(on_the_play=True, players=[PlayerState(True), PlayerState(False)])
    state.active_idx = 0
    art = Permanent(CardDef("Gadget", CardType.ARTIFACT, None, EffectId.FILLER))
    art.slot = 1
    a_land = Permanent(CardDef("A Land", CardType.LAND, None, EffectId.ISLAND, basic=True))
    a_land.slot = 1
    state.players[1].battlefield = [art, a_land]
    state.players[1].library = [CardDef(f"lib{i}", CardType.LAND, None, EffectId.ISLAND) for i in range(3)]
    di2 = registry.CARD_DEFS["Deem Inferior"]
    state.players[0].hand = [di2]
    cast_deem_inferior(state, di2)
    from ..resolution import choose_any_target_creature_options
    opts = choose_any_target_creature_options(state)  # misnamed: really "any matching permanent"
    assert (1, "Gadget", 1) in opts and (1, "A Land", 1) not in opts  # artifact targetable, land not
    _ect(state, 1, "Gadget", 1)
    _resolve_top(state)
    assert state.pending_resolution["kind"] == "tuck_position"
    execute_tuck_position(state, "top2")
    assert art not in state.players[1].battlefield
    assert state.players[1].library[1].name == "Gadget"  # second from top
    print("blue_cards.py Deem Inferior (any nonland permanent: artifact tuck, land excluded) self-check: OK")

    # --- G10 Delver of Secrets: upkeep look -> may transform -> 3/2 flyer ---
    from ..turn import upkeep_step
    from ..effects.triggers import promote_triggers_to_stack
    from ..effects.stats import creature_keywords, permanent_power, permanent_toughness
    from ..resolution import execute_may_transform

    state = GameState(on_the_play=True)
    state.active_idx = 0
    delver = Permanent(registry.CARD_DEFS["Delver of Secrets"])
    delver.slot = 0
    state.players[0].battlefield = [delver]
    state.players[0].library = [CardDef("Bolt", CardType.INSTANT, {"R": 1}, EffectId.FILLER),
                                CardDef("x", CardType.LAND, None, EffectId.ISLAND)]
    upkeep_step(state)  # queues the "at the beginning of your upkeep" trigger
    assert state.trigger_queue and state.trigger_queue[0]["type"] == "upkeep"
    promote_triggers_to_stack(state)
    assert len(state.stack) == 1
    _resolve_top(state)  # top card is an instant -> transform offered
    assert state.pending_resolution["kind"] == "may_transform"
    assert permanent_power(state, delver) == 1 and permanent_toughness(state, delver) == 1  # front face
    execute_may_transform(state, True)
    assert delver.flags.get("transformed")
    assert permanent_power(state, delver) == 3 and permanent_toughness(state, delver) == 2  # Insectile Aberration
    assert "flying" in creature_keywords(state, delver)
    print("blue_cards.py Delver (upkeep reveal I/S -> transform 3/2 flyer) self-check: OK")

    # Non-instant/sorcery on top: look, but no transform choice ever opens.
    state = GameState(on_the_play=True)
    state.active_idx = 0
    d2 = Permanent(registry.CARD_DEFS["Delver of Secrets"])
    d2.slot = 0
    state.players[0].battlefield = [d2]
    state.players[0].library = [CardDef("x", CardType.LAND, None, EffectId.ISLAND)]
    upkeep_step(state)
    promote_triggers_to_stack(state)
    _resolve_top(state)
    assert state.pending_resolution is None and not d2.flags.get("transformed")
    print("blue_cards.py Delver (non-I/S top -> no transform) self-check: OK")

    # --- G12 Ward {2}: an opponent targeting Tolarian Terror must pay {2} or
    # the spell is countered. ---
    from ..effects.casting import cast_targeting_creature
    from ..effects.state_based import destroy_permanent

    def _cast_zap_at_tt(pay):
        state = GameState(on_the_play=True, players=[PlayerState(True), PlayerState(False)])
        tt = Permanent(registry.CARD_DEFS["Tolarian Terror"])
        tt.slot = 1
        state.players[0].battlefield = [tt]
        state.active_idx = 1  # the OPPONENT is casting
        if pay:
            state.players[1].mana_pool = {"U": 2}  # can afford Ward's {2}
        zap = CardDef("Zap", CardType.INSTANT, {"U": 1}, EffectId.FILLER)
        state.players[1].hand = [zap]
        cast_targeting_creature(state, zap, lambda s, perm: destroy_permanent(s, perm))
        _ect(state, 0, "Tolarian Terror", 1)  # lock the target -> Ward triggers, Zap pushed
        assert len(state.stack) == 1 and any(e["type"] == "ward" for e in state.players[1].trigger_queue)
        promote_triggers_to_stack(state)  # Ward goes ON TOP of Zap
        assert len(state.stack) == 2 and state.stack[-1]["card_def"].name == "Tolarian Terror"
        _resolve_top(state)  # Ward resolves -> pay_unless for the caster (idx 1)
        assert state.pending_resolution["kind"] == "pay_unless" and state.active_idx == 1
        return state, tt, zap

    # Decline -> Zap countered, Tolarian Terror survives.
    state, tt, zap = _cast_zap_at_tt(pay=False)
    pay_unless_decline(state)
    assert tt in state.players[0].battlefield  # not destroyed -- Zap was countered
    assert state.stack == [] and zap in state.players[1].graveyard
    print("blue_cards.py Ward {2} (decline -> spell countered) self-check: OK")

    # Pay {2} -> Zap survives, resolves, Tolarian Terror is destroyed.
    state, tt, zap = _cast_zap_at_tt(pay=True)
    pay_unless_pay(state)
    _guard = 0
    while state.pending_resolution is not None and state.pending_resolution["kind"] == "pay_cost":
        _guard += 1
        assert _guard < 10
        execute_pool_spend(state, pool_spend_options(state)[0])
    assert len(state.stack) == 1  # Zap NOT countered
    _resolve_top(state)  # Zap resolves -> destroy Tolarian Terror
    assert tt not in state.players[0].battlefield  # destroyed
    print("blue_cards.py Ward {2} (pay -> spell resolves) self-check: OK")
