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
from ..cards import CardDef, CardType, EffectId, card_subtypes
from ..effects.casting import (
    _log_target_fizzle, capture_any_target, cast_permanent_from_hand, cast_targeting_creature, has_creature_target,
    target_still_legal,
)
from ..effects.shared import (
    affinity_reduction, discard_from_hand_to_graveyard, find_to_hand, graveyard_instant_sorcery_count, mill,
)
from ..effects.stack import counter_spell, push_ability_to_stack, push_to_stack
from ..effects.state_based import departing_card_def, sacrifice_to_graveyard
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

# Back face of Delver of Secrets. Deliberately NOT in BLUE_CARD_CATALOG / CARD_DEFS:
# it's never in a decklist, drawn, or cast -- it only exists as the identity a
# Delver Permanent takes on once it transforms (execute_may_transform swaps the
# Permanent's card_def to this). Same EffectId as the front face so every registry
# lookup (the upkeep trigger, the transform spec, stats) still resolves; same cast
# cost/color identity; 3/2 stats live here AND in the "transform" spec below (the
# spec stays authoritative for combat via effects.stats._transform_spec, this def
# carries them so any direct card_def.extra read stays consistent). A DFC reverts to
# its front face in every zone but the battlefield, so state_based._departing_card_def
# puts the FRONT def back when it leaves.
INSECTILE_ABERRATION_CARD_DEF = CardDef(
    "Insectile Aberration", CardType.CREATURE, {"U": 1}, EffectId.DELVER_OF_SECRETS, power=3, toughness=2,
)


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
            if captured is None or not target_still_legal(state, captured):
                where = (captured[1].card_def.name, captured[1].slot) if captured is not None else None
                _log_target_fizzle(state, card_def, where)
                return
            permanent = captured[1]
            owner_idx = controller_idx(state, permanent)  # controller == owner (no control-changing in this pool)
            state.players[owner_idx].battlefield.remove(permanent)
            # a DFC tucked into the library reverts to its front face (Delver, not Insectile)
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
    graveyard OTHER than `exclude` (the exact graveyard INSTANCE being escaped),
    one at a time (chained begin_choose_graveyard_card). Runs on_complete(state)
    once n are exiled. Excludes `exclude` by object identity -- so a SECOND copy
    of the same card in the graveyard (a distinct instance now) CAN be exiled to
    pay, faithfully."""
    def _step(remaining):
        if remaining == 0:
            on_complete(state)
            return

        def _chosen(state, chosen):
            state.graveyard.remove(chosen)  # the exact chosen instance; exiled, untracked
            state.log_event("zone_move", card=chosen.name, from_zone="graveyard", to_zone="exile_untracked", reason="escape")
            _step(remaining - 1)

        begin_choose_graveyard_card(state, lambda c: c is not exclude, _chosen)

    _step(n)


def _sleep_escape_legal(state):
    """Escape {2}{U}, exile three OTHER cards from your graveyard: legal only
    with 3+ other graveyard cards to exile AND a legal creature to tap
    (a targeted spell needs a target). Mana + Sleep-in-graveyard are checked
    by the generic graveyard-cast machinery (drl_env._flashback_legal)."""
    # 3+ cards OTHER than the one Sleep being escaped -- any other card, INCLUDING
    # a second Sleep copy (distinct instances now). Escape is only offered with a
    # Sleep in the graveyard, so "others" = everything but that one card.
    if not any(c.name == "Sleep of the Dead" for c in state.graveyard):
        return False
    return len(state.graveyard) - 1 >= 3 and has_creature_target(state)


def escape_sleep_of_the_dead(state, inst):
    """Escape: {2}{U} (paid by the graveyard-cast machinery) + exile three
    other graveyard cards, then Sleep leaves the graveyard (escaping; exiled
    after it resolves) and taps a target creature (chosen at cast, fizzles if
    gone).

    inst: the exact graveyard CardInstance being escaped -- see
    flashback_dread_return. Identity matters twice here, not just for the
    removal: it's also the exclusion passed to _exile_n_other_from_graveyard,
    whose predicate is `c is not exclude`, so a SECOND Sleep copy in the
    graveyard stays a legal choice to exile as part of the cost (faithful --
    it genuinely is "another card") while the escaping copy itself never is,
    handed down directly by the caller (drl_env._actions._graveyard_instance)."""
    sleep_inst = inst

    def _after_exile(state):
        state.graveyard.remove(sleep_inst)  # escapes the graveyard; exiled after resolution (untracked)
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
    """Sewer-veillance Cam ETB/LTB: "you may tap or untap target creature."
    Target-at-promotion (etb_targets/ltb_targets: True on both halves): this
    whole function runs AT PROMOTION, as the triggered ability goes on the
    stack (603.3d) -- opening the optional target choice (either side,
    hexproof/shroud-aware) right then, not once the ability actually
    resolves, so an opponent gets a priority window against a known target.
    on resolution it TOGGLES the target's tapped state -- observationally
    identical to the real tap-or-untap choice, since the no-op direction
    (tapping a tapped or untapping an untapped creature) is never the
    meaningful pick -- fizzling if the target's gone by then (608.2b)."""
    idx = state.active_idx

    def _on_target(state, descriptor):
        captured = capture_any_target(state, descriptor)

        def _resolve(st, cd):
            if captured is None or not target_still_legal(st, captured):
                return
            perm = captured[1]
            perm.tapped = not perm.tapped
            st.log_event("tap_or_untap", permanent=(perm.card_def.name, perm.slot), now_tapped=perm.tapped)

        push_to_stack(state, source_card_def, _resolve, reserves_hand_card=False, is_spell=False,
                      targets=() if captured is None else (captured,))

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

        push_to_stack(state, card_def, _resolve, targets=() if entry is None else (("stack_entry", entry),))

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
    """{U}: Target player mills two cards. Draw a card (you). Real "target
    player" -- a genuine begin_choose_target_player choice, locked at CAST
    (precast_choice), not resolution: real Magic (601.2c) chooses targets
    as the spell is announced, before it ever waits on the stack. "Target
    player" is always legal (at minimum yourself), so this never fizzles.
    Whoever is milled, the caster (active player) draws."""
    def _on_player(state, idx):
        def _resolve(state, card_def):
            discard_from_hand_to_graveyard(state, card_def)
            mill(state, 2, idx)
            state.draw(1)  # the caster draws -- active_idx is the controller throughout (no flip)

        push_to_stack(state, card_def, _resolve)

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


def _deep_analysis_choose_and_push(state, card_def, to_graveyard, reserves_hand_card, exiles_on_resolve=False):
    """Choose the target player as Deep Analysis is put on the stack -- real
    "target player" (601.2c: targets are chosen as a spell is cast, before
    it ever waits on the stack), always legal (at minimum yourself), so this
    never fizzles. Shared by the hard cast and Flashback below;
    `to_graveyard`/`reserves_hand_card` say how the card itself reaches the
    graveyard on resolution (hard cast: hand -> graveyard; Flashback:
    already out of the graveyard, exiled after -- Dread Return's own
    convention). Whoever is chosen draws two (cross-player via
    _target_player_draws)."""
    def _on_player(state, idx):
        def _resolve(state, card_def):
            to_graveyard(state, card_def)
            _target_player_draws(state, idx, 2)

        push_to_stack(state, card_def, _resolve, reserves_hand_card=reserves_hand_card, exiles_on_resolve=exiles_on_resolve)

    begin_choose_target_player(state, _on_player)


def cast_deep_analysis(state, card_def):
    """{3}{U}: Target player draws two cards. Locked at CAST
    (precast_choice), not resolution."""
    _deep_analysis_choose_and_push(state, card_def, discard_from_hand_to_graveyard, reserves_hand_card=True)


def flashback_deep_analysis(state, inst):
    """Flashback -- {1}{U}, Pay 3 life. The {1}{U} was already paid by the
    generic mana-flashback path (drl_env._flashback_execute's begin_pay_cost);
    pay the 3-life additional cost now, remove this card from the graveyard
    (it leaves the moment Flashback is chosen; exiled afterward -- untracked,
    Dread Return's precedent), then choose the target player and push the
    same target-player-draws-two effect onto the stack -- Flashback casts
    the spell too, so its target is locked here, at that moment, same as the
    hard cast. Life >= 3 is enforced by the flashback's own legal predicate.

    inst: the exact graveyard CardInstance being flashed back -- see
    flashback_dread_return. Matched and removed by object identity, not a
    by-name lookup, since a graveyard can hold same-named CardInstances that
    must stay distinct."""
    state.graveyard.remove(inst)
    lose_life(state, 3, reason="deep_analysis_flashback")
    _deep_analysis_choose_and_push(state, inst, lambda s, cd: None, reserves_hand_card=False, exiles_on_resolve=True)


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
        # No "pending_kinds" entry: choose_stack_target is POINTER-addressed
        # (rl.action_bridge), not a by-name fixed action, so it needs no
        # per-deck action-table declaration -- see begin_choose_stack_target's
        # own docstring for why (the countered spell is very often the
        # opponent's, which a by-name row could never represent).
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
        "pending_kinds": {"pay_unless"},  # the "unless controller pays {2}" rider -- still a real, live declaration
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
        # via permanent.flags["transformed"], set by execute_may_transform, which
        # also swaps the Permanent's card_def to `card_def` so the game state's own
        # identity -- name, logging, RL perception -- becomes the back face).
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
        "etb_targets": True,  # target creature chosen at promotion (603.3d), not resolution
        "ltb_trigger": lambda state, permanent: sewer_cam_tap_or_untap(state, permanent.card_def),
        "ltb_targets": True,  # same target-at-promotion timing for the leaves-the-battlefield half
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
