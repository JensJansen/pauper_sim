"""Battlefield entry and the direct cast paths: playing a land, casting a
plain permanent, casting an Aura (choose-target-then-resolve), and one ETB
(land bounce) generic enough to live here. tokens.py builds on
enters_battlefield too.

Depends on stack.py (push_to_stack) and win_check.py. Does not depend on
triggers.py -- that dependency points the other way.

References registry.EFFECT_REGISTRY only inside function bodies (registry.py
imports the catalogs, which import this module, so a top-level import would
bind a not-yet-built name) -- see game/registry.py's docstring."""

from .. import registry
from ..cards import CardType
from ..state import Permanent, CardInstance
from .shared import discard_from_hand_to_graveyard
from .stack import push_to_stack
from .stats import can_be_targeted
from .win_check import _check_end_of_game
from .. import resolution


def has_creature_target(state, eligible=lambda p: True):
    """Is there any legal creature target for a "destroy/target creature"
    spell -- on either battlefield, matching `eligible`, and targetable by
    the active player (hexproof/shroud)? Backs the cast's own extra_legal
    (a mandatory-target spell can't be cast without a legal target)."""
    idx = state.active_idx
    return any(
        p.card_type == CardType.CREATURE and eligible(p) and can_be_targeted(state, p, idx)
        for player in state.players for p in player.battlefield
    )


def cast_targeting_creature(state, card_def, on_resolve, eligible=lambda p: True):
    """Shared body for every single-target "target creature" spell: pick a
    creature on either battlefield matching `eligible` (hexproof/shroud-
    aware), locked at cast; on resolution the spell goes to its owner's
    graveyard and, if the target is still legal (608.2b),
    `on_resolve(state, target_permanent)` runs -- else it fizzles. Callers
    supply on_resolve: destroy (Cast Down/Terminate/Snuff Out), put
    counters (Unexpected Fangs), or grant keywords (Toxin Analysis).

    captured can be None even for this mandatory pick: this spell's own
    cost payment (paid after targeting under this engine's cost-then-target
    order) can kill the caster's last legal target between pick and
    payment. target_still_legal(None) returns True by design (for optional-
    pick callers), so it must be checked explicitly here too."""
    idx = state.active_idx

    def _on_target(state, descriptor):
        captured = capture_any_target(state, descriptor)

        def _resolve(state, card_def):
            discard_from_hand_to_graveyard(state, card_def)
            if captured is None or not target_still_legal(state, captured):
                where = (captured[1].card_def.name, captured[1].slot) if captured is not None else None
                _log_target_fizzle(state, card_def, where)
                return
            on_resolve(state, captured[1])

        push_to_stack(state, card_def, _resolve, targets=() if captured is None else (captured,))

    resolution.begin_choose_any_target(
        state,
        lambda p: p.card_type == CardType.CREATURE and eligible(p) and can_be_targeted(state, p, idx),
        _on_target,
        allow_players=False,
    )


def play_land_from_hand(state, card_def):
    state.hand.remove(card_def)
    state.lands_played_this_turn += 1
    return enters_battlefield(state, card_def, from_zone="hand")


def cast_permanent_from_hand(state, card_def):
    """Artifacts/creatures with no additional cost beyond mana and no target
    choices. Run as the spell's resolve off the stack -- the card already
    left hand at cast, so the `if in hand` guard is a no-op there (and
    still lets a self-check call this directly)."""
    if card_def in state.hand:
        state.hand.remove(card_def)
    return enters_battlefield(state, card_def, from_zone="hand")


def _log_target_fizzle(state, card_def, chosen_name_slot):
    """Record of a targeted spell failing to resolve -- otherwise this looks
    identical to "cast a spell that legitimately does nothing." target=None
    when there was never a captured target at all. Routed through
    state.log_event, a no-op when logging is off, so this costs nothing
    during real training runs."""
    state.log_event("target_fizzle", card=card_def.name, target=chosen_name_slot)


# MTG 400.7: a card that changes zones becomes a NEW object. Targeting here
# captures the exact object, so a flickered/returned permanent or graveyard card
# correctly makes an old target fizzle. Not modeled: a LINKED ability that
# deliberately TRACKS an object across a zone change (Adventure, Foretell,
# "exile, then you may play THIS card") -- the returned object is still new,
# but the linked ability references it across the change. No card in this pool
# needs that tracking; add cross-zone object linkage here (and to
# game.state.move_card's minting) if one arrives. See plans/object-identity-zone-model.md.
def capture_any_target(state, target):
    """Cast/activation time: lock a resolution.begin_choose_any_target
    descriptor onto a concrete, identity-stable target to carry on the
    stack. A ("player", idx) is already stable; a ("creature", side, name,
    slot) is resolved to the exact Permanent object it names right now, so
    resolution-time legality (target_still_legal) is an object-identity
    check, never a by-name re-lookup that could latch onto a different
    same-named creature. None (no target chosen) passes straight through."""
    if target is None or target[0] == "player":
        return target
    _, side, name, slot = target
    perm = next(p for p in state.players[side].battlefield if p.card_def.name == name and p.slot == slot)
    _maybe_trigger_ward(state, perm)
    return ("creature", perm)


def _maybe_trigger_ward(state, permanent):
    """Ward (Tolarian Terror): "Whenever this becomes the target of a spell
    or ability an opponent controls, counter it unless that player pays
    [cost]." Fires only when the chooser is an opponent of the permanent's
    controller. Queues a "ward" trigger carrying the payer and cost; it
    lands above the triggering spell and makes the payer pay-or-be-
    countered."""
    ward = registry.EFFECT_REGISTRY.get(permanent.card_def.effect_id, {}).get("ward")
    if ward is None:
        return
    caster_idx = state.active_idx
    controller = next((idx for idx, p in enumerate(state.players) if permanent in p.battlefield), None)
    if controller is None or caster_idx == controller:
        return  # only an opponent's targeting triggers Ward
    # Written into the controller's own queue directly (not the active-
    # player proxy), since Ward belongs to the Warded creature's controller,
    # not the caster who is state.active_idx right now.
    state.players[controller].trigger_queue.append(
        {"type": "ward", "card_def": permanent.card_def, "payer_idx": caster_idx, "cost": ward}
    )
    state.log_event("ward_triggered", permanent=(permanent.card_def.name, permanent.slot), payer=caster_idx)


def target_still_legal(state, captured):
    """Resolution time: is a captured any-target still legal (608.2b -- a
    spell/ability whose targets are all illegal on resolution does
    nothing)? A player is always legal. A creature is legal only while that
    exact captured Permanent is still on some battlefield. None (no
    target) is treated as "nothing to fizzle"."""
    if captured is None or captured[0] == "player":
        return True
    perm = captured[1]
    return any(perm in player.battlefield for player in state.players)


def cast_aura(state, card_def, target_predicate, on_attached=None, no_target_fallback=None):
    """Cast an Aura from hand: pick a legal target via
    resolution.begin_choose_any_target, addressed by the exact (side,
    name, slot) permanent, not just a name. target_predicate is the Aura's
    "enchant WHAT" filter (creature, or land for Utopia Sprawl).

    The target is chosen once, at cast, and re-checked by object identity
    when the spell resolves. If gone by then, or never legal to begin
    with, the spell fails outright, logged via _log_target_fizzle.

    on_attached(state, aura_permanent), if given, runs once attached -- for
    an Aura with its own ETB effect (Abundant Growth's draw).

    no_target_fallback(state, card_def), if given, replaces the default
    "no legal target -> graveyard" fizzle. The one user is Bestow (Nyxborn
    Hydra, 702.103e): a bestow spell with no legal target still enters the
    battlefield as a creature.

    An Aura returns to the graveyard (or, for Rancor, to hand) when
    whatever it enchants leaves the battlefield ("orphaning") -- modeled
    only for combat death (state_based._destroy_creature); other removal
    paths don't currently orphan an enchanted permanent's Auras."""
    def _on_target_chosen(state, target_descriptor):
        captured = capture_any_target(state, target_descriptor)

        def _resolve(state, card_def):
            # Resolving off the stack: already left hand at cast, must not
            # re-enter it. Target was locked at cast (captured).
            if card_def in state.hand:
                state.hand.remove(card_def)
            if captured is None or not target_still_legal(state, captured):
                if no_target_fallback is not None:
                    no_target_fallback(state, card_def)
                    state.log_event("aura_no_target", card=card_def.name, outcome="entered_as_creature")
                    return
                state.move_card(card_def, state.graveyard)
                state.log_event("zone_move", card=card_def.name, from_zone="hand", to_zone="graveyard", reason="fizzle")
                where = (captured[1].card_def.name, captured[1].slot) if captured is not None else None
                _log_target_fizzle(state, card_def, where)
                return
            target = captured[1]
            aura = enters_battlefield(state, card_def, from_zone="hand")
            aura.flags["enchanting"] = target
            state.log_event(
                "aura_attached", aura=(aura.card_def.name, aura.slot), target=(target.card_def.name, target.slot),
            )
            if on_attached is not None:
                on_attached(state, aura)

        push_to_stack(state, card_def, _resolve, targets=() if captured is None else (captured,))

    resolution.begin_choose_any_target(
        state,
        lambda p: target_predicate(p) and can_be_targeted(state, p, state.active_idx),
        _on_target_chosen,
        allow_players=False,
    )


def enters_battlefield(state, card_def, force_tapped=False, from_zone=None):
    """Move a CardDef onto the battlefield as a new Permanent, applying its
    enters-tapped default and queuing its ETB triggered ability onto
    state.trigger_queue for the priority round to put on the stack (not run
    inline -- see the ETB block below), then runs _check_end_of_game.
    Caller has already removed card_def from its previous zone.

    force_tapped=True overrides the registry's enters_tapped default to
    always-tapped -- a one-off per-trigger condition, not a card property
    (Sneaky Snacker enters untapped when cast, but tapped when its own
    "third card drawn" trigger returns it from the graveyard)."""
    # Accept a CardInstance/Permanent or a raw CardDef: the permanent is
    # always minted fresh from the underlying CardDef (a new object, 400.7).
    if isinstance(card_def, CardInstance):
        card_def = card_def.card_def
    spec = registry.EFFECT_REGISTRY.get(card_def.effect_id, {})
    # enters_tapped may be a plain bool or a callable(state) -> bool, for a
    # land whose tapped-ness depends on the board (Gingerbread Cabin).
    # Evaluated before this permanent joins the battlefield below.
    raw_tapped = spec.get("enters_tapped", False)
    tapped = force_tapped or (raw_tapped(state) if callable(raw_tapped) else raw_tapped)
    permanent = state.new_permanent(card_def, tapped=tapped)
    # Record how it entered so an ETB conditioned on entering untapped
    # (Gingerbread Cabin's Food) reads the entry state, not whatever the
    # permanent's tapped-ness has since become.
    permanent.flags["entered_tapped"] = tapped
    # Pooled slot assignment: lowest number not already in use among this
    # player's currently-live permanents of the same name, so slots free up
    # as copies leave rather than growing with turns played.
    used_slots = {p.slot for p in state.battlefield if p.card_def.name == card_def.name}
    slot = 1
    while slot in used_slots:
        slot += 1
    permanent.slot = slot
    state.battlefield.append(permanent)
    state.log_event(
        "zone_move", permanent=(permanent.card_def.name, permanent.slot), from_zone=from_zone,
        to_zone="battlefield", tapped=tapped, card_type=permanent.card_type.name,
        power=card_def.extra.get("power"), toughness=card_def.extra.get("toughness"),
    )
    # Seeded to the printed power/toughness just logged above, so
    # state_based._log_stat_changes's first pass doesn't immediately re-log
    # an unchanged value. If a static-self boost is already true the
    # instant this permanent enters, the seed still mismatches the true
    # computed value, so that first SBA pass correctly fires anyway.
    permanent.flags["_logged_pt"] = (card_def.extra.get("power"), card_def.extra.get("toughness"))

    # ETB (603.3) doesn't resolve inline: it's queued for the next priority
    # round to put on the stack (triggers.py's "etb" branch runs the hook
    # then). The etb_trigger callable receives (state, permanent) so an ETB
    # that needs its own source can reach it (Mesmeric Fiend).
    if spec.get("etb_trigger") is not None:
        state.trigger_queue.append({"type": "etb", "card_def": card_def, "permanent": permanent})

    _check_end_of_game(state)

    return permanent


def bounce_land_etb(state):
    """ETB: return a land you control to hand (Rakdos Carnarium). Not a
    real MTG "target" (no "target" in the ability's own text), so this runs
    synchronously rather than deferred to the stack like cast_aura's
    targeting. enters_battlefield appends the permanent before running its
    ETB trigger, so the land that just entered is itself already a legal
    choice, guaranteeing "always at least one target" for free."""
    def _on_chosen(state, choice):
        if choice is None:
            return  # never reachable: this card is itself a legal choice
        name, slot = choice
        permanent = next(
            p for p in state.battlefield
            if p.card_def.name == name and p.slot == slot and p.card_def.card_type == CardType.LAND
        )
        state.battlefield.remove(permanent)
        state.hand.append(permanent.card_def)
        state.log_event(
            "zone_move", permanent=(permanent.card_def.name, permanent.slot), from_zone="battlefield",
            to_zone="hand", reason="bounce",
        )

    resolution.begin_choose_permanent(state, lambda p: p.card_def.card_type == CardType.LAND, _on_chosen)

