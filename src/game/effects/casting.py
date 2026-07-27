"""Battlefield entry and the direct cast paths: playing a land, casting a
plain permanent, casting an Aura (choose-target-then-resolve), and one ETB
(land bounce) generic enough to live here. tokens.py builds on
enters_battlefield too.

Depends on stack.py (push_to_stack) and win_check.py (the game-end check every
board change can trigger). Does NOT depend on triggers.py -- that dependency
points the other way (promote_triggers_to_stack needs enters_battlefield), so
triggers.py sits above this module.

References registry.EFFECT_REGISTRY only inside function bodies (registry.py
imports the catalogs, which import this module, so a top-level `from .registry
import EFFECT_REGISTRY` would bind a not-yet-built name) -- see
game/registry.py's docstring."""

from .. import registry
from ..cards import CardType
from ..state import Permanent
from .shared import discard_from_hand_to_graveyard
from .stack import push_to_stack
from .stats import can_be_targeted
from .win_check import _check_end_of_game
from .. import resolution


def has_creature_target(state, eligible=lambda p: True):
    """Is there any legal creature target for a "destroy/target creature"
    spell -- on EITHER battlefield, matching `eligible`, and targetable by
    the active player (hexproof/shroud)? A spell with a single creature
    target can't be cast without one (real Magic), so this backs the cast's
    own extra_legal."""
    idx = state.active_idx
    return any(
        p.card_type == CardType.CREATURE and eligible(p) and can_be_targeted(state, p, idx)
        for player in state.players for p in player.battlefield
    )


def cast_targeting_creature(state, card_def, on_resolve, eligible=lambda p: True):
    """Shared body for every single-target "target creature" spell: pick a
    creature on EITHER battlefield matching `eligible` (hexproof/shroud-
    aware), locked at cast (used with precast_choice); on resolution the
    spell goes to its owner's graveyard and, if the target is still a legal
    creature (608.2b), `on_resolve(state, target_permanent)` runs -- else the
    spell fizzles doing nothing. Callers supply on_resolve: destroy
    (Cast Down/Terminate/Snuff Out, via state_based.destroy_permanent), put
    counters (Unexpected Fangs), or grant until-EOT keywords + Investigate
    (Toxin Analysis). `eligible` narrows the legal targets (Snuff Out's
    nonblack); the default is any creature."""
    idx = state.active_idx

    def _on_target(state, descriptor):
        captured = capture_any_target(state, descriptor)

        def _resolve(state, card_def):
            discard_from_hand_to_graveyard(state, card_def)  # the spell itself -> its owner's graveyard
            if not target_still_legal(state, captured):
                where = (captured[1].card_def.name, captured[1].slot) if captured is not None else None
                _log_target_fizzle(state, card_def, where)
                return
            on_resolve(state, captured[1])

        push_to_stack(state, card_def, _resolve)

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
    """Artifacts/creatures with no additional cost beyond mana and no
    target choices. Mana cost is paid by the caller first."""
    state.hand.remove(card_def)
    return enters_battlefield(state, card_def, from_zone="hand")


def _log_target_fizzle(state, card_def, chosen_name_slot):
    """Console-visible record of a targeted spell failing to resolve (see
    cast_aura's own docstring for the rule this enforces) -- otherwise this
    branch is silent and looks, from the outside, identical to "cast a
    spell that legitimately does nothing," which is exactly the kind of
    gap that made the original crash (a stale choose_permanent resolution)
    hard to diagnose. where=None only if begin_choose_permanent's own
    empty-options safety net fired at cast time -- unreachable for an Aura
    today (its own extra_legal already guarantees a target exists then),
    kept here so the message stays accurate if that ever changes."""
    where = f"{chosen_name_slot[0]!r} (slot {chosen_name_slot[1]})" if chosen_name_slot is not None else "no legal target at cast time"
    print(f"[target fizzle] turn {state.turn_number}: {card_def.name} failed to resolve -- target was {where}, not on the battlefield anymore.")


def capture_any_target(state, target):
    """Cast/activation time: lock a resolution.begin_choose_any_target
    descriptor onto a concrete, identity-stable target to carry on the
    stack. A ("player", idx) is already stable; a ("creature", side, name,
    slot) is resolved to the EXACT Permanent object it names right now, so
    resolution-time legality (target_still_legal) is an object-identity
    check, never a fungible-by-name re-lookup that could latch onto a
    different same-named creature. Real Magic: a spell's/ability's targets
    are chosen and locked as it is put on the stack, never re-chosen at
    resolution. None (no target was chosen, e.g. an "up to one" declined)
    passes straight through."""
    if target is None or target[0] == "player":
        return target
    _, side, name, slot = target
    perm = next(p for p in state.players[side].battlefield if p.card_def.name == name and p.slot == slot)
    _maybe_trigger_ward(state, perm)
    return ("creature", perm)


def _maybe_trigger_ward(state, permanent):
    """Ward (Tolarian Terror): "Whenever this becomes the target of a spell or
    ability an OPPONENT controls, counter it unless that player pays [cost]."
    capture_any_target is the moment a creature becomes a target, so this fires
    here -- but only when the chooser (state.active_idx, the caster/activator)
    is an opponent of the permanent's controller (a controller targeting their
    own Warded creature never triggers Ward). Queues a "ward" trigger (resolved
    by game.effects.triggers), carrying the payer (that opponent) and the ward
    cost; it lands on the stack above the triggering spell (which the caller
    pushes right after this returns) and makes the payer pay-or-be-countered.
    Lazy registry read, same convention as the rest of this module."""
    ward = registry.EFFECT_REGISTRY.get(permanent.card_def.effect_id, {}).get("ward")
    if ward is None:
        return
    caster_idx = state.active_idx
    controller = next((idx for idx, p in enumerate(state.players) if permanent in p.battlefield), None)
    if controller is None or caster_idx == controller:
        return  # only an OPPONENT's targeting triggers Ward
    state.trigger_queue.append(
        {"type": "ward", "card_def": permanent.card_def, "payer_idx": caster_idx, "cost": ward}
    )
    state.log_event("ward_triggered", permanent=(permanent.card_def.name, permanent.slot), payer=caster_idx)


def target_still_legal(state, captured):
    """Resolution time: is a captured any-target still a legal target (real
    Magic 608.2b -- a spell/ability whose targets are ALL illegal on
    resolution is removed from the stack and does nothing)? A player is
    always legal (no effect in this pool removes a player from the game). A
    creature is legal only while that exact captured Permanent is still on
    SOME battlefield -- gone (died, sacrificed, bounced, exiled) means the
    target is illegal and its spell/ability fizzles. None (no target) is
    treated as "nothing to fizzle" -- the caller decided 0 targets was legal
    (an "up to one" declined), so it simply resolves with no target."""
    if captured is None or captured[0] == "player":
        return True
    perm = captured[1]
    return any(perm in player.battlefield for player in state.players)


def cast_aura(state, card_def, target_predicate, on_attached=None):
    """Cast an Aura from hand: pick a legal target via
    resolution.begin_choose_any_target -- real "Enchant creature/land"
    targets ANY matching permanent on EITHER battlefield (not just your own),
    hexproof/shroud aware (can_be_targeted), addressed by the EXACT (side,
    name, slot) permanent chosen -- not just a name, since two same-named
    permanents stop being interchangeable the instant an Aura attaches to
    only one of them.
    (target_predicate is the Aura's own "enchant WHAT" filter -- creature for
    the pumps, land/Forest for Utopia Sprawl/Abundant Growth.)

    Real MTG targeting rule, enforced here: the target is chosen once,
    right when the spell is cast -- in this engine, the instant its cost
    finishes paying (drl_env._targeted_cast_execute calls this function
    directly as pay_cost's on_complete, instead of the generic
    _cast_execute's auto-push-to-stack, precisely so target selection runs
    BEFORE the card ever sits on the stack) -- and re-checked by EXACT
    OBJECT IDENTITY only once the spell actually resolves off the stack.
    If that exact Permanent is gone by then (died, was sacrificed, bounced
    -- doesn't matter how), the spell fails outright: no effect, straight
    to the graveyard (real Magic: a spell whose only target is illegal on
    resolution is removed from the stack doing nothing) -- see _resolve's
    own fizzle branch below, logged via _log_target_fizzle so this doesn't
    silently look like "cast a spell that did nothing."

    on_attached(state, aura_permanent), if given, runs once actually
    attached -- for an Aura with its own ETB effect (Abundant Growth's
    draw, Cartouche of Solidarity's token, Utopia Sprawl's chosen color).
    Routed through here rather than the registry's own etb_trigger (which
    only ever receives state, not the permanent) since every one of these
    needs to record something onto the Aura's own Permanent, not just act
    on shared state.

    Real-rules note: an Aura returns to the graveyard (and, for Rancor,
    from there back to hand) when whatever it enchants leaves the
    battlefield ("orphaning") -- modeled for the one reachable case in this
    card pool, combat death. Every OTHER battlefield-removal call
    site in this codebase (sacrifice, bounce, exile -- see their own call
    sites) still doesn't orphan an enchanted permanent's Auras, since none
    of them can currently target a creature that could be enchanted
    (boggles is the only deck with Auras, and none of its own cards
    sacrifice/bounce/exile a creature). Thread the same orphaning logic
    through a removal site if a future card ever makes that reachable."""
    def _on_target_chosen(state, target_descriptor):
        captured = capture_any_target(state, target_descriptor)  # ("permanent-as-'creature'", perm) or None

        def _resolve(state, card_def):
            # Still-in-hand-while-on-stack convention every other cast path
            # here follows (see push_to_stack's own docstring) -- the target
            # is already locked in via `captured`, captured at cast time,
            # well before this ever runs.
            state.hand.remove(card_def)
            if not target_still_legal(state, captured):
                state.graveyard.append(card_def)
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

        push_to_stack(state, card_def, _resolve)

    resolution.begin_choose_any_target(
        state,
        lambda p: target_predicate(p) and can_be_targeted(state, p, state.active_idx),
        _on_target_chosen,
        allow_players=False,
    )


def enters_battlefield(state, card_def, force_tapped=False, from_zone=None):
    """Move a CardDef onto the battlefield as a new Permanent, applying its
    enters-tapped default and QUEUING its ETB triggered ability (via
    game.registry.EFFECT_REGISTRY) onto state.trigger_queue for the priority
    round to put on the stack -- not running it inline (see the ETB block
    below for why), then run _check_end_of_game as a co-located end-of-game
    check after the board mutation (see win_check.py). Caller has already
    removed card_def from its previous zone (hand/library).

    force_tapped=True overrides the registry's own enters_tapped default
    to always-tapped -- a one-off per-trigger condition, not a property of
    the card itself (Sneaky Snacker enters battlefield normally untapped
    when cast, but tapped specifically when its own "third card drawn"
    trigger returns it from the graveyard
    item 7). Every existing caller omits it, unaffected."""
    spec = registry.EFFECT_REGISTRY.get(card_def.effect_id, {})
    # enters_tapped may be a plain bool OR a callable(state) -> bool, for a
    # land whose tapped-ness depends on the board (Gingerbread Cabin: tapped
    # unless you control 3+ other Forests). Evaluated BEFORE this permanent
    # is added to the battlefield below, so "other" counts exclude it.
    raw_tapped = spec.get("enters_tapped", False)
    tapped = force_tapped or (raw_tapped(state) if callable(raw_tapped) else raw_tapped)
    permanent = Permanent(card_def, tapped=tapped)
    # Record how it entered so an ETB conditioned on entering untapped
    # (Gingerbread Cabin's Food) reads the entry state, not whatever the
    # permanent's tapped-ness has since become (it could be tapped for mana
    # in the priority window before the queued ETB resolves).
    permanent.flags["entered_tapped"] = tapped
    # Pooled slot assignment: the lowest number not
    # already in use among this player's currently-live permanents of the
    # same name. Never a running/monotonic count -- a name's slot numbers
    # simply free up once whatever was using them leaves the battlefield,
    # which is what keeps this bounded (by how many can be simultaneously
    # alive, i.e. decklist quantity) even through repeated bounce/blink,
    # rather than growing with how many turns have been played.
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

    # The permanent has now physically entered (above) -- but its ETB
    # triggered ability, faithfully (real Magic 603.3), does NOT resolve
    # inline here: it goes on the stack the next time a player would receive
    # priority, with a response window before it resolves. So this only
    # QUEUES the trigger (state.trigger_queue); game.turn's own priority
    # round promotes it onto the stack once the enclosing action is done
    # (game.effects.triggers._trigger_resolve's own "etb" branch runs the
    # registry hook then). The etb_trigger callable receives (state,
    # permanent) -- the permanent that just entered, so an ETB that needs
    # its own source can reach it (Mesmeric Fiend links its exiled card to
    # this exact permanent for the leaves-the-battlefield return); the great
    # majority of ETBs ignore the second arg.
    if spec.get("etb_trigger") is not None:
        state.trigger_queue.append({"type": "etb", "card_def": card_def, "permanent": permanent})

    _check_end_of_game(state)

    return permanent


def bounce_land_etb(state):
    """ETB: return a land you control to hand (Rakdos Carnarium --
   ). resolution.begin_choose_permanent
    already covers "pick one of my own permanents matching a predicate, by
    exact (name, slot)" exactly -- no new resolution kind needed. Not a
    real MTG "target" (no "target" in this ability's own text -- it's an
    instruction executed entirely as this ETB fires, same as a plain "you
    may" instruction), so no cast-time/resolve-time gap exists here at all
    -- this whole function runs synchronously, unlike cast_aura's own
    deferred-to-the-stack targeting. enters_battlefield appends the
    permanent before running its ETB trigger, so the land that just
    entered is itself already a legal choice here, matching the real-rules
    guarantee "always at least one target" for free. Generic enough to
    share if a second land-bounce card ever needs it -- not
    Rakdos-Carnarium-specific despite currently having one caller."""
    def _on_chosen(state, choice):
        if choice is None:
            return  # begin_choose_permanent's own empty-battlefield safety net -- never reachable in practice, since this card is itself a legal land choice the moment it's on the battlefield
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


if __name__ == "__main__":
    # ponytail self-check: run via `python -m game.effects.casting` from
    # src/. Land bounce, then Aura casting end to end (choose target ->
    # attach -> pt_bonus visible via stats.py) plus the fizzle path (real
    # rule this whole targeting redesign exists to enforce).
    from ..cards import CardDef, EffectId
    from ..state import GameState
    from .stack import resolve_top_of_stack
    from .triggers import promote_triggers_to_stack
    from . import stats

    _filler_backup = registry.EFFECT_REGISTRY[EffectId.FILLER]
    registry.EFFECT_REGISTRY[EffectId.FILLER] = {"etb_trigger": lambda state, permanent: bounce_land_etb(state)}
    try:
        carnarium = CardDef("Fake Carnarium", CardType.LAND, None, EffectId.FILLER)
        state = GameState(on_the_play=True)
        state.hand = [carnarium]
        state.battlefield = [
            Permanent(CardDef("Forest", CardType.LAND, None, EffectId.FOREST)),
            Permanent(CardDef("Swamp", CardType.LAND, None, EffectId.SWAMP)),
        ]

        state.hand.remove(carnarium)
        enters_battlefield(state, carnarium)  # normal ETB path, exactly like play_land_from_hand would drive it
        # The ETB is QUEUED now (faithful timing), not run inline -- it opens
        # its choose_permanent only once promoted to the stack and resolved,
        # exactly as game.turn's priority round drives it in real play.
        assert state.pending_resolution is None
        assert [e["type"] for e in state.trigger_queue] == ["etb"]
        promote_triggers_to_stack(state)
        resolve_top_of_stack(state)
        assert state.pending_resolution["kind"] == "choose_permanent"
        assert resolution.choose_permanent_options(state) == [
            ("Fake Carnarium", 1), ("Forest", 1), ("Swamp", 1),
        ]
        resolution.execute_choose_permanent_option(state, "Swamp", 1)
        assert state.pending_resolution is None
        assert sorted(p.card_def.name for p in state.battlefield) == ["Fake Carnarium", "Forest"]
        assert [c.name for c in state.hand] == ["Swamp"]
    finally:
        registry.EFFECT_REGISTRY[EffectId.FILLER] = _filler_backup

    print("casting.py land-bounce self-check: OK")

    state = GameState(on_the_play=True)
    bogle = Permanent(CardDef("Slippery Bogle", CardType.CREATURE, {"G": 1}, EffectId.SLIPPERY_BOGLE, power=1, toughness=1))
    state.battlefield = [bogle]
    assert stats.permanent_power(state, bogle) == 1
    assert stats.permanent_toughness(state, bogle) == 1

    rancor = CardDef("Rancor", CardType.ENCHANTMENT, {"G": 1}, EffectId.RANCOR)
    state.hand = [rancor]
    cast_aura(state, rancor, lambda p: p.card_def.card_type == CardType.CREATURE)
    assert resolution.choose_any_target_creature_options(state) == [(0, "Slippery Bogle", 1)]  # own creature, side 0
    resolution.execute_choose_any_target_creature(state, 0, "Slippery Bogle", 1)
    assert state.pending_resolution is None
    assert state.hand == [rancor] and len(state.stack) == 1  # still in hand, sitting on the stack
    resolve_top_of_stack(state)
    assert state.hand == []
    rancor_permanent = next(p for p in state.battlefield if p.card_def.name == "Rancor")
    assert rancor_permanent.flags["enchanting"] is bogle
    assert stats.permanent_power(state, bogle) == 3  # 1 base + Rancor's own +2
    assert stats.permanent_toughness(state, bogle) == 1  # unchanged -- Rancor is +2/+0

    print("casting.py Aura self-check: OK")

    # Fizzle: the EXACT permanent chosen as a target is gone by the time
    # the spell resolves -- died, was sacrificed, bounced, doesn't matter
    # how -- the whole spell fails outright, no effect, straight to the
    # graveyard, never even entering the battlefield.
    import contextlib
    import io

    other_bogle = enters_battlefield(
        state, CardDef("Slippery Bogle", CardType.CREATURE, {"G": 1}, EffectId.SLIPPERY_BOGLE, power=1, toughness=1),
    )
    assert other_bogle.slot == 2  # bogle (still on the battlefield) already occupies slot 1

    ethereal_armor = CardDef("Ethereal Armor", CardType.ENCHANTMENT, {"W": 1}, EffectId.ETHEREAL_ARMOR)
    state.hand = [ethereal_armor]
    cast_aura(state, ethereal_armor, lambda p: p.card_def.card_type == CardType.CREATURE)
    assert (0, "Slippery Bogle", 2) in resolution.choose_any_target_creature_options(state)
    resolution.execute_choose_any_target_creature(state, 0, "Slippery Bogle", 2)  # targets other_bogle specifically
    state.battlefield.remove(other_bogle)  # dies before the cast resolves

    fizzle_log = io.StringIO()
    with contextlib.redirect_stdout(fizzle_log):
        resolve_top_of_stack(state)
    assert "fizzle" in fizzle_log.getvalue().lower()
    assert state.hand == []
    assert ethereal_armor in state.graveyard
    assert not any(p.card_def.name == "Ethereal Armor" for p in state.battlefield)
    assert stats.permanent_power(state, bogle) == 3  # unaffected -- the fizzled Aura was never targeting bogle

    print("casting.py Aura target-fizzle self-check: OK")

    # capture_any_target / target_still_legal: the shared cast-time lock +
    # resolution-time legality for begin_choose_any_target. Two same-named
    # creatures on opposite sides -- capture must lock the EXACT one named
    # by (side, name, slot), and legality must flip only when that specific
    # object leaves, not when its same-named twin does.
    from ..state import PlayerState, Permanent
    mine = Permanent(CardDef("Grizzly Bears", CardType.CREATURE, {"G": 1}, EffectId.FILLER, power=2, toughness=2))
    theirs = Permanent(CardDef("Grizzly Bears", CardType.CREATURE, {"G": 1}, EffectId.FILLER, power=2, toughness=2))
    mine.slot = theirs.slot = 1
    tstate = GameState(on_the_play=True, players=[PlayerState(True), PlayerState(False)])
    tstate.players[0].battlefield = [mine]
    tstate.players[1].battlefield = [theirs]
    captured = capture_any_target(tstate, ("creature", 1, "Grizzly Bears", 1))
    assert captured == ("creature", theirs)  # the opponent's copy, by identity -- not mine
    assert target_still_legal(tstate, captured)
    tstate.players[0].battlefield = []  # MY copy leaves -- the captured target (theirs) is untouched
    assert target_still_legal(tstate, captured)
    tstate.players[1].battlefield = []  # the captured copy leaves -> now illegal (fizzle)
    assert not target_still_legal(tstate, captured)
    assert capture_any_target(tstate, ("player", 0)) == ("player", 0)  # players pass through, always legal
    assert target_still_legal(tstate, ("player", 0))
    assert capture_any_target(tstate, None) is None and target_still_legal(tstate, None)  # no target -> no fizzle

    print("casting.py any-target capture/legality self-check: OK")
