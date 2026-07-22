"""Battlefield entry and the cast paths that put something there directly:
playing a land, casting a plain permanent, casting an Aura (with its own
choose-target-then-resolve dance), and one ETB effect (land bounce)
generic enough to live beside enters_battlefield rather than in a color
catalog. tokens.py builds on enters_battlefield too (see that module).

Depends on stack.py for push_to_stack (cast_aura's targeted spell still
goes through the stack like any other cast) and win_check.py for the
game-end check every battlefield change can newly trigger. Does NOT
depend on triggers.py -- triggers.promote_triggers_to_stack needs
enters_battlefield (an automatic trigger can return a card to the
battlefield), so that dependency has to point the other way, or the two
modules would need each other. See triggers.py's own module docstring.

References game.registry.EFFECT_REGISTRY only from inside function
bodies, via `registry.EFFECT_REGISTRY` -- registry.py imports the catalog
modules, which import this module, so a `from .registry import
EFFECT_REGISTRY` here would try to bind a name that doesn't exist yet.
Deferring the lookup to call time breaks the cycle. See game/registry.py's
own module docstring."""

from .. import registry
from ..cards import CardType
from ..state import Permanent
from .stack import push_to_stack
from .win_check import _check_end_of_game
from .. import resolution


def play_land_from_hand(state, card_def):
    state.hand.remove(card_def)
    state.lands_played_this_turn += 1
    return enters_battlefield(state, card_def)


def cast_permanent_from_hand(state, card_def):
    """Artifacts/creatures with no additional cost beyond mana and no
    target choices. Mana cost is paid by the caller first."""
    state.hand.remove(card_def)
    return enters_battlefield(state, card_def)


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


def cast_aura(state, card_def, target_predicate, on_attached=None):
    """Cast an Aura from hand: pick a legal target via
    resolution.begin_choose_permanent, addressed by the EXACT (name, slot)
    permanent chosen -- not just a name, since two same-named permanents
    stop being interchangeable the instant an Aura attaches to only one of
    them (docs/MULTIPLAYER_GAPS.md's "Permanent identity").

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
    card pool, combat death (docs/COMBAT_PLAN.md step 6, see
    state_based._destroy_creature). Every OTHER battlefield-removal call
    site in this codebase (sacrifice, bounce, exile -- see their own call
    sites) still doesn't orphan an enchanted permanent's Auras, since none
    of them can currently target a creature that could be enchanted
    (boggles is the only deck with Auras, and none of its own cards
    sacrifice/bounce/exile a creature). Thread the same orphaning logic
    through a removal site if a future card ever makes that reachable."""
    def _on_target_chosen(state, choice):
        target = None
        if choice is not None:
            name, slot = choice
            target = next(
                p for p in state.battlefield
                if p.card_def.name == name and p.slot == slot and target_predicate(p)
            )

        def _resolve(state, card_def):
            # Still-in-hand-while-on-stack convention every other cast path
            # here follows (see push_to_stack's own docstring) -- the target
            # is already locked in via the `target`/`choice` closure above,
            # captured at cast time, well before this ever runs.
            state.hand.remove(card_def)
            if target is None or target not in state.battlefield:
                state.graveyard.append(card_def)
                _log_target_fizzle(state, card_def, choice)
                return
            aura = enters_battlefield(state, card_def)
            aura.flags["enchanting"] = target
            if on_attached is not None:
                on_attached(state, aura)

        push_to_stack(state, card_def, _resolve)

    resolution.begin_choose_permanent(state, target_predicate, _on_target_chosen)


def enters_battlefield(state, card_def, force_tapped=False):
    """Move a CardDef onto the battlefield as a new Permanent, applying its
    enters-tapped default and ETB trigger (via game.registry.EFFECT_REGISTRY),
    then check _check_end_of_game since a permanent entering is the only
    way a terminated_fn-based win condition can newly become true. Caller
    has already removed card_def from its previous zone (hand/library).

    force_tapped=True overrides the registry's own enters_tapped default
    to always-tapped -- a one-off per-trigger condition, not a property of
    the card itself (Sneaky Snacker enters battlefield normally untapped
    when cast, but tapped specifically when its own "third card drawn"
    trigger returns it from the graveyard -- docs/MADNESS_DECKS_PLAN.md
    item 7). Every existing caller omits it, unaffected."""
    spec = registry.EFFECT_REGISTRY.get(card_def.effect_id, {})
    tapped = force_tapped or spec.get("enters_tapped", False)
    permanent = Permanent(card_def, tapped=tapped)
    # Pooled slot assignment (docs/COMBAT_PLAN.md): the lowest number not
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

    etb_trigger = spec.get("etb_trigger")
    if etb_trigger is not None:
        etb_trigger(state)
    # Bojuka Bog's "exile target player's graveyard" ETB is a documented
    # no-op in both 1- and 2-player games: no card currently reaches into
    # this graveyard-exile mechanic, so it stays unimplemented regardless
    # of whether a real opponent graveyard now exists to target.

    _check_end_of_game(state)

    return permanent


def bounce_land_etb(state):
    """ETB: return a land you control to hand (Rakdos Carnarium --
    docs/MADNESS_DECKS_PLAN.md item 10). resolution.begin_choose_permanent
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

    resolution.begin_choose_permanent(state, lambda p: p.card_def.card_type == CardType.LAND, _on_chosen)


if __name__ == "__main__":
    # ponytail self-check: run via `python -m game.effects.casting` from
    # src/. Land bounce, then Aura casting end to end (choose target ->
    # attach -> pt_bonus visible via stats.py) plus the fizzle path (real
    # rule this whole targeting redesign exists to enforce).
    from ..cards import CardDef, EffectId
    from ..state import GameState
    from .stack import resolve_top_of_stack
    from . import stats

    _filler_backup = registry.EFFECT_REGISTRY[EffectId.FILLER]
    registry.EFFECT_REGISTRY[EffectId.FILLER] = {"etb_trigger": bounce_land_etb}
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
    assert resolution.choose_permanent_options(state) == [("Slippery Bogle", 1)]
    resolution.execute_choose_permanent_option(state, "Slippery Bogle", 1)
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
    assert ("Slippery Bogle", 2) in resolution.choose_permanent_options(state)
    resolution.execute_choose_permanent_option(state, "Slippery Bogle", 2)  # targets other_bogle specifically
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
