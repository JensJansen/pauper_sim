"""Generic effect plumbing every color catalog's own cast/activate
functions call into: battlefield entry, combat, the Madness/Plot/token
mechanics, and the trigger queue. No per-card catalog entries live here
anymore -- every real card (including ones once "shared between decks"
like Forest/Lightning Bolt/Generous Ent) now lives exactly once in its
own color file under game/catalog/, since cards are no longer owned by
decks at all (DECK_REGISTRY_REFRESH_PLAN.md). The two token CardDefs
below (Blood/Robot) are the one exception -- they're never in CARD_DEFS
or any decklist, so a color file isn't the right home for them either.

References game.registry.EFFECT_REGISTRY (the merged registry, built from
every color catalog's own fragment plus registry.py's union) only from
inside function bodies, via `registry.EFFECT_REGISTRY`, never as a bare
name imported at module load time -- registry.py imports the catalog
modules, which import this module, so a `from .registry import
EFFECT_REGISTRY` here would try to bind a name that doesn't exist yet.
Deferring the lookup to call time (Python resolves a name inside a
function body only when that function runs, not when it's defined) breaks
the cycle: by the time anything actually calls enters_battlefield,
game/__init__.py has already finished importing every submodule.

Also holds the Madness "cast for its madness cost" path (execute_madness_
cast), the Plot "pay cost, exile instead of resolving" path (plot_to_
exile), and the trigger-queue drain (drain_trigger_queue) -- these need
game.mana.begin_pay_cost, which game.resolution can't import (mana.py
imports resolution.py at its own top level; the reverse would cycle).
This module is free to depend on both resolution.py and mana.py, so it's
where that orchestration lives instead. See docs/MADNESS_DECKS_PLAN.md
items 1/3/4/7.
"""

from . import mana, registry, resolution
from .cards import CardDef, CardType, EffectId
from .state import Permanent


def play_land_from_hand(state, card_def):
    state.hand.remove(card_def)
    state.lands_played_this_turn += 1
    return enters_battlefield(state, card_def)


def cast_permanent_from_hand(state, card_def):
    """Artifacts/creatures with no additional cost beyond mana and no
    target choices. Mana cost is paid by the caller first."""
    state.hand.remove(card_def)
    return enters_battlefield(state, card_def)


def enters_battlefield(state, card_def, force_tapped=False):
    """Move a CardDef onto the battlefield as a new Permanent, applying its
    enters-tapped default and ETB trigger (via game.registry.EFFECT_REGISTRY),
    then check state.terminated_fn since a permanent entering is the only
    way any win condition discussed so far can newly become true. Caller
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
    state.battlefield.append(permanent)

    etb_trigger = spec.get("etb_trigger")
    if etb_trigger is not None:
        etb_trigger(state)
    # Bojuka Bog's "exile target player's graveyard" ETB is a documented
    # no-op: no opposing graveyard exists in this solitaire simulator.

    if state.terminated_fn is not None and state.turn_won is None and state.terminated_fn(state):
        state.turn_won = state.turn_number

    return permanent


def combat_step(state):
    """Fully automatic mini-combat (rakdos madness / mono red madness
    only -- see game.turn.run_turn's combat_enabled param): every
    creature that is both untapped and not summoning sick deals damage
    equal to its power once, then taps. No attack/block decisions, no
    blockers (this simulator has no opposing board, only the running
    state.damage_dealt counter) -- a creature already tapped for
    something else earlier in the turn is how a model "holds it back"
    instead of attacking. Power lives in card_def.extra["power"]; a
    creature with none set (e.g. an untracked-stats vanilla from another
    deck) contributes 0, so calling this unconditionally would already be
    harmless -- combat_enabled still gates it structurally so no other
    deck's creatures get tapped for a step that isn't theirs.

    Checked against terminated_fn here directly (mirrors
    enters_battlefield's own check) -- combat is a second way
    state.damage_dealt can cross a threshold, not just a permanent
    entering.

    A creature with a registry "haste": True spec (Kitchen Imp) ignores
    its own summoning_sick flag here -- the only other place that flag
    is ever read, so this is the only place haste needs to matter."""
    attackers = [
        p for p in state.battlefield
        if p.card_def.card_type == CardType.CREATURE and not p.tapped
        and (not p.summoning_sick or registry.EFFECT_REGISTRY.get(p.card_def.effect_id, {}).get("haste", False))
    ]
    state.damage_dealt += sum(p.card_def.extra.get("power", 0) for p in attackers)
    for p in attackers:
        p.tapped = True
    if state.terminated_fn is not None and state.turn_won is None and state.terminated_fn(state):
        state.turn_won = state.turn_number


def find_and_remove_by_name(state, name):
    """Search state.library for the first card matching `name`, remove and
    return it (or None if absent). Does not shuffle -- callers shuffle per
    their own card's rules."""
    for i, c in enumerate(state.library):
        if c.name == name:
            return state.library.pop(i)
    return None


def on_cast_trigger(state, card_def):
    """Fires at cast time, before the spell's own resolve runs -- matches
    real Magic timing (the triggered ability goes on the stack above the
    spell it triggered off, so it happens first). Every cast path (normal
    cast, alt_cast, Flashback, and Plot's cast-from-exile) calls this
    identically, from inside drl_env.py's own per-path wrapper functions
    -- so a card like Guttersnipe ("whenever you cast an instant or
    sorcery...") fires the same regardless of which path cast the
    triggering spell. See docs/MADNESS_DECKS_PLAN.md item 11."""
    if card_def.card_type not in (CardType.INSTANT, CardType.SORCERY):
        return
    for permanent in state.battlefield:
        trigger = registry.EFFECT_REGISTRY.get(permanent.card_def.effect_id, {}).get("on_cast")
        if trigger is not None:
            trigger(state, permanent)


def drain_trigger_queue(state):
    """Called once no other resolution is pending -- never mid-resolution
    (see docs/MADNESS_DECKS_PLAN.md items 1/3's cross-cutting rule: a
    card's entire effect finishes before anything it queued gets acted
    on). Runs every "automatic" entry immediately, in order (Sneaky
    Snacker's return, item 7 -- no choice involved, so it never becomes a
    pending_resolution), and stops at the first "decision" entry, turning
    it into a real pending_resolution for the model to act on. A decision
    entry left mid-queue resumes being drained the next time this is
    called with nothing else pending (i.e. once that decision itself
    completes). No-op if the queue is empty or something else is already
    pending."""
    if state.pending_resolution is not None:
        return
    while state.trigger_queue:
        entry = state.trigger_queue.pop(0)
        if entry["type"] == "automatic":
            if entry["kind"] == "on_draw_count":
                card_def = entry["card_def"]
                state.graveyard.remove(card_def)
                enters_battlefield(state, card_def, force_tapped=True)
                continue
            raise ValueError(f"unknown automatic trigger queue entry: {entry}")
        if entry["type"] == "decision":
            if entry["kind"] == "madness":
                resolution.begin_madness_decision(
                    state, entry["card_def"], on_complete=lambda s: drain_trigger_queue(s),
                )
            return  # a decision is now pending -- stop, let the model act on it
        raise ValueError(f"unknown trigger queue entry: {entry}")


def execute_madness_cast(state):
    """Model chose "cast" for a pending madness_decision: pay the card's
    madness cost, then resolve it exactly like any other cast, then let
    the enclosing decision's own on_complete fire (drain_trigger_queue,
    continuing to whatever's queued next). Mirrors drl_env.py's
    _flashback_execute shape (pay, then resolve) -- just entered from
    inside a pending_resolution instead of a top-level action.

    Captures card_def/on_complete from the CURRENT pending_resolution
    before begin_pay_cost overwrites it with its own "pay_cost" one --
    same nested-callback shape flashback_dread_return
    (game.catalog.black_cards) already uses for its own multi-step
    chain."""
    pending = state.pending_resolution
    card_def = pending["card_def"]
    outer_on_complete = pending["on_complete"]
    madness_spec = registry.EFFECT_REGISTRY[card_def.effect_id]["madness"]
    resolution._remove_one_from_exile(state, card_def)

    def _after_pay(s):
        madness_spec["resolve"](s, card_def)
        outer_on_complete(s)

    mana.begin_pay_cost(state, madness_spec["cost"], on_complete=_after_pay)


def plot_to_exile(state, card_def):
    """Plot's own resolve shape (MADNESS_DECKS_PLAN.md item 4): pay the
    plot cost, move hand -> exile with this turn's stamp instead of
    running the card's real effect. Generic Plot-mechanic plumbing (any
    future "plot" card reuses this unchanged, same precedent as
    execute_madness_cast above) -- currently only Highway Robbery
    (game.catalog.red_cards) has a "plot" registry spec."""
    state.hand.remove(card_def)
    state.exile.append((card_def, state.turn_number))


def bounce_land_etb(state):
    """ETB: return a land you control to hand (Rakdos Carnarium --
    docs/MADNESS_DECKS_PLAN.md item 10). resolution.begin_choose_permanent
    already covers "pick one of my own permanents matching a predicate,
    by name" exactly -- no new resolution kind needed. enters_battlefield
    appends the permanent before running its ETB trigger, so the land
    that just entered is itself already a legal target here, matching the
    real-rules guarantee "always at least one target" for free. Generic
    enough to share if a second land-bounce card ever needs it -- not
    Rakdos-Carnarium-specific despite currently having one caller."""
    def _on_chosen(state, name):
        if name is None:
            return  # begin_choose_permanent's own empty-battlefield safety net -- never reachable in practice, since this card is itself a legal land target the moment it's on the battlefield
        permanent = next(p for p in state.battlefield if p.card_def.name == name and p.card_def.card_type == CardType.LAND)
        state.battlefield.remove(permanent)
        state.hand.append(permanent.card_def)

    resolution.begin_choose_permanent(state, lambda p: p.card_def.card_type == CardType.LAND, _on_chosen)


def create_token(state, card_def, tapped=False):
    """A token permanent, not backed by any library/CARD_DEFS card --
    Melded Moxite's Robot, Voldaren Epicure/Vampire's Kiss's Blood
    (docs/MADNESS_DECKS_PLAN.md item 8). Reuses enters_battlefield's full
    battlefield-entry path (ETB dispatch, terminated_fn check) unchanged
    -- a token entering the battlefield is exactly as real as any other
    permanent from here on; only its creation (no hand/library removal
    beforehand) is different. tapped=True covers "Create a TAPPED 2/2
    Robot" (Melded Moxite's own wording); Blood tokens enter untapped."""
    return enters_battlefield(state, card_def, force_tapped=tapped)


def activate_blood_sac(state, permanent):
    """Blood's {1}, {T}, Discard a card, Sacrifice this token: Draw a
    card. The {1} mana and the untapped precondition are both already
    handled generically by drl_env.py's cost_key-based activated-ability
    wiring (same as Candy Trail's own sac ability, which has no {T}
    symbol at all yet gets the identical untapped check for free today)
    -- this only covers what's specific to Blood: sacrifice (a token
    ceases to exist once it leaves the battlefield, per real Magic's own
    state-based action -- never added to the graveyard, unlike a real
    card), then discard a card (reusing item 1's begin_discard directly,
    which is what makes Madness-awareness automatic for whatever gets
    discarded this way), then draw."""
    state.battlefield.remove(permanent)
    resolution.begin_discard(state, 1, optional=False, on_complete=lambda s, _cards: s.draw(1))


BLOOD_TOKEN_CARD_DEF = CardDef("Blood", CardType.ARTIFACT, None, EffectId.BLOOD_TOKEN, sac_ability_cost={"generic": 1})
ROBOT_TOKEN_CARD_DEF = CardDef("Robot", CardType.CREATURE, None, EffectId.ROBOT_TOKEN, power=2)  # 2/2, combat_step's only reader of "power"


if __name__ == "__main__":
    # ponytail self-check: no pytest in this project, mirrors the
    # assert-based demo convention -- run via `python -m game.effects_common`
    # from src/. Exercises the full Madness chain end to end (discard ->
    # exile + queue -> drain -> decision -> pay madness cost -> resolve ->
    # drain again) against a hand-built state. No real madness card exists
    # yet (deck assembly is out of scope for this plan), so this borrows
    # EffectId.FILLER for the fake spell, saving/restoring its real (empty)
    # registry entry around the check; the mana source is a genuine Forest
    # CardDef (EffectId.FOREST already has a real "mana" spec, no faking
    # needed for that half).
    from .cards import CardDef, CardType
    from .state import GameState, Permanent

    resolved_calls = []

    def _fake_resolve(s, c):
        resolved_calls.append(c.name)

    _filler_backup = registry.EFFECT_REGISTRY[EffectId.FILLER]
    registry.EFFECT_REGISTRY[EffectId.FILLER] = {"madness": {"cost": {"G": 1}, "resolve": _fake_resolve}}
    try:
        madness_card = CardDef("Fake Madness Spell", CardType.INSTANT, {"generic": 1, "R": 1}, EffectId.FILLER)
        state = GameState(on_the_play=True)
        state.hand = [madness_card]
        state.battlefield = [Permanent(CardDef("Forest", CardType.LAND, None, EffectId.FOREST))]

        completed = []
        resolution.begin_discard(state, 1, optional=False, on_complete=lambda s, cards: completed.append(cards))
        resolution.execute_discard_option(state, "Fake Madness Spell")
        assert completed == [[madness_card]]
        assert state.pending_resolution is None  # discard's own resolution is done; nothing queued mid-discard
        assert len(state.trigger_queue) == 1 and state.trigger_queue[0]["kind"] == "madness"

        # Top-level orchestration point (drl_env.py's job in real play,
        # once the enclosing top-level action is fully done): drain.
        drain_trigger_queue(state)
        assert state.pending_resolution["kind"] == "madness_decision"
        assert resolution.madness_decision_options(state) == ["cast", "decline"]

        execute_madness_cast(state)
        # begin_pay_cost's own resolution is now pending (paying {G}) --
        # confirms this correctly nests through it via the captured
        # outer_on_complete, instead of crashing on a stale
        # pending_resolution reference once pay_cost's own resolution
        # clears it.
        assert state.pending_resolution["kind"] == "pay_cost"
        assert mana.tap_cost_options(state) == [("Forest", None, False)]
        mana.execute_tap_cost_option(state, "Forest", None, False)

        # Payment complete -> resolve fired -> outer on_complete (drain
        # again, queue now empty) -> fully back to no pending resolution.
        assert resolved_calls == ["Fake Madness Spell"]
        assert state.pending_resolution is None
        assert state.trigger_queue == []
        assert state.exile == []
    finally:
        registry.EFFECT_REGISTRY[EffectId.FILLER] = _filler_backup

    print("effects_common.py madness self-check: OK")

    # Per-turn draw counter + Sneaky Snacker-style automatic return
    # (MADNESS_DECKS_PLAN.md item 7). No real Sneaky Snacker card exists
    # yet, so this borrows EffectId.FILLER again for an "on_draw_count"
    # spec. Two copies sit in the graveyard, to confirm the faithful
    # multi-copy handling (each queues its own return, not capped at one).
    _filler_backup = registry.EFFECT_REGISTRY[EffectId.FILLER]
    registry.EFFECT_REGISTRY[EffectId.FILLER] = {"on_draw_count": {"count": 3}}
    try:
        snacker = CardDef("Fake Snacker", CardType.CREATURE, {"generic": 1}, EffectId.FILLER)
        state = GameState(on_the_play=True)
        state.library = [CardDef(f"Filler {i}", CardType.SORCERY, {}, None) for i in range(5)]
        state.graveyard = [snacker, snacker]  # two physical copies

        state.draw(1)
        assert state.cards_drawn_this_turn == 1 and state.trigger_queue == []
        state.draw(1)
        assert state.cards_drawn_this_turn == 2 and state.trigger_queue == []
        state.draw(1)  # the third card this turn -- both copies trigger
        assert state.cards_drawn_this_turn == 3
        assert len(state.trigger_queue) == 2
        assert all(e == {"type": "automatic", "kind": "on_draw_count", "card_def": snacker} for e in state.trigger_queue)

        state.draw(1)  # a 4th draw must NOT re-trigger (exactly == 3, not >= 3)
        assert len(state.trigger_queue) == 2

        # Draining is fully automatic -- no decision, no pending_resolution
        # ever appears, both copies return to the battlefield tapped.
        drain_trigger_queue(state)
        assert state.pending_resolution is None
        assert state.trigger_queue == []
        assert state.graveyard == []
        assert len(state.battlefield) == 2
        assert all(p.card_def is snacker and p.tapped for p in state.battlefield)
    finally:
        registry.EFFECT_REGISTRY[EffectId.FILLER] = _filler_backup

    print("effects_common.py draw-counter self-check: OK")

    # Tokens (MADNESS_DECKS_PLAN.md item 8): create_token reuses
    # enters_battlefield unchanged; Blood's sac ability reuses
    # begin_discard directly (so a discarded Madness card is still
    # correctly caught -- exercised here via the same FILLER-as-madness-
    # card trick used above).
    state = GameState(on_the_play=True)
    create_token(state, ROBOT_TOKEN_CARD_DEF, tapped=True)
    create_token(state, BLOOD_TOKEN_CARD_DEF)  # untapped by default
    assert [(p.card_def.name, p.tapped) for p in state.battlefield] == [("Robot", True), ("Blood", False)]

    _filler_backup = registry.EFFECT_REGISTRY[EffectId.FILLER]
    registry.EFFECT_REGISTRY[EffectId.FILLER] = {"madness": {"cost": {}, "resolve": lambda s, c: None}}
    try:
        blood = next(p for p in state.battlefield if p.card_def.name == "Blood")
        other_card = CardDef("Fake Madness Card", CardType.SORCERY, {}, EffectId.FILLER)
        state.hand = [other_card]
        state.library = [CardDef("Library Card", CardType.SORCERY, {}, None)]

        drawn_before = len(state.hand)
        activate_blood_sac(state, blood)  # cost payment ({1} mana) is drl_env.py's concern, not this function's -- see build_action_table's token_card_defs
        assert state.pending_resolution["kind"] == "discard"
        assert resolution.discard_options(state) == ["Fake Madness Card"]
        resolution.execute_discard_option(state, "Fake Madness Card")

        # Sacrificed: gone, and never added to any zone (a token ceases
        # to exist, unlike a real card being discarded/sacrificed).
        assert [p.card_def.name for p in state.battlefield] == ["Robot"]
        # The discarded card had a "madness" spec -- queued, not graveyard.
        assert state.graveyard == []
        assert len(state.trigger_queue) == 1 and state.trigger_queue[0]["kind"] == "madness"
        drain_trigger_queue(state)
        resolution.execute_madness_decline(state)  # free madness cost isn't the point of this check -- just confirm routing
        assert [c.name for c in state.graveyard] == ["Fake Madness Card"]
        # Then the draw fires (begin_discard's on_complete): net hand size
        # unchanged (lost the discarded card, gained one drawn).
        assert len(state.hand) == drawn_before  # started at 1 (other_card), discarded it, drew 1 -- still 1
    finally:
        registry.EFFECT_REGISTRY[EffectId.FILLER] = _filler_backup

    print("effects_common.py tokens self-check: OK")

    # Land bounce (item 10): no real Rakdos Carnarium card exists yet, so
    # this borrows EffectId.FILLER's etb_trigger to exercise
    # bounce_land_etb directly through the real enters_battlefield path
    # (same as any other ETB trigger). Two other lands already in play,
    # to confirm the model has a real choice, not just "the card itself."
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
        # All three lands are legal targets, including itself -- it's
        # already on the battlefield by the time its own trigger runs.
        assert resolution.choose_permanent_options(state) == ["Fake Carnarium", "Forest", "Swamp"]

        resolution.execute_choose_permanent_option(state, "Swamp")
        assert state.pending_resolution is None
        assert sorted(p.card_def.name for p in state.battlefield) == ["Fake Carnarium", "Forest"]
        assert [c.name for c in state.hand] == ["Swamp"]

        # Bouncing itself is also legal (real-rules guarantee: always at
        # least one target, satisfied even with only 1 land in play).
        state = GameState(on_the_play=True)
        state.battlefield = [Permanent(CardDef("Fake Carnarium 2", CardType.LAND, None, EffectId.FILLER))]
        bounce_land_etb(state)  # exercised directly this time, not via enters_battlefield -- same underlying call
        resolution.execute_choose_permanent_option(state, "Fake Carnarium 2")
        assert state.battlefield == []
        assert [c.name for c in state.hand] == ["Fake Carnarium 2"]
    finally:
        registry.EFFECT_REGISTRY[EffectId.FILLER] = _filler_backup

    print("effects_common.py land-bounce self-check: OK")

    # Combat (rakdos madness / mono red madness only -- game.turn.run_turn's
    # combat_enabled param gates whether combat_step ever actually gets
    # called; this exercises combat_step itself directly). Five permanents,
    # each proving one eligibility rule.
    state = GameState(on_the_play=True, terminated_fn=lambda s: s.damage_dealt >= 5)
    attacker = Permanent(CardDef("Attacker", CardType.CREATURE, None, EffectId.FILLER, power=3))
    attacker.summoning_sick = False
    sick = Permanent(CardDef("Sick", CardType.CREATURE, None, EffectId.FILLER, power=10))  # summoning_sick=True by construction -- never cleared here (that's untap_step's job)
    already_tapped = Permanent(CardDef("Tapped Out", CardType.CREATURE, None, EffectId.FILLER, power=10), tapped=True)
    already_tapped.summoning_sick = False
    vanilla = Permanent(CardDef("No Stats", CardType.CREATURE, None, EffectId.FILLER))  # no "power" key at all -- untracked-stats precedent (Masked Vandal, Mesmeric Fiend)
    vanilla.summoning_sick = False
    not_a_creature = Permanent(CardDef("Some Land", CardType.LAND, None, EffectId.FILLER, power=10))
    not_a_creature.summoning_sick = False
    state.battlefield = [attacker, sick, already_tapped, vanilla, not_a_creature]

    combat_step(state)
    assert state.damage_dealt == 3  # only Attacker's power counts -- sick/already-tapped/non-creature all excluded
    assert attacker.tapped and vanilla.tapped  # both attacked (vanilla's 0 power still taps it, same as a real 0-power creature attacking)
    assert not sick.tapped and not not_a_creature.tapped  # never touched
    assert state.turn_won is None  # 3 < 5, not lethal yet

    combat_step(state)  # called again with nothing untapped in between (no intervening untap_step) -- everyone still tapped/sick from before, so this must be a no-op
    assert state.damage_dealt == 3

    attacker.tapped = False  # simulate the next turn's untap_step (sick's flag deliberately left alone -- it's still sick until a real untap_step clears it)
    combat_step(state)
    assert state.damage_dealt == 6  # crosses the >=5 threshold
    assert state.turn_won == 0  # terminated_fn caught it here directly, same check enters_battlefield uses -- state.turn_number was never advanced in this synthetic test

    # Haste (Kitchen Imp): a "haste": True registry spec lets a summoning-
    # sick creature attack anyway -- the only place that spec is ever read.
    state = GameState(on_the_play=True)
    _filler_backup = registry.EFFECT_REGISTRY[EffectId.FILLER]
    registry.EFFECT_REGISTRY[EffectId.FILLER] = {"haste": True}
    try:
        hasty = Permanent(CardDef("Hasty", CardType.CREATURE, None, EffectId.FILLER, power=2))
        assert hasty.summoning_sick  # True by construction -- just entered, never untapped
        state.battlefield = [hasty]
        combat_step(state)
        assert state.damage_dealt == 2 and hasty.tapped  # attacked anyway, despite being summoning sick
    finally:
        registry.EFFECT_REGISTRY[EffectId.FILLER] = _filler_backup

    print("effects_common.py combat self-check: OK")
