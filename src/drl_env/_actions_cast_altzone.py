"""Casting from a NON-hand zone or for a non-default cost: Land Grant's free
alt-cost, Dread Return/Sleep of the Dead's Flashback/Escape (graveyard),
Highway Robbery's Plot (exile), and Sagu Wildling/Boulderbranch Golem's
Omen/Prototype (a second cast option for the same hand card, its own cost).
Split out of drl_env._actions_cast -- that module's own docstring still
covers the plain-hand-cast half of table category B (normal/modal/X-cost/
Delve cast) plus categories A/C/D and impulse; this file is everything in
category B that does NOT cast card_def straight out of state.hand for
card_def.cast_cost. Each is a legal(state)/execute(state) factory pair
build_action_table (drl_env._actions_table) calls once per matching card,
same contract as every other _actions_* category module."""

import game

from ._actions_common import _GATE_NO_PENDING, _hand_count_available


def _alt_cast_legal(name, extra_legal, speed):
    """Land Grant's free alt-cost: no mana payment at all, just the
    card's own extra_legal predicate (0 lands in hand).

    Availability must go through _hand_count_available, not a bare
    "any copy in hand" check -- confirmed live via mono_red_madness_mirror
    training: a bare existence check let Fireblast's alt-cost (sacrifice 2
    Mountains) be cast a second time while the first cast's own copy was
    still physically in hand but already reserved on the stack (removal
    deferred to its own resolve, same as every cast-like path -- see
    push_to_stack's docstring), pushing a second stack entry for the same
    physical card. cast_fireblast_alt's own discard_from_hand_to_graveyard
    then ate that shared copy immediately (its eager, non-deferred
    hand-removal), so when the FIRST cast's stack entry finally resolved,
    its own discard_from_hand_to_graveyard found no copy left -- the
    "should be unreachable" RuntimeError."""
    def legal(state):
        if state.pending_resolution is not None:
            return False
        if not game.turn.speed_legal(state, speed):
            return False
        if _hand_count_available(state, name) <= 0:
            return False
        return extra_legal(state)
    legal._pending_gate = _GATE_NO_PENDING
    return legal


def _alt_cast_execute(name, resolve):
    """No generic engine-level cost mechanism for an alt cost (unlike mana's
    begin_pay_cost) -- so, same as _flashback_execute, this calls resolve
    immediately and leaves deferring-onto-the-stack entirely up to resolve
    itself. Alt-cost shapes vary: Land Grant's is free (nothing to pay, so
    its own resolve pushes right away, same as a free Flashback), Fireblast's
    is a real alternate cost (sacrifice 2 Mountains) that must actually be
    paid -- via its own resolution -- before ITS effect gets pushed. Pushing
    generically here, before resolve even runs, would defer Fireblast's own
    cost-payment along with its effect, which is wrong: the cost must be
    paid before anything is fully paid for and put on the stack."""
    def execute(state):
        card_def = game.CARD_DEFS[name]
        game.on_cast_trigger(state, card_def)  # item 11 -- see _actions_cast._cast_execute
        resolve(state, card_def)
    return execute


def _flashback_legal(name, ability_legal, speed, cost=None):
    """Dread Return's Flashback: cast from the graveyard, not hand. Real
    Magic: Flashback follows the same timing as the card itself, not its
    own independent rule -- speed is the same value the card's normal
    cast derived, not a separate default.

    cost (optional): a MANA cost dict for a flashback whose flashback cost
    includes mana (Deep Analysis' {1}{U}, Faithless Looting's {2}{R}). When
    present, its affordability is checked here (plan_payment) exactly like a
    normal cast; the truly free/sacrifice-only flashbacks (Lava Dart, Dread
    Return) leave it None and pay entirely inside their own resolve. Any
    NON-mana additional cost (Deep Analysis' "Pay 3 life") is gated by
    ability_legal instead (it can't be expressed as a mana dict)."""
    def legal(state):
        if state.pending_resolution is not None:
            return False
        if not game.turn.speed_legal(state, speed):
            return False
        if not any(c.name == name for c in state.graveyard):
            return False
        if cost is not None and game.plan_payment(state, cost) is None:
            return False
        return ability_legal(state)
    legal._pending_gate = _GATE_NO_PENDING
    return legal


def _graveyard_instance(state, name):
    """THE type->instance boundary for every graveyard-sourced action.

    The action table is built from card NAMES (it must be -- a deck's action
    space is fixed-width), so an execute closure only ever holds a name and has
    to recover the real game object. For a HAND cast that's free: hand holds
    CardDefs and game.CARD_DEFS[name] IS the object in hand, so type-identity
    and object-identity coincide. For a GRAVEYARD cast they don't -- the
    graveyard holds per-object CardInstances (see plans/object-identity-zone-
    model.md), and game.CARD_DEFS[name] is the interned, one-per-name rules
    definition, never identity-equal to any instance.

    Passing that CardDef onward is what forced every graveyard-cast resolve to
    re-derive the instance itself: six did it by name (via the old
    casting.remove_graveyard_card, or hand-rolled), and the seventh
    (flashback_deep_analysis) forgot and crashed a real pretrain run with
    `ValueError: list.remove(x): x not in list`. Resolving it HERE, once, is
    what lets every downstream removal/capture be a true identity operation.

    Picks the first same-named instance -- correct only when at most one copy
    exists. MTG 400.7 makes same-named graveyard cards interchangeable for "a
    card with this name leaves the graveyard" when there's truly no way to
    tell them apart, but real Magic still lets the PLAYER choose which copy
    when it's observable (a Rooftop Percher target locked on copy A while
    copy B is the one flashed back) -- that choice is handled one level up,
    by this function's sole caller, _with_chosen_copy, which opens a real
    pointer-addressed choose_cast_copy pending whenever 2+ copies exist and
    only calls into here for the no-choice-to-make case (0 or 1 copy)."""
    inst = next((c for c in state.graveyard if c.name == name), None)
    if inst is None:
        # Fail loudly with context, not a bare StopIteration -- same precedent
        # as game.effects.shared.discard_from_hand_to_graveyard's own guard.
        # Unreachable via a legal action (_flashback_legal/_actions_cast.
        # _graveyard_ability_legal both require a same-named graveyard card),
        # so this means a caller's own guarantee broke.
        raise RuntimeError(
            f"_graveyard_instance: no {name!r} in graveyard. "
            f"active_idx={getattr(state, 'active_idx', None)!r} "
            f"turn_number={getattr(state, 'turn_number', None)!r} "
            f"graveyard={[c.name for c in state.graveyard]!r}"
        )
    return inst


def _with_chosen_copy(state, name, proceed):
    """Run `proceed(state, inst)` on the graveyard copy of `name` being cast.

    With 2+ same-named copies this is a REAL agent choice (MTG 601.2a -- the
    object being cast is chosen at announcement), so it opens a pointer-only
    choose_cast_copy pending and continues from its on_complete; the choice is
    made BEFORE any cost is paid, which is both the faithful order and what
    keeps the single pending_resolution slot free for the payment that follows.
    With exactly one copy there is no choice to make, so it proceeds inline:
    this is not a simplification, just the absence of a decision (the harness
    would auto-resolve a one-option pending anyway; skipping it avoids a
    pointless token-set build + mask sweep on the common path).

    Also called from drl_env._actions_cast._graveyard_ability_execute (Bramble
    Wurm's own graveyard-activated ability) -- same identity-recovery need as
    a graveyard cast, just not a cast at all."""
    copies = [c for c in state.graveyard if c.name == name]
    if len(copies) <= 1:
        proceed(state, _graveyard_instance(state, name))
        return
    game.begin_choose_cast_copy(state, name, on_complete=proceed)


def _flashback_execute(name, resolve, cost=None):
    """resolve receives the graveyard CardInstance being cast (NOT the interned
    CardDef) -- see _graveyard_instance. Every flashback/escape resolve removes
    that exact object from the graveyard by identity. WHICH copy, when the
    graveyard holds several, is the agent's own choice (_with_chosen_copy)."""
    def execute(state):
        def _proceed(state, inst):
            if cost is None:
                game.on_cast_trigger(state, inst)  # item 11 -- see _actions_cast._cast_execute
                resolve(state, inst)
                return
            # Mana flashback cost: pay it AFTER the copy is chosen (601.2a
            # announce-then-pay), then fire the on-cast trigger and run the
            # resolve, which pays any further additional cost (life) and pushes
            # the effect.
            def _after_pay(state, inst=inst):
                game.on_cast_trigger(state, inst)
                resolve(state, inst)
            game.begin_pay_cost(state, cost, on_complete=_after_pay)

        _with_chosen_copy(state, name, _proceed)
    return execute


def _plot_legal(name, cost, speed):
    """Plot {cost}: pay it and exile this card from hand (no board
    presence yet) -- legal exactly like a normal cast, just against the
    plot cost instead of card_def.cast_cost. Real Magic: Plot's own
    reminder text is "any time you could cast this card" -- same speed as
    the card's normal cast, not a separate timing rule; the later free
    cast from exile (_cast_from_exile_legal) uses the same speed too."""
    def legal(state):
        if state.pending_resolution is not None:
            return False
        if not game.turn.speed_legal(state, speed):
            return False
        if not any(c.name == name for c in state.hand):
            return False
        return game.plan_payment(state, cost) is not None
    legal._pending_gate = _GATE_NO_PENDING
    return legal


def _plot_execute(name, cost, resolve):
    def execute(state):
        card_def = game.CARD_DEFS[name]
        # Plotting itself isn't casting the spell (it's exiled, not
        # resolved) -- no on_cast_trigger here; that fires from
        # _cast_from_exile_execute below, once it's actually cast.
        game.begin_pay_cost(state, cost, on_complete=lambda s: resolve(s, card_def))
    return execute


def _cast_from_exile_legal(name, extra_legal, speed):
    """Plot's second half: cast a previously-plotted copy, without paying
    its mana cost, on any turn after the one it was plotted on. speed:
    same value _plot_legal used -- see that function's own docstring.

    extra_legal: Plot only waives the MANA cost, not any other cost a
    card's normal "cast" spec gates on (e.g. Highway Robbery's own
    "discard a card" additional cost still needs a card in hand to
    discard) -- reuses the same cast_spec["extra_legal"] the normal cast
    path already checks, so a card needing both never looks payable when
    it secretly isn't. None (every existing Plot card so far) means no
    such gate, unaffected."""
    def legal(state):
        if state.pending_resolution is not None:
            return False
        if not game.turn.speed_legal(state, speed):
            return False
        has_plotted = any(
            c.name == name and stamp is not None and stamp < state.turn_number
            for c, stamp in state.exile
        )
        if not has_plotted:
            return False
        return extra_legal is None or extra_legal(state)
    legal._pending_gate = _GATE_NO_PENDING
    return legal


def _cast_from_exile_execute(name, resolve):
    def execute(state):
        card_def = game.CARD_DEFS[name]
        entry = next(
            e for e in state.exile
            if e[0].name == name and e[1] is not None and e[1] < state.turn_number
        )
        state.exile.remove(entry)
        game.on_cast_trigger(state, card_def)  # item 11 -- see _actions_cast._cast_execute
        # Plot's whole point is that the cost was already paid earlier
        # (when plotted) -- already "fully paid for" now, so push
        # immediately instead of resolving now (see _actions_cast._cast_execute's own
        # stack comment).
        game.push_to_stack(state, card_def, resolve, reserves_hand_card=False)
    return execute


def _omen_cast_legal(hand_name, cost, speed):
    """Sagu Wildling's Omen: real Scryfall reminder text is "(Also shuffle
    this card.)" attached to Roost Seek's own library search -- unlike
    real Adventure, an Omen card does NOT exile itself for a later free-
    zone cast; the resolved sorcery is shuffled directly into the LIBRARY
    (cast_roost_seek), and the real creature half only ever becomes
    castable again once the same physical card is redrawn into HAND, same
    as any ordinary card. So this is really just "the same hand card, a
    second cast option with its own distinct cost" -- checked against
    state.hand, not state.exile. hand_name is the SORCERY side's own
    registered name (the only one ever physically in a zone) -- the
    creature side is a distinct CardDef, never separately registered in
    game.CARD_DEFS (see the "omen" registry spec's own "card_def" key)."""
    def legal(state):
        if state.pending_resolution is not None:
            return False
        if not game.turn.speed_legal(state, speed):
            return False
        if not any(c.name == hand_name for c in state.hand):
            return False
        return game.plan_payment(state, cost) is not None
    legal._pending_gate = _GATE_NO_PENDING
    return legal


def _omen_cast_execute(creature_card_def, cost, resolve):
    """Same begin_pay_cost -> push_to_stack shape as a normal hand cast
    (_actions_cast._cast_execute), just for `creature_card_def` (the distinct
    creature CardDef) instead of game.CARD_DEFS[name]. reserves_hand_card
    defaults True here (unlike Plot/Flashback's own exile/graveyard-sourced
    pushes) -- the physical card genuinely IS still sitting in the caster's
    hand, unresolved, while this is paid for; _hand_count_available matches
    stack entries by NAME, so pushing creature_card_def (same display name
    as whatever's in hand) still correctly reserves it, blocking the
    sorcery-mode cast of the same physical copy in the meantime. `resolve`
    is responsible for removing the matching hand card itself (by NAME,
    not identity -- the object actually sitting in state.hand is the
    sorcery side's own CardDef, a different object from creature_card_def
    despite sharing a display name), same "resolve does its own zone
    removal" convention every other cast path here follows."""
    def execute(state):
        game.on_cast_trigger(state, creature_card_def)  # no-op for a CREATURE card_def (on_cast_trigger only fires for INSTANT/SORCERY) -- called anyway for the same hygiene every other cast path here has

        def _after_pay(s):
            # The physical hand card LEAVES hand at cast, like every other cast
            # (game.push_to_stack) -- never re-entering hand. It shares
            # creature_card_def's display name but is a DIFFERENT object (the
            # hand card is the sorcery/normal side), so push_to_stack's own
            # identity-based removal below misses it; remove it here by name.
            # That is what makes the OTHER mode uncastable while this copy is on
            # the stack, now that the card physically leaving hand is the sole
            # re-cast guard (_hand_count_available is a plain hand tally).
            hand_card = next((c for c in s.hand if c.name == creature_card_def.name), None)
            if hand_card is not None:
                s.hand.remove(hand_card)
            game.push_to_stack(s, creature_card_def, resolve)
        game.begin_pay_cost(state, cost, on_complete=_after_pay)
    return execute


__all__ = [
    '_alt_cast_legal',
    '_alt_cast_execute',
    '_flashback_legal',
    '_graveyard_instance',
    '_with_chosen_copy',
    '_flashback_execute',
    '_plot_legal',
    '_plot_execute',
    '_cast_from_exile_legal',
    '_cast_from_exile_execute',
    '_omen_cast_legal',
    '_omen_cast_execute',
]
