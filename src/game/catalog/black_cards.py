"""Black-identity card catalog: every card whose real mana cost is
mono-black (or, for lands with no cost, whose only mana output is black).
Every card's cost/type/oracle-text below is a direct Scryfall pull,
except creature power/toughness, which is a design choice, not Scryfall
data. Real Jagged Barrens/End the Festivities/Vampire's Kiss/Voldaren
Epicure/Alms of the Vein reference "each opponent"/"target opponent" --
all of these route through win_check.deal_damage_to_opponent, which hits
the opponent's real per-player life_total."""

from .. import resolution
from ..cards import CardDef, CardType, EffectId
from ..effects.casting import _log_target_fizzle, cast_permanent_from_hand, enters_battlefield
from ..effects.shared import discard_from_hand_to_graveyard
from ..effects.stack import push_to_stack
from ..effects.tokens import BLOOD_TOKEN_CARD_DEF, create_token
from ..effects.win_check import deal_damage_to_opponent

BLACK_CARD_CATALOG = {
    "Swamp": CardDef("Swamp", CardType.LAND, None, EffectId.SWAMP),
    "Bojuka Bog": CardDef("Bojuka Bog", CardType.LAND, None, EffectId.BOJUKA_BOG),
    "Balustrade Spy": CardDef(
        "Balustrade Spy", CardType.CREATURE, {"generic": 3, "B": 1}, EffectId.BALUSTRADE_SPY, power=2, toughness=2,
    ),
    "Lotleth Giant": CardDef(
        "Lotleth Giant", CardType.CREATURE, {"generic": 6, "B": 1}, EffectId.LOTLETH_GIANT, power=5, toughness=5,
    ),
    # ETB exiles a nonland from target opponent's hand (tracked); its LTB
    # returns that card when the Fiend leaves the battlefield -- the first
    # leaves-the-battlefield trigger in the engine (mesmeric_fiend_etb/_ltb).
    "Mesmeric Fiend": CardDef(
        "Mesmeric Fiend", CardType.CREATURE, {"generic": 1, "B": 1}, EffectId.MESMERIC_FIEND, power=1, toughness=1,
    ),
    "Dread Return": CardDef("Dread Return", CardType.SORCERY, {"generic": 2, "B": 2}, EffectId.DREAD_RETURN),
    "Kitchen Imp": CardDef(
        "Kitchen Imp", CardType.CREATURE, {"generic": 3, "B": 1}, EffectId.KITCHEN_IMP, power=2, toughness=2,
    ),
    "Vampire's Kiss": CardDef("Vampire's Kiss", CardType.SORCERY, {"generic": 1, "B": 1}, EffectId.VAMPIRES_KISS),
    "Alms of the Vein": CardDef("Alms of the Vein", CardType.SORCERY, {"generic": 2, "B": 1}, EffectId.ALMS_OF_THE_VEIN),
}


def mill_until_land(state):
    """Balustrade Spy's ETB: reveal from the top until a land card, milling
    everything revealed (including the land) to the graveyard. No model
    choice, so a plain loop, not a pending resolution. If the library
    empties before a land turns up, everything left mills and the library
    simply ends up empty -- this deck's own combo enabler. draw() (not
    this function) is what detects and flags actually running out, on
    whatever later draw attempts to pull from the now-empty library."""
    while state.library:
        card = state.library.pop(0)
        state.graveyard.append(card)
        if card.card_type == CardType.LAND:
            break


def lotleth_giant_etb(state):
    """Undergrowth ETB: 1 damage to the opponent per creature card in your
    graveyard."""
    creature_count = sum(1 for c in state.graveyard if c.card_type == CardType.CREATURE)
    deal_damage_to_opponent(state, creature_count)


def bojuka_bog_etb(state):
    """When Bojuka Bog enters, exile target player's graveyard. "Exile a
    graveyard" just empties it here -- exile is untracked, same convention
    as Relic of Progenitus' own graveyard-exile (game.catalog.colorless_
    cards). A REAL target-player choice (begin_choose_target_player), same
    as Relic's exile ability: "yourself" is always legal (true even alone in
    a 1-player game), the opponent becomes a second option once one exists,
    and the model picks explicitly.

    Runs as this ETB triggered ability RESOLVES off the stack (ETBs now go
    through the trigger queue -- casting.enters_battlefield). The target is
    chosen here at resolution rather than locked at stack-placement: "target
    player" is always legal (no way to make a player an illegal target), and
    nothing in this pool manipulates a graveyard at instant speed in response
    to the trigger, so the two orderings are outcome-identical -- no
    observable difference, same reasoning begin_choose_graveyard_card's own
    docstring already applies to Relic's cross-player pick."""
    def _on_player_chosen(state, idx):
        target = state.players[idx]
        exiled = [c.name for c in target.graveyard]
        target.graveyard.clear()
        state.log_event("graveyard_exiled", target_player_idx=idx, exiled=exiled)

    resolution.begin_choose_target_player(state, _on_player_chosen)


def mesmeric_fiend_etb(state, permanent):
    """When Mesmeric Fiend enters, target opponent reveals their hand and you
    choose a nonland card from it. Exile that card -- TRACKED, linked to THIS
    exact Fiend on its own flags; the matching "when this creature leaves the
    battlefield, return the exiled card to its owner's hand" is mesmeric_
    fiend_ltb below.

    Needs a real opponent (2-player) -- "target opponent" has no legal target
    in a 1-player config, so the ETB does nothing there. The nonland card is
    picked from the opponent's hand by reusing begin_choose_graveyard_card (a
    generic "choose one card by name from this list matching a predicate",
    despite its graveyard-flavored name) with the opponent's hand as the list.
    Runs as this ETB resolves off the stack (trigger queue), choosing at
    resolution -- the same ETB convention Pinnacle Kill-Ship / Masked Vandal
    already follow."""
    if len(state.players) < 2:
        return
    opponent = state.opponent
    opponent_idx = state.players.index(opponent)

    def _on_chosen(state, name):
        if name is None:
            return  # no nonland card in the opponent's hand
        card = next(c for c in opponent.hand if c.name == name)
        opponent.hand.remove(card)
        # Tracked exile, linked to this exact Fiend -- returned by its LTB.
        permanent.flags["mesmeric_exiled"] = (card, opponent_idx)
        state.log_event(
            "zone_move", card=card.name, from_zone="hand", to_zone="exile_mesmeric",
            owner_idx=opponent_idx, source=(permanent.card_def.name, permanent.slot),
        )

    resolution.begin_choose_graveyard_card(
        state, lambda c: c.card_type != CardType.LAND, _on_chosen, graveyard=opponent.hand,
    )


def mesmeric_fiend_ltb(state, permanent):
    """When Mesmeric Fiend leaves the battlefield, return the exiled card to
    its owner's hand. Reads the linkage mesmeric_fiend_etb stored on this exact
    permanent -- the object survives leaving the battlefield (state_based.
    _queue_leave_triggers / resolution.execute_sacrifice_option carry it here).
    A no-op if nothing was exiled (an empty/all-land opponent hand at ETB, or a
    1-player game where the ETB never fired)."""
    exiled = permanent.flags.pop("mesmeric_exiled", None)
    if exiled is None:
        return
    card, owner_idx = exiled
    state.players[owner_idx].hand.append(card)
    state.log_event(
        "zone_move", card=card.name, from_zone="exile_mesmeric", to_zone="hand", owner_idx=owner_idx,
        reason="mesmeric_leaves",
    )


def _dread_return_choose_and_push(state, card_def, to_graveyard, reserves_hand_card):
    """Choose the reanimation target -- a creature card in your graveyard --
    as Dread Return is put on the stack, lock it, and push the reanimation
    resolve. On resolution the chosen card returns from the graveyard to the
    battlefield; Dread Return FIZZLES if that card has left the graveyard by
    then (608.2b -- reachable via opponent graveyard hate, e.g. Relic of
    Progenitus exiling graveyards). Dread Return itself, a sorcery, is never a
    legal creature target. `to_graveyard`/`reserves_hand_card` say how the
    Dread Return card reaches the graveyard on resolution: from hand (hard
    cast) or a no-op (Flashback -- exiled, untracked).

    Duplicate-target note: graveyard cards are shared CardDef objects, so two
    copies of the same creature card are indistinguishable -- the fizzle
    check is "no copy of this card remains", not per-physical-copy identity
    (which the shared-CardDef graveyard cannot represent)."""
    def _on_chosen(state, name):
        captured = next((c for c in state.graveyard if c.name == name), None) if name is not None else None

        def _resolve(state, card_def):
            to_graveyard(state, card_def)
            if captured is None or captured not in state.graveyard:
                _log_target_fizzle(state, card_def, (name, "graveyard") if name is not None else None)
                return
            state.graveyard.remove(captured)
            enters_battlefield(state, captured, from_zone="graveyard")

        push_to_stack(state, card_def, _resolve, reserves_hand_card=reserves_hand_card)

    resolution.begin_choose_graveyard_card(state, lambda c: c.card_type == CardType.CREATURE, _on_chosen)


def cast_dread_return(state, card_def):
    """{2}{B}{B}: return target creature card from your graveyard to the
    battlefield. precast_choice -- the target is locked as the spell is cast
    (real Magic), the reanimation waits on the stack, and Dread Return itself
    goes to the graveyard when it resolves."""
    _dread_return_choose_and_push(state, card_def, to_graveyard=discard_from_hand_to_graveyard, reserves_hand_card=True)


def flashback_dread_return(state, card_def):
    """Flashback -- Sacrifice three creatures instead of {2}{B}{B}. Same
    reanimation; the target is chosen as the spell is put on the stack (after
    the sacrifice cost is paid), not at resolution. The newly sacrificed
    creatures are in the graveyard by then, so they're eligible targets (a
    real interaction). Dread Return is exiled afterward (untracked, per its
    own text), so its resolve makes no further zone move for itself."""
    state.graveyard.remove(card_def)  # leaves the graveyard the moment Flashback is chosen; exiled after (untracked)
    resolution.begin_sacrifice(
        state, lambda p: p.card_type == CardType.CREATURE, 3,
        on_complete=lambda s, ok: _dread_return_choose_and_push(
            s, card_def, to_graveyard=lambda st, cd: None, reserves_hand_card=False,
        ) if ok else None,
    )


def madness_kitchen_imp(state, card_def):
    """Kitchen Imp -- Flying, haste. Madness {B}. No ETB at all (real
    Oracle text has no triggered ability beyond Madness itself). Madness
    resolve for a creature: execute_madness_cast has already pulled the
    card out of exile, so this just needs the normal battlefield-entry
    path -- never touches hand, unlike a normal cast."""
    enters_battlefield(state, card_def)


def cast_vampires_kiss(state, card_def):
    """Target player loses 2 life and you gain 2 life. Create two Blood
    tokens. No Madness on this one (only Fiery Temper/Alms of the Vein
    have it)."""
    discard_from_hand_to_graveyard(state, card_def)
    deal_damage_to_opponent(state, 2)
    create_token(state, BLOOD_TOKEN_CARD_DEF)
    create_token(state, BLOOD_TOKEN_CARD_DEF)


def _alms_of_the_vein_damage(state):
    deal_damage_to_opponent(state, 3)


def cast_alms_of_the_vein(state, card_def):
    """Target opponent loses 3 life and you gain 3 life. Madness {B}."""
    discard_from_hand_to_graveyard(state, card_def)
    _alms_of_the_vein_damage(state)


def madness_alms_of_the_vein(state, card_def):
    state.graveyard.append(card_def)
    _alms_of_the_vein_damage(state)


BLACK_EFFECT_REGISTRY = {
    EffectId.SWAMP: {
        "mana": ("fixed", "B"),
    },
    EffectId.BOJUKA_BOG: {
        "mana": ("fixed", "B"),
        "enters_tapped": True,
        "etb_trigger": lambda state, permanent: bojuka_bog_etb(state),
        "pending_kinds": {"choose_target_player"},
    },
    EffectId.BALUSTRADE_SPY: {
        "cast": {"resolve": lambda state, card_def: cast_permanent_from_hand(state, card_def)},
        "etb_trigger": lambda state, permanent: mill_until_land(state),
    },
    EffectId.LOTLETH_GIANT: {
        "cast": {"resolve": lambda state, card_def: cast_permanent_from_hand(state, card_def)},
        "etb_trigger": lambda state, permanent: lotleth_giant_etb(state),
    },
    EffectId.MESMERIC_FIEND: {
        "cast": {"resolve": lambda state, card_def: cast_permanent_from_hand(state, card_def)},
        # ETB: exile a nonland from target opponent's hand (tracked, linked to
        # this Fiend); LTB: return it to its owner's hand.
        "etb_trigger": lambda state, permanent: mesmeric_fiend_etb(state, permanent),
        "ltb_trigger": lambda state, permanent: mesmeric_fiend_ltb(state, permanent),
        "pending_kinds": {"choose_graveyard_card"},
    },
    EffectId.DREAD_RETURN: {
        "cast": {
            "resolve": lambda state, card_def: cast_dread_return(state, card_def),
            "extra_legal": lambda state: any(c.card_type == CardType.CREATURE for c in state.graveyard),
            "precast_choice": True,  # target locked at cast (real Magic), reanimation waits on the stack
        },
        "flashback": {
            "legal": lambda state: sum(1 for p in state.battlefield if p.card_type == CardType.CREATURE) >= 3,
            "resolve": lambda state, card_def: flashback_dread_return(state, card_def),
        },
        "pending_kinds": {"choose_graveyard_card", "sacrifice"},
    },
    EffectId.KITCHEN_IMP: {
        # Real text: Flying, haste.
        "cast": {"resolve": lambda state, card_def: cast_permanent_from_hand(state, card_def)},
        "haste": True,
        "keywords": {"flying"},
        "madness": {"cost": {"B": 1}, "resolve": lambda state, card_def: madness_kitchen_imp(state, card_def)},
        # order_triggers: reachable the
        # instant 2+ Madness cards get discarded at once (Faithless
        # Looting's discard-2) -- both trigger simultaneously and need a
        # real placement-order choice.
        "pending_kinds": {"madness_decision", "order_triggers"},
    },
    EffectId.VAMPIRES_KISS: {
        "cast": {"resolve": lambda state, card_def: cast_vampires_kiss(state, card_def)},
    },
    EffectId.ALMS_OF_THE_VEIN: {
        "cast": {"resolve": lambda state, card_def: cast_alms_of_the_vein(state, card_def)},
        "madness": {"cost": {"B": 1}, "resolve": lambda state, card_def: madness_alms_of_the_vein(state, card_def)},
        "pending_kinds": {"madness_decision", "order_triggers"},  # see EffectId.KITCHEN_IMP's own comment
    },
}


if __name__ == "__main__":
    # ponytail self-check (run via `python -m game.catalog.black_cards` from
    # src/): Dread Return -- target locked at cast, effect on the stack, and
    # the 608.2b fizzle when the chosen creature card leaves the graveyard
    # before it resolves (reachable via opponent graveyard hate).
    import contextlib
    import io

    from ..effects.stack import resolve_top_of_stack
    from ..state import GameState, PlayerState

    dr = CardDef("Dread Return", CardType.SORCERY, {"generic": 2, "B": 2}, EffectId.DREAD_RETURN)
    grizzly = CardDef("Grizzly Bears", CardType.CREATURE, {"G": 1}, EffectId.FILLER, power=2, toughness=2)

    # (a) hard cast reanimates the chosen creature card
    state = GameState(on_the_play=True, players=[PlayerState(True), PlayerState(False)])
    state.hand = [dr]
    state.graveyard = [grizzly]
    cast_dread_return(state, dr)  # precast: begins the graveyard-target choice
    assert state.pending_resolution["kind"] == "choose_graveyard_card"
    resolution.execute_choose_graveyard_card_option(state, "Grizzly Bears")
    assert state.hand == [dr] and len(state.stack) == 1  # Dread Return still in hand, on the stack
    resolve_top_of_stack(state)
    assert dr in state.graveyard  # Dread Return resolved -> graveyard
    assert any(p.card_def is grizzly for p in state.battlefield) and grizzly not in state.graveyard  # reanimated

    # (b) fizzle: the chosen card leaves the graveyard before resolution
    state = GameState(on_the_play=True, players=[PlayerState(True), PlayerState(False)])
    state.hand = [dr]
    state.graveyard = [grizzly]
    cast_dread_return(state, dr)
    resolution.execute_choose_graveyard_card_option(state, "Grizzly Bears")
    state.graveyard.remove(grizzly)  # exiled by graveyard hate before Dread Return resolves
    _log = io.StringIO()
    with contextlib.redirect_stdout(_log):
        resolve_top_of_stack(state)
    assert "fizzle" in _log.getvalue().lower()
    assert dr in state.graveyard  # Dread Return still goes to the graveyard
    assert not any(p.card_def is grizzly for p in state.battlefield)  # nothing reanimated

    print("black_cards.py Dread Return target-at-cast + fizzle self-check: OK")

    # Bojuka Bog: enters tapped, and its ETB "exile target player's
    # graveyard" is a real target-player choice resolved off the stack (ETBs
    # go through the trigger queue now). 2-player so targeting the OPPONENT
    # empties THEIR graveyard, never the active player's own.
    from ..effects.triggers import promote_triggers_to_stack
    from ..resolution import execute_choose_target_player_option

    state = GameState(on_the_play=True, players=[PlayerState(True), PlayerState(False)])
    state.players[0].graveyard = [CardDef("Mine", CardType.CREATURE, None, EffectId.FILLER)]
    state.players[1].graveyard = [CardDef("Theirs", CardType.CREATURE, None, EffectId.FILLER)]
    bog = CardDef("Bojuka Bog", CardType.LAND, None, EffectId.BOJUKA_BOG)
    enters_battlefield(state, bog, from_zone="hand")
    bog_perm = next(p for p in state.battlefield if p.card_def.name == "Bojuka Bog")
    assert bog_perm.tapped  # enters tapped
    # ETB queued (faithful timing), not run inline -- promote + resolve opens the target choice.
    assert state.pending_resolution is None
    assert [e["type"] for e in state.trigger_queue] == ["etb"]
    promote_triggers_to_stack(state)
    resolve_top_of_stack(state)
    assert state.pending_resolution["kind"] == "choose_target_player"
    execute_choose_target_player_option(state, 1)  # target the opponent
    assert state.players[1].graveyard == []  # their graveyard exiled
    assert [c.name for c in state.players[0].graveyard] == ["Mine"]  # own graveyard untouched

    print("black_cards.py Bojuka Bog ETB (exile target player's graveyard) self-check: OK")

    # Mesmeric Fiend: ETB exiles a nonland from target opponent's hand
    # (tracked, linked to this Fiend); its leaves-the-battlefield trigger --
    # the engine's first -- returns that card when the Fiend leaves, whether
    # it DIES (state-based) or is SACRIFICED (resolution.execute_sacrifice_
    # option), the two ways a creature leaves in this pool.
    from ..effects.state_based import check_state_based_actions
    from ..resolution import (
        begin_sacrifice, choose_graveyard_card_options, execute_choose_graveyard_card_option, execute_sacrifice_option,
    )

    fiend_def = CardDef("Mesmeric Fiend", CardType.CREATURE, {"generic": 1, "B": 1}, EffectId.MESMERIC_FIEND, power=1, toughness=1)

    def _enter_fiend_and_exile():
        st = GameState(on_the_play=True, players=[PlayerState(True), PlayerState(False)])
        a_spell = CardDef("Their Spell", CardType.SORCERY, {"B": 1}, EffectId.FILLER)
        a_land = CardDef("Their Land", CardType.LAND, None, EffectId.SWAMP)
        st.players[1].hand = [a_spell, a_land]
        fiend = enters_battlefield(st, fiend_def, from_zone="hand")
        assert [e["type"] for e in st.trigger_queue] == ["etb"]
        promote_triggers_to_stack(st)
        resolve_top_of_stack(st)  # ETB resolves -> choose a nonland from the opponent's hand
        assert st.pending_resolution["kind"] == "choose_graveyard_card"
        assert choose_graveyard_card_options(st) == ["Their Spell"]  # the LAND is excluded (nonland only)
        execute_choose_graveyard_card_option(st, "Their Spell")
        assert [c.name for c in st.players[1].hand] == ["Their Land"]  # nonland exiled from their hand
        assert fiend.flags["mesmeric_exiled"][0] is a_spell  # tracked, linked to this Fiend
        return st, fiend, a_spell

    # (a) the Fiend DIES -> LTB returns the exiled card to its owner's hand.
    state, fiend, a_spell = _enter_fiend_and_exile()
    fiend.damage_marked = fiend_def.extra["toughness"]  # lethal
    check_state_based_actions(state)
    assert fiend not in state.players[0].battlefield  # dead
    assert [e["type"] for e in state.trigger_queue] == ["ltb"]  # LTB queued on leave
    promote_triggers_to_stack(state)
    resolve_top_of_stack(state)
    assert a_spell in state.players[1].hand  # returned to its OWNER's hand
    assert "mesmeric_exiled" not in fiend.flags  # linkage consumed

    # (b) the Fiend is SACRIFICED (e.g. to Dread Return's Flashback) -> same LTB.
    state, fiend, a_spell = _enter_fiend_and_exile()
    begin_sacrifice(state, lambda p: p.card_def.name == "Mesmeric Fiend", 1, on_complete=lambda s, ok: None)
    execute_sacrifice_option(state, "Mesmeric Fiend")
    assert fiend not in state.players[0].battlefield
    assert [e["type"] for e in state.trigger_queue] == ["ltb"]
    promote_triggers_to_stack(state)
    resolve_top_of_stack(state)
    assert a_spell in state.players[1].hand

    # (c) 1-player (no opponent to target): ETB does nothing, no exile.
    solo = GameState(on_the_play=True)
    solo_fiend = enters_battlefield(solo, fiend_def, from_zone="hand")
    promote_triggers_to_stack(solo)
    resolve_top_of_stack(solo)
    assert solo.pending_resolution is None and "mesmeric_exiled" not in solo_fiend.flags

    print("black_cards.py Mesmeric Fiend ETB/LTB self-check: OK")
