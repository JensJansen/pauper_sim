"""Shared card-mechanic helpers reused verbatim by multiple color
catalogs -- no other logical home (each is a few lines, not big enough to
justify its own module, and none of them depend on any other effects
submodule). Zero registry/mana/resolution dependency -- pure state
manipulation."""

from ..cards import CardType


def find_and_remove_by_name(state, name):
    """Search state.library for the first card matching `name`, remove and
    return it (or None if absent). Does not shuffle -- callers shuffle per
    their own card's rules."""
    for i, c in enumerate(state.library):
        if c.name == name:
            return state.library.pop(i)
    return None


def find_to_hand(state, name):
    """Shared tail of every "search library for X, put it into hand,
    shuffle" effect (Generous Ent's forestcycle, Roost Seek, Gatecreeper
    Vine, Land Grant, Expedition Map, Ash Barrens). name=None (a declined
    optional search) still shuffles -- real-rules consequence of having
    searched/revealed the library at all, matching every one of these
    cards' own precedent -- just finds nothing."""
    found = find_and_remove_by_name(state, name) if name is not None else None
    state.rng.shuffle(state.library)
    if found:
        state.hand.append(found)


def discard_from_hand_to_graveyard(state, card_def):
    """Shared opening of nearly every cast_* function: leave hand, land in
    the graveyard as a normally-resolved spell. Not for cards that instead
    exile, get countered/fizzle, or resolve from somewhere other than
    hand (Flashback/Plot/Madness's own resolve paths already skip this)."""
    if card_def not in state.hand:
        # Should be unreachable -- every caller's own legality check
        # guarantees this card is still in hand by the time its resolve
        # runs. Fail loudly with the context needed to find which caller's
        # guarantee actually broke, rather than a bare, contextless
        # ValueError (same "fail loudly, not silently" precedent as
        # drl_env._substitute_and_resolve's own empty-mask check).
        raise RuntimeError(
            f"discard_from_hand_to_graveyard: {card_def.name!r} not in hand. "
            f"active_idx={getattr(state, 'active_idx', None)!r} "
            f"turn_player_idx={getattr(state, 'turn_player_idx', None)!r} "
            f"turn_number={getattr(state, 'turn_number', None)!r} "
            f"pending_resolution={state.pending_resolution!r} "
            f"hand={[c.name for c in state.hand]!r} "
            f"battlefield={[(p.card_def.name, p.slot, p.tapped) for p in state.battlefield]!r} "
            f"stack={[e['card_def'].name for e in state.stack]!r}"
        )
    state.hand.remove(card_def)
    state.graveyard.append(card_def)


def any_creature_on_battlefield(state):
    """Shared "is there a legal Aura/targeted-effect target at all" gate --
    Rancor/Ancestral Mask/Armadillo Cloak/Cartouche of Solidarity/Ethereal
    Armor's own extra_legal all reduce to exactly this."""
    return any(p.card_def.card_type == CardType.CREATURE for p in state.battlefield)


if __name__ == "__main__":
    # ponytail self-check: run via `python -m game.effects.shared` from src/.
    from ..cards import CardDef, CardType, EffectId
    from ..state import GameState

    state = GameState(on_the_play=True)
    state.library = [CardDef("Forest", CardType.LAND, None, EffectId.FOREST), CardDef("Mountain", CardType.LAND, None, EffectId.MOUNTAIN)]
    assert not any_creature_on_battlefield(state)

    find_to_hand(state, "Forest")
    assert [c.name for c in state.hand] == ["Forest"]
    assert [c.name for c in state.library] == ["Mountain"]

    find_to_hand(state, None)  # declined search still shuffles, finds nothing
    assert [c.name for c in state.hand] == ["Forest"]

    forest = state.hand[0]
    state.hand = [forest]
    discard_from_hand_to_graveyard(state, forest)
    assert state.hand == [] and state.graveyard == [forest]

    try:
        discard_from_hand_to_graveyard(state, forest)
        raise AssertionError("expected RuntimeError for a card no longer in hand")
    except RuntimeError:
        pass

    print("shared.py self-check: OK")
