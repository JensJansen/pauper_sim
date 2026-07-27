"""Shared card-mechanic helpers reused verbatim by multiple color
catalogs -- no other logical home (each is a few lines, not big enough to
justify its own module, and none of them depend on any other effects
submodule). Zero registry/mana/resolution dependency -- pure state
manipulation."""

from ..cards import CardType


_COLOR_PIPS = ("W", "U", "B", "R", "G")


def fire_sacrifice_triggers(state, sacrificer_idx, sacrificed_card_def):
    """Fire every "whenever you sacrifice [another permanent / another
    Eldrazi]" trigger (Gixian Infiltrator, Writhing Chrysalis) on the
    SACRIFICING player's own battlefield, for a permanent that was just
    sacrificed. Called from every sacrifice path (sacrifice_to_graveyard,
    execute_sacrifice_option, the token sac abilities). The +1/+1 counter is
    applied inline rather than as a stacked triggered ability -- a deliberate,
    documented timing collapse: it has no target and nothing in this pool can
    respond to it, so on-stack vs inline is unobservable. Lazy registry import
    (shared.py is imported very early)."""
    from .. import registry
    for permanent in state.players[sacrificer_idx].battlefield:
        spec = registry.EFFECT_REGISTRY.get(permanent.card_def.effect_id, {}).get("on_sacrifice")
        if spec is not None:
            spec(state, permanent, sacrificed_card_def)


def affinity_reduction(state):
    """Affinity for artifacts: "this spell costs {1} less for each artifact
    you control." A cost_reduction spec (drl_env._effective_cast_cost) --
    counts the active player's own artifacts (is_artifact), including artifact
    lands / artifact creatures / Treasures."""
    return sum(1 for p in state.battlefield if is_artifact(p.card_def))


def graveyard_instant_sorcery_count(state):
    """"This spell costs {1} less for each instant and sorcery card in your
    graveyard" (Tolarian Terror, Cryptic Serpent) -- a cost_reduction spec."""
    return sum(1 for c in state.graveyard if c.card_type in (CardType.INSTANT, CardType.SORCERY))


def is_artifact(card_def):
    """Whether a card is an artifact -- a plain artifact (card_type ARTIFACT),
    an artifact land (card_type LAND + extra["artifact"]), or an artifact
    creature (card_type CREATURE + extra["artifact"], e.g. Myr Enforcer). The
    single predicate every "artifacts you control" reader uses: Goblin Tomb
    Raider's static boost, affinity, metalcraft, Krark-Clan/Makeshift's
    artifact-sacrifice."""
    return card_def.card_type == CardType.ARTIFACT or card_def.extra.get("artifact", False)


def card_colors(card_def):
    """The card's colors, from the colored pips in its mana cost. A Devoid
    card (extra["devoid"], e.g. Writhing Chrysalis) is colorless regardless
    of its cost's colored pips; a land / cost-less card (cast_cost None) is
    colorless too. Used by "nonblack"/etc. targeting restrictions (Snuff
    Out)."""
    if card_def.extra.get("devoid") or card_def.cast_cost is None:
        return set()
    return {c for c in card_def.cast_cost if c in _COLOR_PIPS}


def card_subtypes(card_def):
    """A card's subtypes (extra["subtypes"], e.g. ("Island","Swamp") or
    ("Elf",)), () if none declared. Only cards that need one for a rules
    check carry it -- basic Island/Forest (for Islandcycling / Gingerbread
    Cabin's Forest count), the U/B dual lands (Island subtype), the Elves
    (tribal counts). Not a full Scryfall subtype line; just the ones the
    engine actually reads."""
    return card_def.extra.get("subtypes", ())


def impulse_exile(state, n, until_next_turn=False):
    """Impulse: exile the top n cards of the active player's library into the
    impulse zone (state.impulse), "you may play them" until a deadline --
    end of THIS turn (until_next_turn=False: Experimental Synthesizer) or end
    of the player's NEXT turn (until_next_turn=True: Reckless Impulse /
    Clockwork Percussionist). The deadline is stored as an absolute
    turn_number: this turn's number, or that number + len(players) for "your
    next turn" (that many turn_numbers away in seat order -- correct for both
    1- and 2-player). game.turn.untap_step prunes entries once turn_number
    passes it. Returns the exiled cards."""
    deadline = state.turn_number + (len(state.players) if until_next_turn else 0)
    exiled = state.library[:n]
    del state.library[:n]
    for card_def in exiled:
        state.impulse.append((card_def, deadline))
    if exiled:
        state.log_event("impulse_exile", cards=[c.name for c in exiled], playable_until=deadline)
    return exiled


def mill(state, n, player_idx=None):
    """Put the top n cards of a player's library into their graveyard (real
    "mill"). player_idx=None mills the active player (Mental Note); an
    explicit index mills a chosen target player (Thought Scour's "target
    player mills two"). Fewer than n in library just mills whatever's there
    -- milling NEVER causes a deck-out (only drawing from an empty library
    does, per state.PlayerState.draw), so this reads/writes the player's
    zones directly and never needs the active_idx flip a cross-player DRAW
    would. Returns the milled cards."""
    idx = state.active_idx if player_idx is None else player_idx
    player = state.players[idx]
    milled = player.library[:n]
    del player.library[:n]
    player.graveyard.extend(milled)
    if milled:
        state.log_event("mill", count=len(milled), player_idx=idx, cards=[c.name for c in milled])
    return milled


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
    return any(p.card_type == CardType.CREATURE for p in state.battlefield)


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
