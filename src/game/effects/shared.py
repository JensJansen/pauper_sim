"""Small card-mechanic helpers reused by multiple color catalogs. Pure
state manipulation -- no registry/mana/resolution dependency."""

from ..cards import CardType, is_artifact


def set_tapped(state, permanent, tapped, reason=None):
    """Choke point for any non-mana, non-turn-structure tapped-state change
    (an activation cost, or an effect tapping/untapping a target). Logs
    "tap_or_untap" with an explicit owner_idx (via stats.controller_idx) so
    replay reconstruction can disambiguate same-named permanents on
    different players' boards (slots are assigned per-player)."""
    from .stats import controller_idx
    permanent.tapped = tapped
    state.log_event(
        "tap_or_untap",
        permanent=(permanent.card_def.name, permanent.slot),
        owner_idx=controller_idx(state, permanent),
        now_tapped=tapped,
        reason=reason,
    )


def tap_for_cost(state, permanent, reason="activate"):
    """Pay a non-mana {T} cost. Thin wrapper over set_tapped."""
    set_tapped(state, permanent, True, reason=reason)


def fire_sacrifice_triggers(state, sacrificer_idx, sacrificed_card_def):
    """Queue every "whenever you sacrifice [a permanent]" trigger (Gixian
    Infiltrator, Writhing Chrysalis) on the sacrificing player's own
    battlefield. Written directly into that player's own trigger_queue
    (not the active-player proxy) since the sacrificer may not be the
    active player. Lazy registry import (shared.py loads very early)."""
    from .. import registry
    for permanent in state.players[sacrificer_idx].battlefield:
        spec = registry.EFFECT_REGISTRY.get(permanent.card_def.effect_id, {}).get("on_sacrifice")
        if spec is not None:
            state.players[sacrificer_idx].trigger_queue.append({
                "type": "on_sacrifice", "card_def": permanent.card_def,
                "permanent": permanent, "sacrificed_card_def": sacrificed_card_def,
            })


def affinity_reduction(state):
    """Affinity for artifacts cost_reduction spec: count of the active
    player's own artifacts (lands / creatures / Treasures included)."""
    return sum(1 for p in state.battlefield if is_artifact(p.card_def))


def graveyard_instant_sorcery_count(state):
    """"Costs {1} less for each instant/sorcery in your graveyard"
    cost_reduction spec (Tolarian Terror, Cryptic Serpent)."""
    return sum(1 for c in state.graveyard if c.card_type in (CardType.INSTANT, CardType.SORCERY))


def impulse_exile(state, n, until_next_turn=False):
    """Exile the top n cards of the active player's library into the impulse
    zone, playable until a deadline: end of this turn (until_next_turn=False)
    or end of the player's next turn (True). Deadline is stored as an
    absolute turn_number; game.turn.untap_step prunes expired entries.
    Returns the exiled cards."""
    deadline = state.turn_number + (len(state.players) if until_next_turn else 0)
    exiled = state.library[:n]
    del state.library[:n]
    for card_def in exiled:
        state.impulse.append((card_def, deadline))
    if exiled:
        state.log_event("impulse_exile", cards=[c.name for c in exiled], playable_until=deadline)
    return exiled


def mill(state, n, player_idx=None):
    """Put the top n cards of a player's library into their graveyard.
    player_idx=None mills the active player; an explicit index mills a
    chosen target player. Fewer than n in library just mills whatever's
    there -- mill never causes a deck-out. Returns the milled cards."""
    idx = state.active_idx if player_idx is None else player_idx
    player = state.players[idx]
    milled = player.library[:n]
    del player.library[:n]
    # move_card mints a fresh graveyard instance per card (zone change).
    milled = [state.move_card(cd, player.graveyard) for cd in milled]
    if milled:
        state.log_event("mill", count=len(milled), player_idx=idx, cards=[c.name for c in milled])
    return milled


def shuffle_library(state, player_idx=None):
    """Shuffle a player's library. The one choke point for shuffling."""
    idx = state.active_idx if player_idx is None else player_idx
    state.rng.shuffle(state.players[idx].library)


def find_and_remove_by_name(state, name):
    """Search state.library for the first card matching `name`, remove and
    return it (or None if absent). Does not shuffle."""
    for i, c in enumerate(state.library):
        if c.name == name:
            return state.library.pop(i)
    return None


def find_to_hand(state, name):
    """Shared tail of every "search library for X, put it into hand,
    shuffle" effect. name=None (a declined optional search) still shuffles,
    just finds nothing."""
    found = find_and_remove_by_name(state, name) if name is not None else None
    shuffle_library(state)
    if found:
        state.hand.append(found)
        state.log_event("zone_move", card=found.name, from_zone="library", to_zone="hand", reason="search")


def discard_from_hand_to_graveyard(state, card_def):
    """Send a card to its controller's graveyard. Two cases: the spell
    currently RESOLVING off the stack (state.resolving_card) already left
    hand at cast, so this just moves it to the graveyard without touching
    hand; any other card is a genuine discard-from-hand (a cost or discard
    effect) and is removed from hand and logged here. Not for cards that
    exile or resolve from elsewhere (Flashback/Plot/Madness skip this)."""
    if card_def is state.resolving_card:
        state.move_card(card_def, state.graveyard)
        return
    if card_def not in state.hand:
        raise RuntimeError(
            f"discard_from_hand_to_graveyard: {card_def.name!r} not in hand and not the resolving spell. "
            f"active_idx={getattr(state, 'active_idx', None)!r} "
            f"turn_player_idx={getattr(state, 'turn_player_idx', None)!r} "
            f"turn_number={getattr(state, 'turn_number', None)!r} "
            f"pending_resolution={state.pending_resolution!r} "
            f"hand={[c.name for c in state.hand]!r} "
            f"battlefield={[(p.card_def.name, p.slot, p.tapped) for p in state.battlefield]!r} "
            f"stack={[e['card_def'].name for e in state.stack]!r}"
        )
    state.hand.remove(card_def)
    state.move_card(card_def, state.graveyard)
    state.log_event("zone_move", card=card_def.name, from_zone="hand", to_zone="graveyard", reason="discard")


def any_creature_on_battlefield(state):
    """"Is there a legal target" gate for an "enchant creature you control"
    restriction (Cartouche of Solidarity). Caster's own battlefield only --
    see any_creature_on_either_battlefield for the unrestricted version."""
    return any(p.card_type == CardType.CREATURE for p in state.battlefield)


def any_creature_on_either_battlefield(state):
    """"Is there a legal target" gate for a plain "enchant/target creature"
    effect (Rancor, Ancestral Mask, Armadillo Cloak, Ethereal Armor, Bestow)
    that can target either side."""
    return any(p.card_type == CardType.CREATURE for player in state.players for p in player.battlefield)


def any_land_on_either_battlefield(state):
    """Land-side counterpart to any_creature_on_either_battlefield
    (Abundant Growth)."""
    return any(p.card_type == CardType.LAND for player in state.players for p in player.battlefield)
