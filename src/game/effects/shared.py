"""Shared card-mechanic helpers reused verbatim by multiple color
catalogs -- no other logical home (each is a few lines, not big enough to
justify its own module, and none of them depend on any other effects
submodule). Zero registry/mana/resolution dependency -- pure state
manipulation."""

from ..cards import CardType, is_artifact


def fire_sacrifice_triggers(state, sacrificer_idx, sacrificed_card_def):
    """Queue every "whenever you sacrifice [another permanent / another
    Eldrazi]" trigger (Gixian Infiltrator, Writhing Chrysalis) on the
    SACRIFICING player's own battlefield, for a permanent that was just
    sacrificed. Called from every sacrifice path: most route through
    sacrifice_to_graveyard (itself the one path resolution.begin_sacrifice/
    Highway Robbery's own discard-or-sacrifice, the token sac abilities, and
    Lotus Petal/Treasure's own consumed-on-tap sacrifice all go through), the
    rest (e.g. Expedition Map, Candy Trail) still call this directly
    alongside their own hand-rolled zone-move. A real triggered
    ability: queued onto the sacrificer's own trigger_queue (written directly,
    not through the state.trigger_queue active-player proxy, in case the
    sacrificer isn't the active player -- same reasoning state_based.
    _queue_leave_triggers already documents) and promoted onto the stack at
    the next priority window (effects.triggers._trigger_resolve's own
    "on_sacrifice" branch runs the registry hook then). Lazy registry import
    (shared.py is imported very early)."""
    from .. import registry
    for permanent in state.players[sacrificer_idx].battlefield:
        spec = registry.EFFECT_REGISTRY.get(permanent.card_def.effect_id, {}).get("on_sacrifice")
        if spec is not None:
            state.players[sacrificer_idx].trigger_queue.append({
                "type": "on_sacrifice", "card_def": permanent.card_def,
                "permanent": permanent, "sacrificed_card_def": sacrificed_card_def,
            })


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
    # Each milled card CHANGES ZONES (library -> graveyard) so it enters the
    # graveyard as a fresh instance (move_card); return those instances, so a
    # caller acting on "the milled cards" acts on the objects now in the
    # graveyard, not the discarded library CardDefs.
    milled = [state.move_card(cd, player.graveyard) for cd in milled]
    if milled:
        state.log_event("mill", count=len(milled), player_idx=idx, cards=[c.name for c in milled])
    return milled


def shuffle_library(state, player_idx=None):
    """Shuffle a player's library. THE one place a library gets shuffled.

    Kept as a single choke point even though the shuffle itself is one line:
    nine call sites used to invoke rng.shuffle directly, and routing them all
    through here is what let a cross-cutting concern be added in one place. It
    carried exactly one such concern -- clearing PlayerState.known_top, since
    shuffling destroys any knowledge of library order -- which went away with
    known_top itself (2026-08-17, when the recurrent policy replaced remembered
    facts with remembered observations)."""
    idx = state.active_idx if player_idx is None else player_idx
    state.rng.shuffle(state.players[idx].library)


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
    shuffle_library(state)
    if found:
        state.hand.append(found)
        # Log the library->hand move (real Magic: the card physically enters hand).
        # Every "search your library, put it into hand" effect routes through here
        # (forestcycle/Islandcycling, Ash Barrens, Land Grant, Expedition Map, ...);
        # without this the fetched card would enter hand with no logged event, so
        # the replay wouldn't track it -- e.g. a later Brainstorm put-back naming
        # it couldn't find it. reason="search" distinguishes it from a normal draw.
        state.log_event("zone_move", card=found.name, from_zone="library", to_zone="hand", reason="search")


def discard_from_hand_to_graveyard(state, card_def):
    """Send a card to its controller's graveyard. Two distinct cases:

    - The spell currently RESOLVING off the stack (state.resolving_card): it
      left hand at cast (push_to_stack) and is now resolving, so it just lands
      in the graveyard -- its hand was never touched here and MUST NOT be (a
      same-named copy still in hand is a different physical card). This is the
      "normally-resolved spell -> graveyard" step at the start of nearly every
      cast_* resolve.
    - Any OTHER card: a genuine discard-FROM-hand (a cost, a discard effect),
      which must be physically in hand and is moved from there to the
      graveyard -- logged here (from_zone="hand"), unlike the resolving-spell
      case above (that departure was already logged at cast/push_to_stack;
      logging it again here would double it). This is the only path a
      Cycling-family cost (Islandcycling/Cycling/Forestcycle -- none of
      which ever touch the stack) takes out of hand, so without this log
      call the replay/event log would silently lose the card: no zone_move
      ever recorded its exit, and it would still read as "in hand" forever.

    Not for cards that instead exile or resolve from somewhere other than hand
    (Flashback/Plot/Madness's own resolve paths already skip this)."""
    if card_def is state.resolving_card:
        state.move_card(card_def, state.graveyard)  # the resolving spell: off hand since cast, stack -> graveyard (new object)
        return
    if card_def not in state.hand:
        # A non-resolving card must be in hand. If it isn't, a caller's own
        # guarantee broke -- fail loudly with the context needed to find which,
        # rather than a bare, contextless ValueError ("fail loudly, not
        # silently", same precedent as drl_env._substitute_and_resolve).
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
    """Shared "is there a legal Aura target at all" gate for an "enchant
    creature YOU CONTROL" restriction (real Magic 601.2c: a mandatory
    target needs at least one legal choice to be castable) -- only
    Cartouche of Solidarity's own extra_legal reduces to this (its real
    text is "Enchant creature you control", unlike every other Aura in this
    pool). Checks state.battlefield (the active/casting player's own zone
    only) -- see any_creature_on_either_battlefield for the "any creature,
    either side" version every unrestricted "Enchant creature" Aura needs."""
    return any(p.card_type == CardType.CREATURE for p in state.battlefield)


def any_creature_on_either_battlefield(state):
    """Shared "is there a legal Aura/targeted-effect target at all" gate for
    an "enchant/target creature" effect that can target EITHER side --
    Rancor/Ancestral Mask/Armadillo Cloak/Ethereal Armor/Nyxborn Hydra's
    Bestow (all plain "Enchant creature", no "you control" restriction --
    verified via Scryfall) all reduce to exactly this. Checking only the
    caster's OWN battlefield (any_creature_on_battlefield) would wrongly
    report "no legal target" whenever the caster controls no creatures but
    the opponent does -- a real, if rarely useful, legal target."""
    return any(p.card_type == CardType.CREATURE for player in state.players for p in player.battlefield)


def any_land_on_either_battlefield(state):
    """Land-side counterpart to any_creature_on_either_battlefield: the
    "is there a legal Aura/targeted-effect target at all" gate for an
    "enchant/target land" effect that can target EITHER side (Abundant
    Growth's real text is "Enchant land", no "you control" restriction).
    Checking only the caster's OWN battlefield would wrongly report "no
    legal target" whenever the caster is land-screwed but the opponent
    controls a land -- a real, if rarely useful, legal target (601.2c)."""
    return any(p.card_type == CardType.LAND for player in state.players for p in player.battlefield)
