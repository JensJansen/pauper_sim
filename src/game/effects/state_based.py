"""State-based actions (creature death) and end-of-turn cleanup. Depends on
stats.py for effective toughness and on registry (lazily) to tell a real card
from a token and to read an orphaned Aura's return-to-hand flag."""

from . import stats
from .. import registry, resolution
from ..cards import CardType

HAND_SIZE_LIMIT = 7  # real Magic's own rule -- not a per-config tunable, no card in this pool ever modifies it


def departing_card_def(permanent):
    """The CardDef that actually goes to another zone when `permanent` leaves the
    battlefield. For a transformed double-faced permanent (Insectile Aberration)
    that's its FRONT face (Delver of Secrets) -- a DFC is only its back face while
    on the battlefield; in every other zone it's the front face (real Magic 712.4a).
    This also keeps the is-token check honest: the back face's name isn't in
    registry.CARD_DEFS, so without reverting here a dying Insectile Aberration would
    be misread as a token and deleted instead of putting Delver in the graveyard."""
    return permanent.flags.get("front_card_def", permanent.card_def)


def check_state_based_actions(state):
    """Creature-death check: every creature on either battlefield with lethal
    marked damage (>= effective toughness), or any damage from a deathtouch
    source, dies -- to the graveyard, its Aura(s) orphaned. Collects all dead
    FIRST, then removes them (real Magic's simultaneous SBA semantics, not a
    one-at-a-time recheck). Scans the whole battlefield -- a strict, cheap
    superset of "just-damaged," since this runs before every priority round,
    not just once per combat."""
    candidates = [
        p for player in state.players for p in player.battlefield if p.card_type == CardType.CREATURE
    ]
    # flags["deathtouched"] (set by combat only on a real deathtouch hit)
    # implies damage was dealt (704.5h); toughness <= 0 is caught too (0 >= 0).
    dead = [
        p for p in candidates
        if p.damage_marked >= stats.permanent_toughness(state, p) or p.flags.get("deathtouched")
    ]
    for permanent in dead:
        _destroy_creature(state, permanent)


def _queue_leave_triggers(state, permanent, owner_idx):
    """Queue a leaves-the-battlefield triggered ability (Mesmeric Fiend's
    exiled-card return) for a permanent that JUST left the battlefield, so
    game.turn's priority round puts it on the stack (real Magic 603.3 -- an
    LTB trigger goes on the stack, doesn't take effect the instant it fires).
    `permanent` is already off the battlefield but still carries whatever the
    trigger needs on its flags (the linked exiled card). No-op unless the
    card's effect_id has an "ltb_trigger" spec.

    Appends to owner_idx's OWN trigger_queue (state.players[owner_idx], not
    the state.trigger_queue active-player proxy): state-based death checks
    scan BOTH battlefields every priority round regardless of whose turn it
    is (check_state_based_actions above), so the dying permanent's owner can
    be the NON-active player (their blocker died in combat, or a removal
    spell killed their creature, on the active player's own turn) -- writing
    through the proxy would silently misfile their trigger into the active
    player's queue, then hand the active player an order_triggers choice
    naming a card from the OPPONENT's deck (drl_env's own coverage guard
    asserts every name it offers a player belongs to that player's own deck,
    so a misfiled trigger here would surface as exactly that kind of
    cross-deck name leak).
    game.effects.triggers.promote_triggers_to_stack reads every player's own
    queue, each ordering only ITS OWN simultaneous triggers (603.3b
    APNAP), so this is the one and only place that needs the true owner
    threaded through instead of the proxy.

    Called from the two -- and, in this card pool, only -- ways a creature
    (the sole card type with an LTB trigger here) leaves: death (below) and
    being sacrificed (sacrifice_to_graveyard, further down this same module
    -- every sacrifice path, including resolution.begin_sacrifice/Highway
    Robbery's own discard-or-sacrifice, routes through it rather than each
    inlining its own copy of this owner-threading logic). No other removal
    path in this pool takes a creature off the battlefield (bounce is
    lands-only; every exile ability exiles its own non-creature source), so
    this is complete, not a simplification -- thread it through a new
    removal site if a future card ever makes one reachable."""
    spec = registry.EFFECT_REGISTRY.get(permanent.card_def.effect_id, {})
    if spec.get("ltb_trigger") is not None:
        state.players[owner_idx].trigger_queue.append(
            {"type": "ltb", "card_def": permanent.card_def, "permanent": permanent})


def _destroy_creature(state, permanent):
    """One creature's death: battlefield -> graveyard, orphaning its Aura(s).
    Finds the owning PlayerState by membership, NOT the active-player proxy --
    combat runs with active_idx on the ATTACKER, but the dying creature can be
    the DEFENDER's blocker, whose zones the proxy would get wrong (or raise on
    battlefield.remove).

    A TOKEN (name not in registry.CARD_DEFS) ceases to exist -- never added to
    a graveyard (real Magic; the observation encoding also keys graveyards by
    real decklist names, so a token there would corrupt the next obs).

    Three orphan outcomes, each a fixed per-card registry flag: a Bestowed
    permanent (Nyxborn Hydra, "becomes_creature_when_orphaned") STAYS and
    reverts to a creature (clear type_override; counters untouched) -- checked
    first, mutually exclusive with the zone moves; Rancor
    ("returns_to_hand_when_orphaned") returns to hand; every other Aura goes to
    its controller's graveyard (the default)."""
    owner = next(player for player in state.players if permanent in player.battlefield)
    owner_idx = state.players.index(owner)
    # Stashed so an ltb_trigger resolving LATER (after a priority window, by
    # which point state.active_idx may no longer be this permanent's owner --
    # see Nihil Spellbomb's dies-trigger) can still recover its true
    # controller instead of misreading whoever's turn it now is.
    permanent.flags["owner_idx"] = owner_idx
    owner.battlefield.remove(permanent)
    departing = departing_card_def(permanent)  # front face for a DFC leaving the battlefield
    is_token = departing.name not in registry.CARD_DEFS
    if not is_token:
        # 400.7 linked-ability tracking: stash the freshly minted graveyard
        # instance on the dying permanent's own flags, so an ltb_trigger that
        # needs to reference the EXACT card that died (Lembas: "shuffles it
        # into their library") can, instead of bridging by name.
        permanent.flags["graveyard_instance"] = state.move_card(departing, owner.graveyard)
    state.log_event(
        "state_based_death", permanent=(permanent.card_def.name, permanent.slot), owner_idx=owner_idx,
        to_zone=("ceases_to_exist" if is_token else "graveyard"),
    )
    _queue_leave_triggers(state, permanent, owner_idx)
    orphaned = [p for p in owner.battlefield if p.flags.get("enchanting") is permanent]
    for aura in orphaned:
        spec = registry.EFFECT_REGISTRY.get(aura.card_def.effect_id, {})
        if spec.get("becomes_creature_when_orphaned", False):
            aura.flags.pop("enchanting", None)
            aura.type_override = None
            state.log_event(
                "aura_orphaned", aura=(aura.card_def.name, aura.slot), target=(permanent.card_def.name, permanent.slot),
                outcome="stays_as_creature",
            )
            continue
        owner.battlefield.remove(aura)
        if spec.get("returns_to_hand_when_orphaned", False):
            owner.hand.append(aura.card_def)
            outcome = "hand"
        else:
            state.move_card(aura.card_def, owner.graveyard)
            outcome = "graveyard"
        state.log_event(
            "aura_orphaned", aura=(aura.card_def.name, aura.slot), target=(permanent.card_def.name, permanent.slot),
            outcome=outcome,
        )


def destroy_permanent(state, permanent):
    """A targeted "destroy" effect (Cast Down/Terminate/Snuff Out destroy a
    creature; Cleansing Wildfire destroys a land). Indestructible permanents
    (extra["indestructible"] -- the four Bridge lands) CAN'T be destroyed:
    the destroy simply does nothing to them (real Magic 701.7c), returning
    False. A creature routes through _destroy_creature (Aura-orphaning, LTB,
    token-ceases-to-exist); any other permanent type (a land) is removed to
    its owner's graveyard directly -- no Aura/LTB in this pool attaches to a
    non-creature. "Can't be regenerated" riders (Terminate/Snuff Out) are a
    no-op: regeneration isn't modeled (no card in this pool grants a regen
    shield), so there's never anything to prevent. Returns True iff actually
    destroyed."""
    if permanent.card_def.extra.get("indestructible"):
        state.log_event("destroy_failed_indestructible", permanent=(permanent.card_def.name, permanent.slot))
        return False
    if permanent.card_type == CardType.CREATURE:
        _destroy_creature(state, permanent)
        return True
    owner = next(player for player in state.players if permanent in player.battlefield)
    owner_idx = state.players.index(owner)
    owner.battlefield.remove(permanent)
    is_token = permanent.card_def.name not in registry.CARD_DEFS
    if not is_token:
        state.move_card(permanent.card_def, owner.graveyard)
    state.log_event(
        "destroy", permanent=(permanent.card_def.name, permanent.slot), owner_idx=owner_idx,
        to_zone=("ceases_to_exist" if is_token else "graveyard"),
    )
    if not is_token:
        _queue_leave_triggers(state, permanent, owner_idx)  # a "put into a graveyard from the battlefield" (dies) trigger, if any
    return True


def sacrifice_to_graveyard(state, permanent):
    """Sacrifice a permanent: battlefield -> its owner's graveyard (or cease,
    for a token), queuing any leaves-the-battlefield / "put into a graveyard
    from the battlefield" (dies) trigger. The single path every "Sacrifice
    this" ability and artifact-sacrifice cost routes through, so a dies
    trigger (Ichor Wellspring, Chromatic Star, Nihil Spellbomb, Lembas) fires
    no matter which effect did the sacrificing -- while battlefield->exile
    paths (Masked Vandal's exile) deliberately do NOT go through here, so they
    correctly don't fire a "put into a graveyard" trigger. Reuses the existing
    ltb_trigger mechanism: in this pool these artifacts only ever leave the
    battlefield by going to the graveyard, so ltb == dies-to-graveyard for
    them."""
    from .shared import fire_sacrifice_triggers

    owner = next(player for player in state.players if permanent in player.battlefield)
    owner_idx = state.players.index(owner)
    permanent.flags["owner_idx"] = owner_idx  # see _destroy_creature's own comment -- true controller for a later-resolving ltb_trigger
    owner.battlefield.remove(permanent)
    departing = departing_card_def(permanent)  # front face for a DFC leaving the battlefield
    is_token = departing.name not in registry.CARD_DEFS
    if not is_token:
        # 400.7 linked-ability tracking: see _destroy_creature's own comment --
        # stash the fresh graveyard instance so an ltb_trigger needing the
        # EXACT card that left (Lembas) can reference it, not bridge by name.
        permanent.flags["graveyard_instance"] = state.move_card(departing, owner.graveyard)
    state.log_event(
        "zone_move", permanent=(permanent.card_def.name, permanent.slot), from_zone="battlefield",
        to_zone=("ceases_to_exist" if is_token else "graveyard"), reason="sacrifice",
    )
    if not is_token:
        _queue_leave_triggers(state, permanent, owner_idx)
    fire_sacrifice_triggers(state, owner_idx, permanent.card_def)  # Gixian Infiltrator / Writhing Chrysalis


def cleanup_step(state):
    """game.turn.Phase.END: clears combat damage off EVERY permanent, both
    players (real Magic: damage clears at cleanup regardless of whose
    turn it is -- iterates state.players directly, not the active-player-
    proxied state.battlefield, which would only ever reach the active
    player's own side), then discards the ACTIVE player down to
    HAND_SIZE_LIMIT, real agency over which cards go via begin_discard --
    the same machinery every other discard effect already uses, not an
    automatic/arbitrary discard. Only the active player discards here,
    matching real Magic's own rule (this runs once per player's own turn
    -- the other player's hand, if any, is untouched until THEIR turn's
    own end); no-op if already at or under the limit (begin_discard's own
    n<=0 short-circuit handles that for free, no guard needed here).

    This is the only ceiling on hand size in an adversarial 2-player game."""
    damaged = [
        (p.card_def.name, p.slot) for player in state.players for p in player.battlefield if p.damage_marked > 0
    ]
    for player in state.players:
        for permanent in player.battlefield:
            permanent.damage_marked = 0
            # "until end of turn" effects wear off now (real Magic 514.2):
            # Agony Warp's -3/-0 / -0/-3, Toxin Analysis' granted deathtouch/
            # lifelink, and the deathtouched damage marker.
            permanent.temp_power = 0
            permanent.temp_toughness = 0
            permanent.temp_keywords = set()
            permanent.flags.pop("deathtouched", None)
    if damaged:
        state.log_event("cleanup_damage_cleared", permanents=damaged)
    n = max(0, len(state.hand) - HAND_SIZE_LIMIT)
    if n > 0:
        # One count per TURN this player over-drew and had to pitch the excess
        # (hoarding proxy for rl.rewards.deploy_reward's loss band) -- only the
        # hand-size cleanup discard, never any other discard effect.
        state.players[state.active_idx].cleanup_discard_turns += 1
    resolution.begin_discard(state, n, optional=False, on_complete=lambda s, _cards: None)
