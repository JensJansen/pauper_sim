"""State-based actions (creature death) and end-of-turn cleanup. Depends on
stats.py for effective toughness and on registry (lazily) to tell a real card
from a token and to read an orphaned Aura's return-to-hand flag."""

from . import stats
from .. import mana, registry, resolution
from ..cards import CardType

HAND_SIZE_LIMIT = 7  # real Magic's own rule; no card in this pool modifies it


def is_token(name):
    """True iff `name` isn't a real card in the decklist registry -- the
    only way to recognize a token, since tokens are never in registry.CARD_DEFS."""
    return name not in registry.CARD_DEFS


def departing_card_def(permanent):
    """The CardDef that goes to another zone when `permanent` leaves the
    battlefield. For a transformed DFC (Insectile Aberration) that's its
    front face (Delver of Secrets) -- a DFC is only its back face while on
    the battlefield (712.4a); also keeps the is-token check honest, since
    the back face's name isn't in registry.CARD_DEFS."""
    return permanent.flags.get("front_card_def", permanent.card_def)


def check_state_based_actions(state):
    """Creature-death check: every creature on either battlefield with
    lethal marked damage, or any deathtouch damage, dies to the graveyard,
    its Auras orphaned. Collects all dead first, then removes them
    (simultaneous SBA semantics). Also the choke point that logs
    "stats_changed" events for the replay viewer (see _log_stat_changes)."""
    candidates = [
        p for player in state.players for p in player.battlefield if p.card_type == CardType.CREATURE
    ]
    enchanting_by_target = stats.enchanting_by_target(state) if candidates else {}
    if state.event_log is not None:
        _log_stat_changes(state, candidates, enchanting_by_target)
    # flags["deathtouched"] (set by combat on a real deathtouch hit) implies
    # damage was dealt (704.5h); toughness <= 0 is caught too (0 >= 0).
    dead = [
        p for p in candidates
        if p.damage_marked >= stats.permanent_toughness(state, p, enchanting_auras=enchanting_by_target.get(id(p), ()))
        or p.flags.get("deathtouched")
    ]
    for permanent in dead:
        _destroy_creature(state, permanent)


def _log_stat_changes(state, candidates, enchanting_by_target):
    """Emits a "stats_changed" event for any creature whose effective
    power/toughness differs from what was last logged (cached on
    flags["_logged_pt"]), so the replay viewer shows current stats instead
    of the printed base logged at zone-entry. Caller gates this on
    state.event_log being on, to skip the cost when nothing reads it."""
    for p in candidates:
        auras = enchanting_by_target.get(id(p), ())
        pt = (stats.permanent_power(state, p, enchanting_auras=auras),
              stats.permanent_toughness(state, p, enchanting_auras=auras))
        if p.flags.get("_logged_pt") != pt:
            p.flags["_logged_pt"] = pt
            state.log_event("stats_changed", permanent=(p.card_def.name, p.slot), power=pt[0], toughness=pt[1])


def _queue_leave_triggers(state, permanent, owner_idx):
    """Queue a leaves-the-battlefield triggered ability (Mesmeric Fiend's
    exiled-card return) for a permanent that just left the battlefield,
    for game.turn's priority round to put on the stack. No-op unless the
    card's effect_id has an "ltb_trigger" spec.

    Appends to owner_idx's own trigger_queue directly, not the
    state.trigger_queue active-player proxy: state-based death checks scan
    both battlefields regardless of whose turn it is, so the dying
    permanent's owner can be the non-active player -- writing through the
    proxy would misfile the trigger into the active player's own queue.

    Called from the two ways a creature leaves: death (below) and being
    sacrificed (sacrifice_to_graveyard, further down this module)."""
    spec = registry.EFFECT_REGISTRY.get(permanent.card_def.effect_id, {})
    if spec.get("ltb_trigger") is not None:
        state.players[owner_idx].trigger_queue.append(
            {"type": "ltb", "card_def": permanent.card_def, "permanent": permanent})


def _destroy_creature(state, permanent):
    """One creature's death: battlefield -> graveyard, orphaning its Auras.
    Finds the owning PlayerState by membership, not the active-player proxy
    (combat can run with active_idx on the attacker while the dying creature
    is the defender's blocker).

    A token ceases to exist -- never added to a graveyard.

    Three orphan outcomes per registry flag: a Bestowed permanent
    ("becomes_creature_when_orphaned") stays and reverts to a creature;
    Rancor ("returns_to_hand_when_orphaned") returns to hand; every other
    Aura goes to its controller's graveyard (the default)."""
    owner_idx = stats.controller_idx(state, permanent)
    assert owner_idx is not None, "_destroy_creature: permanent not found on any battlefield"
    owner = state.players[owner_idx]
    # Stashed so an ltb_trigger resolving later can recover the true
    # controller instead of misreading whoever's turn it now is.
    permanent.flags["owner_idx"] = owner_idx
    owner.battlefield.remove(permanent)
    from .combat import remove_from_combat  # local: combat imports this module
    remove_from_combat(state, permanent)  # 506.4
    departing = departing_card_def(permanent)  # front face for a DFC leaving the battlefield
    departed_is_token = is_token(departing.name)
    if not departed_is_token:
        # 400.7 linked-ability tracking: stash the graveyard instance so an
        # ltb_trigger needing the exact card that died (Lembas) can reference
        # it instead of bridging by name.
        permanent.flags["graveyard_instance"] = state.move_card(departing, owner.graveyard)
    state.log_event(
        "state_based_death", permanent=(permanent.card_def.name, permanent.slot), owner_idx=owner_idx,
        to_zone=("ceases_to_exist" if departed_is_token else "graveyard"),
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
    """A targeted "destroy" effect (Cast Down/Terminate/Snuff Out a
    creature; Cleansing Wildfire a land). Indestructible permanents
    (extra["indestructible"]) can't be destroyed (701.7c), returning False.
    A creature routes through _destroy_creature; any other permanent type
    goes to its owner's graveyard directly. "Can't be regenerated" riders
    are a no-op (regeneration isn't modeled). Returns True iff destroyed."""
    if permanent.card_def.extra.get("indestructible"):
        state.log_event("destroy_failed_indestructible", permanent=(permanent.card_def.name, permanent.slot))
        return False
    if permanent.card_type == CardType.CREATURE:
        _destroy_creature(state, permanent)
        return True
    owner_idx = stats.controller_idx(state, permanent)
    assert owner_idx is not None, "destroy_permanent: permanent not found on any battlefield"
    owner = state.players[owner_idx]
    owner.battlefield.remove(permanent)
    permanent_is_token = is_token(permanent.card_def.name)
    if not permanent_is_token:
        state.move_card(permanent.card_def, owner.graveyard)
    state.log_event(
        "destroy", permanent=(permanent.card_def.name, permanent.slot), owner_idx=owner_idx,
        to_zone=("ceases_to_exist" if permanent_is_token else "graveyard"),
    )
    if not permanent_is_token:
        _queue_leave_triggers(state, permanent, owner_idx)  # a "dies" trigger, if any
    return True


def sacrifice_to_graveyard(state, permanent):
    """Sacrifice a permanent: battlefield -> its owner's graveyard (or
    cease, for a token), queuing any dies trigger. The single path every
    "Sacrifice this" ability and artifact-sacrifice cost routes through, so
    a dies trigger fires no matter which effect did the sacrificing --
    battlefield->exile paths (Masked Vandal) deliberately don't go through
    here.

    Also discounts mana.mana_pool_single_pip for whatever this permanent
    could have produced (mana.discount_departing_source): a mana source
    leaving the battlefield can never have "wastefully" tapped for mana."""
    from .shared import fire_sacrifice_triggers

    owner_idx = stats.controller_idx(state, permanent)
    assert owner_idx is not None, "sacrifice_to_graveyard: permanent not found on any battlefield"
    owner = state.players[owner_idx]
    mana.discount_departing_source(state, permanent, owner_idx)
    permanent.flags["owner_idx"] = owner_idx  # true controller for a later-resolving ltb_trigger
    owner.battlefield.remove(permanent)
    from .combat import remove_from_combat  # local: combat imports this module
    remove_from_combat(state, permanent)  # 506.4
    departing = departing_card_def(permanent)  # front face for a DFC leaving the battlefield
    departed_is_token = is_token(departing.name)
    if not departed_is_token:
        permanent.flags["graveyard_instance"] = state.move_card(departing, owner.graveyard)
    state.log_event(
        "zone_move", permanent=(permanent.card_def.name, permanent.slot), from_zone="battlefield",
        to_zone=("ceases_to_exist" if departed_is_token else "graveyard"), reason="sacrifice",
    )
    if not departed_is_token:
        _queue_leave_triggers(state, permanent, owner_idx)
    fire_sacrifice_triggers(state, owner_idx, permanent.card_def)  # Gixian Infiltrator / Writhing Chrysalis


def cleanup_step(state):
    """game.turn.Phase.END: clears combat damage off every permanent, both
    players, then discards the active player down to HAND_SIZE_LIMIT via
    begin_discard (real agency over which cards go, same machinery every
    other discard effect uses). No-op if already at or under the limit.

    Also runs check_state_based_actions right after the reset, so any
    until-end-of-turn pump/debuff wearing off logs its own "stats_changed"
    event immediately instead of only at the next priority round. Safe to
    call here: damage_marked is already zeroed above, so this can only log,
    never destroy, at this point in cleanup."""
    # Snapshot taken before the clearing loop below; safe since nothing in
    # that loop re-damages a permanent.
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
    check_state_based_actions(state)  # logs stats_changed for any pump/debuff that just wore off
    n = max(0, len(state.hand) - HAND_SIZE_LIMIT)
    if n > 0:
        # Hoarding proxy for rl.rewards.deploy_reward's loss band.
        state.players[state.active_idx].cleanup_discard_turns += 1
    resolution.begin_discard(state, n, optional=False, on_complete=lambda s, _cards: None)
