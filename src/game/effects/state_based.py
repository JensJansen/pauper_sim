"""State-based actions (creature death) and end-of-turn cleanup. Depends on
stats.py for effective toughness and on registry (lazily) to tell a real card
from a token and to read an orphaned Aura's return-to-hand flag."""

from . import stats
from .. import registry, resolution
from ..cards import CardType

HAND_SIZE_LIMIT = 7  # real Magic's own rule -- not a per-config tunable, no card in this pool ever modifies it


def check_state_based_actions(state):
    """Creature-death check: every creature on either battlefield with lethal
    marked damage (>= effective toughness), or any damage from a deathtouch
    source, dies -- to the graveyard, its Aura(s) orphaned. Collects all dead
    FIRST, then removes them (real Magic's simultaneous SBA semantics, not a
    one-at-a-time recheck). Scans the whole battlefield -- a strict, cheap
    superset of "just-damaged," needed now that this runs before every priority
    round, not just once per combat."""
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


def _queue_leave_triggers(state, permanent):
    """Queue a leaves-the-battlefield triggered ability (Mesmeric Fiend's
    exiled-card return) for a permanent that JUST left the battlefield, so
    game.turn's priority round puts it on the stack (real Magic 603.3 -- an
    LTB trigger goes on the stack, doesn't take effect the instant it fires).
    `permanent` is already off the battlefield but still carries whatever the
    trigger needs on its flags (the linked exiled card). No-op unless the
    card's effect_id has an "ltb_trigger" spec.

    Called from the two -- and, in this card pool, only -- ways a creature
    (the sole card type with an LTB trigger here) leaves: death (below) and
    being sacrificed (resolution.execute_sacrifice_option inlines the identical
    three lines, since it can't import this module without a cycle). No other
    removal path in this pool takes a creature off the battlefield (bounce is
    lands-only; every sacrifice/exile ability sacrifices its own non-creature
    source), so this is complete, not a simplification -- thread it through a
    new removal site if a future card ever makes one reachable."""
    spec = registry.EFFECT_REGISTRY.get(permanent.card_def.effect_id, {})
    if spec.get("ltb_trigger") is not None:
        state.trigger_queue.append({"type": "ltb", "card_def": permanent.card_def, "permanent": permanent})


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
    owner.battlefield.remove(permanent)
    is_token = permanent.card_def.name not in registry.CARD_DEFS
    if not is_token:
        owner.graveyard.append(permanent.card_def)
    state.log_event(
        "state_based_death", permanent=(permanent.card_def.name, permanent.slot), owner_idx=owner_idx,
        to_zone=("ceases_to_exist" if is_token else "graveyard"),
    )
    _queue_leave_triggers(state, permanent)
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
            owner.graveyard.append(aura.card_def)
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
        owner.graveyard.append(permanent.card_def)
    state.log_event(
        "destroy", permanent=(permanent.card_def.name, permanent.slot), owner_idx=owner_idx,
        to_zone=("ceases_to_exist" if is_token else "graveyard"),
    )
    if not is_token:
        _queue_leave_triggers(state, permanent)  # a "put into a graveyard from the battlefield" (dies) trigger, if any
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
    owner.battlefield.remove(permanent)
    is_token = permanent.card_def.name not in registry.CARD_DEFS
    if not is_token:
        owner.graveyard.append(permanent.card_def)
    state.log_event(
        "zone_move", permanent=(permanent.card_def.name, permanent.slot), from_zone="battlefield",
        to_zone=("ceases_to_exist" if is_token else "graveyard"), reason="sacrifice",
    )
    if not is_token:
        _queue_leave_triggers(state, permanent)
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

    Newly relevant now that 2-player games run uncapped
    -- without this there was no
    ceiling on hand size in an adversarial game at all."""
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


if __name__ == "__main__":
    # ponytail self-check: run via `python -m game.effects.state_based`
    # from src/. Aura-orphaning and
    # token-death -- the two scenarios that are specific to THIS module
    # (check_state_based_actions/_destroy_creature), as opposed to the
    # combat+SBA handoff exercised together in effects/integration_check.py.
    from ..cards import CardDef, EffectId
    from ..state import GameState, Permanent, PlayerState

    # Aura-orphaning: Rancor returns to its controller's hand
    # (returns_to_hand_when_orphaned, green_cards.py); every other Aura
    # (Ancestral Mask here) to the graveyard, real Magic's default.
    rancor_def = CardDef("Rancor", CardType.ENCHANTMENT, {"G": 1}, EffectId.RANCOR)
    ancestral_mask_def = CardDef("Ancestral Mask", CardType.ENCHANTMENT, {"generic": 2, "G": 1}, EffectId.ANCESTRAL_MASK)
    _card_defs_backup = dict(registry.CARD_DEFS)
    try:
        state = GameState(on_the_play=True, players=[PlayerState(True), PlayerState(False)])
        attacker_with_rancor = Permanent(CardDef("Rancor'd Attacker", CardType.CREATURE, None, EffectId.FILLER, power=1, toughness=1))
        blocker_with_mask = Permanent(CardDef("Masked Blocker", CardType.CREATURE, None, EffectId.FILLER, power=1, toughness=1))
        registry.CARD_DEFS["Rancor'd Attacker"] = attacker_with_rancor.card_def
        registry.CARD_DEFS["Masked Blocker"] = blocker_with_mask.card_def
        rancor_permanent = Permanent(rancor_def)
        rancor_permanent.flags["enchanting"] = attacker_with_rancor
        mask_permanent = Permanent(ancestral_mask_def)
        mask_permanent.flags["enchanting"] = blocker_with_mask
        state.players[0].battlefield = [attacker_with_rancor, rancor_permanent]
        state.players[1].battlefield = [blocker_with_mask, mask_permanent]

        attacker_with_rancor.damage_marked = 1  # lethal (toughness 1)
        blocker_with_mask.damage_marked = 1  # lethal (toughness 1)
        check_state_based_actions(state)

        assert attacker_with_rancor not in state.players[0].battlefield
        assert [c.name for c in state.players[0].graveyard] == ["Rancor'd Attacker"]
        assert rancor_permanent not in state.players[0].battlefield
        assert [c.name for c in state.players[0].hand] == ["Rancor"]  # returned to hand, not the graveyard
        assert rancor_def not in state.players[0].graveyard

        assert blocker_with_mask not in state.players[1].battlefield
        assert mask_permanent not in state.players[1].battlefield
        assert sorted(c.name for c in state.players[1].graveyard) == ["Ancestral Mask", "Masked Blocker"]  # ordinary Aura -- graveyard, not hand
        assert state.players[1].hand == []

        # cleanup_step clears damage_marked for EVERY permanent, both
        # players -- not just the active player's own side.
        cleanup_step(state)
        assert state.pending_resolution is None
    finally:
        registry.CARD_DEFS.clear()
        registry.CARD_DEFS.update(_card_defs_backup)

    print("state_based.py Aura-orphaning + cleanup self-check: OK")

    # Token creature death: ceases to exist entirely -- same real-Magic
    # rule every existing token-removal path already follows, NOT the
    # graveyard-goes-there-normally case above.
    from .tokens import WARRIOR_TOKEN_CARD_DEF

    state = GameState(on_the_play=True, players=[PlayerState(True), PlayerState(False)])
    warrior_token = Permanent(WARRIOR_TOKEN_CARD_DEF)
    state.players[0].battlefield = [warrior_token]
    warrior_token.damage_marked = 1  # lethal (toughness 1)
    check_state_based_actions(state)
    assert warrior_token not in state.players[0].battlefield
    assert state.players[0].graveyard == []  # ceased to exist -- never added to any zone

    print("state_based.py token-death self-check: OK")

    # destroy_permanent: indestructible can't be destroyed; a land goes to its
    # owner's graveyard; a creature routes through _destroy_creature.
    state = GameState(on_the_play=True, players=[PlayerState(True), PlayerState(False)])
    bridge = Permanent(CardDef("Drossforge Bridge", CardType.LAND, None, EffectId.DROSSFORGE_BRIDGE, artifact=True, indestructible=True))
    plain_land = Permanent(CardDef("Great Furnace", CardType.LAND, None, EffectId.GREAT_FURNACE, artifact=True))
    _card_defs_backup = dict(registry.CARD_DEFS)
    registry.CARD_DEFS["Great Furnace"] = plain_land.card_def  # so it goes to graveyard (not treated as a token)
    try:
        state.players[0].battlefield = [bridge, plain_land]
        assert destroy_permanent(state, bridge) is False  # indestructible -- survives
        assert bridge in state.players[0].battlefield
        assert destroy_permanent(state, plain_land) is True
        assert plain_land not in state.players[0].battlefield
        assert [c.name for c in state.players[0].graveyard] == ["Great Furnace"]
    finally:
        registry.CARD_DEFS.clear()
        registry.CARD_DEFS.update(_card_defs_backup)
    print("state_based.py destroy_permanent (indestructible + land) self-check: OK")

    # Deathtouch SBA: a creature with ANY marked damage from a deathtouch
    # source dies, even below its toughness; temp modifiers + the deathtouch
    # marker clear at cleanup.
    state = GameState(on_the_play=True, players=[PlayerState(True), PlayerState(False)])
    big = Permanent(CardDef("Big", CardType.CREATURE, None, EffectId.FILLER, power=5, toughness=5))
    state.players[1].battlefield = [big]
    big.damage_marked = 1
    big.flags["deathtouched"] = True  # as combat.py would set for a deathtouch hit
    check_state_based_actions(state)
    assert big not in state.players[1].battlefield  # 1 < 5 toughness, but deathtouched -> dead

    # temp P/T: -0/-3 to a 3-toughness creature is lethal via SBA (0 toughness).
    state = GameState(on_the_play=True, players=[PlayerState(True), PlayerState(False)])
    warped = Permanent(CardDef("Warped", CardType.CREATURE, None, EffectId.FILLER, power=2, toughness=3))
    state.players[0].battlefield = [warped]
    warped.temp_toughness = -3
    check_state_based_actions(state)
    assert warped not in state.players[0].battlefield

    # cleanup clears temp modifiers + deathtouch marker.
    state = GameState(on_the_play=True, players=[PlayerState(True), PlayerState(False)])
    surv = Permanent(CardDef("Survivor", CardType.CREATURE, None, EffectId.FILLER, power=2, toughness=5))
    surv.temp_power, surv.temp_toughness, surv.temp_keywords = -3, -1, {"deathtouch"}
    surv.flags["deathtouched"] = True
    state.players[0].battlefield = [surv]
    cleanup_step(state)
    assert surv.temp_power == 0 and surv.temp_toughness == 0 and surv.temp_keywords == set()
    assert "deathtouched" not in surv.flags
    print("state_based.py deathtouch SBA + temp-modifier clear self-check: OK")

    # sacrifice_to_graveyard: fires a dies (ltb) trigger AND on_sacrifice
    # triggers on the sacrificer's other battlefield permanents.
    _filler_backup3 = registry.EFFECT_REGISTRY[EffectId.FILLER]
    dies_fired = []
    sac_seen = []
    try:
        state = GameState(on_the_play=True)
        registry.EFFECT_REGISTRY[EffectId.FILLER] = {
            "ltb_trigger": lambda s, p: dies_fired.append(p.card_def.name),
            "on_sacrifice": lambda s, p, sacced: sac_seen.append(sacced.name),
        }
        watcher = Permanent(CardDef("Watcher", CardType.CREATURE, None, EffectId.FILLER, power=1, toughness=1))
        victim = Permanent(CardDef("Victim", CardType.ARTIFACT, {"generic": 1}, EffectId.FILLER))
        registry.CARD_DEFS["Watcher"] = watcher.card_def
        registry.CARD_DEFS["Victim"] = victim.card_def
        state.battlefield = [watcher, victim]
        sacrifice_to_graveyard(state, victim)
        assert victim.card_def in state.graveyard  # real card -> graveyard
        assert [e for e in state.trigger_queue if e["type"] == "ltb"]  # its dies-trigger queued
        assert sac_seen == ["Victim"]  # Watcher's on_sacrifice saw the sacrifice ("another permanent")
    finally:
        registry.EFFECT_REGISTRY[EffectId.FILLER] = _filler_backup3
        registry.CARD_DEFS.pop("Watcher", None)
        registry.CARD_DEFS.pop("Victim", None)
    print("state_based.py sacrifice_to_graveyard (dies + sacrifice triggers) self-check: OK")
