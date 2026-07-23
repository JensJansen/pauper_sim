"""State-based actions (creature death) and end-of-turn cleanup. Depends
on stats.py for effective toughness, and on registry.py's own CARD_DEFS/
EFFECT_REGISTRY (lazily, same convention as every other submodule here --
see game/registry.py's own module docstring) to tell a real card apart
from a token and to look up an orphaned Aura's own return-to-hand flag."""

from . import stats
from .. import registry, resolution
from ..cards import CardType

HAND_SIZE_LIMIT = 7  # real Magic's own rule -- not a per-config tunable, no card in this pool ever modifies it


def check_state_based_actions(state):
    """Creature-death check (docs/COMBAT_PLAN.md step 6, generalized by
    docs/PRIORITY_PLAN.md item 2 to run before every priority
    consultation, not just after combat damage -- real Magic 704.3: SBAs
    are checked every time a player would receive priority): every
    creature on EITHER player's battlefield with damage_marked >= its own
    effective permanent_toughness dies -- removed to the graveyard, its
    own attached Aura(s) orphaned along with it (Aura-orphaning:
    casting.cast_aura's own docstring flagged this as unreachable before
    combat death existed to trigger it). Collects every dead creature
    FIRST, then removes all of them -- matches real Magic's simultaneous
    state-based-action semantics, not a one-at-a-time recheck that could
    let removing one change whether another is considered dead.

    Scans the whole battlefield rather than a passed-in candidate list
    (an earlier, narrower version of this function only checked whichever
    creatures had just taken combat damage) -- simpler and provably
    correct: a creature that wasn't just damaged can't have newly crossed
    its own lethal threshold, so scanning everyone is a strict superset,
    never wrong, and cheap given how few creatures are ever in play at
    once. Needed once this runs unconditionally before every priority
    round, not just once per combat, since there's no single "what just
    changed" set to narrow to anymore."""
    candidates = [
        p for player in state.players for p in player.battlefield if p.card_type == CardType.CREATURE
    ]
    dead = [p for p in candidates if p.damage_marked >= stats.permanent_toughness(state, p)]
    for permanent in dead:
        _destroy_creature(state, permanent)


def _destroy_creature(state, permanent):
    """One creature's actual death: battlefield -> graveyard, plus
    orphaning whatever Aura(s) were enchanting it. Operates on whichever
    PlayerState actually owns `permanent` (found by membership, not
    state.battlefield/state.graveyard/state.hand) -- combat.combat_damage_
    step always runs with state.active_idx on the ATTACKER, but a dying
    permanent can just as easily be the DEFENDER's own blocker, whose
    zones the active-player-proxied accessors would silently get wrong
    (or, for battlefield.remove, raise ValueError outright, since the
    blocker was never in the attacker's own list). An orphaned Aura goes
    to its controller's graveyard by default (every real-Magic Aura's
    default behavior) -- Rancor is the one exception in this card pool
    (returns to hand instead), flagged via a static
    "returns_to_hand_when_orphaned" registry key rather than a callback,
    since it's a fixed per-card fact, never something that needs to be
    computed.

    A TOKEN creature (name absent from registry.CARD_DEFS -- the same
    membership check TOKEN_LIMIT's own accounting uses) never goes to the
    graveyard: real Magic's own rule is that a token ceases to exist
    entirely once it leaves the battlefield, matching every existing
    token-removal path (tokens.activate_blood_sac's own docstring says
    this explicitly). build_observation's graveyard_counts is keyed only
    by real decklist names, same as every other zone-count block there --
    appending a token's card_def would have KeyError'd the very next
    observation build, caught by a live training smoke test with real
    token-creating cards (boggles' own Malevolent Rumble/Cartouche of
    Solidarity) rather than any narrower unit self-check.

    A Bestowed permanent (Nyxborn Hydra) is a third, distinct orphan
    outcome, real Magic's own Bestow fall-off rule: instead of moving to
    any zone at all, it just STAYS on the battlefield and becomes a
    creature again -- flagged via "becomes_creature_when_orphaned" (a fixed
    per-card fact, same shape as "returns_to_hand_when_orphaned"), checked
    first since it's mutually exclusive with the graveyard/hand branches
    below (an orphaned permanent either changes zones or doesn't). Clearing
    type_override back to None is enough on its own -- Nyxborn Hydra's own
    card_def.card_type is already CREATURE, so there's nothing else to
    restore it to. Its own counters (the +1/+1s Bestow entered with) are
    untouched, matching "maintain its own +1/+1s" -- this function never
    touches permanent.counters at all."""
    owner = next(player for player in state.players if permanent in player.battlefield)
    owner.battlefield.remove(permanent)
    if permanent.card_def.name in registry.CARD_DEFS:
        owner.graveyard.append(permanent.card_def)
    orphaned = [p for p in owner.battlefield if p.flags.get("enchanting") is permanent]
    for aura in orphaned:
        spec = registry.EFFECT_REGISTRY.get(aura.card_def.effect_id, {})
        if spec.get("becomes_creature_when_orphaned", False):
            aura.flags.pop("enchanting", None)
            aura.type_override = None
            continue
        owner.battlefield.remove(aura)
        if spec.get("returns_to_hand_when_orphaned", False):
            owner.hand.append(aura.card_def)
        else:
            owner.graveyard.append(aura.card_def)


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
    (docs/MULTIPLAYER_ENGINE_PLAN.md) -- without this there was no
    ceiling on hand size in an adversarial game at all."""
    for player in state.players:
        for permanent in player.battlefield:
            permanent.damage_marked = 0
    n = max(0, len(state.hand) - HAND_SIZE_LIMIT)
    resolution.begin_discard(state, n, optional=False, on_complete=lambda s, _cards: None)


if __name__ == "__main__":
    # ponytail self-check: run via `python -m game.effects.state_based`
    # from src/. Aura-orphaning (docs/COMBAT_PLAN.md step 6) and
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
