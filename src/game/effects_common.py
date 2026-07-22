"""Generic effect plumbing every color catalog's own cast/activate
functions call into: battlefield entry, combat, the Madness/Plot/token
mechanics, and the trigger queue. No per-card catalog entries live here
anymore -- every real card (including ones once "shared between decks"
like Forest/Lightning Bolt/Generous Ent) now lives exactly once in its
own color file under game/catalog/, since cards are no longer owned by
decks at all (DECK_REGISTRY_REFRESH_PLAN.md). The two token CardDefs
below (Blood/Robot) are the one exception -- they're never in CARD_DEFS
or any decklist, so a color file isn't the right home for them either.

References game.registry.EFFECT_REGISTRY (the merged registry, built from
every color catalog's own fragment plus registry.py's union) only from
inside function bodies, via `registry.EFFECT_REGISTRY`, never as a bare
name imported at module load time -- registry.py imports the catalog
modules, which import this module, so a `from .registry import
EFFECT_REGISTRY` here would try to bind a name that doesn't exist yet.
Deferring the lookup to call time (Python resolves a name inside a
function body only when that function runs, not when it's defined) breaks
the cycle: by the time anything actually calls enters_battlefield,
game/__init__.py has already finished importing every submodule.

Also holds the Madness "cast for its madness cost" path (execute_madness_
cast), the Plot "pay cost, exile instead of resolving" path (plot_to_
exile), and the trigger-queue-to-stack promotion (promote_triggers_to_
stack, docs/PRIORITY_PLAN.md item 1) -- these need game.mana.begin_pay_
cost, which game.resolution can't import (mana.py imports resolution.py
at its own top level; the reverse would cycle).
This module is free to depend on both resolution.py and mana.py, so it's
where that orchestration lives instead. See docs/MADNESS_DECKS_PLAN.md
items 1/3/4/7.
"""

from . import mana, registry, resolution
from .cards import CardDef, CardType, EffectId
from .state import Permanent


def play_land_from_hand(state, card_def):
    state.hand.remove(card_def)
    state.lands_played_this_turn += 1
    return enters_battlefield(state, card_def)


def cast_permanent_from_hand(state, card_def):
    """Artifacts/creatures with no additional cost beyond mana and no
    target choices. Mana cost is paid by the caller first."""
    state.hand.remove(card_def)
    return enters_battlefield(state, card_def)


def cast_aura(state, card_def, target_predicate, on_attached=None):
    """Cast an Aura from hand: pick a legal target via
    resolution.begin_choose_permanent (the same "one of my own permanents
    matching a predicate" primitive Crop Rotation's sacrifice-target and
    bounce_land_etb's bounce-target already use), then attach by recording
    the target on the Aura's own flags. Every Aura's registry "cast"/
    "cast_modes" entry pairs this with an extra_legal requiring
    target_predicate to already match something on the battlefield at CAST
    time -- but this function itself only actually runs once the spell
    resolves off state.stack (game.push_to_stack), which can now be well
    after the cast, with other instant-speed responses (or activated
    abilities) in between free to change the battlefield. So a legal
    target is guaranteed at cast time, never guaranteed to still exist by
    the time this runs -- see _on_chosen's own fizzle branch below for
    what happens if it doesn't (unreachable before the stack existed, when
    this always ran in the very same instant as its own extra_legal check).

    on_attached(state, aura_permanent), if given, runs once attached -- for
    an Aura with its own ETB effect (Abundant Growth's draw, Cartouche of
    Solidarity's token, Utopia Sprawl's chosen color). Routed through here
    rather than the registry's own etb_trigger (which only ever receives
    state, not the permanent) since every one of these needs to record
    something onto the Aura's own Permanent, not just act on shared state.

    on_attached(state, aura_permanent), if given, runs once attached -- for
    an Aura with its own ETB effect (Abundant Growth's draw, Cartouche of
    Solidarity's token, Utopia Sprawl's chosen color). Routed through here
    rather than the registry's own etb_trigger (which only ever receives
    state, not the permanent) since every one of these needs to record
    something onto the Aura's own Permanent, not just act on shared state.

    Real-rules note: an Aura returns to the graveyard (and, for Rancor,
    from there back to hand) when whatever it enchants leaves the
    battlefield ("orphaning") -- now modeled for the one reachable case in
    this card pool, combat death (docs/COMBAT_PLAN.md step 6, see
    _destroy_creature). Every OTHER battlefield-removal call site in this
    codebase (sacrifice, bounce, exile -- see their own call sites) still
    doesn't orphan an enchanted permanent's Auras, since none of them can
    currently target a creature that could be enchanted (boggles is the
    only deck with Auras, and none of its own cards sacrifice/bounce/exile
    a creature). Thread the same orphaning logic through a removal site
    if a future card ever makes that reachable."""
    state.hand.remove(card_def)

    def _on_chosen(state, name):
        if name is None:
            # No legal target left by the time this resolves -- real
            # Magic: a spell that resolves with no legal target left does
            # nothing and goes to the graveyard, same as any other spell
            # whose effect happens to do nothing (never enters the
            # battlefield at all here, unlike a normal successful cast).
            state.graveyard.append(card_def)
            return
        target = next(p for p in state.battlefield if p.card_def.name == name and target_predicate(p))
        aura = enters_battlefield(state, card_def)
        aura.flags["enchanting"] = target
        if on_attached is not None:
            on_attached(state, aura)

    resolution.begin_choose_permanent(state, target_predicate, _on_chosen)


def _check_end_of_game(state):
    """Central check for every way the game can end -- called from every
    place board state can change enough to matter: enters_battlefield,
    combat_damage_step, and deal_damage_to_opponent. Replaces what used to
    be the same two lines duplicated at each of those call sites (see
    docs/MULTIPLAYER_ENGINE_PLAN.md).

    Two independent ways to win: the active player's own terminated_fn
    (their deck's combo-completion condition -- Tron assembly, a damage
    threshold) firing, or -- 2-player only -- the opponent's life_total
    hitting 0. No-ops once the game has already ended."""
    if state.turn_won is not None:
        return
    active_idx = state.active_idx
    active = state.players[active_idx]
    if active.terminated_fn is not None and active.terminated_fn(state):
        state.turn_won = state.turn_number
        state.winner = active_idx
        return
    if len(state.players) > 1 and state.opponent.life_total <= 0:
        state.turn_won = state.turn_number
        state.winner = active_idx


def deal_damage_to_opponent(state, n):
    """Every 'deals N damage to the opponent' effect routes through here
    -- the single choke point keeping state.damage_dealt (the historical
    1-player abstraction; terminated.damage_threshold_terminated and every
    burn deck's own win condition still reads it, unchanged) and the real
    per-player life_total (docs/MULTIPLAYER_ENGINE_PLAN.md) in sync. In
    1-player mode there's no second PlayerState to decrement -- the
    damage_dealt bump alone is still the whole story there, exactly as
    before this function existed."""
    state.damage_dealt += n
    if len(state.players) > 1:
        state.opponent.life_total -= n
    _check_end_of_game(state)


def enters_battlefield(state, card_def, force_tapped=False):
    """Move a CardDef onto the battlefield as a new Permanent, applying its
    enters-tapped default and ETB trigger (via game.registry.EFFECT_REGISTRY),
    then check _check_end_of_game since a permanent entering is the only
    way a terminated_fn-based win condition can newly become true. Caller
    has already removed card_def from its previous zone (hand/library).

    force_tapped=True overrides the registry's own enters_tapped default
    to always-tapped -- a one-off per-trigger condition, not a property of
    the card itself (Sneaky Snacker enters battlefield normally untapped
    when cast, but tapped specifically when its own "third card drawn"
    trigger returns it from the graveyard -- docs/MADNESS_DECKS_PLAN.md
    item 7). Every existing caller omits it, unaffected."""
    spec = registry.EFFECT_REGISTRY.get(card_def.effect_id, {})
    tapped = force_tapped or spec.get("enters_tapped", False)
    permanent = Permanent(card_def, tapped=tapped)
    # Pooled slot assignment (docs/COMBAT_PLAN.md): the lowest number not
    # already in use among this player's currently-live permanents of the
    # same name. Never a running/monotonic count -- a name's slot numbers
    # simply free up once whatever was using them leaves the battlefield,
    # which is what keeps this bounded (by how many can be simultaneously
    # alive, i.e. decklist quantity) even through repeated bounce/blink,
    # rather than growing with how many turns have been played.
    used_slots = {p.slot for p in state.battlefield if p.card_def.name == card_def.name}
    slot = 1
    while slot in used_slots:
        slot += 1
    permanent.slot = slot
    state.battlefield.append(permanent)

    etb_trigger = spec.get("etb_trigger")
    if etb_trigger is not None:
        etb_trigger(state)
    # Bojuka Bog's "exile target player's graveyard" ETB is a documented
    # no-op in both 1- and 2-player games: no card currently reaches into
    # this graveyard-exile mechanic, so it stays unimplemented regardless
    # of whether a real opponent graveyard now exists to target.

    _check_end_of_game(state)

    return permanent


def _enchanting_auras(state, permanent):
    """Every Aura currently enchanting `permanent`, searched across BOTH
    players' own battlefields (state.players directly, NOT the
    active-player-proxied state.battlefield). An Aura and whatever it
    enchants are always on the SAME side (cast_aura's own target
    predicate only ever matches the caster's own battlefield at cast
    time) -- but the caller here has no guarantee state.active_idx
    currently points at that side. In particular, combat_damage_step
    always runs with active_idx on the ATTACKER, yet needs a BLOCKER's
    correct effective power/toughness (the defender's own creature) --
    reading state.battlefield there would silently search the wrong
    player's board and miss every Aura bonus on the blocker entirely."""
    for player in state.players:
        for aura in player.battlefield:
            if aura.flags.get("enchanting") is permanent:
                yield aura


def enchantment_count(state, aura):
    """How many ENCHANTMENT-type permanents `aura`'s OWN controller has on
    the battlefield -- shared by every "for each [other] enchantment you
    control" pt_bonus (Ancestral Mask/Ethereal Armor differ only in
    whether the caller subtracts 1 for itself). Takes `aura` (found via
    _enchanting_auras above) so the right controller can be found by
    membership, same reasoning as that function's own docstring -- reading
    state.battlefield instead would count whichever player is currently
    active, not the enchantment's own controller, wrongly conflating the
    two the instant a Blocker's own Ancestral Mask/Ethereal Armor is read
    during combat (attacker active, blocker's Auras on the defender's
    side)."""
    owner = next(player for player in state.players if aura in player.battlefield)
    return sum(1 for p in owner.battlefield if p.card_def.card_type == CardType.ENCHANTMENT)


def permanent_power(state, permanent):
    """A creature's effective power for combat_damage_step (and Ram Through, once
    it's more than a functional blank): its own base power
    (card_def.extra["power"], 0 if absent -- no creature is absent one
    anymore, docs/COMBAT_PLAN.md's full-stats pass, but the default stays
    for FILLER/synthetic self-check permanents) plus every Aura currently
    enchanting it (_enchanting_auras above -- owner-agnostic, correct
    regardless of state.active_idx). Each Aura's registry entry supplies
    its own "pt_bonus" (state, aura_permanent) -> int -- a constant for a
    static bonus (Rancor's +2), a battlefield-wide count for a dynamic one
    (Ancestral Mask/Ethereal Armor's "for each [other] enchantment")."""
    base = permanent.card_def.extra.get("power", 0)
    bonus = sum(
        registry.EFFECT_REGISTRY.get(aura.card_def.effect_id, {}).get("pt_bonus", lambda *_a: 0)(state, aura)
        for aura in _enchanting_auras(state, permanent)
    )
    return base + bonus


def permanent_toughness(state, permanent):
    """A creature's effective toughness -- its own base toughness
    (card_def.extra["toughness"], 0 if absent, same convention as
    permanent_power) plus every Aura currently enchanting it. Deliberately
    NOT the same registry key as permanent_power's own "pt_bonus": real
    Rancor is +2/+0 (power only), so reusing pt_bonus here would wrongly
    also buff toughness. A separate, optional "toughness_bonus" key
    (defaulting to 0, same as pt_bonus's own default) covers the Auras
    that genuinely are symmetric in real Magic (Ancestral Mask/Ethereal
    Armor/Cartouche of Solidarity/Armadillo Cloak are each +X/+X) without
    touching permanent_power's own already-tested logic at all."""
    base = permanent.card_def.extra.get("toughness", 0)
    bonus = sum(
        registry.EFFECT_REGISTRY.get(aura.card_def.effect_id, {}).get("toughness_bonus", lambda *_a: 0)(state, aura)
        for aura in _enchanting_auras(state, permanent)
    )
    return base + bonus


# Real Magic keyword strings this engine models (docs/COMBAT_PLAN.md's
# confirmed scope -- only these four, only on the specific cards that
# already grant one): "vigilance" (Cartouche of Solidarity's Warrior
# token), "flying" (Kitchen Imp's real flying; also used for Silhana
# Ledgewalker's "can't be blocked except by creatures with flying" --
# functionally the identical blocking restriction in a ruleset with no
# reach, so one flag covers both rather than a second near-duplicate),
# "trample" (Rancor, Armadillo Cloak), "first_strike" (Cartouche of
# Solidarity, Ethereal Armor). Deathtouch/double strike/menace/reach:
# no card grants any of them -- not modeled, not a registry key.
def creature_keywords(state, permanent):
    """Union of this permanent's own intrinsic registry "keywords" set
    (a creature's own EFFECT_REGISTRY entry) plus every Aura currently
    enchanting it own GRANTED "keywords" set (an Aura's own EFFECT_REGISTRY
    entry) -- same "own base fact plus every enchanting Aura's own
    contribution" shape as permanent_power/permanent_toughness, reusing
    the same owner-agnostic _enchanting_auras (correct regardless of
    state.active_idx, e.g. reading a blocker's keywords from inside
    combat_damage_step, which always runs with active_idx on the
    attacker)."""
    keywords = set(registry.EFFECT_REGISTRY.get(permanent.card_def.effect_id, {}).get("keywords", ()))
    for aura in _enchanting_auras(state, permanent):
        keywords |= set(registry.EFFECT_REGISTRY.get(aura.card_def.effect_id, {}).get("keywords", ()))
    return keywords


def has_keyword(state, permanent, keyword):
    return keyword in creature_keywords(state, permanent)


def creature_attack_eligible(state, permanent):
    """Untapped, not a Defender (Wall of Roots/Overgrown Battlement/Saruli
    Caretaker/Gatecreeper Vine -- real Magic's own rule: a Defender can
    never attack, full stop, regardless of tapped/summoning-sick status),
    and not summoning sick unless it has a registry "haste": True spec
    (Kitchen Imp) -- the only other place that flag is ever read, so this
    is the only place haste needs to matter. Checked per creature (drl_env's
    "Attack: <name>" actions) so a model can declare SOME eligible
    creatures as attackers and hold others back (as blockers once those
    exist, or as mana sources) instead of the old everyone-eligible-attacks
    wholesale rule."""
    return (
        permanent.card_def.card_type == CardType.CREATURE and not permanent.tapped
        and not permanent.card_def.extra.get("defender", False)
        and (not permanent.summoning_sick or registry.EFFECT_REGISTRY.get(permanent.card_def.effect_id, {}).get("haste", False))
    )


def creature_block_eligible(state, permanent):
    """Untapped and not already assigned to block something else this
    combat -- no gang-blocking/menace modeled, docs/COMBAT_PLAN.md. Reads
    state.opponent.blocked_by, NOT state.blocked_by: this is only ever
    called with state.active_idx already flipped to the defender
    (game.turn._declare_blockers_gen), and PlayerState.blocked_by is keyed
    by the ATTACKING player's own attacker permanents (see its own
    docstring) -- from the flipped-to-defender perspective, that dict
    lives on state.opponent, not on the (active, defending) player
    state.blocked_by itself would read. Deliberately NOT the same
    eligibility as creature_attack_eligible: real Magic lets a Defender
    block (that's its whole point) and lets a summoning-sick creature
    block (summoning sickness only restricts attacking and {T}
    abilities) -- so neither check belongs here."""
    return (
        permanent.card_def.card_type == CardType.CREATURE and not permanent.tapped
        and permanent not in state.opponent.blocked_by.values()
    )


def declare_attacker(state, permanent):
    """Model chose to attack with this specific creature -- addressed by
    (name, slot) at the drl_env.py action-table layer
    (docs/COMBAT_PLAN.md's permanent-identity design), so the caller has
    already picked the exact physical copy it means, not an arbitrary
    same-named match. Tapped here, at declaration, same as real Magic --
    an attacking creature is unavailable for a mana ability etc. for the
    rest of combat, not just tapped as a side effect of dealing damage
    later -- UNLESS it has vigilance (Cartouche of Solidarity's own
    Warrior token), which is real Magic's entire point of the keyword:
    attacking doesn't tap it at all."""
    if not has_keyword(state, permanent, "vigilance"):
        permanent.tapped = True
    state.attackers.append(permanent)


def declare_attackers_step(state):
    """game.turn.Phase.DECLARE_ATTACKERS phase-entry reset (rakdos
    madness / mono red madness / boggles -- gated by combat_enabled, same
    as combat always was): clears last turn's attackers AND blocks (both
    reset together -- a fresh combat has neither yet) so the model's own
    "Attack: <name> (slot k)" actions (drl_env.build_action_table, each
    one checking creature_attack_eligible and calling declare_attacker)
    start this turn's declaration fresh, one creature at a time, rather
    than the old wholesale auto-attack."""
    state.attackers = []
    state.blocked_by = {}


def _is_alive(state, permanent):
    return any(permanent in player.battlefield for player in state.players)


def _attacker_deal_damage(state, attacker, blocker):
    """Attacker deals its combat damage to its blocker -- trample-aware
    (Rancor, Armadillo Cloak): assigns only enough to be lethal (the
    blocker's own remaining toughness), letting any excess spill over to
    the DEFENDING player via deal_damage_to_opponent (state.opponent, from
    the attacker's own active perspective, IS the defender throughout
    combat_damage_step) instead of being wasted on an already-dead
    blocker. Real Magic lets the attacker choose to overkill the blocker
    instead -- never correct without deathtouch (not modeled in this card
    pool), so this always assigns the minimum lethal, no extra decision or
    action needed."""
    power = permanent_power(state, attacker)
    if has_keyword(state, attacker, "trample"):
        lethal = min(power, max(permanent_toughness(state, blocker) - blocker.damage_marked, 0))
        blocker.damage_marked += lethal
        excess = power - lethal
        if excess > 0:
            deal_damage_to_opponent(state, excess)
    else:
        blocker.damage_marked += power


def _blocker_deal_damage(state, blocker, attacker):
    """Blocker deals its combat damage to the attacker it's blocking --
    never tramples through to a player: trample is an attacking-creature
    keyword only, nothing in this card pool grants a blocker-side
    equivalent, and this engine doesn't model one."""
    attacker.damage_marked += permanent_power(state, blocker)


def combat_damage_step(state):
    """game.turn.Phase.COMBAT_DAMAGE: total power (permanent_power(state,
    p) -- base card_def.extra["power"] plus any attached Auras' own
    "pt_bonus") of state.attackers NOT present in state.blocked_by
    (declared in DECLARE_ATTACKERS via declare_attacker; assigned a
    blocker, if any, during the defending player's own consult --
    docs/COMBAT_PLAN.md) hits the opponent via deal_damage_to_opponent
    once; a creature with neither power nor an Aura set (e.g. an
    untracked-stats vanilla from another deck) contributes 0.

    Every blocked attacker/blocker pair fights in up to two sub-steps,
    real Magic's own first-strike ordering: first, whichever side(s) of
    each pair have first_strike (Cartouche of Solidarity, Ethereal Armor)
    deal their damage and a state-based-action check runs -- a first-
    strike kill here means the victim is already gone before the regular
    sub-step, so it never deals damage back. Then, whichever side(s)
    DON'T have first strike deal their damage (their only shot -- a
    first-strike side already had its one shot above), but only if both
    it and its target are still alive, followed by a second SBA check.
    With no first strike anywhere in a given combat this collapses to the
    exact same single simultaneous exchange as before first strike
    existed (every pair's damage all lands in the "regular" sub-step)."""
    unblocked_total = sum(permanent_power(state, p) for p in state.attackers if p not in state.blocked_by)
    state.attackers = []
    deal_damage_to_opponent(state, unblocked_total)

    pairs = list(state.blocked_by.items())

    for attacker, blocker in pairs:
        if has_keyword(state, attacker, "first_strike"):
            _attacker_deal_damage(state, attacker, blocker)
        if has_keyword(state, blocker, "first_strike"):
            _blocker_deal_damage(state, blocker, attacker)
    check_state_based_actions(state)

    for attacker, blocker in pairs:
        attacker_alive, blocker_alive = _is_alive(state, attacker), _is_alive(state, blocker)
        if not has_keyword(state, attacker, "first_strike") and attacker_alive and blocker_alive:
            _attacker_deal_damage(state, attacker, blocker)
        if not has_keyword(state, blocker, "first_strike") and blocker_alive and attacker_alive:
            _blocker_deal_damage(state, blocker, attacker)
    check_state_based_actions(state)


def check_state_based_actions(state):
    """Creature-death check (docs/COMBAT_PLAN.md step 6, generalized by
    docs/PRIORITY_PLAN.md item 2 to run before every priority
    consultation, not just after combat damage -- real Magic 704.3: SBAs
    are checked every time a player would receive priority): every
    creature on EITHER player's battlefield with damage_marked >= its own
    effective permanent_toughness dies -- removed to the graveyard, its
    own attached Aura(s) orphaned along with it (Aura-orphaning:
    cast_aura's own docstring flagged this as unreachable before combat
    death existed to trigger it). Collects every dead creature FIRST,
    then removes all of them -- matches real Magic's simultaneous
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
        p for player in state.players for p in player.battlefield if p.card_def.card_type == CardType.CREATURE
    ]
    dead = [p for p in candidates if p.damage_marked >= permanent_toughness(state, p)]
    for permanent in dead:
        _destroy_creature(state, permanent)


def _destroy_creature(state, permanent):
    """One creature's actual death: battlefield -> graveyard, plus
    orphaning whatever Aura(s) were enchanting it. Operates on whichever
    PlayerState actually owns `permanent` (found by membership, not
    state.battlefield/state.graveyard/state.hand) -- combat_damage_step
    always runs with state.active_idx on the ATTACKER, but a dying
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
    token-removal path (activate_blood_sac's own docstring says this
    explicitly). build_observation's graveyard_counts is keyed only by
    real decklist names, same as every other zone-count block there --
    appending a token's card_def would have KeyError'd the very next
    observation build, caught by a live training smoke test with real
    token-creating cards (boggles' own Malevolent Rumble/Cartouche of
    Solidarity) rather than any narrower unit self-check."""
    owner = next(player for player in state.players if permanent in player.battlefield)
    owner.battlefield.remove(permanent)
    if permanent.card_def.name in registry.CARD_DEFS:
        owner.graveyard.append(permanent.card_def)
    orphaned = [p for p in owner.battlefield if p.flags.get("enchanting") is permanent]
    for aura in orphaned:
        owner.battlefield.remove(aura)
        spec = registry.EFFECT_REGISTRY.get(aura.card_def.effect_id, {})
        if spec.get("returns_to_hand_when_orphaned", False):
            owner.hand.append(aura.card_def)
        else:
            owner.graveyard.append(aura.card_def)


HAND_SIZE_LIMIT = 7  # real Magic's own rule -- not a per-config tunable, no card in this pool ever modifies it


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
    state.hand.remove(card_def)
    state.graveyard.append(card_def)


def any_creature_on_battlefield(state):
    """Shared "is there a legal Aura/targeted-effect target at all" gate --
    Rancor/Ancestral Mask/Armadillo Cloak/Cartouche of Solidarity/Ethereal
    Armor's own extra_legal all reduce to exactly this."""
    return any(p.card_def.card_type == CardType.CREATURE for p in state.battlefield)


def on_cast_trigger(state, card_def):
    """Fires at cast time, before the spell's own resolve runs -- matches
    real Magic timing (the triggered ability goes on the stack above the
    spell it triggered off, so it happens first). Every cast path (normal
    cast, alt_cast, Flashback, and Plot's cast-from-exile) calls this
    identically, from inside drl_env.py's own per-path wrapper functions
    -- so a card like Guttersnipe ("whenever you cast an instant or
    sorcery...") fires the same regardless of which path cast the
    triggering spell. See docs/MADNESS_DECKS_PLAN.md item 11."""
    if card_def.card_type not in (CardType.INSTANT, CardType.SORCERY):
        return
    for permanent in state.battlefield:
        trigger = registry.EFFECT_REGISTRY.get(permanent.card_def.effect_id, {}).get("on_cast")
        if trigger is not None:
            trigger(state, permanent)


def push_to_stack(state, card_def, resolve):
    """A spell is fully paid for (mana or an alternate cost) but not yet
    resolved -- defer `resolve(state, card_def)` onto state.stack instead of
    calling it now, giving the model a chance to respond (cast another
    instant-speed spell) before it resolves. Every cast-like top-level
    action (normal cast, cast_modes, alt_cast, Flashback, Plot's cast-from-
    exile, Madness) pushes here once its own cost-payment is fully done --
    never before (a card whose alt cost is itself a resolution, e.g.
    Fireblast's sacrifice-2-Mountains, must push only from that
    resolution's own on_complete, not from inside the sacrifice itself).

    A pushed card's own hand/graveyard/exile removal still happens inside
    `resolve` itself (unchanged from before the stack existed), not here --
    so a card sitting on the stack, paid for but unresolved, is still
    physically present in whatever zone it came from until it actually
    resolves. Two places correct for that instead of treating it as
    "available": drl_env._hand_count_available (cast legality -- also
    already a non-issue for every Flashback/Plot/Madness path, which each
    remove from their own zone before ever reaching a resolution that
    could push) and resolution.discard_options (an instant-speed activated
    ability, e.g. Blood's sac-for-a-card, isn't blocked by a non-empty
    stack the way a sorcery-speed cast is, so it can still be offered a
    card that's actually already spoken for by an unresolved stack entry).

    Records state.active_idx as this entry's own controller (docs/
    PRIORITY_PLAN.md): a real priority round can flip active_idx through
    both players (whoever's currently deciding to act/pass) between now
    and whenever this entry actually resolves, but state.hand/graveyard/
    battlefield (state.py's own active_idx-proxy) must still resolve
    against the CASTER's zones, not whoever last happened to hold
    priority -- resolve_top_of_stack restores it below."""
    state.stack.append({"card_def": card_def, "resolve": resolve, "controller": state.active_idx})


def resolve_top_of_stack(state):
    """Pop and resolve the most recently pushed spell -- LIFO, no
    reordering action needed (real Magic's own stack order). Called once
    per "Pass" while state.stack is non-empty (game.turn._run_turn_gen),
    never automatically -- the model must explicitly let it happen instead
    of casting something else in response.

    Restores active_idx to this entry's own controller (push_to_stack)
    before resolving: by the time all players have passed in a row,
    active_idx may be sitting on whoever passed last, not the original
    caster (docs/PRIORITY_PLAN.md) -- resolve must run from the
    controller's own zone perspective regardless."""
    entry = state.stack.pop()
    state.active_idx = entry["controller"]
    entry["resolve"](state, entry["card_def"])


def _trigger_resolve(entry):
    """Builds the stack entry's own resolve(state, card_def) function for
    one queued trigger (docs/PRIORITY_PLAN.md item 1) -- deferred until
    THIS SPECIFIC stack entry actually resolves, instead of running the
    instant the trigger was queued (real Magic: triggered abilities go on
    the stack and can be responded to, same as a cast spell).

    "automatic" (Sneaky Snacker's on-draw-count return): runs the exact
    effect this engine always ran immediately before this plan.
    "decision" (Madness): now OPENS the cast-or-decline choice only here,
    matching real Magic's "you may cast it as this ability resolves"
    wording -- not the instant the card was discarded, which is what lets
    an opponent get a real priority window (a chance to respond to the
    trigger itself) before the decision is even offered. Its own
    on_complete is a no-op: the old recursive "keep draining" continuation
    isn't needed anymore -- promote_triggers_to_stack (below) is called
    fresh at the start of every priority round, so anything left queued
    (or queued anew) is picked up there instead of needing to be chained
    through by hand."""
    if entry["type"] == "automatic":
        if entry["kind"] == "on_draw_count":
            def resolve(state, card_def):
                state.graveyard.remove(card_def)
                enters_battlefield(state, card_def, force_tapped=True)
            return resolve
        raise ValueError(f"unknown automatic trigger queue entry: {entry}")
    if entry["type"] == "decision":
        if entry["kind"] == "madness":
            def resolve(state, card_def):
                resolution.begin_madness_decision(state, card_def, on_complete=lambda s: None)
            return resolve
        raise ValueError(f"unknown trigger queue entry: {entry}")
    raise ValueError(f"unknown trigger queue entry: {entry}")


def promote_triggers_to_stack(state):
    """Moves every currently-queued trigger for the active player onto
    state.stack (docs/PRIORITY_PLAN.md item 1), replacing the old
    drain_trigger_queue (which ran each entry's own effect immediately
    instead of deferring it onto the stack -- see _trigger_resolve for
    what changed per trigger kind). Called once per priority round, right
    before anyone would receive priority (game.turn's own priority round),
    matching real Magic's actual ordering (704.3: state-based actions,
    then triggers move to the stack, THEN priority is given).

    Only ever looks at state.trigger_queue (the ACTIVE player's own,
    active-player-proxied) -- callers always invoke this with
    state.active_idx == state.turn_player_idx (priority always resets
    there before this runs), and nothing in the current card pool ever
    queues a trigger for a non-active player (only the active player's
    own draw()/discard() ever populate trigger_queue), so real Magic's own
    APNAP ordering (whose triggers get placed first, when different
    players have simultaneous ones) is moot given what this engine can
    actually produce today -- revisit if a future card changes that.

    2+ queued at once: the active player picks the placement order
    (resolution.begin_order_triggers) -- real Magic's own rule (603.3b),
    not a fixed queue order (a real deck can hit this: Faithless
    Looting's discard-2 landing on two Madness cards at once, or two
    Sneaky Snackers both crossing their own draw-count trigger on the
    same draw). 0 or 1: pushed immediately, no ordering decision needed.
    No-op if the queue is empty -- safe to call unconditionally at the
    start of every priority round."""
    if not state.trigger_queue:
        return
    stack_entries = [{"card_def": entry["card_def"], "resolve": _trigger_resolve(entry)} for entry in state.trigger_queue]
    state.trigger_queue.clear()
    if len(stack_entries) == 1:
        entry = stack_entries[0]
        push_to_stack(state, entry["card_def"], entry["resolve"])
        return
    resolution.begin_order_triggers(state, stack_entries, on_complete=lambda s: None)


def execute_madness_cast(state):
    """Model chose "cast" for a pending madness_decision (itself now only
    ever offered from inside the madness trigger's own stack resolve, see
    _trigger_resolve -- docs/PRIORITY_PLAN.md item 1): pay the card's
    madness cost, then push its effect onto the stack (see push_to_stack)
    instead of resolving it immediately -- a real, independent stack entry
    that gets its own priority round before it resolves, same as any
    other cast. Then calls the enclosing madness_decision's own
    on_complete, a no-op today (see _trigger_resolve's own docstring for
    why the old recursive "keep draining" continuation isn't needed
    anymore).

    Captures card_def/on_complete from the CURRENT pending_resolution
    before begin_pay_cost overwrites it with its own "pay_cost" one --
    same nested-callback shape flashback_dread_return
    (game.catalog.black_cards) already uses for its own multi-step
    chain."""
    pending = state.pending_resolution
    card_def = pending["card_def"]
    outer_on_complete = pending["on_complete"]
    madness_spec = registry.EFFECT_REGISTRY[card_def.effect_id]["madness"]
    resolution._remove_one_from_exile(state, card_def)

    def _after_pay(s):
        push_to_stack(s, card_def, madness_spec["resolve"])
        outer_on_complete(s)

    mana.begin_pay_cost(state, madness_spec["cost"], on_complete=_after_pay)


def plot_to_exile(state, card_def):
    """Plot's own resolve shape (MADNESS_DECKS_PLAN.md item 4): pay the
    plot cost, move hand -> exile with this turn's stamp instead of
    running the card's real effect. Generic Plot-mechanic plumbing (any
    future "plot" card reuses this unchanged, same precedent as
    execute_madness_cast above) -- currently only Highway Robbery
    (game.catalog.red_cards) has a "plot" registry spec."""
    state.hand.remove(card_def)
    state.exile.append((card_def, state.turn_number))


def bounce_land_etb(state):
    """ETB: return a land you control to hand (Rakdos Carnarium --
    docs/MADNESS_DECKS_PLAN.md item 10). resolution.begin_choose_permanent
    already covers "pick one of my own permanents matching a predicate,
    by name" exactly -- no new resolution kind needed. enters_battlefield
    appends the permanent before running its ETB trigger, so the land
    that just entered is itself already a legal target here, matching the
    real-rules guarantee "always at least one target" for free. Generic
    enough to share if a second land-bounce card ever needs it -- not
    Rakdos-Carnarium-specific despite currently having one caller."""
    def _on_chosen(state, name):
        if name is None:
            return  # begin_choose_permanent's own empty-battlefield safety net -- never reachable in practice, since this card is itself a legal land target the moment it's on the battlefield
        permanent = next(p for p in state.battlefield if p.card_def.name == name and p.card_def.card_type == CardType.LAND)
        state.battlefield.remove(permanent)
        state.hand.append(permanent.card_def)

    resolution.begin_choose_permanent(state, lambda p: p.card_def.card_type == CardType.LAND, _on_chosen)


TOKEN_LIMIT = 20  # shared across every token name, not per-name -- see docs/COMBAT_PLAN.md


def create_token(state, card_def, tapped=False):
    """A token permanent, not backed by any library/CARD_DEFS card --
    Melded Moxite's Robot, Voldaren Epicure/Vampire's Kiss's Blood
    (docs/MADNESS_DECKS_PLAN.md item 8). Reuses enters_battlefield's full
    battlefield-entry path (ETB dispatch, terminated_fn check) unchanged
    -- a token entering the battlefield is exactly as real as any other
    permanent from here on; only its creation (no hand/library removal
    beforehand) is different. tapped=True covers "Create a TAPPED 2/2
    Robot" (Melded Moxite's own wording); Blood tokens enter untapped.

    TOKEN_LIMIT caps how many tokens (any name, combined -- an Eldrazi
    Spawn and a Warrior count the same toward this one shared pool) this
    player can have on the battlefield at once (docs/COMBAT_PLAN.md's
    permanent-identity discussion: no per-card token-production math,
    just one flat, generous ceiling). Beyond it, creation fails outright
    -- returns None, never touches the battlefield, never fires an ETB
    trigger, as if it was never attempted at all. No real deck comes
    remotely close today; this exists for whatever degenerate future
    token engine might."""
    token_count = sum(1 for p in state.battlefield if p.card_def.name not in registry.CARD_DEFS)
    if token_count >= TOKEN_LIMIT:
        return None
    return enters_battlefield(state, card_def, force_tapped=tapped)


def activate_blood_sac(state, permanent):
    """Blood's {1}, {T}, Discard a card, Sacrifice this token: Draw a
    card. The {1} mana and the untapped precondition are both already
    handled generically by drl_env.py's cost_key-based activated-ability
    wiring (same as Candy Trail's own sac ability, which has no {T}
    symbol at all yet gets the identical untapped check for free today)
    -- this only covers what's specific to Blood: sacrifice (a token
    ceases to exist once it leaves the battlefield, per real Magic's own
    state-based action -- never added to the graveyard, unlike a real
    card), then discard a card (reusing item 1's begin_discard directly,
    which is what makes Madness-awareness automatic for whatever gets
    discarded this way), then draw."""
    state.battlefield.remove(permanent)
    resolution.begin_discard(state, 1, optional=False, on_complete=lambda s, _cards: s.draw(1))


def activate_eldrazi_spawn_sac(state, permanent):
    """Malevolent Rumble's Eldrazi Spawn token: "Sacrifice this creature:
    Add {C}." No {T} in the real cost -- unlike every other mana source
    in this engine, this doesn't tap (so summoning sickness never gates
    it) and isn't offered through mana.py's tap-based machinery at all.
    Modeled as a standalone no-mana-cost activated ability (same shape
    Quirion Ranger's Forest-bounce already uses) whose only effect is
    floating {C} directly into the mana pool -- reusing state.mana_pool's
    existing "produced now, spent later via a separate action" mechanism
    unchanged, since a sacrifice isn't a tap this engine's interactive
    pay_cost loop has any other way to represent."""
    state.battlefield.remove(permanent)
    state.mana_pool["C"] = state.mana_pool.get("C", 0) + 1


BLOOD_TOKEN_CARD_DEF = CardDef("Blood", CardType.ARTIFACT, None, EffectId.BLOOD_TOKEN, sac_ability_cost={"generic": 1})
ROBOT_TOKEN_CARD_DEF = CardDef("Robot", CardType.CREATURE, None, EffectId.ROBOT_TOKEN, power=2, toughness=2)  # 2/2
WARRIOR_TOKEN_CARD_DEF = CardDef("Warrior", CardType.CREATURE, None, EffectId.WARRIOR_TOKEN, power=1, toughness=1)  # 1/1; vigilance -- see EffectId.WARRIOR_TOKEN's own registry entry (white_cards.py)
ELDRAZI_SPAWN_TOKEN_CARD_DEF = CardDef("Eldrazi Spawn", CardType.CREATURE, None, EffectId.ELDRAZI_SPAWN_TOKEN, power=0, toughness=1)  # 0/1


if __name__ == "__main__":
    # ponytail self-check: no pytest in this project, mirrors the
    # assert-based demo convention -- run via `python -m game.effects_common`
    # from src/. Exercises the full Madness chain end to end (discard ->
    # exile + queue -> drain -> decision -> pay madness cost -> resolve ->
    # drain again) against a hand-built state. No real madness card exists
    # yet (deck assembly is out of scope for this plan), so this borrows
    # EffectId.FILLER for the fake spell, saving/restoring its real (empty)
    # registry entry around the check; the mana source is a genuine Forest
    # CardDef (EffectId.FOREST already has a real "mana" spec, no faking
    # needed for that half).
    from .cards import CardDef, CardType
    from .state import GameState, Permanent

    resolved_calls = []

    def _fake_resolve(s, c):
        resolved_calls.append(c.name)

    _filler_backup = registry.EFFECT_REGISTRY[EffectId.FILLER]
    registry.EFFECT_REGISTRY[EffectId.FILLER] = {"madness": {"cost": {"G": 1}, "resolve": _fake_resolve}}
    try:
        madness_card = CardDef("Fake Madness Spell", CardType.INSTANT, {"generic": 1, "R": 1}, EffectId.FILLER)
        state = GameState(on_the_play=True)
        state.hand = [madness_card]
        state.battlefield = [Permanent(CardDef("Forest", CardType.LAND, None, EffectId.FOREST))]

        completed = []
        resolution.begin_discard(state, 1, optional=False, on_complete=lambda s, cards: completed.append(cards))
        resolution.execute_discard_option(state, "Fake Madness Spell")
        assert completed == [[madness_card]]
        assert state.pending_resolution is None  # discard's own resolution is done; nothing queued mid-discard
        assert len(state.trigger_queue) == 1 and state.trigger_queue[0]["kind"] == "madness"

        # Top-level orchestration point (game.turn's own priority round in
        # real play, once the enclosing top-level action is fully done):
        # promote. The trigger itself is now a real stack entry -- an
        # opponent gets a priority window before the cast-or-decline
        # choice is even offered (docs/PRIORITY_PLAN.md item 1) -- so
        # resolving it is what actually opens the decision, not promotion
        # itself.
        promote_triggers_to_stack(state)
        assert state.pending_resolution is None  # just sitting on the stack, not open yet
        assert len(state.stack) == 1 and state.stack[0]["card_def"] is madness_card
        resolve_top_of_stack(state)
        assert state.pending_resolution["kind"] == "madness_decision"
        assert resolution.madness_decision_options(state) == ["cast", "decline"]

        execute_madness_cast(state)
        # begin_pay_cost's own resolution is now pending (paying {G}) --
        # confirms this correctly nests through it via the captured
        # outer_on_complete, instead of crashing on a stale
        # pending_resolution reference once pay_cost's own resolution
        # clears it.
        assert state.pending_resolution["kind"] == "pay_cost"
        assert mana.tap_cost_options(state) == [("Forest", None, False)]
        mana.execute_tap_cost_option(state, "Forest", None, False)
        # Pool-only model (MANA_POOL_PLAN.md): the tap only floats {G} into
        # the pool -- paying the cost is a separate, explicit spend.
        assert state.pending_resolution["kind"] == "pay_cost"
        mana.execute_pool_spend(state, "G")

        # Payment complete -> pushed to the stack (not resolved yet, a
        # real independent stack entry -- see execute_madness_cast's own
        # docstring) -> the enclosing madness_decision's own on_complete
        # fires (a no-op today) -> fully back to no pending resolution.
        # The effect itself only fires once something actually resolves
        # the stack (a "Pass" in real play).
        assert resolved_calls == []
        assert len(state.stack) == 1 and state.stack[0]["card_def"] is madness_card
        assert state.pending_resolution is None
        assert state.trigger_queue == []
        resolve_top_of_stack(state)
        assert resolved_calls == ["Fake Madness Spell"]
        assert state.stack == []
        assert state.exile == []
    finally:
        registry.EFFECT_REGISTRY[EffectId.FILLER] = _filler_backup

    print("effects_common.py madness self-check: OK")

    # Per-turn draw counter + Sneaky Snacker-style automatic return
    # (MADNESS_DECKS_PLAN.md item 7). No real Sneaky Snacker card exists
    # yet, so this borrows EffectId.FILLER again for an "on_draw_count"
    # spec. Two copies sit in the graveyard, to confirm the faithful
    # multi-copy handling (each queues its own return, not capped at one).
    _filler_backup = registry.EFFECT_REGISTRY[EffectId.FILLER]
    registry.EFFECT_REGISTRY[EffectId.FILLER] = {"on_draw_count": {"count": 3}}
    try:
        snacker = CardDef("Fake Snacker", CardType.CREATURE, {"generic": 1}, EffectId.FILLER)
        state = GameState(on_the_play=True)
        state.library = [CardDef(f"Filler {i}", CardType.SORCERY, {}, None) for i in range(5)]
        state.graveyard = [snacker, snacker]  # two physical copies

        state.draw(1)
        assert state.cards_drawn_this_turn == 1 and state.trigger_queue == []
        state.draw(1)
        assert state.cards_drawn_this_turn == 2 and state.trigger_queue == []
        state.draw(1)  # the third card this turn -- both copies trigger
        assert state.cards_drawn_this_turn == 3
        assert len(state.trigger_queue) == 2
        assert all(e == {"type": "automatic", "kind": "on_draw_count", "card_def": snacker} for e in state.trigger_queue)

        state.draw(1)  # a 4th draw must NOT re-trigger (exactly == 3, not >= 3)
        assert len(state.trigger_queue) == 2

        # 2 simultaneous triggers -- a real placement-order choice
        # (docs/PRIORITY_PLAN.md item 1), not fixed queue order. Both
        # share a name (two physical copies of the same card): by-name
        # fungible, same convention discard/sacrifice already use for
        # interchangeable choices among a player's own copies.
        promote_triggers_to_stack(state)
        assert state.pending_resolution["kind"] == "order_triggers"
        assert resolution.order_triggers_options(state) == ["Fake Snacker"]
        resolution.execute_order_triggers_option(state, "Fake Snacker")
        assert state.pending_resolution["kind"] == "order_triggers"  # one more still to place
        resolution.execute_order_triggers_option(state, "Fake Snacker")
        assert state.pending_resolution is None
        assert len(state.stack) == 2
        assert state.trigger_queue == []

        # No decision at any point once each stack entry resolves -- both
        # copies return to the battlefield tapped.
        while state.stack:
            resolve_top_of_stack(state)
        assert state.pending_resolution is None
        assert state.graveyard == []
        assert len(state.battlefield) == 2
        assert all(p.card_def is snacker and p.tapped for p in state.battlefield)
    finally:
        registry.EFFECT_REGISTRY[EffectId.FILLER] = _filler_backup

    print("effects_common.py draw-counter self-check: OK")

    # Tokens (MADNESS_DECKS_PLAN.md item 8): create_token reuses
    # enters_battlefield unchanged; Blood's sac ability reuses
    # begin_discard directly (so a discarded Madness card is still
    # correctly caught -- exercised here via the same FILLER-as-madness-
    # card trick used above).
    state = GameState(on_the_play=True)
    create_token(state, ROBOT_TOKEN_CARD_DEF, tapped=True)
    create_token(state, BLOOD_TOKEN_CARD_DEF)  # untapped by default
    assert [(p.card_def.name, p.tapped) for p in state.battlefield] == [("Robot", True), ("Blood", False)]

    _filler_backup = registry.EFFECT_REGISTRY[EffectId.FILLER]
    registry.EFFECT_REGISTRY[EffectId.FILLER] = {"madness": {"cost": {}, "resolve": lambda s, c: None}}
    try:
        blood = next(p for p in state.battlefield if p.card_def.name == "Blood")
        other_card = CardDef("Fake Madness Card", CardType.SORCERY, {}, EffectId.FILLER)
        state.hand = [other_card]
        state.library = [CardDef("Library Card", CardType.SORCERY, {}, None)]

        drawn_before = len(state.hand)
        activate_blood_sac(state, blood)  # cost payment ({1} mana) is drl_env.py's concern, not this function's -- see build_action_table's token_card_defs
        assert state.pending_resolution["kind"] == "discard"
        assert resolution.discard_options(state) == ["Fake Madness Card"]
        resolution.execute_discard_option(state, "Fake Madness Card")

        # Sacrificed: gone, and never added to any zone (a token ceases
        # to exist, unlike a real card being discarded/sacrificed).
        assert [p.card_def.name for p in state.battlefield] == ["Robot"]
        # The discarded card had a "madness" spec -- queued, not graveyard.
        assert state.graveyard == []
        assert len(state.trigger_queue) == 1 and state.trigger_queue[0]["kind"] == "madness"
        promote_triggers_to_stack(state)
        assert state.pending_resolution is None and len(state.stack) == 1  # just sitting on the stack
        resolve_top_of_stack(state)  # resolving the trigger is what opens the cast-or-decline choice
        assert state.pending_resolution["kind"] == "madness_decision"
        resolution.execute_madness_decline(state)  # free madness cost isn't the point of this check -- just confirm routing
        assert [c.name for c in state.graveyard] == ["Fake Madness Card"]
        # Then the draw fires (begin_discard's on_complete): net hand size
        # unchanged (lost the discarded card, gained one drawn).
        assert len(state.hand) == drawn_before  # started at 1 (other_card), discarded it, drew 1 -- still 1
    finally:
        registry.EFFECT_REGISTRY[EffectId.FILLER] = _filler_backup

    print("effects_common.py tokens self-check: OK")

    # TOKEN_LIMIT (docs/COMBAT_PLAN.md): a shared pool across every token
    # name, not per-name -- 19 Robots already in play leaves room for
    # exactly 1 more token of ANY kind (a Warrior here), then nothing at
    # all, not even a different name.
    state = GameState(on_the_play=True)
    state.battlefield = [Permanent(ROBOT_TOKEN_CARD_DEF) for _ in range(19)]
    warrior = create_token(state, WARRIOR_TOKEN_CARD_DEF)
    assert warrior is not None and warrior in state.battlefield
    assert len(state.battlefield) == 20
    overflow = create_token(state, ELDRAZI_SPAWN_TOKEN_CARD_DEF)
    assert overflow is None
    assert len(state.battlefield) == 20  # never added -- not even a phantom entry
    assert not any(p.card_def.name == "Eldrazi Spawn" for p in state.battlefield)

    # "Fails outright" also means no ETB trigger fires for the rejected
    # token -- exercised via a synthetic FILLER-backed token whose
    # etb_trigger would be observable if it ran (none of the 4 real
    # tokens happen to have one to test this against directly).
    etb_calls = []
    _filler_backup = registry.EFFECT_REGISTRY[EffectId.FILLER]
    registry.EFFECT_REGISTRY[EffectId.FILLER] = {"etb_trigger": lambda s: etb_calls.append(True)}
    try:
        fake_token = CardDef("Fake Token", CardType.CREATURE, None, EffectId.FILLER)
        result = create_token(state, fake_token)
        assert result is None
        assert etb_calls == []  # never fired -- the creation never happened at all
    finally:
        registry.EFFECT_REGISTRY[EffectId.FILLER] = _filler_backup

    print("effects_common.py TOKEN_LIMIT self-check: OK")

    # Land bounce (item 10): no real Rakdos Carnarium card exists yet, so
    # this borrows EffectId.FILLER's etb_trigger to exercise
    # bounce_land_etb directly through the real enters_battlefield path
    # (same as any other ETB trigger). Two other lands already in play,
    # to confirm the model has a real choice, not just "the card itself."
    _filler_backup = registry.EFFECT_REGISTRY[EffectId.FILLER]
    registry.EFFECT_REGISTRY[EffectId.FILLER] = {"etb_trigger": bounce_land_etb}
    try:
        carnarium = CardDef("Fake Carnarium", CardType.LAND, None, EffectId.FILLER)
        state = GameState(on_the_play=True)
        state.hand = [carnarium]
        state.battlefield = [
            Permanent(CardDef("Forest", CardType.LAND, None, EffectId.FOREST)),
            Permanent(CardDef("Swamp", CardType.LAND, None, EffectId.SWAMP)),
        ]

        state.hand.remove(carnarium)
        enters_battlefield(state, carnarium)  # normal ETB path, exactly like play_land_from_hand would drive it
        assert state.pending_resolution["kind"] == "choose_permanent"
        # All three lands are legal targets, including itself -- it's
        # already on the battlefield by the time its own trigger runs.
        assert resolution.choose_permanent_options(state) == ["Fake Carnarium", "Forest", "Swamp"]

        resolution.execute_choose_permanent_option(state, "Swamp")
        assert state.pending_resolution is None
        assert sorted(p.card_def.name for p in state.battlefield) == ["Fake Carnarium", "Forest"]
        assert [c.name for c in state.hand] == ["Swamp"]

        # Bouncing itself is also legal (real-rules guarantee: always at
        # least one target, satisfied even with only 1 land in play).
        state = GameState(on_the_play=True)
        state.battlefield = [Permanent(CardDef("Fake Carnarium 2", CardType.LAND, None, EffectId.FILLER))]
        bounce_land_etb(state)  # exercised directly this time, not via enters_battlefield -- same underlying call
        resolution.execute_choose_permanent_option(state, "Fake Carnarium 2")
        assert state.battlefield == []
        assert [c.name for c in state.hand] == ["Fake Carnarium 2"]
    finally:
        registry.EFFECT_REGISTRY[EffectId.FILLER] = _filler_backup

    print("effects_common.py land-bounce self-check: OK")

    # Combat (rakdos madness / mono red madness only -- game.turn.Phase's
    # combat_enabled param gates whether DECLARE_ATTACKERS/COMBAT_DAMAGE
    # ever actually run; this exercises declare_attackers_step (now just
    # the phase-entry reset) + creature_attack_eligible + declare_attacker
    # (drl_env's "Attack: <name>" actions call these one creature at a
    # time) + combat_damage_step directly). Five permanents, each proving
    # one eligibility rule.
    state = GameState(on_the_play=True, terminated_fn=lambda s: s.damage_dealt >= 5)
    attacker = Permanent(CardDef("Attacker", CardType.CREATURE, None, EffectId.FILLER, power=3))
    attacker.summoning_sick = False
    sick = Permanent(CardDef("Sick", CardType.CREATURE, None, EffectId.FILLER, power=10))  # summoning_sick=True by construction -- never cleared here (that's untap_step's job)
    already_tapped = Permanent(CardDef("Tapped Out", CardType.CREATURE, None, EffectId.FILLER, power=10), tapped=True)
    already_tapped.summoning_sick = False
    vanilla = Permanent(CardDef("No Stats", CardType.CREATURE, None, EffectId.FILLER))  # no "power" key at all -- untracked-stats precedent (Masked Vandal, Mesmeric Fiend)
    vanilla.summoning_sick = False
    not_a_creature = Permanent(CardDef("Some Land", CardType.LAND, None, EffectId.FILLER, power=10))
    not_a_creature.summoning_sick = False
    defender = Permanent(CardDef("Turtle Wall", CardType.CREATURE, None, EffectId.FILLER, power=10, defender=True))
    defender.summoning_sick = False
    state.battlefield = [attacker, sick, already_tapped, vanilla, not_a_creature, defender]

    declare_attackers_step(state)
    assert state.attackers == []  # phase-entry reset, no auto-population anymore
    assert creature_attack_eligible(state, attacker)
    assert creature_attack_eligible(state, vanilla)  # 0 power still eligible, same as a real 0-power creature
    assert not creature_attack_eligible(state, sick)
    assert not creature_attack_eligible(state, already_tapped)
    assert not creature_attack_eligible(state, not_a_creature)
    # Defender: untapped, not summoning sick, real power -- every other
    # eligibility rule satisfied -- but can never attack, full stop.
    assert not creature_attack_eligible(state, defender)

    # Partial declaration: attack with Attacker only, deliberately leaving
    # the also-eligible vanilla creature back -- a real per-creature
    # choice now, not the old "every eligible creature auto-attacks" rule.
    declare_attacker(state, attacker)
    assert attacker.tapped and attacker in state.attackers
    assert not vanilla.tapped and vanilla not in state.attackers
    combat_damage_step(state)
    assert state.damage_dealt == 3  # only Attacker's power counts
    assert state.attackers == []  # cleared once damage resolves
    assert state.turn_won is None  # 3 < 5, not lethal yet

    declare_attackers_step(state)  # called again with no intervening untap_step -- Attacker is still tapped from last round, so no longer eligible; nothing gets declared this round
    assert not creature_attack_eligible(state, attacker)
    combat_damage_step(state)
    assert state.damage_dealt == 3

    attacker.tapped = False  # simulate the next turn's untap_step (sick's flag deliberately left alone -- it's still sick until a real untap_step clears it)
    declare_attackers_step(state)
    declare_attacker(state, attacker)
    combat_damage_step(state)
    assert state.damage_dealt == 6  # crosses the >=5 threshold
    assert state.turn_won == 0  # terminated_fn caught it here directly, same check enters_battlefield uses -- state.turn_number was never advanced in this synthetic test

    # Haste (Kitchen Imp): a "haste": True registry spec lets a summoning-
    # sick creature be attack-eligible anyway -- the only place that spec
    # is ever read.
    state = GameState(on_the_play=True)
    _filler_backup = registry.EFFECT_REGISTRY[EffectId.FILLER]
    registry.EFFECT_REGISTRY[EffectId.FILLER] = {"haste": True}
    try:
        hasty = Permanent(CardDef("Hasty", CardType.CREATURE, None, EffectId.FILLER, power=2))
        assert hasty.summoning_sick  # True by construction -- just entered, never untapped
        state.battlefield = [hasty]
        declare_attackers_step(state)
        assert creature_attack_eligible(state, hasty)  # eligible despite being summoning sick
        declare_attacker(state, hasty)
        combat_damage_step(state)
        assert state.damage_dealt == 2 and hasty.tapped  # attacked anyway
    finally:
        registry.EFFECT_REGISTRY[EffectId.FILLER] = _filler_backup

    print("effects_common.py combat self-check: OK")

    # Blocking's own mutual combat damage + creature death (docs/COMBAT_
    # PLAN.md steps 5/6): a blocked attacker deals no damage to the
    # opponent (absorbed), but IT and its blocker now fight each other for
    # real -- each takes the other's power as damage_marked, and
    # check_state_based_actions kills whichever side(s) that's lethal for.
    # Needs a genuine 2-player GameState (not the single-player one every
    # earlier check here uses): a dying BLOCKER's own zones belong to the
    # DEFENDER, not whichever side state.active_idx (the attacker,
    # throughout combat_damage_step) currently proxies to -- exactly the
    # bug _destroy_creature's own docstring explains. Two blocked pairs in
    # ONE combat: pair A's attacker dies, its blocker survives; pair B's
    # blocker dies, its attacker survives -- proving the death check
    # applies independently per creature, not "whoever's weaker overall."
    from .state import PlayerState

    # _destroy_creature distinguishes a real card (graveyard-bound) from a
    # token (ceases to exist, see the token-death check below) by
    # registry.CARD_DEFS membership -- so these synthetic FILLER creatures
    # need real, if temporary, CARD_DEFS entries of their own to exercise
    # the "real card" path, same "Fake Plot Spell"-style save/restore
    # convention this file's own Plot-adjacent checks already use.
    fake_names = ["Attacker A", "Attacker B", "Unblocked", "Blocker A", "Blocker B"]
    _card_defs_backup = dict(registry.CARD_DEFS)
    try:
        state = GameState(on_the_play=True, players=[PlayerState(True), PlayerState(False)])
        attacker_a = Permanent(CardDef("Attacker A", CardType.CREATURE, None, EffectId.FILLER, power=1, toughness=3))
        attacker_a.summoning_sick = False
        attacker_b = Permanent(CardDef("Attacker B", CardType.CREATURE, None, EffectId.FILLER, power=5, toughness=3))
        attacker_b.summoning_sick = False
        unblocked_attacker = Permanent(CardDef("Unblocked", CardType.CREATURE, None, EffectId.FILLER, power=2, toughness=1))
        unblocked_attacker.summoning_sick = False
        blocker_a = Permanent(CardDef("Blocker A", CardType.CREATURE, None, EffectId.FILLER, power=5, toughness=2))
        blocker_b = Permanent(CardDef("Blocker B", CardType.CREATURE, None, EffectId.FILLER, power=1, toughness=2))
        for p in (attacker_a, attacker_b, unblocked_attacker, blocker_a, blocker_b):
            registry.CARD_DEFS[p.card_def.name] = p.card_def
        state.players[0].battlefield = [attacker_a, attacker_b, unblocked_attacker]
        state.players[1].battlefield = [blocker_a, blocker_b]

        declare_attackers_step(state)
        declare_attacker(state, attacker_a)
        declare_attacker(state, attacker_b)
        declare_attacker(state, unblocked_attacker)
        state.blocked_by[attacker_a] = blocker_a
        state.blocked_by[attacker_b] = blocker_b
        combat_damage_step(state)

        assert state.damage_dealt == 2  # only the unblocked attacker's power counts -- A/B's damage is absorbed, not dealt to the opponent
        assert state.attackers == []  # cleared same as always, regardless of blocked/unblocked/dead
        # Pair A: attacker_a (toughness 3) took blocker_a's 5 power -> dead.
        # blocker_a (toughness 2) took attacker_a's 1 power -> survives.
        assert attacker_a not in state.players[0].battlefield and blocker_a in state.players[1].battlefield
        assert [c.name for c in state.players[0].graveyard] == ["Attacker A"]  # landed in ITS OWN owner's graveyard, not the active player's by coincidence (both happen to be the same side here, but _destroy_creature doesn't know that)
        assert blocker_a.damage_marked == 1 and not blocker_a.tapped  # damage marked, but very much alive
        # Pair B: attacker_b (toughness 3) took blocker_b's 1 power -> survives.
        # blocker_b (toughness 2) took attacker_b's 5 power -> dead, and its
        # zones are the DEFENDER's, proving _destroy_creature found the right
        # owner rather than assuming state.active_idx (still the attacker here).
        assert attacker_b in state.players[0].battlefield and blocker_b not in state.players[1].battlefield
        assert [c.name for c in state.players[1].graveyard] == ["Blocker B"]
        assert attacker_b.damage_marked == 1 and attacker_b.tapped  # still tapped (attacked), just alive

        # A fresh combat (next turn's declare_attackers_step) resets blocked_by
        # too, not just attackers -- a creature blocked LAST combat isn't still
        # considered blocked this time.
        declare_attackers_step(state)
        assert state.blocked_by == {}

        # cleanup_step clears damage_marked for EVERY permanent, BOTH players --
        # not just the active player's own side (real Magic: cleanup is global,
        # not per-player), and not gated on who's currently active_idx. Neither
        # hand exceeds HAND_SIZE_LIMIT here, so its own discard auto-completes
        # with nothing pending (begin_discard's own n<=0 short-circuit).
        assert blocker_a.damage_marked == 1  # still set from the combat above
        cleanup_step(state)
        assert state.pending_resolution is None
        assert blocker_a.damage_marked == 0
        assert attacker_b.damage_marked == 0
    finally:
        registry.CARD_DEFS.clear()
        registry.CARD_DEFS.update(_card_defs_backup)

    print("effects_common.py blocking mutual-damage + creature-death self-check: OK")

    # Cross-player Aura reads (a real bug found while building mutual
    # combat damage): permanent_power/permanent_toughness/enchantment_count
    # used to read state.battlefield (active-player-proxied) to find
    # enchanting Auras -- silently wrong for a BLOCKER (the defender's own
    # creature), since combat_damage_step always runs with active_idx on
    # the ATTACKER. Confirmed fixed: reading the DEFENDER's Aura'd
    # creature's power/toughness while active_idx is the ATTACKER's still
    # finds every bonus, including Ancestral Mask's own enchantment_count
    # (which needs the DEFENDER's enchantment count specifically, not
    # whichever player is currently active).
    state = GameState(on_the_play=True, players=[PlayerState(True), PlayerState(False)])
    defenders_creature = Permanent(CardDef("Defender's Creature", CardType.CREATURE, None, EffectId.FILLER, power=1, toughness=1))
    rancor_on_defender = Permanent(CardDef("Rancor", CardType.ENCHANTMENT, {"G": 1}, EffectId.RANCOR))
    rancor_on_defender.flags["enchanting"] = defenders_creature
    mask_on_defender = Permanent(CardDef("Ancestral Mask", CardType.ENCHANTMENT, {"generic": 2, "G": 1}, EffectId.ANCESTRAL_MASK))
    mask_on_defender.flags["enchanting"] = defenders_creature
    state.players[1].battlefield = [defenders_creature, rancor_on_defender, mask_on_defender]
    state.active_idx = 0  # the ATTACKER's own perspective -- defender's battlefield is NOT state.battlefield right now

    # power: 1 base + 2 (Rancor) + 2 (Ancestral Mask, 1 OTHER enchantment
    # -- Rancor -- so +2, not +4). toughness: 1 base + 0 (Rancor, power
    # only) + 2 (Ancestral Mask, symmetric).
    assert permanent_power(state, defenders_creature) == 5
    assert permanent_toughness(state, defenders_creature) == 3

    print("effects_common.py cross-player Aura-read self-check: OK")

    # Aura orphaning (docs/COMBAT_PLAN.md step 6): a dying creature's own
    # attached Aura(s) leave the battlefield with it -- Rancor specifically
    # returns to ITS CONTROLLER's hand (returns_to_hand_when_orphaned,
    # green_cards.py), every other Aura to the graveyard (real Magic's
    # default, no special registry flag needed). Uses check_state_based_
    # actions directly (skipping the mutual-damage arithmetic above,
    # already covered) -- just marks lethal damage by hand and confirms
    # the routing.
    rancor_def = CardDef("Rancor", CardType.ENCHANTMENT, {"G": 1}, EffectId.RANCOR)
    ancestral_mask_def = CardDef("Ancestral Mask", CardType.ENCHANTMENT, {"generic": 2, "G": 1}, EffectId.ANCESTRAL_MASK)

    # Same temporary-CARD_DEFS-registration need as the block above --
    # "Rancor'd Attacker"/"Masked Blocker" are synthetic names, and
    # _destroy_creature's real-card-vs-token check would otherwise treat
    # them as tokens (never entering the graveyard at all).
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
        check_state_based_actions(state)  # scans both players' battlefields now, no candidate list needed

        assert attacker_with_rancor not in state.players[0].battlefield
        assert [c.name for c in state.players[0].graveyard] == ["Rancor'd Attacker"]
        assert rancor_permanent not in state.players[0].battlefield
        assert [c.name for c in state.players[0].hand] == ["Rancor"]  # returned to hand, not the graveyard
        assert rancor_def not in state.players[0].graveyard

        assert blocker_with_mask not in state.players[1].battlefield
        assert mask_permanent not in state.players[1].battlefield
        assert sorted(c.name for c in state.players[1].graveyard) == ["Ancestral Mask", "Masked Blocker"]  # ordinary Aura -- graveyard, not hand
        assert state.players[1].hand == []
    finally:
        registry.CARD_DEFS.clear()
        registry.CARD_DEFS.update(_card_defs_backup)

    print("effects_common.py Aura-orphaning self-check: OK")

    # Token creature death (docs/COMBAT_PLAN.md step 6): a token that dies
    # in combat ceases to exist entirely -- same real-Magic rule every
    # existing token-removal path already follows (activate_blood_sac's
    # own docstring), NOT the graveyard-goes-there-normally case above.
    # Uses the real WARRIOR_TOKEN_CARD_DEF (1/1) rather than a synthetic
    # stand-in, since this is exactly the card a real combat death for a
    # token would involve (boggles' own Malevolent Rumble/Cartouche of
    # Solidarity).
    state = GameState(on_the_play=True, players=[PlayerState(True), PlayerState(False)])
    warrior_token = Permanent(WARRIOR_TOKEN_CARD_DEF)
    state.players[0].battlefield = [warrior_token]
    warrior_token.damage_marked = 1  # lethal (toughness 1)
    check_state_based_actions(state)
    assert warrior_token not in state.players[0].battlefield
    assert state.players[0].graveyard == []  # ceased to exist -- never added to any zone

    print("effects_common.py token-death self-check: OK")

    # Vigilance (docs/COMBAT_PLAN.md step 7): attacking a vigilant
    # creature (the real EffectId.WARRIOR_TOKEN registry entry, white_
    # cards.py -- Cartouche of Solidarity's own token) never taps it,
    # unlike an ordinary attacker.
    state = GameState(on_the_play=True)
    vigilant = Permanent(WARRIOR_TOKEN_CARD_DEF)
    vigilant.summoning_sick = False
    ordinary = Permanent(CardDef("Ordinary Attacker", CardType.CREATURE, None, EffectId.FILLER, power=1, toughness=1))
    ordinary.summoning_sick = False
    state.battlefield = [vigilant, ordinary]
    declare_attackers_step(state)
    declare_attacker(state, vigilant)
    declare_attacker(state, ordinary)
    assert not vigilant.tapped and vigilant in state.attackers  # vigilance -- attacked, but never tapped
    assert ordinary.tapped and ordinary in state.attackers  # ordinary attacker -- tapped as always

    print("effects_common.py vigilance self-check: OK")

    # Trample (docs/COMBAT_PLAN.md step 7): a blocked attacker with
    # trample (the real EffectId.RANCOR registry entry) assigns only
    # enough damage to be lethal to its blocker, letting the rest spill
    # over to the DEFENDING player -- confirmed against the real Rancor
    # Aura (+2/+0), not a synthetic keyword stand-in.
    _card_defs_backup = dict(registry.CARD_DEFS)
    try:
        state = GameState(on_the_play=True, players=[PlayerState(True), PlayerState(False)])
        trampler = Permanent(CardDef("Trampler", CardType.CREATURE, None, EffectId.FILLER, power=5, toughness=3))
        trampler.summoning_sick = False
        registry.CARD_DEFS["Trampler"] = trampler.card_def
        rancor_on_trampler = Permanent(CardDef("Rancor", CardType.ENCHANTMENT, {"G": 1}, EffectId.RANCOR))
        rancor_on_trampler.flags["enchanting"] = trampler
        weak_blocker = Permanent(CardDef("Weak Blocker", CardType.CREATURE, None, EffectId.FILLER, power=1, toughness=2))
        registry.CARD_DEFS["Weak Blocker"] = weak_blocker.card_def
        state.players[0].battlefield = [trampler, rancor_on_trampler]
        state.players[1].battlefield = [weak_blocker]

        declare_attackers_step(state)
        declare_attacker(state, trampler)
        state.blocked_by[trampler] = weak_blocker
        combat_damage_step(state)

        # Effective power 7 (5 base + Rancor's +2): 2 assigned as lethal
        # (weak_blocker's own toughness), 5 tramples through to the
        # opponent's life total/damage_dealt.
        assert weak_blocker not in state.players[1].battlefield  # lethal 2 assigned -- dies
        assert state.damage_dealt == 5  # the excess, not the full 7
        assert trampler in state.players[0].battlefield and trampler.damage_marked == 1  # weak_blocker's own power hits back, not enough to matter here
    finally:
        registry.CARD_DEFS.clear()
        registry.CARD_DEFS.update(_card_defs_backup)

    print("effects_common.py trample self-check: OK")

    # First strike (docs/COMBAT_PLAN.md step 7): a blocked attacker with
    # first strike (the real EffectId.CARTOUCHE_OF_SOLIDARITY registry
    # entry) deals its damage BEFORE the blocker gets a chance to --
    # killing the blocker in the first-strike sub-step means the blocker
    # (no first strike of its own) never deals its own damage back at
    # all, even though its power alone would otherwise be lethal to the
    # attacker.
    _card_defs_backup = dict(registry.CARD_DEFS)
    try:
        state = GameState(on_the_play=True, players=[PlayerState(True), PlayerState(False)])
        fs_attacker = Permanent(CardDef("First Striker", CardType.CREATURE, None, EffectId.FILLER, power=4, toughness=1))
        fs_attacker.summoning_sick = False
        registry.CARD_DEFS["First Striker"] = fs_attacker.card_def
        cartouche_on_attacker = Permanent(CardDef("Cartouche of Solidarity", CardType.ENCHANTMENT, {"W": 1}, EffectId.CARTOUCHE_OF_SOLIDARITY))
        cartouche_on_attacker.flags["enchanting"] = fs_attacker
        lethal_blocker = Permanent(CardDef("Would-Be Killer", CardType.CREATURE, None, EffectId.FILLER, power=3, toughness=3))
        registry.CARD_DEFS["Would-Be Killer"] = lethal_blocker.card_def
        state.players[0].battlefield = [fs_attacker, cartouche_on_attacker]
        state.players[1].battlefield = [lethal_blocker]

        declare_attackers_step(state)
        declare_attacker(state, fs_attacker)
        state.blocked_by[fs_attacker] = lethal_blocker
        combat_damage_step(state)

        # Effective power 5 (4 base + Cartouche's +1) >= lethal_blocker's
        # toughness 3 -- dies in the FIRST STRIKE sub-step, before it ever
        # gets to deal its own power-3 damage (which would have exceeded
        # fs_attacker's own toughness 2 -- 1 base + Cartouche's +1).
        assert lethal_blocker not in state.players[1].battlefield
        assert fs_attacker in state.players[0].battlefield and fs_attacker.damage_marked == 0
    finally:
        registry.CARD_DEFS.clear()
        registry.CARD_DEFS.update(_card_defs_backup)

    print("effects_common.py first-strike self-check: OK")

    # Auras (boggles deck): cast_aura's choose-target flow, plus
    # permanent_power correctly summing MULTIPLE attached Auras' own
    # pt_bonus (not just the first one) -- using the real Rancor/Ancestral
    # Mask registry entries (green_cards.py), not a FILLER stand-in, since
    # both now really exist. Also exercises permanent_toughness alongside
    # permanent_power throughout -- Rancor is real-Magic +2/+0 (power
    # only, no toughness_bonus), Ancestral Mask is +2/+2 (both), proving
    # the two stats are tracked independently, not just mirrors of each
    # other (docs/COMBAT_PLAN.md's full-stats pass).
    state = GameState(on_the_play=True)
    bogle = Permanent(CardDef("Slippery Bogle", CardType.CREATURE, {"G": 1}, EffectId.SLIPPERY_BOGLE, power=1, toughness=1))
    state.battlefield = [bogle]
    assert permanent_power(state, bogle) == 1  # no Auras yet -- just its own base power
    assert permanent_toughness(state, bogle) == 1  # ditto, base toughness

    rancor = CardDef("Rancor", CardType.ENCHANTMENT, {"G": 1}, EffectId.RANCOR)
    state.hand = [rancor]
    cast_aura(state, rancor, lambda p: p.card_def.card_type == CardType.CREATURE)
    assert resolution.choose_permanent_options(state) == ["Slippery Bogle"]
    resolution.execute_choose_permanent_option(state, "Slippery Bogle")
    assert state.pending_resolution is None and state.hand == []
    rancor_permanent = next(p for p in state.battlefield if p.card_def.name == "Rancor")
    assert rancor_permanent.flags["enchanting"] is bogle
    assert permanent_power(state, bogle) == 3  # 1 base + Rancor's own +2
    assert permanent_toughness(state, bogle) == 1  # unchanged -- Rancor is +2/+0, power only

    ancestral_mask = CardDef("Ancestral Mask", CardType.ENCHANTMENT, {"generic": 2, "G": 1}, EffectId.ANCESTRAL_MASK)
    state.hand = [ancestral_mask]
    cast_aura(state, ancestral_mask, lambda p: p.card_def.card_type == CardType.CREATURE)
    resolution.execute_choose_permanent_option(state, "Slippery Bogle")
    # 1 base + 2 (Rancor) + 2 (Ancestral Mask's own "+2 per OTHER
    # enchantment" -- exactly one other, Rancor, so +2, not +4) -- proves
    # both Auras are actually summed, not just the most recently attached.
    assert permanent_power(state, bogle) == 5
    # Toughness: 1 base + 0 (Rancor, power only) + 2 (Ancestral Mask, same
    # "+2 per other enchantment" symmetric on both stats).
    assert permanent_toughness(state, bogle) == 3

    print("effects_common.py Aura self-check: OK")
