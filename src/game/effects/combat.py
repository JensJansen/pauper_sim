"""Attack/block declaration, eligibility, and combat damage. Depends on
stats.py for effective power/toughness/keywords, state_based.py for the
state-based-action check combat damage triggers, and win_check.py for the
opponent-facing damage effect (and, for "lifelink", the life-gaining one)."""

from . import stats, state_based
from .win_check import deal_damage_to_opponent, gain_life
from .. import registry
from ..cards import CardType


def creature_attack_eligible(state, permanent):
    """Untapped, not a Defender (Wall of Roots/Overgrown Battlement/Saruli
    Caretaker/Gatecreeper Vine -- real Magic's own rule: a Defender can
    never attack, full stop, regardless of tapped/summoning-sick status),
    and not summoning sick unless it has a registry "haste": True spec
    (Kitchen Imp) -- the only other place that flag is ever read, so this
    is the only place haste needs to matter. Checked per creature (drl_env's
    "Attack: <name>" actions) so a model can declare SOME eligible
    creatures as attackers and hold others back (as blockers once those
    exist, or as mana sources).

    Also excludes anything already in state.attackers -- ordinarily
    redundant with the tapped check above (declare_attacker taps its
    permanent), but vigilance (Cartouche of Solidarity's own Warrior
    token) deliberately skips that tap, so without this explicit guard a
    vigilant creature would stay "eligible" forever within the same
    combat and could be declared an attacker repeatedly, each declaration
    appending a duplicate entry to state.attackers and multiplying its
    power in combat_damage_step's unblocked-damage total. Mirrors the
    explicit, tapped-independent guard creature_block_eligible already
    has for the identical reason (blocking never taps anyone either)."""
    return (
        permanent.card_type == CardType.CREATURE and not permanent.tapped
        and not permanent.card_def.extra.get("defender", False)
        and permanent not in state.attackers
        and (not permanent.summoning_sick or registry.EFFECT_REGISTRY.get(permanent.card_def.effect_id, {}).get("haste", False))
    )


def _blocked_by_creatures(blocked_by):
    """Flatten a blocked_by dict ({attacker: [blockers]}) to the flat set of
    creatures currently committed as blockers. Gang-blocking (multiple
    blockers per one attacker) makes blocked_by's VALUES lists, so a
    membership test can no longer be `in blocked_by.values()` -- each
    blocker still blocks exactly ONE attacker (no creature blocks twice),
    so the flat union is the right 'already committed' set."""
    return {b for blockers in blocked_by.values() for b in blockers}


def can_block(state, blocker, attacker):
    """Whether `blocker` is allowed to block `attacker`. Two real evasion/blocking rules modeled:
    - An attacker that can only be blocked by flying -- real flying (Kitchen
      Imp) OR Silhana Ledgewalker's "can't be blocked except by creatures
      with flying" (which is NOT itself flying) -- may be blocked only by a
      creature with flying or reach.
    - Reach lets a non-flying creature block a flier (Bramble Wurm).
    Menace isn't a per-blocker restriction -- any creature may still commit to
    block a menace attacker -- so it plays no part here; menace's own "needs
    2+ blockers" rule is enforced separately, by menace_block_incomplete and
    enforce_menace below. Shared by creature_block_eligible and drl_env's
    per-attacker choice predicate, so the rule lives in one place."""
    attacker_needs_flying_blocker = (
        stats.has_keyword(state, attacker, "flying")
        or stats.has_keyword(state, attacker, "cant_be_blocked_except_by_flying")
    )
    if not attacker_needs_flying_blocker:
        return True
    return stats.has_keyword(state, blocker, "flying") or stats.has_keyword(state, blocker, "reach")


def creature_block_eligible(state, permanent):
    """A creature untapped, not already committed as a blocker, AND with at
    least one attacker it's actually allowed to block right now. Reads
    state.opponent.blocked_by / state.opponent.attackers, NOT the active
    (defending) player's own: this is only ever called with state.active_idx
    already flipped to the defender (game.turn._declare_blockers_gen), and
    PlayerState.blocked_by/attackers are keyed by the ATTACKING player's own
    permanents (see their own docstrings). Deliberately NOT the same
    eligibility as creature_attack_eligible: real Magic lets a Defender
    block and lets a summoning-sick creature block -- neither check belongs
    here.

    The "has at least one legal target" clause (added with gang-blocking)
    is what stops a blocker with nothing it can legally block -- every
    attacker already blocked, or the only unblocked... no: WITH gang-
    blocking an already-blocked attacker can still take more blockers, so
    the only genuine no-target case is a non-flying blocker when every
    attacker has flying -- from being offered as a legal 'Assign Blocker'.
    That action was a no-op (nested attacker-choice finds no target, auto-
    completes, re-opens blocking, blocker still uncommitted) that a
    stochastic policy could loop on until the declare-blockers action cap
    fired; removing it at the source shrinks the action space and demotes
    that cap to a pure backstop."""
    if not (permanent.card_type == CardType.CREATURE and not permanent.tapped
            and permanent not in _blocked_by_creatures(state.opponent.blocked_by)):
        return False
    return any(can_block(state, permanent, attacker) for attacker in state.opponent.attackers)


def declare_attacker(state, permanent):
    """Model chose to attack with this specific creature -- addressed by
    (name, slot) at the drl_env action-table layer, so the caller has
    already picked the exact physical copy it means, not an arbitrary
    same-named match. Tapped here, at declaration, same as real Magic --
    an attacking creature is unavailable for a mana ability etc. for the
    rest of combat, not just tapped as a side effect of dealing damage
    later -- UNLESS it has vigilance (Cartouche of Solidarity's own
    Warrior token), which is real Magic's entire point of the keyword:
    attacking doesn't tap it at all."""
    if not stats.has_keyword(state, permanent, "vigilance"):
        permanent.tapped = True
    state.attackers.append(permanent)
    state.log_event("attack_declared", attacker=(permanent.card_def.name, permanent.slot), tapped=permanent.tapped)


def declare_attackers_step(state):
    """game.turn.Phase.DECLARE_ATTACKERS phase-entry reset (rakdos
    madness / mono red madness / boggles -- gated by combat_enabled, same
    as combat always was): clears last turn's attackers AND blocks (both
    reset together -- a fresh combat has neither yet) so the model's own
    "Attack: <name> (slot k)" actions (drl_env.build_action_table, each
    one checking creature_attack_eligible and calling declare_attacker)
    start this turn's declaration fresh, one creature at a time."""
    state.attackers = []
    state.blocked_by = {}


def _is_alive(state, permanent):
    return any(permanent in player.battlefield for player in state.players)


def _attacker_deal_damage(state, attacker, blockers, amounts, opponent_amount, attacker_facts):
    """Attacker deals its combat damage across a LIST of `blockers` per the
    parallel `amounts` list (amounts[i] -> blockers[i]), plus `opponent_amount`
    trample-spilled to the defending player. state.opponent, from the
    attacker's own active perspective, IS the defender throughout
    combat_damage_step.

    "lifelink" (Armadillo Cloak's "whenever enchanted creature deals damage,
    you gain that much life" -- a TRIGGERED ability, unlike real lifelink:
    stats.lifelink_count returns however many Cloaks are attached, each an
    independent trigger for the FULL damage dealt, so two Cloaks means 2x life
    gained). The attacker's controller is always the currently ACTIVE player
    throughout combat_damage_step, so gain_life's active-player default is
    already correct; damage ACTUALLY dealt is assigned-to-blockers +
    trample-to-player.

    attacker_facts: combat_damage_step's pre-fetched {power, toughness,
    first_strike, trample, lifelink_count} dict for this attacker -- power/
    trample/lifelink_count read here are ALWAYS the attacker's own; the dict
    exists to avoid re-scanning state.players for its Auras on every blocked
    pair (profiled at ~8-10 redundant stats.py scans per pair otherwise)."""
    # `amounts` is parallel to `blockers`: amounts[i] is the damage assigned to
    # blockers[i] -- the attacking player's own free choice for a multi-blocked
    # attacker (any portion to any blocker, not forced lethal), or
    # _default_damage_assignment's lethal-in-order split for a single blocker
    # / the auto path. `opponent_amount` is the trample share spilled to the
    # defending player. Lifelink counts damage ACTUALLY dealt (to blockers +
    # to the player), which for a full-power assignment is exactly power.
    assigned = 0
    for blocker, amount in zip(blockers, amounts):
        if amount <= 0:
            continue
        blocker.damage_marked += amount
        if attacker_facts["deathtouch"]:
            blocker.flags["deathtouched"] = True  # any damage from a deathtouch source is lethal (SBA)
        assigned += amount
        state.log_event(
            "combat_damage", source=(attacker.card_def.name, attacker.slot),
            target=(blocker.card_def.name, blocker.slot), amount=amount, trample_excess_to_opponent=0,
        )
    if opponent_amount > 0:  # trample (attacker's controller assigned this share to the defending player)
        deal_damage_to_opponent(state, opponent_amount)
        state.log_event(
            "combat_damage", source=(attacker.card_def.name, attacker.slot),
            target="opponent", amount=opponent_amount, trample_excess_to_opponent=opponent_amount,
        )
    lifelink_count = attacker_facts["lifelink_count"]
    if lifelink_count:
        gain_life(state, (assigned + opponent_amount) * lifelink_count)


def _blocker_deal_damage(state, blocker, attacker, blocker_facts):
    """Blocker deals its combat damage to the attacker it's blocking --
    never tramples through to a player: trample is an attacking-creature
    keyword only, nothing in this card pool grants a blocker-side
    equivalent, and this engine doesn't model one.

    "lifelink": unlike the attacker-side case above, the blocker's
    controller is the DEFENDING player -- players[1 - active_idx]
    (== state.opponent) from the currently-active attacker's own
    perspective, not the active player (which would wrongly credit the
    attacker) -- so this passes gain_life that index explicitly instead of
    letting it default to the active player. Routing through gain_life (not
    a raw life_total bump) is what gets this lifegain into the event log
    like every other life change. Multiplied by stats.lifelink_count the
    same stacking way as the attacker-side case above (2 Cloaks on a
    blocker also trigger twice).

    blocker_facts: see _attacker_deal_damage's own docstring -- same
    pre-fetched dict, just the blocker's own this time (no attacker_facts
    needed here: a blocker never tramples, so nothing about the attacker's
    own stats is read in this direction)."""
    power = blocker_facts["power"]
    attacker.damage_marked += power
    if power > 0 and blocker_facts["deathtouch"]:
        attacker.flags["deathtouched"] = True  # any damage from a deathtouch source is lethal (SBA)
    state.log_event(
        "combat_damage", source=(blocker.card_def.name, blocker.slot), target=(attacker.card_def.name, attacker.slot),
        amount=power, trample_excess_to_opponent=0,
    )
    lifelink_count = blocker_facts["lifelink_count"]
    if lifelink_count:
        gain_life(state, power * lifelink_count, player_idx=1 - state.active_idx)


def _default_damage_assignment(attacker_facts, blockers, facts_by_id):
    """Auto split of an attacker's combat damage across its (living)
    blockers -- used for a SINGLE blocker (no choice to make) and as the
    fallback when no explicit model assignment exists. Lethal-in-order to
    maximize kills; a trampler's leftover goes to the player, a non-
    trampler's leftover piles onto the last blocker so all power lands.
    Returns (amounts parallel to
    `blockers`, opponent_amount). For a MULTI-blocked attacker the attacking
    player's OWN freely-chosen split (assign_combat_damage resolution)
    replaces this -- any portion to any blocker, non-lethal allowed."""
    remaining = attacker_facts["power"]
    amounts = [0] * len(blockers)
    for i, blocker in enumerate(blockers):
        if remaining <= 0:
            break
        lethal = max(facts_by_id[id(blocker)]["toughness"] - blocker.damage_marked, 0)
        assign = min(remaining, lethal)
        amounts[i] = assign
        remaining -= assign
    opponent_amount = 0
    if remaining > 0:
        if attacker_facts["trample"]:
            opponent_amount = remaining
        elif blockers:
            amounts[-1] += remaining
    return amounts, opponent_amount


def _damage_assignment_for(attacker, living, attacker_facts, facts_by_id):
    """The (amounts-parallel-to-living, opponent_amount) split this attacker
    deals right now: the attacking player's OWN choice if one was recorded
    (resolution.begin_assign_combat_damage stashes it on
    attacker.flags['combat_damage_split'] for a 2+-blocked attacker --
    popped here, consumed once), else the lethal-in-order default (a single
    blocker, or the model-less path). A model split keyed by blocker maps
    onto `living` (every blocker is alive when its own attacker deals, so
    this is the full assignment)."""
    split = attacker.flags.pop("combat_damage_split", None)
    if split is not None:
        amounts_by_blocker, opponent_amount = split
        return [amounts_by_blocker.get(b, 0) for b in living], opponent_amount
    return _default_damage_assignment(attacker_facts, living, facts_by_id)


def attackers_needing_damage_assignment(state):
    """The (attacker, blockers, power, has_trample) tuples for attackers
    blocked by 2+ creatures with nonzero power -- the ones whose controller
    must freely assign combat damage (resolution.begin_assign_combat_damage,
    driven by turn._assign_combat_damage_gen). A lone blocker or a 0-power
    attacker has no choice, so it's skipped and combat_damage_step auto-
    assigns via _default_damage_assignment. Lives here (not in turn.py) so
    the power/keyword lookups stay in the effects layer that owns stats."""
    out = []
    for attacker in state.attackers:
        blockers = state.blocked_by.get(attacker, [])
        if len(blockers) < 2:
            continue
        power = stats.permanent_power(state, attacker)
        if power <= 0:
            continue
        out.append((attacker, blockers, power, stats.has_keyword(state, attacker, "trample")))
    return out


def combat_damage_step(state):
    """game.turn.Phase.COMBAT_DAMAGE. Unblocked attackers (in state.attackers,
    not in state.blocked_by) deal their total power to the opponent at once; an
    untracked-stats vanilla contributes 0.

    Blocked pairs fight in up to two sub-steps (real Magic first-strike order):
    first-strikers deal, then an SBA check clears the dead (a first-strike kill
    never deals back); then the non-first-strikers deal, if both are still
    alive, and a second SBA check. With no first strike this collapses to one
    simultaneous exchange.

    lifelink on an unblocked attacker is gained here, batched, each attacker's
    power x its lifelink_count (2 Cloaks = 2x).

    Every combatant's {power, toughness, first_strike, trample, deathtouch,
    lifelink_count} is fetched ONCE up front and reused (profiled: avoids ~8-10
    redundant per-pair Aura scans). Safe because nothing here casts or
    (de)attaches an Aura mid-resolution; damage_marked is read fresh."""
    # The Initiative (Avenging Hunter): "Whenever one or more creatures a
    # player controls deal combat damage to you, that player takes the
    # initiative." Snapshot the defending player's life now so any net combat-
    # damage loss below (unblocked hits + trample) can be detected at the end.
    defender_idx = 1 - state.active_idx if len(state.players) > 1 else None
    defender_life_before = state.players[defender_idx].life_total if defender_idx is not None else None

    unblocked = [p for p in state.attackers if p not in state.blocked_by]
    groups = list(state.blocked_by.items())  # [(attacker, [blockers, ...])] -- gang-blocking (list-valued)
    all_combatants = set(state.attackers) | {b for _a, blockers in groups for b in blockers}

    # Scan taken once and reused for every combatant checked below -- see
    # stats.enchanting_by_target's own docstring for the shared caution
    # about why this snapshot is safe today.
    enchanting_by_target = stats.enchanting_by_target(state) if all_combatants else {}

    def _facts(permanent):
        auras = enchanting_by_target.get(id(permanent), ())
        keywords = stats.creature_keywords(state, permanent, enchanting_auras=auras)
        return {
            "power": stats.permanent_power(state, permanent, enchanting_auras=auras),
            "toughness": stats.permanent_toughness(state, permanent, enchanting_auras=auras),
            "first_strike": "first_strike" in keywords,
            "trample": "trample" in keywords,
            "deathtouch": "deathtouch" in keywords,
            "lifelink_count": stats.lifelink_count(state, permanent, enchanting_auras=auras),
        }

    creature_facts = {id(p): _facts(p) for p in all_combatants}

    unblocked_total = sum(creature_facts[id(p)]["power"] for p in unblocked)
    lifelink_total = sum(
        creature_facts[id(p)]["power"] * creature_facts[id(p)]["lifelink_count"] for p in unblocked
    )
    state.attackers = []
    if unblocked_total:
        state.log_event(
            "combat_damage", source=[(p.card_def.name, p.slot) for p in unblocked], target="opponent",
            amount=unblocked_total,
        )
    deal_damage_to_opponent(state, unblocked_total)
    if lifelink_total:
        gain_life(state, lifelink_total)

    # First-strike sub-step: first-strikers on each side deal now, then an
    # SBA check clears the dead before the regular sub-step. Gang-blocking:
    # the attacker splits its damage across its LIVING blockers (the default
    # lethal-in-order split, or the attacking player's own recorded choice),
    # and every blocker deals its own power back to the attacker.
    for attacker, blockers in groups:
        if creature_facts[id(attacker)]["first_strike"]:
            living = [b for b in blockers if _is_alive(state, b)]
            amounts, opp = _damage_assignment_for(attacker, living, creature_facts[id(attacker)], creature_facts)
            _attacker_deal_damage(state, attacker, living, amounts, opp, creature_facts[id(attacker)])
        for blocker in blockers:
            if creature_facts[id(blocker)]["first_strike"]:
                _blocker_deal_damage(state, blocker, attacker, creature_facts[id(blocker)])
    state_based.check_state_based_actions(state)

    for attacker, blockers in groups:
        attacker_alive = _is_alive(state, attacker)
        if not creature_facts[id(attacker)]["first_strike"] and attacker_alive:
            living = [b for b in blockers if _is_alive(state, b)]  # a first-strike attacker's kills are already gone
            if living:
                amounts, opp = _damage_assignment_for(attacker, living, creature_facts[id(attacker)], creature_facts)
                _attacker_deal_damage(state, attacker, living, amounts, opp, creature_facts[id(attacker)])
        for blocker in blockers:
            if not creature_facts[id(blocker)]["first_strike"] and _is_alive(state, blocker) and attacker_alive:
                _blocker_deal_damage(state, blocker, attacker, creature_facts[id(blocker)])
    state_based.check_state_based_actions(state)

    # The Initiative transfer: if the current holder was the defender and took
    # any combat damage this step (life dropped), the attacking player takes
    # the initiative (and ventures). Skipped if the game already ended -- a
    # dead defender's initiative is moot. Lazy import: undercity pulls in
    # casting/tokens, and combat sits underneath those.
    if (defender_idx is not None and state.turn_won is None and state.initiative_idx == defender_idx
            and state.players[defender_idx].life_total < defender_life_before):
        from . import undercity
        undercity.take_initiative(state, state.active_idx)


def menace_block_incomplete(state):
    """True while some menace attacker has exactly ONE blocker committed -- an
    ILLEGAL block declaration ("can't be blocked except by two or more
    creatures", 509.1c: 0 or 2+, never 1). drl_env forbids the defender from
    finishing ("Done blocking") while this holds -- declaration-time
    enforcement, not a post-hoc correction. No undo exists (standing engine
    policy, see todo/no_undo_policy.md): the ONLY way forward once a menace
    attacker has exactly one committed blocker is to add a second one. If none
    is available, this stays illegal until the phase's action cap forces
    completion and enforce_menace (below) drops the illegal lone block --
    bounded, not a softlock, and reachable by a rational policy too, not just
    a pathological one (see enforce_menace's own docstring). Called only during the
    declare-blockers step, where active_idx is the defender, so the
    attacker's own blocked_by/attackers are reached via state.opponent (the
    attacking player, from the defender's point of view -- same accessor the
    block machinery already uses)."""
    return any(
        len(blockers) == 1 and stats.has_keyword(state, attacker, "menace")
        for attacker, blockers in state.opponent.blocked_by.items()
    )


def enforce_menace(state):
    """Backstop, not the primary rule: the declaration-time gate
    (menace_block_incomplete) already makes the defender declare 0 or 2+
    blockers on a menace attacker whenever they CAN, so at combat damage no
    menace attacker should have exactly one blocker on the common path. Fires
    in two cases, only the first of which is pathological: (1)
    game.turn._declare_blockers_gen abandons a partial declaration on its
    action-cap (a policy that never finishes); (2) -- reachable by a
    perfectly rational policy, not just a broken one, since there is no
    Unassign Blocker action to reconsider a commitment (standing engine
    policy, see todo/no_undo_policy.md) -- the defender has exactly one
    eligible blocker left and it's already committed to a menace attacker,
    with no second blocker to add, so declaration
    stays illegal until the action cap forces it through. Either way, this
    drops any lone menace-block so the OUTCOME is still faithful (a lone
    creature can't stop a menace attacker). active_idx is back on the
    attacker here, so state.blocked_by is the attacker's own."""
    for attacker, blockers in list(state.blocked_by.items()):
        if len(blockers) == 1 and stats.has_keyword(state, attacker, "menace"):
            del state.blocked_by[attacker]
            state.log_event("menace_unblocked", attacker=(attacker.card_def.name, attacker.slot))


def has_unfulfilled_goad(state):
    """True if the TURN player, during their OWN declare-attackers step,
    controls a goaded creature that CAN attack but isn't yet declared --
    goad's "attacks each combat if able" then forbids them ending that step
    (drl_env._pass_legal gates Pass on this during DECLARE_ATTACKERS).
    creature_attack_eligible already excludes creatures that can't attack
    (tapped/summoning-sick) and ones already in state.attackers, so a goaded
    creature that literally can't attack -- or has been declared -- never
    blocks the Pass. In 2-player, "attack a player other than you" is
    automatic (the sole opponent), so no attack-target restriction is needed
    beyond forcing the declaration.

    Gated on active_idx == turn_player_idx: the obligation is a turn-based
    action of the turn player alone. DECLARE_ATTACKERS also hands PRIORITY to
    the NON-turn player (game.turn._run_priority_round_gen flips active_idx),
    and state.battlefield proxies to active_idx -- so without this guard, a
    non-turn player who controls a goaded creature (the turn player goaded it)
    would have their priority-Pass blocked here while being unable to declare
    an attacker at all (_attack_legal needs active_idx == turn_player_idx):
    an all-False action mask, a real crash caught in rl.agent._seat_step. Goad
    binds a creature's controller on that controller's own combat, never a
    reactive priority window."""
    if state.active_idx != state.turn_player_idx:
        return False
    return any(
        p.flags.get("goaded_by") is not None and creature_attack_eligible(state, p)
        for p in state.battlefield
    )

    print("combat.py initiative transfer self-check: OK")
