"""Attack/block declaration, eligibility, and combat damage. Depends on
stats.py for effective power/toughness/keywords, state_based.py for the
state-based-action check combat damage triggers, and win_check.py for the
opponent-facing damage effect (and, for "lifelink", the life-gaining one)."""

from . import stats, state_based
from .win_check import deal_damage_to_opponent, gain_life
from ..cards import CardType


def creature_attack_eligible(state, permanent):
    """Untapped, not a Defender (which can never attack regardless of
    tapped/summoning-sick status), and not summoning sick unless it has
    haste (stats.has_haste). Checked per creature so a model can declare
    some eligible creatures as attackers and hold others back.

    Also excludes anything already in state.attackers: vigilance
    (Cartouche of Solidarity's Warrior token) skips the declare-time tap,
    so without this explicit guard a vigilant creature could be declared
    an attacker repeatedly, multiplying its power in combat_damage_step's
    unblocked-damage total. Mirrors creature_block_eligible's identical
    guard."""
    return (
        permanent.card_type == CardType.CREATURE and not permanent.tapped
        and not permanent.card_def.extra.get("defender", False)
        and permanent not in state.attackers
        and (not permanent.summoning_sick or stats.has_haste(state, permanent))
    )


def _blocked_by_creatures(blocked_by):
    """Flatten a blocked_by dict ({attacker: [blockers]}) to the flat set
    of creatures currently committed as blockers. Gang-blocking makes the
    values lists, so this is a union, not a `.values()` membership test."""
    return {b for blockers in blocked_by.values() for b in blockers}


def remove_from_combat(state, permanent):
    """506.4: a permanent removed from the battlefield stops being an
    attacking, blocking, blocked and/or unblocked creature, simultaneously.
    Call this from every battlefield-exit path that can move a creature
    (death, sacrifice, bounce, exile) -- there is no single zone-change
    choke point in this engine.

    Idempotent, and safe to call for a permanent that was never in combat.

    Scope note: this removes `permanent` itself from combat but does not
    drop the blocked_by entry keyed by a removed attacker -- dropping it
    would free that attacker's already-declared blockers for a second
    block, and real Magic declares blockers once, simultaneously (509.1)."""
    for player in state.players:
        if permanent in player.attackers:
            player.attackers.remove(permanent)
        for blockers in player.blocked_by.values():
            if permanent in blockers:
                blockers.remove(permanent)  # 506.4: a removed blocker stops being a blocking creature


def can_block(state, blocker, attacker):
    """Whether `blocker` is allowed to block `attacker`. Two evasion rules
    modeled: an attacker that can only be blocked by flying (real flying,
    or Silhana Ledgewalker's "can't be blocked except by creatures with
    flying") may be blocked only by a creature with flying or reach; reach
    lets a non-flying creature block a flier. Menace isn't a per-blocker
    restriction, so it plays no part here -- see menace_block_incomplete
    and enforce_menace below.

    Shared by creature_block_eligible and drl_env._assign_blocker_execute's
    extra_predicate, so the rule lives in one place -- any future evasion
    rule belongs here, never re-derived by a caller."""
    attacker_needs_flying_blocker = (
        stats.has_keyword(state, attacker, "flying")
        or stats.has_keyword(state, attacker, "cant_be_blocked_except_by_flying")
    )
    if not attacker_needs_flying_blocker:
        return True
    return stats.has_keyword(state, blocker, "flying") or stats.has_keyword(state, blocker, "reach")


def creature_block_eligible(state, permanent):
    """A creature untapped, not already committed as a blocker, and with
    at least one attacker it's actually allowed to block right now. Reads
    state.opponent.blocked_by/attackers, not the defending player's own,
    since those are keyed by the attacking player's permanents. Not the
    same eligibility as creature_attack_eligible: real Magic lets a
    Defender block and lets a summoning-sick creature block.

    The "has at least one legal target" clause stops a blocker with
    nothing it can legally block (e.g. a non-flying blocker when every
    attacker has flying) from being offered as a legal 'Assign Blocker'
    action that would be a no-op."""
    if not (permanent.card_type == CardType.CREATURE and not permanent.tapped
            and permanent not in _blocked_by_creatures(state.opponent.blocked_by)):
        return False
    return any(can_block(state, permanent, attacker) for attacker in state.opponent.attackers)


def declare_attacker(state, permanent):
    """Model chose to attack with this specific creature. Tapped here, at
    declaration, same as real Magic -- an attacking creature is unavailable
    for the rest of combat, not just as a side effect of dealing damage
    later -- unless it has vigilance (Cartouche of Solidarity's Warrior
    token), which doesn't tap on attack at all."""
    if not stats.has_keyword(state, permanent, "vigilance"):
        permanent.tapped = True
    state.attackers.append(permanent)
    state.log_event("attack_declared", attacker=(permanent.card_def.name, permanent.slot), tapped=permanent.tapped)


def declare_attackers_step(state):
    """game.turn.Phase.DECLARE_ATTACKERS phase-entry reset: clears last
    turn's attackers and blocks together, so the model's own "Attack:
    <name>" actions start this turn's declaration fresh."""
    state.attackers = []
    state.blocked_by = {}


def _is_alive(state, permanent):
    return any(permanent in player.battlefield for player in state.players)


def _attacker_deal_damage(state, attacker, blockers, amounts, opponent_amount, attacker_facts):
    """Attacker deals its combat damage across a list of `blockers` per the
    parallel `amounts` list, plus `opponent_amount` trample-spilled to the
    defending player. state.opponent, from the attacker's active
    perspective, is the defender throughout combat_damage_step.

    lifelink (Armadillo Cloak's own triggered ability, unlike real
    lifelink): stats.lifelink_count returns however many Cloaks are
    attached, each an independent trigger for the full damage dealt, so
    two Cloaks means 2x life gained.

    attacker_facts: combat_damage_step's pre-fetched {power, toughness,
    first_strike, trample, lifelink_count} dict for this attacker, avoiding
    a re-scan of state.players' Auras on every blocked pair."""
    # amounts[i] is damage assigned to blockers[i] -- the attacking
    # player's own free choice for a multi-blocked attacker, or
    # _default_damage_assignment's lethal-in-order split otherwise.
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
    never tramples through to a player: trample is attacker-only, and this
    engine doesn't model a blocker-side equivalent.

    lifelink: the blocker's controller is the defending player
    (players[1 - active_idx]), not the active player, so this passes
    gain_life that index explicitly. Multiplied by stats.lifelink_count the
    same way as the attacker-side case above.

    blocker_facts: see _attacker_deal_damage's docstring -- same
    pre-fetched dict, the blocker's own this time."""
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


def _lethal_amount(deathtouch, toughness, damage_marked):
    """Combat damage needed to treat a blocker as 'satisfied' for lethal-
    in-order assignment -- 1 for a deathtouch source (702.2b), else its
    remaining toughness. Shared by the auto path and the free multi-blocker
    resolution (blocker_lethal_capacities) so both agree on the same cap."""
    return 1 if deathtouch else max(toughness - damage_marked, 0)


def _default_damage_assignment(attacker_facts, blockers, facts_by_id):
    """Auto split of an attacker's combat damage across its living
    blockers -- used for a single blocker and as the fallback when no
    explicit model assignment exists. Lethal-in-order to maximize kills; a
    trampler's leftover goes to the player, a non-trampler's leftover
    piles onto the last blocker. Returns (amounts parallel to `blockers`,
    opponent_amount). For a multi-blocked attacker the attacking player's
    own freely-chosen split (assign_combat_damage resolution) replaces
    this instead."""
    remaining = attacker_facts["power"]
    amounts = [0] * len(blockers)
    deathtouch = attacker_facts["deathtouch"]
    for i, blocker in enumerate(blockers):
        if remaining <= 0:
            break
        # 702.2b + 510.1a: any nonzero damage from a deathtouch source
        # counts as lethal, so one point per blocker frees the rest to
        # trample through or kill further blockers. A single-blocked
        # attacker never gets an assign_combat_damage decision, so this
        # auto split is the whole decision -- it should be attacker-optimal.
        lethal = _lethal_amount(deathtouch, facts_by_id[id(blocker)]["toughness"], blocker.damage_marked)
        assign = min(remaining, lethal)
        amounts[i] = assign
        remaining -= assign
    opponent_amount = 0
    if remaining > 0:
        if attacker_facts["trample"]:
            opponent_amount = remaining
        elif blockers:
            # RULES EXCEPTION (owner-approved 2026-08-20): real Magic still
            # requires this dead-letter overkill be assigned to A blocker
            # even though it changes nothing (every blocker here is already
            # at its own lethal cap, so this is strictly cosmetic). Rather
            # than model which specific blocker "in reality" absorbs it,
            # this engine piles all of it onto the last one. Inert in
            # practice -- nothing in this card pool reads how much *excess*
            # damage an already-dead blocker received -- but flagged per
            # this repo's rules-faithfulness mandate rather than silently
            # assumed.
            amounts[-1] += remaining
    return amounts, opponent_amount


def blocker_lethal_capacities(state, attacker, blockers):
    """{blocker: the most combat damage this attacker's free multi-blocker
    assignment may ever put on it} -- the same _lethal_amount cap the auto
    path uses, computed live (once per multi-blocked attacker). Consumed by
    handlers_combat.begin_assign_combat_damage to stop the agent from
    overkilling a blocker or trampling through before every blocker in the
    split has its own lethal share assigned."""
    deathtouch = stats.has_keyword(state, attacker, "deathtouch")
    return {
        b: _lethal_amount(deathtouch, stats.permanent_toughness(state, b), b.damage_marked)
        for b in blockers
    }


def _damage_assignment_for(attacker, living, attacker_facts, facts_by_id):
    """The (amounts-parallel-to-living, opponent_amount) split this
    attacker deals right now: the attacking player's own recorded choice
    (attacker.flags['combat_damage_split'], popped here) if any, else the
    lethal-in-order default. A model split keyed by blocker maps onto
    `living`, which can be a strict subset of the split's own keys: damage
    is assigned during DECLARE_BLOCKERS but not dealt until COMBAT_DAMAGE,
    a phase later, so a blocker the split earmarked damage for can die to
    an instant in that window. Whatever was earmarked for a since-departed
    blocker is added to a trampler's own excess (702.19b); a non-trampler
    simply never deals it (509.1h)."""
    split = attacker.flags.pop("combat_damage_split", None)
    if split is not None:
        amounts_by_blocker, opponent_amount = split
        living_set = set(living)
        orphaned = sum(amount for blocker, amount in amounts_by_blocker.items() if blocker not in living_set)
        if orphaned and attacker_facts["trample"]:
            opponent_amount += orphaned
        return [amounts_by_blocker.get(b, 0) for b in living], opponent_amount
    return _default_damage_assignment(attacker_facts, living, facts_by_id)


def attackers_needing_damage_assignment(state):
    """The (attacker, blockers, power, has_trample) tuples for attackers
    blocked by 2+ creatures with nonzero power -- the ones whose controller
    must freely assign combat damage. A lone blocker or a 0-power attacker
    has no choice, so it's skipped and combat_damage_step auto-assigns via
    _default_damage_assignment."""
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
    """game.turn.Phase.COMBAT_DAMAGE. Unblocked attackers (in
    state.attackers, not in state.blocked_by) deal their total power to
    the opponent at once.

    Blocked pairs fight in up to two sub-steps (real Magic first-strike
    order): first-strikers deal, then an SBA check clears the dead; then
    the non-first-strikers deal, if both are still alive, and a second SBA
    check. With no first strike this collapses to one simultaneous
    exchange.

    lifelink on an unblocked attacker is gained here, batched, each
    attacker's power x its lifelink_count.

    Every combatant's {power, toughness, first_strike, trample, deathtouch,
    lifelink_count} is fetched once up front and reused, avoiding redundant
    per-pair Aura scans. Safe since nothing here casts or (de)attaches an
    Aura mid-resolution."""
    # The Initiative: keyed on combat damage actually dealt to the
    # defending player (unblocked hits + trample overflow), tracked below
    # as damage_dealt_to_defender -- not a net life-total delta, which
    # would miss the transfer whenever life gained elsewhere this step
    # (a lifelink blocker) offsets it.
    defender_idx = 1 - state.active_idx if len(state.players) > 1 else None
    damage_dealt_to_defender = 0

    # Both halves derive from state.attackers, the one authority on what is
    # still attacking (not blocked_by.items(), which can retain a stale
    # entry after 506.4 removes an attacker from combat).
    unblocked = [p for p in state.attackers if p not in state.blocked_by]
    groups = [(a, state.blocked_by[a]) for a in state.attackers if a in state.blocked_by]  # gang-blocking: list-valued
    all_combatants = set(state.attackers) | {b for _a, blockers in groups for b in blockers}

    # Scan taken once and reused for every combatant checked below -- see
    # stats.enchanting_by_target's own docstring for why this is safe.
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
    damage_dealt_to_defender += unblocked_total
    if lifelink_total:
        gain_life(state, lifelink_total)

    # First-strike sub-step: first-strikers on each side deal now, then an
    # SBA check clears the dead before the regular sub-step. Gang-blocking:
    # the attacker splits damage across its living blockers, and every
    # blocker deals its own power back to the attacker.
    for attacker, blockers in groups:
        if creature_facts[id(attacker)]["first_strike"]:
            living = [b for b in blockers if _is_alive(state, b)]
            amounts, opp = _damage_assignment_for(attacker, living, creature_facts[id(attacker)], creature_facts)
            damage_dealt_to_defender += opp
            _attacker_deal_damage(state, attacker, living, amounts, opp, creature_facts[id(attacker)])
        for blocker in blockers:
            if creature_facts[id(blocker)]["first_strike"]:
                _blocker_deal_damage(state, blocker, attacker, creature_facts[id(blocker)])
    state_based.check_state_based_actions(state)

    for attacker, blockers in groups:
        attacker_alive = _is_alive(state, attacker)
        if not creature_facts[id(attacker)]["first_strike"] and attacker_alive:
            living = [b for b in blockers if _is_alive(state, b)]  # a first-strike attacker's kills are already gone
            # Unconditional even when living is empty: a non-trampler then
            # deals no damage (a no-op), but a trampler with no blockers
            # left must have its full power go through to the defending
            # player (702.19b/510.1c), which the same blockers=[] path
            # already produces correctly.
            amounts, opp = _damage_assignment_for(attacker, living, creature_facts[id(attacker)], creature_facts)
            damage_dealt_to_defender += opp
            _attacker_deal_damage(state, attacker, living, amounts, opp, creature_facts[id(attacker)])
        for blocker in blockers:
            if not creature_facts[id(blocker)]["first_strike"] and _is_alive(state, blocker) and attacker_alive:
                _blocker_deal_damage(state, blocker, attacker, creature_facts[id(blocker)])
    state_based.check_state_based_actions(state)

    # The Initiative transfer: if the current holder was the defender and
    # any of the attacker's creatures dealt combat damage to them this
    # step, queue the attacker's "take the initiative" trigger (CR 722.2's
    # second one) -- state.initiative_idx only flips once it resolves off
    # the stack. Skipped if the game already ended. Lazy import: undercity
    # pulls in casting/tokens, and combat sits underneath those.
    if (defender_idx is not None and state.turn_won is None and state.initiative_idx == defender_idx
            and damage_dealt_to_defender > 0):
        from . import undercity
        undercity.queue_take_initiative(state, state.active_idx)


def menace_block_incomplete(state):
    """True while some menace attacker has exactly one blocker committed --
    an illegal block declaration (509.1c: 0 or 2+, never 1). drl_env
    forbids the defender from finishing ("Done blocking") while this
    holds -- declaration-time enforcement, not a post-hoc correction. No
    undo exists (todo/no_undo_policy.md), so the only way forward once a
    menace attacker has one committed blocker is to add a second.

    If none is available, game.turn._declare_blockers_gen's own zero-
    legal-actions check abandons the declaration; enforce_menace below then
    drops the illegal lone block at combat damage. Called only during the
    declare-blockers step, where active_idx is the defender, so the
    attacker's own blocked_by/attackers are reached via state.opponent."""
    return any(
        len(blockers) == 1 and stats.has_keyword(state, attacker, "menace")
        for attacker, blockers in state.opponent.blocked_by.items()
    )


def enforce_menace(state):
    """Backstop, not the primary rule: the declaration-time gate
    (menace_block_incomplete) already makes the defender declare 0 or 2+
    blockers on a menace attacker whenever they can, so at combat damage no
    menace attacker should have exactly one blocker on the common path.
    Fires when a declaration is abandoned (action-cap) or stays illegal
    (no second blocker available, and no Unassign Blocker action to
    reconsider a commitment). Either way, drops any lone menace-block so
    the outcome stays faithful. active_idx is back on the attacker here,
    so state.blocked_by is the attacker's own."""
    for attacker, blockers in list(state.blocked_by.items()):
        if len(blockers) == 1 and stats.has_keyword(state, attacker, "menace"):
            del state.blocked_by[attacker]
            state.log_event("menace_unblocked", attacker=(attacker.card_def.name, attacker.slot))


def has_unfulfilled_goad(state):
    """True if the turn player, during their own declare-attackers step,
    controls a goaded creature that can attack but isn't yet declared --
    goad's "attacks each combat if able" forbids ending that step
    (drl_env._pass_legal gates Pass on this). creature_attack_eligible
    already excludes creatures that can't attack and ones already
    declared, so those never block the Pass.

    Gated on active_idx == turn_player_idx: the obligation is a turn-based
    action of the turn player alone. DECLARE_ATTACKERS also hands priority
    to the non-turn player, so without this guard a non-turn player
    controlling a goaded creature would have their priority-Pass blocked
    here while unable to declare an attacker at all (an all-False action
    mask). Goad binds a creature's controller on that controller's own
    combat, never a reactive priority window."""
    if state.active_idx != state.turn_player_idx:
        return False
    return any(
        p.flags.get("goaded_by") is not None and creature_attack_eligible(state, p)
        for p in state.battlefield
    )

    print("combat.py initiative transfer self-check: OK")
