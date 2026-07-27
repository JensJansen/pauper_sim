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
    exist, or as mana sources) instead of the old everyone-eligible-attacks
    wholesale rule.

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
    Menace/other restrictions aren't modeled (no card grants them). Shared by
    creature_block_eligible and drl_env's per-attacker choice predicate, so
    the rule lives in one place."""
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
    (name, slot) at the drl_env.py action-table layer
   , so the caller has
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
    start this turn's declaration fresh, one creature at a time, rather
    than the old wholesale auto-attack."""
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
    # attacker (Stage B: any portion to any blocker, not forced lethal -- user
    # spec), or _default_damage_assignment's lethal-in-order split for a single
    # blocker / the auto path. `opponent_amount` is the trample share spilled
    # to the defending player. Lifelink counts damage ACTUALLY dealt (to
    # blockers + to the player), which for a full-power assignment (the default
    # / old 1:1 path) is exactly power.
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
    trampler's leftover piles onto the last blocker so all power lands
    (matching the old 1:1 behaviour). Returns (amounts parallel to
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

    enchanting_by_target = {}
    if all_combatants:
        for player in state.players:
            for aura in player.battlefield:
                target = aura.flags.get("enchanting")
                if target is not None:
                    enchanting_by_target.setdefault(id(target), []).append(aura)

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
    # the attacker splits its damage across its LIVING blockers (Stage-A
    # default split; Stage B swaps in the model's own choice), and every
    # blocker deals its own power back to the attacker.
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
    finishing ("Done blocking") while this holds, and offers an "Unassign
    Blocker" action so they can always reach a legal declaration (add a second
    blocker or take the lone one back) -- declaration-time enforcement, not a
    post-hoc correction. Called only during the declare-blockers step, where
    active_idx is the defender, so the attacker's own blocked_by/attackers are
    reached via state.opponent (the attacking player, from the defender's
    point of view -- same accessor the block machinery already uses)."""
    return any(
        len(blockers) == 1 and stats.has_keyword(state, attacker, "menace")
        for attacker, blockers in state.opponent.blocked_by.items()
    )


def enforce_menace(state):
    """Defensive backstop, NOT the primary rule: the declaration-time gate
    (menace_block_incomplete + the Unassign action in drl_env) already makes
    the defender declare 0 or 2+ blockers on a menace attacker, so at combat
    damage no menace attacker should have exactly one blocker. This only ever
    fires if game.turn._declare_blockers_gen abandons a partial declaration on
    its action-cap (a pathological policy that never finishes) -- it then drops
    any lone menace-block so the OUTCOME is still faithful (a lone creature
    can't stop a menace attacker). A no-op on every rational path. active_idx is
    back on the attacker here, so state.blocked_by is the attacker's own."""
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
    an all-False action mask, a real crash caught in rl.train._seat_step. Goad
    binds a creature's controller on that controller's own combat, never a
    reactive priority window."""
    if state.active_idx != state.turn_player_idx:
        return False
    return any(
        p.flags.get("goaded_by") is not None and creature_attack_eligible(state, p)
        for p in state.battlefield
    )


if __name__ == "__main__":
    # ponytail self-check: run via `python -m game.effects.combat` from
    # src/. Attack eligibility + declaration + damage, then the keyword
    # trio (vigilance/trample/first strike) -- everything specific to THIS
    # module. The combat+SBA creature-death handoff lives in
    # effects/integration_check.py instead (it exercises state_based.py
    # just as much as this module).
    from ..cards import CardDef, EffectId
    from ..state import GameState, Permanent, PlayerState

    state = GameState(on_the_play=True, players=[PlayerState(True), PlayerState(False)])
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
    assert not creature_attack_eligible(state, defender)  # every other rule satisfied, but can never attack

    declare_attacker(state, attacker)
    assert attacker.tapped and attacker in state.attackers
    assert not vanilla.tapped and vanilla not in state.attackers  # partial declaration -- vanilla deliberately left back
    combat_damage_step(state)
    assert state.players[1].life_total == 17  # 20 - the lone eligible attacker's power (3)
    assert state.attackers == []
    assert state.turn_won is None

    print("combat.py eligibility + damage self-check: OK")

    # Haste (Kitchen Imp): a "haste": True registry spec lets a summoning-
    # sick creature be attack-eligible anyway -- the only place that spec
    # is ever read.
    state = GameState(on_the_play=True, players=[PlayerState(True), PlayerState(False)])
    _filler_backup = registry.EFFECT_REGISTRY[EffectId.FILLER]
    registry.EFFECT_REGISTRY[EffectId.FILLER] = {"haste": True}
    try:
        hasty = Permanent(CardDef("Hasty", CardType.CREATURE, None, EffectId.FILLER, power=2))
        assert hasty.summoning_sick
        state.battlefield = [hasty]
        declare_attackers_step(state)
        assert creature_attack_eligible(state, hasty)
        declare_attacker(state, hasty)
        combat_damage_step(state)
        assert state.players[1].life_total == 18 and hasty.tapped  # 20 - hasty's power (2)
    finally:
        registry.EFFECT_REGISTRY[EffectId.FILLER] = _filler_backup

    print("combat.py haste self-check: OK")

    # Vigilance: attacking a vigilant
    # creature never taps it, unlike an ordinary attacker.
    from .tokens import WARRIOR_TOKEN_CARD_DEF  # the real EffectId.WARRIOR_TOKEN registry entry (white_cards.py) grants vigilance

    state = GameState(on_the_play=True, players=[PlayerState(True), PlayerState(False)])
    vigilant = Permanent(WARRIOR_TOKEN_CARD_DEF)
    vigilant.summoning_sick = False
    ordinary = Permanent(CardDef("Ordinary Attacker", CardType.CREATURE, None, EffectId.FILLER, power=1, toughness=1))
    ordinary.summoning_sick = False
    state.battlefield = [vigilant, ordinary]
    declare_attackers_step(state)
    declare_attacker(state, vigilant)
    declare_attacker(state, ordinary)
    assert not vigilant.tapped and vigilant in state.attackers
    assert ordinary.tapped and ordinary in state.attackers

    # Regression: a vigilant creature staying untapped must NOT make it
    # re-declarable -- it already attacked this combat, tapped or not.
    # Before creature_attack_eligible's own state.attackers guard existed,
    # this stayed "eligible" forever (vigilance skips the only other
    # exclusion, tapped) and repeated declare_attacker calls silently
    # duplicated it in state.attackers, multiplying its power in
    # combat_damage_step's unblocked-damage total.
    assert not creature_attack_eligible(state, vigilant)
    assert state.attackers.count(vigilant) == 1
    combat_damage_step(state)
    assert state.players[1].life_total == 18  # 20 - (vigilant's power once + ordinary's power once) -- not doubled

    print("combat.py vigilance self-check: OK")

    # Trample (the real EffectId.RANCOR registry entry): a blocked
    # attacker with trample assigns only enough damage to be lethal to its
    # blocker, letting the rest spill over to the DEFENDING player.
    from ..state import PlayerState

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
        state.blocked_by[trampler] = [weak_blocker]
        combat_damage_step(state)

        # Effective power 7 (5 base + Rancor's +2): 2 assigned as lethal
        # (weak_blocker's own toughness), 5 tramples through.
        assert weak_blocker not in state.players[1].battlefield
        assert state.players[1].life_total == 15  # 20 - the 5 that trampled through
        assert trampler in state.players[0].battlefield and trampler.damage_marked == 1
    finally:
        registry.CARD_DEFS.clear()
        registry.CARD_DEFS.update(_card_defs_backup)

    print("combat.py trample self-check: OK")

    # First strike (the real EffectId.CARTOUCHE_OF_SOLIDARITY registry
    # entry): a blocked attacker with first strike deals its damage BEFORE
    # the blocker gets a chance to.
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
        state.blocked_by[fs_attacker] = [lethal_blocker]
        combat_damage_step(state)

        # Effective power 5 (4 base + Cartouche's +1) >= lethal_blocker's
        # toughness 3 -- dies in the FIRST STRIKE sub-step, before it ever
        # deals its own power-3 damage back.
        assert lethal_blocker not in state.players[1].battlefield
        assert fs_attacker in state.players[0].battlefield and fs_attacker.damage_marked == 0
    finally:
        registry.CARD_DEFS.clear()
        registry.CARD_DEFS.update(_card_defs_backup)

    print("combat.py first-strike self-check: OK")

    # Lifelink (the real EffectId.ARMADILLO_CLOAK registry entry):
    # "whenever enchanted creature deals damage, you gain that much life"
    # -- credited to whichever side actually controls the damage-dealing
    # creature, both for an unblocked attacker and for a BLOCKING creature
    # (the defending player, never the currently-active attacker).
    _card_defs_backup = dict(registry.CARD_DEFS)
    try:
        state = GameState(on_the_play=True, players=[PlayerState(True), PlayerState(False)])
        lifelinker = Permanent(CardDef("Lifelinker", CardType.CREATURE, None, EffectId.FILLER, power=3, toughness=3))
        lifelinker.summoning_sick = False
        registry.CARD_DEFS["Lifelinker"] = lifelinker.card_def
        cloak = Permanent(CardDef("Armadillo Cloak", CardType.ENCHANTMENT, {"generic": 1, "G": 1, "W": 1}, EffectId.ARMADILLO_CLOAK))
        cloak.flags["enchanting"] = lifelinker
        state.players[0].battlefield = [lifelinker, cloak]

        declare_attackers_step(state)
        declare_attacker(state, lifelinker)
        combat_damage_step(state)  # unblocked

        # Effective power 5 (3 base + Cloak's own +2, same Aura bonus
        # permanent_power always applies) -- both the damage AND the
        # lifelink gain use this effective total, not the base 3.
        assert state.players[1].life_total == 15  # 20 - the unblocked lifelinker's power (5)
        assert state.players[0].life_total == 25  # STARTING_LIFE (20) + 5, unblocked lifelink

        # STACKING: two Armadillo Cloaks on the SAME creature -- two
        # independent triggers, not a boolean that dedups to one. Real
        # rule (unlike real lifelink, which never stacks): each Cloak's
        # own "whenever enchanted creature deals damage, you gain that
        # much life" fires separately, each for the FULL damage dealt.
        state = GameState(on_the_play=True, players=[PlayerState(True), PlayerState(False)])
        double_cloaked = Permanent(CardDef("Double Cloaked", CardType.CREATURE, None, EffectId.FILLER, power=3, toughness=3))
        double_cloaked.summoning_sick = False
        registry.CARD_DEFS["Double Cloaked"] = double_cloaked.card_def
        cloak_a = Permanent(CardDef("Armadillo Cloak", CardType.ENCHANTMENT, {"generic": 1, "G": 1, "W": 1}, EffectId.ARMADILLO_CLOAK))
        cloak_a.flags["enchanting"] = double_cloaked
        cloak_b = Permanent(CardDef("Armadillo Cloak", CardType.ENCHANTMENT, {"generic": 1, "G": 1, "W": 1}, EffectId.ARMADILLO_CLOAK))
        cloak_b.flags["enchanting"] = double_cloaked
        state.players[0].battlefield = [double_cloaked, cloak_a, cloak_b]

        declare_attackers_step(state)
        declare_attacker(state, double_cloaked)
        combat_damage_step(state)  # unblocked

        # Effective power 7 (3 base + 2+2 from both Cloaks). Life gained:
        # 7 * 2 (one trigger per Cloak) = 14, NOT just 7 (what a boolean
        # "lifelink" keyword would wrongly give, deduped to one trigger).
        assert state.players[1].life_total == 13  # 20 - the double-cloaked lifelinker's power (7)
        assert state.players[0].life_total == 34  # STARTING_LIFE (20) + 14

        # A blocked lifelinker with trample: effective power 5 (3 base +
        # Cloak's own +2) vs a 2-toughness blocker -- 2 assigned as
        # lethal, 3 tramples through, but the FULL 5 is gained as life --
        # Armadillo Cloak's own text has no "excess damage only" carve-out,
        # unlike trample's own player-damage rule.
        state = GameState(on_the_play=True, players=[PlayerState(True), PlayerState(False)])
        lifelinker2 = Permanent(CardDef("Lifelinker2", CardType.CREATURE, None, EffectId.FILLER, power=3, toughness=3))
        lifelinker2.summoning_sick = False
        registry.CARD_DEFS["Lifelinker2"] = lifelinker2.card_def
        cloak2 = Permanent(CardDef("Armadillo Cloak", CardType.ENCHANTMENT, {"generic": 1, "G": 1, "W": 1}, EffectId.ARMADILLO_CLOAK))
        cloak2.flags["enchanting"] = lifelinker2
        blocker = Permanent(CardDef("Blocker", CardType.CREATURE, None, EffectId.FILLER, power=1, toughness=2))
        registry.CARD_DEFS["Blocker"] = blocker.card_def
        state.players[0].battlefield = [lifelinker2, cloak2]
        state.players[1].battlefield = [blocker]

        declare_attackers_step(state)
        declare_attacker(state, lifelinker2)
        state.blocked_by[lifelinker2] = [blocker]
        combat_damage_step(state)

        assert state.players[1].life_total == 17  # 20 - the trampled-through excess (5 power - 2 lethal)
        assert state.players[0].life_total == 25  # +5, the FULL effective power, not just the excess

        # A BLOCKING lifelinker: life goes to the DEFENDING player (index
        # 1 here, since player 0 is the attacker/active side throughout
        # combat_damage_step), never state.life_total (which would
        # silently credit the wrong side).
        state = GameState(on_the_play=True, players=[PlayerState(True), PlayerState(False)])
        attacker3 = Permanent(CardDef("Attacker3", CardType.CREATURE, None, EffectId.FILLER, power=1, toughness=5))
        attacker3.summoning_sick = False
        registry.CARD_DEFS["Attacker3"] = attacker3.card_def
        blocker_lifelinker = Permanent(CardDef("BlockerLifelinker", CardType.CREATURE, None, EffectId.FILLER, power=2, toughness=2))
        registry.CARD_DEFS["BlockerLifelinker"] = blocker_lifelinker.card_def
        cloak3 = Permanent(CardDef("Armadillo Cloak", CardType.ENCHANTMENT, {"generic": 1, "G": 1, "W": 1}, EffectId.ARMADILLO_CLOAK))
        cloak3.flags["enchanting"] = blocker_lifelinker
        state.players[0].battlefield = [attacker3]
        state.players[1].battlefield = [blocker_lifelinker, cloak3]

        declare_attackers_step(state)
        declare_attacker(state, attacker3)
        state.blocked_by[attacker3] = [blocker_lifelinker]
        combat_damage_step(state)

        assert state.players[1].life_total == 24  # +4 (2 base + Cloak's +2) -- the DEFENDING player
        assert state.players[0].life_total == 20  # unaffected -- attacker3 itself has no lifelink

        # --- GANG-BLOCKING: one attacker, two blockers, model-decided split ---
        # A 3/3 attacker gang-blocked by two 2/2s. The attacker's controller
        # assigns 2 damage to the first blocker (lethal -> dies) and only 1 to
        # the second (survives) -- an ARBITRARY, non-lethal-to-all split, the
        # whole point of the model decision. Both blockers deal 2 back, so the
        # attacker takes 4 >= 3 and dies. The split is stashed on the
        # attacker's own flags exactly as resolution.begin_assign_combat_damage
        # records it, and must be consumed (popped) by the damage step.
        gang_state = GameState(on_the_play=True, players=[PlayerState(True), PlayerState(False)])
        gb_atk = Permanent(CardDef("GBAttacker", CardType.CREATURE, None, EffectId.FILLER, power=3, toughness=3))
        gb_atk.summoning_sick = False
        gb_b1 = Permanent(CardDef("GBBlocker1", CardType.CREATURE, None, EffectId.FILLER, power=2, toughness=2))
        gb_b2 = Permanent(CardDef("GBBlocker2", CardType.CREATURE, None, EffectId.FILLER, power=2, toughness=2))
        for _p in (gb_atk, gb_b1, gb_b2):
            registry.CARD_DEFS[_p.card_def.name] = _p.card_def
        gang_state.players[0].battlefield = [gb_atk]
        gang_state.players[1].battlefield = [gb_b1, gb_b2]
        declare_attackers_step(gang_state)
        declare_attacker(gang_state, gb_atk)
        gang_state.blocked_by[gb_atk] = [gb_b1, gb_b2]
        gb_atk.flags["combat_damage_split"] = ({gb_b1: 2, gb_b2: 1}, 0)  # attacker's own arbitrary choice
        combat_damage_step(gang_state)
        assert gb_b1 not in gang_state.players[1].battlefield  # took 2 (lethal) -> dead
        assert gb_b2 in gang_state.players[1].battlefield       # took only 1 -> survives
        assert gb_atk not in gang_state.players[0].battlefield  # took 2+2=4 >= 3 -> dead
        assert "combat_damage_split" not in gb_atk.flags        # consumed (popped) by the damage step
    finally:
        registry.CARD_DEFS.clear()
        registry.CARD_DEFS.update(_card_defs_backup)

    print("combat.py lifelink self-check: OK")
    print("combat.py gang-blocking damage-split self-check: OK")

    # can_block: evasion + reach (C8/C9). A real flier (Kitchen Imp) and
    # Silhana's "can't be blocked except by flying" both demand a flying or
    # reach blocker; reach (Bramble Wurm) satisfies that, a vanilla creature
    # doesn't, and -- crucially -- Silhana itself (evasion, NOT real flying)
    # can NOT block a flier.
    kb_state = GameState(on_the_play=True, players=[PlayerState(True), PlayerState(False)])
    imp = Permanent(CardDef("Imp", CardType.CREATURE, None, EffectId.KITCHEN_IMP, power=2, toughness=2))
    silhana = Permanent(CardDef("Silhana", CardType.CREATURE, None, EffectId.SILHANA_LEDGEWALKER, power=1, toughness=1))
    wurm = Permanent(CardDef("Wurm", CardType.CREATURE, None, EffectId.BRAMBLE_WURM, power=7, toughness=6))
    vanilla_c = Permanent(CardDef("Vanilla", CardType.CREATURE, None, EffectId.FILLER, power=2, toughness=2))
    assert not can_block(kb_state, vanilla_c, imp)      # vanilla can't block a flier
    assert can_block(kb_state, wurm, imp)               # reach blocks a flier
    assert can_block(kb_state, imp, imp)                # flying blocks flying
    assert not can_block(kb_state, vanilla_c, silhana)  # Silhana's evasion needs a flying/reach blocker
    assert can_block(kb_state, wurm, silhana)           # reach satisfies Silhana's evasion too
    assert not can_block(kb_state, silhana, imp)        # Silhana is NOT flying -> can't block a flier (the C8 fix)
    assert can_block(kb_state, vanilla_c, vanilla_c)    # no restriction -> anyone blocks
    print("combat.py can_block evasion/reach self-check: OK")

    # --- G12: menace, goad, initiative transfer ---
    from ..cards import EffectId as _EID

    # Menace (509.1c): a declaration leaving a menace attacker with exactly ONE
    # blocker is illegal. menace_block_incomplete flags it (so drl_env forbids
    # "Done" until fixed -- 0 or 2+); enforce_menace is the cap-abandon backstop
    # that drops a stray lone block. Player 0 attacks, player 1 (the defender,
    # active during blocking) assigns blockers.
    _fb = registry.EFFECT_REGISTRY[_EID.FILLER]
    registry.EFFECT_REGISTRY[_EID.FILLER] = {"keywords": {"menace"}}
    try:
        state = GameState(on_the_play=True, players=[PlayerState(True), PlayerState(False)])
        menacer = Permanent(CardDef("Menacer", CardType.CREATURE, None, _EID.FILLER, power=4, toughness=1))
        menacer.summoning_sick = False
        state.players[0].battlefield = [menacer]
        state.players[0].attackers = [menacer]
        lone = Permanent(CardDef("Lone", CardType.CREATURE, None, None, power=1, toughness=1))
        state.active_idx = 1  # the DEFENDER is active during declare-blockers

        state.players[0].blocked_by = {menacer: [lone]}  # exactly one -> incomplete/illegal
        assert menace_block_incomplete(state)  # "Done" would be forbidden here
        b2 = Permanent(CardDef("B2", CardType.CREATURE, None, None, power=1, toughness=1))
        state.players[0].blocked_by = {menacer: [lone, b2]}  # two -> a legal block
        assert not menace_block_incomplete(state)
        state.players[0].blocked_by = {}  # zero -> also legal (unblocked)
        assert not menace_block_incomplete(state)

        # Backstop: enforce_menace (active back on the attacker) drops a stray
        # lone menace-block, keeping a two-block one.
        state.active_idx = 0
        state.players[0].blocked_by = {menacer: [lone]}
        enforce_menace(state)
        assert menacer not in state.players[0].blocked_by  # unblocked
        state.players[0].blocked_by = {menacer: [lone, b2]}
        enforce_menace(state)
        assert state.players[0].blocked_by == {menacer: [lone, b2]}
    finally:
        registry.EFFECT_REGISTRY[_EID.FILLER] = _fb
    print("combat.py menace (declaration incomplete-gate + backstop) self-check: OK")

    # Goad: a goaded creature that can attack blocks its controller's Pass
    # (has_unfulfilled_goad) until it's declared.
    state = GameState(on_the_play=True, players=[PlayerState(True), PlayerState(False)])
    state.active_idx = state.turn_player_idx = 0
    goaded = Permanent(CardDef("Goaded", CardType.CREATURE, None, _EID.FILLER, power=2, toughness=2))
    goaded.summoning_sick = False
    goaded.flags["goaded_by"] = 1
    state.players[0].battlefield = [goaded]
    declare_attackers_step(state)
    assert has_unfulfilled_goad(state)  # able + undeclared -> must attack
    declare_attacker(state, goaded)
    assert not has_unfulfilled_goad(state)  # now declared -> satisfied
    # a goaded creature that CAN'T attack (tapped) never forces the issue
    state.players[0].attackers = []
    goaded.tapped = True
    assert not has_unfulfilled_goad(state)

    # Regression: goad binds the turn player during THEIR own declare step, not
    # a NON-turn player who merely holds priority during DECLARE_ATTACKERS
    # (game.turn._run_priority_round_gen flips active_idx to them). A forcing
    # goaded creature under the non-turn player must NOT block their priority-
    # Pass -- they cannot declare an attacker at all (_attack_legal needs
    # active_idx == turn_player_idx), so blocking it left an all-False action
    # mask (the rl.train._seat_step crash this guards). turn_player_idx stays 0.
    nonturn_goaded = Permanent(CardDef("NonturnGoaded", CardType.CREATURE, None, _EID.FILLER, power=2, toughness=2))
    nonturn_goaded.summoning_sick = False
    nonturn_goaded.flags["goaded_by"] = 0  # goaded by the turn player (idx 0)
    state.players[1].battlefield = [nonturn_goaded]
    state.active_idx = 1  # non-turn player now holds priority
    assert not has_unfulfilled_goad(state), "goad must never block a NON-turn player's priority-Pass"
    print("combat.py goad (forces declaration) self-check: OK")

    # Initiative transfer: the holder taking combat damage passes it to the
    # attacker, who then has a venture queued.
    state = GameState(on_the_play=True, players=[PlayerState(True), PlayerState(False)])
    state.active_idx = state.turn_player_idx = 0
    state.initiative_idx = 1  # the DEFENDER holds it
    hitter = Permanent(CardDef("Hitter", CardType.CREATURE, None, _EID.FILLER, power=3, toughness=3))
    hitter.summoning_sick = False
    state.players[0].battlefield = [hitter]
    state.players[0].attackers = [hitter]
    combat_damage_step(state)
    assert state.players[1].life_total == 17  # 3 unblocked to the defender
    assert state.initiative_idx == 0  # stolen by the attacker
    assert any(e["type"] == "venture" for e in state.players[0].trigger_queue)
    print("combat.py initiative transfer self-check: OK")
