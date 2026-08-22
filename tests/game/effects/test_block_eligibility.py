"""Who may block -- creature_block_eligible's own rules, deliberately
different from creature_attack_eligible (Defender may block; summoning
sickness doesn't stop blocking). Evasion rules (flying/reach/Silhana) are
covered by test_combat.py's test_can_block_evasion_and_reach.
"""
from game import registry
from game.cards import CardDef, CardType, EffectId
from game.effects.combat import creature_block_eligible, menace_block_incomplete
from game.state import GameState, Permanent, PlayerState


def _defending_state():
    """A state mid-combat with player 0 attacking and player 1 defending, with
    active_idx already flipped to the defender the way
    turn._declare_blockers_gen does before any of this is consulted."""
    state = GameState(on_the_play=True, players=[PlayerState(True), PlayerState(False)])
    attacker = Permanent(CardDef("Attacker", CardType.CREATURE, None, EffectId.FILLER, power=2, toughness=2))
    attacker.summoning_sick = False
    attacker.tapped = True  # already declared
    state.players[0].battlefield = [attacker]
    state.players[0].attackers = [attacker]
    state.active_idx = 1
    return state, attacker


def test_defender_creature_may_block():
    # 509.1a: Defender only forbids attacking, not blocking.
    state, _attacker = _defending_state()
    wall = Permanent(CardDef("Wall of Roots", CardType.CREATURE, None, EffectId.FILLER,
                             power=0, toughness=5, defender=True))
    wall.summoning_sick = False
    state.players[1].battlefield = [wall]
    assert creature_block_eligible(state, wall), "a Defender blocks fine -- it just cannot attack"


def test_summoning_sick_creature_may_block():
    state, _attacker = _defending_state()
    fresh = Permanent(CardDef("Just Cast", CardType.CREATURE, None, EffectId.FILLER, power=2, toughness=2))
    assert fresh.summoning_sick, "fixture assumes a freshly-cast creature"
    state.players[1].battlefield = [fresh]
    assert creature_block_eligible(state, fresh), "summoning sickness never prevents blocking"


def test_tapped_creature_cannot_block():
    # 509.1a: a tapped creature cannot be declared as a blocker.
    state, _attacker = _defending_state()
    tapped = Permanent(CardDef("Tapped Out", CardType.CREATURE, None, EffectId.FILLER,
                               power=2, toughness=2), tapped=True)
    tapped.summoning_sick = False
    state.players[1].battlefield = [tapped]
    assert not creature_block_eligible(state, tapped)


def test_a_creature_already_blocking_cannot_block_again():
    # 509.1: one creature blocks exactly one attacker. blocked_by values are
    # lists (gang-blocking); a creature in any of them is already spent.
    state, attacker = _defending_state()
    blocker = Permanent(CardDef("Blocker", CardType.CREATURE, None, EffectId.FILLER, power=2, toughness=2))
    blocker.summoning_sick = False
    state.players[1].battlefield = [blocker]
    assert creature_block_eligible(state, blocker)

    state.players[0].blocked_by[attacker] = [blocker]
    assert not creature_block_eligible(state, blocker), "already blocking -- cannot be assigned twice"


def test_non_creature_permanent_is_never_block_eligible():
    state, _attacker = _defending_state()
    land = Permanent(CardDef("Forest", CardType.LAND, None, EffectId.FILLER))
    land.summoning_sick = False
    state.players[1].battlefield = [land]
    assert not creature_block_eligible(state, land)


def test_menace_with_only_one_available_blocker_leaves_the_declaration_stuck():
    # a menace attacker with one committed blocker makes "Done" illegal
    # (menace_block_incomplete), and with no other eligible creature there is
    # no legal action left; _declare_blockers_gen must abandon the
    # declaration rather than spin, and enforce_menace drops the illegal lone
    # block at combat damage.
    original = registry.EFFECT_REGISTRY[EffectId.FILLER]
    registry.EFFECT_REGISTRY[EffectId.FILLER] = {"keywords": {"menace"}}
    try:
        state, attacker = _defending_state()
        lone = Permanent(CardDef("Lone", CardType.CREATURE, None, EffectId.FILLER, power=1, toughness=1))
        lone.summoning_sick = False
        state.players[1].battlefield = [lone]

        state.players[0].blocked_by[attacker] = [lone]
        assert menace_block_incomplete(state), "one blocker on a menace attacker is an illegal declaration"
        assert not creature_block_eligible(state, lone), "the only blocker is already committed"
    finally:
        registry.EFFECT_REGISTRY[EffectId.FILLER] = original
