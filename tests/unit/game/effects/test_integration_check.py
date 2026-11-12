"""Scenarios that need two or more game.effects submodules cooperating
together; single-module scenarios live in each submodule's own test_*.py."""
from game import mana, registry, resolution
from game.cards import CardDef, CardType, EffectId
from game.effects import combat, madness_and_plot, stack, state_based, triggers
from game.state import GameState, Permanent, PlayerState


def test_madness_chain_triggers_stack_madness_and_plot_mana_resolution():
    # full madness chain: discard -> exile + queue -> drain -> decision ->
    # pay madness cost -> resolve -> drain again, across triggers.py,
    # stack.py, madness_and_plot.py, mana.py, and resolution.
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

        # promote (game.turn's priority round): the trigger becomes a real
        # stack entry, giving the opponent a priority window before the
        # cast-or-decline choice is offered; resolving it opens the decision.
        triggers.promote_triggers_to_stack(state)
        assert state.pending_resolution is None  # just sitting on the stack, not open yet
        assert len(state.stack) == 1 and state.stack[0]["card_def"] is madness_card
        stack.resolve_top_of_stack(state)
        assert state.pending_resolution["kind"] == "madness_decision"
        assert resolution.madness_decision_options(state) == ["cast", "decline"]

        madness_and_plot.execute_madness_cast(state)
        # begin_pay_cost's resolution (paying {G}) nests through the
        # captured outer_on_complete correctly
        assert state.pending_resolution["kind"] == "pay_cost"
        # float {G} from the Forest (605.3a: a mana ability resolves at once), then spend it
        mana.activate_mana_source(state, state.battlefield[0])
        mana.execute_pool_spend(state, "G")

        # payment complete -> pushed to the stack, not resolved yet -> back
        # to no pending resolution; the effect fires only once the stack resolves
        assert resolved_calls == []
        assert len(state.stack) == 1 and state.stack[0]["card_def"] is madness_card
        assert state.pending_resolution is None
        assert state.trigger_queue == []
        stack.resolve_top_of_stack(state)
        assert resolved_calls == ["Fake Madness Spell"]
        assert state.stack == []
        assert state.exile == []
    finally:
        registry.EFFECT_REGISTRY[EffectId.FILLER] = _filler_backup


def test_combat_and_state_based_mutual_damage_creature_death_handoff():
    # combat_damage_step hands off to check_state_based_actions: a blocked
    # attacker deals no damage to the opponent, but it and its blocker fight
    # each other, and a dying blocker's zones must resolve to the DEFENDER's
    # side, not state.active_idx (the attacker throughout combat_damage_step).
    # Two blocked pairs: pair A's attacker dies, blocker survives; pair B's
    # blocker dies, attacker survives -- death is checked per creature.
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

        combat.declare_attackers_step(state)
        combat.declare_attacker(state, attacker_a)
        combat.declare_attacker(state, attacker_b)
        combat.declare_attacker(state, unblocked_attacker)
        state.blocked_by[attacker_a] = [blocker_a]
        state.blocked_by[attacker_b] = [blocker_b]
        combat.combat_damage_step(state)

        assert state.players[1].life_total == 18  # only the unblocked attacker's power (2) gets through
        assert state.attackers == []
        # pair A: attacker_a (toughness 3) takes 5 -> dies; blocker_a (toughness 2) takes 1 -> survives
        assert attacker_a not in state.players[0].battlefield and blocker_a in state.players[1].battlefield
        assert [c.name for c in state.players[0].graveyard] == ["Attacker A"]  # its own owner's graveyard
        assert blocker_a.damage_marked == 1 and not blocker_a.tapped
        # pair B: attacker_b (toughness 3) takes 1 -> survives; blocker_b (toughness 2) takes 5 -> dies,
        # in the DEFENDER's zones, not state.active_idx (still the attacker here)
        assert attacker_b in state.players[0].battlefield and blocker_b not in state.players[1].battlefield
        assert [c.name for c in state.players[1].graveyard] == ["Blocker B"]
        assert attacker_b.damage_marked == 1 and attacker_b.tapped

        # a fresh combat resets blocked_by too, not just attackers
        combat.declare_attackers_step(state)
        assert state.blocked_by == {}

        # cleanup_step clears damage_marked for every permanent, both players
        assert blocker_a.damage_marked == 1
        state_based.cleanup_step(state)
        assert state.pending_resolution is None
        assert blocker_a.damage_marked == 0
        assert attacker_b.damage_marked == 0
    finally:
        registry.CARD_DEFS.clear()
        registry.CARD_DEFS.update(_card_defs_backup)
