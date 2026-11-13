"""Planted bugs for the ACTION-SPACE component (drl_env/, rl/decision/action_bridge.py and the engine helpers whose
answers the mask is built from). Maintained jointly with the repo owner: when behavior changes, add the mutation that
guards it, and treat any survivor as a hole in the tests, not a nuisance.

Run:  python tests/support/mutation_sweep.py action_space          (all)  |  ... action_space 3 7  |  ... --baseline

Each entry: (file under src/, exact old text -- must occur exactly once --, replacement, what bug it models).
Entries marked KNOWN-GAP are bugs in a family the oracle does not judge yet; they are expected to SURVIVE until that
layer exists, and are listed so the gap is visible rather than assumed away."""

COMPONENT = "action_space"
TIMEOUT = 900  # seconds per mutation; a survivor runs the whole component (~6 min), a hang is a catch

AS = "tests/component/action_space"
TEST_PATHS = [
    f"{AS}/test_a0_structure.py", f"{AS}/test_a4_copies_and_costs.py", f"{AS}/test_a2_oracle.py", f"{AS}/test_a1_purity.py",
    f"{AS}/test_a6_combat.py", f"{AS}/test_a7_scenarios.py", f"{AS}/test_a3_execute.py", f"{AS}/test_a5_mana.py",
    f"{AS}/test_action_table.py", f"{AS}/test_action_bridge.py", f"{AS}/test_action_table_order.py",
]
# One game per ordered pair and no disk cache (the cache is keyed to the REAL source, which a scratch copy would not match).
ENV = {"ACTION_CORPUS_GAMES": "1", "ACTION_CORPUS_NOCACHE": "1", "ACTION_PURITY_EVERY": "12"}

C = "drl_env/_actions_cast.py"
A = "drl_env/_actions_cast_altzone.py"
M = "drl_env/_actions_mana.py"
R = "drl_env/_actions_resolution.py"
T = "drl_env/_actions_table.py"
GM = "game/mana.py"
GT = "game/turn.py"

MUTATIONS = [
    # --- timing ---
    (GT, "            and not state.stack\n        )\n    if speed is Speed.YOUR_TURN:", "        )\n    if speed is Speed.YOUR_TURN:",
     "sorcery-speed actions allowed with something on the stack"),
    (GT, "            state.active_idx == state.turn_player_idx\n            and state.phase in SORCERY_SPEED_PHASES",
     "            state.phase in SORCERY_SPEED_PHASES", "sorcery-speed actions allowed on the opponent's turn"),
    (C, "    return game.turn.Speed.SORCERY\n\n\ndef _land_drop_legal", "    return game.turn.Speed.INSTANT\n\n\ndef _land_drop_legal",
     "every non-instant spell castable at instant speed"),
    (C, "            and state.lands_played_this_turn == 0\n            and any(c.name == name for c in state.hand)",
     "            and any(c.name == name for c in state.hand)", "unlimited land drops"),
    (C, "            and game.turn.speed_legal(state, game.turn.Speed.SORCERY)\n            and state.lands_played_this_turn == 0",
     "            and state.lands_played_this_turn == 0", "land drop ignores timing"),
    (R, "def _pass_legal(state):\n    if state.pending_resolution is not None:\n        return False",
     "def _pass_legal(state):\n    if False:\n        return False", "Pass offered while a decision is pending"),
    (R, "def _pass_legal(state):\n    if state.pending_resolution is not None:\n        return False",
     "def _pass_legal(state):\n    state.rng.random()\n    if state.pending_resolution is not None:\n        return False",
     "computing the mask consumes randomness"),
    # --- casting ---
    (C, "        if game.plan_payment(state, _effective_cast_cost(state, card_def)) is None:\n            return False\n        return extra_legal is None or extra_legal(state)",
     "        return extra_legal is None or extra_legal(state)", "a spell is offered without checking it can be paid for"),
    (C, "        if _hand_count_available(state, name) <= 0:\n            return False\n        card_def = game.CARD_DEFS[name]\n        if game.plan_payment(state, _effective_cast_cost",
     "        card_def = game.CARD_DEFS[name]\n        if game.plan_payment(state, _effective_cast_cost", "a spell is offered with no copy in hand"),
    (C, 'reduced["generic"] = max(0, reduced.get("generic", 0) - reduction)', 'reduced["generic"] = max(0, reduced.get("generic", 0) - 0)',
     "cost reduction ignored"),
    (C, 'reduced["generic"] = max(0, reduced.get("generic", 0) - reduction)', 'reduced["generic"] = max(0, reduced.get("generic", 0) - reduction - 1)',
     "cost reduction applied one too many times"),
    (C, "for n in range(min(max_n, len(state.graveyard)) + 1):", "for n in range(min(max_n, len(state.graveyard)) + 2):",
     "delve may exile one card more than the graveyard holds"),
    (C, "        if not any(c.name == name for c in state.hand):\n            return False\n        card_def = game.CARD_DEFS[name]\n        return game.plan_payment(state, card_def.extra[cost_key]) is not None",
     "        if not any(c.name == name for c in state.hand):\n            return False\n        card_def = game.CARD_DEFS[name]\n        return True", "cycling offered without paying its cost"),
    (C, "        if not any(c.name == name for c in state.graveyard):\n            return False\n        card_def = game.CARD_DEFS[name]\n        return game.plan_payment",
     "        card_def = game.CARD_DEFS[name]\n        return game.plan_payment", "graveyard ability offered with nothing in the graveyard"),
    (A, "        if not any(c.name == name for c in state.graveyard):\n            return False\n        if cost is not None",
     "        if cost is not None", "flashback offered with nothing in the graveyard"),
    (A, "            c.name == name and stamp is not None and stamp < state.turn_number", "            c.name == name and stamp is not None and stamp <= state.turn_number",
     "plot: cast from exile on the same turn"),
    (C, "if cd.name == name and state.turn_number <= deadline:", "if cd.name == name and state.turn_number <= deadline + 1:",
     "impulse window one turn too long"),
    (T, '("forestcycle", "Forestcycle"), ("cycle", "Cycle")', '("forestcycle", "Forestcycle"), ("cycleX", "Cycle")', "cycling rows silently not generated"),
    (T, "for slot in range(1, max_slot + 1):", "for slot in range(1, max_slot):", "one slot row too few per card"),
    # --- mana ---
    (M, "        or (state.phase in game.turn.SORCERY_SPEED_PHASES and state.active_idx == state.turn_player_idx)",
     "        or (state.phase in game.turn.SORCERY_SPEED_PHASES)", "speculative mana ability allowed on the opponent's turn"),
    (M, "    return game.payment_survives(\n        state, game.units_after(state, tapped=[source], produced=produced, own_ability_tap=source))",
     "    return True", "strand rule removed: a tap that makes the payment impossible is offered"),
    (R, '            and color in game.pool_spend_options(state)', "", "spend-from-pool offered for any color"),
    (GM, "if state.mana_pool.get(color, 0) > 0 and (remaining.get(color, 0) > 0 or generic_needed)",
     "if state.mana_pool.get(color, 0) > 0 and (remaining.get(color, 0) > 0)", "a pool color cannot be spent on generic"),
    (GM, 'if len(units) < sum(need.values()) + cost.get("generic", 0):\n        return False', "if False:\n        return False",
     "affordability ignores the total mana needed"),
    (GM, "if sum(need[c] for c in subset) > sum(1 for u in units if u & wanted):", "if sum(need[c] for c in subset) >= sum(1 for u in units if u & wanted):",
     "colored-pip availability off by one (too strict)"),
    (GM, "    if not tappable_for_mana(state, permanent):\n        return ()\n    spec = registry.EFFECT_REGISTRY.get(permanent.card_def.effect_id, {}).get(\"mana\")\n    granted",
     "    spec = registry.EFFECT_REGISTRY.get(permanent.card_def.effect_id, {}).get(\"mana\")\n    granted", "tapped sources still counted as mana supply"),
    (GM, "    return permanent.summoning_sick and not _has_haste(state, permanent)", "    return False", "summoning-sick creatures can tap for mana"),
    (GM, 'output = ["C", "C", "C"] if permanent.card_def.extra["tron_type"] == "Tower" else ["C", "C"]',
     'output = ["C", "C"] if permanent.card_def.extra["tron_type"] == "Tower" else ["C", "C", "C"]', "Tron output amounts swapped"),
    (GM, "output = [symbol] * sum(1 for p in state.battlefield if p not in exclude and predicate(p))", "output = [symbol]",
     "count-based mana source always produces one"),
    # --- purity / hygiene ---
    (C if False else "drl_env/_actions_common.py", "    return sum(1 for c in state.hand if c.name == name)",
     "    state.hand.sort(key=lambda c: c.name)\n    return sum(1 for c in state.hand if c.name == name)", "computing the mask reorders the hand"),
    # --- bridge dispatch ---
    ("rl/decision/action_bridge.py", '"choose_any_target", "choose_graveyard_card", "choose_cast_copy", "choose_stack_target",',
     '"choose_any_target", "choose_graveyard_card", "choose_stack_target",', "any_pointer_legal forgets a pointer kind"),
    ("rl/decision/action_bridge.py", '"declare_blockers", "choose_permanent", "choose_opponent_permanent", "assign_combat_damage",\n        "choose_any_target",\n    ) or',
     '"declare_blockers", "choose_permanent", "choose_opponent_permanent", "assign_combat_damage",\n    ) or', "pointer_kind forgets a pointer kind"),
    # --- which copy / activation cost (the 2026-11-12 fixes) ---
    (GM, "        for player in state.players:\n            for aura in player.battlefield:", "        for player in state.players[:1]:\n            for aura in player.battlefield:",
     "F-01 back: only the first player's Auras count for a land's mana"),
    (GM, "        for player in state.players:\n            for aura in player.battlefield:", "        for player in [state]:\n            for aura in player.battlefield:",
     "F-01 back: only the ACTIVE player's Auras count for a land's mana"),
    (C, "        if taps and (p.tapped or game.tap_summoning_locked(state, p)):\n            continue", "        if False:\n            continue",
     "an ability whose cost includes {T} works on a tapped or summoning-sick permanent"),
    (C, "        if taps and (p.tapped or game.tap_summoning_locked(state, p)):\n            continue", "        if p.tapped or game.tap_summoning_locked(state, p):\n            continue",
     "an ability WITHOUT {T} in its cost is refused on a tapped permanent (the old bug)"),
    (C, "        if ability_legal is not None and not ability_legal(state, p):\n            continue", "        if False:\n            continue",
     "the card's own activation precondition is ignored for the copy"),
    (C, "        return bool(_activation_copies(state, name, taps, ability_legal))",
     "        return bool(_activation_copies(state, name, taps)[:1]) and ability_legal(state, next(p for p in state.battlefield if p.card_def.name == name))",
     "only the FIRST copy is asked whether it can activate (F-02 back)"),
    (C, "    if len(choices) == 1:\n        proceed(state, choices[0])\n        return", "    if True:\n        proceed(state, choices[0])\n        return",
     "activation never asks which of several different copies"),
    (M, "        if len(distinct_copies(state, candidates)) == 1:", "        if True:", "tap never asks which of several different permanents"),
    (M, "        candidates = [p for p in _find_mana_sources(state, name, color) if _tap_survives(state, p, color)]", "        candidates = _find_mana_sources(state, name, color)",
     "the copy choice may offer a permanent whose tap strands the payment"),
    (M, "        return any(_tap_survives(state, p, color) for p in _cached_mana_sources(state, name, color))",
     "        return any(_tap_survives(state, p, color) for p in _cached_mana_sources(state, name, color)[:1])",
     "only the first copy's tap is considered for legality (F-02 back)"),
    ("rl/decision/action_bridge.py", "                if p is not None and any(p is c for c in mana_sub[\"candidates\"]):", "                if False:",
     "choose_source pointer mask offers nothing (all-False)"),
    ("rl/decision/action_bridge.py", "        return mana_sub[\"stage\"] in _POINTER_MANA_STAGES\n    pending = state.pending_resolution\n    if pending is None:\n        return state.phase",
     "        return mana_sub[\"stage\"] == \"choose_target\"\n    pending = state.pending_resolution\n    if pending is None:\n        return state.phase",
     "any_pointer_legal forgets the choose_source stage"),
    (T, "        if mana_sub is not None:\n            mana_sub_stage = mana_sub[\"stage\"]", "        if mana_sub is not None and sub_rows:\n            mana_sub_stage = mana_sub[\"stage\"]",
     "a sub-decision does not suppress the fixed rows of a deck without sub-decision buttons"),
    ("game/catalog/colorless_cards.py", "\"taps\": True, \"resolve\": lambda state, permanent: activate_candy_trail_sac(state, permanent)",
     "\"taps\": False, \"resolve\": lambda state, permanent: activate_candy_trail_sac(state, permanent)", "Candy Trail declared as not tapping (it does)"),
    ("game/catalog/red_cards.py", "\"taps\": False, \"resolve\": lambda state, permanent: activate_reckless_lackey_sac(state, permanent)",
     "\"taps\": True, \"resolve\": lambda state, permanent: activate_reckless_lackey_sac(state, permanent)", "Reckless Lackey declared as tapping (it does not)"),
    # --- combat (A6) and payment search (A5) ---
    ("game/effects/combat.py", "        permanent.card_type == CardType.CREATURE and not permanent.tapped\n        and not permanent.card_def.extra.get(\"defender\", False)",
     "        permanent.card_type == CardType.CREATURE\n        and not permanent.card_def.extra.get(\"defender\", False)", "a tapped creature may attack"),
    ("game/effects/combat.py", "        and not permanent.card_def.extra.get(\"defender\", False)\n        and permanent not in state.attackers",
     "        and permanent not in state.attackers", "a Defender may attack"),
    ("game/effects/combat.py", "        and (not permanent.summoning_sick or stats.has_haste(state, permanent))\n    )", "    )",
     "a summoning-sick creature without haste may attack"),
    ("game/effects/combat.py", "        and permanent not in state.attackers\n", "", "a creature may be declared as an attacker twice"),
    ("game/effects/combat.py", "    return stats.has_keyword(state, blocker, \"flying\") or stats.has_keyword(state, blocker, \"reach\")",
     "    return stats.has_keyword(state, blocker, \"flying\")", "reach no longer lets a creature block a flier"),
    ("game/effects/combat.py", "    if not (permanent.card_type == CardType.CREATURE and not permanent.tapped\n            and permanent not in _blocked_by_creatures(state.opponent.blocked_by)):",
     "    if not (permanent.card_type == CardType.CREATURE\n            and permanent not in _blocked_by_creatures(state.opponent.blocked_by)):", "a tapped creature may block"),
    ("game/effects/combat.py", "    if not (permanent.card_type == CardType.CREATURE and not permanent.tapped\n            and permanent not in _blocked_by_creatures(state.opponent.blocked_by)):",
     "    if not (permanent.card_type == CardType.CREATURE and not permanent.tapped):", "a creature may block twice"),
    ("game/effects/combat.py", "        len(blockers) == 1 and stats.has_keyword(state, attacker, \"menace\")\n        for attacker, blockers in state.opponent.blocked_by.items()",
     "        len(blockers) == 2 and stats.has_keyword(state, attacker, \"menace\")\n        for attacker, blockers in state.opponent.blocked_by.items()", "menace needs the wrong number of blockers"),
    ("game/effects/combat.py", "    if state.active_idx != state.turn_player_idx:\n        return False\n    return any(\n        p.flags.get(\"goaded_by\")", "    return False\n    return any(\n        p.flags.get(\"goaded_by\")", "goad never binds"),
    ("game/effects/combat.py", "    if not stats.has_keyword(state, permanent, \"vigilance\"):\n        permanent.tapped = True", "    if True:\n        permanent.tapped = True", "vigilance ignored when declaring an attacker"),
    ("game/effects/combat.py", "    return 1 if deathtouch else max(toughness - damage_marked, 0)", "    return 1 if deathtouch else max(toughness, 0)", "lethal damage ignores damage already marked"),
    ("rl/decision/action_bridge.py", "amounts.get(p, 0) < lethal_by_blocker[p]", "amounts.get(p, 0) <= lethal_by_blocker[p]", "damage may overkill a blocker by one"),
    ("game/resolution/handlers_combat.py", "        pending[\"opponent\"] += pending[\"remaining\"]", "        pending[\"opponent\"] += 0", "trample damage is lost instead of going to the player"),
    ("drl_env/_actions_combat.py", "    return not game.menace_block_incomplete(state)", "    return True", "Done blocking offered while a menace attacker has a lone blocker"),
    (GM, "    return owed is None or can_pay(units, owed)", "    return True", "payment_survives always true: nothing is ever refused as stranding"),
    (GM, "        elif granted:", "        elif False:", "F-03 back: a grant on a multi-symbol source is dropped from affordability"),
    (GM, "for combo in product(*(c.alternatives for c in choices)))", "for combo in product(*(c.alternatives[:1] for c in choices)))",
     "an either-or supply unit is only ever used its first way"),
    # --- KNOWN-GAP: families the oracle does not judge yet (decision buttons) ---
    (R, "    return pending is not None and pending[\"kind\"] == \"ancient_stirrings\"\n\n\n_decline_legal._pending_gate",
     "    return False\n\n\n_decline_legal._pending_gate", "KNOWN-GAP: Ancient Stirrings' Decline button never legal"),
    (R, "    return pending is not None and pending[\"kind\"] == \"may_transform\"", "    return False", "may-transform buttons never legal (caught: real games then reach an all-False mask)"),
]
