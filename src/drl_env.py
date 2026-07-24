"""Gymnasium environment adapter wrapping game.py's simulator.

Not one of the 4 independent pieces (see DRL_PLAN.md) -- this is assembly
logic that combines a simulator (game.py, imported and never modified)
with an injected reward function (rewards.py's contract) into a
Gym-compatible interface a DRL model can train against.
"""

import os
import random

import numpy as np

import game

# ---------------------------------------------------------------------------
# D2.2 -- Action table (MULTI_DECK_PLAN.md Phase M4e: generated from a
# decklist + game.EFFECT_REGISTRY instead of hand-typed -- this, plus the
# pending-resolution machinery in game.py, is what makes a deck built
# entirely from already-implemented cards need zero new code here.
#
# Categories, in table order:
#   A. Play land: <name>            -- one per distinct land name
#   B. Cast <name>                  -- one per card with a registry "cast" entry
#   C. Activate <name> (<ability>)  -- one per registered activated ability
#   D. Forestcycle <name>           -- one per registry "forestcycle" entry
#   E. Pass
#   F. Choose: <name>               -- shared across every pending-resolution
#      kind that picks a plain card name (paying with a fixed/Tron mana
#      source, search_fetch, ancient_stirrings, and scry/surveil's ordering
#      phase), dispatched by pending_resolution["kind"]
#   G. Choose: <name> as <color>    -- flexible/filter mana sources during
#      a pay_cost resolution specifically (the only kind needing a color)
#   H. Keep / Dispose (scry/surveil)
#   I. Decline (Ancient Stirrings)
#   J. Abandon payment -- cancels a pending pay_cost resolution outright,
#      untapping everything tapped so far. Without this, tapping a
#      flexible/filter source for the wrong color could strand a game
#      with an unpayable remaining cost and zero legal actions -- see
#      game.abandon_pay_cost's docstring.
#   K. Choose target: <name> (slot k) -- exact-(name, slot)-addressed, the
#      "choose_permanent" resolution's own actions (Aura enchant-targets,
#      Crop Rotation's sacrifice cost, land bounce) -- NOT category F,
#      unlike before: two same-named permanents stop being interchangeable
#      the instant an Aura attaches to only one of them, and cast_aura's
#      cast-time-target/resolve-time-fizzle contract depends on knowing
#      exactly which physical permanent was chosen (docs/MULTIPLAYER_GAPS.md
#      "Permanent identity").
#
# spy_combo deck additions: B also covers Winding Way's modal cast (2
# actions, one per mode), Land Grant's free alt-cost, Dread Return's
# Flashback (cast from the graveyard), and Nyxborn Hydra's own
# "x_cast_modes" (one action per (mode, X) pair -- its normal creature cast
# and Bestow, each its own cost distinct from card_def.cast_cost, see
# _x_cast_legal/_x_cast_execute/_x_precast_choice_execute); C also covers
# non-mana activated abilities (Quirion Ranger, Pinnacle Kill-Ship's own
# Station); F/H also cover select_to_hand's own Keep/Bottom pair and its
# ordering phase (Lead the Stampede) and an optional search's Decline
# (Gatecreeper Vine) alongside Ancient Stirrings'; K also covers Pinnacle
# Kill-Ship's own opponent-facing ETB target (choose_opponent_permanent,
# same category as blocking's own cross-player targeting -- correctly a
# no-op in every current 1-player Tron config, since the underlying
# resolution auto-completes with no legal target).
# ---------------------------------------------------------------------------

def _cast_speed(card_def, spec):
    """The game.turn.Speed a cast-like action (cast/cast_modes/alt_cast/
    flashback/plot -- each derived independently, once per action, in
    build_action_table) resolves to: an explicit "speed" key in that
    specific spec dict if a card ever needs to override it, else
    Speed.INSTANT for an actual CardType.INSTANT card (its type line
    already implies instant speed -- no per-card tag needed for the
    common case), else Speed.SORCERY -- the default for every creature/
    artifact/enchantment/sorcery/land absent a Flash-like exception, per
    real Magic's own casting-speed rule. Flashback/Plot deliberately have
    no override in this cube today, so they fall through to the same
    CardType-derived answer the card's normal cast would -- correct per
    real Magic (Flashback/Plot follow the same timing as the card
    itself), not just a convenient default."""
    override = spec.get("speed")
    if override is not None:
        return override
    if card_def.card_type == game.CardType.INSTANT:
        return game.turn.Speed.INSTANT
    return game.turn.Speed.SORCERY


_GATE_NO_PENDING = object()  # sentinel: this closure's own first check is "state.pending_resolution is None" -- see legal_action_mask's own docstring


def _land_drop_legal(name):
    def legal(state):
        return (
            state.pending_resolution is None
            # Real Magic: playing a land is always sorcery-speed (no
            # per-card override exists in this cube) -- speed_legal's own
            # Speed.SORCERY branch already requires state.active_idx ==
            # state.turn_player_idx (docs/PRIORITY_PLAN.md), so this alone
            # already refuses a land drop offered to the non-turn player
            # during a priority window, with no separate check needed here.
            and game.turn.speed_legal(state, game.turn.Speed.SORCERY)
            and state.lands_played_this_turn == 0
            and any(c.name == name for c in state.hand)
        )
    legal._pending_gate = _GATE_NO_PENDING
    return legal


def _land_drop_execute(name):
    def execute(state):
        game.play_land_from_hand(state, game.CARD_DEFS[name])
    return execute


def _hand_count_available(state, name):
    """How many copies of `name` in state.hand are actually still castable
    right now. A cast-like resolve function only removes its card from hand
    when it finally RUNS -- which, since push_to_stack (game.effects.stack)
    defers it, can be well after the cost is paid -- so a copy already
    pushed onto state.stack (paid for, awaiting resolution) is still
    physically present in state.hand but must not count as available: an
    action mask that let the model "cast" that same physical copy a second
    time would push a second stack entry referencing it, and crash once
    both entries eventually try to remove it from hand. Sorcery-speed cards
    are already safe from this via speed_legal's own stack-emptiness check
    (nothing sorcery-speed is ever legal again once anything -- even
    itself -- sits unresolved on the stack); this only actually matters for
    Speed.INSTANT cards, which the stack never blocks re-casting of, but is
    correct (a no-op) to apply uniformly rather than special-casing speed
    here too."""
    hand_count = sum(1 for c in state.hand if c.name == name)
    stacked_count = sum(1 for entry in state.stack if entry["card_def"].name == name and entry["reserves_hand_card"])
    return hand_count - stacked_count


def _cast_legal(name, extra_legal, speed):
    def legal(state):
        if state.pending_resolution is not None:
            return False
        if not game.turn.speed_legal(state, speed):
            return False
        if _hand_count_available(state, name) <= 0:
            return False
        card_def = game.CARD_DEFS[name]
        if game.plan_payment(state, card_def.cast_cost) is None:
            return False
        return extra_legal is None or extra_legal(state)
    legal._pending_gate = _GATE_NO_PENDING
    return legal


def _cast_execute(name, resolve):
    def execute(state):
        card_def = game.CARD_DEFS[name]
        def _after_pay(s):
            # Fires only once mana is actually, irreversibly paid -- NOT
            # the instant this cast is announced. "Abandon payment" can
            # cancel a pending pay_cost resolution outright (see
            # game.abandon_pay_cost), so firing this any earlier let a
            # cast be announced, the trigger collected for free, and
            # payment then abandoned -- repeatable forever. Real MTG
            # (601.2i): a spell isn't "cast" until its cost is paid;
            # failing/declining to pay reverses the whole action as if it
            # was never started, so a "whenever you cast" trigger (e.g.
            # Guttersnipe) never fires either. Every cast path (this one,
            # alt_cast, flashback, plot-from-exile below) fires it
            # identically once its own cost is similarly locked in.
            game.on_cast_trigger(s, card_def)
            # Once mana is fully paid, the spell is "cast" but not yet
            # resolved -- push it onto state.stack (game.push_to_stack)
            # instead of resolving immediately, so the model can respond
            # with another instant-speed action first. Something (a
            # "Pass" -- see game.turn._run_turn_gen) has to actually
            # resolve it later.
            game.push_to_stack(s, card_def, resolve)
        game.begin_pay_cost(state, card_def.cast_cost, on_complete=_after_pay)
    return execute


def _precast_choice_execute(name, resolve):
    """Cast-like execute for a card whose own `resolve` needs to settle
    something -- a real target (cast_aura's "enchant target creature"), or
    an additional cost (cast_crop_rotation's "sacrifice a land") -- BEFORE
    the spell is fully cast, not once it resolves off the stack. Real MTG:
    both targets and additional costs are locked in as part of casting the
    spell, never deferred to resolution; only the spell's own EFFECT waits
    on the stack. Unlike _cast_execute, `resolve` is called directly as
    pay_cost's on_complete and is responsible for its own game.push_to_stack
    call (having already run whatever precast resolution it needs -- see
    cast_aura/cast_crop_rotation's own docstrings for each one's exact
    contract) instead of this function pushing to the stack generically on
    its behalf. Selected via each registry cast/cast_modes spec's own
    "precast_choice": True flag (build_action_table)."""
    def execute(state):
        card_def = game.CARD_DEFS[name]
        def _after_pay(s):
            game.on_cast_trigger(s, card_def)  # only once mana is irreversibly paid -- see _cast_execute's own comment
            resolve(s, card_def)
        game.begin_pay_cost(state, card_def.cast_cost, on_complete=_after_pay)
    return execute


def _x_cast_legal(name, cost, extra_legal, speed):
    """Like _cast_legal, but against an explicit `cost` instead of
    card_def.cast_cost -- one X value's own concrete cost (an
    "x_cast_modes" registry entry's own per-mode base cost plus that X's
    own generic, build_action_table's own loop below), same "a real cost
    distinct from card_def.cast_cost" shape _plot_legal/_omen_cast_legal
    already use for their own alternate costs, not a param bolted onto
    _cast_legal itself."""
    def legal(state):
        if state.pending_resolution is not None:
            return False
        if not game.turn.speed_legal(state, speed):
            return False
        if _hand_count_available(state, name) <= 0:
            return False
        if game.plan_payment(state, cost) is None:
            return False
        return extra_legal is None or extra_legal(state)
    legal._pending_gate = _GATE_NO_PENDING
    return legal


def _x_cast_execute(name, cost, resolve):
    """Same shape as _cast_execute, against an explicit `cost`."""
    def execute(state):
        card_def = game.CARD_DEFS[name]
        def _after_pay(s):
            game.on_cast_trigger(s, card_def)
            game.push_to_stack(s, card_def, resolve)
        game.begin_pay_cost(state, cost, on_complete=_after_pay)
    return execute


def _x_precast_choice_execute(name, cost, resolve):
    """Same shape as _precast_choice_execute, against an explicit `cost` --
    Nyxborn Hydra's own Bestow mode needs both: a real target chosen before
    the stack (cast_aura, same as Rancor/Ancestral Mask/Ethereal Armor) AND
    its own X-dependent cost distinct from the card's normal cast_cost."""
    def execute(state):
        card_def = game.CARD_DEFS[name]
        def _after_pay(s):
            game.on_cast_trigger(s, card_def)
            resolve(s, card_def)
        game.begin_pay_cost(state, cost, on_complete=_after_pay)
    return execute


def _activate_legal(name, cost_key, speed):
    def legal(state):
        if state.pending_resolution is not None:
            return False
        if not game.turn.speed_legal(state, speed):
            return False
        p = next((p for p in state.battlefield if p.card_def.name == name and not p.tapped), None)
        return p is not None and game.plan_payment(state, p.card_def.extra[cost_key]) is not None
    legal._pending_gate = _GATE_NO_PENDING
    return legal


def _activate_execute(name, cost_key, resolve):
    def execute(state):
        p = next(p for p in state.battlefield if p.card_def.name == name and not p.tapped)
        cost = p.card_def.extra[cost_key]
        game.begin_pay_cost(state, cost, on_complete=lambda s: resolve(s, p))
    return execute


def _forestcycle_legal(name, cost_key):
    def legal(state):
        if state.pending_resolution is not None:
            return False
        if not any(c.name == name for c in state.hand):
            return False
        card_def = game.CARD_DEFS[name]
        return game.plan_payment(state, card_def.extra[cost_key]) is not None
    legal._pending_gate = _GATE_NO_PENDING
    return legal


def _forestcycle_execute(name, cost_key, resolve):
    def execute(state):
        card_def = game.CARD_DEFS[name]
        game.begin_pay_cost(state, card_def.extra[cost_key], on_complete=lambda s: resolve(s, card_def))
    return execute


def _graveyard_ability_legal(name, cost_key):
    """Bramble Wurm's own "{2}{G}, Exile this card from your graveyard:
    gain 5 life" -- same hand-zone/cost-key/card_def shape as
    _forestcycle_legal above, just sourced from state.graveyard instead of
    state.hand (a graveyard activated ability, unlike Flashback, never
    recasts the spell -- resolve just runs the ability directly, no
    push_to_stack, matching every other activated ability in this engine).
    No speed gate: every existing activated ability defaults to "any
    time" absent an explicit override (see build_action_table's own
    activated_abilities loop), and this one has none."""
    def legal(state):
        if state.pending_resolution is not None:
            return False
        if not any(c.name == name for c in state.graveyard):
            return False
        card_def = game.CARD_DEFS[name]
        return game.plan_payment(state, card_def.extra[cost_key]) is not None
    legal._pending_gate = _GATE_NO_PENDING
    return legal


def _graveyard_ability_execute(name, cost_key, resolve):
    def execute(state):
        card_def = game.CARD_DEFS[name]
        game.begin_pay_cost(state, card_def.extra[cost_key], on_complete=lambda s: resolve(s, card_def))
    return execute


def _pass_legal(state):
    return state.pending_resolution is None


_pass_legal._pending_gate = _GATE_NO_PENDING


def _pass_execute(state):
    pass  # handled by DeckEnv.step() itself, not via this table


def _choose_name_options(state):
    """Plain (uncolored) 'Choose: X' names currently legal, given whatever
    kind of pending resolution -- if any -- is active. "choose_permanent"
    is NOT handled here -- see _choose_permanent_legal/_choose_permanent_
    execute below: it needs exact (name, slot) addressing (docs/
    MULTIPLAYER_GAPS.md's "Permanent identity"), same as
    "choose_opponent_permanent" already gets, not this generic by-name
    dispatch."""
    pending = state.pending_resolution
    if pending is None:
        return []
    kind = pending["kind"]
    if kind == "pay_cost":
        return [n for n, c, f in _cached_tap_cost_options(state) if c is None and not f]
    if kind == "search_fetch":
        return game.search_fetch_options(state)
    if kind == "choose_graveyard_card":
        return game.choose_graveyard_card_options(state)
    if kind == "sacrifice":
        return game.sacrifice_options(state)
    if kind == "discard":
        return game.discard_options(state)
    if kind == "discard_or_sacrifice":
        # Only the DISCARD half reuses this generic "Choose: X" dispatch
        # (bare hand-card names, same as plain "discard") -- the
        # sacrifice half gets its own distinctly-labeled actions instead
        # (build_action_table's own "Sacrifice (cost): X" loop) precisely
        # to avoid ambiguity if a hand card and a battlefield land ever
        # share a name (e.g. a Mountain in hand while Mountains are also
        # in play) -- two different action strings, never one bare name
        # that could mean either.
        return game.discard_or_sacrifice_discard_options(state)
    if kind == "mulligan_bottom":
        return game.bottom_options(state)
    if kind == "ancient_stirrings":
        return [n for n in game.ancient_stirrings_options(state) if n != "decline"]
    if kind == "malevolent_rumble":
        return [n for n in game.malevolent_rumble_options(state) if n != "decline"]
    if kind in ("scry", "surveil") and pending["ordered"] is not None:
        return game.scry_surveil_options(state)
    if kind == "select_to_hand" and pending["ordered"] is not None:
        return game.select_to_hand_options(state)  # ordering phase only -- "keep"/"bottom" are their own actions
    if kind == "order_triggers":
        return game.order_triggers_options(state)  # docs/PRIORITY_PLAN.md item 1
    return []


def _choose_name_legal(name):
    def legal(state):
        return name in _choose_name_options(state)
    # Matches _choose_name_options' own dispatch table above exactly --
    # every pending kind that function ever returns a non-empty list for.
    legal._pending_gate = frozenset({
        "pay_cost", "search_fetch", "choose_graveyard_card", "sacrifice", "discard", "discard_or_sacrifice",
        "mulligan_bottom", "ancient_stirrings", "malevolent_rumble", "scry", "surveil", "select_to_hand",
        "order_triggers",
    })
    return legal


def _choose_name_execute(name):
    def execute(state):
        kind = state.pending_resolution["kind"]
        if kind == "pay_cost":
            game.execute_tap_cost_option(state, name, None, False)
        elif kind == "search_fetch":
            game.execute_search_fetch_option(state, name)
        elif kind == "choose_graveyard_card":
            game.execute_choose_graveyard_card_option(state, name)
        elif kind == "sacrifice":
            game.execute_sacrifice_option(state, name)
        elif kind == "discard":
            game.execute_discard_option(state, name)
        elif kind == "discard_or_sacrifice":
            game.execute_discard_or_sacrifice_option(state, "discard", name)
        elif kind == "mulligan_bottom":
            game.execute_bottom_option(state, name)
        elif kind == "ancient_stirrings":
            game.execute_ancient_stirrings_option(state, name)
        elif kind == "malevolent_rumble":
            game.execute_malevolent_rumble_option(state, name)
        elif kind == "select_to_hand":
            game.execute_select_to_hand_option(state, name)  # ordering phase only
        elif kind == "order_triggers":
            game.execute_order_triggers_option(state, name)  # docs/PRIORITY_PLAN.md item 1
        else:  # scry / surveil, ordering phase
            game.execute_scry_surveil_option(state, name)
    return execute


def _choose_name_color_options(state):
    """(name, color) pairs currently legal via tap_cost_options's
    flexible/filter entries -- the only pending-resolution kind that ever
    needs a color qualifier."""
    pending = state.pending_resolution
    if pending is None or pending["kind"] != "pay_cost":
        return []
    return [(n, c) for n, c, _f in _cached_tap_cost_options(state) if c is not None]


def _choose_name_color_legal(name, color):
    def legal(state):
        return (name, color) in _choose_name_color_options(state)
    legal._pending_gate = frozenset({"pay_cost"})
    return legal


def _choose_name_color_execute(name, color):
    def execute(state):
        is_filter = next(f for n, c, f in game.tap_cost_options(state) if n == name and c == color)
        game.execute_tap_cost_option(state, name, color, is_filter)
    return execute


def _attack_legal(name, slot):
    """Legal only during Phase.DECLARE_ATTACKERS, and only for the true
    turn owner (state.active_idx == state.turn_player_idx,
    docs/PRIORITY_PLAN.md) -- declaring an attacker is a turn-based
    special action, not a priority action, so the non-turn player must
    never be allowed to declare one just because state.phase (a single
    shared field describing the TURN's phase) happens to match during
    their own priority window. And only if the specific physical
    permanent occupying this (name, slot) -- docs/COMBAT_PLAN.md's
    permanent-identity design -- is currently attack-eligible
    (game.creature_attack_eligible): untapped, and not summoning sick
    unless it has haste. Attacking stays fully optional: a model can leave
    any subset of eligible creatures back, Pass with zero attackers
    declared is still legal (same as always -- state.attackers simply
    starts, and can stay, empty for this turn)."""
    def legal(state):
        if state.phase is not game.turn.Phase.DECLARE_ATTACKERS:
            return False
        if state.active_idx != state.turn_player_idx:
            return False
        p = _cached_battlefield_lookup(state).get((name, slot))
        return p is not None and game.creature_attack_eligible(state, p)
    return legal


def _attack_execute(name, slot):
    """Declares the specific physical permanent occupying this (name,
    slot) as an attacker -- unlike the old arbitrary-pick-by-name
    behavior, this lets a model distinguish an Aura-enchanted copy
    (different effective power) from a plain one of the same name."""
    def execute(state):
        permanent = next(
            p for p in state.battlefield
            if p.card_def.name == name and p.slot == slot and game.creature_attack_eligible(state, p)
        )
        game.declare_attacker(state, permanent)
    return execute


def _choose_permanent_legal(name, slot):
    """The "choose_permanent" resolution's action-table half (Aura
    enchant-targets, Crop Rotation's sacrifice cost, land bounce) -- legal
    only while that kind is pending and (name, slot) is one of its own
    current options. Exact (name, slot) addressing, same reason
    _choose_opponent_permanent_legal below needs it (docs/
    MULTIPLAYER_GAPS.md's "Permanent identity") -- a plain by-name "Choose:
    X" can't tell two same-named permanents apart, and cast_aura's whole
    fizzle-on-invalid-target contract depends on knowing exactly which one
    was chosen."""
    def legal(state):
        pending = state.pending_resolution
        return (
            pending is not None and pending["kind"] == "choose_permanent"
            and (name, slot) in game.choose_permanent_options(state)
        )
    legal._pending_gate = frozenset({"choose_permanent"})
    return legal


def _choose_permanent_execute(name, slot):
    def execute(state):
        game.execute_choose_permanent_option(state, name, slot)
    return execute


def _choose_opponent_permanent_legal(name, slot):
    """The general cross-player targeting primitive's action-table half
    (docs/COMBAT_PLAN.md) -- legal only while a "choose_opponent_permanent"
    resolution is pending and (name, slot) is one of its own current
    options. Only ever correct when the referencing side is already the
    active perspective (game.begin_choose_opponent_permanent's own
    docstring) -- blocking's own defender-decision channel is what
    guarantees that, not this function."""
    def legal(state):
        pending = state.pending_resolution
        return (
            pending is not None and pending["kind"] == "choose_opponent_permanent"
            and (name, slot) in game.choose_opponent_permanent_options(state)
        )
    legal._pending_gate = frozenset({"choose_opponent_permanent"})
    return legal


def _choose_opponent_permanent_execute(name, slot):
    def execute(state):
        game.execute_choose_opponent_permanent_option(state, name, slot)
    return execute


def _assign_blocker_legal(name, slot):
    """One "Assign Blocker: <name> (slot j)" action (docs/COMBAT_PLAN.md's
    blocking design) -- legal only while a "declare_blockers" resolution
    is pending (game.turn._declare_blockers_gen has already flipped
    state.active_idx to the defender by the time this is ever checked)
    and the specific physical permanent at this (name, slot) is currently
    block-eligible (game.creature_block_eligible): untapped, not already
    assigned to block something else this combat. Unlike attacking,
    neither summoning sickness nor Defender excludes a blocker -- see
    creature_block_eligible's own docstring for why."""
    def legal(state):
        pending = state.pending_resolution
        if pending is None or pending["kind"] != "declare_blockers":
            return False
        p = _cached_battlefield_lookup(state).get((name, slot))
        return p is not None and game.creature_block_eligible(state, p)
    legal._pending_gate = frozenset({"declare_blockers"})
    return legal


def _assign_blocker_execute(name, slot):
    """Parks the specific physical permanent at this (name, slot) as a
    blocker, then hands off to game.declare_blocker_assignment, which
    nests a cross-player choose_opponent_permanent sub-resolution to pick
    which of the attacker's declared, not-yet-blocked attackers it
    blocks -- restricted by extra_predicate to attackers this specific
    blocker is actually allowed to block: flying's own restriction
    (docs/COMBAT_PLAN.md step 7) means an attacker with flying can only be
    chosen here if `blocker` itself also has flying (game.has_keyword --
    resolution.py can't compute this itself, see declare_blocker_
    assignment's own docstring for why the predicate has to come from
    here instead). Once that completes, re-opens begin_declare_blockers
    (via the captured outer on_complete) so the defender can assign
    another blocker or choose Done -- same nested-callback shape
    execute_madness_cast already uses for its own multi-step chain."""
    def execute(state):
        blocker = next(
            p for p in state.battlefield
            if p.card_def.name == name and p.slot == slot and game.creature_block_eligible(state, p)
        )
        outer_on_complete = state.pending_resolution["on_complete"]

        def _blockable_by(attacker):
            return not game.has_keyword(state, attacker, "flying") or game.has_keyword(state, blocker, "flying")

        game.declare_blocker_assignment(
            state, blocker, on_complete=lambda s: game.begin_declare_blockers(s, outer_on_complete),
            extra_predicate=_blockable_by,
        )
    return execute


def _done_blocking_legal(state):
    pending = state.pending_resolution
    return pending is not None and pending["kind"] == "declare_blockers"


_done_blocking_legal._pending_gate = frozenset({"declare_blockers"})


def _done_blocking_execute(state):
    game.complete_resolution(state)


def _pool_spend_legal(color):
    def legal(state):
        return (
            state.pending_resolution is not None
            and state.pending_resolution["kind"] == "pay_cost"
            and color in game.pool_spend_options(state)
        )
    legal._pending_gate = frozenset({"pay_cost"})
    return legal


def _pool_spend_execute(color):
    def execute(state):
        game.execute_pool_spend(state, color)
    return execute


def _keep_dispose_legal(state):
    pending = state.pending_resolution
    return pending is not None and pending["kind"] in ("scry", "surveil") and bool(pending["remaining"])


_keep_dispose_legal._pending_gate = frozenset({"scry", "surveil"})


def _keep_execute(state):
    game.execute_scry_surveil_option(state, "keep")


def _dispose_execute(state):
    game.execute_scry_surveil_option(state, "dispose")


def _mulligan_decision_legal(state):
    pending = state.pending_resolution
    return pending is not None and pending["kind"] == "mulligan_decision"


_mulligan_decision_legal._pending_gate = frozenset({"mulligan_decision"})


def _mulligan_take_legal(state):
    # Caps London Mulligan at HAND_SIZE_LIMIT (7): execute_mulligan_take
    # never runs out of library to bound itself (mulliganed cards go back
    # into the library before the redraw), so a deterministically-evaluated
    # policy that argmaxes to "Mulligan" regardless of hand quality would
    # otherwise retake it forever -- confirmed live (a barely-trained
    # MaskablePPO checkpoint did exactly this during evaluate_two_player).
    # Past 7 mulligans, "Keep hand" becomes the only legal action, same
    # illegal-action-gets-substituted fallback every other action already
    # relies on (see model_choose_action).
    return _mulligan_decision_legal(state) and state.mulligans_taken < game.HAND_SIZE_LIMIT


_mulligan_take_legal._pending_gate = frozenset({"mulligan_decision"})


def _mulligan_keep_execute(state):
    game.execute_mulligan_keep(state)


def _mulligan_take_execute(state):
    game.execute_mulligan_take(state)


def _decline_legal(state):
    pending = state.pending_resolution
    return pending is not None and pending["kind"] == "ancient_stirrings"


_decline_legal._pending_gate = frozenset({"ancient_stirrings"})


def _decline_execute(state):
    game.execute_ancient_stirrings_option(state, "decline")


def _decline_malevolent_rumble_legal(state):
    pending = state.pending_resolution
    return pending is not None and pending["kind"] == "malevolent_rumble"


_decline_malevolent_rumble_legal._pending_gate = frozenset({"malevolent_rumble"})


def _decline_malevolent_rumble_execute(state):
    game.execute_malevolent_rumble_option(state, "decline")


def _abandon_payment_legal(state):
    pending = state.pending_resolution
    return pending is not None and pending["kind"] == "pay_cost"


_abandon_payment_legal._pending_gate = frozenset({"pay_cost"})


def _abandon_payment_execute(state):
    game.abandon_pay_cost(state)


# ---------------------------------------------------------------------------
# spy_combo deck additions: select_to_hand's own fixed actions (Lead the
# Stampede), an optional-search decline, non-mana activated abilities
# (Quirion Ranger), Land Grant's free alt-cost, Dread Return's Flashback,
# and Winding Way's modal cast. None of these fire for Tron cards -- each
# is gated on a registry key no Tron EffectId sets.
# ---------------------------------------------------------------------------

def _select_to_hand_keep_legal(state):
    pending = state.pending_resolution
    return (
        pending is not None and pending["kind"] == "select_to_hand"
        and bool(pending["remaining"]) and pending["eligible"](pending["remaining"][0])
    )


_select_to_hand_keep_legal._pending_gate = frozenset({"select_to_hand"})


def _select_to_hand_bottom_legal(state):
    pending = state.pending_resolution
    return pending is not None and pending["kind"] == "select_to_hand" and bool(pending["remaining"])


_select_to_hand_bottom_legal._pending_gate = frozenset({"select_to_hand"})


def _select_to_hand_keep_execute(state):
    game.execute_select_to_hand_option(state, "keep")


def _select_to_hand_bottom_execute(state):
    game.execute_select_to_hand_option(state, "bottom")


def _decline_search_legal(state):
    pending = state.pending_resolution
    return (
        pending is not None and pending["kind"] == "search_fetch" and pending.get("optional")
        and bool(game.search_fetch_options(state))
    )


_decline_search_legal._pending_gate = frozenset({"search_fetch"})


def _decline_search_execute(state):
    game.execute_search_fetch_decline(state)


def _decline_discard_legal(state):
    pending = state.pending_resolution
    return (
        pending is not None and pending["kind"] == "discard" and pending.get("optional")
        and bool(game.discard_options(state))
    )


_decline_discard_legal._pending_gate = frozenset({"discard"})


def _decline_discard_execute(state):
    game.execute_discard_decline(state)


def _target_self_legal(state):
    pending = state.pending_resolution
    return pending is not None and pending["kind"] == "choose_target_player"


_target_self_legal._pending_gate = frozenset({"choose_target_player"})


def _target_self_execute(state):
    game.execute_choose_target_player_option(state, state.active_idx)


def _target_opponent_legal(state):
    """Legal only once a real second PlayerState exists -- "target
    player" genuinely offers a choice the instant one does (Relic of
    Progenitus' own repeatable exile ability), same "only legal with a
    real opponent" gate deal_damage_to_opponent's own 2-player branch
    already uses elsewhere."""
    pending = state.pending_resolution
    return pending is not None and pending["kind"] == "choose_target_player" and len(state.players) > 1


_target_opponent_legal._pending_gate = frozenset({"choose_target_player"})


def _target_opponent_execute(state):
    game.execute_choose_target_player_option(state, 1 - state.active_idx)


def _discard_or_sacrifice_sacrifice_legal(name):
    """The SACRIFICE half of Highway Robbery's own "discard a card or
    sacrifice a land" -- a distinctly-labeled action (build_action_table's
    own "Sacrifice (cost): {name}"), not a reuse of the generic
    "Choose: X" dispatch _choose_name_options/_choose_name_execute give
    the DISCARD half: two different action strings, so a hand card and a
    battlefield land sharing a name (e.g. a Mountain in hand while
    Mountains are also in play) can never be ambiguous about which one a
    single button means."""
    def legal(state):
        pending = state.pending_resolution
        return pending is not None and pending["kind"] == "discard_or_sacrifice" and name in game.discard_or_sacrifice_sacrifice_options(state)
    legal._pending_gate = frozenset({"discard_or_sacrifice"})
    return legal


def _discard_or_sacrifice_sacrifice_execute(name):
    def execute(state):
        game.execute_discard_or_sacrifice_option(state, "sacrifice", name)
    return execute


def _decline_discard_or_sacrifice_legal(state):
    pending = state.pending_resolution
    return pending is not None and pending["kind"] == "discard_or_sacrifice"


_decline_discard_or_sacrifice_legal._pending_gate = frozenset({"discard_or_sacrifice"})


def _decline_discard_or_sacrifice_execute(state):
    game.execute_discard_or_sacrifice_decline(state)


def _madness_cast_legal(state):
    """Legal only if the model can actually afford the exiled card's
    madness cost right now -- same "guaranteed payable, not a maybe"
    contract every other alternate cast path here already follows."""
    pending = state.pending_resolution
    if pending is None or pending["kind"] != "madness_decision":
        return False
    madness_spec = game.EFFECT_REGISTRY[pending["card_def"].effect_id]["madness"]
    return game.plan_payment(state, madness_spec["cost"]) is not None


_madness_cast_legal._pending_gate = frozenset({"madness_decision"})


def _madness_cast_execute(state):
    game.execute_madness_cast(state)


def _madness_decline_legal(state):
    pending = state.pending_resolution
    return pending is not None and pending["kind"] == "madness_decision"


_madness_decline_legal._pending_gate = frozenset({"madness_decision"})


def _madness_decline_execute(state):
    game.execute_madness_decline(state)


def _activate_no_cost_legal(name, ability_legal, speed):
    """Non-mana activated-ability cost (Quirion Ranger's Forest bounce):
    no {T}-of-self assumption, unlike _activate_legal -- the ability's own
    legal(state, permanent) captures its whole cost precondition."""
    def legal(state):
        if state.pending_resolution is not None:
            return False
        if not game.turn.speed_legal(state, speed):
            return False
        p = next((p for p in state.battlefield if p.card_def.name == name), None)
        return p is not None and ability_legal(state, p)
    legal._pending_gate = _GATE_NO_PENDING
    return legal


def _activate_no_cost_execute(name, resolve):
    def execute(state):
        p = next(p for p in state.battlefield if p.card_def.name == name)
        resolve(state, p)
    return execute


def _alt_cast_legal(name, extra_legal, speed):
    """Land Grant's free alt-cost: no mana payment at all, just the
    card's own extra_legal predicate (0 lands in hand).

    Availability must go through _hand_count_available, not a bare
    "any copy in hand" check -- confirmed live via mono_red_madness_mirror
    training: a bare existence check let Fireblast's alt-cost (sacrifice 2
    Mountains) be cast a second time while the first cast's own copy was
    still physically in hand but already reserved on the stack (removal
    deferred to its own resolve, same as every cast-like path -- see
    push_to_stack's docstring), pushing a second stack entry for the same
    physical card. cast_fireblast_alt's own discard_from_hand_to_graveyard
    then ate that shared copy immediately (its eager, non-deferred
    hand-removal), so when the FIRST cast's stack entry finally resolved,
    its own discard_from_hand_to_graveyard found no copy left -- the
    "should be unreachable" RuntimeError."""
    def legal(state):
        if state.pending_resolution is not None:
            return False
        if not game.turn.speed_legal(state, speed):
            return False
        if _hand_count_available(state, name) <= 0:
            return False
        return extra_legal(state)
    legal._pending_gate = _GATE_NO_PENDING
    return legal


def _alt_cast_execute(name, resolve):
    """No generic engine-level cost mechanism for an alt cost (unlike mana's
    begin_pay_cost) -- so, same as _flashback_execute, this calls resolve
    immediately and leaves deferring-onto-the-stack entirely up to resolve
    itself. Alt-cost shapes vary: Land Grant's is free (nothing to pay, so
    its own resolve pushes right away, same as a free Flashback), Fireblast's
    is a real alternate cost (sacrifice 2 Mountains) that must actually be
    paid -- via its own resolution -- before ITS effect gets pushed. Pushing
    generically here, before resolve even runs, would defer Fireblast's own
    cost-payment along with its effect, which is wrong: the cost must be
    paid before anything is fully paid for and put on the stack."""
    def execute(state):
        card_def = game.CARD_DEFS[name]
        game.on_cast_trigger(state, card_def)  # item 11 -- see _cast_execute
        resolve(state, card_def)
    return execute


def _flashback_legal(name, ability_legal, speed):
    """Dread Return's Flashback: cast from the graveyard, not hand. Real
    Magic: Flashback follows the same timing as the card itself, not its
    own independent rule -- speed is the same value the card's normal
    cast derived, not a separate default."""
    def legal(state):
        if state.pending_resolution is not None:
            return False
        if not game.turn.speed_legal(state, speed):
            return False
        if not any(c.name == name for c in state.graveyard):
            return False
        return ability_legal(state)
    legal._pending_gate = _GATE_NO_PENDING
    return legal


def _flashback_execute(name, resolve):
    def execute(state):
        card_def = game.CARD_DEFS[name]
        game.on_cast_trigger(state, card_def)  # item 11 -- see _cast_execute
        resolve(state, card_def)
    return execute


def _plot_legal(name, cost, speed):
    """Plot {cost}: pay it and exile this card from hand (no board
    presence yet) -- legal exactly like a normal cast, just against the
    plot cost instead of card_def.cast_cost. Real Magic: Plot's own
    reminder text is "any time you could cast this card" -- same speed as
    the card's normal cast, not a separate timing rule; the later free
    cast from exile (_cast_from_exile_legal) uses the same speed too."""
    def legal(state):
        if state.pending_resolution is not None:
            return False
        if not game.turn.speed_legal(state, speed):
            return False
        if not any(c.name == name for c in state.hand):
            return False
        return game.plan_payment(state, cost) is not None
    legal._pending_gate = _GATE_NO_PENDING
    return legal


def _plot_execute(name, cost, resolve):
    def execute(state):
        card_def = game.CARD_DEFS[name]
        # Plotting itself isn't casting the spell (it's exiled, not
        # resolved) -- no on_cast_trigger here; that fires from
        # _cast_from_exile_execute below, once it's actually cast.
        game.begin_pay_cost(state, cost, on_complete=lambda s: resolve(s, card_def))
    return execute


def _cast_from_exile_legal(name, extra_legal, speed):
    """Plot's second half: cast a previously-plotted copy, without paying
    its mana cost, on any turn after the one it was plotted on. speed:
    same value _plot_legal used -- see that function's own docstring.

    extra_legal: Plot only waives the MANA cost, not any other cost a
    card's normal "cast" spec gates on (e.g. Highway Robbery's own
    "discard a card" additional cost still needs a card in hand to
    discard) -- reuses the same cast_spec["extra_legal"] the normal cast
    path already checks, so a card needing both never looks payable when
    it secretly isn't. None (every existing Plot card so far) means no
    such gate, unaffected."""
    def legal(state):
        if state.pending_resolution is not None:
            return False
        if not game.turn.speed_legal(state, speed):
            return False
        has_plotted = any(
            c.name == name and stamp is not None and stamp < state.turn_number
            for c, stamp in state.exile
        )
        if not has_plotted:
            return False
        return extra_legal is None or extra_legal(state)
    legal._pending_gate = _GATE_NO_PENDING
    return legal


def _cast_from_exile_execute(name, resolve):
    def execute(state):
        card_def = game.CARD_DEFS[name]
        entry = next(
            e for e in state.exile
            if e[0].name == name and e[1] is not None and e[1] < state.turn_number
        )
        state.exile.remove(entry)
        game.on_cast_trigger(state, card_def)  # item 11 -- see _cast_execute
        # Plot's whole point is that the cost was already paid earlier
        # (when plotted) -- already "fully paid for" now, so push
        # immediately instead of resolving now (see _cast_execute's own
        # stack comment).
        game.push_to_stack(state, card_def, resolve, reserves_hand_card=False)
    return execute


def _omen_cast_legal(hand_name, cost, speed):
    """Sagu Wildling's Omen: real Scryfall reminder text is "(Also shuffle
    this card.)" attached to Roost Seek's own library search -- unlike
    real Adventure, an Omen card does NOT exile itself for a later free-
    zone cast; the resolved sorcery is shuffled directly into the LIBRARY
    (cast_roost_seek), and the real creature half only ever becomes
    castable again once the same physical card is redrawn into HAND, same
    as any ordinary card. So this is really just "the same hand card, a
    second cast option with its own distinct cost" -- checked against
    state.hand, not state.exile. hand_name is the SORCERY side's own
    registered name (the only one ever physically in a zone) -- the
    creature side is a distinct CardDef, never separately registered in
    game.CARD_DEFS (see the "omen" registry spec's own "card_def" key)."""
    def legal(state):
        if state.pending_resolution is not None:
            return False
        if not game.turn.speed_legal(state, speed):
            return False
        if not any(c.name == hand_name for c in state.hand):
            return False
        return game.plan_payment(state, cost) is not None
    legal._pending_gate = _GATE_NO_PENDING
    return legal


def _omen_cast_execute(creature_card_def, cost, resolve):
    """Same begin_pay_cost -> push_to_stack shape as a normal hand cast
    (_cast_execute), just for `creature_card_def` (the distinct creature
    CardDef) instead of game.CARD_DEFS[name]. reserves_hand_card defaults
    True here (unlike Plot/Flashback's own exile/graveyard-sourced pushes)
    -- the physical card genuinely IS still sitting in the caster's hand,
    unresolved, while this is paid for; _hand_count_available matches
    stack entries by NAME, so pushing creature_card_def (same display name
    as whatever's in hand) still correctly reserves it, blocking the
    sorcery-mode cast of the same physical copy in the meantime. `resolve`
    is responsible for removing the matching hand card itself (by NAME,
    not identity -- the object actually sitting in state.hand is the
    sorcery side's own CardDef, a different object from creature_card_def
    despite sharing a display name), same "resolve does its own zone
    removal" convention every other cast path here follows."""
    def execute(state):
        game.on_cast_trigger(state, creature_card_def)  # no-op for a CREATURE card_def (on_cast_trigger only fires for INSTANT/SORCERY) -- called anyway for the same hygiene every other cast path here has
        game.begin_pay_cost(state, cost, on_complete=lambda s: game.push_to_stack(s, creature_card_def, resolve))
    return execute


def build_action_table(decklist, registry, token_card_defs=(), pending_kinds=(),
                        opponent_decklist=None, opponent_token_card_defs=(), extra_choosable_names=()):
    """opponent_decklist/opponent_token_card_defs: the OTHER side's own
    decklist/tokens (docs/COMBAT_PLAN.md's cross-player targeting
    primitive) -- None/() for every 1-player deck (there's no real
    opponent battlefield to reference at all), matching combat_enabled=False
    decks never seeing "Attack: X" become legal. Only ever given by
    TwoPlayerDeckEnv, which already has this data on hand for its own
    opponent_actions table.

    token_card_defs: every token CardDef this deck's own cards can
    create at runtime (Blood, Robot, Warrior, Eldrazi Spawn --
    docs/MADNESS_DECKS_PLAN.md item 8), e.g. (game.BLOOD_TOKEN_CARD_DEF,).
    Tokens are never decklist entries (no quantity, not in game.CARD_DEFS),
    so they can't flow through distinct_names/game.CARD_DEFS[name] the way
    every other action here does -- casting/land-drop/Flashback/etc. stay
    decklist-only, a token is never cast or played as a land.

    Two independent things read this list, for two different reasons: the
    activated-abilities loop below (a token's own ability, e.g. Blood's
    sac-for-a-card or Eldrazi Spawn's sac-for-{C}, needs an action to
    exist at all), and the choosable_names set that both the "Choose: X"
    and "Choose target: X (slot k)" name lists build from (a token can be
    a perfectly legal choose_permanent/sacrifice/discard choice -- e.g. any
    creature-enchanting Aura can enchant a token creature -- despite never
    appearing in the decklist; list a token here even if it has no
    activated ability of its own, like Warrior, so it stays a legal choice
    once it's on the battlefield). Defaults to () so every existing call
    site (Tron, spy_combo -- neither creates tokens) is unaffected.

    pending_kinds: this deck's own extra pending-resolution kinds beyond
    the universal baseline (pay_cost) -- see game.registry.
    derive_pending_kinds -- gates which of the fixed kind-specific actions
    below (Keep/Dispose scry-surveil, Decline Ancient Stirrings, etc.)
    actually get added, so a deck's action table never grows because of a
    pending kind only some other deck can reach."""
    distinct_names = sorted({name for name, *_rest in decklist})
    land_names = sorted({
        name for name in distinct_names if game.CARD_DEFS[name].card_type == game.CardType.LAND
    })

    actions = []

    for name in land_names:
        actions.append((f"Play land: {name}", _land_drop_legal(name), _land_drop_execute(name)))

    for name in distinct_names:
        card_spec = registry.get(game.CARD_DEFS[name].effect_id, {})
        cast_spec = card_spec.get("cast")
        if cast_spec is not None:
            # "precast_choice": True (Auras' real targets, Crop Rotation's
            # sacrifice-as-a-cost) -- resolve must run immediately once paid
            # and manage its own push_to_stack, instead of the generic
            # auto-push _cast_execute does (see _precast_choice_execute's
            # own docstring for exactly why).
            cast_execute_fn = _precast_choice_execute if cast_spec.get("precast_choice") else _cast_execute
            actions.append((
                f"Cast {name}",
                _cast_legal(name, cast_spec.get("extra_legal"), _cast_speed(game.CARD_DEFS[name], cast_spec)),
                cast_execute_fn(name, cast_spec["resolve"]),
            ))
        # Winding Way: a modal cast (choose creature or land) instead of a
        # single "cast" entry -- one action per mode.
        cast_modes = card_spec.get("cast_modes")
        if cast_modes is not None:
            for mode_name, mode_spec in cast_modes.items():
                mode_execute_fn = _precast_choice_execute if mode_spec.get("precast_choice") else _cast_execute
                actions.append((
                    f"Cast {name} (choose {mode_name})",
                    _cast_legal(name, mode_spec.get("extra_legal"), _cast_speed(game.CARD_DEFS[name], mode_spec)),
                    mode_execute_fn(name, mode_spec["resolve"]),
                ))
        # Nyxborn Hydra: X-cost modes (its own normal creature cast AND
        # Bestow, each with a different base cost) -- one action per (mode,
        # X) pair, X in 0..mode_spec["max_x"]. Each mode's own "resolve" is
        # a function OF x returning the (state, card_def) resolve itself
        # (green_cards.cast_nyxborn_hydra_creature/cast_nyxborn_hydra_bestow),
        # not a plain resolve like cast_modes above -- X has to be baked
        # into a distinct closure per action, there's no other way to tell
        # two different X actions apart once they're both just entries in
        # this flat action table. plan_payment (inside _x_cast_legal) is
        # what keeps an unaffordable X from ever being offered -- this loop
        # only bounds the table's own size, not what's ever actually legal.
        x_cast_modes = card_spec.get("x_cast_modes")
        if x_cast_modes is not None:
            for mode_name, mode_spec in x_cast_modes.items():
                mode_execute_fn = _x_precast_choice_execute if mode_spec.get("precast_choice") else _x_cast_execute
                speed = _cast_speed(game.CARD_DEFS[name], mode_spec)
                extra_legal = mode_spec.get("extra_legal")
                make_resolve = mode_spec["resolve"]
                base_cost = mode_spec["cost"]
                for x in range(mode_spec["max_x"] + 1):
                    cost = dict(base_cost)
                    cost["generic"] = cost.get("generic", 0) + x
                    actions.append((
                        f"Cast {name} ({mode_name}, X={x})",
                        _x_cast_legal(name, cost, extra_legal, speed),
                        mode_execute_fn(name, cost, make_resolve(x)),
                    ))
        # Land Grant: a second, free cast path alongside the normal one.
        alt_cast = card_spec.get("alt_cast")
        if alt_cast is not None:
            actions.append((
                f"Cast {name} (free)",
                _alt_cast_legal(name, alt_cast["extra_legal"], _cast_speed(game.CARD_DEFS[name], alt_cast)),
                _alt_cast_execute(name, alt_cast["resolve"]),
            ))
        # Dread Return: Flashback casts from the graveyard, not hand.
        flashback = card_spec.get("flashback")
        if flashback is not None:
            actions.append((
                f"Flashback {name}",
                _flashback_legal(name, flashback["legal"], _cast_speed(game.CARD_DEFS[name], flashback)),
                _flashback_execute(name, flashback["resolve"]),
            ))
        # Highway Robbery: Plot -- pay its plot cost to exile it now,
        # cast it for free from exile on any later turn. The cast-from-
        # exile half reuses this same card's normal "cast" resolve (the
        # real spell effect is identical either way, only how the cost
        # was paid differs) -- so a "plot" entry only makes sense
        # alongside a "cast" entry, never alone.
        plot = card_spec.get("plot")
        if plot is not None:
            plot_speed = _cast_speed(game.CARD_DEFS[name], plot)  # same speed governs both actions below -- Plot's own reminder text ("any time you could cast this card") is one timing rule, not two
            actions.append((
                f"Plot {name}",
                _plot_legal(name, plot["cost"], plot_speed),
                _plot_execute(name, plot["cost"], plot["resolve"]),
            ))
            # cast_from_exile_resolve: optional override for cards whose
            # normal "cast" resolve does state.hand.remove(card_def) (the
            # universal convention for every cast resolve in this codebase)
            # -- wrong once the card already left exile, never hand, by
            # the time this runs (Highway Robbery's own real-world case;
            # every existing Plot self-check's resolve happens to be a
            # no-op, which is why this distinction never mattered before).
            # Falls back to cast_spec["resolve"] unchanged for any card
            # whose resolve doesn't care either way.
            actions.append((
                f"Cast {name} (plotted)",
                _cast_from_exile_legal(name, cast_spec.get("extra_legal"), plot_speed),
                _cast_from_exile_execute(name, plot.get("cast_from_exile_resolve", cast_spec["resolve"])),
            ))
        # Sagu Wildling: Omen -- cast_roost_seek (this same "name"'s own
        # "cast" resolve above) shuffles the card into the LIBRARY instead
        # of graveyarding OR exiling it (real Adventure's own exile
        # doesn't apply to Omen -- see _omen_cast_legal's own docstring).
        # This second action is just an ordinary-looking second cast
        # option for the same hand card, its own real cost, offered
        # whenever a same-named card is back in hand (redrawn after being
        # shuffled in) -- never shares a resolve with the sorcery mode,
        # the creature side is a wholly different spell.
        omen = card_spec.get("omen")
        if omen is not None:
            omen_speed = _cast_speed(omen["card_def"], omen)
            actions.append((
                f"Cast {omen['card_def'].name} (omen)",
                _omen_cast_legal(name, omen["cost"], omen_speed),
                _omen_cast_execute(omen["card_def"], omen["cost"], omen["resolve"]),
            ))

    activatable = [(name, game.CARD_DEFS[name].effect_id) for name in distinct_names]
    activatable += [(cd.name, cd.effect_id) for cd in token_card_defs]
    for name, effect_id in activatable:
        abilities = registry.get(effect_id, {}).get("activated_abilities", {})
        for ability_name, spec in abilities.items():
            # Real Magic's own default for activated (non-mana) abilities is
            # the opposite of a spell's: any time, unless the card says
            # "activate only as a sorcery" -- an explicit "speed" key in the
            # ability's own spec is that opt-in override; every existing
            # ability (Blood, Candy Trail, Expedition Map, Bonders'
            # Ornament, Quirion Ranger, Barrels) has none, so all keep
            # working in every phase exactly as before this feature existed.
            speed = spec.get("speed", game.turn.Speed.INSTANT)
            if "cost_key" in spec:
                actions.append((
                    f"Activate {name} ({ability_name})",
                    _activate_legal(name, spec["cost_key"], speed),
                    _activate_execute(name, spec["cost_key"], spec["resolve"]),
                ))
            else:
                # Non-mana cost (Quirion Ranger: return a Forest to hand).
                actions.append((
                    f"Activate {name} ({ability_name})",
                    _activate_no_cost_legal(name, spec["legal"], speed),
                    _activate_no_cost_execute(name, spec["resolve"]),
                ))

    for name in distinct_names:
        fc_spec = registry.get(game.CARD_DEFS[name].effect_id, {}).get("forestcycle")
        if fc_spec is not None:
            actions.append((
                f"Forestcycle {name}",
                _forestcycle_legal(name, fc_spec["cost_key"]),
                _forestcycle_execute(name, fc_spec["cost_key"], fc_spec["resolve"]),
            ))

    # Bramble Wurm: an activated ability usable from the graveyard, not
    # the battlefield (unlike every "activated_abilities" entry above) or
    # hand (unlike Forestcycle) -- its own "graveyard_ability" registry key.
    for name in distinct_names:
        gy_spec = registry.get(game.CARD_DEFS[name].effect_id, {}).get("graveyard_ability")
        if gy_spec is not None:
            actions.append((
                f"Activate {name} (graveyard)",
                _graveyard_ability_legal(name, gy_spec["cost_key"]),
                _graveyard_ability_execute(name, gy_spec["cost_key"], gy_spec["resolve"]),
            ))

    actions.append(("Pass", _pass_legal, _pass_execute))

    # "Choose: X" needs to cover every name a sacrifice/discard/search_fetch/
    # etc. resolution could ever offer -- not just decklist names. (NOT
    # choose_permanent -- that's the "Choose target: X (slot k)" block
    # below, exact-(name, slot) addressed.) A token (e.g. boggles' Eldrazi
    # Spawn) is a perfectly legal sacrifice/discard choice despite never
    # appearing in CARD_DEFS/the decklist; omitting token names here left
    # exactly that case legal-to-create but impossible-to-choose once a
    # token was the only eligible option, softlocking the game.
    # extra_choosable_names: card names that can be a "Choose: X" option
    # despite not being in THIS deck (nor its tokens) -- specifically an
    # OPPONENT's graveyard cards, reachable by a choose_graveyard_card
    # resolution that targets a player (Relic of Progenitus' exile ability,
    # colorless_cards.py). Without them, a cross-deck game where the acting
    # player exiles from the OPPONENT's graveyard has zero legal actions for
    # names outside its own deck -> empty action mask -> dead state (a real
    # softlock, confirmed via a monster_tron-vs-mono_red smoke game). Passed
    # as the whole league's card universe (token_pool.build_pool), not a
    # specific opponent's deck: bounded, fixed per trained model, and still
    # runtime-masked to only-legal-when-actually-in-the-targeted-graveyard.
    choosable_names = sorted(set(distinct_names) | {cd.name for cd in token_card_defs} | set(extra_choosable_names))
    for name in choosable_names:
        actions.append((f"Choose: {name}", _choose_name_legal(name), _choose_name_execute(name)))

    # "Attack: X (slot k)" -- one per (creature name, slot) pair
    # (docs/COMBAT_PLAN.md's permanent-identity design), legal only during
    # Phase.DECLARE_ATTACKERS (see _attack_legal). k runs 1..that card's
    # own decklist quantity for a real card -- the pooled slot scheme
    # means this is a hard, correct bound even through repeated bounce/
    # blink, since only that many physical copies can ever be
    # simultaneously alive. A token has no decklist quantity to read, so
    # k instead runs 1..TOKEN_LIMIT -- a shared pool across every token
    # name combined, so any single name could in principle claim all of
    # it, and each name's own registered range has to cover that worst
    # case independently. A deck whose own phase sequence never includes
    # DECLARE_ATTACKERS (combat_enabled=False) simply never sees any of
    # these become legal -- same "phase not in this deck's own sequence"
    # degrade every other phase-gated action already relies on.
    card_type_by_name = {name: game.CARD_DEFS[name].card_type for name in distinct_names}
    card_type_by_name.update({cd.name: cd.card_type for cd in token_card_defs})
    qty_by_name = {name: qty for name, qty, *_rest in decklist}

    # "Choose target: X (slot k)" -- the "choose_permanent" resolution's own
    # exact-(name, slot) addressed actions (Aura enchant-targets, Crop
    # Rotation's sacrifice cost, land bounce), same shape/reasoning as
    # "Choose opponent's: X (slot k)" below just scoped to THIS side's own
    # battlefield. Registered for every choosable name, not just creatures
    # (unlike "Attack:"/"Assign Blocker:" below) -- Utopia Sprawl/Abundant
    # Growth target lands, not creatures -- and legal() gates precisely at
    # runtime against whichever predicate the actual pending choose_permanent
    # resolution holds, same "pre-register broadly, mask precisely" pattern
    # "Choose: X as color" below already uses.
    for name in choosable_names:
        max_slot = qty_by_name.get(name, game.TOKEN_LIMIT)
        for slot in range(1, max_slot + 1):
            actions.append((
                f"Choose target: {name} (slot {slot})",
                _choose_permanent_legal(name, slot),
                _choose_permanent_execute(name, slot),
            ))

    attackable_names = sorted(name for name in choosable_names if card_type_by_name[name] == game.CardType.CREATURE)
    for name in attackable_names:
        max_slot = qty_by_name.get(name, game.TOKEN_LIMIT)
        for slot in range(1, max_slot + 1):
            actions.append((
                f"Attack: {name} (slot {slot})",
                _attack_legal(name, slot),
                _attack_execute(name, slot),
            ))

    # "Assign Blocker: X (slot j)" -- same own-creature (name, slot)
    # addressing as "Attack: X (slot k)" above, since blocking is a
    # decision about this player's OWN creatures (docs/COMBAT_PLAN.md),
    # just legal at a different point (once _declare_blockers_gen has
    # flipped state.active_idx to the defender and a "declare_blockers"
    # resolution is pending -- see _assign_blocker_legal). "Done blocking"
    # is the explicit action that closes the consult, same "Done" precedent
    # as scry/surveil's own keep-then-order decomposition.
    for name in attackable_names:
        max_slot = qty_by_name.get(name, game.TOKEN_LIMIT)
        for slot in range(1, max_slot + 1):
            actions.append((
                f"Assign Blocker: {name} (slot {slot})",
                _assign_blocker_legal(name, slot),
                _assign_blocker_execute(name, slot),
            ))
    actions.append(("Done blocking", _done_blocking_legal, _done_blocking_execute))

    # "Choose opponent's: X (slot k)" -- the general cross-player
    # targeting primitive (docs/COMBAT_PLAN.md), one per (opponent
    # creature name, slot), built from the OPPONENT's own decklist/tokens
    # instead of this side's own -- blocking's first consumer. Same
    # quantity-or-TOKEN_LIMIT bound as the attack registration above, just
    # applied to the other side's card pool. None/() (the default for
    # every 1-player deck) registers nothing at all -- there's no real
    # opponent battlefield to ever reference in that mode.
    if opponent_decklist is not None:
        opponent_distinct_names = sorted({name for name, *_rest in opponent_decklist})
        opponent_card_type_by_name = {name: game.CARD_DEFS[name].card_type for name in opponent_distinct_names}
        opponent_card_type_by_name.update({cd.name: cd.card_type for cd in opponent_token_card_defs})
        opponent_qty_by_name = {name: qty for name, qty, *_rest in opponent_decklist}
        opponent_choosable_names = sorted(
            set(opponent_distinct_names) | {cd.name for cd in opponent_token_card_defs}
        )
        opponent_targetable_names = sorted(
            name for name in opponent_choosable_names
            if opponent_card_type_by_name[name] == game.CardType.CREATURE
        )
        for name in opponent_targetable_names:
            max_slot = opponent_qty_by_name.get(name, game.TOKEN_LIMIT)
            for slot in range(1, max_slot + 1):
                actions.append((
                    f"Choose opponent's: {name} (slot {slot})",
                    _choose_opponent_permanent_legal(name, slot),
                    _choose_opponent_permanent_execute(name, slot),
                ))

    # Abundant Growth's own grant: a runtime, per-instance fact (which
    # specific land, if any, ends up enchanted) that can't be known when
    # this table is built, before any game state exists -- so every land
    # name gets a "Choose: X as color" slot for every color ANY card in
    # this decklist can ever grant, pre-registered here and masked
    # legal/illegal at runtime by mana.tap_cost_options actually seeing
    # (or not seeing) an attached grant.
    grantable_colors = set()
    for name in distinct_names:
        grantable_colors |= registry.get(game.CARD_DEFS[name].effect_id, {}).get("grants_mana_colors", set())

    for name in distinct_names:
        spec = registry.get(game.CARD_DEFS[name].effect_id, {})
        colors = set()
        mana = spec.get("mana")
        if mana is not None and mana[0] == "flexible":
            colors |= mana[1]
        filter_mana = spec.get("filter_mana")
        if filter_mana is not None:
            colors |= filter_mana["colors"]
        if game.CARD_DEFS[name].card_type == game.CardType.LAND:
            colors |= grantable_colors
        for color in sorted(colors):
            actions.append((
                f"Choose: {name} as {color}",
                _choose_name_color_legal(name, color),
                _choose_name_color_execute(name, color),
            ))

    for color in game.POOL_COLORS:
        actions.append((
            f"Spend {color} from pool",
            _pool_spend_legal(color),
            _pool_spend_execute(color),
        ))

    if "scry" in pending_kinds or "surveil" in pending_kinds:
        actions.append(("Keep (scry/surveil)", _keep_dispose_legal, _keep_execute))
        actions.append(("Dispose (scry/surveil)", _keep_dispose_legal, _dispose_execute))
    if "ancient_stirrings" in pending_kinds:
        actions.append(("Decline (Ancient Stirrings)", _decline_legal, _decline_execute))
    if "malevolent_rumble" in pending_kinds:
        actions.append(("Decline (Malevolent Rumble)", _decline_malevolent_rumble_legal, _decline_malevolent_rumble_execute))
    if "select_to_hand" in pending_kinds:
        actions.append(("Keep (select to hand)", _select_to_hand_keep_legal, _select_to_hand_keep_execute))
        actions.append(("Bottom (select to hand)", _select_to_hand_bottom_legal, _select_to_hand_bottom_execute))
    if "search_fetch" in pending_kinds:
        # Gated on "search_fetch" membership alone, not per-deck optionality
        # (Tron's own search_fetch uses are never optional=True, so this
        # stays present-but-permanently-illegal for Tron -- same as it was
        # unconditionally before this change; both current decks already
        # share "search_fetch" either way, so this isn't a growth vector).
        actions.append(("Decline (search)", _decline_search_legal, _decline_search_execute))
    actions.append(("Abandon payment", _abandon_payment_legal, _abandon_payment_execute))  # pay_cost is baseline, always present
    # mulligan_decision/mulligan_bottom are baseline too (BASELINE_PENDING_KINDS,
    # game.turn.run_mulligan_phase) -- every deck goes through the pregame
    # mulligan phase, so these are unconditional, same footing as "Abandon
    # payment" above. Bottoming itself reuses the existing "Choose: X"
    # action (_choose_name_options/_choose_name_execute's own "mulligan_bottom"
    # branch) -- no separate action needed for it.
    actions.append(("Keep hand", _mulligan_decision_legal, _mulligan_keep_execute))
    actions.append(("Mulligan", _mulligan_take_legal, _mulligan_take_execute))
    if "discard" in pending_kinds:
        actions.append(("Decline (discard)", _decline_discard_legal, _decline_discard_execute))
    if "discard_or_sacrifice" in pending_kinds:
        # The DISCARD half reuses the generic "Choose: X" action built
        # above (bare hand-card names); only the SACRIFICE half needs its
        # own distinctly-labeled actions here (see
        # _discard_or_sacrifice_sacrifice_legal's own docstring for why).
        for name in land_names:
            actions.append((
                f"Sacrifice (cost): {name}",
                _discard_or_sacrifice_sacrifice_legal(name),
                _discard_or_sacrifice_sacrifice_execute(name),
            ))
        actions.append((
            "Decline (discard or sacrifice)",
            _decline_discard_or_sacrifice_legal,
            _decline_discard_or_sacrifice_execute,
        ))
    if "madness_decision" in pending_kinds:
        actions.append(("Cast (madness)", _madness_cast_legal, _madness_cast_execute))
        actions.append(("Decline (madness)", _madness_decline_legal, _madness_decline_execute))
    if "choose_target_player" in pending_kinds:
        # "Target: yourself" is always legal the instant this pending
        # kind is reached (a real Magic legality fact -- "target player"
        # never excludes its own caster), even alone in a 1-player game;
        # "Target: opponent" only becomes legal once a real second
        # PlayerState exists. Two fixed actions, not a per-name loop --
        # there are only ever at most 2 possible players, never more.
        actions.append(("Target: yourself", _target_self_legal, _target_self_execute))
        actions.append(("Target: opponent", _target_opponent_legal, _target_opponent_execute))

    return tuple(actions)


_battlefield_lookup_cache = None  # (state, {(name, slot): Permanent}) -- valid only for the duration of one legal_action_mask sweep, same lifecycle as _tap_cost_options_cache below


def _cached_battlefield_lookup(state):
    """Sweep-scoped {(name, slot): Permanent} lookup for state.battlefield --
    same "profiled, not guessed" caching pattern as _cached_tap_cost_options
    just below (docs/PRIORITY_PLAN.md item 6): _attack_legal/
    _assign_blocker_legal each independently scanned the WHOLE battlefield
    with any(...) to find one specific (name, slot), once per action-table
    entry -- for a deck with many creature copies (boggles' Auras/tokens)
    that's O(action_table_size x battlefield_size) repeated work every
    sweep (profiled: 2 closures alone accounted for ~3.4M calls across a
    single 8192-step training burst). Building this dict once per sweep
    turns each of those checks into an O(1) lookup. Safe for the same
    reason _cached_tap_cost_options is: a legal_action_mask sweep only ever
    calls legal_fns, never an execute_* function, so state can't change
    mid-sweep. (name, slot) is a safe dict key here the same way
    _creature_slot_block's own by_slot lookup already relies on it being
    unique per name -- state.battlefield is always ONE side's own,
    active-relative zone (see this module's other active-relative
    docstrings), never two players' permanents mixed in one sweep."""
    global _battlefield_lookup_cache
    if _battlefield_lookup_cache is None or _battlefield_lookup_cache[0] is not state:
        _battlefield_lookup_cache = (state, {(p.card_def.name, p.slot): p for p in state.battlefield})
    return _battlefield_lookup_cache[1]


_tap_cost_options_cache = None  # (state, result) -- valid only for the duration of one legal_action_mask sweep, see _cached_tap_cost_options


def _cached_tap_cost_options(state):
    """Memoizes game.tap_cost_options(state) for the exact duration of one
    legal_action_mask sweep (docs/PRIORITY_PLAN.md item 6 -- profiled,
    not guessed: mana.tap_cost_options was called 480,942 times against
    only 78,969 mask builds, ~6x more than needed and ~10% of total
    training time by itself). _choose_name_legal/_choose_name_color_legal
    (the "Choose: X"/"Choose: X as color" mana-source actions) each
    independently call this from scratch, once per candidate name/color,
    so one sweep recomputes the identical list several times over.
    Provably safe to cache for exactly this scope: a legal_action_mask
    sweep only ever calls legal_fns, never an execute_* function, so state
    can't change mid-sweep -- legal_action_mask resets this cache before
    and after its own sweep (see there), so nothing outside a sweep (an
    actual execute_fn call, a later sweep against mutated state) can ever
    see a stale hit."""
    global _tap_cost_options_cache
    if _tap_cost_options_cache is None or _tap_cost_options_cache[0] is not state:
        _tap_cost_options_cache = (state, game.tap_cost_options(state))
    return _tap_cost_options_cache[1]


def legal_action_mask(state, actions):
    """Stateless: usable both by DeckEnv.action_masks() and by
    harness.evaluate(), which plays games directly through game.run_game,
    not through env.step (see DRL_CHECKLIST.md's D6 implementation note).
    `actions` is any table built by build_action_table -- every deck's own
    table, none privileged as a default (a caller with its own decklist
    always has its own table to pass, e.g. harness.py's self.actions).

    Category-gating (profiled, not guessed: this table can run ~300 entries
    long, and every single one of those closures gets called on every
    sweep regardless of relevance -- see docs/GPU_VECENV_INVESTIGATION.md's
    training-speed followup): most `_X_legal` closures start with a cheap,
    static check of state.pending_resolution (either "must be None" or
    "must be one specific kind/set of kinds") before doing any real work.
    Each such closure is stamped with a `._pending_gate` attribute at
    creation time -- `_GATE_NO_PENDING`, or a frozenset of the
    pending_resolution["kind"] values it could possibly be legal under --
    copied directly from that closure's own first-line check, changing WHEN
    it gets called, never WHAT it returns. A closure with no `._pending_gate`
    stamped (attack, and anything this fix's own audit didn't touch) is
    always called, exactly like every closure was before this fix -- the
    fail-safe default, not an optimization gap that can go wrong.

    Resets _tap_cost_options_cache, _battlefield_lookup_cache, and
    game.mana's own _enchanting_cache (game.reset_mana_cache) before AND
    after the sweep itself (not just before): guarantees none of these
    caches can ever leak past this call's own scope into a later
    execute_fn call or an unrelated sweep against a different/mutated
    state, even though nothing in the current single-threaded, synchronous
    call pattern would actually trigger that -- belt-and-suspenders for a
    module-level global, not load-bearing. mana.py's own cache is reset
    from here, not self-invalidating there, for the same reason the other
    two aren't: see game.mana._enchanting's own docstring."""
    global _tap_cost_options_cache, _battlefield_lookup_cache
    _tap_cost_options_cache = None
    _battlefield_lookup_cache = None
    game.reset_mana_cache()
    pending = state.pending_resolution
    pending_kind = pending["kind"] if pending is not None else None
    try:
        mask = np.zeros(len(actions), dtype=bool)
        for idx, (_name, legal_fn, _execute) in enumerate(actions):
            gate = getattr(legal_fn, "_pending_gate", None)
            if gate is _GATE_NO_PENDING:
                if pending is not None:
                    continue
            elif gate is not None and pending_kind not in gate:
                continue
            mask[idx] = legal_fn(state)
        return mask
    finally:
        _tap_cost_options_cache = None
        _battlefield_lookup_cache = None
        game.reset_mana_cache()


# ---------------------------------------------------------------------------
# D2.3 / D2.4 -- DeckEnv
# ---------------------------------------------------------------------------

def _lost(state, seat_idx):
    """True once someone has won and it wasn't seat_idx -- the one thing
    every existing 1-player reward_fn (rewards.py) can't tell on its own:
    state.turn_won/turn_number don't say WHO won, only that the game
    ended. A win (state.winner == seat_idx) or "nobody yet" (state.winner
    is None, including the still-in-progress case) both fall through
    unchanged to whatever the wrapped reward_fn would already compute --
    only an actual loss needs to be forced to 0 here."""
    return state.winner is not None and state.winner != seat_idx


def _for_player(state, player_idx, fn):
    """Runs fn(state) with state.active_idx temporarily set to player_idx,
    then restores it -- lets existing active-player-proxied logic
    (rewards.resource_quality_components, game.permanent_power's own
    aura-enchanting search, mana.py's Tron-awareness via state.battlefield,
    ...) be reused for a NON-active player (here: whichever seat's
    OPPONENT this observation is being built for) instead of a second,
    parallel implementation of any of it. Safe even though state.stack/
    pending_resolution are shared, not per-player -- this is only ever
    called between turns (never during a resolution), and every property
    this flip actually affects (hand/battlefield/graveyard/library/
    mana_pool/etc.) is genuinely per-player."""
    original = state.active_idx
    state.active_idx = player_idx
    try:
        return fn(state)
    finally:
        state.active_idx = original


if __name__ == "__main__":
    # ponytail self-check: no pytest in this project, mirrors the
    # assert-based demo convention -- run via `python drl_env.py` from
    # src/. Exercises Plot (MADNESS_DECKS_PLAN.md item 4) and the on-cast
    # trigger hook (item 11) through the REAL _plot_legal/_plot_execute/
    # _cast_from_exile_legal/_cast_from_exile_execute functions -- not a
    # parallel reimplementation. No real Plot/Guttersnipe card exists yet
    # (deck assembly out of scope), so this temporarily injects into the
    # global game.CARD_DEFS/game.EFFECT_REGISTRY, saving/restoring both.
    from game.cards import CardDef, CardType, EffectId
    from game.state import GameState, Permanent, PlayerState

    _card_defs_backup = dict(game.CARD_DEFS)
    _filler_backup = game.EFFECT_REGISTRY[EffectId.FILLER]
    _generous_ent_backup = game.EFFECT_REGISTRY[EffectId.GENEROUS_ENT]

    PLOT_COST = {"generic": 1, "B": 1}  # {B}, not {R} -- EffectId.SWAMP is a real, already-correctly-wired
    # mana source (registry.py's derived views like SIMPLE_MANA_SOURCE_EFFECTS/_FIXED_SOURCE_COLOR are built
    # once at import time; injecting a fake "mana" spec onto FILLER here wouldn't be reflected in them, so the
    # legality pre-check (plan_payment) would wrongly see no valid source -- reusing a real fixed-color land
    # sidesteps that entirely rather than also having to patch the derived views to match).

    on_cast_calls = []
    plot_spell = CardDef("Fake Plot Spell", CardType.SORCERY, PLOT_COST, EffectId.FILLER)
    game.CARD_DEFS["Fake Plot Spell"] = plot_spell
    game.EFFECT_REGISTRY[EffectId.FILLER] = {
        "cast": {"resolve": lambda s, c: None},
        "plot": {"cost": PLOT_COST, "resolve": lambda s, c: (s.hand.remove(c), s.exile.append((c, s.turn_number)))},
    }
    # Guttersnipe stand-in: a permanent whose registry entry has an
    # "on_cast" trigger -- borrows EffectId.GENEROUS_ENT for the duration.
    game.EFFECT_REGISTRY[EffectId.GENEROUS_ENT] = {
        "on_cast": lambda s, permanent: on_cast_calls.append(permanent.card_def.name),
    }
    try:
        state = GameState(on_the_play=True)
        state.phase = game.turn.Phase.MAIN1  # Plot Speed defaults to SORCERY (a CardType.SORCERY card, no override) -- needs a sorcery-speed phase to be legal at all now
        state.hand = [plot_spell]
        state.battlefield = [
            Permanent(CardDef("Swamp", CardType.LAND, None, EffectId.SWAMP)),
            Permanent(CardDef("Swamp", CardType.LAND, None, EffectId.SWAMP)),
            Permanent(CardDef("Guttersnipe-ish", CardType.CREATURE, None, EffectId.GENEROUS_ENT)),
        ]

        # Plot it: pay {1}{B}, exile with this turn's stamp. Both Swamps
        # are needed (1 generic + 1 B); pay_cost is always interactive
        # regardless of what the legality pre-check found, so this taps
        # them one at a time.
        assert _plot_legal("Fake Plot Spell", PLOT_COST, game.turn.Speed.SORCERY)(state)
        _plot_execute("Fake Plot Spell", PLOT_COST, game.EFFECT_REGISTRY[EffectId.FILLER]["plot"]["resolve"])(state)
        assert state.pending_resolution["kind"] == "pay_cost"
        while state.pending_resolution is not None:
            tap_opts = game.tap_cost_options(state)
            if tap_opts:
                name, _color, is_filter = tap_opts[0]
                game.execute_tap_cost_option(state, name, None, is_filter)
            else:
                # Both Swamps produce only B -- the 2nd tap's B floats
                # into the pool instead of auto-filling the outstanding
                # {generic:1} pip (this engine deliberately never
                # auto-spends floated mana toward generic -- mana.py's
                # own documented design). Spend it explicitly.
                game.execute_pool_spend(state, game.pool_spend_options(state)[0])
        assert state.pending_resolution is None
        assert state.hand == []
        assert [c.name for c, _stamp in state.exile] == ["Fake Plot Spell"]
        assert on_cast_calls == []  # plotting itself never fires on_cast -- it isn't casting the spell

        # Same turn: not castable yet ("on a later turn").
        assert not _cast_from_exile_legal("Fake Plot Spell", None, game.turn.Speed.SORCERY)(state)

        # A later turn: castable for free, fires on_cast_trigger (Guttersnipe).
        state.turn_number += 1
        assert _cast_from_exile_legal("Fake Plot Spell", None, game.turn.Speed.SORCERY)(state)
        _cast_from_exile_execute("Fake Plot Spell", game.EFFECT_REGISTRY[EffectId.FILLER]["cast"]["resolve"])(state)
        assert state.exile == []
        assert on_cast_calls == ["Guttersnipe-ish"]

        # extra_legal gate on the cast-from-exile path (Highway Robbery's
        # own need: Plot waives the mana cost, not other costs a normal
        # cast's extra_legal already checks). Re-plot, then simulate an
        # extra_legal that's never satisfiable.
        game.EFFECT_REGISTRY[EffectId.FILLER] = {
            "cast": {"resolve": lambda s, c: None, "extra_legal": lambda s: False},
            "plot": {"cost": PLOT_COST, "resolve": lambda s, c: (s.hand.remove(c), s.exile.append((c, s.turn_number)))},
        }
        state = GameState(on_the_play=True)
        state.phase = game.turn.Phase.MAIN1
        state.hand = [plot_spell]
        state.battlefield = [
            Permanent(CardDef("Swamp", CardType.LAND, None, EffectId.SWAMP)),
            Permanent(CardDef("Swamp", CardType.LAND, None, EffectId.SWAMP)),
        ]
        _plot_execute("Fake Plot Spell", PLOT_COST, game.EFFECT_REGISTRY[EffectId.FILLER]["plot"]["resolve"])(state)
        while state.pending_resolution is not None:
            tap_opts = game.tap_cost_options(state)
            if tap_opts:
                name, _color, is_filter = tap_opts[0]
                game.execute_tap_cost_option(state, name, None, is_filter)
            else:
                game.execute_pool_spend(state, game.pool_spend_options(state)[0])
        state.turn_number += 1
        assert not _cast_from_exile_legal("Fake Plot Spell", game.EFFECT_REGISTRY[EffectId.FILLER]["cast"]["extra_legal"], game.turn.Speed.SORCERY)(state)
    finally:
        game.CARD_DEFS.clear()
        game.CARD_DEFS.update(_card_defs_backup)
        game.EFFECT_REGISTRY[EffectId.FILLER] = _filler_backup
        game.EFFECT_REGISTRY[EffectId.GENEROUS_ENT] = _generous_ent_backup

    print("drl_env.py Plot + on-cast-trigger self-check: OK")

    # Regression: on_cast_trigger (Guttersnipe) must only fire once a
    # spell's cost is actually, irreversibly paid. Casting a spell and
    # then choosing "Abandon payment" (game.abandon_pay_cost) must NOT
    # have collected the trigger for free -- and must be repeatable
    # without ever firing it, since the card never actually left hand and
    # no mana was ever spent. Before the fix, _cast_execute fired
    # on_cast_trigger BEFORE begin_pay_cost even started, so this exact
    # cast-then-abandon loop collected Guttersnipe's damage for free,
    # indefinitely.
    _card_defs_backup = dict(game.CARD_DEFS)
    _filler_backup = game.EFFECT_REGISTRY[EffectId.FILLER]
    _generous_ent_backup = game.EFFECT_REGISTRY[EffectId.GENEROUS_ENT]

    on_cast_calls = []
    fake_bolt = CardDef("Fake Bolt", CardType.INSTANT, {"B": 1}, EffectId.FILLER)
    game.CARD_DEFS["Fake Bolt"] = fake_bolt
    game.EFFECT_REGISTRY[EffectId.FILLER] = {"cast": {"resolve": lambda s, c: None}}
    game.EFFECT_REGISTRY[EffectId.GENEROUS_ENT] = {
        "on_cast": lambda s, permanent: on_cast_calls.append(permanent.card_def.name),
    }
    try:
        state = GameState(on_the_play=True)
        state.phase = game.turn.Phase.MAIN1
        state.hand = [fake_bolt]
        state.battlefield = [
            Permanent(CardDef("Swamp", CardType.LAND, None, EffectId.SWAMP)),
            Permanent(CardDef("Guttersnipe-ish", CardType.CREATURE, None, EffectId.GENEROUS_ENT)),
        ]
        cast_legal = _cast_legal("Fake Bolt", None, game.turn.Speed.INSTANT)
        cast_execute = _cast_execute("Fake Bolt", game.EFFECT_REGISTRY[EffectId.FILLER]["cast"]["resolve"])

        for _ in range(5):  # "infinitely" -- a handful of reps proves the loop, not just one
            assert cast_legal(state)
            cast_execute(state)
            assert on_cast_calls == []  # not fired yet -- cost isn't paid
            assert state.pending_resolution["kind"] == "pay_cost"
            game.abandon_pay_cost(state)
            assert state.pending_resolution is None
            assert on_cast_calls == []  # declining payment must never have collected it
            assert state.hand == [fake_bolt]  # never actually cast -- still sitting in hand

        # Actually pay this time -- now, and only now, it fires (once).
        assert cast_legal(state)
        cast_execute(state)
        while state.pending_resolution is not None:
            tap_opts = game.tap_cost_options(state)
            if tap_opts:
                name, color, is_filter = tap_opts[0]
                game.execute_tap_cost_option(state, name, color, is_filter)
            else:
                # A tap only floats mana into the pool -- never auto-spends
                # it toward the cost (mana.py's own design, see the Plot
                # check above) -- spend it explicitly.
                game.execute_pool_spend(state, game.pool_spend_options(state)[0])
        assert on_cast_calls == ["Guttersnipe-ish"]
        assert len(state.stack) == 1
    finally:
        game.CARD_DEFS.clear()
        game.CARD_DEFS.update(_card_defs_backup)
        game.EFFECT_REGISTRY[EffectId.FILLER] = _filler_backup
        game.EFFECT_REGISTRY[EffectId.GENEROUS_ENT] = _generous_ent_backup

    print("drl_env.py abandon-payment on-cast-trigger regression: OK")

    # Tokens (item 8): build_action_table's token_card_defs param is what
    # actually makes "Activate Blood (sac)" exist as an action at all --
    # "Blood" is never a decklist name, so the plain distinct_names-driven
    # loop alone (used by every other activated ability) can't find it.
    empty_decklist = []
    no_token_actions = build_action_table(empty_decklist, game.EFFECT_REGISTRY)
    assert not any("Blood" in nm for nm, _l, _e in no_token_actions)  # opt-in: omitted => absent, zero effect on existing decks

    token_actions = build_action_table(empty_decklist, game.EFFECT_REGISTRY, token_card_defs=(game.BLOOD_TOKEN_CARD_DEF,))
    activate_name, activate_legal, activate_execute = next(
        (nm, lg, ex) for nm, lg, ex in token_actions if nm == "Activate Blood (sac)"
    )

    state = GameState(on_the_play=True)
    game.create_token(state, game.BLOOD_TOKEN_CARD_DEF)
    state.battlefield.append(Permanent(CardDef("Swamp", CardType.LAND, None, EffectId.SWAMP)))
    state.hand = [CardDef("Card To Discard", CardType.SORCERY, {}, None)]
    state.library = [CardDef("Library Card", CardType.SORCERY, {}, None)]

    assert activate_legal(state)
    activate_execute(state)  # pays {1} via the real begin_pay_cost path, same as every other cost_key ability
    assert state.pending_resolution["kind"] == "pay_cost"
    tap_name, tap_color, tap_filter = game.tap_cost_options(state)[0]
    game.execute_tap_cost_option(state, tap_name, tap_color, tap_filter)
    if state.pending_resolution is not None:
        # Swamp produces B, which floats into the pool instead of
        # auto-filling the {generic:1} need -- same lesson as the Plot
        # check above. Spend it explicitly.
        game.execute_pool_spend(state, game.pool_spend_options(state)[0])

    assert state.pending_resolution["kind"] == "discard"  # Blood's own effect: discard a card
    game.execute_discard_option(state, "Card To Discard")
    assert state.pending_resolution is None
    assert [p.card_def.name for p in state.battlefield] == ["Swamp"]  # Blood is gone, never added to any zone
    assert [c.name for c in state.hand] == ["Library Card"]  # discarded one, drew one

    print("drl_env.py tokens self-check: OK")


    # Cross-player targeting (docs/COMBAT_PLAN.md): build_action_table's
    # opponent_decklist/opponent_token_card_defs params register "Choose
    # opponent's: X (slot k)" actions from the OTHER side's own card pool
    # -- blocking's first consumer, but exercised standalone here since
    # blocking itself isn't built yet. Boggles on both sides -- what's
    # under test is MY OWN action table's opponent-facing entries, not
    # anything about my own cards, so a real decklist with real creature
    # quantities on both sides is all this needs.
    boggles_decklist = game.parse_decklist_file(os.path.join(os.path.dirname(__file__), "..", "data", "boggles.txt"))

    no_opponent_actions = build_action_table(boggles_decklist, game.EFFECT_REGISTRY)
    assert not any(nm.startswith("Choose opponent's:") for nm, _l, _e in no_opponent_actions)  # 1p mode: never registered at all

    my_actions = build_action_table(boggles_decklist, game.EFFECT_REGISTRY, opponent_decklist=boggles_decklist)
    bogle_slot_actions = [nm for nm, _l, _e in my_actions if nm.startswith("Choose opponent's: Slippery Bogle")]
    assert bogle_slot_actions == [f"Choose opponent's: Slippery Bogle (slot {k})" for k in range(1, 5)]  # boggles.txt: 4 copies
    assert not any(nm.startswith("Choose opponent's: Forest") for nm, _l, _e in my_actions)  # a land, never a targetable creature

    def _midx(action_name):
        return next(i for i, (nm, _l, _e) in enumerate(my_actions) if nm == action_name)

    target_slot_2 = _midx("Choose opponent's: Slippery Bogle (slot 2)")
    target_slot_1 = _midx("Choose opponent's: Slippery Bogle (slot 1)")

    attacker_bogle_1 = Permanent(game.CARD_DEFS["Slippery Bogle"])
    attacker_bogle_2 = Permanent(game.CARD_DEFS["Slippery Bogle"])
    attacker_bogle_2.slot = 2
    state = GameState(on_the_play=True, players=[PlayerState(True), PlayerState(False)])
    state.players[0].battlefield = [attacker_bogle_1, attacker_bogle_2]
    state.active_idx = 1  # simulating the defender's own already-flipped perspective (see game.begin_choose_opponent_permanent's own docstring)

    _, legal_slot_2, execute_slot_2 = my_actions[target_slot_2]
    _, legal_slot_1, _ = my_actions[target_slot_1]
    assert not legal_slot_2(state) and not legal_slot_1(state)  # nothing pending yet

    completed = []
    game.begin_choose_opponent_permanent(
        state, lambda p: p.card_def.card_type == game.CardType.CREATURE, lambda s, choice: completed.append(choice),
    )
    assert legal_slot_1(state) and legal_slot_2(state)
    execute_slot_2(state)
    assert completed == [("Slippery Bogle", 2)]  # the specific slot targeted, not an arbitrary same-named match
    assert not legal_slot_2(state)  # resolution is complete, nothing pending anymore

    print("drl_env.py cross-player targeting self-check: OK")

    # Turn-owner / priority-holder split (docs/PRIORITY_PLAN.md item 0):
    # _land_drop_legal (via speed_legal) and _attack_legal must both
    # refuse the non-turn player even when state.phase/state.
    # lands_played_this_turn/their own eligible creature would otherwise
    # look legal -- simulates a priority consult (active_idx flipped away
    # from turn_player_idx) without needing the full priority round built
    # yet.
    play_forest_idx = _midx("Play land: Forest")
    attack_bogle_idx = _midx("Attack: Slippery Bogle (slot 1)")
    _, play_forest_legal, _ = my_actions[play_forest_idx]
    _, attack_bogle_legal, _ = my_actions[attack_bogle_idx]

    turn_owner_state = GameState(on_the_play=True, players=[PlayerState(True), PlayerState(False)])
    turn_owner_state.turn_player_idx = 0
    turn_owner_state.active_idx = 0
    turn_owner_state.phase = game.turn.Phase.DECLARE_ATTACKERS
    attacking_bogle = Permanent(game.CARD_DEFS["Slippery Bogle"])
    attacking_bogle.summoning_sick = False
    turn_owner_state.players[0].hand = [game.CARD_DEFS["Forest"]]
    turn_owner_state.players[0].battlefield = [attacking_bogle]
    assert attack_bogle_legal(turn_owner_state)  # the turn player's own creature, their own DECLARE_ATTACKERS -- legal

    turn_owner_state.phase = game.turn.Phase.MAIN1
    assert play_forest_legal(turn_owner_state)  # the turn player's own MAIN1, land in hand, none played yet -- legal

    turn_owner_state.active_idx = 1  # simulating a priority consult of the OTHER player
    turn_owner_state.players[1].hand = [game.CARD_DEFS["Forest"]]  # even with their OWN land available
    assert not play_forest_legal(turn_owner_state)  # refused -- not their turn, regardless of their own hand/lands_played_this_turn

    turn_owner_state.phase = game.turn.Phase.DECLARE_ATTACKERS
    non_turn_bogle = Permanent(game.CARD_DEFS["Slippery Bogle"])
    non_turn_bogle.summoning_sick = False
    turn_owner_state.players[1].battlefield = [non_turn_bogle]  # even with their OWN eligible creature at the same (name, slot)
    assert not attack_bogle_legal(turn_owner_state)  # refused -- declaring attackers is the turn player's own special action

    print("drl_env.py turn-owner (land drop / declare attacker) self-check: OK")


    # Blocking (docs/COMBAT_PLAN.md): build_action_table's "Assign Blocker:
    # <name> (slot j)" / "Done blocking" entries, end to end through the
    # REAL production functions (_assign_blocker_legal/_execute,
    # _done_blocking_legal/_execute) -- not a parallel reimplementation.
    # Two attacking Slippery Bogles (real power=1 stats), one defending
    # Slippery Bogle blocks only ONE of them.
    boggles_pending_kinds = game.derive_pending_kinds(boggles_decklist)
    assign_slot_1 = _midx("Assign Blocker: Slippery Bogle (slot 1)")
    done_blocking_idx = _midx("Done blocking")
    _, assign_legal, assign_execute = my_actions[assign_slot_1]
    _, done_legal, done_execute = my_actions[done_blocking_idx]

    atk_bogle_1 = Permanent(game.CARD_DEFS["Slippery Bogle"])
    atk_bogle_2 = Permanent(game.CARD_DEFS["Slippery Bogle"])
    atk_bogle_2.slot = 2
    defender_bogle = Permanent(game.CARD_DEFS["Slippery Bogle"])  # slot 1 by default
    state = GameState(on_the_play=True, players=[PlayerState(True), PlayerState(False)])
    state.players[0].battlefield = [atk_bogle_1, atk_bogle_2]
    state.players[0].attackers = [atk_bogle_1, atk_bogle_2]
    atk_bogle_1.tapped = True
    atk_bogle_2.tapped = True  # declare_attacker's own effect -- simulated directly, attacking itself isn't under test here
    state.players[1].battlefield = [defender_bogle]
    state.active_idx = 1  # simulating _declare_blockers_gen's own flip to the defender

    assert not assign_legal(state) and not done_legal(state)  # nothing pending yet

    completed = []
    game.begin_declare_blockers(state, on_complete=lambda s: completed.append(True))
    assert completed == []
    assert assign_legal(state) and done_legal(state)

    assign_execute(state)  # "Assign Blocker: Slippery Bogle (slot 1)" -- parks defender_bogle as a blocker
    assert state.pending_resolution["kind"] == "choose_opponent_permanent"
    target_slot_1 = _midx("Choose opponent's: Slippery Bogle (slot 1)")
    _, target_legal, target_execute = my_actions[target_slot_1]
    assert target_legal(state)
    target_execute(state)  # assigns it to block atk_bogle_1 specifically, not atk_bogle_2

    # Re-opened (drl_env._assign_blocker_execute's own nested on_complete):
    # defender_bogle is now spoken for, so it's no longer offered again --
    # confirms creature_block_eligible actually gates the SAME action a
    # second time, not just the first.
    assert state.pending_resolution["kind"] == "declare_blockers"
    assert not assign_legal(state)
    assert done_legal(state)
    done_execute(state)  # "Done blocking" -- atk_bogle_2 goes unblocked
    assert completed == [True]
    assert state.pending_resolution is None
    assert state.players[0].blocked_by == {atk_bogle_1: defender_bogle}


    print("drl_env.py blocking self-check: OK")

    # Flying (docs/COMBAT_PLAN.md step 7): _assign_blocker_execute's own
    # extra_predicate (game.has_keyword), end to end through the REAL
    # action table -- Silhana Ledgewalker (real "can't be blocked except
    # by creatures with flying," modeled as the "flying" keyword) can only
    # be blocked by a creature that itself has flying (Kitchen Imp, real
    # flying) -- a plain Slippery Bogle is otherwise a perfectly legal
    # (untapped) blocker, but can never be assigned to THIS specific
    # attacker. Mixes cards from different real color catalogs
    # (green/multicolor + black) purely to exercise the engine mechanism
    # -- not a claim either card is ever actually run together in a real
    # deck.
    flying_decklist = [("Silhana Ledgewalker", 2), ("Slippery Bogle", 2), ("Kitchen Imp", 2)]
    flying_actions = build_action_table(flying_decklist, game.EFFECT_REGISTRY, opponent_decklist=flying_decklist)

    def _fidx(action_name):
        return next(i for i, (nm, _l, _e) in enumerate(flying_actions) if nm == action_name)

    _, bogle_legal, bogle_execute = flying_actions[_fidx("Assign Blocker: Slippery Bogle (slot 1)")]
    _, imp_legal, imp_execute = flying_actions[_fidx("Assign Blocker: Kitchen Imp (slot 1)")]

    attacking_ledgewalker = Permanent(game.CARD_DEFS["Silhana Ledgewalker"])
    attacking_ledgewalker.tapped = True  # already attacked
    defending_bogle = Permanent(game.CARD_DEFS["Slippery Bogle"])
    defending_imp = Permanent(game.CARD_DEFS["Kitchen Imp"])
    state = GameState(on_the_play=True, players=[PlayerState(True), PlayerState(False)])
    state.players[0].battlefield = [attacking_ledgewalker]
    state.players[0].attackers = [attacking_ledgewalker]
    state.players[1].battlefield = [defending_bogle, defending_imp]
    state.active_idx = 1  # simulating _declare_blockers_gen's own flip to the defender

    game.begin_declare_blockers(state, on_complete=lambda s: None)
    assert bogle_legal(state) and imp_legal(state)  # both otherwise-eligible blockers (untapped, unused)

    bogle_execute(state)  # parks the Bogle -- but it can't legally block a flyer, so this fizzles
    assert state.pending_resolution["kind"] == "declare_blockers"  # re-opened, nothing left pending
    assert state.players[0].blocked_by == {}  # nothing assigned -- the Bogle was never a legal choice for this attacker

    imp_execute(state)  # Kitchen Imp HAS flying -- opens a real nested choice
    assert state.pending_resolution["kind"] == "choose_opponent_permanent"
    assert game.choose_opponent_permanent_options(state) == [("Silhana Ledgewalker", 1)]
    game.execute_choose_opponent_permanent_option(state, "Silhana Ledgewalker", 1)
    assert state.players[0].blocked_by == {attacking_ledgewalker: defending_imp}

    print("drl_env.py flying self-check: OK")

    # Targeting (real MTG rule, per drl_env._precast_choice_execute /
    # game.effects.casting.cast_aura's own docstrings): a target is chosen once,
    # at cast time, exact (name, slot) addressed -- not just by name -- and
    # re-validated by identity only once the spell resolves off the stack.
    # End to end through the REAL action table: "Cast Rancor" pays its {G}
    # cost, then (precast_choice, not deferred) immediately opens
    # choose_permanent with BOTH Slippery Bogles offered by their own
    # distinct slot; "Choose target: Slippery Bogle (slot 2)" picks the
    # specific one, which pushes to the stack (not yet attached, still in
    # hand); resolving the stack attaches it to exactly the one chosen, not
    # an arbitrary same-named match.
    targeting_decklist = [("Slippery Bogle", 2), ("Rancor", 2), ("Forest", 10)]
    targeting_actions = build_action_table(targeting_decklist, game.EFFECT_REGISTRY)

    def _gidx(action_name):
        return next(i for i, (nm, _l, _e) in enumerate(targeting_actions) if nm == action_name)

    bogle_1 = Permanent(game.CARD_DEFS["Slippery Bogle"])
    bogle_2 = Permanent(game.CARD_DEFS["Slippery Bogle"])
    bogle_2.slot = 2
    forest = Permanent(game.CARD_DEFS["Forest"])
    rancor_card = game.CARD_DEFS["Rancor"]
    state = GameState(on_the_play=True)
    state.phase = game.turn.Phase.MAIN1  # sorcery-speed cast requires this -- GameState defaults phase=None
    state.battlefield = [bogle_1, bogle_2, forest]
    state.hand = [rancor_card]

    _, cast_rancor_legal, cast_rancor_execute = targeting_actions[_gidx("Cast Rancor")]
    assert cast_rancor_legal(state)
    cast_rancor_execute(state)
    assert state.pending_resolution["kind"] == "pay_cost"
    targeting_actions[_gidx("Choose: Forest")][2](state)  # tap the Forest -- floats {G}
    targeting_actions[_gidx("Spend G from pool")][2](state)  # pays Rancor's {G}

    # Cost fully paid -- precast_choice means cast_aura runs its target
    # choice IMMEDIATELY here, NOT deferred to when this eventually pops
    # off the stack (that's the whole point of this redesign).
    assert state.pending_resolution["kind"] == "choose_permanent"
    assert set(game.choose_permanent_options(state)) == {("Slippery Bogle", 1), ("Slippery Bogle", 2)}

    _, choose_slot_2_legal, choose_slot_2_execute = targeting_actions[_gidx("Choose target: Slippery Bogle (slot 2)")]
    assert choose_slot_2_legal(state)
    choose_slot_2_execute(state)
    # Target chosen -- pushed to the stack, not yet attached (still
    # physically in hand, same "still in hand while on stack" convention
    # every other cast path here follows).
    assert state.pending_resolution is None
    assert state.hand == [rancor_card] and len(state.stack) == 1

    game.resolve_top_of_stack(state)
    assert state.hand == []
    rancor_permanent = next(p for p in state.battlefield if p.card_def.name == "Rancor")
    assert rancor_permanent.flags["enchanting"] is bogle_2  # the SPECIFIC one chosen -- not bogle_1, despite the identical name

    print("drl_env.py Aura targeting (exact slot addressing) self-check: OK")

    # Fizzle, same end-to-end path: the exact chosen permanent (bogle_1
    # this time) is gone by the time the cast resolves -- the whole spell
    # fails, no effect, straight to the graveyard, never attaches.
    state = GameState(on_the_play=True)
    state.phase = game.turn.Phase.MAIN1
    bogle_1 = Permanent(game.CARD_DEFS["Slippery Bogle"])
    forest = Permanent(game.CARD_DEFS["Forest"])
    state.battlefield = [bogle_1, forest]
    state.hand = [rancor_card]

    targeting_actions[_gidx("Cast Rancor")][2](state)
    targeting_actions[_gidx("Choose: Forest")][2](state)
    targeting_actions[_gidx("Spend G from pool")][2](state)
    targeting_actions[_gidx("Choose target: Slippery Bogle (slot 1)")][2](state)
    assert len(state.stack) == 1
    state.battlefield.remove(bogle_1)  # dies before the cast resolves

    game.resolve_top_of_stack(state)
    assert state.hand == []
    assert rancor_card in state.graveyard
    assert not any(p.card_def.name == "Rancor" for p in state.battlefield)

    print("drl_env.py Aura target-fizzle (end to end) self-check: OK")

    # _lost: true once someone has won and it wasn't seat_idx -- still used
    # directly by token_train.py's own reward attribution.
    assert _lost(type("S", (), {"winner": 1})(), 0) is True
    assert _lost(type("S", (), {"winner": 0})(), 0) is False
    assert _lost(type("S", (), {"winner": None})(), 0) is False
    print("drl_env.py _lost self-check: OK")

    # tap_cost_options memoization never returns a stale answer (docs/
    # PRIORITY_PLAN.md item 6): build a pay_cost resolution with exactly 1
    # untapped Mountain, sweep the mask (populating the cache -- "Choose:
    # Mountain" legal), tap it (a real mutation -- zero untapped sources
    # left, so tap_cost_options itself now returns empty), then sweep
    # again -- the second sweep must see the mutation, not the first
    # sweep's cached answer, proving the cache doesn't leak across
    # separate legal_action_mask calls.
    perf_decklist = [("Mountain", 10), ("Lightning Bolt", 5)]
    perf_pending = game.derive_pending_kinds(perf_decklist)
    perf_actions = build_action_table(perf_decklist, game.EFFECT_REGISTRY, pending_kinds=perf_pending)
    perf_choose_mountain = next(i for i, (nm, _l, _e) in enumerate(perf_actions) if nm == "Choose: Mountain")
    perf_state = GameState(on_the_play=True, players=[PlayerState(True)])
    perf_state.hand = [game.CARD_DEFS["Lightning Bolt"]]
    perf_state.battlefield = [Permanent(game.CARD_DEFS["Mountain"])]
    game.begin_pay_cost(perf_state, {"R": 1}, on_complete=lambda s: None)
    assert legal_action_mask(perf_state, perf_actions)[perf_choose_mountain]
    assert _tap_cost_options_cache is None  # cleared again once the sweep itself returns
    game.execute_tap_cost_option(perf_state, "Mountain", None, False)  # taps the only Mountain -- 0 untapped sources left
    assert game.tap_cost_options(perf_state) == []  # ground truth: nothing left to tap
    assert not legal_action_mask(perf_state, perf_actions)[perf_choose_mountain]  # would be wrongly True if the first sweep's stale cache leaked through

    print("drl_env.py tap_cost_options cache self-check: OK")

