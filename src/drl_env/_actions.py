"""The token/attention DRL action layer: per-action legal/execute/options
factories, build_action_table (assembles the flat fixed action table), and
legal_action_mask (+ its sweep-scoped caches). Split out of drl_env/__init__
unchanged; re-exported there so `drl_env.X` keeps working."""

import numpy as np

import game


# ---------------------------------------------------------------------------
# Action table: generated from a decklist + game.EFFECT_REGISTRY instead of
# hand-typed -- this, plus the pending-resolution machinery in the game
# engine, is what makes a deck built entirely from already-implemented cards
# need zero new code here.
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
#      exactly which physical permanent was chosen.
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
            # state.turn_player_idx, so this alone
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
    """How many copies of `name` in state.hand are castable right now -- just
    the count in hand. A spell LEAVES hand the instant it is put on the stack
    (game.effects.stack.push_to_stack removes it at cast), so a copy already
    paid-for and awaiting resolution is simply no longer in state.hand and can't
    be re-cast -- the card physically leaving hand IS the re-cast guard. (Before
    that, a cast copy stayed in hand until its resolve ran, so this had to
    subtract on-stack copies by hand to stop the model casting the same physical
    copy twice -- which still crashed once two entries both tried to remove it.
    That bookkeeping is now subsumed by the faithful zone move.)"""
    return sum(1 for c in state.hand if c.name == name)


def _effective_cast_cost(state, card_def):
    """A card's cast cost after any registry "cost_reduction" -- a
    lambda(state) -> int (affinity = # artifacts you control; the graveyard
    instant/sorcery count for Tolarian Terror / Cryptic Serpent; cards drawn
    this turn for Deem Inferior). The reduction lowers ONLY the generic pips,
    floored at 0 -- colored pips are never reduced (real "costs {N} less").
    A card with no such spec (every existing card, and every G1-G6 card) pays
    card_def.cast_cost unchanged, so this is a transparent no-op for them --
    the single reason it's safe to route the whole cast path through it."""
    cost = card_def.cast_cost
    spec = game.EFFECT_REGISTRY.get(card_def.effect_id, {}).get("cost_reduction")
    if spec is None or cost is None:
        return cost
    reduction = spec(state)
    if reduction <= 0:
        return cost
    reduced = dict(cost)
    reduced["generic"] = max(0, reduced.get("generic", 0) - reduction)
    return reduced


def _cast_legal(name, extra_legal, speed):
    def legal(state):
        if state.pending_resolution is not None:
            return False
        if not game.turn.speed_legal(state, speed):
            return False
        if _hand_count_available(state, name) <= 0:
            return False
        card_def = game.CARD_DEFS[name]
        if game.plan_payment(state, _effective_cast_cost(state, card_def)) is None:
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
        game.begin_pay_cost(state, _effective_cast_cost(state, card_def), on_complete=_after_pay)
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
        game.begin_pay_cost(state, _effective_cast_cost(state, card_def), on_complete=_after_pay)
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


def _delve_reduced_cost(card_def, n):
    """Delve pays {1} of the generic cost per graveyard card exiled -- reduce
    the generic pips by n, floored at 0 (colored pips untouched)."""
    cost = dict(card_def.cast_cost)
    cost["generic"] = max(0, cost.get("generic", 0) - n)
    return cost


def _delve_legal(name, n, speed):
    """Cast this spell delving exactly n cards -- legal only with n cards in
    the graveyard to exile AND the {generic-n} remainder affordable. One
    action per n (0..delve["max"]); plan_payment masks the unaffordable ones,
    same as x_cast_modes' own per-X enumeration."""
    def legal(state):
        if state.pending_resolution is not None:
            return False
        if not game.turn.speed_legal(state, speed):
            return False
        if _hand_count_available(state, name) <= 0:
            return False
        if len(state.graveyard) < n:
            return False
        return game.plan_payment(state, _delve_reduced_cost(game.CARD_DEFS[name], n)) is not None
    legal._pending_gate = _GATE_NO_PENDING
    return legal


def _delve_execute(name, n, resolve):
    """Exile n graveyard cards (the model chooses which -- begin_exile_n_from_
    graveyard), then pay the {generic-n} remainder, then cast normally. The
    exile is a cost, paid first; abandoning the mana payment afterward leaves
    those cards exiled (same as any other paid cost)."""
    def execute(state):
        card_def = game.CARD_DEFS[name]

        def _after_exile(s):
            def _after_pay(s2):
                game.on_cast_trigger(s2, card_def)
                game.push_to_stack(s2, card_def, resolve)
            game.begin_pay_cost(s, _delve_reduced_cost(card_def, n), on_complete=_after_pay)

        game.begin_exile_n_from_graveyard(state, n, _after_exile)
    return execute


def _tuck_position_legal(state):
    pending = state.pending_resolution
    return pending is not None and pending["kind"] == "tuck_position"


_tuck_position_legal._pending_gate = frozenset({"tuck_position"})


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
    if state.pending_resolution is not None:
        return False
    # Goad (Undercity Arena): a goaded creature that CAN attack must be
    # declared -- its controller may not end their own declare-attackers step
    # (Pass) while one is still able and undeclared. game.has_unfulfilled_goad
    # only ever returns True during DECLARE_ATTACKERS for the turn player, so
    # this is a no-op everywhere else.
    if state.phase is game.turn.Phase.DECLARE_ATTACKERS and game.has_unfulfilled_goad(state):
        return False
    return True


_pass_legal._pending_gate = _GATE_NO_PENDING


def _pass_execute(state):
    pass  # no-op: a Pass is signalled by choose_action returning None (the game loop then advances), never by invoking this execute fn


def _choose_name_options(state):
    """Plain (uncolored) 'Choose: X' names currently legal, given whatever
    kind of pending resolution -- if any -- is active. "choose_permanent"
    is NOT handled here -- see _choose_permanent_legal/_choose_permanent_
    execute below: it needs exact (name, slot) addressing (docs/
    "Permanent identity"), same as
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
    if kind == "throne_reveal":  # Undercity Throne: pick a creature card from the revealed top 10
        return game.throne_reveal_options(state)
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
    if kind == "ancient_stirrings":
        return [n for n in game.ancient_stirrings_options(state) if n != "decline"]
    if kind == "malevolent_rumble":
        return [n for n in game.malevolent_rumble_options(state) if n != "decline"]
    if kind in ("scry", "surveil") and pending["ordered"] is not None:
        return game.scry_surveil_options(state)
    if kind == "select_to_hand" and pending["ordered"] is not None:
        return game.select_to_hand_options(state)  # ordering phase only -- "keep"/"bottom" are their own actions
    if kind == "order_triggers":
        return game.order_triggers_options(state)
    if kind == "put_on_top":  # Brainstorm: which hand card to put on top next
        return game.put_on_top_options(state)
    if kind == "ponder":  # Ponder: which revealed card to place on top next ("Shuffle (Ponder)" is its own action)
        return game.ponder_options(state)
    if kind == "choose_stack_target":  # Counterspell/Dispel/Spell Pierce: which spell on the stack to counter
        return game.choose_stack_target_options(state)
    return []


def _choose_name_legal(name):
    def legal(state):
        return name in _choose_name_options(state)
    # Matches _choose_name_options' own dispatch table above exactly --
    # every pending kind that function ever returns a non-empty list for.
    legal._pending_gate = frozenset({
        "pay_cost", "search_fetch", "throne_reveal", "choose_graveyard_card", "sacrifice", "discard",
        "discard_or_sacrifice", "ancient_stirrings", "malevolent_rumble", "scry", "surveil",
        "select_to_hand", "order_triggers", "put_on_top", "ponder", "choose_stack_target",
    })
    return legal


def _choose_name_execute(name):
    def execute(state):
        kind = state.pending_resolution["kind"]
        if kind == "pay_cost":
            game.execute_tap_cost_option(state, name, None, False)
        elif kind == "search_fetch":
            game.execute_search_fetch_option(state, name)
        elif kind == "throne_reveal":
            game.execute_throne_reveal_option(state, name)
        elif kind == "choose_graveyard_card":
            game.execute_choose_graveyard_card_option(state, name)
        elif kind == "sacrifice":
            game.execute_sacrifice_option(state, name)
        elif kind == "discard":
            game.execute_discard_option(state, name)
        elif kind == "discard_or_sacrifice":
            game.execute_discard_or_sacrifice_option(state, "discard", name)
        elif kind == "ancient_stirrings":
            game.execute_ancient_stirrings_option(state, name)
        elif kind == "malevolent_rumble":
            game.execute_malevolent_rumble_option(state, name)
        elif kind == "select_to_hand":
            game.execute_select_to_hand_option(state, name)  # ordering phase only
        elif kind == "order_triggers":
            game.execute_order_triggers_option(state, name)
        elif kind == "put_on_top":
            game.execute_put_on_top_option(state, name)
        elif kind == "ponder":
            game.execute_ponder_option(state, name)
        elif kind == "choose_stack_target":
            game.execute_choose_stack_target_option(state, name)
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
   ) -- declaring an attacker is a turn-based
    special action, not a priority action, so the non-turn player must
    never be allowed to declare one just because state.phase (a single
    shared field describing the TURN's phase) happens to match during
    their own priority window. And only if the specific physical
    permanent occupying this (name, slot)
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
    "Permanent identity") -- a plain by-name "Choose:
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
    -- legal only while a "choose_opponent_permanent"
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
    """One "Assign Blocker: <name> (slot j)" action -- legal only while a "declare_blockers" resolution
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
    means an attacker with flying can only be
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
    if pending is None or pending["kind"] != "declare_blockers":
        return False
    # Menace (509.1c): can't FINISH a block declaration that leaves a menace
    # attacker blocked by exactly one creature -- the defender must add a
    # second blocker or unassign the lone one first.
    return not game.menace_block_incomplete(state)


_done_blocking_legal._pending_gate = frozenset({"declare_blockers"})


def _done_blocking_execute(state):
    game.complete_resolution(state)


def _unassign_blocker_legal(name, slot):
    """One "Unassign Blocker: <name> (slot j)" action -- take a committed
    blocker back OUT of the block declaration (before Done). Legal while a
    declare_blockers resolution is pending and the specific physical permanent
    at this (name, slot) is currently committed as a blocker. Exists so the
    menace 0-or-2+ rule (see _done_blocking_legal) is never a softlock: the
    defender can always rearrange a lone menace-block into a legal declaration."""
    def legal(state):
        pending = state.pending_resolution
        if pending is None or pending["kind"] != "declare_blockers":
            return False
        p = _cached_battlefield_lookup(state).get((name, slot))
        return p is not None and any(p in blockers for blockers in state.opponent.blocked_by.values())
    legal._pending_gate = frozenset({"declare_blockers"})
    return legal


def _unassign_blocker_execute(name, slot):
    def execute(state):
        p = next(pp for pp in state.battlefield if pp.card_def.name == name and pp.slot == slot)
        blocked_by = state.opponent.blocked_by
        for attacker, blockers in list(blocked_by.items()):
            if p in blockers:
                blockers.remove(p)
                if not blockers:
                    del blocked_by[attacker]
                state.log_event("block_unassigned", blocker=(name, slot), attacker=(attacker.card_def.name, attacker.slot))
                break
    return execute


def _assign_damage_to_opponent_legal(state):
    """The trample "assign this combat-damage point to the defending player"
    action -- legal only during an assign_combat_damage resolution whose
    attacker HAS trample and still has points to assign (the blockers are
    the pointer half of this decision; this fixed action is the player
    half). NOT a targeting-prefixed name, so build_fixed_action_table keeps
    it in the fixed table rather than stripping it to the pointer scheme."""
    pending = state.pending_resolution
    return (
        pending is not None and pending["kind"] == "assign_combat_damage"
        and pending["has_trample"] and pending["remaining"] > 0
    )


_assign_damage_to_opponent_legal._pending_gate = frozenset({"assign_combat_damage"})


def _assign_damage_to_opponent_execute(state):
    game.execute_assign_combat_damage_to_player(state)


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


# NOTE: the _mulligan_decision_legal / _mulligan_take_legal / _mulligan_keep_execute
# / _mulligan_take_execute helpers were removed with the pregame fixed-table actions
# (harness refactor Phase 4) -- the MulliganNet (rl.mulligan) owns the pregame phase
# now, so nothing in the action table references them. The engine's own mulligan
# (game.execute_mulligan_keep/take, game.turn.run_mulligan_phase) is untouched.


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


def _ponder_shuffle_legal(state):
    """Ponder's "you may shuffle" -- an alternative to ordering the revealed
    cards, so legal only BEFORE any card has been placed on top (ordered
    still empty). Once ordering has begun, that choice is made."""
    pending = state.pending_resolution
    return pending is not None and pending["kind"] == "ponder" and not pending["ordered"]


_ponder_shuffle_legal._pending_gate = frozenset({"ponder"})


def _ponder_shuffle_execute(state):
    game.execute_ponder_shuffle(state)


def _pay_unless_pay_legal(state):
    """"Pay {N}" for the Spell Pierce / Ward rider -- legal only while a
    pay_unless resolution is open AND the payer can actually afford the {N}
    (active_idx is already flipped to the payer, so plan_payment reads THEIR
    sources)."""
    pending = state.pending_resolution
    if pending is None or pending["kind"] != "pay_unless":
        return False
    return game.plan_payment(state, pending["cost"]) is not None


_pay_unless_pay_legal._pending_gate = frozenset({"pay_unless"})


def _pay_unless_pay_execute(state):
    game.pay_unless_pay(state)


def _pay_unless_decline_legal(state):
    pending = state.pending_resolution
    return pending is not None and pending["kind"] == "pay_unless"


_pay_unless_decline_legal._pending_gate = frozenset({"pay_unless"})


def _pay_unless_decline_execute(state):
    game.pay_unless_decline(state)


def _may_transform_legal(state):
    pending = state.pending_resolution
    return pending is not None and pending["kind"] == "may_transform"


_may_transform_legal._pending_gate = frozenset({"may_transform"})


def _may_copy_legal(state):
    pending = state.pending_resolution
    return pending is not None and pending["kind"] == "may_copy"


_may_copy_legal._pending_gate = frozenset({"may_copy"})


def _choose_room_legal(room):
    def legal(state):
        pending = state.pending_resolution
        return pending is not None and pending["kind"] == "choose_room" and room in pending["options"]
    legal._pending_gate = frozenset({"choose_room"})
    return legal


def _choose_room_execute(room):
    return lambda state: game.execute_choose_room_option(state, room)


def _choose_mana_color_legal(color):
    def legal(state):
        pending = state.pending_resolution
        return pending is not None and pending["kind"] == "choose_mana_color"
    legal._pending_gate = frozenset({"choose_mana_color"})
    return legal


def _choose_mana_color_execute(color):
    return lambda state: game.execute_choose_mana_color(state, color)


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


def _decline_graveyard_card_legal(state):
    """Only for an OPTIONAL choose_graveyard_card (Masked Vandal's "you may
    exile a creature card from your graveyard") with real options to decline
    -- gated on pending["optional"] so it never appears for Dread Return /
    Relic's own MANDATORY graveyard picks, which share the same kind."""
    pending = state.pending_resolution
    return (
        pending is not None and pending["kind"] == "choose_graveyard_card" and pending.get("optional")
        and bool(game.choose_graveyard_card_options(state))
    )


_decline_graveyard_card_legal._pending_gate = frozenset({"choose_graveyard_card"})


def _decline_graveyard_card_execute(state):
    game.execute_choose_graveyard_card_decline(state)


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


def _target_any_self_legal(state):
    """The "any target" player half (real Magic: a player is always a legal
    "any target", including yourself -- Lightning Bolt to your own face is
    legal, if rarely wise). Only offered when the pending choose_any_target
    allows players (a "target creature"-only choice sets allow_players=False
    and this stays masked). The creature half of the same choice rides the
    identity pointer scheme (rl.action_bridge), not a fixed action."""
    pending = state.pending_resolution
    return pending is not None and pending["kind"] == "choose_any_target" and pending["allow_players"]


_target_any_self_legal._pending_gate = frozenset({"choose_any_target"})


def _target_any_self_execute(state):
    game.execute_choose_any_target_player(state, state.active_idx)


def _target_any_opponent_legal(state):
    pending = state.pending_resolution
    return (
        pending is not None and pending["kind"] == "choose_any_target"
        and pending["allow_players"] and len(state.players) > 1
    )


_target_any_opponent_legal._pending_gate = frozenset({"choose_any_target"})


def _target_any_opponent_execute(state):
    game.execute_choose_any_target_player(state, 1 - state.active_idx)


def _target_any_decline_legal(state):
    """Decline an "up to one target" (optional) choose_any_target -- e.g.
    Pinnacle Kill-Ship's ETB choosing zero targets. Only legal when the
    pending was begun optional=True."""
    pending = state.pending_resolution
    return pending is not None and pending["kind"] == "choose_any_target" and pending.get("optional", False)


_target_any_decline_legal._pending_gate = frozenset({"choose_any_target"})


def _target_any_decline_execute(state):
    game.execute_choose_any_target_decline(state)


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


def _flashback_legal(name, ability_legal, speed, cost=None):
    """Dread Return's Flashback: cast from the graveyard, not hand. Real
    Magic: Flashback follows the same timing as the card itself, not its
    own independent rule -- speed is the same value the card's normal
    cast derived, not a separate default.

    cost (optional): a MANA cost dict for a flashback whose flashback cost
    includes mana (Deep Analysis' {1}{U}). When present, its affordability
    is checked here (plan_payment) exactly like a normal cast; the free/
    sacrifice-only flashbacks (Faithless Looting, Lava Dart, Dread Return)
    leave it None and pay entirely inside their own resolve. Any NON-mana
    additional cost (Deep Analysis' "Pay 3 life") is gated by ability_legal
    instead (it can't be expressed as a mana dict)."""
    def legal(state):
        if state.pending_resolution is not None:
            return False
        if not game.turn.speed_legal(state, speed):
            return False
        if not any(c.name == name for c in state.graveyard):
            return False
        if cost is not None and game.plan_payment(state, cost) is None:
            return False
        return ability_legal(state)
    legal._pending_gate = _GATE_NO_PENDING
    return legal


def _flashback_execute(name, resolve, cost=None):
    def execute(state):
        card_def = game.CARD_DEFS[name]
        if cost is None:
            game.on_cast_trigger(state, card_def)  # item 11 -- see _cast_execute
            resolve(state, card_def)
        else:
            # Mana flashback cost: pay it first (like a normal cast), then
            # fire the on-cast trigger and run the resolve, which pays any
            # further additional cost (life) and pushes the effect.
            def _after_pay(state):
                game.on_cast_trigger(state, card_def)
                resolve(state, card_def)
            game.begin_pay_cost(state, cost, on_complete=_after_pay)
    return execute


def _impulse_entry(state, name):
    """The topmost still-unexpired impulse entry (card_def, deadline) for
    `name`, or None. Expired entries are pruned at untap, but this also
    re-checks the deadline defensively."""
    for cd, deadline in reversed(state.impulse):
        if cd.name == name and state.turn_number <= deadline:
            return (cd, deadline)
    return None


def _play_impulse_land_legal(name):
    def legal(state):
        if state.pending_resolution is not None:
            return False
        if _impulse_entry(state, name) is None:
            return False
        if not game.turn.speed_legal(state, game.turn.Speed.SORCERY):  # playing a land is sorcery-speed, your turn
            return False
        return state.lands_played_this_turn == 0
    legal._pending_gate = _GATE_NO_PENDING
    return legal


def _play_impulse_land_execute(name):
    def execute(state):
        entry = _impulse_entry(state, name)
        state.impulse.remove(entry)
        state.hand.append(entry[0])  # source it via hand so play_land_from_hand works (Cascade-style insertion)
        game.play_land_from_hand(state, entry[0])
    return execute


def _play_impulse_cast_legal(name, cost, extra_legal, speed):
    def legal(state):
        if state.pending_resolution is not None:
            return False
        if _impulse_entry(state, name) is None:
            return False
        if not game.turn.speed_legal(state, speed):
            return False
        if game.plan_payment(state, cost) is None:
            return False
        return extra_legal is None or extra_legal(state)
    legal._pending_gate = _GATE_NO_PENDING
    return legal


def _play_impulse_cast_execute(name, cost, resolve, precast):
    """Cast an impulse-exiled spell for `cost` (its normal cost -- impulse,
    unlike Plot, is NOT free). Mirrors _cast_execute/_precast_choice_execute,
    but the card is removed from the impulse zone only in _after_pay (once
    mana is irreversibly paid) -- so abandoning payment leaves it in impulse,
    no leak. Then it's inserted into hand (Cascade-style, so the card's own
    resolve, written for a hand cast, finds and removes it) and either pushed
    to the stack (non-precast) or resolved directly (precast, which pushes
    itself)."""
    def execute(state):
        entry = _impulse_entry(state, name)

        def _after_pay(s):
            if entry in s.impulse:
                s.impulse.remove(entry)
            s.hand.append(entry[0])
            game.on_cast_trigger(s, entry[0])
            if precast:
                resolve(s, entry[0])
            else:
                game.push_to_stack(s, entry[0], resolve)

        game.begin_pay_cost(state, cost, on_complete=_after_pay)
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

        def _after_pay(s):
            # The physical hand card LEAVES hand at cast, like every other cast
            # (game.push_to_stack) -- never re-entering hand. It shares
            # creature_card_def's display name but is a DIFFERENT object (the
            # hand card is the sorcery/normal side), so push_to_stack's own
            # identity-based removal below misses it; remove it here by name.
            # That is what makes the OTHER mode uncastable while this copy is on
            # the stack, now that the card physically leaving hand is the sole
            # re-cast guard (_hand_count_available is a plain hand tally).
            hand_card = next((c for c in s.hand if c.name == creature_card_def.name), None)
            if hand_card is not None:
                s.hand.remove(hand_card)
            game.push_to_stack(s, creature_card_def, resolve)
        game.begin_pay_cost(state, cost, on_complete=_after_pay)
    return execute


def build_action_table(decklist, registry, token_card_defs=(), pending_kinds=(),
                        opponent_decklist=None, opponent_token_card_defs=(), extra_choosable_names=()):
    """opponent_decklist/opponent_token_card_defs: the OTHER side's own
    decklist/tokens -- None/() for every 1-player deck (there's no real
    opponent battlefield to reference at all), matching combat_enabled=False
    decks never seeing "Attack: X" become legal. The token/pointer pipeline
    passes None here (cross-player targeting moved to the pointer head, not
    a fixed opponent action table) -- kept for a fixed "Choose opponent's:
    X" table if one is ever wanted again.

    token_card_defs: every token CardDef this deck's own cards can
    create at runtime (Blood, Robot, Warrior, Eldrazi Spawn --
   ), e.g. (game.BLOOD_TOKEN_CARD_DEF,).
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
                speed = _cast_speed(game.CARD_DEFS[name], mode_spec)
                extra_legal = mode_spec.get("extra_legal")
                # A mode may override the card's cast_cost (Kicker: the kicked
                # mode costs more) -- if it does, route through the explicit-
                # cost helpers (_x_cast_*), same ones x_cast_modes uses; else
                # the default card_def.cast_cost path (Winding Way/Utopia Sprawl).
                mode_cost = mode_spec.get("cost")
                if mode_cost is not None:
                    legal = _x_cast_legal(name, mode_cost, extra_legal, speed)
                    ex_fn = _x_precast_choice_execute if mode_spec.get("precast_choice") else _x_cast_execute
                    execute = ex_fn(name, mode_cost, mode_spec["resolve"])
                else:
                    legal = _cast_legal(name, extra_legal, speed)
                    ex_fn = _precast_choice_execute if mode_spec.get("precast_choice") else _cast_execute
                    execute = ex_fn(name, mode_spec["resolve"])
                actions.append((f"Cast {name} (choose {mode_name})", legal, execute))
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
        # Gurmag Angler: Delve -- one "Cast X (delve N)" action per N in
        # 0..delve["max"] (the generic cost), each exiling N graveyard cards
        # to pay {N} of the generic. Same per-value enumeration + plan_payment
        # masking as x_cast_modes.
        delve = card_spec.get("delve")
        if delve is not None:
            delve_speed = _cast_speed(game.CARD_DEFS[name], delve)
            for n in range(delve["max"] + 1):
                actions.append((
                    f"Cast {name} (delve {n})",
                    _delve_legal(name, n, delve_speed),
                    _delve_execute(name, n, delve["resolve"]),
                ))
        # Land Grant: a second, free cast path alongside the normal one.
        alt_cast = card_spec.get("alt_cast")
        if alt_cast is not None:
            actions.append((
                f"Cast {name} (free)",
                _alt_cast_legal(name, alt_cast["extra_legal"], _cast_speed(game.CARD_DEFS[name], alt_cast)),
                _alt_cast_execute(name, alt_cast["resolve"]),
            ))
        # Dread Return: Flashback casts from the graveyard, not hand. Escape
        # (Sleep of the Dead) is the same graveyard-cast machinery with a mana
        # cost + its own additional cost handled inside the resolve -- shares
        # _flashback_legal/_execute, only the action label differs.
        for gy_cast_key, gy_cast_label in (("flashback", "Flashback"), ("escape", "Escape")):
            gy_cast = card_spec.get(gy_cast_key)
            if gy_cast is not None:
                gc_cost = gy_cast.get("cost")  # mana cost dict, if the cost includes mana (Deep Analysis, Sleep of the Dead)
                actions.append((
                    f"{gy_cast_label} {name}",
                    _flashback_legal(name, gy_cast["legal"], _cast_speed(game.CARD_DEFS[name], gy_cast), gc_cost),
                    _flashback_execute(name, gy_cast["resolve"], gc_cost),
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
        # Boulderbranch Golem: Prototype -- a second cast option for the same
        # hand card, its own cheaper cost, producing a DIFFERENT CardDef (the
        # smaller 3/3 with its own ETB). Structurally identical to Omen ("cast
        # this hand card for an alternate cost as a different creature"), so it
        # reuses the same _omen_cast_legal/_omen_cast_execute helpers -- only
        # the resolve differs (no library shuffle; the prototype creature just
        # enters). Real reminder text: "You may cast this spell with different
        # mana cost, color, and size. It keeps its abilities and types."
        prototype = card_spec.get("prototype")
        if prototype is not None:
            proto_speed = _cast_speed(prototype["card_def"], prototype)
            actions.append((
                f"Cast {name} (prototype)",
                _omen_cast_legal(name, prototype["cost"], proto_speed),
                _omen_cast_execute(prototype["card_def"], prototype["cost"], prototype["resolve"]),
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

    # "Discard this card from hand: <do something>" cycling-family actions.
    # Both keys share the identical hand-zone/cost-key/resolve plumbing
    # (_forestcycle_legal/_execute) -- they differ only in the action label:
    #   "forestcycle" -- basic-land-to-hand search (Generous Ent, Ash Barrens)
    #   "cycle"       -- plain Cycling (discard, draw) and typed cycling like
    #                    Islandcycling (Lorien Revealed) / Twisted Landscape
    for name in distinct_names:
        for spec_key, label in (("forestcycle", "Forestcycle"), ("cycle", "Cycle")):
            cyc_spec = registry.get(game.CARD_DEFS[name].effect_id, {}).get(spec_key)
            if cyc_spec is not None:
                actions.append((
                    f"{label} {name}",
                    _forestcycle_legal(name, cyc_spec["cost_key"]),
                    _forestcycle_execute(name, cyc_spec["cost_key"], cyc_spec["resolve"]),
                ))

    # Impulse: "you may play the exiled cards" (Reckless Impulse / Experimental
    # Synthesizer / Clockwork Percussionist). Only emitted for a deck that can
    # actually impulse (its impulse-source card declares pending_kinds
    # {"impulse"}), so decks without one never carry these mostly-illegal
    # actions. One action per deck card name: a land play, or a cast per the
    # card's own cast/cast_modes spec (paying its NORMAL cost -- impulse, unlike
    # Plot, is not free; x_cast_modes cards, none in the impulse decks, aren't
    # offered from impulse).
    if "impulse" in pending_kinds:
        for name in distinct_names:
            card_def = game.CARD_DEFS[name]
            if card_def.card_type == game.CardType.LAND:
                actions.append((f"Play from exile: {name}", _play_impulse_land_legal(name), _play_impulse_land_execute(name)))
                continue
            spec = registry.get(card_def.effect_id, {})
            cast_spec = spec.get("cast")
            if cast_spec is not None:
                speed = _cast_speed(card_def, cast_spec)
                actions.append((
                    f"Play from exile: {name}",
                    _play_impulse_cast_legal(name, card_def.cast_cost, cast_spec.get("extra_legal"), speed),
                    _play_impulse_cast_execute(name, card_def.cast_cost, cast_spec["resolve"], cast_spec.get("precast_choice", False)),
                ))
            for mode_name, mode_spec in (spec.get("cast_modes") or {}).items():
                mode_cost = mode_spec.get("cost", card_def.cast_cost)
                speed = _cast_speed(card_def, mode_spec)
                actions.append((
                    f"Play from exile: {name} ({mode_name})",
                    _play_impulse_cast_legal(name, mode_cost, mode_spec.get("extra_legal"), speed),
                    _play_impulse_cast_execute(name, mode_cost, mode_spec["resolve"], mode_spec.get("precast_choice", False)),
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
    # as the whole league's card universe (rl.pool.build_pool), not a
    # specific opponent's deck: bounded, fixed per trained model, and still
    # runtime-masked to only-legal-when-actually-in-the-targeted-graveyard.
    choosable_names = sorted(set(distinct_names) | {cd.name for cd in token_card_defs})
    # extra_choosable_names goes into the "Choose: X" loop ONLY, never the
    # shared choosable_names set below -- that set also drives "Choose
    # target: X (slot k)" and "Attack: X (slot k)", which are strictly
    # THIS side's OWN battlefield permanents (an opponent's card is never a
    # legal choose_permanent/attack target of mine) and index a
    # card_type_by_name map built from this deck alone. A foreign name only
    # ever belongs to the by-name "Choose: X" resolution (choose_graveyard_
    # card over an opponent's graveyard), nothing permanent-scoped.
    choose_by_name = sorted(set(choosable_names) | set(extra_choosable_names))
    for name in choose_by_name:
        actions.append((f"Choose: {name}", _choose_name_legal(name), _choose_name_execute(name)))

    # "Attack: X (slot k)" -- one per (creature name, slot) pair
    #, legal only during
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
    # decision about this player's OWN creatures,
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
            # "Unassign Blocker: X (slot j)" -- take a committed blocker back
            # out (menace's 0-or-2+ rule needs this to never be a softlock, see
            # _unassign_blocker_legal). Same (name, slot) own-creature domain.
            actions.append((
                f"Unassign Blocker: {name} (slot {slot})",
                _unassign_blocker_legal(name, slot),
                _unassign_blocker_execute(name, slot),
            ))
    actions.append(("Done blocking", _done_blocking_legal, _done_blocking_execute))
    # Trample's "assign a combat-damage point to the defending player" half
    # of a gang-blocking damage assignment (the blocker half is the pointer
    # scheme). One fixed action, runtime-gated to a trampling attacker mid-
    # assign_combat_damage -- masked illegal otherwise.
    actions.append((
        "Assign combat damage to opponent", _assign_damage_to_opponent_legal, _assign_damage_to_opponent_execute,
    ))

    # "Choose opponent's: X (slot k)" -- the general cross-player
    # targeting primitive, one per (opponent
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
    if "ponder" in pending_kinds:
        actions.append(("Shuffle (Ponder)", _ponder_shuffle_legal, _ponder_shuffle_execute))
    if "pay_unless" in pending_kinds:  # Spell Pierce / Ward "unless controller pays {N}"
        actions.append(("Pay (unless)", _pay_unless_pay_legal, _pay_unless_pay_execute))
        actions.append(("Don't pay (unless)", _pay_unless_decline_legal, _pay_unless_decline_execute))
    if "tuck_position" in pending_kinds:  # Deem Inferior: owner picks 2nd-from-top or bottom
        actions.append(("Tuck: 2nd from top", _tuck_position_legal, lambda state: game.execute_tuck_position(state, "top2")))
        actions.append(("Tuck: bottom", _tuck_position_legal, lambda state: game.execute_tuck_position(state, "bottom")))
    if "may_transform" in pending_kinds:  # Delver of Secrets: revealed an instant/sorcery, may flip
        actions.append(("Transform", _may_transform_legal, lambda state: game.execute_may_transform(state, True)))
        actions.append(("Don't transform", _may_transform_legal, lambda state: game.execute_may_transform(state, False)))
    if "may_copy" in pending_kinds:  # Chain Lightning: after paying {R}{R}, may copy the spell
        actions.append(("Copy spell", _may_copy_legal, lambda state: game.execute_may_copy(state, True)))
        actions.append(("Don't copy spell", _may_copy_legal, lambda state: game.execute_may_copy(state, False)))
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
    if "choose_graveyard_card" in pending_kinds:
        # Only ever legal for an OPTIONAL choose_graveyard_card (Masked
        # Vandal's "you may exile a creature from your graveyard"); the
        # legal_fn itself gates on pending["optional"], so it stays present-
        # but-permanently-illegal for decks whose only graveyard picks are
        # mandatory (Dread Return, Relic), same footing as "Decline (search)".
        actions.append(("Decline (graveyard)", _decline_graveyard_card_legal, _decline_graveyard_card_execute))
    actions.append(("Abandon payment", _abandon_payment_legal, _abandon_payment_execute))  # pay_cost is baseline, always present
    # NOTE: the pregame mulligan actions ("Keep hand" / "Mulligan") and the
    # "mulligan_bottom" branch of the generic "Choose: X" action were REMOVED from
    # this table (harness refactor Phase 4). The per-deck MulliganNet (rl.mulligan)
    # now OWNS every pregame decision -- rl.agent.SeatAgent intercepts the pregame
    # pending kinds before the main net's forward -- so the main policy's action
    # space contains ZERO pregame actions and a game can never fall back to a
    # fixed-table mulligan. The _mulligan_*_legal/_execute helpers are retained
    # (still exported) but no longer wired into any table.
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
    if "choose_any_target" in pending_kinds:
        # The player half of an "any target" choice (Lightning Bolt etc.) --
        # two fixed actions, same shape/reasoning as the choose_target_player
        # pair above. The creature half (either battlefield) rides the
        # identity pointer scheme (rl.action_bridge), not fixed actions.
        actions.append(("Target any: yourself", _target_any_self_legal, _target_any_self_execute))
        actions.append(("Target any: opponent", _target_any_opponent_legal, _target_any_opponent_execute))
        actions.append(("Choose no target", _target_any_decline_legal, _target_any_decline_execute))  # "up to one" decline
    if "choose_room" in pending_kinds:  # Undercity venture: which next room to enter (a ≤2-way branch)
        for room in game.ROOM_NAMES:
            actions.append((f"Enter room: {room}", _choose_room_legal(room), _choose_room_execute(room)))
    if "choose_mana_color" in pending_kinds:  # Chromatic Star: "add one mana of any color"
        for color in game.COLORS:
            actions.append((f"Add mana: {color}", _choose_mana_color_legal(color), _choose_mana_color_execute(color)))

    return tuple(actions)


_battlefield_lookup_cache = None  # (state, {(name, slot): Permanent}) -- valid only for the duration of one legal_action_mask sweep, same lifecycle as _tap_cost_options_cache below


def _cached_battlefield_lookup(state):
    """Sweep-scoped {(name, slot): Permanent} lookup for state.battlefield --
    same "profiled, not guessed" caching pattern as _cached_tap_cost_options
    just below: _attack_legal/
    _assign_blocker_legal each independently scanned the WHOLE battlefield
    with any(...) to find one specific (name, slot), once per action-table
    entry -- for a deck with many creature copies (boggles' Auras/tokens)
    that's O(action_table_size x battlefield_size) repeated work every
    sweep (profiled: 2 closures alone accounted for ~3.4M calls across a
    single 8192-step training burst). Building this dict once per sweep
    turns each of those checks into an O(1) lookup. Safe for the same
    reason _cached_tap_cost_options is: a legal_action_mask sweep only ever
    calls legal_fns, never an execute_* function, so state can't change
    mid-sweep. (name, slot) is a safe dict key here because it is unique
    per side -- state.battlefield is always ONE side's own,
    active-relative zone (see this module's other active-relative
    docstrings), never two players' permanents mixed in one sweep."""
    global _battlefield_lookup_cache
    if _battlefield_lookup_cache is None or _battlefield_lookup_cache[0] is not state:
        _battlefield_lookup_cache = (state, {(p.card_def.name, p.slot): p for p in state.battlefield})
    return _battlefield_lookup_cache[1]


_tap_cost_options_cache = None  # (state, result) -- valid only for the duration of one legal_action_mask sweep, see _cached_tap_cost_options


def _cached_tap_cost_options(state):
    """Memoizes game.tap_cost_options(state) for the exact duration of one
    legal_action_mask sweep. _choose_name_legal/_choose_name_color_legal
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
    """Stateless -- takes the action table explicitly, so any caller (the
    token pipeline's own _seat_step, a direct game-loop driver, ...)
    can use it. `actions` is any table built by build_action_table -- every
    deck's own table, none privileged as a default (a caller with its own
    decklist always has its own table to pass).

    Category-gating (profiled, not guessed: this table can run ~300 entries
    long, and every single one of those closures gets called on every
    sweep regardless of relevance
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


__all__ = [
    '_cast_speed',
    '_GATE_NO_PENDING',
    '_land_drop_legal',
    '_land_drop_execute',
    '_hand_count_available',
    '_effective_cast_cost',
    '_cast_legal',
    '_cast_execute',
    '_precast_choice_execute',
    '_x_cast_legal',
    '_x_cast_execute',
    '_x_precast_choice_execute',
    '_delve_reduced_cost',
    '_delve_legal',
    '_delve_execute',
    '_tuck_position_legal',
    '_activate_legal',
    '_activate_execute',
    '_forestcycle_legal',
    '_forestcycle_execute',
    '_graveyard_ability_legal',
    '_graveyard_ability_execute',
    '_pass_legal',
    '_pass_execute',
    '_choose_name_options',
    '_choose_name_legal',
    '_choose_name_execute',
    '_choose_name_color_options',
    '_choose_name_color_legal',
    '_choose_name_color_execute',
    '_attack_legal',
    '_attack_execute',
    '_choose_permanent_legal',
    '_choose_permanent_execute',
    '_choose_opponent_permanent_legal',
    '_choose_opponent_permanent_execute',
    '_assign_blocker_legal',
    '_assign_blocker_execute',
    '_done_blocking_legal',
    '_done_blocking_execute',
    '_unassign_blocker_legal',
    '_unassign_blocker_execute',
    '_assign_damage_to_opponent_legal',
    '_assign_damage_to_opponent_execute',
    '_pool_spend_legal',
    '_pool_spend_execute',
    '_keep_dispose_legal',
    '_keep_execute',
    '_dispose_execute',
    '_decline_legal',
    '_decline_execute',
    '_decline_malevolent_rumble_legal',
    '_decline_malevolent_rumble_execute',
    '_abandon_payment_legal',
    '_abandon_payment_execute',
    '_ponder_shuffle_legal',
    '_ponder_shuffle_execute',
    '_pay_unless_pay_legal',
    '_pay_unless_pay_execute',
    '_pay_unless_decline_legal',
    '_pay_unless_decline_execute',
    '_may_transform_legal',
    '_may_copy_legal',
    '_choose_room_legal',
    '_choose_room_execute',
    '_choose_mana_color_legal',
    '_choose_mana_color_execute',
    '_select_to_hand_keep_legal',
    '_select_to_hand_bottom_legal',
    '_select_to_hand_keep_execute',
    '_select_to_hand_bottom_execute',
    '_decline_search_legal',
    '_decline_search_execute',
    '_decline_graveyard_card_legal',
    '_decline_graveyard_card_execute',
    '_decline_discard_legal',
    '_decline_discard_execute',
    '_target_self_legal',
    '_target_self_execute',
    '_target_opponent_legal',
    '_target_opponent_execute',
    '_target_any_self_legal',
    '_target_any_self_execute',
    '_target_any_opponent_legal',
    '_target_any_opponent_execute',
    '_target_any_decline_legal',
    '_target_any_decline_execute',
    '_discard_or_sacrifice_sacrifice_legal',
    '_discard_or_sacrifice_sacrifice_execute',
    '_decline_discard_or_sacrifice_legal',
    '_decline_discard_or_sacrifice_execute',
    '_madness_cast_legal',
    '_madness_cast_execute',
    '_madness_decline_legal',
    '_madness_decline_execute',
    '_activate_no_cost_legal',
    '_activate_no_cost_execute',
    '_alt_cast_legal',
    '_alt_cast_execute',
    '_flashback_legal',
    '_flashback_execute',
    '_impulse_entry',
    '_play_impulse_land_legal',
    '_play_impulse_land_execute',
    '_play_impulse_cast_legal',
    '_play_impulse_cast_execute',
    '_plot_legal',
    '_plot_execute',
    '_cast_from_exile_legal',
    '_cast_from_exile_execute',
    '_omen_cast_legal',
    '_omen_cast_execute',
    'build_action_table',
    '_battlefield_lookup_cache',
    '_cached_battlefield_lookup',
    '_tap_cost_options_cache',
    '_cached_tap_cost_options',
    'legal_action_mask',
]
