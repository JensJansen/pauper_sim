"""Gymnasium environment adapter wrapping game.py's simulator.

Not one of the 4 independent pieces (see DRL_PLAN.md) -- this is assembly
logic that combines a simulator (game.py, imported and never modified)
with an injected reward function (rewards.py's contract) into a
Gym-compatible interface a DRL model can train against.
"""

import random

import numpy as np
import gymnasium
from gymnasium import spaces

import game

# ---------------------------------------------------------------------------
# D2.1 -- Card indexing and observation builder (MULTI_DECK_PLAN.md Phase
# M4f: build_observation takes a decklist explicitly, sized to its
# distinct-card count, instead of a hardcoded 90-dim vector.
# ---------------------------------------------------------------------------

CARD_NAMES = sorted({name for name, *_ in game.TRON_DECKLIST})
CARD_COPIES = {name: qty for name, qty, *_ in game.TRON_DECKLIST}

# Every kind string game.py's pending-resolution mechanism can produce,
# plus "none" for "nothing pending" -- part of the engine's own vocabulary
# (see game.py's begin_resolution), not decklist-specific.
PENDING_KINDS = (
    "none", "pay_cost", "search_fetch", "choose_permanent", "ancient_stirrings", "scry", "surveil",
    # spy_combo deck additions:
    "select_to_hand", "choose_graveyard_card", "sacrifice_creatures",
)

OBSERVATION_DIM = len(CARD_NAMES) * 4 + 2 + len(PENDING_KINDS)


def observation_dim_for(decklist):
    """Same formula OBSERVATION_DIM uses for Tron, generalized to any
    decklist -- shared by TronEnv.__init__ and harness.py's load() so a
    second deck's dimension is computed identically in both places."""
    return len({name for name, *_rest in decklist}) * 4 + 2 + len(PENDING_KINDS)


def build_observation(state, decklist, horizon):
    card_names = sorted({name for name, *_rest in decklist})
    card_copies = {name: qty for name, qty, *_rest in decklist}
    dim = len(card_names) * 4 + 2 + len(PENDING_KINDS)
    obs = np.zeros(dim, dtype=np.float32)

    hand_counts = {name: 0 for name in card_names}
    for card_def in state.hand:
        hand_counts[card_def.name] += 1

    bf_untapped = {name: 0 for name in card_names}
    bf_tapped = {name: 0 for name in card_names}
    for p in state.battlefield:
        if p.tapped:
            bf_tapped[p.card_def.name] += 1
        else:
            bf_untapped[p.card_def.name] += 1

    graveyard_counts = {name: 0 for name in card_names}
    for card_def in state.graveyard:
        graveyard_counts[card_def.name] += 1

    i = 0
    for name in card_names:
        obs[i] = hand_counts[name] / card_copies[name]
        i += 1
    for name in card_names:
        obs[i] = bf_untapped[name] / card_copies[name]
        i += 1
        obs[i] = bf_tapped[name] / card_copies[name]
        i += 1
    for name in card_names:
        remaining = card_copies[name] - hand_counts[name] - bf_untapped[name] - bf_tapped[name] - graveyard_counts[name]
        obs[i] = remaining / card_copies[name]
        i += 1
    obs[i] = state.turn_number / horizon
    i += 1
    obs[i] = 1.0 if state.lands_played_this_turn > 0 else 0.0
    i += 1

    # Which kind of pending resolution (if any) is active right now -- the
    # only signal in the observation itself that a decision like "which
    # tap source" or "keep or dispose" is underway; the action mask alone
    # tells the model *what's* legal but not *why* (MaskablePPO's network
    # never sees the mask as an input feature, only the observation).
    pending_kind = state.pending_resolution["kind"] if state.pending_resolution is not None else "none"
    for kind in PENDING_KINDS:
        obs[i] = 1.0 if kind == pending_kind else 0.0
        i += 1

    return obs


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
#      source, search_fetch, choose_permanent, ancient_stirrings, and
#      scry/surveil's ordering phase), dispatched by pending_resolution["kind"]
#   G. Choose: <name> as <color>    -- flexible/filter mana sources during
#      a pay_cost resolution specifically (the only kind needing a color)
#   H. Keep / Dispose (scry/surveil)
#   I. Decline (Ancient Stirrings)
#   J. Abandon payment -- cancels a pending pay_cost resolution outright,
#      untapping everything tapped so far. Without this, tapping a
#      flexible/filter source for the wrong color could strand a game
#      with an unpayable remaining cost and zero legal actions -- see
#      game.abandon_pay_cost's docstring.
#
# spy_combo deck additions: B also covers Winding Way's modal cast (2
# actions, one per mode), Land Grant's free alt-cost, and Dread Return's
# Flashback (cast from the graveyard); C also covers non-mana activated
# abilities (Quirion Ranger); F/H also cover select_to_hand's own
# Keep/Bottom pair and its ordering phase (Lead the Stampede) and an
# optional search's Decline (Gatecreeper Vine) alongside Ancient
# Stirrings'.
# ---------------------------------------------------------------------------

def _land_drop_legal(name):
    def legal(state):
        return (
            state.pending_resolution is None
            and state.lands_played_this_turn == 0
            and any(c.name == name for c in state.hand)
        )
    return legal


def _land_drop_execute(name):
    def execute(state):
        game.play_land_from_hand(state, game.CARD_DEFS[name])
    return execute


def _cast_legal(name, extra_legal):
    def legal(state):
        if state.pending_resolution is not None:
            return False
        if not any(c.name == name for c in state.hand):
            return False
        card_def = game.CARD_DEFS[name]
        if game.plan_payment(state, card_def.cast_cost) is None:
            return False
        return extra_legal is None or extra_legal(state)
    return legal


def _cast_execute(name, resolve):
    def execute(state):
        card_def = game.CARD_DEFS[name]
        game.begin_pay_cost(state, card_def.cast_cost, on_complete=lambda s: resolve(s, card_def))
    return execute


def _activate_legal(name, cost_key):
    def legal(state):
        if state.pending_resolution is not None:
            return False
        p = next((p for p in state.battlefield if p.card_def.name == name and not p.tapped), None)
        return p is not None and game.plan_payment(state, p.card_def.extra[cost_key]) is not None
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
    return legal


def _forestcycle_execute(name, cost_key, resolve):
    def execute(state):
        card_def = game.CARD_DEFS[name]
        game.begin_pay_cost(state, card_def.extra[cost_key], on_complete=lambda s: resolve(s, card_def))
    return execute


def _pass_legal(state):
    return state.pending_resolution is None


def _pass_execute(state):
    pass  # handled by TronEnv.step() itself, not via this table


def _choose_name_options(state):
    """Plain (uncolored) 'Choose: X' names currently legal, given whatever
    kind of pending resolution -- if any -- is active."""
    pending = state.pending_resolution
    if pending is None:
        return []
    kind = pending["kind"]
    if kind == "pay_cost":
        return [n for n, c, f in game.tap_cost_options(state) if c is None and not f]
    if kind == "search_fetch":
        return game.search_fetch_options(state)
    if kind == "choose_permanent":
        return game.choose_permanent_options(state)
    if kind == "choose_graveyard_card":
        return game.choose_graveyard_card_options(state)
    if kind == "sacrifice_creatures":
        return game.sacrifice_creatures_options(state)
    if kind == "ancient_stirrings":
        return [n for n in game.ancient_stirrings_options(state) if n != "decline"]
    if kind in ("scry", "surveil") and pending["ordered"] is not None:
        return game.scry_surveil_options(state)
    if kind == "select_to_hand" and pending["ordered"] is not None:
        return game.select_to_hand_options(state)  # ordering phase only -- "keep"/"bottom" are their own actions
    return []


def _choose_name_legal(name):
    def legal(state):
        return name in _choose_name_options(state)
    return legal


def _choose_name_execute(name):
    def execute(state):
        kind = state.pending_resolution["kind"]
        if kind == "pay_cost":
            game.execute_tap_cost_option(state, name, None, False)
        elif kind == "search_fetch":
            game.execute_search_fetch_option(state, name)
        elif kind == "choose_permanent":
            game.execute_choose_permanent_option(state, name)
        elif kind == "choose_graveyard_card":
            game.execute_choose_graveyard_card_option(state, name)
        elif kind == "sacrifice_creatures":
            game.execute_sacrifice_creatures_option(state, name)
        elif kind == "ancient_stirrings":
            game.execute_ancient_stirrings_option(state, name)
        elif kind == "select_to_hand":
            game.execute_select_to_hand_option(state, name)  # ordering phase only
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
    return [(n, c) for n, c, _f in game.tap_cost_options(state) if c is not None]


def _choose_name_color_legal(name, color):
    def legal(state):
        return (name, color) in _choose_name_color_options(state)
    return legal


def _choose_name_color_execute(name, color):
    def execute(state):
        is_filter = next(f for n, c, f in game.tap_cost_options(state) if n == name and c == color)
        game.execute_tap_cost_option(state, name, color, is_filter)
    return execute


def _keep_dispose_legal(state):
    pending = state.pending_resolution
    return pending is not None and pending["kind"] in ("scry", "surveil") and bool(pending["remaining"])


def _keep_execute(state):
    game.execute_scry_surveil_option(state, "keep")


def _dispose_execute(state):
    game.execute_scry_surveil_option(state, "dispose")


def _decline_legal(state):
    pending = state.pending_resolution
    return pending is not None and pending["kind"] == "ancient_stirrings"


def _decline_execute(state):
    game.execute_ancient_stirrings_option(state, "decline")


def _abandon_payment_legal(state):
    pending = state.pending_resolution
    return pending is not None and pending["kind"] == "pay_cost"


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


def _select_to_hand_bottom_legal(state):
    pending = state.pending_resolution
    return pending is not None and pending["kind"] == "select_to_hand" and bool(pending["remaining"])


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


def _decline_search_execute(state):
    game.execute_search_fetch_decline(state)


def _activate_no_cost_legal(name, ability_legal):
    """Non-mana activated-ability cost (Quirion Ranger's Forest bounce):
    no {T}-of-self assumption, unlike _activate_legal -- the ability's own
    legal(state, permanent) captures its whole cost precondition."""
    def legal(state):
        if state.pending_resolution is not None:
            return False
        p = next((p for p in state.battlefield if p.card_def.name == name), None)
        return p is not None and ability_legal(state, p)
    return legal


def _activate_no_cost_execute(name, resolve):
    def execute(state):
        p = next(p for p in state.battlefield if p.card_def.name == name)
        resolve(state, p)
    return execute


def _alt_cast_legal(name, extra_legal):
    """Land Grant's free alt-cost: no mana payment at all, just the
    card's own extra_legal predicate (0 lands in hand)."""
    def legal(state):
        if state.pending_resolution is not None:
            return False
        if not any(c.name == name for c in state.hand):
            return False
        return extra_legal(state)
    return legal


def _alt_cast_execute(name, resolve):
    def execute(state):
        resolve(state, game.CARD_DEFS[name])
    return execute


def _flashback_legal(name, ability_legal):
    """Dread Return's Flashback: cast from the graveyard, not hand."""
    def legal(state):
        if state.pending_resolution is not None:
            return False
        if not any(c.name == name for c in state.graveyard):
            return False
        return ability_legal(state)
    return legal


def _flashback_execute(name, resolve):
    def execute(state):
        resolve(state, game.CARD_DEFS[name])
    return execute


def build_action_table(decklist, registry):
    distinct_names = sorted({name for name, *_rest in decklist})
    land_names = sorted({
        name for name, _qty, card_type, _cost, _effect, _extra in decklist
        if card_type == game.CardType.LAND
    })

    actions = []

    for name in land_names:
        actions.append((f"Play land: {name}", _land_drop_legal(name), _land_drop_execute(name)))

    for name in distinct_names:
        card_spec = registry.get(game.CARD_DEFS[name].effect_id, {})
        cast_spec = card_spec.get("cast")
        if cast_spec is not None:
            actions.append((
                f"Cast {name}",
                _cast_legal(name, cast_spec.get("extra_legal")),
                _cast_execute(name, cast_spec["resolve"]),
            ))
        # Winding Way: a modal cast (choose creature or land) instead of a
        # single "cast" entry -- one action per mode.
        cast_modes = card_spec.get("cast_modes")
        if cast_modes is not None:
            for mode_name, mode_spec in cast_modes.items():
                actions.append((
                    f"Cast {name} (choose {mode_name})",
                    _cast_legal(name, mode_spec.get("extra_legal")),
                    _cast_execute(name, mode_spec["resolve"]),
                ))
        # Land Grant: a second, free cast path alongside the normal one.
        alt_cast = card_spec.get("alt_cast")
        if alt_cast is not None:
            actions.append((
                f"Cast {name} (free)",
                _alt_cast_legal(name, alt_cast["extra_legal"]),
                _alt_cast_execute(name, alt_cast["resolve"]),
            ))
        # Dread Return: Flashback casts from the graveyard, not hand.
        flashback = card_spec.get("flashback")
        if flashback is not None:
            actions.append((
                f"Flashback {name}",
                _flashback_legal(name, flashback["legal"]),
                _flashback_execute(name, flashback["resolve"]),
            ))

    for name in distinct_names:
        abilities = registry.get(game.CARD_DEFS[name].effect_id, {}).get("activated_abilities", {})
        for ability_name, spec in abilities.items():
            if "cost_key" in spec:
                actions.append((
                    f"Activate {name} ({ability_name})",
                    _activate_legal(name, spec["cost_key"]),
                    _activate_execute(name, spec["cost_key"], spec["resolve"]),
                ))
            else:
                # Non-mana cost (Quirion Ranger: return a Forest to hand).
                actions.append((
                    f"Activate {name} ({ability_name})",
                    _activate_no_cost_legal(name, spec["legal"]),
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

    actions.append(("Pass", _pass_legal, _pass_execute))

    for name in distinct_names:
        actions.append((f"Choose: {name}", _choose_name_legal(name), _choose_name_execute(name)))

    for name in distinct_names:
        spec = registry.get(game.CARD_DEFS[name].effect_id, {})
        colors = set()
        mana = spec.get("mana")
        if mana is not None and mana[0] == "flexible":
            colors |= mana[1]
        filter_mana = spec.get("filter_mana")
        if filter_mana is not None:
            colors |= filter_mana["colors"]
        for color in sorted(colors):
            actions.append((
                f"Choose: {name} as {color}",
                _choose_name_color_legal(name, color),
                _choose_name_color_execute(name, color),
            ))

    actions.append(("Keep (scry/surveil)", _keep_dispose_legal, _keep_execute))
    actions.append(("Dispose (scry/surveil)", _keep_dispose_legal, _dispose_execute))
    actions.append(("Decline (Ancient Stirrings)", _decline_legal, _decline_execute))
    actions.append(("Keep (select to hand)", _select_to_hand_keep_legal, _select_to_hand_keep_execute))
    actions.append(("Bottom (select to hand)", _select_to_hand_bottom_legal, _select_to_hand_bottom_execute))
    actions.append(("Decline (search)", _decline_search_legal, _decline_search_execute))
    actions.append(("Abandon payment", _abandon_payment_legal, _abandon_payment_execute))

    return tuple(actions)


ACTIONS = build_action_table(game.TRON_DECKLIST, game.EFFECT_REGISTRY)
PASS_ACTION = next(i for i, (name, _legal, _execute) in enumerate(ACTIONS) if name == "Pass")


def legal_action_mask(state, actions=ACTIONS):
    """Stateless: usable both by TronEnv.action_masks() and by
    harness.evaluate(), which plays games directly through game.run_game,
    not through env.step (see DRL_CHECKLIST.md's D6 implementation note).
    `actions` defaults to the Tron table but accepts any table built by
    build_action_table, so a second deck's env can reuse this function."""
    return np.array([legal_fn(state) for _, legal_fn, _ in actions], dtype=bool)


# ---------------------------------------------------------------------------
# D2.3 / D2.4 -- TronEnv
# ---------------------------------------------------------------------------

def _start_turn(state):
    """Exactly game.run_turn's preamble (composing its public primitives in
    the same order), stopping short of running a main-phase loop -- that
    loop is what TronEnv.step() replaces, one action at a time."""
    state.turn_number += 1
    state.lands_played_this_turn = 0
    game.untap_step(state)
    game.draw_step(state)


class TronEnv(gymnasium.Env):
    # Deck-parameterized (MULTI_DECK_PLAN.md Phase M4/M7): decklist/
    # terminated_fn default to Tron's so every existing call site keeps
    # working unchanged, but a second deck's env is just a different
    # decklist/terminated_fn passed in here -- its own action table and
    # observation dim are built fresh per instance, not read from the
    # module-level Tron-specific ACTIONS/OBSERVATION_DIM globals.
    def __init__(self, reward_fn, decklist=game.TRON_DECKLIST, terminated_fn=game.tron_terminated,
                 horizon=6, on_the_play=True, seed=None):
        super().__init__()
        self.reward_fn = reward_fn
        self.decklist = decklist
        self.terminated_fn = terminated_fn
        self.horizon = horizon
        self.on_the_play = on_the_play
        self._rng = random.Random(seed)
        self.state = None
        self.actions = build_action_table(decklist, game.EFFECT_REGISTRY)
        self.pass_action = next(i for i, (name, _legal, _execute) in enumerate(self.actions) if name == "Pass")
        self.observation_dim = observation_dim_for(decklist)
        self.observation_space = spaces.Box(low=0.0, high=1.0, shape=(self.observation_dim,), dtype=np.float32)
        self.action_space = spaces.Discrete(len(self.actions))

    def action_masks(self):
        if self.state is None:
            raise RuntimeError("action_masks() called before reset()")
        return legal_action_mask(self.state, self.actions)

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        if seed is not None:
            self._rng = random.Random(seed)
        self.state = game.new_game_state(self.decklist, self.terminated_fn, self.on_the_play, self._rng)
        _start_turn(self.state)
        return build_observation(self.state, self.decklist, self.horizon), {}

    def step(self, action):
        mask = legal_action_mask(self.state, self.actions)
        if not mask[action]:
            # Illegal action -> substitute the first currently-legal one
            # (works for any SB3 algorithm, maskable or not). MUST NOT
            # assume PASS_ACTION specifically is always a safe substitute
            # (MULTI_DECK_PLAN.md Phase M4e): Pass is illegal whenever a
            # resolution is pending, and blindly "passing" in that state
            # would abandon the resolution mid-flight and desync
            # state.pending_resolution from the turn loop. legal_indices
            # is never empty: with no resolution pending, Pass itself is
            # always legal; with one pending, every kind guarantees at
            # least one option (pay_cost by construction, search_fetch/
            # choose_permanent's empty case auto-fizzles instead of
            # leaving a stuck resolution, ancient_stirrings always offers
            # "decline", scry/surveil always offers keep/dispose or a
            # nonempty ordering set).
            legal_indices = np.flatnonzero(mask)
            action = int(legal_indices[0])

        game_over = False
        if action == self.pass_action:
            if self.state.turn_number < self.horizon:
                _start_turn(self.state)
            else:
                game_over = True  # just passed during the final turn -- no more turns left
        else:
            _, _, execute_fn = self.actions[action]
            execute_fn(self.state)

        done = self.state.turn_won is not None or game_over or self.state.decked_out
        reward = self.reward_fn(self.state, done, self.horizon)
        obs = build_observation(self.state, self.decklist, self.horizon)
        return obs, reward, done, False, {}


def _phase_d2_sanity_check():
    import rewards

    # D2.1: observation shape/range.
    rng = random.Random(0)
    state = game.new_game_state(game.TRON_DECKLIST, game.tron_terminated, on_the_play=True, rng=rng)
    obs = build_observation(state, game.TRON_DECKLIST, horizon=6)
    assert obs.shape == (OBSERVATION_DIM,), obs.shape
    assert OBSERVATION_DIM == len(CARD_NAMES) * 4 + 2 + len(PENDING_KINDS), OBSERVATION_DIM
    assert np.all(obs >= 0.0) and np.all(obs <= 1.0), obs

    # D2.2: legality on a hand-built state (only one Tron land in hand, no
    # mana in play at all -- only that land drop and Pass should be legal).
    # Checked by name, not index: MULTI_DECK_PLAN.md Phase M4e's generated
    # table doesn't guarantee the same ordering the old hand-typed one did.
    state2 = game.GameState(on_the_play=True, rng=random.Random(0))
    state2.turn_number = 1
    state2.hand = [game.CARD_DEFS["Urza's Mine"], game.CARD_DEFS["Rooftop Percher"]]
    state2.library = [game.CARD_DEFS["Rooftop Percher"]] * 53
    mask = legal_action_mask(state2)
    legal_names = sorted(ACTIONS[i][0] for i, ok in enumerate(mask) if ok)
    assert legal_names == ["Pass", "Play land: Urza's Mine"], legal_names

    # D2.4: an all-Pass episode advances turns exactly like game.run_game
    # with an always-pass policy (cross-checked against game._phase4_sanity_check).
    env = TronEnv(rewards.assembled_with_resource_quality, horizon=6, on_the_play=True, seed=0)
    obs, info = env.reset()
    done = False
    steps = 0
    while not done:
        obs, reward, done, truncated, info = env.step(PASS_ACTION)
        steps += 1
        assert steps < 1000, "runaway episode"
    assert env.state.turn_number == 6, env.state.turn_number
    assert env.state.turn_won is None
    assert len(env.state.hand) == 7 + 5, len(env.state.hand)
    assert reward == 0.0, reward

    # D2.4: 50 uniform-random-legal-action episodes run to completion, and
    # never get stuck with a pending resolution and zero legal actions
    # (MULTI_DECK_PLAN.md Phase M4e's illegal-action fallback and the
    # search_fetch/choose_permanent empty-options safety net both depend
    # on this invariant -- a fuller, thousands-of-games version of this
    # same check is Phase M4g's job).
    rng2 = random.Random(1)
    for _ in range(50):
        env2 = TronEnv(rewards.assembled_with_resource_quality, horizon=6, on_the_play=True, seed=rng2.random())
        obs, info = env2.reset()
        done = False
        steps = 0
        while not done:
            mask = env2.action_masks()
            legal = [i for i, ok in enumerate(mask) if ok]
            assert legal, "no legal action available -- stuck state"
            action = rng2.choice(legal)
            obs, reward, done, truncated, info = env2.step(action)
            steps += 1
            assert steps < 1000, "runaway episode"
        assert env2.state.pending_resolution is None, "episode ended with a resolution still pending"
        assert reward == 0.0 or reward > 0.0


def _find_action(name):
    return next(i for i, (n, _legal, _execute) in enumerate(ACTIONS) if n == name)


def _phase_m4g_sanity_check():
    """MULTI_DECK_PLAN.md Phase M4g: large-scale random-rollout verification
    of the whole generic action table (not just game.py's own primitives,
    which Phases M4a-M4d already covered in detail) plus one full
    end-to-end walk of a multi-step action through the real ACTIONS table,
    to prove the wiring itself -- not just the underlying primitives --
    is correct."""
    import rewards

    # Thousands of uniform-random-legal-action episodes: zero stuck states,
    # zero runaway episodes, every episode ends with no resolution pending,
    # and the termination-turn distribution is plausible (not degenerate --
    # some games assemble, most don't, under pure random play).
    rng = random.Random(7)
    num_episodes = 2000
    turn_won_counts = {}
    never = 0
    for _ in range(num_episodes):
        env = TronEnv(rewards.assembled_with_resource_quality, horizon=6, on_the_play=True, seed=rng.random())
        obs, info = env.reset()
        done = False
        steps = 0
        while not done:
            mask = env.action_masks()
            legal = [i for i, ok in enumerate(mask) if ok]
            assert legal, "no legal action available -- stuck state"
            action = rng.choice(legal)
            obs, reward, done, truncated, info = env.step(action)
            steps += 1
            assert steps < 1000, "runaway episode"
        assert env.state.pending_resolution is None, "episode ended with a resolution still pending"
        if env.state.turn_won is None:
            never += 1
        else:
            turn_won_counts[env.state.turn_won] = turn_won_counts.get(env.state.turn_won, 0) + 1

    assembled = num_episodes - never
    assert 0 < assembled < num_episodes, (
        f"degenerate distribution: {assembled}/{num_episodes} assembled under random play "
        "(expected some but not all/none)"
    )
    assert all(1 <= turn <= 6 for turn in turn_won_counts), turn_won_counts

    # End-to-end: cast Candy Trail through the REAL action table (not
    # game.py's primitives directly) -- pay its {1} cost with a tapped
    # Forest, then resolve its ETB scry 2, keeping one card in a
    # model-chosen order, dispatched purely by walking legal_action_mask
    # and looking up actions by name, exactly like a real policy would.
    state = game.GameState(on_the_play=True, rng=random.Random(0))
    state.turn_number = 2
    state.hand = [game.CARD_DEFS["Candy Trail"]]
    state.battlefield = [game.Permanent(game.CARD_DEFS["Forest"])]
    state.library = [game.CARD_DEFS["Urza's Tower"], game.CARD_DEFS["Rooftop Percher"]] + [game.CARD_DEFS["Rooftop Percher"]] * 10

    mask = legal_action_mask(state)
    assert ACTIONS[_find_action("Cast Candy Trail")][1](state)  # legal_fn directly, no dispatch needed
    ACTIONS[_find_action("Cast Candy Trail")][2](state)  # execute_fn
    assert state.pending_resolution["kind"] == "pay_cost"

    mask = legal_action_mask(state)
    forest_action = _find_action("Choose: Forest")
    assert mask[forest_action], "Forest should be offered to cover the {1} generic cost"
    ACTIONS[forest_action][2](state)
    assert state.pending_resolution["kind"] == "scry", "payment complete -- Candy Trail's ETB scry begins"

    mask = legal_action_mask(state)
    keep_action = _find_action("Keep (scry/surveil)")
    dispose_action = _find_action("Dispose (scry/surveil)")
    assert mask[keep_action] and mask[dispose_action]
    ACTIONS[keep_action][2](state)   # keep Urza's Tower
    ACTIONS[dispose_action][2](state)  # dispose the Rooftop Percher
    assert state.pending_resolution is None, "only 1 kept -- no ordering phase needed"
    assert state.library[0].name == "Urza's Tower"
    assert any(p.card_def.name == "Candy Trail" for p in state.battlefield)
    assert any(p.card_def.name == "Forest" and p.tapped for p in state.battlefield)

    # Abandon payment through the real table: tap Bonder's Ornament for
    # the wrong color, confirm "Abandon payment" is offered and reverses it.
    state2 = game.GameState(on_the_play=True, rng=random.Random(0))
    state2.hand = [game.CARD_DEFS["Ancient Stirrings"]]
    state2.battlefield = [game.Permanent(game.CARD_DEFS["Bonder's Ornament"])]
    state2.library = [game.CARD_DEFS["Rooftop Percher"]] * 10
    ACTIONS[_find_action("Cast Ancient Stirrings")][2](state2)
    assert state2.pending_resolution["kind"] == "pay_cost"
    ACTIONS[_find_action("Choose: Bonder's Ornament as W")][2](state2)
    ornament = next(p for p in state2.battlefield if p.card_def.name == "Bonder's Ornament")
    assert ornament.tapped is True
    assert state2.pending_resolution is not None, "W doesn't satisfy the {G} cost"
    abandon_action = _find_action("Abandon payment")
    assert legal_action_mask(state2)[abandon_action]
    ACTIONS[abandon_action][2](state2)
    assert state2.pending_resolution is None
    assert ornament.tapped is False
    assert state2.hand == [game.CARD_DEFS["Ancient Stirrings"]], "the card never left hand -- abandon is a full undo"


def _phase_multi_deck_sanity_check():
    """MULTI_DECK_PLAN.md Phase M4/M7: a second, differently-shaped deck
    (built entirely from card names/effects TRON_DECKLIST already defines,
    so no game.py changes are needed just to prove the plumbing) gets its
    own action table and observation dim, entirely independent of the
    module-level Tron ACTIONS/OBSERVATION_DIM globals, and its own
    terminated_fn drives its own win condition -- not Tron's."""
    import rewards

    toy_decklist = [
        ("Forest", 40, game.CardType.LAND, None, game.EffectId.FOREST, {}),
        ("Rooftop Percher", 20, game.CardType.FILLER, None, game.EffectId.FILLER, {}),
    ]

    def toy_terminated(state):
        return sum(1 for p in state.battlefield if p.card_def.name == "Forest") >= 3

    toy_env = TronEnv(rewards.assembled_with_resource_quality, decklist=toy_decklist,
                       terminated_fn=toy_terminated, horizon=6, on_the_play=True, seed=0)

    # A wholly separate action table/observation space from Tron's -- a toy
    # 2-distinct-card deck has far fewer actions and a smaller observation.
    assert len(toy_env.actions) != len(ACTIONS), "toy deck should not collide with Tron's action count"
    assert toy_env.observation_dim != OBSERVATION_DIM, "toy deck should not collide with Tron's observation dim"
    assert toy_env.observation_dim == observation_dim_for(toy_decklist)

    obs, info = toy_env.reset()
    assert obs.shape == (toy_env.observation_dim,)

    # Play every available land drop each turn (only "Forest" and "Pass"
    # are ever legal in this toy deck) until either the toy win condition
    # fires or the horizon is reached -- proving reset/step/masking all
    # work end-to-end against a non-Tron decklist/terminated_fn.
    done = False
    steps = 0
    while not done:
        mask = toy_env.action_masks()
        legal = [i for i, ok in enumerate(mask) if ok]
        assert legal, "no legal action available -- stuck state"
        land_action = next((i for i in legal if toy_env.actions[i][0] == "Play land: Forest"), None)
        action = land_action if land_action is not None else toy_env.pass_action
        obs, reward, done, truncated, info = toy_env.step(action)
        steps += 1
        assert steps < 1000, "runaway episode"
    assert toy_env.state.turn_won is not None, "expected the toy win condition (3+ Forests) to fire"

    # Meanwhile Tron's own env, built with no decklist/terminated_fn
    # arguments at all, is completely unaffected -- still the same
    # module-level ACTIONS/OBSERVATION_DIM/PASS_ACTION.
    tron_env_instance = TronEnv(rewards.assembled_with_resource_quality, horizon=6, on_the_play=True, seed=0)
    assert len(tron_env_instance.actions) == len(ACTIONS)
    assert tron_env_instance.observation_dim == OBSERVATION_DIM
    assert tron_env_instance.pass_action == PASS_ACTION


def _phase_spy_combo_fuzz_sanity_check():
    """Real spy_combo deck (game.SPY_COMBO_DECKLIST/spy_combo_terminated),
    driven through the real, fully-generated action table exactly like
    Phase M4g does for Tron: random-legal-action rollouts, primarily as a
    robustness check (no stuck states, no runaways, no crashes across
    every new mechanic's action-table wiring) rather than a policy-quality
    one -- reaching 20 damage under pure random play within a modest
    horizon is a bonus, not asserted, since the real combo is specific
    enough that random play may rarely or never find it.

    Uses a trivial placeholder reward_fn, not rewards.assembled_with_
    resource_quality: that function's resource_quality_components helper
    hardcodes Tron's own two flexible mana sources (Wooded Ridgeline,
    Bonder's Ornament) and raises on spy_combo's new ones (Saruli
    Caretaker, Lotus Petal). Generalizing it is rewards.py work, out of
    scope here (the real spy_combo reward function is deferred, same as
    the rest of rewards.py) -- this env-robustness check doesn't need a
    real reward at all."""
    def _dummy_reward(state, done, horizon):
        return 0.0

    rng = random.Random(11)
    num_episodes = 500
    horizon = 15
    wins = 0
    for _ in range(num_episodes):
        env = TronEnv(
            _dummy_reward, decklist=game.SPY_COMBO_DECKLIST,
            terminated_fn=game.spy_combo_terminated, horizon=horizon, on_the_play=True, seed=rng.random(),
        )
        obs, info = env.reset()
        done = False
        steps = 0
        while not done:
            mask = env.action_masks()
            legal = [i for i, ok in enumerate(mask) if ok]
            assert legal, "no legal action available -- stuck state"
            action = rng.choice(legal)
            obs, reward, done, truncated, info = env.step(action)
            steps += 1
            assert steps < 2000, "runaway episode"
        assert env.state.pending_resolution is None, "episode ended with a resolution still pending"
        if env.state.turn_won is not None:
            wins += 1
    print(f"  (informational: {wins}/{num_episodes} reached 20 damage under pure random play)")


if __name__ == "__main__":
    _phase_d2_sanity_check()
    print("Phase D2 OK: observation/action spaces, masking, reset/step all behave correctly.")

    _phase_m4g_sanity_check()
    print("Phase M4g OK: 2000 random episodes with zero stuck states/runaways and a plausible termination "
          "distribution; a full Candy Trail cast-then-scry and an abandoned payment both verified end-to-end "
          "through the real ACTIONS table.")

    _phase_multi_deck_sanity_check()
    print("Multi-deck OK: a second decklist/terminated_fn produces its own independent action table and "
          "observation dim, resolves to its own win condition, and Tron's own env is unaffected.")

    _phase_spy_combo_fuzz_sanity_check()
    print("spy_combo fuzz OK: 500 random episodes against the real 60-card deck through the real action "
          "table, zero stuck states/runaways/leftover pending resolutions.")
