"""Per-seat decision dispatch -- the ONE place that routes a decision to the
right sub-model: pregame (mulligan keep/bottom) goes to the deck's mulligan
decider, everything else to its main DeckNetwork policy.

A SeatAgent bundles a seat's main policy, its pregame decider (a MulliganNet or
AlwaysKeep), and its deck_ctx. It owns the mulligan-vs-policy routing but NOT
reward attribution -- the rollout keeps that, because the mulligan reward is a
whole-game bandit and the PPO reward is per-decision terminal-flushed (two
different rules). `decide()` returns a DecisionResult carrying the executor plus
whichever transition (PPO or mulligan) the caller should record, or None to
record nothing (a forced move, an AlwaysKeep pregame pick, or a non-recording
eval play).

This module also owns the decision-side primitives (_seat_step and the token/
mask/feature builders it needs), keeping them out of rl.train so that rl.train
can import SeatAgent without a circular dependency (rl.agent depends on
nothing in rl.train)."""

from collections import namedtuple

import numpy as np
import torch

import drl_env
import game
from rl.action_bridge import any_pointer_legal, execute_pointer_choice, pointer_kind, pointer_legal_mask
from rl.arch import pad_token_batch
from rl.features import MANA_PIP_CAP, _stack_target_map, build_token_set
from rl import mulligan as mulligan_mod

DECK_SIZE_CAP = 60  # every decklist in data/ is exactly 60 cards (see rl.features's own cap-then-normalize idiom)
OPPONENT_HAND_SIZE_CAP = 12  # generous headroom over the 7-card cleanup limit for a mid-turn multi-draw hand


def _scalar_features(state, seat_idx, horizon):
    """Non-tokenized globals -- turn number, lands-played, mulligans, am-I-
    turn-player, my/opponent floating mana pool, phase one-hot, my/opponent
    life, my/opponent library size, opponent's hand size, whether the stack
    currently targets me/the opponent AS A PLAYER. Same composition
    rl.deck.SCALAR_FEATURE_DIM documents (mana-pool cap of 8, matched here).
    state.mana_pool is a GameState property proxying to
    state.players[state.active_idx] (game/state.py's _active_player_property)
    -- read unconditionally, not gated, since _for_player below already
    guarantees active_idx == seat_idx for the whole duration of _read.
    other.mana_pool reads straight off THAT PlayerState instead (game/
    state.py: every PlayerState owns its own mana_pool dict; state.mana_pool
    is just the active-player convenience proxy over the same dicts), so
    it's correct regardless of active_idx too. Floating mana is public
    information in real Magic (either player can see how much of each color
    the other has floating) -- both pools belong here on that basis alone,
    same reasoning library/hand SIZE below already follows.

    Library/hand SIZE (not contents) and a declared player-target are all
    public in real Magic (either player can count a library or a hand, and a
    spell's targets are known the instant they're chosen) -- unlike the
    per-card token set, which deliberately keeps OPPONENT hand/library
    CONTENTS hidden (rl.features.build_token_set's own docstring). My own
    hand size isn't included here: build_token_set now tokenizes my own
    hand card-by-card, so a redundant aggregate count here would add
    nothing a sum over those tokens doesn't already give the network."""
    def _read(s):
        me = s.players[seat_idx]
        other = s.players[1 - seat_idx]
        _obj_controllers, player_controllers = _stack_target_map(s)
        out = [
            min(s.turn_number / horizon, 1.0),
            1.0 if s.lands_played_this_turn > 0 else 0.0,
            min(s.mulligans_taken, 7) / 7,
            1.0 if s.active_idx == s.turn_player_idx else 0.0,
        ]
        for color in game.POOL_COLORS:
            out.append(min(s.mana_pool.get(color, 0), 8) / 8)
        for color in game.POOL_COLORS:
            out.append(min(other.mana_pool.get(color, 0), 8) / 8)
        for phase in game.turn.Phase:
            out.append(1.0 if phase == s.phase else 0.0)
        out.append(max(me.life_total, 0) / game.state.STARTING_LIFE)
        out.append(max(other.life_total, 0) / game.state.STARTING_LIFE)
        out.append(min(len(me.library), DECK_SIZE_CAP) / DECK_SIZE_CAP)
        out.append(min(len(other.library), DECK_SIZE_CAP) / DECK_SIZE_CAP)
        out.append(min(len(other.hand), OPPONENT_HAND_SIZE_CAP) / OPPONENT_HAND_SIZE_CAP)
        out.append(1.0 if player_controllers[seat_idx] else 0.0)
        out.append(1.0 if player_controllers[1 - seat_idx] else 0.0)
        return out
    return drl_env._for_player(state, seat_idx, _read)


def _hand_cost_reduction_deltas(state, seat_idx):
    """{card name: cost_reduction_delta} for every DISTINCT card currently in
    seat_idx's own hand with a live registry "cost_reduction" spec
    (drl_env._effective_cast_cost, which reduces ONLY generic pips -- its own
    docstring) -- rl.features._token_row's cost_reduction_delta slot for a
    hand token, letting the static block's printed cost plus this one number
    reconstruct a card's TRUE current cost (Tolarian Terror's graveyard
    count, affinity, Deem Inferior's cards-drawn count, ...) instead of
    always showing the printed, possibly-stale one. Names with no reduction
    right now (the overwhelming majority, including every card with no
    cost_reduction spec at all) are simply absent, read back as 0.0 via
    dict.get in build_token_set.

    Deduped by CardDef object, which for hand cards (still game.CARD_DEFS's
    shared/interned objects -- see rl.features.build_token_set's own
    docstring on why hand tokens carry no real identity) is the same as
    deduping by name: two copies of a reduced-cost card always compute to
    the same delta this instant, so there's no reason to recompute it twice."""
    def _read(s):
        out = {}
        for card_def in {c for c in s.players[seat_idx].hand}:
            cost = card_def.cast_cost
            if cost is None or game.EFFECT_REGISTRY.get(card_def.effect_id, {}).get("cost_reduction") is None:
                continue
            effective = drl_env._effective_cast_cost(s, card_def)
            delta = cost.get("generic", 0) - effective.get("generic", 0)
            if delta:
                out[card_def.name] = delta / MANA_PIP_CAP
        return out
    return drl_env._for_player(state, seat_idx, _read)


# A decision's forward-FREE part: everything needed to (a) tell whether it's
# forced and (b) later run the policy forward, without having run it yet.
# full_mask is the raw-width (n_fixed + raw_token_count) legal mask, the exact
# array stored in the buffer and re-padded by ppo_update. sole_action is the
# lone legal index when n_legal == 1 (a forced move), else None.
_Decision = namedtuple("_Decision", "tokens scalar full_mask identities fixed_table n_fixed n_legal sole_action")


def _raise_all_false(state, seat):
    # DIAGNOSTIC (temporary): an all-False mask means the engine reached a
    # decision state the action space can't represent AT ALL -- a real gap.
    # masked_fill(-1e8) then Categorical would otherwise sample UNIFORMLY over
    # every (illegal) position and crash downstream (execute_pointer_choice)
    # with a misleading error. Surface the true culprit precisely instead.
    pend = state.pending_resolution
    print("  *** ALL-FALSE MASK ***", flush=True)
    print(f"    pending_kind={pend['kind'] if pend else None} phase={state.phase} seat={seat} "
          f"turn={state.turn_number} active_idx={state.active_idx} turn_player={state.turn_player_idx}", flush=True)
    if pend:
        print(f"    pending keys={list(pend.keys())}", flush=True)
        for k in ("remaining", "ordered", "kept", "disposed"):
            if k in pend:
                v = pend[k]
                print(f"    pending[{k}]={[getattr(c, 'name', c) for c in v] if isinstance(v, list) else v}", flush=True)
    # A STRANDED PAYMENT (pending_kind=pay_cost) is the one all-False shape whose
    # cause is invisible from the pending alone: it depends on what mana is
    # floating and what could still be tapped. Float-first mana has no "Abandon
    # payment" action, so a payment the agent cannot finish is unescapable by
    # construction -- meaning the mana state at this instant IS the bug report.
    # Dump both pools, every mana source and whether it is still untapped, and
    # what each untapped one could produce.
    print(f"    mana_pool(active)={dict(state.mana_pool)}", flush=True)
    for idx, player in enumerate(state.players):
        print(f"    seat{idx} pool={dict(player.mana_pool)} life={player.life_total}", flush=True)
        sources = []
        for p in player.battlefield:
            spec = game.EFFECT_REGISTRY.get(p.card_def.effect_id, {})
            if "mana" in spec:
                try:
                    out = game.mana_output(p, state) if not p.tapped else []
                except Exception as exc:  # a source whose output needs state it can't read here
                    out = f"<{type(exc).__name__}>"
                sources.append(f"{p.card_def.name}#{p.slot}{'(T)' if p.tapped else ''}->{out}")
            elif "filter_mana" in spec or "mana_extra_choose" in spec:
                # Pool->pool converters (filter_mana) have no mana_output at all
                # -- that helper only understands a "mana" spec, so calling it
                # here unconditionally would throw ValueError and blind this
                # exact dump to a filter's real state. Report whether it's spent
                # for the turn instead -- the one fact that actually matters for
                # "why can't this pay {G}".
                used = p.flags.get("used_this_turn", p.tapped)
                sources.append(f"{p.card_def.name}#{p.slot}{'(used)' if used else '(available)'}")
        print(f"    seat{idx} mana sources={sources}", flush=True)
    raise RuntimeError(f"all-False action mask for pending kind {pend['kind'] if pend else None!r}")


def _executor_for(state, action_idx, fixed_table, identities):
    """The zero-arg callable that applies action_idx: a fixed-table entry's own
    execute closure, or execute_pointer_choice on the chosen permanent (pointer
    half, indexed past the fixed table)."""
    if action_idx < len(fixed_table):
        execute_fn = fixed_table[action_idx][2]
        return (lambda state=state, execute_fn=execute_fn: execute_fn(state))
    chosen = identities[action_idx - len(fixed_table)]
    return (lambda state=state, chosen=chosen: execute_pointer_choice(state, chosen))


def _is_pass(action_idx, fixed_table):
    return action_idx < len(fixed_table) and fixed_table[action_idx][0] == "Pass"


def _build_decision(state, seat, deck_ctx, horizon):
    """Everything a decision needs that does NOT require the (expensive) policy
    forward: the token set + scalar features and the legal-action mask (fixed
    half via ONE drl_env.legal_action_mask sweep, pointer half via
    pointer_legal_mask). Forward-free by design so a FORCED decision
    (n_legal == 1 -- overwhelmingly a priority Pass with nothing to play) skips
    the network entirely.
    identities is read straight off the token set (each token is (vocab_idx,
    feature_row, identity)); the pointer mask -- and its buffer-stored width --
    are the RAW token count, exactly what ppo_update re-pads per minibatch.

    Two-pass token set: the mask (and thus forced-ness) only needs each
    token's identity, not its full dynamic feature row (permanent_power/
    toughness, blocked-by sets -- build_token_set's expensive part). So the
    first pass asks for identities only (include_rows=False); the real
    per-token rows + scalar features -- and the hand cost-reduction lookup
    those rows need (_hand_cost_reduction_deltas) -- are only built once we
    know a forward pass will actually consume them (sole is None). A forced
    decision never reads dec.tokens/dec.scalar (_seat_step returns before
    touching either), so those stay None for it rather than paying for
    values nobody reads."""
    vocab, fixed_table = deck_ctx
    cheap_tokens = build_token_set(state, seat, vocab, include_rows=False)
    identities = [identity for _idx, _row, identity in cheap_tokens]

    fixed_mask = np.asarray(drl_env.legal_action_mask(state, fixed_table), dtype=bool)
    pointer_mask = pointer_legal_mask(state, identities) if any_pointer_legal(state) else [False] * len(identities)
    full_mask = np.concatenate([fixed_mask, np.asarray(pointer_mask, dtype=bool)])

    legal = np.flatnonzero(full_mask)
    if legal.size == 0:
        _raise_all_false(state, seat)
    sole = int(legal[0]) if legal.size == 1 else None
    if sole is not None:
        return _Decision(None, None, full_mask, identities, fixed_table, len(fixed_table), int(legal.size), sole)
    hand_cost_reduction = _hand_cost_reduction_deltas(state, seat)
    tokens = build_token_set(state, seat, vocab, hand_cost_reduction=hand_cost_reduction)
    scalar = _scalar_features(state, seat, horizon)
    return _Decision(tokens, scalar, full_mask, identities, fixed_table, len(fixed_table), int(legal.size), sole)


def _padded_full_mask(full_mask_np, n_fixed, n_padded_tokens, device):
    """Widen a decision's raw full_mask (n_fixed + raw_token_count) to the
    padded token width the network's logits actually span (n_fixed +
    n_padded_tokens), padding the extra token slots False -- the same
    convention ppo_update uses to rebuild a minibatch mask, and the reason the
    empty-board case (pad_token_batch pads 0 tokens to 1 dummy slot) is safe."""
    out = torch.zeros((n_fixed + n_padded_tokens,), dtype=torch.bool, device=device)
    out[:n_fixed] = torch.as_tensor(full_mask_np[:n_fixed], dtype=torch.bool, device=device)
    ptr = full_mask_np[n_fixed:]
    out[n_fixed:n_fixed + len(ptr)] = torch.as_tensor(ptr, dtype=torch.bool, device=device)
    return out


def _resolve_pointer_identity(state, obj):
    """A pointer candidate's identity object (see rl.features.build_token_set's
    own docstring for the possible shapes) -> {name, slot, controller} for
    decision-weight logging. Only a live battlefield Permanent has a `slot`;
    a graveyard CardInstance, a stack-entry dict, and a bare revealed-hand
    CardDef (the deferred Mesmeric-Fiend path) don't, and none of them carry
    their owner on the object itself -- controller is resolved by membership
    search. That search is necessary, not just careful: Rooftop Percher's ETB
    pools BOTH players' graveyards into one pending
    (`combined = [c for pl in state.players for c in pl.graveyard]`,
    game/catalog/colorless_cards.py), so a single choose_graveyard_card
    decision can genuinely offer candidates from either side -- there's no
    shortcut like "check which player's list the pending references"."""
    if obj is None:
        return None
    if isinstance(obj, dict):  # a real stack entry (game/effects/stack.py's push_to_stack)
        card_def = obj.get("card_def")
        return {"name": card_def.name if card_def else None, "slot": None, "controller": obj.get("controller")}
    if isinstance(obj, game.Permanent):
        for i, p in enumerate(state.players):
            if obj in p.battlefield:
                return {"name": obj.name, "slot": obj.slot, "controller": i}
        return {"name": obj.name, "slot": obj.slot, "controller": None}
    for i, p in enumerate(state.players):
        if obj in p.graveyard or obj in p.hand:
            return {"name": obj.name, "slot": None, "controller": i}
    return {"name": getattr(obj, "name", None), "slot": None, "controller": None}


def _log_decision_weights(state, dec, dist, value, action_idx, full_mask_t):
    """Opt-in instrumentation (state.event_log is not None -- --log eval/
    matchup runs only, see game/state.py's log_event docstring): the top-5
    candidate actions by the network's own post-mask probability, the one
    actually taken, and the critic's value estimate -- all already computed
    in the SAME forward pass _seat_step just ran, so this adds no inference
    call and no randomness. See todo/game_visualization.md's "Decision-point
    overlay" section for the full design.

    entropy is dist.entropy() (nats, same units PPO's own entropy bonus and
    training-time logging use) -- the TRUE full-distribution entropy over
    every legal action, not derivable from the top-5 `candidates` truncation
    below. Same free-lunch reasoning as value_estimate: dist already exists
    from this decision's own forward pass."""
    if state.event_log is None:
        return
    probs = dist.probs[0]
    legal_idx = full_mask_t[0].nonzero(as_tuple=True)[0]
    top = torch.topk(probs[legal_idx], min(5, legal_idx.numel()))
    candidates = []
    for idx, prob in zip(legal_idx[top.indices].tolist(), top.values.tolist()):
        if idx < dec.n_fixed:
            candidates.append({"index": idx, "probability": prob, "fixed_label": dec.fixed_table[idx][0], "pointer_identity": None})
        else:
            token_idx = idx - dec.n_fixed
            identity = dec.identities[token_idx] if token_idx < len(dec.identities) else None
            candidates.append({"index": idx, "probability": prob, "fixed_label": None,
                                "pointer_identity": _resolve_pointer_identity(state, identity)})
    state.log_event(
        "decision_weights", network="main", chosen_index=action_idx, value_estimate=float(value.item()),
        entropy=float(dist.entropy().item()), candidates=candidates, pointer_kind=pointer_kind(state),
    )


def _seat_step(state, seat, deck_ctx, net, horizon, device, greedy=False):
    """One seat's main-policy decision, batch-of-1 (the sequential collector).
    Builds the decision (mask first -- see _build_decision), takes the sole legal
    action WITHOUT a forward when the state is forced, else runs the shared stack
    + per-deck net, masks, and samples (or argmaxes, when greedy). Returns
    (executor, buffer_entry, is_pass); buffer_entry is None for a forced move
    (record nothing). greedy=True is for eval (deterministic argmax); training
    always samples. Forced decisions never call _log_decision_weights -- no
    forward pass ran, so there's nothing to log (and nothing informative
    about a decision with one legal option anyway)."""
    dec = _build_decision(state, seat, deck_ctx, horizon)
    if dec.sole_action is not None:
        return (_executor_for(state, dec.sole_action, dec.fixed_table, dec.identities), None,
                _is_pass(dec.sole_action, dec.fixed_table))

    vocab_idx, features, key_padding_mask, _identities = pad_token_batch([dec.tokens], device=device)
    side_flag = features[:, :, -1]
    full_mask = _padded_full_mask(dec.full_mask, dec.n_fixed, vocab_idx.shape[1], device).unsqueeze(0)
    with torch.inference_mode():
        mine_summary, theirs_summary, token_reps = net.shared_stack(vocab_idx, features, key_padding_mask, side_flag)
        scalar_t = torch.as_tensor(dec.scalar, dtype=torch.float32, device=device).unsqueeze(0)
        logits, value = net(mine_summary, theirs_summary, scalar_t, token_reps, full_mask[:, dec.n_fixed:])
        masked_logits = logits.masked_fill(~full_mask, -1e8)
        dist = torch.distributions.Categorical(logits=masked_logits)
        action = masked_logits.argmax(dim=-1) if greedy else dist.sample()
        logp = dist.log_prob(action)
        action_idx = int(action.item())
        _log_decision_weights(state, dec, dist, value, action_idx, full_mask)

    buffer_entry = (dec.tokens, dec.scalar, dec.full_mask, action_idx, float(logp.item()), float(value.item()))
    return (_executor_for(state, action_idx, dec.fixed_table, dec.identities), buffer_entry,
            _is_pass(action_idx, dec.fixed_table))


# --- the agent: one dispatch, no attribution ---

# Pregame pending kinds owned by the mulligan decider, never the main policy.
# The authoritative set: SeatAgent.decide intercepts these before the main net's
# fixed table is ever consulted, which is what lets the fixed table drop its
# mulligan actions entirely.
PREGAME_KINDS = ("mulligan_decision", "mulligan_bottom")

# executor: zero-arg callable applying the chosen action (None => Pass).
# ppo_entry: main-policy buffer entry to record, or None (forced move / pregame).
# mull_entry: mulligan transition to record, or None (not a pregame decision, or
#   AlwaysKeep, which trains nothing). Its reward slot is filled at game end.
# is_pass: True if the chosen action is a Pass (executor is None).
DecisionResult = namedtuple("DecisionResult", "executor ppo_entry mull_entry is_pass")


class AlwaysKeep:
    """Trivial pregame decider: always keep the opening hand (0 mulligans),
    train nothing. Used wherever mulligan training is not wanted (pretrain,
    self-checks, an eval before a trained mulligan model exists). mulligan_bottom
    is UNREACHABLE under keep-always -- keeping with 0 mulligans taken never opens
    a bottom (game.resolution.handlers_mulligan: n = min(mulligans_taken, ...)) -- so it
    asserts rather than guess if ever asked."""

    def decide(self, state):
        pend = state.pending_resolution
        assert pend is not None and pend["kind"] == "mulligan_decision", (
            f"AlwaysKeep only handles mulligan_decision; got {pend['kind'] if pend else None!r} "
            "(mulligan_bottom is unreachable under keep-always)"
        )
        return (lambda state=state: game.execute_mulligan_keep(state))


class SeatAgent:
    """One seat's decision-maker: main policy (DeckNetwork) + pregame decider
    (MulliganNet or AlwaysKeep) + deck_ctx (vocab, fixed_table)."""

    def __init__(self, main, mulligan, deck_ctx):
        self.main = main            # DeckNetwork (None only in decide()'s pregame-only unit tests)
        self.mulligan = mulligan    # MulliganNet or AlwaysKeep
        self.deck_ctx = deck_ctx
        self.vocab = deck_ctx[0]

    def decide(self, state, seat, horizon, device, greedy=False):
        pend = state.pending_resolution
        if pend is not None and pend["kind"] in PREGAME_KINDS:
            if isinstance(self.mulligan, AlwaysKeep):
                return DecisionResult(self.mulligan.decide(state), None, None, is_pass=False)
            # MulliganNet: capture its (reward-slot-None) transition via a local
            # sink so attribution stays the rollout's job, not the agent's. The
            # list object is returned by reference -- the rollout fills entry[5]
            # at game end.
            sink = []
            executor = mulligan_mod.decide(self.mulligan, self.vocab, state, seat, sink.append, greedy=greedy)
            return DecisionResult(executor, None, sink[0] if sink else None, is_pass=False)
        executor, ppo_entry, is_pass = _seat_step(state, seat, self.deck_ctx, self.main, horizon, device, greedy=greedy)
        return DecisionResult(executor, ppo_entry, None, is_pass)


# --- a hand-authored, non-learned opponent (the gauntlet's tier-1 member) ---

def _card_name_from_cast_action(name):
    """"Cast X" / "Cast X (free)" / "Cast X (plotted)" / "Cast X (omen)" /
    "Cast X (prototype)" -> X, matching drl_env.build_action_table's own
    f"Cast {name}" / f"Cast {name} ({suffix})" naming. Only ever called on a
    name already known to start with "Cast " -- a parenthesized suffix is the
    only other thing that can follow the card name in that scheme."""
    rest = name[len("Cast "):]
    paren = rest.find(" (")
    return rest[:paren] if paren != -1 else rest


def _cmc(card_def):
    return sum(card_def.cast_cost.values()) if card_def.cast_cost else 0


def _fixed_tier(name):
    """Rough priority for a legal fixed-table action when no pointer rule
    applies: a land drop first, then a cast (ties broken by CMC below --
    "cast the highest-cost thing you can afford"; legality already means
    affordable), then anything else fixed (Activate, Forestcycle, Decline,
    Abandon payment, ...), Pass only as an actual last resort -- the owner's
    own rule order ends "...; else pass"."""
    if name.startswith("Play land: "):
        return 3
    if name.startswith("Cast "):
        return 2
    if name == "Pass":
        return 0
    return 1


def _best_fixed_index(fixed_table, fixed_legal):
    def key(i):
        name = fixed_table[i][0]
        cmc = _cmc(game.CARD_DEFS[_card_name_from_cast_action(name)]) if name.startswith("Cast ") else 0
        return (_fixed_tier(name), cmc)
    return max(fixed_legal, key=key)


def _fixed_index_named(fixed_table, fixed_legal, name):
    for i in fixed_legal:
        if fixed_table[i][0] == name:
            return i
    return None


def _could_be_blocked_by(state, attacker, blocker):
    """The one block restriction this engine encodes beyond raw eligibility
    (declare_blocker_assignment's own extra_predicate; see action_bridge.py's
    execute_pointer_choice for the same check on the real assignment path):
    a flying attacker can only be blocked by a flying blocker."""
    return not game.has_keyword(state, attacker, "flying") or game.has_keyword(state, blocker, "flying")


def _attack_is_worthwhile(state, attacker):
    """Owner's rule: attack if the creature is SAFE (nothing can kill it).
    If not safe, attack only if the CHEAPEST creature that can profitably
    kill it costs at least as much mana as the attacker itself -- an
    equal-or-worse-for-them trade is fine, a strictly cheaper answer is not."""
    my_toughness = game.permanent_toughness(state, attacker)
    threats = [
        p for p in state.opponent.battlefield
        if p.card_def.card_type == game.CardType.CREATURE and not p.tapped
        and _could_be_blocked_by(state, attacker, p)
        and game.permanent_power(state, p) >= my_toughness
    ]
    if not threats:
        return True
    my_cmc = _cmc(attacker.card_def)
    cheapest_threat_cmc = min(_cmc(p.card_def) for p in threats)
    return cheapest_threat_cmc >= my_cmc


def _blocker_worth_assigning(state, blocker):
    """"Block to kill when possible" -- does this potential blocker
    (state.battlefield, already block-eligible per the engine's own mask)
    have enough power to kill at least one of the opponent's currently
    declared attackers. Deliberately rough: doesn't re-derive already-
    assigned/gang-block state itself -- the engine's own legal-action mask
    is what actually restricts which blocker/attacker pairings are offered,
    so picking a legal-but-redundant blocker here is at worst a wasted
    action, never an illegal or crashing one."""
    return any(
        _could_be_blocked_by(state, attacker, blocker)
        and game.permanent_power(state, blocker) >= game.permanent_toughness(state, attacker)
        for attacker in state.opponent.attackers
    )


def _heuristic_action_index(state, dec):
    """Picks one legal index (fixed or pointer half) per the owner's rough
    rule set: play a land if possible; else cast the highest-cost thing
    affordable; attack a creature only if safe or a fair-or-better trade;
    block to kill when possible; else pass. Any pointer category outside
    attack/block/"which attacker does my just-assigned blocker hit" (a
    resolving spell's own target choice, a sacrifice cost, ...) has no rule
    from the owner -- falls back to the first legal option, deterministic
    rather than guessed-at, since this bot was only asked to cover general
    play principles, not reimplement optimal targeting for every card."""
    legal = np.flatnonzero(dec.full_mask)
    n_fixed = dec.n_fixed
    fixed_legal = [i for i in legal if i < n_fixed]
    pointer_legal = [i for i in legal if i >= n_fixed]
    pending = state.pending_resolution

    if (pending is None and pointer_legal and state.phase is game.turn.Phase.DECLARE_ATTACKERS
            and state.active_idx == state.turn_player_idx):
        worthwhile = [i for i in pointer_legal if _attack_is_worthwhile(state, dec.identities[i - n_fixed])]
        if worthwhile:
            return max(worthwhile, key=lambda i: game.permanent_power(state, dec.identities[i - n_fixed]))
        named = _fixed_index_named(dec.fixed_table, fixed_legal, "Pass")
        if named is not None:
            return named

    elif pending is not None and pending["kind"] == "declare_blockers" and pointer_legal:
        worthwhile = [i for i in pointer_legal if _blocker_worth_assigning(state, dec.identities[i - n_fixed])]
        if worthwhile:
            return max(worthwhile, key=lambda i: game.permanent_power(state, dec.identities[i - n_fixed]))
        named = _fixed_index_named(dec.fixed_table, fixed_legal, "Done blocking")
        if named is not None:
            return named

    elif pending is not None and pending["kind"] == "choose_opponent_permanent" and pointer_legal:
        # The common real use of this primitive here is "which of my just-
        # assigned blocker's legal attackers does it block" -- prefer the
        # biggest threat among whatever's actually offered (already filtered
        # to what THIS specific blocker may legally block).
        return max(pointer_legal, key=lambda i: game.permanent_power(state, dec.identities[i - n_fixed]))

    if fixed_legal:
        return _best_fixed_index(dec.fixed_table, fixed_legal)
    return int(legal[0])  # a pure pointer decision with no fixed fallback at all -- first legal, deterministic


class HeuristicAgent:
    """A hand-authored, non-learned opponent -- the gauntlet mechanism's
    tier-1 member (see README's Gauntlet section). Reuses the SAME legal-
    action machinery a trained SeatAgent's main policy does (_build_decision,
    _executor_for) but scores among the legal choices by the owner's own
    rough, general-principle rules instead of a trained policy: a fixed,
    non-self-play reference for catching population-wide blind spots a
    hand-authored, never-adapting strategy wouldn't share. Pregame (mulligan)
    delegates to AlwaysKeep -- no heuristic pregame logic was asked for."""

    def __init__(self, deck_ctx):
        self.deck_ctx = deck_ctx
        self.mulligan = AlwaysKeep()

    def decide(self, state, seat, horizon, device, greedy=False):
        pend = state.pending_resolution
        if pend is not None and pend["kind"] in PREGAME_KINDS:
            return DecisionResult(self.mulligan.decide(state), None, None, is_pass=False)
        dec = _build_decision(state, seat, self.deck_ctx, horizon)
        action_idx = dec.sole_action if dec.sole_action is not None else _heuristic_action_index(state, dec)
        executor = _executor_for(state, action_idx, dec.fixed_table, dec.identities)
        return DecisionResult(executor, None, None, _is_pass(action_idx, dec.fixed_table))
