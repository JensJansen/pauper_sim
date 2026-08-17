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
    currently targets me/the opponent AS A PLAYER, am-I-on-the-play, and the
    opponent's mulligans-taken / cleanup-discard-turns. Same composition
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
        # Three public facts the main policy could not see (2026-08-13). All
        # are things a real player knows at the table without any hidden
        # information, so these are observation-fidelity fixes, not new
        # capabilities: each is a deterministic function of publicly
        # observable state, consumed only as a network input, never an
        # evaluation.
        #
        # on_the_play: a plain oversight -- the MULLIGAN net has always seen it
        # (rl.mulligan.MulliganNet.N_SCALAR) while the main policy did not,
        # despite it changing correct play on essentially every turn of the
        # game (who wins the race, whether to hold up interaction).
        out.append(1.0 if me.on_the_play else 0.0)
        # Opponent mulligans: public -- you watch them shuffle back and draw
        # one fewer. Directly informs how hard to press. `s.mulligans_taken`
        # is the ACTIVE-player proxy (game/state.py:732) and _for_player pins
        # active_idx == seat_idx, so it is already mine; this is the other seat.
        out.append(min(other.mulligans_taken, 7) / 7)
        # Opponent cleanup discards: also public (discarding to hand size at
        # cleanup happens in the open). One count per TURN over the limit, not
        # per card -- a hoarding proxy the reward function already reads
        # (rl.rewards.deploy_reward's q term) but the policy could not observe
        # about its opponent. Capped at the same 7 as mulligans purely to keep
        # the input in [0, 1]; games rarely exceed it.
        out.append(min(other.cleanup_discard_turns, 7) / 7)
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


def _raise_all_false(state, seat, deck_vocab=None):
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
    # A live mana_subdecision SHORT-CIRCUITS any_pointer_legal -- it returns
    # `stage == "choose_target"` and never looks at pending_resolution at all,
    # so any other stage forces the whole pointer half of the mask to False no
    # matter what the pending wants. That is correct only while the FIXED color
    # actions cover the decision; if can_produce rejects every color at the same
    # moment, both halves are empty and this fires. Invisible from the pending
    # alone, so print it (2026-08-16: two assign_combat_damage all-Falses whose
    # blockers were all alive, i.e. NOT the 510.1a case turn.py now filters).
    sub = state.mana_subdecision
    if sub is None:
        print("    mana_subdecision=None", flush=True)
    else:
        src, tgt = sub.get("source"), sub.get("target")
        print(f"    mana_subdecision stage={sub.get('stage')!r} "
              f"source={(src.card_def.name, src.slot, 'T' if src.tapped else 'U') if src is not None else None} "
              f"target={(tgt.card_def.name, tgt.slot) if tgt is not None else None} "
              f"keys={list(sub.keys())}", flush=True)
        cp = sub.get("can_produce")
        if cp is not None:
            producible = []
            for color in ("W", "U", "B", "R", "G"):
                try:
                    if cp(state, color):
                        producible.append(color)
                except Exception as exc:
                    producible.append(f"{color}:<{type(exc).__name__}>")
            print(f"    mana_subdecision can_produce -> {producible or 'NOTHING (this is the all-False)'}", flush=True)
    # The pending's own targets, and whether each is still reachable: a blocker
    # that is alive but NOT in build_token_set cannot be addressed by the
    # pointer head even though it is a legal choice.
    if pend and "blockers" in pend:
        from rl.features import build_token_set
        try:
            ids = {id(i) for _v, _r, i in build_token_set(state, seat, deck_vocab, include_rows=False) if i is not None}
        except Exception as exc:
            ids = f"<{type(exc).__name__}>"
        for b in pend["blockers"]:
            on_bf = any(b in p.battlefield for p in state.players)
            tok = (id(b) in ids) if isinstance(ids, set) else ids
            print(f"    blocker {b.card_def.name}#{b.slot} on_battlefield={on_bf} tokenized={tok}", flush=True)
    # A STRANDED PAYMENT is not a generic "the action space can't represent this
    # state" -- it is specifically a LEGALITY-MASK BUG, and it is the one
    # all-False shape whose cause is invisible from the pending alone. There is
    # no "Abandon payment" action, so a payment the agent cannot finish is
    # unescapable by construction; game.mana's STRANDING INVARIANT says this is
    # unreachable, so reaching it means one of exactly two things is wrong:
    #
    #   (a) plan_payment said yes when it should have said no, so the cast was
    #       never legal -- a bug in available_mana_units (a source counted that
    #       shouldn't be, or counted as producing more than it can) or in
    #       can_pay (the feasibility test itself);
    #   (b) plan_payment was right, and something consumed supply AFTER the
    #       announcement that its own gate should have refused -- a missing or
    #       wrong payment_survives check on some mid-payment action.
    #
    # `announced` (stashed by begin_pay_cost) vs `remaining` separates them: if
    # the units below cannot cover `announced` either, it is (a); if they could
    # have then but cannot now, it is (b). Everything printed here is the
    # SOLVER'S OWN VIEW, not just the board -- the board alone shows what is
    # there, and the bug is in the model of what is there.
    if pend and pend["kind"] == "pay_cost":
        units = game.available_mana_units(state)
        announced = pend.get("announced")
        print("    --- STRANDED PAYMENT (legality-mask bug) ---", flush=True)
        print(f"    announced_cost={announced} remaining_cost={pend['remaining']}", flush=True)
        print(f"    solver units ({len(units)}) = {sorted(''.join(sorted(u)) for u in units)}", flush=True)
        print(f"    can_pay(units, remaining)={game.can_pay(units, pend['remaining'])}"
              f"  can_pay(units, announced)="
              f"{game.can_pay(units, announced) if announced is not None else '<not recorded>'}", flush=True)
        if announced is not None:
            print("    => cause is (a) plan_payment/available_mana_units/can_pay was wrong at announce time"
                  if not game.can_pay(units, announced)
                  else "    => cause is (b) supply was consumed after announcement by an action whose"
                       " payment_survives gate is missing or wrong", flush=True)
        # Why each candidate source is or is not in that unit list. This is what
        # localises (a): compare the board against the model of the board.
        for p in state.battlefield:
            spec = game.EFFECT_REGISTRY.get(p.card_def.effect_id, {})
            if "mana" not in spec and "filter_mana" not in spec and "mana_extra_choose" not in spec:
                continue
            contributed = game.source_mana_units(state, p)
            if contributed:
                reason = f"IN as {[''.join(sorted(u)) for u in contributed]}"
            elif p.tapped:
                reason = "OUT: tapped"
            elif game.tap_summoning_locked(state, p):
                reason = "OUT: summoning-locked"
            elif spec.get("mana_extra_choose") is not None:
                reason = "OUT: mana_extra_choose (excluded by design -- its cost taps another source)"
            elif spec.get("mana_extra_available") is not None:
                reason = "OUT: mana_extra_available() false (already used this turn?)"
            elif "mana" not in spec:
                reason = "OUT: filter only, produces no mana of its own"
            else:
                reason = "OUT: no reason found -- SUSPECT, this is a source_mana_units bug"
            print(f"      {p.card_def.name}#{p.slot}: {reason}", flush=True)
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
    if pend and pend["kind"] == "pay_cost":
        raise RuntimeError(
            f"STRANDED PAYMENT: owed {pend['remaining']} (announced {pend.get('announced')}) with no legal "
            f"way to pay it. game.mana's STRANDING INVARIANT says this is unreachable, so either "
            f"plan_payment allowed a cast it should not have, or a mid-payment action consumed supply "
            f"without a payment_survives gate -- see the dump above for which."
        )
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
        _raise_all_false(state, seat, vocab)
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


# HeuristicAgent -- the hand-authored, non-learned gauntlet opponent -- lives in
# rl/heuristic_agent.py as of 2026-08-17. It imports the decision plumbing above
# (_build_decision / _executor_for / _is_pass / AlwaysKeep); this file stays the
# LEARNED path only.
