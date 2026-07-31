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
mask/feature builders it needs) -- relocated here from rl.train so that
rl.train can import SeatAgent without a circular dependency (rl.agent depends on
nothing in rl.train)."""

from collections import namedtuple

import numpy as np
import torch

import drl_env
import game
from rl.action_bridge import any_pointer_legal, execute_pointer_choice, pointer_legal_mask
from rl.arch import pad_token_batch
from rl.features import _stack_target_map, build_token_set
from rl import mulligan as mulligan_mod

DECK_SIZE_CAP = 60  # every decklist in data/ is exactly 60 cards (see rl.features's own cap-then-normalize idiom)
OPPONENT_HAND_SIZE_CAP = 12  # generous headroom over the 7-card cleanup limit for a mid-turn multi-draw hand


def _scalar_features(state, seat_idx, horizon):
    """Non-tokenized globals -- turn number, lands-played, mulligans, am-I-
    turn-player, floating mana pool, phase one-hot, my/opponent life,
    my/opponent library size, opponent's hand size, whether the stack
    currently targets me/the opponent AS A PLAYER. Same composition
    rl.deck.SCALAR_FEATURE_DIM documents (mana-pool cap of 8, matched here).
    state.mana_pool is a GameState property proxying to
    state.players[state.active_idx] (game/state.py's _active_player_property)
    -- read unconditionally, not gated, since _for_player below already
    guarantees active_idx == seat_idx for the whole duration of _read.

    Library/hand SIZE (not contents) and a declared player-target are all
    public in real Magic (either player can count a library or a hand, and a
    spell's targets are known the instant they're chosen) -- unlike the
    per-card token set, which deliberately keeps hand/library CONTENTS
    hidden (rl.features.build_token_set's own docstring). My own hand size
    isn't included: it's not hidden information from myself, so there's
    nothing to surface there."""
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
    # floating and what could still be tapped. Float-first removed "Abandon
    # payment", so a payment the agent cannot finish is unescapable by
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
                # here unconditionally used to throw ValueError and blind this
                # exact dump to a filter's real state (confirmed live:
                # monster_tron, "Barrels of Blasting Jelly#1-><ValueError>" on
                # the very crash this diagnostic exists to explain). Report
                # whether it's spent for the turn instead -- the one fact that
                # actually matters for "why can't this pay {G}".
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
    are the RAW token count, exactly what ppo_update re-pads per minibatch."""
    vocab, fixed_table, _pending_kinds = deck_ctx
    tokens = build_token_set(state, seat, vocab)
    scalar = _scalar_features(state, seat, horizon)
    identities = [identity for _idx, _row, identity in tokens]

    fixed_mask = np.asarray(drl_env.legal_action_mask(state, fixed_table), dtype=bool)
    pointer_mask = pointer_legal_mask(state, identities) if any_pointer_legal(state) else [False] * len(identities)
    full_mask = np.concatenate([fixed_mask, np.asarray(pointer_mask, dtype=bool)])

    legal = np.flatnonzero(full_mask)
    if legal.size == 0:
        _raise_all_false(state, seat)
    sole = int(legal[0]) if legal.size == 1 else None
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


def _seat_step(state, seat, deck_ctx, net, horizon, device, greedy=False):
    """One seat's main-policy decision, batch-of-1 (the sequential collector).
    Builds the decision (mask first -- see _build_decision), takes the sole legal
    action WITHOUT a forward when the state is forced, else runs the shared stack
    + per-deck net, masks, and samples (or argmaxes, when greedy). Returns
    (executor, buffer_entry, is_pass); buffer_entry is None for a forced move
    (record nothing). greedy=True is for eval (deterministic argmax); training
    always samples."""
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
    buffer_entry = (dec.tokens, dec.scalar, dec.full_mask, action_idx, float(logp.item()), float(value.item()))
    return (_executor_for(state, action_idx, dec.fixed_table, dec.identities), buffer_entry,
            _is_pass(action_idx, dec.fixed_table))


# --- the agent: one dispatch, no attribution ---

# Pregame pending kinds owned by the mulligan decider, never the main policy.
# The authoritative set: SeatAgent.decide intercepts these before the main net's
# fixed table is ever consulted, which is what lets the fixed table drop its
# mulligan actions entirely (see the harness-refactor plan, structural removal).
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
    a bottom (game.resolution.handlers: n = min(mulligans_taken, ...)) -- so it
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
    (MulliganNet or AlwaysKeep) + deck_ctx (vocab, fixed_table, pending_kinds)."""

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
