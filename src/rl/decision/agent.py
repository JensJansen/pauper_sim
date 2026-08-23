"""Per-seat decision dispatch -- routes a decision to the right sub-model:
pregame (mulligan keep/bottom) goes to the deck's mulligan decider, everything
else to its main DeckNetwork policy.

A SeatAgent bundles a seat's main policy, its pregame decider (a MulliganNet or
AlwaysKeep), and its deck_ctx. It owns the mulligan-vs-policy routing but not
reward attribution -- the rollout keeps that, since the mulligan reward is a
whole-game bandit and the PPO reward is per-decision terminal-flushed.
`decide()` returns a DecisionResult carrying the executor plus whichever
transition (PPO or mulligan) the caller should record, or None to record
nothing (a forced move, an AlwaysKeep pregame pick, or a non-recording eval
play).

This module also owns the decision-side primitives (_seat_step and the
token/mask/feature builders it needs), kept out of rl.training.train to avoid
a circular import."""

from collections import namedtuple

import numpy as np
import torch

import drl_env
import game
from rl.decision.action_bridge import any_pointer_legal, execute_pointer_choice, pointer_kind, pointer_legal_mask
from rl.model.arch import pad_token_batch
from rl.model.features import MANA_PIP_CAP, _stack_target_map, build_token_set
from rl.model import mulligan as mulligan_mod

DECK_SIZE_CAP = 60  # every decklist in data/ is exactly 60 cards (see rl.model.features's own cap-then-normalize idiom)
OPPONENT_HAND_SIZE_CAP = 12  # generous headroom over the 7-card cleanup limit for a mid-turn multi-draw hand


def _scalar_features(state, seat_idx, horizon):
    """Non-tokenized globals: turn number, lands-played, mulligans, am-I-
    turn-player, my/opponent floating mana pool, phase one-hot, my/opponent
    life, my/opponent library size, opponent's hand size, whether the stack
    currently targets me/the opponent as a player, am-I-on-the-play, and the
    opponent's mulligans-taken / cleanup-discard-turns. Length matches
    rl.model.deck.SCALAR_FEATURE_DIM.

    state.mana_pool proxies to state.players[state.active_idx]; read
    unconditionally since _for_player guarantees active_idx == seat_idx for
    the duration of _read. other.mana_pool reads the other seat's own
    PlayerState directly. My own hand size isn't included -- build_token_set
    already tokenizes my hand card-by-card."""
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
        # on_the_play, opponent mulligans taken, and opponent cleanup
        # discards this turn -- all public information, capped at 7 to stay
        # in [0, 1].
        out.append(1.0 if me.on_the_play else 0.0)
        out.append(min(other.mulligans_taken, 7) / 7)
        out.append(min(other.cleanup_discard_turns, 7) / 7)
        return out
    return drl_env._for_player(state, seat_idx, _read)


def _hand_cost_reduction_deltas(state, seat_idx):
    """{card name: cost_reduction_delta} for every distinct card in
    seat_idx's hand with a live "cost_reduction" registry spec
    (drl_env._effective_cast_cost, which reduces only generic pips). Lets
    build_token_set reconstruct a card's true current cost from its printed
    cost plus this delta. Cards with no reduction are absent, read back as
    0.0 via dict.get.

    Deduped by CardDef object, which for hand cards is the same as deduping
    by name."""
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


# A decision's forward-free part: enough to tell whether it's forced and, if
# not, to run the policy forward. full_mask is the raw-width (n_fixed +
# raw_token_count) legal mask stored in the buffer and re-padded by
# ppo_update. sole_action is the lone legal index when n_legal == 1, else None.
_Decision = namedtuple("_Decision", "tokens scalar full_mask identities fixed_table n_fixed n_legal sole_action")


def _raise_all_false(state, seat):
    """An all-False mask means the engine reached a decision state the action
    space can't represent. Raises with the pending kind and board context
    instead of letting Categorical sample uniformly over illegal positions
    and crash downstream with a misleading error."""
    pend = state.pending_resolution
    raise RuntimeError(
        f"all-False action mask for pending kind {pend['kind'] if pend else None!r} "
        f"(phase={state.phase} seat={seat} turn={state.turn_number} "
        f"active_idx={state.active_idx} turn_player={state.turn_player_idx})"
    )


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
    """Everything a decision needs that doesn't require the (expensive)
    policy forward: the token set + scalar features and the legal-action
    mask (fixed half via one drl_env.legal_action_mask sweep, pointer half
    via pointer_legal_mask). A FORCED decision (n_legal == 1) skips the
    network entirely.

    Two-pass token set: the mask only needs each token's identity, not its
    full dynamic feature row, so the first pass asks for identities only
    (include_rows=False). The real per-token rows + scalar features are only
    built once a forward pass will actually consume them (sole is None); a
    forced decision leaves dec.tokens/dec.scalar as None."""
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
    padded token width the network's logits span (n_fixed + n_padded_tokens),
    padding the extra slots False. Same convention ppo_update uses to
    rebuild a minibatch mask."""
    out = torch.zeros((n_fixed + n_padded_tokens,), dtype=torch.bool, device=device)
    out[:n_fixed] = torch.as_tensor(full_mask_np[:n_fixed], dtype=torch.bool, device=device)
    ptr = full_mask_np[n_fixed:]
    out[n_fixed:n_fixed + len(ptr)] = torch.as_tensor(ptr, dtype=torch.bool, device=device)
    return out


def _resolve_pointer_identity(state, obj):
    """A pointer candidate's identity object -> {name, slot, controller} for
    decision-weight logging. Only a live battlefield Permanent has a `slot`;
    none of the possible object shapes carry their owner directly, so
    controller is resolved by membership search -- necessary because a
    single choose_graveyard_card decision can offer candidates from either
    side's graveyard (e.g. Rooftop Percher pools both)."""
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
    matchup runs only): logs the top-5 candidate actions by post-mask
    probability, the one actually taken, and the critic's value estimate --
    all read off the forward pass _seat_step already ran, adding no extra
    inference call.

    entropy is dist.entropy() (nats), the true full-distribution entropy over
    every legal action, not derivable from the top-5 truncation below."""
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


def _seat_step(state, seat, deck_ctx, net, horizon, device, greedy=False, hidden=None, prev_action=None):
    """One seat's main-policy decision, batch-of-1. Builds the decision (mask
    first), takes the sole legal action without a forward when forced, else
    runs this deck's encoder + net, masks, and samples (or argmaxes when
    greedy). Returns (executor, buffer_entry, is_pass, hidden, prev_action);
    buffer_entry is None for a forced move.

    hidden: this seat's recurrent state (DeckNetwork's GRU), threaded in and
    returned advanced. prev_action: this seat's previous recorded action
    index, or None at game start. A forced decision returns both unchanged --
    it runs no forward and is recorded in no buffer, so ppo_update's replay
    (which derives prev_action from the preceding recorded one) must not
    advance either for it."""
    dec = _build_decision(state, seat, deck_ctx, horizon)
    if dec.sole_action is not None:
        return (_executor_for(state, dec.sole_action, dec.fixed_table, dec.identities), None,
                _is_pass(dec.sole_action, dec.fixed_table), hidden, prev_action)

    vocab_idx, features, key_padding_mask, _identities = pad_token_batch([dec.tokens], device=device)
    side_flag = features[:, :, -1]
    full_mask = _padded_full_mask(dec.full_mask, dec.n_fixed, vocab_idx.shape[1], device).unsqueeze(0)
    with torch.inference_mode():
        mine_summary, theirs_summary, token_reps = net.encoder(vocab_idx, features, key_padding_mask, side_flag)
        scalar_t = torch.as_tensor(dec.scalar, dtype=torch.float32, device=device).unsqueeze(0)
        logits, value, hidden = net(mine_summary, theirs_summary, scalar_t, token_reps,
                                    full_mask[:, dec.n_fixed:], hidden=hidden,
                                    prev_action=net.prev_action_symbols([prev_action], device))
        masked_logits = logits.masked_fill(~full_mask, -1e8)
        dist = torch.distributions.Categorical(logits=masked_logits)
        action = masked_logits.argmax(dim=-1) if greedy else dist.sample()
        logp = dist.log_prob(action)
        action_idx = int(action.item())
        _log_decision_weights(state, dec, dist, value, action_idx, full_mask)
    # Clone out of inference_mode: this hidden state carries into the next
    # decision's forward, and inference-mode tensors can't enter an autograd
    # graph.
    hidden = hidden.clone()

    buffer_entry = (dec.tokens, dec.scalar, dec.full_mask, action_idx, float(logp.item()), float(value.item()))
    return (_executor_for(state, action_idx, dec.fixed_table, dec.identities), buffer_entry,
            _is_pass(action_idx, dec.fixed_table), hidden, action_idx)


# --- the agent: one dispatch, no attribution ---

# Pregame pending kinds owned by the mulligan decider, never the main policy.
# SeatAgent.decide intercepts these before the main net's fixed table is
# consulted, letting the fixed table drop its mulligan actions entirely.
PREGAME_KINDS = ("mulligan_decision", "mulligan_bottom")

# executor: zero-arg callable applying the chosen action (None => Pass).
# ppo_entry: main-policy buffer entry to record, or None (forced move / pregame).
# mull_entry: mulligan transition to record, or None (not a pregame decision,
#   or AlwaysKeep). Its reward slot is filled at game end.
# is_pass: True if the chosen action is a Pass (executor is None).
DecisionResult = namedtuple("DecisionResult", "executor ppo_entry mull_entry is_pass")


class AlwaysKeep:
    """Trivial pregame decider: always keeps the opening hand, trains
    nothing. mulligan_bottom is unreachable under keep-always (0 mulligans
    never opens a bottom), so it asserts rather than guess if ever asked."""

    def decide(self, state):
        pend = state.pending_resolution
        assert pend is not None and pend["kind"] == "mulligan_decision", (
            f"AlwaysKeep only handles mulligan_decision; got {pend['kind'] if pend else None!r} "
            "(mulligan_bottom is unreachable under keep-always)"
        )
        return (lambda state=state: game.execute_mulligan_keep(state))


class RandomMulligan:
    """Baseline pregame decider: uniform-random keep/mulligan (respecting the
    London cap) and uniform-random bottom-card choice. Trains nothing -- a
    pure-noise floor to evaluate a trained MulliganNet against. Unlike
    AlwaysKeep, mulligan_bottom is reachable here, so both pending kinds are
    handled. Takes an rng for reproducibility under a seeded caller."""

    def __init__(self, rng):
        self.rng = rng

    def decide(self, state):
        pend = state.pending_resolution
        if pend["kind"] == "mulligan_decision":
            mull_legal = state.mulligans_taken < game.HAND_SIZE_LIMIT
            if mull_legal and self.rng.random() < 0.5:
                return (lambda state=state: game.execute_mulligan_take(state))
            return (lambda state=state: game.execute_mulligan_keep(state))
        name = self.rng.choice(game.bottom_options(state))
        return (lambda state=state, name=name: game.execute_bottom_option(state, name))


class MulliganZeroLands:
    """Baseline pregame decider: mulligans a 0-land hand, keeps anything
    else. Checked fresh at every decision, so a post-mulligan redraw that's
    still 0 lands mulligans again too. Trains nothing -- an opponent-side
    floor that removes the one unambiguous hand-quality mistake, so a game
    the opponent loses reflects the training net's decision quality rather
    than the opponent tanking on an unkeepable hand. Bottom-card choice is
    uniform-random, same as RandomMulligan. Takes an rng for the bottom
    pick."""

    def __init__(self, rng):
        self.rng = rng

    def decide(self, state):
        pend = state.pending_resolution
        if pend["kind"] == "mulligan_decision":
            mull_legal = state.mulligans_taken < game.HAND_SIZE_LIMIT
            no_lands = all(c.card_type.name != "LAND" for c in state.hand)
            if mull_legal and no_lands:
                return (lambda state=state: game.execute_mulligan_take(state))
            return (lambda state=state: game.execute_mulligan_keep(state))
        name = self.rng.choice(game.bottom_options(state))
        return (lambda state=state, name=name: game.execute_bottom_option(state, name))


class SeatAgent:
    """One seat's decision-maker: main policy (DeckNetwork) + pregame decider
    (a MulliganNet, or a simple non-network decider like AlwaysKeep/
    RandomMulligan) + deck_ctx (vocab, fixed_table) + this seat's live
    recurrent state for the game currently being played."""

    def __init__(self, main, mulligan, deck_ctx):
        self.main = main            # DeckNetwork (None only in decide()'s pregame-only unit tests)
        self.mulligan = mulligan    # MulliganNet, or a simple decider (AlwaysKeep/RandomMulligan)
        self.deck_ctx = deck_ctx
        self.vocab = deck_ctx[0]
        self.hidden = {}            # seat -> that seat's GRU state this game; see reset()
        self.prev_action = {}       # seat -> its own last RECORDED action index this game

    def reset(self):
        """Clears the recurrent state. Must be called at the start of every
        game -- collect_rollout's game loop does this. Not optional: a
        SeatAgent is reused across games (LeaguePool caches loaded snapshots
        by path), so without this the GRU would carry one game's state into
        the next, and ppo_update -- which replays every episode from a zero
        state -- would recompute hidden states that never occurred."""
        self.hidden = {}
        self.prev_action = {}

    def decide(self, state, seat, horizon, device, greedy=False):
        pend = state.pending_resolution
        if pend is not None and pend["kind"] in PREGAME_KINDS:
            # Anything that isn't a real MulliganNet is a simple decider
            # (AlwaysKeep, RandomMulligan, ...) satisfying only the narrower
            # decide(state) contract.
            if not isinstance(self.mulligan, mulligan_mod.MulliganNet):
                return DecisionResult(self.mulligan.decide(state), None, None, is_pass=False)
            # MulliganNet: capture its transition (reward slot filled later
            # by the rollout) via a local sink list.
            sink = []
            executor = mulligan_mod.decide(self.mulligan, self.vocab, state, seat, sink.append, greedy=greedy)
            return DecisionResult(executor, None, sink[0] if sink else None, is_pass=False)
        # Keyed by seat, not one state per agent: a mirror pairing puts this
        # same object on both seats, and a single shared state would leak
        # seat 0's hidden information into seat 1's conditioning and break
        # ppo_update's per-seat episode replay.
        executor, ppo_entry, is_pass, self.hidden[seat], self.prev_action[seat] = _seat_step(
            state, seat, self.deck_ctx, self.main, horizon, device, greedy=greedy,
            hidden=self.hidden.get(seat), prev_action=self.prev_action.get(seat))
        return DecisionResult(executor, ppo_entry, None, is_pass)
