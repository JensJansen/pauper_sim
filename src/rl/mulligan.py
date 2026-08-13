"""Deck-specific mulligan model + its REINFORCE trainer.

The mulligan is a near-bandit: ONE pregame decision (keep/mulligan each round,
plus which cards to bottom on a keep) whose reward is the whole game's outcome.
So its credit assignment is DIRECT -- the game that follows is a black box that
turns "the hand I kept" into a single number -- unlike the main in-game policy,
whose terminal reward has to survive ~100 steps of GAE discounting to reach the
mulligan choice, too diluted a signal to train a mulligan decision through (see
rl.rewards's deploy_reward docstring).

This model OWNS the pregame phase (rl.agent.SeatAgent routes mulligan_decision /
mulligan_bottom decisions here instead of the main net) and is trained by
REINFORCE with its OWN reward, decoupled from the main PPO update:

    reward(seat) = WIN_REWARD * (1 if seat won else 0)  -  MULLIGAN_COST * mulligans_taken**2

WIN_REWARD (1.0) dominates so "mulligan to win" stays reinforced; the penalty is
CONVEX (quadratic) so the 1st mulligan is nearly free but each further one hurts
more than the last -- the model mulligans a hand iff doing so raises its win
probability by more than that (rising) marginal cost, and mulliganing toward zero
is strongly discouraged. The reward goes NEGATIVE on a mulligan-heavy loss.

It reuses the frozen shared stack's card embeddings to represent the hand (so it
inherits the card semantics the stack already learned); everything else is a
small per-deck head. Card indices come from rl.features.CardVocab.
"""
from __future__ import annotations

import game
import torch
import torch.nn as nn

HAND = game.HAND_SIZE_LIMIT  # 7 -- London mulligan cap and hand-size normalizer
WIN_REWARD = 1.0
# Convex (quadratic) per-mulligan penalty: total = MULLIGAN_COST * mulligans**2, so
# the marginal cost of the Nth mulligan grows ~linearly and the 1st is nearly free.
# 1->0.02, 2->0.08, 3->0.18, 4->0.32, 5->0.50, 6->0.72, 7->0.98 (~the whole win).
MULLIGAN_COST = 0.02
ENTROPY_COEF = 0.01  # keep some exploration so the keep/mull head doesn't re-collapse


def mulligan_reward(won, mulligans_taken):
    """This seat's terminal reward for its pregame decisions: a big win payout
    minus a CONVEX per-mulligan penalty (quadratic, so each extra mulligan costs
    more than the last). Negative on a mulligan-heavy loss; a win stays >=0 even
    at the 7-mulligan cap."""
    return WIN_REWARD * (1.0 if won else 0.0) - MULLIGAN_COST * mulligans_taken ** 2


class MulliganNet(nn.Module):
    """One per deck. Reuses shared_stack.embedding (frozen) to encode the hand;
    a small trunk feeds three heads: keep/mulligan logits, a value baseline, and
    a pointer query used to score bottom candidates against their embeddings."""

    N_SCALAR = 2  # mulligans_taken/HAND, on_the_play

    def __init__(self, shared_stack, hidden=64):
        super().__init__()
        # Plain reference, not a registered child -- see rl.deck.DeckNetwork's
        # own comment for why nn.Module.__setattr__ is avoided here.
        object.__setattr__(self, "shared_stack", shared_stack)
        d = shared_stack.d_model
        self.trunk = nn.Sequential(
            nn.Linear(d + self.N_SCALAR, hidden), nn.Tanh(),
            nn.Linear(hidden, hidden), nn.Tanh(),
        )
        self.keep_head = nn.Linear(hidden, 2)     # [keep, mulligan]
        self.value_head = nn.Linear(hidden, 1)    # REINFORCE baseline
        self.bottom_query = nn.Linear(hidden, d)  # scores bottom candidates
        self.d_model = d

    def _encode(self, hand_idx):
        """hand_idx: LongTensor [B, H] (0 = padding). Returns pooled hand
        embedding [B, d] (mean over real cards), masking padding."""
        embs = self.shared_stack.embedding(hand_idx)          # [B, H, d]
        real = (hand_idx != 0).float().unsqueeze(-1)          # [B, H, 1]
        return (embs * real).sum(1) / real.sum(1).clamp(min=1.0)

    def trunk_out(self, hand_idx, scalars):
        pooled = self._encode(hand_idx)
        return self.trunk(torch.cat([pooled, scalars], dim=-1))  # [B, hidden]

    def decision(self, hand_idx, scalars):
        """Returns ([B,2] keep/mull logits, [B] value)."""
        h = self.trunk_out(hand_idx, scalars)
        return self.keep_head(h), self.value_head(h).squeeze(-1)

    def bottom(self, hand_idx, scalars, cand_idx, cand_mask):
        """Score each bottom candidate. cand_idx [B,K] vocab indices, cand_mask
        [B,K] bool (real candidate). Returns ([B,K] masked scores, [B] value)."""
        h = self.trunk_out(hand_idx, scalars)
        query = self.bottom_query(h).unsqueeze(1)              # [B, 1, d]
        cand_embs = self.shared_stack.embedding(cand_idx)      # [B, K, d]
        scores = torch.bmm(query, cand_embs.transpose(1, 2)).squeeze(1) / (self.d_model ** 0.5)
        return scores.masked_fill(~cand_mask, -1e8), self.value_head(h).squeeze(-1)


# --- collection (runs during self-play, in the main process or a worker) ---

def _scalars(state, seat):
    return [min(state.mulligans_taken, HAND) / HAND, 1.0 if state.players[seat].on_the_play else 0.0]


def _log_keep_or_mulligan(state, logits, value, action):
    """Opt-in instrumentation (state.event_log is not None, same gate as
    rl.agent's decision_weights logging) for the keep-vs-mulligan choice.
    Only 2 actions ever exist, so no top-K truncation needed. Skipped
    entirely by the caller when the choice was forced (mull_legal is
    False -- past the mulligan cap, "keep" is the only legal option) --
    nothing informative to log about a decision with one real option."""
    if state.event_log is None:
        return
    probs = torch.softmax(logits, dim=-1)[0].tolist()
    labels = ["Keep", "Mulligan"]
    candidates = [{"index": i, "probability": probs[i], "fixed_label": labels[i], "pointer_identity": None}
                  for i in range(2)]
    state.log_event("decision_weights", network="mulligan_keep", chosen_index=action,
                     value_estimate=float(value.item()), candidates=candidates, pointer_kind=None)


def _log_bottom_pick(state, scores, value, chosen, cands):
    """Same instrumentation for a bottom-card pick -- one candidate per
    unique card name in hand (cands, from game.bottom_options), top-5 by
    probability. Caller skips this entirely when len(cands) <= 1 (forced,
    nothing to log)."""
    if state.event_log is None:
        return
    probs = torch.softmax(scores, dim=-1)[0]
    top = torch.topk(probs, min(5, len(cands)))
    candidates = [{"index": i, "probability": p, "fixed_label": cands[i], "pointer_identity": None}
                  for i, p in zip(top.indices.tolist(), top.values.tolist())]
    state.log_event("decision_weights", network="mulligan_bottom", chosen_index=chosen,
                     value_estimate=float(value.item()), candidates=candidates, pointer_kind=None)


def decide(net, vocab, state, seat, record, greedy=False):
    """Make the pending mulligan-phase decision with `net`, append a plain-data
    transition via record(entry), and return the zero-arg executor that applies
    it. Called by rl.agent.SeatAgent.decide when the pending kind is
    mulligan_decision / mulligan_bottom. Samples during training (exploration);
    greedy=True (argmax) for evaluation. `record` gets a tuple whose reward slot
    is filled in later (finish, at game end); pass a no-op to just evaluate."""
    pend = state.pending_resolution
    hand_idx = [vocab.index(c.name) for c in state.hand]
    scalars = _scalars(state, seat)
    hi = torch.tensor([hand_idx], dtype=torch.long)
    sc = torch.tensor([scalars], dtype=torch.float32)
    with torch.inference_mode():
        if pend["kind"] == "mulligan_decision":
            logits, value = net.decision(hi, sc)
            mull_legal = state.mulligans_taken < HAND
            if not mull_legal:
                logits = logits.clone()
                logits[0, 1] = -1e8  # past the cap only "keep" is legal
            action = int(logits.argmax(-1).item()) if greedy else int(torch.distributions.Categorical(logits=logits).sample().item())
            if mull_legal:
                _log_keep_or_mulligan(state, logits, value, action)
            record(["decision", hand_idx, scalars, mull_legal, action, None])
            if action == 0:
                return lambda: game.execute_mulligan_keep(state)
            return lambda: game.execute_mulligan_take(state)

        # mulligan_bottom: pick one card NAME to put on the bottom
        cands = game.bottom_options(state)  # sorted unique names in hand
        cand_idx = [vocab.index(n) for n in cands]
        ci = torch.tensor([cand_idx], dtype=torch.long)
        cm = torch.ones((1, len(cand_idx)), dtype=torch.bool)
        scores, value = net.bottom(hi, sc, ci, cm)
        k = int(scores.argmax(-1).item()) if greedy else int(torch.distributions.Categorical(logits=scores).sample().item())
        name = cands[k]
        if len(cands) > 1:
            _log_bottom_pick(state, scores, value, k, cands)
        record(["bottom", hand_idx, scalars, cand_idx, k, None])
        return lambda: game.execute_bottom_option(state, name)


# --- REINFORCE update (runs in the main process, after collection) ---

def _pad(rows, fill=0):
    w = max((len(r) for r in rows), default=1) or 1
    return [r + [fill] * (w - len(r)) for r in rows]


def update(net, optimizer, transitions):
    """One REINFORCE step over a batch of transitions (each: [kind, hand_idx,
    scalars, extra, action, reward]). logp and value are RE-computed here from
    the stored observation (on-policy: the net is unchanged since collection this
    round), so nothing stale is stored. Returns a small stats dict; a no-op on an
    empty batch."""
    dec = [t for t in transitions if t[0] == "decision"]
    bot = [t for t in transitions if t[0] == "bottom"]
    if not dec and not bot:
        return {"n": 0}

    device = next(net.parameters()).device
    policy_loss = torch.zeros((), device=device)
    value_loss = torch.zeros((), device=device)
    entropy = torch.zeros((), device=device)
    n = 0

    if dec:
        hand = torch.tensor(_pad([t[1] for t in dec]), dtype=torch.long, device=device)
        sc = torch.tensor([t[2] for t in dec], dtype=torch.float32, device=device)
        logits, value = net.decision(hand, sc)
        for i, t in enumerate(dec):  # re-apply the collect-time legality mask
            if not t[3]:
                logits = logits.clone(); logits[i, 1] = -1e8
        actions = torch.tensor([t[4] for t in dec], dtype=torch.long, device=device)
        rewards = torch.tensor([t[5] for t in dec], dtype=torch.float32, device=device)
        distn = torch.distributions.Categorical(logits=logits)
        logp = distn.log_prob(actions)
        adv = (rewards - value).detach()
        policy_loss = policy_loss - (logp * adv).sum()
        value_loss = value_loss + ((value - rewards) ** 2).sum()
        entropy = entropy + distn.entropy().sum()
        n += len(dec)

    if bot:
        hand = torch.tensor(_pad([t[1] for t in bot]), dtype=torch.long, device=device)
        sc = torch.tensor([t[2] for t in bot], dtype=torch.float32, device=device)
        K = max(len(t[3]) for t in bot)
        cand = torch.tensor([t[3] + [0] * (K - len(t[3])) for t in bot], dtype=torch.long, device=device)
        cmask = torch.tensor([[True] * len(t[3]) + [False] * (K - len(t[3])) for t in bot], dtype=torch.bool, device=device)
        scores, value = net.bottom(hand, sc, cand, cmask)
        actions = torch.tensor([t[4] for t in bot], dtype=torch.long, device=device)
        rewards = torch.tensor([t[5] for t in bot], dtype=torch.float32, device=device)
        distn = torch.distributions.Categorical(logits=scores)
        logp = distn.log_prob(actions)
        adv = (rewards - value).detach()
        policy_loss = policy_loss - (logp * adv).sum()
        value_loss = value_loss + ((value - rewards) ** 2).sum()
        entropy = entropy + distn.entropy().sum()
        n += len(bot)

    loss = (policy_loss + 0.5 * value_loss - ENTROPY_COEF * entropy) / max(1, n)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    return {"n": n, "loss": float(loss.item()),
            "policy_loss": float(policy_loss.item() / max(1, n)),
            "value_loss": float(value_loss.item() / max(1, n))}
