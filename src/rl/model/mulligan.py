"""Deck-specific mulligan model + its REINFORCE trainer.

The mulligan is a near-bandit: one pregame decision (keep/mulligan each
round, plus which cards to bottom on a keep) whose reward is the whole
game's outcome, unlike the main in-game policy whose terminal reward has to
survive ~100 steps of GAE discounting to reach the mulligan choice.

This model owns the pregame phase (rl.decision.agent.SeatAgent routes
mulligan_decision / mulligan_bottom decisions here) and is trained by
REINFORCE with its own reward, decoupled from the main PPO update:

    reward(seat) = WIN_REWARD * (1 if seat won else 0)

No per-mulligan-count penalty. mulligans_taken remains a net input
(_scalars, below).

It reads the same structured, self-attended hand representation the main
policy sees at every in-game decision (rl.model.features.build_token_set's
rows run through this deck's own SetTransformer). See MulliganNet's
docstring for the encoder-sharing mechanics.
"""
from __future__ import annotations

import game
import torch
import torch.nn as nn

from rl.model.arch import pad_token_batch
from rl.model.features import build_token_set

HAND = game.HAND_SIZE_LIMIT  # 7 -- London mulligan cap and hand-size normalizer
WIN_REWARD = 1.0
ENTROPY_COEF = 1.0  # 2026-08-24: 0.01 -- tiny-league's keep/mull head kept re-collapsing
# toward "always keep" even with stratify_0land_pct forcing exposure (entropy fell from ~0.86
# to ~0.35 bits at 0 lands between 20k and 30k games/deck despite the critic's own value spread
# continuing to widen over the same span -- the entropy bonus vanishes as the policy sharpens,
# so mere forced exposure wasn't enough to hold the gain). Tried 3.0 first: within ~2.5k games
# it flooded P(mulligan) to a flat ~50/50 at every land count (entropy pinned at ~1.0 bit
# everywhere) -- too strong, drowning the land-count signal instead of preserving it, confirming
# logs/mulligan_stratify_0p2_entropy3_2k.json's earlier finding on a different deck/setup.
# Dropped to 1.0 to look for a middle ground that holds differentiation without going fully
# uniform -- still experimental, not yet validated at this magnitude on the main league loop.

# Bumped whenever what MulliganNet's trunk reads changes shape-compatibly (a
# shape/param-count mismatch already fails loudly via load_state_dict; a
# same-shape representation change would not). Registered as a buffer so it
# rides state_dict()/load_state_dict() -- an old checkpoint lacking the key
# fails strict loading loudly instead of silently loading weights trained
# against a different input distribution.
HAND_REPR_VERSION = 2


def mulligan_reward(won):
    """This seat's terminal reward for its pregame decisions: WIN_REWARD if
    it won, 0.0 otherwise."""
    return WIN_REWARD if won else 0.0


class MulliganNet(nn.Module):
    """One per deck. Reads its deck's own SetTransformer forward over the
    hand's token set (encode() below), pools to one hand summary, then a
    small trunk feeds three heads: keep/mulligan logits, a value baseline,
    and a pointer query that scores bottom candidates against the encoder's
    post-attention per-card representations (token_reps) -- the same
    pointer mechanism rl.model.deck.DeckNetwork uses for its targeting head.

    The encoder is held as a plain reference, not a registered child, so
    this net's REINFORCE optimizer never trains it -- the main PPO update
    owns it. encode() wraps the encoder forward in torch.no_grad() so
    REINFORCE's backward pass reaches the trunk/heads but stops at the
    encoder. Consequence: PPO moves the encoder's weights between mulligan
    updates, so this net's input representation is non-stationary under it."""

    N_SCALAR = 2  # mulligans_taken/HAND, on_the_play

    def __init__(self, encoder, hidden=64):
        super().__init__()
        # Plain reference, not a registered child -- see class docstring.
        object.__setattr__(self, "encoder", encoder)
        d = encoder.d_model
        self.trunk = nn.Sequential(
            nn.Linear(d + self.N_SCALAR, hidden), nn.Tanh(),
            nn.Linear(hidden, hidden), nn.Tanh(),
        )
        self.keep_head = nn.Linear(hidden, 2)     # [keep, mulligan]
        self.value_head = nn.Linear(hidden, 1)    # REINFORCE baseline
        self.bottom_query = nn.Linear(hidden, d)  # scores bottom candidates
        self.d_model = d
        self.register_buffer("hand_repr_version", torch.tensor(HAND_REPR_VERSION))  # see HAND_REPR_VERSION comment

    def encode(self, vocab_idx, features, key_padding_mask, side_flag):
        """Runs this deck's shared encoder forward, wrapped in no_grad (see
        class docstring). Returns (mine_summary [B,d], token_reps [B,T,d]);
        theirs_summary is discarded -- pregame there is no opponent board/
        hand to summarize."""
        with torch.no_grad():
            mine_summary, _theirs_summary, token_reps = self.encoder(
                vocab_idx, features, key_padding_mask, side_flag)
        return mine_summary, token_reps

    def trunk_out(self, mine_summary, scalars):
        return self.trunk(torch.cat([mine_summary, scalars], dim=-1))  # [B, hidden]

    def decision(self, mine_summary, scalars):
        """Returns ([B,2] keep/mull logits, [B] value)."""
        h = self.trunk_out(mine_summary, scalars)
        return self.keep_head(h), self.value_head(h).squeeze(-1)

    def bottom(self, mine_summary, scalars, cand_reps, cand_mask):
        """Score each bottom candidate. cand_reps [B,K,d] -- each candidate's
        OWN post-attention token representation (gathered from encode()'s
        token_reps, one row per distinct hand card name -- see mulligan.decide/
        update for how the row is picked), cand_mask [B,K] bool (real
        candidate). Returns ([B,K] masked scores, [B] value)."""
        h = self.trunk_out(mine_summary, scalars)
        query = self.bottom_query(h).unsqueeze(1)              # [B, 1, d]
        scores = torch.bmm(query, cand_reps.transpose(1, 2)).squeeze(1) / (self.d_model ** 0.5)
        return scores.masked_fill(~cand_mask, -1e8), self.value_head(h).squeeze(-1)


# --- collection (runs during self-play, in the main process or a worker) ---

def _scalars(state, seat):
    return [min(state.mulligans_taken, HAND) / HAND, 1.0 if state.players[seat].on_the_play else 0.0]


def _log_keep_or_mulligan(state, logits, value, action):
    """Opt-in instrumentation (state.event_log is not None) for the
    keep-vs-mulligan choice. Only 2 actions exist, so no top-K truncation.
    Caller skips this entirely when the choice was forced."""
    if state.event_log is None:
        return
    probs = torch.softmax(logits, dim=-1)[0].tolist()
    labels = ["Keep", "Mulligan"]
    candidates = [{"index": i, "probability": probs[i], "fixed_label": labels[i], "pointer_identity": None}
                  for i in range(2)]
    state.log_event("decision_weights", network="mulligan_keep", chosen_index=action,
                     value_estimate=float(value.item()), candidates=candidates, pointer_kind=None)


def _log_bottom_pick(state, scores, value, chosen, cands):
    """Same instrumentation for a bottom-card pick: one candidate per unique
    card name in hand, top-5 by probability. Skipped when len(cands) <= 1."""
    if state.event_log is None:
        return
    probs = torch.softmax(scores, dim=-1)[0]
    top = torch.topk(probs, min(5, len(cands)))
    candidates = [{"index": i, "probability": p, "fixed_label": cands[i], "pointer_identity": None}
                  for i, p in zip(top.indices.tolist(), top.values.tolist())]
    state.log_event("decision_weights", network="mulligan_bottom", chosen_index=chosen,
                     value_estimate=float(value.item()), candidates=candidates, pointer_kind=None)


def decide(net, vocab, state, seat, record, greedy=False):
    """Make the pending mulligan-phase decision with `net`, append a
    plain-data transition via record(entry), and return the zero-arg
    executor that applies it. Called by rl.decision.agent.SeatAgent.decide
    when the pending kind is mulligan_decision / mulligan_bottom. Samples
    during training; greedy=True (argmax) for evaluation. `record` gets a
    tuple whose reward slot is filled in later at game end; pass a no-op to
    just evaluate.

    Builds the same per-card token set (rl.model.features.build_token_set)
    and runs it through the same encoder forward (net.encode) any in-game
    decision uses -- pregame, every zone but my own hand is empty. tokens
    (not a pooled vector) is what gets recorded, so update()'s on-policy
    replay reruns the real encoder forward against current weights."""
    pend = state.pending_resolution
    tokens = build_token_set(state, seat, vocab)  # pregame: my hand is the only nonempty zone
    scalars = _scalars(state, seat)
    vocab_idx, features, key_padding_mask, _identities = pad_token_batch([tokens])
    side_flag = features[:, :, -1]
    sc = torch.tensor([scalars], dtype=torch.float32)
    with torch.inference_mode():
        mine_summary, token_reps = net.encode(vocab_idx, features, key_padding_mask, side_flag)
        if pend["kind"] == "mulligan_decision":
            logits, value = net.decision(mine_summary, sc)
            mull_legal = state.mulligans_taken < HAND
            if not mull_legal:
                logits = logits.clone()
                logits[0, 1] = -1e8  # past the cap only "keep" is legal
            action = int(logits.argmax(-1).item()) if greedy else int(torch.distributions.Categorical(logits=logits).sample().item())
            if mull_legal:
                _log_keep_or_mulligan(state, logits, value, action)
            record(["decision", tokens, scalars, mull_legal, action, None])
            if action == 0:
                return lambda: game.execute_mulligan_keep(state)
            return lambda: game.execute_mulligan_take(state)

        # mulligan_bottom: pick one card NAME to bottom, scored against its
        # own post-attention token representation (one row per distinct
        # name, the first hand token whose vocab index matches). Duplicate-
        # named hand cards are feature-identical post-attention, so any one
        # of them stands for the whole group.
        cands = game.bottom_options(state)  # sorted unique names in hand
        cand_pos = [next(i for i, t in enumerate(tokens) if t[0] == vocab.index(n)) for n in cands]
        cand_reps = token_reps[:, cand_pos, :]
        cm = torch.ones((1, len(cand_pos)), dtype=torch.bool)
        scores, value = net.bottom(mine_summary, sc, cand_reps, cm)
        k = int(scores.argmax(-1).item()) if greedy else int(torch.distributions.Categorical(logits=scores).sample().item())
        name = cands[k]
        if len(cands) > 1:
            _log_bottom_pick(state, scores, value, k, cands)
        record(["bottom", tokens, scalars, cand_pos, k, None])
        return lambda: game.execute_bottom_option(state, name)


# --- REINFORCE update (runs in the main process, after collection) ---

def update(net, optimizer, transitions):
    """One REINFORCE step over a batch of transitions (each: [kind, tokens,
    scalars, extra, action, reward]). tokens is the exact build_token_set
    output recorded at collection time; pad_token_batch + net.encode rebuild
    mine_summary/token_reps against the current weights (still no-grad past
    the encoder -- see MulliganNet.encode). logp and value are recomputed
    from the stored observation. Returns a small stats dict; a no-op on an
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
        vocab_idx, features, key_padding_mask, _identities = pad_token_batch([t[1] for t in dec], device=device)
        side_flag = features[:, :, -1]
        mine_summary, _token_reps = net.encode(vocab_idx, features, key_padding_mask, side_flag)
        sc = torch.tensor([t[2] for t in dec], dtype=torch.float32, device=device)
        logits, value = net.decision(mine_summary, sc)
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
        vocab_idx, features, key_padding_mask, _identities = pad_token_batch([t[1] for t in bot], device=device)
        side_flag = features[:, :, -1]
        mine_summary, token_reps = net.encode(vocab_idx, features, key_padding_mask, side_flag)
        sc = torch.tensor([t[2] for t in bot], dtype=torch.float32, device=device)
        K = max(len(t[3]) for t in bot)
        d = token_reps.shape[-1]
        # Padded with row-index 0 (always a valid row); cmask blanks its score to -1e8 regardless.
        cand_pos = torch.tensor([t[3] + [0] * (K - len(t[3])) for t in bot], dtype=torch.long, device=device)
        cmask = torch.tensor([[True] * len(t[3]) + [False] * (K - len(t[3])) for t in bot], dtype=torch.bool, device=device)
        cand_reps = torch.gather(token_reps, 1, cand_pos.unsqueeze(-1).expand(-1, -1, d))
        scores, value = net.bottom(mine_summary, sc, cand_reps, cmask)
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
