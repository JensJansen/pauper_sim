"""Deck-specific mulligan model + its REINFORCE trainer.

The mulligan is a near-bandit: ONE pregame decision (keep/mulligan each round,
plus which cards to bottom on a keep) whose reward is the whole game's outcome.
So its credit assignment is DIRECT -- the game that follows is a black box that
turns "the hand I kept" into a single number -- unlike the main in-game policy,
whose terminal reward has to survive ~100 steps of GAE discounting to reach the
mulligan choice, too diluted a signal to train a mulligan decision through (see
rl.rewards's deploy_reward docstring).

This model OWNS the pregame phase (rl.decision.agent.SeatAgent routes mulligan_decision /
mulligan_bottom decisions here instead of the main net) and is trained by
REINFORCE with its OWN reward, decoupled from the main PPO update:

    reward(seat) = WIN_REWARD * (1 if seat won else 0)

REMOVED (2026-08-21) the convex per-mulligan-count penalty this reward used to
subtract: a controlled ablation (cost=0.02 vs cost=0, no stratification) had
already shown byte-identical outcomes, but that run never learned a working
policy either way, so it didn't distinguish "the penalty doesn't matter" from
"nothing mattered yet, penalty included." Once stratify_0land_pct/twin=
MulliganZeroLands/enough games (~16-25k) produced a genuinely working, decisive
0-land-discriminating policy, the SAME reward-shape question resurfaced with a
real policy to test it against: attaching a checkpoint trained under cost=0
into league training (whose rl.model.mulligan module default was still cost=0.02)
would silently change the reward function AND the loaded Adam optimizer's
momentum out from under it the moment training resumed. Rather than leave that
mismatch in place, the penalty is gone from the reward entirely, not just
zeroed -- see mulligan_reward. mulligans_taken remains a NET INPUT (_scalars,
below) so the model still observes how many mulligans it has already taken
this game; only the REWARD no longer penalizes taking them.

It reads the SAME structured, self-attended hand representation the main policy
sees at every in-game decision -- rl.model.features.build_token_set's full per-card
rows (mana production/colors, card type, ...) run through this deck's own
SetTransformer -- not a hand-rolled summary. Before 2026-08-20 it only mean-
pooled bare card-identity embeddings, which carries no card-type signal at all;
that net could not tell a land from a spell except by whatever incidentally
survived averaging, and manual review + a log audit confirmed it: 0-land hands
kept 50% of the time (0/8 wins), 1-land hands kept 94% of the time, both at
>=90% confidence, greedy. See MulliganNet's own docstring for the encoder-
sharing mechanics. Card indices come from rl.model.features.CardVocab.
"""
from __future__ import annotations

import game
import torch
import torch.nn as nn

from rl.model.arch import pad_token_batch
from rl.model.features import build_token_set

HAND = game.HAND_SIZE_LIMIT  # 7 -- London mulligan cap and hand-size normalizer
WIN_REWARD = 1.0
ENTROPY_COEF = 0.01  # keep some exploration so the keep/mull head doesn't re-collapse

# Bumped whenever what MulliganNet's trunk actually reads changes shape-
# compatibly (a plain shape/param-count mismatch already fails loudly via
# load_state_dict; a same-shape REPRESENTATION change would not). Registered
# as a buffer below so it rides state_dict()/load_state_dict() for free --
# an old checkpoint simply lacks the key and strict loading raises "Missing
# key(s)" instead of silently loading pre-fix weights into a net that now
# feeds them a completely different input distribution. 1 = mean-pooled bare
# card-identity embeddings (no card-type signal at all); 2 = this deck's own
# SetTransformer forward over full per-card structured features (2026-08-20).
HAND_REPR_VERSION = 2


def mulligan_reward(won):
    """This seat's terminal reward for its pregame decisions: WIN_REWARD if it
    won, 0.0 otherwise. No per-mulligan penalty (removed 2026-08-21 -- see this
    module's own docstring for why); a mulligan is only ever discouraged by its
    own effect on win probability, never by a flat count-based cost."""
    return WIN_REWARD if won else 0.0


class MulliganNet(nn.Module):
    """One per deck. Reads its deck's own SetTransformer forward over the
    hand's full structured token set (rl.model.features.build_token_set -- the SAME
    per-card rows and self-attention the main policy sees at every in-game
    decision, run via encode() below), pools to ONE hand summary, then a
    small trunk feeds three heads: keep/mulligan logits, a value baseline,
    and a pointer query that scores bottom candidates against the encoder's
    own post-attention per-card representations (token_reps) -- the same
    pointer mechanism rl.model.deck.DeckNetwork uses for its own targeting head,
    just reused here for "which card to bottom" instead of "which permanent
    to target".

    The encoder is held as a PLAIN REFERENCE, not a registered child, so this
    net's own REINFORCE optimizer never trains it -- the main PPO update owns
    it (rl.model.deck.DeckNetwork registers it). That is a deliberate asymmetry: a
    mulligan decision is one near-bandit sample per game, far too little
    signal to be steering a 117k-parameter perception encoder that ~100
    in-game decisions per game are also steering. encode() below is the ONE
    place that boundary is enforced now that this net's forward pass actually
    runs the encoder: wrapped in torch.no_grad() so REINFORCE's backward pass
    can reach the trunk/heads but stops dead at the encoder, exactly as if it
    were still a detached embedding lookup.

    Known consequence, unchanged from the embedding-only design this
    replaced (accepted 2026-08-17, still true 2026-08-20): PPO moves the
    encoder's weights between mulligan updates, so this net's input
    representation is non-stationary under it. The mulligan trunk retrains
    against the drift, and the drift is slow relative to how long a mulligan
    policy takes to converge."""

    N_SCALAR = 2  # mulligans_taken/HAND, on_the_play

    def __init__(self, encoder, hidden=64):
        super().__init__()
        # Plain reference, NOT a registered child -- see the class docstring:
        # keeps the encoder out of this net's optimizer and out of its
        # state_dict (the DeckNetwork checkpoint already carries it).
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
        # See HAND_REPR_VERSION's own comment: a plain int buffer so an old
        # (pre-2026-08-20) checkpoint fails load_state_dict loudly -- its
        # saved state_dict simply has no "hand_repr_version" key -- instead of
        # silently loading shape-compatible weights that were trained against
        # a completely different input distribution.
        self.register_buffer("hand_repr_version", torch.tensor(HAND_REPR_VERSION))

    def encode(self, vocab_idx, features, key_padding_mask, side_flag):
        """Runs this deck's shared encoder forward -- see the class docstring
        for why this is wrapped in no_grad rather than left to the caller to
        remember. Returns (mine_summary [B,d], token_reps [B,T,d]); theirs_
        summary is discarded -- pregame there is no opponent board/hand to
        summarize, so it carries nothing this net's trunk (no FiLM layer)
        would use anyway."""
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
    """Opt-in instrumentation (state.event_log is not None, same gate as
    rl.decision.agent's decision_weights logging) for the keep-vs-mulligan choice.
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
    it. Called by rl.decision.agent.SeatAgent.decide when the pending kind is
    mulligan_decision / mulligan_bottom. Samples during training (exploration);
    greedy=True (argmax) for evaluation. `record` gets a tuple whose reward slot
    is filled in later (finish, at game end); pass a no-op to just evaluate.

    Builds the SAME per-card token set (rl.model.features.build_token_set) and runs
    it through the SAME encoder forward (net.encode) any in-game decision
    uses -- pregame, every zone but my own hand is empty, so this is exactly
    "my hand, fully featured" with no separate code path. tokens (not
    hand_idx) is what gets recorded, so update()'s on-policy replay reruns
    the real encoder forward against the current weights instead of trusting
    a stale pooled vector."""
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

        # mulligan_bottom: pick one card NAME to put on the bottom, scored
        # against ITS OWN post-attention token representation (token_reps),
        # one row per distinct name (game.bottom_options) -- the first hand
        # token whose vocab index matches. Duplicate-named hand cards feed
        # the identical row into self-attention over the identical token
        # set, so they come out feature-identical post-attention too; any
        # one of them stands for the whole group, same as the pre-fix code's
        # one-embedding-per-name shortcut.
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
    mine_summary/token_reps from it here, against the CURRENT weights (still
    no-grad past the encoder boundary -- see MulliganNet.encode). logp and
    value are RE-computed here from the stored observation (on-policy: the
    net is unchanged since collection this round), so nothing stale is
    stored. Returns a small stats dict; a no-op on an empty batch."""
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
        # Padded with row-index 0 -- always a VALID (if masked-out) row, same
        # convention the pre-fix vocab-index padding used; cmask blanks its
        # score to -1e8 regardless of what real card ends up gathered there.
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
