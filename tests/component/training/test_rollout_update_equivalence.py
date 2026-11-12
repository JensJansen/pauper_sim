"""The encoder is run twice per decision: once while playing (batch of 1, inference mode, token list as built) and
again inside ppo_update (episodes padded together into minibatches, grad mode, recurrent state replayed from zero).
PPO is only valid if those two agree: at the moment of the update the new policy equals the old one, so every
importance ratio is exactly 1. ppo_update itself reports the KL it measures; with a zero learning rate the weights
don't move, so that KL is pure replay disagreement."""
import random

import pytest
import torch

import game
from rl.decision.action_bridge import build_fixed_action_table
from rl.decision.agent import AlwaysKeep, SeatAgent
from rl.league.league_runner import build_deck_net
from rl.model.features import CardVocab
from rl.rewards import flat_win_loss_reward
from rl.roster import TOKEN_DEFS
from rl.training.ppo import ppo_update
from rl.training.train import RolloutBuffer, _constant_pairing, collect_rollout

ENC = {"d_model": 32, "n_heads": 2, "n_layers": 2, "dim_feedforward": 64}
TOKENS = TOKEN_DEFS  # every token/pseudo-card the engine can put on a battlefield or the stack


def _play(seed, deck_a, deck_b, n_games=3, horizon=40):
    torch.manual_seed(seed)
    lists = [game.parse_decklist_file(f"../data/{d}.txt") for d in (deck_a, deck_b)]
    vocab = CardVocab(lists, token_card_defs=TOKENS)
    same = deck_a == deck_b
    nets, agents = [], []
    for dl in lists[:1] if same else lists:  # a mirror is ONE network playing both seats (both record into bucket "m")
        table = build_fixed_action_table(dl, token_card_defs=TOKENS)
        net = build_deck_net(vocab.size, len(table), trunk_hidden=(32, 32), encoder=ENC)
        net.eval()
        nets.append(net)
        agents.append(SeatAgent(net, AlwaysKeep(), (vocab, table)))
    if same:
        agents = agents * 2
    pairing = _constant_pairing(agents, lists, [flat_win_loss_reward()] * 2, ["m", "m" if same else None])
    buffers, _mull, _n = collect_rollout(pairing, n_games, horizon=horizon, rng=random.Random(seed), device="cpu")
    return nets[0], buffers["m"]


def _episodes(buf):
    """The buffer split into one sub-buffer per episode (a trajectory ends at done=True)."""
    out, cur = [], RolloutBuffer()
    for i in range(len(buf)):
        cur.add(buf.token_lists[i], buf.scalar[i], buf.mask[i], buf.action[i], buf.logp[i], buf.value[i], buf.reward[i], buf.done[i])
        if buf.done[i]:
            out.append(cur)
            cur = RolloutBuffer()
    assert len(cur) == 0, "buffer ends mid-episode"
    return out


def _worst_kl(net, buf, group):
    """Largest approx_kl over every minibatch of `group` whole episodes. ppo_update reports only its LAST minibatch's
    KL, so each group is run as its own update (one minibatch each), covering every episode."""
    net.train()
    opt = torch.optim.SGD(net.parameters(), lr=0.0)  # weights never move: any KL is replay disagreement
    eps = _episodes(buf)
    worst, clip = 0.0, 0.0
    for i in range(0, len(eps), group):
        sub = RolloutBuffer()
        for e in eps[i:i + group]:
            sub.extend(e)
        out = ppo_update(net, opt, sub, "cpu", n_epochs=1, batch_size=group, target_kl=None)
        worst, clip = max(worst, out[3]), max(clip, out[4])
    return worst, clip


@pytest.mark.parametrize("decks", [("mono_red_madness", "mono_red_madness"), ("elves", "rakdos_madness")])
@pytest.mark.parametrize("group", [1, 2, 64])
def test_replaying_a_rollout_through_the_update_path_reproduces_the_policy(decks, group):
    net, buf = _play(0, *decks)
    assert len(buf) >= 20, f"only {len(buf)} recorded decisions"
    n_fixed = net.non_targeting_head.out_features
    assert any(a >= n_fixed for a in buf.action), "no pointer pick recorded: the previous-pointer memory went untested"
    kl, clip = _worst_kl(net, buf, group)
    assert kl < 1e-6, f"replay disagrees with rollout: approx_kl={kl:.2e}"
    assert clip == 0.0


def test_the_previous_pointer_input_is_part_of_what_the_update_replays(monkeypatch):
    """Negative control for the previous-pointer memory: if the update path stopped feeding it, the replay would no
    longer match the rollout (which does feed it)."""
    import rl.training.ppo as ppo_mod
    net, buf = _play(0, "mono_red_madness", "mono_red_madness")
    real = ppo_mod.previous_pointer_inputs
    monkeypatch.setattr(ppo_mod, "previous_pointer_inputs", lambda v, f, p, seq, n: tuple(torch.zeros_like(t) for t in real(v, f, p, seq, n)))
    kl, _clip = _worst_kl(net, buf, 64)
    assert kl > 1e-5, f"dropping the previous-pointer input went undetected: approx_kl={kl:.2e}"


def test_the_equivalence_measure_is_sensitive():
    """Negative control: nudge the encoder after collection and the same measure must light up."""
    net, buf = _play(1, "mono_red_madness", "mono_red_madness")
    with torch.no_grad():
        for p in net.encoder.parameters():
            p.add_(0.05 * torch.randn_like(p))
    kl, _clip = _worst_kl(net, buf, 64)
    assert kl > 1e-4, f"a perturbed encoder went undetected: approx_kl={kl:.2e}"
