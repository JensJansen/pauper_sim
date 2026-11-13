"""Rollout-collection invariants on real games (plan P2): what collect_rollout records is well-formed, attributed to the
right seat and bucket, and legal under its own stored mask. The update-side replay of the same data is
tests/component/training/test_rollout_update_equivalence.py; the maths of the update is tests/unit/rl/training/test_ppo_math.py.

Rewards use flat_win_loss_reward, so every episode's total is exactly +1 (the winner), -1 (the loser) or -1 (a horizon timeout
scores -1 for BOTH seats: the owner-confirmed contract, 2026-11-12 -- a timeout should never occur, decking is an instant loss,
and a stalled game is never better than a loss)."""
import random

import numpy as np
import pytest
import torch

import game
from rl.decision.action_bridge import build_fixed_action_table
from rl.decision.agent import AlwaysKeep, SeatAgent
from rl.league.league_runner import build_deck_net
from rl.model.features import CardVocab
from rl.rewards import flat_win_loss_reward
from rl.roster import TOKEN_DEFS
from rl.training.train import _constant_pairing, collect_rollout

ENC = {"d_model": 32, "n_heads": 2, "n_layers": 2, "dim_feedforward": 64}


def _run(deck_a, deck_b, record_as, n_games=4, horizon=60, seed=0):
    torch.manual_seed(seed)
    lists = [game.parse_decklist_file(f"../data/{d}.txt") for d in (deck_a, deck_b)]
    vocab = CardVocab(lists, token_card_defs=TOKEN_DEFS)
    same = deck_a == deck_b
    agents = []
    for dl in lists[:1] if same else lists:
        table = build_fixed_action_table(dl, token_card_defs=TOKEN_DEFS)
        net = build_deck_net(vocab.size, len(table), trunk_hidden=(32, 32), encoder=ENC)
        net.eval()
        agents.append(SeatAgent(net, AlwaysKeep(), (vocab, table)))
    if same:
        agents = agents * 2
    winners = []
    pairing = _constant_pairing(agents, lists, [flat_win_loss_reward()] * 2, record_as)
    buffers, _mull, played = collect_rollout(pairing, n_games, horizon, random.Random(seed), device="cpu",
                                             on_game_end=lambda state: winners.append(state.winner))
    return buffers, winners, played


def _episodes(buf):
    """[(reward_sum, rewards, length)] per episode, in recorded order; asserts the buffer never ends mid-episode."""
    out, cur = [], []
    for r, done in zip(buf.reward, buf.done, strict=True):
        cur.append(r)
        if done:
            out.append((sum(cur), cur, len(cur)))
            cur = []
    assert not cur, "the buffer ends mid-episode: the last transition of a game must carry done=True"
    return out


CASES = [("mono_red_madness", "mono_red_madness", ["m", "m"]),
         ("elves", "rakdos_madness", ["elves", "rakdos_madness"])]


@pytest.mark.parametrize("deck_a,deck_b,record_as", CASES)
def test_every_episode_is_closed_and_pays_only_at_the_terminal_step(deck_a, deck_b, record_as):
    buffers, winners, played = _run(deck_a, deck_b, record_as)
    assert played == 4 == len(winners)
    for name, buf in buffers.items():
        assert len(buf) > 0, name
        for total, rewards, _n in _episodes(buf):
            assert set(rewards[:-1]) == {0.0} or len(rewards) == 1, f"{name}: a reward was paid before the last step"
            assert total in (1.0, -1.0), f"{name}: episode total {total}"


@pytest.mark.parametrize("deck_a,deck_b,record_as", CASES)
def test_each_seat_is_paid_for_its_own_result(deck_a, deck_b, record_as):
    """Episodes are appended game by game, seat 0's trajectory then seat 1's, each into its bucket. Winner +1, loser -1,
    a game with no winner -1 for both."""
    buffers, winners, _played = _run(deck_a, deck_b, record_as, n_games=5, seed=1)
    per_bucket = {name: iter(_episodes(buf)) for name, buf in buffers.items()}
    for g, winner in enumerate(winners):
        for seat in (0, 1):
            total, _r, _n = next(per_bucket[record_as[seat]])
            expected = 1.0 if winner == seat else -1.0
            assert total == expected, f"game {g} seat {seat}: winner={winner}, paid {total}"
    assert all(next(it, None) is None for it in per_bucket.values()), "episodes were recorded that no game accounts for"


def test_a_seat_with_no_bucket_is_not_recorded():
    buffers, _w, played = _run("elves", "rakdos_madness", ["elves", None], n_games=4, seed=2)
    assert set(buffers) == {"elves"} and len(_episodes(buffers["elves"])) == played == 4


def test_a_mirror_pools_both_seats_into_one_bucket():
    buffers, _w, played = _run("mono_red_madness", "mono_red_madness", ["m", "m"], n_games=3, seed=3)
    assert set(buffers) == {"m"} and len(_episodes(buffers["m"])) == 2 * played


def test_a_horizon_timeout_scores_minus_one_for_both_seats_and_is_well_formed():
    """Owner-confirmed contract (plan Q3): a cut-off game has no winner and each recorded seat gets -1 on its last step."""
    buffers, winners, _p = _run("mono_red_madness", "mono_red_madness", ["m", "m"], n_games=3, horizon=2, seed=4)
    assert winners == [None] * 3
    eps = _episodes(buffers["m"])
    # at a 2-turn horizon a seat may never have faced a real choice, and then records nothing at all
    assert 3 <= len(eps) <= 6 and all(total == -1.0 for total, _r, _n in eps)


@pytest.mark.parametrize("deck_a,deck_b,record_as", CASES)
def test_every_recorded_action_is_legal_under_its_own_stored_mask(deck_a, deck_b, record_as):
    """The property ppo_update now enforces (plan Q6), checked at the source on real games."""
    buffers, _w, _p = _run(deck_a, deck_b, record_as, n_games=3, seed=5)
    for name, buf in buffers.items():
        for i in range(len(buf)):
            mask = np.asarray(buf.mask[i])
            assert 0 <= buf.action[i] < len(mask) and mask[buf.action[i]], f"{name}[{i}]: action {buf.action[i]} is masked"
            assert mask.any(), f"{name}[{i}]: an all-False mask was recorded"


@pytest.mark.parametrize("deck_a,deck_b,record_as", CASES)
def test_forced_decisions_are_not_recorded(deck_a, deck_b, record_as):
    """A decision with exactly one legal action teaches nothing: collect_rollout leaves it out (ppo_entry is None), so every
    recorded transition had a real choice. A recorded one-legal-action transition would mean the shortcut stopped firing."""
    buffers, _w, _p = _run(deck_a, deck_b, record_as, n_games=3, seed=6)
    for name, buf in buffers.items():
        assert all(np.asarray(m).sum() >= 2 for m in buf.mask), f"{name}: a forced decision was recorded"


@pytest.mark.parametrize("deck_a,deck_b,record_as", CASES)
def test_recorded_numbers_are_finite_and_the_log_probability_is_a_log_probability(deck_a, deck_b, record_as):
    buffers, _w, _p = _run(deck_a, deck_b, record_as, n_games=3, seed=7)
    for name, buf in buffers.items():
        assert np.isfinite(buf.logp).all() and (np.array(buf.logp) <= 1e-6).all(), name
        assert np.isfinite(buf.value).all() and np.isfinite(buf.reward).all(), name
        assert all(np.isfinite(s).all() for s in buf.scalar), name


# ---- league pairing: who is recorded -----------------------------------------------------------------------------------------------

def _league_buckets(tmp_path, checkpoint_rate, n_games=6):
    """collect_rollout_league over a two-deck pool: returns the bucket names recorded and the opponents drawn."""
    from rl.league.league import LeaguePool
    from rl.model.arch import SetTransformer
    from rl.model.deck import DeckNetwork
    from rl.roster import build_pool
    from rl.training.train import collect_rollout_league

    decklists, vocab, deck_ctxs, tables = build_pool()
    names = ["elves", "boggles"]
    torch.manual_seed(0)
    nets = {n: DeckNetwork(SetTransformer(vocab.size, **ENC), non_targeting_n_actions=len(tables[n]), trunk_hidden=(32, 32)).eval()
            for n in names}
    pool = LeaguePool(str(tmp_path), names)
    for n in names:
        pool.register_snapshot(n, nets[n])
    buffers, _m, played, outcomes = collect_rollout_league(
        "elves", nets, None, deck_ctxs, decklists, pool, flat_win_loss_reward(), 40, n_games, random.Random(1),
        device="cpu", checkpoint_rate=checkpoint_rate)
    return buffers, played, outcomes


def test_a_frozen_snapshot_opponent_records_nothing_only_the_training_deck_does(tmp_path):
    buffers, played, outcomes = _league_buckets(tmp_path, checkpoint_rate=1.0)
    assert set(buffers) == {"elves"}, "every opponent was a frozen snapshot, so no other bucket may exist"
    assert all(opp_snapshot is not None for _opp, opp_snapshot, _won, _strat in outcomes)
    assert played == 6


def test_a_live_opponent_deck_records_under_its_own_name(tmp_path):
    buffers, _played, outcomes = _league_buckets(tmp_path, checkpoint_rate=0.0, n_games=10)
    assert all(snap is None for _opp, snap, _won, _strat in outcomes), "checkpoint_rate 0: never a snapshot"
    assert set(buffers) == {"elves", "boggles"}, "a game against the other deck's live net also trains that deck (seeded draw)"
