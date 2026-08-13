"""One-off: verify WHERE rl.rewards.deploy_reward_v3's dense mana-burn charge
actually lands in the PPO buffer, relative to the "Tap X" actions that cause
it. Uses the REAL production reward function and the REAL collect_rollout
pending-reward bookkeeping (rl.train), not a re-derivation -- runs a few real
dmir_terror games with record=True and reads the resulting RolloutBuffer's
own (action, reward) sequence directly.

Hypothesis under test (see rl.train.collect_rollout's choose_action closure):
reward for the action pending at seat 0's decision point k is computed via
reward_fn(state, ...) where `state` is whatever seat 0's OWN NEXT decision
point k+1 sees -- and game.turn._empty_mana_pools (which actually increments
mana_burnt_this_turn_single_pip, the dense penalty's real input) only fires
at a PHASE BOUNDARY, not immediately after each tap. Since a seat keeps
priority after any non-Pass action (only Pass hands it to the other player),
a whole phase's worth of "Tap" actions happen back-to-back with the burn
counter still unchanged between them -- so EVERY tap's own reward should be
~0 for the dense term, and the entire phase's accumulated burn charge should
land on whichever action is pending when the phase-boundary-crossing next
decision point arrives (typically that seat's own final Pass ending the
phase). This script checks that directly against real reward values.

Usage: python analysis/check_credit_assignment.py [--games N] [--seed N]
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # src/, for `repo_paths` / `rl.*` -- these live one level up now that this script sits in analysis/
import argparse
import random

from repo_paths import CHECKPOINTS_DIR
from rl.pool import build_pool
from rl.train import _constant_pairing, collect_rollout
from rl.league_runner import HORIZON, load_frozen_stack, D_MODEL
from rl.rewards import deploy_reward_v3
from rl.agent import SeatAgent
from rl import checkpoint as ckpt_io
from rl.deck import DeckNetwork
from rl.mulligan import MulliganNet


def _load_deck(league_dir, name, shared, fixed_tables):
    net = DeckNetwork(shared, film_condition_dim=D_MODEL, non_targeting_n_actions=len(fixed_tables[name]))
    ckpt_io.load_deck_checkpoint(f"{league_dir}/{name}/live.pt", net)
    net.eval()
    mnet = MulliganNet(shared)
    ckpt_io.load_deck_checkpoint(f"{league_dir}/{name}/mulligan.pt", mnet)
    mnet.eval()
    return net, mnet


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--opponent", default="rakdos_madness")
    p.add_argument("--games", type=int, default=5)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    decklists, vocab, deck_ctxs, fixed_tables = build_pool()
    shared = load_frozen_stack(vocab.size)
    rng = random.Random(args.seed)

    net, mnet = _load_deck(CHECKPOINTS_DIR / "4_deck_subleague_test", "dmir_terror", shared, fixed_tables)
    onet, omnet = _load_deck(CHECKPOINTS_DIR / "4_deck_subleague_gauntlet", args.opponent, shared, fixed_tables)
    fixed_table = fixed_tables["dmir_terror"]
    n_fixed = len(fixed_table)

    def label(action_idx):
        return fixed_table[action_idx][0] if action_idx < n_fixed else "<pointer/target action>"

    pairing = _constant_pairing(
        [SeatAgent(net, mnet, deck_ctxs["dmir_terror"]), SeatAgent(onet, omnet, deck_ctxs[args.opponent])],
        [decklists["dmir_terror"], decklists[args.opponent]],
        [deploy_reward_v3, None], ["dmir_terror", None],
    )

    buffers_by_deck, _mull, played = collect_rollout(
        pairing, args.games, HORIZON, rng, device="cpu", record=True, greedy=True)
    buf = buffers_by_deck["dmir_terror"]
    print(f"{played} games, {len(buf)} recorded dmir_terror transitions\n")

    nonzero_dense = 0
    tap_rewards, pass_rewards, other_rewards = [], [], []
    for i in range(len(buf)):
        a, r, done = buf.action[i], buf.reward[i], buf.done[i]
        lab = label(a)
        dense_component = r if not done else None  # terminal transitions also carry win/loss -- skip those from the "which action carries the dense charge" tally
        if lab.startswith("Tap"):
            tap_rewards.append(r)
        elif lab == "Pass":
            pass_rewards.append(r)
        else:
            other_rewards.append(r)
        if not done and abs(r) > 1e-9:
            nonzero_dense += 1

    print(f"transitions with a nonzero dense charge (non-terminal): {nonzero_dense}/{len(buf)}")
    print(f"mean reward when the action taken was Tap*:  {sum(tap_rewards)/len(tap_rewards) if tap_rewards else 0:.5f}  (n={len(tap_rewards)}, nonzero={sum(1 for x in tap_rewards if abs(x)>1e-9)})")
    print(f"mean reward when the action taken was Pass:  {sum(pass_rewards)/len(pass_rewards) if pass_rewards else 0:.5f}  (n={len(pass_rewards)}, nonzero={sum(1 for x in pass_rewards if abs(x)>1e-9)})")
    print(f"mean reward when the action was other (cast/target/etc): {sum(other_rewards)/len(other_rewards) if other_rewards else 0:.5f}  (n={len(other_rewards)}, nonzero={sum(1 for x in other_rewards if abs(x)>1e-9)})")

    # Print a concrete window: the first stretch of several consecutive Tap
    # actions followed by a nonzero-reward transition, so the pattern is
    # visible directly, not just averaged away.
    print("\n--- first Tap-run + whatever transition finally carries a nonzero reward ---")
    i = 0
    shown = 0
    while i < len(buf) and shown < 3:
        if label(buf.action[i]).startswith("Tap"):
            start = i
            while i < len(buf) and label(buf.action[i]).startswith("Tap"):
                i += 1
            # print the tap run plus the next few transitions until reward != 0 or done
            j = start
            end = min(i + 6, len(buf))
            while j < end:
                print(f"  [{j}] action={label(buf.action[j]):30s} reward={buf.reward[j]:+.5f} done={buf.done[j]}")
                if abs(buf.reward[j]) > 1e-9 or buf.done[j]:
                    break
                j += 1
            print()
            shown += 1
        else:
            i += 1


if __name__ == "__main__":
    main()
