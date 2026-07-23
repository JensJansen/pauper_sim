"""Lean, CleanRL-style masked PPO -- a drop-in `model_cls` alternative to
sb3_contrib.MaskablePPO for TrainingHarness/run.py (see docs/
GPU_VECENV_INVESTIGATION.md's training-speed followup). Implements just
enough of MaskablePPO's surface (constructor shape, .predict/.learn/.save/
.load/.get_env/.num_timesteps) to be swapped in via a config's own
"model_cls": "lean_ppo" key -- harness.py's self-play wiring (set_own_model/
set_opponent_model/the SubprocVecEnv snapshot-sync path) needs zero changes,
since it only ever calls that same surface, never anything MaskablePPO-
specific.

The one deliberate difference: masking is applied by setting illegal logits
to a large negative value before a plain torch.distributions.Categorical,
instead of sb3-contrib's own MaskableCategorical/apply_masking/logsumexp
stack -- profiled at ~12% of total training wall-clock on its own, and
measured (a throwaway single-env prototype, same env/hyperparameters) at a
real ~26.6% throughput win end to end. Everything else (env-stepping,
observation building, action-table masking, two-player self-play) is
identical code, reused directly from drl_env.py/game/ -- this file only
replaces the RL algorithm's own implementation.

Hyperparameter defaults match sb3_contrib.MaskablePPO's own actual defaults
(verified against the installed package): lr=3e-4, n_steps=2048, batch_
size=64, n_epochs=10, gamma=0.99, gae_lambda=0.95, clip_range=0.2,
ent_coef=0.0, vf_coef=0.5, max_grad_norm=0.5.
"""

import json
import os
import random
import shutil
import tempfile
from datetime import datetime

import gymnasium
import numpy as np
import torch
import torch.nn as nn
from gymnasium import spaces
from sb3_contrib.common.maskable.utils import get_action_masks
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecEnv

import game
import drl_env


class ActorCritic(nn.Module):
    """Two independent MLPs (actor -> n_actions logits, critic -> scalar
    value) sized from net_arch -- not a shared trunk, matching SB3's own
    net_arch convention (a flat list gives pi/vf each their own separate
    network of that shape) for a fair apples-to-apples forward-pass cost
    comparison. nn.Tanh -- SB3 MlpPolicy's own default activation."""

    def __init__(self, obs_dim, n_actions, net_arch=(64, 64)):
        super().__init__()
        self.actor = self._mlp(obs_dim, net_arch, n_actions)
        self.critic = self._mlp(obs_dim, net_arch, 1)

    @staticmethod
    def _mlp(in_dim, hidden, out_dim):
        layers = []
        prev = in_dim
        for size in hidden:
            layers += [nn.Linear(prev, size), nn.Tanh()]
            prev = size
        layers.append(nn.Linear(prev, out_dim))
        return nn.Sequential(*layers)

    def forward(self, obs):
        return self.actor(obs), self.critic(obs).squeeze(-1)


def _ensure_vec_env(env):
    """Mirrors SB3's own BaseAlgorithm._wrap_env: a bare (non-VecEnv) env --
    TrainingHarness.__init__ hands one over at n_envs==1, same as it does
    for real MaskablePPO today -- gets Monitor-wrapped (episode_count()'s
    env_method("get_episode_rewards") depends on this) then DummyVecEnv-
    wrapped, wrapping the SAME instance, not rebuilding a fresh one."""
    if isinstance(env, VecEnv):
        return env
    return DummyVecEnv([lambda: Monitor(env)])


_DEFAULT_HYPERPARAMS = dict(
    learning_rate=3e-4, n_steps=2048, batch_size=64, n_epochs=10, gamma=0.99, gae_lambda=0.95,
    clip_range=0.2, ent_coef=0.0, vf_coef=0.5, max_grad_norm=0.5,
)


class LeanMaskablePPO:
    def __init__(self, policy, env, policy_kwargs=None, device="auto", **model_kwargs):
        self.env = _ensure_vec_env(env)
        self.net_arch = tuple((policy_kwargs or {}).get("net_arch", (64, 64)))
        self.device = self._resolve_device(device)

        hp = dict(_DEFAULT_HYPERPARAMS)
        hp.update({k: v for k, v in model_kwargs.items() if k in hp})
        self.learning_rate = hp["learning_rate"]
        self.n_steps = hp["n_steps"]
        self.batch_size = hp["batch_size"]
        self.n_epochs = hp["n_epochs"]
        self.gamma = hp["gamma"]
        self.gae_lambda = hp["gae_lambda"]
        self.clip_range = hp["clip_range"]
        self.ent_coef = hp["ent_coef"]
        self.vf_coef = hp["vf_coef"]
        self.max_grad_norm = hp["max_grad_norm"]

        self.obs_dim = self.env.observation_space.shape[0]
        self.n_actions = self.env.action_space.n
        self.net = ActorCritic(self.obs_dim, self.n_actions, self.net_arch).to(self.device)
        self.optimizer = torch.optim.Adam(self.net.parameters(), lr=self.learning_rate)

        self.num_timesteps = 0
        self._last_obs = None

    @staticmethod
    def _resolve_device(device):
        if device == "cpu":
            return torch.device("cpu")
        if device == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("device='cuda' requested but torch.cuda.is_available() is False")
        if device in ("auto", "cuda"):
            return torch.device("cuda" if torch.cuda.is_available() else "cpu")
        return torch.device(device)

    def get_env(self):
        return self.env

    # -- inference ---------------------------------------------------------

    def predict(self, obs, action_masks=None, deterministic=False):
        """Same contract as SB3 model.predict(): single-obs in (drl_env.py's
        model_choose_action, TwoPlayerDeckEnv's own_model/opponent_model
        calls) -> (action, None). Masking: illegal logits set to a large
        negative value, then a plain torch.distributions.Categorical --
        the whole point of this class, not sb3-contrib's MaskableCategorical
        stack."""
        with torch.no_grad():
            obs_t = torch.as_tensor(np.asarray(obs), dtype=torch.float32, device=self.device)
            single = obs_t.dim() == 1
            if single:
                obs_t = obs_t.unsqueeze(0)
            logits, _value = self.net(obs_t)
            if action_masks is not None:
                mask_t = torch.as_tensor(np.asarray(action_masks), dtype=torch.bool, device=self.device)
                if single and mask_t.dim() == 1:
                    mask_t = mask_t.unsqueeze(0)
                logits = logits.masked_fill(~mask_t, -1e8)
            if deterministic:
                action = logits.argmax(dim=-1)
            else:
                action = torch.distributions.Categorical(logits=logits).sample()
        action_np = action.cpu().numpy()
        return (action_np[0], None) if single else (action_np, None)

    # -- training ------------------------------------------------------------

    def learn(self, total_timesteps, callback=None, reset_num_timesteps=True):
        """total_timesteps is an upper bound reached in n_steps*n_envs-sized
        increments (may overshoot slightly to the next rollout boundary) --
        same "floor guarantee, may overshoot" contract this codebase's own
        harness.py/train_boggles_mirror_segments.py already use elsewhere,
        and matches SB3's own real .learn() behavior (it doesn't stop
        mid-rollout either). callback unused -- 1-player TrainingHarness.
        train()'s StopTrainingOnMaxEpisodes path isn't exercised by any
        current two-player config; two-player training drives episode
        counting itself via harness.episode_count()."""
        if reset_num_timesteps:
            self.num_timesteps = 0
            self._last_obs = None
        if self._last_obs is None:
            self._last_obs = self.env.reset()
        n_envs = self.env.num_envs
        target = self.num_timesteps + total_timesteps
        while self.num_timesteps < target:
            rollout = self._collect_rollout(self.n_steps)
            self._update(*rollout)
            self.num_timesteps += self.n_steps * n_envs
        return self

    def _collect_rollout(self, n_steps):
        n_envs = self.env.num_envs
        obs_buf = np.zeros((n_steps, n_envs, self.obs_dim), dtype=np.float32)
        act_buf = np.zeros((n_steps, n_envs), dtype=np.int64)
        logp_buf = np.zeros((n_steps, n_envs), dtype=np.float32)
        val_buf = np.zeros((n_steps, n_envs), dtype=np.float32)
        rew_buf = np.zeros((n_steps, n_envs), dtype=np.float32)
        done_buf = np.zeros((n_steps, n_envs), dtype=np.float32)
        mask_buf = np.zeros((n_steps, n_envs, self.n_actions), dtype=bool)

        for t in range(n_steps):
            masks = get_action_masks(self.env)
            with torch.no_grad():
                obs_t = torch.as_tensor(self._last_obs, dtype=torch.float32, device=self.device)
                logits, values = self.net(obs_t)
                mask_t = torch.as_tensor(masks, dtype=torch.bool, device=self.device)
                logits = logits.masked_fill(~mask_t, -1e8)
                dist = torch.distributions.Categorical(logits=logits)
                actions = dist.sample()
                logps = dist.log_prob(actions)

            obs_buf[t] = self._last_obs
            act_buf[t] = actions.cpu().numpy()
            logp_buf[t] = logps.cpu().numpy()
            val_buf[t] = values.cpu().numpy()
            mask_buf[t] = masks

            next_obs, rewards, dones, _infos = self.env.step(actions.cpu().numpy())
            rew_buf[t] = rewards
            done_buf[t] = dones.astype(np.float32)
            self._last_obs = next_obs

        with torch.no_grad():
            obs_t = torch.as_tensor(self._last_obs, dtype=torch.float32, device=self.device)
            _logits, last_values = self.net(obs_t)
        return obs_buf, act_buf, logp_buf, val_buf, rew_buf, done_buf, mask_buf, last_values.cpu().numpy()

    def _update(self, obs_buf, act_buf, old_logp_buf, val_buf, rew_buf, done_buf, mask_buf, last_values):
        n_steps, _n_envs = rew_buf.shape
        adv_buf = np.zeros_like(rew_buf)
        last_gae = np.zeros_like(last_values)
        for t in reversed(range(n_steps)):
            next_value = last_values if t == n_steps - 1 else val_buf[t + 1]
            next_nonterminal = 1.0 - done_buf[t]
            delta = rew_buf[t] + self.gamma * next_value * next_nonterminal - val_buf[t]
            last_gae = delta + self.gamma * self.gae_lambda * next_nonterminal * last_gae
            adv_buf[t] = last_gae
        ret_buf = adv_buf + val_buf

        b_obs = obs_buf.reshape(-1, self.obs_dim)
        b_act = act_buf.reshape(-1)
        b_logp = old_logp_buf.reshape(-1)
        b_adv = adv_buf.reshape(-1)
        b_adv = (b_adv - b_adv.mean()) / (b_adv.std() + 1e-8)
        b_ret = ret_buf.reshape(-1)
        b_mask = mask_buf.reshape(-1, self.n_actions)

        obs_t = torch.as_tensor(b_obs, dtype=torch.float32, device=self.device)
        act_t = torch.as_tensor(b_act, dtype=torch.int64, device=self.device)
        old_logp_t = torch.as_tensor(b_logp, dtype=torch.float32, device=self.device)
        adv_t = torch.as_tensor(b_adv, dtype=torch.float32, device=self.device)
        ret_t = torch.as_tensor(b_ret, dtype=torch.float32, device=self.device)
        mask_t = torch.as_tensor(b_mask, dtype=torch.bool, device=self.device)

        total = len(b_act)
        indices = np.arange(total)
        for _epoch in range(self.n_epochs):
            np.random.shuffle(indices)
            for start in range(0, total, self.batch_size):
                mb = indices[start:start + self.batch_size]
                logits, values = self.net(obs_t[mb])
                logits = logits.masked_fill(~mask_t[mb], -1e8)
                dist = torch.distributions.Categorical(logits=logits)
                new_logp = dist.log_prob(act_t[mb])
                entropy = dist.entropy().mean()

                ratio = torch.exp(new_logp - old_logp_t[mb])
                surr1 = ratio * adv_t[mb]
                surr2 = torch.clamp(ratio, 1 - self.clip_range, 1 + self.clip_range) * adv_t[mb]
                policy_loss = -torch.min(surr1, surr2).mean()
                value_loss = ((values - ret_t[mb]) ** 2).mean()
                loss = policy_loss + self.vf_coef * value_loss - self.ent_coef * entropy

                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.net.parameters(), self.max_grad_norm)
                self.optimizer.step()

    # -- persistence ---------------------------------------------------------

    def save(self, path):
        torch.save({
            "net_state": self.net.state_dict(),
            "obs_dim": self.obs_dim,
            "n_actions": self.n_actions,
            "net_arch": self.net_arch,
            "hyperparams": {
                "learning_rate": self.learning_rate, "n_steps": self.n_steps, "batch_size": self.batch_size,
                "n_epochs": self.n_epochs, "gamma": self.gamma, "gae_lambda": self.gae_lambda,
                "clip_range": self.clip_range, "ent_coef": self.ent_coef, "vf_coef": self.vf_coef,
                "max_grad_norm": self.max_grad_norm,
            },
            "num_timesteps": self.num_timesteps,
        }, path)

    @classmethod
    def load(cls, path, env=None, device="auto", **_ignored):
        """env=None is the inference-only reload shape TwoPlayerDeckEnv.
        reload_own_model/reload_opponent_model already use for SubprocVecEnv
        periodic-snapshot self-play (harness.py's train_two_player) --
        builds just enough (a net, no env/optimizer/rollout machinery) to
        answer .predict() calls."""
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
        if env is not None:
            model = cls(
                "MlpPolicy", env, policy_kwargs={"net_arch": checkpoint["net_arch"]}, device=device,
                **checkpoint["hyperparams"],
            )
        else:
            model = cls.__new__(cls)
            model.env = None
            model.obs_dim = checkpoint["obs_dim"]
            model.n_actions = checkpoint["n_actions"]
            model.net_arch = tuple(checkpoint["net_arch"])
            model.device = cls._resolve_device(device)
            model.net = ActorCritic(model.obs_dim, model.n_actions, model.net_arch).to(model.device)
        model.net.load_state_dict(checkpoint["net_state"])
        model.num_timesteps = checkpoint.get("num_timesteps", 0)
        return model


# ---------------------------------------------------------------------------
# Simultaneous two-player self-play (opt-in "selfplay_mode": "simultaneous"
# in a config's own JSON, dispatched by run.py's _train_two_player) -- BOTH
# sides collect training data from the SAME stream of real games, instead of
# harness.train_two_player's alternating single-buffer bursts (there, only
# whichever side is mid-.learn() ever records a transition; the other side's
# decisions during that burst are pure no_grad() inference -- see harness.
# py's TwoPlayerDeckEnv.step, which only ever builds an observation/reward
# for self.my_seat_idx). Confirmed by a throwaway single-process spike:
# every decision, by either seat, becomes a transition when collected this
# way (100% utilization vs. today's structural ~50%).
#
# Asymmetric decks supported: decklist/reward_fn/terminated_fn/pending_kinds/
# token_card_defs/net_arch are all independent per seat -- same "_2"-suffixed
# per-side convention run.py's own _load_side already uses for the old
# alternating-burst path (a mirror match is just both sides happening to be
# given identical values, not a separate code path). lean_ppo only, not
# MaskablePPO: this needs direct access to ActorCritic/the rollout buffer to
# feed two simultaneous learners from one game stream, a seam SB3's own
# opaque .learn() doesn't expose.
# ---------------------------------------------------------------------------

class _SelfPlayBuffer:
    def __init__(self):
        self.obs, self.act, self.logp, self.val, self.mask, self.rew, self.done = [], [], [], [], [], [], []

    def __len__(self):
        return len(self.obs)

    def add(self, obs, act, logp, val, mask, rew, done):
        self.obs.append(obs); self.act.append(act); self.logp.append(logp)
        self.val.append(val); self.mask.append(mask); self.rew.append(rew); self.done.append(done)

    def extend_from_dict(self, d):
        self.obs.extend(d["obs"]); self.act.extend(d["act"]); self.logp.extend(d["logp"])
        self.val.extend(d["val"]); self.mask.extend(d["mask"]); self.rew.extend(d["rew"]); self.done.extend(d["done"])

    def clear(self):
        self.__init__()


class _SelfPlayWorker(gymnasium.Env):
    """One SubprocVecEnv worker for train_simultaneous_selfplay: plays real
    self-play games entirely inside this worker process, using its own
    LOCAL copies of both sides' nets. collect() is the actual work method
    (called via VecEnv.env_method, the same RPC harness.py's own
    _sync_subproc_snapshots already uses for reload_own_model/reload_
    opponent_model) -- reset()/step() are never called by anything here;
    they only exist because SubprocVecEnv.__init__ unconditionally queries
    observation_space/action_space off a real gymnasium.Env at construction
    time.

    Every constructor arg comes in a (side_a, side_b) pair, index-by-seat
    throughout (self.decklists[seat], self.nets[seat], ...) -- a mirror
    match is just both elements of every pair happening to be equal, not a
    separate code path. decklists/terminated_fns/reward_fns/token_card_defs
    are the real objects (not names) -- the env_fn constructor thunk
    crosses the process boundary via cloudpickle (SB3's own
    CloudpickleWrapper), which handles plain data and module-level function
    references alike; only collect()'s OWN return value has to survive the
    plain-pickle RPC channel back, and that's just (obs, action, logp,
    value, mask, reward, done) tuples -- numbers and numpy arrays, never a
    closure."""

    def __init__(self, decklists, terminated_fns, reward_fns, pending_kinds_list, horizon, token_card_defs_list,
                 on_the_play, net_archs, seed):
        super().__init__()
        self.decklists = decklists
        self.terminated_fns = terminated_fns
        self.reward_fns = reward_fns
        self.horizon = horizon
        self.on_the_play = on_the_play
        self.pending_kinds_list = pending_kinds_list

        # Each side's own action table built with the OTHER side's decklist/
        # token_card_defs as opponent_decklist/opponent_token_card_defs --
        # same shape TrainingHarness.__init__ already builds (self.actions/
        # self.opponent_actions) for the old alternating-burst path.
        self.actions_list = [
            drl_env.build_action_table(
                decklists[seat], game.EFFECT_REGISTRY, token_card_defs=token_card_defs_list[seat],
                pending_kinds=pending_kinds_list[seat], opponent_decklist=decklists[1 - seat],
                opponent_token_card_defs=token_card_defs_list[1 - seat],
            )
            for seat in (0, 1)
        ]
        self.pass_actions = [
            next(i for i, (n, _l, _e) in enumerate(self.actions_list[seat]) if n == "Pass") for seat in (0, 1)
        ]
        self.obs_dims = [
            drl_env.two_player_observation_dim(decklists[seat], pending_kinds_list[seat], decklists[1 - seat])
            for seat in (0, 1)
        ]
        self.n_actions_list = [len(self.actions_list[seat]) for seat in (0, 1)]

        # opp_kw_list[seat]: the aggregate-feature kwargs build_two_player_
        # observation needs to describe seat's OWN opponent (the OTHER
        # side's decklist) -- same _opponent_aggregate_features inputs
        # TrainingHarness.__init__ precomputes for both directions.
        self.opp_kw_list = []
        for seat in (0, 1):
            opp_decklist = decklists[1 - seat]
            card_names, card_copies = drl_env._card_lookup(opp_decklist)
            creature_names, creature_copies = drl_env.creature_names_and_copies(opp_decklist)
            total_cards = sum(qty for _n, qty, *_r in opp_decklist)
            self.opp_kw_list.append(dict(
                opponent_total_cards=total_cards, opponent_creature_names=creature_names,
                opponent_creature_copies=creature_copies, opponent_card_names=card_names,
                opponent_card_copies=card_copies,
            ))

        # Formality only -- SubprocVecEnv.__init__ queries these off worker 0
        # to satisfy the VecEnv constructor contract, but nothing in this
        # collector ever reads vec.observation_space/vec.action_space (every
        # call goes through env_method, not step()/reset()); reporting
        # side 0's own shape is arbitrary and functionally inert.
        self.observation_space = spaces.Box(low=0.0, high=1.0, shape=(self.obs_dims[0],), dtype=np.float32)
        self.action_space = spaces.Discrete(self.n_actions_list[0])

        self.nets = [
            ActorCritic(self.obs_dims[seat], self.n_actions_list[seat], net_archs[seat]) for seat in (0, 1)
        ]
        self.rng = random.Random(seed)

    def sync_weights(self, path_a, path_b):
        """SubprocVecEnv counterpart to a live model reference -- no live
        object crosses a process boundary, so the main process periodically
        saves both sides' current weights and every worker reloads them
        from disk. Same pattern as TwoPlayerDeckEnv.reload_own_model/
        reload_opponent_model."""
        self.nets[0].load_state_dict(torch.load(path_a, map_location="cpu", weights_only=True))
        self.nets[1].load_state_dict(torch.load(path_b, map_location="cpu", weights_only=True))

    def _seat_forward(self, seat, state):
        obs = drl_env.build_two_player_observation(
            state, seat, self.decklists[seat], self.horizon, self.pending_kinds_list[seat],
            **self.opp_kw_list[seat],
        )
        mask = drl_env.legal_action_mask(state, self.actions_list[seat])
        with torch.no_grad():
            obs_t = torch.as_tensor(obs, dtype=torch.float32).unsqueeze(0)
            mask_t = torch.as_tensor(mask, dtype=torch.bool).unsqueeze(0)
            logits, value = self.nets[seat](obs_t)
            logits = logits.masked_fill(~mask_t, -1e8)
            dist = torch.distributions.Categorical(logits=logits)
            action = dist.sample()
            logp = dist.log_prob(action)
        return obs, mask, int(action.item()), float(logp.item()), float(value.item())

    def _reward_for(self, seat, state, done):
        """Mirrors drl_env.TwoPlayerDeckEnv.step()'s own reward formula
        exactly: 0.0 if this seat has lost (drl_env._lost, applied
        unconditionally on every call -- not just at the true end of a
        game, same defensive-on-every-step guard step() itself always
        applies), else reward_fn(state, done, horizon). Deliberately NOT
        harness.finalize_scores: that helper additionally forces every
        score to 0.0 whenever state.turn_won is None (a horizon-capped
        draw, never truly concluded) -- correct for finalize_scores' own
        callers (which score AFTER a game is fully over and want that
        extra "never terminated" gate), but wrong here, since a call with
        done=False is a completely ordinary mid-game reward read, and even
        the done=True call should behave exactly like TwoPlayerDeckEnv.
        step()'s own final step -- trusting reward_fn's own internal
        done-handling, not adding a second independent zeroing rule on
        top of it.

        At done=True, state.active_idx may be pointing at whichever seat
        just played when the game ended (game.run_multiplayer_game's own
        lazy-flip convention), not necessarily THIS seat -- drl_env.
        _for_player flips to this seat's own perspective for the read,
        the same primitive harness.finalize_scores itself uses for the
        identical reason. Not needed at done=False: choose_action's own
        seat = state.active_idx already guarantees this seat IS the
        active one at that moment."""
        if drl_env._lost(state, seat):
            return 0.0
        if done:
            return drl_env._for_player(state, seat, lambda s: self.reward_fns[seat](s, True, self.horizon))
        return self.reward_fns[seat](state, False, self.horizon)

    def collect(self, n_games):
        """Plays n_games real self-play games locally (game.run_multiplayer_
        game -- the same driver harness.evaluate_two_player already uses),
        recording a transition into whichever seat's own buffer made each
        decision, reward attributed at that seat's OWN next decision (state.
        active_idx == seat at the moment it's asked, the same invariant
        drl_env.TwoPlayerDeckEnv's _opponent_choose_action/_own_choose_action
        already rely on) or at the game's true end (_reward_for's own
        done=True branch)."""
        bufs = [_SelfPlayBuffer(), _SelfPlayBuffer()]
        pending = [None, None]

        def choose_action(state):
            seat = state.active_idx
            if pending[seat] is not None:
                reward = self._reward_for(seat, state, False)
                p = pending[seat]
                bufs[seat].add(p["obs"], p["action"], p["logp"], p["value"], p["mask"], reward, False)
                pending[seat] = None
            obs, mask, action, logp, value = self._seat_forward(seat, state)
            pending[seat] = {"obs": obs, "action": action, "logp": logp, "value": value, "mask": mask}
            if action == self.pass_actions[seat]:
                return None
            _name, _legal, execute_fn = self.actions_list[seat][action]
            return lambda state=state, execute_fn=execute_fn: execute_fn(state)

        for _ in range(n_games):
            if self.on_the_play is None:
                starting_idx = self.rng.randint(0, 1)
            else:
                starting_idx = 0 if self.on_the_play else 1
            state = game.run_multiplayer_game(
                decklists=self.decklists, terminated_fns=self.terminated_fns,
                rng=self.rng, starting_player_idx=starting_idx, choose_action=choose_action,
                horizon=self.horizon, combat_enabled=True,
            )
            for seat in (0, 1):
                if pending[seat] is not None:
                    reward = self._reward_for(seat, state, True)
                    p = pending[seat]
                    bufs[seat].add(p["obs"], p["action"], p["logp"], p["value"], p["mask"], reward, True)
                    pending[seat] = None

        return tuple(
            {"obs": b.obs, "act": b.act, "logp": b.logp, "val": b.val, "mask": b.mask, "rew": b.rew, "done": b.done}
            for b in bufs
        )

    def reset(self, *, seed=None, options=None):
        raise NotImplementedError("_SelfPlayWorker only supports collect()/sync_weights() via env_method")

    def step(self, action):
        raise NotImplementedError("_SelfPlayWorker only supports collect()/sync_weights() via env_method")


def _selfplay_gae(rewards_, values_, dones_, gamma, gae_lambda):
    """Safe to run as one flat pass over a buffer stitched together from
    N_WORKERS separate processes' own game streams: every worker's
    collect() call flushes a done=True transition after EVERY game (not
    just its last one), so each worker's own contribution to the buffer
    always ends on a real episode boundary -- exactly what stops this
    reverse pass from ever bootstrapping across into a different worker's
    (or a different game's) trajectory, the same way it already wouldn't
    bootstrap across two ordinary episodes back to back."""
    n = len(rewards_)
    adv = np.zeros(n, dtype=np.float32)
    last_gae = 0.0
    for t in reversed(range(n)):
        next_value = 0.0 if dones_[t] or t + 1 >= n else values_[t + 1]
        next_nonterminal = 0.0 if dones_[t] else 1.0
        delta = rewards_[t] + gamma * next_value * next_nonterminal - values_[t]
        last_gae = delta + gamma * gae_lambda * next_nonterminal * last_gae
        adv[t] = last_gae
    return adv


def _selfplay_ppo_update(net, optimizer, buf, device, n_epochs, batch_size, gamma, gae_lambda, clip_range,
                          ent_coef, vf_coef, max_grad_norm):
    values = np.array(buf.val, dtype=np.float32)
    rewards_ = np.array(buf.rew, dtype=np.float32)
    dones = np.array(buf.done, dtype=np.float32)
    adv = _selfplay_gae(rewards_, values, dones, gamma, gae_lambda)
    ret = adv + values
    adv = (adv - adv.mean()) / (adv.std() + 1e-8)

    obs_t = torch.as_tensor(np.array(buf.obs, dtype=np.float32), device=device)
    act_t = torch.as_tensor(np.array(buf.act, dtype=np.int64), device=device)
    old_logp_t = torch.as_tensor(np.array(buf.logp, dtype=np.float32), device=device)
    adv_t = torch.as_tensor(adv, device=device)
    ret_t = torch.as_tensor(ret, dtype=torch.float32, device=device)
    mask_t = torch.as_tensor(np.array(buf.mask, dtype=bool), device=device)

    total = len(buf)
    indices = np.arange(total)
    for _epoch in range(n_epochs):
        np.random.shuffle(indices)
        for start in range(0, total, batch_size):
            mb = indices[start:start + batch_size]
            logits, values_pred = net(obs_t[mb])
            logits = logits.masked_fill(~mask_t[mb], -1e8)
            dist = torch.distributions.Categorical(logits=logits)
            new_logp = dist.log_prob(act_t[mb])
            entropy = dist.entropy().mean()

            ratio = torch.exp(new_logp - old_logp_t[mb])
            surr1 = ratio * adv_t[mb]
            surr2 = torch.clamp(ratio, 1 - clip_range, 1 + clip_range) * adv_t[mb]
            policy_loss = -torch.min(surr1, surr2).mean()
            value_loss = ((values_pred - ret_t[mb]) ** 2).mean()
            loss = policy_loss + vf_coef * value_loss - ent_coef * entropy

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(net.parameters(), max_grad_norm)
            optimizer.step()


def _save_selfplay_checkpoint(path, net, obs_dim, n_actions, net_arch, hyperparams, num_timesteps):
    """Same dict shape as LeanMaskablePPO.save() -- so the result loads
    through LeanMaskablePPO.load() unmodified, which is what TrainingHarness.
    load() calls under the hood."""
    torch.save({
        "net_state": net.state_dict(), "obs_dim": obs_dim, "n_actions": n_actions, "net_arch": tuple(net_arch),
        "hyperparams": hyperparams, "num_timesteps": num_timesteps,
    }, path)


def train_simultaneous_selfplay(
    decklist_a, decklist_b, terminated_fn_a, terminated_fn_b, reward_fn_a, reward_fn_b, pending_kinds_a,
    pending_kinds_b, horizon, token_card_defs_a=(), token_card_defs_b=(), on_the_play=None,
    net_arch_a=(64, 64), net_arch_b=(64, 64), n_envs=8, vec_env_cls=SubprocVecEnv, seed=0, device="cpu",
    hyperparams_a=None, hyperparams_b=None, total_timesteps=None, max_episodes=None,
    games_per_worker_per_round=1, resume_path_a=None, resume_path_b=None, save_path_a=None, save_path_b=None,
):
    """Two-player self-play with BOTH sides trained from ONE shared stream
    of real games (see this section's own module-level comment for why, vs.
    harness.train_two_player's alternating bursts). Every per-side argument
    (decklist/terminated_fn/reward_fn/pending_kinds/token_card_defs/net_arch/
    hyperparams) is fully independent -- a mirror match is just calling this
    with side_a's values equal to side_b's, not a separate mode. horizon/
    on_the_play/n_envs/vec_env_cls/seed/device are genuinely shared (one
    game, one env-count, one training-execution setup), same "shared vs.
    per-side" split run.py's own load_config already uses.

    hyperparams_a/hyperparams_b: any dict, e.g. a raw config's own
    model_kwargs (policy_kwargs/device/verbose and all) -- every key NOT in
    _DEFAULT_HYPERPARAMS is ignored, same filtering LeanMaskablePPO.__init__
    already applies, so callers never need to pre-filter down to just the
    PPO hyperparams themselves. Each side's own
    n_steps sets ITS OWN buffer-fill threshold (n_steps * n_envs) -- the two
    sides can have different rollout/update cadences if their hyperparams
    differ, same as they can already have different net sizes.

    max_episodes is TOTAL real games played (not per-side, unlike harness.
    train_two_player's own max_episodes) -- since every game feeds both
    sides equally here, "how many games" and "how much data per side" are
    close to the same number (exactly the same in a mirror match; an
    asymmetric match's two sides can still end up with different total
    transition counts if their own decision density per game differs).
    total_timesteps is a per-side safety-cap upper bound, same role as
    harness.train_two_player's own parameter, checked once per round
    alongside max_episodes.

    Returns (net_a, net_b, timesteps_a, timesteps_b, games_played). If
    save_path_a/save_path_b are given, also writes a TrainingHarness.load()-
    compatible models/<config>/agent_a/ and .../agent_b/ (model.zip +
    metadata.json, the same 4 fields TrainingHarness.load()'s mismatch-check
    verifies: reward_fn name, horizon, action_space_size, observation_dim,
    two_player) -- so run.py's existing evaluate_two_player path works on
    the result unmodified, even though training never built a TrainingHarness
    at all."""
    decklists, terminated_fns, reward_fns = [decklist_a, decklist_b], [terminated_fn_a, terminated_fn_b], [
        reward_fn_a, reward_fn_b,
    ]
    pending_kinds_list = [pending_kinds_a, pending_kinds_b]
    token_card_defs_list = [token_card_defs_a, token_card_defs_b]
    net_archs = [tuple(net_arch_a), tuple(net_arch_b)]
    hyperparams_list = [
        {**_DEFAULT_HYPERPARAMS, **{k: v for k, v in (raw or {}).items() if k in _DEFAULT_HYPERPARAMS}}
        for raw in (hyperparams_a, hyperparams_b)
    ]

    obs_dims = [
        drl_env.two_player_observation_dim(decklists[seat], pending_kinds_list[seat], decklists[1 - seat])
        for seat in (0, 1)
    ]
    actions_list = [
        drl_env.build_action_table(
            decklists[seat], game.EFFECT_REGISTRY, token_card_defs=token_card_defs_list[seat],
            pending_kinds=pending_kinds_list[seat], opponent_decklist=decklists[1 - seat],
            opponent_token_card_defs=token_card_defs_list[1 - seat],
        )
        for seat in (0, 1)
    ]
    n_actions_list = [len(actions_list[seat]) for seat in (0, 1)]

    torch.manual_seed(seed)
    nets = [ActorCritic(obs_dims[seat], n_actions_list[seat], net_archs[seat]).to(device) for seat in (0, 1)]
    timesteps = [0, 0]
    for seat, resume_path in enumerate((resume_path_a, resume_path_b)):
        if resume_path is not None:
            checkpoint = torch.load(resume_path, map_location="cpu", weights_only=False)
            saved_shape = (checkpoint["obs_dim"], checkpoint["n_actions"], tuple(checkpoint["net_arch"]))
            current_shape = (obs_dims[seat], n_actions_list[seat], net_archs[seat])
            if saved_shape != current_shape:
                # Fails loudly with the actual mismatch, before ever calling
                # load_state_dict -- otherwise this surfaces as a bare
                # PyTorch RuntimeError ("Error(s) in loading state_dict...")
                # with no indication of WHICH config value changed. Same
                # "fail before any training work starts" spirit as
                # TrainingHarness.load()'s own mismatch-check.
                raise ValueError(
                    f"Cannot resume from {resume_path} -- saved (obs_dim, n_actions, net_arch)="
                    f"{saved_shape} doesn't match this side's current {current_shape}. The decklist, "
                    f"token_card_defs, pending_kinds, or net_arch for this side must have changed since it "
                    f"was last trained here."
                )
            nets[seat].load_state_dict(checkpoint["net_state"])
            timesteps[seat] = checkpoint.get("num_timesteps", 0)
    optimizers = [
        torch.optim.Adam(nets[seat].parameters(), lr=hyperparams_list[seat]["learning_rate"]) for seat in (0, 1)
    ]

    n_steps_per_side = [hyperparams_list[seat]["n_steps"] * n_envs for seat in (0, 1)]

    tmp_dir = tempfile.mkdtemp(prefix="selfplay_sync_")
    path_a, path_b = os.path.join(tmp_dir, "net_a.pt"), os.path.join(tmp_dir, "net_b.pt")

    def _save_weights():
        torch.save(nets[0].state_dict(), path_a)
        torch.save(nets[1].state_dict(), path_b)

    def _make_worker(worker_seed):
        def _init():
            return _SelfPlayWorker(
                decklists, terminated_fns, reward_fns, pending_kinds_list, horizon, token_card_defs_list,
                on_the_play, net_archs, worker_seed,
            )
        return _init

    vec = vec_env_cls([_make_worker(seed + i) for i in range(n_envs)])
    buffers = [_SelfPlayBuffer(), _SelfPlayBuffer()]
    games_played = 0

    try:
        _save_weights()
        vec.env_method("sync_weights", path_a, path_b)

        while True:
            if max_episodes is not None and games_played >= max_episodes:
                break
            if total_timesteps is not None and (timesteps[0] >= total_timesteps or timesteps[1] >= total_timesteps):
                break

            results = vec.env_method("collect", games_per_worker_per_round)
            games_played += n_envs * games_per_worker_per_round
            for worker_result in results:
                for seat in (0, 1):
                    d = worker_result[seat]
                    buffers[seat].extend_from_dict(d)

            updated_this_round = False
            for seat in (0, 1):
                if len(buffers[seat]) >= n_steps_per_side[seat]:
                    hp = hyperparams_list[seat]
                    _selfplay_ppo_update(
                        nets[seat], optimizers[seat], buffers[seat], device, hp["n_epochs"], hp["batch_size"],
                        hp["gamma"], hp["gae_lambda"], hp["clip_range"], hp["ent_coef"], hp["vf_coef"],
                        hp["max_grad_norm"],
                    )
                    timesteps[seat] += len(buffers[seat])
                    buffers[seat].clear()
                    updated_this_round = True

            if updated_this_round:
                _save_weights()
                vec.env_method("sync_weights", path_a, path_b)
    finally:
        vec.close()
        shutil.rmtree(tmp_dir, ignore_errors=True)

    if save_path_a is not None and save_path_b is not None:
        for path, seat in ((save_path_a, 0), (save_path_b, 1)):
            os.makedirs(path, exist_ok=True)
            _save_selfplay_checkpoint(
                os.path.join(path, "model.zip"), nets[seat], obs_dims[seat], n_actions_list[seat],
                net_archs[seat], hyperparams_list[seat], timesteps[seat],
            )
            metadata = {
                "reward_fn": reward_fns[seat].__name__, "model_cls": "LeanMaskablePPO", "model_kwargs": {
                    "policy_kwargs": {"net_arch": list(net_archs[seat])}, "device": device,
                    **hyperparams_list[seat],
                },
                "horizon": horizon, "on_the_play": on_the_play, "action_space_size": n_actions_list[seat],
                "observation_dim": obs_dims[seat], "total_timesteps_trained": timesteps[seat], "train_seed": seed,
                "timestamp": datetime.now().isoformat(), "scoring_fns": [], "two_player": True,
                "my_seat_idx": seat, "shaping_weight": 0.0,
            }
            with open(os.path.join(path, "metadata.json"), "w") as f:
                json.dump(metadata, f, indent=2)

    return nets[0], nets[1], timesteps[0], timesteps[1], games_played


if __name__ == "__main__":
    # ponytail self-check: no pytest in this project, mirrors the
    # assert-based demo convention every other module here uses -- run via
    # `python lean_ppo.py` from src/.
    import sys

    import rewards
    import terminated

    decklist = [("Mountain", 20), ("Lightning Bolt", 10)]
    opponent_decklist = [("Mountain", 20)]
    pending = game.derive_pending_kinds(decklist)
    opponent_pending = game.derive_pending_kinds(opponent_decklist)
    tiny_kwargs = {"policy_kwargs": {"net_arch": [8, 8]}, "device": "cpu", "n_steps": 16, "batch_size": 8, "n_epochs": 2}

    # n_envs=1: bare env auto-wrap path.
    env1 = drl_env.TwoPlayerDeckEnv(
        rewards.strict_binary_reward, decklist=decklist, terminated_fn=lambda s: False, pending_kinds=pending,
        opponent_decklist=opponent_decklist, opponent_terminated_fn=lambda s: False,
        opponent_pending_kinds=opponent_pending, my_seat_idx=0, horizon=20, on_the_play=True, seed=0,
    )
    model1 = LeanMaskablePPO("MlpPolicy", env1, **tiny_kwargs)
    assert isinstance(model1.get_env(), VecEnv) and model1.get_env() is not env1  # bare env got auto-wrapped
    model1.learn(total_timesteps=32, reset_num_timesteps=False)
    assert model1.num_timesteps >= 32
    # episode_count()-style access must work post-auto-wrap.
    assert isinstance(sum(len(r) for r in model1.get_env().env_method("get_episode_rewards")), int)
    print(f"lean_ppo.py n_envs=1 auto-wrap self-check: OK (num_timesteps={model1.num_timesteps})")

    # n_envs>1: real DummyVecEnv, matches how TrainingHarness builds n_envs>1 configs.
    def _make_env(seed):
        def _init():
            return Monitor(drl_env.TwoPlayerDeckEnv(
                rewards.strict_binary_reward, decklist=decklist, terminated_fn=lambda s: False,
                pending_kinds=pending, opponent_decklist=opponent_decklist, opponent_terminated_fn=lambda s: False,
                opponent_pending_kinds=opponent_pending, my_seat_idx=0, horizon=20, on_the_play=True, seed=seed,
            ))
        return _init

    vec_env = DummyVecEnv([_make_env(0), _make_env(1)])
    model2 = LeanMaskablePPO("MlpPolicy", vec_env, **tiny_kwargs)
    assert model2.get_env() is vec_env
    model2.learn(total_timesteps=32, reset_num_timesteps=False)
    assert model2.num_timesteps >= 32

    # Masking correctness: sample many predictions against random masks, every chosen action must be legal.
    rng = np.random.default_rng(0)
    for _ in range(200):
        obs = rng.random(model2.obs_dim).astype(np.float32)
        mask = rng.random(model2.n_actions) > 0.5
        if not mask.any():
            mask[0] = True
        action, _ = model2.predict(obs, action_masks=mask, deterministic=False)
        assert mask[action], "predict() sampled an action outside its own legal mask"
        action_det, _ = model2.predict(obs, action_masks=mask, deterministic=True)
        assert mask[action_det]
    print("lean_ppo.py n_envs>1 + masking-correctness self-check: OK")

    # Save/load round-trip: reloaded net must produce IDENTICAL predictions (same weights, deterministic).
    tmp_dir = tempfile.mkdtemp(prefix="lean_ppo_selfcheck_")
    try:
        save_path = os.path.join(tmp_dir, "model.zip")
        model2.save(save_path)
        loaded_with_env = LeanMaskablePPO.load(save_path, env=vec_env, device="cpu")
        loaded_inference_only = LeanMaskablePPO.load(save_path, env=None, device="cpu")
        obs = rng.random(model2.obs_dim).astype(np.float32)
        mask = np.ones(model2.n_actions, dtype=bool)
        a1, _ = model2.predict(obs, action_masks=mask, deterministic=True)
        a2, _ = loaded_with_env.predict(obs, action_masks=mask, deterministic=True)
        a3, _ = loaded_inference_only.predict(obs, action_masks=mask, deterministic=True)
        assert a1 == a2 == a3, "save/load round-trip changed predictions -- weights didn't survive intact"
        assert loaded_with_env.num_timesteps == model2.num_timesteps
        print(f"lean_ppo.py save/load round-trip self-check: OK (num_timesteps={loaded_with_env.num_timesteps})")
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)
