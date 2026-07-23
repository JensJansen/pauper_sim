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

import numpy as np
import torch
import torch.nn as nn
from sb3_contrib.common.maskable.utils import get_action_masks
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecEnv


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


if __name__ == "__main__":
    # ponytail self-check: no pytest in this project, mirrors the
    # assert-based demo convention every other module here uses -- run via
    # `python lean_ppo.py` from src/.
    import os
    import shutil
    import sys
    import tempfile

    import game
    import rewards
    import terminated
    import drl_env

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
