import gymnasium as gym
import numpy as np
import torch

from .vec_env import VecEnv


class SimpleGymVecEnv(VecEnv):
    """Minimal wrapper for gym.make("CartPole-v1") to GopsEA VecEnv."""

    def __init__(self, env: gym.Env, device="cpu"):
        self.env = env
        self.device = torch.device(device)

        # ---- fixed for single-env ----
        self.num_envs = 1

        # ---- spaces ----
        self.num_obs = gym.spaces.flatdim(env.observation_space)
        self.num_actions = gym.spaces.flatdim(env.action_space)

        self.num_privileged_obs = 0
        self.max_episode_length = env.spec.max_episode_steps

        self.episode_length_buf = torch.zeros(1, device=self.device, dtype=torch.long)

        self.reset()

    # ------------------------------------------------------------------ #
    # Core API
    # ------------------------------------------------------------------ #

    def reset(self):
        obs, _ = self.env.reset()
        self.episode_length_buf.zero_()

        obs = torch.tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)
        return obs, {"observations": obs}

    def step(self, actions: torch.Tensor):
        # CartPole action is discrete → int
        action = int(actions[0].item())

        obs, reward, terminated, truncated, _ = self.env.step(action)

        done = terminated or truncated

        self.episode_length_buf += 1
        if done:
            self.episode_length_buf.zero_()

        obs = torch.tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)
        reward = torch.tensor([reward], dtype=torch.float32, device=self.device)
        done = torch.tensor([done], dtype=torch.long, device=self.device)

        extras = {
            "observations": obs,
            "termination": torch.tensor([terminated], device=self.device),
            "truncated": torch.tensor([truncated], device=self.device),
            "timeout": torch.tensor([truncated], device=self.device),
        }

        return obs, reward, done, extras

    def seed(self, seed: int = -1):
        self.env.reset(seed=seed)
        return seed

    def close(self):
        self.env.close()

    @property
    def dim_params(self):
        dim_params = {
            "policy_dim": self.observation_space["policy"].shape[-1],
            "critic_dim": self.observation_space.get(
                "critic", self.observation_space["policy"]
            ).shape[-1],
            "dynamic_dim": self.observation_space.get(
                "dynamic", self.observation_space["policy"]
            ).shape[-1],
            "action_dim": self.action_space.shape[-1],
            "rewards_dim": self.rewards_shape,
        }
        return dim_params
    
class SimpleGymWrapper:
    def __init__(self, env: gym.Env, device="cpu"):
        self.env = env
        self.num_envs = 1
        self.device = device
        self.observation_space = env.observation_space
        self.action_space = env.action_space
        self._last_obs = None
    def reset(self):
        obs, _ = self.env.reset()
        obs = torch.as_tensor(obs, dtype=torch.float32, device=self.device)
        self._last_obs = obs
        return obs, {"observations": {}}
    def get_observations(self):
        if self._last_obs is None: obs, _ = self.reset()
        else: obs = self._last_obs
        return obs, {"observations": {}}
    def step(self, actions: torch.Tensor):
        actions_np = actions.detach().cpu().numpy()
        obs, reward, terminated, truncated, _ = self.env.step(actions_np)
        self.done_flag = done_flag = terminated or truncated
        obs_tensor = torch.as_tensor(obs, dtype=torch.float32, device=self.device) if not self.done_flag else self.reset()[0]
        reward = torch.as_tensor(reward, dtype=torch.float32, device=self.device).view(1)
        done = torch.as_tensor(done_flag, dtype=torch.long, device=self.device).view(1)
        extras = {
            "td_observations": {
                "policy": torch.as_tensor(obs, dtype=torch.float32, device=self.device)
            },
            "observations": {},
            "termination": torch.as_tensor(terminated, device=self.device).view(1),
            "timeout": torch.as_tensor(truncated, device=self.device).view(1),
        }
        self._last_obs = obs_tensor
        return obs_tensor, reward, done, extras
    
    def post_step(self):
        if self.done_flag:
            return self.reset()
        return self._last_obs, {}
    
    @property
    def dim_params(self):
        return {
            "policy_dim": self.observation_space.shape[-1],
            "critic_dim": self.observation_space.shape[-1],
            "action_dim": self.action_space.shape[-1],
        }