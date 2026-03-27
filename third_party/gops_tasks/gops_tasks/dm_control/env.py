from __future__ import annotations

from collections import OrderedDict
from typing import Any

import gymnasium as gym
import numpy as np

try:
    import dm_env
    from dm_control import suite
    from dm_control.suite.wrappers import action_scale
    from dm_env import specs
except Exception as exc:  # pragma: no cover
    dm_env = None
    suite = None
    action_scale = None
    specs = None
    _IMPORT_ERROR = exc
else:
    _IMPORT_ERROR = None

_DM_ENV_BASE = dm_env.Environment if dm_env is not None else object


class ActionRepeatWrapper(_DM_ENV_BASE):
    def __init__(self, env: dm_env.Environment, num_repeats: int):
        self._env = env
        self._num_repeats = num_repeats

    def step(self, action):
        reward = 0.0
        discount = 1.0
        time_step = None
        for _ in range(self._num_repeats):
            time_step = self._env.step(action)
            reward += (time_step.reward or 0.0) * discount
            discount *= time_step.discount
            if time_step.last():
                break
        return time_step._replace(reward=reward, discount=discount)

    def observation_spec(self):
        return self._env.observation_spec()

    def action_spec(self):
        return self._env.action_spec()

    def reset(self):
        return self._env.reset()

    def __getattr__(self, name):
        return getattr(self._env, name)


class ActionDTypeWrapper(_DM_ENV_BASE):
    def __init__(self, env: dm_env.Environment, dtype):
        self._env = env
        wrapped_action_spec = env.action_spec()
        self._action_spec = specs.BoundedArray(
            wrapped_action_spec.shape,
            dtype,
            wrapped_action_spec.minimum,
            wrapped_action_spec.maximum,
            "action",
        )

    def step(self, action):
        return self._env.step(action.astype(self._env.action_spec().dtype))

    def observation_spec(self):
        return self._env.observation_spec()

    def action_spec(self):
        return self._action_spec

    def reset(self):
        return self._env.reset()

    def __getattr__(self, name):
        return getattr(self._env, name)


class DMCGymEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}

    def __init__(
        self,
        domain_name: str,
        task_name: str,
        seed: int | None = None,
        frame_skip: int = 2,
        max_episode_steps: int = 500,
        render_mode: str | None = None,
        camera_id: int = 0,
        width: int = 640,
        height: int = 480,
        GopsEA_entry_point=None,
    ):
        if _IMPORT_ERROR is not None:
            raise ImportError("dm_control is required for dm_control tasks.") from _IMPORT_ERROR
        if render_mode not in (None, "rgb_array"):
            raise ValueError("Only `rgb_array` render_mode is supported.")

        self.domain_name = domain_name
        self.task_name = task_name
        self._seed = seed
        self._frame_skip = int(frame_skip)
        self.max_episode_steps = int(max_episode_steps)
        self.render_mode = render_mode
        self._camera_id = int(camera_id)
        self._width = int(width)
        self._height = int(height)
        self._t = 0

        self._build_env()
        self.observation_space = gym.spaces.Box(
            low=np.full(self._obs_shape, -np.inf, dtype=np.float32),
            high=np.full(self._obs_shape, np.inf, dtype=np.float32),
            dtype=np.float32,
        )
        self.action_space = gym.spaces.Box(
            low=np.full(self._env.action_spec().shape, -1.0, dtype=np.float32),
            high=np.full(self._env.action_spec().shape, 1.0, dtype=np.float32),
            dtype=np.float32,
        )

    def _build_env(self):
        self._env = suite.load(
            self.domain_name,
            self.task_name,
            task_kwargs={"random": self._seed},
            visualize_reward=False,
        )
        self._env = ActionDTypeWrapper(self._env, np.float32)
        self._env = ActionRepeatWrapper(self._env, self._frame_skip)
        self._env = action_scale.Wrapper(self._env, minimum=-1.0, maximum=1.0)
        self._obs_shape = (
            int(np.sum([np.prod(v.shape) if v.shape else 1 for v in self._env.observation_spec().values()])),
        )

    @staticmethod
    def _flatten_obs(obs: OrderedDict[str, np.ndarray]) -> np.ndarray:
        return np.concatenate([np.asarray(v).reshape(-1) for v in obs.values()]).astype(np.float32)

    def reset(self, *, seed: int | None = None, options: dict[str, Any] | None = None):
        if seed is not None:
            self._seed = seed
            self._build_env()
        self._t = 0
        time_step = self._env.reset()
        return self._flatten_obs(time_step.observation), {}

    def step(self, action):
        self._t += 1
        time_step = self._env.step(np.asarray(action, dtype=np.float32))
        terminated = bool(time_step.last())
        truncated = self._t >= self.max_episode_steps
        return (
            self._flatten_obs(time_step.observation),
            float(time_step.reward or 0.0),
            terminated,
            truncated,
            {},
        )

    def render(self):
        if self.render_mode != "rgb_array":
            return None
        camera_id = 2 if self.domain_name == "quadruped" else self._camera_id
        return self._env.physics.render(self._height, self._width, camera_id)

    def close(self):
        return None
