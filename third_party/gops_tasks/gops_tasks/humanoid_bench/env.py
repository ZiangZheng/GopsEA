from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any

import gymnasium as gym
import numpy as np


def _prepare_humanoid_bench_import() -> None:
    """Best-effort local import support for third_party/humanoid-bench."""
    try:
        import humanoid_bench  # noqa: F401
        return
    except Exception:
        pass

    repo_root = Path(__file__).resolve().parents[4]
    local_pkg_root = repo_root / "third_party" / "humanoid-bench"
    if local_pkg_root.exists():
        local_pkg_root_str = str(local_pkg_root)
        if local_pkg_root_str not in sys.path:
            sys.path.insert(0, local_pkg_root_str)


class HumanoidBenchWrapper(gym.Wrapper):
    """Thin wrapper for stable dtype/action behavior."""

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(np.asarray(action).copy())
        obs = np.asarray(obs, dtype=np.float32)
        reward = float(reward)
        return obs, reward, terminated, truncated, info

    def reset(self, *, seed: int | None = None, options: dict[str, Any] | None = None):
        obs, info = self.env.reset(seed=seed, options=options)
        return np.asarray(obs, dtype=np.float32), info


def make_humanoid_env(
    robot: str,
    task: str,
    control: str | None = None,
    render_mode: str | None = None,
    policy_path: str | None = None,
    mean_path: str | None = None,
    var_path: str | None = None,
    policy_type: str | None = None,
    small_obs: str | bool | None = None,
    GopsEA_entry_point=None,
    **kwargs,
):
    """Create humanoid-bench env through local/external humanoid_bench package."""
    _prepare_humanoid_bench_import()
    import humanoid_bench  # noqa: F401

    # Match upstream defaults.
    if control is None:
        control = "torque" if robot in {"g1", "digit"} else "pos"

    if small_obs is not None:
        small_obs = str(small_obs)

    env_id = f"{robot}-{task}-v0"
    env = gym.make(
        env_id,
        render_mode=render_mode or "rgb_array",
        policy_path=policy_path,
        mean_path=mean_path,
        var_path=var_path,
        policy_type=policy_type,
        small_obs=small_obs,
        control=control,
        **kwargs,
    )
    env = HumanoidBenchWrapper(env)
    env.max_episode_steps = env.get_wrapper_attr("_max_episode_steps")

    # Headless Linux default for mujoco rendering.
    if sys.platform != "darwin" and "MUJOCO_GL" not in os.environ:
        os.environ["MUJOCO_GL"] = "egl"

    return env
