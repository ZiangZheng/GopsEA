from __future__ import annotations

import gymnasium as gym
import os


def _register_once(env_id: str, entry_point: str, kwargs: dict, max_episode_steps=None) -> None:
    if env_id in gym.envs.registry:
        return
    register_kwargs = {
        "id": env_id,
        "entry_point": entry_point,
        "kwargs": kwargs,
        "disable_env_checker": True,
    }
    if max_episode_steps is not None:
        register_kwargs["max_episode_steps"] = max_episode_steps
    gym.register(**register_kwargs)


def create_original_env(env_id: str, **kwargs):
    """Bridge GOPS original env registry to gymnasium.make."""
    from gops_tasks.original.create_pkg.create_env import create_env

    # Convert old gym API envs to gymnasium API by default.
    return create_env(env_id=env_id, gym2gymnasium=True, **kwargs)


def register_original_envs() -> int:
    """Register all old GOPS original envs into gymnasium registry."""
    count = 0
    # Preferred path: use original GOPS registry if available.
    try:
        from gops_tasks.original.create_pkg.create_env import registry as original_registry
    except Exception:
        original_registry = {}

    for env_id in sorted(original_registry.keys()):
        _register_once(
            env_id=env_id,
            entry_point="gops_tasks.gym:create_original_env",
            kwargs={"env_id": env_id},
        )
        if env_id in gym.envs.registry:
            count += 1

    # Fallback path: manual registration from env_gym python files.
    # This keeps legacy gym_* ids available even when `gym` dependency
    # required by original.create_pkg is not installed.
    env_gym_dir = os.path.join(
        os.path.dirname(__file__), "..", "original", "env", "env_gym"
    )
    env_gym_dir = os.path.abspath(env_gym_dir)
    if os.path.isdir(env_gym_dir):
        for file in sorted(os.listdir(env_gym_dir)):
            if not file.endswith(".py") or file.startswith("_"):
                continue
            env_id = file[:-3]
            _register_once(
                env_id=env_id,
                entry_point=f"gops_tasks.original.env.env_gym.{env_id}:env_creator",
                kwargs={},
            )
            if env_id in gym.envs.registry:
                count += 1
    return count


def register_third_party_envs() -> None:
    """Trigger registration side effects for third-party env packages."""
    # IsaacLab task registrations.
    try:
        import gops_tasks.isaaclab.locomotion  # noqa: F401
    except Exception:
        pass

    # DeepMind Control task registrations.
    try:
        import gops_tasks.dm_control  # noqa: F401
    except Exception:
        pass

    # Humanoid-bench task registrations.
    try:
        import gops_tasks.humanoid_bench  # noqa: F401
    except Exception:
        pass


# One-pass global registration.
register_third_party_envs()
register_original_envs()

