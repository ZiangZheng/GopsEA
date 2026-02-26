import torch
from .vecenv_wrapper import VecEnvWrapper
from typing import TYPE_CHECKING


class GopsEAEnvWrapper(VecEnvWrapper):
    def __init__(self, env, clip_actions = None):
        super().__init__(env, clip_actions)
        self.rewards_shape = self.get_rewards().shape[-1]
        self.commad_shape = self.get_commands().shape[-1]

    def get_rewards(self):
        return self.unwrapped.reward_manager._step_reward

    def get_commands(self):
        commands = []
        for k, v in self.unwrapped.command_manager._terms.items():
            commands.append(v.command)
        if commands:
            return torch.cat(commands, dim=1)
        else:
            return torch.zeros((self.num_envs, 0),)
    
    def get_observations(self) -> tuple[torch.Tensor, dict]:
        """Returns the current observations of the environment."""
        if hasattr(self.unwrapped, "observation_manager"):
            obs_dict = self.unwrapped.observation_manager.compute()
        else:
            obs_dict = self.unwrapped._get_observations()
        return obs_dict["policy"], {"observations": obs_dict}
    
    def step(self, actions: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, dict]:
        # clip actions
        if self.clip_actions is not None:
            actions = torch.clamp(actions, -self.clip_actions, self.clip_actions)
        # record step information
        obs_dict, rew, terminated, truncated, extras = self.env.step(actions)
        # compute dones for compatibility
        dones = (terminated | truncated).to(dtype=torch.long)
        # move extra observations to the extras dict
        obs = obs_dict["policy"]
        extras["observations"] = obs_dict
        extras["termination"] = terminated
        extras["command"] = self.get_commands()
        # move time out information to the extras dict
        # this is only needed for infinite horizon tasks
        if not self.unwrapped.cfg.is_finite_horizon:
            extras["time_outs"] = truncated
            extras["timeout"] = truncated

        # return the step information
        return obs, rew, dones, extras
    
    @property
    def dim_params(self):
        dim_params = {
            "policy_dim": self.observation_space["policy"].shape[-1],
            "critic_dim": self.observation_space.get("critic", self.observation_space["policy"]).shape[-1],
            "dynamic_dim": self.observation_space.get("dynamic", self.observation_space["policy"]).shape[-1],
            "action_dim": self.action_space.shape[-1],
            "rewards_dim": self.rewards_shape,
        }
        return dim_params