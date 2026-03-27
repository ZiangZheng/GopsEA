from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass
class TransitionBatch:
    obs: torch.Tensor
    critic_obs: torch.Tensor
    actions: torch.Tensor
    rewards: torch.Tensor
    next_obs: torch.Tensor
    next_critic_obs: torch.Tensor
    termination: torch.Tensor
    timeout: torch.Tensor

    def normalize(self) -> "TransitionBatch":
        rewards = self.rewards if self.rewards.dim() > 1 else self.rewards.unsqueeze(-1)
        termination = self.termination if self.termination.dim() > 1 else self.termination.unsqueeze(-1)
        timeout = self.timeout if self.timeout.dim() > 1 else self.timeout.unsqueeze(-1)
        return TransitionBatch(
            obs=self.obs,
            critic_obs=self.critic_obs,
            actions=self.actions,
            rewards=rewards.float(),
            next_obs=self.next_obs,
            next_critic_obs=self.next_critic_obs,
            termination=termination.float(),
            timeout=timeout.float(),
        )

    def as_dict(self) -> dict[str, torch.Tensor]:
        return {
            "obs": self.obs,
            "critic_obs": self.critic_obs,
            "actions": self.actions,
            "rewards": self.rewards,
            "next_obs": self.next_obs,
            "next_critic_obs": self.next_critic_obs,
            "termination": self.termination,
            "timeout": self.timeout,
        }
