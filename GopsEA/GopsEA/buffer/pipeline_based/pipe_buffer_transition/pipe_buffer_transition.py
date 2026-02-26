from __future__ import annotations
import torch
from GopsEA import configclass
from dataclasses import dataclass
from GopsEA.utils.template.module_base import ModuleBase, ModuleBaseCfg

from GopsEA.buffer.pipeline_based.pipe_buffer_base import PipeBufferBase, PipeBufferBaseCfg


class PipeBufferTransition(PipeBufferBase):
    """
    Minimal transition replay buffer for SAC:
    stores (obs, critic_obs, action, reward, next_obs, next_critic_obs, termination, timeout)
    """

    def __init__(
        self,
        capacity: int,
        obs_dim: int,
        critic_obs_dim: int,
        action_dim: int,
        device: str,
    ):
        self.capacity = capacity
        self.device = device
        self.ptr = 0
        self.size = 0

        self.obs = torch.zeros((capacity, obs_dim), device=device)
        self.critic_obs = torch.zeros((capacity, critic_obs_dim), device=device)
        self.actions = torch.zeros((capacity, action_dim), device=device)
        self.rewards = torch.zeros((capacity, 1), device=device)
        self.next_obs = torch.zeros((capacity, obs_dim), device=device)
        self.next_critic_obs = torch.zeros((capacity, critic_obs_dim), device=device)
        self.termination = torch.zeros((capacity, 1), device=device)
        self.timeout = torch.zeros((capacity, 1), device=device)

    @torch.no_grad()
    def add(
        self,
        obs,
        critic_obs,
        action,
        reward,
        next_obs,
        next_critic_obs,
        termination,
        timeout,
    ):
        n = obs.shape[0]  # num envs

        idx = (torch.arange(n, device=self.device) + self.ptr) % self.capacity

        self.obs[idx] = obs
        self.critic_obs[idx] = critic_obs
        self.actions[idx] = action
        self.rewards[idx] = reward.unsqueeze(-1)
        self.next_obs[idx] = next_obs
        self.next_critic_obs[idx] = next_critic_obs
        self.termination[idx] = termination.unsqueeze(-1).float()
        self.timeout[idx] = timeout.unsqueeze(-1).float()

        self.ptr = (self.ptr + n) % self.capacity
        self.size = min(self.size + n, self.capacity)

    @dataclass
    class TransitionData:
        obs             : torch.Tensor
        critic_obs      : torch.Tensor
        actions         : torch.Tensor
        rewards         : torch.Tensor
        next_obs        : torch.Tensor
        next_critic_obs : torch.Tensor
        termination     : torch.Tensor
        timeout         : torch.Tensor
        
        def to_dict(self):
            return dict(
                obs=self.obs,
                critic_obs=self.critic_obs,
                actions=self.actions,
                rewards=self.rewards,
                next_obs=self.next_obs,
                next_critic_obs=self.next_critic_obs,
                termination=self.termination,
                timeout=self.timeout,
            )

    def sample(self, batch_size: int):
        assert self.size > 0
        idx = torch.randint(0, self.size, (batch_size,), device=self.device)

        return self.TransitionData(
            obs=self.obs[idx],
            critic_obs=self.critic_obs[idx],
            actions=self.actions[idx],
            rewards=self.rewards[idx],
            next_obs=self.next_obs[idx],
            next_critic_obs=self.next_critic_obs[idx],
            termination=self.termination[idx],
            timeout=self.timeout[idx],
        )

    def mini_batch_generator(
        self,
        num_epochs: int,
        batch_size: int,
        max_batches_per_epoch: int = None,
        **kwargs
    ):
        assert self.size > 0, "ReplayBuffer is empty; call add() first."

        for _ in range(num_epochs):
            perm = torch.randperm(self.size, device=self.device)
            batches_yielded = 0

            for start in range(0, self.size, batch_size):
                if max_batches_per_epoch is not None and batches_yielded >= max_batches_per_epoch:
                    break

                ids = perm[start:start + batch_size]

                batch = self.TransitionData(
                    obs=self.obs[ids],
                    critic_obs=self.critic_obs[ids],
                    actions=self.actions[ids],
                    rewards=self.rewards[ids],
                    next_obs=self.next_obs[ids],
                    next_critic_obs=self.next_critic_obs[ids],
                    termination=self.termination[ids],
                    timeout=self.timeout[ids],
                )

                batches_yielded += 1
                yield batch

@configclass
class PipeBufferTransitionCfg(ModuleBaseCfg):
    class_type: type[PipeBufferTransition] = PipeBufferTransition
    capacity: int = 4096 * 128
    
    def construct_from_cfg(self, *args, env_dim_params:dict=None, device=None, **kwargs):
        return PipeBufferTransition(
            self.capacity,
            obs_dim=env_dim_params["policy_dim"],
            critic_obs_dim=env_dim_params["critic_dim"],
            action_dim=env_dim_params["action_dim"],
            device=device
        )