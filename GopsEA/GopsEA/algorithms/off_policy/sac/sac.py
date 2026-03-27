from __future__ import annotations

import copy
from typing import Dict, Generator, Union

import torch
import torch.nn as nn
import torch.optim as optim

from GopsEA import configclass
from GopsEA.algorithms.algorithm_base import AlgorithmBase, AlgorithmBaseCfg
from GopsEA.components.actor import SACActor
from GopsEA.components.critic import MultiQNetwork
from GopsEA.networks.optimizer import GroupedOptimizerCfg, GroupedOptimizer


class SAC(AlgorithmBase):
    cfg: "SACCfg"
    actor: SACActor
    critic: MultiQNetwork

    def __init__(self, cfg: "SACCfg", actor: SACActor, critic: MultiQNetwork, device: str):
        super().__init__()
        self.cfg = cfg
        self.device = device

        self.actor = actor.to(device)
        self.critics = critic.to(device)
        self.target_critics = copy.deepcopy(self.critics).to(device)
        for p in self.target_critics.parameters():
            p.requires_grad_(False)

        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=cfg.actor_lr)
        self.critic_optimizer = optim.Adam(self.critics.parameters(), lr=cfg.critic_lr)
        self.optimizer = GroupedOptimizer(
            {"actor": self.actor_optimizer, "critic": self.critic_optimizer},
            self.cfg.optimzers_cfg,
        )

        self.auto_entropy = cfg.auto_entropy
        if self.auto_entropy:
            self.target_entropy = (
                cfg.target_entropy if cfg.target_entropy is not None else -float(actor.action_dim)
            )
            self.log_alpha = torch.zeros(1, requires_grad=True, device=device)
            self.alpha = self.log_alpha.exp()
            self.alpha_optimizer = optim.Adam([self.log_alpha], lr=cfg.alpha_lr)
        else:
            self.alpha = torch.tensor(cfg.alpha, device=device)

    def act(self, obs, critic_obs):
        return self.actor.sample(obs)[0]

    @staticmethod
    def _flatten_time_batch(x: torch.Tensor) -> torch.Tensor:
        # [T, B, ...] -> [T*B, ...]
        return x.reshape(-1, *x.shape[2:])

    def _seq_to_trans_payload(self, minib: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        obs = minib["obs"].transpose(0, 1)
        critic_obs = minib["critic_obs"].transpose(0, 1)
        action = minib["action"].transpose(0, 1)
        reward = minib["reward"].transpose(0, 1)
        termination = minib["termination"].transpose(0, 1)
        timeout = minib["timeout"].transpose(0, 1)
        return {
            "obs": self._flatten_time_batch(obs[:-1]),
            "critic_obs": self._flatten_time_batch(critic_obs[:-1]),
            "actions": self._flatten_time_batch(action[:-1]),
            "rewards": self._flatten_time_batch(reward[:-1]),
            "next_obs": self._flatten_time_batch(obs[1:]),
            "next_critic_obs": self._flatten_time_batch(critic_obs[1:]),
            "termination": self._flatten_time_batch(termination[:-1]),
            "timeout": self._flatten_time_batch(timeout[:-1]),
        }

    def _normalize_payload(self, minib: Union[dict, object]) -> Dict[str, torch.Tensor]:
        if isinstance(minib, dict):
            payload = minib
        elif hasattr(minib, "to_dict"):
            payload = minib.to_dict()
        else:
            raise NotImplementedError("SAC received unsupported replay batch type.")

        # Transition-style payload: already has next_* tensors.
        if "next_obs" in payload and "next_critic_obs" in payload:
            return payload
        # Sequence-style payload: convert to transition-style.
        if "action" in payload and "reward" in payload:
            return self._seq_to_trans_payload(payload)
        raise KeyError("SAC batch missing required keys for trans/seq update.")

    def _update_at_trans(
        self,
        obs: torch.Tensor,
        critic_obs: torch.Tensor,
        actions: torch.Tensor,
        rewards: torch.Tensor,
        next_obs: torch.Tensor,
        next_critic_obs: torch.Tensor,
        termination: torch.Tensor,
        timeout: torch.Tensor,
    ):
        critic_loss, q_mean, target_q_mean = self._update_critic(
            obs, critic_obs, actions, rewards, next_obs, next_critic_obs, termination, timeout
        )
        if self._is_update_actor:
            actor_loss, alpha_loss, entropy = self._update_actor(obs, critic_obs)
        else:
            actor_loss, alpha_loss, entropy = None, None, None
        if self._is_target_update:
            self._soft_update()
        return critic_loss, q_mean, target_q_mean, actor_loss, alpha_loss, self.alpha.item(), entropy

    def update(self, generator: Generator[Union[dict, object], None, None]):
        self.ptr_update = 0
        critic_losses, q_means, target_q_means, actor_losses, alpha_losses, alphas, entropies = (
            [],
            [],
            [],
            [],
            [],
            [],
            [],
        )
        for minib in generator:
            payload = self._normalize_payload(minib)
            critic_loss, q_mean, target_q_mean, actor_loss, alpha_loss, alpha, entropy = self._update_at_trans(
                **payload
            )
            critic_losses.append(critic_loss)
            q_means.append(q_mean)
            target_q_means.append(target_q_mean)
            if actor_loss is not None:
                actor_losses.append(actor_loss)
            if alpha_loss is not None:
                alpha_losses.append(alpha_loss)
            if entropy is not None:
                entropies.append(entropy)
            alphas.append(alpha)
            self.ptr_update += 1

        return {
            "critic_loss": sum(critic_losses) / len(critic_losses),
            "actor_loss": sum(actor_losses) / len(actor_losses) if actor_losses else 0.0,
            "q_mean": sum(q_means) / len(q_means),
            "target_q_mean": sum(target_q_means) / len(target_q_means),
            "alpha_loss": sum(alpha_losses) / len(alpha_losses) if alpha_losses else 0.0,
            "alpha": sum(alphas) / len(alphas),
            "entropy": sum(entropies) / len(entropies) if entropies else 0.0,
            "mini_batch_num": self.ptr_update,
        }

    @torch.no_grad()
    def _soft_update(self):
        tau = self.cfg.tau
        for p, tp in zip(self.critics.parameters(), self.target_critics.parameters()):
            tp.data.mul_(1.0 - tau)
            tp.data.add_(tau * p.data)

    def _update_critic(
        self,
        obs: torch.Tensor,
        critic_obs: torch.Tensor,
        actions: torch.Tensor,
        rewards: torch.Tensor,
        next_obs: torch.Tensor,
        next_critic_obs: torch.Tensor,
        termination: torch.Tensor,
        timeout: torch.Tensor,
    ):
        gamma = self.cfg.gamma
        bootstrap_mask = 1.0 - termination.float()
        with torch.no_grad():
            next_actions, next_log_prob = self.actor.sample(next_obs)
            target_q_all = self.target_critics(next_critic_obs, next_actions)
            target_q_min = torch.min(target_q_all, dim=0).values
            target_q = rewards + gamma * bootstrap_mask * (target_q_min - self.alpha * next_log_prob)

        current_q_all: torch.Tensor = self.critics(critic_obs, actions)
        critic_loss = 0.0
        for q in current_q_all:
            critic_loss += (q - target_q).pow(2).mean()

        self.critic_optimizer.zero_grad(set_to_none=True)
        critic_loss.backward()
        nn.utils.clip_grad_norm_(self.critics.parameters(), self.cfg.max_grad_norm)
        self.critic_optimizer.step()

        return float(critic_loss.item()), float(current_q_all.mean()), float(target_q.mean())

    def _update_actor(self, obs: torch.Tensor, critic_obs: torch.Tensor):
        new_actions, log_prob = self.actor.sample(obs)
        q_new_all = self.critics(critic_obs, new_actions)
        q_new_min = torch.min(q_new_all, dim=0).values
        alpha_reg = self.alpha.detach() * log_prob
        actor_loss = (alpha_reg - q_new_min).mean()

        self.actor_optimizer.zero_grad(set_to_none=True)
        actor_loss.backward()
        nn.utils.clip_grad_norm_(self.actor.parameters(), self.cfg.max_grad_norm)
        self.actor_optimizer.step()

        alpha_loss = None
        if self.auto_entropy:
            alpha_loss = -(self.log_alpha * (log_prob + self.target_entropy).detach()).mean()
            self.alpha_optimizer.zero_grad(set_to_none=True)
            alpha_loss.backward()
            self.alpha_optimizer.step()
            self.alpha = self.log_alpha.exp()

        entropy = float(-log_prob.mean().item())
        return float(actor_loss.item()), 0.0 if alpha_loss is None else float(alpha_loss.item()), entropy

    @property
    def _is_update_actor(self):
        return (self.ptr_update % self.cfg.actor_update_freq) == 0

    @property
    def _is_target_update(self):
        return (self.ptr_update % self.cfg.target_update_freq) == 0


@configclass
class SACCfg(AlgorithmBaseCfg):
    class_type: type[SAC] = SAC
    optimzers_cfg: GroupedOptimizerCfg = GroupedOptimizerCfg()

    gamma: float = 0.99
    tau: float = 0.005

    actor_lr: float = 3e-4
    critic_lr: float = 3e-4
    alpha_lr: float = 3e-4
    actor_update_freq: int = 4
    target_update_freq: int = 4

    auto_entropy: bool = True
    target_entropy: float | None = None
    alpha: float = 0.2

    max_grad_norm: float = 1.0
