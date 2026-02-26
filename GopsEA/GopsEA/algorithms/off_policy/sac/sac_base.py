from __future__ import annotations
import copy
import torch
import torch.nn as nn
import torch.optim as optim
from typing import Dict, Generator

from GopsEA import configclass
from dataclasses import MISSING

from GopsEA.utils.template.module_base import ModuleBase, ModuleBaseCfg
from GopsEA.components.actor import SACActor
from GopsEA.components.critic import MultiQNetwork
from GopsEA.networks.optimizer import GroupedOptimizerCfg, GroupedOptimizer
from GopsEA.algorithms.algorithm_base import AlgorithmBase, AlgorithmBaseCfg

# -----------------------------------------------------
# Flatten (T-1, B) -> (N,)
# -----------------------------------------------------
def flatten(x):
    return x.reshape(-1, *x.shape[2:])

class SAC(AlgorithmBase):
    """
    Soft Actor-Critic (off-policy, update-once)

    - Explicit asymmetric observations:
        * policy_obs  -> actor
        * critic_obs  -> critics
    - Actor / critics / target critics are built internally
    - Replay buffer and batching are handled externally
    """
    cfg: "SACCfg"
    actor: SACActor
    critic: MultiQNetwork

    def __init__(
        self,
        cfg: SACCfg,
        actor: SACActor,
        critic: MultiQNetwork,
        device: str,
    ):
        super().__init__()
        self.cfg = cfg
        self.device = device
        
        self.actor = actor.to(device)
        self.critics = critic.to(device)
        self.target_critics = copy.deepcopy(self.critics).to(device)
        for p in self.target_critics.parameters():
            p.requires_grad_(False)

        # =====================================================
        # Optimizers
        # =====================================================
        self.actor_optimizer = optim.Adam(
            self.actor.parameters(), lr=cfg.actor_lr
        )
        self.critic_optimizer = optim.Adam(
            self.critics.parameters(), lr=cfg.critic_lr
        )
        # Only for save and loss
        self.optimizer = GroupedOptimizer(
            {
                "actor": self.actor_optimizer,
                "critic": self.critic_optimizer
            },
            self.cfg.optimzers_cfg
        )

        # =====================================================
        # Entropy temperature (alpha)
        # =====================================================
        self.auto_entropy = cfg.auto_entropy
        if self.auto_entropy:
            self.target_entropy = cfg.target_entropy if cfg.target_entropy is not None else -float(actor.action_dim)
            self.log_alpha = torch.zeros(1, requires_grad=True, device=device)
            self.alpha = self.log_alpha.exp()
            self.alpha_optimizer = optim.Adam(
                [self.log_alpha], lr=cfg.alpha_lr
            )
        else:
            self.alpha = torch.tensor(cfg.alpha, device=device)

    def act(self, obs, critic_obs):
        return self.actor.sample(obs)[0]

    # =====================================================
    # Soft update of target critics
    # =====================================================
    @torch.no_grad()
    def _soft_update(self):
        tau = self.cfg.tau
        for p, tp in zip(self.critics.parameters(), self.target_critics.parameters()):
            tp.data.mul_(1.0 - tau)
            tp.data.add_(tau * p.data)

    def update(self, generator: Generator[Dict[str, torch.Tensor], None, None]):
        self.ptr_update = 0
        critic_losses, q_means, target_q_means, actor_losses, alpha_losses, alphas = [], [], [], [], [], []
        for minib in generator:
            params_dict: Dict[str, torch.Tensor] = minib
            critic_loss, q_mean, target_q_mean, actor_loss, alpha_loss, alpha = self.update_seq(
                **params_dict,
            )
            critic_losses.append(critic_loss)
            q_means.append(q_mean)
            target_q_means.append(target_q_mean)
            if actor_loss is not None: actor_losses.append(actor_loss)
            if alpha_loss is not None: alpha_losses.append(alpha_loss)
            alphas.append(alpha)
            self.ptr_update += 1
        return {
            "critic_loss": sum(critic_losses) / len(critic_losses),
            "actor_loss": sum(actor_losses) / len(actor_losses),
            "q_mean": sum(q_means) / len(q_means),
            "target_q_mean": sum(target_q_means) / len(target_q_means),
            "alpha_loss": sum(alpha_losses) / len(alpha_losses),
            "alpha": sum(alphas) / len(alphas),
            "mini_batch_num": self.ptr_update
        }

    def update_seq(
        self,
        obs: torch.Tensor,
        critic_obs: torch.Tensor,
        action: torch.Tensor,
        reward: torch.Tensor,
        termination: torch.Tensor,
        timeout: torch.Tensor,
    ) -> Dict[str, float]:
        """
        Sequence SAC update.
        Inputs are full trajectories [B, T, D].
        TD backup is constructed via time shift inside the sequence.
        """
        obs = obs.transpose(0, 1)
        critic_obs = critic_obs.transpose(0, 1)
        action = action.transpose(0, 1)
        reward = reward.transpose(0, 1)
        termination = termination.transpose(0, 1)
        timeout = timeout.transpose(0, 1)

        return self.update_trans(
            obs             = flatten(obs[:-1]),
            critic_obs      = flatten(critic_obs[:-1]),
            actions         = flatten(action[:-1]),
            rewards         = flatten(reward[:-1]),
            next_obs        = flatten(obs[1:]),
            next_critic_obs = flatten(critic_obs[1:]),
            termination     = flatten(termination[:-1]),
            timeout         = flatten(timeout[:-1])
        )

    def update_trans(
        self,
        obs: torch.Tensor,
        critic_obs: torch.Tensor,
        actions: torch.Tensor,
        rewards: torch.Tensor,
        next_obs: torch.Tensor,
        next_critic_obs: torch.Tensor,
        termination: torch.Tensor,
        timeout: torch.Tensor,   # termination OR timeout
    ) -> Dict[str, float]:

        critic_loss, q_mean, target_q_mean \
            = self.update_critic(obs, critic_obs, actions, rewards, next_obs, next_critic_obs, termination, timeout)

        if self._is_update_actor:
            actor_loss, alpha_loss = self.update_actor(obs, critic_obs)
        else:
            actor_loss, alpha_loss = None, None


        if self._is_target_update:
            self._soft_update()

        return critic_loss, q_mean, target_q_mean, actor_loss, alpha_loss, self.alpha.item()

        # return {
        #     "critic_loss": float(critic_loss.item()),
        #     "actor_loss": float(actor_loss.item()),
        #     "alpha_loss": 0.0 if alpha_loss is None else float(alpha_loss.item()),
        #     "alpha": float(self.alpha.item()),
        # }

    def update_critic(
        self,
        obs: torch.Tensor,
        critic_obs: torch.Tensor,
        actions: torch.Tensor,
        rewards: torch.Tensor,
        next_obs: torch.Tensor,
        next_critic_obs: torch.Tensor,
        termination: torch.Tensor,
        timeout: torch.Tensor,   # termination OR timeout
    ) -> Dict[str, float]:
        gamma = self.cfg.gamma

        # dones = (timeout + termination > 0.8).float()
        bootstrap_mask = (1.0 - termination)
        
        # -----------------------------------------------------
        # 1. Critic update
        # -----------------------------------------------------
        with torch.no_grad():
            next_actions, next_log_prob = self.actor.sample(next_obs)
            target_q_all = self.target_critics(next_critic_obs, next_actions)
            target_q_min = torch.min(target_q_all, dim=0).values

            target_q = rewards + gamma * bootstrap_mask * (
                target_q_min - self.alpha * next_log_prob
            )
            
            # target_q = torch.clamp(target_q, -100.0, 100.0)

        current_q_all: torch.Tensor = self.critics(critic_obs, actions)  # (Q, B)
        loss = 0.0
        for q in current_q_all:
            loss += (q - target_q).pow(2).mean()
        critic_loss = loss

        self.critic_optimizer.zero_grad(set_to_none=True)
        critic_loss.backward()
        nn.utils.clip_grad_norm_(self.critics.parameters(), self.cfg.max_grad_norm)
        self.critic_optimizer.step()
        
        return float(critic_loss.item()), float(current_q_all.mean()), float(target_q.mean())

    @property
    def _is_update_actor(self):
        return (self.ptr_update % self.cfg.actor_update_freq) == 0

    @property
    def _is_target_update(self):
        return (self.ptr_update % self.cfg.target_update_freq) == 0

    def update_actor(
        self,
        obs: torch.Tensor,
        critic_obs: torch.Tensor,
    ) -> Dict[str, float]:
        new_actions, log_prob = self.actor.sample(obs)
        q_new_all = self.critics(critic_obs, new_actions)
        q_new_min = torch.min(q_new_all, dim=0).values
        alpha_reg = self.alpha.detach() * log_prob
        actor_loss = (alpha_reg - q_new_min).mean()

        self.actor_optimizer.zero_grad(set_to_none=True)
        actor_loss.backward()
        nn.utils.clip_grad_norm_(
            self.actor.parameters(), self.cfg.max_grad_norm
        )
        self.actor_optimizer.step()

        alpha_loss = None
        if self.auto_entropy:
            alpha_loss = -(
                self.log_alpha * (log_prob + self.target_entropy).detach()
            ).mean()

            self.alpha_optimizer.zero_grad(set_to_none=True)
            alpha_loss.backward()
            self.alpha_optimizer.step()
            self.alpha = self.log_alpha.exp()
        return (
            float(actor_loss.item()),
            0.0 if alpha_loss is None else float(alpha_loss.item())
        )

@configclass
class SACCfg(AlgorithmBaseCfg):
    """
    SAC configuration (algorithm + network hyperparameters only).
    Observation / action dimensions are provided at runtime.
    """

    class_type: type[SAC] = SAC
    optimzers_cfg: GroupedOptimizerCfg = GroupedOptimizerCfg()
    # -------------------------
    # SAC hyperparameters
    # -------------------------

    gamma               : float = 0.99
    tau                 : float = 0.005

    actor_lr            : float = 3e-4
    critic_lr           : float = 3e-4
    alpha_lr            : float = 3e-4
    actor_update_freq   : int = 4
    target_update_freq  : int = 4

    auto_entropy        : bool = True
    target_entropy      : float | None = None  # None -> -action_dim
    alpha               : float = 0.2

    max_grad_norm       : float = 1.0
    
