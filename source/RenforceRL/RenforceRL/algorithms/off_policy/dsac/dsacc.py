from __future__ import annotations

import torch
from RenforceRL import configclass
from typing import Generator, Dict, Tuple, Union
from RenforceRL.buffer import PipeBufferTransition

import copy
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Normal
from typing import Dict, Generator

from RenforceRL import configclass
from dataclasses import MISSING

from RenforceRL.utils.template.module_base import ModuleBase, ModuleBaseCfg
from RenforceRL.components.actor import SACActor
from RenforceRL.components.critic import GaussianQNetwork
from RenforceRL.networks.optimizer import GroupedOptimizerCfg, GroupedOptimizer
from RenforceRL.algorithms.algorithm_base import AlgorithmBase, AlgorithmBaseCfg

# -----------------------------------------------------
# Flatten (T-1, B) -> (N,)
# -----------------------------------------------------
def flatten(x):
    return x.reshape(-1, *x.shape[2:])

class DSAC(AlgorithmBase):
    """
    Distributional Soft Actor-Critic (off-policy, update-once)

    - Explicit asymmetric observations:
        * policy_obs  -> actor
        * critic_obs  -> critic
    - Actor / critic / target critic are built internally
    - Replay buffer and batching are handled externally
    """
    cfg: "DSACCfg"
    actor: SACActor
    critic: GaussianQNetwork

    def __init__(
        self,
        cfg: DSACCfg,
        actor: SACActor,
        critic: GaussianQNetwork,
        device: str,
    ):
        super().__init__()
        self.cfg = cfg
        self.device = device
        
        self.actor = actor.to(device)
        self.critic = critic.to(device)
        self.target_critic = copy.deepcopy(self.critic).to(device)
        self.target_actor = copy.deepcopy(self.actor).to(device)
        for p in self.target_critic.parameters():
            p.requires_grad_(False)
        for p in self.target_actor.parameters():
            p.requires_grad_(False)

        # =====================================================
        # Optimizers
        # =====================================================
        self.actor_optimizer = optim.Adam(
            self.actor.parameters(), lr=cfg.actor_lr, betas=(0.9, 0.999)
        )
        self.critic_optimizer = optim.Adam(
            self.critic.parameters(), lr=cfg.critic_lr, betas=(0.9, 0.999)
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
                [self.log_alpha], lr=cfg.alpha_lr, betas=(0.9, 0.999)
            )
        else:
            self.alpha = torch.tensor(cfg.alpha, device=device)

    def act(self, obs, critic_obs):
        return self.actor.sample(obs)[0]

    # =====================================================
    # Soft update of target critic
    # =====================================================
    @torch.no_grad()
    def _soft_update(self):
        tau = self.cfg.tau
        for p, tp in zip(self.critic.parameters(), self.target_critic.parameters()):
            tp.data.mul_(1.0 - tau)
            tp.data.add_(tau * p.data)
        for p, tp in zip(self.actor.parameters(), self.target_actor.parameters()):
            tp.data.mul_(1.0 - tau)
            tp.data.add_(tau * p.data)

    def update(self, generator: Generator[Union[PipeBufferTransition.TransitionData, dict], None, None]):
        self.ptr_update = 0
        critic_losses, q_mean_means, target_q_means, actor_losses, alpha_losses, alphas, entropies = [], [], [], [], [], [], []
        for minib in generator:
            if isinstance(minib, dict):
                critic_loss, q_mean_mean, target_q_mean, actor_loss, alpha_loss, alpha, entropy = self.update_trans(**minib)
            else:
                raise NotImplementedError("DSAC get bad replay buffer.")
            critic_losses.append(critic_loss)
            q_mean_means.append(q_mean_mean)
            target_q_means.append(target_q_mean)
            if actor_loss is not None: actor_losses.append(actor_loss)
            if alpha_loss is not None: alpha_losses.append(alpha_loss)
            alphas.append(alpha)
            if entropy is not None: entropies.append(entropy)
            self.ptr_update += 1
        return {
            "critic_loss": sum(critic_losses) / len(critic_losses),
            "actor_loss": sum(actor_losses) / len(actor_losses) if actor_losses else 0.0,
            "q_mean": sum(q_mean_means) / len(q_mean_means),
            "target_q_mean": sum(target_q_means) / len(target_q_means),
            "alpha_loss": sum(alpha_losses) / len(alpha_losses) if alpha_losses else 0.0,
            "alpha": sum(alphas) / len(alphas),
            "entropy": sum(entropies) / len(entropies) if entropies else 0.0,
            "mini_batch_num": self.ptr_update
        }

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

        critic_loss, q_mean_mean, target_q_mean \
            = self.update_critic(obs, critic_obs, actions, rewards, next_obs, next_critic_obs, termination, timeout)

        if self._is_update_actor:
            actor_loss, alpha_loss, entropy = self.update_actor(obs, critic_obs)
        else:
            actor_loss, alpha_loss, entropy = None, None, None

        if self._is_target_update:
            self._soft_update()

        return critic_loss, q_mean_mean, target_q_mean, actor_loss, alpha_loss, self.alpha.item(), entropy

    def _q_evaluate(self, critic_obs, actions, critic, use_min=False):
        q_mean, q_std = critic(critic_obs, actions)
        normal = Normal(torch.zeros_like(q_mean), torch.ones_like(q_std))
        if use_min:
            z = -torch.abs(normal.sample())
        else:
            z = normal.sample()
            z = torch.clamp(z, -3, 3)
        q_sample = q_mean + torch.mul(z, q_std)  # reparameterization trick
        return q_mean, q_std, q_sample
    
    # @torch.no_grad()
    def _compute_target_q(self, r, done, q_mean, q_std, q_next_sample, log_prob_next):
        target_q = r + (1 - done) * self.cfg.gamma * (
            q_next_sample - self.alpha.detach() * log_prob_next
        )
        td_bound = 3 * torch.mean(q_std)
        difference = torch.clamp(target_q - q_mean, -td_bound, td_bound)
        target_q_bound = q_mean + difference
        return target_q.detach(), target_q_bound.detach()
    
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

        done = termination.float()
        
        # -----------------------------------------------------
        # 1. Critic update
        # -----------------------------------------------------
        next_actions, next_log_prob = self.target_actor.sample(next_obs)
        q_mean, q_std, _ = self._q_evaluate(critic_obs, actions, self.critic, use_min=False)
        _, _, q_next_sample = self._q_evaluate(
            next_critic_obs, next_actions, self.target_critic, use_min=False
        )
        target_q, target_q_bound = self._compute_target_q(
            rewards,
            done,
            q_mean.detach(),
            q_std.detach(),
            q_next_sample.detach(),
            next_log_prob.detach(),
        )
        
        if self.cfg.bound:
            critic_loss = torch.mean(
                torch.pow(q_mean - target_q, 2) / (2 * torch.pow(q_std.detach(), 2))
                + torch.pow(q_mean.detach() - target_q_bound, 2) / (2 * torch.pow(q_std, 2))
                + torch.log(q_std)
            )
        else:
            critic_loss = -Normal(q_mean, q_std).log_prob(target_q).mean()
            
        self.critic_optimizer.zero_grad(set_to_none=True)
        critic_loss.backward()
        nn.utils.clip_grad_norm_(self.critic.parameters(), self.cfg.max_grad_norm)
        self.critic_optimizer.step()
        
        return float(critic_loss.item()), float(q_mean.detach().mean()), float(target_q.detach().mean())

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
        actions, log_prob = self.actor.sample(obs)
        q_mean, _, _ = self._q_evaluate(critic_obs, actions, self.critic, use_min=False)
        actor_loss = (self.alpha.detach() * log_prob - q_mean).mean()

        self.actor_optimizer.zero_grad(set_to_none=True)
        actor_loss.backward()
        nn.utils.clip_grad_norm_(self.actor.parameters(), self.cfg.max_grad_norm)
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
            
        entropy = float(-log_prob.mean().item())
            
        return (
            float(actor_loss.item()),
            0.0 if alpha_loss is None else float(alpha_loss.item()),
            entropy
        )

@configclass
class DSACCfg(AlgorithmBaseCfg):
    """
    DSAC configuration (algorithm + network hyperparameters only).
    Observation / action dimensions are provided at runtime.
    """

    class_type: type[DSAC] = DSAC
    optimzers_cfg: GroupedOptimizerCfg = GroupedOptimizerCfg()
    # -------------------------
    # DSAC hyperparameters
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
    bound               : bool = True

    max_grad_norm       : float = 1.0