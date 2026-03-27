from __future__ import annotations

import copy

import torch
import torch.optim as optim
from torch.distributions import Normal
from torch.nn.functional import huber_loss

from GopsEA import configclass
from GopsEA.algorithms.algorithm_base import AlgorithmBase, AlgorithmBaseCfg
from GopsEA.components.actor import SACActor
from GopsEA.components.critic import GaussianQNetwork
from GopsEA.networks.optimizer import GroupedOptimizerCfg, GroupedOptimizer


class DSACT(AlgorithmBase):
    cfg: "DSACTCfg"
    actor: SACActor
    critic1: GaussianQNetwork
    critic2: GaussianQNetwork

    def __init__(
        self,
        cfg: "DSACTCfg",
        actor: SACActor,
        critic1: GaussianQNetwork,
        critic2: GaussianQNetwork,
        device: str,
    ):
        super().__init__()
        self.cfg = cfg
        self.device = device

        self.actor = actor.to(device)
        self.critic1 = critic1.to(device)
        self.critic2 = critic2.to(device)
        self.target_critic1 = copy.deepcopy(self.critic1).to(device)
        self.target_critic2 = copy.deepcopy(self.critic2).to(device)
        self.target_actor = copy.deepcopy(self.actor).to(device)
        for p in self.target_critic1.parameters():
            p.requires_grad_(False)
        for p in self.target_critic2.parameters():
            p.requires_grad_(False)
        for p in self.target_actor.parameters():
            p.requires_grad_(False)

        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=cfg.actor_lr, betas=(0.9, 0.999))
        self.critic1_optimizer = optim.Adam(self.critic1.parameters(), lr=cfg.critic_lr, betas=(0.9, 0.999))
        self.critic2_optimizer = optim.Adam(self.critic2.parameters(), lr=cfg.critic_lr, betas=(0.9, 0.999))
        self.optimizer = GroupedOptimizer(
            {"actor": self.actor_optimizer, "critic1": self.critic1_optimizer, "critic2": self.critic2_optimizer},
            self.cfg.optimzers_cfg,
        )

        self.auto_entropy = cfg.auto_entropy
        if self.auto_entropy:
            self.target_entropy = (
                cfg.target_entropy if cfg.target_entropy is not None else -float(actor.action_dim)
            )
            self.log_alpha = torch.zeros(1, requires_grad=True, device=device)
            self.alpha = self.log_alpha.exp()
            self.alpha_optimizer = optim.Adam([self.log_alpha], lr=cfg.alpha_lr, betas=(0.9, 0.999))
        else:
            self.alpha = torch.tensor(cfg.alpha, device=device)

        self.sample_actions = None
        self.sample_log_prob = None
        self.avg_std1 = None
        self.avg_std2 = None

    def act(self, obs, critic_obs):
        return self.actor.sample(obs)[0]

    @torch.no_grad()
    def _soft_update(self):
        tau = self.cfg.tau
        for p, tp in zip(self.critic1.parameters(), self.target_critic1.parameters()):
            tp.data.mul_(1.0 - tau)
            tp.data.add_(tau * p.data)
        for p, tp in zip(self.critic2.parameters(), self.target_critic2.parameters()):
            tp.data.mul_(1.0 - tau)
            tp.data.add_(tau * p.data)
        for p, tp in zip(self.actor.parameters(), self.target_actor.parameters()):
            tp.data.mul_(1.0 - tau)
            tp.data.add_(tau * p.data)

    def update(self, generator):
        self.ptr_update = 0
        critic_losses, avg_q1_mean_s, avg_q2_mean_s, avg_q1_std_s, avg_q2_std_s, actor_losses, alpha_losses, alphas, entropies = (
            [],
            [],
            [],
            [],
            [],
            [],
            [],
            [],
            [],
        )
        for minib in generator:
            if not isinstance(minib, dict):
                raise NotImplementedError("DSAC get bad replay buffer.")
            (
                critic_loss,
                avg_q1_mean,
                avg_q2_mean,
                avg_q1_std,
                avg_q2_std,
                _,
                _,
                actor_loss,
                alpha_loss,
                alpha,
                entropy,
            ) = self.compute_gradient(**minib)
            self.update_trans()
            critic_losses.append(critic_loss)
            avg_q1_mean_s.append(avg_q1_mean)
            avg_q2_mean_s.append(avg_q2_mean)
            avg_q1_std_s.append(avg_q1_std)
            avg_q2_std_s.append(avg_q2_std)
            if actor_loss is not None:
                actor_losses.append(float(actor_loss.item()))
            if alpha_loss is not None:
                alpha_losses.append(float(alpha_loss.item()))
            alphas.append(alpha)
            if entropy is not None:
                entropies.append(entropy)
        self.ptr_update += 1
        return {
            "critic_loss": sum(critic_losses) / len(critic_losses),
            "actor_loss": sum(actor_losses) / len(actor_losses) if actor_losses else 0.0,
            "q1_mean": sum(avg_q1_mean_s) / len(avg_q1_mean_s),
            "q2_mean": sum(avg_q2_mean_s) / len(avg_q2_mean_s),
            "q1_std": sum(avg_q1_std_s) / len(avg_q1_std_s),
            "q2_std": sum(avg_q2_std_s) / len(avg_q2_std_s),
            "alpha_loss": sum(alpha_losses) / len(alpha_losses) if alpha_losses else 0.0,
            "alpha": sum(alphas) / len(alphas),
            "entropy": sum(entropies) / len(entropies) if entropies else 0.0,
            "mini_batch_num": self.ptr_update,
        }

    def compute_gradient(
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
        sample_actions, sample_log_prob = self.actor.sample(obs)
        self.sample_actions = sample_actions
        self.sample_log_prob = sample_log_prob

        self.critic1_optimizer.zero_grad()
        self.critic2_optimizer.zero_grad()
        critic_loss, avg_q1_mean, avg_q2_mean, avg_q1_std, avg_q2_std, min_q1_std, min_q2_std = self.compute_loss_critic(
            obs, critic_obs, actions, rewards, next_obs, next_critic_obs, termination, timeout
        )
        critic_loss.backward()

        for p in self.critic1.parameters():
            p.requires_grad = False
        for p in self.critic2.parameters():
            p.requires_grad = False

        self.actor_optimizer.zero_grad()
        actor_loss, entropy = self.compute_loss_actor(obs, critic_obs)
        actor_loss.backward()

        for p in self.critic1.parameters():
            p.requires_grad = True
        for p in self.critic2.parameters():
            p.requires_grad = True

        if self.auto_entropy:
            self.alpha_optimizer.zero_grad()
            alpha_loss = self.compute_loss_alpha(obs)
            alpha_loss.backward()
        else:
            alpha_loss = None

        return (
            critic_loss,
            avg_q1_mean,
            avg_q2_mean,
            avg_q1_std,
            avg_q2_std,
            min_q1_std,
            min_q2_std,
            actor_loss,
            alpha_loss,
            self._get_alpha(),
            entropy,
        )

    def update_trans(self):
        self.critic1_optimizer.step()
        self.critic2_optimizer.step()
        if self._is_update_actor:
            self.actor_optimizer.step()
            self.alpha_optimizer.step()
        if self._is_target_update:
            self._soft_update()

    def _get_alpha(self, requires_grad: bool = False):
        alpha = self.log_alpha.exp()
        return alpha if requires_grad else alpha.item()

    @property
    def _is_update_actor(self):
        return (self.ptr_update % self.cfg.actor_update_freq) == 0

    @property
    def _is_target_update(self):
        return (self.ptr_update % self.cfg.target_update_freq) == 0

    def _q_evaluate(self, critic_obs, actions, critic, use_min=False):
        q_mean, q_std = critic(critic_obs, actions)
        normal = Normal(torch.zeros_like(q_mean), torch.ones_like(q_std))
        if use_min:
            z = -torch.abs(normal.sample())
        else:
            z = torch.clamp(normal.sample(), -3, 3)
        q_sample = q_mean + torch.mul(z, q_std)
        return q_mean, q_std, q_sample

    def _compute_target_q(self, r, done, q_mean, q_std, q_mean_next, q_sample_next, log_prob_next):
        target_q = r + (1 - done) * self.cfg.gamma * (q_mean_next - self._get_alpha() * log_prob_next)
        target_q_sample = r + (1 - done) * self.cfg.gamma * (q_sample_next - self._get_alpha() * log_prob_next)
        td_bound = 3 * q_std
        difference = torch.clamp(target_q_sample - q_mean, -td_bound, td_bound)
        target_q_bound = q_mean + difference
        return target_q.detach(), target_q_bound.detach()

    def compute_loss_critic(
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
        done = termination.float()
        next_actions, next_log_prob = self.target_actor.sample(next_obs)
        next_log_prob = next_log_prob.unsqueeze(-1)
        q1_mean, q1_std, _ = self._q_evaluate(critic_obs, actions, self.critic1, use_min=False)
        q2_mean, q2_std, _ = self._q_evaluate(critic_obs, actions, self.critic2, use_min=False)
        if self.avg_std1 is None:
            self.avg_std1 = torch.mean(q1_std.detach())
        else:
            self.avg_std1 = (1 - self.cfg.tau_b) * self.avg_std1 + self.cfg.tau_b * torch.mean(q1_std.detach())
        if self.avg_std2 is None:
            self.avg_std2 = torch.mean(q2_std.detach())
        else:
            self.avg_std2 = (1 - self.cfg.tau_b) * self.avg_std2 + self.cfg.tau_b * torch.mean(q2_std.detach())

        q1_mean_next, _, q1_sample_next = self._q_evaluate(next_critic_obs, next_actions, self.target_critic1, use_min=False)
        q2_mean_next, _, q2_sample_next = self._q_evaluate(next_critic_obs, next_actions, self.target_critic2, use_min=False)
        q_mean_next = torch.min(q1_mean_next, q2_mean_next)
        q_sample_next = torch.where(q1_mean_next < q2_mean_next, q1_sample_next, q2_sample_next)

        target_q1, target_q1_bound = self._compute_target_q(
            rewards,
            done,
            q1_mean.detach(),
            self.avg_std1.detach(),
            q_mean_next.detach(),
            q_sample_next.detach(),
            next_log_prob.detach(),
        )
        target_q2, target_q2_bound = self._compute_target_q(
            rewards,
            done,
            q2_mean.detach(),
            self.avg_std2.detach(),
            q_mean_next.detach(),
            q_sample_next.detach(),
            next_log_prob.detach(),
        )

        q1_std_detach = torch.clamp(q1_std, min=0.0).detach()
        q2_std_detach = torch.clamp(q2_std, min=0.0).detach()
        bias = 0.1
        ratio1 = (torch.pow(self.avg_std1, 2) / (torch.pow(q1_std_detach, 2) + bias)).clamp(min=0.1, max=10)
        ratio2 = (torch.pow(self.avg_std2, 2) / (torch.pow(q2_std_detach, 2) + bias)).clamp(min=0.1, max=10)

        critic1_loss = torch.mean(
            ratio1
            * (
                huber_loss(q1_mean, target_q1, delta=50, reduction="none")
                + q1_std
                * (q1_std_detach.pow(2) - huber_loss(q1_mean.detach(), target_q1_bound, delta=50, reduction="none"))
                / (q1_std_detach + bias)
            )
        )
        critic2_loss = torch.mean(
            ratio2
            * (
                huber_loss(q2_mean, target_q2, delta=50, reduction="none")
                + q2_std
                * (q2_std_detach.pow(2) - huber_loss(q2_mean.detach(), target_q2_bound, delta=50, reduction="none"))
                / (q2_std_detach + bias)
            )
        )
        critic_loss = critic1_loss + critic2_loss

        return (
            critic_loss,
            float(q1_mean.detach().mean()),
            float(q2_mean.detach().mean()),
            float(q1_std.detach().mean()),
            float(q2_std.detach().mean()),
            float(q1_std.min().detach()),
            float(q2_std.min().detach()),
        )

    def compute_loss_actor(self, obs: torch.Tensor, critic_obs: torch.Tensor):
        actions = self.sample_actions
        log_prob = self.sample_log_prob
        q1_mean, _, _ = self._q_evaluate(critic_obs, actions, self.critic1, use_min=False)
        q2_mean, _, _ = self._q_evaluate(critic_obs, actions, self.critic2, use_min=False)
        actor_loss = (self._get_alpha() * log_prob - torch.min(q1_mean, q2_mean)).mean()
        entropy = float(-log_prob.mean().item())
        return actor_loss, entropy

    def compute_loss_alpha(self, obs: torch.Tensor):
        log_prob = self.sample_log_prob
        return -(self.log_alpha * (log_prob.detach() + self.target_entropy)).mean()


@configclass
class DSACTCfg(AlgorithmBaseCfg):
    class_type: type[DSACT] = DSACT
    optimzers_cfg: GroupedOptimizerCfg = GroupedOptimizerCfg()

    gamma: float = 0.99
    tau: float = 0.005
    tau_b: float = 0.005

    actor_lr: float = 3e-4
    critic_lr: float = 3e-4
    alpha_lr: float = 3e-4
    actor_update_freq: int = 4
    target_update_freq: int = 4

    auto_entropy: bool = True
    target_entropy: float | None = None
    alpha: float = 0.2

    max_grad_norm: float = 1.0

    def construct_from_cfg(self, actor_critic, device, *args, **kwargs):
        return DSACT(
            self,
            actor=actor_critic.actor,
            critic1=actor_critic.critic[0],
            critic2=actor_critic.critic[1],
            device=device,
            *args,
            **kwargs,
        )
