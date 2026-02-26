from __future__ import annotations

import torch
import torch.nn as nn
import torch.optim as optim

from GopsEA import configclass

from GopsEA.utils.template.module_base import ModuleBase, ModuleBaseCfg
from GopsEA.components.actor import StateIndStdActor
from GopsEA.components.critic import VNetwork
from GopsEA.buffer.online_rollout.rollout_storage import RolloutStorage
from GopsEA.utils.logging import timeit

from GopsEA.algorithms.algorithm_base import AlgorithmBase, AlgorithmBaseCfg

class PPO(AlgorithmBase):
    actor: StateIndStdActor
    critic: VNetwork

    def __init__(
        self,
        cfg: "PPOCfg",
        actor: "StateIndStdActor",
        critic: "VNetwork",
        device: str = "cpu",
    ):
        super().__init__()
        self.cfg = cfg
        self.device = device

        # components
        self.actor = actor.to(device)
        self.critic = critic.to(device)

        # rollout storage (lazy init)
        self.storage = None
        self.transition = RolloutStorage.Transition()

        # optimizer
        self.learning_rate = cfg.learning_rate
        self.optimizer = optim.Adam(
            list(self.actor.parameters()) + list(self.critic.parameters()),
            lr=self.learning_rate,
        )

    # --------------------------------------------------------------------- #
    # rollout & mode
    # --------------------------------------------------------------------- #
    def init_storage(
        self,
        num_envs: int,
        num_transitions_per_env: int,
        actor_obs_shape,
        critic_obs_shape,
        action_shape,
    ):
        self.storage = RolloutStorage(
            num_envs,
            num_transitions_per_env,
            actor_obs_shape,
            critic_obs_shape,
            action_shape,
            self.device,
        )

    def train_mode(self):
        self.actor.train()
        self.critic.train()

    def test_mode(self):
        self.actor.eval()
        self.critic.eval()

    # --------------------------------------------------------------------- #
    # interaction
    # --------------------------------------------------------------------- #
    @torch.no_grad()
    def act(self, obs, critic_obs):
        self.transition.actions = self.actor.act(obs).detach()
        self.transition.values = self.critic(critic_obs).detach()
        self.transition.actions_log_prob = self.actor.get_actions_log_prob(self.transition.actions).detach()
        self.transition.action_mean = self.actor.action_mean.detach()
        self.transition.action_sigma = self.actor.action_std.detach()
        self.transition.observations = obs
        self.transition.critic_observations = critic_obs

        return self.transition.actions

    def process_env_step(self, rewards, dones, infos):
        self.transition.rewards = rewards.clone()
        self.transition.dones = dones
        # bootstrap on timeouts
        if "timeout" in infos:
            self.transition.rewards += (
                self.cfg.gamma
                * self.transition.values.squeeze(-1)
                * infos["timeout"].to(self.device)
            )
        self.storage.add_transitions(self.transition)
        self.transition.clear()
        self.actor.reset(dones); self.critic.reset(dones)

    @torch.no_grad()
    def compute_returns(self, last_critic_obs):
        last_values = self.critic(last_critic_obs).detach()
        self.storage.compute_returns(last_values, self.cfg.gamma, self.cfg.lam)

    # --------------------------------------------------------------------- #
    # update
    # --------------------------------------------------------------------- #
    @timeit("update_time")
    def update(self):
        mean_value_loss = 0.0
        mean_surrogate_loss = 0.0
        generator = self.storage.mini_batch_generator(
            self.cfg.num_mini_batches, self.cfg.num_learning_epochs
        )
        for batch in generator:
            (
                obs_batch,
                critic_obs_batch,
                actions_batch,
                old_values_batch,
                advantages_batch,
                returns_batch,
                old_actions_log_prob_batch,
                old_mu_batch,
                old_sigma_batch,
            ) = batch
            self.actor.act(obs_batch)
            actions_log_prob_batch        = self.actor.get_actions_log_prob(actions_batch)
            entropy         = self.actor.entropy()
            value_pred           = self.critic(critic_obs_batch)
            mu_batch        = self.actor.action_mean
            sigma_batch     = self.actor.action_std
            
            # adaptive KL
            if self.cfg.schedule == "adaptive" and self.cfg.desired_kl is not None:
                with torch.inference_mode():
                    kl = torch.sum(
                        torch.log(sigma_batch / old_sigma_batch + 1.0e-5)
                        + (torch.square(old_sigma_batch) + torch.square(old_mu_batch - mu_batch))
                        / (2.0 * torch.square(sigma_batch))
                        - 0.5,
                        axis=-1,
                    )
                    kl_mean = torch.mean(kl)
                    if kl_mean > self.cfg.desired_kl * 2.0:
                        self.learning_rate = max(1e-5, self.learning_rate / 1.5)
                    elif kl_mean < self.cfg.desired_kl / 2.0 and kl_mean > 0.0:
                        self.learning_rate = min(1e-2, self.learning_rate * 1.5)

                    for param_group in self.optimizer.param_groups:
                        param_group["lr"] = self.learning_rate

            # surrogate loss
            ratio = torch.exp(actions_log_prob_batch - torch.squeeze(old_actions_log_prob_batch))
            surrogate = -torch.squeeze(advantages_batch) * ratio
            surrogate_clipped = -torch.squeeze(advantages_batch) * torch.clamp(
                ratio,
                1.0 - self.cfg.clip_param,
                1.0 + self.cfg.clip_param,
            )
            surrogate_loss = torch.max(surrogate, surrogate_clipped).mean()

            # value loss
            if self.cfg.use_clipped_value_loss:
                value_clipped = old_values_batch + (value_pred - old_values_batch).clamp(
                    -self.cfg.clip_param, self.cfg.clip_param
                )
                value_loss = torch.max(
                    (value_pred - returns_batch).pow(2),
                    (value_clipped - returns_batch).pow(2),
                ).mean()
            else:
                value_loss = (returns_batch - value_pred).pow(2).mean()

            loss = (
                surrogate_loss
                + self.cfg.value_loss_coef * value_loss
                - self.cfg.entropy_coef * entropy.mean()
            )

            # backward
            self.optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(
                list(self.actor.parameters()) + list(self.critic.parameters()),
                self.cfg.max_grad_norm,
            )
            self.optimizer.step()

            mean_value_loss += value_loss.item()
            mean_surrogate_loss += surrogate_loss.item()

        num_updates = self.cfg.num_learning_epochs * self.cfg.num_mini_batches
        self.storage.clear()
        
        return {
            "mean_value_loss" : mean_value_loss / num_updates,
            "mean_surrogate_loss" : mean_surrogate_loss / num_updates,
            "learning_rate": self.learning_rate,
            "mean_std": self.actor.action_std.mean()
        }

@configclass
class PPOCfg(AlgorithmBaseCfg):
    class_type: type[PPO] = PPO
    
    # optimization
    learning_rate: float = 3e-4
    min_learning_rate: float = 1e-5
    max_learning_rate: float = 1e-2

    # PPO specific
    clip_param: float = 0.2
    num_learning_epochs: int = 5
    num_mini_batches: int = 4
    value_loss_coef: float = 1.0
    entropy_coef: float = 0.0
    max_grad_norm: float = 1.0

    # return / advantage
    gamma: float = 0.99
    lam: float = 0.95
    use_clipped_value_loss: bool = True

    # KL control
    schedule: str = "fixed"  # fixed | adaptive
    desired_kl: float | None = 0.01