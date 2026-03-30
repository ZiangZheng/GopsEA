"""Model-based PPO (MBPO) on-policy algorithm.

Extends PPO with a learned system dynamics model and optional imagination rollouts.
"""

from __future__ import annotations
from dataclasses import MISSING

import torch
import torch.nn as nn
import torch.optim as optim

from GopsEA import configclass

from GopsEA.buffer.direct_based.dynamic_replay_buffer import DynamicReplayBuffer
from GopsEA.components.actor import StateIndStdActor
from GopsEA.components.critic import VNetwork
from GopsEA.components.actor_critic_pack import ActorCritic
from GopsEA.components.normalizer import NormalizerBase
from GopsEA.components.world_models.system_dynamics import SystemDynamicsMLP
from GopsEA.utils.template.module_base import ModuleBase
from GopsEA.buffer.online_rollout.rollout_storage import RolloutStorage

from GopsEA.algorithms.on_policy.ppo import PPO, PPOCfg
from GopsEA.algorithms.world_model_trainer.system_dynamics_trainer import (
    SystemDynamicsTrainer,
    SystemDynamicsTrainerCfg,
)


class MBPO(PPO):
    """Model-based PPO (MBPO) on-policy algorithm.

    Extends PPO with a learned system dynamics model and optional imagination rollouts.
    """

    actor: StateIndStdActor
    critic: VNetwork

    def __init__(
        self,
        cfg: "MBPOCfg",
        actor: "StateIndStdActor",
        critic: "VNetwork",
        system_dynamics: "SystemDynamicsMLP",
        replay_buffer: "DynamicReplayBuffer",
        device: str,
    ):
        super().__init__(cfg=cfg, actor=actor, critic=critic, device=device)

        self.cfg = cfg
        # system_dynamics can be left as None when only running vanilla PPO.
        self.system_dynamics = (
            system_dynamics.to(device) if system_dynamics is not None else None
        )

        self.replay_buffer: "DynamicReplayBuffer" = replay_buffer

        # Create SystemDynamicsTrainer if system_dynamics is provided
        self.system_dynamics_trainer: SystemDynamicsTrainer = (
            self.cfg.system_dynamics_trainer_cfg.construct_from_cfg(
                replay_buffer=replay_buffer,
                system_dynamics=system_dynamics,
            )
        )
        self.imagination_storage: RolloutStorage | None = None
        self.imagination_transition = RolloutStorage.Transition()

    def init_imagination_storage(
        self,
        num_envs: int,
        num_transitions_per_env: int,
        actor_obs_shape,
        critic_obs_shape,
        action_shape,
    ):
        """Initialize rollout storage for imagination trajectories."""
        self.imagination_storage = RolloutStorage(
            num_envs,
            num_transitions_per_env,
            actor_obs_shape,
            critic_obs_shape,
            action_shape,
            self.device,
        )

    # --------------------------------------------------------------------- #
    # interaction with env / imagination
    # --------------------------------------------------------------------- #

    @torch.no_grad()
    def act_imagination(self, obs, critic_obs):
        """Act in imagination; mirrors `act` but writes into imagination transition."""
        self.imagination_transition.actions = self.actor.act(obs).detach()
        self.imagination_transition.values = self.critic(critic_obs).detach()
        self.imagination_transition.actions_log_prob = (
            self.actor.get_actions_log_prob(self.imagination_transition.actions).detach()
        )
        self.imagination_transition.action_mean = self.actor.action_mean.detach()
        self.imagination_transition.action_sigma = self.actor.action_std.detach()
        self.imagination_transition.observations = obs
        self.imagination_transition.critic_observations = critic_obs
        return self.imagination_transition.actions

    def process_env_step(
        self,
        rewards: torch.Tensor,
        dones: torch.Tensor,
        infos: dict,
        imagination: bool = False,
    ):
        if imagination:
            # Mirror PPO.process_env_step but using imagination_transition/storage.
            self.imagination_transition.rewards = rewards.clone()
            self.imagination_transition.dones = dones
            if "timeout" in infos:
                self.imagination_transition.rewards += (
                    self.cfg.gamma
                    * self.imagination_transition.values.squeeze(-1)
                    * infos["timeout"].to(self.device)
                )
            self.imagination_storage.add_transitions(self.imagination_transition)
            self.imagination_transition.clear()
            self.actor.reset(dones)
            self.critic.reset(dones)
        else:
            return super().process_env_step(rewards, dones, infos)

    @torch.no_grad()
    def compute_imagination_returns(self, last_critic_obs: torch.Tensor):
        """Compute returns for imagination trajectories."""
        last_values = self.critic(last_critic_obs).detach()
        self.imagination_storage.compute_returns(
            last_values, self.cfg.gamma, self.cfg.lam
        )

    @torch.no_grad()
    def prepare_imagination(self):
        """Prepare initial dynamic/action history for imagination rollouts.

        Samples initial dynamic and action histories from the system replay buffer.
        Returns:
            dynamic_history: [num_imagination_envs, history_horizon, dynamic_dim]
            action_history: [num_imagination_envs, history_horizon, action_dim]
        """
        horizon = self.system_dynamics.history_horizon
        imagination_generator = self.replay_buffer.mini_batch_generator(
            sequence_length=horizon,
            num_mini_batches=1,
            mini_batch_size=self.imagination_storage.num_envs,
        )
        imagination_dynamic_history, imagination_action_history = next(imagination_generator)[:2]
        return imagination_dynamic_history, imagination_action_history

    def update_system_dynamics(self):
        """Update the system dynamics model from the replay buffer."""
        if self.system_dynamics_trainer is None:
            return {
                "state_loss": 0.0,
                "sequence_loss": 0.0,
                "bound_loss": 0.0,
                "extension_loss": 0.0,
                "contact_loss": 0.0,
                "termination_loss": 0.0,
                "reward_loss": 0.0,
            }
        return self.system_dynamics_trainer.update_system_dynamics()

    def evaluate_system_dynamics(self):
        """Evaluate system dynamics with autoregressive prediction.

        Returns:
            dict: Evaluation metrics including autoregressive errors
        """
        if self.system_dynamics_trainer is None:
            return {
                "traj_autoregressive_error": float("inf"),
                "traj_autoregressive_error_noised": {},
            }
        return self.system_dynamics_trainer.evaluate_system_dynamics()

    # --------------------------------------------------------------------- #
    # PPO update with optional imagination data
    # --------------------------------------------------------------------- #

    def _combined_mini_batch_generator(self):
        """Combine real and imagination storage mini-batches by concatenation."""
        assert self.imagination_storage is not None

        real_gen = self.storage.mini_batch_generator(
            self.cfg.num_mini_batches, self.cfg.num_learning_epochs
        )
        imag_gen = self.imagination_storage.mini_batch_generator(
            self.cfg.num_mini_batches, self.cfg.num_learning_epochs
        )

        for real_batch, imag_batch in zip(real_gen, imag_gen):
            (
                obs_r,
                critic_obs_r,
                actions_r,
                old_values_r,
                advantages_r,
                returns_r,
                old_log_prob_r,
                old_mu_r,
                old_sigma_r,
            ) = real_batch
            (
                obs_i,
                critic_obs_i,
                actions_i,
                old_values_i,
                advantages_i,
                returns_i,
                old_log_prob_i,
                old_mu_i,
                old_sigma_i,
            ) = imag_batch

            obs = torch.cat([obs_r, obs_i], dim=0)
            critic_obs = torch.cat([critic_obs_r, critic_obs_i], dim=0)
            actions = torch.cat([actions_r, actions_i], dim=0)
            old_values = torch.cat([old_values_r, old_values_i], dim=0)
            advantages = torch.cat([advantages_r, advantages_i], dim=0)
            returns = torch.cat([returns_r, returns_i], dim=0)
            old_log_prob = torch.cat([old_log_prob_r, old_log_prob_i], dim=0)
            old_mu = torch.cat([old_mu_r, old_mu_i], dim=0)
            old_sigma = torch.cat([old_sigma_r, old_sigma_i], dim=0)

            yield (
                obs,
                critic_obs,
                actions,
                old_values,
                advantages,
                returns,
                old_log_prob,
                old_mu,
                old_sigma,
            )

    def update(self, use_imagination: bool = False):
        """Run a PPO update; optionally mix in imagination data."""
        if not use_imagination or self.imagination_storage is None:
            return super().update()

        mean_value_loss = 0.0
        mean_surrogate_loss = 0.0

        generator = self._combined_mini_batch_generator()

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
            actions_log_prob_batch = self.actor.get_actions_log_prob(actions_batch)
            entropy = self.actor.entropy()
            value_pred = self.critic(critic_obs_batch)
            mu_batch = self.actor.action_mean
            sigma_batch = self.actor.action_std

            # adaptive KL (same as PPO)
            if self.cfg.schedule == "adaptive" and self.cfg.desired_kl is not None:
                with torch.inference_mode():
                    kl = torch.sum(
                        torch.log(sigma_batch / (old_sigma_batch + 1.0e-8) + 1.0e-5)
                        + (torch.square(old_sigma_batch) + torch.square(old_mu_batch - mu_batch))
                        / (2.0 * torch.square(sigma_batch))
                        - 0.5,
                        axis=-1,
                    )
                    kl_mean = torch.mean(kl)
                    if kl_mean > self.cfg.desired_kl * 2.0:
                        self.learning_rate = max(
                            self.cfg.min_learning_rate,
                            self.learning_rate / 1.5,
                        )
                    elif kl_mean < self.cfg.desired_kl / 2.0 and kl_mean > 0.0:
                        self.learning_rate = min(
                            self.cfg.max_learning_rate,
                            self.learning_rate * 1.5,
                        )
                    for param_group in self.optimizer.param_groups:
                        param_group["lr"] = self.learning_rate

            # surrogate loss
            ratio = torch.exp(
                actions_log_prob_batch - torch.squeeze(old_actions_log_prob_batch)
            )
            surrogate = -torch.squeeze(advantages_batch) * ratio
            surrogate_clipped = -torch.squeeze(advantages_batch) * torch.clamp(
                ratio,
                1.0 - self.cfg.clip_param,
                1.0 + self.cfg.clip_param,
            )
            surrogate_loss = torch.max(surrogate, surrogate_clipped).mean()

            # value loss (reuse PPO logic)
            if self.cfg.use_clipped_value_loss:
                value_clipped = old_values_batch + (value_pred - old_values_batch).clamp(
                    -self.cfg.clip_param,
                    self.cfg.clip_param,
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

            self.optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(
                list(self.actor.parameters()) + list(self.critic.parameters()),
                self.cfg.max_grad_norm,
            )
            self.optimizer.step()

            mean_value_loss += float(value_loss.detach())
            mean_surrogate_loss += float(surrogate_loss.detach())

        num_updates = self.cfg.num_learning_epochs * self.cfg.num_mini_batches
        self.storage.clear()
        if self.imagination_storage is not None:
            self.imagination_storage.clear()

        return {
            "mean_value_loss": mean_value_loss / num_updates,
            "mean_surrogate_loss": mean_surrogate_loss / num_updates,
            "learning_rate": self.learning_rate,
            "mean_std": self.actor.action_std.mean(),
        }


@configclass
class MBPOCfg(PPOCfg):
    """Configuration for MBPO algorithm."""

    class_type: type["MBPO"] = MBPO

    system_dynamics_trainer_cfg: SystemDynamicsTrainerCfg = SystemDynamicsTrainerCfg()

    def construct_from_cfg(
        self,
        actor_critic: "ActorCritic",
        system_dynamics: "ModuleBase",
        replay_buffer: "DynamicReplayBuffer",
        device: str,
        *args,
        **kwargs,
    ):
        return self.class_type(
            cfg=self,
            actor=actor_critic.actor,
            critic=actor_critic.critic,
            system_dynamics=system_dynamics,
            replay_buffer=replay_buffer,
            device=device,
            *args,
            **kwargs,
        )
