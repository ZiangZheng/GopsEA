from __future__ import annotations

import torch
import torch.nn as nn
import torch.optim as optim

from RenforceRL import configclass

from RenforceRL.components.actor import StateIndStdActor
from RenforceRL.components.critic import VNetwork
from RenforceRL.components.actor_critic_pack import ActorCritic
from RenforceRL.components.normalizer import NormalizerBase
from RenforceRL.utils.template.module_base import ModuleBaseCfg
from RenforceRL.buffer.online_rollout.rollout_storage import RolloutStorage

from RenforceRL.algorithms.on_policy.ppo import PPO, PPOCfg

from RenforceRL.buffer.direct_based.simple_system_replay_buffer import SimpleSystemReplayBuffer

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
        system_dynamics: nn.Module | None,
        state_normalizer: NormalizerBase | None = None,
        action_normalizer: NormalizerBase | None = None,
        device: str = "cpu",
    ):
        super().__init__(cfg=cfg, actor=actor, critic=critic, device=device)

        self.mbpo_cfg = cfg
        # system_dynamics can be left as None when only running vanilla PPO.
        self.system_dynamics = (
            system_dynamics.to(device) if system_dynamics is not None else None
        )
        self.state_normalizer = state_normalizer
        self.action_normalizer = action_normalizer

        # System dynamics optimizer and replay buffer are lazy-initialized once
        # we know the state/action dimensions from the first call to
        # `init_system_replay_buffer`.
        self.system_dynamics_optimizer: optim.Optimizer | None = None
        self.system_replay_buffer: SimpleSystemReplayBuffer | None = None

        # Imagination rollout storage
        self.imagination_storage: RolloutStorage | None = None
        self.imagination_transition = RolloutStorage.Transition()

    # --------------------------------------------------------------------- #
    # replay buffer / storage helpers
    # --------------------------------------------------------------------- #

    def init_system_replay_buffer(
        self,
        state_dim: int,
        action_dim: int,
        extension_dim: int = 0,
        contact_dim: int = 0,
        termination_dim: int = 0,
    ):
        """Initialize the replay buffer and optimizer for system dynamics."""
        # If no dynamics model is provided, skip.
        if self.system_dynamics is None or self.system_replay_buffer is not None:
            return

        self.system_replay_buffer = SimpleSystemReplayBuffer(
            state_dim=state_dim,
            action_dim=action_dim,
            extension_dim=extension_dim,
            contact_dim=contact_dim,
            termination_dim=termination_dim,
            capacity=self.mbpo_cfg.system_dynamics_replay_buffer_size,
            device=self.device,
        )
        self.system_dynamics_optimizer = optim.Adam(
            self.system_dynamics.parameters(),
            lr=self.mbpo_cfg.system_dynamics_learning_rate,
            weight_decay=self.mbpo_cfg.system_dynamics_weight_decay,
        )

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
        """Extend PPO's `process_env_step` to support imagination storage."""
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
            assert (
                self.imagination_storage is not None
            ), "Imagination storage not initialized"
            self.imagination_storage.add_transitions(self.imagination_transition)
            self.imagination_transition.clear()
            self.actor.reset(dones)
            self.critic.reset(dones)
        else:
            # Fall back to standard PPO behaviour.
            return super().process_env_step(rewards, dones, infos)

    @torch.no_grad()
    def compute_imagination_returns(self, last_critic_obs: torch.Tensor):
        """Compute returns for imagination trajectories."""
        assert self.imagination_storage is not None
        last_values = self.critic(last_critic_obs).detach()
        self.imagination_storage.compute_returns(
            last_values, self.cfg.gamma, self.cfg.lam
        )

    @torch.no_grad()
    def prepare_imagination(self):
        """Prepare initial state/action history for imagination rollouts.
        
        Samples initial state and action histories from the system replay buffer.
        Returns:
            state_history: [num_imagination_envs, history_horizon, state_dim]
            action_history: [num_imagination_envs, history_horizon, action_dim]
        """
        assert self.imagination_storage is not None
        assert self.system_replay_buffer is not None
        assert self.system_dynamics is not None
        
        # Get history horizon from system dynamics config
        history_horizon = (
            self.mbpo_cfg.system_dynamics_history_horizon
            + self.mbpo_cfg.system_dynamics_forecast_horizon
        )
        
        # Sample initial histories from replay buffer
        imagination_generator = self.system_replay_buffer.mini_batch_generator(
            sequence_length=history_horizon,
            num_mini_batches=1,
            mini_batch_size=self.imagination_storage.num_envs,
        )
        imagination_state_history, imagination_action_history = next(imagination_generator)[:2]
        
        return imagination_state_history, imagination_action_history

    # --------------------------------------------------------------------- #
    # system dynamics training
    # --------------------------------------------------------------------- #

    @torch.no_grad()
    def fill_history_buffer(
        self,
        system_state: torch.Tensor,
        system_action: torch.Tensor,
        system_extension: torch.Tensor | None = None,
        system_contact: torch.Tensor | None = None,
        system_termination: torch.Tensor | None = None,
    ):
        """Push a batch of system transitions into replay buffer."""
        assert (
            self.system_replay_buffer is not None
        ), "System replay buffer not initialized"

        # Normalize if normalizers are provided.
        if self.state_normalizer is not None:
            system_state = self.state_normalizer(system_state)
        if self.action_normalizer is not None:
            system_action = self.action_normalizer(system_action)

        # Flatten over env dimension.
        B = system_state.shape[0]
        state_flat = system_state.view(B, -1)
        action_flat = system_action.view(B, -1)
        ext_flat = (
            system_extension.view(B, -1) if system_extension is not None else None
        )
        contact_flat = (
            system_contact.view(B, -1) if system_contact is not None else None
        )
        term_flat = (
            system_termination.view(B, -1) if system_termination is not None else None
        )

        self.system_replay_buffer.insert(
            state_flat, action_flat, ext_flat, contact_flat, term_flat
        )

    def update_system_dynamics(self):
        """Update the system dynamics model from the replay buffer."""
        if (
            self.system_replay_buffer is None
            or self.system_dynamics_optimizer is None
            or self.system_dynamics is None
        ):
            return {}

        horizon = (
            self.mbpo_cfg.system_dynamics_history_horizon
            + self.mbpo_cfg.system_dynamics_forecast_horizon
        )

        mean_losses = {
            "state_loss": 0.0,
            "sequence_loss": 0.0,
            "bound_loss": 0.0,
            "extension_loss": 0.0,
            "contact_loss": 0.0,
            "termination_loss": 0.0,
        }
        num_updates = 0

        generator = self.system_replay_buffer.mini_batch_generator(
            sequence_length=horizon,
            num_mini_batches=self.mbpo_cfg.system_dynamics_num_mini_batches,
            mini_batch_size=self.mbpo_cfg.system_dynamics_mini_batch_size,
        )

        for (
            state_batch,
            action_batch,
            extension_batch,
            contact_batch,
            termination_batch,
        ) in generator:
            # Expect system_dynamics to implement `compute_loss` with a similar signature.
            losses = self.system_dynamics.compute_loss(
                state_batch,
                action_batch,
                extension_batch,
                contact_batch,
                termination_batch,
            )

            # `losses` can be a dict or a tuple; we normalize it to a dict.
            if isinstance(losses, dict):
                state_loss = losses.get("state_loss", 0.0)
                sequence_loss = losses.get("sequence_loss", 0.0)
                bound_loss = losses.get("bound_loss", 0.0)
                extension_loss = losses.get("extension_loss", 0.0)
                contact_loss = losses.get("contact_loss", 0.0)
                termination_loss = losses.get("termination_loss", 0.0)
            else:
                (
                    state_loss,
                    sequence_loss,
                    bound_loss,
                    extension_loss,
                    contact_loss,
                    termination_loss,
                ) = losses

            loss = (
                self.mbpo_cfg.system_dynamics_loss_weights["state"] * state_loss
                + self.mbpo_cfg.system_dynamics_loss_weights["sequence"]
                * sequence_loss
                + self.mbpo_cfg.system_dynamics_loss_weights["bound"] * bound_loss
                + self.mbpo_cfg.system_dynamics_loss_weights["extension"]
                * extension_loss
                + self.mbpo_cfg.system_dynamics_loss_weights["contact"]
                * contact_loss
                + self.mbpo_cfg.system_dynamics_loss_weights["termination"]
                * termination_loss
            )

            self.system_dynamics_optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(
                self.system_dynamics.parameters(), self.mbpo_cfg.max_grad_norm
            )
            self.system_dynamics_optimizer.step()

            mean_losses["state_loss"] += float(state_loss.detach())
            mean_losses["sequence_loss"] += float(sequence_loss.detach())
            mean_losses["bound_loss"] += float(bound_loss.detach())
            mean_losses["extension_loss"] += float(extension_loss.detach())
            mean_losses["contact_loss"] += float(contact_loss.detach())
            mean_losses["termination_loss"] += float(termination_loss.detach())
            num_updates += 1

        if num_updates == 0:
            return {k: 0.0 for k in mean_losses}

        for k in mean_losses:
            mean_losses[k] /= num_updates

        return mean_losses

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
    class_type: type["MBPO"] = None  # set in __post_init__

    # System dynamics optimisation
    system_dynamics_learning_rate: float = 1e-3
    system_dynamics_weight_decay: float = 0.0
    system_dynamics_replay_buffer_size: int = 100_000

    # Horizons
    system_dynamics_history_horizon: int = 1
    system_dynamics_forecast_horizon: int = 1

    # Loss weights
    system_dynamics_loss_weights: dict = {
        "state": 1.0,
        "sequence": 1.0,
        "bound": 1.0,
        "extension": 1.0,
        "contact": 1.0,
        "termination": 1.0,
    }

    # Mini-batch settings
    system_dynamics_num_mini_batches: int = 8
    system_dynamics_mini_batch_size: int = 1024

    def construct_from_cfg(
        self,
        actor_critic: "ActorCritic",
        system_dynamics,
        state_normalizer,
        action_normalizer,
        device,
        *args,
        **kwargs,
    ):
        """Construct MBPO algorithm from configs and shared ActorCritic."""
        return self.class_type(
            cfg=self,
            actor=actor_critic.actor,
            critic=actor_critic.critic,
            system_dynamics=system_dynamics,
            state_normalizer=state_normalizer,
            action_normalizer=action_normalizer,
            device=device,
            *args,
            **kwargs,
        )

    def __post_init__(self):
        # Defer import to avoid circulars during module loading.
        from RenforceRL.algorithms.on_policy.mbpo.mbpo import MBPO as _MBPO

        self.class_type = _MBPO

