"""System dynamics trainer for MBPO.

This module provides training logic for system dynamics models used in MBPO.
"""

import torch
from torch import nn
from dataclasses import MISSING
from GopsEA import configclass
from GopsEA.utils.template import ClassTemplateBaseCfg
import torch.optim as optim
from GopsEA.buffer.direct_based.dynamic_replay_buffer import DynamicReplayBuffer


class SystemDynamicsTrainer:
    """Trainer for system dynamics models."""

    cfg: "SystemDynamicsTrainerCfg"

    def __init__(
        self,
        cfg: "SystemDynamicsTrainerCfg",
        replay_buffer: DynamicReplayBuffer,
        system_dynamics: nn.Module,
    ) -> None:
        self.cfg = cfg
        self.replay_buffer: DynamicReplayBuffer = replay_buffer
        self.system_dynamics = system_dynamics

        self.system_dynamics_optimizer = optim.Adam(
            self.system_dynamics.parameters(),
            lr=self.cfg.dynamic_update_learning_rate,
            weight_decay=self.cfg.dynamic_update_weight_decay,
        )

    def update_system_dynamics(self):
        """Update the system dynamics model from the replay buffer."""
        total_horizon = self.system_dynamics.history_horizon + self.cfg.dynamic_update_forecast_horizon
        mean_losses = {
            "state_loss": 0.0,
            "sequence_loss": 0.0,
            "bound_loss": 0.0,
            "extension_loss": 0.0,
            "contact_loss": 0.0,
            "termination_loss": 0.0,
            "reward_loss": 0.0,
        }
        num_updates = 0
        generator = self.replay_buffer.mini_batch_generator(
            sequence_length=total_horizon,
            num_mini_batches=self.cfg.dynamic_update_num_mini_batches,
            mini_batch_size=self.cfg.dynamic_update_mini_batch_size,
        )

        for (
            dynamic_batch,
            action_batch,
            extension_batch,
            contact_batch,
            termination_batch,
            reward_batch,
        ) in generator:
            # Expect system_dynamics to implement `compute_loss` with a similar signature.
            losses = self.system_dynamics.compute_loss(
                dynamic_batch,
                action_batch,
                extension_batch,
                contact_batch,
                termination_batch,
                reward_batch,
            )

            # `losses` can be a dict or a tuple; we normalize it to a dict.
            if isinstance(losses, dict):
                dynamic_loss = losses.get("dynamic_loss", 0.0)
                sequence_loss = losses.get("sequence_loss", 0.0)
                bound_loss = losses.get("bound_loss", 0.0)
                extension_loss = losses.get("extension_loss", 0.0)
                contact_loss = losses.get("contact_loss", 0.0)
                termination_loss = losses.get("termination_loss", 0.0)
                reward_loss = losses.get("reward_loss", 0.0)
            else:
                (
                    dynamic_loss,
                    sequence_loss,
                    bound_loss,
                    extension_loss,
                    contact_loss,
                    termination_loss,
                    reward_loss,
                ) = losses

            loss = (
                self.cfg.system_dynamics_loss_weights["dynamic"] * dynamic_loss
                + self.cfg.system_dynamics_loss_weights["sequence"] * sequence_loss
                + self.cfg.system_dynamics_loss_weights["bound"] * bound_loss
                + self.cfg.system_dynamics_loss_weights["extension"] * extension_loss
                + self.cfg.system_dynamics_loss_weights["contact"] * contact_loss
                + self.cfg.system_dynamics_loss_weights["termination"] * termination_loss
                + self.cfg.system_dynamics_loss_weights.get("reward", 1.0) * reward_loss
            )

            self.system_dynamics_optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(
                self.system_dynamics.parameters(), self.cfg.max_grad_norm
            )
            self.system_dynamics_optimizer.step()

            mean_losses["state_loss"] += float(dynamic_loss.detach())
            mean_losses["sequence_loss"] += float(sequence_loss.detach())
            mean_losses["bound_loss"] += float(bound_loss.detach())
            mean_losses["extension_loss"] += float(extension_loss.detach())
            mean_losses["contact_loss"] += float(contact_loss.detach())
            mean_losses["termination_loss"] += float(termination_loss.detach())
            mean_losses["reward_loss"] += float(reward_loss.detach())
            num_updates += 1

        if num_updates == 0:
            return {k: 0.0 for k in mean_losses}

        for k in mean_losses:
            mean_losses[k] /= num_updates

        return mean_losses

    @torch.no_grad()
    def evaluate_system_dynamics(self):
        """Evaluate system dynamics with autoregressive prediction.

        Returns:
            dict: Evaluation metrics including autoregressive errors
        """
        if (
            not hasattr(self.cfg, "eval_trajectory_length")
            or self.cfg.eval_trajectory_length is None
        ):
            # Default: use a reasonable evaluation length
            eval_length = self.system_dynamics.history_horizon + 50
        else:
            eval_length = self.cfg.eval_trajectory_length

        if (
            not hasattr(self.cfg, "num_eval_trajectories")
            or self.cfg.num_eval_trajectories is None
        ):
            num_eval_trajectories = 10
        else:
            num_eval_trajectories = self.cfg.num_eval_trajectories

        # Get evaluation trajectories from replay buffer
        eval_generator = self.replay_buffer.mini_batch_generator(
            sequence_length=eval_length,
            num_mini_batches=1,
            mini_batch_size=num_eval_trajectories,
        )

        try:
            (
                dynamic_traj,
                action_traj,
                extension_traj,
                contact_traj,
                termination_traj,
                reward_traj,
            ) = next(eval_generator)
        except StopIteration:
            # No data available for evaluation
            return {
                "traj_autoregressive_error": float("inf"),
                "traj_autoregressive_error_noised": {},
            }

        # Perform autoregressive prediction
        if hasattr(self.system_dynamics, "autoregressive_prediction"):
            (
                dynamic_pred,
                extension_pred,
                contact_pred,
                termination_pred,
                reward_pred,
            ) = self.system_dynamics.autoregressive_prediction(
                dynamic_traj,
                action_traj,
                extension_traj,
                contact_traj,
                termination_traj,
            )
        else:
            # Fallback: use forward method step by step (less efficient)
            dynamic_pred = dynamic_traj.clone()
            H = self.system_dynamics.history_horizon
            x_state_batch = dynamic_traj[:, :H].clone()

            for i in range(H, eval_length):
                action_window = action_traj[:, i - H + 1 : i + 1]
                next_dynamic, _, _, _, _ = self.system_dynamics._forward_core(
                    x_state_batch, action_window
                )
                dynamic_pred[:, i] = next_dynamic
                x_state_batch = torch.cat(
                    [x_state_batch[:, 1:].clone(), next_dynamic.unsqueeze(1)], dim=1
                )
            extension_pred = extension_traj.clone() if extension_traj is not None else None
            contact_pred = contact_traj.clone() if contact_traj is not None else None
            termination_pred = termination_traj.clone() if termination_traj is not None else None
            reward_pred = reward_traj.clone() if reward_traj is not None else None

        # Compute autoregressive error (only for predicted steps, skip history horizon)
        H = self.system_dynamics.history_horizon
        pred_dynamic = dynamic_pred[:, H:]
        gt_dynamic = dynamic_traj[:, H:]

        # Normalized error: sum of absolute errors / sum of absolute ground truth
        traj_autoregressive_error = (
            (pred_dynamic - gt_dynamic).abs().sum(dim=-1)
            / (gt_dynamic.abs().sum(dim=-1) + 1e-8)
        ).mean().item()

        # Evaluate with noise (if configured)
        traj_autoregressive_error_noised_dict = {}
        if hasattr(self.cfg, "eval_traj_noise_scale") and self.cfg.eval_traj_noise_scale:
            for noise_scale in self.cfg.eval_traj_noise_scale:
                dynamic_traj_noised = dynamic_traj + torch.randn_like(dynamic_traj) * noise_scale
                action_traj_noised = action_traj + torch.randn_like(action_traj) * noise_scale

                if hasattr(self.system_dynamics, "autoregressive_prediction"):
                    (
                        dynamic_pred_noised,
                        _,
                        _,
                        _,
                        _,
                    ) = self.system_dynamics.autoregressive_prediction(
                        dynamic_traj_noised,
                        action_traj_noised,
                        extension_traj,
                        contact_traj,
                        termination_traj,
                    )
                else:
                    # Fallback
                    dynamic_pred_noised = dynamic_traj_noised.clone()
                    x_state_batch = dynamic_traj_noised[:, :H].clone()
                    for i in range(H, eval_length):
                        action_window = action_traj_noised[:, i - H + 1 : i + 1]
                        next_dynamic, _, _, _, _ = self.system_dynamics._forward_core(
                            x_state_batch, action_window
                        )
                        dynamic_pred_noised[:, i] = next_dynamic

                pred_dynamic_noised = dynamic_pred_noised[:, H:]
                gt_dynamic_noised = dynamic_traj_noised[:, H:]
                traj_autoregressive_error_noised = (
                    (pred_dynamic_noised - gt_dynamic_noised).abs().sum(dim=-1)
                    / (gt_dynamic_noised.abs().sum(dim=-1) + 1e-8)
                ).mean().item()
                traj_autoregressive_error_noised_dict[noise_scale] = traj_autoregressive_error_noised

        return {
            "traj_autoregressive_error": traj_autoregressive_error,
            "traj_autoregressive_error_noised": traj_autoregressive_error_noised_dict,
        }


@configclass
class SystemDynamicsTrainerCfg(ClassTemplateBaseCfg):
    """Configuration for system dynamics trainer."""

    class_type: type[SystemDynamicsTrainer] = SystemDynamicsTrainer

    # System dynamics optimisation
    dynamic_update_learning_rate: float = 1e-3
    dynamic_update_weight_decay: float = 0.0
    # Used for update the dynamics, would affect the replay length.
    dynamic_update_forecast_horizon: int = MISSING
    dynamic_update_num_mini_batches: int = MISSING
    dynamic_update_mini_batch_size: int = MISSING

    # Loss weights
    system_dynamics_loss_weights: dict = {
        "dynamic": 1.0,
        "sequence": 1.0,
        "bound": 1.0,
        "extension": 1.0,
        "contact": 1.0,
        "termination": 1.0,
        "reward": 1.0,
    }

    # Gradient clipping
    max_grad_norm: float = 1.0

    # Evaluation parameters
    eval_trajectory_length: int | None = None  # If None, uses history_horizon + 50
    num_eval_trajectories: int | None = None  # If None, uses 10
    eval_traj_noise_scale: list[float] | None = None  # List of noise scales for robustness testing
