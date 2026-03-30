"""MLP-based system dynamics model for MBPO.

This module provides a simple MLP-based system dynamics model that predicts
next dynamic state and optional auxiliary signals.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from dataclasses import MISSING

from GopsEA import configclass
from GopsEA.networks.mlp import MLP, MLPCfg
from GopsEA.utils.template.module_base import ModuleBaseCfg
from GopsEA.components.world_models.system_dynamics.system_dynamics_base import (
    SystemDynamicsBase,
    SystemDynamicsBaseCfg,
)


class SystemDynamicsMLP(SystemDynamicsBase):
    """A simple MLP-based system dynamics model for MBPO.

    This module is intentionally much simpler than ensemble-based models:
    - no ensemble
    - no KL / RSSM terms
    - predicts next state (and optional auxiliary signals) with plain MSE / BCE.

    It only needs to support:
    - `compute_loss(state_batch, action_batch, extension_batch, contact_batch, termination_batch, reward_batch)`
      returning a 7-tuple of loss scalars compatible with `MBPO.update_system_dynamics`.
    """

    def __init__(self, cfg: "SystemDynamicsMLPCfg", dim_params: dict, device: str = "cpu"):
        # Infer dimensions from env dim_params + cfg overrides.
        dynamic_dim = dim_params.get("dynamic_dim")
        action_dim = dim_params.get("action_dim")
        extension_dim = dim_params.get("extension_dim", 0)
        contact_dim = dim_params.get("contact_dim", 0)
        termination_dim = dim_params.get("termination_dim", 0)
        reward_dim = dim_params.get("reward_dim", 1)
        history_horizon = cfg.history_horizon

        # Initialize base class
        super().__init__(
            state_dim=dynamic_dim,
            action_dim=action_dim,
            extension_dim=extension_dim,
            contact_dim=contact_dim,
            termination_dim=termination_dim,
            reward_dim=reward_dim,
            history_horizon=history_horizon,
            device=device,
        )

        self.cfg = cfg
        # Alias for consistency with naming convention
        self.dynamic_dim = self.state_dim

        # Input dimension depends on history_horizon
        # For history_horizon > 1, we flatten the history
        if history_horizon > 1:
            input_dim = history_horizon * (self.dynamic_dim + self.action_dim)
        else:
            input_dim = self.dynamic_dim + self.action_dim

        # Determine trunk output dimension (last hidden layer size)
        if len(cfg.backbone_cfg.hidden_features) > 0:
            trunk_output_dim = cfg.backbone_cfg.hidden_features[-1]
        else:
            # If no hidden layers, use input_dim as output
            trunk_output_dim = input_dim

        # Core MLP trunk using backbone_cfg with construct_from_cfg
        self.trunk = cfg.backbone_cfg.construct_from_cfg(
            in_feature=input_dim, out_feature=trunk_output_dim
        )

        # Store trunk output dimension for head construction
        last_dim = trunk_output_dim

        # Heads.
        self.dynamic_head = nn.Linear(last_dim, self.dynamic_dim)
        self.extension_head = (
            nn.Linear(last_dim, self.extension_dim) if self.extension_dim > 0 else None
        )
        self.contact_head = (
            nn.Linear(last_dim, self.contact_dim) if self.contact_dim > 0 else None
        )
        self.termination_head = (
            nn.Linear(last_dim, self.termination_dim) if self.termination_dim > 0 else None
        )
        self.reward_head = (
            nn.Linear(last_dim, self.reward_dim) if self.reward_dim > 0 else None
        )

        self.to(device)

    def _forward_core(self, dynamic_input: torch.Tensor):
        """Forward core network given flattened dynamic-action input.

        Args:
            dynamic_input: [B, input_dim] where input_dim = history_horizon * (dynamic_dim + action_dim)
                        or [B, dynamic_dim + action_dim] if history_horizon == 1
        """
        feat = self.trunk(dynamic_input)
        next_dynamic = self.dynamic_head(feat)
        extension = self.extension_head(feat) if self.extension_head is not None else None
        contact = self.contact_head(feat) if self.contact_head is not None else None
        termination = (
            self.termination_head(feat) if self.termination_head is not None else None
        )
        reward = self.reward_head(feat) if self.reward_head is not None else None
        return next_dynamic, extension, contact, termination, reward

    def forward(self, dynamic_seq: torch.Tensor, action_seq: torch.Tensor):
        """Predict next dynamic state using history.

        Args:
            dynamic_seq: [B, T, dynamic_dim] where T >= history_horizon
            action_seq: [B, T, action_dim] where T >= history_horizon

        Returns:
            next_dynamic_pred: [B, dynamic_dim]
            extension_pred: Optional[torch.Tensor]
            contact_pred: Optional[torch.Tensor]
            termination_pred: Optional[torch.Tensor]
            reward_pred: Optional[torch.Tensor]
        """
        B = dynamic_seq.shape[0]

        if self.history_horizon > 1:
            # Use last history_horizon steps
            dynamic_history = dynamic_seq[:, -self.history_horizon :]  # [B, history_horizon, dynamic_dim]
            action_history = action_seq[:, -self.history_horizon :]  # [B, history_horizon, action_dim]

            # Flatten history: [B, history_horizon * (dynamic_dim + action_dim)]
            dynamic_input = torch.cat(
                [dynamic_history, action_history], dim=-1
            )  # [B, history_horizon, dynamic_dim + action_dim]
            dynamic_input = dynamic_input.reshape(B, -1)  # [B, history_horizon * (dynamic_dim + action_dim)]
        else:
            # Single-step: use last dynamic state and action
            dynamic_t = dynamic_seq[:, -1]  # [B, dynamic_dim]
            action_t = action_seq[:, -1]  # [B, action_dim]
            dynamic_input = torch.cat([dynamic_t, action_t], dim=-1)  # [B, dynamic_dim + action_dim]

        return self._forward_core(dynamic_input)

    def compute_loss(
        self,
        dynamic_batch: torch.Tensor,
        action_batch: torch.Tensor,
        extension_batch: torch.Tensor | None = None,
        contact_batch: torch.Tensor | None = None,
        termination_batch: torch.Tensor | None = None,
        reward_batch: torch.Tensor | None = None,
    ):
        """Compute per-component losses over a sequence batch.

        The buffer provides sequences of length `T`. We use all T-1 transitions:
        (dynamic_t, action_t) -> dynamic_{t+1}, and average MSE/BCE over them.

        Shapes:
            dynamic_batch: [B, T, dynamic_dim]
            action_batch: [B, T, action_dim]
            reward_batch: Optional [B, T, reward_dim]
        """
        # Targets: next dynamic state and optional auxiliaries.
        dynamic_t = dynamic_batch[:, :-1]  # [B, T-1, D_d]
        dynamic_tp1 = dynamic_batch[:, 1:]  # [B, T-1, D_d]
        action_t = action_batch[:, :-1]  # [B, T-1, D_a]

        B, Tm1, _ = dynamic_t.shape

        # Prepare inputs for each transition
        if self.history_horizon > 1:
            # For each transition, we need history_horizon steps
            # We'll create sliding windows
            dynamic_inputs = []
            for i in range(Tm1):
                # Get history_horizon steps ending at step i
                start_idx = max(0, i + 1 - self.history_horizon)
                end_idx = i + 1
                dynamic_window = dynamic_batch[:, start_idx:end_idx]  # [B, window_len, dynamic_dim]
                action_window = action_batch[:, start_idx:end_idx]  # [B, window_len, action_dim]

                # Pad if necessary (shouldn't happen if sequence_length >= history_horizon)
                if dynamic_window.shape[1] < self.history_horizon:
                    pad_len = self.history_horizon - dynamic_window.shape[1]
                    dynamic_pad = dynamic_window[:, :1].repeat(1, pad_len, 1)
                    action_pad = action_window[:, :1].repeat(1, pad_len, 1)
                    dynamic_window = torch.cat([dynamic_pad, dynamic_window], dim=1)
                    action_window = torch.cat([action_pad, action_window], dim=1)

                # Flatten
                combined = torch.cat(
                    [dynamic_window, action_window], dim=-1
                )  # [B, history_horizon, dynamic_dim + action_dim]
                combined_flat = combined.reshape(B, -1)  # [B, history_horizon * (dynamic_dim + action_dim)]
                dynamic_inputs.append(combined_flat)

            dynamic_input = torch.stack(dynamic_inputs, dim=1)  # [B, Tm1, history_horizon * (dynamic_dim + action_dim)]
            dynamic_input = dynamic_input.reshape(B * Tm1, -1)  # [B * Tm1, history_horizon * (dynamic_dim + action_dim)]
        else:
            # Single-step: flatten all transitions
            dynamic_t_flat = dynamic_t.reshape(B * Tm1, self.dynamic_dim)
            action_t_flat = action_t.reshape(B * Tm1, self.action_dim)
            dynamic_input = torch.cat([dynamic_t_flat, action_t_flat], dim=-1)  # [B * Tm1, dynamic_dim + action_dim]

        dynamic_tp1_flat = dynamic_tp1.reshape(B * Tm1, self.dynamic_dim)

        next_dynamic_pred, ext_pred, contact_pred, term_pred, reward_pred = self._forward_core(dynamic_input)

        # Dynamic loss (MSE).
        dynamic_loss = torch.mean((next_dynamic_pred - dynamic_tp1_flat) ** 2)
        # We don't implement a separate "sequence" loss; reuse dynamic_loss.
        sequence_loss = dynamic_loss

        # No explicit bound regularization; keep it zero.
        bound_loss = torch.tensor(0.0, device=self.device)

        # Auxiliary losses.
        if self.extension_dim > 0 and extension_batch is not None:
            ext_target = extension_batch[:, 1:].reshape(B * Tm1, self.extension_dim)
            extension_loss = torch.mean((ext_pred - ext_target) ** 2)
        else:
            extension_loss = torch.tensor(0.0, device=self.device)

        bce = nn.BCEWithLogitsLoss()
        if self.contact_dim > 0 and contact_batch is not None:
            contact_target = contact_batch[:, 1:].reshape(B * Tm1, self.contact_dim)
            contact_loss = bce(contact_pred, contact_target)
        else:
            contact_loss = torch.tensor(0.0, device=self.device)

        if self.termination_dim > 0 and termination_batch is not None:
            term_target = termination_batch[:, 1:].reshape(B * Tm1, self.termination_dim)
            termination_loss = bce(term_pred, term_target)
        else:
            termination_loss = torch.tensor(0.0, device=self.device)

        # Reward loss (MSE).
        if self.reward_dim > 0 and reward_batch is not None:
            reward_target = reward_batch[:, 1:].reshape(B * Tm1, self.reward_dim)
            reward_loss = torch.mean((reward_pred - reward_target) ** 2)
        else:
            reward_loss = torch.tensor(0.0, device=self.device)

        return (
            dynamic_loss,
            sequence_loss,
            bound_loss,
            extension_loss,
            contact_loss,
            termination_loss,
            reward_loss,
        )


@configclass
class SystemDynamicsMLPCfg(SystemDynamicsBaseCfg):
    """Configuration for MLP-based system dynamics model.

    A simple MLP implementation of system dynamics that predicts next state
    and optional auxiliary signals (extensions, contacts, terminations).
    """

    class_type: type[nn.Module] = SystemDynamicsMLP

    # MLP backbone configuration for the trunk network
    backbone_cfg: MLPCfg = MLPCfg(
        hidden_features=[256, 256],
        activations=[[("ReLU", {})], [("ReLU", {})], []],  # No activation on last layer
    )
