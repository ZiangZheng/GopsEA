from __future__ import annotations

import torch
import torch.nn as nn
from dataclasses import MISSING

from RenforceRL import configclass
from RenforceRL.utils.template.module_base import ModuleBaseCfg
from RenforceRL.components.world_models.system_dynamics.system_dynamics_base import (
    SystemDynamicsBase,
    SystemDynamicsBaseCfg,
)


class SystemDynamicsMLP(SystemDynamicsBase):
    """A simple MLP-based system dynamics model for MBPO.

    This module is intentionally much simpler than the reference
    `SystemDynamicsEnsemble`:
    - no ensemble
    - no KL / RSSM terms
    - predicts next state (and optional auxiliary signals) with plain MSE / BCE.

    It only needs to support:
    - `compute_loss(state_batch, action_batch, extension_batch, contact_batch, termination_batch)`
      returning a 6-tuple of loss scalars compatible with `MBPO.update_system_dynamics`.
    """

    def __init__(self, cfg: "SystemDynamicsMLPCfg", dim_params: dict, device: str = "cpu"):
        # Infer dimensions from env dim_params + cfg overrides.
        state_dim = dim_params.get(cfg.state_dim_key, dim_params.get("dynamic_dim"))
        action_dim = dim_params.get(cfg.action_dim_key, dim_params.get("action_dim"))
        extension_dim = cfg.extension_dim
        contact_dim = cfg.contact_dim
        termination_dim = cfg.termination_dim
        history_horizon = cfg.history_horizon
        
        # Initialize base class
        super().__init__(
            state_dim=state_dim,
            action_dim=action_dim,
            extension_dim=extension_dim,
            contact_dim=contact_dim,
            termination_dim=termination_dim,
            history_horizon=history_horizon,
            device=device,
        )
        
        self.cfg = cfg

        # Input dimension depends on history_horizon
        # For history_horizon > 1, we flatten the history
        if history_horizon > 1:
            input_dim = history_horizon * (self.state_dim + self.action_dim)
        else:
            input_dim = self.state_dim + self.action_dim

        # Core MLP trunk.
        layers: list[nn.Module] = []
        last_dim = input_dim
        for h in cfg.hidden_sizes:
            layers.append(nn.Linear(last_dim, h))
            layers.append(nn.ReLU())
            last_dim = h
        self.trunk = nn.Sequential(*layers)

        # Heads.
        self.state_head = nn.Linear(last_dim, self.state_dim)
        self.extension_head = (
            nn.Linear(last_dim, self.extension_dim) if self.extension_dim > 0 else None
        )
        self.contact_head = (
            nn.Linear(last_dim, self.contact_dim) if self.contact_dim > 0 else None
        )
        self.termination_head = (
            nn.Linear(last_dim, self.termination_dim) if self.termination_dim > 0 else None
        )

        self.to(device)

    def _forward_core(self, state_input: torch.Tensor):
        """Forward core network given flattened state-action input.
        
        Args:
            state_input: [B, input_dim] where input_dim = history_horizon * (state_dim + action_dim)
                        or [B, state_dim + action_dim] if history_horizon == 1
        """
        feat = self.trunk(state_input)
        next_state = self.state_head(feat)
        extension = self.extension_head(feat) if self.extension_head is not None else None
        contact = self.contact_head(feat) if self.contact_head is not None else None
        termination = (
            self.termination_head(feat) if self.termination_head is not None else None
        )
        return next_state, extension, contact, termination

    def forward(self, state_seq: torch.Tensor, action_seq: torch.Tensor):
        """Predict next state using history.

        Args:
            state_seq: [B, T, state_dim] where T >= history_horizon
            action_seq: [B, T, action_dim] where T >= history_horizon

        Returns:
            next_state_pred: [B, state_dim]
            extension_pred: Optional[torch.Tensor]
            contact_pred: Optional[torch.Tensor]
            termination_pred: Optional[torch.Tensor]
        """
        B = state_seq.shape[0]
        
        if self.history_horizon > 1:
            # Use last history_horizon steps
            state_history = state_seq[:, -self.history_horizon:]  # [B, history_horizon, state_dim]
            action_history = action_seq[:, -self.history_horizon:]  # [B, history_horizon, action_dim]
            
            # Flatten history: [B, history_horizon * (state_dim + action_dim)]
            state_input = torch.cat([state_history, action_history], dim=-1)  # [B, history_horizon, state_dim + action_dim]
            state_input = state_input.reshape(B, -1)  # [B, history_horizon * (state_dim + action_dim)]
        else:
            # Single-step: use last state and action
            state_t = state_seq[:, -1]  # [B, state_dim]
            action_t = action_seq[:, -1]  # [B, action_dim]
            state_input = torch.cat([state_t, action_t], dim=-1)  # [B, state_dim + action_dim]
        
        return self._forward_core(state_input)

    def compute_loss(
        self,
        state_batch: torch.Tensor,
        action_batch: torch.Tensor,
        extension_batch: torch.Tensor | None,
        contact_batch: torch.Tensor | None,
        termination_batch: torch.Tensor | None,
    ):
        """Compute per-component losses over a sequence batch.

        The buffer provides sequences of length `T`. We use all T-1 transitions:
        (state_t, action_t) -> state_{t+1}, and average MSE/BCE over them.

        Shapes:
            state_batch: [B, T, state_dim]
            action_batch: [B, T, action_dim]
        """
        # Targets: next state and optional auxiliaries.
        state_t = state_batch[:, :-1]  # [B, T-1, D_s]
        state_tp1 = state_batch[:, 1:]  # [B, T-1, D_s]
        action_t = action_batch[:, :-1]  # [B, T-1, D_a]

        B, Tm1, _ = state_t.shape
        
        # Prepare inputs for each transition
        if self.history_horizon > 1:
            # For each transition, we need history_horizon steps
            # We'll create sliding windows
            state_inputs = []
            for i in range(Tm1):
                # Get history_horizon steps ending at step i
                start_idx = max(0, i + 1 - self.history_horizon)
                end_idx = i + 1
                state_window = state_batch[:, start_idx:end_idx]  # [B, window_len, state_dim]
                action_window = action_batch[:, start_idx:end_idx]  # [B, window_len, action_dim]
                
                # Pad if necessary (shouldn't happen if sequence_length >= history_horizon)
                if state_window.shape[1] < self.history_horizon:
                    pad_len = self.history_horizon - state_window.shape[1]
                    state_pad = state_window[:, :1].repeat(1, pad_len, 1)
                    action_pad = action_window[:, :1].repeat(1, pad_len, 1)
                    state_window = torch.cat([state_pad, state_window], dim=1)
                    action_window = torch.cat([action_pad, action_window], dim=1)
                
                # Flatten
                combined = torch.cat([state_window, action_window], dim=-1)  # [B, history_horizon, state_dim + action_dim]
                combined_flat = combined.reshape(B, -1)  # [B, history_horizon * (state_dim + action_dim)]
                state_inputs.append(combined_flat)
            
            state_input = torch.stack(state_inputs, dim=1)  # [B, Tm1, history_horizon * (state_dim + action_dim)]
            state_input = state_input.reshape(B * Tm1, -1)  # [B * Tm1, history_horizon * (state_dim + action_dim)]
        else:
            # Single-step: flatten all transitions
            state_t_flat = state_t.reshape(B * Tm1, self.state_dim)
            action_t_flat = action_t.reshape(B * Tm1, self.action_dim)
            state_input = torch.cat([state_t_flat, action_t_flat], dim=-1)  # [B * Tm1, state_dim + action_dim]
        
        state_tp1_flat = state_tp1.reshape(B * Tm1, self.state_dim)

        next_state_pred, ext_pred, contact_pred, term_pred = self._forward_core(state_input)

        # State loss (MSE).
        state_loss = torch.mean((next_state_pred - state_tp1_flat) ** 2)
        # We don't implement a separate "sequence" loss; reuse state_loss.
        sequence_loss = state_loss

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

        return (
            state_loss,
            sequence_loss,
            bound_loss,
            extension_loss,
            contact_loss,
            termination_loss,
        )


@configclass
class SystemDynamicsMLPCfg(SystemDynamicsBaseCfg):
    """Configuration for MLP-based system dynamics model.
    
    A simple MLP implementation of system dynamics that predicts next state
    and optional auxiliary signals (extensions, contacts, terminations).
    """
    
    class_type: type[nn.Module] = SystemDynamicsMLP

    # Simple MLP architecture.
    hidden_sizes: list[int] = (256, 256)
    
    def construct_from_cfg(self, dim_params: dict = None, device: str = "cpu", **kwargs):
        """Construct SystemDynamicsMLP from config.
        
        Args:
            dim_params: Dictionary containing dimension parameters from environment
            device: Device to run on
            **kwargs: Additional arguments passed to SystemDynamicsMLP constructor
        
        Returns:
            SystemDynamicsMLP: Constructed system dynamics model instance
        """
        if dim_params is None:
            raise ValueError("dim_params must be provided to construct system dynamics model")
        
        # Construct model instance (SystemDynamicsMLP accepts cfg and dim_params)
        return self.class_type(
            cfg=self,
            dim_params=dim_params,
            device=device,
            **kwargs
        )
