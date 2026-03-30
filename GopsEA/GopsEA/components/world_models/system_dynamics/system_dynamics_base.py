"""Base class for system dynamics models.

System dynamics models predict the next dynamic state given current dynamic state and action.
Unlike world models that use latent variables, system dynamics work with explicit
physics terms (e.g., positions, velocities, torques).

The model can optionally predict:
- Extensions: Additional dynamic information (e.g., joint extensions)
- Contacts: Contact signals (binary)
- Terminations: Termination signals (binary)

Rewards are typically computed from predicted dynamic states using rule-based functions,
not learned by the model.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from abc import ABC, abstractmethod
from typing import Tuple, Optional
from dataclasses import MISSING

from GopsEA import configclass
from GopsEA.utils.template.module_base import ModuleBaseCfg


class SystemDynamicsBase(nn.Module, ABC):
    """Base class for system dynamics models.

    System dynamics models predict dynamic state transitions:
        d_{t+1} = f(d_t, a_t)

    They work with explicit physics dynamic states (not latent representations),
    and can optionally predict auxiliary signals like contacts and terminations.
    """

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        extension_dim: int = 0,
        contact_dim: int = 0,
        termination_dim: int = 0,
        reward_dim: int = 1,
        history_horizon: int = 1,
        device: str = "cpu",
    ):
        """Initialize system dynamics model.

        Args:
            state_dim: Dimension of system dynamic state
            action_dim: Dimension of action
            extension_dim: Dimension of extension signals (0 if not used)
            contact_dim: Dimension of contact signals (0 if not used)
            termination_dim: Dimension of termination signals (0 if not used)
            reward_dim: Dimension of reward (1 if scalar reward)
            history_horizon: Number of historical steps to use (1 = single-step)
            device: Device to run on
        """
        super().__init__()
        self.state_dim = state_dim
        self.dynamic_dim = state_dim  # Alias for consistency
        self.action_dim = action_dim
        self.extension_dim = extension_dim
        self.contact_dim = contact_dim
        self.termination_dim = termination_dim
        self.reward_dim = reward_dim
        self.history_horizon = history_horizon
        self.device = device

    @abstractmethod
    def forward(
        self,
        dynamic_seq: torch.Tensor,
        action_seq: torch.Tensor,
    ) -> Tuple[
        torch.Tensor,  # next_dynamic: [B, dynamic_dim]
        Optional[torch.Tensor],  # extension: [B, extension_dim] or None
        Optional[torch.Tensor],  # contact: [B, contact_dim] or None
        Optional[torch.Tensor],  # termination: [B, termination_dim] or None
        Optional[torch.Tensor],  # reward: [B, reward_dim] or None
    ]:
        """Predict next dynamic state and optional auxiliary signals.

        Args:
            dynamic_seq: [B, T, dynamic_dim] - Dynamic state sequence (T >= history_horizon)
            action_seq: [B, T, action_dim] - Action sequence (T >= history_horizon)

        Returns:
            next_dynamic: [B, dynamic_dim] - Predicted next dynamic state
            extension: Optional [B, extension_dim] - Predicted extension signals
            contact: Optional [B, contact_dim] - Predicted contact signals
            termination: Optional [B, termination_dim] - Predicted termination signals
            reward: Optional [B, reward_dim] - Predicted reward
        """
        pass

    @abstractmethod
    def compute_loss(
        self,
        dynamic_batch: torch.Tensor,
        action_batch: torch.Tensor,
        extension_batch: Optional[torch.Tensor] = None,
        contact_batch: Optional[torch.Tensor] = None,
        termination_batch: Optional[torch.Tensor] = None,
        reward_batch: Optional[torch.Tensor] = None,
    ) -> Tuple[
        torch.Tensor,  # dynamic_loss
        torch.Tensor,  # sequence_loss
        torch.Tensor,  # bound_loss
        torch.Tensor,  # extension_loss
        torch.Tensor,  # contact_loss
        torch.Tensor,  # termination_loss
        torch.Tensor,  # reward_loss
    ]:
        """Compute training losses.

        Args:
            dynamic_batch: [B, T, dynamic_dim] - Dynamic state sequence
            action_batch: [B, T, action_dim] - Action sequence
            extension_batch: Optional [B, T, extension_dim]
            contact_batch: Optional [B, T, contact_dim]
            termination_batch: Optional [B, T, termination_dim]
            reward_batch: Optional [B, T, reward_dim]

        Returns:
            7-tuple of loss scalars compatible with MBPO.update_system_dynamics
        """
        pass


@configclass
class SystemDynamicsBaseCfg(ModuleBaseCfg):
    """Base configuration for system dynamics models.

    This config class defines the common interface for all system dynamics models.
    Subclasses should inherit from this and add model-specific parameters.
    """

    class_type: type[nn.Module] = SystemDynamicsBase

    # History horizon for using past dynamic states/actions
    history_horizon: int = 1
