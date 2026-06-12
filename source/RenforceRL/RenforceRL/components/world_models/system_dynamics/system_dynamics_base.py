"""Base class for system dynamics models.

System dynamics models predict the next state given current state and action.
Unlike world models that use latent variables, system dynamics work with explicit
physics terms (e.g., positions, velocities, torques).

The model can optionally predict:
- Extensions: Additional state information (e.g., joint extensions)
- Contacts: Contact signals (binary)
- Terminations: Termination signals (binary)

Rewards are typically computed from predicted states using rule-based functions,
not learned by the model.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from abc import ABC, abstractmethod
from typing import Tuple, Optional
from dataclasses import MISSING

from RenforceRL import configclass
from RenforceRL.utils.template.module_base import ModuleBaseCfg


class SystemDynamicsBase(nn.Module, ABC):
    """Base class for system dynamics models.
    
    System dynamics models predict state transitions:
        s_{t+1} = f(s_t, a_t)
    
    They work with explicit physics states (not latent representations),
    and can optionally predict auxiliary signals like contacts and terminations.
    """
    
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        extension_dim: int = 0,
        contact_dim: int = 0,
        termination_dim: int = 0,
        history_horizon: int = 1,
        device: str = "cpu",
    ):
        """Initialize system dynamics model.
        
        Args:
            state_dim: Dimension of system state
            action_dim: Dimension of action
            extension_dim: Dimension of extension signals (0 if not used)
            contact_dim: Dimension of contact signals (0 if not used)
            termination_dim: Dimension of termination signals (0 if not used)
            history_horizon: Number of historical steps to use (1 = single-step)
            device: Device to run on
        """
        super().__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.extension_dim = extension_dim
        self.contact_dim = contact_dim
        self.termination_dim = termination_dim
        self.history_horizon = history_horizon
        self.device = device
    
    @abstractmethod
    def forward(
        self,
        state_seq: torch.Tensor,
        action_seq: torch.Tensor,
    ) -> Tuple[
        torch.Tensor,  # next_state: [B, state_dim]
        Optional[torch.Tensor],  # extension: [B, extension_dim] or None
        Optional[torch.Tensor],  # contact: [B, contact_dim] or None
        Optional[torch.Tensor],  # termination: [B, termination_dim] or None
    ]:
        """Predict next state and optional auxiliary signals.
        
        Args:
            state_seq: [B, T, state_dim] - State sequence (T >= history_horizon)
            action_seq: [B, T, action_dim] - Action sequence (T >= history_horizon)
        
        Returns:
            next_state: [B, state_dim] - Predicted next state
            extension: Optional [B, extension_dim] - Predicted extension signals
            contact: Optional [B, contact_dim] - Predicted contact signals
            termination: Optional [B, termination_dim] - Predicted termination signals
        """
        pass
    
    @abstractmethod
    def compute_loss(
        self,
        state_batch: torch.Tensor,
        action_batch: torch.Tensor,
        extension_batch: Optional[torch.Tensor] = None,
        contact_batch: Optional[torch.Tensor] = None,
        termination_batch: Optional[torch.Tensor] = None,
    ) -> Tuple[
        torch.Tensor,  # state_loss
        torch.Tensor,  # sequence_loss
        torch.Tensor,  # bound_loss
        torch.Tensor,  # extension_loss
        torch.Tensor,  # contact_loss
        torch.Tensor,  # termination_loss
    ]:
        """Compute training losses.
        
        Args:
            state_batch: [B, T, state_dim] - State sequence
            action_batch: [B, T, action_dim] - Action sequence
            extension_batch: Optional [B, T, extension_dim]
            contact_batch: Optional [B, T, contact_dim]
            termination_batch: Optional [B, T, termination_dim]
        
        Returns:
            6-tuple of loss scalars compatible with MBPO.update_system_dynamics
        """
        pass


@configclass
class SystemDynamicsBaseCfg(ModuleBaseCfg):
    """Base configuration for system dynamics models.
    
    This config class defines the common interface for all system dynamics models.
    Subclasses should inherit from this and add model-specific parameters.
    """
    
    class_type: type[nn.Module] = SystemDynamicsBase
    
    # Dimension keys for extracting from dim_params
    state_dim_key: str = "dynamic_dim"
    action_dim_key: str = "action_dim"
    
    # Optional auxiliary output dimensions
    extension_dim: int = 0
    contact_dim: int = 0
    termination_dim: int = 0
    
    # History horizon for using past states/actions
    history_horizon: int = 1
    
    def construct_from_cfg(self, dim_params: dict = None, device: str = "cpu", **kwargs):
        """Construct system dynamics model from config.
        
        This is a base implementation that should be overridden by subclasses
        to match their specific constructor signatures.
        
        Args:
            dim_params: Dictionary containing dimension parameters from environment
            device: Device to run on
            **kwargs: Additional arguments passed to class_type constructor
        
        Returns:
            SystemDynamicsBase: Constructed system dynamics model instance
        
        Note:
            Subclasses must override this method to match their constructor signature.
            The base implementation raises NotImplementedError.
        """
        raise NotImplementedError(
            f"{self.__class__.__name__}.construct_from_cfg must be implemented by subclasses. "
            f"Base class {SystemDynamicsBase.__name__} cannot be instantiated directly."
        )
