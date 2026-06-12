from __future__ import annotations

import torch
import torch.nn as nn
from abc import ABC, abstractmethod


class ActorBase(nn.Module, ABC):
    """
    Base class for all actor implementations.
    
    All actors should inherit from this class and implement:
        - forward: compute action distribution or action
        - act_inference: deterministic action for inference/play
    """

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
    ):
        """
        Args:
            state_dim (int): Input state feature dimension.
            action_dim (int): Action dimension.
        """
        super().__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim

    @abstractmethod
    def forward(self, state: torch.Tensor):
        """
        Forward pass: compute action distribution or action.
        
        Args:
            state: Input state tensor
            
        Returns:
            Action distribution or action tensor
        """
        pass

    @torch.no_grad()
    def act_inference(self, state: torch.Tensor) -> torch.Tensor:
        """
        Deterministic action for inference/play.
        
        This method should return deterministic actions (typically the mean
        of the action distribution) for use during evaluation/play.
        
        Args:
            state: Input state tensor
            
        Returns:
            Deterministic action tensor
        """
        # Default implementation: try to use sample with deterministic=True
        # Subclasses should override this if they have a better implementation
        if hasattr(self, 'sample'):
            return self.sample(state, deterministic=True)
        elif hasattr(self, 'act'):
            return self.act(state)
        else:
            # Fallback: use forward and extract mean if it's a distribution
            result = self(state)
            if hasattr(result, 'mean'):
                if hasattr(result, 'base_dist'):
                    # TransformedDistribution case
                    return result.base_dist.mean.tanh()
                return result.mean
            return result
