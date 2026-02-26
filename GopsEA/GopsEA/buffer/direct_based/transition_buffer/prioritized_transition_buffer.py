from __future__ import annotations

import torch
import numpy as np
from typing import Dict, Generator, Optional
from GopsEA import configclass
from dataclasses import MISSING
from .direct_transition_buffer import DirectTransitionBuffer, DirectTransitionBufferCfg


class PrioritizedTransitionBuffer(DirectTransitionBuffer):
    """
    Prioritized transition replay buffer for off-policy algorithms.
    Extends DirectTransitionBuffer with priority-based sampling.
    
    Each transition has a priority score, and sampling is weighted by these priorities.
    Supports importance sampling weights to correct for the bias introduced by prioritized sampling.
    """
    
    def __init__(
        self,
        cfg: "PrioritizedTransitionBufferCfg",
        device,
        obs_dim: int = None,
        critic_obs_dim: int = None,
        action_dim: int = None,
        **kwargs
    ):
        super().__init__(cfg, device, obs_dim, critic_obs_dim, action_dim, **kwargs)
        
        # Priority-related parameters
        self.alpha = cfg.alpha  # Priority exponent (0 = uniform, 1 = fully prioritized)
        self.beta = cfg.beta  # Importance sampling exponent (0 = no correction, 1 = full correction)
        self.beta_schedule = cfg.beta_schedule  # How beta increases over time
        self.epsilon = cfg.epsilon  # Small constant to ensure non-zero priorities
        self.max_priority = cfg.max_priority  # Initial priority for new transitions
        
        # Track beta updates
        self.beta_start = cfg.beta
        self.beta_end = cfg.beta_end if hasattr(cfg, 'beta_end') else cfg.beta
        self.total_updates = 0
    
    def create_buffer(self):
        """Initialize storage tensors for transitions, including priority buffer."""
        super().create_buffer()
        
        # Priority buffer: stores priority scores for each transition
        capacity = self.max_steps
        self.priority_buffer = torch.full(
            (capacity,), 
            self.max_priority, 
            device=self.device, 
            dtype=torch.float32
        )
    
    def update_trans(
        self,
        obs: torch.Tensor,
        critic_obs: torch.Tensor,
        actions: torch.Tensor,
        rewards: torch.Tensor,
        next_obs: torch.Tensor,
        next_critic_obs: torch.Tensor,
        termination: torch.Tensor,
        timeout: torch.Tensor,
        priorities: Optional[torch.Tensor] = None,
    ) -> Dict[str, float]:
        """
        Update buffer with new transitions.
        
        Args:
            obs: Current observations, shape (num_envs, obs_dim)
            critic_obs: Current critic observations, shape (num_envs, critic_obs_dim)
            actions: Actions taken, shape (num_envs, action_dim)
            rewards: Rewards received, shape (num_envs, 1) or (num_envs,)
            next_obs: Next observations, shape (num_envs, obs_dim)
            next_critic_obs: Next critic observations, shape (num_envs, critic_obs_dim)
            termination: Termination flags, shape (num_envs, 1) or (num_envs,)
            timeout: Timeout flags, shape (num_envs, 1) or (num_envs,)
            priorities: Optional priority scores, shape (num_envs,) or (num_envs, 1). 
                       If None, uses max_priority.
        
        Returns:
            Dictionary with statistics about the addition
        """
        num_envs = obs.shape[0]
        
        # Calculate indices for circular buffer BEFORE calling super() (which updates self.ptr)
        indices = (torch.arange(num_envs, device=self.device) + self.ptr) % self.max_steps
        
        # Store transitions (call parent method)
        result = super().update_trans(
            obs, critic_obs, actions, rewards, next_obs, next_critic_obs, termination, timeout
        )
        
        # Prepare priorities
        if priorities is None:
            priorities = torch.full((num_envs,), self.max_priority, device=self.device, dtype=torch.float32)
        else:
            if priorities.dim() > 1:
                priorities = priorities.squeeze(-1)
            priorities = priorities.float()
            # Ensure priorities are positive and add epsilon to avoid zero probabilities
            priorities = torch.clamp(priorities, min=0.0) + self.epsilon
        
        # Store priorities at the same indices (calculated before super() call)
        self.priority_buffer[indices] = priorities
        
        return result
    
    def update_priorities(self, indices: torch.Tensor, priorities: torch.Tensor):
        """
        Update priority scores for existing transitions.
        
        Args:
            indices: Indices of transitions to update, shape (N,)
            priorities: New priority scores, shape (N,) or (N, 1)
        """
        if priorities.dim() > 1:
            priorities = priorities.squeeze(-1)
        priorities = priorities.float()
        # Ensure priorities are positive and add epsilon to avoid zero probabilities
        priorities = torch.clamp(priorities, min=0.0) + self.epsilon
        
        # Handle circular buffer: indices might be out of current length
        valid_mask = indices < self.length
        if valid_mask.any():
            self.priority_buffer[indices[valid_mask]] = priorities[valid_mask]
    
    def _get_priorities(self, indices: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Get priority scores for given indices, or all valid priorities.
        
        Args:
            indices: Optional indices to get priorities for. If None, returns all valid priorities.
        
        Returns:
            Priority scores
        """
        if indices is None:
            return self.priority_buffer[:self.length]
        else:
            return self.priority_buffer[indices]
    
    def _compute_sampling_probs(self, priorities: torch.Tensor) -> torch.Tensor:
        """
        Compute sampling probabilities from priorities.
        
        Args:
            priorities: Priority scores, shape (N,)
        
        Returns:
            Sampling probabilities, shape (N,)
        """
        # p(i) = priority_i^alpha / sum(priority_j^alpha)
        priorities_alpha = torch.pow(priorities, self.alpha)
        probs = priorities_alpha / priorities_alpha.sum()
        return probs
    
    def _sample_indices_by_priority(self, batch_size: int) -> torch.Tensor:
        """
        Sample indices based on priority scores.
        
        Args:
            batch_size: Number of indices to sample
        
        Returns:
            Sampled indices, shape (batch_size,)
        """
        if self.length == 0:
            raise ValueError("Buffer is empty, cannot sample.")
        
        # Get priorities for valid transitions
        priorities = self._get_priorities()
        
        # Compute sampling probabilities
        probs = self._compute_sampling_probs(priorities)
        
        # Sample indices according to probabilities
        indices = torch.multinomial(probs, batch_size, replacement=True)
        
        return indices
    
    def _compute_importance_weights(self, indices: torch.Tensor) -> torch.Tensor:
        """
        Compute importance sampling weights to correct for prioritized sampling bias.
        
        Args:
            indices: Sampled indices, shape (batch_size,)
        
        Returns:
            Importance sampling weights, shape (batch_size,)
        """
        if self.length == 0:
            raise ValueError("Buffer is empty, cannot compute weights.")
        
        # Get current beta (may increase over time)
        current_beta = self._get_current_beta()
        
        # Get priorities for sampled indices
        sampled_priorities = self._get_priorities(indices)
        
        # Get all valid priorities
        all_priorities = self._get_priorities()
        
        # Compute sampling probabilities
        probs = self._compute_sampling_probs(all_priorities)
        sampled_probs = probs[indices]
        
        # Importance sampling weight: w(i) = (N * p(i))^(-beta)
        weights = torch.pow(self.length * sampled_probs, -current_beta)
        
        # Normalize by max weight to stabilize training
        max_weight = weights.max()
        if max_weight > 0:
            weights = weights / max_weight
        
        return weights
    
    def _get_current_beta(self) -> float:
        """
        Get current beta value (may increase over time according to schedule).
        
        Returns:
            Current beta value
        """
        if self.beta_schedule == "linear" and hasattr(self, 'beta_end'):
            # Linear schedule: interpolate between beta_start and beta_end
            # This would need total training steps, for now just return beta
            # In practice, you'd pass total_steps and current_step
            return self.beta
        else:
            return self.beta
    
    def sample_batch(self, batch_size: int, return_weights: bool = True) -> Dict[str, torch.Tensor]:
        """
        Sample a batch of transitions from the buffer using prioritized sampling.
        
        Args:
            batch_size: Number of transitions to sample
            return_weights: If True, include importance sampling weights in the batch
        
        Returns:
            Dictionary with keys: obs, critic_obs, actions, rewards, next_obs, 
            next_critic_obs, termination, timeout, and optionally weights, indices
        """
        if self.length == 0:
            raise ValueError("Buffer is empty, cannot sample.")
        
        # Sample indices based on priorities
        indices = self._sample_indices_by_priority(batch_size)
        
        # Get transitions
        batch = {
            "obs": self.obs_buffer[indices],
            "critic_obs": self.critic_obs_buffer[indices],
            "actions": self.actions_buffer[indices],
            "rewards": self.rewards_buffer[indices],
            "next_obs": self.next_obs_buffer[indices],
            "next_critic_obs": self.next_critic_obs_buffer[indices],
            "termination": self.termination_buffer[indices],
            "timeout": self.timeout_buffer[indices],
            "indices": indices,  # Return indices so priorities can be updated later
        }
        
        # Add importance sampling weights if requested
        if return_weights:
            batch["weights"] = self._compute_importance_weights(indices)
        
        return batch
    
    def mini_batch_generator(
        self,
        num_epochs: int,
        batch_size: int,
        max_batches_per_epoch: int = None,
        return_weights: bool = True,
        **kwargs
    ) -> Generator[Dict[str, torch.Tensor], None, None]:
        """
        Generate mini-batches for training using prioritized sampling.
        
        Args:
            num_epochs: Number of epochs to iterate
            batch_size: Size of each batch
            max_batches_per_epoch: Maximum number of batches per epoch (None = all available)
            return_weights: If True, include importance sampling weights in batches
            **kwargs: Additional arguments (unused)
        
        Yields:
            Dictionary with batched transitions, including weights and indices
        """
        if self.length == 0:
            return
        
        # Calculate number of batches per epoch
        if max_batches_per_epoch is None:
            batches_per_epoch = (self.length + batch_size - 1) // batch_size
        else:
            batches_per_epoch = min(max_batches_per_epoch, (self.length + batch_size - 1) // batch_size)
        
        for epoch in range(num_epochs):
            batches_yielded = 0
            
            for _ in range(batches_per_epoch):
                if max_batches_per_epoch is not None and batches_yielded >= max_batches_per_epoch:
                    break
                
                # Sample indices based on priorities
                indices = self._sample_indices_by_priority(batch_size)
                
                batch = {
                    "obs": self.obs_buffer[indices],
                    "critic_obs": self.critic_obs_buffer[indices],
                    "actions": self.actions_buffer[indices],
                    "rewards": self.rewards_buffer[indices],
                    "next_obs": self.next_obs_buffer[indices],
                    "next_critic_obs": self.next_critic_obs_buffer[indices],
                    "termination": self.termination_buffer[indices],
                    "timeout": self.timeout_buffer[indices],
                    "indices": indices,  # Return indices so priorities can be updated later
                }
                
                # Add importance sampling weights if requested
                if return_weights:
                    batch["weights"] = self._compute_importance_weights(indices)
                
                batches_yielded += 1
                yield batch
    
    def clear(self):
        """Clear the buffer and reset pointers."""
        super().clear()
        # Reset priorities to max_priority
        self.priority_buffer.fill_(self.max_priority)
    
    @staticmethod
    def construct_from_cfg(cfg: "PrioritizedTransitionBufferCfg", dim_params: dict, device, *args, num_envs=1, **kwargs):
        """
        Construct buffer from config.
        
        Args:
            cfg: Configuration object
            dim_params: Dictionary with dimension parameters (policy_dim, critic_dim, action_dim)
            device: Device to store buffers on
            num_envs: Number of parallel environments (unused for direct buffer)
            **kwargs: Additional arguments
        """
        obs_dim = dim_params.get("policy_dim", cfg.obs_dim)
        critic_obs_dim = dim_params.get("critic_dim", cfg.critic_obs_dim)
        action_dim = dim_params.get("action_dim", cfg.action_dim)
        
        return PrioritizedTransitionBuffer(
            cfg=cfg,
            device=device,
            obs_dim=obs_dim,
            critic_obs_dim=critic_obs_dim,
            action_dim=action_dim,
            **kwargs
        )


@configclass
class PrioritizedTransitionBufferCfg(DirectTransitionBufferCfg):
    class_type: type[DirectTransitionBuffer] = PrioritizedTransitionBuffer
    
    # Priority sampling parameters
    alpha: float = 0.6  # Priority exponent: 0 = uniform sampling, 1 = fully prioritized
    beta: float = 0.4  # Importance sampling exponent: 0 = no correction, 1 = full correction
    beta_schedule: str = "constant"  # How beta changes: "constant" or "linear"
    epsilon: float = 1e-6  # Small constant added to priorities to ensure non-zero probability
    max_priority: float = 1.0  # Initial priority for new transitions
    
    def construct_from_cfg(self, dim_params: dict, device, *args, num_envs=1, **kwargs):
        return PrioritizedTransitionBuffer.construct_from_cfg(self, dim_params, device, num_envs=num_envs, **kwargs)
