from __future__ import annotations

import torch
from typing import Dict, Generator
from GopsEA import configclass
from dataclasses import MISSING
from GopsEA.buffer.replay_buffer_base import ReplayBufferBase, ReplayBufferBaseCfg


class DirectTransitionBuffer(ReplayBufferBase):
    """
    Direct transition replay buffer for off-policy algorithms.
    Stores (obs, critic_obs, action, reward, next_obs, next_critic_obs, termination, timeout)
    as individual transitions in a circular buffer.
    """
    
    def __init__(
        self,
        cfg: "DirectTransitionBufferCfg",
        device,
        obs_dim: int = None,
        critic_obs_dim: int = None,
        action_dim: int = None,
        **kwargs
    ):
        super().__init__(cfg, device, **kwargs)
        
        # Set dimensions from cfg or parameters
        self.obs_dim = obs_dim or cfg.obs_dim
        self.critic_obs_dim = critic_obs_dim or cfg.critic_obs_dim
        self.action_dim = action_dim or cfg.action_dim
        
        # Set buffer capacity and warmup
        self.max_steps = cfg.max_steps
        self.warmup_steps = cfg.warmup_steps if cfg.warmup_steps is not None else cfg.max_steps // 10
        
        # Create buffers
        self.create_buffer()
        
        # Initialize pointers and length
        self.clear()
    
    def create_buffer(self):
        """Initialize storage tensors for transitions."""
        capacity = self.max_steps
        
        self.obs_buffer = torch.zeros((capacity, self.obs_dim), device=self.device, dtype=torch.float32)
        self.critic_obs_buffer = torch.zeros((capacity, self.critic_obs_dim), device=self.device, dtype=torch.float32)
        self.actions_buffer = torch.zeros((capacity, self.action_dim), device=self.device, dtype=torch.float32)
        self.rewards_buffer = torch.zeros((capacity, 1), device=self.device, dtype=torch.float32)
        self.next_obs_buffer = torch.zeros((capacity, self.obs_dim), device=self.device, dtype=torch.float32)
        self.next_critic_obs_buffer = torch.zeros((capacity, self.critic_obs_dim), device=self.device, dtype=torch.float32)
        self.termination_buffer = torch.zeros((capacity, 1), device=self.device, dtype=torch.float32)
        self.timeout_buffer = torch.zeros((capacity, 1), device=self.device, dtype=torch.float32)
    
    def add(
        self,
        obs: torch.Tensor,
        critic_obs: torch.Tensor,
        actions: torch.Tensor,
        rewards: torch.Tensor,
        next_obs: torch.Tensor,
        next_critic_obs: torch.Tensor,
        termination: torch.Tensor,
        timeout: torch.Tensor,
    ) -> Dict[str, float]:
        """
        Add transitions to the buffer.
        
        Args:
            obs: Current observations, shape (num_envs, obs_dim)
            critic_obs: Current critic observations, shape (num_envs, critic_obs_dim)
            actions: Actions taken, shape (num_envs, action_dim)
            rewards: Rewards received, shape (num_envs, 1) or (num_envs,)
            next_obs: Next observations, shape (num_envs, obs_dim)
            next_critic_obs: Next critic observations, shape (num_envs, critic_obs_dim)
            termination: Termination flags, shape (num_envs, 1) or (num_envs,)
            timeout: Timeout flags, shape (num_envs, 1) or (num_envs,)
        
        Returns:
            Dictionary with statistics about the addition
        """
        return self.update_trans(obs, critic_obs, actions, rewards, next_obs, next_critic_obs, termination, timeout)
    
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
    ) -> Dict[str, float]:
        """
        Update buffer with new transitions (alias for add).
        
        Args:
            obs: Current observations, shape (num_envs, obs_dim)
            critic_obs: Current critic observations, shape (num_envs, critic_obs_dim)
            actions: Actions taken, shape (num_envs, action_dim)
            rewards: Rewards received, shape (num_envs, 1) or (num_envs,)
            next_obs: Next observations, shape (num_envs, obs_dim)
            next_critic_obs: Next critic observations, shape (num_envs, critic_obs_dim)
            termination: Termination flags, shape (num_envs, 1) or (num_envs,)
            timeout: Timeout flags, shape (num_envs, 1) or (num_envs,)
        
        Returns:
            Dictionary with statistics about the addition
        """
        num_envs = obs.shape[0]
        
        # Ensure rewards, termination, timeout are 2D
        if rewards.dim() == 1:
            rewards = rewards.unsqueeze(-1)
        if termination.dim() == 1:
            termination = termination.unsqueeze(-1)
        if timeout.dim() == 1:
            timeout = timeout.unsqueeze(-1)
        
        # Calculate indices for circular buffer
        indices = (torch.arange(num_envs, device=self.device) + self.ptr) % self.max_steps
        
        # Store transitions
        self.obs_buffer[indices] = obs
        self.critic_obs_buffer[indices] = critic_obs
        self.actions_buffer[indices] = actions
        self.rewards_buffer[indices] = rewards.float()
        self.next_obs_buffer[indices] = next_obs
        self.next_critic_obs_buffer[indices] = next_critic_obs
        self.termination_buffer[indices] = termination.float()
        self.timeout_buffer[indices] = timeout.float()
        
        # Update pointer and size
        self.ptr = (self.ptr + num_envs) % self.max_steps
        self.length = min(self.length + num_envs, self.max_steps)
        
        return {
            "buffer_length": self.length,
            "buffer_capacity": self.max_steps,
            "utilization": self.length / self.max_steps,
        }
    
    def clear(self):
        """Clear the buffer and reset pointers."""
        self.ptr = 0
        self.length = 0
    
    def is_warmingup(self) -> bool:
        """Return True if buffer has not yet reached warmup_length."""
        return self.length < self.warmup_steps
    
    def is_full(self) -> bool:
        """Return True if buffer is completely full."""
        return self.length >= self.max_steps
    
    def sample_batch(self, batch_size: int) -> Dict[str, torch.Tensor]:
        """
        Sample a batch of transitions from the buffer.
        
        Args:
            batch_size: Number of transitions to sample
            
        Returns:
            Dictionary with keys: obs, critic_obs, actions, rewards, next_obs, next_critic_obs, termination, timeout
        """
        if self.length == 0:
            raise ValueError("Buffer is empty, cannot sample.")
        
        # Sample random indices
        indices = torch.randint(0, self.length, (batch_size,), device=self.device)
        
        return {
            "obs": self.obs_buffer[indices],
            "critic_obs": self.critic_obs_buffer[indices],
            "actions": self.actions_buffer[indices],
            "rewards": self.rewards_buffer[indices],
            "next_obs": self.next_obs_buffer[indices],
            "next_critic_obs": self.next_critic_obs_buffer[indices],
            "termination": self.termination_buffer[indices],
            "timeout": self.timeout_buffer[indices],
        }
    
    def mini_batch_generator(
        self,
        num_epochs: int,
        batch_size: int,
        max_batches_per_epoch: int = None,
        **kwargs
    ) -> Generator[Dict[str, torch.Tensor], None, None]:
        """
        Generate mini-batches for training.
        
        Args:
            num_epochs: Number of epochs to iterate
            batch_size: Size of each batch
            max_batches_per_epoch: Maximum number of batches per epoch (None = all available)
            **kwargs: Additional arguments (unused)
        
        Yields:
            Dictionary with batched transitions
        """
        if self.length == 0:
            return
        
        # Calculate number of batches per epoch
        if max_batches_per_epoch is None:
            batches_per_epoch = (self.length + batch_size - 1) // batch_size
        else:
            batches_per_epoch = min(max_batches_per_epoch, (self.length + batch_size - 1) // batch_size)
        
        for epoch in range(num_epochs):
            # Shuffle indices for this epoch
            perm = torch.randperm(self.length, device=self.device)
            
            batches_yielded = 0
            for start_idx in range(0, self.length, batch_size):
                if max_batches_per_epoch is not None and batches_yielded >= max_batches_per_epoch:
                    break
                
                end_idx = min(start_idx + batch_size, self.length)
                indices = perm[start_idx:end_idx]
                
                # If batch is smaller than batch_size, pad with random samples
                if len(indices) < batch_size:
                    additional_indices = torch.randint(0, self.length, (batch_size - len(indices),), device=self.device)
                    indices = torch.cat([indices, additional_indices])
                
                batch = {
                    "obs": self.obs_buffer[indices],
                    "critic_obs": self.critic_obs_buffer[indices],
                    "actions": self.actions_buffer[indices],
                    "rewards": self.rewards_buffer[indices],
                    "next_obs": self.next_obs_buffer[indices],
                    "next_critic_obs": self.next_critic_obs_buffer[indices],
                    "termination": self.termination_buffer[indices],
                    "timeout": self.timeout_buffer[indices],
                }
                
                batches_yielded += 1
                yield batch
    
    @staticmethod
    def construct_from_cfg(cfg: "DirectTransitionBufferCfg", dim_params: dict, device, *args, num_envs=1, **kwargs):
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
        
        return DirectTransitionBuffer(
            cfg=cfg,
            device=device,
            obs_dim=obs_dim,
            critic_obs_dim=critic_obs_dim,
            action_dim=action_dim,
            **kwargs
        )


@configclass
class DirectTransitionBufferCfg(ReplayBufferBaseCfg):
    class_type: type[ReplayBufferBase] = DirectTransitionBuffer
    
    max_steps: int = MISSING  # Maximum number of transitions to store
    warmup_steps: int = None  # Number of steps before training starts (default: max_steps // 10)
    obs_dim: int = MISSING  # Observation dimension
    critic_obs_dim: int = MISSING  # Critic observation dimension
    action_dim: int = MISSING  # Action dimension
    
    def construct_from_cfg(self, dim_params: dict, device, *args, num_envs=1, **kwargs):
        return DirectTransitionBuffer.construct_from_cfg(self, dim_params, device, num_envs=num_envs, **kwargs)
