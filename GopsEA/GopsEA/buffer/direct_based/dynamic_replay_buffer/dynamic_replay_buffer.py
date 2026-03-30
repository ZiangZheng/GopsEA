"""Environment-based replay buffer for system dynamics training.

Stores data with shape [num_envs, buffer_size, dim] - each environment has its own buffer row.
Supports batch insertion and avoids sampling sequences that cross environment reset boundaries.
"""

from __future__ import annotations

import torch
import numpy as np
from GopsEA import configclass
from dataclasses import MISSING
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from GopsEA.buffer.replay_buffer_base import ReplayBufferBaseCfg


class DynamicReplayBuffer:
    """Environment-based replay buffer for system dynamics training.

    Stores data with shape [num_envs, buffer_size, dim] - each environment has its own buffer row.
    Supports batch insertion and avoids sampling sequences that cross environment reset boundaries.

    This implementation is based on the reference implementation from rsl_rl_rwm,
    maintaining the same mechanism for per-environment storage and reset-aware sampling.
    """

    def __init__(
        self,
        dynamic_dim: int,
        action_dim: int,
        extension_dim: int = 0,
        contact_dim: int = 0,
        termination_dim: int = 0,
        reward_dim: int = 1,
        buffer_size: int = 100_000,
        device: str = "cpu",
    ):
        """Initialize the replay buffer.

        Args:
            dynamic_dim: Dimension of dynamic state
            action_dim: Dimension of action
            extension_dim: Dimension of extension data (0 if not used)
            contact_dim: Dimension of contact data (0 if not used)
            termination_dim: Dimension of termination data (0 if not used)
            reward_dim: Dimension of reward (default 1)
            buffer_size: Maximum number of transitions per environment
            device: Device to store the buffer on
        """
        self.device = device
        self.buffer_size = buffer_size

        # Store dimensions
        self.dynamic_dim = dynamic_dim
        self.action_dim = action_dim
        self.extension_dim = extension_dim
        self.contact_dim = contact_dim
        self.termination_dim = termination_dim
        self.reward_dim = reward_dim

        # Buffer will be initialized lazily when first data is inserted
        self.replay_bufs = None
        self.num_envs = None
        self.step = 0
        self.num_transitions = 0

    def _initialize_buffer(self, num_envs: int):
        """Initialize the replay buffers dynamically based on first insertion."""
        self.num_envs = num_envs

        # Create separate buffers for each field
        self.replay_bufs = {
            "dynamic": torch.zeros(num_envs, self.buffer_size, self.dynamic_dim, device=self.device),
            "action": torch.zeros(num_envs, self.buffer_size, self.action_dim, device=self.device),
        }

        if self.extension_dim > 0:
            self.replay_bufs["extension"] = torch.zeros(
                num_envs, self.buffer_size, self.extension_dim, device=self.device
            )
        else:
            self.replay_bufs["extension"] = None

        if self.contact_dim > 0:
            self.replay_bufs["contact"] = torch.zeros(
                num_envs, self.buffer_size, self.contact_dim, device=self.device
            )
        else:
            self.replay_bufs["contact"] = None

        if self.termination_dim > 0:
            self.replay_bufs["termination"] = torch.zeros(
                num_envs, self.buffer_size, self.termination_dim, device=self.device
            )
        else:
            self.replay_bufs["termination"] = None

        if self.reward_dim > 0:
            self.replay_bufs["reward"] = torch.zeros(
                num_envs, self.buffer_size, self.reward_dim, device=self.device
            )
        else:
            self.replay_bufs["reward"] = None

    def insert(
        self,
        dynamic: torch.Tensor,
        action: torch.Tensor,
        extension: torch.Tensor | None = None,
        contact: torch.Tensor | None = None,
        termination: torch.Tensor | None = None,
        reward: torch.Tensor | None = None,
    ):
        """Insert new transitions into the buffer in a circular manner.

        Args:
            dynamic: Dynamic states, shape [num_envs, num_steps, dynamic_dim] or [num_envs, dynamic_dim]
            action: Actions, shape [num_envs, num_steps, action_dim] or [num_envs, action_dim]
            extension: Extension data, shape [num_envs, num_steps, extension_dim] or [num_envs, extension_dim] (optional)
            contact: Contact data, shape [num_envs, num_steps, contact_dim] or [num_envs, contact_dim] (optional)
            termination: Termination flags, shape [num_envs, num_steps, termination_dim] or [num_envs, termination_dim] (optional)
            reward: Rewards, shape [num_envs, num_steps, reward_dim] or [num_envs, reward_dim] (optional)
        """
        # Initialize buffer if needed
        if self.replay_bufs is None:
            num_envs = dynamic.shape[0]
            self._initialize_buffer(num_envs)

        # Ensure 3D: [num_envs, num_steps, dim]
        if dynamic.dim() == 2:
            dynamic = dynamic.unsqueeze(1)
        if action.dim() == 2:
            action = action.unsqueeze(1)
        if extension is not None and extension.dim() == 2:
            extension = extension.unsqueeze(1)
        if contact is not None and contact.dim() == 2:
            contact = contact.unsqueeze(1)
        if termination is not None and termination.dim() == 2:
            termination = termination.unsqueeze(1)
        if reward is not None and reward.dim() == 2:
            reward = reward.unsqueeze(1)

        num_inputs = dynamic.shape[1]

        def _insert_into_buffer(r_buf, i_buf):
            """Insert data into a single buffer, handling circular wrapping."""
            if r_buf is None or i_buf is None:
                return
            end_idx = self.step + num_inputs
            if end_idx > self.buffer_size:
                # Handle circular wrapping: split into two segments
                r_buf[:, self.step : self.buffer_size] = i_buf[:, : self.buffer_size - self.step]
                r_buf[:, : end_idx - self.buffer_size] = i_buf[:, self.buffer_size - self.step :]
            else:
                r_buf[:, self.step : end_idx] = i_buf

        # Insert into each buffer
        _insert_into_buffer(self.replay_bufs["dynamic"], dynamic)
        _insert_into_buffer(self.replay_bufs["action"], action)
        _insert_into_buffer(self.replay_bufs["extension"], extension)
        _insert_into_buffer(self.replay_bufs["contact"], contact)
        _insert_into_buffer(self.replay_bufs["termination"], termination)
        _insert_into_buffer(self.replay_bufs["reward"], reward)

        # Update counters
        self.num_transitions = min(self.buffer_size, self.num_transitions + num_inputs)
        self.step = (self.step + num_inputs) % self.buffer_size

    def mini_batch_generator(
        self,
        sequence_length: int,
        num_mini_batches: int,
        mini_batch_size: int,
        reset_data: torch.Tensor | None = None,
    ):
        """Yield mini-batches of sequences for training.

        Args:
            sequence_length: Length of sequences to sample
            num_mini_batches: Number of mini-batches to generate
            mini_batch_size: Size of each mini-batch
            reset_data: Reset flags, shape [num_envs, buffer_size] or [num_envs, num_transitions]
                       If provided, sequences crossing reset boundaries will be avoided.

        Yields:
            Tuple of (dynamic_batch, action_batch, extension_batch, contact_batch,
                     termination_batch, reward_batch)
            Each batch has shape [mini_batch_size, sequence_length, dim]
        """
        assert self.replay_bufs is not None, "Replay buffer is not initialized."

        # Pad buffer if needed
        replay_bufs = self._pad_and_get_replay_bufs(sequence_length)

        # Generate valid indices if reset data is provided
        valid_indices = None
        if reset_data is not None:
            valid_indices = self._generate_valid_indices(reset_data, sequence_length)

        for _ in range(num_mini_batches):
            yield self._generate_batch(
                replay_bufs, valid_indices, sequence_length, mini_batch_size
            )

    def _pad_and_get_replay_bufs(self, sequence_length: int):
        """Pad buffers if num_transitions < sequence_length."""
        if self.num_transitions >= sequence_length:
            return self.replay_bufs

        padding_size = sequence_length - self.num_transitions
        padded_bufs = {}

        for key, buf in self.replay_bufs.items():
            if buf is None:
                padded_bufs[key] = None
            else:
                padding = torch.zeros(
                    buf.shape[0], padding_size, buf.shape[-1], device=self.device
                )
                padded_bufs[key] = torch.cat(
                    [padding, buf[:, : self.num_transitions]], dim=1
                )

        return padded_bufs

    def _generate_valid_indices(
        self, reset_data: torch.Tensor, sequence_length: int
    ):
        """Generate valid start indices that don't cross reset boundaries.

        Args:
            reset_data: Reset flags, shape [num_envs, buffer_size] or [num_envs, num_transitions]
            sequence_length: Length of sequences to sample

        Returns:
            Tuple of (env_indices, start_indices) where each sequence is valid
        """
        # Ensure reset_data has the right shape
        if reset_data.shape[1] < max(self.num_transitions, sequence_length):
            # Pad if needed
            pad_size = max(self.num_transitions, sequence_length) - reset_data.shape[1]
            padding = torch.zeros(
                reset_data.shape[0], pad_size, device=self.device, dtype=reset_data.dtype
            )
            reset_data = torch.cat([padding, reset_data], dim=1)

        # Extract the relevant portion
        max_len = max(self.num_transitions, sequence_length)
        reset_flags = reset_data[:, :max_len].to(torch.bool)

        # Use unfold to create sliding windows and check if any contain a reset
        # unfold(1, sequence_length, 1) creates windows of size sequence_length
        windows = reset_flags.unfold(1, sequence_length, 1)  # [num_envs, num_windows, sequence_length]

        # Check if any window contains a reset (except the last position which is the next state)
        # We want sequences that don't have resets in positions [0, sequence_length-1)
        valid_mask = ~windows[:, :, :-1].any(dim=2)  # [num_envs, num_windows]

        # Get valid (env_idx, start_idx) pairs
        env_indices, start_indices = torch.where(valid_mask)
        return env_indices, start_indices

    def _generate_batch(
        self,
        replay_bufs: dict,
        valid_indices: tuple | None,
        sequence_length: int,
        mini_batch_size: int,
    ):
        """Generate a mini-batch by sampling sequences.

        Args:
            replay_bufs: Dictionary of buffers
            valid_indices: Tuple of (env_indices, start_indices) if reset-aware sampling is used
            sequence_length: Length of sequences
            mini_batch_size: Size of mini-batch

        Returns:
            Tuple of batches for each field
        """
        if valid_indices is None:
            # Simple random sampling without reset awareness
            max_start_idx = max(self.num_transitions - sequence_length, 0) + 1
            sampled_envs = torch.tensor(
                np.random.choice(self.num_envs, size=mini_batch_size),
                device=self.device,
            )
            sampled_starts = torch.tensor(
                np.random.choice(max_start_idx, size=mini_batch_size),
                device=self.device,
            )
        else:
            # Sample from valid indices (avoiding reset boundaries)
            env_indices, start_indices = valid_indices
            if len(env_indices) == 0:
                # No valid sequences, fall back to simple sampling
                max_start_idx = max(self.num_transitions - sequence_length, 0) + 1
                sampled_envs = torch.tensor(
                    np.random.choice(self.num_envs, size=mini_batch_size),
                    device=self.device,
                )
                sampled_starts = torch.tensor(
                    np.random.choice(max_start_idx, size=mini_batch_size),
                    device=self.device,
                )
            else:
                sampled_idxs = torch.tensor(
                    np.random.choice(len(env_indices), size=mini_batch_size),
                    device=self.device,
                )
                sampled_envs = env_indices[sampled_idxs]
                sampled_starts = start_indices[sampled_idxs]

        # Create offsets for sequence extraction
        offsets = torch.arange(sequence_length, device=self.device)

        # Extract sequences using advanced indexing
        def _extract_sequence(buf):
            if buf is None:
                return None
            # buf: [num_envs, buffer_size, dim]
            # sampled_envs: [mini_batch_size]
            # sampled_starts: [mini_batch_size]
            # offsets: [sequence_length]
            # Result: [mini_batch_size, sequence_length, dim]
            return buf[sampled_envs[:, None], sampled_starts[:, None] + offsets]

        return (
            _extract_sequence(replay_bufs["dynamic"]),
            _extract_sequence(replay_bufs["action"]),
            _extract_sequence(replay_bufs["extension"]),
            _extract_sequence(replay_bufs["contact"]),
            _extract_sequence(replay_bufs["termination"]),
            _extract_sequence(replay_bufs["reward"]),
        )

    def _num_valid(self) -> int:
        """Return the number of valid transitions in the buffer."""
        return self.buffer_size if self.num_transitions >= self.buffer_size else self.num_transitions

    @staticmethod
    def construct_from_cfg(
        cfg: "DynamicReplayBufferCfg",
        dim_params: dict,
        device: str,
        *args,
        num_envs: int = 1,
        **kwargs,
    ):
        """Construct buffer from config.

        Args:
            cfg: Configuration object
            dim_params: Dictionary with dimension parameters:
                - dynamic_dim: Dimension of dynamic state (required)
                - action_dim: Dimension of action (required)
                - extension_dim: Dimension of extension data (optional, uses cfg default)
                - contact_dim: Dimension of contact data (optional, uses cfg default)
                - termination_dim: Dimension of termination data (optional, uses cfg default)
                - reward_dim: Dimension of reward (optional, uses cfg default)
            device: Device to store buffers on
            num_envs: Number of parallel environments (unused, buffer initializes lazily)
            **kwargs: Additional arguments

        Returns:
            DynamicReplayBuffer instance
        """
        # Required dimensions must be provided in dim_params
        dynamic_dim = dim_params.get("dynamic_dim")
        action_dim = dim_params.get("action_dim")

        if dynamic_dim is None:
            raise ValueError("dynamic_dim must be provided in dim_params")
        if action_dim is None:
            raise ValueError("action_dim must be provided in dim_params")

        return DynamicReplayBuffer(
            dynamic_dim=dynamic_dim,
            action_dim=action_dim,
            extension_dim=dim_params.get("extension_dim", cfg.extension_dim),
            contact_dim=dim_params.get("contact_dim", cfg.contact_dim),
            termination_dim=dim_params.get("termination_dim", cfg.termination_dim),
            reward_dim=dim_params.get("reward_dim", cfg.reward_dim),
            buffer_size=cfg.buffer_size,
            device=device,
            **kwargs,
        )


@configclass
class DynamicReplayBufferCfg:
    """Configuration for DynamicReplayBuffer.

    This buffer stores data with shape [num_envs, buffer_size, dim] - each environment
    has its own buffer row. Supports batch insertion and avoids sampling sequences
    that cross environment reset boundaries.
    """

    class_type: type[DynamicReplayBuffer] = DynamicReplayBuffer

    # Buffer capacity (per environment)
    buffer_size: int = MISSING  # Maximum number of transitions per environment

    # Optional dimension defaults (can be overridden by dim_params)
    extension_dim: int = 0  # Dimension of extension data (0 if not used)
    contact_dim: int = 0  # Dimension of contact data (0 if not used)
    termination_dim: int = 0  # Dimension of termination data (0 if not used)
    reward_dim: int = 1  # Dimension of reward (default 1)

    def construct_from_cfg(
        self, dim_params: dict, device: str, *args, num_envs: int = 1, **kwargs
    ):
        """Construct buffer from config and dimension parameters."""
        return DynamicReplayBuffer.construct_from_cfg(
            self, dim_params, device, num_envs=num_envs, **kwargs
        )
