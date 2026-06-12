from __future__ import annotations

import torch
from abc import ABC, abstractmethod
from typing import NamedTuple, List, Union, Dict, Generator
from RenforceRL import configclass
from dataclasses import MISSING
from typing import Callable
from .pipe_buffer_base import PipeBufferBase, PipeBufferBaseCfg 

from RenforceRL.runners.world_model.trainer.utils import get_valid_mask_from_termination
from RenforceRL.utils.isaaclab.trajectory import load_hdf5_trajectories, DATA_TRAJ_MAPPING

class PipeBufferStep(PipeBufferBase):
    cfg: PipeBufferStepCfg
    def __init__(
            self, cfg: PipeBufferStepCfg,
            comp_shapes, num_envs, device, **kwargs
        ):
        """
        Replay Buffer at step level, but samples at chunk level. 
        Stores data in a (num_envs, max_steps, D) structure.
        """
        self.warmup_steps = cfg.warmup_chunks * cfg.chunk_size * num_envs
        self.max_steps = cfg.max_buffer_chunks * cfg.chunk_size * num_envs
        self.max_step_per_env = cfg.max_buffer_chunks * cfg.chunk_size
        super().__init__(cfg, comp_shapes=comp_shapes, num_envs=num_envs, device=device)

        # Ensure termination is boolean
        if hasattr(self, 'termination_buffer'):
            self.termination_buffer = self.termination_buffer.to(torch.bool)
        
        # Initialize step pointers for each environment
        self.clear()
    
    # Override base clear to ensure env pointers are reset
    def clear(self):
        self.length = 0 # Total steps
        self.env_pointers = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self.env_lengths = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        
    def create_buffer(self):
        """Initializes the (Num_Envs, Max_Steps, D) buffers."""
        c_buffer = lambda shape, dtype=torch.float32, func=torch.empty: func(
            shape, dtype=dtype, device=self.device, requires_grad=False
        )

        comp_shapes = [shape if isinstance(shape, tuple) else (shape,) for shape in self.comp_shapes]
        
        # Buffer shape: (num_envs, max_steps, *D)
        self.comp_shapes_full = [[self.num_envs, self.max_step_per_env, *shape] for shape in comp_shapes]
        
        for idx, cname in enumerate(self.cfg.component_cfg.comp_names):
            cshape = self.comp_shapes_full[idx]
            dtype = getattr(torch, self.cfg.component_cfg.comp_dtype[idx])
            setattr(self, f"{cname}_buffer", c_buffer(cshape, dtype=dtype))

    def _add2buffer_sequence(self, data, buffer, env_idx, start, end):
        """Adds a sequence (T steps) of data for a single environment."""
        # data shape: (T, *D)
        # buffer shape: (num_envs, max_steps, *D)
        
        dlen = end - start
        
        # Circular wrap logic (not needed if T <= max_steps, but good practice)
        if end <= self.max_step_per_env:
            buffer[env_idx, start:end, ...] = data[:dlen, ...]
        else:
            # Wrap around: (max_steps - start) steps at the end, (end - max_steps) steps at the start
            wrap_len = self.max_step_per_env - start
            buffer[env_idx, start:self.max_step_per_env, ...] = data[:wrap_len, ...]
            buffer[env_idx, 0:end - self.max_step_per_env, ...] = data[wrap_len:dlen, ...]

    def _update_pointers_and_length(self, env_idx: torch.Tensor, steps_added: int):
        """Updates pointers and total length after adding data for specific environments."""
        # Update individual environment pointers
        self.env_pointers[env_idx] = (self.env_pointers[env_idx] + steps_added) % self.max_step_per_env
        
        # Update individual environment lengths (capped at max_steps)
        self.env_lengths[env_idx] = torch.min(self.env_lengths[env_idx] + steps_added, torch.tensor(self.max_step_per_env, device=self.device))
        
        # Update total length
        self.length = torch.sum(self.env_lengths).item()

    # --- 1. Add [B, D] Steps ---

    def add_steps(self, samples: Dict[str, torch.Tensor]):
        """
        Adds a batch of steps (B_envs, D) collected from parallel environments.
        
        Args:
            samples (Dict[str, torch.Tensor]): Dictionary of components, each of shape (B_envs, *D).
        """
        B_envs = samples[self.cfg.component_cfg.comp_names[0]].shape[0]
        
        # If the incoming batch is smaller than num_envs, it's an error in typical parallel setups
        assert B_envs == self.num_envs, "Input batch size must match num_envs for add_steps."

        # Get the current pointers for all environments
        current_pointers = self.env_pointers

        for name in self.cfg.component_cfg.comp_names:
            data = samples[name] # Shape (B_envs, *D)
            buffer = getattr(self, f"{name}_buffer") # Shape (B_envs, L_max, *D)
            
            if len(data.shape) == 1:
                data = data.reshape(-1, 1)
            
            # Use advanced indexing to place one step (at T=pointer) for all B_envs simultaneously
            # The indices are [Env_idx, Step_idx, ...]
            # Env_idx: torch.arange(B_envs)
            # Step_idx: current_pointers
            buffer[torch.arange(B_envs), current_pointers, ...] = data

        # All environments added 1 step
        self._update_pointers_and_length(torch.arange(B_envs), steps_added=1)

    # --- 2. Add [B, T, D] Steps (Chunk/Sequence) ---

    def add_chunk(self, samples: Dict[str, torch.Tensor]):
        # TODO ERROR! do not use this
        ...


    def add_trajectory(self, env_idx: int, traj: Dict[str, torch.Tensor]):
        """
        Add a single trajectory (T, D) into a specific environment slot in the buffer.
        Internally it uses the same circular-buffer logic as add_chunk, but restricted
        to one environment.

        Args:
            env_idx (int): Which environment index to write into.
            traj (Dict[str, torch.Tensor]): Dict of arrays of shape (T, D) or (T,)
        """

        assert 0 <= env_idx < self.num_envs, f"env_idx {env_idx} out of range (num_envs={self.num_envs})"

        # --- Prepare shapes: convert (T, D) into (T, *D) consistently ---
        T = None
        for k, v in traj.items():
            if T is None:
                T = v.shape[0]
            else:
                assert v.shape[0] == T, "All fields must have same T length."

        # --- Insert trajectory into circular buffer for this env ---
        start_ptr = self.env_pointers[env_idx].item()
        end_ptr = start_ptr + T

        for name in self.cfg.component_cfg.comp_names:
            data = traj[name]
            if data.ndim == 1:
                data = data[:, None]      # (T,) -> (T,1)

            buffer = getattr(self, f"{name}_buffer")  # (num_envs, max_steps, *D)

            if end_ptr <= self.max_step_per_env:
                # Continuous write
                buffer[env_idx, start_ptr:end_ptr, ...] = data
            else:
                # Wrap around
                wrap_len = self.max_step_per_env - start_ptr
                buffer[env_idx, start_ptr:self.max_step_per_env, ...] = data[:wrap_len]
                buffer[env_idx, 0:T - wrap_len, ...] = data[wrap_len:]

        # --- Update pointers and lengths (only for env_idx) ---
        self._update_pointers_and_length(
            torch.tensor([env_idx], device=self.device), 
            steps_added=T
        )


    # --- Mini-Batch Generator (World Model Sequence Sampling) ---
    
    def mini_batch_generator(self, num_epochs: int, batch_size: int, **kwargs) -> Generator[Dict[str, torch.Tensor], None, None]:
        """
        Generates mini-batches (sequences of length self.chunk_size) for World Model training.
        Fully vectorized implementation.
        """
        # 1. Warmup Check
        if self.length < self.warmup_chunks:
            return 

        K = self.chunk_size

        # 2. Calculate valid sampling range per environment
        # If buffer is full (circular), we can start anywhere (0 to max_steps-1).
        # If not full (linear), we can start up to (length - chunk_size).
        # Shape: (num_envs,)
        valid_range_ends = torch.where(
            self.env_lengths == self.max_step_per_env,
            self.max_step_per_env, # Circular: allow wrapping
            self.env_lengths - K + 1 # Linear: strict bound
        )
        
        # Filter out environments that don't have enough data yet
        valid_env_mask = valid_range_ends > 0
        valid_env_indices = torch.nonzero(valid_env_mask).squeeze(-1) # Indices of envs with enough data
        
        if valid_env_indices.numel() == 0:
            return

        # Estimate batches per epoch based on total valid chunks
        total_valid_chunks = valid_range_ends.sum().item()
        batches_per_epoch = max(1, int(total_valid_chunks // batch_size))

        # Pre-calculate time offsets for the chunk: [0, 1, ..., chunk_size-1]
        # Shape: (1, chunk_size)
        time_offsets = torch.arange(K, device=self.device).unsqueeze(0)

        for epoch in range(num_epochs):
            for _ in range(batches_per_epoch):
                # --- A. Vectorized Sampling of Indices ---
                
                # 1. Sample Environment Indices randomly from valid environments
                # Shape: (batch_size,)
                rand_env_select = torch.randint(0, valid_env_indices.numel(), (batch_size,), device=self.device)
                batch_env_idx = valid_env_indices[rand_env_select]

                # 2. Sample Start Steps for each selected environment
                # Get the valid range for each selected env
                batch_valid_ranges = valid_range_ends[batch_env_idx]
                # Sample random start step: [0, range)
                batch_start_step = (torch.rand(batch_size, device=self.device) * batch_valid_ranges).long()

                # 3. Construct Time Indices Grid (Handling Wrapping via Modulo)
                # Shape: (batch_size, chunk_size)
                # base_start (B, 1) + offset (1, T) -> (B, T)
                raw_time_indices = batch_start_step.unsqueeze(1) + time_offsets
                batch_time_idx = raw_time_indices % self.max_step_per_env # Handle circular wrapping automatically

                # --- B. Vectorized Data Gathering ---
                
                batch_data = {}
                
                # Expand env indices to match time indices for gather/indexing
                # Shape: (batch_size, chunk_size)
                batch_env_idx_expanded = batch_env_idx.unsqueeze(1).expand(-1, K)

                for name in self.cfg.component_cfg.comp_names:
                    buffer = getattr(self, f"{name}_buffer") # (num_envs, max_steps, D)
                    
                    # Advanced Indexing: buffer[ (B,T), (B,T) ] -> (B, T, D)
                    # This replaces the slow Python loop and cat
                    batch_data[name] = buffer[batch_env_idx_expanded, batch_time_idx]

                # --- C. Validity Masking ---
                
                is_valid = torch.ones_like(batch_data["reward"], dtype=torch.bool)
                
                # Check termination to mask out transitions across episodes
                if 'termination' in batch_data:
                    termination_batch = batch_data['termination'].squeeze(-1)
                    is_valid = torch.logical_and(get_valid_mask_from_termination(termination_batch), is_valid)
                
                if 'timeout' in batch_data:
                    timeout_batch = batch_data['timeout'].squeeze(-1)
                    is_valid = torch.logical_and(get_valid_mask_from_termination(timeout_batch), is_valid)

                # Also mask out the "Seam" (Write Pointer Crossing) if needed
                # If we wrapped around, we might have crossed the write pointer (oldest -> newest data jump).
                # However, usually termination signals handle this naturally. 
                # If strict correctness is needed, we would check if batch_time_idx crosses env_pointers[batch_env_idx].
                
                batch_data['is_valid'] = is_valid.unsqueeze(-1)
                
                yield batch_data
                
    def init_data_loader(self, dataset_path: str):
        self.dataset_path = dataset_path
        self.data_generator = self._infinite_loader(dataset_path)
        self.load_ptr = 0
        print(f"[ReplayBuffer] Data loader initialized from: {dataset_path}")

    def _infinite_loader(self, path):
        while True:
            gen = load_hdf5_trajectories(path)
            count = 0
            for sample in gen:
                yield sample
                count += 1
            if count == 0:
                raise ValueError(f"Dataset at {path} is empty!")
            print(f"[ReplayBuffer] Dataset exhausted (read {count} trajs), restarting loop...")

    def stream_data(self, num_trajectories: int):
        if not hasattr(self, 'data_generator'):
            raise RuntimeError("Data loader not initialized! Call init_data_loader() first.")

        for _ in range(num_trajectories):
            sample = next(self.data_generator)
            
            data = {
                DATA_TRAJ_MAPPING.get(k, k): torch.from_numpy(v).to(torch.float32)
                for k, v in sample.items()
                if DATA_TRAJ_MAPPING.__contains__(k)
            }
            
            self.add_trajectory(self.load_ptr, data)
            self.load_ptr = (self.load_ptr + 1) % self.num_envs

@configclass
class PipeBufferStepCfg(PipeBufferBaseCfg):
    class_type: type[PipeBufferBase] = PipeBufferStep