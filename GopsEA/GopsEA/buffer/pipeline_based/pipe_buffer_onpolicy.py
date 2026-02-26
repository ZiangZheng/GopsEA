from __future__ import annotations

import torch
from abc import ABC, abstractmethod
from typing import Dict, NamedTuple, List, Union
from GopsEA import configclass
from dataclasses import MISSING
from typing import Callable
from .pipe_buffer_base import PipeBufferBase, PipeBufferBaseCfg

class PipeBufferOnpolicy(PipeBufferBase):
    cfg: PipeBufferOnpolicyCfg
    def __init__(
            self, cfg: PipeBufferOnpolicyCfg,
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

    def add_chunk(self, samples: Dict[str, torch.Tensor]):
        B_envs, T = samples[self.cfg.component_cfg.comp_names[0]].shape[:2]
        assert B_envs == self.num_envs, "Input batch size must match num_envs for add_chunk."
        for name in self.cfg.component_cfg.comp_names:
            data = samples[name] # Shape (B_envs, *D)
            buffer = getattr(self, f"{name}_buffer") # Shape (B_envs, L_max, *D)
            buffer[:, :T] = data
            
        self.length = B_envs * T
        self.pointer = T
        
    def mini_batch_generator(self, num_epochs, mini_batch_size, **kwargs):
        """
        Mini-batches at the CHUNK level.
        Each sample is a full trajectory of shape (T, D),
        not step-level data.

        Output batch shapes:
            batch[cname]: (B_env, T, D...)
        """
        num_envs = self.num_envs
        T = self.pointer
        assert T > 0, "ReplayBuffer is empty; call add_chunk() before training."

        env_indices = torch.arange(num_envs, device=self.device)

        for _ in range(num_epochs):
            # Shuffle envs each epoch
            perm = torch.randperm(num_envs, device=self.device)
            env_indices = env_indices[perm]

            # Mini-batch slicing (chunk-level)
            for start in range(0, num_envs, mini_batch_size):
                mb_env_ids = env_indices[start:start + mini_batch_size]

                batch = {}

                for cname in self.cfg.component_cfg.comp_names:
                    buf = getattr(self, f"{cname}_buffer")  # (E, T, ...)
                    # Select entire trajectory for each env
                    batch[cname] = buf[mb_env_ids, :T]       # (B_env, T, ...)

                yield batch

    def clear(self):
        self.length = 0
        self.pointer = 0

@configclass
class PipeBufferOnpolicyCfg(PipeBufferBaseCfg):
    class_type: type[PipeBufferBase] = PipeBufferOnpolicy
