from __future__ import annotations

import torch
from typing import Dict, Literal, NamedTuple, List, Union
from GopsEA import configclass
from dataclasses import MISSING
from typing import Callable

from GopsEA.buffer.pipeline_based.pipe_buffer_base import PipeBufferBase, PipeBufferBaseCfg
from . import rep_func

class PipeBufferChunk(PipeBufferBase):
    cfg: PipeBufferChunkCfg

    def __init__(
        self,
        cfg: PipeBufferChunkCfg,
        *args,
        **kwargs
    ):
        """
        Chunk-level replay buffer:
        - Stores [max_buffer_chunks, chunk_size, ...]
        - add_chunk() will write chunks at indexes chosen by a replacement policy.
        """
        self.max_steps         = cfg.max_buffer_chunks * cfg.chunk_size
        self.warmup_steps      = cfg.warmup_chunks * cfg.chunk_size

        super().__init__(cfg, *args, **kwargs)

        self.valid_chunks = 0
        self.chunk_weights = torch.ones(
            self.max_buffer_chunks, device=self.device, dtype=torch.float32
        )
        self.init_rep_fn()
        
    rep_fn: Callable = None
    def init_rep_fn(self):
        if self.cfg.replace_mode == "fifo": 
            self.rep_fn = rep_func.rep_fn_fipo
        elif self.cfg.replace_mode == "weighted": 
            self.rep_fn = rep_func.rep_fn_weighted

    def create_buffer(self):
        """Initializes the (max_chunks, chunk_size, D) buffers."""
        c_buffer = lambda shape, dtype=torch.float32, func=torch.empty: func(
            shape, dtype=dtype, device=self.device, requires_grad=False
        )
        comp_shapes = [
            shape if isinstance(shape, tuple) else (shape,) for shape in self.comp_shapes
        ]
        self.comp_shapes_full = [
            [self.max_buffer_chunks, self.chunk_size, *shape] for shape in comp_shapes
        ]
        for idx, cname in enumerate(self.component_cfg.comp_names):
            cshape = self.comp_shapes_full[idx]
            dtype  = getattr(torch, self.component_cfg.comp_dtype[idx])
            setattr(self, f"{cname}_buffer", c_buffer(cshape, dtype=dtype))

    def select_replace_indices(self, num_new: int) -> torch.Tensor:
        """
        Decide which chunk indices to write new data into.

        Default behavior:
            - If buffer not full: sequential fill [valid_chunks, valid_chunks+num_new)
            - If full: FIFO by default OR weighted sampling (configurable)

        NOTE: This method is the expected extension point for algorithms
              that require different replacement policies.
        """
        if self.valid_chunks < self.max_buffer_chunks:
            # Still space → append sequentially
            start = self.valid_chunks
            end   = min(self.valid_chunks + num_new, self.max_buffer_chunks)
            idx = torch.arange(start, end, device=self.device)
            if end - start < num_new:
                idx = torch.concat([idx, self.rep_fn(self, num_new)], dim=0)
            return idx

        return self.rep_fn(self, num_new)

    def add_chunk(self, samples: Dict[str, torch.Tensor]):
        """
        samples[name]: shape (B_env, chunk_size, ...)
        """
        B_envs, T = samples[self.component_cfg.comp_names[0]].shape[:2]
        assert T == self.chunk_size, f"Chunk size mismatch: got {T}, expect {self.chunk_size}"

        # Step 1: choose write positions
        replace_ids = self.select_replace_indices(B_envs)

        # Step 2: write data to those chunk positions
        for cname in self.component_cfg.comp_names:
            buf  = getattr(self, f"{cname}_buffer")
            data = samples[cname]     # (B_env, T, ...)
            buf[replace_ids] = data

        # Update counters
        if self.valid_chunks < self.max_buffer_chunks:
            self.valid_chunks = min(self.valid_chunks + B_envs, self.max_buffer_chunks)

        self.pointer = (self.pointer + B_envs) % self.max_buffer_chunks
        self.length  = self.valid_chunks * self.chunk_size

    def mini_batch_generator(
        self,
        num_epochs,
        batch_size,
        max_batches_per_epoch: int = None,
        **kwargs
    ):
        """
        Return batches:
            batch[cname]: (B_chunk, chunk_size, ...)
        """
        num_chunks = self.valid_chunks
        assert num_chunks > 0, "ReplayBuffer is empty; call add_chunk() first."

        chunk_indices = torch.arange(num_chunks, device=self.device)

        for _ in range(num_epochs):
            perm = torch.randperm(num_chunks, device=self.device)
            chunk_indices = chunk_indices[perm]
            batches_yielded = 0
            for start in range(0, num_chunks, batch_size):
                if max_batches_per_epoch is not None and batches_yielded >= max_batches_per_epoch:
                    break
                mb_ids = chunk_indices[start:start + batch_size]
                batch = {}
                for cname in self.component_cfg.comp_names:
                    buf = getattr(self, f"{cname}_buffer")  # (N, T, ...)
                    batch[cname] = buf[mb_ids]
                batches_yielded += 1
                yield batch

    def mini_trans_batch_generator(
        self,
        num_epochs,
        batch_size,
        max_batches_per_epoch: int = None,
        **kwargs
    ):
        """
        Return batches:
            batch[cname]: (B, 2, ...)
        """
        num_chunks = self.valid_chunks
        assert num_chunks > 0, "ReplayBuffer is empty; call add_chunk() first."

        T = self.chunk_size
        device = self.device
        for _ in range(num_epochs):
            batches_yielded = 0
            total_transitions = num_chunks * (T - 1)
            perm = torch.randperm(total_transitions, device=device)
            for start in range(0, total_transitions, batch_size):
                if max_batches_per_epoch is not None and batches_yielded >= max_batches_per_epoch:
                    break
                ids = perm[start:start + batch_size]
                chunk_ids = ids // (T - 1)
                t_ids = ids % (T - 1)
                batch = {}
                for cname in self.component_cfg.comp_names:
                    buf = getattr(self, f"{cname}_buffer")  # (N_chunk, T, ...)
                    batch[cname] = torch.stack(
                        [
                            buf[chunk_ids, t_ids],
                            buf[chunk_ids, t_ids + 1]
                        ],
                        dim=1
                    )
                batches_yielded += 1
                yield batch

    def clear(self):
        self.valid_chunks = 0
        self.length = 0
        self.pointer = 0


@configclass
class PipeBufferChunkCfg(PipeBufferBaseCfg):
    class_type: type[PipeBufferBase] = PipeBufferChunk

    max_buffer_chunks: int = 4096 * 2
    chunk_size: int = 32
    warmup_chunks: int = 10
    
    replace_mode: rep_func.RepFuncTypes = "fifo"
