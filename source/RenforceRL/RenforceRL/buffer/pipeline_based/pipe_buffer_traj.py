from __future__ import annotations

import torch
from typing import NamedTuple, List, Union, Dict, Generator
from RenforceRL import configclass
from dataclasses import MISSING
from typing import Callable

from .pipe_buffer_base import PipeBufferBase, PipeBufferBaseCfg
from RenforceRL.runners.world_model.trainer.utils import get_valid_mask_from_termination


class PipeBufferTraj(PipeBufferBase):

    def __init__(self, cfg: PipeBufferTrajCfg, comp_shapes, num_envs, device, **kwargs):
        self.max_steps = cfg.max_buffer_chunks * cfg.chunk_size
        self.warmup_steps = cfg.warmup_chunks * cfg.chunk_size
        self.sample_size = cfg.sample_size if cfg.sample_size is not None else cfg.chunk_size
        self.sample_stride = cfg.sample_stride if cfg.sample_stride is not None else 1
        super().__init__(cfg, comp_shapes=comp_shapes, num_envs=num_envs, device=device)

    # ------------------------------------------------------- #
    #        Create flat buffer: [L, D]
    # ------------------------------------------------------- #
    def create_buffer(self):
        c_buffer = lambda shape, dtype=torch.float32: torch.zeros(
            shape, dtype=dtype, device=self.device
        )

        comp_shapes = [
            shape if isinstance(shape, tuple) else (shape,)
            for shape in self.comp_shapes
        ]

        self.comp_shapes_full = [[self.max_steps, *shape] for shape in comp_shapes]
        self.length = 0
        self.ptr = 0

        for idx, cname in enumerate(self.cfg.component_cfg.comp_names):
            dtype = getattr(torch, self.cfg.component_cfg.comp_dtype[idx])
            shape = self.comp_shapes_full[idx]
            setattr(self, f"{cname}_buffer", c_buffer(shape, dtype=dtype))

    def clear(self):
        self.length = 0
        self.ptr = 0
        for cname in self.cfg.component_cfg.comp_names:
            getattr(self, f"{cname}_buffer").zero_()

    def _add2buffer_traj(self, data: torch.Tensor, buffer: torch.Tensor):
        T = data.shape[0]
        if self.ptr + T <= self.max_steps:
            buffer[self.ptr:self.ptr + T] = data
        else:
            first = self.max_steps - self.ptr
            buffer[self.ptr:] = data[:first]
            buffer[:T - first] = data[first:]
        self.ptr = (self.ptr + T) % self.max_steps
        self.length = min(self.length + T, self.max_steps)

    def add_traj(self, samples: Dict[str, torch.Tensor]):
        B = None
        T = None
        for cname in self.cfg.component_cfg.comp_names:
            x = samples[cname]
            assert x.dim() >= 2, f"{cname} must be [B, T, ...]"
            if B is None:
                B, T = x.shape[:2]
            else:
                assert x.shape[0] == B and x.shape[1] == T
            if x.dim() == 2:
                samples[cname] = x.unsqueeze(-1)

        for cname in self.cfg.component_cfg.comp_names:
            x = samples[cname]           # [B, T, D]
            x = x.reshape(B*T, *x.shape[2:])
            buf = getattr(self, f"{cname}_buffer")
            self._add2buffer_traj(x, buf)

    def sample_batch_seq(self, batch_size: int, K=None) -> Dict[str, torch.Tensor]:
        """
        Sample a batch of sequential data with length K.
        No try/except; missing fields raise errors naturally.
        """
        assert self.length >= self.warmup_steps, \
            f"Insufficient data: need warmup {self.warmup_steps}, have {self.length}"

        K = K if K is not None else self.sample_size
        assert self.length >= K, f"length {self.length} < K {K}"

        starts = self._get_valid_start_indices(K)
        assert starts.numel() > 0, "No valid start indices."

        perm = torch.randperm(starts.numel(), device=self.device)
        batch_starts = starts[perm[:batch_size]]  # [B]

        idx = self._compute_indices(batch_starts, K)  # [B, K]

        batch = self._gather_batch(idx)
        batch["is_valid"] = self._compute_valid_mask(batch)

        return batch

    def _get_valid_start_indices(self, K):
        stride = self.sample_stride

        if stride > 1:
            max_start = max((self.length - K) // stride, 0) * stride
            return torch.arange(0, max_start, stride, device=self.device)
        else:
            max_start = max(self.length - K, 0)
            return torch.arange(max_start, device=self.device)

    def _compute_indices(self, batch_starts, K):
        L = self.max_steps
        arange_K = torch.arange(K, device=self.device)
        return (batch_starts[:, None] + arange_K[None, :]) % L

    def _gather_batch(self, idx):
        batch = {}
        for cname in self.cfg.component_cfg.comp_names:
            buf = getattr(self, f"{cname}_buffer")
            batch[cname] = buf[idx]
        return batch

    def _compute_valid_mask(self, batch):
        B, K = batch[list(batch.keys())[0]].shape[:2]
        is_valid = torch.ones((B, K), dtype=torch.bool, device=self.device)

        if "termination" in batch:
            tm = batch["termination"].squeeze(-1)
            is_valid = torch.logical_and(get_valid_mask_from_termination(tm), is_valid)

        if "timeout" in batch:
            tt = batch["timeout"].squeeze(-1)
            is_valid = torch.logical_and(get_valid_mask_from_termination(tt), is_valid)

        return is_valid.unsqueeze(-1)

    def mini_batch_generator(self, num_epochs: int, batch_size: int, K=None, **kwargs):
        assert self.length >= self.warmup_steps

        L = self.max_steps
        K = K if K is not None else self.sample_size
        stride = self.sample_stride

        if stride > 1:
            max_start = max((self.length - K) // stride, 0) * stride
            starts_all = torch.arange(0, max_start, stride, device=self.device)
        else:
            max_start = max(self.length - K, 0)
            starts_all = torch.arange(max_start, device=self.device)

        arange_K = torch.arange(K, device=self.device)

        for _ in range(num_epochs):
            perm = torch.randperm(starts_all.numel(), device=self.device)
            starts_shuffled = starts_all[perm]

            for i in range(0, starts_shuffled.numel(), batch_size):
                batch_starts = starts_shuffled[i:i + batch_size]
                if batch_starts.numel() == 0:
                    continue

                idx = (batch_starts[:, None] + arange_K[None, :]) % L

                batch_ret = {}
                for cname in self.cfg.component_cfg.comp_names:
                    buf = getattr(self, f"{cname}_buffer")
                    batch_ret[cname] = buf[idx]

                is_valid = torch.ones_like(batch_ret["reward"], dtype=torch.bool)

                if 'termination' in batch_ret:
                    tm = batch_ret['termination'].squeeze(-1)
                    is_valid = torch.logical_and(get_valid_mask_from_termination(tm), is_valid)

                if 'timeout' in batch_ret:
                    tt = batch_ret['timeout'].squeeze(-1)
                    is_valid = torch.logical_and(get_valid_mask_from_termination(tt), is_valid)

                batch_ret['is_valid'] = is_valid.unsqueeze(-1)
                yield batch_ret


@configclass
class PipeBufferTrajCfg(PipeBufferBaseCfg):
    class_type: type[PipeBufferBase] = PipeBufferTraj
    
    sample_size: Union[int, None] = None
    sample_stride: Union[int, None] = None
