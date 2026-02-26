from __future__ import annotations

import torch
from abc import ABC, abstractmethod
from typing import Dict, NamedTuple, List, Union
from GopsEA import configclass
from dataclasses import MISSING
from typing import TYPE_CHECKING
from GopsEA.utils.template.module_base import ModuleBase, ModuleBaseCfg
from GopsEA.buffer.replay_buffer_base import ReplayBufferBase, ReplayBufferBaseCfg

if TYPE_CHECKING:
    from .component_cfg import ComponentCfg

class PipeBufferBase(ReplayBufferBase):
    cfg: PipeBufferBaseCfg
    component_cfg: "ComponentCfg"
    
    max_steps: int = None
    warmup_steps: int = None
    max_buffer_chunks: int = None
    warmup_chunks: int = None
    chunk_size: int = None
    
    def __init__(self, 
                cfg: PipeBufferBaseCfg,
                comp_shapes,
                component_cfg,
                num_envs,
                device, 
                **kwargs):
        """
        Replay Buffer for storing and sampling experiences in reinforcement learning.
        
        Arguments:
            device (torch.device): Device (CPU or GPU) to store the buffer
            
        Notes:
            The buffer is construct at shape (B, L, D).
            Where the L is the chunck size for world model max prediction.
        """
        super().__init__(cfg, device)
        # Ensure no variable is None during initialization
        # assert all(value is not None for value in locals().values()), "Some local variables are None"
        
        if kwargs: 
            print(f"Warning: Unused kwargs: {list(kwargs.keys())}")  # Warning for any unused arguments

        self.cfg = cfg
        self.component_cfg = component_cfg
        self.num_envs = num_envs

        # Assign values to object attributes
        self.max_buffer_chunks = cfg.max_buffer_chunks
        self.warmup_chunks = cfg.warmup_chunks
        self.chunk_size = cfg.chunk_size
        self.num_envs = num_envs

        self.comp_shapes = comp_shapes
        self.device = device
        
        # Create the buffer
        self.create_buffer()

        # Initialize buffer-related variables
        self.clear()
        self.fullidx = False

    @property
    def dim_params(self):
        return {k + "_dim": v for k, v in zip(self.component_cfg.comp_names, self.comp_shapes)}

    @staticmethod
    def construct_from_cfg(cfg: PipeBufferBaseCfg, dim_params: dict, device, *args, num_envs=1):
        comp_shapes = cfg.component_cfg.get_shape(**dim_params)
        replay_buffer: PipeBufferBase = \
            cfg.class_type(cfg=cfg, comp_shapes=comp_shapes, num_envs=num_envs, device=device)
        return replay_buffer

    @abstractmethod
    def create_buffer(self):
        ...

    @abstractmethod
    def add_traj(self, samples: Dict[str, torch.Tensor]):
        ...

    @abstractmethod
    def add_trans(self, samples: Dict[str, torch.Tensor]):
        """Step only batch"""
        ...

    @abstractmethod
    def add_steps(self, samples: Dict[str, torch.Tensor]):
        """Step with num envs"""
        ...
        
    def add_chunk(self, samples: Dict[str, torch.Tensor]):
        ...

    @abstractmethod
    def mini_batch_generator(self, num_epochs: int, batch_size: int, **kwargs):
        ...

    def sample_batch_seq(self, batch_size: int) -> Dict[str, torch.Tensor]:
        raise NotImplementedError("sample_batch_seq not implemented in base class.")

    def clear(self):
        self.length = 0
        self.last_pointer = 0
        
    def is_warmingup(self) -> bool:
        """Return True if buffer has not yet reached warmup_length."""
        """Warm up should reach certain chuck level, where we sample at chunck."""
        return self.length < self.warmup_steps

    def is_full(self) -> bool:
        """Return True if buffer is completely full."""
        return self.length >= self.max_steps

    def report(self) -> dict:
        """
        Return a dictionary containing replay buffer statistics.
        """
        total_capacity = self.max_steps
        total_length = int(self.length)
        remaining = int(total_capacity - total_length)

        return {
            "total_length": total_length,
            "total_capacity": total_capacity,
            "utilization": total_length / total_capacity,
            "is_full": self.is_full(),
            "is_warmingup": self.is_warmingup(),
            "remaining_space": remaining,
        }

    def pretty_report(self, shown_num=1) -> str:
        """
        Return a human-readable string describing the replay buffer status.
        """
        info = self.report()

        total_length = info["total_length"]
        total_capacity = info["total_capacity"]
        utilization = info["utilization"]
        remaining = info["remaining_space"]

        is_full = info["is_full"]
        is_warmingup = info["is_warmingup"]

        lines = []
        lines.append("=" * 60)
        lines.append("Replay Buffer Status")
        lines.append("=" * 60)
        lines.append(f"Total Capacity : {total_capacity}")
        lines.append(f"Total Length   : {total_length}")
        lines.append(f"Utilization    : {utilization*100:.2f}%")
        lines.append(f"Remaining Space: {remaining}")
        lines.append(f"Is Full        : {is_full}")
        lines.append(f"Is Warmup      : {is_warmingup}  (warmup_steps={self.warmup_steps})")
        lines.append("-" * 60)
        return "\n".join(lines)

    def sample_batch(self, batch_size):
        """
        Sample a batch of individual transitions from the buffer.
        Returns a dict with keys as component names, values as tensors of shape (batch_size, ...)
        """
        if self.length < batch_size:
            raise ValueError("Not enough data in buffer")
        
        # Sample random indices
        indices = torch.randint(0, self.length, (batch_size,), device=self.device)
        
        batch = {}
        for cname in self.component_cfg.comp_names:
            buf = getattr(self, f"{cname}_buffer")
            # buf shape: (max_chunks, chunk_size, ...)
            # Flatten to (max_chunks * chunk_size, ...)
            flat_buf = buf.view(-1, *buf.shape[2:])
            batch[cname] = flat_buf[indices]
        return batch

@configclass
class PipeBufferBaseCfg(ReplayBufferBaseCfg):
    class_type: type[PipeBufferBase] = PipeBufferBase

    max_buffer_chunks: int = MISSING
    # Maximum size of the buffer
    warmup_chunks: int = MISSING
    # The number of samples required to fill the buffer before training
    chunk_size: int = MISSING
    # recomended for match the transformer max length

    def construct_from_cfg(self, component_cfg:ComponentCfg, dim_params: dict, device, *args, num_envs=1, **kwargs):
        comp_shapes = component_cfg.get_shape(**dim_params)
        replay_buffer: PipeBufferBase = \
            self.class_type(cfg=self, component_cfg=component_cfg, comp_shapes=comp_shapes, num_envs=num_envs, device=device)
        return replay_buffer