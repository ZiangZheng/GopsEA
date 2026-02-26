from __future__ import annotations

import torch
from abc import ABC, abstractmethod
from typing import Dict, NamedTuple, List, Union
from GopsEA import configclass
from dataclasses import MISSING
from typing import TYPE_CHECKING
from GopsEA.utils.template import ClassTemplateBase, ClassTemplateBaseCfg


class ReplayBufferBase(ClassTemplateBase):
    cfg: ReplayBufferBaseCfg
    
    max_steps: int = None
    warmup_steps: int = None
    max_buffer_chunks: int = None
    warmup_chunks: int = None
    chunk_size: int = None
    
    def __init__(self, 
                cfg: ReplayBufferBaseCfg,
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
        super().__init__()
        # Ensure no variable is None during initialization
        # assert all(value is not None for value in locals().values()), "Some local variables are None"
        
        if kwargs: 
            print(f"Warning: Unused kwargs: {list(kwargs.keys())}")  # Warning for any unused arguments

        self.cfg = cfg
        self.device = device

    @staticmethod
    def construct_from_cfg(cfg: ReplayBufferBaseCfg, dim_params: dict, device, *args, num_envs=1):
        ...

    @abstractmethod
    def create_buffer(self):
        ...

    @abstractmethod
    def add(self, *args, **kwargs):
        ...

    def add_traj(self, samples: Dict[str, torch.Tensor]):
        ...

    def add_trans(self, samples: Dict[str, torch.Tensor]):
        ...

    def add_steps(self, samples: Dict[str, torch.Tensor]):
        ...
        
    def add_chunk(self, samples: Dict[str, torch.Tensor]):
        ...

    @abstractmethod
    def mini_batch_generator(self, num_epochs: int, batch_size: int, **kwargs):
        ...

    def sample_batch(self, batch_size):
        ...

    def sample_batch_seq(self, batch_size: int) -> Dict[str, torch.Tensor]:
        raise NotImplementedError("sample_batch_seq not implemented in base class.")

    @abstractmethod
    def clear(self):
        ...
        
    def is_warmingup(self) -> bool:
        """Return True if buffer has not yet reached warmup_length."""
        """Warm up should reach certain chuck level, where we sample at chunck."""
        ...

    @abstractmethod
    def is_full(self) -> bool:
        ...

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

    def __str__(self):
        return self.pretty_report()


@configclass
class ReplayBufferBaseCfg(ClassTemplateBaseCfg):
    class_type: type[ReplayBufferBase] = ReplayBufferBase

    def construct_from_cfg(self, dim_params: dict, device, *args, num_envs=1, **kwargs):
        ...