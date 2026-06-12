from __future__ import annotations
import torch
from typing import Literal, TYPE_CHECKING

if TYPE_CHECKING:
    from .pipe_buffer_chunk import PipeBufferChunk

RepFuncTypes = Literal["fipo", "weighted"]

def rep_fn_fipo(self: PipeBufferChunk, num_new):
    idx = torch.arange(num_new, device=self.device)
    return (self.pointer + idx) % self.max_buffer_chunks

def rep_fn_weighted(self: PipeBufferChunk, num_new):
    prob = self.chunk_weights / torch.sum(self.chunk_weights)
    return torch.multinomial(prob, num_samples=num_new, replacement=False)
