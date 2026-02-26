from __future__ import annotations

import torch
import torch.nn as nn
from typing import Optional, Tuple
from dataclasses import dataclass
from GopsEA import configclass
from GopsEA.utils.template.module_base import ModuleBaseCfg


class SimpleRNN(nn.Module):
    """
    A minimal, high-performance RNN/LSTM/GRU encoder.

    Design goals:
      - Pure RNN (no positional encoding / no caching logic / no projection stacks)
      - Use PyTorch's optimized cudnn RNN kernels
      - Provide clean full-sequence + single-step API
    """

    def __init__(self, cfg: "SimpleRNNCfg", input_dim: int):
        super().__init__()
        self.cfg = cfg
        self.input_dim = input_dim
        self.hidden_size = cfg.hidden_size
        self.rnn_type = cfg.rnn_type

        rnn_cls = {
            "lstm": nn.LSTM,
            "gru": nn.GRU,
            "rnn_tanh": lambda *a, **k: nn.RNN(*a, nonlinearity="tanh", **k),
            "rnn_relu": lambda *a, **k: nn.RNN(*a, nonlinearity="relu", **k),
        }[cfg.rnn_type]

        self.rnn = rnn_cls(
            input_size=self.input_dim,
            hidden_size=self.hidden_size,
            num_layers=cfg.num_layers,
            batch_first=True,
            dropout=cfg.dropout if cfg.num_layers > 1 else 0.0,
            bidirectional=cfg.bidirectional,
        )

        out_dim = self.hidden_size * (2 if cfg.bidirectional else 1)
        self.out_norm = nn.LayerNorm(out_dim)

        self._cached_hidden: Optional[Tuple[torch.Tensor, Optional[torch.Tensor]]] = None
        self._init_weights()

    def _init_weights(self):
        for name, param in self.rnn.named_parameters():
            if "weight_ih" in name:
                nn.init.xavier_uniform_(param.data)
            elif "weight_hh" in name:
                nn.init.orthogonal_(param.data)
            elif "bias" in name:
                nn.init.constant_(param.data, 0.0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, L, input_dim)
        Returns:
            out: (B, L, hidden_size * directions)
        """
        out, _ = self.rnn(x)
        out = self.out_norm(out)
        return out

    def reset_hidden(self, batch_size: int, device: torch.device, dtype=torch.float32):
        num_dirs = 2 if self.cfg.bidirectional else 1
        layers = self.cfg.num_layers * num_dirs
        h = torch.zeros(layers, batch_size, self.hidden_size, device=device, dtype=dtype)
        if self.rnn_type == "lstm":
            c = torch.zeros_like(h)
            self._cached_hidden = (h, c)
        else:
            self._cached_hidden = (h, None)

    def forward_step(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, 1, input_dim)
        Returns:
            (B, 1, hidden_size * directions)
        """
        assert x.shape[1] == 1
        if self._cached_hidden is None:
            self.reset_hidden(x.shape[0], x.device, x.dtype)

        out, new_hidden = self.rnn(x, self._cached_hidden)

        if self.rnn_type == "lstm":
            self._cached_hidden = (new_hidden[0].detach(), new_hidden[1].detach())
        else:
            self._cached_hidden = (new_hidden.detach(), None)

        out = self.out_norm(out)
        return out


@configclass
class SimpleRNNCfg(ModuleBaseCfg):
    class_type: type[nn.Module] = SimpleRNN

    hidden_size: int = 512
    num_layers: int = 1
    bidirectional: bool = False
    dropout: float = 0.0
    rnn_type: str = "lstm"  # lstm/gru/rnn_tanh/rnn_relu
