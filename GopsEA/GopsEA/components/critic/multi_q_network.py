from __future__ import annotations

import torch
import torch.nn as nn
from typing import Literal
from dataclasses import MISSING

from GopsEA import configclass
from GopsEA.utils.template.module_base import ModuleBase, ModuleBaseCfg
from GopsEA.networks.mlp import MLP, MLPCfg


class MultiQNetwork(ModuleBase):
    """
    Multi-head Q-network for clipped double Q-learning.

    Forward returns:
        - all Qs:   [num_q, ..., 1]
        - min Q:    [..., 1]
        - mean Q:   [..., 1]
    """

    def __init__(
        self,
        cfg: MultiQNetworkCfg,
        state_dim: int,
        action_dim: int,
        out_feature: int = 1,
    ):
        super().__init__()

        self.cfg = cfg
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.in_feature = state_dim + action_dim
        self.out_feature = out_feature
        self.num_q = cfg.num_q

        # Independent Q-heads (no parameter sharing)
        self.q_heads = nn.ModuleList([
            cfg.backbone_cfg.class_type(
                in_feature=self.in_feature,
                out_feature=out_feature,
                cfg=cfg.backbone_cfg
            )
            for _ in range(self.num_q)
        ])

    def forward(
        self,
        state: torch.Tensor,
        action: torch.Tensor,
        return_type: Literal["all", "min", "mean"] = "all",
    ) -> torch.Tensor:
        """
        Args:
            state:  (..., state_dim)
            action: (..., action_dim)

        Returns:
            - all:  (num_q, ..., 1)
            - min:  (..., 1)
            - mean: (..., 1)
        """
        sa = torch.cat([state, action], dim=-1)

        qs = torch.stack(
            [q(sa) for q in self.q_heads],
            dim=0
        )  # [num_q, ..., 1]

        if return_type == "all":
            return qs
        elif return_type == "min":
            return qs.min(dim=0).values
        elif return_type == "mean":
            return qs.mean(dim=0)
        else:
            raise ValueError(f"Unknown return_type: {return_type}")

@configclass
class MultiQNetworkCfg(ModuleBaseCfg):
    class_type: type[nn.Module] = MultiQNetwork

    num_q: int = 2   # clipped double Q

    backbone_cfg: MLPCfg = MLPCfg(
        hidden_features=[256, 256],
        activations=[
            [('LayerNorm', {}), ('ReLU', {})],
            [('LayerNorm', {}), ('ReLU', {})],
        ]
    )

    def construct_from_cfg(self, *args, dim_params=None, reward_dim=1, **kwargs):
        return MultiQNetwork(self, dim_params["critic_dim"], dim_params["action_dim"], reward_dim)