from __future__ import annotations

import torch
import torch.nn as nn
import torch.distributions as D

from GopsEA import configclass
from GopsEA.networks.mlp import MLPCfg
from GopsEA.utils.template.module_base import ModuleBaseCfg
from GopsEA.components.actor.actor_base import ActorBase
from dataclasses import MISSING


class GaussianActor(ActorBase):
    """
    Gaussian policy actor: π(a|s)

    This module directly maps state to a Gaussian (optionally squashed)
    action distribution and supports:
        - reparameterized sampling
        - log-probability computation
        - entropy computation

    Design:
        state
          ↓
        Pi backbone (MLP)
          ↓
        mean, log_std
          ↓
        Normal / Tanh-Normal distribution
    """

    def __init__(
        self,
        cfg: GaussianActorCfg,
        state_dim: int,
        action_dim: int,
    ):
        super().__init__(state_dim=state_dim, action_dim=action_dim)

        self.cfg = cfg

        # -------- backbone --------
        self.backbone = cfg.backbone_cfg.class_type(
            in_feature=state_dim,
            out_feature=action_dim * 2,
            cfg=cfg.backbone_cfg,
        )

        # -------- std parameterization --------
        self.state_independent_std = cfg.state_independent_std
        self.log_std_min = cfg.log_std_min
        self.log_std_max = cfg.log_std_max
        self.min_std = cfg.min_std

        if self.state_independent_std:
            self.log_std = nn.Parameter(
                torch.ones(action_dim) * cfg.init_log_std
            )

        # -------- distribution options --------
        self.squash = cfg.squash
        
        self.act = self.sample

    # --------------------------------------------------------------------- #
    # core forward
    # --------------------------------------------------------------------- #

    def forward(self, state: torch.Tensor) -> D.Distribution:
        """
        Returns:
            torch.distributions.Distribution representing π(a|s)
        """

        logits = self.backbone(state)
        mean, log_std = torch.split(logits, self.action_dim, dim=-1)

        if self.state_independent_std:
            log_std = self.log_std.expand_as(mean)
        else:
            log_std = torch.clamp(log_std, self.log_std_min, self.log_std_max)

        std = log_std.exp()
        if self.min_std is not None:
            std = torch.clamp(std, min=self.min_std)

        base_dist = D.Normal(mean, std)

        if self.squash:
            base_dist = D.TransformedDistribution(
                base_dist,
                [D.transforms.TanhTransform(cache_size=1)],
            )

        self.act_dist = base_dist
        self.action_std = std
        return base_dist

    # --------------------------------------------------------------------- #
    # algorithm-facing helpers
    # --------------------------------------------------------------------- #

    def sample(
        self,
        state: torch.Tensor,
        deterministic: bool = False,
    ) -> torch.Tensor:
        dist = self(state)

        if deterministic:
            if isinstance(dist, D.TransformedDistribution):
                return dist.base_dist.mean.tanh()
            return dist.mean

        return dist.rsample()

    def get_actions_log_prob(self, action):
        return self.act_dist.log_prob(action).sum(dim=-1)


    @property
    def action_mean(self):
        return self.act_dist.mean

    def log_prob(
        self,
        state: torch.Tensor,
        action: torch.Tensor,
    ) -> torch.Tensor:
        dist = self(state)
        return dist.log_prob(action).sum(dim=-1)

    def entropy(
        self,
        state: torch.Tensor = None,
    ) -> torch.Tensor:
        dist = self(state) if state is not None else self.act_dist
        return dist.entropy().sum(dim=-1)
    
    def reset(self, *args, **kwargs):
        pass
    
    @torch.no_grad()
    def act_inference(self, state: torch.Tensor) -> torch.Tensor:
        """
        Deterministic action for inference/play.
        Returns the mean of the action distribution (with tanh if squashed).
        """
        dist = self(state)
        if isinstance(dist, D.TransformedDistribution):
            return dist.base_dist.mean.tanh()
        return dist.mean


# ===================================================================== #
# Config
# ===================================================================== #

@configclass
class GaussianActorCfg(ModuleBaseCfg):
    """Configuration for GaussianActor."""

    class_type: type[nn.Module] = GaussianActor

    backbone_cfg: MLPCfg = MISSING

    # -------- std parameterization --------
    state_independent_std: bool = False
    init_log_std: float = 0.0

    log_std_min: float = -10.0
    log_std_max: float = 2.0
    min_std: float | None = None

    # -------- distribution transform --------
    squash: bool = False

    def construct_from_cfg(self, *args, dim_params: dict=None, **kwargs):
        if dim_params is None:
            return super().construct_from_cfg(*args, **kwargs)
        return GaussianActor(self, dim_params["policy_dim"], dim_params["action_dim"])
    