from __future__ import annotations

import torch
import torch.nn as nn
import torch.distributions as D

from GopsEA import configclass
from GopsEA.networks.mlp import MLPCfg
from GopsEA.utils.template.module_base import ModuleBaseCfg
from GopsEA.components.actor.actor_base import ActorBase
from dataclasses import MISSING


class StateIndStdActor(ActorBase):
    """
    Gaussian policy actor: π(a|s)

    Design:
        state
          ↓
        Pi backbone (MLP)
          ↓
        action mean
          ↓
        Normal / Tanh-Normal distribution

    Notes:
        - std is state-independent
        - parameterized either as log_std or std (configurable)
    """

    def __init__(
        self,
        cfg: StateIndStdActorCfg,
        state_dim: int,
        action_dim: int,
    ):
        super().__init__(state_dim=state_dim, action_dim=action_dim)

        self.cfg = cfg

        # ------------------------------------------------------------------ #
        # backbone: state -> mean
        # ------------------------------------------------------------------ #
        self.backbone = cfg.backbone_cfg.class_type(
            in_feature=state_dim,
            out_feature=action_dim,
            cfg=cfg.backbone_cfg,
        )

        # ------------------------------------------------------------------ #
        # std parameterization
        # ------------------------------------------------------------------ #
        self.use_log_std = cfg.use_log_std
        self.log_std_min = cfg.log_std_min
        self.log_std_max = cfg.log_std_max

        self.std_param = nn.Parameter(
            torch.ones(action_dim) * cfg.init_std
        )

        # ------------------------------------------------------------------ #
        # distribution options
        # ------------------------------------------------------------------ #
        self.squash = cfg.squash

        self.act_dist: D.Distribution | None = None

    # --------------------------------------------------------------------- #
    # core forward
    # --------------------------------------------------------------------- #

    def forward(self, state: torch.Tensor) -> D.Distribution:
        mean = self.backbone(state)

        if self.use_log_std:
            log_std = torch.clamp(
                self.std_param,
                self.log_std_min,
                self.log_std_max,
            )
            std = log_std.exp()
        else:
            std = self.std_param
            if self.cfg.min_std is not None:
                std = torch.clamp(std, min=self.cfg.min_std, max=self.cfg.max_std)

        std = std.expand_as(mean)

        base_dist = D.Normal(mean, std)

        if self.squash:
            base_dist = D.TransformedDistribution(
                base_dist,
                [D.transforms.TanhTransform(cache_size=1)],
            )

        self.act_dist = base_dist

        return base_dist

    # --------------------------------------------------------------------- #
    # algorithm-facing helpers
    # --------------------------------------------------------------------- #

    def sample(
        self,
        state: torch.Tensor,
        deterministic: bool = False,
    ) -> torch.Tensor:
        dist: D.Distribution = self(state)

        if deterministic:
            if isinstance(dist, D.TransformedDistribution):
                return dist.base_dist.mean.tanh()
            return dist.mean

        return dist.rsample()
    
    def act(self, state):
        dist: D.Distribution = self(state)
        return dist.sample()

    def get_actions_log_prob(self, action: torch.Tensor) -> torch.Tensor:
        return self.act_dist.log_prob(action).sum(dim=-1)

    @property
    def action_mean(self) -> torch.Tensor:
        return self.act_dist.mean
    
    @property
    def action_std(self):
        return self.act_dist.stddev

    def entropy(
        self,
        state: torch.Tensor | None = None,
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
class StateIndStdActorCfg(ModuleBaseCfg):
    """Configuration for NoiseActor."""

    class_type: type[nn.Module] = StateIndStdActor

    # PPO style backbone
    backbone_cfg: MLPCfg = MLPCfg(
        hidden_features=[512, 256, 128],
        activations=[
            [('ELU', {})]
        ] * 3 + [ [] ]
    )

    # -------- std parameterization --------
    use_log_std: bool = True

    init_log_std: float = 0.0
    init_std: float = 1.0

    log_std_min: float = -10.0
    log_std_max: float = 2.0
    min_std: float = 0
    max_std: float = 10

    # -------- distribution transform --------
    squash: bool = False

    def construct_from_cfg(self, *args, dim_params: dict = None, **kwargs):
        if dim_params is None:
            return super().construct_from_cfg(*args, **kwargs)
        return StateIndStdActor(
            self,
            dim_params["policy_dim"],
            dim_params["action_dim"],
        )
