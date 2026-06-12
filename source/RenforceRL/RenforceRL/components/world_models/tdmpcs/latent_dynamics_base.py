from __future__ import annotations
import copy
from dataclasses import MISSING

import torch
import torch.nn as nn
from typing import Optional, Tuple, Dict

from RenforceRL import configclass
from RenforceRL.utils.template.module_base import ModuleBaseCfg
from RenforceRL.components.world_models.world_model_base import WorldModelBase, WorldModelBaseCfg
from RenforceRL.components.encoder.vec_state_encoder import VecStateEncoderCfg
from RenforceRL.components.decoder.continuous_vec_decoder import ContinuousVecDecoderCfg
from RenforceRL.components.decoder.transition_vec_decoder import TransitionVecDecoderCfg

from RenforceRL.components.actor.pi_network import PiNetwork, PiNetworkCfg
from RenforceRL.components.critic.q_network import QNetwork, QNetworkCfg
from RenforceRL.components.normalizer import NormalizerBaseCfg, NormalizerBase

from RenforceRL.networks.optimizer import GroupedOptimizer, GroupedOptimizerCfg

class LatentDynamicsBase(WorldModelBase):
    cfg: LatentDynamicsBaseCfg

    def __init__(
        self,
        cfg: LatentDynamicsBaseCfg,
        obs_dim: int,
        action_dim: int,
    ):
        self.cfg = cfg
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.reward_dim = 1
        self.latent_dim = cfg.latent_dim
        super().__init__(cfg)

    encoder           : nn.Module
    transition        : nn.Module
    reward_head       : nn.Module
    term_head         : nn.Module
    q_network         : QNetwork
    obs_normalizer    : NormalizerBase

    target_q_network  : QNetwork

    def init_components(self):
        self.encoder     : nn.Module = self.cfg.encoder_cfg.class_type(     self.cfg.encoder_cfg,     in_feature=self.obs_dim,                      out_feature=self.latent_dim)
        self.reward_head : nn.Module = self.cfg.reward_head_cfg.class_type( self.cfg.reward_head_cfg, in_feature=self.latent_dim + self.action_dim, out_feature=self.reward_dim)
        self.term_head   : nn.Module = self.cfg.term_head_cfg.class_type(   self.cfg.term_head_cfg,   in_feature=self.latent_dim, out_feature=1)
        
        self.transition  : nn.Module = \
            self.cfg.transition_cfg.class_type(self.cfg.transition_cfg, in_feature=self.latent_dim + self.action_dim, out_feature=self.latent_dim)

        self.q_network  : QNetwork = \
            QNetwork(self.cfg.critic_cfg, state_dim=self.latent_dim, action_dim=self.action_dim, out_feature=self.reward_dim)
            
        self.obs_normalizer = NormalizerBase.construct_from_cfg(self.cfg.obs_normalizer_cfg, shape=self.obs_dim)

        self.target_q_network = copy.deepcopy(self.q_network)
        for p in self.target_q_network.parameters():
            p.requires_grad = False

    def init_optimizers(self):
        ...

    @property
    def comp_dim(self) -> Dict[str, int]:
        return {
            # dims used for replay buffer storage
            "obs": self.obs_dim,
            "action": self.action_dim,
            "reward": self.reward_dim,
            "termination": 1,
        }

    def update(self, **kwargs) -> Dict[str, float]:
        """
        Dispatch to step-update or sequence-update based on input shape.
        """
        obs = kwargs["obs"]
        if obs.ndim == 2: return self.update_step(**kwargs)
        elif obs.ndim == 3: return self.update_seq(**kwargs)
        else: raise ValueError(f"Unsupported obs shape: {obs.shape}")

    def update_step(self, obs, action, next_obs, reward, termination, **kwargs) -> Dict[str, float]:
        ...
        
    def update_seq(self, obs, action, reward, termination, is_valid, **kwargs) -> Dict[str, float]:
        ...

    @staticmethod
    def _ema_update(online: nn.Module, target: nn.Module, ema_m):
        m = ema_m
        for p_o, p_t in zip(online.parameters(), target.parameters()):
            p_t.data.mul_(m).add_((1 - m) * p_o.data)

    @torch.no_grad()
    def predict_next(
        self,
        action: torch.Tensor,
        state: torch.Tensor,
        latent: torch.Tensor = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        latent = self.encode(state) if state.size(-1) == self.obs_dim else state
        sa = torch.cat([latent, action], dim=-1)
        next_latent = self.transition(sa)
        dist_feat = next_latent
        reward = self.reward_head(next_latent)
        term_logits = self.term_head(next_latent)
        termination = torch.sigmoid(term_logits) > 0.5
        return next_latent, reward, termination, {"term_logits": term_logits, "dist_feat": dist_feat}

    @torch.no_grad()
    def predict_q(self, latent: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        return self.q_network(latent, action)
    
    def save_world_model(self, path, infos=None):
        saved_dict = {
            "infos": infos,
            "world_model_dict": self.state_dict(),
            "world_optim_state_dict": {
                "optim": self.optimizer.state_dict(),
                "scaler": self.scaler.state_dict()
            }
        }
        torch.save(saved_dict, path)

    def load_world_model(self, path, load_optim=False):
        loaded_dict = torch.load(path)
        self.load_state_dict(loaded_dict["world_model_state_dict"])
        if load_optim:
            self.optimizer.load_state_dict(loaded_dict["world_optim_state_dict"]["optim"])
            self.scaler.load_state_dict(loaded_dict["world_optim_state_dict"]["scaler"])
        return loaded_dict["infos"]

@configclass
class LatentDynamicsBaseCfg(WorldModelBaseCfg):
    """
    Config for a TD-MPC style latent deterministic dynamics model.
    All networks must be provided via config (import from external modules).
    """

    class_type                  : type[nn.Module] = LatentDynamicsBase
    latent_dim                  : int = 128
    critic_cfg                  : QNetworkCfg = MISSING
    encoder_cfg                 : VecStateEncoderCfg = MISSING
    transition_cfg              : VecStateEncoderCfg = MISSING
    reward_head_cfg             : TransitionVecDecoderCfg = MISSING
    term_head_cfg               : ContinuousVecDecoderCfg = MISSING
    obs_normalizer_cfg          : NormalizerBaseCfg = NormalizerBaseCfg()
    
    # ------- Optimizers -------
    dyn_learning_rate           : float = 1e-4           # dynamics optimizer
    dyn_weight_decay            : float = 0.05
    q_learning_rate             : float = 1e-4         # critic optimizer
    q_weight_decay              : float = 0.05
        
    optimizers_cfg              : GroupedOptimizerCfg = GroupedOptimizerCfg(
        grad_clip_norm=10.0,
        grad_clip_norms={
            "optim_dyn": 10.0,
            "optim_q": 1.0
        }
    )

    # AMP
    use_amp: bool = True