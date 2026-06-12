from __future__ import annotations

import torch
import torch.nn as nn
from typing import Optional, Tuple, Dict

from RenforceRL import configclass
from RenforceRL.utils.template.module_base import ModuleBaseCfg
from RenforceRL.components.world_models.world_model_base import WorldModelBase, WorldModelBaseCfg
from RenforceRL.components.encoder.vec_state_encoder import VecStateEncoderCfg
from RenforceRL.components.decoder.continuous_vec_decoder import ContinuousVecDecoderCfg

class LatentDynamicsTDMPC1(WorldModelBase):
    """
    TD-MPC latent deterministic dynamics model (fully modular / config-injected).

    Exposed interfaces:
        encode(obs) -> latent (B, latent_dim)

        predict_next(latent_or_obs, action)
           -> (next_latent, reward, term_logits, dist_feat_opt)

        init_kv_cache(batch_size)
           -> {"h": hidden_state}

        forward_with_kv_cache(latent_or_obs, action, kv_cache)
           -> (next_latent, reward, term_logits, dist_feat_opt, kv_cache)
    """

    cfg: LatentDynamicsTDMPC1Cfg

    def __init__(
        self,
        cfg: LatentDynamicsTDMPC1Cfg,
        obs_dim: int,
        action_dim: int,
        reward_dim: int = 1
    ):
        
        self.cfg = cfg
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.reward_dim = reward_dim
        self.latent_dim = cfg.latent_dim
        super(LatentDynamicsTDMPC1, self).__init__(cfg)

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
        ...

@configclass
class LatentDynamicsTDMPC1Cfg(WorldModelBaseCfg):
    """
    Config for a TD-MPC style latent deterministic dynamics model.
    All networks must be provided via config (import from external modules).
    """

    class_type: type[nn.Module] = None

    # latent size
    latent_dim: int = 128

    encoder_cfg: Optional[VecStateEncoderCfg] = None
    transition_cfg: Optional[VecStateEncoderCfg] = None
    reward_head_cfg: Optional[ContinuousVecDecoderCfg] = None
    term_head_cfg: Optional[ContinuousVecDecoderCfg] = None
    reconstruct_decoder_cfg: Optional[ContinuousVecDecoderCfg] = None
    
    loss_latent_coef: float = 1.0
    loss_reward_coef: float = 1.0
    loss_term_coef: float = 1.0
    loss_recon_coef: float = 1.0
    