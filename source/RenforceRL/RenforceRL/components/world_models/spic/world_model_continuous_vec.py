from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import autocast
from torch.distributions import OneHotCategorical, Normal
from einops import rearrange, repeat, reduce
from einops.layers.torch import Rearrange
from typing import List, Union, Dict, Literal, Tuple
from dataclasses import MISSING
from RenforceRL import configclass
from RenforceRL.utils.template.module_base import ModuleBaseCfg

from RenforceRL.components.encoder import VecStateEncoder, VecStateEncoderCfg
from RenforceRL.components.decoder import (
    BinaryVecDecoder, BinaryVecDecoderCfg, 
    ContinuousVecDecoder, ContinuousVecDecoderCfg
)
from RenforceRL.networks.transformers.lsa_transformer_kvc import \
    TransformerEncoderKVCache, TransformerEncoderKVCacheCfg
from RenforceRL.networks.transformers.attention_blocks import \
    get_subsequent_mask_with_batch_length

from RenforceRL.components.world_models.world_model_base import WorldModelBase, WorldModelBaseCfg


class WorldModelContinuousVec(WorldModelBase):
    cfg: WorldModelContinuousVecCfg
    def __init__(self, cfg: WorldModelContinuousVecCfg, obs_dim: int, action_dim: int, latent_dim: int):
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.latent_dim = latent_dim
        super().__init__(cfg)

    @property
    def comp_dim(self) -> Dict[str, int]:
        return {"obs_dim": self.obs_dim, "action_dim": self.action_dim, "latent_dim": self.action_dim}

    def init_components(self):
        self.state_encoder: VecStateEncoder = \
            VecStateEncoder(self.cfg.state_encoder_cfg, self.obs_dim, self.latent_dim)
        
        self.storm_transformer:TransformerEncoderKVCache = \
            TransformerEncoderKVCache(self.cfg.transformer_cfg, self.latent_dim, self.action_dim)
        
        self.feat_dim = self.cfg.transformer_cfg.feat_dim
        
        self.state_decoder: ContinuousVecDecoder = \
            ContinuousVecDecoder(self.cfg.state_decoder_cfg, self.feat_dim, self.obs_dim)
        self.reward_decoder: ContinuousVecDecoder = \
            ContinuousVecDecoder(self.cfg.reward_decoder_cfg, self.feat_dim, 1)
        self.termination_decoder: BinaryVecDecoder = \
            BinaryVecDecoder(self.cfg.termination_decoder_cfg, self.feat_dim, 1)

    def update(
            self, obs: torch.Tensor, action: torch.Tensor, 
            reward: torch.Tensor, next_obs: torch.Tensor, 
            termination: torch.Tensor, is_valid: torch.Tensor=None
        ) -> Dict[str, float]:
            """
            Calculates the World Model loss, performs backpropagation, and updates 
            the model parameters for one mini-batch.
            
            Args:
                samples (Dict[str, torch.Tensor]): The mini-batch sampled from the buffer, 
                    containing 'obs1', 'action', 'reward', 'obs2' (next_obs), 'termination', 
                    and the crucial 'is_valid' mask.
                    
            Returns:
                Dict[str, float]: Dictionary containing logging information (loss values).
            """
            self.train()
            
            # Fallback: Treat all steps as valid (only safe if all trajectories are full length)
            mask = is_valid if is_valid is not None else torch.ones_like(reward, dtype=torch.float32) 
            
            batch_size, traj_length = obs.shape[:2]
            device = obs.device

            with torch.autocast(device_type='cuda', dtype=self.tensor_dtype, enabled=self.use_amp):
                
                # 2. Forward Pass
                temporal_mask = get_subsequent_mask_with_batch_length(traj_length, device)
                
                latent = self.state_encoder(obs)
                dist_feat = self.storm_transformer(latent, action, temporal_mask)
                
                # Prediction Heads
                prior_logits = self.state_decoder(dist_feat)         
                reward_logits = self.reward_decoder(dist_feat)           
                termination_logits = self.termination_decoder(dist_feat) 

                # 3. Masked Loss Calculation
                
                # Dynamics Loss (State Transition)
                dynamics_loss = self.masked_mean_loss(
                    self.mse_loss_func, prior_logits, next_obs.detach(), mask
                )

                # Reward Loss
                reward_loss = self.masked_mean_loss(
                    self.mse_loss_func, reward_logits, reward, mask
                )

                # Termination Loss
                termination_loss = self.masked_mean_loss(
                    self.bce_with_logits_loss_func, termination_logits, termination, mask
                )

                total_loss = reward_loss + termination_loss + dynamics_loss

            # 4. Gradient Descent
            self.scaler.scale(total_loss).backward()
            self.scaler.unscale_(self.optimizer) 
            torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=self.cfg.gradient_clip_norm) 
            self.scaler.step(self.optimizer)
            self.scaler.update()
            self.optimizer.zero_grad(set_to_none=True)

            # 5. Logging Information
            tb_info = {
                "reward_loss": reward_loss.item(),
                "termination_loss": termination_loss.item(),
                "dynamics_loss": dynamics_loss.item(),
                "total_loss": total_loss.item(),
            }
            return tb_info
    
    def predict_next(self, obs: torch.Tensor, action: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Predicts the next observation, reward, and termination probability based on 
        the current observation (obs) and action. This is typically an open-loop prediction.

        Args:
            obs (torch.Tensor): Current observation (s_t). Shape (B, D_obs).
            action (torch.Tensor): Action taken (a_t). Shape (B, D_act).
            
        Returns:
            Tuple: (next_obs, reward_hat, termination_prob, dist_feat)
        """
        self.eval() # Ensure model is in evaluation mode
        
        # Use model's internal AMP settings
        with torch.no_grad(), torch.autocast(device_type='cuda', dtype=self.tensor_dtype, enabled=self.use_amp):
            
            # 1. Encode Observation to Latent State (z_t)
            latent = self.state_encoder(obs)
            
            # 2. Dynamics Transition: h_t -> h_{t+1} (dist_feat)
            # Note: This requires a Transformer setup that handles a single step input (z_t, a_t)
            # and typically updates an internal KV cache for sequence continuation.
            dist_feat = self.storm_transformer.forward_with_kv_cache(latent, action)

            # 3. Decode Predictions (from h_{t+1})
            next_obs = self.state_decoder(dist_feat)
            reward_hat = self.reward_decoder(dist_feat)
            termination_logits = self.termination_decoder(dist_feat)

        return next_obs, reward_hat, termination_logits, dist_feat
    
    def evaluate(
        self, obs: torch.Tensor, action: torch.Tensor, 
        reward: torch.Tensor, next_obs: torch.Tensor, 
        termination: torch.Tensor, is_valid: torch.Tensor=None
    ) -> Dict[str, float]:
        """
        Evaluates the World Model on a batch of data without updating parameters.
        Computes losses and additional error metrics for analysis.
        
        Args:
            obs, action, reward, next_obs, termination: Input tensors from replay buffer.
            is_valid: Mask indicating valid steps in the trajectory.
                
        Returns:
            Dict[str, float]: Dictionary containing loss values and error metrics.
        """
        # Ensure model is in evaluation mode (using train(False) to avoid recursion if named eval)
        self.train(False)
        
        # Fallback: Treat all steps as valid (only safe if all trajectories are full length)
        mask = is_valid if is_valid is not None else torch.ones_like(reward, dtype=torch.float32) 
        
        batch_size, traj_length = obs.shape[:2]
        device = obs.device

        with torch.no_grad(), torch.autocast(device_type='cuda', dtype=self.tensor_dtype, enabled=self.use_amp):
            
            # 1. Forward Pass
            temporal_mask = get_subsequent_mask_with_batch_length(traj_length, device)
            
            latent = self.state_encoder(obs)
            dist_feat = self.storm_transformer(latent, action, temporal_mask)
            
            # Prediction Heads
            prior_logits = self.state_decoder(dist_feat)         
            reward_logits = self.reward_decoder(dist_feat)           
            termination_logits = self.termination_decoder(dist_feat) 
            
            # 2. Loss Calculation (Standard Metrics)
            dynamics_loss = self.masked_mean_loss(
                self.mse_loss_func, prior_logits, next_obs, mask
            )
            reward_loss = self.masked_mean_loss(
                self.mse_loss_func, reward_logits, reward, mask
            )
            termination_loss = self.masked_mean_loss(
                self.bce_with_logits_loss_func, termination_logits, termination, mask
            )
            total_loss = reward_loss + termination_loss + dynamics_loss

            # 3. Error Analysis (Interpretability Metrics)
            valid_count = mask.sum().clamp(min=1.0)
            
            # Observation Reconstruction Error (MSE & MAE)
            obs_err = (prior_logits - next_obs)
            mask_obs = mask.expand_as(obs_err) # Expand mask to (B, T, D)
            
            obs_mse = (obs_err.pow(2) * mask_obs).sum() / (valid_count * self.obs_dim)
            obs_mae = (obs_err.abs() * mask_obs).sum() / (valid_count * self.obs_dim)

            # Reward Error (MAE)
            reward_mae = ((reward_logits - reward).abs() * mask).sum() / valid_count

            # Termination Accuracy
            term_probs = torch.sigmoid(termination_logits)
            term_preds = (term_probs > 0.5).float()
            term_acc = ((term_preds == termination).float() * mask).sum() / valid_count

        return {
            "eval/total_loss": total_loss.item(),
            "eval/dynamics_loss": dynamics_loss.item(),
            "eval/reward_loss": reward_loss.item(),
            "eval/termination_loss": termination_loss.item(),
            "eval/obs_mse": obs_mse.item(),
            "eval/obs_mae": obs_mae.item(),
            "eval/reward_mae": reward_mae.item(),
            "eval/term_acc": term_acc.item(),
        }
    
@configclass
class WorldModelContinuousVecCfg(WorldModelBaseCfg):
    class_type:type[nn.Module] = WorldModelContinuousVec
    
    gradient_clip_norm: float = 1.0
    
    state_encoder_cfg:ModuleBaseCfg = VecStateEncoderCfg()
    state_decoder_cfg:ModuleBaseCfg = ContinuousVecDecoderCfg()
    
    reward_decoder_cfg:ModuleBaseCfg = ContinuousVecDecoderCfg()
    termination_decoder_cfg:ModuleBaseCfg = BinaryVecDecoderCfg()
    
    transformer_cfg:TransformerEncoderKVCacheCfg = TransformerEncoderKVCacheCfg()
