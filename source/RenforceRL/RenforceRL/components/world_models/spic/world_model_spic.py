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

from RenforceRL.utils.isaaclab.trajectory import make_next_obs


class WorldModelSPiC(WorldModelBase):
    cfg: WorldModelSPiCCfg
    def __init__(
            self, cfg: WorldModelSPiCCfg, 
            policy_obs_dim, action_dim, rewards_dim, 
            terminations_dim=1, dynamic_obs_dim=None, critic_obs_dim=None,
            **kwargs
        ):
        self.policy_dim = policy_obs_dim
        self.action_dim = action_dim
        self.rewards_dim = rewards_dim
        self.terminations_dim = terminations_dim
        self.latent_dim = cfg.latent_dim if cfg.latent_dim is not None else policy_obs_dim
        self.dynamic_dim = dynamic_obs_dim if dynamic_obs_dim is not None else policy_obs_dim
        self.critic_dim = critic_obs_dim if critic_obs_dim is not None else critic_obs_dim
        super().__init__(cfg)

    @property
    def comp_dim(self) -> Dict[str, int]:
        """
        Should consistent with the replay buffer size. 
        """
        return {
            "policy":       self.policy_dim,
            "action":       self.action_dim,
            "rewards":      self.rewards_dim, 
            "terminations": self.terminations_dim, 
            "critic":       self.critic_dim, 
            "dynamic":      self.dynamic_dim,
        }

    def init_components(self):
        self.feat_dim = self.cfg.transformer_cfg.feat_dim
        
        self.state_encoder: VecStateEncoder = \
            self.cfg.state_encoder_cfg.class_type(self.cfg.state_encoder_cfg, self.policy_dim, self.latent_dim)
            
        self.storm_transformer:TransformerEncoderKVCache = \
            TransformerEncoderKVCache(self.cfg.transformer_cfg, self.latent_dim, self.action_dim)
        
        self.state_decoder: ContinuousVecDecoder = \
            ContinuousVecDecoder(self.cfg.state_decoder_cfg, self.feat_dim, self.dynamic_dim)
        self.rewards_decoder: ContinuousVecDecoder = \
            ContinuousVecDecoder(self.cfg.rewards_decoder_cfg, self.feat_dim, self.rewards_dim)
        self.termination_decoder: ContinuousVecDecoder = \
            ContinuousVecDecoder(self.cfg.termination_decoder_cfg, self.feat_dim, self.terminations_dim)

    def update(
            self, 
            policy: torch.Tensor, action: torch.Tensor, 
            reward: torch.Tensor, timeout: torch.Tensor,
            rewards: torch.Tensor, dynamic: torch.Tensor, 
            termination: torch.Tensor, is_valid: torch.Tensor, **kwargs
        ) -> Dict[str, float]:
            """
            Calculates the World Model loss, performs backpropagation, and updates 
            the model parameters for one mini-batch.
                    
            Returns:
                Dict[str, float]: Dictionary containing logging information (loss values).
            """
            self.train()
            
            # Fallback: Treat all steps as valid (only safe if all trajectories are full length)
            mask = is_valid
            
            # Shift inputs and targets for next-step prediction
            # Inputs: s_t, a_t (0 to T-1)
            policy_input = policy[:, :-1]
            action_input = action[:, :-1]
            
            # Targets: s_{t+1} (1 to T)
            dynamic_target = dynamic[:, 1:]
            
            # Targets: r_t, term_t (0 to T-1) - aligned with inputs
            rewards_target = rewards[:, :-1]
            termination_target = termination[:, :-1]
            
            # Mask: We need the target to be valid. 
            # Assuming is_valid marks valid steps. If step t+1 is valid, then transition t->t+1 is valid.
            mask_target = mask[:, 1:]

            batch_size, traj_length = policy_input.shape[:2]
            device = policy.device

            with torch.autocast(device_type='cuda', dtype=self.tensor_dtype, enabled=self.use_amp):
                
                # 2. Forward Pass
                temporal_mask = get_subsequent_mask_with_batch_length(traj_length, device)
                
                latent = self.state_encoder(policy_input)
                dist_feat = self.storm_transformer(latent, action_input, temporal_mask)
                
                # Prediction Heads
                prior_logits = self.state_decoder(dist_feat)         
                reward_logits = self.rewards_decoder(dist_feat)           
                termination_logits = self.termination_decoder(dist_feat) 

                # 3. Masked Loss Calculation
                
                # Dynamics Loss (State Transition)
                dynamics_loss = self.masked_mean_loss(
                    self.mse_loss_func, prior_logits, dynamic_target.detach(), mask_target
                )

                # Reward Loss
                reward_loss = self.masked_mean_loss(
                    self.mse_loss_func, reward_logits, rewards_target, mask_target
                )

                # Termination Loss
                termination_loss = self.masked_mean_loss(
                    self.bce_with_logits_loss_func, termination_logits, termination_target, mask_target
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
    
    def evaluate(
        self, 
        policy: torch.Tensor, action: torch.Tensor, 
        rewards: torch.Tensor, dynamic: torch.Tensor, 
        termination: torch.Tensor, is_valid: torch.Tensor=None, **kwargs
    ) -> Dict[str, float]:
        """
        Evaluates the SPiC World Model on a batch of data without updating parameters.
        
        Args:
            policy: Input state/policy obs features (B, T, policy_dim).
            action: Actions taken (B, T, action_dim).
            rewards: Target rewards (B, T, rewards_dim).
            dynamic: Target dynamic features (B, T, dynamic_dim).
            termination: Target termination signals (B, T, terminations_dim).
            is_valid: Mask indicating valid steps.
                
        Returns:
            Dict[str, float]: Dictionary containing loss values and error metrics.
        """
        self.train(False)
        
        def adjust_dims(a, b):
            while a.dim() < b.dim():
                a = a.unsqueeze(-1)
            while a.dim() > b.dim():
                a = a.squeeze(-1)
            return a

        # Fallback: Treat all steps as valid if mask is not provided
        mask = is_valid if is_valid is not None else torch.ones_like(termination, dtype=torch.float32)
        termination = termination.to(torch.float32)
        
        # Shift inputs and targets
        policy_input = policy[:, :-1]
        action_input = action[:, :-1]
        dynamic_target = dynamic[:, 1:]
        rewards_target = rewards[:, :-1]
        termination_target = termination[:, :-1]
        mask_target = mask[:, 1:]
        
        batch_size, traj_length = policy_input.shape[:2]
        device = policy.device

        with torch.no_grad(), torch.autocast(device_type='cuda', dtype=self.tensor_dtype, enabled=self.use_amp):
            
            # 1. Forward Pass
            temporal_mask = get_subsequent_mask_with_batch_length(traj_length, device)
            
            latent = self.state_encoder(policy_input)
            dist_feat = self.storm_transformer(latent, action_input, temporal_mask)
            
            # Prediction Heads
            prior_logits = self.state_decoder(dist_feat)         
            reward_logits = self.rewards_decoder(dist_feat)           
            termination_logits = self.termination_decoder(dist_feat) 

            # 2. Loss Calculation
            dynamics_loss = self.masked_mean_loss(
                self.mse_loss_func, prior_logits, dynamic_target, mask_target
            )
            reward_loss = self.masked_mean_loss(
                self.mse_loss_func, reward_logits, rewards_target, mask_target
            )
            termination_loss = self.masked_mean_loss(
                self.bce_with_logits_loss_func, termination_logits, termination_target, mask_target
            )
            total_loss = reward_loss + termination_loss + dynamics_loss

            # 3. Error Analysis
            valid_count = mask_target.sum().clamp(min=1.0)
            
            # Dynamics Reconstruction Error (MSE & MAE)
            dyn_err = (prior_logits - dynamic_target)
            mask_dyn = mask_target
            mask_dyn = adjust_dims(mask_dyn, dyn_err)
            mask_dyn = mask_dyn.expand_as(dyn_err)
            
            dynamic_mse = (dyn_err.pow(2) * mask_dyn).sum() / (valid_count * self.dynamic_dim)
            dynamic_mae = (dyn_err.abs() * mask_dyn).sum() / (valid_count * self.dynamic_dim)

            # Reward Error (MAE)
            rew_err = (reward_logits - rewards_target)
            mask_rew = mask_target
            mask_rew = adjust_dims(mask_rew, rew_err)
            mask_rew = mask_rew.expand_as(rew_err)
            reward_mae = (rew_err.abs() * mask_rew).sum() / (valid_count * self.rewards_dim)

            # Termination Accuracy
            term_probs = torch.sigmoid(termination_logits)
            term_preds = (term_probs > 0.5).float()
            mask_term = mask_target
            mask_term = adjust_dims(mask_term, term_preds)
            mask_term = mask_term.expand_as(term_preds)
            term_acc = ((term_preds == termination_target).float() * mask_term).sum() / (valid_count * self.terminations_dim)

        return {
            "eval/total_loss": total_loss.item(),
            "eval/dynamics_loss": dynamics_loss.item(),
            "eval/reward_loss": reward_loss.item(),
            "eval/termination_loss": termination_loss.item(),
            "eval/dynamic_mse": dynamic_mse.item(),
            "eval/dynamic_mae": dynamic_mae.item(),
            "eval/reward_mae": reward_mae.item(),
            "eval/term_acc": term_acc.item(),
        }

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
            next_dynamics = self.state_decoder(dist_feat)
            rewards_hat = self.rewards_decoder(dist_feat)
            termination_logits = self.termination_decoder(dist_feat)

        return next_dynamics, rewards_hat, termination_logits, dist_feat
    
@configclass
class WorldModelSPiCCfg(WorldModelBaseCfg):
    class_type:type[nn.Module] = WorldModelSPiC
    gradient_clip_norm: float = 1.0
    
    latent_dim:                  int = MISSING
    state_encoder_cfg:           ModuleBaseCfg = VecStateEncoderCfg()
    state_decoder_cfg:           ContinuousVecDecoderCfg = ContinuousVecDecoderCfg()
    rewards_decoder_cfg:         ContinuousVecDecoderCfg = ContinuousVecDecoderCfg()
    termination_decoder_cfg:     ContinuousVecDecoderCfg = ContinuousVecDecoderCfg()
    transformer_cfg:             TransformerEncoderKVCacheCfg = TransformerEncoderKVCacheCfg()
