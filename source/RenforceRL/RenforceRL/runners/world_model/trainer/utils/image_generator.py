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

from RenforceRL.components.world_models.spic.world_model_continuous_vec import WorldModelContinuousVec

@torch.no_grad()
def image_data_generator(
    world_model: WorldModelContinuousVec,
    initial_obs: torch.Tensor,
    actions: torch.Tensor
) -> Dict[str, torch.Tensor]:
    """
    Generates an 'imagined' trajectory by looping the World Model's single-step prediction.

    The sequence starts with the prediction *from* the initial_obs. The final output 
    sequence length T will match the action sequence length.

    Args:
        world_model (WorldModelBasic): The trained World Model instance.
        initial_obs (torch.Tensor): Initial observation(s) (s_0). Shape (B, D_obs).
        actions (torch.Tensor): Sequence of actions (a_0, a_1, ..., a_{T-1}). Shape (B, T, D_act).
        
    Returns:
        Dict[str, torch.Tensor]: The generated sequence for 'obs', 'reward', 'termination', 'latent'.
                                 Shapes: (B, T, D).
    """
    world_model.eval()
    B, T, D_act = actions.shape
    device = initial_obs.device

    # --- 1. Preparation and KV Cache Reset ---
    
    # Crucial step: Reset the Transformer's KV Cache for the new prediction sequence.
    # We assume storm_transformer has a method to reset its internal cache for batch size B.
    # Note: If storm_transformer is part of WorldModelBasic, we call its method.
    world_model.storm_transformer.reset_kv_cache(batch_size=B) 

    # Prepare lists to collect the imagined sequence
    imagined_obs = []
    imagined_reward = []
    imagined_termination = []
    imagined_latent_feat = []

    # The prediction starts from s_0.
    # s_t is the observation used to predict s_{t+1}, r_t, d_t.
    current_obs = initial_obs.clone()

    # --- 2. Step-by-Step Prediction Loop ---
    
    for t in range(T):
        action_t = actions[:, t, :] # Action a_t (Shape: B, D_act)

        # Use the single-step prediction interface (predict_next)
        with torch.no_grad(), torch.autocast(device_type=device.type, 
                                            dtype=world_model.tensor_dtype, 
                                            enabled=world_model.use_amp):
            
            # Predict next step based on current_obs (s_t) and action_t (a_t)
            # This prediction predicts s_{t+1}, r_t, d_t, and features h_{t+1}
            # The KV Cache is updated internally here.
            
            # Assuming predict_next returns: (next_obs, reward_hat, termination_prob, dist_feat)
            next_obs_t, reward_t, termination_t, dist_feat_t = \
                world_model.predict_next(current_obs, action_t)
            
        # 3. Collect Results
        imagined_obs.append(next_obs_t)
        imagined_reward.append(reward_t)
        imagined_termination.append(termination_t)
        imagined_latent_feat.append(dist_feat_t)
        
        # 4. Update current_obs for the next time step (t+1)
        # The predicted state becomes the input observation for the next loop iteration (s_{t+1})
        current_obs = next_obs_t.detach() 

    # --- 5. Concatenate and Return ---

    # Stack the lists into tensors of shape (B, T, D)
    imagined_data = {
        'obs': torch.stack(imagined_obs, dim=1),
        'reward': torch.stack(imagined_reward, dim=1),
        'termination': torch.stack(imagined_termination, dim=1),
        'latent': torch.stack(imagined_latent_feat, dim=1),
        'action': actions # Include actions for completeness
    }
    
    return imagined_data