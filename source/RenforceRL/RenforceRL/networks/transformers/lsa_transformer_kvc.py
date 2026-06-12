from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from typing import List, Union, Dict, Literal, Tuple
from dataclasses import MISSING
from RenforceRL import configclass
from RenforceRL.utils.template.module_base import ModuleBaseCfg
from RenforceRL.networks.mlp import MLP, MLPCfg

from .attention_blocks import PositionalEncoding1D, AttentionBlock, AttentionBlockKVCache


"""
Input is latent state and action
"""


class TransformerEncoderKVCache(nn.Module):
    """
    A Transformer Encoder model designed for sequential processing with Key/Value caching
    for efficient single-step (autoregressive) inference.
    
    Input: Concatenation of latent embedding and action vector.
    """
    
    def __init__(self, cfg: TransformerEncoderKVCacheCfg, latent_dim:int, action_dim:int):
        super().__init__()
        
        self.cfg = cfg
        self.latent_dim = latent_dim
        self.action_dim = action_dim
        self.input_dim = latent_dim + action_dim
        self.feat_dim = cfg.feat_dim
        
        # 1. Feature Projection (Stem)
        # Mixes latent_embedding and action, projects to feat_dim
        self.stem = nn.Sequential(
            nn.Linear(self.input_dim, cfg.feat_dim, bias=False),
            nn.LayerNorm(cfg.feat_dim),
            nn.ReLU(inplace=True),
            nn.Linear(cfg.feat_dim, cfg.feat_dim, bias=False),
            nn.LayerNorm(cfg.feat_dim)
        )
        
        # 2. Positional Encoding
        self.position_encoding = PositionalEncoding1D(
            max_length=cfg.max_length, 
            embed_dim=cfg.feat_dim
        )
        
        # 3. Transformer Layer Stack
        self.layer_stack = nn.ModuleList([
            AttentionBlockKVCache(
                feat_dim=cfg.feat_dim, 
                hidden_dim=cfg.feat_dim * 2, 
                num_heads=cfg.num_heads, 
                dropout=cfg.dropout
            ) for _ in range(cfg.num_layers)
        ])
        
        # 4. Final LayerNorm after positional encoding
        self.layer_norm = nn.LayerNorm(cfg.feat_dim, eps=1e-6) 
        
        # K/V Cache state (initialized via reset_kv_cache_list)
        self.kv_cache_list: List[torch.Tensor] = []

    def forward(self, latent: torch.Tensor, action: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """
        Standard forward pass for full sequences (training).
        
        Args:
            latent (torch.Tensor): Latent state tensor (B, L, latent_dim).
            action (torch.Tensor): Action tensor (B, L, action_dim).
            mask (torch.Tensor): Attention mask (e.g., for padding or causal attention).
            
        Returns:
            torch.Tensor: Output features (B, L, feat_dim).
        """
        # Combine and project features
        feats = self.stem(torch.cat([latent, action], dim=-1))
        
        # Apply positional encoding and final layer norm
        feats = self.position_encoding(feats)
        feats = self.layer_norm(feats) # LayerNorm before transformer layers (common practice)

        # Pass through transformer layers
        for layer in self.layer_stack:
            # Note: For full sequence, K=V=Q=feats.
            # mask should be a causal/padding mask if L > 1.
            feats, _ = layer(query=feats, key=feats, value=feats, attn_mask=mask)

        return feats

    def reset_kv_cache_list(self, batch_size: int, dtype: torch.dtype):
        """
        Resets the internal list of Key/Value caches for all layers. 
        Should be called before starting a new sequence generation.
        """
        device = next(self.parameters()).device
        self.kv_cache_list = []
        for _ in self.layer_stack:
            # Initialize with an empty sequence (B, 0, C)
            self.kv_cache_list.append(
                torch.zeros(
                    size=(batch_size, 0, self.feat_dim), 
                    dtype=dtype, device=device
                )
            )

    def forward_with_kv_cache(self, latent: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        """
        Autoregressive forward pass for single steps (inference). 
        The K/V cache is updated internally for subsequent steps.
        
        Args:
            latent (torch.Tensor): Latent state tensor (B, 1, latent_dim).
            action (torch.Tensor): Action tensor (B, 1, action_dim).
            
        Returns:
            torch.Tensor: Output features for the current step (B, 1, feat_dim).
        """
        # Assertion: input must be a single step (L=1)
        assert latent.shape[1] == 1, "Input sequence length must be 1 for forward_with_kv_cache."
        
        # 1. Feature Projection
        # Input shape (B, 1, input_dim) -> feats shape (B, 1, feat_dim)
        feats = self.stem(torch.cat([latent, action], dim=-1))
        
        # 2. Positional Encoding
        # Get the current sequence length (the size of the cache before update)
        current_position = self.kv_cache_list[0].shape[1] 
        feats = self.position_encoding.forward_with_position(feats, position=current_position)
        feats = self.layer_norm(feats)

        # 3. Create Attention Mask (only needed if sequence is long, not usually for single step)
        # If L=1, Q only needs to attend to itself and the cache. 
        # Causal mask is implicitly handled by using cache for K/V and Q=current step.
        # Since this is for cross/self-attention where Q is always the latest token, we 
        # don't typically need an explicit mask if the cache length is what we want.
        # We will use the original mask structure (if necessary, check AttentionBlock for mask format)
        # Note: If AttentionBlock expects a causal mask over Q, K, this simple setup might need adjustment.
        
        for idx, layer in enumerate(self.layer_stack):
            # Key/Value are the concatenation of previous cache and current features (feats)
            # K_full = V_full = [K_cache; Current_feats]
            K_full = torch.cat([self.kv_cache_list[idx], feats], dim=1)
            
            # Update the K/V cache for the *next* time step
            self.kv_cache_list[idx] = K_full 
            
            # Query is just the current feature (feats), K/V are the full sequence history
            # Output shape: (B, 1, feat_dim)
            feats, _ = layer(query=feats, key=K_full, value=K_full)

        return feats

@configclass
class TransformerEncoderKVCacheCfg(ModuleBaseCfg):
    """
    Configuration for the Transformer Encoder with K/V Caching.
    """
    class_type: type[nn.Module] = TransformerEncoderKVCache # Reference to the class name
    
    # Core Transformer parameters
    
    feat_dim: int = 512        # The hidden dimension (d_model) for the transformer
    num_layers: int = 4        # Number of transformer layers
    num_heads: int = 8         # Number of attention heads
    max_length: int = 1024     # Maximum sequence length for positional encoding
    dropout: float = 0.1       # Dropout rate
