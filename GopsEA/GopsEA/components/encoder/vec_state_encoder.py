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
from GopsEA import configclass
from GopsEA.utils.template.module_base import ModuleBase, ModuleBaseCfg

# Assuming the corrected MLP and MLPCfg from previous iteration
from GopsEA.networks.mlp import MLP, MLPCfg 

class VecStateEncoder(ModuleBase):
    """
    Encodes a vector state (typically observation data) using an MLP backbone.
    
    Handles inputs of shape (B, C) or (B, L, C), where:
    - B: Batch size
    - L: Sequence/Look-back length (L=1 if input is (B, C))
    - C: Input feature dimension
    
    The MLP operates on the (B*L, C) tensor and the output is reshaped back to (B, L, C) or (B, C).
    """
    def __init__(self, cfg: VecStateEncoderCfg, in_feature: int, out_feature: int):
        # 1. FIX: Must call super().__init__() for nn.Module
        super().__init__()
        
        self.cfg = cfg
        self.in_feature = in_feature
        self.out_feature = out_feature
        
        # Initialize the MLP backbone
        # The MLP maps the input feature size to the output feature size.
        self.backbone: MLP = cfg.backbone_cfg.class_type(
            cfg=cfg.backbone_cfg,
            in_feature=in_feature, 
            out_feature=out_feature, 
        )

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        """
        Performs the encoding of the vector state.

        Args:
            input (torch.Tensor): Input tensor of shape (B, C) or (B, L, C).

        Returns:
            torch.Tensor: Encoded tensor of shape (B, out_feature) or (B, L, out_feature).
        """
        # Determine the input dimensionality (2D or 3D)
        num_dims = input.ndim
        
        if num_dims == 2:
            # Case (B, C): Reshape to (B, L=1, C) to unify processing
            B, C = input.shape
            input_3d = input.unsqueeze(dim=1)  # Shape (B, 1, C)
            L = 1
        elif num_dims == 3:
            # Case (B, L, C): Use directly
            B, L, C = input.shape
            input_3d = input
        else:
            raise ValueError(f"Wrong input shape for VecStateEncoder. Expected 2D or 3D, got {input.shape}.")

        # 2. Reshape/Rearrange the 3D tensor to 2D for the MLP backbone
        # Shape: (B, L, C) -> (B * L, C)
        states_2d = rearrange(input_3d, "B L C -> (B L) C")
        
        # Pass the (B*L, C) tensor through the MLP.
        # Output shape: (B * L, out_feature)
        encoded_2d = self.backbone(states_2d)
        
        # Reshape/Rearrange the encoded 2D tensor back to 3D
        # Shape: (B * L, out_feature) -> (B, L, out_feature)
        output_3d = rearrange(encoded_2d, "(B L) C_out -> B L C_out", B=B, L=L)
        
        if num_dims == 2:
            # 3. FIX: If input was (B, C), output should be (B, C_out). Squeeze is not in-place.
            output = output_3d.squeeze(dim=1) # Shape (B, C_out)
        else:
            # If input was (B, L, C), output is (B, L, C_out)
            output = output_3d
            
        return output
    
@configclass
class VecStateEncoderCfg(ModuleBaseCfg):
    """Configuration for the Vector State Encoder."""
    class_type: type[nn.Module] = VecStateEncoder
    # Configuration for the internal MLP backbone
    backbone_cfg: MLPCfg = MISSING