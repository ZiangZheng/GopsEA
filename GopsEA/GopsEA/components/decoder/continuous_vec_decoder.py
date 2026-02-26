from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from typing import List, Union, Dict, Literal, Tuple
from dataclasses import MISSING
from GopsEA import configclass
from GopsEA.utils.template.module_base import ModuleBaseCfg

from GopsEA.networks.mlp import MLP, MLPCfg 


class ContinuousVecDecoder(nn.Module):
    """
    Decodes an input vector (e.g., a latent representation or encoded state) 
    into a continuous vector output using an MLP backbone.

    Handles inputs of shape (B, C) or (B, L, C). The output feature dimension 
    (out_feature) should correspond to the desired continuous output size (e.g., action space size).
    """
    def __init__(self, cfg: ContinuousVecDecoderCfg, in_feature: int, out_feature: int):
        """
        Initializes the Continuous Vector Decoder.

        Args:
            in_feature (int): The size of the input features (e.g., the latent dimension).
            out_feature (int): The size of the continuous vector output (e.g., action dimension).
            cfg (ContinuousVecDecoderCfg): The configuration object.
        """
        super().__init__()
        
        self.cfg = cfg
        self.in_feature = in_feature
        self.out_feature = out_feature
        
        # Initialize the MLP backbone which acts as the decoder
        # It maps the input feature size (in_feature) to the desired output feature size (out_feature).
        self.backbone: MLP = cfg.backbone_cfg.class_type(
            in_feature=in_feature, 
            out_feature=out_feature, 
            cfg=cfg.backbone_cfg
        )

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        """
        Performs the decoding to a continuous vector.

        Args:
            input (torch.Tensor): Input tensor of shape (B, C) or (B, L, C).

        Returns:
            torch.Tensor: Decoded continuous vector of shape (B, out_feature) or (B, L, out_feature).
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
            raise ValueError(f"Wrong input shape for ContinuousVecDecoder. Expected 2D or 3D, got {input.shape}.")

        # 1. Reshape/Rearrange the 3D tensor to 2D for the MLP backbone
        # Shape: (B, L, C) -> (B * L, C)
        input_2d = rearrange(input_3d, "B L C -> (B L) C")
        
        # 2. Pass the (B*L, C) tensor through the MLP decoder.
        # Output shape: (B * L, out_feature)
        decoded_2d = self.backbone(input_2d)
        
        # 3. Reshape/Rearrange the decoded 2D tensor back to 3D
        # Shape: (B * L, out_feature) -> (B, L, out_feature)
        output_3d = rearrange(decoded_2d, "(B L) C_out -> B L C_out", B=B, L=L)
        
        if num_dims == 2:
            # If input was (B, C), output should be (B, C_out)
            output = output_3d.squeeze(dim=1) # Shape (B, C_out)
        else:
            # If input was (B, L, C), output is (B, L, C_out)
            output = output_3d
            
        return output
    
@configclass
class ContinuousVecDecoderCfg(ModuleBaseCfg):
    """Configuration for the Continuous Vector Decoder."""
    class_type: type[nn.Module] = ContinuousVecDecoder
    # Configuration for the internal MLP backbone (decoder network)
    backbone_cfg: MLPCfg = MISSING