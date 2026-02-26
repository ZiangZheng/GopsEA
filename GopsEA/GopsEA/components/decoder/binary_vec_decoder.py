from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Bernoulli # For binary sampling
from einops import rearrange
from typing import List, Union, Dict, Literal, Tuple
from dataclasses import MISSING, field
from GopsEA import configclass
from GopsEA.utils.template.module_base import ModuleBaseCfg

from GopsEA.networks.mlp import MLP, MLPCfg 


class BinaryVecDecoder(nn.Module):
    """
    Decodes an input vector into a binary vector (a vector of 0s and 1s) or
    a vector of probabilities (P(bit=1)).

    The MLP backbone must be configured to output raw logits, and a Sigmoid
    activation is applied in the forward pass to get probabilities.
    """
    def __init__(self, cfg: BinaryVecDecoderCfg, in_feature: int, out_feature: int):
        """
        Initializes the Binary Vector Decoder.

        Args:
            in_feature (int): The size of the input features (e.g., the latent dimension).
            out_feature (int): The size of the binary vector output.
            cfg (BinaryVecDecoderCfg): The configuration object.
        """
        super().__init__()
        
        self.cfg = cfg
        self.in_feature = in_feature
        self.out_feature = out_feature
        
        # Initialize the MLP backbone which acts as the decoder.
        # It maps in_feature to out_feature (logits).
        self.backbone: MLP = cfg.backbone_cfg.class_type(
            in_feature=in_feature, 
            out_feature=out_feature, 
            cfg=cfg.backbone_cfg
        )

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        """
        Performs the decoding to a binary vector or probability vector.

        Args:
            input (torch.Tensor): Input tensor of shape (B, C) or (B, L, C).

        Returns:
            torch.Tensor: Decoded vector of shape (B, out_feature) or (B, L, out_feature).
                          Values are probabilities [0, 1] or sampled binary [0, 1].
        """
        # Determine the input dimensionality (2D or 3D)
        num_dims = input.ndim
        
        if num_dims == 2:
            B, C = input.shape
            input_3d = input.unsqueeze(dim=1)  # Shape (B, 1, C)
            L = 1
        elif num_dims == 3:
            B, L, C = input.shape
            input_3d = input
        else:
            raise ValueError(f"Wrong input shape for BinaryVecDecoder. Expected 2D or 3D, got {input.shape}.")

        # 1. Reshape to (B * L, C) for the MLP
        input_2d = rearrange(input_3d, "B L C -> (B L) C")
        
        # 2. Pass through MLP to get raw logits
        # Output shape: (B * L, out_feature)
        logits_2d = self.backbone(input_2d)
        
        # No sigmoid for forward, returns logits
        
        # 3. Apply Sigmoid to convert logits to probabilities P(bit=1)
        # Probabilities are in [0, 1]
        # probs_2d = torch.sigmoid(logits_2d)
        
        # if self.cfg.use_sampling:
        #     # 4a. Perform binary sampling using Bernoulli distribution
        #     dist = Bernoulli(probs=probs_2d)
        #     output_2d = dist.sample()
        # else:
        #     # 4b. Return the raw probabilities
        #     output_2d = probs_2d
        
        # 5. Reshape back to 3D
        output_3d = rearrange(logits_2d, "(B L) C_out -> B L C_out", B=B, L=L)
        
        if num_dims == 2:
            # If input was (B, C), output should be (B, C_out)
            output = output_3d.squeeze(dim=1)
        else:
            # If input was (B, L, C), output is (B, L, C_out)
            output = output_3d
            
        return output
    
@configclass
class BinaryVecDecoderCfg(ModuleBaseCfg):
    """Configuration for the Binary Vector Decoder."""
    class_type: type[nn.Module] = BinaryVecDecoder
    # Configuration for the internal MLP backbone.
    # The last layer of this MLP should output raw logits (no activation).
    backbone_cfg: MLPCfg = MISSING
    # If True, a hard binary sample (0 or 1) is returned. If False, the probability (0 to 1) is returned.
    # use_sampling: bool = False