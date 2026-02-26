from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import autocast
from torch.distributions import OneHotCategorical, Normal
from einops import rearrange, repeat, reduce
from einops.layers.torch import Rearrange
from typing import List, Union, Dict, Literal, Tuple
from dataclasses import MISSING, field, dataclass
from GopsEA import configclass
from GopsEA.utils.template.module_base import ModuleBaseCfg

from GopsEA.networks.activations import get_activation, ACTIVATION_TYPES


class MLP(nn.Sequential):
    """
    Multi-Layer Perceptron (MLP) module with configurable hidden layers and activations.
    The input and output feature sizes are passed during initialization, while hidden sizes 
    and activations are defined in the configuration (MLPCfg).
    """
    def __init__(self, cfg: MLPCfg, in_feature: int, out_feature: int):
        """
        Initializes the MLP.

        Args:
            in_feature (int): The size of the input features.
            out_feature (int): The size of the output features.
            cfg (MLPCfg): The configuration object defining hidden sizes and activations.
        """
        self.cfg: MLPCfg = cfg
        self.in_feature: int = in_feature
        self.out_feature: int = out_feature
        
        # Determine all feature sizes including input and output
        # E.g., in_feature=10, hidden_features=[64, 32], out_feature=2
        # all_features = [10, 64, 32, 2]
        all_features = [self.in_feature] + self.cfg.hidden_features + [self.out_feature]
        
        # Ensure the number of activation sets matches the number of layers
        num_layers = len(all_features) - 1
        if len(self.cfg.activations) != num_layers:
            raise ValueError(
                f"The number of activation sets ({len(self.cfg.activations)}) "
                f"must match the number of layers ({num_layers})."
            )
            
        layers = []
        
        # Iterate over all linear layers (from input to output)
        for i in range(num_layers):
            layer_in_feature = all_features[i]
            layer_out_feature = all_features[i+1]
            layer_activations = self.cfg.activations[i]
            
            _info = {
                "layer_in_feature": layer_in_feature,
                "layer_out_feature": layer_out_feature
            }
            
            # 1. Add the Linear layer
            _layer = [
                nn.Linear(layer_in_feature, layer_out_feature)
            ]
            
            # 2. Add the Activation layers
            _layer += [
                get_activation(name, _info, params) for name, params in layer_activations
            ]
            
            # Extend the main layers list
            layers.extend(_layer)
        
        # Note: The last layer's activations (if present in cfg.activations) are applied 
        # *after* the final Linear layer to produce the output.
        super().__init__(*layers)
        
@configclass
class MLPCfg(ModuleBaseCfg):
    """
    Configuration for the Multi-Layer Perceptron (MLP) module.
    
    If you want to leave the network finally will not pass the activatetion leave last layer [[]] 
    ```python
    ac_backbone_cfg = MLPCfg(
                hidden_features=[512, 256, 128],
                activations=[
                    [('ELU', {})]
                ] * 3 + [[]]
    )
    ```
    """
    class_type: type[nn.Module] = MLP
    
    # List of hidden layer sizes. The input and output features are passed directly to __init__.
    # The list format is: [hidden_size_1, hidden_size_2, ..., hidden_size_N]
    hidden_features: List[int] = field(default_factory=list)

    # Activation functions for each layer. 
    # Must have len(activations) == len(hidden_features) + 1 (for input -> first hidden, ..., last hidden -> output).
    # Each element is a list of (name, params) tuples for potential multiple activations per linear layer.
    activations: List[List[Tuple[ACTIVATION_TYPES, dict]]] = MISSING