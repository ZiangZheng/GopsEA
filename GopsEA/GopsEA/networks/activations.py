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


ACTIVATION_TYPES = Literal["LayerNorm", "GELU", "Dropout", "ReLU"]

def get_activation(name: Union[str, nn.Module], losc, kwargs):
    if isinstance(name, str):
        if name == "LayerNorm":
            ret = nn.LayerNorm(losc["layer_out_feature"], **kwargs)
        elif name == "GELU":
            ret = nn.GELU(**kwargs)
        elif name == "ELU":
            ret = nn.ELU(**kwargs)
        elif name == "ReLU":
            ret = nn.ReLU(**kwargs)
        elif name == "SiLU":
            ret = nn.SiLU(**kwargs)
        elif name == "Dropout":
            ret = nn.Dropout(kwargs["p"], **kwargs)
        else:
            raise ValueError(f"{name} not surrport for activation.")
    else:
        ret = name
    return ret