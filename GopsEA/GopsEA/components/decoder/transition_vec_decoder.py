from __future__ import annotations

import torch
from GopsEA import configclass

from GopsEA.components.decoder.continuous_vec_decoder import ContinuousVecDecoder, ContinuousVecDecoderCfg

class TransitionVecDecoder(ContinuousVecDecoder):
    cfg: TransitionVecDecoderCfg
    def __init__(self, cfg, state_feature, action_feature):
        super().__init__(cfg, state_feature + action_feature, state_feature)
        
    def forward(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        super_input = torch.cat([state, action], dim=-1)
        return super().forward(super_input)
    
@configclass
class  TransitionVecDecoderCfg(ContinuousVecDecoderCfg):
    class_type: type[TransitionVecDecoder] = TransitionVecDecoder
