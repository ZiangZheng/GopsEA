from GopsEA import configclass

from .sac import SAC, SACCfg


class SACSeq(SAC):
    """Compatibility alias. Use SAC directly."""


@configclass
class SACSeqCfg(SACCfg):
    class_type: type[SAC] = SACSeq
