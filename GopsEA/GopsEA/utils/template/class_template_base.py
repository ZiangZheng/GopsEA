from __future__ import annotations

import pickle
from GopsEA import configclass


class ClassTemplateBase(object):
    cfg: ClassTemplateBaseCfg
    @staticmethod
    def construct_from_cfg(cfg: ClassTemplateBaseCfg, *args, **kwargs):
        return cfg.class_type(cfg, *args, **kwargs)

    def save_cfg(self, path):
        """
        Save cfg from obj
        """
        save_dict = {
            "cfg": self.cfg
        }
        
class ClassTemplateBaseCfg:
    class_type: type[ClassTemplateBase] = ClassTemplateBase
    
    def construct_from_cfg(self, *args, **kwargs):
        return self.class_type(self, *args, **kwargs)