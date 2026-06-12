from RenforceRL import configclass
from RenforceRL.utils.template.module_base import ModuleBase, ModuleBaseCfg

@configclass
class ModuleList(ModuleBaseCfg):
    module_list: list[ModuleBase] = [] # type: ignore
    
    def construct_from_cfg(self, *args, **kwargs):
        return [module.construct_from_cfg(self, *args, **kwargs) for module in self.module_list]