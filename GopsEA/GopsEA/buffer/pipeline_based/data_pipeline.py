from dataclasses import dataclass
from GopsEA import configclass
from typing import Dict, List
from .component_cfg import ComponentCfg
from GopsEA.utils.mapping import transform_dict_mapping

ENV_DIM_KEYS = [
    "policy_dim", "critic_dim", "dynamic_dim", "action_dim", "rewards_dim",
]

ENV_DATA_KEYS = [
    "timeout", "termination", "policy", "critic", "dynamic", "action", "reward", "rewards"
]

@dataclass
class DataPipeline:
    """
    Docstring for DataPipeline, {ENV_DIM_KEYS}
    """
    
    REPLAY_BUFFER_COMP: ComponentCfg
    MAPPING_CONSTRUCTION_ENV_DIM_2_REPLAYBUFFER_DIM: Dict[str, str]
    MAPPING_PROCESS_ENV_SAMPLE_2_REPLAYBUFFER_DATA: Dict[str, str]

    def buffer_dim_params(self, dim_params):
        return transform_dict_mapping(dim_params,  self.MAPPING_CONSTRUCTION_ENV_DIM_2_REPLAYBUFFER_DIM)

    @classmethod
    def construct_from_names(
            cls, 
            comp_cfg=ComponentCfg, 
            replay_buffer_dim_names=List[str], 
            replay_buffer_data_names=List[str], 
        ):
        """
        Docstring for construct_from_names
        
        :param cls: Description
        :param comp_cfg: Description
        :param replay_buffer_dim_names: Description {ENV_DIM_KEYS}
        :param replay_buffer_data_names: Description {ENV_DATA_KEYS}
        """
        MAPPING_CONSTRUCTION_ENV_DIM_2_REPLAYBUFFER_DIM = {k: v for k, v in zip(ENV_DIM_KEYS, replay_buffer_dim_names) if v is not None}
        MAPPING_PROCESS_ENV_SAMPLE_2_REPLAYBUFFER_DATA = {k: v for k, v in zip(ENV_DATA_KEYS, replay_buffer_data_names) if v is not None}
        return cls(comp_cfg, MAPPING_CONSTRUCTION_ENV_DIM_2_REPLAYBUFFER_DIM, MAPPING_PROCESS_ENV_SAMPLE_2_REPLAYBUFFER_DATA)
        
    def verbose(cfg) -> str:
        def _block(title: str) -> str:
            width = len(title) + 4
            return (
                " " + "┌" + "─" * width + "┐\n"
                + f" │  {title}  │\n"
                + " " + "└" + "─" * width + "┘"
            )

        def _mapping(title: str, d: dict) -> str:
            lines = [title]
            max_k = max(len(k) for k in d)
            for k, v in d.items():
                lines.append(f"   {k:<{max_k}} → {v}")
            return "\n".join(lines)

        text = []

        text.append("\n┌──────────────────────────────────────────────┐")
        text.append("│               DATA PIPELINE                  │")
        text.append("└──────────────────────────────────────────────┘\n")

        text.append(_block("ENV (dimension fields)"))
        text.append("      │")
        text.append("      ▼")
        text.append(_mapping(
            "[ENV DIM → REPLAY BUFFER DIM]",
            cfg.MAPPING_CONSTRUCTION_ENV_DIM_2_REPLAYBUFFER_DIM
        ))
        text.append("      │")
        text.append("      ▼")
        text.append(_block("REPLAY BUFFER (components)"))

        text.append(_block("ENV (sample fields)"))
        text.append("      │")
        text.append("      ▼")
        text.append(_mapping(
            "[ENV SAMPLE → REPLAY BUFFER DATA]",
            cfg.MAPPING_PROCESS_ENV_SAMPLE_2_REPLAYBUFFER_DATA
        ))
        text.append("      │")
        text.append("      ▼")
        text.append(_block("REPLAY BUFFER (data)"))

        text.append(_block("REPLAY BUFFER COMPONENT SCHEMA"))
        text.append("   " + str(cfg.REPLAY_BUFFER_COMP))

        return "\n".join(text)

    
    def check(cfg) -> str:
        return

@configclass
class DataPipelineCfg:
    component_cfg               : ComponentCfg = None
    replay_buffer_dim_names     : List[str] = None
    replay_buffer_data_names    : List[str] = None
    
    def construct_from_cfg(self):
        return DataPipeline.construct_from_names(
            self.component_cfg, self.replay_buffer_dim_names, self.replay_buffer_data_names
        )


if __name__ == "__main__":
    pipeline = \
    DataPipeline(
        REPLAY_BUFFER_COMP = ComponentCfg(
            ["policy",  "dynamic", "action",  "reward",  "rewards", "termination", "timeout"],
            ["float32", "float32", "float32", "float32", "float32", "bool",        "bool"   ],
            [-1,        -1,        -1,        1,         -1,        1,             1        ]
        ),
        MAPPING_PROCESS_ENV_SAMPLE_2_REPLAYBUFFER_DATA = {
            "policy":         "policy", 
            "dynamic":        "dynamic",
            "reward":         "reward",
            "rewards":        "rewards",
            "timeout":        "timeout",
            "action":         "action",
            "termination":    "termination",
        },
        MAPPING_CONSTRUCTION_ENV_DIM_2_REPLAYBUFFER_DIM = {
            "dynamic_dim"      : "dynamic",
            "policy_dim"       : "policy",
            "action_dim"       : "action",
            "rewards_dim"      : "rewards",
        },
    )

    print(pipeline.verbose())