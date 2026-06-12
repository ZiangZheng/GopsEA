from typing import Callable, List, Dict

def transform_dict_mapping(obj: dict, mapping_dict):
    if mapping_dict is None: return obj
    return \
    {mapping_dict.get(k, k): v for k, v in obj.items() if k in mapping_dict}
    
def tranverse_dict_value(obj: dict, func: Callable):
    return \
    {k: func(v) for k, v in obj.items()}
    
def check_mapping_valid(mapping: Dict[str, str], names: List[str]):
    for k, v in mapping.items():
        assert v in names, f"mapping target : {v} not appeared. avialable terms {names}"