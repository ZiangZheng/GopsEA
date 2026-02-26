#  Copyright (c). All Rights Reserved.
#  General Optimal control Problem Solver (GOPS)
#  Intelligent Driving Lab (iDLab), Tsinghua University
#
#  Creator: iDLab
#  Lab Leader: Prof. Shengbo Eben Li
#  Email: lisb04@gmail.com
#
#  Description: vehicle 3DOF data environment with tracking error constraint
#  Update: 2022-11-14, Yujie Yang: create environment

from typing import Any, Dict, Optional

import numpy as np

from gops_tasks.original.env.env_ocp.pyth_veh3dofconti import PythVeh3dofconti


class PythVeh3dofcontiErrCstr(PythVeh3dofconti):
    def __init__(
        self,
        pre_horizon: int = 10,
        path_para: Optional[Dict[str, Dict]] = None,
        u_para: Optional[Dict[str, Dict]] = None,
        y_error_tol: float = 0.2,
        u_error_tol: float = 2.0,
        **kwargs: Any,
    ):
        super().__init__(pre_horizon, path_para, u_para, **kwargs)
        self.y_error_tol = y_error_tol
        self.u_error_tol = u_error_tol
    
    @property
    def additional_info(self) -> Dict[str, Dict]:
        additional_info = super().additional_info
        return additional_info.update({"constraint": {"shape": (2,), "dtype": np.float32},})

    def get_constraint(self) -> np.ndarray:
        y, u = self.state[1], self.state[3]
        y_ref, u_ref = self.ref_points[0, 1], self.ref_points[0, 3]
        constraint = np.array(
            [abs(y - y_ref) - self.y_error_tol, abs(u - u_ref) - self.u_error_tol,],
            dtype=np.float32,
        )
        return constraint

    @property
    def info(self):
        info = super().info
        info.update(
            {"constraint": self.get_constraint(),}
        )
        return info


def env_creator(**kwargs):
    return PythVeh3dofcontiErrCstr(**kwargs)
