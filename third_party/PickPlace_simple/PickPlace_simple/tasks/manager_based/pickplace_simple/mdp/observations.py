# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from isaaclab.assets import RigidObject
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils.math import subtract_frame_transforms

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def object_position_in_robot_root_frame(
    env: ManagerBasedRLEnv,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
) -> torch.Tensor:
    """The position of the object in the robot's root frame."""
    robot: RigidObject = env.scene[robot_cfg.name]
    obj: RigidObject = env.scene[object_cfg.name]

    # 全部改用 root_state_w 提取
    object_pos_w = obj.data.root_state_w[:, :3]
    object_quat_w = obj.data.root_state_w[:, 3:7]

    # 计算相对变换
    object_pos_b, _ = subtract_frame_transforms(
        robot.data.root_state_w[:, :3],
        robot.data.root_state_w[:, 3:7],
        object_pos_w,
        object_quat_w,
    )
    return object_pos_b


def object_quaternion_in_robot_root_frame(
    env: ManagerBasedRLEnv,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
) -> torch.Tensor:
    """The quaternion of the object in the robot's root frame."""
    robot: RigidObject = env.scene[robot_cfg.name]
    obj: RigidObject = env.scene[object_cfg.name]

    # 全部改用 root_state_w 提取
    object_pos_w = obj.data.root_state_w[:, :3]
    object_quat_w = obj.data.root_state_w[:, 3:7]

    # 计算相对变换
    _, object_quat_b = subtract_frame_transforms(
        robot.data.root_state_w[:, :3],
        robot.data.root_state_w[:, 3:7],
        object_pos_w,
        object_quat_w,
    )
    return object_quat_b


def mission_accomplished_obs(env: ManagerBasedRLEnv) -> torch.Tensor:
    if hasattr(env, "_mission_accomplished"):
        return env._mission_accomplished.float().unsqueeze(-1)
    return torch.zeros((env.num_envs, 1), device=env.device)