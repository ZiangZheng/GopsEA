# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from isaaclab.assets import RigidObject
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import FrameTransformer
from isaaclab.utils.math import combine_frame_transforms

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def _get_tensor(tensor_data, device):
    """辅助函数，确保数据转换为 Tensor 并在正确的设备上"""
    return torch.tensor(tensor_data, device=device, dtype=torch.float32)


def object_ee_distance(
    env: ManagerBasedRLEnv,
    std: float,
    success_threshold: float,
    command_name: str,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
    ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame"),
) -> torch.Tensor:
    robot: RigidObject = env.scene[robot_cfg.name]
    object: RigidObject = env.scene[object_cfg.name]
    command = env.command_manager.get_command(command_name)
    ee_frame: FrameTransformer = env.scene[ee_frame_cfg.name]

    target_pos_w, _ = combine_frame_transforms(
        robot.data.root_state_w[:, :3], robot.data.root_state_w[:, 3:7], command[:, :3]
    )

    distance_xy = torch.norm(target_pos_w[:, :2] - object.data.root_pos_w[:, :2], dim=1)

    is_far_enough = (distance_xy > success_threshold).float()

    cube_pos_w = object.data.root_pos_w
    ee_w = ee_frame.data.target_pos_w[..., 0, :]

    object_ee_distance = torch.norm(cube_pos_w - ee_w, dim=1)

    return is_far_enough * (1 - torch.tanh(object_ee_distance / std))


def object_is_lifted(
    env: ManagerBasedRLEnv,
    success_threshold: float,
    minimal_height: float,
    command_name: str,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
) -> torch.Tensor:
    robot: RigidObject = env.scene[robot_cfg.name]
    object: RigidObject = env.scene[object_cfg.name]
    command = env.command_manager.get_command(command_name)

    target_pos_w, _ = combine_frame_transforms(
        robot.data.root_state_w[:, :3], robot.data.root_state_w[:, 3:7], command[:, :3]
    )

    # [优化] 计算 XY 平面距离 (2D)，防止提起动作干扰距离判断
    # 假设我们只关心物体是否被移开了原来的平面位置
    distance_xy = torch.norm(target_pos_w[:, :2] - object.data.root_pos_w[:, :2], dim=1)

    is_high_enough = object.data.root_pos_w[:, 2] > minimal_height
    is_far_enough = distance_xy > success_threshold

    return torch.where(is_high_enough & is_far_enough, 1.0, 0.0)


def object_at_target(
    env: ManagerBasedRLEnv,
    std: float,
    success_threshold: float,
    minimal_height: float,
    command_name: str,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
) -> torch.Tensor:
    robot: RigidObject = env.scene[robot_cfg.name]
    object: RigidObject = env.scene[object_cfg.name]
    command = env.command_manager.get_command(command_name)

    target_pos_w, _ = combine_frame_transforms(
        robot.data.root_state_w[:, :3], robot.data.root_state_w[:, 3:7], command[:, :3]
    )

    distance = torch.norm(target_pos_w - object.data.root_pos_w[:, :3], dim=1)
    distance_xy = torch.norm(target_pos_w[:, :2] - object.data.root_pos_w[:, :2], dim=1)

    is_lifted = (object.data.root_pos_w[:, 2] > minimal_height).float()
    is_close_enough = (distance_xy < success_threshold).float()

    condition_mask = torch.max(is_lifted, is_close_enough)

    return condition_mask * (1 - torch.tanh(distance / std))


def go_back(
    env: ManagerBasedRLEnv,
    std: float,
    success_threshold: float,
    command_name: str,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
    ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame"),
) -> torch.Tensor:
    robot: RigidObject = env.scene[robot_cfg.name]
    object: RigidObject = env.scene[object_cfg.name]
    command = env.command_manager.get_command(command_name)
    ee_frame: FrameTransformer = env.scene[ee_frame_cfg.name]

    target_pos_w, _ = combine_frame_transforms(
        robot.data.root_state_w[:, :3], robot.data.root_state_w[:, 3:7], command[:, :3]
    )

    distance_xy = torch.norm(target_pos_w[:, :2] - object.data.root_pos_w[:, :2], dim=1)
    # 物体距离目标足够近（即任务完成）
    is_close_enough = (distance_xy < success_threshold).float()

    ee_w = ee_frame.data.target_pos_w[..., 0, :]

    home_pos_b = torch.tensor([0.0, 0.5, 0.5], device=env.device).repeat(
        env.num_envs, 1
    )

    home_pos_w, _ = combine_frame_transforms(
        robot.data.root_state_w[:, :3], robot.data.root_state_w[:, 3:7], home_pos_b
    )

    # 获取 EE 的世界坐标
    ee_w = ee_frame.data.target_pos_w[..., 0, :]

    # 计算距离
    dist_to_home = torch.norm(ee_w - home_pos_w, dim=1)

    return is_close_enough * (1 - torch.tanh(dist_to_home / std))
