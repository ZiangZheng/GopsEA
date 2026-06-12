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
from isaaclab.utils.math import combine_frame_transforms, quat_error_magnitude, quat_mul

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv

"""
设置奖励终止条件是不科学的，如果需要改变，应加入更大的反向奖励，或者设置平台
可以使用势能控制来限制轨迹空间
可以通过写小惩罚来避免“抄近路”
tanh给的是趋势；如果需要稳定，则需要给一个大的实现奖励
如果奖励不易实现，可在途中增减离散奖励平台，避免回跳
"""


# 靠近物体
def object_ee_distance(
    env: ManagerBasedRLEnv,
    std: float,
    command_name: str,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
    ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame"),
) -> torch.Tensor:

    robot: RigidObject = env.scene[robot_cfg.name]
    object: RigidObject = env.scene[object_cfg.name]
    command = env.command_manager.get_command(command_name)
    ee_frame: FrameTransformer = env.scene[ee_frame_cfg.name]

    # 引导机械臂末端接近物体
    target_pos_w, target_rot_w = combine_frame_transforms(
        robot.data.root_state_w[:, :3],
        robot.data.root_state_w[:, 3:7],
        command[:, :3],
        command[:, 3:7],
    )

    distance_xy = torch.norm(target_pos_w[:, :2] - object.data.root_pos_w[:, :2], dim=1)
    rot_error = quat_error_magnitude(object.data.root_quat_w, target_rot_w)

    # 距离目标情况
    is_close_enough = (
        (distance_xy < 0.04) & (object.data.root_pos_w[:, 2] < 0.04)
    ).float()

    # 与目标姿态关系
    is_directed = (rot_error < 0.2).float()

    # 机械臂与物体距离
    cube_pos_w = object.data.root_pos_w
    ee_w = ee_frame.data.target_pos_w[..., 0, :]

    object_ee_distance = torch.norm(cube_pos_w - ee_w, dim=1)

    # 末端姿态与理想距离
    ee_w_quat = ee_frame.data.target_quat_w[..., 0, :]
    obj_w_quat = object.data.root_quat_w

    x_flip_quat = torch.tensor([0.0, 1.0, 0.0, 0.0], device=env.device).repeat(
        obj_w_quat.shape[0], 1
    )

    target_ee_quat = quat_mul(obj_w_quat, x_flip_quat)

    rot_error_eetar = quat_error_magnitude(ee_w_quat, target_ee_quat)

    # 机械臂与物体距离
    is_close_enough_objee = (object_ee_distance < 0.1).float()

    # 进入空间
    is_inline = ((distance_xy < 0.04) | (object.data.root_pos_w[:, 2] > 0.04)).float()

    # 隔开最初一段
    rot_mask = torch.max(is_close_enough_objee, is_inline)

    dist_temp = torch.tanh(object_ee_distance / std)
    # if (is_close_enough * is_directed == 1.0) & (object_ee_distance > 0.1):
    #     dist_temp = 0.8  # 或1

    # condition = (is_close_enough * is_directed == 1.0) & (object_ee_distance > 0.1)
    # dist_temp = torch.where(condition, torch.full_like(dist_temp, 0.8), dist_temp)

    finger_pos = robot.data.joint_pos[:, 7:9]
    gripper_opening = finger_pos.mean(dim=1)
    is_gripper_open = gripper_opening > 0.03

    return (
        1
        - 20 * (0.05 - is_close_enough * is_directed * is_gripper_open) * dist_temp
        + rot_mask * (3.14 - (1 - is_inline) * rot_error_eetar) * 0.1
    )


# 进入势能空间
def object_is_lifted(
    env: ManagerBasedRLEnv,
    minimal_height: float,
    command_name: str,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
    ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame"),
) -> torch.Tensor:
    robot: RigidObject = env.scene[robot_cfg.name]
    object: RigidObject = env.scene[object_cfg.name]
    command = env.command_manager.get_command(command_name)
    ee_frame: FrameTransformer = env.scene[ee_frame_cfg.name]

    # 判断物体是否进入势能谷
    target_pos_w, _ = combine_frame_transforms(
        robot.data.root_state_w[:, :3], robot.data.root_state_w[:, 3:7], command[:, :3]
    )

    distance_xy = torch.norm(target_pos_w[:, :2] - object.data.root_pos_w[:, :2], dim=1)

    is_high_enough = object.data.root_pos_w[:, 2] > minimal_height
    is_close_enough = distance_xy < 0.04

    ee_w_quat = ee_frame.data.target_quat_w[..., 0, :]
    obj_w_quat = object.data.root_quat_w

    x_flip_quat = torch.tensor([0.0, 1.0, 0.0, 0.0], device=env.device).repeat(
        obj_w_quat.shape[0], 1
    )

    target_ee_quat = quat_mul(obj_w_quat, x_flip_quat)

    rot_error = quat_error_magnitude(ee_w_quat, target_ee_quat)

    return torch.where(
        ((is_high_enough) & (rot_error < 0.77) | is_close_enough), 1.0, 0.0
    )


# 拖动惩罚
def object_move_on_table(
    env: ManagerBasedRLEnv,
    minimal_height: float,
    command_name: str,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
) -> torch.Tensor:
    robot: RigidObject = env.scene[robot_cfg.name]
    object: RigidObject = env.scene[object_cfg.name]
    command = env.command_manager.get_command(command_name)

    # 惩罚物体在桌子上推动/拖动
    target_pos_w, _ = combine_frame_transforms(
        robot.data.root_state_w[:, :3], robot.data.root_state_w[:, 3:7], command[:, :3]
    )

    distance_y = torch.abs(target_pos_w[:, 1] - 0.5)

    in_range = distance_y < 0.05
    object_height = object.data.root_pos_w[:, 2]
    on_table = object_height < minimal_height
    return torch.where(in_range & on_table, 1.0, 0.0)


# 靠近物体
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

    # 计算物体目标距离
    target_pos_w, _ = combine_frame_transforms(
        robot.data.root_state_w[:, :3], robot.data.root_state_w[:, 3:7], command[:, :3]
    )

    distance = torch.norm(target_pos_w - object.data.root_pos_w[:, :3], dim=1)
    distance_xy = torch.norm(target_pos_w[:, :2] - object.data.root_pos_w[:, :2], dim=1)

    # 判断物体是否进入势能谷
    is_high_enough = object.data.root_pos_w[:, 2] > minimal_height
    is_close_enough = distance_xy < success_threshold

    is_inline = (is_high_enough | is_close_enough).float()

    return is_inline * (1 - torch.tanh(distance / std))


def orientation_error(
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

    # 计算物体目标距离
    target_pos_w, target_rot_w = combine_frame_transforms(
        robot.data.root_state_w[:, :3],
        robot.data.root_state_w[:, 3:7],
        command[:, :3],
        command[:, 3:7],
    )

    distance_xy = torch.norm(target_pos_w[:, :2] - object.data.root_pos_w[:, :2], dim=1)

    # 判断物体是否进入势能谷
    # is_high_enough = object.data.root_pos_w[:, 2] > minimal_height
    # is_close_enough = distance_xy < success_threshold

    # is_inline = (is_high_enough | is_close_enough).float()

    rot_error = quat_error_magnitude(object.data.root_quat_w, target_rot_w)

    is_directed = (rot_error < 0.2).float()

    # 综合奖励
    return (distance_xy < 0.1).float() * (1 - torch.tanh(rot_error) + is_directed)


# 放下物体
def object_put_down(
    env: ManagerBasedRLEnv,
    success_threshold: float,
    command_name: str,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
) -> torch.Tensor:

    robot: RigidObject = env.scene[robot_cfg.name]
    object: RigidObject = env.scene[object_cfg.name]
    command = env.command_manager.get_command(command_name)

    # 判断物体是否放好
    target_pos_w, _ = combine_frame_transforms(
        robot.data.root_state_w[:, :3], robot.data.root_state_w[:, 3:7], command[:, :3]
    )

    distance_xy = torch.norm(target_pos_w[:, :2] - object.data.root_pos_w[:, :2], dim=1)
    is_close_enough = (distance_xy < success_threshold).float()

    # 物体放下奖励
    is_put_down = (object.data.root_pos_w[:, 2] < 0.04).float()

    return is_close_enough * is_put_down


# 松开物体
def let_object_go(
    env: ManagerBasedRLEnv,
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

    # 判断物体是否放好
    # 引导机械臂末端接近物体
    target_pos_w, target_rot_w = combine_frame_transforms(
        robot.data.root_state_w[:, :3],
        robot.data.root_state_w[:, 3:7],
        command[:, :3],
        command[:, 3:7],
    )

    distance_xy = torch.norm(target_pos_w[:, :2] - object.data.root_pos_w[:, :2], dim=1)
    rot_error = quat_error_magnitude(object.data.root_quat_w, target_rot_w)

    is_close_enough = (distance_xy < success_threshold).float()
    is_directed = (rot_error < 0.2).float()

    # 机械臂末端远离物体奖励
    cube_pos_w = object.data.root_pos_w
    ee_w = ee_frame.data.target_pos_w[..., 0, :]

    object_ee_distance = torch.norm(cube_pos_w - ee_w, dim=1)

    is_put_down = (object.data.root_pos_w[:, 2] < 0.04).float()

    finger_pos = robot.data.joint_pos[:, 7:9]
    gripper_opening = finger_pos.mean(dim=1)
    is_gripper_open = gripper_opening > 0.03

    is_far_enough = (object_ee_distance > 0.1).float()

    # return is_close_enough * is_directed * is_put_down * (1 + is_far_enough)
    return (
        is_close_enough
        * is_directed
        * is_put_down
        * (1 + gripper_opening / 0.1 + is_gripper_open * (1 + is_far_enough))
    )


# 完成放置后回中
def go_back(
    env: ManagerBasedRLEnv,
    std: float,
    command_name: str,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
    ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame"),
) -> torch.Tensor:

    robot: RigidObject = env.scene[robot_cfg.name]
    object: RigidObject = env.scene[object_cfg.name]
    command = env.command_manager.get_command(command_name)
    ee_frame: FrameTransformer = env.scene[ee_frame_cfg.name]

    # 判断物体是否放好
    target_pos_w, target_rot_w = combine_frame_transforms(
        robot.data.root_state_w[:, :3],
        robot.data.root_state_w[:, 3:7],
        command[:, :3],
        command[:, 3:7],
    )

    distance_xy = torch.norm(target_pos_w[:, :2] - object.data.root_pos_w[:, :2], dim=1)
    rot_error = quat_error_magnitude(object.data.root_quat_w, target_rot_w)

    is_close_enough = (
        (distance_xy < 0.04) & (object.data.root_pos_w[:, 2] < 0.04)
    ).float()
    is_directed = (rot_error < 0.2).float()

    # 引导回中
    home_pos_b = torch.tensor([0.5, 0.0, 0.4], device=env.device).repeat(
        env.num_envs, 1
    )
    home_pos_w, _ = combine_frame_transforms(
        robot.data.root_state_w[:, :3], robot.data.root_state_w[:, 3:7], home_pos_b
    )
    ee_w = ee_frame.data.target_pos_w[..., 0, :]
    dist_to_home = torch.norm(ee_w - home_pos_w, dim=1)

    # 完成回中后奖励
    reach_the_dest = (dist_to_home < 0.04).float()

    # cube_pos_w = object.data.root_pos_w
    # ee_w = ee_frame.data.target_pos_w[..., 0, :]

    # object_ee_distance = torch.norm(cube_pos_w - ee_w, dim=1)
    # is_far_enough = (object_ee_distance > 0.1).float()

    finger_pos = robot.data.joint_pos[:, 7:9]
    gripper_opening = finger_pos.mean(dim=1)
    is_gripper_open = gripper_opening > 0.03

    return (
        is_close_enough
        * is_directed
        * is_gripper_open
        * (1 - torch.tanh(dist_to_home / std) + reach_the_dest)
    )
