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


# 公共计算缓存
def _get_cached_basic_info(
    env: ManagerBasedRLEnv,
    command_name: str,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    # 1. 初始化缓存容器
    if not hasattr(env, "_custom_reward"):
        env._custom_reward = {
            "step_index": -1,
            "distance_xy": None,
            "distance_z": None,
            "rot_error": None,
        }

    cache = env._custom_reward
    current_step = int(env.common_step_counter)

    # 2. 检查是否需要更新
    if cache["step_index"] != current_step:
        robot: RigidObject = env.scene[robot_cfg.name]
        object: RigidObject = env.scene[object_cfg.name]
        command = env.command_manager.get_command(command_name)

        target_pos_w, target_quat_w = combine_frame_transforms(
            robot.data.root_state_w[:, :3],
            robot.data.root_state_w[:, 3:7],
            command[:, :3],
            command[:, 3:7],  # 传入指令中的姿态部分
        )

        distance_xy = torch.norm(
            target_pos_w[:, :2] - object.data.root_pos_w[:, :2], dim=1
        )
        distance_z = object.data.root_pos_w[:, 2]
        rot_error = quat_error_magnitude(object.data.root_quat_w, target_quat_w)

        # 更新缓存
        cache["step_index"] = current_step
        cache["distance_xy"] = distance_xy
        cache["distance_z"] = distance_z
        cache["rot_error"] = rot_error

    return cache["distance_xy"], cache["distance_z"], cache["rot_error"]


def _update_cached_mission_acomplishment(
    env: ManagerBasedRLEnv,
    command_name: str,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
):

    distance_xy, distance_z, rot_error = _get_cached_basic_info(
        env, command_name, robot_cfg, object_cfg
    )

    if not hasattr(env, "_mission_accomplished"):
        env._mission_accomplished = torch.zeros(
            env.num_envs, dtype=torch.bool, device=env.device
        )

    reset_indices = env.episode_length_buf == 1
    env._mission_accomplished[reset_indices] = False

    is_close_enough_h = distance_xy < 0.04
    is_put_down = distance_z < 0.04
    is_directed = rot_error < 0.2

    env._mission_accomplished = env._mission_accomplished | (
        is_close_enough_h & is_put_down & is_directed
    )

    # if env._mission_accomplished[0]:
    #     print(
    #         f"Env 0: Accomplished | Step: {env.episode_length_buf[0].item()} | Height: {distance_z[0].item():.4f}"
    #     )


def _get_cached_mission_acomplishment(
    env: ManagerBasedRLEnv,
) -> torch.Tensor:
    if not hasattr(env, "_mission_accomplished"):
        env._mission_accomplished = torch.zeros(
            env.num_envs, dtype=torch.bool, device=env.device
        )
    return env._mission_accomplished


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
    ee_frame: FrameTransformer = env.scene[ee_frame_cfg.name]

    distance_xy, distance_z, rot_error = _get_cached_basic_info(
        env, command_name, robot_cfg, object_cfg
    )

    # 物块-执行器距离计算
    cube_pos_w = object.data.root_pos_w
    ee_w = ee_frame.data.target_pos_w[..., 0, :]
    object_ee_distance = torch.norm(cube_pos_w - ee_w, dim=1)

    # 执行器-目标姿态距离计算
    obj_w_quat = object.data.root_quat_w
    ee_w_quat = ee_frame.data.target_quat_w[..., 0, :]
    x_flip_quat = torch.tensor([0.0, 1.0, 0.0, 0.0], device=env.device).repeat(
        obj_w_quat.shape[0], 1
    )
    target_ee_quat = quat_mul(obj_w_quat, x_flip_quat)
    rot_error_eetar = quat_error_magnitude(ee_w_quat, target_ee_quat)

    # 姿态作用域计算
    is_inline = ((distance_xy < 0.04) | (distance_z > 0.04)).float()
    is_close_enough_objee = (object_ee_distance < 0.1).float()
    rot_mask = torch.max(is_close_enough_objee, is_inline)

    _update_cached_mission_acomplishment(env, command_name, robot_cfg, object_cfg)

    return torch.where(
        _get_cached_mission_acomplishment(env),
        2.0,
        1
        - (1 - is_inline) * torch.tanh(object_ee_distance / std)
        + rot_mask * (3.14 - (1 - is_inline) * rot_error_eetar) * 0.3,
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
    ee_frame: FrameTransformer = env.scene[ee_frame_cfg.name]

    distance_xy, distance_z, rot_error = _get_cached_basic_info(
        env, command_name, robot_cfg, object_cfg
    )

    # 物块-目标位姿计算
    is_high_enough = distance_z > minimal_height
    is_close_enough_h = distance_xy < 0.04

    # 物块-夹爪位姿计算
    ee_w_quat = ee_frame.data.target_quat_w[..., 0, :]
    obj_w_quat = object.data.root_quat_w
    x_flip_quat = torch.tensor([0.0, 1.0, 0.0, 0.0], device=env.device).repeat(
        obj_w_quat.shape[0], 1
    )
    target_ee_quat = quat_mul(obj_w_quat, x_flip_quat)
    rot_error_eetar = quat_error_magnitude(ee_w_quat, target_ee_quat)

    _update_cached_mission_acomplishment(env, command_name, robot_cfg, object_cfg)

    return torch.where(
        (
            _get_cached_mission_acomplishment(env)
            | (is_high_enough & (rot_error_eetar < 0.73))
            | is_close_enough_h
        ),
        1.0,
        0.0,
    )


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

    if not hasattr(env, "_drag_penalty_history"):
        env._drag_penalty_history = torch.zeros(
            env.num_envs, dtype=torch.bool, device=env.device
        )

    reset_indices = env.episode_length_buf == 1
    env._drag_penalty_history[reset_indices] = False

    target_pos_w, _ = combine_frame_transforms(
        robot.data.root_state_w[:, :3], robot.data.root_state_w[:, 3:7], command[:, :3]
    )

    distance_y = torch.abs(target_pos_w[:, 1])

    in_range = distance_y < 0.05
    object_height = object.data.root_pos_w[:, 2]
    on_table = object_height < 0.04  # minimal_height

    current_violation = in_range & on_table

    env._drag_penalty_history = env._drag_penalty_history | current_violation

    return torch.where(env._drag_penalty_history, 1.0, 0.0)


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
    object: RigidObject = env.scene[object_cfg.name]

    distance_xy, distance_z, _ = _get_cached_basic_info(
        env, command_name, robot_cfg, object_cfg
    )

    distance = torch.sqrt(distance_xy**2 + distance_z**2)

    # 判断物体是否进入势能谷
    is_high_enough = object.data.root_pos_w[:, 2] > minimal_height
    is_close_enough_h = distance_xy < 0.04

    is_inline = (is_high_enough | is_close_enough_h).float()

    _update_cached_mission_acomplishment(env, command_name, robot_cfg, object_cfg)

    return torch.where(
        _get_cached_mission_acomplishment(env),
        2.0,
        is_inline * (1 - torch.tanh(distance / std) + is_close_enough_h),
    )


def orientation_error(
    env: ManagerBasedRLEnv,
    success_threshold: float,
    minimal_height: float,
    command_name: str,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
) -> torch.Tensor:

    distance_xy, _, rot_error = _get_cached_basic_info(
        env, command_name, robot_cfg, object_cfg
    )

    is_close_enough_h = (distance_xy < 0.1).float()

    is_directed = (rot_error < 0.2).float()

    mission_accomplished = getattr(
        env, "_mission_accomplished", torch.zeros_like(rot_error, dtype=torch.bool)
    )

    decay_scale = torch.where(mission_accomplished, 0.1, 1.0)
    effective_rot_error = rot_error * decay_scale

    _update_cached_mission_acomplishment(env, command_name, robot_cfg, object_cfg)

    # 综合奖励
    return torch.where(
        _get_cached_mission_acomplishment(env),
        2.0,
        is_close_enough_h * (1 - torch.tanh(effective_rot_error) + is_directed),
    )


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
    distance_xy, distance_z, rot_error = _get_cached_basic_info(
        env, command_name, robot_cfg, object_cfg
    )

    is_close_enough_h = (distance_xy < 0.04).float()

    is_directed = (rot_error < 0.2).float()

    # 物体放下奖励
    is_put_down = (distance_z < 0.04).float()

    _update_cached_mission_acomplishment(env, command_name, robot_cfg, object_cfg)

    return torch.where(
        _get_cached_mission_acomplishment(env),
        1,
        is_close_enough_h * is_put_down,
    )


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
    ee_frame: FrameTransformer = env.scene[ee_frame_cfg.name]

    # 判断物体是否放好
    # 引导机械臂末端接近物体
    # distance_xy, distance_z, rot_error = _get_cached_basic_info(
    #     env, command_name, robot_cfg, object_cfg
    # )

    # is_close_enough_h = (distance_xy < 0.04).float()
    # is_directed = (rot_error < 0.2).float()

    # 机械臂末端远离物体奖励
    cube_pos_w = object.data.root_pos_w
    ee_w = ee_frame.data.target_pos_w[..., 0, :]
    object_ee_distance = torch.norm(cube_pos_w - ee_w, dim=1)
    is_far_enough = (object_ee_distance > 0.1).float()

    # is_put_down = (distance_z < 0.04).float()

    finger_pos = robot.data.joint_pos[:, 7:9]
    gripper_opening = finger_pos.mean(dim=1)
    # is_gripper_open = (gripper_opening > 0.03).float()

    _update_cached_mission_acomplishment(env, command_name, robot_cfg, object_cfg)

    return torch.where(
        _get_cached_mission_acomplishment(env),
        gripper_opening / 0.01 + is_far_enough,
        0,
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
    distance_xy, distance_z, rot_error = _get_cached_basic_info(
        env, command_name, robot_cfg, object_cfg
    )

    # is_close_enough = ((distance_xy < 0.04) & (distance_z < 0.04)).float()
    is_put_down = (distance_z < 0.04).float()
    # is_directed = (rot_error < 0.4).float()

    # 引导回中
    home_pos_b = torch.tensor([0.5, -0.2, 0.3], device=env.device).repeat(
        env.num_envs, 1
    )
    home_pos_w, _ = combine_frame_transforms(
        robot.data.root_state_w[:, :3], robot.data.root_state_w[:, 3:7], home_pos_b
    )
    ee_w = ee_frame.data.target_pos_w[..., 0, :]
    dist_to_home = torch.norm(ee_w - home_pos_w, dim=1)

    # 完成回中后奖励
    reach_the_dest = (dist_to_home < 0.04).float()

    finger_pos = robot.data.joint_pos[:, 7:9]
    gripper_opening = finger_pos.mean(dim=1)
    is_gripper_open = (gripper_opening > 0.03).float()

    # cube_pos_w = object.data.root_pos_w
    # ee_w = ee_frame.data.target_pos_w[..., 0, :]

    # object_ee_distance = torch.norm(cube_pos_w - ee_w, dim=1)

    _update_cached_mission_acomplishment(env, command_name, robot_cfg, object_cfg)

    return torch.where(
        _get_cached_mission_acomplishment(env),
        is_put_down * (1 - torch.tanh(dist_to_home / 0.3) + reach_the_dest),
        0,
    )
