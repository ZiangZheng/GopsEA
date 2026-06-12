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


"""
weights:
1 15 -16 20 6 30 30 100
"""


# 公共计算缓存
def _get_cached_basic_info(
    env: ManagerBasedRLEnv,
    command_name: str,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    # 1. 初始化缓存容器
    if not hasattr(env, "_custom_reward_cache"):
        env._custom_reward_cache = {
            "step_index": -1,
            "distance_xy": None,
            "distance_z": None,
            "rot_error": None,
        }

    cache = env._custom_reward_cache
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

    # 物块-目标位姿计算
    is_inline = ((distance_xy < 0.04) | (distance_z > 0.04)).float()
    is_close_enough_h = ((distance_xy < 0.04) & (distance_z < 0.04)).float()
    is_directed = (rot_error < 0.2).float()

    # 物块-夹爪位姿计算
    cube_pos_w = object.data.root_pos_w
    ee_w = ee_frame.data.target_pos_w[..., 0, :]
    object_ee_distance = torch.norm(cube_pos_w - ee_w, dim=1)
    dist_temp = torch.tanh(object_ee_distance / std)
    is_close_enough_objee = (object_ee_distance < 0.1).float()

    obj_w_quat = object.data.root_quat_w
    ee_w_quat = ee_frame.data.target_quat_w[..., 0, :]
    x_flip_quat = torch.tensor([0.0, 1.0, 0.0, 0.0], device=env.device).repeat(
        obj_w_quat.shape[0], 1
    )
    target_ee_quat = quat_mul(obj_w_quat, x_flip_quat)
    rot_error_eetar = quat_error_magnitude(ee_w_quat, target_ee_quat)

    # 夹爪计算
    finger_pos = robot.data.joint_pos[:, 7:9]
    gripper_opening = finger_pos.mean(dim=1)
    is_gripper_open = (gripper_opening > 0.03).float()

    rot_mask = torch.max(is_close_enough_objee, is_inline)

    return (
        1
        - (1 - is_close_enough_h * is_directed * is_gripper_open) * dist_temp
        + rot_mask * (3.14 - (1 - is_inline) * rot_error_eetar) * 0.3
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
    object: RigidObject = env.scene[object_cfg.name]
    ee_frame: FrameTransformer = env.scene[ee_frame_cfg.name]

    distance_xy, distance_z, _ = _get_cached_basic_info(
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

    return torch.where(
        ((is_high_enough) & (rot_error_eetar < 0.73) | is_close_enough_h), 1.0, 0.0
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

    # --- 1. 初始化持久化历史记录 (如果不存在) ---
    if not hasattr(env, "_drag_penalty_history"):
        # 创建一个全 False 的布尔张量，形状为 (num_envs,)
        env._drag_penalty_history = torch.zeros(
            env.num_envs, dtype=torch.bool, device=env.device
        )

    # --- 2. 处理环境重置 ---
    # 如果当前步数是 0，说明刚刚重置过，必须清除历史记录
    # 注意：必须在计算当前步之前清除，否则会误删当前步刚产生的违规
    reset_indices = env.episode_length_buf == 0
    env._drag_penalty_history[reset_indices] = False

    # --- 3. 计算当前步的违规情况 (原有逻辑) ---
    target_pos_w, _ = combine_frame_transforms(
        robot.data.root_state_w[:, :3], robot.data.root_state_w[:, 3:7], command[:, :3]
    )

    distance_y = torch.abs(target_pos_w[:, 1])

    in_range = distance_y < 0.05
    object_height = object.data.root_pos_w[:, 2]
    on_table = object_height < 0.04  # minimal_height

    # 当前这一帧是否违规
    current_violation = in_range & on_table

    # --- 4. 更新历史记录 (状态锁定) ---
    # 逻辑：只要 历史上有过违规 或者 当前违规，就标记为 True
    env._drag_penalty_history = env._drag_penalty_history | current_violation

    # --- 5. 返回结果 ---
    # 返回记录的状态：只要触发过，就一直返回 1.0
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

    return is_inline * (1 - torch.tanh(distance / std))


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

    # 综合奖励
    return is_close_enough_h * (1 - torch.tanh(rot_error) + is_directed)


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

    return is_close_enough_h * is_put_down * (1 + is_directed)


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
    distance_xy, distance_z, rot_error = _get_cached_basic_info(
        env, command_name, robot_cfg, object_cfg
    )

    is_close_enough_h = (distance_xy < 0.04).float()
    is_directed = (rot_error < 0.2).float()

    # 机械臂末端远离物体奖励
    cube_pos_w = object.data.root_pos_w
    ee_w = ee_frame.data.target_pos_w[..., 0, :]
    object_ee_distance = torch.norm(cube_pos_w - ee_w, dim=1)
    is_far_enough = (object_ee_distance > 0.1).float()

    is_put_down = (distance_z < 0.04).float()

    finger_pos = robot.data.joint_pos[:, 7:9]
    gripper_opening = finger_pos.mean(dim=1)
    is_gripper_open = (gripper_opening > 0.03).float()

    # return is_close_enough_h * is_directed * is_put_down * is_far_enough
    return is_close_enough_h * is_directed * is_put_down * gripper_opening / 0.01


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

    is_close_enough = ((distance_xy < 0.06) & (distance_z < 0.04)).float()
    is_directed = (rot_error < 0.4).float()

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
    reach_the_dest = (dist_to_home < 0.05).float()

    finger_pos = robot.data.joint_pos[:, 7:9]
    gripper_opening = finger_pos.mean(dim=1)
    is_gripper_open = (gripper_opening > 0.03).float()

    # cube_pos_w = object.data.root_pos_w
    # ee_w = ee_frame.data.target_pos_w[..., 0, :]

    # object_ee_distance = torch.norm(cube_pos_w - ee_w, dim=1)

    return (
        is_close_enough
        * is_directed
        * is_gripper_open
        * (
            torch.tanh((0.3 - dist_to_home) / 0.3)
            + 1
            - torch.tanh(dist_to_home / 0.3)
            + reach_the_dest
        )
    )
