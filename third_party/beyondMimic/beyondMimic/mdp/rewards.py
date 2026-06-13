from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import ContactSensor
from isaaclab.utils.math import quat_error_magnitude
from isaaclab.assets import Articulation, RigidObject
from isaaclab.utils.math import combine_frame_transforms, quat_error_magnitude, quat_mul, quat_apply

from beyondMimic.mdp.commands import MotionCommand

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def _get_body_indexes(command: MotionCommand, body_names: list[str] | None) -> list[int]:
    return [i for i, name in enumerate(command.cfg.body_names) if (body_names is None) or (name in body_names)]


def motion_global_anchor_position_error_exp(env: ManagerBasedRLEnv, command_name: str, std: float) -> torch.Tensor:
    command: MotionCommand = env.command_manager.get_term(command_name)
    error = torch.sum(torch.square(command.anchor_pos_w - command.robot_anchor_pos_w), dim=-1)
    return torch.exp(-error / std**2)


def motion_global_anchor_orientation_error_exp(env: ManagerBasedRLEnv, command_name: str, std: float) -> torch.Tensor:
    command: MotionCommand = env.command_manager.get_term(command_name)
    error = quat_error_magnitude(command.anchor_quat_w, command.robot_anchor_quat_w) ** 2
    return torch.exp(-error / std**2)


def motion_relative_body_position_error_exp(
    env: ManagerBasedRLEnv, command_name: str, std: float, body_names: list[str] | None = None
) -> torch.Tensor:
    command: MotionCommand = env.command_manager.get_term(command_name)
    body_indexes = _get_body_indexes(command, body_names)
    error = torch.sum(
        torch.square(command.body_pos_relative_w[:, body_indexes] - command.robot_body_pos_w[:, body_indexes]), dim=-1
    )
    return torch.exp(-error.mean(-1) / std**2)


def motion_relative_body_orientation_error_exp(
    env: ManagerBasedRLEnv, command_name: str, std: float, body_names: list[str] | None = None
) -> torch.Tensor:
    command: MotionCommand = env.command_manager.get_term(command_name)
    body_indexes = _get_body_indexes(command, body_names)
    error = (
        quat_error_magnitude(command.body_quat_relative_w[:, body_indexes], command.robot_body_quat_w[:, body_indexes])
        ** 2
    )
    return torch.exp(-error.mean(-1) / std**2)


def motion_global_body_linear_velocity_error_exp(
    env: ManagerBasedRLEnv, command_name: str, std: float, body_names: list[str] | None = None
) -> torch.Tensor:
    command: MotionCommand = env.command_manager.get_term(command_name)
    body_indexes = _get_body_indexes(command, body_names)
    error = torch.sum(
        torch.square(command.body_lin_vel_w[:, body_indexes] - command.robot_body_lin_vel_w[:, body_indexes]), dim=-1
    )
    return torch.exp(-error.mean(-1) / std**2)


def motion_global_body_angular_velocity_error_exp(
    env: ManagerBasedRLEnv, command_name: str, std: float, body_names: list[str] | None = None
) -> torch.Tensor:
    command: MotionCommand = env.command_manager.get_term(command_name)
    body_indexes = _get_body_indexes(command, body_names)
    error = torch.sum(
        torch.square(command.body_ang_vel_w[:, body_indexes] - command.robot_body_ang_vel_w[:, body_indexes]), dim=-1
    )
    return torch.exp(-error.mean(-1) / std**2)


def feet_contact_time(env: ManagerBasedRLEnv, sensor_cfg: SceneEntityCfg, threshold: float) -> torch.Tensor:
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    first_air = contact_sensor.compute_first_air(env.step_dt, env.physics_dt)[:, sensor_cfg.body_ids]
    last_contact_time = contact_sensor.data.last_contact_time[:, sensor_cfg.body_ids]
    reward = torch.sum((last_contact_time < threshold) * first_air, dim=-1)
    return reward

def body_ang_acc_l2(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Penalize the angular acceleration of bodies using L2-kernel."""
    asset: Articulation = env.scene[asset_cfg.name]
    return torch.sum(torch.norm(asset.data.body_ang_acc_w[:, asset_cfg.body_ids, :], dim=-1), dim=1)

def position_command_error_exp(env: ManagerBasedRLEnv, command_name: str, asset_cfg: SceneEntityCfg, std:float) -> torch.Tensor:
    """Penalize tracking of the position error using exp.

    The function computes the position error between the desired position (from the command) and the
    current position of the asset's body (in world frame). 
    """
    # extract the asset (to enable type hinting)
    asset: RigidObject = env.scene[asset_cfg.name]
    command = env.command_manager.get_command(command_name)
    # obtain the desired and current positions
    des_pos_b = command[:, :3]
    des_pos_w, _ = combine_frame_transforms(asset.data.root_pos_w, asset.data.root_quat_w, des_pos_b)
    curr_pos_w = asset.data.body_pos_w[:, asset_cfg.body_ids[0]]  # type: ignore
    distance = torch.norm(curr_pos_w - des_pos_w, dim=1)
    return torch.exp(-distance**2 / std**2)

def orientation_command_error_exp(env: ManagerBasedRLEnv, command_name: str, asset_cfg: SceneEntityCfg, std:float) -> torch.Tensor:
    """Penalize tracking of the orientation error using exp.

    The function computes the orientation error between the desired orientation (from the command) and the
    current orientation of the asset's body (in world frame). The orientation error is computed as the exp of the
    shortest path between the desired and current orientations.
    """
    # extract the asset (to enable type hinting)
    asset: RigidObject = env.scene[asset_cfg.name]
    command = env.command_manager.get_command(command_name)
    # obtain the desired and current orientations
    des_quat_b = command[:, 3:7]
    des_quat_w = quat_mul(asset.data.root_quat_w, des_quat_b)
    curr_quat_w = asset.data.body_quat_w[:, asset_cfg.body_ids[0]]  # type: ignore    
    
    ref_error = quat_error_magnitude(curr_quat_w, des_quat_w)
    # ref_error = torch.square(offset_body_vel_w[:,indices] - body_vel[:,indices])
    exp_ref_error = torch.exp(-ref_error/std**2)
    # rews = torch.mean(exp_ref_error, dim=[-1, -2])
    rews = exp_ref_error
    if torch.isnan(rews).any(): 
        print("nan in reference_traj")
    return rews

def linear_velocity_command_error_exp(env: ManagerBasedRLEnv, command_name: str, asset_cfg: SceneEntityCfg, std:float) -> torch.Tensor:
    """Penalize tracking of the linear velocity error using exp.

    The function computes the linear velocity error between the desired linear velocity (from the command) and the
    current linear velocity of the asset's body (in world frame). The linear velocity error is computed as the exp of the
    L2-norm of the difference between the desired and current linear velocities.
    """
    asset: RigidObject = env.scene[asset_cfg.name]
    command = env.command_manager.get_command(command_name)
    # obtain the desired and current positions
    des_lin_vel_b = command[:, 7:10]
    des_lin_vel_w = quat_apply(asset.data.root_quat_w, des_lin_vel_b)
    curr_lin_vel_w = asset.data.body_vel_w[:, asset_cfg.body_ids[0], :3]  # type: ignore
    distance = torch.norm(curr_lin_vel_w - des_lin_vel_w, dim=1)
    return torch.exp(-distance**2 / std**2)

def angular_velocity_command_error_exp(env: ManagerBasedRLEnv, command_name: str, asset_cfg: SceneEntityCfg, std:float) -> torch.Tensor:
    """Penalize tracking of the angular velocity error using exp.

    The function computes the angular velocity error between the desired angular velocity (from the command) and the
    current angular velocity of the asset's body (in world frame). The angular velocity error is computed as the exp of the
    L2-norm of the difference between the desired and current angular velocities.
    """
    asset: RigidObject = env.scene[asset_cfg.name]
    command = env.command_manager.get_command(command_name)
    # obtain the desired and current positions
    des_ang_vel_b = command[:, 10:13]
    des_ang_vel_w = quat_apply(asset.data.root_quat_w, des_ang_vel_b)
    curr_ang_vel_w = asset.data.body_vel_w[:, asset_cfg.body_ids[0], 3:]  # type: ignore
    distance = torch.norm(curr_ang_vel_w - des_ang_vel_w, dim=1)
    return torch.exp(-distance**2 / std**2)
