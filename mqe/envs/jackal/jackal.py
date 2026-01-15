"""
Jackal Wheeled Robot Environment

Clearpath Robotics Jackal - Differential drive robot
Action space: [vx, vy, vyaw] (same as Go1 for compatibility)
Low-level controller converts to wheel velocities

Author: Claude
Date: 2026-01-15
"""

import numpy as np
import torch

from isaacgym.torch_utils import *
from isaacgym import gymtorch, gymapi, gymutil

import torch
from torch import Tensor
from typing import Tuple, Dict
from copy import copy

from mqe import LEGGED_GYM_ROOT_DIR, envs
from mqe.envs.base.legged_robot import LeggedRobot
from mqe.envs.field.legged_robot_field import LeggedRobotField
from mqe.envs.jackal.jackal_config import JackalCfg
from mqe.utils.math import quat_apply_yaw, wrap_to_pi, torch_rand_sqrt_float
from mqe.utils.helpers import class_to_dict


class Jackal(LeggedRobotField):
    """Jackal wheeled robot for MAPush.

    Differential drive robot with high-level velocity control [vx, vy, vyaw].
    A differential drive controller converts these to wheel velocities.

    Action space: [vx, vy, vyaw] - same as Go1 for unified learning
    - vx: forward/backward velocity (m/s)
    - vy: lateral velocity (m/s) - limited due to non-holonomic constraint
    - vyaw: yaw rate (rad/s) - crucial for orientation control
    """

    def __init__(self, cfg: JackalCfg, sim_params, physics_engine, sim_device, headless):
        self.cfg = cfg
        self.env_name = cfg.env.env_name

        # Jackal physical parameters for differential drive
        self.wheel_radius = 0.098  # meters
        self.track_width = 0.37559  # meters (distance between wheels)

        super().__init__(cfg, sim_params, physics_engine, sim_device, headless)

        self.obs_buf = copy(self.cfg.obs)
        self.privileged_obs_buf = copy(self.cfg.privileged_obs)

        print(f"[Jackal] Initialized with {self.num_agents} agents, {self.num_dof} DOF per agent")
        print(f"[Jackal] Control: High-level [vx, vy, vyaw] with differential drive controller")
        print(f"[Jackal] Wheel radius: {self.wheel_radius}m, Track width: {self.track_width}m")

    def differential_drive_controller(self, vx, vy, vyaw):
        """Convert high-level velocity commands to wheel velocities.

        Differential drive kinematics for a two-wheeled robot:
        - Only vx (forward) and vyaw (rotation) can be directly achieved
        - vy (lateral) is constrained by non-holonomic dynamics

        Args:
            vx: Forward velocity (m/s) - shape [num_envs, num_agents]
            vy: Lateral velocity (m/s) - shape [num_envs, num_agents] (limited for diff drive)
            vyaw: Yaw rate (rad/s) - shape [num_envs, num_agents]

        Returns:
            wheel_velocities: Tensor of shape [num_envs, num_agents, 2]
                             where [:, :, 0] = left wheel, [:, :, 1] = right wheel (rad/s)
        """
        # For differential drive, we primarily use vx and vyaw
        # vy is difficult to achieve (requires coordinated forward motion + rotation)
        # For now, we ignore vy and warn if it's large

        # Differential drive equations:
        # left_wheel_vel = (vx - vyaw * track_width/2) / wheel_radius
        # right_wheel_vel = (vx + vyaw * track_width/2) / wheel_radius

        left_wheel_vel = (vx - vyaw * self.track_width / 2) / self.wheel_radius
        right_wheel_vel = (vx + vyaw * self.track_width / 2) / self.wheel_radius

        # Stack into [num_envs, num_agents, 2]
        wheel_velocities = torch.stack([left_wheel_vel, right_wheel_vel], dim=-1)

        return wheel_velocities

    def step(self, action):
        """Step the Jackal environment.

        Args:
            action: Tensor of shape [num_envs, num_agents, 3]
                    where each action is [vx, vy, vyaw]
        """
        # Convert high-level commands [vx, vy, vyaw] to wheel velocities
        # action shape: [num_envs, num_agents, 3]
        vx = action[..., 0]    # [num_envs, num_agents]
        vy = action[..., 1]    # [num_envs, num_agents]
        vyaw = action[..., 2]  # [num_envs, num_agents]

        # Convert to wheel velocities using differential drive controller
        wheel_actions = self.differential_drive_controller(vx, vy, vyaw)
        # wheel_actions shape: [num_envs, num_agents, 2]

        # Reshape for physics engine
        actions = wheel_actions.reshape(self.num_envs, -1)
        self.pre_physics_step(actions)

        # Step physics and render
        self.render()
        for dec_i in range(self.decimation):
            # Compute torques from wheel velocities
            self.torques = self._compute_torques(self.actions).view(self.torques.shape)

            # Add NPC actions if needed (for objects)
            if self.num_actions_npc != 0:
                torques = torch.cat((self.torques, torch.zeros((self.num_envs, self.num_actions_npc), dtype=torch.long, device=self.device)), dim=1)
            else:
                torques = self.torques

            self.gym.set_dof_actuation_force_tensor(self.sim, gymtorch.unwrap_tensor(torques))
            self.gym.simulate(self.sim)

            if self.device == 'cpu':
                self.gym.fetch_results(self.sim, True)
            self.gym.refresh_dof_state_tensor(self.sim)

            self.post_decimation_step(dec_i)

        self.post_physics_step()
        return self.obs_buf, self.rew_buf, self.reset_buf, self.extras
