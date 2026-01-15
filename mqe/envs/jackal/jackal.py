"""
Jackal Wheeled Robot Environment

Clearpath Robotics Jackal - Differential drive robot
2 DOF: [left_wheel_velocity, right_wheel_velocity]

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

    Differential drive robot with 2 DOF (left/right wheel velocities).
    Unlike Go1's hierarchical control, Jackal uses direct velocity control.
    """

    def __init__(self, cfg: JackalCfg, sim_params, physics_engine, sim_device, headless):
        self.cfg = cfg
        self.env_name = cfg.env.env_name
        super().__init__(cfg, sim_params, physics_engine, sim_device, headless)

        self.obs_buf = copy(self.cfg.obs)
        self.privileged_obs_buf = copy(self.cfg.privileged_obs)

        # Jackal uses direct wheel velocity control (no hierarchical policy needed)
        # Actions are directly [left_wheel_vel, right_wheel_vel]
        print(f"[Jackal] Initialized with {self.num_agents} agents, {self.num_dof} DOF per agent")
        print(f"[Jackal] Control type: Direct wheel velocity control")

    def step(self, action):
        """Step the Jackal environment.

        Args:
            action: Tensor of shape [num_envs, num_agents, 2]
                    where each action is [left_wheel_vel, right_wheel_vel]
        """
        # Clip and reshape actions
        actions = action.reshape(self.num_envs, -1)
        self.pre_physics_step(actions)

        # Step physics and render
        self.render()
        for dec_i in range(self.decimation):
            # For wheel velocity control, torques are proportional to desired velocities
            # (Isaac Gym can handle velocity control natively)
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
