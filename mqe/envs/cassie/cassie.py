# Cassie Biped Robot for MAPush
#
# Based on AnymalC implementation pattern (48-dim legged_gym observation)
# Key differences from quadrupeds:
# - 12 DOFs (6 per leg) instead of 12 DOFs (3 per leg x 4 legs)
# - Spawn height: 1.0m (taller than quadrupeds)
# - Uses PD control (no actuator network)

import numpy as np
import torch

from isaacgym.torch_utils import *
from isaacgym import gymtorch, gymapi, gymutil

from torch import Tensor
from typing import Tuple, Dict
from copy import copy

from mqe import LEGGED_GYM_ROOT_DIR, envs
from mqe.envs.base.legged_robot import LeggedRobot
from mqe.envs.field.legged_robot_field import LeggedRobotField
from mqe.envs.cassie.cassie_config import CassieCfg
from mqe.utils.math import quat_apply_yaw, wrap_to_pi, torch_rand_sqrt_float
from mqe.utils.helpers import class_to_dict


class Cassie(LeggedRobotField):
    def __init__(self, cfg: CassieCfg, sim_params, physics_engine, sim_device, headless):

        self.cfg = cfg
        self.env_name = cfg.env.env_name
        super().__init__(cfg, sim_params, physics_engine, sim_device, headless)

        self.obs_buf = copy(self.cfg.obs)
        self.privileged_obs_buf = copy(self.cfg.privileged_obs)

        self.last_locomotion_action = torch.zeros(self.num_envs * self.num_agents, 12, dtype=torch.float, device=self.device, requires_grad=False)
        self.last_two_locomotion_action = torch.zeros(self.num_envs * self.num_agents, 12, dtype=torch.float, device=self.device, requires_grad=False)

        if self.cfg.control.control_type == "C":
            self._prepare_locomotion_policy()

    def step(self, action):

        if self.cfg.control.control_type == "C":
            action = torch.clip(action, -1, 1)
            action = self.preprocess_action(action)
            clip_actions = self.cfg.normalization.clip_actions
            self.actions = torch.clip(action, -clip_actions, clip_actions).reshape(self.num_envs, -1).to(self.device)
        else:
            actions = action.reshape(self.num_envs, -1)
            self.pre_physics_step(actions)
        # step physics and render each frame
        self.render()
        for dec_i in range(self.decimation):
            self.torques = self._compute_torques(self.actions).view(self.torques.shape)
            torques = torch.cat((self.torques, torch.zeros((self.num_envs, self.num_actions_npc), dtype=torch.long, device=self.device)), dim=1) if self.num_actions_npc != 0 else self.torques

            self.gym.set_dof_actuation_force_tensor(self.sim, gymtorch.unwrap_tensor(torques))
            self.gym.simulate(self.sim)
            if self.device == 'cpu':
                self.gym.fetch_results(self.sim, True)
            self.gym.refresh_dof_state_tensor(self.sim)

            self.post_decimation_step(dec_i)

        self.post_physics_step()
        return self.obs_buf, self.rew_buf, self.reset_buf, self.extras

    def preprocess_action(self, actions):
        """
        Preprocess mid-level actions into locomotion policy observations.

        Cassie uses legged_gym default 48-dim observation structure:
          [0:3]   base_lin_vel (scaled)
          [3:6]   base_ang_vel (scaled)
          [6:9]   projected_gravity
          [9:12]  commands [vx, vy, vyaw] (scaled)
          [12:24] dof_pos (relative to default, scaled)
          [24:36] dof_vel (scaled)
          [36:48] last_action
        """

        # Fill velocity commands from mid-level actions [vx, vy, vyaw]
        if self.cfg.command.cfg.vel:
            self.locomotion_obs[:, 9:12] = actions[:, self.vel_idx : self.vel_idx + 3] * torch.tensor(
                [self.cfg.control.obs_scales.lin_vel,
                 self.cfg.control.obs_scales.lin_vel,
                 self.cfg.control.obs_scales.ang_vel],
                device=self.device
            )

        # Fill core observations
        self.locomotion_obs[:, 0:3] = self.obs_buf.lin_vel
        self.locomotion_obs[:, 3:6] = self.obs_buf.ang_vel
        self.locomotion_obs[:, 6:9] = self.obs_buf.projected_gravity
        self.locomotion_obs[:, 12:24] = self.obs_buf.dof_pos
        self.locomotion_obs[:, 24:36] = self.obs_buf.dof_vel
        self.locomotion_obs[:, 36:48] = self.last_locomotion_action

        # Call policy
        locomotion_action = self.locomotion_policy(self.locomotion_obs)

        self.last_two_locomotion_action = self.last_locomotion_action
        self.last_locomotion_action = locomotion_action
        return locomotion_action

    def reset_idx(self, env_ids):
        """ Reset some environments.
            Calls self._reset_dofs(env_ids), self._reset_root_states(env_ids), and self._resample_commands(env_ids)
            [Optional] calls self._update_terrain_curriculum(env_ids), self.update_command_curriculum(env_ids) and
            Logs episode info
            Resets some buffers

        Args:
            env_ids (list[int]): List of environment ids which must be reset
        """
        self.reset_ids = env_ids
        if len(env_ids) == 0:
            return
        # update curriculum
        if self.cfg.terrain.curriculum:
            self._update_terrain_curriculum(env_ids)

        self._fill_extras(env_ids)

        # reset robot states
        self._reset_dofs(env_ids)
        self._reset_root_states(env_ids)

        self._reset_buffers(env_ids)

        self.store_recording(env_ids)

    def _reset_buffers(self, env_ids):
        super()._reset_buffers(env_ids)
        agent_ids = self.env_agent_indices[env_ids].reshape(-1)
        # Reset locomotion action history
        self.last_locomotion_action[agent_ids] = 0
        self.last_two_locomotion_action[agent_ids] = 0

    def reset(self):
        """ Reset all robots"""
        self.reset_idx(torch.arange(self.num_envs, device=self.device))
        self.compute_observations()
        return self.obs_buf

    def compute_observations(self):
        """ Computes observations
        """
        assert self.dof_pos.shape[1] % self.num_agents == 0, "DOF number is not compatible with agent number"
        dof_num = self.dof_pos.shape[1] // self.num_agents
        dof_pos = (self.dof_pos - self.default_dof_pos).reshape(-1, dof_num)
        dof_vel = self.dof_vel.reshape(-1, dof_num)

        if self.cfg.obs.cfgs.base_pos:
            self.obs_buf.base_pos = (self.base_pos - self.env_origins_repeat) * self.cfg.obs.scales.base_pos

        if self.cfg.obs.cfgs.base_quat:
            self.obs_buf.base_quat = self.base_quat * self.cfg.obs.scales.base_quat

        if self.cfg.obs.cfgs.dof_pos or self.cfg.control.control_type == "C":
            self.obs_buf.dof_pos = dof_pos * self.obs_scales.dof_pos

        if self.cfg.obs.cfgs.dof_vel or self.cfg.control.control_type == "C":
            self.obs_buf.dof_vel = dof_vel * self.obs_scales.dof_vel

        if self.cfg.obs.cfgs.lin_vel or self.cfg.control.control_type == "C":
            self.obs_buf.lin_vel = self.base_lin_vel * self.obs_scales.lin_vel

        if self.cfg.obs.cfgs.ang_vel or self.cfg.control.control_type == "C":
            self.obs_buf.ang_vel = self.base_ang_vel * self.obs_scales.ang_vel

        if self.cfg.obs.cfgs.last_action or self.cfg.control.control_type == "C":
            self.obs_buf.last_action = self.actions.reshape(-1, dof_num)

        if self.cfg.obs.cfgs.last_last_action or self.cfg.control.control_type == "C":
            self.obs_buf.last_last_action = self.last_actions.reshape(-1, dof_num)

        if self.cfg.obs.cfgs.projected_gravity or self.cfg.control.control_type == "C":
            self.obs_buf.projected_gravity = copy(self.projected_gravity)

        if self.cfg.obs.cfgs.base_rpy:
            self.obs_buf.base_rpy = torch.stack(get_euler_xyz(self.base_quat), dim=1)

        if self.cfg.obs.cfgs.env_info and hasattr(self, "env_info"):
            self.obs_buf.env_info = self.env_info

    def _prepare_reward_function(self):
        """ Prepares a list of reward functions, whcih will be called to compute the total reward.
            Looks for self._reward_<REWARD_NAME>, where <REWARD_NAME> are names of all non zero reward scales in the cfg.
        """
        # remove zero scales + multiply non-zero ones by dt
        for key in list(self.reward_scales.keys()):
            scale = self.reward_scales[key]
            if scale==0:
                self.reward_scales.pop(key)
            else:
                self.reward_scales[key] *= self.dt
        # prepare list of functions
        self.reward_functions = []
        self.reward_names = []
        for name, scale in self.reward_scales.items():
            if name=="termination":
                continue
            self.reward_names.append(name)

        # reward episode sums
        self.episode_sums = {name: torch.zeros(self.num_envs * self.num_agents, dtype=torch.float, device=self.device, requires_grad=False)
                             for name in self.reward_scales.keys()}

    def _post_physics_step_callback(self):
        """ Callback called before computing terminations, rewards, and observations
            Default behaviour: Compute ang vel command based on target and heading, compute measured terrain heights and randomly push robots
        """
        if self.cfg.domain_rand.push_robots and (self.common_step_counter % self.cfg.domain_rand.push_interval == 0):
            self._push_robots()

    def _compute_torques(self, actions):
        """ Compute torques from actions using PD control.
            Cassie uses PD control without actuator network.

        Args:
            actions (torch.Tensor): Actions (joint position targets from locomotion policy)

        Returns:
            [torch.Tensor]: Torques sent to the simulation
        """
        if isinstance(self.cfg.control.action_scale, (tuple, list)):
            self.cfg.control.action_scale = torch.tensor(self.cfg.control.action_scale, device=self.sim_device)

        actions_scaled = actions * self.cfg.control.action_scale
        actions_scaled = actions_scaled.reshape(self.num_envs, -1)
        control_type = self.cfg.control.control_type

        if control_type == "C" or control_type == "P":
            # PD control
            self.joint_pos_target = actions_scaled + self.default_dof_pos
            torques = self.p_gains * (self.joint_pos_target - self.dof_pos) - self.d_gains * self.dof_vel
            return torch.clip(torques, -self.torque_limits, self.torque_limits)
        else:
            return super()._compute_torques(actions)

    def _init_buffers(self):
        """ Initialize torch tensors which will contain simulation states and processed quantities
        """
        super()._init_buffers()

        # Cassie uses PD control, no actuator network needed
        # Just initialize the joint position target buffer
        self.joint_pos_target = torch.zeros_like(self.dof_pos)

    def _prepare_locomotion_policy(self):
        """
        Prepare Cassie locomotion policy (legged_gym default).

        Cassie uses simple MLP policy with 48-dim current observations.
        Structure:
          [0:3]   base_lin_vel (scaled)
          [3:6]   base_ang_vel (scaled)
          [6:9]   projected_gravity
          [9:12]  commands [vx, vy, vyaw] (scaled)
          [12:24] dof_pos (relative to default, scaled)
          [24:36] dof_vel (scaled)
          [36:48] last_action
        """
        assert self.cfg.control.locomotion_policy_dir is not None, "No locomotion policy provided for Cassie."

        # Initialize 48-dim locomotion observation buffer
        locomotion_obs = self._fill_command_obs()
        self.locomotion_obs = locomotion_obs.repeat([self.num_envs * self.num_agents, 1])

        # Load JIT policy file
        policy_path = self.cfg.control.locomotion_policy_dir + '/policy_1.pt'
        print(f"[Cassie] Loading locomotion policy from: {policy_path}")
        policy_model = torch.jit.load(policy_path, map_location=self.device)

        def policy(obs, info={}):
            with torch.no_grad():
                action = policy_model.forward(obs)
            return action

        self.locomotion_policy = policy
        print(f"[Cassie] Locomotion policy loaded successfully (48-dim obs -> 12-dim action)")

    def _fill_command_obs(self):
        """
        Fill command in locomotion observation with default command.

        Cassie policy expects 48 dims (legged_gym default):
          [0:3]   base_lin_vel (3)
          [3:6]   base_ang_vel (3)
          [6:9]   projected_gravity (3)
          [9:12]  commands [vx, vy, vyaw] (3)
          [12:24] dof_pos (12)
          [24:36] dof_vel (12)
          [36:48] last_action (12)
        """

        idx = 0
        locomotion_obs = torch.zeros(1, 48, dtype=torch.float, device=self.device, requires_grad=False)

        # Default commands for velocity (will be overwritten by mid-level policy)
        if not self.cfg.command.cfg.vel:
            locomotion_obs[0, 9] = self.cfg.control.default_command.lin_vel_x * self.cfg.control.obs_scales.lin_vel
            locomotion_obs[0, 10] = self.cfg.control.default_command.lin_vel_y * self.cfg.control.obs_scales.lin_vel
            locomotion_obs[0, 11] = self.cfg.control.default_command.ang_vel * self.cfg.control.obs_scales.ang_vel
        else:
            self.vel_idx = idx
            idx += 3

        return locomotion_obs
