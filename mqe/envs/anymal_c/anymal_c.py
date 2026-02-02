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
from mqe.envs.anymal_c.anymal_c_config import AnymalCCfg
from mqe.utils.math import quat_apply_yaw, wrap_to_pi, torch_rand_sqrt_float
from mqe.utils.helpers import class_to_dict

class AnymalC(LeggedRobotField):
    def __init__(self, cfg: AnymalCCfg, sim_params, physics_engine, sim_device, headless):

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

        legged_gym default observation structure (48 dims):
        Structure (MUST match legged_gym/envs/base/legged_robot.py):
          [0:3]   base_lin_vel (scaled by obs_scales.lin_vel)
          [3:6]   base_ang_vel (scaled by obs_scales.ang_vel)
          [6:9]   projected_gravity (NOT scaled)
          [9:12]  commands [vx, vy, vyaw] (scaled by command_scales)
          [12:24] dof_pos (scaled by obs_scales.dof_pos)
          [24:36] dof_vel (scaled by obs_scales.dof_vel)
          [36:48] previous actions (raw policy outputs)
        """

        # [0:3] base_lin_vel (already scaled in obs_buf)
        self.locomotion_obs[:, 0:3] = self.obs_buf.lin_vel

        # [3:6] base_ang_vel (already scaled in obs_buf)
        self.locomotion_obs[:, 3:6] = self.obs_buf.ang_vel

        # [6:9] projected_gravity (NOT scaled)
        self.locomotion_obs[:, 6:9] = self.obs_buf.projected_gravity

        # [9:12] Fill velocity commands from mid-level actions [vx, vy, vyaw]
        if self.cfg.command.cfg.vel:
            self.locomotion_obs[:, 9:12] = actions[:, self.vel_idx : self.vel_idx + 3] * torch.tensor(
                [self.cfg.control.obs_scales.lin_vel,
                 self.cfg.control.obs_scales.lin_vel,
                 self.cfg.control.obs_scales.ang_vel],
                device=self.device
            )

        # [12:24] dof_pos (already scaled in obs_buf)
        self.locomotion_obs[:, 12:24] = self.obs_buf.dof_pos

        # [24:36] dof_vel (already scaled in obs_buf)
        self.locomotion_obs[:, 24:36] = self.obs_buf.dof_vel

        # [36:48] previous actions (raw policy outputs from last step)
        self.locomotion_obs[:, 36:48] = self.last_locomotion_action

        # Call policy (no history buffer needed for legged_gym default policy)
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
        self.gait_indices[agent_ids] = 0

        # Reset LSTM actuator network hidden states for reset environments
        if hasattr(self, 'sea_hidden_state_per_env'):
            self.sea_hidden_state_per_env[:, agent_ids] = 0.0
            self.sea_cell_state_per_env[:, agent_ids] = 0.0

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

        if self.cfg.obs.cfgs.clock_inputs or self.cfg.control.control_type == "C":
            assert self.cfg.control.control_type == "C", "To active clock_inputs, control_type should be set to \"C\" instead of \"{}\"".format(self.cfg.control.control_type)
            self.obs_buf.clock_inputs = copy(self.clock_inputs)

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
        self._step_contact_targets()

        if self.cfg.domain_rand.push_robots and  (self.common_step_counter % self.cfg.domain_rand.push_interval == 0):
            self._push_robots()

    def _step_contact_targets(self):
        if self.cfg.obs.cfgs.clock_inputs or self.cfg.control.control_type == "C":
            frequencies = self.locomotion_obs[:, 7]
            phases = self.locomotion_obs[:, 8]
            offsets = self.locomotion_obs[:, 9]
            bounds = self.locomotion_obs[:, 10]
            durations = self.locomotion_obs[:, 11]
            self.gait_indices = torch.remainder(self.gait_indices + self.dt * frequencies, 1.0)

            foot_indices = [self.gait_indices + phases + offsets + bounds,
                            self.gait_indices + offsets,
                            self.gait_indices + bounds,
                            self.gait_indices + phases]

            self.clock_inputs[:, 0] = torch.sin(2 * np.pi * foot_indices[0])
            self.clock_inputs[:, 1] = torch.sin(2 * np.pi * foot_indices[1])
            self.clock_inputs[:, 2] = torch.sin(2 * np.pi * foot_indices[2])
            self.clock_inputs[:, 3] = torch.sin(2 * np.pi * foot_indices[3])

    def _compute_torques(self, actions):
        """ Compute torques from actions.
            Actions can be interpreted as position or velocity targets given to a PD controller, or directly as scaled torques.
            [NOTE]: torques must have the same dimension as the number of DOFs, even if some DOFs are not actuated.

        Args:
            actions (torch.Tensor): Actions

        Returns:
            [torch.Tensor]: Torques sent to the simulation
        """
        #pd controller
        if isinstance(self.cfg.control.action_scale, (tuple, list)):
            self.cfg.control.action_scale = torch.tensor(self.cfg.control.action_scale, device= self.sim_device)
        actions_scaled = actions * self.cfg.control.action_scale
        actions_scaled = actions_scaled.reshape(-1, 12)
        actions_scaled[:, [0, 3, 6, 9]] *= self.cfg.control.hip_scale_reduction
        actions_scaled = actions_scaled.reshape(self.num_envs, -1)
        control_type = self.cfg.control.control_type

        if control_type == "C" or control_type == "control_net":

            if self.cfg.domain_rand.randomize_lag_timesteps:
                self.lag_buffer = self.lag_buffer[1:] + [actions_scaled.clone()]
                self.joint_pos_target = self.lag_buffer[0] + self.default_dof_pos
            else:
                self.joint_pos_target = actions_scaled + self.default_dof_pos

            # Compute position error and velocity for LSTM actuator network
            self.joint_pos_err = (self.dof_pos - self.joint_pos_target).reshape([-1, 12])
            self.joint_vel = self.dof_vel.reshape([-1, 12])

            # LSTM actuator network only needs current error and velocity (maintains internal state)
            torques = self.actuator_network(self.joint_pos_err, self.joint_vel)
            return torch.clip(torques, -self.torque_limits, self.torque_limits)
        else:
            return super()._compute_torques(actions)

    #----------------------------------------
    def _init_buffers(self):
        """ Initialize torch tensors which will contain simulation states and processed quantities
        """
        ### get gym GPU state tensors ###
        super()._init_buffers()

        self.lag_buffer = [torch.zeros_like(self.dof_pos, device=self.device) for i in range(self.cfg.domain_rand.lag_timesteps + 1)]

        if self.cfg.control.control_type == "actuator_net" or self.cfg.control.control_type == "C":

            # Anymal C uses LSTM actuator network (anydrive_v3_lstm.pt)
            # This matches the original legged_gym training setup
            actuator_network = torch.jit.load(self.cfg.control.actuator_network_path + "/anydrive_v3_lstm.pt", map_location=self.device)

            # Initialize LSTM hidden states
            # Shape: [2 layers, num_envs * num_agents * num_dof, 8 hidden]
            self.sea_input = torch.zeros(self.num_envs * self.num_agents * 12, 1, 2, device=self.device, requires_grad=False)
            self.sea_hidden_state = torch.zeros(2, self.num_envs * self.num_agents * 12, 8, device=self.device, requires_grad=False)
            self.sea_cell_state = torch.zeros(2, self.num_envs * self.num_agents * 12, 8, device=self.device, requires_grad=False)
            # View for per-environment reset
            self.sea_hidden_state_per_env = self.sea_hidden_state.view(2, self.num_envs * self.num_agents, 12, 8)
            self.sea_cell_state_per_env = self.sea_cell_state.view(2, self.num_envs * self.num_agents, 12, 8)

            def eval_actuator_network(joint_pos_err, joint_vel):
                # LSTM expects error as (target - actual), but we pass (actual - target)
                # So we need to NEGATE the position error
                self.sea_input[:, 0, 0] = -joint_pos_err.flatten()  # Negate for correct sign
                self.sea_input[:, 0, 1] = joint_vel.flatten()

                with torch.inference_mode():
                    torques, (self.sea_hidden_state[:], self.sea_cell_state[:]) = actuator_network(
                        self.sea_input, (self.sea_hidden_state, self.sea_cell_state)
                    )
                return torques.view(self.num_envs, self.num_actuated_dof)

            self.actuator_network = eval_actuator_network

    def _prepare_locomotion_policy(self):
        """
        Prepare Anymal C locomotion policy (legged_gym default).

        Unlike Go1's walk_these_ways (with history + adaptation module),
        Anymal C uses simple MLP policy with 48-dim current observations.
        """
        assert self.cfg.control.locomotion_policy_dir != None, "No locomotion policy provided."

        # Initialize 48-dim locomotion observation buffer
        locomotion_obs = self._fill_command_obs()
        self.locomotion_obs = locomotion_obs.repeat([self.num_envs * self.num_agents, 1])

        # Load single JIT policy file
        policy_model = torch.jit.load(self.cfg.control.locomotion_policy_dir + '/policy_1.pt', map_location=self.device)

        def policy(obs, info={}):
            with torch.no_grad():
                action = policy_model.forward(obs)
            return action

        self.locomotion_policy = policy

    def _fill_command_obs(self):
        """
        Fill command in locomotion observation with default values.

        legged_gym default observation structure (48 dims):
        Structure (MUST match legged_gym/envs/base/legged_robot.py):
          [0:3]   base_lin_vel (scaled)
          [3:6]   base_ang_vel (scaled)
          [6:9]   projected_gravity
          [9:12]  commands [vx, vy, vyaw] (scaled)
          [12:24] dof_pos (scaled)
          [24:36] dof_vel (scaled)
          [36:48] previous actions
        """

        idx = 0
        locomotion_obs = torch.zeros(1, 48, dtype=torch.float, device=self.device, requires_grad=False)

        # Default commands for velocity at indices [9:12] (will be overwritten by mid-level policy)
        if not self.cfg.command.cfg.vel:
            locomotion_obs[0, 9] = self.cfg.control.default_command.lin_vel_x * self.cfg.control.obs_scales.lin_vel
            locomotion_obs[0, 10] = self.cfg.control.default_command.lin_vel_y * self.cfg.control.obs_scales.lin_vel
            locomotion_obs[0, 11] = self.cfg.control.default_command.ang_vel * self.cfg.control.obs_scales.ang_vel
        else:
            self.vel_idx = idx
            idx += 3

        return locomotion_obs
