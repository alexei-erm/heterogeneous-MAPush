"""
Heterogeneous Robot Base Class

This class extends LeggedRobotField to support heterogeneous agents
(multiple robot types in the same environment).

Author: Claude
Date: 2026-01-15
"""

import numpy as np
import torch
import os
from typing import List, Dict, Any

from isaacgym.torch_utils import *
from isaacgym import gymtorch, gymapi

from mqe import LEGGED_GYM_ROOT_DIR
from mqe.envs.field.legged_robot_field import LeggedRobotField
from mqe.envs.robot_registry import get_robot_class, get_robot_info
from mqe.utils.hetero_config import (
    get_hetero_asset_paths,
    get_hetero_action_dims,
    get_max_action_dim,
    validate_hetero_agents
)


class HeteroRobot(LeggedRobotField):
    """
    Heterogeneous robot environment that supports multiple robot types.

    This class extends LeggedRobotField to handle:
    - Loading different URDF files per agent
    - Different DOF counts per agent
    - Different action dimensions per agent
    - Proper indexing and state management for mixed robots
    """

    def __init__(self, cfg, sim_params, physics_engine, sim_device, headless):
        """
        Initialize heterogeneous robot environment.

        Args:
            cfg: Configuration with hetero_agent_types attribute
            sim_params: Isaac Gym simulation parameters
            physics_engine: Physics engine type
            sim_device: Simulation device
            headless: Whether to run headless
        """
        # Validate hetero configuration
        if not hasattr(cfg, 'hetero_agent_types'):
            raise ValueError(
                "Config must have 'hetero_agent_types' attribute for HeteroRobot. "
                "Use create_hetero_config() to create proper config."
            )

        self.hetero_agent_types = cfg.hetero_agent_types

        # Validate agents
        is_valid, message = validate_hetero_agents(self.hetero_agent_types)
        if not is_valid:
            raise ValueError(f"Invalid hetero agents: {message}")

        # Get action dimensions for each agent type
        self.hetero_action_dims = get_hetero_action_dims(self.hetero_agent_types)
        self.max_action_dim = get_max_action_dim(self.hetero_agent_types)

        # Store robot info for each agent type
        self.hetero_robot_info = [
            get_robot_info(robot_name) for robot_name in self.hetero_agent_types
        ]

        print(f"\n[HeteroRobot] Initializing with {len(self.hetero_agent_types)} agent types:")
        for i, robot_name in enumerate(self.hetero_agent_types):
            print(f"  Agent {i}: {robot_name} ({self.hetero_action_dims[i]} DOF)")

        # Initialize parent class
        super().__init__(cfg, sim_params, physics_engine, sim_device, headless)

        # Create action padding masks for each agent type
        self._create_action_masks()

        # Load locomotion policies for agents that need them
        if self.cfg.control.control_type == "C":
            self._load_locomotion_policies()

    def _load_locomotion_policies(self):
        """
        Load locomotion policies for each heterogeneous robot type.

        For each robot with control_type='C', load its locomotion policy.
        """
        from mqe.envs.robot_registry import get_robot_config

        self.locomotion_policies = []
        self.locomotion_obs_buffers = []

        for agent_idx, robot_type in enumerate(self.hetero_agent_types):
            robot_info = self.hetero_robot_info[agent_idx]

            if robot_info['default_control'] == 'C':
                # Load robot-specific config to get locomotion policy path
                robot_config = get_robot_config(robot_type)

                if hasattr(robot_config.control, 'locomotion_policy_dir') and robot_config.control.locomotion_policy_dir is not None:
                    policy_dir = robot_config.control.locomotion_policy_dir

                    # Check robot type to determine policy loading method
                    if robot_type == 'go1':
                        # Go1 uses walk_these_ways (body + adaptation_module)
                        body = torch.jit.load(policy_dir + '/body_latest.jit', map_location=self.device)
                        adaptation_module = torch.jit.load(policy_dir + '/adaptation_module_latest.jit', map_location=self.device)

                        def go1_policy(obs, info={}):
                            with torch.no_grad():
                                latent = adaptation_module.forward(obs)
                                action = body.forward(torch.cat((obs, latent), dim=-1))
                            info['latent'] = latent
                            return action

                        self.locomotion_policies.append(go1_policy)
                        # Go1 needs 70-dim obs + 2100-dim history
                        obs_buffer = torch.zeros(self.num_envs, 70, dtype=torch.float, device=self.device)
                        history_buffer = torch.zeros(self.num_envs, 2100, dtype=torch.float, device=self.device)
                        self.locomotion_obs_buffers.append({'obs': obs_buffer, 'history': history_buffer})

                        # Initialize gait indices for Go1 clock computation
                        if not hasattr(self, 'go1_gait_indices'):
                            self.go1_gait_indices = torch.zeros(self.num_envs, dtype=torch.float, device=self.device)
                            self.go1_clock_inputs = torch.zeros(self.num_envs, 4, dtype=torch.float, device=self.device)

                        # Load Go1's actuator network
                        if hasattr(robot_config.control, 'actuator_network_path'):
                            self._load_actuator_network(robot_config.control.actuator_network_path, robot_type, agent_idx)

                    elif robot_type == 'anymal_c':
                        # Anymal C uses single JIT policy (48-dim obs)
                        policy_model = torch.jit.load(policy_dir + '/policy_1.pt', map_location=self.device)

                        def anymal_policy(obs, info={}):
                            with torch.no_grad():
                                action = policy_model.forward(obs)
                            return action

                        self.locomotion_policies.append(anymal_policy)
                        # Anymal C needs 48-dim obs (no history buffer)
                        obs_buffer = torch.zeros(self.num_envs, 48, dtype=torch.float, device=self.device)
                        self.locomotion_obs_buffers.append({'obs': obs_buffer, 'history': None})

                        # Load Anymal C's actuator network (different from Go1!)
                        if hasattr(robot_config.control, 'actuator_network_path'):
                            self._load_actuator_network(robot_config.control.actuator_network_path, robot_type, agent_idx)

                    else:
                        raise NotImplementedError(f"Locomotion policy loading for {robot_type} not implemented")

                    print(f"[HeteroRobot] Loaded locomotion policy for agent {agent_idx} ({robot_type})")
                else:
                    raise ValueError(f"Robot {robot_type} has control_type='C' but no locomotion_policy_dir specified")
            else:
                # No locomotion policy needed (direct control)
                self.locomotion_policies.append(None)
                self.locomotion_obs_buffers.append(None)

    def _load_actuator_network(self, actuator_network_path, robot_type, agent_idx):
        """
        Load per-robot actuator network for torque computation.
        Different robots use different actuator networks:
        - Go1 -> unitree_go1.pt (feedforward, input: [pos_err, vel] x 3 timesteps)
        - Anymal C -> anydrive_v3_lstm.pt (LSTM, input: [pos_err, vel], needs hidden state)
        """
        # Initialize dict if needed
        if not hasattr(self, 'actuator_networks'):
            self.actuator_networks = {}

        # Determine which actuator network file to use
        if robot_type == 'go1':
            actuator_file = actuator_network_path + "/unitree_go1.pt"
            actuator_net = torch.jit.load(actuator_file, map_location=self.device)

            # Go1 uses feedforward network with history
            def eval_go1_actuator(joint_pos_err, joint_pos_err_last, joint_pos_err_last_last,
                                  joint_vel, joint_vel_last, joint_vel_last_last):
                # Stack inputs: [num_envs, num_dof, 6]
                xs = torch.cat((joint_pos_err.unsqueeze(-1),
                               joint_pos_err_last.unsqueeze(-1),
                               joint_pos_err_last_last.unsqueeze(-1),
                               joint_vel.unsqueeze(-1),
                               joint_vel_last.unsqueeze(-1),
                               joint_vel_last_last.unsqueeze(-1)), dim=-1)
                # Reshape for network: [num_envs * num_dof, 6]
                num_envs = joint_pos_err.shape[0]
                num_dof = joint_pos_err.shape[1]
                with torch.no_grad():
                    torques = actuator_net(xs.view(num_envs * num_dof, 6))
                return torques.view(num_envs, num_dof)

            self.actuator_networks[agent_idx] = eval_go1_actuator

        elif robot_type == 'anymal_c':
            actuator_file = actuator_network_path + "/anydrive_v3_lstm.pt"
            actuator_net = torch.jit.load(actuator_file, map_location=self.device)

            # Anymal C uses LSTM network - initialize hidden states
            # Shape: [2 layers, num_envs * num_dof, 8 hidden]
            num_dof = 12  # Anymal C has 12 DOFs
            sea_hidden = torch.zeros(2, self.num_envs * num_dof, 8, device=self.device)
            sea_cell = torch.zeros(2, self.num_envs * num_dof, 8, device=self.device)

            # Store hidden states for this agent
            setattr(self, f'anymal_sea_hidden_{agent_idx}', sea_hidden)
            setattr(self, f'anymal_sea_cell_{agent_idx}', sea_cell)

            # Anymal C LSTM actuator - different input format!
            # CRITICAL: legged_gym computes pos_err as (target - actual), but we compute (actual - target)
            # So we need to NEGATE the position error for Anymal C's LSTM
            def eval_anymal_actuator(joint_pos_err, joint_pos_err_last, joint_pos_err_last_last,
                                     joint_vel, joint_vel_last, joint_vel_last_last):
                num_envs = joint_pos_err.shape[0]
                num_dof = joint_pos_err.shape[1]

                # Get hidden states
                sea_hidden = getattr(self, f'anymal_sea_hidden_{agent_idx}')
                sea_cell = getattr(self, f'anymal_sea_cell_{agent_idx}')

                # Input format: [num_envs * num_dof, 1, 2] with [pos_err, vel]
                # NEGATE pos_err because legged_gym uses (target - actual) but we pass (actual - target)
                sea_input = torch.zeros(num_envs * num_dof, 1, 2, device=self.device)
                sea_input[:, 0, 0] = -joint_pos_err.flatten()  # NEGATED!
                sea_input[:, 0, 1] = joint_vel.flatten()

                with torch.inference_mode():
                    torques, (new_hidden, new_cell) = actuator_net(sea_input, (sea_hidden, sea_cell))

                # Update hidden states
                setattr(self, f'anymal_sea_hidden_{agent_idx}', new_hidden)
                setattr(self, f'anymal_sea_cell_{agent_idx}', new_cell)

                return torques.view(num_envs, num_dof)

            self.actuator_networks[agent_idx] = eval_anymal_actuator

        else:
            raise ValueError(f"Unknown robot type for actuator network: {robot_type}")

        print(f"[HeteroRobot] Loaded actuator network for agent {agent_idx} ({robot_type}) from {actuator_file}")

    def _create_action_masks(self):
        """
        Create masks for padding/unpadding actions to handle different action dimensions.

        For example, if agent0 has 3 DOF and agent1 has 2 DOF:
        - Networks see max_action_dim=3 for both
        - We mask out the extra dimension for agent1 before sending to sim
        """
        self.action_masks = []

        for i, action_dim in enumerate(self.hetero_action_dims):
            mask = torch.zeros(self.max_action_dim, dtype=torch.bool, device=self.device)
            mask[:action_dim] = True
            self.action_masks.append(mask)

        # Create per-environment masks (shape: [num_envs, num_agents, max_action_dim])
        self.env_action_masks = torch.zeros(
            self.num_envs, self.num_agents, self.max_action_dim,
            dtype=torch.bool, device=self.device
        )

        for j in range(self.num_agents):
            self.env_action_masks[:, j, :] = self.action_masks[j]

    def _create_envs(self):
        """
        Override _create_envs to load different URDF files per agent.

        This is the key modification that enables heterogeneous agents.
        """
        # Get asset paths for each agent type
        asset_paths = get_hetero_asset_paths(self.hetero_agent_types)

        # Load all robot assets
        robot_assets = []
        robot_dof_props_list = []
        robot_rigid_shape_props_list = []
        robot_num_dofs = []
        robot_num_bodies = []
        robot_torque_limits = []  # Store per-robot torque limits

        for i, asset_path_template in enumerate(asset_paths):
            asset_path = asset_path_template.format(LEGGED_GYM_ROOT_DIR=LEGGED_GYM_ROOT_DIR)
            asset_root = os.path.dirname(asset_path)
            asset_file = os.path.basename(asset_path)

            # Get per-robot config for this agent type
            from mqe.envs.robot_registry import get_robot_config
            robot_config = get_robot_config(self.hetero_agent_types[i])

            # Create asset options using per-robot config
            asset_options = gymapi.AssetOptions()
            asset_options.default_dof_drive_mode = robot_config.asset.default_dof_drive_mode
            asset_options.collapse_fixed_joints = robot_config.asset.collapse_fixed_joints
            asset_options.replace_cylinder_with_capsule = robot_config.asset.replace_cylinder_with_capsule
            asset_options.flip_visual_attachments = robot_config.asset.flip_visual_attachments
            asset_options.fix_base_link = robot_config.asset.fix_base_link
            asset_options.density = robot_config.asset.density
            asset_options.angular_damping = robot_config.asset.angular_damping
            asset_options.linear_damping = robot_config.asset.linear_damping
            asset_options.max_angular_velocity = robot_config.asset.max_angular_velocity
            asset_options.max_linear_velocity = robot_config.asset.max_linear_velocity
            asset_options.armature = robot_config.asset.armature
            asset_options.thickness = robot_config.asset.thickness
            asset_options.disable_gravity = robot_config.asset.disable_gravity

            # Load asset
            robot_asset = self.gym.load_asset(self.sim, asset_root, asset_file, asset_options)
            robot_assets.append(robot_asset)

            # Get DOF and body info
            num_dof = self.gym.get_asset_dof_count(robot_asset)
            num_bodies = self.gym.get_asset_rigid_body_count(robot_asset)
            dof_props = self.gym.get_asset_dof_properties(robot_asset)
            rigid_shape_props = self.gym.get_asset_rigid_shape_properties(robot_asset)

            robot_num_dofs.append(num_dof)
            robot_num_bodies.append(num_bodies)
            robot_dof_props_list.append(dof_props)
            robot_rigid_shape_props_list.append(rigid_shape_props)

            # Get torque limits for this robot type (robot_config already loaded above)
            if hasattr(robot_config.control, 'torque_limits'):
                robot_torque_limits.append(robot_config.control.torque_limits)
            else:
                # Default: no torque limits (will use None)
                robot_torque_limits.append(None)

            print(f"[HeteroRobot] Loaded {self.hetero_agent_types[i]}: {num_dof} DOF, {num_bodies} bodies")

        # Store for later use
        self.robot_assets = robot_assets
        self.robot_num_dofs = robot_num_dofs
        self.robot_num_bodies = robot_num_bodies
        self.robot_dof_props_list = robot_dof_props_list
        self.robot_rigid_shape_props_list = robot_rigid_shape_props_list
        self.robot_torque_limits = robot_torque_limits

        # For compatibility, use first robot as "primary"
        self.dof_names = self.gym.get_asset_dof_names(robot_assets[0])
        self.num_dof = robot_num_dofs[0]  # Primary robot's DOF
        self.num_actuated_dof = sum(robot_num_dofs)  # Total DOF across all agents

        # Prepare NPCs
        self._prepare_npc()

        # Get body names and indices from primary robot
        body_names = self.gym.get_asset_rigid_body_names(robot_assets[0])
        self.num_bodies = robot_num_bodies[0]

        feet_names = [s for s in body_names if self.cfg.asset.foot_name in s]
        penalized_contact_names = []
        for name in self.cfg.asset.penalize_contacts_on:
            penalized_contact_names.extend([s for s in body_names if name in s])
        termination_contact_names = []
        for name in self.cfg.asset.terminate_after_contacts_on:
            termination_contact_names.extend([s for s in body_names if name in s])

        # Initialize base states
        if getattr(self.cfg.init_state, "multi_init_state", False):
            init_state_list = []
            for idx, init_state in enumerate(self.cfg.init_state.init_states):
                base_init_state_list = init_state.pos + init_state.rot + init_state.lin_vel + init_state.ang_vel
                base_init_state = to_torch(base_init_state_list, device=self.device, requires_grad=False)

                # HETERO FIX: Override z-position with robot-specific init height
                # Each robot type has its own proper ground clearance
                from mqe.envs.robot_registry import get_robot_config
                robot_config = get_robot_config(self.hetero_agent_types[idx])
                robot_init_height = robot_config.init_state.pos[2]
                base_init_state[2] = robot_init_height
                # print(f"[HeteroRobot] Agent {idx} ({self.hetero_agent_types[idx]}): init height set to {robot_init_height}m")

                init_state_list.append(base_init_state)
                if idx == 0:
                    start_pose = gymapi.Transform()
                    start_pose.p = gymapi.Vec3(*base_init_state[:3])
            self.base_init_state = torch.stack(init_state_list, dim=0).repeat(self.num_envs, 1)
        else:
            base_init_state_list = self.cfg.init_state.pos + self.cfg.init_state.rot + self.cfg.init_state.lin_vel + self.cfg.init_state.ang_vel
            base_init_state = to_torch(base_init_state_list, device=self.device, requires_grad=False)
            start_pose = gymapi.Transform()
            start_pose.p = gymapi.Vec3(*base_init_state[:3])
            self.base_init_state = base_init_state.unsqueeze(0).repeat(self.num_agents * self.num_envs, 1)

        self._get_env_origins()
        env_lower = gymapi.Vec3(0., 0., 0.)
        env_upper = gymapi.Vec3(0., 0., 0.)

        self.npc_handles = []
        self.sensor_handles = []
        self.actor_handles = []
        self.envs = []

        self.default_friction = robot_rigid_shape_props_list[0][1].friction
        self.default_restitution = robot_rigid_shape_props_list[0][1].restitution

        self._init_custom_buffers__()

        # Create env and agent indices
        self.env_agent_indices = torch.zeros(self.num_envs, self.num_agents, dtype=torch.long, device=self.device)
        self.env_npc_indices = torch.zeros(self.num_envs, self.num_npcs, dtype=torch.long, device=self.device)

        for i in range(self.num_envs):
            for j in range(self.num_agents):
                self.env_agent_indices[i, j] = i * self.num_agents + j
            for j in range(self.num_npcs):
                self.env_npc_indices[i, j] = i * self.num_npcs + j

        # Create environments
        for i in range(self.num_envs):
            env_handle = self.gym.create_env(self.sim, env_lower, env_upper, int(np.sqrt(self.num_envs)))

            agent_handles = []
            sensor_handle_dicts = []

            # Create each agent with its specific robot type
            for j in range(self.num_agents):
                # Get the robot asset for this agent
                robot_asset = robot_assets[j]
                rigid_shape_props = robot_rigid_shape_props_list[j]
                dof_props_asset = robot_dof_props_list[j]

                # Process properties
                rigid_shape_props = self._process_rigid_shape_props(rigid_shape_props, i)
                self.gym.set_asset_rigid_shape_properties(robot_asset, rigid_shape_props)

                # Set initial position
                pos = self.env_origins[i].clone()
                pos[0:1] += torch_rand_float(-self.cfg.terrain.x_init_range, self.cfg.terrain.x_init_range, (1, 1), device=self.device).squeeze(1)
                pos[1:2] += torch_rand_float(-self.cfg.terrain.y_init_range, self.cfg.terrain.y_init_range, (1, 1), device=self.device).squeeze(1)

                # CRITICAL FIX: Set agent-specific spawn height
                from mqe.envs.robot_registry import get_robot_config
                robot_config = get_robot_config(self.hetero_agent_types[j])
                pos[2] = robot_config.init_state.pos[2]  # Set correct z-height for this agent

                start_pose.p = gymapi.Vec3(*pos)

                # Create actor with specific robot asset
                agent_name = f"{self.hetero_agent_types[j]}_agent{j}"
                agent_handle = self.gym.create_actor(
                    env_handle, robot_asset, start_pose,
                    agent_name, i, self.cfg.asset.self_collisions, 0
                )

                # Set DOF properties
                # Track current agent for _process_dof_props
                self._current_agent_idx = j
                dof_props = self._process_dof_props(dof_props_asset, i)
                self.gym.set_actor_dof_properties(env_handle, agent_handle, dof_props)

                # Set body properties
                body_props = self.gym.get_actor_rigid_body_properties(env_handle, agent_handle)
                body_props = self._process_rigid_body_props(body_props, i)
                self.gym.set_actor_rigid_body_properties(env_handle, agent_handle, body_props, recomputeInertia=True)

                # Create sensors
                sensor_handle_dict = self._create_sensors(env_handle, agent_handle)

                agent_handles.append(agent_handle)
                sensor_handle_dicts.append(sensor_handle_dict)

            # Create NPCs
            npc_handles = self._create_npc(env_handle, i)

            self.envs.append(env_handle)
            self.actor_handles.append(agent_handles + npc_handles)
            self.sensor_handles.append(sensor_handle_dicts)
            self.npc_handles.append(npc_handles)

        # Create actor indices
        self.actor_indices = torch.zeros(self.num_envs, self.num_agents + self.num_npcs, dtype=torch.int32, device=self.device)
        self.agent_indices = torch.zeros(self.num_envs, self.num_agents, dtype=torch.int32, device=self.device)
        self.npc_indices = torch.zeros(self.num_envs, self.num_npcs, dtype=torch.int32, device=self.device)

        for i in range(self.num_envs):
            for j in range(self.num_agents + self.num_npcs):
                self.actor_indices[i, j] = self.gym.get_actor_index(self.envs[i], self.actor_handles[i][j], gymapi.DOMAIN_SIM)
                if j < self.num_agents:
                    self.agent_indices[i, j] = self.gym.get_actor_index(self.envs[i], self.actor_handles[i][j], gymapi.DOMAIN_SIM)
                if j >= self.num_agents:
                    self.npc_indices[i, j - self.num_agents] = self.gym.get_actor_index(self.envs[i], self.actor_handles[i][j], gymapi.DOMAIN_SIM)

        # Get feet indices (from primary robot)
        self.feet_indices = torch.zeros(len(feet_names), dtype=torch.long, device=self.device)
        for i in range(len(feet_names)):
            self.feet_indices[i] = self.gym.find_actor_rigid_body_handle(self.envs[0], self.actor_handles[0][0], feet_names[i])

        # Get penalized contact indices
        self.penalised_contact_indices = torch.zeros(len(penalized_contact_names), dtype=torch.long, device=self.device)
        for i in range(len(penalized_contact_names)):
            self.penalised_contact_indices[i] = self.gym.find_actor_rigid_body_handle(self.envs[0], self.actor_handles[0][0], penalized_contact_names[i])

        # Get termination contact indices
        self.termination_contact_indices = torch.zeros(len(termination_contact_names), dtype=torch.long, device=self.device)
        for i in range(len(termination_contact_names)):
            self.termination_contact_indices[i] = self.gym.find_actor_rigid_body_handle(self.envs[0], self.actor_handles[0][0], termination_contact_names[i])

        # Initialize video recording attributes
        if self.cfg.env.record_video:
            from mqe.utils.helpers import FloatingCameraSensor
            self.rendering_camera = FloatingCameraSensor(self)

        self.video_writer = None
        self.video_frames = []
        self.complete_video_frames = []

        print(f"[HeteroRobot] Created {self.num_envs} environments with {self.num_agents} heterogeneous agents each")

    def pre_physics_step(self, actions):
        """
        Override to handle different action dimensions per agent.

        Args:
            actions: Tensor of shape [num_envs, num_agents, max_action_dim]
                     (padded to max dimension)
        """
        # Mask out padding for agents with fewer actions
        actions_masked = actions * self.env_action_masks.float()

        # Call parent implementation
        clip_actions = self.cfg.normalization.clip_actions
        self.actions = torch.clip(actions_masked, -clip_actions, clip_actions).to(self.device)

    def _process_dof_props(self, props, env_id):
        """
        Override to handle per-agent torque limits for heterogeneous robots.

        This method is called during environment creation for each agent.
        We use self._current_agent_idx to determine which robot's torque limits to use.

        Args:
            props: DOF properties from asset
            env_id: Environment ID

        Returns:
            Processed DOF properties
        """
        # CRITICAL FIX: Only call parent on first agent to avoid resetting torque_limits
        # Parent class initializes torque_limits = zeros(num_actuated_dof) which would
        # overwrite our per-agent torque limits setup
        if env_id == 0 and self._current_agent_idx == 0:
            # Call parent's parent (LeggedRobot._process_dof_props) only for first agent
            from mqe.envs.base.legged_robot import LeggedRobot
            props = LeggedRobot._process_dof_props(self, props, env_id)

            # Parent created buffers with total DOF size, we need to delete and rebuild
            # for per-agent handling
            if hasattr(self, 'torque_limits'):
                delattr(self, 'torque_limits')

        # Only process on first environment to set up torque limits
        if env_id == 0 and hasattr(self, '_current_agent_idx'):
            agent_idx = self._current_agent_idx

            torque_limits_cfg = self.robot_torque_limits[agent_idx]
            num_dof_this_agent = self.robot_num_dofs[agent_idx]

            if torque_limits_cfg is not None:
                if not isinstance(torque_limits_cfg, (tuple, list)):
                    # Scalar torque limit - apply to all DOFs of this robot
                    torque_limits_agent = torch.ones(num_dof_this_agent, dtype=torch.float, device=self.device) * torque_limits_cfg
                else:
                    # List of torque limits - repeat pattern to match DOF count
                    if num_dof_this_agent % len(torque_limits_cfg) == 0:
                        torque_limits_agent = torch.tensor(
                            torque_limits_cfg * (num_dof_this_agent // len(torque_limits_cfg)),
                            dtype=torch.float, device=self.device
                        )
                    else:
                        # DOF doesn't divide evenly, just use the first values
                        torque_limits_agent = torch.tensor(
                            torque_limits_cfg[:num_dof_this_agent],
                            dtype=torch.float, device=self.device
                        )

                # Initialize or extend torque_limits tensor
                if not hasattr(self, 'torque_limits'):
                    # First agent - initialize
                    self.torque_limits = torque_limits_agent
                else:
                    # Subsequent agents - concatenate
                    self.torque_limits = torch.cat([self.torque_limits, torque_limits_agent])

                print(f"[HeteroRobot] Agent {agent_idx} ({self.hetero_agent_types[agent_idx]}): "
                      f"{num_dof_this_agent} DOF, torque_limits: {torque_limits_agent.tolist()}")

            # Print final shape only after last agent
            if agent_idx == self.num_agents - 1:
                print(f"[HeteroRobot _create_envs] FINAL self.torque_limits.shape: {self.torque_limits.shape}")

        return props

    def _init_buffers(self):
        """
        Override to handle heterogeneous DOF counts when initializing buffers.

        We copy the entire parent _init_buffers() method but modify the section
        that initializes default_dof_pos, p_gains, and d_gains to support
        heterogeneous DOF counts.
        """
        ### get gym GPU state tensors ###
        actor_root_state = self.gym.acquire_actor_root_state_tensor(self.sim)
        dof_state_tensor = self.gym.acquire_dof_state_tensor(self.sim)
        net_contact_forces = self.gym.acquire_net_contact_force_tensor(self.sim)
        rigid_body_state = self.gym.acquire_rigid_body_state_tensor(self.sim)
        self.gym.refresh_dof_state_tensor(self.sim)
        self.gym.refresh_actor_root_state_tensor(self.sim)
        self.gym.refresh_net_contact_force_tensor(self.sim)
        self.gym.refresh_rigid_body_state_tensor(self.sim)
        self.gym.render_all_camera_sensors(self.sim)

        ### create some wrapper tensors for different slices ###

        # root state
        self.all_root_states = gymtorch.wrap_tensor(actor_root_state)
        self.root_states = self.all_root_states.view(self.num_envs, -1, 13)[:, :self.num_agents, :].reshape(-1, 13)
        self.base_pos = self.root_states[:, 0:3]
        self.base_quat = self.root_states[:, 3:7]
        self.prev_base_pos = self.base_pos.clone()
        self.root_states_npc = self.all_root_states.view(self.num_envs, -1, 13)[:, self.num_agents:, :].reshape(-1, 13)
        self.base_pos_npc = self.root_states_npc[:, 0:3]
        self.base_quat_npc = self.root_states_npc[:, 3:7]

        # dof state
        self.all_dof_states = gymtorch.wrap_tensor(dof_state_tensor)
        self.dof_state = self.all_dof_states.view(self.num_envs, -1, 2)[:, :self.num_actuated_dof, :]
        self.dof_pos = self.dof_state[:, :, 0]
        self.dof_vel = self.dof_state[:, :, 1]

        if self.num_actions_npc > 0:
            self.dof_state_npc = self.all_dof_states.view(self.num_envs, -1, 2)[:, self.num_actuated_dof:, :]
            self.dof_pos_npc = self.dof_state_npc[:, :, 0]
            self.dof_vel_npc = self.dof_state_npc[:, :, 1]

        # contact force
        self.contact_forces = gymtorch.wrap_tensor(net_contact_forces).view(self.num_envs, -1, 3)

        # initialize some data used later on
        self.common_step_counter = 0
        self.extras = {}

        self.gravity_vec = to_torch(get_axis_params(-1., self.up_axis_idx), device=self.device).repeat((self.num_envs * self.num_agents, 1))
        self.forward_vec = to_torch([1., 0., 0.], device=self.device).repeat((self.num_envs, 1))

        # For heterogeneous agents, torques should match actuated DOFs, not high-level actions
        self.torques = torch.zeros(self.num_envs, self.num_actuated_dof, dtype=torch.float, device=self.device, requires_grad=False)
        self.p_gains = torch.zeros(self.num_actuated_dof, dtype=torch.float, device=self.device, requires_grad=False)
        self.d_gains = torch.zeros(self.num_actuated_dof, dtype=torch.float, device=self.device, requires_grad=False)

        self.action = torch.zeros(self.num_envs * self.num_agents, self.num_action, dtype=torch.float, device=self.device, requires_grad=False)
        self.actions = torch.zeros(self.num_envs, self.num_actions, dtype=torch.float, device=self.device, requires_grad=False)
        self.last_actions = torch.zeros(self.num_envs, self.num_actions, dtype=torch.float, device=self.device, requires_grad=False)
        self.last_dof_vel = torch.zeros_like(self.dof_vel)
        self.last_root_vel = torch.zeros_like(self.root_states[:, 7:13])

        self.commands = torch.zeros(self.num_envs * self.num_agents, self.cfg.commands.num_commands, dtype=torch.float, device=self.device, requires_grad=False)
        self.commands_scale = torch.tensor([self.obs_scales.lin_vel, self.obs_scales.lin_vel, self.obs_scales.ang_vel], device=self.device, requires_grad=False,)
        self.desired_contact_states = torch.zeros(self.num_envs, 4, dtype=torch.float, device=self.device, requires_grad=False,)
        self.feet_air_time = torch.zeros(self.num_envs, self.feet_indices.shape[0], dtype=torch.float, device=self.device, requires_grad=False)
        self.last_contacts = torch.zeros(self.num_envs, len(self.feet_indices), dtype=torch.bool, device=self.device, requires_grad=False)
        self.base_lin_vel = quat_rotate_inverse(self.base_quat, self.root_states[:, 7:10])
        self.base_ang_vel = quat_rotate_inverse(self.base_quat, self.root_states[:, 10:13])
        self.projected_gravity = quat_rotate_inverse(self.base_quat, self.gravity_vec)
        self.substep_torques = torch.zeros(self.num_envs, self.decimation, self.num_actuated_dof, dtype=torch.float, device=self.device, requires_grad=False)
        self.substep_dof_vel = torch.zeros(self.num_envs, self.decimation, self.num_actuated_dof, dtype=torch.float, device=self.device, requires_grad=False)
        self.substep_exceed_dof_pos_limits = torch.zeros(self.num_envs, self.decimation, self.num_actuated_dof, dtype=torch.bool, device=self.device, requires_grad=False)

        # ============================================================
        # HETEROGENEOUS DOF HANDLING (MODIFIED SECTION)
        # ============================================================
        # Initialize default_dof_pos, p_gains, d_gains for heterogeneous agents

        # Get robot configs for each agent type
        from mqe.envs.robot_registry import get_robot_config
        robot_configs = [get_robot_config(robot_name) for robot_name in self.hetero_agent_types]

        # Compute DOF offsets for each agent
        dof_offsets = [0]
        for num_dof in self.robot_num_dofs:
            dof_offsets.append(dof_offsets[-1] + num_dof)

        # Initialize default_dof_pos tensor
        self.default_dof_pos = torch.zeros(
            self.num_actuated_dof,
            dtype=torch.float,
            device=self.device,
            requires_grad=False
        )

        # For each agent type
        for agent_idx in range(self.num_agents):
            robot_config = robot_configs[agent_idx]
            num_dof_agent = self.robot_num_dofs[agent_idx]
            dof_offset = dof_offsets[agent_idx]

            # Get DOF names for this robot
            robot_asset = self.robot_assets[agent_idx]
            dof_names_agent = self.gym.get_asset_dof_names(robot_asset)

            # Set default positions and PD gains for each DOF of this agent
            for i in range(num_dof_agent):
                dof_name = dof_names_agent[i]
                global_idx = dof_offset + i

                # Set default joint angle
                if hasattr(robot_config.init_state, 'default_joint_angles'):
                    if dof_name in robot_config.init_state.default_joint_angles:
                        angle = robot_config.init_state.default_joint_angles[dof_name]
                        self.default_dof_pos[global_idx] = angle
                    else:
                        # Default to 0.0 if not specified
                        self.default_dof_pos[global_idx] = 0.0
                else:
                    self.default_dof_pos[global_idx] = 0.0

                # Set PD gains
                found = False
                if hasattr(robot_config.control, 'stiffness') and hasattr(robot_config.control, 'damping'):
                    for dof_pattern in robot_config.control.stiffness.keys():
                        if dof_pattern in dof_name:
                            self.p_gains[global_idx] = robot_config.control.stiffness[dof_pattern]
                            self.d_gains[global_idx] = robot_config.control.damping[dof_pattern]
                            found = True
                            break

                if not found:
                    self.p_gains[global_idx] = 0.0
                    self.d_gains[global_idx] = 0.0

        # Unsqueeze for batch dimension
        self.default_dof_pos = self.default_dof_pos.unsqueeze(0)

        print(f"[HeteroRobot] Initialized buffers for {self.num_agents} heterogeneous agents:")
        for agent_idx in range(self.num_agents):
            dof_offset = dof_offsets[agent_idx]
            num_dof_agent = self.robot_num_dofs[agent_idx]
            print(f"  Agent {agent_idx} ({self.hetero_agent_types[agent_idx]}): "
                  f"DOF {dof_offset}-{dof_offset + num_dof_agent - 1} "
                  f"({num_dof_agent} DOFs)")

    def compute_observations(self):
        """
        Override compute_observations to handle heterogeneous action spaces.

        For heterogeneous robots with unified action space (e.g., both use 3 DOF [vx, vy, vyaw]),
        we need to use the action dimension instead of DOF count when reshaping actions.
        """
        # Check if parent class has compute_observations
        if not hasattr(super(), 'compute_observations'):
            return

        # For heterogeneous agents, we use unified action space (self.num_action per agent)
        # instead of varying DOF counts

        # First, call parent's logic but we'll need to patch the dof_num calculation
        # Save the original assertion check
        assert self.dof_pos.shape[1] == self.num_actuated_dof, \
            f"DOF mismatch: dof_pos has {self.dof_pos.shape[1]} but expected {self.num_actuated_dof}"

        # For heterogeneous agents, use action dimension (unified across agents)
        # instead of per-agent DOF count
        action_dim_per_agent = self.num_action  # This is 3 for both Go1 and Jackal

        # Compute per-agent DOF positions/velocities differently for hetero
        # We need to handle each agent's DOF separately
        dof_offsets = [0]
        for num_dof in self.robot_num_dofs:
            dof_offsets.append(dof_offsets[-1] + num_dof)

        # Create lists to hold per-agent observations
        dof_pos_per_agent = []
        dof_vel_per_agent = []

        for agent_idx in range(self.num_agents):
            start_idx = dof_offsets[agent_idx]
            end_idx = dof_offsets[agent_idx + 1]
            num_dof_agent = self.robot_num_dofs[agent_idx]

            # Extract this agent's DOFs for all environments
            agent_dof_pos = self.dof_pos[:, start_idx:end_idx]  # [num_envs, num_dof_agent]
            agent_dof_vel = self.dof_vel[:, start_idx:end_idx]

            # Subtract default positions for this agent
            default_pos_agent = self.default_dof_pos[0, start_idx:end_idx]
            agent_dof_pos = agent_dof_pos - default_pos_agent

            dof_pos_per_agent.append(agent_dof_pos)
            dof_vel_per_agent.append(agent_dof_vel)

        # Stack to create [num_envs * num_agents, max_dof] tensors
        # For now, we'll pad to the max DOF count
        max_dof = max(self.robot_num_dofs)
        dof_pos_padded = []
        dof_vel_padded = []

        for env_idx in range(self.num_envs):
            for agent_idx in range(self.num_agents):
                agent_dof_pos = dof_pos_per_agent[agent_idx][env_idx]
                agent_dof_vel = dof_vel_per_agent[agent_idx][env_idx]

                # Pad to max_dof
                if len(agent_dof_pos) < max_dof:
                    agent_dof_pos = torch.cat([
                        agent_dof_pos,
                        torch.zeros(max_dof - len(agent_dof_pos), device=self.device)
                    ])
                    agent_dof_vel = torch.cat([
                        agent_dof_vel,
                        torch.zeros(max_dof - len(agent_dof_vel), device=self.device)
                    ])

                dof_pos_padded.append(agent_dof_pos)
                dof_vel_padded.append(agent_dof_vel)

        dof_pos = torch.stack(dof_pos_padded)  # [num_envs * num_agents, max_dof]
        dof_vel = torch.stack(dof_vel_padded)

        # CRITICAL FIX: Recompute velocities from current root_states
        # These were only computed once during __init__ and become stale
        self.base_lin_vel[:] = quat_rotate_inverse(self.base_quat, self.root_states[:, 7:10])
        self.base_ang_vel[:] = quat_rotate_inverse(self.base_quat, self.root_states[:, 10:13])
        self.projected_gravity[:] = quat_rotate_inverse(self.base_quat, self.gravity_vec)

        # Now set observations using the unified action dimension
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
            # DEBUG: Print velocities on first observation computation
            if not hasattr(self, '_debug_printed_vel'):
                print(f"\n[DEBUG compute_observations] First call:")
                print(f"  root_states[0, 7:10] (global lin vel agent 0): {self.root_states[0, 7:10]}")
                print(f"  root_states[1, 7:10] (global lin vel agent 1): {self.root_states[1, 7:10]}")
                print(f"  base_lin_vel[0] (body frame agent 0): {self.base_lin_vel[0]}")
                print(f"  base_lin_vel[1] (body frame agent 1): {self.base_lin_vel[1]}")
                print(f"  obs_buf.lin_vel[0] (scaled agent 0): {self.obs_buf.lin_vel[0]}")
                print(f"  obs_buf.lin_vel[1] (scaled agent 1): {self.obs_buf.lin_vel[1]}")
                self._debug_printed_vel = True

        if self.cfg.obs.cfgs.ang_vel or self.cfg.control.control_type == "C":
            self.obs_buf.ang_vel = self.base_ang_vel * self.obs_scales.ang_vel

        # KEY FIX: Use action_dim_per_agent instead of dof_num for action reshaping
        if self.cfg.obs.cfgs.last_action or self.cfg.control.control_type == "C":
            self.obs_buf.last_action = self.actions.reshape(-1, action_dim_per_agent)

        if self.cfg.obs.cfgs.last_last_action or self.cfg.control.control_type == "C":
            self.obs_buf.last_last_action = self.last_actions.reshape(-1, action_dim_per_agent)

        if self.cfg.obs.cfgs.projected_gravity or self.cfg.control.control_type == "C":
            self.obs_buf.projected_gravity = self.projected_gravity

        if hasattr(self.cfg.obs.cfgs, 'clock_inputs') and (self.cfg.obs.cfgs.clock_inputs or self.cfg.control.control_type == "C"):
            if hasattr(self, 'clock_inputs'):
                from copy import copy
                self.obs_buf.clock_inputs = copy(self.clock_inputs)

        if hasattr(self.cfg.obs.cfgs, 'base_rpy') and self.cfg.obs.cfgs.base_rpy:
            from isaacgym.torch_utils import get_euler_xyz
            self.obs_buf.base_rpy = torch.stack(get_euler_xyz(self.base_quat), dim=1)

        if hasattr(self.cfg.obs.cfgs, 'env_info') and self.cfg.obs.cfgs.env_info and hasattr(self, "env_info"):
            self.obs_buf.env_info = self.env_info

    def _compute_torques(self, actions):
        """
        Override to handle heterogeneous DOF counts when computing torques.

        For heterogeneous agents, we need to:
        1. Process each agent's actions according to their DOF count
        2. Handle different control types per agent (hierarchical vs direct)
        3. Combine torques respecting the heterogeneous DOF layout
        """
        # Check control type
        control_type = self.cfg.control.control_type

        # For hierarchical control mode "C", we need special handling
        if control_type == "C" or control_type == "control_net":
            # actions shape: [num_envs, total_dofs] (joint positions from locomotion policies)
            # We need to process each agent separately with their specific action_scale

            # Compute DOF offsets for indexing
            dof_offsets = [0]
            for num_dof in self.robot_num_dofs:
                dof_offsets.append(dof_offsets[-1] + num_dof)

            # Process each agent's actions to get joint targets
            # CRITICAL: Apply per-robot action_scale, NOT task-level action_scale
            all_joint_targets = []

            # Get per-robot action scales
            from mqe.envs.robot_registry import get_robot_config

            for agent_idx in range(self.num_agents):
                # Get this agent's robot info
                robot_type = self.hetero_agent_types[agent_idx]
                robot_info = self.hetero_robot_info[agent_idx]
                num_dof_agent = self.robot_num_dofs[agent_idx]
                dof_start = dof_offsets[agent_idx]
                dof_end = dof_offsets[agent_idx + 1]

                # Get this robot's specific action_scale from its config
                robot_config = get_robot_config(robot_type)
                robot_action_scale = robot_config.control.action_scale

                # Get default positions for this agent
                default_pos_agent = self.default_dof_pos[0, dof_start:dof_end]  # [num_dof_agent]

                # For robots with hierarchical control
                # Actions are joint position residuals from locomotion policy
                if robot_info['default_control'] == 'C':
                    # Extract this agent's joint position residuals from actions
                    # actions shape: [num_envs, total_dofs]
                    joint_residuals = actions[:, dof_start:dof_end].clone()  # [num_envs, num_dof_agent]

                    # Apply robot-specific action scale
                    joint_residuals_scaled = joint_residuals * robot_action_scale

                    # Apply hip_scale_reduction for Go1 (and Anymal C has same structure)
                    # Hip joints are at indices 0, 3, 6, 9 for 12-DOF quadrupeds
                    if hasattr(robot_config.control, 'hip_scale_reduction') and num_dof_agent == 12:
                        hip_scale = robot_config.control.hip_scale_reduction
                        joint_residuals_scaled[:, [0, 3, 6, 9]] *= hip_scale

                    # Add to default positions to get joint targets
                    # target = residual * scale + default
                    joint_targets = joint_residuals_scaled + default_pos_agent
                    all_joint_targets.append(joint_targets)

                else:
                    # All robots in hetero mode should use hierarchical control
                    raise NotImplementedError(f"Robot type {robot_type} does not use hierarchical control")

            # Concatenate all joint targets
            self.joint_pos_target = torch.cat(all_joint_targets, dim=1)  # [num_envs, total_dofs]

            # DEBUG: Print actual joint targets on first step
            if not hasattr(self, '_debug_printed_targets'):
                print(f"\n[DEBUG _compute_torques] Joint targets:")
                print(f"  Go1 targets: {self.joint_pos_target[0, :12]}")
                print(f"  Go1 defaults: {self.default_dof_pos[0, :12]}")
                print(f"  Anymal C targets: {self.joint_pos_target[0, 12:24]}")
                print(f"  Anymal C defaults: {self.default_dof_pos[0, 12:24]}")
                self._debug_printed_targets = True

            # Compute torques for each agent
            all_torques = []

            for agent_idx in range(self.num_agents):
                robot_info = self.hetero_robot_info[agent_idx]
                dof_start = dof_offsets[agent_idx]
                dof_end = dof_offsets[agent_idx + 1]
                num_dof_agent = self.robot_num_dofs[agent_idx]

                # Extract this agent's DOF positions, velocities, and targets
                joint_pos_err_agent = (
                    self.dof_pos[:, dof_start:dof_end]
                    - self.joint_pos_target[:, dof_start:dof_end]
                )
                joint_vel_agent = self.dof_vel[:, dof_start:dof_end]

                # Compute torques based on control type
                if robot_info['default_control'] == 'C':
                    # Hierarchical control: use per-robot actuator network
                    if hasattr(self, 'actuator_networks') and agent_idx in self.actuator_networks:
                        # Initialize history buffers if needed
                        if not hasattr(self, f'joint_pos_err_last_{agent_idx}'):
                            setattr(self, f'joint_pos_err_last_{agent_idx}', joint_pos_err_agent.clone())
                            setattr(self, f'joint_pos_err_last_last_{agent_idx}', joint_pos_err_agent.clone())
                            setattr(self, f'joint_vel_last_{agent_idx}', joint_vel_agent.clone())
                            setattr(self, f'joint_vel_last_last_{agent_idx}', joint_vel_agent.clone())

                        # Get history
                        joint_pos_err_last = getattr(self, f'joint_pos_err_last_{agent_idx}')
                        joint_pos_err_last_last = getattr(self, f'joint_pos_err_last_last_{agent_idx}')
                        joint_vel_last = getattr(self, f'joint_vel_last_{agent_idx}')
                        joint_vel_last_last = getattr(self, f'joint_vel_last_last_{agent_idx}')

                        # Apply this agent's specific actuator network
                        torques_agent = self.actuator_networks[agent_idx](
                            joint_pos_err_agent,
                            joint_pos_err_last,
                            joint_pos_err_last_last,
                            joint_vel_agent,
                            joint_vel_last,
                            joint_vel_last_last
                        )

                        # Update history
                        setattr(self, f'joint_pos_err_last_last_{agent_idx}', joint_pos_err_last.clone())
                        setattr(self, f'joint_pos_err_last_{agent_idx}', joint_pos_err_agent.clone())
                        setattr(self, f'joint_vel_last_last_{agent_idx}', joint_vel_last.clone())
                        setattr(self, f'joint_vel_last_{agent_idx}', joint_vel_agent.clone())
                    else:
                        # Fallback to PD control
                        torques_agent = (
                            self.p_gains[dof_start:dof_end] * joint_pos_err_agent
                            - self.d_gains[dof_start:dof_end] * joint_vel_agent
                        )

                    all_torques.append(torques_agent)

                else:
                    # All robots in hetero mode should use hierarchical control
                    raise NotImplementedError(f"Robot type does not use hierarchical control")

            # Concatenate all torques
            torques = torch.cat(all_torques, dim=1)  # [num_envs, total_dofs]

            # Clip to torque limits
            torque_limits_expanded = self.torque_limits.unsqueeze(0).expand(self.num_envs, -1)
            return torch.clip(torques, -torque_limits_expanded, torque_limits_expanded)

        else:
            # For non-hierarchical control, use parent implementation
            return super()._compute_torques(actions)

    def step(self, action):
        """
        Override step to handle per-agent locomotion policies in heterogeneous mode.

        CRITICAL: In heterogeneous mode, we need to:
        1. Extract per-agent high-level actions [vx, vy, vyaw]
        2. Call each agent's locomotion policy to get joint positions
        3. Concatenate joint positions into self.actions [num_envs, total_dofs]
        4. Then run physics simulation
        """
        if self.cfg.control.control_type == "C":
            action = torch.clip(action, -1, 1)

            # Actions come in as [num_envs * num_agents, action_dim]
            # Need to extract each agent's actions across all environments

            # Process actions for each agent type separately to get joint positions
            joint_positions_list = []

            for agent_idx in range(self.num_agents):
                # Extract this agent's high-level actions from all environments
                # action shape: [num_envs * num_agents, 3]
                # We need rows corresponding to this agent: [env0-agentN, env1-agentN, ...]
                agent_env_indices = torch.arange(self.num_envs, device=self.device) * self.num_agents + agent_idx
                agent_actions = action[agent_env_indices, :]  # [num_envs, 3]

                # Get robot info
                robot_type = self.hetero_agent_types[agent_idx]
                robot_info = self.hetero_robot_info[agent_idx]
                num_dof_agent = self.robot_num_dofs[agent_idx]

                # Call locomotion policy to convert high-level → joint positions
                locomotion_policy = self.locomotion_policies[agent_idx]
                obs_buffer_dict = self.locomotion_obs_buffers[agent_idx]

                # Get per-agent observations (extract from shared buffers)
                agent_env_indices = torch.arange(self.num_envs, device=self.device) * self.num_agents + agent_idx

                if robot_type == 'go1':
                    # Build 70-dim locomotion observation for Go1 walk_these_ways policy
                    # Structure must match go1.py _fill_command_obs() and preprocess_action()
                    loc_obs = obs_buffer_dict['obs']
                    history = obs_buffer_dict['history']

                    # [0:3] projected_gravity
                    loc_obs[:, 0:3] = self.obs_buf.projected_gravity[agent_env_indices]

                    # [3:6] velocity commands scaled by [2.0, 2.0, 0.25]
                    loc_obs[:, 3:6] = agent_actions * self.commands_scale

                    # [6] body_height (default 0.0 * 2.0 = 0.0)
                    loc_obs[:, 6] = 0.0

                    # [7] gait_freq (default 3.0 * 1.0 = 3.0)
                    loc_obs[:, 7] = 3.0

                    # [8:12] gait params for trotting: [phase=0.5, offset=0, bound=0, duration=0.5]
                    loc_obs[:, 8] = 0.5   # phase
                    loc_obs[:, 9] = 0.0   # offset
                    loc_obs[:, 10] = 0.0  # bound
                    loc_obs[:, 11] = 0.5  # duration

                    # [12] footswing_height (default 0.08 * 0.15 = 0.012)
                    loc_obs[:, 12] = 0.08 * 0.15

                    # [13:15] body_pitch, body_roll (default 0.0)
                    loc_obs[:, 13] = 0.0
                    loc_obs[:, 14] = 0.0

                    # [15] stance_width (default 0.25 * 1.0 = 0.25)
                    loc_obs[:, 15] = 0.25

                    # [16] stance_length (default 0.428 * 1.0 = 0.428)
                    loc_obs[:, 16] = 0.428

                    # [17] aux_reward (default 0.0)
                    loc_obs[:, 17] = 0.0

                    # [18:30] dof_pos (scaled errors from default)
                    loc_obs[:, 18:30] = self.obs_buf.dof_pos[agent_env_indices]

                    # [30:42] dof_vel (scaled)
                    loc_obs[:, 30:42] = self.obs_buf.dof_vel[agent_env_indices]

                    # [42:54] last_locomotion_action (t-1)
                    if 'last_action' in obs_buffer_dict:
                        loc_obs[:, 42:54] = obs_buffer_dict['last_action']
                    else:
                        loc_obs[:, 42:54] = 0.0

                    # [54:66] last_two_locomotion_action (t-2)
                    if 'last_two_action' in obs_buffer_dict:
                        loc_obs[:, 54:66] = obs_buffer_dict['last_two_action']
                    else:
                        loc_obs[:, 54:66] = 0.0

                    # [66:70] clock_inputs - compute from gait phase
                    # Must match go1.py _step_contact_targets() exactly
                    if hasattr(self, 'go1_gait_indices'):
                        # Get gait params from locomotion_obs (same as homogeneous)
                        frequencies = loc_obs[:, 7]   # gait_freq
                        phases = loc_obs[:, 8]        # phase
                        offsets = loc_obs[:, 9]       # offset
                        bounds = loc_obs[:, 10]       # bound
                        durations = loc_obs[:, 11]    # duration

                        # Update gait index
                        self.go1_gait_indices = torch.remainder(self.go1_gait_indices + self.dt * frequencies, 1.0)

                        # Compute foot indices exactly like go1.py _step_contact_targets
                        foot_indices = [
                            self.go1_gait_indices + phases + offsets + bounds,  # FL
                            self.go1_gait_indices + offsets,                     # FR
                            self.go1_gait_indices + bounds,                      # RL
                            self.go1_gait_indices + phases                       # RR
                        ]

                        # Apply duration warping like go1.py
                        for idxs in foot_indices:
                            stance_idxs = torch.remainder(idxs, 1) < durations
                            swing_idxs = torch.remainder(idxs, 1) >= durations
                            idxs[stance_idxs] = torch.remainder(idxs[stance_idxs], 1) * (0.5 / durations[stance_idxs])
                            idxs[swing_idxs] = 0.5 + (torch.remainder(idxs[swing_idxs], 1) - durations[swing_idxs]) * (
                                0.5 / (1 - durations[swing_idxs]))

                        # Compute clock inputs (sinusoidal)
                        self.go1_clock_inputs[:, 0] = torch.sin(2 * np.pi * foot_indices[0])
                        self.go1_clock_inputs[:, 1] = torch.sin(2 * np.pi * foot_indices[1])
                        self.go1_clock_inputs[:, 2] = torch.sin(2 * np.pi * foot_indices[2])
                        self.go1_clock_inputs[:, 3] = torch.sin(2 * np.pi * foot_indices[3])

                        loc_obs[:, 66:70] = self.go1_clock_inputs
                    else:
                        loc_obs[:, 66:70] = 0.0

                    # Update history (rolling window of 30 observations = 2100 dims)
                    history = torch.cat((history[:, 70:], loc_obs), dim=-1)
                    obs_buffer_dict['history'] = history

                    # Call policy with history
                    joint_positions = locomotion_policy(history)

                    # Store last actions for next step (need t-1 and t-2)
                    if 'last_action' in obs_buffer_dict:
                        obs_buffer_dict['last_two_action'] = obs_buffer_dict['last_action'].clone()
                    obs_buffer_dict['last_action'] = joint_positions.clone()

                    # DEBUG: Print on first step only
                    if not hasattr(self, '_debug_printed_go1'):
                        print(f"\n[DEBUG Go1] First step:")
                        print(f"  Commands (scaled): {loc_obs[0, 3:6]}")
                        print(f"  Gait freq: {loc_obs[0, 7]}, Gait params: {loc_obs[0, 8:12]}")
                        print(f"  DOF pos (scaled errors): {loc_obs[0, 18:30]}")
                        print(f"  Policy output joint targets: {joint_positions[0]}")
                        print(f"  Default joint angles: {self.default_dof_pos[0, :12]}")
                        self._debug_printed_go1 = True

                elif robot_type == 'anymal_c':
                    # Build 48-dim locomotion observation for Anymal C
                    # Use SAME method as Go1 - use obs_buf which is already processed!
                    loc_obs = obs_buffer_dict['obs']

                    # Use obs_buf like Go1 does - it handles scaling and per-agent indexing
                    # Observation structure for legged_gym:
                    # [0:3] base_lin_vel (already scaled in obs_buf)
                    # [3:6] base_ang_vel (already scaled in obs_buf)
                    # [6:9] projected_gravity
                    # [9:12] commands
                    # [12:24] dof_pos (already scaled in obs_buf)
                    # [24:36] dof_vel (already scaled in obs_buf)
                    # [36:48] previous actions

                    # Get values from obs_buf (already per-agent and scaled)
                    loc_obs[:, 0:3] = self.obs_buf.lin_vel[agent_env_indices]  # Already scaled
                    loc_obs[:, 3:6] = self.obs_buf.ang_vel[agent_env_indices]  # Already scaled
                    loc_obs[:, 6:9] = self.obs_buf.projected_gravity[agent_env_indices]
                    # Scale commands like legged_gym does: [lin_vel, lin_vel, ang_vel] * [2.0, 2.0, 0.25]
                    loc_obs[:, 9:12] = agent_actions * self.commands_scale  # Commands scaled
                    loc_obs[:, 12:24] = self.obs_buf.dof_pos[agent_env_indices]  # Already scaled
                    loc_obs[:, 24:36] = self.obs_buf.dof_vel[agent_env_indices]  # Already scaled

                    # [36:48] previous actions (raw policy outputs from last step)
                    if 'last_joint_targets' in obs_buffer_dict:
                        loc_obs[:, 36:48] = obs_buffer_dict['last_joint_targets']
                    else:
                        # First step - use zeros
                        loc_obs[:, 36:48] = torch.zeros(self.num_envs, 12, device=self.device)

                    # Call policy
                    joint_positions = locomotion_policy(loc_obs)

                    # DEBUG: Print on first step only
                    if not hasattr(self, '_debug_printed_anymal'):
                        print(f"\n[DEBUG Anymal C] First step:")
                        print(f"  Lin vel (scaled): {loc_obs[0, 0:3]}")
                        print(f"  Ang vel (scaled): {loc_obs[0, 3:6]}")
                        print(f"  Projected gravity: {loc_obs[0, 6:9]}")
                        print(f"  Commands (scaled): {loc_obs[0, 9:12]}")
                        print(f"  DOF pos (scaled errors): {loc_obs[0, 12:24]}")
                        print(f"  DOF vel (scaled): {loc_obs[0, 24:36]}")
                        print(f"  Last actions: {loc_obs[0, 36:48]}")
                        print(f"  Policy output joint targets: {joint_positions[0]}")
                        print(f"  Default joint angles: {self.default_dof_pos[0, 12:24]}")
                        self._debug_printed_anymal = True

                    # Store for next step
                    obs_buffer_dict['last_joint_targets'] = joint_positions.clone()

                else:
                    raise NotImplementedError(f"Locomotion policy for {robot_type} not implemented")

                # Append joint positions for this agent
                joint_positions_list.append(joint_positions)

            # Concatenate all agents' joint positions
            all_joint_positions = torch.cat(joint_positions_list, dim=1)  # [num_envs, total_dofs]

            clip_actions = self.cfg.normalization.clip_actions
            self.actions = torch.clip(all_joint_positions, -clip_actions, clip_actions).reshape(self.num_envs, -1).to(self.device)
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

    def close(self):
        """
        Close the environment and cleanup resources.
        Required for gym wrapper compatibility.
        """
        # Isaac Gym environments don't need explicit cleanup
        # but gym wrapper expects this method
        pass
