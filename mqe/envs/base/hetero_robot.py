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

            # Create asset options (use config from base or per-robot if available)
            asset_options = gymapi.AssetOptions()
            asset_options.default_dof_drive_mode = self.cfg.asset.default_dof_drive_mode
            asset_options.collapse_fixed_joints = self.cfg.asset.collapse_fixed_joints
            asset_options.replace_cylinder_with_capsule = self.cfg.asset.replace_cylinder_with_capsule
            asset_options.flip_visual_attachments = self.cfg.asset.flip_visual_attachments
            asset_options.fix_base_link = self.cfg.asset.fix_base_link
            asset_options.density = self.cfg.asset.density
            asset_options.angular_damping = self.cfg.asset.angular_damping
            asset_options.linear_damping = self.cfg.asset.linear_damping
            asset_options.max_angular_velocity = self.cfg.asset.max_angular_velocity
            asset_options.max_linear_velocity = self.cfg.asset.max_linear_velocity
            asset_options.armature = self.cfg.asset.armature
            asset_options.thickness = self.cfg.asset.thickness
            asset_options.disable_gravity = self.cfg.asset.disable_gravity

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

            # Get torque limits for this robot type
            from mqe.envs.robot_registry import get_robot_config
            robot_config = get_robot_config(self.hetero_agent_types[i])
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
                print(f"[HeteroRobot] Agent {idx} ({self.hetero_agent_types[idx]}): init height set to {robot_init_height}m")

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
        # Call parent's parent (LeggedRobot._process_dof_props) to skip LeggedRobotField's assertion
        from mqe.envs.base.legged_robot import LeggedRobot
        props = LeggedRobot._process_dof_props(self, props, env_id)

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
            # actions shape: [num_envs, num_agents * num_action]
            # We need to process each agent separately

            # Scale actions
            if isinstance(self.cfg.control.action_scale, (tuple, list)):
                self.cfg.control.action_scale = torch.tensor(
                    self.cfg.control.action_scale, device=self.sim_device
                )
            actions_scaled = actions * self.cfg.control.action_scale

            # Compute DOF offsets for indexing
            dof_offsets = [0]
            for num_dof in self.robot_num_dofs:
                dof_offsets.append(dof_offsets[-1] + num_dof)

            # Process each agent's actions to get joint targets
            all_joint_targets = []

            for agent_idx in range(self.num_agents):
                # Get this agent's robot info
                robot_info = self.hetero_robot_info[agent_idx]
                num_dof_agent = self.robot_num_dofs[agent_idx]
                dof_start = dof_offsets[agent_idx]
                dof_end = dof_offsets[agent_idx + 1]

                # Extract this agent's actions for all environments
                # actions_scaled shape: [num_envs, num_agents * action_dim]
                action_start = agent_idx * self.num_action
                action_end = (agent_idx + 1) * self.num_action
                agent_actions = actions_scaled[:, action_start:action_end]  # [num_envs, action_dim]

                # Get default positions for this agent
                default_pos_agent = self.default_dof_pos[0, dof_start:dof_end]  # [num_dof_agent]

                # For Go1 (hierarchical control), actions are processed by locomotion policy
                if robot_info['default_control'] == 'C':
                    # Go1: actions are high-level commands [vx, vy, vyaw]
                    # The locomotion policy converts these to 12 joint positions
                    # Reshape to per-agent for locomotion policy
                    agent_actions_reshaped = agent_actions.reshape(-1, num_dof_agent)

                    # Apply hip scale reduction for Go1
                    if hasattr(self.cfg.control, 'hip_scale_reduction'):
                        agent_actions_reshaped[:, [0, 3, 6, 9]] *= self.cfg.control.hip_scale_reduction

                    # Flatten back
                    agent_actions_processed = agent_actions_reshaped.reshape(self.num_envs, num_dof_agent)

                    # Add to default positions to get joint targets
                    joint_targets = agent_actions_processed + default_pos_agent
                    all_joint_targets.append(joint_targets)

                else:
                    # Jackal (or other direct control): actions are wheel velocities
                    # For position control, we use actions directly as position targets
                    # Expand actions to match DOF count if needed
                    if agent_actions.shape[1] != num_dof_agent:
                        # Actions are high-level [vx, vy, vyaw], but we have 2 wheel DOFs
                        # We need to convert high-level to low-level
                        # For now, just use zero targets (differential drive handled elsewhere)
                        joint_targets = default_pos_agent.unsqueeze(0).repeat(self.num_envs, 1)
                    else:
                        joint_targets = agent_actions + default_pos_agent

                    all_joint_targets.append(joint_targets)

            # Concatenate all joint targets
            self.joint_pos_target = torch.cat(all_joint_targets, dim=1)  # [num_envs, total_dofs]

            # Compute position errors and velocities for actuator network
            # For Go1 agent (first agent), use actuator network
            go1_dof_start = 0
            go1_dof_end = self.robot_num_dofs[0]  # Should be 12 for Go1

            # Extract Go1's DOFs
            joint_pos_err_go1 = (
                self.dof_pos[:, go1_dof_start:go1_dof_end]
                - self.joint_pos_target[:, go1_dof_start:go1_dof_end]
            )
            joint_vel_go1 = self.dof_vel[:, go1_dof_start:go1_dof_end]

            # Apply actuator network to Go1
            if hasattr(self, 'actuator_network'):
                torques_go1 = self.actuator_network(
                    joint_pos_err_go1,
                    self.joint_pos_err_last if hasattr(self, 'joint_pos_err_last') else joint_pos_err_go1,
                    self.joint_pos_err_last_last if hasattr(self, 'joint_pos_err_last_last') else joint_pos_err_go1,
                    joint_vel_go1,
                    self.joint_vel_last if hasattr(self, 'joint_vel_last') else joint_vel_go1,
                    self.joint_vel_last_last if hasattr(self, 'joint_vel_last_last') else joint_vel_go1
                )

                # Update history for Go1
                self.joint_pos_err_last_last = torch.clone(self.joint_pos_err_last) if hasattr(self, 'joint_pos_err_last') else joint_pos_err_go1.clone()
                self.joint_pos_err_last = torch.clone(joint_pos_err_go1)
                self.joint_vel_last_last = torch.clone(self.joint_vel_last) if hasattr(self, 'joint_vel_last') else joint_vel_go1.clone()
                self.joint_vel_last = torch.clone(joint_vel_go1)
            else:
                # Fallback to PD control
                torques_go1 = self.p_gains[go1_dof_start:go1_dof_end] * joint_pos_err_go1 - self.d_gains[go1_dof_start:go1_dof_end] * joint_vel_go1

            # For Jackal (second agent), use simple PD control
            jackal_dof_start = self.robot_num_dofs[0]
            jackal_dof_end = jackal_dof_start + self.robot_num_dofs[1]

            joint_pos_err_jackal = (
                self.dof_pos[:, jackal_dof_start:jackal_dof_end]
                - self.joint_pos_target[:, jackal_dof_start:jackal_dof_end]
            )
            joint_vel_jackal = self.dof_vel[:, jackal_dof_start:jackal_dof_end]

            torques_jackal = (
                self.p_gains[jackal_dof_start:jackal_dof_end] * joint_pos_err_jackal
                - self.d_gains[jackal_dof_start:jackal_dof_end] * joint_vel_jackal
            )

            # Concatenate torques
            torques = torch.cat([torques_go1, torques_jackal], dim=1)

            # Clip to torque limits
            return torch.clip(torques, -self.torque_limits, self.torque_limits)

        else:
            # For non-hierarchical control, use parent implementation
            return super()._compute_torques(actions)
