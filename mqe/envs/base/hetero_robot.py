"""
Heterogeneous Robot Base Class

This class extends LeggedRobot to support heterogeneous agents
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
from mqe.envs.base.legged_robot import LeggedRobot
from mqe.envs.robot_registry import get_robot_class, get_robot_info
from mqe.utils.hetero_config import (
    get_hetero_asset_paths,
    get_hetero_action_dims,
    get_max_action_dim,
    validate_hetero_agents
)


class HeteroRobot(LeggedRobot):
    """
    Heterogeneous robot environment that supports multiple robot types.

    This class extends LeggedRobot to handle:
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

            print(f"[HeteroRobot] Loaded {self.hetero_agent_types[i]}: {num_dof} DOF, {num_bodies} bodies")

        # Store for later use
        self.robot_assets = robot_assets
        self.robot_num_dofs = robot_num_dofs
        self.robot_num_bodies = robot_num_bodies
        self.robot_dof_props_list = robot_dof_props_list
        self.robot_rigid_shape_props_list = robot_rigid_shape_props_list

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
