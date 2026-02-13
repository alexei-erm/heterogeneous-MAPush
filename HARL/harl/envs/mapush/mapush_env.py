"""MAPush environment wrapper for HARL."""
import torch
import numpy as np
from typing import Tuple, List, Optional
from collections import deque
import sys
# INSERT at beginning to override PYTHONPATH pollution
sys.path.insert(0, '/home/gvlab/new-universal-MAPush')

from mqe.envs.utils import make_mqe_env
from task.cuboid.config import Go1PushMidCfg


class MAPushEnv:
    """MAPush environment for HARL HAPPO training.

    This wrapper adapts the MAPush environment to HARL's interface requirements.
    """

    def __init__(self, env_args):
        """Initialize MAPush environment.

        Args:
            env_args: dict with keys:
                - task: str (e.g., "go1push_mid")
                - n_threads: int (number of parallel environments)
        """
        self.env_args = env_args

        # Import Isaac Gym before other imports
        import isaacgym  # Must import before torch
        from isaacgym import gymapi
        from mqe.envs.utils import custom_cfg

        # Create MAPush environment - use argparse.Namespace like old code
        import argparse
        args = argparse.Namespace()

        # Get num_envs from either 'num_envs' or 'n_threads'
        num_envs = env_args.get("num_envs", env_args.get("n_threads", 500))

        args.task = env_args.get("task", "go1push_mid")
        args.num_envs = num_envs
        args.seed = env_args.get("seed", 1)
        args.headless = env_args.get("headless", True)
        args.record_video = False  # Disable video recording during training

        # Device configuration
        device = env_args.get("device", "cuda:0")
        args.rl_device = device
        args.sim_device = "cuda:0"
        args.device = "cuda"  # For parse_sim_params
        args.compute_device_id = 0
        args.sim_device_type = "cuda"
        args.use_gpu_pipeline = True
        args.physics_engine = gymapi.SIM_PHYSX  # Use PhysX
        args.use_gpu = True
        args.subscenes = 0  # Number of PhysX subscenes
        args.num_threads = 0  # Number of cores used by PhysX

        # Check for heterogeneous agent mode using agent0/agent1 flags
        agent0 = env_args.get("agent0", "go1")
        agent1 = env_args.get("agent1", "go1")
        self.is_hetero = (agent0 != agent1)
        self.agent_types = [agent0, agent1]

        # Create MQE environment with custom config
        individualized_rewards = env_args.get("individualized_rewards", False)
        shared_gated_rewards = env_args.get("shared_gated_rewards", False)
        cooperation_rewards = env_args.get("cooperation_rewards", False)
        mapush_og_rewards_teamified = env_args.get("mapush_og_rewards_teamified", False)
        reward_scale_testing = env_args.get("reward_scale_testing", False)
        collaboration_rewards = env_args.get("collaboration_rewards", False)
        positive_approachtobox_reward = env_args.get("positive_approachtobox_reward", False)
        require_both_contact_for_success = env_args.get("require_both_contact_for_success", False)

        # Velocity task parameters
        vel_speed_min = env_args.get("vel_speed_min", None)
        vel_speed_max = env_args.get("vel_speed_max", None)
        vel_tracking_scale = env_args.get("vel_tracking_scale", None)
        vel_angular_penalty_scale = env_args.get("vel_angular_penalty_scale", None)

        if self.is_hetero:
            # Use make_hetero_env for heterogeneous agents
            from mqe.envs.utils import make_hetero_env
            print(f"[HARL MAPushEnv] Creating heterogeneous environment: {self.agent_types}")

            self.env, self.env_cfg = make_hetero_env(
                args.task,
                self.agent_types,
                args,
                custom_cfg=custom_cfg(args, individualized_rewards=individualized_rewards,
                                      shared_gated_rewards=shared_gated_rewards,
                                      cooperation_rewards=cooperation_rewards,
                                      mapush_og_rewards_teamified=mapush_og_rewards_teamified,
                                      reward_scale_testing=reward_scale_testing,
                                      collaboration_rewards=collaboration_rewards,
                                      positive_approachtobox_reward=positive_approachtobox_reward,
                                      agent0=agent0, agent1=agent1,
                                      require_both_contact_for_success=require_both_contact_for_success,
                                      vel_speed_min=vel_speed_min, vel_speed_max=vel_speed_max,
                                      vel_tracking_scale=vel_tracking_scale,
                                      vel_angular_penalty_scale=vel_angular_penalty_scale)
            )
        else:
            # Standard homogeneous environment
            self.env, self.env_cfg = make_mqe_env(
                args.task,
                args,
                custom_cfg=custom_cfg(args, individualized_rewards=individualized_rewards,
                                      shared_gated_rewards=shared_gated_rewards,
                                      cooperation_rewards=cooperation_rewards,
                                      mapush_og_rewards_teamified=mapush_og_rewards_teamified,
                                      reward_scale_testing=reward_scale_testing,
                                      collaboration_rewards=collaboration_rewards,
                                      positive_approachtobox_reward=positive_approachtobox_reward,
                                      agent0=agent0, agent1=agent1,
                                      require_both_contact_for_success=require_both_contact_for_success,
                                      vel_speed_min=vel_speed_min, vel_speed_max=vel_speed_max,
                                      vel_tracking_scale=vel_tracking_scale,
                                      vel_angular_penalty_scale=vel_angular_penalty_scale)
            )

        self.n_envs = self.env.num_envs
        self.n_agents = self.env.num_agents

        # Detect velocity task
        self.is_velocity_task = (args.task == "go1push_vel")

        # HARL expects list of spaces (one per agent)
        # In hetero mode, spaces are already lists; in homo mode, we need to duplicate
        if self.is_hetero and isinstance(self.env.observation_space, list):
            # Heterogeneous: spaces already different per agent
            self.observation_space = self.env.observation_space
            self.action_space = self.env.action_space
        else:
            # Homogeneous: same space for all agents
            self.observation_space = [self.env.observation_space] * self.n_agents
            self.action_space = [self.env.action_space] * self.n_agents

        if self.is_hetero:
            print(f"[MAPushEnv] Heterogeneous agents with unified action space:")
            print(f"  Agent 0 ({agent0}): 3 DOF [vx, vy, vyaw]")
            print(f"  Agent 1 ({agent1}): 3 DOF [vx, vy, vyaw]")

        # Flags to control critic input coordinate system
        # Priority: relative_obs > concat_observations > goal_centered > box_centered > absolute
        # use_relative_obs_critic: CRITIC11 (9 dims) - Relative observations with inter-robot distance
        # use_concat_agent_observations_critic: CRITIC10 (16 dims) - Concatenated agent local observations
        # use_goal_centered_critic: CRITIC16 (9 dims) - Goal-centered coordinates (stationary reference frame)
        # use_box_centered_critic: CRITIC9 (9 dims) - Box-centered coordinates (translation invariant)
        # None: CRITIC7 (11 dims) - Absolute world frame coordinates
        # DEFAULT: All False (absolute coordinates)
        self.use_relative_obs_critic = env_args.get("use_relative_obs_critic", False)
        self.use_concat_agent_observations_critic = env_args.get("use_concat_agent_observations_critic", False)
        self.use_goal_centered_critic = env_args.get("use_goal_centered_critic", False)
        self.use_box_centered_critic = env_args.get("use_box_centered_critic", False)

        # Share observation space (for centralized critic)
        from gym import spaces
        if self.is_velocity_task:
            # Velocity task global state: [cmd(3), box_vel(3), agent0_rel(2), agent1_rel(2), inter_agent_dist(1)]
            global_state_dim = 3 + 3 + 2 * self.n_agents + 1  # = 11 for 2 agents
        elif self.use_relative_obs_critic:
            # CRITIC11: Relative observations with explicit inter-robot distance
            # Structure: [robot1_to_box(3), robot2_to_box(3), inter_robot_dist(1), goal_to_box(2)]
            #
            # robot_to_box: (dx, dy, dψ) - agent position and yaw relative to box
            # inter_robot_dist: scalar - Euclidean distance between agents
            # goal_to_box: (dx, dy) - target position relative to box
            #
            # Total: 3 + 3 + 1 + 2 = 9 dims
            global_state_dim = 3 + 3 + 1 + 2  # robot1(3) + robot2(3) + dist(1) + goal(2)
        elif self.use_concat_agent_observations_critic:
            # CRITIC10: Concatenated agent local observations
            # Simply concatenate all agents' local observations without modification
            # Each agent observation is in its own frame of reference (already rotated to local frame)
            #
            # Each agent observation (8 dims for 2 agents):
            #   [target_x, target_y,                  = 2 dims (target relative to agent)
            #    box_x, box_y,                        = 2 dims (box relative to agent)
            #    box_yaw,                             = 1 dim  (box yaw relative to agent)
            #    other_agent_x, other_agent_y, other_agent_yaw] = 3 dims (other agent relative to this agent)
            #
            # Global state = [agent0_obs, agent1_obs] = 8 * n_agents = 16 dims (for 2 agents)
            obs_dim = self.env.observation_space.shape[0]  # 8 dims per agent
            global_state_dim = obs_dim * self.n_agents     # 8 * 2 = 16 dims
        elif self.use_goal_centered_critic:
            # CRITIC16: Goal-centered global state
            # Express everything relative to the goal (the stationary target)
            # This provides:
            # 1. Translation invariance (goal at different world positions = same state)
            # 2. True global view (not just concatenated local perspectives)
            # 3. Task-centric representation (value depends on box-goal distance and agent-goal positions)
            # 4. Stationary reference frame (goal never moves, unlike box which moves during episode)
            #
            # Global state structure (9 dims for 2 agents):
            #   [box_rel_x, box_rel_y,                = 2 dims (box relative to goal)
            #    box_rel_yaw,                         = 1 dim  (box yaw relative to goal)
            #    agent0_rel_x, agent0_rel_y,          = 2 dims (agent0 relative to goal)
            #    agent0_rel_yaw,                      = 1 dim  (agent0 yaw relative to goal)
            #    agent1_rel_x, agent1_rel_y,          = 2 dims (agent1 relative to goal)
            #    agent1_rel_yaw]                      = 1 dim  (agent1 yaw relative to goal)
            # Total: 3 + 3*n_agents = 9 dims (for 2 agents)
            global_state_dim = 3 + 3 * self.n_agents  # box(3) + agents(3 each)
        elif self.use_box_centered_critic:
            # CRITIC9: Box-centered global state
            # Express everything relative to the box (the object being pushed)
            # This provides:
            # 1. Translation invariance (box at different positions = same state)
            # 2. True global view (not just concatenated local perspectives)
            # 3. Task-centric representation (value depends on box-target distance and agent-box positions)
            #
            # Global state structure (9 dims for 2 agents):
            #   [target_rel_x, target_rel_y,           = 2 dims (target relative to box)
            #    agent0_rel_x, agent0_rel_y,           = 2 dims (agent0 relative to box)
            #    agent0_rel_yaw,                       = 1 dim  (agent0 yaw relative to box)
            #    agent1_rel_x, agent1_rel_y,           = 2 dims (agent1 relative to box)
            #    agent1_rel_yaw,                       = 1 dim  (agent1 yaw relative to box)
            #    box_yaw]                              = 1 dim  (box absolute orientation)
            # Total: 2 + 3*n_agents + 1 = 9 dims (for 2 agents)
            global_state_dim = 2 + 3 * self.n_agents + 1  # target(2) + agents(3 each) + box_yaw(1)
        else:
            # CRITIC7: Absolute (world frame) global state
            # All positions in global coordinates
            # Global state structure (11 dims for 2 agents):
            #   [box_x, box_y, box_yaw,                = 3 dims (box absolute position)
            #    target_x, target_y,                   = 2 dims (target absolute position)
            #    agent0_x, agent0_y, agent0_yaw,       = 3 dims (agent0 absolute position)
            #    agent1_x, agent1_y, agent1_yaw]       = 3 dims (agent1 absolute position)
            # Total: 3 + 2 + 3*n_agents = 11 dims (for 2 agents)
            global_state_dim = 3 + 2 + 3 * self.n_agents  # box(3) + target(2) + agents(3 each)

        self.share_observation_space = [
            spaces.Box(low=-float('inf'), high=float('inf'),
                      shape=(global_state_dim,), dtype=np.float32)
        ] * self.n_agents

        # Statistics tracking (for calculator mode)
        self.reset_statistics()

    def _construct_global_state(self) -> np.ndarray:
        """Construct global state from environment internals.

        Depending on self.use_box_centered_critic flag:

        If True (CRITIC9 - Box-centered):
            Global state structure (9 dims for 2 agents):
            - Target relative to box: [target_x - box_x, target_y - box_y]           = 2 dims
            - Agent 0 relative to box: [agent0_x - box_x, agent0_y - box_y, agent0_yaw - box_yaw] = 3 dims
            - Agent 1 relative to box: [agent1_x - box_x, agent1_y - box_y, agent1_yaw - box_yaw] = 3 dims
            - Box orientation: [box_yaw]                                             = 1 dim
            Total: 2 + 3*n_agents + 1 = 9 dims

        If False (CRITIC7 - Absolute):
            Global state structure (11 dims for 2 agents):
            - Box: [x, y, yaw]                                                       = 3 dims
            - Target: [x, y]                                                         = 2 dims
            - Agent 0: [x, y, yaw]                                                   = 3 dims
            - Agent 1: [x, y, yaw]                                                   = 3 dims
            Total: 3 + 2 + 3*n_agents = 11 dims

        Returns:
            global_state: [n_envs, global_state_dim] numpy array
        """
        # Access underlying wrapper to get global state information
        wrapper = self.env

        # Get NPC states (box and target) from root_states_npc
        # root_states_npc shape: [num_envs * num_npcs, 13] (pos, quat, lin_vel, ang_vel)
        # root_states_npc is in WORLD FRAME (includes env_origins offset)
        npc_states = wrapper.root_states_npc.reshape(self.n_envs, wrapper.num_npcs, -1)

        # Box state (NPC 0)
        # SUBTRACT env_origins to convert to environment-relative frame
        # This matches the coordinate frame used by obs_buf.base_pos (which also subtracts env_origins)
        box_pos_global = npc_states[:, 0, :3] - wrapper.env.env_origins  # [n_envs, 3]
        box_quat = npc_states[:, 0, 3:7]  # [n_envs, 4]

        # Target state (NPC 1)
        # SUBTRACT env_origins to convert to environment-relative frame
        target_pos_global = npc_states[:, 1, :3] - wrapper.env.env_origins  # [n_envs, 3]
        target_quat = npc_states[:, 1, 3:7]  # [n_envs, 4]

        # Convert quaternions to yaw using Isaac Gym utils
        from isaacgym.torch_utils import get_euler_xyz
        box_rpy = torch.stack(get_euler_xyz(box_quat), dim=1)  # [n_envs, 3]
        target_rpy = torch.stack(get_euler_xyz(target_quat), dim=1)  # [n_envs, 3]

        # Get agent states from obs_buf (includes position, velocity, rpy)
        # We need to access the raw observation buffer from the base environment
        obs_buf = wrapper.env.obs_buf if hasattr(wrapper, 'env') else wrapper.obs_buf

        # Agent position and orientation
        # NOTE: obs_buf.base_pos ALREADY has env_origins subtracted (see go1.py:161)
        # So it's in environment-relative frame, matching box_pos_global and target_pos_global above
        base_pos = obs_buf.base_pos.reshape(self.n_envs, self.n_agents, 3)  # [n_envs, n_agents, 3]
        base_rpy = obs_buf.base_rpy.reshape(self.n_envs, self.n_agents, 3)  # [n_envs, n_agents, 3]

        # Agent velocities (linear and angular)
        # These should be [n_envs * n_agents, 3] and we reshape to [n_envs, n_agents, 3]
        try:
            base_lin_vel = obs_buf.lin_vel.reshape(self.n_envs, self.n_agents, 3)  # [n_envs, n_agents, 3]
            base_ang_vel = obs_buf.ang_vel.reshape(self.n_envs, self.n_agents, 3)  # [n_envs, n_agents, 3]
        except Exception as e:
            print(f"\nERROR accessing velocities:")
            print(f"  obs_buf.lin_vel shape: {obs_buf.lin_vel.shape if hasattr(obs_buf, 'lin_vel') else 'DOES NOT EXIST'}")
            print(f"  obs_buf.ang_vel shape: {obs_buf.ang_vel.shape if hasattr(obs_buf, 'ang_vel') else 'DOES NOT EXIST'}")
            print(f"  Expected reshape: [{self.n_envs}, {self.n_agents}, 3]")
            print(f"  Error: {e}")
            raise

        # Construct global state based on coordinate system flag
        # Priority: relative_obs > concat_observations > goal_centered > box_centered > absolute
        # NOTE: This method is NOT called when use_concat_agent_observations_critic=True or use_relative_obs_critic=True
        # For CRITIC10 and CRITIC11, global state is constructed directly in step() and reset()
        if self.use_relative_obs_critic:
            # CRITIC11: Relative observations with inter-robot distance
            # This should not be reached as CRITIC11 constructs state directly in step()/reset()
            raise RuntimeError("CRITIC11 should construct state directly in step()/reset(), not call _construct_global_state()")
        elif self.use_goal_centered_critic:
            # CRITIC16: Goal-centered (relative) global state
            # Express everything relative to the goal (stationary reference frame)
            # Structure: [box_rel(3), agent0_rel(3), agent1_rel(3), ...]

            # Goal relative to... nothing! Goal is the origin (0, 0)
            # But we need goal position and orientation for transformation

            # Box relative to goal
            box_rel = box_pos_global[:, :2] - target_pos_global[:, :2]  # [n_envs, 2]
            box_yaw_rel = box_rpy[:, 2:3] - target_rpy[:, 2:3]  # [n_envs, 1]

            # Start with box relative position and yaw
            global_state_list = [box_rel, box_yaw_rel]

            # Add each agent's position and orientation relative to goal
            for agent_id in range(self.n_agents):
                # Agent position relative to goal
                agent_pos_rel = base_pos[:, agent_id, :2] - target_pos_global[:, :2]  # [n_envs, 2]
                # Agent yaw relative to goal yaw
                agent_yaw_rel = base_rpy[:, agent_id, 2:3] - target_rpy[:, 2:3]  # [n_envs, 1]

                global_state_list.append(agent_pos_rel)
                global_state_list.append(agent_yaw_rel)

            # Concatenate into single tensor
            global_state_torch = torch.cat(global_state_list, dim=1)  # [n_envs, 3 + 3*n_agents]
        elif self.use_box_centered_critic:
            # CRITIC9: Box-centered (relative) global state
            # Express everything relative to the box (translation invariant)
            # Structure: [target_rel(2), agent0_rel(3), agent1_rel(3), ..., box_yaw(1)]

            # Target relative to box
            target_rel = target_pos_global[:, :2] - box_pos_global[:, :2]  # [n_envs, 2]

            # Start with target relative position
            global_state_list = [target_rel]

            # Add each agent's position and orientation relative to box
            for agent_id in range(self.n_agents):
                # Agent position relative to box
                agent_pos_rel = base_pos[:, agent_id, :2] - box_pos_global[:, :2]  # [n_envs, 2]
                # Agent yaw relative to box yaw
                agent_yaw_rel = base_rpy[:, agent_id, 2:3] - box_rpy[:, 2:3]  # [n_envs, 1]

                global_state_list.append(agent_pos_rel)
                global_state_list.append(agent_yaw_rel)

            # Add box absolute orientation (for reference)
            global_state_list.append(box_rpy[:, 2:3])  # [n_envs, 1]

            # Concatenate into single tensor
            global_state_torch = torch.cat(global_state_list, dim=1)  # [n_envs, 2 + 3*n_agents + 1]
        else:
            # CRITIC7: Absolute (world frame) global state
            # All positions in global coordinates
            # Structure: [box(3), target(2), agent0(3), agent1(3), ...]

            global_state_list = [
                box_pos_global[:, :2],         # box x, y
                box_rpy[:, 2:3],               # box yaw
                target_pos_global[:, :2],      # target x, y
            ]

            # Add all agents' positions (NO velocities)
            for agent_id in range(self.n_agents):
                global_state_list.append(base_pos[:, agent_id, :2])      # agent x, y
                global_state_list.append(base_rpy[:, agent_id, 2:3])     # agent yaw

            # Concatenate into single tensor
            global_state_torch = torch.cat(global_state_list, dim=1)  # [n_envs, 3 + 2 + 3*n_agents]

        # Convert to numpy
        import numpy as np
        global_state_np = global_state_torch.cpu().numpy().astype(np.float32)

        # Diagnostic logging (first call only)
        if not hasattr(self, '_logged_global_state'):
            print("\n" + "="*80)
            if self.use_concat_agent_observations_critic:
                print("GLOBAL STATE DIAGNOSTIC (First Step) - CRITIC8: Concatenated Agent Observations")
                print("="*80)
                print(f"Global state shape: {global_state_np.shape}")
                obs_dim = self.env.observation_space.shape[0]
                print(f"Expected: [{self.n_envs}, {obs_dim * self.n_agents}] for {self.n_agents} agents (concatenated local obs)")
                print(f"\nEnvironment 0 global state ({obs_dim * self.n_agents} dims):")
                print(f"  Agent0 obs ({obs_dim} dims): {global_state_np[0, :obs_dim]}")
                print(f"  Agent1 obs ({obs_dim} dims): {global_state_np[0, obs_dim:obs_dim*2]}")
                print(f"\nAgent observation structure (each {obs_dim} dims):")
                print(f"  - Target (x, y) relative to agent: 2 dims")
                print(f"  - Box (x, y) relative to agent: 2 dims")
                print(f"  - Box yaw relative to agent: 1 dim")
                print(f"  - Other agent (x, y, yaw) relative to this agent: 3 dims")
            elif self.use_goal_centered_critic:
                print("GLOBAL STATE DIAGNOSTIC (First Step) - CRITIC16: Goal-centered")
                print("="*80)
                print(f"Global state shape: {global_state_np.shape}")
                print(f"Expected: [{self.n_envs}, 9] for 2 agents (goal-centered coordinates)")
                print(f"\nEnvironment 0 global state (9 dims):")
                print(f"  Box rel to goal:    x={global_state_np[0,0]:.3f}, y={global_state_np[0,1]:.3f}, yaw={global_state_np[0,2]:.3f}")
                print(f"  Agent0 rel to goal: x={global_state_np[0,3]:.3f}, y={global_state_np[0,4]:.3f}, yaw={global_state_np[0,5]:.3f}")
                print(f"  Agent1 rel to goal: x={global_state_np[0,6]:.3f}, y={global_state_np[0,7]:.3f}, yaw={global_state_np[0,8]:.3f}")
                print(f"\nNote: Goal is at origin (0, 0) in this frame. Box at (0, 0) = SUCCESS!")
            elif self.use_box_centered_critic:
                print("GLOBAL STATE DIAGNOSTIC (First Step) - CRITIC9: Box-centered")
                print("="*80)
                print(f"Global state shape: {global_state_np.shape}")
                print(f"Expected: [500, 9] for 2 agents (box-centered coordinates)")
                print(f"\nEnvironment 0 global state (9 dims):")
                print(f"  Target rel to box:  x={global_state_np[0,0]:.3f}, y={global_state_np[0,1]:.3f}")
                print(f"  Agent0 rel to box:  x={global_state_np[0,2]:.3f}, y={global_state_np[0,3]:.3f}, yaw={global_state_np[0,4]:.3f}")
                print(f"  Agent1 rel to box:  x={global_state_np[0,5]:.3f}, y={global_state_np[0,6]:.3f}, yaw={global_state_np[0,7]:.3f}")
                print(f"  Box yaw (abs):      {global_state_np[0,8]:.3f}")
            else:
                print("GLOBAL STATE DIAGNOSTIC (First Step) - CRITIC7: Absolute coordinates")
                print("="*80)
                print(f"Global state shape: {global_state_np.shape}")
                print(f"Expected: [500, 11] for 2 agents (absolute world frame)")
                print(f"\nEnvironment 0 global state (11 dims):")
                print(f"  Box:    x={global_state_np[0,0]:.3f}, y={global_state_np[0,1]:.3f}, yaw={global_state_np[0,2]:.3f}")
                print(f"  Target: x={global_state_np[0,3]:.3f}, y={global_state_np[0,4]:.3f}")
                print(f"  Agent0: x={global_state_np[0,5]:.3f}, y={global_state_np[0,6]:.3f}, yaw={global_state_np[0,7]:.3f}")
                print(f"  Agent1: x={global_state_np[0,8]:.3f}, y={global_state_np[0,9]:.3f}, yaw={global_state_np[0,10]:.3f}")
            print(f"\nStatistics across all {self.n_envs} environments:")
            print(f"  Min values:  {np.min(global_state_np, axis=0)}")
            print(f"  Max values:  {np.max(global_state_np, axis=0)}")
            print(f"  Mean values: {np.mean(global_state_np, axis=0)}")
            print(f"  Std values:  {np.std(global_state_np, axis=0)}")
            print(f"\nNaN count: {np.isnan(global_state_np).sum()}")
            print(f"Inf count: {np.isinf(global_state_np).sum()}")
            print("="*80 + "\n")
            self._logged_global_state = True

        # Handle NaN and Inf
        nan_count = np.isnan(global_state_np).sum()
        inf_count = np.isinf(global_state_np).sum()
        if nan_count > 0 or inf_count > 0:
            print(f"WARNING: Found {nan_count} NaN and {inf_count} Inf values in global state!")
        global_state_np[np.isnan(global_state_np)] = 0.0
        global_state_np[np.isinf(global_state_np)] = 0.0

        return global_state_np

    def _construct_relative_obs_state(self) -> np.ndarray:
        """Construct CRITIC11 relative observations with inter-robot distance.

        Structure (9 dims for 2 agents):
            [robot1_to_box(3), robot2_to_box(3), inter_robot_dist(1), goal_to_box(2)]

        - robot_to_box: (dx, dy, dψ) - agent position and yaw relative to box
        - inter_robot_dist: scalar - Euclidean distance between agents
        - goal_to_box: (dx, dy) - target position relative to box

        Returns:
            global_state_np: [n_envs, 9] numpy array
        """
        # Access underlying wrapper to get global state information
        wrapper = self.env

        # Get NPC states (box and target) from root_states_npc
        # root_states_npc shape: [num_envs * num_npcs, 13] (pos, quat, lin_vel, ang_vel)
        # root_states_npc is in WORLD FRAME (includes env_origins offset)
        npc_states = wrapper.root_states_npc.reshape(self.n_envs, wrapper.num_npcs, -1)

        # Box state (NPC 0)
        # SUBTRACT env_origins to convert to environment-relative frame
        box_pos_global = npc_states[:, 0, :3] - wrapper.env.env_origins  # [n_envs, 3]
        box_quat = npc_states[:, 0, 3:7]  # [n_envs, 4]

        # Target state (NPC 1)
        # SUBTRACT env_origins to convert to environment-relative frame
        target_pos_global = npc_states[:, 1, :3] - wrapper.env.env_origins  # [n_envs, 3]

        # Convert quaternions to yaw
        from isaacgym.torch_utils import get_euler_xyz
        box_rpy = torch.stack(get_euler_xyz(box_quat), dim=1)  # [n_envs, 3]

        # Get agent states from obs_buf
        obs_buf = wrapper.env.obs_buf if hasattr(wrapper, 'env') else wrapper.obs_buf

        # Agent position and orientation
        # NOTE: obs_buf.base_pos ALREADY has env_origins subtracted
        base_pos = obs_buf.base_pos.reshape(self.n_envs, self.n_agents, 3)  # [n_envs, n_agents, 3]
        base_rpy = obs_buf.base_rpy.reshape(self.n_envs, self.n_agents, 3)  # [n_envs, n_agents, 3]

        # Convert to numpy
        import numpy as np
        box_pos_np = box_pos_global.cpu().numpy()  # [n_envs, 3]
        box_yaw_np = box_rpy[:, 2].cpu().numpy()  # [n_envs]
        target_pos_np = target_pos_global.cpu().numpy()  # [n_envs, 3]
        agent_pos_np = base_pos.cpu().numpy()  # [n_envs, n_agents, 3]
        agent_yaw_np = base_rpy[:, :, 2].cpu().numpy()  # [n_envs, n_agents]

        # Construct state components
        global_state_list = []

        # For each agent: position and yaw relative to box
        for agent_id in range(self.n_agents):
            # Agent position relative to box (dx, dy)
            agent_to_box_pos = agent_pos_np[:, agent_id, :2] - box_pos_np[:, :2]  # [n_envs, 2]

            # Agent yaw relative to box yaw (dψ)
            agent_to_box_yaw = agent_yaw_np[:, agent_id] - box_yaw_np  # [n_envs]
            agent_to_box_yaw = agent_to_box_yaw[:, np.newaxis]  # [n_envs, 1]

            # Concatenate: [dx, dy, dψ]
            robot_to_box = np.concatenate([agent_to_box_pos, agent_to_box_yaw], axis=1)  # [n_envs, 3]
            global_state_list.append(robot_to_box)

        # Inter-robot distance: ||robot1_pos - robot2_pos||
        inter_robot_diff = agent_pos_np[:, 0, :2] - agent_pos_np[:, 1, :2]  # [n_envs, 2]
        inter_robot_dist = np.linalg.norm(inter_robot_diff, axis=1, keepdims=True)  # [n_envs, 1]
        global_state_list.append(inter_robot_dist)

        # Goal/target relative to box (dx, dy)
        goal_to_box = target_pos_np[:, :2] - box_pos_np[:, :2]  # [n_envs, 2]
        global_state_list.append(goal_to_box)

        # Concatenate all components: [robot1(3), robot2(3), dist(1), goal(2)] = 9 dims
        global_state_np = np.concatenate(global_state_list, axis=1).astype(np.float32)

        # Diagnostic logging (first call only)
        if not hasattr(self, '_logged_global_state'):
            print("\n" + "="*80)
            print("GLOBAL STATE DIAGNOSTIC (First Step) - CRITIC11: Relative Observations")
            print("="*80)
            print(f"Global state shape: {global_state_np.shape}")
            print(f"Expected: [{self.n_envs}, 9] for 2 agents (relative observations)")
            print(f"\nEnvironment 0 global state (9 dims):")
            print(f"  Robot1 to box:  dx={global_state_np[0,0]:.3f}, dy={global_state_np[0,1]:.3f}, dψ={global_state_np[0,2]:.3f}")
            print(f"  Robot2 to box:  dx={global_state_np[0,3]:.3f}, dy={global_state_np[0,4]:.3f}, dψ={global_state_np[0,5]:.3f}")
            print(f"  Inter-robot distance: {global_state_np[0,6]:.3f}")
            print(f"  Goal to box:    dx={global_state_np[0,7]:.3f}, dy={global_state_np[0,8]:.3f}")
            print(f"\nStatistics across all {self.n_envs} environments:")
            print(f"  Min values:  {np.min(global_state_np, axis=0)}")
            print(f"  Max values:  {np.max(global_state_np, axis=0)}")
            print(f"  Mean values: {np.mean(global_state_np, axis=0)}")
            print(f"  Std values:  {np.std(global_state_np, axis=0)}")
            print(f"\nNaN count: {np.isnan(global_state_np).sum()}")
            print(f"Inf count: {np.isinf(global_state_np).sum()}")
            print("="*80 + "\n")
            self._logged_global_state = True

        # Handle NaN and Inf
        nan_count = np.isnan(global_state_np).sum()
        inf_count = np.isinf(global_state_np).sum()
        if nan_count > 0 or inf_count > 0:
            print(f"WARNING: Found {nan_count} NaN and {inf_count} Inf values in global state!")
        global_state_np[np.isnan(global_state_np)] = 0.0
        global_state_np[np.isinf(global_state_np)] = 0.0

        return global_state_np

    def _construct_vel_global_state(self) -> np.ndarray:
        """Construct global state for velocity task.

        Structure (11 dims for 2 agents):
            [cos(cmd_dir), sin(cmd_dir), cmd_speed,   # velocity command (3)
             box_vx, box_vy, box_ang_vel_z,           # box velocity (3)
             agent0_rel_x, agent0_rel_y,              # agent0 relative to box (2)
             agent1_rel_x, agent1_rel_y,              # agent1 relative to box (2)
             inter_agent_dist]                         # distance between agents (1)

        Returns:
            global_state_np: [n_envs, 11] numpy array
        """
        wrapper = self.env

        # Velocity command from wrapper
        cmd_dir = wrapper.cmd_direction  # (n_envs,)
        cmd_speed = wrapper.cmd_speed    # (n_envs,)

        # NPC states
        npc_states = wrapper.root_states_npc.reshape(self.n_envs, wrapper.num_npcs, -1)
        box_pos_global = npc_states[:, 0, :3] - wrapper.env.env_origins
        box_lin_vel = npc_states[:, 0, 7:10]   # (n_envs, 3)
        box_ang_vel = npc_states[:, 0, 10:13]  # (n_envs, 3)

        # Agent states
        obs_buf = wrapper.env.obs_buf if hasattr(wrapper, 'env') else wrapper.obs_buf
        base_pos = obs_buf.base_pos.reshape(self.n_envs, self.n_agents, 3)

        # Build state components
        state_list = [
            torch.cos(cmd_dir).unsqueeze(1),          # cos(cmd_dir) (n_envs, 1)
            torch.sin(cmd_dir).unsqueeze(1),           # sin(cmd_dir) (n_envs, 1)
            cmd_speed.unsqueeze(1),                     # cmd_speed (n_envs, 1)
            box_lin_vel[:, :2],                         # box vx, vy (n_envs, 2)
            box_ang_vel[:, 2:3],                        # box angular vel z (n_envs, 1)
        ]

        # Agent positions relative to box
        for agent_id in range(self.n_agents):
            agent_rel = base_pos[:, agent_id, :2] - box_pos_global[:, :2]
            state_list.append(agent_rel)  # (n_envs, 2)

        # Inter-agent distance
        inter_agent_dist = torch.norm(
            base_pos[:, 0, :2] - base_pos[:, 1, :2], dim=1, keepdim=True
        )
        state_list.append(inter_agent_dist)  # (n_envs, 1)

        global_state_torch = torch.cat(state_list, dim=1)
        global_state_np = global_state_torch.cpu().numpy().astype(np.float32)

        # Diagnostic logging (first call only)
        if not hasattr(self, '_logged_global_state'):
            print("\n" + "=" * 80)
            print("GLOBAL STATE DIAGNOSTIC (First Step) - Velocity Task")
            print("=" * 80)
            print(f"Global state shape: {global_state_np.shape}")
            print(f"Expected: [{self.n_envs}, 11] for 2 agents")
            print(f"\nEnvironment 0 global state (11 dims):")
            print(f"  cmd: cos={global_state_np[0,0]:.3f}, sin={global_state_np[0,1]:.3f}, speed={global_state_np[0,2]:.3f}")
            print(f"  box_vel: vx={global_state_np[0,3]:.3f}, vy={global_state_np[0,4]:.3f}, ang_z={global_state_np[0,5]:.3f}")
            print(f"  agent0_rel: x={global_state_np[0,6]:.3f}, y={global_state_np[0,7]:.3f}")
            print(f"  agent1_rel: x={global_state_np[0,8]:.3f}, y={global_state_np[0,9]:.3f}")
            print(f"  inter_agent_dist: {global_state_np[0,10]:.3f}")
            print("=" * 80 + "\n")
            self._logged_global_state = True

        # Handle NaN/Inf
        global_state_np[np.isnan(global_state_np)] = 0.0
        global_state_np[np.isinf(global_state_np)] = 0.0

        return global_state_np

    def step(self, actions: np.ndarray) -> Tuple:
        """Step the environment.

        Args:
            actions: [n_envs, n_agents, action_dim]
                     For hetero mode: action_dim varies per agent (e.g., [3, 2])
                     HARL may pad to max_dim, which we'll handle here

        Returns:
            obs: [n_envs, n_agents, obs_dim]
            state: [n_envs, global_state_dim] - TRUE GLOBAL STATE
            rewards: [n_envs, n_agents, 1]
            dones: [n_envs, n_agents]
            infos: list of dicts
            available_actions: None
        """
        # Convert to torch: actions already in [n_envs, n_agents, action_dim] format
        actions_torch = torch.from_numpy(actions).cuda()

        # For heterogeneous mode: If actions are padded to max_dim, extract per-agent dims
        # HAPPO's separate actor networks output correct dimensions per agent
        # but they may be padded when batched. The environment wrapper handles this.
        if self.is_hetero:
            # Actions shape: [n_envs, n_agents, max_action_dim]
            # Each agent uses only its actual action dimensions
            # The Go1PushMidWrapper will handle masking/extraction
            pass  # Wrapper handles it automatically

        # CRITIC6 REVERTED (Dec 21, 2025): Action scaling BROKE learning!
        # Evidence: critic3 (no scaling) achieved 20% success in 100M steps
        #           critic6 (with 0.5x scaling) achieved 0% success in 100M steps
        # The 0.5x scaling made actions too small to push the box effectively.
        # MQE wrapper already applies 0.5x scaling, so effective range is [-0.5, 0.5] m/s
        # This is sufficient for the task. Do NOT add additional scaling here.
        #
        # DISABLED:
        # actions_torch = (0.5 * actions_torch).clamp(-1.0, 1.0)

        # Step environment
        obs, rewards, dones, infos = self.env.step(actions_torch)

        # Convert to numpy
        obs_np = obs.cpu().numpy()  # [n_envs, n_agents, obs_dim]
        rewards_np = rewards.cpu().numpy()  # [n_envs, n_agents]
        dones_np = dones.cpu().numpy()  # [n_envs]

        # Construct global state based on task type and critic mode
        if self.is_velocity_task:
            global_state_np = self._construct_vel_global_state()
        elif self.use_relative_obs_critic:
            # CRITIC11: Relative observations with inter-robot distance
            # Construct: [robot1_to_box(3), robot2_to_box(3), inter_robot_dist(1), goal_to_box(2)]
            global_state_np = self._construct_relative_obs_state()
        elif self.use_concat_agent_observations_critic:
            # CRITIC10: Simply concatenate agent observations (no modification)
            # obs_np is [n_envs, n_agents, obs_dim], flatten to [n_envs, n_agents * obs_dim]
            global_state_np = obs_np.reshape(self.n_envs, -1)
        else:
            # CRITIC16/CRITIC9/CRITIC7: Use goal-centered, box-centered, or absolute global state
            # Construct global state with everything relative to goal, box, or in world frame
            global_state_np = self._construct_global_state()

        # For HARL EP mode compatibility, broadcast to [n_envs, n_agents, global_state_dim]
        # The runner will use state[:, 0] to get the global state
        state_np = np.broadcast_to(
            global_state_np[:, np.newaxis, :],
            (self.n_envs, self.n_agents, global_state_np.shape[1])
        )

        # Reshape rewards: [n_envs, n_agents] -> [n_envs, n_agents, 1]
        rewards_np = rewards_np[..., np.newaxis]

        # Dones - broadcast to all agents
        dones_np = np.broadcast_to(dones_np[:, np.newaxis], (self.n_envs, self.n_agents))

        # Infos - HARL expects list of dicts with agent ID keys
        # For EP mode: info[0] is used; for FP mode: info[agent_id] for all agents
        infos_list = [{agent_id: {} for agent_id in range(self.n_agents)} for _ in range(self.n_envs)]

        # Track statistics for episodes that just finished
        for env_idx in range(self.n_envs):
            if dones_np[env_idx, 0]:  # Episode done
                if self.is_velocity_task:
                    # Velocity task: no success/failure, just track episode completion
                    self.episode_success.append(False)  # No binary success metric
                else:
                    # Track success (finished_buf from wrapper)
                    success = False
                    if hasattr(self.env, 'finished_buf'):
                        success = bool(self.env.finished_buf[env_idx].item())
                    self.episode_success.append(success)

                # Note: Episode length, collision, and collaboration stats are not tracked
                # for mid-level task. High-level task will implement these.

        # Available actions - None for continuous action space
        return obs_np, state_np, rewards_np, dones_np, infos_list, None

    def reset(self) -> Tuple:
        """Reset the environment.

        Returns:
            obs: [n_envs, n_agents, obs_dim]
            state: [n_envs, n_agents, global_state_dim] - Concatenated local observations
            available_actions: None (continuous action space)
        """
        obs = self.env.reset()
        obs_np = obs.cpu().numpy()

        # Construct global state based on task type and critic mode
        if self.is_velocity_task:
            global_state_np = self._construct_vel_global_state()
        elif self.use_relative_obs_critic:
            # CRITIC11: Relative observations with inter-robot distance
            # Construct: [robot1_to_box(3), robot2_to_box(3), inter_robot_dist(1), goal_to_box(2)]
            global_state_np = self._construct_relative_obs_state()
        elif self.use_concat_agent_observations_critic:
            # CRITIC10: Simply concatenate agent observations (no modification)
            # obs_np is [n_envs, n_agents, obs_dim], flatten to [n_envs, n_agents * obs_dim]
            global_state_np = obs_np.reshape(self.n_envs, -1)
        else:
            # CRITIC16/CRITIC9/CRITIC7: Use goal-centered, box-centered, or absolute global state
            # Construct global state with everything relative to goal, box, or in world frame
            global_state_np = self._construct_global_state()

        # For HARL EP mode compatibility, broadcast to [n_envs, n_agents, global_state_dim]
        state_np = np.broadcast_to(
            global_state_np[:, np.newaxis, :],
            (self.n_envs, self.n_agents, global_state_np.shape[1])
        )

        return obs_np, state_np, None

    def seed(self, seed: int):
        """Set random seed."""
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        np.random.seed(seed)

    def close(self):
        """Close the environment."""
        # MAPush environments don't have a close method, just pass
        pass

    def get_statistics(self):
        """Get accumulated statistics.

        Returns:
            dict with keys: success_rate, collision_rate, avg_episode_length, collaboration_degree
        """
        num_success = sum(self.episode_success) if self.episode_success else 0

        stats = {
            'success_rate': np.mean(self.episode_success) if self.episode_success else 0.0,
            'collision_rate': np.mean(self.episode_collision) if self.episode_collision else 0.0,
            'avg_episode_length': np.mean(self.episode_lengths) if self.episode_lengths else 0.0,
            'collaboration_degree': np.mean(self.episode_collaboration) if self.episode_collaboration else 0.0,
            'num_episodes': len(self.episode_success),
            'num_success': num_success,
            'num_collision_tracked': len(self.episode_collision),
            'num_collab_tracked': len(self.episode_collaboration),
        }
        return stats

    def reset_statistics(self):
        """Reset statistics buffers.

        Using deque with maxlen=1000 to prevent unbounded memory growth.
        This limits statistics to the most recent 1000 episodes.
        """
        self.episode_success = deque(maxlen=1000)
        self.episode_collision = deque(maxlen=1000)
        self.episode_lengths = deque(maxlen=1000)
        self.episode_collaboration = deque(maxlen=1000)
