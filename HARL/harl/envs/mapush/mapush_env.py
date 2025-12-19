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

        # Create MQE environment with custom config
        individualized_rewards = env_args.get("individualized_rewards", False)
        shared_gated_rewards = env_args.get("shared_gated_rewards", False)
        self.env, self.env_cfg = make_mqe_env(
            args.task,
            args,
            custom_cfg=custom_cfg(args, individualized_rewards=individualized_rewards,
                                  shared_gated_rewards=shared_gated_rewards)
        )

        self.n_envs = self.env.num_envs
        self.n_agents = self.env.num_agents

        # HARL expects list of spaces (one per agent)
        # For homogeneous case (both agents have same spaces), duplicate
        self.observation_space = [self.env.observation_space] * self.n_agents
        self.action_space = [self.env.action_space] * self.n_agents

        # Share observation space (for centralized critic)
        # Global state: box(3) + target(2) + agents(6 each) = 3 + 2 + 6*n_agents dims
        # [box_x, box_y, box_yaw,                                    = 3 dims
        #  target_x, target_y,                                       = 2 dims (NO yaw for target!)
        #  agent0_x, agent0_y, agent0_yaw, vx, vy, vyaw,            = 6 dims
        #  agent1_x, agent1_y, agent1_yaw, vx, vy, vyaw]            = 6 dims
        # Total for 2 agents: 3 + 2 + 6 + 6 = 17 dims
        from gym import spaces
        global_state_dim = 3 + 2 + 6 * self.n_agents  # box + target + all agents (pos + vel)
        self.share_observation_space = [
            spaces.Box(low=-float('inf'), high=float('inf'),
                      shape=(global_state_dim,), dtype=np.float32)
        ] * self.n_agents

        # Statistics tracking (for calculator mode)
        self.reset_statistics()

    def _construct_global_state(self) -> np.ndarray:
        """Construct global state from environment internals.

        Global state contains positions and orientations in global frame:
        - Box: [x, y, yaw]
        - Target: [x, y, yaw]
        - Agent 0: [x, y, yaw]
        - Agent 1: [x, y, yaw]
        ...for all agents

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

        # Construct global state: [box(3), target(2), agent0(6), agent1(6), ...]
        # Only use x, y, yaw for 2D projection
        global_state_list = [
            box_pos_global[:, :2],         # box x, y
            box_rpy[:, 2:3],               # box yaw
            target_pos_global[:, :2],      # target x, y (NO yaw! Target is just a point)
        ]

        # Add all agents' states (position + velocity)
        for agent_id in range(self.n_agents):
            global_state_list.append(base_pos[:, agent_id, :2])      # agent x, y
            global_state_list.append(base_rpy[:, agent_id, 2:3])     # agent yaw
            global_state_list.append(base_lin_vel[:, agent_id, :2])  # agent vx, vy
            global_state_list.append(base_ang_vel[:, agent_id, 2:3]) # agent vyaw (angular velocity around z)

        # Concatenate into single tensor
        global_state_torch = torch.cat(global_state_list, dim=1)  # [n_envs, 6 + 3*n_agents]

        # Convert to numpy
        import numpy as np
        global_state_np = global_state_torch.cpu().numpy().astype(np.float32)

        # Diagnostic logging (first call only)
        if not hasattr(self, '_logged_global_state'):
            print("\n" + "="*80)
            print("GLOBAL STATE DIAGNOSTIC (First Step)")
            print("="*80)
            print(f"Global state shape: {global_state_np.shape}")
            print(f"Expected: [500, 17] for 2 agents")
            print(f"\nEnvironment 0 global state (17 dims):")
            print(f"  Box:    x={global_state_np[0,0]:.3f}, y={global_state_np[0,1]:.3f}, yaw={global_state_np[0,2]:.3f}")
            print(f"  Target: x={global_state_np[0,3]:.3f}, y={global_state_np[0,4]:.3f}")
            print(f"  Agent0: x={global_state_np[0,5]:.3f}, y={global_state_np[0,6]:.3f}, yaw={global_state_np[0,7]:.3f}, vx={global_state_np[0,8]:.3f}, vy={global_state_np[0,9]:.3f}, vyaw={global_state_np[0,10]:.3f}")
            print(f"  Agent1: x={global_state_np[0,11]:.3f}, y={global_state_np[0,12]:.3f}, yaw={global_state_np[0,13]:.3f}, vx={global_state_np[0,14]:.3f}, vy={global_state_np[0,15]:.3f}, vyaw={global_state_np[0,16]:.3f}")
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

    def step(self, actions: np.ndarray) -> Tuple:
        """Step the environment.

        Args:
            actions: [n_envs, n_agents, action_dim]

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

        # CRITIC6 (Dec 19, 2025): Match OpenRL action scaling
        # OpenRL applies 0.5x scale + clip in wrapper BEFORE MQE's 0.5x scale
        # This makes effective range [-0.25, 0.25] instead of [-0.5, 0.5]
        # Without this, HARL agents move 2x faster than OpenRL agents
        actions_torch = (0.5 * actions_torch).clamp(-1.0, 1.0)

        # Step environment
        obs, rewards, dones, infos = self.env.step(actions_torch)

        # Convert to numpy
        obs_np = obs.cpu().numpy()  # [n_envs, n_agents, obs_dim]
        rewards_np = rewards.cpu().numpy()  # [n_envs, n_agents]
        dones_np = dones.cpu().numpy()  # [n_envs]

        # Construct TRUE global state in global coordinate frame
        # Shape: [n_envs, global_state_dim]
        # Content: [box_x, box_y, box_yaw, target_x, target_y, target_yaw,
        #           agent0_x, agent0_y, agent0_yaw, agent1_x, agent1_y, agent1_yaw, ...]
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
            state: [n_envs, n_agents, global_state_dim] - TRUE GLOBAL STATE (broadcasted)
            available_actions: None (continuous action space)
        """
        obs = self.env.reset()
        obs_np = obs.cpu().numpy()

        # Construct TRUE global state in global coordinate frame
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
