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
        self.env, self.env_cfg = make_mqe_env(
            args.task,
            args,
            custom_cfg=custom_cfg(args, individualized_rewards=individualized_rewards)
        )

        self.n_envs = self.env.num_envs
        self.n_agents = self.env.num_agents

        # HARL expects list of spaces (one per agent)
        # For homogeneous case (both agents have same spaces), duplicate
        self.observation_space = [self.env.observation_space] * self.n_agents
        self.action_space = [self.env.action_space] * self.n_agents

        # Share observation space (for centralized critic)
        # Using same as observation for now (can be extended for global state)
        self.share_observation_space = [self.env.observation_space] * self.n_agents

        # Statistics tracking (for calculator mode)
        self.reset_statistics()

    def step(self, actions: np.ndarray) -> Tuple:
        """Step the environment.

        Args:
            actions: [n_envs, n_agents, action_dim]

        Returns:
            obs: [n_envs, n_agents, obs_dim]
            state: [n_envs, n_agents, state_dim]
            rewards: [n_envs, n_agents, 1]
            dones: [n_envs, n_agents]
            infos: list of dicts
            available_actions: None
        """
        # Convert to torch: actions already in [n_envs, n_agents, action_dim] format
        actions_torch = torch.from_numpy(actions).cuda()

        # Step environment
        obs, rewards, dones, infos = self.env.step(actions_torch)

        # Convert to numpy
        obs_np = obs.cpu().numpy()  # [n_envs, n_agents, obs_dim]
        rewards_np = rewards.cpu().numpy()  # [n_envs, n_agents]
        dones_np = dones.cpu().numpy()  # [n_envs]

        # State (same as obs for now - can be customized for global state)
        state_np = obs_np.copy()

        # Reshape rewards: [n_envs, n_agents] -> [n_envs, n_agents, 1]
        rewards_np = rewards_np[..., np.newaxis]

        # Dones - broadcast to all agents
        dones_np = np.broadcast_to(dones_np[:, np.newaxis], (self.n_envs, self.n_agents))

        # Infos - HARL expects list of dicts with agent ID keys for EP mode
        # For EP (Environment Provided) state, info[0] contains shared info
        infos_list = [{0: {}} for _ in range(self.n_envs)]

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
            state: [n_envs, n_agents, state_dim]
            available_actions: None (continuous action space)
        """
        obs = self.env.reset()
        obs_np = obs.cpu().numpy()
        state_np = obs_np.copy()
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
