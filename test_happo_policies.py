#!/usr/bin/env python3
"""
Test heterogeneous environment with trained HAPPO mid-level policies.

This will test if robots can coordinate with the actual trained policies
instead of using zero-velocity locomotion policies.
"""

import sys
sys.path.append('/home/gvlab/new-universal-MAPush')
sys.path.append('/home/gvlab/new-universal-MAPush/HARL')

# Import Isaac Gym FIRST
from isaacgym import gymapi
from isaacgym import gymutil

import torch
import torch.nn as nn
from mqe.envs.utils import make_hetero_env
from mqe.utils.helpers import get_args
from harl.models.policy_models.stochastic_mlp_policy import StochasticMlpPolicy
from gym import spaces
import numpy as np


class SimpleActor:
    """Simple wrapper to load and use trained HAPPO actor."""

    def __init__(self, checkpoint_path, obs_dim, action_dim, device='cuda'):
        """Load actor from checkpoint.

        Args:
            checkpoint_path: Path to actor .pt file
            obs_dim: Observation dimension
            action_dim: Action dimension (3 for velocity commands)
            device: Device to run on
        """
        self.device = device

        # Create model architecture matching training
        # These args match the HAPPO config
        args = {
            'hidden_sizes': [256, 256],
            'gain': 0.01,
            'initialization_method': 'orthogonal_',
            'use_feature_normalization': True,
            'use_recurrent_policy': False,
            'use_naive_recurrent_policy': False,
            'recurrent_n': 1,
            'use_policy_active_masks': True,
            'use_policy_vhead': False,
            'use_orthogonal': True,
            'activation_func': 'relu',  # Fixed: was activation_id
            'std_x_coef': 1.0,
            'std_y_coef': 0.5,
        }

        # Create spaces
        obs_space = spaces.Box(low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32)
        action_space = spaces.Box(low=-1.0, high=1.0, shape=(action_dim,), dtype=np.float32)

        # Create model
        self.model = StochasticMlpPolicy(args, obs_space, action_space, device=torch.device(device))

        # Load checkpoint
        state_dict = torch.load(checkpoint_path, map_location=device)
        self.model.load_state_dict(state_dict)
        self.model.eval()

        print(f"✅ Loaded actor from: {checkpoint_path}")
        print(f"   Obs dim: {obs_dim}, Action dim: {action_dim}")

    def get_actions(self, obs, deterministic=False):
        """Get actions from observations.

        Args:
            obs: Observations [batch, obs_dim]
            deterministic: Whether to use deterministic policy

        Returns:
            actions: Actions [batch, action_dim]
        """
        with torch.no_grad():
            actions = self.model(obs, stochastic=not deterministic)
        return actions


def test_happo_policies():
    """Test with trained HAPPO mid-level policies."""

    print("\n" + "="*60)
    print("TESTING HETERO ENV WITH TRAINED HAPPO POLICIES")
    print("="*60)

    # Get proper args with defaults
    args = get_args()
    args.task = 'go1push_mid'
    args.headless = False  # Show viewer
    args.num_envs = 1
    args.record_video = False

    # Create hetero environment (Go1 + Anymal C)
    print("\n[1] Creating heterogeneous environment...")
    print("    Agent 0: Go1")
    print("    Agent 1: Anymal C")

    env, env_cfg = make_hetero_env(
        env_name='go1push_mid',
        agent_types=['go1', 'anymal_c'],
        args=args
    )

    print(f"✅ Environment created")
    print(f"   Num envs: {env.num_envs}")
    print(f"   Num agents: {env.num_agents}")

    # Load trained HAPPO actors
    print("\n[2] Loading trained HAPPO policies...")
    checkpoint_dir = "results/mapush/go1push_mid/happo/go1_anymalc_hetero_v1/seed-00001-2026-01-29-15-02-43/checkpoints/10M"

    # Note: Both agents have same observation space in HAPPO (they use shared critic)
    # But we need to check what obs dim was used during training
    # For MAPush mid-level, observation is typically: [box_pos(2), box_vel(2), agent_pos(2), agent_vel(2)] = 8 dim
    # But let's check the actual trained model

    # Check obs dim from checkpoint
    test_checkpoint = torch.load(f"{checkpoint_dir}/actor_agent0.pt", map_location='cpu')
    # First layer weight shape: [hidden_size, obs_dim]
    obs_dim = test_checkpoint['base.mlp.fc.0.weight'].shape[1]
    print(f"   Detected obs_dim from checkpoint: {obs_dim}")

    action_dim = 3  # [vx, vy, vyaw]

    actor_agent0 = SimpleActor(
        f"{checkpoint_dir}/actor_agent0.pt",
        obs_dim=obs_dim,
        action_dim=action_dim,
        device='cuda'
    )

    actor_agent1 = SimpleActor(
        f"{checkpoint_dir}/actor_agent1.pt",
        obs_dim=obs_dim,
        action_dim=action_dim,
        device='cuda'
    )

    # Reset environment
    print("\n[3] Resetting environment...")
    obs = env.reset()
    print(f"✅ Reset successful")
    print(f"   Obs shape: {obs.shape}")
    print(f"   Expected obs shape per agent: [{env.num_envs}, {obs_dim}]")

    # Check if obs dimension matches
    if obs.shape[-1] != obs_dim:
        print(f"\n⚠️  WARNING: Observation dimension mismatch!")
        print(f"   Environment provides: {obs.shape[-1]} dims")
        print(f"   Policy expects: {obs_dim} dims")
        print(f"\n   This might be because:")
        print(f"   1. Environment provides full state, policy was trained on partial state")
        print(f"   2. Environment changed since training")
        print(f"   We'll try to extract the first {obs_dim} dims as the policy input")

    # Run test
    print("\n[4] Running test with HAPPO policies...")
    num_steps = 500  # ~10 seconds at 50Hz

    reset_count = 0
    step_count = 0

    for step in range(num_steps):
        # Get observations for each agent
        # obs shape: [num_envs, num_agents, obs_dim] or [num_envs * num_agents, obs_dim]
        if len(obs.shape) == 3:
            obs_agent0 = obs[:, 0, :obs_dim]  # [num_envs, obs_dim]
            obs_agent1 = obs[:, 1, :obs_dim]
        else:
            # Flatten format
            obs_agent0 = obs[0::2, :obs_dim]  # Even indices
            obs_agent1 = obs[1::2, :obs_dim]  # Odd indices

        # Get actions from policies
        actions_agent0 = actor_agent0.get_actions(obs_agent0, deterministic=False)
        actions_agent1 = actor_agent1.get_actions(obs_agent1, deterministic=False)

        # Combine actions: [num_envs, num_agents, action_dim]
        actions = torch.stack([actions_agent0, actions_agent1], dim=1)

        # Print first action for debugging
        if step == 0:
            print(f"\n   First step observations:")
            print(f"   Agent 0 (Go1) obs: {obs_agent0[0]}")
            print(f"   Agent 1 (Anymal C) obs: {obs_agent1[0]}")
            print(f"\n   First step actions:")
            print(f"   Agent 0 (Go1): {actions_agent0[0]}")
            print(f"   Agent 1 (Anymal C): {actions_agent1[0]}")

        # Step environment
        obs, rewards, dones, infos = env.step(actions)
        step_count += 1

        # Print first few rewards for debugging
        if step < 10:
            print(f"   Step {step}: reward={rewards[0]}, done={dones[0]}")

        if dones.any():
            reset_count += 1
            # Print why it reset
            if step < 50:  # Only print early resets
                print(f"   [RESET at step {step}] reward={rewards[0]}")
            obs = env.reset()
            step_count = 0

        # Print progress every 100 steps
        if step % 100 == 0:
            print(f"  Step {step}/{num_steps} - Resets: {reset_count}")

    print("\n" + "="*60)
    print("TEST RESULTS")
    print("="*60)
    print(f"Total steps: {num_steps}")
    print(f"Total resets: {reset_count}")

    if reset_count == 0:
        print("\n✅ SUCCESS: Completed entire test without resets!")
        print("   -> HAPPO policies work correctly with heterogeneous environment")
    elif reset_count < 5:
        print(f"\n⚠️  PARTIAL SUCCESS: Only {reset_count} resets")
        print("   -> Policies mostly stable, some edge cases might need tuning")
    else:
        print(f"\n❌ FAILURE: {reset_count} resets in {num_steps} steps")
        print("   -> Policies might not match environment observations")
        print("   -> Check observation space consistency")

    print("="*60 + "\n")

    env.close()


if __name__ == "__main__":
    test_happo_policies()
