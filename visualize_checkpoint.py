#!/usr/bin/env python3
"""
Visualize trained heterogeneous policy from checkpoint.
Shows what the learned policy commands the robots to do.

Usage:
    python visualize_checkpoint.py --checkpoint ./results/.../checkpoints/10M
"""

import argparse
import os
import sys

# CRITICAL: Import isaacgym-related modules BEFORE torch
from mqe.envs.utils import make_hetero_env
from mqe.utils.helpers import get_args

# Now safe to import torch
import torch
import numpy as np

def load_policy(checkpoint_dir, device='cuda'):
    """Load actor model from checkpoint directory."""
    # HAPPO saves per-agent actors
    actor_files = []
    for i in range(2):  # 2 agents
        actor_path = os.path.join(checkpoint_dir, f'actor_agent{i}.pt')
        if os.path.exists(actor_path):
            actor_files.append(actor_path)
        else:
            print(f"Warning: Actor file not found: {actor_path}")

    if len(actor_files) == 0:
        raise FileNotFoundError(f"No actor models found in {checkpoint_dir}")

    # Load actor models
    actors = []
    for actor_file in actor_files:
        print(f"Loading actor: {actor_file}")
        actor_state = torch.load(actor_file, map_location=device)
        actors.append(actor_state)

    return actors

def visualize_policy(checkpoint_dir, num_steps=2000):
    """Run environment with loaded policy and viewer enabled."""

    print("="*80)
    print("VISUALIZING TRAINED HETEROGENEOUS POLICY (Go1 + Anymal C)")
    print("="*80)
    print(f"Checkpoint: {checkpoint_dir}")
    print(f"Running {num_steps} steps with LEARNED policy")
    print("Watch for:")
    print("  1. What actions does the policy command?")
    print("  2. Do robots coordinate to push the box?")
    print("  3. When does physics explode?")
    print("  4. Do robots collide violently?")
    print("="*80)

    # Get args with viewer enabled
    sys.argv = [
        'visualize_checkpoint.py',
        '--task=go1push_mid',
        '--num_envs=1',  # Single environment for easier viewing
        '--seed=1',
    ]
    args = get_args()
    args.record_video = False
    args.headless = False  # Enable viewer

    # Create environment with custom config to disable video recording
    print("\nCreating heterogeneous environment...")

    def disable_recording_cfg(cfg):
        """Disable video recording for faster visualization."""
        cfg.env.record_video = False
        return cfg

    env, env_cfg = make_hetero_env(
        env_name='go1push_mid',
        agent_types=['go1', 'anymal_c'],
        args=args,
        custom_cfg=disable_recording_cfg
    )

    # Load policy
    print("\nLoading trained policy...")
    try:
        actors = load_policy(checkpoint_dir, device='cuda')
        print(f"✓ Loaded {len(actors)} actor models")
    except Exception as e:
        print(f"✗ Failed to load policy: {e}")
        print("\nFalling back to RANDOM actions (like visualize_hetero.py)")
        actors = None

    print("\n[Viewer] Environment created. Viewer should open.")
    print("[Viewer] Press 'V' to toggle viewer camera")
    print("[Viewer] Use mouse to rotate/pan camera")
    print(f"[Viewer] Running {num_steps} steps with ZERO actions (standing still)...")
    print("[Viewer] This tests if spawn heights are correct.")
    print("\nExpected heights:")
    print("  Go1: ~0.42m")
    print("  Anymal C: ~0.6m")
    print("\nBoth robots should stand still without falling.\n")

    # Reset
    obs = env.reset()

    # Run with learned policy
    for step in range(num_steps):
        # TEMPORARY: Use ZERO actions to test spawn heights
        # Both robots should just stand still at their spawn heights
        actions = torch.zeros(2, 3, device='cuda')  # [num_envs * num_agents, 3] = [2, 3]
        # actions[0, :] = Go1 actions [vx, vy, vyaw]
        # actions[1, :] = Anymal C actions [vx, vy, vyaw]

        if step == 0:
            print("[INFO] Using ZERO actions - robots should stand still")

        # Step environment
        obs, reward, done, info = env.step(actions)

        # Check for NaN
        if torch.isnan(obs).any():
            print(f"[Step {step}] ⚠️  NaN detected in observations!")
        if torch.isnan(reward).any():
            print(f"[Step {step}] ⚠️  NaN detected in rewards!")

        # Print every 100 steps
        if step % 100 == 0:
            print(f"[Step {step}] Reward: {reward.mean():.4f}")

            # Check robot states
            base_pos = env.env.obs_buf.base_pos.reshape(1, 2, -1)
            go1_height = base_pos[0, 0, 2].item()
            anymal_height = base_pos[0, 1, 2].item()

            print(f"  Go1 height: {go1_height:.3f}m")
            print(f"  Anymal C height: {anymal_height:.3f}m")

            # Check for physics breakdown
            if go1_height < 0.1 or anymal_height < 0.1:
                print("  ⚠️  Robot(s) too low - may have fallen through ground!")
            if go1_height > 1.0 or anymal_height > 1.0:
                print("  ⚠️  Robot(s) too high - physics may be unstable!")

        if done.any():
            print(f"[Step {step}] Episode done, resetting...")
            obs = env.reset()

    print("\n" + "="*80)
    print("Visualization complete!")
    print("="*80)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Visualize trained heterogeneous policy')
    parser.add_argument(
        '--checkpoint',
        type=str,
        default='./results/mapush/go1push_mid/happo/go1_jackal_hetero_concat_critic/seed-00001-2026-01-17-22-26-05/checkpoints/10M',
        help='Path to checkpoint directory'
    )
    parser.add_argument('--steps', type=int, default=2000, help='Number of steps to run')

    args = parser.parse_args()

    visualize_policy(args.checkpoint, args.steps)
