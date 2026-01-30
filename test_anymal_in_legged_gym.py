#!/usr/bin/env python3
"""
Test Anymal C policy in ORIGINAL legged_gym environment.
This will show us what observations the policy actually expects.
"""

import sys
sys.path.append('/home/gvlab/legged_gym')

from isaacgym import gymapi
import torch
import os

# Set environment variable for legged_gym
os.environ['LEGGED_GYM_ROOT_DIR'] = '/home/gvlab/legged_gym'

from legged_gym.envs import *
from legged_gym import LEGGED_GYM_ROOT_DIR
from legged_gym.utils import get_args, task_registry

def test_anymal_in_legged_gym():
    print("\n" + "="*60)
    print("TESTING ANYMAL C IN ORIGINAL LEGGED_GYM")
    print("="*60)

    # Get args
    args = get_args()
    args.task = 'anymal_c_flat'
    args.headless = True
    args.num_envs = 1

    # Create environment
    print("\n[1] Creating legged_gym Anymal C environment...")
    env, env_cfg = task_registry.make_env(name=args.task, args=args)

    print(f"✅ Environment created")
    print(f"   Num envs: {env.num_envs}")

    # Reset
    print("\n[2] Resetting...")
    obs_tuple = env.reset()
    if isinstance(obs_tuple, tuple):
        obs = obs_tuple[0]
    else:
        obs = obs_tuple

    print(f"✅ Reset successful")
    print(f"   Obs shape: {obs.shape}")

    # Print observation structure
    print("\n[3] Observation at spawn (zero commands):")
    print(f"   Full obs shape: {obs.shape}")
    print(f"   Obs[0]: {obs[0]}")

    # Break down the 48-dim observation
    print("\n   Observation breakdown:")
    print(f"   [0:3] base_lin_vel * scale: {obs[0, 0:3]}")
    print(f"   [3:6] base_ang_vel * scale: {obs[0, 3:6]}")
    print(f"   [6:9] projected_gravity: {obs[0, 6:9]}")
    print(f"   [9:12] commands * scale: {obs[0, 9:12]}")
    print(f"   [12:24] (dof_pos - default) * scale: {obs[0, 12:24]}")
    print(f"   [24:36] dof_vel * scale: {obs[0, 24:36]}")
    print(f"   [36:48] actions (last step): {obs[0, 36:48]}")

    # Check raw values before scaling
    print("\n[4] Raw state values:")
    print(f"   base_lin_vel (raw): {env.base_lin_vel[0]}")
    print(f"   base_ang_vel (raw): {env.base_ang_vel[0]}")
    print(f"   dof_pos (raw): {env.dof_pos[0]}")
    print(f"   default_dof_pos: {env.default_dof_pos[0]}")
    print(f"   dof_pos - default: {env.dof_pos[0] - env.default_dof_pos[0]}")
    print(f"   commands: {env.commands[0]}")

    # Check scales
    print("\n[5] Observation scales:")
    print(f"   lin_vel scale: {env.obs_scales.lin_vel}")
    print(f"   ang_vel scale: {env.obs_scales.ang_vel}")
    print(f"   dof_pos scale: {env.obs_scales.dof_pos}")
    print(f"   dof_vel scale: {env.obs_scales.dof_vel}")

    # Step with zero commands
    print("\n[6] Stepping with zero actions...")
    action = torch.zeros(1, 12, device='cuda')
    obs, rewards, dones, infos = env.step(action)

    print(f"   Obs after step: {obs[0]}")
    print(f"   Reward: {rewards[0]}")
    print(f"   Done: {dones[0]}")

    env.close()
    print("\n" + "="*60)

if __name__ == "__main__":
    test_anymal_in_legged_gym()
