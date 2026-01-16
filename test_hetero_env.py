#!/usr/bin/env python3
"""
Quick test script to verify heterogeneous environment creation.

Usage:
    conda activate mapush
    python test_hetero_env.py
"""

import sys
from types import SimpleNamespace

# CRITICAL: Import isaacgym FIRST before torch!
import isaacgym
import torch

# Test 1: Import test
print("="*60)
print("Test 1: Testing imports...")
print("="*60)

try:
    from mqe.envs.jackal import Jackal, JackalCfg
    print("✓ Jackal import successful")
except Exception as e:
    print(f"✗ Jackal import failed: {e}")
    sys.exit(1)

try:
    from mqe.envs.robot_registry import list_available_robots, get_robot_info
    robots = list_available_robots()
    print(f"✓ Robot registry successful. Available robots: {robots}")
except Exception as e:
    print(f"✗ Robot registry failed: {e}")
    sys.exit(1)

# Test 2: Robot info test
print("\n" + "="*60)
print("Test 2: Testing robot info retrieval...")
print("="*60)

try:
    go1_info = get_robot_info('go1')
    print(f"✓ Go1 info: {go1_info['num_actions']} actions, control: {go1_info['default_control']}")

    jackal_info = get_robot_info('jackal')
    print(f"✓ Jackal info: {jackal_info['num_actions']} actions, control: {jackal_info['default_control']}")
except Exception as e:
    print(f"✗ Robot info retrieval failed: {e}")
    sys.exit(1)

# Test 3: Heterogeneous validation
print("\n" + "="*60)
print("Test 3: Testing heterogeneous validation...")
print("="*60)

try:
    from mqe.utils.hetero_config import validate_hetero_agents
    is_valid, message = validate_hetero_agents(['go1', 'jackal'])
    if is_valid:
        print(f"✓ Heterogeneous agents validated: {message}")
    else:
        print(f"✗ Validation failed: {message}")
        sys.exit(1)
except Exception as e:
    print(f"✗ Validation test failed: {e}")
    sys.exit(1)

# Test 4: Environment creation (the main test!)
print("\n" + "="*60)
print("Test 4: Creating heterogeneous environment (Go1 + Jackal)...")
print("="*60)

try:
    from mqe.envs.utils import make_hetero_env
    from mqe.utils.helpers import get_args

    # Get proper args with all defaults
    import sys
    sys.argv = [
        'test_hetero_env.py',
        '--task=go1push_mid',
        '--headless',
        '--num_envs=2',
        '--seed=1',
    ]
    args = get_args()
    args.record_video = False

    print("Creating heterogeneous environment...")
    env, env_cfg = make_hetero_env(
        env_name='go1push_mid',
        agent_types=['go1', 'jackal'],
        args=args
    )

    print(f"✓ Environment created successfully!")
    print(f"  - Number of environments: {env.num_envs}")
    print(f"  - Number of agents: {env.num_agents}")
    print(f"  - Agent types: {env.hetero_agent_types}")
    print(f"  - Total actuated DOFs: {env.num_actuated_dof}")
    print(f"  - Robot DOFs: {env.robot_num_dofs}")

    # Test 5: Reset environment
    print("\n" + "="*60)
    print("Test 5: Testing environment reset...")
    print("="*60)

    obs = env.reset()
    print(f"✓ Environment reset successful!")
    print(f"  - Observation shape: {obs.shape}")

    # Test 6: Single step
    print("\n" + "="*60)
    print("Test 6: Testing single environment step...")
    print("="*60)

    # Create random actions (shape: [num_envs, num_agents, action_dim])
    actions = torch.rand(env.num_envs, env.num_agents, 3, device=env.device) * 2 - 1

    obs, rewards, dones, infos = env.step(actions)
    print(f"✓ Environment step successful!")
    print(f"  - Observation shape: {obs.shape}")
    print(f"  - Rewards shape: {rewards.shape}")
    print(f"  - Dones shape: {dones.shape}")

    # Close environment (optional - not critical for training)
    # env.close()
    print("\n" + "="*60)
    print("✓ ALL TESTS PASSED!")
    print("="*60)
    print("\nHeterogeneous environment (Go1 + Jackal) is working correctly!")
    print("You can now proceed with training.")

except Exception as e:
    print(f"✗ Environment creation/test failed!")
    print(f"Error: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
