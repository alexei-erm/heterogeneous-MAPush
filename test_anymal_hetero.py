"""
Test script for Anymal C heterogeneous environment (Go1 + Anymal C)
"""
import sys

# CRITICAL: Import isaacgym before torch
import isaacgym

import torch

print("="*70)
print("Test: Anymal C Heterogeneous Environment (Go1 + Anymal C)")
print("="*70)

# Test 1: Import Anymal C
print("\nTest 1: Importing Anymal C...")
try:
    from mqe.envs.anymal_c import AnymalC, AnymalCCfg
    print("✓ Anymal C import successful")
except Exception as e:
    print(f"✗ Anymal C import failed: {e}")
    sys.exit(1)

# Test 2: Check robot registry
print("\nTest 2: Checking robot registry...")
try:
    from mqe.envs.robot_registry import get_robot_info, list_available_robots
    robots = list_available_robots()
    print(f"✓ Available robots: {robots}")

    anymal_info = get_robot_info('anymal_c')
    print(f"✓ Anymal C registered with control type: {anymal_info['default_control']}")
except Exception as e:
    print(f"✗ Registry check failed: {e}")
    sys.exit(1)

# Test 3: Validate heterogeneous agent combination
print("\nTest 3: Validating heterogeneous agents (Go1 + Anymal C)...")
try:
    from mqe.utils.hetero_config import validate_hetero_agents
    is_valid, message = validate_hetero_agents(['go1', 'anymal_c'])
    if is_valid:
        print(f"✓ {message}")
    else:
        print(f"✗ Validation failed: {message}")
        sys.exit(1)
except Exception as e:
    print(f"✗ Validation failed: {e}")
    sys.exit(1)

# Test 4: Check policy file exists
print("\nTest 4: Checking policy file...")
import os
policy_path = "./resources/robots/anymal_c/policy_500.jit/policy_1.pt"
if os.path.exists(policy_path):
    print(f"✓ Policy file found at {policy_path}")
    file_size = os.path.getsize(policy_path) / (1024 * 1024)  # MB
    print(f"  Size: {file_size:.2f} MB")
else:
    print(f"✗ Policy file NOT found at {policy_path}")
    print("  Please ensure the policy was exported correctly")
    sys.exit(1)

# Test 5: Create heterogeneous environment
print("\nTest 5: Creating heterogeneous environment (Go1 + Anymal C)...")
print("  This may take a moment to initialize Isaac Gym...")

try:
    import argparse
    from isaacgym import gymapi
    from mqe.envs.utils import make_mqe_env
    from task.cuboid.config import Go1PushMidCfg

    args = argparse.Namespace()
    args.task = "go1push_mid"
    args.num_envs = 2  # Small number for testing
    args.seed = 1
    args.headless = True
    args.record_video = False
    args.rl_device = "cuda:0"
    args.sim_device = "cuda:0"
    args.device = "cuda"
    args.compute_device_id = 0
    args.sim_device_type = "cuda"
    args.use_gpu_pipeline = True
    args.physics_engine = gymapi.SIM_PHYSX
    args.use_gpu = True
    args.subscenes = 0
    args.num_threads = 0

    print("  Creating environment with 2 envs × 2 agents (Go1 + Anymal C)...")
    from mqe.envs.utils import make_hetero_env, custom_cfg

    agent_types = ['go1', 'anymal_c']
    env, env_cfg = make_hetero_env(
        args.task,
        agent_types,
        args,
        custom_cfg=custom_cfg(args, hetero_agent='anymal_c')
    )

    print(f"✓ Environment created successfully!")
    print(f"  Num envs: {env.num_envs}")
    print(f"  Num agents: {env.num_agents}")
    print(f"  Total DOFs: {env.num_dof}")

    # Test 6: Reset environment
    print("\nTest 6: Resetting environment...")
    obs = env.reset()
    print(f"✓ Environment reset successful")
    print(f"  Observation shape: {obs.shape}")

    # Test 7: Step environment
    print("\nTest 7: Stepping environment...")
    actions = torch.zeros((env.num_envs, env.num_agents, 3), device='cuda:0')
    obs, rewards, dones, infos = env.step(actions)
    print(f"✓ Environment step successful")
    print(f"  Observations: {obs.shape}")
    print(f"  Rewards: {rewards.shape}")
    print(f"  Dones: {dones.shape}")

    # Close environment (if method exists)
    if hasattr(env, 'close'):
        env.close()
    print("\n✓ All tests passed! Anymal C heterogeneous environment is ready.")

except Exception as e:
    print(f"✗ Environment creation/testing failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print("\n" + "="*70)
print("SUCCESS: Ready to train with Go1 + Anymal C!")
print("="*70)
