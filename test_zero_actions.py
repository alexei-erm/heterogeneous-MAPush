"""
Test heterogeneous environment with ZERO actions.

This isolates whether the problem is:
1. Locomotion policy issues -> If works with zero actions, policy is broken
2. Physics/spawn issues -> If fails with zero actions, spawn/physics problem
"""

import sys
sys.path.append('/home/gvlab/new-universal-MAPush')

# Import Isaac Gym FIRST
from isaacgym import gymapi
from isaacgym import gymutil

import torch
from mqe.envs.utils import make_hetero_env
from mqe.utils.helpers import get_args

def test_zero_actions():
    """Test with zero actions to see if robots can stand still."""

    print("\n" + "="*60)
    print("TESTING HETERO ENV WITH ZERO ACTIONS")
    print("="*60)

    # Get proper args with defaults
    args = get_args()
    args.task = 'go1push_mid'
    args.headless = False  # Show viewer
    args.num_envs = 1
    args.record_video = False  # Disable video recording

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
    print(f"   Total DOFs: {env.num_actuated_dof}")

    # Reset environment
    print("\n[2] Resetting environment...")
    obs = env.reset()
    print(f"✅ Reset successful")
    print(f"   Obs shape: {obs.shape}")

    # Test with SMALL CONSTANT velocity commands
    print("\n[3] Stepping with SMALL CONSTANT velocity commands...")
    print("    Testing if policies work better with non-zero commands")

    num_steps = 500  # ~10 seconds at 50Hz

    # Small constant velocity: [num_envs, num_agents, action_dim]
    # action_dim = 3 for both agents [vx, vy, vyaw]
    # Try small forward velocity (0.2 m/s) instead of zero
    constant_actions = torch.zeros(env.num_envs, env.num_agents, 3, device='cuda')
    constant_actions[:, :, 0] = 0.2  # Small forward velocity

    print(f"    Command: vx=0.2 m/s, vy=0.0 m/s, vyaw=0.0 rad/s")

    reset_count = 0
    step_count = 0

    for step in range(num_steps):
        obs, rewards, dones, infos = env.step(constant_actions)
        step_count += 1

        if dones.any():
            reset_count += 1
            # Reset and continue (silent)
            obs = env.reset()
            step_count = 0

        # Print progress every 100 steps (~2 seconds) - minimal output
        if step % 100 == 0:
            print(f"  Step {step}/{num_steps} - Resets: {reset_count}")

    print("\n" + "="*60)
    print("TEST RESULTS")
    print("="*60)
    print(f"Total steps: {num_steps}")
    print(f"Total resets: {reset_count}")

    if reset_count == 0:
        print("\n✅ SUCCESS: Robots stood stable for entire test!")
        print("   -> Problem is likely in locomotion policy or action processing")
    elif reset_count < 5:
        print(f"\n⚠️  PARTIAL SUCCESS: Only {reset_count} resets")
        print("   -> Spawn might be slightly unstable but not catastrophic")
    else:
        print(f"\n❌ FAILURE: {reset_count} resets in {num_steps} steps")
        print("   -> Physics/spawn issue - robots can't even stand still")

        if reset_count > 20:
            print("   -> CRITICAL: Immediate collapse on spawn")

    print("="*60 + "\n")

    env.close()

if __name__ == "__main__":
    test_zero_actions()
