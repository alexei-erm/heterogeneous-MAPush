"""
Test heterogeneous environment with constant velocity actions.

Tests Go1 + Cassie walking together in MAPush environment.
"""

import sys
sys.path.append('/home/gvlab/new-universal-MAPush')

# Import Isaac Gym FIRST
from isaacgym import gymapi
from isaacgym import gymutil

import torch
from mqe.envs.utils import make_hetero_env
from mqe.utils.helpers import get_args

def test_go1_cassie():
    """Test Go1 + Cassie walking with constant velocity."""

    print("\n" + "="*60)
    print("TESTING GO1 + CASSIE HETERO ENV")
    print("="*60)

    # Get proper args with defaults
    args = get_args()
    args.task = 'go1push_mid'
    args.headless = False  # Show viewer
    args.num_envs = 1
    args.record_video = False  # Disable video recording

    # Create hetero environment (Go1 + Cassie)
    print("\n[1] Creating heterogeneous environment...")
    print("    Agent 0: Go1 (quadruped)")
    print("    Agent 1: Cassie (biped)")

    env, env_cfg = make_hetero_env(
        env_name='go1push_mid',
        agent_types=['go1', 'cassie'],
        args=args
    )

    print(f"Environment created")
    print(f"   Num envs: {env.num_envs}")
    print(f"   Num agents: {env.num_agents}")
    print(f"   Total DOFs: {env.num_actuated_dof}")

    # Reset environment
    print("\n[2] Resetting environment...")
    obs = env.reset()
    print(f"Reset successful")

    # Test with constant forward velocity
    print("\n[3] Stepping with constant forward velocity...")
    print("    Both robots walking forward at 0.5 m/s")

    num_steps = 1000  # ~20 seconds at 50Hz

    # Constant velocity: [num_envs, num_agents, action_dim]
    # action_dim = 3 for both agents [vx, vy, vyaw]
    constant_actions = torch.zeros(env.num_envs, env.num_agents, 3, device='cuda')
    constant_actions[:, :, 0] = 0.5  # Forward velocity 0.5 m/s

    print(f"    Command: vx=0.5 m/s, vy=0.0 m/s, vyaw=0.0 rad/s")
    print(f"    Running for {num_steps} steps (~20 seconds)")
    print()

    reset_count = 0

    for step in range(num_steps):
        obs, rewards, dones, infos = env.step(constant_actions)

        if dones.any():
            reset_count += 1
            obs = env.reset()

        # Print progress every 200 steps
        if step % 200 == 0:
            print(f"  Step {step}/{num_steps} - Resets: {reset_count}")

    print("\n" + "="*60)
    print("TEST RESULTS")
    print("="*60)
    print(f"Total steps: {num_steps}")
    print(f"Total resets: {reset_count}")

    if reset_count == 0:
        print("\nSUCCESS: Both robots walked stable for entire test!")
    elif reset_count < 5:
        print(f"\nPARTIAL SUCCESS: Only {reset_count} resets")
    else:
        print(f"\nISSUE: {reset_count} resets - check locomotion policies")

    print("="*60 + "\n")

    env.close()

if __name__ == "__main__":
    test_go1_cassie()
