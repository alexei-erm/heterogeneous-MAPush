"""
Test multi-agent environment with constant velocity actions.

Supports any robot combination via agent_types parameter.
- Homogeneous (same robots): walks at 1.0 m/s
- Heterogeneous (different robots): walks at 0.5 m/s
"""

import sys
sys.path.append('/home/gvlab/new-universal-MAPush')

# Import Isaac Gym FIRST
from isaacgym import gymapi
from isaacgym import gymutil

import torch
from mqe.envs.utils import make_hetero_env
from mqe.utils.helpers import get_args

def test_env():
    """Test robot walking with constant velocity."""

    # Get proper args with defaults
    args = get_args()
    args.task = 'go1push_mid'
    args.headless = False  # Show viewer
    args.num_envs = 1
    args.record_video = False  # Disable video recording

    # ============================================================
    # CONFIGURE AGENT TYPES HERE
    # ============================================================
    # Homogeneous examples (same robot x2 -> uses native robot class):
    #   agent_types = ['go1', 'go1']
    #   agent_types = ['cassie', 'cassie']
    #   agent_types = ['anymal_c', 'anymal_c']
    #
    # Heterogeneous examples (different robots -> uses HeteroRobot):
    #   agent_types = ['go1', 'cassie']
    #   agent_types = ['anymal_c', 'cassie']
    #   agent_types = ['go1', 'anymal_c']
    # ============================================================
    agent_types = ['go1', 'go1']

    # Detect if homogeneous or heterogeneous
    is_homogeneous = (agent_types[0] == agent_types[1])

    print("\n" + "="*60)
    if is_homogeneous:
        print(f"TESTING HOMOGENEOUS ENV: {agent_types[0]} x 2")
        print("(Uses native robot class, velocity = 1.0 m/s)")
    else:
        print(f"TESTING HETEROGENEOUS ENV: {agent_types[0]} + {agent_types[1]}")
        print("(Uses HeteroRobot class, velocity = 0.5 m/s)")
    print("="*60)

    print(f"\n[1] Creating environment...")
    print(f"    Agent 0: {agent_types[0]}")
    print(f"    Agent 1: {agent_types[1]}")

    env, env_cfg = make_hetero_env(
        env_name='go1push_mid',
        agent_types=agent_types,
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

    # Set velocity based on env type
    if is_homogeneous:
        velocity = 1.0  # Homogeneous: 1.0 m/s (faster)
    else:
        velocity = 0.5  # Heterogeneous: 0.5 m/s (slower)

    print(f"\n[3] Stepping with constant forward velocity...")
    print(f"    {'HOMOGENEOUS' if is_homogeneous else 'HETEROGENEOUS'} mode")
    print(f"    Velocity: {velocity} m/s")

    num_steps = 1000  # ~20 seconds at 50Hz

    # Constant velocity: [num_envs, num_agents, action_dim]
    # action_dim = 3 for both agents [vx, vy, vyaw]
    constant_actions = torch.zeros(env.num_envs, env.num_agents, 3, device='cuda')
    constant_actions[:, :, 0] = velocity  # Forward velocity

    print(f"    Command: vx={velocity} m/s, vy=0.0 m/s, vyaw=0.0 rad/s")
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
    print(f"Mode: {'HOMOGENEOUS' if is_homogeneous else 'HETEROGENEOUS'}")
    print(f"Agents: {agent_types}")
    print(f"Velocity: {velocity} m/s")
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
    test_env()
