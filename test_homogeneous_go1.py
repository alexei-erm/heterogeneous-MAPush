#!/usr/bin/env python3
"""
Test HOMOGENEOUS environment (2x Go1) with ZERO actions.
This uses the same method as test_zero_actions.py but with homogeneous setup.
"""

import sys
sys.path.append('/home/gvlab/new-universal-MAPush')

# Import Isaac Gym FIRST
from isaacgym import gymapi
from isaacgym import gymutil

import torch
from mqe.envs.utils import custom_cfg, make_env
from mqe.utils.helpers import get_args
from task.cuboid.config import Go1PushMidCfg

def test_homogeneous_go1():
    """Test with homogeneous Go1s and zero actions to see if robots can stand still."""

    print("\n" + "="*60)
    print("TESTING HOMOGENEOUS 2x GO1 WITH ZERO ACTIONS")
    print("="*60)

    # Get proper args with defaults
    args = get_args()
    args.task = 'go1push_mid'
    args.headless = False  # Show viewer
    args.num_envs = 1
    args.record_video = False  # Don't record video for this test

    # Create HOMOGENEOUS environment (2x Go1) - NO hetero_agent flag
    print("\n[1] Creating HOMOGENEOUS environment...")
    print("    Agent 0: Go1")
    print("    Agent 1: Go1")

    # Use custom_cfg without hetero_agent parameter
    cfg = custom_cfg(args)

    # Import the environment
    from mqe.envs.wrappers.go1_push_mid_wrapper import Go1PushMidWrapper
    from mqe.envs.npc.go1_object import Go1Object

    # Create environment using the same pattern as hetero mode
    env, env_cfg = make_env(Go1Object, cfg(Go1PushMidCfg), args)

    # Apply wrapper
    env = Go1PushMidWrapper(env)

    print(f"✅ Environment created")
    print(f"   Num envs: {env.num_envs}")
    print(f"   Num agents: {env.num_agents}")
    print(f"   Total DOFs: {env.num_actuated_dof}")

    # Reset environment
    print("\n[2] Resetting environment...")
    obs = env.reset()
    print(f"✅ Reset successful")
    print(f"   Obs shape: {obs.shape}")

    # Test with CONSTANT FORWARD velocity for both robots
    print("\n[3] Stepping with CONSTANT FORWARD velocity [1, 0, 0]...")
    print("    Both Go1 robots should walk forward")

    num_steps = 500  # ~10 seconds at 50Hz

    # Constant forward velocity: [num_envs, num_agents, action_dim]
    # action_dim = 3 for both agents [vx, vy, vyaw]
    constant_actions = torch.zeros(env.num_envs, env.num_agents, 3, device='cuda')
    constant_actions[:, :, 0] = 1.0  # vx = 1.0 m/s forward

    reset_count = 0
    step_count = 0

    for step in range(num_steps):
        obs, rewards, dones, infos = env.step(constant_actions)
        step_count += 1

        if dones.any():
            reset_count += 1
            print(f"\n⚠️  Episode reset at step {step_count}")
            print(f"    Done flags: {dones}")

            # Check termination reasons if available
            if hasattr(env, 'reset_buf'):
                print(f"    Reset buffer: {env.reset_buf}")

            # Check heights
            if hasattr(env, 'base_pos'):
                for agent_idx in range(env.num_agents):
                    agent_z = env.base_pos[agent_idx, 2].item()
                    print(f"    Agent {agent_idx} height: {agent_z:.3f}m")

            # Reset and continue
            obs = env.reset()
            step_count = 0

        # Print progress every 50 steps (~1 second)
        if step % 50 == 0:
            print(f"  Step {step}/{num_steps} - Resets: {reset_count}")

            # Print agent heights
            if hasattr(env, 'base_pos'):
                heights = []
                for agent_idx in range(env.num_agents):
                    agent_z = env.base_pos[agent_idx, 2].item()
                    heights.append(f"Agent{agent_idx}={agent_z:.3f}m")
                print(f"    Heights: {', '.join(heights)}")

    print(f"\n✅ Test completed")
    print(f"   Total resets: {reset_count}/{num_steps}")
    print(f"   Reset rate: {reset_count/num_steps*100:.1f}%")

    if reset_count == 0:
        print("\n🎉 SUCCESS! Both Go1s stood completely still!")
    elif reset_count < 5:
        print("\n✅ MOSTLY STABLE - only a few resets")
    else:
        print("\n⚠️  UNSTABLE - frequent resets indicate a problem")

    env.close()

if __name__ == "__main__":
    test_homogeneous_go1()
