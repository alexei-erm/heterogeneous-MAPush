#!/usr/bin/env python3
"""
Test JUST standing - no movement command at all.
Feed the locomotion policies [0, 0, 0] and see if robots can stand.
"""

import sys
sys.path.append('/home/gvlab/new-universal-MAPush')

from isaacgym import gymapi
import torch
from mqe.envs.utils import make_hetero_env
from mqe.utils.helpers import get_args


def test_standing():
    print("\n" + "="*60)
    print("TESTING PURE STANDING WITH ZERO VELOCITY COMMANDS")
    print("="*60)

    args = get_args()
    args.task = 'go1push_mid'
    args.headless = False
    args.num_envs = 1
    args.record_video = False

    print("\n[1] Creating heterogeneous environment...")
    env, env_cfg = make_hetero_env(
        env_name='go1push_mid',
        agent_types=['go1', 'anymal_c'],
        args=args
    )

    print(f"✅ Environment created")

    print("\n[2] Resetting environment...")
    obs = env.reset()
    print(f"✅ Reset successful")

    print("\n[3] Stepping with CONSTANT velocity [0, 1, 0.05] (sideways + slight turn)...")
    num_steps = 200  # 4 seconds

    # Constant velocity for both agents: [vx, vy, vyaw]
    constant_actions = torch.zeros(env.num_envs, env.num_agents, 3, device='cuda')
    constant_actions[:, :, 0] = 0.0   # vx = 0 m/s
    constant_actions[:, :, 1] = 0.5   # vy = 1.0 m/s sideways
    constant_actions[:, :, 2] = 0.5  # vyaw = 0.05 rad/s turn

    reset_count = 0
    for step in range(num_steps):
        obs, rewards, dones, infos = env.step(constant_actions)

        if step < 10:
            print(f"  Step {step}: reward={rewards[0]}, done={dones[0]}")

        if dones.any():
            reset_count += 1
            if step < 50:
                print(f"  [RESET at step {step}]")
            obs = env.reset()

        if step % 50 == 0:
            print(f"  Step {step}/{num_steps} - Resets: {reset_count}")

    print("\n" + "="*60)
    print(f"STANDING TEST RESULTS: {reset_count} resets in {num_steps} steps")
    if reset_count == 0:
        print("✅ Robots can stand still!")
    elif reset_count < 3:
        print("⚠️  Mostly stable, minor issues")
    else:
        print("❌ Robots cannot stand - locomotion policy issue")
    print("="*60 + "\n")

    env.close()


if __name__ == "__main__":
    test_standing()
