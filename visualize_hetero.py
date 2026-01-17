#!/usr/bin/env python3
"""
Visualize heterogeneous Go1 + Jackal setup to debug physics issues.
Run with: python visualize_hetero.py
"""

from mqe.envs.utils import make_hetero_env
from mqe.utils.helpers import get_args
import torch  # Import torch AFTER isaacgym modules
import sys

def visualize_hetero():
    """Run heterogeneous environment with viewer to inspect physics behavior."""

    print("="*80)
    print("HETEROGENEOUS ENVIRONMENT VISUALIZATION")
    print("="*80)
    print("Creating Go1 + Jackal environment with viewer enabled...")
    print("Watch for:")
    print("  1. Jackal falling through ground (wrong init height)")
    print("  2. Jackal flipping over (unstable)")
    print("  3. Agents not moving (control issues)")
    print("  4. NaN positions (physics breakdown)")
    print("="*80)

    # Get proper args with viewer enabled
    sys.argv = [
        'visualize_hetero.py',
        '--task=go1push_mid',
        '--num_envs=1',  # Single environment for easier viewing
        '--seed=1',
    ]
    args = get_args()
    args.record_video = False
    args.headless = False  # Enable viewer

    # Create environment
    env, env_cfg = make_hetero_env(
        env_name='go1push_mid',
        agent_types=['go1', 'jackal'],
        args=args
    )

    print("\n[Viewer] Environment created. Viewer should open.")
    print("[Viewer] Press 'V' to toggle viewer camera")
    print("[Viewer] Use mouse to rotate/pan camera")
    print("[Viewer] Running 1000 steps with random actions...")
    print("\nObserve the robots and watch for physics anomalies!\n")

    # Reset
    obs = env.reset()

    # Run for some steps with random actions
    for step in range(1000):
        # Random actions for both agents
        actions = torch.rand(1, 2, 3, device='cuda') * 2 - 1  # [-1, 1]

        # Step environment
        obs, reward, done, info = env.step(actions)

        # Check for NaN
        if torch.isnan(obs).any():
            print(f"[Step {step}] NaN detected in observations!")
        if torch.isnan(reward).any():
            print(f"[Step {step}] NaN detected in rewards!")

        # Print every 100 steps
        if step % 100 == 0:
            print(f"[Step {step}] Reward: {reward.mean():.4f}")

            # Check robot states
            base_pos = env.env.obs_buf.base_pos.reshape(1, 2, -1)
            go1_height = base_pos[0, 0, 2].item()
            jackal_height = base_pos[0, 1, 2].item()

            print(f"  Go1 height: {go1_height:.3f}m (should be ~0.3-0.4m)")
            print(f"  Jackal height: {jackal_height:.3f}m (should be ~0.10m)")

            # Check for physics breakdown
            if go1_height < 0.1 or jackal_height < 0.0:
                print("  ⚠️  WARNING: Robot(s) too low - may have fallen through ground!")
            if go1_height > 1.0 or jackal_height > 0.5:
                print("  ⚠️  WARNING: Robot(s) too high - physics may be unstable!")

        if done.any():
            print(f"[Step {step}] Episode done, resetting...")
            obs = env.reset()

    print("\n" + "="*80)
    print("Visualization complete!")
    print("="*80)

if __name__ == "__main__":
    visualize_hetero()
