"""
Debug Cassie locomotion policy integration.

Diagnoses:
1. Observation values being passed to policy
2. Action outputs from policy
3. Compare to expected ranges
"""

import sys
sys.path.append('/home/gvlab/new-universal-MAPush')

from isaacgym import gymapi
from isaacgym import gymutil

import torch
import numpy as np

def test_policy_standalone():
    """Test the Cassie policy in isolation with known good inputs."""

    print("\n" + "="*60)
    print("CASSIE POLICY STANDALONE TEST")
    print("="*60)

    # Load policy
    policy_path = '/home/gvlab/new-universal-MAPush/resources/robots/cassie/policy/policy_1.pt'
    policy = torch.jit.load(policy_path, map_location='cpu')

    # Create test observation matching legged_gym training
    # [0:3]   base_lin_vel * 2.0
    # [3:6]   base_ang_vel * 0.25
    # [6:9]   projected_gravity
    # [9:12]  commands * [2.0, 2.0, 0.25]
    # [12:24] dof_pos * 1.0
    # [24:36] dof_vel * 0.05
    # [36:48] last_actions

    obs = torch.zeros(1, 48)

    # Standing still
    obs[0, 0:3] = torch.tensor([0, 0, 0]) * 2.0  # lin_vel
    obs[0, 3:6] = torch.tensor([0, 0, 0]) * 0.25  # ang_vel
    obs[0, 6:9] = torch.tensor([0, 0, -1])  # gravity pointing down (standing upright)
    obs[0, 9:12] = torch.tensor([0.5, 0, 0]) * torch.tensor([2.0, 2.0, 0.25])  # command: forward 0.5 m/s
    obs[0, 12:24] = 0  # dof_pos at default
    obs[0, 24:36] = 0  # dof_vel = 0
    obs[0, 36:48] = 0  # last actions = 0

    print("\n[1] Input observation (standing, command forward 0.5 m/s):")
    print(f"    lin_vel (scaled):     {obs[0, 0:3].tolist()}")
    print(f"    ang_vel (scaled):     {obs[0, 3:6].tolist()}")
    print(f"    projected_gravity:    {obs[0, 6:9].tolist()}")
    print(f"    commands (scaled):    {obs[0, 9:12].tolist()}")
    print(f"    dof_pos (first 6):    {obs[0, 12:18].tolist()}")
    print(f"    dof_vel (first 6):    {obs[0, 24:30].tolist()}")

    # Run policy
    with torch.no_grad():
        actions = policy(obs)

    print(f"\n[2] Policy output (12 joint position deltas):")
    print(f"    Left leg:  {actions[0, 0:6].tolist()}")
    print(f"    Right leg: {actions[0, 6:12].tolist()}")
    print(f"    Action range: [{actions.min().item():.3f}, {actions.max().item():.3f}]")

    # Check if actions are reasonable (should be in [-1, 1] roughly)
    if actions.abs().max() > 5:
        print("\n    WARNING: Actions seem very large!")

    # Test with default joint angles from config
    default_angles = torch.tensor([
        0.1, 0., 1., -1.8, 1.57, -1.57,  # Left leg
        -0.1, 0., 1., -1.8, 1.57, -1.57   # Right leg
    ])

    # Apply action scale (0.5) and add to defaults
    action_scale = 0.5
    target_positions = actions[0] * action_scale + default_angles

    print(f"\n[3] Joint position targets (action_scale=0.5):")
    print(f"    Defaults:     {default_angles.tolist()}")
    print(f"    Targets:      {target_positions.tolist()}")
    print(f"    Delta:        {(target_positions - default_angles).tolist()}")

    # Check if deltas are reasonable
    deltas = (target_positions - default_angles).abs()
    if deltas.max() > 1.0:
        print(f"\n    WARNING: Large joint deltas detected! Max: {deltas.max():.3f} rad")

    return True


def test_env_observations():
    """Test what observations the env is actually generating."""

    print("\n" + "="*60)
    print("TESTING ENV OBSERVATION GENERATION")
    print("="*60)

    from mqe.envs.utils import make_hetero_env
    from mqe.utils.helpers import get_args

    args = get_args()
    args.task = 'go1push_mid'
    args.headless = True
    args.num_envs = 1

    # Create env with just Cassie
    print("\n[1] Creating Cassie-only environment...")

    env, env_cfg = make_hetero_env(
        env_name='go1push_mid',
        agent_types=['cassie', 'cassie'],  # Two Cassies
        args=args
    )

    print(f"    Num envs: {env.num_envs}")
    print(f"    Num agents: {env.num_agents}")

    # Reset and get observations
    obs = env.reset()

    print(f"\n[2] Environment observation buffer:")
    print(f"    obs_buf type: {type(env.obs_buf)}")

    # Check if Cassie locomotion obs is being populated
    if hasattr(env, 'locomotion_obs'):
        loc_obs = env.locomotion_obs
        print(f"\n[3] Locomotion observation buffer:")
        print(f"    Shape: {loc_obs.shape}")
        print(f"    lin_vel [0:3]:        {loc_obs[0, 0:3].tolist()}")
        print(f"    ang_vel [3:6]:        {loc_obs[0, 3:6].tolist()}")
        print(f"    gravity [6:9]:        {loc_obs[0, 6:9].tolist()}")
        print(f"    commands [9:12]:      {loc_obs[0, 9:12].tolist()}")
        print(f"    dof_pos [12:18]:      {loc_obs[0, 12:18].tolist()}")
        print(f"    dof_vel [24:30]:      {loc_obs[0, 24:30].tolist()}")
        print(f"    last_action [36:42]:  {loc_obs[0, 36:42].tolist()}")
    else:
        print("\n    ERROR: No locomotion_obs buffer found!")

    # Step with constant forward velocity
    print(f"\n[4] Stepping with forward velocity command...")
    constant_actions = torch.zeros(env.num_envs, env.num_agents, 3, device='cuda')
    constant_actions[:, :, 0] = 0.5  # Forward

    for i in range(5):
        obs, rewards, dones, infos = env.step(constant_actions)

        if hasattr(env, 'locomotion_obs'):
            loc_obs = env.locomotion_obs
            print(f"\n    Step {i+1}:")
            print(f"      commands [9:12]: {loc_obs[0, 9:12].tolist()}")
            print(f"      gravity [6:9]:   {loc_obs[0, 6:9].tolist()}")

            if hasattr(env, 'last_locomotion_action'):
                print(f"      last_action[0:6]: {env.last_locomotion_action[0, 0:6].tolist()}")

        if dones.any():
            print(f"      RESET triggered!")
            break

    env.close()


if __name__ == "__main__":
    # Test 1: Policy in isolation
    test_policy_standalone()

    # Test 2: Environment observations
    print("\n" + "="*60)
    test_env_observations()
