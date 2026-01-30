#!/usr/bin/env python3
"""
Test Anymal C policy carefully with different observation inputs.
Figure out what observations make it output reasonable joint angles.
"""

import torch

# Load the policy
policy_path = "/home/gvlab/new-universal-MAPush/resources/robots/anymal_c/policy_500.jit/policy_1.pt"
policy = torch.jit.load(policy_path).to('cuda')
policy.eval()

print("="*60)
print("TESTING ANYMAL C POLICY WITH VARIOUS OBSERVATIONS")
print("="*60)

# Default joint angles for Anymal C
# Order: LF_HAA, LF_HFE, LF_KFE, RF_HAA, RF_HFE, RF_KFE, LH_HAA, LH_HFE, LH_KFE, RH_HAA, RH_HFE, RH_KFE
default_angles = torch.tensor([0.0, 0.4, -0.8, 0.0, -0.4, 0.8, 0.0, 0.4, -0.8, 0.0, -0.4, 0.8], device='cuda')
action_scale = 0.5

# Observation structure (48 dims):
# [0:3]   base_lin_vel * 2.0
# [3:6]   base_ang_vel * 0.25
# [6:9]   projected_gravity (normalized, z should be -1 when upright)
# [9:12]  commands * [2.0, 2.0, 0.25] - [vx, vy, vyaw]
# [12:24] (dof_pos - default) * 1.0
# [24:36] dof_vel * 0.05
# [36:48] last_actions (previous policy outputs)

def test_policy(obs, name):
    """Test policy with given observation and print results."""
    with torch.no_grad():
        output = policy(obs.unsqueeze(0))
    output = output[0]

    # Compute actual joint targets
    targets = output * action_scale + default_angles

    print(f"\n{name}:")
    print(f"  Policy raw output: {output}")
    print(f"  Joint targets: {targets}")
    print(f"  Target - Default: {targets - default_angles}")
    max_diff = (targets - default_angles).abs().max().item()
    print(f"  Max deviation from default: {max_diff:.3f} rad ({max_diff * 57.3:.1f} deg)")

    return output, targets

# Test 1: Perfect standing (what we're sending)
print("\n" + "-"*60)
obs1 = torch.zeros(48, device='cuda')
obs1[6] = 0.0   # gravity x
obs1[7] = 0.0   # gravity y
obs1[8] = -1.0  # gravity z (pointing down when upright)
# Commands are zero
# DOF positions are zero (at default)
# DOF velocities are zero
# Last actions are zero
test_policy(obs1, "Test 1: Perfect standing (all zeros except gravity)")

# Test 2: With small random noise on dof_pos (like domain randomization)
print("\n" + "-"*60)
obs2 = obs1.clone()
obs2[12:24] = torch.randn(12, device='cuda') * 0.1  # Small random DOF errors
test_policy(obs2, "Test 2: Small random DOF position noise")

# Test 3: With typical velocities (robot slightly moving)
print("\n" + "-"*60)
obs3 = obs1.clone()
obs3[0:3] = torch.tensor([0.1, 0.0, 0.0], device='cuda')  # Small forward velocity
obs3[24:36] = torch.randn(12, device='cuda') * 0.1  # Small random DOF velocities
test_policy(obs3, "Test 3: Small linear velocity + DOF velocity noise")

# Test 4: With nonzero last_actions (what legged_gym does)
print("\n" + "-"*60)
obs4 = obs1.clone()
obs4[36:48] = default_angles.clone()  # Previous actions were at default
# Wait, last_actions should be policy OUTPUTS not targets!
# Let's try with zeros to start

# Test 5: Typical training observation (from domain randomization)
print("\n" + "-"*60)
obs5 = torch.zeros(48, device='cuda')
obs5[0:3] = torch.randn(3, device='cuda') * 0.2  # Random linear velocity
obs5[3:6] = torch.randn(3, device='cuda') * 0.1  # Random angular velocity
obs5[6:9] = torch.tensor([0.05, 0.0, -1.0], device='cuda')  # Slight tilt
obs5[9:12] = torch.tensor([0.5, 0.0, 0.0], device='cuda')  # Forward command (scaled)
obs5[12:24] = torch.randn(12, device='cuda') * 0.2  # Random DOF errors
obs5[24:36] = torch.randn(12, device='cuda') * 0.5  # Random DOF velocities
obs5[36:48] = torch.randn(12, device='cuda') * 0.3  # Random last actions
test_policy(obs5, "Test 5: Typical training observation (domain randomization)")

# Test 6: Non-zero command (walking forward)
print("\n" + "-"*60)
obs6 = obs1.clone()
obs6[9] = 1.0   # Forward velocity command (scaled: 0.5 m/s * 2.0 = 1.0)
obs6[10] = 0.0  # No sideways
obs6[11] = 0.0  # No turning
test_policy(obs6, "Test 6: Forward velocity command (0.5 m/s)")

print("\n" + "="*60)
print("CONCLUSION:")
print("If the policy outputs reasonable values only with noisy observations,")
print("it was trained with domain randomization and expects noise!")
print("="*60)
