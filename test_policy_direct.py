#!/usr/bin/env python3
"""
Test Anymal C policy with exact same observations as legged_gym.
This will prove if the policy itself works or if there's an observation mismatch.
"""

import torch

# Load the Anymal C policy
policy_path = "/home/gvlab/new-universal-MAPush/resources/robots/anymal_c/policy_500.jit/policy_1.pt"
print(f"Loading policy from: {policy_path}")
policy = torch.jit.load(policy_path).to('cuda')
print("✅ Policy loaded")

# Create the EXACT observation from legged_gym (from previous test output)
# This is what legged_gym gave to the policy at spawn
obs_from_legged_gym = torch.tensor([
    [-4.4687e-01,  5.1689e-02, -2.0279e-01,  1.5771e-01,  1.8341e-02,
     3.9321e-02, -3.3760e-02,  3.3483e-02, -1.0310e+00,  1.0305e+00,
    -1.0089e+00, -2.5479e-01, -7.4661e-03, -1.0264e-01, -1.2509e-01,
     1.0623e-02, -1.4739e-01,  7.1902e-02, -8.1463e-03,  2.9841e-03,
    -4.0249e-02,  7.7580e-03, -1.2454e-01, -1.8155e-01, -6.8218e-02,
     2.3572e-02,  1.8822e-02,  7.3160e-02,  1.5326e-02, -1.0023e-01,
     8.2682e-04,  6.5016e-02, -5.7836e-03,  1.3480e-02,  1.0343e-01,
     3.4537e-02,  0.0000e+00,  0.0000e+00,  0.0000e+00,  0.0000e+00,
     0.0000e+00,  0.0000e+00,  0.0000e+00,  0.0000e+00,  0.0000e+00,
     0.0000e+00,  0.0000e+00,  0.0000e+00]
], device='cuda')

print(f"\nInput observation shape: {obs_from_legged_gym.shape}")
print(f"Input observation: {obs_from_legged_gym[0]}")

# Call policy
with torch.no_grad():
    output = policy(obs_from_legged_gym)

print(f"\nPolicy output shape: {output.shape}")
print(f"Policy output: {output[0]}")

# Expected default joint angles
default_angles = torch.tensor([0.0, 0.4, -0.8, 0.0, -0.4, 0.8, 0.0, 0.4, -0.8, 0.0, -0.4, 0.8], device='cuda')
print(f"\nDefault joint angles: {default_angles}")
print(f"Difference from default: {output[0] - default_angles}")

print("\n" + "="*60)
print("Now test with OUR perfect observations (all zeros):")
print("="*60)

# Create perfect observation (all zeros except projected_gravity)
obs_perfect = torch.zeros(1, 48, device='cuda')
obs_perfect[0, 6:9] = torch.tensor([0.0, 0.0, -1.0], device='cuda')  # projected_gravity

print(f"\nInput observation: {obs_perfect[0]}")

with torch.no_grad():
    output_perfect = policy(obs_perfect)

print(f"\nPolicy output: {output_perfect[0]}")
print(f"Default joint angles: {default_angles}")
print(f"Difference from default: {output_perfect[0] - default_angles}")
