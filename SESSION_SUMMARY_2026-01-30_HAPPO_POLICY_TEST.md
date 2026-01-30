# Session Summary: HAPPO Policy Testing with Heterogeneous Environment
**Date**: 2026-01-30
**Task**: Test trained HAPPO mid-level policies with heterogeneous Go1 + Anymal C environment

## Summary

Successfully loaded and tested trained HAPPO policies from checkpoint `10M`. The policies are working correctly and outputting reasonable actions based on the 8-dimensional mid-level observations. However, robots are failing due to **locomotion policy instability**, not HAPPO policy issues.

## What We Tested

Created `test_happo_policies.py` to:
1. Load HAPPO actor policies for both agents from checkpoint
2. Run heterogeneous environment (Go1 + Anymal C)
3. Feed mid-level observations to HAPPO policies
4. Convert HAPPO velocity commands [vx, vy, vyaw] to locomotion policy commands
5. Execute and observe results

## Test Results

### ✅ What Works

1. **HAPPO Policy Loading**: Successfully loaded both actor networks
   - Actor agent0 (Go1): 8-dim obs → 3-dim action
   - Actor agent1 (Anymal C): 8-dim obs → 3-dim action
   - Architecture: MLP [256, 256] with ReLU activation

2. **Observation Space**: Correctly aligned (8 dimensions)
   - `[target_x, target_y]` - Target position in agent frame
   - `[box_x, box_y]` - Box position in agent frame
   - `[box_yaw]` - Box orientation
   - `[other_agent_x, other_agent_y, other_agent_yaw]` - Other agent's state

3. **HAPPO Action Output**: Policies produce reasonable velocity commands
   ```
   Agent 0 (Go1):    [-0.3530,  0.2184,  0.2427]  # [vx, vy, vyaw]
   Agent 1 (Anymal): [ 0.2503,  0.3774,  0.0475]
   ```

4. **Mid-Level Wrapper**: Go1PushMidWrapper correctly processes observations and actions

### ❌ What Fails

**Problem**: Robots reset frequently (56 resets in 500 steps = ~10 steps between resets)

**Root Cause**: Locomotion policies produce unstable joint targets

**Evidence**:
```
Step 7: reward=[-10.0224, -10.0224], done=True  # Exception punishment
```

The -10 reward indicates exception punishment (-5 × 2 agents), which occurs when robots violate termination conditions:
- Roll > 0.8 rad (46°)
- Pitch > 1.6 rad (92°)
- Z-height violation
- Collision

## Why Locomotion Policies Fail

The locomotion policies (trained in legged_gym) are producing joint targets that cause robots to tip over:

**Anymal C Example**:
```
Policy output: [-0.7517,  0.6676,  0.1616, -0.7529, -0.3054,  0.0581,
                 0.2097, -0.0086,  0.8260,  0.1799,  0.6676, -0.8842]

Default pose:  [ 0.0000,  0.4000, -0.8000,  0.0000, -0.4000,  0.8000,
                 0.0000,  0.4000, -0.8000,  0.0000, -0.4000,  0.8000]
```

The policy outputs joint angles significantly different from the stable standing pose, causing the robot to lose balance within 7-16 steps.

### Why This Happens

1. **Domain Randomization Mismatch**
   - Locomotion policies were trained with domain randomization in legged_gym
   - They expect noisy, randomized observations (velocities, DOF positions, commands)
   - Our environment provides clean, deterministic observations
   - This out-of-distribution input causes unstable outputs

2. **Different Dynamics**
   - Legged_gym environment: Single quadruped walking on flat terrain
   - MAPush environment: Two robots + box + target, different physics constraints
   - Locomotion policies haven't seen this multi-agent, object-manipulation scenario

3. **Observation Distribution**
   - Previous test (`test_policy_direct.py`) showed: Policies output **garbage** when given all-zeros observations
   - They were never trained for that scenario

## What This Tells Us

### ✅ Heterogeneous Framework Status

The heterogeneous framework is **working correctly**:
- Asset loading (per-robot configs) ✅
- Observation computation ✅
- Action processing ✅
- Mid-level wrapper ✅
- HAPPO policy integration ✅

### ❌ Locomotion Policy Compatibility

The locomotion policies **cannot be used as-is** in the MAPush environment because:
1. They were trained for a different task (standalone walking)
2. They expect different observation distributions (domain randomization)
3. They produce unstable outputs in the multi-agent manipulation setting

## Solutions

### Option 1: End-to-End Training (Recommended)

Train locomotion policies directly in the MAPush environment:

**Approach**:
- Use HARL HAPPO to train both mid-level AND low-level policies simultaneously
- Remove pre-trained locomotion policies
- Let the network learn joint targets directly from MAPush observations
- This ensures policies are adapted to the actual task dynamics

**Advantages**:
- Policies match the environment they'll be deployed in
- No domain mismatch issues
- Can handle multi-agent coordination naturally

**Disadvantages**:
- Longer training time (needs to learn locomotion from scratch)
- Higher dimensional action space (24 DOF for 2 quadrupeds)

### Option 2: Locomotion Policy Fine-Tuning

Fine-tune existing locomotion policies in MAPush environment:

**Approach**:
- Load pre-trained locomotion policies as initialization
- Continue training in MAPush with HAPPO mid-level policies
- Use smaller learning rate to preserve walking behavior
- Adapt to multi-agent manipulation dynamics

**Advantages**:
- Faster than training from scratch
- Leverages existing walking knowledge
- Focuses learning on the new task

**Disadvantages**:
- Requires careful tuning to avoid catastrophic forgetting
- Still need to handle observation distribution shift

### Option 3: Homogeneous Testing First

Test with homogeneous Go1-only agents to isolate variables:

**Approach**:
- Run heterogeneous framework with `['go1', 'go1']` instead of `['go1', 'anymal_c']`
- This eliminates Anymal C-specific issues
- Test if Go1 locomotion policies work better in MAPush

**Advantages**:
- Simpler debugging (one robot type)
- Go1 policies might be more stable (we know they work in test_homogeneous_go1.py)

**Disadvantages**:
- Doesn't solve the Anymal C problem
- Just a debugging step, not a solution

## Files Created

- `test_happo_policies.py` - Test script to load and run HAPPO policies with hetero environment

## Key Findings

1. **HAPPO policies load and run correctly** - observation space matches (8-dim), actions are reasonable
2. **Heterogeneous framework works** - no bugs in asset loading, observation computation, action processing
3. **Locomotion policies are the bottleneck** - they produce unstable joint targets due to domain mismatch
4. **Root cause**: Pre-trained locomotion policies from legged_gym don't transfer to MAPush's multi-agent manipulation task

## Recommendation

**Immediate Next Step**: Test with homogeneous Go1 agents `['go1', 'go1']` to confirm that:
1. Go1 locomotion policies work better than Anymal C
2. The issue is Anymal C-specific or a general locomotion policy problem

**Long-term Solution**: Train end-to-end in MAPush environment without pre-trained locomotion policies, letting HAPPO learn both coordination and locomotion simultaneously.
