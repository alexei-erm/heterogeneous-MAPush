# Wheeled Robot Investigation Summary

**Date:** 2026-01-19
**Investigated:** Turtlebot3 Burger, Jackal
**Outcome:** ❌ **Not viable for heterogeneous multi-agent RL**

---

## Problem Statement

Attempted to integrate wheeled robots (Turtlebot3, Jackal) as alternative agent for heterogeneous training alongside Go1. Both robots produce instant NaN values at initialization.

---

## Investigation Process

### Tests Performed

1. **Jackal (Clearpath Robotics)**
   - 2 DOF differential drive
   - Multiple URDF iterations
   - Result: NaN at step 0 ❌

2. **Turtlebot3 Burger**
   - 2 DOF differential drive
   - Progressively simplified URDF:
     - Original ROS URDF (with base_footprint)
     - Fixed version (removed base_footprint)
     - Minimal version (primitives only, no meshes)
     - Revolute joints (instead of continuous)
   - Result: NaN at step 0 for all versions ❌

3. **Control Modes Tested**
   - Velocity control (DOF_MODE_VEL) ❌
   - Position control (DOF_MODE_POS) ❌

4. **Joint Types Tested**
   - Continuous joints (type="continuous") ❌
   - Revolute joints with explicit limits ❌

5. **Spawn Heights Tested**
   - Low spawn (z=0.05m) ❌
   - Medium spawn (z=0.1m) ❌
   - High spawn (z=0.5m, same as working Go1) ❌

6. **Sanity Check: Go1**
   - Spawned at z=0.5m with floating base
   - Result: ✅ **Works perfectly** (no NaN)

### Final Test: Fixed Base

**Configuration:**
```python
asset_options.fix_base_link = True  # Fix base to world
```

**Result:** ✅ **Robot loads without NaN!**

---

## Root Cause

**Isaac Gym has a bug/limitation with floating-base wheeled robots.**

The NaN occurs during initialization of floating-base dynamics for wheeled robots. This does NOT affect:
- Legged robots (Go1, Anymal) - floating base works ✅
- Fixed-base robots - works but unusable for mobile tasks ✅

---

## Technical Details

### What Works
- Go1 (12 DOF, floating base, position control)
- Anymal (12 DOF, floating base, spawns correctly)
- Turtlebot3 (2 DOF, **FIXED** base, position control)

### What Fails
- Turtlebot3 (2 DOF, floating base) - NaN before simulation starts
- Jackal (2 DOF, floating base) - NaN before simulation starts

### Error Pattern
```
Root states shape: torch.Size([1, 13])
DOF states shape: torch.Size([2, 2])
Initial position: [nan, nan, nan]  <-- NaN BEFORE simulation
Initial rotation: [nan, nan, nan, nan]
```

NaN appears in the root state tensor **before any simulation steps**, indicating an initialization problem in Isaac Gym's floating-base dynamics.

---

## Why This Matters

For our heterogeneous multi-agent RL task:
- ❌ **Cannot use wheeled robots** (Turtlebot3, Jackal, any differential drive)
- ✅ **Can use quadrupeds** (Go1, Anymal, A1)
- ❌ **Fixed base defeats purpose** (robot can't move around environment)

---

## Possible Workarounds (All Rejected)

### 1. Use Fixed Base
- **Why rejected:** Robot can't move - defeats purpose of mobile agent

### 2. Train Locomotion Policy for Wheeled Robot
- **Why rejected:** Doesn't solve floating-base initialization bug

### 3. Use Different Physics Engine
- **Why rejected:** MAPush is deeply integrated with Isaac Gym PhysX

### 4. Report Bug to NVIDIA
- **Why rejected:** Isaac Gym is deprecated, no longer maintained

---

## Recommendations

### Option A: Homogeneous Training (2× Go1) ✅
**Pros:**
- Proven to work
- Same control stack
- Existing checkpoints/baselines

**Cons:**
- No heterogeneity (doesn't meet original goal)

### Option B: Heterogeneous Quadrupeds (Go1 + Anymal) ⚠️
**Pros:**
- Both are quadrupeds (similar but different)
- Anymal spawns without NaN
- Different morphology

**Cons:**
- **Missing Anymal locomotion policy** (critical blocker)
- Would need hierarchical control_type='C' for consistency
- No pre-trained policy available

**Implementation if chosen:**
1. Set Anymal to `control_type='P'` (direct joint control)
2. RL policy directly outputs 12D joint targets
3. Different architectures: Go1 hierarchical, Anymal direct
4. May not learn coordination well

### Option C: Homogeneous with Observation Masking (2× Go1, different obs) ✅
**Pros:**
- Both agents are Go1 (works!)
- One agent gets full observations, other gets masked/reduced
- Tests heterogeneous learning with asymmetric information

**Cons:**
- Same robot morphology
- Less dramatic than different robot types

---

## Conclusion

**Wheeled robots are not viable in Isaac Gym for floating-base multi-agent RL.**

Recommend **Option A (homogeneous 2× Go1)** or **Option C (Go1 + Go1 with observation masking)** to proceed with training.

If true heterogeneity is required, **Option B (Go1 + Anymal)** is possible but requires accepting different control modes (hierarchical vs direct).

---

## Files Created During Investigation

- `resources/robots/turtlebot3/urdf/` - Multiple URDF iterations
- `resources/robots/turtlebot3/meshes/` - Mesh files
- `test_turtlebot3.py` - Initial test script
- `diagnose_turtlebot3.py` - Diagnostic script
- `test_go1_sanity.py` - Sanity check
- `WHEELED_ROBOT_INVESTIGATION.md` - This document

---

**Status:** Investigation complete - wheeled robots ruled out
**Next Decision:** Choose from Options A, B, or C above
