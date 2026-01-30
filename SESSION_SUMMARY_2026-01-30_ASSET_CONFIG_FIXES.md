# Session Summary: Anymal C Asset Configuration Fixes
**Date:** 2026-01-30
**Focus:** Fixing Anymal C visual deformation by matching legged_gym asset configuration exactly

---

## Session Context

Continued from `SESSION_SUMMARY_2026-01-30_ANYMAL_PD_GAINS_FIX.md` where we had fixed:
- PD gains (P=80, D=2.0)
- Hind leg joint angles (HFE=-0.4, KFE=0.8)
- Action scale (0.5)
- Spawn height (0.62m)
- Observation structure (changed to use obs_buf like Go1)

**New Problem:** After replacing URDF with IsaacGymEnvs reference, Anymal C appeared "completely fucked up and deform" visually.

---

## Root Cause Analysis

The issue was NOT the URDF itself, but **asset loading configuration mismatches** between our code and legged_gym's training environment.

### Investigation Method

Compared three configuration sources:
1. **Training config:** `/home/gvlab/legged_gym/legged_gym/envs/anymal_c/mixed_terrains/anymal_c_rough_config.py`
2. **Parent class defaults:** `/home/gvlab/legged_gym/legged_gym/envs/base/legged_robot_config.py` (lines 98-118)
3. **Our config:** `mqe/envs/anymal_c/anymal_c_config.py`

---

## Bugs Found and Fixed

### Bug #1: Self-Collisions Enabled (Should Be Disabled)
**Location:** `mqe/envs/anymal_c/anymal_c_config.py:70`

**Problem:**
```python
self_collisions = 0  # 0 = enabled (robot parts can collide with each other)
```

**Fix:**
```python
self_collisions = 1  # 1 = disabled - MUST match legged_gym (disabled)
```

**Impact:** Enabled self-collisions can cause physics instability and visual artifacts when robot parts interfere with each other.

---

### Bug #2: Visual Attachments Not Flipped
**Location:** `mqe/envs/anymal_c/anymal_c_config.py:73`

**Problem:**
```python
flip_visual_attachments = False
```

**Fix:**
```python
flip_visual_attachments = True  # MUST match legged_gym parent class default
```

**Impact:** Anymal C's mesh files (.obj) are y-up and need to be flipped to z-up for Isaac Gym. This is the **primary cause of visual deformation**.

**Evidence from legged_gym:**
- Parent class `LeggedRobotCfg.asset` sets `flip_visual_attachments = True` (line 110)
- Anymal C config inherits this and doesn't override it
- Therefore, training used `flip_visual_attachments = True`

---

### Bug #3: Wrong Contact Penalty Bodies
**Location:** `mqe/envs/anymal_c/anymal_c_config.py:63`

**Problem:**
```python
penalize_contacts_on = ["base", "THIGH"]
```

**Fix:**
```python
penalize_contacts_on = ["SHANK", "THIGH"]  # MUST match legged_gym training config
```

**Impact:** Minor - affects reward calculation but not visual appearance.

**Evidence from legged_gym:**
- Training config explicitly sets: `penalize_contacts_on = ["SHANK", "THIGH"]` (line 75)

---

## All Asset Configuration Parameters (Now Verified)

Comparison with legged_gym training config:

| Parameter | Legged_gym Training | Our Config (After Fix) | Status |
|-----------|---------------------|------------------------|--------|
| **file** | `anymal_c.urdf` | `anymal_c.urdf` | ✅ Match |
| **name** | `"anymal_c"` | `"anymal_c"` | ✅ Match |
| **foot_name** | `"FOOT"` | `"FOOT"` | ✅ Match |
| **penalize_contacts_on** | `["SHANK", "THIGH"]` | `["SHANK", "THIGH"]` | ✅ **FIXED** |
| **terminate_after_contacts_on** | `["base"]` | `["base"]` | ✅ Match |
| **self_collisions** | `1` (disabled) | `1` (disabled) | ✅ **FIXED** |
| **collapse_fixed_joints** | `True` | `True` | ✅ Match |
| **fix_base_link** | `False` | `False` | ✅ Match |
| **default_dof_drive_mode** | `3` (effort) | `3` (effort) | ✅ Match |
| **replace_cylinder_with_capsule** | `True` | `True` | ✅ Match |
| **flip_visual_attachments** | `True` | `True` | ✅ **FIXED** |
| **disable_gravity** | `False` | `False` | ✅ Match |
| **density** | `0.001` | `0.001` | ✅ Match |
| **angular_damping** | `0.` | `0.` | ✅ Match |
| **linear_damping** | `0.` | `0.` | ✅ Match |
| **max_angular_velocity** | `1000.` | `1000.` | ✅ Match |
| **max_linear_velocity** | `1000.` | `1000.` | ✅ Match |
| **armature** | `0.` | `0.` | ✅ Match |
| **thickness** | `0.01` | `0.01` | ✅ Match |

---

## Changes Made This Session

### File: `mqe/envs/anymal_c/anymal_c_config.py`

**Lines 63, 70, 73:**
```python
# Before:
penalize_contacts_on = ["base", "THIGH"]
self_collisions = 0  # enabled
flip_visual_attachments = False

# After:
penalize_contacts_on = ["SHANK", "THIGH"]  # MUST match legged_gym training config
self_collisions = 1  # 1 to disable, 0 to enable - MUST match legged_gym (disabled)
flip_visual_attachments = True  # MUST match legged_gym parent class default (True)
```

**Line 84 (spawn height reset to normal):**
```python
# Before:
pos = [0.0, 0.0, 1] # testing height

# After:
pos = [0.0, 0.0, 0.62] # x,y,z [m] - Matches legged_gym training config (0.6m)
```

---

## Current Status

### ✅ Verified Fixes (All Sessions Combined)

**Configuration Parameters:**
- ✅ Spawn height: 0.62m
- ✅ Default joint angles: All 12 joints correct (including hind leg fix)
- ✅ PD gains: P=80, D=2.0 with pattern '_'
- ✅ Action scale: 0.5
- ✅ Foot name: "FOOT"
- ✅ Self-collisions: Disabled (1)
- ✅ Flip visual attachments: True
- ✅ Penalize contacts: ["SHANK", "THIGH"]

**Code Fixes:**
- ✅ Observation structure: Changed to use obs_buf (like Go1)
- ✅ URDF: Using IsaacGymEnvs reference (1600 lines)

### ❌ Known Remaining Issues

1. **CRITICAL: Visual appearance still wrong** (user reported after config fixes)
   - Possible causes:
     - URDF mesh references incorrect
     - Mesh files corrupted or wrong version
     - Asset loading code has additional issues

2. **Anymal C still falling** (observation fix not yet tested)
   - Last fix (obs_buf) may resolve this
   - Needs user testing with `test_zero_actions.py`

---

## Key Learning: Configuration Inheritance

**Important discovery about legged_gym configs:**

Anymal C training config (`anymal_c_rough_config.py`) inherits from `LeggedRobotCfg`:
```python
class AnymalCRoughCfg(LeggedRobotCfg):
    class asset(LeggedRobotCfg.asset):
        # Only overrides specific parameters
        file = "..."
        name = "anymal_c"
        foot_name = "FOOT"
        penalize_contacts_on = ["SHANK", "THIGH"]
        # All other parameters inherited from parent!
```

**This means:**
- Parameters NOT explicitly overridden come from parent class
- We must check parent class (`legged_robot_config.py:98-118`) for defaults
- Our config must match BOTH explicit overrides AND inherited defaults

---

## Testing Instructions

**Test Scripts (ONLY use these 2):**
1. `test_homogeneous_go1.py` - Tests 2x Go1 homogeneous mode
2. `test_zero_actions.py` - Tests Go1 + Anymal C heterogeneous mode

**Next Test (User to perform):**
```bash
python test_zero_actions.py
```

**What to check:**
1. Visual appearance: Does Anymal C look correct? (legs connected, not deformed)
2. Physics: Does Anymal C stand or fall?
3. Stability: Are there frequent episode resets?

---

## Files Modified

- `mqe/envs/anymal_c/anymal_c_config.py` - Fixed 3 asset parameters

---

## Next Steps (When Resumed)

### If Visual Issues Persist:

1. **Check URDF mesh references** - Verify mesh file paths in URDF
2. **Compare mesh files** - Diff our meshes vs legged_gym meshes
3. **Check asset loading code** - Verify `hetero_robot.py` lines 195-224 applies config correctly
4. **Consider reverting URDF** - Test if original URDF + correct config works better

### If Anymal C Still Falls:

1. **Debug observation values** - Use existing `dump_anymal_policy_input.py`
2. **Compare with legged_gym observations** - Modify legged_gym play.py to print obs
3. **Check command scaling** - Commands in obs[9:12] might need scaling
4. **Verify obs_buf indexing** - Ensure agent_env_indices extracts Anymal data correctly

---

## References

- Training config: `/home/gvlab/legged_gym/legged_gym/envs/anymal_c/mixed_terrains/anymal_c_rough_config.py`
- Parent class: `/home/gvlab/legged_gym/legged_gym/envs/base/legged_robot_config.py`
- Our config: `mqe/envs/anymal_c/anymal_c_config.py`
- URDF: `resources/robots/anymal_c/urdf/anymal_c.urdf` (1600 lines, from IsaacGymEnvs)
- Debug status: `ANYMAL_C_DEBUG_STATUS.md`
