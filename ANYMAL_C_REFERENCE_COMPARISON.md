# Anymal C Implementation Comparison: IsaacGymEnvs vs MAPush

**Date:** 2026-01-30
**Purpose:** Identify differences between the reference IsaacGymEnvs Anymal implementation and our MAPush implementation

---

## 🔍 Key Findings from IsaacGymEnvs Reference

### 1. **Critical Asset Loading Setting**

**IsaacGymEnvs** (`anymal.py:192`):
```python
extremity_name = "SHANK" if asset_options.collapse_fixed_joints else "FOOT"
feet_names = [s for s in body_names if extremity_name in s]
```

**Key Insight:** When `collapse_fixed_joints=True`, the FOOT bodies get merged into SHANK!

**IsaacGymEnvs Config** (`Anymal.yaml:53`):
```yaml
urdfAsset:
  collapseFixedJoints: True  # ← This merges FOOT into SHANK
```

**Our Config** (`anymal_c_config.py:67`):
```python
collapse_fixed_joints = True  # ✅ Same
foot_name = "FOOT"            # ❌ WRONG! Should be "SHANK" when collapse_fixed_joints=True
```

---

## 🐛 CRITICAL BUG IDENTIFIED

### **The Foot Contact Bug**

**Problem:** We use `foot_name = "FOOT"` but with `collapse_fixed_joints=True`, FOOT bodies are merged into SHANK bodies!

**Result:**
- Environment looks for bodies containing "FOOT"
- No such bodies exist (they were collapsed into SHANK)
- Contact detection fails
- Feet sink through ground

**Fix:** Either:
1. **Option A:** Use `foot_name = "SHANK"` (like IsaacGymEnvs)
2. **Option B:** Set `collapse_fixed_joints = False` and keep `foot_name = "FOOT"`

**Recommendation:** Use Option A (matches reference implementation)

---

## 📊 Complete Configuration Comparison

### **Base Init State**

| Parameter | IsaacGymEnvs | Our Config | Status |
|-----------|--------------|------------|--------|
| **pos[2] (height)** | 0.62m | 0.55m | ⚠️ Different |
| **rot** | [0,0,0,1] | Same | ✅ Match |

**Analysis:**
- Reference spawns at 0.62m (higher)
- We spawn at 0.55m (lower)
- Both are reasonable, but reference is safer

**Recommendation:** Try 0.62m if still having issues

---

### **PD Control Parameters**

| Parameter | IsaacGymEnvs | Our Config (After Fix) | Status |
|-----------|--------------|------------------------|--------|
| **stiffness** | 85.0 | 85.0 | ✅ Match |
| **damping** | 2.0 | 2.0 | ✅ Match |
| **actionScale** | 0.5 | 0.25 | ⚠️ Different |

**Analysis:**
- PD gains now match perfectly ✅
- Action scale differs: 0.5 vs 0.25 (we're more conservative)

**Recommendation:** Keep 0.25 for now, increase to 0.5 if movements too slow

---

### **Default Joint Angles**

| Joint | IsaacGymEnvs | Our Config | Status |
|-------|--------------|------------|--------|
| **LF_HAA** | 0.03 | 0.0 | ⚠️ Different |
| **LH_HAA** | 0.03 | 0.0 | ⚠️ Different |
| **RF_HAA** | -0.03 | 0.0 | ⚠️ Different |
| **RH_HAA** | -0.03 | 0.0 | ⚠️ Different |
| **LF_HFE** | 0.4 | 0.4 | ✅ Match |
| **LH_HFE** | -0.4 | 0.4 | ❌ WRONG! |
| **RF_HFE** | 0.4 | 0.4 | ✅ Match |
| **RH_HFE** | -0.4 | 0.4 | ❌ WRONG! |
| **LF_KFE** | -0.8 | -0.8 | ✅ Match |
| **LH_KFE** | 0.8 | -0.8 | ❌ WRONG! |
| **RF_KFE** | -0.8 | -0.8 | ✅ Match |
| **RH_KFE** | 0.8 | -0.8 | ❌ WRONG! |

**CRITICAL BUG:** Hind leg angles are symmetric with front legs in our config!

**Reference pattern:**
- Front legs: HFE=0.4, KFE=-0.8
- Hind legs: HFE=-0.4, KFE=0.8 (opposite sign)

**Our config:** All legs use same angles (wrong!)

**Result:** Anymal tries to stand with wrong leg configuration → collapses

---

### **Asset Options**

| Parameter | IsaacGymEnvs | Our Config | Status |
|-----------|--------------|------------|--------|
| **default_dof_drive_mode** | DOF_MODE_NONE (0) | 3 (effort) | ⚠️ Different |
| **collapse_fixed_joints** | True | True | ✅ Match |
| **replace_cylinder_with_capsule** | True | True | ✅ Match |
| **flip_visual_attachments** | True | False | ⚠️ Different |
| **fix_base_link** | False | False | ✅ Match |
| **density** | 0.001 | 0.001 | ✅ Match |
| **angular_damping** | 0.0 | 0.0 | ✅ Match |
| **linear_damping** | 0.0 | 0.0 | ✅ Match |
| **armature** | 0.0 | 0.0 | ✅ Match |
| **thickness** | 0.01 | 0.01 | ✅ Match |
| **disable_gravity** | False | False | ✅ Match |

**Note:** `default_dof_drive_mode = DOF_MODE_NONE` in asset options, then set to `DOF_MODE_POS` in dof_props later

---

### **Observation Scales**

| Parameter | IsaacGymEnvs | Our Config | Status |
|-----------|--------------|------------|--------|
| **linearVelocityScale** | 2.0 | 2.0 (lin_vel) | ✅ Match |
| **angularVelocityScale** | 0.25 | 0.25 (ang_vel) | ✅ Match |
| **dofPositionScale** | 1.0 | 1.0 | ✅ Match |
| **dofVelocityScale** | 0.05 | 0.05 | ✅ Match |

---

## 🔧 Action Items (Prioritized)

### **CRITICAL (Must Fix)**

1. ✅ **PD Gains Pattern** - FIXED (changed 'joint' → '_')
2. ✅ **PD Gains Values** - FIXED (P=85, D=2.0)
3. ❌ **Foot Contact Name** - MUST FIX
   ```python
   # File: mqe/envs/anymal_c/anymal_c_config.py:62
   foot_name = "SHANK"  # Change from "FOOT" to "SHANK"
   ```

4. ❌ **Default Joint Angles (Hind Legs)** - MUST FIX
   ```python
   # File: mqe/envs/anymal_c/anymal_c_config.py:97-104
   # Left Hind leg
   'LH_HAA': 0.03,    # Add slight abduction
   'LH_HFE': -0.4,    # FLIP SIGN (currently 0.4)
   'LH_KFE': 0.8,     # FLIP SIGN (currently -0.8)

   # Right Hind leg
   'RH_HAA': -0.03,   # Add slight adduction
   'RH_HFE': -0.4,    # FLIP SIGN (currently 0.4)
   'RH_KFE': 0.8,     # FLIP SIGN (currently -0.8)
   ```

### **HIGH Priority (Recommended)**

5. ⚠️ **Spawn Height** - Consider increasing
   ```python
   # File: mqe/envs/anymal_c/anymal_c_config.py:84
   pos = [0.0, 0.0, 0.62]  # Increase from 0.55 to 0.62 (reference value)
   ```

6. ⚠️ **Front Leg HAA Joints** - Add slight abduction/adduction
   ```python
   'LF_HAA': 0.03,    # Add slight abduction (currently 0.0)
   'RF_HAA': -0.03,   # Add slight adduction (currently 0.0)
   ```

### **MEDIUM Priority (Optional)**

7. ⚠️ **Action Scale** - Consider matching reference
   ```python
   # File: mqe/envs/anymal_c/anymal_c_config.py:117
   action_scale = 0.5  # Increase from 0.25 to 0.5 if movements too slow
   ```

8. ⚠️ **flip_visual_attachments** - May need for correct rendering
   ```python
   # File: mqe/envs/anymal_c/anymal_c_config.py
   flip_visual_attachments = True  # Change from False
   ```

---

## 🧪 How IsaacGymEnvs Sets DOF Properties

**Reference code** (`anymal.py:199-203`):
```python
dof_props = self.gym.get_asset_dof_properties(anymal_asset)
for i in range(self.num_dof):
    dof_props['driveMode'][i] = gymapi.DOF_MODE_POS
    dof_props['stiffness'][i] = self.cfg["env"]["control"]["stiffness"]
    dof_props['damping'][i] = self.cfg["env"]["control"]["damping"]
```

**Then applies per actor:**
```python
self.gym.set_actor_dof_properties(env_ptr, anymal_handle, dof_props)
```

**Our approach:** We use pattern matching which works the same way, just different syntax.

---

## 🎯 Expected Leg Configuration

**Anymal C Standing Pose (Top View):**
```
     FRONT
    LF    RF
    /\    /\
   /  \  /  \
  |    ||    |
   \  /  \  /
    \/    \/
    LH    RH
     REAR

Front legs: HFE=+0.4, KFE=-0.8  (bent forward)
Hind legs:  HFE=-0.4, KFE=+0.8  (bent backward)
```

**Current wrong config:** All legs bent forward → unstable!

---

## 📝 Summary

### **Why Anymal C Was Falling:**

1. ✅ **PD gains too weak** - FIXED
2. ✅ **PD gains not applied (pattern mismatch)** - FIXED
3. ❌ **Foot contacts wrong (FOOT vs SHANK)** - NEEDS FIX
4. ❌ **Hind legs configured wrong** - NEEDS FIX
5. ⚠️ **Spawn height slightly low** - RECOMMENDED FIX

### **Confidence Level:**

With items 3 & 4 fixed, Anymal C should:
- ✅ Stand upright stably
- ✅ Detect ground contacts properly
- ✅ Not sink through ground
- ✅ Have correct leg posture

---

## 🔗 References

**IsaacGymEnvs Repository:**
- Code: `/home/gvlab/IsaacGymEnvs/isaacgymenvs/tasks/anymal.py`
- Config: `/home/gvlab/IsaacGymEnvs/isaacgymenvs/cfg/task/Anymal.yaml`
- URDF: `/home/gvlab/IsaacGymEnvs/assets/urdf/anymal_c/urdf/anymal.urdf`

**Official IsaacGymEnvs:**
- GitHub: https://github.com/isaac-sim/IsaacGymEnvs
- Anymal Task: https://github.com/isaac-sim/IsaacGymEnvs/blob/main/isaacgymenvs/tasks/anymal.py

---

**Next Step:** Apply fixes 3 & 4 (critical), then test!
