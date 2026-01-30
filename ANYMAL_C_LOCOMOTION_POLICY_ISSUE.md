# Anymal C Locomotion Policy Problem - Investigation Summary

**Date:** 2026-01-30
**Issue:** Anymal C falls when given zero velocity commands
**Root Cause:** Locomotion policy outputs incorrect joint targets

---

## 🔍 Investigation Timeline

### 1. **Initial Problem: Robots Falling**
- Both Go1 and Anymal C fall to ground with zero actions
- Heights drop from ~0.42m/0.62m to ~0.20m/0.45m within 10 steps

### 2. **Config Fixes Applied**
✅ **PD Gains Pattern:** Changed from 'joint' → '_' (now matches DOF names)
✅ **PD Gains Values:** P=85→80, D=2.0 (now matches training config)
✅ **Hind Leg Angles:** Fixed to use opposite signs from front legs
✅ **HAA Offsets:** Changed from ±0.03 → 0.0 (matches training)
✅ **Spawn Height:** 0.55m → 0.62m (matches training)
✅ **Action Scale:** 0.25 → 0.5 (matches training config)

### 3. **Training Config Comparison**

Compared with `/home/gvlab/legged_gym/legged_gym/envs/anymal_c/mixed_terrains/anymal_c_rough_config.py`:

| Parameter | Training Config | Our Config (Final) | Status |
|-----------|----------------|-------------------|--------|
| LF/RF/LH/RH_HAA | 0.0 | 0.0 | ✅ Match |
| LF/RF_HFE | 0.4 | 0.4 | ✅ Match |
| LH/RH_HFE | -0.4 | -0.4 | ✅ Match |
| LF/RF_KFE | -0.8 | -0.8 | ✅ Match |
| LH/RH_KFE | 0.8 | 0.8 | ✅ Match |
| PD Stiffness | 80 | 80 | ✅ Match |
| PD Damping | 2.0 | 2.0 | ✅ Match |
| Action Scale | 0.5 | 0.5 | ✅ Match |
| Spawn Height | 0.6 | 0.62 | ✅ Close |

**All critical parameters now match the training config!**

---

## ❌ Current Problem: Wrong Policy Outputs

Even with ALL config parameters matching, the locomotion policy outputs **incorrect joint targets**:

### Test Results (Zero Actions Test):

**Input:** `[vx=0, vy=0, vyaw=0]` (stand still)

**Expected Output:** Close to default joint angles
```
[ 0.0, 0.4, -0.8, 0.0, -0.4, 0.8, 0.0, 0.4, -0.8, 0.0, -0.4, 0.8]
```

**Actual Policy Output:**
```
[-0.13, -0.19, 1.06, -1.20, 0.19, -0.86, 0.66, -0.59, 1.44, 0.13, 0.30, -1.09]
```

**Difference:** COMPLETELY WRONG! ❌

This explains why the robot falls - the policy is commanding impossible joint configurations!

---

## 🤔 Possible Root Causes

### Hypothesis 1: Policy File Mismatch
- **File:** `resources/robots/anymal_c/policy_500.jit`
- **Training:** `/home/gvlab/legged_gym/logs/flat_anymal_c_rtx2070/Jan19_18-51-31_/policy_500.jit`
- **Could be:** Wrong export, incompatible version, or for different config

### Hypothesis 2: Observation Space Mismatch
- Policy expects 48 observations
- Maybe our environment computes observations differently?
- Different observation scaling could cause wrong policy outputs

### Hypothesis 3: Actuator Network Issue
- Training uses `anydrive_v3_lstm.pt` actuator network
- File exists in `resources/actuator_nets/`
- Maybe not being loaded/used correctly?

### Hypothesis 4: Control Type Issue
- Training uses hierarchical control with actuator network
- We're using control_type='C' (command mode)
- Maybe there's a mode mismatch in how actions are processed?

---

## 🧪 Debug Output

```
[2] Default DOF positions (target when action=0):
  Anymal C: [ 0.   0.4 -0.8  0.  -0.4  0.8  0.   0.4 -0.8  0.  -0.4  0.8]

[3] Initial actual DOF positions:
  Anymal C: [0.00, 0.38, -0.60, 0.00, -0.32, 0.60, 0.00, 0.29, -0.97, 0.00, -0.49, 0.71]
  ✅ Close to defaults (randomized slightly during reset)

[5] After step - Policy commanded targets:
  Anymal C: [-0.13, -0.19, 1.06, -1.20, 0.19, -0.86, 0.66, -0.59, 1.44, 0.13, 0.30, -1.09]
  ❌ COMPLETELY WRONG!

[6] Result after 10 steps:
  Heights: Go1=0.198m, Anymal=0.475m
  ⚠️ BOTH FALLING (should be ~0.42m/0.62m)
```

---

## 📋 Next Steps to Debug

### Option 1: Test Policy Directly in legged_gym
```bash
cd /home/gvlab/legged_gym
python legged_gym/scripts/play.py --task=anymal_c_flat_rtx2070
```
This will show if the policy works in its original training environment.

### Option 2: Re-export Policy
If Option 1 works, re-export the policy to ensure correct format:
```bash
cd /home/gvlab/legged_gym
python export_anymal_policy.py
```

### Option 3: Compare Observations
Check if our environment computes observations the same way as legged_gym:
- 48 observations expected
- Check observation composition and scaling

### Option 4: Try Different Policy Checkpoint
Try earlier checkpoints (model_400.pt, model_450.pt) in case model_500.pt has issues.

### Option 5: Test Without Locomotion Policy
Temporarily use control_type='P' (direct position control) to verify PD gains work:
- This bypasses the locomotion policy
- Tests if robot can stand with default joint angles + PD control
- Isolates whether problem is physics or policy

---

## 🔗 Key Files

**Training Config:**
- `/home/gvlab/legged_gym/legged_gym/envs/anymal_c/flat/anymal_c_flat_rtx2070_config.py`
- `/home/gvlab/legged_gym/legged_gym/envs/anymal_c/mixed_terrains/anymal_c_rough_config.py`

**Policy File:**
- Our version: `/home/gvlab/new-universal-MAPush/resources/robots/anymal_c/policy_500.jit`
- Original: `/home/gvlab/legged_gym/logs/flat_anymal_c_rtx2070/Jan19_18-51-31_/policy_500.jit`

**Our Config:**
- `/home/gvlab/new-universal-MAPush/mqe/envs/anymal_c/anymal_c_config.py`

**Debug Scripts:**
- `debug_locomotion_output.py` - Shows what policy outputs
- `verify_config_loaded.py` - Confirms config values loaded correctly

---

## ✅ What's Working

1. **Config values load correctly** - Verified via debug script
2. **PD gains applied properly** - P=80, D=2.0 for all Anymal DOFs
3. **Default joint angles correct** - Match training config exactly
4. **Spawn heights correct** - 0.42m (Go1), 0.62m (Anymal)
5. **Asset loading works** - 12 DOF, 17 bodies for Anymal C

---

## ❓ Open Questions

1. **Why does the policy output wrong joint targets with zero input?**
2. **Is the policy file corrupted or incompatible?**
3. **Are observations being computed correctly?**
4. **Is the actuator network being used?**

---

**Recommendation:** Test the policy in its original legged_gym environment (Option 1) to determine if the policy itself is broken or if there's an integration issue with MAPush.
