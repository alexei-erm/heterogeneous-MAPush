# Anymal C Debug Status - Comprehensive Summary

**Date:** 2026-01-30
**Current Status:** Anymal C falls in hetero mode, but Go1 works perfectly

---

## ✅ VERIFIED - What Works Correctly

### 1. **Homogeneous 2x Go1 Mode**
- ✅ Both Go1 robots spawn correctly at 0.45m
- ✅ Both stand perfectly still with zero actions
- ✅ No falling, no resets
- ✅ Heights remain stable at ~0.285m (crouched but stable)
- **Test:** `test_homogeneous_go1.py` - 0 resets in 500 steps

### 2. **Go1 in Heterogeneous Mode**
- ✅ Go1 spawns correctly
- ✅ Go1 locomotion policy works
- ✅ No issues observed with Go1 agent

### 3. **Anymal C Configuration Matches Training**
All parameters verified to match `/home/gvlab/legged_gym/legged_gym/envs/anymal_c/mixed_terrains/anymal_c_rough_config.py`:

| Parameter | Training Config | Our Config | Status |
|-----------|----------------|------------|--------|
| **Spawn height** | 0.6m | 0.62m | ✅ Close enough |
| **Default joint angles** | | | |
| - HAA joints | 0.0 | 0.0 | ✅ Match |
| - Front HFE | 0.4 | 0.4 | ✅ Match |
| - Hind HFE | -0.4 | -0.4 | ✅ Match |
| - Front KFE | -0.8 | -0.8 | ✅ Match |
| - Hind KFE | 0.8 | 0.8 | ✅ Match |
| **PD Gains** | | | |
| - Stiffness | 80.0 | 80.0 | ✅ Match |
| - Damping | 2.0 | 2.0 | ✅ Match |
| - Pattern | 'HAA'/'HFE'/'KFE' | '_' (matches all) | ✅ Works |
| **Action scale** | 0.5 | 0.5 | ✅ Match |
| **Foot name** | "FOOT" | "FOOT" | ✅ Match |

### 4. **Anymal C Policy Works in Original Environment**
- ✅ Tested `python legged_gym/scripts/play.py --task=anymal_c_flat_rtx2070`
- ✅ **Policy walks normally in legged_gym!**
- ✅ Policy file (`policy_500.jit`) is NOT corrupted

### 5. **Asset Loading**
- ✅ Anymal C URDF loads correctly
- ✅ 12 DOF detected correctly
- ✅ 17 bodies detected correctly
- ✅ Body names correct: ['base', 'LF_HIP', 'LF_THIGH', 'LF_SHANK', 'LF_FOOT', ...]
- ✅ Both SHANK and FOOT bodies exist (collapse_fixed_joints didn't merge them in our URDF)

### 6. **Buffer Initialization**
- ✅ `default_dof_pos` correctly initialized with Anymal C angles
- ✅ PD gains correctly set for DOFs 12-23
- ✅ Torque limits correctly set to 80 N·m for Anymal C

---

## ❌ VERIFIED - What Has Been Ruled Out

### 1. **NOT a Physics/PD Gains Problem**
- ❌ PD gains are correct (P=80, D=2.0)
- ❌ PD gains ARE being applied (verified with debug prints)
- ❌ Not a general physics issue (Go1 works fine)

### 2. **NOT a Config File Problem**
- ❌ All config parameters match training
- ❌ Spawn height is correct
- ❌ Joint angles are correct
- ❌ Action scale is correct

### 3. **NOT a Policy File Problem**
- ❌ Policy works perfectly in legged_gym
- ❌ Policy file is not corrupted

### 4. **NOT an Asset/URDF Problem**
- ❌ URDF loads correctly
- ❌ DOF count is correct
- ❌ Body structure is correct

### 5. **NOT a Homogeneous vs Heterogeneous Framework Issue**
- ❌ Go1 works in both homo and hetero modes
- ❌ Hetero framework is sound

---

## 🔍 POSSIBLE REMAINING ISSUES

### **Issue #1: Observation Structure Mismatch**
**Status:** Most likely root cause

**Evidence:**
- Policy works in legged_gym with same config
- Policy fails in our environment
- **Difference must be in observations fed to policy**

**Latest Attempt (Just Made):**
- Changed Anymal C to use `obs_buf` (like Go1) instead of raw state tensors
- Code now at `mqe/envs/base/hetero_robot.py:1028-1056`
- **NOT YET TESTED if this fixes the issue**

**Specific Concerns:**
```python
# Current implementation uses obs_buf (like Go1):
loc_obs[:, 0:3] = self.obs_buf.lin_vel[agent_env_indices]   # Already scaled
loc_obs[:, 3:6] = self.obs_buf.ang_vel[agent_env_indices]   # Already scaled
loc_obs[:, 6:9] = self.obs_buf.projected_gravity[agent_env_indices]
loc_obs[:, 9:12] = agent_actions  # Commands
loc_obs[:, 12:24] = self.obs_buf.dof_pos[agent_env_indices]  # Already scaled
loc_obs[:, 24:36] = self.obs_buf.dof_vel[agent_env_indices]  # Already scaled
loc_obs[:, 36:48] = last_joint_targets  # Previous actions
```

**Potential Issues:**
1. **Double scaling?** - `obs_buf` values might already be scaled, but we're not checking if legged_gym expects unscaled values
2. **`obs_buf` indexing** - `agent_env_indices` might not map correctly to Anymal C's data
3. **Commands not scaled?** - Line with `agent_actions` might need scaling
4. **Previous actions format** - Should be joint targets, not velocity commands

### **Issue #2: Per-Agent Observation Indexing**
**Status:** Partially addressed, needs verification

**Problem:**
- Heterogeneous setup has shape: `[num_envs * num_agents, ...]` for states
- Anymal C is agent_idx=1
- `agent_env_indices = torch.arange(num_envs) * num_agents + 1`
- For 1 env, 2 agents: `agent_env_indices = [1]`

**Questions:**
- Does `obs_buf.dof_pos[agent_env_indices]` return Anymal C DOFs or Go1 DOFs?
- Is `obs_buf` structured per-agent or per-DOF?

### **Issue #3: Observation Buffer Structure**
**Status:** Unknown

**Questions:**
- What is `obs_buf`? Is it a SimpleNamespace? A tensor?
- How is `obs_buf.dof_pos` populated for heterogeneous agents?
- Does it account for different DOF counts per agent?

**Location to check:**
- `mqe/envs/base/hetero_robot.py:732-770` - Where `obs_buf` is populated

### **Issue #4: Commands/Actions Confusion**
**Status:** Needs clarification

**In our code:**
- `agent_actions` = `[vx, vy, vyaw]` (high-level velocity commands)
- These go into observation slot [9:12]

**In legged_gym:**
- Observation [9:12] expects `commands * commands_scale`
- Commands might need scaling: `[vx * lin_vel_scale, vy * lin_vel_scale, vyaw * ang_vel_scale]`

**Currently:** We pass unscaled commands!

### **Issue #5: Last Actions vs Last Joint Targets**
**Status:** Implementation unclear

**In legged_gym observation [36:48]:**
- Expects last **actions** (the raw policy outputs from previous step)
- NOT velocity commands, NOT scaled values

**Our implementation:**
- Stores `joint_positions.clone()` as `last_joint_targets`
- Are these the raw policy outputs or processed values?

---

## 🧪 NEXT DEBUGGING STEPS

### Priority 1: Verify Observation Values
**Action:** Print actual observation values being fed to Anymal C policy

**Script:** `dump_anymal_policy_input.py` (already created)

**Check:**
1. Are velocities reasonable? (not 0.9 m/s when standing still)
2. Are DOF positions correct? (close to default angles)
3. Are DOF errors correct? (dof_pos - default should be small)
4. Are commands zero?

### Priority 2: Compare with Legged Gym Observations
**Action:** Run Anymal C in legged_gym and capture observations

**Steps:**
1. Modify `legged_gym/scripts/play.py` to print observations
2. Compare values with our environment
3. Find discrepancies

### Priority 3: Test If Commands Need Scaling
**Action:** Try scaling commands in observation

**Change in `hetero_robot.py:1052`:**
```python
# Current:
loc_obs[:, 9:12] = agent_actions

# Try:
loc_obs[:, 9] = agent_actions[:, 0] * 2.0  # vx * lin_vel_scale
loc_obs[:, 10] = agent_actions[:, 1] * 2.0  # vy * lin_vel_scale
loc_obs[:, 11] = agent_actions[:, 2] * 0.25  # vyaw * ang_vel_scale
```

### Priority 4: Verify obs_buf Structure
**Action:** Add debug prints to understand obs_buf layout

**Questions to answer:**
- What shape is `obs_buf.dof_pos`?
- Does `obs_buf.dof_pos[1]` give Anymal C's DOFs or something else?
- Is it already accounting for heterogeneous layout?

### Priority 5: Try Direct State Access (Fallback)
**Action:** If obs_buf is wrong, compute observations directly from state tensors

**Approach:**
```python
# Instead of using obs_buf, use raw states:
anymal_dof_pos = self.dof_pos[:, 12:24]
anymal_dof_vel = self.dof_vel[:, 12:24]
anymal_lin_vel = self.base_lin_vel[agent_env_indices]
anymal_ang_vel = self.base_ang_vel[agent_env_indices]
```

---

## 📊 Test Results Summary

| Test | Go1 Behavior | Anymal C Behavior | Result |
|------|--------------|-------------------|--------|
| **Homogeneous 2x Go1** | ✅ Stable at 0.285m | N/A | ✅ PASS |
| **Hetero Go1 + Anymal (old obs)** | ✅ Stable | ❌ Falls, resets frequently | ❌ FAIL |
| **Hetero Go1 + Anymal (new obs)** | ? | ? | ⏳ NOT TESTED YET |

---

## 🎯 CRITICAL QUESTION

**Why does Go1 work but Anymal C doesn't when both use the same observation construction method now?**

Possible answers:
1. **obs_buf indexing is still wrong for Anymal C**
2. **Commands need scaling (Go1 might ignore unscaled commands)**
3. **Anymal C policy expects different observation format than we implemented**
4. **obs_buf structure doesn't account for heterogeneous DOF counts**

---

## 📁 Key Files

**Config Files:**
- Anymal C: `mqe/envs/anymal_c/anymal_c_config.py`
- Go1: `mqe/envs/go1/go1_config.py`
- Task: `task/cuboid/config.py`
- Training reference: `/home/gvlab/legged_gym/legged_gym/envs/anymal_c/mixed_terrains/anymal_c_rough_config.py`

**Code Files:**
- Observation construction: `mqe/envs/base/hetero_robot.py:1028-1056`
- Policy loading: `mqe/envs/base/hetero_robot.py:95-150`
- obs_buf population: `mqe/envs/base/hetero_robot.py:732-770`

**Test Files:**
- Homogeneous: `test_homogeneous_go1.py`
- Heterogeneous: `test_zero_actions.py`
- Debug: `dump_anymal_policy_input.py`, `debug_locomotion_output.py`

**Policy File:**
- Location: `resources/robots/anymal_c/policy_500.jit`
- Source: `/home/gvlab/legged_gym/logs/flat_anymal_c_rtx2070/Jan19_18-51-31_/policy_500.jit`

---

## 🔄 Latest Code Changes

### Session Today (2026-01-30):

1. ✅ Fixed PD gains pattern: `'joint'` → `'_'`
2. ✅ Fixed PD gains values: P=85→80, D=2.0
3. ✅ Fixed hind leg joint angles: HFE=-0.4, KFE=0.8
4. ✅ Fixed HAA offsets: 0.0 (matching training)
5. ✅ Fixed spawn height: 0.62m
6. ✅ Fixed action scale: 0.5
7. ✅ Changed foot_name: "SHANK" → "FOOT" (reverted based on URDF structure)
8. ✅ **Rewrote Anymal C observation construction to match Go1 method (using obs_buf)**

**Last Change Location:** `mqe/envs/base/hetero_robot.py:1028-1056`

---

## ⏭️ IMMEDIATE NEXT STEP

**TEST THE LATEST CHANGES:**
```bash
conda run -n mapush python test_zero_actions.py
```

**Expected behavior if fixed:**
- Anymal C should stand at ~0.60-0.62m
- Minimal or no episode resets
- Similar stability to Go1

**If still fails:**
- Proceed with Priority debugging steps above
- Most likely: Commands need scaling or obs_buf indexing is wrong
