# Session Summary - 2026-01-19

## Current Task
Implementing heterogeneous multi-agent RL training (Go1 + second robot) for MAPush environment.

## Session Progress

### What Happened
1. **Abandoned Jackal**: After extensive debugging, determined Jackal URDF is fundamentally broken in Isaac Gym (instant NaN at step 0)
2. **Switched to Anymal**: User decided to use Anymal as second robot instead of Jackal
3. **Investigated Anymal controller**: Checked if Anymal has required low-level locomotion controller

## Critical Findings

### Architecture Requirements (User Clarification)
**The system needs:**
- Mid-level RL policy outputs **velocity commands** [vx, vy, vyaw]
- Low-level controller converts these to either:
  - **Go1**: Joint torques via locomotion_policy + actuator_network
  - **Wheeled robot**: Wheel velocities via differential_drive_controller

**Control type MUST be 'C' (hierarchical) for both robots**, not 'P'

### Jackal Issues Found
1. **WRONG control_type**: Jackal config has `control_type='P'` but should be `'C'`
   - Location: `mqe/envs/jackal/jackal_config.py:86`
2. **Has differential drive controller**: `jackal.py:60-90` implements conversion from [vx, vy, vyaw] → wheel velocities
3. **URDF broken**: Instant NaN even in isolation tests

### Anymal Status
**Available:**
- ✅ URDF: `/home/gvlab/new-universal-MAPush/resources/robots/anymal_b/urdf/anymal_b.urdf`
- ✅ Spawns correctly in Isaac Gym (tested in `test_anymal_solo.py`, z≈0.6m stable)
- ✅ 12 DOFs (quadruped like Go1)

**Missing (CRITICAL BLOCKER):**
- ❌ **No locomotion policy** (only `walk_these_ways` exists for Go1)
  - Go1 has: `./mqe/utils/locomotion_checkpoints/walk_these_ways/{body_latest.jit, adaptation_module_latest.jit}`
  - Anymal: Nothing found
- ❌ **No actuator network** (only `./resources/actuator_nets/unitree_go1.pt` exists)

### Go1 Architecture (Reference)
```
RL Policy → [vx, vy, vyaw, ...]
  ↓ preprocess_action() (go1.py:63-93)
  ↓ locomotion_policy (go1.py:388-408) - walk_these_ways
  ↓ 12D joint position targets
  ↓ actuator_network (go1.py:366-381) - unitree_go1.pt
  ↓ Torques
```

## Current Blocker

**Cannot proceed with Anymal** because it lacks:
1. Locomotion policy to convert [vx, vy, vyaw] → 12D joint targets
2. Actuator network to convert joint targets → realistic torques

## Options to Proceed

### Option A: Homogeneous Go1 + Go1
- Both have full hierarchical control stack
- Proven to work
- No heterogeneity

### Option B: Train Anymal locomotion policy
- Requires training walk_these_ways policy for Anymal
- Significant effort (weeks?)
- Would need actuator network too

### Option C: Anymal with direct control
- Set Anymal control_type='P'
- RL policy directly outputs 12D joint targets
- Skip locomotion policy layer
- **Different architectures** for Go1 vs Anymal (hetero control types)
- May not work well for object pushing task

### Option D: Find pre-trained Anymal policy
- Search Isaac Gym examples/online
- Check if ANYbotics provides policies
- Unlikely to exist in compatible format

## Files Modified This Session
- None (only investigation/testing)

## Test Scripts Created Then Deleted
- `test_simple_box.py` (✅ worked)
- `test_go1_solo.py` (✅ worked)
- `test_anymal_solo.py` (✅ worked)
- `test_jackal_solo.py` (❌ NaN)
- `resources/robots/simple_wheeler/` (❌ NaN)

All deleted as requested.

## Previous Bugs Fixed (Earlier Session)
1. ✅ HARL wrapper observation space bug (`HARL/harl/envs/mapush/mapush_env.py:107-119`)
2. ✅ Viewer mode for heterogeneous agents (`HARL/harl_mapush/test.py:227-268`)
3. ✅ Jackal mesh paths (didn't matter, URDF fundamentally broken)

## Key Code Locations

### Go1 Hierarchical Control
- Config: `mqe/envs/go1/go1_config.py:109` - `control_type = 'C'`
- Locomotion policy loading: `go1.py:388-408`
- Actuator network loading: `go1.py:366-381`
- Torque computation: `go1.py:314-353`
- Action preprocessing: `go1.py:63-93`

### Jackal (Broken)
- Config: `mqe/envs/jackal/jackal_config.py:86` - `control_type = 'P'` ⚠️ WRONG
- Differential drive: `jackal.py:60-90`
- Step function: `jackal.py:92-135`

### Base Classes
- Control type handling: `mqe/envs/base/legged_robot.py:438-445`
- PD control: `torques = p_gains*(actions + default_pos - dof_pos) - d_gains*dof_vel`

## Next Session TODO

**User needs to decide:**
1. Which option to proceed with (A/B/C/D)?
2. If Option C (different control types), is that acceptable for heterogeneous training?
3. If Option A (homogeneous), abandon heterogeneous agents entirely?

**If proceeding with Anymal (any option):**
1. Create `mqe/envs/anymal/anymal_config.py`
2. Create `mqe/envs/anymal/anymal.py`
3. Register in robot registry
4. Implement appropriate control architecture based on choice

## Background Process Status
- Background Bash 025019 running: `conda run -n mapush python test_hetero_env.py 2>&1 | grep -A 250 "Test 4:"`
- Status: running, has output available

## User's Last Statement
"make a quick summary of session and latest work for enxt session to start over at"

---
**Session ended:** 2026-01-19
**Next action:** User decision on which option to pursue for second robot
