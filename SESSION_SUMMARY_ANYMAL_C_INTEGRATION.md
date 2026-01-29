# Anymal C Integration for Heterogeneous Multi-Agent RL

**Date**: 2026-01-19
**Project**: MAPush Heterogeneous Agent Training (Go1 + Anymal C)
**Status**: Policy trained and verified, ready for MAPush integration

---

## Executive Summary

Successfully trained an Anymal C locomotion policy in Isaac Gym and exported it to JIT format. The policy is verified working and ready to be integrated into the MAPush heterogeneous environment alongside Go1. This session focused on training preparation, while the next session should implement the Anymal C robot class in MAPush.

---

## Completed Tasks

### 1. Environment Setup ✅
- **Environment Name**: `anymal_training`
- **Python Version**: 3.8 (Isaac Gym requirement)
- **Activation**: `conda activate anymal_training`

### 2. Dependencies Installed ✅
- PyTorch 1.13.1+cu116
- Isaac Gym 1.0rc4
- rsl_rl 1.0.2 (downgraded from 3.3.0 for compatibility)
- legged_gym 1.0.0
- tensorboard

### 3. Compatibility Issues Fixed ✅
See `/home/gvlab/new-universal-MAPush/ANYMAL_C_TRAINING_SETUP.md` for detailed fixes:
- Python 3.8 type annotation compatibility in rsl_rl
- Torch version requirement (2.6.0 → 1.10.0)
- rsl_rl version downgrade (3.3.0 → 1.0.2)
- Package name mismatch resolution

### 4. Policy Training ✅
**Training Command**:
```bash
cd /home/gvlab/legged_gym
conda activate anymal_training
python legged_gym/scripts/train.py --task=anymal_c_flat_rtx2070 --headless --pipeline gpu
```

**Training Results**:
- Total iterations: 500
- Training time: 8.5 minutes (511.92 seconds)
- Final mean reward: 17.83
- Final episode length: 962 steps
- Computation speed: 96,589 steps/s
- VRAM usage: ~3.5GB / 8GB (RTX 2070)

**Config Used**: `/home/gvlab/legged_gym/legged_gym/envs/anymal_c/flat/anymal_c_flat_rtx2070_config.py`
- num_envs: 4096
- num_observations: 48
- num_actions: 12
- Terrain: Flat plane

### 5. Policy Export ✅
**Export Script**: `/home/gvlab/legged_gym/export_anymal_policy.py`

**Exported Policy Location**:
```
/home/gvlab/new-universal-MAPush/resources/robots/anymal_c/policy_500.jit/
```

**Policy Interface**:
- Input: 48-dimensional observation vector
- Output: 12-dimensional action vector (joint torques)

### 6. Policy Verification ✅
**Verification Command**:
```bash
cd /home/gvlab/legged_gym
conda activate anymal_training
python legged_gym/scripts/play.py --task=anymal_c_flat_rtx2070 --load_run Jan19_18-51-31_
```

Result: Policy successfully controls Anymal C for locomotion

---

## Current State

### File Structure
```
/home/gvlab/new-universal-MAPush/
├── resources/robots/anymal_c/
│   ├── urdf/anymal_c.urdf                    # Robot URDF
│   ├── meshes/                               # Visual meshes
│   └── policy_500.jit/                       # ✅ Trained locomotion policy
│       └── policy_1.pt
│
├── mqe/envs/
│   ├── go1/                                  # Existing Go1 implementation
│   │   ├── go1.py                           # Go1 robot class
│   │   └── go1_config.py                    # Go1 configuration
│   │
│   ├── anymal_c/                            # ⏳ TODO: Create this directory
│   │   ├── anymal_c.py                      # ⏳ TODO: Anymal C robot class
│   │   └── anymal_c_config.py               # ⏳ TODO: Anymal C configuration
│   │
│   └── robot_registry.py                    # ⏳ TODO: Register Anymal C
│
└── ANYMAL_C_TRAINING_SETUP.md               # ✅ Training documentation
```

### Training Checkpoints
```
/home/gvlab/legged_gym/logs/flat_anymal_c_rtx2070/Jan19_18-51-31_/
├── model_50.pt
├── model_100.pt
├── model_150.pt
├── model_200.pt
├── model_250.pt
├── model_300.pt
├── model_350.pt
├── model_400.pt
├── model_450.pt
└── model_500.pt                              # Final checkpoint
```

---

## Next Steps (Critical Path)

### Phase 1: Create Anymal C Robot Class in MAPush

#### Step 1.1: Create Directory Structure
```bash
mkdir -p /home/gvlab/new-universal-MAPush/mqe/envs/anymal_c
```

#### Step 1.2: Create `anymal_c_config.py`
**Reference**: `/home/gvlab/new-universal-MAPush/mqe/envs/go1/go1_config.py`

**Key configurations to mirror from Go1**:
- Inherit from `LeggedRobotFieldCfg`
- Set `num_actions = 12` (Anymal C has 12 DOF)
- Set `num_observations` based on MAPush requirements
- Configure `control.control_type = "C"` (hierarchical control)
- Set `locomotion_policy_dir` to point to Anymal C policy
- Configure default joint angles for Anymal C
- Set torque limits for Anymal C actuators

**Important differences from Go1**:
- Different URDF path: `"{LEGGED_GYM_ROOT_DIR}/resources/robots/anymal_c/urdf/anymal_c.urdf"`
- Different joint names (Anymal C naming convention)
- Different default standing pose
- Different foot name (check URDF)
- May need different torque limits

#### Step 1.3: Create `anymal_c.py`
**Reference**: `/home/gvlab/new-universal-MAPush/mqe/envs/go1/go1.py`

**Key components to implement**:
1. **Class definition**: `class AnymalC(LeggedRobotField)`
2. **__init__**: Load locomotion policy (similar to Go1:388-408)
3. **_prepare_locomotion_policy()**: Load JIT model from policy_500.jit
4. **preprocess_action()**: Convert high-level commands to locomotion observations
5. **step()**: Execute actions and step physics
6. **compute_observations()**: Compute robot observations
7. **_compute_torques()**: Use actuator network to compute torques

**Critical sections from Go1 to adapt**:
- Lines 32-33: Load locomotion policy
- Lines 388-408: `_prepare_locomotion_policy()` function
- Lines 63-107: `preprocess_action()` - maps mid-level commands to locomotion obs
- Lines 314-353: `_compute_torques()` - actuator network integration

**Policy loading modification needed**:
```python
# Go1 uses walk_these_ways with body + adaptation module
body = torch.jit.load(self.cfg.control.locomotion_policy_dir + '/body_latest.jit')
adaptation_module = torch.jit.load(self.cfg.control.locomotion_policy_dir + '/adaptation_module_latest.jit')

# Anymal C uses single policy file (simpler)
policy = torch.jit.load(self.cfg.control.locomotion_policy_dir + '/policy_500.jit/policy_1.pt')
```

#### Step 1.4: Register Anymal C in Robot Registry
**File**: `/home/gvlab/new-universal-MAPush/mqe/envs/robot_registry.py`

Add Anymal C similar to how Go1 is registered.

#### Step 1.5: Update MAPush __init__.py
Ensure Anymal C can be imported and instantiated.

### Phase 2: Test Anymal C Standalone

#### Step 2.1: Create Test Script
```python
# test_anymal_standalone.py
# Test Anymal C in MAPush without heterogeneous setup
```

**Test checklist**:
- Environment creation succeeds
- Policy loads correctly
- Observations computed correctly
- Actions execute without errors
- Robot doesn't fall immediately
- Locomotion commands work

#### Step 2.2: Debug Common Issues
- URDF loading errors
- Policy input/output dimension mismatches
- Joint name mismatches
- Actuator network compatibility
- Observation scaling issues

### Phase 3: Heterogeneous Integration (Go1 + Anymal C)

#### Step 3.1: Update Heterogeneous Robot Configuration
**File**: `/home/gvlab/new-universal-MAPush/mqe/envs/base/hetero_robot.py`

Ensure it can handle:
- Different DOF counts (Go1: 12, Anymal C: 12) ✅ Same DOF
- Different observation spaces
- Different action spaces
- Different torque limits

#### Step 3.2: Test Heterogeneous Environment
```python
# Test Go1 + Anymal C in same environment
# Verify both robots can be controlled independently
```

#### Step 3.3: HARL Training Integration
Update HARL config to support Anymal C as an agent type.

---

## Key Technical Details

### Anymal C Specifications
- **DOF**: 12 (3 joints per leg × 4 legs)
- **Joint layout**: HAA (Hip Abduction/Adduction), HFE (Hip Flexion/Extension), KFE (Knee Flexion/Extension)
- **Standing height**: ~0.5m (check URDF for exact value)
- **Mass**: ~30-35kg (check URDF)
- **Policy observation dim**: 48
- **Policy action dim**: 12

### Policy Architecture (Anymal C)
Unlike Go1's walk_these_ways (which uses body + adaptation module), the Anymal C policy is a single JIT model:
- Trained with legged_gym default PPO
- Actor network: [512, 256, 128] hidden dims
- Activation: ELU
- No recurrence
- No adaptation module

### Control Hierarchy
```
HARL Policy (Multi-Agent RL)
    ↓
Mid-level commands [vx, vy, vyaw, ...]
    ↓
Locomotion Policy (Anymal C)
    ↓
Joint position targets [12D]
    ↓
Actuator Network
    ↓
Joint torques [12D]
    ↓
Isaac Gym Physics
```

### Observation Space Mapping
The locomotion policy expects 48-dim observations:
```
[0:3]    projected_gravity      (3)
[3:5]    lin_vel_command        (2)  [vx, vy]
[5]      ang_vel_command        (1)  [vyaw]
[6]      body_height_command    (1)
[7]      gait_frequency         (1)
[8:12]   gait_phase_params      (4)
[12]     footswing_height       (1)
[13:15]  body_pose_cmd          (2)  [pitch, roll]
[15]     stance_width           (1)
[16]     stance_length          (1)
[17]     aux_reward             (1)
[18:30]  dof_pos                (12)
[30:42]  dof_vel                (12)
[42:54]  last_action            (12)
[54:66]  last_last_action       (12)
[66:70]  clock_inputs           (4)
```

**History buffer**: Policy uses 2100-dim history (last 30 observations concatenated)

---

## Important File Locations

### Training Environment (anymal_training conda env)
```
/home/gvlab/legged_gym/
├── legged_gym/envs/anymal_c/flat/anymal_c_flat_rtx2070_config.py
├── legged_gym/scripts/train.py
├── legged_gym/scripts/play.py
├── export_anymal_policy.py
└── logs/flat_anymal_c_rtx2070/Jan19_18-51-31_/

/home/gvlab/rsl_rl/
└── (modified for Python 3.8 compatibility)
```

### MAPush Environment (mapush conda env)
```
/home/gvlab/new-universal-MAPush/
├── resources/robots/anymal_c/
├── mqe/envs/go1/              # Reference implementation
├── mqe/envs/anymal_c/         # TODO: Create this
├── mqe/envs/base/hetero_robot.py
└── HARL/harl/envs/mapush/
```

---

## Potential Issues & Solutions

### Issue 1: Actuator Network Path
**Problem**: Anymal C may need different actuator network than Go1
**Solution**:
- Check if Go1 actuator network works (similar hardware)
- If not, may need to train Anymal C specific actuator network
- For now, try using Go1's: `./resources/actuator_nets/unitree_go1.pt`

### Issue 2: Policy Input/Output Dimension Mismatch
**Problem**: MAPush observation space may differ from training
**Solution**:
- Carefully match observation dimensions in `preprocess_action()`
- Ensure all 48 dims are correctly populated
- Use same observation scaling as training

### Issue 3: Joint Name Mismatches
**Problem**: Anymal C joint names differ from Go1
**Solution**:
- Parse URDF to get exact joint names
- Update `default_joint_angles` dictionary in config
- Ensure joint order matches policy expectations

### Issue 4: Different Control Frequency
**Problem**: MAPush control dt may differ from training
**Solution**:
- Check decimation settings
- Training used dt=0.005, decimation=4 → policy dt=0.02
- Match this in MAPush config

### Issue 5: History Buffer Initialization
**Problem**: Policy expects 2100-dim history, need proper initialization
**Solution**:
- Initialize `history_locomotion_obs` in `_prepare_locomotion_policy()`
- Reset to zero in `_reset_buffers()`
- Same as Go1 line 394

---

## Testing Checklist (for next session)

### Anymal C Standalone Tests
- [ ] Environment creation succeeds
- [ ] URDF loads without errors
- [ ] Policy loads from JIT file
- [ ] Robot spawns at correct height
- [ ] Observations computed correctly (48 dims)
- [ ] Actions execute without NaN/Inf
- [ ] Robot maintains balance for 10+ seconds
- [ ] Velocity commands work (vx, vy, vyaw)
- [ ] No physics explosions or instabilities
- [ ] Torques within limits

### Heterogeneous Environment Tests
- [ ] Go1 + Anymal C spawn together
- [ ] Both agents observe correctly
- [ ] Both agents act independently
- [ ] No buffer size conflicts
- [ ] No tensor shape mismatches
- [ ] Environment resets work
- [ ] Rewards computed correctly

### HARL Integration Tests
- [ ] HARL can create heterogeneous env
- [ ] Training loop starts without errors
- [ ] Gradients flow correctly
- [ ] Checkpoints save/load
- [ ] TensorBoard logging works

---

## Questions to Resolve

1. **Actuator Network**: Does Anymal C need its own actuator network or can it use Go1's?
   - Likely can share if both use similar motors (Unitree-style)
   - Check torque-position characteristics

2. **Observation Space**: What observations does HARL policy need from Anymal C?
   - Currently Go1 uses ~235 dims
   - Anymal C may need similar set
   - Need to design heterogeneous observation space

3. **Reward Function**: Should Anymal C have different rewards than Go1?
   - Different locomotion characteristics
   - May need different reward weights

4. **Terrain**: Can Anymal C handle same terrain as Go1?
   - Trained on flat terrain
   - May need curriculum for rough terrain

---

## Key Learnings

1. **Isaac Gym is deprecated** - locked to Python 3.6-3.8, future work should migrate to Isaac Lab
2. **rsl_rl versioning matters** - v3.3.0 incompatible with legged_gym, use v1.0.2
3. **Fast iterations at training start are normal** - robots fall quickly initially
4. **VRAM usage varies with episode length** - longer episodes need more memory
5. **GPU pipeline flag** - use `--pipeline gpu` not `--use_gpu_pipeline`
6. **Import order critical** - isaacgym must be imported before torch

---

## Resources

### Documentation
- `/home/gvlab/new-universal-MAPush/ANYMAL_C_TRAINING_SETUP.md` - Detailed training setup
- `/home/gvlab/new-universal-MAPush/SESSION_SUMMARY_2026-01-19.md` - Previous session notes

### Reference Code
- Go1 implementation: `/home/gvlab/new-universal-MAPush/mqe/envs/go1/`
- Hetero robot base: `/home/gvlab/new-universal-MAPush/mqe/envs/base/hetero_robot.py`
- Legged gym Anymal: `/home/gvlab/legged_gym/legged_gym/envs/anymal_c/`

### External Links
- legged_gym: https://github.com/leggedrobotics/legged_gym
- rsl_rl: https://github.com/leggedrobotics/rsl_rl
- Anymal C: https://www.anybotics.com/anymal

---

## Next Session Immediate Actions

**Start here**:
1. Create `/home/gvlab/new-universal-MAPush/mqe/envs/anymal_c/` directory
2. Copy Go1 config as template for `anymal_c_config.py`
3. Modify config for Anymal C specifics (URDF path, joint names, policy path)
4. Copy Go1 class as template for `anymal_c.py`
5. Adapt policy loading to use single JIT file instead of body+adaptation
6. Test standalone Anymal C environment

**Command to start**:
```bash
cd /home/gvlab/new-universal-MAPush
mkdir -p mqe/envs/anymal_c
# Then proceed with file creation
```

---

**Session End**: 2026-01-19
**Next Session**: Anymal C MAPush Integration
