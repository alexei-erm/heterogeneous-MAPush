# Anymal C Locomotion Policy Training Setup

**Date**: 2026-01-19
**Purpose**: Train Anymal C quadruped locomotion policy for heterogeneous agent setup with Go1 in MAPush

---

## Summary

Successfully configured environment and dependencies for training Anymal C locomotion policy in Isaac Gym. The policy will take velocity commands [vx, vy, vyaw] and output joint torques, similar to the existing Go1 setup.

---

## Environment Setup

**Environment Name**: `anymal_training`
**Python Version**: 3.8 (required by Isaac Gym <3.9 constraint)

### Dependencies Installed
1. PyTorch 1.13.1+cu116
2. torchvision 0.14.1+cu116
3. Isaac Gym 1.0rc4
4. rsl_rl 3.3.0 (RL library)
5. legged_gym 1.0.0 (training framework)
6. matplotlib, scipy, numpy, etc.

### Activation Command
```bash
conda activate anymal_training
```

---

## Training Configuration

**Config File**: `/home/gvlab/legged_gym/legged_gym/envs/anymal_c/flat/anymal_c_flat_rtx2070_config.py`

**Key Parameters**:
- `num_envs`: 2048 (reduced from 4096 for 8GB VRAM)
- `num_observations`: 48
- `num_actions`: 12 (Anymal C has 12 DOF)
- `max_iterations`: 500
- `save_interval`: 50
- Terrain: Flat plane (faster training than rough terrain)

**Training Command**:
```bash
cd /home/gvlab/legged_gym
python legged_gym/scripts/train.py --task=anymal_c_flat_rtx2070 --headless
```

**Expected Training Time**: 3-5 hours on RTX 2070 8GB

**Checkpoint Location**: `/home/gvlab/legged_gym/logs/flat_anymal_c_rtx2070/`

---

## Problems Encountered & Solutions

### Problem 1: Python Version Compatibility
**Issue**: Attempted Python 3.10 → Isaac Gym requires <3.9
**Attempted**: Python 3.9 → Isaac Gym requires <3.9 (exclusive)
**Solution**: Used Python 3.8 (Isaac Gym supports 3.6-3.8)

### Problem 2: rsl_rl Type Annotations (Python 3.8)
**Error**: `TypeError: 'type' object is not subscriptable`
**Location**: `/home/gvlab/rsl_rl/rsl_rl/networks/memory.py:14`
**Issue**: Code used `tuple[...]` syntax (Python 3.9+ only) instead of `Tuple[...]` from typing

**Fixed Lines** (memory.py):
```python
from typing import Union, Tuple, Optional

HiddenState = Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor], None]  # Fixed line 14

def forward(self, input: torch.Tensor, masks: Optional[torch.Tensor] = None, ...) -> torch.Tensor:  # Fixed line 36

def reset(self, dones: Optional[torch.Tensor] = None, ...) -> None:  # Fixed line 51

def detach_hidden_state(self, dones: Optional[torch.Tensor] = None) -> None:  # Fixed line 69
```

### Problem 3: rsl_rl Dependency Version Mismatch
**Error**: `ERROR: No matching distribution found for torch>=2.6.0`
**Issue**: rsl_rl pyproject.toml required torch>=2.6.0 (doesn't exist yet)

**Fixed**: `/home/gvlab/rsl_rl/pyproject.toml`
```toml
# Before:
dependencies = [
    "torch>=2.6.0",
    "tensordict>=0.7.0",
    ...
]

# After:
dependencies = [
    "torch>=1.10.0",  # Relaxed requirement
    "tensordict",      # Removed version constraint
    ...
]
```

Also fixed:
- `license = "BSD-3-Clause"` → `license = {text = "BSD-3-Clause"}`
- `requires-python = ">=3.9"` → `requires-python = ">=3.8"`
- Package name: `"rsl-rl-lib"` → `"rsl-rl"` (to match legged_gym expectation)

### Problem 4: Missing `obs_groups` Parameter
**Error**: `KeyError: 'obs_groups'`
**Location**: `/home/gvlab/rsl_rl/rsl_rl/runners/on_policy_runner.py:43`
**Root Cause**: rsl_rl 3.3.0 expects `obs_groups` parameter at root level of PPO config, but legged_gym was designed for older rsl_rl version

**Attempted Fixes** (Failed):
1. ❌ Added `obs_groups = {}` to `runner` class (wrong location)
2. ❌ Python cache not clearing

**Working Solution**: Added `obs_groups = {}` at ROOT level of PPO config class

**Fixed**: `/home/gvlab/legged_gym/legged_gym/envs/base/legged_robot_config.py:205`
```python
class LeggedRobotCfgPPO(BaseConfig):
    seed = 1
    runner_class_name = 'OnPolicyRunner'
    obs_groups = {}  # ← CRITICAL: Must be at root level, not inside runner!
    class policy:
        ...
```

**Why This Was Tricky**: The config structure had nested classes, and it wasn't obvious that `obs_groups` needed to be at the same level as `seed`, not inside the `runner` nested class. The error message didn't clarify this hierarchy requirement.

---

## File Modifications Summary

### Modified Files
1. `/home/gvlab/rsl_rl/rsl_rl/networks/memory.py`
   - Fixed Python 3.8 type annotation compatibility

2. `/home/gvlab/rsl_rl/pyproject.toml`
   - Relaxed torch version requirement
   - Fixed license format
   - Changed package name to "rsl-rl"
   - Lowered Python requirement to 3.8

3. `/home/gvlab/legged_gym/legged_gym/envs/base/legged_robot_config.py`
   - Added `obs_groups = {}` at root level of `LeggedRobotCfgPPO`

### Created Files
1. `/home/gvlab/legged_gym/legged_gym/envs/anymal_c/flat/anymal_c_flat_rtx2070_config.py`
   - Custom config optimized for RTX 2070 8GB VRAM
   - Reduced num_envs from 4096 to 2048
   - Increased max_iterations to 500

2. `/home/gvlab/legged_gym/legged_gym/envs/__init__.py`
   - Registered `anymal_c_flat_rtx2070` task

---

## Next Steps

1. ✅ Environment setup complete
2. ✅ Dependency issues resolved
3. ⏳ **CURRENT**: Start training
4. ⏳ Monitor VRAM usage (may need to reduce num_envs further if OOM)
5. ⏳ Export trained policy to .jit format
6. ⏳ Create Anymal C robot class in MAPush (mirror Go1 structure)
7. ⏳ Integrate Anymal policy into MAPush heterogeneous environment
8. ⏳ Test heterogeneous training (Go1 + Anymal C)

---

## Training Monitoring

**Check VRAM Usage**:
```bash
watch -n 1 nvidia-smi
```

If training crashes with OOM error, reduce `num_envs` in config:
- Current: 2048
- Try: 1024 (if OOM occurs)
- Minimum viable: 512 (slower but stable)

**Monitor Training Progress**:
```bash
# Checkpoints saved every 50 iterations
ls -lh /home/gvlab/legged_gym/logs/flat_anymal_c_rtx2070/
```

---

## Key Learnings

1. **Isaac Gym is deprecated** - Locked to Python 3.6-3.8, PyTorch 1.x
2. **rsl_rl 3.3.0 compatibility** - Newer than legged_gym expects, required config modifications
3. **Config class hierarchy matters** - `obs_groups` placement was critical
4. **Python bytecode caching** - Must clear `__pycache__` after config edits
5. **VRAM constraints** - RTX 2070 8GB is at the lower limit for RL training (A100 40GB is recommended)

---

## References

- **legged_gym**: https://github.com/leggedrobotics/legged_gym
- **rsl_rl**: https://github.com/leggedrobotics/rsl_rl
- **Isaac Gym**: NVIDIA deprecated physics simulator
- **Anymal C**: ANYbotics quadruped robot (12 DOF)
