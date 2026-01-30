# Anymal C Integration Notes

Summary of issues encountered and solutions when integrating Anymal C into MAPush heterogeneous environment.

---

## Training Setup

**Environment**: `anymal_training` (Python 3.8, PyTorch 1.13.1+cu116)

**Training command**:
```bash
cd /home/gvlab/legged_gym
conda activate anymal_training
python legged_gym/scripts/train.py --task=anymal_c_flat_rtx2070 --headless
```

**Checkpoints**: `/home/gvlab/legged_gym/logs/flat_anymal_c_rtx2070/`

---

## Critical Config Parameters (Must Match Training)

| Parameter | Value | Notes |
|-----------|-------|-------|
| action_scale | 0.5 | Scales policy output before adding to defaults |
| stiffness | 80.0 | PD P-gain |
| damping | 2.0 | PD D-gain |
| spawn height | 0.62m | Initial z position |

**Default Joint Angles** (front vs hind legs have OPPOSITE signs for HFE/KFE):
```
Front: HAA=0.0, HFE=+0.4, KFE=-0.8
Hind:  HAA=0.0, HFE=-0.4, KFE=+0.8
```

---

## Key Integration Issues & Solutions

### 1. Actuator Network Sign Convention
**Problem**: Anymal C LSTM actuator expects `target - actual`, but code computed `actual - target`
**Solution**: Negate position error in `_load_actuator_network()`:
```python
sea_input[:, 0, 0] = -joint_pos_err.flatten()  # NEGATED
```

### 2. LSTM Actuator Network Requires Hidden State
**Problem**: `anydrive_v3_lstm.pt` is LSTM, not feedforward like Go1's
**Solution**: Initialize and maintain hidden/cell states:
```python
sea_hidden = torch.zeros(2, num_envs * 12, 8, device=device)
sea_cell = torch.zeros(2, num_envs * 12, 8, device=device)
torques, (h, c) = actuator_net(sea_input, (sea_hidden, sea_cell))
```

### 3. No hip_scale_reduction
**Problem**: Config had `hip_scale_reduction=0.5` but Anymal wasn't trained with it
**Solution**: Remove from config (only Go1 uses this)

### 4. Observation Structure (48-dim)
```
[0:3]   base_lin_vel * 2.0
[3:6]   base_ang_vel * 0.25
[6:9]   projected_gravity (NOT scaled)
[9:12]  commands * [2.0, 2.0, 0.25]
[12:24] (dof_pos - default) * 1.0
[24:36] dof_vel * 0.05
[36:48] previous_actions (raw policy outputs)
```

### 5. foot_name Setting
With `collapse_fixed_joints=True`, use `foot_name="FOOT"` (verified FOOT bodies exist in our URDF).

---

## Files

| Component | Path |
|-----------|------|
| Config | `mqe/envs/anymal_c/anymal_c_config.py` |
| Policy | `resources/robots/anymal_c/policy_500.jit/policy_1.pt` |
| Actuator Net | `resources/actuator_nets/anydrive_v3_lstm.pt` |
| URDF | `resources/robots/anymal_c/urdf/anymal_c.urdf` |

---

## legged_gym Dependency Fixes

For Python 3.8 compatibility with rsl_rl 3.3.0:

1. **rsl_rl/networks/memory.py**: Change `tuple[...]` to `Tuple[...]` (import from typing)
2. **rsl_rl/pyproject.toml**: Relax `torch>=2.6.0` to `torch>=1.10.0`
3. **legged_gym config**: Add `obs_groups = {}` at ROOT level of `LeggedRobotCfgPPO` class
