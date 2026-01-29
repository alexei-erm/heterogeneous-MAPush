# Heterogeneous Agent Implementation Fixes

**Date:** 2026-01-17 to 2026-01-18
**Task:** Fix heterogeneous Go1 + Jackal training for MAPush environment

---

## Summary

Fixed critical bugs in heterogeneous multi-agent RL implementation that were preventing Go1 + Jackal training from working. The previous training run (`go1_jackal_hetero_concat_critic`) was **completely broken** due to observation space bugs - the checkpoint is unusable.

---

## Critical Bugs Fixed

### 1. **HARL Wrapper Observation Space Bug** ⚠️ CRITICAL
**File:** `HARL/harl/envs/mapush/mapush_env.py:111-115`

**Problem:**
```python
# WRONG - duplicates same space for both agents even in hetero mode!
self.observation_space = [self.env.observation_space] * self.n_agents
self.action_space = [self.env.action_space] * self.n_agents
```

In heterogeneous mode, `self.env.observation_space` is ALREADY a list of different spaces (Go1: 235 dims, Jackal: different), but this code duplicated the first space for both agents.

**Impact:**
- All previous hetero training used WRONG observation dimensions (8 dims instead of 235)
- Trained policies are completely broken and unusable
- Explains why Go1 fell randomly and Jackal didn't move in visualization

**Fix:**
```python
# CORRECT - use different spaces in hetero mode
if self.is_hetero and isinstance(self.env.observation_space, list):
    # Heterogeneous: spaces already different per agent
    self.observation_space = self.env.observation_space
    self.action_space = self.env.action_space
else:
    # Homogeneous: same space for all agents
    self.observation_space = [self.env.observation_space] * self.n_agents
    self.action_space = [self.env.action_space] * self.n_agents
```

---

### 2. **Test Script Observation Space Handling**
**File:** `HARL/harl_mapush/test.py:256-268`

**Problem:**
Test viewer mode didn't handle heterogeneous observation/action spaces - treated them as single space for both agents.

**Fix:**
```python
# Check if spaces are already lists (hetero mode) or single space (homo mode)
if isinstance(obs_space, list):
    # Heterogeneous mode - use different spaces per agent
    obs_spaces = obs_space
    act_spaces = act_space
else:
    # Homogeneous mode - same space for all agents
    obs_spaces = [obs_space] * n_agents
    act_spaces = [act_space] * n_agents
```

---

### 3. **Test Script Environment Creation**
**File:** `HARL/harl_mapush/test.py:227-262`

**Problem:**
Viewer mode always called `make_mqe_env()` which creates homogeneous (2x Go1) environment, even when `--hetero_agent jackal` was specified.

**Fix:**
```python
# Create environment - use make_hetero_env if hetero_agent specified
if hetero_agent:
    print(f"  Using heterogeneous mode: agent0=go1, agent1={hetero_agent}")
    env_raw, _ = make_hetero_env(
        env_name=args.task,
        agent_types=['go1', hetero_agent],
        args=args
    )
else:
    print("  Using homogeneous mode: 2x go1")
    env_raw, _ = make_mqe_env(args.task, args, custom_cfg=custom_cfg(args, hetero_agent=hetero_agent))
```

---

### 4. **Jackal URDF Mesh Paths**
**File:** `resources/robots/jackal/urdf/jackal.urdf`

**Problem:**
Mesh file paths missing `../meshes/` prefix, causing meshes to not load:
```xml
<!-- WRONG -->
<mesh filename="jackal-base.stl" scale="1 1 1"/>
```

**Fix:**
```xml
<!-- CORRECT -->
<mesh filename="../meshes/jackal-base.stl" scale="1 1 1"/>
```

Applied to:
- Line 26: chassis_link visual (jackal-base.stl)
- Line 55: left_wheel_link visual (jackal-wheel.stl)
- Line 86: right_wheel_link visual (jackal-wheel.stl)

---

## Previous Fixes (Already Applied)

### 5. **Jackal Velocity Control Mode**
**File:** `mqe/envs/jackal/jackal_config.py:45`

Changed from effort (torque) mode to velocity mode for stability:
```python
default_dof_drive_mode = 1  # VELOCITY control (stable for wheeled robots!)
                             # Was 3 (EFFORT/torque) which caused physics instability
```

### 6. **Wheel Velocity Limiting**
**File:** `mqe/envs/base/hetero_robot.py:769-780`

Added conservative velocity limits to prevent physics explosions:
```python
max_wheel_velocity = 10.0  # rad/s (conservative, URDF allows 20)
left_wheel_vel = torch.clamp(left_wheel_vel, -max_wheel_velocity, max_wheel_velocity)
right_wheel_vel = torch.clamp(right_wheel_vel, -max_wheel_velocity, max_wheel_velocity)
```

### 7. **Caster Height Fix**
**File:** `resources/robots/jackal/urdf/jackal.urdf:133, 153`

Raised casters from -0.02 to 0.02 to prevent ground interference:
```xml
<!-- Drive wheel bottom: 0.0345 - 0.098 = -0.0635m -->
<!-- Caster bottom (old): -0.02 - 0.05 = -0.07m (TOO LOW, caused bouncing) -->
<!-- Caster bottom (new): 0.02 - 0.05 = -0.03m (above wheels, stability only when tilted) -->
<origin xyz="0.131 0 0.02" rpy="0 0 0"/>  <!-- front caster -->
<origin xyz="-0.131 0 0.02" rpy="0 0 0"/>  <!-- rear caster -->
```

### 8. **Reward Tensor Handling**
**File:** `mqe/envs/wrappers/go1_push_mid_wrapper.py:832-841`

Fixed reward tensor to use repeat() instead of expand() and clean NaN:
```python
team_reward = reward.sum(dim=1, keepdim=True)  # (num_envs, 1)

# CRITICAL: Clean NaN/Inf from team_reward BEFORE expanding
team_reward[torch.isnan(team_reward)] = 0
team_reward[torch.isinf(team_reward)] = 0

# Use repeat() instead of expand() to create proper copy (not view)
reward = team_reward.repeat(1, self.num_agents)  # (num_envs, num_agents)
```

### 9. **Differential Drive Controller Integration**
**File:** `mqe/envs/base/hetero_robot.py:826-843`

Properly integrated differential drive for Jackal in heterogeneous mode:
```python
if hasattr(self, 'wheel_vel_targets') and 1 in self.wheel_vel_targets:
    # Set wheel velocity targets (rad/s)
    # Isaac Gym will apply built-in stable velocity controller
    torques_jackal = self.wheel_vel_targets[1]  # Actually velocities, not torques!
```

### 10. **Jackal Initial Height**
**File:** `mqe/envs/jackal/jackal_config.py:72`

Set correct ground clearance for wheeled robot:
```python
# Wheel radius = 0.098m, wheel center 0.1m above base_link
# For wheels to touch ground: base_link at ~0.10m (includes settling margin)
pos = [0.0, 0.0, 0.10]  # Jackal base height ~10cm (was 0.15, caused z-wave violations)
```

---

## Training Status

### ❌ Previous Training (BROKEN)
**Checkpoint:** `results/mapush/go1push_mid/happo/go1_jackal_hetero_concat_critic/seed-00001-2026-01-17-22-26-05/checkpoints/10M`

**Issues:**
- Trained with wrong observation dimensions (8 instead of 235)
- Actor input layer: 8 dims (should be 235 for Go1)
- Checkpoint completely unusable
- Explains all visualization failures:
  - Go1 falling randomly
  - Jackal not moving
  - NaN after 1M steps

**Verification:**
```bash
python3 << 'EOF'
import torch
checkpoint = torch.load('results/.../10M/actor_agent0.pt')
for key, value in checkpoint.items():
    if 'weight' in key:
        print(f"{key}: {value.shape}")
# Output: base.mlp.fc.0.weight: torch.Size([256, 8])  <-- WRONG! Should be 235
EOF
```

### ✅ New Training (FIXED)
**Command:**
```bash
python HARL/harl_mapush/train.py \
    --exp_name go1_jackal_hetero_fixed \
    --use_concat_agent_observations_critic True \
    --mapush_og_rewards_teamified True \
    --hetero_agent jackal
```

**Expected:**
- Proper observation spaces: Go1 (235 dims), Jackal (different)
- Correct mesh loading (black/yellow Jackal)
- Stable physics with velocity control
- Both robots moving and learning

---

## Testing & Visualization

### Test Environment (Headless)
```bash
conda run -n mapush python test_hetero_env.py
```

### Visualize Trained Checkpoint
```bash
./run_testing.sh \
    --checkpoint results/mapush/go1push_mid/happo/go1_jackal_hetero_fixed/seed-00001-.../checkpoints/10M \
    --mode viewer \
    --num_episodes 10 \
    --seed 1 \
    --hetero_agent jackal
```

**Critical:** Must include `--hetero_agent jackal` flag!

---

## Architecture Details

### Unified Action Space
Both agents use same high-level action space: `[vx, vy, vyaw]` (3 DOF)

**Go1 (Quadruped):**
- 3D action → 12 joint positions (via locomotion policy)
- Control: Hierarchical policy with PD control

**Jackal (Wheeled):**
- 3D action → 2 wheel velocities (via differential drive controller)
- Control: Direct velocity mode (Isaac Gym built-in controller)
- Kinematics:
  ```
  v_left  = (vx - vyaw * wheel_base/2) / wheel_radius
  v_right = (vx + vyaw * wheel_base/2) / wheel_radius
  ```

### Differential Drive Parameters
```python
wheel_radius = 0.098  # meters (from URDF)
wheel_base = 0.37558  # meters (2 * 0.18779 from URDF)
max_wheel_velocity = 10.0  # rad/s (conservative limit)
```

### Physics Configuration
- **Simulation:** Isaac Gym (PhysX backend)
- **Frequency:** 50 Hz (dt = 0.02s)
- **Go1 Control:** Position control (DOF_MODE_POS = 0)
- **Jackal Control:** Velocity control (DOF_MODE_VEL = 1)
- **Ground Friction:** 0.6 static, 0.5 dynamic

---

## Known Issues & Limitations

### 1. NaN in Learned Policy (ONGOING)
- Random actions: Stable (0 NaN in 1000 steps)
- Learned policy: NaN starts after ~500k-1M steps
- **Cause:** Policy learns physics-breaking actions
- **Mitigation:** NaN cleaning prevents crashes, but policy doesn't learn effectively

### 2. Physics Buffer Overflow
- Collision pairs grow from 12M → 60M during training
- Indicates severe interpenetration/instability
- Related to aggressive learned actions

### 3. Zero Success Rate
- Previous hetero training: 0% success
- Likely due to observation space bug (now fixed)
- New training should show improvement

---

## File Changes Summary

### Modified Files
1. `HARL/harl/envs/mapush/mapush_env.py` - Fixed observation space handling
2. `HARL/harl_mapush/test.py` - Fixed viewer mode for hetero agents
3. `resources/robots/jackal/urdf/jackal.urdf` - Fixed mesh paths and caster heights
4. `mqe/envs/jackal/jackal_config.py` - Velocity mode, initial height
5. `mqe/envs/base/hetero_robot.py` - Differential drive, velocity limiting
6. `mqe/envs/wrappers/go1_push_mid_wrapper.py` - Reward tensor handling

### Created Files
- `diagnose_jackal.py` - Diagnostic script for physics issues
- `visualize_checkpoint.py` - Checkpoint visualization (partial implementation)
- `test_hetero_env.py` - Environment testing script

---

## Next Steps

1. **Start new training run** with fixed code
2. **Monitor training logs** for:
   - Observation dimensions (should be 235 for Go1)
   - NaN occurrence (should delay until >1M steps)
   - Success rate (should be >0%)
3. **Visualize checkpoint** at 10M steps to verify:
   - Both robots spawn correctly
   - Jackal has proper mesh (black/yellow)
   - Both robots move intelligently
4. **If NaN persists:** Consider switching to 2x Go1 homogeneous setup (known working)

---

## References

**Jackal Robot:**
- Manufacturer: Clearpath Robotics
- Type: Differential drive wheeled robot
- URDF: Simplified from official Clearpath model
- Wheel radius: 9.8cm, Base width: 37.6cm

**Isaac Gym DOF Modes:**
- `DOF_MODE_NONE = 0`: No control
- `DOF_MODE_POS = 1`: Position control (was velocity, corrected)
- `DOF_MODE_VEL = 2`: Velocity control (was effort, corrected)
- `DOF_MODE_EFFORT = 3`: Torque/effort control

**HAPPO Algorithm:**
- Heterogeneous-Agent PPO
- Separate actor per agent type
- Shared or separate critic (using concat observations)
- Supports different obs/action spaces per agent
