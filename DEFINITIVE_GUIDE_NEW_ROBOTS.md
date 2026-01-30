# Definitive Guide: Adding New Robots to MAPush Heterogeneous Environment

**Last Updated:** 2026-01-30
**Based on:** Go1, Anymal C, and Jackal integrations

This is the authoritative reference for adding new robot types to the MAPush heterogeneous multi-agent environment. It consolidates lessons learned from multiple robot integrations.

---

## Table of Contents

1. [Prerequisites](#prerequisites)
2. [Architecture Overview](#architecture-overview)
3. [Step-by-Step Implementation](#step-by-step-implementation)
4. [Critical Configuration Parameters](#critical-configuration-parameters)
5. [Locomotion Policy Integration](#locomotion-policy-integration)
6. [Actuator Network Integration](#actuator-network-integration)
7. [Reset Handling](#reset-handling)
8. [Testing & Validation](#testing--validation)
9. [Common Pitfalls & Solutions](#common-pitfalls--solutions)
10. [File Locations Reference](#file-locations-reference)
11. [Complete Checklist](#complete-checklist)

---

## Prerequisites

Before adding a new robot, you need:

1. **URDF file** - Robot description with correct joint definitions
2. **Trained locomotion policy** - Converts `[vx, vy, vyaw]` commands to joint targets
3. **Actuator network** (if applicable) - Converts joint position errors to torques
4. **Training configuration** - Know the exact parameters used to train the policy

**Critical:** The observation structure and scaling MUST match exactly what was used during locomotion policy training.

---

## Architecture Overview

### Control Hierarchy

```
High-Level Actions [vx, vy, vyaw]
         │
         ▼
┌─────────────────────────────────┐
│     Locomotion Policy           │  ← Converts velocity commands to joint targets
│  (robot-specific, pre-trained)  │
└─────────────────────────────────┘
         │
         ▼
   Joint Position Targets [num_dof]
         │
         ▼
┌─────────────────────────────────┐
│     Actuator Network            │  ← Converts position errors to torques
│  (robot-specific, pre-trained)  │
└─────────────────────────────────┘
         │
         ▼
      Joint Torques [num_dof]
         │
         ▼
      Isaac Gym Physics
```

### Robot Types Currently Supported

| Robot | Type | DOFs | Control | Actuator Network |
|-------|------|------|---------|------------------|
| Go1 | Quadruped | 12 | Hierarchical (C) | Feedforward (`unitree_go1.pt`) |
| Anymal C | Quadruped | 12 | Hierarchical (C) | LSTM (`anydrive_v3_lstm.pt`) |
| Jackal | Wheeled | 2 | Differential Drive (P) | None (velocity control) |

---

## Step-by-Step Implementation

### Step 1: Prepare Robot Assets

Create directory structure:
```
resources/robots/<robot_name>/
├── urdf/
│   └── <robot_name>.urdf
├── meshes/
│   └── *.stl (or *.obj, *.dae)
└── policy_<version>.jit/
    └── policy_1.pt (or body_latest.jit + adaptation_module_latest.jit)
```

**URDF Checklist:**
- [ ] Mesh paths are correct (use relative paths like `../meshes/file.stl`)
- [ ] Joint names match what locomotion policy expects
- [ ] `collapse_fixed_joints` setting matches training
- [ ] Collision geometries are reasonable

### Step 2: Create Robot Configuration

Create `mqe/envs/<robot_name>/<robot_name>_config.py`:

```python
from mqe.envs.field.legged_robot_field_config import LeggedRobotFieldCfg

class NewRobotCfg(LeggedRobotFieldCfg):

    class asset:
        file = "{LEGGED_GYM_ROOT_DIR}/resources/robots/<robot_name>/urdf/<robot_name>.urdf"
        name = "<robot_name>"
        foot_name = "FOOT"  # Name of foot bodies in URDF (case-sensitive!)
        penalize_contacts_on = ["SHANK", "THIGH"]  # MUST match training config
        terminate_after_contacts_on = ["base"]
        collapse_fixed_joints = True  # MUST match training config
        flip_visual_attachments = True  # MUST match training config
        default_dof_drive_mode = 3  # 3=effort for actuator network control
        self_collisions = 1  # 1=disabled, 0=enabled - MUST match training

    class init_state(LeggedRobotFieldCfg.init_state):
        pos = [0.0, 0.0, 0.5]  # [x, y, z] - z is spawn height
        default_joint_angles = {
            # MUST match training config EXACTLY
            # Use URDF joint names
            'joint_name_1': 0.0,
            'joint_name_2': 0.4,
            # ... all joints
        }

    class control(LeggedRobotFieldCfg.control):
        control_type = 'C'  # 'C' = hierarchical with locomotion policy

        # PD gains - use pattern matching for joint names
        # The key is a substring that appears in joint names
        stiffness = {'_': 80.}  # e.g., '_' matches 'LF_HAA', 'RF_HFE', etc.
        damping = {'_': 2.0}

        # CRITICAL: Must match training config
        action_scale = 0.5

        # Torque limits per joint [j0, j1, j2, ...] * num_legs
        torque_limits = [80., 80., 80.] * 4

        # Only add if robot was trained with this
        # hip_scale_reduction = 0.5  # Go1 uses this, Anymal C does NOT

        # Policy and actuator paths
        locomotion_policy_dir = "./resources/robots/<robot_name>/policy_500.jit"
        actuator_network_path = "./resources/actuator_nets"

        # Decimation (physics steps per control step)
        decimation = 4  # MUST match training config
```

### Step 3: Register the Robot

Add to `mqe/envs/robot_registry.py`:

```python
ROBOT_REGISTRY = {
    # ... existing robots ...

    '<robot_name>': {
        'class_path': 'mqe.envs.<robot_name>.<robot_name>.NewRobot',
        'config_path': 'mqe.envs.<robot_name>.<robot_name>_config.NewRobotCfg',
        'default_control': 'C',  # 'C' for hierarchical, 'P' for direct
        'num_actions': 3,  # [vx, vy, vyaw]
        'description': 'Description of the robot'
    },
}
```

### Step 4: Add Locomotion Policy Loading

In `mqe/envs/base/hetero_robot.py`, add to `_load_locomotion_policies()`:

```python
elif robot_type == '<robot_name>':
    # Load policy (adapt based on policy architecture)
    policy_model = torch.jit.load(policy_dir + '/policy_1.pt', map_location=self.device)

    def new_robot_policy(obs, info={}):
        with torch.no_grad():
            action = policy_model.forward(obs)
        return action

    self.locomotion_policies.append(new_robot_policy)

    # Create observation buffer with correct size
    # OBS_DIM must match training exactly!
    obs_buffer = torch.zeros(self.num_envs, OBS_DIM, dtype=torch.float, device=self.device)
    self.locomotion_obs_buffers.append({'obs': obs_buffer, 'history': None})

    # Load actuator network
    if hasattr(robot_config.control, 'actuator_network_path'):
        self._load_actuator_network(robot_config.control.actuator_network_path, robot_type, agent_idx)
```

### Step 5: Add Observation Construction

In `mqe/envs/base/hetero_robot.py`, add to `step()` function:

```python
elif robot_type == '<robot_name>':
    loc_obs = obs_buffer_dict['obs']

    # Fill observation according to training config
    # SCALING MUST MATCH TRAINING EXACTLY!
    loc_obs[:, 0:3] = self.obs_buf.lin_vel[agent_env_indices] * lin_vel_scale
    loc_obs[:, 3:6] = self.obs_buf.ang_vel[agent_env_indices] * ang_vel_scale
    loc_obs[:, 6:9] = self.obs_buf.projected_gravity[agent_env_indices]
    loc_obs[:, 9:12] = agent_actions * self.commands_scale
    loc_obs[:, 12:24] = self.obs_buf.dof_pos[agent_env_indices] * dof_pos_scale
    loc_obs[:, 24:36] = self.obs_buf.dof_vel[agent_env_indices] * dof_vel_scale

    # Previous actions (if policy expects them)
    if 'last_joint_targets' in obs_buffer_dict:
        loc_obs[:, 36:48] = obs_buffer_dict['last_joint_targets']
    else:
        loc_obs[:, 36:48] = torch.zeros(self.num_envs, 12, device=self.device)

    # Call policy
    joint_positions = locomotion_policy(loc_obs)

    # Store for next step
    obs_buffer_dict['last_joint_targets'] = joint_positions.clone()
```

### Step 6: Add Actuator Network Loading

In `mqe/envs/base/hetero_robot.py`, add to `_load_actuator_network()`:

```python
elif robot_type == '<robot_name>':
    actuator_file = actuator_network_path + "/<robot_name>_actuator.pt"
    actuator_net = torch.jit.load(actuator_file, map_location=self.device)

    # For LSTM networks: initialize hidden states
    if is_lstm:
        num_dof = 12
        sea_hidden = torch.zeros(2, self.num_envs * num_dof, 8, device=self.device)
        sea_cell = torch.zeros(2, self.num_envs * num_dof, 8, device=self.device)
        setattr(self, f'<robot>_sea_hidden_{agent_idx}', sea_hidden)
        setattr(self, f'<robot>_sea_cell_{agent_idx}', sea_cell)

    def eval_new_robot_actuator(joint_pos_err, joint_pos_err_last, joint_pos_err_last_last,
                                 joint_vel, joint_vel_last, joint_vel_last_last):
        # CRITICAL: Check sign convention!
        # hetero_robot.py computes: actual - target
        # If training used: target - actual, you must NEGATE

        # For feedforward networks (like Go1):
        xs = torch.cat((joint_pos_err.unsqueeze(-1),
                       joint_pos_err_last.unsqueeze(-1),
                       joint_pos_err_last_last.unsqueeze(-1),
                       joint_vel.unsqueeze(-1),
                       joint_vel_last.unsqueeze(-1),
                       joint_vel_last_last.unsqueeze(-1)), dim=-1)
        with torch.no_grad():
            torques = actuator_net(xs.view(-1, 6))
        return torques.view(num_envs, num_dof)

        # For LSTM networks (like Anymal C):
        # sea_input[:, 0, 0] = -joint_pos_err.flatten()  # NEGATED!
        # sea_input[:, 0, 1] = joint_vel.flatten()
        # torques, (h, c) = actuator_net(sea_input, (hidden, cell))

    self.actuator_networks[agent_idx] = eval_new_robot_actuator
```

### Step 7: Add Reset Handling

In `mqe/envs/base/hetero_robot.py`, add to `_reset_hetero_buffers()`:

```python
if robot_type == '<robot_name>':
    # Reset LSTM hidden states (if applicable)
    if hasattr(self, f'<robot>_sea_hidden_{agent_idx}'):
        sea_hidden = getattr(self, f'<robot>_sea_hidden_{agent_idx}').clone()
        sea_cell = getattr(self, f'<robot>_sea_cell_{agent_idx}').clone()

        for env_id in env_ids:
            start_idx = env_id * num_dof_agent
            end_idx = (env_id + 1) * num_dof_agent
            sea_hidden[:, start_idx:end_idx, :] = 0.0
            sea_cell[:, start_idx:end_idx, :] = 0.0

        setattr(self, f'<robot>_sea_hidden_{agent_idx}', sea_hidden)
        setattr(self, f'<robot>_sea_cell_{agent_idx}', sea_cell)
```

**IMPORTANT:** Use `.clone()` before modifying LSTM states to avoid "Inplace update to inference tensor" error!

---

## Critical Configuration Parameters

### Parameters That MUST Match Training

| Parameter | Where to Find | Impact if Wrong |
|-----------|---------------|-----------------|
| `action_scale` | Training config | Robot moves too much/little |
| `default_joint_angles` | Training config | Robot falls immediately |
| `stiffness` / `damping` | Training config (PD gains) | Unstable joint control |
| `decimation` | Training config | Wrong control frequency |
| `hip_scale_reduction` | Training config | Erratic hip movement |
| `collapse_fixed_joints` | Training config | Wrong body/joint count |
| Observation scaling | Training code | Policy outputs garbage |
| Position error sign | Training code | Torques explode |

### Observation Scaling Reference

**Go1 (Walk-These-Ways):**
```python
lin_vel_scale = 2.0
ang_vel_scale = 0.25
dof_pos_scale = 1.0
dof_vel_scale = 0.05
commands_scale = [2.0, 2.0, 0.25]  # [vx, vy, vyaw]
```

**Anymal C (legged_gym standard):**
```python
lin_vel_scale = 2.0
ang_vel_scale = 0.25
dof_pos_scale = 1.0
dof_vel_scale = 0.05
commands_scale = [2.0, 2.0, 0.25]
# Note: projected_gravity is NOT scaled
```

---

## Locomotion Policy Integration

### Policy Type A: Walk-These-Ways (Go1)

**Files:** `body_latest.jit`, `adaptation_module_latest.jit`
**Observation:** 70 dims + 2100 history buffer

```
Observation Structure:
[0:3]   projected_gravity
[3:6]   velocity commands * [2.0, 2.0, 0.25]
[6]     body_height * 2.0
[7]     gait_freq * 1.0
[8:12]  gait params [phase, offset, bound, duration]
[12]    footswing_height * 0.15
[13:15] body_pitch, body_roll
[15]    stance_width
[16]    stance_length
[17]    aux_reward
[18:30] dof_pos (error from default) * 1.0
[30:42] dof_vel * 0.05
[42:54] last_action (t-1)
[54:66] last_two_action (t-2)
[66:70] clock_inputs (sinusoidal gait phase)
```

**Policy Call:**
```python
latent = adaptation_module.forward(obs_70)
history_buffer = update_history(history_buffer, obs_70)  # 2100 dims
action = body.forward(torch.cat([history_buffer, latent], dim=-1))
```

### Policy Type B: Standard legged_gym (Anymal C)

**Files:** `policy_1.pt` (single JIT file)
**Observation:** 48 dims, no history

```
Observation Structure:
[0:3]   base_lin_vel * 2.0
[3:6]   base_ang_vel * 0.25
[6:9]   projected_gravity (NOT scaled)
[9:12]  commands * [2.0, 2.0, 0.25]
[12:24] dof_pos (error from default) * 1.0
[24:36] dof_vel * 0.05
[36:48] previous_actions (raw policy outputs)
```

**Policy Call:**
```python
action = policy_model.forward(obs_48)
```

---

## Actuator Network Integration

### Type A: Feedforward (Go1 - unitree_go1.pt)

- **Input:** 6 values per joint `[pos_err, pos_err_t-1, pos_err_t-2, vel, vel_t-1, vel_t-2]`
- **Sign convention:** `pos_err = actual - target`
- **Shape:** `[num_envs * num_dof, 6]` → `[num_envs * num_dof, 1]`
- **No state:** Stateless, no reset needed

### Type B: LSTM (Anymal C - anydrive_v3_lstm.pt)

- **Input:** 2 values per joint `[pos_err, vel]`
- **Sign convention:** `pos_err = target - actual` (OPPOSITE of Go1!)
- **Shape:** `[num_envs * num_dof, 1, 2]` → `[num_envs * num_dof, 1]`
- **Hidden state:** `(h, c)` of shape `[2, num_envs * num_dof, 8]`
- **MUST reset** hidden states on episode termination!

### Position Error Sign Convention

| Source | Formula | When to Negate |
|--------|---------|----------------|
| hetero_robot.py | `actual - target` | Base convention |
| Go1 actuator | `actual - target` | No negation needed |
| Anymal C actuator | `target - actual` | MUST negate! |

**How to check:** Look at the training code where actuator network was trained.

---

## Reset Handling

Episode resets require clearing:

1. **LSTM hidden states** (for LSTM-based actuator networks)
2. **Observation buffers** (`last_action`, `last_joint_targets`, `history`)
3. **Actuator history** (`joint_pos_err_last`, `joint_vel_last`)
4. **Gait indices** (for robots with clock-based gaits)

**Critical:** Use `.clone()` when modifying tensors that may be in inference mode:

```python
# WRONG - will crash with "Inplace update to inference tensor"
sea_hidden[:, start_idx:end_idx, :] = 0.0

# CORRECT - clone first
sea_hidden = sea_hidden.clone()
sea_hidden[:, start_idx:end_idx, :] = 0.0
```

---

## Testing & Validation

### Test 1: Standalone Standing Test

```python
# test_standing.py
from mqe.envs.utils import make_hetero_env

env, _ = make_hetero_env('go1push_mid', ['go1', '<new_robot>'], args)
obs = env.reset()

# Send zero velocity commands
actions = torch.zeros(env.num_envs, env.num_agents, 3, device='cuda')
for _ in range(200):
    obs, rewards, dones, infos = env.step(actions)
    if dones.any():
        print("Robot fell!")
        break
else:
    print("Robot can stand!")
```

### Test 2: Movement Test

```python
# Test forward movement
actions[:, :, 0] = 0.5  # vx = 0.5 m/s
for _ in range(200):
    obs, rewards, dones, infos = env.step(actions)
```

### Test 3: Full Training Test

```bash
python HARL/harl_mapush/train.py \
    --exp_name test_new_robot \
    --hetero_agent <new_robot> \
    --n_rollout_threads 50 \
    --num_env_steps 10000
```

---

## Common Pitfalls & Solutions

### Robot Falls Immediately

| Cause | Solution |
|-------|----------|
| Wrong `default_joint_angles` | Copy exactly from training config |
| Wrong `action_scale` | Check training config |
| Wrong spawn height | Adjust `init_state.pos[2]` |
| Wrong sign convention | Check and negate if needed |

### Robot Moves Erratically

| Cause | Solution |
|-------|----------|
| Observation structure mismatch | Match training exactly |
| Missing observation scaling | Add correct scaling factors |
| Wrong `hip_scale_reduction` | Only use if trained with it |
| Wrong `decimation` | Match training config |

### Robot Doesn't Respond to Commands

| Cause | Solution |
|-------|----------|
| Commands not scaled | Apply `commands_scale = [2.0, 2.0, 0.25]` |
| Policy input dimension mismatch | Check obs buffer size |
| Wrong policy file | Verify correct checkpoint |

### Torques Explode / NaN

| Cause | Solution |
|-------|----------|
| Wrong sign convention | Negate position error |
| Missing torque clipping | Add `torque_limits` to config |
| LSTM states not reset | Add reset logic for LSTM |
| Missing `.clone()` | Clone before in-place modification |

### Training Crashes on Reset

| Cause | Solution |
|-------|----------|
| Inference mode tensor modification | Use `.clone()` before modifying |
| Missing reset handler | Add robot to `_reset_hetero_buffers()` |

---

## File Locations Reference

| Component | Location |
|-----------|----------|
| Robot config | `mqe/envs/<robot>/<robot>_config.py` |
| Robot registry | `mqe/envs/robot_registry.py` |
| Hetero robot logic | `mqe/envs/base/hetero_robot.py` |
| URDF files | `resources/robots/<robot>/urdf/` |
| Locomotion policies | `resources/robots/<robot>/` |
| Actuator networks | `resources/actuator_nets/` |
| Training script | `HARL/harl_mapush/train.py` |
| Environment wrapper | `HARL/harl/envs/mapush/mapush_env.py` |

---

## Complete Checklist

### Pre-Implementation
- [ ] Have URDF file with correct joint names
- [ ] Have trained locomotion policy
- [ ] Know the exact training configuration
- [ ] Know the observation structure and scaling
- [ ] Know the position error sign convention

### Implementation
- [ ] URDF placed in `resources/robots/<name>/urdf/`
- [ ] Mesh paths in URDF are correct
- [ ] Config file created at `mqe/envs/<name>/<name>_config.py`
- [ ] `default_joint_angles` match training EXACTLY
- [ ] `action_scale` matches training
- [ ] `stiffness` and `damping` match training
- [ ] `torque_limits` set correctly
- [ ] `decimation` matches training
- [ ] `hip_scale_reduction` only if trained with it
- [ ] Robot registered in `robot_registry.py`
- [ ] Policy loading added to `_load_locomotion_policies()`
- [ ] Observation construction added to `step()`
- [ ] Observation scaling matches training EXACTLY
- [ ] Actuator network loading added to `_load_actuator_network()`
- [ ] Position error sign convention correct
- [ ] Reset handling added to `_reset_hetero_buffers()`
- [ ] `.clone()` used for LSTM state modifications

### Testing
- [ ] Robot can stand with zero commands
- [ ] Robot moves forward with positive vx
- [ ] Robot turns with non-zero vyaw
- [ ] No crashes during episode resets
- [ ] Training runs without NaN
- [ ] No "inference tensor" errors

---

## Quick Reference: Adding a legged_gym-style Robot

If your robot was trained with standard legged_gym (like Anymal C), follow this minimal path:

1. Copy `anymal_c_config.py` → `<robot>_config.py`
2. Update: `file`, `name`, `foot_name`, `default_joint_angles`, `torque_limits`, `locomotion_policy_dir`
3. Add robot to `robot_registry.py`
4. In `hetero_robot.py`:
   - Copy Anymal C policy loading block
   - Copy Anymal C observation construction block
   - Copy Anymal C actuator network loading (if using LSTM)
   - Copy Anymal C reset handling
5. Update observation dimensions if different from 48
6. Test standing, movement, and training

---

**Document Version:** 1.0
**Based on integrations:** Go1, Anymal C, Jackal
