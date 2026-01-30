# Guide for Adding New Robots to MAPush Heterogeneous Environment

This guide documents all the critical details needed to add a new robot type to the MAPush heterogeneous multi-agent environment.

## Overview

The heterogeneous MAPush system allows different robot types (e.g., Go1, Anymal C) to work together in the same environment. Each robot uses:
1. A **locomotion policy** - converts velocity commands [vx, vy, vyaw] to joint position targets
2. An **actuator network** - converts joint position errors to torques

---

## Step 1: Create Robot Configuration

Create a new config file at `mqe/envs/<robot_name>/<robot_name>_config.py`:

```python
from mqe.envs.field.legged_robot_field_config import LeggedRobotFieldCfg

class NewRobotCfg(LeggedRobotFieldCfg):

    class asset:
        file = "{LEGGED_GYM_ROOT_DIR}/resources/robots/<robot_name>/urdf/<robot>.urdf"
        name = "<robot_name>"
        foot_name = "foot"  # Name of foot bodies in URDF
        penalize_contacts_on = ["base", "thigh"]
        terminate_after_contacts_on = ["base"]
        # ... other asset options

    class init_state(LeggedRobotFieldCfg.init_state):
        pos = [0.0, 0.0, 0.5]  # Initial height
        default_joint_angles = {
            'joint1': 0.0,
            'joint2': 0.8,
            # ... all joints with their default angles
        }

    class control(LeggedRobotFieldCfg.control):
        control_type = 'C'  # Hierarchical control
        stiffness = {'joint': 20.}  # PD gains (for fallback)
        damping = {'joint': 0.5}
        action_scale = 0.25  # MUST match training config
        torque_limits = [20., 20., 25.] * 4  # Per-joint limits

        # CRITICAL: Path to trained locomotion policy
        locomotion_policy_dir = "./path/to/policy"
        actuator_network_path = "./resources/actuator_nets"

        # Only if the robot was trained with hip_scale_reduction
        # hip_scale_reduction = 0.5  # Go1 uses this, Anymal C does NOT
```

### Critical Config Parameters

| Parameter | Description | How to Find |
|-----------|-------------|-------------|
| `action_scale` | Scales policy outputs before adding to defaults | Check training config |
| `hip_scale_reduction` | Extra scaling for hip joints (indices 0,3,6,9) | Check if used during training |
| `default_joint_angles` | Rest pose joint positions | From URDF or training config |
| `torque_limits` | Max torque per joint | From robot specs/URDF |

---

## Step 2: Register the Robot

Add the robot to `mqe/envs/robot_registry.py`:

```python
ROBOT_REGISTRY = {
    'go1': {
        'config': 'mqe.envs.go1.go1_config.Go1Cfg',
        'asset_path': '{LEGGED_GYM_ROOT_DIR}/resources/robots/go1/urdf/go1.urdf',
        'num_dof': 12,
        'action_dim': 3,  # [vx, vy, vyaw]
        'default_control': 'C',
    },
    'anymal_c': {
        'config': 'mqe.envs.anymal_c.anymal_c_config.AnymalCCfg',
        'asset_path': '{LEGGED_GYM_ROOT_DIR}/resources/robots/anymal_c/urdf/anymal_c.urdf',
        'num_dof': 12,
        'action_dim': 3,
        'default_control': 'C',
    },
    # ADD YOUR NEW ROBOT HERE
    'new_robot': {
        'config': 'mqe.envs.new_robot.new_robot_config.NewRobotCfg',
        'asset_path': '{LEGGED_GYM_ROOT_DIR}/resources/robots/new_robot/urdf/new_robot.urdf',
        'num_dof': 12,  # Number of actuated DOFs
        'action_dim': 3,
        'default_control': 'C',
    },
}
```

---

## Step 3: Locomotion Policy Integration

### Policy Types

Different robots may use different policy architectures:

#### Type A: Walk-These-Ways (Go1)
- **Files**: `body_latest.jit`, `adaptation_module_latest.jit`
- **Observation**: 70 dims + 2100 history buffer
- **Observation structure**:
  ```
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
- **Policy call**: `policy(history_2100_dims)`

#### Type B: Standard legged_gym (Anymal C)
- **Files**: `policy_1.pt` (single JIT file)
- **Observation**: 48 dims, no history
- **Observation structure**:
  ```
  [0:3]   base_lin_vel * 2.0
  [3:6]   base_ang_vel * 0.25
  [6:9]   projected_gravity (NOT scaled)
  [9:12]  commands * [2.0, 2.0, 0.25]
  [12:24] dof_pos (error from default) * 1.0
  [24:36] dof_vel * 0.05
  [36:48] previous_actions (raw policy outputs)
  ```
- **Policy call**: `policy(obs_48_dims)`

### Adding Policy Loading in hetero_robot.py

In `_load_locomotion_policies()`, add a new branch:

```python
elif robot_type == 'new_robot':
    # Load policy
    policy_model = torch.jit.load(policy_dir + '/policy.pt', map_location=self.device)

    def new_robot_policy(obs, info={}):
        with torch.no_grad():
            action = policy_model.forward(obs)
        return action

    self.locomotion_policies.append(new_robot_policy)

    # Create observation buffer with correct size
    obs_buffer = torch.zeros(self.num_envs, OBS_DIM, dtype=torch.float, device=self.device)
    self.locomotion_obs_buffers.append({'obs': obs_buffer, 'history': None})
```

### Adding Observation Construction in step()

In the `step()` function, add observation construction:

```python
elif robot_type == 'new_robot':
    loc_obs = obs_buffer_dict['obs']

    # Fill observation according to training config
    loc_obs[:, 0:3] = self.obs_buf.lin_vel[agent_env_indices]
    loc_obs[:, 3:6] = self.obs_buf.ang_vel[agent_env_indices]
    # ... etc

    joint_positions = locomotion_policy(loc_obs)
    obs_buffer_dict['last_action'] = joint_positions.clone()
```

---

## Step 4: Actuator Network Integration

### Actuator Network Types

#### Type A: Feedforward (Go1 - unitree_go1.pt)
- **Input**: 6 values per joint `[pos_err, pos_err_t-1, pos_err_t-2, vel, vel_t-1, vel_t-2]`
- **Sign convention**: `pos_err = actual - target`
- **Shape**: `[num_envs * num_dof, 6]` -> `[num_envs * num_dof, 1]`

#### Type B: LSTM (Anymal C - anydrive_v3_lstm.pt)
- **Input**: 2 values per joint `[pos_err, vel]`
- **Sign convention**: `pos_err = target - actual` (OPPOSITE of Go1!)
- **Shape**: `[num_envs * num_dof, 1, 2]` -> `[num_envs * num_dof, 1]`
- **Requires**: Hidden state `(h, c)` of shape `[2, num_envs * num_dof, 8]`

### Adding Actuator Network in hetero_robot.py

In `_load_actuator_network()`:

```python
elif robot_type == 'new_robot':
    actuator_file = actuator_network_path + "/new_robot_actuator.pt"
    actuator_net = torch.jit.load(actuator_file, map_location=self.device)

    # Check the sign convention used during training!
    # legged_gym typically uses: target - actual
    # Some implementations use: actual - target

    def eval_new_robot_actuator(joint_pos_err, joint_pos_err_last, joint_pos_err_last_last,
                                 joint_vel, joint_vel_last, joint_vel_last_last):
        # Implement based on actuator network architecture
        # ...
        return torques

    self.actuator_networks[agent_idx] = eval_new_robot_actuator
```

### CRITICAL: Position Error Sign Convention

| Source | Formula | When to Negate |
|--------|---------|----------------|
| hetero_robot.py | `actual - target` | Base convention |
| Go1 actuator | `actual - target` | No negation needed |
| Anymal C actuator | `target - actual` | MUST negate! |

Check your training code to determine which convention was used!

---

## Step 5: Torque Computation

The `_compute_torques()` function handles per-robot torque computation:

1. Joint targets are computed: `target = policy_output * action_scale + default_pos`
2. Position error: `error = actual_pos - target` (may need negation for some actuators)
3. Actuator network converts error + velocity to torques
4. Torques are clipped to `torque_limits`

### Hip Scale Reduction

Go1 uses `hip_scale_reduction = 0.5` which reduces action scale for hip joints (indices 0, 3, 6, 9). This is applied AFTER `action_scale`:

```python
if hasattr(robot_config.control, 'hip_scale_reduction') and num_dof_agent == 12:
    hip_scale = robot_config.control.hip_scale_reduction
    joint_residuals_scaled[:, [0, 3, 6, 9]] *= hip_scale
```

**Only add this if your robot was trained with it!**

---

## Step 6: Reset Handling

For LSTM-based actuator networks, hidden states must be reset when environments reset. Add to reset logic:

```python
# Reset LSTM hidden states for Anymal C
if hasattr(self, f'anymal_sea_hidden_{agent_idx}'):
    self.anymal_sea_hidden_{agent_idx}[:, env_ids] = 0.
    self.anymal_sea_cell_{agent_idx}[:, env_ids] = 0.
```

---

## Checklist for New Robot

- [ ] URDF file placed in `resources/robots/<name>/urdf/`
- [ ] Config file created with correct `default_joint_angles`
- [ ] Robot registered in `robot_registry.py`
- [ ] Locomotion policy files copied to specified path
- [ ] Actuator network file available (or use existing one if compatible)
- [ ] `action_scale` matches training config
- [ ] `hip_scale_reduction` only if trained with it
- [ ] Observation structure matches training exactly
- [ ] Position error sign convention correct for actuator network
- [ ] Torque limits set correctly
- [ ] Policy loading added to `_load_locomotion_policies()`
- [ ] Actuator loading added to `_load_actuator_network()`
- [ ] Observation construction added to `step()`
- [ ] LSTM reset logic added (if applicable)

---

## Common Issues

### Robot falls immediately
- Check `default_joint_angles` match training config
- Verify `action_scale` is correct
- Check actuator network sign convention

### Robot moves erratically
- Observation structure doesn't match training
- Missing observation scaling (lin_vel * 2.0, ang_vel * 0.25, etc.)
- Wrong `hip_scale_reduction` setting

### Robot doesn't respond to commands
- Commands not scaled correctly (should be * [2.0, 2.0, 0.25])
- Policy input dimension mismatch

### Torques explode
- Wrong sign convention in actuator network
- Missing torque clipping
- LSTM hidden states not reset on episode reset

---

## File Locations Summary

| Component | Location |
|-----------|----------|
| Robot config | `mqe/envs/<robot>/` |
| Robot registry | `mqe/envs/robot_registry.py` |
| Hetero robot logic | `mqe/envs/base/hetero_robot.py` |
| URDF files | `resources/robots/<robot>/urdf/` |
| Locomotion policies | `resources/robots/<robot>/` or custom path |
| Actuator networks | `resources/actuator_nets/` |
