# Cassie Biped Robot Integration Notes

**Date:** 2026-02-01
**Status:** Integration complete, awaiting locomotion policy training

---

## Overview

Cassie is a bipedal robot from Agility Robotics. This document details its integration into the MAPush framework for heterogeneous multi-agent pushing tasks.

### Key Differences from Quadrupeds

| Aspect | Quadrupeds (Go1/Anymal C) | Cassie (Biped) |
|--------|---------------------------|----------------|
| Legs | 4 | 2 |
| DOFs | 12 (3 per leg) | 12 (6 per leg) |
| Spawn height | 0.42m / 0.62m | 1.0m |
| Stability | More stable | Less stable (narrower base) |
| Locomotion | Trotting gait | Walking/running gait |

---

## 1. Training Setup (legged_gym)

### 1.1 Created Flat Terrain Config

**File:** `/home/gvlab/legged_gym/legged_gym/envs/cassie/cassie_flat_config.py`

```python
from legged_gym.envs.cassie.cassie_config import CassieRoughCfg, CassieRoughCfgPPO

class CassieFlatCfg(CassieRoughCfg):
    class env(CassieRoughCfg.env):
        num_envs = 4096
        num_observations = 48  # Flat terrain (no height measurements)

    class terrain(CassieRoughCfg.terrain):
        mesh_type = 'plane'
        measure_heights = False

    class asset(CassieRoughCfg.asset):
        self_collisions = 0  # Enable self-collisions

    class rewards(CassieRoughCfg.rewards):
        max_contact_force = 350.
        class scales(CassieRoughCfg.rewards.scales):
            orientation = -5.0
            torques = -0.000025
            feet_air_time = 2.

    class commands(CassieRoughCfg.commands):
        heading_command = False
        resampling_time = 4.
        class ranges(CassieRoughCfg.commands.ranges):
            lin_vel_x = [-1.0, 1.0]
            lin_vel_y = [-0.5, 0.5]  # Reduced for biped stability
            ang_vel_yaw = [-1.0, 1.0]

    class domain_rand(CassieRoughCfg.domain_rand):
        friction_range = [0.5, 1.25]

class CassieFlatCfgPPO(CassieRoughCfgPPO):
    class policy(CassieRoughCfgPPO.policy):
        actor_hidden_dims = [128, 64, 32]
        critic_hidden_dims = [128, 64, 32]
        activation = 'elu'

    class algorithm(CassieRoughCfgPPO.algorithm):
        entropy_coef = 0.01

    class runner(CassieRoughCfgPPO.runner):
        run_name = ''
        experiment_name = 'flat_cassie'
        max_iterations = 500
```

### 1.2 Registered Task

**File:** `/home/gvlab/legged_gym/legged_gym/envs/__init__.py`

Added:
```python
from .cassie.cassie_flat_config import CassieFlatCfg, CassieFlatCfgPPO
task_registry.register("cassie_flat", Cassie, CassieFlatCfg(), CassieFlatCfgPPO())
```

### 1.3 Training Command

```bash
cd /home/gvlab/legged_gym
conda activate anymal_training  # or your legged_gym conda environment
python legged_gym/scripts/train.py --task=cassie_flat --headless
```

### 1.4 After Training

Copy the trained policy to MAPush:
```bash
cp /home/gvlab/legged_gym/logs/flat_cassie/<run_folder>/model_<iter>.pt \
   /home/gvlab/new-universal-MAPush/resources/robots/cassie/policy/policy_1.pt
```

---

## 2. MAPush Integration

### 2.1 Assets Copied

**Source:** `/home/gvlab/legged_gym/resources/robots/cassie/`
**Destination:** `/home/gvlab/new-universal-MAPush/resources/robots/cassie/`

```
resources/robots/cassie/
├── urdf/
│   └── cassie.urdf
├── meshes/
│   ├── pelvis.stl
│   ├── thigh.stl
│   ├── shin-bone.stl
│   ├── toe.stl
│   └── ... (22 mesh files total)
├── policy/
│   └── policy_1.pt  (TO BE ADDED after training)
└── cassie_license.txt
```

### 2.2 Configuration File

**File:** `/home/gvlab/new-universal-MAPush/mqe/envs/cassie/cassie_config.py`

Key parameters that MUST match training:

```python
class CassieCfg(LeggedRobotFieldCfg):
    class init_state:
        pos = [0.0, 0.0, 1.0]  # Spawn height 1.0m (biped is tall)
        default_joint_angles = {
            # Left leg
            'hip_abduction_left': 0.1,
            'hip_rotation_left': 0.,
            'hip_flexion_left': 1.,
            'thigh_joint_left': -1.8,
            'ankle_joint_left': 1.57,
            'toe_joint_left': -1.57,
            # Right leg
            'hip_abduction_right': -0.1,
            'hip_rotation_right': 0.,
            'hip_flexion_right': 1.,
            'thigh_joint_right': -1.8,
            'ankle_joint_right': 1.57,
            'toe_joint_right': -1.57,
        }

    class control:
        control_type = 'C'  # Hierarchical control
        action_scale = 0.5  # MUST match training
        stiffness = {
            'hip_abduction': 100.0,
            'hip_rotation': 100.0,
            'hip_flexion': 200.,
            'thigh_joint': 200.,
            'ankle_joint': 200.,
            'toe_joint': 40.
        }
        damping = {
            'hip_abduction': 3.0,
            'hip_rotation': 3.0,
            'hip_flexion': 6.,
            'thigh_joint': 6.,
            'ankle_joint': 6.,
            'toe_joint': 1.
        }
        decimation = 4
        locomotion_policy_dir = "./resources/robots/cassie/policy"
        actuator_network_path = None  # Uses PD control
```

### 2.3 Robot Registry Entry

**File:** `/home/gvlab/new-universal-MAPush/mqe/envs/robot_registry.py`

```python
ROBOT_REGISTRY = {
    # ... existing robots ...
    'cassie': {
        'class_path': 'mqe.envs.go1.go1.Go1',  # Uses Go1 base class
        'config_path': 'mqe.envs.cassie.cassie_config.CassieCfg',
        'default_control': 'C',
        'num_actions': 3,  # [vx, vy, vyaw]
        'description': 'Agility Robotics Cassie biped robot with trained locomotion policy'
    },
}
```

### 2.4 HeteroRobot Integration

**File:** `/home/gvlab/new-universal-MAPush/mqe/envs/base/hetero_robot.py`

#### Policy Loading (added to `_load_locomotion_policies`)

```python
elif robot_type == 'cassie':
    # Cassie uses standard legged_gym policy (48-dim obs)
    policy_model = torch.jit.load(policy_dir + '/policy_1.pt', map_location=self.device)

    def cassie_policy(obs, info={}):
        with torch.no_grad():
            action = policy_model.forward(obs)
        return action

    self.locomotion_policies.append(cassie_policy)
    obs_buffer = torch.zeros(self.num_envs, 48, dtype=torch.float, device=self.device)
    self.locomotion_obs_buffers.append({'obs': obs_buffer, 'history': None})
```

#### Actuator Network (PD Control Fallback)

```python
elif robot_type == 'cassie':
    # Cassie uses PD control (no actuator network)
    def eval_cassie_pd(joint_pos_err, ...):
        torques = (
            self.p_gains[dof_start:dof_end] * (-joint_pos_err)
            - self.d_gains[dof_start:dof_end] * joint_vel
        )
        return torques

    self.actuator_networks[agent_idx] = eval_cassie_pd
```

#### Observation Construction (in `step`)

```python
elif robot_type == 'cassie':
    # 48-dim observation (same structure as Anymal C)
    # [0:3] base_lin_vel, [3:6] base_ang_vel, [6:9] projected_gravity,
    # [9:12] commands, [12:24] dof_pos, [24:36] dof_vel, [36:48] previous actions
    loc_obs = obs_buffer_dict['obs']
    loc_obs[:, 0:3] = self.obs_buf.lin_vel[agent_env_indices]
    loc_obs[:, 3:6] = self.obs_buf.ang_vel[agent_env_indices]
    loc_obs[:, 6:9] = self.obs_buf.projected_gravity[agent_env_indices]
    loc_obs[:, 9:12] = agent_actions * self.commands_scale
    loc_obs[:, 12:24] = self.obs_buf.dof_pos[agent_env_indices]
    loc_obs[:, 24:36] = self.obs_buf.dof_vel[agent_env_indices]
    loc_obs[:, 36:48] = obs_buffer_dict.get('last_joint_targets', zeros)

    joint_positions = locomotion_policy(loc_obs)
    obs_buffer_dict['last_joint_targets'] = joint_positions.clone()
```

#### Reset Handling (in `_reset_hetero_buffers`)

```python
if robot_type == 'cassie':
    if self.locomotion_obs_buffers[agent_idx] is not None:
        obs_buffer_dict = self.locomotion_obs_buffers[agent_idx]
        if 'last_joint_targets' in obs_buffer_dict:
            obs_buffer_dict['last_joint_targets'][env_ids] = 0.0
```

---

## 3. Observation Structure

Cassie uses the same 48-dim observation as Anymal C (flat terrain legged_gym policy):

| Index | Dimension | Content | Scale |
|-------|-----------|---------|-------|
| 0-2 | 3 | base_lin_vel | 2.0 |
| 3-5 | 3 | base_ang_vel | 0.25 |
| 6-8 | 3 | projected_gravity | 1.0 |
| 9-11 | 3 | commands [vx, vy, vyaw] | [2.0, 2.0, 0.25] |
| 12-23 | 12 | dof_pos (scaled) | 1.0 |
| 24-35 | 12 | dof_vel (scaled) | 0.05 |
| 36-47 | 12 | previous_actions | 1.0 |

---

## 4. Joint Order

Cassie has 12 actuated DOFs in this order:

| Index | Joint Name | Type |
|-------|------------|------|
| 0 | hip_abduction_left | HAA |
| 1 | hip_rotation_left | HRO |
| 2 | hip_flexion_left | HFE |
| 3 | thigh_joint_left | KFE |
| 4 | ankle_joint_left | AFE |
| 5 | toe_joint_left | TFE |
| 6 | hip_abduction_right | HAA |
| 7 | hip_rotation_right | HRO |
| 8 | hip_flexion_right | HFE |
| 9 | thigh_joint_right | KFE |
| 10 | ankle_joint_right | AFE |
| 11 | toe_joint_right | TFE |

---

## 5. Usage Examples

### HAPPO (HARL) Training

```bash
# Go1 + Cassie heterogeneous
python HARL/harl_mapush/train.py \
    --agent0 go1 \
    --agent1 cassie \
    --exp_name go1_cassie_hetero \
    --n_rollout_threads 500 \
    --num_env_steps 100000000

# Cassie + Anymal C heterogeneous
python HARL/harl_mapush/train.py \
    --agent0 cassie \
    --agent1 anymal_c \
    --exp_name cassie_anymal_hetero

# Cassie homogeneous (2x Cassie)
python HARL/harl_mapush/train.py \
    --agent0 cassie \
    --agent1 cassie \
    --exp_name cassie_homo
```

### MAPPO (OpenRL) Training

```bash
# Go1 + Cassie
python ./openrl_ws/train.py \
    --agent0 go1 \
    --agent1 cassie \
    --algo ppo \
    --task go1push_mid \
    --config ./openrl_ws/cfgs/ppo.yaml \
    --use_tensorboard \
    --headless

# Cassie + Cassie
python ./openrl_ws/train.py \
    --agent0 cassie \
    --agent1 cassie \
    --algo ppo \
    --task go1push_mid
```

### Testing

```bash
# HAPPO testing
python HARL/harl_mapush/test.py \
    --checkpoint <path>/checkpoints/50M \
    --agent0 go1 \
    --agent1 cassie \
    --mode viewer

# MAPPO testing
python ./openrl_ws/test.py \
    --checkpoint <path>/checkpoints/rl_model_XXXXX_steps/module.pt \
    --agent0 go1 \
    --agent1 cassie \
    --test_mode viewer
```

---

## 6. Troubleshooting

### Robot Falls Over Immediately

- **Cause:** Locomotion policy not trained or wrong policy loaded
- **Fix:** Train Cassie policy using `cassie_flat` task first

### Wrong Joint Positions

- **Cause:** `action_scale` or `default_joint_angles` mismatch
- **Fix:** Verify values in `cassie_config.py` match legged_gym training config

### Robot Spawns Underground or in Air

- **Cause:** Wrong spawn height
- **Fix:** Ensure `init_state.pos[2] = 1.0` in config

### Torques Too Weak/Strong

- **Cause:** PD gains mismatch
- **Fix:** Verify `stiffness` and `damping` dicts match training config

### Observation Dimension Error

- **Cause:** Policy expects different observation size
- **Fix:** Verify policy was trained with 48-dim obs (flat terrain config)

---

## 7. Files Modified/Created

### Created Files

| File | Purpose |
|------|---------|
| `/home/gvlab/legged_gym/.../cassie_flat_config.py` | Training config |
| `/home/gvlab/new-universal-MAPush/mqe/envs/cassie/__init__.py` | Module init |
| `/home/gvlab/new-universal-MAPush/mqe/envs/cassie/cassie_config.py` | MAPush config |
| `/home/gvlab/new-universal-MAPush/resources/robots/cassie/` | Assets folder |

### Modified Files

| File | Changes |
|------|---------|
| `/home/gvlab/legged_gym/.../envs/__init__.py` | Added `cassie_flat` registration |
| `/home/gvlab/new-universal-MAPush/mqe/envs/robot_registry.py` | Added Cassie entry |
| `/home/gvlab/new-universal-MAPush/mqe/envs/base/hetero_robot.py` | Added Cassie handling |

---

## 8. TODO

1. [ ] Train Cassie locomotion policy using `cassie_flat` task
2. [ ] Copy trained policy to `resources/robots/cassie/policy/policy_1.pt`
3. [ ] Test Cassie in MAPush with viewer mode
4. [ ] Fine-tune PD gains if needed
5. [ ] Consider training with actuator network for better torque prediction

---

## 9. References

- Cassie URDF: `/home/gvlab/legged_gym/resources/robots/cassie/urdf/cassie.urdf`
- Original Cassie config: `/home/gvlab/legged_gym/legged_gym/envs/cassie/cassie_config.py`
- Anymal C integration (similar pattern): `/home/gvlab/new-universal-MAPush/ANYMAL_C_INTEGRATION_NOTES.md`
- Robot integration guide: `/home/gvlab/new-universal-MAPush/DEFINITIVE_GUIDE_NEW_ROBOTS.md`
