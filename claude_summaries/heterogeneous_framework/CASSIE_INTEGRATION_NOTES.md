# Cassie Biped Robot Integration Notes

**Date:** 2026-02-01
**Updated:** 2026-02-02
**Status:** ✅ COMPLETE - Locomotion policy trained and integrated

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
| Control | Actuator network | PD control |

---

## 1. Training Setup (legged_gym)

### 1.1 Training Configuration

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

### 1.2 Task Registration

**File:** `/home/gvlab/legged_gym/legged_gym/envs/__init__.py`

```python
from .cassie.cassie_flat_config import CassieFlatCfg, CassieFlatCfgPPO
task_registry.register("cassie_flat", Cassie, CassieFlatCfg(), CassieFlatCfgPPO())
```

### 1.3 Training Command

```bash
cd /home/gvlab/legged_gym
conda activate anymal_training
python legged_gym/scripts/train.py --task=cassie_flat --headless
```

### 1.4 Training Results (2026-02-02)

**Run folder:** `/home/gvlab/legged_gym/logs/flat_cassie/Feb02_11-59-29_/`

Training was run for 2000 iterations but **training collapsed after ~1200 iterations**.

#### Training Metrics Analysis

```
CASSIE TRAINING METRICS (2000 iterations)
======================================================================

Train/mean_reward:
  Step 0:    -0.1045
  Step 1250: 29.0157  <-- PEAK PERFORMANCE
  Step 1999: 11.2296  <-- Collapsed

Episode/rew_tracking_lin_vel:
  Step 0:    0.0047
  Step 1250: 0.8641   <-- PEAK
  Step 1999: 0.4615   <-- Degraded

Episode/rew_termination:
  Step 1250: -0.0028  <-- Minimal falls
  Step 1999: -0.0945  <-- More falls (collapsed)
```

**ASCII Training Curves:**
```
MEAN REWARD (peaked at ~30, collapsed to ~11)
│                                       ●
│           ●●●●●●   ●●●●●●●●●●●●●●●●●●● ●●●●
│  ●●●●●●●●●      ●●●                        │
│  │                                         ●  ●
│ ●                                                  ●●●●●●●●
│●
└────────────────────────────────────────────────────────────
 0                        1200                           2000
```

**Best model:** `model_1200.pt` (before collapse)

### 1.5 Policy Export

The raw checkpoint was converted to JIT format for MAPush inference:

```python
import torch
import torch.nn as nn

class Actor(nn.Module):
    def __init__(self, num_obs=48, num_actions=12):
        super().__init__()
        self.actor = nn.Sequential(
            nn.Linear(num_obs, 128),
            nn.ELU(),
            nn.Linear(128, 64),
            nn.ELU(),
            nn.Linear(64, 32),
            nn.ELU(),
            nn.Linear(32, num_actions)
        )

    def forward(self, obs):
        return self.actor(obs)

# Load checkpoint and export actor only
checkpoint = torch.load('model_1200.pt', map_location='cpu')
actor = Actor()
actor.load_state_dict({k: v for k, v in checkpoint['model_state_dict'].items() if k.startswith('actor.')})
actor.eval()

# Trace and save as JIT
traced_actor = torch.jit.trace(actor, torch.zeros(1, 48))
traced_actor.save('policy_1.pt')
```

---

## 2. MAPush Integration

### 2.1 Assets Structure

**Location:** `/home/gvlab/new-universal-MAPush/resources/robots/cassie/`

```
resources/robots/cassie/
├── urdf/
│   └── cassie.urdf
├── meshes/
│   ├── pelvis.stl
│   ├── abduction.stl
│   ├── abduction_mirror.stl
│   ├── hip.stl
│   ├── hip_mirror.stl
│   ├── achilles-rod.stl
│   ├── knee-output.stl
│   ├── knee-output_mirror.stl
│   ├── shin-bone.stl
│   ├── toe.stl
│   └── ... (22 mesh files total)
├── policy/
│   └── policy_1.pt  ✅ (JIT format, from model_1200)
└── cassie_license.txt
```

### 2.2 Robot Class

**File:** `/home/gvlab/new-universal-MAPush/mqe/envs/cassie/cassie.py`

Key implementation details:

```python
class Cassie(LeggedRobotField):
    """
    Cassie biped robot for MAPush.

    - 12 DOFs (6 per leg)
    - 48-dim observation (legged_gym standard)
    - PD control (no actuator network)
    - Spawn height: 1.0m
    """

    def preprocess_action(self, actions):
        """
        Observation structure (48 dims):
          [0:3]   base_lin_vel (scaled)
          [3:6]   base_ang_vel (scaled)
          [6:9]   projected_gravity
          [9:12]  commands [vx, vy, vyaw] (scaled)
          [12:24] dof_pos (relative to default, scaled)
          [24:36] dof_vel (scaled)
          [36:48] last_action
        """
        # Fill observations and call locomotion policy
        self.locomotion_obs[:, 0:3] = self.obs_buf.lin_vel
        self.locomotion_obs[:, 3:6] = self.obs_buf.ang_vel
        self.locomotion_obs[:, 6:9] = self.obs_buf.projected_gravity
        self.locomotion_obs[:, 12:24] = self.obs_buf.dof_pos
        self.locomotion_obs[:, 24:36] = self.obs_buf.dof_vel
        self.locomotion_obs[:, 36:48] = self.last_locomotion_action

        locomotion_action = self.locomotion_policy(self.locomotion_obs)
        return locomotion_action

    def _compute_torques(self, actions):
        """PD control (no actuator network)"""
        actions_scaled = actions * self.cfg.control.action_scale
        self.joint_pos_target = actions_scaled + self.default_dof_pos
        torques = self.p_gains * (self.joint_pos_target - self.dof_pos) - self.d_gains * self.dof_vel
        return torch.clip(torques, -self.torque_limits, self.torque_limits)
```

### 2.3 Configuration File

**File:** `/home/gvlab/new-universal-MAPush/mqe/envs/cassie/cassie_config.py`

Key parameters (MUST match training):

```python
class CassieCfg(LeggedRobotFieldCfg):
    class init_state:
        pos = [0.0, 0.0, 1.0]  # Spawn height 1.0m
        default_joint_angles = {
            'hip_abduction_left': 0.1,
            'hip_rotation_left': 0.,
            'hip_flexion_left': 1.,
            'thigh_joint_left': -1.8,
            'ankle_joint_left': 1.57,
            'toe_joint_left': -1.57,
            'hip_abduction_right': -0.1,
            'hip_rotation_right': 0.,
            'hip_flexion_right': 1.,
            'thigh_joint_right': -1.8,
            'ankle_joint_right': 1.57,
            'toe_joint_right': -1.57,
        }

    class control:
        control_type = 'C'
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

### 2.4 Robot Registry Entry

**File:** `/home/gvlab/new-universal-MAPush/mqe/envs/robot_registry.py`

```python
ROBOT_REGISTRY = {
    # ... existing robots ...
    'cassie': {
        'class_path': 'mqe.envs.cassie.cassie.Cassie',
        'config_path': 'mqe.envs.cassie.cassie_config.CassieCfg',
        'default_control': 'C',
        'num_actions': 3,  # [vx, vy, vyaw] mid-level
        'description': 'Agility Robotics Cassie biped robot with trained locomotion policy (12 DOF)'
    },
}
```

### 2.5 Module Init

**File:** `/home/gvlab/new-universal-MAPush/mqe/envs/cassie/__init__.py`

```python
from mqe.envs.cassie.cassie import Cassie
from mqe.envs.cassie.cassie_config import CassieCfg

__all__ = ["Cassie", "CassieCfg"]
```

---

## 3. Observation Structure

Cassie uses the standard 48-dim legged_gym observation (flat terrain):

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
cd /home/gvlab/new-universal-MAPush

# Go1 + Cassie heterogeneous
conda run -n mapush python HARL/harl_mapush/train.py \
    --agent0 go1 \
    --agent1 cassie \
    --exp_name go1_cassie_hetero \
    --n_rollout_threads 500 \
    --num_env_steps 100000000

# Cassie + Anymal C heterogeneous
conda run -n mapush python HARL/harl_mapush/train.py \
    --agent0 cassie \
    --agent1 anymal_c \
    --exp_name cassie_anymal_hetero

# Cassie homogeneous (2x Cassie)
conda run -n mapush python HARL/harl_mapush/train.py \
    --agent0 cassie \
    --agent1 cassie \
    --exp_name cassie_homo
```

### MAPPO (OpenRL) Training

```bash
cd /home/gvlab/new-universal-MAPush

# Go1 + Cassie
conda run -n mapush python openrl_ws/train.py \
    --agent0 go1 \
    --agent1 cassie \
    --algo ppo \
    --task go1push_mid \
    --use_tensorboard \
    --headless

# Cassie + Cassie
conda run -n mapush python openrl_ws/train.py \
    --agent0 cassie \
    --agent1 cassie \
    --algo ppo \
    --task go1push_mid
```

### Testing

```bash
# HAPPO testing
conda run -n mapush python HARL/harl_mapush/test.py \
    --checkpoint <path>/checkpoints/50M \
    --agent0 go1 \
    --agent1 cassie \
    --mode viewer

# MAPPO testing
conda run -n mapush python openrl_ws/test.py \
    --checkpoint <path>/checkpoints/rl_model_XXXXX_steps/module.pt \
    --agent0 go1 \
    --agent1 cassie \
    --test_mode viewer
```

### Visualize Locomotion Policy Only (legged_gym)

```bash
cd /home/gvlab/legged_gym
conda activate anymal_training
python legged_gym/scripts/play.py --task=cassie_flat
```

---

## 6. Troubleshooting

### Robot Falls Over Immediately

- **Cause:** Locomotion policy not loaded or wrong policy
- **Fix:** Verify `resources/robots/cassie/policy/policy_1.pt` exists and is JIT format

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

### "Policy not in JIT format" Error

- **Cause:** Raw checkpoint copied instead of exported JIT model
- **Fix:** Re-export using the JIT export script in Section 1.5

### Robot Jumps/Extends Legs and Falls (FIXED 2026-02-02)

- **Cause:** PD control sign error in `hetero_robot.py`
- **Symptom:** Cassie would jump, extend legs incorrectly, and fall within 2 seconds
- **Root Cause:** The PD control formula was computing `torque = Kp * (current - target)` instead of `torque = Kp * (target - current)`
- **Fix:** Updated `mqe/envs/base/hetero_robot.py` line ~1148 to use correct sign:
  ```python
  # BEFORE (wrong):
  joint_pos_err_agent = self.dof_pos - self.joint_pos_target
  torques = p_gains * joint_pos_err_agent - d_gains * vel  # Moves AWAY from target!

  # AFTER (correct):
  joint_pos_err_agent = self.joint_pos_target - self.dof_pos
  torques = p_gains * joint_pos_err_agent - d_gains * vel  # Moves TOWARD target
  ```
- **Note:** Actuator networks (Go1, Anymal C) use the opposite sign convention (`current - target`) because they were trained that way

---

## 7. Files Modified/Created

### Created Files

| File | Purpose |
|------|---------|
| `mqe/envs/cassie/cassie.py` | Cassie robot class |
| `mqe/envs/cassie/cassie_config.py` | MAPush configuration |
| `mqe/envs/cassie/__init__.py` | Module init |
| `resources/robots/cassie/policy/policy_1.pt` | JIT locomotion policy |

### Modified Files

| File | Changes |
|------|---------|
| `mqe/envs/robot_registry.py` | Updated Cassie to use `Cassie` class |
| `mqe/envs/base/hetero_robot.py` | Fixed PD control sign for Cassie (line ~1148) |

### legged_gym Files (Training Only)

| File | Purpose |
|------|---------|
| `legged_gym/envs/cassie/cassie_flat_config.py` | Flat terrain training config |
| `legged_gym/envs/__init__.py` | Task registration |

---

## 8. Completed Tasks

- [x] Train Cassie locomotion policy using `cassie_flat` task
- [x] Analyze training metrics and select best checkpoint (model_1200)
- [x] Export policy to JIT format
- [x] Copy trained policy to `resources/robots/cassie/policy/policy_1.pt`
- [x] Create `cassie.py` robot class
- [x] Update `__init__.py` with Cassie imports
- [x] Update robot registry with correct class path
- [x] Test Cassie import and policy loading

---

## 9. Future Improvements

1. **Retrain with better hyperparameters** - Current training collapsed after 1200 iterations
2. **Add actuator network** - Could improve torque prediction accuracy
3. **Domain randomization** - Add push perturbations for robustness
4. **Tune velocity ranges** - Current `lin_vel_y` range may be too aggressive for biped

---

## 10. References

- Cassie URDF: `/home/gvlab/legged_gym/resources/robots/cassie/urdf/cassie.urdf`
- Original Cassie config: `/home/gvlab/legged_gym/legged_gym/envs/cassie/cassie_config.py`
- Anymal C integration (similar pattern): `mqe/envs/anymal_c/`
- Training logs: `/home/gvlab/legged_gym/logs/flat_cassie/Feb02_11-59-29_/`

---

## 11. Integration Verification

```bash
# Test import (requires isaacgym)
conda run -n mapush python -c "
import isaacgym
from mqe.envs.robot_registry import get_robot_class, get_robot_config

cassie_class = get_robot_class('cassie')
cassie_config = get_robot_config('cassie')
cfg = cassie_config()

print(f'Class: {cassie_class}')
print(f'Spawn height: {cfg.init_state.pos[2]}m')
print(f'Control type: {cfg.control.control_type}')
print('Cassie integration verified!')
"

# Test policy loading
conda run -n mapush python -c "
import torch
policy = torch.jit.load('./resources/robots/cassie/policy/policy_1.pt')
out = policy(torch.zeros(1, 48))
print(f'Policy: 48-dim obs -> {out.shape[1]}-dim action')
print('Policy verified!')
"
```

**Output:**
```
Class: <class 'mqe.envs.cassie.cassie.Cassie'>
Spawn height: 1.0m
Control type: C
Cassie integration verified!

Policy: 48-dim obs -> 12-dim action
Policy verified!
```

---

**Integration Complete:** 2026-02-02
**Author:** Claude (Anthropic)
