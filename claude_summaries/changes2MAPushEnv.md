# Changes to MAPush Environment

This document tracks permanent modifications made to the MAPush environment codebase.

---

## 1. Agent Spawn Angular Separation (Permanent)

**Date:** 2026-01-31
**File Modified:** `mqe/envs/base/legged_robot.py` (lines 622-662)

### Problem

When using heterogeneous agents (e.g., Go1 + Anymal C), agents could spawn on top of each other. The original spawn mechanism sampled each agent's angle (`theta`) around the box independently from `[0, 2π]`, allowing both agents to receive nearly identical angles.

With spawn heights of:
- Go1: Z = 0.45m
- Anymal C: Z = 0.62m

And similar XY positions, the robots' bodies would overlap/intersect at spawn, causing physics instability.

### Solution

Implemented **guaranteed minimum angular separation** between agents spawning around the box.

#### How It Works:

1. Sample a single random `base_theta` per environment
2. Each agent is offset by `agent_idx * min_angular_separation`
3. Small jitter (±30% of separation, max ±0.5 rad) added for training diversity

```
Agent 0: base_theta + jitter
Agent 1: base_theta + min_angular_separation + jitter
Agent N: base_theta + N * min_angular_separation + jitter
```

#### Default Configuration:

- `min_agent_angular_separation = 1.57` (π/2 radians = 90 degrees)

With `r ≈ 1.25m` spawn radius and 90° separation, minimum XY distance between agents is approximately **1.77m**.

### Code Change

```python
# In _reset_root_states(), after sampling radii:

# Minimum angular separation between agents (default: pi/2 = 90 degrees)
min_agent_angular_sep = getattr(self.cfg.domain_rand, "min_agent_angular_separation", 1.57)

if self.num_agents > 1 and min_agent_angular_sep > 0:
    theta_range = random_base_theta_from_init[1] - random_base_theta_from_init[0]
    base_theta = torch.rand(self.num_envs, device=self.device) * theta_range + random_base_theta_from_init[0]

    jitter_range = min(0.5, min_agent_angular_sep * 0.3)

    agent_thetas = torch.zeros(self.num_envs, self.num_agents, device=self.device)
    for agent_idx in range(self.num_agents):
        jitter = (torch.rand(self.num_envs, device=self.device) - 0.5) * 2 * jitter_range
        agent_thetas[:, agent_idx] = base_theta + agent_idx * min_agent_angular_sep + jitter

    base_init_state[:, 1] = agent_thetas.reshape(-1)
```

### Optional Override

While this change is permanent (always active), the separation angle can be configured per-task in `domain_rand`:

```python
class domain_rand:
    min_agent_angular_separation = 2.09  # 120 degrees instead of default 90
```

Set to `0` to disable (not recommended).

### Visual Comparison

```
         Before                      After
           Box                        Box
            |                          |
     t=1.5--+--t=1.52            t=0.3-+-t=1.87 (pi/2 apart)
            |                          |
    [Both agents here]         Agent0    Agent1
      COLLISION!               (separated by ~1.77m)
```

---

## 2. Box Mass Override Option (Optional Flag)

**Date:** 2026-01-31
**Files Modified:**
- `mqe/envs/configs/go1_push_mid_config.py` - added `npc_mass_override` parameter (this is the config used by ENV_DICT)
- `mqe/envs/npc/go1_object.py` - implemented runtime mass override

**Note:** The config in `task/cuboid/config.py` is NOT used by the hetero environment. The actual config used is `mqe/envs/configs/go1_push_mid_config.py`.

### Problem

With HAPPO's separate actor networks (unlike MAPPO's shared network), agents don't naturally learn to collaborate. The default box mass of **4 kg** is trivially easy for either robot to push solo:
- Go1: ~12 kg
- Anymal C: ~50 kg

This allows free-rider behavior where one agent does nothing while the other pushes.

### Solution

Added optional **runtime box mass override** that can make the box heavy enough to require both agents to push together.

### Configuration

In `mqe/envs/configs/go1_push_mid_config.py`:

```python
class asset(Go1Cfg.asset):
    # Box mass override: None = use URDF default (4kg)
    # Set to higher value (e.g., 50.0) to require collaboration
    # Box dimensions: 1.2m x 1.2m x 0.5m
    npc_mass_override = None  # or set to desired mass in kg
```

### Usage

**Test old checkpoint with light box (default):**
```python
class asset:
    npc_mass_override = None  # Uses URDF default (4 kg)
```

**Train/test with heavy box:**
```python
class asset:
    npc_mass_override = 50.0  # 50 kg box - requires collaboration
```

### Recommended Mass Values

| Mass | Effect | Use Case |
|------|--------|----------|
| None / 4 kg | Trivial for single robot | Baseline / old checkpoints |
| 20-30 kg | Challenging for Go1 solo, easy for Anymal | Mild collaboration pressure |
| 40-60 kg | Hard for either solo, needs both | Strong collaboration signal |
| 80+ kg | Very difficult even together | Stress test |

### Implementation Details

When `npc_mass_override` is set (not None), the code:
1. Overrides the box mass from URDF
2. Recalculates correct inertia tensor for the new mass using box dimensions (1.2m × 1.2m × 0.5m)
3. Prints confirmation message: `[Box Mass Override] Box mass set to X kg (URDF default: 4 kg)`

```python
# In go1_object.py _create_npc():
if npc_mass_override is not None:
    rigid_body_props[0].mass = npc_mass_override
    # Recalculate inertia: I = (1/12) * m * (a² + b²)
    m = npc_mass_override
    w, d, h = 1.2, 1.2, 0.5
    ixx = (1.0/12.0) * m * (d*d + h*h)
    iyy = (1.0/12.0) * m * (w*w + h*h)
    izz = (1.0/12.0) * m * (w*w + d*d)
```

### Backwards Compatibility

- Default is `None` (no override) → existing checkpoints work unchanged
- Old models trained with 4 kg box can be tested with heavy box by setting the flag
- New models can be trained with heavy box from the start

---

## 3. Per-Agent Type Flags (--agent0/--agent1)

**Date:** 2026-01-31 (Updated: 2026-02-02)
**Files Modified:**
- `HARL/harl_mapush/train.py` - Replaced `--hetero_agent` with `--agent0`/`--agent1`
- `HARL/harl_mapush/test.py` - Same flag changes
- `HARL/harl/envs/mapush/mapush_env.py` - Updated to use agent0/agent1
- `mqe/envs/utils.py` - Updated `custom_cfg()` signature, `make_hetero_env()` for routing
- `mqe/envs/robot_registry.py` - Central robot registry for all supported robots
- `mqe/envs/base/hetero_robot.py` - Heterogeneous robot environment class
- `HARL/harl_mapush/utils/run_config_saver.py` - Updated for new flags

### Problem

The original `--hetero_agent` flag hardcoded agent0 as Go1:
```bash
# Old approach - agent0 always Go1
python train.py --hetero_agent anymal_c  # → agent0=go1, agent1=anymal_c
```

This prevented:
- Testing arbitrary agent orderings (e.g., anymal_c as agent0)
- Homogeneous training with non-Go1 robots (e.g., both anymal_c)
- Future multi-robot combinations

### Solution

Replaced `--hetero_agent` with explicit `--agent0` and `--agent1` flags:

```bash
# New approach - full flexibility
python train.py --agent0 go1 --agent1 anymal_c      # Heterogeneous
python train.py --agent0 anymal_c --agent1 go1      # Reversed order
python train.py --agent0 anymal_c --agent1 anymal_c # Both Anymal C
python train.py --agent0 cassie --agent1 cassie     # Both Cassie
python train.py --agent0 go1 --agent1 cassie        # Go1 + Cassie
python train.py                                      # Default: both go1
```

### Environment Routing Logic

The system automatically detects whether to create a homogeneous or heterogeneous environment:

```python
# In make_hetero_env():
is_homogeneous = (agent0 == agent1)  # Same robot type

if is_homogeneous and robot_type == 'go1':
    # TRUE HOMOGENEOUS: Use native Go1Object class directly
    # This is the original MAPush environment
    task_class = Go1Object

else:
    # HETERO PATH: Use HeteroRobot for:
    # 1. Different robots (e.g., go1 + cassie)
    # 2. Same non-Go1 robots (e.g., cassie + cassie, anymal_c + anymal_c)
    task_class = HeteroRobot + Go1Object
```

**Why this routing?**
- `Go1Object` inherits from `Go1`, so go1+go1 uses the native class directly
- Non-Go1 homogeneous pairs (cassie+cassie) use HeteroRobot to avoid MRO conflicts
- All heterogeneous pairs use HeteroRobot

### Usage Examples

**Training:**
```bash
# Homogeneous (default - both Go1)
python HARL/harl_mapush/train.py --exp_name baseline

# Homogeneous Cassie
python HARL/harl_mapush/train.py \
  --exp_name cassie_pair \
  --agent0 cassie \
  --agent1 cassie

# Heterogeneous (Go1 + Anymal C)
python HARL/harl_mapush/train.py \
  --exp_name go1_anymal \
  --agent0 go1 \
  --agent1 anymal_c

# Heterogeneous (Cassie + Anymal C)
python HARL/harl_mapush/train.py \
  --exp_name cassie_anymal \
  --agent0 cassie \
  --agent1 anymal_c
```

**Testing:**
```bash
python HARL/harl_mapush/test.py \
  --checkpoint ./results/.../checkpoints/50M \
  --agent0 go1 \
  --agent1 anymal_c \
  --mode calculator \
  --num_episodes 100
```

### Detection Logic

Heterogeneous mode is now detected by comparing agent types:
```python
is_hetero = (agent0 != agent1)
```

### Backwards Compatibility

- Default values: `--agent0 go1 --agent1 go1` (homogeneous Go1)
- Old `--hetero_agent` flag removed (not backwards compatible)
- Existing checkpoints still work - just specify correct agent types when testing

### Available Robot Types

| Robot | Description | Control | DOF |
|-------|-------------|---------|-----|
| `go1` | Unitree Go1 quadruped | Hierarchical (walk_these_ways) | 12 |
| `anymal_c` | ANYmal C quadruped | Hierarchical (legged_gym LSTM) | 12 |
| `cassie` | Agility Robotics Cassie biped | Hierarchical (legged_gym) | 10 |

### Adding New Robots

To add a new robot to the system:

#### Step 1: Create Robot Files
```
mqe/envs/<robot_name>/
├── <robot_name>.py          # Robot class (inherits LeggedRobotField)
├── <robot_name>_config.py   # Robot configuration
```

#### Step 2: Register in Robot Registry
Edit `mqe/envs/robot_registry.py`:
```python
ROBOT_REGISTRY = {
    # ... existing robots ...
    'new_robot': {
        'class_path': 'mqe.envs.new_robot.new_robot.NewRobot',
        'config_path': 'mqe.envs.new_robot.new_robot_config.NewRobotCfg',
        'default_control': 'C',  # 'C' for hierarchical, 'P' for direct
        'num_actions': 3,        # Mid-level action dim [vx, vy, vyaw]
        'description': 'New robot description'
    },
}
```

#### Step 3: Add Locomotion Policy Loading (if control_type='C')
Edit `mqe/envs/base/hetero_robot.py` in `_load_locomotion_policies()`:
```python
elif robot_type == 'new_robot':
    # Load locomotion policy
    policy_model = torch.jit.load(policy_dir + '/policy_1.pt', map_location=self.device)

    def new_robot_policy(obs, info={}, _model=policy_model):
        with torch.no_grad():
            action = _model.forward(obs)
        return action

    self.locomotion_policies.append(new_robot_policy)
    obs_buffer = torch.zeros(self.num_envs, OBS_DIM, dtype=torch.float, device=self.device)
    self.locomotion_obs_buffers.append({'obs': obs_buffer, 'history': None})
```

#### Step 4: Add Actuator Network (if applicable)
In `_load_actuator_network()`:
```python
elif robot_type == 'new_robot':
    # Load actuator network or use PD control fallback
    ...
```

#### Step 5: Provide Assets
```
resources/robots/<robot_name>/
├── urdf/<robot_name>.urdf
├── policy/policy_1.pt        # Trained locomotion policy
└── actuator_net/...          # If using actuator network
```

### Interchangeability Matrix

All combinations are supported:

|          | go1 | anymal_c | cassie |
|----------|-----|----------|--------|
| **go1**      | ✅ Homo | ✅ Hetero | ✅ Hetero |
| **anymal_c** | ✅ Hetero | ✅ Homo* | ✅ Hetero |
| **cassie**   | ✅ Hetero | ✅ Hetero | ✅ Homo* |

*Non-Go1 homogeneous pairs use HeteroRobot internally but function as homogeneous environments.

---

## 4. Run Configuration Saver (HARL Training)

**Date:** 2026-01-31
**Files Created/Modified:**
- `HARL/harl_mapush/utils/run_config_saver.py` - New utility for saving run configuration
- `HARL/harl_mapush/runners/mapush_happo_runner.py` - Integration into training

### Problem

With many command-line flags and configuration options (reward types, critic modes, heterogeneous agents, box mass, etc.), it became difficult to:
1. Remember what flags were used for a specific training run
2. Reproduce a training run later
3. Verify test settings match training settings

### Solution

Implemented automatic **run configuration saving** at training start. Two files are created in the run directory:

#### 1. `command.txt`
```
# Training command - copy-paste to reproduce
# Generated: 2026-01-31 14:30:00
# Run directory: HARL/results/mapush/cuboid/happo/hetero_go1_anymal/...

python HARL/harl_mapush/train.py --algo happo --exp_name hetero_go1_anymal --hetero_agent anymal_c ...

# Testing command template (update checkpoint path):
# python HARL/harl_mapush/test.py \
#   --checkpoint .../checkpoints/LATEST \
#   --hetero_agent anymal_c \
#   --mode viewer \
#   --num_episodes 5
```

#### 2. `run_config.yaml`
Complete configuration snapshot including:
- **Agents**: agent0 type, agent1 type, is_heterogeneous
- **Algorithm**: name, seed, learning rates, PPO params
- **Training**: num_env_steps, n_rollout_threads, episode_length
- **Network**: hidden_sizes, activation, recurrent settings
- **Environment args**: all command-line flags (reward modes, critic modes, etc.)
- **Environment config** (from Python class):
  - **Box**: `effective_mass_kg`, dimensions, collision/gravity settings
  - **Spawn**: angular separation, position range, friction
  - **Rewards**: scales for all reward components
  - **Termination**: thresholds for z_wave, collision
  - **Goal**: distance range, threshold

### Key Feature: Box Mass Tracking

The run config explicitly tracks box mass:
```yaml
environment:
  box:
    mass_override_kg: 50.0
    urdf_default_mass_kg: 4.0
    effective_mass_kg: 50.0  # ← The actual mass used
    dimensions_m: [1.2, 1.2, 0.5]
```

### Usage

Configuration is saved automatically when training starts. To check what settings a run used:

```bash
# View the command used
cat HARL/results/.../command.txt

# View full configuration
cat HARL/results/.../run_config.yaml
```

To load configuration in Python (e.g., for testing):
```python
from harl_mapush.utils.run_config_saver import load_run_config, print_config_summary

config = load_run_config("/path/to/run_dir")
print_config_summary(config)

# Access specific values
box_mass = config["environment"]["box"]["effective_mass_kg"]
hetero_agent = config["agents"]["agent1"]  # e.g., "anymal_c"
```

### Files in Run Directory

After training starts, the run directory structure is:
```
HARL/results/mapush/cuboid/happo/<exp_name>/seed-1-<timestamp>/
├── command.txt         # ← Training command for reproduction
├── run_config.yaml     # ← Complete configuration snapshot
├── checkpoints/
│   ├── 10M/
│   ├── 20M/
│   └── ...
└── logs/
```

---

## 5. Baseline MAPPO Rewards Flag (--baseline_mappo_rewards)

**Date:** 2026-02-01
**Files Modified:**
- `openrl_ws/utils.py` - Added `--baseline_mappo_rewards` argument (default: True)
- `openrl_ws/train.py` - Pass flag to `custom_cfg()`
- `mqe/envs/utils.py` - Accept flag in `custom_cfg()`, set config values and original scales
- `mqe/envs/wrappers/go1_push_mid_wrapper.py` - Check flag and disable all HAPPO-specific rewards
- `openrl_ws/run_utils/run_config_saver.py` - Save flag in run config

**Note:** This flag is **ONLY for MAPPO (OpenRL)**. HAPPO has its own `--mapush_og_rewards_teamified` flag.

### Problem

The MAPush codebase evolved to include many HAPPO-specific rewards (proximity_penalty, goal_push_bonus, cooperation bonuses, etc.) that were added to the shared `go1_push_mid_wrapper.py`. These rewards were **enabled by default** via non-zero scale values in the config:

```python
# In config - these were contaminating MAPPO runs!
proximity_penalty_scale = 0.002  # HAPPO reward - should be OFF for MAPPO
ocb_reward_scale = 0.01  # Was changed from original 0.004
```

When running MAPPO training, it was unknowingly using these HAPPO rewards instead of the original 7 MAPush rewards.

### Solution

Added `--baseline_mappo_rewards` flag (default: True) that guarantees:

1. **ONLY the original 7 MAPush rewards** are used
2. **Original scales** are restored (some had been changed)
3. **ALL HAPPO-specific rewards** are disabled (set to 0)

### Original 7 MAPush Rewards (Baseline)

| Reward | Original Scale | Description |
|--------|---------------|-------------|
| `target_reward` | 0.00325 | Distance to target |
| `approach_reward` | 0.00075 | Distance to box |
| `collision_punishment` | -0.0025 | Agent-agent collision |
| `push_reward` | 0.0015 | Push force |
| `ocb_reward` | **0.004** | Orientation-conditioned bonus |
| `reach_target_reward` | 10 | Success bonus |
| `exception_punishment` | -5 | Physics exception penalty |

**Note:** `ocb_reward_scale` was changed to 0.01 at some point - the baseline restores it to the original 0.004.

### Disabled HAPPO Rewards

When `--baseline_mappo_rewards True`, the following are set to 0:

| HAPPO Reward | Was Scale | Now |
|--------------|-----------|-----|
| `proximity_penalty` | 0.002 | 0 |
| `goal_push_bonus` | 0.01 | 0 |
| `engagement_bonus` | 0.0004 | 0 |
| `cooperation_bonus` | 0.0002 | 0 |
| `same_side_bonus` | 0.0004 | 0 |
| `blocking_penalty` | -0.001 | 0 |
| `directional_progress` | 0.003 | 0 |
| `dual_push_bonus` | 0.005 | 0 |
| `gaussian_proximity_bonus` | 0.006 | 0 |

Also disables flags: `individualized_rewards`, `shared_gated_rewards`, `cooperation_rewards`, `collaboration_rewards`, `reward_scale_testing`, `positive_approachtobox_reward`

### Usage

```bash
# Default (baseline rewards) - RECOMMENDED for fair MAPPO comparison
python ./openrl_ws/train.py --algo ppo --task go1push_mid ...

# Explicitly enable baseline (same as default)
python ./openrl_ws/train.py --algo ppo --task go1push_mid --baseline_mappo_rewards True

# Disable baseline to use HAPPO rewards with MAPPO (for experiments)
python ./openrl_ws/train.py --algo ppo --task go1push_mid --baseline_mappo_rewards False
```

### Output

When enabled, you'll see:
```
[MAPPO Training] Reward mode: BASELINE (original 7 MAPush rewards)
[custom_cfg] BASELINE MAPPO REWARDS MODE: Using original 7 rewards with original scales
[Go1PushMidWrapper] BASELINE MAPPO REWARDS MODE ACTIVE
  Using ONLY 7 original rewards: target, approach, collision, push, ocb, reach_target, exception
  Scales: target=0.00325, approach=0.00075, collision=-0.0025, push=0.0015, ocb=0.004, reach=10, exception=-5
```

### Run Config Tracking

The flag is saved in `run_config.yaml`:
```yaml
rewards:
  baseline_mappo_rewards: true
  description: "baseline_mappo_rewards=True uses ONLY original 7 MAPush rewards with original scales"
```

### Backwards Compatibility

- Default is `True` (baseline mode) for all new MAPPO runs
- To replicate old contaminated runs, use `--baseline_mappo_rewards False`
- HAPPO is unaffected - continue using `--mapush_og_rewards_teamified` for HAPPO baseline

---
