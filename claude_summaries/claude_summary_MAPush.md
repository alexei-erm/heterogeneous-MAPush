# MAPush Repository Quick-Start Guide

**Last Updated:** 2025-12-13

## Project Overview

This repository implements a hierarchical Multi-Agent Reinforcement Learning (MARL) framework for multi-quadruped collaborative pushing tasks in Isaac Gym. The codebase is based on [MQE (Multi-agent Quadruped Environment)](https://github.com/ziyanx02/multiagent-quadruped-environment).

**Paper:** [Learning Multi-Agent Loco-Manipulation for Long-Horizon Quadrupedal Pushing](https://arxiv.org/pdf/2411.07104)
**Website:** https://collaborative-mapush.github.io/

---

## Your Objectives & Use Case

### Setting Configuration
- **2 agents** (quadrupeds)
- **2 randomly located obstacles**
- **Cuboid task** (primary focus)

### Goals (in order)
1. **Replace MAPPO with HAPPO** - Swap out the current MARL algorithm
2. **Working HAPPO model (homogeneous)** - Both robots identical, same observation space
3. **Heterogeneous observations** - Two Go1s with different observation spaces (e.g., one has one less input)
4. **Heterogeneous robots** - agent0: Go1, agent1: wheeled robot (any type)
5. **Real-world deployment** - Deploy with MuJoCo (MC) in the lab with 2 different physical robots

---

## Repository Structure

```
/home/gvlab/new-universal-MAPush/
├── mqe/                          # Main environment package
│   ├── envs/
│   │   ├── base/                 # Base classes for environments
│   │   │   ├── base_task.py      # BaseTask abstract class
│   │   │   ├── base_config.py    # Base configuration
│   │   │   ├── legged_robot.py   # LeggedRobot base class (multi-agent support)
│   │   │   └── legged_robot_config.py
│   │   ├── go1/                  # Unitree Go1 quadruped specific code
│   │   │   ├── go1.py            # Go1 class with locomotion policy
│   │   │   └── go1_config.py     # Go1-specific configurations
│   │   ├── field/                # Field environment extensions
│   │   │   ├── legged_robot_field.py
│   │   │   └── legged_robot_field_config.py
│   │   ├── npc/                  # Non-player characters (objects)
│   │   │   └── go1_object.py     # Interactive objects
│   │   ├── wrappers/             # Task-specific wrappers (obs, rewards, actions)
│   │   │   ├── empty_wrapper.py
│   │   │   ├── go1_push_mid_wrapper.py    # Mid-level controller wrapper
│   │   │   ├── go1_push_upper_wrapper.py  # High-level controller wrapper
│   │   │   └── utils/
│   │   │       ├── trajectory.py
│   │   │       └── rrt.py
│   │   ├── configs/              # Environment configuration files
│   │   │   ├── go1_push_mid_config.py     # Mid-level config
│   │   │   └── go1_push_upper_config.py   # High-level config
│   │   ├── utils.py              # Env utilities (make_mqe_env)
│   │   └── utils_dist.py         # Distance calculation utilities
│   └── utils/                    # General utilities
│       ├── terrain/              # Terrain generation
│       ├── helpers.py
│       ├── math.py
│       └── task_registry.py
├── openrl_ws/                    # OpenRL integration workspace
│   ├── train.py                  # Main training script
│   ├── test.py                   # Testing/evaluation script
│   ├── utils.py                  # OpenRL wrapper (mqe_openrl_wrapper)
│   ├── update_config.py          # Config update utility
│   └── cfgs/                     # RL algorithm configurations
│       ├── ppo.yaml              # PPO hyperparameters (CURRENTLY USED)
│       ├── dppo.yaml             # Distributed PPO
│       ├── mat.yaml              # Multi-Agent Transformer
│       └── jrpo.yaml             # Joint RPO
├── task/                         # Task-specific configurations
│   ├── cuboid/                   # YOUR PRIMARY TASK
│   │   ├── config.py             # Task configuration (Go1PushMidCfg)
│   │   └── train.sh              # Training script (wrapper around openrl_ws/train.py)
│   ├── Tblock/
│   │   ├── config.py
│   │   └── train.sh
│   └── cylinder/
│       ├── config.py
│       └── train.sh
├── resources/                    # Assets and pretrained models
│   ├── robots/                   # Robot URDF/mesh files
│   │   ├── go1/                  # Unitree Go1 (YOUR PRIMARY ROBOT)
│   │   ├── a1/                   # Unitree A1
│   │   ├── anymal_b/             # ANYmal B
│   │   ├── anymal_c/             # ANYmal C
│   │   └── cassie/               # Cassie
│   ├── objects/                  # Object URDF files
│   │   ├── cuboid/SmallBox.urdf
│   │   ├── Tblock/
│   │   └── cylinder/
│   ├── command_nets/             # Mid-level controller checkpoints
│   ├── goals_net/                # High-level controller checkpoints (pretrained)
│   └── actuator_nets/            # Locomotion policy networks
├── results/                      # Training outputs (auto-generated)
│   └── <mm-dd-hh_object>/        # Timestamped results
│       ├── checkpoints/          # Model checkpoints
│       ├── task/                 # Task config snapshot
│       ├── success_rate.txt      # Success rate logs
│       └── log.txt
├── script/                       # Legacy/alternative training scripts
│   ├── train.py
│   └── utils/
├── HARL/                         # HARL library (IGNORED per your request)
├── setup.py                      # Package installation
└── README.md                     # Original documentation
```

---

## Key Components Explained

### 1. Hierarchical Control Architecture

The system uses a **two-level hierarchical control**:

#### **Mid-Level Controller** (Your main focus)
- **File:** `mqe/envs/wrappers/go1_push_mid_wrapper.py`
- **Config:** `task/cuboid/config.py` (inherits from `Go1PushMidCfg`)
- **Purpose:** Outputs velocity commands for coordinated pushing
- **Observation Space:**
  - Without `general_dist`: `(2 + 3 * num_agents,)` dimensions
    - Target position (x, y)
    - Box position (x, y) + box yaw
    - Other agents' positions (x, y) + yaws
  - With `general_dist`: `(3 + 3 * num_agents,)` (includes target yaw)
- **Action Space:** `(3,)` - [linear_vel_x, linear_vel_y, angular_vel_z]
- **Rewards:**
  - `target_reward`: Box moving toward target
  - `approach_reward`: Robots approaching box
  - `collision_punishment`: Inter-robot collision avoidance
  - `push_reward`: Reward for pushing (box velocity > 0.1)
  - `ocb_reward`: Optimal contact point reward (pushing perpendicular to box edges)
  - `reach_target_reward`: Large bonus for task completion
  - `exception_punishment`: Penalty for failures (rollover, pitch, collision)

#### **High-Level Controller** (Optional, for long-horizon planning)
- **File:** `mqe/envs/wrappers/go1_push_upper_wrapper.py`
- **Config:** `mqe/envs/configs/go1_push_upper_config.py`
- **Purpose:** Plans subgoals; mid-level executes them
- **Note:** You likely won't need this for your objectives

### 2. Environment Configuration System

Configurations use **inheritance**:
```
BaseConfig → LeggedRobotCfg → Go1Cfg → Go1PushMidCfg
```

**Key config file:** `task/cuboid/config.py`

#### Important Configuration Classes:
- **`env`**: `num_envs`, `num_agents` (2 for you), `num_npcs`, `episode_length_s`
- **`asset`**: Robot/object URDF paths, collision settings
- **`terrain`**: Map size, obstacles, friction
- **`init_state`**: Initial positions for agents and objects
- **`domain_rand`**: Randomization ranges (agent position, yaw, box yaw, friction)
- **`goal`**: Goal position settings (static/random/received/sequential)
- **`rewards.scales`**: Reward weights (tune these!)
- **`termination`**: Termination conditions (roll, pitch, collision, z_wave)
- **`command`**: Control type (velocity, body height, gait, etc.)
- **`control`**: `control_type = 'C'` (uses locomotion policy)

### 3. Training Pipeline

#### Training Command (from `task/cuboid/train.sh`):
```bash
source task/cuboid/train.sh False
```

**Workflow:**
1. `train.sh` calls `openrl_ws/update_config.py` → Updates `mqe/envs/configs/go1_push_mid_config.py` from `task/cuboid/config.py`
2. Calls `openrl_ws/train.py` with args:
   - `--algo ppo` (currently PPO, you'll change to HAPPO)
   - `--task go1push_mid`
   - `--num_envs 500` (parallel environments)
   - `--train_timesteps 100000000`
   - `--config ./openrl_ws/cfgs/ppo.yaml`
3. Training logs → `./log/`
4. Final results → `./results/<mm-dd-hh>_cuboid/`
5. Auto-evaluates checkpoints → `success_rate.txt`

#### Testing Command:
```bash
source results/10-15-23_cuboid/task/train.sh True
```

### 4. RL Algorithm Integration (OpenRL)

**Current:** Uses OpenRL library with **PPO** (`openrl.modules.common.PPONet`)

**Your task:** Replace with **HAPPO**

**Key files to modify:**
- `openrl_ws/train.py:33-37` - Algorithm selection logic
- `openrl_ws/cfgs/` - Add `happo.yaml` config
- May need to modify wrapper in `openrl_ws/utils.py`

**Algorithm configs available:**
- `ppo.yaml` - Current (homogeneous MAPPO-style)
- `dppo.yaml` - Distributed PPO
- `mat.yaml` - Multi-Agent Transformer
- `jrpo.yaml` - Joint RPO

### 5. Observation Space Customization

**Location:** `mqe/envs/wrappers/go1_push_mid_wrapper.py`

**For Goal 3 (heterogeneous observations):**
- Modify `reset()` and `step()` methods (lines 119-391)
- Currently, all agents get identical observation dimensions
- You'll need to create per-agent observation spaces
- Options:
  1. Modify wrapper to support `observation_space` as list/dict
  2. Mask certain observations for specific agents
  3. Create separate wrapper classes per agent type

### 6. Robot Assets

**Available robots:** (in `resources/robots/`)
- `go1/` - Unitree Go1 (your primary)
- `a1/` - Unitree A1 (similar to Go1)
- `anymal_b/`, `anymal_c/` - ANYmal quadrupeds
- `cassie/` - Bipedal robot

**For Goal 4 (Go1 + wheeled robot):**
- You'll need to add wheeled robot URDF to `resources/robots/`
- Create new robot class (like `mqe/envs/wheeled_robot/`) following Go1 structure
- Modify environment to support heterogeneous robot types
- Update `asset` config to specify different URDFs per agent

---

## Important Code Locations

### Multi-Agent Logic
- **Agent initialization:** `mqe/envs/base/legged_robot.py:100-200`
- **Multi-agent state management:** `mqe/envs/base/legged_robot.py` (root_states shape: `[num_envs * num_agents, ...]`)
- **Per-agent observations:** `mqe/envs/wrappers/go1_push_mid_wrapper.py:162-189` (rotation to local frame)

### MAPPO → HAPPO Replacement Points
1. **Algorithm selection:** `openrl_ws/train.py:32-38`
2. **Network definition:** `openrl_ws/train.py:36` (`PPONet`)
3. **Agent definition:** `openrl_ws/train.py:37` (`PPOAgent`)
4. **Config file:** Add `openrl_ws/cfgs/happo.yaml`

### Distance Calculation (for rewards)
- **File:** `mqe/envs/utils_dist.py`
- **Class:** `dist_calculator`
- **Used in:** `go1_push_mid_wrapper.py:328` for target distance reward

### Locomotion Policy
- **Loaded in:** `mqe/envs/go1/go1.py:33` (`_prepare_locomotion_policy()`)
- **Type:** Walk-these-ways style locomotion controller
- **Action preprocessing:** `go1.py:63-99` (converts high-level commands to joint actions)

---

## Training Workflow Summary

### For Cuboid Task (your case):
1. **Edit config:** `task/cuboid/config.py`
   - Set `num_agents = 2`
   - Set `num_npcs = 2` (box + target)
   - Configure domain randomization ranges
   - Tune reward scales in `rewards.scales`

2. **Train:**
   ```bash
   source task/cuboid/train.sh False
   ```

3. **Monitor:** TensorBoard logs in `./log/`

4. **Results:** Auto-saved to `./results/<timestamp>_cuboid/`
   - Checkpoints every 20M steps
   - Success rate calculations in `success_rate.txt`

5. **Test:**
   ```bash
   source results/<timestamp>_cuboid/task/train.sh True
   ```
   Edit the checkpoint filename in the script first.

---

## Key Parameters for Your Objectives

### Goal 1 & 2: HAPPO (Homogeneous)
- **Focus:** `openrl_ws/train.py`, `openrl_ws/cfgs/happo.yaml`
- **No env changes needed** - current homogeneous setup works
- **Verify:** `task/cuboid/config.py` has `num_agents = 2`

### Goal 3: Heterogeneous Observations (2 Go1s, different inputs)
- **Modify:** `mqe/envs/wrappers/go1_push_mid_wrapper.py`
  - Lines 54-58: Define per-agent observation spaces
  - Lines 181-188: Customize observation construction per agent
- **Option:** Create agent-specific observation masks
- **Test:** Ensure HAPPO supports asymmetric observations

### Goal 4: Heterogeneous Robots (Go1 + Wheeled)
- **Add wheeled robot:**
  1. Create `mqe/envs/wheeled_robot/wheeled_robot.py` (similar to `go1/go1.py`)
  2. Add URDF to `resources/robots/wheeled_robot/`
  3. Modify `task/cuboid/config.py`:
     - `asset.file` for each agent
     - Potentially different action spaces
  4. Update `mqe/envs/base/legged_robot.py` to handle mixed robot types
- **Challenge:** Different action spaces require HAPPO modification or action padding

---

## Utilities & Helpers

### Environment Creation
- **Function:** `mqe/envs/utils.py:make_mqe_env(task_name, args, custom_cfg)`
- **Task registry:** `mqe/utils/task_registry.py`
- **Wrapper chain:** `make_mqe_env` → `Go1` → `Go1PushMidWrapper` → `mqe_openrl_wrapper`

### Config Merging
- **Function:** `mqe/utils/helpers.py:merge_dict()`
- **Used for:** Merging child config with parent configs

### Math Utilities
- **File:** `mqe/utils/math.py`
- **Key functions:** `quat_apply_yaw()`, `wrap_to_pi()`, `torch_rand_sqrt_float()`

---

## Common Pitfalls & Tips

### 1. NumPy Version Conflict
**Issue:** Isaac Gym requires numpy ≤ 1.19.5
**Fix:**
```bash
pip install numpy==1.19.5
# OR modify isaacgym/python/isaacgym/torch_utils.py: np.float → np.float32
```

### 2. Rendering on A100/A800
**Issue:** Segmentation fault during rendering
**Fix:** Use GeForce GPUs for rendering, A100 for headless training

### 3. Config Not Updating
**Issue:** Changes to `task/cuboid/config.py` don't take effect
**Fix:** `train.sh` copies config to `mqe/envs/configs/go1_push_mid_config.py` - ensure this step runs

### 4. Multi-Agent Indexing
**Shape conventions:**
- `root_states`: `[num_envs * num_agents, 13]` (flattened)
- `obs_buf`: `[num_envs, num_agents, obs_dim]`
- Rewards: `[num_envs, num_agents]`

### 5. Observation Space for Heterogeneous Agents
OpenRL expects uniform observation spaces. For asymmetric obs:
- Pad to same dimension + mask
- Use dictionary observation spaces (if OpenRL supports)
- Modify wrapper to handle per-agent spaces

---

## Next Steps for Your Goals

### Immediate (Goal 1 & 2):
1. Research HAPPO implementation (check OpenRL or implement custom)
2. Create `openrl_ws/cfgs/happo.yaml`
3. Modify `openrl_ws/train.py` to support HAPPO algorithm
4. Test with current homogeneous 2-agent setup
5. Verify training convergence

### Short-term (Goal 3):
1. Design observation space difference (which input to remove?)
2. Modify `go1_push_mid_wrapper.py` to support per-agent observations
3. Ensure HAPPO handles asymmetric observations
4. Re-train and validate performance

### Medium-term (Goal 4):
1. Select wheeled robot type
2. Obtain/create URDF
3. Implement wheeled robot environment class
4. Handle heterogeneous action spaces
5. Test in simulation

### Long-term (Goal 5):
1. MuJoCo integration (separate from Isaac Gym)
2. Sim-to-real transfer considerations
3. Real robot interface setup

---

## Questions to Resolve

1. **HAPPO availability:** Does OpenRL support HAPPO natively, or do you need a custom implementation?
2. **Observation asymmetry:** Which specific input will be removed from one Go1?
3. **Wheeled robot:** Which wheeled robot model (TurtleBot, Jackal, custom)?
4. **Action space heterogeneity:** How to handle different action spaces in HAPPO?

---

## Useful Commands

```bash
# Training (headless)
source task/cuboid/train.sh False

# Testing (with visualization)
source results/<folder>/task/train.sh True

# TensorBoard
tensorboard --logdir=./log

# Check GPU usage
nvidia-smi

# Find latest checkpoint
ls -lt results/*/checkpoints/

# Quick test (few envs)
python openrl_ws/test.py --num_envs 1 --algo ppo --task go1push_mid \
  --checkpoint <path> --test_mode viewer
```

---

## Additional Resources

- **Original README:** `/home/gvlab/new-universal-MAPush/README.md`
- **Paper:** arXiv:2411.07104
- **MQE Base:** https://github.com/ziyanx02/multiagent-quadruped-environment
- **Isaac Gym Docs:** https://developer.nvidia.com/isaac-gym

---

**Good luck with the HAPPO integration!** 🚀
