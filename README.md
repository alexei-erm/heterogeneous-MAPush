# MAPush: Multi-Agent Loco-Manipulation for Quadrupedal Pushing

A hierarchical multi-agent reinforcement learning framework for multi-robot collaborative pushing in Isaac Gym. Supports **homogeneous and heterogeneous** robot teams with two algorithm backends (**HAPPO** and **MAPPO**) across three task levels.

Based on [MQE](https://github.com/ziyanx02/multiagent-quadruped-environment). Original paper: [Learning Multi-Agent Loco-Manipulation for Long-Horizon Quadrupedal Pushing](https://arxiv.org/pdf/2411.07104).

---

## Table of Contents

1. [Installation](#installation)
2. [Architecture Overview](#architecture-overview)
3. [Tasks](#tasks)
4. [Algorithms](#algorithms)
5. [Heterogeneous Robot Support](#heterogeneous-robot-support)
6. [Training](#training)
7. [Testing](#testing)
8. [Configuration Reference](#configuration-reference)
9. [Repository Structure](#repository-structure)
10. [Key Technical Notes](#key-technical-notes)
11. [Troubleshooting](#troubleshooting)
12. [Citation](#citation)

---

## Installation

1. Create a conda environment with Python 3.8:
   ```bash
   conda create -n mapush python=3.8
   conda activate mapush
   ```

2. Install PyTorch (compatible version) from https://pytorch.org/.

3. Install Isaac Gym Preview 4 from https://developer.nvidia.com/isaac-gym:
   ```bash
   tar -xf IsaacGym_Preview_4_Package.tar.gz
   cd isaacgym/python && pip install -e .
   ```

4. Verify Isaac Gym:
   ```bash
   cd isaacgym/python/examples && python 1080_balls_of_solitude.py
   ```

5. Install MAPush:
   ```bash
   cd /path/to/new-universal-MAPush
   pip install -e .
   ```

6. Install HARL (for HAPPO):
   ```bash
   cd HARL && pip install -e .
   ```

---

## Architecture Overview

### Hierarchical Control

```
Upper Level (go1push_upper) ─── 1 HARL agent, plans sub-goal waypoints
  |
  v
Mid Level (go1push_mid) ─────── 2 agents, velocity commands toward goal
  |
  v
Locomotion Policy ────────────── Per-robot (Walk-These-Ways / legged_gym)
  |
  v
Joint Torques ────────────────── Isaac Gym physics
```

Each level can be trained independently. The upper level loads frozen mid-level actors at runtime.

### Two Algorithm Backends

| Backend | Algorithm | Entry Point | Config |
|---------|-----------|-------------|--------|
| HARL | HAPPO | `HARL/harl_mapush/train.py` | `HARL/harl/configs/algos_cfgs/happo.yaml` |
| OpenRL | MAPPO | `openrl_ws/train.py` | `openrl_ws/cfgs/ppo.yaml` |

**HAPPO** (Heterogeneous-Agent PPO) is the primary algorithm. It uses sequential agent updates with per-agent actor networks and a shared centralized critic. This naturally supports heterogeneous agents without parameter sharing.

**MAPPO** (Multi-Agent PPO) uses simultaneous updates with shared parameters. It remains available for homogeneous baselines.

---

## Tasks

### go1push_mid (Goal-Based Pushing)

Two agents cooperatively push a box to a target position.

| Property | Value |
|----------|-------|
| Agents | 2 |
| Actor obs | 8 dims (egocentric: target, box, other agent positions/yaws) |
| Action | 3 dims `[vx, vy, vyaw]` |
| Success | Box within threshold of target |
| Episode | Success or timeout (200 steps) |

**Rewards:** distance-to-target, approach-to-box, push, optimal-contact-behind (OCB), reach-target bonus, collision penalty, exception penalty.

### go1push_vel (Velocity Tracking)

Two agents push a box in a commanded direction at a commanded speed. Designed to make cooperation **mechanically necessary** --- a single agent pushing off-center creates net torque, while two agents flanking the box cancel each other's torques for clean linear motion.

| Property | Value |
|----------|-------|
| Agents | 2 |
| Actor obs | 15 dims (egocentric: command direction, box state, velocities, other agent) |
| Critic state | 15 dims (box-centered frame) |
| Action | 3 dims `[vx, vy, vyaw]` |
| Success | None (continuous tracking) |
| Episode | Timeout only |

**Rewards:** velocity tracking (cosine similarity), angular velocity penalty, approach, push, collision, exception.

**Legacy obs mode:** `--legacy_vel_obs` restores old 16-dim actor / 18-dim critic state for testing pre-direction-only checkpoints.

### go1push_upper (High-Level Planner)

A single planner agent outputs sub-goal waypoints. Frozen mid-level actors execute them.

| Property | Value |
|----------|-------|
| Agents | 1 (HARL sees single agent) |
| Actor obs | 26 dims (global: robots, box, obstacles, trajectory) |
| Action | 2 dims `[sub_goal_x, sub_goal_y]` |
| Episode | 160s timeout or target reached |

**Requires:** `--mid_level_checkpoint <path> --mid_level_format happo` (or `openrl`).

---

## Algorithms

### HAPPO (Primary)

Heterogeneous-Agent Proximal Policy Optimization. Key properties:

- **Per-agent actor networks** --- each agent has its own policy, no parameter sharing required
- **Shared centralized critic** --- single value network sees global state
- **Sequential updates** --- agents update one at a time in random order, each accounting for previous agents' policy changes via importance-weighted modified advantages
- **Monotonic improvement guarantee** --- provably converges to Nash equilibrium

Update loop per iteration:
1. Collect rollouts with current joint policy
2. Compute joint advantages via GAE using shared critic
3. Draw random agent permutation
4. For each agent in sequence: update policy with PPO-clip objective using modified advantage, then propagate importance weights to next agent
5. Update shared critic

### MAPPO (Baseline)

Standard Multi-Agent PPO with simultaneous updates and parameter sharing. Used for homogeneous baselines via OpenRL.

### Centralized Critic Modes (HAPPO)

The shared critic can receive global state in 5 different representations:

| Mode | Flag | Dims | Description |
|------|------|------|-------------|
| Absolute | (default) | 11 | box(3) + target(2) + agents(3 each) |
| Box-centered | `--use_box_centered_critic` | 9 | target relative to box + agents relative to box |
| Goal-centered | `--use_goal_centered_critic` | 9 | everything relative to goal |
| Relative | `--use_relative_obs_critic` | 9 | robot-to-box vectors + inter-robot dist + goal-to-box |
| Concat-agent | `--use_concat_agent_observations_critic` | 16 | concatenate both agents' 8-dim actor obs |

---

## Heterogeneous Robot Support

Four robot types are registered and can be used in any combination via `--agent0` / `--agent1` flags:

| Robot | Type | DOFs | Locomotion Policy | Actuator |
|-------|------|------|-------------------|----------|
| `go1` | Quadruped | 12 | Walk-These-Ways (70-dim obs + 2100 history) | Feedforward (`unitree_go1.pt`) |
| `anymal_c` | Quadruped | 12 | legged_gym MLP (48-dim obs) | LSTM (`anydrive_v3_lstm.pt`) |
| `cassie` | Biped | 12 | legged_gym MLP [128,64,32] ELU (48-dim obs) | PD control |

All robots expose a unified 3-action interface `[vx, vy, vyaw]`, so HARL sees identical action spaces regardless of robot type.

```bash
# Homogeneous (default)
python HARL/harl_mapush/train.py --agent0 go1 --agent1 go1

# Heterogeneous
python HARL/harl_mapush/train.py --agent0 go1 --agent1 anymal_c

# Any combination
python HARL/harl_mapush/train.py --agent0 cassie --agent1 anymal_c
```

**Routing:** `go1 + go1` uses native `Go1Object` class. Any other combination uses `HeteroRobot` (multi-URDF, per-agent bodies, independent locomotion policies and actuator networks).

### Adding New Robots

1. Create config + class in `mqe/envs/<robot_name>/`
2. Register in `mqe/envs/robot_registry.py`
3. Add locomotion policy loading, observation construction, actuator network, and reset handling in `mqe/envs/base/hetero_robot.py`

Critical: observation scaling, sign conventions, and `.clone()` for LSTM state modifications must match the robot's locomotion training exactly. See `claude_summaries/heterogeneous_framework/DEFINITIVE_GUIDE_NEW_ROBOTS.md` for the full checklist.

---

## Training

### HAPPO (Recommended)

```bash
# Homogeneous Go1 --- goal-based task
python HARL/harl_mapush/train.py \
    --task go1push_mid \
    --exp_name homogeneous_baseline \
    --n_rollout_threads 500 \
    --num_env_steps 200000000

# Heterogeneous Go1 + Anymal C with anti-freeloading
python HARL/harl_mapush/train.py \
    --task go1push_mid \
    --agent0 go1 --agent1 anymal_c \
    --contact_force_gating True --contact_force_gating_alpha 0.3 \
    --use_concat_agent_observations_critic True \
    --mapush_og_rewards_teamified True \
    --box_mass_range 8 30 \
    --exp_name hetero_go1_anymal \
    --num_env_steps 200000000

# Velocity task
python HARL/harl_mapush/train.py \
    --task go1push_vel \
    --exp_name velocity_baseline \
    --n_rollout_threads 500 \
    --num_env_steps 200000000

# Upper-level planner with frozen HAPPO mid-level
python HARL/harl_mapush/train.py \
    --task go1push_upper \
    --mid_level_checkpoint results/mapush/go1push_mid/happo/<RUN>/checkpoints/190M \
    --mid_level_format happo \
    --n_rollout_threads 10 \
    --num_env_steps 200000000 \
    --exp_name upper_planner

# Resume from checkpoint
python HARL/harl_mapush/train.py \
    --checkpoint results/mapush/.../checkpoints/100M \
    --exp_name resumed_run
```

### MAPPO (Baseline)

```bash
# Via task script
source task/cuboid/train.sh False

# Direct
python openrl_ws/train.py \
    --algo ppo \
    --task go1push_mid \
    --num_envs 500 \
    --train_timesteps 200000000 \
    --use_tensorboard \
    --headless
```

### Output Structure

HAPPO training saves to:
```
results/mapush/<task>/happo/<exp_name>/seed-<seed>-<timestamp>/
  checkpoints/
    10M/
      actor_agent0.pt
      actor_agent1.pt
      critic_agent.pt
      value_normalizer.pt
    20M/, 30M/, ..., 200M/
  logs/                      (TensorBoard)
  config.json                (full experiment config)
  command.txt                (CLI command for reproduction)
  run_config.yaml            (human-readable config)
```

MAPPO saves to `log/MQE/<task>/...` with `rl_model_<steps>_steps/module.pt` checkpoints.

---

## Testing

### HAPPO

```bash
# Calculator mode --- fast parallel statistics
python HARL/harl_mapush/test.py \
    --checkpoint results/mapush/.../checkpoints/100M \
    --task go1push_mid \
    --mode calculator \
    --num_envs 300 --num_episodes 100 \
    --agent0 go1 --agent1 go1

# Viewer mode --- visualization with optional video
python HARL/harl_mapush/test.py \
    --checkpoint results/mapush/.../checkpoints/100M \
    --task go1push_mid \
    --mode viewer \
    --num_episodes 5 \
    --record_video

# Batch test all checkpoints
python HARL/harl_mapush/test.py \
    --checkpoint results/mapush/.../seed-00001-.../ \
    --task go1push_mid \
    --mode calculator \
    --num_envs 300 \
    --all_checkpoints

# Upper task testing (requires mid-level checkpoint)
python HARL/harl_mapush/test.py \
    --checkpoint results/mapush/go1push_upper/.../checkpoints/10M \
    --task go1push_upper \
    --mid_level_checkpoint results/mapush/go1push_mid/.../checkpoints/190M \
    --mode viewer --num_episodes 3

# Velocity task with legacy obs (old checkpoints)
python HARL/harl_mapush/test.py \
    --checkpoint results/mapush/.../checkpoints/190M \
    --task go1push_vel \
    --mode calculator \
    --legacy_vel_obs
```

### MAPPO

```bash
# Viewer
python openrl_ws/test.py \
    --num_envs 1 --algo ppo --task go1push_mid \
    --checkpoint <path>/module.pt \
    --test_mode viewer

# Calculator
python openrl_ws/test.py \
    --num_envs 300 --algo ppo --task go1push_mid \
    --checkpoint <path>/module.pt \
    --test_mode calculator --headless
```

### Statistics

| Task | Metrics |
|------|---------|
| go1push_mid | success rate, collision rate, avg episode length, collaboration degree |
| go1push_vel | direction error (rad/deg), speed error (m/s), box angular vel (rad/s), velocity success rate (%) |
| go1push_upper | success rate, distance-to-target reward, trajectory reward, reach-target reward |

---

## Configuration Reference

### Core Flags

| Flag | Default | Description |
|------|---------|-------------|
| `--task` | `go1push_mid` | Task: `go1push_mid`, `go1push_vel`, `go1push_upper` |
| `--exp_name` | `cuboid_happo` | Experiment name for logging |
| `--seed` | `1` | Random seed |
| `--n_rollout_threads` | 500 | Parallel environments |
| `--num_env_steps` | 200000000 | Total training steps |
| `--episode_length` | 200 | Steps per episode |

### Agent Flags

| Flag | Default | Description |
|------|---------|-------------|
| `--agent0` | `go1` | Robot type for agent 0 |
| `--agent1` | `go1` | Robot type for agent 1 |

### Box Physics

| Flag | Default | Description |
|------|---------|-------------|
| `--box_mass` | None | Fixed box mass override (kg) |
| `--box_mass_range` | None | Uniform random mass range `MIN MAX` per env |

### Reward Shaping

| Flag | Default | Description |
|------|---------|-------------|
| `--mapush_og_rewards_teamified` | False | Use original 7 MAPush rewards as team rewards |
| `--individualized_rewards` | False | Per-agent reward components |
| `--reward_scale_testing` | False | Tuned scales for heavy-box experiments |
| `--collaboration_rewards` | False | Dual-pushing bonus |
| `--cooperation_rewards` | False | Three-tier cooperation bonuses |

### Anti-Freeloading (Contact-Force Gating)

| Flag | Default | Description |
|------|---------|-------------|
| `--contact_force_gating` | False | Enable contact-force balance gating |
| `--contact_force_gating_alpha` | 0.3 | Min gate value (freeloader gets at most alpha of gated rewards) |

**Mechanism:** Reads Isaac Gym `net_contact_force_tensor`, computes horizontal XY contact forces on non-foot bodies per agent, normalizes by agent mass (N/kg), computes balance ratio `min/max`, gates pushing-related rewards. Activates only when box is moving (speed > 0.01 m/s).

### Critic Mode

| Flag | Default | Description |
|------|---------|-------------|
| `--use_box_centered_critic` | False | Box-centered relative coordinates (9d) |
| `--use_goal_centered_critic` | False | Goal-centered relative coordinates (9d) |
| `--use_relative_obs_critic` | False | Robot-to-box relative observations (9d) |
| `--use_concat_agent_observations_critic` | False | Concatenate agent observations (16d) |

### Upper Task

| Flag | Default | Description |
|------|---------|-------------|
| `--mid_level_checkpoint` | None | Path to frozen mid-level actor checkpoint |
| `--mid_level_format` | `openrl` | Mid-level format: `happo` or `openrl` |

### Velocity Task

| Flag | Default | Description |
|------|---------|-------------|
| `--vel_speed_min` | 0.3 | Min commanded speed (m/s) |
| `--vel_speed_max` | 1.0 | Max commanded speed (m/s) |
| `--vel_tracking_scale` | 0.01 | Velocity tracking reward scale |
| `--vel_angular_penalty_scale` | -0.005 | Angular velocity penalty scale |
| `--legacy_vel_obs` | False | Restore old 16/18-dim obs for testing old checkpoints |

### Logging

| Flag | Default | Description |
|------|---------|-------------|
| `--use_tensorboard` | True | Enable TensorBoard logging |
| `--checkpoint` | None | Resume training from checkpoint path |

---

## Repository Structure

```
new-universal-MAPush/
  mqe/                              # Core environment package
    envs/
      base/
        base_task.py                # BaseTask abstract class
        legged_robot.py             # LeggedRobot (multi-agent support)
        hetero_robot.py             # HeteroRobot (multi-URDF, per-agent control)
      go1/
        go1.py                      # Go1 class with Walk-These-Ways locomotion
        go1_config.py
      anymal_c/                     # Anymal C robot class + config
      cassie/                       # Cassie biped robot class + config
      field/
        legged_robot_field.py       # Field environment extensions
      npc/
        go1_object.py               # Interactive objects (box)
      wrappers/
        go1_push_mid_wrapper.py     # Mid-level: obs, rewards, actions
        go1_push_vel_wrapper.py     # Velocity task: direction tracking
        go1_push_upper_wrapper.py   # Upper level: sub-goal planning
        empty_wrapper.py            # Base wrapper class
      configs/
        go1_push_mid_config.py      # Mid-level config
        go1_push_vel_config.py      # Velocity task config
        go1_push_upper_config.py    # Upper level config
      robot_registry.py             # Central robot type registry
      utils.py                      # make_mqe_env(), make_hetero_env(), custom_cfg()
    utils/
      hetero_config.py              # Config merging for heterogeneous setups
      task_registry.py              # Task registration
      terrain/                      # Terrain generation

  HARL/                             # HARL library + MAPush integration
    harl/                           # Core HARL (minimally modified)
      algorithms/actors/happo.py    # HAPPO algorithm
      runners/on_policy_ha_runner.py # Sequential update runner
      models/policy_models/         # Actor network architectures
      envs/mapush/
        mapush_env.py               # MAPush-to-HARL wrapper (~1000 lines)
        mapush_logger.py            # TensorBoard logging
      configs/algos_cfgs/happo.yaml # HAPPO hyperparameters
    harl_mapush/                    # MAPush-specific scripts
      train.py                      # Training entry point (40+ CLI flags)
      test.py                       # Testing (calculator, viewer, batch, video)
      runners/
        mapush_happo_runner.py      # Step-based checkpointing

  openrl_ws/                        # OpenRL / MAPPO integration
    train.py                        # MAPPO training script
    test.py                         # MAPPO testing script
    utils.py                        # OpenRL wrapper
    cfgs/
      ppo.yaml                      # PPO hyperparameters

  task/                             # Task-specific configs and scripts
    cuboid/
      config.py                     # Mid-task overrides
      train.sh                      # MAPPO training launcher
    velocity/
      config.py                     # Velocity task overrides
      train.sh                      # MAPPO velocity training
      train_happo.sh                # HAPPO velocity training

  resources/
    robots/                         # Robot URDFs, meshes, locomotion policies
      go1/, anymal_c/, cassie/, ...
    objects/                        # Box URDFs
      cuboid/SmallBox.urdf          # Default 1.2m box
      arrow.urdf                    # Velocity task direction marker
    actuator_nets/                  # Actuator network weights
    command_nets/                   # Mid-level controller checkpoints
    goals_net/                      # Pretrained upper-level policy

  results/                          # Training outputs
    mapush/<task>/happo/<exp>/      # HAPPO results

  plot_tb.py                        # TensorBoard batch plotting
  plot_success.py                   # Success rate curve plotting
```

---

## Key Technical Notes

### Isaac Gym Buffered State Writes

`set_actor_root_state_tensor_indexed` is a **buffered** command in Isaac Gym's GPU pipeline. A second call before `simulate()` **replaces** the first, it does not merge. Any wrapper that modifies actor states after `_reset_root_states()` must include ALL actors in its indexed call, not just the ones it changed.

### `root_states_npc` is a Copy

`root_states_npc = all_root_states.view(N, -1, 13)[:, A:, :].reshape(-1, 13)` creates a **copy** (non-contiguous slice + reshape). Always read/write live physics data from `all_root_states` directly.

### LSTM Actuator State Resets

For robots with LSTM-based actuator networks (Anymal C), hidden states must be reset on episode termination. Use `.clone()` before in-place modification to avoid "Inplace update to inference tensor" errors.

### OpenRL Checkpoint Frequency

`CheckpointCallback.save_freq` counts `n_calls` (one per `env.step()`), NOT timesteps. Formula: `save_freq = desired_timestep_interval / num_envs`.

### Contact-Force Gating

The anti-freeloading mechanism uses mass-normalized contact forces: `contribution = force / agent_mass` (N/kg). An earlier version used `force * mass_fraction` which amplified the heavier robot's contribution --- if you see this pattern in old code, it is a known bug.

### Velocity Task: Direction-Only Reward

The velocity tracking reward uses `cos_sim` only (no `exp(-speed_error)` term). The speed magnitude term was removed because it uniformly suppressed reward for heterogeneous teams that physically couldn't reach the commanded speed, causing weaker agents to give up entirely.

---

## Troubleshooting

1. **`ImportError: libpython3.8m.so.1.0`**
   ```bash
   export LD_LIBRARY_PATH=/path/to/conda/envs/mapush/lib
   ```

2. **numpy version conflict** --- Isaac Gym requires `numpy <= 1.19.5`, or modify `isaacgym/python/isaacgym/torch_utils.py`: change `np.float` to `np.float32`.

3. **Segfault during rendering** --- A100/A800 GPUs don't support Isaac Gym rendering. Use GeForce GPUs for viewer mode.

4. **OpenRL callback import error** (`partially initialized module ... has no attribute 'BaseCallback'`) --- Comment out `from openrl.runners.common.base_agent import BaseAgent` in `openrl/utils/callback/callback.py`.

5. **PYTHONPATH issues** --- If imports fail, ensure clean environment:
   ```bash
   unset PYTHONPATH
   conda run -n mapush python HARL/harl_mapush/train.py ...
   ```

6. **NaN crashes in upper task** --- Physics corruption (degenerate quaternions) can cascade through the observation pipeline. NaN sanitization is applied automatically in `mapush_env.py` and `test.py`, but if you see NaN-related errors, check that mid-level checkpoint paths are correct and that the frozen actors produce valid output.

---

## Citation

```bibtex
@article{mapush2024,
  title={Learning Multi-Agent Loco-Manipulation for Long-Horizon Quadrupedal Pushing},
  author={Feng, Yuming and Hong, Chuye and Niu, Yaru and Liu, Shiqi and Yang, Yuxiang and Yu, Wenhao and Zhang, Tingnan and Tan, Jie and Zhao, Ding},
  journal={arXiv preprint arXiv:2411.07104},
  year={2024}
}
```

HARL (Heterogeneous-Agent Reinforcement Learning) — used for HAPPO training:

```bibtex
@article{zhong2024harl,
  title={Heterogeneous-Agent Reinforcement Learning},
  author={Zhong, Yifan and Kuba, Jakub Grudzien and Feng, Xidong and Hu, Siyi and Ji, Jiaming and Yang, Yaodong},
  journal={Journal of Machine Learning Research},
  volume={25},
  pages={1--67},
  year={2024},
  url={http://jmlr.org/papers/v25/23-0488.html}
}
```
GitHub: https://github.com/PKU-MARL/HARL

legged_gym — used for training Anymal C and Cassie locomotion policies:

```bibtex
@inproceedings{rudin2022legged,
  title={Learning to Walk in Minutes Using Massively Parallel Deep Reinforcement Learning},
  author={Rudin, Nikita and Hoeller, David and Reist, Philipp and Hutter, Marco},
  booktitle={Conference on Robot Learning (CoRL)},
  year={2022},
  url={https://arxiv.org/abs/2109.11978}
}
```
GitHub: https://github.com/leggedrobotics/legged_gym
