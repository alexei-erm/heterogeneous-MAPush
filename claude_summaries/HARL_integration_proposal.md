# HARL Integration into MAPush — Implementation Report

**Original Proposal Date:** 2025-12-13
**Last Verified:** 2026-03-12
**Status:** ✅ FULLY IMPLEMENTED AND PRODUCTION-READY
**Objective:** Integrate HAPPO from HARL for mid-level and velocity-level controller training (k=2 agents, cuboid task, homogeneous and heterogeneous robot teams)

---

## 1. Requirements — All Met

### Training Requirements
| Requirement | Status | Implementation |
|-------------|--------|----------------|
| Save models/logs to organized folder | ✅ | `results/mapush/<task>/happo/<exp_name>/seed-<seed>-<timestamp>/` |
| Checkpoints every 10M steps | ✅ | Step-based in `MAPushHAPPORunner.run()`, interval = 10,000,000 |
| 3 NNs per checkpoint | ✅ | `actor_agent0.pt`, `actor_agent1.pt`, `critic_agent.pt` + `value_normalizer.pt` |
| Checkpoint folders named `10M/`, `20M/`, etc. | ✅ | `checkpoints/{steps // 1M}M/` |
| TensorBoard logging | ✅ | Per-agent actor metrics, critic metrics, HAPPO factor, custom reward components |
| Config saved with run | ✅ | `config.json` + `command.txt` + `run_config.yaml` per run |

### Testing Requirements
| Requirement | Status | Implementation |
|-------------|--------|----------------|
| Calculator mode (multi-env stats) | ✅ | `--mode calculator --num_envs 300 --num_episodes 100` |
| Viewer mode (sequential visualization) | ✅ | `--mode viewer --num_episodes 5` |
| Success rate | ✅ | From `env.finished_buf` per episode |
| Collision rate | ✅ | From `env.collision_degree_buf` |
| Finished time | ✅ | `avg_episode_length * 0.02s` |
| Collaboration degree | ✅ | From `env.collaboration_degree_buf` |
| Batch checkpoint testing | ✅ | `--all_checkpoints` tests all 10M/20M/.../200M and outputs summary table + `test_results.txt` |
| Video recording | ✅ | `--record_video` saves MP4s via imageio to `docs/video/` |
| Velocity task metrics | ✅ | Direction error, speed error, box angular velocity, velocity success rate |

---

## 2. Architecture — Option A Implemented (with Extensions)

The original proposal recommended **Option A: Minimal Modification** — wrap MAPush as a HARL environment, keep MAPush-specific code separate. This was implemented with significant extensions beyond the original scope.

### Implemented Directory Structure

```
HARL/
├── harl/                                    # Core HARL library (minimally modified)
│   ├── envs/
│   │   └── mapush/                          # MAPush environment interface
│   │       ├── __init__.py                  # Exports MAPushEnv, MAPushLogger
│   │       ├── mapush_env.py                # Environment wrapper (~900 lines)
│   │       └── mapush_logger.py             # Custom logger (~145 lines)
│   ├── utils/
│   │   └── envs_tools.py                    # Modified: mapush registration
│   └── envs/
│       └── __init__.py                      # Modified: LOGGER_REGISTRY["mapush"]
│
├── harl_mapush/                             # MAPush-specific training/testing
│   ├── train.py                             # Training script (~275 lines, 40+ CLI flags)
│   ├── test.py                              # Testing script (~700+ lines, 2 modes)
│   └── runners/
│       └── mapush_happo_runner.py           # Custom runner with step-based checkpoints
│
└── results/
    └── mapush/
        ├── go1push_mid/happo/               # 69 seed runs across 43 experiments
        └── go1push_vel/happo/               # 19 seed runs across 12 experiments
```

### What Changed vs. the Original Proposal

| Component | Proposal (Dec 2025) | Actual Implementation (Mar 2026) |
|-----------|---------------------|----------------------------------|
| `mapush_env.py` | ~80 lines, basic wrapper | ~900 lines: 5 critic modes, hetero routing, velocity global state, statistics tracking |
| `mapush_logger.py` | ~15 lines, minimal | ~145 lines: per-reward TensorBoard logging, velocity metrics, OpenRL-compatible reward buffer reading |
| `train.py` | ~50 lines, 6 flags | ~275 lines, **40+ CLI flags** for rewards, critic, heterogeneous agents, box mass, velocity task |
| `test.py` | ~250 lines, basic calc/viewer | ~700+ lines: `--all_checkpoints` batch testing, video recording, velocity-specific metrics, `--legacy_vel_obs` |
| `mapush_happo_runner.py` | ~120 lines, basic override | Full run() override with step-based checkpointing, config saving, `command.txt` generation |
| YAML config | `mapush.yaml` proposed | **Not used** — all config via `env_args` dict and CLI flags (more flexible) |
| Tasks supported | `go1push_mid` only | `go1push_mid` + `go1push_vel` (velocity-MAPush) |
| Robot support | Homogeneous Go1 only | **3 robots**: go1, anymal_c, cassie via `--agent0`/`--agent1` |
| Critic modes | EP only, obs=actor_obs | 5 modes: absolute (11d), box-centered (9d), goal-centered (9d), relative (9d), concat-agent (16d) |
| Reward variants | None | 10+ reward configurations: individualized, teamified, gated, collaboration bonuses, contact-force gating |

---

## 3. Key Components — Detailed

### 3.1 Environment Wrapper (`mapush_env.py`)

**Role:** Bridges Isaac Gym MAPush environment to HARL's interface.

**Environment Creation — Two Paths:**
```python
agent0 = env_args.get("agent0", "go1")
agent1 = env_args.get("agent1", "go1")
is_hetero = (agent0 != agent1) or (agent0 != 'go1')

if is_hetero:
    env, env_cfg = make_hetero_env(task, [agent0, agent1], args, custom_cfg)
else:
    env, env_cfg = make_mqe_env(task, args, custom_cfg)
```

**Interface Shapes (HARL standard):**
| Method | Input | Output |
|--------|-------|--------|
| `step(actions)` | `[n_envs, n_agents, 3]` | obs `[n_envs, n_agents, obs_dim]`, state `[n_envs, n_agents, global_dim]`, rewards `[n_envs, n_agents, 1]`, dones `[n_envs, n_agents]`, infos `[list]`, available_actions `None` |
| `reset()` | — | obs, state, available_actions |

**Global State (Centralized Critic) — 5 Modes:**

| Mode | CLI Flag | Dims | Description |
|------|----------|------|-------------|
| Absolute (default) | — | 11 | box(3) + target(2) + agent0(3) + agent1(3) |
| Box-centered | `--use_box_centered_critic` | 9 | target_rel(2) + agents_rel(3 each) + box_yaw(1) |
| Goal-centered | `--use_goal_centered_critic` | 9 | box_rel(3) + agents_rel(3 each) |
| Relative | `--use_relative_obs_critic` | 9 | robots_to_box(3 each) + inter_dist(1) + goal_to_box(2) |
| Concat-agent | `--use_concat_agent_observations_critic` | 16 | agent0_obs(8) + agent1_obs(8) |

**Velocity Task Global State:**
- Modern (default): 15 dims — cmd_dir(2) + box_dynamics(3) + agents(5 each), all in box frame
- Legacy (`--legacy_vel_obs`): 18 dims — adds cmd_speed(1) + speed_error(1) + dir_error(1)

**State Type:** Always EP (Environment Provided) — single global state broadcast to all agents.

### 3.2 Custom Runner (`mapush_happo_runner.py`)

**Extends:** `OnPolicyHARunner` → `OnPolicyBaseRunner`

**Key customizations:**
1. **Step-based checkpoints** (not episode-based): checks every rollout step, saves when `total_steps >= last_checkpoint + 10M`
2. **Disables HARL evaluation** (`use_eval = False`) — Isaac Gym can't run multiple env instances
3. **Saves full config** at init: `config.json`, `command.txt`, `run_config.yaml`
4. **Checkpoint contents**: `actor_agent{i}.pt` + `critic_agent.pt` + `value_normalizer.pt`

**Training loop:** Standard HARL on-policy HA runner with sequential agent updates (HAPPO's importance-weighted scheme), plus the step counter and checkpoint logic inserted in the inner loop.

### 3.3 Training Script (`train.py`)

**40+ CLI flags organized in groups:**

| Group | Key Flags |
|-------|-----------|
| Core | `--algo happo`, `--task {go1push_mid, go1push_vel}`, `--exp_name`, `--seed` |
| Training | `--n_rollout_threads 500`, `--num_env_steps 200000000`, `--episode_length 200` |
| Heterogeneous | `--agent0 {go1, anymal_c, cassie}`, `--agent1 {same}` |
| Box physics | `--box_mass N`, `--box_mass_range MIN MAX` |
| Reward shaping | `--individualized_rewards`, `--mapush_og_rewards_teamified`, `--reward_scale_testing`, `--collaboration_rewards`, `--cooperation_rewards` |
| Contact gating | `--contact_force_gating`, `--contact_force_gating_alpha 0.3` |
| Critic mode | `--use_box_centered_critic`, `--use_goal_centered_critic`, `--use_relative_obs_critic`, `--use_concat_agent_observations_critic` |
| Velocity task | `--vel_speed_min`, `--vel_speed_max`, `--vel_tracking_scale`, `--vel_angular_penalty_scale`, `--legacy_vel_obs` |
| Logging | `--use_tensorboard True`, `--checkpoint <path>` (resume) |

**Flow:** Parse args → load HAPPO YAML defaults → build `env_args` dict → `import isaacgym` → create `MAPushHAPPORunner(args, algo_args, env_args)` → optionally `restore_checkpoint()` → `runner.run()` → `runner.close()`

### 3.4 Testing Script (`test.py`)

**Modes:**

| Mode | Usage | What It Does |
|------|-------|--------------|
| Calculator | `--mode calculator --num_envs 300` | Runs N parallel envs, collects statistics over `--num_episodes` episodes, prints summary |
| Viewer | `--mode viewer --num_episodes 5` | Single env with rendering, plays episodes sequentially |
| Batch | `--all_checkpoints --checkpoint <run_dir>` | Tests every checkpoint (10M..200M), prints summary table, saves `test_results.txt` |

**Model Loading:** Reads `config.json` from 2 levels above checkpoint dir → creates HAPPO actors with matching architecture → loads `actor_agent{i}.pt` weights → eval mode.

**Statistics (go1push_mid):** success_rate, collision_rate, avg_episode_length, collaboration_degree
**Statistics (go1push_vel):** direction_error (rad/deg), speed_error (m/s), box_angular_vel (rad/s), velocity_success_rate (%)

### 3.5 Logger (`mapush_logger.py`)

**Extends:** `BaseLogger`

**Custom behavior:**
- Reads reward components from wrapper's `reward_buffer` dict (OpenRL-compatible accumulation pattern)
- Logs each reward term to TensorBoard as `rewards/<component_name>`
- Logs aggregate `average_step_reward`
- For velocity task: logs direction error, speed error, box angular velocity
- For mid task: logs success rate from `envs.get_statistics()`

### 3.6 HARL Registration Points

| File | Registration |
|------|-------------|
| `harl/envs/__init__.py` | `LOGGER_REGISTRY["mapush"] = MAPushLogger` |
| `harl/utils/envs_tools.py` | `make_train_env()` → imports and returns `MAPushEnv` |
| `harl/utils/envs_tools.py` | `make_eval_env()` → raises `NotImplementedError` (Isaac Gym single-instance) |
| `harl/utils/envs_tools.py` | `get_num_agents()` → returns `envs.n_agents` |

**Note:** No `mapush.yaml` config file exists. Configuration is handled entirely through the `env_args` dictionary constructed in `train.py`, which is more flexible for the many experimental flags used.

---

## 4. Results Directory Structure

```
results/mapush/<task>/happo/<exp_name>/seed-<seed>-<timestamp>/
├── checkpoints/
│   ├── 10M/
│   │   ├── actor_agent0.pt      (~278 KB)
│   │   ├── actor_agent1.pt      (~278 KB)
│   │   ├── critic_agent.pt      (~414 KB)
│   │   └── value_normalizer.pt  (~513 B)
│   ├── 20M/, 30M/, ..., 200M/
│   └── test_results.txt         (from --all_checkpoints testing)
├── logs/
│   ├── agent0/                  (dist_entropy, policy_loss, actor_grad_norm, ratio)
│   ├── agent1/                  (same structure)
│   ├── critic/                  (critic_grad_norm, value_loss)
│   ├── happo_factor/            (agent0/, agent1/ — mean, max, min)
│   └── events.out.tfevents.*    (main TensorBoard file)
├── models/                      (unused — HARL default, we use checkpoints/ instead)
├── config.json                  (~4.4 KB, full algo_args + env_args + main_args)
├── command.txt                  (~1.2 KB, exact CLI command for reproduction)
├── run_config.yaml              (~2.9 KB, human-readable config)
└── progress.txt                 (empty)
```

**Scale:** 88 total seed runs across 55 experiments (69 mid-task + 19 velocity-task).

---

## 5. Usage Examples

### Training

```bash
# Homogeneous Go1 — goal-based task
python HARL/harl_mapush/train.py \
    --task go1push_mid \
    --exp_name homogeneous_baseline \
    --n_rollout_threads 500 --num_env_steps 200000000

# Heterogeneous Go1+Anymal — with contact-force anti-freeloading
python HARL/harl_mapush/train.py \
    --task go1push_mid \
    --agent0 go1 --agent1 anymal_c \
    --contact_force_gating True --contact_force_gating_alpha 0.3 \
    --use_concat_agent_observations_critic True \
    --mapush_og_rewards_teamified True \
    --box_mass_range 8 30 \
    --exp_name hetero_go1_anymal --num_env_steps 200000000

# Velocity task — homogeneous
python HARL/harl_mapush/train.py \
    --task go1push_vel \
    --exp_name velocity_homogeneous \
    --n_rollout_threads 500 --num_env_steps 200000000

# Resume from checkpoint
python HARL/harl_mapush/train.py \
    --checkpoint results/mapush/.../checkpoints/100M \
    --exp_name resumed_run
```

### Testing

```bash
# Calculator mode — single checkpoint
python HARL/harl_mapush/test.py \
    --checkpoint results/mapush/.../checkpoints/100M \
    --task go1push_mid --mode calculator \
    --num_envs 300 --num_episodes 100 \
    --agent0 go1 --agent1 go1

# Viewer mode with video recording
python HARL/harl_mapush/test.py \
    --checkpoint results/mapush/.../checkpoints/100M \
    --task go1push_mid --mode viewer \
    --num_episodes 5 --record_video

# Batch test all checkpoints
python HARL/harl_mapush/test.py \
    --checkpoint results/mapush/.../seed-00001-2026-03-08/ \
    --task go1push_mid --mode calculator \
    --num_envs 300 --all_checkpoints

# Velocity task with legacy obs (old checkpoints)
python HARL/harl_mapush/test.py \
    --checkpoint results/mapush/.../checkpoints/190M \
    --task go1push_vel --mode calculator \
    --box_mass 8 --legacy_vel_obs
```

---

## 6. Heterogeneous Robot Support

The integration supports **3 robot types** via `--agent0`/`--agent1` flags:

| Robot | Type | DOFs | Control | Actuator Network |
|-------|------|------|---------|------------------|
| Robot | Type | DOFs | Locomotion Policy | Actuator Stage |
|-------|------|------|-------------------|----------------|
| go1 | Quadruped | 12 | Walk-These-Ways MLP (70-dim obs + 2100 history) | Feedforward network (`unitree_go1.pt`) |
| anymal_c | Quadruped | 12 | legged_gym MLP (48-dim obs) | LSTM network (`anydrive_v3_lstm.pt`) |
| cassie | Biped | 12 | legged_gym MLP [128,64,32] ELU (48-dim obs) | PD control (no actuator network) |

**All robots expose a unified 3-action interface** `[vx, vy, vyaw]`, so HARL sees homogeneous action spaces regardless of robot heterogeneity.

**Routing logic in `mapush_env.py`:**
- `go1 + go1` → `make_mqe_env()` (native Go1Object class)
- Any other combination → `make_hetero_env()` (HeteroRobot base, 1444 lines)

**Core hetero infrastructure** (outside HARL, in mqe/):
- `mqe/envs/robot_registry.py` (287 lines) — robot class/config lookup
- `mqe/envs/base/hetero_robot.py` (1444 lines) — multi-URDF, per-agent bodies, action routing
- `mqe/utils/hetero_config.py` (332 lines) — config merging, validation

---

## 7. Anti-Freeloading: Contact-Force Gating

**Problem:** In heterogeneous teams (e.g., Go1 12kg + Anymal 50kg), the weaker agent freeloads because the stronger agent can solve the task solo.

**Mechanism:** Reads Isaac Gym `net_contact_force_tensor`, computes horizontal XY contact forces on non-foot bodies per agent, normalizes by agent mass (N/kg), computes balance ratio, gates reward.

```
contribution_i = horizontal_force_i / agent_mass_i
balance = min(contributions) / max(contributions)
gate = alpha + (1 - alpha) * balance
gated_rewards *= gate
```

**Trigger:** Box velocity > 0.01 m/s (replaces earlier force threshold).

**Evolution:**
1. **v1 (2026-02-25):** `force × mass_fraction` — WRONG, amplified heavy robot
2. **v2 (2026-03-07):** `force / agent_mass` — CORRECT, equal effort → balance≈1.0
3. **v3 (2026-03-12):** Box-velocity trigger instead of force threshold

**Flags:** `--contact_force_gating True --contact_force_gating_alpha 0.3`

**Gated rewards (mid task):** `push_reward`, `distance_to_target_reward`
**Gated rewards (vel task):** `velocity_tracking_reward` (but found less effective — velocity task better served by heavy box mass randomization)

---

## 8. Velocity-MAPush Task Integration

Added alongside the original `go1push_mid` task with full HARL support:

| Aspect | go1push_mid (Goal) | go1push_vel (Velocity) |
|--------|-------------------|----------------------|
| Objective | Push box to target position | Push box in commanded direction |
| Actor obs | 8 dims (egocentric) | 15 dims (egocentric, direction-only) |
| Critic state | 9-16 dims (5 modes) | 15 dims (box-frame) or 18 dims (legacy) |
| Primary reward | Distance to target | Cosine similarity with commanded direction |
| Success metric | Binary (reached target) | Velocity tracking success rate (%) |
| Episode end | Reached target or timeout | Timeout only |

**Key design decision:** Velocity tracking reward uses `cos_sim` only (direction), NOT `cos_sim × exp(-speed_error)`. The `exp(-speed_error)` term uniformly suppressed reward for heterogeneous teams that physically couldn't reach commanded speed, causing weaker agents to give up.

---

## 9. Resolved Design Questions

| Question (from original proposal) | Resolution |
|------------------------------------|------------|
| EP vs FP state type? | **EP always** — single global state broadcast to all agents |
| Save value normalizer? | **Yes** — `value_normalizer.pt` in every checkpoint |
| Checkpoint on failure? | **Not implemented** — no crash-save handler (training is stable enough) |
| TensorBoard: default + custom? | **Custom only** — `MAPushLogger` reads from wrapper `reward_buffer`, logs all components individually |
| YAML config file? | **Skipped** — `env_args` dict from CLI flags is more flexible for 40+ experimental parameters |

---

## 10. Comparison: Proposal vs. Implementation

| Aspect | Original Proposal (Dec 2025) | Final Implementation (Mar 2026) |
|--------|-----------------------------|---------------------------------|
| Scope | 1 task, homogeneous Go1 | 2 tasks, 3 robots, heterogeneous |
| `mapush_env.py` size | ~80 lines | ~900 lines |
| `train.py` flags | 6 flags | 40+ flags |
| `test.py` features | Basic calc/viewer | + batch testing, video, velocity metrics, legacy obs |
| Critic modes | 1 (EP, obs=actor_obs) | 5 modes (absolute, box-centered, goal-centered, relative, concat) |
| Reward variants | None (default rewards) | 10+ configurations including contact-force gating |
| Anti-freeloading | Not considered | Contact-force gating with mass normalization |
| Robot types | Go1 only | go1, anymal_c, cassie |
| Experiments run | 0 | 88 seed runs across 55 experiments |
| Estimated timeline | 7-10 days | Implemented over ~3 months of iterative development |

---

## 11. File Reference

### Created Files (HARL-side)

| File | Lines | Purpose |
|------|-------|---------|
| `HARL/harl/envs/mapush/__init__.py` | ~10 | Exports MAPushEnv, MAPushLogger |
| `HARL/harl/envs/mapush/mapush_env.py` | ~900 | Environment wrapper with 5 critic modes, hetero routing, velocity state |
| `HARL/harl/envs/mapush/mapush_logger.py` | ~145 | TensorBoard logging, reward component tracking |
| `HARL/harl_mapush/train.py` | ~275 | Training entry point with 40+ CLI flags |
| `HARL/harl_mapush/test.py` | ~700+ | Testing with calculator, viewer, batch, video modes |
| `HARL/harl_mapush/runners/mapush_happo_runner.py` | ~240 | Custom runner with step-based checkpoints |

### Modified Files (HARL core — minimal)

| File | Change |
|------|--------|
| `HARL/harl/envs/__init__.py` | Added `"mapush": MAPushLogger` to `LOGGER_REGISTRY` |
| `HARL/harl/utils/envs_tools.py` | Added mapush to `make_train_env()`, `make_eval_env()` (raises NotImplementedError), `get_num_agents()` |

### Supporting Infrastructure (mqe-side, outside HARL)

| File | Lines | Purpose |
|------|-------|---------|
| `mqe/envs/robot_registry.py` | 287 | Robot class/config registry (3 working robots: go1, anymal_c, cassie) |
| `mqe/envs/base/hetero_robot.py` | 1444 | Multi-robot environment with different URDFs |
| `mqe/utils/hetero_config.py` | 332 | Config merging for heterogeneous setups |
| `mqe/envs/utils.py` | 300+ | `make_mqe_env()`, `make_hetero_env()`, `custom_cfg()` |
| `mqe/envs/wrappers/go1_push_mid_wrapper.py` | — | Mid-task wrapper with contact-force gating |
| `mqe/envs/wrappers/go1_push_vel_wrapper.py` | — | Velocity-task wrapper with gating + direction-only tracking |
