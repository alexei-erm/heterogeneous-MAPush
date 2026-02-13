# Velocity-MAPush: Implementation Guide

How the code is structured, what each file does, and how they connect.

---

## Overview

Velocity-MAPush (`go1push_vel`) is a new task variant registered alongside the existing `go1push_mid` task. It shares the same robot/NPC/terrain infrastructure but has entirely different observations, rewards, and success criteria. The implementation touches **6 new files** and **5 modified files** across two training pipelines (MAPPO via OpenRL, HAPPO via HARL).

---

## File Map

```
NEW FILES:
  mqe/envs/configs/go1_push_vel_config.py    Config (inherits Go1PushMidCfg)
  mqe/envs/wrappers/go1_push_vel_wrapper.py  Wrapper (inherits EmptyWrapper)
  task/velocity/config.py                     Task-specific overrides
  task/velocity/train.sh                      MAPPO training script
  task/velocity/train_happo.sh                HAPPO training script

MODIFIED FILES:
  mqe/envs/utils.py                           Registration + custom_cfg velocity params
  HARL/harl/envs/mapush/mapush_env.py         Critic global state + velocity routing
  HARL/harl_mapush/train.py                   Velocity CLI flags
  HARL/harl_mapush/test.py                    Velocity evaluation metrics
  openrl_ws/train.py                          Velocity param extraction
  openrl_ws/utils.py                          Velocity CLI param definitions
```

---

## Dependency Graph

```
go1_push_vel_config.py
    │ inherits Go1PushMidCfg
    │ defines: velocity_command, goal (disabled), reward scales
    ▼
go1_push_vel_wrapper.py
    │ reads config, builds obs (16-dim), computes rewards (6 terms)
    │ manages velocity commands, arrow marker
    ▼
mqe/envs/utils.py
    │ registers "go1push_vel" → {Go1Object, Go1PushVelCfg, Go1PushVelWrapper}
    │ custom_cfg() handles velocity param overrides
    ▼
┌───────────────────────────────┐    ┌───────────────────────────────┐
│ HARL Pipeline (HAPPO)         │    │ OpenRL Pipeline (MAPPO)       │
│                               │    │                               │
│ mapush_env.py                 │    │ openrl_ws/train.py            │
│   is_velocity_task flag       │    │   vel param extraction        │
│   _construct_vel_global_state │    │   disables mid-task rewards   │
│   18-dim critic state         │    │                               │
│                               │    │ openrl_ws/utils.py            │
│ HARL/harl_mapush/train.py     │    │   CLI param definitions       │
│   --task go1push_vel          │    │                               │
│   --vel_speed_min/max etc.    │    │                               │
│                               │    │                               │
│ HARL/harl_mapush/test.py      │    │                               │
│   velocity metrics display    │    │                               │
└───────────────────────────────┘    └───────────────────────────────┘
    ▲                                     ▲
    │                                     │
task/velocity/train_happo.sh         task/velocity/train.sh
task/velocity/config.py (box mass=8kg, shared by both pipelines)
```

---

## File-by-File Details

### 1. `mqe/envs/configs/go1_push_vel_config.py` — NEW

**Class:** `Go1PushVelCfg(Go1PushMidCfg)`

Inherits everything from the mid-task config (terrain, asset, init_state, domain_rand, termination, command, control). Overrides three things:

**`velocity_command` class (new):**
- `speed_range = [0.3, 1.0]` — commanded speed sampled uniformly in m/s
- `direction_range = [0, 2*pi]` — commanded direction sampled uniformly in radians
- `arrow_offset = 2.0` — visual marker placed 2m ahead of box in commanded direction

**`goal` class (override — disables goal-reaching):**
- All four goal modes set to `False`
- `THRESHOLD = 99999.0` — makes `finished_buf` always False (box never "reaches" the target)
- No `check_setting` validation (the parent's runs at import time on the parent's own goal class, which still has `random_goal_pos=True`)

**`rewards.scales` class (override):**
| Scale | Value | Notes |
|-------|-------|-------|
| `velocity_tracking_scale` | 0.01 | NEW — primary reward |
| `angular_velocity_penalty_scale` | -0.005 | NEW — cooperation mechanism |
| `approach_reward_scale` | 0.00075 | Reused from mid |
| `collision_punishment_scale` | -0.0025 | Reused from mid |
| `push_reward_scale` | 0.0015 | Reused from mid |
| `exception_punishment_scale` | -5 | Reused from mid |
| `target_reward_scale` | 0.0 | Disabled (no target) |
| `reach_target_reward_scale` | 0.0 | Disabled |
| `ocb_reward_scale` | 0.0 | Disabled |
| `proximity_penalty_scale` | 0.0 | Disabled |

---

### 2. `mqe/envs/wrappers/go1_push_vel_wrapper.py` — NEW (~490 lines)

**Class:** `Go1PushVelWrapper(EmptyWrapper)`

Does **not** inherit from `Go1PushMidWrapper`. Clean implementation with velocity-specific logic.

**Key state buffers:**
- `cmd_direction` — `(num_envs,)` tensor, direction in [0, 2pi] world frame
- `cmd_speed` — `(num_envs,)` tensor, speed in m/s
- `physics_exception_buf` — NaN detection
- `reward_buffer` — dict accumulating reward terms + metrics for tensorboard

**Key methods:**

| Method | Purpose |
|--------|---------|
| `_sample_velocity_commands(env_ids)` | Sample random direction + speed for given envs |
| `_update_arrow_marker()` | Reposition target NPC as directional arrow via `set_actor_root_state_tensor_indexed` |
| `_build_obs(base_pos, base_rpy, base_vel, box_lin_vel, box_ang_vel_z)` | Build 16-dim agent-centric observations |
| `reset()` | Reset all envs, sample new commands, build obs with zero velocities |
| `step(action)` | Step physics, detect resets, build obs, compute 6 reward terms |

**Arrow marker implementation detail:**
The target NPC (index 1 in `root_states_npc`) is repurposed as a visual arrow. Each step its position is set to `box_pos + [cos(theta), sin(theta)] * arrow_offset`. Because `root_states_npc` is a **copy** (not a view) of `all_root_states` due to non-contiguous slicing, the code must explicitly write the target NPC state back to `all_root_states` via the target actor's flat index, then call `set_actor_root_state_tensor_indexed`.

**Reset detection:**
Uses `self.env.reset_ids` (set in `legged_robot.py:243`) to detect which envs just auto-reset mid-episode. New velocity commands are sampled only for those envs.

**Team reward pattern:**
All 6 reward terms are computed per-environment, then broadcast identically to all agents:
```python
reward[:, :] += term.unsqueeze(1).repeat(1, num_agents)
```
Finally, rewards are summed across agents and re-broadcast:
```python
team_reward = reward.sum(dim=1, keepdim=True)
reward = team_reward.repeat(1, num_agents)
```

---

### 3. `mqe/envs/utils.py` — MODIFIED

**Registration (ENV_DICT):**
```python
"go1push_vel": {
    "class": Go1Object,
    "config": Go1PushVelCfg,
    "wrapper": Go1PushVelWrapper
}
```

**`custom_cfg()` signature extended** with 4 optional velocity params:
- `vel_speed_min`, `vel_speed_max` — override `cfg.velocity_command.speed_range`
- `vel_tracking_scale`, `vel_angular_penalty_scale` — override `cfg.rewards.scales.*`

These are only applied when `hasattr(cfg, 'velocity_command')` is True, so they're silently ignored for non-velocity tasks.

---

### 4. `HARL/harl/envs/mapush/mapush_env.py` — MODIFIED

**Detection:**
```python
self.is_velocity_task = (args.task == "go1push_vel")
```

**Critic state dimension:**
```python
if self.is_velocity_task:
    global_state_dim = 3 + 3 + 5 * self.n_agents + 2  # = 18 for 2 agents
```

**`_construct_vel_global_state()` method:**
Produces an 18-dim state in the **box-centered frame** (all positions, velocities, and commands rotated by `-box_yaw`). This gives the critic translation + rotation invariance. See the RL-explained doc for the full breakdown.

**Routing in `step()` and `reset()`:**
```python
if self.is_velocity_task:
    global_state_np = self._construct_vel_global_state()
elif self.use_relative_obs_critic:
    ...
```

**Episode tracking:** Velocity task episodes append `success=False` since there's no binary success metric.

---

### 5. `HARL/harl_mapush/train.py` — MODIFIED

**New CLI args:**
- `--task go1push_vel` (added to choices)
- `--vel_speed_min`, `--vel_speed_max`, `--vel_tracking_scale`, `--vel_angular_penalty_scale`

These are extracted, passed into `env_args`, and flow through to `custom_cfg()`.

---

### 6. `HARL/harl_mapush/test.py` — MODIFIED

**New `--task` argument** with choices `["go1push_mid", "go1push_vel"]`.

**Velocity metrics display** (when `is_velocity_task` is True):
- Avg direction error (rad and deg)
- Avg speed error (m/s)
- Avg box angular velocity (rad/s)
- Avg velocity tracking reward
- Avg angular velocity penalty

---

### 7. `openrl_ws/train.py` — MODIFIED

- Extracts velocity params from args via `getattr`
- Disables `baseline_mappo_rewards` and `mappo_heavybox_rewards` for velocity task (these are mid-task-specific reward presets)
- Passes velocity params through to `custom_cfg()`
- Task folder copy handles both `go1push_mid` and `go1push_vel`

---

### 8. `openrl_ws/utils.py` — MODIFIED

Added 4 velocity CLI parameter definitions to the `custom_parameters` list:
`--vel_speed_min`, `--vel_speed_max`, `--vel_tracking_scale`, `--vel_angular_penalty_scale`

---

### 9. `task/velocity/config.py` — NEW

Task-specific config for experiments. Currently overrides:
- `npc_mass_override = 8` (lighter 8kg box for faster learning; default is heavier)

Class shadows the imported parent name (`Go1PushVelCfg(Go1PushVelCfg)`) — same pattern as `task/cuboid/config.py`.

---

### 10. `task/velocity/train.sh` — NEW

MAPPO training script. Key settings:
- 500 parallel envs, 200M training steps
- 2-layer MLP with hidden size 128
- Seeds 1, task `go1push_vel`
- After training: runs calculator mode at every 10M step checkpoint
- **Does not call `update_config.py`** (that script writes to `go1_push_mid_config.py`, wrong target)

---

### 11. `task/velocity/train_happo.sh` — NEW

HAPPO training script. Key settings:
- 500 rollout threads, 200M env steps
- `--use_relative_obs_critic False` (velocity task uses its own dedicated critic state)

---

## Key Implementation Decisions

### Why EmptyWrapper instead of Go1PushMidWrapper?
The mid-task wrapper's observations, rewards, and episode logic are fundamentally different. Inheriting would mean overriding almost everything while carrying dead code and risking subtle interactions. A clean EmptyWrapper base is simpler and safer.

### Why `THRESHOLD = 99999.0` instead of modifying `check_termination()`?
The termination check in `legged_robot.py` compares box-target distance against `THRESHOLD`. Setting it to 99999 makes `finished_buf` always False without changing any base class code. Episodes run to `max_episode_length` and only reset on timeout or physics exceptions.

### Why `root_states_npc` needs explicit writeback?
`root_states_npc = all_root_states.view(N, -1, 13)[:, A:, :].reshape(-1, 13)` — the view + non-contiguous slicing + reshape creates a **copy**, not a view. Writing to `root_states_npc` does NOT update `all_root_states`. The arrow marker code must explicitly copy the target NPC state back to `all_root_states` at the correct flat index before calling `set_actor_root_state_tensor_indexed`.

### Why zero velocities on reset?
After `env.reset()`, the physics simulation hasn't stepped yet — all velocities are zero. Passing zero tensors to `_build_obs` on reset is physically correct and avoids reading stale velocity data from the previous episode.

---

## Running

**MAPPO training:**
```bash
cd /home/gvlab/new-universal-MAPush
bash task/velocity/train.sh
```

**HAPPO training:**
```bash
cd /home/gvlab/new-universal-MAPush
bash task/velocity/train_happo.sh
```

**Quick sanity check (no GPU needed for this):**
```bash
python -c "from mqe.envs.utils import ENV_DICT; print('go1push_vel' in ENV_DICT)"
# Should print: True
```

**Visual test (single env, renders):**
```bash
bash task/velocity/train.sh True
```
