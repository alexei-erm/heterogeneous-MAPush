# Session Summary — 2026-02-13 (Part 2)

## Context
Continuing from part 1. MAPPO velocity training had been launched. This session focused on fixing the test framework, debugging missing checkpoints, adding a velocity success rate metric, and tuning reward functions.

## Bugs Found & Fixed

### 1. No checkpoints saved during MAPPO training (CRITICAL)
- `openrl_ws/train.py` had `CheckpointCallback(save_freq=10000000)`
- `save_freq` counts `n_calls` (one per `env.step()` call), NOT timesteps
- With `num_envs=500`, each `env.step()` = 500 timesteps, so max `n_calls` = 400,000 for 200M training — never reaches 10M threshold
- **Fix**: Changed `save_freq` from `10000000` to `20000` (= 10M timesteps / 500 envs)
- This produces checkpoints at 10M, 20M, ..., 200M timesteps (20 total)
- The mid-task worked because at commit `582be23` the value was `20000` — it was incorrectly changed to `10000000` on the velocity branch
- File: `openrl_ws/train.py` line 123

### 2. openrl_ws/test.py calculator mode crashes on velocity task
- Calculator mode accessed `env.init_finished_buf`, `env.collision_degree_buf`, `env.init_episode_length_buf` which don't exist on velocity wrapper
- **Fix**: Added `is_velocity_task` detection, velocity-specific calculator using `env.env.reward_buffer`
- File: `openrl_ws/test.py`

### 3. openrl_ws/test.py viewer mode crashes on velocity task
- Viewer mode accessed `env.finished_buf` which doesn't exist on velocity wrapper
- **Fix**: Velocity branch prints metrics from `reward_buffer` instead
- File: `openrl_ws/test.py`

### 4. HARL test.py --all_checkpoints summary meaningless for velocity
- Printed `success_rate=0` for velocity task
- **Fix**: Detects velocity task, prints velocity metric columns + success rate instead
- File: `HARL/harl_mapush/test.py`

### 5. test_all_checkpoints_velocity.sh path handling
- Script expected run folder but user passed checkpoints dir → looked in `checkpoints/checkpoints/`
- **Fix**: Auto-detects if `basename` is "checkpoints" and goes up one level
- File: `test_all_checkpoints_velocity.sh`

## New Features

### Velocity Tracking Success Rate metric
- Formula: `avg_vel_track_reward_per_step / velocity_tracking_scale`
- Perfect tracking (cos_sim=1, speed_error=0) → 100%
- Uses config value (`velocity_tracking_scale`) at runtime — auto-adapts if scale is changed
- Added to:
  - `openrl_ws/utils.py` — tensorboard logging via `batch_rewards()` as `velocity_tracking_success_rate`
  - `openrl_ws/test.py` — calculator output + viewer episode printout
  - `HARL/harl_mapush/test.py` — single calculator, `--all_checkpoints` summary table + file output, viewer mode

### Post-training calculator loop
- `task/velocity/train.sh` now runs calculator mode on all checkpoints after training completes
- Saves results to `<run_folder>/velocity_metrics.txt`

### test_all_checkpoints_velocity.sh (new script)
- Standalone script to run MAPPO calculator on all checkpoints in a run folder
- Parses agent0/agent1/algo from `command.txt` with defaults
- Usage: `./test_all_checkpoints_velocity.sh log/MQE/go1push_vel/velocity/run2`

## Reward Tuning

### Collision punishment made sharper
- **Old**: `1 / (0.02 + d/3)` — penalty at 1m = -0.003, at 2m = -0.001
- **New**: `1 / (0.02 + d/0.5)` — penalty at 1m = -0.0005, at 2m = -0.00025
- Same close-range (d=0) penalty (-0.05), but 6x less at working distance (1m)
- Agents flanking box at normal distance barely penalized
- File: `mqe/envs/wrappers/go1_push_vel_wrapper.py`

### OCB reward redesign (DISCUSSED, NOT YET IMPLEMENTED)
- Current: linear dot product with push direction — peaks at 180° (directly behind box)
- Problem: best cooperative pushing is at ~120° (flanking), not directly behind
- Proposed: `cos(angle - 120°)` — peaks at 120°, negative in front (0-60°), +0.5 at 180°
- Pending implementation

## Joint reward analysis at optimal position
At optimal pushing position (each agent ~0.4m from box, 0.8m apart):
- Approach penalty: -0.000608
- Collision penalty: -0.000617
- **Total: -0.00123** (~12% of max tracking reward 0.01)
- This is the minimum of the joint approach+collision function — insignificant tax

## Cooperation ideas discussed (not implemented)
1. **Symmetric positioning** — reward agents for being ~180° apart around box
2. **Lateral force cancellation** — penalize net perpendicular force on box
3. **Velocity alignment efficiency** — ratio of useful velocity to total velocity
4. **Contact symmetry** — reward both agents being in contact with box
5. **Mutual information** — penalize when actions are independent of partner state

Recommendation: symmetric positioning + angular velocity penalty (cause + effect)

## Files Modified
- `openrl_ws/train.py` — checkpoint save_freq fix (20000)
- `openrl_ws/test.py` — velocity calculator, viewer, success rate
- `openrl_ws/utils.py` — success rate in tensorboard batch_rewards
- `HARL/harl_mapush/test.py` — velocity calculator, viewer, all_checkpoints summary, success rate
- `mqe/envs/wrappers/go1_push_vel_wrapper.py` — sharper collision punishment

## Files Created
- `test_all_checkpoints_velocity.sh` — standalone MAPPO checkpoint testing script

## Test Commands Reference

### MAPPO
```bash
# Viewer (single env, visual)
python ./openrl_ws/test.py --num_envs 1 --algo ppo --task go1push_vel \
    --checkpoint log/MQE/go1push_vel/velocity/run2/checkpoints/rl_model_10000000_steps/module.pt \
    --agent0 go1 --agent1 go1 --test_mode viewer

# Calculator (stats)
python ./openrl_ws/test.py --num_envs 300 --algo ppo --task go1push_vel \
    --checkpoint log/MQE/go1push_vel/velocity/run2/checkpoints/rl_model_10000000_steps/module.pt \
    --agent0 go1 --agent1 go1 --test_mode calculator --headless

# All checkpoints
./test_all_checkpoints_velocity.sh log/MQE/go1push_vel/velocity/run2
```

### HAPPO
```bash
# Calculator (single checkpoint)
python HARL/harl_mapush/test.py --task go1push_vel \
    --checkpoint <path_to_checkpoint_dir> \
    --mode calculator --num_envs 300 --agent0 go1 --agent1 go1

# All checkpoints (saves test_results.txt)
python HARL/harl_mapush/test.py --task go1push_vel \
    --checkpoint <path_to_models_dir> \
    --mode calculator --num_envs 300 --agent0 go1 --agent1 go1 --all_checkpoints

# Viewer
python HARL/harl_mapush/test.py --task go1push_vel \
    --checkpoint <path_to_checkpoint_dir> \
    --mode viewer --num_envs 1 --num_episodes 3 --agent0 go1 --agent1 go1
```

## Training Results
- MAPPO with 2 Go1s learned the velocity task well in **100-150M steps**
- HAPPO training not yet started

## Key Technical Insights

### OpenRL CheckpointCallback
`save_freq` counts `n_calls` (one per `env.step()`), NOT timesteps. The checkpoint filename uses `num_time_steps` (= n_calls × num_envs). Formula: `save_freq = desired_timestep_interval / num_envs`.

### MAPPO vs HAPPO critic for velocity task
- **MAPPO**: critic sees same 16-dim egocentric actor observation (parameter sharing)
- **HAPPO**: critic gets dedicated 18-dim box-centered global state with velocity error vectors — richer privileged view

### All velocity rewards are team rewards
Every reward term is computed per-env, broadcast to all agents, summed, then re-broadcast identically. No individual reward components exist.
