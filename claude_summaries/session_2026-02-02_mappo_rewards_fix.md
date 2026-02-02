# Session Summary - 2026-02-02

## Overview
This session focused on fixing MAPPO reward contamination from HAPPO-specific changes, and investigating exception_punishment being 0 in heterogeneous runs.

## Key Work Completed

### 1. Fixed MAPPO Baseline Rewards (`--baseline_mappo_rewards` flag)

**Problem**: MAPPO runs were using HAPPO-specific reward modifications because both frameworks share the same wrapper file (`mqe/envs/wrappers/go1_push_mid_wrapper.py`).

**Root Cause**:
```
MAPPO (openrl_ws):  make_env() → make_mqe_env() → Go1PushMidWrapper
HAPPO (HARL):       MAPushEnv() → make_mqe_env() → Go1PushMidWrapper
```
Both frameworks use the **same wrapper file**. Changes made for HAPPO affected MAPPO's default behavior.

**Solution**: Implemented proper original MAPush reward behavior when `--baseline_mappo_rewards True` (default for MAPPO).

### 2. Original MaPush Rewards - Now Correctly Restored

Reference: `/home/gvlab/backup_MAPush/mqe/envs/wrappers/go1_push_mid_wrapper.py`

| Reward | Type | MAPPO Implementation | HAPPO Implementation |
|--------|------|---------------------|---------------------|
| `reach_target_reward` | Shared | `reward[finished, :] +=` | Same |
| `exception_punishment` | Shared | `reward[exception, :] +=` | Same |
| `distance_to_target_reward` | Shared | `reward[:, :] +=` | Same (+ optional gating) |
| `approach_to_box_reward` | **Per-agent** | `reward[:, i] +=` | Team reward (sum/avg) |
| `collision_punishment` | **Per-agent** | `reward[:, i] +=` and `reward[:, j] +=` | Team reward |
| `push_reward` | Shared | `reward[:, :] +=` | Same (+ optional gating) |
| `ocb_reward` | **Per-agent** | `reward[:, i] +=` | Team reward (binary or continuous) |

**Key Insight**: MAPPO uses per-agent critics, so rewards like `approach_to_box`, `collision_punishment`, and `ocb_reward` must be computed **per-agent**. HAPPO uses a global critic, so those were "teamified" to be shared.

### 3. Code Changes in `go1_push_mid_wrapper.py`

#### approach_to_box_reward (lines ~591-640)
```python
if self.baseline_mappo_rewards:
    # ORIGINAL MAPPO: Per-agent reward - each agent gets its own distance penalty
    reward_logger = []
    for i in range(self.num_agents):
        distance = torch.norm(box_pos - base_pos[:, i, :], dim=1, keepdim=True)
        distance_reward = (-(distance + 0.5)**2) * self.approach_reward_scale
        reward_logger.append(torch.sum(distance_reward).cpu())
        reward[:, i] += distance_reward.squeeze(-1)
    self.reward_buffer["approach_to_box_reward"] += np.sum(np.array(reward_logger))
else:
    # HAPPO: Team reward (sum or average)
    ...
```

#### collision_punishment (lines ~642-670)
```python
if self.baseline_mappo_rewards:
    # ORIGINAL MAPPO: Per-agent collision punishment (both agents in pair get it)
    punishment_logger = []
    for i in range(self.num_agents):
        for j in range(i+1, self.num_agents):
            distance = torch.norm(base_pos[:, i, :] - base_pos[:, j, :], dim=1, keepdim=True)
            collision_punishment = (1 / (0.02 + distance/3)) * self.collision_punishment_scale
            punishment_logger.append(torch.sum(collision_punishment).cpu())
            reward[:, i] += collision_punishment.squeeze(-1)
            reward[:, j] += collision_punishment.squeeze(-1)
    self.reward_buffer["collision_punishment"] += np.sum(np.array(punishment_logger))
else:
    # HAPPO: Team reward
    ...
```

#### ocb_reward (lines ~792-835)
```python
if self.baseline_mappo_rewards:
    # ORIGINAL MAPPO: Per-agent OCB reward
    reward_logger = []
    for i in range(self.num_agents):
        ocb_reward = raw_ocb_list[i] * self.ocb_reward_scale
        reward[:, i] += ocb_reward
        reward_logger.append(torch.sum(ocb_reward).cpu())
    self.reward_buffer["ocb_reward"] += np.sum(np.array(reward_logger))
elif self.mapush_og_rewards_teamified:
    # HAPPO teamified version (continuous, averaged)
    ...
else:
    # HAPPO binary joint OCB
    ...
```

#### distance_to_target_reward (lines ~568-584)
```python
if (self.mapush_og_rewards_teamified or self.baseline_mappo_rewards) and self.target_reward_scale != 0:
    # Shared reward (both agents get same) - correct for both MAPPO and HAPPO
    distance_reward = self.target_reward_scale * 100 * (2 * (past_distance - distance) - 0.01 * distance)
    # Gating NOT applied for baseline_mappo_rewards
    if self.shared_gated_rewards and not self.baseline_mappo_rewards:
        distance_reward = distance_reward * gating_factor
    reward[:, :] += distance_reward.unsqueeze(1).repeat(1, self.num_agents)
```

### 4. Investigation: exception_punishment = 0 in Heterogeneous Runs

**Observation**: User reported `exception_punishment: 0.0` in new heterogeneous runs, which was never 0 before.

**Investigation Steps**:
1. Verified hetero config DOES have termination class (tested via Python script)
2. Confirmed termination_terms = ['roll', 'pitch', 'z_wave', 'collision']
3. Checked inheritance chain: HeteroTask → HeteroRobot → LeggedRobotField → LeggedRobot

**Key Finding from git history** (commit 198194b):
```python
# HETERO FIX: Override z-position with robot-specific init height
# Each robot type has its own proper ground clearance
from mqe.envs.robot_registry import get_robot_config
robot_config = get_robot_config(self.hetero_agent_types[idx])
robot_init_height = robot_config.init_state.pos[2]
base_init_state[2] = robot_init_height
```

**Termination Check Flow**:
1. `LeggedRobot.post_physics_step()` calls `self.check_termination()`
2. `LeggedRobotField.check_termination()` does roll/pitch/z_wave/collision checks
3. Sets `self.exception_buf` which wrapper reads for punishment

**z_wave threshold for hetero**:
- Go1PushMidCfg: threshold = 0.35m
- AnymalCCfg: threshold = 2.0m (relaxed for taller robot)
- Currently uses single threshold from base config for all agents
- But `base_init_state` is now correctly set per-robot, so z_wave compares each robot against its **own** init position

**Added Debug Logging** (lines ~540-573 in wrapper):
```python
# DEBUG: Track exception counts
if not hasattr(self, 'debug_exception_step_counter'):
    self.debug_exception_step_counter = 0
    self.debug_sim_exception_count = 0
    self.debug_nan_exception_count = 0

num_sim_exceptions = self.exception_buf.sum().item()
num_nan_exceptions = self.value_exception_buf.sum().item()
# ... logs every 500 steps showing counts and config status
```

**Status**: Debug code added, awaiting test run to see output.

### 5. Potential Issue: Per-Agent Termination Thresholds

For true heterogeneous support, termination thresholds should be per-agent:
- Go1 z_wave threshold: 0.35m (shorter robot)
- Anymal z_wave threshold: 2.0m (taller robot)

Currently all agents use the same threshold from Go1PushMidCfg. This may need future work.

## Files Modified This Session

1. **`mqe/envs/wrappers/go1_push_mid_wrapper.py`**
   - Fixed per-agent rewards for baseline MAPPO (approach_to_box, collision_punishment, ocb_reward)
   - Fixed distance_to_target_reward condition
   - Added debug logging for exception tracking

## Quick Reference Commands

```bash
# MAPPO training with baseline rewards (default)
cd /home/gvlab/new-universal-MAPush
PYTHONPATH=/home/gvlab/new-universal-MAPush:$PYTHONPATH python ./openrl_ws/train.py \
    --task go1push_mid \
    --agent0 go1 --agent1 go1 \
    --num_envs 100 \
    --train_timesteps 100000 \
    --exp_name test_baseline \
    --use_tensorboard True

# HAPPO heterogeneous training
cd /home/gvlab/new-universal-MAPush
python HARL/harl_mapush/train.py \
    --agent0 go1 --agent1 anymal_c \
    --num_envs 100
```

## Pending/Next Steps

1. **Run test to see debug output** for exception_punishment investigation
2. **Consider per-agent termination thresholds** for true heterogeneous support
3. **Verify MAPPO training** works correctly with baseline rewards after these fixes
4. **Remove debug logging** after investigation complete

## Related Files for Context

- Original untouched wrapper: `/home/gvlab/backup_MAPush/mqe/envs/wrappers/go1_push_mid_wrapper.py`
- Hetero config creation: `mqe/utils/hetero_config.py`
- Termination checks: `mqe/envs/field/legged_robot_field.py` (lines 138-230)
- HeteroRobot init heights: `mqe/envs/base/hetero_robot.py` (lines 204-225)
