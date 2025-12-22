# CRITIC8: Concatenated Local Observations + Full Reversion

> **Date:** December 22, 2025
> **Goal:** Match OpenRL's approach - use concatenated local observations for critic
> **Status:** TESTING

---

## Summary of All Changes

This version reverts ALL previous critic fixes and implements a new approach that more closely matches OpenRL's working implementation.

---

## Part 1: Reverted Changes

### 1.1 Value Normalizer Fix (critic3) - REVERTED

**File:** `HARL/harl/algorithms/critics/v_critic.py`

**What was changed:**
- critic3 moved `value_normalizer.update()` to run ONCE before the training loop
- This was meant to prevent non-stationary targets

**Reverted to:**
- `value_normalizer.update()` now runs inside `cal_value_loss()` as originally designed

```python
# REVERTED: value_normalizer.update() back inside cal_value_loss()
if value_normalizer is not None:
    value_normalizer.update(return_batch)  # <- Back to original location
    error_clipped = (
        value_normalizer.normalize(return_batch) - value_pred_clipped
    )
```

### 1.2 Actor Update Interval (critic3) - REVERTED

**File:** `HARL/harl/runners/on_policy_ha_runner.py`

**What was changed:**
- critic3 added `actor_update_interval` to slow down actor updates
- Actors only updated every N iterations

**Reverted to:**
- Actors update EVERY iteration (no interval logic)
- `train()` method no longer takes `episode` parameter

```python
# REVERTED: train() no longer takes episode parameter
def train(self):  # Was: def train(self, episode=None)
    # ... actor updates happen every iteration now
```

**File:** `HARL/harl/runners/on_policy_base_runner.py`

```python
# REVERTED: train() called without episode
actor_train_infos, critic_train_info = self.train()  # Was: self.train(episode)
```

### 1.3 Config Parameters - REVERTED

**File:** `HARL/harl/configs/algos_cfgs/happo.yaml`

| Parameter | critic5/6/7 Value | REVERTED Value |
|-----------|-------------------|----------------|
| `critic_epoch` | 30 | **5** |
| `clip_param` | 0.1 | **0.2** |
| `value_loss_coef` | 3.0 | **1.0** |
| `max_grad_norm` | 5.0 | **10.0** |
| `gae_lambda` | 0.9 | **0.95** |
| `actor_update_interval` | 5 | **REMOVED** |

### 1.4 What Was KEPT

| Feature | Status | Reason |
|---------|--------|--------|
| Separate actor/critic hidden sizes | ✅ Kept | Useful flexibility |
| Action scaling reverted (no 0.5x) | ✅ Kept | 0.5x scaling broke learning |

---

## Part 2: New Critic Input (critic8)

### The Problem with Previous Approaches

| Approach | Issue |
|----------|-------|
| **critic7 (11 dims, global frame)** | Same situation at different positions = different state vectors |
| **OpenRL (8 dims per agent)** | Uses local observations, translation invariant |

### The Solution: Concatenated Local Observations

Instead of constructing a global state in absolute coordinates, we concatenate all agents' local observations.

**File:** `HARL/harl/envs/mapush/mapush_env.py`

```python
# CRITIC8: Concatenate all agents' local observations
obs_dim = self.env.observation_space.shape[0]  # 8 dims per agent
global_state_dim = obs_dim * self.n_agents     # 8 * 2 = 16 dims
```

### Before vs After

#### BEFORE (critic7): Global Absolute Frame (11 dims)
```
Global State:
├── Box:     [x, y, yaw]           = 3 dims (ABSOLUTE position)
├── Target:  [x, y]                = 2 dims (ABSOLUTE position)
├── Agent 0: [x, y, yaw]           = 3 dims (ABSOLUTE position)
└── Agent 1: [x, y, yaw]           = 3 dims (ABSOLUTE position)
                            TOTAL = 11 dims

Problem: Agent at (5,3) vs (1,1) with same relative setup = DIFFERENT states!
```

#### AFTER (critic8): Concatenated Local Observations (16 dims)
```
Global State = [Agent0_local_obs, Agent1_local_obs]

Agent 0's local obs (8 dims):
├── Target:      [rel_x, rel_y]         = 2 dims (relative to agent0)
├── Box:         [rel_x, rel_y]         = 2 dims (relative to agent0)
├── Box yaw:     [rel_yaw]              = 1 dim  (relative to agent0)
└── Other agent: [rel_x, rel_y, rel_yaw] = 3 dims (agent1 relative to agent0)

Agent 1's local obs (8 dims):
├── Target:      [rel_x, rel_y]         = 2 dims (relative to agent1)
├── Box:         [rel_x, rel_y]         = 2 dims (relative to agent1)
├── Box yaw:     [rel_yaw]              = 1 dim  (relative to agent1)
└── Other agent: [rel_x, rel_y, rel_yaw] = 3 dims (agent0 relative to agent1)

                                  TOTAL = 16 dims

Advantage: Same relative setup ANYWHERE on map = SAME state!
```

### Index Mapping (16 dims)

| Index | Content | Frame |
|-------|---------|-------|
| 0 | target_rel_x (agent0) | Agent0 local |
| 1 | target_rel_y (agent0) | Agent0 local |
| 2 | box_rel_x (agent0) | Agent0 local |
| 3 | box_rel_y (agent0) | Agent0 local |
| 4 | box_rel_yaw (agent0) | Agent0 local |
| 5 | agent1_rel_x (agent0) | Agent0 local |
| 6 | agent1_rel_y (agent0) | Agent0 local |
| 7 | agent1_rel_yaw (agent0) | Agent0 local |
| 8 | target_rel_x (agent1) | Agent1 local |
| 9 | target_rel_y (agent1) | Agent1 local |
| 10 | box_rel_x (agent1) | Agent1 local |
| 11 | box_rel_y (agent1) | Agent1 local |
| 12 | box_rel_yaw (agent1) | Agent1 local |
| 13 | agent0_rel_x (agent1) | Agent1 local |
| 14 | agent0_rel_y (agent1) | Agent1 local |
| 15 | agent0_rel_yaw (agent1) | Agent1 local |

---

## Why This Should Work Better

### 1. Translation Invariance
```
Scenario: Both agents 1m from box, box 2m from target

At map position (5, 5):
  State = [1, 0, 2, 0, 0, -1, 0, 0,   # Agent0 sees box ahead, target further
           1, 0, 2, 0, 0, 1, 0, 0]    # Agent1 sees same

At map position (100, 100):
  State = [1, 0, 2, 0, 0, -1, 0, 0,   # SAME STATE!
           1, 0, 2, 0, 0, 1, 0, 0]

Critic learns: "this configuration = this value" (generalizes!)
```

### 2. Rotation Invariance
Local observations are rotated to each agent's heading, so the agent always "faces forward."

### 3. Matches OpenRL
OpenRL's working MAPPO uses per-agent local observations. By concatenating them, we give the centralized critic the same information.

### 4. Simpler Implementation
No complex global state construction needed - just flatten the actor observations.

---

## Implementation Details

### In `__init__`:
```python
obs_dim = self.env.observation_space.shape[0]  # 8
global_state_dim = obs_dim * self.n_agents     # 16
```

### In `step()` and `reset()`:
```python
# obs_np shape: [n_envs, n_agents, obs_dim] e.g. [500, 2, 8]
# Flatten to: [n_envs, n_agents * obs_dim] e.g. [500, 16]
global_state_np = obs_np.reshape(self.n_envs, -1)
```

---

## Current Configuration

```yaml
# happo.yaml - Standard values
ppo_epoch: 5
critic_epoch: 5
clip_param: 0.2
value_loss_coef: 1.0
max_grad_norm: 10.0
gae_lambda: 0.95
entropy_coef: 0.01

# Separate architectures (kept)
actor_hidden_sizes: [128, 128]
critic_hidden_sizes: [256, 256, 128]
```

---

## Expected Results

- **Better generalization**: Same relative scenario = same value prediction
- **Faster learning**: Critic doesn't need to learn "position doesn't matter"
- **More stable**: Less variance in value estimates across the map

---

## Files Modified

1. `HARL/harl/envs/mapush/mapush_env.py`
   - Changed `share_observation_space` to 16 dims
   - Modified `step()` and `reset()` to use concatenated local obs

2. `HARL/harl/algorithms/critics/v_critic.py`
   - Reverted value normalizer to original location

3. `HARL/harl/runners/on_policy_ha_runner.py`
   - Removed actor_update_interval logic

4. `HARL/harl/runners/on_policy_base_runner.py`
   - Reverted train() call

5. `HARL/harl/configs/algos_cfgs/happo.yaml`
   - Reset all parameters to standard values

---

## History

| Version | Date | Key Changes | Result |
|---------|------|-------------|--------|
| critic1 | Dec 18 | critic_epoch: 25 | Failed |
| critic2 | Dec 18 | value_loss_coef: 5.0 | Failed |
| critic3 | Dec 18 | Value normalizer fix + actor_update_interval | 20% success |
| critic4 | Dec 19 | Higher LR | Failed |
| critic5 | Dec 19 | Full stability config | Failed |
| critic6 | Dec 19 | Action scaling 0.5x | Failed (0% success) |
| critic7 | Dec 21 | Remove velocities (11 dims) | Failed |
| **critic8** | Dec 22 | **Concatenated local obs (16 dims) + full revert** | Testing |
