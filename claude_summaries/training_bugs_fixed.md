# Training Bugs Fixed - HARL MAPush Integration

**Date:** 2025-12-13
**Session:** Phase 1-3 Debugging and Fixes

---

## Overview

This document details all bugs encountered and fixed during the HARL-MAPush training integration. Training now runs successfully with proper statistics logging and checkpoint saving.

---

## Bug 1: Segmentation Fault During Environment Initialization

### Symptom
```
+++ Using GPU PhysX
Physics Engine: PhysX
Physics Device: cuda:0
GPU Pipeline: enabled
Segmentation fault (core dumped)
```

Training crashed immediately after GPU PhysX initialization, before "Setting seed" message.

### Root Cause
**File:** `mqe/envs/base/base_task.py:44`

```python
self.sim_params.physx.max_gpu_contact_pairs *= 5
```

The code multiplied GPU contact pairs by 5x, causing excessive GPU memory allocation that led to CUDA OOM errors and segfaults on 8GB VRAM GPUs.

### Fix
```python
# Reduced from *5 to *1 to avoid CUDA OOM on 8GB VRAM GPUs
self.sim_params.physx.max_gpu_contact_pairs *= 1
```

**Impact:** Environment initialization now succeeds without memory errors.

---

## Bug 2: Memory Leak - Unbounded Statistics Growth

### Symptom
Training would start successfully but vRAM usage would grow continuously:
- 6000 MB → 7500 MB → 7700 MB → crash (segfault)
- Reducing parallel environments or model size didn't help

### Root Cause
**File:** `HARL/harl/envs/mapush/mapush_env.py:116-134`

```python
# Track statistics for episodes that just finished
for env_idx in range(self.n_envs):
    if dones_np[env_idx, 0]:  # Episode done
        self.episode_success.append(success)      # UNBOUNDED GROWTH
        self.episode_lengths.append(length)       # UNBOUNDED GROWTH
        self.episode_collision.append(collision)  # UNBOUNDED GROWTH
        self.episode_collaboration.append(collab) # UNBOUNDED GROWTH
```

Statistics were accumulated in Python lists that grew without bounds. With 500 parallel environments and `log_interval=5` rollouts, thousands of episodes accumulated before reset, causing memory fragmentation and vRAM growth.

**Calculation:**
- Log interval: 5 rollouts
- Steps per rollout: 200
- Total steps between resets: 5 × 200 = 1,000 steps
- Episodes per env per 1000 steps: ~1
- Total episodes accumulated: 500 envs × 1 episode = 500+ entries
- Over time: unbounded growth

### Fix
**File:** `HARL/harl/envs/mapush/mapush_env.py:185-188`

Changed from unbounded lists to bounded deques:

```python
from collections import deque

def reset_statistics(self):
    """Reset statistics buffers.

    Using deque with maxlen=1000 to prevent unbounded memory growth.
    This limits statistics to the most recent 1000 episodes.
    """
    self.episode_success = deque(maxlen=1000)
    self.episode_collision = deque(maxlen=1000)
    self.episode_lengths = deque(maxlen=1000)
    self.episode_collaboration = deque(maxlen=1000)
```

**Impact:** Memory usage stays bounded at ~32KB maximum. Training can run indefinitely without memory leaks.

---

## Bug 3: Environment Setup - Wrong Namespace Type

### Symptom
Environment creation would fail or behave incorrectly.

### Root Cause
**File:** `HARL/harl/envs/mapush/mapush_env.py:30-51`

Used `types.SimpleNamespace` instead of `argparse.Namespace`, and didn't call `custom_cfg()`:

```python
from types import SimpleNamespace
args = SimpleNamespace(
    task=env_args.get("task", "go1push_mid"),
    headless=True,
    num_envs=env_args.get("n_threads", 500),
    # ...
)

self.env, self.env_cfg = make_mqe_env(args.task, args)  # Missing custom_cfg!
```

### Fix
```python
import argparse
from mqe.envs.utils import custom_cfg

args = argparse.Namespace()
args.task = env_args.get("task", "go1push_mid")
args.num_envs = env_args.get("num_envs", env_args.get("n_threads", 500))
args.seed = env_args.get("seed", 1)
args.headless = env_args.get("headless", True)
args.record_video = False
# ... device configuration ...

# Create MQE environment with custom config
self.env, self.env_cfg = make_mqe_env(
    args.task,
    args,
    custom_cfg=custom_cfg(args)  # ← Added
)
```

**Impact:** Environment properly configures num_envs and other settings.

---

## Bug 4: Action Tensor Shape Mismatch

### Symptom
```
RuntimeError: The size of tensor a (500) must match the size of tensor b (2) at non-singleton dimension 1
```

### Root Cause
**File:** `HARL/harl/envs/mapush/mapush_env.py:97-98`

Incorrectly transposed actions:

```python
# WRONG: Transpose [n_envs, n_agents, action_dim] -> [n_agents, n_envs, action_dim]
actions_transposed = actions.transpose(1, 0, 2)
actions_torch = torch.from_numpy(actions_transposed).cuda()
```

**Debug output showed:**
```
DEBUG: actions.shape = (500, 2, 3)          # Input from runner
DEBUG: actions_torch.shape = torch.Size([2, 500, 3])  # WRONG!
```

The wrapper expected `[n_envs, n_agents, action_dim]` but we were transposing it incorrectly.

### Fix
```python
# Convert to torch: actions already in [n_envs, n_agents, action_dim] format
actions_torch = torch.from_numpy(actions).cuda()
```

**Impact:** Actions now have correct shape for environment step.

---

## Bug 5: Info Dict Format - KeyError

### Symptom
```
KeyError: 0
File "on_policy_base_runner.py", line 415, in <listcomp>
    if "bad_transition" in info[0].keys()
```

### Root Cause
**File:** `HARL/harl/envs/mapush/mapush_env.py:119`

For Environment Provided (EP) state mode, the runner expects info dicts with integer keys for agent IDs:

```python
infos_list = [{} for _ in range(self.n_envs)]  # WRONG
```

The runner tried to access `info[0]` to check for shared environment info.

### Fix
```python
# Infos - HARL expects list of dicts with agent ID keys for EP mode
# For EP (Environment Provided) state, info[0] contains shared info
infos_list = [{0: {}} for _ in range(self.n_envs)]
```

**Impact:** Info dicts properly formatted for EP state mode.

---

## Bug 6: Available Actions - TypeError on None

### Symptom
```
TypeError: 'NoneType' object is not subscriptable
File "on_policy_base_runner.py", line 444
    if available_actions[0] is not None
```

### Root Cause - Part 1: Environment
**File:** `HARL/harl/envs/mapush/mapush_env.py`

Initially returned `np.array([[None] * n_agents] * n_envs)`, but:
- Runner checks `available_actions[0] is not None`
- `array[0] = [None, None]` which is NOT None (it's a list)
- This caused the runner to pass the array to buffer
- Buffer tried to assign to `self.available_actions[step]` but `self.available_actions` was `None` for continuous action spaces

### Root Cause - Part 2: Runner
**File:** `HARL/harl/runners/on_policy_base_runner.py:444`

```python
available_actions[:, agent_id]
if available_actions[0] is not None  # Can't index None!
else None,
```

The check assumed `available_actions` was an array, not `None`.

### Fix - Part 1: Environment
```python
# Available actions - None for continuous action space
return obs_np, state_np, rewards_np, dones_np, infos_list, None
```

### Fix - Part 2: Runner
```python
available_actions[:, agent_id]
if available_actions is not None and available_actions[0] is not None
else None,
```

**Impact:** Continuous action spaces properly handled with `None` available actions.

---

## Bug 7: Logger Missing Environment Reference

### Symptom
```
AttributeError: 'MAPushLogger' object has no attribute 'envs'
File "mapush_logger.py", line 26
    if hasattr(self.envs, 'get_statistics'):
```

### Root Cause
The `MAPushLogger` needed to call `self.envs.get_statistics()` and `self.envs.reset_statistics()` but didn't have a reference to the environment.

### Fix - Part 1: Logger
**File:** `HARL/harl/envs/mapush/mapush_logger.py:9-16`

```python
def __init__(self, args, algo_args, env_args, num_agents, writter, run_dir):
    """Initialize MAPush logger."""
    super().__init__(args, algo_args, env_args, num_agents, writter, run_dir)
    self.envs = None  # Will be set by runner

def set_envs(self, envs):
    """Set the environment reference for statistics tracking."""
    self.envs = envs
```

### Fix - Part 2: Runner
**File:** `HARL/harl_mapush/runners/mapush_happo_runner.py:30-32`

```python
# Set envs reference in logger for MAPush statistics tracking
if hasattr(self.logger, 'set_envs'):
    self.logger.set_envs(self.envs)
```

**Impact:** Logger can now access environment statistics for custom MAPush metrics.

---

## Bug 8: Missing mapush Support in configs_tools

### Symptom
```
UnboundLocalError: local variable 'task' referenced before assignment
File "harl/utils/configs_tools.py", line 69
```

### Root Cause
**File:** `HARL/harl/utils/configs_tools.py`

The `get_task_name()` function had cases for different environments but not mapush, causing `task` variable to be undefined.

### Fix
```python
elif env == "mapush":
    task = env_args.get("task", "go1push_mid")
```

**Impact:** HARL properly recognizes mapush environment.

---

## Complete File Changes Summary

### Files Modified

1. **`mqe/envs/base/base_task.py`** (line 44-45)
   - GPU memory fix: `max_gpu_contact_pairs *= 1`

2. **`HARL/harl/envs/mapush/mapush_env.py`**
   - Import deque (line 5)
   - Environment setup with argparse.Namespace and custom_cfg (lines 30-66)
   - Removed action transpose (line 98)
   - Fixed info dict format for EP mode (line 119)
   - Return None for available_actions (line 148, 161)
   - Bounded deques in reset_statistics() (lines 185-188)

3. **`HARL/harl/envs/mapush/mapush_logger.py`**
   - Added __init__ and set_envs() method (lines 9-16)

4. **`HARL/harl_mapush/runners/mapush_happo_runner.py`**
   - Call logger.set_envs() after initialization (lines 30-32)

5. **`HARL/harl/runners/on_policy_base_runner.py`** (line 444)
   - Check available_actions is not None before indexing

6. **`HARL/harl/utils/configs_tools.py`** (lines 69-70)
   - Added mapush case

---

## Training Status: ✅ WORKING

Training now runs successfully with:
- 500 parallel environments
- No memory leaks
- Proper statistics logging (success rate, collision rate, etc.)
- Checkpoint saving every 10M steps
- TensorBoard metrics tracking

**Example output:**
```
Env mapush Task go1push_mid Algo happo Exp cuboid_happo updates 5/1000 episodes,
total num timesteps 500000/100000000, FPS 5644.
Average step reward is -0.016547411680221558.
Some episodes done, average episode reward is -6.5947160790208725.
[MAPush Stats] Success: 0.120 | Collision: 0.045 | Episodes: 50
```

---

## Lessons Learned

1. **Memory Management:** Always use bounded buffers for statistics accumulation in multi-environment settings
2. **GPU Memory:** Be conservative with GPU memory allocations (contact pairs, etc.)
3. **Data Shapes:** Carefully verify tensor/array shapes match expected formats
4. **Environment Interfaces:** Follow framework conventions exactly (info dicts, available actions, etc.)
5. **Testing:** Test with actual environment before assuming compatibility
6. **Debugging:** Add debug prints for shapes/types when encountering runtime errors

---

## Next Steps

**Phase 4: Testing Framework** (Remaining)
- Implement `HARL/harl_mapush/test.py`
- Viewer mode (sequential episode visualization)
- Calculator mode (multi-env statistics)
- Configurable seed, episodes, envs
- Statistics: success rate, collision rate, finished time, collaboration degree
