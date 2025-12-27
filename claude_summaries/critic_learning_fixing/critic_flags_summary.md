# Critic Input Mode Flags - Implementation Summary

> **Date:** December 26, 2025
> **Feature:** Flag-controlled critic input modes for easy experimentation

---

## Overview

All critic fundamental changes are now controlled by command-line flags. This allows easy switching between different critic input representations without modifying code.

---

## Available Critic Modes

### Flag Priority

When multiple flags are set, priority is:
1. `--use_concat_agent_observations_critic` (highest priority)
2. `--use_box_centered_critic`
3. Neither flag (default: absolute coordinates)

---

## Mode 1: CRITIC8 - Concatenated Agent Observations

**Flag:** `--use_concat_agent_observations_critic True`
**Default:** `False`
**Dimensions:** 16 dims (for 2 agents)

### Description
Simply concatenates all agents' local observations without any modification. Each agent's observation is already in its own frame of reference (rotated to local frame in the wrapper).

### Structure
```
Global State = [Agent0_obs, Agent1_obs]

Each agent observation (8 dims):
  - Target (x, y) relative to agent: 2 dims
  - Box (x, y) relative to agent: 2 dims
  - Box yaw relative to agent: 1 dim
  - Other agent (x, y, yaw) relative to this agent: 3 dims

Total: 8 * 2 = 16 dims
```

### Properties
- ✅ **Translation invariant** (observations are relative to each agent)
- ✅ **Rotation invariant** (each observation rotated to agent's heading)
- ✅ **Unmodified observations** (exactly what actors see)
- ✅ **Two perspectives** (each agent's view of the world)

### Usage
```bash
./run_training.sh \
    --algo happo \
    --env mapush \
    --exp_name critic8_test \
    --use_concat_agent_observations_critic True
```

---

## Mode 2: CRITIC9 - Box-Centered Coordinates

**Flag:** `--use_box_centered_critic True`
**Default:** `False`
**Dimensions:** 9 dims (for 2 agents)

### Description
Expresses everything relative to the box (the object being pushed). Provides translation invariance with a truly global view.

### Structure
```
Global State:
  - Target relative to box: (x, y) = 2 dims
  - Agent 0 relative to box: (x, y, yaw) = 3 dims
  - Agent 1 relative to box: (x, y, yaw) = 3 dims
  - Box absolute orientation: yaw = 1 dim

Total: 2 + 3*n_agents + 1 = 9 dims
```

### Properties
- ✅ **Translation invariant** (everything relative to box)
- ✅ **Task-centric** (value depends on box-target distance)
- ✅ **True global view** (single unified perspective)
- ✅ **Compact** (9 dims vs 16 for CRITIC8)

### Usage
```bash
./run_training.sh \
    --algo happo \
    --env mapush \
    --exp_name critic9_test \
    --use_box_centered_critic True
```

---

## Mode 3: CRITIC7 - Absolute Coordinates (DEFAULT)

**Flags:** Both `False` (or not specified)
**Default:** `True` (active when no flags used)
**Dimensions:** 11 dims (for 2 agents)

### Description
All positions in absolute world coordinates. No translation invariance.

### Structure
```
Global State:
  - Box: (x, y, yaw) = 3 dims
  - Target: (x, y) = 2 dims
  - Agent 0: (x, y, yaw) = 3 dims
  - Agent 1: (x, y, yaw) = 3 dims

Total: 3 + 2 + 3*n_agents = 11 dims
```

### Properties
- ❌ **NOT translation invariant** (absolute positions)
- ✅ **Simpler to understand** (raw world coordinates)
- ⚠️ **Requires learning** (critic must learn position doesn't matter)

### Usage
```bash
# Default - no flags needed
./run_training.sh \
    --algo happo \
    --env mapush \
    --exp_name critic7_test

# Or explicitly
./run_training.sh \
    --algo happo \
    --env mapush \
    --exp_name critic7_test \
    --use_box_centered_critic False \
    --use_concat_agent_observations_critic False
```

---

## Comparison Table

| Mode | Flag | Dims | Translation Inv. | Rotation Inv. | Global View | Observations |
|------|------|------|------------------|---------------|-------------|--------------|
| **CRITIC8** | `--use_concat_agent_observations_critic True` | 16 | ✅ | ✅ | Two perspectives | Unmodified actor obs |
| **CRITIC9** | `--use_box_centered_critic True` | 9 | ✅ | ⚠️ Partial | ✅ True global | Constructed from box frame |
| **CRITIC7** | Neither (default) | 11 | ❌ | ❌ | ✅ True global | Absolute world coordinates |

---

## Implementation Files Modified

1. **`HARL/harl_mapush/train.py`**
   - Added `--use_concat_agent_observations_critic` argument
   - Updated `env_args` to pass both flags
   - Both flags default to `False`

2. **`HARL/harl/envs/mapush/mapush_env.py`**
   - Added `use_concat_agent_observations_critic` flag reading
   - Updated `__init__()` to handle three modes
   - Modified `step()` to use `obs_np.reshape()` for CRITIC8
   - Modified `reset()` to use `obs_np.reshape()` for CRITIC8
   - Updated diagnostic logging for all three modes

---

## Diagnostic Output

On first step, you'll see different diagnostic output based on mode:

### CRITIC8
```
================================================================================
GLOBAL STATE DIAGNOSTIC (First Step) - CRITIC8: Concatenated Agent Observations
================================================================================
Global state shape: (500, 16)
Expected: [500, 16] for 2 agents (concatenated local obs)

Environment 0 global state (16 dims):
  Agent0 obs (8 dims): [1.234 -0.456 2.345 0.123 -0.234 -1.234 0.567 0.890]
  Agent1 obs (8 dims): [1.345 0.567 2.234 -0.234 0.123 1.234 -0.567 -0.890]

Agent observation structure (each 8 dims):
  - Target (x, y) relative to agent: 2 dims
  - Box (x, y) relative to agent: 2 dims
  - Box yaw relative to agent: 1 dim
  - Other agent (x, y, yaw) relative to this agent: 3 dims
```

### CRITIC9
```
================================================================================
GLOBAL STATE DIAGNOSTIC (First Step) - CRITIC9: Box-centered
================================================================================
Global state shape: (500, 9)
Expected: [500, 9] for 2 agents (box-centered coordinates)

Environment 0 global state (9 dims):
  Target rel to box:  x=2.155, y=-0.444
  Agent0 rel to box:  x=-2.833, y=-0.428, yaw=5.327
  Agent1 rel to box:  x=2.845, y=-0.532, yaw=-0.324
  Box yaw (abs):      -0.451
```

### CRITIC7
```
================================================================================
GLOBAL STATE DIAGNOSTIC (First Step) - CRITIC7: Absolute coordinates
================================================================================
Global state shape: (500, 11)
Expected: [500, 11] for 2 agents (absolute world frame)

Environment 0 global state (11 dims):
  Box:    x=2.155, y=1.456, yaw=-0.451
  Target: x=-1.205, y=1.012
  Agent0: x=-0.678, y=1.028, yaw=4.876
  Agent1: x=5.000, y=0.924, yaw=-0.775
```

---

## Quick Reference Commands

```bash
# CRITIC8 (concatenated observations)
./run_training.sh --exp_name test_c8 --use_concat_agent_observations_critic True

# CRITIC9 (box-centered)
./run_training.sh --exp_name test_c9 --use_box_centered_critic True

# CRITIC7 (absolute) - DEFAULT
./run_training.sh --exp_name test_c7

# Check config after training starts
cat results/mapush/go1push_mid/happo/<exp_name>/seed-*/config.json | grep "critic"
```

---

## Testing Different Modes

To compare all three modes:

```bash
# Start three parallel training runs
./run_training.sh --exp_name critic7_abs --seed 1 &
./run_training.sh --exp_name critic8_concat --seed 1 --use_concat_agent_observations_critic True &
./run_training.sh --exp_name critic9_box --seed 1 --use_box_centered_critic True &

# Monitor with tensorboard
tensorboard --logdir results/mapush/go1push_mid/happo/
```

---

## Notes

- **Default is CRITIC7** (absolute coordinates) when no flags are specified
- **CRITIC8 takes priority** if both flags are True (though this is not recommended)
- All modes work with the same actor observations (only critic input changes)
- Config is saved in `config.json` for reproducibility
