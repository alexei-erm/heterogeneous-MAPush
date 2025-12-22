# Critic Coordinate System Flag

> **Date:** December 22, 2025
> **Feature:** Switchable critic input coordinate system

---

## Overview

Added a flag `use_box_centered_critic` to quickly switch between two critic input representations:

| Mode | Flag Value | Dims | Description |
|------|-----------|------|-------------|
| **Box-centered (CRITIC9)** | `True` (default) | 9 | Everything relative to box - translation invariant |
| **Absolute (CRITIC7)** | `False` | 11 | Absolute world frame coordinates |

---

## Usage

### Option 1: Command Line (Recommended)

Add the flag when running training:

```bash
# Use box-centered coordinates (default - CRITIC9)
python HARL/harl_mapush/train.py \
    --algo_name happo \
    --env_name mapush \
    --exp_name critic9 \
    --use_box_centered_critic True

# Use absolute coordinates (CRITIC7)
python HARL/harl_mapush/train.py \
    --algo_name happo \
    --env_name mapush \
    --exp_name critic7_absolute \
    --use_box_centered_critic False
```

### Option 2: Modify train.py

Edit `HARL/harl_mapush/train.py` line ~88:

```python
env_args = {
    "task": args.get("task", "go1push_mid"),
    "n_threads": n_threads,
    "state_type": "EP",
    "individualized_rewards": use_individual,
    "shared_gated_rewards": use_shared_gated,
    "use_box_centered_critic": True,  # <-- Add this line (True or False)
}
```

---

## Coordinate Systems Explained

### Box-Centered (CRITIC9) - `use_box_centered_critic=True`

**Global state (9 dims):**
```
[target_x - box_x, target_y - box_y,              # 2: Target relative to box
 agent0_x - box_x, agent0_y - box_y,              # 2: Agent0 relative to box
 agent0_yaw - box_yaw,                            # 1: Agent0 yaw relative to box
 agent1_x - box_x, agent1_y - box_y,              # 2: Agent1 relative to box
 agent1_yaw - box_yaw,                            # 1: Agent1 yaw relative to box
 box_yaw]                                         # 1: Box absolute orientation
```

**Advantages:**
- ✅ Translation invariant (box at different positions = same state)
- ✅ Task-centric (value depends on box-target distance)
- ✅ More compact (9 dims vs 11)

**Example:**
```
Box at (5, 5), target at (7, 5), agents around box:
  State = [2, 0, -0.5, 0, 0, 0.5, 0, 0, 0]  # Target 2m ahead of box

Box at (100, 100), target at (102, 100), same setup:
  State = [2, 0, -0.5, 0, 0, 0.5, 0, 0, 0]  # SAME STATE!
```

### Absolute (CRITIC7) - `use_box_centered_critic=False`

**Global state (11 dims):**
```
[box_x, box_y, box_yaw,                           # 3: Box position
 target_x, target_y,                              # 2: Target position
 agent0_x, agent0_y, agent0_yaw,                  # 3: Agent0 position
 agent1_x, agent1_y, agent1_yaw]                  # 3: Agent1 position
```

**Characteristics:**
- ❌ Not translation invariant (same relative setup at different positions = different states)
- ✅ Simpler to understand (raw world coordinates)
- ⚠️ Requires critic to learn "position doesn't matter"

**Example:**
```
Box at (5, 5), target at (7, 5):
  State = [5, 5, 0, 7, 5, 4.5, 5, 0, 5.5, 5, 0]

Box at (100, 100), target at (102, 100), same relative setup:
  State = [100, 100, 0, 102, 100, 99.5, 100, 0, 100.5, 100, 0]  # DIFFERENT!
```

---

## Implementation Details

**File:** `HARL/harl/envs/mapush/mapush_env.py`

### Lines 82 (flag reading):
```python
self.use_box_centered_critic = env_args.get("use_box_centered_critic", True)
```

### Lines 86-112 (dimension setup):
```python
if self.use_box_centered_critic:
    global_state_dim = 2 + 3 * self.n_agents + 1  # 9 dims for 2 agents
else:
    global_state_dim = 3 + 2 + 3 * self.n_agents  # 11 dims for 2 agents
```

### Lines 194-237 (state construction):
```python
if self.use_box_centered_critic:
    # Construct box-centered state
    target_rel = target_pos - box_pos
    agent_rel = agent_pos - box_pos
    # ...
else:
    # Construct absolute state
    global_state = [box_pos, box_yaw, target_pos, agent0_pos, agent0_yaw, ...]
    # ...
```

---

## Diagnostic Output

On first step, you'll see different diagnostic prints:

### Box-centered (CRITIC9):
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

### Absolute (CRITIC7):
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

## Recommendations

**Default: Box-centered (CRITIC9)**
- Better for learning (translation invariant)
- Proven effective in similar tasks
- More compact representation

**Use Absolute (CRITIC7) only if:**
- You want to baseline against absolute coordinates
- Debugging coordinate transformations
- Comparing with other implementations that use absolute frames

---

## Quick Test

To verify the flag works:

```bash
# Test box-centered
python HARL/harl_mapush/train.py --algo_name happo --env_name mapush --exp_name test_bc --use_box_centered_critic True

# Should see: "CRITIC9: Box-centered" in diagnostic output
# State shape: (500, 9)

# Test absolute
python HARL/harl_mapush/train.py --algo_name happo --env_name mapush --exp_name test_abs --use_box_centered_critic False

# Should see: "CRITIC7: Absolute coordinates" in diagnostic output
# State shape: (500, 11)
```
