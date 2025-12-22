# CRITIC9: Box-Centered Global State

> **Date:** December 22, 2025
> **Goal:** Alternative approach - express everything relative to the box
> **Status:** IMPLEMENTED

---

## Concept

Instead of using absolute coordinates (critic7) or concatenated local observations (critic8), express everything **relative to the box** - the object being pushed.

**Key insight:** The box is the center of the task. The value of a state depends on:
1. How far the box is from the target
2. Where the agents are relative to the box
3. The orientations

By using box-centered coordinates, we get translation invariance while maintaining a truly global view.

---

## Proposed Structure (9 dims)

```
Box-Centered Global State:
├── Target relative to box:
│   └── [target_x - box_x, target_y - box_y]    = 2 dims
├── Agent 0 relative to box:
│   ├── [agent0_x - box_x, agent0_y - box_y]    = 2 dims
│   └── [agent0_yaw - box_yaw]                  = 1 dim
├── Agent 1 relative to box:
│   ├── [agent1_x - box_x, agent1_y - box_y]    = 2 dims
│   └── [agent1_yaw - box_yaw]                  = 1 dim
└── Box orientation:
    └── [box_yaw]                                = 1 dim

TOTAL: 2 + 3*n_agents + 1 = 9 dims (for 2 agents)
```

---

## Why Box-Centered?

### 1. Translation Invariance
```
Box at (5, 5), target at (7, 5), agents around box:
  State = [2, 0, -0.5, 0, 0, 0.5, 0, 0, ...]  # Target 2m ahead of box

Box at (100, 100), target at (102, 100), same setup:
  State = [2, 0, -0.5, 0, 0, 0.5, 0, 0, ...]  # SAME STATE!
```

### 2. Task-Centric
The VALUE of a state depends primarily on:
- How far is box from target? (box-target distance)
- Can agents push the box? (agent-box relative positions)

Box-centered coordinates capture exactly this.

### 3. True Global View
Unlike concatenated local obs (critic8), this gives ONE unified view of the entire scene, not two separate agent perspectives.

### 4. Compact
Only 9 dims vs 16 dims for critic8.

---

## Comparison to Other Approaches

| Approach | Dims | Translation Invariant | Rotation Invariant | True Global |
|----------|------|----------------------|-------------------|-------------|
| critic7 (absolute) | 11 | ❌ | ❌ | ✅ |
| critic8 (concat local) | 16 | ✅ | ✅ | ⚠️ Two views |
| **critic9 (box-centered)** | 9 | ✅ | ⚠️ Partial | ✅ |

---

## Implementation Plan

### In `__init__`:
```python
# CRITIC9: Box-centered global state
# [target_rel(2), agent0_rel(3), agent1_rel(3), box_yaw(1)] = 9 dims (for 2 agents)
global_state_dim = 2 + 3 * self.n_agents + 1  # target(2) + agents(3 each) + box_yaw(1)
```

### In `step()` and `reset()`:

```python
def _construct_box_centered_state(self) -> np.ndarray:
    """Construct global state relative to box position and orientation."""

    # Get box state
    box_pos = ...   # [n_envs, 3]
    box_yaw = ...   # [n_envs, 1]

    # Get target state
    target_pos = ...  # [n_envs, 3]

    # Get agent states
    agent_pos = ...   # [n_envs, n_agents, 3]
    agent_yaw = ...   # [n_envs, n_agents, 1]

    # Compute relative positions (optionally rotate to box frame)
    target_rel = target_pos[:, :2] - box_pos[:, :2]  # [n_envs, 2]

    agent_rel = []
    for i in range(self.n_agents):
        pos_rel = agent_pos[:, i, :2] - box_pos[:, :2]  # [n_envs, 2]
        yaw_rel = agent_yaw[:, i] - box_yaw            # [n_envs, 1]
        agent_rel.append(np.concatenate([pos_rel, yaw_rel], axis=1))

    # Concatenate: [target_rel(2), agent0_rel(3), agent1_rel(3), box_yaw(1)]
    global_state = np.concatenate([
        target_rel,      # 2 dims
        *agent_rel,      # 3 * n_agents dims
        box_yaw,         # 1 dim (absolute orientation)
    ], axis=1)

    return global_state
```

---

## Optional: Full Rotation to Box Frame

For full rotation invariance, we could rotate all positions to the box's frame:

```python
# Rotate to box's frame (box always "faces forward")
cos_yaw = np.cos(-box_yaw)
sin_yaw = np.sin(-box_yaw)

# Rotate target
target_rotated_x = (target_pos[:,0] - box_pos[:,0]) * cos_yaw - (target_pos[:,1] - box_pos[:,1]) * sin_yaw
target_rotated_y = (target_pos[:,0] - box_pos[:,0]) * sin_yaw + (target_pos[:,1] - box_pos[:,1]) * cos_yaw

# Similarly for agents...
```

This would make the state fully rotation invariant (same scenario with box rotated differently = same state).

---

## When to Try This

Implement critic9 if:
1. critic8 (concatenated local obs) fails
2. We want a more compact representation
3. We want a true "global" view rather than two agent perspectives

---

## Potential Issues

1. **Rotation handling:** Need to decide if we rotate to box frame or just use relative positions
2. **Box yaw reference:** Including box_yaw as absolute might reintroduce some variance
3. **Agent coordination:** Two-agent perspectives in critic8 might help with coordination

---

## Files to Modify (when implementing)

1. `HARL/harl/envs/mapush/mapush_env.py`
   - Change `share_observation_space` dimension
   - Implement `_construct_box_centered_state()` or modify `_construct_global_state()`
   - Update `step()` and `reset()` to use new state

---

## Implementation Details (COMPLETED)

### Files Modified:

**1. `HARL/harl/envs/mapush/mapush_env.py`**

#### Changed share_observation_space (lines 79-100):
```python
# Changed from 16 dims (critic8) to 9 dims (critic9)
global_state_dim = 2 + 3 * self.n_agents + 1  # target(2) + agents(3 each) + box_yaw(1) = 9 dims for 2 agents
```

#### Modified `_construct_global_state()` (lines 105-193):
- Target relative to box: `target_pos - box_pos`
- Each agent relative to box: `agent_pos - box_pos`, `agent_yaw - box_yaw`
- Box absolute yaw: `box_yaw`

#### Updated `step()` method (line 268):
```python
global_state_np = self._construct_global_state()  # Changed from obs_np.reshape()
```

#### Updated `reset()` method (line 316):
```python
global_state_np = self._construct_global_state()  # Changed from obs_np.reshape()
```

---

## History

| Version | Approach | Dims | Status |
|---------|----------|------|--------|
| critic7 | Absolute global | 11 | Failed |
| critic8 | Concatenated local | 16 | Testing |
| **critic9** | **Box-centered** | **9** | **IMPLEMENTED** |
