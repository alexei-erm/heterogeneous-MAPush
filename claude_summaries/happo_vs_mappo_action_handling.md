# HAPPO vs MAPPO Action Handling for Heterogeneous Agents

**Date:** 2026-01-15
**Context:** Heterogeneous agent support (Go1 + Jackal)
**Status:** ✅ Simplified - Both use unified 3 DOF action space

---

## Overview

After initial implementation with per-agent action dimensions, we simplified the design: **both Go1 and Jackal now use the same 3 DOF action space [vx, vy, vyaw]**. This eliminates the need for masking or per-agent network dimensions.

---

## Key Design Decision

**The user insight:** "I'm not sure about the handling of the actor network... vyaw is useless for jackal. Is it better to adapt the network's dimensions to output 2 when the flag is used or do masking?"

**The realization:** vyaw (yaw rate) is **NOT** useless for Jackal! It's crucial for controlling the robot's orientation in the 2D plane. A differential drive robot needs vyaw to turn.

**The solution:** Both agents use [vx, vy, vyaw] action space. The difference is in the **low-level controller**, not the high-level action space.

---

## Unified Action Space Design

### Both HAPPO and MAPPO: Same Action Space
**Network Architecture:** Doesn't matter - both use 3 DOF
**Approach:** Unified high-level commands

```python
# Both agents use same action space
action_space = Box(low=-1, high=1, shape=(3,))  # [vx, vy, vyaw]
action_spaces = [action_space] * n_agents
```

**Benefits:**
- ✅ Simple: No masking, no per-agent dimensions
- ✅ Unified: Same abstraction level for all agents
- ✅ Meaningful: vyaw is crucial for both Go1 and Jackal
- ✅ Scalable: Easy to add new robots with same action interface

**Implementation Location:**
- HAPPO: `HARL/harl/envs/mapush/mapush_env.py:113-120`
- MAPPO: `mqe/envs/wrappers/go1_push_mid_wrapper.py:56-66`

---

## How It Works

### High-Level Actions (Learned by Network)
Both Go1 and Jackal receive the same high-level velocity commands:
- **vx:** Forward/backward velocity (m/s)
- **vy:** Lateral velocity (m/s)
- **vyaw:** Yaw rate / angular velocity (rad/s)

### Low-Level Controllers (Robot-Specific)

#### Go1: Learned Locomotion Policy
```python
# Go1 uses a neural network policy (pre-trained)
joint_torques = locomotion_policy([vx, vy, vyaw])
# Outputs 12 joint torques for 4 legs
```

#### Jackal: Differential Drive Kinematics
```python
# Jackal uses kinematic equations (differential_drive_controller)
left_wheel_vel = (vx - vyaw * track_width/2) / wheel_radius
right_wheel_vel = (vx + vyaw * track_width/2) / wheel_radius
# Outputs 2 wheel velocities
```

**Note:** vy (lateral velocity) is difficult for differential drive due to non-holonomic constraints, but the network can learn to minimize it or use coordinated motion.

---

## Technical Implementation

### Unified Action Space

**File:** `HARL/harl/envs/mapush/mapush_env.py`

```python
# Action space: Both Go1 and Jackal use [vx, vy, vyaw] (3 DOF)
# Jackal's differential drive controller internally converts to wheel velocities
self.action_space = [self.env.action_space] * self.n_agents

if self.is_hetero:
    print(f"[MAPushEnv] Heterogeneous agents with unified action space:")
    print(f"  Agent 0 (Go1): 3 DOF [vx, vy, vyaw] → Locomotion policy")
    print(f"  Agent 1 ({hetero_agent}): 3 DOF [vx, vy, vyaw] → Differential drive controller")
```

**File:** `mqe/envs/jackal/jackal.py`

```python
def differential_drive_controller(self, vx, vy, vyaw):
    """Convert high-level velocity commands to wheel velocities.

    Differential drive kinematics:
    - vx and vyaw can be directly achieved
    - vy is constrained by non-holonomic dynamics
    """
    left_wheel_vel = (vx - vyaw * self.track_width / 2) / self.wheel_radius
    right_wheel_vel = (vx + vyaw * self.track_width / 2) / self.wheel_radius

    wheel_velocities = torch.stack([left_wheel_vel, right_wheel_vel], dim=-1)
    return wheel_velocities

def step(self, action):
    # action shape: [num_envs, num_agents, 3]
    vx = action[..., 0]
    vy = action[..., 1]
    vyaw = action[..., 2]

    # Convert to wheel velocities
    wheel_actions = self.differential_drive_controller(vx, vy, vyaw)
    # wheel_actions shape: [num_envs, num_agents, 2]

    # Apply to physics engine
    ...
```

---

## Comparison: Before vs After

### Before (Overcomplicated)
| Aspect | HAPPO | MAPPO |
|--------|-------|-------|
| Action Space | Per-agent (Go1: 3, Jackal: 2) | Padded to max (both: 3) |
| Network Output | 3 for Go1, 2 for Jackal | 3 for both (Jackal masks 3rd) |
| Complexity | Moderate | High (masking logic) |
| Scalability | Good | Poor (more masking) |

### After (Simplified)
| Aspect | HAPPO | MAPPO |
|--------|-------|-------|
| Action Space | Unified (both: 3) | Unified (both: 3) |
| Network Output | 3 for both | 3 for both |
| Complexity | Low | Low |
| Scalability | Excellent | Excellent |

---

## Why This Is Better

### 1. **Semantic Correctness**
- vyaw (yaw rate) is **essential** for Jackal to turn
- Not just Go1 feature - fundamental for any mobile robot
- Both agents benefit from orientation control

### 2. **Simpler Implementation**
- No masking logic needed
- No per-agent action dimension tracking
- Less code, fewer bugs

### 3. **Better Learning**
- Network learns same high-level interface for all agents
- Transfer learning potential
- Easier to add new robots

### 4. **Cleaner Abstraction**
- High-level: What the agent should do (velocities)
- Low-level: How to achieve it (robot-specific)
- Separation of concerns

---

## Code Locations Summary

| Component | Location | Purpose |
|-----------|----------|---------|
| Unified action space (HAPPO) | `HARL/harl/envs/mapush/mapush_env.py:113-120` | Same for both agents |
| Unified action space (MAPPO) | `mqe/envs/wrappers/go1_push_mid_wrapper.py:56-66` | Same for both agents |
| Differential drive controller | `mqe/envs/jackal/jackal.py:60-90` | Converts [vx,vy,vyaw] to wheels |
| Jackal configuration | `mqe/envs/jackal/jackal_config.py:24` | num_actions = 3 |
| Robot registry | `mqe/envs/robot_registry.py:40` | num_actions: 3 |

---

## Testing Commands

### HAPPO Training
```bash
cd /home/gvlab/new-universal-MAPush/HARL/harl_mapush

# Heterogeneous (Go1 + Jackal) - Unified action space
python train.py \
  --exp_name test_go1_jackal_unified \
  --hetero_agent jackal \
  --num_env_steps 10000 \
  --n_rollout_threads 10
```

### MAPPO Training
```bash
cd /home/gvlab/new-universal-MAPush

# Heterogeneous (Go1 + Jackal) - Unified action space
python openrl_ws/train.py \
  --algo ppo \
  --task go1push_mid \
  --num_envs 500 \
  --hetero_agent jackal
```

---

## Differential Drive Kinematics

For a differential drive robot:
- **Track width (L):** 0.37559 m (distance between wheels)
- **Wheel radius (r):** 0.098 m

Given desired velocities:
- **vx:** Linear velocity in forward direction
- **vyaw:** Angular velocity around z-axis

Wheel velocities:
```
ω_left = (vx - vyaw * L/2) / r
ω_right = (vx + vyaw * L/2) / r
```

**Forward motion:** vx > 0, vyaw = 0 → both wheels same speed
**Rotation:** vx = 0, vyaw > 0 → wheels opposite direction
**Arc motion:** vx > 0, vyaw > 0 → turning while moving

---

## Future Robots

When adding a new robot with different low-level DOF:

**Step 1:** Keep high-level action space as [vx, vy, vyaw] (3 DOF)
**Step 2:** Implement robot-specific low-level controller
**Step 3:** Register in robot_registry with num_actions: 3

Examples:
- **Omnidirectional wheeled robot:** Mecanum wheels controller
- **Ackermann steering robot:** Car-like kinematics controller
- **Bipedal robot:** Walking policy controller

All use the same [vx, vy, vyaw] interface!

---

## References

- Differential Drive Kinematics: Standard robotics textbook
- Jackal Implementation: `mqe/envs/jackal/jackal.py`
- User Design Decision: Session 2026-01-15
- Simplified Design: Inspired by user questioning vyaw necessity

---

**Status:** ✅ **Simplified and production-ready**

Both HAPPO and MAPPO now use the same clean, unified action space design. No masking, no per-agent dimensions - just a clear separation between high-level commands and robot-specific controllers.
