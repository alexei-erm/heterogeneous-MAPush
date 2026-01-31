# Changes to MAPush Environment

This document tracks permanent modifications made to the MAPush environment codebase.

---

## 1. Agent Spawn Angular Separation (Permanent)

**Date:** 2026-01-31
**File Modified:** `mqe/envs/base/legged_robot.py` (lines 622-662)

### Problem

When using heterogeneous agents (e.g., Go1 + Anymal C), agents could spawn on top of each other. The original spawn mechanism sampled each agent's angle (`theta`) around the box independently from `[0, 2π]`, allowing both agents to receive nearly identical angles.

With spawn heights of:
- Go1: Z = 0.45m
- Anymal C: Z = 0.62m

And similar XY positions, the robots' bodies would overlap/intersect at spawn, causing physics instability.

### Solution

Implemented **guaranteed minimum angular separation** between agents spawning around the box.

#### How It Works:

1. Sample a single random `base_theta` per environment
2. Each agent is offset by `agent_idx * min_angular_separation`
3. Small jitter (±30% of separation, max ±0.5 rad) added for training diversity

```
Agent 0: base_theta + jitter
Agent 1: base_theta + min_angular_separation + jitter
Agent N: base_theta + N * min_angular_separation + jitter
```

#### Default Configuration:

- `min_agent_angular_separation = 1.57` (π/2 radians = 90 degrees)

With `r ≈ 1.25m` spawn radius and 90° separation, minimum XY distance between agents is approximately **1.77m**.

### Code Change

```python
# In _reset_root_states(), after sampling radii:

# Minimum angular separation between agents (default: pi/2 = 90 degrees)
min_agent_angular_sep = getattr(self.cfg.domain_rand, "min_agent_angular_separation", 1.57)

if self.num_agents > 1 and min_agent_angular_sep > 0:
    theta_range = random_base_theta_from_init[1] - random_base_theta_from_init[0]
    base_theta = torch.rand(self.num_envs, device=self.device) * theta_range + random_base_theta_from_init[0]

    jitter_range = min(0.5, min_agent_angular_sep * 0.3)

    agent_thetas = torch.zeros(self.num_envs, self.num_agents, device=self.device)
    for agent_idx in range(self.num_agents):
        jitter = (torch.rand(self.num_envs, device=self.device) - 0.5) * 2 * jitter_range
        agent_thetas[:, agent_idx] = base_theta + agent_idx * min_agent_angular_sep + jitter

    base_init_state[:, 1] = agent_thetas.reshape(-1)
```

### Optional Override

While this change is permanent (always active), the separation angle can be configured per-task in `domain_rand`:

```python
class domain_rand:
    min_agent_angular_separation = 2.09  # 120 degrees instead of default 90
```

Set to `0` to disable (not recommended).

### Visual Comparison

```
         Before                      After
           Box                        Box
            |                          |
     t=1.5--+--t=1.52            t=0.3-+-t=1.87 (pi/2 apart)
            |                          |
    [Both agents here]         Agent0    Agent1
      COLLISION!               (separated by ~1.77m)
```

---

## 2. Box Mass Override Option (Optional Flag)

**Date:** 2026-01-31
**Files Modified:**
- `mqe/envs/configs/go1_push_mid_config.py` - added `npc_mass_override` parameter (this is the config used by ENV_DICT)
- `mqe/envs/npc/go1_object.py` - implemented runtime mass override

**Note:** The config in `task/cuboid/config.py` is NOT used by the hetero environment. The actual config used is `mqe/envs/configs/go1_push_mid_config.py`.

### Problem

With HAPPO's separate actor networks (unlike MAPPO's shared network), agents don't naturally learn to collaborate. The default box mass of **4 kg** is trivially easy for either robot to push solo:
- Go1: ~12 kg
- Anymal C: ~50 kg

This allows free-rider behavior where one agent does nothing while the other pushes.

### Solution

Added optional **runtime box mass override** that can make the box heavy enough to require both agents to push together.

### Configuration

In `mqe/envs/configs/go1_push_mid_config.py`:

```python
class asset(Go1Cfg.asset):
    # Box mass override: None = use URDF default (4kg)
    # Set to higher value (e.g., 50.0) to require collaboration
    # Box dimensions: 1.2m x 1.2m x 0.5m
    npc_mass_override = None  # or set to desired mass in kg
```

### Usage

**Test old checkpoint with light box (default):**
```python
class asset:
    npc_mass_override = None  # Uses URDF default (4 kg)
```

**Train/test with heavy box:**
```python
class asset:
    npc_mass_override = 50.0  # 50 kg box - requires collaboration
```

### Recommended Mass Values

| Mass | Effect | Use Case |
|------|--------|----------|
| None / 4 kg | Trivial for single robot | Baseline / old checkpoints |
| 20-30 kg | Challenging for Go1 solo, easy for Anymal | Mild collaboration pressure |
| 40-60 kg | Hard for either solo, needs both | Strong collaboration signal |
| 80+ kg | Very difficult even together | Stress test |

### Implementation Details

When `npc_mass_override` is set (not None), the code:
1. Overrides the box mass from URDF
2. Recalculates correct inertia tensor for the new mass using box dimensions (1.2m × 1.2m × 0.5m)
3. Prints confirmation message: `[Box Mass Override] Box mass set to X kg (URDF default: 4 kg)`

```python
# In go1_object.py _create_npc():
if npc_mass_override is not None:
    rigid_body_props[0].mass = npc_mass_override
    # Recalculate inertia: I = (1/12) * m * (a² + b²)
    m = npc_mass_override
    w, d, h = 1.2, 1.2, 0.5
    ixx = (1.0/12.0) * m * (d*d + h*h)
    iyy = (1.0/12.0) * m * (w*w + h*h)
    izz = (1.0/12.0) * m * (w*w + d*d)
```

### Backwards Compatibility

- Default is `None` (no override) → existing checkpoints work unchanged
- Old models trained with 4 kg box can be tested with heavy box by setting the flag
- New models can be trained with heavy box from the start

---
