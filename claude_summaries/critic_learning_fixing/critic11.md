# CRITIC11: Relative Observations with Inter-Robot Distance

> **Date:** December 27, 2025
> **Goal:** Provide critic with relative observations and explicit inter-robot distance
> **Status:** IMPLEMENTED
> **Flag:** `--use_relative_obs_critic True`

---

## Overview

CRITIC11 implements a **relative observation** approach for the centralized critic, adding **explicit inter-robot distance** as a feature. This combines benefits of relative coordinates (translation invariance) with explicit coordination information.

---

## Structure (9 dimensions)

```
critic_state = [
    robot1_to_box,           # (dx, dy, dψ) = 3 dims
    robot2_to_box,           # (dx, dy, dψ) = 3 dims
    inter_robot_distance,    # scalar = 1 dim
    goal_to_box,             # (dx, dy) = 2 dims
]

Total: 3 + 3 + 1 + 2 = 9 dims
```

### Component Details

1. **robot_to_box** (per agent, 3 dims each)
   - `dx, dy`: Agent position relative to box (agent.pos - box.pos)
   - `dψ`: Agent yaw relative to box yaw (agent.yaw - box.yaw)

2. **inter_robot_distance** (1 dim)
   - Euclidean distance between agents: `||robot1.pos - robot2.pos||`
   - Explicit coordination signal

3. **goal_to_box** (2 dims)
   - `dx, dy`: Target position relative to box (target.pos - box.pos)
   - Task progress indicator

---

## Key Properties

### ✅ Translation Invariance

Everything is expressed relative to the box, so the same configuration at different map positions produces the same state vector.

```
Box at (5, 5), target at (7, 5), agents around box:
  State = [robot1_rel, robot2_rel, dist, target_rel]

Box at (100, 100), target at (102, 100), same setup:
  State = [robot1_rel, robot2_rel, dist, target_rel]  # SAME!
```

### ✅ Explicit Coordination Information

Unlike CRITIC9 which only has agent positions relative to box, CRITIC11 **explicitly includes inter-robot distance**.

**Why this matters:**
- Critic can directly observe agent separation
- Coordination states (close together vs far apart) are explicit features
- No need to learn to compute distance from positions

### ✅ Task-Centric

Like CRITIC9, everything is relative to the box (the object being pushed):
- Value depends on: box-target distance, agent-box positions, agent-agent distance
- Invariant to global position

### ✅ Compact Representation

Only 9 dims, same as CRITIC9, less than CRITIC10 (16 dims).

---

## Comparison to Other Critics

| Aspect | CRITIC7 (11D) | CRITIC9 (9D) | CRITIC10 (16D) | CRITIC11 (9D) |
|--------|---------------|--------------|----------------|---------------|
| **Frame** | Absolute world | Box-relative | Agent-relative (2 views) | Box-relative |
| **Translation Inv.** | ❌ | ✅ | ✅ | ✅ |
| **Explicit Inter-Robot Dist** | ❌ | ❌ | ❌ | ✅ |
| **Coordination Signal** | Implicit | Implicit | Implicit | **Explicit** |
| **Dims** | 11 | 9 | 16 | 9 |
| **View** | Single global | Single global | Two agent views | Single global |

### CRITIC9 vs CRITIC11

**Same:**
- Both 9 dims
- Both box-relative
- Both translation invariant

**Different:**
- CRITIC9: `[target_rel(2), agent0_rel(3), agent1_rel(3), box_yaw(1)]`
- CRITIC11: `[agent0_rel(3), agent1_rel(3), inter_dist(1), target_rel(2)]`

**Key advantage of CRITIC11:**
- **Explicit inter-robot distance** as a feature
- No need for critic to compute it from positions
- Direct signal for coordination state

---

## Why This Might Help with Freeloading

### The Freeloading Problem

In CRITIC10's 60% success run, one agent freeloaded because:
- Critic evaluated joint policy as "good enough" (60% success)
- No explicit signal that agents should be coordinated
- Small positive advantages for hovering agent

### How CRITIC11 Addresses This

**Explicit inter-robot distance provides:**

1. **Direct coordination signal**
   - When agents are too far: distance is large
   - When agents are close (coordinating): distance is small
   - Critic can learn: "small distance + both pushing = high value"

2. **Easier to learn coordination value**
   - CRITIC9: Must learn distance from `[x1-box, y1-box, x2-box, y2-box]`
   - CRITIC11: Distance is a direct input feature
   - Reduces learning complexity

3. **Distinguishes solo vs cooperative**
   - Solo pushing: Large inter-robot distance
   - Coordinated pushing: Small inter-robot distance
   - Easier for critic to differentiate value

**Expected behavior:**
- States with small inter-robot distance → higher value (cooperation)
- States with large inter-robot distance → lower value (solo/freeloading)
- Drives both agents to coordinate

---

## Implementation Details

### Files Modified

1. **`HARL/harl_mapush/train.py`**
   - Added `--use_relative_obs_critic` flag (line 50-51)
   - Added flag to env_args (line 94, 105)

2. **`HARL/harl/envs/mapush/mapush_env.py`**
   - Added flag reading (line 86)
   - Updated priority comment (line 80-85)
   - Added dimension calculation (line 92-101)
   - Implemented `_construct_relative_obs_state()` method (line 337-445)
   - Updated `step()` to use new method (line 473-476)
   - Updated `reset()` to use new method (line 531-534)

### State Construction Algorithm

```python
def _construct_relative_obs_state(self):
    # Get box, target, and agent positions from environment
    box_pos, box_yaw = get_box_state()
    target_pos = get_target_state()
    agent_pos, agent_yaw = get_agent_states()  # [n_envs, 2, 3]

    # Robot 1 to box
    robot1_to_box = [
        agent_pos[0] - box_pos,  # dx, dy
        agent_yaw[0] - box_yaw   # dψ
    ]  # 3 dims

    # Robot 2 to box
    robot2_to_box = [
        agent_pos[1] - box_pos,  # dx, dy
        agent_yaw[1] - box_yaw   # dψ
    ]  # 3 dims

    # Inter-robot distance
    inter_dist = ||agent_pos[0] - agent_pos[1]||  # 1 dim

    # Goal to box
    goal_to_box = target_pos - box_pos  # dx, dy: 2 dims

    # Concatenate
    return [robot1_to_box, robot2_to_box, inter_dist, goal_to_box]  # 9 dims
```

### Diagnostic Output

On first step:
```
================================================================================
GLOBAL STATE DIAGNOSTIC (First Step) - CRITIC11: Relative Observations
================================================================================
Global state shape: (500, 9)
Expected: [500, 9] for 2 agents (relative observations)

Environment 0 global state (9 dims):
  Robot1 to box:  dx=-2.833, dy=-0.428, dψ=5.327
  Robot2 to box:  dx=2.845, dy=-0.532, dψ=-0.324
  Inter-robot distance: 5.682
  Goal to box:    dx=2.155, dy=-0.444

Statistics across all 500 environments:
  Min values:  [...]
  Max values:  [...]
  Mean values: [...]
  Std values:  [...]
```

---

## Usage

### Training with CRITIC11

```bash
# Basic usage
./run_training.sh \
    --algo happo \
    --env mapush \
    --exp_name critic11_test \
    --use_relative_obs_critic True \
    --seed 1

# With additional parameters
./run_training.sh \
    --algo happo \
    --env mapush \
    --exp_name critic11_antifreeriding \
    --use_relative_obs_critic True \
    --n_rollout_threads 500 \
    --seed 42
```

### Flag Priority

When multiple flags are set:
1. `--use_relative_obs_critic` (CRITIC11) - **Highest priority**
2. `--use_concat_agent_observations_critic` (CRITIC10)
3. `--use_box_centered_critic` (CRITIC9)
4. None (CRITIC7 - absolute coordinates) - **Default**

### Verify Configuration

```bash
# Check config file after training starts
cat results/mapush/go1push_mid/happo/critic11_test/seed-*/config.json | grep "use_relative_obs_critic"

# Should show:
# "use_relative_obs_critic": true
```

---

## Expected Benefits

### 1. Reduce Freeloading

**Hypothesis:** Explicit inter-robot distance will help critic distinguish:
- Coordinated states (small distance, both pushing) → High value
- Freeloading states (large distance, one hovering) → Medium value
- Both agents get advantage signal to coordinate

### 2. Faster Value Function Convergence

**Reasoning:**
- Simpler input features (distance is explicit, not computed)
- More direct supervision for coordination
- Fewer parameters to learn than CRITIC10 (9 vs 16 dims)

### 3. Better Credit Assignment

**Why:**
- Critic can directly observe whether agents are coordinating
- Distance feature correlates with cooperation
- Clearer advantage signals for both agents

### 4. Translation Invariance

Like CRITIC9, maintains translation invariance for generalization.

---

## Potential Concerns

### 1. May Not Fully Solve Freeloading

**Issue:** Explicit distance alone may not be enough if:
- Solo pushing still achieves 60% success
- Cooperation rewards are still disabled
- Sequential HAPPO updates still allow asymmetric convergence

**Mitigation:** Combine with cooperation reward shaping.

### 2. Distance Metric Choice

**Question:** Should we use:
- Euclidean distance? (current implementation)
- Manhattan distance?
- Projected distance along push direction?

**Current choice:** Euclidean - simple, symmetric, standard.

### 3. Same Dimensionality as CRITIC9

**Observation:** Both are 9 dims, so capacity is the same.

**Difference:** Feature representation, not capacity.
- CRITIC9 relies on box_yaw (absolute orientation)
- CRITIC11 uses inter_robot_distance (coordination signal)

---

## Experiment Design

### Test 1: CRITIC11 vs CRITIC10 (Freeloading)

**Goal:** Does explicit distance reduce freeloading?

```bash
# Run both in parallel
./run_training.sh --exp_name critic10_baseline --use_concat_agent_observations_critic True --seed 1 &
./run_training.sh --exp_name critic11_distance --use_relative_obs_critic True --seed 1 &
```

**Metrics to compare:**
- Agent 0 vs Agent 1 policy loss magnitude difference
- Agent 0 vs Agent 1 gradient norm trends
- Success rate progression
- Value function convergence speed

**Expected result:**
- CRITIC11: More symmetric agent losses (both working)
- CRITIC11: Higher final success rate (80-90% vs 60%)
- CRITIC11: Faster value function convergence

### Test 2: CRITIC11 vs CRITIC9

**Goal:** Does explicit distance help vs implicit distance?

```bash
./run_training.sh --exp_name critic9_implicit --use_box_centered_critic True --seed 1 &
./run_training.sh --exp_name critic11_explicit --use_relative_obs_critic True --seed 1 &
```

**Expected result:**
- CRITIC11: Faster learning (explicit feature easier)
- CRITIC11: Better coordination (distance signal clearer)

---

## Configuration

All standard HAPPO parameters from `happo.yaml` apply:

```yaml
# Actor/Critic architectures
actor_hidden_sizes: [128, 128]
critic_hidden_sizes: [256, 256, 128]

# Training parameters
lr: 0.005
critic_lr: 0.005
ppo_epoch: 5
critic_epoch: 5
clip_param: 0.2
value_loss_coef: 1.0
max_grad_norm: 10.0
gae_lambda: 0.95
entropy_coef: 0.01
```

---

## Summary

**CRITIC11 = Box-relative observations + Explicit inter-robot distance**

**Key Innovation:** Making coordination state (agent-agent distance) an **explicit input feature** rather than letting the critic compute it from positions.

**Motivation:** Address freeloading by providing a direct signal for whether agents are coordinating or not.

**Structure:**
```
[robot1_to_box(3), robot2_to_box(3), inter_robot_dist(1), goal_to_box(2)]
```

**Expected impact:**
- Reduce freeloading (explicit coordination signal)
- Faster value convergence (simpler features)
- Better credit assignment (direct observation of cooperation)

**Next steps:**
- Test against CRITIC10 (compare freeloading behavior)
- Test against CRITIC9 (compare learning speed)
- Combine with cooperation reward shaping if needed
