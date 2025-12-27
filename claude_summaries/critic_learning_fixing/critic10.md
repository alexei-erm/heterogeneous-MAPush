# CRITIC10: Flag-Based Concatenated Agent Observations

> **Date:** December 26, 2025
> **Goal:** Use unmodified concatenated agent observations for critic input
> **Status:** IMPLEMENTED
> **Flag:** `--use_concat_agent_observations_critic True`

---

## Overview

CRITIC10 implements a **flag-controlled system** where the critic receives concatenated agent observations **exactly as the actors see them** - no modifications, no transformations, just pure concatenation.

This is similar to CRITIC8 conceptually, but now properly implemented with:
1. **Flag-based control** - Easy to switch on/off
2. **Verified agent-centric observations** - Each obs is in agent's local frame
3. **Clean architecture** - Part of unified flag system with CRITIC7 and CRITIC9

---

## Key Principle

**The critic sees exactly what the actors see, concatenated together.**

No additional state construction, no global frame transformations - just `[agent0_obs, agent1_obs]`.

---

## The Flag System

### Three Critic Modes Available

| Mode | Flag | Default | Dimensions |
|------|------|---------|-----------|
| **CRITIC10** | `--use_concat_agent_observations_critic True` | **False** | 16 dims |
| **CRITIC9** | `--use_box_centered_critic True` | False | 9 dims |
| **CRITIC7** | Neither (both False) | **TRUE** | 11 dims |

### Flag Priority

When multiple flags are set:
1. `use_concat_agent_observations_critic` takes **highest priority**
2. `use_box_centered_critic` takes second priority
3. Neither → defaults to absolute coordinates (CRITIC7)

---

## Observation Structure

### Per-Agent Observation (8 dims)

From `mqe/envs/wrappers/go1_push_mid_wrapper.py`, each agent's observation is **already in local frame**:

```python
# Lines 175-217 in go1_push_mid_wrapper.py
# Observations are rotated to each agent's local frame:
rotated_box_pos = ...  # Box position rotated to agent's heading
rotated_target_pos = ... # Target position rotated to agent's heading
rotated_box_rpy[:, 2] = box_rpy[:, 2] - base_rpy[:, 2]  # Box yaw relative to agent

# Final observation construction:
obs = torch.cat([
    rotated_target_pos[:, :, :2],        # Target (x, y) in agent's frame: 2 dims
    rotated_box_pos[:, :, :2],           # Box (x, y) in agent's frame: 2 dims
    rotated_box_rpy[:, :, 2].unsqueeze(2), # Box yaw relative to agent: 1 dim
    all_base_info                        # Other agents' states in this agent's frame: 3 dims
], dim=2)
```

**Each agent observation (8 dims):**
```
[target_x, target_y,              # 2 dims: Target relative to agent
 box_x, box_y,                    # 2 dims: Box relative to agent
 box_yaw,                         # 1 dim:  Box yaw relative to agent heading
 other_agent_x, other_agent_y,    # 2 dims: Other agent position relative to this agent
 other_agent_yaw]                 # 1 dim:  Other agent yaw relative to this agent
```

### Concatenated Global State (16 dims)

```
Global State = [Agent0_obs, Agent1_obs]

Index | Content | Frame
------|---------|-------
0-1   | Target (x, y) from agent0's perspective | Agent0 local
2-3   | Box (x, y) from agent0's perspective | Agent0 local
4     | Box yaw from agent0's perspective | Agent0 local
5-7   | Agent1 (x, y, yaw) from agent0's perspective | Agent0 local
8-9   | Target (x, y) from agent1's perspective | Agent1 local
10-11 | Box (x, y) from agent1's perspective | Agent1 local
12    | Box yaw from agent1's perspective | Agent1 local
13-15 | Agent0 (x, y, yaw) from agent1's perspective | Agent1 local

Total: 16 dims
```

---

## Why This Should Work

### 1. Translation Invariance ✅

Because each observation is **relative to the agent**, the same scenario at different map positions produces the **same state vector**.

```
Scenario: Both agents 1m from box, box 2m from target

At position (5, 5):
  State = [1, 0, 2, 0, 0, -1, 0, 0,    # Agent0's view
           1, 0, 2, 0, 0, 1, 0, 0]     # Agent1's view

At position (100, 100):
  State = [1, 0, 2, 0, 0, -1, 0, 0,    # SAME!
           1, 0, 2, 0, 0, 1, 0, 0]
```

### 2. Rotation Invariance ✅

Each observation is **rotated to the agent's heading** (lines 175-183 in wrapper). The agent always "faces forward" in its own observation.

### 3. Matches Actor Information ✅

The critic sees **exactly what the actors use to make decisions**. No information mismatch between actor and critic.

### 4. Two Perspectives 👁️👁️

Unlike CRITIC7 or CRITIC9 which give a single global view, CRITIC10 provides **both agents' perspectives**. This may help the critic understand:
- What each agent can see
- Coordination challenges (different viewpoints)
- Symmetry in the task

---

## Comparison to Previous Critics

| Aspect | CRITIC7 (Absolute) | CRITIC9 (Box-centered) | CRITIC10 (Concat Obs) |
|--------|-------------------|----------------------|---------------------|
| **Dimensions** | 11 | 9 | 16 |
| **Frame** | World absolute | Box-relative | Agent-relative (two views) |
| **Translation Inv.** | ❌ | ✅ | ✅ |
| **Rotation Inv.** | ❌ | ⚠️ Partial | ✅ |
| **Actor-Critic Match** | ❌ Different | ❌ Different | ✅ **Same** |
| **Global View** | ✅ Single | ✅ Single | ⚠️ Two perspectives |
| **Complexity** | Simple | Medium | Simple (just concat) |

---

## Implementation Details

### In `__init__()` (mapush_env.py:85-103)

```python
self.use_concat_agent_observations_critic = env_args.get("use_concat_agent_observations_critic", False)
self.use_box_centered_critic = env_args.get("use_box_centered_critic", False)

if self.use_concat_agent_observations_critic:
    # CRITIC10: Concatenated agent local observations
    obs_dim = self.env.observation_space.shape[0]  # 8 dims per agent
    global_state_dim = obs_dim * self.n_agents     # 8 * 2 = 16 dims
elif self.use_box_centered_critic:
    # CRITIC9: Box-centered
    global_state_dim = 2 + 3 * self.n_agents + 1  # 9 dims
else:
    # CRITIC7: Absolute
    global_state_dim = 3 + 2 + 3 * self.n_agents  # 11 dims
```

### In `step()` (mapush_env.py:348-356)

```python
# Construct global state based on critic mode
if self.use_concat_agent_observations_critic:
    # CRITIC10: Simply concatenate agent observations (no modification)
    # obs_np is [n_envs, n_agents, obs_dim], flatten to [n_envs, n_agents * obs_dim]
    global_state_np = obs_np.reshape(self.n_envs, -1)
else:
    # CRITIC9 or CRITIC7: Use box-centered or absolute global state
    global_state_np = self._construct_global_state()
```

### In `reset()` (mapush_env.py:403-410)

```python
# Construct global state based on critic mode
if self.use_concat_agent_observations_critic:
    # CRITIC10: Simply concatenate agent observations (no modification)
    global_state_np = obs_np.reshape(self.n_envs, -1)
else:
    # CRITIC9 or CRITIC7: Use box-centered or absolute global state
    global_state_np = self._construct_global_state()
```

---

## Diagnostic Output

When training starts with CRITIC10, you'll see:

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

Statistics across all 500 environments:
  Min values:  [...]
  Max values:  [...]
  Mean values: [...]
  Std values:  [...]
```

---

## Files Modified

### 1. `HARL/harl_mapush/train.py`

**Line 48-49:** Added flag argument
```python
parser.add_argument("--use_concat_agent_observations_critic", type=lambda x: (str(x).lower() == 'true'), default=False,
                   help="Use concatenated agent observations for critic (CRITIC10). Takes priority over other critic modes. DEFAULT: False")
```

**Line 91:** Read flag value
```python
use_concat_obs = args.get("use_concat_agent_observations_critic", False)
```

**Line 101:** Pass flag to environment
```python
"use_concat_agent_observations_critic": use_concat_obs,  # CRITIC10: Concatenated agent observations (takes priority)
```

### 2. `HARL/harl/envs/mapush/mapush_env.py`

**Lines 79-86:** Flag setup and priority comment
```python
# Flags to control critic input coordinate system
# Priority: concat_observations > box_centered > absolute
# use_concat_agent_observations_critic: CRITIC10 (16 dims) - Concatenated agent local observations
# use_box_centered_critic: CRITIC9 (9 dims) - Box-centered coordinates (translation invariant)
# Neither: CRITIC7 (11 dims) - Absolute world frame coordinates
# DEFAULT: Both False (absolute coordinates)
self.use_concat_agent_observations_critic = env_args.get("use_concat_agent_observations_critic", False)
self.use_box_centered_critic = env_args.get("use_box_centered_critic", False)
```

**Lines 90-103:** Dimension calculation with priority
```python
if self.use_concat_agent_observations_critic:
    # CRITIC10: Concatenated agent local observations
    obs_dim = self.env.observation_space.shape[0]  # 8 dims per agent
    global_state_dim = obs_dim * self.n_agents     # 16 dims
elif self.use_box_centered_critic:
    # CRITIC9: Box-centered
    global_state_dim = 2 + 3 * self.n_agents + 1  # 9 dims
else:
    # CRITIC7: Absolute
    global_state_dim = 3 + 2 + 3 * self.n_agents  # 11 dims
```

**Lines 348-356, 403-410:** State construction in `step()` and `reset()`
```python
if self.use_concat_agent_observations_critic:
    global_state_np = obs_np.reshape(self.n_envs, -1)
else:
    global_state_np = self._construct_global_state()
```

**Lines 273-286:** Updated diagnostic logging
```python
if self.use_concat_agent_observations_critic:
    print("GLOBAL STATE DIAGNOSTIC (First Step) - CRITIC8: Concatenated Agent Observations")
    # ... detailed output
elif self.use_box_centered_critic:
    print("GLOBAL STATE DIAGNOSTIC (First Step) - CRITIC9: Box-centered")
    # ... detailed output
else:
    print("GLOBAL STATE DIAGNOSTIC (First Step) - CRITIC7: Absolute coordinates")
    # ... detailed output
```

---

## Usage

### Training with CRITIC10

```bash
# Basic usage
./run_training.sh \
    --algo happo \
    --env mapush \
    --exp_name critic10_test \
    --use_concat_agent_observations_critic True

# With additional parameters
./run_training.sh \
    --algo happo \
    --env mapush \
    --exp_name critic10_lr_test \
    --use_concat_agent_observations_critic True \
    --seed 42 \
    --n_rollout_threads 500
```

### Verify Configuration

After training starts:

```bash
# Check config file
cat results/mapush/go1push_mid/happo/critic10_test/seed-*/config.json | grep "use_concat_agent_observations_critic"

# Should show:
# "use_concat_agent_observations_critic": true
```

### Testing the Checkpoint

```bash
# Same as before - test.py reads config automatically
./run_testing.sh \
    --checkpoint results/mapush/go1push_mid/happo/critic10_test/seed-*/checkpoints/100M \
    --mode viewer \
    --num_episodes 10
```

---

## Expected Benefits

1. **Actor-Critic Consistency**
   - Critic uses same information as actors
   - No distribution mismatch between what's learned and what's used

2. **Translation & Rotation Invariance**
   - Observations are already in agent-local frames
   - Same relative scenario = same state vector

3. **Coordination Awareness**
   - Two perspectives may help critic understand:
     - What each agent can see
     - Asymmetric information
     - Coordination requirements

4. **Simplicity**
   - No complex state construction
   - Just `reshape()` - minimal room for bugs

---

## Potential Concerns

1. **Redundant Information**
   - Some info appears twice (box position from both perspectives)
   - May make learning slower initially

2. **Higher Dimensionality**
   - 16 dims vs 11 (CRITIC7) or 9 (CRITIC9)
   - Larger critic network may need more samples

3. **Two Perspectives vs One Global**
   - Unlike CRITIC7/9 which give single global view
   - Critic must "fuse" two viewpoints

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

## History Context

| Version | Date | Approach | Dims | Result |
|---------|------|----------|------|--------|
| critic1 | Dec 18 | Increased critic_epoch | 17 | Failed |
| critic2 | Dec 18 | Increased value_loss_coef | 17 | Failed |
| critic3 | Dec 18 | Value normalizer fix | 17 | 20% success |
| critic4 | Dec 19 | Actor update interval | 17 | Failed |
| critic5 | Dec 19 | Full stability config | 17 | Failed |
| critic6 | Dec 19 | Action scaling 0.5x | 17 | 0% success (broke learning) |
| critic7 | Dec 21 | Absolute coordinates (removed velocities) | 11 | Testing |
| critic8 | Dec 22 | Concat observations + reverted all fixes | 16 | Testing |
| critic9 | Dec 22 | Box-centered coordinates | 9 | Testing |
| **critic10** | Dec 26 | **Flag-based concat observations** | 16 | **IMPLEMENTED** |

---

## Quick Reference

```bash
# CRITIC10 (concatenated observations)
./run_training.sh --exp_name c10 --use_concat_agent_observations_critic True

# CRITIC9 (box-centered)
./run_training.sh --exp_name c9 --use_box_centered_critic True

# CRITIC7 (absolute - DEFAULT)
./run_training.sh --exp_name c7

# Compare all three in parallel
./run_training.sh --exp_name c7_compare --seed 1 &
./run_training.sh --exp_name c9_compare --seed 1 --use_box_centered_critic True &
./run_training.sh --exp_name c10_compare --seed 1 --use_concat_agent_observations_critic True &
```

---

## Notes

- **CRITIC10 = CRITIC8 concept**, but properly implemented with flags
- All previous critic fixes (epochs, learning rates, etc.) have been **reverted to defaults**
- The key difference now is **only the critic input representation**, not training hyperparameters
- This allows clean A/B testing of critic input modes
