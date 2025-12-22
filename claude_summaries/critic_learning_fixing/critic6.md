# CRITIC6: Action Scaling Fix (Match OpenRL) - **FAILED, REVERTED**

> **Date:** December 19, 2025
> **Goal:** Match OpenRL's action scaling to fix 2x velocity difference
> **Impact:** ~~Agents now move at same speed as OpenRL baseline~~
>
> ## **RESULT: FAILED - REVERTED ON DEC 21, 2025**
>
> This change **completely broke learning**. Evidence:
> - **critic3 (no scaling)**: 2.5% → **20.2% success** in 100M steps
> - **critic6 (with 0.5x scaling)**: 0% → **0% success** in 100M steps
>
> The 0.5x action scaling made actions too small (±0.25 m/s) to effectively push the box.
> The entropy increased by 65.8%, indicating the policy found no useful actions and drifted to maximum randomness.
>
> **Lesson learned**: Don't blindly match OpenRL scaling. HARL's bounded std (max ~0.5) combined with action scaling creates insufficient exploration.

---

## Problem Discovered

While comparing OpenRL (backup_MAPush) and HARL implementations, we found a **critical difference in action scaling**:

| Framework | Wrapper Scale | MQE Scale | **Total** | Effective Range |
|-----------|---------------|-----------|-----------|-----------------|
| OpenRL | 0.5x | 0.5x | **0.25x** | [-0.25, 0.25] |
| HARL | None | 0.5x | **0.5x** | [-0.5, 0.5] |

**HARL agents were moving 2x faster than OpenRL agents!**

This could explain training instability - faster movements mean:
- Larger state changes per step
- Harder for critic to predict value
- More variance in returns
- Potentially overshooting targets

---

## Solution

Added OpenRL-compatible action scaling in `mapush_env.py`:

```python
# CRITIC6 (Dec 19, 2025): Match OpenRL action scaling
# OpenRL applies 0.5x scale + clip in wrapper BEFORE MQE's 0.5x scale
# This makes effective range [-0.25, 0.25] instead of [-0.5, 0.5]
# Without this, HARL agents move 2x faster than OpenRL agents
actions_torch = (0.5 * actions_torch).clamp(-1.0, 1.0)
```

---

## Action Flow Comparison

### Before (HARL only)

```
Network output → MQE wrapper (0.5x) → Simulation
                 ↓
         Range: [-0.5, 0.5] m/s
```

### After (Matching OpenRL)

```
Network output → HARL wrapper (0.5x + clip) → MQE wrapper (0.5x) → Simulation
                 ↓
         Range: [-0.25, 0.25] m/s
```

---

## File Changes

**Modified:** `HARL/harl/envs/mapush/mapush_env.py`

```python
# In step() method, after converting actions to torch:
actions_torch = (0.5 * actions_torch).clamp(-1.0, 1.0)
```

---

## Expected Impact

1. **Slower agent movements** - More controlled, less aggressive
2. **Smoother trajectories** - Smaller velocity changes
3. **Better credit assignment** - Smaller state transitions easier to learn
4. **Matches OpenRL baseline** - Fair comparison possible

---

## Additional Difference (Not Changed)

The DiagGaussian std parameterization is also different:

| Framework | Std Formula | Behavior |
|-----------|-------------|----------|
| OpenRL | `exp(log_std)` | Unbounded, can be very large |
| HARL | `sigmoid(log_std) * 0.5` | Bounded in (0, 0.5) |

This was **NOT changed** because:
- Bounded std provides more stable exploration
- The clipping after sampling limits extreme actions anyway
- Action scaling fix is the more critical change

If needed, this can be addressed in a future update by modifying `HARL/harl/models/base/distributions.py`.

---

## Combined with critic5

critic6 should be run with critic5 settings:

| Parameter | Value |
|-----------|-------|
| `lr` | 0.0005 |
| `critic_lr` | 0.003 |
| `critic_epoch` | 30 |
| `actor_update_interval` | 5 |
| `clip_param` | 0.1 |
| `value_loss_coef` | 3.0 |
| `max_grad_norm` | 5.0 |
| `gae_lambda` | 0.9 |
| `actor_hidden_sizes` | [256, 256] |
| `critic_hidden_sizes` | [256, 256, 128] |
| **+ Action scaling** | 0.5x (new) |

---

## History

| Version | Date | Key Changes |
|---------|------|-------------|
| critic1 | Dec 18 | Critic epoch increase |
| critic2 | Dec 18 | Value loss coefficient increase |
| critic3 | Dec 18 | Value normalizer fix |
| critic4 | Dec 19 | Actor update interval |
| critic5 | Dec 19 | Full stability config + separate architectures |
| **critic6** | Dec 19 | **Action scaling fix (match OpenRL) - FAILED** |
| **critic7** | Dec 21 | **Remove velocities from critic input** |

---

# CRITIC7: Remove Agent Velocities from Critic Input

> **Date:** December 21, 2025
> **Goal:** Simplify critic input to match OpenRL baseline more closely
> **Status:** TESTING

---

## Rationale

The critic was receiving agent velocities (vx, vy, vyaw) as input, but:

1. **OpenRL baseline doesn't use velocities** - The working OpenRL critic uses only positions
2. **Velocities ≈ Actions** - In MAPush, actions ARE velocity commands. Including velocities blurs the line between V(s) and Q(s,a)
3. **Simpler is better** - 11 dims vs 17 dims means fewer parameters to learn

---

## Before vs After Comparison

### BEFORE (17 dimensions)
```
Global State:
├── Box:     [x, y, yaw]                           = 3 dims
├── Target:  [x, y]                                = 2 dims
├── Agent 0: [x, y, yaw, vx, vy, vyaw]             = 6 dims  ← velocities included
└── Agent 1: [x, y, yaw, vx, vy, vyaw]             = 6 dims  ← velocities included
                                            TOTAL = 17 dims
```

### AFTER (11 dimensions)
```
Global State:
├── Box:     [x, y, yaw]                           = 3 dims
├── Target:  [x, y]                                = 2 dims
├── Agent 0: [x, y, yaw]                           = 3 dims  ← positions only
└── Agent 1: [x, y, yaw]                           = 3 dims  ← positions only
                                            TOTAL = 11 dims
```

---

## Index Mapping

| Index | Before (17-dim) | After (11-dim) |
|-------|-----------------|----------------|
| 0 | box_x | box_x |
| 1 | box_y | box_y |
| 2 | box_yaw | box_yaw |
| 3 | target_x | target_x |
| 4 | target_y | target_y |
| 5 | agent0_x | agent0_x |
| 6 | agent0_y | agent0_y |
| 7 | agent0_yaw | agent0_yaw |
| 8 | agent0_vx | agent1_x |
| 9 | agent0_vy | agent1_y |
| 10 | agent0_vyaw | agent1_yaw |
| 11 | agent1_x | - |
| 12 | agent1_y | - |
| 13 | agent1_yaw | - |
| 14 | agent1_vx | - |
| 15 | agent1_vy | - |
| 16 | agent1_vyaw | - |

---

## File Changes

**Modified:** `HARL/harl/envs/mapush/mapush_env.py`

```python
# Line 91: Changed dimension calculation
global_state_dim = 3 + 2 + 3 * self.n_agents  # was: 6 * self.n_agents

# Lines 172-174: Removed velocity collection
# REMOVED:
# global_state_list.append(base_lin_vel[:, agent_id, :2])  # agent vx, vy
# global_state_list.append(base_ang_vel[:, agent_id, 2:3]) # agent vyaw
```

---

## Expected Impact

1. **Simpler value function** - 11 inputs vs 17 = ~35% reduction in input size
2. **Cleaner V(s) estimation** - No action-like information in state
3. **Faster training** - Smaller network input
4. **Better match to OpenRL** - Same information available to critic

---

## Combined Changes (critic7)

Building on critic3 (best performing so far):

| Change | Status |
|--------|--------|
| Action scaling (0.5x) | ❌ REVERTED (broke learning) |
| Remove velocities from critic | ✅ NEW |
| Keep: value_loss_coef = 5.0 | ✅ |
| Keep: critic_epoch = 25 | ✅ |
| Keep: value normalizer fix | ✅ |
