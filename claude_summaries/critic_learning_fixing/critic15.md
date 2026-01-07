# CRITIC15: Original MAPush Rewards (Teamified)

> **Date:** January 7, 2026
> **Philosophy:** Return to vanilla MAPush reward structure, properly teamified for centralized critic
> **Goal:** Establish baseline with original reward design before adding new signals
> **Status:** IMPLEMENTED
> **Parent:** Derived from CRITIC14 (team rewards fix)

---

## Motivation

CRITIC14 showed that team rewards improve learning (approach_to_box learned in 25M steps). However, the reward structure had drifted significantly from original MAPush:
- Added `goal_push_bonus` (0.01) - not original
- Added `proximity_penalty` (0.002) - not original
- Modified `ocb_reward` to asymmetric +0.01/-0.004 - original was ±0.004
- Disabled `distance_to_target_reward` - was original
- Changed `collision_punishment` scale from -0.0025 to -0.0008

**Problem observed:** Agents struggle when spawning between box and target. They must navigate behind the box, leading to collisions and pushing box away from goal.

**Hypothesis:** Before adding more complexity, establish a clean baseline with original MAPush rewards properly teamified.

---

## The Fix

New flag: `--mapush_og_rewards_teamified True`

When enabled:
1. **Re-enable** `distance_to_target_reward` with original formula
2. **Average** (not sum) `approach_to_box_reward` to preserve magnitude
3. **Restore** `collision_punishment` to original scale -0.0025
4. **Use symmetric** OCB ±0.004 (original scale)
5. **Disable** `goal_push_bonus` (not in original)
6. **Disable** `proximity_penalty` (not in original)

---

## CRITIC14 vs CRITIC15 Comparison

| Reward | CRITIC14 | CRITIC15 (OG Teamified) |
|--------|----------|------------------------|
| `reach_target_reward` | 10 | 10 |
| `distance_to_target_reward` | **DISABLED** | **0.00325 (RE-ENABLED)** |
| `approach_to_box_reward` | 0.00075 (sum) | 0.00075 **(AVERAGE)** |
| `collision_punishment` | -0.0025 | **-0.0025** |
| `push_reward` | 0.0015 | 0.0015 |
| `goal_push_bonus` | 0.01 | **DISABLED** |
| `ocb_reward` | +0.01/-0.004 (asymmetric) | **±0.004 (symmetric)** |
| `proximity_penalty` | 0.002 | **DISABLED** |
| `exception_punishment` | -5 | -5 |
| **Total Active** | 8 | **7** |

---

## Reward Details

### 1. `reach_target_reward` (scale: 10) - UNCHANGED
```python
reward[self.finished_buf, :] += 10  # TEAM
```

### 2. `distance_to_target_reward` (scale: 0.00325) - RE-ENABLED
```python
# Original formula with distance penalty term
distance_reward = scale * 100 * (2 * (past_distance - distance) - 0.01 * distance)
reward[:, :] += distance_reward  # TEAM
```
**Why re-enabled:** Provides continuous progress signal. `goal_push_bonus` measured velocity, this measures position delta.

### 3. `approach_to_box_reward` (scale: 0.00075) - AVERAGED
```python
total_penalty = sum of both agents' penalties
avg_penalty = total_penalty / num_agents  # AVERAGE to preserve magnitude
reward[:, :] += avg_penalty  # TEAM
```
**Why averaged:** Original was per-agent. Summing would double the effective scale. Averaging preserves designed magnitude.

### 4. `collision_punishment` (scale: -0.0025) - RESTORED
```python
collision_scale = -0.0025  # Original scale (was changed to -0.0008 in Iter4)
collision_punishment = (1 / (0.02 + agent_distance / 3)) * collision_scale
reward[:, :] += collision_punishment  # TEAM
```
**Why restored:** Original MAPush used -0.0025. Iter4 reduction may have allowed too much agent proximity.

### 5. `push_reward` (scale: 0.0015) - UNCHANGED
```python
push_reward[box_moving] = 0.0015
reward[:, :] += push_reward  # TEAM
```

### 6. `ocb_reward` (scale: ±0.004) - SYMMETRIC (JOINT BINARY)
```python
# Joint binary: both must be on correct side
both_correct = (raw_ocb_0 > 0) & (raw_ocb_1 > 0)
joint_ocb_reward = torch.where(
    both_correct,
    +0.004,  # Both correct (was +0.01)
    -0.004   # Any wrong (unchanged)
)
reward[:, :] += joint_ocb_reward  # TEAM
```
**Why symmetric:** Original MAPush used 0.004 scale. Asymmetric +0.01/-0.004 was experimental.

**IMPORTANT NOTE:** This is **joint binary** OCB (from CRITIC12 v5), NOT the original continuous per-agent OCB. The original was:
```python
# Original MAPush (per-agent continuous):
for i in range(num_agents):
    ocb_reward = dot(target_direction, normal_vector) * 0.004
    reward[:, i] += ocb_reward  # Each agent gets their own continuous value
```
See v2 proposal below for restoring continuous OCB.

### 7. `exception_punishment` (scale: -5) - UNCHANGED
```python
reward[exception_buf, :] += -5  # TEAM
```

---

## Disabled Rewards

### `goal_push_bonus` - NOT ORIGINAL
Added in CRITIC12 v7. Rewards box velocity toward goal. Redundant with `distance_to_target_reward`.

### `proximity_penalty` - NOT ORIGINAL
Added in CRITIC13 v3. Quadratic penalty for agents being too far apart. Novel addition, not in original.

---

## Implementation

### Flag: `--mapush_og_rewards_teamified`

```bash
# Training command
./run_training.sh --algo happo --env mapush --exp_name critic15 \
    --use_concat_agent_observations_critic True \
    --mapush_og_rewards_teamified True \
    --seed 7
```

### Code Changes

**Files modified:**
1. `HARL/harl_mapush/train.py` - Added argument
2. `mqe/envs/utils.py` - Added to `custom_cfg()`
3. `HARL/harl/envs/mapush/mapush_env.py` - Pass flag through
4. `mqe/envs/wrappers/go1_push_mid_wrapper.py` - Implement reward logic

### Wrapper Logic Summary

```python
# In go1_push_mid_wrapper.py

# distance_to_target: only when flag is True
if self.mapush_og_rewards_teamified and self.target_reward_scale != 0:
    # Original formula with penalty term
    distance_reward = scale * 100 * (2*(past-curr) - 0.01*dist)

# approach_to_box: average when flag is True
if self.mapush_og_rewards_teamified:
    total_approach_penalty = total_approach_penalty / self.num_agents

# collision: hardcode -0.0025 when flag is True
collision_scale = -0.0025 if self.mapush_og_rewards_teamified else self.collision_punishment_scale

# goal_push_bonus: skip when flag is True
if self.goal_push_bonus_scale != 0 and not self.mapush_og_rewards_teamified:

# ocb: symmetric ±0.004 when flag is True
if self.mapush_og_rewards_teamified:
    joint_ocb_reward = torch.where(both_correct, +0.004, -0.004)

# proximity_penalty: skip when flag is True
if self.proximity_penalty_scale != 0 and not self.mapush_og_rewards_teamified:
```

---

## Expected Behavior

### Compared to Original MAPush (MAPPO)
- Same 7 rewards with same scales
- Team rewards instead of per-agent (for centralized critic compatibility)
- `approach_to_box` averaged to preserve per-agent magnitude
- `ocb` is joint binary (both must be correct) instead of per-agent continuous

### Compared to CRITIC14
- Simpler reward structure (7 vs 8 rewards)
- No velocity-based directional signal (goal_push_bonus disabled)
- No proximity enforcement (proximity_penalty disabled)
- Stronger collision avoidance (-0.0025 vs config value)
- Balanced OCB (±0.004 symmetric vs +0.01/-0.004 asymmetric)

---

## Success Metrics

| Metric | CRITIC14 | CRITIC15 Target |
|--------|----------|-----------------|
| Success Rate | TBD | Baseline |
| approach_to_box convergence | 25M steps | Similar or better |
| Behind-box navigation | Poor | Improved (hypothesis) |
| Agent collisions | Some | Reduced (stronger penalty) |

---

## Version History

| Run Directory | Version | Key Config |
|---------------|---------|------------|
| TBD | v1 | Original MAPush rewards teamified: 7 rewards, symmetric OCB, -0.0025 collision |

---

## If This Works

This establishes a clean baseline. Future iterations can:
1. Add back `goal_push_bonus` if directional signal needed
2. Add back `proximity_penalty` if agents drift apart
3. Tune individual scales with clear understanding of baseline

## If This Fails

Possible issues:
1. No velocity signal → agents push slowly → add back `goal_push_bonus`
2. Agents drift apart → no cooperation → add back `proximity_penalty`
3. Symmetric OCB too weak → restore asymmetric +0.01/-0.004
4. Joint binary OCB too coarse → restore continuous OCB (v2 proposal)

---

## v2 Proposal: Restore Continuous OCB (RECOMMENDED)

**Motivation:** Current v1 uses joint binary OCB (from CRITIC12 v5), not the original continuous per-agent OCB. Joint binary only gives feedback when BOTH agents are positioned correctly, which may be too sparse.

**Why Averaged Continuous is Better:**
1. **Consistent with other rewards:** Like `approach_to_box_reward`, we average to preserve scale
2. **More robust:** Doesn't rely solely on critic for credit assignment - each agent gets partial immediate feedback
3. **Simpler:** No complex hybrid schemes or minimum operations
4. **Closer to original:** Matches original MAPush formula, just teamified

**Change:**
```python
# v2: Restore continuous OCB (original formula) and average for team reward
total_ocb = torch.zeros(self.num_envs, device=self.device)
for i in range(self.num_agents):
    gf_pos = base_pos[:, i, :2] - box_pos[:, :2]
    rotation_matrix = rotation_matrix_2D(-box_rpy[:, 2])
    box_relative_pos = torch.bmm(rotation_matrix, gf_pos.unsqueeze(2)).squeeze(2)
    normal_vector = self.calc_normal_vector_for_obc_reward(vertex_list, box_relative_pos)
    rotation_matrix = rotation_matrix_2D(box_rpy[:, 2])
    normal_vector = torch.bmm(rotation_matrix, normal_vector.to(rotation_matrix.device).unsqueeze(2)).squeeze(2)

    # Continuous OCB: dot product can range from -1 to +1
    raw_ocb = torch.sum(target_direction * normal_vector, dim=1)
    ocb_reward = raw_ocb * 0.004  # Original scale
    total_ocb += ocb_reward

# Team reward: AVERAGE to preserve scale magnitude
avg_ocb = total_ocb / self.num_agents  # Average (not sum) to preserve designed scale
reward[:, :] += avg_ocb.unsqueeze(1).repeat(1, self.num_agents)
```

**Behavior Examples:**

| Agent 0 OCB | Agent 1 OCB | Avg OCB | Reward (×0.004) | Interpretation |
|-------------|-------------|---------|-----------------|----------------|
| +1.0 | +1.0 | +1.0 | **+0.004** | Both perfectly positioned |
| +1.0 | +0.5 | +0.75 | **+0.003** | One perfect, one good |
| +1.0 | 0.0 | +0.5 | **+0.002** | One perfect, one neutral |
| +1.0 | -0.5 | +0.25 | **+0.001** | One perfect, one wrong (still positive!) |
| +0.5 | -0.5 | 0.0 | **0.0** | Balanced |
| -0.5 | -0.5 | -0.5 | **-0.002** | Both wrong |
| -1.0 | -1.0 | -1.0 | **-0.004** | Both maximally wrong |

**Key Advantage:** Agent 0's good action (+1.0) still gives positive team reward even if Agent 1 is wrong (-0.5). This provides **immediate partial credit** and doesn't rely solely on the critic for credit assignment.

**Expected difference:**
- **v1 (joint binary):** Only two reward values: +0.004 or -0.004, depends on critic for all credit assignment
- **v2 (continuous averaged):** Smooth gradient from -0.004 to +0.004, each agent gets partial immediate credit

**When to use v2:**
- If v1 shows poor OCB learning (agents don't learn positioning)
- If you want more robust credit assignment (less critic-dependent)
- To truly match original MAPush behavior (continuous OCB)
