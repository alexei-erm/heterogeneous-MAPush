# Iteration 4: Direction-Aware Approach + Reduced Collision

**Date:** 2025-12-14
**Status:** Ready to train

---

## Why Iter3 Failed

Observed behavior:
1. **Freeloading agent walks away**: Leaves the scene entirely in a straight line
2. **Pushing agent ignores direction**: Approaches from any side, pushes randomly (sometimes away from goal)

**Root cause analysis:**

The idle penalty was useless - agent was MOVING away, not idle. But why escape?

**Collision penalty was dominating:**
```
collision_punishment = (1 / (0.02 + distance/3)) * -0.0025
```

At 0.5m inter-agent distance: **-0.0134/step**
At 5m from box (escaped): approach penalty only **-0.0227/step**

Agent rationally escapes because:
- Far away = no collision penalty
- Approach penalty alone is similar magnitude to collision penalty
- Escaping is the optimal policy under these reward scales!

---

## Iter4 Strategy: Two Fixes

### Fix 1: Direction-Aware Approach Penalty

The `approach_reward` is actually a PENALTY (negative): `-(distance+0.5)^2 * scale`

- Normal penalty (1x) if on **CORRECT side** (behind box relative to target)
- DOUBLE penalty (2x) if on **WRONG side** (in front of box)

This discourages positioning on the wrong side of box.

### Fix 2: Reduce Collision Penalty

Reduce `collision_punishment_scale` from `-0.0025` to `-0.0008` (3x weaker).

Now agents tolerate being close to each other while pushing.

---

## Changes Summary

| Component | Iter3 | Iter4 |
|-----------|-------|-------|
| `approach_reward` | Direction-agnostic | **Direction-aware** (2x penalty on wrong side) |
| `collision_punishment_scale` | -0.0025 | **-0.0008** (3x weaker) |
| `idle_penalty` | -0.005 | **REMOVED** (redundant) |

---

## Implementation

### Direction-aware approach (go1_push_mid_wrapper.py:353-378)

```python
# Iter4: Direction-aware approach PENALTY
box_to_target = target_pos[:, :2] - box_pos[:, :2]
box_to_target_norm = box_to_target / (torch.norm(...) + 1e-6)

for i in range(self.num_agents):
    agent_to_box = box_pos[:, :2] - agent_pos_2d

    # dot > 0 means agent is behind box (correct pushing position)
    dot_product = torch.sum(agent_to_box * box_to_target_norm, dim=1)
    on_wrong_side = (dot_product <= 0).float()

    base_distance_penalty = (-(distance+0.5)**2) * approach_scale  # NEGATIVE

    # 1x penalty if correct side, 2x penalty if wrong side
    direction_multiplier = 1.0 + on_wrong_side * 1.0
    distance_reward = base_distance_penalty * direction_multiplier
```

### Collision scale (go1_push_mid_config.py:109)

```python
collision_punishment_scale = -0.0008  # was -0.0025
```

---

## Math Analysis

**New collision penalty at 0.5m:**
```
(1 / (0.02 + 0.167)) * -0.0008 = -0.0043/step  (was -0.0134)
```

**Approach penalty at 1m from box (near box, pushing):**
```
-(1.5)^2 * 0.00075 = -0.00169/step
```

**Total for staying and pushing:** -0.006/step

**Approach penalty at 5m (escaped):** -0.0227/step

**Verdict:** Staying is now ~4x better than escaping!

---

## Files Modified

| File | Change |
|------|--------|
| `mqe/envs/wrappers/go1_push_mid_wrapper.py:353-378` | Direction-aware approach penalty |
| `mqe/envs/wrappers/go1_push_mid_wrapper.py:412-413` | Removed idle penalty |
| `mqe/envs/configs/go1_push_mid_config.py:109` | collision_punishment_scale: -0.0025 → -0.0008 |

---

## Usage

```bash
./run_training.sh --exp_name happo_iter4 --individualized_rewards
```

---

## Expected Behavior

1. **No more escaping**: Staying near box is clearly better than running away
2. **Correct positioning**: 2x penalty on wrong side encourages circling to pushing side
3. **Tolerate proximity**: Agents can be close to each other without excessive penalty
4. **Collaborative pushing**: Both agents engaged, pushing from correct side

---

## Monitoring

Watch in TensorBoard:
- `train_episode_rewards` - Should be stable/increasing
- `mapush/success_rate` - Key metric
- `rewards/approach_box` - Direction-weighted penalty
- `rewards/collision_punishment` - Should be smaller magnitude now

---

## If This Doesn't Work (Iter5 Ideas)

1. **Increase wrong-side multiplier** to 3x or 4x
2. **Further reduce collision** to -0.0005 or remove entirely
3. **Boost OCB reward scale** - already rewards correct positioning
4. **Add explicit "push toward goal" reward** - reward when box velocity aligns with target direction
