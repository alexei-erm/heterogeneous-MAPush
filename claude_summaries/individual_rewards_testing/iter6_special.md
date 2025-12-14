# Iteration 6 Special: Positive Reinforcement + Velocity-Based Rewards

**Date:** 2025-12-14
**Status:** Ready to train
**Based on:** Successful Iter10 from previous experiments

---

## Why Iter5 Failed

From iter5 analysis:
- Agent 0: -7.76 → -6.95 (stable)
- Agent 1: -7.49 → **-18.66** (tanking!)

Agent 1 learned to escape/freeload. The approach penalty wasn't enough - agent got stuck in local minimum where escaping was "easier" than learning to push.

**Key insight from your Iter10:** Positive reinforcement works better than punishment.

---

## Iter6 Philosophy

**From Iter10's success:**
1. **Engagement bonus** - POSITIVE reward for being near box
2. **Cooperation bonus** - POSITIVE reward when BOTH agents near box
3. **Same-side bonus** - POSITIVE reward when both on push side
4. **Blocking penalty** - Only penalty is for being in wrong position
5. **Velocity-based push** - Reward actual box velocity toward goal
6. **Shared directional progress** - Some shared reward aligns team goals

---

## New Reward Structure

| Component | Scale | Type | Description |
|-----------|-------|------|-------------|
| `engagement_bonus` | 0.02 | per-agent | Being close to box (< 1.5m) |
| `cooperation_bonus` | 0.01 | per-agent | BOTH agents near box |
| `same_side_bonus` | 0.02 | per-agent | BOTH agents on push side |
| `blocking_penalty` | -0.05 | per-agent | Agent between box and goal |
| `goal_push_bonus` | 0.003 | per-agent | Box velocity toward goal * contact_weight |
| `directional_progress` | 0.15 | **SHARED** | Box distance to goal decreased |

**Old rewards kept:**
- `reach_target_reward` - individual (contact-weighted)
- `target_reward` - individual (contact-weighted)
- `push_reward` - individual (contact-weighted)
- `ocb_reward` - individual
- `approach_reward` - individual (penalty for being far)
- `collision_punishment` - individual

---

## Implementation Details

### 1. Engagement Bonus (Positive Proximity!)
```python
engagement_threshold = 1.5  # meters
engagement = clamp(1.0 - agent_dist / engagement_threshold, 0.0, 1.0)
reward[:, i] += engagement_bonus_scale * engagement
```

### 2. Cooperation Bonus
```python
both_near = (dist_agent0 < 1.5) & (dist_agent1 < 1.5)
reward[:, :] += cooperation_bonus_scale * both_near.float()
```

### 3. Same Side Bonus
```python
# Agent on push side if: dot(agent_to_box, box_to_target) > 0
both_on_push_side = on_push_side[0] & on_push_side[1]
reward[:, :] += same_side_bonus_scale * both_on_push_side.float()
```

### 4. Blocking Penalty
```python
# Agent blocking if: dot(box_to_agent, box_to_target) > 0 (in front of box)
reward[:, i] += blocking_penalty_scale * is_blocking.float()  # negative scale
```

### 5. Velocity-Based Push Contribution
```python
# KEY FIX: Use actual box velocity direction, not agent position!
box_vel_norm = box_vel / (box_speed + 1e-6)
velocity_alignment = dot(box_vel_norm, box_to_target_norm)  # -1 to 1
push_contribution = scale * velocity_alignment * box_speed * contact_weight
push_contribution = clamp(push_contribution, min=0)  # only positive
```

### 6. Shared Directional Progress
```python
old_dist = norm(last_box_pos - target_pos)
new_dist = norm(box_pos - target_pos)
progress = old_dist - new_dist  # positive = closer to goal
reward[:, :] += directional_progress_scale * progress  # SHARED
```

---

## Files Modified

| File | Change |
|------|--------|
| `mqe/envs/wrappers/go1_push_mid_wrapper.py:71-85` | New reward buffer entries |
| `mqe/envs/wrappers/go1_push_mid_wrapper.py:97-102` | New reward scales |
| `mqe/envs/wrappers/go1_push_mid_wrapper.py:445-541` | Iter6 reward calculations |
| `HARL/harl/envs/mapush/mapush_logger.py:116-120` | New reward logging |

---

## Usage

```bash
./run_training.sh --exp_name happo_iter6_special --individualized_rewards
```

---

## Monitoring

Watch in TensorBoard:
- `train_episode_rewards/agent_0` and `agent_1` - Should be more balanced now
- `rewards/engagement_bonus` - Should be positive, increasing
- `rewards/cooperation_bonus` - Indicates both agents near box
- `rewards/same_side_bonus` - Indicates both on correct side
- `rewards/blocking_penalty` - Should decrease as agents learn
- `rewards/goal_push_bonus` - Should increase as agents push toward goal
- `mapush/success_rate` - The ultimate metric

---

## Expected Behavior

**Early (0-15M):**
- Agents learn engagement bonus → stay near box
- Cooperation bonus reinforces both staying

**Mid (15-30M):**
- Same-side bonus → both position behind box
- Blocking penalty → don't stand in front

**Late (30M+):**
- Velocity-based push + directional progress → push toward goal
- Success rate should climb

---

## Why This Should Work

1. **Positive > Negative**: Engagement/cooperation bonuses pull agents toward box instead of pushing them away from escaping

2. **Clear signals**: Each reward component has a clear purpose and doesn't conflict

3. **Velocity-based**: Rewards actual pushing direction, not just position

4. **Some shared reward**: Directional progress aligns team goals without causing freeloading (dominated by per-agent rewards 3:1)

---

## If This Doesn't Work (Iter7 Ideas)

1. Increase `engagement_bonus_scale` to 0.05
2. Reduce `directional_progress_scale` to 0.05 (less shared)
3. Add curriculum: start with closer targets
4. Tune `contact_threshold` - maybe 1.0m instead of 0.8m
