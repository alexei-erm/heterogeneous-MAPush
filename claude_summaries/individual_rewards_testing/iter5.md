# Iteration 5: All Positive Rewards Individual + Goal Push Bonus

**Date:** 2025-12-14
**Status:** Ready to train

---

## Philosophy Change

**Iter1-4 lesson:** Negative reinforcement doesn't work well. Agents optimize to avoid penalties rather than learn the task.

**Iter5 philosophy:**
- All POSITIVE rewards → INDIVIDUAL (only agents contributing get credit)
- All NEGATIVE penalties → SHARED (keep as is, don't amplify)
- Add NEW positive reward for pushing toward goal

---

## Changes Summary

| Reward | Sign | Iter4 | Iter5 |
|--------|------|-------|-------|
| `reach_target_reward` (+10) | POSITIVE | Shared | **INDIVIDUAL** |
| `target_reward` (distance) | POSITIVE | Shared | **INDIVIDUAL** |
| `push_reward` | POSITIVE | Shared | **INDIVIDUAL** |
| `ocb_reward` | POSITIVE | Individual | Individual (unchanged) |
| **NEW: goal_push_bonus** | POSITIVE | N/A | **INDIVIDUAL** |
| `approach_reward` | NEGATIVE | Direction-aware | Individual (simplified, no direction multiplier) |
| `collision_punishment` | NEGATIVE | Individual | Individual (both agents penalized on collision) |
| `exception_punishment` | NEGATIVE | Shared | Shared (simulation errors) |

**All rewards are now individual except exception_punishment (which is for simulation errors).**

---

## New Reward: goal_push_bonus

The missing signal! Rewards agents when box moves **TOWARD** the goal, not just moves.

```python
# Box velocity toward target
box_vel = box_velocity[:, :2]
box_to_target = target_pos[:, :2] - box_pos[:, :2]
box_to_target_norm = normalize(box_to_target)

# How much box is moving toward goal
goal_velocity = dot(box_vel, box_to_target_norm)
goal_velocity_positive = clamp(goal_velocity, min=0)  # only reward positive

# Only agents close to box get credit
for i in range(num_agents):
    contact_weight = clamp(1.0 - (dist - threshold) / threshold, 0, 1)
    goal_push_reward = goal_push_bonus_scale * goal_velocity_positive * contact_weight
    reward[:, i] += goal_push_reward
```

**Parameters:**
- `goal_push_bonus_scale = 0.003`
- `contact_threshold = 0.8m`

---

## Individual Positive Rewards Logic

All positive rewards use the same contact-based weighting:

```python
contact_weight = clamp(1.0 - (agent_box_dist - threshold) / threshold, 0.0, 1.0)
```

- Agent within 0.8m → weight = 1.0 (full reward)
- Agent at 1.6m → weight = 0.0 (no reward)
- Smooth linear transition between

---

## Per-Agent Logging

Added per-agent reward tracking to TensorBoard:
- `train_episode_rewards/agent_0` - Agent 0's episode reward
- `train_episode_rewards/agent_1` - Agent 1's episode reward

This helps detect freeloading - if one agent consistently gets more reward, the other is freeloading.

---

## Files Modified

| File | Change |
|------|--------|
| `mqe/envs/wrappers/go1_push_mid_wrapper.py:323-335` | reach_target_reward → individual |
| `mqe/envs/wrappers/go1_push_mid_wrapper.py:345-363` | target_reward → individual |
| `mqe/envs/wrappers/go1_push_mid_wrapper.py:365-374` | approach_reward → simplified (no direction multiplier) |
| `mqe/envs/wrappers/go1_push_mid_wrapper.py:388-407` | push_reward → individual |
| `mqe/envs/wrappers/go1_push_mid_wrapper.py:433-455` | NEW: goal_push_bonus |
| `mqe/envs/wrappers/go1_push_mid_wrapper.py:89-90` | goal_push_bonus_scale config |
| `HARL/harl/envs/mapush/mapush_logger.py:27-67` | Per-agent reward tracking |
| `HARL/harl/envs/mapush/mapush_logger.py:74-83` | Per-agent TensorBoard logging |
| `HARL/harl/envs/mapush/mapush_logger.py:115` | goal_push_bonus logging |

---

## Usage

```bash
./run_training.sh --exp_name happo_iter5 --individualized_rewards
```

---

## Monitoring

Watch in TensorBoard:
- `train_episode_rewards/agent_0` - Agent 0 reward (NEW!)
- `train_episode_rewards/agent_1` - Agent 1 reward (NEW!)
- `train_episode_rewards/aver_rewards` - Mean (existing)
- `mapush/success_rate` - Key metric
- `rewards/goal_push_bonus` - New reward signal (NEW!)

**Freeloading detection:** If agent_0 >> agent_1 (or vice versa), one is freeloading.

---

## Expected Behavior

1. **Both agents engaged**: Individual rewards mean freeloading gets 0 reward
2. **Correct positioning**: Agents must be near box to get any positive reward
3. **Goal-directed pushing**: goal_push_bonus rewards pushing TOWARD target
4. **Balanced rewards**: Per-agent logging will show if both agents are contributing equally

---

## If This Doesn't Work (Iter6 Ideas)

1. **Increase goal_push_bonus_scale** - make direction more important
2. **Decrease contact_threshold** to 0.6m - stricter credit assignment
3. **Add proximity bonus** - explicit positive reward for being close (instead of penalty for being far)
4. **Remove approach penalty entirely** - rely only on positive rewards
