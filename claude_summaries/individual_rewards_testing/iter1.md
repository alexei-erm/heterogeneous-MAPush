# Iteration 1: Contact-Based Individualized Rewards

**Date:** 2025-12-14
**Status:** Training in progress

---

## Goal

Prevent freeloading behavior in HAPPO where one agent does nothing while the other pushes the box.

---

## Changes from Original Reward System

### Original (Shared Rewards)

All three rewards were broadcast equally to both agents:

```python
# reach_target_reward - when box reaches target
reward[self.finished_buf, :] += self.reach_target_reward_scale  # Both agents get +10

# target_reward - distance progress toward target
reward[:, :] += distance_reward.unsqueeze(1).repeat(1, self.num_agents)  # Both get same

# push_reward - when box is moving
reward[:, :] += push_reward.unsqueeze(1).repeat(1, self.num_agents)  # Both get same
```

### New (Individualized Rewards)

Rewards are weighted by agent proximity to box (contact-based attribution):

```python
contact_weight = clamp(1.0 - (agent_box_dist - threshold) / threshold, 0.0, 1.0)
# Agent within 0.8m → weight = 1.0 (full reward)
# Agent at 1.6m → weight = 0.0 (no reward)
```

**Modified rewards:**
1. `reach_target_reward` - Only agents near box at episode end get bonus
2. `target_reward` - Distance progress weighted by contact
3. `push_reward` - Only agents in contact get push reward

---

## Implementation

### Files Modified

| File | Change |
|------|--------|
| `mqe/envs/wrappers/go1_push_mid_wrapper.py` | Added `if self.individualized_rewards:` branches for 3 rewards |
| `mqe/envs/configs/go1_push_mid_config.py` | Added `individualized_rewards=False`, `contact_threshold=0.8` |
| `mqe/envs/utils.py` | Updated `custom_cfg()` to accept `individualized_rewards` param |
| `HARL/harl_mapush/train.py` | Added `--individualized_rewards` flag |
| `HARL/harl/envs/mapush/mapush_env.py` | Pass flag through to custom_cfg |
| `HARL/harl/envs/mapush/mapush_logger.py` | Added reward component logging to TensorBoard |

### Key Parameters

- `contact_threshold = 0.8` meters (agents within this distance get full reward)
- Linear decay beyond threshold (smooth, no discontinuity)

---

## Usage

```bash
# With individualized rewards (new)
./run_training.sh --exp_name happo_individual --individualized_rewards

# With shared rewards (original behavior)
./run_training.sh --exp_name happo_shared
```

---

## TensorBoard Metrics

Now logging:
- `mapush/success_rate` - Episode success rate
- `rewards/target_distance` - Box-to-target distance reward
- `rewards/approach_box` - Agent approach reward (already individual)
- `rewards/push` - Box movement reward
- `rewards/reach_target` - Episode completion bonus
- `rewards/ocb` - Optimal collaborative behavior (already individual)
- `rewards/collision_punishment` - Inter-agent collision (already individual)
- `rewards/exception_punishment` - Exception penalties
- `config/individualized_rewards` - Flag (0 or 1)

---

## Training Run

**Experiment:** `happo_individual` (seed-00001-2025-12-14-01-34-50)
**Config:**
- 500 parallel environments
- 100M total steps
- Individualized rewards: ON
- Contact threshold: 0.8m

---

## Expected Outcome

With individualized rewards:
- Both agents should learn to approach and push the box
- No freeloading (one agent idle while other works)
- Potentially slower initial learning (both must learn to contribute)
- Better long-term collaboration

---

## Issues Encountered

1. **Output buffering** - Fixed with `--no-capture-output` and `python -u`
2. **Separate task registration hanging** - Removed `go1push_mid_happo` task, used flag instead
3. **Logger not connected** - Fixed `set_envs()` call timing
4. **Logger early return bug** - Fixed indentation issue in `episode_log()`

---

## Next Steps (Iter 2)

If freeloading persists:
- Try smaller `contact_threshold` (0.5m) for stricter attribution
- Try exponential decay instead of linear
- Add velocity-based attribution (reward agents moving toward box)
- Consider true contact force detection from Isaac Gym
