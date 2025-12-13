# Iteration 2: Individualize Only Push Reward

**Date:** 2025-12-14
**Status:** Ready to train

---

## Why Iter1 Failed

Iter1 individualized ALL three shared rewards (reach_target, target_distance, push). This was too aggressive:
- Agents got near-zero reward when far from box
- But they need rewards to LEARN to approach in the first place
- Conflicting signals: approach_reward says "get close", but other rewards punish being far
- Result: train_episode_rewards falling rapidly

---

## Iter2 Strategy: Minimal Intervention

**Only individualize push_reward** - the most direct anti-freeloading signal.

| Reward | Iter1 | Iter2 | Rationale |
|--------|-------|-------|-----------|
| `reach_target_reward` | Individual | **SHARED** | Both agents celebrate success together |
| `target_reward` (distance) | Individual | **SHARED** | Both benefit from progress toward goal |
| `push_reward` | Individual | **INDIVIDUAL** | Only agents near box get credit for movement |
| `approach_reward` | Individual | Individual | (unchanged - already per-agent) |
| `ocb_reward` | Individual | Individual | (unchanged - already per-agent) |
| `collision_punishment` | Individual | Individual | (unchanged - already per-agent) |

---

## Logic

**Key insight:** If box is moving and agent is far away, that agent is NOT pushing.

```python
if self.individualized_rewards:
    # Only agents near box get push reward
    for i in range(self.num_agents):
        agent_box_dist = torch.norm(box_pos - base_pos[:, i, :], dim=1)
        contact_weight = clamp(1.0 - (dist - 0.8) / 0.8, 0.0, 1.0)
        reward[box_moving, i] += push_reward_scale * contact_weight[box_moving]
else:
    # Original: both agents get push reward
    reward[:, :] += push_reward
```

---

## Expected Behavior

1. **Early training:** Agents learn to approach box (shared target_distance reward)
2. **Mid training:** Agents near box get push_reward, freeloaders don't
3. **Late training:** Both agents learn to stay near box and push together

The shared rewards provide stable learning signal, while individualized push_reward prevents freeloading.

---

## Changes from Iter1

```python
# reach_target_reward - REVERTED to shared
reward[self.finished_buf, :] += self.reach_target_reward_scale

# target_reward (distance) - REVERTED to shared
reward[:, :] += distance_reward.unsqueeze(1).repeat(1, self.num_agents)

# push_reward - KEPT individual (same as iter1)
# Only agents within contact_threshold get credit when box moves
```

---

## Usage

```bash
# Same command - flag still controls push_reward individualization
./run_training.sh --exp_name happo_iter2 --individualized_rewards
```

---

## Parameters

- `contact_threshold = 0.8m` (unchanged from iter1)
- `push_reward_scale = 0.0015` (unchanged)

---

## Monitoring

Watch in TensorBoard:
- `train_episode_rewards` - Should NOT fall rapidly like iter1
- `mapush/success_rate` - Should increase over time
- `rewards/push` - Individual push reward signal

---

## If This Doesn't Work (Iter3 Ideas)

1. **Increase contact_threshold** to 1.2m (more forgiving)
2. **Add minimum floor** to push_reward weight (e.g., min 0.2)
3. **Binary push attribution** - if either agent is close, both get reward; if neither is close, no reward
4. **Velocity-based** - reward agents moving toward box, not just near it
