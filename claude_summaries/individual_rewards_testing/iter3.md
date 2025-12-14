# Iteration 3: Idle Penalty (Symmetric Punishment)

**Date:** 2025-12-14
**Status:** Ready to train

---

## Why Iter1 & Iter2 Failed

**Iter1** (all 3 rewards individualized): Too aggressive, agents got near-zero reward when far from box, destabilized learning.

**Iter2** (only push_reward individualized): Still failed. The problem: push_reward is sparse and noisy. Box only moves sometimes, and attributing it to "nearby" agents creates unstable credit assignment.

**Root cause:** Individualizing positive rewards creates learning instability. Agents need consistent positive signal to learn.

---

## Iter3 Strategy: Symmetric Penalty

**Keep ALL rewards shared.** Instead, add a small **idle penalty** when agent is far from box AND box isn't moving.

| Reward | Iter2 | Iter3 | Rationale |
|--------|-------|-------|-----------|
| `reach_target_reward` | SHARED | SHARED | Both agents celebrate success |
| `target_reward` (distance) | SHARED | SHARED | Both benefit from progress |
| `push_reward` | INDIVIDUAL | **SHARED** | Reverted - shared is more stable |
| `approach_reward` | Individual | Individual | (already per-agent) |
| `ocb_reward` | Individual | Individual | (already per-agent) |
| `collision_punishment` | Individual | Individual | (already per-agent) |
| **NEW: idle_penalty** | N/A | **Individual** | Punish agents far from box when stationary |

---

## Logic

```python
# Iter3: Idle penalty - punish agents far from box when box is NOT moving
if self.individualized_rewards and self.idle_penalty_scale != 0:
    box_stationary = ~box_moving  # box velocity < 0.1

    for i in range(self.num_agents):
        agent_box_dist = torch.norm(box_pos - base_pos[:, i, :], dim=1)
        is_far = agent_box_dist > self.idle_distance_threshold  # default 1.0m
        should_penalize = box_stationary & is_far
        reward[should_penalize, i] += self.idle_penalty_scale  # default -0.001
```

---

## Key Insight

**Don't mess with positive rewards** - they provide stable learning signal.

**Add a small nudge** to discourage freeloading:
- If box is moving → no penalty (someone is pushing, all good)
- If box is stationary AND agent is far → small penalty

This is gentler than iter1/iter2 because:
1. Shared rewards remain stable
2. Penalty is small (-0.001) vs reward scales (0.0015-0.01)
3. Penalty only applies when agent is clearly not contributing

---

## Parameters

- `idle_penalty_scale = -0.001` (small negative, configurable)
- `idle_distance_threshold = 1.0m` (beyond this = "far" from box)
- `individualized_rewards = True` still enables this

---

## Usage

```bash
# Same flag - now controls idle penalty instead of push individualization
./run_training.sh --exp_name happo_iter3 --individualized_rewards
```

---

## Monitoring

Watch in TensorBoard:
- `train_episode_rewards` - Should NOT fall rapidly
- `mapush/success_rate` - Should increase over time
- `rewards/idle_penalty` - New metric (should be small negative)
- `rewards/push` - Now shared again

---

## If This Doesn't Work (Iter4 Ideas)

1. **Increase penalty** to -0.005 (stronger nudge)
2. **Decrease idle_distance_threshold** to 0.8m (stricter)
3. **Boost approach_reward_scale** instead (Option B from original proposal)
4. **Mutual proximity bonus** - both agents must be near box to get full reward (Option C)
