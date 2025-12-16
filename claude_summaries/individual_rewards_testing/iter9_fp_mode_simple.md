# Iteration 9: FP Mode with Simple Individual Rewards

**Date:** 2025-12-15
**Status:** Ready to train
**Based on:** Iter8 failed - reward averaging diluted individual accountability

---

## Problems with Iter8

1. **Reward averaging killed individual accountability**
   - Agent 1 walks away → gets -100 penalty
   - Agent 0 stays close → gets -6 penalty
   - Average = -53 for BOTH agents
   - Agent 1 doesn't feel full pain of its bad decision

2. **EP mode gives same advantages to both actors**
   - Both agents trained with identical advantage estimates
   - No way to learn different behaviors

3. **Gating caused instability**
   - Value loss oscillated wildly (0.03 → 0.16 → 0.05)
   - Non-stationary reward landscape

---

## Iter9 Solution: FP Mode with Simple Rewards

**Key insight:** Use FP (Feature Pruned) mode so each actor gets advantages computed from ITS OWN rewards.

### FP Mode Flow:
1. Critic buffer stores per-agent rewards: `(steps, envs, agents, 1)`
2. Returns computed per-agent from individual rewards
3. Advantages computed per-agent: `returns[:,:,i] - values[:,:,i]`
4. Each actor trains with `advantages[:, :, agent_id]`

**This gives individual accountability** - if Agent 1 walks away, Agent 1's advantages reflect that.

---

## Reward Structure (Simplified)

Back to MAPPO's original rewards, but with contact-based individualization:

| Reward | Type | Description |
|--------|------|-------------|
| `approach_reward` | Individual | Penalty for distance to box (per agent) |
| `push_reward` | **Individual** | Contact-weighted: only agents near box get credit |
| `reach_target` | **Individual** | Contact-weighted: only agents near box get success bonus |
| `target_distance` | Shared | Box progress toward goal (team reward) |
| `ocb_reward` | Individual | Orientation reward (per agent) |
| `collision_punishment` | Individual | Both agents penalized equally |

### Contact Weighting:
```python
contact_weight = clamp(1.0 - (distance - threshold) / threshold, 0, 1)
# At threshold distance: weight = 1.0 (full credit)
# At 2x threshold: weight = 0.0 (no credit)
```

---

## Changes Made

### 1. Removed reward averaging (wrapper)
```python
# REMOVED:
# if not self.individualized_rewards:
#     mean_reward = reward.mean(dim=1, keepdim=True)
#     reward = mean_reward.expand(-1, self.num_agents)
```

### 2. Removed iter6 complex rewards
- No engagement_bonus, cooperation_bonus, same_side_bonus, blocking_penalty
- No velocity-based goal_push_bonus
- Just the original MAPPO rewards

### 3. Made push_reward contact-weighted
```python
if self.individualized_rewards:
    for i in range(self.num_agents):
        contact_weight = clamp(1.0 - (dist - threshold) / threshold, 0, 1)
        push_reward = scale * box_moving * contact_weight
        reward[:, i] += push_reward
```

### 4. reach_target already contact-weighted (from iter5)

---

## Why This Should Work

1. **FP mode = individual accountability**
   - Each agent's critic predicts value from its own rewards
   - Each agent's advantages reflect its own actions
   - Walking away → bad advantages → policy learns to stay

2. **Simple reward structure**
   - Same rewards that work for MAPPO
   - Just individualized where needed (push, reach_target)

3. **Stable training**
   - No gating causing non-stationary rewards
   - No complex bonuses to tune

---

## Usage

```bash
cd /home/gvlab/new-universal-MAPush
python HARL/harl_mapush/train.py --algo happo --exp_name happo_iter9_fp --individualized_rewards
```

---

## Monitoring

Watch for:
1. `train_episode_rewards/agent_0` vs `agent_1` - should be DIFFERENT (not identical)
2. `critic/value_loss` - should decrease steadily (not oscillate)
3. `mapush/success_rate` - should climb
4. Both agents staying near box in visual inspection

---

## Comparison

| Iter | Mode | Rewards | Problem |
|------|------|---------|---------|
| 1-6 | EP | Complex individual | Critic only saw agent 0's rewards |
| 7 | FP | Complex individual | Rewards too complex, symmetric |
| 8 | EP | Shared + gated | Averaging killed accountability, gating unstable |
| **9** | **FP** | **Simple + contact-weighted** | Should work |
