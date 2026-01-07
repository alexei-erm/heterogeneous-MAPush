# CRITIC14: All Team Rewards (Centralized Critic Compatibility)

> **Date:** January 6, 2026
> **Philosophy:** All rewards must be team rewards for centralized critic
> **Goal:** Fix reward-critic mismatch that caused high variance in advantage estimation
> **Status:** IMPLEMENTED
> **Parent:** Derived from CRITIC13 v3

---

## The Problem

CRITIC13 (and all previous versions) had a **reward-critic mismatch**:

```
Centralized Critic: V(s) = E[team return]

But approach_to_box_reward was INDIVIDUAL:
  Agent 0 reward: approach_penalty_0 + shared_rewards
  Agent 1 reward: approach_penalty_1 + shared_rewards

Advantage computation:
  A_0 = r_0 + γV(s') - V(s)  ← r_0 is individual
  A_1 = r_1 + γV(s') - V(s)  ← r_1 is individual, V(s) is team!
```

The centralized critic estimates **team value**, but individual rewards create per-agent returns. The baseline doesn't match → **higher variance in advantage estimation**.

---

## The Fix

**Convert ALL rewards to team rewards.**

### Changes Made

| Reward | Before | After |
|--------|--------|-------|
| `approach_to_box_reward` | `reward[:, i]` (individual) | `reward[:, :]` (team: sum of both) |
| `collision_punishment` | `reward[:, i]` + `reward[:, j]` | `reward[:, :]` (explicit team) |

### Code Changes

**approach_to_box_reward** (go1_push_mid_wrapper.py:434-445):
```python
# BEFORE (individual):
for i in range(self.num_agents):
    distance_reward = (-(distance+0.5)**2) * scale
    reward[:, i] += distance_reward  # per-agent

# AFTER (team):
total_approach_penalty = torch.zeros(self.num_envs, device=self.device)
for i in range(self.num_agents):
    distance_penalty = (-(distance + 0.5)**2) * scale
    total_approach_penalty += distance_penalty
reward[:, :] += total_approach_penalty.unsqueeze(1).repeat(1, self.num_agents)  # team
```

**collision_punishment** (go1_push_mid_wrapper.py:447-455):
```python
# BEFORE (both agents, but separate assignments):
reward[:, i] += collision_punishment
reward[:, j] += collision_punishment

# AFTER (explicit team):
reward[:, :] += collision_punishment.unsqueeze(1).repeat(1, self.num_agents)
```

---

## All Rewards Now Team

| # | Reward | Scale | Type |
|---|--------|-------|------|
| 1 | `reach_target_reward` | 10 | TEAM |
| 2 | `approach_to_box_reward` | 0.00075 | **TEAM (FIXED)** |
| 3 | `push_reward` | 0.0015 | TEAM |
| 4 | `goal_push_bonus` | 0.01 | TEAM |
| 5 | `ocb_reward` | +0.01/-0.004 | TEAM |
| 6 | `proximity_penalty` | -0.002 | TEAM |
| 7 | `collision_punishment` | -0.0025 | **TEAM (FIXED)** |
| 8 | `exception_punishment` | -5 | TEAM |

**All 8 rewards are now team rewards.**

---

## Expected Impact

### Before (CRITIC13)
- Centralized V(s) estimates team value
- Individual approach reward causes per-agent return variance
- Advantage estimation has higher variance
- Critic struggles to predict value accurately

### After (CRITIC14)
- All rewards are team → all returns are identical per-agent
- V(s) perfectly matches the return structure
- Lower variance in advantage estimation
- Critic can learn more accurate value estimates

---

## Behavioral Change

### approach_to_box_reward

**Before:** Each agent penalized only for their own distance
- Agent far from box: only that agent penalized
- Freeloading possible (one stays far, doesn't affect other's reward)

**After:** Both agents share sum of penalties
- Agent far from box: BOTH agents penalized
- Mutual accountability (if partner drifts, you suffer too)

This actually STRENGTHENS anti-freeloading:
- Before: "I'm close, I'm fine"
- After: "We're both responsible for staying close"

---

## Version History

| Run Directory | Version | Key Change |
|---------------|---------|------------|
| TBD | v1 | All rewards converted to team (approach + collision) |

---

## Training Command

```bash
./run_training.sh --algo happo --env mapush --exp_name critic14 --use_concat_agent_observations_critic True --seed 7
```

---

## Comparison: CRITIC13 vs CRITIC14

| Aspect | CRITIC13 | CRITIC14 |
|--------|----------|----------|
| approach_to_box | Individual | Team (sum) |
| collision_punishment | Both (separate) | Team (explicit) |
| Critic compatibility | Mismatch | Perfect match |
| Advantage variance | Higher | Lower |
| Anti-freeloading | Individual penalty | Mutual penalty |

---

## If This Works

This fix should provide:
1. Lower variance in policy gradient updates
2. More stable training
3. Better critic predictions
4. Possibly faster convergence

## If This Fails

The team reward structure might:
1. Dilute individual responsibility (unlikely - mutual penalty is stronger)
2. Need scale adjustments for the doubled penalty magnitude

Fallback: Adjust approach_reward_scale from 0.00075 → 0.000375 to compensate for summing.
