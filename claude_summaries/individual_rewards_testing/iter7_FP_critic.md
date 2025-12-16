# Iteration 7: FP Critic Mode

**Date:** 2025-12-14
**Status:** Ready to train
**Based on:** Iter6 analysis - discovered critic was only seeing Agent 0's rewards

---

## Problem Discovered in Iter6

Iter6 had a fundamental bug: **EP (Environment Provided) state_type + individualized rewards = broken**

```python
# EP mode (line 453 in on_policy_base_runner.py)
self.critic_buffer.insert(
    share_obs[:, 0],
    rnn_states_critic,
    values,
    rewards[:, 0],  # ← Only Agent 0's reward!
    masks[:, 0],
    bad_masks,
)
```

**Result:**
- Critic trained only on Agent 0's rewards
- Agent 1 got garbage advantage estimates
- Value loss exploded: 0.01 → 0.25 (+170%)
- Both agents had nearly identical rewards (0.1% diff) despite "individualized" rewards

---

## Iter7 Fix: FP Mode

**Change in `HARL/harl_mapush/train.py`:**

```python
# Before (Iter6)
"state_type": "EP"

# After (Iter7)
"state_type": "FP" if use_individual else "EP"
```

**FP (Feature Pruned) mode:**
- Critic sees ALL agents' rewards: `rewards` instead of `rewards[:, 0]`
- Critic learns: V(state, agent_i) → expected return for agent i
- Each agent gets advantage estimates based on ITS OWN expected returns

---

## Expected Behavior

| Metric | Iter6 (EP) | Iter7 (FP) Expected |
|--------|------------|---------------------|
| Value loss | 0.01 → 0.25 (explode) | Should decrease/stabilize |
| Agent rewards | Identical (0.1% diff) | Should diverge meaningfully |
| Success rate | Peaked 1.3%, fell back | Should climb steadily |
| Entropy | Rising (policy collapse) | Should stabilize |

---

## What Stays the Same

All iter6 reward structure unchanged:
- engagement_bonus_scale: 0.0004
- cooperation_bonus_scale: 0.0002
- same_side_bonus_scale: 0.0004
- blocking_penalty_scale: -0.001
- goal_push_bonus_scale: 0.003
- directional_progress_scale: 0.003

---

## Monitoring

Watch for:
1. **Value loss** - should decrease over time (like MAPPO: 1.14 → 0.004)
2. **Agent reward difference** - should see >1% difference between agents
3. **Success rate** - should show steady upward trend, not sporadic spikes

---

## Usage

```bash
cd /home/gvlab/new-universal-MAPush/HARL
python harl_mapush/train.py --algo happo --exp_name happo_iter7_FP --individualized_rewards
```

---

## Potential Iter8 Changes (if Iter7 insufficient)

If FP alone doesn't solve learning:

1. **Lower entropy_coef: 0.01 → 0.001**
   - Reduce "stay random" pressure
   - Help policy converge instead of entropy explosion

2. **Stronger goal rewards (10x)**
   - goal_push_bonus_scale: 0.003 → 0.03
   - directional_progress_scale: 0.003 → 0.03
   - Give clearer gradient signal for goal-directed pushing

---

## Reference: MAPPO Baseline Comparison

| Metric | MAPPO (90% success) | HAPPO Iter6 (broken) |
|--------|---------------------|----------------------|
| Value loss | 1.14 → 0.004 (-95%) | 0.01 → 0.25 (+170%) |
| Entropy | 4.6 → 49.7 (rises, still learns) | 1.3 → 2.2 (rises) |
| Actor grad norm | 0.31 → 0.05 | 0.28 → 0.04 |
| Success | 90% | 1.3% peak |

Key insight: Entropy rising isn't the problem (MAPPO has it too). The VALUE LOSS diverging was the real issue, caused by EP mode + individual rewards mismatch.
