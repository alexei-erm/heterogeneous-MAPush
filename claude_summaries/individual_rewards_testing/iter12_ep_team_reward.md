# Iteration 12: EP Mode + Team Reward (Correct HAPPO)

**Date:** 2025-12-16
**Status:** Running
**Command:** `python HARL/harl_mapush/train.py --algo happo --exp_name iter12_ep-mode_teamreward`

---

## Critical Fix: Proper HAPPO Implementation

This iteration fixes **two fundamental issues** that broke all previous HAPPO runs (iter1-11):

### Issue 1: FP Mode (Fixed in train.py)
- **Before:** `state_type: "FP"` when using individualized rewards
- **After:** `state_type: "EP"` always (hardcoded)
- **Why:** HAPPO requires single global critic, not per-agent value estimation

### Issue 2: Per-Agent Rewards (Fixed in wrapper)
- **Before:** Each agent received different rewards (approach, collision, OCB computed per-agent)
- **After:** All rewards summed into team reward, identical for all agents
- **Why:** HAPPO's credit assignment comes from sequential importance weighting, NOT from reward decomposition

---

## Changes Made

### 1. `HARL/harl_mapush/train.py` (line 91)
```python
# ALWAYS EP mode - HAPPO uses single global critic
"state_type": "EP",
```

### 2. `mqe/envs/wrappers/go1_push_mid_wrapper.py` (end of step())
```python
# HAPPO/MAPPO TEAM REWARD: Sum per-agent rewards into team reward
# All agents receive IDENTICAL team reward for proper CTDE training
team_reward = reward.sum(dim=1, keepdim=True)  # (num_envs, 1)
reward = team_reward.expand(-1, self.num_agents)  # (num_envs, num_agents)
```

---

## Configuration

| Parameter | Value | Notes |
|-----------|-------|-------|
| Algorithm | HAPPO | Separate actors, shared critic |
| state_type | EP | Single global critic |
| lr | 0.0005 | Same as MAPPO |
| critic_lr | 0.0005 | Same as MAPPO |
| Rewards | Team (summed) | Identical for all agents |
| share_param | False | Separate actor networks |

---

## Why This Is Correct (Per HAPPO Theory)

From `HAPPO_Credit_Assignment_Explained.md`:

> **CRITICAL INSIGHT**: In HAPPO, credit assignment does NOT come from giving individual rewards to actors. It comes from the **sequential update mechanism with importance weighting**.

### The Correct Flow:
1. **Team reward** computed (sum of all components)
2. **All agents** receive identical team reward
3. **Single critic** learns V(s) predicting team return
4. **Same advantages** computed for all agents: `A(s,a) = R - V(s)`
5. **Sequential update** with importance weighting handles credit assignment:
   - Agent 1 updates with full advantage
   - Agent 2 updates with reweighted advantage: `M₂ = [π¹_new/π¹_old] × A`

### What Was Wrong Before (iter9-11):
- FP mode → per-agent value estimation → critic divergence
- Per-agent rewards → inconsistent advantages → broken sequential update
- Critic saw only agent 0's reward → not representative of team performance

---

## Expected Results

| Metric | iter9-11 (broken) | iter12 (correct) |
|--------|-------------------|------------------|
| Critic value_loss | Diverging (0.01→0.27) | Should converge (like MAPPO) |
| Success rate | 15-23% plateau | Should improve significantly |
| Training stability | Oscillating | Stable |

---

## Monitoring

**Critical:** Watch `critic/value_loss`
- Should **decrease** over training (0.5 → 0.03 like MAPPO)
- If still increasing, something else is wrong

**Success metric:** Should approach MAPPO's ~90% success rate since the only difference is separate vs shared actor networks.

---

## Comparison: MAPPO vs HAPPO (iter12)

| Aspect | MAPPO | HAPPO iter12 |
|--------|-------|--------------|
| Actor networks | 1 shared | 2 separate |
| Critic | Single global | Single global |
| Rewards | Team (sum) | Team (sum) |
| Advantages | Same for all | Same for all |
| Update | Simultaneous | Sequential with importance weighting |
| Credit assignment | Implicit | Via sequential update |

The ONLY difference now is separate actor networks + sequential update. This should allow heterogeneous policies while maintaining stable training.

---

## References

- `claude_summaries/HAPPO_Critic_Detailed.md` - Critic architecture
- `claude_summaries/HAPPO_Credit_Assignment_Explained.md` - Why team rewards are required
