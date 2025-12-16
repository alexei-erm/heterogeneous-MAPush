# Iteration 8: Gated Shared Rewards

**Date:** 2025-12-15
**Status:** Ready to train
**Based on:** HAPPO + shared rewards showed freeloading (one agent runs away)

---

## Problem Discovered

HAPPO with shared rewards at 44M steps:
- Success rate: 0% → 10% (learning happening!)
- Episode reward: -6 → -24 (getting WORSE)
- Visual inspection: **One agent runs away while other pushes**

**Root cause:** With shared rewards, if Agent 0 succeeds, Agent 1 gets +10 reach_target for FREE.
Agent 1 learned: "Let Agent 0 do the work, I'll collect the reward."

---

## Iter8 Solution: Gated Shared Rewards

**Concept:** Multiply ALL shared rewards by `min(engagement_agent0, engagement_agent1)`

```python
gating_factor = min(engagement_all_agents)  # 0 to 1

# If ANY agent is far from box:
# - gating_factor → 0
# - ALL rewards → 0 for BOTH agents

# If BOTH agents near box:
# - gating_factor → 1
# - Full rewards for BOTH
```

**This creates peer pressure:** Agent 0 learns "I need Agent 1 nearby or we BOTH suffer"

---

## Implementation

**New flag:** `--shared_gated_rewards`

**Gating threshold:** 2.0 meters (configurable)

**Rewards gated:**
- `reach_target_reward` - success bonus
- `target_reward` (distance_to_target) - box progress
- `push_reward` - box movement
- `ocb_reward` - orientation control

**Rewards NOT gated:**
- `approach_reward` - individual penalty for being far (still applies)
- `collision_punishment` - safety (always applies)
- `exception_punishment` - errors (always applies)

---

## Expected Behavior

**Without gating (old):**
- Agent 1 runs away → still gets reward if Agent 0 succeeds
- Freeloading is optimal strategy

**With gating (iter8):**
- Agent 1 runs away → gating_factor=0 → BOTH get 0 reward
- Agent 1 learns staying near box is necessary
- No freeloading possible

---

## Monitoring

Watch for:
1. `rewards/gating_factor` - should climb from ~0.3 to ~0.8+ as both learn to stay engaged
2. `success_rate` - should climb without episode reward tanking
3. Episode rewards - should go UP as success increases (not down like before)

---

## Usage

```bash
cd /home/gvlab/new-universal-MAPush/HARL
python harl_mapush/train.py --algo happo --exp_name happo_iter8_gated --shared_gated_rewards
```

---

## Comparison

| Run | Flag | Behavior |
|-----|------|----------|
| MAPPO baseline | (none) | Works - 90% success |
| HAPPO + shared | (none) | Freeloading - one runs away |
| HAPPO + individual | `--individualized_rewards` | Broken critic (iter1-7) |
| **HAPPO + gated** | `--shared_gated_rewards` | Prevents freeloading |

---

## Why This Should Work

1. **Still shared rewards** - both agents get same reward, critic works correctly (EP mode)
2. **Cooperation enforced** - can't get reward unless BOTH engaged
3. **No competition** - not pitting agents against each other
4. **Simple mechanism** - just a multiplier on existing rewards

---

## Iter8 Fix: Reward Averaging for EP Critic

**Problem discovered during iter8 testing:**
- `approach_reward` and `ocb_reward` were still per-agent (each agent's distance to box)
- EP mode critic only sees `rewards[:, 0]` (Agent 0's reward)
- Agent 1's advantages computed from wrong reward signal

**Fix:** Average rewards across agents before returning from wrapper:
```python
# In go1_push_mid_wrapper.py, before return
if not self.individualized_rewards:
    mean_reward = reward.mean(dim=1, keepdim=True)
    reward = mean_reward.expand(-1, self.num_agents)
```

**Why this works:**
- Critic predicts team value V(s) = expected team return
- Team return = average of all agents' rewards
- Both agents now get correct advantage estimates
- Standard practice for cooperative MARL with shared critic

**After fix:**
- `train_episode_rewards/agent_0` == `train_episode_rewards/agent_1` (identical)
- Critic sees true team reward
- Gating still applied before averaging
