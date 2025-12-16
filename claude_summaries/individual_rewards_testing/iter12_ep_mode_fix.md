# Iteration 12: EP Mode Fix (Correct HAPPO Implementation)

**Date:** 2025-12-16
**Status:** Ready to train
**Based on:** Discovery that FP mode causes critic divergence; HAPPO documentation requires EP mode

---

## Critical Discovery

All previous HAPPO iterations (iter9-11) used **FP mode** which is incorrect for HAPPO.

From `HAPPO_Critic_Detailed.md`:
> HAPPO uses a **SINGLE CENTRALIZED GLOBAL VALUE NETWORK** (not per-agent critics)

FP mode was causing:
- Critic value_loss to **increase** (0.014 → 0.27) instead of converge
- Per-agent value estimation instability
- Different advantages for each agent (breaking HAPPO's design)

---

## What Changed

### 1. train.py - Always EP Mode
```python
# OLD (broken):
"state_type": "FP" if use_individual else "EP"

# NEW (correct):
"state_type": "EP",  # ALWAYS EP - HAPPO uses single global critic
```

### 2. go1_push_mid_wrapper.py - Team Reward for Critic
```python
# When individualized_rewards=True, average rewards for stable critic
if self.individualized_rewards:
    team_reward = reward.mean(dim=1, keepdim=True)
    reward = team_reward.expand(-1, self.num_agents)
```

---

## How HAPPO Now Works

| Component | Configuration |
|-----------|---------------|
| Critic | Single global V(s), sees team reward |
| Advantages | Same for all agents (from global value function) |
| Actors | Separate networks, different observations |
| Credit assignment | Via HAPPO's sequential importance weighting |

**Key insight:** HAPPO's sequential update mechanism handles credit assignment, NOT per-agent rewards. The contact-weighted individual rewards are averaged to give the critic a stable training signal.

---

## Expected Behavior

1. **Critic value_loss should CONVERGE** (like MAPPO: 0.53 → 0.03)
2. **Stable training** - no more divergence
3. **Separate actor networks** may still specialize through:
   - Different observations (agent-local coordinates)
   - HAPPO's sequential update with importance weighting
   - Natural interaction dynamics

---

## Usage

```bash
cd /home/gvlab/new-universal-MAPush
python HARL/harl_mapush/train.py --algo happo --exp_name happo_iter12_ep --individualized_rewards
```

Note: `--individualized_rewards` now means contact-weighted rewards are computed but then averaged for the critic. This affects reward shaping/logging but critic sees team reward.

---

## Comparison with Previous Iterations

| Aspect | Iter9-11 (FP, broken) | Iter12 (EP, correct) |
|--------|----------------------|---------------------|
| state_type | FP | EP |
| Critic input | Per-agent states | Global state |
| Rewards to critic | Per-agent | Team average |
| Advantages | Different per agent | Same for all |
| Value loss | Diverging (0.01→0.27) | Should converge |
| Expected success | 15-23% plateau | Should improve |

---

## Monitoring

**Critical metric:** `critic/value_loss`
- Should decrease over training (like MAPPO)
- If still increasing, something else is wrong

**Watch for:**
- Whether agents naturally specialize despite same advantages
- If freeloading persists, may need different approach (not reward-based)

---

## If Freeloading Still Occurs

HAPPO with EP mode may result in both agents learning similar behavior (like MAPPO's shared policy). If this happens and both agents push, that's actually fine!

If one agent still freeloads, potential next steps:
1. Asymmetric observations (give agents different info)
2. Role-based reward bonuses
3. Communication mechanism between agents
4. Different algorithm entirely (QMIX, COMA, etc.)
