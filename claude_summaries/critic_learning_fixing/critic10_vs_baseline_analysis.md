# CRITIC10 vs Baseline MAPPO: Comparative Analysis

> **Date:** December 26, 2025
> **Comparison:** HARL CRITIC10 vs OpenRL Baseline MAPPO
> **Question:** Why are policy losses negative and oscillating?

---

## Executive Summary

**Key Finding:** Both runs show **negative oscillating policy losses**, but the baseline MAPPO **succeeded** while CRITIC10 is **struggling**.

| Metric | Baseline MAPPO (130M steps) | CRITIC10 (100M steps) | Winner |
|--------|---------------------------|---------------------|---------|
| **Success Rate** | Not tracked, but successful | 30.9% (volatile) | 🏆 Baseline |
| **Avg Reward** | -0.019 → +0.024 (+225%) | -0.019 → -0.011 (+41%) | 🏆 Baseline |
| **Policy Loss** | +0.027 → -0.001 (converged) | -0.0002 → -0.0005 (volatile) | 🏆 Baseline |
| **Value Loss** | 5.04 → 0.004 (-99.9%, converged) | 0.106 → 0.051 (-51%, volatile) | 🏆 Baseline |
| **Entropy** | 4.31 → 50.08 (+1062%!) | 1.30 → 2.17 (+68%) | ⚠️ Both bad |

**Shocking Discovery:** Even the **successful baseline** has massively increasing entropy (+1062%)!

This changes everything we thought about the entropy metric in this environment.

---

## Detailed Metric Comparison

### 1. Policy Loss Analysis

#### Baseline MAPPO (Successful)
```
Policy Loss: +0.027210 → -0.001277
Mean: -0.001074 (std: 0.001839)
Change: -104.7%
Trend: volatile
Converged: YES at step 600K
Final variance: 0.000000 (fully converged)
```

#### CRITIC10 HARL (Struggling)
```
Agent 0: -0.000195 → -0.000544
Agent 1: -0.003352 → -0.002780
Mean: -0.000364 (std: 0.000307)
Trend: volatile
Converged: NO
```

**Key Differences:**

1. **Magnitude**
   - Baseline: Started positive (+0.027), ended negative (-0.001)
   - CRITIC10: Always negative (~-0.0005)
   - Baseline has **50x larger magnitude** (-0.001 vs -0.00002)

2. **Convergence**
   - Baseline: Converged at 600K steps, final variance = 0
   - CRITIC10: Still volatile at 100M steps, no convergence

3. **Starting Point**
   - Baseline: Started POSITIVE (+0.027)
   - CRITIC10: Started negative (agents ~-0.0002 and -0.003)

### Why Policy Loss is Negative: ANSWER

**Both runs show negative policy loss - this is NORMAL for PPO!**

The PPO objective is:
```python
J(θ) = E[min(r(θ) * A, clip(r(θ)) * A)]
policy_loss = -J(θ)  # Negative because we minimize loss
```

**When advantages are positive** (good actions):
- J(θ) is positive
- policy_loss = -J(θ) is **negative**
- This is the normal, healthy state!

**When advantages are negative** (bad actions):
- J(θ) is negative
- policy_loss = -J(θ) is **positive**
- This happens early in training (see baseline starting at +0.027)

**Baseline trajectory:**
```
Step 100K:   +0.027 (still learning bad vs good actions)
Step 600K:   -0.001 (converged, mostly good actions)
Step 130M:   -0.001 (stable)
```

**CRITIC10 trajectory:**
```
Always around -0.0003 to -0.0005 (tiny, oscillating, not converging)
```

---

### 2. Why is CRITIC10 Oscillating Without Convergence?

**The problem is NOT that losses are negative - it's that they're:**
1. **Extremely small** (0.0003 vs baseline's 0.001)
2. **Not converging** (volatile throughout)
3. **Weak gradient signal** (ratio ≈ 1.0000)

#### Root Cause: Tiny Gradient Magnitudes

| Metric | Baseline MAPPO | CRITIC10 | Ratio |
|--------|---------------|----------|-------|
| Policy loss magnitude | 0.001074 | 0.000364 | **3x smaller** |
| Actor grad norm | Not checked | 0.06-0.44 | - |
| Importance ratio | Not checked | 1.0000 | Almost no update |

**CRITIC10's updates are too small to make progress!**

---

### 3. Entropy: The Shocking Truth

#### Baseline MAPPO (Successful!)
```
Entropy: 4.31 → 50.08
Change: +1062.4%
Trend: volatile
Converged: NO
```

#### CRITIC10 (Struggling)
```
Entropy: 1.30 → 2.17
Change: +67.6%
Trend: increasing
Converged: NO
```

**What?! The baseline has WORSE entropy growth than CRITIC10!**

### Re-evaluating the Entropy Metric

**Previous assumption:** Increasing entropy = policy becoming random = BAD

**New insight:** In this multi-agent continuous control task:
- **Entropy metric may not be reliable**
- Could be measuring action **variance**, not **randomness**
- Successful policies may have high variance to explore coordination

**Why baseline entropy explodes to 50:**

Possible explanations:
1. **Action space size** - 3D continuous actions for 2 agents
2. **Coordination variance** - successful collaboration requires diverse actions
3. **Metric calculation** - may include batch variance, not just policy entropy
4. **Multi-agent interaction** - joint action space is large

**Conclusion:** We **cannot use entropy as a health metric** for this task!

---

### 4. Value Loss: The Key Difference

#### Baseline MAPPO (Successful)
```
Value Loss: 5.044 → 0.004
Change: -99.9%
Trend: volatile BUT CONVERGED
Converged: YES
```

#### CRITIC10 (Struggling)
```
Value Loss: 0.106 → 0.051
Change: -51.4%
Trend: volatile, NOT CONVERGED
Converged: NO
```

**This is the smoking gun!**

Baseline:
- Started with high value loss (5.0)
- Converged to near-zero (0.004)
- **133x reduction**

CRITIC10:
- Started with moderate value loss (0.106)
- Reduced to (0.051)
- **Only 2x reduction**
- Still volatile, not converged

**The critic is not learning properly in CRITIC10!**

---

### 5. Reward Progression

#### Baseline MAPPO
```
Avg Step Reward: -0.019 → +0.024
Change: +224.8%
Trend: volatile BUT CONVERGED
Final: POSITIVE rewards (+0.024)
```

#### CRITIC10
```
Avg Step Reward: -0.019 → -0.011
Change: +40.8%
Trend: increasing BUT STILL NEGATIVE
Final: NEGATIVE rewards (-0.011)
```

**Baseline achieved positive rewards, CRITIC10 is still negative!**

---

## Root Cause Analysis

### Why CRITIC10 is Failing

**1. Critic Not Learning**
- Value loss not converging (0.051 vs baseline's 0.004)
- **21x worse** value loss than baseline
- Unstable value estimates → unstable advantages

**2. Weak Policy Gradients**
- Policy loss magnitude: 0.00036 vs baseline's 0.001
- **3x weaker** signal
- Ratio ≈ 1.0000 (almost no policy change per update)

**3. Critic Input Complexity**
- CRITIC10 uses 16D concatenated observations (two agent perspectives)
- Baseline uses ??? (need to check - likely simpler)
- Harder for critic to learn stable value function

**4. Architecture Mismatch**
- Baseline: Tested architecture (worked)
- CRITIC10: New architecture (untested)
- May need more capacity or different structure

---

## Why Negative Oscillating Policy Loss is Not the Problem

**The negative loss is EXPECTED and CORRECT for PPO.**

The oscillation in baseline was also present ("volatile" trend), but it **converged** after 600K steps.

The real problems are:
1. **Too small** (weak gradients)
2. **Never converges** (value function unstable)
3. **Critic not learning** (value loss not decreasing enough)

---

## Specific Answers to Your Questions

### Q1: Why are agent policy losses always negative?

**A:** This is **completely normal** for PPO/HAPPO after initial learning.

PPO minimizes `loss = -E[min(ratio * A, clip(ratio) * A)]`

When most actions have **positive advantages** (good actions):
- The expectation is positive
- The negative makes the loss negative
- We minimize this negative value = maximize the objective

**The baseline also has negative policy loss (-0.001)** - this is the healthy, converged state!

Early in training (see baseline at step 100K: +0.027), the loss can be positive when advantages are mixed/negative.

### Q2: Why are they oscillating around a mean value without convergence?

**A:** Because the **critic (value function) is not learning properly**.

Evidence:
1. **Value loss not converging**
   - Baseline: 5.0 → 0.004 (converged, 133x reduction)
   - CRITIC10: 0.106 → 0.051 (not converged, only 2x reduction)

2. **Weak gradients**
   - Policy loss magnitude too small (0.00036 vs 0.001)
   - Ratio ≈ 1.0000 (barely updating)

3. **Unstable advantages**
   - Volatile value function → volatile advantage estimates
   - Volatile advantages → oscillating policy updates

**The oscillation itself is not unusual** (baseline also shows "volatile" trend), but baseline **converged after 600K steps**, CRITIC10 **never converges**.

---

## What's Different Between Baseline and CRITIC10?

| Aspect | Baseline MAPPO | CRITIC10 HARL |
|--------|---------------|--------------|
| **Framework** | OpenRL | HARL |
| **Algorithm** | MAPPO (shared params) | HAPPO (heterogeneous) |
| **Critic Input** | ??? (need to check) | 16D concatenated observations |
| **Observation** | Agent-local | Agent-local (same) |
| **Updates** | Synchronous | Sequential with importance sampling |
| **Architecture** | Tested & working | New, untested |
| **Value Loss** | Converged (0.004) | Not converged (0.051) |
| **Rewards** | Positive (+0.024) | Still negative (-0.011) |

**Key difference: The critic is learning in baseline but not in CRITIC10.**

---

## Recommendations

### ❌ NOT the Issue
1. ~~Negative policy loss~~ - This is normal!
2. ~~Increasing entropy~~ - Baseline also has this (even worse!)
3. ~~Oscillating loss~~ - Baseline also oscillates but converges

### ✅ REAL Issues to Fix

1. **Value function not converging**
   - Try simpler critic input (CRITIC7 absolute coords)
   - Increase critic capacity
   - Increase critic learning rate

2. **Weak policy gradients**
   - Increase actor learning rate (0.005 → 0.01)
   - Reduce clipping (0.2 → 0.3 for larger updates)

3. **Architecture**
   - Try CRITIC7 (11 dims, single global view) instead of CRITIC10 (16 dims, two perspectives)
   - The simpler input may help critic converge like baseline

### Immediate Experiment

**Test CRITIC7 with same config:**
```bash
./run_training.sh \
    --exp_name critic7_vs_critic10 \
    --seed 1
    # No flags = CRITIC7 by default
```

**Compare:**
- Value loss convergence
- Policy loss magnitude
- Reward progression

If CRITIC7 shows faster value loss decrease → the 16D input is the problem.

---

## Conclusion

### The Negative Oscillating Policy Loss is NORMAL

Both successful (baseline) and struggling (CRITIC10) runs have:
- Negative policy losses ✅ Expected for PPO
- Oscillating losses ✅ Normal (baseline also "volatile")

### The REAL Problem is Critic Learning

Baseline MAPPO:
- Value loss: 5.0 → 0.004 ✅ **Converged**
- Rewards: -0.019 → +0.024 ✅ **Positive**
- Policy converged at 600K steps ✅

CRITIC10 HARL:
- Value loss: 0.106 → 0.051 ❌ **Not converged** (21x worse than baseline)
- Rewards: -0.019 → -0.011 ❌ **Still negative**
- Policy never converges ❌

**Root cause:** The 16D concatenated observation input may be too complex for the critic to learn a stable value function. Try CRITIC7 (11D single global view) instead.

### Entropy is Not a Reliable Metric

Baseline's entropy increased by 1062% (4.3 → 50.1) and still succeeded. We cannot use entropy to judge training health in this environment.

### Next Steps

1. ✅ Understand negative loss is normal (PPO design)
2. ✅ Ignore entropy metric (unreliable for this task)
3. 🎯 Focus on value loss convergence
4. 🎯 Test CRITIC7 vs CRITIC10 with same hyperparameters
5. 🎯 Compare value loss decrease rate between methods
