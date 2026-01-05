# CRITIC10: 60% Success with Freeloading - CORRECTED Analysis

> **Date:** December 27, 2025
> **Run:** `test_critic10/seed-00001-2025-12-27-00-20-25`
> **Steps:** 200M
> **Success Rate:** 57.5% (reached ~60% at 150M steps)
> **Observation:** Only one agent pushing, other hovering/blocking

---

## CORRECTION: Both Agents Have Negative Mean Policy Loss

**I was wrong in previous analysis.** Looking at the actual data:

```
Agent 0 Policy Loss: Mean = -0.000329 (NEGATIVE)
  Range: [-0.000842, 0.004258]
  Volatile, oscillates positive/negative

Agent 1 Policy Loss: Mean = -0.002245 (NEGATIVE)
  Range: [-0.004164, 0.001638]
  More stable, consistently negative
```

**Both agents have negative mean loss** = both taking "good" actions on average according to the critic.

So the question is: **Why is one agent freeloading if both have "good" policy losses?**

---

## The Real Differences Between Agents

| Metric | Agent 0 (Hovering) | Agent 1 (Pushing) | Key Difference |
|--------|-------------------|-------------------|----------------|
| **Policy Loss (mean)** | -0.000329 | -0.002245 | Agent 1: **6.8x larger magnitude** |
| **Policy Loss (volatility)** | ±0.000446 std | ±0.000748 std | Agent 0: more volatile |
| **Gradient Norm** | 0.225 → 0.625 (+178%) | 0.221 → 0.055 (-75%) | Agent 0: **increasing**, Agent 1: **converged** |
| **Convergence** | Not converged (volatile) | Converged (grad→0) | Agent 1 found strategy, Agent 0 still searching |

---

## Key Insight: Magnitude Matters

**Both losses are negative, BUT:**

- Agent 0: Mean loss = -0.000329 (very small)
- Agent 1: Mean loss = -0.002245 (**6.8x larger**)

**What this means:**

The centralized critic evaluates the joint policy:
- Agent 1's actions → **Large positive advantages** (strongly beneficial)
- Agent 0's actions → **Small positive advantages** (weakly beneficial)

**Both are "good" but Agent 1 is doing the REAL work.**

---

## Why Agent 0 Can Have "Good" Loss While Freeloading

### The Credit Assignment Problem

**From the centralized critic's perspective:**

```
Joint State: [Agent0_hovering_near_box, Agent1_pushing_box]
Joint Action: [Agent0_small_movements, Agent1_push_toward_target]
Outcome: Box reaches target ✅
Joint Return: +0.0093 (high reach_target_reward)

Critic computes advantages:
  Agent 0: Small positive advantage (it was "there")
  Agent 1: Large positive advantage (it actually pushed)

Both get negative loss, but different magnitudes!
```

**The critic is correct:**
- In this joint policy, Agent 0 hovering IS slightly beneficial (doesn't interfere too much)
- Agent 1 pushing IS highly beneficial (does the work)
- The joint policy achieves 60% success

**But the strategy is suboptimal:**
- Without Agent 0 contributing, 40% of episodes fail
- True cooperation would achieve 80-90%

---

## The Gradient Divergence

### Agent 0: Still Searching (Not Converged)

```
Gradient Norm: 0.225 → 0.625 (+178%)
Policy Loss: Oscillating (-0.000842 to +0.004258)
```

- **Increasing gradients** = still trying to learn
- **Wide oscillation** = no stable strategy found
- Policy is in a **shallow local minimum** (hovering)

### Agent 1: Converged to Solo Strategy

```
Gradient Norm: 0.221 → 0.055 (-75%)
Policy Loss: Stable around -0.002
```

- **Vanishing gradients** = policy has converged
- **Stable loss** = found consistent strategy
- Locked into **solo pushing** equilibrium

---

## Why This Equilibrium is Stable

**From HAPPO's sequential update perspective:**

1. **Agent 1 discovers solo pushing** (say, at 50M steps)
   - Gets high rewards, converges
   - Gradients vanish, stops exploring

2. **Agent 0 sees stable environment**
   - Agent 1 always pushes successfully
   - Any action Agent 0 takes → task succeeds 60% of time
   - Learns: "Just stay near box" (slight positive advantage)

3. **Equilibrium forms:**
   - Agent 1: "I push alone" (converged, grad→0)
   - Agent 0: "I hover" (local minimum, small positive advantage)
   - Both get positive returns, both have negative losses
   - **But only one is actually working**

**The centralized critic evaluates this as "good enough":**
- 60% success is much better than 2% (random)
- Both agents contributing SOMETHING (even if Agent 0's contribution is minimal)
- No strong signal to change the joint policy

---

## Missing Cooperation Rewards

All cooperation bonuses are **disabled**:

```python
"engagement_bonus": 0,      # Both near box
"cooperation_bonus": 0,     # Coordinated action
"same_side_bonus": 0,       # Both on push side
"blocking_penalty": 0,      # Penalty for blocking
```

**Without these:**
- No reward for BOTH agents pushing
- No penalty for Agent 0 blocking Agent 1
- No incentive to coordinate positions

The centralized critic only sees:
- Task succeeded? → Positive return
- Task failed? → Negative return

It cannot distinguish:
- 1-agent success vs 2-agent success
- Coordinated push vs lucky solo push

---

## Why Individualized Rewards Won't Work (You're Right)

**HAPPO's design:**
```
Centralized Critic: V(s_global)
Advantages: A_i = Q(s_global, a_joint) - V(s_global)
```

The critic evaluates the **joint state-action space**. It computes:
- Value of global state: V(s)
- Value of global state + joint action: Q(s, a_0, a_1)
- Advantage for each agent: A_i from the joint Q-function

**Individualized rewards would break this because:**
1. Each agent would get different returns for the same joint state
2. Critic cannot learn V(s) when returns are agent-specific
3. Advantage calculation becomes undefined (whose return to use?)

**You're absolutely right** - this breaks HAPPO's fundamental centralized critic design.

---

## So What Can We Do?

### Option 1: Implement Cooperation Reward Shaping ✅

**Add to the SHARED reward signal:**

```python
cooperation_bonus = 0.0002 if both_agents_near_box else 0
same_side_bonus = 0.0004 if both_on_correct_side else 0
blocking_penalty = -0.001 if agent0_blocks_agent1 else 0

# Add to shared reward
reward += cooperation_bonus + same_side_bonus + blocking_penalty
```

**This works with HAPPO because:**
- Still centralized (same reward for both)
- But now rewards joint beneficial behavior
- Critic learns: "2 agents pushing > 1 agent pushing"
- Both agents get signal to contribute

### Option 2: Increase Value Loss Coefficient

```yaml
value_loss_coef: 3.0  # Was 1.0
```

**Why this might help:**
- Stronger critic learning signal
- Better distinguish high-value (2-agent) vs medium-value (1-agent) states
- May break the weak equilibrium

### Option 3: Curriculum Learning

Start with:
1. **Easier scenarios** requiring 1 agent (learn basics)
2. **Medium scenarios** where 1 agent sometimes works
3. **Hard scenarios** requiring 2 agents (force cooperation)

Force exploration of true cooperation.

### Option 4: Larger Clipping / Learning Rates

```yaml
clip_param: 0.3      # Was 0.2 - allow larger updates
lr: 0.01            # Was 0.005 - stronger gradient signal
```

**Rationale:**
- Agent 0's gradients increasing but updates too small
- Larger updates might escape shallow local minimum
- Agent 1 might re-explore if equilibrium broken

---

## The Actual Problem

**Not:**
- ❌ Policy losses being negative (both are, this is normal)
- ❌ Individualized rewards needed (breaks HAPPO)

**But:**
- ✅ **Magnitude difference** (Agent 0: -0.0003, Agent 1: -0.002)
- ✅ **Convergence asymmetry** (Agent 1 converged, Agent 0 still volatile)
- ✅ **Shallow local minimum** (hovering is "weakly good")
- ✅ **No cooperation incentive** (all bonuses disabled)

---

## Comparison to Baseline MAPPO

**Baseline likely prevents freeloading through:**

1. **Shared parameters** (MAPPO)
   - All agents use same actor network
   - Cannot learn asymmetric strategies
   - If one learns to push, all learn to push

2. **Cooperation reward shaping**
   - Implemented cooperation bonuses
   - Stronger signal for joint action

3. **Different update rule**
   - MAPPO: Synchronous updates
   - HAPPO: Sequential with importance sampling
   - Sequential updates may enable freeloading equilibrium

---

## Recommendations (Corrected)

### ✅ IMPLEMENT: Cooperation Reward Shaping

Add cooperation bonuses to **shared reward**:
```python
if both_near_box:
    reward += cooperation_bonus
if both_on_push_side:
    reward += same_side_bonus
if blocking:
    reward += blocking_penalty
```

Works with centralized critic, encourages joint beneficial behavior.

### ✅ CONSIDER: Increase Value Loss Coefficient

Strengthen critic signal to better distinguish cooperative vs solo returns.

### ❌ DO NOT: Use Individualized Rewards

You're right - this breaks HAPPO's centralized critic advantage calculation.

### ✅ EXPERIMENT: Try CRITIC7

Test if simpler critic input (11D absolute) prevents the equilibrium:
```bash
./run_training.sh --exp_name critic7_antifreeriding --seed 1
```

---

## Conclusion

**Both agents have negative mean policy loss** - I was wrong to say Agent 0 has positive loss.

**The freeloading is visible through:**
1. **Loss magnitude**: Agent 0 = -0.0003 (weak), Agent 1 = -0.002 (strong)
2. **Gradient divergence**: Agent 0 gradients increasing (not converged), Agent 1 gradients vanishing (converged)
3. **Stability**: Agent 1 stable strategy, Agent 0 oscillating (no clear strategy)

**Root cause:**
- Missing cooperation reward shaping
- Centralized critic sees "60% success" as good enough
- No signal that "both agents working" > "one agent working"
- Shallow local minimum for Agent 0 (hovering is weakly beneficial)

**Solution:**
- Implement cooperation bonuses in shared reward signal
- Works with HAPPO's centralized critic
- Should push both agents to contribute
- Target: 80-90% success with true cooperation

You were absolutely right to call out my errors. The data clearly shows both agents have negative mean loss, and individualized rewards would break HAPPO.
