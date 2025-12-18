# Critic Fix #1: Aggressive Value Function Fitting

**Date:** December 18, 2025
**Branch:** critic-fixing
**Status:** Implemented, ready for testing

---

## Problem Statement

HAPPO's critic is showing instability during training:
- Value loss increases over first 40M steps despite fixes to global state
- High oscillations in loss curves
- Gradient norms increasing
- After value normalizer fix (Fix #0), still seeing critic struggle

**Root Cause:** HAPPO's sequential agent updates create a "moving target" problem for the centralized critic.

### Why HAPPO is Harder Than MAPPO

**HAPPO's Sequential Updates:**
```
Iteration N:
  1. Agent 0 updates policy → π₀ changes
  2. Agent 1 updates policy → π₁ changes
  3. Critic sees: V(s; π₀_new, π₁_new)

Iteration N+1:
  1. Agent 0 updates again → π₀ changes more
  2. Agent 1 updates again → π₁ changes more
  3. Critic sees: V(s; π₀_newer, π₁_newer)
```

**Problem:** Critic is learning to predict returns under a **constantly shifting joint policy**. By the time critic adapts to iteration N's policies, agents have already updated twice more in iteration N+1.

**MAPPO (for comparison):** All agents update synchronously → critic faces slower policy drift.

---

## The Fix: Aggressive Critic Training

**Idea:** Give the critic **way more training** to keep up with the fast-changing policies.

### Changes Made

**File:** `HARL/harl/configs/algos_cfgs/happo.yaml`

```yaml
algo:
  ppo_epoch: 5              # Actors: 5 epochs (unchanged)
  critic_epoch: 25          # Critic: 25 epochs (UP from 5) ⬆️
  value_loss_coef: 5.0      # Loss weight: 5.0 (UP from 1.0) ⬆️
```

**Effect:**
- **5x more critic training** per data collection
- **5x higher loss weight** → stronger value function fitting
- Actors still train normally (5 epochs)

---

## Why This Should Work

### 1. More Epochs = Better Tracking
With 25 epochs instead of 5:
- Critic has 5x more gradient steps to adapt
- Can better fit the complex value landscape
- Reduces lag behind policy updates

### 2. Higher Loss Weight = Stronger Signal
With `value_loss_coef: 5.0` instead of `1.0`:
- Critic optimization prioritizes value fitting
- Backprop gives stronger gradients to value network
- Faster convergence per epoch

### 3. Addressing the Time Asymmetry
**Before:**
- Actors: 5 epochs × 2 agents = 10 total policy updates
- Critic: 5 epochs = 5 value updates
- **Ratio:** 2:1 (policies update 2x more)

**After:**
- Actors: 5 epochs × 2 agents = 10 total policy updates
- Critic: 25 epochs = 25 value updates
- **Ratio:** 1:2.5 (critic updates 2.5x MORE than any single policy)

This rebalances the learning dynamics!

---

## Expected Results

### Metrics to Watch

**Good signs (what we want to see):**
1. ✅ **Critic loss decreases steadily** instead of increasing
2. ✅ **Lower oscillations** in loss curves
3. ✅ **Stable gradient norms** (not exploding)
4. ✅ **Value predictions track returns** more closely
5. ✅ **Advantages have reasonable std** (not growing unbounded)

**Potential issues (what to watch for):**
1. ⚠️ **Critic overfitting** - loss plateaus but eval performance drops
2. ⚠️ **Slower wall-clock time** - 5x more critic training takes time
3. ⚠️ **Policy degradation** - if critic dominates too much

### Success Criteria

**Minimum success:**
- Critic loss shows **decreasing trend** over 20M steps
- Oscillations reduced by **>50%**

**Full success:**
- Critic loss decreases steadily to 40M+ steps
- Success rate improves over training
- No signs of instability or divergence

---

## Implementation Details

### Code Changes
**Modified:** `HARL/harl/configs/algos_cfgs/happo.yaml`
- Line 85: `critic_epoch: 5` → `critic_epoch: 25`
- Line 98: `value_loss_coef: 1` → `value_loss_coef: 5.0`

**No other changes needed** - pure config adjustment!

### Training Impact
- **Sample efficiency:** Unchanged (same data collection)
- **Wall-clock time:** ~20-30% slower (more critic training)
- **Memory:** Unchanged (same batch sizes)
- **Convergence:** Should improve (better value estimates)

---

## Comparison to Other Fixes

| Fix | Complexity | Impact | Implementation | Risk |
|-----|-----------|--------|----------------|------|
| **Fix #1 (this)** | 🟢 Trivial | 🔴 High | Config only | 🟢 Low |
| Fix #2 (pre-training) | 🟡 Moderate | 🔴 High | Code changes | 🟢 Low |
| Fix #3 (slower actors) | 🟡 Moderate | 🟡 Medium | Code changes | 🟡 Medium |
| Fix #4 (target critic) | 🔴 High | 🟡 Medium | Major rewrite | 🔴 High |
| Fix #5 (sync batch) | 🔴 Very High | 🔴 High | Algorithm change | 🔴 High |

**Fix #1 is the clear winner for first attempt:** Minimal effort, high impact, low risk.

---

## Next Steps

### Testing Plan

1. **Run training with Fix #1**
   ```bash
   # Start training run
   python train_script.py --exp_name critic_fix1_epochs25_coef5
   ```

2. **Monitor for 10-20M steps**
   - Watch TensorBoard: `critic/value_loss`
   - Compare to previous runs (iter17)
   - Look for decreasing trend

3. **Evaluate Results**
   - If **good:** Continue to 40M+ steps
   - If **partial:** Add Fix #2 (pre-training)
   - If **bad:** Revert and try Fix #3 (slower actors)

### Rollback Instructions

If this makes things worse:

```yaml
# Revert to original values
critic_epoch: 5
value_loss_coef: 1.0
```

---

## Theoretical Justification

### From HAPPO Paper
The original HAPPO paper (Kuba et al., 2022) doesn't specify critic-actor epoch ratios, but:
- Uses centralized critic with sequential policy updates
- Acknowledges "moving target" challenge
- Recommends "sufficient value function training"

### From Related Work

**IPPO (Independent PPO):**
- Uses 1:1 actor-critic epoch ratio
- Works because agents train independently (no coordination)

**MAPPO (Multi-Agent PPO):**
- Uses 1:1 ratio but synchronized updates
- Less moving target problem than HAPPO

**HAPPO (Heterogeneous-Agent PPO):**
- **Should use asymmetric ratios** due to sequential updates
- Our hypothesis: 1:5 ratio (actor:critic) is appropriate

### Mathematical Intuition

Critic loss gradient:
```
∇_θ L_critic = ∇_θ (V_θ(s) - R)²
```

With sequential updates, the return R is a function of **changing policies**:
```
R(s,a; π₀, π₁) = r + γ V(s'; π₀', π₁')
```

Where π₀', π₁' are the **updated** policies (different from π₀, π₁).

**More critic epochs** means more chances to fit V to the current policy landscape before it shifts again.

---

## Related Fixes

**Prerequisites (already applied):**
- ✅ Fix #0: Value normalizer moved outside loop (Dec 18)
- ✅ Global state fixes: Coordinate frame + velocities (Dec 17)

**Future fixes (if needed):**
- Fix #2: Mini-batch critic pre-training
- Fix #3: Slower agent update cycles
- Fix #4: Target critic network
- Fix #5: Synchronized update batching

---

## References

- HAPPO paper: Kuba et al. (2022) "Trust Region Policy Optimisation in Multi-Agent Reinforcement Learning"
- MAPPO paper: Yu et al. (2021) "The Surprising Effectiveness of PPO in Cooperative Multi-Agent Games"
- PPO paper: Schulman et al. (2017) "Proximal Policy Optimization Algorithms"

---

## Changelog

**December 18, 2025:**
- Initial implementation of Fix #1
- `critic_epoch: 5 → 25`
- `value_loss_coef: 1.0 → 5.0`
- Ready for testing on critic-fixing branch
