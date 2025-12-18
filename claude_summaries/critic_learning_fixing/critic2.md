# Critic Fix #2: Mini-Batch Critic Pre-Training

**Date:** December 18, 2025
**Branch:** critic-fixing
**Status:** Implemented, ready for testing
**Previous Fix:** Fix #1 (critic_epoch: 25, value_loss_coef: 5.0) - **FAILED** (loss still increasing)

---

## Why Fix #1 Failed

**Fix #1 approach:** Train critic for 25 epochs instead of 5.

**Result:** Critic loss still increased.

**Diagnosis:** More training on a **moving target** doesn't help if the target moves faster than the critic can adapt. HAPPO's sequential agent updates shift the joint policy landscape too rapidly - by the time the critic fits iteration N's policies, agents have already updated multiple times.

**Key insight:** The problem isn't training *amount*, it's training *timing*.

---

## Fix #2: Pre-Training Before Policy Shifts

### The Idea

Instead of just doing more epochs *after* data collection, **stabilize the critic BEFORE each agent updates**:

```
Old (Fix #1):
  Collect data
  Agent 0 updates → π₀ shifts
  Agent 1 updates → π₁ shifts
  Train critic 25 epochs ← trying to fit already-shifted policies

New (Fix #2):
  Collect data
  PRE-TRAIN critic 5 epochs ← get good baseline values
  Agent 0 updates → π₀ shifts (critic ready)
  Agent 1 updates → π₁ shifts (critic ready)
  Train critic 25 epochs ← final fitting
```

**Why this helps:** Critic gets a "warm start" with good value estimates *before* the ground shifts under it. This reduces the tracking lag.

---

## Implementation

### Code Changes

**File:** `HARL/harl/runners/on_policy_ha_runner.py`

**Location:** Lines 52-64 (before the agent update loop)

```python
# CRITIC FIX 2 (Dec 18, 2025): Pre-train critic before agent updates
# Do 5 extra critic-only updates to stabilize value estimates before policies shift
# This helps critic "warm up" before each agent changes the joint policy landscape
critic_pretrain_epochs = getattr(self.algo_args["algo"], "critic_pretrain_epochs", 5)
if critic_pretrain_epochs > 0:
    # Store original critic_epoch
    original_critic_epoch = self.critic.critic_epoch
    # Temporarily set to pretrain epochs
    self.critic.critic_epoch = critic_pretrain_epochs
    # Pre-train critic
    _ = self.critic.train(self.critic_buffer, self.value_normalizer)
    # Restore original critic_epoch for later
    self.critic.critic_epoch = original_critic_epoch
```

**Config:** `HARL/harl/configs/algos_cfgs/happo.yaml`

```yaml
algo:
  ppo_epoch: 5                    # Actors: 5 epochs
  critic_epoch: 25                # Critic final: 25 epochs (from Fix #1)
  critic_pretrain_epochs: 5       # Critic pre-train: 5 epochs (NEW)
  value_loss_coef: 5.0            # Loss weight: 5.0 (from Fix #1)
```

---

## How It Works

### Training Sequence Per Iteration:

**Before (Fix #1 only):**
```
1. Collect 100K transitions (500 envs × 200 steps)
2. Compute returns and advantages
3. Agent 0 updates (5 epochs)
4. Agent 1 updates (5 epochs)
5. Critic trains (25 epochs) ← single phase
   Total critic updates: 25
```

**After (Fix #1 + Fix #2):**
```
1. Collect 100K transitions (500 envs × 200 steps)
2. Compute returns and advantages
3. Critic PRE-TRAINS (5 epochs) ← NEW! Stabilize baseline
4. Agent 0 updates (5 epochs)
5. Agent 1 updates (5 epochs)
6. Critic trains (25 epochs) ← final fitting
   Total critic updates: 5 + 25 = 30
```

### Effective Training:

- **Critic total:** 30 epochs per iteration (5 pretrain + 25 final)
- **Actor total:** 10 epochs per iteration (5 each × 2 agents)
- **Ratio:** 3:1 (critic trains 3x more than actors)

---

## Why This Should Work Better Than Fix #1

### Problem with Fix #1:
By the time critic trains (after both agents update), the joint policy has shifted significantly:
```
π(t=0) → collect data → π(t=1) [agent 0] → π(t=2) [agent 1] → train critic on π(t=2)
                                                                   ↑
                                               Trying to fit policies that already shifted twice!
```

Critic is always **lagging 2 updates behind**.

### Solution in Fix #2:
Pre-training gives critic a head start:
```
π(t=0) → collect data → pretrain critic on π(t=0) → π(t=1) [agent 0] → π(t=2) [agent 1] → final train
                        ↑                                                                    ↑
                    Warm start with good                                           Refinement
                    baseline values                                                on shifted policies
```

Critic is only **lagging 2 updates** instead of starting from scratch after 2 updates.

### Analogy:
**Fix #1:** Like trying to hit a moving target that's already far away.

**Fix #2:** Like warming up your aim on the target while it's still close, then tracking it as it moves.

---

## Expected Results

### Metrics to Watch:

**Good signs:**
1. ✅ **Critic loss decreases steadily** (not increasing like Fix #1)
2. ✅ **Lower oscillations** in loss curves
3. ✅ **Value predictions closer to returns** from the start
4. ✅ **Stable gradient norms**

**Success criteria:**
- Critic loss shows **decreasing trend** over 20M steps
- Loss stays below previous runs after 10M steps
- No divergence or explosion

### Comparison to Fix #1:

| Metric | Fix #1 (Failed) | Fix #2 (Expected) |
|--------|-----------------|-------------------|
| Critic loss @ 10M | Increasing | Decreasing |
| Loss oscillations | High | Lower |
| Value tracking | Poor | Good |
| Training stable | No | Yes |

---

## Training Time Impact

### Compute Cost:

**Fix #1 only:**
- Critic: 25 epochs per iteration
- Wall-clock: ~20-30% slower than baseline

**Fix #1 + Fix #2:**
- Critic: 30 epochs per iteration (5 pretrain + 25 final)
- Wall-clock: ~25-35% slower than baseline
- **Additional cost vs Fix #1:** ~5-10% slower

**Trade-off:** Slightly more compute, but should actually converge **faster** in terms of environment steps if it works.

---

## Theoretical Justification

### From Multi-Agent RL Literature:

**QMIX (Rashid et al., 2018):**
- Uses centralized value function
- Stabilizes Q-network with target networks
- Similar idea: stabilize before policy shifts

**MADDPG (Lowe et al., 2017):**
- Centralized critic with decentralized actors
- Updates critic more frequently than actors
- Ratio: 1 actor update per 5-10 critic updates

**Our approach:**
- Pre-training + heavy training = 30 critic epochs
- 2 sequential actor updates = 10 total actor epochs
- Ratio: 3:1 (critic:actor)
- Similar to MADDPG's asymmetric update schedule

### Why Pre-Training Helps:

Critic value function approximation error:
```
Error(t) = |V_θ(s) - R(s; π_t)|
```

With sequential updates, the policy π_t shifts rapidly. Pre-training ensures:
```
Error(t=0) is SMALL before π starts shifting
→ Easier to track π(t=1), π(t=2) from a good starting point
```

Without pre-training (Fix #1):
```
Error(t=0) is LARGE (random init or stale from prev iteration)
→ Hard to track π(t=1), π(t=2) starting from a bad baseline
```

---

## Implementation Details

### Config Parameters:

```yaml
critic_pretrain_epochs: 5  # Epochs BEFORE agent updates
critic_epoch: 25           # Epochs AFTER agent updates
value_loss_coef: 5.0       # Loss weight (from Fix #1)
```

**Tuning knobs:**
- Increase `critic_pretrain_epochs` (5 → 10) if still unstable
- Decrease if training too slow and results are good
- Keep `critic_epoch: 25` from Fix #1

### Code Logic:

The implementation uses a temporary override:
1. Store original `critic.critic_epoch = 25`
2. Temporarily set `critic.critic_epoch = 5`
3. Run `critic.train()` → does 5 epochs
4. Restore `critic.critic_epoch = 25`
5. Later: Final `critic.train()` → does 25 epochs

This is cleaner than modifying the train() function signature.

---

## Rollback Instructions

If Fix #2 makes things worse:

### Revert Code Change:
```bash
cd /home/gvlab/new-universal-MAPush
git diff HARL/harl/runners/on_policy_ha_runner.py
# If on git, revert:
git checkout HARL/harl/runners/on_policy_ha_runner.py
```

### Or manually remove lines 52-64:
Delete the entire block:
```python
# CRITIC FIX 2 ... (lines 52-64)
```

### Revert Config:
```yaml
# Remove this line:
critic_pretrain_epochs: 5
```

---

## Next Steps if Fix #2 Fails

If pre-training doesn't work either, the options are:

### **Fix #3: Slower Actor Updates**
- Update actors every 2-3 iterations instead of every iteration
- Gives critic more time to stabilize between policy shifts
- Trade-off: Slower learning but more stable

### **Fix #4: Target Critic Network**
- Add slow-moving target network (like DQN/TD3)
- Stabilizes bootstrapped returns
- Significant code changes required

### **Fix #5: Reduce Sequential Nature**
- Batch compute both agents' gradients
- Update critic once on joint data
- Apply both actor updates
- Less "whiplash" for critic
- Changes HAPPO algorithm fundamentally

---

## Relation to Fix #1

**Fix #1 is KEPT in Fix #2:**
- `critic_epoch: 25` (from Fix #1)
- `value_loss_coef: 5.0` (from Fix #1)

**Fix #2 ADDS:**
- Pre-training phase before agent updates
- Total: 5 + 25 = 30 critic epochs

**Combined effect:**
- More total training (30 vs 25)
- Better timing (before + after vs just after)
- Should address both quantity and quality of critic updates

---

## Testing Plan

1. **Launch training:**
   ```bash
   python train_script.py --exp_name critic_fix2_pretrain5
   ```

2. **Monitor for 10M steps:**
   - TensorBoard: `critic/value_loss`
   - Should see decreasing trend early (by 5M)

3. **Compare to Fix #1:**
   - Fix #1: Loss increased by 10M
   - Fix #2: Loss should decrease by 10M

4. **Decision point @ 10M:**
   - ✅ Decreasing → Continue to 40M+
   - ⚠️ Flat → Give it 10M more
   - ❌ Increasing → Move to Fix #3

---

## Changelog

**December 18, 2025:**
- Implemented Fix #2 on top of Fix #1
- Added `critic_pretrain_epochs: 5` config parameter
- Modified `on_policy_ha_runner.py` to pre-train before agent loop
- Ready for testing

**Status:** Implemented, awaiting test results
