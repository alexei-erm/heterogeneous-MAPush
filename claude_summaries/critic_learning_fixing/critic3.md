# Critic Fix #3: Slow Actor Updates

**Date:** December 19, 2025
**Branch:** critic-fixing
**Status:** Implemented, ready for testing
**Previous Fixes:**
- Fix #1 (critic_epoch: 25, value_loss_coef: 5.0) - **FAILED** (loss increasing)
- Fix #2 (critic_pretrain_epochs: 5) - **FAILED** (loss flat at 0.25, 15% success @ 76.5M steps)

---

## Why Fix #1 and Fix #2 Failed

**Fix #1 approach:** Train critic for 25 epochs instead of 5.
**Fix #2 approach:** Pre-train critic 5 epochs before actor updates, then train 25 epochs after.

**Combined result:** Both fixes failed. Critic loss plateaued around 0.25 with high oscillations. Success rate stuck at ~15% after 76.5M steps.

**Root cause:** Both fixes tried to make the critic train **harder**, but didn't address the fundamental problem - **the target is moving too fast**.

### The Core Problem

HAPPO's sequential updates create a rapidly shifting joint policy landscape:

```
Iteration N:
  Collect data with π(t)
  Agent 0 updates → π₀ becomes π₀'
  Agent 1 updates → π₁ becomes π₁'
  Critic tries to fit V(s; π₀', π₁')

Iteration N+1:
  Collect data with π'(t)
  Agent 0 updates → π₀' becomes π₀''
  Agent 1 updates → π₁' becomes π₁''
  Critic tries to fit V(s; π₀'', π₁'')
```

**Every iteration, the joint policy shifts twice** (once per agent). The critic can't keep up no matter how many epochs we train it within the same iteration.

**Analogy:** You're trying to hit a moving target that accelerates away from you. Training harder (Fix #1) or warming up first (Fix #2) doesn't help if the target is still accelerating faster than you can aim.

---

## Fix #3: Slow Down the Target

### The Idea

Instead of trying to make the critic faster, **slow down how fast the joint policy moves**:

```
Old (Every iteration):
  Iteration 1: Collect → Update actors → Update critic
  Iteration 2: Collect → Update actors → Update critic
  Iteration 3: Collect → Update actors → Update critic
  ...
  Joint policy shifts EVERY iteration

New (Every 3 iterations):
  Iteration 1: Collect → Update critic only
  Iteration 2: Collect → Update critic only
  Iteration 3: Collect → Update actors + critic
  Iteration 4: Collect → Update critic only
  Iteration 5: Collect → Update critic only
  Iteration 6: Collect → Update actors + critic
  ...
  Joint policy shifts every 3 iterations
```

**Why this helps:** Critic gets 2 full iterations to stabilize on a **fixed target** before the policies shift.

---

## Implementation

### Code Changes

**File:** `HARL/harl/runners/on_policy_ha_runner.py`

**Modified `train()` signature** (line 11):
```python
def train(self, episode=None):
    """Train the model."""
    actor_train_infos = []

    # CRITIC FIX 3 (Dec 19, 2025): Slow down actor updates to stabilize critic
    # Only update actors every N iterations to give critic breathing room
    actor_update_interval = getattr(self.algo_args["algo"], "actor_update_interval", 1)
    should_update_actors = (episode is None) or (episode % actor_update_interval == 0)
```

**Wrapped actor update loop** (lines 57-133):
```python
# Only update actors if we're on an update iteration
if should_update_actors:
    for agent_id in agent_order:
        # ... (all actor update code) ...
```

**File:** `HARL/harl/runners/on_policy_base_runner.py`

**Pass episode number to train()** (line 249):
```python
actor_train_infos, critic_train_info = self.train(episode)
```

**Config:** `HARL/harl/configs/algos_cfgs/happo.yaml`

```yaml
algo:
  ppo_epoch: 5                    # Actors: 5 epochs when they update
  critic_epoch: 25                # Critic: 25 epochs every iteration (from Fix #1)
  actor_update_interval: 3        # Actors update every 3 iterations (NEW)
  value_loss_coef: 5.0            # Loss weight: 5.0 (from Fix #1)
```

**Removed from Fix #2:**
- Deleted `critic_pretrain_epochs: 5`
- Removed pre-training code block (was lines 52-64 in runner)

---

## How It Works

### Training Sequence Per 3 Iterations:

**Iteration 1 (episode % 3 == 1):**
```
1. Collect 100K transitions (500 envs × 200 steps)
2. Compute returns and advantages
3. SKIP actor updates
4. Critic trains (25 epochs)
   Policies: π₀, π₁ (unchanged)
```

**Iteration 2 (episode % 3 == 2):**
```
1. Collect 100K transitions (500 envs × 200 steps)
2. Compute returns and advantages
3. SKIP actor updates
4. Critic trains (25 epochs)
   Policies: π₀, π₁ (still unchanged)
```

**Iteration 3 (episode % 3 == 0):**
```
1. Collect 100K transitions (500 envs × 200 steps)
2. Compute returns and advantages
3. Agent 0 updates (5 epochs) → π₀ becomes π₀'
4. Agent 1 updates (5 epochs) → π₁ becomes π₁'
5. Critic trains (25 epochs)
   Policies: π₀', π₁' (updated)
```

### Effective Training:

**Per 3 iterations:**
- Critic total: 75 epochs (25 × 3 iterations)
- Actor total: 10 epochs (5 × 2 agents, only on iteration 3)
- Ratio: 7.5:1 (critic trains 7.5x more than actors per update cycle)

**Joint policy shifts:** Once every 3 iterations instead of every iteration

---

## Why This Should Work Better Than Fix #1 and Fix #2

### Problem with Fix #1 + Fix #2:
Even with 30 epochs of critic training per iteration, the joint policy was shifting **every single iteration**. By the time critic converged, policies had already moved again.

```
Timeline with Fix #2:
t=0: π(v0) → Collect → Pretrain critic → Update actors → π(v1) → Train critic
t=1: π(v1) → Collect → Pretrain critic → Update actors → π(v2) → Train critic
t=2: π(v2) → Collect → Pretrain critic → Update actors → π(v3) → Train critic
...

Policy version changes every iteration. Critic is always chasing.
```

### Solution in Fix #3:
Policies stay **frozen for 2 iterations** while critic trains on stable targets:

```
Timeline with Fix #3:
t=0: π(v0) → Collect → Train critic only
t=1: π(v0) → Collect → Train critic only (same policies!)
t=2: π(v0) → Collect → Update actors → π(v1) → Train critic
t=3: π(v1) → Collect → Train critic only
t=4: π(v1) → Collect → Train critic only (same policies!)
t=5: π(v1) → Collect → Update actors → π(v2) → Train critic
...

Policy version changes every 3 iterations. Critic gets breathing room.
```

Critic trains on the **same joint policy** for 2 consecutive iterations before it shifts, giving it time to converge before the target moves.

### Analogy:
**Fix #1 + Fix #2:** Like trying to photograph a car speeding past you by taking more photos faster.

**Fix #3:** Like asking the car to stop every few meters so you can take clear photos.

---

## Expected Results

### Metrics to Watch:

**Good signs:**
1. ✅ **Critic loss decreases steadily** over the first 10M steps
2. ✅ **Lower oscillations** - loss should be smoother than Fix #1 and Fix #2
3. ✅ **Value predictions closer to returns**
4. ✅ **Success rate improves** (baseline: 15% @ 76M)

**Success criteria:**
- Critic loss shows **clear decreasing trend** by 10M steps
- Loss gets below 0.20 (Fix #2 plateaued at 0.25)
- Success rate reaches >25% by 40M steps
- No divergence or explosion

### Comparison to Previous Fixes:

| Metric | Fix #1 (Failed) | Fix #2 (Failed) | Fix #3 (Expected) |
|--------|-----------------|-----------------|-------------------|
| Critic loss @ 10M | Increasing | Flat (0.25) | Decreasing |
| Critic loss @ 76M | N/A | 0.25 | <0.20 |
| Success rate @ 76M | N/A | 15.64% | >25% |
| Loss oscillations | Very high | High | Lower |
| Training stable | No | Plateaued | Yes |

---

## Training Time Impact

### Compute Cost:

**Fix #1 + Fix #2:**
- Critic: 30 epochs per iteration (5 pretrain + 25 final)
- Actors: 10 epochs per iteration (5 × 2 agents)
- Every iteration

**Fix #3:**
- Critic: 25 epochs per iteration
- Actors: 10 epochs per 3 iterations (5 × 2 agents)
- Every 3rd iteration for actors

**Wall-clock comparison:**

Since actors are the computational bottleneck in this environment:
- Fix #2: 100% actor compute every iteration
- Fix #3: 33% actor compute (only every 3rd iteration)

**Expected:** Fix #3 should be **~15-20% faster** per iteration than Fix #2, but **~30-40% slower to convergence** in terms of environment steps because actors update less frequently.

**Trade-off:** Slower policy learning, but much better critic stability should lead to better final performance.

---

## Theoretical Justification

### From Multi-Agent RL Literature:

**A2C/A3C (Mnih et al., 2016):**
- Asynchronous updates with stale policies
- Showed that training on slightly outdated policies is acceptable
- Our approach: Intentionally use "stale" policies for critic training

**PPO (Schulman et al., 2017):**
- Multiple epochs of updates on same batch
- Works because of clipping that prevents policy from shifting too far
- Our approach: Multiple iterations on same policy version

**QMIX (Rashid et al., 2018):**
- Target network updated every K steps
- Stabilizes learning by fixing target for multiple updates
- Our approach: Actor networks stay fixed for K iterations

### Why Slowing Updates Helps:

Critic loss is driven by temporal difference error:
```
L = E[(V_θ(s) - (r + γV_θ(s')))²]
```

With sequential updates, the policy π in the data changes rapidly, making V_θ's target non-stationary:
```
Iteration t:   V_θ should approximate V^{π_t}
Iteration t+1: V_θ should approximate V^{π_{t+1}}
```

If π_{t+1} is very different from π_t, V_θ has to "unlearn" and "relearn" every iteration.

**Fix #3 reduces non-stationarity:** π stays the same for 3 iterations, so V_θ can actually converge to V^π before π changes.

---

## Implementation Details

### Config Parameters:

```yaml
actor_update_interval: 3  # Update actors every 3 iterations
critic_epoch: 25          # Critic trains 25 epochs per iteration (from Fix #1)
value_loss_coef: 5.0      # Loss weight (from Fix #1)
```

**Tuning knobs:**
- Increase `actor_update_interval` (3 → 5) if still unstable
- Decrease (3 → 2) if learning too slow
- Keep `critic_epoch: 25` and `value_loss_coef: 5.0` from Fix #1

### Code Logic:

The implementation checks if current iteration is divisible by `actor_update_interval`:
```python
should_update_actors = (episode % actor_update_interval == 0)
```

- Iteration 1 (1 % 3 = 1): False → Skip actors
- Iteration 2 (2 % 3 = 2): False → Skip actors
- Iteration 3 (3 % 3 = 0): True → Update actors
- Iteration 4 (4 % 3 = 1): False → Skip actors
- ...

Critic **always** trains regardless of `should_update_actors`.

---

## Rollback Instructions

If Fix #3 makes things worse:

### Revert Code Changes:
```bash
cd /home/gvlab/new-universal-MAPush
git diff HARL/harl/runners/on_policy_ha_runner.py
git diff HARL/harl/runners/on_policy_base_runner.py
# If on git, revert:
git checkout HARL/harl/runners/on_policy_ha_runner.py
git checkout HARL/harl/runners/on_policy_base_runner.py
```

### Revert Config:
```yaml
# Remove this line:
actor_update_interval: 3
```

---

## Next Steps if Fix #3 Fails

If slowing actor updates doesn't work, the options are:

### **Fix #4: Target Critic Network**
- Maintain two critic networks: online and target
- Update target slowly (Polyak averaging: θ_target = 0.995 * θ_target + 0.005 * θ_online)
- Compute returns using target network (like TD3/SAC)
- Stabilizes bootstrapped values
- Significant code changes required

### **Fix #5: Synchronized Batch Updates**
- Compute gradients for both agents simultaneously
- Apply both actor updates at once (instead of sequentially)
- Update critic once after both actors change
- Reduces "whiplash" effect on critic
- Changes HAPPO algorithm fundamentally (no longer sequential)

### **Fix #6: Reduce Critic Learning Rate**
- Lower `critic_lr` from 0.0005 to 0.0001
- Slower critic updates = less overreaction to policy shifts
- Simpler than target networks
- May need more total training time

---

## Relation to Previous Fixes

**Fix #1 (KEPT):**
- `critic_epoch: 25`
- `value_loss_coef: 5.0`

**Fix #2 (REMOVED):**
- Deleted `critic_pretrain_epochs: 5`
- Removed pre-training code

**Fix #3 (ADDED):**
- `actor_update_interval: 3`
- Conditional actor updates
- Episode number passed to `train()`

**Combined effect:**
- Less frequent policy shifts (every 3 iterations vs every iteration)
- Critic still trains heavily (25 epochs) each iteration
- Should address root cause: non-stationary target problem

---

## Testing Plan

1. **Launch training:**
   ```bash
   python train_script.py --exp_name critic_fix3_interval3
   ```

2. **Monitor for 10M steps:**
   - TensorBoard: `critic/value_loss`
   - Should see **decreasing trend** (not flat like Fix #2)
   - Should see **lower oscillations**

3. **Compare to Fix #2:**
   - Fix #2: Loss flat at 0.25, success 15% @ 76M
   - Fix #3: Loss should drop below 0.20, success >20% @ 40M

4. **Decision point @ 20M:**
   - ✅ Decreasing and below 0.20 → Continue to 100M
   - ⚠️ Decreasing but slow → Give it 20M more
   - ❌ Flat or increasing → Move to Fix #4

---

## Changelog

**December 19, 2025:**
- Removed Fix #2 (pre-training) - did not work
- Implemented Fix #3 (slow actor updates)
- Added `actor_update_interval: 3` config parameter
- Modified `on_policy_ha_runner.py` to conditionally skip actor updates
- Modified `on_policy_base_runner.py` to pass episode number to `train()`
- Ready for testing

**Status:** Implemented, awaiting test results
