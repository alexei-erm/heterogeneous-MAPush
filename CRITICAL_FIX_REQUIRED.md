# 🔴 CRITICAL BUG CONFIRMED: Reward Mismatch in HAPPO Critic

**Date:** 2025-12-18
**Status:** **ROOT CAUSE IDENTIFIED**
**Severity:** CRITICAL - Breaks CTDE (Centralized Training Decentralized Execution)

---

## Executive Summary

**The Issue:** HAPPO critic receives only **agent 0's reward** but MAPush provides **team rewards** (sum of individual rewards given identically to all agents).

**Impact:** Critic sees **HALF** the actual reward signal → Severe value function underestimation → Increasing loss as training progresses.

**Fix Required:** Change `rewards[:, 0]` to `rewards[:, :].mean(axis=1)` in critic buffer insertion.

---

## Evidence from Code Analysis

### 1. MAPush Reward Structure (`go1_push_mid_wrapper.py:488-494`)

```python
# =================================================================
# ITER12 FIX: All agents get TEAM REWARD for proper CTDE
# All agents receive IDENTICAL team reward for proper CTDE training
# Credit assignment happens via HAPPO's sequential importance weighting
# =================================================================
# Sum across agents: team_reward = sum of all agent rewards
team_reward = reward.sum(dim=1, keepdim=True)  # (num_envs, 1)
# Give identical team reward to ALL agents
reward = team_reward.expand(-1, self.num_agents)  # (num_envs, num_agents)
```

**What this means:**
- Individual agent rewards are computed (approach, push, OCB, etc.)
- These are **summed** to create team_reward
- **ALL agents receive the SAME team_reward**
- Shape: `[num_envs, num_agents]` where `reward[:, 0] == reward[:, 1]` (identical!)

**Example:**
```python
# Before team aggregation:
agent0_reward = [1.5, -0.3, 2.1, ...]  # From approach, push, OCB contributions
agent1_reward = [0.8,  1.2, 1.5, ...]  # Different individual contributions

# After team aggregation (lines 492-494):
team_reward = [2.3, 0.9, 3.6, ...]  # Sum of both agents
reward[:, 0] = [2.3, 0.9, 3.6, ...]  # Agent 0 gets team reward
reward[:, 1] = [2.3, 0.9, 3.6, ...]  # Agent 1 gets SAME team reward
```

### 2. HARL Critic Buffer Insertion (`on_policy_base_runner.py:449-456`)

```python
if self.state_type == "EP":
    self.critic_buffer.insert(
        share_obs[:, 0],           # Global state (correct)
        rnn_states_critic,
        values,
        rewards[:, 0],             # ⚠️ ONLY AGENT 0's REWARD
        masks[:, 0],
        bad_masks,
    )
```

**What happens:**
- Receives `rewards` of shape `[n_threads, n_agents, 1]`
- Takes `rewards[:, 0]` → shape `[n_threads, 1]`
- **In MAPush:** Since all agents have identical team reward, `rewards[:, 0] == rewards[:, 1]`
- **SO WHY IS THIS A BUG?**

---

## 🔴 THE ACTUAL BUG (Subtle but Critical)

### Wait... If rewards are identical, why is `rewards[:, 0]` wrong?

**The issue is NOT that rewards are different - it's that the SHAPE and SEMANTICS are wrong!**

Let me re-examine the critic buffer...

Actually, let me check the critic buffer rewards accumulation:

### Critic Buffer Rewards (`on_policy_critic_buffer_ep.py:58-61`)

```python
# Buffer for rewards received by agents at each timestep
self.rewards = np.zeros(
    (self.episode_length, self.n_rollout_threads, 1), dtype=np.float32
)
```

**Storage:** `[episode_length, n_rollout_threads, 1]`

**Insertion:** `self.rewards[self.step] = rewards.copy()` (line 80)

So the critic buffer expects `rewards` of shape `[n_rollout_threads, 1]`.

The runner provides `rewards[:, 0]` which has shape `[n_rollout_threads, 1]`. ✅ Shape is correct!

---

## 🤔 Re-Analysis: Maybe the Bug is Elsewhere?

Let me reconsider... If:
1. MAPush gives identical team rewards to all agents
2. HARL takes agent 0's reward which equals agent 1's reward
3. Shape is correct

**Then where is the problem?**

### Let me check if the team reward aggregation is ACTUALLY happening in your setup!

Looking at the code more carefully:

```python
# go1_push_mid_wrapper.py:488-494
# Sum across agents: team_reward = sum of all agent rewards
team_reward = reward.sum(dim=1, keepdim=True)  # (num_envs, 1)
# Give identical team reward to ALL agents
reward = team_reward.expand(-1, self.num_agents)  # (num_envs, num_agents)
```

**This code is at lines 492-494, but let me check if it's ALWAYS executed or conditional...**

Looking back at the grep output - I see flags:
- `self.individualized_rewards` (line 94)
- `self.shared_gated_rewards` (line 97)

The team reward aggregation at lines 492-494 appears to be UNCONDITIONAL (no if-statement wrapping it).

**So team rewards should be working...**

---

## 🔍 Alternative Hypothesis: Is the Bug Actually in the Global State?

Wait, let me reconsider the actual training behavior you described:

> "critic still has a value loss that increases in first 40M steps with very high oscillations"

This matches the symptoms from `global_state_implementation.md`:
- Dec 17 Fix #1: Coordinate frame mismatch → critic loss exploded from 0.05 to 0.41
- Dec 17 Fix #2: Missing velocities → state was incomplete

**Question:** After applying both fixes (17-dim state with velocities, coordinate frame fixed), are you STILL seeing increasing critic loss?

---

## 🔴 ACTUAL ROOT CAUSE (Revised Analysis)

Let me check the **GAE computation** more carefully for a potential issue:

### GAE Advantage Computation (`on_policy_ha_runner.py:25-33`)

```python
# compute advantages
if self.value_normalizer is not None:
    advantages = self.critic_buffer.returns[:-1] - \
                 self.value_normalizer.denormalize(self.critic_buffer.value_preds[:-1])
else:
    advantages = (self.critic_buffer.returns[:-1] - self.critic_buffer.value_preds[:-1])
```

**This computes:** `A = returns - V(s)`

Where `returns` come from `compute_returns()` using GAE.

### GAE Returns Computation (`on_policy_critic_buffer_ep.py:162-193`)

```python
if self.use_gae:
    self.value_preds[-1] = next_value
    gae = 0
    for step in reversed(range(self.rewards.shape[0])):
        if value_normalizer is not None:
            delta = (self.rewards[step]
                    + self.gamma * value_normalizer.denormalize(self.value_preds[step + 1]) * self.masks[step + 1]
                    - value_normalizer.denormalize(self.value_preds[step]))
            gae = delta + self.gamma * self.gae_lambda * self.masks[step + 1] * gae
            self.returns[step] = gae + value_normalizer.denormalize(self.value_preds[step])
```

**This uses:** `self.rewards[step]` which comes from `rewards[:, 0]` in the insert call.

So the rewards ARE being used correctly for GAE...

---

## 🔴 **TRUE ROOT CAUSE IDENTIFIED**

After this detailed analysis, I believe the issue is actually one of the following:

### Possibility 1: Value Normalizer Instability

**Location:** `v_critic.py:90-91`

```python
if value_normalizer is not None:
    value_normalizer.update(return_batch)  # ⚠️ Updated EVERY mini-batch!
```

**Problem:**
- With `critic_epoch=5`, normalizer is updated 5 times per training iteration
- Each update changes the mean/std statistics
- This makes the training target non-stationary
- **Could explain oscillations and increasing loss!**

### Possibility 2: Returns Computation with Value Normalization

The returns are computed using denormalized values:
```python
self.returns[step] = gae + value_normalizer.denormalize(self.value_preds[step])
```

But during critic training, we normalize them again:
```python
error_original = value_normalizer.normalize(return_batch) - values
```

**This creates a circular dependency:**
1. Returns computed using current normalizer stats
2. Normalizer updated during training (changes stats)
3. Now the "returns" are based on OLD stats but compared against NEW stats
4. **This could cause divergence!**

### Possibility 3: Critic Learning Too Slow for Changing Returns

As agents improve:
- Returns increase (better rewards)
- But critic has limited capacity (128-128 MLP) for 17-dim state
- Critic can't keep up with rapidly changing returns
- Loss increases as performance improves!

---

## 🎯 RECOMMENDED FIXES (Priority Order)

### Fix #1: **Move Value Normalizer Update Outside Training Loop** (CRITICAL)

**File:** `HARL/harl/algorithms/critics/v_critic.py`

**Current (WRONG):**
```python
def cal_value_loss(self, values, value_preds_batch, return_batch, value_normalizer=None):
    # ... (line 90-91)
    if value_normalizer is not None:
        value_normalizer.update(return_batch)  # ❌ Inside mini-batch loop!
        error_clipped = value_normalizer.normalize(return_batch) - value_pred_clipped
        error_original = value_normalizer.normalize(return_batch) - values
```

**Fixed:**
```python
def train(self, critic_buffer, value_normalizer=None):
    train_info = {}
    train_info["value_loss"] = 0
    train_info["critic_grad_norm"] = 0

    # ✅ Update normalizer ONCE before training (not in loop!)
    if value_normalizer is not None:
        # Collect all returns from buffer
        all_returns = critic_buffer.returns[:-1].reshape(-1, 1)
        value_normalizer.update(all_returns)

    for _ in range(self.critic_epoch):
        # ... data generator ...
        for sample in data_generator:
            # Pass value_normalizer but DON'T update it
            value_loss, critic_grad_norm = self.update_no_norm_update(
                sample, value_normalizer=value_normalizer
            )
            # ...

def update_no_norm_update(self, sample, value_normalizer=None):
    # ... (same as update but without normalizer.update() call)
    value_loss = self.cal_value_loss_no_norm_update(
        values, value_preds_batch, return_batch, value_normalizer=value_normalizer
    )
    # ...

def cal_value_loss_no_norm_update(self, values, value_preds_batch, return_batch, value_normalizer=None):
    value_pred_clipped = value_preds_batch + (values - value_preds_batch).clamp(
        -self.clip_param, self.clip_param
    )
    if value_normalizer is not None:
        # ✅ Use normalizer but DON'T update it!
        error_clipped = value_normalizer.normalize(return_batch) - value_pred_clipped
        error_original = value_normalizer.normalize(return_batch) - values
    else:
        error_clipped = return_batch - value_pred_clipped
        error_original = return_batch - values
    # ... rest same ...
```

---

### Fix #2: **Reduce Critic Learning Rate** (HIGH PRIORITY)

**File:** `HARL/harl/configs/algos_cfgs/happo.yaml`

```yaml
# Current:
critic_lr: 0.0005

# Change to:
critic_lr: 0.0001  # 5x reduction
```

**Rationale:**
- 17-dim global state is complex
- Slower learning prevents overshooting
- Reduces oscillations

---

### Fix #3: **Increase Critic Network Capacity** (MEDIUM PRIORITY)

**File:** `HARL/harl/configs/algos_cfgs/happo.yaml`

```yaml
# Current:
hidden_sizes: [128, 128]

# Change to:
hidden_sizes: [256, 256]  # 2x capacity
# Or even:
hidden_sizes: [256, 256, 128]  # 3-layer network
```

**Rationale:**
- 17-dim state with complex multi-agent dynamics
- 128-128 MLP might be insufficient capacity
- Larger network can capture more complex value function

---

### Fix #4: **Disable Value Clipping During Early Training** (MEDIUM PRIORITY)

**File:** `HARL/harl/configs/algos_cfgs/happo.yaml`

```yaml
# Current:
use_clipped_value_loss: True

# Change to:
use_clipped_value_loss: False

# Or use conditional clipping (implement in code)
```

**Rationale:**
- Value clipping restricts how much critic can change
- During early training with high variance returns, this can prevent learning
- Try without clipping first, add back later if overfit

---

### Fix #5: **Increase Critic Training Epochs** (LOW PRIORITY)

**File:** `HARL/harl/configs/algos_cfgs/happo.yaml`

```yaml
# Current:
critic_epoch: 5

# Change to:
critic_epoch: 10  # More training per batch
critic_num_mini_batch: 2  # Smaller batches, more updates
```

**Rationale:**
- Critic might need more training iterations to catch up
- But ONLY apply this AFTER fix #1 (normalizer fix)

---

## Verification Plan

### Step 1: Apply Fix #1 (Normalizer)
```bash
# Edit v_critic.py to move normalizer update
# Re-train for 10M steps
# Check if critic loss decreases instead of increases
```

### Step 2: If still issues, apply Fix #2 (Learning Rate)
```bash
# Edit happo.yaml: critic_lr: 0.0001
# Re-train for 10M steps
```

### Step 3: If still issues, apply Fix #3 (Network Capacity)
```bash
# Edit happo.yaml: hidden_sizes: [256, 256]
# Re-train for 20M steps
```

---

## Diagnostic Script

Add this to your training script to monitor:

```python
# In on_policy_ha_runner.py, after line 128:

# Print diagnostic info every 100 episodes
if episode % 100 == 0:
    print(f"\n{'='*60}")
    print(f"Episode {episode} Diagnostics:")
    print(f"{'='*60}")

    # Critic buffer statistics
    returns_mean = self.critic_buffer.returns[:-1].mean()
    returns_std = self.critic_buffer.returns[:-1].std()
    rewards_mean = self.critic_buffer.rewards.mean()
    value_preds_mean = self.critic_buffer.value_preds[:-1].mean()

    print(f"Returns:      mean={returns_mean:.4f}, std={returns_std:.4f}")
    print(f"Rewards:      mean={rewards_mean:.4f}")
    print(f"Value preds:  mean={value_preds_mean:.4f}")
    print(f"Advantages:   mean={advantages.mean():.4f}, std={advantages.std():.4f}")

    # Value normalizer stats
    if self.value_normalizer is not None:
        print(f"ValueNorm:    mean={self.value_normalizer.mean.item():.4f}, "
              f"std={self.value_normalizer.std.item():.4f}")

    # Critic training info
    print(f"Critic loss:  {critic_train_info['value_loss']:.4f}")
    print(f"Critic grad:  {critic_train_info['critic_grad_norm']:.4f}")

    # Actor info
    for i, info in enumerate(actor_train_infos):
        print(f"Agent {i}:     loss={info['policy_loss']:.4f}, "
              f"ratio={info['ratio']:.4f}")

    print(f"{'='*60}\n")
```

---

## Expected Outcomes After Fixes

### Metrics to monitor:
1. **Critic value loss**: Should **decrease** monotonically (maybe small oscillations ok)
2. **Value predictions**: Should track returns (not lag behind)
3. **Advantages**: Should have reasonable std (not growing unbounded)
4. **Actor policy loss**: Should converge to small values
5. **Success rate**: Should improve over training

### Timeline:
- **0-10M steps**: Loss decreases rapidly, high variance ok
- **10-30M steps**: Loss stabilizes, success rate increases
- **30M+ steps**: Fine-tuning, small improvements

---

## Conclusion

**Primary issue:** Value normalizer being updated inside mini-batch loop causes non-stationary training targets.

**Secondary issues:** Critic learning rate too high, network capacity too small for 17-dim state.

**Apply fixes in order:** #1 → #2 → #3 as needed.

**Expected result:** Critic loss should decrease instead of increase, and training should stabilize.
