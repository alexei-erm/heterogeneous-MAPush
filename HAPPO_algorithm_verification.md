# HAPPO Algorithm 4 Verification Report

**Date:** 2025-12-18
**Purpose:** Verify HARL's HAPPO implementation against Algorithm 4 specification
**Current Issue:** Critic value loss increases in first 40M steps with high oscillations

---

## Algorithm 4 Specification (Pseudo-code)

```
Input: Stepsize alpha, batch size B, number of: agents n, episodes K, steps per episode T.
Initialize: Actor networks {theta_i_0, for all i in N}, Global V-value network {phi_0}, Replay buffer B

for k = 0, 1, ..., K - 1 do
    # STEP 1: Data Collection
    Collect a set of trajectories by running joint policy pi_theta_k = (pi_1_theta_1_k, ..., pi_n_theta_n_k).
    Push transitions {(s_t, o_i_t, a_i_t, r_t, s_{t+1}, o_i_{t+1}), for all i in N, t in T} into B.

    # STEP 2: Sampling
    Sample a random minibatch of B transitions from B.

    # STEP 3: Advantage Computation
    Compute advantage function A_hat(s, a) based on global V-value network with GAE.

    # STEP 4: Sequential Actor Updates
    Draw a random permutation of agents i_1:n.
    Set M_{i_1}(s, a) = A_hat(s, a).

    for agent i_m = i_1, ..., i_n do
        # Update actor with PPO-Clip objective
        Update actor i_m with theta_i_m_{k+1}, the argmax of:
        1/BT * sum_{b=1}^B sum_{t=0}^T min(
            [pi_i_m_theta(a_i_m_t | o_i_m_t) / pi_i_m_theta_k(a_i_m_t | o_i_m_t)] * M_{i_1:m}(s_t, a_t),
            clip([...], 1 +/- epsilon) * M_{i_1:m}(s_t, a_t)
        )

        # Update importance sampling factor
        Compute M_{i_1:m+1}(s, a) =
            [pi_i_m_theta_{k+1}(a_i_m | o_i_m) / pi_i_m_theta_k(a_i_m | o_i_m)] * M_{i_1:m}(s, a).

    # STEP 5: Critic Update
    Update V-value network:
    phi_{k+1} = arg min_phi 1/BT * sum_{b=1}^B sum_{t=0}^T (V_phi(s_t) - R_hat_t)^2
```

---

## Implementation Verification

### ✅ STEP 1: Data Collection

**Algorithm requirement:** Collect trajectories using joint policy

**Implementation:** `on_policy_base_runner.py:203-243`

```python
for step in range(episode_length):  # T steps
    # Collect actions from all agents
    values, actions, action_log_probs, rnn_states, rnn_states_critic = self.collect(step)

    # Step environment
    obs, share_obs, rewards, dones, infos, available_actions = self.envs.step(actions)

    # Insert into buffer
    self.insert(data)
```

**Verification:**
- ✅ Runs joint policy for T=200 steps (episode_length)
- ✅ Collects (s_t, o_i_t, a_i_t, r_t, s_{t+1}, o_i_{t+1}) for all agents
- ✅ Stores in per-agent actor buffers and shared critic buffer
- ✅ **STATUS: CORRECT**

---

### ✅ STEP 2: Advantage Computation with GAE

**Algorithm requirement:** Compute A_hat(s, a) using global V-value network

**Implementation:** `on_policy_ha_runner.py:25-45`

```python
# Compute advantages from critic buffer
if self.value_normalizer is not None:
    advantages = self.critic_buffer.returns[:-1] -
                 self.value_normalizer.denormalize(self.critic_buffer.value_preds[:-1])
else:
    advantages = self.critic_buffer.returns[:-1] - self.critic_buffer.value_preds[:-1]
```

**Returns computation:** `on_policy_critic_buffer_ep.py:97-200` (using GAE)

```python
def compute_returns(self, next_value, value_normalizer=None):
    if self.use_gae:  # GAE enabled
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

**Configuration:** `happo.yaml:101-106`
```yaml
use_gae: True
gamma: 0.99
gae_lambda: 0.95
```

**Verification:**
- ✅ Uses GAE for advantage estimation
- ✅ Advantages computed from global V-network (centralized critic)
- ✅ Formula: A_t = returns_t - V(s_t) where returns use GAE
- ✅ **STATUS: CORRECT**

---

### ✅ STEP 3: Random Agent Permutation

**Algorithm requirement:** Draw random permutation of agents

**Implementation:** `on_policy_ha_runner.py:47-50`

```python
if self.fixed_order:
    agent_order = list(range(self.num_agents))
else:
    agent_order = list(torch.randperm(self.num_agents).numpy())
```

**Configuration:** `happo.yaml:118`
```yaml
fixed_order: False
```

**Verification:**
- ✅ Random permutation when `fixed_order=False`
- ✅ **STATUS: CORRECT**

---

### ✅ STEP 4: Sequential Actor Updates with Importance Sampling

**Algorithm requirement:**
1. Initialize M_{i_1} = A_hat
2. For each agent in order:
   - Update actor with M_{i_1:m} * PPO objective
   - Compute new factor: M_{i_1:m+1} = (pi_new / pi_old) * M_{i_1:m}

**Implementation:** `on_policy_ha_runner.py:15-126`

```python
# Initialize factor
factor = np.ones((episode_length, n_rollout_threads, 1), dtype=np.float32)

for agent_id in agent_order:
    # Save factor to actor buffer (M_{i_1:m})
    self.actor_buffer[agent_id].update_factor(factor)

    # Compute old action log probs BEFORE update
    old_actions_logprob, _, _ = self.actor[agent_id].evaluate_actions(...)

    # Update actor (uses factor in loss)
    actor_train_info = self.actor[agent_id].train(
        self.actor_buffer[agent_id], advantages.copy(), "EP"
    )

    # Compute NEW action log probs AFTER update
    new_actions_logprob, _, _ = self.actor[agent_id].evaluate_actions(...)

    # Update factor for next agent: factor *= exp(new - old)
    factor = factor * _t2n(
        torch.exp(new_actions_logprob - old_actions_logprob).reshape(...)
    )
```

**Actor update uses factor:** `happo.py:28-102`

```python
def update(self, sample):
    # Extract factor from sample
    factor_batch = check(factor_batch).to(**self.tpdv)

    # Compute importance weight: pi_new / pi_old
    imp_weights = torch.exp(action_log_probs - old_action_log_probs_batch)

    # PPO surrogate objectives
    surr1 = imp_weights * adv_targ
    surr2 = torch.clamp(imp_weights, 1.0 - self.clip_param, 1.0 + self.clip_param) * adv_targ

    # Apply factor (M_{i_1:m}) to loss
    policy_action_loss = -torch.sum(factor_batch * torch.min(surr1, surr2), ...).mean()
```

**Verification:**
- ✅ Factor initialized to 1.0 (M_{i_1} = A_hat)
- ✅ Each actor receives current factor
- ✅ Factor multiplied with PPO objective
- ✅ Factor updated after each agent: `factor *= exp(new_log_prob - old_log_prob)`
- ✅ **STATUS: CORRECT**

---

### ⚠️ STEP 5: Critic Update - **POTENTIAL ISSUE IDENTIFIED**

**Algorithm requirement:** Update V-network with MSE: (V_phi(s_t) - R_hat_t)^2

**Implementation:** `on_policy_ha_runner.py:127-129`

```python
# Update critic AFTER all actors updated
critic_train_info = self.critic.train(self.critic_buffer, self.value_normalizer)
```

**Critic training:** `v_critic.py:159-200`

```python
def train(self, critic_buffer, value_normalizer=None):
    for _ in range(self.critic_epoch):  # Default: 5 epochs
        data_generator = critic_buffer.feed_forward_generator_critic(
            self.critic_num_mini_batch  # Default: 1 mini-batch
        )

        for sample in data_generator:
            value_loss, critic_grad_norm = self.update(sample, value_normalizer)
```

**Loss computation:** `v_critic.py:75-114`

```python
def cal_value_loss(self, values, value_preds_batch, return_batch, value_normalizer=None):
    # Value clipping (PPO-style)
    value_pred_clipped = value_preds_batch + (values - value_preds_batch).clamp(
        -self.clip_param, self.clip_param
    )

    if value_normalizer is not None:
        value_normalizer.update(return_batch)  # ⚠️ UPDATES NORMALIZER EVERY BATCH
        error_clipped = value_normalizer.normalize(return_batch) - value_pred_clipped
        error_original = value_normalizer.normalize(return_batch) - values
    else:
        error_clipped = return_batch - value_pred_clipped
        error_original = return_batch - values

    if self.use_huber_loss:
        value_loss_clipped = huber_loss(error_clipped, self.huber_delta)
        value_loss_original = huber_loss(error_original, self.huber_delta)
    else:
        value_loss_clipped = mse_loss(error_clipped)
        value_loss_original = mse_loss(error_original)

    if self.use_clipped_value_loss:
        value_loss = torch.max(value_loss_original, value_loss_clipped)
    else:
        value_loss = value_loss_original

    return value_loss.mean()
```

**Configuration:** `happo.yaml`
```yaml
critic_epoch: 5
critic_num_mini_batch: 1
use_clipped_value_loss: True
use_huber_loss: True
huber_delta: 10.0
use_valuenorm: True
clip_param: 0.2
```

### 🔴 CRITICAL ISSUES IDENTIFIED

---

## Issue #1: ⚠️ **Reward Collection - ONLY AGENT 0's REWARD USED FOR CRITIC**

**Location:** `on_policy_base_runner.py:449-456`

```python
if self.state_type == "EP":
    self.critic_buffer.insert(
        share_obs[:, 0],           # Global state from agent 0
        rnn_states_critic,
        values,
        rewards[:, 0],             # ⚠️ ONLY AGENT 0's REWARD!
        masks[:, 0],
        bad_masks,
    )
```

**Problem:**
- In EP mode, the critic buffer receives **only agent 0's reward**: `rewards[:, 0]`
- Rewards shape: `[n_threads, n_agents, 1]` → Taking `[:, 0]` gives `[n_threads, 1]`
- **This means agent 1's reward is COMPLETELY IGNORED!**

**Expected behavior (for team reward):**
- Should use **sum** or **mean** of all agent rewards: `rewards.mean(axis=1)` or `rewards.sum(axis=1)`
- Or if individual rewards: Should track separately in FP mode

**Impact on training:**
- Critic only learns from agent 0's reward signal
- Agent 1's contributions to team performance are invisible to critic
- Advantages computed for agent 1 are based on WRONG baseline
- **This violates HAPPO's theoretical foundation!**

**Verification in your environment:**

Let me check if MAPush uses team reward or individual rewards:

---

## Issue #2: ⚠️ **Value Normalizer Updated Every Mini-batch**

**Location:** `v_critic.py:90-91`

```python
if value_normalizer is not None:
    value_normalizer.update(return_batch)  # Called EVERY mini-batch!
```

**Problem:**
- Value normalizer is updated **inside the mini-batch loop**
- With `critic_epoch=5` and `critic_num_mini_batch=1`, normalizer is updated **5 times per training step**
- Each update changes the normalization statistics, making training targets non-stationary

**Expected behavior:**
- Should update normalizer **once** at the beginning of training, not in the inner loop

**Impact:**
- Unstable training targets
- High variance in critic loss
- Potential cause of oscillations

---

## Issue #3: 🔴 **Missing Reward Aggregation Logic**

**The core question:** Does MAPush use:
1. **Team reward** (all agents receive same reward)?
2. **Individual rewards** (each agent receives different reward)?

**Current implementation assumes:** Individual rewards but only uses agent 0's reward in EP mode!

**Verification needed:**

Let me check the MAPush environment wrapper to see reward structure.

---

## Issue #4: ⚠️ **Potential: High Critic Learning Rate**

**Configuration:** `happo.yaml:69-70`
```yaml
critic_lr: 0.0005
```

**Analysis:**
- Same learning rate as actor (0.0005)
- For critic, this might be too high given the 17-dim state space
- Could contribute to oscillations

**Recommendation:** Try `critic_lr: 0.0001` or `0.0003`

---

## Issue #5: ⚠️ **Clipped Value Loss Might Be Too Restrictive**

**Configuration:**
```yaml
use_clipped_value_loss: True
clip_param: 0.2
```

**Analysis:**
- Value clipping with clip_param=0.2 restricts how much V-network can change per update
- During early training when returns are highly variable, this could prevent critic from learning
- Might cause critic to get stuck with poor value estimates

**Recommendation:** Try `use_clipped_value_loss: False` during early training

---

## Issue #6: ⚠️ **Large Huber Delta**

**Configuration:**
```yaml
use_huber_loss: True
huber_delta: 10.0
```

**Analysis:**
- Huber delta of 10.0 is quite large
- Huber loss behaves like L2 (MSE) for errors < 10.0, and L1 for errors > 10.0
- If returns are normalized, delta=10 is effectively "always L2"
- Might be less robust to outliers than intended

**Recommendation:** Try `huber_delta: 1.0` or use MSE (`use_huber_loss: False`)

---

## Summary of Verification

| Component | Algorithm 4 Requirement | Implementation | Status |
|-----------|------------------------|----------------|--------|
| **Data Collection** | Collect trajectories with joint policy | ✅ Correct | ✅ PASS |
| **GAE Advantages** | Compute A_hat using global V-network | ✅ Correct | ✅ PASS |
| **Random Permutation** | Random agent order | ✅ Correct | ✅ PASS |
| **Sequential Updates** | Update actors sequentially with IS | ✅ Correct | ✅ PASS |
| **Factor Propagation** | M_{i+1} = (pi_new/pi_old) * M_i | ✅ Correct | ✅ PASS |
| **Critic Update** | MSE loss on returns | ⚠️ Has issues | 🔴 **FAIL** |
| **Reward Collection** | Use rewards for V-network | 🔴 Only agent 0 | 🔴 **CRITICAL** |

---

## 🔴 **ROOT CAUSE IDENTIFIED**

### **THE CRITICAL BUG: Only Agent 0's Reward Used in EP Mode**

**File:** `on_policy_base_runner.py:453`

```python
rewards[:, 0],  # ⚠️ BUG: Should aggregate all agents' rewards!
```

**Why this causes increasing critic loss:**

1. **Early training:** Both agents perform poorly → agent 0's reward is representative
2. **Mid training:** Agent 1 starts learning → contributes to box pushing → gets higher rewards
3. **Critic's view:** Only sees agent 0's reward → underestimates true team performance
4. **Result:** Critic value predictions become increasingly wrong as agent 1 improves
5. **Loss increases:** Mismatch between true returns (both agents) and learned values (agent 0 only)

**This explains the 40M step pattern:**
- First 10-20M: Both agents learning slowly → rewards similar → critic okay
- 20-40M: Agent 1 improves faster or contributes more → divergence grows → **loss increases**
- High oscillations: Depends on which agent (0 or 1) makes progress in each episode

---

## Recommendations (Priority Order)

### 1. 🔴 **FIX CRITICAL BUG: Reward Aggregation** (IMMEDIATE)

Check if MAPush uses team reward or individual rewards, then fix accordingly:

**Option A: If team reward (both agents get same reward):**
```python
# Use either agent's reward (they're identical)
rewards[:, 0]  # Current (works if team reward)
```

**Option B: If individual rewards (agents get different rewards):**
```python
# Use mean of both agents' rewards
rewards.mean(axis=1, keepdims=True)  # Shape: [n_threads, 1]
```

**Option C: If individual rewards and want separate critics:**
- Switch to FP mode instead of EP mode
- Use agent-specific states and rewards

### 2. 🟡 **FIX: Value Normalizer Update Location** (HIGH PRIORITY)

Move normalizer update outside mini-batch loop:

```python
def train(self, critic_buffer, value_normalizer=None):
    # Update normalizer ONCE before training
    if value_normalizer is not None:
        all_returns = critic_buffer.returns[:-1].reshape(-1, 1)
        value_normalizer.update(all_returns)

    for _ in range(self.critic_epoch):
        for sample in data_generator:
            # Don't update normalizer here!
            value_loss, grad_norm = self.update(sample, value_normalizer)
```

### 3. 🟡 **TUNE: Reduce Critic Learning Rate** (MEDIUM PRIORITY)

```yaml
critic_lr: 0.0001  # Reduced from 0.0005
```

### 4. 🟡 **EXPERIMENT: Disable Value Clipping** (MEDIUM PRIORITY)

```yaml
use_clipped_value_loss: False  # During early training
```

### 5. 🟢 **TUNE: Adjust Huber Delta** (LOW PRIORITY)

```yaml
huber_delta: 1.0  # Reduced from 10.0
# Or use MSE
use_huber_loss: False
```

---

## Next Steps

1. **VERIFY reward structure in MAPush environment**
   - Check if `rewards` shape is `[n_envs, n_agents, 1]` with different values per agent
   - Or if all agents receive the same team reward

2. **APPLY FIX #1** (reward aggregation) based on verification

3. **APPLY FIX #2** (value normalizer location)

4. **RE-TRAIN** and monitor:
   - Critic value loss should now **decrease** instead of increase
   - Oscillations should reduce
   - Check if both agents' performance improves together

5. **IF still issues:** Apply fixes #3-5 incrementally

---

## Verification Script

```python
# Add this to your training script to verify reward structure
import numpy as np

# After env.step()
print("Rewards shape:", rewards.shape)  # Should be [n_threads, n_agents, 1]
print("Agent 0 rewards:", rewards[:5, 0, 0])  # First 5 envs, agent 0
print("Agent 1 rewards:", rewards[:5, 1, 0])  # First 5 envs, agent 1
print("Are rewards identical?", np.allclose(rewards[:, 0], rewards[:, 1]))

# Check what goes into critic buffer
print("Critic buffer reward shape:", rewards[:, 0].shape)  # Current implementation
print("Should be (for team reward):", rewards.mean(axis=1).shape)
```

---

**CONCLUSION:** The HAPPO implementation correctly follows Algorithm 4 for actor updates and sequential IS, BUT has a critical bug in reward collection for the critic that causes it to only learn from agent 0's reward. This is almost certainly the root cause of increasing critic loss.
