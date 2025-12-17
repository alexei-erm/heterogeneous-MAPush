# Critic Comparison: OpenRL (backup_MAPush) vs HARL (new-universal-MAPush)

This document compares the critic implementations in MAPush's original OpenRL-based setup versus the new HARL-based HAPPO implementation.

---

## Executive Summary

| Aspect | OpenRL (MAPPO) | HARL (HAPPO) |
|--------|----------------|--------------|
| **Critic Architecture** | Single `ValueNetwork` | Single `VCritic` + `VNet` |
| **Input** | `critic_obs` per agent | `share_obs` (EP mode: agent 0's obs) |
| **Output** | V(s) per agent | V(s) per state |
| **Buffer Shape** | `[T, N, M, dim]` | EP: `[T, N, dim]` |
| **Advantage** | Per-agent | Shared (same for all agents) |
| **Policy Update** | Simultaneous | **Sequential with factor** |
| **GAE** | Same formula | Same formula |
| **Value Loss** | Clipped Huber/MSE | Clipped Huber/MSE |

**Key Difference**: OpenRL/MAPPO updates all agents simultaneously with per-agent advantages. HARL/HAPPO uses sequential updates with an importance-weighted factor multiplier.

---

## 1. Network Architecture Comparison

### OpenRL `ValueNetwork`
**File**: `openrl/modules/networks/value_network.py`

```python
class ValueNetwork(BaseValueNetwork):
    def __init__(self, cfg, input_space, ...):
        # Base feature extractor
        self.base = MLPBase(cfg, critic_obs_shape) if not image else CNNBase(...)

        # Optional RNN
        if self._use_recurrent_policy:
            self.rnn = RNNLayer(input_size, hidden_size, recurrent_N)

        # Output head
        if self._use_popart:
            self.v_out = PopArt(input_size, 1)  # Adaptive normalization
        else:
            self.v_out = nn.Linear(input_size, 1)

    def forward(self, critic_obs, rnn_states, masks):
        critic_features = self.base(critic_obs)
        if self._use_recurrent_policy:
            critic_features, rnn_states = self.rnn(critic_features, rnn_states, masks)
        values = self.v_out(critic_features)
        return values, rnn_states
```

### HARL `VNet`
**File**: `harl/models/value_function_models/v_net.py`

```python
class VNet(nn.Module):
    def __init__(self, args, cent_obs_space, device):
        # Base feature extractor
        self.base = MLPBase(args, cent_obs_shape) if not image else CNNBase(...)

        # Optional RNN
        if self.use_recurrent_policy:
            self.rnn = RNNLayer(hidden_sizes[-1], hidden_sizes[-1], recurrent_n)

        # Output head (no PopArt option - uses ValueNorm externally)
        self.v_out = init_(nn.Linear(hidden_sizes[-1], 1))

    def forward(self, cent_obs, rnn_states, masks):
        critic_features = self.base(cent_obs)
        if self.use_recurrent_policy:
            critic_features, rnn_states = self.rnn(critic_features, rnn_states, masks)
        values = self.v_out(critic_features)
        return values, rnn_states
```

**Differences**:
- OpenRL supports PopArt (built-in normalization); HARL uses external `ValueNorm`
- Architecture is otherwise nearly identical

---

## 2. Buffer & Data Storage

### OpenRL `ReplayData`
**File**: `openrl/buffers/replay_data.py`

```python
# Buffer shapes include agent dimension
self.critic_obs = np.zeros(
    (episode_length + 1, n_rollout_threads, num_agents, *critic_obs_shape),
    dtype=np.float32
)  # Shape: [T+1, N, M, obs_dim]

self.value_preds = np.zeros(
    (episode_length + 1, n_rollout_threads, num_agents, 1),
    dtype=np.float32
)  # Shape: [T+1, N, M, 1]

self.rewards = np.zeros(
    (episode_length, n_rollout_threads, num_agents, 1),
    dtype=np.float32
)  # Shape: [T, N, M, 1]
```

### HARL `OnPolicyCriticBufferEP`
**File**: `harl/common/buffers/on_policy_critic_buffer_ep.py`

```python
# EP mode: NO agent dimension - single value for entire state
self.share_obs = np.zeros(
    (episode_length + 1, n_rollout_threads, *share_obs_shape),
    dtype=np.float32
)  # Shape: [T+1, N, obs_dim]

self.value_preds = np.zeros(
    (episode_length + 1, n_rollout_threads, 1),
    dtype=np.float32
)  # Shape: [T+1, N, 1]

self.rewards = np.zeros(
    (episode_length, n_rollout_threads, 1),
    dtype=np.float32
)  # Shape: [T, N, 1]
```

**Key Difference**:
- OpenRL stores per-agent data: `[T, N, M, dim]`
- HARL EP mode stores per-state data: `[T, N, dim]` (single value for all agents)

---

## 3. Data Insertion

### OpenRL
**File**: `openrl/buffers/replay_data.py:245-284`

```python
def insert(self, raw_obs, rnn_states, rnn_states_critic, actions,
           action_log_probs, value_preds, rewards, masks, ...):
    critic_obs = get_critic_obs(raw_obs)  # Extract critic observation
    self.critic_obs[self.step + 1] = critic_obs.copy()
    self.value_preds[self.step] = value_preds.copy()  # Per-agent values
    self.rewards[self.step] = rewards.copy()          # Per-agent rewards
    self.masks[self.step + 1] = masks.copy()
```

### HARL (EP mode)
**File**: `harl/runners/on_policy_base_runner.py:448-456`

```python
if self.state_type == "EP":
    self.critic_buffer.insert(
        share_obs[:, 0],      # Only agent 0's observation as global state
        rnn_states_critic,
        values,               # Single value per state
        rewards[:, 0],        # Only agent 0's reward (team reward)
        masks[:, 0],
        bad_masks,
    )
```

**Key Difference**:
- OpenRL inserts all agents' data
- HARL EP mode uses agent 0's data as representative (assumes homogeneous/team reward)

---

## 4. GAE Returns Computation

### OpenRL
**File**: `openrl/buffers/replay_data.py:320-423`

```python
def compute_returns(self, next_value, value_normalizer=None):
    if self._use_gae:
        self.value_preds[-1] = next_value
        gae = 0
        for step in reversed(range(self.rewards.shape[0])):
            if self._use_valuenorm and value_normalizer is not None:
                delta = (
                    self.rewards[step]
                    + self.gamma * value_normalizer.denormalize(self.value_preds[step + 1])
                    * self.masks[step + 1]
                    - value_normalizer.denormalize(self.value_preds[step])
                )
                gae = delta + self.gamma * self.gae_lambda * gae * self.masks[step + 1]
                gae = gae * self.bad_masks[step + 1]  # Handle truncation
                self.returns[step] = gae + value_normalizer.denormalize(self.value_preds[step])
```

### HARL
**File**: `harl/common/buffers/on_policy_critic_buffer_ep.py:97-200`

```python
def compute_returns(self, next_value, value_normalizer=None):
    if self.use_gae:
        self.value_preds[-1] = next_value
        gae = 0
        for step in reversed(range(self.rewards.shape[0])):
            if value_normalizer is not None:
                delta = (
                    self.rewards[step]
                    + self.gamma * value_normalizer.denormalize(self.value_preds[step + 1])
                    * self.masks[step + 1]
                    - value_normalizer.denormalize(self.value_preds[step])
                )
                gae = delta + self.gamma * self.gae_lambda * self.masks[step + 1] * gae
                gae = self.bad_masks[step + 1] * gae  # Handle truncation
                self.returns[step] = gae + value_normalizer.denormalize(self.value_preds[step])
```

**Result**: Identical GAE formula

---

## 5. Advantage Computation

### OpenRL (in PPOAlgorithm.train_ppo)
**File**: `openrl/algorithms/ppo.py:394-409`

```python
def train_ppo(self, buffer, turn_on):
    if self._use_valuenorm and value_normalizer is not None:
        advantages = buffer.returns[:-1] - value_normalizer.denormalize(
            buffer.value_preds[:-1]
        )
    else:
        advantages = buffer.returns[:-1] - buffer.value_preds[:-1]

    # Normalize advantages (with active mask handling)
    advantages_copy = advantages.copy()
    advantages_copy[buffer.active_masks[:-1] == 0.0] = np.nan
    mean_advantages = np.nanmean(advantages_copy)
    std_advantages = np.nanstd(advantages_copy)
    advantages = (advantages - mean_advantages) / (std_advantages + 1e-5)
```

### HARL (in OnPolicyHARunner.train)
**File**: `harl/runners/on_policy_ha_runner.py:25-45`

```python
def train(self):
    # Compute advantages
    if self.value_normalizer is not None:
        advantages = self.critic_buffer.returns[:-1] \
                   - self.value_normalizer.denormalize(self.critic_buffer.value_preds[:-1])
    else:
        advantages = self.critic_buffer.returns[:-1] - self.critic_buffer.value_preds[:-1]

    # For FP mode, normalize with active masks (similar to OpenRL)
    if self.state_type == "FP":
        active_masks_array = np.stack([self.actor_buffer[i].active_masks for i in range(self.num_agents)], axis=2)
        advantages_copy = advantages.copy()
        advantages_copy[active_masks_array[:-1] == 0.0] = np.nan
        mean_advantages = np.nanmean(advantages_copy)
        std_advantages = np.nanstd(advantages_copy)
        advantages = (advantages - mean_advantages) / (std_advantages + 1e-5)
```

**Result**: Nearly identical advantage computation

---

## 6. Policy Update (THE KEY DIFFERENCE)

### OpenRL (MAPPO) - Simultaneous Update
**File**: `openrl/algorithms/ppo.py:238-361`

```python
def prepare_loss(self, critic_obs_batch, obs_batch, ...):
    # Get values and action log probs for ALL agents at once
    values, action_log_probs, dist_entropy, policy_values = self.algo_module.evaluate_actions(
        critic_obs_batch, obs_batch, rnn_states_batch, ...
    )

    # PPO ratio and surrogate loss
    ratio = torch.exp(action_log_probs - old_action_log_probs_batch)
    surr1 = ratio * adv_targ
    surr2 = torch.clamp(ratio, 1.0 - self.clip_param, 1.0 + self.clip_param) * adv_targ
    policy_loss = -torch.min(surr1, surr2).mean()

    # Value loss
    value_loss = self.cal_value_loss(value_normalizer, values, value_preds_batch, return_batch, ...)

    # Both losses computed together, backprop together
    return [policy_loss, value_loss], ...
```

**All agents updated simultaneously with same gradient step**

### HARL (HAPPO) - Sequential Update with Factor
**File**: `harl/runners/on_policy_ha_runner.py:47-130`

```python
def train(self):
    # Initialize factor (importance weight multiplier)
    factor = np.ones((episode_length, n_rollout_threads, 1), dtype=np.float32)

    # Random or fixed agent order
    agent_order = list(torch.randperm(self.num_agents).numpy())  # e.g., [1, 0]

    for agent_id in agent_order:
        # Store current factor for this agent's update
        self.actor_buffer[agent_id].update_factor(factor)

        # Get OLD action log probs BEFORE update
        old_actions_logprob, _, _ = self.actor[agent_id].evaluate_actions(...)

        # Update THIS actor using advantages * factor
        actor_train_info = self.actor[agent_id].train(
            self.actor_buffer[agent_id],
            advantages.copy(),  # Same advantage, but factor is in buffer
            "EP"
        )

        # Get NEW action log probs AFTER update
        new_actions_logprob, _, _ = self.actor[agent_id].evaluate_actions(...)

        # UPDATE FACTOR for next agent
        # factor *= exp(new_log_prob - old_log_prob)
        factor = factor * _t2n(
            torch.prod(
                torch.exp(new_actions_logprob - old_actions_logprob),
                dim=-1
            ).reshape(episode_length, n_rollout_threads, 1)
        )

    # Critic updated ONCE after all actors
    critic_train_info = self.critic.train(self.critic_buffer, self.value_normalizer)
```

**Agents updated sequentially, each seeing the cumulative policy change of previous agents**

---

## 7. Value Loss Computation

### OpenRL
**File**: `openrl/algorithms/ppo.py:178-220`

```python
def cal_value_loss(self, value_normalizer, values, value_preds_batch, return_batch, active_masks_batch):
    # Clipped value predictions
    value_pred_clipped = value_preds_batch + (values - value_preds_batch).clamp(
        -self.clip_param, self.clip_param
    )

    if self._use_valuenorm and value_normalizer is not None:
        value_normalizer.update(return_batch)
        error_clipped = value_normalizer.normalize(return_batch) - value_pred_clipped
        error_original = value_normalizer.normalize(return_batch) - values
    else:
        error_clipped = return_batch - value_pred_clipped
        error_original = return_batch - values

    if self._use_huber_loss:
        value_loss_clipped = huber_loss(error_clipped, self.huber_delta)
        value_loss_original = huber_loss(error_original, self.huber_delta)
    else:
        value_loss_clipped = mse_loss(error_clipped)
        value_loss_original = mse_loss(error_original)

    if self._use_clipped_value_loss:
        value_loss = torch.max(value_loss_original, value_loss_clipped)
    else:
        value_loss = value_loss_original

    return value_loss.mean()
```

### HARL
**File**: `harl/algorithms/critics/v_critic.py:75-114`

```python
def cal_value_loss(self, values, value_preds_batch, return_batch, value_normalizer=None):
    # Clipped value predictions
    value_pred_clipped = value_preds_batch + (values - value_preds_batch).clamp(
        -self.clip_param, self.clip_param
    )

    if value_normalizer is not None:
        value_normalizer.update(return_batch)
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

**Result**: Identical value loss computation

---

## 8. Summary Table

| Component | OpenRL (MAPPO) | HARL (HAPPO) |
|-----------|----------------|--------------|
| **Critic Network** | `ValueNetwork` (supports PopArt) | `VNet` (uses external ValueNorm) |
| **Buffer Dim** | `[T, N, M, dim]` (per-agent) | `[T, N, dim]` (per-state, EP mode) |
| **Critic Input** | `critic_obs` per agent | `share_obs[:, 0]` (agent 0 only) |
| **Value Output** | Per-agent V(s) | Single V(s) per state |
| **Reward Input** | Per-agent rewards | Team reward (agent 0) |
| **GAE Formula** | Standard | Standard (identical) |
| **Advantage Norm** | Yes, with active masks | Yes, with active masks (FP mode) |
| **Value Loss** | Clipped Huber/MSE | Clipped Huber/MSE (identical) |
| **Actor Update** | **Simultaneous** | **Sequential with factor** |
| **Credit Assignment** | Implicit | **Explicit via importance weighting** |

---

## 9. Implications for MAPush

1. **Team Reward Assumption**: HARL EP mode assumes all agents receive the same team reward. This is correct for cooperative tasks like MAPush.

2. **Sequential Update Benefit**: HAPPO's sequential update provides better credit assignment than MAPPO's simultaneous update, theoretically leading to better coordination.

3. **State Representation**: Currently, HARL uses agent 0's local observation as `share_obs`. For better critic performance, this could be extended to use a true global state.

4. **Compatibility**: Both use identical GAE and value loss formulas, so the core critic learning is equivalent. The difference is in how advantages are used during policy updates.
