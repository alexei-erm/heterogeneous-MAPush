# Current HAPPO Critic Implementation in This Repository

This document details how the HAPPO critic works in this specific codebase (`new-universal-MAPush`), tracing the actual code paths for input, output, and training.

---

## 1. Critic Architecture Overview

**The implementation matches the theory**: A **single centralized VNet** estimates V(s) for all agents.

```
                    ┌─────────────────────────────────────────┐
                    │            VCritic Class                │
                    │   (harl/algorithms/critics/v_critic.py) │
                    └─────────────────────────────────────────┘
                                       │
                                       ▼
                    ┌─────────────────────────────────────────┐
                    │              VNet Model                 │
                    │ (harl/models/value_function_models/     │
                    │                     v_net.py)           │
                    └─────────────────────────────────────────┘
                                       │
                    ┌──────────────────┼──────────────────┐
                    ▼                  ▼                  ▼
              ┌──────────┐      ┌──────────┐      ┌──────────┐
              │  MLPBase │  or  │  CNNBase │  +   │ (RNNLayer)│
              │ [128,128]│      │          │      │ optional  │
              └──────────┘      └──────────┘      └──────────┘
                    │                                  │
                    └────────────────┬─────────────────┘
                                     ▼
                              ┌─────────────┐
                              │  Linear(1)  │  → V(s) scalar
                              │   v_out     │
                              └─────────────┘
```

---

## 2. Critic Input

### 2.1 Input Source: `share_obs` (Global State)

**File**: `harl/runners/on_policy_base_runner.py:448-460`

```python
if self.state_type == "EP":  # Environment-Provided state
    self.critic_buffer.insert(
        share_obs[:, 0],          # [n_envs, share_obs_dim]
        rnn_states_critic,
        values,
        rewards[:, 0],            # Team reward (from agent 0, same for all)
        masks[:, 0],
        bad_masks,
    )
```

For MAPush with EP mode:
- `share_obs` comes from `MAPushEnv.step()` as `state_np = obs_np.copy()`
- Currently uses **local observation as global state** (can be extended)

### 2.2 Input Dimensions (MAPush Mid-Level)

**File**: `harl/envs/mapush/mapush_env.py:81`

```python
self.share_observation_space = [self.env.observation_space] * self.n_agents
```

From MAPush wrapper (`mqe/envs/wrappers/go1_push_mid_wrapper.py`):
- Observation space: `Box(shape=(8,))` or `Box(shape=(9,))` with yaw
- 8-dim obs: `[dx_box, dy_box, dx_target, dy_target, dx_other, dy_other, cos_yaw, sin_yaw]`

### 2.3 Buffer Storage

**File**: `harl/common/buffers/on_policy_critic_buffer_ep.py:32-35`

```python
self.share_obs = np.zeros(
    (self.episode_length + 1, self.n_rollout_threads, *share_obs_shape),
    dtype=np.float32,
)
# Shape: [201, 500, 8] for episode_length=200, n_rollout_threads=500, obs_dim=8
```

---

## 3. Critic Output

### 3.1 Value Function Output

**File**: `harl/models/value_function_models/v_net.py:48-67`

```python
def forward(self, cent_obs, rnn_states, masks):
    """
    Args:
        cent_obs: [batch_size, share_obs_dim]
        rnn_states: [batch_size, recurrent_n, rnn_hidden_size] (if RNN used)
        masks: [batch_size, 1]

    Returns:
        values: [batch_size, 1]  ← Single scalar V(s) per state
        rnn_states: updated RNN states (if RNN used)
    """
    critic_features = self.base(cent_obs)  # MLPBase: [batch, 128]
    if self.use_recurrent_policy:
        critic_features, rnn_states = self.rnn(critic_features, rnn_states, masks)
    values = self.v_out(critic_features)   # Linear: [batch, 1]
    return values, rnn_states
```

### 3.2 Value Storage

**File**: `harl/common/buffers/on_policy_critic_buffer_ep.py:49-51`

```python
self.value_preds = np.zeros(
    (self.episode_length + 1, self.n_rollout_threads, 1), dtype=np.float32
)
# Shape: [201, 500, 1]
```

---

## 4. Returns & Advantage Computation

### 4.1 Returns Computation (GAE)

**File**: `harl/common/buffers/on_policy_critic_buffer_ep.py:97-200`

```python
def compute_returns(self, next_value, value_normalizer=None):
    """Compute GAE returns."""
    if self.use_gae:
        self.value_preds[-1] = next_value
        gae = 0
        for step in reversed(range(self.rewards.shape[0])):
            if value_normalizer is not None:
                # TD error: δ_t = r_t + γ*V(s_{t+1}) - V(s_t)
                delta = (
                    self.rewards[step]
                    + self.gamma * value_normalizer.denormalize(self.value_preds[step + 1])
                    * self.masks[step + 1]
                    - value_normalizer.denormalize(self.value_preds[step])
                )
                # GAE accumulation: A_t = δ_t + (γλ)*A_{t+1}
                gae = delta + self.gamma * self.gae_lambda * self.masks[step + 1] * gae
                # Return: R̂_t = A_t + V(s_t)
                self.returns[step] = gae + value_normalizer.denormalize(self.value_preds[step])
```

**Hyperparameters** (from `happo.yaml`):
- `gamma`: 0.99
- `gae_lambda`: 0.95
- `use_gae`: True
- `use_valuenorm`: True

### 4.2 Advantage Computation

**File**: `harl/runners/on_policy_ha_runner.py:25-33`

```python
def train(self):
    # Compute advantages: A_t = R̂_t - V(s_t)
    if self.value_normalizer is not None:
        advantages = self.critic_buffer.returns[:-1] \
                   - self.value_normalizer.denormalize(self.critic_buffer.value_preds[:-1])
    else:
        advantages = self.critic_buffer.returns[:-1] - self.critic_buffer.value_preds[:-1]

    # Shape: [episode_length, n_rollout_threads, 1] = [200, 500, 1]
```

---

## 5. Critic Training

### 5.1 Training Loop

**File**: `harl/algorithms/critics/v_critic.py:159-200`

```python
def train(self, critic_buffer, value_normalizer=None):
    """Perform a training update using minibatch GD."""
    train_info = {"value_loss": 0, "critic_grad_norm": 0}

    for _ in range(self.critic_epoch):  # 5 epochs by default
        data_generator = critic_buffer.feed_forward_generator_critic(
            self.critic_num_mini_batch  # 1 mini-batch by default
        )

        for sample in data_generator:
            value_loss, critic_grad_norm = self.update(sample, value_normalizer)
            train_info["value_loss"] += value_loss.item()
            train_info["critic_grad_norm"] += critic_grad_norm

    num_updates = self.critic_epoch * self.critic_num_mini_batch  # 5 * 1 = 5
    for k in train_info.keys():
        train_info[k] /= num_updates

    return train_info
```

### 5.2 Critic Update (Single Batch)

**File**: `harl/algorithms/critics/v_critic.py:116-157`

```python
def update(self, sample, value_normalizer=None):
    """Update critic network."""
    (
        share_obs_batch,       # [batch_size, share_obs_dim]
        rnn_states_critic_batch,
        value_preds_batch,     # [batch_size, 1] - old predictions (for clipping)
        return_batch,          # [batch_size, 1] - GAE returns (targets)
        masks_batch,
    ) = sample

    # Forward pass
    values, _ = self.get_values(share_obs_batch, rnn_states_critic_batch, masks_batch)

    # Compute loss
    value_loss = self.cal_value_loss(
        values, value_preds_batch, return_batch, value_normalizer
    )

    # Backward pass
    self.critic_optimizer.zero_grad()
    (value_loss * self.value_loss_coef).backward()  # value_loss_coef = 1.0

    # Gradient clipping
    if self.use_max_grad_norm:
        critic_grad_norm = nn.utils.clip_grad_norm_(
            self.critic.parameters(), self.max_grad_norm  # 10.0
        )

    self.critic_optimizer.step()
    return value_loss, critic_grad_norm
```

### 5.3 Value Loss Function

**File**: `harl/algorithms/critics/v_critic.py:75-114`

```python
def cal_value_loss(self, values, value_preds_batch, return_batch, value_normalizer=None):
    """Calculate clipped Huber loss for value function."""

    # Clipped value predictions (PPO-style)
    value_pred_clipped = value_preds_batch + (values - value_preds_batch).clamp(
        -self.clip_param, self.clip_param  # clip_param = 0.2
    )

    # Normalize targets if using ValueNorm
    if value_normalizer is not None:
        value_normalizer.update(return_batch)
        error_clipped = value_normalizer.normalize(return_batch) - value_pred_clipped
        error_original = value_normalizer.normalize(return_batch) - values
    else:
        error_clipped = return_batch - value_pred_clipped
        error_original = return_batch - values

    # Huber loss (more robust than MSE)
    if self.use_huber_loss:
        value_loss_clipped = huber_loss(error_clipped, self.huber_delta)  # delta=10.0
        value_loss_original = huber_loss(error_original, self.huber_delta)
    else:
        value_loss_clipped = mse_loss(error_clipped)
        value_loss_original = mse_loss(error_original)

    # Take max of clipped and unclipped (conservative update)
    if self.use_clipped_value_loss:
        value_loss = torch.max(value_loss_original, value_loss_clipped)
    else:
        value_loss = value_loss_original

    return value_loss.mean()
```

---

## 6. Sequential Policy Update (HAPPO-Specific)

### 6.1 Factor Initialization

**File**: `harl/runners/on_policy_ha_runner.py:15-23`

```python
def train(self):
    actor_train_infos = []

    # Factor starts at 1.0 for all timesteps/environments
    # This will be modified as agents update sequentially
    factor = np.ones(
        (episode_length, n_rollout_threads, 1),  # [200, 500, 1]
        dtype=np.float32,
    )
```

### 6.2 Sequential Agent Updates

**File**: `harl/runners/on_policy_ha_runner.py:47-124`

```python
    # Random or fixed agent order
    if self.fixed_order:
        agent_order = list(range(self.num_agents))  # [0, 1]
    else:
        agent_order = list(torch.randperm(self.num_agents).numpy())  # e.g., [1, 0]

    for agent_id in agent_order:
        # Store current factor in actor buffer (used during actor training)
        self.actor_buffer[agent_id].update_factor(factor)

        # Get old action log probs BEFORE update
        old_actions_logprob, _, _ = self.actor[agent_id].evaluate_actions(
            obs, rnn_states, actions, masks, available_actions, active_masks
        )

        # Update this actor using advantages * factor
        if self.state_type == "EP":
            actor_train_info = self.actor[agent_id].train(
                self.actor_buffer[agent_id],
                advantages.copy(),  # SAME advantages for all agents
                "EP"
            )

        # Get new action log probs AFTER update
        new_actions_logprob, _, _ = self.actor[agent_id].evaluate_actions(
            obs, rnn_states, actions, masks, available_actions, active_masks
        )

        # Update factor for NEXT agent: M_{next} = (π_new/π_old) * M_{current}
        factor = factor * _t2n(
            torch.prod(  # or mean, based on action_aggregation
                torch.exp(new_actions_logprob - old_actions_logprob),
                dim=-1
            ).reshape(episode_length, n_rollout_threads, 1)
        )

        actor_train_infos.append(actor_train_info)
```

### 6.3 Critic Update (After All Actors)

**File**: `harl/runners/on_policy_ha_runner.py:127-130`

```python
    # Update critic ONCE after all actors have updated
    critic_train_info = self.critic.train(self.critic_buffer, self.value_normalizer)

    return actor_train_infos, critic_train_info
```

---

## 7. Data Flow Diagram

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                              ROLLOUT COLLECTION                              │
└─────────────────────────────────────────────────────────────────────────────┘
                                       │
                                       ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│  MAPushEnv.step(actions)                                                    │
│    │                                                                        │
│    ├── obs_np:    [n_envs, n_agents, obs_dim] = [500, 2, 8]                │
│    ├── state_np:  [n_envs, n_agents, obs_dim] = [500, 2, 8]  (same as obs) │
│    ├── rewards:   [n_envs, n_agents, 1]       = [500, 2, 1]                │
│    └── dones:     [n_envs, n_agents]          = [500, 2]                   │
└─────────────────────────────────────────────────────────────────────────────┘
                                       │
                                       ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│  OnPolicyBaseRunner.insert()                                                │
│    │                                                                        │
│    ├── critic_buffer.insert(                                               │
│    │       share_obs[:, 0],    # [500, 8] - uses agent 0's obs as state   │
│    │       rnn_states_critic,                                              │
│    │       values,             # [500, 1] - V(s) predictions              │
│    │       rewards[:, 0],      # [500, 1] - team reward (agent 0)         │
│    │       masks[:, 0],                                                    │
│    │       bad_masks                                                       │
│    │   )                                                                   │
│    │                                                                        │
│    └── actor_buffer[i].insert(...)  # Per-agent obs, actions, etc.        │
└─────────────────────────────────────────────────────────────────────────────┘
                                       │
                                       ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│  OnPolicyCriticBufferEP  (after episode_length=200 steps)                   │
│    │                                                                        │
│    ├── share_obs:   [201, 500, 8]                                          │
│    ├── value_preds: [201, 500, 1]                                          │
│    ├── rewards:     [200, 500, 1]                                          │
│    ├── returns:     [201, 500, 1]  (computed via GAE)                      │
│    └── masks:       [201, 500, 1]                                          │
└─────────────────────────────────────────────────────────────────────────────┘
                                       │
                                       ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│  compute_returns(next_value)                                                │
│    │                                                                        │
│    │  for step in reversed(range(200)):                                    │
│    │      δ_t = r_t + γ*V(s_{t+1}) - V(s_t)                                │
│    │      gae = δ_t + γλ*gae                                               │
│    │      returns[step] = gae + V(s_t)                                     │
│    │                                                                        │
│    └── Output: returns[200, 500, 1]                                        │
└─────────────────────────────────────────────────────────────────────────────┘
                                       │
                                       ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│  OnPolicyHARunner.train()                                                   │
│    │                                                                        │
│    ├── advantages = returns[:-1] - value_preds[:-1]  # [200, 500, 1]       │
│    │                                                                        │
│    ├── for agent_id in agent_order:                                        │
│    │       actor_buffer[agent_id].update_factor(factor)                    │
│    │       old_logprob = actor.evaluate_actions(...)                       │
│    │       actor.train(actor_buffer, advantages, "EP")                     │
│    │       new_logprob = actor.evaluate_actions(...)                       │
│    │       factor *= exp(new_logprob - old_logprob)  # HAPPO multiplier   │
│    │                                                                        │
│    └── critic.train(critic_buffer, value_normalizer)                       │
└─────────────────────────────────────────────────────────────────────────────┘
                                       │
                                       ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│  VCritic.train(critic_buffer)                                               │
│    │                                                                        │
│    │  for epoch in range(5):  # critic_epoch                               │
│    │      for (share_obs_batch, ..., return_batch, ...) in data_generator: │
│    │          values = VNet(share_obs_batch)  # [batch, 1]                 │
│    │          loss = HuberLoss(values, returns)                            │
│    │          loss.backward()                                              │
│    │          optimizer.step()                                             │
│    │                                                                        │
│    └── Output: {"value_loss": avg_loss, "critic_grad_norm": avg_grad}      │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## 8. Hyperparameters Summary (from `happo.yaml`)

| Parameter | Value | Location |
|-----------|-------|----------|
| **Network** | | |
| `hidden_sizes` | [128, 128] | model |
| `activation_func` | relu | model |
| `initialization_method` | orthogonal_ | model |
| `gain` | 0.01 | model |
| **Training** | | |
| `critic_lr` | 0.001 | model |
| `critic_epoch` | 5 | algo |
| `critic_num_mini_batch` | 1 | algo |
| `n_rollout_threads` | 500 | train |
| `episode_length` | 200 | train |
| **Loss** | | |
| `use_clipped_value_loss` | True | algo |
| `clip_param` | 0.2 | algo |
| `use_huber_loss` | True | algo |
| `huber_delta` | 10.0 | algo |
| `value_loss_coef` | 1.0 | algo |
| **GAE** | | |
| `use_gae` | True | algo |
| `gamma` | 0.99 | algo |
| `gae_lambda` | 0.95 | algo |
| **Optimization** | | |
| `use_max_grad_norm` | True | algo |
| `max_grad_norm` | 10.0 | algo |
| `opti_eps` | 1e-5 | model |
| `weight_decay` | 0 | model |
| `use_valuenorm` | True | train |

---

## 9. Verification: Alignment with Theory

| Theory Requirement | Implementation Status |
|-------------------|----------------------|
| Single centralized critic | **YES** - `VCritic` with single `VNet` |
| Critic input: global state s | **YES** - `share_obs` (currently local obs, can extend) |
| Critic output: V(s) scalar | **YES** - `v_out = Linear(hidden, 1)` |
| Same advantage for all agents | **YES** - `advantages.copy()` passed to each actor |
| Sequential actor updates | **YES** - `for agent_id in agent_order` loop |
| Factor/multiplier update | **YES** - `factor *= exp(new_logprob - old_logprob)` |
| Critic updated after actors | **YES** - `critic.train()` called last in `train()` |
| Team reward (not individual) | **DEPENDS** - uses `rewards[:, 0]` (agent 0's reward) |

---

## 10. Key Code Locations

| Component | File Path |
|-----------|-----------|
| VCritic class | `HARL/harl/algorithms/critics/v_critic.py` |
| VNet model | `HARL/harl/models/value_function_models/v_net.py` |
| Critic buffer (EP) | `HARL/harl/common/buffers/on_policy_critic_buffer_ep.py` |
| HA Runner (sequential update) | `HARL/harl/runners/on_policy_ha_runner.py` |
| Base Runner (data collection) | `HARL/harl/runners/on_policy_base_runner.py` |
| MAPush env wrapper | `HARL/harl/envs/mapush/mapush_env.py` |
| HAPPO config | `HARL/harl/configs/algos_cfgs/happo.yaml` |
