# Comprehensive Critic Comparison: OpenRL (backup_MAPush) vs HARL HAPPO

> **Generated:** December 19, 2025
> **Purpose:** Deep analysis of critic implementations to identify differences that may impact training stability and convergence

---

## Table of Contents

1. [Executive Summary](#1-executive-summary)
2. [Model Architecture](#2-model-architecture)
3. [Input/Output Dimensions](#3-inputoutput-dimensions)
4. [Training Loop & Update Method](#4-training-loop--update-method)
5. [Loss Functions](#5-loss-functions)
6. [Value Normalization](#6-value-normalization)
7. [GAE & Return Computation](#7-gae--return-computation)
8. [Hyperparameters Comparison](#8-hyperparameters-comparison)
9. [Buffer Implementation](#9-buffer-implementation)
10. [Key Differences Summary](#10-key-differences-summary)
11. [Potential Issues & Recommendations](#11-potential-issues--recommendations)

---

## 1. Executive Summary

| Aspect | OpenRL (backup_MAPush) | HARL HAPPO |
|--------|------------------------|------------|
| **Framework** | OpenRL library | HARL library |
| **Critic Type** | Integrated with PPO | Separate VCritic class |
| **Architecture** | Single hidden layer (64 units) | 3 hidden layers (256, 256, 128) |
| **Value Norm Update** | Inside mini-batch loop | Once before training loop (FIXED) |
| **Actor/Critic Training** | Joint (same loop) | Separate (different epochs) |
| **Critic Epochs** | Same as ppo_epoch (10) | 25 (5x more than actor) |
| **Value Loss Coef** | 0.5 | 5.0 (10x higher) |
| **Learning Rates** | Equal (5e-3 both) | Different (critic 5x higher) |

---

## 2. Model Architecture

### 2.1 OpenRL ValueNetwork

**File:** `/home/gvlab/miniconda3/envs/mapush/lib/python3.8/site-packages/openrl/modules/networks/value_network.py`

```python
class ValueNetwork(BaseValueNetwork):
    def __init__(self, cfg, input_space, ...):
        self.hidden_size = cfg.hidden_size  # Default: 64

        # Base network: MLPBase or CNNBase
        self.base = MLPBase(cfg, critic_obs_shape, use_attn_internal=True)

        # Optional RNN layer
        if self._use_recurrent_policy:
            self.rnn = RNNLayer(input_size, self.hidden_size, ...)

        # Output head
        if self._use_popart:
            self.v_out = PopArt(input_size, 1, device=device)
        else:
            self.v_out = nn.Linear(input_size, 1)
```

**MLPBase Structure (OpenRL):**
```python
class MLPBase:
    def __init__(self, cfg, obs_shape):
        self._layer_N = cfg.layer_N  # Default: 1

        # Optional feature normalization
        if self._use_feature_normalization:
            self.feature_norm = nn.LayerNorm(obs_dim)

        # MLP layers
        self.mlp = MLPLayer(obs_dim, hidden_size, layer_N, ...)
```

**MLPLayer Structure (OpenRL):**
```python
class MLPLayer:
    def __init__(self, input_dim, hidden_size, layer_N, ...):
        # First layer
        self.fc1 = nn.Sequential(
            nn.Linear(input_dim, hidden_size),
            activation,          # ReLU default
            nn.LayerNorm(hidden_size)
        )

        # Hidden layers (if layer_N > 1)
        if self._layer_N > 1:
            self.fc_h = nn.Sequential(
                nn.Linear(hidden_size, hidden_size),
                activation,
                nn.LayerNorm(hidden_size)
            )
            self.fc2 = get_clones(self.fc_h, layer_N - 1)

        # Final layer (no activation, just LayerNorm)
        self.fc3 = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.LayerNorm(hidden_size)
        )
```

**OpenRL Architecture Summary:**
```
Input (obs_dim)
    -> LayerNorm (if use_feature_normalization)
    -> Linear(obs_dim, 64) + ReLU + LayerNorm
    -> [Optional: Linear(64, 64) + ReLU + LayerNorm] x (layer_N - 1)
    -> Linear(64, 64) + LayerNorm
    -> [Optional: RNN(64, 64)]
    -> Linear(64, 1) -> Value
```

---

### 2.2 HARL VNet

**File:** `/home/gvlab/new-universal-MAPush/HARL/harl/models/value_function_models/v_net.py`

```python
class VNet(nn.Module):
    def __init__(self, args, cent_obs_space, device):
        self.hidden_sizes = args["hidden_sizes"]  # [256, 256, 128]

        # Base network
        self.base = MLPBase(args, cent_obs_shape)

        # Optional RNN
        if self.use_recurrent_policy:
            self.rnn = RNNLayer(hidden_sizes[-1], hidden_sizes[-1], ...)

        # Output head
        self.v_out = nn.Linear(hidden_sizes[-1], 1)  # Linear(128, 1)
```

**MLPBase Structure (HARL):**
```python
class MLPBase:
    def __init__(self, args, obs_shape):
        self.hidden_sizes = args["hidden_sizes"]  # [256, 256, 128]

        # Feature normalization
        if self.use_feature_normalization:
            self.feature_norm = nn.LayerNorm(obs_dim)

        # MLP layers
        self.mlp = MLPLayer(obs_dim, hidden_sizes, ...)
```

**MLPLayer Structure (HARL):**
```python
class MLPLayer:
    def __init__(self, input_dim, hidden_sizes, ...):
        # Build layers dynamically
        layers = [
            nn.Linear(input_dim, hidden_sizes[0]),
            activation,          # ReLU
            nn.LayerNorm(hidden_sizes[0])
        ]

        for i in range(1, len(hidden_sizes)):
            layers += [
                nn.Linear(hidden_sizes[i-1], hidden_sizes[i]),
                activation,
                nn.LayerNorm(hidden_sizes[i])
            ]

        self.fc = nn.Sequential(*layers)
```

**HARL Architecture Summary:**
```
Input (17 dims for MAPush 2-agent)
    -> LayerNorm(17)
    -> Linear(17, 256) + ReLU + LayerNorm(256)
    -> Linear(256, 256) + ReLU + LayerNorm(256)
    -> Linear(256, 128) + ReLU + LayerNorm(128)
    -> Linear(128, 1) -> Value
```

---

### 2.3 Architecture Comparison Table

| Component | OpenRL | HARL |
|-----------|--------|------|
| **Hidden Sizes** | [64] (single layer) | [256, 256, 128] |
| **Total Hidden Layers** | 1-2 (configurable) | 3 (fixed by config) |
| **Total Parameters (17-dim input)** | ~4,353 | ~134,401 |
| **Feature Normalization** | LayerNorm (optional, default False) | LayerNorm (enabled) |
| **Inter-layer Normalization** | LayerNorm after each layer | LayerNorm after each layer |
| **Activation** | ReLU (activation_id=1) | ReLU |
| **Initialization** | Orthogonal | Orthogonal |
| **Output Layer Init** | Standard | gain=0.01 |
| **PopArt Support** | Yes (optional) | No |
| **Influence Policy** | Yes (optional) | No |

**Parameter Count Calculation:**

*OpenRL (64 hidden, 1 layer):*
```
Layer 1: 17 * 64 + 64 = 1,152
Layer 2: 64 * 64 + 64 = 4,160 (fc3)
Output:  64 * 1 + 1 = 65
LayerNorms: 64*2 + 64*2 = 256
Total: ~5,633 parameters
```

*HARL (256, 256, 128):*
```
Layer 1: 17 * 256 + 256 = 4,608
Layer 2: 256 * 256 + 256 = 65,792
Layer 3: 256 * 128 + 128 = 32,896
Output:  128 * 1 + 1 = 129
LayerNorms: 17*2 + 256*2 + 256*2 + 128*2 = 1,314
Total: ~104,739 parameters
```

**HARL critic has ~19x more parameters than OpenRL critic!**

---

## 3. Input/Output Dimensions

### 3.1 OpenRL Critic Input

OpenRL uses the standard observation as critic input (not a separate global state):

```python
# From openrl/buffers/utils/util.py
def get_critic_obs_space(input_space):
    # Returns the observation space for critic
    # In MAPush, this is the agent's local observation
```

**Input Dimension:** Agent's local observation (varies, likely 8 dims in MAPush)

### 3.2 HARL Critic Input (Global State)

HARL uses a **centralized global state** for the critic (CTDE paradigm):

**File:** `/home/gvlab/new-universal-MAPush/HARL/harl/envs/mapush/mapush_env.py`

```python
def _construct_global_state(self) -> np.ndarray:
    """
    Global state = [box state] + [target state] + [all agent states]

    For 2-agent MAPush:
    - Box: [x, y, yaw] = 3 dims
    - Target: [x, y] = 2 dims (NO yaw!)
    - Agent 0: [x, y, yaw, vx, vy, vyaw] = 6 dims
    - Agent 1: [x, y, yaw, vx, vy, vyaw] = 6 dims

    TOTAL: 17 dims
    """
```

**Input Dimension:** 17 (global state for 2-agent MAPush)

### 3.3 Comparison

| Aspect | OpenRL | HARL |
|--------|--------|------|
| **Input Type** | Local observation | Global state (centralized) |
| **Input Dim (2-agent)** | ~8 (local obs) | 17 (global state) |
| **Contains Velocities** | Depends on obs | Yes (vx, vy, vyaw for each agent) |
| **Contains Other Agents** | No | Yes (all agent positions/velocities) |
| **Contains Task Objects** | Partial | Full (box + target) |
| **CTDE Paradigm** | No (decentralized) | Yes (centralized critic) |

---

## 4. Training Loop & Update Method

### 4.1 OpenRL Training Loop

**File:** `/home/gvlab/miniconda3/envs/mapush/lib/python3.8/site-packages/openrl/algorithms/ppo.py`

```python
def train_ppo(self, buffer, turn_on):
    # Compute advantages
    if self._use_valuenorm:
        advantages = buffer.returns[:-1] - value_normalizer.denormalize(
            buffer.value_preds[:-1]
        )

    # Normalize advantages (ALWAYS done, with active_masks)
    advantages_copy = advantages.copy()
    advantages_copy[buffer.active_masks[:-1] == 0.0] = np.nan
    mean_advantages = np.nanmean(advantages_copy)
    std_advantages = np.nanstd(advantages_copy)
    advantages = (advantages - mean_advantages) / (std_advantages + 1e-5)

    # Training loop
    for _ in range(self.ppo_epoch):  # Default: 10
        data_generator = self.get_data_generator(buffer, advantages)

        for sample in data_generator:
            # JOINT update: actor AND critic in same call
            value_loss, critic_grad_norm, policy_loss, ... = self.ppo_update(sample)
```

**Key Characteristics:**
- Actor and critic updated together in `ppo_update()`
- Same number of epochs for both (ppo_epoch)
- Advantages ALWAYS normalized with active_masks
- Value normalizer updated INSIDE `cal_value_loss()`

### 4.2 HARL Training Loop

**File:** `/home/gvlab/new-universal-MAPush/HARL/harl/runners/on_policy_ha_runner.py`

```python
def train(self, episode=None):
    # Compute advantages
    if self.value_normalizer is not None:
        advantages = self.critic_buffer.returns[:-1] - \
            self.value_normalizer.denormalize(self.critic_buffer.value_preds[:-1])

    # CRITIC FIX 3: Only update actors every N iterations
    actor_update_interval = self.algo_args["algo"].get("actor_update_interval", 1)
    should_update_actors = (episode % actor_update_interval == 0)

    # Update actors (if on correct iteration)
    if should_update_actors:
        for agent_id in agent_order:
            actor_train_info = self.actor[agent_id].train(...)

    # ALWAYS update critic (separate from actors)
    critic_train_info = self.critic.train(self.critic_buffer, self.value_normalizer)
```

**VCritic.train():**
```python
def train(self, critic_buffer, value_normalizer=None):
    # FIX (Dec 18, 2025): Update value normalizer ONCE before training loop
    if value_normalizer is not None:
        all_returns = critic_buffer.returns[:-1].reshape(-1, 1)
        value_normalizer.update(all_returns)

    for _ in range(self.critic_epoch):  # 25 epochs
        data_generator = critic_buffer.feed_forward_generator_critic(...)

        for sample in data_generator:
            value_loss, critic_grad_norm = self.update(sample, value_normalizer)
```

**Key Characteristics:**
- Actor and critic trained SEPARATELY
- Different epoch counts (actor: 5, critic: 25)
- Actor updates can be slowed down (actor_update_interval=3)
- Value normalizer updated ONCE before training loop (FIXED)

### 4.3 Training Loop Comparison

| Aspect | OpenRL | HARL |
|--------|--------|------|
| **Actor/Critic Coupling** | Joint (same update call) | Separate (different methods) |
| **Actor Epochs** | ppo_epoch (10) | ppo_epoch (5) |
| **Critic Epochs** | ppo_epoch (10) | critic_epoch (25) |
| **Actor Update Frequency** | Every iteration | Every 3rd iteration |
| **Advantage Normalization** | Always (with active_masks) | Not applied |
| **Value Norm Update** | Inside mini-batch loop | Once before training |

---

## 5. Loss Functions

### 5.1 OpenRL Value Loss

```python
def cal_value_loss(self, value_normalizer, values, value_preds_batch,
                   return_batch, active_masks_batch):
    # Clipped value prediction
    value_pred_clipped = value_preds_batch + (values - value_preds_batch).clamp(
        -self.clip_param, self.clip_param
    )

    # Normalize returns (AND UPDATE NORMALIZER!)
    if self._use_valuenorm and value_normalizer is not None:
        value_normalizer.update(return_batch)  # <-- UPDATES HERE (inside loop!)
        error_clipped = value_normalizer.normalize(return_batch) - value_pred_clipped
        error_original = value_normalizer.normalize(return_batch) - values
    else:
        error_clipped = return_batch - value_pred_clipped
        error_original = return_batch - values

    # Loss computation
    if self._use_huber_loss:
        value_loss_clipped = huber_loss(error_clipped, self.huber_delta)
        value_loss_original = huber_loss(error_original, self.huber_delta)
    else:
        value_loss_clipped = mse_loss(error_clipped)
        value_loss_original = mse_loss(error_original)

    # Clipped loss
    if self._use_clipped_value_loss:
        value_loss = torch.max(value_loss_original, value_loss_clipped)
    else:
        value_loss = value_loss_original

    # Apply active masks
    if self._use_value_active_masks:
        value_loss = (value_loss * active_masks_batch).sum() / active_masks_batch.sum()
    else:
        value_loss = value_loss.mean()

    return value_loss
```

### 5.2 HARL Value Loss

```python
def cal_value_loss(self, values, value_preds_batch, return_batch, value_normalizer=None):
    # Clipped value prediction
    value_pred_clipped = value_preds_batch + (values - value_preds_batch).clamp(
        -self.clip_param, self.clip_param
    )

    # Normalize returns (NO UPDATE - done before loop!)
    if value_normalizer is not None:
        # FIX: Removed value_normalizer.update() from here
        error_clipped = value_normalizer.normalize(return_batch) - value_pred_clipped
        error_original = value_normalizer.normalize(return_batch) - values
    else:
        error_clipped = return_batch - value_pred_clipped
        error_original = return_batch - values

    # Loss computation
    if self.use_huber_loss:
        value_loss_clipped = huber_loss(error_clipped, self.huber_delta)
        value_loss_original = huber_loss(error_original, self.huber_delta)
    else:
        value_loss_clipped = mse_loss(error_clipped)
        value_loss_original = mse_loss(error_original)

    # Clipped loss
    if self.use_clipped_value_loss:
        value_loss = torch.max(value_loss_original, value_loss_clipped)
    else:
        value_loss = value_loss_original

    return value_loss.mean()  # Simple mean, no active masks
```

### 5.3 Loss Function Comparison

| Aspect | OpenRL | HARL |
|--------|--------|------|
| **Value Clipping** | Yes (clip_param=0.2) | Yes (clip_param=0.2) |
| **Huber Loss** | Yes (huber_delta=10.0) | Yes (huber_delta=10.0) |
| **Clipped Loss (max)** | Yes | Yes |
| **Active Masks** | Yes (use_value_active_masks=True) | No |
| **Value Normalizer Update** | Inside cal_value_loss() | Before training loop |
| **Final Aggregation** | Weighted mean (active_masks) | Simple mean |

---

## 6. Value Normalization

### 6.1 OpenRL ValueNorm

OpenRL uses the same `ValueNorm` concept but updates it **inside the mini-batch loop**:

```python
# Inside cal_value_loss():
if self._use_valuenorm and value_normalizer is not None:
    value_normalizer.update(return_batch)  # Called every mini-batch!
```

**Problem:** With 10 epochs and 1 mini-batch, the normalizer is updated 10 times per rollout. With shuffled data, this causes non-stationary training targets.

### 6.2 HARL ValueNorm

**File:** `/home/gvlab/new-universal-MAPush/HARL/harl/common/valuenorm.py`

```python
class ValueNorm(nn.Module):
    def __init__(self, input_shape, beta=0.99999, ...):
        self.running_mean = nn.Parameter(torch.zeros(input_shape), requires_grad=False)
        self.running_mean_sq = nn.Parameter(torch.zeros(input_shape), requires_grad=False)
        self.debiasing_term = nn.Parameter(torch.tensor(0.0), requires_grad=False)
        self.beta = beta  # EMA coefficient

    def update(self, input_vector):
        batch_mean = input_vector.mean(...)
        batch_sq_mean = (input_vector**2).mean(...)

        # Exponential moving average
        self.running_mean = weight * self.running_mean + (1 - weight) * batch_mean
        self.running_mean_sq = weight * self.running_mean_sq + (1 - weight) * batch_sq_mean

    def normalize(self, input_vector):
        mean, var = self.running_mean_var()
        return (input_vector - mean) / torch.sqrt(var)

    def denormalize(self, input_vector):
        mean, var = self.running_mean_var()
        return input_vector * torch.sqrt(var) + mean
```

**HARL FIX (Dec 18, 2025):** Value normalizer updated **once before training loop**:

```python
# In VCritic.train():
if value_normalizer is not None:
    all_returns = critic_buffer.returns[:-1].reshape(-1, 1)
    value_normalizer.update(all_returns)  # ONCE with all returns

for _ in range(self.critic_epoch):
    # Training loop - normalizer NOT updated inside
```

### 6.3 Value Normalization Comparison

| Aspect | OpenRL | HARL |
|--------|--------|------|
| **Normalization Method** | EMA | EMA |
| **Beta (EMA coefficient)** | 0.99999 | 0.99999 |
| **Update Frequency** | Every mini-batch (10x/rollout) | Once per rollout |
| **Target Stationarity** | Non-stationary | Stationary (FIXED) |
| **Update Location** | Inside cal_value_loss() | Before training loop |

---

## 7. GAE & Return Computation

### 7.1 OpenRL GAE

```python
# From openrl/buffers/normal_buffer.py
def compute_returns(self, next_value, value_normalizer):
    if self._use_gae:
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
            else:
                delta = ...

            gae = delta + self.gamma * self.gae_lambda * self.masks[step + 1] * gae
            self.returns[step] = gae + value_normalizer.denormalize(self.value_preds[step])
```

### 7.2 HARL GAE

```python
# From harl/common/buffers/on_policy_critic_buffer_ep.py
def compute_returns(self, next_value, value_normalizer=None):
    if self.use_proper_time_limits:  # Handle truncation vs termination
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
                    gae = self.bad_masks[step + 1] * gae  # Handle truncation!
                    self.returns[step] = gae + value_normalizer.denormalize(
                        self.value_preds[step]
                    )
```

### 7.3 GAE Comparison

| Aspect | OpenRL | HARL |
|--------|--------|------|
| **GAE Used** | Yes | Yes |
| **Gamma** | 0.99 | 0.99 |
| **Lambda** | 0.95 | 0.95 |
| **Proper Time Limits** | No (default False) | Yes (use_proper_time_limits=True) |
| **Bad Masks (Truncation)** | No | Yes |
| **Bootstrap on Truncation** | No | Yes |

---

## 8. Hyperparameters Comparison

### 8.1 Full Hyperparameter Table

| Parameter | OpenRL (ppo.yaml) | HARL (happo.yaml) | Notes |
|-----------|-------------------|-------------------|-------|
| **lr** (actor) | 5e-3 | 1e-3 | OpenRL 5x higher |
| **critic_lr** | 5e-3 | 5e-3 | Same |
| **Critic/Actor LR Ratio** | 1.0 | 5.0 | HARL prioritizes critic |
| **episode_length** | 200 | 200 | Same |
| **ppo_epoch** | 10 (default) | 5 | Actor epochs |
| **critic_epoch** | (same as ppo_epoch) | 25 | HARL: 5x more critic epochs |
| **actor_update_interval** | 1 | 3 | HARL slows actor updates |
| **value_loss_coef** | 0.5 | 5.0 | HARL 10x higher |
| **entropy_coef** | 0.01 | 0.01 | Same |
| **clip_param** | 0.2 | 0.2 | Same |
| **num_mini_batch** | 1 | 1 | Same |
| **use_gae** | True | True | Same |
| **gamma** | 0.99 | 0.99 | Same |
| **gae_lambda** | 0.95 | 0.95 | Same |
| **use_huber_loss** | True | True | Same |
| **huber_delta** | 10.0 | 10.0 | Same |
| **use_clipped_value_loss** | True | True | Same |
| **max_grad_norm** | 10.0 | 10.0 | Same |
| **use_valuenorm** | True | True | Same |
| **use_adv_normalize** | False | N/A | OpenRL has option |
| **use_recurrent_policy** | False | False | Same |
| **hidden_size** | 64 | [256, 256, 128] | HARL much larger |
| **layer_N** | 1 | 3 | HARL deeper |
| **activation** | ReLU (id=1) | ReLU | Same |
| **use_feature_normalization** | False (default) | True | HARL normalizes inputs |
| **use_orthogonal** | True | True | Same (orthogonal init) |
| **gain** | 0.01 | 0.01 | Same |
| **n_rollout_threads** | varies | 500 | HARL uses 500 envs |
| **use_proper_time_limits** | False | True | HARL handles truncation |

---

## 9. Buffer Implementation

### 9.1 OpenRL Buffer

**File:** `/home/gvlab/miniconda3/envs/mapush/lib/python3.8/site-packages/openrl/buffers/normal_buffer.py`

```python
class NormalReplayBuffer:
    def __init__(self, cfg, ...):
        # Stores both actor and critic data
        self.obs = np.zeros((T+1, N, num_agents, *obs_shape))
        self.critic_obs = np.zeros((T+1, N, num_agents, *critic_obs_shape))
        self.value_preds = np.zeros((T+1, N, num_agents, 1))
        self.returns = np.zeros((T+1, N, num_agents, 1))
        self.rewards = np.zeros((T, N, num_agents, 1))
        self.masks = np.ones((T+1, N, num_agents, 1))
        self.active_masks = np.ones_like(self.masks)
        # ... actions, log_probs, etc.
```

### 9.2 HARL Buffer

HARL uses **separate buffers** for actor and critic:

**Actor Buffer:** `OnPolicyActorBuffer` (per agent)
**Critic Buffer:** `OnPolicyCriticBufferEP` (shared, EP = Environment-Provided state)

```python
# Critic Buffer (EP mode - same global state for all agents)
class OnPolicyCriticBufferEP:
    def __init__(self, args, share_obs_space):
        self.share_obs = np.zeros((T+1, N, *share_obs_shape))  # No agent dim!
        self.rnn_states_critic = np.zeros((T+1, N, recurrent_n, rnn_hidden_size))
        self.value_preds = np.zeros((T+1, N, 1))
        self.returns = np.zeros((T+1, N, 1))
        self.rewards = np.zeros((T, N, 1))  # Shared reward
        self.masks = np.ones((T+1, N, 1))
        self.bad_masks = np.ones_like(self.masks)  # For truncation
```

### 9.3 Buffer Comparison

| Aspect | OpenRL | HARL |
|--------|--------|------|
| **Buffer Structure** | Single unified buffer | Separate actor/critic buffers |
| **Agent Dimension** | Included (num_agents) | Removed in EP mode |
| **Critic State** | Per-agent critic_obs | Shared global state |
| **Active Masks** | Yes | No (in critic buffer) |
| **Bad Masks** | No | Yes (for truncation) |
| **Data Generator** | feed_forward_generator | feed_forward_generator_critic |

---

## 10. Key Differences Summary

### 10.1 Critical Differences

| # | Aspect | OpenRL | HARL | Impact |
|---|--------|--------|------|--------|
| 1 | **Critic Capacity** | 64 hidden units, 1 layer | 256-256-128, 3 layers | HARL 19x more params |
| 2 | **Critic Input** | Local observation | Global state (17 dims) | HARL has more info |
| 3 | **Value Norm Update** | Inside mini-batch loop | Once before training | HARL more stable |
| 4 | **Critic Epochs** | Same as actor (10) | 5x more than actor (25) | HARL fits value better |
| 5 | **Value Loss Coef** | 0.5 | 5.0 | HARL stronger gradient |
| 6 | **Actor/Critic Coupling** | Joint training | Separate training | HARL more flexible |
| 7 | **Actor Update Rate** | Every iteration | Every 3rd iteration | HARL stabilizes critic |
| 8 | **Proper Time Limits** | No | Yes | HARL handles truncation |
| 9 | **Feature Normalization** | Off (default) | On | HARL normalizes inputs |
| 10 | **Active Masks** | Used in value loss | Not used | OpenRL more selective |

### 10.2 Effective Critic Updates Per Rollout

**OpenRL:**
```
Updates = ppo_epoch * num_mini_batch = 10 * 1 = 10
Value norm updates = 10 (inside loop)
```

**HARL:**
```
Updates = critic_epoch * critic_num_mini_batch = 25 * 1 = 25
Value norm updates = 1 (before loop)
```

**HARL trains critic 2.5x more often with stable normalization targets!**

---

## 11. Potential Issues & Recommendations

### 11.1 Issues Identified in HARL Implementation

1. **Value Normalizer Timing (FIXED Dec 18, 2025)**
   - Original: Updated inside mini-batch loop (non-stationary targets)
   - Fixed: Updated once before training loop

2. **Critic Epochs (FIXED Dec 18, 2025)**
   - Original: 5 epochs (same as actor)
   - Fixed: 25 epochs (5x more)

3. **Value Loss Coefficient (FIXED Dec 18, 2025)**
   - Original: 1.0
   - Fixed: 5.0 (stronger gradient)

4. **Actor Update Interval (FIXED Dec 19, 2025)**
   - Original: 1 (every iteration)
   - Fixed: 3 (every 3rd iteration)

### 11.2 Remaining Differences to Consider

1. **Advantage Normalization**
   - OpenRL: Always normalizes with active_masks
   - HARL: Does not normalize advantages
   - **Recommendation:** Consider adding advantage normalization to HARL

2. **Active Masks in Value Loss**
   - OpenRL: Uses active_masks to weight value loss
   - HARL: Simple mean over all samples
   - **Recommendation:** May not be critical for MAPush (all agents active)

3. **Critic Architecture**
   - HARL has much larger network (19x params)
   - May be overkill for 17-dim input
   - **Recommendation:** Consider if smaller network would suffice

4. **Learning Rate Schedule**
   - Both support linear decay but have it disabled
   - **Recommendation:** May help with late-training stability

### 11.3 Summary of Applied Fixes

| Fix | Date | Change | Rationale |
|-----|------|--------|-----------|
| **CRITIC FIX 1** | Dec 18, 2025 | critic_epoch: 5 -> 25, value_loss_coef: 1.0 -> 5.0 | Give critic more training to reduce value error |
| **CRITIC FIX 2** | Dec 18, 2025 | Value normalizer update moved before training loop | Prevent non-stationary training targets |
| **CRITIC FIX 3** | Dec 19, 2025 | actor_update_interval: 1 -> 3 | Give critic breathing room, policies stay fixed while critic stabilizes |

---

## Appendix: File Locations

### OpenRL (backup_MAPush)
```
/home/gvlab/miniconda3/envs/mapush/lib/python3.8/site-packages/openrl/
├── algorithms/ppo.py                    # PPO training algorithm
├── modules/networks/value_network.py    # Critic network
├── modules/networks/utils/mlp.py        # MLP layers
├── buffers/normal_buffer.py             # Replay buffer
├── buffers/replay_data.py               # Data structures
└── configs/config.py                    # Default hyperparameters

/home/gvlab/backup_MAPush/openrl_ws/
├── train.py                             # Training script
├── utils.py                             # Environment wrapper
└── cfgs/ppo.yaml                        # Config overrides
```

### HARL (new-universal-MAPush)
```
/home/gvlab/new-universal-MAPush/HARL/harl/
├── algorithms/critics/v_critic.py       # Critic training
├── models/value_function_models/v_net.py # Critic network
├── models/base/mlp.py                   # MLP layers
├── common/buffers/
│   ├── on_policy_critic_buffer_ep.py    # EP critic buffer
│   └── on_policy_critic_buffer_fp.py    # FP critic buffer
├── common/valuenorm.py                  # Value normalization
├── runners/
│   ├── on_policy_base_runner.py         # Base runner
│   └── on_policy_ha_runner.py           # HA runner (HAPPO)
├── envs/mapush/mapush_env.py            # Environment wrapper
├── configs/algos_cfgs/happo.yaml        # HAPPO config
└── utils/models_tools.py                # Loss functions
```
