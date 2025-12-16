# HAPPO Critic Architecture for MAPush (2-Agent Cuboid Pushing)

## Overview

**CRITICAL**: HAPPO uses a **SINGLE CENTRALIZED GLOBAL VALUE NETWORK** (not per-agent critics). This is fundamentally different from having separate critics per agent. The key innovation is in how the advantages are used during sequential policy updates, not in having multiple critics.

---

## 1. Critic Network Architecture

### Network Structure
```python
class GlobalValueNetwork(nn.Module):
    """
    Single centralized critic that estimates the value of global states
    """
    def __init__(self, global_state_dim, hidden_sizes=[128, 128]):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(global_state_dim, hidden_sizes[0]),
            nn.ReLU(),
            nn.Linear(hidden_sizes[0], hidden_sizes[1]),
            nn.ReLU(),
            nn.Linear(hidden_sizes[1], 1)  # Single scalar output
        )
    
    def forward(self, global_state):
        """
        Args:
            global_state: [batch_size, global_state_dim]
        Returns:
            value: [batch_size, 1] - scalar value for each state
        """
        return self.network(global_state)
```

**Key Parameters (from hyperparameters):**
- `hidden_sizes`: [128, 128] (for MPE/MAMuJoCo tasks)
- `activation`: ReLU
- `initialization`: Orthogonal with gain 0.01
- `use_valuenorm`: True (normalize value targets)

---

## 2. Critic Input Format (MAPush 2-Agent Cuboid)

### Global State Representation

The critic takes the **global state** `s_t`, which contains complete environment information:

```python
# For 2 agents pushing a cuboid:
global_state_dim = 11 + num_obstacles * 3  # Base: 11, +3 per obstacle

global_state = [
    # === OBJECT STATE (global frame) ===
    x_object,          # [1] object x position
    y_object,          # [1] object y position  
    ψ_object,          # [1] object yaw angle
    
    # === AGENT 1 STATE (global frame) ===
    x_1,               # [1] robot 1 x position
    y_1,               # [1] robot 1 y position
    ψ_1,               # [1] robot 1 yaw angle
    
    # === AGENT 2 STATE (global frame) ===
    x_2,               # [1] robot 2 x position
    y_2,               # [1] robot 2 y position
    ψ_2,               # [1] robot 2 yaw angle
    
    # === SUBGOAL (from high-level controller) ===
    x_subgoal,         # [1] target x position
    y_subgoal,         # [1] target y position
    
    # === OBSTACLE STATES (for each obstacle) ===
    x_obs_i,           # [1] obstacle i x position
    y_obs_i,           # [1] obstacle i y position
    r_obs_i,           # [1] obstacle i radius
    ...                # repeat for all obstacles
]
# Total dimension: 11 + 3*num_obstacles
```

**Important Notes:**
- The critic does NOT take actions as input (unlike Q-learning methods)
- It only estimates V(s), not Q(s,a)
- During training, actions are used to compute advantages, but not as direct input to the value network
- All positions are in the global coordinate frame

---

## 3. Critic Output

### Value Function Output

```python
V_φ(s_t) → scalar value ∈ ℝ

# Shape: [batch_size, 1]
# Interpretation: Expected discounted return from state s_t
```

**What this value represents:**
- Expected cumulative reward from state `s_t` under the current joint policy
- Conditioned on the global state (all agents' positions, object state, obstacles)
- Used to compute advantages for BOTH agents

---

## 4. Advantage Computation (GAE)

Before updating policies, HAPPO computes advantages using **Generalized Advantage Estimation (GAE)**:

### GAE Formula

```python
def compute_advantages(rewards, values, dones, gamma=0.99, gae_lambda=0.95):
    """
    Compute GAE advantages from trajectory data
    
    Args:
        rewards: [T] - rewards at each timestep
        values: [T+1] - value estimates (including bootstrap value)
        dones: [T] - done flags
        gamma: discount factor
        gae_lambda: GAE lambda parameter
    
    Returns:
        advantages: [T] - advantage estimates
        returns: [T] - empirical returns (for value function targets)
    """
    advantages = []
    gae = 0
    
    for t in reversed(range(len(rewards))):
        if t == len(rewards) - 1:
            next_value = values[t + 1]
        else:
            next_value = values[t + 1]
        
        # TD error: δ_t = r_t + γ*V(s_{t+1}) - V(s_t)
        delta = rewards[t] + gamma * next_value * (1 - dones[t]) - values[t]
        
        # GAE: A_t = δ_t + (γλ)*δ_{t+1} + (γλ)²*δ_{t+2} + ...
        gae = delta + gamma * gae_lambda * (1 - dones[t]) * gae
        advantages.insert(0, gae)
    
    advantages = torch.tensor(advantages)
    returns = advantages + values[:-1]  # R̂_t = A_t + V(s_t)
    
    return advantages, returns
```

**Key insight:** 
- The advantage `Â(s_t, a_t)` represents "how much better is this joint action compared to the expected value?"
- This is the **SAME** advantage used for both agents initially
- But HAPPO modifies it sequentially during updates (see Section 6)

---

## 5. Critic Loss Function

### Mean Squared Error Loss

The critic is trained to minimize the TD error between predicted values and empirical returns:

```python
def compute_critic_loss(value_network, global_states, returns, 
                       use_clipped_value_loss=True, clip_param=0.2,
                       use_huber_loss=True, huber_delta=10.0):
    """
    Compute the value function loss
    
    Args:
        value_network: Global V-value network φ
        global_states: [batch_size, global_state_dim] 
        returns: [batch_size] - empirical returns R̂_t
        use_clipped_value_loss: Whether to clip value updates (like PPO)
        clip_param: Clipping range
        use_huber_loss: Whether to use Huber loss instead of MSE
        huber_delta: Huber loss delta parameter
    
    Returns:
        value_loss: Scalar loss value
    """
    # Forward pass through value network
    values_pred = value_network(global_states).squeeze(-1)  # [batch_size]
    
    if use_clipped_value_loss:
        # PPO-style clipped value loss
        # Get old value predictions (from before this update)
        with torch.no_grad():
            values_old = value_network_old(global_states).squeeze(-1)
        
        # Clip the value predictions
        values_clipped = values_old + torch.clamp(
            values_pred - values_old, 
            -clip_param, 
            clip_param
        )
        
        # Compute losses for both clipped and unclipped
        if use_huber_loss:
            loss_unclipped = F.huber_loss(values_pred, returns, 
                                         delta=huber_delta, reduction='none')
            loss_clipped = F.huber_loss(values_clipped, returns,
                                       delta=huber_delta, reduction='none')
        else:
            loss_unclipped = (values_pred - returns) ** 2
            loss_clipped = (values_clipped - returns) ** 2
        
        # Take maximum of clipped and unclipped losses
        value_loss = torch.max(loss_unclipped, loss_clipped).mean()
    else:
        # Standard MSE or Huber loss
        if use_huber_loss:
            value_loss = F.huber_loss(values_pred, returns, 
                                     delta=huber_delta, reduction='mean')
        else:
            value_loss = F.mse_loss(values_pred, returns)
    
    return value_loss


# Full loss computation as in HAPPO pseudocode:
# φ_{k+1} = arg min_φ (1/BT) Σ_{b=1}^B Σ_{t=0}^T [V_φ(s_t) - R̂_t]²
```

**Loss components (from hyperparameters):**
- `use_clipped_value_loss`: True
- `use_huber_loss`: True (more robust to outliers than MSE)
- `huber_delta`: 10.0
- `value_loss_coef`: 1.0 (coefficient in total loss)
- `use_valuenorm`: True (normalize targets)

---

## 6. Gradient Computation

### Backward Pass Through Critic

```python
def update_critic(value_network, optimizer, global_states, returns, 
                 value_loss_coef=1.0, max_grad_norm=10.0):
    """
    Update the global value network using gradient descent
    
    Args:
        value_network: Global V-value network φ
        optimizer: Adam optimizer (lr=5e-4 typically)
        global_states: [batch_size, global_state_dim]
        returns: [batch_size] - empirical returns R̂_t
        value_loss_coef: Coefficient for value loss in total loss
        max_grad_norm: Maximum gradient norm for clipping
    
    Returns:
        value_loss: Scalar loss value (for logging)
    """
    # Compute value loss
    value_loss = compute_critic_loss(value_network, global_states, returns)
    
    # Scale by coefficient
    total_loss = value_loss_coef * value_loss
    
    # Zero gradients
    optimizer.zero_grad()
    
    # Backward pass
    total_loss.backward()
    
    # Gradient clipping (prevent exploding gradients)
    if max_grad_norm is not None:
        torch.nn.utils.clip_grad_norm_(
            value_network.parameters(), 
            max_grad_norm
        )
    
    # Update parameters
    optimizer.step()
    
    return value_loss.item()
```

### Gradient Flow

```
Returns R̂_t (targets)
    ↓
Value Loss: L = (1/BT) Σ [V_φ(s_t) - R̂_t]²
    ↓
∂L/∂φ = (2/BT) Σ [V_φ(s_t) - R̂_t] · ∂V_φ(s_t)/∂φ
    ↓
Gradient Clipping (max_grad_norm=10.0)
    ↓
Adam Optimizer Update: φ ← φ - α·∂L/∂φ
```

**Gradient computation details:**
- Use autograd for automatic differentiation
- Adam optimizer with learning rate `critic_lr = 5e-4`
- Gradient clipping with `max_grad_norm = 10.0`
- Optimizer epsilon: `optim_eps = 1e-5`
- No weight decay (`weight_decay = 0`)

---

## 7. Sequential Update Mechanism (HAPPO-Specific)

### How HAPPO Uses the Critic Differently from MAPPO

**Key Innovation:** While the critic itself is identical to MAPPO, HAPPO uses the advantages differently during policy updates:

```python
def happo_sequential_update(actors, value_network, global_states, 
                           observations, actions, advantages):
    """
    HAPPO's sequential policy update scheme
    
    Args:
        actors: List of N actor networks [π^1, π^2, ..., π^N]
        value_network: Global V-value network (SINGLE network for all agents)
        global_states: [batch_size, global_state_dim]
        observations: List of [batch_size, obs_dim_i] for each agent i
        actions: List of [batch_size, action_dim_i] for each agent i  
        advantages: [batch_size] - joint advantage Â(s,a) from critic
    """
    # Random permutation of agents
    n_agents = len(actors)
    agent_order = torch.randperm(n_agents)  # e.g., [1, 0] or [0, 1]
    
    # Initialize multiplier with full advantage
    M = advantages.clone()  # M_{i1}(s,a) = Â(s,a)
    
    # Sequential update
    for idx, agent_i in enumerate(agent_order):
        # ===== UPDATE AGENT i =====
        # Get old policy probabilities
        with torch.no_grad():
            old_log_probs = actors[agent_i].get_log_prob(
                observations[agent_i], 
                actions[agent_i]
            )
        
        # Current policy probabilities  
        new_log_probs = actors[agent_i].get_log_prob(
            observations[agent_i],
            actions[agent_i]
        )
        
        # Importance ratio
        ratio = torch.exp(new_log_probs - old_log_probs)
        
        # PPO-Clip objective using current multiplier M
        surr1 = ratio * M
        surr2 = torch.clamp(ratio, 1.0 - clip_param, 1.0 + clip_param) * M
        actor_loss = -torch.min(surr1, surr2).mean()
        
        # Update actor i
        actor_optimizer[agent_i].zero_grad()
        actor_loss.backward()
        torch.nn.utils.clip_grad_norm_(actors[agent_i].parameters(), max_grad_norm)
        actor_optimizer[agent_i].step()
        
        # ===== UPDATE MULTIPLIER FOR NEXT AGENT =====
        if idx < n_agents - 1:  # Don't update after last agent
            # Get NEW policy probabilities after update
            with torch.no_grad():
                updated_log_probs = actors[agent_i].get_log_prob(
                    observations[agent_i],
                    actions[agent_i]
                )
            
            # Importance weight for updated policy
            importance_weight = torch.exp(updated_log_probs - old_log_probs)
            
            # Update multiplier: M_{i1:m+1} = (π^{im}_{new}/π^{im}_{old}) * M_{i1:m}
            M = importance_weight * M
```

**Critical Understanding:**

1. **Single Critic**: There is ONE value network that estimates V(s) for the global state

2. **Joint Advantage**: Initially, `Â(s,a) = Â(s, a¹, a²)` is computed using this single critic

3. **Sequential Modification**: As agents update sequentially, the advantage is reweighted:
   - Agent 1 uses: `M₁(s,a) = Â(s,a)`
   - Agent 2 uses: `M₂(s,a) = [π¹_new(a¹)/π¹_old(a¹)] × Â(s,a)`

4. **Credit Assignment**: The multiplier accounts for agent 1's policy change when updating agent 2

---

## 8. Complete Training Loop

### Full HAPPO Algorithm with Critic

```python
def happo_training_step(actors, value_network, 
                       actor_optimizers, critic_optimizer,
                       rollout_buffer, n_epochs=5):
    """
    Complete HAPPO training step
    
    Args:
        actors: List of actor networks [π^1, π^2]
        value_network: Single global V-value network φ
        actor_optimizers: List of optimizers for each actor
        critic_optimizer: Optimizer for value network
        rollout_buffer: Buffer containing trajectory data
        n_epochs: Number of epochs to train (5 for critic, 10 for actors in MAPush)
    """
    # 1. GET TRAJECTORY DATA FROM BUFFER
    global_states = rollout_buffer.global_states      # [B*T, state_dim]
    observations = rollout_buffer.observations         # List of [B*T, obs_dim_i]
    actions = rollout_buffer.actions                   # List of [B*T, act_dim_i]
    rewards = rollout_buffer.rewards                   # [B*T]
    dones = rollout_buffer.dones                       # [B*T]
    
    # 2. COMPUTE VALUE ESTIMATES
    with torch.no_grad():
        values = value_network(global_states).squeeze(-1)  # [B*T]
        next_values = value_network(rollout_buffer.next_global_states).squeeze(-1)
        values_with_bootstrap = torch.cat([values, next_values[-1:]])
    
    # 3. COMPUTE ADVANTAGES USING GAE
    advantages, returns = compute_advantages(
        rewards, values_with_bootstrap, dones,
        gamma=0.99, gae_lambda=0.95
    )
    
    # Normalize advantages
    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
    
    # 4. UPDATE CRITIC (5 epochs)
    for epoch in range(5):  # critic_epoch from hyperparameters
        for batch in rollout_buffer.get_batches(critic_mini_batch_size):
            critic_loss = update_critic(
                value_network, 
                critic_optimizer,
                batch.global_states,
                batch.returns,
                value_loss_coef=1.0,
                max_grad_norm=10.0
            )
    
    # 5. UPDATE ACTORS SEQUENTIALLY (10 epochs in MAPush)
    for epoch in range(10):  # ppo_epoch from hyperparameters
        for batch in rollout_buffer.get_batches(actor_mini_batch_size):
            happo_sequential_update(
                actors,
                value_network,  # Used to maintain advantages, not updated here
                batch.global_states,
                batch.observations,
                batch.actions,
                batch.advantages
            )
    
    return {
        'critic_loss': critic_loss,
        'mean_advantages': advantages.mean().item(),
        'mean_returns': returns.mean().item()
    }
```

---

## 9. Key Hyperparameters for MAPush

From the project documentation and HARL paper:

```python
HAPPO_CRITIC_CONFIG = {
    # === NETWORK ARCHITECTURE ===
    'hidden_sizes': [128, 128],            # Two hidden layers
    'activation': 'ReLU',
    'initialization': 'orthogonal',
    'init_gain': 0.01,
    
    # === TRAINING ===
    'critic_lr': 5e-4,                     # Learning rate
    'critic_epochs': 10,                    # MAPush uses 10 (Table VIII)
    'critic_mini_batch': 1,                # Mini-batches per epoch
    'batch_size': 500 * 200 * 2,           # num_envs * episode_len * num_agents
    
    # === LOSS FUNCTION ===
    'use_clipped_value_loss': True,
    'use_huber_loss': True,
    'huber_delta': 10.0,
    'value_loss_coef': 0.5,                # MAPush uses 0.5 (Table VIII)
    'clip_param': 0.2,
    
    # === ADVANTAGE ESTIMATION ===
    'use_gae': True,
    'gae_lambda': 0.95,
    'gamma': 0.99,
    
    # === OPTIMIZATION ===
    'use_max_grad_norm': True,
    'max_grad_norm': 10.0,
    'optim_eps': 1e-5,
    'weight_decay': 0,
    
    # === NORMALIZATION ===
    'use_valuenorm': True,                 # Normalize value targets
    'use_feature_normalization': True,     # Normalize input features
}
```

---

## 10. Summary: HAPPO vs MAPPO Critic

| Aspect | MAPPO | HAPPO |
|--------|-------|-------|
| **Critic Architecture** | Single global V-network | Single global V-network (SAME!) |
| **Critic Input** | Global state s | Global state s (SAME!) |
| **Critic Output** | V(s) | V(s) (SAME!) |
| **Critic Loss** | MSE/Huber on returns | MSE/Huber on returns (SAME!) |
| **Advantage Computation** | Standard GAE | Standard GAE (SAME!) |
| **Policy Update** | **Simultaneous all agents** | **Sequential with multiplier** |
| **Credit Assignment** | Implicit via joint advantage | **Explicit via importance weighting** |

**The key difference is NOT in the critic itself, but in HOW the advantages from the critic are used during sequential policy updates!**

---

## 11. Practical Implementation Tips

### For MAPush 2-Agent Cuboid Pushing:

1. **Critic input dimension**: 
   - Base: 11 (3 object + 3 agent₁ + 3 agent₂ + 2 subgoal)
   - Add 3 per obstacle: +6 for 2 obstacles
   - Total: **17 dimensions**

2. **Batch processing**:
   - Collect rollouts from 500 parallel environments
   - Episode length: 200 steps
   - Total batch: 500 × 200 × 2 = 200,000 transitions

3. **Memory efficiency**:
   - Store global states only once per timestep (not per agent)
   - Observations stored per agent (in local frames)

4. **Training frequency**:
   - Collect full batch before any updates
   - Update critic 10 epochs with mini-batch size 1
   - Update actors sequentially 10 epochs

5. **Monitoring**:
   - Track critic loss convergence
   - Monitor advantage magnitude (should be normalized ~N(0,1))
   - Check value function predictions vs empirical returns

---

## References

- HAPPO Paper: "Heterogeneous-Agent Reinforcement Learning" (Kuba et al., 2022)
- HAPPO Algorithm Pseudocode: Algorithm 4 in paper
- MAPush Hyperparameters: Table VIII in MAPush paper
- HARL Implementation: https://github.com/PKU-MARL/HARL
