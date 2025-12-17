# EXACT Critic Input Comparison: OpenRL vs HARL

## TL;DR - THE KEY DIFFERENCE

| System | What Critic Sees |
|--------|------------------|
| **OpenRL (MAPPO)** | Each agent's local observation (ego-centric view) |
| **HARL (HAPPO)** | Agent 0's local observation only (NOT global state) |

**BOTH systems use LOCAL observations, NOT global state**. The difference is:
- OpenRL: Critic processes ALL agents' local views
- HARL: Critic processes ONLY agent 0's local view

---

## Detailed Breakdown

### Environment Output (Same for Both)

From `mqe/envs/wrappers/go1_push_mid_wrapper.py:186-187`:

```python
obs = torch.cat([
    rotated_target_pos[:,:,:2],      # Target position in agent's frame [dx_target, dy_target]
    rotated_box_pos[:,:,:2],         # Box position in agent's frame [dx_box, dy_box]
    rotated_box_rpy[:,:,2].unsqueeze(2),  # Box yaw in agent's frame [yaw_box]
    all_base_info                    # Other agents' positions in agent's frame [dx_other, dy_other, yaw_other]
], dim=2)
```

**Shape**: `[num_envs, num_agents, obs_dim]`

For 2 agents:
- **obs_dim = 8**:
  - `[0:2]` = `[dx_target, dy_target]` - target pos relative to THIS agent
  - `[2:4]` = `[dx_box, dy_box]` - box pos relative to THIS agent
  - `[4]` = `yaw_box` - box yaw relative to THIS agent
  - `[5:8]` = `[dx_other, dy_other, yaw_other]` - other agent pos relative to THIS agent

**Key property**: Each agent's observation is **ego-centric** (rotated to agent's local frame).

---

## OpenRL (MAPPO) Critic Input

### Step 1: Environment Returns Observations

From `openrl_ws/utils.py:65-67`:
```python
obs, reward, termination, info = self.env.step(actions)
obs = obs.cpu().numpy()  # Shape: [num_envs, num_agents, obs_dim]
```

**Concrete values for env_id=0**:
```python
obs[0, 0, :] = [dx_target_agent0, dy_target_agent0, dx_box_agent0, dy_box_agent0, yaw_box_agent0, dx_agent1, dy_agent1, yaw_agent1]
obs[0, 1, :] = [dx_target_agent1, dy_target_agent1, dx_box_agent1, dy_box_agent1, yaw_box_agent1, dx_agent0, dy_agent0, yaw_agent0]
```

### Step 2: Buffer Insertion

From `openrl/buffers/replay_data.py:259-267`:
```python
def insert(self, raw_obs, ...):
    critic_obs = get_critic_obs(raw_obs)  # Returns raw_obs directly (no dict)
    policy_obs = get_policy_obs(raw_obs)  # Returns raw_obs directly (no dict)

    self.critic_obs[self.step + 1] = critic_obs.copy()
    # Shape: [episode_length+1, n_rollout_threads, num_agents, obs_dim]
    #      = [201, 500, 2, 8]
```

### Step 3: Critic Forward Pass

From `openrl/modules/networks/value_network.py:113-136`:
```python
def forward(self, critic_obs, rnn_states, masks):
    # critic_obs shape: [batch_size, obs_dim]
    #                 = [n_rollout_threads * num_agents, obs_dim]
    #                 = [500 * 2, 8] = [1000, 8]

    critic_features = self.base(critic_obs)  # MLPBase processes each observation
    values = self.v_out(critic_features)     # Output: [1000, 1]
    return values, rnn_states
```

**What the critic actually sees (batch of 1000 observations)**:
```
obs[0] = agent0_env0: [dx_target, dy_target, dx_box, dy_box, yaw_box, dx_other, dy_other, yaw_other]
obs[1] = agent1_env0: [dx_target, dy_target, dx_box, dy_box, yaw_box, dx_other, dy_other, yaw_other]
obs[2] = agent0_env1: [dx_target, dy_target, dx_box, dy_box, yaw_box, dx_other, dy_other, yaw_other]
obs[3] = agent1_env1: [dx_target, dy_target, dx_box, dy_box, yaw_box, dx_other, dy_other, yaw_other]
...
obs[1000] = agent1_env499: [...]
```

**Each row is a different agent's ego-centric view**.

### Step 4: Value Predictions Stored

From `openrl/buffers/replay_data.py:142-145`:
```python
self.value_preds = np.zeros(
    (episode_length + 1, n_rollout_threads, num_agents, 1),
    dtype=np.float32
)  # Shape: [201, 500, 2, 1]
```

**Concrete values**:
```python
value_preds[step, env_id, agent_id, 0] = V(obs[env_id, agent_id, :])
```

For env_id=0:
```python
value_preds[t, 0, 0, 0] = V([dx_target_agent0, dy_target_agent0, ..., dx_agent1, dy_agent1, yaw_agent1])
value_preds[t, 0, 1, 0] = V([dx_target_agent1, dy_target_agent1, ..., dx_agent0, dy_agent0, yaw_agent0])
```

**Two separate value estimates**, one for each agent's perspective.

---

## HARL (HAPPO) Critic Input

### Step 1: Environment Returns Observations

From `HARL/harl/envs/mapush/mapush_env.py:104-112`:
```python
obs, rewards, dones, infos = self.env.step(actions_torch)
obs_np = obs.cpu().numpy()  # Shape: [n_envs, n_agents, obs_dim] = [500, 2, 8]
state_np = obs_np.copy()    # State = obs (not a true global state!)
```

**Concrete values for env_id=0**:
```python
obs_np[0, 0, :] = [dx_target_agent0, dy_target_agent0, dx_box_agent0, dy_box_agent0, yaw_box_agent0, dx_agent1, dy_agent1, yaw_agent1]
obs_np[0, 1, :] = [dx_target_agent1, dy_target_agent1, dx_box_agent1, dy_box_agent1, yaw_box_agent1, dx_agent0, dy_agent0, yaw_agent0]

state_np[0, 0, :] = SAME as obs_np[0, 0, :]
state_np[0, 1, :] = SAME as obs_np[0, 1, :]
```

### Step 2: Buffer Insertion (EP Mode)

From `HARL/harl/runners/on_policy_base_runner.py:448-456`:
```python
if self.state_type == "EP":  # Environment-Provided state
    self.critic_buffer.insert(
        share_obs[:, 0],      # ONLY agent 0's observation!
        rnn_states_critic,
        values,
        rewards[:, 0],        # ONLY agent 0's reward
        masks[:, 0],
        bad_masks,
    )
```

From `HARL/harl/common/buffers/on_policy_critic_buffer_ep.py:32-35`:
```python
self.share_obs = np.zeros(
    (episode_length + 1, n_rollout_threads, *share_obs_shape),
    dtype=np.float32
)  # Shape: [201, 500, 8]  <- NO agent dimension!
```

**Concrete values stored in buffer**:
```python
self.share_obs[step, 0, :] = state_np[0, 0, :]
                           = [dx_target_agent0, dy_target_agent0, dx_box_agent0, dy_box_agent0, yaw_box_agent0, dx_agent1, dy_agent1, yaw_agent1]
```

**Agent 1's observation is DISCARDED!**

### Step 3: Critic Forward Pass

From `HARL/harl/models/value_function_models/v_net.py:48-67`:
```python
def forward(self, cent_obs, rnn_states, masks):
    # cent_obs shape: [batch_size, share_obs_dim]
    #               = [n_rollout_threads, share_obs_dim]
    #               = [500, 8]

    critic_features = self.base(cent_obs)  # MLPBase processes each observation
    values = self.v_out(critic_features)   # Output: [500, 1]
    return values, rnn_states
```

**What the critic actually sees (batch of 500 observations)**:
```
obs[0] = agent0_env0: [dx_target, dy_target, dx_box, dy_box, yaw_box, dx_other, dy_other, yaw_other]
obs[1] = agent0_env1: [dx_target, dy_target, dx_box, dy_box, yaw_box, dx_other, dy_other, yaw_other]
obs[2] = agent0_env2: [dx_target, dy_target, dx_box, dy_box, yaw_box, dx_other, dy_other, yaw_other]
...
obs[499] = agent0_env499: [...]
```

**Only agent 0's perspectives**, agent 1 never seen by critic!

### Step 4: Value Predictions Stored

From `HARL/harl/common/buffers/on_policy_critic_buffer_ep.py:49-51`:
```python
self.value_preds = np.zeros(
    (episode_length + 1, n_rollout_threads, 1),
    dtype=np.float32
)  # Shape: [201, 500, 1]  <- NO agent dimension!
```

**Concrete values**:
```python
value_preds[t, 0, 0] = V([dx_target_agent0, dy_target_agent0, ..., dx_agent1, dy_agent1, yaw_agent1])
# Same value used for BOTH agents!
```

**One value estimate per environment**, shared by all agents.

---

## Side-by-Side Comparison

### Environment 0, Timestep t

**Observations from MAPush**:
```python
obs[0, 0, :] = [0.5, 1.2, 0.3, 0.8, 0.1, -0.5, -1.2, -0.1]  # Agent 0's view
obs[0, 1, :] = [0.4, 1.3, 0.2, 0.9, 0.1,  0.5,  1.2,  0.1]  # Agent 1's view
```

### OpenRL Critic Input

**Batch contains BOTH agent views**:
```python
critic_obs[0, :] = [0.5, 1.2, 0.3, 0.8, 0.1, -0.5, -1.2, -0.1]  # Agent 0 perspective
critic_obs[1, :] = [0.4, 1.3, 0.2, 0.9, 0.1,  0.5,  1.2,  0.1]  # Agent 1 perspective
```

**Value predictions**:
```python
value_preds[t, 0, 0, 0] = V([0.5, 1.2, 0.3, 0.8, 0.1, -0.5, -1.2, -0.1]) = e.g., 2.3
value_preds[t, 0, 1, 0] = V([0.4, 1.3, 0.2, 0.9, 0.1,  0.5,  1.2,  0.1]) = e.g., 2.1
```

### HARL Critic Input (EP Mode)

**Batch contains ONLY agent 0's view**:
```python
critic_obs[0, :] = [0.5, 1.2, 0.3, 0.8, 0.1, -0.5, -1.2, -0.1]  # Agent 0 perspective only
```

**Value prediction**:
```python
value_preds[t, 0, 0] = V([0.5, 1.2, 0.3, 0.8, 0.1, -0.5, -1.2, -0.1]) = e.g., 2.3
# This SAME value is used for BOTH agents in advantage calculation!
```

---

## The Critical Problem

### OpenRL's Critic:
- **Sees**: Both agents' ego-centric observations
- **Learns**: To estimate value from EACH agent's perspective
- **Produces**: Two value estimates (one per agent)
- **Advantage**: Each agent gets advantage based on its own perspective

### HARL's Critic (Current Implementation):
- **Sees**: ONLY agent 0's ego-centric observation
- **Learns**: To estimate value from ONLY agent 0's perspective
- **Produces**: ONE value estimate (from agent 0's view)
- **Advantage**: BOTH agents get advantage based on agent 0's perspective

**This is INCORRECT for HAPPO theory**, which requires:
- Critic should see a **global state** (not agent 0's local view)
- All agents should get advantages based on the SAME global state estimate

---

## What SHOULD Happen (True HAPPO)

### Ideal Input (Global State)

**Environment should provide**:
```python
global_state = [
    box_x_global, box_y_global, box_yaw_global,        # Box state in global frame
    target_x_global, target_y_global, target_yaw_global,  # Target state in global frame
    agent0_x_global, agent0_y_global, agent0_yaw_global,  # Agent 0 in global frame
    agent1_x_global, agent1_y_global, agent1_yaw_global,  # Agent 1 in global frame
]
# Shape: [num_envs, global_state_dim] = [500, 12]
```

**Properties**:
- **Invariant** to agent order/identity
- **Complete** information about task-relevant state
- **Global** coordinate frame (not ego-centric)

### HARL Critic Would Then See:

```python
critic_obs[0, :] = [box_x, box_y, box_yaw, target_x, target_y, target_yaw,
                   agent0_x, agent0_y, agent0_yaw, agent1_x, agent1_y, agent1_yaw]
# Same observation for all environments
# V(s) estimates joint state value, not individual agent perspective
```

**Advantage for both agents based on SAME global state**:
```python
advantages[t, 0] = returns[t, 0] - V(global_state[t, 0])
# Both agents use SAME baseline for advantage estimation
```

---

## Summary Table

| Aspect | OpenRL (Current) | HARL (Current) | HAPPO (Theory) |
|--------|------------------|----------------|----------------|
| **Critic Input** | All agents' local obs | Agent 0's local obs | Global state |
| **Input Frame** | Ego-centric (per agent) | Ego-centric (agent 0) | Global frame |
| **Input Dim** | `[batch, obs_dim]` where batch includes all agents | `[batch, obs_dim]` where batch is num_envs | `[batch, global_state_dim]` |
| **Value Output** | Per-agent values | Single value per env | Single value per env |
| **Advantage** | Per-agent (from own view) | Shared (from agent 0's view) | Shared (from global state) |
| **Correctness** | OK for MAPPO | **WRONG** (should use global state) | Correct |

---

## Action Items

To fix HARL's HAPPO implementation:

1. **Modify MAPush wrapper** to provide true global state:
   - Box position/orientation in global frame
   - Target position/orientation in global frame
   - All agents' positions/orientations in global frame

2. **Update `MAPushEnv.step()`** to return proper `state_np`:
   ```python
   state_np = construct_global_state(box_state, target_state, agent_states)
   # Shape: [n_envs, global_state_dim]
   ```

3. **Verify buffer insertion** uses global state correctly (already does via `share_obs[:, 0]`, but input will be global not local)

4. **Verify advantage calculation** uses single value for all agents (already correct in HARL)

The current implementation is functionally closer to "MAPPO with agent 0's observation" than true HAPPO.
