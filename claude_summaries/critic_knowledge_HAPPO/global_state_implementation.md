# Global State Implementation for HARL HAPPO Critic

## Summary

Implemented **true global state** for HAPPO critic in HARL, replacing the incorrect agent 0's local observation with a global coordinate frame representation.

---

## Global State Format

### Dimensions
- **Total**: 12 dimensions for 2 agents
- **Formula**: `3 (box) + 3 (target) + 3 * n_agents`

### Content (in order)
```
[0:2]   box_x, box_y           - Box position in global frame
[2]     box_yaw                - Box orientation (yaw angle)
[3:5]   target_x, target_y     - Target position in global frame
[5]     target_yaw             - Target orientation (yaw angle)
[6:8]   agent0_x, agent0_y     - Agent 0 position in global frame
[8]     agent0_yaw             - Agent 0 orientation (yaw angle)
[9:11]  agent1_x, agent1_y     - Agent 1 position in global frame
[11]    agent1_yaw             - Agent 1 orientation (yaw angle)
```

### Properties
1. **Global coordinate frame** - All positions/orientations are absolute, not relative to any agent
2. **Agent-order invariant** - State fully describes the environment
3. **Complete information** - Contains all task-relevant state
4. **2D projection** - Uses x, y, yaw (ignoring z for ground-based robots)

---

## Implementation Details

### File: `HARL/harl/envs/mapush/mapush_env.py`

#### 1. Updated Share Observation Space

**Before**:
```python
self.share_observation_space = [self.env.observation_space] * self.n_agents
# Shape: (8,) - local ego-centric observation
```

**After**:
```python
global_state_dim = 3 + 3 + 3 * self.n_agents  # box + target + all agents
self.share_observation_space = [
    spaces.Box(low=-float('inf'), high=float('inf'),
              shape=(global_state_dim,), dtype=np.float32)
] * self.n_agents
# Shape: (12,) for 2 agents - global state
```

#### 2. New Method: `_construct_global_state()`

```python
def _construct_global_state(self) -> np.ndarray:
    """Construct global state from environment internals.

    Returns:
        global_state: [n_envs, global_state_dim] numpy array
    """
```

**Data Sources**:
- `wrapper.root_states_npc` - Box and target states in world frame
- `obs_buf.base_pos` - Agent positions from observation buffer
- `obs_buf.base_rpy` - Agent orientations from observation buffer

**Process**:
1. Extract box position and quaternion from NPC states
2. Extract target position and quaternion from NPC states
3. Convert quaternions to Euler angles (yaw)
4. Extract all agents' positions and orientations
5. Concatenate into single global state vector
6. Handle NaN/Inf values
7. Return numpy array `[n_envs, 12]`

#### 3. Updated `step()` Method

**Before**:
```python
state_np = obs_np.copy()  # Local observations
```

**After**:
```python
global_state_np = self._construct_global_state()  # [n_envs, 12]

# Broadcast to [n_envs, n_agents, 12] for EP mode compatibility
state_np = np.broadcast_to(
    global_state_np[:, np.newaxis, :],
    (self.n_envs, self.n_agents, global_state_np.shape[1])
)
```

**Why broadcast?**: HARL runner expects `share_obs` shape `[n_envs, n_agents, state_dim]` and uses `share_obs[:, 0]` to extract state for EP mode. Broadcasting ensures all agents have the same global state.

#### 4. Updated `reset()` Method

Same changes as `step()` - constructs and broadcasts global state.

---

## Data Flow Verification

### Environment Output
```python
obs_np:   [500, 2, 8]  - Per-agent local observations (unchanged)
state_np: [500, 2, 12] - Global state (broadcasted, all identical)
```

### HARL Runner (EP Mode)
```python
# on_policy_base_runner.py:450
self.critic_buffer.insert(
    share_obs[:, 0],  # [500, 12] - Global state from agent 0 index
    ...
)
```

### Critic Buffer (EP Mode)
```python
# on_policy_critic_buffer_ep.py:32-35
self.share_obs = np.zeros(
    (episode_length + 1, n_rollout_threads, *share_obs_shape),
    dtype=np.float32
)
# Shape: [201, 500, 12] - True global state
```

### VNet Critic Input
```python
# v_net.py:48-67
def forward(self, cent_obs, rnn_states, masks):
    # cent_obs shape: [batch_size, 12]
    #               = [500, 12] for 500 envs
```

**What critic sees per batch sample**:
```
obs[0] = [box_x, box_y, box_yaw, target_x, target_y, target_yaw,
          agent0_x, agent0_y, agent0_yaw, agent1_x, agent1_y, agent1_yaw]  # env 0
obs[1] = [box_x, box_y, box_yaw, target_x, target_y, target_yaw,
          agent0_x, agent0_y, agent0_yaw, agent1_x, agent1_y, agent1_yaw]  # env 1
...
obs[499] = [...]  # env 499
```

**Each row is the SAME global state for that environment, not agent-specific!**

---

## Comparison: Before vs After

| Aspect | Before (WRONG) | After (CORRECT) |
|--------|----------------|-----------------|
| **Critic Input** | Agent 0's local observation | Global state |
| **Coordinate Frame** | Ego-centric (agent 0) | Global (world) |
| **Input Shape** | `[500, 8]` | `[500, 12]` |
| **Agent 1 Info** | As relative position to agent 0 | Absolute position in world |
| **Box/Target Info** | Relative to agent 0 | Absolute position in world |
| **Invariance** | NOT agent-order invariant | Agent-order invariant |
| **HAPPO Theory** | VIOLATED | SATISFIED |

### Example State Comparison

**Scenario**:
- Box at (1.0, 2.0, 0.5 rad)
- Target at (5.0, 6.0, 1.0 rad)
- Agent 0 at (0.5, 1.0, 0.2 rad)
- Agent 1 at (1.5, 3.0, 0.8 rad)

**Before (Agent 0's local view)**:
```python
state[0, 0, :] = [
    4.8, 5.1,      # target relative to agent 0 (rotated)
    0.6, 1.1,      # box relative to agent 0 (rotated)
    0.3,           # box yaw relative to agent 0
    1.1, 2.1, 0.6  # agent 1 relative to agent 0 (rotated)
]  # Shape: (8,)
```

**After (Global state)**:
```python
state[0, 0, :] = [
    1.0, 2.0, 0.5,   # box absolute
    5.0, 6.0, 1.0,   # target absolute
    0.5, 1.0, 0.2,   # agent 0 absolute
    1.5, 3.0, 0.8    # agent 1 absolute
]  # Shape: (12,)

# state[0, 1, :] is IDENTICAL (broadcasted)
```

---

## Impact on HAPPO Training

### Advantages (from same global state)

**Before**:
```python
# Advantage for agent 0
A_agent0[t] = R[t, agent0] - V(agent0_local_obs[t])

# Advantage for agent 1
A_agent1[t] = R[t, agent1] - V(agent0_local_obs[t])  # WRONG! Uses agent 0's view
```

**After**:
```python
# Advantage for both agents
A_agent0[t] = R[t, agent0] - V(global_state[t])
A_agent1[t] = R[t, agent1] - V(global_state[t])  # CORRECT! Same baseline
```

### Sequential Policy Update

**Before**:
```python
factor_agent0 = 1.0
factor_agent1 = exp(new_logprob_agent0 - old_logprob_agent0)

# But baseline was agent 0's view - inconsistent!
```

**After**:
```python
factor_agent0 = 1.0
factor_agent1 = exp(new_logprob_agent0 - old_logprob_agent0)

# Baseline is global state - CONSISTENT with HAPPO theory!
```

---

## Testing Verification

To verify the implementation is correct:

```python
import numpy as np

# After environment step
obs, state, rewards, dones, infos, _ = env.step(actions)

# Check shapes
assert obs.shape == (500, 2, 8), "Observations should be [n_envs, n_agents, 8]"
assert state.shape == (500, 2, 12), "State should be [n_envs, n_agents, 12]"

# Check global state is broadcasted (all agents see same state)
assert np.allclose(state[:, 0, :], state[:, 1, :]), "All agents should see same global state"

# Check state contains global coordinates
# Box should NOT be at origin (would be if ego-centric)
box_positions = state[:, 0, 0:2]  # [n_envs, 2]
assert not np.allclose(box_positions, 0.0), "Box should have non-zero global position"

print("Global state verification PASSED!")
```

---

## Remaining Considerations

1. **Permutation Invariance**: Current implementation orders agents as [agent0, agent1]. For true permutation invariance, could use a set-based encoding or attention mechanism. However, for 2 agents with fixed roles, current approach is sufficient.

2. **Scaling to More Agents**: Formula `3 + 3 + 3*n_agents` scales linearly. For many agents, consider graph neural network or attention-based critic.

3. **Velocity Information**: Current state only includes positions and orientations. Could extend to include velocities if needed:
   ```python
   global_state_dim = 3 + 3 + 6 * n_agents  # pos + vel for each agent
   ```

4. **Task-Specific State**: For different tasks, may want different global state components (e.g., goal regions, obstacles).

---

## Conclusion

The implementation now correctly provides HAPPO's critic with a **true global state** in a **global coordinate frame**, satisfying the theoretical requirements for proper credit assignment and centralized value estimation.

This resolves the critical issue identified in `EXACT_critic_input_comparison.md` where the critic was incorrectly using agent 0's ego-centric observation instead of a global state.
