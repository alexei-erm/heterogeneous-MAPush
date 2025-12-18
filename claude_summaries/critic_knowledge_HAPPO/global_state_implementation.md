# Global State Implementation for HARL HAPPO Critic

## Summary

Implemented **true global state** for HAPPO critic in HARL, replacing the incorrect agent 0's local observation with a consistent environment-relative coordinate frame representation including agent dynamics.

**CRITICAL FIXES:**
- **Dec 17, 2025 (Fix #1)**: Fixed coordinate frame mismatch where box/target were in world frame but agents were in environment-relative frame
- **Dec 17, 2025 (Fix #2)**: Added agent velocities and fixed target representation (target is a point, not an oriented object)

---

## Global State Format (FINAL - 17 Dimensions)

### Dimensions
- **Total**: 17 dimensions for 2 agents
- **Formula**: `3 (box) + 2 (target) + 6 * n_agents (position + velocity per agent)`

### Content (in order)
```
[0:2]   box_x, box_y                    - Box position (relative to env origin)
[2]     box_yaw                         - Box orientation (yaw angle, absolute)
[3:4]   target_x, target_y              - Target position (relative to env origin, NO yaw!)
[5:7]   agent0_x, agent0_y, agent0_yaw  - Agent 0 position & orientation
[8:10]  agent0_vx, agent0_vy, agent0_vyaw - Agent 0 linear & angular velocity
[11:13] agent1_x, agent1_y, agent1_yaw  - Agent 1 position & orientation
[14:16] agent1_vx, agent1_vy, agent1_vyaw - Agent 1 linear & angular velocity
```

### Properties
1. **Environment-relative coordinate frame** - All positions are relative to environment origin (`env_origins` subtracted)
2. **Consistent frame** - All entities (box, target, agents) use same coordinate system
3. **Agent-order invariant** - State fully describes the environment
4. **Complete information** - Includes both kinematics (position/orientation) AND dynamics (velocities)
5. **Correct target representation** - Target is 2D point (x, y), NOT an oriented object
6. **2D projection** - Uses x, y, yaw for positions, vx, vy, vyaw for velocities (ignoring z for ground-based task)

---

## CRITICAL BUG FIX - Coordinate Frame Mismatch (Dec 17, 2025)

### The Problem

**Symptom**: Critic value loss exploded from 0.05 to 0.41 (711% increase), indicating the critic was learning garbage.

**Root Cause**: Mixed coordinate frames in global state construction:
- **Box/Target positions**: Were in **world frame** (included `env_origins` offset)
- **Agent positions**: Were in **environment-relative frame** (`env_origins` already subtracted)

This created an inconsistent state representation that prevented the critic from learning meaningful value predictions.

### Technical Details

#### Isaac Gym Coordinate Frames

Isaac Gym places multiple environments in a grid with `env_origins` offsets:
```
Environment 0: env_origins = [0.0, 0.0, 0.0]
Environment 1: env_origins = [5.0, 0.0, 0.0]
Environment 2: env_origins = [10.0, 0.0, 0.0]
...
```

All actor positions from `gym.acquire_actor_root_state_tensor()` are in **world frame** (absolute coordinates including these offsets).

#### The Wrapper's Processing

In `mqe/envs/go1/go1.py:161`, the observation buffer processes positions:
```python
self.obs_buf.base_pos = (self.base_pos - self.env_origins_repeat) * self.cfg.obs.scales.base_pos
```

So `obs_buf.base_pos` has `env_origins` **already subtracted**, converting to environment-relative frame.

Similarly, in `mqe/envs/wrappers/go1_push_mid_wrapper.py:163-164`:
```python
box_pos = npc_pos[:,0,:] - self.env.env_origins
target_pos = npc_pos[:,1,:] - self.env.env_origins
```

The wrapper subtracts `env_origins` to get environment-relative positions.

#### The Bug in `mapush_env.py` (BEFORE FIX)

**Lines 115, 119 (BUGGY)**:
```python
# Box state - directly from root_states_npc (WORLD FRAME)
box_pos_global = npc_states[:, 0, :3]  # Includes env_origins offset!

# Target state - directly from root_states_npc (WORLD FRAME)
target_pos_global = npc_states[:, 1, :3]  # Includes env_origins offset!

# Agent state - from obs_buf (ENVIRONMENT-RELATIVE FRAME)
base_pos = obs_buf.base_pos.reshape(...)  # env_origins already subtracted!
```

**Result**: Critic received mixed frames:
```python
# Environment 1 (env_origins = [5.0, 0.0, 0.0]):
state = [
    6.0, 2.0, 0.5,    # box in world frame (actual: 1.0m from env origin)
    10.0, 8.0, 1.0,   # target in world frame (actual: 5.0m from env origin)
    0.5, 1.0, 0.2,    # agent0 in env-relative frame
    1.5, 3.0, 0.8     # agent1 in env-relative frame
]
```

This is **meaningless** - the critic couldn't learn that a box at `[6.0, 2.0]` and agent at `[0.5, 1.0]` are only 0.7m apart in reality!

### The Fix

**Lines 117, 122 (FIXED)**:
```python
# Box state - SUBTRACT env_origins to match agent frame
box_pos_global = npc_states[:, 0, :3] - wrapper.env.env_origins  # [n_envs, 3]

# Target state - SUBTRACT env_origins to match agent frame
target_pos_global = npc_states[:, 1, :3] - wrapper.env.env_origins  # [n_envs, 3]

# Agent state - already in environment-relative frame
base_pos = obs_buf.base_pos.reshape(self.n_envs, self.n_agents, 3)  # [n_envs, n_agents, 3]
```

**Result**: Critic now receives consistent environment-relative frame:
```python
# Environment 1 (env_origins = [5.0, 0.0, 0.0]):
state = [
    1.0, 2.0, 0.5,    # box in env-relative frame
    5.0, 8.0, 1.0,    # target in env-relative frame
    0.5, 1.0, 0.2,    # agent0 in env-relative frame
    1.5, 3.0, 0.8     # agent1 in env-relative frame
]
```

Now the critic can correctly learn spatial relationships!

### Impact on Training

**Before Fix**:
- Critic couldn't learn because state representation was incoherent
- Value loss increased instead of decreased (0.05 → 0.41)
- Different environments had completely different "meaning" for same state values

**After Fix**:
- All environments use consistent coordinate frame
- Spatial relationships are preserved and learnable
- Critic should now converge properly

---

## CRITICAL FIX #2 - Missing Velocities & Wrong Target Representation (Dec 17, 2025)

### The Problem

**Discovered Issue**: The 12-dim global state was fundamentally incomplete for a dynamics task!

**Three Critical Errors:**

1. **Target had orientation (WRONG!)**
   - Implemented: `[target_x, target_y, target_yaw]` = 3 dims
   - Reality: Target is a 2D goal point, NOT an oriented object
   - Should be: `[target_x, target_y]` = 2 dims

2. **Missing agent velocities (CRITICAL!)**
   - Implemented: Only positions `[x, y, yaw]` per agent
   - Problem: Critic cannot distinguish between:
     - Agent moving toward box at 0.5 m/s (good!)
     - Agent stationary near box (bad!)
   - Both have same position → Same state → Same value prediction (WRONG!)

3. **Incomplete state for dynamics prediction**
   - Pushing task requires understanding momentum, collision dynamics
   - Position-only state cannot predict:
     - If agents will successfully push box
     - Impact of agent velocities on box motion
     - Coordination quality (are agents approaching from good angles with good speeds?)

### Why This Matters for HAPPO

HAPPO's critic estimates **value function** V(s), which predicts expected future returns.

**With position-only state (12 dims):**
```python
State at t=0: [box at (1, 2), agents at (0.5, 1) and (1.5, 3)]
State at t=1: [box at (1, 2), agents at (0.6, 1.1) and (1.4, 2.9)]

Critic sees: "Agents moved slightly, box didn't move"
Missing: Were agents accelerating toward box? Or slowing down?
```

**With position + velocity state (17 dims):**
```python
State at t=0: [box at (1, 2), agents at (0.5, 1) with v=(0.2, 0.1) and (1.5, 3) with v=(-0.1, -0.2)]
State at t=1: [box at (1, 2), agents at (0.6, 1.1) with v=(0.3, 0.2) and (1.4, 2.9) with v=(-0.2, -0.3)]

Critic sees: "Agents accelerating toward box from opposite sides - high value, imminent push!"
```

### The Fix

**Lines 87, 144-154 in `mapush_env.py`:**

**BEFORE (12 dims):**
```python
global_state_dim = 3 + 3 + 3 * self.n_agents  # box + target + agents

global_state_list = [
    box_pos_global[:, :2],      # box x, y
    box_rpy[:, 2:3],            # box yaw
    target_pos_global[:, :2],   # target x, y
    target_rpy[:, 2:3],         # target yaw (WRONG!)
]

for agent_id in range(self.n_agents):
    global_state_list.append(base_pos[:, agent_id, :2])     # x, y
    global_state_list.append(base_rpy[:, agent_id, 2:3])    # yaw
    # MISSING: velocities!
```

**AFTER (17 dims):**
```python
global_state_dim = 3 + 2 + 6 * self.n_agents  # box + target + agents(pos+vel)

# Get velocities from obs_buf
base_lin_vel = obs_buf.lin_vel.reshape(self.n_envs, self.n_agents, 3)
base_ang_vel = obs_buf.ang_vel.reshape(self.n_envs, self.n_agents, 3)

global_state_list = [
    box_pos_global[:, :2],      # box x, y
    box_rpy[:, 2:3],            # box yaw
    target_pos_global[:, :2],   # target x, y (NO yaw!)
]

for agent_id in range(self.n_agents):
    global_state_list.append(base_pos[:, agent_id, :2])       # x, y
    global_state_list.append(base_rpy[:, agent_id, 2:3])      # yaw
    global_state_list.append(base_lin_vel[:, agent_id, :2])   # vx, vy
    global_state_list.append(base_ang_vel[:, agent_id, 2:3])  # vyaw
```

### Data Source Verification

Velocities come from `obs_buf` which is populated in `mqe/envs/go1/go1.py:172-176`:

```python
if self.cfg.obs.cfgs.lin_vel or self.cfg.control.control_type == "C":
    self.obs_buf.lin_vel = self.base_lin_vel * self.obs_scales.lin_vel

if self.cfg.obs.cfgs.ang_vel or self.cfg.control.control_type == "C":
    self.obs_buf.ang_vel = self.base_ang_vel * self.obs_scales.ang_vel
```

Since `control_type = 'C'` in MAPush config, these fields are **always populated**.

Velocities are in **body frame** (robot-centric), which is appropriate since:
- Each robot's policy uses body-frame velocities for control
- Critic needs to understand each robot's motion relative to its orientation

### Verification Output

From actual training run:
```
Global state shape: (500, 17)
Expected: [500, 17] for 2 agents

Environment 0 global state (17 dims):
  Box:    x=12.000, y=0.000, yaw=0.475
  Target: x=14.155, y=1.456                     ← NO yaw! Correct!
  Agent0: x=11.549, y=-1.205, yaw=1.487, vx=-0.030, vy=-0.098, vyaw=0.050  ← Velocities!
  Agent1: x=11.322, y=1.028, yaw=5.350, vx=0.005, vy=0.038, vyaw=-0.006    ← Velocities!

NaN count: 0
Inf count: 0

✓ CORRECT: Critic receiving 17-dim global state with velocities
```

### Expected Impact

**With velocities, the critic can now:**

1. **Predict push outcomes**: High velocity + good angle = high value
2. **Distinguish strategies**:
   - Slow coordinated approach (medium value, safer)
   - Fast aggressive push (high value if coordinated, low if not)
3. **Evaluate momentum**: Fast-moving box = higher/lower value depending on direction
4. **Credit assignment**: Agent contributing velocity toward goal = higher value contribution
5. **Learn dynamics**: Box response to agent forces, collision effects, etc.

This should significantly improve:
- Value function accuracy
- Credit assignment between agents
- Training stability (more informative state → better gradients)
- Final task performance (better coordination through better value estimates)

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
| **Critic Input** | Agent 0's local observation | Centralized global state |
| **Coordinate Frame** | Ego-centric (agent 0) | Environment-relative (consistent) |
| **Input Shape** | `[500, 8]` | `[500, 12]` |
| **Agent 1 Info** | As relative position to agent 0 | Position relative to env origin |
| **Box/Target Info** | Relative to agent 0 | Relative to env origin (consistent!) |
| **Frame Consistency** | Mixed frames (bug: world + relative) | Single frame (all env-relative) |
| **Invariance** | NOT agent-order invariant | Agent-order invariant |
| **HAPPO Theory** | VIOLATED | SATISFIED |

### Example State Comparison

**Scenario** (all positions relative to environment origin):
- Box at (1.0, 2.0, 0.5 rad)
- Target at (5.0, 6.0, 1.0 rad)
- Agent 0 at (0.5, 1.0, 0.2 rad)
- Agent 1 at (1.5, 3.0, 0.8 rad)

**Before (Agent 0's ego-centric local view)**:
```python
state[0, 0, :] = [
    4.8, 5.1,      # target relative to agent 0 (rotated by agent0's yaw)
    0.6, 1.1,      # box relative to agent 0 (rotated by agent0's yaw)
    0.3,           # box yaw relative to agent 0
    1.1, 2.1, 0.6  # agent 1 relative to agent 0 (rotated by agent0's yaw)
]  # Shape: (8,)
# Everything in agent 0's local coordinate frame - NOT suitable for centralized critic!
```

**After (Environment-relative centralized state)**:
```python
state[0, 0, :] = [
    1.0, 2.0, 0.5,   # box position (env-relative) + yaw (absolute)
    5.0, 6.0, 1.0,   # target position (env-relative) + yaw (absolute)
    0.5, 1.0, 0.2,   # agent 0 position (env-relative) + yaw (absolute)
    1.5, 3.0, 0.8    # agent 1 position (env-relative) + yaw (absolute)
]  # Shape: (12,)

# state[0, 1, :] is IDENTICAL (broadcasted) - all agents see same global state
# All positions in CONSISTENT environment-relative frame!
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

The implementation now correctly provides HAPPO's critic with a **true centralized state** in a **consistent environment-relative coordinate frame**, satisfying the theoretical requirements for proper credit assignment and centralized value estimation.

**Key fixes applied**:
1. ✅ Replaced agent 0's ego-centric observation with centralized global state
2. ✅ Fixed coordinate frame mismatch (all entities now use environment-relative frame)
3. ✅ Ensured consistent spatial relationships across all parallel environments
4. ✅ Enabled proper credit assignment for multi-agent HAPPO training

This resolves two critical issues:
1. **Original issue**: Critic was using agent 0's ego-centric observation instead of global state
2. **Coordinate frame bug (Dec 17)**: Mixed world frame and environment-relative frame in state construction

With these fixes, the critic should now be able to learn meaningful value predictions for the collaborative pushing task.

---

## CRITICAL FIX #3 - Value Normalizer Non-Stationary Targets (Dec 18, 2025)

### The Problem

**Symptom**: Despite fixes #1 and #2, critic value loss continued to increase over first 40M steps with high oscillations.

**Root Cause**: Value normalizer was being updated **inside the mini-batch training loop**, creating non-stationary training targets.

**Location**: `HARL/harl/algorithms/critics/v_critic.py:91` (in `cal_value_loss()`)

**Bug Code**:
```python
if value_normalizer is not None:
    value_normalizer.update(return_batch)  # ❌ Called EVERY mini-batch!
    error_original = value_normalizer.normalize(return_batch) - values
```

**Impact**:
- With `critic_epoch=5`, normalizer updated **5 times per training iteration**
- Each update changed mean/std statistics during training
- Training targets became non-stationary (goal posts moved mid-training)
- Result: High variance, oscillations, increasing critic loss

### The Fix

**File**: `HARL/harl/algorithms/critics/v_critic.py`

**Change 1**: Removed normalizer update from `cal_value_loss()` (line 91)
```python
if value_normalizer is not None:
    # FIX (Dec 18, 2025): Removed value_normalizer.update() from here
    # Normalizer now updated ONCE in train() before the training loop
    error_original = value_normalizer.normalize(return_batch) - values
```

**Change 2**: Added single normalizer update in `train()` method (lines 175-182)
```python
def train(self, critic_buffer, value_normalizer=None):
    train_info = {}
    train_info["value_loss"] = 0
    train_info["critic_grad_norm"] = 0

    # FIX (Dec 18, 2025): Update value normalizer ONCE before training loop
    if value_normalizer is not None:
        all_returns = critic_buffer.returns[:-1].reshape(-1, 1)
        all_returns_tensor = check(all_returns).to(**self.tpdv)
        value_normalizer.update(all_returns_tensor)

    for _ in range(self.critic_epoch):
        # ... training loop with FIXED normalizer stats ...
```

### Why This Works

**Before**:
```
train():
  epoch 1: update normalizer (mean=10.2) → train
  epoch 2: update normalizer (mean=10.5) → train  # Changed!
  epoch 3: update normalizer (mean=10.8) → train  # Changed!
  epoch 4: update normalizer (mean=11.1) → train  # Changed!
  epoch 5: update normalizer (mean=11.3) → train  # Changed!
```
Non-stationary targets → unstable training

**After**:
```
train():
  update normalizer ONCE (mean=10.5, std=5.3)
  epoch 1: train with mean=10.5  # Fixed
  epoch 2: train with mean=10.5  # Fixed
  epoch 3: train with mean=10.5  # Fixed
  epoch 4: train with mean=10.5  # Fixed
  epoch 5: train with mean=10.5  # Fixed
```
Stationary targets → stable training

### Expected Impact

**Before Fix**:
- ❌ Critic loss: Increasing or highly oscillating
- ❌ Training: Unstable, high variance
- ❌ Value predictions: Inconsistent

**After Fix**:
- ✅ Critic loss: **Decreasing steadily**
- ✅ Training: Stable, lower variance
- ✅ Value predictions: Consistent, tracking returns properly

This fix is critical for HAPPO because unstable value predictions lead to noisy advantages, which propagate through the sequential importance-weighted policy updates.
