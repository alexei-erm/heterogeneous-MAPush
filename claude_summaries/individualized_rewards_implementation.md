# Individualized Rewards Implementation for HAPPO

**Date:** 2025-12-14
**Status:** ✅ Implementation Complete

---

## Problem Statement

### Freeloading Behavior in HAPPO

When using HAPPO (Heterogeneous-Agent Proximal Policy Optimization) with **shared rewards**, agents can learn to freeload:

- **Scenario:** Two agents pushing a box collaboratively
- **Issue:** Rewards are broadcast equally to both agents regardless of contribution
- **Result:** One agent learns to do nothing while the other pushes
- **Root Cause:** HAPPO uses separate neural networks per agent, so shared rewards don't properly attribute credit

### Original Shared Rewards

Three rewards were shared equally between agents:

1. **`reach_target_reward`** (+10 when box reaches target)
2. **`target_reward`** (distance-based progress toward target)
3. **`push_reward`** (+0.0015 when box is moving)

```python
# Before (shared):
reward[:, :] += distance_reward.unsqueeze(1).repeat(1, self.num_agents)
# Both agents get same reward regardless of who pushed
```

---

## Solution: Contact-Based Reward Attribution

### Core Concept

**Agents only get credit for actions when they are in contact with the box.**

- Uses **proximity to box** as proxy for contact
- Agents within `contact_threshold` distance get full reward
- Agents farther away get partial/zero reward
- Smooth linear decay to avoid discontinuities

### Contact Weight Formula

```python
agent_box_dist = torch.norm(box_pos - base_pos[:, i, :], dim=1)
contact_weight = torch.clamp(
    1.0 - (agent_box_dist - contact_threshold) / contact_threshold,
    0.0,
    1.0
)
```

**Examples (with `contact_threshold = 0.8m`):**
- Agent at `0.5m` from box → weight = `1.0` (full reward)
- Agent at `0.8m` from box → weight = `1.0` (full reward)
- Agent at `1.2m` from box → weight = `0.5` (half reward)
- Agent at `1.6m` from box → weight = `0.0` (no reward)
- Agent at `2.0m+` from box → weight = `0.0` (no reward)

---

## Implementation Details

### 1. Configuration Options

**File:** `mqe/envs/configs/go1_push_mid_config.py`

```python
class rewards(Go1Cfg.rewards):
    # Enable individualized rewards (default: False for backward compatibility)
    individualized_rewards = False

    # Distance threshold for contact detection (meters)
    contact_threshold = 0.8

    class scales:
        target_reward_scale = 0.00325
        approach_reward_scale = 0.00075
        collision_punishment_scale = -0.0025
        push_reward_scale = 0.0015
        ocb_reward_scale = 0.004
        reach_target_reward_scale = 10
        exception_punishment_scale = -5
```

### 2. Modified Rewards

**File:** `mqe/envs/wrappers/go1_push_mid_wrapper.py`

#### A. Reach Target Reward (lines 316-331)

**Before:**
```python
reward[self.finished_buf, :] += self.reach_target_reward_scale
```

**After:**
```python
if self.individualized_rewards:
    for i in range(self.num_agents):
        agent_box_dist = torch.norm(box_pos - base_pos[:, i, :], dim=1)
        contact_weight = torch.clamp(
            1.0 - (agent_box_dist - self.contact_threshold) / self.contact_threshold,
            0.0, 1.0
        )
        individual_reward = self.reach_target_reward_scale * contact_weight
        reward[self.finished_buf, i] += individual_reward[self.finished_buf]
else:
    # Original shared reward
    reward[self.finished_buf, :] += self.reach_target_reward_scale
```

**Explanation:**
- When episode succeeds, only agents near the box get the +10 bonus
- Agent far from box gets reduced/zero bonus (didn't help push)

#### B. Target Reward (Distance Progress) (lines 341-360)

**Before:**
```python
distance_reward = self.target_reward_scale * 100 * (2 * (past_distance - distance) - 0.01 * distance)
reward[:, :] += distance_reward.unsqueeze(1).repeat(1, self.num_agents)
```

**After:**
```python
distance_reward = self.target_reward_scale * 100 * (2 * (past_distance - distance) - 0.01 * distance)

if self.individualized_rewards:
    for i in range(self.num_agents):
        agent_box_dist = torch.norm(box_pos - base_pos[:, i, :], dim=1)
        contact_weight = torch.clamp(
            1.0 - (agent_box_dist - self.contact_threshold) / self.contact_threshold,
            0.0, 1.0
        )
        reward[:, i] += distance_reward * contact_weight
else:
    # Original shared reward
    reward[:, :] += distance_reward.unsqueeze(1).repeat(1, self.num_agents)
```

**Explanation:**
- Box moves toward target → only agents near box get credit
- Agent far away → reduced/zero credit for progress

#### C. Push Reward (lines 384-404)

**Before:**
```python
box_moving = torch.norm(...[:, 0, 7:9], dim=1) > 0.1
push_reward = torch.zeros((self.env.num_envs,), device=self.env.device)
push_reward[box_moving] = self.push_reward_scale
reward[:, :] += push_reward.unsqueeze(1).repeat(1, self.num_agents)
```

**After:**
```python
box_moving = torch.norm(self.root_states_npc.reshape(self.num_envs, self.num_npcs, -1)[:, 0, 7:9], dim=1) > 0.1

if self.individualized_rewards:
    for i in range(self.num_agents):
        agent_box_dist = torch.norm(box_pos - base_pos[:, i, :], dim=1)
        contact_weight = torch.clamp(
            1.0 - (agent_box_dist - self.contact_threshold) / self.contact_threshold,
            0.0, 1.0
        )
        individual_push_reward = self.push_reward_scale * contact_weight
        reward[box_moving, i] += individual_push_reward[box_moving]
else:
    # Original shared reward
    push_reward = torch.zeros((self.env.num_envs,), device=self.env.device)
    push_reward[box_moving] = self.push_reward_scale
    reward[:, :] += push_reward.unsqueeze(1).repeat(1, self.num_agents)
```

**Explanation:**
- Box is moving → only agents in contact get push reward
- Agent far away → doesn't get credit for box motion

### 3. Rewards That Remain Individual

These rewards were **already individualized** (unchanged):

- **`approach_reward`** - Encourages each agent to approach box individually
- **`collision_punishment`** - Penalizes inter-agent collisions (per-agent)
- **`ocb_reward`** (Optimal Collaborative Behavior) - Per-agent based on push angle

### 4. HAPPO-Specific Task Config

**File:** `mqe/envs/configs/go1_push_mid_happo_config.py` (NEW)

```python
from mqe.envs.configs.go1_push_mid_config import Go1PushMidCfg

class Go1PushMidHappoCfg(Go1PushMidCfg):
    """HAPPO configuration with individualized rewards enabled."""

    class rewards(Go1PushMidCfg.rewards):
        expanded_ocb_reward = False
        individualized_rewards = True  # ← ENABLED
        contact_threshold = 0.8
        class scales:
            target_reward_scale = 0.00325
            approach_reward_scale = 0.00075
            collision_punishment_scale = -0.0025
            push_reward_scale = 0.0015
            ocb_reward_scale = 0.004
            reach_target_reward_scale = 10
            exception_punishment_scale = -5
```

**Registered as:** `go1push_mid_happo` in `mqe/envs/utils.py`

---

## Usage

### Training with Individualized Rewards

**Option 1: Use HAPPO task (recommended)**
```bash
./run_training.sh --exp_name happo_individual --task go1push_mid_happo
```

**Option 2: Modify config manually**
```python
# In task/cuboid/config.py or mqe/envs/configs/go1_push_mid_config.py
class rewards(Go1Cfg.rewards):
    individualized_rewards = True  # ← Set to True
    contact_threshold = 0.8
```

Then run:
```bash
./run_training.sh --exp_name happo_individual --task go1push_mid
```

### Training with Shared Rewards (Original Behavior)

```bash
./run_training.sh --exp_name happo_shared --task go1push_mid
```

**Note:** Default is `individualized_rewards = False` for backward compatibility.

---

## Expected Behavior

### With Shared Rewards (Original)

**Episode Example:**
1. Both agents start far from box
2. Agent A approaches box and pushes toward target
3. Agent B stays far away and does nothing
4. Box reaches target
5. **Both agents get +10 reach_target_reward**
6. **Agent B learns:** "I can get reward without doing anything"

**Result:** Freeloading behavior emerges

### With Individualized Rewards (New)

**Episode Example:**
1. Both agents start far from box
2. Agent A approaches box and pushes toward target
3. Agent B stays far away (2m from box)
4. Box reaches target
5. **Agent A gets +10 reach_target_reward** (contact_weight = 1.0)
6. **Agent B gets +0 reach_target_reward** (contact_weight = 0.0)
7. **Agent B learns:** "I need to approach and push to get reward"

**Result:** Both agents learn to contribute

---

## Testing

### Unit Testing (Not Yet Done)

To verify the implementation works:

1. **Test contact weight calculation:**
   ```python
   # Agent at various distances
   distances = [0.5, 0.8, 1.2, 1.6, 2.0]
   for dist in distances:
       weight = max(0.0, min(1.0, 1.0 - (dist - 0.8) / 0.8))
       print(f"Distance: {dist}m → Weight: {weight}")
   ```

2. **Test reward attribution:**
   - Create scenario where Agent A is at 0.6m from box
   - Agent B is at 2.0m from box
   - Box moves toward target
   - Verify: Agent A gets full reward, Agent B gets zero

### Integration Testing

**After training with individualized rewards:**

1. **Observe agent behavior in viewer mode:**
   ```bash
   ./run_testing.sh --checkpoint HARL/results/.../checkpoints/50M --mode viewer --num_episodes 5
   ```
   - Both agents should approach and push box
   - No agent should stay idle

2. **Compare with shared rewards baseline:**
   - Train two models: one with `individualized_rewards=True`, one with `False`
   - Test both and observe behavior
   - Individualized should show less freeloading

---

## Tuning Parameters

### Contact Threshold

**Current:** `0.8m`

**Effects of changing:**

- **Smaller (e.g., 0.5m):**
  - Agents must be very close to get reward
  - More precise credit assignment
  - Might be too strict (agents get zero reward even when helping)

- **Larger (e.g., 1.2m):**
  - Agents can be farther and still get reward
  - Less precise credit assignment
  - Approaches shared reward behavior

**Recommended:** Keep at `0.8m` initially, tune if needed based on:
- Agent body size (~0.5m)
- Box size (1.2m diagonal)
- Typical push distance

### Weight Function

**Current:** Linear decay
```python
weight = clamp(1.0 - (dist - threshold) / threshold, 0.0, 1.0)
```

**Alternative:** Exponential decay (sharper cutoff)
```python
weight = exp(-((dist - threshold) / decay_rate) ** 2)
```

**Alternative:** Step function (binary)
```python
weight = 1.0 if dist < threshold else 0.0
```

---

## Files Modified

| File | Changes | Lines |
|------|---------|-------|
| `mqe/envs/wrappers/go1_push_mid_wrapper.py` | Added individualized reward logic | 82-86, 316-404 |
| `mqe/envs/configs/go1_push_mid_config.py` | Added config options | 99-103 |
| `task/cuboid/config.py` | Added config options | 99-103 |
| `mqe/envs/configs/go1_push_mid_happo_config.py` | NEW: HAPPO config | All |
| `task/cuboid/config_happo.py` | NEW: HAPPO config (duplicate) | All |
| `mqe/envs/utils.py` | Registered `go1push_mid_happo` task | 10, 28-32 |

---

## Next Steps

### 1. Short Test Run (Recommended)

Verify implementation with 1M step training:

```bash
# Test individualized rewards
./run_training.sh \
    --exp_name test_individual \
    --task go1push_mid_happo \
    --num_env_steps 1000000 \
    --n_rollout_threads 500

# Test shared rewards (baseline)
./run_training.sh \
    --exp_name test_shared \
    --task go1push_mid \
    --num_env_steps 1000000 \
    --n_rollout_threads 500
```

Then test both:
```bash
./run_testing.sh --checkpoint .../test_individual/checkpoints/1M --mode viewer --num_episodes 3
./run_testing.sh --checkpoint .../test_shared/checkpoints/1M --mode viewer --num_episodes 3
```

**Look for:**
- Individualized: Both agents approach and push
- Shared: Possible freeloading behavior

### 2. Full Training

If short test looks good:

```bash
./run_training.sh \
    --exp_name happo_individual_full \
    --task go1push_mid_happo \
    --num_env_steps 100000000 \
    --seed 1
```

### 3. Evaluation

Compare final performance:
- Success rate (should be similar or better)
- Collision rate (might be lower with better coordination)
- Collaboration degree (should be higher)
- Visual inspection (both agents working)

---

## Troubleshooting

### Issue: Training Crashes

**Error:** `KeyError: 'individualized_rewards'`

**Solution:** Config not loaded properly. Ensure using updated config:
```bash
# Check which config is loaded
python -c "from task.cuboid.config import Go1PushMidCfg; print(Go1PushMidCfg.rewards.individualized_rewards)"
# Should print: False

python -c "from task.cuboid.config_happo import Go1PushMidHappoCfg; print(Go1PushMidHappoCfg.rewards.individualized_rewards)"
# Should print: True
```

### Issue: No Difference in Behavior

**Possible causes:**
1. `individualized_rewards` still set to `False`
2. `contact_threshold` too large (agents always in contact)
3. Other rewards dominating (approach, OCB)

**Debug:**
```python
# Add print in wrapper __init__
print(f"Individualized rewards: {self.individualized_rewards}")
print(f"Contact threshold: {self.contact_threshold}")
```

### Issue: Agents Don't Approach Box

**Possible cause:** Contact threshold too strict

**Solution:** Increase threshold:
```python
contact_threshold = 1.2  # instead of 0.8
```

---

## Implementation Quality

### Strengths

✅ **Backward compatible:** Default is `individualized_rewards = False`
✅ **Clean separation:** Original code preserved in else branches
✅ **Configurable:** Easy to tune via `contact_threshold`
✅ **Smooth weighting:** Linear decay avoids discontinuities
✅ **Well-documented:** Comments explain intent

### Potential Improvements (Future Work)

- **True contact detection:** Use Isaac Gym contact forces instead of distance proxy
- **Force-based attribution:** Weight by actual force applied to box
- **Velocity-based attribution:** Credit agents based on relative velocity toward box
- **Adaptive threshold:** Learn optimal contact threshold during training

---

## Comparison with Alternatives

### Alternative 1: True Contact Forces

**Pros:**
- Most accurate credit assignment
- Directly measures contribution

**Cons:**
- Requires Isaac Gym contact sensor integration
- More complex implementation
- Computational overhead

### Alternative 2: Velocity-Based Attribution

```python
# Credit based on agent velocity toward box
agent_vel_to_box = (box_pos - base_pos[:, i, :]) / (agent_box_dist + 1e-6)
velocity_alignment = torch.sum(base_vel[:, i, :] * agent_vel_to_box, dim=1)
weight = sigmoid(velocity_alignment)
```

**Pros:**
- Captures intent to approach/push
- More dynamic than static distance

**Cons:**
- Agents might get credit for moving toward box but not helping
- More complex to tune

### Alternative 3: Binary Step Function

```python
weight = 1.0 if agent_box_dist < contact_threshold else 0.0
```

**Pros:**
- Simple, clear threshold
- Easy to understand

**Cons:**
- Discontinuous (agents might oscillate around boundary)
- Not differentiable for gradient-based analysis

**Our Choice:** Smooth linear decay balances simplicity and continuity.

---

## Performance Expectations

### Metrics to Track

During training with individualized rewards:

1. **Success Rate:** Should be similar or better than shared rewards
   - Shared: ~80-90% at 100M steps
   - Individualized: Target ~85-95%

2. **Collision Rate:** Should be lower (better coordination)
   - Shared: ~8-12%
   - Individualized: Target ~5-10%

3. **Collaboration Degree:** Should be higher
   - Shared: ~0.70-0.80
   - Individualized: Target ~0.80-0.90

4. **Episode Length:** Might be slightly longer initially (both agents learning)
   - Should converge to similar values

### TensorBoard Monitoring

```bash
tensorboard --logdir HARL/results/mapush/cuboid/happo/
```

**Watch for:**
- `mapush/success_rate` - Should increase steadily
- `mapush/collision_rate` - Should decrease
- `average_episode_rewards` - Should increase (become less negative)

---

## Summary

Implemented **contact-based individualized rewards** to prevent freeloading in HAPPO multi-agent training:

- **3 rewards individualized:** reach_target, target (distance), push
- **Contact weight formula:** Linear decay based on distance to box
- **Configurable:** `individualized_rewards` flag and `contact_threshold` parameter
- **Backward compatible:** Default behavior unchanged
- **New task:** `go1push_mid_happo` with individualized rewards enabled

**Status:** Implementation complete, ready for testing.

**Next:** Run 1M step test to verify behavior, then full 100M training.
