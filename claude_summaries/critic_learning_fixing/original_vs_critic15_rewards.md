# Original MAPush vs CRITIC15: Reward Comparison

> **Purpose:** Complete reference for how each reward was calculated in original MAPush vs CRITIC15 teamified implementation
> **Date:** January 7, 2026

---

## Active Rewards (7 total)

| # | Reward Name | Original Scale | CRITIC15 Scale | Status |
|---|-------------|---------------|----------------|--------|
| 1 | `reach_target_reward` | 10 | 10 | ✅ Identical |
| 2 | `distance_to_target_reward` | 0.00325 | 0.00325 | ✅ Identical formula |
| 3 | `approach_to_box_reward` | 0.00075 | 0.00075 | ⚠️ Teamified (averaged) |
| 4 | `collision_punishment` | -0.0025 | -0.0025 | ✅ Already team in original |
| 5 | `push_reward` | 0.0015 | 0.0015 | ✅ Already team in original |
| 6 | `ocb_reward` | 0.004 | ±0.004 | ⚠️ Changed to joint binary |
| 7 | `exception_punishment` | -5 | -5 | ✅ Identical |

---

## Detailed Comparison

### 1. `reach_target_reward`

**Scale:** 10

**Original MAPush:**
```python
# Shared team reward when box reaches target
if self.reach_target_reward_scale != 0:
    reward[self.finished_buf, :] += self.reach_target_reward_scale  # 10
    # Both agents get +10 when task succeeds
```

**CRITIC15:**
```python
# Identical - already team reward
if self.reach_target_reward_scale != 0:
    reward[self.finished_buf, :] += self.reach_target_reward_scale  # 10
```

**Difference:** ✅ **None - identical implementation**

---

### 2. `distance_to_target_reward`

**Scale:** 0.00325

**Original MAPush:**
```python
# Progress shaping + urgency penalty
if self.target_reward_scale != 0:
    if self.last_box_state is None:
        self.last_box_state = copy(box_state)

    past_distance = self.env.dist_calculator.cal_dist(self.last_box_state, target_state)
    distance = self.env.dist_calculator.cal_dist(box_state, target_state)

    # Formula: progress + urgency
    distance_reward = scale * 100 * (2 * (past_distance - distance) - 0.01 * distance)
    #                                  ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^  ^^^^^^^^^^^^^^
    #                                  Progress shaping (dense)         Urgency penalty (sparse)

    # Shared team reward
    reward[:, :] += distance_reward.unsqueeze(1).repeat(1, self.num_agents)
```

**CRITIC15:**
```python
# Identical formula - was disabled in CRITIC13, re-enabled in CRITIC15
if self.mapush_og_rewards_teamified and self.target_reward_scale != 0:
    # Same calculation as original
    distance_reward = self.target_reward_scale * 100 * (2 * (past_distance - distance) - 0.01 * distance)
    reward[:, :] += distance_reward.unsqueeze(1).repeat(1, self.num_agents)
```

**Difference:** ✅ **None - identical formula, same team structure**

**Components:**
- `2 * (past - curr)`: Dense progress shaping (+reward for moving closer)
- `-0.01 * distance`: Sparse urgency penalty (-reward for being far away)
- Net effect: Encourages both speed and progress

---

### 3. `approach_to_box_reward`

**Scale:** 0.00075

**Original MAPush:**
```python
# INDIVIDUAL penalty for each agent's distance to box
if self.approach_reward_scale != 0:
    for i in range(self.num_agents):
        distance = torch.norm(box_pos - base_pos[:, i, :], dim=1, keepdim=True)

        # Quadratic penalty: -(distance + 0.5)²
        distance_reward = (-(distance + 0.5)**2) * self.approach_reward_scale

        # Each agent gets their OWN penalty
        reward[:, i] += distance_reward.squeeze(-1)
```

**CRITIC15:**
```python
# TEAM reward: AVERAGE of both penalties (preserves scale magnitude)
if self.approach_reward_scale != 0:
    total_approach_penalty = torch.zeros(self.num_envs, device=self.device)

    for i in range(self.num_agents):
        distance = torch.norm(box_pos - base_pos[:, i, :], dim=1)

        # Same quadratic formula
        distance_penalty = (-(distance + 0.5)**2) * self.approach_reward_scale
        total_approach_penalty += distance_penalty

    # AVERAGE to preserve designed scale magnitude
    if self.mapush_og_rewards_teamified:
        total_approach_penalty = total_approach_penalty / self.num_agents

    # Both agents get the AVERAGED penalty
    reward[:, :] += total_approach_penalty.unsqueeze(1).repeat(1, self.num_agents)
```

**Difference:** ⚠️ **Teamified via averaging**

**Why averaging?**
- Original: Each agent penalized individually for their distance
- CRITIC15: Sum both penalties, then divide by 2
- Preserves per-agent scale magnitude (0.00075)
- Maintains centralized critic compatibility (team rewards)

**Example:**
```
Agent 0 distance: 2.0m → penalty = -(2.5)² * 0.00075 = -0.0047
Agent 1 distance: 1.0m → penalty = -(1.5)² * 0.00075 = -0.0017

Original:
  reward[0] = -0.0047 (agent 0 only)
  reward[1] = -0.0017 (agent 1 only)

CRITIC15:
  avg_penalty = (-0.0047 + -0.0017) / 2 = -0.0032
  reward[0] = reward[1] = -0.0032 (both agents)
```

---

### 4. `collision_punishment`

**Scale:** -0.0025

**Original MAPush:**
```python
# Penalty for agents being too close (inter-robot collision)
if self.collision_punishment_scale != 0:
    for i in range(self.num_agents):
        for j in range(i+1, self.num_agents):
            distance = torch.norm(base_pos[:, i, :] - base_pos[:, j, :], dim=1, keepdim=True)

            # Inverse distance penalty: closer = worse
            collision_punishment = (1 / (0.02 + distance/3)) * self.collision_punishment_scale

            # BOTH agents get the same punishment
            reward[:, i] += collision_punishment.squeeze(-1)
            reward[:, j] += collision_punishment.squeeze(-1)
```

**CRITIC15:**
```python
# Identical structure (was already team reward in original)
if self.collision_punishment_scale != 0:
    # Only 2 agents, so single pair calculation
    agent_distance = torch.norm(base_pos[:, 0, :] - base_pos[:, 1, :], dim=1)

    # Override scale with original -0.0025 when flag is True
    collision_scale = -0.0025 if self.mapush_og_rewards_teamified else self.collision_punishment_scale

    collision_punishment = (1 / (0.02 + agent_distance / 3)) * collision_scale

    # Both agents get the same punishment (explicit broadcast)
    reward[:, :] += collision_punishment.unsqueeze(1).repeat(1, self.num_agents)
```

**Difference:** ✅ **Identical behavior - original already gave both agents same punishment**

**Note:** Scale was changed to -0.0008 in Iter4, CRITIC15 restores original -0.0025

**Penalty curve:**
```
Distance | Penalty (scale=-0.0025)
---------|------------------------
0.1m     | -0.0476  (very close!)
0.3m     | -0.0208
0.6m     | -0.0114
1.2m     | -0.0060
2.4m     | -0.0030
```

---

### 5. `push_reward`

**Scale:** 0.0015

**Original MAPush:**
```python
# Shared reward when box is moving (velocity > 0.1 m/s)
if self.push_reward_scale != 0:
    push_reward = torch.zeros((self.env.num_envs,), device=self.env.device)

    # Check if box velocity exceeds threshold
    box_velocity = torch.norm(self.root_states_npc.reshape(self.num_envs, self.num_npcs, -1)[:, 0, 7:9], dim=1)
    push_reward[box_velocity > 0.1] = self.push_reward_scale  # Binary: 0.0015 or 0

    # Both agents get same reward
    reward[:, :] += push_reward.unsqueeze(1).repeat(1, self.num_agents)
```

**CRITIC15:**
```python
# Identical - already team reward in original
if self.push_reward_scale != 0:
    # Check if box is moving
    box_moving = torch.norm(self.root_states_npc.reshape(self.num_envs, self.num_npcs, -1)[:, 0, 7:9], dim=1) > 0.1

    push_reward = torch.zeros((self.env.num_envs,), device=self.env.device)
    push_reward[box_moving] = self.push_reward_scale

    reward[:, :] += push_reward.unsqueeze(1).repeat(1, self.num_agents)
```

**Difference:** ✅ **None - identical implementation**

**Behavior:** Binary reward for any box motion (direction-agnostic)

---

### 6. `ocb_reward` (Optimal Contact Behavior)

**Scale:** 0.004

**Original MAPush:**
```python
# INDIVIDUAL continuous reward based on agent positioning quality
if self.ocb_reward_scale != 0:
    # Calculate push direction (box → target)
    target_direction = (target_pos[:, :2] - box_pos[:, :2]) / torch.norm((target_pos[:, :2] - box_pos[:, :2]), dim=1, keepdim=True)

    vertex_list = self.cfg.asset.vertex_list  # Box corners

    for i in range(self.num_agents):
        # 1. Get agent position relative to box
        gf_pos = base_pos[:, i, :2] - box_pos[:, :2]

        # 2. Rotate to box frame
        rotation_matrix = rotation_matrix_2D(-box_rpy[:, 2])
        box_relative_pos = torch.bmm(rotation_matrix, gf_pos.unsqueeze(2)).squeeze(2)

        # 3. Find nearest box edge and its normal vector
        normal_vector = self.calc_normal_vector_for_obc_reward(vertex_list, box_relative_pos)

        # 4. Rotate normal back to global frame
        rotation_matrix = rotation_matrix_2D(box_rpy[:, 2])
        normal_vector = torch.bmm(rotation_matrix, normal_vector.unsqueeze(2)).squeeze(2)

        # 5. CONTINUOUS reward: dot product of normal with target direction
        #    Positive = behind box (correct), Negative = in front (wrong)
        ocb_reward = torch.sum(target_direction * normal_vector, dim=1) * self.ocb_reward_scale

        # Each agent gets their OWN continuous OCB value
        reward[:, i] += ocb_reward  # Range: -0.004 to +0.004 per agent
```

**CRITIC15:**
```python
# JOINT BINARY reward - both agents must be on correct side
if self.ocb_reward_scale != 0:
    target_direction = (target_pos[:, :2] - box_pos[:, :2]) / torch.norm((target_pos[:, :2] - box_pos[:, :2]), dim=1, keepdim=True)
    vertex_list = self.cfg.asset.vertex_list

    # Calculate raw OCB for each agent (same as original)
    raw_ocb_list = []
    for i in range(self.num_agents):
        gf_pos = base_pos[:, i, :2] - box_pos[:, :2]
        rotation_matrix = rotation_matrix_2D(-box_rpy[:, 2])
        box_relative_pos = torch.bmm(rotation_matrix, gf_pos.unsqueeze(2)).squeeze(2)
        normal_vector = self.calc_normal_vector_for_obc_reward(vertex_list, box_relative_pos)
        rotation_matrix = rotation_matrix_2D(box_rpy[:, 2])
        normal_vector = torch.bmm(rotation_matrix, normal_vector.unsqueeze(2)).squeeze(2)

        raw_ocb = torch.sum(target_direction * normal_vector, dim=1)  # -1 to +1
        raw_ocb_list.append(raw_ocb)

    # NEW: Joint binary condition - BOTH must be correct
    both_correct = (raw_ocb_list[0] > 0) & (raw_ocb_list[1] > 0)

    # NEW: Binary reward (only 2 values possible)
    if self.mapush_og_rewards_teamified:
        # CRITIC15: Symmetric ±0.004
        joint_ocb_reward = torch.where(
            both_correct,
            torch.full_like(raw_ocb_list[0], 0.004),   # Both correct: +0.004
            torch.full_like(raw_ocb_list[0], -0.004)   # Any wrong: -0.004
        )

    # Both agents get the SAME binary reward
    reward[:, :] += joint_ocb_reward.unsqueeze(1).repeat(1, self.num_agents)
```

**Difference:** ⚠️ **MAJOR CHANGE - continuous → joint binary**

| Aspect | Original | CRITIC15 |
|--------|----------|----------|
| Type | Continuous per-agent | Binary joint |
| Values | -0.004 to +0.004 (smooth) | Only ±0.004 (discrete) |
| Assignment | Individual | Team (both get same) |
| Condition | Agent's own positioning | BOTH agents must be correct |

**Example:**
```
Agent 0 raw_ocb: +0.8 (behind box, good positioning)
Agent 1 raw_ocb: -0.3 (in front, bad positioning)

Original:
  reward[0] = +0.8 * 0.004 = +0.0032 ✅
  reward[1] = -0.3 * 0.004 = -0.0012 ❌

CRITIC15:
  both_correct = False (agent 1 is wrong)
  reward[0] = reward[1] = -0.004 ❌ Both punished!
```

**Why changed?**
- Added in CRITIC12 v5 to enforce cooperation
- Original per-agent OCB could cancel out when summed for team reward
- Joint binary ensures agents must coordinate positioning

**v2 Proposal (in critic15.md):**
- Restore continuous OCB
- Average both agents' continuous values
- Better credit assignment, closer to original

---

### 7. `exception_punishment`

**Scale:** -5

**Original MAPush:**
```python
# Large penalty for simulation exceptions (NaN, Inf, robot flip, etc.)
if self.exception_punishment_scale != 0:
    reward[self.exception_buf, :] += self.exception_punishment_scale  # -5
    reward[self.value_exception_buf, :] += self.exception_punishment_scale  # -5
```

**CRITIC15:**
```python
# Identical
if self.exception_punishment_scale != 0:
    reward[self.exception_buf, :] += self.exception_punishment_scale
    reward[self.value_exception_buf, :] += self.exception_punishment_scale
```

**Difference:** ✅ **None - identical implementation**

---

## Summary Table

| Reward | Original Type | CRITIC15 Type | Formula Change | Scale Change |
|--------|---------------|---------------|----------------|--------------|
| `reach_target` | Team | Team | None | None |
| `distance_to_target` | Team | Team | None | None |
| `approach_to_box` | **Individual** | **Team (avg)** | None | None |
| `collision` | Team | Team | None | Restored -0.0025 |
| `push` | Team | Team | None | None |
| `ocb` | **Individual continuous** | **Team binary** | **Continuous → Binary** | None |
| `exception` | Team | Team | None | None |

---

## Disabled in CRITIC15 (Not Original)

| Reward | Added In | Scale | Why Disabled |
|--------|----------|-------|--------------|
| `goal_push_bonus` | CRITIC12 v7 | 0.01 | Not in original MAPush |
| `proximity_penalty` | CRITIC13 v3 | 0.002 | Not in original MAPush |

---

## Key Takeaways

1. **5/7 rewards identical** to original (reach_target, distance_to_target, collision, push, exception)
2. **2/7 rewards teamified** for centralized critic compatibility:
   - `approach_to_box`: Individual → Team (averaged)
   - `ocb_reward`: Individual continuous → Team binary
3. **All scales preserved** from original MAPush (except collision restored from -0.0008 to -0.0025)
4. **Team reward structure** enables centralized critic to learn accurate value estimates

---

## References

- Original MAPush: `/home/gvlab/backup_MAPush/mqe/envs/wrappers/go1_push_mid_wrapper.py`
- CRITIC15: `/home/gvlab/new-universal-MAPush/mqe/envs/wrappers/go1_push_mid_wrapper.py`
- Config: `/home/gvlab/new-universal-MAPush/task/cuboid/config.py`
