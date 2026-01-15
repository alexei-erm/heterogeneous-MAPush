# CRITIC16: Goal-Centered Critic Representation

**Date**: January 9, 2026
**Status**: Training Started (seed 7, 900 envs, 200M steps)
**Branch**: `critic-fixing`

---

## Motivation

### Problems with CRITIC15 (Concatenated Ego-Centric Frames)

CRITIC15 used `use_concat_agent_observations_critic=True` (CRITIC10 approach):
- **Input**: 16 dims = `[agent0_ego_view(8d), agent1_ego_view(8d)]`
- **Problem**: Two different ego-centric coordinate frames concatenated together
- **Result**: Critic heatmaps showed NO clear spatial structure

#### Example Issue:
When varying agent 0's position while fixing agent 1:
```
Agent0's view: [target: 2.5m, box: 3.0m ahead, ...]  ← box far
Agent1's view: [target: 1.0m, box: 0.5m ahead, ...]  ← box near (SAME box, different frame!)
```

The critic sees the **same box** encoded in **two different coordinate systems** simultaneously. This makes learning spatial value structure unnecessarily complex.

### Key Insight: The Goal Never Moves!

Unlike the box (which agents push around), the **goal is stationary**. Using it as the reference frame provides:

1. **Stationary reference**: Goal position never changes during episode
2. **Task-aligned**: Success = box at goal = box at (0, 0) in goal frame
3. **Single coherent frame**: Everything in one consistent coordinate system
4. **Translation invariance**: Same relative configuration anywhere = same state

### Comparison with MAPPO Baseline

MAPPO critic shows **clear spatial structure** in heatmaps:
- Far from box → Low value (clear gradient)
- Near other agent → Very bad value (collision avoidance)
- Behind box on push side → High value (optimal positioning)

CRITIC15 failed to learn this despite 85% success rate and using the same rewards.

**Hypothesis**: Concatenated ego frames are too complex for the critic to extract clear spatial gradients, even though the policy still learns via HAPPO's sequential importance weighting.

---

## CRITIC16: Goal-Centered Representation

### Global State Structure (9 dimensions)

```
[x_box, y_box, yaw_box,           # Box position & orientation relative to goal (3 dims)
 x_agent0, y_agent0, yaw_agent0,  # Agent0 position & orientation relative to goal (3 dims)
 x_agent1, y_agent1, yaw_agent1]  # Agent1 position & orientation relative to goal (3 dims)
```

All coordinates in **goal reference frame**:
- Goal is at origin: `(0, 0)`
- Goal orientation is reference: `yaw_goal = 0`
- Success state: Box at `(0, 0, 0)` = box at goal with correct orientation

### Why Include Box Yaw?

Yaw = rotation around z-axis (vertical axis). For object manipulation:
- Position match: `(x_box, y_box) ≈ (0, 0)`
- Orientation match: `yaw_box ≈ 0`

Both required for task success. Box at goal but rotated 90° = failure.

### Comparison with Other Representations

| Critic | Reference Frame | Dims | Pros | Cons |
|--------|----------------|------|------|------|
| CRITIC7 | Absolute (world) | 11 | Simple | No translation invariance |
| CRITIC9 | Box-centered | 9 | Task-focused | Box moves during episode |
| CRITIC10/15 | Dual ego-centric | 16 | Rich info | Too complex, mixed frames |
| CRITIC11 | Relative observations | 9 | Compact | Less interpretable |
| **CRITIC16** | **Goal-centered** | **9** | **Stationary frame, task-aligned, single coherent system** | **None identified yet** |

---

## Implementation

### Command-Line Flag

```bash
--use_goal_centered_critic True
```

### Priority Order
When multiple flags are set, priority is:
```
relative_obs > concat_observations > goal_centered > box_centered > absolute
```

### Code Location

**Files Modified**:
1. `HARL/harl_mapush/train.py` (lines 46-47, 102, 118)
   - Added `--use_goal_centered_critic` argument
   - Pass flag to env_args

2. `HARL/harl/envs/mapush/mapush_env.py`:
   - Line 97: Added `use_goal_centered_critic` flag
   - Lines 126-143: Global state dimension calculation
   - Lines 259-285: Goal-centered state construction in `_construct_global_state()`
   - Lines 352-361: Diagnostic logging for CRITIC16

### State Construction

```python
# Box relative to goal
box_rel = box_pos_global[:, :2] - target_pos_global[:, :2]  # [n_envs, 2]
box_yaw_rel = box_rpy[:, 2:3] - target_rpy[:, 2:3]  # [n_envs, 1]

# Start with box relative position and yaw
global_state_list = [box_rel, box_yaw_rel]

# Add each agent's position and orientation relative to goal
for agent_id in range(self.n_agents):
    # Agent position relative to goal
    agent_pos_rel = base_pos[:, agent_id, :2] - target_pos_global[:, :2]  # [n_envs, 2]
    # Agent yaw relative to goal yaw
    agent_yaw_rel = base_rpy[:, agent_id, 2:3] - target_rpy[:, 2:3]  # [n_envs, 1]

    global_state_list.append(agent_pos_rel)
    global_state_list.append(agent_yaw_rel)

# Concatenate into single tensor
global_state_torch = torch.cat(global_state_list, dim=1)  # [n_envs, 9]
```

---

## Training Configuration

### Run Command
```bash
./run_training.sh --algo happo --env mapush --exp_name critic16 \
      --use_goal_centered_critic True \
      --mapush_og_rewards_teamified True \
      --seed 7
```

### Hyperparameters
- **Environments**: 900 parallel
- **Total steps**: 200M
- **Episode length**: 200 steps
- **Algorithm**: HAPPO
- **Seed**: 7
- **Critic architecture**: [256, 256, 128]
- **Actor architecture**: [256, 256]

### Reward Configuration
- **Base**: `--mapush_og_rewards_teamified True`
  - 7 original MAPush rewards converted to team rewards
  - Approach: AVERAGE of agent distance penalties
  - Collision: -0.0025 (original scale)
  - OCB: Continuous ±0.004 (averaged, original scale)
- **Collaboration bonus**: **DISABLED** (testing base case first)

### Important Notes
- **Actor observations**: Remain **agent-centric** (8 dims per agent, unchanged)
- **Critic input**: Goal-centered global state (9 dims)
- **Rewards**: Identical to CRITIC15 (teamified original MAPush)
- **Only difference**: Critic representation (concat ego → goal-centered)

---

## Expected Results

### Hypothesis: Clearer Spatial Value Structure

CRITIC16's goal-centered representation should enable the critic to learn clear spatial gradients:

**Expected heatmap patterns**:
1. **Distance to goal**: Box far from origin (goal) → LOW value
2. **Optimal positioning**: Agents behind box, pushing toward goal → HIGH value
3. **Collision zones**: Agents too close → BAD value
4. **Success region**: Box near (0, 0) → HIGHEST value

### Comparison Points

**MAPPO Baseline** (8-dim ego-centric, per-agent critic):
- ✓ Clear spatial structure in heatmaps
- ✓ Distance punishment visible
- ✓ Collision avoidance zones apparent
- Success rate: ~90% (OpenRL baseline)

**CRITIC15** (16-dim dual ego-centric, team critic):
- ✗ No clear spatial structure in heatmaps
- ✗ Values all negative, sometimes backwards
- ✓ 85% success rate (policy still learns via HAPPO)
- Problem: Dual ego frames too complex for critic

**CRITIC16** (9-dim goal-centered, team critic):
- ? Expected: Clear spatial structure (like MAPPO)
- ? Expected: Proper value landscape learning
- ? Expected: ≥85% success rate
- Advantage: Single coherent coordinate frame

### Success Criteria

1. **Heatmap Quality**: Clear spatial gradients in critic value heatmap
   - Far from goal → negative values
   - Near goal → positive values
   - Smooth gradients pointing toward (0, 0)

2. **Performance**: Match or exceed CRITIC15's 85% success rate

3. **Training Stability**: Smooth value function learning, no divergence

4. **Interpretability**: Value landscape makes intuitive sense
   - Box at (0, 0) should have highest value
   - Agents positioned correctly should increase value
   - Poor configurations should have low/negative value

---

## Related Work

### Previous Critics

- **CRITIC7**: Absolute world frame (11 dims) - No translation invariance
- **CRITIC9**: Box-centered (9 dims) - Moving reference frame
- **CRITIC10**: Concatenated ego-centric (16 dims) - Mixed frames, complex
- **CRITIC11**: Relative observations (9 dims) - Compact but less interpretable
- **CRITIC12**: Added cooperation bonuses (multi-tier)
- **CRITIC13**: Proximity penalty, disabled cooperation bonuses
- **CRITIC14**: Joint binary OCB (abandoned for continuous)
- **CRITIC15**: Continuous OCB + dual pushing bonus (85% SR, poor heatmaps)

### Key Design Decisions

1. **Goal vs Box as Reference**:
   - Goal: Stationary ✓
   - Box: Moves during episode ✗

2. **Single Frame vs Dual Frames**:
   - Single goal-centered: Coherent ✓
   - Dual ego-centric: Complex ✗

3. **Include Box Yaw**:
   - Yes: Full state representation ✓
   - No: Missing orientation info ✗

---

## Next Steps

1. **Monitor Training**:
   - Check diagnostic output at start (goal-centered state construction)
   - Track success rate progression
   - Watch for training instabilities

2. **Visualize Critic Values**:
   - Generate heatmaps at 10M, 50M, 100M, 200M steps
   - Compare with CRITIC15 and MAPPO baseline
   - Verify spatial structure emerges

3. **Ablation Study** (if CRITIC16 succeeds):
   - Add `--collaboration_rewards True` (dual pushing bonus)
   - Compare with/without collaboration bonus
   - Measure impact on success rate and episode length

4. **Hyperparameter Tuning** (if needed):
   - Critic learning rate (currently 0.005)
   - Critic architecture (currently [256, 256, 128])
   - Value normalization settings

---

## Files Modified

```
HARL/harl_mapush/train.py
HARL/harl/envs/mapush/mapush_env.py
claude_summaries/critic_learning_fixing/critic16.md  (this file)
```

## Git Status

Branch: `critic-fixing`

Modified files:
- `HARL/harl_mapush/train.py`
- `HARL/harl/envs/mapush/mapush_env.py`
- `claude_summaries/critic_learning_fixing/critic16.md`

Ready for commit after training validation.

---

## References

- CRITIC15 analysis: `claude_summaries/critic_learning_fixing/critic15.md`
- HAPPO algorithm: HARL framework
- Original MAPush: `backup_MAPush/` (OpenRL MAPPO baseline)
- Reward analysis: Issue with dual ego-centric frames making spatial learning difficult
