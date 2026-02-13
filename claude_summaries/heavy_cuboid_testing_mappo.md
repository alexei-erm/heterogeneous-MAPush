# Heavy Cuboid Testing Documentation

**Date Started:** 2026-02-04
**Objective:** Force collaboration between agents by making the box too heavy for one agent to push alone

---

## Background

### Problem: Freeloader Behavior
With HAPPO's separate actor networks (unlike MAPPO's shared network), agents don't naturally learn to collaborate. The default box mass of **4 kg** is trivially easy for either robot to push solo:
- Go1: ~12 kg body weigh
- Anymal C: ~50 kg body weight

This allows free-rider behavior where one agent does nothing while the other pushes.

### Hypothesis
Making the box heavy enough (8 kg) should require both agents to push together, forcing collaboration.

---

## Experiment 1: Heavy Box with Original Reward Scales

**Run Directory:** `log/MQE/go1push_mid/heavy_homogen_MAPPObaseline_go1/run1`

### Configuration
- **Box mass:** 8 kg (via `npc_mass_override = 8`)
- **Reward scales:** Original MAPush baseline values
  - `target_reward_scale = 0.00325`
  - `push_reward_scale = 0.0015`
  - `ocb_reward_scale = 0.004`

### Results
- **Success rate:** ~2% across all checkpoints
- **Observed behavior:** Agents learned to:
  - Stay apart (collision avoidance reward)
  - Position behind box (OCB reward)
  - **NOT push toward goal**

### Diagnosis
OCB reward dominated learning. Agents found local optimum of "just stand behind box" without actually pushing. The push_reward signal was too weak relative to OCB.

---

## Experiment 2: Adjusted Reward Scales (3x target, 4x push, 0.75x OCB)

**Run Directory:** `log/MQE/go1push_mid/heavyrewards_homogen_MAPPObaseline_go1/run1`

### Note: Config Override Bug Discovered & Fixed
Initially, the `--baseline_mappo_rewards True` flag in `train.sh` was **HARD-OVERRIDING** all reward scales with the original baseline values in `mqe/envs/utils.py:228-237`, ignoring the config file completely. This was fixed by modifying `custom_cfg()` to use the adjusted scales.

### Configuration
- **Box mass:** 8 kg
- **Reward scales (actually applied this time):**
  - `target_reward_scale = 0.01` (3x increase from 0.00325)
  - `push_reward_scale = 0.006` (4x increase from 0.0015)
  - `ocb_reward_scale = 0.0075` (0.75x from 0.01, ~1.875x from original 0.004)

### Observations (Early Training ~10M steps)
- **push_reward:** Increased significantly! ~0.00035 vs ~0.000006 in Experiment 1
  - This is ~58x higher, indicating agents ARE pushing more
- **distance_to_target_reward:** Stuck at **-0.022** instead of improving
  - In Experiment 1 it was ~-0.007 and improved over time
  - The 3x scale increase also 3x'd the distance PENALTY component

### Problem Discovered: Target Reward Formula Issue

The formula in `go1_push_mid_wrapper.py:585`:
```python
distance_reward = self.target_reward_scale * 100 * (2 * (past_distance - distance) - 0.01 * distance)
```

Has two components:
1. `2 * (past_distance - distance)` - progress reward (good to scale up)
2. `-0.01 * distance` - constant distance penalty (BAD to scale up)

When `target_reward_scale` increased 3x:
- Progress term: 3x stronger (good!)
- Distance penalty: 3x stronger (counterproductive!)

The penalty term dominated early training, keeping reward stuck at -0.022.

---

## Experiment 3: Push Reward Only

**Run Directory:** `log/MQE/go1push_mid/heavyrewards_homogen_MAPPObaseline_go1/run2`

### Rationale
Since push_reward showed the most promising improvement (58x increase in actual pushing behavior), focus on that alone without the problematic target_reward scaling.

### Configuration
- **Box mass:** 8 kg
- **Reward scales:**
  - `target_reward_scale = 0.00325` (reverted to original)
  - `push_reward_scale = 0.006` (4x increase - ONLY CHANGE)
  - `ocb_reward_scale = 0.004` (reverted to original)
  - `reach_target_reward_scale = 10` (original)

### Results
- **Success rate:** 20-27% across checkpoints
- **Observed behavior:**
  - Agents learned to **push** (major improvement from Exp 1!)
  - Sometimes push **together** (collaboration emerging)
  - But **NOT necessarily toward the goal** - pushing in random directions

### Diagnosis
The 4x `push_reward` successfully incentivized pushing behavior, but the reward doesn't care about push *direction*. The `target_reward` (goal direction signal) and `reach_target_reward` (success bonus) were too weak relative to the amplified push reward.

---

## Experiment 4: Goal-Directed Pushing (Current)

**Run Directory:** `log/MQE/go1push_mid/heavyrewards_homogen_MAPPObaseline_go1/run3`

### Rationale
Exp 3 showed agents CAN push (20-27% SR), but lack goal direction. Two adjustments:
1. **Reduce `push_reward`** from 4x to 2.5x - still encourages pushing but less dominant
2. **Increase `reach_target_reward`** from 10 to 50 (5x) - stronger sparse signal that reaching the goal matters

Why not increase `target_reward_scale`? The formula has a penalty term that also scales (discovered in Exp 2), causing counterproductive behavior.

### Configuration
- **Box mass:** 8 kg
- **Reward scales:**
  - `target_reward_scale = 0.00325` (original - avoid penalty scaling issue)
  - `push_reward_scale = 0.004` (2.5x, reduced from Exp 3's 4x)
  - `ocb_reward_scale = 0.004` (original)
  - `reach_target_reward_scale = 50` (5x increase from 10)

### Scale Comparison Table

| Reward | Original | Exp 3 | Exp 4 |
|--------|----------|-------|-------|
| `target_reward_scale` | 0.00325 | 0.00325 | 0.00325 |
| `push_reward_scale` | 0.0015 | 0.006 (4x) | **0.004 (2.5x)** |
| `ocb_reward_scale` | 0.004 | 0.004 | 0.004 |
| `reach_target_reward_scale` | 10 | 10 | **50 (5x)** |

### Expectations
- Agents should still push (2.5x push reward maintains incentive)
- Stronger goal signal (5x success bonus) should bias pushing toward target
- Success rate should improve beyond 27%

### Results
- **Success rate:** 24-29% across checkpoints (90M: 29%, 100M: 24.7%)
- **Collaboration degree:** Improved! 0.12 → 0.24 at 100M
- **Observed behavior:**
  - Agents push together more (collaboration doubled)
  - Still not consistently goal-directed
  - Similar success rate to Exp 3 despite 5x reach bonus

### Diagnosis
The sparse `reach_target_reward` (only on success) wasn't enough to guide pushing direction. The dense `target_reward` signal is needed, but the penalty term in the formula was blocking it from being scaled up effectively.

---

## Experiment 5: Remove Distance Penalty Term (Current)

**Run Directory:** `log/MQE/go1push_mid/heavyrewards_homogen_MAPPObaseline_go1/run4`

### Rationale
The root cause identified: the `distance_reward` formula has a **penalty term** that scales with `target_reward_scale`:

```python
# BEFORE (problematic):
distance_reward = scale * 100 * (2 * (past_distance - distance) - 0.01 * distance)
#                                    ↑ PROGRESS (good)           ↑ PENALTY (bad to scale)
```

When scaling up `target_reward_scale`:
- Progress term gets stronger (good!)
- Penalty term ALSO gets stronger (bad - constant punishment for being far)

With heavy box, agents can't move it early in training, so:
- Progress ≈ 0 (box barely moves)
- Penalty = constant negative → agents always punished

**Solution:** Remove the penalty term entirely.

### Code Change

**File:** `mqe/envs/wrappers/go1_push_mid_wrapper.py` line 585

```python
# BEFORE:
distance_reward = self.target_reward_scale * 100 * (2 * (past_distance - distance) - 0.01 * distance)

# AFTER (Exp 5):
distance_reward = self.target_reward_scale * 100 * 2 * (past_distance - distance)
```

### Configuration
- **Box mass:** 8 kg
- **Reward scales:**
  - `target_reward_scale = 0.01` (3x increase - NOW SAFE without penalty!)
  - `push_reward_scale = 0.004` (2.5x, same as Exp 4)
  - `ocb_reward_scale = 0.004` (original)
  - `reach_target_reward_scale = 50` (5x, same as Exp 4)

### Scale Comparison Table

| Reward | Original | Exp 3 | Exp 4 | Exp 5 |
|--------|----------|-------|-------|-------|
| `target_reward_scale` | 0.00325 | 0.00325 | 0.00325 | **0.01 (3x)** |
| `push_reward_scale` | 0.0015 | 0.006 (4x) | 0.004 (2.5x) | 0.004 (2.5x) |
| `ocb_reward_scale` | 0.004 | 0.004 | 0.004 | 0.004 |
| `reach_target_reward_scale` | 10 | 10 | 50 (5x) | 50 (5x) |
| **Penalty term** | Yes | Yes | Yes | **REMOVED** |

### Expectations
- 3x stronger progress signal should guide pushing toward goal
- No penalty means agents aren't punished for distance early in training
- Combined with 2.5x push + 5x reach, should see significant SR improvement
- Target: >40% success rate

### Results
*Stopped early - moved to Exp 6*

---

## Experiment 6: Reduce Variance (Revert Reach Bonus)

**Run Directory:** `log/MQE/go1push_mid/heavyrewards_homogen_MAPPObaseline_go1/run5`

### Rationale
Exp 4 showed a critical pattern: **everything peaked at 40M then collapsed together** (SR, push, OCB, avg step reward all fell). This is NOT reward hacking - it's policy instability/divergence.

**Root cause identified:** The 5x `reach_target_reward = 50` creates **high variance** in episode returns:
- Success episode: ~52 return (50 sparse + ~2 dense)
- Failure episode: ~1-2 return (dense only)

This ~25-50x variance destabilizes the value function → unstable policy → collapse.

### Solution
Keep the Exp 5 improvements (penalty removal, 3x target, 2.5x push) but **revert reach_target back to 10** to reduce variance.

### Configuration
- **Box mass:** 8 kg
- **Reward scales:**
  - `target_reward_scale = 0.01` (3x increase - penalty term REMOVED)
  - `push_reward_scale = 0.004` (2.5x increase)
  - `ocb_reward_scale = 0.004` (original)
  - `reach_target_reward_scale = 10` (REVERTED from 50)
- **Penalty term:** REMOVED (from Exp 5)

### Scale Comparison Table

| Reward | Original | Exp 4 | Exp 5 | Exp 6 |
|--------|----------|-------|-------|-------|
| `target_reward_scale` | 0.00325 | 0.00325 | 0.01 (3x) | 0.01 (3x) |
| `push_reward_scale` | 0.0015 | 0.004 (2.5x) | 0.004 (2.5x) | 0.004 (2.5x) |
| `ocb_reward_scale` | 0.004 | 0.004 | 0.004 | 0.004 |
| `reach_target_reward_scale` | 10 | 50 (5x) | 50 (5x) | **10 (1x)** |
| **Penalty term** | Yes | Yes | NO | NO |

### Expectations
- More stable training (lower variance from sparse reward)
- 3x target reward (without penalty) should guide goal direction
- Should NOT collapse after 40M like Exp 4
- Target: stable >30% SR without collapse

### Results
- **Success rate:** ~60%
- **Observed behavior:**
  - Agents push toward goal (target reward working!)
  - Agents collaborate (heavy box requires both)
  - **BUT:** Agents not pushing from behind (OCB too weak relative to other rewards)

### Diagnosis
The balance shifted too far - now push/target dominate and OCB is too weak to encourage optimal pushing position (from behind the box).

---

## Experiment 7: Increase OCB for Optimal Positioning

**Run Directory:** `log/MQE/go1push_mid/heavyrewards_homogen_MAPPObaseline_go1/run6`

### Rationale
Exp 6 achieved 60% SR but agents push from suboptimal angles. Need to bring OCB back up to guide *where* they push from while maintaining the goal-directed pushing behavior.

### Configuration
- **Box mass:** 8 kg
- **Reward scales:**
  - `target_reward_scale = 0.01` (3x - same as Exp 6)
  - `push_reward_scale = 0.004` (2.5x - same as Exp 6)
  - `ocb_reward_scale = 0.008` (2x increase from 0.004)
  - `reach_target_reward_scale = 10` (original - same as Exp 6)
- **Penalty term:** REMOVED (same as Exp 6)

### Scale Comparison Table

| Reward | Original | Exp 6 | Exp 7 |
|--------|----------|-------|-------|
| `target_reward_scale` | 0.00325 | 0.01 (3x) | 0.01 (3x) |
| `push_reward_scale` | 0.0015 | 0.004 (2.5x) | 0.004 (2.5x) |
| `ocb_reward_scale` | 0.004 | 0.004 (1x) | **0.008 (2x)** |
| `reach_target_reward_scale` | 10 | 10 (1x) | 10 (1x) |
| **Penalty term** | Yes | NO | NO |

### Expectations
- Maintain 60%+ SR from Exp 6
- Agents should now push from behind the box (optimal position)
- Better collaboration due to proper positioning
- Target: >65% SR with optimal pushing behavior

### Results
*Pending - experiment in progress*

---

## Configuration Reference

### MAPPO Reward Flags (Updated 2026-02-11)

**Two flags available for MAPPO:**

| Flag | Description | Distance Penalty | Scales |
|------|-------------|------------------|--------|
| `--baseline_mappo_rewards True` | TRUE ORIGINAL MAPush rewards | **INCLUDED** | Original |
| `--mappo_heavybox_rewards True` | Heavy box training (8kg) | **REMOVED** | Exp 7 |

### Flag Details

**`--baseline_mappo_rewards True` (DEFAULT):**
- Uses TRUE ORIGINAL 7 MAPush rewards
- Uses ORIGINAL scales: target=0.00325, push=0.0015, ocb=0.004
- Uses ORIGINAL distance formula WITH penalty term
- For standard 4kg box training

**`--mappo_heavybox_rewards True`:**
- Uses 6 rewards (distance_to_target COMPLETELY DISABLED)
- Scales: push=0.004 (2.5x), ocb=0.008 (2x)
- distance_to_target_reward: COMPLETELY DISABLED (not just penalty removed)
- For heavy 8kg box training

### Usage Examples

```bash
# Standard training (4kg box) - uses TRUE ORIGINAL rewards
python ./openrl_ws/train.py --baseline_mappo_rewards True ...

# Heavy box training (8kg box) - uses Exp 7 optimized scales
python ./openrl_ws/train.py --mappo_heavybox_rewards True ...
```

### Box Mass Override
- Location: `task/cuboid/config.py` (for MAPPO) or `mqe/envs/configs/go1_push_mid_config.py` (for HAPPO)
- Parameter: `asset.npc_mass_override`
- Default: `None` (uses URDF default of 4 kg)
- For heavy box experiments: `8` kg

---

## Key Learnings

1. **Always verify config is actually being used** - The `--baseline_mappo_rewards` flag silently overrides config file values

2. **Check reward formulas before scaling** - Scaling a reward scale may affect multiple terms in unexpected ways (e.g., penalty terms getting scaled too)

3. **Push reward is the key signal** - Even with heavy box, the 4x push reward increase showed 58x actual improvement in pushing behavior

4. **OCB reward can dominate** - With original scales, agents optimize for positioning (OCB) rather than actual task completion (pushing to goal)

5. **Distance penalty hurts heavy box training** - The constant penalty term punishes agents when box is hard to move, leading to poor learning

---

## Files Modified

| File | Changes |
|------|---------|
| `mqe/envs/utils.py` | Added `--mappo_heavybox_rewards` flag; restored `--baseline_mappo_rewards` to TRUE ORIGINAL scales |
| `mqe/envs/wrappers/go1_push_mid_wrapper.py` | Added `mappo_heavybox_rewards` handling; distance penalty only removed for heavy box mode |
| `openrl_ws/utils.py` | Added `--mappo_heavybox_rewards` argument |
| `openrl_ws/train.py` | Added `mappo_heavybox_rewards` flag handling |
| `task/cuboid/config.py` | Set `npc_mass_override = 8` for heavy box experiments |

---

## Experiment Summary Table

| Exp | Target Scale | Push Scale | OCB Scale | Reach Scale | Penalty | Result | Issue |
|-----|--------------|------------|-----------|-------------|---------|--------|-------|
| 1 | 0.00325 (1x) | 0.0015 (1x) | 0.004 (1x) | 10 | Yes | ~2% SR | Agents just stand behind box |
| 2 | 0.01 (3x) | 0.006 (4x) | 0.003 (0.75x) | 10 | Yes | N/A | Penalty term scaling broke training |
| 3 | 0.00325 (1x) | 0.006 (4x) | 0.004 (1x) | 10 | Yes | 20-27% SR | Push works but not goal-directed |
| 4 | 0.00325 (1x) | 0.004 (2.5x) | 0.004 (1x) | 50 (5x) | Yes | Peak 29% @ 40M → **COLLAPSED** | High variance from 50x sparse reward |
| 5 | 0.01 (3x) | 0.004 (2.5x) | 0.004 (1x) | 50 (5x) | NO | *stopped* | Moved to Exp 6 |
| 6 | 0.01 (3x) | 0.004 (2.5x) | 0.004 (1x) | 10 (1x) | NO | **~60% SR** | Agents not pushing from behind |
| 7 | 0.01 (3x) | 0.004 (2.5x) | **0.008 (2x)** | 10 (1x) | NO | *pending* | Increase OCB for optimal positioning |
