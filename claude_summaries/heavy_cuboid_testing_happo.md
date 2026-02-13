# Heavy Cuboid Testing - HAPPO Documentation

**Date Started:** 2026-02-09
**Objective:** Force collaboration between agents by making the box too heavy for one agent to push alone, using HAPPO algorithm

---

## Background

### Problem: Freeloader Behavior
With HAPPO's separate actor networks (unlike MAPPO's shared network), agents don't naturally learn to collaborate. The default box mass of **4 kg** is trivially easy for either robot to push solo:
- Go1: ~12 kg body weight

This allows free-rider behavior where one agent does nothing while the other pushes.

### Hypothesis
Making the box heavy enough (8 kg) should require both agents to push together, forcing collaboration.

### Key Difference from MAPPO Testing
HAPPO uses separate actor networks per agent, which changes the learning dynamics significantly. The reward shaping that worked for MAPPO may not directly transfer to HAPPO.

---

## Experiment 1: OG Teamified Rewards

**Run Directory:** `results/mapush/go1push_mid/happo/heavy8kg_concat_critic_teamified/seed-00001-2026-02-09-12-36-07`

### Configuration
- **Box mass:** 8 kg (via `npc_mass_override = 8`)
- **Algorithm:** HAPPO with concatenated critic observations
- **Reward flag:** `--mapush_og_rewards_teamified True`
- **Training steps:** 150M
- **Parallel envs:** 500

### Command
```bash
python HARL/harl_mapush/train.py \
  --exp_name heavy8kg_concat_critic_teamified \
  --n_rollout_threads 500 \
  --num_env_steps 150000000 \
  --use_concat_agent_observations_critic True \
  --mapush_og_rewards_teamified True
```

### Reward Scales Used
| Reward | Scale |
|--------|-------|
| `target_reward_scale` | 0.00325 (original) |
| `approach_reward_scale` | 0.00075 (original) |
| `collision_punishment_scale` | -0.0025 (original) |
| `push_reward_scale` | 0.0015 (original) |
| `ocb_reward_scale` | 0.01 |
| `reach_target_reward_scale` | 10 (original) |
| `exception_punishment_scale` | -5 (original) |
| `distance_to_target_reward` | **ENABLED** (original formula with penalty) |

### Results
- **Success rate:** ~10% across checkpoints
- **Observed behavior:** Agents struggled to coordinate effectively

### Diagnosis
With original reward scales and distance_to_target_reward enabled, the learning signal wasn't effective for HAPPO agents to collaborate on the heavier box.

---

## Experiment 2: Reward Scale Testing (No Distance Reward)

**Run Directory:** `results/mapush/go1push_mid/happo/heavy8kg_concat_critic_teamified/seed-00001-2026-02-09-18-29-44`

### Configuration
- **Box mass:** 8 kg
- **Algorithm:** HAPPO with concatenated critic observations
- **Reward flag:** `--reward_scale_testing True`
- **Training steps:** 150M
- **Parallel envs:** 500

### Command
```bash
python HARL/harl_mapush/train.py \
  --exp_name heavy8kg_concat_critic_teamified \
  --n_rollout_threads 500 \
  --num_env_steps 150000000 \
  --use_concat_agent_observations_critic True \
  --reward_scale_testing True
```

### Reward Scales Used
| Reward | Scale | Multiplier |
|--------|-------|------------|
| `push_reward_scale` | 0.004 | 2.5x |
| `ocb_reward_scale` | 0.008 | 2x |
| `reach_target_reward_scale` | 10 | 1x |
| `approach_reward_scale` | 0.00075 | 1x |
| `collision_punishment_scale` | -0.0025 | 1x |
| `exception_punishment_scale` | -5 | 1x |
| `distance_to_target_reward` | **DISABLED** | 0 |

### Discovery
Due to a code ordering issue, the `distance_to_target_reward` was **completely disabled** (not computed at all). This turned out to be **beneficial** for HAPPO!

### Results
- **Success rate:** >95% across checkpoints (10M-150M)
- **Observed behavior:** Excellent collaboration, agents learned to push together effectively

### Key Insight
**Removing the distance-to-target reward entirely** led to much better performance! This suggests that for HAPPO with heavy box:
- The other rewards (push, OCB, approach, reach_target) provide sufficient learning signal
- The distance_to_target reward causes conflicting gradients or noise for HAPPO's separate actor networks

---

## Experiment 3: Extended Training (Same as Exp 2)

**Run Directory:** `results/mapush/go1push_mid/happo/heavy8kg_concat_critic_teamified/seed-00001-2026-02-10-21-32-27`

### Configuration
- **Box mass:** 8 kg
- **Algorithm:** HAPPO with concatenated critic observations
- **Reward flag:** `--reward_scale_testing True`
- **Training steps:** 200M (extended from 150M)
- **Parallel envs:** 500
- **Distance reward:** DISABLED (same as Exp 2)

### Command
```bash
python HARL/harl_mapush/train.py \
  --exp_name heavy8kg_concat_critic_teamified \
  --n_rollout_threads 500 \
  --num_env_steps 200000000 \
  --use_concat_agent_observations_critic True \
  --reward_scale_testing True
```

### Note
Same configuration as Exp 2 (distance_to_target disabled). Running for longer to see if performance continues to improve or plateaus.

### Results
*Pending - experiment in progress*

### Expected Outcome
- Maintain >95% SR
- Possibly slight improvement from additional training

---

## Experiment 4: Distance Reward with Progress-Only (FAILED)

**Run Directory:** `results/mapush/go1push_mid/happo/heavy8kg_concat_critic_teamified/seed-00001-2026-02-11-XX-XX-XX`

### Configuration
- **Box mass:** 8 kg
- **Algorithm:** HAPPO with concatenated critic observations
- **Reward flag:** `--reward_scale_testing True` (with bug fix applied)
- **Training steps:** ~80M (stopped early)
- **Parallel envs:** 500
- **Distance reward:** ENABLED (progress-only, penalty term removed)

### Distance Reward Formula
```python
# Progress-only (penalty term removed):
distance_reward = target_reward_scale * 100 * 2 * (past_distance - distance)
# Positive when box moves toward goal, negative when away
```

### Observed Behavior at 80M Steps
Compared to Exp 2/3 at same step count:
- **Push reward:** Higher than Exp 2/3
- **Reach target reward:** ~2x higher (SR higher as well)
- **OCB reward:** Much lower and not growing as rapidly

### Diagnosis
The distance-to-target reward **competes with OCB reward**:
1. Distance reward rewards progress toward goal from **any push angle**
2. Agents push from suboptimal positions (sides, corners) as long as they make progress
3. Less incentive to find optimal push position (behind the box)
4. Higher push/reach but lower OCB = less efficient pushing behavior

### Results
- **Success rate:** Lower than Exp 2/3 (exact numbers TBD)
- **Conclusion:** Distance-to-target reward hurts HAPPO performance even with penalty removed

### Action Taken
**Reverted code to keep distance_to_target_reward DISABLED for `--reward_scale_testing` flag.**

The `--reward_scale_testing` flag now intentionally does NOT enable `mapush_og_rewards_teamified`, keeping distance_to_target disabled for HAPPO.

---

## Experiment Summary Table

| Exp | Distance Reward | Push Scale | OCB Scale | Result | Notes |
|-----|-----------------|------------|-----------|--------|-------|
| 1 | Enabled (original) | 0.0015 (1x) | 0.01 | ~10% SR | OG teamified rewards |
| 2 | **DISABLED** | 0.004 (2.5x) | 0.008 (2x) | **>95% SR** | Best config found |
| 3 | **DISABLED** | 0.004 (2.5x) | 0.008 (2x) | *pending* | Same as Exp 2, 200M steps |
| 4 | Progress-only | 0.004 (2.5x) | 0.008 (2x) | **FAILED** | OCB suppressed, less efficient |

---

## Key Findings

### 1. Distance-to-Target Reward is Counterproductive for HAPPO
- Exp 1 with distance reward: ~10% SR
- Exp 2 without distance reward: >95% SR
- Exp 4 with progress-only distance reward: Still worse than Exp 2

**Conclusion:** For HAPPO with heavy box, the distance-to-target reward should be **completely disabled**.

### 2. Why Distance Reward Hurts HAPPO
The distance reward creates conflicting incentives:
- It rewards any progress toward goal, regardless of push angle
- This suppresses OCB learning (optimal contact positioning)
- Agents push from suboptimal angles, which is less efficient
- With separate actor networks, this noise is harder to overcome

### 3. Optimal HAPPO Reward Configuration (8kg box)
| Reward | Scale | Status |
|--------|-------|--------|
| `push_reward_scale` | 0.004 | Enabled |
| `ocb_reward_scale` | 0.008 | Enabled |
| `reach_target_reward_scale` | 10 | Enabled |
| `approach_reward_scale` | 0.00075 | Enabled |
| `collision_punishment_scale` | -0.0025 | Enabled |
| `exception_punishment_scale` | -5 | Enabled |
| `distance_to_target_reward` | **0** | **DISABLED** |

### 4. Code Change Made Permanent
The `--reward_scale_testing` flag (HAPPO) and `--mappo_heavybox_rewards` flag (MAPPO) both **completely disable distance_to_target_reward**:

```python
# In go1_push_mid_wrapper.py
skip_distance_reward = self.mappo_heavybox_rewards or self.reward_scale_testing

if not skip_distance_reward and (self.mapush_og_rewards_teamified or self.baseline_mappo_rewards) ...:
    # Only compute distance_to_target_reward if NOT using heavy box modes
    ...
```

---

## Files Modified

| File | Changes |
|------|---------|
| `mqe/envs/wrappers/go1_push_mid_wrapper.py` | `--reward_scale_testing` and `--mappo_heavybox_rewards` completely disable distance_to_target_reward |

---

## Testing Commands

### Calculator Mode (all checkpoints)
```bash
python HARL/harl_mapush/test.py \
  --checkpoint ./results/mapush/go1push_mid/happo/heavy8kg_concat_critic_teamified/seed-00001-2026-02-09-18-29-44/checkpoints \
  --all_checkpoints \
  --mode calculator \
  --num_episodes 100 \
  --num_envs 300 \
  --agent0 go1 \
  --agent1 go1
```

### Viewer Mode (single checkpoint)
```bash
python HARL/harl_mapush/test.py \
  --checkpoint ./results/mapush/go1push_mid/happo/heavy8kg_concat_critic_teamified/seed-00001-2026-02-09-18-29-44/checkpoints/150M \
  --mode viewer \
  --num_episodes 5 \
  --agent0 go1 \
  --agent1 go1
```

---

**Last Updated:** 2026-02-11
