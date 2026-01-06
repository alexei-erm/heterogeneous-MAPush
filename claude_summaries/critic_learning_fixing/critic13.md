# CRITIC13: Minimal Essentials Reward Subset

> **Date:** January 5, 2026
> **Philosophy:** Remove all redundancy, keep only non-overlapping core signals
> **Goal:** Simplify reward structure while maintaining all essential signals
> **Status:** IMPLEMENTED v2 - curriculum learning approach
> **Parent:** Derived from CRITIC12 v5 (best run)

---

## What Changed From CRITIC12

### Core Philosophy Shift

**CRITIC12:** Rich cooperation signals with intentional redundancy
**CRITIC13:** Minimal non-overlapping signals, zero redundancy

### Removed Rewards (4 removed)

| Removed | Scale | Reason |
|---------|-------|--------|
| ❌ `push_reward` | 0.0015 | Redundant with `goal_push_bonus` (direction-agnostic version) |
| ❌ `dual_engagement_bonus` | 0.003 | Redundant with `approach_to_box_reward` |
| ❌ `synchronized_contact_bonus` | 0.008 | Redundant with `approach_to_box_reward` |
| ❌ `bilateral_push_bonus` | 0.01 | Redundant with `goal_push_bonus` |

### Modified Rewards (1 modified)

| Modified | Old | New | Change |
|----------|-----|-----|--------|
| `distance_to_target_reward` | `scale * 100 * (2*(past-curr) - 0.01*dist)` | `scale * 100 * 2*(past-curr)` | **REMOVED `-0.01*distance` penalty term** |

**Why:** The distance penalty was causing net negative rewards even when making progress. Let progress shaping be purely positive.

### Kept Rewards (6 kept)

| Reward | Scale | Purpose |
|--------|-------|---------|
| ✅ `reach_target_reward` | 10 | Sparse success signal (required) |
| ✅ `distance_to_target_reward` | 0.00325 | Progress metric (penalty term removed) |
| ✅ `approach_to_box_reward` | 0.00075 | Individual engagement (prevents freeloading) |
| ✅ `goal_push_bonus` | 0.01 | Directional velocity (replaces push_reward) |
| ✅ `ocb_reward` | +0.01/-0.004 | Positioning signal (asymmetric) |
| ✅ `collision_punishment` | -0.0025 | Safety |
| ✅ `exception_punishment` | -5 | Safety |

**Total Active Rewards:** 7 (down from 11 in CRITIC12)

---

## CRITIC12 vs CRITIC13 Comparison

| Aspect | CRITIC12 v5 (best) | CRITIC13 v1 |
|--------|-------------------|-------------|
| **Total Rewards** | 11 active | 7 active |
| **Cooperation Bonuses** | 3 (dual, sync, bilateral) | 0 |
| **OCB** | Symmetric ±0.004 | Asymmetric +0.01/-0.004 |
| **Distance Penalty** | Yes (-0.01*dist) | No (removed) |
| **Push Rewards** | 2 (push_reward + bilateral) | 1 (goal_push_bonus only) |
| **Philosophy** | Rich cooperation signals | Minimal essentials |
| **Freeloading Prevention** | Team bonuses | Individual approach penalty |

---

## Hypothesis: Why This Might Work Better

### Problem with CRITIC12 v6-v7

After removing `dual_engagement_bonus` in v6:
- OCB went negative (-0.002)
- Agents stopped exploring near box
- Never learned correct positioning

### CRITIC13 Approach

1. **Keep `approach_to_box_reward`** - individual penalty prevents freeloading
2. **Remove cooperation bonuses** - eliminate redundancy with approach
3. **Remove distance penalty** - stop punishing agents for being far from goal
4. **Boost directional push** - 0.01 scale makes it primary shaping signal
5. **Asymmetric OCB** - strong positive for correct positioning

### Expected Outcome

- Simpler reward landscape → clearer learning signal
- `goal_push_bonus` (0.01) becomes dominant directional signal
- `approach_to_box_reward` prevents freeloading without cooperation bonuses
- No distance penalty → agents not discouraged when far from goal

---

## Implementation Details

### Code Changes Required

1. **Remove cooperation bonus code block** (lines 523-552 in wrapper)
2. **Modify distance_to_target_reward** (line 406):
   ```python
   # Old:
   distance_reward = scale * 100 * (2 * (past - curr) - 0.01 * distance)

   # New:
   distance_reward = scale * 100 * 2 * (past - curr)
   ```
3. **Comment out push_reward** (lines 438-462)
4. **Keep goal_push_bonus** at scale 0.01 (already set in v7b)
5. **Keep asymmetric OCB** (+0.01/-0.004)

### Training Command

```bash
./run_training.sh --algo happo --env mapush --exp_name critic13 --use_concat_agent_observations_critic True --seed 7
```

**Note:** No `--cooperation_rewards` flag needed (feature disabled)

---

## Risk Assessment

| Risk | Likelihood | Mitigation |
|------|-----------|-----------|
| Freeloading without cooperation bonuses | Medium | `approach_to_box_reward` individual penalty |
| Exploration too sparse | Low | `goal_push_bonus` provides dense signal |
| OCB stays negative | Medium | Monitor; may need to restore `dual_engagement` |
| Agents don't push together | Low | `goal_push_bonus` rewards any box motion toward goal |

---

## Success Metrics

Compare to CRITIC12 v5 baseline:

| Metric | CRITIC12 v5 | CRITIC13 v1 Target |
|--------|-------------|-------------------|
| Success Rate | 79.8% | >75% |
| OCB Reward | +0.00916 | >0 (positive) |
| Avg Step Reward | +0.0226 | >0 (positive) |
| Freeloading Rate | Low | Low (via approach penalty) |
| Pushing Direction | Good | Better (stronger directional signal) |

---

## Version History

| Run Directory | Version | Steps | SR | Key Config |
|---------------|---------|-------|-----|------------|
| `seed-00007-2026-01-05-19-32-27` | v1 | - | FAIL | Minimal essentials: 7 rewards, no cooperation bonuses, no distance penalty - **TOO SPARSE** |
| `seed-00007-2026-01-06-02-13-24` | v2 | 180-190M | ~20%? | v1 + push_reward (curriculum) - distance_to_target (redundant). Some collaborative pushing seen |
| `seed-00007-2026-01-06-14-46-18` | **v3** | - | - | **v2 + proximity_penalty (quadratic, 0.002) - encourage agents to stay close** |
| TBD | v4 | - | - | v3 + increased goal_push_bonus (0.02) - emphasize speed toward goal |

### Version Details

- **v1** (`seed-00007-2026-01-05-19-32-27`): Minimal essentials subset. Removed 4 redundant rewards (push, 3 cooperation bonuses). Removed distance penalty term. Kept asymmetric OCB +0.01/-0.004 and directional push 0.01. **FAILED - too sparse, agents didn't learn to push**
- **v2** (`seed-00007-2026-01-06-02-13-24`): Fixed v1. Re-added `push_reward` (0.0015) for curriculum learning (initial: just push, then directional takes over). Removed `distance_to_target_reward` (redundant with goal_push_bonus - both measure box moving toward goal). **Some collaborative pushing at 180-190M, but only ~1/5 episodes**
- **v3** (`seed-00007-2026-01-06-14-46-18`): Added `proximity_penalty` (0.002, quadratic) to encourage agents to stay within 1.2m (box side length). Penalty = `-scale * max(0, dist - 1.2)²`. Zero when close, grows quadratically when apart.
- **v4** (PLANNED): Increase `goal_push_bonus` from 0.01 → 0.02 to emphasize speed toward goal. See "v4 Proposal" section below.

---

## v1 → v2 Changes

### What Changed

| Change | Rationale |
|--------|-----------|
| ➕ Re-add `push_reward` (0.0015) | **Curriculum learning**: weak "just push" signal, then goal_push_bonus (0.01, 6.7x stronger) takes over for direction |
| ➖ Remove `distance_to_target_reward` | **Redundant**: Both measure "box moving toward goal" - distance uses position delta, goal_push uses velocity. Keep the physics-based one. |

### Active Rewards Comparison

| Reward | v1 | v2 |
|--------|----|----|
| reach_target_reward | ✅ | ✅ |
| distance_to_target_reward | ✅ | ❌ REMOVED |
| approach_to_box_reward | ✅ | ✅ |
| push_reward | ❌ | ✅ RE-ADDED |
| goal_push_bonus | ✅ | ✅ |
| ocb_reward | ✅ | ✅ |
| collision_punishment | ✅ | ✅ |
| exception_punishment | ✅ | ✅ |

**Count:** Both have 7 rewards, better composition in v2

### Why v1 Failed

Without `push_reward`, agents had no initial "just move the box" signal. The directional signal (`goal_push_bonus`) wasn't enough to bootstrap pushing behavior.

### v2 Curriculum

1. **Early training:** `push_reward` (0.0015) teaches "push box = good"
2. **Mid training:** `goal_push_bonus` (0.01) starts dominating "push toward goal = much better"
3. **Late training:** Direction fully learned, push_reward becomes negligible

---

## v2 → v3 Changes

### What Changed

| Change | Rationale |
|--------|-----------|
| ➕ Add `proximity_penalty` (0.002) | **Encourage cooperation**: quadratic penalty keeps agents close to box side length (1.2m) |

### Active Rewards Comparison

| Reward | v2 | v3 |
|--------|----|----|
| reach_target_reward | ✅ | ✅ |
| approach_to_box_reward | ✅ | ✅ |
| push_reward | ✅ | ✅ |
| goal_push_bonus | ✅ | ✅ |
| ocb_reward | ✅ | ✅ |
| proximity_penalty | ❌ | ✅ NEW |
| collision_punishment | ✅ | ✅ |
| exception_punishment | ✅ | ✅ |

**Count:** v2 = 7 rewards, v3 = 8 rewards

### Proximity Penalty Formula

```python
optimal_distance = 1.2  # box side length
agent_distance = norm(agent0_pos - agent1_pos)
excess_distance = max(0, agent_distance - optimal_distance)
penalty = -scale * excess_distance²  # quadratic, 0 when close
```

### Why v2 Only Worked 1/5 Episodes

Agents learned to push but not to stay together. When one agent drifted away, no signal brought them back. Proximity penalty provides continuous "stay close" shaping.

---

## If This Fails

**Fallback plan:**
1. Restore `dual_engagement_bonus` (0.003) - the missing piece from CRITIC12 v5
2. Keep other simplifications
3. This would be "CRITIC13 v5: Minimal + Engagement"

---

## v4 Proposal: Emphasize Speed Toward Goal

### Motivation

User observed that agents push but not fast enough. Want to reward faster box movement toward goal more strongly.

### Analysis of Current Velocity Rewards

| Reward | Scale | What it measures | Per-step value (at 0.3 m/s) |
|--------|-------|------------------|----------------------------|
| `push_reward` | 0.0015 | Box velocity magnitude | ~0.00045 |
| `goal_push_bonus` | 0.01 | Box velocity toward goal | ~0.003 |

Compared to other per-step rewards:
- `ocb_reward`: +0.01 or -0.004 (fixed)
- `proximity_penalty`: ~0 to -0.002 (quadratic)

### Proposed Change

Increase `goal_push_bonus` scale: **0.01 → 0.02**

| Scale | Per-step at 0.3 m/s | Effect |
|-------|---------------------|--------|
| 0.01 (current) | ~0.003 | Baseline |
| **0.02 (proposed)** | **~0.006** | **2x emphasis on speed** |
| 0.03 (aggressive) | ~0.009 | Matches OCB magnitude |

### Why Not a New Reward?

`goal_push_bonus` already measures exactly what we want (velocity toward goal). Adding another speed reward would be redundant.

### Alternative: Time Penalty

Could add per-step cost to make faster completion directly valuable:
```python
time_penalty = -0.001  # every step costs something
```
But this adds complexity. Try scale increase first.

### v4 Implementation

1. Change `goal_push_bonus_scale` in reward calculation or add to config
2. Increase `proximity_penalty_scale` in task/cuboid/config.py
3. Current goal_push uses hardcoded 0.01 in wrapper

### Active Rewards for v4 (if implemented)

| Reward | Scale | Change |
|--------|-------|--------|
| reach_target_reward | 10 | - |
| approach_to_box_reward | 0.00075 | - |
| push_reward | 0.0015 | - |
| **goal_push_bonus** | **0.02** | **↑ from 0.01 (2x)** |
| ocb_reward | +0.01/-0.004 | - |
| **proximity_penalty** | **-0.005 or -0.01** | **↑ from -0.002 (2.5-5x)** |
| collision_punishment | -0.0025 | - |
| exception_punishment | -5 | - |

---

## v4 Proximity Scale Analysis

### Problem in v3

Agents stabilized at ~1.9m apart (0.7m beyond optimal 1.2m).
Proximity penalty at this distance: -0.002 * 0.7² = **-0.001** (too weak)

### Goal

Push agents closer to optimal 1.2m by increasing penalty strength.

### Scale Options

| Scale | Penalty at 1.9m (0.7m excess) | Penalty at 1.5m (0.3m excess) | Effect |
|-------|-------------------------------|-------------------------------|--------|
| 0.002 (current) | -0.001 | -0.0002 | Too weak |
| **0.005** | **-0.0025** | -0.00045 | Moderate (2.5x) |
| **0.01** | **-0.005** | -0.0009 | Strong (5x), matches OCB |

### Recommendation

Try **0.005** first (conservative). If agents still stay at 1.9m, increase to **0.01**.

At 0.01 scale, penalty at 1.9m would be -0.005/step, comparable to OCB reward magnitude.
