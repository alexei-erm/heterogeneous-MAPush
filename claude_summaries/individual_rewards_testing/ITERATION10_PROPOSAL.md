# Iteration 10: Velocity-Based Rewards

**Date**: 2025-12-07
**Problem**: Agents push together from same side (Iter9 success!) but not toward goal

---

## What Iter9 Achieved

✅ Both agents on push side (blocking fixed)
✅ Both agents pushing together
❌ Pushing in random direction, not toward goal
❌ 0% success rate still

---

## Root Cause Analysis

The `push_contribution` reward was broken:

```python
# OLD CODE (Iter9 and before):
force_direction = agent_to_box  # Where agent IS relative to box
alignment = dot(force_direction, target_direction)
contribution = alignment * box_speed * scale
```

**The flaw**: This rewards agent POSITION, not actual push DIRECTION.

An agent positioned "behind" the box (good position) could push sideways or weakly, and still get rewarded because:
- `agent_to_box` points toward target ✓
- `box_speed > 0` ✓
- But box velocity could be perpendicular to target!

---

## Iter10 Fixes

### Fix 1: Restore Shared Directional Progress Reward

```python
directional_progress_scale = 0.15  # Was 0.0 since Iter6
```

**How it works**:
- Measures actual distance change: `old_distance - new_distance`
- Box moves toward goal → BOTH agents get +0.15 * progress
- Box moves away → BOTH agents get -0.15 * progress
- Simple, clear, shared signal that aligns both agents

### Fix 2: Push Contribution Uses Actual Box Velocity

```python
# NEW CODE (Iter10):
box_velocity_direction = box_velocity / (norm(box_velocity) + 1e-6)
velocity_alignment = dot(box_velocity_direction, target_direction)
contribution = velocity_alignment * box_speed * scale
```

**The fix**: Reward based on where box is ACTUALLY moving, not where agent is positioned.

---

## Reward Structure Iter10

| Component | Scale | Type | Signal |
|-----------|-------|------|--------|
| push_contribution | 0.15 | per-agent | Box velocity toward goal |
| directional_progress | 0.15 | **SHARED** | Box distance to goal decreased |
| engagement_bonus | 0.02 | per-agent | Near box |
| cooperation_bonus | 0.01 | per-agent | Both near box |
| same_side_bonus | 0.02 | per-agent | Both on push side |
| blocking_penalty | -0.05 | per-agent | Between box and goal |

**Key insight**: Now we have BOTH:
- Per-agent rewards (push_contribution) for individual accountability
- Shared reward (directional_progress) for team goal alignment

---

## Expected Behavior

**Early (0-10M)**:
- Agents explore, learn that box→goal = reward
- directional_progress provides clear shared signal

**Mid (10-30M)**:
- Agents coordinate to push toward goal
- Push contribution reinforces correct velocity

**Late (30M+)**:
- Consistent goal-directed pushing
- Success rate should start climbing

---

## Success Criteria

| Milestone | Metric | Target |
|-----------|--------|--------|
| 10M | Distance to target | < 2.0m (improving) |
| 20M | Reward | > +15 (above Iter9's +13) |
| 30M | Success rate | > 5% |
| 50M | Success rate | > 15% |

---

## Training Command

```bash
cd /home/gvlab/universal-MAPush/HARL
python examples/train.py --algo happo --env mapush \
    --exp_name iteration10_velocity_rewards \
    --n_rollout_threads 500 \
    --num_env_steps 100000000 \
    --lr 0.005 \
    --episode_length 200
```

---

## Validation Print

Look for at startup:
```
================================================================================
[ITERATION 10 - VELOCITY-BASED REWARDS]
================================================================================
use_per_agent_rewards:    True
push_reward_scale:        0.15
reach_target_scale:       2.0
engagement_bonus_scale:   0.02
cooperation_bonus_scale:  0.01
blocking_penalty_scale:   0.05
same_side_bonus_scale:    0.02
directional_progress:     0.15  ← RESTORED: Shared reward for box→goal

ITERATION 10: Velocity-based rewards
  Iter9 problem: Agents push together but not toward goal
  Iter10 fix #1: directional_progress=0.15 (SHARED reward when box→goal)
  Iter10 fix #2: push_contribution uses ACTUAL box velocity, not agent position
  Expected: Agents learn to push box TOWARD goal
================================================================================
```

---

## Summary

| Iteration | Problem | Fix | Result |
|-----------|---------|-----|--------|
| Iter8 | One agent blocking | - | Failed |
| Iter9 | Add blocking penalty | Both push same side | Partial success |
| Iter10 | Push random direction | Velocity-based rewards | ? |

**Confidence**: 80% - This directly addresses the observed behavior with proper reward signal.
