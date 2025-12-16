# Iteration 10: Reduced Collision Punishment

**Date:** 2025-12-15
**Status:** Ready to train
**Based on:** Iter9 showed agents avoiding each other instead of pushing together

---

## Iter10 Configuration Summary

**Iter10 = Original MAPPO rewards + HAPPO with FP mode + reduced collision scale**

| Component | Setting |
|-----------|---------|
| Algorithm | HAPPO |
| Critic mode | FP (Feature Pruned) - per-agent advantages |
| Rewards | Original MAPPO reward structure |
| `push_reward` | Contact-weighted (only agents near box get credit) |
| `reach_target` | Contact-weighted (only agents near box get success bonus) |
| `collision_punishment_scale` | **-0.0005** (was -0.0025, 5x reduction) |
| All other reward scales | Unchanged from original |

**Flag:** `--individualized_rewards`

---

## Problem with Iter9

Iter9 reached 18% success at 100M steps, but:
- One agent hovers near box without pushing
- Both agents sometimes not pushing
- Collision punishment may be preventing close collaboration

**Analysis:**
```
collision_punishment = (1 / (0.02 + distance/3)) * scale

At 0.5m distance with scale=-0.0025:
  Per step: -0.0134
  Per episode (1000 steps): -13.4

This is roughly equal to the entire episode reward!
```

Agents learned: "staying close to each other = bad"

---

## Iter10 Fix

**Changed in `task/cuboid/config.py`:**
```python
collision_punishment_scale = -0.0005  # was -0.0025 (5x reduction)
```

**New penalties:**
| Distance | Old penalty/step | New penalty/step |
|----------|------------------|------------------|
| 0.3m     | -0.0208          | -0.0042          |
| 0.5m     | -0.0134          | -0.0027          |
| 1.0m     | -0.0071          | -0.0014          |

**Per-episode at 0.5m distance:** -2.7 (was -13.4)

---

## Why Smooth Reduction (not threshold)

Hard thresholds create discontinuities that hurt critic learning:
- Reward jumps suddenly at threshold
- High variance in value estimates
- Noisy gradients

Scaling keeps the smooth curve, just weaker.

---

## Difference from Original MAPPO

| Aspect | Original MAPPO | Iter10 HAPPO |
|--------|----------------|--------------|
| Policy | Shared (1 actor) | Separate (2 actors) |
| Critic | EP mode (shared reward) | FP mode (per-agent reward) |
| `push_reward` | Shared | Contact-weighted individual |
| `reach_target` | Shared | Contact-weighted individual |
| `collision_scale` | -0.0025 | **-0.0005** |

---

## Usage

```bash
cd /home/gvlab/new-universal-MAPush
python HARL/harl_mapush/train.py --algo happo --exp_name happo_iter10_lowcollision --individualized_rewards
```

---

## Expected Behavior

- Agents should be more willing to get close to each other
- Both agents actively pushing (not just hovering)
- Success rate should improve beyond iter9's 18%

---

## Monitoring

Watch `rewards/collision_punishment`:
- Iter9: ~-0.7 per step average
- Iter10: should be ~-0.14 (5x lower)

If agents still avoid each other, may need further reduction.
