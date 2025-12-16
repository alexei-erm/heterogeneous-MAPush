# Iteration 11: Lower Learning Rates

**Date:** 2025-12-15
**Status:** Ready to train
**Based on:** Iter9/10 critic value_loss climbing instead of converging

---

## Iter11 Configuration Summary

**Iter11 = Iter10 + lower learning rates to match MAPPO**

| Component | Setting |
|-----------|---------|
| Algorithm | HAPPO |
| Critic mode | FP (Feature Pruned) |
| Rewards | Original MAPPO + contact-weighted push/reach_target |
| `collision_punishment_scale` | -0.0005 (from iter10) |
| `lr` (actor) | **0.0005** (was 0.005, 10x reduction) |
| `critic_lr` | **0.0005** (was 0.005, 10x reduction) |

---

## Problem with Iter9/10

Critic value_loss was **increasing** instead of converging:

| Run | Value Loss Trend |
|-----|------------------|
| MAPPO | 0.53 → 0.03 (converges) |
| HAPPO iter9 | 0.11 → 0.10 (flat/oscillating) |
| HAPPO iter10 | 0.10 → 0.19 (diverging!) |

**Root cause discovered:** HAPPO default learning rates are 10x higher than MAPPO:

| Parameter | MAPPO | HAPPO (old) | HAPPO (iter11) |
|-----------|-------|-------------|----------------|
| `lr` | 0.0005 | 0.005 | **0.0005** |
| `critic_lr` | 0.0005 | 0.005 | **0.0005** |

High learning rate causes critic to overshoot and diverge.

---

## Changes Made

**In `HARL/harl/configs/algos_cfgs/happo.yaml`:**
```yaml
lr: 0.0005  # was 0.005
critic_lr: 0.0005  # was 0.005
```

---

## Expected Behavior

- Critic value_loss should **decrease** like MAPPO (0.5 → 0.03)
- More stable training
- Better success rate

---

## Usage

```bash
cd /home/gvlab/new-universal-MAPush
python HARL/harl_mapush/train.py --algo happo --exp_name happo_iter11_lowlr --individualized_rewards --n_rollout_threads 700
```

---

## Monitoring

**Critical metric:** `critic/value_loss`
- Should decrease over time (like MAPPO)
- If still increasing, need further investigation

---

## Summary of All Changes from Original MAPPO

| Aspect | Original MAPPO | Iter11 HAPPO |
|--------|----------------|--------------|
| Policy | Shared (1 actor) | Separate (2 actors) |
| Critic | EP mode | FP mode |
| `push_reward` | Shared | Contact-weighted individual |
| `reach_target` | Shared | Contact-weighted individual |
| `collision_scale` | -0.0025 | -0.0005 |
| `lr` | 0.0005 | 0.0005 (now same) |
| `critic_lr` | 0.0005 | 0.0005 (now same) |
