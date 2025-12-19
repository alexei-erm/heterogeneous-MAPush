# CRITIC6: Action Scaling Fix (Match OpenRL)

> **Date:** December 19, 2025
> **Goal:** Match OpenRL's action scaling to fix 2x velocity difference
> **Impact:** Agents now move at same speed as OpenRL baseline

---

## Problem Discovered

While comparing OpenRL (backup_MAPush) and HARL implementations, we found a **critical difference in action scaling**:

| Framework | Wrapper Scale | MQE Scale | **Total** | Effective Range |
|-----------|---------------|-----------|-----------|-----------------|
| OpenRL | 0.5x | 0.5x | **0.25x** | [-0.25, 0.25] |
| HARL | None | 0.5x | **0.5x** | [-0.5, 0.5] |

**HARL agents were moving 2x faster than OpenRL agents!**

This could explain training instability - faster movements mean:
- Larger state changes per step
- Harder for critic to predict value
- More variance in returns
- Potentially overshooting targets

---

## Solution

Added OpenRL-compatible action scaling in `mapush_env.py`:

```python
# CRITIC6 (Dec 19, 2025): Match OpenRL action scaling
# OpenRL applies 0.5x scale + clip in wrapper BEFORE MQE's 0.5x scale
# This makes effective range [-0.25, 0.25] instead of [-0.5, 0.5]
# Without this, HARL agents move 2x faster than OpenRL agents
actions_torch = (0.5 * actions_torch).clamp(-1.0, 1.0)
```

---

## Action Flow Comparison

### Before (HARL only)

```
Network output → MQE wrapper (0.5x) → Simulation
                 ↓
         Range: [-0.5, 0.5] m/s
```

### After (Matching OpenRL)

```
Network output → HARL wrapper (0.5x + clip) → MQE wrapper (0.5x) → Simulation
                 ↓
         Range: [-0.25, 0.25] m/s
```

---

## File Changes

**Modified:** `HARL/harl/envs/mapush/mapush_env.py`

```python
# In step() method, after converting actions to torch:
actions_torch = (0.5 * actions_torch).clamp(-1.0, 1.0)
```

---

## Expected Impact

1. **Slower agent movements** - More controlled, less aggressive
2. **Smoother trajectories** - Smaller velocity changes
3. **Better credit assignment** - Smaller state transitions easier to learn
4. **Matches OpenRL baseline** - Fair comparison possible

---

## Additional Difference (Not Changed)

The DiagGaussian std parameterization is also different:

| Framework | Std Formula | Behavior |
|-----------|-------------|----------|
| OpenRL | `exp(log_std)` | Unbounded, can be very large |
| HARL | `sigmoid(log_std) * 0.5` | Bounded in (0, 0.5) |

This was **NOT changed** because:
- Bounded std provides more stable exploration
- The clipping after sampling limits extreme actions anyway
- Action scaling fix is the more critical change

If needed, this can be addressed in a future update by modifying `HARL/harl/models/base/distributions.py`.

---

## Combined with critic5

critic6 should be run with critic5 settings:

| Parameter | Value |
|-----------|-------|
| `lr` | 0.0005 |
| `critic_lr` | 0.003 |
| `critic_epoch` | 30 |
| `actor_update_interval` | 5 |
| `clip_param` | 0.1 |
| `value_loss_coef` | 3.0 |
| `max_grad_norm` | 5.0 |
| `gae_lambda` | 0.9 |
| `actor_hidden_sizes` | [256, 256] |
| `critic_hidden_sizes` | [256, 256, 128] |
| **+ Action scaling** | 0.5x (new) |

---

## History

| Version | Date | Key Changes |
|---------|------|-------------|
| critic1 | Dec 18 | Critic epoch increase |
| critic2 | Dec 18 | Value loss coefficient increase |
| critic3 | Dec 18 | Value normalizer fix |
| critic4 | Dec 19 | Actor update interval |
| critic5 | Dec 19 | Full stability config + separate architectures |
| **critic6** | Dec 19 | **Action scaling fix (match OpenRL)** |
