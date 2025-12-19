# CRITIC5: Stability-Focused Configuration

> **Date:** December 19, 2025
> **Goal:** Maximum critic stability with smooth value loss convergence
> **Trade-off:** Slower learning in exchange for stability

---

## Philosophy

```
STABILITY = Slow updates + Tight constraints + Stationary targets
```

The critic5 configuration prioritizes training stability over learning speed. This is motivated by observing value loss rising for 20-40M steps in previous configurations, indicating critic instability.

---

## Configuration Changes

### Summary Table

| Parameter | critic3/4 | critic5 | Change | Rationale |
|-----------|-----------|---------|--------|-----------|
| `lr` (actor) | 0.001 | **0.0005** | -50% | Slower policy change |
| `critic_lr` | 0.005 | **0.003** | -40% | More stable critic updates |
| `critic_epoch` | 25 | **30** | +20% | Better fitting per rollout |
| `actor_update_interval` | 3 | **5** | +67% | Even slower moving target |
| `clip_param` | 0.2 | **0.1** | -50% | Tighter value prediction bounds |
| `value_loss_coef` | 5.0 | **3.0** | -40% | Balance with tighter clip |
| `max_grad_norm` | 10.0 | **5.0** | -50% | Prevent large gradient updates |
| `gae_lambda` | 0.95 | **0.9** | -5% | Lower variance advantages |
| `actor_hidden_sizes` | [256,256,128] | **[256,256]** | -1 layer | Simpler actor network |
| `critic_hidden_sizes` | [256,256,128] | **[256,256,128]** | same | Keep critic capacity |

---

## Detailed Rationale

### 0. Separate Actor/Critic Architectures (NEW FEATURE)

**Problem:** Actor and critic share the same network architecture, but they have different requirements:
- Actor: Maps local observation (8 dims) to action distribution
- Critic: Maps global state (17 dims) to scalar value

**Solution:** Implement separate `actor_hidden_sizes` and `critic_hidden_sizes` in config.

```yaml
# happo.yaml
model:
  hidden_sizes: [256, 256, 128]        # Fallback (backward compatible)
  actor_hidden_sizes: [256, 256]       # Simpler actor (2 layers)
  critic_hidden_sizes: [256, 256, 128] # Keep critic capacity (3 layers)
```

**Implementation:** Modified `on_policy_base_runner.py` to pass separate hidden_sizes to actor and critic:

```python
# Actor gets actor_hidden_sizes
actor_model_args = {**algo_args["model"], "hidden_sizes": self.actor_hidden_sizes}

# Critic gets critic_hidden_sizes
critic_model_args = {**algo_args["model"], "hidden_sizes": self.critic_hidden_sizes}
```

**Rationale for critic5:**
- Actor with 2 layers: Simpler policy, less overfitting to noise
- Critic with 3 layers: Maintains capacity for value function approximation
- Reduces actor parameter count, making policy updates more stable

### 1. Learning Rates (lr: 0.001→0.0005, critic_lr: 0.005→0.003)

**Problem:** Fast learning rates cause rapid changes in both policy and value function.

**Solution:** Reduce both learning rates:
- Actor LR halved: policy changes more slowly
- Critic LR reduced by 40%: value predictions update more gradually
- Critic still learns faster than actor (6:1 ratio maintained)

### 2. Critic Epochs (25→30)

**Problem:** Critic may not fully fit the value function before next rollout.

**Solution:** Increase critic training epochs:
- More gradient steps per rollout
- Better value function approximation
- Compensates for lower learning rate

### 3. Actor Update Interval (3→5)

**Problem:** Policy changes cause non-stationary targets for critic.

**Solution:** Update actors less frequently:
- Actors only update every 5th iteration
- Critic gets 5 rollouts of stable policy to learn from
- Reduces "chasing a moving target" problem

### 4. Value Clip Parameter (0.2→0.1)

**Problem:** Value predictions can jump too far between updates.

**Solution:** Tighter clipping:
```
Before: V_new can differ from V_old by ±0.2
After:  V_new can differ from V_old by ±0.1
```
- Prevents value function oscillation
- Smoother value predictions over time

### 5. Value Loss Coefficient (5.0→3.0)

**Problem:** With tighter clip_param, high value_loss_coef creates conflicting gradients.

**Solution:** Reduce coefficient to balance:
- Still prioritizes critic over actor (3.0 vs entropy 0.01)
- Works harmoniously with tighter clipping
- Prevents "fighting" between clip constraint and loss magnitude

### 6. Max Gradient Norm (10.0→5.0)

**Problem:** Large gradients (40+) being scaled to 10 may still be too aggressive.

**Solution:** Tighter gradient clipping:
- Gradients now scaled to max norm of 5.0
- If pre-clip norm is 40, scaling factor is 0.125 (was 0.25)
- More conservative parameter updates

### 7. GAE Lambda (0.95→0.9)

**Problem:** High lambda means high variance in advantage estimates.

**Solution:** Lower lambda for bias-variance trade-off:
```
lambda = 1.0: Full Monte Carlo returns (high variance)
lambda = 0.0: 1-step TD (high bias)
lambda = 0.9: More bias toward TD, lower variance
```
- More stable advantage estimates
- Critic sees less noisy targets

---

## Expected Behavior

### Pros
- Smoother value loss curve
- Less oscillation in critic predictions
- More stable policy updates
- Better late-training behavior

### Cons
- Slower initial learning (1.5-2x more steps to converge)
- May plateau if too conservative
- Actor may lag behind optimal policy

---

## Monitoring Checklist

When running critic5, monitor these metrics:

1. **value_loss**: Should be smoother, may start higher but decrease steadily
2. **critic_grad_norm**: Should stay closer to 5.0 (pre-clip values lower overall)
3. **average_episode_rewards**: May improve more slowly but more consistently
4. **policy_loss**: Should be stable without large spikes

---

## Rollback Plan

If critic5 is too conservative (learning stalls), try intermediate settings:

```yaml
# critic5-lite: Less aggressive stability
lr: 0.0008                  # Between 0.0005 and 0.001
critic_lr: 0.004            # Between 0.003 and 0.005
actor_update_interval: 4    # Between 3 and 5
clip_param: 0.15            # Between 0.1 and 0.2
max_grad_norm: 7.0          # Between 5.0 and 10.0
gae_lambda: 0.92            # Between 0.9 and 0.95
```

---

## File Changes

**Modified:**
1. `HARL/harl/configs/algos_cfgs/happo.yaml`
   - Added `actor_hidden_sizes` and `critic_hidden_sizes` parameters
   - All changes documented inline with `# CRITIC5 (Dec 19, 2025):` comments

2. `HARL/harl/runners/on_policy_base_runner.py`
   - Added extraction of `actor_hidden_sizes` and `critic_hidden_sizes` with fallback
   - Modified actor initialization to use `actor_hidden_sizes`
   - Modified critic initialization to use `critic_hidden_sizes`
   - Modified buffer initialization to use appropriate hidden sizes

---

## History

| Version | Date | Key Changes |
|---------|------|-------------|
| critic1 | Dec 18 | Initial critic epoch increase |
| critic2 | Dec 18 | Value loss coefficient increase |
| critic3 | Dec 18 | Value normalizer fix (update once before loop) |
| critic4 | Dec 19 | Actor update interval introduced |
| **critic5** | Dec 19 | Full stability-focused configuration |
