# MAPush HAPPO Training - Episode vs Rollout Length

## Crystal Clear Explanation

### MAPush Environment (Ground Truth)
- **Actual episode length**: 1000 steps
- **Episode duration**: 20 seconds
- **Control frequency**: 50Hz (0.02s period)
- **Calculation**: 20s ÷ 0.02s = 1000 steps

### HARL Training Configuration
- **Rollout length**: 200 steps (parameter name: `episode_length`)
- **Why 200?**: Matches original MAPPO training setup
- **What happens**: Collect 200 steps → compute GAE → update networks → repeat

### Key Insight: Rollout ≠ Episode

**One MAPush episode (1000 steps) = Five HARL rollouts (5 × 200 steps)**

```
MAPush Episode Timeline:
|--- Rollout 1 (200 steps) → UPDATE ---|
|--- Rollout 2 (200 steps) → UPDATE ---|
|--- Rollout 3 (200 steps) → UPDATE ---|
|--- Rollout 4 (200 steps) → UPDATE ---|
|--- Rollout 5 (200 steps) → UPDATE ---|
[Episode ends at step 1000, env resets]
```

### Why This Works

**Standard RL practice**: You don't need full episodes to update
- PPO/HAPPO uses **n-step returns** and **GAE** (Generalized Advantage Estimation)
- These only require a fixed-length trajectory (rollout)
- If episode ends mid-rollout, it's handled via `done` flags and bootstrap values
- If rollout ends mid-episode, continue collecting in next rollout (no reset)

### Training Flow Example (500 envs, 100M total steps)

1. **Rollouts**: 100M ÷ (500 × 200) = 1,000 rollouts
2. **Actual MAPush episodes**: ~200 episodes (1000 rollouts ÷ 5)
3. **Updates**: 1,000 updates (one per rollout)
4. **Checkpoints**: 10 checkpoints (10M, 20M, ..., 100M steps)

### Terminology in Code

⚠️ **Confusing HARL parameter naming**:
- Parameter: `episode_length` (historical name from HARL)
- **Actually means**: rollout length / buffer size
- **Not**: actual environment episode length

We've updated prints to say "rollout" for clarity, but the parameter name stays `episode_length` to match HARL's interface.

### Comparison with Original MAPPO

| Aspect | Original MAPPO | HARL HAPPO |
|--------|---------------|------------|
| Rollout length | 200 steps | 200 steps ✓ |
| Parallel envs | 500 | 500 ✓ |
| Total steps | 100M | 100M ✓ |
| Update frequency | Every 200 steps | Every 200 steps ✓ |
| MAPush episodes | ~200 | ~200 ✓ |

**Configuration matches original training setup perfectly.**

## Command Example

```bash
# This trains with 200-step rollouts (matches original MAPPO)
python HARL/harl_mapush/train.py \
    --exp_name my_experiment \
    --n_rollout_threads 500 \
    --num_env_steps 100000000 \
    --episode_length 200  # ← This is ROLLOUT length, not episode length
```

## Summary

✅ **Rollout length**: 200 steps (what we configure)
✅ **Episode length**: 1000 steps (what MAPush environment does)
✅ **Ratio**: 5 rollouts per episode
✅ **Matches**: Original MAPPO training configuration
