# HAPPO MAPush Testing Guide

This guide explains how to test trained HAPPO models on the MAPush cuboid pushing task.

---

## Overview

The `test.py` script provides two testing modes:

1. **Calculator Mode** - Compute statistics over many episodes using parallel environments
2. **Viewer Mode** - Visualize episodes sequentially with rendering

---

## Quick Start

### Calculator Mode (Statistics)

Compute success rate, collision rate, and other metrics:

```bash
./run_testing.sh \
    --checkpoint HARL/results/mapush/cuboid/happo/my_exp/seed-00001-.../checkpoints/10M \
    --mode calculator \
    --num_episodes 100 \
    --num_envs 300 \
    --seed 1
```

**Output:**
```
======================================================================
Statistics Summary (over 100 episodes)
======================================================================
  Success Rate:         0.8567 (85.67%)
  Collision Rate:       0.1234 (12.34%)
  Avg Episode Length:   156.23 steps
  Collaboration Degree: 0.7891
======================================================================
```

### Viewer Mode (Visualization)

Watch the robots push the box (renders in Isaac Gym viewer):

```bash
./run_testing.sh \
    --checkpoint HARL/results/mapush/cuboid/happo/my_exp/seed-00001-.../checkpoints/10M \
    --mode viewer \
    --num_episodes 5 \
    --seed 1
```

**Note:** Viewer mode requires a display and may not work on headless servers or with A100/A800 GPUs.

---

## Command Line Arguments

### Required Arguments

- `--checkpoint` - Path to checkpoint directory containing `actor_agent0.pt` and `actor_agent1.pt`
  - Example: `HARL/results/mapush/cuboid/happo/exp1/seed-00001-20251213_123456/checkpoints/10M`

### Mode Selection

- `--mode` - Testing mode (default: `calculator`)
  - `calculator` - Multi-environment statistics computation
  - `viewer` - Sequential visualization with rendering

### Episode Configuration

- `--num_episodes` - Number of episodes (default: 100)
  - **Calculator mode:** Target number of episodes (may run slightly more due to parallel envs)
  - **Viewer mode:** Exact number of episodes to visualize

- `--num_envs` - Number of parallel environments (default: 300)
  - **Calculator mode only:** More envs = faster statistics computation
  - Ignored in viewer mode (always uses 1 env)

### Other Options

- `--seed` - Random seed (default: 1)

---

## Examples

### Test Latest Checkpoint

```bash
# Find latest checkpoint
LATEST=$(ls -td HARL/results/mapush/cuboid/happo/*/seed-*/checkpoints/* | head -1)

# Run calculator mode
./run_testing.sh --checkpoint $LATEST --mode calculator --num_episodes 200
```

### Test All Checkpoints

```bash
# Test 10M, 20M, 30M checkpoints
for checkpoint in HARL/results/mapush/cuboid/happo/exp1/seed-00001-*/checkpoints/{10M,20M,30M}; do
    echo "Testing: $checkpoint"
    ./run_testing.sh --checkpoint $checkpoint --mode calculator --num_episodes 100
done
```

### Quick Visual Check

```bash
# Watch a few episodes with the trained model
./run_testing.sh \
    --checkpoint HARL/results/.../checkpoints/50M \
    --mode viewer \
    --num_episodes 3
```

### Comprehensive Evaluation

```bash
# Run 500 episodes for robust statistics
./run_testing.sh \
    --checkpoint HARL/results/.../checkpoints/100M \
    --mode calculator \
    --num_episodes 500 \
    --num_envs 500
```

---

## Output Statistics Explained

### Success Rate
- Percentage of episodes where the box reached the target position
- **Higher is better** (target: > 80%)

### Collision Rate
- Average fraction of steps where agents collided with each other
- Computed only for successful episodes
- **Lower is better** (target: < 15%)

### Avg Episode Length
- Average number of steps to complete successful episodes
- MAPush episodes can be up to 1000 steps (20 seconds @ 50 Hz)
- **Lower is better** (indicates efficient pushing)

### Collaboration Degree
- Measure of how well agents coordinate their pushing efforts
- Computed based on optimal contact points and synchronized pushing
- **Higher is better** (target: > 0.75)

---

## Checkpoint Directory Structure

Expected structure for checkpoints:

```
HARL/results/mapush/cuboid/happo/<exp_name>/seed-<seed>-<timestamp>/
├── checkpoints/
│   ├── 10M/
│   │   ├── actor_agent0.pt      ← Required
│   │   ├── actor_agent1.pt      ← Required
│   │   ├── critic_agent.pt      (not used in testing)
│   │   └── value_normalizer.pt  (not used in testing)
│   ├── 20M/
│   ├── 30M/
│   └── ...
├── logs/  (TensorBoard logs)
└── config.json
```

**Note:** Only the `actor_agent*.pt` files are needed for testing.

---

## Troubleshooting

### Error: "Checkpoint directory not found"

**Problem:** Invalid checkpoint path

**Solution:** Verify the path exists and contains `actor_agent0.pt` and `actor_agent1.pt`

```bash
ls HARL/results/mapush/cuboid/happo/*/seed-*/checkpoints/*/actor_agent*.pt
```

### Error: "Segmentation fault" (Viewer Mode)

**Problem:** A100/A800 GPUs don't support Isaac Gym rendering

**Solution:** Use calculator mode instead, or test on a machine with GeForce GPU

### Error: "ImportError" or "ModuleNotFoundError"

**Problem:** Python path pollution from old projects

**Solution:** Use the `run_testing.sh` wrapper which automatically cleans PYTHONPATH

### Calculator Mode Too Slow

**Problem:** Taking too long to compute statistics

**Solution:** Increase `--num_envs` for faster parallel execution

```bash
./run_testing.sh --checkpoint ... --mode calculator --num_envs 500
```

### Viewer Mode Shows Nothing

**Problem:** Running on headless server or SSH without X11 forwarding

**Solution:**
- Run calculator mode instead
- Or use X11 forwarding: `ssh -X user@host`
- Or run from a machine with display

---

## Comparing with MAPPO Baseline

To compare HAPPO with the original MAPPO implementation:

### 1. Test MAPPO Model

```bash
# Original MAPush testing (OpenRL)
cd /home/gvlab/new-universal-MAPush
source results/<mm-dd-hh>_cuboid/task/train.sh True
```

### 2. Test HAPPO Model

```bash
./run_testing.sh \
    --checkpoint HARL/results/.../checkpoints/50M \
    --mode calculator \
    --num_episodes 100 \
    --seed 1
```

### 3. Compare Results

Record metrics from both and compare:
- Success rate (should be similar or better)
- Collision rate (should be lower with HAPPO)
- Collaboration degree (should be higher with HAPPO)

---

## Advanced Usage

### Custom Task Configuration

To test with different environment settings, modify the task config before testing:

```python
# In test.py, line ~240, modify Go1PushMidCfg before creating environment
from task.cuboid.config import Go1PushMidCfg

# Example: Test with different number of obstacles
Go1PushMidCfg.env.num_npcs = 4  # More obstacles
```

### Different Random Seeds

Test model robustness across different random seeds:

```bash
for seed in {1..10}; do
    ./run_testing.sh \
        --checkpoint HARL/results/.../checkpoints/50M \
        --mode calculator \
        --num_episodes 50 \
        --seed $seed
done
```

### Save Statistics to File

```bash
./run_testing.sh \
    --checkpoint HARL/results/.../checkpoints/50M \
    --mode calculator \
    --num_episodes 100 \
    | tee results/happo_50M_stats.txt
```

---

## Next Steps

After testing:

1. **If performance is good** → Train to completion (100M steps)
2. **If performance is poor** → Tune hyperparameters and retrain
3. **For heterogeneous agents** → Modify environment wrapper to support different observation spaces
4. **For real-world deployment** → Export models and test in MuJoCo

---

## Tips for Better Results

### Calculator Mode
- Use **many episodes** (≥100) for reliable statistics
- Use **many envs** (≥300) for faster computation
- Use **multiple seeds** to test robustness

### Viewer Mode
- Use **few episodes** (3-5) for quick visual inspection
- Check if agents are:
  - Approaching the box efficiently
  - Pushing from optimal contact points
  - Coordinating their movements
  - Avoiding collisions

### Model Selection
- Test **multiple checkpoints** (10M, 20M, ..., 100M)
- Best model may not be the final one
- Look for trade-off between success rate and collision rate

---

## Support

If you encounter issues:

1. Check this guide's troubleshooting section
2. Review the error messages carefully
3. Verify your checkpoint path is correct
4. Ensure you're running from the project root directory

For Python path issues, always use the `run_testing.sh` wrapper script.
