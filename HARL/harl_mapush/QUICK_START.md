# HAPPO MAPush Quick Start Guide

Complete workflow for training and testing HAPPO on MAPush cuboid pushing task.

---

## Prerequisites

1. **Conda environment activated:**
   ```bash
   conda activate mapush
   ```

2. **In project root directory:**
   ```bash
   cd /home/gvlab/new-universal-MAPush
   ```

3. **PYTHONPATH clean** (wrapper scripts handle this automatically)

---

## Training

### Quick Start - Short Test Run (1M steps, ~10 minutes)

```bash
./run_training.sh --exp_name test_1M --num_env_steps 1000000 --n_rollout_threads 500
```

### Full Training (100M steps, ~8-10 hours)

```bash
./run_training.sh --exp_name cuboid_happo_v1 --num_env_steps 100000000 --n_rollout_threads 500
```

### Custom Configuration

```bash
./run_training.sh \
    --exp_name my_experiment \
    --seed 42 \
    --n_rollout_threads 500 \
    --num_env_steps 100000000 \
    --episode_length 200
```

### Training Output

Results saved to:
```
HARL/results/mapush/cuboid/happo/<exp_name>/seed-<seed>-<timestamp>/
├── checkpoints/
│   ├── 10M/  ← Checkpoints every 10M steps
│   ├── 20M/
│   └── ...
└── logs/     ← TensorBoard logs
```

---

## Monitoring Training

### TensorBoard

```bash
tensorboard --logdir HARL/results/mapush/cuboid/happo/<exp_name>/seed-*/logs
```

Open browser to `http://localhost:6006`

**Key Metrics to Watch:**
- `mapush/success_rate` - Should increase over time
- `mapush/collision_rate` - Should decrease over time
- `average_episode_rewards` - Should increase (become less negative)
- `mapush/collaboration_degree` - Should increase

### Check Progress

```bash
# Find latest checkpoint
ls -lt HARL/results/mapush/cuboid/happo/*/seed-*/checkpoints/

# Count steps trained
ls HARL/results/mapush/cuboid/happo/*/seed-*/checkpoints/ | tail -1
```

---

## Testing

### Calculator Mode (Statistics)

Fast statistics computation using parallel environments:

```bash
./run_testing.sh \
    --checkpoint HARL/results/mapush/cuboid/happo/cuboid_happo_v1/seed-00001-*/checkpoints/50M \
    --mode calculator \
    --num_episodes 100 \
    --num_envs 300
```

**Output:**
```
Success Rate:         0.8567 (85.67%)
Collision Rate:       0.1234 (12.34%)
Avg Episode Length:   156.23 steps
Collaboration Degree: 0.7891
```

### Viewer Mode (Visualization)

Watch the robots in action:

```bash
./run_testing.sh \
    --checkpoint HARL/results/.../checkpoints/50M \
    --mode viewer \
    --num_episodes 5
```

**Note:** Requires display (won't work on headless servers or A100/A800 GPUs)

---

## Typical Workflow

### 1. Short Test Run

Verify everything works:

```bash
# Train for 1M steps (~10 minutes)
./run_training.sh --exp_name test_1M --num_env_steps 1000000

# Test the checkpoint
CKPT=$(ls -d HARL/results/mapush/cuboid/happo/test_1M/seed-*/checkpoints/* | tail -1)
./run_testing.sh --checkpoint $CKPT --mode calculator --num_episodes 10
```

### 2. Full Training Run

Once verified:

```bash
# Train for 100M steps (~8-10 hours)
./run_training.sh --exp_name cuboid_happo_final --num_env_steps 100000000 --seed 1
```

### 3. Monitor During Training

In a separate terminal:

```bash
# TensorBoard
tensorboard --logdir HARL/results/mapush/cuboid/happo/cuboid_happo_final/

# Or watch checkpoints appear
watch -n 60 'ls -lh HARL/results/mapush/cuboid/happo/cuboid_happo_final/seed-*/checkpoints/'
```

### 4. Evaluate Checkpoints

After training completes:

```bash
# Test all checkpoints
for ckpt in HARL/results/mapush/cuboid/happo/cuboid_happo_final/seed-*/checkpoints/*; do
    echo "Testing: $(basename $ckpt)"
    ./run_testing.sh --checkpoint $ckpt --mode calculator --num_episodes 100
done
```

### 5. Select Best Model

Based on test results:
- Highest success rate with acceptable collision rate
- Usually around 50M-80M steps

---

## Expected Performance

### Training Progress

| Checkpoint | Success Rate | Collision Rate | Notes |
|------------|--------------|----------------|-------|
| 10M steps  | ~30-50%      | ~20-30%        | Learning basics |
| 20M steps  | ~50-70%      | ~15-25%        | Improving |
| 50M steps  | ~70-85%      | ~10-15%        | Good performance |
| 100M steps | ~80-90%      | ~8-12%         | Mature policy |

**Note:** These are approximate. Actual results depend on hyperparameters and random seed.

### Comparison with MAPPO

HAPPO should achieve:
- **Similar or better** success rate (±5%)
- **Lower** collision rate (sequential updates help coordination)
- **Higher** collaboration degree (better multi-agent coordination)

---

## Troubleshooting

### Training Issues

**Problem:** Training crashes immediately
```bash
# Check GPU memory
nvidia-smi

# Try fewer environments
./run_training.sh --n_rollout_threads 300  # instead of 500
```

**Problem:** PYTHONPATH errors
```bash
# Always use the wrapper script
./run_training.sh ...  # NOT: python HARL/harl_mapush/train.py
```

**Problem:** Training is slow
```bash
# Check if using GPU
nvidia-smi

# Training should use ~6-7 GB VRAM with 500 envs
# If using CPU, check Isaac Gym installation
```

### Testing Issues

**Problem:** Checkpoint not found
```bash
# List available checkpoints
find HARL/results -name "actor_agent0.pt" -type f

# Use full path
./run_testing.sh --checkpoint $(pwd)/HARL/results/.../checkpoints/10M
```

**Problem:** Viewer mode segfaults
```bash
# Use calculator mode instead
./run_testing.sh --checkpoint ... --mode calculator
```

---

## File Structure

```
/home/gvlab/new-universal-MAPush/
├── run_training.sh          ← Training wrapper (use this!)
├── run_testing.sh           ← Testing wrapper (use this!)
├── HARL/
│   └── harl_mapush/
│       ├── train.py         ← Training script
│       ├── test.py          ← Testing script
│       ├── QUICK_START.md   ← This file
│       ├── TESTING_GUIDE.md ← Detailed testing guide
│       └── runners/
│           └── mapush_happo_runner.py
└── results/  (old MAPPO results for comparison)
```

---

## Command Reference

### Training Commands

```bash
# Basic training
./run_training.sh --exp_name NAME

# Custom parameters
./run_training.sh --exp_name NAME --seed 1 --n_rollout_threads 500 --num_env_steps 100000000

# Resume from checkpoint (if implemented)
./run_training.sh --exp_name NAME --checkpoint path/to/checkpoint
```

### Testing Commands

```bash
# Calculator mode (fast statistics)
./run_testing.sh --checkpoint PATH --mode calculator --num_episodes 100 --num_envs 300

# Viewer mode (visualization)
./run_testing.sh --checkpoint PATH --mode viewer --num_episodes 5

# Different seed
./run_testing.sh --checkpoint PATH --mode calculator --seed 42
```

### Monitoring Commands

```bash
# TensorBoard
tensorboard --logdir HARL/results/mapush/cuboid/happo/EXP_NAME/seed-*/logs

# List checkpoints
ls HARL/results/mapush/cuboid/happo/*/seed-*/checkpoints/

# GPU usage
nvidia-smi -l 1  # Update every second
```

---

## Next Steps

1. ✅ **Test infrastructure** - Run 1M step test
2. ⏳ **Full training** - Train to 100M steps
3. ⏳ **Evaluation** - Test all checkpoints
4. ⏳ **Comparison** - Compare with MAPPO baseline
5. ⏳ **Heterogeneous agents** - Different observation/action spaces
6. ⏳ **Real robots** - Deploy in lab

---

## Tips

### Training Tips
- Use **high-quality GPU** (RTX 3090/4090 recommended)
- Monitor **TensorBoard** during training
- Save **multiple seeds** for robustness
- Keep **checkpoints** from different stages

### Testing Tips
- Use **calculator mode** for statistics (faster)
- Use **viewer mode** for debugging (visual)
- Test **multiple checkpoints** to find best
- Use **different seeds** to test robustness

### Debugging Tips
- Check **PYTHONPATH** is clean (use wrapper scripts)
- Verify **GPU memory** is sufficient (~7GB for 500 envs)
- Check **Isaac Gym** installation if training won't start
- Review **TensorBoard** logs for training issues

---

## Help

- **Training details:** See `TRAINING_EXPLANATION.md`
- **Testing details:** See `TESTING_GUIDE.md`
- **Integration details:** See `claude_summaries/HARL_integration_proposal.md`
- **Bug fixes:** See `claude_summaries/training_bugs_fixed.md`

For issues, check the troubleshooting sections above or review the error messages carefully.
