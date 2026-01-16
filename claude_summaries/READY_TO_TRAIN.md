# 🚀 READY TO TRAIN - Quick Start Guide

**Date:** 2026-01-16
**Status:** ✅ All tests passing, ready for production training

---

## ✅ What's Been Completed

- ✅ Heterogeneous agent infrastructure (Go1 + Jackal)
- ✅ Buffer initialization for mixed DOF counts
- ✅ Observation computation for heterogeneous agents
- ✅ Torque computation for mixed control types
- ✅ Environment creation, reset, and step verified
- ✅ All integration tests passing

**Test Results:**
```
✅ Environment created: 2 envs × 2 agents (Go1 + Jackal)
✅ Reset successful: Observations [2, 2, 8]
✅ Step successful: Rewards [2, 2], Dones [2]
✅ Total DOFs: 14 (12 Go1 + 2 Jackal)
```

---

## 🎯 Recommended Training Command

Copy-paste this to start training:

```bash
cd /home/gvlab/new-universal-MAPush

conda run -n mapush python HARL/harl_mapush/train.py \
  --exp_name go1_jackal_hetero_v1 \
  --hetero_agent jackal \
  --use_concat_agent_observations_critic True \
  --mapush_og_rewards_teamified True \
  --n_rollout_threads 500 \
  --num_env_steps 100000000 \
  --seed 1
```

**Expected runtime:** ~24-48 hours for 100M steps

---

## 🧪 Quick Test (10K steps - 5 minutes)

Before starting the full run, verify everything works:

```bash
cd /home/gvlab/new-universal-MAPush

conda run -n mapush python HARL/harl_mapush/train.py \
  --exp_name go1_jackal_quick_test \
  --hetero_agent jackal \
  --use_concat_agent_observations_critic True \
  --mapush_og_rewards_teamified True \
  --n_rollout_threads 50 \
  --num_env_steps 10000 \
  --seed 1
```

---

## 📊 Monitor Training

### TensorBoard
```bash
# In a separate terminal
tensorboard --logdir HARL/results/mapush/cuboid/happo/go1_jackal_hetero_v1
```

Then open: http://localhost:6006

### Key Metrics to Watch
- `train/episode_rewards` - Should increase over time
- `train/critic_loss` - Should decrease and stabilize
- `train/actor_loss_agent0` - Go1 policy loss
- `train/actor_loss_agent1` - Jackal policy loss

---

## 🔍 Verify Environment (Optional)

If you want to double-check everything is working:

```bash
cd /home/gvlab/new-universal-MAPush
conda run -n mapush python test_hetero_env.py
```

Should see:
```
✓ ALL TESTS PASSED!
Heterogeneous environment (Go1 + Jackal) is working correctly!
```

---

## 📁 Output Location

Training outputs will be saved to:
```
HARL/results/mapush/cuboid/happo/go1_jackal_hetero_v1/seed-1-<timestamp>/
├── checkpoints/
│   ├── 10M/
│   │   ├── actor_agent0.pt  (Go1 policy)
│   │   ├── actor_agent1.pt  (Jackal policy)
│   │   └── critic_agent.pt  (Shared critic)
│   ├── 20M/
│   └── ...
└── tensorboard/
```

---

## 🎛️ Alternative Configurations

### Different Critic Architecture
```bash
# Goal-centered critic
--use_goal_centered_critic True

# Box-centered critic
--use_box_centered_critic True

# Relative observations critic
--use_relative_obs_critic True
```

### Different Reward Configurations
```bash
# Add cooperation bonuses
--cooperation_rewards True

# Add collaboration bonus (dual pushing)
--collaboration_rewards True

# Prevent freeloading
--individualized_rewards --shared_gated_rewards
```

### Different Training Scale
```bash
# Shorter training (10M steps)
--num_env_steps 10000000

# Longer training (500M steps)
--num_env_steps 500000000

# Fewer environments (for debugging)
--n_rollout_threads 100
```

---

## 🐛 Troubleshooting

### Training crashes immediately
```bash
# Run environment test
python test_hetero_env.py

# Check CUDA memory
nvidia-smi

# Reduce environments if OOM
--n_rollout_threads 200
```

### NaN in losses
- Check tensorboard for when it starts
- Try reducing learning rate
- Try different critic architecture

### One agent not learning
```bash
# Enable anti-freeloading mechanisms
--individualized_rewards --shared_gated_rewards
```

### Poor cooperation
```bash
# Enable cooperation rewards
--cooperation_rewards True --collaboration_rewards True
```

---

## 📚 Complete Documentation

- **All Training Flags:** `claude_summaries/training_flags_reference.md`
- **Implementation Details:** `claude_summaries/heterogeneous_agent_implementation.md`
- **Jackal Integration:** `claude_summaries/jackal_integration.md`
- **HARL Overview:** `claude_summaries/claude_summary_HARL.md`
- **MAPush Overview:** `claude_summaries/claude_summary_MAPush.md`

---

## 🧬 What Makes This Different

### Agent Specifications
| Property | Go1 | Jackal |
|----------|-----|--------|
| Type | Quadruped | Wheeled differential drive |
| Low-level DOF | 12 (leg joints) | 2 (wheel velocities) |
| High-level Actions | 3 [vx, vy, vyaw] | 3 [vx, vy, vyaw] |
| Control Type | Hierarchical (locomotion policy) | Direct (kinematics) |
| Mobility | Omnidirectional | Non-holonomic |
| Terrain | Rough terrain capable | Flat terrain preferred |

### Key Innovation
**Unified Action Space:** Both agents use the same 3 DOF high-level commands [vx, vy, vyaw], but convert them differently:
- **Go1:** Locomotion neural network → 12 joint torques
- **Jackal:** Differential drive kinematics → 2 wheel velocities

This design:
- ✅ Simplifies training (no per-agent network sizing)
- ✅ Enables policy transfer
- ✅ Natural abstraction for multi-robot coordination

---

## 🎯 Success Criteria

Training is successful when:
1. ✅ Episode rewards increase over time
2. ✅ Both agents actively participate (no freeloading)
3. ✅ Successful box pushing to goal
4. ✅ Stable critic and actor losses

---

## 💾 Checkpoints

Models are saved every 10M steps. To resume training:

```bash
python HARL/harl_mapush/train.py \
  --exp_name go1_jackal_hetero_v1 \
  --hetero_agent jackal \
  --use_concat_agent_observations_critic True \
  --mapush_og_rewards_teamified True \
  --checkpoint ./HARL/results/mapush/cuboid/happo/go1_jackal_hetero_v1/seed-1-<timestamp>/checkpoints/50M \
  --n_rollout_threads 500 \
  --num_env_steps 100000000
```

---

## 🧪 Testing Trained Model

After training completes:

```bash
cd HARL/harl_mapush

# Viewer mode (visualize)
python test.py \
  --checkpoint ./results/mapush/cuboid/happo/go1_jackal_hetero_v1/seed-1-<timestamp>/checkpoints/100M \
  --mode viewer \
  --num_episodes 10 \
  --hetero_agent jackal

# Calculator mode (batch evaluation)
python test.py \
  --checkpoint ./results/mapush/cuboid/happo/go1_jackal_hetero_v1/seed-1-<timestamp>/checkpoints/100M \
  --mode calculator \
  --num_episodes 100 \
  --num_envs 300 \
  --hetero_agent jackal
```

---

## 🚀 FINAL COMMAND TO COPY

**Full training (100M steps):**
```bash
cd /home/gvlab/new-universal-MAPush && conda run -n mapush python HARL/harl_mapush/train.py --exp_name go1_jackal_hetero_v1 --hetero_agent jackal --use_concat_agent_observations_critic True --mapush_og_rewards_teamified True --n_rollout_threads 500 --num_env_steps 100000000 --seed 1
```

**Quick test (10K steps):**
```bash
cd /home/gvlab/new-universal-MAPush && conda run -n mapush python HARL/harl_mapush/train.py --exp_name go1_jackal_quick_test --hetero_agent jackal --use_concat_agent_observations_critic True --mapush_og_rewards_teamified True --n_rollout_threads 50 --num_env_steps 10000 --seed 1
```

---

**Ready to train!** 🎉

Good luck with your heterogeneous multi-agent training!

---

**Last Updated:** 2026-01-16
**Status:** 🟢 Production Ready
