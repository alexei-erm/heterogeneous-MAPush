# HAPPO Training Flags Reference

**Date:** 2026-01-16
**Purpose:** Comprehensive reference for all HAPPO training flags in MAPush with heterogeneous agents support

---

## 🚀 Quick Start Commands

### Full Training (100M steps)
```bash
cd /home/gvlab/new-universal-MAPush

# Heterogeneous: Go1 + Jackal with Concatenated Critic + OG MAPush Rewards
conda run -n mapush python HARL/harl_mapush/train.py \
  --exp_name go1_jackal_hetero_concat_critic \
  --hetero_agent jackal \
  --use_concat_agent_observations_critic True \
  --mapush_og_rewards_teamified True \
  --n_rollout_threads 500 \
  --num_env_steps 100000000 \
  --seed 1
```

### Quick Test (10K steps)
```bash
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

## 📋 All Available Flags

### Core Experiment Flags

| Flag | Type | Default | Description |
|------|------|---------|-------------|
| `--algo` | str | `"happo"` | Algorithm to use (only HAPPO supported) |
| `--env` | str | `"mapush"` | Environment name |
| `--exp_name` | str | `"cuboid_happo"` | Experiment name (used for logging/checkpoints) |
| `--task` | str | `"go1push_mid"` | MAPush task variant |
| `--seed` | int | `1` | Random seed for reproducibility |

**Example:**
```bash
--exp_name my_experiment --seed 42
```

---

### 🤖 Heterogeneous Agents

| Flag | Type | Default | Description |
|------|------|---------|-------------|
| `--hetero_agent` | str | `None` | Enable heterogeneous agents. Specify second robot type (e.g., `jackal`). Agent0=Go1, Agent1=specified robot. |

**Available Robot Types:**
- `jackal` - Clearpath Jackal (differential drive, 2 DOF wheels)
- `go1` - Unitree Go1 (default, quadruped, 12 DOF)

**Example:**
```bash
--hetero_agent jackal  # Go1 + Jackal heterogeneous team
```

---

### 🧠 Critic Architecture Flags (Choose ONE)

| Flag | Type | Default | ID | Description |
|------|------|---------|-------|-------------|
| `--use_concat_agent_observations_critic` | bool | `False` | **CRITIC10** | Concatenate agent observations for critic input |
| `--use_goal_centered_critic` | bool | `False` | **CRITIC16** | Goal-centered coordinates: Everything relative to goal (stationary reference frame). 9 dims: [box_rel(3), agent0_rel(3), agent1_rel(3)] |
| `--use_box_centered_critic` | bool | `False` | **CRITIC9** | Box-centered (relative) coordinates for critic. Set False for absolute coords (CRITIC7) |
| `--use_relative_obs_critic` | bool | `False` | **CRITIC11** | Relative observations: [robot1_to_box, robot2_to_box, inter_robot_dist, goal_to_box]. **Takes highest priority** |

**Note:** These are mutually exclusive. Choose the one that best fits your experiment.

**Examples:**
```bash
# Concatenated critic (recommended for heterogeneous agents)
--use_concat_agent_observations_critic True

# Goal-centered critic
--use_goal_centered_critic True

# Box-centered critic
--use_box_centered_critic True

# Relative observations critic (highest priority if multiple are set)
--use_relative_obs_critic True
```

---

### 🎁 Reward Shaping Flags

| Flag | Type | Default | ID | Description |
|------|------|---------|-------|-------------|
| `--individualized_rewards` | bool | `False` | - | Enable individualized rewards for HAPPO (prevents freeloading) |
| `--shared_gated_rewards` | bool | `False` | **Iter8** | Gate all shared rewards by min agent engagement (prevents freeloading) |
| `--mapush_og_rewards_teamified` | bool | `False` | - | **Use original 7 MAPush rewards converted to team rewards**. Disables goal_push_bonus, proximity_penalty. Uses symmetric OCB ±0.004, original collision scale -0.0025 |
| `--cooperation_rewards` | bool | `False` | **CRITIC12** | Three-tier cooperation bonuses: dual_engagement, synchronized_contact, bilateral_push |
| `--collaboration_rewards` | bool | `False` | **CRITIC15 v4** | Dual pushing bonus when both agents push toward goal simultaneously. **Use with --mapush_og_rewards_teamified True** |
| `--reward_scale_testing` | bool | `False` | - | Reserved for future reward scale experiments (currently no effect) |

**Recommended Combination:**
```bash
# Original MAPush rewards (teamified) + collaboration bonus
--mapush_og_rewards_teamified True --collaboration_rewards True
```

---

### ⚙️ Training Hyperparameters

| Flag | Type | Default | Description |
|------|------|---------|-------------|
| `--n_rollout_threads` | int | Config | Number of parallel environments (typical: 50-500) |
| `--num_env_steps` | int | Config | Total number of environment steps to train |
| `--episode_length` | int | Config | Rollout length: steps to collect before update |

**Examples:**
```bash
# Quick test
--n_rollout_threads 50 --num_env_steps 10000

# Short training
--n_rollout_threads 200 --num_env_steps 10000000  # 10M

# Full training
--n_rollout_threads 500 --num_env_steps 100000000  # 100M

# Long training
--n_rollout_threads 500 --num_env_steps 500000000  # 500M
```

---

### 📊 Logging & Checkpointing

| Flag | Type | Default | Description |
|------|------|---------|-------------|
| `--use_tensorboard` | bool | `True` | Use TensorBoard for logging |
| `--checkpoint` | str | `None` | Path to checkpoint to resume training from |

**Examples:**
```bash
# Resume from checkpoint
--checkpoint ./HARL/results/mapush/cuboid/happo/my_exp/seed-1-20260116_123456/checkpoints/10M

# Disable tensorboard
--use_tensorboard False
```

---

## 🧪 Experiment Recipes

### 1. Baseline Homogeneous (2x Go1)
```bash
python HARL/harl_mapush/train.py \
  --exp_name baseline_homogeneous \
  --n_rollout_threads 500 \
  --num_env_steps 100000000
```

### 2. Heterogeneous with Concatenated Critic
```bash
python HARL/harl_mapush/train.py \
  --exp_name hetero_concat_critic \
  --hetero_agent jackal \
  --use_concat_agent_observations_critic True \
  --n_rollout_threads 500 \
  --num_env_steps 100000000
```

### 3. Heterogeneous with OG MAPush Rewards (Teamified)
```bash
python HARL/harl_mapush/train.py \
  --exp_name hetero_og_rewards \
  --hetero_agent jackal \
  --mapush_og_rewards_teamified True \
  --n_rollout_threads 500 \
  --num_env_steps 100000000
```

### 4. Heterogeneous with Full Cooperation Stack
```bash
python HARL/harl_mapush/train.py \
  --exp_name hetero_full_coop \
  --hetero_agent jackal \
  --use_concat_agent_observations_critic True \
  --mapush_og_rewards_teamified True \
  --cooperation_rewards True \
  --collaboration_rewards True \
  --n_rollout_threads 500 \
  --num_env_steps 100000000
```

### 5. Heterogeneous with Goal-Centered Critic
```bash
python HARL/harl_mapush/train.py \
  --exp_name hetero_goal_centered \
  --hetero_agent jackal \
  --use_goal_centered_critic True \
  --mapush_og_rewards_teamified True \
  --n_rollout_threads 500 \
  --num_env_steps 100000000
```

### 6. Quick Debug Test (1K steps)
```bash
python HARL/harl_mapush/train.py \
  --exp_name debug_test \
  --hetero_agent jackal \
  --n_rollout_threads 10 \
  --num_env_steps 1000
```

---

## 🔬 Experimental Combinations

### Anti-Freeloading Stack
```bash
# Individualized rewards + gated shared rewards
--individualized_rewards \
--shared_gated_rewards
```

### Maximum Cooperation
```bash
# All cooperation mechanisms enabled
--cooperation_rewards True \
--collaboration_rewards True \
--mapush_og_rewards_teamified True
```

### Relative Observation Stack (CRITIC11 takes priority)
```bash
# Relative observations for critic (highest priority)
--use_relative_obs_critic True \
--cooperation_rewards True
```

---

## 📁 Output Structure

Training outputs are saved to:
```
HARL/results/mapush/cuboid/happo/<exp_name>/seed-<seed>-<timestamp>/
├── checkpoints/
│   ├── 10M/
│   │   ├── actor_agent0.pt
│   │   ├── actor_agent1.pt
│   │   └── critic_agent.pt
│   ├── 20M/
│   ├── 30M/
│   └── ...
├── logs/
└── tensorboard/
```

---

## 🐛 Common Issues & Solutions

### Issue: Training crashes at start
**Solution:** Check environment creation with test script:
```bash
python test_hetero_env.py
```

### Issue: NaN in critic loss
**Solution:**
- Reduce learning rate
- Check critic architecture (try different critic flags)
- Ensure proper normalization

### Issue: One agent not learning (freeloading)
**Solution:**
```bash
--individualized_rewards --shared_gated_rewards
```

### Issue: Poor cooperation
**Solution:**
```bash
--cooperation_rewards True --collaboration_rewards True
```

---

## 📊 Monitoring Training

### TensorBoard
```bash
tensorboard --logdir HARL/results/mapush/cuboid/happo/<exp_name>/seed-<seed>-<timestamp>/tensorboard
```

### Key Metrics to Watch
- `train/episode_rewards` - Total episode reward
- `train/critic_loss` - Critic convergence
- `train/actor_loss_agent0` - Agent 0 policy loss
- `train/actor_loss_agent1` - Agent 1 policy loss
- `eval/success_rate` - Task success rate (if implemented)

---

## 🔍 Testing Trained Models

See `HARL/harl_mapush/test.py` for testing commands.

**Example:**
```bash
cd HARL/harl_mapush

python test.py \
  --checkpoint ./results/mapush/cuboid/happo/my_exp/seed-1-20260116_123456/checkpoints/100M \
  --mode viewer \
  --num_episodes 10 \
  --hetero_agent jackal
```

---

## 📚 Related Documentation

- **Heterogeneous Implementation:** `claude_summaries/heterogeneous_agent_implementation.md`
- **Jackal Integration:** `claude_summaries/jackal_integration.md`
- **HARL Overview:** `claude_summaries/claude_summary_HARL.md`
- **MAPush Overview:** `claude_summaries/claude_summary_MAPush.md`

---

## ✅ Implementation Status (2026-01-16)

- ✅ All core training flags implemented
- ✅ Heterogeneous agent support (`--hetero_agent jackal`)
- ✅ All critic architectures (CRITIC7-16) working
- ✅ All reward configurations tested
- ✅ Environment creation & stepping verified
- ✅ Buffer initialization for mixed DOF counts
- ✅ Observation computation for heterogeneous agents
- ✅ Torque computation for mixed control types

**Status:** 🟢 **READY FOR TRAINING**

---

## 🎯 Recommended Starting Point

For heterogeneous Go1 + Jackal training:

```bash
cd /home/gvlab/new-universal-MAPush

conda run -n mapush python HARL/harl_mapush/train.py \
  --exp_name go1_jackal_concat_v1 \
  --hetero_agent jackal \
  --use_concat_agent_observations_critic True \
  --mapush_og_rewards_teamified True \
  --n_rollout_threads 500 \
  --num_env_steps 100000000 \
  --seed 1
```

This configuration:
- ✅ Uses concatenated observations (handles heterogeneity well)
- ✅ Uses proven original MAPush rewards
- ✅ 500 parallel environments (good throughput)
- ✅ 100M steps (sufficient for convergence)

**Estimated training time:** ~24-48 hours on GPU

---

**Last Updated:** 2026-01-16
**Author:** Claude (Anthropic)
