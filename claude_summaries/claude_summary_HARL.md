# HARL (Heterogeneous-Agent Reinforcement Learning) Repository Summary

**Last Updated:** 2025-12-13

## Overview

HARL is a standalone PyTorch library implementing **Heterogeneous-Agent Reinforcement Learning** algorithms. This is a separate repository from MAPush that you want to **integrate** to replace the current MAPPO/PPO implementation with **HAPPO** (and potentially other HARL algorithms).

**Repository:** https://github.com/PKU-MARL/HARL
**Papers:**
- [HARL (JMLR 2024)](https://jmlr.org/papers/v25/23-0488.html)
- [MEHARL (ICLR 2024 spotlight)](https://openreview.net/forum?id=tmqOhBC4a5)

---

## Why HARL? Key Advantages

### vs. MAPPO (current MAPush approach):
1. **Heterogeneous Agent Support** - Designed for agents with different observation/action spaces
2. **Sequential Update Scheme** - Agents update in sequence, accounting for each other's policy changes
3. **Theoretical Guarantees** - Monotonic improvement and convergence to equilibrium
4. **Superior Performance** - Demonstrated across 7 benchmarks (SMAC, MAMuJoCo, MPE, GRF, Bi-DexHands, etc.)
5. **No Parameter Sharing Required** - Doesn't rely on the restrictive parameter-sharing trick

### Sequential vs. Simultaneous Updates:
- **MAPPO/MADDPG:** All agents update simultaneously (can lead to instability)
- **HARL Algorithms:** Agents update sequentially with importance sampling to account for policy changes
- **Result:** Better coordination, more stable training

---

## HARL Repository Structure

```
/home/gvlab/new-universal-MAPush/HARL/
├── harl/                          # Main HARL package
│   ├── algorithms/                # Algorithm implementations
│   │   ├── actors/                # Actor (policy) algorithms
│   │   │   ├── happo.py           # HAPPO (YOUR PRIMARY TARGET)
│   │   │   ├── hatrpo.py          # HA Trust Region Policy Optimization
│   │   │   ├── haa2c.py           # HA Advantage Actor-Critic
│   │   │   ├── haddpg.py          # HA Deep Deterministic Policy Gradient
│   │   │   ├── hatd3.py           # HA Twin Delayed DDPG
│   │   │   ├── hasac.py           # HA Soft Actor-Critic
│   │   │   ├── had3qn.py          # HA Dueling DQN
│   │   │   ├── mappo.py           # MAPPO (for comparison)
│   │   │   ├── on_policy_base.py  # Base class for on-policy actors
│   │   │   └── off_policy_base.py # Base class for off-policy actors
│   │   └── critics/               # Critic (value function) implementations
│   │       ├── v_critic.py        # State-value critic
│   │       ├── continuous_q_critic.py
│   │       ├── discrete_q_critic.py
│   │       └── ...
│   ├── runners/                   # Training loop runners
│   │   ├── on_policy_base_runner.py   # Base runner for on-policy algos
│   │   ├── on_policy_ha_runner.py     # HAPPO/HATRPO/HAA2C runner
│   │   ├── on_policy_ma_runner.py     # MAPPO runner
│   │   ├── off_policy_base_runner.py
│   │   ├── off_policy_ha_runner.py    # HADDPG/HATD3/HASAC runner
│   │   └── __init__.py                # RUNNER_REGISTRY
│   ├── envs/                      # Environment interfaces
│   │   ├── smac/                  # StarCraft Multi-Agent Challenge
│   │   ├── smacv2/                # SMAC v2
│   │   ├── mamujoco/              # Multi-Agent MuJoCo
│   │   ├── pettingzoo_mpe/        # Multi-Particle Env (MPE)
│   │   ├── gym/                   # Single-agent Gym
│   │   ├── football/              # Google Research Football
│   │   ├── dexhands/              # Bi-DexterousHands (Isaac Gym)
│   │   ├── lag/                   # Light Aircraft Game
│   │   └── env_wrappers.py        # Vec env wrappers
│   ├── models/                    # Neural network architectures
│   │   ├── policy_models/         # Policy network models
│   │   │   ├── stochastic_policy.py
│   │   │   ├── deterministic_policy.py
│   │   │   └── squashed_gaussian_policy.py
│   │   ├── value_function_models/ # Value network models
│   │   └── base/                  # Base network components
│   ├── common/                    # Common utilities
│   │   ├── buffers/               # Replay buffers
│   │   │   ├── on_policy_actor_buffer.py    # Per-agent buffer
│   │   │   ├── on_policy_critic_buffer_ep.py # Centralized critic (EP)
│   │   │   ├── on_policy_critic_buffer_fp.py # Feature-pruned critic (FP)
│   │   │   ├── off_policy_buffer_*.py
│   │   │   └── ...
│   │   ├── base_logger.py         # Base logger class
│   │   └── valuenorm.py           # Value normalization
│   ├── utils/                     # Utility functions
│   │   ├── envs_tools.py          # Env creation, seeding
│   │   ├── configs_tools.py       # Config loading/saving
│   │   ├── models_tools.py        # Model utilities
│   │   └── trans_tools.py         # Tensor transformations
│   └── configs/                   # Configuration files
│       ├── algos_cfgs/            # Algorithm configs
│       │   ├── happo.yaml         # HAPPO hyperparameters
│       │   ├── hatrpo.yaml
│       │   ├── mappo.yaml
│       │   └── ...
│       └── envs_cfgs/             # Environment configs
│           ├── mamujoco.yaml
│           ├── pettingzoo_mpe.yaml
│           ├── dexhands.yaml
│           └── ...
├── examples/                      # Training entry points
│   └── train.py                   # Main training script
├── tuned_configs/                 # Pre-tuned hyperparameters
│   ├── mamujoco/
│   ├── smac/
│   ├── dexhands/
│   └── ...
├── setup.py                       # Package installation
└── README.md                      # Documentation
```

---

## Key Components Deep Dive

### 1. HAPPO Algorithm (`harl/algorithms/actors/happo.py`)

**Class:** `HAPPO(OnPolicyBase)`

**Key Features:**
- **Inherits from:** `OnPolicyBase` (provides common policy update logic)
- **Sequential Update:** Updates agents one at a time, using importance sampling
- **PPO-style Clipping:** Uses clipped surrogate objective for stability
- **Entropy Regularization:** Encourages exploration

**Key Parameters (from `happo.yaml`):**
- `clip_param`: 0.2 - PPO clip parameter
- `ppo_epoch`: 5 - Number of update epochs per data batch
- `actor_num_mini_batch`: 1 - Mini-batches for actor update
- `entropy_coef`: 0.01 - Entropy bonus coefficient
- `lr`: 0.0005 - Actor learning rate
- `critic_lr`: 0.0005 - Critic learning rate
- `use_gae`: True - Use Generalized Advantage Estimation
- `gamma`: 0.99 - Discount factor
- `gae_lambda`: 0.95 - GAE lambda
- `share_param`: False - Whether to share parameters across agents
- `fixed_order`: False - Whether to use fixed agent update order

**Update Method:**
```python
def update(self, sample):
    # Computes:
    # 1. Action log probabilities
    # 2. Importance weights: exp(new_log_prob - old_log_prob)
    # 3. Surrogate loss with clipping
    # 4. Entropy bonus
    # 5. Gradient update
```

**Train Method:**
```python
def train(self, actor_buffer, advantages, state_type):
    # Performs ppo_epoch epochs of minibatch updates
    # Normalizes advantages (if state_type == "EP")
```

### 2. Sequential Update Scheme (`harl/runners/on_policy_ha_runner.py`)

**Class:** `OnPolicyHARunner(OnPolicyBaseRunner)`

**Training Loop (Simplified):**
```python
def train(self):
    # 1. Compute advantages from critic
    # 2. Initialize factor = 1 (importance sampling accumulator)
    # 3. For each agent in random order:
    #    a. Compute old action log probs
    #    b. Update actor with current factor
    #    c. Compute new action log probs
    #    d. Update factor *= exp(new - old)
    # 4. Update shared critic
```

**Key Insight:** Factor accumulates the product of importance weights from all previously updated agents. This ensures each agent's update accounts for changes made by others.

### 3. Buffers

**OnPolicyActorBuffer:** Per-agent buffer storing (obs, actions, rewards, masks, etc.)
- Each agent has its own buffer
- Shape: `[episode_length, n_rollout_threads, ...]`

**OnPolicyCriticBuffer (EP/FP):**
- **EP (Environment Provided):** All agents share the same global state
- **FP (Feature Pruned):** Each agent has a different state (centralized obs)
- Stores (state, value_preds, returns, advantages)

### 4. Runner System

**RUNNER_REGISTRY** (in `harl/runners/__init__.py`):
```python
RUNNER_REGISTRY = {
    "happo": OnPolicyHARunner,    # <-- Your target
    "hatrpo": OnPolicyHARunner,
    "haa2c": OnPolicyHARunner,
    "mappo": OnPolicyMARunner,    # Simultaneous update
    ...
}
```

**OnPolicyBaseRunner:**
- Initializes environments, actors, critics, buffers
- Implements `run()`, `collect()`, `eval()` methods
- Handles logging, checkpointing, rendering

**OnPolicyHARunner:**
- Extends base with sequential update logic
- Overrides `train()` method

---

## Environment Interface Requirements

To integrate HARL with MAPush, you need to create an environment wrapper that matches HARL's interface:

```python
class Env:
    def __init__(self, env_args):
        self.n_agents = ...                      # Number of agents
        self.share_observation_space = ...       # List of global state spaces (one per agent)
        self.observation_space = ...             # List of observation spaces (one per agent)
        self.action_space = ...                  # List of action spaces (one per agent)

    def step(self, actions):
        """
        Args:
            actions: np.array of shape [n_agents, n_envs, action_dim]
        Returns:
            obs: np.array [n_envs, n_agents, obs_dim]
            state: np.array [n_envs, n_agents, state_dim] or [n_envs, state_dim]
            rewards: np.array [n_envs, n_agents, 1]
            dones: np.array [n_envs, n_agents] (bool)
            infos: list of dicts
            available_actions: None or np.array
        """
        return obs, state, rewards, dones, infos, available_actions

    def reset(self):
        """
        Returns:
            obs: np.array [n_envs, n_agents, obs_dim]
            state: np.array [n_envs, n_agents, state_dim] or [n_envs, state_dim]
            available_actions: None or np.array
        """
        return obs, state, available_actions

    def seed(self, seed):
        pass

    def render(self):
        pass

    def close(self):
        pass
```

**Key Differences from MAPush (OpenRL):**
- **Observation shape:** HARL expects `[n_envs, n_agents, obs_dim]`, MAPush uses `[n_envs, n_agents, obs_dim]` ✓ (same)
- **Action shape:** HARL takes `[n_agents, n_envs, action_dim]`, MAPush gives `[n_envs, n_agents, action_dim]` (need transpose)
- **Reward shape:** HARL expects `[n_envs, n_agents, 1]`, MAPush has `[n_envs, n_agents]` (need unsqueeze)
- **State/share_obs:** HARL uses `share_observation_space` (centralized critic state), MAPush doesn't explicitly use this
- **Available actions:** HARL supports discrete action masking, MAPush doesn't use this

---

## Installation & Setup

### 1. Install HARL

```bash
cd /home/gvlab/new-universal-MAPush/HARL
pip install -e .
```

**Dependencies (from `setup.py`):**
- `torch>=1.9.0`
- `pyyaml>=5.3.1`
- `tensorboard>=2.2.1`
- `tensorboardX`
- `setproctitle`

### 2. Compatibility with Isaac Gym

HARL already supports Isaac Gym via the **dexhands** environment:
- See `harl/envs/dexhands/dexhands_env.py` for reference implementation
- HARL imports `isaacgym` before PyTorch (required)
- Should be compatible with your MAPush setup

---

## Usage (Standalone HARL)

### Training Command:
```bash
cd /home/gvlab/new-universal-MAPush/HARL/examples
python train.py --algo happo --env pettingzoo_mpe --exp_name test
```

### Load from Config:
```bash
python train.py --load_config /path/to/config.json --exp_name test
```

### Batch Running (Multiple Seeds):
```bash
for seed in $(seq 1 3); do
    python train.py --algo happo --env mamujoco --exp_name test --seed $seed
done
```

### Override Config Parameters:
```bash
python train.py --algo happo --env pettingzoo_mpe --exp_name test \
    --n_rollout_threads 100 \
    --num_env_steps 50000000 \
    --lr 0.0003
```

---

## Integration Strategy: HARL + MAPush

### Option 1: Wrap MAPush as a HARL Environment (Recommended)

**Steps:**
1. Create `harl/envs/mapush/mapush_env.py` following `dexhands_env.py` structure
2. Create `harl/envs/mapush/mapush_logger.py` for logging
3. Create `harl/configs/envs_cfgs/mapush.yaml` for env config
4. Register env in `harl/utils/envs_tools.py:make_train_env()`
5. Register logger in `harl/envs/__init__.py`
6. Use HARL's training script directly

**Advantages:**
- Clean separation, use HARL exactly as designed
- Easy to switch between HARL algorithms
- Leverage HARL's tuned configs and logging

**Disadvantages:**
- More refactoring required
- Need to adapt MAPush's observation/reward structure

### Option 2: Port HAPPO into MAPush (OpenRL-style)

**Steps:**
1. Copy `harl/algorithms/actors/happo.py` → `openrl_ws/algorithms/`
2. Copy `harl/runners/on_policy_ha_runner.py` → `openrl_ws/runners/`
3. Adapt buffers from HARL to OpenRL format
4. Modify `openrl_ws/train.py` to support HAPPO
5. Create `openrl_ws/cfgs/happo.yaml`

**Advantages:**
- Minimal changes to existing MAPush workflow
- Keep current environment interface

**Disadvantages:**
- Lose HARL's ecosystem benefits
- Manual porting may introduce bugs
- Harder to maintain/update

### Option 3: Hybrid Approach

**Steps:**
1. Create thin wrapper converting MAPush env → HARL interface
2. Import HARL algorithms directly in MAPush
3. Use HARL runners but keep MAPush env unchanged

---

## Recommended Integration Path (Option 1 Detailed)

### Step 1: Create MAPush Environment Wrapper

**File:** `HARL/harl/envs/mapush/mapush_env.py`

```python
import torch
import numpy as np
from mqe.envs.utils import make_mqe_env
from mqe.utils.task_registry import task_registry

class MAPushEnv:
    def __init__(self, env_args):
        # env_args should contain: task, num_envs, etc.
        self.env_args = env_args

        # Create MAPush environment
        from types import SimpleNamespace
        args = SimpleNamespace(
            task=env_args["task"],  # e.g., "go1push_mid"
            headless=True,
            # ... other args from env_args
        )
        self.env, self.env_cfg = make_mqe_env(env_args["task"], args)

        self.n_envs = self.env.num_envs
        self.n_agents = self.env.num_agents

        # HARL expects list of spaces (one per agent)
        # For homogeneous case, duplicate
        self.observation_space = [self.env.observation_space] * self.n_agents
        self.action_space = [self.env.action_space] * self.n_agents

        # Share observation space (for centralized critic)
        # Option 1: Use concatenated obs of all agents
        # Option 2: Use global state if available
        self.share_observation_space = [self.observation_space[0]] * self.n_agents

    def step(self, actions):
        # actions: [n_agents, n_envs, action_dim] -> need [n_envs, n_agents, action_dim]
        actions = torch.from_numpy(actions.transpose(1, 0, 2)).cuda()

        obs, rewards, dones, infos = self.env.step(actions)

        # Convert to numpy
        obs_np = obs.cpu().numpy()  # [n_envs, n_agents, obs_dim]
        rewards_np = rewards.cpu().numpy()  # [n_envs, n_agents]
        dones_np = dones.cpu().numpy()  # [n_envs]

        # Prepare state (for centralized critic)
        state_np = obs_np  # Or construct global state

        # Reshape rewards
        rewards_np = rewards_np[..., np.newaxis]  # [n_envs, n_agents, 1]

        # Dones - broadcast to all agents
        dones_np = np.broadcast_to(dones_np[:, np.newaxis], (self.n_envs, self.n_agents))

        # Infos - list of dicts
        infos_list = [{} for _ in range(self.n_envs)]

        return obs_np, state_np, rewards_np, dones_np, infos_list, None

    def reset(self):
        obs = self.env.reset()
        obs_np = obs.cpu().numpy()
        state_np = obs_np
        return obs_np, state_np, None

    def seed(self, seed):
        pass

    def close(self):
        self.env.close()
```

### Step 2: Create Logger

**File:** `HARL/harl/envs/mapush/mapush_logger.py`

```python
from harl.common.base_logger import BaseLogger

class MAPushLogger(BaseLogger):
    def get_task_name(self):
        return self.env_args.get("task", "go1push_mid")
```

### Step 3: Register Environment

**Edit:** `HARL/harl/utils/envs_tools.py`

Add to `make_train_env()`:
```python
elif env_name == "mapush":
    from harl.envs.mapush.mapush_env import MAPushEnv
    env = MAPushEnv(env_args)
```

Add to `get_num_agents()`:
```python
elif env == "mapush":
    return envs.n_agents
```

**Edit:** `HARL/harl/envs/__init__.py`

```python
from harl.envs.mapush.mapush_logger import MAPushLogger

LOGGER_REGISTRY = {
    ...
    "mapush": MAPushLogger,
}
```

**Edit:** `HARL/examples/train.py`

Add to choices:
```python
parser.add_argument(
    "--env",
    choices=[..., "mapush"],
    ...
)
```

### Step 4: Create Environment Config

**File:** `HARL/harl/configs/envs_cfgs/mapush.yaml`

```yaml
task: go1push_mid
num_envs: 500
# Add other MAPush-specific configs
```

### Step 5: Train with HAPPO

```bash
cd /home/gvlab/new-universal-MAPush/HARL/examples
python train.py --algo happo --env mapush --exp_name happo_go1push \
    --n_rollout_threads 500 \
    --num_env_steps 100000000 \
    --episode_length 200
```

---

## Key Configuration Parameters

### HAPPO-specific (from `happo.yaml`):

**Training:**
- `n_rollout_threads`: 20 → **Set to 500 for MAPush** (parallel envs)
- `num_env_steps`: 10000000 → **Set to 100000000 for MAPush**
- `episode_length`: 200 → **Matches MAPush** ✓
- `use_valuenorm`: True - Normalize value function
- `use_linear_lr_decay`: False

**Model:**
- `hidden_sizes`: [128, 128] - MLP layer sizes
- `activation_func`: relu
- `lr`: 0.0005 - Actor learning rate
- `critic_lr`: 0.0005 - Critic learning rate

**Algorithm:**
- `ppo_epoch`: 5 - Update epochs per data collection
- `critic_epoch`: 5
- `clip_param`: 0.2 - PPO clip parameter
- `entropy_coef`: 0.01 - Entropy regularization
- `gamma`: 0.99 - Discount factor
- `gae_lambda`: 0.95 - GAE parameter
- `share_param`: False - **Set False for heterogeneous** ✓
- `fixed_order`: False - Random agent update order

---

## Heterogeneous Agent Support

### For Goal 3: Different Observations (2 Go1s, one with fewer inputs)

**HARL natively supports this!**

1. **Different observation spaces:**
   ```python
   self.observation_space = [
       gym.spaces.Box(..., shape=(8,)),  # Agent 0: full obs
       gym.spaces.Box(..., shape=(7,)),  # Agent 1: reduced obs
   ]
   ```

2. **Set `share_param: False` in config**

3. **HARL automatically creates separate actor networks per agent**

### For Goal 4: Different Robot Types (Go1 + Wheeled)

**HARL supports this too!**

1. **Different action spaces:**
   ```python
   self.action_space = [
       gym.spaces.Box(..., shape=(3,)),  # Go1: [vx, vy, w]
       gym.spaces.Box(..., shape=(2,)),  # Wheeled: [v, w]
   ]
   ```

2. **HARL creates separate policies automatically**

3. **Shared critic can still use global state**

---

## State Type: EP vs FP

**EP (Environment Provided):**
- All agents receive the **same global state** for centralized critic
- Use when environment provides a shared observation
- Example: `state = [box_pos, target_pos, all_agent_pos]` (same for all)

**FP (Feature Pruned):**
- Each agent has **different centralized observations**
- Use when agents have different privileged info
- Example: `state_0 = [box_pos, agent0_pos, ...]`, `state_1 = [box_pos, agent1_pos, ...]`

**For MAPush:** Likely use **EP** with shared global state.

**Set in:** `HARL/harl/configs/envs_cfgs/mapush.yaml`
```yaml
state_type: "EP"  # or "FP"
```

---

## Logging and Output

### Training Logs:
- **Location:** `HARL/results/<env>/<task>/<algo>/<exp_name>/seed-<seed>-<timestamp>/`
- **TensorBoard:** `logs/` subdirectory
- **Models:** `models/` subdirectory (checkpoints)
- **Config:** `config.json` (full experiment config)

### Logging Output:
- Episode rewards
- Success rate (if you implement in logger)
- Actor/critic losses
- Gradient norms

### Custom Logging:
Extend `MAPushLogger` to add custom metrics:
```python
class MAPushLogger(BaseLogger):
    def episode_log(self, episode, actor_train_infos, critic_train_info, ...):
        # Add MAPush-specific logging (e.g., success rate, push metrics)
        super().episode_log(...)
```

---

## Algorithm Comparison

| Algorithm | Type       | Action Space             | Characteristics                      |
|-----------|------------|--------------------------|--------------------------------------|
| **HAPPO** | On-policy  | Continuous/Discrete/Multi| PPO-based, sequential update         |
| HATRPO    | On-policy  | Continuous/Discrete      | Trust region, sequential update      |
| HAA2C     | On-policy  | Continuous/Discrete/Multi| A2C-based, sequential update         |
| HASAC     | Off-policy | Continuous/Discrete/Multi| SAC-based, maximum entropy           |
| HADDPG    | Off-policy | Continuous               | DDPG-based, deterministic policies   |
| HATD3     | Off-policy | Continuous               | TD3-based, twin Q-networks           |
| MAPPO     | On-policy  | Continuous/Discrete/Multi| Simultaneous update (no heterogeneity)|

**For MAPush (continuous actions, on-policy):** **HAPPO** is the best choice ✓

---

## Testing & Validation

### 1. Test HARL Standalone on Simple Env

```bash
cd /home/gvlab/new-universal-MAPush/HARL/examples
python train.py --algo happo --env pettingzoo_mpe --exp_name test \
    --num_env_steps 1000000 --n_rollout_threads 10
```

Verify HARL works on your system.

### 2. Test MAPush Wrapper

Create minimal test:
```python
from harl.envs.mapush.mapush_env import MAPushEnv

env_args = {"task": "go1push_mid", "num_envs": 10}
env = MAPushEnv(env_args)

obs, state, _ = env.reset()
print("Obs shape:", obs.shape)  # Should be [10, 2, obs_dim]

actions = np.random.randn(2, 10, 3)  # [n_agents, n_envs, action_dim]
obs, state, rewards, dones, infos, _ = env.step(actions)
print("Reward shape:", rewards.shape)  # Should be [10, 2, 1]
```

### 3. Short Training Run

```bash
python train.py --algo happo --env mapush --exp_name test \
    --num_env_steps 1000000 --n_rollout_threads 10 --episode_length 200
```

Monitor for errors, check learning curves.

---

## Troubleshooting Integration

### Issue: Shape Mismatches

**Symptom:** Tensor shape errors during training

**Solution:** Check wrapper's observation/action/reward shapes match HARL expectations

### Issue: Isaac Gym Import Errors

**Symptom:** `ImportError: libpython3.8m.so.1.0`

**Solution:**
```bash
export LD_LIBRARY_PATH=/path/to/conda/envs/mapush/lib
```

### Issue: Shared Observation Space Undefined

**Symptom:** `AttributeError: 'MAPushEnv' object has no attribute 'share_observation_space'`

**Solution:** Ensure wrapper defines all required attributes

### Issue: Performance Degradation

**Symptom:** HAPPO performs worse than MAPPO

**Solution:**
- Tune hyperparameters (lr, clip_param, entropy_coef)
- Check reward scaling
- Verify advantage computation
- Try longer training (HAPPO may need more steps initially)

---

## Hyperparameter Tuning Tips

### Start with HARL defaults from `happo.yaml`

### Key parameters to tune for MAPush:

1. **Learning rates:**
   - `lr`: 0.0005 (actor)
   - `critic_lr`: 0.0005
   - Try: [0.0001, 0.0003, 0.0005, 0.001]

2. **Clip parameter:**
   - `clip_param`: 0.2
   - Try: [0.1, 0.2, 0.3]

3. **Entropy coefficient:**
   - `entropy_coef`: 0.01
   - Try: [0.001, 0.01, 0.05]

4. **GAE parameters:**
   - `gamma`: 0.99 (discount)
   - `gae_lambda`: 0.95
   - Usually keep these standard

5. **Update epochs:**
   - `ppo_epoch`: 5
   - Try: [3, 5, 10]

6. **Batch size:**
   - `actor_num_mini_batch`: 1
   - `critic_num_mini_batch`: 1
   - Try: [1, 2, 4] (larger = more stable but slower)

---

## Comparison: HARL vs Current MAPush Setup

| Aspect                  | Current MAPush (OpenRL)       | HARL Integration              |
|-------------------------|-------------------------------|-------------------------------|
| **Algorithm**           | PPO/MAPPO                     | HAPPO                         |
| **Update Scheme**       | Simultaneous                  | Sequential                    |
| **Heterogeneity**       | Limited (param sharing)       | Native support                |
| **Training Script**     | `openrl_ws/train.py`          | `HARL/examples/train.py`      |
| **Config Location**     | `openrl_ws/cfgs/ppo.yaml`     | `HARL/harl/configs/algos_cfgs/happo.yaml` |
| **Env Interface**       | OpenRL wrapper                | HARL wrapper                  |
| **Logging**             | OpenRL logger                 | HARL logger + TensorBoard     |
| **Checkpointing**       | Custom                        | HARL runner handles           |
| **Multi-agent Buffers** | Shared buffer                 | Per-agent actor buffers       |
| **Critic**              | Centralized (implicit)        | Centralized (explicit EP/FP)  |

---

## Next Steps Summary

### Phase 1: Setup & Testing (1-2 days)
1. Install HARL in your environment
2. Test HARL standalone on simple env (e.g., MPE)
3. Verify Isaac Gym compatibility

### Phase 2: Integration (3-5 days)
1. Create `mapush_env.py` wrapper
2. Create `mapush_logger.py`
3. Register environment in HARL
4. Create `mapush.yaml` config
5. Test wrapper with dummy actions

### Phase 3: Training (1-2 days)
1. Short training run (1M steps) to validate
2. Full training run (100M steps)
3. Compare with MAPPO baseline

### Phase 4: Heterogeneous Setup (2-3 days)
1. Implement different observation spaces
2. Test with HAPPO
3. Implement different robot types
4. Validate training

---

## References & Resources

**HARL Repository:** https://github.com/PKU-MARL/HARL

**Papers:**
- HARL: https://jmlr.org/papers/v25/23-0488.html
- MEHARL: https://openreview.net/forum?id=tmqOhBC4a5

**Example Environments:**
- DexHands (Isaac Gym): `HARL/harl/envs/dexhands/` - Most similar to your use case
- MAMuJoCo: `HARL/harl/envs/mamujoco/` - Multi-agent continuous control
- MPE: `HARL/harl/envs/pettingzoo_mpe/` - Simple multi-agent

**Tuned Configs:**
- `HARL/tuned_configs/dexhands/` - Closest to your domain
- `HARL/tuned_configs/mamujoco/` - Continuous control reference

---

## Contact & Support

**HARL Issues:** https://github.com/PKU-MARL/HARL/issues

**PKU-MARL Lab:** https://github.com/PKU-MARL

---

**Good luck with the HAPPO integration! The sequential update scheme should provide better coordination for your multi-agent pushing task.** 🚀
