# HARL Integration Proposal for MAPush

**Date:** 2025-12-13
**Objective:** Integrate HAPPO from HARL for mid-level controller training (k=2 agents, cuboid task)

---

## Requirements Summary

### 1. Training Requirements
- ✅ Save models and logs to `HARL/results/...` folder
- ✅ Save checkpoints every 10M steps
- ✅ 3 neural networks per checkpoint: `actor_agent0.pt`, `actor_agent1.pt`, `critic_agent.pt`
- ✅ Checkpoint folders named: `10M/`, `20M/`, `30M/`, etc.

### 2. Testing Requirements
- ✅ `test.py` file with two modes:
  - **Viewer mode:** Show N episodes sequentially (one at a time)
  - **Calculator mode:** Compute statistics over multiple environments
- ✅ Configurable: seed, number of episodes (viewer), number of envs (calc)
- ✅ Statistics to show (same as MAPush):
  - **Success rate** (priority)
  - **Collision rate** (priority)
  - Finished time
  - Collaboration degree

---

## Analysis: Current State

### MAPush Structure
```
openrl_ws/
├── train.py           # Training entry point
├── test.py            # Testing with viewer/calculator modes
├── utils.py           # Environment wrapper (mqe_openrl_wrapper)
└── cfgs/ppo.yaml      # PPO config

results/
└── <mm-dd-hh>_cuboid/
    ├── checkpoints/
    │   ├── rl_model_10000000_steps/
    │   │   └── module.pt           # Single file checkpoint
    │   ├── rl_model_20000000_steps/
    │   └── ...
    ├── task/
    ├── success_rate.txt
    └── log.txt
```

### HARL Structure
```
HARL/
├── examples/train.py  # Training entry point
├── harl/
│   ├── runners/
│   │   └── on_policy_base_runner.py  # save() at line 724
│   ├── envs/          # Environment interfaces
│   └── configs/
└── results/
    └── <env>/<task>/<algo>/<exp_name>/seed-<seed>-<timestamp>/
        ├── logs/      # TensorBoard logs
        ├── models/    # Model checkpoints
        │   ├── actor_agent0.pt
        │   ├── actor_agent1.pt
        │   └── critic_agent.pt
        └── config.json
```

### Key Differences
| Aspect | MAPush | HARL |
|--------|--------|------|
| Checkpoint frequency | Every 10M steps | Every `eval_interval` episodes |
| Checkpoint naming | `rl_model_<steps>_steps/` | Single `models/` folder |
| Model files | `module.pt` (single) | `actor_agent0.pt`, `actor_agent1.pt`, `critic_agent.pt` |
| Test script | `openrl_ws/test.py` | None (uses render mode in train.py) |
| Statistics tracking | Custom buffers in env | Logger-based |

---

## Proposed Integration Architecture

### Option A: Minimal Modification (Recommended)

**Philosophy:** Wrap MAPush environment for HARL, customize HARL runner for your requirements

**Directory Structure:**
```
HARL/
├── harl/
│   └── envs/
│       └── mapush/
│           ├── __init__.py
│           ├── mapush_env.py         # Environment wrapper
│           └── mapush_logger.py      # Custom logger for success/collision rate
├── harl_mapush/                      # NEW: Custom scripts for MAPush
│   ├── train.py                      # Modified HARL training script
│   ├── test.py                       # NEW: Testing script (viewer + calc modes)
│   ├── runners/
│   │   └── mapush_happo_runner.py   # Custom runner with 10M checkpoint saving
│   └── configs/
│       └── mapush_cuboid.yaml        # MAPush-specific config
└── results/
    └── mapush/
        └── cuboid/
            └── happo/
                └── <exp_name>/
                    └── seed-<seed>-<timestamp>/
                        ├── checkpoints/
                        │   ├── 10M/
                        │   │   ├── actor_agent0.pt
                        │   │   ├── actor_agent1.pt
                        │   │   └── critic_agent.pt
                        │   ├── 20M/
                        │   └── ...
                        ├── logs/
                        └── config.json
```

**Pros:**
- Clean separation between HARL core and MAPush-specific code
- Easy to maintain and update HARL independently
- Follows HARL's architecture patterns

**Cons:**
- Requires creating new folder structure
- Need to duplicate some HARL code

---

### Option B: In-Place Modification

**Philosophy:** Modify HARL's existing structure directly

**Changes:**
- Modify `harl/runners/on_policy_base_runner.py` to support step-based checkpointing
- Add test.py to `HARL/examples/`
- Add mapush env to `harl/envs/`

**Pros:**
- Everything in one place
- Minimal file duplication

**Cons:**
- Harder to update HARL from upstream
- Mixes MAPush-specific logic with HARL core

---

## Recommended Approach: Option A with Extensions

I recommend **Option A** with the following implementation:

---

## Detailed Implementation Plan

### Phase 1: Environment Wrapper (1-2 days)

#### File: `HARL/harl/envs/mapush/mapush_env.py`

```python
"""MAPush environment wrapper for HARL."""
import torch
import numpy as np
from typing import Tuple, List, Optional
import sys
sys.path.append('/home/gvlab/new-universal-MAPush')

from mqe.envs.utils import make_mqe_env
from task.cuboid.config import Go1PushMidCfg


class MAPushEnv:
    """MAPush environment for HARL HAPPO training."""

    def __init__(self, env_args):
        """
        Args:
            env_args: dict with keys:
                - task: str (e.g., "go1push_mid")
                - n_threads: int (number of parallel environments)
        """
        self.env_args = env_args

        # Create MAPush environment
        from types import SimpleNamespace
        args = SimpleNamespace(
            task=env_args.get("task", "go1push_mid"),
            headless=True,
            num_envs=env_args.get("n_threads", 500),
            rl_device="cuda:0",
            sim_device="cuda:0",
            graphics_device_id=0,
        )

        # Use custom config for cuboid
        custom_config = Go1PushMidCfg

        self.env, self.env_cfg = make_mqe_env(args.task, args, custom_cfg=custom_config)

        self.n_envs = self.env.num_envs
        self.n_agents = self.env.num_agents

        # HARL expects list of spaces (one per agent)
        self.observation_space = [self.env.observation_space] * self.n_agents
        self.action_space = [self.env.action_space] * self.n_agents

        # Share observation space (for centralized critic) - use same as obs for now
        self.share_observation_space = [self.env.observation_space] * self.n_agents

        # Statistics tracking (for calculator mode)
        self.episode_rewards = []
        self.episode_lengths = []
        self.episode_success = []
        self.episode_collision = []
        self.episode_collaboration = []

    def step(self, actions: np.ndarray) -> Tuple:
        """
        Args:
            actions: [n_agents, n_envs, action_dim]

        Returns:
            obs: [n_envs, n_agents, obs_dim]
            state: [n_envs, n_agents, state_dim]
            rewards: [n_envs, n_agents, 1]
            dones: [n_envs, n_agents]
            infos: list of dicts
            available_actions: None
        """
        # Transpose: [n_agents, n_envs, action_dim] -> [n_envs, n_agents, action_dim]
        actions_transposed = actions.transpose(1, 0, 2)
        actions_torch = torch.from_numpy(actions_transposed).cuda()

        # Step environment
        obs, rewards, dones, infos = self.env.step(actions_torch)

        # Convert to numpy
        obs_np = obs.cpu().numpy()  # [n_envs, n_agents, obs_dim]
        rewards_np = rewards.cpu().numpy()  # [n_envs, n_agents]
        dones_np = dones.cpu().numpy()  # [n_envs]

        # State (same as obs for now, can be customized)
        state_np = obs_np.copy()

        # Reshape rewards: [n_envs, n_agents] -> [n_envs, n_agents, 1]
        rewards_np = rewards_np[..., np.newaxis]

        # Dones - broadcast to all agents
        dones_np = np.broadcast_to(dones_np[:, np.newaxis], (self.n_envs, self.n_agents))

        # Infos
        infos_list = [{} for _ in range(self.n_envs)]

        # Track statistics for episodes that just finished
        for env_idx in range(self.n_envs):
            if dones_np[env_idx, 0]:  # Episode done
                # Extract statistics from environment
                if hasattr(self.env, 'finished_buf'):
                    success = self.env.finished_buf[env_idx].item()
                    self.episode_success.append(success)

                if hasattr(self.env, 'episode_length_buf'):
                    length = self.env.episode_length_buf[env_idx].item()
                    self.episode_lengths.append(length)

                if hasattr(self.env, 'collision_degree_buf'):
                    collision = self.env.collision_degree_buf[env_idx].item()
                    self.episode_collision.append(collision)

                if hasattr(self.env, 'collaboration_degree_buf'):
                    collab = self.env.collaboration_degree_buf[env_idx].item()
                    self.episode_collaboration.append(collab)

        return obs_np, state_np, rewards_np, dones_np, infos_list, None

    def reset(self) -> Tuple:
        """
        Returns:
            obs: [n_envs, n_agents, obs_dim]
            state: [n_envs, n_agents, state_dim]
            available_actions: None
        """
        obs = self.env.reset()
        obs_np = obs.cpu().numpy()
        state_np = obs_np.copy()
        return obs_np, state_np, None

    def seed(self, seed: int):
        """Set random seed."""
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        np.random.seed(seed)

    def close(self):
        """Close the environment."""
        if hasattr(self.env, 'close'):
            self.env.close()

    def get_statistics(self):
        """Get accumulated statistics."""
        stats = {
            'success_rate': np.mean(self.episode_success) if self.episode_success else 0.0,
            'collision_rate': np.mean(self.episode_collision) if self.episode_collision else 0.0,
            'avg_episode_length': np.mean(self.episode_lengths) if self.episode_lengths else 0.0,
            'collaboration_degree': np.mean(self.episode_collaboration) if self.episode_collaboration else 0.0,
        }
        return stats

    def reset_statistics(self):
        """Reset statistics buffers."""
        self.episode_success = []
        self.episode_collision = []
        self.episode_lengths = []
        self.episode_collaboration = []
```

#### File: `HARL/harl/envs/mapush/mapush_logger.py`

```python
"""MAPush logger for HARL."""
from harl.common.base_logger import BaseLogger
import numpy as np


class MAPushLogger(BaseLogger):
    """Logger for MAPush environment."""

    def get_task_name(self):
        """Get task name."""
        return self.env_args.get("task", "cuboid")

    def episode_log(self, actor_train_infos, critic_train_info, buffer, *args):
        """Log episode information."""
        super().episode_log(actor_train_infos, critic_train_info, buffer, *args)

        # Add MAPush-specific logging
        # Access environment statistics if available
        if hasattr(self.envs, 'get_statistics'):
            stats = self.envs.get_statistics()
            self.writter.add_scalar('mapush/success_rate', stats['success_rate'], self.total_num_steps)
            self.writter.add_scalar('mapush/collision_rate', stats['collision_rate'], self.total_num_steps)
            self.writter.add_scalar('mapush/avg_episode_length', stats['avg_episode_length'], self.total_num_steps)
            self.writter.add_scalar('mapush/collaboration_degree', stats['collaboration_degree'], self.total_num_steps)
```

#### File: `HARL/harl/envs/mapush/__init__.py`

```python
"""MAPush environment for HARL."""
from harl.envs.mapush.mapush_env import MAPushEnv
from harl.envs.mapush.mapush_logger import MAPushLogger

__all__ = ["MAPushEnv", "MAPushLogger"]
```

---

### Phase 2: Custom Runner with 10M Checkpointing (2-3 days)

#### File: `HARL/harl_mapush/runners/mapush_happo_runner.py`

```python
"""Custom HAPPO runner for MAPush with step-based checkpointing."""
import os
import torch
import numpy as np
from harl.runners.on_policy_ha_runner import OnPolicyHARunner


class MAPushHAPPORunner(OnPolicyHARunner):
    """HAPPO runner with MAPush-specific checkpointing."""

    def __init__(self, args, algo_args, env_args):
        super().__init__(args, algo_args, env_args)

        # Checkpoint configuration
        self.checkpoint_interval = 10_000_000  # 10M steps
        self.last_checkpoint_step = 0

        # Create checkpoints directory
        self.checkpoints_dir = os.path.join(self.run_dir, "checkpoints")
        os.makedirs(self.checkpoints_dir, exist_ok=True)

        # Track total steps
        self.total_steps = 0

    def run(self):
        """Run training with step-based checkpointing."""
        self.warmup()

        start = self.start_episode
        episodes = (
            int(self.algo_args["train"]["num_env_steps"])
            // self.algo_args["train"]["episode_length"]
            // self.algo_args["train"]["n_rollout_threads"]
        )

        for episode in range(start, episodes):
            if self.algo_args["train"]["use_linear_lr_decay"]:
                self.actor[0].lr_decay(episode, episodes)
                if not self.share_param:
                    for agent_id in range(1, self.num_agents):
                        self.actor[agent_id].lr_decay(episode, episodes)
                self.critic.lr_decay(episode, episodes)

            self.logger.episode_init(episode)

            self.prep_rollout()
            for step in range(self.algo_args["train"]["episode_length"]):
                # Collect actions, obs, rewards
                (
                    values,
                    actions,
                    action_log_probs,
                    rnn_states,
                    rnn_states_critic,
                ) = self.collect(step)

                (
                    obs,
                    share_obs,
                    rewards,
                    dones,
                    infos,
                    available_actions,
                ) = self.envs.step(actions)

                data = (
                    obs,
                    share_obs,
                    rewards,
                    dones,
                    infos,
                    available_actions,
                    values,
                    actions,
                    action_log_probs,
                    rnn_states,
                    rnn_states_critic,
                )
                self.insert(data)

                # Update total steps
                self.total_steps += self.algo_args["train"]["n_rollout_threads"]

                # Check if we should save checkpoint
                if self.total_steps - self.last_checkpoint_step >= self.checkpoint_interval:
                    self.save_checkpoint(self.total_steps)
                    self.last_checkpoint_step = self.total_steps

            self.compute()

            self.prep_training()

            actor_train_infos, critic_train_info = self.train()

            # Log
            if episode % self.algo_args["train"]["log_interval"] == 0:
                self.logger.episode_log(
                    actor_train_infos,
                    critic_train_info,
                    self.actor_buffer,
                    self.critic_buffer,
                )

            # Eval
            if episode % self.algo_args["train"]["eval_interval"] == 0:
                if self.algo_args["eval"]["use_eval"]:
                    self.prep_rollout()
                    self.eval()

            self.after_update()

        # Save final checkpoint
        self.save_checkpoint(self.total_steps)

    def save_checkpoint(self, steps):
        """Save checkpoint at specific step count."""
        # Create checkpoint directory
        checkpoint_name = f"{steps // 1_000_000}M"
        checkpoint_path = os.path.join(self.checkpoints_dir, checkpoint_name)
        os.makedirs(checkpoint_path, exist_ok=True)

        # Save actor models
        for agent_id in range(self.num_agents):
            policy_actor = self.actor[agent_id].actor
            torch.save(
                policy_actor.state_dict(),
                os.path.join(checkpoint_path, f"actor_agent{agent_id}.pt"),
            )

        # Save critic model
        policy_critic = self.critic.critic
        torch.save(
            policy_critic.state_dict(),
            os.path.join(checkpoint_path, "critic_agent.pt"),
        )

        # Save value normalizer if exists
        if self.value_normalizer is not None:
            torch.save(
                self.value_normalizer.state_dict(),
                os.path.join(checkpoint_path, "value_normalizer.pt"),
            )

        print(f"[Checkpoint] Saved models at {steps} steps to {checkpoint_path}")
```

---

### Phase 3: Training Script (1 day)

#### File: `HARL/harl_mapush/train.py`

```python
"""Training script for MAPush with HAPPO."""
import argparse
import sys
sys.path.append('/home/gvlab/new-universal-MAPush')
sys.path.append('/home/gvlab/new-universal-MAPush/HARL')

from harl.utils.configs_tools import get_defaults_yaml_args, update_args


def main():
    """Main training function."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--algo", type=str, default="happo", choices=["happo"])
    parser.add_argument("--env", type=str, default="mapush")
    parser.add_argument("--exp_name", type=str, default="cuboid_happo")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--n_rollout_threads", type=int, default=500)
    parser.add_argument("--num_env_steps", type=int, default=100_000_000)

    args, unparsed_args = parser.parse_known_args()

    # Process unparsed args
    def process(arg):
        try:
            return eval(arg)
        except:
            return arg

    keys = [k[2:] for k in unparsed_args[0::2]]
    values = [process(v) for v in unparsed_args[1::2]]
    unparsed_dict = {k: v for k, v in zip(keys, values)}
    args = vars(args)

    # Load configs
    algo_args, _ = get_defaults_yaml_args(args["algo"], "pettingzoo_mpe")  # Use as base

    # Override with MAPush-specific config
    env_args = {
        "task": "go1push_mid",
        "n_threads": args.get("n_rollout_threads", 500),
        "state_type": "EP",  # Environment Provided state
    }

    # Update from command line
    update_args(unparsed_dict, algo_args, env_args)

    # Import Isaac Gym before PyTorch
    import isaacgym

    # Start training with custom runner
    from harl_mapush.runners.mapush_happo_runner import MAPushHAPPORunner

    runner = MAPushHAPPORunner(args, algo_args, env_args)
    runner.run()
    runner.close()


if __name__ == "__main__":
    main()
```

---

### Phase 4: Testing Script (2-3 days)

#### File: `HARL/harl_mapush/test.py`

```python
"""Testing script for MAPush with HAPPO."""
import argparse
import torch
import numpy as np
import sys
import os
sys.path.append('/home/gvlab/new-universal-MAPush')
sys.path.append('/home/gvlab/new-universal-MAPush/HARL')

import isaacgym
from harl.envs.mapush.mapush_env import MAPushEnv
from harl.algorithms.actors.happo import HAPPO


def load_models(checkpoint_dir, n_agents, obs_space, act_space, device="cuda"):
    """Load actor models from checkpoint."""
    actors = []

    for agent_id in range(n_agents):
        # Create actor
        actor_args = {
            "hidden_sizes": [128, 128],
            "activation_func": "relu",
            "use_feature_normalization": True,
            "initialization_method": "orthogonal_",
            "gain": 0.01,
            "use_naive_recurrent_policy": False,
            "use_recurrent_policy": False,
            "recurrent_n": 1,
            "data_chunk_length": 10,
            "lr": 0.0005,
            "opti_eps": 1e-5,
            "weight_decay": 0,
            "std_x_coef": 1,
            "std_y_coef": 0.5,
            "clip_param": 0.2,
            "ppo_epoch": 5,
            "actor_num_mini_batch": 1,
            "entropy_coef": 0.01,
            "use_max_grad_norm": True,
            "max_grad_norm": 10.0,
            "use_gae": True,
            "gamma": 0.99,
            "gae_lambda": 0.95,
            "use_huber_loss": True,
            "use_policy_active_masks": True,
            "huber_delta": 10.0,
            "action_aggregation": "prod",
        }

        actor = HAPPO(actor_args, obs_space[agent_id], act_space[agent_id], device=device)

        # Load weights
        model_path = os.path.join(checkpoint_dir, f"actor_agent{agent_id}.pt")
        actor.actor.load_state_dict(torch.load(model_path))
        actor.actor.eval()

        actors.append(actor)

    return actors


def test_calculator_mode(actors, env, num_episodes, seed):
    """Run calculator mode to compute statistics."""
    print(f"\n{'='*60}")
    print(f"Calculator Mode - Running {num_episodes} episodes")
    print(f"{'='*60}\n")

    env.seed(seed)
    env.reset_statistics()

    n_agents = env.n_agents
    n_envs = env.n_envs
    recurrent_n = 1
    rnn_hidden_size = 128

    episodes_completed = 0

    obs, _, _ = env.reset()
    rnn_states = np.zeros((n_envs, n_agents, recurrent_n, rnn_hidden_size), dtype=np.float32)
    masks = np.ones((n_envs, n_agents, 1), dtype=np.float32)

    while episodes_completed < num_episodes:
        # Collect actions from all actors
        actions_collector = []

        for agent_id in range(n_agents):
            action, rnn_state = actors[agent_id].act(
                obs[:, agent_id],
                rnn_states[:, agent_id],
                masks[:, agent_id],
                None,
                deterministic=True,
            )
            actions_collector.append(action.cpu().numpy())
            rnn_states[:, agent_id] = rnn_state.cpu().numpy()

        # Stack actions: (n_agents, n_envs, action_dim)
        actions = np.array(actions_collector)

        # Step environment
        obs, _, rewards, dones, infos, _ = env.step(actions)

        # Reset RNN states for done environments
        dones_env = np.all(dones, axis=1)
        rnn_states[dones_env] = np.zeros(
            ((dones_env).sum(), n_agents, recurrent_n, rnn_hidden_size),
            dtype=np.float32
        )

        masks = np.ones((n_envs, n_agents, 1), dtype=np.float32)
        masks[dones_env] = np.zeros(((dones_env).sum(), n_agents, 1), dtype=np.float32)

        # Count episodes
        episodes_completed += dones_env.sum()

        if episodes_completed >= num_episodes:
            break

    # Get statistics
    stats = env.get_statistics()

    print(f"\n{'='*60}")
    print(f"Statistics (over {num_episodes} episodes):")
    print(f"{'='*60}")
    print(f"Success Rate:         {stats['success_rate']:.4f}")
    print(f"Collision Rate:       {stats['collision_rate']:.4f}")
    print(f"Avg Episode Length:   {stats['avg_episode_length']:.2f}")
    print(f"Collaboration Degree: {stats['collaboration_degree']:.4f}")
    print(f"{'='*60}\n")


def test_viewer_mode(actors, env_args, num_episodes, seed, record_video=False):
    """Run viewer mode to show episodes sequentially."""
    print(f"\n{'='*60}")
    print(f"Viewer Mode - Showing {num_episodes} episodes")
    print(f"{'='*60}\n")

    # Create single-env version for visualization
    env_args_single = env_args.copy()
    env_args_single["n_threads"] = 1

    from types import SimpleNamespace
    from mqe.envs.utils import make_mqe_env
    from task.cuboid.config import Go1PushMidCfg

    args = SimpleNamespace(
        task="go1push_mid",
        headless=False,  # Enable rendering
        num_envs=1,
        rl_device="cuda:0",
        sim_device="cuda:0",
        graphics_device_id=0,
    )

    env_raw, _ = make_mqe_env(args.task, args, custom_cfg=Go1PushMidCfg)

    n_agents = env_raw.num_agents
    recurrent_n = 1
    rnn_hidden_size = 128

    for episode_idx in range(num_episodes):
        print(f"\nEpisode {episode_idx + 1}/{num_episodes}")

        obs = env_raw.reset()
        obs_np = obs.cpu().numpy()

        rnn_states = np.zeros((1, n_agents, recurrent_n, rnn_hidden_size), dtype=np.float32)
        masks = np.ones((1, n_agents, 1), dtype=np.float32)

        done = False
        step_count = 0

        while not done:
            # Collect actions
            actions_collector = []

            for agent_id in range(n_agents):
                obs_agent = obs_np.reshape(1, n_agents, -1)[:, agent_id]
                action, rnn_state = actors[agent_id].act(
                    obs_agent,
                    rnn_states[:, agent_id],
                    masks[:, agent_id],
                    None,
                    deterministic=True,
                )
                actions_collector.append(action.cpu().numpy())
                rnn_states[:, agent_id] = rnn_state.cpu().numpy()

            actions = torch.from_numpy(
                np.array(actions_collector).transpose(1, 0, 2)[0]
            ).cuda()

            # Step
            obs, rewards, dones_raw, infos = env_raw.step(actions)
            obs_np = obs.cpu().numpy()

            done = dones_raw.cpu().item()
            step_count += 1

        # Check success
        if hasattr(env_raw, 'finished_buf'):
            success = env_raw.finished_buf[0].item()
            print(f"  Result: {'SUCCESS' if success else 'FAILED'}")
            print(f"  Steps: {step_count}")

    env_raw.close()
    print(f"\n{'='*60}\n")


def main():
    """Main testing function."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to checkpoint folder (e.g., HARL/results/.../checkpoints/10M)")
    parser.add_argument("--mode", type=str, default="calculator", choices=["calculator", "viewer"])
    parser.add_argument("--num_episodes", type=int, default=10, help="Number of episodes (viewer mode)")
    parser.add_argument("--num_envs", type=int, default=300, help="Number of parallel envs (calculator mode)")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--record_video", action="store_true")

    args = parser.parse_args()

    # Environment config
    env_args = {
        "task": "go1push_mid",
        "n_threads": args.num_envs if args.mode == "calculator" else 1,
    }

    if args.mode == "calculator":
        # Create environment
        env = MAPushEnv(env_args)

        # Load models
        actors = load_models(
            args.checkpoint,
            env.n_agents,
            env.observation_space,
            env.action_space,
        )

        # Run calculator mode
        test_calculator_mode(actors, env, args.num_episodes, args.seed)

        env.close()

    else:  # viewer mode
        # Load models first (using dummy env to get spaces)
        env_dummy = MAPushEnv({"task": "go1push_mid", "n_threads": 1})
        actors = load_models(
            args.checkpoint,
            env_dummy.n_agents,
            env_dummy.observation_space,
            env_dummy.action_space,
        )
        env_dummy.close()

        # Run viewer mode
        test_viewer_mode(actors, env_args, args.num_episodes, args.seed, args.record_video)


if __name__ == "__main__":
    main()
```

---

### Phase 5: Registration and Configuration (1 day)

#### File: `HARL/harl/utils/envs_tools.py` (Modify)

Add to `make_train_env()`:

```python
elif env_name == "mapush":
    from harl.envs.mapush.mapush_env import MAPushEnv
    return MAPushEnv({"n_threads": n_threads, **env_args})
```

Add to `get_num_agents()`:

```python
elif env == "mapush":
    return envs.n_agents
```

#### File: `HARL/harl/envs/__init__.py` (Modify)

```python
from harl.envs.mapush.mapush_logger import MAPushLogger

LOGGER_REGISTRY = {
    ...
    "mapush": MAPushLogger,
}
```

#### File: `HARL/harl_mapush/configs/mapush_cuboid.yaml`

```yaml
# MAPush cuboid task configuration
task: go1push_mid
n_threads: 500
state_type: "EP"  # Environment Provided state (all agents see same global state)
```

---

## Usage Examples

### Training

```bash
cd /home/gvlab/new-universal-MAPush/HARL/harl_mapush

# Basic training
python train.py --exp_name my_experiment --seed 1

# Custom configuration
python train.py \
    --exp_name my_experiment \
    --seed 1 \
    --n_rollout_threads 500 \
    --num_env_steps 100000000
```

**Output:**
```
HARL/results/mapush/cuboid/happo/my_experiment/seed-00001-<timestamp>/
├── checkpoints/
│   ├── 10M/
│   │   ├── actor_agent0.pt
│   │   ├── actor_agent1.pt
│   │   └── critic_agent.pt
│   ├── 20M/
│   ├── 30M/
│   └── ...
├── logs/
│   └── events.out.tfevents...
└── config.json
```

### Testing - Calculator Mode

```bash
python test.py \
    --checkpoint /home/gvlab/new-universal-MAPush/HARL/results/mapush/cuboid/happo/my_experiment/seed-00001-.../checkpoints/50M \
    --mode calculator \
    --num_envs 300 \
    --seed 1
```

**Output:**
```
============================================================
Calculator Mode - Running 300 episodes
============================================================

============================================================
Statistics (over 300 episodes):
============================================================
Success Rate:         0.8567
Collision Rate:       0.1234
Avg Episode Length:   156.23
Collaboration Degree: 0.7891
============================================================
```

### Testing - Viewer Mode

```bash
python test.py \
    --checkpoint .../checkpoints/50M \
    --mode viewer \
    --num_episodes 5 \
    --seed 1
```

**Output:**
```
============================================================
Viewer Mode - Showing 5 episodes
============================================================

Episode 1/5
  Result: SUCCESS
  Steps: 142

Episode 2/5
  Result: FAILED
  Steps: 200

...
```

---

## Implementation Timeline

| Phase | Task | Duration | Files Created/Modified |
|-------|------|----------|------------------------|
| **1** | Environment Wrapper | 1-2 days | `mapush_env.py`, `mapush_logger.py`, `__init__.py` |
| **2** | Custom Runner | 2-3 days | `mapush_happo_runner.py` |
| **3** | Training Script | 1 day | `train.py` |
| **4** | Testing Script | 2-3 days | `test.py` |
| **5** | Registration & Config | 1 day | Modify HARL files, create configs |

**Total:** 7-10 days

---

## Advantages of This Approach

1. ✅ **Clean separation:** MAPush-specific code in `harl_mapush/`, doesn't pollute HARL core
2. ✅ **Checkpoint compatibility:** Saves in format you specified (10M, 20M, etc.)
3. ✅ **Testing flexibility:** Separate test.py with viewer/calculator modes
4. ✅ **Statistics tracking:** Same metrics as MAPush (success rate, collision rate, etc.)
5. ✅ **Easy to maintain:** Can update HARL independently
6. ✅ **Extensible:** Easy to add more algorithms later (HATRPO, HAA2C, etc.)

---

## Alternative: Simpler Approach (If Tight on Time)

If you need something faster, I can propose a **minimal integration** that:
- Skips custom runner, modifies HARL's base runner directly
- Uses HARL's existing save mechanism with wrapper to rename folders
- Creates simpler test.py

Would take ~3-5 days but less clean.

---

## Questions to Resolve

1. **State representation:** Should we use EP (same global state for all) or FP (different states per agent)?
   - Recommendation: EP for now (simpler)

2. **Value normalizer:** Should we save it in checkpoints?
   - Recommendation: Yes (already included in code above)

3. **Checkpoint on failure:** Should we save checkpoint if training crashes?
   - Recommendation: Yes, add exception handling

4. **TensorBoard logs:** Keep HARL's default logging or customize?
   - Recommendation: Keep default + add MAPush metrics

---

## Next Steps

1. **Review this proposal** - Let me know if architecture works for you
2. **Clarify requirements** - Any changes to what I've proposed?
3. **Start implementation** - I can begin with Phase 1 (environment wrapper)

Would you like me to proceed with this plan, or would you prefer modifications?
