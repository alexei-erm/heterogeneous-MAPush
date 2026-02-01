"""
Run Configuration Saver for MAPPO/OpenRL Training

Saves a complete configuration snapshot at training start, including:
- Command used to launch training
- Agent types and heterogeneous configuration
- Algorithm hyperparameters (hidden_size, layer_N, lr, etc.)
- Environment configuration

This allows for:
1. Full reproducibility (copy-paste command)
2. Quick reference for what a run used
3. Easy comparison between runs
"""

import os
import sys
import yaml
from datetime import datetime
from typing import Dict, Any, Optional


def save_run_config(
    run_dir: str,
    args,
    env_cfg: Optional[Any] = None,
):
    """
    Save complete run configuration to the run directory.

    Creates two files:
    - command.txt: Exact command to reproduce the run
    - run_config.yaml: Complete configuration snapshot

    Args:
        run_dir: Directory where training results are saved
        args: Parsed command-line arguments (argparse Namespace)
        env_cfg: Environment configuration class (Go1PushMidCfg or similar)
    """
    os.makedirs(run_dir, exist_ok=True)

    # Get timestamp
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # Get command
    command_str = " ".join(sys.argv)

    # =========================================================================
    # Save command.txt
    # =========================================================================
    command_file = os.path.join(run_dir, "command.txt")
    with open(command_file, "w") as f:
        f.write(f"# MAPPO Training command - copy-paste to reproduce\n")
        f.write(f"# Generated: {timestamp}\n")
        f.write(f"# Run directory: {run_dir}\n")
        f.write(f"\n")
        f.write(f"cd /home/gvlab/new-universal-MAPush\n")
        f.write(f"export PYTHONPATH=/home/gvlab/new-universal-MAPush:$PYTHONPATH\n")
        f.write(f"{command_str}\n")
        f.write(f"\n")
        f.write(f"# Testing command template (update checkpoint path):\n")

        # Generate test command based on training args
        agent0 = getattr(args, 'agent0', 'go1')
        agent1 = getattr(args, 'agent1', 'go1')
        f.write(f"# python ./openrl_ws/test.py \\\n")
        f.write(f"#   --checkpoint {run_dir}/checkpoints/rl_model_XXXXX_steps/module.pt \\\n")
        f.write(f"#   --agent0 {agent0} \\\n")
        f.write(f"#   --agent1 {agent1} \\\n")
        f.write(f"#   --algo {getattr(args, 'algo', 'ppo')} \\\n")
        f.write(f"#   --task {getattr(args, 'task', 'go1push_mid')} \\\n")
        f.write(f"#   --num_envs 1 \\\n")
        f.write(f"#   --test_mode viewer\n")

    # =========================================================================
    # Build run_config.yaml
    # =========================================================================
    config = {}

    # Header
    config["_meta"] = {
        "generated": timestamp,
        "command": command_str,
        "run_dir": run_dir,
        "framework": "MAPPO/OpenRL",
    }

    # -------------------------------------------------------------------------
    # Agents
    # -------------------------------------------------------------------------
    agent0 = getattr(args, 'agent0', 'go1')
    agent1 = getattr(args, 'agent1', 'go1')
    is_hetero = (agent0 != agent1)
    config["agents"] = {
        "agent0": agent0,
        "agent1": agent1,
        "is_heterogeneous": is_hetero,
    }

    # -------------------------------------------------------------------------
    # Algorithm & Network
    # -------------------------------------------------------------------------
    config["algorithm"] = {
        "name": getattr(args, 'algo', 'ppo'),
        "seed": getattr(args, 'seed', None),
    }

    config["network"] = {
        "hidden_size": getattr(args, 'hidden_size', 64),
        "layer_N": getattr(args, 'layer_N', 1),
        "use_recurrent_policy": getattr(args, 'use_recurrent_policy', False),
    }

    config["training"] = {
        "train_timesteps": getattr(args, 'train_timesteps', None),
        "num_envs": getattr(args, 'num_envs', None),
        "episode_length": getattr(args, 'episode_length', 200),
        "lr": getattr(args, 'lr', None),
        "critic_lr": getattr(args, 'critic_lr', None),
    }

    config["logging"] = {
        "use_wandb": getattr(args, 'use_wandb', False),
        "use_tensorboard": getattr(args, 'use_tensorboard', False),
        "exp_name": getattr(args, 'exp_name', 'default'),
    }

    config["rewards"] = {
        "baseline_mappo_rewards": getattr(args, 'baseline_mappo_rewards', True),
        "description": "baseline_mappo_rewards=True uses ONLY original 7 MAPush rewards with original scales",
    }

    # -------------------------------------------------------------------------
    # Environment Config (from Python config class)
    # -------------------------------------------------------------------------
    if env_cfg is not None:
        config["environment"] = extract_env_config(env_cfg)

    # -------------------------------------------------------------------------
    # Save YAML
    # -------------------------------------------------------------------------
    config_file = os.path.join(run_dir, "run_config.yaml")
    with open(config_file, "w") as f:
        f.write("# MAPPO/OpenRL Training Configuration\n")
        f.write(f"# Auto-generated at training start - DO NOT EDIT\n")
        f.write(f"# Timestamp: {timestamp}\n")
        f.write(f"\n")
        yaml.dump(config, f, default_flow_style=False, sort_keys=False, allow_unicode=True)

    print(f"[Run Config] Saved configuration to {run_dir}/")
    print(f"  - command.txt: Exact training command")
    print(f"  - run_config.yaml: Complete configuration snapshot")


def extract_env_config(env_cfg) -> Dict[str, Any]:
    """
    Extract relevant configuration from environment config class.

    Args:
        env_cfg: Environment configuration class (e.g., Go1PushMidCfg)

    Returns:
        Dictionary with extracted configuration
    """
    config = {}

    # -------------------------------------------------------------------------
    # Box configuration
    # -------------------------------------------------------------------------
    if hasattr(env_cfg, 'asset'):
        asset = env_cfg.asset
        mass_override = getattr(asset, "npc_mass_override", None)
        config["box"] = {
            "mass_override_kg": mass_override,
            "effective_mass_kg": mass_override if mass_override is not None else 4.0,
        }

    # -------------------------------------------------------------------------
    # Episode configuration
    # -------------------------------------------------------------------------
    if hasattr(env_cfg, 'env'):
        env = env_cfg.env
        config["episode"] = {
            "num_agents": getattr(env, "num_agents", 2),
            "episode_length_s": getattr(env, "episode_length_s", 20),
        }

    # -------------------------------------------------------------------------
    # Hetero configuration
    # -------------------------------------------------------------------------
    if hasattr(env_cfg, 'hetero'):
        hetero = env_cfg.hetero
        config["hetero"] = {
            "use_hetero": getattr(hetero, "use_hetero", False),
            "hetero_agent_types": getattr(hetero, "hetero_agent_types", ['go1', 'go1']),
        }

    # -------------------------------------------------------------------------
    # Rewards configuration
    # -------------------------------------------------------------------------
    if hasattr(env_cfg, 'rewards'):
        rewards = env_cfg.rewards
        config["rewards"] = {
            "individualized_rewards": getattr(rewards, "individualized_rewards", False),
        }

    return config


def load_run_config(run_dir: str) -> Optional[Dict[str, Any]]:
    """
    Load run configuration from a run directory.

    Args:
        run_dir: Directory containing run_config.yaml

    Returns:
        Configuration dictionary or None if not found
    """
    config_file = os.path.join(run_dir, "run_config.yaml")

    if not os.path.exists(config_file):
        return None

    with open(config_file, "r") as f:
        config = yaml.safe_load(f)

    return config


def print_config_comparison(run_dir1: str, run_dir2: str):
    """Print a side-by-side comparison of two run configurations."""
    config1 = load_run_config(run_dir1)
    config2 = load_run_config(run_dir2)

    if config1 is None:
        print(f"No config found in {run_dir1}")
        return
    if config2 is None:
        print(f"No config found in {run_dir2}")
        return

    print("\n" + "=" * 70)
    print(f"Configuration Comparison")
    print("=" * 70)
    print(f"{'Parameter':<30} {'Run 1':<18} {'Run 2':<18}")
    print("-" * 70)

    # Compare key parameters
    comparisons = [
        ("Agent 0", config1.get("agents", {}).get("agent0"), config2.get("agents", {}).get("agent0")),
        ("Agent 1", config1.get("agents", {}).get("agent1"), config2.get("agents", {}).get("agent1")),
        ("Heterogeneous", config1.get("agents", {}).get("is_heterogeneous"), config2.get("agents", {}).get("is_heterogeneous")),
        ("Algorithm", config1.get("algorithm", {}).get("name"), config2.get("algorithm", {}).get("name")),
        ("Seed", config1.get("algorithm", {}).get("seed"), config2.get("algorithm", {}).get("seed")),
        ("Hidden Size", config1.get("network", {}).get("hidden_size"), config2.get("network", {}).get("hidden_size")),
        ("Layer N", config1.get("network", {}).get("layer_N"), config2.get("network", {}).get("layer_N")),
        ("Train Timesteps", config1.get("training", {}).get("train_timesteps"), config2.get("training", {}).get("train_timesteps")),
        ("Num Envs", config1.get("training", {}).get("num_envs"), config2.get("training", {}).get("num_envs")),
        ("LR", config1.get("training", {}).get("lr"), config2.get("training", {}).get("lr")),
    ]

    for param, val1, val2 in comparisons:
        diff = " <--" if val1 != val2 else ""
        print(f"{param:<30} {str(val1):<18} {str(val2):<18}{diff}")

    print("=" * 70 + "\n")
