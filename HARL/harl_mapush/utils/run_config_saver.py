"""
Run Configuration Saver for MAPush Training

Saves a complete configuration snapshot at training start, including:
- Command used to launch training
- Agent types and heterogeneous configuration
- Algorithm hyperparameters
- Environment configuration (box mass, spawn settings, etc.)
- Reward configuration

This allows for:
1. Full reproducibility (copy-paste command)
2. Quick reference for what a run used
3. Safety check when testing (verify agent types match)
"""

import os
import sys
import yaml
import json
from datetime import datetime
from typing import Dict, Any, Optional


def save_run_config(
    run_dir: str,
    args: Dict[str, Any],
    algo_args: Dict[str, Any],
    env_args: Dict[str, Any],
    env_cfg: Optional[Any] = None,
    command_args: Optional[list] = None
):
    """
    Save complete run configuration to the run directory.

    Creates two files:
    - command.txt: Exact command to reproduce the run
    - run_config.yaml: Complete configuration snapshot

    Args:
        run_dir: Directory where training results are saved
        args: Main arguments (algo, env, exp_name, etc.)
        algo_args: Algorithm hyperparameters
        env_args: Environment arguments passed to MAPushEnv
        env_cfg: Environment configuration class (Go1PushMidCfg or similar)
        command_args: sys.argv or equivalent command line arguments
    """
    os.makedirs(run_dir, exist_ok=True)

    # Get timestamp
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # Reconstruct command if not provided
    if command_args is None:
        command_args = sys.argv

    command_str = " ".join(command_args)

    # =========================================================================
    # Save command.txt
    # =========================================================================
    command_file = os.path.join(run_dir, "command.txt")
    with open(command_file, "w") as f:
        f.write(f"# Training command - copy-paste to reproduce\n")
        f.write(f"# Generated: {timestamp}\n")
        f.write(f"# Run directory: {run_dir}\n")
        f.write(f"\n")
        f.write(f"{command_str}\n")
        f.write(f"\n")
        f.write(f"# Testing command template (update checkpoint path):\n")

        # Generate test command based on training args
        hetero_agent = env_args.get("hetero_agent")
        if hetero_agent:
            f.write(f"# python HARL/harl_mapush/test.py \\\n")
            f.write(f"#   --checkpoint {run_dir}/checkpoints/LATEST \\\n")
            f.write(f"#   --hetero_agent {hetero_agent} \\\n")
            f.write(f"#   --mode viewer \\\n")
            f.write(f"#   --num_episodes 5\n")
        else:
            f.write(f"# python HARL/harl_mapush/test.py \\\n")
            f.write(f"#   --checkpoint {run_dir}/checkpoints/LATEST \\\n")
            f.write(f"#   --mode viewer \\\n")
            f.write(f"#   --num_episodes 5\n")

    # =========================================================================
    # Build run_config.yaml
    # =========================================================================
    config = {}

    # Header
    config["_meta"] = {
        "generated": timestamp,
        "command": command_str,
        "run_dir": run_dir,
    }

    # -------------------------------------------------------------------------
    # Agents
    # -------------------------------------------------------------------------
    hetero_agent = env_args.get("hetero_agent")
    if hetero_agent:
        config["agents"] = {
            "agent0": "go1",
            "agent1": hetero_agent,
            "is_heterogeneous": True,
        }
    else:
        config["agents"] = {
            "agent0": "go1",
            "agent1": "go1",
            "is_heterogeneous": False,
        }

    # -------------------------------------------------------------------------
    # Algorithm
    # -------------------------------------------------------------------------
    config["algorithm"] = {
        "name": args.get("algo", "happo"),
        "seed": algo_args.get("seed", {}).get("seed", 1),
    }

    # Extract training params
    train_args = algo_args.get("train", {})
    config["training"] = {
        "num_env_steps": train_args.get("num_env_steps"),
        "n_rollout_threads": train_args.get("n_rollout_threads"),
        "episode_length": train_args.get("episode_length"),
        "n_eval_rollout_threads": train_args.get("n_eval_rollout_threads"),
        "eval_interval": train_args.get("eval_interval"),
        "log_interval": train_args.get("log_interval"),
    }

    # Extract model/network params
    model_args = algo_args.get("model", {})
    config["network"] = {
        "hidden_sizes": model_args.get("hidden_sizes"),
        "activation": model_args.get("activation"),
        "use_recurrent_policy": model_args.get("use_recurrent_policy"),
        "recurrent_n": model_args.get("recurrent_n"),
        "use_naive_recurrent_policy": model_args.get("use_naive_recurrent_policy"),
    }

    # Extract algo-specific params
    algo_specific = algo_args.get("algo", {})
    config["algorithm"].update({
        "lr": algo_specific.get("lr"),
        "critic_lr": algo_specific.get("critic_lr"),
        "gamma": algo_specific.get("gamma"),
        "gae_lambda": algo_specific.get("gae_lambda"),
        "clip_param": algo_specific.get("clip_param"),
        "ppo_epoch": algo_specific.get("ppo_epoch"),
        "num_mini_batch": algo_specific.get("num_mini_batch"),
        "entropy_coef": algo_specific.get("entropy_coef"),
        "value_loss_coef": algo_specific.get("value_loss_coef"),
        "max_grad_norm": algo_specific.get("max_grad_norm"),
        "use_max_grad_norm": algo_specific.get("use_max_grad_norm"),
    })

    # -------------------------------------------------------------------------
    # Environment Args (from command line flags)
    # -------------------------------------------------------------------------
    config["env_args"] = {
        "task": env_args.get("task", "go1push_mid"),
        "state_type": env_args.get("state_type"),
        "individualized_rewards": env_args.get("individualized_rewards", False),
        "shared_gated_rewards": env_args.get("shared_gated_rewards", False),
        "cooperation_rewards": env_args.get("cooperation_rewards", False),
        "collaboration_rewards": env_args.get("collaboration_rewards", False),
        "mapush_og_rewards_teamified": env_args.get("mapush_og_rewards_teamified", False),
        "positive_approachtobox_reward": env_args.get("positive_approachtobox_reward", False),
        "use_goal_centered_critic": env_args.get("use_goal_centered_critic", False),
        "use_box_centered_critic": env_args.get("use_box_centered_critic", False),
        "use_concat_agent_observations_critic": env_args.get("use_concat_agent_observations_critic", False),
        "use_relative_obs_critic": env_args.get("use_relative_obs_critic", False),
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
        f.write("# MAPush Training Configuration\n")
        f.write(f"# Auto-generated at training start - DO NOT EDIT\n")
        f.write(f"# Timestamp: {timestamp}\n")
        f.write(f"# Command: {command_str}\n")
        f.write("\n")
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
        config["box"] = {
            "mass_override_kg": getattr(asset, "npc_mass_override", None),
            "urdf_default_mass_kg": 4.0,  # From SmallBox.urdf
            "dimensions_m": [1.2, 1.2, 0.5],  # From SmallBox.urdf
            "npc_collision": getattr(asset, "npc_collision", True),
            "npc_gravity": getattr(asset, "npc_gravity", True),
        }
        # Add effective mass
        mass_override = getattr(asset, "npc_mass_override", None)
        config["box"]["effective_mass_kg"] = mass_override if mass_override is not None else 4.0

    # -------------------------------------------------------------------------
    # Spawn configuration
    # -------------------------------------------------------------------------
    if hasattr(env_cfg, 'domain_rand'):
        domain_rand = env_cfg.domain_rand

        spawn_config = {}

        # Angular separation
        spawn_config["min_agent_angular_separation_rad"] = getattr(
            domain_rand, "min_agent_angular_separation", 1.57
        )

        # Position range
        if hasattr(domain_rand, "init_base_pos_range"):
            pos_range = domain_rand.init_base_pos_range
            spawn_config["init_base_pos_range_r"] = pos_range.get("r", [1.2, 1.3])
            spawn_config["init_base_pos_range_theta"] = pos_range.get("theta", [0, 6.28])

        # Friction
        spawn_config["friction_range"] = getattr(domain_rand, "friction_range", [0.5, 0.6])

        config["spawn"] = spawn_config

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
    # Rewards configuration
    # -------------------------------------------------------------------------
    if hasattr(env_cfg, 'rewards'):
        rewards = env_cfg.rewards

        reward_config = {
            "individualized_rewards": getattr(rewards, "individualized_rewards", False),
            "contact_threshold": getattr(rewards, "contact_threshold", 0.8),
        }

        # Extract scales
        if hasattr(rewards, 'scales'):
            scales = rewards.scales
            reward_config["scales"] = {
                "target_reward": getattr(scales, "target_reward_scale", None),
                "approach_reward": getattr(scales, "approach_reward_scale", None),
                "collision_punishment": getattr(scales, "collision_punishment_scale", None),
                "push_reward": getattr(scales, "push_reward_scale", None),
                "ocb_reward": getattr(scales, "ocb_reward_scale", None),
                "reach_target_reward": getattr(scales, "reach_target_reward_scale", None),
                "exception_punishment": getattr(scales, "exception_punishment_scale", None),
            }

        config["rewards"] = reward_config

    # -------------------------------------------------------------------------
    # Termination configuration
    # -------------------------------------------------------------------------
    if hasattr(env_cfg, 'termination'):
        termination = env_cfg.termination
        config["termination"] = {
            "terms": getattr(termination, "termination_terms", []),
        }
        if hasattr(termination, "z_wave_kwargs"):
            config["termination"]["z_wave_threshold"] = termination.z_wave_kwargs.get("threshold")
        if hasattr(termination, "collision_kwargs"):
            config["termination"]["collision_threshold"] = termination.collision_kwargs.get("threshold")

    # -------------------------------------------------------------------------
    # Goal configuration
    # -------------------------------------------------------------------------
    if hasattr(env_cfg, 'goal'):
        goal = env_cfg.goal
        config["goal"] = {
            "random_goal_pos": getattr(goal, "random_goal_pos", True),
            "random_goal_distance_from_init": getattr(goal, "random_goal_distance_from_init", None),
            "threshold": getattr(goal, "THRESHOLD", 1.0),
        }

    return config


def load_run_config(run_dir: str) -> Optional[Dict[str, Any]]:
    """
    Load run configuration from a checkpoint directory.

    Args:
        run_dir: Directory containing run_config.yaml

    Returns:
        Configuration dictionary or None if not found
    """
    config_file = os.path.join(run_dir, "run_config.yaml")

    if not os.path.exists(config_file):
        # Try parent directory (if run_dir points to checkpoints/10M/)
        parent_config = os.path.join(os.path.dirname(os.path.dirname(run_dir)), "run_config.yaml")
        if os.path.exists(parent_config):
            config_file = parent_config
        else:
            return None

    with open(config_file, "r") as f:
        config = yaml.safe_load(f)

    return config


def print_config_summary(config: Dict[str, Any]):
    """Print a human-readable summary of the configuration."""
    print("\n" + "=" * 60)
    print("Run Configuration Summary")
    print("=" * 60)

    if "_meta" in config:
        print(f"Generated: {config['_meta'].get('generated', 'Unknown')}")

    if "agents" in config:
        agents = config["agents"]
        print(f"\nAgents:")
        print(f"  Agent 0: {agents.get('agent0', 'Unknown')}")
        print(f"  Agent 1: {agents.get('agent1', 'Unknown')}")
        print(f"  Heterogeneous: {agents.get('is_heterogeneous', False)}")

    if "environment" in config and "box" in config["environment"]:
        box = config["environment"]["box"]
        print(f"\nBox:")
        print(f"  Effective mass: {box.get('effective_mass_kg', 4.0)} kg")

    if "training" in config:
        train = config["training"]
        print(f"\nTraining:")
        print(f"  Total steps: {train.get('num_env_steps', 'Unknown'):,}")
        print(f"  Parallel envs: {train.get('n_rollout_threads', 'Unknown')}")

    print("=" * 60 + "\n")
