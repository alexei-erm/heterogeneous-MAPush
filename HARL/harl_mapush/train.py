"""Training script for MAPush with HAPPO.

This script trains HAPPO on the MAPush cuboid pushing task.
Models are saved every 10M steps in the format:
    HARL/results/mapush/cuboid/happo/<exp_name>/seed-<seed>-<timestamp>/checkpoints/
        ├── 10M/
        │   ├── actor_agent0.pt
        │   ├── actor_agent1.pt
        │   └── critic_agent.pt
        ├── 20M/
        └── ...
"""
import argparse
import sys
import os

# Add paths - INSERT at beginning to override any PYTHONPATH pollution
sys.path.insert(0, '/home/gvlab/new-universal-MAPush/HARL')
sys.path.insert(0, '/home/gvlab/new-universal-MAPush')

from harl.utils.configs_tools import get_defaults_yaml_args, update_args


def main():
    """Main training function."""
    parser = argparse.ArgumentParser(
        description="Train HAPPO on MAPush cuboid task",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    # Main arguments
    parser.add_argument("--algo", type=str, default="happo",
                       choices=["happo"],
                       help="Algorithm (only HAPPO supported for now)")
    parser.add_argument("--env", type=str, default="mapush",
                       help="Environment name")
    parser.add_argument("--exp_name", type=str, default="cuboid_happo",
                       help="Experiment name")
    parser.add_argument("--task", type=str, default="go1push_mid",
                       choices=["go1push_mid"],
                       help="MAPush task variant")
    parser.add_argument("--individualized_rewards", action="store_true", default=False,
                       help="Enable individualized rewards for HAPPO (prevents freeloading)")
    parser.add_argument("--seed", type=int, default=1,
                       help="Random seed")

    # Training arguments
    parser.add_argument("--n_rollout_threads", type=int, default=500,
                       help="Number of parallel environments")
    parser.add_argument("--num_env_steps", type=int, default=100_000_000,
                       help="Total number of environment steps")
    parser.add_argument("--episode_length", type=int, default=200,
                       help="Rollout length: steps to collect before update (MAPush episodes: 1000 steps, split into 5 rollouts)")

    # Optional arguments
    parser.add_argument("--use_tensorboard", action="store_true", default=True,
                       help="Use TensorBoard for logging")
    parser.add_argument("--checkpoint", type=str, default=None,
                       help="Path to checkpoint to resume from")

    args, unparsed_args = parser.parse_known_args()

    # Process unparsed arguments
    def process(arg):
        try:
            return eval(arg)
        except:
            return arg

    keys = [k[2:] for k in unparsed_args[0::2]]  # remove -- from argument
    values = [process(v) for v in unparsed_args[1::2]]
    unparsed_dict = {k: v for k, v in zip(keys, values)}
    args = vars(args)  # convert to dict

    # Load HAPPO config as base
    algo_args, _ = get_defaults_yaml_args(args["algo"], "pettingzoo_mpe")

    # Override with MAPush-specific environment config
    env_args = {
        "task": args.get("task", "go1push_mid"),  # Use task from command line
        "n_threads": args.get("n_rollout_threads", 500),
        "state_type": "EP",  # Environment Provided state (all agents see same global state)
        "individualized_rewards": args.get("individualized_rewards", False),  # For HAPPO
    }

    # Override training parameters
    algo_args["train"]["n_rollout_threads"] = args.get("n_rollout_threads", 500)
    algo_args["train"]["num_env_steps"] = args.get("num_env_steps", 100_000_000)
    algo_args["train"]["episode_length"] = args.get("episode_length", 200)

    # Set seed
    algo_args["seed"]["seed"] = args.get("seed", 1)
    algo_args["seed"]["seed_specify"] = True

    # Update from command line
    update_args(unparsed_dict, algo_args, env_args)

    # Print configuration
    print("\n" + "="*60)
    print("MAPush HAPPO Training Configuration")
    print("="*60)
    print(f"Algorithm: {args['algo']}")
    print(f"Environment: {env_args['task']}")
    print(f"Experiment: {args['exp_name']}")
    print(f"Individualized rewards: {env_args['individualized_rewards']}")
    print(f"Seed: {algo_args['seed']['seed']}")
    print(f"Parallel envs: {algo_args['train']['n_rollout_threads']}")
    print(f"Total steps: {algo_args['train']['num_env_steps']:,}")
    print(f"Rollout length: {algo_args['train']['episode_length']} steps/rollout")
    print(f"  (MAPush actual episodes: 1000 steps = 20s @ 50Hz)")
    print(f"  (Each episode = {1000 // algo_args['train']['episode_length']} rollouts)")
    print("="*60 + "\n")

    # Import Isaac Gym before PyTorch (required)
    import isaacgym

    # Import runner
    from harl_mapush.runners.mapush_happo_runner import MAPushHAPPORunner

    # Create and run
    runner = MAPushHAPPORunner(args, algo_args, env_args)

    # Restore from checkpoint if specified
    if args.get("checkpoint") is not None:
        if os.path.exists(args["checkpoint"]):
            runner.restore_checkpoint(args["checkpoint"])
        else:
            print(f"Warning: Checkpoint {args['checkpoint']} not found. Starting from scratch.")

    # Run training
    runner.run()

    # Close
    runner.close()

    print("\n" + "="*60)
    print("Training completed successfully!")
    print("="*60 + "\n")


if __name__ == "__main__":
    # Must run from MAPush root directory for relative paths to work
    if not os.path.exists("./resources/actuator_nets"):
        print("\n" + "="*60)
        print("ERROR: Must run from MAPush root directory!")
        print("="*60)
        print("\nPlease run:")
        print("  cd /home/gvlab/new-universal-MAPush")
        print("  conda run -n mapush python HARL/harl_mapush/train.py [args]")
        print("\n" + "="*60 + "\n")
        sys.exit(1)

    main()
