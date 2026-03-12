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
                       choices=["go1push_mid", "go1push_vel"],
                       help="MAPush task variant")
    parser.add_argument("--individualized_rewards", action="store_true", default=False,
                       help="Enable individualized rewards for HAPPO (prevents freeloading)")
    parser.add_argument("--shared_gated_rewards", action="store_true", default=False,
                       help="Iter8: Gate all shared rewards by min agent engagement (prevents freeloading)")
    parser.add_argument("--use_goal_centered_critic", type=lambda x: (str(x).lower() == 'true'), default=False,
                       help="Use goal-centered coordinates for critic (CRITIC16): Everything relative to goal (stationary reference frame). 9 dims: [box_rel(3), agent0_rel(3), agent1_rel(3)]. DEFAULT: False")
    parser.add_argument("--use_box_centered_critic", type=lambda x: (str(x).lower() == 'true'), default=False,
                       help="Use box-centered (relative) coordinates for critic (CRITIC9). Set to False for absolute coordinates (CRITIC7). DEFAULT: False (absolute)")
    parser.add_argument("--use_concat_agent_observations_critic", type=lambda x: (str(x).lower() == 'true'), default=False,
                       help="Use concatenated agent observations for critic (CRITIC10). DEFAULT: False")
    parser.add_argument("--use_relative_obs_critic", type=lambda x: (str(x).lower() == 'true'), default=False,
                       help="Use relative observations for critic (CRITIC11): [robot1_to_box, robot2_to_box, inter_robot_dist, goal_to_box]. Takes highest priority. DEFAULT: False")
    parser.add_argument("--cooperation_rewards", type=lambda x: (str(x).lower() == 'true'), default=False,
                       help="CRITIC12: Enable three-tier cooperation bonuses (dual_engagement, synchronized_contact, bilateral_push). DEFAULT: False")
    parser.add_argument("--mapush_og_rewards_teamified", type=lambda x: (str(x).lower() == 'true'), default=False,
                       help="Use original MAPush rewards (7 total) converted to team rewards. Disables goal_push_bonus, proximity_penalty. Uses symmetric OCB ±0.004, original collision scale -0.0025. DEFAULT: False")
    parser.add_argument("--reward_scale_testing", type=lambda x: (str(x).lower() == 'true'), default=False,
                       help="Use same 7 rewards as mapush_og_rewards_teamified but with heavy_cuboid Exp 7 scales: target=0.01(3x), push=0.004(2.5x), ocb=0.008(2x). Penalty term REMOVED. DEFAULT: False")
    parser.add_argument("--collaboration_rewards", type=lambda x: (str(x).lower() == 'true'), default=False,
                       help="CRITIC15 v4: Add dual pushing bonus. Rewards when both agents push toward goal simultaneously. Use with --mapush_og_rewards_teamified True. DEFAULT: False")
    parser.add_argument("--require_both_contact_for_success", type=lambda x: (str(x).lower() == 'true'), default=False,
                       help="COLLABORATION ENFORCEMENT: Only give reach_target_reward if BOTH agents are within contact_threshold of box at success. Solo pushing = NO reward. DEFAULT: False")
    parser.add_argument("--contact_force_gating", type=lambda x: (str(x).lower() == 'true'), default=False,
                       help="Gate push_reward and target_reward by contact-force balance ratio. Prevents freeloading in heterogeneous teams. DEFAULT: False")
    parser.add_argument("--contact_force_gating_alpha", type=float, default=0.3,
                       help="Minimum gate value when balance=0 (freeloader gets alpha fraction of gated rewards). 1.0=disabled, 0.0=full gating. DEFAULT: 0.3")
    parser.add_argument("--box_mass", type=float, default=None,
                       help="Override box mass in kg. None = use config default (8kg). DEFAULT: None")
    parser.add_argument("--box_mass_range", type=float, nargs=2, default=None,
                       help="Randomize box mass uniformly in [min, max] kg per env. Overrides --box_mass. Example: --box_mass_range 4 20")
    parser.add_argument("--agent0", type=str, default="go1",
                       help="Robot type for agent 0. Options: go1, anymal_c. DEFAULT: go1")
    parser.add_argument("--agent1", type=str, default="go1",
                       help="Robot type for agent 1. Options: go1, anymal_c. DEFAULT: go1")
    # Velocity task arguments
    parser.add_argument("--vel_speed_min", type=float, default=None,
                       help="Velocity task: minimum commanded speed (m/s). DEFAULT: config value (0.3)")
    parser.add_argument("--vel_speed_max", type=float, default=None,
                       help="Velocity task: maximum commanded speed (m/s). DEFAULT: config value (1.0)")
    parser.add_argument("--vel_tracking_scale", type=float, default=None,
                       help="Velocity task: velocity tracking reward scale. DEFAULT: config value (0.01)")
    parser.add_argument("--vel_angular_penalty_scale", type=float, default=None,
                       help="Velocity task: angular velocity penalty scale. DEFAULT: config value (-0.005)")
    parser.add_argument("--legacy_vel_obs", action="store_true", default=False,
                       help="Use legacy 16-dim velocity obs (includes cmd_speed) for loading old checkpoints")
    parser.add_argument("--seed", type=int, default=1,
                       help="Random seed")

    # Training arguments
    parser.add_argument("--n_rollout_threads", type=int, default=None,
                       help="Number of parallel environments (default: from YAML config)")
    parser.add_argument("--num_env_steps", type=int, default=None,
                       help="Total number of environment steps (default: from YAML config)")
    parser.add_argument("--episode_length", type=int, default=None,
                       help="Rollout length: steps to collect before update (default: from YAML config)")

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
    # ALWAYS use EP mode for critic (single global value function)
    # FP mode causes critic divergence - per-agent value estimation is unstable
    # Individualized rewards only affects reward shaping, NOT critic mode
    use_individual = args.get("individualized_rewards", False)
    use_shared_gated = args.get("shared_gated_rewards", False)
    use_goal_centered = args.get("use_goal_centered_critic", False)
    use_box_centered = args.get("use_box_centered_critic", False)
    use_concat_obs = args.get("use_concat_agent_observations_critic", False)
    use_relative_obs = args.get("use_relative_obs_critic", False)
    use_cooperation = args.get("cooperation_rewards", False)
    use_og_rewards = args.get("mapush_og_rewards_teamified", False)
    use_reward_testing = args.get("reward_scale_testing", False)
    use_collaboration = args.get("collaboration_rewards", False)
    use_positive_approach = args.get("positive_approachtobox_reward", False)
    use_require_both_contact = args.get("require_both_contact_for_success", False)
    use_contact_force_gating = args.get("contact_force_gating", False)
    contact_force_gating_alpha = args.get("contact_force_gating_alpha", 0.3)
    box_mass = args.get("box_mass", None)
    box_mass_range = args.get("box_mass_range", None)

    # Velocity task parameters
    vel_speed_min = args.get("vel_speed_min", None)
    vel_speed_max = args.get("vel_speed_max", None)
    vel_tracking_scale = args.get("vel_tracking_scale", None)
    vel_angular_penalty_scale = args.get("vel_angular_penalty_scale", None)

    # Agent types for heterogeneous training
    agent0 = args.get("agent0", "go1")
    agent1 = args.get("agent1", "go1")
    is_hetero = (agent0 != agent1)

    # n_threads defaults to YAML config value if not specified on command line
    n_threads = args.get("n_rollout_threads") or algo_args["train"]["n_rollout_threads"]
    env_args = {
        "task": args.get("task", "go1push_mid"),  # Use task from command line
        "n_threads": n_threads,
        "state_type": "EP",  # ALWAYS EP - HAPPO uses single global critic per documentation
        "individualized_rewards": use_individual,  # For reward shaping only
        "shared_gated_rewards": use_shared_gated,  # Iter8: Gate shared rewards by min engagement
        "use_goal_centered_critic": use_goal_centered,  # CRITIC16: Goal-centered coordinates (stationary reference frame)
        "use_box_centered_critic": use_box_centered,  # CRITIC9: Box-centered (True) vs CRITIC7: Absolute (False)
        "use_concat_agent_observations_critic": use_concat_obs,  # CRITIC10: Concatenated agent observations
        "use_relative_obs_critic": use_relative_obs,  # CRITIC11: Relative observations with inter-robot distance (highest priority)
        "cooperation_rewards": use_cooperation,  # CRITIC12: Three-tier cooperation bonuses
        "mapush_og_rewards_teamified": use_og_rewards,  # Original MAPush rewards converted to team rewards
        "reward_scale_testing": use_reward_testing,  # Reserved for future experiments
        "collaboration_rewards": use_collaboration,  # CRITIC15 v4: Dual pushing bonus (use with mapush_og_rewards_teamified)
        "positive_approachtobox_reward": use_positive_approach,  # CRITIC17: Positive inverse distance reward instead of negative penalty
        "agent0": agent0,  # Robot type for agent 0
        "agent1": agent1,  # Robot type for agent 1
        "require_both_contact_for_success": use_require_both_contact,  # COLLABORATION ENFORCEMENT: Only give reach_target_reward if BOTH agents in contact
        "contact_force_gating": use_contact_force_gating,  # Gate push/target rewards by contact-force balance
        "contact_force_gating_alpha": contact_force_gating_alpha,  # Min gate value (0.3 = freeloader gets 30%)
        "box_mass": box_mass,  # Override box mass in kg (None = config default)
        "box_mass_range": box_mass_range,  # Randomize box mass in [min, max] kg per env
        # Velocity task parameters
        "vel_speed_min": vel_speed_min,
        "vel_speed_max": vel_speed_max,
        "vel_tracking_scale": vel_tracking_scale,
        "vel_angular_penalty_scale": vel_angular_penalty_scale,
        "legacy_vel_obs": args.get("legacy_vel_obs", False),
    }

    # Override training parameters only if specified on command line
    if args.get("n_rollout_threads") is not None:
        algo_args["train"]["n_rollout_threads"] = args["n_rollout_threads"]
    if args.get("num_env_steps") is not None:
        algo_args["train"]["num_env_steps"] = args["num_env_steps"]
    if args.get("episode_length") is not None:
        algo_args["train"]["episode_length"] = args["episode_length"]

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
    if is_hetero:
        print(f"Agent types: HETEROGENEOUS (agent0={agent0}, agent1={agent1})")
    else:
        print(f"Agent types: HOMOGENEOUS (both {agent0})")
    print(f"Critic mode: {env_args['state_type']} (EP = single global critic)")
    print(f"Individualized rewards: {env_args['individualized_rewards']}")
    if env_args['individualized_rewards']:
        print(f"  → Contact-weighted rewards averaged to team reward for stable critic")
    print(f"Original MAPush rewards (teamified): {env_args['mapush_og_rewards_teamified']}")
    if env_args['mapush_og_rewards_teamified']:
        print(f"  → 7 rewards: reach_target, distance_to_target, approach_to_box (avg),")
        print(f"               collision (-0.0025), push, ocb (±0.004), exception")
        print(f"  → Disabled: goal_push_bonus, proximity_penalty")
    if env_args['task'] == 'go1push_vel':
        print(f"Task type: VELOCITY (push box in commanded direction/speed)")
        if vel_speed_min is not None or vel_speed_max is not None:
            print(f"  Speed range override: [{vel_speed_min}, {vel_speed_max}]")
        if vel_tracking_scale is not None:
            print(f"  Tracking scale override: {vel_tracking_scale}")
        if vel_angular_penalty_scale is not None:
            print(f"  Angular penalty scale override: {vel_angular_penalty_scale}")
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
