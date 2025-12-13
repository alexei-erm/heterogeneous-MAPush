"""Testing script for MAPush with HAPPO.

This script provides two testing modes:
1. Viewer mode: Visualize episodes sequentially (one at a time)
2. Calculator mode: Compute statistics over multiple parallel environments

Usage:
    # Calculator mode (statistics over many episodes)
    python test.py --checkpoint path/to/checkpoint/10M --mode calculator --num_episodes 100 --num_envs 300

    # Viewer mode (visualize episodes)
    python test.py --checkpoint path/to/checkpoint/10M --mode viewer --num_episodes 5
"""
import sys
import os

# Add paths - INSERT at beginning to override any PYTHONPATH pollution
sys.path.insert(0, '/home/gvlab/new-universal-MAPush/HARL')
sys.path.insert(0, '/home/gvlab/new-universal-MAPush')

# CRITICAL: Import Isaac Gym BEFORE any other imports (especially torch)
import isaacgym

# Now safe to import torch and other modules
import argparse
import torch
import numpy as np

from harl.envs.mapush.mapush_env import MAPushEnv
from harl.algorithms.actors.happo import HAPPO
from harl.utils.configs_tools import get_defaults_yaml_args


def load_models(checkpoint_dir, n_agents, obs_spaces, act_spaces, device="cuda"):
    """Load HAPPO actor models from checkpoint directory.

    Args:
        checkpoint_dir: Path to checkpoint directory (e.g., .../checkpoints/10M)
        n_agents: Number of agents
        obs_spaces: List of observation spaces (one per agent)
        act_spaces: List of action spaces (one per agent)
        device: Device to load models on

    Returns:
        List of HAPPO actor objects
    """
    print(f"\nLoading models from: {checkpoint_dir}")

    # Load default HAPPO args
    algo_args, _ = get_defaults_yaml_args("happo", "pettingzoo_mpe")

    # Merge algo and model args (HAPPO needs both)
    actor_args = {**algo_args["model"], **algo_args["algo"]}

    actors = []

    for agent_id in range(n_agents):
        # Create HAPPO actor
        actor = HAPPO(
            actor_args,
            obs_spaces[agent_id],
            act_spaces[agent_id],
            device=torch.device(device)
        )

        # Load checkpoint
        model_path = os.path.join(checkpoint_dir, f"actor_agent{agent_id}.pt")
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Actor model not found: {model_path}")

        print(f"  Loading actor_agent{agent_id}.pt")
        actor.actor.load_state_dict(torch.load(model_path, map_location=device))
        actor.actor.eval()  # Set to evaluation mode

        actors.append(actor)

    print(f"Successfully loaded {n_agents} actor models\n")
    return actors


def test_calculator_mode(actors, env, num_episodes, seed):
    """Run calculator mode to compute statistics over many episodes.

    Args:
        actors: List of HAPPO actors
        env: MAPushEnv instance
        num_episodes: Number of episodes to run
        seed: Random seed
    """
    print(f"\n{'='*70}")
    print(f"Calculator Mode - Computing statistics over {num_episodes} episodes")
    print(f"{'='*70}\n")

    # Seed environment
    env.seed(seed)
    env.reset_statistics()

    n_agents = env.n_agents
    n_envs = env.n_envs

    # Initialize RNN states (HAPPO uses recurrent policy option)
    # Check if using recurrent policy
    use_recurrent = actors[0].actor.rnn is not None if hasattr(actors[0].actor, 'rnn') else False

    if use_recurrent:
        rnn_hidden_size = actors[0].actor.rnn_hidden_size
        recurrent_n = actors[0].actor.recurrent_n
    else:
        rnn_hidden_size = 64  # Default
        recurrent_n = 1

    episodes_completed = 0
    step_count = 0

    # Reset environment
    obs, _, _ = env.reset()
    rnn_states = np.zeros((n_envs, n_agents, recurrent_n, rnn_hidden_size), dtype=np.float32)
    masks = np.ones((n_envs, n_agents, 1), dtype=np.float32)

    print(f"Running with {n_envs} parallel environments...")
    print(f"Target episodes: {num_episodes}")
    print(f"Progress: ", end="", flush=True)

    while episodes_completed < num_episodes:
        # Collect actions from all actors
        actions_list = []

        for agent_id in range(n_agents):
            # Get action from actor (deterministic for testing)
            with torch.no_grad():
                action, rnn_state = actors[agent_id].act(
                    obs[:, agent_id],  # [n_envs, obs_dim]
                    rnn_states[:, agent_id],  # [n_envs, recurrent_n, rnn_hidden_size]
                    masks[:, agent_id],  # [n_envs, 1]
                    available_actions=None,
                    deterministic=True  # Use deterministic policy for evaluation
                )

            actions_list.append(action.cpu().numpy())
            rnn_states[:, agent_id] = rnn_state.cpu().numpy()

        # Stack actions: [n_envs, n_agents, action_dim]
        actions = np.stack(actions_list, axis=1)

        # Step environment
        obs, _, rewards, dones, infos, _ = env.step(actions)

        # Update masks for done environments
        dones_env = np.all(dones, axis=1)  # [n_envs]

        # Reset RNN states for done environments
        if np.any(dones_env):
            rnn_states[dones_env] = np.zeros(
                (dones_env.sum(), n_agents, recurrent_n, rnn_hidden_size),
                dtype=np.float32
            )
            masks[dones_env] = np.zeros((dones_env.sum(), n_agents, 1), dtype=np.float32)
        else:
            masks = np.ones((n_envs, n_agents, 1), dtype=np.float32)

        # Count completed episodes
        new_episodes = dones_env.sum()
        episodes_completed += new_episodes
        step_count += 1

        # Progress indicator
        if episodes_completed % max(1, num_episodes // 20) == 0:
            print(".", end="", flush=True)

        if episodes_completed >= num_episodes:
            break

    print(" Done!\n")

    # Get final statistics
    stats = env.get_statistics()

    # Print results
    # MAPush runs at 50 Hz (dt = 0.02s)
    dt = 0.02  # seconds per step
    avg_time = stats['avg_episode_length'] * dt

    print(f"\n{'='*70}")
    print(f"Statistics Summary (over {stats['num_episodes']} episodes)")
    print(f"{'='*70}")
    print(f"  Success Rate:         {stats['success_rate']:.4f} ({stats['success_rate']*100:.2f}%)")
    print(f"                        [{stats['num_success']}/{stats['num_episodes']} episodes succeeded]")
    print(f"\n  Episode Metrics (high-level task only):")
    print(f"    Avg Episode Length:   {stats['avg_episode_length']:.1f} steps ({avg_time:.2f}s)")
    print(f"    Collision Rate:       {stats['collision_rate']:.4f}")
    print(f"    Collaboration Degree: {stats['collaboration_degree']:.4f}")
    print(f"{'='*70}\n")

    return stats


def test_viewer_mode(checkpoint_dir, num_episodes, seed):
    """Run viewer mode to visualize episodes sequentially.

    Args:
        checkpoint_dir: Path to checkpoint directory
        num_episodes: Number of episodes to visualize
        seed: Random seed
    """
    print(f"\n{'='*70}")
    print(f"Viewer Mode - Visualizing {num_episodes} episodes")
    print(f"{'='*70}\n")

    # Create single-environment version for visualization
    from types import SimpleNamespace
    from mqe.envs.utils import make_mqe_env, custom_cfg
    from task.cuboid.config import Go1PushMidCfg

    # Create args for visualization
    args = argparse.Namespace()
    args.task = "go1push_mid"
    args.num_envs = 1
    args.seed = seed
    args.headless = False  # Enable rendering
    args.record_video = False
    args.rl_device = "cuda:0"
    args.sim_device = "cuda:0"
    args.device = "cuda"
    args.compute_device_id = 0
    args.sim_device_type = "cuda"
    args.use_gpu_pipeline = True

    from isaacgym import gymapi
    args.physics_engine = gymapi.SIM_PHYSX
    args.use_gpu = True
    args.subscenes = 0
    args.num_threads = 0

    # Create environment
    print("Creating visualization environment...")
    env_raw, _ = make_mqe_env(args.task, args, custom_cfg=custom_cfg(args))

    n_agents = env_raw.num_agents

    # Get observation and action spaces from env
    obs_space = env_raw.observation_space
    act_space = env_raw.action_space
    obs_spaces = [obs_space] * n_agents
    act_spaces = [act_space] * n_agents

    # Load models
    actors = load_models(checkpoint_dir, n_agents, obs_spaces, act_spaces, device="cuda:0")

    # Check if using recurrent policy
    use_recurrent = actors[0].actor.rnn is not None if hasattr(actors[0].actor, 'rnn') else False

    if use_recurrent:
        rnn_hidden_size = actors[0].actor.rnn_hidden_size
        recurrent_n = actors[0].actor.recurrent_n
    else:
        rnn_hidden_size = 64
        recurrent_n = 1

    # Run episodes
    for episode_idx in range(num_episodes):
        print(f"\n{'─'*70}")
        print(f"Episode {episode_idx + 1}/{num_episodes}")
        print(f"{'─'*70}")

        # Reset environment
        obs = env_raw.reset()
        obs_np = obs.cpu().numpy()  # [1, n_agents, obs_dim]

        # Initialize RNN states and masks
        rnn_states = np.zeros((1, n_agents, recurrent_n, rnn_hidden_size), dtype=np.float32)
        masks = np.ones((1, n_agents, 1), dtype=np.float32)

        done = False
        step_count = 0
        episode_reward = 0.0

        while not done:
            # Collect actions from all actors
            actions_list = []

            for agent_id in range(n_agents):
                with torch.no_grad():
                    action, rnn_state = actors[agent_id].act(
                        obs_np[:, agent_id],  # [1, obs_dim]
                        rnn_states[:, agent_id],  # [1, recurrent_n, rnn_hidden_size]
                        masks[:, agent_id],  # [1, 1]
                        available_actions=None,
                        deterministic=True
                    )

                actions_list.append(action.cpu().numpy())
                rnn_states[:, agent_id] = rnn_state.cpu().numpy()

            # Stack and convert to torch
            actions_np = np.stack(actions_list, axis=1)  # [1, n_agents, action_dim]
            actions_torch = torch.from_numpy(actions_np[0]).cuda()  # [n_agents, action_dim]

            # Step environment
            obs, rewards, dones_raw, infos = env_raw.step(actions_torch)
            obs_np = obs.cpu().numpy()

            done = dones_raw.cpu().item()
            episode_reward += rewards.cpu().numpy().sum()
            step_count += 1

            # Update masks
            if done:
                masks = np.zeros((1, n_agents, 1), dtype=np.float32)

        # Print episode results
        success = env_raw.finished_buf[0].item() if hasattr(env_raw, 'finished_buf') else False

        print(f"  Steps:   {step_count}")
        print(f"  Reward:  {episode_reward:.2f}")
        print(f"  Result:  {'✓ SUCCESS' if success else '✗ FAILED'}")

        if hasattr(env_raw, 'collision_degree_buf') and success:
            collision = env_raw.collision_degree_buf[0].item() / step_count
            print(f"  Collision Rate: {collision:.4f}")

        if hasattr(env_raw, 'collaboration_degree_buf') and success:
            collab = env_raw.collaboration_degree_buf[0].item() / step_count
            print(f"  Collaboration:  {collab:.4f}")

    print(f"\n{'='*70}\n")

    # Close environment
    env_raw.close()


def main():
    """Main testing function."""
    parser = argparse.ArgumentParser(
        description="Test trained HAPPO models on MAPush",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    # Required arguments
    parser.add_argument("--checkpoint", type=str, required=True,
                       help="Path to checkpoint directory or parent directory with --all_checkpoints")

    # Mode selection
    parser.add_argument("--mode", type=str, default="calculator",
                       choices=["calculator", "viewer"],
                       help="Testing mode: calculator (statistics) or viewer (visualization)")

    # Checkpoint selection
    parser.add_argument("--all_checkpoints", action="store_true",
                       help="Test all checkpoint subdirectories (10M, 20M, etc.) in the given path")

    # Episode configuration
    parser.add_argument("--num_episodes", type=int, default=100,
                       help="Number of episodes to run (viewer mode) or target (calculator mode)")
    parser.add_argument("--num_envs", type=int, default=300,
                       help="Number of parallel environments (calculator mode only)")

    # Other
    parser.add_argument("--seed", type=int, default=1,
                       help="Random seed")

    args = parser.parse_args()

    # Verify checkpoint path exists
    if not os.path.exists(args.checkpoint):
        print(f"\nERROR: Path not found: {args.checkpoint}\n")
        sys.exit(1)

    # Get list of checkpoints to test
    checkpoints_to_test = []

    if args.all_checkpoints:
        # Find all checkpoint subdirectories (10M, 20M, etc.)
        if not os.path.isdir(args.checkpoint):
            print(f"\nERROR: --all_checkpoints requires a directory path\n")
            sys.exit(1)

        # Look for subdirectories that contain actor_agent0.pt
        for item in sorted(os.listdir(args.checkpoint)):
            item_path = os.path.join(args.checkpoint, item)
            if os.path.isdir(item_path):
                actor0 = os.path.join(item_path, "actor_agent0.pt")
                actor1 = os.path.join(item_path, "actor_agent1.pt")
                if os.path.exists(actor0) and os.path.exists(actor1):
                    checkpoints_to_test.append(item_path)

        if not checkpoints_to_test:
            print(f"\nERROR: No valid checkpoints found in: {args.checkpoint}")
            print("Looking for subdirectories containing actor_agent0.pt and actor_agent1.pt\n")
            sys.exit(1)

        print(f"\nFound {len(checkpoints_to_test)} checkpoints to test:")
        for ckpt in checkpoints_to_test:
            print(f"  - {os.path.basename(ckpt)}")
        print()

    else:
        # Single checkpoint mode
        actor0_path = os.path.join(args.checkpoint, "actor_agent0.pt")
        actor1_path = os.path.join(args.checkpoint, "actor_agent1.pt")

        if not os.path.exists(actor0_path) or not os.path.exists(actor1_path):
            print(f"\nERROR: Actor models not found in checkpoint directory")
            print(f"  Looking for: actor_agent0.pt, actor_agent1.pt")
            print(f"  In directory: {args.checkpoint}")
            print("\nTip: Use --all_checkpoints to test all checkpoints in a directory\n")
            sys.exit(1)

        checkpoints_to_test = [args.checkpoint]

    # Loop over all checkpoints
    all_results = []

    for idx, checkpoint_path in enumerate(checkpoints_to_test):
        checkpoint_name = os.path.basename(checkpoint_path)

        # Print configuration
        if len(checkpoints_to_test) > 1:
            print(f"\n{'='*70}")
            print(f"Testing Checkpoint [{idx+1}/{len(checkpoints_to_test)}]: {checkpoint_name}")
            print(f"{'='*70}")
        else:
            print("\n" + "="*70)
            print("MAPush HAPPO Testing Configuration")
            print("="*70)
            print(f"Checkpoint:    {checkpoint_path}")
            print(f"Mode:          {args.mode}")
            print(f"Seed:          {args.seed}")

            if args.mode == "calculator":
                print(f"Episodes:      {args.num_episodes} (target)")
                print(f"Parallel envs: {args.num_envs}")
            else:
                print(f"Episodes:      {args.num_episodes}")

            print("="*70)

        # Run appropriate mode
        if args.mode == "calculator":
            # Create multi-env environment (only once for all checkpoints)
            if idx == 0:
                env_args = {
                    "task": "go1push_mid",
                    "n_threads": args.num_envs,
                    "headless": True,
                }
                print("\nInitializing calculator mode environment...")
                env = MAPushEnv(env_args)

            # Reset statistics for this checkpoint
            env.reset_statistics()

            # Load models
            actors = load_models(
                checkpoint_path,
                env.n_agents,
                env.observation_space,
                env.action_space,
                device="cuda:0"
            )

            # Run calculator mode
            stats = test_calculator_mode(actors, env, args.num_episodes, args.seed)
            all_results.append({
                'checkpoint': checkpoint_name,
                'success_rate': stats['success_rate'],
                'num_episodes': stats['num_episodes']
            })

        else:  # viewer mode
            if len(checkpoints_to_test) > 1:
                print(f"\nWARNING: Viewer mode with --all_checkpoints will show each checkpoint sequentially")
                print("This may take a long time. Consider using calculator mode instead.\n")

            # Run viewer mode (creates its own single-env environment)
            test_viewer_mode(checkpoint_path, args.num_episodes, args.seed)

    # Clean up
    if args.mode == "calculator":
        try:
            env.close()
        except:
            pass

    # Print summary if testing multiple checkpoints
    if len(checkpoints_to_test) > 1 and args.mode == "calculator":
        print("\n" + "="*70)
        print("Summary of All Checkpoints")
        print("="*70)
        for result in all_results:
            print(f"  {result['checkpoint']:8s}  Success Rate: {result['success_rate']:.4f} ({result['success_rate']*100:.2f}%)  [{int(result['success_rate']*result['num_episodes'])}/{result['num_episodes']} episodes]")
        print("="*70)

        # Save results to file
        output_dir = os.path.dirname(checkpoints_to_test[0])
        output_file = os.path.join(output_dir, "test_results.txt")

        with open(output_file, 'w') as f:
            import datetime
            f.write("="*70 + "\n")
            f.write("MAPush HAPPO Testing Results\n")
            f.write("="*70 + "\n")
            f.write(f"Test Date: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Num Episodes per Checkpoint: {args.num_episodes}\n")
            f.write(f"Num Parallel Envs: {args.num_envs}\n")
            f.write(f"Seed: {args.seed}\n")
            f.write("="*70 + "\n\n")

            f.write("Results by Checkpoint:\n")
            f.write("-"*70 + "\n")
            for result in all_results:
                f.write(f"{result['checkpoint']:8s}  Success Rate: {result['success_rate']:.4f} ({result['success_rate']*100:.2f}%)  ")
                f.write(f"[{int(result['success_rate']*result['num_episodes'])}/{result['num_episodes']} episodes]\n")
            f.write("="*70 + "\n")

        print(f"\nResults saved to: {output_file}")

    print("\nTesting completed successfully!\n")


if __name__ == "__main__":
    # Check running from correct directory
    if not os.path.exists("./resources/actuator_nets"):
        print("\n" + "="*70)
        print("ERROR: Must run from MAPush root directory!")
        print("="*70)
        print("\nPlease run:")
        print("  cd /home/gvlab/new-universal-MAPush")
        print("  conda run -n mapush python HARL/harl_mapush/test.py [args]")
        print("\n" + "="*70 + "\n")
        sys.exit(1)

    main()
