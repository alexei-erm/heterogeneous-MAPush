"""Custom HAPPO runner for MAPush with step-based checkpointing.

This runner extends HARL's OnPolicyHARunner to add:
- Checkpoint saving every 10M steps (instead of episode-based)
- Checkpoint folder naming: 10M, 20M, 30M, etc.
- Three models per checkpoint: actor_agent0.pt, actor_agent1.pt, critic_agent.pt
"""
import os
import torch
import numpy as np
from harl.runners.on_policy_ha_runner import OnPolicyHARunner


class MAPushHAPPORunner(OnPolicyHARunner):
    """HAPPO runner with MAPush-specific checkpointing every 10M steps."""

    def __init__(self, args, algo_args, env_args):
        """Initialize runner with step-based checkpointing.

        Args:
            args: Main arguments (algo, env, exp_name)
            algo_args: Algorithm configuration
            env_args: Environment configuration
        """
        # Disable evaluation (Isaac Gym doesn't support multiple instances)
        algo_args["eval"]["use_eval"] = False

        super().__init__(args, algo_args, env_args)

        # Set envs reference in logger for MAPush statistics tracking
        if hasattr(self.logger, 'set_envs'):
            self.logger.set_envs(self.envs)

        # Checkpoint configuration
        self.checkpoint_interval = 10_000_000  # 10M steps
        self.last_checkpoint_step = 0

        # Create checkpoints directory
        self.checkpoints_dir = os.path.join(self.run_dir, "checkpoints")
        os.makedirs(self.checkpoints_dir, exist_ok=True)

        # Track total steps across all episodes
        self.total_steps = 0

        print(f"\n{'='*60}")
        print(f"MAPush HAPPO Runner Initialized")
        print(f"{'='*60}")
        print(f"Checkpoint interval: {self.checkpoint_interval:,} steps")
        print(f"Checkpoints will be saved to: {self.checkpoints_dir}")
        print(f"{'='*60}\n")

    def run(self):
        """Run training with step-based checkpointing.

        Training Structure (MAPush mid-level):
        - MAPush actual episodes: 1000 steps (20 seconds @ 50Hz control frequency)
        - HARL rollout length: 200 steps (matches original MAPPO training)
        - Each rollout: collect 200 steps across all parallel environments, then update
        - Each MAPush episode spans 5 rollouts (1000 / 200 = 5)
        - Checkpoint: every 10M total steps (not rollouts)

        Example: 100M steps with 500 envs, 200 steps/rollout:
        - Total rollouts: 100M / (500 * 200) = 1000 rollouts
        - Actual MAPush episodes: ~200 episodes (1000 rollouts / 5 = 200)
        - Checkpoints at: 10M, 20M, ..., 100M steps
        """
        self.warmup()

        rollouts = (
            int(self.algo_args["train"]["num_env_steps"])
            // self.algo_args["train"]["episode_length"]
            // self.algo_args["train"]["n_rollout_threads"]
        )

        self.logger.init(rollouts)  # logger callback at the beginning of training

        print(f"\nStarting training for {rollouts} rollouts (~{rollouts // 5} MAPush episodes)")
        print(f"Total environment steps: {self.algo_args['train']['num_env_steps']:,}")
        print(f"Rollout length: {self.algo_args['train']['episode_length']} steps/rollout")
        print(f"MAPush episode length: 1000 steps (20s @ 50Hz)")
        print(f"Parallel envs: {self.algo_args['train']['n_rollout_threads']}")
        print(f"Update frequency: every {self.algo_args['train']['episode_length']} steps\n")

        for rollout in range(1, rollouts + 1):
            # Learning rate decay
            if self.algo_args["train"]["use_linear_lr_decay"]:
                self.actor[0].lr_decay(rollout, rollouts)
                if not self.share_param:
                    for agent_id in range(1, self.num_agents):
                        self.actor[agent_id].lr_decay(rollout, rollouts)
                self.critic.lr_decay(rollout, rollouts)

            self.logger.episode_init(rollout)  # Still called episode_init in base class

            # Rollout phase
            self.prep_rollout()
            for step in range(self.algo_args["train"]["episode_length"]):
                # Collect actions, observations, rewards
                (
                    values,
                    actions,
                    action_log_probs,
                    rnn_states,
                    rnn_states_critic,
                ) = self.collect(step)

                # Step environments
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

                self.logger.per_step(data)  # logger callback at each step
                self.insert(data)

                # Update total steps
                self.total_steps += self.algo_args["train"]["n_rollout_threads"]

                # Check if we should save checkpoint (every 10M steps)
                if self.total_steps - self.last_checkpoint_step >= self.checkpoint_interval:
                    self.save_checkpoint(self.total_steps)
                    self.last_checkpoint_step = self.total_steps

            # Compute returns and advantages
            self.compute()

            # Training phase
            self.prep_training()
            actor_train_infos, critic_train_info = self.train()

            # Logging
            if rollout % self.algo_args["train"]["log_interval"] == 0:
                self.logger.episode_log(
                    actor_train_infos,
                    critic_train_info,
                    self.actor_buffer,
                    self.critic_buffer,
                )

                # Print progress
                progress = rollout / rollouts * 100
                mapush_eps = (rollout * self.algo_args["train"]["episode_length"] *
                             self.algo_args["train"]["n_rollout_threads"]) // 1000
                print(f"[Rollout {rollout}/{rollouts}] "
                      f"Progress: {progress:.1f}% | "
                      f"Steps: {self.total_steps:,} | "
                      f"~MAPush episodes: {mapush_eps:,}")

            # Evaluation
            if rollout % self.algo_args["train"]["eval_interval"] == 0:
                if self.algo_args["eval"]["use_eval"]:
                    self.prep_rollout()
                    self.eval()

            self.after_update()

        # Save final checkpoint
        print(f"\nTraining complete! Saving final checkpoint...")
        self.save_checkpoint(self.total_steps)

        print(f"\n{'='*60}")
        print(f"Training Summary")
        print(f"{'='*60}")
        print(f"Total episodes: {episodes}")
        print(f"Total steps: {self.total_steps:,}")
        print(f"Checkpoints saved: {self.total_steps // self.checkpoint_interval}")
        print(f"{'='*60}\n")

    def save_checkpoint(self, steps):
        """Save checkpoint at specific step count.

        Args:
            steps: Current total step count
        """
        # Create checkpoint directory named by millions of steps
        checkpoint_name = f"{steps // 1_000_000}M"
        checkpoint_path = os.path.join(self.checkpoints_dir, checkpoint_name)
        os.makedirs(checkpoint_path, exist_ok=True)

        # Save actor models (one per agent)
        for agent_id in range(self.num_agents):
            policy_actor = self.actor[agent_id].actor
            save_path = os.path.join(checkpoint_path, f"actor_agent{agent_id}.pt")
            torch.save(policy_actor.state_dict(), save_path)

        # Save critic model
        policy_critic = self.critic.critic
        critic_path = os.path.join(checkpoint_path, "critic_agent.pt")
        torch.save(policy_critic.state_dict(), critic_path)

        # Save value normalizer if it exists
        if self.value_normalizer is not None:
            normalizer_path = os.path.join(checkpoint_path, "value_normalizer.pt")
            torch.save(self.value_normalizer.state_dict(), normalizer_path)

        print(f"[Checkpoint] Saved models at {steps:,} steps → {checkpoint_path}")

    def restore_checkpoint(self, checkpoint_path):
        """Restore models from a checkpoint directory.

        Args:
            checkpoint_path: Path to checkpoint folder (e.g., .../checkpoints/50M/)
        """
        print(f"\nRestoring checkpoint from: {checkpoint_path}")

        # Load actor models
        for agent_id in range(self.num_agents):
            actor_path = os.path.join(checkpoint_path, f"actor_agent{agent_id}.pt")
            if os.path.exists(actor_path):
                state_dict = torch.load(actor_path)
                self.actor[agent_id].actor.load_state_dict(state_dict)
                print(f"  ✓ Loaded actor_agent{agent_id}.pt")
            else:
                print(f"  ✗ Warning: {actor_path} not found")

        # Load critic model
        critic_path = os.path.join(checkpoint_path, "critic_agent.pt")
        if os.path.exists(critic_path):
            state_dict = torch.load(critic_path)
            self.critic.critic.load_state_dict(state_dict)
            print(f"  ✓ Loaded critic_agent.pt")
        else:
            print(f"  ✗ Warning: {critic_path} not found")

        # Load value normalizer if it exists
        normalizer_path = os.path.join(checkpoint_path, "value_normalizer.pt")
        if os.path.exists(normalizer_path) and self.value_normalizer is not None:
            state_dict = torch.load(normalizer_path)
            self.value_normalizer.load_state_dict(state_dict)
            print(f"  ✓ Loaded value_normalizer.pt")

        print(f"Checkpoint restored successfully!\n")
