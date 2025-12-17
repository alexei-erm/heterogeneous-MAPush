"""MAPush logger for HARL.

Matches OpenRL's reward logging style exactly:
- Logs average_step_reward (same calculation as OpenRL's batch_rewards)
- Logs individual reward components averaged per step
- Does NOT log episode rewards (removed to match OpenRL)
"""
import time
import os
from harl.common.base_logger import BaseLogger
import numpy as np


class MAPushLogger(BaseLogger):
    """Logger for MAPush environment matching OpenRL's reward logging style."""

    def __init__(self, args, algo_args, env_args, num_agents, writter, run_dir):
        """Initialize MAPush logger."""
        super().__init__(args, algo_args, env_args, num_agents, writter, run_dir)
        self.envs = None  # Will be set by runner

    def set_envs(self, envs):
        """Set the environment reference for statistics tracking."""
        self.envs = envs

    def get_task_name(self):
        """Get task name for logging."""
        return self.env_args.get("task", "cuboid")

    def init(self, episodes):
        """Initialize the logger - skip episode reward tracking from parent."""
        self.start = time.time()
        self.episodes = episodes
        # We don't track episode rewards - only step rewards like OpenRL

    def per_step(self, data):
        """Process data per step - no episode reward accumulation needed."""
        # We don't accumulate episode rewards like the parent does
        # OpenRL calculates rewards from reward_buffer, not from step-by-step accumulation
        pass

    def episode_log(self, actor_train_infos, critic_train_info, actor_buffer, critic_buffer):
        """Log episode information matching OpenRL's batch_rewards output exactly."""
        # Calculate total_num_steps
        self.total_num_steps = (
            self.episode
            * self.algo_args["train"]["episode_length"]
            * self.algo_args["train"]["n_rollout_threads"]
        )
        self.end = time.time()

        # Print status line (same as parent)
        print(
            "Env {} Task {} Algo {} Exp {} updates {}/{} episodes, total num timesteps {}/{}, FPS {}.".format(
                self.args["env"],
                self.task_name,
                self.args["algo"],
                self.args["exp_name"],
                self.episode,
                self.episodes,
                self.total_num_steps,
                self.algo_args["train"]["num_env_steps"],
                int(self.total_num_steps / (self.end - self.start)),
            )
        )

        # Log actor/critic training info (same as parent)
        self.log_train(actor_train_infos, critic_train_info)

        # ============================================================
        # OpenRL-style reward logging from reward_buffer
        # Matches: openrl_ws/utils.py batch_rewards() exactly
        # ============================================================
        wrapper = getattr(self.envs, 'env', None) if self.envs else None

        if wrapper is not None and hasattr(wrapper, 'reward_buffer'):
            rb = wrapper.reward_buffer
            step_count = rb.get("step_count", 0)
            num_envs = wrapper.num_envs

            if step_count > 0:
                # Calculate average_step_reward exactly like OpenRL:
                # reward_dict[k] = reward_buffer[k] / (num_envs * step_count)
                # average_step_reward = sum of all reward/punishment components

                average_step_reward = 0.0
                reward_dict = {}

                for k in rb.keys():
                    if k == "step_count":
                        continue
                    # Exact OpenRL formula
                    reward_dict[k] = rb[k] / (num_envs * step_count)
                    # Sum up rewards and punishments for average_step_reward
                    if "reward" in k or "punishment" in k or "bonus" in k or "penalty" in k:
                        average_step_reward += reward_dict[k]

                # Print average_step_reward (matches OpenRL console output)
                print(f"Average step reward is {average_step_reward}.")

                # Log to tensorboard - average_step_reward
                self.writter.add_scalar(
                    'average_step_reward',
                    average_step_reward,
                    self.total_num_steps
                )

                # Log individual reward components (same names as OpenRL)
                for k, v in reward_dict.items():
                    self.writter.add_scalar(f'rewards/{k}', v, self.total_num_steps)

                # Reset reward buffer (same as OpenRL)
                for key in rb:
                    rb[key] = 0
            else:
                print("Average step reward is N/A (no steps recorded).")
        else:
            # Fallback: use critic buffer mean (less accurate but works without wrapper)
            avg_step_reward = critic_buffer.get_mean_rewards()
            print(f"Average step reward is {avg_step_reward}.")
            self.writter.add_scalar('average_step_reward', avg_step_reward, self.total_num_steps)

        # ============================================================
        # MAPush-specific statistics (success rate, etc.)
        # ============================================================
        if self.envs is not None and hasattr(self.envs, 'get_statistics'):
            stats = self.envs.get_statistics()
            self.writter.add_scalar('mapush/success_rate', stats['success_rate'], self.total_num_steps)
            print(f"  [MAPush] Success: {stats['success_rate']:.3f} ({stats['num_success']}/{stats['num_episodes']} eps)\n")
            self.envs.reset_statistics()
        else:
            print()  # Newline for formatting
