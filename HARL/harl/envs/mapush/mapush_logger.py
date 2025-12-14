"""MAPush logger for HARL."""
from harl.common.base_logger import BaseLogger
import numpy as np


class MAPushLogger(BaseLogger):
    """Logger for MAPush environment with custom statistics tracking."""

    def __init__(self, args, algo_args, env_args, num_agents, writter, run_dir):
        """Initialize MAPush logger."""
        super().__init__(args, algo_args, env_args, num_agents, writter, run_dir)
        self.envs = None  # Will be set by runner
        self.individualized_rewards = env_args.get("individualized_rewards", False)

        # Per-agent reward tracking
        self.train_episode_rewards_per_agent = None  # Will init in init()
        self.done_episodes_rewards_per_agent = None

    def set_envs(self, envs):
        """Set the environment reference for statistics tracking."""
        self.envs = envs

    def get_task_name(self):
        """Get task name for logging."""
        return self.env_args.get("task", "cuboid")

    def init(self, episodes):
        """Initialize the logger with per-agent tracking."""
        super().init(episodes)
        # Initialize per-agent reward tracking: (n_rollout_threads, num_agents)
        n_threads = self.algo_args["train"]["n_rollout_threads"]
        self.train_episode_rewards_per_agent = np.zeros((n_threads, self.num_agents))
        self.done_episodes_rewards_per_agent = [[] for _ in range(self.num_agents)]

    def per_step(self, data):
        """Process data per step with per-agent reward tracking."""
        # Call parent (handles mean reward)
        super().per_step(data)

        # Also track per-agent rewards
        (
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
        ) = data

        dones_env = np.all(dones, axis=1)
        # rewards shape: (n_threads, num_agents, 1)
        rewards_squeezed = rewards.squeeze(-1) if rewards.ndim == 3 else rewards

        self.train_episode_rewards_per_agent += rewards_squeezed

        for t in range(self.algo_args["train"]["n_rollout_threads"]):
            if dones_env[t]:
                for agent_id in range(self.num_agents):
                    self.done_episodes_rewards_per_agent[agent_id].append(
                        self.train_episode_rewards_per_agent[t, agent_id]
                    )
                self.train_episode_rewards_per_agent[t] = 0

    def episode_log(self, actor_train_infos, critic_train_info, actor_buffer, critic_buffer):
        """Log episode information including MAPush-specific metrics."""
        # Call parent logging
        super().episode_log(actor_train_infos, critic_train_info, actor_buffer, critic_buffer)

        # Log per-agent rewards
        for agent_id in range(self.num_agents):
            if len(self.done_episodes_rewards_per_agent[agent_id]) > 0:
                avg_reward = np.mean(self.done_episodes_rewards_per_agent[agent_id])
                self.writter.add_scalar(
                    f'train_episode_rewards/agent_{agent_id}',
                    avg_reward,
                    self.total_num_steps
                )
                self.done_episodes_rewards_per_agent[agent_id] = []

        # Add MAPush-specific logging
        if self.envs is None:
            return

        # Log success rate
        if hasattr(self.envs, 'get_statistics'):
            stats = self.envs.get_statistics()
            self.writter.add_scalar('mapush/success_rate', stats['success_rate'], self.total_num_steps)

            # Print to console
            print(f"  [MAPush] Success: {stats['success_rate']:.3f} ({stats['num_success']}/{stats['num_episodes']} eps)")

            # Reset statistics after logging
            self.envs.reset_statistics()

        # Log reward components from wrapper
        # Path: self.envs (MAPushEnv) -> self.envs.env (Go1PushMidWrapper) -> reward_buffer
        wrapper = getattr(self.envs, 'env', None)
        if wrapper is not None and hasattr(wrapper, 'reward_buffer'):
            rb = wrapper.reward_buffer
            step_count = rb.get("step_count", 1)
            if step_count > 0:
                # Log each reward component (averaged over steps)
                self.writter.add_scalar('rewards/target_distance', rb["distance_to_target_reward"] / step_count, self.total_num_steps)
                self.writter.add_scalar('rewards/approach_box', rb["approach_to_box_reward"] / step_count, self.total_num_steps)
                self.writter.add_scalar('rewards/push', rb["push_reward"] / step_count, self.total_num_steps)
                self.writter.add_scalar('rewards/reach_target', rb["reach_target_reward"] / step_count, self.total_num_steps)
                self.writter.add_scalar('rewards/ocb', rb["ocb_reward"] / step_count, self.total_num_steps)
                self.writter.add_scalar('rewards/collision_punishment', rb["collision_punishment"] / step_count, self.total_num_steps)
                self.writter.add_scalar('rewards/exception_punishment', rb["exception_punishment"] / step_count, self.total_num_steps)
                self.writter.add_scalar('rewards/goal_push_bonus', rb.get("goal_push_bonus", 0) / step_count, self.total_num_steps)
                # Iter6 new rewards
                self.writter.add_scalar('rewards/engagement_bonus', rb.get("engagement_bonus", 0) / step_count, self.total_num_steps)
                self.writter.add_scalar('rewards/cooperation_bonus', rb.get("cooperation_bonus", 0) / step_count, self.total_num_steps)
                self.writter.add_scalar('rewards/same_side_bonus', rb.get("same_side_bonus", 0) / step_count, self.total_num_steps)
                self.writter.add_scalar('rewards/blocking_penalty', rb.get("blocking_penalty", 0) / step_count, self.total_num_steps)

                # Log if individualized rewards are enabled
                if hasattr(wrapper, 'individualized_rewards'):
                    self.writter.add_scalar('config/individualized_rewards', float(wrapper.individualized_rewards), self.total_num_steps)

                # Reset reward buffer
                for key in rb:
                    rb[key] = 0
