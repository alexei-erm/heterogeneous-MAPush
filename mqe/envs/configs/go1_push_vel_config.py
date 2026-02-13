import numpy as np
from mqe.envs.configs.go1_push_mid_config import Go1PushMidCfg


class Go1PushVelCfg(Go1PushMidCfg):
    """Configuration for Velocity-MAPush task.

    Inherits terrain, asset, init_state, domain_rand, termination, command,
    control from Go1PushMidCfg. Overrides goal (disabled), rewards (velocity-specific),
    and adds velocity_command parameters.

    Task: Push box in a commanded direction at a commanded speed for the full episode.
    Cooperation mechanism: Single agent induces torque (penalized by angular_velocity_penalty),
    while two agents from complementary positions achieve clean linear motion.
    """

    class velocity_command:
        """Velocity command parameters sampled per episode."""
        speed_range = [0.3, 1.0]        # [min, max] m/s commanded speed
        direction_range = [0, 2 * np.pi]  # [min, max] radians for commanded direction
        arrow_offset = 2.0               # meters ahead of box to place arrow marker

    class goal:
        """Goal disabled for velocity task — episodes run to full max_episode_length."""
        static_goal_pos = False
        goal_pos = [12.1, 0.0, 0.1]
        goal_rpy = [0.0, 0.0, 0.0]
        random_goal_pos = False
        random_goal_distance_from_init = [1.5, 3.0]
        random_goal_theta_from_init = [0, 2 * np.pi]
        random_goal_rpy_range = dict(
            r=[-0.01, 0.01],
            p=[-0.01, 0.01],
            y=[-0.01, 0.01],
        )
        received_goal_pos = False
        received_final_pos = [9.0, 0.0, 0.1]
        sequential_goal_pos = False
        goal_poses = [
            [3.0, 0.0, 0.1],
            [4.0, 0.0, 0.1],
            [5.0, 0.0, 0.1],
            [6.0, 0.0, 0.1],
            [7.0, 0.0, 0.1],
        ]
        general_dist = False
        yaw_active = True
        THRESHOLD = 99999.0  # Effectively disables finished_buf (box never "reaches" target)
        # NOTE: No check_setting validation — all four goal modes are False by design

    class rewards(Go1PushMidCfg.rewards):
        """Velocity-specific reward scales."""
        class scales:
            # Velocity-specific rewards
            velocity_tracking_scale = 0.01
            angular_velocity_penalty_scale = -0.005

            # Reuse from mid task
            approach_reward_scale = 0.00075
            collision_punishment_scale = -0.0025
            push_reward_scale = 0.0015
            exception_punishment_scale = -5

            # Disabled rewards (set to 0)
            target_reward_scale = 0.0
            reach_target_reward_scale = 0.0
            ocb_reward_scale = 0.0
            proximity_penalty_scale = 0.0
