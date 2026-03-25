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

    class asset(Go1PushMidCfg.asset):
        """Override box mass to 8kg for faster learning.
        Override target NPC to use arrow marker URDF (green elongated box) instead
        of the red circle. Unfix base link so we can reposition + rotate it each
        step, and disable gravity so it floats in place."""
        # --- 1.5x bigger box (1.8m) — DISABLED, using default SmallBox (1.2m x 1.2m x 0.5m) ---
        # file_npc = "{LEGGED_GYM_ROOT_DIR}/resources/objects/cuboid/VelBox.urdf"
        # npc_box_dimensions = [1.8, 1.8, 0.75]  # must match VelBox.urdf
        npc_mass_override = 8
        _file_npc = "{LEGGED_GYM_ROOT_DIR}/resources/objects/arrow.urdf"
        _fix_npc_base_link = False  # Allow teleporting target NPC (arrow marker)
        _npc_gravity = False        # Prevent target NPC from falling

    class domain_rand(Go1PushMidCfg.domain_rand):
        """Spawn radius for 1.2m box (edge at 0.6m from center).
        ~0.6m clearance: 0.6 + 0.6 = 1.2m minimum radius."""
        random_base_init_state = True
        init_base_pos_range = dict(
            r=[1.2, 1.3],  # reverted from [1.5, 1.6] (was for 1.8m big box)
            theta=[-0.01, 2 * np.pi],
        )

    class termination(Go1PushMidCfg.termination):
        """Override termination: remove z_wave since arrow marker (target NPC)
        gets repositioned each step, which would always trigger z_wave."""
        termination_terms = [
            "roll",
            "pitch",
            # "z_wave",  — REMOVED: arrow marker repositioning triggers false z_wave on target NPC
            "collision",
        ]

    class velocity_command:
        """Velocity command parameters sampled per episode."""
        speed_range = [0.5, 0.5]        # [min, max] m/s — fixed speed (no randomization)
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
        THRESHOLD = -1.0  # Negative → distance is never < -1 → finished_buf always False
        # NOTE: No check_setting validation — all four goal modes are False by design

    class rewards(Go1PushMidCfg.rewards):
        """Velocity-specific reward scales."""

        # Contact-force gating is now controlled by CLI flags:
        #   --contact_force_gating True --contact_force_gating_alpha 0.1
        # See go1_push_vel_wrapper.py for implementation.

        # Sharpness exponent for positive cosine similarity: cos^n
        # n=1: linear (old behavior), n=4: rewards precise alignment sharply
        vel_tracking_sharpness = 4

        class scales:
            # Velocity-specific rewards
            velocity_tracking_scale = 0.05      # Dominant reward (was 0.1, before that 0.01)
            angular_velocity_penalty_scale = 0.0   # Disabled
            velocity_ocb_scale = 0.004          # Positioning: be on push side of box
            flanking_separation_scale = 0.002   # Reward agents for ~90° separation around box (cross-product)

            # Reuse from mid task
            approach_reward_scale = 0.00075
            collision_punishment_scale = -0.001  # Lightened (was -0.0025)
            push_reward_scale = 0.0015          # Re-enabled: reward when box moving > 0.1 m/s
            exception_punishment_scale = -5

            # Disabled rewards (set to 0)
            target_reward_scale = 0.0
            reach_target_reward_scale = 0.0
            ocb_reward_scale = 0.0
            proximity_penalty_scale = 0.0
