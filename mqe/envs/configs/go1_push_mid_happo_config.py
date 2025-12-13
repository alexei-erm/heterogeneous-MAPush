"""HAPPO-specific configuration for MAPush cuboid task.

This config enables individualized rewards which are critical for HAPPO
to prevent freeloading behavior (one agent doing nothing while the other pushes).

Usage:
    Use task="go1push_mid_happo" to use this config with individualized rewards.
"""
import numpy as np
from mqe.envs.configs.go1_push_mid_config import Go1PushMidCfg


class Go1PushMidHappoCfg(Go1PushMidCfg):
    """HAPPO configuration with individualized rewards.

    Key differences from base config:
    - individualized_rewards = True: Shared rewards are weighted by agent proximity to box
    - contact_threshold = 0.8: Agents within 0.8m of box center get full reward

    This prevents freeloading by ensuring agents only get credit for actions
    when they are actually in contact with (pushing) the box.
    """

    class rewards(Go1PushMidCfg.rewards):
        expanded_ocb_reward = False
        # Enable individualized rewards for HAPPO
        individualized_rewards = True
        contact_threshold = 0.8  # meters
        class scales:
            target_reward_scale = 0.00325
            approach_reward_scale = 0.00075
            collision_punishment_scale = -0.0025
            push_reward_scale = 0.0015
            ocb_reward_scale = 0.004
            reach_target_reward_scale = 10
            exception_punishment_scale = -5
