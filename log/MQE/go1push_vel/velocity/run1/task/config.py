import numpy as np
from mqe.envs.configs.go1_push_vel_config import Go1PushVelCfg


class Go1PushVelCfg(Go1PushVelCfg):
    """Task-specific overrides for Velocity-MAPush MAPPO training.

    Inherits everything from the base Go1PushVelCfg.
    Override box mass or other parameters here for experiments.
    """

    class asset(Go1PushVelCfg.asset):
        # Box mass for MAPPO velocity training
        # Lighter box (8kg) is easier to push = faster learning
        npc_mass_override = 8
