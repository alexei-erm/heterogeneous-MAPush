# environments
from mqe.envs.field.legged_robot_field import LeggedRobotField
from mqe.envs.go1.go1 import Go1
from mqe.envs.npc.go1_object import Go1Object

# configs
from mqe.envs.field.legged_robot_field_config import LeggedRobotFieldCfg
from mqe.envs.configs.go1_push_mid_config import Go1PushMidCfg
from mqe.envs.configs.go1_push_upper_config import Go1PushUpperCfg


# wrappers
from mqe.envs.wrappers.empty_wrapper import EmptyWrapper
from mqe.envs.wrappers.go1_push_mid_wrapper import Go1PushMidWrapper
from mqe.envs.wrappers.go1_push_upper_wrapper import Go1PushUpperWrapper

from mqe.utils import make_env

from typing import Tuple

ENV_DICT = {
    "go1push_mid": {
        "class": Go1Object,
        "config": Go1PushMidCfg,
        "wrapper": Go1PushMidWrapper
    },
    "go1push_upper": {
        "class": Go1Object,
        "config": Go1PushUpperCfg,
        "wrapper": Go1PushUpperWrapper
    },
}

def make_mqe_env(env_name: str, args=None, custom_cfg=None) -> Tuple[LeggedRobotField, LeggedRobotFieldCfg]:
    env_dict = ENV_DICT[env_name]

    if callable(custom_cfg):
        env_dict["config"] = custom_cfg(env_dict["config"])

    env, env_cfg = make_env(env_dict["class"], env_dict["config"], args)

    env = env_dict["wrapper"](env)

    return env, env_cfg

def custom_cfg(args, individualized_rewards=False):

    def fn(cfg:LeggedRobotFieldCfg):

        if getattr(args, "num_envs", None) is not None:
            cfg.env.num_envs = args.num_envs

        cfg.env.record_video = args.record_video

        # Enable individualized rewards for HAPPO if requested
        if individualized_rewards and hasattr(cfg, 'rewards'):
            cfg.rewards.individualized_rewards = True

        return cfg

    return fn

